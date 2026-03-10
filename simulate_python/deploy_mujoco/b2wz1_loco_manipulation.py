"""
Sim2sim test for B2WZ1 loco-manipulation ONNX policy in MuJoCo.

Run from simulate_python/:

    python3 deploy_mujoco/b2wz1_loco_manipulation.py \
        configs/b2wz1_loco_manipulation.yaml \
        --mode pd-stand

    python3 deploy_mujoco/b2wz1_loco_manipulation.py \
        configs/b2wz1_loco_manipulation.yaml \
        --mode lock-arm-policy

    python3 deploy_mujoco/b2wz1_loco_manipulation.py \
        configs/b2wz1_loco_manipulation.yaml \
        --mode full-policy
"""

import os
import sys
import time
import argparse
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from utilities.math import (
    quat_apply_inverse_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_from_rotmat_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_rotate_inverse_numpy,
    quat_unique_wxyz,
    euler_xyz_from_quat_wxyz,
    quat_from_yaw_wxyz,
    normalize,
    quat_slerp_wxyz,
    quat_angle_wxyz,
    quat_from_keypoints_lb,
)

# ============================================================
# Basic helpers
# ============================================================

def get_sensor_slice(model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str) -> np.ndarray:
    """Read one sensor by name."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid < 0:
        raise ValueError(f"Sensor not found: {sensor_name}")
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return data.sensordata[adr : adr + dim].copy()


def debug_print_obs(tag: str, obs_step: np.ndarray):
    i = 0
    base_ang_vel_b = obs_step[i:i + 3]; i += 3
    projected_gravity_b = obs_step[i:i + 3]; i += 3
    base_cmd = obs_step[i:i + 3]; i += 3
    ee_cmd_lb = obs_step[i:i + 9]; i += 9
    ee_cur_lb = obs_step[i:i + 9]; i += 9
    joint_pos_rel = obs_step[i:i + 18]; i += 18
    joint_vel = obs_step[i:i + 22]; i += 22
    last_action = obs_step[i:i + 22]; i += 22

    print(f"\n[DEBUG {tag}]")
    print(f"  base_ang_vel_b      : {base_ang_vel_b}")
    print(f"  projected_gravity_b : {projected_gravity_b}")
    print(f"  base_cmd            : {base_cmd}")
    print(f"  ee_cmd_lb kp0       : {ee_cmd_lb[0:3]}")
    print(f"  ee_cmd_lb kp1       : {ee_cmd_lb[3:6]}")
    print(f"  ee_cmd_lb kp2       : {ee_cmd_lb[6:9]}")
    print(f"  ee_cur_lb kp0       : {ee_cur_lb[0:3]}")
    print(f"  ee_cur_lb kp1       : {ee_cur_lb[3:6]}")
    print(f"  ee_cur_lb kp2       : {ee_cur_lb[6:9]}")
    print(f"  joint_pos_rel       : {joint_pos_rel}")
    print(f"  joint_vel           : {joint_vel}")
    print(f"  last_action         : {last_action}")
    print(f"  |joint_pos_rel|max  : {np.abs(joint_pos_rel).max()}")
    print(f"  |joint_vel|max      : {np.abs(joint_vel).max()}")
    print(f"  |last_action|max    : {np.abs(last_action).max()}")


# ============================================================
# IsaacLab-style EE command sampler (single-env NumPy version)
# ============================================================

class PresampledKeypointsInterpolateCommandLBSim:
    """
    Single-environment NumPy implementation of IsaacLab's
    PresampledKeypointsInterpolateCommandLB.

    Table format:
        npy shape = (N, 9)
        row = [kp0_xyz, kp1_xyz, kp2_xyz] in level-base frame
    """

    def __init__(
        self,
        file_path: str,
        kp_dx: float = 0.30,
        kp_dz: float = 0.30,
        kp0_threshold: float = 0.20,
        rot_threshold: float = 0.40,
        seed: int = 0,
    ):
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(
                f"[PresampledKeypointsInterpolateCommandLBSim] "
                f"Expected npy shape (N,9), got {arr.shape} from '{file_path}'."
            )

        self._table = arr
        self._num_rows = int(arr.shape[0])

        self._dx = float(kp_dx)
        self._dz = float(kp_dz)
        self._kp0_threshold = float(kp0_threshold)
        self._rot_threshold = float(rot_threshold)

        self._rng = np.random.default_rng(seed)

        self.keypoints_command_lb = np.zeros(9, dtype=np.float32)
        self._has_cmd = False

    def __str__(self) -> str:
        msg = "PresampledKeypointsInterpolateCommandLBSim:\n"
        msg += f"  table rows      : {self._num_rows}\n"
        msg += f"  kp_dx / kp_dz   : {self._dx:.3f} / {self._dz:.3f}\n"
        msg += f"  kp0_threshold   : {self._kp0_threshold:.3f} m\n"
        msg += f"  rot_threshold   : {self._rot_threshold:.3f} rad\n"
        return msg

    @property
    def command(self) -> np.ndarray:
        return self.keypoints_command_lb.copy()

    def _pick_indices(self, k: int) -> np.ndarray:
        return self._rng.integers(0, self._num_rows, size=(k,), endpoint=False)

    @staticmethod
    def _split_kps(kps_9: np.ndarray):
        kp0 = kps_9[0:3]
        kp1 = kps_9[3:6]
        kp2 = kps_9[6:9]
        return kp0, kp1, kp2

    @staticmethod
    def _pack_kps(kp0: np.ndarray, kp1: np.ndarray, kp2: np.ndarray) -> np.ndarray:
        return np.concatenate([kp0, kp1, kp2], dtype=np.float32)

    def _kps_from_pose(self, kp0: np.ndarray, quat: np.ndarray):
        off_x = np.array([self._dx, 0.0, 0.0], dtype=np.float32)
        off_z = np.array([0.0, 0.0, self._dz], dtype=np.float32)
        kp1 = kp0 + quat_apply_wxyz(quat, off_x)
        kp2 = kp0 + quat_apply_wxyz(quat, off_z)
        return kp1.astype(np.float32), kp2.astype(np.float32)

    def reset(self):
        idx = int(self._pick_indices(1)[0])
        self.keypoints_command_lb = self._table[idx].copy()
        self._has_cmd = True

    def resample(self):
        idx = int(self._pick_indices(1)[0])
        kps_s = self._table[idx].copy()

        if not self._has_cmd:
            self.keypoints_command_lb = kps_s
            self._has_cmd = True
            return

        kp0_s, kp1_s, kp2_s = self._split_kps(kps_s)
        kp0_p, kp1_p, kp2_p = self._split_kps(self.keypoints_command_lb)

        quat_p = quat_from_keypoints_lb(kp0_p, kp1_p, kp2_p, self._dx, self._dz)
        quat_s = quat_from_keypoints_lb(kp0_s, kp1_s, kp2_s, self._dx, self._dz)

        delta = kp0_s - kp0_p
        dist = max(float(np.linalg.norm(delta)), 1e-8)
        alpha_pos = min(self._kp0_threshold / dist, 1.0)

        ang = max(quat_angle_wxyz(quat_p, quat_s), 1e-8)
        alpha_rot = min(self._rot_threshold / ang, 1.0)

        alpha = min(alpha_pos, alpha_rot)

        need_interp = (dist > self._kp0_threshold) or (ang > self._rot_threshold)
        alpha_eff = alpha if need_interp else 1.0

        kp0_new = kp0_p + alpha_eff * delta
        quat_new = quat_slerp_wxyz(quat_p, quat_s, float(alpha_eff))
        kp1_new, kp2_new = self._kps_from_pose(kp0_new, quat_new)

        self.keypoints_command_lb = self._pack_kps(kp0_new, kp1_new, kp2_new)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path to b2wz1_loco_manipulation.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default="full-policy",
        choices=["pd-stand", "lock-arm-policy", "full-policy"],
        help="Run mode",
    )
    args = parser.parse_args()

    # --------------------------------------------------------
    # 1) Load config
    # --------------------------------------------------------
    yaml_path = os.path.abspath(args.yaml_path)
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    policy_path = cfg["policy_path"]
    xml_path = cfg["xml_path"]

    if not os.path.isabs(policy_path):
        policy_path = os.path.abspath(os.path.join(project_root, policy_path))
    if not os.path.isabs(xml_path):
        xml_path = os.path.abspath(os.path.join(project_root, xml_path))

    sim_duration = float(cfg["simulation_duration"])
    dt = float(cfg["simulation_dt"])
    decim = int(cfg["control_decimation"])

    history_len = int(cfg["history_length"])
    obs_dim_per_step = int(cfg["obs_dim_per_step"])
    obs_dim = int(cfg["obs_dim"])
    act_dim = int(cfg["action_dim"])

    base_cmd = np.array(cfg["base_command"], dtype=np.float32)

    root_pos = np.array(cfg["root_pos"], dtype=np.float32)
    root_quat = np.array(cfg["root_quat_wxyz"], dtype=np.float32)

    default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)

    leg_scale = float(cfg["leg_action_scale"])
    arm_scale = float(cfg["arm_action_scale"])
    wheel_scale = float(cfg["wheel_action_scale"])

    leg_kps = np.array(cfg["leg_kps"], dtype=np.float32)
    leg_kds = np.array(cfg["leg_kds"], dtype=np.float32)
    arm_kps = np.array(cfg["arm_kps"], dtype=np.float32)
    arm_kds = np.array(cfg["arm_kds"], dtype=np.float32)

    leg_torque_limits = np.array(cfg["leg_torque_limits"], dtype=np.float32)
    arm_torque_limit = float(cfg["arm_torque_limit"])
    wheel_vel_limit = float(cfg["wheel_vel_limit"])

    # 改成 body 名称，不再用 site
    ee_body_name = str(cfg.get("ee_body", "gripperStator"))

    # command sampler config
    ee_command_path = cfg["ee_command_path"]
    if not os.path.isabs(ee_command_path):
        ee_command_path = os.path.abspath(os.path.join(project_root, ee_command_path))
    ee_kp_dx = float(cfg.get("ee_kp_dx", 0.30))
    ee_kp_dz = float(cfg.get("ee_kp_dz", 0.30))
    ee_kp0_threshold = float(cfg.get("ee_kp0_threshold", 0.20))
    ee_rot_threshold = float(cfg.get("ee_rot_threshold", 0.40))
    ee_command_seed = int(cfg.get("ee_command_seed", 0))
    ee_resample_interval = int(cfg.get("ee_resample_interval", 50))

    assert obs_dim_per_step == 89, f"Expected obs_dim_per_step=89, got {obs_dim_per_step}"
    assert obs_dim == obs_dim_per_step * history_len, (
        f"Expected obs_dim={obs_dim_per_step * history_len}, got {obs_dim}"
    )
    assert act_dim == 22, f"Expected act_dim=22, got {act_dim}"

    print("=" * 70)
    print("B2WZ1 Loco-Manip - ONNX Policy (MuJoCo sim2sim)")
    print("=" * 70)
    print(f"Mode:   {args.mode}")
    print(f"Policy: {policy_path}")
    print(f"XML:    {xml_path}")
    print(f"Control freq: {1.0 / (dt * decim):.1f} Hz")
    print("=" * 70)

    use_policy = args.mode in ["lock-arm-policy", "full-policy"]

    # --------------------------------------------------------
    # 2) Load ONNX
    # --------------------------------------------------------
    sess = None
    input_name = None
    output_name = None
    if use_policy:
        sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        print("ONNX loaded:")
        print(" input :", input_name, sess.get_inputs()[0].shape)
        print(" output:", output_name, sess.get_outputs()[0].shape)
    else:
        print("Policy disabled in pd-stand mode.")
    print("=" * 70)

    # --------------------------------------------------------
    # 3) Load MuJoCo model
    # --------------------------------------------------------
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = dt

    # 直接找 body，不再找 site
    ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    if ee_bid < 0:
        raise ValueError(f"Body not found: {ee_body_name}")
    print(f"EE body: {ee_body_name}, body_id={ee_bid}")

    # --------------------------------------------------------
    # 4) Joint mapping
    # --------------------------------------------------------
    mujoco_joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper",
    ]

    ctrl_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper",
    ]

    policy_joint_pos_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]

    policy_joint_vel_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "joint1",
        "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
        "joint2", "joint3", "joint4", "joint5", "joint6",
    ]

    policy_action_semantic = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
    ]
    assert len(policy_action_semantic) == act_dim

    leg_action_joint_names = policy_action_semantic[:12]
    arm_action_joint_names = policy_action_semantic[12:18]
    wheel_action_joint_names = policy_action_semantic[18:22]

    num_leg_joints = 12
    num_arm_joints = 6
    num_wheel_joints = 4

    leg_action_policy_indices = list(range(12))
    arm_action_policy_indices = list(range(12, 18))
    wheel_action_policy_indices = list(range(18, 22))

    mujoco_to_policy_joint_pos_indices = [
        mujoco_joint_names.index(name) for name in policy_joint_pos_names
    ]
    mujoco_to_policy_joint_vel_indices = [
        mujoco_joint_names.index(name) for name in policy_joint_vel_names
    ]

    leg_mujoco_joint_indices = [mujoco_joint_names.index(name) for name in leg_action_joint_names]
    arm_mujoco_joint_indices = [mujoco_joint_names.index(name) for name in arm_action_joint_names]
    wheel_mujoco_joint_indices = [mujoco_joint_names.index(name) for name in wheel_action_joint_names]

    control_source_joint_names = leg_action_joint_names + arm_action_joint_names + wheel_action_joint_names
    ctrl_source_index_by_name = {name: i for i, name in enumerate(control_source_joint_names)}
    ctrl_src_indices_or_none = [
        ctrl_source_index_by_name[name] if name in ctrl_source_index_by_name else None
        for name in ctrl_joint_names
    ]

    assert len(default_joint_pos) == len(mujoco_joint_names), (
        f"default_joint_pos length mismatch: {len(default_joint_pos)} vs {len(mujoco_joint_names)}"
    )

    default_joint_pos_policy_jointpos = default_joint_pos[mujoco_to_policy_joint_pos_indices]

    default_leg_pos_policy = np.array(
        [default_joint_pos[mujoco_joint_names.index(name)] for name in leg_action_joint_names],
        dtype=np.float32,
    )
    default_arm_pos_policy = np.array(
        [default_joint_pos[mujoco_joint_names.index(name)] for name in arm_action_joint_names],
        dtype=np.float32,
    )
    default_wheel_pos_policy = np.array(
        [default_joint_pos[mujoco_joint_names.index(name)] for name in wheel_action_joint_names],
        dtype=np.float32,
    )

    print("Joint mapping:")
    print(f" MuJoCo joint order         : {mujoco_joint_names}\n")
    print(f" Ctrl joint order           : {ctrl_joint_names}\n")
    print(f" Policy joint_pos order     : {policy_joint_pos_names}\n")
    print(f" Policy joint_vel order     : {policy_joint_vel_names}\n")
    print(f" Policy action semantic     : {policy_action_semantic}\n")
    print(f" default_leg_pos_policy     : {default_leg_pos_policy}")
    print(f" default_arm_pos_policy     : {default_arm_pos_policy}")
    print(f" default_wheel_pos_policy   : {default_wheel_pos_policy}")
    print("=" * 70)

    # --------------------------------------------------------
    # 5) Initialize state
    # --------------------------------------------------------
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0

    d.qpos[0:3] = root_pos
    d.qpos[3:7] = root_quat
    d.qpos[7:7 + len(mujoco_joint_names)] = default_joint_pos

    mujoco.mj_forward(m, d)
    print(f"Initialized height z = {d.qpos[2]:.3f} m")

    # --------------------------------------------------------
    # 6) EE current keypoints in level-base frame
    # --------------------------------------------------------
    def compute_ee_current_kp_lb() -> np.ndarray:
        base_pos_w = d.qpos[0:3].copy().astype(np.float32)
        base_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))

        _, _, yaw = euler_xyz_from_quat_wxyz(base_quat_w)
        lb_quat_w = quat_from_yaw_wxyz(yaw)
        lb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(lb_quat_w))

        # 用 body pose，不再用 site pose
        ee_pos_w = d.xpos[ee_bid].copy().astype(np.float32)
        ee_rot_w = d.xmat[ee_bid].reshape(3, 3).copy().astype(np.float32)
        ee_quat_w = quat_from_rotmat_wxyz(ee_rot_w)

        ee_pos_lb = quat_apply_inverse_wxyz(lb_quat_w, ee_pos_w - base_pos_w)
        ee_quat_lb = quat_mul_wxyz(quat_conjugate_wxyz(lb_quat_w), ee_quat_w)
        ee_quat_lb = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_lb))

        off_x = np.array([ee_kp_dx, 0.0, 0.0], dtype=np.float32)
        off_z = np.array([0.0, 0.0, ee_kp_dz], dtype=np.float32)

        kp0 = ee_pos_lb
        kp1 = ee_pos_lb + quat_apply_wxyz(ee_quat_lb, off_x)
        kp2 = ee_pos_lb + quat_apply_wxyz(ee_quat_lb, off_z)
        return np.concatenate([kp0, kp1, kp2]).astype(np.float32)

    # --------------------------------------------------------
    # 7) EE command sampler
    # --------------------------------------------------------
    ee_cmd_sampler = PresampledKeypointsInterpolateCommandLBSim(
        file_path=ee_command_path,
        kp_dx=ee_kp_dx,
        kp_dz=ee_kp_dz,
        kp0_threshold=ee_kp0_threshold,
        rot_threshold=ee_rot_threshold,
        seed=ee_command_seed,
    )
    ee_cmd_sampler.reset()
    print(ee_cmd_sampler)
    print(f"Initial ee command lb: {ee_cmd_sampler.command}")

    # --------------------------------------------------------
    # 8) Build one-step obs
    # --------------------------------------------------------
    last_action = np.zeros(act_dim, dtype=np.float32)

    def build_obs_step(ee_cmd_lb: np.ndarray) -> np.ndarray:
        qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy()
        qvel_mujoco = d.qvel[6:6 + len(mujoco_joint_names)].copy()

        base_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)

        base_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(base_quat_w, gravity_w)

        ee_cur_lb = compute_ee_current_kp_lb()
        # ee_cur_lb = ee_cmd_lb.copy()  # for debugging

        joint_pos_policy = qpos_mujoco[mujoco_to_policy_joint_pos_indices]
        joint_pos_rel = joint_pos_policy - default_joint_pos_policy_jointpos

        joint_vel_policy = qvel_mujoco[mujoco_to_policy_joint_vel_indices]

        obs = np.concatenate(
            [
                base_ang_vel_b,
                projected_gravity_b,
                base_cmd,
                ee_cmd_lb,
                ee_cur_lb,
                joint_pos_rel,
                joint_vel_policy,
                last_action,
            ],
            dtype=np.float32,
        )

        assert obs.shape[0] == obs_dim_per_step, (
            f"Obs dim mismatch: {obs.shape[0]} vs {obs_dim_per_step}"
        )
        return obs

    # --------------------------------------------------------
    # 9) Initial targets / history
    # --------------------------------------------------------
    leg_target = default_leg_pos_policy.copy()
    arm_target = default_arm_pos_policy.copy()
    wheel_cmd = np.zeros(num_wheel_joints, dtype=np.float32)

    ee_cmd_lb_current = ee_cmd_sampler.command.copy()

    obs0 = build_obs_step(ee_cmd_lb_current)
    debug_print_obs("obs0_init", obs0)

    i = 0
    obs0_base_ang_vel = obs0[i:i+3].copy(); i += 3
    obs0_projected_gravity = obs0[i:i+3].copy(); i += 3
    obs0_base_cmd = obs0[i:i+3].copy(); i += 3
    obs0_ee_cmd = obs0[i:i+9].copy(); i += 9
    obs0_ee_cur = obs0[i:i+9].copy(); i += 9
    obs0_joint_pos = obs0[i:i+18].copy(); i += 18
    obs0_joint_vel = obs0[i:i+22].copy(); i += 22
    obs0_last_action = obs0[i:i+22].copy(); i += 22

    ang_vel_hist = deque(maxlen=history_len)
    gravity_hist = deque(maxlen=history_len)
    cmd_hist = deque(maxlen=history_len)
    ee_cmd_hist = deque(maxlen=history_len)
    ee_cur_hist = deque(maxlen=history_len)
    jpos_hist = deque(maxlen=history_len)
    jvel_hist = deque(maxlen=history_len)
    act_hist = deque(maxlen=history_len)

    for _ in range(history_len):
        ang_vel_hist.append(obs0_base_ang_vel.copy())
        gravity_hist.append(obs0_projected_gravity.copy())
        cmd_hist.append(obs0_base_cmd.copy())
        ee_cmd_hist.append(obs0_ee_cmd.copy())
        ee_cur_hist.append(obs0_ee_cur.copy())
        jpos_hist.append(obs0_joint_pos.copy())
        jvel_hist.append(obs0_joint_vel.copy())
        act_hist.append(obs0_last_action.copy())

    # --------------------------------------------------------
    # 10) Startup blend-in
    # --------------------------------------------------------
    startup_hold_s = 1.0
    startup_blend_s = 3.0

    # --------------------------------------------------------
    # 11) Simulation loop
    # --------------------------------------------------------
    counter = 0
    policy_tick = 0
    sim_time = 0.0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = d.qpos[:3]

        while viewer.is_running() and sim_time < sim_duration:
            step_start = time.time()

            qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy()
            qvel_mujoco = d.qvel[6:6 + len(mujoco_joint_names)].copy()

            leg_pos = qpos_mujoco[leg_mujoco_joint_indices]
            leg_vel = qvel_mujoco[leg_mujoco_joint_indices]

            arm_pos = qpos_mujoco[arm_mujoco_joint_indices]
            arm_vel = qvel_mujoco[arm_mujoco_joint_indices]

            if sim_time < startup_hold_s:
                blend = 0.0
            elif sim_time < startup_hold_s + startup_blend_s:
                blend = (sim_time - startup_hold_s) / startup_blend_s
            else:
                blend = 1.0
            blend = float(np.clip(blend, 0.0, 1.0))

            leg_tau = leg_kps * (leg_target - leg_pos) - leg_kds * leg_vel
            arm_tau = arm_kps * (arm_target - arm_pos) - arm_kds * arm_vel

            leg_tau = np.clip(
                leg_tau,
                -np.repeat(leg_torque_limits, 4),
                np.repeat(leg_torque_limits, 4),
            )
            arm_tau = np.clip(arm_tau, -arm_torque_limit, arm_torque_limit)
            wheel_ctrl = np.clip(wheel_cmd, -wheel_vel_limit, wheel_vel_limit)

            ctrl_source = np.concatenate(
                [
                    leg_tau,
                    arm_tau,
                    wheel_ctrl,
                ],
                dtype=np.float32,
            )

            d.ctrl[:] = 0.0
            for ctrl_i, src_i in enumerate(ctrl_src_indices_or_none):
                if src_i is not None:
                    d.ctrl[ctrl_i] = ctrl_source[src_i]
                else:
                    d.ctrl[ctrl_i] = 0.0

            mujoco.mj_step(m, d)
            viewer.cam.lookat[:] = d.qpos[:3]
            sim_time += dt

            if counter % decim == 0:
                if counter <= 20:
                    qvel_ang = d.qvel[3:6].copy()
                    gyro_ang = get_sensor_slice(m, d, "imu_gyro")
                    print(f"\n[DEBUG gyro_vs_qvel counter={counter}]")
                    print(f"  qvel[3:6] = {qvel_ang}")
                    print(f"  imu_gyro  = {gyro_ang}")
                    print(f"  diff      = {qvel_ang - gyro_ang}")

                if policy_tick > 0 and (policy_tick % ee_resample_interval == 0):
                    ee_cmd_sampler.resample()
                ee_cmd_lb_current = ee_cmd_sampler.command.copy()

                obs_step = build_obs_step(ee_cmd_lb_current)

                if counter <= 20:
                    debug_print_obs(f"obs_step_counter_{counter}", obs_step)

                i = 0
                ang = obs_step[i:i+3]; i += 3
                grav = obs_step[i:i+3]; i += 3
                cmd = obs_step[i:i+3]; i += 3
                ee_cmd = obs_step[i:i+9]; i += 9
                ee_cur = obs_step[i:i+9]; i += 9
                jpos = obs_step[i:i+18]; i += 18
                jvel = obs_step[i:i+22]; i += 22
                act = obs_step[i:i+22]; i += 22

                ang_vel_hist.append(ang.copy())
                gravity_hist.append(grav.copy())
                cmd_hist.append(cmd.copy())
                ee_cmd_hist.append(ee_cmd.copy())
                ee_cur_hist.append(ee_cur.copy())
                jpos_hist.append(jpos.copy())
                jvel_hist.append(jvel.copy())
                act_hist.append(act.copy())

                obs_stack = np.concatenate(
                    [
                        np.array(ang_vel_hist).reshape(-1),
                        np.array(gravity_hist).reshape(-1),
                        np.array(cmd_hist).reshape(-1),
                        np.array(ee_cmd_hist).reshape(-1),
                        np.array(ee_cur_hist).reshape(-1),
                        np.array(jpos_hist).reshape(-1),
                        np.array(jvel_hist).reshape(-1),
                        np.array(act_hist).reshape(-1),
                    ],
                    dtype=np.float32,
                )

                assert obs_stack.shape[0] == obs_dim, (
                    f"obs_stack dim mismatch: {obs_stack.shape[0]} vs {obs_dim}"
                )

                if args.mode == "pd-stand":
                    action = np.zeros(act_dim, dtype=np.float32)
                else:
                    action = sess.run(
                        [output_name],
                        {input_name: obs_stack[None, :]},
                    )[0][0].astype(np.float32)

                last_action[:] = action

                leg_act = action[leg_action_policy_indices]
                arm_act = action[arm_action_policy_indices]
                wheel_act = action[wheel_action_policy_indices]

                if args.mode == "pd-stand":
                    leg_target = default_leg_pos_policy.copy()
                    arm_target = default_arm_pos_policy.copy()
                    wheel_cmd = np.zeros(num_wheel_joints, dtype=np.float32)

                elif args.mode == "lock-arm-policy":
                    leg_target = default_leg_pos_policy + blend * (leg_scale * leg_act)
                    wheel_cmd = blend * (wheel_scale * wheel_act)
                    arm_target = default_arm_pos_policy.copy()

                elif args.mode == "full-policy":
                    leg_target = default_leg_pos_policy + blend * (leg_scale * leg_act)
                    wheel_cmd = blend * (wheel_scale * wheel_act)
                    arm_target = default_arm_pos_policy + blend * (arm_scale * arm_act)

                policy_tick += 1

            if counter % 200 == 0:
                arm_vals = last_action[arm_action_policy_indices]
                print(
                    f"[{counter:6d}] "
                    f"t={sim_time:6.2f}s "
                    f"z={d.qpos[2]:.3f} | "
                    f"leg=[{last_action[leg_action_policy_indices].min():+.2f},"
                    f"{last_action[leg_action_policy_indices].max():+.2f}] "
                    f"wheel=[{last_action[wheel_action_policy_indices].min():+.2f},"
                    f"{last_action[wheel_action_policy_indices].max():+.2f}] "
                    f"arm=[{arm_vals.min():+.2f},{arm_vals.max():+.2f}] "
                    f"blend={blend:.2f}"
                )
                print(f"  leg_tau    = {np.array2string(leg_tau, precision=3)}")
                print(f"  wheel_ctrl = {np.array2string(wheel_ctrl, precision=3)}")
                print(f"  arm_tau    = {np.array2string(arm_tau, precision=3)}")
                print(f"  arm_target = {np.array2string(arm_target, precision=3)}")
                print(f"  ee_cmd_lb  = {np.array2string(ee_cmd_lb_current, precision=3)}")
                print(f"  d.ctrl     = {np.array2string(d.ctrl, precision=3)}")

            counter += 1
            viewer.sync()

            time_until_next = dt - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)