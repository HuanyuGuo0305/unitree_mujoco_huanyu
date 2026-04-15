"""
Sim2sim deployment for B2WZ1 loco-manipulation ONNX policy in MuJoCo.

Run from simulate_python/:

    python3 deploy_mujoco/b2wz1_loco_manipulation.py configs/b2wz1_loco_manipulation.yaml --mode pd-stand

    python3 deploy_mujoco/b2wz1_loco_manipulation.py configs/b2wz1_loco_manipulation.yaml --mode lock-arm-policy

    python3 deploy_mujoco/b2wz1_loco_manipulation.py configs/b2wz1_loco_manipulation.yaml --mode full-policy
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

# Add project root to sys.path
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
    quat_slerp_wxyz,
    quat_angle_wxyz,
    quat_from_keypoints_lb,
)
from utilities.mujoco_helper import (
    get_sensor_slice,
    make_arrow_mat,
)


class PresampledKeypointsCubicTrajectoryCommandLBSim:
    """Single-environment NumPy version of PresampledKeypointsCubicTrajectoryCommandLB.

    Behavior:
      - sample raw target from presampled table
      - apply adjacent target limit on kp0 / rotation
      - generate cubic trajectory from current command to accepted target
      - hold at target for hold_duration_s
      - auto-resample when one cycle finishes
    """

    def __init__(
        self,
        file_path: str,
        control_dt: float,
        kp_dx: float = 0.30,
        kp_dz: float = 0.30,
        kp0_threshold: float = 0.20,
        rot_threshold: float = 0.40,
        traj_duration_s: float = 4.0,
        hold_duration_s: float = 4.0,
        seed: int = 0,
    ):
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(f"Expected npy shape (N,9), got {arr.shape} from '{file_path}'.")

        self._table = arr
        self._num_rows = int(arr.shape[0])

        self._dx = float(kp_dx)
        self._dz = float(kp_dz)
        self._kp0_threshold = float(kp0_threshold)
        self._rot_threshold = float(rot_threshold)

        self._traj_duration_s = float(traj_duration_s)
        self._hold_duration_s = float(hold_duration_s)
        self._cycle_duration_s = self._traj_duration_s + self._hold_duration_s

        self._control_dt = float(control_dt)
        self._cycle_steps = max(1, int(round(self._cycle_duration_s / self._control_dt)))

        self._rng = np.random.default_rng(seed)

        self._has_cmd = False
        self._step_in_cycle = 0

        self.keypoints_command_lb = np.zeros(9, dtype=np.float32)

        self._traj_start_pos_lb = np.zeros(3, dtype=np.float32)
        self._traj_start_quat_lb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._traj_end_pos_lb = np.zeros(3, dtype=np.float32)
        self._traj_end_quat_lb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @property
    def command(self) -> np.ndarray:
        return self.keypoints_command_lb.copy()

    def _pick_index(self) -> int:
        return int(self._rng.integers(0, self._num_rows))

    @staticmethod
    def _split_kps(kps_9: np.ndarray):
        kp0 = kps_9[0:3]
        kp1 = kps_9[3:6]
        kp2 = kps_9[6:9]
        return kp0, kp1, kp2

    @staticmethod
    def _pack_kps(kp0: np.ndarray, kp1: np.ndarray, kp2: np.ndarray) -> np.ndarray:
        return np.concatenate([kp0, kp1, kp2]).astype(np.float32)

    def _kps_from_pose(self, kp0: np.ndarray, quat: np.ndarray):
        off_x = np.array([self._dx, 0.0, 0.0], dtype=np.float32)
        off_z = np.array([0.0, 0.0, self._dz], dtype=np.float32)
        kp1 = kp0 + quat_apply_wxyz(quat, off_x)
        kp2 = kp0 + quat_apply_wxyz(quat, off_z)
        return kp1.astype(np.float32), kp2.astype(np.float32)

    @staticmethod
    def _cubic_time_scaling(tau: float) -> float:
        tau = float(np.clip(tau, 0.0, 1.0))
        return 3.0 * tau * tau - 2.0 * tau * tau * tau

    def _apply_adjacent_target_limit(
        self,
        kp0_ref: np.ndarray,
        quat_ref: np.ndarray,
        kp0_raw: np.ndarray,
        quat_raw: np.ndarray,
    ):
        delta = kp0_raw - kp0_ref
        dist = max(float(np.linalg.norm(delta)), 1e-8)
        alpha_pos = min(self._kp0_threshold / dist, 1.0)

        ang = max(float(quat_angle_wxyz(quat_ref, quat_raw)), 1e-8)
        alpha_rot = min(self._rot_threshold / ang, 1.0)

        alpha = min(alpha_pos, alpha_rot)
        within = (dist <= self._kp0_threshold) and (ang <= self._rot_threshold)
        alpha_eff = 1.0 if within else alpha

        kp0_new = kp0_ref + alpha_eff * delta
        quat_new = quat_slerp_wxyz(quat_ref, quat_raw, float(alpha_eff))
        return kp0_new.astype(np.float32), quat_new.astype(np.float32)

    def _start_new_cycle_from_reference(self, ref_kps_lb: np.ndarray):
        kp0_ref, kp1_ref, kp2_ref = self._split_kps(ref_kps_lb)
        quat_ref = quat_from_keypoints_lb(kp0_ref, kp1_ref, kp2_ref, self._dx, self._dz)

        sampled = self._table[self._pick_index()].copy()
        kp0_raw, kp1_raw, kp2_raw = self._split_kps(sampled)
        quat_raw = quat_from_keypoints_lb(kp0_raw, kp1_raw, kp2_raw, self._dx, self._dz)

        kp0_end, quat_end = self._apply_adjacent_target_limit(
            kp0_ref=kp0_ref,
            quat_ref=quat_ref,
            kp0_raw=kp0_raw,
            quat_raw=quat_raw,
        )

        self._traj_start_pos_lb = kp0_ref.astype(np.float32)
        self._traj_start_quat_lb = quat_ref.astype(np.float32)
        self._traj_end_pos_lb = kp0_end.astype(np.float32)
        self._traj_end_quat_lb = quat_end.astype(np.float32)

        kp1_start, kp2_start = self._kps_from_pose(self._traj_start_pos_lb, self._traj_start_quat_lb)
        self.keypoints_command_lb = self._pack_kps(self._traj_start_pos_lb, kp1_start, kp2_start)

        self._step_in_cycle = 0
        self._has_cmd = True

    def reset(self, initial_kps_lb: np.ndarray, sample_first: bool = True):
        initial_kps_lb = np.asarray(initial_kps_lb, dtype=np.float32).reshape(9,)
        self._has_cmd = False
        self._step_in_cycle = 0
        self.keypoints_command_lb = initial_kps_lb.copy()

        if sample_first:
            self._start_new_cycle_from_reference(initial_kps_lb)
        else:
            self._has_cmd = True

    def _eval_current_command(self):
        if self._step_in_cycle <= 0:
            tau = 0.0
        else:
            t = min(self._step_in_cycle * self._control_dt, self._cycle_duration_s)
            tau = min(t / max(self._traj_duration_s, 1e-6), 1.0)

        s = self._cubic_time_scaling(tau)

        pos = self._traj_start_pos_lb + s * (self._traj_end_pos_lb - self._traj_start_pos_lb)
        quat = quat_slerp_wxyz(self._traj_start_quat_lb, self._traj_end_quat_lb, s)

        kp1, kp2 = self._kps_from_pose(pos, quat)
        self.keypoints_command_lb = self._pack_kps(pos, kp1, kp2)

    def update(self) -> np.ndarray:
        if not self._has_cmd:
            raise RuntimeError("Command sampler not initialized. Call reset() first.")

        self._eval_current_command()

        self._step_in_cycle += 1
        if self._step_in_cycle >= self._cycle_steps:
            self._start_new_cycle_from_reference(self.keypoints_command_lb.copy())

        return self.command


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="Path to yaml config")
    parser.add_argument(
        "--mode",
        type=str,
        default="full-policy",
        choices=["pd-stand", "lock-arm-policy", "full-policy"],
        help="Run mode",
    )
    args = parser.parse_args()

    # 1) Load config
    yaml_path = os.path.abspath(args.yaml_path)
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    policy_path = cfg["policy_path"]
    xml_path = cfg["xml_path"]

    if not os.path.isabs(policy_path):
        policy_path = os.path.abspath(os.path.join(project_root, policy_path))
    if not os.path.isabs(xml_path):
        xml_path = os.path.abspath(os.path.join(project_root, xml_path))

    simulation_duration = float(cfg["simulation_duration"])
    simulation_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])
    control_dt = simulation_dt * control_decimation

    history_length = int(cfg["history_length"])
    obs_dim_per_step = int(cfg["obs_dim_per_step"])
    obs_dim = int(cfg["obs_dim"])
    action_dim = int(cfg["action_dim"])

    base_command = np.array(cfg["base_command"], dtype=np.float32)

    root_pos = np.array(cfg["root_pos"], dtype=np.float32)
    root_quat = np.array(cfg["root_quat_wxyz"], dtype=np.float32)

    default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)
    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)

    leg_action_scale = float(cfg["leg_action_scale"])
    arm_action_scale = np.array(cfg["arm_action_scale"], dtype=np.float32)
    wheel_action_scale = float(cfg["wheel_action_scale"])

    leg_torque_limits = np.array(cfg["leg_torque_limits"], dtype=np.float32)
    arm_torque_limits = np.array(cfg["arm_torque_limits"], dtype=np.float32)
    gripper_torque_limit = float(cfg.get("gripper_torque_limit", 30.0))
    wheel_velocity_limits = np.array(cfg["wheel_velocity_limits"], dtype=np.float32)

    ee_body_name = str(cfg.get("ee_body", "gripperStator"))
    ee_command_path = cfg["ee_command_path"]
    if not os.path.isabs(ee_command_path):
        ee_command_path = os.path.abspath(os.path.join(project_root, ee_command_path))

    ee_kp_dx = float(cfg.get("ee_kp_dx", 0.30))
    ee_kp_dz = float(cfg.get("ee_kp_dz", 0.30))
    ee_kp0_threshold = float(cfg.get("ee_kp0_threshold", 0.20))
    ee_rot_threshold = float(cfg.get("ee_rot_threshold", 0.40))
    ee_command_seed = int(cfg.get("ee_command_seed", 0))
    ee_traj_duration_s = float(cfg.get("ee_traj_duration_s", 4.0))
    ee_hold_duration_s = float(cfg.get("ee_hold_duration_s", 4.0))

    startup_hold_s = float(cfg.get("startup_hold_s", 1.0))
    startup_blend_s = float(cfg.get("startup_blend_s", 2.0))

    # Visualization config
    vis_ee_target = bool(cfg.get("vis_ee_target", True))
    vis_ee_target_axis_len = float(cfg.get("vis_ee_target_axis_len", 0.20))
    vis_ee_target_axis_radius = float(cfg.get("vis_ee_target_axis_radius", 0.01))
    vis_ee_target_sphere_size = float(cfg.get("vis_ee_target_sphere_size", 0.03))
    vis_ee_target_alpha = float(cfg.get("vis_ee_target_alpha", 0.9))
    vis_ee_target_show_current = bool(cfg.get("vis_ee_target_show_current", True))

    assert obs_dim_per_step == 98, f"Expected obs_dim_per_step=98, got {obs_dim_per_step}"
    assert obs_dim == obs_dim_per_step * history_length, f"Expected obs_dim={obs_dim_per_step * history_length}, got {obs_dim}"
    assert action_dim == 22, f"Expected action_dim=22, got {action_dim}"
    assert len(default_joint_pos) == 23
    assert len(kps) == 23
    assert len(kds) == 23
    assert len(leg_torque_limits) == 12
    assert len(arm_torque_limits) == 6
    assert len(wheel_velocity_limits) == 4
    assert arm_action_scale.shape == (6,), f"Expected arm_action_scale shape (6,), got {arm_action_scale.shape}"

    print("=" * 72)
    print("B2WZ1 Loco-Manipulation - ONNX Policy (MuJoCo sim2sim)")
    print("=" * 72)
    print(f"Mode:            {args.mode}")
    print(f"Policy:          {policy_path}")
    print(f"XML:             {xml_path}")
    print(f"Control freq:    {1.0 / control_dt:.1f} Hz")
    print(f"Obs per step:    {obs_dim_per_step}")
    print(f"Obs stacked dim: {obs_dim}")
    print(f"Action dim:      {action_dim}")
    print(f"Arm scales:      {arm_action_scale}")
    print(f"EE cycle:        {ee_traj_duration_s + ee_hold_duration_s:.2f}s")
    print(f"EE cycle steps:  {int(round((ee_traj_duration_s + ee_hold_duration_s) / control_dt))}")
    print(f"Vis EE target:   {vis_ee_target}")
    print("=" * 72)

    use_policy = args.mode in ["lock-arm-policy", "full-policy"]

    # 2) Load ONNX
    sess = None
    input_name = None
    output_name = None
    if use_policy:
        sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        print("ONNX loaded:")
        print("  input :", input_name, sess.get_inputs()[0].shape)
        print("  output:", output_name, sess.get_outputs()[0].shape)
    else:
        print("Policy disabled in pd-stand mode.")
    print("=" * 72)

    # 3) Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    if ee_bid < 0:
        raise ValueError(f"Body not found: {ee_body_name}")

    # 4) Joint mapping
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

    # Policy order
    leg_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    arm_joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]
    wheel_joint_names = [
        "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
    ]

    policy_joint_pos_names = leg_joint_names + arm_joint_names
    policy_joint_vel_names = leg_joint_names + arm_joint_names + wheel_joint_names
    policy_action_names = leg_joint_names + arm_joint_names + wheel_joint_names

    assert len(policy_joint_pos_names) == 18
    assert len(policy_joint_vel_names) == 22
    assert len(policy_action_names) == 22

    policy_joint_pos_mujoco_indices = [mujoco_joint_names.index(n) for n in policy_joint_pos_names]
    policy_joint_vel_mujoco_indices = [mujoco_joint_names.index(n) for n in policy_joint_vel_names]

    leg_mujoco_indices = [mujoco_joint_names.index(n) for n in leg_joint_names]
    arm_mujoco_indices = [mujoco_joint_names.index(n) for n in arm_joint_names]
    wheel_mujoco_indices = [mujoco_joint_names.index(n) for n in wheel_joint_names]
    gripper_mujoco_index = mujoco_joint_names.index("jointGripper")

    leg_action_indices = list(range(0, 12))
    arm_action_indices = list(range(12, 18))
    wheel_action_indices = list(range(18, 22))

    control_source_joint_names = (
        leg_joint_names
        + arm_joint_names
        + wheel_joint_names
        + ["jointGripper"]
    )

    ctrl_source_index_by_name = {name: i for i, name in enumerate(control_source_joint_names)}
    ctrl_src_indices_or_none = [
        ctrl_source_index_by_name[name] if name in ctrl_source_index_by_name else None
        for name in ctrl_joint_names
    ]

    default_joint_pos_policy_pos = default_joint_pos[policy_joint_pos_mujoco_indices]
    default_leg_pos = default_joint_pos[leg_mujoco_indices]
    default_arm_pos = default_joint_pos[arm_mujoco_indices]
    default_gripper_pos = float(default_joint_pos[gripper_mujoco_index])

    print("Joint mapping:")
    print(f"  MuJoCo qpos order      : {mujoco_joint_names}")
    print(f"  MuJoCo ctrl order      : {ctrl_joint_names}")
    print(f"  Policy joint_pos order : {policy_joint_pos_names}")
    print(f"  Policy joint_vel order : {policy_joint_vel_names}")
    print(f"  Policy action order    : {policy_action_names}")
    print("=" * 72)

    # 5) Initialize state
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0

    d.qpos[0:3] = root_pos
    d.qpos[3:7] = root_quat
    d.qpos[7:7 + len(mujoco_joint_names)] = default_joint_pos

    mujoco.mj_forward(m, d)
    print(f"Initialized height z = {d.qpos[2]:.3f} m")

    # 6) Compute current EE keypoints in level-base frame
    def compute_ee_current_kp_lb() -> np.ndarray:
        base_pos_w = d.qpos[0:3].copy().astype(np.float32)
        base_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))

        _, _, yaw = euler_xyz_from_quat_wxyz(base_quat_w)
        lb_quat_w = quat_from_yaw_wxyz(yaw)
        lb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(lb_quat_w))

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

    def get_level_base_pose_world():
        """Return level-base origin/orientation in world frame."""
        base_pos_w = d.qpos[0:3].copy().astype(np.float32)
        base_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))

        _, _, yaw = euler_xyz_from_quat_wxyz(base_quat_w)
        lb_quat_w = quat_from_yaw_wxyz(yaw)
        lb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(lb_quat_w))
        return base_pos_w, lb_quat_w

    def ee_kp_lb_to_world(ee_kp_lb: np.ndarray):
        """Convert EE keypoints from level-base frame to world frame."""
        base_pos_w, lb_quat_w = get_level_base_pose_world()

        kp0_lb = ee_kp_lb[0:3].astype(np.float32)
        kp1_lb = ee_kp_lb[3:6].astype(np.float32)
        kp2_lb = ee_kp_lb[6:9].astype(np.float32)

        kp0_w = base_pos_w + quat_apply_wxyz(lb_quat_w, kp0_lb)
        kp1_w = base_pos_w + quat_apply_wxyz(lb_quat_w, kp1_lb)
        kp2_w = base_pos_w + quat_apply_wxyz(lb_quat_w, kp2_lb)

        return kp0_w, kp1_w, kp2_w

    def add_capsule_to_scene(scene, p0, p1, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return

        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        diff = p1 - p0
        length = np.linalg.norm(diff)
        if length < 1e-8:
            return

        pos = 0.5 * (p0 + p1)
        mat = make_arrow_mat(diff)

        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([radius, 0.5 * length, 0.0], dtype=np.float64),
            pos,
            mat.reshape(-1),
            np.array(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    def add_sphere_to_scene(scene, pos, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return

        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0], dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.array(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    def visualize_ee_pose(scene, ee_kp_lb: np.ndarray, axis_len: float, axis_radius: float,
                          sphere_size: float, alpha: float,
                          pos_rgba, x_rgba, z_rgba):
        kp0_w, kp1_w, kp2_w = ee_kp_lb_to_world(ee_kp_lb)

        x_dir = kp1_w - kp0_w
        z_dir = kp2_w - kp0_w

        x_norm = np.linalg.norm(x_dir)
        z_norm = np.linalg.norm(z_dir)
        if x_norm < 1e-8 or z_norm < 1e-8:
            return

        x_dir = x_dir / x_norm
        z_dir = z_dir / z_norm

        x_end = kp0_w + axis_len * x_dir
        z_end = kp0_w + axis_len * z_dir

        add_sphere_to_scene(scene, kp0_w, sphere_size, [pos_rgba[0], pos_rgba[1], pos_rgba[2], alpha])
        add_capsule_to_scene(scene, kp0_w, x_end, axis_radius, [x_rgba[0], x_rgba[1], x_rgba[2], alpha])
        add_capsule_to_scene(scene, kp0_w, z_end, axis_radius, [z_rgba[0], z_rgba[1], z_rgba[2], alpha])

    def update_custom_visualization(viewer, ee_cmd_lb: np.ndarray):
        viewer.user_scn.ngeom = 0

        if not vis_ee_target:
            return

        # Target pose: red
        visualize_ee_pose(
            viewer.user_scn,
            ee_kp_lb=ee_cmd_lb,
            axis_len=vis_ee_target_axis_len,
            axis_radius=vis_ee_target_axis_radius,
            sphere_size=vis_ee_target_sphere_size,
            alpha=vis_ee_target_alpha,
            pos_rgba=[1.0, 0.0, 0.0, vis_ee_target_alpha],
            x_rgba=[1.0, 0.0, 0.0, vis_ee_target_alpha],
            z_rgba=[1.0, 0.0, 0.0, vis_ee_target_alpha],
        )

        if vis_ee_target_show_current:
            ee_cur_lb = compute_ee_current_kp_lb()
            visualize_ee_pose(
                viewer.user_scn,
                ee_kp_lb=ee_cur_lb,
                axis_len=vis_ee_target_axis_len * 0.85,
                axis_radius=vis_ee_target_axis_radius * 0.8,
                sphere_size=vis_ee_target_sphere_size * 0.85,
                alpha=min(1.0, vis_ee_target_alpha),
                pos_rgba=[0.0, 0.0, 1.0, vis_ee_target_alpha],
                x_rgba=[0.0, 0.0, 1.0, vis_ee_target_alpha],
                z_rgba=[0.0, 0.0, 1.0, vis_ee_target_alpha],
            )

    # 7) EE command sampler
    ee_cmd_sampler = PresampledKeypointsCubicTrajectoryCommandLBSim(
        file_path=ee_command_path,
        control_dt=control_dt,
        kp_dx=ee_kp_dx,
        kp_dz=ee_kp_dz,
        kp0_threshold=ee_kp0_threshold,
        rot_threshold=ee_rot_threshold,
        traj_duration_s=ee_traj_duration_s,
        hold_duration_s=ee_hold_duration_s,
        seed=ee_command_seed,
    )

    ee_cur_init_lb = compute_ee_current_kp_lb()
    ee_cmd_sampler.reset(initial_kps_lb=ee_cur_init_lb, sample_first=True)
    ee_cmd_lb_current = ee_cmd_sampler.command.copy()

    # 8) Build one-step observation
    last_action = np.zeros(action_dim, dtype=np.float32)

    def build_obs_step(ee_cmd_lb: np.ndarray) -> np.ndarray:
        qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy().astype(np.float32)
        qvel_mujoco = d.qvel[6:6 + len(mujoco_joint_names)].copy().astype(np.float32)

        base_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)

        base_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(base_quat_w, gravity_w).astype(np.float32)

        ee_cur_lb = compute_ee_current_kp_lb()
        ee_err_lb = (ee_cmd_lb - ee_cur_lb).astype(np.float32)

        joint_pos_policy = qpos_mujoco[policy_joint_pos_mujoco_indices]
        joint_pos_rel = joint_pos_policy - default_joint_pos_policy_pos
        joint_pos_leg_rel = joint_pos_rel[:12]
        joint_pos_arm_rel = joint_pos_rel[12:18]

        joint_vel_policy = qvel_mujoco[policy_joint_vel_mujoco_indices]
        joint_vel_leg = joint_vel_policy[:12]
        joint_vel_arm = joint_vel_policy[12:18]
        joint_vel_wheel = joint_vel_policy[18:22]

        obs = np.concatenate(
            [
                base_ang_vel_b,          # 3
                projected_gravity_b,     # 3
                base_command,            # 3
                ee_cmd_lb,               # 9
                ee_cur_lb,               # 9
                ee_err_lb,               # 9
                joint_pos_leg_rel,       # 12
                joint_pos_arm_rel,       # 6
                joint_vel_leg,           # 12
                joint_vel_arm,           # 6
                joint_vel_wheel,         # 4
                last_action,             # 22
            ],
            dtype=np.float32,
        )

        assert obs.shape[0] == obs_dim_per_step, f"Obs dim mismatch: {obs.shape[0]} vs {obs_dim_per_step}"
        return obs

    # 9) Initial targets and per-term history
    leg_target = default_leg_pos.copy()
    arm_target = default_arm_pos.copy()
    wheel_cmd = np.zeros(4, dtype=np.float32)
    gripper_target = default_gripper_pos

    obs0 = build_obs_step(ee_cmd_lb_current)

    i = 0
    obs0_base_ang_vel = obs0[i:i + 3]; i += 3
    obs0_projected_gravity = obs0[i:i + 3]; i += 3
    obs0_base_cmd = obs0[i:i + 3]; i += 3
    obs0_ee_cmd = obs0[i:i + 9]; i += 9
    obs0_ee_cur = obs0[i:i + 9]; i += 9
    obs0_ee_err = obs0[i:i + 9]; i += 9
    obs0_joint_pos_leg = obs0[i:i + 12]; i += 12
    obs0_joint_pos_arm = obs0[i:i + 6]; i += 6
    obs0_joint_vel_leg = obs0[i:i + 12]; i += 12
    obs0_joint_vel_arm = obs0[i:i + 6]; i += 6
    obs0_joint_vel_wheel = obs0[i:i + 4]; i += 4
    obs0_last_action = obs0[i:i + 22]; i += 22

    base_ang_vel_hist = deque(maxlen=history_length)
    projected_gravity_hist = deque(maxlen=history_length)
    base_cmd_hist = deque(maxlen=history_length)
    ee_cmd_hist = deque(maxlen=history_length)
    ee_cur_hist = deque(maxlen=history_length)
    ee_err_hist = deque(maxlen=history_length)
    joint_pos_leg_hist = deque(maxlen=history_length)
    joint_pos_arm_hist = deque(maxlen=history_length)
    joint_vel_leg_hist = deque(maxlen=history_length)
    joint_vel_arm_hist = deque(maxlen=history_length)
    joint_vel_wheel_hist = deque(maxlen=history_length)
    last_action_hist = deque(maxlen=history_length)

    for _ in range(history_length):
        base_ang_vel_hist.append(obs0_base_ang_vel.copy())
        projected_gravity_hist.append(obs0_projected_gravity.copy())
        base_cmd_hist.append(obs0_base_cmd.copy())
        ee_cmd_hist.append(obs0_ee_cmd.copy())
        ee_cur_hist.append(obs0_ee_cur.copy())
        ee_err_hist.append(obs0_ee_err.copy())
        joint_pos_leg_hist.append(obs0_joint_pos_leg.copy())
        joint_pos_arm_hist.append(obs0_joint_pos_arm.copy())
        joint_vel_leg_hist.append(obs0_joint_vel_leg.copy())
        joint_vel_arm_hist.append(obs0_joint_vel_arm.copy())
        joint_vel_wheel_hist.append(obs0_joint_vel_wheel.copy())
        last_action_hist.append(obs0_last_action.copy())

    # 10) Main simulation loop
    counter = 0
    policy_tick = 0
    sim_time = 0.0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = d.qpos[:3]

        while viewer.is_running() and sim_time < simulation_duration:
            step_start = time.time()

            if sim_time < startup_hold_s:
                blend = 0.0
            elif sim_time < startup_hold_s + startup_blend_s:
                blend = (sim_time - startup_hold_s) / startup_blend_s
            else:
                blend = 1.0
            blend = float(np.clip(blend, 0.0, 1.0))

            # Low-level control (external PD)
            qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy().astype(np.float32)
            qvel_mujoco = d.qvel[6:6 + len(mujoco_joint_names)].copy().astype(np.float32)

            leg_pos = qpos_mujoco[leg_mujoco_indices]
            leg_vel = qvel_mujoco[leg_mujoco_indices]

            arm_pos = qpos_mujoco[arm_mujoco_indices]
            arm_vel = qvel_mujoco[arm_mujoco_indices]

            gripper_pos = float(qpos_mujoco[gripper_mujoco_index])
            gripper_vel = float(qvel_mujoco[gripper_mujoco_index])

            leg_tau = kps[leg_mujoco_indices] * (leg_target - leg_pos) - kds[leg_mujoco_indices] * leg_vel
            leg_tau = np.clip(leg_tau, -leg_torque_limits, leg_torque_limits).astype(np.float32)

            arm_tau = kps[arm_mujoco_indices] * (arm_target - arm_pos) - kds[arm_mujoco_indices] * arm_vel
            arm_tau = np.clip(arm_tau, -arm_torque_limits, arm_torque_limits).astype(np.float32)

            gripper_tau = kps[gripper_mujoco_index] * (gripper_target - gripper_pos) - kds[gripper_mujoco_index] * gripper_vel
            gripper_tau = float(np.clip(gripper_tau, -gripper_torque_limit, gripper_torque_limit))

            wheel_ctrl = np.clip(wheel_cmd, -wheel_velocity_limits, wheel_velocity_limits).astype(np.float32)

            ctrl_source = np.concatenate(
                [
                    leg_tau,
                    arm_tau,
                    wheel_ctrl,
                    np.array([gripper_tau], dtype=np.float32),
                ],
                dtype=np.float32,
            )

            d.ctrl[:] = 0.0
            for ctrl_i, src_i in enumerate(ctrl_src_indices_or_none):
                if src_i is not None:
                    d.ctrl[ctrl_i] = ctrl_source[src_i]

            mujoco.mj_step(m, d)
            viewer.cam.lookat[:] = d.qpos[:3]
            sim_time += simulation_dt

            # Policy inference
            if counter % control_decimation == 0:
                ee_cmd_lb_current = ee_cmd_sampler.update()

                obs_step = build_obs_step(ee_cmd_lb_current)

                i = 0
                curr_base_ang_vel = obs_step[i:i + 3]; i += 3
                curr_projected_gravity = obs_step[i:i + 3]; i += 3
                curr_base_cmd = obs_step[i:i + 3]; i += 3
                curr_ee_cmd = obs_step[i:i + 9]; i += 9
                curr_ee_cur = obs_step[i:i + 9]; i += 9
                curr_ee_err = obs_step[i:i + 9]; i += 9
                curr_joint_pos_leg = obs_step[i:i + 12]; i += 12
                curr_joint_pos_arm = obs_step[i:i + 6]; i += 6
                curr_joint_vel_leg = obs_step[i:i + 12]; i += 12
                curr_joint_vel_arm = obs_step[i:i + 6]; i += 6
                curr_joint_vel_wheel = obs_step[i:i + 4]; i += 4
                curr_last_action = obs_step[i:i + 22]; i += 22

                base_ang_vel_hist.append(curr_base_ang_vel.copy())
                projected_gravity_hist.append(curr_projected_gravity.copy())
                base_cmd_hist.append(curr_base_cmd.copy())
                ee_cmd_hist.append(curr_ee_cmd.copy())
                ee_cur_hist.append(curr_ee_cur.copy())
                ee_err_hist.append(curr_ee_err.copy())
                joint_pos_leg_hist.append(curr_joint_pos_leg.copy())
                joint_pos_arm_hist.append(curr_joint_pos_arm.copy())
                joint_vel_leg_hist.append(curr_joint_vel_leg.copy())
                joint_vel_arm_hist.append(curr_joint_vel_arm.copy())
                joint_vel_wheel_hist.append(curr_joint_vel_wheel.copy())
                last_action_hist.append(curr_last_action.copy())

                obs_stack = np.concatenate(
                    [
                        np.array(base_ang_vel_hist).reshape(-1),
                        np.array(projected_gravity_hist).reshape(-1),
                        np.array(base_cmd_hist).reshape(-1),
                        np.array(ee_cmd_hist).reshape(-1),
                        np.array(ee_cur_hist).reshape(-1),
                        np.array(ee_err_hist).reshape(-1),
                        np.array(joint_pos_leg_hist).reshape(-1),
                        np.array(joint_pos_arm_hist).reshape(-1),
                        np.array(joint_vel_leg_hist).reshape(-1),
                        np.array(joint_vel_arm_hist).reshape(-1),
                        np.array(joint_vel_wheel_hist).reshape(-1),
                        np.array(last_action_hist).reshape(-1),
                    ],
                    dtype=np.float32,
                )

                assert obs_stack.shape[0] == obs_dim, f"obs_stack dim mismatch: {obs_stack.shape[0]} vs {obs_dim}"

                if args.mode == "pd-stand":
                    action = np.zeros(action_dim, dtype=np.float32)
                else:
                    action = sess.run(
                        [output_name],
                        {input_name: obs_stack[None, :]},
                    )[0][0].astype(np.float32)

                last_action[:] = action

                leg_act = action[leg_action_indices].copy()
                arm_act = action[arm_action_indices].copy()
                wheel_act = action[wheel_action_indices].copy()

                if args.mode == "pd-stand":
                    leg_target = default_leg_pos.copy()
                    arm_target = default_arm_pos.copy()
                    wheel_cmd[:] = 0.0

                elif args.mode == "lock-arm-policy":
                    leg_target = default_leg_pos + blend * (leg_action_scale * leg_act)
                    arm_target = default_arm_pos.copy()
                    wheel_cmd = blend * (wheel_action_scale * wheel_act)

                elif args.mode == "full-policy":
                    leg_target = default_leg_pos + blend * (leg_action_scale * leg_act)
                    arm_target = default_arm_pos + blend * (arm_action_scale * arm_act)
                    wheel_cmd = blend * (wheel_action_scale * wheel_act)

                policy_tick += 1

            # Visualization
            update_custom_visualization(viewer, ee_cmd_lb_current)

            # Logging
            if counter % 200 == 0:
                current_leg_pos = d.qpos[7:7 + len(mujoco_joint_names)].copy()[leg_mujoco_indices]
                current_arm_pos = d.qpos[7:7 + len(mujoco_joint_names)].copy()[arm_mujoco_indices]
                leg_err = leg_target - current_leg_pos
                arm_err = arm_target - current_arm_pos
                ee_err_norm = np.linalg.norm((ee_cmd_lb_current - compute_ee_current_kp_lb()).reshape(3, 3), axis=1)

                print(
                    f"[{counter:6d}] "
                    f"t={sim_time:6.2f}s | "
                    f"z={d.qpos[2]:.3f} | "
                    f"blend={blend:.2f} | "
                    f"leg_act=[{last_action[leg_action_indices].min():+.2f},{last_action[leg_action_indices].max():+.2f}] | "
                    f"arm_act=[{last_action[arm_action_indices].min():+.2f},{last_action[arm_action_indices].max():+.2f}] | "
                    f"wheel_act=[{last_action[wheel_action_indices].min():+.2f},{last_action[wheel_action_indices].max():+.2f}]"
                )
                print(
                    f"           max_leg_err={np.max(np.abs(leg_err)):.4f} | "
                    f"max_arm_err={np.max(np.abs(arm_err)):.4f} | "
                    f"ee_err=[{ee_err_norm[0]:.4f},{ee_err_norm[1]:.4f},{ee_err_norm[2]:.4f}] | "
                    f"max_ctrl={np.max(np.abs(d.ctrl[:])):.2f}"
                )

            counter += 1
            viewer.sync()

            time_until_next_step = simulation_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)