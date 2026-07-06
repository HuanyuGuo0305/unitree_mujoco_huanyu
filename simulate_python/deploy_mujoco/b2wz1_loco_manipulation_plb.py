"""
Sim2sim deployment for B2WZ1 + Z1 loco-manipulation ONNX policy in MuJoCo,
aligned with the second-version PLB training setup.

PLB frame:
  - origin = [base_x, base_y, ground_z]
  - orientation = yaw-only(base_quat)

Run from simulate_python/:

    python3 deploy_mujoco/b2wz1_loco_manipulation_plb.py configs/b2wz1_loco_manipulation_plb.yaml --mode pd-stand
    python3 deploy_mujoco/b2wz1_loco_manipulation_plb.py configs/b2wz1_loco_manipulation_plb.yaml --mode lock-arm-policy
    python3 deploy_mujoco/b2wz1_loco_manipulation_plb.py configs/b2wz1_loco_manipulation_plb.yaml --mode full-policy
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
)
from utilities.mujoco_helper import (
    get_sensor_slice,
    make_arrow_mat,
)


class SequentialKeypointsTrajectoryCommandPLBSim:
    """
    Sequential trajectory sampler in PLB frame:
      - follow rows in npy sequentially
      - cubic interpolation between consecutive rows
      - optional hold at each waypoint
      - loop forever
    """

    def __init__(
        self,
        file_path: str,
        control_dt: float,
        traj_duration_s: float = 2.0,
        hold_duration_s: float = 1.0,
    ):
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(f"Expected npy shape (N,9), got {arr.shape} from '{file_path}'.")

        self.table = arr
        self.N = arr.shape[0]

        self.control_dt = float(control_dt)
        self.traj_duration_s = float(traj_duration_s)
        self.hold_duration_s = float(hold_duration_s)

        self.steps_per_traj = max(1, int(round(self.traj_duration_s / self.control_dt)))
        self.steps_per_hold = max(0, int(round(self.hold_duration_s / self.control_dt)))

        self.idx = 0
        self.step = 0
        self.phase = "move"
        self._has_cmd = False

        self.current = self.table[0].copy()
        self.start = self.table[0].copy()
        self.target = self.table[0].copy()

    @property
    def command(self) -> np.ndarray:
        return self.current.copy()

    @staticmethod
    def cubic(t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        return 3.0 * t * t - 2.0 * t * t * t

    def reset(self, initial_kps_plb: np.ndarray, sample_first: bool = True):
        initial_kps_plb = np.asarray(initial_kps_plb, dtype=np.float32).reshape(9,)

        self.idx = 0
        self.step = 0
        self.phase = "move"
        self._has_cmd = True

        self.current = initial_kps_plb.copy()
        self.start = initial_kps_plb.copy()
        self.target = self.table[0].copy() if sample_first else initial_kps_plb.copy()

    def update(self) -> np.ndarray:
        if not self._has_cmd:
            raise RuntimeError("Command sampler not initialized. Call reset() first.")

        if self.phase == "move":
            tau = self.step / max(1, self.steps_per_traj)
            s = self.cubic(tau)

            self.current = ((1.0 - s) * self.start + s * self.target).astype(np.float32)

            self.step += 1
            if self.step >= self.steps_per_traj:
                self.current = self.target.copy()
                self.phase = "hold"
                self.step = 0

        elif self.phase == "hold":
            self.current = self.target.copy()
            self.step += 1

            if self.step >= self.steps_per_hold:
                self.phase = "move"
                self.step = 0
                self.idx = (self.idx + 1) % self.N
                self.start = self.target.copy()
                self.target = self.table[self.idx].copy()

        return self.current.copy()


class PresampledKeypointsDirectCommandPLBSim:
    """
    Directly sampled EE keypoint command in PLB frame.

    PLB frame:
      - origin = [base_x, base_y, ground_z]
      - orientation = yaw-only(base_quat)

    Behavior:
      1) Randomly sample one row from a presampled reachable PLB keypoint table
      2) Hold this sampled keypoint command until next resampling
      3) No cubic interpolation
      4) No adjacent-target threshold clipping
    """

    def __init__(
        self,
        file_path: str,
        control_dt: float,
        cycle_duration_s: float = 8.0,
        seed: int = 0,
    ):
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(f"Expected npy shape (N,9), got {arr.shape} from '{file_path}'.")

        self._table = arr
        self._num_rows = int(arr.shape[0])
        self._control_dt = float(control_dt)
        self._cycle_duration_s = float(cycle_duration_s)
        if self._cycle_duration_s <= 0.0:
            raise ValueError(f"Invalid cycle_duration_s={self._cycle_duration_s}")

        self._cycle_steps = max(1, int(round(self._cycle_duration_s / self._control_dt)))
        self._rng = np.random.default_rng(seed)

        self._has_cmd = False
        self._step_in_cycle = 0
        self.keypoints_command_plb = np.zeros(9, dtype=np.float32)

    @property
    def command(self) -> np.ndarray:
        return self.keypoints_command_plb.copy()

    def _pick_index(self) -> int:
        return int(self._rng.integers(0, self._num_rows))

    def _sample_new_command(self):
        self.keypoints_command_plb = self._table[self._pick_index()].copy().astype(np.float32)
        self._step_in_cycle = 0
        self._has_cmd = True

    def reset(self, initial_kps_plb: np.ndarray, sample_first: bool = True):
        initial_kps_plb = np.asarray(initial_kps_plb, dtype=np.float32).reshape(9,)
        self._step_in_cycle = 0
        self._has_cmd = True

        if sample_first:
            self._sample_new_command()
        else:
            self.keypoints_command_plb = initial_kps_plb.copy()

    def update(self) -> np.ndarray:
        if not self._has_cmd:
            raise RuntimeError("Command sampler not initialized. Call reset() first.")

        self._step_in_cycle += 1
        if self._step_in_cycle >= self._cycle_steps:
            self._sample_new_command()

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

    enable_arm_target_rate_limit = bool(cfg.get("enable_arm_target_rate_limit", False))
    arm_target_rate_limit = np.array(
        cfg.get("arm_target_rate_limit", [np.inf] * 6),
        dtype=np.float32,
    )

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
    ee_ground_z = float(cfg.get("ee_ground_z", cfg.get("ground_z", 0.0)))
    ee_command_seed = int(cfg.get("ee_command_seed", 0))
    ee_cycle_duration_s = float(cfg.get("ee_cycle_duration_s", 8.0))

    startup_hold_s = float(cfg.get("startup_hold_s", 1.0))
    startup_blend_s = float(cfg.get("startup_blend_s", 2.0))

    vis_ee_target = bool(cfg.get("vis_ee_target", True))
    vis_ee_target_axis_len = float(cfg.get("vis_ee_target_axis_len", 0.20))
    vis_ee_target_axis_radius = float(cfg.get("vis_ee_target_axis_radius", 0.01))
    vis_ee_target_sphere_size = float(cfg.get("vis_ee_target_sphere_size", 0.03))
    vis_ee_target_alpha = float(cfg.get("vis_ee_target_alpha", 0.9))
    vis_ee_target_show_current = bool(cfg.get("vis_ee_target_show_current", True))

    assert obs_dim_per_step == 80, f"Expected obs_dim_per_step=80, got {obs_dim_per_step}"
    assert obs_dim == obs_dim_per_step * history_length, f"Expected obs_dim={obs_dim_per_step * history_length}, got {obs_dim}"
    assert action_dim == 22, f"Expected action_dim=22, got {action_dim}"
    assert len(default_joint_pos) == 23
    assert len(kps) == 23
    assert len(kds) == 23
    assert len(leg_torque_limits) == 12
    assert len(arm_torque_limits) == 6
    assert len(wheel_velocity_limits) == 4
    assert arm_action_scale.shape == (6,), f"Expected arm_action_scale shape (6,), got {arm_action_scale.shape}"
    assert arm_target_rate_limit.shape == (6,), (
        f"Expected arm_target_rate_limit shape (6,), got {arm_target_rate_limit.shape}"
    )
    if enable_arm_target_rate_limit and np.any(arm_target_rate_limit <= 0.0):
        raise ValueError(
            f"arm_target_rate_limit must be positive when enabled, got {arm_target_rate_limit}."
        )

    ee_command_mode = str(cfg.get("ee_command_mode", "direct"))

    print("=" * 80)
    print("B2WZ1 Loco-Manipulation - ONNX Policy (MuJoCo sim2sim, PLB whole-body)")
    print("=" * 80)
    print(f"Mode:            {args.mode}")
    print(f"Policy:          {policy_path}")
    print(f"XML:             {xml_path}")
    print(f"Control freq:    {1.0 / control_dt:.1f} Hz")
    print(f"Obs per step:    {obs_dim_per_step}")
    print(f"Obs stacked dim: {obs_dim}")
    print(f"Action dim:      {action_dim}")
    print(f"Leg scale:       {leg_action_scale}")
    print(f"Arm scales:      {arm_action_scale}")
    print(f"Arm target RL:   {enable_arm_target_rate_limit}")
    if enable_arm_target_rate_limit:
        print(f"Arm target rate: {arm_target_rate_limit} rad/s")
    print(f"Wheel scale:     {wheel_action_scale}")
    print(f"EE mode:         {ee_command_mode}")
    print(f"EE frame:        PLB")
    print(f"EE ground_z:     {ee_ground_z:.3f}")
    print(f"EE path:         {ee_command_path}")
    if ee_command_mode == "direct":
        print(f"EE cycle:        {ee_cycle_duration_s:.2f}s")
        print("EE behavior:     direct sample-and-hold")
    elif ee_command_mode == "sequential":
        print(
            f"EE cycle:        "
            f"{float(cfg.get('ee_sequential_traj_duration_s', 2.0)) + float(cfg.get('ee_sequential_hold_duration_s', 1.0)):.2f}s"
        )
        print("EE behavior:     sequential cubic interpolation")
    else:
        raise ValueError(f"Unsupported ee_command_mode: {ee_command_mode}")
    print(f"Vis EE target:   {vis_ee_target}")
    print("=" * 80)

    use_policy = args.mode in ["lock-arm-policy", "full-policy"]

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
    print("=" * 80)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    if ee_bid < 0:
        raise ValueError(f"Body not found: {ee_body_name}")

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

    leg_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    wheel_joint_names = ["FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"]

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

    control_source_joint_names = leg_joint_names + arm_joint_names + wheel_joint_names + ["jointGripper"]
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
    print("=" * 80)

    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0

    d.qpos[0:3] = root_pos
    d.qpos[3:7] = root_quat
    d.qpos[7:7 + len(mujoco_joint_names)] = default_joint_pos

    mujoco.mj_forward(m, d)
    print(f"Initialized height z = {d.qpos[2]:.3f} m")

    def get_projected_level_base_pose_world():
        base_pos_w = d.qpos[0:3].copy().astype(np.float32)
        base_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))

        _, _, yaw = euler_xyz_from_quat_wxyz(base_quat_w)
        plb_quat_w = quat_from_yaw_wxyz(yaw)
        plb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(plb_quat_w))

        plb_pos_w = base_pos_w.copy()
        plb_pos_w[2] = ee_ground_z

        return plb_pos_w, plb_quat_w

    def compute_ee_current_kp_plb() -> np.ndarray:
        plb_pos_w, plb_quat_w = get_projected_level_base_pose_world()

        ee_pos_w = d.xpos[ee_bid].copy().astype(np.float32)
        ee_rot_w = d.xmat[ee_bid].reshape(3, 3).copy().astype(np.float32)
        ee_quat_w = quat_from_rotmat_wxyz(ee_rot_w)

        ee_pos_plb = quat_apply_inverse_wxyz(plb_quat_w, ee_pos_w - plb_pos_w)
        ee_quat_plb = quat_mul_wxyz(quat_conjugate_wxyz(plb_quat_w), ee_quat_w)
        ee_quat_plb = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_plb))

        off_x = np.array([ee_kp_dx, 0.0, 0.0], dtype=np.float32)
        off_z = np.array([0.0, 0.0, ee_kp_dz], dtype=np.float32)

        kp0 = ee_pos_plb
        kp1 = ee_pos_plb + quat_apply_wxyz(ee_quat_plb, off_x)
        kp2 = ee_pos_plb + quat_apply_wxyz(ee_quat_plb, off_z)
        return np.concatenate([kp0, kp1, kp2]).astype(np.float32)

    def ee_kp_plb_to_world(ee_kp_plb: np.ndarray):
        plb_pos_w, plb_quat_w = get_projected_level_base_pose_world()

        kp0_plb = ee_kp_plb[0:3].astype(np.float32)
        kp1_plb = ee_kp_plb[3:6].astype(np.float32)
        kp2_plb = ee_kp_plb[6:9].astype(np.float32)

        kp0_w = plb_pos_w + quat_apply_wxyz(plb_quat_w, kp0_plb)
        kp1_w = plb_pos_w + quat_apply_wxyz(plb_quat_w, kp1_plb)
        kp2_w = plb_pos_w + quat_apply_wxyz(plb_quat_w, kp2_plb)

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

    def visualize_ee_pose(
        scene,
        ee_kp_plb: np.ndarray,
        axis_len: float,
        axis_radius: float,
        sphere_size: float,
        alpha: float,
        pos_rgba,
        x_rgba,
        z_rgba,
    ):
        kp0_w, kp1_w, kp2_w = ee_kp_plb_to_world(ee_kp_plb)

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

    def update_custom_visualization(viewer, ee_cmd_plb: np.ndarray):
        viewer.user_scn.ngeom = 0

        if not vis_ee_target:
            return

        visualize_ee_pose(
            viewer.user_scn,
            ee_kp_plb=ee_cmd_plb,
            axis_len=vis_ee_target_axis_len,
            axis_radius=vis_ee_target_axis_radius,
            sphere_size=vis_ee_target_sphere_size,
            alpha=vis_ee_target_alpha,
            pos_rgba=[1.0, 0.0, 0.0, vis_ee_target_alpha],
            x_rgba=[1.0, 0.0, 0.0, vis_ee_target_alpha],
            z_rgba=[1.0, 0.0, 0.0, vis_ee_target_alpha],
        )

        if vis_ee_target_show_current:
            ee_cur_plb = compute_ee_current_kp_plb()
            visualize_ee_pose(
                viewer.user_scn,
                ee_kp_plb=ee_cur_plb,
                axis_len=vis_ee_target_axis_len * 0.85,
                axis_radius=vis_ee_target_axis_radius * 0.8,
                sphere_size=vis_ee_target_sphere_size * 0.85,
                alpha=min(1.0, vis_ee_target_alpha),
                pos_rgba=[0.0, 0.0, 1.0, vis_ee_target_alpha],
                x_rgba=[0.0, 0.0, 1.0, vis_ee_target_alpha],
                z_rgba=[0.0, 0.0, 1.0, vis_ee_target_alpha],
            )

    if ee_command_mode == "sequential":
        ee_cmd_sampler = SequentialKeypointsTrajectoryCommandPLBSim(
            file_path=ee_command_path,
            control_dt=control_dt,
            traj_duration_s=float(cfg.get("ee_sequential_traj_duration_s", 2.0)),
            hold_duration_s=float(cfg.get("ee_sequential_hold_duration_s", 1.0)),
        )
    elif ee_command_mode == "direct":
        ee_cmd_sampler = PresampledKeypointsDirectCommandPLBSim(
            file_path=ee_command_path,
            control_dt=control_dt,
            cycle_duration_s=ee_cycle_duration_s,
            seed=ee_command_seed,
        )
    else:
        raise ValueError(f"Unsupported ee_command_mode: {ee_command_mode}")

    ee_cur_init_plb = compute_ee_current_kp_plb()
    sample_first = False if args.mode == "pd-stand" else True
    ee_cmd_sampler.reset(initial_kps_plb=ee_cur_init_plb, sample_first=sample_first)
    ee_cmd_plb_current = ee_cmd_sampler.command.copy()

    last_action = np.zeros(action_dim, dtype=np.float32)

    prev_arm_target_buf = [default_arm_pos.copy().astype(np.float32)]

    def apply_arm_target_rate_limit(raw_target: np.ndarray) -> np.ndarray:
        """Rate-limit arm joint position target exactly like training action term.

        Limits the final joint target, not the raw policy action:
            limited = prev + clip(raw - prev, +/- rate_limit * control_dt)
        """
        raw_target = np.asarray(raw_target, dtype=np.float32).reshape(6,)
        if not enable_arm_target_rate_limit:
            prev_arm_target_buf[0] = raw_target.copy()
            return raw_target.copy()

        max_delta = arm_target_rate_limit * control_dt
        prev_arm_target = prev_arm_target_buf[0]
        delta = np.clip(raw_target - prev_arm_target, -max_delta, max_delta)
        limited_target = (prev_arm_target + delta).astype(np.float32)
        prev_arm_target_buf[0] = limited_target.copy()
        return limited_target

    def reset_arm_target_rate_limiter(target: np.ndarray | None = None):
        """Reset limiter state to avoid artificial jumps during non-policy modes/startup."""
        if target is None:
            target = default_arm_pos
        prev_arm_target_buf[0] = np.asarray(target, dtype=np.float32).reshape(6,).copy()

    def build_obs_step(ee_cmd_plb: np.ndarray) -> np.ndarray:
        qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy().astype(np.float32)
        qvel_mujoco = d.qvel[6:6 + len(mujoco_joint_names)].copy().astype(np.float32)

        base_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)

        base_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(base_quat_w, gravity_w).astype(np.float32)

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
                base_ang_vel_b,       # 3
                projected_gravity_b,  # 3
                base_command,         # 3
                ee_cmd_plb,           # 9
                joint_pos_leg_rel,    # 12
                joint_pos_arm_rel,    # 6
                joint_vel_leg,        # 12
                joint_vel_arm,        # 6
                joint_vel_wheel,      # 4
                last_action,          # 22
            ],
            dtype=np.float32,
        )

        assert obs.shape[0] == obs_dim_per_step, f"Obs dim mismatch: {obs.shape[0]} vs {obs_dim_per_step}"
        return obs

    leg_target = default_leg_pos.copy()
    arm_target = default_arm_pos.copy()
    wheel_cmd = np.zeros(4, dtype=np.float32)
    gripper_target = default_gripper_pos

    obs0 = build_obs_step(ee_cmd_plb_current)

    i = 0
    obs0_base_ang_vel = obs0[i:i + 3]; i += 3
    obs0_projected_gravity = obs0[i:i + 3]; i += 3
    obs0_base_cmd = obs0[i:i + 3]; i += 3
    obs0_ee_cmd = obs0[i:i + 9]; i += 9
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
        joint_pos_leg_hist.append(obs0_joint_pos_leg.copy())
        joint_pos_arm_hist.append(obs0_joint_pos_arm.copy())
        joint_vel_leg_hist.append(obs0_joint_vel_leg.copy())
        joint_vel_arm_hist.append(obs0_joint_vel_arm.copy())
        joint_vel_wheel_hist.append(obs0_joint_vel_wheel.copy())
        last_action_hist.append(obs0_last_action.copy())

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

            if counter % control_decimation == 0:
                if args.mode == "pd-stand":
                    ee_cmd_plb_current = ee_cmd_sampler.command.copy()
                else:
                    ee_cmd_plb_current = ee_cmd_sampler.update()

                obs_step = build_obs_step(ee_cmd_plb_current)

                i = 0
                curr_base_ang_vel = obs_step[i:i + 3]; i += 3
                curr_projected_gravity = obs_step[i:i + 3]; i += 3
                curr_base_cmd = obs_step[i:i + 3]; i += 3
                curr_ee_cmd = obs_step[i:i + 9]; i += 9
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
                    action = sess.run([output_name], {input_name: obs_stack[None, :]})[0][0].astype(np.float32)

                last_action[:] = action

                leg_act = action[leg_action_indices].copy()
                arm_act = action[arm_action_indices].copy()
                wheel_act = action[wheel_action_indices].copy()

                if args.mode == "pd-stand":
                    leg_target = default_leg_pos.copy()
                    arm_target = default_arm_pos.copy()
                    reset_arm_target_rate_limiter(arm_target)
                    wheel_cmd[:] = 0.0

                elif args.mode == "lock-arm-policy":
                    leg_target = default_leg_pos + blend * (leg_action_scale * leg_act)
                    arm_target = default_arm_pos.copy()
                    reset_arm_target_rate_limiter(arm_target)
                    wheel_cmd = blend * (wheel_action_scale * wheel_act)

                elif args.mode == "full-policy":
                    leg_target = default_leg_pos + blend * (leg_action_scale * leg_act)
                    raw_arm_target = default_arm_pos + blend * (arm_action_scale * arm_act)
                    arm_target = apply_arm_target_rate_limit(raw_arm_target)
                    wheel_cmd = blend * (wheel_action_scale * wheel_act)

                else:
                    raise ValueError(f"Unsupported mode: {args.mode}")

                policy_tick += 1

            update_custom_visualization(viewer, ee_cmd_plb_current)

            if counter % 200 == 0:
                current_leg_pos = d.qpos[7:7 + len(mujoco_joint_names)].copy()[leg_mujoco_indices]
                current_arm_pos = d.qpos[7:7 + len(mujoco_joint_names)].copy()[arm_mujoco_indices]
                leg_err = leg_target - current_leg_pos
                arm_err = arm_target - current_arm_pos
                ee_err_norm = np.linalg.norm((ee_cmd_plb_current - compute_ee_current_kp_plb()).reshape(3, 3), axis=1)

                extra_cmd_info = ""
                if ee_command_mode == "direct":
                    extra_cmd_info = f" | cmd_hold={ee_cycle_duration_s:.2f}s"
                if enable_arm_target_rate_limit:
                    extra_cmd_info += f" | arm_target_rl=on"

                print(
                    f"[{counter:6d}] "
                    f"t={sim_time:6.2f}s | "
                    f"z={d.qpos[2]:.3f} | "
                    f"blend={blend:.2f}{extra_cmd_info} | "
                    f"leg_act=[{last_action[leg_action_indices].min():+.2f},{last_action[leg_action_indices].max():+.2f}] | "
                    f"arm_act=[{last_action[arm_action_indices].min():+.2f},{last_action[arm_action_indices].max():+.2f}] | "
                    f"wheel_act=[{last_action[wheel_action_indices].min():+.2f},{last_action[wheel_action_indices].max():+.2f}]"
                )
                print(
                    f"           max_leg_err={np.max(np.abs(leg_err)):.4f} | "
                    f"max_arm_err={np.max(np.abs(arm_err)):.4f} | "
                    f"ee_err_plb=[{ee_err_norm[0]:.4f},{ee_err_norm[1]:.4f},{ee_err_norm[2]:.4f}] | "
                    f"max_ctrl={np.max(np.abs(d.ctrl[:])):.2f}"
                    f"ee_cmd_plb=[{ee_cmd_plb_current[0]:.3f},{ee_cmd_plb_current[1]:.3f},{ee_cmd_plb_current[2]:.3f}"
                )

            counter += 1
            viewer.sync()

            time_until_next_step = simulation_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
