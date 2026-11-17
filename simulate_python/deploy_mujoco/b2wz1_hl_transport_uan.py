"""MuJoCo sim2sim for the 6-D EE-pose-residual B2W/Z1 transport policy.

Matches the updated Isaac Lab four-canonical-pose training task:
  - physics / UAN: 250 Hz; frozen WBC: 50 Hz; HL Beta policy: 10 Hz;
  - 64-D/frame actor observation, 3-frame FEATURE-MAJOR history = 192-D;
  - HL action = [dx, dy, dz, dyaw, dpitch, droll] normalized in [-1,1];
  - base velocity reference is sampled independently every 3-5 s (no HL base action);
  - HIGH (reset-settled live EE kp0), LOW, LEFT, RIGHT sampled every 4-5 s;
  - position transitions use 1.5-s (HIGH/LEFT/RIGHT) / 2.0-s (LOW)
    fifth-order minimum-jerk; orientation references stay constant;
  - puck disk-frame center position visible to actor; no inference-time noise;
  - puck physical radius/height checked against the YAML and compiled MJCF.

Run from unitree_mujoco/simulate_python:
  python3 deploy_mujoco/b2wz1_hl_transport_uan.py \
      configs/b2wz1_hl_transport_uan.yaml

E4 sim2sim alignment (2026-11-14), see
simulate_python/transport_experiment/docs/ALIGNMENT_REPORT.md:
  - frozen WBC path corrected in the YAML (training WBC run 2026-10-27_12-47-31);
  - LEFT/RIGHT canonical x corrected to 0.50 in the YAML;
  - puck priority/solimp + multiccd in the MJCF (puck friction now effective);
  - disk-puck and base contact use the NET contact force between the two
    bodies, like the Isaac Lab ContactSensor force_matrix_w / net_forces_w;
  - timeout after max_episode_length - 1 = 149 completed 10-Hz steps, exactly
    like DirectRLEnv (episode_length_buf >= max_episode_length - 1).
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import time
from collections import deque
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import torch
import yaml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)
from utilities.math import (  # noqa: E402
    euler_xyz_from_quat_wxyz,
    quat_apply_inverse_wxyz,
    quat_apply_wxyz,
    quat_from_rotmat_wxyz,
    quat_from_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_rotate_inverse_numpy,
    quat_unique_wxyz,
)
from utilities.mujoco_helper import get_sensor_slice, make_arrow_mat  # noqa: E402
def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))
def build_keypoints_from_kp0_yaw_pitch_roll_plb(
    kp0: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Exact NumPy equivalent of the dynamic-transport training adapter.
    Convention:
      kp1 = kp0 + kp_dx * local +X
      kp2 = kp0 + kp_dz * local +Z
    yaw/pitch define local +X directly.  Zero-roll local +Z is the most-upward
    direction orthogonal to +X.  roll then rotates +Z about local +X by the
    right-hand rule.
    """
    kp0 = np.asarray(kp0, dtype=np.float32).reshape(3,)
    yaw = float(yaw)
    pitch = float(pitch)
    roll = float(roll)
    cp = math.cos(pitch)
    vx = np.array(
        [cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)],
        dtype=np.float32,
    )
    vx /= max(float(np.linalg.norm(vx)), eps)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    vz0 = up - float(np.dot(up, vx)) * vx
    vz0_norm = float(np.linalg.norm(vz0))
    if vz0_norm > eps:
        vz0 /= vz0_norm
    else:
        vz0 = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)
        vz0 /= max(float(np.linalg.norm(vz0)), eps)
    vy0 = np.cross(vz0, vx).astype(np.float32)
    vy0 /= max(float(np.linalg.norm(vy0)), eps)
    vz = math.cos(roll) * vz0 - math.sin(roll) * vy0
    vz = vz.astype(np.float32)
    vz /= max(float(np.linalg.norm(vz)), eps)
    kp1 = kp0 + float(kp_dx) * vx
    kp2 = kp0 + float(kp_dz) * vz
    return np.concatenate([kp0, kp1, kp2]).astype(np.float32)
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
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
def add_capsule_to_scene(scene, p0, p1, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    diff = p1 - p0
    length = float(np.linalg.norm(diff))
    if length < 1.0e-8:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, 0.5 * length, 0.0], dtype=np.float64),
        0.5 * (p0 + p1),
        make_arrow_mat(diff).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="Path to hierarchical sim2sim YAML")
    args = parser.parse_args()
    yaml_path = os.path.abspath(args.yaml_path)
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    # ------------------------------------------------------------------
    # Paths / hierarchy / dimensions
    # ------------------------------------------------------------------
    low_policy_path = resolve_project_path(cfg["low_policy_path"])
    high_policy_path = resolve_project_path(cfg["high_policy_path"])
    uan_model_path = resolve_project_path(cfg["uan_model_path"])
    xml_path = resolve_project_path(cfg["xml_path"])
    simulation_duration = float(cfg["simulation_duration"])
    simulation_dt = float(cfg["simulation_dt"])
    ll_control_decimation = int(cfg["ll_control_decimation"])
    ll_steps_per_hl_step = int(cfg["ll_steps_per_hl_step"])
    ll_control_dt = simulation_dt * ll_control_decimation
    hl_control_dt = ll_control_dt * ll_steps_per_hl_step
    ll_history_length = int(cfg["ll_history_length"])
    ll_obs_dim_per_step = int(cfg["ll_obs_dim_per_step"])
    ll_obs_dim = int(cfg["ll_obs_dim"])
    ll_action_dim = int(cfg["ll_action_dim"])
    hl_history_length = int(cfg["hl_history_length"])
    hl_obs_dim_per_step = int(cfg["hl_obs_dim_per_step"])
    hl_obs_dim = int(cfg["hl_obs_dim"])
    hl_action_dim = int(cfg["hl_action_dim"])
    if abs(simulation_dt - 0.004) > 1.0e-9:
        raise ValueError(f"Expected 250-Hz physics dt=0.004, got {simulation_dt}")
    if ll_control_decimation != 5 or ll_steps_per_hl_step != 5:
        raise ValueError(
            "Expected hierarchy 250 Hz -> 50 Hz -> 10 Hz, i.e. decimations 5 and 5."
        )
    assert abs(1.0 / ll_control_dt - 50.0) < 1.0e-5
    assert abs(1.0 / hl_control_dt - 10.0) < 1.0e-5
    assert ll_obs_dim_per_step == 80
    assert ll_obs_dim == 400
    assert ll_action_dim == 22
    if hl_obs_dim_per_step != 64:
        raise ValueError("New transport actor requires hl_obs_dim_per_step=64.")
    if hl_history_length <= 0 or hl_obs_dim != hl_history_length * hl_obs_dim_per_step:
        raise ValueError(
            f"Inconsistent high-level history: hl_history_length={hl_history_length}, "
            f"hl_obs_dim_per_step={hl_obs_dim_per_step}, hl_obs_dim={hl_obs_dim}. "
            f"Expected hl_obs_dim={hl_history_length * hl_obs_dim_per_step}."
        )
    if hl_action_dim != 6:
        raise ValueError("New transport policy requires hl_action_dim=6.")
    for key, path in (
        ("low_policy_path", low_policy_path),
        ("high_policy_path", high_policy_path),
        ("uan_model_path", uan_model_path),
        ("xml_path", xml_path),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{key} does not exist: {path}")
    # ------------------------------------------------------------------
    # Robot / actuator configuration
    # ------------------------------------------------------------------
    root_pos_reset = np.asarray(cfg["root_pos"], dtype=np.float32)
    root_quat_reset = np.asarray(cfg["root_quat_wxyz"], dtype=np.float32)
    default_joint_pos = np.asarray(cfg["default_joint_pos"], dtype=np.float32)
    kps = np.asarray(cfg["kps"], dtype=np.float32)
    kds = np.asarray(cfg["kds"], dtype=np.float32)
    leg_action_scale = float(cfg["leg_action_scale"])
    arm_action_scale = np.asarray(cfg["arm_action_scale"], dtype=np.float32)
    wheel_action_scale = float(cfg["wheel_action_scale"])
    leg_torque_limits = np.asarray(cfg["leg_torque_limits"], dtype=np.float32)
    wheel_velocity_limits = np.asarray(cfg["wheel_velocity_limits"], dtype=np.float32)
    uan_device = str(cfg.get("uan_device", "cpu"))
    uan_history_length = int(cfg.get("uan_history_length", 25))
    uan_pos_scale = float(cfg.get("uan_pos_scale", 1.0))
    uan_vel_scale = float(cfg.get("uan_vel_scale", 0.05))
    uan_action_scale = float(cfg.get("uan_action_scale", 0.20))
    uan_kp = np.asarray(cfg["uan_kp"], dtype=np.float32)
    uan_kd = np.asarray(cfg["uan_kd"], dtype=np.float32)
    uan_nominal_torque_limits = np.asarray(
        cfg["uan_nominal_torque_limits"], dtype=np.float32
    )
    uan_final_torque_limits = np.asarray(
        cfg["uan_final_torque_limits"], dtype=np.float32
    )
    uan_action_saturation_threshold = float(
        cfg.get("uan_action_saturation_threshold", 0.95)
    )
    sync_mujoco_arm_limits_to_uan = bool(
        cfg.get("sync_mujoco_arm_limits_to_uan", True)
    )
    gripper_hold_pos = float(cfg["gripper_hold_pos"])
    gripper_mode = str(cfg.get("gripper_mode", "pd_position")).strip().lower()
    if gripper_mode not in ("dc_motor", "pd_position"):
        raise ValueError("gripper_mode must be 'dc_motor' or 'pd_position'.")
    gripper_torque_limit = float(cfg.get("gripper_torque_limit", 30.0))
    gripper_saturation_effort = float(
        cfg.get("gripper_saturation_effort", gripper_torque_limit)
    )
    gripper_velocity_limit = float(cfg.get("gripper_velocity_limit", 2.0))
    gripper_pd_kp_cfg = cfg.get("gripper_pd_kp", None)
    gripper_pd_kd_cfg = cfg.get("gripper_pd_kd", None)
    gripper_pd_torque_limit = float(
        cfg.get("gripper_pd_torque_limit", gripper_torque_limit)
    )
    # ------------------------------------------------------------------
    # Transport command/reference semantics
    # ------------------------------------------------------------------
    ground_z = float(cfg["ground_z"])
    ee_body_name = str(cfg.get("ee_body", "gripperStator"))
    ee_kp_dx = float(cfg["ee_kp_dx"])
    ee_kp_dz = float(cfg["ee_kp_dz"])
    neutral_kp0 = np.asarray(cfg["neutral_kp0"], dtype=np.float32)
    neutral_ee_yaw = float(cfg["neutral_ee_yaw"])
    neutral_ee_pitch = float(cfg["neutral_ee_pitch"])
    neutral_ee_roll = float(cfg["neutral_ee_roll"])
    base_cmd_scale = np.asarray(cfg["base_cmd_scale"], dtype=np.float32)
    ee_pos_action_residual_scale = np.asarray(
        cfg["ee_pos_action_residual_scale"], dtype=np.float32
    )
    ee_rot_action_residual_scale = np.asarray(
        cfg["ee_rot_action_residual_scale"], dtype=np.float32
    )
    kp0_x_range = np.asarray(cfg["kp0_x_range"], dtype=np.float32)
    kp0_y_range = np.asarray(cfg["kp0_y_range"], dtype=np.float32)
    kp0_z_range = np.asarray(cfg["kp0_z_range"], dtype=np.float32)
    ee_yaw_range = np.asarray(cfg["ee_yaw_range"], dtype=np.float32)
    ee_pitch_range = np.asarray(cfg["ee_pitch_range"], dtype=np.float32)
    ee_roll_range = np.asarray(cfg["ee_roll_range"], dtype=np.float32)
    reset_settle_time_s = float(cfg["reset_settle_time_s"])
    base_ref_resample_time_range_s = np.asarray(
        cfg["base_ref_resample_time_range_s"], dtype=np.float32
    )
    base_ref_ranges = np.asarray(cfg["base_ref_ranges"], dtype=np.float32)
    ee_ref_resample_time_range_s = np.asarray(
        cfg["ee_ref_resample_time_range_s"], dtype=np.float32
    )
    # Training uses 4-5 s, sampled independently for each episode/target.
    ee_fast_transition_time_s = float(cfg.get("ee_fast_transition_time_s", 1.5))
    ee_low_transition_time_s = float(cfg.get("ee_low_transition_time_s", 2.0))
    canonical_ee_positions = np.asarray(cfg["canonical_ee_positions"], dtype=np.float32)
    canonical_ee_yaw = float(cfg["canonical_ee_yaw"])
    canonical_ee_pitch = float(cfg["canonical_ee_pitch"])
    canonical_ee_roll = float(cfg["canonical_ee_roll"])
    settle_steps = int(round(reset_settle_time_s / hl_control_dt))
    base_ref_min_steps = int(round(float(base_ref_resample_time_range_s[0]) / hl_control_dt))
    base_ref_max_steps = int(round(float(base_ref_resample_time_range_s[1]) / hl_control_dt))
    ee_ref_min_steps = int(round(float(ee_ref_resample_time_range_s[0]) / hl_control_dt))
    ee_ref_max_steps = int(round(float(ee_ref_resample_time_range_s[1]) / hl_control_dt))
    ee_fast_transition_steps = int(round(ee_fast_transition_time_s / hl_control_dt))
    ee_low_transition_steps = int(round(ee_low_transition_time_s / hl_control_dt))
    if ee_ref_resample_time_range_s.shape != (2,) or ee_ref_min_steps <= 0 or ee_ref_max_steps < ee_ref_min_steps:
        raise ValueError("ee_ref_resample_time_range_s must be ordered positive [min, max].")
    if ee_fast_transition_steps <= 0:
        raise ValueError("ee_fast_transition_time_s must be positive.")
    if ee_low_transition_steps <= 0:
        raise ValueError("ee_low_transition_time_s must be positive.")
    if max(ee_fast_transition_steps, ee_low_transition_steps) > ee_ref_min_steps:
        raise ValueError("EE transitions must not exceed the minimum EE reference resampling period.")
    if canonical_ee_positions.shape != (4, 3):
        raise ValueError("canonical_ee_positions must have shape [4,3].")
    if base_ref_ranges.shape != (3, 2):
        raise ValueError("base_ref_ranges must have shape [3,2].")
    if np.any(base_ref_ranges[:, 0] < -base_cmd_scale) or np.any(
        base_ref_ranges[:, 1] > base_cmd_scale
    ):
        raise ValueError("base_ref_ranges must stay inside frozen-WBC base bounds.")
    ee_pos_cmd_min = np.array(
        [kp0_x_range[0], kp0_y_range[0], kp0_z_range[0]], dtype=np.float32
    )
    ee_pos_cmd_max = np.array(
        [kp0_x_range[1], kp0_y_range[1], kp0_z_range[1]], dtype=np.float32
    )
    ee_rot_cmd_min = np.array(
        [ee_yaw_range[0], ee_pitch_range[0], ee_roll_range[0]], dtype=np.float32
    )
    ee_rot_cmd_max = np.array(
        [ee_yaw_range[1], ee_pitch_range[1], ee_roll_range[1]], dtype=np.float32
    )
    # ------------------------------------------------------------------
    # Disk / puck
    # ------------------------------------------------------------------
    disk_center_offset_gripper = np.asarray(
        cfg["disk_center_offset_gripper"], dtype=np.float32
    )
    disk_x_axis_gripper = np.asarray(cfg["disk_x_axis_gripper"], dtype=np.float32)
    disk_y_axis_gripper = np.asarray(cfg["disk_y_axis_gripper"], dtype=np.float32)
    disk_z_axis_gripper = np.asarray(cfg["disk_z_axis_gripper"], dtype=np.float32)
    disk_thickness = float(cfg["disk_thickness"])
    puck_body_name = str(cfg.get("puck_body", "transport_puck"))
    puck_freejoint_name = str(cfg.get("puck_freejoint", "transport_puck_freejoint"))
    puck_geom_name = str(cfg.get("puck_geom", "transport_puck_geom"))
    disk_collision_geom_name = str(cfg.get("disk_collision_geom", "disk_collision"))
    puck_mass = float(cfg["puck_mass"])
    puck_radius = float(cfg["puck_radius"])
    puck_height = float(cfg["puck_height"])
    puck_friction = float(cfg.get("puck_friction", 0.60))
    puck_spawn_xy_randomization = float(cfg["puck_spawn_xy_randomization"])
    puck_spawn_clearance = float(cfg["puck_spawn_clearance"])
    puck_disk_contact_force_threshold = float(cfg["puck_disk_contact_force_threshold"])
    puck_drop_no_contact_steps = int(cfg["puck_drop_no_contact_steps"])
    base_contact_force_threshold = float(cfg.get("base_contact_force_threshold", 1.0))
    if min(puck_mass, puck_radius, puck_height) <= 0.0:
        raise ValueError("puck_mass, puck_radius and puck_height must all be positive.")
    if puck_friction < 0.0:
        raise ValueError("puck_friction must be non-negative.")
    if puck_drop_no_contact_steps <= 0:
        raise ValueError("puck_drop_no_contact_steps must be positive.")
    # ------------------------------------------------------------------
    # Episode / diagnostics
    # ------------------------------------------------------------------
    episode_length_s = float(cfg["episode_length_s"])
    # Isaac Lab DirectRLEnv: max_episode_length = ceil(T / step_dt); timeout is
    # raised when episode_length_buf >= max_episode_length - 1.
    max_episode_hl_steps = int(math.ceil(round(episode_length_s / hl_control_dt, 9)))
    auto_reset = bool(cfg.get("auto_reset", True))
    terminate_on_puck_drop = bool(cfg.get("terminate_on_puck_drop", True))
    terminate_on_base_contact = bool(cfg.get("terminate_on_base_contact", True))
    rng = np.random.default_rng(int(cfg.get("random_seed", 0)))
    vis_task_ee_reference = bool(cfg.get("vis_task_ee_reference", True))
    vis_wbc_ee_command = bool(cfg.get("vis_wbc_ee_command", True))
    vis_axis_len = float(cfg.get("vis_axis_len", 0.18))
    vis_axis_radius = float(cfg.get("vis_axis_radius", 0.008))
    vis_sphere_size = float(cfg.get("vis_sphere_size", 0.025))
    print_interval_sim_steps = int(cfg.get("print_interval_sim_steps", 200))
    debug_tracking_logging = bool(cfg.get("debug_tracking_logging", False))
    debug_tracking_log_every_ll_steps = int(
        cfg.get("debug_tracking_log_every_ll_steps", 1)
    )
    debug_tracking_csv_path = None
    if debug_tracking_logging:
        debug_tracking_csv_path = resolve_project_path(
            str(
                cfg.get(
                    "debug_tracking_csv_path",
                    "logs/b2wz1_hl_transport_uan_tracking_debug.csv",
                )
            )
        )
        os.makedirs(os.path.dirname(debug_tracking_csv_path), exist_ok=True)
    # Shape checks.
    assert root_pos_reset.shape == (3,)
    assert root_quat_reset.shape == (4,)
    assert default_joint_pos.shape == (23,)
    assert kps.shape == (23,)
    assert kds.shape == (23,)
    assert arm_action_scale.shape == (6,)
    assert leg_torque_limits.shape == (12,)
    assert wheel_velocity_limits.shape == (4,)
    assert uan_history_length == 25
    assert uan_kp.shape == (6,)
    assert uan_kd.shape == (6,)
    assert uan_nominal_torque_limits.shape == (6,)
    assert uan_final_torque_limits.shape == (6,)
    assert base_cmd_scale.shape == (3,)
    assert ee_pos_action_residual_scale.shape == (3,)
    assert ee_rot_action_residual_scale.shape == (3,)
    # ------------------------------------------------------------------
    # Networks
    # ------------------------------------------------------------------
    low_sess = ort.InferenceSession(low_policy_path, providers=["CPUExecutionProvider"])
    high_sess = ort.InferenceSession(high_policy_path, providers=["CPUExecutionProvider"])
    low_input_name = low_sess.get_inputs()[0].name
    low_output_name = low_sess.get_outputs()[0].name
    high_input_name = high_sess.get_inputs()[0].name
    high_output_name = high_sess.get_outputs()[0].name
    low_in_shape = low_sess.get_inputs()[0].shape
    low_out_shape = low_sess.get_outputs()[0].shape
    high_in_shape = high_sess.get_inputs()[0].shape
    high_out_shape = high_sess.get_outputs()[0].shape
    if isinstance(low_in_shape[-1], int):
        assert low_in_shape[-1] == ll_obs_dim, low_in_shape
    if isinstance(low_out_shape[-1], int):
        assert low_out_shape[-1] == ll_action_dim, low_out_shape
    if isinstance(high_in_shape[-1], int) and high_in_shape[-1] != hl_obs_dim:
        raise ValueError(
            f"High-level ONNX input dimension mismatch: model expects "
            f"{high_in_shape[-1]}, but YAML requests {hl_obs_dim} "
            f"({hl_history_length} frames x {hl_obs_dim_per_step} features). "
            "Verify that high_policy_path points to the policy exported from "
            "the current training run, not an older 5-frame checkpoint."
        )
    if isinstance(high_out_shape[-1], int):
        assert high_out_shape[-1] == hl_action_dim, high_out_shape
    if uan_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"UAN device {uan_device!r} requested but CUDA is unavailable.")
    uan_model = torch.jit.load(uan_model_path, map_location=uan_device).eval()
    for param in uan_model.parameters():
        param.requires_grad_(False)
    with torch.inference_mode():
        dummy_u = torch.zeros(
            (1, 6 * uan_history_length * 2),
            dtype=torch.float32,
            device=uan_device,
        )
        dummy_out = uan_model(dummy_u)
        if isinstance(dummy_out, (tuple, list)):
            dummy_out = dummy_out[0]
        if tuple(dummy_out.shape) != (1, 6):
            raise RuntimeError(f"Expected UAN output shape (1,6), got {tuple(dummy_out.shape)}")
    # ------------------------------------------------------------------
    # MuJoCo model and identifiers
    # ------------------------------------------------------------------
    m = mujoco.MjModel.from_xml_path(xml_path)
    m.opt.timestep = simulation_dt
    ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    puck_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, puck_body_name)
    puck_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, puck_freejoint_name)
    puck_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, puck_geom_name)
    # Standalone sim2sim needs an explicit collision floor in the MJCF.
    # Without it the floating-base robot falls under gravity, the disk leaves
    # the reset puck underneath/above it, and the 0.5-s no-contact detector
    # repeatedly triggers puck_drop during settle.
    ground_geom_name = str(cfg.get("ground_geom", "ground"))
    ground_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, ground_geom_name)
    if ground_gid < 0:
        raise ValueError(
            f"Ground geom not found: {ground_geom_name!r}. "
            "The transport sim2sim requires an explicit plane at z=0."
        )
    disk_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, disk_collision_geom_name)
    for what, idx in (
        (f"body {ee_body_name}", ee_bid),
        ("body base_link", base_bid),
        (f"body {puck_body_name}", puck_bid),
        (f"joint {puck_freejoint_name}", puck_jid),
        (f"geom {puck_geom_name}", puck_gid),
        (f"geom {disk_collision_geom_name}", disk_gid),
    ):
        if idx < 0:
            raise ValueError(f"MuJoCo {what} not found in {xml_path}")
    # MuJoCo geom.size for a cylinder is [radius, HALF-height, ...].
    # Fail loudly if the older 2.5-cm-radius / 5-cm-high MJCF is loaded:
    # changing puck_height in YAML alone does not change collision geometry.
    if m.geom_type[puck_gid] != mujoco.mjtGeom.mjGEOM_CYLINDER:
        raise ValueError("Expected puck collision geom to be a MuJoCo cylinder.")
    actual_r = float(m.geom_size[puck_gid, 0])
    actual_h = 2.0 * float(m.geom_size[puck_gid, 1])
    if abs(actual_r - puck_radius) > 1.0e-6 or abs(actual_h - puck_height) > 1.0e-6:
        raise ValueError(
            f"Puck MJCF collision size mismatch: radius={actual_r:.4f} m, "
            f"height={actual_h:.4f} m; YAML/training expects radius={puck_radius:.4f} m, "
            f"height={puck_height:.4f} m. Use the supplied four-pose grounded MJCF."
        )
    puck_xml_mass = float(m.body_mass[puck_bid])
    if puck_xml_mass <= 0.0:
        raise RuntimeError("Compiled puck reference mass must be > 0.")
    mass_ratio = puck_mass / puck_xml_mass
    # Analytic inertia of a solid homogeneous cylinder, about its COM,
    # aligned with puck local +Z. Correct for BOTH YAML mass and geometry.
    inertia_xy = puck_mass * (3.0 * puck_radius**2 + puck_height**2) / 12.0
    inertia_z = 0.5 * puck_mass * puck_radius**2
    m.body_mass[puck_bid] = puck_mass
    m.body_inertia[puck_bid] = np.array([inertia_xy, inertia_xy, inertia_z])
    # Sliding-friction coefficient (training range [0.3, 1.0]).
    m.geom_friction[puck_gid, 0] = puck_friction
    d = mujoco.MjData(m)
    mujoco.mj_setConst(m, d)
    puck_qpos_adr = int(m.jnt_qposadr[puck_jid])
    puck_dof_adr = int(m.jnt_dofadr[puck_jid])
    # Validated robot ordering from the retrieval sim2sim.  The newly appended
    # puck freejoint comes after these robot coordinates and does not alter them.
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
    policy_joint_pos_mujoco_indices = [mujoco_joint_names.index(n) for n in policy_joint_pos_names]
    policy_joint_vel_mujoco_indices = [mujoco_joint_names.index(n) for n in policy_joint_vel_names]
    leg_mujoco_indices = [mujoco_joint_names.index(n) for n in leg_joint_names]
    arm_mujoco_indices = [mujoco_joint_names.index(n) for n in arm_joint_names]
    wheel_mujoco_indices = [mujoco_joint_names.index(n) for n in wheel_joint_names]
    gripper_mujoco_index = mujoco_joint_names.index("jointGripper")
    leg_action_indices = np.arange(0, 12)
    arm_action_indices = np.arange(12, 18)
    wheel_action_indices = np.arange(18, 22)
    control_source_joint_names = leg_joint_names + arm_joint_names + wheel_joint_names + ["jointGripper"]
    ctrl_source_index_by_name = {name: i for i, name in enumerate(control_source_joint_names)}
    ctrl_src_indices_or_none = [
        ctrl_source_index_by_name[name] if name in ctrl_source_index_by_name else None
        for name in ctrl_joint_names
    ]
    default_joint_pos_policy_pos = default_joint_pos[policy_joint_pos_mujoco_indices]
    default_leg_pos = default_joint_pos[leg_mujoco_indices]
    default_arm_pos = default_joint_pos[arm_mujoco_indices]
    gripper_pd_kp = (
        float(gripper_pd_kp_cfg)
        if gripper_pd_kp_cfg is not None
        else float(kps[gripper_mujoco_index])
    )
    gripper_pd_kd = (
        float(gripper_pd_kd_cfg)
        if gripper_pd_kd_cfg is not None
        else float(kds[gripper_mujoco_index])
    )
    # Synchronize the MuJoCo Z1 effort limits to the frozen UAN envelope.
    print("MuJoCo arm torque-limit audit:")
    for j, joint_name in enumerate(arm_joint_names):
        ctrl_i = ctrl_joint_names.index(joint_name)
        final_lim = float(uan_final_torque_limits[j])
        if hasattr(m, "actuator_ctrllimited") and bool(m.actuator_ctrllimited[ctrl_i]):
            old_range = m.actuator_ctrlrange[ctrl_i].copy()
            if old_range[0] > -final_lim + 1e-6 or old_range[1] < final_lim - 1e-6:
                if not sync_mujoco_arm_limits_to_uan:
                    raise RuntimeError(
                        f"MuJoCo actuator ctrlrange for {joint_name} {old_range} "
                        f"is narrower than UAN final +/-{final_lim}."
                    )
                m.actuator_ctrlrange[ctrl_i, :] = (-final_lim, final_lim)
                print(f"  {joint_name}: ctrlrange {old_range} -> [-{final_lim}, +{final_lim}]")
        if hasattr(m, "actuator_forcelimited") and bool(m.actuator_forcelimited[ctrl_i]):
            old_range = m.actuator_forcerange[ctrl_i].copy()
            if old_range[0] > -final_lim + 1e-6 or old_range[1] < final_lim - 1e-6:
                if not sync_mujoco_arm_limits_to_uan:
                    raise RuntimeError(
                        f"MuJoCo actuator forcerange for {joint_name} {old_range} "
                        f"is narrower than UAN final +/-{final_lim}."
                    )
                m.actuator_forcerange[ctrl_i, :] = (-final_lim, final_lim)
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if (
            joint_id >= 0
            and hasattr(m, "jnt_actfrclimited")
            and hasattr(m, "jnt_actfrcrange")
            and bool(m.jnt_actfrclimited[joint_id])
        ):
            old_range = m.jnt_actfrcrange[joint_id].copy()
            if old_range[0] > -final_lim + 1e-6 or old_range[1] < final_lim - 1e-6:
                if not sync_mujoco_arm_limits_to_uan:
                    raise RuntimeError(
                        f"MuJoCo joint force range for {joint_name} {old_range} "
                        f"is narrower than UAN final +/-{final_lim}."
                    )
                m.jnt_actfrcrange[joint_id, :] = (-final_lim, final_lim)
    print("  PASS: MuJoCo will not clip inside the UAN final torque envelope.")
    print("=" * 92)
    print("B2WZ1 Dynamic Transport - Hierarchical WBC + UAN MuJoCo sim2sim")
    print(f"XML:               {xml_path}")
    print(f"Low policy:        {low_policy_path}")
    print(f"High policy:       {high_policy_path}")
    print(f"UAN model:         {uan_model_path}")
    print(f"Physics / UAN:     {1.0 / simulation_dt:.1f} Hz")
    print(f"Low-level WBC:     {1.0 / ll_control_dt:.1f} Hz, obs={ll_obs_dim}, act={ll_action_dim}")
    print(f"High-level:        {1.0 / hl_control_dt:.1f} Hz, obs={hl_obs_dim}, act={hl_action_dim}")
    print("Low ONNX:          ", low_in_shape, "->", low_out_shape)
    print("High ONNX:         ", high_in_shape, "->", high_out_shape)
    print(f"Puck mass:         {puck_mass:.4f} kg (XML nominal {puck_xml_mass:.4f} kg)")
    print(f"Puck radius/height:{puck_radius:.4f} / {puck_height:.4f} m; friction={puck_friction:.2f}")
    print(f"Ground geom:       {ground_geom_name} (id={ground_gid}, z=0 plane expected)")
    print(f"Puck inertia scale:{mass_ratio:.3f}x")
    print(f"Settle:            {settle_steps} HL steps = {settle_steps * hl_control_dt:.2f} s")
    print(f"Episode:           timeout after {max_episode_hl_steps - 1} HL steps (Isaac Lab semantics)")
    print(f"Base ref hold:     {base_ref_min_steps}-{base_ref_max_steps} HL steps")
    print(f"EE ref hold:       {ee_ref_min_steps}-{ee_ref_max_steps} HL steps (uniform)")
    print(
        f"EE transitions:    fast={ee_fast_transition_steps} HL steps "
        f"({ee_fast_transition_time_s:.2f}s), LOW={ee_low_transition_steps} HL steps "
        f"({ee_low_transition_time_s:.2f}s), minimum-jerk"
    )
    print("HL semantics:      base reference direct; EE pose reference + bounded 6-D residual")
    print(
        f"HL actor obs:      {hl_obs_dim_per_step}-D/frame x {hl_history_length} "
        f"feature-major = {hl_obs_dim}-D; puck center_D observed"
    )
    print("HIGH target:       measured EE kp0 at reset, re-measured after dynamic settle")
    print("=" * 92)
    # ------------------------------------------------------------------
    # State buffers
    # ------------------------------------------------------------------
    base_velocity_ref = np.zeros(3, dtype=np.float32)
    ee_kp0_ref = canonical_ee_positions[0].copy()
    ee_yaw_ref = canonical_ee_yaw
    ee_pitch_ref = canonical_ee_pitch
    ee_roll_ref = canonical_ee_roll

    # Per-episode HIGH kp0 in PLB.  canonical_ee_positions[0] is only the
    # nominal fallback/template until the live MuJoCo EE pose is measured.
    episode_high_kp0 = canonical_ee_positions[0].copy()

    current_ee_pose_index = 0
    settle_steps_remaining = 0
    base_ref_steps_remaining = 0
    ee_ref_steps_remaining = 0
    # EE minimum-jerk interpolation state.  ee_kp0_ref is always the CURRENT
    # reference seen by the HL actor and used by the residual decoder.
    ee_transition_start_kp0 = canonical_ee_positions[0].copy()
    ee_transition_target_kp0 = canonical_ee_positions[0].copy()
    ee_transition_step = 0
    ee_transition_total_steps = 1
    base_command = np.zeros(3, dtype=np.float32)
    kp0_cmd = ee_kp0_ref.copy()
    ee_yaw_cmd = ee_yaw_ref
    ee_pitch_cmd = ee_pitch_ref
    ee_roll_cmd = ee_roll_ref
    ee_cmd_plb_current = build_keypoints_from_kp0_yaw_pitch_roll_plb(
        kp0_cmd, ee_yaw_cmd, ee_pitch_cmd, ee_roll_cmd, ee_kp_dx, ee_kp_dz
    )
    raw_hl_action = np.zeros(hl_action_dim, dtype=np.float32)
    effective_hl_action = np.zeros(hl_action_dim, dtype=np.float32)
    executed_hl_action = np.zeros(hl_action_dim, dtype=np.float32)
    last_ll_action = np.zeros(ll_action_dim, dtype=np.float32)
    leg_target = default_leg_pos.copy()
    arm_target = default_arm_pos.copy()
    wheel_cmd = np.zeros(4, dtype=np.float32)
    gripper_target = gripper_hold_pos
    puck_no_contact_steps = 0
    puck_drop = False
    base_contact = False
    episode_start_time = 0.0
    episode_hl_steps = 0
    episode_index = 0
    # Feature-major history, exactly matching the training flattening.
    ll_feature_dims = [3, 3, 3, 9, 12, 6, 12, 6, 4, 22]
    hl_feature_dims = [3, 3, 12, 6, 12, 6, 4, 3, 3, 3, 6, 3]
    assert sum(ll_feature_dims) == ll_obs_dim_per_step
    assert sum(hl_feature_dims) == hl_obs_dim_per_step
    ll_histories = [deque(maxlen=ll_history_length) for _ in ll_feature_dims]
    hl_histories = [deque(maxlen=hl_history_length) for _ in hl_feature_dims]
    # UAN state: joint-major, oldest -> newest, [position_error, scaled_velocity].
    uan_history = np.zeros((6, uan_history_length, 2), dtype=np.float32)
    last_uan_action = np.zeros(6, dtype=np.float32)
    last_uan_delta_tau = np.zeros(6, dtype=np.float32)
    last_uan_tau_pd = np.zeros(6, dtype=np.float32)
    last_uan_tau_nom = np.zeros(6, dtype=np.float32)
    def split_features(frame: np.ndarray, dims: list[int]) -> list[np.ndarray]:
        out = []
        i = 0
        for dim in dims:
            out.append(frame[i : i + dim].copy())
            i += dim
        assert i == frame.shape[0]
        return out
    def fill_histories(histories, frame: np.ndarray, dims: list[int], length: int):
        for hist, feature in zip(histories, split_features(frame, dims)):
            hist.clear()
            for _ in range(length):
                hist.append(feature.copy())
    def append_histories(histories, frame: np.ndarray, dims: list[int]):
        for hist, feature in zip(histories, split_features(frame, dims)):
            hist.append(feature.copy())
    def flatten_feature_major(histories) -> np.ndarray:
        return np.concatenate(
            [np.asarray(hist, dtype=np.float32).reshape(-1) for hist in histories],
            dtype=np.float32,
        )
    def get_root_pose():
        pos_w = d.qpos[0:3].copy().astype(np.float32)
        quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        return pos_w, quat_w
    def get_plb_pose_world():
        root_pos_w, root_quat_w = get_root_pose()
        _, _, yaw = euler_xyz_from_quat_wxyz(root_quat_w)
        plb_quat_w = quat_unique_wxyz(
            quat_normalize_wxyz(quat_from_yaw_wxyz(yaw))
        )
        plb_pos_w = root_pos_w.copy()
        plb_pos_w[2] = ground_z
        return plb_pos_w, plb_quat_w
    def ee_kp_plb_to_world(kps_plb: np.ndarray):
        plb_pos_w, plb_quat_w = get_plb_pose_world()
        kps = np.asarray(kps_plb, dtype=np.float32).reshape(3, 3)
        return np.stack(
            [plb_pos_w + quat_apply_wxyz(plb_quat_w, p) for p in kps], axis=0
        )
    def compute_actual_ee_keypoints_plb():
        plb_pos_w, plb_quat_w = get_plb_pose_world()
        ee_pos_w = d.xpos[ee_bid].copy().astype(np.float32)
        ee_rot_w = d.xmat[ee_bid].reshape(3, 3).copy().astype(np.float32)
        kp0_w = ee_pos_w
        kp1_w = ee_pos_w + ee_rot_w @ np.array([ee_kp_dx, 0.0, 0.0], dtype=np.float32)
        kp2_w = ee_pos_w + ee_rot_w @ np.array([0.0, 0.0, ee_kp_dz], dtype=np.float32)
        return np.concatenate(
            [
                quat_apply_inverse_wxyz(plb_quat_w, kp0_w - plb_pos_w),
                quat_apply_inverse_wxyz(plb_quat_w, kp1_w - plb_pos_w),
                quat_apply_inverse_wxyz(plb_quat_w, kp2_w - plb_pos_w),
            ]
        ).astype(np.float32)
    def measure_ee_kp0_plb() -> np.ndarray:
        """Measure the live gripperStator origin (kp0) in the current PLB frame."""
        return compute_actual_ee_keypoints_plb()[0:3].copy().astype(np.float32)

    def sync_ll_commands_from_references():
        """Copy current task references directly to frozen-WBC commands.

        This is the zero-residual path used during reset/settle synchronization.
        """
        nonlocal base_command, kp0_cmd
        nonlocal ee_yaw_cmd, ee_pitch_cmd, ee_roll_cmd
        nonlocal ee_cmd_plb_current, gripper_target

        base_command = base_velocity_ref.copy()
        kp0_cmd = ee_kp0_ref.copy()
        ee_yaw_cmd = ee_yaw_ref
        ee_pitch_cmd = ee_pitch_ref
        ee_roll_cmd = ee_roll_ref
        ee_cmd_plb_current = build_keypoints_from_kp0_yaw_pitch_roll_plb(
            kp0_cmd,
            ee_yaw_cmd,
            ee_pitch_cmd,
            ee_roll_cmd,
            ee_kp_dx,
            ee_kp_dz,
        )
        gripper_target = gripper_hold_pos

    def set_episode_high_from_measured_ee(*, update_current_reference: bool):
        """Capture the live EE kp0 as this episode's HIGH position.

        At reset this gives the temporary settle-hold reference.  After the last
        settle transition it is measured again and becomes the final HIGH target
        used for the remainder of the episode.
        """
        nonlocal episode_high_kp0, ee_kp0_ref, current_ee_pose_index
        nonlocal ee_transition_start_kp0, ee_transition_target_kp0
        nonlocal ee_transition_step, ee_transition_total_steps

        measured_high = measure_ee_kp0_plb()
        episode_high_kp0 = measured_high.copy()

        if update_current_reference:
            ee_kp0_ref = measured_high.copy()
            current_ee_pose_index = 0
            ee_transition_start_kp0 = measured_high.copy()
            ee_transition_target_kp0 = measured_high.copy()
            ee_transition_step = 0
            ee_transition_total_steps = 1
            sync_ll_commands_from_references()

    def get_disk_frame_world():
        stator_pos_w = d.xpos[ee_bid].copy().astype(np.float32)
        stator_rot_w = d.xmat[ee_bid].reshape(3, 3).copy().astype(np.float32)
        center_w = stator_pos_w + stator_rot_w @ disk_center_offset_gripper
        x_w = stator_rot_w @ disk_x_axis_gripper
        y_w = stator_rot_w @ disk_y_axis_gripper
        z_w = stator_rot_w @ disk_z_axis_gripper
        R_w_d = np.stack([x_w, y_w, z_w], axis=1).astype(np.float32)
        q_w_d = quat_from_rotmat_wxyz(R_w_d)
        q_w_d = quat_unique_wxyz(quat_normalize_wxyz(q_w_d))
        return center_w.astype(np.float32), x_w.astype(np.float32), y_w.astype(np.float32), z_w.astype(np.float32), q_w_d
    def get_puck_pos_w():
        return d.xpos[puck_bid].copy().astype(np.float32)
    def get_puck_pos_disk():
        center_w, x_w, y_w, z_w, _ = get_disk_frame_world()
        rel = get_puck_pos_w() - center_w
        return np.array(
            [np.dot(rel, x_w), np.dot(rel, y_w), np.dot(rel, z_w)],
            dtype=np.float32,
        )
    def contact_force_for_pair(geom_a: int, geom_b: int) -> float:
        """Norm of the NET world-frame contact force between two geoms.

        Mirrors the PhysX filtered force matrix (net force between two bodies).
        With multiccd the flat puck/disk contact is split over several points,
        so the net force (not the per-point maximum) is the comparable quantity.
        """
        force6 = np.zeros(6, dtype=np.float64)
        net = np.zeros(3, dtype=np.float64)
        for ci in range(d.ncon):
            c = d.contact[ci]
            if (c.geom1 == geom_a and c.geom2 == geom_b) or (
                c.geom1 == geom_b and c.geom2 == geom_a
            ):
                mujoco.mj_contactForce(m, d, ci, force6)
                # mj_contactForce/frame give the force acting ON geom2.
                f_w = c.frame.reshape(3, 3).T @ force6[:3]
                net += f_w if c.geom2 == geom_b else -f_w
        return float(np.linalg.norm(net))
    def puck_disk_contact_now() -> tuple[bool, float]:
        f = contact_force_for_pair(disk_gid, puck_gid)
        return f > puck_disk_contact_force_threshold, f
    def base_contact_now() -> tuple[bool, float]:
        """Net world-frame contact force acting on base_link (any partner)."""
        force6 = np.zeros(6, dtype=np.float64)
        net = np.zeros(3, dtype=np.float64)
        for ci in range(d.ncon):
            c = d.contact[ci]
            b1 = int(m.geom_bodyid[c.geom1])
            b2 = int(m.geom_bodyid[c.geom2])
            if b1 == base_bid or b2 == base_bid:
                mujoco.mj_contactForce(m, d, ci, force6)
                f_w = c.frame.reshape(3, 3).T @ force6[:3]
                # Force reported on geom2 by geom1 convention; sign it so the
                # sum is the force acting ON base_link.
                net += f_w if b2 == base_bid else -f_w
        best = float(np.linalg.norm(net))
        return best > base_contact_force_threshold, best
    def sample_base_velocity_reference():
        nonlocal base_velocity_ref, base_ref_steps_remaining
        low = base_ref_ranges[:, 0]
        high = base_ref_ranges[:, 1]
        base_velocity_ref = rng.uniform(low, high).astype(np.float32)
        if base_ref_min_steps == base_ref_max_steps:
            base_ref_steps_remaining = base_ref_min_steps
        else:
            base_ref_steps_remaining = int(
                rng.integers(base_ref_min_steps, base_ref_max_steps + 1)
            )
    def minimum_jerk_blend(tau: float) -> float:
        """Fifth-order smoothstep with zero velocity/acceleration at both endpoints."""
        tau = float(np.clip(tau, 0.0, 1.0))
        return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
    def sample_ee_pose_reference():
        """Sample a canonical target and start the training-matched interpolation.
        Sampling cadence is independent U(4,5)-second intervals, matching
        training. HIGH/LOW/LEFT/RIGHT are uniform; repeated targets allowed.
        The current reference
        does NOT jump.  It becomes the start point of a minimum-jerk trajectory.
        Duration depends only on the newly sampled target:
          HIGH / LEFT / RIGHT -> ee_fast_transition_time_s
          LOW                 -> ee_low_transition_time_s
        """
        nonlocal ee_kp0_ref, ee_yaw_ref, ee_pitch_ref, ee_roll_ref
        nonlocal current_ee_pose_index, ee_ref_steps_remaining
        nonlocal ee_transition_start_kp0, ee_transition_target_kp0
        nonlocal ee_transition_step, ee_transition_total_steps
        current_ee_pose_index = int(rng.integers(0, canonical_ee_positions.shape[0]))
        if current_ee_pose_index == 0:
            # HIGH is environment/episode specific: use the dynamically measured
            # settled reset EE position, not the nominal locked-FK template.
            target_kp0 = episode_high_kp0.copy()
        else:
            target_kp0 = canonical_ee_positions[current_ee_pose_index].copy()

        # Start exactly from the reference that was visible/executed immediately
        # before this resample, preserving reference continuity.
        ee_transition_start_kp0 = ee_kp0_ref.copy()
        ee_transition_target_kp0 = target_kp0
        ee_transition_step = 0
        ee_transition_total_steps = (
            ee_low_transition_steps if current_ee_pose_index == 1
            else ee_fast_transition_steps
        )
        # tau=0: keep current position reference unchanged.  It advances only
        # after each completed 10-Hz transition, matching training ordering.
        ee_kp0_ref = ee_transition_start_kp0.copy()
        # Orientation is unchanged because all four canonical poses share it.
        ee_yaw_ref = canonical_ee_yaw
        ee_pitch_ref = canonical_ee_pitch
        ee_roll_ref = canonical_ee_roll
        # The next target is resampled after a new inclusive [4,5] s hold.
        if ee_ref_min_steps == ee_ref_max_steps:
            ee_ref_steps_remaining = ee_ref_min_steps
        else:
            ee_ref_steps_remaining = int(
                rng.integers(ee_ref_min_steps, ee_ref_max_steps + 1)
            )
    def advance_ee_pose_interpolation():
        """Advance CURRENT EE position reference by one 10-Hz minimum-jerk sample."""
        nonlocal ee_kp0_ref, ee_transition_step
        total_steps = max(int(ee_transition_total_steps), 1)
        ee_transition_step = min(int(ee_transition_step) + 1, total_steps)
        tau = float(ee_transition_step) / float(total_steps)
        blend = minimum_jerk_blend(tau)
        ee_kp0_ref = (
            ee_transition_start_kp0
            + blend * (ee_transition_target_kp0 - ee_transition_start_kp0)
        ).astype(np.float32)
    def reset_command_references():
        nonlocal base_velocity_ref, ee_kp0_ref, ee_yaw_ref, ee_pitch_ref, ee_roll_ref
        nonlocal episode_high_kp0, current_ee_pose_index
        nonlocal settle_steps_remaining, base_ref_steps_remaining, ee_ref_steps_remaining
        nonlocal ee_transition_start_kp0, ee_transition_target_kp0
        nonlocal ee_transition_step, ee_transition_total_steps
        base_velocity_ref = np.zeros(3, dtype=np.float32)
        episode_high_kp0 = canonical_ee_positions[0].copy()
        ee_kp0_ref = episode_high_kp0.copy()
        ee_yaw_ref = canonical_ee_yaw
        ee_pitch_ref = canonical_ee_pitch
        ee_roll_ref = canonical_ee_roll
        current_ee_pose_index = 0
        ee_transition_start_kp0 = episode_high_kp0.copy()
        ee_transition_target_kp0 = episode_high_kp0.copy()
        ee_transition_step = 0
        ee_transition_total_steps = 1
        settle_steps_remaining = settle_steps
        base_ref_steps_remaining = 0
        ee_ref_steps_remaining = 0
    def advance_command_references_after_transition():
        nonlocal settle_steps_remaining, base_ref_steps_remaining, ee_ref_steps_remaining
        if settle_steps_remaining > 0:
            settle_steps_remaining -= 1
            if settle_steps_remaining <= 0:
                # Match training: after the final 0.1-s settle transition, use
                # the live dynamically settled MuJoCo EE kp0 in PLB as this
                # episode's final HIGH target.  Keep orientation canonical.
                set_episode_high_from_measured_ee(
                    update_current_reference=True
                )

                # Then sample the first active targets.  EE interpolation starts
                # at tau=0 from the just-measured settled HIGH.
                sample_base_velocity_reference()
                sample_ee_pose_reference()
            return
        # The CURRENT references have just been executed for one HL step.
        base_ref_steps_remaining -= 1
        ee_ref_steps_remaining -= 1
        # Advance interpolation AFTER executing the previous reference and BEFORE
        # constructing the next actor observation.  This exactly preserves:
        # obs_t(ref_t) -> action_t -> execute_t -> advance -> obs_(t+1)(ref_(t+1)).
        advance_ee_pose_interpolation()
        if base_ref_steps_remaining <= 0:
            sample_base_velocity_reference()
        if ee_ref_steps_remaining <= 0:
            sample_ee_pose_reference()
    def decode_hl_action(action: np.ndarray):
        """Decode the CURRENT 6-D action against the CURRENT task references.

        This updates executed_hl_action immediately after decoding. The next
        high-level observation will read that value as the previous action.
        """
        nonlocal raw_hl_action, effective_hl_action, executed_hl_action
        nonlocal base_command, kp0_cmd, ee_yaw_cmd, ee_pitch_cmd, ee_roll_cmd
        nonlocal ee_cmd_plb_current, gripper_target
        action = np.asarray(action, dtype=np.float32).reshape(hl_action_dim,)
        if not np.isfinite(action).all():
            raise FloatingPointError(f"Non-finite high-level action: {action}")
        raw_hl_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        effective_hl_action = raw_hl_action.copy()
        if settle_steps_remaining > 0:
            effective_hl_action[:] = 0.0

        # Base command is exclusively exogenous, with NO learned residual.
        base_command = base_velocity_ref.copy()

        # EE kp0 position in PLB: normalized action[:3] -> +/-10 cm by default.
        pos_raw = ee_kp0_ref + effective_hl_action[0:3] * ee_pos_action_residual_scale
        kp0_cmd = np.clip(pos_raw, ee_pos_cmd_min, ee_pos_cmd_max).astype(np.float32)
        # Training deliberately holds the *measured* EE kp0 exactly during settle.
        if settle_steps_remaining > 0:
            kp0_cmd = ee_kp0_ref.copy()

        # EE orientation, training adapter convention [yaw, pitch, roll].
        rot_ref = np.array([ee_yaw_ref, ee_pitch_ref, ee_roll_ref], dtype=np.float32)
        rot_raw = rot_ref + effective_hl_action[3:6] * ee_rot_action_residual_scale
        rot_cmd = np.clip(rot_raw, ee_rot_cmd_min, ee_rot_cmd_max).astype(np.float32)
        ee_yaw_cmd, ee_pitch_cmd, ee_roll_cmd = map(float, rot_cmd)

        # Executed normalized residual (AFTER WBC absolute command bounds).
        # Observation o_(t+1) sees exactly a_t^exec, not a_(t-1)^exec.
        executed_hl_action = np.clip(
            np.concatenate([
                (kp0_cmd - ee_kp0_ref) / ee_pos_action_residual_scale,
                (rot_cmd - rot_ref) / ee_rot_action_residual_scale,
            ]),
            -1.0,
            1.0,
        ).astype(np.float32)

        ee_cmd_plb_current = build_keypoints_from_kp0_yaw_pitch_roll_plb(
            kp0_cmd, ee_yaw_cmd, ee_pitch_cmd, ee_roll_cmd, ee_kp_dx, ee_kp_dz
        )
        gripper_target = gripper_hold_pos

    def build_hl_obs_frame():
        root_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        root_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(root_quat_w, gravity_w).astype(np.float32)
        qpos_mujoco = d.qpos[7 : 7 + len(mujoco_joint_names)].copy().astype(np.float32)
        qvel_mujoco = d.qvel[6 : 6 + len(mujoco_joint_names)].copy().astype(np.float32)
        leg_pos_rel = (
            qpos_mujoco[leg_mujoco_indices] - default_joint_pos[leg_mujoco_indices]
        ).astype(np.float32)
        arm_pos_rel = (
            qpos_mujoco[arm_mujoco_indices] - default_joint_pos[arm_mujoco_indices]
        ).astype(np.float32)
        leg_joint_vel = qvel_mujoco[leg_mujoco_indices].astype(np.float32)
        arm_joint_vel = qvel_mujoco[arm_mujoco_indices].astype(np.float32)
        wheel_joint_vel = qvel_mujoco[wheel_mujoco_indices].astype(np.float32)
        ee_orientation_ref = np.array(
            [ee_yaw_ref, ee_pitch_ref, ee_roll_ref], dtype=np.float32
        )
        # The training actor observes puck center position in the live disk frame.
        # Training applies up to 1 cm
        # component-wise uniform noise to this channel; sim2sim deployment uses
        # the exact MuJoCo state (equivalent to noiseless play/evaluation mode).
        puck_center_pos_disk = get_puck_pos_disk()
        obs = np.concatenate(
            [
                root_ang_vel_b,          # 3
                projected_gravity_b,     # 3
                leg_pos_rel,             # 12
                arm_pos_rel,             # 6
                leg_joint_vel,           # 12
                arm_joint_vel,           # 6
                wheel_joint_vel,         # 4
                base_velocity_ref,       # 3
                ee_kp0_ref,              # 3
                ee_orientation_ref,      # 3
                executed_hl_action,      # 6
                puck_center_pos_disk,    # 3
            ],
            dtype=np.float32,
        )
        assert obs.shape == (hl_obs_dim_per_step,), obs.shape
        return obs
    def build_ll_obs_frame():
        qpos_mujoco = d.qpos[7 : 7 + len(mujoco_joint_names)].copy().astype(np.float32)
        qvel_mujoco = d.qvel[6 : 6 + len(mujoco_joint_names)].copy().astype(np.float32)
        base_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)
        root_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(root_quat_w, gravity_w).astype(np.float32)
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
                base_ang_vel_b,
                projected_gravity_b,
                base_command,
                ee_cmd_plb_current,
                joint_pos_leg_rel,
                joint_pos_arm_rel,
                joint_vel_leg,
                joint_vel_arm,
                joint_vel_wheel,
                last_ll_action,
            ],
            dtype=np.float32,
        )
        assert obs.shape == (ll_obs_dim_per_step,), obs.shape
        return obs
    def compute_uan_arm_torque(q_des, q, qd):
        q_des = np.asarray(q_des, dtype=np.float32).reshape(6,)
        q = np.asarray(q, dtype=np.float32).reshape(6,)
        qd = np.asarray(qd, dtype=np.float32).reshape(6,)
        pos_err = (q_des - q) * uan_pos_scale
        vel = qd * uan_vel_scale
        uan_history[:, :-1, :] = uan_history[:, 1:, :]
        uan_history[:, -1, 0] = pos_err
        uan_history[:, -1, 1] = vel
        obs_np = np.ascontiguousarray(uan_history.reshape(1, -1), dtype=np.float32)
        obs_t = torch.from_numpy(obs_np).to(device=uan_device)
        with torch.inference_mode():
            out = uan_model(obs_t)
            if isinstance(out, (tuple, list)):
                out = out[0]
        action = out.detach().cpu().numpy().reshape(6,).astype(np.float32)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError(f"Non-finite UAN action: {action}")
        if np.max(np.abs(action)) > 1.001:
            raise RuntimeError(f"UAN action outside expected [-1,1]: {action}")
        tau_pd = uan_kp * (q_des - q) - uan_kd * qd
        tau_nom = np.clip(tau_pd, -uan_nominal_torque_limits, uan_nominal_torque_limits)
        delta_tau = action * uan_action_scale * uan_kp
        tau = np.clip(
            tau_nom + delta_tau, -uan_final_torque_limits, uan_final_torque_limits
        ).astype(np.float32)
        last_uan_action[:] = action
        last_uan_delta_tau[:] = delta_tau
        last_uan_tau_pd[:] = tau_pd
        last_uan_tau_nom[:] = tau_nom
        return tau
    def run_high_level_policy():
        hl_frame = build_hl_obs_frame()
        append_histories(hl_histories, hl_frame, hl_feature_dims)
        hl_obs_stack = flatten_feature_major(hl_histories)
        assert hl_obs_stack.shape == (hl_obs_dim,), hl_obs_stack.shape
        action = high_sess.run(
            [high_output_name], {high_input_name: hl_obs_stack[None, :]}
        )[0][0].astype(np.float32)
        decode_hl_action(action)
    def run_low_level_policy():
        nonlocal last_ll_action, leg_target, arm_target, wheel_cmd
        ll_frame = build_ll_obs_frame()
        append_histories(ll_histories, ll_frame, ll_feature_dims)
        ll_obs_stack = flatten_feature_major(ll_histories)
        assert ll_obs_stack.shape == (ll_obs_dim,), ll_obs_stack.shape
        action = low_sess.run(
            [low_output_name], {low_input_name: ll_obs_stack[None, :]}
        )[0][0].astype(np.float32)
        last_ll_action = action
        leg_target = default_leg_pos + leg_action_scale * action[leg_action_indices]
        arm_target = default_arm_pos + arm_action_scale * action[arm_action_indices]
        wheel_cmd = wheel_action_scale * action[wheel_action_indices]
    def apply_low_level_actuation():
        qpos_mujoco = d.qpos[7 : 7 + len(mujoco_joint_names)].copy().astype(np.float32)
        qvel_mujoco = d.qvel[6 : 6 + len(mujoco_joint_names)].copy().astype(np.float32)
        leg_pos = qpos_mujoco[leg_mujoco_indices]
        leg_vel = qvel_mujoco[leg_mujoco_indices]
        arm_pos = qpos_mujoco[arm_mujoco_indices]
        arm_vel = qvel_mujoco[arm_mujoco_indices]
        gripper_pos = float(qpos_mujoco[gripper_mujoco_index])
        gripper_vel = float(qvel_mujoco[gripper_mujoco_index])
        leg_tau = kps[leg_mujoco_indices] * (leg_target - leg_pos) - kds[leg_mujoco_indices] * leg_vel
        leg_tau = np.clip(leg_tau, -leg_torque_limits, leg_torque_limits).astype(np.float32)
        arm_tau = compute_uan_arm_torque(arm_target, arm_pos, arm_vel)
        if gripper_mode == "pd_position":
            gripper_tau = float(
                np.clip(
                    gripper_pd_kp * (gripper_target - gripper_pos) - gripper_pd_kd * gripper_vel,
                    -gripper_pd_torque_limit,
                    gripper_pd_torque_limit,
                )
            )
        else:
            if gripper_velocity_limit <= 0.0 or gripper_saturation_effort <= 0.0:
                raise ValueError("Invalid DCMotor gripper parameters.")
            tau_computed = kps[gripper_mujoco_index] * (gripper_target - gripper_pos) - kds[gripper_mujoco_index] * gripper_vel
            vel_at_effort_lim = gripper_velocity_limit * (
                1.0 + gripper_torque_limit / gripper_saturation_effort
            )
            vel_for_limit = float(np.clip(gripper_vel, -vel_at_effort_lim, vel_at_effort_lim))
            tau_top = gripper_saturation_effort * (1.0 - vel_for_limit / gripper_velocity_limit)
            tau_bottom = gripper_saturation_effort * (-1.0 - vel_for_limit / gripper_velocity_limit)
            tau_max = min(tau_top, gripper_torque_limit)
            tau_min = max(tau_bottom, -gripper_torque_limit)
            gripper_tau = float(np.clip(tau_computed, tau_min, tau_max))
        wheel_ctrl = np.clip(wheel_cmd, -wheel_velocity_limits, wheel_velocity_limits).astype(np.float32)
        ctrl_source = np.concatenate(
            [leg_tau, arm_tau, wheel_ctrl, np.array([gripper_tau], dtype=np.float32)],
            dtype=np.float32,
        )
        d.ctrl[:] = 0.0
        for ctrl_i, src_i in enumerate(ctrl_src_indices_or_none):
            if src_i is not None:
                d.ctrl[ctrl_i] = ctrl_source[src_i]
    def spawn_puck_on_current_disk():
        center_w, x_w, y_w, z_w, q_w_d = get_disk_frame_world()
        xy = rng.uniform(
            -puck_spawn_xy_randomization,
            puck_spawn_xy_randomization,
            size=2,
        ).astype(np.float32)
        normal_offset = 0.5 * disk_thickness + 0.5 * puck_height + puck_spawn_clearance
        puck_pos_w = center_w + xy[0] * x_w + xy[1] * y_w + normal_offset * z_w
        d.qpos[puck_qpos_adr : puck_qpos_adr + 3] = puck_pos_w
        d.qpos[puck_qpos_adr + 3 : puck_qpos_adr + 7] = q_w_d
        d.qvel[puck_dof_adr : puck_dof_adr + 6] = 0.0
    def reset_episode(sim_time: float, reason: str = "start"):
        nonlocal last_ll_action, leg_target, arm_target, wheel_cmd, gripper_target
        nonlocal raw_hl_action, effective_hl_action, executed_hl_action
        nonlocal base_command, kp0_cmd, ee_yaw_cmd, ee_pitch_cmd, ee_roll_cmd, ee_cmd_plb_current
        nonlocal puck_no_contact_steps, puck_drop, base_contact
        nonlocal episode_start_time, episode_index, episode_hl_steps
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        d.qpos[0:3] = root_pos_reset
        d.qpos[3:7] = root_quat_reset
        d.qpos[7 : 7 + len(mujoco_joint_names)] = default_joint_pos
        d.qpos[7 + gripper_mujoco_index] = gripper_hold_pos
        # Safe temporary puck pose before forward kinematics.
        d.qpos[puck_qpos_adr : puck_qpos_adr + 3] = np.array([0.0, 0.0, 2.0])
        d.qpos[puck_qpos_adr + 3 : puck_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        d.qvel[puck_dof_adr : puck_dof_adr + 6] = 0.0
        reset_command_references()
        base_command = base_velocity_ref.copy()
        kp0_cmd = ee_kp0_ref.copy()
        ee_yaw_cmd = ee_yaw_ref
        ee_pitch_cmd = ee_pitch_ref
        ee_roll_cmd = ee_roll_ref
        ee_cmd_plb_current = build_keypoints_from_kp0_yaw_pitch_roll_plb(
            kp0_cmd, ee_yaw_cmd, ee_pitch_cmd, ee_roll_cmd, ee_kp_dx, ee_kp_dz
        )
        raw_hl_action = np.zeros(hl_action_dim, dtype=np.float32)
        effective_hl_action = np.zeros(hl_action_dim, dtype=np.float32)
        executed_hl_action = np.zeros(hl_action_dim, dtype=np.float32)
        last_ll_action = np.zeros(ll_action_dim, dtype=np.float32)
        leg_target = default_leg_pos.copy()
        arm_target = default_arm_pos.copy()
        wheel_cmd = np.zeros(4, dtype=np.float32)
        gripper_target = gripper_hold_pos
        mujoco.mj_forward(m, d)

        # Replace the nominal locked-FK HIGH with the ACTUAL reset EE kp0 in PLB.
        # This becomes the temporary hold reference during the 0.5-s settle.
        # At settle completion it is measured again under real dynamics.
        set_episode_high_from_measured_ee(
            update_current_reference=True
        )

        spawn_puck_on_current_disk()
        mujoco.mj_forward(m, d)
        uan_history.fill(0.0)
        last_uan_action.fill(0.0)
        last_uan_delta_tau.fill(0.0)
        last_uan_tau_pd.fill(0.0)
        last_uan_tau_nom.fill(0.0)
        puck_no_contact_steps = 0
        puck_drop = False
        base_contact = False
        ll_frame0 = build_ll_obs_frame()
        hl_frame0 = build_hl_obs_frame()
        fill_histories(ll_histories, ll_frame0, ll_feature_dims, ll_history_length)
        fill_histories(hl_histories, hl_frame0, hl_feature_dims, hl_history_length)
        episode_start_time = sim_time
        episode_hl_steps = 0
        episode_index += 1
        puck_d = get_puck_pos_disk()
        print(
            f"[RESET #{episode_index}] reason={reason} puck_mass={puck_mass:.3f}kg "
            f"HIGH_DYN=({episode_high_kp0[0]:+.4f},"
            f"{episode_high_kp0[1]:+.4f},{episode_high_kp0[2]:+.4f}) "
            f"puck_D=({puck_d[0]:+.4f},{puck_d[1]:+.4f},{puck_d[2]:+.4f})"
        )
    def infer_first_hl_and_ll_after_reset():
        # Reset history already contains the valid reset observation.  Infer the
        # first HL action from it, mask the residual because settle is active,
        # then infer the first 50-Hz WBC action before any physics step.
        hl_obs_stack0 = flatten_feature_major(hl_histories)
        action = high_sess.run(
            [high_output_name], {high_input_name: hl_obs_stack0[None, :]}
        )[0][0].astype(np.float32)
        decode_hl_action(action)
        run_low_level_policy()
    # Optional CSV diagnostics.
    debug_csv_file = None
    debug_csv_writer = None
    debug_ll_sample_count = 0
    if debug_tracking_logging:
        debug_csv_file = open(debug_tracking_csv_path, "w", newline="")
        debug_csv_writer = csv.writer(debug_csv_file)
        debug_csv_writer.writerow(
            [
                "sim_time_s", "episode", "settle_steps_remaining", "ee_pose_index",
                "puck_mass_kg", "puck_x_D", "puck_y_D", "puck_z_D", "puck_rho_D",
                *[f"base_ref_{x}" for x in ("vx", "vy", "wz")],
                *[f"base_cmd_{x}" for x in ("vx", "vy", "wz")],
                *[f"hl_raw_{i}" for i in range(hl_action_dim)],
                *[f"hl_eff_{i}" for i in range(hl_action_dim)],
                *[f"hl_exec_{i}" for i in range(hl_action_dim)],
                *[f"kp0_ref_{x}" for x in ("x", "y", "z")],
                *[f"kp0_cmd_{x}" for x in ("x", "y", "z")],
                "yaw_ref", "pitch_ref", "roll_ref", "yaw_cmd", "pitch_cmd", "roll_cmd",
                "disk_puck_contact", "disk_puck_force_N", "puck_no_contact_steps",
                *[f"uan_a{j+1}" for j in range(6)],
                *[f"uan_dtau{j+1}" for j in range(6)],
            ]
        )
        debug_csv_file.flush()
    def write_tracking_debug_row(sim_time_now: float):
        nonlocal debug_ll_sample_count
        if not debug_tracking_logging or debug_csv_writer is None:
            return
        debug_ll_sample_count += 1
        if debug_ll_sample_count % debug_tracking_log_every_ll_steps != 0:
            return
        puck_d = get_puck_pos_disk()
        contact, force = puck_disk_contact_now()
        debug_csv_writer.writerow(
            [
                float(sim_time_now), int(episode_index), int(settle_steps_remaining), int(current_ee_pose_index),
                float(puck_mass), *puck_d.tolist(), float(np.linalg.norm(puck_d[:2])),
                *base_velocity_ref.tolist(), *base_command.tolist(),
                *raw_hl_action.tolist(), *effective_hl_action.tolist(), *executed_hl_action.tolist(),
                *ee_kp0_ref.tolist(), *kp0_cmd.tolist(),
                float(ee_yaw_ref), float(ee_pitch_ref), float(ee_roll_ref),
                float(ee_yaw_cmd), float(ee_pitch_cmd), float(ee_roll_cmd),
                int(contact), float(force), int(puck_no_contact_steps),
                *last_uan_action.tolist(), *last_uan_delta_tau.tolist(),
            ]
        )
        debug_csv_file.flush()
    def update_custom_visualization(viewer):
        viewer.user_scn.ngeom = 0
        def draw_kp_axes(kps_plb, sphere_rgba, x_rgba, z_rgba):
            kps_w = ee_kp_plb_to_world(kps_plb)
            kp0_w, kp1_w, kp2_w = kps_w
            xdir = kp1_w - kp0_w
            zdir = kp2_w - kp0_w
            xn = float(np.linalg.norm(xdir))
            zn = float(np.linalg.norm(zdir))
            add_sphere_to_scene(viewer.user_scn, kp0_w, vis_sphere_size, sphere_rgba)
            if xn > 1.0e-8:
                add_capsule_to_scene(
                    viewer.user_scn,
                    kp0_w,
                    kp0_w + vis_axis_len * xdir / xn,
                    vis_axis_radius,
                    x_rgba,
                )
            if zn > 1.0e-8:
                add_capsule_to_scene(
                    viewer.user_scn,
                    kp0_w,
                    kp0_w + vis_axis_len * zdir / zn,
                    vis_axis_radius,
                    z_rgba,
                )
        if vis_task_ee_reference:
            ref_kps = build_keypoints_from_kp0_yaw_pitch_roll_plb(
                ee_kp0_ref,
                ee_yaw_ref,
                ee_pitch_ref,
                ee_roll_ref,
                ee_kp_dx,
                ee_kp_dz,
            )
            draw_kp_axes(
                ref_kps,
                [1.0, 0.0, 1.0, 0.9],
                [1.0, 0.0, 1.0, 0.9],
                [0.7, 0.0, 1.0, 0.9],
            )
        if vis_wbc_ee_command:
            draw_kp_axes(
                ee_cmd_plb_current,
                [1.0, 0.2, 0.2, 0.9],
                [1.0, 0.0, 0.0, 0.9],
                [0.0, 0.0, 1.0, 0.9],
            )
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    sim_time = 0.0
    sim_counter = 0
    ll_blocks_completed_in_hl_interval = 0
    reset_episode(sim_time, reason="start")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = d.qpos[:3]
        infer_first_hl_and_ll_after_reset()
        write_tracking_debug_row(sim_time)
        while viewer.is_running() and sim_time < simulation_duration:
            step_start = time.time()
            apply_low_level_actuation()
            mujoco.mj_step(m, d)
            sim_time += simulation_dt
            sim_counter += 1
            viewer.cam.lookat[:] = d.qpos[:3]
            if sim_counter % ll_control_decimation == 0:
                ll_blocks_completed_in_hl_interval += 1
                if ll_blocks_completed_in_hl_interval >= ll_steps_per_hl_step:
                    # One complete 10-Hz transition has ended. Contact/terminal
                    # state is evaluated BEFORE references advance, as in training.
                    puck_contact, puck_force = puck_disk_contact_now()
                    # Training only counts no-contact steps in ACTIVE transport;
                    # the 0.5-s settle interval clears the counter instead.
                    if settle_steps_remaining > 0 or puck_contact:
                        puck_no_contact_steps = 0
                    else:
                        puck_no_contact_steps += 1
                    puck_drop = puck_no_contact_steps >= puck_drop_no_contact_steps
                    base_contact, base_force = base_contact_now()
                    episode_hl_steps += 1
                    timeout = episode_hl_steps >= max_episode_hl_steps - 1
                    reset_reason = None
                    if terminate_on_base_contact and base_contact:
                        reset_reason = f"base_contact({base_force:.2f}N)"
                    elif terminate_on_puck_drop and puck_drop:
                        reset_reason = "puck_drop"
                    elif timeout:
                        reset_reason = "time_out"
                    if auto_reset and reset_reason is not None:
                        reset_episode(sim_time, reason=reset_reason)
                        infer_first_hl_and_ll_after_reset()
                        ll_blocks_completed_in_hl_interval = 0
                        write_tracking_debug_row(sim_time)
                    else:
                        # Training advances references AFTER the completed
                        # transition and BEFORE constructing the next actor obs.
                        # This includes one 10-Hz minimum-jerk EE interpolation step;
                        # the previous-action channel already holds just-executed action.
                        advance_command_references_after_transition()
                        run_high_level_policy()
                        run_low_level_policy()
                        ll_blocks_completed_in_hl_interval = 0
                        write_tracking_debug_row(sim_time)
                else:
                    run_low_level_policy()
                    write_tracking_debug_row(sim_time)
            update_custom_visualization(viewer)
            if print_interval_sim_steps > 0 and sim_counter % print_interval_sim_steps == 0:
                puck_d = get_puck_pos_disk()
                puck_contact_dbg, puck_force_dbg = puck_disk_contact_now()
                actual_kps = compute_actual_ee_keypoints_plb()
                kp0_err = actual_kps[:3] - ee_cmd_plb_current[:3]
                print(
                    f"[{sim_counter:7d}] t={sim_time:7.2f}s ep={episode_index:03d} "
                    f"settle={settle_steps_remaining:02d} ee_idx={current_ee_pose_index} "
                    f"puck_D=({puck_d[0]:+.3f},{puck_d[1]:+.3f},{puck_d[2]:+.3f}) "
                    f"rho={np.linalg.norm(puck_d[:2]):.3f} "
                    f"contact={int(puck_contact_dbg)}({puck_force_dbg:.2f}N) "
                    f"no_contact={puck_no_contact_steps}"
                )
                print(
                    "            base_ref="
                    + np.array2string(base_velocity_ref, precision=3, suppress_small=True)
                    + " base_cmd="
                    + np.array2string(base_command, precision=3, suppress_small=True)
                    + " | HL_raw="
                    + np.array2string(raw_hl_action, precision=2, suppress_small=True)
                )
                print(
                    "            HL_eff="
                    + np.array2string(effective_hl_action, precision=2, suppress_small=True)
                    + " HL_exec="
                    + np.array2string(executed_hl_action, precision=2, suppress_small=True)
                    + f" | kp0_err={np.linalg.norm(kp0_err):.3f}m"
                )
                print(
                    f"            UAN_sat={np.mean(np.abs(last_uan_action) > uan_action_saturation_threshold):.3f} "
                    f"UAN_nomclip={np.mean(np.abs(last_uan_tau_pd) > uan_nominal_torque_limits):.3f} "
                    f"UAN_dTau={np.mean(np.abs(last_uan_delta_tau)):.2f}Nm"
                )
            viewer.sync()
            remaining = simulation_dt - (time.time() - step_start)
            if remaining > 0.0:
                time.sleep(remaining)
    if debug_csv_file is not None:
        debug_csv_file.close()
        print(f"[DEBUG] Tracking CSV written to: {debug_tracking_csv_path}")
if __name__ == "__main__":
    main()