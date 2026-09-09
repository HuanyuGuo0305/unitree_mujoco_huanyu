"""
Hierarchical MuJoCo sim2sim deployment for B2WZ1 retrieval with frozen WBC + UAN.

Control hierarchy
-----------------
MuJoCo / UAN actuator: 250 Hz
Frozen low-level WBC ONNX: 50 Hz
High-level retrieval ONNX: 10 Hz

The high-level policy matches the ABS Isaac Lab training interface:
56-D/frame actor observation, 3-frame feature-major history, and 9-D action.
The only behavioral change relative to the previous UAN sim2sim is the 5-D
high-level arm/EE action semantics: normalized actions map directly to absolute
PLB XYZ/yaw/pitch command ranges rather than being deltas from measured EE pose.

The low-level WBC outputs:
  - 12 leg position targets
  - 6 Z1 arm position targets
  - 4 wheel velocity targets

For the Z1 arm, MuJoCo does NOT use the legacy ideal-PD/DCMotor arm path.
Instead, at every 4-ms physics step it reproduces the frozen UAN actuator:

    tau_pd  = Kp * (q_des - q) - Kd * qd
    tau_nom = clip(tau_pd, +/- nominal_limit)
    delta   = UAN(history) * action_scale * Kp
    tau_arm = clip(tau_nom + delta, +/- final_limit)

The 50-Hz WBC q_des is held for five 250-Hz UAN/physics steps.  UAN history is
6 joints x 25 samples x (position_error, scaled_velocity), joint-major,
newest sample last, and is zeroed on every episode reset.

Intentionally NOT implemented:
  - startup policy blend-in
  - arm joint-target rate limiter
  - observation noise / mocap noise / latency

Run from unitree_mujoco/simulate_python:

    python3 deploy_mujoco/b2wz1_hl_retrieval_abs_uan.py configs/b2wz1_hl_retrieval_abs_uan.yaml
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
    quat_apply_inverse_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_from_rotmat_wxyz,
    quat_from_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_rotate_inverse_numpy,
    quat_unique_wxyz,
    euler_xyz_from_quat_wxyz,
)
from utilities.mujoco_helper import get_sensor_slice, make_arrow_mat  # noqa: E402


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def quat_from_euler_xyz_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Standard XYZ roll/pitch/yaw to wxyz quaternion."""
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)

    q = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )
    return quat_unique_wxyz(quat_normalize_wxyz(q))


def build_keypoints_from_kp0_yaw_pitch_plb(
    kp0: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    kp_dx: float,
    kp_dz: float,
    pitch_to_euler_sign: float = 1.0,
) -> np.ndarray:
    """Build [kp0,kp1,kp2] in PLB from the commanded EE pose.

    ``pitch`` is the high-level policy/training pitch coordinate.  The Z1
    gripper convention used by this deployment treats negative pitch as
    tilting the gripper local +X (finger-forward) axis downward.  Standard
    XYZ Euler ``Ry(pitch)`` has the opposite sign for that geometric effect,
    so ``pitch_to_euler_sign=-1`` converts the policy coordinate to the
    geometric Euler angle before constructing the quaternion.
    """
    kp0 = np.asarray(kp0, dtype=np.float32).reshape(3,)
    geometric_pitch = float(pitch_to_euler_sign) * float(pitch)
    q_plb = quat_from_euler_xyz_wxyz(
        float(roll), geometric_pitch, float(yaw)
    )

    off_x = np.array([kp_dx, 0.0, 0.0], dtype=np.float32)
    off_z = np.array([0.0, 0.0, kp_dz], dtype=np.float32)

    kp1 = kp0 + quat_apply_wxyz(q_plb, off_x)
    kp2 = kp0 + quat_apply_wxyz(q_plb, off_z)
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

    assert abs(1.0 / ll_control_dt - 50.0) < 1.0e-5
    assert abs(1.0 / hl_control_dt - 10.0) < 1.0e-5
    assert ll_obs_dim_per_step == 80
    assert ll_obs_dim == 400
    assert ll_action_dim == 22
    assert hl_obs_dim_per_step == 56
    assert hl_obs_dim == 168
    assert hl_action_dim == 9

    root_pos_reset = np.asarray(cfg["root_pos"], dtype=np.float32)
    root_quat_reset = np.asarray(cfg["root_quat_wxyz"], dtype=np.float32)

    default_joint_pos = np.asarray(cfg["default_joint_pos"], dtype=np.float32)
    kps = np.asarray(cfg["kps"], dtype=np.float32)
    kds = np.asarray(cfg["kds"], dtype=np.float32)

    leg_action_scale = float(cfg["leg_action_scale"])
    arm_action_scale = np.asarray(cfg["arm_action_scale"], dtype=np.float32)
    wheel_action_scale = float(cfg["wheel_action_scale"])

    leg_torque_limits = np.asarray(cfg["leg_torque_limits"], dtype=np.float32)

    # Frozen UAN actuator configuration. These values must match the
    # UNITREE_B2WZ1_UAN_CFG / UANActuator used during WBC and HL training.
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

    gripper_torque_limit = float(cfg["gripper_torque_limit"])
    # Match IsaacLab DCMotorCfg used by jointGripper.
    # In training:
    #   effort_limit      = 30.0
    #   saturation_effort = 30.0
    #   velocity_limit    = 2.0 rad/s
    #
    # Keep backward compatibility with the existing YAML: if the two new keys are
    # absent, use the exact training values / existing effort limit.
    gripper_saturation_effort = float(
        cfg.get("gripper_saturation_effort", gripper_torque_limit)
    )
    gripper_velocity_limit = float(cfg.get("gripper_velocity_limit", 2.0))
    wheel_velocity_limits = np.asarray(cfg["wheel_velocity_limits"], dtype=np.float32)

    base_cmd_scale = np.asarray(cfg["base_cmd_scale"], dtype=np.float32)
    kp0_x_range = np.asarray(cfg["kp0_x_range"], dtype=np.float32)
    kp0_y_range = np.asarray(cfg["kp0_y_range"], dtype=np.float32)
    kp0_z_range = np.asarray(cfg["kp0_z_range"], dtype=np.float32)
    ee_yaw_range = np.asarray(cfg["ee_yaw_range"], dtype=np.float32)
    ee_pitch_range = np.asarray(cfg["ee_pitch_range"], dtype=np.float32)
    # Convert the policy/training pitch coordinate to the geometric XYZ-Euler
    # pitch used by the local quaternion helper.  For the Z1 gripper frame,
    # negative policy pitch is intended to tilt local +X downward.
    ee_pitch_to_euler_sign = float(cfg.get("ee_pitch_to_euler_sign", -1.0))
    if ee_pitch_to_euler_sign not in (-1.0, 1.0):
        raise ValueError(
            "ee_pitch_to_euler_sign must be exactly -1.0 or +1.0, "
            f"got {ee_pitch_to_euler_sign}."
        )

    neutral_kp0 = np.asarray(cfg["neutral_kp0"], dtype=np.float32)
    neutral_ee_yaw = float(cfg["neutral_ee_yaw"])
    neutral_ee_pitch = float(cfg["neutral_ee_pitch"])
    fixed_ee_roll = float(cfg["fixed_ee_roll"])
    ee_kp_dx = float(cfg["ee_kp_dx"])
    ee_kp_dz = float(cfg["ee_kp_dz"])
    ground_z = float(cfg["ground_z"])

    def action_to_range(x: float, limits: np.ndarray) -> float:
        """Map normalized action x in [-1,1] affinely to [low, high]."""
        low, high = float(limits[0]), float(limits[1])
        if high <= low:
            raise ValueError(f"Invalid absolute action range [{low}, {high}].")
        return low + 0.5 * (float(x) + 1.0) * (high - low)

    def range_to_action(value: float, limits: np.ndarray) -> float:
        """Inverse affine map, used to encode the neutral reset EE command."""
        low, high = float(limits[0]), float(limits[1])
        if high <= low:
            raise ValueError(f"Invalid absolute action range [{low}, {high}].")
        return 2.0 * (float(value) - low) / (high - low) - 1.0

    # Training reset semantics for the ABS policy: previous_hl_action[3:8]
    # must encode the neutral EE command in normalized ABS coordinates, not zeros.
    neutral_arm_action = np.array(
        [
            range_to_action(neutral_kp0[0], kp0_x_range),
            range_to_action(neutral_kp0[1], kp0_y_range),
            range_to_action(neutral_kp0[2], kp0_z_range),
            range_to_action(neutral_ee_yaw, ee_yaw_range),
            range_to_action(neutral_ee_pitch, ee_pitch_range),
        ],
        dtype=np.float32,
    )
    if np.any(neutral_arm_action < -1.0 - 1.0e-6) or np.any(neutral_arm_action > 1.0 + 1.0e-6):
        raise ValueError(
            "Neutral EE command lies outside configured ABS command ranges: "
            f"normalized={neutral_arm_action}"
        )

    gripper_open_pos = float(cfg["gripper_open_pos"])
    gripper_close_pos = float(cfg["gripper_close_pos"])
    gripper_binary_threshold = float(cfg["gripper_binary_threshold"])
    stage2_force_gripper_close_enabled = bool(
        cfg.get("stage2_force_gripper_close_enabled", True)
    )

    gripper_center_offset_local = np.asarray(
        cfg["gripper_center_offset_local"], dtype=np.float32
    )
    grasp_proxy_error_threshold = float(cfg["grasp_proxy_error_threshold"])
    gripper_not_fully_closed_angle_threshold = float(
        cfg["gripper_not_fully_closed_angle_threshold"]
    )
    gripper_angle_hold_threshold = float(cfg["gripper_angle_hold_threshold"])
    grasp_proxy_enter_steps = int(cfg["grasp_proxy_enter_steps"])
    grasp_proxy_exit_steps = int(cfg["grasp_proxy_exit_steps"])

    object_sampling_center_x = float(cfg["object_sampling_center_x"])
    object_sampling_center_y = float(cfg["object_sampling_center_y"])
    object_sampling_r_range = np.asarray(cfg["object_sampling_r_range"], dtype=np.float32)
    object_sampling_theta_max = float(cfg["object_sampling_theta_max"])
    object_half_extent = float(cfg["object_half_extent"])
    object_spawn_z_epsilon = float(cfg["object_spawn_z_epsilon"])

    # Retrieval-target reset distribution.
    # Legacy mode preserves the original fixed-Z / 0.25-m-disk behavior.
    # Enlarged mode matches the new training distribution.
    enlarged_retrieval_target_sampling_range = bool(
        cfg.get("enlarged_retrieval_target_sampling_range", False)
    )

    retrieval_target_radius = float(cfg["retrieval_target_radius"])
    retrieval_target_z_w = float(cfg["retrieval_target_z_w"])

    enlarged_retrieval_target_radius = float(
        cfg.get("enlarged_retrieval_target_radius", 0.50)
    )
    enlarged_retrieval_target_z_range_w = np.asarray(
        cfg.get("enlarged_retrieval_target_z_range_w", [0.30, 0.80]),
        dtype=np.float32,
    )
    if enlarged_retrieval_target_z_range_w.shape != (2,):
        raise ValueError(
            "enlarged_retrieval_target_z_range_w must contain exactly [z_min, z_max]."
        )
    if enlarged_retrieval_target_radius < 0.0:
        raise ValueError("enlarged_retrieval_target_radius must be >= 0.")
    if enlarged_retrieval_target_z_range_w[1] < enlarged_retrieval_target_z_range_w[0]:
        raise ValueError(
            "enlarged_retrieval_target_z_range_w must satisfy z_max >= z_min."
        )

    episode_length_s = float(cfg["episode_length_s"])
    auto_reset = bool(cfg.get("auto_reset", True))
    rng = np.random.default_rng(int(cfg.get("random_seed", 0)))

    vis_retrieval_target = bool(cfg.get("vis_retrieval_target", True))
    vis_retrieval_target_radius = float(cfg.get("vis_retrieval_target_radius", 0.05))
    vis_ee_target = bool(cfg.get("vis_ee_target", True))
    vis_ee_target_axis_len = float(cfg.get("vis_ee_target_axis_len", 0.20))
    vis_ee_target_axis_radius = float(cfg.get("vis_ee_target_axis_radius", 0.01))
    vis_ee_target_sphere_size = float(cfg.get("vis_ee_target_sphere_size", 0.03))

    # P2 grasp-height / close-timing diagnostics. These are visualization-only and
    # do not alter observations, actions, targets, or physics.
    vis_grasp_debug = bool(cfg.get("vis_grasp_debug", True))
    vis_gripper_center_radius = float(cfg.get("vis_gripper_center_radius", 0.018))
    vis_cube_center_radius = float(cfg.get("vis_cube_center_radius", 0.015))

    print_interval_sim_steps = int(cfg.get("print_interval_sim_steps", 200))

    # Tracking diagnostics for HL -> WBC -> arm debugging. Logging is observation-only:
    # it does not modify policy inputs, actions, commands, or physics.
    debug_tracking_logging = bool(cfg.get("debug_tracking_logging", True))
    debug_tracking_log_every_ll_steps = int(
        cfg.get("debug_tracking_log_every_ll_steps", 1)
    )
    if debug_tracking_log_every_ll_steps <= 0:
        raise ValueError("debug_tracking_log_every_ll_steps must be >= 1.")

    debug_tracking_csv_path = None
    if debug_tracking_logging:
        debug_tracking_csv_path = resolve_project_path(
            str(
                cfg.get(
                    "debug_tracking_csv_path",
                    "logs/b2wz1_hl_retrieval_abs_uan_tracking_debug.csv",
                )
            )
        )
        os.makedirs(os.path.dirname(debug_tracking_csv_path), exist_ok=True)

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

    if np.any(uan_nominal_torque_limits <= 0.0):
        raise ValueError("uan_nominal_torque_limits must all be > 0.")
    if np.any(uan_final_torque_limits < uan_nominal_torque_limits):
        raise ValueError(
            "uan_final_torque_limits must be >= uan_nominal_torque_limits."
        )

    # Exact hierarchy used by the UAN WBC / HL training:
    #   250-Hz physics+UAN -> 50-Hz frozen WBC -> 10-Hz high level.
    if abs(simulation_dt - 0.004) > 1.0e-9 or ll_control_decimation != 5:
        raise ValueError(
            "UAN hierarchy timing mismatch: expected simulation_dt=0.004 and "
            f"ll_control_decimation=5, got {simulation_dt} and "
            f"{ll_control_decimation}."
        )

    # Fail early on bad paths.
    for key, path in (
        ("low_policy_path", low_policy_path),
        ("high_policy_path", high_policy_path),
        ("uan_model_path", uan_model_path),
        ("xml_path", xml_path),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{key} does not exist: {path}")

    # ONNX sessions.
    low_sess = ort.InferenceSession(low_policy_path, providers=["CPUExecutionProvider"])
    high_sess = ort.InferenceSession(high_policy_path, providers=["CPUExecutionProvider"])

    low_input_name = low_sess.get_inputs()[0].name
    low_output_name = low_sess.get_outputs()[0].name
    high_input_name = high_sess.get_inputs()[0].name
    high_output_name = high_sess.get_outputs()[0].name

    # Exported deterministic UAN wrapper: includes observation normalization
    # and Beta-mean actor inference.
    if uan_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"UAN device {uan_device!r} requested but CUDA is unavailable."
        )
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
            raise RuntimeError(
                f"Expected UAN output shape (1,6), got {tuple(dummy_out.shape)}"
            )
        if not torch.isfinite(dummy_out).all():
            raise RuntimeError("UAN returned non-finite zero-history output.")

    print("=" * 88)
    print("B2WZ1 Retrieval ABS - Hierarchical WBC + UAN MuJoCo sim2sim")
    print(f"XML:               {xml_path}")
    print(f"Low policy:        {low_policy_path}")
    print(f"High policy:       {high_policy_path}")
    print(f"UAN model:         {uan_model_path}")
    print(f"Physics / UAN:     {1.0 / simulation_dt:.1f} Hz")
    print(f"Low-level WBC:     {1.0 / ll_control_dt:.1f} Hz, obs={ll_obs_dim}, act={ll_action_dim}")
    print(f"High-level:        {1.0 / hl_control_dt:.1f} Hz, obs={hl_obs_dim}, act={hl_action_dim}")
    print("Low ONNX:          ", low_sess.get_inputs()[0].shape, "->", low_sess.get_outputs()[0].shape)
    print("High ONNX:         ", high_sess.get_inputs()[0].shape, "->", high_sess.get_outputs()[0].shape)
    print("UAN TorchScript:   ", tuple(dummy_out.shape), "zero=", np.array2string(dummy_out.detach().cpu().numpy()[0], precision=3))
    print(f"UAN history:       {uan_history_length} x {simulation_dt * 1000.0:.1f} ms = {uan_history_length * simulation_dt * 1000.0:.1f} ms")
    print(f"UAN Kp:            {uan_kp}")
    print(f"UAN Kd:            {uan_kd}")
    print(f"UAN nominal cap:   {uan_nominal_torque_limits}")
    print(f"UAN final cap:     {uan_final_torque_limits}")
    print(f"Force close proxy: {stage2_force_gripper_close_enabled}")
    print("HL arm semantics:  ABS normalized action -> PLB XYZ/yaw/pitch ranges")
    print(
        "EE pitch mapping:   policy pitch * "
        f"{ee_pitch_to_euler_sign:+.0f} -> geometric XYZ-Euler pitch "
        "(Z1 local +X toward fingertips)"
    )
    print(f"ABS neutral action:{neutral_arm_action}")
    if enlarged_retrieval_target_sampling_range:
        print(
            "Retrieval target:  enlarged "
            f"XY radius={enlarged_retrieval_target_radius:.3f} m, "
            f"Z=[{float(enlarged_retrieval_target_z_range_w[0]):.3f}, "
            f"{float(enlarged_retrieval_target_z_range_w[1]):.3f}] m"
        )
    else:
        print(
            "Retrieval target:  legacy "
            f"XY radius={retrieval_target_radius:.3f} m, "
            f"Z={retrieval_target_z_w:.3f} m"
        )
    print(
        "Gripper DCMotor:    "
        f"effort={gripper_torque_limit:.3f} Nm, "
        f"stall={gripper_saturation_effort:.3f} Nm, "
        f"no-load vel={gripper_velocity_limit:.3f} rad/s"
    )
    if debug_tracking_logging:
        print(
            "Tracking debug:     50-Hz CSV every "
            f"{debug_tracking_log_every_ll_steps} LL step(s) -> "
            f"{debug_tracking_csv_path}"
        )
    print("=" * 88)

    # Fail loudly on static ONNX dimensions if available.
    low_in_shape = low_sess.get_inputs()[0].shape
    low_out_shape = low_sess.get_outputs()[0].shape
    high_in_shape = high_sess.get_inputs()[0].shape
    high_out_shape = high_sess.get_outputs()[0].shape
    if isinstance(low_in_shape[-1], int):
        assert low_in_shape[-1] == ll_obs_dim, low_in_shape
    if isinstance(low_out_shape[-1], int):
        assert low_out_shape[-1] == ll_action_dim, low_out_shape
    if isinstance(high_in_shape[-1], int):
        assert high_in_shape[-1] == hl_obs_dim, high_in_shape
    if isinstance(high_out_shape[-1], int):
        assert high_out_shape[-1] == hl_action_dim, high_out_shape

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    ee_body_name = str(cfg.get("ee_body", "gripperStator"))
    cube_body_name = str(cfg.get("cube_body", "retrieval_cube"))
    cube_freejoint_name = str(cfg.get("cube_freejoint", "retrieval_cube_freejoint"))

    ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cube_body_name)
    cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, cube_freejoint_name)
    if ee_bid < 0:
        raise ValueError(f"Body not found: {ee_body_name}")
    if cube_bid < 0:
        raise ValueError(f"Cube body not found: {cube_body_name}")
    if cube_jid < 0:
        raise ValueError(f"Cube freejoint not found: {cube_freejoint_name}")

    cube_qpos_adr = int(m.jnt_qposadr[cube_jid])
    cube_dof_adr = int(m.jnt_dofadr[cube_jid])

    # This is the validated low-level MuJoCo model/joint ordering.
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

    policy_joint_pos_mujoco_indices = [
        mujoco_joint_names.index(n) for n in policy_joint_pos_names
    ]
    policy_joint_vel_mujoco_indices = [
        mujoco_joint_names.index(n) for n in policy_joint_vel_names
    ]
    leg_mujoco_indices = [mujoco_joint_names.index(n) for n in leg_joint_names]
    arm_mujoco_indices = [mujoco_joint_names.index(n) for n in arm_joint_names]
    wheel_mujoco_indices = [mujoco_joint_names.index(n) for n in wheel_joint_names]
    gripper_mujoco_index = mujoco_joint_names.index("jointGripper")

    leg_action_indices = np.arange(0, 12)
    arm_action_indices = np.arange(12, 18)
    wheel_action_indices = np.arange(18, 22)

    control_source_joint_names = (
        leg_joint_names + arm_joint_names + wheel_joint_names + ["jointGripper"]
    )
    ctrl_source_index_by_name = {
        name: i for i, name in enumerate(control_source_joint_names)
    }
    ctrl_src_indices_or_none = [
        ctrl_source_index_by_name[name] if name in ctrl_source_index_by_name else None
        for name in ctrl_joint_names
    ]

    default_joint_pos_policy_pos = default_joint_pos[policy_joint_pos_mujoco_indices]
    default_leg_pos = default_joint_pos[leg_mujoco_indices]
    default_arm_pos = default_joint_pos[arm_mujoco_indices]
    default_gripper_pos = float(default_joint_pos[gripper_mujoco_index])

    # The retrieval XML may still contain the legacy +/-30/60 Nm Z1 limits.
    # The explicit UAN law already owns the final torque clamp, so synchronize
    # any narrower MuJoCo-side limits to the UAN final envelope.
    print("MuJoCo arm torque-limit audit:")
    for j, joint_name in enumerate(arm_joint_names):
        ctrl_i = ctrl_joint_names.index(joint_name)
        final_lim = float(uan_final_torque_limits[j])

        if hasattr(m, "actuator_ctrllimited") and bool(m.actuator_ctrllimited[ctrl_i]):
            old_range = m.actuator_ctrlrange[ctrl_i].copy()
            if old_range[0] > -final_lim + 1e-6 or old_range[1] < final_lim - 1e-6:
                if not sync_mujoco_arm_limits_to_uan:
                    raise RuntimeError(
                        f"MuJoCo actuator ctrlrange for {joint_name} is "
                        f"[{old_range[0]}, {old_range[1]}], narrower than "
                        f"UAN final +/-{final_lim} Nm."
                    )
                m.actuator_ctrlrange[ctrl_i, :] = (-final_lim, final_lim)
                print(
                    f"  {joint_name}: ctrlrange "
                    f"[{old_range[0]:.1f}, {old_range[1]:.1f}] -> "
                    f"[-{final_lim:.1f}, +{final_lim:.1f}]"
                )

        if hasattr(m, "actuator_forcelimited") and bool(m.actuator_forcelimited[ctrl_i]):
            old_range = m.actuator_forcerange[ctrl_i].copy()
            if old_range[0] > -final_lim + 1e-6 or old_range[1] < final_lim - 1e-6:
                if not sync_mujoco_arm_limits_to_uan:
                    raise RuntimeError(
                        f"MuJoCo actuator forcerange for {joint_name} is "
                        f"[{old_range[0]}, {old_range[1]}], narrower than "
                        f"UAN final +/-{final_lim} Nm."
                    )
                m.actuator_forcerange[ctrl_i, :] = (-final_lim, final_lim)
                print(
                    f"  {joint_name}: forcerange "
                    f"[{old_range[0]:.1f}, {old_range[1]:.1f}] -> "
                    f"[-{final_lim:.1f}, +{final_lim:.1f}]"
                )

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
                        f"MuJoCo joint actuator-force range for {joint_name} is "
                        f"[{old_range[0]}, {old_range[1]}], narrower than "
                        f"UAN final +/-{final_lim} Nm."
                    )
                m.jnt_actfrcrange[joint_id, :] = (-final_lim, final_lim)
                print(
                    f"  {joint_name}: joint actfrcrange "
                    f"[{old_range[0]:.1f}, {old_range[1]:.1f}] -> "
                    f"[-{final_lim:.1f}, +{final_lim:.1f}]"
                )

    print("  PASS: MuJoCo will not clip inside the UAN final torque envelope.")

    # Low-level state.
    base_command = np.zeros(3, dtype=np.float32)
    ee_cmd_plb_current = np.zeros(9, dtype=np.float32)
    last_ll_action = np.zeros(ll_action_dim, dtype=np.float32)

    leg_target = default_leg_pos.copy()
    arm_target = default_arm_pos.copy()
    wheel_cmd = np.zeros(4, dtype=np.float32)
    gripper_target = gripper_open_pos

    # High-level state (ABS EE action semantics).
    current_hl_action = np.zeros(hl_action_dim, dtype=np.float32)
    executed_gripper_cmd_norm = -1.0
    raw_gripper_action = -1.0

    grasp_confidence_proxy = False
    grasp_proxy_enter_count = 0
    grasp_proxy_exit_count = 0
    prev_gripper_joint_pos = gripper_open_pos

    retrieval_target_pos_w = np.zeros(3, dtype=np.float32)
    episode_start_time = 0.0
    episode_index = 0

    # History buffers: one deque per feature group, matching training feature-major flatten.
    ll_feature_dims = [3, 3, 3, 9, 12, 6, 12, 6, 4, 22]
    # High-level actor observation matches training with root_lin_vel removed.
    # Feature order per frame:
    #   root_ang_vel_b(3), projected_gravity_b(3), leg_pos_rel(12),
    #   arm_pos_rel(6), gripper_pos_rel(1), arm_joint_vel(6),
    #   object_center_pos_base(3), gripper_orientation_base(6),
    #   gripper_center_pos_base(3), retrieval_target_pos_base(3),
    #   previous_hl_action(9), grasp_confidence_proxy(1) = 56.
    hl_feature_dims = [3, 3, 12, 6, 1, 6, 3, 6, 3, 3, 9, 1]
    assert sum(ll_feature_dims) == ll_obs_dim_per_step
    assert sum(hl_feature_dims) == hl_obs_dim_per_step

    ll_histories = [deque(maxlen=ll_history_length) for _ in ll_feature_dims]
    hl_histories = [deque(maxlen=hl_history_length) for _ in hl_feature_dims]

    # UAN state: (joint, time, feature), joint-major, newest sample last.
    # Feature 0 = position error * pos_scale
    # Feature 1 = joint velocity * vel_scale
    uan_history = np.zeros((6, uan_history_length, 2), dtype=np.float32)
    last_uan_action = np.zeros(6, dtype=np.float32)
    last_uan_delta_tau = np.zeros(6, dtype=np.float32)
    last_uan_tau_pd = np.zeros(6, dtype=np.float32)
    last_uan_tau_nom = np.zeros(6, dtype=np.float32)

    def compute_uan_arm_torque(
        q_des: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> np.ndarray:
        """Run one 250-Hz frozen UAN actuator step."""
        q_des = np.asarray(q_des, dtype=np.float32).reshape(6,)
        q = np.asarray(q, dtype=np.float32).reshape(6,)
        qd = np.asarray(qd, dtype=np.float32).reshape(6,)

        pos_err = (q_des - q) * uan_pos_scale
        vel = qd * uan_vel_scale

        uan_history[:, :-1, :] = uan_history[:, 1:, :]
        uan_history[:, -1, 0] = pos_err
        uan_history[:, -1, 1] = vel

        obs_np = np.ascontiguousarray(
            uan_history.reshape(1, -1),
            dtype=np.float32,
        )
        obs_t = torch.from_numpy(obs_np).to(device=uan_device)

        with torch.inference_mode():
            out = uan_model(obs_t)
            if isinstance(out, (tuple, list)):
                out = out[0]

        action = out.detach().cpu().numpy().reshape(6,).astype(np.float32)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError(f"Non-finite UAN action: {action}")
        if np.max(np.abs(action)) > 1.001:
            raise RuntimeError(
                f"UAN action outside expected Beta-mean [-1,1]: {action}"
            )

        tau_pd = uan_kp * (q_des - q) - uan_kd * qd
        tau_nom = np.clip(
            tau_pd,
            -uan_nominal_torque_limits,
            uan_nominal_torque_limits,
        )
        delta_tau = action * uan_action_scale * uan_kp
        tau = np.clip(
            tau_nom + delta_tau,
            -uan_final_torque_limits,
            uan_final_torque_limits,
        ).astype(np.float32)

        last_uan_action[:] = action
        last_uan_delta_tau[:] = delta_tau
        last_uan_tau_pd[:] = tau_pd
        last_uan_tau_nom[:] = tau_nom
        return tau

    def split_features(frame: np.ndarray, dims: list[int]) -> list[np.ndarray]:
        out = []
        i = 0
        for dim in dims:
            out.append(frame[i:i + dim].copy())
            i += dim
        assert i == frame.shape[0]
        return out

    def fill_histories(histories, frame: np.ndarray, dims: list[int], length: int):
        features = split_features(frame, dims)
        for hist, feature in zip(histories, features):
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

    def compute_actual_ee_pose_plb():
        plb_pos_w, plb_quat_w = get_plb_pose_world()
        ee_pos_w = d.xpos[ee_bid].copy().astype(np.float32)
        ee_rot_w = d.xmat[ee_bid].reshape(3, 3).copy().astype(np.float32)
        ee_quat_w = quat_from_rotmat_wxyz(ee_rot_w)

        ee_pos_plb = quat_apply_inverse_wxyz(plb_quat_w, ee_pos_w - plb_pos_w)
        ee_quat_plb = quat_mul_wxyz(
            quat_conjugate_wxyz(plb_quat_w), ee_quat_w
        )
        ee_quat_plb = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_plb))
        _, geometric_pitch, yaw = euler_xyz_from_quat_wxyz(ee_quat_plb)
        # Inverse of the command conversion above.  Since the sign is +/-1,
        # its inverse is itself.  This keeps debug pitch in policy coordinates.
        policy_pitch = ee_pitch_to_euler_sign * float(geometric_pitch)
        return ee_pos_plb.astype(np.float32), float(yaw), policy_pitch

    def compute_actual_ee_keypoints_plb():
        """Return actual [kp0,kp1,kp2] of gripperStator in the current PLB.

        This mirrors the frozen WBC command semantics exactly:
          kp0 = gripperStator origin
          kp1 = kp0 + R_ee * [kp_dx, 0, 0]
          kp2 = kp0 + R_ee * [0, 0, kp_dz]
        and then expresses all three world points in the yaw-only PLB frame.
        """
        plb_pos_w, plb_quat_w = get_plb_pose_world()
        ee_pos_w = d.xpos[ee_bid].copy().astype(np.float32)
        ee_rot_w = d.xmat[ee_bid].reshape(3, 3).copy().astype(np.float32)

        kp0_w = ee_pos_w
        kp1_w = ee_pos_w + ee_rot_w @ np.array(
            [ee_kp_dx, 0.0, 0.0], dtype=np.float32
        )
        kp2_w = ee_pos_w + ee_rot_w @ np.array(
            [0.0, 0.0, ee_kp_dz], dtype=np.float32
        )

        kps_w = (kp0_w, kp1_w, kp2_w)
        kps_plb = [
            quat_apply_inverse_wxyz(plb_quat_w, kp_w - plb_pos_w)
            for kp_w in kps_w
        ]
        return np.concatenate(kps_plb).astype(np.float32)

    def ee_kp_plb_to_world(kps_plb: np.ndarray):
        plb_pos_w, plb_quat_w = get_plb_pose_world()
        kps = np.asarray(kps_plb, dtype=np.float32).reshape(3, 3)
        return np.stack(
            [plb_pos_w + quat_apply_wxyz(plb_quat_w, p) for p in kps],
            axis=0,
        )

    def get_cube_pos_w():
        return d.xpos[cube_bid].copy().astype(np.float32)

    def get_gripper_geometry():
        root_pos_w, root_quat_w = get_root_pose()

        stator_pos_w = d.xpos[ee_bid].copy().astype(np.float32)
        stator_rot_w = d.xmat[ee_bid].reshape(3, 3).copy().astype(np.float32)

        gripper_center_pos_w = (
            stator_pos_w + stator_rot_w @ gripper_center_offset_local
        ).astype(np.float32)

        object_pos_w = get_cube_pos_w()

        object_center_pos_base = quat_apply_inverse_wxyz(
            root_quat_w, object_pos_w - root_pos_w
        ).astype(np.float32)
        gripper_center_pos_base = quat_apply_inverse_wxyz(
            root_quat_w, gripper_center_pos_w - root_pos_w
        ).astype(np.float32)

        # Gripper local +X and +Y expressed in current root/body frame.
        gripper_x_axis_w = stator_rot_w[:, 0].astype(np.float32)
        gripper_y_axis_w = stator_rot_w[:, 1].astype(np.float32)
        gripper_x_axis_base = quat_apply_inverse_wxyz(
            root_quat_w, gripper_x_axis_w
        ).astype(np.float32)
        gripper_y_axis_base = quat_apply_inverse_wxyz(
            root_quat_w, gripper_y_axis_w
        ).astype(np.float32)
        gripper_orientation_base = np.concatenate(
            [gripper_x_axis_base, gripper_y_axis_base]
        ).astype(np.float32)

        grasp_error = float(np.linalg.norm(object_pos_w - gripper_center_pos_w))

        return (
            object_center_pos_base,
            gripper_orientation_base,
            gripper_center_pos_base,
            gripper_center_pos_w,
            grasp_error,
        )

    def build_neutral_ee_command():
        return build_keypoints_from_kp0_yaw_pitch_plb(
            kp0=neutral_kp0,
            yaw=neutral_ee_yaw,
            pitch=neutral_ee_pitch,
            roll=fixed_ee_roll,
            kp_dx=ee_kp_dx,
            kp_dz=ee_kp_dz,
            pitch_to_euler_sign=ee_pitch_to_euler_sign,
        )

    def decode_hl_action(action: np.ndarray):
        nonlocal base_command
        nonlocal ee_cmd_plb_current
        nonlocal gripper_target
        nonlocal executed_gripper_cmd_norm
        nonlocal raw_gripper_action

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        base_command = (action[0:3] * base_cmd_scale).astype(np.float32)

        # ABS high-level arm/EE semantics: each normalized action maps directly
        # to the configured PLB command range.  There is NO measured-EE re-anchoring.
        # The logical pitch coordinate is converted to the Z1 gripper geometric
        # frame inside build_keypoints_from_kp0_yaw_pitch_plb().
        kp0_cmd = np.array(
            [
                action_to_range(action[3], kp0_x_range),
                action_to_range(action[4], kp0_y_range),
                action_to_range(action[5], kp0_z_range),
            ],
            dtype=np.float32,
        )
        yaw_cmd = action_to_range(action[6], ee_yaw_range)
        pitch_cmd = action_to_range(action[7], ee_pitch_range)

        ee_cmd_plb_current = build_keypoints_from_kp0_yaw_pitch_plb(
            kp0=kp0_cmd,
            yaw=float(yaw_cmd),
            pitch=float(pitch_cmd),
            roll=fixed_ee_roll,
            kp_dx=ee_kp_dx,
            kp_dz=ee_kp_dz,
            pitch_to_euler_sign=ee_pitch_to_euler_sign,
        )

        raw_gripper_action = float(action[8])
        # P1 alignment with training: strictly positive closes; zero remains open.
        binary_close = raw_gripper_action > gripper_binary_threshold
        executed_close = binary_close or (
            stage2_force_gripper_close_enabled and grasp_confidence_proxy
        )
        executed_gripper_cmd_norm = 1.0 if executed_close else -1.0
        gripper_target = gripper_close_pos if executed_close else gripper_open_pos

    def build_previous_hl_action():
        # Training observation semantics:
        # actual executed normalized base command, raw arm action[3:8],
        # and actually executed binary gripper action.
        denom = np.maximum(np.abs(base_cmd_scale), 1.0e-6)
        effective_base_action = np.clip(base_command / denom, -1.0, 1.0)
        return np.concatenate(
            [
                effective_base_action,
                current_hl_action[3:8],
                np.array([executed_gripper_cmd_norm], dtype=np.float32),
            ],
            dtype=np.float32,
        )

    def update_grasp_confidence_proxy():
        nonlocal grasp_confidence_proxy
        nonlocal grasp_proxy_enter_count
        nonlocal grasp_proxy_exit_count
        nonlocal prev_gripper_joint_pos

        qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy().astype(np.float32)
        gripper_joint_pos = float(qpos_mujoco[gripper_mujoco_index])

        _, _, _, _, grasp_error = get_gripper_geometry()

        close_commanded = executed_gripper_cmd_norm > 0.0
        gripper_angle_delta = abs(gripper_joint_pos - prev_gripper_joint_pos)
        gripper_not_fully_closed = (
            gripper_joint_pos < gripper_not_fully_closed_angle_threshold
        )
        gripper_angle_holding = gripper_angle_delta < gripper_angle_hold_threshold

        proxy_candidate = (
            close_commanded
            and grasp_error < grasp_proxy_error_threshold
            and gripper_not_fully_closed
            and gripper_angle_holding
        )

        if not grasp_confidence_proxy:
            if proxy_candidate:
                grasp_proxy_enter_count += 1
            else:
                grasp_proxy_enter_count = 0

            if grasp_proxy_enter_count >= grasp_proxy_enter_steps:
                grasp_confidence_proxy = True
                grasp_proxy_enter_count = 0
        else:
            if not proxy_candidate:
                grasp_proxy_exit_count += 1
            else:
                grasp_proxy_exit_count = 0

            if grasp_proxy_exit_count >= grasp_proxy_exit_steps:
                grasp_confidence_proxy = False
                grasp_proxy_exit_count = 0

        prev_gripper_joint_pos = gripper_joint_pos

    def build_hl_obs_frame():
        root_pos_w, root_quat_w = get_root_pose()

        # root_lin_vel is intentionally omitted from the high-level actor observation
        # to match the retrained policy. All remaining observation terms/order are unchanged.

        # Same IMU gyro source used by the validated low-level sim2sim.
        root_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)

        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(
            root_quat_w, gravity_w
        ).astype(np.float32)

        qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy().astype(np.float32)
        qvel_mujoco = d.qvel[6:6 + len(mujoco_joint_names)].copy().astype(np.float32)

        leg_pos_rel = (
            qpos_mujoco[leg_mujoco_indices]
            - default_joint_pos[leg_mujoco_indices]
        ).astype(np.float32)
        arm_pos_rel = (
            qpos_mujoco[arm_mujoco_indices]
            - default_joint_pos[arm_mujoco_indices]
        ).astype(np.float32)
        gripper_pos_rel = np.array(
            [qpos_mujoco[gripper_mujoco_index] - default_gripper_pos],
            dtype=np.float32,
        )
        arm_joint_vel = qvel_mujoco[arm_mujoco_indices].astype(np.float32)

        (
            object_center_pos_base,
            gripper_orientation_base,
            gripper_center_pos_base,
            _,
            _,
        ) = get_gripper_geometry()

        retrieval_target_pos_base = quat_apply_inverse_wxyz(
            root_quat_w, retrieval_target_pos_w - root_pos_w
        ).astype(np.float32)

        previous_hl_action = build_previous_hl_action()

        obs = np.concatenate(
            [
                root_ang_vel_b,                              # 3
                projected_gravity_b,                         # 3
                leg_pos_rel,                                 # 12
                arm_pos_rel,                                 # 6
                gripper_pos_rel,                             # 1
                arm_joint_vel,                               # 6
                object_center_pos_base,                      # 3
                gripper_orientation_base,                    # 6
                gripper_center_pos_base,                     # 3
                retrieval_target_pos_base,                   # 3
                previous_hl_action,                          # 9
                np.array([float(grasp_confidence_proxy)], dtype=np.float32), # 1
            ],
            dtype=np.float32,
        )
        assert obs.shape == (hl_obs_dim_per_step,), obs.shape
        return obs

    def build_ll_obs_frame():
        qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy().astype(np.float32)
        qvel_mujoco = d.qvel[6:6 + len(mujoco_joint_names)].copy().astype(np.float32)

        base_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)

        root_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(
            root_quat_w, gravity_w
        ).astype(np.float32)

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
                ee_cmd_plb_current,   # 9
                joint_pos_leg_rel,    # 12
                joint_pos_arm_rel,    # 6
                joint_vel_leg,        # 12
                joint_vel_arm,        # 6
                joint_vel_wheel,      # 4
                last_ll_action,       # 22
            ],
            dtype=np.float32,
        )
        assert obs.shape == (ll_obs_dim_per_step,), obs.shape
        return obs

    def sample_task_reset():
        nonlocal retrieval_target_pos_w

        root_pos_w, root_quat_w = get_root_pose()
        _, _, root_yaw = euler_xyz_from_quat_wxyz(root_quat_w)
        plb_quat_w = quat_unique_wxyz(
            quat_normalize_wxyz(quat_from_yaw_wxyz(root_yaw))
        )
        plb_pos_w = root_pos_w.copy()
        plb_pos_w[2] = ground_z

        # Cube reset: r uniform, theta uniform in front-sector, exactly like training.
        r_min, r_max = float(object_sampling_r_range[0]), float(object_sampling_r_range[1])
        r = float(rng.uniform(r_min, r_max))
        theta = float(rng.uniform(-object_sampling_theta_max, object_sampling_theta_max))
        object_rel_plb = np.array(
            [
                object_sampling_center_x + r * math.cos(theta),
                object_sampling_center_y + r * math.sin(theta),
                object_half_extent + object_spawn_z_epsilon,
            ],
            dtype=np.float32,
        )
        object_pos_w = (
            plb_pos_w + quat_apply_wxyz(plb_quat_w, object_rel_plb)
        ).astype(np.float32)

        d.qpos[cube_qpos_adr:cube_qpos_adr + 3] = object_pos_w
        d.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        d.qvel[cube_dof_adr:cube_dof_adr + 6] = 0.0

        # Retrieval target: uniform by AREA in a disk around initialized root world XY.
        #
        # enlarged_retrieval_target_sampling_range == False:
        #   preserve the original sim2sim logic exactly:
        #     XY radius = retrieval_target_radius (legacy 0.25 m)
        #     Z         = retrieval_target_z_w (legacy 0.50 m)
        #
        # enlarged_retrieval_target_sampling_range == True:
        #   match the enlarged training distribution:
        #     XY radius = enlarged_retrieval_target_radius (0.50 m)
        #     Z         ~ Uniform(enlarged_retrieval_target_z_range_w)
        target_theta = float(rng.uniform(-math.pi, math.pi))

        if enlarged_retrieval_target_sampling_range:
            active_target_radius = enlarged_retrieval_target_radius
            target_z_w = float(
                rng.uniform(
                    float(enlarged_retrieval_target_z_range_w[0]),
                    float(enlarged_retrieval_target_z_range_w[1]),
                )
            )
        else:
            active_target_radius = retrieval_target_radius
            target_z_w = retrieval_target_z_w

        target_radius = active_target_radius * math.sqrt(
            float(rng.uniform(0.0, 1.0))
        )
        retrieval_target_pos_w = np.array(
            [
                root_pos_w[0] + target_radius * math.cos(target_theta),
                root_pos_w[1] + target_radius * math.sin(target_theta),
                target_z_w,
            ],
            dtype=np.float32,
        )

    def reset_episode(sim_time: float):
        nonlocal base_command
        nonlocal ee_cmd_plb_current
        nonlocal last_ll_action
        nonlocal leg_target, arm_target, wheel_cmd, gripper_target
        nonlocal current_hl_action, raw_gripper_action, executed_gripper_cmd_norm
        nonlocal grasp_confidence_proxy, grasp_proxy_enter_count, grasp_proxy_exit_count
        nonlocal prev_gripper_joint_pos
        nonlocal episode_start_time, episode_index

        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0

        d.qpos[0:3] = root_pos_reset
        d.qpos[3:7] = root_quat_reset
        d.qpos[7:7 + len(mujoco_joint_names)] = default_joint_pos

        # P0 alignment with training:
        # The Isaac Lab environment explicitly resets jointGripper to the fully-open
        # physical position before the first valid observation is built.  Do the same
        # here; setting only gripper_target is not sufficient because the policy
        # observes the measured joint position.
        d.qpos[7 + gripper_mujoco_index] = gripper_open_pos

        # Give the cube a valid temporary state before mj_forward.
        d.qpos[cube_qpos_adr:cube_qpos_adr + 3] = np.array(
            [0.8, 0.0, object_half_extent + object_spawn_z_epsilon],
            dtype=np.float32,
        )
        d.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        d.qvel[cube_dof_adr:cube_dof_adr + 6] = 0.0

        mujoco.mj_forward(m, d)

        sample_task_reset()
        mujoco.mj_forward(m, d)

        base_command = np.zeros(3, dtype=np.float32)
        ee_cmd_plb_current = build_neutral_ee_command()
        last_ll_action = np.zeros(ll_action_dim, dtype=np.float32)

        leg_target = default_leg_pos.copy()
        arm_target = default_arm_pos.copy()
        wheel_cmd = np.zeros(4, dtype=np.float32)
        gripper_target = gripper_open_pos

        current_hl_action = np.zeros(hl_action_dim, dtype=np.float32)
        current_hl_action[3:8] = neutral_arm_action
        raw_gripper_action = -1.0
        executed_gripper_cmd_norm = -1.0

        grasp_confidence_proxy = False
        grasp_proxy_enter_count = 0
        grasp_proxy_exit_count = 0
        prev_gripper_joint_pos = gripper_open_pos

        # Match UAN actuator reset semantics from Isaac Lab: zero history.
        uan_history.fill(0.0)
        last_uan_action.fill(0.0)
        last_uan_delta_tau.fill(0.0)
        last_uan_tau_pd.fill(0.0)
        last_uan_tau_nom.fill(0.0)

        # Initialize histories exactly like training: repeat the first valid frame.
        ll_frame0 = build_ll_obs_frame()
        hl_frame0 = build_hl_obs_frame()
        fill_histories(ll_histories, ll_frame0, ll_feature_dims, ll_history_length)
        fill_histories(hl_histories, hl_frame0, hl_feature_dims, hl_history_length)

        episode_start_time = sim_time
        episode_index += 1

        cube_pos = get_cube_pos_w()
        print(
            f"[RESET #{episode_index}] root=({d.qpos[0]:+.3f},{d.qpos[1]:+.3f}) "
            f"cube_w=({cube_pos[0]:+.3f},{cube_pos[1]:+.3f},{cube_pos[2]:+.3f}) "
            f"target_w=({retrieval_target_pos_w[0]:+.3f},"
            f"{retrieval_target_pos_w[1]:+.3f},{retrieval_target_pos_w[2]:+.3f})"
        )

    debug_csv_file = None
    debug_csv_writer = None
    debug_ll_sample_count = 0

    if debug_tracking_logging:
        debug_csv_file = open(debug_tracking_csv_path, "w", newline="")
        debug_csv_writer = csv.writer(debug_csv_file)

        debug_header = [
            "sim_time_s", "episode", "root_z",
            *[f"hl_a{i}" for i in range(9)],
            *[f"ee_cmd_kp{i}_{axis}" for i in range(3) for axis in ("x", "y", "z")],
            *[f"ee_act_kp{i}_{axis}" for i in range(3) for axis in ("x", "y", "z")],
            "kp0_err_x", "kp0_err_y", "kp0_err_z", "kp0_err_norm",
            "kp0_cmd_z", "kp0_actual_z",
            "ee_yaw_cmd", "ee_pitch_cmd", "ee_yaw_actual", "ee_pitch_actual",
            "grasp_error", "grasp_dz",
            *[f"ll_arm_a{j+1}" for j in range(6)],
            *[f"arm_q_des{j+1}" for j in range(6)],
            *[f"arm_q{j+1}" for j in range(6)],
            *[f"arm_qd{j+1}" for j in range(6)],
            *[f"arm_q_err{j+1}" for j in range(6)],
            *[f"uan_a{j+1}" for j in range(6)],
            *[f"uan_dtau{j+1}" for j in range(6)],
        ]
        debug_csv_writer.writerow(debug_header)
        debug_csv_file.flush()

    def write_tracking_debug_row(sim_time_now: float):
        nonlocal debug_ll_sample_count
        if not debug_tracking_logging or debug_csv_writer is None:
            return

        debug_ll_sample_count += 1
        if debug_ll_sample_count % debug_tracking_log_every_ll_steps != 0:
            return

        qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy().astype(np.float32)
        qvel_mujoco = d.qvel[6:6 + len(mujoco_joint_names)].copy().astype(np.float32)
        arm_q = qpos_mujoco[arm_mujoco_indices]
        arm_qd = qvel_mujoco[arm_mujoco_indices]
        arm_q_err = arm_target - arm_q
        ll_arm_action = last_ll_action[arm_action_indices]

        ee_cmd_kps = ee_cmd_plb_current.copy().astype(np.float32)
        ee_act_kps = compute_actual_ee_keypoints_plb()
        kp0_cmd = ee_cmd_kps[0:3]
        kp0_act = ee_act_kps[0:3]
        kp0_err = kp0_act - kp0_cmd

        ee_pos_plb, ee_yaw_actual, ee_pitch_actual = compute_actual_ee_pose_plb()
        del ee_pos_plb
        ee_yaw_cmd = action_to_range(current_hl_action[6], ee_yaw_range)
        ee_pitch_cmd = action_to_range(current_hl_action[7], ee_pitch_range)

        _, _, _, gripper_center_pos_w, grasp_error = get_gripper_geometry()
        cube_center_pos_w = get_cube_pos_w()
        grasp_dz = float(gripper_center_pos_w[2] - cube_center_pos_w[2])

        row = [
            float(sim_time_now), int(episode_index), float(d.qpos[2]),
            *current_hl_action.tolist(),
            *ee_cmd_kps.tolist(),
            *ee_act_kps.tolist(),
            *kp0_err.tolist(), float(np.linalg.norm(kp0_err)),
            float(kp0_cmd[2]), float(kp0_act[2]),
            float(ee_yaw_cmd), float(ee_pitch_cmd),
            float(ee_yaw_actual), float(ee_pitch_actual),
            float(grasp_error), float(grasp_dz),
            *ll_arm_action.tolist(),
            *arm_target.tolist(),
            *arm_q.tolist(),
            *arm_qd.tolist(),
            *arm_q_err.tolist(),
            *last_uan_action.tolist(),
            *last_uan_delta_tau.tolist(),
        ]
        debug_csv_writer.writerow(row)
        debug_csv_file.flush()

    def run_high_level_policy():
        nonlocal current_hl_action

        # Proxy is estimated from the command and joint state over the preceding HL interval.
        update_grasp_confidence_proxy()

        hl_frame = build_hl_obs_frame()
        append_histories(hl_histories, hl_frame, hl_feature_dims)
        hl_obs_stack = flatten_feature_major(hl_histories)
        assert hl_obs_stack.shape == (hl_obs_dim,), hl_obs_stack.shape

        action = high_sess.run(
            [high_output_name],
            {high_input_name: hl_obs_stack[None, :]},
        )[0][0].astype(np.float32)
        current_hl_action = np.clip(action, -1.0, 1.0)
        decode_hl_action(current_hl_action)

    def run_low_level_policy():
        nonlocal last_ll_action, leg_target, arm_target, wheel_cmd

        ll_frame = build_ll_obs_frame()
        append_histories(ll_histories, ll_frame, ll_feature_dims)
        ll_obs_stack = flatten_feature_major(ll_histories)
        assert ll_obs_stack.shape == (ll_obs_dim,), ll_obs_stack.shape

        action = low_sess.run(
            [low_output_name],
            {low_input_name: ll_obs_stack[None, :]},
        )[0][0].astype(np.float32)
        last_ll_action = action

        leg_act = action[leg_action_indices]
        arm_act = action[arm_action_indices]
        wheel_act = action[wheel_action_indices]

        # No deployment blend-in and no arm-target rate limiter in this version.
        leg_target = default_leg_pos + leg_action_scale * leg_act
        arm_target = default_arm_pos + arm_action_scale * arm_act
        wheel_cmd = wheel_action_scale * wheel_act

    def apply_low_level_actuation():
        qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy().astype(np.float32)
        qvel_mujoco = d.qvel[6:6 + len(mujoco_joint_names)].copy().astype(np.float32)

        leg_pos = qpos_mujoco[leg_mujoco_indices]
        leg_vel = qvel_mujoco[leg_mujoco_indices]
        arm_pos = qpos_mujoco[arm_mujoco_indices]
        arm_vel = qvel_mujoco[arm_mujoco_indices]
        gripper_pos = float(qpos_mujoco[gripper_mujoco_index])
        gripper_vel = float(qvel_mujoco[gripper_mujoco_index])

        leg_tau = (
            kps[leg_mujoco_indices] * (leg_target - leg_pos)
            - kds[leg_mujoco_indices] * leg_vel
        )
        leg_tau = np.clip(
            leg_tau, -leg_torque_limits, leg_torque_limits
        ).astype(np.float32)

        # Frozen UAN actuator, exactly once per 4-ms MuJoCo physics step.
        # The latest 50-Hz WBC arm_target is zero-order-held across five steps.
        arm_tau = compute_uan_arm_torque(
            arm_target,
            arm_pos,
            arm_vel,
        )

        # jointGripper actuator dynamics aligned with IsaacLab DCMotor.
        #
        # IsaacLab first computes the IdealPD requested effort:
        #   tau_computed = Kp * (q_des - q) - Kd * qdot
        #
        # It then clips that effort with the linear four-quadrant DC-motor
        # torque-speed envelope:
        #   tau_top    = tau_stall * ( 1 - qdot / qdot_max )
        #   tau_bottom = tau_stall * (-1 - qdot / qdot_max )
        #   tau_max    = min(tau_top,    +tau_cont)
        #   tau_min    = max(tau_bottom, -tau_cont)
        #   tau_applied = clip(tau_computed, tau_min, tau_max)
        #
        # Before evaluating the envelope, IsaacLab clips joint velocity to:
        #   +/- velocity_limit * (1 + effort_limit / saturation_effort)
        gripper_tau_computed = (
            kps[gripper_mujoco_index] * (gripper_target - gripper_pos)
            - kds[gripper_mujoco_index] * gripper_vel
        )

        if gripper_velocity_limit <= 0.0:
            raise ValueError(
                f"gripper_velocity_limit must be > 0, got {gripper_velocity_limit}"
            )
        if gripper_saturation_effort <= 0.0:
            raise ValueError(
                "gripper_saturation_effort must be > 0, "
                f"got {gripper_saturation_effort}"
            )

        gripper_vel_at_effort_lim = gripper_velocity_limit * (
            1.0 + gripper_torque_limit / gripper_saturation_effort
        )
        gripper_vel_for_limit = float(
            np.clip(
                gripper_vel,
                -gripper_vel_at_effort_lim,
                gripper_vel_at_effort_lim,
            )
        )

        gripper_torque_speed_top = gripper_saturation_effort * (
            1.0 - gripper_vel_for_limit / gripper_velocity_limit
        )
        gripper_torque_speed_bottom = gripper_saturation_effort * (
            -1.0 - gripper_vel_for_limit / gripper_velocity_limit
        )

        gripper_max_effort = min(
            gripper_torque_speed_top,
            gripper_torque_limit,
        )
        gripper_min_effort = max(
            gripper_torque_speed_bottom,
            -gripper_torque_limit,
        )

        gripper_tau = float(
            np.clip(
                gripper_tau_computed,
                gripper_min_effort,
                gripper_max_effort,
            )
        )

        wheel_ctrl = np.clip(
            wheel_cmd, -wheel_velocity_limits, wheel_velocity_limits
        ).astype(np.float32)

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

    def update_custom_visualization(viewer):
        viewer.user_scn.ngeom = 0

        if vis_retrieval_target:
            add_sphere_to_scene(
                viewer.user_scn,
                retrieval_target_pos_w,
                vis_retrieval_target_radius,
                [1.0, 0.0, 1.0, 0.9],
            )

        if vis_ee_target:
            kps_w = ee_kp_plb_to_world(ee_cmd_plb_current)
            kp0, kp1, kp2 = kps_w
            x_dir = kp1 - kp0
            z_dir = kp2 - kp0
            x_norm = float(np.linalg.norm(x_dir))
            z_norm = float(np.linalg.norm(z_dir))
            if x_norm > 1.0e-8 and z_norm > 1.0e-8:
                x_end = kp0 + vis_ee_target_axis_len * x_dir / x_norm
                z_end = kp0 + vis_ee_target_axis_len * z_dir / z_norm
                add_sphere_to_scene(
                    viewer.user_scn,
                    kp0,
                    vis_ee_target_sphere_size,
                    [1.0, 0.0, 0.0, 0.9],
                )
                add_capsule_to_scene(
                    viewer.user_scn, kp0, x_end,
                    vis_ee_target_axis_radius, [1.0, 0.0, 0.0, 0.9]
                )
                add_capsule_to_scene(
                    viewer.user_scn, kp0, z_end,
                    vis_ee_target_axis_radius, [0.0, 0.0, 1.0, 0.9]
                )

        if vis_grasp_debug:
            # Yellow: measured training-defined grasp center
            # Cyan:   measured cube center
            _, _, _, gripper_center_pos_w, _ = get_gripper_geometry()
            cube_center_pos_w = get_cube_pos_w()
            add_sphere_to_scene(
                viewer.user_scn,
                gripper_center_pos_w,
                vis_gripper_center_radius,
                [1.0, 1.0, 0.0, 0.95],
            )
            add_sphere_to_scene(
                viewer.user_scn,
                cube_center_pos_w,
                vis_cube_center_radius,
                [0.0, 1.0, 1.0, 0.95],
            )

    sim_time = 0.0
    sim_counter = 0
    ll_blocks_completed_in_hl_interval = 0

    reset_episode(sim_time)

    def infer_first_hl_and_ll_after_reset():
        """Match training's action/LL ordering at the start of an episode.

        Training semantics:
            reset observation -> HL action/decode -> LL obs/inference -> physics x5

        Therefore the first LL policy inference must happen before the first physics
        step, after the first HL command has already been decoded.
        """
        nonlocal current_hl_action
        nonlocal ll_blocks_completed_in_hl_interval

        hl_obs_stack0 = flatten_feature_major(hl_histories)
        first_hl_action = high_sess.run(
            [high_output_name],
            {high_input_name: hl_obs_stack0[None, :]},
        )[0][0].astype(np.float32)
        current_hl_action = np.clip(first_hl_action, -1.0, 1.0)
        decode_hl_action(current_hl_action)

        # P0 timing alignment: unlike the previous deployment version, do not spend
        # the first 20 ms executing reset/default targets. Build the LL observation
        # with the just-decoded HL command and infer the LL action immediately.
        run_low_level_policy()
        ll_blocks_completed_in_hl_interval = 0

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

            # The LL target/action for this 20-ms block has already been computed
            # before entering the block, matching training.
            apply_low_level_actuation()
            mujoco.mj_step(m, d)

            sim_time += simulation_dt
            sim_counter += 1
            viewer.cam.lookat[:] = d.qpos[:3]

            if sim_counter % ll_control_decimation == 0:
                # One 50-Hz LL block (5 UAN/physics steps) has just completed.
                ll_blocks_completed_in_hl_interval += 1

                # Training executes exactly five LL blocks for each HL action.
                # At the 0.1-s boundary, update task/proxy state, infer/decode the
                # next HL action, then immediately build the next LL observation.
                if ll_blocks_completed_in_hl_interval >= ll_steps_per_hl_step:
                    run_high_level_policy()
                    ll_blocks_completed_in_hl_interval = 0

                # Whether or not HL changed, the next LL inference occurs BEFORE
                # the next physics block.
                run_low_level_policy()
                write_tracking_debug_row(sim_time)

            if auto_reset and (sim_time - episode_start_time) >= episode_length_s:
                reset_episode(sim_time)
                infer_first_hl_and_ll_after_reset()

            update_custom_visualization(viewer)

            if print_interval_sim_steps > 0 and sim_counter % print_interval_sim_steps == 0:
                cube_pos_w = get_cube_pos_w()
                retrieval_error = float(
                    np.linalg.norm(cube_pos_w - retrieval_target_pos_w)
                )
                _, _, _, gripper_center_pos_w, grasp_error = get_gripper_geometry()
                cube_center_pos_w = get_cube_pos_w()
                grasp_dz = float(gripper_center_pos_w[2] - cube_center_pos_w[2])
                qpos_mujoco_dbg = d.qpos[
                    7:7 + len(mujoco_joint_names)
                ].copy().astype(np.float32)
                gripper_q_dbg = float(qpos_mujoco_dbg[gripper_mujoco_index])
                print(
                    f"[{sim_counter:7d}] t={sim_time:7.2f}s "
                    f"| ep={episode_index:03d} "
                    f"| z={d.qpos[2]:.3f} "
                    f"| grasp_err={grasp_error:.3f} "
                    f"| grasp_dz={grasp_dz:+.3f} "
                    f"| grip_q={gripper_q_dbg:+.3f} "
                    f"| proxy={int(grasp_confidence_proxy)} "
                    f"| grip_raw={raw_gripper_action:+.3f} "
                    f"| grip_exec={executed_gripper_cmd_norm:+.0f} "
                    f"| retrieval_err={retrieval_error:.3f} "
                    f"| UAN_sat={np.mean(np.abs(last_uan_action) > uan_action_saturation_threshold):.3f} "
                    f"| UAN_nomclip={np.mean(np.abs(last_uan_tau_pd) > uan_nominal_torque_limits):.3f} "
                    f"| UAN_dTau={np.mean(np.abs(last_uan_delta_tau)):.2f}Nm"
                )
                print(
                    "            HL="
                    + np.array2string(
                        current_hl_action,
                        precision=2,
                        suppress_small=True,
                        max_line_width=180,
                    )
                    + f" | base_cmd={np.array2string(base_command, precision=3)}"
                )
                ee_actual_kps_dbg = compute_actual_ee_keypoints_plb()
                kp0_cmd_dbg = ee_cmd_plb_current[0:3]
                kp0_act_dbg = ee_actual_kps_dbg[0:3]
                kp0_err_dbg = kp0_act_dbg - kp0_cmd_dbg
                qvel_mujoco_dbg = d.qvel[
                    6:6 + len(mujoco_joint_names)
                ].copy().astype(np.float32)
                arm_q_dbg = qpos_mujoco_dbg[arm_mujoco_indices]
                arm_qd_dbg = qvel_mujoco_dbg[arm_mujoco_indices]
                ll_arm_dbg = last_ll_action[arm_action_indices]
                print(
                    "            TRACK kp0_cmd="
                    + np.array2string(kp0_cmd_dbg, precision=3, suppress_small=True)
                    + " kp0_act="
                    + np.array2string(kp0_act_dbg, precision=3, suppress_small=True)
                    + " err(act-cmd)="
                    + np.array2string(kp0_err_dbg, precision=3, suppress_small=True)
                    + f" | |e|={np.linalg.norm(kp0_err_dbg):.3f}"
                )
                print(
                    "            LL_arm="
                    + np.array2string(ll_arm_dbg, precision=3, suppress_small=True)
                    + " q_des="
                    + np.array2string(arm_target, precision=3, suppress_small=True)
                    + " q="
                    + np.array2string(arm_q_dbg, precision=3, suppress_small=True)
                    + " qd="
                    + np.array2string(arm_qd_dbg, precision=3, suppress_small=True)
                )
                kp_errs_dbg = ee_actual_kps_dbg.reshape(3, 3) - ee_cmd_plb_current.reshape(3, 3)
                print(
                    "            KP_ERR_NORM="
                    + np.array2string(
                        np.linalg.norm(kp_errs_dbg, axis=1),
                        precision=3,
                        suppress_small=True,
                    )
                    + " | UAN_a="
                    + np.array2string(last_uan_action, precision=3, suppress_small=True)
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