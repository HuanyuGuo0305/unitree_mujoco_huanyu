#!/usr/bin/env python3
"""
Headless motion-profile evaluation for one hierarchical B2WZ1 retrieval episode.

Purpose
-------
Visualize whether the high-level retrieval policy produces genuinely dynamic,
simultaneous base-arm behavior. The deployed control hierarchy is unchanged:

    MuJoCo / frozen UAN : 250 Hz
    frozen low-level WBC:  50 Hz
    high-level retrieval:  10 Hz

One retrieval episode is run headlessly and the following ACTUAL robot motion
signals are logged at the 250-Hz physics rate:

1) Base planar speed + signed base yaw rate
2) EE planar speed in world XY
3) Executed gripper action (0=open, 1=close)

The script also reports how much of the episode contains simultaneous base and
EE motion. This diagnostic is not needed for the plot but is useful when
arguing that the policy is dynamic rather than move-stop-manipulate.

Cube geometry, mass and inertia are scaled consistently with constant density:

    mass = density * side^3

Success uses the same task-level definition as the task-performance evaluator:
the cube center must remain within the retrieval-target radius for a continuous
dwell interval. Regrasp/recovery is allowed.

Outputs
-------
- hl_motion_profile.png
- hl_motion_profile.pdf
- hl_motion_profile.csv
- hl_motion_profile.npz
- hl_motion_profile_summary.json

Example
-------
Run from unitree_mujoco/simulate_python:

    python3 deploy_mujoco/b2wz1_hl_motion_profile.py \
        configs/b2wz1_hl_retrieval_abs_uan.yaml \
        --seed 0 \
        --cube-side-cm 5.0 \
        --out results/hl_motion_profile
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import json
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
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




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="Path to hierarchical sim2sim YAML")
    parser.add_argument("--seed", type=int, default=0, help="Task reset seed.")
    parser.add_argument(
        "--cube-side-cm",
        type=float,
        default=5.0,
        help="Fixed cube side length for this episode [cm].",
    )
    parser.add_argument(
        "--object-density",
        type=float,
        default=None,
        help="Cube density [kg/m^3]. Default: YAML object_density or 800.",
    )
    parser.add_argument(
        "--success-radius-m",
        type=float,
        default=0.10,
        help="3-D cube-to-target distance required for task success.",
    )
    parser.add_argument(
        "--success-dwell-s",
        type=float,
        default=0.30,
        help="Continuous dwell required inside success radius.",
    )
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=0.04,
        help="Moving-average window used only for plotted velocity signals.",
    )
    parser.add_argument(
        "--base-motion-threshold-mps",
        type=float,
        default=0.05,
        help="Threshold used for simultaneous-motion diagnostic.",
    )
    parser.add_argument(
        "--yaw-motion-threshold-rps",
        type=float,
        default=0.10,
        help="Absolute yaw-rate threshold used for simultaneous-motion diagnostic.",
    )
    parser.add_argument(
        "--ee-motion-threshold-mps",
        type=float,
        default=0.05,
        help="EE planar-speed threshold used for simultaneous-motion diagnostic.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/hl_motion_profile",
        help="Output directory.",
    )
    args = parser.parse_args()

    if args.cube_side_cm <= 0.0:
        raise ValueError("--cube-side-cm must be > 0")
    if args.success_radius_m <= 0.0:
        raise ValueError("--success-radius-m must be > 0")
    if args.success_dwell_s <= 0.0:
        raise ValueError("--success-dwell-s must be > 0")
    if args.smooth_window_s < 0.0:
        raise ValueError("--smooth-window-s must be >= 0")

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
    assert hl_obs_dim_per_step == 55
    assert hl_obs_dim == 165
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

    # jointGripper actuator model.
    #   "dc_motor"    -> IsaacLab DCMotor torque-speed envelope, as trained.
    #   "pd_position" -> ideal PD position actuator, effort-clipped only.
    gripper_mode = str(cfg.get("gripper_mode", "dc_motor")).strip().lower()
    if gripper_mode not in ("dc_motor", "pd_position"):
        raise ValueError(
            "gripper_mode must be 'dc_motor' or 'pd_position', got "
            f"{cfg.get('gripper_mode')!r}."
        )

    # pd_position parameters.  The gains default to the jointGripper entries of
    # the kps/kds tables and are resolved once the MuJoCo joint order is known.
    gripper_pd_kp_cfg = cfg.get("gripper_pd_kp", None)
    gripper_pd_kd_cfg = cfg.get("gripper_pd_kd", None)
    gripper_pd_torque_limit = float(
        cfg.get("gripper_pd_torque_limit", gripper_torque_limit)
    )
    if gripper_mode == "pd_position" and gripper_pd_torque_limit <= 0.0:
        raise ValueError(
            f"gripper_pd_torque_limit must be > 0, got {gripper_pd_torque_limit}"
        )
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
    auto_reset = False  # batch evaluator owns reset/termination
    object_density = float(
        args.object_density
        if args.object_density is not None
        else cfg.get("object_density", 800.0)
    )
    if object_density <= 0.0:
        raise ValueError(f"object density must be > 0, got {object_density}")

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Visualization/debug logging disabled in this headless evaluator.

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
    print("HL actor obs:      55-D/frame x 3 = 165-D; grasp proxy NOT exposed")
    print("Internal proxy:     retained for force-close/task logic")
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
    if gripper_mode == "pd_position":
        print(
            "Gripper actuator:   pd_position, "
            f"effort=+/-{gripper_pd_torque_limit:.3f} Nm, "
            "no torque-speed envelope"
        )
    else:
        print(
            "Gripper actuator:   dc_motor, "
            f"effort={gripper_torque_limit:.3f} Nm, "
            f"stall={gripper_saturation_effort:.3f} Nm, "
            f"no-load vel={gripper_velocity_limit:.3f} rad/s"
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

    # Resolve the cube collision geom attached to retrieval_cube.  Runtime
    # resizing is applied to this box geom; body mass/inertia are updated
    # consistently with constant density.
    cube_geom_cfg = cfg.get("cube_geom", None)
    if cube_geom_cfg is not None:
        cube_gid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_GEOM, str(cube_geom_cfg)
        )
        if cube_gid < 0:
            raise ValueError(f"Configured cube_geom not found: {cube_geom_cfg}")
    else:
        geom_adr = int(m.body_geomadr[cube_bid])
        geom_num = int(m.body_geomnum[cube_bid])
        candidates = [
            gid
            for gid in range(geom_adr, geom_adr + geom_num)
            if int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        ]
        if len(candidates) != 1:
            names = [
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
                for gid in range(geom_adr, geom_adr + geom_num)
            ]
            raise RuntimeError(
                "Expected exactly one BOX geom on retrieval_cube; "
                f"found {len(candidates)} among {names}. "
                "Set cube_geom in YAML to disambiguate."
            )
        cube_gid = int(candidates[0])

    cube_geom_name = mujoco.mj_id2name(
        m, mujoco.mjtObj.mjOBJ_GEOM, cube_gid
    )

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

    # jointGripper PD gains.  Both modes evaluate the same ideal-PD law first,
    # so the kps/kds table entries are the default for either; pd_position can
    # override them without touching the DCMotor path.
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
    if gripper_mode == "pd_position":
        if gripper_pd_kp <= 0.0 or gripper_pd_kd < 0.0:
            raise ValueError(
                "Invalid jointGripper PD gains: "
                f"kp={gripper_pd_kp}, kd={gripper_pd_kd}."
            )
        print(
            f"Gripper pd_position: kp={gripper_pd_kp:.3f}, "
            f"kd={gripper_pd_kd:.3f}, "
            f"effort=+/-{gripper_pd_torque_limit:.3f} Nm"
        )
        # The PD law writes a torque into a MuJoCo <motor>, so a narrower
        # ctrlrange in the XML silently caps the requested effort.
        if "jointGripper" in ctrl_joint_names:
            gripper_ctrl_i = ctrl_joint_names.index("jointGripper")
            if hasattr(m, "actuator_ctrllimited") and bool(
                m.actuator_ctrllimited[gripper_ctrl_i]
            ):
                gripper_ctrl_range = m.actuator_ctrlrange[gripper_ctrl_i]
                if (
                    gripper_ctrl_range[0] > -gripper_pd_torque_limit + 1e-6
                    or gripper_ctrl_range[1] < gripper_pd_torque_limit - 1e-6
                ):
                    print(
                        "  WARNING: MuJoCo jointGripper actuator ctrlrange "
                        f"[{gripper_ctrl_range[0]:.1f}, "
                        f"{gripper_ctrl_range[1]:.1f}] is narrower than "
                        f"+/-{gripper_pd_torque_limit:.1f} Nm; MuJoCo clamps."
                    )

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
    # High-level actor observation matches the no-proxy training interface.
    # Feature order per frame:
    #   root_ang_vel_b(3), projected_gravity_b(3), leg_pos_rel(12),
    #   arm_pos_rel(6), gripper_pos_rel(1), arm_joint_vel(6),
    #   object_center_pos_base(3), gripper_orientation_base(6),
    #   gripper_center_pos_base(3), retrieval_target_pos_base(3),
    #   previous_hl_action(9) = 55.
    #
    # grasp_confidence_proxy is intentionally NOT part of the actor observation.
    # It is still maintained below for unchanged internal force-close/task logic.
    hl_feature_dims = [3, 3, 12, 6, 1, 6, 3, 6, 3, 3, 9]
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

        # root_lin_vel and grasp_confidence_proxy are intentionally omitted from the
        # high-level actor observation to match the retrained no-proxy policy.
        # All remaining observation terms/order are unchanged.

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

        if gripper_mode == "pd_position":
            # Ideal PD position actuator: the requested effort is clipped
            # only to the symmetric effort limit, with no DC-motor
            # torque-speed envelope and no velocity pre-clip.
            gripper_tau = float(
                np.clip(
                    gripper_pd_kp * (gripper_target - gripper_pos)
                    - gripper_pd_kd * gripper_vel,
                    -gripper_pd_torque_limit,
                    gripper_pd_torque_limit,
                )
            )
        else:
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

    # ------------------------------------------------------------------
    # Headless one-episode motion-profile evaluator
    # ------------------------------------------------------------------

    def set_cube_physical_properties(side_m: float) -> tuple[float, float]:
        """Resize cube and update constant-density mass/inertia."""
        side_m = float(side_m)
        half = 0.5 * side_m
        mass = object_density * side_m ** 3
        inertia_diag = mass * side_m ** 2 / 6.0

        if int(m.geom_type[cube_gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise RuntimeError("Resolved cube geom is not a MuJoCo BOX")

        m.geom_size[cube_gid, 0:3] = half
        m.body_mass[cube_bid] = mass
        m.body_inertia[cube_bid, 0:3] = inertia_diag
        if hasattr(m, "geom_rbound"):
            m.geom_rbound[cube_gid] = math.sqrt(3.0) * half

        mujoco.mj_setConst(m, d)
        return half, mass

    def sample_episode_spec(seed: int) -> dict[str, float]:
        """Sample one task reset using the same distributions as deployment."""
        spec_rng = np.random.default_rng(int(seed))

        r_min = float(object_sampling_r_range[0])
        r_max = float(object_sampling_r_range[1])

        object_r = float(spec_rng.uniform(r_min, r_max))
        object_theta = float(
            spec_rng.uniform(
                -object_sampling_theta_max,
                object_sampling_theta_max,
            )
        )

        target_theta = float(spec_rng.uniform(-math.pi, math.pi))
        if enlarged_retrieval_target_sampling_range:
            target_radius_max = enlarged_retrieval_target_radius
            target_z = float(
                spec_rng.uniform(
                    float(enlarged_retrieval_target_z_range_w[0]),
                    float(enlarged_retrieval_target_z_range_w[1]),
                )
            )
        else:
            target_radius_max = retrieval_target_radius
            target_z = retrieval_target_z_w

        # Uniform by AREA in disk, matching deployment.
        target_radius = target_radius_max * math.sqrt(
            float(spec_rng.uniform(0.0, 1.0))
        )

        return {
            "object_r": object_r,
            "object_theta": object_theta,
            "target_theta": target_theta,
            "target_radius": target_radius,
            "target_z": target_z,
        }

    def reset_profile_episode(
        spec: dict[str, float],
        side_m: float,
    ) -> tuple[float, float]:
        """Reset one episode with deterministic task spec."""
        nonlocal base_command
        nonlocal ee_cmd_plb_current
        nonlocal last_ll_action
        nonlocal leg_target, arm_target, wheel_cmd, gripper_target
        nonlocal current_hl_action, raw_gripper_action, executed_gripper_cmd_norm
        nonlocal grasp_confidence_proxy, grasp_proxy_enter_count, grasp_proxy_exit_count
        nonlocal prev_gripper_joint_pos
        nonlocal retrieval_target_pos_w

        half_extent, mass = set_cube_physical_properties(side_m)

        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0

        d.qpos[0:3] = root_pos_reset
        d.qpos[3:7] = root_quat_reset
        d.qpos[7:7 + len(mujoco_joint_names)] = default_joint_pos
        d.qpos[7 + gripper_mujoco_index] = gripper_open_pos

        d.qpos[cube_qpos_adr:cube_qpos_adr + 3] = np.array(
            [0.8, 0.0, half_extent + object_spawn_z_epsilon],
            dtype=np.float32,
        )
        d.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        d.qvel[cube_dof_adr:cube_dof_adr + 6] = 0.0
        mujoco.mj_forward(m, d)

        root_pos_w, root_quat_w = get_root_pose()
        _, _, root_yaw = euler_xyz_from_quat_wxyz(root_quat_w)
        plb_quat_w = quat_unique_wxyz(
            quat_normalize_wxyz(quat_from_yaw_wxyz(root_yaw))
        )
        plb_pos_w = root_pos_w.copy()
        plb_pos_w[2] = ground_z

        r = float(spec["object_r"])
        theta = float(spec["object_theta"])
        object_rel_plb = np.array(
            [
                object_sampling_center_x + r * math.cos(theta),
                object_sampling_center_y + r * math.sin(theta),
                half_extent + object_spawn_z_epsilon,
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

        target_theta = float(spec["target_theta"])
        target_radius = float(spec["target_radius"])
        retrieval_target_pos_w = np.array(
            [
                root_pos_w[0] + target_radius * math.cos(target_theta),
                root_pos_w[1] + target_radius * math.sin(target_theta),
                float(spec["target_z"]),
            ],
            dtype=np.float32,
        )
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

        uan_history.fill(0.0)
        last_uan_action.fill(0.0)
        last_uan_delta_tau.fill(0.0)
        last_uan_tau_pd.fill(0.0)
        last_uan_tau_nom.fill(0.0)

        ll_frame0 = build_ll_obs_frame()
        hl_frame0 = build_hl_obs_frame()
        fill_histories(
            ll_histories, ll_frame0, ll_feature_dims, ll_history_length
        )
        fill_histories(
            hl_histories, hl_frame0, hl_feature_dims, hl_history_length
        )
        return half_extent, mass

    def infer_first_hl_and_ll_after_reset_profile() -> None:
        """Match training ordering: reset obs -> HL -> LL -> physics."""
        nonlocal current_hl_action

        hl_obs_stack0 = flatten_feature_major(hl_histories)
        first_hl_action = high_sess.run(
            [high_output_name],
            {high_input_name: hl_obs_stack0[None, :]},
        )[0][0].astype(np.float32)
        current_hl_action = np.clip(first_hl_action, -1.0, 1.0)
        decode_hl_action(current_hl_action)
        run_low_level_policy()

    def moving_average_same(x: np.ndarray, window_s: float) -> np.ndarray:
        """Centered edge-padded moving average, used only for visualization."""
        x = np.asarray(x, dtype=np.float64)
        if window_s <= 0.0:
            return x.copy()

        n = max(1, int(round(window_s / simulation_dt)))
        if n <= 1:
            return x.copy()
        if n % 2 == 0:
            n += 1

        pad = n // 2
        x_pad = np.pad(x, (pad, pad), mode="edge")
        kernel = np.ones(n, dtype=np.float64) / float(n)
        return np.convolve(x_pad, kernel, mode="valid")

    def make_motion_profile_plot(
        time_s: np.ndarray,
        base_speed_mps: np.ndarray,
        base_yaw_rate_rps: np.ndarray,
        ee_speed_mps: np.ndarray,
        gripper_close: np.ndarray,
        success: bool,
        completion_time_s: float,
        side_cm: float,
        mass_kg: float,
    ) -> None:
        """Three-panel slide-ready motion profile."""
        base_plot = moving_average_same(
            base_speed_mps, args.smooth_window_s
        )
        yaw_plot = moving_average_same(
            base_yaw_rate_rps, args.smooth_window_s
        )
        ee_plot = moving_average_same(
            ee_speed_mps, args.smooth_window_s
        )

        fig, axes = plt.subplots(
            3, 1, figsize=(10.8, 7.2), sharex=True,
            gridspec_kw={"height_ratios": [1.25, 1.0, 0.75]},
        )

        # Top: actual base planar speed + signed yaw rate.
        ax = axes[0]
        line_speed, = ax.plot(
            time_s, base_plot, linewidth=1.7,
            label="Base planar speed"
        )
        ax.set_ylabel("Planar speed [m/s]")
        ax.grid(True, alpha=0.25)

        ax_yaw = ax.twinx()
        line_yaw, = ax_yaw.plot(
            time_s, yaw_plot, linestyle="--", linewidth=1.5,
            label="Base yaw rate"
        )
        ax_yaw.set_ylabel("Yaw rate [rad/s]")
        ax.set_title("Base Motion")
        ax.legend(
            [line_speed, line_yaw],
            [line_speed.get_label(), line_yaw.get_label()],
            frameon=False,
            loc="upper right",
        )

        # Middle: actual EE world-XY speed.
        ax = axes[1]
        ax.plot(
            time_s, ee_plot, linewidth=1.7,
            label="EE planar speed"
        )
        ax.set_ylabel("EE speed [m/s]")
        ax.set_title("End-Effector Motion")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="upper right")

        # Bottom: actually executed binary gripper action.
        ax = axes[2]
        ax.step(
            time_s, gripper_close,
            where="post", linewidth=1.7,
            label="Gripper action"
        )
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["Open", "Close"])
        ax.set_ylim(-0.15, 1.15)
        ax.set_ylabel("Gripper")
        ax.set_xlabel("Time [s]")
        ax.set_title("Gripper Action")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="upper right")

        if success and np.isfinite(completion_time_s):
            for panel in axes:
                panel.axvline(
                    completion_time_s,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.75,
                )

        status = (
            f"success, completion={completion_time_s:.2f} s"
            if success and np.isfinite(completion_time_s)
            else "failure / timeout"
        )
        fig.suptitle(
            f"Motion Profile During Retrieval Episode  |  "
            f"cube={side_cm:.1f} cm, mass={mass_kg:.3f} kg  |  {status}",
            fontsize=13,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

        fig.savefig(
            os.path.join(out_dir, "hl_motion_profile.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(out_dir, "hl_motion_profile.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # Run one episode and log all motion signals at 250 Hz.
    # ------------------------------------------------------------------
    side_m = float(args.cube_side_cm) / 100.0
    spec = sample_episode_spec(args.seed)
    _, cube_mass_kg = reset_profile_episode(spec, side_m)
    infer_first_hl_and_ll_after_reset_profile()

    max_steps = int(math.ceil(episode_length_s / simulation_dt))
    dwell_steps_required = max(
        1, int(math.ceil(args.success_dwell_s / simulation_dt))
    )

    time_log = []
    base_planar_speed_log = []
    base_yaw_rate_log = []
    ee_planar_speed_log = []
    gripper_close_log = []
    gripper_raw_log = []
    retrieval_error_log = []
    base_cmd_log = []
    ee_pos_world_log = []

    ll_blocks_completed = 0
    success_dwell_steps = 0
    success = False
    completion_time_s = float("nan")

    prev_ee_pos_w = d.xpos[ee_bid].copy().astype(np.float64)

    for step in range(max_steps):
        apply_low_level_actuation()
        mujoco.mj_step(m, d)

        if (step + 1) % ll_control_decimation == 0:
            ll_blocks_completed += 1
            if ll_blocks_completed >= ll_steps_per_hl_step:
                run_high_level_policy()
                ll_blocks_completed = 0
            run_low_level_policy()

        t = (step + 1) * simulation_dt

        # Actual base planar speed. The XY norm is invariant to yaw-frame choice.
        base_planar_speed = float(np.linalg.norm(d.qvel[0:2]))

        # Signed body-frame yaw rate from the same IMU gyro used by the policy.
        base_yaw_rate = float(
            get_sensor_slice(m, d, "imu_gyro").astype(np.float32)[2]
        )

        # Actual EE planar speed in world XY from finite differences.
        ee_pos_w = d.xpos[ee_bid].copy().astype(np.float64)
        ee_vel_xy = (ee_pos_w[0:2] - prev_ee_pos_w[0:2]) / simulation_dt
        ee_planar_speed = float(np.linalg.norm(ee_vel_xy))
        prev_ee_pos_w = ee_pos_w

        gripper_close = 1.0 if executed_gripper_cmd_norm > 0.0 else 0.0
        retrieval_error = float(
            np.linalg.norm(get_cube_pos_w() - retrieval_target_pos_w)
        )

        time_log.append(t)
        base_planar_speed_log.append(base_planar_speed)
        base_yaw_rate_log.append(base_yaw_rate)
        ee_planar_speed_log.append(ee_planar_speed)
        gripper_close_log.append(gripper_close)
        gripper_raw_log.append(float(raw_gripper_action))
        retrieval_error_log.append(retrieval_error)
        base_cmd_log.append(base_command.copy())
        ee_pos_world_log.append(ee_pos_w.copy())

        if retrieval_error <= args.success_radius_m:
            success_dwell_steps += 1
        else:
            success_dwell_steps = 0

        if success_dwell_steps >= dwell_steps_required:
            success = True
            completion_time_s = t
            break

    time_s = np.asarray(time_log, dtype=np.float64)
    base_speed_mps = np.asarray(base_planar_speed_log, dtype=np.float64)
    base_yaw_rate_rps = np.asarray(base_yaw_rate_log, dtype=np.float64)
    ee_speed_mps = np.asarray(ee_planar_speed_log, dtype=np.float64)
    gripper_close = np.asarray(gripper_close_log, dtype=np.float64)
    gripper_raw = np.asarray(gripper_raw_log, dtype=np.float64)
    retrieval_error_m = np.asarray(retrieval_error_log, dtype=np.float64)
    base_cmd_arr = np.asarray(base_cmd_log, dtype=np.float64)
    ee_pos_world = np.asarray(ee_pos_world_log, dtype=np.float64)

    # Quantify simultaneous whole-body motion without changing the plot.
    base_is_moving = (
        (base_speed_mps >= args.base_motion_threshold_mps)
        | (np.abs(base_yaw_rate_rps) >= args.yaw_motion_threshold_rps)
    )
    ee_is_moving = ee_speed_mps >= args.ee_motion_threshold_mps
    simultaneous = base_is_moving & ee_is_moving

    simultaneous_time_s = float(np.sum(simultaneous) * simulation_dt)
    simultaneous_fraction = (
        float(np.mean(simultaneous)) if simultaneous.size else float("nan")
    )

    summary = {
        "seed": int(args.seed),
        "cube_side_cm": float(args.cube_side_cm),
        "cube_mass_kg": float(cube_mass_kg),
        "object_density_kg_m3": float(object_density),
        "success": bool(success),
        "completion_time_s": (
            float(completion_time_s)
            if np.isfinite(completion_time_s)
            else None
        ),
        "episode_logged_duration_s": float(time_s[-1]) if len(time_s) else 0.0,
        "peak_base_planar_speed_mps": float(np.max(base_speed_mps)),
        "peak_abs_base_yaw_rate_rps": float(np.max(np.abs(base_yaw_rate_rps))),
        "peak_ee_planar_speed_mps": float(np.max(ee_speed_mps)),
        "simultaneous_motion_time_s": simultaneous_time_s,
        "simultaneous_motion_fraction": simultaneous_fraction,
        "base_motion_threshold_mps": float(args.base_motion_threshold_mps),
        "yaw_motion_threshold_rps": float(args.yaw_motion_threshold_rps),
        "ee_motion_threshold_mps": float(args.ee_motion_threshold_mps),
        "success_radius_m": float(args.success_radius_m),
        "success_dwell_s": float(args.success_dwell_s),
        "object_spawn_r_m": float(spec["object_r"]),
        "object_spawn_theta_rad": float(spec["object_theta"]),
        "retrieval_target_radius_m": float(spec["target_radius"]),
        "retrieval_target_theta_rad": float(spec["target_theta"]),
        "retrieval_target_z_m": float(spec["target_z"]),
    }

    # CSV is convenient for quick inspection / replotting outside Python.
    csv_path = os.path.join(out_dir, "hl_motion_profile.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "base_planar_speed_mps",
                "base_yaw_rate_rps",
                "ee_planar_speed_mps",
                "gripper_close_01",
                "gripper_raw_action",
                "retrieval_error_m",
                "base_cmd_vx",
                "base_cmd_vy",
                "base_cmd_wz",
                "ee_world_x",
                "ee_world_y",
                "ee_world_z",
            ]
        )
        for i in range(len(time_s)):
            writer.writerow(
                [
                    time_s[i],
                    base_speed_mps[i],
                    base_yaw_rate_rps[i],
                    ee_speed_mps[i],
                    gripper_close[i],
                    gripper_raw[i],
                    retrieval_error_m[i],
                    base_cmd_arr[i, 0],
                    base_cmd_arr[i, 1],
                    base_cmd_arr[i, 2],
                    ee_pos_world[i, 0],
                    ee_pos_world[i, 1],
                    ee_pos_world[i, 2],
                ]
            )

    np.savez(
        os.path.join(out_dir, "hl_motion_profile.npz"),
        time_s=time_s.astype(np.float32),
        base_planar_speed_mps=base_speed_mps.astype(np.float32),
        base_yaw_rate_rps=base_yaw_rate_rps.astype(np.float32),
        ee_planar_speed_mps=ee_speed_mps.astype(np.float32),
        gripper_close_01=gripper_close.astype(np.float32),
        gripper_raw_action=gripper_raw.astype(np.float32),
        retrieval_error_m=retrieval_error_m.astype(np.float32),
        base_command=base_cmd_arr.astype(np.float32),
        ee_position_world=ee_pos_world.astype(np.float32),
        simultaneous_motion_mask=simultaneous,
    )

    with open(
        os.path.join(out_dir, "hl_motion_profile_summary.json"), "w"
    ) as f:
        json.dump(summary, f, indent=2)

    make_motion_profile_plot(
        time_s=time_s,
        base_speed_mps=base_speed_mps,
        base_yaw_rate_rps=base_yaw_rate_rps,
        ee_speed_mps=ee_speed_mps,
        gripper_close=gripper_close,
        success=success,
        completion_time_s=completion_time_s,
        side_cm=float(args.cube_side_cm),
        mass_kg=float(cube_mass_kg),
    )

    print("=" * 88)
    print("B2WZ1 HIGH-LEVEL RETRIEVAL MOTION PROFILE")
    print("=" * 88)
    print(f"Seed:                      {args.seed}")
    print(f"Cube:                      {args.cube_side_cm:.1f} cm, {cube_mass_kg:.4f} kg")
    print(f"Success:                   {success}")
    if success:
        print(f"Completion time:           {completion_time_s:.3f} s")
    else:
        print(f"Completion time:           timeout ({episode_length_s:.2f} s horizon)")
    print(f"Peak base planar speed:    {np.max(base_speed_mps):.3f} m/s")
    print(f"Peak |base yaw rate|:      {np.max(np.abs(base_yaw_rate_rps)):.3f} rad/s")
    print(f"Peak EE planar speed:      {np.max(ee_speed_mps):.3f} m/s")
    print(f"Simultaneous motion time:  {simultaneous_time_s:.3f} s")
    print(f"Simultaneous motion frac.: {100.0*simultaneous_fraction:.1f}%")
    print("=" * 88)
    print(f"Saved: {os.path.join(out_dir, 'hl_motion_profile.png')}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
