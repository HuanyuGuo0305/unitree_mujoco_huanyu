#!/usr/bin/env python3
"""
Low-level WBC EE tracking-accuracy evaluation for B2W + Z1 in MuJoCo.

Evaluation protocol
-------------------
- Load the same WBC ONNX policy and frozen UAN actuator used by the normal
  B2WZ1 sim2sim deployment.
- Load the presampled 9-D PLB EE command table from cfg['ee_command_path'].
- Randomly choose N UNIQUE targets (default 50) with a fixed seed.
- Apply each target as a DIRECT sample-and-hold command (no interpolation).
- For each target, evaluate only the final steady-state window (default 0.5 s).

Outputs
-------
1) ee_position_tracking_3d.png/.pdf
   3-D desired-vs-achieved steady-state EE positions in PLB. Targets above
   the position-error P90 are excluded before reporting the final metric.
   The reported headline metric is the inlier-only position MAE.
2) ee_orientation_error_hist.png/.pdf
   Histogram of per-target steady-state orientation MAE values. Targets above
   the orientation-error P90 are excluded before reporting the final metric.
   The reported headline metric is the inlier-only orientation MAE.
3) ee_tracking_results.npz
4) metrics.json
5) selected_ee_commands.npy

Run from simulate_python/:

  python3 deploy_mujoco/wbc_ee_tracking_accuracy.py \
      configs/b2wz1_wbc_uan_plb.yaml \
      --num-targets 50 \
      --seed 0 \
      --out results/wbc_ee_tracking_accuracy

Relative paths inside the YAML are resolved from the unitree_mujoco project
root, matching the existing sim2sim script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import onnxruntime as ort
import torch
import yaml


# -----------------------------------------------------------------------------
# Project imports / repository layout
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parents[2]
PROJECT_ROOT = DEFAULT_PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.math import (  # noqa: E402
    euler_xyz_from_quat_wxyz,
    quat_apply_inverse_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_from_rotmat_wxyz,
    quat_from_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_rotate_inverse_numpy,
    quat_unique_wxyz,
)
from utilities.mujoco_helper import get_sensor_slice  # noqa: E402


MUJOCO_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper",
]

CTRL_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper",
]

LEG_JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
WHEEL_JOINT_NAMES = ["FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"]


def resolve_from_project(path: str | os.PathLike, project_root: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def keypoints_to_pose_plb(kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert [kp0, kp1, kp2] to PLB position + quaternion (wxyz).

    The command representation is the same as the existing sim2sim code:
      kp0 = EE origin
      kp1 = kp0 + R * [ee_kp_dx, 0, 0]
      kp2 = kp0 + R * [0, 0, ee_kp_dz]

    We recover x and z axes from the keypoints, orthogonalize them for
    numerical robustness, and form R=[x y z] with y = z x x.
    """
    kps = np.asarray(kps, dtype=np.float64).reshape(9)
    p = kps[0:3].copy()
    x = kps[3:6] - p
    z = kps[6:9] - p

    nx = np.linalg.norm(x)
    nz = np.linalg.norm(z)
    if nx < 1e-9 or nz < 1e-9:
        raise ValueError(f"Degenerate keypoint command: {kps}")

    z = z / nz
    x = x / nx
    # Remove any small non-orthogonal component introduced numerically.
    x = x - np.dot(x, z) * z
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        raise ValueError(f"Degenerate x/z axes in keypoint command: {kps}")
    x = x / nx

    y = np.cross(z, x)
    ny = np.linalg.norm(y)
    if ny < 1e-9:
        raise ValueError(f"Degenerate reconstructed y-axis: {kps}")
    y = y / ny
    # Recompute z to guarantee an orthonormal right-handed frame.
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)

    R = np.column_stack((x, y, z)).astype(np.float32)
    q = quat_from_rotmat_wxyz(R)
    q = quat_unique_wxyz(quat_normalize_wxyz(q.astype(np.float32)))
    return p.astype(np.float32), q.astype(np.float32)


def quat_geodesic_error_rad(q_des: np.ndarray, q_act: np.ndarray) -> float:
    q_des = quat_normalize_wxyz(np.asarray(q_des, dtype=np.float32))
    q_act = quat_normalize_wxyz(np.asarray(q_act, dtype=np.float32))
    dot = float(abs(np.dot(q_des, q_act)))
    dot = float(np.clip(dot, 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def percentile(x: np.ndarray, p: float) -> float:
    return float(np.percentile(np.asarray(x, dtype=np.float64), p))


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x))))


def sync_mujoco_arm_limits(
    m: mujoco.MjModel,
    final_limits: np.ndarray,
    sync_limits: bool,
) -> None:
    print("MuJoCo arm torque-limit audit:")
    for j, joint_name in enumerate(ARM_JOINT_NAMES):
        ctrl_i = CTRL_JOINT_NAMES.index(joint_name)
        final_lim = float(final_limits[j])

        if bool(m.actuator_ctrllimited[ctrl_i]):
            old = m.actuator_ctrlrange[ctrl_i].copy()
            if old[0] > -final_lim + 1e-6 or old[1] < final_lim - 1e-6:
                if not sync_limits:
                    raise RuntimeError(
                        f"{joint_name} ctrlrange {old} is narrower than +/-{final_lim} Nm"
                    )
                m.actuator_ctrlrange[ctrl_i, :] = (-final_lim, final_lim)

        if bool(m.actuator_forcelimited[ctrl_i]):
            old = m.actuator_forcerange[ctrl_i].copy()
            if old[0] > -final_lim + 1e-6 or old[1] < final_lim - 1e-6:
                if not sync_limits:
                    raise RuntimeError(
                        f"{joint_name} forcerange {old} is narrower than +/-{final_lim} Nm"
                    )
                m.actuator_forcerange[ctrl_i, :] = (-final_lim, final_lim)

        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if (
            joint_id >= 0
            and hasattr(m, "jnt_actfrclimited")
            and hasattr(m, "jnt_actfrcrange")
            and bool(m.jnt_actfrclimited[joint_id])
        ):
            old = m.jnt_actfrcrange[joint_id].copy()
            if old[0] > -final_lim + 1e-6 or old[1] < final_lim - 1e-6:
                if not sync_limits:
                    raise RuntimeError(
                        f"{joint_name} joint actfrcrange {old} is narrower than +/-{final_lim} Nm"
                    )
                m.jnt_actfrcrange[joint_id, :] = (-final_lim, final_lim)

    print("  PASS: MuJoCo will not clip inside the explicit UAN final envelope.")


class WBCUANTrackingEvaluator:
    def __init__(self, cfg: dict, project_root: Path):
        self.cfg = cfg
        self.project_root = project_root

        # Paths.
        self.policy_path = resolve_from_project(cfg["policy_path"], project_root)
        self.uan_model_path = resolve_from_project(cfg["uan_model_path"], project_root)
        self.xml_path = resolve_from_project(cfg["xml_path"], project_root)
        self.command_path = resolve_from_project(cfg["ee_command_path"], project_root)

        for name, p in (
            ("policy_path", self.policy_path),
            ("uan_model_path", self.uan_model_path),
            ("xml_path", self.xml_path),
            ("ee_command_path", self.command_path),
        ):
            if not p.is_file():
                raise FileNotFoundError(f"{name} not found: {p}")

        # Timing / dimensions.
        self.simulation_dt = float(cfg["simulation_dt"])
        self.control_decimation = int(cfg["control_decimation"])
        self.control_dt = self.simulation_dt * self.control_decimation
        if abs(self.simulation_dt - 0.004) > 1e-9 or self.control_decimation != 5:
            raise ValueError(
                "Expected the trained WBC-UAN timing: 250 Hz physics/UAN and 50 Hz WBC."
            )

        self.history_length = int(cfg["history_length"])
        self.obs_dim_per_step = int(cfg["obs_dim_per_step"])
        self.obs_dim = int(cfg["obs_dim"])
        self.action_dim = int(cfg["action_dim"])
        if self.obs_dim_per_step != 80 or self.obs_dim != 400 or self.action_dim != 22:
            raise ValueError(
                f"Unexpected WBC dimensions: step={self.obs_dim_per_step}, "
                f"stack={self.obs_dim}, action={self.action_dim}"
            )

        self.startup_hold_s = float(cfg.get("startup_hold_s", 1.0))
        self.startup_blend_s = float(cfg.get("startup_blend_s", 2.0))
        self.base_command = np.asarray(cfg["base_command"], dtype=np.float32)

        self.root_pos = np.asarray(cfg["root_pos"], dtype=np.float32)
        self.root_quat = np.asarray(cfg["root_quat_wxyz"], dtype=np.float32)
        self.default_joint_pos = np.asarray(cfg["default_joint_pos"], dtype=np.float32)
        self.kps = np.asarray(cfg["kps"], dtype=np.float32)
        self.kds = np.asarray(cfg["kds"], dtype=np.float32)
        self.leg_action_scale = float(cfg["leg_action_scale"])
        self.arm_action_scale = np.asarray(cfg["arm_action_scale"], dtype=np.float32)
        self.wheel_action_scale = float(cfg["wheel_action_scale"])
        self.leg_torque_limits = np.asarray(cfg["leg_torque_limits"], dtype=np.float32)
        self.gripper_torque_limit = float(cfg.get("gripper_torque_limit", 30.0))
        self.wheel_velocity_limits = np.asarray(cfg["wheel_velocity_limits"], dtype=np.float32)

        # UAN.
        self.uan_device = str(cfg.get("uan_device", "cpu"))
        if self.uan_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"UAN device '{self.uan_device}' requested but CUDA unavailable")
        self.uan_history_length = int(cfg.get("uan_history_length", 25))
        self.uan_pos_scale = float(cfg.get("uan_pos_scale", 1.0))
        self.uan_vel_scale = float(cfg.get("uan_vel_scale", 0.05))
        self.uan_action_scale = float(cfg.get("uan_action_scale", 0.20))
        self.uan_kp = np.asarray(cfg["uan_kp"], dtype=np.float32)
        self.uan_kd = np.asarray(cfg["uan_kd"], dtype=np.float32)
        self.uan_nominal_torque_limits = np.asarray(
            cfg["uan_nominal_torque_limits"], dtype=np.float32
        )
        self.uan_final_torque_limits = np.asarray(
            cfg["uan_final_torque_limits"], dtype=np.float32
        )
        if self.uan_history_length != 25:
            raise ValueError(f"Expected UAN history_length=25, got {self.uan_history_length}")

        self.ee_body_name = str(cfg.get("ee_body", "gripperStator"))
        self.ee_ground_z = float(cfg.get("ee_ground_z", 0.0))
        self.ee_kp_dx = float(cfg.get("ee_kp_dx", 0.30))
        self.ee_kp_dz = float(cfg.get("ee_kp_dz", 0.30))

        # Load WBC policy.
        self.sess = ort.InferenceSession(
            str(self.policy_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

        # Load deterministic frozen UAN wrapper.
        self.uan_model = torch.jit.load(
            str(self.uan_model_path), map_location=self.uan_device
        ).eval()
        for p in self.uan_model.parameters():
            p.requires_grad_(False)

        # MuJoCo.
        self.m = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt
        sync_mujoco_arm_limits(
            self.m,
            self.uan_final_torque_limits,
            bool(cfg.get("sync_mujoco_arm_limits_to_uan", True)),
        )

        self.ee_bid = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name
        )
        if self.ee_bid < 0:
            raise ValueError(f"EE body not found: {self.ee_body_name}")

        # Joint mappings copied from the working sim2sim implementation.
        self.policy_joint_pos_names = LEG_JOINT_NAMES + ARM_JOINT_NAMES
        self.policy_joint_vel_names = LEG_JOINT_NAMES + ARM_JOINT_NAMES + WHEEL_JOINT_NAMES
        self.policy_joint_pos_mujoco_indices = [
            MUJOCO_JOINT_NAMES.index(n) for n in self.policy_joint_pos_names
        ]
        self.policy_joint_vel_mujoco_indices = [
            MUJOCO_JOINT_NAMES.index(n) for n in self.policy_joint_vel_names
        ]
        self.leg_mujoco_indices = [MUJOCO_JOINT_NAMES.index(n) for n in LEG_JOINT_NAMES]
        self.arm_mujoco_indices = [MUJOCO_JOINT_NAMES.index(n) for n in ARM_JOINT_NAMES]
        self.wheel_mujoco_indices = [MUJOCO_JOINT_NAMES.index(n) for n in WHEEL_JOINT_NAMES]
        self.gripper_mujoco_index = MUJOCO_JOINT_NAMES.index("jointGripper")

        self.leg_action_indices = list(range(0, 12))
        self.arm_action_indices = list(range(12, 18))
        self.wheel_action_indices = list(range(18, 22))

        control_source_joint_names = (
            LEG_JOINT_NAMES + ARM_JOINT_NAMES + WHEEL_JOINT_NAMES + ["jointGripper"]
        )
        ctrl_source_index_by_name = {
            name: i for i, name in enumerate(control_source_joint_names)
        }
        self.ctrl_src_indices_or_none = [
            ctrl_source_index_by_name[name] if name in ctrl_source_index_by_name else None
            for name in CTRL_JOINT_NAMES
        ]

        self.default_joint_pos_policy_pos = self.default_joint_pos[
            self.policy_joint_pos_mujoco_indices
        ]
        self.default_leg_pos = self.default_joint_pos[self.leg_mujoco_indices]
        self.default_arm_pos = self.default_joint_pos[self.arm_mujoco_indices]
        self.default_gripper_pos = float(
            self.default_joint_pos[self.gripper_mujoco_index]
        )

        # Runtime state.
        self.uan_history = np.zeros(
            (6, self.uan_history_length, 2), dtype=np.float32
        )
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.leg_target = self.default_leg_pos.copy()
        self.arm_target = self.default_arm_pos.copy()
        self.wheel_cmd = np.zeros(4, dtype=np.float32)
        self.gripper_target = self.default_gripper_pos
        self.counter = 0
        self.sim_time = 0.0

        self._init_sim_state()
        self._init_wbc_history()

    def _init_sim_state(self) -> None:
        self.d.qpos[:] = 0.0
        self.d.qvel[:] = 0.0
        self.d.ctrl[:] = 0.0
        self.d.qpos[0:3] = self.root_pos
        self.d.qpos[3:7] = self.root_quat
        self.d.qpos[7:7 + len(MUJOCO_JOINT_NAMES)] = self.default_joint_pos
        mujoco.mj_forward(self.m, self.d)
        self.uan_history.fill(0.0)

    def get_projected_level_base_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        base_pos_w = self.d.qpos[0:3].copy().astype(np.float32)
        base_quat_w = quat_unique_wxyz(
            self.d.qpos[3:7].copy().astype(np.float32)
        )
        _, _, yaw = euler_xyz_from_quat_wxyz(base_quat_w)
        plb_quat_w = quat_from_yaw_wxyz(yaw)
        plb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(plb_quat_w))
        plb_pos_w = base_pos_w.copy()
        plb_pos_w[2] = self.ee_ground_z
        return plb_pos_w, plb_quat_w

    def compute_ee_pose_plb(self) -> tuple[np.ndarray, np.ndarray]:
        plb_pos_w, plb_quat_w = self.get_projected_level_base_pose_world()
        ee_pos_w = self.d.xpos[self.ee_bid].copy().astype(np.float32)
        ee_rot_w = self.d.xmat[self.ee_bid].reshape(3, 3).copy().astype(np.float32)
        ee_quat_w = quat_from_rotmat_wxyz(ee_rot_w)

        ee_pos_plb = quat_apply_inverse_wxyz(
            plb_quat_w, ee_pos_w - plb_pos_w
        ).astype(np.float32)
        ee_quat_plb = quat_mul_wxyz(
            quat_conjugate_wxyz(plb_quat_w), ee_quat_w
        )
        ee_quat_plb = quat_unique_wxyz(
            quat_normalize_wxyz(ee_quat_plb.astype(np.float32))
        )
        return ee_pos_plb, ee_quat_plb.astype(np.float32)

    def compute_ee_keypoints_plb(self) -> np.ndarray:
        p, q = self.compute_ee_pose_plb()
        off_x = np.array([self.ee_kp_dx, 0.0, 0.0], dtype=np.float32)
        off_z = np.array([0.0, 0.0, self.ee_kp_dz], dtype=np.float32)
        kp0 = p
        kp1 = p + quat_apply_wxyz(q, off_x)
        kp2 = p + quat_apply_wxyz(q, off_z)
        return np.concatenate([kp0, kp1, kp2]).astype(np.float32)

    def compute_uan_arm_torque(
        self, q_des: np.ndarray, q: np.ndarray, qd: np.ndarray
    ) -> np.ndarray:
        q_des = np.asarray(q_des, dtype=np.float32).reshape(6)
        q = np.asarray(q, dtype=np.float32).reshape(6)
        qd = np.asarray(qd, dtype=np.float32).reshape(6)

        pos_err = (q_des - q) * self.uan_pos_scale
        vel = qd * self.uan_vel_scale
        self.uan_history[:, :-1, :] = self.uan_history[:, 1:, :]
        self.uan_history[:, -1, 0] = pos_err
        self.uan_history[:, -1, 1] = vel

        obs_np = np.ascontiguousarray(
            self.uan_history.reshape(1, -1), dtype=np.float32
        )
        obs_t = torch.from_numpy(obs_np).to(self.uan_device)
        with torch.inference_mode():
            out = self.uan_model(obs_t)
            if isinstance(out, (tuple, list)):
                out = out[0]
        action = out.detach().cpu().numpy().reshape(6).astype(np.float32)

        if not np.all(np.isfinite(action)):
            raise FloatingPointError(f"Non-finite UAN action: {action}")
        if np.max(np.abs(action)) > 1.001:
            raise RuntimeError(
                f"UAN action outside expected [-1,1] range: {action}"
            )

        tau_pd = self.uan_kp * (q_des - q) - self.uan_kd * qd
        tau_nom = np.clip(
            tau_pd,
            -self.uan_nominal_torque_limits,
            self.uan_nominal_torque_limits,
        )
        delta_tau = action * self.uan_action_scale * self.uan_kp
        tau = np.clip(
            tau_nom + delta_tau,
            -self.uan_final_torque_limits,
            self.uan_final_torque_limits,
        )
        return tau.astype(np.float32)

    def build_obs_step(self, ee_cmd_plb: np.ndarray) -> np.ndarray:
        qpos_mujoco = self.d.qpos[
            7:7 + len(MUJOCO_JOINT_NAMES)
        ].copy().astype(np.float32)
        qvel_mujoco = self.d.qvel[
            6:6 + len(MUJOCO_JOINT_NAMES)
        ].copy().astype(np.float32)

        base_ang_vel_b = get_sensor_slice(
            self.m, self.d, "imu_gyro"
        ).astype(np.float32)
        base_quat_w = quat_unique_wxyz(
            self.d.qpos[3:7].copy().astype(np.float32)
        )
        projected_gravity_b = quat_rotate_inverse_numpy(
            base_quat_w, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        ).astype(np.float32)

        joint_pos_policy = qpos_mujoco[self.policy_joint_pos_mujoco_indices]
        joint_pos_rel = joint_pos_policy - self.default_joint_pos_policy_pos
        joint_pos_leg_rel = joint_pos_rel[:12]
        joint_pos_arm_rel = joint_pos_rel[12:18]

        joint_vel_policy = qvel_mujoco[self.policy_joint_vel_mujoco_indices]
        joint_vel_leg = joint_vel_policy[:12]
        joint_vel_arm = joint_vel_policy[12:18]
        joint_vel_wheel = joint_vel_policy[18:22]

        obs = np.concatenate(
            [
                base_ang_vel_b,
                projected_gravity_b,
                self.base_command,
                ee_cmd_plb,
                joint_pos_leg_rel,
                joint_pos_arm_rel,
                joint_vel_leg,
                joint_vel_arm,
                joint_vel_wheel,
                self.last_action,
            ],
            dtype=np.float32,
        )
        if obs.shape[0] != self.obs_dim_per_step:
            raise RuntimeError(
                f"Obs dim mismatch: {obs.shape[0]} vs {self.obs_dim_per_step}"
            )
        return obs

    def _split_obs_step(self, obs: np.ndarray) -> tuple[np.ndarray, ...]:
        i = 0
        blocks = []
        for n in (3, 3, 3, 9, 12, 6, 12, 6, 4, 22):
            blocks.append(obs[i:i+n].copy())
            i += n
        return tuple(blocks)

    def _init_wbc_history(self) -> None:
        ee_cmd_init = self.compute_ee_keypoints_plb()
        obs0 = self.build_obs_step(ee_cmd_init)
        blocks = self._split_obs_step(obs0)
        self.obs_hists = [deque(maxlen=self.history_length) for _ in blocks]
        for hist, block in zip(self.obs_hists, blocks):
            for _ in range(self.history_length):
                hist.append(block.copy())

    def _update_wbc_policy(self, ee_cmd_plb: np.ndarray, blend: float) -> None:
        obs_step = self.build_obs_step(ee_cmd_plb)
        blocks = self._split_obs_step(obs_step)
        for hist, block in zip(self.obs_hists, blocks):
            hist.append(block.copy())

        obs_stack = np.concatenate(
            [np.asarray(hist).reshape(-1) for hist in self.obs_hists],
            dtype=np.float32,
        )
        if obs_stack.shape[0] != self.obs_dim:
            raise RuntimeError(
                f"Stacked obs dim mismatch: {obs_stack.shape[0]} vs {self.obs_dim}"
            )

        action = self.sess.run(
            [self.output_name], {self.input_name: obs_stack[None, :]}
        )[0][0].astype(np.float32)
        if action.shape != (self.action_dim,):
            raise RuntimeError(f"Unexpected WBC action shape: {action.shape}")
        self.last_action[:] = action

        leg_act = action[self.leg_action_indices]
        arm_act = action[self.arm_action_indices]
        wheel_act = action[self.wheel_action_indices]

        self.leg_target = (
            self.default_leg_pos + blend * (self.leg_action_scale * leg_act)
        )
        self.arm_target = (
            self.default_arm_pos + blend * (self.arm_action_scale * arm_act)
        )
        self.wheel_cmd = blend * (self.wheel_action_scale * wheel_act)

    def _one_physics_step(self, ee_cmd_plb: np.ndarray) -> None:
        # Same startup blending semantics as the original sim2sim script.
        if self.sim_time < self.startup_hold_s:
            blend = 0.0
        elif self.sim_time < self.startup_hold_s + self.startup_blend_s:
            blend = (self.sim_time - self.startup_hold_s) / self.startup_blend_s
        else:
            blend = 1.0
        blend = float(np.clip(blend, 0.0, 1.0))

        qpos_mujoco = self.d.qpos[
            7:7 + len(MUJOCO_JOINT_NAMES)
        ].copy().astype(np.float32)
        qvel_mujoco = self.d.qvel[
            6:6 + len(MUJOCO_JOINT_NAMES)
        ].copy().astype(np.float32)

        leg_pos = qpos_mujoco[self.leg_mujoco_indices]
        leg_vel = qvel_mujoco[self.leg_mujoco_indices]
        arm_pos = qpos_mujoco[self.arm_mujoco_indices]
        arm_vel = qvel_mujoco[self.arm_mujoco_indices]
        gripper_pos = float(qpos_mujoco[self.gripper_mujoco_index])
        gripper_vel = float(qvel_mujoco[self.gripper_mujoco_index])

        # Intentionally mirrors the current sim2sim implementation.
        leg_tau = (
            self.kps[self.leg_mujoco_indices] * (self.leg_target - leg_pos)
            - self.kds[self.leg_mujoco_indices] * leg_vel
        )
        leg_tau = np.clip(
            leg_tau, -self.leg_torque_limits, self.leg_torque_limits
        ).astype(np.float32)

        arm_tau = self.compute_uan_arm_torque(
            self.arm_target, arm_pos, arm_vel
        )

        gripper_tau = (
            self.kps[self.gripper_mujoco_index]
            * (self.gripper_target - gripper_pos)
            - self.kds[self.gripper_mujoco_index] * gripper_vel
        )
        gripper_tau = float(
            np.clip(
                gripper_tau,
                -self.gripper_torque_limit,
                self.gripper_torque_limit,
            )
        )
        wheel_ctrl = np.clip(
            self.wheel_cmd,
            -self.wheel_velocity_limits,
            self.wheel_velocity_limits,
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

        self.d.ctrl[:] = 0.0
        for ctrl_i, src_i in enumerate(self.ctrl_src_indices_or_none):
            if src_i is not None:
                self.d.ctrl[ctrl_i] = ctrl_source[src_i]

        mujoco.mj_step(self.m, self.d)
        self.sim_time += self.simulation_dt

        if self.counter % self.control_decimation == 0:
            self._update_wbc_policy(ee_cmd_plb, blend)

        self.counter += 1

    def warmup(self) -> None:
        ee_cmd = self.compute_ee_keypoints_plb()
        warmup_s = self.startup_hold_s + self.startup_blend_s
        n = int(round(warmup_s / self.simulation_dt))
        print(f"Warmup: {warmup_s:.2f} s ({n} physics steps)")
        for _ in range(n):
            self._one_physics_step(ee_cmd)

    def evaluate_target(
        self,
        target_kps: np.ndarray,
        hold_s: float,
        eval_window_s: float,
    ) -> dict:
        target_kps = np.asarray(target_kps, dtype=np.float32).reshape(9)
        target_pos, target_quat = keypoints_to_pose_plb(target_kps)

        hold_steps = int(round(hold_s / self.simulation_dt))
        eval_steps = int(round(eval_window_s / self.simulation_dt))
        if hold_steps < 1 or eval_steps < 1 or eval_steps > hold_steps:
            raise ValueError(
                f"Invalid hold/eval windows: hold_s={hold_s}, eval_window_s={eval_window_s}"
            )

        pos_samples = []
        quat_samples = []
        pos_err_samples = []
        ori_err_samples = []

        eval_start = hold_steps - eval_steps
        for k in range(hold_steps):
            self._one_physics_step(target_kps)
            if k >= eval_start:
                p, q = self.compute_ee_pose_plb()
                pos_samples.append(p)
                quat_samples.append(q)
                pos_err_samples.append(float(np.linalg.norm(p - target_pos)))
                ori_err_samples.append(quat_geodesic_error_rad(target_quat, q))

        pos_samples = np.asarray(pos_samples, dtype=np.float32)
        quat_samples = np.asarray(quat_samples, dtype=np.float32)
        pos_err_samples = np.asarray(pos_err_samples, dtype=np.float32)
        ori_err_samples = np.asarray(ori_err_samples, dtype=np.float32)

        achieved_pos_mean = np.mean(pos_samples, axis=0)

        # Per-target steady-state MAE over the evaluation window.
        # pos_err_samples and ori_err_samples are already non-negative errors.
        position_mae_m = float(np.mean(pos_err_samples))
        orientation_mae_rad = float(np.mean(ori_err_samples))

        return {
            "target_pos": target_pos,
            "target_quat": target_quat,
            "achieved_pos_mean": achieved_pos_mean.astype(np.float32),
            "position_mae_m": position_mae_m,
            "orientation_mae_rad": orientation_mae_rad,
            "eval_pos_samples": pos_samples,
            "eval_quat_samples": quat_samples,
        }


def make_position_plot(
    out_dir: Path,
    desired: np.ndarray,
    achieved: np.ndarray,
    pos_mae_m: np.ndarray,
    overall_position_mae_m: float,
) -> None:
    """3D scatter using only inliers after excluding targets above position-error P90."""
    p90 = float(np.percentile(pos_mae_m, 90))
    inlier_mask = pos_mae_m <= p90

    desired_in = desired[inlier_mask]
    achieved_in = achieved[inlier_mask]

    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        desired_in[:, 0], desired_in[:, 1], desired_in[:, 2],
        marker="x", s=42, label="Desired"
    )
    ax.scatter(
        achieved_in[:, 0], achieved_in[:, 1], achieved_in[:, 2],
        marker="o", s=30, label="Achieved"
    )

    xs, ys, zs = [], [], []
    for p0, p1 in zip(desired_in, achieved_in):
        xs.extend([p0[0], p1[0], np.nan])
        ys.extend([p0[1], p1[1], np.nan])
        zs.extend([p0[2], p1[2], np.nan])
    ax.plot(xs, ys, zs, linewidth=0.7, alpha=0.35, label="Tracking offset")

    ax.set_xlabel("EE x in PLB [m]")
    ax.set_ylabel("EE y in PLB [m]")
    ax.set_zlabel("EE z in PLB [m]")
    ax.set_title(
        "Desired vs. Achieved EE Positions\n"
        f"Inlier-only position MAE = {overall_position_mae_m*100.0:.2f} cm"
    )
    ax.legend(frameon=False)

    if len(desired_in) > 0:
        spans = np.ptp(np.vstack([desired_in, achieved_in]), axis=0)
        spans = np.maximum(spans, 1e-3)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(spans)

    fig.tight_layout()
    fig.savefig(out_dir / "ee_position_tracking_3d.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "ee_position_tracking_3d.pdf", bbox_inches="tight")
    plt.close(fig)


def make_orientation_histogram(
    out_dir: Path,
    orientation_mae_deg: np.ndarray,
    overall_orientation_mae_deg: float,
    bin_width_deg: float = 2.5,
) -> None:
    """Histogram of inlier orientation MAE values, normalized to proportions."""
    p90 = float(np.percentile(orientation_mae_deg, 90))
    errors_in = orientation_mae_deg[orientation_mae_deg <= p90]

    if len(errors_in) == 0:
        raise RuntimeError("No inlier orientation errors remain after P90 exclusion")

    max_edge = float(np.max(errors_in))
    # Ensure at least one non-empty bin.
    upper = max(bin_width_deg, np.ceil(max_edge / bin_width_deg) * bin_width_deg)
    bins = np.arange(0.0, upper + bin_width_deg, bin_width_deg)
    counts, edges = np.histogram(errors_in, bins=bins)
    proportions = counts.astype(np.float64) / float(len(errors_in))
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(
        centers,
        proportions,
        width=bin_width_deg * 0.9,
        align="center",
        label="Inlier targets",
    )
    ax.axvline(
        overall_orientation_mae_deg,
        linestyle="--",
        linewidth=1.5,
        label=f"Inlier-only MAE = {overall_orientation_mae_deg:.2f} deg",
    )
    ax.set_xlabel(f"Steady-state orientation MAE [deg] (bin width = {bin_width_deg:.1f}°)")
    ax.set_ylabel("Proportion of targets")
    ax.set_title("EE Orientation Tracking Accuracy")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "ee_orientation_error_hist.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "ee_orientation_error_hist.pdf", bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="Existing WBC-UAN sim2sim YAML")
    parser.add_argument("--num-targets", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hold-s",
        type=float,
        default=None,
        help="Direct hold time per target; default: cfg ee_cycle_duration_s",
    )
    parser.add_argument(
        "--eval-window-s",
        type=float,
        default=0.5,
        help="Steady-state window at the end of each hold",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/wbc_ee_tracking_accuracy",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Default: two levels above this script (unitree_mujoco root)",
    )
    args = parser.parse_args()

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root is not None
        else DEFAULT_PROJECT_ROOT
    )
    # Ensure project_root wins if explicitly overridden.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    yaml_path = Path(args.yaml_path).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    command_path = resolve_from_project(cfg["ee_command_path"], project_root)
    command_table = np.load(command_path).astype(np.float32)
    if command_table.ndim != 2 or command_table.shape[1] != 9:
        raise ValueError(
            f"Expected EE command table shape (N,9), got {command_table.shape}"
        )
    if args.num_targets < 1:
        raise ValueError("--num-targets must be >= 1")
    if args.num_targets > command_table.shape[0]:
        raise ValueError(
            f"Requested {args.num_targets} unique targets but table has only "
            f"{command_table.shape[0]} rows"
        )

    hold_s = (
        float(args.hold_s)
        if args.hold_s is not None
        else float(cfg.get("ee_cycle_duration_s", 8.0))
    )
    eval_window_s = float(args.eval_window_s)
    if eval_window_s <= 0.0 or eval_window_s > hold_s:
        raise ValueError("Need 0 < eval_window_s <= hold_s")

    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(
        command_table.shape[0], size=args.num_targets, replace=False
    )
    selected_commands = command_table[selected_indices].copy()
    np.save(out_dir / "selected_ee_commands.npy", selected_commands)

    print("=" * 80)
    print("Low-level WBC EE tracking-accuracy evaluation")
    print("=" * 80)
    print(f"YAML:              {yaml_path}")
    print(f"Command table:     {command_path}")
    print(f"Targets:           {args.num_targets} unique samples")
    print(f"Random seed:       {args.seed}")
    print(f"Hold per target:   {hold_s:.2f} s")
    print(f"Eval window:       final {eval_window_s:.2f} s")
    print(f"Base command:      {cfg['base_command']}")
    print(f"Output:            {out_dir}")
    print("=" * 80)

    evaluator = WBCUANTrackingEvaluator(cfg, project_root)
    evaluator.warmup()

    desired_positions = []
    desired_quats = []
    achieved_positions = []
    pos_mae_m = []
    ori_mae_rad = []

    for i, (table_idx, target) in enumerate(
        zip(selected_indices, selected_commands), start=1
    ):
        result = evaluator.evaluate_target(
            target_kps=target,
            hold_s=hold_s,
            eval_window_s=eval_window_s,
        )
        desired_positions.append(result["target_pos"])
        desired_quats.append(result["target_quat"])
        achieved_positions.append(result["achieved_pos_mean"])
        pos_mae_m.append(result["position_mae_m"])
        ori_mae_rad.append(result["orientation_mae_rad"])

        print(
            f"[{i:02d}/{args.num_targets}] evaluated table_idx={int(table_idx):5d}"
        )

    desired_positions = np.asarray(desired_positions, dtype=np.float32)
    desired_quats = np.asarray(desired_quats, dtype=np.float32)
    achieved_positions = np.asarray(achieved_positions, dtype=np.float32)
    pos_mae_m = np.asarray(pos_mae_m, dtype=np.float32)
    ori_mae_rad = np.asarray(ori_mae_rad, dtype=np.float32)
    ori_mae_deg = np.degrees(ori_mae_rad).astype(np.float32)

    # Exclude targets above P90 before computing the final headline metrics.
    position_p90_m = float(np.percentile(pos_mae_m, 90))
    orientation_p90_deg = float(np.percentile(ori_mae_deg, 90))

    position_inlier_mask = pos_mae_m <= position_p90_m
    orientation_inlier_mask = ori_mae_deg <= orientation_p90_deg

    overall_position_mae_m = float(np.mean(pos_mae_m[position_inlier_mask]))
    overall_orientation_mae_deg = float(np.mean(ori_mae_deg[orientation_inlier_mask]))
    overall_orientation_mae_rad = float(np.radians(overall_orientation_mae_deg))

    metrics = {
        "position_tracking_mae_cm": overall_position_mae_m * 100.0,
        "orientation_tracking_mae_deg": overall_orientation_mae_deg,
    }

    print("\nSummary")
    print("-" * 80)
    print(f"Position tracking MAE    : {metrics['position_tracking_mae_cm']:.2f} cm")
    print(f"Orientation tracking MAE : {metrics['orientation_tracking_mae_deg']:.2f} deg")
    print("-" * 80)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        out_dir / "ee_tracking_results.npz",
        selected_indices=selected_indices.astype(np.int64),
        selected_commands=selected_commands,
        desired_positions=desired_positions,
        desired_quaternions_wxyz=desired_quats,
        achieved_positions_mean=achieved_positions,
        position_mae_m=pos_mae_m,
        orientation_mae_rad=ori_mae_rad,
        orientation_mae_deg=ori_mae_deg,
        position_p90_m=np.float32(position_p90_m),
        orientation_p90_deg=np.float32(orientation_p90_deg),
        position_inlier_mask=position_inlier_mask,
        orientation_inlier_mask=orientation_inlier_mask,
    )

    make_position_plot(
        out_dir,
        desired_positions,
        achieved_positions,
        pos_mae_m,
        overall_position_mae_m,
    )
    make_orientation_histogram(
        out_dir,
        ori_mae_deg,
        overall_orientation_mae_deg,
        bin_width_deg=2.5,
    )

    print(f"\nSaved to {out_dir}")
    print("  ee_position_tracking_3d.png / .pdf")
    print("  ee_orientation_error_hist.png / .pdf")
    print("  ee_tracking_results.npz")
    print("  selected_ee_commands.npy")
    print("  metrics.json")


if __name__ == "__main__":
    main()
