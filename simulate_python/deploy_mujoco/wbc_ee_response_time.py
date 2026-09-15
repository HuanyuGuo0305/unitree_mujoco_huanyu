#!/usr/bin/env python3
"""
Low-level WBC EE response-time evaluation for B2W + Z1 in MuJoCo.

Goal
----
Quantify how quickly the low-level WBC reacts to abrupt EE pose-command
switches. This evaluation intentionally matches the training/deployment
interface: commands are sampled from the existing presampled 9-D PLB command
set and switched directly, with NO interpolation.

Protocol
--------
1) Load the same WBC ONNX policy + frozen UAN actuator used by sim2sim.
2) Randomly choose N+1 UNIQUE commands from cfg["ee_command_path"].
3) Use the first command only to precondition/settle the robot.
4) Execute N measured direct target switches (default: 50).
5) Record EE position/orientation error at 250 Hz after every switch.
6) Estimate the final steady-state error from the last 0.5 s.
7) Normalize the reducible transient error:

       e_norm(t) = (e(t) - e_inf) / (e(0) - e_inf)

8) T90 is the first time e_norm <= 0.1 and remains <= 0.1 continuously
   for a short dwell window (default: 0.1 s).
9) Trivial command changes are excluded separately for position/orientation.
10) Extreme response-time outliers are removed using a P90 cutoff on finite
    T90 values. Headline response times are the mean T90 of the remaining set.

Outputs
-------
- ee_response_time.png / .pdf
  Two-panel normalized transient response plot (position + orientation).
  Curves show the median response with a 10-90 percentile band after filtering.
- ee_response_results.npz
  Full per-transition traces, T90 values, masks, selected commands.
- selected_ee_response_commands.npy
  The exact N+1 presampled commands used, for reproducibility / hardware replay.
- metrics.json
  Headline mean position/orientation T90 plus diagnostic counts.

Run from simulate_python/:

  python3 deploy_mujoco/wbc_ee_response_time.py \
      configs/b2wz1_wbc_uan_plb.yaml \
      --num-transitions 50 \
      --seed 0 \
      --hold-s 3.0 \
      --out results/wbc_ee_response_time

Relative YAML paths are resolved from the unitree_mujoco project root.
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


class WBCUANResponseEvaluator:
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


    def hold_command(self, target_kps: np.ndarray, duration_s: float) -> None:
        """Apply one direct EE command for a fixed duration without logging."""
        target_kps = np.asarray(target_kps, dtype=np.float32).reshape(9)
        n_steps = int(round(float(duration_s) / self.simulation_dt))
        if n_steps < 1:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        for _ in range(n_steps):
            self._one_physics_step(target_kps)

    def evaluate_transition(
        self,
        target_kps: np.ndarray,
        hold_s: float,
        steady_window_s: float,
        t90_threshold: float,
        t90_dwell_s: float,
    ) -> dict:
        """Apply an abrupt direct target switch and log the full transient.

        The first sample is taken immediately before the first physics step
        under the new target and therefore corresponds to t=0.
        """
        target_kps = np.asarray(target_kps, dtype=np.float32).reshape(9)
        target_pos, target_quat = keypoints_to_pose_plb(target_kps)

        hold_steps = int(round(float(hold_s) / self.simulation_dt))
        steady_steps = int(round(float(steady_window_s) / self.simulation_dt))
        dwell_steps = max(1, int(round(float(t90_dwell_s) / self.simulation_dt)))

        if hold_steps < 2:
            raise ValueError("hold_s is too short")
        if steady_steps < 1 or steady_steps >= hold_steps:
            raise ValueError(
                "Need 0 < steady_window_s < hold_s; got "
                f"{steady_window_s} and {hold_s}"
            )
        if not (0.0 < t90_threshold < 1.0):
            raise ValueError(f"t90_threshold must be in (0,1), got {t90_threshold}")

        n_samples = hold_steps + 1
        time_s = np.arange(n_samples, dtype=np.float64) * self.simulation_dt
        pos_error_m = np.zeros(n_samples, dtype=np.float64)
        ori_error_rad = np.zeros(n_samples, dtype=np.float64)

        # t = 0: state immediately before the first physics step with new command.
        p, q = self.compute_ee_pose_plb()
        pos_error_m[0] = float(np.linalg.norm(p - target_pos))
        ori_error_rad[0] = quat_geodesic_error_rad(target_quat, q)

        for k in range(hold_steps):
            self._one_physics_step(target_kps)
            p, q = self.compute_ee_pose_plb()
            pos_error_m[k + 1] = float(np.linalg.norm(p - target_pos))
            ori_error_rad[k + 1] = quat_geodesic_error_rad(target_quat, q)

        # Final steady-state error is estimated from the tail of the hold.
        pos_inf_m = float(np.mean(pos_error_m[-steady_steps:]))
        ori_inf_rad = float(np.mean(ori_error_rad[-steady_steps:]))

        pos_reducible_m = float(pos_error_m[0] - pos_inf_m)
        ori_reducible_rad = float(ori_error_rad[0] - ori_inf_rad)

        def normalize(error: np.ndarray, e_inf: float, reducible: float) -> np.ndarray:
            if reducible <= 1e-9:
                return np.full_like(error, np.nan, dtype=np.float64)
            return ((error - e_inf) / reducible).astype(np.float64)

        pos_norm = normalize(pos_error_m, pos_inf_m, pos_reducible_m)
        ori_norm = normalize(ori_error_rad, ori_inf_rad, ori_reducible_rad)

        def sustained_t90(norm_error: np.ndarray) -> float:
            if not np.all(np.isfinite(norm_error)):
                return float("nan")
            last_start = len(norm_error) - dwell_steps
            for i in range(max(0, last_start + 1)):
                window = norm_error[i:i + dwell_steps]
                if np.all(window <= t90_threshold):
                    return float(time_s[i])
            return float("nan")

        return {
            "time_s": time_s,
            "target_pos": target_pos.astype(np.float32),
            "target_quat": target_quat.astype(np.float32),
            "position_error_m": pos_error_m.astype(np.float32),
            "orientation_error_rad": ori_error_rad.astype(np.float32),
            "position_normalized_error": pos_norm.astype(np.float32),
            "orientation_normalized_error": ori_norm.astype(np.float32),
            "position_initial_error_m": float(pos_error_m[0]),
            "orientation_initial_error_rad": float(ori_error_rad[0]),
            "position_steady_error_m": pos_inf_m,
            "orientation_steady_error_rad": ori_inf_rad,
            "position_reducible_error_m": pos_reducible_m,
            "orientation_reducible_error_rad": ori_reducible_rad,
            "position_t90_s": sustained_t90(pos_norm),
            "orientation_t90_s": sustained_t90(ori_norm),
        }



def compute_t90_inlier_mask(
    t90_s: np.ndarray,
    eligible_mask: np.ndarray,
    outlier_percentile: float,
) -> tuple[np.ndarray, float, int]:
    """Return finite eligible T90 values below the chosen percentile cutoff.

    Non-finite T90 values (threshold not reached within the hold) are not
    silently counted as normal samples; they are returned via the diagnostic
    count so the user can inspect whether failures are becoming common.
    """
    t90_s = np.asarray(t90_s, dtype=np.float64)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)

    finite_eligible = eligible_mask & np.isfinite(t90_s)
    num_not_reached = int(np.sum(eligible_mask & ~np.isfinite(t90_s)))

    vals = t90_s[finite_eligible]
    if vals.size == 0:
        raise RuntimeError("No finite eligible T90 values are available.")

    cutoff = float(np.percentile(vals, outlier_percentile))
    inlier_mask = finite_eligible & (t90_s <= cutoff)
    return inlier_mask, cutoff, num_not_reached


def aggregate_normalized_response(
    traces: np.ndarray,
    inlier_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median and 10/90 percentile envelopes across selected transitions."""
    selected = np.asarray(traces, dtype=np.float64)[np.asarray(inlier_mask, dtype=bool)]
    if selected.shape[0] == 0:
        raise RuntimeError("No response traces remain after filtering.")
    median = np.nanmedian(selected, axis=0)
    low = np.nanpercentile(selected, 10, axis=0)
    high = np.nanpercentile(selected, 90, axis=0)
    return median, low, high


def make_response_plot(
    out_dir: Path,
    time_s: np.ndarray,
    pos_norm: np.ndarray,
    ori_norm: np.ndarray,
    pos_inlier_mask: np.ndarray,
    ori_inlier_mask: np.ndarray,
    mean_pos_t90_s: float,
    mean_ori_t90_s: float,
    t90_threshold: float,
    plot_window_s: float,
) -> None:
    """Create a cleaner slide-ready normalized transient-response figure."""
    pos_med, pos_lo, pos_hi = aggregate_normalized_response(
        pos_norm, pos_inlier_mask
    )
    ori_med, ori_lo, ori_hi = aggregate_normalized_response(
        ori_norm, ori_inlier_mask
    )

    plot_mask = time_s <= float(plot_window_s)
    t = time_s[plot_mask]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3), sharey=True)

    # ----- Left: position -----
    ax = axes[0]
    band_pos = ax.fill_between(
        t, pos_lo[plot_mask], pos_hi[plot_mask], alpha=0.18
    )
    line_pos, = ax.plot(t, pos_med[plot_mask], linewidth=2.0)
    thr_pos = ax.axhline(
        t90_threshold, linestyle="--", linewidth=1.2
    )
    ax.axvline(
        mean_pos_t90_s, linestyle=":", linewidth=1.5
    )
    ax.text(
        mean_pos_t90_s + 0.02,
        0.93,
        rf"$T_{{90}}^{{pos}}={mean_pos_t90_s:.2f}\,$s",
        fontsize=10,
        ha="left",
        va="top",
    )

    ax.set_title("EE Position Response")
    ax.set_xlabel("Time after target switch [s]")
    ax.set_ylabel("Normalized tracking error")
    ax.grid(True, alpha=0.25)

    # ----- Right: orientation -----
    ax = axes[1]
    band_ori = ax.fill_between(
        t, ori_lo[plot_mask], ori_hi[plot_mask], alpha=0.18
    )
    line_ori, = ax.plot(t, ori_med[plot_mask], linewidth=2.0)
    thr_ori = ax.axhline(
        t90_threshold, linestyle="--", linewidth=1.2
    )
    ax.axvline(
        mean_ori_t90_s, linestyle=":", linewidth=1.5
    )
    ax.text(
        mean_ori_t90_s + 0.02,
        0.93,
        rf"$T_{{90}}^{{ori}}={mean_ori_t90_s:.2f}\,$s",
        fontsize=10,
        ha="left",
        va="top",
    )

    ax.set_title("EE Orientation Response")
    ax.set_xlabel("Time after target switch [s]")
    ax.grid(True, alpha=0.25)

    # Keep the normalized response visually focused while still allowing
    # modest overshoot below zero / initial transient above one.
    axes[0].set_ylim(-0.10, 1.15)

    # One shared legend for the entire figure.
    fig.legend(
        [line_pos, band_pos, thr_pos],
        ["Median response", "10-90 percentile", "90% error reduction"],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(out_dir / "ee_response_time.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "ee_response_time.pdf", bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="Existing WBC-UAN sim2sim YAML")
    parser.add_argument(
        "--num-transitions",
        type=int,
        default=50,
        help="Number of measured direct EE target switches",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hold-s",
        type=float,
        default=3.0,
        help="Hold time after every measured target switch",
    )
    parser.add_argument(
        "--precondition-s",
        type=float,
        default=None,
        help="Hold first sampled command before measurements; default: hold-s",
    )
    parser.add_argument(
        "--steady-window-s",
        type=float,
        default=0.5,
        help="Tail window used to estimate steady-state error",
    )
    parser.add_argument(
        "--t90-dwell-s",
        type=float,
        default=0.10,
        help="Required continuous dwell below the 10%% normalized-error threshold",
    )
    parser.add_argument(
        "--min-position-step-m",
        type=float,
        default=0.05,
        help="Ignore trivial position-command switches smaller than this",
    )
    parser.add_argument(
        "--min-orientation-step-deg",
        type=float,
        default=5.0,
        help="Ignore trivial orientation-command switches smaller than this",
    )
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=90.0,
        help="Exclude finite T90 values above this percentile",
    )
    parser.add_argument(
        "--plot-window-s",
        type=float,
        default=2.0,
        help="Time window shown in the response plot",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/wbc_ee_response_time",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Default: two levels above this script (unitree_mujoco root)",
    )
    args = parser.parse_args()

    if args.num_transitions < 1:
        raise ValueError("--num-transitions must be >= 1")
    if args.hold_s <= 0.0:
        raise ValueError("--hold-s must be positive")
    if not (0.0 < args.steady_window_s < args.hold_s):
        raise ValueError("Need 0 < steady-window-s < hold-s")
    if args.t90_dwell_s <= 0.0:
        raise ValueError("--t90-dwell-s must be positive")
    if not (50.0 < args.outlier_percentile <= 100.0):
        raise ValueError("--outlier-percentile should be in (50, 100]")
    if args.plot_window_s <= 0.0:
        raise ValueError("--plot-window-s must be positive")

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root is not None
        else DEFAULT_PROJECT_ROOT
    )
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    yaml_path = Path(args.yaml_path).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_command = np.asarray(cfg["base_command"], dtype=np.float32)
    if np.linalg.norm(base_command) > 1e-6:
        raise ValueError(
            "EE response-time evaluation expects a stationary base; "
            f"cfg base_command is {base_command.tolist()}"
        )

    command_path = resolve_from_project(cfg["ee_command_path"], project_root)
    command_table = np.load(command_path).astype(np.float32)
    if command_table.ndim != 2 or command_table.shape[1] != 9:
        raise ValueError(
            f"Expected EE command table shape (N,9), got {command_table.shape}"
        )

    # N measured switches require N+1 presampled commands so every measured
    # transient is a command-to-command transition.
    num_commands = args.num_transitions + 1
    if num_commands > command_table.shape[0]:
        raise ValueError(
            f"Need {num_commands} unique commands for {args.num_transitions} "
            f"transitions, but table only has {command_table.shape[0]} rows."
        )

    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(
        command_table.shape[0], size=num_commands, replace=False
    )
    selected_commands = command_table[selected_indices].copy()
    np.save(out_dir / "selected_ee_response_commands.npy", selected_commands)

    selected_positions = []
    selected_quats = []
    for cmd in selected_commands:
        p, q = keypoints_to_pose_plb(cmd)
        selected_positions.append(p)
        selected_quats.append(q)
    selected_positions = np.asarray(selected_positions, dtype=np.float32)
    selected_quats = np.asarray(selected_quats, dtype=np.float32)

    precondition_s = (
        float(args.precondition_s)
        if args.precondition_s is not None
        else float(args.hold_s)
    )
    t90_threshold = 0.10

    print("=" * 80)
    print("Low-level WBC EE response-time evaluation")
    print("=" * 80)
    print(f"YAML:                    {yaml_path}")
    print(f"Command table:           {command_path}")
    print(f"Measured transitions:    {args.num_transitions}")
    print(f"Random seed:             {args.seed}")
    print(f"Precondition hold:        {precondition_s:.2f} s")
    print(f"Hold per target:         {args.hold_s:.2f} s")
    print(f"Steady-state tail:        {args.steady_window_s:.2f} s")
    print(f"T90 dwell:               {args.t90_dwell_s:.2f} s")
    print(f"Min position step:        {args.min_position_step_m:.3f} m")
    print(f"Min orientation step:     {args.min_orientation_step_deg:.1f} deg")
    print(f"Outlier cutoff:           P{args.outlier_percentile:g}")
    print(f"Output:                   {out_dir}")
    print("=" * 80)

    evaluator = WBCUANResponseEvaluator(cfg, project_root)
    evaluator.warmup()

    print(
        f"Preconditioning on table_idx={int(selected_indices[0])} "
        f"for {precondition_s:.2f} s ..."
    )
    evaluator.hold_command(selected_commands[0], precondition_s)

    all_pos_err = []
    all_ori_err_rad = []
    all_pos_norm = []
    all_ori_norm = []
    pos_t90_s = []
    ori_t90_s = []
    pos_initial_m = []
    ori_initial_rad = []
    pos_steady_m = []
    ori_steady_rad = []
    pos_reducible_m = []
    ori_reducible_rad = []
    pos_step_m = []
    ori_step_rad = []

    common_time_s = None

    for i in range(args.num_transitions):
        prev_pos = selected_positions[i]
        prev_quat = selected_quats[i]
        target_pos = selected_positions[i + 1]
        target_quat = selected_quats[i + 1]

        command_pos_step = float(np.linalg.norm(target_pos - prev_pos))
        command_ori_step = quat_geodesic_error_rad(target_quat, prev_quat)

        result = evaluator.evaluate_transition(
            target_kps=selected_commands[i + 1],
            hold_s=args.hold_s,
            steady_window_s=args.steady_window_s,
            t90_threshold=t90_threshold,
            t90_dwell_s=args.t90_dwell_s,
        )

        if common_time_s is None:
            common_time_s = result["time_s"]
        elif len(common_time_s) != len(result["time_s"]):
            raise RuntimeError("Transition time grids unexpectedly differ.")

        all_pos_err.append(result["position_error_m"])
        all_ori_err_rad.append(result["orientation_error_rad"])
        all_pos_norm.append(result["position_normalized_error"])
        all_ori_norm.append(result["orientation_normalized_error"])
        pos_t90_s.append(result["position_t90_s"])
        ori_t90_s.append(result["orientation_t90_s"])
        pos_initial_m.append(result["position_initial_error_m"])
        ori_initial_rad.append(result["orientation_initial_error_rad"])
        pos_steady_m.append(result["position_steady_error_m"])
        ori_steady_rad.append(result["orientation_steady_error_rad"])
        pos_reducible_m.append(result["position_reducible_error_m"])
        ori_reducible_rad.append(result["orientation_reducible_error_rad"])
        pos_step_m.append(command_pos_step)
        ori_step_rad.append(command_ori_step)

        print(
            f"[{i+1:02d}/{args.num_transitions}] "
            f"{int(selected_indices[i])} -> {int(selected_indices[i+1])} | "
            f"dpos={command_pos_step*100:.1f} cm | "
            f"dori={np.degrees(command_ori_step):.1f} deg | "
            f"T90_pos={result['position_t90_s']:.3f} s | "
            f"T90_ori={result['orientation_t90_s']:.3f} s"
        )

    time_s = np.asarray(common_time_s, dtype=np.float64)
    pos_err_m = np.asarray(all_pos_err, dtype=np.float32)
    ori_err_rad = np.asarray(all_ori_err_rad, dtype=np.float32)
    pos_norm = np.asarray(all_pos_norm, dtype=np.float32)
    ori_norm = np.asarray(all_ori_norm, dtype=np.float32)

    pos_t90_s = np.asarray(pos_t90_s, dtype=np.float64)
    ori_t90_s = np.asarray(ori_t90_s, dtype=np.float64)
    pos_initial_m = np.asarray(pos_initial_m, dtype=np.float32)
    ori_initial_rad = np.asarray(ori_initial_rad, dtype=np.float32)
    pos_steady_m = np.asarray(pos_steady_m, dtype=np.float32)
    ori_steady_rad = np.asarray(ori_steady_rad, dtype=np.float32)
    pos_reducible_m = np.asarray(pos_reducible_m, dtype=np.float32)
    ori_reducible_rad = np.asarray(ori_reducible_rad, dtype=np.float32)
    pos_step_m = np.asarray(pos_step_m, dtype=np.float32)
    ori_step_rad = np.asarray(ori_step_rad, dtype=np.float32)
    ori_step_deg = np.degrees(ori_step_rad)

    # "Trivial" means the desired command itself barely changed. Also require
    # positive reducible error so the normalized response is well-defined.
    pos_eligible = (
        (pos_step_m >= float(args.min_position_step_m))
        & (pos_reducible_m > 1e-6)
    )
    ori_eligible = (
        (ori_step_deg >= float(args.min_orientation_step_deg))
        & (ori_reducible_rad > 1e-6)
    )

    pos_inlier, pos_t90_pctl_s, pos_not_reached = compute_t90_inlier_mask(
        pos_t90_s, pos_eligible, args.outlier_percentile
    )
    ori_inlier, ori_t90_pctl_s, ori_not_reached = compute_t90_inlier_mask(
        ori_t90_s, ori_eligible, args.outlier_percentile
    )

    mean_pos_t90_s = float(np.mean(pos_t90_s[pos_inlier]))
    mean_ori_t90_s = float(np.mean(ori_t90_s[ori_inlier]))

    metrics = {
        "position_mean_t90_s": mean_pos_t90_s,
        "orientation_mean_t90_s": mean_ori_t90_s,
        "position_eligible_transitions": int(np.sum(pos_eligible)),
        "orientation_eligible_transitions": int(np.sum(ori_eligible)),
        "position_inlier_transitions": int(np.sum(pos_inlier)),
        "orientation_inlier_transitions": int(np.sum(ori_inlier)),
        "position_t90_not_reached": int(pos_not_reached),
        "orientation_t90_not_reached": int(ori_not_reached),
        "position_outlier_cutoff_s": float(pos_t90_pctl_s),
        "orientation_outlier_cutoff_s": float(ori_t90_pctl_s),
        "outlier_percentile": float(args.outlier_percentile),
    }

    print("\nSummary")
    print("-" * 80)
    print(
        f"Position $T_90$     : {mean_pos_t90_s:.3f} s "
        f"({int(np.sum(pos_inlier))} retained)"
    )
    print(
        f"Orientation $T_90$  : {mean_ori_t90_s:.3f} s "
        f"({int(np.sum(ori_inlier))} retained)"
    )
    print(
        f"Position T90 cutoff  : {pos_t90_pctl_s:.3f} s | "
        f"not reached: {pos_not_reached}"
    )
    print(
        f"Orientation cutoff   : {ori_t90_pctl_s:.3f} s | "
        f"not reached: {ori_not_reached}"
    )
    print("-" * 80)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        out_dir / "ee_response_results.npz",
        selected_indices=selected_indices.astype(np.int64),
        selected_commands=selected_commands,
        selected_positions=selected_positions,
        selected_quaternions_wxyz=selected_quats,
        time_s=time_s,
        position_error_m=pos_err_m,
        orientation_error_rad=ori_err_rad,
        orientation_error_deg=np.degrees(ori_err_rad).astype(np.float32),
        position_normalized_error=pos_norm,
        orientation_normalized_error=ori_norm,
        position_t90_s=pos_t90_s,
        orientation_t90_s=ori_t90_s,
        position_command_step_m=pos_step_m,
        orientation_command_step_rad=ori_step_rad,
        orientation_command_step_deg=ori_step_deg.astype(np.float32),
        position_initial_error_m=pos_initial_m,
        orientation_initial_error_rad=ori_initial_rad,
        position_steady_error_m=pos_steady_m,
        orientation_steady_error_rad=ori_steady_rad,
        position_reducible_error_m=pos_reducible_m,
        orientation_reducible_error_rad=ori_reducible_rad,
        position_eligible_mask=pos_eligible,
        orientation_eligible_mask=ori_eligible,
        position_inlier_mask=pos_inlier,
        orientation_inlier_mask=ori_inlier,
        position_t90_outlier_cutoff_s=np.float32(pos_t90_pctl_s),
        orientation_t90_outlier_cutoff_s=np.float32(ori_t90_pctl_s),
    )

    make_response_plot(
        out_dir=out_dir,
        time_s=time_s,
        pos_norm=pos_norm,
        ori_norm=ori_norm,
        pos_inlier_mask=pos_inlier,
        ori_inlier_mask=ori_inlier,
        mean_pos_t90_s=mean_pos_t90_s,
        mean_ori_t90_s=mean_ori_t90_s,
        t90_threshold=t90_threshold,
        plot_window_s=min(float(args.plot_window_s), float(args.hold_s)),
    )

    print(f"\nSaved to {out_dir}")
    print("  ee_response_time.png / .pdf")
    print("  ee_response_results.npz")
    print("  selected_ee_response_commands.npy")
    print("  metrics.json")


if __name__ == "__main__":
    main()
