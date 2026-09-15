#!/usr/bin/env python3
"""
UAN joint-level sim-to-real replay validation in MuJoCo.

Purpose
-------
Replay one hardware-recorded 6-DoF Z1 q_des trajectory twice from the same
initial state:

  1) nominal PD only       ("Sim w/o UAN")
  2) nominal PD + UAN      ("Sim w/ UAN")

The script compares both MuJoCo responses against the recorded real Z1
response and automatically writes:

  - joint2_response.png
  - joint2_sim2real_error.png
  - joint2_uan_validation_combined.png
  - mujoco_replay_results.npz
  - metrics.json

The actuator path matches the supplied WBC-UAN sim2sim code:

  tau_pd  = Kp * (q_des - q) - Kd * qd
  tau_nom = clip(tau_pd, +/- nominal_limit)
  delta   = UAN(history) * action_scale * Kp
  tau     = clip(tau_nom + delta, +/- final_limit)

For the no-UAN run, delta = 0 and all other simulator/controller settings
remain identical.

Example
-------
python3 deploy_mujoco/uan_joint2_replay_validation.py \
    configs/b2wz1_wbc_uan_plb.yaml \
    --data utilities/data/uan_joint2_seg89_replay.npz \
    --out results/uan_joint2_seg89
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import yaml


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


def resolve_from_project(path: str, project_root: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def audit_and_sync_arm_limits(
    m: mujoco.MjModel,
    final_limits: np.ndarray,
    sync_limits: bool,
) -> None:
    """
    Keep MuJoCo's redundant arm limits from clipping inside the explicit
    UAN final torque envelope.
    """
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
                print(
                    f"  {joint_name}: ctrlrange "
                    f"[{old[0]:.1f}, {old[1]:.1f}] -> "
                    f"[-{final_lim:.1f}, +{final_lim:.1f}]"
                )

        if bool(m.actuator_forcelimited[ctrl_i]):
            old = m.actuator_forcerange[ctrl_i].copy()
            if old[0] > -final_lim + 1e-6 or old[1] < final_lim - 1e-6:
                if not sync_limits:
                    raise RuntimeError(
                        f"{joint_name} forcerange {old} is narrower than +/-{final_lim} Nm"
                    )
                m.actuator_forcerange[ctrl_i, :] = (-final_lim, final_lim)
                print(
                    f"  {joint_name}: forcerange "
                    f"[{old[0]:.1f}, {old[1]:.1f}] -> "
                    f"[-{final_lim:.1f}, +{final_lim:.1f}]"
                )

        joint_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
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
                        f"{joint_name} joint actfrcrange {old} is narrower "
                        f"than +/-{final_lim} Nm"
                    )
                m.jnt_actfrcrange[joint_id, :] = (-final_lim, final_lim)
                print(
                    f"  {joint_name}: joint actfrcrange "
                    f"[{old[0]:.1f}, {old[1]:.1f}] -> "
                    f"[-{final_lim:.1f}, +{final_lim:.1f}]"
                )

    print("  PASS: explicit actuator law owns the arm torque limits.")


class UANActuator:
    def __init__(self, cfg: dict, model_path: Path):
        self.device = str(cfg.get("uan_device", "cpu"))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"UAN device '{self.device}' requested but CUDA is unavailable")

        self.history_length = int(cfg.get("uan_history_length", 25))
        self.pos_scale = float(cfg.get("uan_pos_scale", 1.0))
        self.vel_scale = float(cfg.get("uan_vel_scale", 0.05))
        self.action_scale = float(cfg.get("uan_action_scale", 0.20))

        self.kp = np.asarray(cfg["uan_kp"], dtype=np.float32)
        self.kd = np.asarray(cfg["uan_kd"], dtype=np.float32)
        self.nominal_limits = np.asarray(
            cfg["uan_nominal_torque_limits"], dtype=np.float32
        )
        self.final_limits = np.asarray(
            cfg["uan_final_torque_limits"], dtype=np.float32
        )

        for name, arr in (
            ("uan_kp", self.kp),
            ("uan_kd", self.kd),
            ("uan_nominal_torque_limits", self.nominal_limits),
            ("uan_final_torque_limits", self.final_limits),
        ):
            if arr.shape != (6,):
                raise ValueError(f"{name} must have shape (6,), got {arr.shape}")

        if self.history_length != 25:
            raise ValueError(
                f"Expected 25-step UAN history, got {self.history_length}"
            )

        self.model = torch.jit.load(
            str(model_path), map_location=self.device
        ).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.history = np.zeros(
            (6, self.history_length, 2), dtype=np.float32
        )

        with torch.inference_mode():
            dummy = torch.zeros(
                (1, 6 * self.history_length * 2),
                dtype=torch.float32,
                device=self.device,
            )
            out = self.model(dummy)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if tuple(out.shape) != (1, 6):
                raise ValueError(
                    f"Expected UAN output shape (1,6), got {tuple(out.shape)}"
                )

    def reset(self) -> None:
        self.history.fill(0.0)

    def nominal_torque(
        self, q_des: np.ndarray, q: np.ndarray, qd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        tau_pd = self.kp * (q_des - q) - self.kd * qd
        tau_nom = np.clip(
            tau_pd, -self.nominal_limits, self.nominal_limits
        ).astype(np.float32)
        return tau_pd.astype(np.float32), tau_nom

    def step(
        self,
        q_des: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        use_uan: bool,
    ) -> tuple[np.ndarray, dict]:
        q_des = np.asarray(q_des, dtype=np.float32).reshape(6)
        q = np.asarray(q, dtype=np.float32).reshape(6)
        qd = np.asarray(qd, dtype=np.float32).reshape(6)

        tau_pd, tau_nom = self.nominal_torque(q_des, q, qd)

        if use_uan:
            pos_err = (q_des - q) * self.pos_scale
            vel = qd * self.vel_scale

            self.history[:, :-1, :] = self.history[:, 1:, :]
            self.history[:, -1, 0] = pos_err
            self.history[:, -1, 1] = vel

            obs = np.ascontiguousarray(
                self.history.reshape(1, -1), dtype=np.float32
            )
            obs_t = torch.from_numpy(obs).to(self.device)

            with torch.inference_mode():
                action = self.model(obs_t)
                if isinstance(action, (tuple, list)):
                    action = action[0]
            action = (
                action.detach().cpu().numpy().reshape(6).astype(np.float32)
            )

            if not np.all(np.isfinite(action)):
                raise FloatingPointError(f"Non-finite UAN action: {action}")
            if np.max(np.abs(action)) > 1.001:
                raise RuntimeError(
                    f"UAN action outside expected [-1,1] range: {action}"
                )

            delta_tau = action * self.action_scale * self.kp
        else:
            # Keep the baseline identical except for the learned residual.
            action = np.zeros(6, dtype=np.float32)
            delta_tau = np.zeros(6, dtype=np.float32)

        tau = np.clip(
            tau_nom + delta_tau,
            -self.final_limits,
            self.final_limits,
        ).astype(np.float32)

        diag = {
            "tau_pd": tau_pd,
            "tau_nom": tau_nom,
            "uan_action": action,
            "delta_tau": delta_tau.astype(np.float32),
            "tau": tau,
        }
        return tau, diag


def build_model(cfg: dict, xml_path: Path) -> mujoco.MjModel:
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    dt = float(cfg["simulation_dt"])
    if abs(dt - 0.004) > 1e-9:
        raise ValueError(
            f"Replay expects simulation_dt=0.004 (250 Hz), got {dt}"
        )
    m.opt.timestep = dt

    final_limits = np.asarray(
        cfg["uan_final_torque_limits"], dtype=np.float32
    )
    audit_and_sync_arm_limits(
        m,
        final_limits,
        bool(cfg.get("sync_mujoco_arm_limits_to_uan", True)),
    )
    return m


def get_joint_qpos_qvel(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    q = []
    qd = []
    for name in names:
        jid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if jid < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {name}")
        q.append(float(d.qpos[m.jnt_qposadr[jid]]))
        qd.append(float(d.qvel[m.jnt_dofadr[jid]]))
    return np.asarray(q, dtype=np.float32), np.asarray(qd, dtype=np.float32)


def set_joint_state(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    names: list[str],
    q: np.ndarray,
    qd: np.ndarray,
) -> None:
    for name, qi, qdi in zip(names, q, qd):
        jid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if jid < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {name}")
        d.qpos[m.jnt_qposadr[jid]] = float(qi)
        d.qvel[m.jnt_dofadr[jid]] = float(qdi)


def initialize_state(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    cfg: dict,
    q_arm0: np.ndarray,
    qd_arm0: np.ndarray,
) -> None:
    root_pos = np.asarray(cfg["root_pos"], dtype=np.float64)
    root_quat = np.asarray(cfg["root_quat_wxyz"], dtype=np.float64)
    default_joint_pos = np.asarray(
        cfg["default_joint_pos"], dtype=np.float64
    )

    if default_joint_pos.shape != (23,):
        raise ValueError(
            f"default_joint_pos must have 23 entries, got {default_joint_pos.shape}"
        )

    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0

    if m.nq < 7 + 23:
        raise ValueError(
            f"Model nq={m.nq} is too small for free root + 23 joints"
        )

    d.qpos[0:3] = root_pos
    d.qpos[3:7] = root_quat
    d.qpos[7:7 + 23] = default_joint_pos

    # Overwrite the six arm joints with the measured hardware initial state.
    set_joint_state(
        m, d, ARM_JOINT_NAMES, q_arm0, qd_arm0
    )

    mujoco.mj_forward(m, d)


def run_replay(
    m: mujoco.MjModel,
    cfg: dict,
    uan: UANActuator,
    q_des: np.ndarray,
    q_real0: np.ndarray,
    qd_real0: np.ndarray,
    use_uan: bool,
) -> dict:
    d = mujoco.MjData(m)
    initialize_state(m, d, cfg, q_real0, qd_real0)
    uan.reset()

    n = q_des.shape[0]
    q_sim = np.zeros((n, 6), dtype=np.float32)
    qd_sim = np.zeros((n, 6), dtype=np.float32)
    tau_arm = np.zeros((n - 1, 6), dtype=np.float32)
    tau_nom = np.zeros((n - 1, 6), dtype=np.float32)
    delta_tau = np.zeros((n - 1, 6), dtype=np.float32)
    uan_action = np.zeros((n - 1, 6), dtype=np.float32)

    q_sim[0], qd_sim[0] = get_joint_qpos_qvel(
        m, d, ARM_JOINT_NAMES
    )

    kps = np.asarray(cfg["kps"], dtype=np.float32)
    kds = np.asarray(cfg["kds"], dtype=np.float32)
    default_joint_pos = np.asarray(
        cfg["default_joint_pos"], dtype=np.float32
    )

    leg_torque_limits_cfg = np.asarray(
        cfg["leg_torque_limits"], dtype=np.float32
    )
    if leg_torque_limits_cfg.shape != (12,):
        raise ValueError(
            f"leg_torque_limits must have 12 entries, got {leg_torque_limits_cfg.shape}"
        )
    # YAML order: FL(h,t,c), FR(h,t,c), RL(h,t,c), RR(h,t,c).
    leg_limit_order = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    leg_limit_by_name = {
        name: leg_torque_limits_cfg[i]
        for i, name in enumerate(leg_limit_order)
    }
    leg_torque_limits = np.asarray(
        [leg_limit_by_name[name] for name in LEG_JOINT_NAMES],
        dtype=np.float32,
    )
    gripper_torque_limit = float(
        cfg.get("gripper_torque_limit", 30.0)
    )

    # Same mappings as the supplied working sim2sim deployment script.
    leg_mj_idx = [MUJOCO_JOINT_NAMES.index(nm) for nm in LEG_JOINT_NAMES]
    gripper_mj_idx = MUJOCO_JOINT_NAMES.index("jointGripper")

    default_leg_pos = default_joint_pos[leg_mj_idx]
    default_gripper_pos = float(default_joint_pos[gripper_mj_idx])

    # Control source order used by the working WBC-UAN script.
    control_source_joint_names = (
        LEG_JOINT_NAMES
        + ARM_JOINT_NAMES
        + WHEEL_JOINT_NAMES
        + ["jointGripper"]
    )
    ctrl_source_index_by_name = {
        name: i for i, name in enumerate(control_source_joint_names)
    }
    ctrl_src_indices_or_none = [
        ctrl_source_index_by_name.get(name, None)
        for name in CTRL_JOINT_NAMES
    ]

    for k in range(n - 1):
        arm_pos, arm_vel = get_joint_qpos_qvel(
            m, d, ARM_JOINT_NAMES
        )

        # Hold the B2-W legs at the same standing posture used by pd-stand.
        leg_pos, leg_vel = get_joint_qpos_qvel(
            m, d, LEG_JOINT_NAMES
        )
        leg_kp = kps[leg_mj_idx]
        leg_kd = kds[leg_mj_idx]
        leg_tau = (
            leg_kp * (default_leg_pos - leg_pos)
            - leg_kd * leg_vel
        )
        leg_tau = np.clip(
            leg_tau, -leg_torque_limits, leg_torque_limits
        ).astype(np.float32)

        arm_tau_k, diag = uan.step(
            q_des=q_des[k],
            q=arm_pos,
            qd=arm_vel,
            use_uan=use_uan,
        )

        gripper_pos, gripper_vel = get_joint_qpos_qvel(
            m, d, ["jointGripper"]
        )
        g_kp = float(kps[gripper_mj_idx])
        g_kd = float(kds[gripper_mj_idx])
        gripper_tau = (
            g_kp * (default_gripper_pos - float(gripper_pos[0]))
            - g_kd * float(gripper_vel[0])
        )
        gripper_tau = float(
            np.clip(
                gripper_tau,
                -gripper_torque_limit,
                gripper_torque_limit,
            )
        )

        wheel_ctrl = np.zeros(4, dtype=np.float32)

        ctrl_source = np.concatenate(
            [
                leg_tau,
                arm_tau_k,
                wheel_ctrl,
                np.array([gripper_tau], dtype=np.float32),
            ]
        )

        d.ctrl[:] = 0.0
        for ctrl_i, src_i in enumerate(ctrl_src_indices_or_none):
            if src_i is not None:
                d.ctrl[ctrl_i] = ctrl_source[src_i]

        mujoco.mj_step(m, d)

        q_sim[k + 1], qd_sim[k + 1] = get_joint_qpos_qvel(
            m, d, ARM_JOINT_NAMES
        )
        tau_arm[k] = diag["tau"]
        tau_nom[k] = diag["tau_nom"]
        delta_tau[k] = diag["delta_tau"]
        uan_action[k] = diag["uan_action"]

    return {
        "q": q_sim,
        "qd": qd_sim,
        "tau": tau_arm,
        "tau_nom": tau_nom,
        "delta_tau": delta_tau,
        "uan_action": uan_action,
    }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b))))


def make_plots(
    out_dir: Path,
    t: np.ndarray,
    q_des_j2: np.ndarray,
    q_real_j2: np.ndarray,
    q_no_j2: np.ndarray,
    q_uan_j2: np.ndarray,
    rmse_no: float,
    rmse_uan: float,
) -> None:
    # Plot 1: joint-2 response.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(t, q_des_j2, label="Command")
    ax.plot(t, q_real_j2, label="Real robot", linewidth=2.0)
    ax.plot(t, q_no_j2, label="Sim w/o UAN")
    ax.plot(t, q_uan_j2, label="Sim w/ UAN")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint-2 position [rad]")
    ax.set_title("Joint-2 Response")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "joint2_response.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "joint2_response.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot 2: sim-to-real error.
    err_no = q_no_j2 - q_real_j2
    err_uan = q_uan_j2 - q_real_j2

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(t, err_no * 1000.0, label=f"Sim w/o UAN ({rmse_no*1000:.1f} mrad RMSE)")
    ax.plot(t, err_uan * 1000.0, label=f"Sim w/ UAN ({rmse_uan*1000:.1f} mrad RMSE)")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Sim-to-real error [mrad]")
    ax.set_title("Joint-2 Sim-to-Real Position Error")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        out_dir / "joint2_sim2real_error.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "joint2_sim2real_error.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Combined slide-friendly two-panel figure.
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.2))

    axes[0].plot(t, q_des_j2, label="Command")
    axes[0].plot(t, q_real_j2, label="Real robot", linewidth=2.0)
    axes[0].plot(t, q_no_j2, label="Sim w/o UAN")
    axes[0].plot(t, q_uan_j2, label="Sim w/ UAN")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Joint-2 position [rad]")
    axes[0].set_title("Joint-2 Response")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        t,
        err_no * 1000.0,
        label=f"w/o UAN: {rmse_no*1000:.1f} mrad",
    )
    axes[1].plot(
        t,
        err_uan * 1000.0,
        label=f"w/ UAN: {rmse_uan*1000:.1f} mrad",
    )
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Sim-to-real error [mrad]")
    axes[1].set_title("Sim-to-Real Position Error")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        out_dir / "joint2_uan_validation_combined.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "joint2_uan_validation_combined.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "yaml_path",
        type=str,
        help="Existing b2wz1_wbc_uan_plb.yaml",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=str,
        help="Extracted seg replay .npz (relative paths are resolved from project root)",
    )
    parser.add_argument(
        "--out",
        default="results/uan_joint2_seg89",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        type=str,
        help=(
            "Project root used to resolve xml_path / uan_model_path. "
            "Default: two levels above this script."
        ),
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    if args.project_root is None:
        project_root = script_path.parents[2]
    else:
        project_root = Path(args.project_root).resolve()

    # yaml_path and --out remain relative to the current working directory
    # (normally simulate_python/), while replay data follows the same
    # project-root convention as xml_path / uan_model_path.
    yaml_path = Path(args.yaml_path).resolve()
    data_path = resolve_from_project(args.data, project_root)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    xml_path = resolve_from_project(cfg["xml_path"], project_root)
    uan_model_path = resolve_from_project(
        cfg["uan_model_path"], project_root
    )

    for name, p in (
        ("xml_path", xml_path),
        ("uan_model_path", uan_model_path),
        ("data", data_path),
    ):
        if not p.is_file():
            raise FileNotFoundError(f"{name} not found: {p}")

    hw = np.load(data_path)
    q_des = np.asarray(hw["q_des"], dtype=np.float32)
    q_real = np.asarray(hw["q_real"], dtype=np.float32)
    qd_real = np.asarray(hw["qd_real"], dtype=np.float32)

    if q_des.ndim != 2 or q_des.shape[1] != 6:
        raise ValueError(f"q_des must be (N,6), got {q_des.shape}")
    if q_real.shape != q_des.shape or qd_real.shape != q_des.shape:
        raise ValueError(
            "q_des, q_real, qd_real must all have the same (N,6) shape"
        )

    dt = float(cfg["simulation_dt"])
    logged_dt = float(hw["dt"]) if "dt" in hw.files else dt
    if abs(dt - logged_dt) > 1e-9:
        raise ValueError(
            f"Timing mismatch: sim dt={dt}, replay data dt={logged_dt}"
        )

    n = q_des.shape[0]
    t = np.arange(n, dtype=np.float64) * dt

    print("=" * 78)
    print("UAN joint-2 sim-to-real replay")
    print("=" * 78)
    print(f"YAML:      {yaml_path}")
    print(f"XML:       {xml_path}")
    print(f"UAN model: {uan_model_path}")
    print(f"Replay:    {data_path}")
    print(f"Samples:   {n}")
    print(f"dt:        {dt:.6f} s ({1.0/dt:.1f} Hz)")
    print(f"Duration:  {t[-1]:.3f} s")
    print(f"Initial q: {q_real[0]}")
    print(f"Initial qd:{qd_real[0]}")
    print("=" * 78)

    # One model definition, two fresh MjData states, identical initial conditions.
    m = build_model(cfg, xml_path)
    uan = UANActuator(cfg, uan_model_path)

    print("\nRunning baseline: nominal PD only ...")
    no_uan = run_replay(
        m=m,
        cfg=cfg,
        uan=uan,
        q_des=q_des,
        q_real0=q_real[0],
        qd_real0=qd_real[0],
        use_uan=False,
    )

    print("Running UAN: nominal PD + residual ...")
    with_uan = run_replay(
        m=m,
        cfg=cfg,
        uan=uan,
        q_des=q_des,
        q_real0=q_real[0],
        qd_real0=qd_real[0],
        use_uan=True,
    )

    j2 = 1
    rmse_no = rmse(no_uan["q"][:, j2], q_real[:, j2])
    rmse_uan = rmse(with_uan["q"][:, j2], q_real[:, j2])
    improvement = (
        100.0 * (rmse_no - rmse_uan) / rmse_no
        if rmse_no > 0.0
        else float("nan")
    )

    print("\nResults")
    print("-" * 78)
    print(f"Joint-2 RMSE, Sim w/o UAN : {rmse_no*1000.0:.3f} mrad")
    print(f"Joint-2 RMSE, Sim w/ UAN  : {rmse_uan*1000.0:.3f} mrad")
    print(f"Relative error reduction  : {improvement:.2f} %")
    print("-" * 78)

    metrics = {
        "joint": "joint2",
        "num_samples": int(n),
        "dt_s": float(dt),
        "duration_s": float(t[-1]),
        "rmse_no_uan_rad": rmse_no,
        "rmse_with_uan_rad": rmse_uan,
        "rmse_no_uan_mrad": rmse_no * 1000.0,
        "rmse_with_uan_mrad": rmse_uan * 1000.0,
        "relative_error_reduction_percent": improvement,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        out_dir / "mujoco_replay_results.npz",
        t=t,
        q_des=q_des,
        q_real=q_real,
        qd_real=qd_real,
        q_no_uan=no_uan["q"],
        qd_no_uan=no_uan["qd"],
        tau_no_uan=no_uan["tau"],
        q_with_uan=with_uan["q"],
        qd_with_uan=with_uan["qd"],
        tau_with_uan=with_uan["tau"],
        tau_nom_with_uan=with_uan["tau_nom"],
        delta_tau_with_uan=with_uan["delta_tau"],
        uan_action=with_uan["uan_action"],
    )

    make_plots(
        out_dir=out_dir,
        t=t,
        q_des_j2=q_des[:, j2],
        q_real_j2=q_real[:, j2],
        q_no_j2=no_uan["q"][:, j2],
        q_uan_j2=with_uan["q"][:, j2],
        rmse_no=rmse_no,
        rmse_uan=rmse_uan,
    )

    print(f"\nSaved outputs to: {out_dir}")
    print("  joint2_response.png / .pdf")
    print("  joint2_sim2real_error.png / .pdf")
    print("  joint2_uan_validation_combined.png / .pdf")
    print("  mujoco_replay_results.npz")
    print("  metrics.json")


if __name__ == "__main__":
    main()
