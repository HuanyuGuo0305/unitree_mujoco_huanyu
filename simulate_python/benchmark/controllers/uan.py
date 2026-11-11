"""Frozen UAN arm actuator -- shared, unmodified, by all six baselines.

Verbatim port of ``compute_uan_arm_torque`` (C1 b2wz1_wbc_plb_uan.py:782-821,
identical law in C2-C6). Runs once per 250 Hz physics step.

    pos_err = (q_des - q) * pos_scale
    vel     = qd * vel_scale
    history <- shift left, newest last, layout (6, 25, 2) joint-major
    action  = torchscript(history.reshape(1, 300))     # Beta-mean in [-1, 1]
    tau_pd  = Kp (q_des - q) - Kd qd
    tau_nom = clip(tau_pd, +- nominal_limits)
    dtau    = action * action_scale * Kp
    tau     = clip(tau_nom + dtau, +- final_limits)
"""

from __future__ import annotations

import numpy as np
import torch


class FrozenUAN:
    def __init__(self, model_path: str, cfg: dict, device: str = "cpu",
                 pd_dtype=np.float32):
        # C1/C2/C5/C6 compute tau_pd in float32 from float32 joint states;
        # C3/C4 compute it in float64 from the raw float64 states
        # (b2wz1_vbc_plb_uan.py:819). Matching this exactly is required for
        # bit-level parity with the original scripts.
        self.pd_dtype = np.dtype(pd_dtype)
        self.device = str(cfg.get("uan_device", device))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"UAN device '{self.device}' requested but CUDA unavailable.")

        self.history_length = int(cfg.get("uan_history_length", 25))
        if self.history_length != 25:
            raise ValueError(f"Expected UAN history_length=25, got {self.history_length}")

        self.pos_scale = float(cfg.get("uan_pos_scale", 1.0))
        self.vel_scale = float(cfg.get("uan_vel_scale", 0.05))
        self.action_scale = float(cfg.get("uan_action_scale", 0.20))
        self.kp = np.asarray(cfg["uan_kp"], dtype=np.float32).reshape(6)
        self.kd = np.asarray(cfg["uan_kd"], dtype=np.float32).reshape(6)
        self.nominal_limits = np.asarray(
            cfg["uan_nominal_torque_limits"], dtype=np.float32
        ).reshape(6)
        self.final_limits = np.asarray(
            cfg["uan_final_torque_limits"], dtype=np.float32
        ).reshape(6)
        if np.any(self.final_limits < self.nominal_limits):
            raise ValueError("UAN final torque limits must be >= nominal limits.")
        self.saturation_threshold = float(cfg.get("uan_action_saturation_threshold", 0.95))

        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        with torch.inference_mode():
            dummy = torch.zeros(
                (1, 6 * self.history_length * 2), dtype=torch.float32, device=self.device
            )
            out = self.model(dummy)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if tuple(out.shape) != (1, 6):
                raise ValueError(f"Expected UAN output shape (1,6), got {tuple(out.shape)}")
            if not torch.isfinite(out).all():
                raise ValueError("UAN returned non-finite output for zero-history smoke test.")

        self.history = np.zeros((6, self.history_length, 2), dtype=np.float32)
        self.last_action = np.zeros(6, dtype=np.float32)
        self.last_delta_tau = np.zeros(6, dtype=np.float32)
        self.last_tau_pd = np.zeros(6, dtype=np.float32)
        self.last_tau_nom = np.zeros(6, dtype=np.float32)
        self.n_calls = 0

    def reset(self):
        """Zero history -- NOT pre-filled, matching the frozen model semantics."""
        self.history[:] = 0.0
        self.last_action[:] = 0.0
        self.last_delta_tau[:] = 0.0
        self.last_tau_pd[:] = 0.0
        self.last_tau_nom[:] = 0.0
        self.n_calls = 0

    def torque(self, q_des: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        q_des_raw = np.asarray(q_des).reshape(6)
        q_raw = np.asarray(q).reshape(6)
        qd_raw = np.asarray(qd).reshape(6)
        q_des = q_des_raw.astype(np.float32)
        q = q_raw.astype(np.float32)
        qd = qd_raw.astype(np.float32)

        raw_q_des, raw_q, raw_qd = q_des_raw, q_raw, qd_raw
        pos_err = (q_des - q) * self.pos_scale
        vel = qd * self.vel_scale

        self.history[:, :-1, :] = self.history[:, 1:, :]
        self.history[:, -1, 0] = pos_err
        self.history[:, -1, 1] = vel

        obs = np.ascontiguousarray(self.history.reshape(1, -1), dtype=np.float32)
        with torch.inference_mode():
            out = self.model(torch.from_numpy(obs).to(device=self.device))
            if isinstance(out, (tuple, list)):
                out = out[0]
        action = out.detach().cpu().numpy().reshape(6).astype(np.float32)

        if not np.all(np.isfinite(action)):
            raise FloatingPointError(f"Non-finite UAN action: {action}")
        if np.max(np.abs(action)) > 1.001:
            raise RuntimeError(f"UAN action left expected [-1,1] Beta-mean range: {action}")

        if self.pd_dtype == np.float64:
            tau_pd = self.kp * (
                raw_q_des.astype(np.float64) - raw_q.astype(np.float64)
            ) - self.kd * raw_qd.astype(np.float64)
        else:
            tau_pd = self.kp * (q_des - q) - self.kd * qd
        tau_nom = np.clip(tau_pd, -self.nominal_limits, self.nominal_limits)
        delta_tau = action * self.action_scale * self.kp
        tau = np.clip(tau_nom + delta_tau, -self.final_limits, self.final_limits).astype(
            self.pd_dtype
        )

        self.last_action[:] = action
        self.last_delta_tau[:] = delta_tau
        self.last_tau_nom[:] = tau_nom
        self.last_tau_pd[:] = tau_pd
        self.n_calls += 1
        return tau


def sync_mujoco_arm_limits(
    model, arm_joint_names, actuator_for_joint, final_limits, enabled: bool = True, verbose: bool = False
):
    """Port of the arm torque-limit synchronisation every script performs.

    The explicit UAN law owns both nominal and final saturation, so MuJoCo's
    redundant actuator/joint limits must not clip inside that envelope.
    """
    import mujoco

    for j, joint_name in enumerate(arm_joint_names):
        act_id = actuator_for_joint[joint_name]
        limit = float(final_limits[j])

        for limited, ranges in (
            (model.actuator_ctrllimited, model.actuator_ctrlrange),
            (model.actuator_forcelimited, model.actuator_forcerange),
        ):
            if bool(limited[act_id]):
                lo, hi = ranges[act_id]
                if lo > -limit + 1e-6 or hi < limit - 1e-6:
                    if not enabled:
                        raise RuntimeError(
                            f"MuJoCo range for {joint_name} is [{lo}, {hi}], narrower "
                            f"than the UAN final +/-{limit} Nm."
                        )
                    ranges[act_id] = (-limit, limit)
                    if verbose:
                        print(f"  {joint_name}: [{lo:.1f},{hi:.1f}] -> [-{limit:.1f},{limit:.1f}]")

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if (
            jid >= 0
            and hasattr(model, "jnt_actfrclimited")
            and bool(model.jnt_actfrclimited[jid])
        ):
            lo, hi = model.jnt_actfrcrange[jid]
            if lo > -limit + 1e-6 or hi < limit - 1e-6:
                if not enabled:
                    raise RuntimeError(
                        f"MuJoCo jnt_actfrcrange for {joint_name} is [{lo}, {hi}], "
                        f"narrower than the UAN final +/-{limit} Nm."
                    )
                model.jnt_actfrcrange[jid] = (-limit, limit)

    # Post-condition: MuJoCo may not clip inside the UAN final envelope.
    for j, joint_name in enumerate(arm_joint_names):
        act_id = actuator_for_joint[joint_name]
        limit = float(final_limits[j])
        if bool(model.actuator_ctrllimited[act_id]):
            lo, hi = model.actuator_ctrlrange[act_id]
            assert lo <= -limit + 1e-6 and hi >= limit - 1e-6
        if bool(model.actuator_forcelimited[act_id]):
            lo, hi = model.actuator_forcerange[act_id]
            assert lo <= -limit + 1e-6 and hi >= limit - 1e-6
