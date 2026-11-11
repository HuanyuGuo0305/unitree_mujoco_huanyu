"""C5 / C6 -- RoboDuet-inspired cooperative dual-PPO.

Ported from ``deploy_mujoco/b2wz1_roboduet_plb_uan.py`` (C5) and
``b2wz1_roboduet_plb_legged_uan.py`` (C6).

The inference order is load-bearing and must not be reordered
(b2wz1_roboduet_plb_uan.py:626-637):

    1. one physical state read -> ARM and LOCO term values
    2. append both histories
    3. arm_raw      = ARM_onnx(arm_hist.stack())
    4. new_guidance = posture.inject(arm_raw)      # stateful 50 Hz slew
    5. loco_obs     = patch_posture_history(loco_hist.stack(), new_guidance)
    6. loco_raw     = LOCO_onnx(loco_obs)

``posture.inject`` is a stateful rate limiter: it is called exactly once per
policy tick, enforced by an assertion on the tick counter.
"""

from __future__ import annotations

import math

import numpy as np

from benchmark.controllers.base import BaselineAdapter, OnnxPolicy, TermMajorHistory, resolve_repo_path
from benchmark.core.frames import rotmat_from_quat


class PostureGuidance:
    """Verbatim port: raw arm[6:8] -> tanh/angle -> clip -> slew at 50 Hz."""

    def __init__(self, dt: float, slew_rate: float = 0.35):
        self.dt = float(dt)
        self.slew_rate = float(slew_rate)
        self.command = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self.n_injections = 0

    def reset(self):
        self.command[:] = 0.0
        self.target[:] = 0.0
        self.n_injections = 0

    @staticmethod
    def decode(arm_raw: np.ndarray) -> np.ndarray:
        a = np.asarray(arm_raw, dtype=np.float32).reshape(8)
        x = np.tanh(a[6:8].astype(np.float64))
        pitch = float(x[0]) * math.radians(25.0 if x[0] >= 0 else 5.0)
        roll = float(x[1]) * math.radians(3.0)
        return np.array([pitch, roll], dtype=np.float32)

    def inject(self, arm_raw: np.ndarray) -> np.ndarray:
        self.target = self.decode(arm_raw)
        lower = np.deg2rad([-5.0, -3.0])
        upper = np.deg2rad([25.0, 3.0])
        bounded = np.clip(self.target.astype(np.float64), lower, upper)
        self.command = (
            self.command.astype(np.float64)
            + np.clip(
                bounded - self.command.astype(np.float64),
                -self.slew_rate * self.dt,
                self.slew_rate * self.dt,
            )
        ).astype(np.float32)
        self.n_injections += 1
        return self.command.copy()


class RoboDuetAdapter(BaselineAdapter):
    baseline_id = "C5"
    has_wheels = True
    blend_before_step = False
    state_dtype = np.float64
    uan_pd_dtype = np.float32

    def _load_policies(self):
        arm_path = resolve_repo_path(self.cfg["arm_policy_path"], self.repo_root)
        loco_path = resolve_repo_path(self.cfg["loco_policy_path"], self.repo_root)

        if self.has_wheels:
            self.loco_terms = (
                ("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3),
                ("base_posture_commands", 2), ("joint_pos_leg", 12), ("joint_pos_arm", 6),
                ("joint_vel_leg", 12), ("joint_vel_arm", 6), ("joint_vel_wheel", 4),
                ("actions", 16),
            )
            self.arm_terms = (
                ("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3),
                ("ee_kp_commands", 9), ("joint_pos_leg", 12), ("joint_pos_arm", 6),
                ("joint_vel_leg", 12), ("joint_vel_arm", 6), ("joint_vel_wheel", 4),
                ("last_arm_joint_action", 6), ("previous_guidance", 2),
            )
            self.loco_action_dim = 16
        else:
            self.loco_terms = (
                ("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3),
                ("base_posture_commands", 2), ("joint_pos_leg", 12), ("joint_pos_arm", 6),
                ("joint_vel_leg", 12), ("joint_vel_arm", 6),
                ("actions", 12),
            )
            self.arm_terms = (
                ("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3),
                ("ee_kp_commands", 9), ("joint_pos_leg", 12), ("joint_pos_arm", 6),
                ("joint_vel_leg", 12), ("joint_vel_arm", 6),
                ("last_arm_joint_action", 6), ("previous_guidance", 2),
            )
            self.loco_action_dim = 12

        self.arm_obs_dim = sum(d for _, d in self.arm_terms) * self.history_length
        self.loco_obs_dim = sum(d for _, d in self.loco_terms) * self.history_length
        self.arm_policy = OnnxPolicy(arm_path, self.arm_obs_dim, 8)
        self.loco_policy = OnnxPolicy(loco_path, self.loco_obs_dim, self.loco_action_dim)
        self.asset_hashes["arm_policy"] = self.arm_policy.sha256
        self.asset_hashes["loco_policy"] = self.loco_policy.sha256
        self.asset_paths = {"arm_policy": arm_path, "loco_policy": loco_path}

        # base_posture_commands history occupies [45:55] (C5) / [45:55] (C6):
        # 5 * (3 + 3 + 3) = 45 precedes it, and it is 5 * 2 wide.
        self._posture_slice_start = 45
        self._posture_slice_end = 55

    def _setup(self):
        arm_scale = self.cfg["arm_action_scale"]
        self.arm_action_scale = (
            float(arm_scale) if np.isscalar(arm_scale) else np.asarray(arm_scale, dtype=np.float64)
        )
        self.posture = PostureGuidance(
            self.control_dt, slew_rate=float(self.cfg.get("guidance_rate_rad_s", 0.35))
        )
        self.arm_history = TermMajorHistory(self.arm_terms, self.history_length)
        self.loco_history = TermMajorHistory(self.loco_terms, self.history_length)
        self.last_arm = np.zeros(6, dtype=np.float32)
        self.last_loco = np.zeros(self.loco_action_dim, dtype=np.float32)

    def _projected_gravity(self) -> np.ndarray:
        # C5/C6 use R(q).T @ [0,0,-1] directly, without quat_unique_wxyz.
        R = rotmat_from_quat(self.data.qpos[3:7])
        return (R.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

    def _common_terms(self, base_command) -> dict:
        d = self.data
        out = {
            "base_ang_vel": self._base_ang_vel(),
            "projected_gravity": self._projected_gravity(),
            "velocity_commands": np.asarray(base_command, dtype=np.float32),
            "joint_pos_leg": (d.qpos[self.leg_qidx] - self.leg_default).astype(np.float32),
            "joint_pos_arm": (d.qpos[self.arm_qidx] - self.arm_default).astype(np.float32),
            "joint_vel_leg": d.qvel[self.leg_vidx].astype(np.float32),
            "joint_vel_arm": d.qvel[self.arm_vidx].astype(np.float32),
        }
        if self.has_wheels:
            out["joint_vel_wheel"] = d.qvel[self.wheel_vidx].astype(np.float32)
        return out

    def _frame_pair(self, base_command, ee_command) -> tuple[dict, dict]:
        """One physical state read, shared by both groups, BEFORE new guidance."""
        values = self._common_terms(base_command)
        loco = dict(values)
        loco.update(
            {
                "base_posture_commands": self.posture.command.copy(),
                "actions": self.last_loco.copy(),
            }
        )
        arm = dict(values)
        arm.update(
            {
                "ee_kp_commands": np.asarray(ee_command, dtype=np.float32),
                "last_arm_joint_action": self.last_arm.copy(),
                "previous_guidance": self.posture.command.copy(),
            }
        )
        return arm, loco

    def _patch_loco_posture_history(
        self, raw_loco: np.ndarray, current_guidance: np.ndarray
    ) -> np.ndarray:
        """Shift ONLY the posture term by one frame and write the new guidance.

        Operates on a COPY of the stacked vector; the posture deque itself is
        never shifted twice.
        """
        if raw_loco.shape != (self.loco_obs_dim,):
            raise ValueError(f"LOCO obs dim {raw_loco.shape} != {self.loco_obs_dim}")
        a, b = self._posture_slice_start, self._posture_slice_end
        out = raw_loco.copy()
        out[a : b - 2] = raw_loco[a + 2 : b]
        out[b - 2 : b] = np.asarray(current_guidance, dtype=np.float32).reshape(2)
        return out

    def _reset_controller(self):
        self.posture.reset()
        self.last_arm[:] = 0.0
        self.last_loco[:] = 0.0
        arm, loco = self._frame_pair(self.last_base_command, self.last_ee_command)
        self.arm_history.seed(arm)
        self.loco_history.seed(loco)

    def _policy_update(self, base_command, ee_command):
        expected_injections = self.policy_ticks
        if self.posture.n_injections != expected_injections:
            raise RuntimeError(
                f"{self.baseline_id}: posture guidance injected "
                f"{self.posture.n_injections} times for {expected_injections} ticks "
                "-- the stateful rate limiter was double-updated."
            )

        arm_terms, loco_terms = self._frame_pair(base_command, ee_command)
        self.arm_history.append(arm_terms)
        self.loco_history.append(loco_terms)

        arm_obs = self.arm_history.stack()
        arm_raw = self.arm_policy(arm_obs)
        if arm_raw.shape != (8,) or not np.isfinite(arm_raw).all():
            raise FloatingPointError(f"{self.baseline_id}: invalid ARM action {arm_raw}")

        new_guidance = self.posture.inject(arm_raw)
        loco_obs = self._patch_loco_posture_history(self.loco_history.stack(), new_guidance)
        loco_raw = self.loco_policy(loco_obs)
        if loco_raw.shape != (self.loco_action_dim,) or not np.isfinite(loco_raw).all():
            raise FloatingPointError(f"{self.baseline_id}: invalid LOCO action {loco_raw}")

        self.last_arm[:] = arm_raw[:6]
        self.last_loco[:] = loco_raw

        blend = self._blend
        self.leg_target = self.leg_default + blend * self.leg_action_scale * loco_raw[
            0:12
        ].astype(np.float64)
        self.arm_target = self.arm_default + blend * self.arm_action_scale * arm_raw[
            0:6
        ].astype(np.float64)
        if self.has_wheels:
            self.wheel_cmd = (
                blend * self.wheel_action_scale * loco_raw[12:16].astype(np.float64)
            )

    def last_policy_action(self) -> np.ndarray:
        return self.last_loco

    def diagnostics(self) -> dict:
        out = super().diagnostics()
        out["guidance_exec_pitch_deg"] = float(np.degrees(self.posture.command[0]))
        out["guidance_exec_roll_deg"] = float(np.degrees(self.posture.command[1]))
        out["guidance_req_pitch_deg"] = float(np.degrees(self.posture.target[0]))
        return out


class C5Adapter(RoboDuetAdapter):
    baseline_id = "C5"
    has_wheels = True


class C6Adapter(RoboDuetAdapter):
    baseline_id = "C6"
    has_wheels = False
