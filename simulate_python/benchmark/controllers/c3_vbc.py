"""C3 / C4 -- VBC-inspired locomotion PPO + stateful arm DLS IK.

Ported from ``deploy_mujoco/b2wz1_vbc_plb_uan.py`` (C3) and
``b2wz1_vbc_plb_legged_uan.py`` (C4).

The 6-D IK targets are NOT PPO actions: the locomotion policy emits only leg
(and, for C3, wheel) actions, and the arm target is owned entirely by a
*stateful* damped-least-squares IK whose ``ik_reference`` persists across ticks.
That statefulness is the part most easily broken by a benchmark harness, so the
IK is called exactly once per 50 Hz tick, after the ONNX inference, matching
``b2wz1_vbc_plb_uan.py:996``.
"""

from __future__ import annotations

import mujoco
import numpy as np

from benchmark.controllers.base import BaselineAdapter, OnnxPolicy, TermMajorHistory, resolve_repo_path
from benchmark.core.frames import plb_pose_from_base, plb_rotmat, rotation_from_keypoints
from utilities.math import (
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_from_rotmat_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
)


def _rotation_error_axis_angle(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Verbatim port of C3's ``_rotation_error_axis_angle`` (q_err = q_t * conj(q_c))."""
    q_current = quat_from_rotmat_wxyz(np.asarray(R_current, dtype=np.float32))
    q_target = quat_from_rotmat_wxyz(np.asarray(R_target, dtype=np.float32))
    q_err = quat_normalize_wxyz(quat_mul_wxyz(q_target, quat_conjugate_wxyz(q_current)))
    q_err = np.asarray(q_err, dtype=np.float64)
    if q_err[0] < 0.0:
        q_err = -q_err
    imag = q_err[1:4]
    magnitude = float(np.linalg.norm(imag))
    if magnitude < 1e-9:
        return (2.0 * imag).astype(np.float64)
    angle = 2.0 * np.arctan2(magnitude, max(0.0, float(q_err[0])))
    return (angle / magnitude) * imag


class VBCAdapter(BaselineAdapter):
    baseline_id = "C3"
    has_wheels = True
    blend_before_step = False  # C3/C4 compute blend inside the policy tick
    state_dtype = np.float64
    # C3/C4 compute tau_pd in float64 from the raw float64 joint states
    # (b2wz1_vbc_plb_uan.py:819); the UAN history input is still float32.
    uan_pd_dtype = np.float64

    def _load_policies(self):
        path = resolve_repo_path(self.cfg["policy_path"], self.repo_root)
        self.obs_dim_per_step = int(self.cfg["obs_dim_per_step"])
        self.obs_dim = int(self.cfg["obs_dim"])
        self.action_dim = int(self.cfg["action_dim"])
        expected_per_step = 74 if self.has_wheels else 66
        expected_action = 16 if self.has_wheels else 12
        if self.obs_dim_per_step != expected_per_step or self.action_dim != expected_action:
            raise ValueError(
                f"{self.baseline_id}: expected obs_dim_per_step={expected_per_step}, "
                f"action_dim={expected_action}"
            )
        self.policy = OnnxPolicy(path, self.obs_dim, self.action_dim)
        self.asset_hashes["policy"] = self.policy.sha256
        self.asset_paths = {"policy": path}

    def _setup(self):
        ik = self.cfg["arm_ik"]
        if str(ik.get("control_mode", "stateful")) != "stateful":
            raise ValueError(f"{self.baseline_id}: expected stateful arm IK")
        self.ik_damping = float(ik["damping"])
        self.ik_pos_gain = float(ik["position_gain"])
        self.ik_ori_gain = float(ik["orientation_gain"])
        self.ik_max_delta = float(ik["max_joint_delta"])
        self.ik_margin = float(ik["joint_limit_margin"])
        self.ik_reference_gain = float(ik["reference_gain"])
        self.ik_max_lead = float(ik["max_reference_lead"])
        hard = np.asarray(ik["arm_hard_joint_limits"], dtype=np.float64).reshape(6, 2)
        soft_factor = float(ik["soft_joint_pos_limit_factor"])
        if not 0.0 < soft_factor <= 1.0:
            raise ValueError("Invalid soft_joint_pos_limit_factor")
        mid = hard.mean(axis=1)
        half = (hard[:, 1] - hard[:, 0]) * (soft_factor / 2.0)
        self.ik_lower = mid - half + self.ik_margin
        self.ik_upper = mid + half - self.ik_margin
        if np.any(self.ik_lower >= self.ik_upper):
            raise ValueError("Arm soft limits invalid")

        schema = [
            ("base_ang_vel", 3),
            ("projected_gravity", 3),
            ("velocity_commands", 3),
            ("ee_kp_commands", 9),
            ("joint_pos_leg", 12),
            ("joint_pos_arm", 6),
            ("joint_vel_leg", 12),
            ("joint_vel_arm", 6),
        ]
        if self.has_wheels:
            schema.append(("joint_vel_wheel", 4))
        schema.append(("actions", self.action_dim))
        assert sum(dim for _, dim in schema) == self.obs_dim_per_step
        self.schema = tuple(schema)
        self.history = TermMajorHistory(self.schema, self.history_length)
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self.model.nv), dtype=np.float64)

    def _build_terms(self, base_command, ee_command) -> dict:
        d = self.data
        terms = {
            "base_ang_vel": self._base_ang_vel(),
            "projected_gravity": self._projected_gravity(),
            "velocity_commands": np.asarray(base_command, dtype=np.float32),
            "ee_kp_commands": np.asarray(ee_command, dtype=np.float32),
            "joint_pos_leg": (d.qpos[self.leg_qidx] - self.leg_default).astype(np.float32),
            "joint_pos_arm": (d.qpos[self.arm_qidx] - self.arm_default).astype(np.float32),
            "joint_vel_leg": d.qvel[self.leg_vidx].astype(np.float32),
            "joint_vel_arm": d.qvel[self.arm_vidx].astype(np.float32),
            "actions": self.last_action.copy(),
        }
        if self.has_wheels:
            terms["joint_vel_wheel"] = d.qvel[self.wheel_vidx].astype(np.float32)
        return terms

    def _reset_controller(self):
        self.last_action[:] = 0.0
        # Native C3 initialises the arm target from the measured state, not the
        # default pose, because the IK owns it (b2wz1_vbc_plb_uan.py:907).
        self.arm_target = self.data.qpos[self.arm_qidx].copy().astype(np.float64)
        self.ik_reference = self.data.qpos[self.arm_qidx].copy().astype(np.float64)
        self.ik_just_reset = True
        self.ik_last_delta = np.zeros(6, dtype=np.float64)
        self.history.seed(self._build_terms(self.last_base_command, self.last_ee_command))

    def _update_stateful_arm_ik(self, ee_command: np.ndarray) -> np.ndarray:
        m, d = self.model, self.data
        q_current = d.qpos[self.arm_qidx].copy().astype(np.float64)

        plb_pos_w, plb_quat_w = plb_pose_from_base(
            d.qpos[0:3], d.qpos[3:7], self.ee_ground_z
        )
        desired_plb = np.asarray(ee_command, dtype=np.float64).reshape(3, 3)
        target_pos_w = plb_pos_w.astype(np.float64) + quat_apply_wxyz(
            plb_quat_w, desired_plb[0].astype(np.float32)
        ).astype(np.float64)
        R_plb = plb_rotmat(d.qpos[3:7])
        target_rot_w = R_plb @ rotation_from_keypoints(ee_command)

        actual_pos_w = d.xpos[self.ee_bid].copy().astype(np.float64)
        actual_rot_w = d.xmat[self.ee_bid].reshape(3, 3).copy().astype(np.float64)

        pos_error_w = target_pos_w - actual_pos_w
        rot_error_w = _rotation_error_axis_angle(actual_rot_w, target_rot_w)
        task_error = np.concatenate(
            (self.ik_pos_gain * pos_error_w, self.ik_ori_gain * rot_error_w)
        )

        self._jacp[:] = 0.0
        self._jacr[:] = 0.0
        mujoco.mj_jac(m, d, self._jacp, self._jacr, actual_pos_w, self.ee_bid)
        J = np.vstack((self._jacp[:, self.arm_vidx], self._jacr[:, self.arm_vidx]))
        if J.shape != (6, 6):
            raise RuntimeError(f"Wrong world-frame arm Jacobian shape: {J.shape}")

        dq = J.T @ np.linalg.solve(
            J @ J.T + (self.ik_damping**2) * np.eye(6), task_error
        )
        largest = float(np.max(np.abs(dq)))
        if largest > self.ik_max_delta:
            dq *= self.ik_max_delta / max(largest, 1e-6)
        self.ik_last_delta[:] = dq

        if self.ik_just_reset:
            self.ik_reference = q_current.copy()
            self.ik_just_reset = False
        else:
            q_candidate = np.clip(
                self.ik_reference + self.ik_reference_gain * dq, self.ik_lower, self.ik_upper
            )
            lead = np.clip(q_candidate - q_current, -self.ik_max_lead, self.ik_max_lead)
            self.ik_reference = q_current + lead
        if not np.isfinite(self.ik_reference).all():
            self.ik_reference = q_current.copy()
        return self.ik_reference.copy()

    def _policy_update(self, base_command, ee_command):
        self.history.append(self._build_terms(base_command, ee_command))
        obs = self.history.stack()
        if obs.shape != (self.obs_dim,) or not np.isfinite(obs).all():
            raise RuntimeError(f"{self.baseline_id}: invalid observation {obs.shape}")

        raw_action = self.policy(obs)
        if raw_action.shape != (self.action_dim,) or not np.isfinite(raw_action).all():
            raise FloatingPointError(f"{self.baseline_id}: invalid action {raw_action}")
        self.last_action[:] = raw_action  # observation stores the unscaled raw action

        blend = self._blend
        self.leg_target = self.leg_default + blend * self.leg_action_scale * raw_action[
            0:12
        ].astype(np.float64)
        if self.has_wheels:
            self.wheel_cmd = (
                blend * self.wheel_action_scale * raw_action[12:16].astype(np.float64)
            )
        # No RL arm action: the stateful DLS IK always owns the arm target.
        self.arm_target = self._update_stateful_arm_ik(ee_command)

    def diagnostics(self) -> dict:
        out = super().diagnostics()
        out["ik_delta_max"] = float(np.max(np.abs(self.ik_last_delta)))
        return out


class C3Adapter(VBCAdapter):
    baseline_id = "C3"
    has_wheels = True


class C4Adapter(VBCAdapter):
    baseline_id = "C4"
    has_wheels = False
