"""C1 / C2 -- unified whole-body PPO (ours), wheeled-legged and fixed-wheel.

Ported from ``deploy_mujoco/b2wz1_wbc_plb_uan.py`` (C1) and
``deploy_mujoco/b2wz1_wbc_plb_legged_uan.py`` (C2), mode ``full-policy``.

Observation (per step, term-major 5-frame history):
    base_ang_vel 3 | projected_gravity 3 | base_command 3 | ee_kp_commands 9
    joint_pos_leg 12 | joint_pos_arm 6 | joint_vel_leg 12 | joint_vel_arm 6
    [joint_vel_wheel 4 -- C1 only] | last_action 22 (C1) / 18 (C2)

Action: legs[0:12] * 0.25, arm[12:18] * 0.10, wheels[18:22] * 4.0 (C1 only),
each scaled by the startup blend and added to the default joint position.
"""

from __future__ import annotations

import numpy as np

from benchmark.controllers.base import BaselineAdapter, OnnxPolicy, TermMajorHistory, resolve_repo_path


class WBCAdapter(BaselineAdapter):
    baseline_id = "C1"
    has_wheels = True
    blend_before_step = True   # C1/C2 compute blend before the physics step
    state_dtype = np.float32   # C1/C2 run the whole control path in float32
    uan_pd_dtype = np.float32

    def _load_policies(self):
        path = resolve_repo_path(self.cfg["policy_path"], self.repo_root)
        self.obs_dim_per_step = int(self.cfg["obs_dim_per_step"])
        self.obs_dim = int(self.cfg["obs_dim"])
        self.action_dim = int(self.cfg["action_dim"])
        expected_per_step = 80 if self.has_wheels else 72
        expected_action = 22 if self.has_wheels else 18
        if self.obs_dim_per_step != expected_per_step or self.action_dim != expected_action:
            raise ValueError(
                f"{self.baseline_id}: expected obs_dim_per_step={expected_per_step} and "
                f"action_dim={expected_action}, got {self.obs_dim_per_step} / {self.action_dim}"
            )
        if self.obs_dim != self.obs_dim_per_step * self.history_length:
            raise ValueError(f"{self.baseline_id}: obs_dim mismatch")
        self.policy = OnnxPolicy(path, self.obs_dim, self.action_dim)
        self.asset_hashes["policy"] = self.policy.sha256
        self.asset_paths = {"policy": path}

    def _setup(self):
        arm_scale = self.cfg["arm_action_scale"]
        self.arm_action_scale = (
            np.full(6, float(arm_scale), dtype=np.float32)
            if np.isscalar(arm_scale)
            else np.asarray(arm_scale, dtype=np.float32).reshape(6)
        )
        schema = [
            ("base_ang_vel", 3),
            ("projected_gravity", 3),
            ("base_command", 3),
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

    def _build_terms(self, base_command, ee_command) -> dict:
        d = self.data
        terms = {
            "base_ang_vel": self._base_ang_vel(),
            "projected_gravity": self._projected_gravity(),
            "base_command": np.asarray(base_command, dtype=np.float32),
            "ee_kp_commands": np.asarray(ee_command, dtype=np.float32),
            # C1/C2 cast qpos to float32 BEFORE subtracting the default
            # (b2wz1_wbc_plb_uan.py:825-835), unlike C3-C6.
            "joint_pos_leg": d.qpos[self.leg_qidx].astype(np.float32) - self.leg_default,
            "joint_pos_arm": d.qpos[self.arm_qidx].astype(np.float32) - self.arm_default,
            "joint_vel_leg": d.qvel[self.leg_vidx].astype(np.float32),
            "joint_vel_arm": d.qvel[self.arm_vidx].astype(np.float32),
            "actions": self.last_action.copy(),
        }
        if self.has_wheels:
            terms["joint_vel_wheel"] = d.qvel[self.wheel_vidx].astype(np.float32)
        return terms

    def _reset_controller(self):
        self.last_action[:] = 0.0
        self.history.seed(self._build_terms(self.last_base_command, self.last_ee_command))

    def _policy_update(self, base_command, ee_command):
        self.history.append(self._build_terms(base_command, ee_command))
        obs = self.history.stack()
        if obs.shape != (self.obs_dim,) or not np.isfinite(obs).all():
            raise RuntimeError(f"{self.baseline_id}: invalid observation {obs.shape}")

        action = self.policy(obs)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise FloatingPointError(f"{self.baseline_id}: invalid action {action}")
        self.last_action[:] = action

        blend = self._blend
        self.leg_target = self.leg_default + blend * (self.leg_action_scale * action[0:12])
        self.arm_target = self.arm_default + blend * (self.arm_action_scale * action[12:18])
        if self.has_wheels:
            self.wheel_cmd = blend * (self.wheel_action_scale * action[18:22])


class C1Adapter(WBCAdapter):
    baseline_id = "C1"
    has_wheels = True
    # C1 reads the gripper gain out of a float32 array.
    gripper_gain_dtype = np.float32


class C2Adapter(WBCAdapter):
    baseline_id = "C2"
    has_wheels = False
    # C2 wraps the gripper gains in float(), so its gripper law is float64.
    gripper_gain_dtype = np.float64
