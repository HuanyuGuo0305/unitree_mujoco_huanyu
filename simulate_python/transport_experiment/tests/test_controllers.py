"""Controller / decoder checks for the four E4 methods.

    python3 -m unittest transport_experiment.tests.test_controllers -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../..")))

from transport_experiment.controllers import METHODS, build_methods, decode_residual  # noqa: E402
from transport_experiment.sim import TransportSim, resolve_repo_path  # noqa: E402
from transport_experiment.tests.test_alignment import _nominal_scenario  # noqa: E402

EXP = yaml.safe_load(open(os.path.join(HERE, "../config.yaml")))
DEP = yaml.safe_load(open(resolve_repo_path(EXP["deploy_config"])))


def isaac_decode(variant, actions, kp0_ref, rot_ref, settling, pos_scale, rot_scale, pmin, pmax, rmin, rmax):
    """Transcription of _pre_physics_step + _decode_hl_action_to_ll_commands of each Isaac Lab env (torch)."""
    a = torch.as_tensor(actions)[None].clamp(-1.0, 1.0)
    if settling:
        a = torch.zeros_like(a)
    ref = torch.as_tensor(kp0_ref)[None]
    rref = torch.as_tensor(rot_ref)[None]
    ps, rs = torch.as_tensor(pos_scale), torch.as_tensor(rot_scale)
    if variant == "full":
        pos_raw = ref + a[:, 0:3] * ps
        rot_raw = rref + a[:, 3:6] * rs
    elif variant == "pos_only":
        pos_raw = ref + a[:, 0:3] * ps
        rot_raw = rref
    else:  # ori_only
        pos_raw = ref
        rot_raw = rref + a[:, 0:3] * rs
    pos = torch.maximum(torch.minimum(pos_raw, torch.as_tensor(pmax)), torch.as_tensor(pmin))
    if settling:
        pos = ref
    rot = torch.maximum(torch.minimum(rot_raw, torch.as_tensor(rmax)), torch.as_tensor(rmin))
    ep = ((pos - ref) / ps).clamp(-1, 1)
    er = ((rot - rref) / rs).clamp(-1, 1)
    ex = {"full": torch.cat([ep, er], -1), "pos_only": ep, "ori_only": er}[variant]
    return pos[0].numpy(), rot[0].numpy(), ex[0].numpy()


class TestControllers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.sim = TransportSim(DEP, low_policy_path=EXP["low_policy"]["path"])
        cls.methods = build_methods(EXP)

    def _active_sim(self):
        s = self.sim
        s.reset(_nominal_scenario())
        while s.settling:                     # run the settle with zero residual
            self.methods["no_residual"].reset(s) if s.hl_steps_completed == 0 else None
            self.methods["no_residual"].step(s)
            s.run_hl_interval()
            s.advance_references()
        return s

    def test_models_and_dims(self):
        expect = {"no_residual": (None, 0), "pos_only": (183, 3), "ori_only": (183, 3), "full": (192, 6)}
        for name in METHODS:
            m = self.methods[name]
            obs_dim, adim = expect[name]
            self.assertEqual(m.action_dim, adim)
            if adim:
                self.assertEqual(m.obs_dim, obs_dim)
                self.assertEqual(m.policy_sha256, EXP["methods"][name]["sha256"])
                self.assertEqual(m.sess.get_outputs()[0].shape[-1], adim)

    def test_policy_outputs_and_commands(self):
        s = self._active_sim()
        for name in METHODS:
            m = self.methods[name]
            m.reset(s)
            if m.action_dim:
                frame = s.hl_obs_frame(m.executed)
                self.assertEqual(frame.shape, (m.frame_dim,))
                self.assertEqual(frame.shape[0], 64 if name == "full" else 61)
            base_ref = s.base_velocity_ref.copy()
            out = m.step(s)
            self.assertTrue(np.all(np.isfinite(out["raw"])))
            self.assertEqual(out["raw"].shape, (m.action_dim,))
            self.assertEqual(out["executed"].shape, (m.action_dim,))
            self.assertEqual(out["diag6"].shape, (6,))
            np.testing.assert_array_equal(s.base_command, base_ref)            # base reference untouched
            if name == "no_residual":
                np.testing.assert_array_equal(out["diag6"], 0)
            if name == "pos_only":
                np.testing.assert_array_equal(out["diag6"][3:], 0)
                np.testing.assert_array_equal(s.rot_cmd, np.clip(s.rot_ref, s.ee_rot_cmd_min, s.ee_rot_cmd_max))
            if name == "ori_only":
                np.testing.assert_array_equal(out["diag6"][:3], 0)
                np.testing.assert_array_equal(s.kp0_cmd, np.clip(s.ee_kp0_ref, s.ee_pos_cmd_min, s.ee_pos_cmd_max))
            # WBC 9-D keypoint command is the PLB encoding of the decoded pose.
            from transport_experiment.sim import build_keypoints_from_kp0_yaw_pitch_roll_plb as kp
            np.testing.assert_array_equal(s.ee_cmd_plb, kp(s.kp0_cmd, *map(float, s.rot_cmd), 0.3, 0.3))

    def test_residual_scale_and_clipping_match_isaac(self):
        s = self._active_sim()
        rng = np.random.default_rng(0)
        for variant, adim in (("full", 6), ("pos_only", 3), ("ori_only", 3)):
            for trial in range(300):
                a = rng.uniform(-1.4, 1.4, adim).astype(np.float32)
                s.ee_kp0_ref = rng.uniform([0.35, -0.45, 0.05], [0.95, 0.45, 1.25]).astype(np.float32)
                s.ee_yaw_ref, s.ee_pitch_ref, s.ee_roll_ref = map(float, rng.uniform([-1.2, -0.9, 0.6], [1.2, 0.9, 2.5]))
                settling = trial % 7 == 0
                s.settle_steps_remaining = 1 if settling else 0
                raw, eff, kp0, rot, ex, diag = decode_residual(variant, a, s)
                pk, pr, pe = isaac_decode(variant, a, s.ee_kp0_ref, s.rot_ref, settling, s.ee_pos_scale, s.ee_rot_scale,
                                          s.ee_pos_cmd_min, s.ee_pos_cmd_max, s.ee_rot_cmd_min, s.ee_rot_cmd_max)
                np.testing.assert_allclose(kp0, pk, atol=1e-7)
                np.testing.assert_allclose(rot, pr, atol=1e-7)
                np.testing.assert_allclose(ex, pe, atol=1e-6)
                if settling:
                    np.testing.assert_array_equal(eff, 0)
        # Explicit residual scale: +1 -> +10 cm / +45 deg when far from the bounds.
        s.settle_steps_remaining = 0
        s.ee_kp0_ref = np.array([0.6, 0.0, 0.7], np.float32)
        s.ee_yaw_ref, s.ee_pitch_ref, s.ee_roll_ref = 0.0, -0.06, np.pi / 2
        _, _, kp0, rot, _, _ = decode_residual("full", np.array([1, -1, 0.5, 0.5, 0.2, -0.5], np.float32), s)
        np.testing.assert_allclose(kp0, [0.7, -0.1, 0.75], atol=1e-6)
        np.testing.assert_allclose(rot, [np.pi / 8, -0.06 + 0.2 * np.pi / 4, np.pi / 2 - np.pi / 8], atol=1e-6)

    def test_previous_action_is_last_executed(self):
        s = self._active_sim()
        for name in ("pos_only", "ori_only", "full"):
            m = self.methods[name]
            m.reset(s)
            out1 = m.step(s)
            s.run_hl_interval()
            s.advance_references()
            m.step(s)
            # newest history frame holds a_t^exec of the PREVIOUS step (feature group 10)
            newest = np.asarray(m.history.h[10][-1])
            np.testing.assert_array_equal(newest, out1["executed"])


if __name__ == "__main__":
    unittest.main()
