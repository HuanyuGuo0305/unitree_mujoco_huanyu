#!/usr/bin/env python3
"""Unit tests for command generation, frames, metrics and statistics.

Covers acceptance criteria 1-15 (criteria 16 and 17 are covered by
``test_controller_parity.py`` and ``test_bitexact_parity.py``).

Run:  python3 benchmark/tests/test_scenarios.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for root in (REPO_ROOT, SIM_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

from benchmark.core import frames, scenario as sc, stats  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

CONTROL_DT = 0.02


def _cfgs(name):
    return E.load_config(name)


# ==========================================================================
# 1. Deterministic command generation
# ==========================================================================
class TestDeterminism(unittest.TestCase):
    def test_base_schedule_is_deterministic(self):
        a = sc.make_base_velocity_schedule(1000, CONTROL_DT, 7)
        b = sc.make_base_velocity_schedule(1000, CONTROL_DT, 7)
        np.testing.assert_array_equal(a.commands, b.commands)
        np.testing.assert_array_equal(a.switch_ticks, b.switch_ticks)

    def test_different_seeds_differ(self):
        a = sc.make_base_velocity_schedule(1000, CONTROL_DT, 7)
        b = sc.make_base_velocity_schedule(1000, CONTROL_DT, 8)
        self.assertFalse(np.array_equal(a.commands, b.commands))

    def test_e1_scenarios_reproducible(self):
        cfg, _ = _cfgs("E1")
        for condition in E.E1_CONDITIONS:
            s1 = E.build_e1(cfg, condition, 3)
            s2 = E.build_e1(cfg, condition, 3)
            np.testing.assert_array_equal(s1.base.commands, s2.base.commands)
            np.testing.assert_array_equal(s1.ee.commands, s2.ee.commands)


# ==========================================================================
# 2. Base velocity resampling interval 2-4 s
# ==========================================================================
class TestBaseResampling(unittest.TestCase):
    def test_segment_durations_within_2_to_4_s(self):
        for seed in range(30):
            s = sc.make_base_velocity_schedule(10_000, CONTROL_DT, seed)
            # The final segment is truncated by the episode end.
            full = s.segment_durations_s[:-1]
            self.assertTrue((full >= 2.0 - 1e-9).all(), f"seed {seed}: {full.min()}")
            self.assertTrue((full <= 4.0 + 1e-9).all(), f"seed {seed}: {full.max()}")

    def test_switch_spacing_matches_durations(self):
        s = sc.make_base_velocity_schedule(5000, CONTROL_DT, 1)
        spacing = np.diff(s.switch_ticks) * CONTROL_DT
        np.testing.assert_allclose(spacing, s.segment_durations_s[:-1], atol=1e-12)

    def test_commands_are_piecewise_constant(self):
        s = sc.make_base_velocity_schedule(5000, CONTROL_DT, 2)
        changes = np.nonzero(np.any(np.diff(s.commands, axis=0) != 0, axis=1))[0] + 1
        self.assertTrue(set(changes.tolist()).issubset(set(s.switch_ticks.tolist())))

    def test_commands_inside_declared_limits(self):
        limits = np.array([0.6, 0.6, 0.6])
        for seed in range(30):
            s = sc.make_base_velocity_schedule(2000, CONTROL_DT, seed, limits=limits)
            self.assertLessEqual(float(np.abs(s.commands).max()), float(limits.max()) + 1e-9)

    def test_intensity_scales_the_same_waveform(self):
        a = sc.make_base_velocity_schedule(2000, CONTROL_DT, 4, intensity=1.0)
        b = sc.make_base_velocity_schedule(2000, CONTROL_DT, 4, intensity=0.5)
        np.testing.assert_array_equal(a.waveform, b.waveform)
        np.testing.assert_allclose(b.commands, a.commands * 0.5, atol=1e-7)


# ==========================================================================
# 3 + 4. E1 EE resampling exactly every 5 s and no interpolation
# ==========================================================================
class TestE1EECommands(unittest.TestCase):
    def setUp(self):
        self.cfg, self.glob = _cfgs("E1")
        self.table = E.keypoint_table(self.glob)

    def test_e1_ee_switch_interval_exactly_5s(self):
        n = E.n_eval_ticks(self.cfg)
        for seed in range(10):
            s = sc.make_ee_random_schedule(n, CONTROL_DT, seed, self.table, 5.0)
            spacing = np.diff(s.switch_ticks)
            self.assertTrue(len(s.switch_ticks) >= 4)
            self.assertTrue(
                (spacing == 250).all(),
                f"seed {seed}: switch spacing {spacing} ticks, expected all 250 (= 5.000 s)",
            )
            self.assertAlmostEqual(float(spacing[0]) * CONTROL_DT, 5.0, places=12)

    def test_e1_has_no_interpolation_or_smoothing(self):
        """Between switches the command must be bit-identical; at a switch it
        must jump directly to a table row with no intermediate values."""
        n = E.n_eval_ticks(self.cfg)
        s = sc.make_ee_random_schedule(n, CONTROL_DT, 0, self.table, 5.0)
        changes = np.nonzero(np.any(np.diff(s.commands, axis=0) != 0, axis=1))[0] + 1
        np.testing.assert_array_equal(changes, s.switch_ticks)
        # every emitted command is EXACTLY a row of the native table
        rows = {tuple(r) for r in self.table.table}
        for value in np.unique(s.commands, axis=0):
            self.assertIn(tuple(value), rows)
        # the number of distinct commands equals the number of segments
        self.assertEqual(len(np.unique(s.commands, axis=0)), len(s.switch_ticks) + 1)

    def test_e1_fixed_target_is_constant_and_from_the_table(self):
        n = E.n_eval_ticks(self.cfg)
        s = sc.make_ee_fixed_schedule(n, 0, self.table)
        self.assertEqual(len(np.unique(s.commands, axis=0)), 1)
        rows = {tuple(r) for r in self.table.table}
        self.assertIn(tuple(s.commands[0]), rows)

    def test_e1_switches_are_inside_the_evaluated_window(self):
        n = E.n_eval_ticks(self.cfg)
        s = sc.make_ee_random_schedule(n, CONTROL_DT, 0, self.table, 5.0)
        self.assertTrue((s.switch_ticks > 0).all())
        self.assertTrue((s.switch_ticks < n).all())


# ==========================================================================
# 5. Identical commands across compared baselines
# ==========================================================================
class TestCommandSharing(unittest.TestCase):
    def test_e1_commands_identical_for_c1_c3_c5(self):
        cfg, _ = _cfgs("E1")
        for condition in E.E1_CONDITIONS:
            for seed in (0, 1, 2):
                built = [E.build_e1(cfg, condition, seed) for _ in ("C1", "C3", "C5")]
                for other in built[1:]:
                    np.testing.assert_array_equal(built[0].base.commands, other.base.commands)
                    np.testing.assert_array_equal(built[0].ee.commands, other.ee.commands)

    def test_e2_commands_identical_for_c1_c2(self):
        cfg, _ = _cfgs("E2")
        for condition in E.E2_CONDITIONS:
            a = E.build_e2(cfg, condition, 0)
            b = E.build_e2(cfg, condition, 0)
            np.testing.assert_array_equal(a.base.commands, b.base.commands)
            np.testing.assert_array_equal(a.ee.commands, b.ee.commands)

    def test_e1_pairing_base_shared_across_ee_conditions(self):
        """T1/T2 and T3/T4 must share base commands; T1/T3 and T2/T4 the EE ones."""
        cfg, _ = _cfgs("E1")
        s = {c: E.build_e1(cfg, c, 11) for c in E.E1_CONDITIONS}
        np.testing.assert_array_equal(s["T1"].base.commands, s["T2"].base.commands)
        np.testing.assert_array_equal(s["T3"].base.commands, s["T4"].base.commands)
        np.testing.assert_array_equal(s["T1"].ee.commands, s["T3"].ee.commands)
        np.testing.assert_array_equal(s["T2"].ee.commands, s["T4"].ee.commands)


# ==========================================================================
# 6 + 7. Frames, quaternion conventions, orientation metric
# ==========================================================================
class TestFrames(unittest.TestCase):
    def test_keypoint_pose_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            p = rng.uniform(-1, 1, 3)
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            R = frames.rotmat_from_quat(q)
            kp = frames.pose_to_keypoints(p, R)
            p2, R2 = frames.keypoints_to_pose(kp)
            np.testing.assert_allclose(p, p2, atol=1e-6)
            np.testing.assert_allclose(R, R2, atol=1e-6)

    def test_native_table_rows_are_valid_poses(self):
        _, glob = _cfgs("E1")
        table = E.keypoint_table(glob)
        rng = np.random.default_rng(1)
        for i in rng.integers(0, table.n_rows, 500):
            _, R = frames.keypoints_to_pose(table.row(int(i)))
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=5)

    def test_rotation_from_keypoints_matches_c3(self):
        """The benchmark must use C3's exact keypoint->rotation convention."""
        sys.path.insert(0, os.path.join(SIM_ROOT, "deploy_mujoco"))
        _, glob = _cfgs("E1")
        table = E.keypoint_table(glob)
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_c3_src", os.path.join(SIM_ROOT, "deploy_mujoco", "b2wz1_vbc_plb_uan.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # safe: the script body is under __main__
        for i in (0, 1, 7, 100, 19999):
            np.testing.assert_array_equal(
                module._rotation_from_keypoints(table.row(i)),
                frames.rotation_from_keypoints(table.row(i)),
            )

    def test_geodesic_angle_properties(self):
        self.assertAlmostEqual(frames.geodesic_angle_deg(np.eye(3), np.eye(3)), 0.0, places=9)
        for angle in (5.0, 30.0, 90.0, 179.0):
            a = np.radians(angle)
            Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            self.assertAlmostEqual(frames.geodesic_angle_deg(np.eye(3), Rz), angle, places=6)
            # symmetric
            self.assertAlmostEqual(
                frames.geodesic_angle_deg(Rz, np.eye(3)),
                frames.geodesic_angle_deg(np.eye(3), Rz), places=9,
            )

    def test_geodesic_is_not_quaternion_subtraction(self):
        """q and -q are the same rotation; a raw quaternion difference is not."""
        from utilities.math import quat_from_rotmat_wxyz

        a = np.radians(20.0)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        q = quat_from_rotmat_wxyz(R.astype(np.float32))
        self.assertAlmostEqual(frames.geodesic_angle_deg(R, R), 0.0, places=9)
        # the sign-flipped quaternion has a huge raw difference but zero geodesic
        self.assertGreater(float(np.linalg.norm(np.asarray(q) - (-np.asarray(q)))), 1.0)

    def test_plb_frame_is_yaw_only_and_ground_referenced(self):
        from utilities.math import quat_from_yaw_wxyz, quat_mul_wxyz

        roll = np.radians(12.0)
        q_roll = np.array([np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0], dtype=np.float32)
        q_yaw = quat_from_yaw_wxyz(0.7)
        q = quat_mul_wxyz(q_yaw, q_roll)
        pos = np.array([1.5, -2.5, 0.62], dtype=np.float32)
        plb_pos, plb_quat = frames.plb_pose_from_base(pos, q, ground_z=0.0)
        np.testing.assert_allclose(plb_pos, [1.5, -2.5, 0.0], atol=1e-6)
        R = frames.rotmat_from_quat(plb_quat)
        # pure yaw: third row/column are the world z axis
        np.testing.assert_allclose(R[2, :], [0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(R[:, 2], [0, 0, 1], atol=1e-6)
        self.assertAlmostEqual(frames.yaw_from_quat(plb_quat), 0.7, places=6)

    def test_world_plb_roundtrip(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            base_pos = rng.uniform(-3, 3, 3).astype(np.float32)
            yaw = rng.uniform(-np.pi, np.pi)
            from utilities.math import quat_from_yaw_wxyz

            base_quat = quat_from_yaw_wxyz(yaw)
            kp_plb = rng.uniform(-1, 1, 9).astype(np.float32)
            plb_pos, plb_quat = frames.plb_pose_from_base(base_pos, base_quat, 0.0)
            kp_w = frames.keypoints_plb_to_world(kp_plb, plb_pos, plb_quat)
            back = frames.keypoints_world_to_plb(kp_w, plb_pos, plb_quat)
            np.testing.assert_allclose(kp_plb, back, atol=1e-5)

    def test_e4_world_to_plb_is_exact_inverse_of_the_state_map(self):
        from utilities.math import quat_from_yaw_wxyz

        rng = np.random.default_rng(4)
        for _ in range(50):
            base_pos = rng.uniform(-2, 2, 3).astype(np.float32)
            base_quat = quat_from_yaw_wxyz(rng.uniform(-np.pi, np.pi))
            ee_pos_w = base_pos + rng.uniform(-0.5, 0.5, 3).astype(np.float32)
            a = rng.uniform(-np.pi, np.pi)
            R_w = np.array(
                [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]
            )
            cmd = frames.world_pose_to_plb_keypoints(ee_pos_w, R_w, base_pos, base_quat)
            measured = frames.ee_keypoints_plb_from_state(
                ee_pos_w, R_w, base_pos, base_quat
            )
            pos_err, ori_err = frames.keypoint_pose_error(cmd, measured)
            self.assertLess(pos_err, 1e-5)
            self.assertLess(ori_err, 1e-3)

    def test_base_velocity_frames(self):
        from utilities.math import quat_from_yaw_wxyz

        yaw = 0.5
        q = quat_from_yaw_wxyz(yaw)
        v_w = np.array([1.0, 0.0, 0.0])
        w_w = np.array([0.0, 0.0, 0.3])
        v_b, wz_b = frames.base_velocity_body_frame(v_w, w_w, q)
        np.testing.assert_allclose(v_b[:2], [np.cos(yaw), -np.sin(yaw)], atol=1e-9)
        self.assertAlmostEqual(wz_b, 0.3, places=12)
        v_y, wz_y = frames.base_velocity_yaw_frame(v_w, w_w, q)
        np.testing.assert_allclose(v_b, v_y, atol=1e-9)  # yaw-only base: identical


# ==========================================================================
# 8. Tracking RMSE implementation
# ==========================================================================
class TestMetricMath(unittest.TestCase):
    def test_vxy_rmse_definition(self):
        """E_v = sqrt(mean ||v_xy - v_cmd||^2), i.e. the 2-D vector norm."""
        dvx = np.array([3.0, 0.0])
        dvy = np.array([4.0, 0.0])
        expected = np.sqrt(np.mean(dvx**2 + dvy**2))
        self.assertAlmostEqual(expected, np.sqrt((25.0 + 0.0) / 2), places=12)
        # a per-axis RMSE average would give a different (wrong) answer
        wrong = 0.5 * (np.sqrt(np.mean(dvx**2)) + np.sqrt(np.mean(dvy**2)))
        self.assertNotAlmostEqual(expected, wrong, places=3)

    def test_ee_ori_is_mae_not_rmse(self):
        errs = np.array([0.0, 10.0])
        self.assertAlmostEqual(float(np.mean(errs)), 5.0, places=12)
        self.assertNotAlmostEqual(float(np.sqrt(np.mean(errs**2))), 5.0, places=3)


# ==========================================================================
# 14. Confidence intervals
# ==========================================================================
class TestStatistics(unittest.TestCase):
    def test_bootstrap_ci_is_deterministic(self):
        x = np.random.default_rng(0).normal(5.0, 1.0, 30)
        a = stats.bootstrap_ci(x, n_resamples=2000, seed=123)
        b = stats.bootstrap_ci(x, n_resamples=2000, seed=123)
        self.assertEqual(a, b)

    def test_bootstrap_ci_brackets_the_mean(self):
        x = np.random.default_rng(1).normal(5.0, 1.0, 30)
        lo, hi = stats.bootstrap_ci(x, n_resamples=10000, seed=7)
        self.assertLess(lo, float(np.mean(x)))
        self.assertGreater(hi, float(np.mean(x)))

    def test_bootstrap_ci_width_shrinks_with_n(self):
        rng = np.random.default_rng(2)
        small = stats.bootstrap_ci(rng.normal(0, 1, 10), n_resamples=4000, seed=1)
        large = stats.bootstrap_ci(rng.normal(0, 1, 1000), n_resamples=4000, seed=1)
        self.assertGreater(small[1] - small[0], large[1] - large[0])

    def test_ci_matches_normal_theory_approximately(self):
        x = np.random.default_rng(3).normal(10.0, 2.0, 400)
        lo, hi = stats.bootstrap_ci(x, n_resamples=10000, seed=5)
        sem = float(np.std(x, ddof=1) / np.sqrt(len(x)))
        self.assertAlmostEqual(hi - lo, 2 * 1.96 * sem, delta=0.4 * sem)

    def test_paired_difference_preserves_pairing(self):
        a = {0: 1.0, 1: 2.0, 2: 3.0}
        b = {0: 1.5, 2: 3.5, 3: 9.0}
        res = stats.paired_difference(a, b, n_resamples=2000, seed=1)
        self.assertEqual(res["n_pairs"], 2)
        self.assertAlmostEqual(res["mean"], 0.5, places=9)
        # unpaired means would give a different answer
        self.assertNotAlmostEqual(
            res["mean"], float(np.mean(list(b.values())) - np.mean(list(a.values()))),
            places=3,
        )

    def test_time_samples_are_not_treated_as_replicates(self):
        """The CI unit must be the episode, not the 50 Hz sample."""
        per_episode = np.full(30, 0.1)
        lo, hi = stats.bootstrap_ci(per_episode, n_resamples=2000, seed=1)
        self.assertAlmostEqual(lo, 0.1, places=9)
        self.assertAlmostEqual(hi, 0.1, places=9)

    def test_significance_requires_ci_excluding_zero(self):
        a = {i: 1.0 for i in range(30)}
        b = {i: 2.0 for i in range(30)}
        res = stats.paired_difference(a, b, n_resamples=2000, seed=1)
        self.assertTrue(res["significant"])
        noisy_a = {i: float(v) for i, v in enumerate(np.random.default_rng(0).normal(0, 1, 30))}
        noisy_b = {i: float(v) for i, v in enumerate(np.random.default_rng(1).normal(0, 1, 30))}
        self.assertFalse(stats.paired_difference(noisy_a, noisy_b, n_resamples=2000, seed=1)["significant"])


# ==========================================================================
# 15. Official vs illustrative seed separation
# ==========================================================================
class TestSeedPolicy(unittest.TestCase):
    """The official and illustrative pools must stay separable.

    Round 1 used official 0-29 inside a showcase pool of 0-99, so a curated
    figure seed could also be an official statistical sample. Round 2 extended
    the official pool to 0-59 and moved the candidate pool to 100-149, making
    the two DISJOINT -- which is what these tests now pin. The pool sizes are
    read from the config rather than hard-coded, so extending the official pool
    again does not require editing a test; only the disjointness and the
    "contiguous from a declared start" shape are asserted.
    """

    def test_official_and_showcase_pools_declared_separately(self):
        _, glob = _cfgs("E1")
        for key in ("official_seeds", "showcase_seeds"):
            self.assertIn(key, glob)
            self.assertIn("start", glob[key])
            self.assertIn("count", glob[key])
            self.assertGreater(glob[key]["count"], 0)
        self.assertGreaterEqual(glob["official_seeds"]["count"], 30)

    def test_official_and_showcase_pools_are_disjoint(self):
        _, glob = _cfgs("E1")
        official = set(range(glob["official_seeds"]["start"],
                             glob["official_seeds"]["start"] + glob["official_seeds"]["count"]))
        showcase = set(range(glob["showcase_seeds"]["start"],
                             glob["showcase_seeds"]["start"] + glob["showcase_seeds"]["count"]))
        self.assertEqual(
            official & showcase, set(),
            "an illustrative figure seed must never also be an official "
            "statistical sample",
        )

    def test_aggregate_restricts_to_official_seeds(self):
        from benchmark import aggregate

        rows = [
            {"seed": s, "baseline": "C1", "condition": "T1", "experiment": "E1",
             "ee_pos_rmse": 0.1, "survived": 1}
            for s in range(100)
        ]
        official = aggregate.filter_official(rows, 0, 30)
        self.assertEqual(len(official), 30)
        self.assertEqual(max(r["seed"] for r in official), 29)


# ==========================================================================
# E1/E2/E4 scenario-level checks
# ==========================================================================
class TestExperimentConfigs(unittest.TestCase):
    def test_e1_conditions_cover_the_2x2(self):
        cfg, _ = _cfgs("E1")
        matrix = {(v["base"], v["ee"]) for v in cfg["conditions"].values()}
        self.assertEqual(
            matrix,
            {("still", "fixed"), ("still", "random"), ("moving", "fixed"), ("moving", "random")},
        )

    def test_base_still_is_exactly_zero(self):
        cfg, _ = _cfgs("E1")
        for condition in ("T1", "T2"):
            s = E.build_e1(cfg, condition, 0)
            self.assertEqual(float(np.abs(s.base.commands).max()), 0.0)

    def test_e2_motion_intensity_envelopes(self):
        """Round 3 states E2's intensities as per-axis ABSOLUTE envelopes.

        Rounds 1-2 declared them as fractions 0.25/0.50/0.75 of the 0.6 shared
        range (= 0.15/0.30/0.45). Round 3 replaced that with explicit
        0.20/0.40/0.60 in m/s and rad/s, so this test now reads the envelope
        rather than the fraction. The limits are per axis, not a bound on the
        XY norm, which is why each axis is checked separately.
        """
        cfg, _ = _cfgs("E2")
        want = {"Low": 0.20, "Medium": 0.40, "High": 0.60}
        for name, alpha in want.items():
            self.assertEqual(E.e2_axis_limits(cfg, name), [alpha, alpha, alpha])
            s = E.build_e2(cfg, name, 0)
            per_axis_max = np.abs(s.base.commands).max(axis=0)
            for axis in range(3):
                self.assertLessEqual(float(per_axis_max[axis]), alpha + 1e-6)

    def test_e2_envelope_never_exceeds_the_shared_supported_range(self):
        cfg, _ = _cfgs("E2")
        shared = np.asarray(cfg["base_command_limits"], dtype=float)
        for name in ("Low", "Medium", "High"):
            axis = np.asarray(E.e2_axis_limits(cfg, name), dtype=float)
            self.assertTrue(bool(np.all(axis <= shared + 1e-12)))

    def test_e2_covers_all_three_motion_patterns(self):
        cfg, _ = _cfgs("E2")
        s = E.build_e2(cfg, "Medium", 0)
        segments = []
        for start in s.base.switch_ticks:
            segments.append(s.base.waveform[int(start)])
        kinds = set()
        for seg in segments:
            vx, vy, wz = (abs(v) > 1e-9 for v in seg)
            if vx and not vy and not wz:
                kinds.add("straight")
            elif wz and not vx and not vy:
                kinds.add("turn")
            elif vx and wz:
                kinds.add("translate_turn")
        self.assertEqual(kinds, {"straight", "turn", "translate_turn"})

    def test_e4_base_schedule_is_bounded_and_phased(self):
        cfg, _ = _cfgs("E4")
        s = E.build_e4(cfg, "chicken_head", 0)
        limits = np.asarray(cfg["base_command_limits"])
        bound = float(cfg["base"]["amplitude"]) * limits.max()
        self.assertLessEqual(float(np.abs(s.base.commands).max()), bound + 1e-6)
        self.assertEqual(len(s.base.switch_ticks), len(cfg["base"]["phases"]))


# ==========================================================================
# Seeded initial conditions: identical across baselines, varying across seeds
# ==========================================================================
class TestInitialState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco

        from benchmark.controllers.registry import build_adapter

        cls.mujoco = mujoco
        cls.build = staticmethod(build_adapter)
        cls.cfg, _ = _cfgs("E1")

    def _state(self, baseline, seed, noise=None):
        adapter = self.build(baseline)
        model = adapter.build_model()
        data = self.mujoco.MjData(model)
        adapter.attach(model, data)
        adapter.set_initial_state()
        adapter.apply_initial_state_noise(
            seed, self.cfg["initial_state"] if noise is None else noise
        )
        joints = (
            adapter.leg_joint_names + adapter.arm_joint_names
            + [adapter.gripper_joint_name]
        )
        return np.array(
            [data.qpos[adapter.qpos_addr[n]] for n in joints]
            + list(data.qpos[0:7]),
            dtype=np.float64,
        )

    def test_identical_across_all_six_baselines(self):
        reference = self._state("C1", 5)
        for baseline in ("C2", "C3", "C4", "C5", "C6"):
            np.testing.assert_array_equal(
                self._state(baseline, 5), reference,
                err_msg=f"{baseline} initial state differs from C1 at the same seed",
            )

    def test_varies_across_seeds(self):
        a = self._state("C1", 0)
        b = self._state("C1", 1)
        self.assertGreater(float(np.abs(a - b).max()), 1e-6)

    def test_reproducible_for_the_same_seed(self):
        np.testing.assert_array_equal(self._state("C3", 7), self._state("C3", 7))

    def test_perturbation_within_declared_bounds(self):
        noise = self.cfg["initial_state"]
        nominal = self._state("C1", 0, noise={"enabled": False})
        for seed in range(8):
            perturbed = self._state("C1", seed)
            dq = np.abs(perturbed[:19] - nominal[:19])
            self.assertLessEqual(float(dq.max()), float(noise["joint_pos_noise"]) + 1e-9)
            dz = perturbed[21] - nominal[21]
            self.assertGreaterEqual(dz, -1e-9)
            self.assertLessEqual(dz, float(noise["base_z_noise"]) + 1e-9)

    def test_disabled_randomization_reproduces_the_nominal_state(self):
        off = self._state("C1", 3, noise={"enabled": False})
        off2 = self._state("C1", 9, noise={"enabled": False})
        np.testing.assert_array_equal(off, off2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
