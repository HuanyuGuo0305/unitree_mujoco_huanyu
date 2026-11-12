#!/usr/bin/env python3
"""Unit tests for the E1 command protocol, cross-baseline command identity, the
statistics conventions and the atomic writers (tests the CODE and CONFIG, not results).

Run:  python3 -m unittest benchmark.tests.test_protocol     (from simulate_python/)
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for _root in (REPO_ROOT, SIM_ROOT):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from benchmark.core import stats  # noqa: E402
from benchmark.round3 import common  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

DT = 0.02


def _cfg(experiment):
    return E.load_config(experiment)[0]


class TestE1BaseResampling(unittest.TestCase):
    """Validation item 1: base commands resample every 2-3 s."""

    def test_config_declares_two_to_three_seconds(self):
        self.assertEqual(_cfg("E1")["base"]["segment_duration_range"], [2.0, 3.0])

    def test_every_segment_duration_is_within_two_to_three_seconds(self):
        cfg = _cfg("E1")
        for condition in ("T3", "T4"):
            for seed in range(0, 50, 7):
                s = E.build_e1(cfg, condition, seed)
                d = s.base.segment_durations_s
                self.assertGreaterEqual(float(d.min()), 2.0 - 1e-9)
                self.assertLessEqual(float(d.max()), 3.0 + 1e-9)

    def test_switch_intervals_measured_from_the_command_signal_agree(self):
        """Not just the declared durations: the realised command signal too."""
        cfg = _cfg("E1")
        s = E.build_e1(cfg, "T4", 3)
        cmd = s.base.commands
        switches = np.nonzero(np.any(np.abs(np.diff(cmd, axis=0)) > 0, axis=1))[0] + 1
        intervals = np.diff(np.concatenate([[0], switches])) * DT
        self.assertGreaterEqual(float(intervals.min()), 2.0 - 1e-9)
        self.assertLessEqual(float(intervals.max()), 3.0 + 1e-9)

    def test_base_still_conditions_are_exactly_zero(self):
        cfg = _cfg("E1")
        for condition in ("T1", "T2"):
            for seed in (0, 25, 49):
                s = E.build_e1(cfg, condition, seed)
                self.assertEqual(float(np.abs(s.base.commands).max()), 0.0)


class TestE1EEResampling(unittest.TestCase):
    """Validation items 2 and 3: EE targets replaced every 3 s, no interpolation."""

    def test_config_declares_three_seconds_and_no_interpolation(self):
        ee = _cfg("E1")["ee"]
        self.assertEqual(float(ee["resample_interval_s"]), 3.0)
        self.assertEqual(ee["interpolation"], "none")

    def test_inter_switch_interval_is_exactly_three_seconds(self):
        cfg = _cfg("E1")
        for seed in (0, 11, 49):
            s = E.build_e1(cfg, "T4", seed)
            cmd = s.ee.commands
            switches = np.nonzero(np.any(np.abs(np.diff(cmd, axis=0)) > 0, axis=1))[0] + 1
            intervals = np.diff(switches) * DT
            self.assertTrue(np.allclose(intervals, 3.0, atol=1e-12))

    def test_first_hold_is_one_control_step_short_by_native_convention(self):
        """The native sample-and-hold counter is index = (k+1)//cycle_steps.

        That makes the FIRST hold 149 ticks (2.98 s) and every later one
        exactly 150 ticks (3.00 s). This is the validated native behaviour,
        preserved deliberately rather than "corrected" -- and it is identical
        for every baseline, so it cannot bias a comparison. The test pins it so
        the convention cannot drift unnoticed.
        """
        s = E.build_e1(_cfg("E1"), "T4", 0)
        cmd = s.ee.commands
        switches = np.nonzero(np.any(np.abs(np.diff(cmd, axis=0)) > 0, axis=1))[0] + 1
        self.assertEqual(int(switches[0]), 149)
        self.assertAlmostEqual(float(switches[0] * DT), 2.98, places=12)

    def test_command_is_piecewise_constant_with_hard_switches(self):
        """No interpolation: between switches the 9-D command does not change
        at all, and at a switch it changes in one control step."""
        s = E.build_e1(_cfg("E1"), "T4", 5)
        cmd = s.ee.commands
        diffs = np.abs(np.diff(cmd, axis=0)).max(axis=1)
        changed = diffs > 0
        self.assertEqual(int(changed.sum()), 9)
        # Every non-switch step is bit-identical, not merely small.
        self.assertEqual(float(diffs[~changed].max()), 0.0)

    def test_ee_fixed_target_is_a_single_constant_row(self):
        for seed in (0, 49):
            s = E.build_e1(_cfg("E1"), "T3", seed)
            self.assertEqual(len(np.unique(s.ee.commands, axis=0)), 1)

    def test_targets_come_from_the_native_keypoint_table(self):
        cfg = _cfg("E1")
        table = E.keypoint_table(cfg["_global"])
        s = E.build_e1(cfg, "T4", 2)
        for row in np.unique(s.ee.commands, axis=0):
            self.assertTrue(
                bool(np.any(np.all(np.abs(table.table - row) < 1e-6, axis=1))),
                "an EE target is not a row of the native command table",
            )


class TestIdenticalCommandsAcrossBaselines(unittest.TestCase):
    """Validation item 7: the schedule depends on (experiment, condition, seed)
    only, so every compared baseline is handed bit-identical commands."""

    def test_schedules_are_a_function_of_condition_and_seed_alone(self):
        for experiment, conditions in (("E1", ("T1", "T4")), ("E2", ("dynamic",)),
                                       ("E3", ("B0T0", "B4T4"))):
            cfg = _cfg(experiment)
            for condition in conditions:
                a = E.build_scenario(experiment, cfg, condition, 9).arrays()
                b = E.build_scenario(experiment, cfg, condition, 9).arrays()
                for key in a:
                    self.assertTrue(np.array_equal(a[key], b[key]),
                                    f"{experiment}/{condition}/{key} not reproducible")

    def test_changing_the_ee_condition_does_not_perturb_the_base_schedule(self):
        """T1 vs T2 and T3 vs T4 share a base schedule, which is what makes the
        EE-condition comparison paired."""
        cfg = _cfg("E1")
        for still, moving in (("T1", "T2"), ("T3", "T4")):
            a = E.build_e1(cfg, still, 6).base.commands
            b = E.build_e1(cfg, moving, 6).base.commands
            self.assertTrue(np.array_equal(a, b))


class TestStatisticsConventions(unittest.TestCase):
    def test_paired_advantage_sign_convention(self):
        """advantage = rival - reference, so positive means C1 (reference) won."""
        ref = {s: 1.0 for s in range(10)}
        rival = {s: 1.5 for s in range(10)}
        res = stats.paired_advantage(ref, rival, n_resamples=200, seed=1)
        self.assertAlmostEqual(res["mean"], 0.5)
        self.assertEqual(res["frac_positive"], 1.0)

    def test_pairing_uses_matching_seed_ids_only(self):
        ref = {0: 1.0, 1: 2.0, 2: 3.0}
        rival = {1: 2.5, 2: 3.5, 7: 99.0}
        res = stats.paired_advantage(ref, rival, n_resamples=200, seed=1)
        self.assertEqual(res["n_pairs"], 2)
        self.assertEqual(res["seeds"], [1, 2])

    def test_bootstrap_is_deterministic_across_calls(self):
        values = list(np.random.default_rng(1).normal(size=40))
        a = stats.bootstrap_ci(values, 10000, 20261106)
        b = stats.bootstrap_ci(values, 10000, 20261106)
        self.assertEqual(a, b)


class TestAtomicWrites(unittest.TestCase):
    def test_csv_write_is_atomic_and_leaves_no_temp_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x", "out.csv")
            common.write_csv(path, [{"a": 1.5, "b": "t"}])
            self.assertTrue(os.path.isfile(path))
            self.assertFalse(os.path.exists(path + ".tmp"))
            self.assertEqual(common.read_csv(path)[0]["b"], "t")

    def test_floats_are_written_at_full_precision(self):
        import tempfile

        value = 0.1234567890123456789
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            common.write_csv(path, [{"v": value}])
            self.assertEqual(float(common.read_csv(path)[0]["v"]), value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
