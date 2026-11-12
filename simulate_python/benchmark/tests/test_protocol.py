#!/usr/bin/env python3
"""Unit tests for the round-3 protocol changes and the round-3 analysis code.

These test the CODE and the CONFIG, not the measured results. The measured
results are checked by ``round3/validate.py``, which reads the actual output
files.

Run:  python3 benchmark/tests/test_round3_protocol.py
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
from benchmark.round3 import common, export, select3  # noqa: E402
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


class TestE2Envelopes(unittest.TestCase):
    """Validation item 4: the new 0.20 / 0.40 / 0.60 per-axis envelopes."""

    def test_axis_limits(self):
        cfg = _cfg("E2")
        for name, alpha in (("Low", 0.20), ("Medium", 0.40), ("High", 0.60)):
            self.assertEqual(E.e2_axis_limits(cfg, name), [alpha, alpha, alpha])

    def test_commands_respect_each_axis_separately(self):
        cfg = _cfg("E2")
        for name, alpha in (("Low", 0.20), ("Medium", 0.40), ("High", 0.60)):
            for seed in range(0, 50, 11):
                cmd = E.build_e2(cfg, name, seed).base.commands
                self.assertLessEqual(float(np.abs(cmd).max()), alpha + 1e-6)

    def test_envelope_is_per_axis_not_a_norm_bound(self):
        """A translate+turn segment may command |vx| = |vy| near the limit at
        the same time, so the XY norm can exceed the per-axis value. That is
        intended; this pins it so a future 'fix' cannot silently add a norm
        constraint the brief did not ask for."""
        cfg = _cfg("E2")
        worst = 0.0
        for seed in range(50):
            cmd = E.build_e2(cfg, "High", seed).base.commands
            worst = max(worst, float(np.hypot(cmd[:, 0], cmd[:, 1]).max()))
        self.assertGreater(worst, 0.60 + 1e-6)

    def test_scaling_changes_only_the_amplitude(self):
        """At one seed the three intensities must be the SAME waveform."""
        cfg = _cfg("E2")
        lo = E.build_e2(cfg, "Low", 4).base.commands
        hi = E.build_e2(cfg, "High", 4).base.commands
        self.assertTrue(np.allclose(hi * (0.20 / 0.60), lo, atol=1e-6))

    def test_rejects_an_envelope_outside_the_shared_supported_range(self):
        cfg = _cfg("E2")
        bad = {k: dict(v) for k, v in cfg["base"]["motion_intensity"].items()}
        bad["High"]["max_abs_vx_mps"] = 0.9
        cfg = dict(cfg)
        cfg["base"] = dict(cfg["base"], motion_intensity=bad)
        with self.assertRaises(ValueError):
            E.e2_axis_limits(cfg, "High")


class TestSeedPolicy(unittest.TestCase):
    """Validation item 6: exactly 50 official seeds, 0-49."""

    def test_official_pool_is_zero_to_fortynine(self):
        seeds = common.official_seeds()
        self.assertEqual(len(seeds), 50)
        self.assertEqual(seeds[0], 0)
        self.assertEqual(seeds[-1], 49)

    def test_official_and_showcase_pools_are_disjoint(self):
        self.assertFalse(set(common.official_seeds()) & set(common.showcase_seeds()))


class TestIdenticalCommandsAcrossBaselines(unittest.TestCase):
    """Validation item 7: the schedule depends on (experiment, condition, seed)
    only, so every compared baseline is handed bit-identical commands."""

    def test_schedules_are_a_function_of_condition_and_seed_alone(self):
        for experiment, conditions in (("E1", ("T1", "T4")),
                                       ("E2", ("Low", "High"))):
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


class TestIntervalDiagnostics(unittest.TestCase):
    """Validation item 8: the full-window RMSE includes the switch transients."""

    def test_intervals_tile_the_window_without_gaps_or_overlap(self):
        arr = {
            "t": np.arange(10) * DT + 3.0,
            "vx_cmd": np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=float),
            "vy_cmd": np.zeros(10), "wz_cmd": np.zeros(10),
            "ee_cmd_kp": np.zeros((10, 9)),
        }
        bounds = export._interval_bounds(arr, "base_segment")
        self.assertEqual(bounds, [(0, 3), (3, 7), (7, 10)])
        self.assertEqual(sum(e - s for s, e in bounds), 10)

    def test_first_interval_starts_at_the_window_opening(self):
        arr = {
            "t": np.arange(6) * DT + 3.0,
            "vx_cmd": np.array([5, 5, 7, 7, 7, 7], dtype=float),
            "vy_cmd": np.zeros(6), "wz_cmd": np.zeros(6),
            "ee_cmd_kp": np.zeros((6, 9)),
        }
        bounds = export._interval_bounds(arr, "base_segment")
        self.assertEqual(bounds[0][0], 0)

    def test_weighted_recombination_reproduces_the_full_window_rmse(self):
        """The ONLY correct way to recombine interval RMSEs is the
        sample-count-weighted quadratic mean. An unweighted mean does not
        reproduce the full-window value, which is why the headline metric is
        never an unweighted average of intervals."""
        rng = np.random.default_rng(0)
        err = rng.normal(0, 1, 100)
        bounds = [(0, 30), (30, 55), (55, 100)]
        full = np.sqrt(np.mean(err**2))
        per = [np.sqrt(np.mean(err[s:e] ** 2)) for s, e in bounds]
        ns = [e - s for s, e in bounds]
        weighted = np.sqrt(sum(n * r**2 for n, r in zip(ns, per)) / sum(ns))
        self.assertAlmostEqual(weighted, full, places=12)
        self.assertNotAlmostEqual(float(np.mean(per)), full, places=3)


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


class TestSelectionProcedure(unittest.TestCase):
    def test_advantage_requires_beating_every_rival(self):
        pb = {"C1": {"m": 1.0}, "C3": {"m": 2.0}, "C5": {"m": 0.5}}
        self.assertEqual(select3._advantage(pb, "m", ["C3", "C5"]), -0.5)

    def test_min_combine_cannot_be_carried_by_one_term(self):
        case = {"terms": [("a", 1.0), ("b", 1.0)], "combine": "min"}
        pb = {"C1": {"a": 0.0, "b": 1.0}, "C3": {"a": 100.0, "b": 0.9}}
        score, _ = select3._score(case, pb, ["C3"], {"a": 1.0, "b": 1.0})
        self.assertAlmostEqual(score, -0.1)

    def test_require_negative_gate_selects_a_contradicting_trial(self):
        """The E2 trade-off criterion must REJECT a trial where C1 also wins
        on tracking -- it exists to surface a case that contradicts a uniform
        C1 advantage."""
        case = {"terms": [("p", 1.0)], "combine": "mean",
                "require_positive": ["p"], "require_negative": ["t"]}
        scale = {"p": 1.0, "t": 1.0}
        good = {"C1": {"p": 1.0, "t": 2.0, "survived": 1},
                "C2": {"p": 2.0, "t": 1.0, "survived": 1}}
        bad = {"C1": {"p": 1.0, "t": 1.0, "survived": 1},
               "C2": {"p": 2.0, "t": 2.0, "survived": 1}}
        cfg = {"min_base_segments": 0, "min_ee_switches": 0,
               "min_rms_fraction_of_condition_median": 0.0}
        _s, terms = select3._score(case, good, ["C2"], scale)
        self.assertEqual(select3._gates(case, good, ["C2"], None, scale, cfg, terms), [])
        _s, terms = select3._score(case, bad, ["C2"], scale)
        self.assertTrue(any("require_negative" in r for r in
                            select3._gates(case, bad, ["C2"], None, scale, cfg, terms)))

    def test_terminated_episode_is_gated_out(self):
        case = {"terms": [("m", 1.0)], "combine": "mean", "require_positive": []}
        pb = {"C1": {"m": 1.0, "survived": 1, "termination_reason": "completed"},
              "C3": {"m": 2.0, "survived": 0, "termination_reason": "base_height"}}
        cfg = {"min_base_segments": 0, "min_ee_switches": 0,
               "min_rms_fraction_of_condition_median": 0.0}
        _s, terms = select3._score(case, pb, ["C3"], {"m": 1.0})
        self.assertTrue(any(r.startswith("G1_") for r in
                            select3._gates(case, pb, ["C3"], None, {"m": 1.0}, cfg, terms)))

    def test_nontriviality_threshold_is_condition_relative(self):
        """An ABSOLUTE commanded-motion cut would reject a whole low-intensity
        condition by construction; the round-2 build did exactly that. The rule
        must compare against the condition's own pool median."""
        self.assertIn("min_rms_fraction_of_condition_median", select3.NONTRIVIAL["E1"])
        self.assertIn("min_rms_fraction_of_condition_median", select3.NONTRIVIAL["E2"])


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
