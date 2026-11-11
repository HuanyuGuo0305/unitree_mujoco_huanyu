#!/usr/bin/env python3
"""Unit tests for the round-2 additions.

Covers the sign conventions, the pairing, the illustrative-seed selection
procedure and the pool-disjointness guard. These are the places where a silent
error would corrupt a reported conclusion rather than crash.

Run:  python3 benchmark/tests/test_round2_analysis.py
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

from benchmark import analysis, select_cases  # noqa: E402
from benchmark.core import stats  # noqa: E402
from benchmark.plots import figures_round2 as figs  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

CFG_STATS = {"bootstrap_resamples": 2000, "bootstrap_seed": 11, "confidence": 0.95}


def _e3_rows(c1_ee, c3_ee, c1_base, c3_base, n_seeds=20,
             families=("line", "circle", "semicircle")):
    """Synthetic E3 rows with a known per-seed structure."""
    rows = []
    rng = np.random.default_rng(0)
    for family in families:
        for seed in range(n_seeds):
            jitter = float(rng.normal(0.0, 1e-4))
            for ab in (0.0, 0.5, 1.0):
                for ae in (0.0, 0.5, 1.0):
                    for baseline, ee, base in (("C1", c1_ee, c1_base),
                                               ("C3", c3_ee, c3_base)):
                        rows.append({
                            "experiment": "E3", "baseline": baseline, "seed": seed,
                            "ee_family": family, "alpha_base": ab, "alpha_ee": ae,
                            "condition": f"S{int(ab*2)}{int(ae*2)}_{family}",
                            "survived": 1, "termination_reason": "completed",
                            "ee_pos_rmse": ee + 0.01 * ab + jitter,
                            "base_vxy_rmse": base + 0.02 * ae + jitter,
                            "ee_ori_mae_deg": 10.0, "base_wz_rmse": 0.05,
                        })
    return rows


# ==========================================================================
class TestPairedAdvantage(unittest.TestCase):
    def test_positive_means_reference_better(self):
        ref = {i: 0.10 for i in range(20)}
        rival = {i: 0.15 for i in range(20)}
        res = stats.paired_advantage(ref, rival, n_resamples=500, seed=1)
        self.assertAlmostEqual(res["mean"], 0.05, places=9)
        self.assertEqual(res["frac_positive"], 1.0)
        self.assertTrue(res["significant"])

    def test_negative_means_rival_better(self):
        ref = {i: 0.20 for i in range(20)}
        rival = {i: 0.15 for i in range(20)}
        res = stats.paired_advantage(ref, rival, n_resamples=500, seed=1)
        self.assertLess(res["mean"], 0.0)
        self.assertEqual(res["frac_positive"], 0.0)

    def test_frac_positive_counts_trials_not_magnitude(self):
        # One huge win, nineteen small losses: the mean favours the reference,
        # the win fraction does not. Both must be reported.
        ref = {i: 0.10 for i in range(20)}
        rival = {0: 5.0}
        rival.update({i: 0.09 for i in range(1, 20)})
        res = stats.paired_advantage(ref, rival, n_resamples=500, seed=1)
        self.assertGreater(res["mean"], 0.0)
        self.assertAlmostEqual(res["frac_positive"], 0.05, places=9)
        self.assertLess(res["median"], 0.0)

    def test_pairing_is_preserved(self):
        # Same marginal distributions, opposite pairing: an unpaired test would
        # see no difference, the paired one must.
        ref = {i: float(i) for i in range(10)}
        rival = {i: float(i) + 1.0 for i in range(10)}
        res = stats.paired_advantage(ref, rival, n_resamples=500, seed=1)
        self.assertAlmostEqual(res["mean"], 1.0, places=9)
        self.assertTrue(res["significant"])

    def test_only_shared_seeds_are_used(self):
        res = stats.paired_advantage({1: 1.0, 2: 2.0}, {2: 3.0, 3: 4.0},
                                     n_resamples=200, seed=1)
        self.assertEqual(res["n_pairs"], 1)

    def test_zero_difference_is_not_significant(self):
        ref = {i: 0.1 for i in range(20)}
        res = stats.paired_advantage(ref, dict(ref), n_resamples=500, seed=1)
        self.assertFalse(res["significant"])
        self.assertEqual(res["mean"], 0.0)


class TestAnalysisTable(unittest.TestCase):
    def test_advantage_sign_matches_definition(self):
        rows = _e3_rows(c1_ee=0.10, c3_ee=0.13, c1_base=0.08, c3_base=0.09)
        table = analysis.paired_advantage_table(rows, "C1", "C3", CFG_STATS)
        ee = [r for r in table if r["quantity"] == "A_EE" and r["cell"] == "S22"
              and not r["pooled_over_families"]]
        self.assertTrue(ee)
        for row in ee:
            self.assertAlmostEqual(row["mean_advantage"], 0.03, places=3)
            self.assertEqual(row["favours"], "C1")
            self.assertEqual(row["advantage_definition"],
                             "C3_ee_pos_rmse - C1_ee_pos_rmse")

    def test_rival_better_is_reported_as_such(self):
        rows = _e3_rows(c1_ee=0.15, c3_ee=0.10, c1_base=0.08, c3_base=0.09)
        table = analysis.paired_advantage_table(rows, "C1", "C3", CFG_STATS)
        ee = [r for r in table if r["quantity"] == "A_EE" and r["cell"] == "S11"][0]
        self.assertLess(ee["mean_advantage"], 0.0)
        self.assertEqual(ee["favours"], "C3")

    def test_pooled_rows_have_three_times_the_pairs(self):
        rows = _e3_rows(c1_ee=0.10, c3_ee=0.13, c1_base=0.08, c3_base=0.09, n_seeds=12)
        table = analysis.paired_advantage_table(rows, "C1", "C3", CFG_STATS)
        per_family = [r for r in table if r["quantity"] == "A_EE"
                      and r["cell"] == "S22" and r["ee_family"] == "line"][0]
        pooled = [r for r in table if r["quantity"] == "A_EE"
                  and r["cell"] == "S22" and r["pooled_over_families"]][0]
        self.assertEqual(per_family["n_pairs"], 12)
        self.assertEqual(pooled["n_pairs"], 36)
        self.assertTrue(pooled["pooled_over_families"])

    def test_degradation_quantities_are_present_and_signed(self):
        # c1_ee has +0.01*alpha_base baked in, so BOTH methods degrade by the
        # same +0.01 and the advantage must be ~0, not the absolute gap.
        rows = _e3_rows(c1_ee=0.10, c3_ee=0.13, c1_base=0.08, c3_base=0.09)
        # load_e3 reads from disk, so the degradation merge is exercised here
        # through the same helper it calls.
        from benchmark import aggregate
        degradation = aggregate.e3_degradation(rows, None)
        index = {(r["baseline"], r["ee_family"], r["seed"],
                  r["alpha_base"], r["alpha_ee"]): r for r in degradation}
        for row in rows:
            extra = index.get((row["baseline"], row["ee_family"], row["seed"],
                               row["alpha_base"], row["alpha_ee"]))
            if extra and "delta_ee_pos_rmse_from_base" in extra:
                row["delta_ee_pos_rmse_from_base"] = extra["delta_ee_pos_rmse_from_base"]
                row["delta_base_vxy_rmse_from_ee"] = extra["delta_base_vxy_rmse_from_ee"]
        table = analysis.paired_advantage_table(rows, "C1", "C3", CFG_STATS)
        deg = [r for r in table if r["quantity"] == "A_Delta_EE"
               and r["cell"] == "S22" and r["ee_family"] == "line"]
        self.assertTrue(deg)
        self.assertAlmostEqual(deg[0]["mean_advantage"], 0.0, places=6)

    def test_absolute_table_covers_the_full_matrix(self):
        rows = _e3_rows(c1_ee=0.10, c3_ee=0.13, c1_base=0.08, c3_base=0.09, n_seeds=5)
        table = analysis.absolute_tracking_table(rows, CFG_STATS)
        self.assertEqual(len(table), 3 * 9 * 2)   # families x cells x baselines
        demanding = {r["cell"] for r in table if r["demanding_cell"]}
        self.assertEqual(demanding, {"S11", "S12", "S21", "S22"})


class TestFigureSeriesHelpers(unittest.TestCase):
    def test_family_pooling_averages_before_bootstrapping(self):
        rows = _e3_rows(c1_ee=0.10, c3_ee=0.13, c1_base=0.08, c3_base=0.09, n_seeds=8)
        subset = figs._cell_rows(rows, ab=1.0, ae=1.0, baseline="C1")
        per_seed = figs._per_seed_pooled(subset, "ee_pos_rmse")
        # One value per seed, not one per episode.
        self.assertEqual(len(per_seed), 8)
        self.assertEqual(len(subset), 8 * 3)

    def test_paired_series_differences_before_pooling(self):
        rows = _e3_rows(c1_ee=0.10, c3_ee=0.13, c1_base=0.08, c3_base=0.09, n_seeds=8)
        mean, lo, hi, n, sig = figs._paired_series_pooled(
            rows, "C1", "C3", "ee_pos_rmse", "ab", [1.0], {"ae": 1.0}, CFG_STATS)
        self.assertEqual(int(n[0]), 8)
        self.assertAlmostEqual(float(mean[0]), 0.03, places=3)
        self.assertTrue(sig[0])

    def test_cell_filter_is_numeric_safe(self):
        rows = _e3_rows(c1_ee=0.1, c3_ee=0.1, c1_base=0.1, c3_base=0.1, n_seeds=2)
        for row in rows:
            row["alpha_base"] = str(row["alpha_base"])
        self.assertEqual(len(figs._cell_rows(rows, ab=0.5, ae=0.5, baseline="C1")), 6)


class TestPowerReconstruction(unittest.TestCase):
    """The 250 Hz power curve must agree with the episode metric it averages to.

    A slice that is one block off reads accelerations as velocities and
    inflates the curve by orders of magnitude while still producing a
    plausible-looking plot, so the reconstruction is checked numerically rather
    than by eye.
    """

    @staticmethod
    def _write_episode(tmp, n_leg=12, n_arm=6, n_rows=50, shift=0):
        import json

        rng = np.random.default_rng(3)
        n_total = n_leg + n_arm
        qpos = rng.normal(0, 1, (n_rows, n_total))
        qvel = rng.normal(0, 2, (n_rows, n_total))
        qacc = rng.normal(0, 50, (n_rows, n_total))
        tau = rng.normal(0, 10, (n_rows, n_total))
        t = np.arange(n_rows) * 0.004
        data = np.column_stack([t, qpos, qvel, qacc, tau]).astype(np.float32)
        np.savez_compressed(os.path.join(tmp, "physics.npz"), data=data)
        used = qacc if shift else qvel
        metrics = {
            "leg_power": float(np.mean(np.sum(np.abs(
                tau[:, :n_leg] * used[:, :n_leg]), axis=1))),
            "arm_power": float(np.mean(np.sum(np.abs(
                tau[:, n_leg:] * used[:, n_leg:]), axis=1))),
        }
        with open(os.path.join(tmp, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle)
        return metrics

    def test_reconstruction_matches_the_episode_metric(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            metrics = self._write_episode(tmp)
            result = figs._leg_arm_power(tmp)
            self.assertAlmostEqual(float(np.mean(result["leg_power"])),
                                   metrics["leg_power"], places=3)
            self.assertAlmostEqual(float(np.mean(result["arm_power"])),
                                   metrics["arm_power"], places=3)
            self.assertEqual(result["n_leg"], 12)

    def test_mismatch_raises_instead_of_plotting_a_wrong_curve(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            # metrics computed from the ACCELERATION block: the reconstruction
            # must refuse rather than silently disagree.
            self._write_episode(tmp, shift=1)
            with self.assertRaises(ValueError):
                figs._leg_arm_power(tmp)

    def test_missing_physics_log_returns_none(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(figs._leg_arm_power(tmp))


class TestSelectionProcedure(unittest.TestCase):
    @staticmethod
    def _candidates(values):
        """values: {seed: {baseline: {metric: value}}}"""
        out = []
        for seed, per_baseline in values.items():
            out.append({
                "experiment": "E1", "condition": "T4", "seed": seed,
                "gated_out": [], "metrics": per_baseline,
                "survived": {b: 1 for b in per_baseline},
                "paths": {b: f"/tmp/{b}/{seed}" for b in per_baseline},
                "commands": {"n_base_segments": 9, "n_ee_command_changes": 5,
                             "rms_commanded_fraction_of_range": 0.5},
            })
        return out

    def _spec(self):
        return select_cases.CRITERIA["E1"]

    def test_scores_rank_the_largest_worst_case_margin_first(self):
        base = {"base_vxy_rmse": 0.10, "base_wz_rmse": 0.10, "ee_pos_rmse": 0.10,
                "ee_ori_mae_deg": 10.0}
        values = {}
        for seed, gain in ((1, 0.01), (2, 0.05), (3, 0.03)):
            values[seed] = {
                "C1": {**base, "base_vxy_rmse": 0.10 - gain,
                       "base_wz_rmse": 0.10 - gain, "ee_pos_rmse": 0.10 - gain},
                "C3": dict(base), "C5": dict(base),
            }
        spec = self._spec()
        cands = self._candidates(values)
        scales = select_cases.compute_scales(cands, spec)
        cands = select_cases.score_candidates(cands, spec, scales)
        records, _ = select_cases.select(cands, spec)
        self.assertEqual(records[0]["selected"]["seed"], 2)
        self.assertEqual([r["seed"] for r in records[0]["ranking_top10"]], [2, 3, 1])

    def test_min_rule_requires_all_metrics_simultaneously(self):
        base = {"base_vxy_rmse": 0.10, "base_wz_rmse": 0.10, "ee_pos_rmse": 0.10,
                "ee_ori_mae_deg": 10.0}
        # seed 1: huge base win but LOSES on EE -> min rule must reject it.
        values = {
            1: {"C1": {**base, "base_vxy_rmse": 0.01, "base_wz_rmse": 0.01,
                       "ee_pos_rmse": 0.20},
                "C3": dict(base), "C5": dict(base)},
            2: {"C1": {**base, "base_vxy_rmse": 0.09, "base_wz_rmse": 0.09,
                       "ee_pos_rmse": 0.09},
                "C3": dict(base), "C5": dict(base)},
        }
        spec = self._spec()
        cands = self._candidates(values)
        scales = select_cases.compute_scales(cands, spec)
        cands = select_cases.score_candidates(cands, spec, scales)
        records, _ = select_cases.select(cands, spec)
        simultaneous = records[0]
        self.assertEqual(simultaneous["selected"]["seed"], 2)

    def test_require_positive_gate_excludes_a_candidate(self):
        base = {"base_vxy_rmse": 0.10, "base_wz_rmse": 0.10, "ee_pos_rmse": 0.10,
                "ee_ori_mae_deg": 10.0}
        values = {
            1: {"C1": {**base, "base_vxy_rmse": 0.05, "base_wz_rmse": 0.05,
                       "ee_pos_rmse": 0.11},   # worse EE -> fails example2 gate
                "C3": dict(base), "C5": dict(base)},
        }
        spec = self._spec()
        cands = self._candidates(values)
        scales = select_cases.compute_scales(cands, spec)
        cands = select_cases.score_candidates(cands, spec, scales)
        locomotion = [c for c in spec["cases"]
                      if c["id"] == "example2_locomotion_advantage"][0]
        score = cands[0]["scores"][locomotion["id"]]
        self.assertFalse(score["eligible"])
        self.assertIn("not better on required metric", score["reason"])

    def test_no_qualifying_candidate_is_reported_not_backfilled(self):
        base = {"base_vxy_rmse": 0.10, "base_wz_rmse": 0.10, "ee_pos_rmse": 0.10,
                "ee_ori_mae_deg": 10.0}
        # C1 worse everywhere: nothing may be selected.
        values = {1: {"C1": {k: v * 1.5 for k, v in base.items()},
                      "C3": dict(base), "C5": dict(base)}}
        spec = self._spec()
        cands = self._candidates(values)
        scales = select_cases.compute_scales(cands, spec)
        cands = select_cases.score_candidates(cands, spec, scales)
        records, _ = select_cases.select(cands, spec)
        for record in records:
            self.assertIsNone(record["selected"])
            self.assertIn("No gated candidate", record["selection_failure_reason"])

    def test_each_criterion_takes_a_distinct_seed(self):
        base = {"base_vxy_rmse": 0.10, "base_wz_rmse": 0.10, "ee_pos_rmse": 0.10,
                "ee_ori_mae_deg": 10.0}
        values = {}
        for seed, gain in ((1, 0.05), (2, 0.04), (3, 0.03)):
            values[seed] = {
                "C1": {**base, "base_vxy_rmse": 0.10 - gain,
                       "base_wz_rmse": 0.10 - gain, "ee_pos_rmse": 0.10 - gain},
                "C3": dict(base), "C5": dict(base)}
        spec = self._spec()
        cands = self._candidates(values)
        scales = select_cases.compute_scales(cands, spec)
        cands = select_cases.score_candidates(cands, spec, scales)
        records, _ = select_cases.select(cands, spec)
        seeds = [r["selected"]["seed"] for r in records if r["selected"]]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_selection_is_deterministic(self):
        base = {"base_vxy_rmse": 0.10, "base_wz_rmse": 0.10, "ee_pos_rmse": 0.10,
                "ee_ori_mae_deg": 10.0}
        values = {s: {"C1": {**base, "base_vxy_rmse": 0.10 - 0.001 * s,
                             "base_wz_rmse": 0.10 - 0.001 * s,
                             "ee_pos_rmse": 0.10 - 0.001 * s},
                      "C3": dict(base), "C5": dict(base)} for s in range(1, 9)}
        spec = self._spec()
        runs = []
        for _ in range(2):
            cands = self._candidates(values)
            scales = select_cases.compute_scales(cands, spec)
            cands = select_cases.score_candidates(cands, spec, scales)
            records, _ = select_cases.select(cands, spec)
            runs.append([r["selected"]["seed"] for r in records if r["selected"]])
        self.assertEqual(runs[0], runs[1])

    def test_ties_break_on_the_lower_seed(self):
        base = {"base_vxy_rmse": 0.10, "base_wz_rmse": 0.10, "ee_pos_rmse": 0.10,
                "ee_ori_mae_deg": 10.0}
        better = {**base, "base_vxy_rmse": 0.08, "base_wz_rmse": 0.08,
                  "ee_pos_rmse": 0.08}
        values = {7: {"C1": dict(better), "C3": dict(base), "C5": dict(base)},
                  3: {"C1": dict(better), "C3": dict(base), "C5": dict(base)}}
        spec = self._spec()
        cands = self._candidates(values)
        scales = select_cases.compute_scales(cands, spec)
        cands = select_cases.score_candidates(cands, spec, scales)
        records, _ = select_cases.select(cands, spec)
        self.assertEqual(records[0]["selected"]["seed"], 3)


class TestPoolDisjointness(unittest.TestCase):
    def test_configured_pools_do_not_overlap(self):
        _, global_cfg = E.load_config("E1")
        official = set(range(global_cfg["official_seeds"]["start"],
                             global_cfg["official_seeds"]["start"]
                             + global_cfg["official_seeds"]["count"]))
        showcase = set(range(global_cfg["showcase_seeds"]["start"],
                             global_cfg["showcase_seeds"]["start"]
                             + global_cfg["showcase_seeds"]["count"]))
        self.assertEqual(official & showcase, set())

    def test_overlapping_pools_raise(self):
        _, global_cfg = E.load_config("E1")
        bad_pool = dict(global_cfg["official_seeds"])
        with self.assertRaises(SystemExit):
            select_cases.build_candidates(
                "E1", "/nonexistent", bad_pool, select_cases.CRITERIA["E1"],
                global_cfg["base_command_limits"], global_cfg)

    def test_official_filter_keeps_terminated_episodes(self):
        from benchmark import aggregate
        rows = [{"seed": 0, "survived": 0}, {"seed": 1, "survived": 1},
                {"seed": 99, "survived": 1}]
        kept = aggregate.filter_official(rows, 0, 60)
        self.assertEqual([r["seed"] for r in kept], [0, 1])
        self.assertEqual(sum(1 for r in kept if not r["survived"]), 1)


class TestDivergingScale(unittest.TestCase):
    def test_midpoint_is_neutral_and_poles_are_opposite(self):
        from benchmark.plots import style
        mid = np.array(style.DIVERGING(0.5)[:3])
        self.assertLess(float(np.std(mid)), 0.02)      # gray: channels agree
        low = np.array(style.DIVERGING(0.0)[:3])
        high = np.array(style.DIVERGING(1.0)[:3])
        self.assertGreater(low[0], low[2])             # red pole
        self.assertGreater(high[2], high[0])           # blue pole

    def test_equal_steps_per_arm(self):
        from benchmark.plots import style
        steps = style.DIVERGING_STEPS
        self.assertEqual(len(steps) % 2, 1)
        half = len(steps) // 2
        self.assertEqual(half, len(steps) - half - 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
