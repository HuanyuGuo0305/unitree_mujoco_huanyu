#!/usr/bin/env python3
"""Did adding seeds 30-59 change any conclusion?

Recomputes every paired comparison on seeds 0-29 and on seeds 0-59 and reports,
per (experiment, condition, metric, pair):

* the mean difference and 95% CI in each round;
* whether the SIGN of the point estimate changed;
* whether the SIGNIFICANCE verdict changed (CI excluding zero or not);
* the CI half-width in each round, i.e. how much precision the extra seeds
  actually bought.

This is run AFTER the expansion and reported as-is. It is not a stopping rule:
the seed counts were fixed in advance (0-29, then 30-59), and no comparison was
re-run until a preferred verdict appeared.

    python3 benchmark/compare_rounds.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for root in (REPO_ROOT, SIM_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

from benchmark import aggregate  # noqa: E402
from benchmark.core import stats  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

RESULTS_ROOT = os.path.join(SIM_ROOT, "benchmark", "results")

SPECS = {
    "E1": (["base_vxy_rmse", "base_wz_rmse", "ee_pos_rmse", "ee_ori_mae_deg"],
           ["condition"], [("C1", "C3"), ("C1", "C5")]),
    "E2": (["base_vxy_rmse", "base_wz_rmse", "total_power", "leg_power",
            "leg_acc_rms", "arm_acc_rms"],
           ["condition"], [("C1", "C2")]),
}


def _pairs(rows, metric, group_keys, a, b, cfg_stats):
    out = {}
    for key, items in aggregate.group(rows, group_keys).items():
        va = {r["seed"]: r[metric] for r in items
              if r["baseline"] == a and metric in r}
        vb = {r["seed"]: r[metric] for r in items
              if r["baseline"] == b and metric in r}
        if not va or not vb:
            continue
        out[key] = stats.paired_difference(
            va, vb,
            n_resamples=cfg_stats["bootstrap_resamples"],
            seed=cfg_stats["bootstrap_seed"],
            confidence=cfg_stats["confidence"],
        )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=os.path.join(RESULTS_ROOT, "raw"))
    parser.add_argument("--out", default=os.path.join(
        RESULTS_ROOT, "summary", "round_comparison_30_vs_60.csv"))
    parser.add_argument("--round1", type=int, default=30)
    parser.add_argument("--round2", type=int, default=60)
    args = parser.parse_args(argv)

    _, global_cfg = E.load_config("E1")
    cfg_stats = global_cfg["statistics"]

    records = []
    for experiment, (metrics, group_keys, pairs) in SPECS.items():
        rows_all = aggregate.load_raw(args.results, [experiment])
        r1 = aggregate.filter_official(rows_all, 0, args.round1)
        r2 = aggregate.filter_official(rows_all, 0, args.round2)
        if not r1 or not r2:
            print(f"{experiment}: missing data, skipping")
            continue
        n1 = len({r["seed"] for r in r1})
        n2 = len({r["seed"] for r in r2})
        print(f"\n=== {experiment}: {n1} seeds vs {n2} seeds ===")
        flips = []
        for metric in metrics:
            for a, b in pairs:
                p1 = _pairs(r1, metric, group_keys, a, b, cfg_stats)
                p2 = _pairs(r2, metric, group_keys, a, b, cfg_stats)
                for key in sorted(set(p1) & set(p2), key=lambda k: tuple(map(str, k))):
                    x, y = p1[key], p2[key]
                    sign_flip = bool(np.sign(x["mean"]) != np.sign(y["mean"]))
                    sig_flip = bool(x["significant"] != y["significant"])
                    hw1 = (x["ci_high"] - x["ci_low"]) / 2.0
                    hw2 = (y["ci_high"] - y["ci_low"]) / 2.0
                    record = {
                        "experiment": experiment,
                        **dict(zip(group_keys, key)),
                        "metric": metric, "baseline_a": a, "baseline_b": b,
                        f"n_seeds_round1": x["n_pairs"],
                        f"n_seeds_round2": y["n_pairs"],
                        "mean_round1": x["mean"], "ci_low_round1": x["ci_low"],
                        "ci_high_round1": x["ci_high"],
                        "significant_round1": x["significant"],
                        "mean_round2": y["mean"], "ci_low_round2": y["ci_low"],
                        "ci_high_round2": y["ci_high"],
                        "significant_round2": y["significant"],
                        "ci_halfwidth_round1": hw1,
                        "ci_halfwidth_round2": hw2,
                        "ci_halfwidth_ratio": (hw2 / hw1) if hw1 else float("nan"),
                        "sign_changed": sign_flip,
                        "significance_changed": sig_flip,
                        "conclusion_changed": sign_flip or sig_flip,
                    }
                    records.append(record)
                    if record["conclusion_changed"]:
                        flips.append(record)
        total = sum(1 for r in records if r["experiment"] == experiment)
        ratios = [r["ci_halfwidth_ratio"] for r in records
                  if r["experiment"] == experiment and np.isfinite(r["ci_halfwidth_ratio"])]
        print(f"  {total} paired comparisons; "
              f"{len(flips)} changed a conclusion")
        if ratios:
            print(f"  CI half-width ratio (60-seed / 30-seed): "
                  f"median {np.median(ratios):.3f}, "
                  f"expected ~{1/np.sqrt(2):.3f} for independent sampling")
        for record in flips:
            label = " ".join(f"{k}={record[k]}" for k in group_keys)
            print(f"    CHANGED  {label} {record['metric']} "
                  f"{record['baseline_a']}-{record['baseline_b']}: "
                  f"{record['mean_round1']:+.4f}"
                  f"[{record['ci_low_round1']:+.4f},{record['ci_high_round1']:+.4f}]"
                  f" sig={record['significant_round1']}"
                  f"  ->  {record['mean_round2']:+.4f}"
                  f"[{record['ci_low_round2']:+.4f},{record['ci_high_round2']:+.4f}]"
                  f" sig={record['significant_round2']}")

    aggregate.write_csv(args.out, records)
    changed = sum(1 for r in records if r["conclusion_changed"])
    print(f"\n{len(records)} comparisons total, {changed} conclusion changes")
    print(f"-> {os.path.relpath(args.out, SIM_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
