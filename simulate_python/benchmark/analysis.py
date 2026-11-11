#!/usr/bin/env python3
"""E3 round-2 analysis: absolute tracking and explicit C1-vs-baseline pairing.

Adds the three things the degradation-only view could not answer:

1. **Absolute tracking** per matrix cell (``E3_absolute_tracking.csv``) -- a
   controller must not look good merely by degrading little from an already
   poor reference.
2. **Paired advantage** of C1 over C3 and over C5
   (``E3_C1_vs_C3_paired.csv``, ``E3_C1_vs_C5_paired.csv``) with mean, median,
   bootstrap CI and the fraction of paired trials C1 wins, per trajectory
   family and per cell.
3. The same treatment for the two **directional degradation** quantities, so
   the coupling claim and the absolute claim are reported side by side.

Sign convention everywhere: ``advantage = rival - C1``, for metrics where lower
is better, so POSITIVE means C1 was better on that seed. Nothing is declared
significant on a point estimate; significance is the mean advantage's 95%
bootstrap CI excluding zero.

    python3 benchmark/analysis.py                      # official pool
    python3 benchmark/analysis.py --seed-start 0 --num-seeds 30   # compare rounds
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

# (column in the per-episode row, short name used in the paper, units)
ABSOLUTE_QUANTITIES = [
    ("ee_pos_rmse", "A_EE", "m"),
    ("base_vxy_rmse", "A_Base", "m/s"),
    ("ee_ori_mae_deg", "A_EE_ori", "deg"),
    ("base_wz_rmse", "A_Base_wz", "rad/s"),
]
DEGRADATION_QUANTITIES = [
    ("delta_ee_pos_rmse_from_base", "A_Delta_EE", "m"),
    ("delta_base_vxy_rmse_from_ee", "A_Delta_Base", "m/s"),
]
FAMILIES = ("line", "circle", "semicircle")
CELL_NAME = {(0.0, 0.0): "S00", (0.0, 0.5): "S01", (0.0, 1.0): "S02",
             (0.5, 0.0): "S10", (0.5, 0.5): "S11", (0.5, 1.0): "S12",
             (1.0, 0.0): "S20", (1.0, 0.5): "S21", (1.0, 1.0): "S22"}
DEMANDING = ("S11", "S12", "S21", "S22")


def _cell(row) -> str:
    return CELL_NAME.get((float(row["alpha_base"]), float(row["alpha_ee"])), "?")


def load_e3(raw_root, seed_start, seed_count):
    rows = aggregate.load_raw(raw_root, ["E3"])
    rows = aggregate.filter_official(rows, seed_start, seed_count)
    degradation = aggregate.e3_degradation(rows, None)
    # Merge the two degradation columns onto the per-episode rows so one
    # function can treat absolute and degradation quantities identically.
    index = {(r["baseline"], r["ee_family"], r["seed"],
              float(r["alpha_base"]), float(r["alpha_ee"])): r for r in degradation}
    for row in rows:
        extra = index.get((row["baseline"], row["ee_family"], row["seed"],
                           float(row["alpha_base"]), float(row["alpha_ee"])))
        if not extra:
            continue
        for key, _, _ in DEGRADATION_QUANTITIES:
            if key in extra:
                row[key] = extra[key]
    return rows


def absolute_tracking_table(rows, cfg_stats) -> list[dict]:
    """Mean + 95% CI of every absolute metric, per cell, family and baseline."""
    metrics = [m for m, _, _ in ABSOLUTE_QUANTITIES]
    out = []
    groups: dict = {}
    for row in rows:
        key = (row["ee_family"], float(row["alpha_base"]), float(row["alpha_ee"]),
               row["baseline"])
        groups.setdefault(key, []).append(row)
    for (family, ab, ae, baseline), items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        record = {
            "ee_family": family, "cell": CELL_NAME.get((ab, ae), "?"),
            "alpha_base": ab, "alpha_ee": ae, "baseline": baseline,
            "n_episodes": len(items),
            "n_survived": sum(int(r["survived"]) for r in items),
            "demanding_cell": CELL_NAME.get((ab, ae), "?") in DEMANDING,
        }
        for metric in metrics:
            s = stats.summarize(
                [r[metric] for r in items if metric in r],
                n_resamples=cfg_stats["bootstrap_resamples"],
                seed=cfg_stats["bootstrap_seed"],
                confidence=cfg_stats["confidence"],
            )
            record[f"{metric}_mean"] = s["mean"]
            record[f"{metric}_median"] = s["median"]
            record[f"{metric}_ci_low"] = s["ci_low"]
            record[f"{metric}_ci_high"] = s["ci_high"]
            record[f"{metric}_std"] = s["std"]
        out.append(record)
    return out


def paired_advantage_table(rows, reference, rival, cfg_stats) -> list[dict]:
    """``rival - reference`` per matched (seed, family) for every quantity.

    Also emits ``ee_family = all`` rows, which pool the three families by
    pairing on (seed, family) -- 3x the pairs, same cell. Those rows are a
    pooled summary, not an independent replication of the per-family rows, and
    are labelled ``pooled_over_families``.
    """
    quantities = ABSOLUTE_QUANTITIES + DEGRADATION_QUANTITIES
    by_cell: dict = {}
    for row in rows:
        key = (float(row["alpha_base"]), float(row["alpha_ee"]))
        by_cell.setdefault(key, []).append(row)

    out = []
    for (ab, ae), items in sorted(by_cell.items()):
        cell = CELL_NAME.get((ab, ae), "?")
        for family in list(FAMILIES) + ["all"]:
            scope = items if family == "all" else [r for r in items if r["ee_family"] == family]
            if not scope:
                continue
            for metric, short, unit in quantities:
                # Key on (seed, family) so the pooled rows stay correctly paired.
                ref = {(r["seed"], r["ee_family"]): r[metric]
                       for r in scope if r["baseline"] == reference and metric in r}
                oth = {(r["seed"], r["ee_family"]): r[metric]
                       for r in scope if r["baseline"] == rival and metric in r}
                if not ref or not oth:
                    continue
                res = stats.paired_advantage(
                    ref, oth,
                    n_resamples=cfg_stats["bootstrap_resamples"],
                    seed=cfg_stats["bootstrap_seed"],
                    confidence=cfg_stats["confidence"],
                )
                out.append({
                    "cell": cell, "alpha_base": ab, "alpha_ee": ae,
                    "ee_family": family,
                    "pooled_over_families": family == "all",
                    "demanding_cell": cell in DEMANDING,
                    "reference": reference, "rival": rival,
                    "quantity": short, "metric": metric, "unit": unit,
                    "advantage_definition": f"{rival}_{metric} - {reference}_{metric}",
                    "mean_advantage": res["mean"],
                    "median_advantage": res["median"],
                    "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                    "median_ci_low": res["median_ci_low"],
                    "median_ci_high": res["median_ci_high"],
                    "frac_trials_reference_better": res["frac_positive"],
                    "n_trials_reference_better": res["n_positive"],
                    "n_pairs": res["n_pairs"],
                    "significant_95": res["significant"],
                    "favours": (
                        "neither (CI spans 0)" if not res["significant"]
                        else (reference if res["mean"] > 0 else rival)
                    ),
                })
    return out


def _print_paired(table, reference, rival, quantity, cells=DEMANDING):
    sel = [r for r in table if r["quantity"] == quantity and r["cell"] in cells
           and not r["pooled_over_families"]]
    if not sel:
        return
    unit = sel[0]["unit"]
    print(f"\n{quantity} = {rival} - {reference}  [{unit}]   "
          f"(positive => {reference} better)")
    print(f"{'cell':5s} {'family':11s} {'mean':>9s} {'95% CI':>21s} "
          f"{'median':>9s} {'win':>7s}  favours")
    for r in sorted(sel, key=lambda r: (r["cell"], r["ee_family"])):
        print(f"{r['cell']:5s} {r['ee_family']:11s} {r['mean_advantage']:+9.4f} "
              f"[{r['ci_low']:+9.4f},{r['ci_high']:+9.4f}] "
              f"{r['median_advantage']:+9.4f} "
              f"{r['frac_trials_reference_better']*100:6.0f}%  {r['favours']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=os.path.join(RESULTS_ROOT, "raw"))
    parser.add_argument("--out", default=RESULTS_ROOT)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--suffix", default="", help="Appended to output file names")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    _, global_cfg = E.load_config("E3")
    cfg_stats = global_cfg["statistics"]
    seed_start = args.seed_start if args.seed_start is not None else global_cfg["official_seeds"]["start"]
    seed_count = args.num_seeds if args.num_seeds is not None else global_cfg["official_seeds"]["count"]

    rows = load_e3(args.results, seed_start, seed_count)
    if not rows:
        print(f"No E3 episodes in seeds {seed_start}-{seed_start+seed_count-1}")
        return 1
    seeds = sorted({r["seed"] for r in rows})
    print(f"E3: {len(rows)} episodes, {len(seeds)} seeds "
          f"({seeds[0]}-{seeds[-1]}), {len({r['baseline'] for r in rows})} baselines, "
          f"{sum(1 for r in rows if not r['survived'])} early terminations")

    tables_dir = os.path.join(args.out, "tables")
    sfx = args.suffix

    absolute = absolute_tracking_table(rows, cfg_stats)
    aggregate.write_csv(os.path.join(tables_dir, f"E3_absolute_tracking{sfx}.csv"), absolute)

    for rival in ("C3", "C5"):
        table = paired_advantage_table(rows, "C1", rival, cfg_stats)
        aggregate.write_csv(
            os.path.join(tables_dir, f"E3_C1_vs_{rival}_paired{sfx}.csv"), table
        )
        if not args.quiet:
            print("\n" + "=" * 78)
            print(f"C1 vs {rival} -- demanding cells only (full matrix in the CSV)")
            print("=" * 78)
            for _, short, _ in ABSOLUTE_QUANTITIES + DEGRADATION_QUANTITIES:
                _print_paired(table, "C1", rival, short)

    print(f"\nTables -> {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
