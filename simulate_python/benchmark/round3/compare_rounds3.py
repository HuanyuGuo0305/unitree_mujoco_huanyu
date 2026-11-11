#!/usr/bin/env python3
"""Round 2 vs round 3: what changed, and why.

Two kinds of difference are reported SEPARATELY, because conflating them would
be misleading:

A. **E1 and E2 -- protocol change.** The commands themselves changed (E1 base
   resampling 2-4 s -> 2-3 s and EE targets 5 s -> 3 s; E2 envelope
   0.15/0.30/0.45 -> 0.20/0.40/0.60 per axis). A round-2 number and a round-3
   number therefore answer DIFFERENT questions, and the difference between
   them is an effect of the task, not an improvement or a regression in any
   controller. Reported as context only.

B. **E3 -- seed-pool change only.** The protocol is byte-identical and the
   episodes are the same deterministic runs, so round 2's 60 seeds (0-59) and
   round 3's 50 (0-49) are a controlled comparison: the only difference is
   which seeds are aggregated. Every paired comparison is recomputed on both
   pools from the same raw data, and any CONCLUSION CHANGE -- a sign flip or a
   significance flip -- is listed individually. This is what shows whether
   narrowing the pool to 50 cost anything.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from benchmark.core import stats
from benchmark.round3 import aggregate3, common

ARCHIVE = os.path.join(common.BENCH, "results", "archive_round2")

E1_MAP = {
    "base_vxy_rmse_mps": "base_vxy_rmse_mean",
    "base_yaw_rmse_radps": "base_wz_rmse_mean",
    "ee_position_rmse_m": "ee_pos_rmse_mean",
    "ee_orientation_mae_deg": "ee_ori_mae_deg_mean",
    "survived": "survived_mean",
}
E2_MAP = {
    "base_vxy_rmse_mps": "base_vxy_rmse_mean",
    "base_yaw_rmse_radps": "base_wz_rmse_mean",
    "total_power_w": "total_power_mean",
    "leg_power_w": "leg_power_mean",
    "arm_power_w": "arm_power_mean",
    "wheel_power_w": "wheel_power_mean",
    "leg_acc_rms_radps2": "leg_acc_rms_mean",
    "arm_acc_rms_radps2": "arm_acc_rms_mean",
}


def protocol_comparison(experiment, mapping, archive_csv, out_rows):
    old = common.read_csv(os.path.join(ARCHIVE, "tables", archive_csv))
    old_idx = {(r["condition"], r["baseline"]): r for r in old}
    new = common.read_csv(os.path.join(common.OUT, experiment, "summary_50seeds.csv"))
    for r in new:
        if r["metric"] not in mapping:
            continue
        key = (r["condition"], r["baseline"])
        if key not in old_idx:
            continue
        old_value = old_idx[key].get(mapping[r["metric"]])
        if old_value in (None, "", "nan"):
            continue
        old_value = float(old_value)
        new_value = float(r["mean"])
        out_rows.append(
            {
                "experiment": experiment, "comparison_kind": "protocol_change",
                "condition": r["condition"], "baseline": r["baseline"],
                "metric": r["metric"],
                "round2_mean": old_value, "round2_n_seeds": int(old_idx[key].get("n_episodes", 0)),
                "round3_mean": new_value, "round3_n_seeds": int(r["n_seeds"]),
                "absolute_change": new_value - old_value,
                "relative_change_pct": (100.0 * (new_value - old_value) / old_value
                                        if old_value else float("nan")),
                "note": "different command protocol; not a controller change",
            }
        )


def e3_seed_pool_comparison(out_rows, cfg):
    """Recompute every E3 paired comparison on 50 and on 60 seeds."""
    rows = common.read_csv(os.path.join(common.OUT, "E3", "per_seed.csv"))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["alpha_base"] = float(r["alpha_base"])
        r["alpha_ee"] = float(r["alpha_ee"])
    # Round 3's per_seed.csv holds 0-49 only, so seeds 50-59 are re-read here
    # from the raw tree, which still contains them untouched.
    from benchmark.scenarios import experiments as E

    exp_cfg, _ = E.load_config("E3")
    cells = E.e3_conditions(exp_cfg)
    extra_seeds = list(range(50, 60))
    extra, _problems = common.load_official(
        "E3", list(exp_cfg["baselines"]), [c for c, *_ in cells], seeds=extra_seeds
    )
    meta = {c: (ab, ae, f) for c, ab, ae, f in cells}
    for r in extra:
        ab, ae, family = meta[r["condition"]]
        rows.append(
            {
                "condition": r["condition"], "baseline": r["baseline"],
                "seed": int(r["seed"]), "trajectory_family": family,
                "alpha_base": float(ab), "alpha_ee": float(ae),
                "survived": int(r["survived"]),
                "ee_position_rmse_m": float(r["ee_pos_rmse"]),
                "base_vxy_rmse_mps": float(r["base_vxy_rmse"]),
                "base_yaw_rmse_radps": float(r["base_wz_rmse"]),
                "ee_orientation_mae_deg": float(r["ee_ori_mae_deg"]),
            }
        )
    n60 = len({r["seed"] for r in rows})

    pools = {"round3_50seeds": set(range(50)), "round2_60seeds": set(range(60))}
    flips = []
    metrics = [("ee_position_rmse_m", "m"), ("base_vxy_rmse_mps", "m/s"),
               ("base_yaw_rmse_radps", "rad/s"), ("ee_orientation_mae_deg", "deg")]
    by_cell: dict = {}
    for r in rows:
        by_cell.setdefault((r["alpha_base"], r["alpha_ee"], r["trajectory_family"]), []).append(r)

    for (ab, ae, family), cell in sorted(by_cell.items()):
        tag = aggregate3._cell_tag(ab, ae)
        for rival in ("C3", "C5"):
            for metric, unit in metrics:
                result = {}
                for name, pool in pools.items():
                    sub = [r for r in cell if r["seed"] in pool]
                    ref = common.per_seed_map(sub, "C1", metric)
                    riv = common.per_seed_map(sub, rival, metric)
                    result[name] = stats.paired_advantage(
                        ref, riv, n_resamples=cfg["bootstrap_resamples"],
                        seed=cfg["bootstrap_seed"], confidence=cfg["confidence"])
                a, b = result["round3_50seeds"], result["round2_60seeds"]
                sign_flip = int(np.sign(a["mean"]) != np.sign(b["mean"]))
                sig_flip = int(bool(a["significant"]) != bool(b["significant"]))
                row = {
                    "experiment": "E3", "comparison_kind": "seed_pool_only",
                    "condition": tag, "trajectory_family": family,
                    "alpha_base": ab, "alpha_ee": ae,
                    "comparison": f"{rival}_minus_C1", "metric": metric, "unit": unit,
                    "round3_n_pairs": a["n_pairs"], "round3_mean": a["mean"],
                    "round3_ci_low": a["ci_low"], "round3_ci_high": a["ci_high"],
                    "round3_significant": int(a["significant"]),
                    "round2_n_pairs": b["n_pairs"], "round2_mean": b["mean"],
                    "round2_ci_low": b["ci_low"], "round2_ci_high": b["ci_high"],
                    "round2_significant": int(b["significant"]),
                    "mean_shift": a["mean"] - b["mean"],
                    "sign_flip": sign_flip, "significance_flip": sig_flip,
                    "conclusion_change": int(bool(sign_flip or sig_flip)),
                    "ci_halfwidth_ratio_50_over_60": (
                        ((a["ci_high"] - a["ci_low"]) / (b["ci_high"] - b["ci_low"]))
                        if (b["ci_high"] - b["ci_low"]) else float("nan")
                    ),
                }
                out_rows.append(row)
                if row["conclusion_change"]:
                    flips.append(row)
    return flips, n60


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(common.OUT, "validation"))
    args = parser.parse_args(argv)
    cfg = common.stats_config()

    protocol_rows: list[dict] = []
    protocol_comparison("E1", E1_MAP, "E1_tracking_summary.csv", protocol_rows)
    protocol_comparison("E2", E2_MAP, "E2_wheel_actuation_summary.csv", protocol_rows)
    common.write_csv(os.path.join(args.out, "round2_vs_round3_protocol_change.csv"),
                     protocol_rows)

    seed_rows: list[dict] = []
    flips, n60 = e3_seed_pool_comparison(seed_rows, cfg)
    common.write_csv(os.path.join(args.out, "round2_vs_round3_E3_seed_pool.csv"),
                     seed_rows)

    print("=" * 78)
    print("ROUND 2 vs ROUND 3")
    print("=" * 78)
    print("\nA. E1/E2 -- PROTOCOL CHANGE (different commands; not a controller change)")
    for experiment in ("E1", "E2"):
        sel = [r for r in protocol_rows if r["experiment"] == experiment]
        print(f"  {experiment}: {len(sel)} (condition, baseline, metric) means compared")
        worst = sorted(sel, key=lambda r: -abs(r["relative_change_pct"]))[:4]
        for r in worst:
            print(f"    {r['condition']:7s} {r['baseline']:3s} {r['metric']:24s} "
                  f"{r['round2_mean']:9.4f} -> {r['round3_mean']:9.4f} "
                  f"({r['relative_change_pct']:+.1f}%)")

    print(f"\nB. E3 -- SEED POOL ONLY (identical protocol, {n60} seeds available)")
    print(f"  {len(seed_rows)} paired comparisons recomputed on both pools")
    print(f"  conclusion changes (sign or significance flip): {len(flips)} "
          f"({100.0*len(flips)/max(len(seed_rows),1):.1f}%)")
    for r in flips[:12]:
        print(f"    {r['condition']} {r['trajectory_family']:10s} {r['comparison']} "
              f"{r['metric']:22s} 50s {r['round3_mean']:+.5f} "
              f"(sig={r['round3_significant']}) vs 60s {r['round2_mean']:+.5f} "
              f"(sig={r['round2_significant']})")
    ratios = [r["ci_halfwidth_ratio_50_over_60"] for r in seed_rows
              if np.isfinite(r["ci_halfwidth_ratio_50_over_60"])]
    if ratios:
        print(f"  CI width ratio 50/60 seeds: median {np.median(ratios):.3f}, "
              f"mean {np.mean(ratios):.3f}  (sqrt(60/50) = "
              f"{np.sqrt(60/50):.3f} expected)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
