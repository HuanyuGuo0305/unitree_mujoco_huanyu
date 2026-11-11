#!/usr/bin/env python3
"""Round 2 vs round 3: what changed, and why.

The comparison covers E1 and E2, whose protocols changed:

**E1 and E2 -- protocol change.** The commands themselves changed (E1 base
resampling 2-4 s -> 2-3 s and EE targets 5 s -> 3 s; E2 envelope
0.15/0.30/0.45 -> 0.20/0.40/0.60 per axis). A round-2 number and a round-3
number therefore answer DIFFERENT questions, and the difference between
them is an effect of the task, not an improvement or a regression in any
controller. Reported as context only.

(The former part B, the E3 seed-pool comparison, was removed with the
superseded E3 v1 protocol; see docs/E3_V3_CLEANUP.md.)
"""

from __future__ import annotations

import argparse
import os

from benchmark.round3 import common

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(common.OUT, "validation"))
    args = parser.parse_args(argv)

    protocol_rows: list[dict] = []
    protocol_comparison("E1", E1_MAP, "E1_tracking_summary.csv", protocol_rows)
    protocol_comparison("E2", E2_MAP, "E2_wheel_actuation_summary.csv", protocol_rows)
    common.write_csv(os.path.join(args.out, "round2_vs_round3_protocol_change.csv"),
                     protocol_rows)

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

    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
