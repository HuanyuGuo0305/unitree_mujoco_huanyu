#!/usr/bin/env python3
"""Aggregate raw episode results into tidy CSVs, summaries and LaTeX tables.

    python3 benchmark/aggregate.py --results benchmark/results/raw
    python3 benchmark/aggregate.py --experiments E1 --tables

Outputs (under ``benchmark/results``):

    tables/E1_tracking_all.csv        every official episode, every metric
    tables/E1_tracking_summary.csv    per (condition, baseline) statistics
    tables/E1_tracking_compact.tex    the four-panel IEEE table
    tables/E2_wheel_actuation.{tex,csv}
    tables/E4_stabilization.csv
    summary/<EXP>_paired_differences.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for root in (REPO_ROOT, SIM_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

from benchmark.controllers.registry import baseline_label  # noqa: E402
from benchmark.core import stats  # noqa: E402
from benchmark.plots import tables as table_gen  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

RESULTS_ROOT = os.path.join(SIM_ROOT, "benchmark", "results")
DEFAULT_RAW = os.path.join(RESULTS_ROOT, "raw")


# --------------------------------------------------------------------------
def load_raw(raw_root: str, experiments=None) -> list[dict]:
    """Load every ``metrics.json`` + ``metadata.json`` pair under ``raw_root``."""
    rows = []
    if not os.path.isdir(raw_root):
        return rows
    for experiment in sorted(os.listdir(raw_root)):
        if experiment.startswith("_"):
            continue
        if experiments and experiment not in experiments:
            continue
        exp_dir = os.path.join(raw_root, experiment)
        if not os.path.isdir(exp_dir):
            continue
        for baseline in sorted(os.listdir(exp_dir)):
            base_dir = os.path.join(exp_dir, baseline)
            if not os.path.isdir(base_dir):
                continue
            for condition in sorted(os.listdir(base_dir)):
                cond_dir = os.path.join(base_dir, condition)
                if not os.path.isdir(cond_dir):
                    continue
                for seed_dir in sorted(os.listdir(cond_dir)):
                    path = os.path.join(cond_dir, seed_dir)
                    mpath = os.path.join(path, "metrics.json")
                    dpath = os.path.join(path, "metadata.json")
                    if not (os.path.isfile(mpath) and os.path.isfile(dpath)):
                        continue
                    with open(mpath, "r", encoding="utf-8") as handle:
                        metrics = json.load(handle)
                    with open(dpath, "r", encoding="utf-8") as handle:
                        meta = json.load(handle)
                    row = dict(metrics)
                    row.update(
                        {
                            "experiment": experiment,
                            "baseline": baseline,
                            "condition": condition,
                            "seed": int(meta["seed"]),
                            "settle_s": float(meta["settle_s"]),
                            "duration_s": float(meta["duration_s"]),
                            "path": os.path.relpath(path, SIM_ROOT),
                        }
                    )
                    scen = meta.get("scenario", {})
                    for key in (
                        "alpha_base", "alpha_ee", "ee_family", "base_intensity",
                        "ee_mean_speed_mps", "base_mode", "ee_mode",
                    ):
                        if key in scen:
                            row[key] = scen[key]
                    rows.append(row)
    return rows


def filter_official(rows, seed_start: int, seed_count: int) -> list[dict]:
    """Restrict to the official seed pool. No other filtering is ever applied:
    early-terminated episodes are retained and reported."""
    allowed = set(range(seed_start, seed_start + seed_count))
    return [r for r in rows if r["seed"] in allowed]


def write_csv(path: str, rows: list[dict], columns=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------
METRIC_LABELS = {
    "base_vxy_rmse": ("Base XY vel. RMSE", "m/s", True),
    "base_wz_rmse": ("Base yaw-rate RMSE", "rad/s", True),
    "ee_pos_rmse": ("EE pos. RMSE", "m", True),
    "ee_ori_mae_deg": ("EE ori. MAE", "deg", True),
    "total_power": ("Mech. power proxy", "W", True),
    "leg_power": ("Leg power proxy", "W", True),
    "arm_power": ("Arm power proxy", "W", True),
    "wheel_power": ("Wheel power proxy", "W", True),
    "leg_acc_rms": ("Leg joint acc. RMS", "rad/s^2", True),
    "arm_acc_rms": ("Arm joint acc. RMS", "rad/s^2", True),
    "survived": ("Survival rate", "-", False),
}


def group(rows, keys):
    out: dict = {}
    for row in rows:
        out.setdefault(tuple(row[k] for k in keys), []).append(row)
    return out


def summarize_groups(rows, metrics, group_keys, cfg_stats) -> list[dict]:
    summary = []
    for key, items in sorted(group(rows, group_keys).items(), key=lambda kv: tuple(map(str, kv[0]))):
        record = dict(zip(group_keys, key))
        record["n_episodes"] = len(items)
        if all("survived" in r and r["survived"] is not None for r in items):
            n_surv = sum(int(r["survived"]) for r in items)
            record["n_survived"] = n_surv
            record["survival_rate"] = n_surv / len(items)
            lo, hi = stats.wilson_interval(n_surv, len(items), cfg_stats["confidence"])
            record["survival_ci_low"], record["survival_ci_high"] = lo, hi
            record["n_terminated"] = len(items) - n_surv
            reasons = {}
            for r in items:
                if not int(r["survived"]):
                    reason = r.get("termination_reason", "unknown")
                    reasons[reason] = reasons.get(reason, 0) + 1
            record["termination_reasons"] = ";".join(
                f"{k}={v}" for k, v in sorted(reasons.items())
            )
        if all("valid_eval_duration_s" in r for r in items):
            record["mean_valid_eval_duration_s"] = float(
                np.mean([x["valid_eval_duration_s"] for x in items])
            )
        for metric in metrics:
            values = [r[metric] for r in items if metric in r]
            s = stats.summarize(
                values,
                n_resamples=cfg_stats["bootstrap_resamples"],
                seed=cfg_stats["bootstrap_seed"],
                confidence=cfg_stats["confidence"],
            )
            for stat_name, value in s.items():
                record[f"{metric}_{stat_name}"] = value
        summary.append(record)
    return summary


def paired_table(rows, metrics, group_keys, baselines, cfg_stats) -> list[dict]:
    """Every pairwise paired difference, preserving seed pairing."""
    out = []
    for key, items in sorted(group(rows, group_keys).items(), key=lambda kv: tuple(map(str, kv[0]))):
        by_baseline = {}
        for row in items:
            by_baseline.setdefault(row["baseline"], {})[row["seed"]] = row
        present = [b for b in baselines if b in by_baseline]
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                for metric in metrics:
                    va = {s: r[metric] for s, r in by_baseline[a].items() if metric in r}
                    vb = {s: r[metric] for s, r in by_baseline[b].items() if metric in r}
                    res = stats.paired_difference(
                        va, vb,
                        n_resamples=cfg_stats["bootstrap_resamples"],
                        seed=cfg_stats["bootstrap_seed"],
                        confidence=cfg_stats["confidence"],
                    )
                    record = dict(zip(group_keys, key))
                    record.update(
                        {
                            "metric": metric,
                            "baseline_a": a,
                            "baseline_b": b,
                            "mean_b_minus_a": res["mean"],
                            "ci_low": res["ci_low"],
                            "ci_high": res["ci_high"],
                            "n_pairs": res["n_pairs"],
                            "significant_95": res["significant"],
                        }
                    )
                    out.append(record)
    return out


# --------------------------------------------------------------------------
E1_METRICS = ["base_vxy_rmse", "base_wz_rmse", "ee_pos_rmse", "ee_ori_mae_deg"]
E1_SECONDARY = [
    "survived", "valid_eval_duration_s", "ee_pos_err_post_switch_mean",
    "ee_ori_err_post_switch_mean_deg", "base_vxy_err_post_switch_mean",
    "n_ee_switches_in_eval", "n_base_switches_in_eval",
    "base_vxy_rmse_yawframe", "base_wz_rmse_yawframe",
]
E2_METRICS = [
    "base_vxy_rmse", "base_wz_rmse", "total_power", "leg_power", "arm_power",
    "wheel_power", "leg_acc_rms", "arm_acc_rms", "survived",
]
E4_METRICS = [
    "ee_world_pos_rmse", "ee_world_pos_mae", "ee_world_pos_max",
    "ee_world_ori_mae_deg", "ee_world_ori_max_deg",
    "base_vxy_rmse", "base_wz_rmse", "survived",
]


# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=DEFAULT_RAW)
    parser.add_argument("--out", default=RESULTS_ROOT)
    parser.add_argument("--experiments", default=None, help="Comma-separated, default all found")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--tables", action="store_true", help="Also emit LaTeX tables")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Aggregate every seed found instead of the official pool")
    args = parser.parse_args(argv)

    experiments = (
        [e.strip() for e in args.experiments.split(",")] if args.experiments else None
    )
    _, global_cfg = E.load_config("E1")
    cfg_stats = global_cfg["statistics"]
    seed_start = (
        args.seed_start if args.seed_start is not None else global_cfg["official_seeds"]["start"]
    )
    seed_count = (
        args.num_seeds if args.num_seeds is not None else global_cfg["official_seeds"]["count"]
    )

    rows = load_raw(args.results, experiments)
    if not rows:
        print(f"No episode results found under {args.results}")
        return 1
    found = sorted({r["experiment"] for r in rows})
    print(f"Loaded {len(rows)} episodes for experiments {found}")

    tables_dir = os.path.join(args.out, "tables")
    summary_dir = os.path.join(args.out, "summary")

    for experiment in found:
        exp_rows = [r for r in rows if r["experiment"] == experiment]
        if args.all_seeds:
            official = exp_rows
            scope = f"all {len(exp_rows)} episodes found"
        else:
            official = filter_official(exp_rows, seed_start, seed_count)
            scope = (
                f"official seeds {seed_start}-{seed_start + seed_count - 1}: "
                f"{len(official)}/{len(exp_rows)} episodes"
            )
        print(f"\n=== {experiment} ({scope}) ===")
        if not official:
            print("  no episodes in the official seed pool; skipping")
            continue

        seeds_present = sorted({r["seed"] for r in official})
        n_terminated = sum(1 for r in official if not r["survived"])
        print(f"  seeds present: {len(seeds_present)}  early terminations: {n_terminated}")

        if experiment == "E1":
            metrics = E1_METRICS + E1_SECONDARY
            write_csv(os.path.join(tables_dir, "E1_tracking_all.csv"), official)
            summary = summarize_groups(official, metrics, ["condition", "baseline"], cfg_stats)
            write_csv(os.path.join(tables_dir, "E1_tracking_summary.csv"), summary)
            pairs = paired_table(
                official, E1_METRICS, ["condition"], ["C1", "C3", "C5"], cfg_stats
            )
            write_csv(os.path.join(summary_dir, "E1_paired_differences.csv"), pairs)
            if args.tables:
                table_gen.e1_compact_table(official, tables_dir, cfg_stats)
            _print_summary(summary, E1_METRICS, ["condition", "baseline"])

        elif experiment == "E2":
            write_csv(os.path.join(tables_dir, "E2_wheel_actuation_all.csv"), official)
            summary = summarize_groups(official, E2_METRICS, ["condition", "baseline"], cfg_stats)
            write_csv(os.path.join(tables_dir, "E2_wheel_actuation_summary.csv"), summary)
            pairs = paired_table(official, E2_METRICS, ["condition"], ["C1", "C2"], cfg_stats)
            write_csv(os.path.join(summary_dir, "E2_paired_differences.csv"), pairs)
            if args.tables:
                exp_cfg, _ = E.load_config("E2")
                table_gen.e2_table(
                    official, tables_dir, cfg_stats,
                    primary_condition=exp_cfg["primary_table_condition"],
                )
            _print_summary(summary, E2_METRICS[:6], ["condition", "baseline"])

        elif experiment == "E4":
            write_csv(os.path.join(tables_dir, "E4_stabilization_all.csv"), official)
            summary = summarize_groups(official, E4_METRICS, ["condition", "baseline"], cfg_stats)
            write_csv(os.path.join(tables_dir, "E4_stabilization.csv"), summary)
            if args.tables:
                table_gen.e4_table(official, tables_dir, cfg_stats)
            _print_summary(summary, E4_METRICS[:5], ["condition", "baseline"])

    print(f"\nTables -> {tables_dir}\nSummaries -> {summary_dir}")
    return 0


def _print_summary(summary, metrics, keys, limit=30):
    header = " ".join(f"{k:>12s}" for k in keys) + "   n  surv " + " ".join(
        f"{m[:18]:>20s}" for m in metrics
    )
    print(header)
    print("-" * len(header))
    for record in summary[:limit]:
        line = " ".join(f"{str(record[k]):>12s}" for k in keys)
        line += f" {record['n_episodes']:3d} {record.get('survival_rate', float('nan')):5.2f} "
        for metric in metrics:
            mean = record.get(f"{metric}_mean", float("nan"))
            lo = record.get(f"{metric}_ci_low", float("nan"))
            hi = record.get(f"{metric}_ci_high", float("nan"))
            line += f" {mean:8.4f}[{lo:.4f},{hi:.4f}]"
        print(line)
    if len(summary) > limit:
        print(f"... {len(summary) - limit} more rows (see the CSVs)")


if __name__ == "__main__":
    raise SystemExit(main())
