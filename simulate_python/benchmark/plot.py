#!/usr/bin/env python3
"""Figure generation. Each experiment's figures can be produced independently.

    python3 benchmark/plot.py --results benchmark/results/raw
    python3 benchmark/plot.py --experiments E4 --seed 7 --snapshots --video
    python3 benchmark/plot.py --side-by-side E1 --condition T4 --seed 12
"""

from __future__ import annotations

import argparse
import os
import sys

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for root in (REPO_ROOT, SIM_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

from benchmark import aggregate  # noqa: E402
from benchmark.plots import figures  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

RESULTS_ROOT = os.path.join(SIM_ROOT, "benchmark", "results")
FIG_DIR = os.path.join(RESULTS_ROOT, "figures")
TABLE_DIR = os.path.join(RESULTS_ROOT, "tables")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=os.path.join(RESULTS_ROOT, "raw"))
    parser.add_argument("--out", default=FIG_DIR)
    parser.add_argument("--experiments", default=None, help="Comma-separated subset")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Seed for E4 snapshots/video")
    parser.add_argument("--baseline", default="C1", help="Baseline for E4 snapshots/video")
    parser.add_argument("--snapshots", action="store_true", help="Render E4 snapshots")
    parser.add_argument("--video", action="store_true", help="Render an E4 MP4")
    parser.add_argument("--side-by-side", default=None,
                        help="Experiment ID for a curated side-by-side comparison")
    parser.add_argument("--condition", default=None, help="Condition for --side-by-side")
    args = parser.parse_args(argv)

    _, global_cfg = E.load_config("E1")
    cfg_stats = global_cfg["statistics"]
    seed_start = args.seed_start if args.seed_start is not None else global_cfg["official_seeds"]["start"]
    seed_count = args.num_seeds if args.num_seeds is not None else global_cfg["official_seeds"]["count"]

    if args.side_by_side:
        return _side_by_side(args, global_cfg)

    experiments = [e.strip() for e in args.experiments.split(",")] if args.experiments else None
    rows = aggregate.load_raw(args.results, experiments)
    if not rows:
        print(f"No episode results under {args.results}")
        return 1
    found = sorted({r["experiment"] for r in rows})
    os.makedirs(args.out, exist_ok=True)

    for experiment in found:
        exp_rows = [r for r in rows if r["experiment"] == experiment]
        official = (
            exp_rows if args.all_seeds
            else aggregate.filter_official(exp_rows, seed_start, seed_count)
        )
        if not official:
            print(f"{experiment}: no episodes in the official seed pool")
            continue
        print(f"\n=== {experiment} figures ({len(official)} episodes) ===")

        if experiment == "E1":
            summary = aggregate.summarize_groups(
                official, aggregate.E1_METRICS, ["condition", "baseline"], cfg_stats
            )
            figures.e1_table_preview(summary, args.out)
            figures.e1_condition_bars(summary, args.out)

        elif experiment == "E2":
            figures.e2_intensity_curves(official, args.out, cfg_stats)

        elif experiment == "E4":
            exp_cfg, _ = E.load_config("E4")
            for baseline in sorted({r["baseline"] for r in official}):
                episode = os.path.join(
                    args.results, "E4", baseline, "chicken_head", f"seed_{args.seed:03d}"
                )
                if os.path.isdir(episode):
                    figures.e4_figures(episode, args.out, baseline)
            if args.snapshots:
                figures.e4_snapshots(args.baseline, args.seed, args.out, exp_cfg)
            if args.video:
                figures.e4_video(
                    args.baseline, args.seed,
                    os.path.join(RESULTS_ROOT, "videos",
                                 f"E4_{args.baseline}_seed{args.seed:03d}.mp4"),
                    exp_cfg,
                )

    print(f"\nFigures -> {args.out}")
    return 0


def _side_by_side(args, global_cfg):
    experiment = args.side_by_side
    condition = args.condition
    if condition is None:
        raise SystemExit("--side-by-side requires --condition")
    base_dir = os.path.join(args.results, experiment)
    if not os.path.isdir(base_dir):
        raise SystemExit(f"no results for {experiment} under {args.results}")
    episode_dirs = {}
    for baseline in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, baseline, condition, f"seed_{args.seed:03d}")
        if os.path.isdir(path):
            episode_dirs[baseline] = path
    if not episode_dirs:
        raise SystemExit(
            f"no episodes for {experiment}/{condition}/seed_{args.seed:03d}"
        )
    figures.side_by_side_trajectories(
        episode_dirs, args.out,
        f"{experiment}_side_by_side_{condition}_seed{args.seed:03d}",
        title=f"{experiment} {condition}, seed {args.seed}",
    )
    print(f"\nFigures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
