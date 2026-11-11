#!/usr/bin/env python3
"""Curated illustrative-seed selection -- kept strictly separate from statistics.

The official quantitative results use seeds 0-29, every applicable seed, with no
exclusions. This utility is for choosing *figures*: it searches a larger
candidate pool (0-99 by default) for

  A. representative seeds  -- principal error metric closest to that method's
     own median, so the example is typical of the method rather than flattering;
  B. high-contrast seeds   -- seeds where C1 shows a clear, measured
     coordination advantage over C3/C5 under identical commands.

Every selection records the candidate pool, the criterion, and the metric values
for every compared baseline, so a reader can see exactly why a figure's seed was
chosen. Nothing here feeds the aggregate statistics.

    python3 benchmark/showcase.py --experiment E3 --baselines C1,C3,C5
    python3 benchmark/showcase.py --experiment E1 --criterion representative
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for root in (REPO_ROOT, SIM_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

from benchmark import aggregate  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

RESULTS_ROOT = os.path.join(SIM_ROOT, "benchmark", "results")

PRIMARY_METRIC = {
    "E1": "ee_pos_rmse",
    "E2": "base_vxy_rmse",
    "E3": "ee_pos_rmse",
    "E4": "ee_world_pos_rmse",
}
CONTRAST_CRITERIA = (
    "ee_error_advantage",
    "base_velocity_advantage",
    "locomotion_induced_ee_degradation_advantage",
    "manipulation_induced_base_degradation_advantage",
)


def _index(rows):
    out: dict = {}
    for row in rows:
        out.setdefault((row["condition"], row["seed"]), {})[row["baseline"]] = row
    return out


def representative_seeds(rows, baselines, metric, top_k=3):
    """Seeds whose metric is closest to the method's own median."""
    out = []
    by_condition: dict = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    for condition, items in sorted(by_condition.items()):
        for baseline in baselines:
            values = {
                r["seed"]: r[metric]
                for r in items
                if r["baseline"] == baseline and metric in r and np.isfinite(r[metric])
            }
            if len(values) < 3:
                continue
            median = float(np.median(list(values.values())))
            ranked = sorted(values.items(), key=lambda kv: abs(kv[1] - median))
            for seed, value in ranked[:top_k]:
                row = next(
                    r for r in items if r["baseline"] == baseline and r["seed"] == seed
                )
                out.append(
                    {
                        "condition": condition,
                        "baseline": baseline,
                        "seed": int(seed),
                        "criterion": "representative",
                        "criterion_detail": (
                            f"|{metric} - median({metric})| minimal for this method; "
                            f"median = {median:.6g}"
                        ),
                        "metric": metric,
                        "value": float(value),
                        "method_median": median,
                        "survived": int(row["survived"]),
                        "termination_reason": row["termination_reason"],
                    }
                )
    return out


def high_contrast_seeds(rows, reference, others, criterion, top_k=3,
                        require_all_survive=True):
    """Seeds where ``reference`` beats every method in ``others`` by the most.

    All compared baselines must have run the identical commands (same condition,
    same seed) and -- unless disabled -- must all have completed, so the figure
    cannot be a comparison against a crashed run.
    """
    metric_for = {
        "ee_error_advantage": "ee_pos_rmse",
        "base_velocity_advantage": "base_vxy_rmse",
        "locomotion_induced_ee_degradation_advantage": "delta_ee_pos_rmse_from_base",
        "manipulation_induced_base_degradation_advantage": "delta_base_vxy_rmse_from_ee",
    }
    metric = metric_for[criterion]
    index = _index(rows)
    candidates = []
    for (condition, seed), per_baseline in index.items():
        if reference not in per_baseline:
            continue
        if not all(b in per_baseline for b in others):
            continue
        rows_here = [per_baseline[b] for b in [reference] + list(others)]
        if any(metric not in r or not np.isfinite(r[metric]) for r in rows_here):
            continue
        if require_all_survive and not all(int(r["survived"]) for r in rows_here):
            continue
        ref_value = float(per_baseline[reference][metric])
        margins = {
            b: float(per_baseline[b][metric]) - ref_value for b in others
        }
        worst_margin = min(margins.values())  # the advantage over the closest rival
        if worst_margin <= 0:
            continue
        candidates.append(
            {
                "condition": condition,
                "seed": int(seed),
                "criterion": criterion,
                "criterion_detail": (
                    f"{reference} has the lowest {metric}; margin over the closest "
                    f"rival = {worst_margin:.6g}"
                ),
                "metric": metric,
                "reference": reference,
                "margin_over_closest_rival": worst_margin,
                "values": {
                    b: float(per_baseline[b][metric]) for b in [reference] + list(others)
                },
                "margins": margins,
                "survived": {
                    b: int(per_baseline[b]["survived"]) for b in [reference] + list(others)
                },
                "termination_reasons": {
                    b: per_baseline[b]["termination_reason"]
                    for b in [reference] + list(others)
                },
            }
        )
    candidates.sort(key=lambda c: -c["margin_over_closest_rival"])

    by_condition: dict = {}
    for candidate in candidates:
        by_condition.setdefault(candidate["condition"], []).append(candidate)
    out = []
    for condition in sorted(by_condition):
        out.extend(by_condition[condition][:top_k])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=["E1", "E2", "E3", "E4"])
    parser.add_argument("--results", default=os.path.join(RESULTS_ROOT, "raw"))
    parser.add_argument("--baselines", default="C1,C3,C5")
    parser.add_argument("--reference", default="C1")
    parser.add_argument("--criterion", default="all",
                        choices=("all", "representative") + CONTRAST_CRITERIA)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--allow-terminated", action="store_true",
                        help="Allow high-contrast seeds where a baseline terminated early")
    parser.add_argument("--out", default=os.path.join(RESULTS_ROOT, "showcase",
                                                      "selected_seeds.json"))
    args = parser.parse_args(argv)

    _, global_cfg = E.load_config("E1")
    pool = global_cfg["showcase_seeds"]
    official = global_cfg["official_seeds"]
    baselines = [b.strip() for b in args.baselines.split(",")]

    rows = aggregate.load_raw(args.results, [args.experiment])
    rows = [r for r in rows if r["baseline"] in baselines]
    if not rows:
        print(f"No {args.experiment} episodes under {args.results} for {baselines}")
        return 1

    # Degradation is computed on the FULL set, before any condition filter: the
    # alpha_base = 0 and alpha_ee = 0 references live in other conditions, so
    # filtering first would silently drop both degradation metrics.
    if args.experiment == "E3":
        _, stats_cfg = E.load_config("E3")
        degradation = aggregate.e3_degradation(rows, stats_cfg["statistics"])
        deg_index = {
            (r["baseline"], r["ee_family"], r["seed"], r["alpha_base"], r["alpha_ee"]): r
            for r in degradation
        }
        for row in rows:
            key = (row["baseline"], row.get("ee_family"), row["seed"],
                   row.get("alpha_base"), row.get("alpha_ee"))
            extra = deg_index.get(key)
            if extra:
                for metric in ("delta_ee_pos_rmse_from_base",
                               "delta_base_vxy_rmse_from_ee"):
                    if metric in extra:
                        row[metric] = extra[metric]

    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        rows = [r for r in rows if r["condition"] in wanted]
        if not rows:
            print(f"No {args.experiment} episodes for conditions {sorted(wanted)}")
            return 1

    seeds_found = sorted({r["seed"] for r in rows})
    metric = PRIMARY_METRIC[args.experiment]
    selections = {
        "experiment": args.experiment,
        "generated_by": "benchmark/showcase.py",
        "purpose": (
            "Curated illustrative examples for figures and video ONLY. These "
            "seeds are NOT used for, and do not replace, the official aggregate "
            "statistics."
        ),
        "candidate_seed_pool": {
            "start": pool["start"], "count": pool["count"],
            "seeds_actually_present": seeds_found,
        },
        "official_seed_pool": {"start": official["start"], "count": official["count"]},
        "baselines_compared": baselines,
        "reference_baseline": args.reference,
        "primary_metric": metric,
        "require_all_baselines_survive": not args.allow_terminated,
        "figure_caption_requirement": (
            "Label figures built from these seeds as illustrative / curated "
            "examples, not as random representative trials."
        ),
        "selections": {},
    }

    if args.criterion in ("all", "representative"):
        selections["selections"]["representative"] = representative_seeds(
            rows, baselines, metric, args.top_k
        )
    others = [b for b in baselines if b != args.reference]
    criteria = CONTRAST_CRITERIA if args.criterion == "all" else (
        () if args.criterion == "representative" else (args.criterion,)
    )
    for criterion in criteria:
        metric_name = {
            "ee_error_advantage": "ee_pos_rmse",
            "base_velocity_advantage": "base_vxy_rmse",
            "locomotion_induced_ee_degradation_advantage": "delta_ee_pos_rmse_from_base",
            "manipulation_induced_base_degradation_advantage": "delta_base_vxy_rmse_from_ee",
        }[criterion]
        if not any(metric_name in r for r in rows):
            selections["selections"][criterion] = {
                "skipped": f"metric '{metric_name}' not available for {args.experiment}"
            }
            continue
        if not others:
            selections["selections"][criterion] = {"skipped": "no rival baselines given"}
            continue
        selections["selections"][criterion] = high_contrast_seeds(
            rows, args.reference, others, criterion, args.top_k,
            require_all_survive=not args.allow_terminated,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    existing = {}
    if os.path.isfile(args.out):
        with open(args.out, "r", encoding="utf-8") as handle:
            try:
                existing = json.load(handle)
            except json.JSONDecodeError:
                existing = {}
    if not isinstance(existing, dict) or "experiments" not in existing:
        existing = {"experiments": {}}
    existing["experiments"][args.experiment] = selections
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)

    print(f"Candidate pool: seeds {seeds_found[0]}-{seeds_found[-1]} "
          f"({len(seeds_found)} present)")
    for criterion, items in selections["selections"].items():
        if isinstance(items, dict):
            print(f"\n{criterion}: {items.get('skipped')}")
            continue
        print(f"\n{criterion} ({len(items)} selected):")
        for item in items[:12]:
            if criterion == "representative":
                print(f"  {item['condition']:>16s} {item['baseline']} "
                      f"seed {item['seed']:3d}  {item['metric']}={item['value']:.4f} "
                      f"(median {item['method_median']:.4f})")
            else:
                values = " ".join(f"{b}={v:.4f}" for b, v in sorted(item["values"].items()))
                print(f"  {item['condition']:>16s} seed {item['seed']:3d}  "
                      f"margin={item['margin_over_closest_rival']:.4f}  {values}")
    print(f"\nMetadata -> {os.path.relpath(args.out, SIM_ROOT)}")
    print("Reminder: these are curated illustrative examples, not official statistics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
