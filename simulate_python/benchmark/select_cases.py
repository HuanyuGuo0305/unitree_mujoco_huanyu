#!/usr/bin/env python3
"""Illustrative-case selection for E1 and E2 figures.

This is NOT part of the quantitative evaluation. The official statistics use
every seed in the official pool (0-59) with no exclusions; this utility picks
example trials for figures from a **disjoint** candidate pool (100-149) and
records the full ranking so a reader can see exactly why each figure's seed was
chosen.

The selection procedure is declared in ``CRITERIA`` below and is applied
mechanically:

1. **Gates** (pre-declared, applied before any ranking)
   G1 every compared baseline completed the episode;
   G2 every scored metric is finite for every compared baseline;
   G3 the commands are nontrivial -- for the seeded-random experiments, the
      episode must contain several base-command segments, must actually
      command motion, and (E1 T4) must switch the EE target inside the
      evaluation window;
2. **Scale** per metric: the median, over the gated candidates, of the mean of
   that metric across the compared baselines. A scale, taken from the pool
   itself, so the three metrics are comparable; it is not a result.
3. **Margin** per seed and metric: ``min over scored rivals (rival - C1)``,
   i.e. C1's worst-case advantage, divided by the scale.
4. **Score** per criterion: either the minimum of the normalised margins (C1
   must lead on every listed metric simultaneously) or their mean, subject to
   any ``require_positive`` metrics on which C1 must not be worse.
5. **Selection**: highest score, ties broken by the lower seed; each criterion
   takes a distinct seed, criteria resolved in listed order.

If a criterion has no candidate that satisfies it, that is reported as
``selected: null`` with the reason -- never back-filled with a weaker case.

    python3 benchmark/select_cases.py --experiment E1
    python3 benchmark/select_cases.py --experiment E1 --experiment E2
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

from benchmark import aggregate  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

RESULTS_ROOT = os.path.join(SIM_ROOT, "benchmark", "results")

LOWER_IS_BETTER = {
    "base_vxy_rmse": True, "base_wz_rmse": True, "ee_pos_rmse": True,
    "ee_ori_mae_deg": True, "total_power": True, "leg_power": True,
    "leg_acc_rms": True, "arm_acc_rms": True,
}

# ---------------------------------------------------------------- procedure
CRITERIA = {
    "E1": {
        "conditions": ["T4"],
        "primary_condition_rationale": (
            "T4 (base moving + EE random targets) is the only condition that "
            "exercises locomotion and manipulation command tracking "
            "simultaneously, so it is where an illustrative trial is "
            "informative about whole-body coordination."
        ),
        "compared": ["C1", "C3", "C5"],
        "scored_rivals": ["C3", "C5"],
        "report_metrics": ["base_vxy_rmse", "base_wz_rmse", "ee_pos_rmse",
                           "ee_ori_mae_deg"],
        "cases": [
            {
                "id": "example1_simultaneous_tracking",
                "title": "Strong simultaneous tracking",
                "intent": "C1 clearly ahead on base velocity, yaw rate AND EE "
                          "position at the same time.",
                "score": "min",
                "metrics": ["base_vxy_rmse", "base_wz_rmse", "ee_pos_rmse"],
                "require_positive": [],
            },
            {
                "id": "example2_locomotion_advantage",
                "title": "Locomotion tracking advantage",
                "intent": "C1 tracks the base velocity and yaw-rate commands "
                          "clearly better while the EE target keeps changing, "
                          "without giving up EE position accuracy.",
                "score": "mean",
                "metrics": ["base_vxy_rmse", "base_wz_rmse"],
                "require_positive": ["ee_pos_rmse"],
            },
            {
                "id": "example3_manipulation_advantage",
                "title": "Manipulation tracking advantage",
                "intent": "C1 tracks the switching EE position targets clearly "
                          "better during base motion, while still not losing "
                          "base tracking.",
                "score": "mean",
                "metrics": ["ee_pos_rmse"],
                "require_positive": ["base_vxy_rmse", "base_wz_rmse"],
            },
        ],
    },
    "E2": {
        "conditions": ["Medium"],
        "primary_condition_rationale": (
            "Medium intensity (0.50 of the shared supported command range) is "
            "the condition the official 60-seed statistics show the clearest "
            "wheel-actuation benefit in, and it is mid-range rather than at "
            "the edge of the command envelope."
        ),
        "compared": ["C1", "C2"],
        "scored_rivals": ["C2"],
        "report_metrics": ["base_vxy_rmse", "base_wz_rmse", "total_power",
                           "leg_power", "wheel_power", "leg_acc_rms", "arm_acc_rms"],
        "cases": [
            {
                "id": "case1_power_reduction",
                "title": "Reduced total mechanical power proxy",
                "intent": "Active wheels lower the total |tau * qdot| proxy.",
                "score": "mean",
                "metrics": ["total_power"],
                "require_positive": [],
            },
            {
                "id": "case2_leg_acceleration_reduction",
                "title": "Reduced leg joint acceleration",
                "intent": "Active wheels lower the leg joint-acceleration RMS.",
                "score": "mean",
                "metrics": ["leg_acc_rms"],
                "require_positive": [],
            },
            {
                "id": "case3_tracking_efficiency_tradeoff",
                "title": "Favourable tracking-efficiency trade-off",
                "intent": "C1 is simultaneously better on total power and on "
                          "both base tracking metrics -- no trade-off is being "
                          "paid for the efficiency gain.",
                "score": "min",
                "metrics": ["total_power", "base_vxy_rmse", "base_wz_rmse"],
                "require_positive": [],
            },
        ],
    },
}

# G3 nontriviality thresholds, pre-declared.
#
# The commanded-speed threshold is RELATIVE TO THE CONDITION, not an absolute
# fraction of the full command range. A fixed absolute threshold is wrong here:
# E2's intensities scale the command amplitude (Low 0.25x, Medium 0.50x, High
# 0.75x), so the expected RMS commanded fraction at Medium is ~0.29 and an
# absolute 0.30 cut rejects essentially every Medium episode by construction
# rather than for anything about that episode. (That is exactly what it did on
# the first run: 47 of 50 candidates rejected, leaving one eligible candidate
# for two of the three criteria, which would have made the "selection"
# vacuous.) The rule is therefore "at least 70% of the median RMS commanded
# speed of this condition's own candidate pool", which still rejects a
# degenerate near-stationary draw but does not depend on the intensity.
#
# The segment and switch counts stay absolute: they are intensity-independent
# properties of the schedule.
NONTRIVIAL = {
    "E1": {"min_base_segments": 4, "min_ee_switches": 2,
           "min_rms_fraction_of_condition_median": 0.70},
    "E2": {"min_base_segments": 4, "min_ee_switches": 0,
           "min_rms_fraction_of_condition_median": 0.70},
}


# ---------------------------------------------------------------- utilities
def command_diagnostics(episode_path: str, limits) -> dict:
    """Read the logged command schedule back and describe it."""
    full = os.path.join(SIM_ROOT, episode_path) if not os.path.isabs(episode_path) else episode_path
    path = os.path.join(full, "commands.csv")
    if not os.path.isfile(path):
        return {}
    t, vx, vy, wz, ee = [], [], [], [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["evaluated"])) != 1:
                continue
            t.append(float(row["t"]))
            vx.append(float(row["vx_cmd"]))
            vy.append(float(row["vy_cmd"]))
            wz.append(float(row["wz_cmd"]))
            ee.append([float(row[f"ee_cmd_kp{i}"]) for i in range(9)])
    if not t:
        return {}
    cmd = np.column_stack([vx, vy, wz])
    ee = np.asarray(ee)
    changes = int(np.sum(np.any(np.abs(np.diff(cmd, axis=0)) > 1e-9, axis=1)))
    ee_changes = int(np.sum(np.any(np.abs(np.diff(ee, axis=0)) > 1e-9, axis=1)))
    lim = np.asarray(limits, dtype=float)
    rms_frac = float(np.sqrt(np.mean(np.sum((cmd / lim) ** 2, axis=1) / 3.0)))
    return {
        "n_base_command_changes": changes,
        "n_base_segments": changes + 1,
        "n_ee_command_changes": ee_changes,
        "rms_commanded_fraction_of_range": rms_frac,
        "max_abs_vx_cmd": float(np.max(np.abs(vx))),
        "max_abs_vy_cmd": float(np.max(np.abs(vy))),
        "max_abs_wz_cmd": float(np.max(np.abs(wz))),
        "eval_samples": len(t),
    }


def _index_by_seed_cond(rows):
    out: dict = {}
    for row in rows:
        out.setdefault((row["condition"], row["seed"]), {})[row["baseline"]] = row
    return out


def build_candidates(experiment, raw_root, pool, spec, limits, global_cfg):
    """Gate the candidate pool and return (candidates, gate_log)."""
    rows = aggregate.load_raw(raw_root, [experiment])
    rows = aggregate.filter_official(rows, pool["start"], pool["count"])

    official = global_cfg["official_seeds"]
    overlap = set(range(pool["start"], pool["start"] + pool["count"])) & set(
        range(official["start"], official["start"] + official["count"])
    )
    if overlap:
        raise SystemExit(
            f"Candidate pool overlaps the official pool at seeds {sorted(overlap)}. "
            "An illustrative figure seed must not also be an official statistical "
            "sample; fix showcase_seeds in configs/baselines.yaml."
        )

    compared = spec["compared"]
    gate_log = []
    candidates = []

    thresholds = NONTRIVIAL[experiment]
    index = _index_by_seed_cond(rows)

    # Pass 1: read every candidate's command schedule, so the commanded-speed
    # gate can be expressed relative to this condition's own pool.
    entries = []
    for (condition, seed), per_baseline in sorted(index.items()):
        if condition not in spec["conditions"]:
            continue
        entry = {"experiment": experiment, "condition": condition, "seed": int(seed),
                 "_per_baseline": per_baseline}
        if all(b in per_baseline for b in compared):
            entry["commands"] = command_diagnostics(
                per_baseline[compared[0]]["path"], limits)
        else:
            entry["commands"] = {}
        entries.append(entry)

    condition_median_rms = {}
    for condition in spec["conditions"]:
        values = [e["commands"]["rms_commanded_fraction_of_range"]
                  for e in entries
                  if e["condition"] == condition
                  and "rms_commanded_fraction_of_range" in e["commands"]]
        condition_median_rms[condition] = float(np.median(values)) if values else float("nan")

    # Pass 2: gate.
    metrics_needed = sorted({m for case in spec["cases"] for m in
                             case["metrics"] + case["require_positive"]})
    for entry in entries:
        per_baseline = entry.pop("_per_baseline")
        condition = entry["condition"]
        reasons = []
        if not all(b in per_baseline for b in compared):
            reasons.append("G1: a compared baseline is missing")
        elif not all(int(per_baseline[b]["survived"]) for b in compared):
            reasons.append("G1: a compared baseline terminated early")
        if all(b in per_baseline for b in compared):
            for metric in metrics_needed:
                if not all(np.isfinite(per_baseline[b].get(metric, np.nan))
                           for b in compared):
                    reasons.append(f"G2: {metric} not finite")
            diag = entry["commands"]
            if diag:
                floor = thresholds["min_rms_fraction_of_condition_median"] * \
                    condition_median_rms[condition]
                entry["condition_median_rms_commanded_fraction"] = \
                    condition_median_rms[condition]
                entry["rms_floor_applied"] = floor
                if diag["n_base_segments"] < thresholds["min_base_segments"]:
                    reasons.append(
                        f"G3: only {diag['n_base_segments']} base segments "
                        f"(< {thresholds['min_base_segments']})")
                if diag["n_ee_command_changes"] < thresholds["min_ee_switches"]:
                    reasons.append(
                        f"G3: only {diag['n_ee_command_changes']} EE switches "
                        f"(< {thresholds['min_ee_switches']})")
                if np.isfinite(floor) and \
                        diag["rms_commanded_fraction_of_range"] < floor:
                    reasons.append(
                        f"G3: RMS commanded speed "
                        f"{diag['rms_commanded_fraction_of_range']:.3f} of range "
                        f"(< {floor:.3f}, i.e. "
                        f"{thresholds['min_rms_fraction_of_condition_median']:.0%} "
                        f"of this condition's pool median "
                        f"{condition_median_rms[condition]:.3f})")
            entry["metrics"] = {
                b: {m: per_baseline[b].get(m) for m in spec["report_metrics"]}
                for b in compared
            }
            entry["survived"] = {b: int(per_baseline[b]["survived"]) for b in compared}
            entry["paths"] = {b: per_baseline[b]["path"] for b in compared}
        entry["gated_out"] = reasons
        gate_log.append(entry)
        if not reasons:
            candidates.append(entry)
    return candidates, gate_log


def compute_scales(candidates, spec):
    """Median across gated candidates of the across-baseline mean, per metric."""
    metrics = sorted({m for case in spec["cases"]
                      for m in case["metrics"] + case["require_positive"]})
    scales = {}
    for metric in metrics:
        pooled = []
        for cand in candidates:
            values = [cand["metrics"][b][metric] for b in spec["compared"]
                      if cand["metrics"][b].get(metric) is not None]
            if values:
                pooled.append(float(np.mean(values)))
        scales[metric] = float(np.median(pooled)) if pooled else float("nan")
    return scales


def score_candidates(candidates, spec, scales):
    """Attach the normalised worst-case margins and every criterion's score."""
    for cand in candidates:
        margins, z = {}, {}
        for metric, scale in scales.items():
            ref = cand["metrics"]["C1"].get(metric)
            if ref is None:
                continue
            rival_margins = {
                r: float(cand["metrics"][r][metric]) - float(ref)
                for r in spec["scored_rivals"]
                if cand["metrics"][r].get(metric) is not None
            }
            if not rival_margins:
                continue
            worst = min(rival_margins.values())
            margins[metric] = {"per_rival": rival_margins, "worst_case": worst}
            z[metric] = worst / scale if scale and np.isfinite(scale) and scale != 0 else float("nan")
        cand["margins"] = margins
        cand["z"] = z
        cand["scores"] = {}
        for case in spec["cases"]:
            if "cells" in case and cand.get("cell") not in case["cells"]:
                cand["scores"][case["id"]] = {
                    "eligible": False, "score": float("-inf"),
                    "reason": f"cell {cand.get('cell')} not in {case['cells']}",
                }
                continue
            zs = [z.get(m, float("nan")) for m in case["metrics"]]
            gate = [z.get(m, float("nan")) for m in case["require_positive"]]
            if any(not np.isfinite(v) for v in zs + gate):
                cand["scores"][case["id"]] = {
                    "eligible": False, "score": float("-inf"),
                    "reason": "a scored metric is not available",
                }
                continue
            if any(v <= 0 for v in gate):
                failed = [m for m, v in zip(case["require_positive"], gate) if v <= 0]
                cand["scores"][case["id"]] = {
                    "eligible": False, "score": float("-inf"),
                    "reason": f"C1 not better on required metric(s) {failed}",
                }
                continue
            value = float(min(zs)) if case["score"] == "min" else float(np.mean(zs))
            cand["scores"][case["id"]] = {
                "eligible": True, "score": value,
                "normalised_margins": {m: float(z[m]) for m in case["metrics"]},
                "reason": "",
            }
    return candidates


def select(candidates, spec):
    """Highest score per criterion, distinct seeds, criteria in listed order."""
    chosen, used_seeds, out = {}, set(), []
    for case in spec["cases"]:
        eligible = [c for c in candidates
                    if c["scores"][case["id"]]["eligible"]
                    and c["scores"][case["id"]]["score"] > 0
                    and c["seed"] not in used_seeds]
        ranking = sorted(
            eligible, key=lambda c: (-c["scores"][case["id"]]["score"], c["seed"])
        )
        record = {
            "case_id": case["id"], "title": case["title"], "intent": case["intent"],
            "score_rule": (
                f"{case['score']}(normalised worst-case margin over "
                f"{case['metrics']})"
                + (f", required C1 better on {case['require_positive']}"
                   if case["require_positive"] else "")
            ),
            "n_eligible": len(eligible),
            "n_excluded_already_used": sum(
                1 for c in candidates
                if c["scores"][case["id"]]["eligible"]
                and c["scores"][case["id"]]["score"] > 0 and c["seed"] in used_seeds),
            "ranking_top10": [
                {"seed": c["seed"], "condition": c["condition"],
                 "score": c["scores"][case["id"]]["score"],
                 "normalised_margins": c["scores"][case["id"]].get("normalised_margins")}
                for c in ranking[:10]
            ],
        }
        if not ranking:
            record["selected"] = None
            record["selection_failure_reason"] = (
                "No gated candidate in the pool has a positive score for this "
                "criterion. Reported as-is; no weaker case was substituted."
            )
        else:
            best = ranking[0]
            used_seeds.add(best["seed"])
            chosen[case["id"]] = best
            record["selected"] = {
                "seed": best["seed"], "condition": best["condition"],
                "cell": best.get("cell"), "ee_family": best.get("ee_family"),
                "score": best["scores"][case["id"]]["score"],
                "metrics": best["metrics"], "survived": best["survived"],
                "normalised_margins": best["scores"][case["id"]]["normalised_margins"],
                "worst_case_margins": {
                    m: best["margins"][m]["worst_case"] for m in best["margins"]},
                "per_rival_margins": {
                    m: best["margins"][m]["per_rival"] for m in best["margins"]},
                "paths": best["paths"],
                "commands": best.get("commands", {}),
            }
        out.append(record)
    return out, chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", action="append", required=True,
                        choices=["E1", "E2"])
    parser.add_argument("--results", default=os.path.join(RESULTS_ROOT, "raw"))
    parser.add_argument("--out", default=RESULTS_ROOT)
    args = parser.parse_args(argv)

    _, global_cfg = E.load_config("E1")
    pool = global_cfg["showcase_seeds"]
    limits = global_cfg["base_command_limits"]

    for experiment in args.experiment:
        spec = CRITERIA[experiment]
        print("=" * 78)
        print(f"{experiment} illustrative-case selection  |  candidate pool "
              f"{pool['start']}-{pool['start']+pool['count']-1}")
        print("=" * 78)
        candidates, gate_log = build_candidates(
            experiment, args.results, pool, spec, limits, global_cfg
        )
        if not candidates:
            print("  no candidate survived the gates; nothing selected")
        scales = compute_scales(candidates, spec)
        candidates = score_candidates(candidates, spec, scales)
        records, chosen = select(candidates, spec)

        gate_reasons: dict = {}
        for entry in gate_log:
            for reason in entry["gated_out"]:
                gate_reasons[reason.split(":")[0]] = gate_reasons.get(
                    reason.split(":")[0], 0) + 1
        manifest = {
            "experiment": experiment,
            "generated_by": "benchmark/select_cases.py",
            "status": "ILLUSTRATIVE EXAMPLES ONLY",
            "label_required_on_every_artifact": (
                "Selected illustrative trials from a separately evaluated "
                "candidate seed pool. Not benchmark average performance."
            ),
            "candidate_seed_pool": {
                "start": pool["start"], "count": pool["count"],
                "seeds_present": sorted({c["seed"] for c in gate_log}),
                "disjoint_from_official_pool": True,
                "official_pool": global_cfg["official_seeds"],
            },
            "conditions_searched": spec["conditions"],
            "primary_condition_rationale": spec["primary_condition_rationale"],
            "baselines_compared": spec["compared"],
            "baselines_scored": spec["scored_rivals"],
            "procedure": {
                "gates": {
                    "G1": "every compared baseline completed the episode",
                    "G2": "every scored metric finite for every baseline",
                    "G3": NONTRIVIAL.get(experiment, "not applicable (deterministic commands)"),
                    "G4": "not applicable",
                },
                "margin": "min over scored rivals of (rival - C1), per metric",
                "normalisation_scale": "median over gated candidates of the "
                                       "across-baseline mean of that metric",
                "scales_used": scales,
                "tie_break": "lower seed id",
                "dedup": "each criterion takes a distinct seed, criteria in listed order",
            },
            "gate_summary": {
                "n_candidate_episodes_examined": len(gate_log),
                "n_passed_all_gates": len(candidates),
                "n_gated_out_by_gate": gate_reasons,
            },
            "all_candidates": [
                {
                    "seed": c["seed"], "condition": c["condition"],
                    "gated_out": c["gated_out"],
                    "metrics": c.get("metrics"),
                    "normalised_margins": c.get("z"),
                    "scores": {k: v["score"] for k, v in c.get("scores", {}).items()},
                }
                for c in sorted(gate_log, key=lambda c: (c["condition"], c["seed"]))
            ],
            "cases": records,
        }
        show_dir = os.path.join(args.out, "showcase")
        os.makedirs(show_dir, exist_ok=True)
        mpath = os.path.join(show_dir, f"{experiment}_selection_manifest.json")
        with open(mpath, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, default=float)

        flat = []
        for record in records:
            sel = record["selected"]
            if not sel:
                continue
            for baseline in spec["compared"]:
                row = {
                    "case_id": record["case_id"], "case_title": record["title"],
                    "experiment": experiment, "condition": sel["condition"],
                    "seed": sel["seed"], "baseline": baseline,
                    "survived": sel["survived"][baseline],
                    "selection_score": record["selected"]["score"],
                    "status": "illustrative_example_not_population_average",
                }
                row.update({m: sel["metrics"][baseline].get(m)
                            for m in spec["report_metrics"]})
                flat.append(row)
        csv_dir = os.path.join(args.out, "tables")
        csv_name = f"{experiment}_selected_3seeds.csv"
        aggregate.write_csv(os.path.join(csv_dir, csv_name), flat)

        print(f"  examined {len(gate_log)} candidate episodes, "
              f"{len(candidates)} passed all gates")
        if gate_reasons:
            print(f"  gated out: {gate_reasons}")
        print(f"  scales: " + ", ".join(f"{k}={v:.4g}" for k, v in scales.items()))
        for record in records:
            sel = record["selected"]
            print(f"\n  [{record['case_id']}] {record['title']}")
            print(f"    rule: {record['score_rule']}")
            print(f"    eligible candidates: {record['n_eligible']}")
            if not sel:
                print(f"    SELECTED: none -- {record['selection_failure_reason']}")
                continue
            print(f"    SELECTED seed {sel['seed']} ({sel['condition']}), "
                  f"score {sel['score']:.3f}")
            for metric in spec["report_metrics"]:
                values = " ".join(
                    f"{b}={sel['metrics'][b][metric]:.4f}" if sel['metrics'][b].get(metric) is not None else f"{b}=n/a"
                    for b in spec["compared"])
                print(f"      {metric:34s} {values}")
        print(f"\n  manifest -> {os.path.relpath(mpath, SIM_ROOT)}")
        print(f"  selected table -> {os.path.relpath(os.path.join(csv_dir, csv_name), SIM_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
