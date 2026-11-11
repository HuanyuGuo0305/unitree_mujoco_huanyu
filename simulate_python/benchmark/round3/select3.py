#!/usr/bin/env python3
"""Phase F: pick three illustrative trials per experiment, reproducibly.

These are **selected illustrative trials**. They are single episodes chosen to
show a behaviour clearly in a figure. They are NOT evidence about average
performance -- that is what the 50-seed summaries are for -- and every output
file and figure caption says so.

Honesty rules built into the procedure
--------------------------------------
1. The criteria below are declared in code BEFORE any ranking is computed, and
   are not edited to change which seed wins. The scoring function, the gates
   and the tie-break are fixed.
2. The candidate pool is the official 50 seeds. Selecting an illustrative seed
   does not remove it from, or change, any aggregate: the summaries always use
   all 50. Because the pool is the official population, every selected seed is
   reported together with its RANK and PERCENTILE among the 50, so a reader can
   see immediately that it is an extreme draw rather than a typical one.
3. The complete ranking of every candidate -- selected or not, eligible or not,
   with its gate verdict and its score -- is written to
   ``showcase/<E>_candidate_ranking.csv``. Nothing is hidden.
4. Gates reject trials that would make a figure misleading (an early
   termination, a non-finite metric, a near-stationary command). A gate can
   reject a trial that would have flattered C1; that is the point, and the
   rejections are listed.
5. If no candidate satisfies a criterion, the criterion is reported as
   UNSATISFIED with the reason. No substitute is invented and no threshold is
   relaxed afterwards.
6. Every selected case stores ALL measured metrics for ALL compared baselines,
   including the metrics on which C1 loses.

Scoring
-------
Each criterion is a list of terms ``(metric, weight)``. For a term, the per-seed
advantage is ``min over rivals (rival - C1)`` for a lower-is-better metric, so
a term only scores well if C1 beats EVERY rival on it. The advantage is
normalised by the condition's pooled median of that metric across all
baselines and seeds, which makes terms in different units comparable and keeps
the score invariant to unit choice.

``combine = "min"`` scores the criterion as the SMALLEST normalised term, so a
"simultaneous" criterion cannot be won by one huge term and one bad one.
``combine = "mean"`` averages them. ``require_positive`` lists terms that must
be strictly positive, and ``require_negative`` lists terms that must be
strictly negative -- used by the trade-off criterion, which deliberately
requires C1 to be WORSE on tracking while better on power.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from benchmark.round3 import common

# ==========================================================================
# Pre-declared criteria
# ==========================================================================
CRITERIA = {
    "E1": {
        "condition": "T4",
        "rivals": ["C3", "C5"],
        "pool_note": (
            "T4 (base moving + EE random targets) is the only condition in which "
            "both subsystems are commanded to move, so it is the only one where a "
            "simultaneous-tracking claim can be illustrated."
        ),
        "cases": [
            {
                "name": "case1_simultaneous_tracking",
                "intent": (
                    "C1 ahead of BOTH C3 and C5 on base XY velocity AND on EE "
                    "position in the same trial -- the simultaneous-tracking claim."
                ),
                "terms": [("base_vxy_rmse_mps", 1.0), ("ee_position_rmse_m", 1.0)],
                "combine": "min",
                "require_positive": ["base_vxy_rmse_mps", "ee_position_rmse_m"],
            },
            {
                "name": "case2_base_velocity_advantage",
                "intent": (
                    "Largest C1 locomotion-tracking advantage: base XY velocity and "
                    "base yaw rate, against both rivals."
                ),
                "terms": [("base_vxy_rmse_mps", 1.0), ("base_yaw_rmse_radps", 1.0)],
                "combine": "min",
                "require_positive": ["base_vxy_rmse_mps", "base_yaw_rmse_radps"],
            },
            {
                "name": "case3_dynamic_ee_position_advantage",
                "intent": (
                    "Largest C1 advantage in tracking a target that is replaced "
                    "every 3 s, i.e. dynamic EE position tracking."
                ),
                "terms": [("ee_position_rmse_m", 1.0)],
                "combine": "mean",
                "require_positive": ["ee_position_rmse_m"],
            },
        ],
    },
    "E2": {
        "condition": "Medium",
        "rivals": ["C2"],
        "pool_note": (
            "Medium (per-axis envelope 0.40 m/s and 0.40 rad/s) is E2's primary "
            "table condition and sits between the two extremes of the sweep."
        ),
        "cases": [
            {
                "name": "case1_mechanical_power_advantage",
                "intent": "Largest reduction in the total mechanical power proxy.",
                "terms": [("total_power_w", 1.0)],
                "combine": "mean",
                "require_positive": ["total_power_w"],
            },
            {
                "name": "case2_leg_smoothness_advantage",
                "intent": (
                    "Largest reduction in leg joint-acceleration RMS together with "
                    "a reduction in leg mechanical power -- the smoothness claim."
                ),
                "terms": [("leg_acc_rms_radps2", 1.0), ("leg_power_w", 1.0)],
                "combine": "min",
                "require_positive": ["leg_acc_rms_radps2", "leg_power_w"],
            },
            {
                "name": "case3_tracking_efficiency_tradeoff",
                "intent": (
                    "An honest trade-off: C1 uses less total power while tracking "
                    "base XY velocity WORSE than C2. Requires the tracking term to "
                    "be negative, so this criterion can only ever select a trial "
                    "that contradicts a uniform-C1-advantage reading."
                ),
                "terms": [("total_power_w", 1.0)],
                "combine": "mean",
                "require_positive": ["total_power_w"],
                "require_negative": ["base_vxy_rmse_mps"],
            },
        ],
    },
    "E3": {
        "rivals": ["C3", "C5"],
        "pool_note": (
            "Each E3 criterion names the intensity cell it must come from, chosen "
            "for what the cell means, not for which seed wins in it."
        ),
        "cases": [
            {
                "name": "case1_high_speed_ee_tracking",
                "cell": "S02",
                "intent": (
                    "Fastest EE trajectory with the base still (alpha_ee = 1, "
                    "alpha_base = 0): strong C1 dynamic EE position tracking, "
                    "isolated from any locomotion effect."
                ),
                "terms": [("ee_position_rmse_m", 1.0)],
                "combine": "mean",
                "require_positive": ["ee_position_rmse_m"],
            },
            {
                "name": "case2_combined_motion_base_tracking",
                "cell": "S22",
                "intent": (
                    "Both subsystems at full intensity: lower C1 base XY velocity "
                    "tracking error while the arm is also moving at full speed."
                ),
                "terms": [("base_vxy_rmse_mps", 1.0)],
                "combine": "mean",
                "require_positive": ["base_vxy_rmse_mps"],
            },
            {
                "name": "case3_reduced_coupling",
                "cell": "S22",
                "intent": (
                    "Reduced locomotion-manipulation coupling: C1 shows a smaller "
                    "locomotion-induced EE error increase AND a smaller "
                    "manipulation-induced base error increase than both rivals, in "
                    "the same trial."
                ),
                "terms": [("locomotion_induced_ee", 1.0),
                          ("manipulation_induced_base", 1.0)],
                "combine": "min",
                "require_positive": ["locomotion_induced_ee",
                                     "manipulation_induced_base"],
            },
        ],
    },
}

# Nontriviality gate. A figure of a trial in which nothing was commanded to
# move would be meaningless, so a candidate must carry a real task.
#
# The commanded-motion threshold is CONDITION-RELATIVE (a fraction of that
# condition's own pool median), not absolute. An absolute threshold is wrong
# here: an intensity level scales the command amplitude, so one fixed cut
# either rejects a whole low-intensity condition by construction or waves a
# high-intensity one through. Round 2 hit exactly that failure -- an absolute
# 0.30-of-range cut rejected 47 of 50 Medium E2 candidates, because the
# expected RMS at 0.50x intensity is ~0.29 -- and the relative rule replaced it.
NONTRIVIAL = {
    "E1": {"min_base_segments": 8, "min_ee_switches": 8,
           "min_rms_fraction_of_condition_median": 0.70},
    "E2": {"min_base_segments": 6, "min_ee_switches": 0,
           "min_rms_fraction_of_condition_median": 0.70},
    "E3": {"min_base_segments": 0, "min_ee_switches": 0,
           "min_rms_fraction_of_condition_median": 0.70},
}


# ==========================================================================
def _commanded_motion_rms(path: str) -> dict:
    """Commanded-motion descriptors, read from the episode's own commands.csv.

    Segment and switch counts are counted HERE rather than taken from an
    exported metrics column, so the gate cannot be weakened by a column that a
    particular experiment's per-seed export happens not to carry. (It was:
    E2's export has no ``n_base_switches_in_eval`` column, so reading the count
    from the row silently gave 1 segment for every E2 candidate and the gate
    rejected all 50.)
    """
    cmd = common.read_csv(os.path.join(path, "commands.csv"))
    keep = [i for i, r in enumerate(cmd) if int(r["evaluated"]) == 1]
    vx = np.asarray([float(cmd[i]["vx_cmd"]) for i in keep])
    vy = np.asarray([float(cmd[i]["vy_cmd"]) for i in keep])
    wz = np.asarray([float(cmd[i]["wz_cmd"]) for i in keep])
    ee = np.stack([[float(cmd[i][f"ee_cmd_kp{j}"]) for j in range(9)] for i in keep])
    ee_pos = ee[:, 0:3]
    ee_speed = (np.linalg.norm(np.diff(ee_pos, axis=0), axis=1) / 0.02
                if len(ee_pos) > 1 else np.zeros(1))
    base_cmd = np.stack([vx, vy, wz], axis=1)
    n_base_changes = (int(np.sum(np.any(np.abs(np.diff(base_cmd, axis=0)) > 1e-12, axis=1)))
                      if len(base_cmd) > 1 else 0)
    n_ee_changes = (int(np.sum(np.any(np.abs(np.diff(ee, axis=0)) > 1e-9, axis=1)))
                    if len(ee) > 1 else 0)
    return {
        "rms_commanded_base_speed": float(np.sqrt(np.mean(vx**2 + vy**2))),
        "rms_commanded_yaw_rate": float(np.sqrt(np.mean(wz**2))),
        "rms_commanded_ee_speed": float(np.sqrt(np.mean(ee_speed**2))),
        "n_base_segments": n_base_changes + 1,
        "n_ee_switches": n_ee_changes,
    }


def _advantage(per_baseline: dict, metric: str, rivals: list[str]) -> float:
    """min over rivals of (rival - C1): positive only if C1 beat every rival."""
    c1 = per_baseline["C1"][metric]
    vals = [per_baseline[r][metric] - c1 for r in rivals if r in per_baseline]
    return float(min(vals)) if vals else float("nan")


def _score(case, per_baseline, rivals, scale) -> tuple[float, dict]:
    terms = {}
    for metric, weight in case["terms"]:
        adv = _advantage(per_baseline, metric, rivals)
        denom = scale.get(metric, float("nan"))
        terms[metric] = {
            "advantage": adv,
            "normalised": float(weight * adv / denom) if denom and np.isfinite(denom) else float("nan"),
        }
    values = [t["normalised"] for t in terms.values()]
    if not values or not all(np.isfinite(v) for v in values):
        return float("nan"), terms
    score = float(min(values)) if case["combine"] == "min" else float(np.mean(values))
    return score, terms


def _gates(case, per_baseline, rivals, motion, scale, cfg_nontrivial, terms) -> list[str]:
    """Reasons this candidate is NOT eligible. Empty list = eligible."""
    reasons = []
    for name, row in per_baseline.items():
        if not int(row.get("survived", 0)):
            reasons.append(f"G1_terminated_{name}({row.get('termination_reason','')})")
    for name, row in per_baseline.items():
        for metric, _w in case["terms"]:
            v = row.get(metric)
            if v is None or not np.isfinite(float(v)):
                reasons.append(f"G2_nonfinite_{name}_{metric}")
    if motion:
        if motion["n_base_segments"] < cfg_nontrivial["min_base_segments"]:
            reasons.append(
                f"G3_too_few_base_segments({motion['n_base_segments']}"
                f"<{cfg_nontrivial['min_base_segments']})"
            )
        if motion["n_ee_switches"] < cfg_nontrivial["min_ee_switches"]:
            reasons.append(
                f"G3_too_few_ee_switches({motion['n_ee_switches']}"
                f"<{cfg_nontrivial['min_ee_switches']})"
            )
        frac = cfg_nontrivial["min_rms_fraction_of_condition_median"]
        for key in ("rms_commanded_base_speed", "rms_commanded_ee_speed"):
            median = scale.get(key)
            if median is None or not np.isfinite(median) or median <= 1e-12:
                continue  # that channel is not commanded to move in this condition
            if motion[key] < frac * median:
                reasons.append(
                    f"G3_{key}_below_{frac:g}x_condition_median"
                    f"({motion[key]:.4f}<{frac*median:.4f})"
                )
    for metric in case.get("require_positive", []):
        adv = terms.get(metric, {}).get("advantage", float("nan"))
        if not (np.isfinite(adv) and adv > 0.0):
            reasons.append(f"G4_require_positive_{metric}({adv:.6g})")
    for metric in case.get("require_negative", []):
        adv = _advantage(per_baseline, metric, rivals)
        if not (np.isfinite(adv) and adv < 0.0):
            reasons.append(f"G4_require_negative_{metric}({adv:.6g})")
    return reasons


# ==========================================================================
def _build_candidates(experiment, rows, condition_filter, metrics_for_scale,
                      rivals, need_motion=True):
    """(seed -> {baseline -> row}) plus the condition-relative scale."""
    sel = [r for r in rows if condition_filter(r)]
    by_seed: dict = {}
    for r in sel:
        by_seed.setdefault(int(r["seed"]), {})[r["baseline"]] = r
    # Scale: pooled median across ALL baselines and seeds in this condition.
    scale = {}
    for metric in metrics_for_scale:
        vals = [float(r[metric]) for r in sel
                if metric in r and np.isfinite(float(r[metric]))]
        scale[metric] = float(np.median(vals)) if vals else float("nan")

    motion: dict = {}
    if need_motion:
        for seed, per_baseline in by_seed.items():
            ref = per_baseline.get("C1")
            if ref is None:
                continue
            path = os.path.join(common.BENCH, ref["path"])
            motion[seed] = _commanded_motion_rms(path)
        for key in ("rms_commanded_base_speed", "rms_commanded_ee_speed",
                    "rms_commanded_yaw_rate"):
            vals = [m[key] for m in motion.values()]
            scale[key] = float(np.median(vals)) if vals else float("nan")
    return by_seed, scale, motion


def select_experiment(experiment, out_dir, showcase_dir) -> dict:
    spec = CRITERIA[experiment]
    rivals = spec["rivals"]
    rows = common.read_csv(os.path.join(out_dir, "per_seed.csv"))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["survived"] = int(r["survived"])

    deg_index: dict = {}
    if experiment == "E3":
        deg = common.read_csv(os.path.join(out_dir, "per_seed_degradation.csv"))
        for r in deg:
            key = (r["baseline"], r["trajectory_family"], int(r["seed"]), r["condition"])
            deg_index.setdefault(key, {})[r["degradation_type"]] = float(r["degradation"])

    ranking_rows, selected, unsatisfied = [], [], []
    used_seeds: set = set()

    for case in spec["cases"]:
        cell = case.get("cell")
        condition = case.get("condition", spec.get("condition"))
        if experiment == "E3":
            groups = sorted({r["trajectory_family"] for r in rows})
        else:
            groups = [None]

        metrics_for_scale = sorted({m for m, _ in case["terms"]}
                                   | set(case.get("require_negative", [])))
        metrics_for_scale = [m for m in metrics_for_scale if not m.startswith(
            ("locomotion_induced", "manipulation_induced"))]
        if experiment == "E1":
            metrics_for_scale = list({*metrics_for_scale, "base_vxy_rmse_mps",
                                      "ee_position_rmse_m", "base_yaw_rmse_radps"})

        candidates = []
        for family in groups:
            def keep(r, cell=cell, condition=condition, family=family):
                if experiment == "E3":
                    return r["condition"].startswith(cell + "_") and \
                        r["trajectory_family"] == family
                return r["condition"] == condition
            by_seed, scale, motion = _build_candidates(
                experiment, rows, keep, metrics_for_scale, rivals,
                need_motion=(experiment != "E3"),
            )
            for seed, per_baseline in sorted(by_seed.items()):
                if "C1" not in per_baseline:
                    continue
                pb = {k: {kk: (float(vv) if _isnum(vv) else vv) for kk, vv in v.items()}
                      for k, v in per_baseline.items()}
                if experiment == "E3":
                    cond_id = per_baseline["C1"]["condition"]
                    for name, row in pb.items():
                        d = deg_index.get((name, family, seed, cond_id), {})
                        row["locomotion_induced_ee"] = d.get(
                            "locomotion_induced_ee", float("nan"))
                        row["manipulation_induced_base"] = d.get(
                            "manipulation_induced_base", float("nan"))
                    for metric, _w in case["terms"]:
                        if metric.startswith(("locomotion_induced", "manipulation_induced")):
                            vals = [abs(r[metric]) for r in pb.values()
                                    if np.isfinite(r[metric])]
                            scale.setdefault(metric, float(np.median(vals)) if vals else float("nan"))
                score, terms = _score(case, pb, rivals, scale)
                gates = _gates(case, pb, rivals, motion.get(seed), scale,
                               NONTRIVIAL[experiment], terms)
                candidates.append(
                    {
                        "experiment": experiment, "case": case["name"],
                        "condition": per_baseline["C1"]["condition"],
                        "trajectory_family": family or "",
                        "seed": seed, "selection_score": score,
                        "eligible": int(not gates),
                        "gate_failures": "|".join(gates),
                        "terms": terms, "per_baseline": pb,
                        "motion": motion.get(seed, {}),
                    }
                )

        eligible = [c for c in candidates if c["eligible"] and np.isfinite(c["selection_score"])]
        # Deterministic order: score desc, then seed asc, then family -- no ties
        # can be broken by dictionary order.
        eligible.sort(key=lambda c: (-c["selection_score"], c["seed"], c["trajectory_family"]))
        for rank, c in enumerate(sorted(candidates,
                                        key=lambda c: (-(c["selection_score"]
                                                         if np.isfinite(c["selection_score"])
                                                         else -np.inf),
                                                       c["seed"], c["trajectory_family"])), 1):
            ranking_rows.append(_ranking_row(c, rank, rivals, len(candidates)))

        # One seed is never reused for two criteria, so the three cases are
        # three distinct trials.
        pick = next((c for c in eligible if c["seed"] not in used_seeds), None)
        if pick is None:
            unsatisfied.append(
                {
                    "experiment": experiment, "case": case["name"],
                    "intent": case["intent"],
                    "n_candidates": len(candidates),
                    "n_eligible": len(eligible),
                    "reason": (
                        "no candidate passed the gates"
                        if not eligible
                        else "every eligible candidate was already selected for an "
                             "earlier criterion"
                    ),
                    "most_common_gate_failures": _common_failures(candidates),
                }
            )
            print(f"  {experiment}/{case['name']}: UNSATISFIED -- "
                  f"{len(eligible)}/{len(candidates)} eligible")
            continue
        used_seeds.add(pick["seed"])
        rank = 1 + sum(
            1 for c in eligible if (-c["selection_score"], c["seed"]) <
            (-pick["selection_score"], pick["seed"])
        )
        selected.append(_selected_record(pick, case, rivals, rank, len(eligible),
                                         len(candidates), candidates))
        print(f"  {experiment}/{case['name']}: seed {pick['seed']} "
              f"{pick['condition']}{'/' + pick['trajectory_family'] if pick['trajectory_family'] else ''}"
              f"  score={pick['selection_score']:.4f} "
              f"(rank {rank}/{len(eligible)} eligible of {len(candidates)} candidates)")

    common.write_csv(os.path.join(showcase_dir, f"{experiment}_candidate_ranking.csv"),
                     ranking_rows)
    common.write_csv(os.path.join(out_dir, "selected_3_cases.csv"),
                     [_flat(r) for r in selected])
    manifest = {
        "experiment": experiment,
        "label": "SELECTED ILLUSTRATIVE TRIALS -- not population averages",
        "candidate_pool": {
            "seeds": [common.official_seeds()[0], common.official_seeds()[-1]],
            "n_seeds": len(common.official_seeds()),
            "note": (
                "The candidate pool is the official 50-seed population. Selecting a "
                "seed here does not remove it from, or alter, any aggregate: every "
                "summary uses all 50 seeds. Each selected case reports its rank and "
                "percentile within the 50 so its position in the distribution is "
                "explicit."
            ),
            "condition_note": spec["pool_note"],
        },
        "nontriviality_gate": NONTRIVIAL[experiment],
        "criteria": [
            {k: v for k, v in case.items() if k != "terms"}
            | {"terms": [{"metric": m, "weight": w} for m, w in case["terms"]]}
            for case in spec["cases"]
        ],
        "selected": selected,
        "unsatisfied_criteria": unsatisfied,
        "n_candidate_rows": len(ranking_rows),
    }
    common.write_json(os.path.join(showcase_dir, f"{experiment}_selection_manifest.json"),
                      manifest)
    return manifest


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _common_failures(candidates) -> dict:
    counts: dict = {}
    for c in candidates:
        for f in c["gate_failures"].split("|"):
            if f:
                counts[f.split("(")[0]] = counts.get(f.split("(")[0], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:6])


def _ranking_row(c, rank, rivals, n_total) -> dict:
    row = {
        "experiment": c["experiment"], "case": c["case"], "rank": rank,
        "n_candidates": n_total, "condition": c["condition"],
        "trajectory_family": c["trajectory_family"], "seed": c["seed"],
        "selection_score": c["selection_score"], "eligible": c["eligible"],
        "gate_failures": c["gate_failures"],
    }
    for metric, t in c["terms"].items():
        row[f"adv_{metric}"] = t["advantage"]
        row[f"adv_{metric}_normalised"] = t["normalised"]
    for name, pb in sorted(c["per_baseline"].items()):
        for metric in c["terms"]:
            row[f"{name}_{metric}"] = pb.get(metric)
        row[f"{name}_survived"] = pb.get("survived")
    for key, value in (c["motion"] or {}).items():
        row[f"cmd_{key}"] = value
    return row


REPORT_METRICS = {
    "E1": ["base_vxy_rmse_mps", "base_yaw_rmse_radps", "ee_position_rmse_m",
           "ee_orientation_mae_deg", "ee_position_mae_m", "ee_position_p95_m",
           "ee_orientation_rmse_deg", "valid_evaluation_time_s"],
    "E2": ["base_vxy_rmse_mps", "base_yaw_rmse_radps", "total_power_w",
           "leg_power_w", "arm_power_w", "wheel_power_w",
           "leg_acc_rms_radps2", "arm_acc_rms_radps2",
           "ee_position_rmse_m", "ee_orientation_mae_deg", "valid_evaluation_time_s"],
    "E3": ["ee_position_rmse_m", "base_vxy_rmse_mps", "base_yaw_rmse_radps",
           "ee_orientation_mae_deg", "locomotion_induced_ee",
           "manipulation_induced_base", "valid_evaluation_time_s"],
}


def _selected_record(pick, case, rivals, rank, n_eligible, n_candidates, candidates) -> dict:
    exp = pick["experiment"]
    metrics = REPORT_METRICS[exp]
    pb = pick["per_baseline"]
    # Percentile of this seed's score among ALL candidates with a finite score.
    scores = [c["selection_score"] for c in candidates
              if np.isfinite(c["selection_score"])]
    pct = (100.0 * sum(1 for s in scores if s <= pick["selection_score"]) / len(scores)
           if scores else float("nan"))
    paired = {}
    for rival in rivals:
        if rival not in pb:
            continue
        paired[f"{rival}_minus_C1"] = {
            m: (float(pb[rival][m]) - float(pb["C1"][m]))
            if _isnum(pb[rival].get(m)) and _isnum(pb["C1"].get(m)) else None
            for m in metrics
        }
    return {
        "selected_seed": pick["seed"],
        "experiment": exp,
        "case": case["name"],
        "condition": pick["condition"],
        "trajectory_family": pick["trajectory_family"],
        "selection_reason": case["intent"],
        "selection_score": pick["selection_score"],
        "rank_among_eligible": rank,
        "n_eligible": n_eligible,
        "n_candidates": n_candidates,
        "score_percentile_among_candidates": pct,
        "label": "selected illustrative trial",
        "C1_metrics": {m: _num(pb["C1"].get(m)) for m in metrics},
        "baseline_metrics": {
            r: {m: _num(pb[r].get(m)) for m in metrics} for r in rivals if r in pb
        },
        "paired_differences_rival_minus_C1": paired,
        "survived": {k: int(v.get("survived", 0)) for k, v in sorted(pb.items())},
        "termination_reason": {k: v.get("termination_reason", "") for k, v in sorted(pb.items())},
        "commanded_motion": pick["motion"],
        "episode_paths": {k: v.get("path", "") for k, v in sorted(pb.items())},
    }


def _num(v):
    return float(v) if _isnum(v) else None


def _flat(rec) -> dict:
    out = {
        "experiment": rec["experiment"], "case": rec["case"],
        "selected_seed": rec["selected_seed"], "condition": rec["condition"],
        "trajectory_family": rec["trajectory_family"],
        "label": rec["label"],
        "selection_reason": rec["selection_reason"],
        "selection_score": rec["selection_score"],
        "rank_among_eligible": rec["rank_among_eligible"],
        "n_eligible": rec["n_eligible"], "n_candidates": rec["n_candidates"],
        "score_percentile_among_candidates": rec["score_percentile_among_candidates"],
    }
    for metric, value in rec["C1_metrics"].items():
        out[f"C1_{metric}"] = value
    for rival, metrics in rec["baseline_metrics"].items():
        for metric, value in metrics.items():
            out[f"{rival}_{metric}"] = value
    for rival, metrics in rec["paired_differences_rival_minus_C1"].items():
        for metric, value in metrics.items():
            out[f"diff_{rival}_{metric}"] = value
    for name, value in rec["survived"].items():
        out[f"survived_{name}"] = value
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", default="E1,E2,E3")
    parser.add_argument("--out", default=common.OUT)
    args = parser.parse_args(argv)
    showcase = os.path.join(args.out, "showcase")
    print("=" * 78)
    print("ROUND 3 ILLUSTRATIVE CASE SELECTION  (selected illustrative trials; "
          "the 50-seed summaries remain the evidence for general claims)")
    print("=" * 78)
    n_unsat = 0
    for experiment in [e.strip() for e in args.experiments.split(",")]:
        manifest = select_experiment(experiment, os.path.join(args.out, experiment),
                                     showcase)
        n_unsat += len(manifest["unsatisfied_criteria"])
    print("=" * 78)
    print(f"unsatisfied criteria: {n_unsat} (reported, not substituted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
