#!/usr/bin/env python3
"""E1 paper tables: Table A (official 50 seeds) and Table B (3 selected seeds).

Reads ONLY the existing round-3 E1 outputs and the round-3 raw episodes. No
simulation is run, no controller, policy or metric definition is touched.

Table A and Table B have the IDENTICAL layout -- same 2x2 condition grouping,
same three methods, same four metrics, same precision. They differ in exactly
two things: the seed population they average, and how uncertainty is reported
(Table A: 95% bootstrap CI over 50 seeds; Table B: standard deviation over the
3 selected seeds, which is descriptive only at n = 3).

What Table B is and is not
--------------------------
Table B is an **illustrative** subset. Its three seeds are drawn from the same
official 0-49 pool, so choosing them changes no aggregate: Table A always uses
all 50. The three seeds are the SAME for all four conditions and all three
baselines, so Table B is a coherent paired subset rather than a different
cherry-pick per row. It is never a population estimate, and §9 of the analysis
report quantifies by how much it overstates the typical advantage.

Selection
---------
Ranked on T4 only (the one condition where base and EE are both commanded to
move). Because the metrics have different units, each per-seed advantage is
divided by a declared reference scale before the terms are combined; the scales
are the T4 pooled medians defined in ``REFERENCE_SCALES_DOC`` and printed into
the manifest. Criteria, gates and the tie-break are fixed in code before any
ranking is computed and were not revised after seeing the result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

from benchmark.core import stats
from benchmark.round3 import common

OUT_DIR = os.path.join(common.OUT, "E1", "paper_tables")

CONDITIONS = ["T1", "T2", "T3", "T4"]
CONDITION_TITLES = {
    "T1": "Base still / EE fixed",
    "T2": "Base still / EE random",
    "T3": "Base moving / EE fixed",
    "T4": "Base moving / EE random",
}
# Plain-text units for the rendered preview (the .tex uses the LaTeX forms).
PREVIEW_UNITS = {
    "base_vxy_rmse_mps": "m/s",
    "base_yaw_rmse_radps": "rad/s",
    "ee_position_rmse_m": "m",
    "ee_orientation_mae_deg": "deg",
}

# Compact descriptors for the table's grouping column.
CONDITION_SHORT = {
    "T1": "still / fixed",
    "T2": "still / random",
    "T3": "moving / fixed",
    "T4": "moving / random",
}
BASELINES = ["C1", "C3", "C5"]
METHOD_LABEL = {
    "C1": "C1 Unified RL WBC (Ours)",
    "C3": "C3 RL Locomotion + IK",
    "C5": "C5 Cooperative Dual-PPO",
}
METHOD_SHORT_TEX = {
    "C1": r"C1 (\textbf{Ours})",
    "C3": "C3",
    "C5": "C5",
}

# (csv column, short header, SI unit, decimals)
METRICS = [
    ("base_vxy_rmse_mps", "Base XY", r"m\,s$^{-1}$", 4),
    ("base_yaw_rmse_radps", "Yaw", r"rad\,s$^{-1}$", 4),
    ("ee_position_rmse_m", "EE Pos", "m", 4),
    ("ee_orientation_mae_deg", "EE Ori", r"$^{\circ}$", 2),
]
METRIC_KEYS = [m for m, _h, _u, _d in METRICS]

# --------------------------------------------------------------------------
# The round-3 E1 protocol this module refuses to run against anything else.
# --------------------------------------------------------------------------
EXPECTED_PROTOCOL = {
    "duration_s": 30.0,
    "settle_s": 3.0,
    "base_segment_duration_range_s": [2.0, 3.0],
    "ee_resample_interval_s": 3.0,
    "ee_interpolation": "none",
    "base_frame": "body",
    "ee_frame": "plb",
    "official_seeds": list(range(50)),
}

# --------------------------------------------------------------------------
# Selection, declared before ranking
# --------------------------------------------------------------------------
RANKING_CONDITION = "T4"

REFERENCE_SCALES_DOC = (
    "Each metric's reference scale is the T4 pooled median of that metric over "
    "all three baselines and all 50 official seeds. It is a deterministic "
    "function of the official data, is computed before any ranking, and is "
    "recorded in the manifest. Dividing by it makes the m/s, rad/s, m and "
    "degree terms dimensionless and comparable, so no raw sum of unlike units "
    "is ever formed."
)

CRITERIA = [
    {
        "seed_label": "A",
        "name": "combined_tracking_advantage",
        "intent": (
            "C1 ahead of BOTH C3 and C5 on base XY velocity AND on EE position "
            "in the same T4 trial: the simultaneous locomotion + manipulation "
            "tracking claim."
        ),
        "terms": ["base_vxy_rmse_mps", "ee_position_rmse_m"],
        "combine": "min",
        "require_positive": ["base_vxy_rmse_mps", "ee_position_rmse_m"],
    },
    {
        "seed_label": "B",
        "name": "base_tracking_advantage",
        "intent": (
            "C1 ahead of BOTH rivals on base XY velocity AND base yaw rate "
            "while the EE target keeps changing every 3 s."
        ),
        "terms": ["base_vxy_rmse_mps", "base_yaw_rmse_radps"],
        "combine": "min",
        "require_positive": ["base_vxy_rmse_mps", "base_yaw_rmse_radps"],
    },
    {
        "seed_label": "C",
        "name": "dynamic_ee_tracking_advantage",
        "intent": (
            "C1 ahead of BOTH rivals on EE position RMSE while the base is "
            "moving and the target is replaced every 3 s."
        ),
        "terms": ["ee_position_rmse_m"],
        "combine": "mean",
        "require_positive": ["ee_position_rmse_m"],
    },
]

GATES_DOC = {
    "G1_all_twelve_episodes_survived": (
        "Every one of the seed's 12 episodes (4 conditions x 3 baselines) must "
        "have run the full [3,30) s window. Table B averages the same seeds "
        "across all four conditions, so one early termination anywhere would "
        "make a whole column survivorship-conditioned. 27 of the 50 seeds are "
        "rejected by this gate, almost all of them C5 falling in T3/T4 -- a "
        "gate that removes trials where C1 would have looked strongest."
    ),
    "G2_all_metrics_finite": "All four metrics finite for all three baselines.",
    "G3_nontrivial_T4_commands": (
        "At least 8 base-command segments and at least 8 EE target switches in "
        "the T4 evaluation window, and an RMS commanded base speed of at least "
        "0.70x the T4 pool median. The speed threshold is condition-relative, "
        "not absolute, so it cannot reject a whole condition by construction."
    ),
    "G4_criterion_sign": (
        "Every metric named in the criterion's require_positive list must have "
        "a strictly positive advantage against BOTH rivals."
    ),
    "G5_distinct_seeds": "A seed selected for one criterion is not reused.",
}

NONTRIVIAL = {
    "min_base_segments": 8,
    "min_ee_switches": 8,
    "min_rms_fraction_of_condition_median": 0.70,
}


class ValidationError(SystemExit):
    pass


# ==========================================================================
# 1. Load and validate
# ==========================================================================
def load_and_validate(verbose=True) -> list[dict]:
    """Return the 600 official E1 per-seed records, or raise."""
    path = os.path.join(common.OUT, "E1", "per_seed.csv")
    if not os.path.isfile(path):
        raise ValidationError(f"round-3 E1 per-seed file not found: {path}")
    rows = common.read_csv(path)
    problems: list[str] = []

    # --- shape ---
    want_seeds = EXPECTED_PROTOCOL["official_seeds"]
    if len(rows) != len(CONDITIONS) * len(BASELINES) * len(want_seeds):
        problems.append(
            f"expected {len(CONDITIONS)*len(BASELINES)*len(want_seeds)} rows, "
            f"found {len(rows)}"
        )
    keys = set()
    for r in rows:
        r["seed"] = int(r["seed"])
        r["survived"] = int(r["survived"])
        for m in METRIC_KEYS:
            r[m] = float(r[m])
        key = (r["condition"], r["baseline"], r["seed"])
        if key in keys:
            problems.append(f"duplicate per-seed record: {key}")
        keys.add(key)
    for condition in CONDITIONS:
        for baseline in BASELINES:
            got = sorted(r["seed"] for r in rows
                         if r["condition"] == condition and r["baseline"] == baseline)
            if got != want_seeds:
                missing = sorted(set(want_seeds) - set(got))
                extra = sorted(set(got) - set(want_seeds))
                problems.append(
                    f"{condition}/{baseline}: {len(got)} seeds "
                    f"(missing {missing[:5]}, unexpected {extra[:5]})"
                )
    unexpected_conditions = {r["condition"] for r in rows} - set(CONDITIONS)
    unexpected_baselines = {r["baseline"] for r in rows} - set(BASELINES)
    if unexpected_conditions:
        problems.append(f"unexpected conditions: {sorted(unexpected_conditions)}")
    if unexpected_baselines:
        problems.append(f"unexpected baselines: {sorted(unexpected_baselines)}")

    # --- metric sanity ---
    for r in rows:
        for m in METRIC_KEYS:
            v = r[m]
            if not np.isfinite(v):
                problems.append(f"non-finite {m} at "
                                f"{r['condition']}/{r['baseline']}/seed {r['seed']}")
            elif v < 0.0:
                problems.append(f"negative {m} at "
                                f"{r['condition']}/{r['baseline']}/seed {r['seed']}")
        if r["ee_orientation_mae_deg"] > 180.0:
            problems.append(
                f"EE orientation MAE {r['ee_orientation_mae_deg']:.2f} deg exceeds the "
                f"geodesic maximum at {r['condition']}/{r['baseline']}/seed {r['seed']}"
            )
        if r["survived"] and abs(float(r["valid_evaluation_time_s"]) - 27.0) > 1e-9:
            problems.append(
                f"survived episode with a {r['valid_evaluation_time_s']} s window at "
                f"{r['condition']}/{r['baseline']}/seed {r['seed']}"
            )

    # --- config: the round-3 protocol, not round 2 ---
    cfg_path = os.path.join(common.BENCH, "configs", "E1.yaml")
    with open(cfg_path, encoding="utf-8") as h:
        e1_cfg = yaml.safe_load(h)
    g = common.load_global_config()
    checks = [
        ("E1.yaml base.segment_duration_range",
         list(e1_cfg["base"]["segment_duration_range"]),
         EXPECTED_PROTOCOL["base_segment_duration_range_s"]),
        ("E1.yaml ee.resample_interval_s",
         float(e1_cfg["ee"]["resample_interval_s"]),
         EXPECTED_PROTOCOL["ee_resample_interval_s"]),
        ("E1.yaml ee.interpolation", e1_cfg["ee"]["interpolation"],
         EXPECTED_PROTOCOL["ee_interpolation"]),
        ("E1.yaml duration_s", float(e1_cfg["duration_s"]),
         EXPECTED_PROTOCOL["duration_s"]),
        ("baselines.yaml settle_s", float(g["settle_s"]),
         EXPECTED_PROTOCOL["settle_s"]),
        ("baselines.yaml metrics.base_frame", g["metrics"]["base_frame"],
         EXPECTED_PROTOCOL["base_frame"]),
        ("baselines.yaml metrics.ee_frame", g["metrics"]["ee_frame"],
         EXPECTED_PROTOCOL["ee_frame"]),
        ("baselines.yaml official_seeds", common.official_seeds(),
         EXPECTED_PROTOCOL["official_seeds"]),
    ]
    for name, got, want in checks:
        if got != want:
            problems.append(f"CONFIG MISMATCH {name}: {got!r} != round-3 {want!r}")

    # --- per-episode metadata: every one of the 600 must record round 3 ---
    stale = 0
    for r in rows:
        meta_path = os.path.join(
            common.BENCH, r["path"].rstrip("/"), "metadata.json"
        ) if r.get("path") else None
        if not meta_path or not os.path.isfile(meta_path):
            problems.append(f"missing metadata.json for {r['condition']}/"
                            f"{r['baseline']}/seed {r['seed']}")
            continue
        with open(meta_path, encoding="utf-8") as h:
            meta = json.load(h)
        scen = meta.get("scenario", {})
        if list(scen.get("base_segment_duration_range_s", [])) != [2.0, 3.0] or \
           float(scen.get("ee_resample_interval_s", -1)) != 3.0 or \
           scen.get("ee_interpolation") != "none" or \
           float(meta.get("duration_s", -1)) != 30.0 or \
           float(meta.get("settle_s", -1)) != 3.0:
            stale += 1
            if stale <= 5:
                problems.append(
                    f"ROUND-2 CONTAMINATION: {r['condition']}/{r['baseline']}/"
                    f"seed {r['seed']} records base "
                    f"{scen.get('base_segment_duration_range_s')}, EE "
                    f"{scen.get('ee_resample_interval_s')} s, "
                    f"{meta.get('duration_s')} s / {meta.get('settle_s')} s settle"
                )
    if stale > 5:
        problems.append(f"... and {stale - 5} further contaminated episodes")

    if problems:
        print("E1 ROUND-3 VALIDATION FAILED:", file=sys.stderr)
        for line in problems[:25]:
            print("  " + line, file=sys.stderr)
        raise ValidationError(
            f"{len(problems)} validation problem(s); refusing to build tables "
            f"from data that is not the final round-3 E1 setup. No substitution "
            f"of older results was made."
        )
    if verbose:
        print(f"  validated {len(rows)} official E1 records: "
              f"{len(CONDITIONS)} conditions x {len(BASELINES)} baselines x "
              f"{len(want_seeds)} seeds (0-{want_seeds[-1]}), no duplicates")
        print("  protocol confirmed: base resampling 2-3 s, EE targets every "
              "3.0 s with no interpolation, PLB EE frame, body base frame, "
              "30 s episodes, 3 s settling, window [3,30) s")
        print("  all 600 episode metadata records carry the round-3 protocol "
              "(no round-2 contamination)")
    return rows


def _index(rows):
    return {(r["condition"], r["baseline"], r["seed"]): r for r in rows}


# ==========================================================================
# 2. Statistics
# ==========================================================================
def official_means(rows, cfg) -> list[dict]:
    """Per (condition, baseline, metric): mean / SD / 95% bootstrap CI over 50 seeds."""
    out = []
    for condition in CONDITIONS:
        for baseline in BASELINES:
            sel = sorted((r for r in rows
                          if r["condition"] == condition and r["baseline"] == baseline),
                         key=lambda r: r["seed"])
            seeds = [r["seed"] for r in sel]
            for metric, _h, unit, _d in METRICS:
                values = np.asarray([r[metric] for r in sel], dtype=np.float64)
                lo, hi = stats.bootstrap_ci(
                    values, n_resamples=cfg["bootstrap_resamples"],
                    seed=cfg["bootstrap_seed"], confidence=cfg["confidence"])
                out.append({
                    "table": "A_official_50seeds",
                    "condition": condition,
                    "condition_title": CONDITION_TITLES[condition],
                    "baseline": baseline, "method": METHOD_LABEL[baseline],
                    "metric": metric, "unit": unit.replace("\\,", " ").replace("$", ""),
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "ci95_lower": lo, "ci95_upper": hi,
                    "median": float(np.median(values)),
                    "min": float(values.min()), "max": float(values.max()),
                    "n_survived": int(sum(r["survived"] for r in sel)),
                    "seed_min": min(seeds), "seed_max": max(seeds),
                    "bootstrap_resamples": cfg["bootstrap_resamples"],
                    "bootstrap_seed": cfg["bootstrap_seed"],
                })
    return out


def selected_means(rows, selected_seeds) -> list[dict]:
    """Per (condition, baseline, metric): mean / SD over the 3 selected seeds."""
    out = []
    for condition in CONDITIONS:
        for baseline in BASELINES:
            sel = [r for r in rows
                   if r["condition"] == condition and r["baseline"] == baseline
                   and r["seed"] in selected_seeds]
            sel.sort(key=lambda r: selected_seeds.index(r["seed"]))
            if len(sel) != len(selected_seeds):
                raise ValidationError(
                    f"{condition}/{baseline}: {len(sel)} of {len(selected_seeds)} "
                    f"selected seeds present"
                )
            for metric, _h, unit, _d in METRICS:
                values = np.asarray([r[metric] for r in sel], dtype=np.float64)
                row = {
                    "table": "B_selected_3seeds",
                    "condition": condition,
                    "condition_title": CONDITION_TITLES[condition],
                    "baseline": baseline, "method": METHOD_LABEL[baseline],
                    "metric": metric, "unit": unit.replace("\\,", " ").replace("$", ""),
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "sd_descriptive_n3": float(values.std(ddof=1)),
                    "selected_seeds": ";".join(str(s) for s in selected_seeds),
                }
                for i, (s, v) in enumerate(zip(selected_seeds, values)):
                    row[f"seed{i+1}_id"] = s
                    row[f"seed{i+1}_value"] = float(v)
                out.append(row)
    return out


def recompute_check(rows, official, selected, selected_seeds) -> list[str]:
    """Independently re-derive every published mean from the 600-row records."""
    idx = {}
    for r in rows:
        idx.setdefault((r["condition"], r["baseline"]), {})[r["seed"]] = r
    problems = []
    for entry in official:
        per_seed = idx[(entry["condition"], entry["baseline"])]
        want = float(np.mean([per_seed[s][entry["metric"]] for s in range(50)]))
        if abs(want - entry["mean"]) > 1e-12:
            problems.append(f"A {entry['condition']}/{entry['baseline']}/"
                            f"{entry['metric']}: {entry['mean']!r} != {want!r}")
    for entry in selected:
        per_seed = idx[(entry["condition"], entry["baseline"])]
        want = float(np.mean([per_seed[s][entry["metric"]] for s in selected_seeds]))
        if abs(want - entry["mean"]) > 1e-12:
            problems.append(f"B {entry['condition']}/{entry['baseline']}/"
                            f"{entry['metric']}: {entry['mean']!r} != {want!r}")
    return problems


def paired_advantages(rows, selected_seeds, cfg) -> list[dict]:
    """(rival - C1) per condition and metric: 50-seed inferential, 3-seed descriptive."""
    out = []
    for condition in CONDITIONS:
        c1 = {r["seed"]: r for r in rows
              if r["condition"] == condition and r["baseline"] == "C1"}
        for rival in ("C3", "C5"):
            rv = {r["seed"]: r for r in rows
                  if r["condition"] == condition and r["baseline"] == rival}
            for metric, _h, unit, _d in METRICS:
                ref = {s: c1[s][metric] for s in c1}
                oth = {s: rv[s][metric] for s in rv}
                res = stats.paired_advantage(
                    ref, oth, n_resamples=cfg["bootstrap_resamples"],
                    seed=cfg["bootstrap_seed"], confidence=cfg["confidence"])
                diffs3 = [oth[s] - ref[s] for s in selected_seeds]
                out.append({
                    "condition": condition,
                    "condition_title": CONDITION_TITLES[condition],
                    "comparison": f"{rival}_minus_C1",
                    "metric": metric,
                    "unit": unit.replace("\\,", " ").replace("$", ""),
                    "sign_convention": "positive => C1 has the lower error",
                    # --- official 50 seeds: inferential ---
                    "n_pairs_50seed": res["n_pairs"],
                    "mean_difference_50seed": res["mean"],
                    "median_difference_50seed": res["median"],
                    "sd_difference_50seed": res.get("std", float("nan")),
                    "ci95_lower_50seed": res["ci_low"],
                    "ci95_upper_50seed": res["ci_high"],
                    "frac_seeds_C1_better_50seed": res["frac_positive"],
                    "significant_50seed": int(res["significant"]),
                    "C1_mean_50seed": float(np.mean(list(ref.values()))),
                    f"{rival}_mean_50seed": float(np.mean(list(oth.values()))),
                    # --- 3 selected seeds: DESCRIPTIVE ONLY ---
                    "selected_seeds": ";".join(str(s) for s in selected_seeds),
                    "mean_difference_3seed_descriptive": float(np.mean(diffs3)),
                    "sd_difference_3seed_descriptive": float(np.std(diffs3, ddof=1)),
                    "n_of_3_seeds_C1_better": int(sum(1 for d in diffs3 if d > 0)),
                    "descriptive_only_note": (
                        "n = 3: reported as a description of the selected trials. "
                        "No confirmatory test, no CI, and not a population estimate."
                    ),
                })
    return out


# ==========================================================================
# 3. Selection
# ==========================================================================
def _commanded_motion(path: str) -> dict:
    cmd = common.read_csv(os.path.join(common.BENCH, path, "commands.csv"))
    keep = [i for i, r in enumerate(cmd) if int(r["evaluated"]) == 1]
    vx = np.asarray([float(cmd[i]["vx_cmd"]) for i in keep])
    vy = np.asarray([float(cmd[i]["vy_cmd"]) for i in keep])
    wz = np.asarray([float(cmd[i]["wz_cmd"]) for i in keep])
    ee = np.asarray([[float(cmd[i][f"ee_cmd_kp{j}"]) for j in range(9)] for i in keep])
    base = np.stack([vx, vy, wz], axis=1)
    n_base = int(np.sum(np.any(np.abs(np.diff(base, axis=0)) > 1e-12, axis=1))) + 1
    n_ee = int(np.sum(np.any(np.abs(np.diff(ee, axis=0)) > 1e-9, axis=1)))
    return {
        "rms_commanded_base_speed": float(np.sqrt(np.mean(vx**2 + vy**2))),
        "rms_commanded_yaw_rate": float(np.sqrt(np.mean(wz**2))),
        "n_base_segments": n_base, "n_ee_switches": n_ee,
    }


def select_three_seeds(rows, verbose=True):
    idx = _index(rows)
    t4 = [r for r in rows if r["condition"] == RANKING_CONDITION]

    # --- reference scales, computed before any ranking ---
    scales = {m: float(np.median([r[m] for r in t4])) for m in METRIC_KEYS}

    # --- per-seed commanded motion (read from C1's own commands.csv) ---
    motion = {}
    for seed in range(50):
        motion[seed] = _commanded_motion(idx[(RANKING_CONDITION, "C1", seed)]["path"])
    speed_median = float(np.median([m["rms_commanded_base_speed"]
                                    for m in motion.values()]))
    scales["rms_commanded_base_speed_T4_median"] = speed_median

    # --- gates that do not depend on the criterion ---
    base_gates = {}
    for seed in range(50):
        reasons = []
        episodes = [idx[(c, b, seed)] for c in CONDITIONS for b in BASELINES]
        dead = [(e["condition"], e["baseline"], e["termination_reason"])
                for e in episodes if not e["survived"]]
        if dead:
            reasons.append("G1_terminated:" + ",".join(f"{c}/{b}({w})" for c, b, w in dead))
        for e in episodes:
            for m in METRIC_KEYS:
                if not np.isfinite(e[m]):
                    reasons.append(f"G2_nonfinite_{e['condition']}/{e['baseline']}_{m}")
        mo = motion[seed]
        if mo["n_base_segments"] < NONTRIVIAL["min_base_segments"]:
            reasons.append(f"G3_base_segments({mo['n_base_segments']}"
                           f"<{NONTRIVIAL['min_base_segments']})")
        if mo["n_ee_switches"] < NONTRIVIAL["min_ee_switches"]:
            reasons.append(f"G3_ee_switches({mo['n_ee_switches']}"
                           f"<{NONTRIVIAL['min_ee_switches']})")
        floor = NONTRIVIAL["min_rms_fraction_of_condition_median"] * speed_median
        if mo["rms_commanded_base_speed"] < floor:
            reasons.append(f"G3_commanded_speed({mo['rms_commanded_base_speed']:.4f}"
                           f"<{floor:.4f})")
        base_gates[seed] = reasons

    # --- rank each criterion ---
    ranking_rows, selected = [], []
    used: set = set()
    unsatisfied = []
    for crit in CRITERIA:
        candidates = []
        for seed in range(50):
            per_baseline = {b: idx[(RANKING_CONDITION, b, seed)] for b in BASELINES}
            terms = {}
            for metric in crit["terms"]:
                adv = min(per_baseline[r][metric] - per_baseline["C1"][metric]
                          for r in ("C3", "C5"))
                terms[metric] = {"advantage": adv,
                                 "normalised": adv / scales[metric]}
            values = [t["normalised"] for t in terms.values()]
            score = (float(min(values)) if crit["combine"] == "min"
                     else float(np.mean(values)))
            reasons = list(base_gates[seed])
            for metric in crit["require_positive"]:
                if not terms[metric]["advantage"] > 0.0:
                    reasons.append(
                        f"G4_require_positive_{metric}"
                        f"({terms[metric]['advantage']:+.6g})")
            candidates.append({
                "seed": seed, "score": score, "terms": terms,
                "gate_failures": reasons, "eligible": not reasons,
                "per_baseline": per_baseline, "motion": motion[seed],
            })
        order = sorted(candidates, key=lambda c: (-c["score"], c["seed"]))
        for rank, c in enumerate(order, 1):
            row = {
                "criterion": crit["name"], "seed_label": crit["seed_label"],
                "rank": rank, "n_candidates": len(order),
                "condition": RANKING_CONDITION, "seed": c["seed"],
                "selection_score": c["score"], "eligible": int(c["eligible"]),
                "gate_failures": "|".join(c["gate_failures"]),
            }
            for metric, t in c["terms"].items():
                row[f"advantage_{metric}"] = t["advantage"]
                row[f"advantage_{metric}_normalised"] = t["normalised"]
                row[f"reference_scale_{metric}"] = scales[metric]
            for b in BASELINES:
                for metric in METRIC_KEYS:
                    row[f"{b}_{metric}"] = c["per_baseline"][b][metric]
                row[f"{b}_survived_T4"] = c["per_baseline"][b]["survived"]
            row["cmd_n_base_segments"] = c["motion"]["n_base_segments"]
            row["cmd_n_ee_switches"] = c["motion"]["n_ee_switches"]
            row["cmd_rms_base_speed"] = c["motion"]["rms_commanded_base_speed"]
            ranking_rows.append(row)

        eligible = [c for c in order if c["eligible"]]
        pick = next((c for c in eligible if c["seed"] not in used), None)
        if pick is None:
            unsatisfied.append({
                "criterion": crit["name"], "seed_label": crit["seed_label"],
                "intent": crit["intent"], "n_candidates": len(order),
                "n_eligible": len(eligible),
                "reason": ("no candidate passed the gates" if not eligible else
                           "every eligible candidate was already selected"),
            })
            if verbose:
                print(f"  criterion {crit['seed_label']} ({crit['name']}): "
                      f"UNSATISFIED -- {len(eligible)}/{len(order)} eligible")
            continue
        used.add(pick["seed"])
        rank_among_eligible = 1 + sum(
            1 for c in eligible
            if (-c["score"], c["seed"]) < (-pick["score"], pick["seed"]))
        selected.append({
            "seed_label": crit["seed_label"], "criterion": crit["name"],
            "intent": crit["intent"], "seed": pick["seed"],
            "ranking_condition": RANKING_CONDITION,
            "selection_score": pick["score"],
            "rank_among_eligible": rank_among_eligible,
            "n_eligible": len(eligible), "n_candidates": len(order),
            "terms": {m: {"advantage": t["advantage"],
                          "reference_scale": scales[m],
                          "normalised": t["normalised"]}
                      for m, t in pick["terms"].items()},
            "T4_metrics": {b: {m: pick["per_baseline"][b][m] for m in METRIC_KEYS}
                           for b in BASELINES},
            "commanded_motion_T4": pick["motion"],
        })
        if verbose:
            print(f"  criterion {crit['seed_label']} ({crit['name']}): "
                  f"seed {pick['seed']}  score {pick['score']:.4f}  "
                  f"rank {rank_among_eligible}/{len(eligible)} eligible of "
                  f"{len(order)}")

    n_gated = sum(1 for s in range(50) if base_gates[s])
    manifest = {
        "label": "SELECTED ILLUSTRATIVE TRIALS -- not a population estimate",
        "ranking_condition": RANKING_CONDITION,
        "ranking_condition_title": CONDITION_TITLES[RANKING_CONDITION],
        "candidate_pool": {
            "seeds": [0, 49], "n": 50,
            "note": ("The candidate pool is the official 50-seed population. "
                     "Choosing a seed here changes no aggregate: Table A always "
                     "uses all 50 seeds. Each selected seed reports its rank "
                     "among the eligible candidates."),
        },
        "reference_scales": scales,
        "reference_scales_doc": REFERENCE_SCALES_DOC,
        "gates": GATES_DOC,
        "nontriviality_thresholds": NONTRIVIAL,
        "n_seeds_rejected_by_base_gates": n_gated,
        "criteria": CRITERIA,
        "selected": selected,
        "unsatisfied_criteria": unsatisfied,
        "selected_seed_ids": [s["seed"] for s in selected],
        "same_seeds_for_all_conditions_and_baselines": True,
    }
    return [s["seed"] for s in selected], manifest, ranking_rows


# ==========================================================================
# 4. LaTeX
# ==========================================================================
def _fmt(value, digits, bold=False):
    if value is None or not np.isfinite(float(value)):
        return "--"
    text = f"{float(value):.{digits}f}"
    return r"\textbf{" + text + "}" if bold else text


def _cell(mean, lo, hi, digits, bold, style):
    """One table cell: the mean, then its uncertainty.

    ``style`` selects what follows the mean:

    * ``"sd50"`` -- Table A. ``$mean \pm SD$`` over the 50 official seeds, the
      whole cell in inline math so the operator is typeset as an operator. The
      SD is the sample standard deviation (``ddof=1``) of the 50 per-seed
      values; it is NOT derived from the bootstrap CI. Bold uses
      ``\mathbf`` because the mean sits inside math mode here.
    * ``"ci"``   -- the 95% bootstrap CI in brackets, kept available for any
      table that wants the inferential interval instead of the dispersion.
    * ``"sd"``   -- Table B, standard deviation over the 3 selected seeds,
      parenthesised and flagged as descriptive at n = 3.
    """
    if style == "sd50":
        if mean is None or not np.isfinite(float(mean)):
            return "--"
        text = f"{float(mean):.{digits}f}"
        head = r"\mathbf{" + text + "}" if bold else text
        tail = rf" \pm {lo:.{digits}f}" if np.isfinite(lo) else ""
        return "$" + head + tail + "$"
    head = _fmt(mean, digits, bold)
    if style == "ci":
        tail = (rf"\,{{\scriptsize$[{lo:.{digits}f},{hi:.{digits}f}]$}}"
                if np.isfinite(lo) else "")
    else:
        tail = rf"\,{{\scriptsize$(\pm{lo:.{digits}f})$}}" if np.isfinite(lo) else ""
    return head + tail


def build_table(entries, selected_seeds, style, title, label, caption,
                n_seeds) -> str:
    idx = {(e["condition"], e["baseline"], e["metric"]): e for e in entries}
    lines = [
        r"% Requires: booktabs, multirow. No vertical rules by design.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        r"\begin{tabular}{@{}ll" + "r" * len(METRICS) + r"@{}}",
        r"\toprule",
        r"& & \multicolumn{" + str(len(METRICS))
        + r"}{c}{Tracking error (lower is better)} \\",
        r"\cmidrule(l){3-" + str(2 + len(METRICS)) + r"}",
        r"Condition & Method & " + " & ".join(h for _m, h, _u, _d in METRICS) + r" \\",
        r" & & " + " & ".join(f"[{u}]" for _m, _h, u, _d in METRICS) + r" \\",
        r"\midrule",
    ]
    for ci, condition in enumerate(CONDITIONS):
        # One line, no \\ inside the cell: a line break inside \shortstack is
        # legal LaTeX but it makes the row ambiguous to parse and is fragile in
        # a \multirow. The short descriptor fits on one line instead.
        head = (r"\multirow{3}{*}{" + condition
                + r"\,{\scriptsize " + CONDITION_SHORT[condition] + r"}}")
        best = {}
        for metric, _h, _u, _d in METRICS:
            means = {b: idx[(condition, b, metric)]["mean"] for b in BASELINES}
            best[metric] = min(means, key=means.get)
        for bi, baseline in enumerate(BASELINES):
            cells = []
            for metric, _h, _u, digits in METRICS:
                e = idx[(condition, baseline, metric)]
                if style == "sd50":
                    lo, hi = e["sd"], float("nan")
                elif style == "ci":
                    lo, hi = e["ci95_lower"], e["ci95_upper"]
                else:
                    lo, hi = e["sd_descriptive_n3"], float("nan")
                cells.append(_cell(e["mean"], lo, hi, digits,
                                   baseline == best[metric], style))
            prefix = head if bi == 0 else ""
            lines.append(prefix + " & " + METHOD_SHORT_TEX[baseline] + " & "
                         + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace" if ci < len(CONDITIONS) - 1 else "")
    lines = [ln for ln in lines if ln != ""]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


# Caption text fixed by the paper brief; reproduced verbatim apart from the
# LaTeX spelling of "+-" and "%". The protocol details that the earlier
# mean-[CI] caption carried (command resampling, evaluation window, frames,
# per-seed CSV) now live only in ROUND3_REPORT.md and E1_TABLE_ANALYSIS.md.
TABLE_A_CAPTION = (
    r"Base-velocity and EE-pose tracking errors (mean $\pm$ standard "
    r"deviation) over 50 random seeds. Bold values indicate the lowest mean "
    r"error."
)

TABLE_B_CAPTION = (
    r"\textbf{Selected Illustrative Tracking Cases (3 Paired Seeds).} "
    r"Values are the mean of three selected paired evaluation seeds chosen "
    r"using T4 simultaneous tracking criteria. The full 50-seed evaluation is "
    r"reported separately. "
    r"The same three seeds are used for every condition and every method, so "
    r"the subset is paired rather than re-selected per row; the bracketed "
    r"value is the standard deviation over those three seeds and is "
    r"descriptive only at $n=3$. \textbf{This table is illustrative and is "
    r"not a population estimate}: it carries no inferential claim and does not "
    r"supersede the 50-seed results. \textbf{Bold} marks the lowest measured "
    r"mean per metric and condition, whichever method attains it. Underlying "
    r"individual seed values: \texttt{E1\_selected3\_raw\_36rows.csv}."
)


# ==========================================================================
# 5. Preview rendering (not a LaTeX compile)
# ==========================================================================
def render_preview(entries, style, title, subtitle, out_path) -> str | None:
    """Render a visual preview of the table with matplotlib.

    This is NOT a LaTeX compile. No LaTeX distribution is installed in this
    environment, so the .tex files cannot be typeset here; this preview exists
    so the layout and the numbers can be eyeballed, and it says so on its face.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = {(e["condition"], e["baseline"], e["metric"]): e for e in entries}
    headers = ["Condition", "Method"] + [
        f"{h}\n[{PREVIEW_UNITS[m]}]" for m, h, _u, _d in METRICS
    ]
    body, bolds = [], []
    for condition in CONDITIONS:
        best = {}
        for metric, _h, _u, _d in METRICS:
            means = {b: idx[(condition, b, metric)]["mean"] for b in BASELINES}
            best[metric] = min(means, key=means.get)
        for bi, baseline in enumerate(BASELINES):
            row = [f"{condition}  {CONDITION_TITLES[condition]}" if bi == 0 else "",
                   METHOD_LABEL[baseline]]
            flags = [False, False]
            for metric, _h, _u, digits in METRICS:
                e = idx[(condition, baseline, metric)]
                if style == "sd50":
                    tail = f" ± {e['sd']:.{digits}f}"
                elif style == "ci":
                    tail = f"\n[{e['ci95_lower']:.{digits}f}, {e['ci95_upper']:.{digits}f}]"
                else:
                    tail = f"\n(±{e['sd_descriptive_n3']:.{digits}f})"
                row.append(f"{e['mean']:.{digits}f}{tail}")
                flags.append(baseline == best[metric])
            body.append(row)
            bolds.append(flags)

    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=headers, cellLoc="center", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 2.05)
    ncols = len(headers)
    for c in range(ncols):
        cell = table[0, c]
        cell.set_text_props(weight="bold", color="#0b0b0b")
        cell.set_facecolor("#f0efec")
        cell.set_edgecolor("#b9b8b1")
    for r, flags in enumerate(bolds, start=1):
        for c in range(ncols):
            cell = table[r, c]
            cell.set_edgecolor("#dededa")
            if flags[c]:
                cell.set_text_props(weight="bold")
            if (r - 1) // 3 % 2 == 1:
                cell.set_facecolor("#fbfbfa")
            if c == 1 and body[r - 1][1].startswith("C1"):
                cell.set_text_props(weight="bold")
    for c in range(ncols):
        table[0, c].set_height(table[0, c].get_height() * 1.25)
    fig.suptitle(title, fontsize=11, y=0.97)
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=8, color="#52514e")
    fig.text(0.5, 0.03,
             "RENDERED PREVIEW — this is NOT a LaTeX compile. No LaTeX "
             "distribution is installed in this environment; the .tex file is "
             "the publication artefact.",
             ha="center", fontsize=7.5, color="#8a8982", style="italic")
    fig.tight_layout(rect=(0, 0.055, 1, 0.915))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 6. Analysis report
# ==========================================================================
def _md_table(entries, style, selected_seeds=None) -> list[str]:
    idx = {(e["condition"], e["baseline"], e["metric"]): e for e in entries}
    head = ("| Condition | Method | Base XY [m/s] | Yaw [rad/s] | "
            "EE Pos [m] | EE Ori [deg] |")
    lines = [head, "|---|---|---|---|---|---|"]
    for condition in CONDITIONS:
        best = {}
        for metric, _h, _u, _d in METRICS:
            means = {b: idx[(condition, b, metric)]["mean"] for b in BASELINES}
            best[metric] = min(means, key=means.get)
        for bi, baseline in enumerate(BASELINES):
            cells = []
            for metric, _h, _u, digits in METRICS:
                e = idx[(condition, baseline, metric)]
                if style == "ci":
                    # Published form is mean +- SD; the bootstrap CI stays on
                    # record here because every significance claim below rests
                    # on it, and SD and CI are not interchangeable.
                    txt = (f"{e['mean']:.{digits}f} ± {e['sd']:.{digits}f} "
                           f"[CI {e['ci95_lower']:.{digits}f}, "
                           f"{e['ci95_upper']:.{digits}f}]")
                else:
                    txt = f"{e['mean']:.{digits}f} (±{e['sd_descriptive_n3']:.{digits}f})"
                cells.append(f"**{txt}**" if baseline == best[metric] else txt)
            label = (f"**{condition}** {CONDITION_TITLES[condition]}" if bi == 0 else "")
            name = ("**C1 Unified RL WBC (Ours)**" if baseline == "C1"
                    else METHOD_LABEL[baseline])
            lines.append(f"| {label} | {name} | " + " | ".join(cells) + " |")
    return lines


def write_analysis(out_dir, rows, official, selected, advantages, manifest,
                   selected_seeds) -> str:
    idx_a = {(e["condition"], e["baseline"], e["metric"]): e for e in official}
    idx_b = {(e["condition"], e["baseline"], e["metric"]): e for e in selected}
    adv = {(a["condition"], a["comparison"], a["metric"]): a for a in advantages}
    metric_name = {m: h for m, h, _u, _d in METRICS}
    digits = {m: d for m, _h, _u, d in METRICS}

    L: list[str] = []
    L += [
        "# E1 Round-3 tracking tables: Table A (50 seeds) vs Table B (3 selected seeds)",
        "",
        "Generated by `benchmark/round3/paper_tables_e1.py` from the existing "
        "round-3 E1 results. **No simulation was run and no controller, policy "
        "or metric definition was touched.**",
        "",
        "Both tables use the same four lower-is-better metrics over the full "
        "`[3, 30)` s evaluation window, including every transient after a base "
        "command switch (every 2–3 s) and after an EE target replacement "
        "(every 3 s, direct sample-and-hold, no interpolation). Base velocity "
        "is in the body frame, the EE pose in the PLB frame. Per-episode RMSE "
        "values are averaged across seeds arithmetically; no pooled RMSE is "
        "recomputed by concatenating timesteps.",
        "",
        "---",
        "",
        "## 1. Table A — official 50-seed mean",
        "",
        "Mean ± sample standard deviation (ddof = 1) over all 50 official "
        "seeds (0–49) — the form published in the paper table. The 95% "
        "bootstrap CI is kept here in square brackets because every "
        "significance statement below rests on it, and an SD is not a "
        "substitute for one (10,000 resamples, fixed seed "
        f"{manifest.get('bootstrap_seed', 20261106)}). **Bold** = lowest "
        "measured mean for that metric within the condition, whichever method "
        "attains it.",
        "",
    ]
    L += _md_table(official, "ci")
    L += [
        "",
        "## 2. Table B — selected 3-seed mean",
        "",
        f"Seeds **{selected_seeds[0]}, {selected_seeds[1]}, {selected_seeds[2]}**, "
        "the same three for every condition and every method. Bracketed value "
        "is the standard deviation over those three seeds and is **descriptive "
        "only at n = 3**. This table is illustrative; it is not a population "
        "estimate and does not supersede Table A.",
        "",
    ]
    L += _md_table(selected, "sd", selected_seeds)

    # ---- 3. seed IDs ----
    L += ["", "## 3. Selected seed IDs", ""]
    L += ["| Label | Seed | Criterion | Rank among eligible |", "|---|---|---|---|"]
    for entry in manifest["selected"]:
        L.append(f"| {entry['seed_label']} | **{entry['seed']}** | "
                 f"`{entry['criterion']}` | {entry['rank_among_eligible']} of "
                 f"{entry['n_eligible']} eligible (pool 50) |")

    # ---- 4. criteria and scores ----
    L += [
        "",
        "## 4. Selection criteria and ranking scores",
        "",
        f"Ranked on **{RANKING_CONDITION}** ({CONDITION_TITLES[RANKING_CONDITION]}) "
        "only — the sole condition in which both subsystems are commanded to "
        "move, so it is the only one where a simultaneous-tracking claim can be "
        "ranked. The chosen seeds are then applied unchanged to T1–T3.",
        "",
        "**Normalisation.** " + REFERENCE_SCALES_DOC,
        "",
        "| Metric | Reference scale (T4 pooled median) |",
        "|---|---|",
    ]
    for metric, _h, _u, d in METRICS:
        L.append(f"| {metric_name[metric]} | "
                 f"{manifest['reference_scales'][metric]:.{d}f} |")
    L += [
        "",
        "For a term, the per-seed advantage is `min over rivals (rival − C1)`, "
        "so a term scores above zero only if C1 beat **both** C3 and C5 on it. "
        "`combine = min` scores the criterion as its smallest normalised term, "
        "so a 'simultaneous' criterion cannot be carried by one large term and "
        "one bad one.",
        "",
        "### Gates (applied before ranking)",
        "",
    ]
    for key, text in GATES_DOC.items():
        L.append(f"* **{key}** — {text}")
    L += [
        "",
        f"{manifest['n_seeds_rejected_by_base_gates']} of the 50 seeds were "
        "rejected by the criterion-independent gates, almost all of them by G1 "
        "(a C5 early termination somewhere in T1–T4).",
        "",
        "### Scores of the selected seeds",
        "",
        "| Label | Seed | Term | Advantage (raw) | Reference scale | Normalised | Criterion score |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in manifest["selected"]:
        first = True
        for metric, t in entry["terms"].items():
            d = digits[metric]
            L.append(
                f"| {entry['seed_label'] if first else ''} | "
                f"{entry['seed'] if first else ''} | {metric_name[metric]} | "
                f"{t['advantage']:+.{d}f} | {t['reference_scale']:.{d}f} | "
                f"{t['normalised']:+.4f} | "
                f"{entry['selection_score']:.4f}{'' if first else ''} |"
            )
            first = False
    if manifest["unsatisfied_criteria"]:
        L += ["", "### Criteria that could not be satisfied", ""]
        for u in manifest["unsatisfied_criteria"]:
            L.append(f"* **{u['seed_label']} / `{u['criterion']}`** — "
                     f"{u['reason']} ({u['n_eligible']} of {u['n_candidates']} "
                     f"eligible). No substitute seed was invented.")
    else:
        L += ["", "All three criteria were satisfied; no threshold was relaxed "
              "and no criterion was revised after seeing the result.", ""]

    # ---- 5. paired comparisons ----
    L += [
        "",
        "## 5. C1 vs C3 and C1 vs C5 (official 50 seeds)",
        "",
        "Paired difference `rival − C1`, formed per seed before averaging. "
        "**Positive ⇒ C1 has the lower error.** `n.s.` = the 95% CI contains "
        "zero. 'C1 better' is the fraction of the 50 paired seeds on which C1 "
        "had the lower value.",
        "",
        "| Cond. | Comparison | Metric | Mean diff. | 95% CI | C1 better | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition in CONDITIONS:
        for rival in ("C3", "C5"):
            for metric, _h, _u, d in METRICS:
                a = adv[(condition, f"{rival}_minus_C1", metric)]
                m = a["mean_difference_50seed"]
                sig = int(a["significant_50seed"])
                verdict = ("**C1**" if sig and m > 0 else
                           f"**{rival}**" if sig and m < 0 else "n.s.")
                L.append(
                    f"| {condition} | {rival} − C1 | {metric_name[metric]} | "
                    f"{m:+.{d}f} | [{a['ci95_lower_50seed']:+.{d}f}, "
                    f"{a['ci95_upper_50seed']:+.{d}f}] | "
                    f"{100*a['frac_seeds_C1_better_50seed']:.0f}% | {verdict} |"
                )

    # ---- 6 / 7. which metrics favour whom ----
    favours_c1, favours_rival, ns = [], [], []
    for key, a in adv.items():
        condition, comparison, metric = key
        m = a["mean_difference_50seed"]
        entry = (condition, comparison.replace("_minus_C1", ""), metric, m,
                 a["ci95_lower_50seed"], a["ci95_upper_50seed"])
        if not int(a["significant_50seed"]):
            ns.append(entry)
        elif m > 0:
            favours_c1.append(entry)
        else:
            favours_rival.append(entry)
    L += [
        "",
        "## 6. Metrics that favour C1",
        "",
        f"**{len(favours_c1)} of the 32 paired comparisons are significant in "
        "C1's favour.**",
        "",
        "* **Base XY velocity RMSE and base yaw-rate RMSE: C1 wins all 16 of "
        "these comparisons** (4 conditions × 2 rivals × 2 metrics), every one "
        "significant. Largest margins: T3 C5−C1 = "
        f"{adv[('T3','C5_minus_C1','base_vxy_rmse_mps')]['mean_difference_50seed']:+.4f} m/s, "
        "T2 C5−C1 = "
        f"{adv[('T2','C5_minus_C1','base_yaw_rmse_radps')]['mean_difference_50seed']:+.4f} rad/s.",
        "* **EE position RMSE whenever the target moves** (T2, T4): C1 ahead of "
        "C3 by "
        f"{adv[('T4','C3_minus_C1','ee_position_rmse_m')]['mean_difference_50seed']:+.4f} m "
        "at T4 and of C5 by "
        f"{adv[('T4','C5_minus_C1','ee_position_rmse_m')]['mean_difference_50seed']:+.4f} m, "
        "with C1 lower on "
        f"{100*adv[('T4','C3_minus_C1','ee_position_rmse_m')]['frac_seeds_C1_better_50seed']:.0f}% "
        "and "
        f"{100*adv[('T4','C5_minus_C1','ee_position_rmse_m')]['frac_seeds_C1_better_50seed']:.0f}% "
        "of the 50 paired seeds respectively.",
        "* **EE orientation MAE with a moving target** (T2, T4): C1 ahead of "
        "both rivals, significantly.",
        "",
        "## 7. Metrics that favour a baseline",
        "",
        "These are preserved exactly as measured and are **bold in both "
        "tables** wherever the baseline attains the lowest mean.",
        "",
    ]
    for condition, rival, metric, m, lo, hi in sorted(favours_rival):
        d = digits[metric]
        L.append(f"* **{condition}, {metric_name[metric]}: {rival} beats C1** "
                 f"by {abs(m):.{d}f} (CI [{lo:+.{d}f}, {hi:+.{d}f}]).")
    L += ["", "Comparisons with no significant difference:", ""]
    for condition, rival, metric, m, lo, hi in sorted(ns):
        d = digits[metric]
        L.append(f"* {condition}, {metric_name[metric]}, {rival} − C1 = "
                 f"{m:+.{d}f} (CI [{lo:+.{d}f}, {hi:+.{d}f}]) — **statistically "
                 f"tied**.")
    L += [
        "",
        "The dominant finding here is the **static-target orientation "
        "trade-off**: with a fixed EE target, C3's stateful IK holds "
        "orientation far better than the unified policy — by "
        f"{abs(adv[('T1','C3_minus_C1','ee_orientation_mae_deg')]['mean_difference_50seed']):.2f}° "
        "at T1 and "
        f"{abs(adv[('T3','C3_minus_C1','ee_orientation_mae_deg')]['mean_difference_50seed']):.2f}° "
        "at T3. C1's orientation advantage appears only once the target starts "
        "moving. C1 and C3 are also statistically tied on EE **position** with "
        "a fixed target (T1 and T3).",
        "",
    ]

    # ---- 8 / 9. how B differs from A ----
    L += [
        "## 8. How the 3-seed subset differs from the 50-seed means",
        "",
        "Per-cell difference `Table B mean − Table A mean`, as a percentage of "
        "the Table A mean. A positive value means the selected subset reports a "
        "**larger error** than the population.",
        "",
        "| Cond. | Metric | C1 | C3 | C5 |",
        "|---|---|---|---|---|",
    ]
    for condition in CONDITIONS:
        for metric, _h, _u, d in METRICS:
            cells = []
            for baseline in BASELINES:
                a_mean = idx_a[(condition, baseline, metric)]["mean"]
                b_mean = idx_b[(condition, baseline, metric)]["mean"]
                pct = 100.0 * (b_mean - a_mean) / a_mean if a_mean else float("nan")
                cells.append(f"{b_mean:.{d}f} ({pct:+.0f}%)")
            L.append(f"| {condition} | {metric_name[metric]} | "
                     + " | ".join(cells) + " |")

    ratios = []
    for key, a in adv.items():
        condition, comparison, metric = key
        m50 = a["mean_difference_50seed"]
        m3 = a["mean_difference_3seed_descriptive"]
        ratios.append((condition, comparison.replace("_minus_C1", ""), metric,
                       m50, m3, (m3 / m50) if m50 else float("nan")))
    overstated = [r for r in ratios if np.isfinite(r[5]) and r[5] > 1.5]
    understated = [r for r in ratios if np.isfinite(r[5]) and 0 <= r[5] < 0.5]
    flipped = [r for r in ratios if np.isfinite(r[5]) and r[5] < 0]
    L += [
        "",
        "## 9. Does the selected subset overstate the typical advantage?",
        "",
        "**Yes, in places — and it understates in others, and flips sign in "
        "one.** The comparison below is the paired advantage `rival − C1` "
        "measured on the 3 selected seeds against the same quantity measured on "
        "all 50. The 3-seed column is *descriptive*: n = 3, no CI, no test.",
        "",
        "| Cond. | Comparison | Metric | 50-seed | 3-seed | Ratio |",
        "|---|---|---|---|---|---|",
    ]
    for condition in CONDITIONS:
        for rival in ("C3", "C5"):
            for metric, _h, _u, d in METRICS:
                a = adv[(condition, f"{rival}_minus_C1", metric)]
                m50 = a["mean_difference_50seed"]
                m3 = a["mean_difference_3seed_descriptive"]
                ratio = (m3 / m50) if m50 else float("nan")
                flag = (" **overstates**" if ratio > 1.5 else
                        " *understates*" if 0 <= ratio < 0.5 else
                        " **SIGN FLIP**" if ratio < 0 else "")
                L.append(f"| {condition} | {rival} − C1 | {metric_name[metric]} | "
                         f"{m50:+.{d}f} | {m3:+.{d}f} | {ratio:.2f}×{flag} |")
    L += [
        "",
        f"**Summary: of the 32 paired quantities, {len(overstated)} are "
        f"overstated by more than 1.5×, {len(understated)} are understated by "
        f"more than 2×, and {len(flipped)} changes sign relative to the "
        "population.**",
        "",
        "The specific cases that matter for how the subset should be read:",
        "",
    ]
    for condition, rival, metric, m50, m3, ratio in sorted(flipped):
        d = digits[metric]
        a = adv[(condition, f"{rival}_minus_C1", metric)]
        L.append(
            f"* **Sign flip — {condition}, {metric_name[metric]}, {rival} − C1.** "
            f"The 50-seed population gives {m50:+.{d}f} "
            f"(CI [{a['ci95_lower_50seed']:+.{d}f}, {a['ci95_upper_50seed']:+.{d}f}], "
            f"{'significant' if int(a['significant_50seed']) else 'not significant'}), "
            f"but the three selected seeds give {m3:+.{d}f}. "
            f"The subset therefore points the opposite way from the population "
            f"on this metric and must not be read as evidence about it."
        )
    worst = sorted(overstated, key=lambda r: -r[5])[:4]
    for condition, rival, metric, m50, m3, ratio in worst:
        d = digits[metric]
        L.append(f"* **Overstated — {condition}, {metric_name[metric]}, "
                 f"{rival} − C1:** {m3:+.{d}f} on the subset against "
                 f"{m50:+.{d}f} on the population ({ratio:.1f}×).")
    L += [
        "",
        "This is exactly what a criterion that ranks seeds by C1's T4 advantage "
        "is expected to do, and it is why **Table A, not Table B, is the "
        "evidence for every general claim**. Table B's role is to show what the "
        "behaviour looks like in a single coherent trio of trials, with the "
        "contrary cells (C3's orientation advantage at T1 and T3) still visible "
        "in it.",
        "",
        "## 10. Proposed IEEE RA-L results paragraph",
        "",
        "> Across 50 official seeds with identical seeded command sequences, the "
        "unified RL whole-body controller (C1) attains the lowest base-velocity "
        "tracking error in all four E1 conditions: base XY velocity RMSE of "
        f"{idx_a[('T4','C1','base_vxy_rmse_mps')]['mean']:.3f} m/s and yaw-rate "
        f"RMSE of {idx_a[('T4','C1','base_yaw_rmse_radps')]['mean']:.3f} rad/s "
        "under simultaneous base motion and 3 s end-effector target "
        "replacement (T4), against "
        f"{idx_a[('T4','C3','base_vxy_rmse_mps')]['mean']:.3f} m/s for RL "
        f"locomotion with stateful IK (C3) and "
        f"{idx_a[('T4','C5','base_vxy_rmse_mps')]['mean']:.3f} m/s for the "
        "cooperative dual-PPO baseline (C5); all sixteen base-tracking paired "
        "differences are significant at 95% by a paired bootstrap. When the "
        "end-effector target is dynamic, C1 also leads on end-effector "
        "position, by "
        f"{adv[('T4','C3_minus_C1','ee_position_rmse_m')]['mean_difference_50seed']:.3f} m "
        f"over C3 (95% CI "
        f"[{adv[('T4','C3_minus_C1','ee_position_rmse_m')]['ci95_lower_50seed']:.3f}, "
        f"{adv[('T4','C3_minus_C1','ee_position_rmse_m')]['ci95_upper_50seed']:.3f}], "
        "lower on "
        f"{100*adv[('T4','C3_minus_C1','ee_position_rmse_m')]['frac_seeds_C1_better_50seed']:.0f}% "
        "of the 50 paired seeds), and on orientation. The advantage is "
        "specific to dynamic targets: with a fixed target, C3's stateful IK "
        "holds end-effector orientation markedly better "
        f"({idx_a[('T1','C3','ee_orientation_mae_deg')]['mean']:.1f}° versus "
        f"{idx_a[('T1','C1','ee_orientation_mae_deg')]['mean']:.1f}° at T1), and "
        "the two are statistically indistinguishable on end-effector position. "
        "C5's errors under base motion must be read alongside its survival "
        f"rate, which falls to "
        f"{idx_a[('T4','C5','base_vxy_rmse_mps')]['n_survived']}/50 "
        "in T4, so its reported errors there are conditioned on the episodes "
        "that completed. Errors are computed over the full post-settling "
        "window and include every transient following a command switch.",
        "",
        "---",
        "",
        "## Generated files",
        "",
        "All under `benchmark/results/final_round3/E1/paper_tables/`:",
        "",
        "| File | Contents |",
        "|---|---|",
        "| `E1_per_seed_600rows.csv` | every official E1 episode: 4 conditions × 3 baselines × 50 seeds |",
        "| `E1_50seed_mean.csv` | Table A values: mean, SD, 95% CI, median, min, max per cell |",
        "| `E1_50seed_mean.tex` | Table A, IEEE `table*`, booktabs, no vertical rules |",
        "| `E1_50seed_mean_preview.pdf` | rendered preview of Table A (**not** a LaTeX compile) |",
        "| `E1_selected3_mean.csv` | Table B values: mean, descriptive SD, and the three per-seed values |",
        "| `E1_selected3_mean.tex` | Table B, identical layout to Table A |",
        "| `E1_selected3_mean_preview.pdf` | rendered preview of Table B (**not** a LaTeX compile) |",
        "| `E1_selected3_raw_36rows.csv` | the 36 underlying records: 3 seeds × 4 conditions × 3 baselines |",
        "| `E1_selected3_seed_ids.json` | seed IDs, criteria, reference scales, gates, scores, ranks |",
        "| `E1_selected3_candidate_ranking.csv` | the complete ranking of all 50 candidates per criterion, selected or not, with gate verdicts |",
        "| `E1_paired_advantages_50seeds.csv` | C3−C1 and C5−C1 per condition and metric: 50-seed mean/CI/win fraction plus the 3-seed descriptive difference |",
        "| `E1_TABLE_ANALYSIS.md` | this report |",
        "",
        "**LaTeX status: the two `.tex` files were NOT compiled.** No LaTeX "
        "distribution is installed in this environment. They pass a structural "
        "validator (environment nesting, brace and `$` balance, per-row cell "
        "counts against the column specification, unescaped underscores), but "
        "that is not a substitute for a compile. The `_preview.pdf` files are "
        "matplotlib renderings provided so the layout and numbers can be "
        "checked; they are not typeset output.",
        "",
    ]
    return common.write_text(os.path.join(out_dir, "E1_TABLE_ANALYSIS.md"),
                             "\n".join(L) + "\n")


def quality_assurance(rows, official, selected, selected_seeds, raw36,
                      advantages) -> list[str]:
    """The §8 checklist, run against the built artefacts."""
    problems = []
    if len(rows) != 600:
        problems.append(f"official records: {len(rows)} != 600")
    for condition in CONDITIONS:
        for baseline in BASELINES:
            seeds = {r["seed"] for r in rows
                     if r["condition"] == condition and r["baseline"] == baseline}
            if len(seeds) != 50:
                problems.append(f"{condition}/{baseline}: {len(seeds)} unique seeds")
    if len(selected_seeds) != 3 or len(set(selected_seeds)) != 3:
        problems.append(f"selected seeds not 3 distinct IDs: {selected_seeds}")
    if any(s not in range(50) for s in selected_seeds):
        problems.append(f"selected seed outside the official pool: {selected_seeds}")
    if len(raw36) != 36:
        problems.append(f"selected raw export: {len(raw36)} != 36 rows")
    combos = {(r["condition"], r["baseline"]) for r in raw36}
    if len(combos) != 12:
        problems.append(f"selected raw export covers {len(combos)} != 12 "
                        f"(condition, baseline) combinations")
    for condition in CONDITIONS:
        for baseline in BASELINES:
            got = sorted(r["seed"] for r in raw36
                         if r["condition"] == condition and r["baseline"] == baseline)
            if got != sorted(selected_seeds):
                problems.append(f"{condition}/{baseline} uses seeds {got}, not "
                                f"{sorted(selected_seeds)}")
    for e in official:
        if e["n_seeds"] != 50:
            problems.append(f"Table A cell with n = {e['n_seeds']}")
        if not (np.isfinite(e["ci95_lower"]) and np.isfinite(e["ci95_upper"])):
            problems.append(f"non-finite CI in {e['condition']}/{e['baseline']}")
        if not (e["ci95_lower"] <= e["mean"] <= e["ci95_upper"]):
            problems.append(f"mean outside its CI in {e['condition']}/{e['baseline']}"
                            f"/{e['metric']}")
    for e in selected:
        if e["n_seeds"] != 3:
            problems.append(f"Table B cell with n = {e['n_seeds']}")
        vals = [e[f"seed{i}_value"] for i in (1, 2, 3)]
        if abs(float(np.mean(vals)) - e["mean"]) > 1e-12:
            problems.append(f"Table B mean != mean of its three stored values in "
                            f"{e['condition']}/{e['baseline']}/{e['metric']}")
    for a in advantages:
        if a["sign_convention"] != "positive => C1 has the lower error":
            problems.append("paired-advantage sign convention not recorded")
            break
    return problems


# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args(argv)
    cfg = common.stats_config()
    os.makedirs(args.out, exist_ok=True)
    written = []

    print("=" * 78)
    print("E1 PAPER TABLES — Table A (50 official seeds) + Table B (3 selected)")
    print("=" * 78)
    print("\n[1] validating the round-3 E1 dataset")
    rows = load_and_validate()

    print("\n[2] selecting three illustrative seeds (ranked on T4)")
    selected_seeds, manifest, ranking_rows = select_three_seeds(rows)
    if len(selected_seeds) != 3:
        raise ValidationError(
            f"only {len(selected_seeds)} of 3 criteria could be satisfied; "
            f"see the manifest's unsatisfied_criteria. No substitute seed was "
            f"invented."
        )
    print(f"  selected seed IDs (same for all 4 conditions and all 3 methods): "
          f"{selected_seeds}")

    print("\n[3] computing statistics")
    official = official_means(rows, cfg)
    selected = selected_means(rows, selected_seeds)
    problems = recompute_check(rows, official, selected, selected_seeds)
    if problems:
        for line in problems[:10]:
            print("  MISMATCH " + line, file=sys.stderr)
        raise ValidationError(f"{len(problems)} published means disagree with the "
                              f"600-row per-seed records")
    print(f"  all {len(official)} Table-A and {len(selected)} Table-B means "
          f"re-derived from the 600-row records to within 1e-12")
    advantages = paired_advantages(rows, selected_seeds, cfg)

    print("\n[4] writing data exports")
    per_seed_cols = ["condition", "condition_title", "baseline", "method", "seed"] \
        + METRIC_KEYS + ["survived", "termination_reason", "valid_evaluation_time_s",
                         "n_eval_samples", "n_base_switches_in_eval",
                         "n_ee_switches_in_eval", "path"]
    export_rows = []
    for r in sorted(rows, key=lambda r: (CONDITIONS.index(r["condition"]),
                                         BASELINES.index(r["baseline"]), r["seed"])):
        out = {k: r.get(k) for k in per_seed_cols}
        out["condition_title"] = CONDITION_TITLES[r["condition"]]
        out["method"] = METHOD_LABEL[r["baseline"]]
        export_rows.append(out)
    written.append(common.write_csv(
        os.path.join(args.out, "E1_per_seed_600rows.csv"), export_rows, per_seed_cols))

    written.append(common.write_csv(
        os.path.join(args.out, "E1_50seed_mean.csv"), official))
    written.append(common.write_csv(
        os.path.join(args.out, "E1_selected3_mean.csv"), selected))

    raw36 = []
    for seed in selected_seeds:
        for condition in CONDITIONS:
            for baseline in BASELINES:
                r = _index(rows)[(condition, baseline, seed)]
                raw36.append({
                    "selected_seed_label": next(
                        s["seed_label"] for s in manifest["selected"]
                        if s["seed"] == seed),
                    "seed": seed, "condition": condition,
                    "condition_title": CONDITION_TITLES[condition],
                    "baseline": baseline, "method": METHOD_LABEL[baseline],
                    **{m: r[m] for m in METRIC_KEYS},
                    "survived": r["survived"],
                    "termination_reason": r["termination_reason"],
                    "valid_evaluation_time_s": r["valid_evaluation_time_s"],
                })
    if len(raw36) != 36:
        raise ValidationError(f"selected raw export has {len(raw36)} rows, expected 36")
    written.append(common.write_csv(
        os.path.join(args.out, "E1_selected3_raw_36rows.csv"), raw36))
    written.append(common.write_csv(
        os.path.join(args.out, "E1_paired_advantages_50seeds.csv"), advantages))
    written.append(common.write_csv(
        os.path.join(args.out, "E1_selected3_candidate_ranking.csv"), ranking_rows))
    written.append(common.write_json(
        os.path.join(args.out, "E1_selected3_seed_ids.json"), manifest))

    print("\n[5] writing LaTeX")
    written.append(common.write_text(
        os.path.join(args.out, "E1_50seed_mean.tex"),
        build_table(official, None, "sd50",
                    "E1 Tracking Performance (50 Seeds)",
                    "tab:e1-50seed-mean", TABLE_A_CAPTION, 50)))
    written.append(common.write_text(
        os.path.join(args.out, "E1_selected3_mean.tex"),
        build_table(selected, selected_seeds, "sd",
                    "Selected Illustrative Tracking Cases (3 Paired Seeds)",
                    "tab:e1-selected3-mean", TABLE_B_CAPTION, 3)))

    print("\n[6] rendering previews (not a LaTeX compile)")
    written.append(render_preview(
        official, "sd50", "Table A — E1 Tracking Performance (50 Seeds)",
        "Mean ± sample SD (ddof = 1) over official seeds 0–49. "
        "Bold = lowest measured mean per metric and condition.",
        os.path.join(args.out, "E1_50seed_mean_preview.pdf")))
    written.append(render_preview(
        selected, "sd",
        "Table B — Selected Illustrative Tracking Cases (3 Paired Seeds)",
        f"Mean over selected seeds {selected_seeds}; (±SD, descriptive at n=3). "
        "Illustrative subset — not a population estimate.",
        os.path.join(args.out, "E1_selected3_mean_preview.pdf")))
    print("\n[7] quality assurance")
    qa = quality_assurance(rows, official, selected, selected_seeds, raw36, advantages)
    if qa:
        for line in qa[:15]:
            print("  QA FAIL " + line, file=sys.stderr)
        raise ValidationError(f"{len(qa)} quality-assurance failure(s)")
    print("  600 official records, 50 unique seeds per cell, 3 distinct selected "
          "seeds, 36 selected records")
    print("  same 3 seeds used for all 4 conditions and all 3 baselines")
    print("  every Table-A mean lies inside its own 95% CI; every Table-B mean "
          "equals the mean of its 3 stored per-seed values")

    print("\n[8] analysis report")
    written.append(write_analysis(args.out, rows, official, selected, advantages,
                                  manifest, selected_seeds))

    print("\n[9] files written")
    for path in written:
        print(f"  {os.path.relpath(path, common.BENCH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
