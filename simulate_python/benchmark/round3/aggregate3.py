#!/usr/bin/env python3
"""Phase E: 50-seed aggregate statistics and paired comparisons.

Read exclusively from the round-3 per-seed CSVs written by ``export.py``, so
every published number is reproducible from the per-seed files alone -- which
is also checked, by recomputing the means straight from the CSV.

Statistical rules
-----------------
* The experimental unit is the seed. Timesteps within an episode are never
  treated as independent trials.
* 95% intervals are percentile bootstrap, 10,000 resamples, fixed seed from
  ``baselines.yaml`` -- so a rerun gives byte-identical intervals.
* Paired comparisons difference the SAME seed first, then bootstrap whole
  pairs. Only seeds present for both methods enter; the pair count is reported.
* Sign convention, stated in every output file: ``difference = rival - C1``.
  For a lower-is-better metric a POSITIVE difference means C1 performed better.
  ``c1_minus_rival_mean`` is also emitted for readers who want the literal
  "C1 - C3" ordering.
* ``significant`` means the paired-difference CI excludes zero. A near-zero or
  negative difference is reported as measured, never suppressed.
* Survival is a proportion, so it gets a Wilson interval rather than a
  bootstrap of a 0/1 mean.
* E3 family pooling uses a CLUSTER bootstrap over seeds: the three trajectory
  families at one seed share that seed's initial state, so they are correlated
  and are averaged within the seed before resampling. Treating them as 150
  independent trials would shrink the intervals by about sqrt(3).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from benchmark.core import stats
from benchmark.round3 import common

SIGN_CONVENTION = (
    "difference = rival - C1; positive => C1 better for a lower-is-better metric"
)

E1_METRICS = [
    ("base_vxy_rmse_mps", "m/s", True),
    ("base_yaw_rmse_radps", "rad/s", True),
    ("ee_position_rmse_m", "m", True),
    ("ee_orientation_mae_deg", "deg", True),
    ("valid_evaluation_time_s", "s", False),
    ("survived", "-", False),
]
E2_METRICS = [
    ("base_vxy_rmse_mps", "m/s", True),
    ("base_yaw_rmse_radps", "rad/s", True),
    ("total_power_w", "W", True),
    ("leg_power_w", "W", True),
    ("arm_power_w", "W", True),
    ("wheel_power_w", "W", True),
    ("leg_acc_rms_radps2", "rad/s^2", True),
    ("arm_acc_rms_radps2", "rad/s^2", True),
    ("ee_position_rmse_m", "m", True),
    ("ee_orientation_mae_deg", "deg", True),
    ("valid_evaluation_time_s", "s", False),
    ("survived", "-", False),
]
E3_METRICS = [
    ("ee_position_rmse_m", "m", True),
    ("base_vxy_rmse_mps", "m/s", True),
    ("base_yaw_rmse_radps", "rad/s", True),
    ("ee_orientation_mae_deg", "deg", True),
    ("valid_evaluation_time_s", "s", False),
    ("survived", "-", False),
]

SUMMARY_COLUMNS = [
    "experiment", "condition", "trajectory_family", "alpha_base", "alpha_ee",
    "baseline", "metric", "unit", "n_seeds", "mean", "standard_deviation",
    "ci95_lower", "ci95_upper", "median", "min", "max",
    "n_survived", "survival_wilson_lower", "survival_wilson_upper",
    "seed_min", "seed_max", "bootstrap_resamples", "bootstrap_seed",
]

PAIRED_COLUMNS = [
    "experiment", "condition", "trajectory_family", "alpha_base", "alpha_ee",
    "comparison", "reference", "rival", "metric", "unit", "lower_is_better",
    "n_pairs", "mean_difference", "c1_minus_rival_mean", "median_difference",
    "sd_difference", "ci95_lower", "ci95_upper",
    "median_ci95_lower", "median_ci95_upper",
    "frac_pairs_c1_better", "n_pairs_c1_better", "significant",
    "reference_mean", "rival_mean", "relative_difference_pct",
    "sign_convention", "bootstrap_resamples", "bootstrap_seed",
]


def _rows_to_float(rows, metric):
    return np.asarray([float(r[metric]) for r in rows], dtype=np.float64)


def summarize_cell(rows, metric, unit, cfg, keys) -> dict:
    values = _rows_to_float(rows, metric)
    seeds = [int(r["seed"]) for r in rows]
    finite = values[np.isfinite(values)]
    res = stats.summarize(finite, n_resamples=cfg["bootstrap_resamples"],
                          seed=cfg["bootstrap_seed"], confidence=cfg["confidence"])
    surv = np.asarray([int(r["survived"]) for r in rows])
    wl, wh = stats.wilson_interval(int(surv.sum()), int(surv.size),
                                   confidence=cfg["confidence"])
    out = dict(keys)
    out.update(
        {
            "metric": metric, "unit": unit,
            "n_seeds": int(len(rows)),
            "mean": res["mean"], "standard_deviation": res["std"],
            "ci95_lower": res["ci_low"], "ci95_upper": res["ci_high"],
            "median": res["median"],
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
            "n_survived": int(surv.sum()),
            "survival_wilson_lower": wl, "survival_wilson_upper": wh,
            "seed_min": min(seeds), "seed_max": max(seeds),
            "bootstrap_resamples": cfg["bootstrap_resamples"],
            "bootstrap_seed": cfg["bootstrap_seed"],
        }
    )
    return out


def paired_cell(rows, reference, rival, metric, unit, lower_is_better, cfg, keys) -> dict:
    ref_map = common.per_seed_map(rows, reference, metric)
    riv_map = common.per_seed_map(rows, rival, metric)
    res = stats.paired_advantage(
        ref_map, riv_map, n_resamples=cfg["bootstrap_resamples"],
        seed=cfg["bootstrap_seed"], confidence=cfg["confidence"],
    )
    shared = sorted(set(ref_map) & set(riv_map))
    ref_mean = float(np.mean([ref_map[s] for s in shared])) if shared else float("nan")
    riv_mean = float(np.mean([riv_map[s] for s in shared])) if shared else float("nan")
    out = dict(keys)
    out.update(
        {
            "comparison": f"{rival}_minus_{reference}",
            "reference": reference, "rival": rival,
            "metric": metric, "unit": unit,
            "lower_is_better": int(bool(lower_is_better)),
            "n_pairs": res["n_pairs"],
            "mean_difference": res["mean"],
            "c1_minus_rival_mean": -res["mean"],
            "median_difference": res["median"],
            "sd_difference": res.get("std", float("nan")),
            "ci95_lower": res["ci_low"], "ci95_upper": res["ci_high"],
            "median_ci95_lower": res["median_ci_low"],
            "median_ci95_upper": res["median_ci_high"],
            "frac_pairs_c1_better": res["frac_positive"],
            "n_pairs_c1_better": res["n_positive"],
            "significant": int(res["significant"]),
            "reference_mean": ref_mean, "rival_mean": riv_mean,
            "relative_difference_pct": (
                100.0 * res["mean"] / ref_mean if np.isfinite(ref_mean) and ref_mean != 0
                else float("nan")
            ),
            "sign_convention": SIGN_CONVENTION,
            "bootstrap_resamples": cfg["bootstrap_resamples"],
            "bootstrap_seed": cfg["bootstrap_seed"],
        }
    )
    return out


def _group(rows, keys):
    out: dict = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


# ==========================================================================
def run_experiment(experiment, metrics, condition_order, rivals, out_dir, cfg):
    rows = common.read_csv(os.path.join(out_dir, "per_seed.csv"))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["survived"] = int(r["survived"])
    summary, paired = [], []
    for (condition,), cell in sorted(
        _group(rows, ["condition"]).items(),
        key=lambda kv: condition_order.index(kv[0][0]) if kv[0][0] in condition_order else 99,
    ):
        for (baseline,), per_baseline in sorted(_group(cell, ["baseline"]).items()):
            for metric, unit, lib in metrics:
                summary.append(
                    summarize_cell(per_baseline, metric, unit, cfg,
                                   {"experiment": experiment, "condition": condition,
                                    "baseline": baseline})
                )
        for rival in rivals:
            if not any(r["baseline"] == rival for r in cell):
                continue
            for metric, unit, lib in metrics:
                paired.append(
                    paired_cell(cell, "C1", rival, metric, unit, lib, cfg,
                                {"experiment": experiment, "condition": condition})
                )
    common.write_csv(os.path.join(out_dir, "summary_50seeds.csv"), summary, SUMMARY_COLUMNS)
    common.write_csv(os.path.join(out_dir, "paired_comparisons.csv"), paired, PAIRED_COLUMNS)
    return summary, paired


# ==========================================================================
# E3: per-cell, per-family, and family-pooled (cluster bootstrap over seeds)
# ==========================================================================
def _pool_families_by_seed(rows, metric):
    """seed -> mean over the trajectory families present at that seed.

    The three families at one seed share that seed's initial state, so they are
    a cluster, not three independent trials. Averaging within the seed and then
    bootstrapping seeds is the cluster bootstrap; it keeps the effective sample
    size at the number of seeds.
    """
    acc: dict = {}
    for r in rows:
        acc.setdefault(int(r["seed"]), []).append(float(r[metric]))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def _pooled_summary(rows, metric, unit, cfg, keys) -> dict:
    per_seed = _pool_families_by_seed(rows, metric)
    seeds = sorted(per_seed)
    values = np.asarray([per_seed[s] for s in seeds], dtype=np.float64)
    res = stats.summarize(values, n_resamples=cfg["bootstrap_resamples"],
                          seed=cfg["bootstrap_seed"], confidence=cfg["confidence"])
    surv = np.asarray([int(r["survived"]) for r in rows])
    wl, wh = stats.wilson_interval(int(surv.sum()), int(surv.size), cfg["confidence"])
    out = dict(keys)
    out.update(
        {
            "metric": metric, "unit": unit, "n_seeds": len(seeds),
            "mean": res["mean"], "standard_deviation": res["std"],
            "ci95_lower": res["ci_low"], "ci95_upper": res["ci_high"],
            "median": res["median"],
            "min": float(values.min()) if values.size else float("nan"),
            "max": float(values.max()) if values.size else float("nan"),
            "n_survived": int(surv.sum()),
            "survival_wilson_lower": wl, "survival_wilson_upper": wh,
            "seed_min": min(seeds), "seed_max": max(seeds),
            "bootstrap_resamples": cfg["bootstrap_resamples"],
            "bootstrap_seed": cfg["bootstrap_seed"],
        }
    )
    return out


def _pooled_paired(rows, reference, rival, metric, unit, lib, cfg, keys) -> dict:
    ref = _pool_families_by_seed([r for r in rows if r["baseline"] == reference], metric)
    riv = _pool_families_by_seed([r for r in rows if r["baseline"] == rival], metric)
    res = stats.paired_advantage(ref, riv, n_resamples=cfg["bootstrap_resamples"],
                                 seed=cfg["bootstrap_seed"], confidence=cfg["confidence"])
    shared = sorted(set(ref) & set(riv))
    ref_mean = float(np.mean([ref[s] for s in shared])) if shared else float("nan")
    riv_mean = float(np.mean([riv[s] for s in shared])) if shared else float("nan")
    out = dict(keys)
    out.update(
        {
            "comparison": f"{rival}_minus_{reference}", "reference": reference,
            "rival": rival, "metric": metric, "unit": unit,
            "lower_is_better": int(bool(lib)), "n_pairs": res["n_pairs"],
            "mean_difference": res["mean"], "c1_minus_rival_mean": -res["mean"],
            "median_difference": res["median"], "sd_difference": res.get("std", float("nan")),
            "ci95_lower": res["ci_low"], "ci95_upper": res["ci_high"],
            "median_ci95_lower": res["median_ci_low"],
            "median_ci95_upper": res["median_ci_high"],
            "frac_pairs_c1_better": res["frac_positive"],
            "n_pairs_c1_better": res["n_positive"],
            "significant": int(res["significant"]),
            "reference_mean": ref_mean, "rival_mean": riv_mean,
            "relative_difference_pct": (
                100.0 * res["mean"] / ref_mean if np.isfinite(ref_mean) and ref_mean else float("nan")
            ),
            "sign_convention": SIGN_CONVENTION,
            "bootstrap_resamples": cfg["bootstrap_resamples"],
            "bootstrap_seed": cfg["bootstrap_seed"],
        }
    )
    return out


def run_e3(out_dir, cfg):
    rows = common.read_csv(os.path.join(out_dir, "per_seed.csv"))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["survived"] = int(r["survived"])
        r["alpha_base"] = float(r["alpha_base"])
        r["alpha_ee"] = float(r["alpha_ee"])
    summary, paired = [], []

    # ---- per (cell, family) and per cell pooled over families ----
    for (ab, ae), cell in sorted(_group(rows, ["alpha_base", "alpha_ee"]).items()):
        tag = _cell_tag(ab, ae)
        for (family,), fam in sorted(_group(cell, ["trajectory_family"]).items()):
            keys = {"experiment": "E3", "condition": tag, "trajectory_family": family,
                    "alpha_base": ab, "alpha_ee": ae}
            for (baseline,), per_b in sorted(_group(fam, ["baseline"]).items()):
                for metric, unit, lib in E3_METRICS:
                    summary.append(summarize_cell(per_b, metric, unit, cfg,
                                                  dict(keys, baseline=baseline)))
            for rival in ("C3", "C5"):
                for metric, unit, lib in E3_METRICS:
                    paired.append(paired_cell(fam, "C1", rival, metric, unit, lib, cfg, keys))
        keys = {"experiment": "E3", "condition": tag, "trajectory_family": "all(pooled)",
                "alpha_base": ab, "alpha_ee": ae}
        for (baseline,), per_b in sorted(_group(cell, ["baseline"]).items()):
            for metric, unit, lib in E3_METRICS:
                summary.append(_pooled_summary(per_b, metric, unit, cfg,
                                               dict(keys, baseline=baseline)))
        for rival in ("C3", "C5"):
            for metric, unit, lib in E3_METRICS:
                paired.append(_pooled_paired(cell, "C1", rival, metric, unit, lib, cfg, keys))

    common.write_csv(os.path.join(out_dir, "summary_50seeds.csv"), summary, SUMMARY_COLUMNS)

    # ---- degradation ----
    deg = common.read_csv(os.path.join(out_dir, "per_seed_degradation.csv"))
    for r in deg:
        r["seed"] = int(r["seed"])
        r["survived"] = int(r["condition_survived"])
        r["degradation"] = float(r["degradation"])
        r["alpha_base"] = float(r["alpha_base"])
        r["alpha_ee"] = float(r["alpha_ee"])
    deg_summary, deg_paired = [], []
    for (kind, ab, ae), cell in sorted(
        _group(deg, ["degradation_type", "alpha_base", "alpha_ee"]).items()
    ):
        tag = _cell_tag(ab, ae)
        ref_tag = _cell_tag(0.0, ae) if kind == "locomotion_induced_ee" else _cell_tag(ab, 0.0)
        for (family,), fam in sorted(_group(cell, ["trajectory_family"]).items()):
            keys = {"experiment": "E3", "condition": tag, "trajectory_family": family,
                    "alpha_base": ab, "alpha_ee": ae,
                    "degradation_type": kind, "reference_condition": ref_tag}
            for (baseline,), per_b in sorted(_group(fam, ["baseline"]).items()):
                deg_summary.append(summarize_cell(per_b, "degradation", _deg_unit(kind), cfg,
                                                  dict(keys, baseline=baseline)))
            for rival in ("C3", "C5"):
                deg_paired.append(
                    paired_cell(fam, "C1", rival, "degradation", _deg_unit(kind), True, cfg, keys)
                )
        keys = {"experiment": "E3", "condition": tag, "trajectory_family": "all(pooled)",
                "alpha_base": ab, "alpha_ee": ae,
                "degradation_type": kind, "reference_condition": ref_tag}
        for (baseline,), per_b in sorted(_group(cell, ["baseline"]).items()):
            deg_summary.append(_pooled_summary(per_b, "degradation", _deg_unit(kind), cfg,
                                               dict(keys, baseline=baseline)))
        for rival in ("C3", "C5"):
            deg_paired.append(
                _pooled_paired(cell, "C1", rival, "degradation", _deg_unit(kind), True, cfg, keys)
            )

    common.write_csv(os.path.join(out_dir, "summary_degradation.csv"), deg_summary,
                     SUMMARY_COLUMNS[:1] + ["degradation_type", "reference_condition"]
                     + SUMMARY_COLUMNS[1:])
    common.write_csv(os.path.join(out_dir, "paired_comparisons.csv"),
                     paired + deg_paired,
                     PAIRED_COLUMNS[:1] + ["degradation_type", "reference_condition"]
                     + PAIRED_COLUMNS[1:])
    return summary, paired, deg_summary, deg_paired


def _cell_tag(ab: float, ae: float) -> str:
    levels = {0.0: 0, 0.5: 1, 1.0: 2}
    return f"S{levels[round(ab,6)]}{levels[round(ae,6)]}"


def _deg_unit(kind: str) -> str:
    return "m" if kind == "locomotion_induced_ee" else "m/s"


# ==========================================================================
def verify_means_from_csv(out_dir, experiment, tol=1e-12) -> list[str]:
    """Independently recompute every summary mean straight from per_seed.csv.

    Validation item 10: the published aggregates must be reproducible from the
    per-seed file alone, with no access to this module's internals.
    """
    per_seed = common.read_csv(os.path.join(out_dir, "per_seed.csv"))
    summary = common.read_csv(os.path.join(out_dir, "summary_50seeds.csv"))
    problems = []
    for s in summary:
        if s.get("trajectory_family") == "all(pooled)":
            continue  # pooled means are checked separately below
        sel = [
            r for r in per_seed
            if r["baseline"] == s["baseline"]
            and (s.get("trajectory_family", "") in ("", r.get("trajectory_family", "")))
            and _cell_matches(r, s, experiment)
        ]
        values = [float(r[s["metric"]]) for r in sel if np.isfinite(float(r[s["metric"]]))]
        if len(sel) != int(s["n_seeds"]):
            problems.append(
                f"{experiment} {s['condition']}/{s.get('trajectory_family','')}/"
                f"{s['baseline']}/{s['metric']}: n {len(sel)} != {s['n_seeds']}"
            )
            continue
        got = float(np.mean(values))
        want = float(s["mean"])
        if abs(got - want) > max(tol, tol * abs(want)):
            problems.append(
                f"{experiment} {s['condition']}/{s.get('trajectory_family','')}/"
                f"{s['baseline']}/{s['metric']}: mean {got!r} != {want!r}"
            )
    return problems


def _cell_matches(row, summary_row, experiment) -> bool:
    if experiment != "E3":
        return row["condition"] == summary_row["condition"]
    return (
        abs(float(row["alpha_base"]) - float(summary_row["alpha_base"])) < 1e-12
        and abs(float(row["alpha_ee"]) - float(summary_row["alpha_ee"])) < 1e-12
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", default="E1,E2,E3")
    parser.add_argument("--out", default=common.OUT)
    args = parser.parse_args(argv)
    wanted = [e.strip() for e in args.experiments.split(",")]
    cfg = common.stats_config()

    print("=" * 78)
    print(f"ROUND 3 AGGREGATE STATISTICS  "
          f"({cfg['bootstrap_resamples']} bootstrap resamples, "
          f"fixed seed {cfg['bootstrap_seed']})")
    print(f"sign convention: {SIGN_CONVENTION}")
    print("=" * 78)

    problems = []
    if "E1" in wanted:
        s, p = run_experiment("E1", E1_METRICS, ["T1", "T2", "T3", "T4"],
                              ["C3", "C5"], os.path.join(args.out, "E1"), cfg)
        print(f"  E1: {len(s)} summary rows, {len(p)} paired comparisons")
        problems += verify_means_from_csv(os.path.join(args.out, "E1"), "E1")
    if "E2" in wanted:
        s, p = run_experiment("E2", E2_METRICS, ["Low", "Medium", "High"],
                              ["C2"], os.path.join(args.out, "E2"), cfg)
        print(f"  E2: {len(s)} summary rows, {len(p)} paired comparisons")
        problems += verify_means_from_csv(os.path.join(args.out, "E2"), "E2")
    if "E3" in wanted:
        s, p, ds, dp = run_e3(os.path.join(args.out, "E3"), cfg)
        print(f"  E3: {len(s)} summary rows ({len(p)} paired) "
              f"+ {len(ds)} degradation summary rows ({len(dp)} paired)")
        problems += verify_means_from_csv(os.path.join(args.out, "E3"), "E3")

    print("-" * 78)
    if problems:
        print(f"MEAN REPRODUCIBILITY: {len(problems)} MISMATCHES")
        for line in problems[:10]:
            print("  " + line)
        return 2
    print("MEAN REPRODUCIBILITY: every published mean recomputed from per_seed.csv "
          "to within 1e-12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
