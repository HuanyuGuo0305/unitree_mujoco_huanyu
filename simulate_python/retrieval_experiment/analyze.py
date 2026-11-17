"""Analysis of an E5 run directory (reproducible from saved logs; no simulation).

    cd unitree_mujoco/simulate_python
    python3 -m retrieval_experiment.analyze retrieval_experiment/results/<run>

Writes <run>/analysis/:
    summary_table.csv / summary_table.tex   paper table: size interval, task success %, one-shot success %,
                                            completion time mean +- SD (successful trials)
    group_metrics.csv                       counts, Wilson 95 % CIs, completion-time n, grasp / drop / retention
    failure_breakdown.csv                   counts per failure reason and interval
    stage_decomposition.csv                 approach / grasp / post-grasp failures
    density_mass_bins.csv                   success vs sampled density and mass (quartile bins, pooled)
    paired_tests.json                       Cochran's Q, McNemar (Holm), paired bootstrap, logistic model
    oneshot_audit.json                      task-success vs one-shot equivalence audit
    consistency.json                        internal + MuJoCo read-back + trajectory recomputation checks
    ANALYSIS.md                             human-readable report
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from retrieval_experiment import retention as RT  # noqa: E402
from retrieval_experiment.metrics import FAILURE_REASONS, mean_std, wilson_ci  # noqa: E402

HL_DT = 0.1


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _b(x):
    return str(x).strip().lower() in ("true", "1")


def load_rows(run_dir):
    with open(os.path.join(run_dir, "episodes.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["scenario_id"] = int(r["scenario_id"])
        r["g"] = int(r["group"])
        r["side"] = _f(r["cube_side_m"])
        r["rho"] = _f(r["cube_density"])
        r["mass"] = _f(r["cube_mass_kg"])
        r["success_b"] = _b(r["success"])
        r["grasp_b"] = _b(r["grasp_confirmed"])
        r["s1_b"] = _b(r["reached_stage1"])
        r["lost_b"] = r["failure_reason"] == "grasp_lost_after_confirmation"
    return rows


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def cochran_q(mat: np.ndarray):
    from scipy.stats import chi2
    n, k = mat.shape
    col, row, tot = mat.sum(0), mat.sum(1), mat.sum()
    den = k * tot - (row ** 2).sum()
    if den == 0:
        return float("nan"), float("nan")
    q = (k - 1) * (k * (col ** 2).sum() - tot ** 2) / den
    return float(q), float(chi2.sf(q, k - 1))


def mcnemar_exact(a: np.ndarray, b: np.ndarray):
    from scipy.stats import binomtest
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(binomtest(n01, n01 + n10, 0.5).pvalue)


def paired_bootstrap(x, y, n_boot=10000, seed=0):
    """95 % percentile CI of mean(y - x) over matched pairs."""
    d = np.asarray(y, float) - np.asarray(x, float)
    if d.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def logistic(X, y, names, iters=50):
    """Logistic regression (Newton / IRLS) with Wald SEs; X without intercept, standardized."""
    from scipy.stats import norm
    mu, sd = X.mean(0), X.std(0)
    Z = np.column_stack([np.ones(len(y)), (X - mu) / sd])
    b = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ b))
        W = p * (1 - p)
        H = Z.T @ (Z * W[:, None]) + 1e-9 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, Z.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-Z @ b))
    cov = np.linalg.inv(Z.T @ (Z * (p * (1 - p))[:, None]))
    se = np.sqrt(np.diag(cov))
    out = {"outcome": "failure (1 - task success)", "n": int(len(y)), "standardized_predictors": True}
    for i, nm in enumerate(["intercept"] + names):
        z = b[i] / se[i]
        out[nm] = {"coef_per_SD": float(b[i]), "se": float(se[i]), "z": float(z), "p": float(2 * norm.sf(abs(z))),
                   **({"predictor_sd": float(sd[i - 1])} if i else {})}
    return out


def recompute_from_trajectory(npz_path, succ_thr=0.15, succ_n=3):
    z = np.load(npz_path)
    stage, rerr = z["stage"], z["retrieval_error"]
    count, first = 0, None
    for i, (s, e) in enumerate(zip(stage, rerr)):
        count = count + 1 if (s == 2 and e < succ_thr) else 0
        if count >= succ_n:
            first = i + 1
            break
    g = np.nonzero(np.diff(np.concatenate([[0], (stage == 2).astype(int)])) == 1)[0]
    grasp = int(g[0]) + 1 if g.size else None
    return first, grasp, len(stage), float(z["cube_side"]), float(z["cube_density"]), float(z["cube_mass"])


def fmt_ct(m, s, n):
    if n == 0:
        return "--"
    return f"{m:.2f} $\\pm$ {s:.2f}" if n > 1 else f"{m:.2f}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python3 -m retrieval_experiment.analyze <run_dir>")
    run_dir = os.path.abspath(argv[0])
    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))["experiment_config"]
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    rows = load_rows(run_dir)
    intervals = [tuple(iv) for iv in cfg["size_intervals_cm"]]
    groups = list(range(1, len(intervals) + 1))
    label = {g: f"[{intervals[g - 1][0]:.1f}, {intervals[g - 1][1]:.1f}]" for g in groups}
    by = defaultdict(list)
    for r in rows:
        by[r["g"]].append(r)
    noise = sorted({r["observation_noise"] for r in rows})
    if len(noise) != 1:
        raise RuntimeError(f"mixed observation-noise settings in one run: {noise}")
    rho_lo, rho_hi = cfg["cube"]["density_range"]

    def wcsv(name, recs):
        with open(os.path.join(out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)

    # ------------------------------------------------------------------ physical retention (250-Hz log)
    ret_cfg = dict(RT.DEFAULTS, **cfg.get("retention", {}))
    traj_dir = os.path.join(run_dir, "trajectories")
    PH = {}
    ret_rows = []
    for r in rows:
        z = np.load(os.path.join(traj_dir, f"s{r['scenario_id']:04d}_g{r['g']}.npz"))
        if "ph_q" not in z.files:
            raise RuntimeError("this run has no 250-Hz retention log (ph_*): One-Shot Success cannot be evaluated; "
                               "re-run the scenario bank with the current run.py")
        ph = {k: z[k] for k in ("ph_f_stator", "ph_f_mover", "ph_q", "ph_rel_pos", "ph_cmd", "ph_dt")}
        PH[(r["scenario_id"], r["g"])] = ph
        a = RT.analyze_retention(ph, r["side"], r["success_b"], ret_cfg)
        st = RT.analyze_retention(ph, r["side"], r["success_b"], ret_cfg, settled=False)
        r["ret"] = a
        r["oneshot_b"] = a["one_shot_success"]
        r["oneshot_strict_b"] = st["one_shot_success"]
        r["acq_b"] = a["acquired"]
        r["drop_b"] = a["releases"] > 0
        ret_rows.append({
            "scenario_id": r["scenario_id"], "group": r["g"], "interval_cm": label[r["g"]], "cube_side_cm": 100 * r["side"],
            "cube_density": r["rho"], "cube_mass_kg": r["mass"], "task_success": r["success_b"],
            "one_shot_success": a["one_shot_success"], "one_shot_strict": st["one_shot_success"],
            "failure_reason": r["failure_reason"], "completion_time_s": _f(r["completion_time_s"]),
            "physically_acquired": a["acquired"], "first_acquisition_s": a["first_acquisition_s"],
            "training_grasp_confirmation_s": _f(r["time_to_grasp_s"]),
            "attempts_before_acquisition": a["attempts_before_acquisition"],
            "drops": a["releases"], "drop_times_s": ";".join(f"{t:.3f}" for t in a["release_times_s"]),
            "drop_gap_ms": ";".join(f"{t:.0f}" for t in a["release_gap_ms"]),
            "drop_rel_disp_mm": ";".join(f"{t:.1f}" for t in a["release_disp_mm"]),
            "drop_jaw_closing_deg": ";".join(f"{t:.1f}" for t in a["release_jaw_closing_deg"]),
            "regrasps": a["reacquisitions"], "regrasp_times_s": ";".join(f"{t:.3f}" for t in a["reacquisition_times_s"]),
            "retained_at_end": a["retained_at_end"], "flicker_gaps": a["flicker_gaps"], "flicker_max_ms": a["flicker_max_ms"],
            "close_to_open_cmd_after_acquisition": a["close_to_open_cmd_after_acquisition"],
            "close_to_open_cmd_while_held": a["close_to_open_cmd_while_held"],
            "close_to_open_cmd_while_held_harmless": a["close_to_open_cmd_while_held_harmless"],
            "touch_without_pinch_after_drop_s": a["touch_without_pinch_after_release_s"],
        })
    wcsv("episodes_retention.csv", ret_rows)
    # threshold sensitivity of the One-Shot count (same logs, one parameter changed at a time)
    variants = [("official", {}), ("hold_force 0.3 N (sensor threshold)", {"hold_force_threshold": 0.3}),
                ("hold_force 3 N", {"hold_force_threshold": 3.0}), ("hold_force 30 N", {"hold_force_threshold": 30.0}),
                ("hold_force 100 N", {"hold_force_threshold": 100.0}),
                ("release gap 24 ms", {"release_min_gap_s": 0.024}), ("release gap 60 ms", {"release_min_gap_s": 0.06}),
                ("release gap 96 ms", {"release_min_gap_s": 0.096}),
                ("jaw closing 3 deg", {"release_jaw_closing_deg": 3.0}), ("jaw closing 8 deg", {"release_jaw_closing_deg": 8.0}),
                ("displacement side/4", {"release_rel_disp_fraction_of_side": 0.25}),
                ("displacement 1 side", {"release_rel_disp_fraction_of_side": 1.0}),
                ("acquisition jaw range 2 deg", {"acquisition_max_jaw_range_deg": 2.0}),
                ("acquisition rel. disp 5 mm", {"acquisition_max_rel_disp_m": 0.005}),
                ("acquisition window 0.2 s", {"acquisition_window_s": 0.2})]
    sens = []
    for name, over in variants + [("strict: any 0.1-s pinch is an acquisition", None)]:
        c = dict(ret_cfg, **(over or {}))
        cnt = {g: 0 for g in groups}
        for r in rows:
            a = RT.analyze_retention(PH[(r["scenario_id"], r["g"])], r["side"], r["success_b"], c, settled=over is not None)
            cnt[r["g"]] += a["one_shot_success"]
        sens.append({"variant": name, **{f"one_shot_G{g}": cnt[g] for g in groups}, "one_shot_total": sum(cnt.values()),
                     "task_success_total": sum(r["success_b"] for r in rows)})
    wcsv("oneshot_sensitivity.csv", sens)

    # ------------------------------------------------------------------ per group
    summary, gm = [], []
    for g in groups:
        rs = by[g]
        n = len(rs)
        k = sum(r["success_b"] for r in rs)
        k1 = sum(r["oneshot_b"] for r in rs)
        lo, hi = wilson_ci(k, n)
        lo1, hi1 = wilson_ci(k1, n)
        ct_m, ct_sd, ct_n = mean_std([_f(r["completion_time_s"]) for r in rs if r["success_b"]])
        conf = [r for r in rs if r["grasp_b"]]
        nconf = len(conf)
        glo, ghi = wilson_ci(nconf, n)
        acqd = [r for r in rs if r["acq_b"]]
        nacq = len(acqd)
        dropped = sum(r["drop_b"] for r in acqd)
        ta = mean_std([r["ret"]["first_acquisition_s"] for r in acqd])
        summary.append({"size_interval_cm": label[g], "n_trials": n, "task_successes": k, "one_shot_successes": k1,
                        "task_success_rate_pct": 100.0 * k / n, "one_shot_success_rate_pct": 100.0 * k1 / n,
                        "completion_time_mean_s": ct_m, "completion_time_sd_s": ct_sd, "completion_time_n": ct_n})
        tg = mean_std([_f(r["time_to_grasp_s"]) for r in conf])
        rp = mean_std([_f(r["retrieval_phase_time_s"]) for r in rs if r["success_b"]])
        gm.append({
            "size_interval_cm": label[g], "n_trials": n,
            "side_cm_min": min(100 * r["side"] for r in rs), "side_cm_max": max(100 * r["side"] for r in rs),
            "density_min": min(r["rho"] for r in rs), "density_max": max(r["rho"] for r in rs),
            "mass_kg_min": min(r["mass"] for r in rs), "mass_kg_max": max(r["mass"] for r in rs),
            "task_successes": k, "task_success_pct": 100.0 * k / n, "task_wilson95_lo_pct": 100 * lo, "task_wilson95_hi_pct": 100 * hi,
            "one_shot_successes": k1, "one_shot_pct": 100.0 * k1 / n, "one_shot_wilson95_lo_pct": 100 * lo1,
            "one_shot_wilson95_hi_pct": 100 * hi1,
            "completion_time_mean_s": ct_m, "completion_time_sd_s": ct_sd, "completion_time_n": ct_n,
            "completion_time_median_s": float(np.median([_f(r["completion_time_s"]) for r in rs if r["success_b"]])) if k else float("nan"),
            "grasp_confirmed": nconf, "grasp_confirmation_pct": 100.0 * nconf / n,
            "grasp_wilson95_lo_pct": 100 * glo, "grasp_wilson95_hi_pct": 100 * ghi,
            "time_to_first_grasp_mean_s": tg[0], "time_to_first_grasp_sd_s": tg[1],
            "grasp_to_success_mean_s": rp[0], "grasp_to_success_sd_s": rp[1],
            "one_shot_strict_successes": sum(r["oneshot_strict_b"] for r in rs),
            "physically_acquired": nacq, "physical_acquisition_pct": 100.0 * nacq / n,
            "time_to_first_acquisition_mean_s": ta[0], "time_to_first_acquisition_sd_s": ta[1],
            "drop_episodes": dropped, "drop_pct_of_acquired": 100.0 * dropped / nacq if nacq else float("nan"),
            "first_acquisition_retention_pct_of_acquired": 100.0 * (nacq - dropped) / nacq if nacq else float("nan"),
            "drop_events": sum(r["ret"]["releases"] for r in rs),
            "regrasp_episodes": sum(r["ret"]["reacquisitions"] > 0 for r in rs),
            "successful_after_regrasp": sum(r["success_b"] and r["drop_b"] for r in rs),
            "training_grasp_loss_terminations": sum(r["lost_b"] for r in rs),
            "episodes_with_flicker": sum(r["ret"]["flicker_gaps"] > 0 for r in rs),
            "flicker_gaps_total": sum(r["ret"]["flicker_gaps"] for r in rs),
            "episodes_with_failed_attempts_before_acquisition": sum(r["ret"]["attempts_before_acquisition"] > 0 for r in acqd),
            "close_to_open_cmd_while_held": sum(r["ret"]["close_to_open_cmd_while_held"] for r in rs),
            "timeout_pct": 100.0 * sum(r["failure_reason"] in ("timeout", "grasp_not_achieved") for r in rs) / n,
            "s2_mean_grip_force_N": mean_std([0.5 * (_f(r["s2_mean_f_stator"]) + _f(r["s2_mean_f_mover"])) for r in conf])[0],
            "s2_max_penetration_mm_p95": float(np.nanpercentile([_f(r["s2_max_penetration_mm"]) for r in conf], 95)) if nconf else float("nan"),
            "s2_cube_slip_mm_median": float(np.nanmedian([_f(r["s2_cube_slip_mm"]) for r in conf])) if nconf else float("nan"),
            "s2_dual_contact_frac_mean": mean_std([_f(r["s2_dual_contact_frac"]) for r in conf])[0],
            "s2_gripper_angle_deg_mean": mean_std([_f(r["s2_mean_gripper_angle_deg"]) for r in conf])[0],
        })
    wcsv("summary_table.csv", summary)
    wcsv("group_metrics.csv", gm)
    fb = []
    for g in groups:
        c = Counter(r["failure_reason"] for r in by[g])
        fb.append({"size_interval_cm": label[g], "n_trials": len(by[g]), **{k: c.get(k, 0) for k in FAILURE_REASONS}})
    wcsv("failure_breakdown.csv", fb)
    sd = []
    for g in groups:
        fail = [r for r in by[g] if not r["success_b"]]
        sd.append({"size_interval_cm": label[g], "n_failures": len(fail),
                   "approach_never_stage1": sum(not r["s1_b"] for r in fail),
                   "grasp_stage1_not_confirmed": sum(r["s1_b"] and not r["grasp_b"] for r in fail),
                   "post_grasp": sum(r["grasp_b"] for r in fail),
                   "post_grasp_lost": sum(r["lost_b"] for r in fail),
                   "post_grasp_timeout": sum(r["failure_reason"] == "timeout" for r in fail),
                   "other_termination": sum(r["failure_reason"] in ("base_contact_failure", "grasp_error_out_of_bounds",
                                                                     "numerical_failure") for r in fail)})
    wcsv("stage_decomposition.csv", sd)

    # ------------------------------------------------------------------ density / mass diagnostics
    from scipy.stats import mannwhitneyu
    dm = []
    for var, key, unit in (("density", "rho", "kg/m^3"), ("mass", "mass", "kg")):
        vals = np.array([r[key] for r in rows])
        edges = np.quantile(vals, [0, 0.25, 0.5, 0.75, 1.0])
        for i in range(4):
            sel = [r for r in rows if (edges[i] <= r[key] < edges[i + 1]) or (i == 3 and r[key] == edges[4])]
            k = sum(r["success_b"] for r in sel)
            lo, hi = wilson_ci(k, len(sel))
            dm.append({"variable": var, "unit": unit, "bin": f"Q{i + 1}", "lo": float(edges[i]), "hi": float(edges[i + 1]),
                       "n": len(sel), "task_success_pct": 100.0 * k / len(sel), "wilson_lo_pct": 100 * lo, "wilson_hi_pct": 100 * hi,
                       "one_shot_pct": 100.0 * sum(r["oneshot_b"] for r in sel) / len(sel),
                       "post_grasp_loss_pct": 100.0 * sum(r["lost_b"] for r in sel) / len(sel)})
    wcsv("density_mass_bins.csv", dm)

    # ------------------------------------------------------------------ paired tests
    ids = sorted({r["scenario_id"] for r in rows})
    idx = {g: {r["scenario_id"]: r for r in by[g]} for g in groups}
    complete = [i for i in ids if all(i in idx[g] for g in groups)]
    tests = {"n_paired_scenarios": len(complete), "groups": {g: label[g] for g in groups}}
    for metric, key in (("task_success", "success_b"), ("one_shot", "oneshot_b")):
        mat = np.array([[idx[g][i][key] for g in groups] for i in complete], dtype=bool)
        q, p = cochran_q(mat.astype(int))
        pairs = [(1, 2), (2, 3), (3, 4), (1, 4)] if len(groups) == 4 else list(zip(groups[:-1], groups[1:]))
        mc, raw = {}, {}
        for a, b in pairs:
            n01, n10, pv = mcnemar_exact(mat[:, a - 1], mat[:, b - 1])
            key2 = f"G{a}_vs_G{b}"
            mc[key2] = {"only_G%d_success" % b: n01, "only_G%d_success" % a: n10, "p_exact": pv}
            raw[key2] = pv
        for k2, v in holm(raw).items():
            mc[k2]["p_holm"] = v
        tests[metric] = {"cochran_q": {"Q": q, "df": len(groups) - 1, "p": p}, "mcnemar": mc,
                         "success_at_all_groups": int(mat.all(1).sum()), "success_at_no_group": int((~mat.any(1)).sum())}
    boot = {}
    for a, b in [(1, 2), (2, 3), (3, 4), (1, 4)]:
        m_ids = [i for i in complete if idx[a][i]["success_b"] and idx[b][i]["success_b"]]
        x = [_f(idx[a][i]["completion_time_s"]) for i in m_ids]
        y = [_f(idx[b][i]["completion_time_s"]) for i in m_ids]
        d, lo, hi = paired_bootstrap(x, y, seed=a * 10 + b)
        boot[f"G{b}_minus_G{a}"] = {"n_matched": len(m_ids), "mean_diff_s": d, "ci95_lo_s": lo, "ci95_hi_s": hi}
    tests["completion_time_paired_bootstrap"] = boot
    X = np.array([[100 * r["side"], r["rho"]] for r in rows])
    y = np.array([0.0 if r["success_b"] else 1.0 for r in rows])
    tests["logistic_failure_on_side_and_density"] = logistic(X, y, ["side_cm", "density"]) if 0 < y.sum() < len(y) else "no variation"
    for var, key in (("density", "rho"), ("mass", "mass"), ("side", "side")):
        a = [r[key] for r in rows if r["success_b"]]
        b = [r[key] for r in rows if not r["success_b"]]
        tests[f"mannwhitney_{var}_success_vs_failure"] = (
            {"median_success": float(np.median(a)), "median_failure": float(np.median(b)),
             "p": float(mannwhitneyu(a, b).pvalue)} if a and b else "n/a")
    with open(os.path.join(out_dir, "paired_tests.json"), "w") as f:
        json.dump(tests, f, indent=1)

    # ------------------------------------------------------------------ one-shot audit
    diff = [r for r in rows if r["success_b"] and not r["oneshot_b"]]
    rr = [x for x in ret_rows]
    audit = {
        "definition": ("One-shot success = task success AND continuous physical retention (load-bearing pinch, "
                       "flicker-tolerant) from the first physical acquisition to the success instant, i.e. no "
                       "drop and re-grasp. Evaluated from the 250-Hz log, independent of the training Stage-2 state."),
        "task_success_total": sum(r["success_b"] for r in rows), "one_shot_total": sum(r["oneshot_b"] for r in rows),
        "one_shot_strict_total": sum(r["oneshot_strict_b"] for r in rows),
        "task_success_but_not_one_shot": len(diff),
        "one_shot_without_task_success": sum(r["oneshot_b"] and not r["success_b"] for r in rows),
        "not_one_shot_episodes": [{k: x[k] for k in ("scenario_id", "group", "first_acquisition_s", "drop_times_s",
                                                     "drop_gap_ms", "drop_rel_disp_mm", "drop_jaw_closing_deg",
                                                     "regrasp_times_s", "training_grasp_confirmation_s", "completion_time_s")}
                                  for x in rr if x["task_success"] and not x["one_shot_success"]],
        "drops_before_training_confirmation_in_successes": sum(
            1 for x in rr if x["task_success"] and x["drops"] and float(x["drop_times_s"].split(";")[0]) < x["training_grasp_confirmation_s"]),
        "training_grasp_loss_terminations": sum(r["lost_b"] for r in rows),
        "training_grasp_loss_terminations_with_physical_drop": sum(r["lost_b"] and r["drop_b"] for r in rows),
        "confirmed_grasps_without_physical_acquisition": sum(r["grasp_b"] and not r["acq_b"] for r in rows),
        "max_flicker_gap_ms": max(x["flicker_max_ms"] for x in rr),
        "close_to_open_cmd_while_held": sum(x["close_to_open_cmd_while_held"] for x in rr),
        "close_to_open_cmd_after_acquisition": sum(x["close_to_open_cmd_after_acquisition"] for x in rr),
        "previous_definition_note": ("The previous metric started at the training Stage-2 confirmation, sampled contacts "
                                     "at 10 Hz and required both a contact-free jaw pair and grasp error > 0.10 m to call "
                                     "a drop; it therefore missed drops before confirmation and slips that stay within "
                                     "0.10 m or keep single-jaw contact, and reported one-shot == task success."),
    }
    with open(os.path.join(out_dir, "oneshot_audit.json"), "w") as f:
        json.dump(audit, f, indent=1)

    # ------------------------------------------------------------------ consistency
    issues, findings = [], []
    max_steps = int(meta["timing"]["timeout_after_hl_steps"])
    for r in rows:
        tag = f"s{r['scenario_id']}/G{r['g']}"
        lo_cm, hi_cm = intervals[r["g"] - 1]
        cm = 100 * r["side"]
        if not (lo_cm <= cm < hi_cm or (r["g"] == len(groups) and lo_cm <= cm <= hi_cm)):
            issues.append(f"{tag}: side {cm} outside interval")
        if abs(cm - (lo_cm + _f(r["size_u"]) * (hi_cm - lo_cm))) > 1e-9:
            issues.append(f"{tag}: side not reproducible from size_u")
        if not (rho_lo <= r["rho"] <= rho_hi) or abs(r["rho"] - (rho_lo + _f(r["density_u"]) * (rho_hi - rho_lo))) > 1e-9:
            issues.append(f"{tag}: density")
        if abs(_f(r["mujoco_mass_kg"]) - r["rho"] * r["side"] ** 3) > 1e-12 or abs(r["mass"] - _f(r["mujoco_mass_kg"])) > 1e-12:
            issues.append(f"{tag}: MuJoCo mass read-back")
        if abs(_f(r["mujoco_inertia"]) - _f(r["expected_inertia"])) > 1e-9 * abs(_f(r["expected_inertia"])):
            issues.append(f"{tag}: MuJoCo inertia read-back")
        ct = _f(r["completion_time_s"])
        if r["success_b"] != (r["failure_reason"] == "success"):
            issues.append(f"{tag}: success flag vs reason")
        if r["oneshot_b"] and not (r["success_b"] and r["grasp_b"] and r["acq_b"]):
            issues.append(f"{tag}: one-shot without success/grasp/acquisition")
        if r["oneshot_b"] and r["drop_b"]:
            issues.append(f"{tag}: one-shot with a physical drop")
        # Training-proxy vs physical-retention disagreements are physical findings, not data errors:
        # e.g. oversized cubes squeezed out of the jaws while the 3-deg/step proxy confirms a "grasp".
        if r["lost_b"] and not r["drop_b"]:
            findings.append(f"{tag}: training grasp-loss termination without a physical drop "
                            f"({'never physically acquired' if not r['acq_b'] else 'acquired, retained'})")
        if r["grasp_b"] and not r["acq_b"]:
            findings.append(f"{tag}: training-confirmed grasp without physical acquisition (pinch never settled)")
        if r["success_b"] and not r["oneshot_b"] and not r["drop_b"]:
            issues.append(f"{tag}: task success not one-shot without a physical drop")
        if r["oneshot_strict_b"] and not r["oneshot_b"]:
            issues.append(f"{tag}: strict one-shot but not one-shot")
        if r["failure_reason"] not in FAILURE_REASONS:
            issues.append(f"{tag}: unknown reason")
        if r["success_b"]:
            if not (np.isfinite(ct) and 0 < ct <= max_steps * HL_DT + 1e-9) or abs(ct - _f(r["episode_time_s"])) > 1e-9:
                issues.append(f"{tag}: completion time {ct}")
            if not r["grasp_b"] or not (_f(r["time_to_grasp_s"]) < ct):
                issues.append(f"{tag}: success without earlier grasp")
        elif np.isfinite(ct):
            issues.append(f"{tag}: completion time for failed episode")
        if r["failure_reason"] in ("timeout", "grasp_not_achieved") and int(r["hl_steps"]) != max_steps:
            issues.append(f"{tag}: timeout at {r['hl_steps']} steps")
        if r["error"]:
            issues.append(f"{tag}: engine error {r['error']}")
    n_traj, mism = 0, []
    task = cfg["task"]
    for r in rows:
        p = os.path.join(traj_dir, f"s{r['scenario_id']:04d}_g{r['g']}.npz")
        if not os.path.isfile(p):
            continue
        n_traj += 1
        first, grasp, n, side, rho, mass = recompute_from_trajectory(
            p, float(task["retrieval_success_error_threshold"]), int(task["retrieval_success_consecutive_steps"]))
        ok = n == int(r["hl_steps"]) and side == r["side"] and rho == r["rho"] and mass == _f(r["mujoco_mass_kg"])
        if r["success_b"]:
            ok &= first is not None and abs(first * HL_DT - _f(r["completion_time_s"])) < 1e-9
        elif r["failure_reason"] != "numerical_failure":
            ok &= first is None or first == n
        if r["grasp_b"]:
            ok &= grasp is not None and abs(grasp * HL_DT - _f(r["time_to_grasp_s"])) < 1e-9
        ok &= len(PH[(r["scenario_id"], r["g"])]["ph_q"]) == 25 * n
        if not ok:
            mism.append(f"s{r['scenario_id']}/G{r['g']}")
    consistency = {"n_rows": len(rows), "n_expected": len(ids) * len(groups), "row_issues": issues,
                   "physical_findings": findings,
                   "trajectories_checked": n_traj, "trajectory_mismatches": mism,
                   "one_shot_le_task_success_every_group": all(s["one_shot_successes"] <= s["task_successes"] for s in summary),
                   "single_noise_setting": noise[0],
                   "ok": not issues and not mism and len(rows) == len(ids) * len(groups)}
    with open(os.path.join(out_dir, "consistency.json"), "w") as f:
        json.dump(consistency, f, indent=1)

    # ------------------------------------------------------------------ LaTeX
    tex = [
        "% E5 dynamic object retrieval -- generated by retrieval_experiment/analyze.py",
        f"% run: {os.path.basename(run_dir)}; {len(ids)} paired scenarios per interval; side ~ U(interval), "
        f"density ~ U({rho_lo:.0f}, {rho_hi:.0f}) kg/m^3 (training: {cfg['cube']['training_density']:.0f});",
        f"% observation noise: {noise[0]}; MuJoCo noslip iterations: {cfg.get('mujoco_noslip_iterations')}; intervals are"
        " [lo, hi) except the last [5.5, 6.0].",
        "% SR: task success; One-Shot: task success with continuous physical retention (no drop and re-grasp) after the",
        "% first physical acquisition, from the 250-Hz contact/geometry log (all trials);",
        "% Time: completion time mean +- SD over successful trials only.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Dynamic object retrieval performance across different object sizes and densities.}",
        "\\label{tab:retrieval_size_density}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{cccc}",
        "\\toprule",
        "Size (cm) & SR (\\%) $\\uparrow$ & One-Shot SR (\\%) $\\uparrow$ & Time (s) $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for s in summary:
        tex.append(f"{{{s['size_interval_cm']}}} & {s['task_success_rate_pct']:.1f} & {s['one_shot_success_rate_pct']:.1f} & "
                   f"{fmt_ct(s['completion_time_mean_s'], s['completion_time_sd_s'], s['completion_time_n'])} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(os.path.join(out_dir, "summary_table.tex"), "w") as f:
        f.write("\n".join(tex))

    # ------------------------------------------------------------------ Markdown
    def pct(x):
        return "–" if not np.isfinite(x) else f"{x:.1f}"
    md = [f"# E5 dynamic object retrieval -- analysis of `{os.path.basename(run_dir)}`", "",
          f"- Episodes: {len(rows)} = {len(ids)} paired base scenarios x {len(groups)} size intervals "
          f"({', '.join(label[g] for g in groups)} cm; side ~ U(interval), unrounded)",
          f"- Cube density ~ U({rho_lo:.0f}, {rho_hi:.0f}) kg/m^3 per base scenario, shared by all intervals "
          f"(evaluation-time perturbation; training used {cfg['cube']['training_density']:.0f}); mass = density x side^3",
          f"- Observation noise: **{noise[0]}**; MuJoCo noslip iterations: **{cfg.get('mujoco_noslip_iterations')}**",
          f"- Horizon {max_steps} HL steps ({max_steps * HL_DT:.1f} s). Task success: Stage 2 and ||cube - target|| < "
          f"{task['retrieval_success_error_threshold']} m for {task['retrieval_success_consecutive_steps']} consecutive steps. "
          "One-shot success: task success AND continuous physical retention (load-bearing pinch, flicker-tolerant) from the "
          "first physical acquisition to the success instant -- no drop and re-grasp (retention.py, 250-Hz log). Both over all trials.",
          f"- Consistency checks: **{'PASS' if consistency['ok'] else 'FAIL'}** ({n_traj} trajectories re-evaluated, "
          f"{len(mism)} mismatches, {len(issues)} row issues; MuJoCo mass/inertia read-back checked per episode)", "",
          "## Primary table", "",
          "| Object Size (cm) | Success Rate (%) ↑ | One-Shot Success (%) ↑ | Completion Time (s) ↓ | n (succ. / one-shot / trials) |",
          "|---|---:|---:|---:|---:|"]
    for s in summary:
        ct = "–" if s["completion_time_n"] == 0 else f"{s['completion_time_mean_s']:.2f} ± {s['completion_time_sd_s']:.2f}"
        md.append(f"| {s['size_interval_cm']} | {s['task_success_rate_pct']:.1f} | {s['one_shot_success_rate_pct']:.1f} | {ct} | "
                  f"{s['task_successes']} / {s['one_shot_successes']} / {s['n_trials']} |")
    md += ["", "## Per-interval statistics", "",
           "| Interval | side (cm) | mass (kg) | Task 95% CI | One-shot 95% CI | t_compl n | Physically acquired (%) | t→first acquisition (s) | Training grasp confirmed (%) | grasp→success (s) | Drop after acquisition (% acq.) | First-acquisition retention (% acq.) |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in gm:
        md.append(f"| {x['size_interval_cm']} | {x['side_cm_min']:.3f}–{x['side_cm_max']:.3f} | {x['mass_kg_min']:.4f}–{x['mass_kg_max']:.4f} "
                  f"| {x['task_wilson95_lo_pct']:.1f}–{x['task_wilson95_hi_pct']:.1f} | {x['one_shot_wilson95_lo_pct']:.1f}–{x['one_shot_wilson95_hi_pct']:.1f} "
                  f"| {x['completion_time_n']} | {x['physical_acquisition_pct']:.1f} | {x['time_to_first_acquisition_mean_s']:.2f} ± {x['time_to_first_acquisition_sd_s']:.2f} "
                  f"| {x['grasp_confirmation_pct']:.1f} | {x['grasp_to_success_mean_s']:.2f} ± {x['grasp_to_success_sd_s']:.2f} "
                  f"| {pct(x['drop_pct_of_acquired'])} | {pct(x['first_acquisition_retention_pct_of_acquired'])} |")
    md += ["", "Physical retention events (250-Hz log; all episodes of the interval):", "",
           "| Interval | drop events | episodes with drop | re-grasp episodes | successful after drop + re-grasp | training grasp-loss terminations | episodes with flicker (gaps) | failed pinch attempts before acquisition (episodes) | close→open commands while held | strict one-shot |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in gm:
        md.append(f"| {x['size_interval_cm']} | {x['drop_events']} | {x['drop_episodes']} | {x['regrasp_episodes']} | "
                  f"{x['successful_after_regrasp']} | {x['training_grasp_loss_terminations']} | {x['episodes_with_flicker']} ({x['flicker_gaps_total']}) | "
                  f"{x['episodes_with_failed_attempts_before_acquisition']} | {x['close_to_open_cmd_while_held']} | {x['one_shot_strict_successes']} |")
    md += ["", "One-Shot threshold sensitivity (same logs; one parameter changed at a time):", "",
           "| variant | " + " | ".join(f"G{g}" for g in groups) + " | total one-shot | task successes |", "|---|" + "---:|" * (len(groups) + 2)]
    for x in sens:
        md.append(f"| {x['variant']} | " + " | ".join(str(x[f'one_shot_G{g}']) for g in groups) + f" | {x['one_shot_total']} | {x['task_success_total']} |")
    md += ["", "In-episode grasp physics (Stage-2 steps of grasp-confirmed episodes):", "",
           "| Interval | mean jaw normal force (N) | p95 max penetration (mm) | median cube slip (mm) | dual-contact fraction | mean gripper angle (deg) |",
           "|---|---:|---:|---:|---:|---:|"]
    for x in gm:
        md.append(f"| {x['size_interval_cm']} | {x['s2_mean_grip_force_N']:.0f} | {x['s2_max_penetration_mm_p95']:.2f} | "
                  f"{x['s2_cube_slip_mm_median']:.1f} | {x['s2_dual_contact_frac_mean']:.3f} | {x['s2_gripper_angle_deg_mean']:.1f} |")
    md += ["", "## Failure breakdown", "", "| Interval | " + " | ".join(FAILURE_REASONS) + " |",
           "|---|" + "---:|" * len(FAILURE_REASONS)]
    for x in fb:
        md.append(f"| {x['size_interval_cm']} | " + " | ".join(str(x[k]) for k in FAILURE_REASONS) + " |")
    md += ["", "| Interval | failures | approach (never S1) | grasp (S1, never confirmed) | post-grasp (lost / timeout) | other |",
           "|---|---:|---:|---:|---:|---:|"]
    for x in sd:
        md.append(f"| {x['size_interval_cm']} | {x['n_failures']} | {x['approach_never_stage1']} | {x['grasp_stage1_not_confirmed']} | "
                  f"{x['post_grasp']} ({x['post_grasp_lost']} / {x['post_grasp_timeout']}) | {x['other_termination']} |")
    md += ["", "## Success vs sampled density and mass (pooled over intervals, quartile bins)", "",
           "| variable | bin | range | n | task success (%) | 95% CI | one-shot (%) | post-grasp loss (%) |",
           "|---|---|---|---:|---:|---:|---:|---:|"]
    for x in dm:
        fmt = "{:.0f}" if x["variable"] == "density" else "{:.4f}"
        md.append(f"| {x['variable']} ({x['unit']}) | {x['bin']} | {fmt.format(x['lo'])}–{fmt.format(x['hi'])} | {x['n']} | "
                  f"{x['task_success_pct']:.1f} | {x['wilson_lo_pct']:.1f}–{x['wilson_hi_pct']:.1f} | {x['one_shot_pct']:.1f} | {x['post_grasp_loss_pct']:.1f} |")
    md += ["", "## Paired statistics", "", "```json", json.dumps(tests, indent=1), "```", "",
           "## Task success vs one-shot success", "", "```json", json.dumps(audit, indent=1), "```", ""]
    if findings:
        md += ["## Physical findings (training proxy vs physical retention; not data errors)", "",
               "The training grasp proxy tolerates 3 deg of jaw motion per HL step; a cube that is pinched but keeps "
               "sliding out of the jaws (jaw closing, cube moving relative to the jaws) can be 'confirmed' and then "
               "lost without ever being physically acquired. These episodes are failures under both metrics.", ""] + \
              [f"- {x}" for x in findings] + [""]
    if issues or mism:
        md += ["## Consistency issues", ""] + [f"- {i}" for i in issues[:50]] + [f"- trajectory mismatch {t}" for t in mism[:50]]
    with open(os.path.join(out_dir, "ANALYSIS.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[: 14 + len(summary)]))
    print(f"[E5 analyze] consistency {'PASS' if consistency['ok'] else 'FAIL'} -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
