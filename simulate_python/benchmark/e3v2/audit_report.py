#!/usr/bin/env python3
"""E3 v2 audit: summaries, sensitivity analyses and audit figures.

Inputs: results/E3_v2/{data,stats,validation,protocol} and the raw audit tables
from benchmark.e3v2.audit. Writes ONLY results/E3_v2/audit/ (tables, figures,
audit_numbers.json, audit_tables.md). Never modifies the canonical dataset.
Run:  python3 -m benchmark.e3v2.audit_report
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy import stats as sps

from benchmark.round3 import common  # noqa: F401
from benchmark.core import stats  # noqa: E402
from benchmark.e3v2 import analyze as A  # noqa: E402
from benchmark.e3v2.audit import PRACTICAL_REL, NEAR_FALL  # noqa: E402
from benchmark.plots import style  # noqa: E402
from benchmark.plots.style import plt  # noqa: E402

ROOT, OUT = A.ROOT, os.path.join(A.ROOT, "audit")
FIGD = os.path.join(OUT, "figures")
BLEV, VLEV, FAMS, CTRL, MK = A.BLEV, A.VLEV, A.FAMILIES, A.BASELINES, A.MK
NUM = {}
MD = []


def rd(path):
    with open(path) as h:
        rows = list(csv.DictReader(h))
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, str) and v.startswith("np.float64(") and v.endswith(")"):
                v = v[len("np.float64("):-1]   # numpy repr written by select_orientation
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def pct(k, n):
    lo, hi = stats.wilson_interval(int(k), int(n))
    return f"{100 * k / n:.1f} % [{100 * lo:.1f}, {100 * hi:.1f}] (n={int(n)})" if n else "n/a"


def table(head, rows):
    MD.append("| " + " | ".join(head) + " |")
    MD.append("|" + "---|" * len(head))
    for r in rows:
        MD.append("| " + " | ".join(str(x) for x in r) + " |")
    MD.append("")


# ======================================================================== A1
def a1():
    MD.append("## A1 Feasibility at the base states actually encountered in E3 v2\n")
    rows = rd(os.path.join(OUT, "A1_feasibility_states.csv"))
    res = {}
    out = []
    for stratum in ("random", "pre-termination"):
        for state in ("normal", "near-fall"):
            sel = [r for r in rows if r["stratum"] == stratum and r["state"] == state]
            if not sel:
                continue
            n = len(sel)
            k_hw = sum(r["reach_hw"] for r in sel)
            k_sf = sum(r["reach_c3_soft"] for r in sel)
            k_sg = sum(r["near_singular"] for r in sel)
            res[f"{stratum}/{state}"] = {"n": n, "reach_hw": k_hw / n, "reach_c3_soft": k_sf / n,
                                         "near_singular": k_sg / n}
            out.append([stratum, state, n, pct(k_hw, n), pct(k_sf, n), pct(k_sg, n),
                        f"{np.percentile([r['joint_margin_hw_rad'] for r in sel], 5):.3f}"])
    table(["sample", "base state", "n states", "reachable (hardware range)", "reachable (C3 IK soft band)",
           "near-singular (σmin<0.05)", "p5 joint margin [rad]"], out)
    NUM["A1"] = res
    # by trajectory x base level (random stratum, normal states)
    MD.append("Reachability in normal states (random sample), by trajectory and base bound — hardware / C3 soft band:\n")
    out = []
    for f in FAMS:
        line = [f]
        for B in BLEV:
            sel = [r for r in rows if r["stratum"] == "random" and r["state"] == "normal" and r["trajectory"] == f
                   and abs(r["base_level"] - B) < 1e-9]
            line.append(f"{100 * np.mean([r['reach_hw'] for r in sel]):.1f} / "
                        f"{100 * np.mean([r['reach_c3_soft'] for r in sel]):.1f} (n={len(sel)})")
        out.append(line)
    table(["trajectory"] + [f"B={B:.2f}" for B in BLEV], out)
    out = []
    for f in FAMS:
        line = [f]
        for v in VLEV:
            sel = [r for r in rows if r["stratum"] == "random" and r["state"] == "normal" and r["trajectory"] == f
                   and abs(r["ee_speed_mps"] - v) < 1e-9]
            line.append(f"{100 * np.mean([r['reach_hw'] for r in sel]):.1f} / "
                        f"{100 * np.mean([r['reach_c3_soft'] for r in sel]):.1f}")
        out.append(line)
    MD.append("Same, by EE speed:\n")
    table(["trajectory"] + [f"v={v:.2f}" for v in VLEV], out)
    # by source controller (whose base states)
    out = []
    for c in CTRL:
        for state in ("normal", "near-fall"):
            sel = [r for r in rows if r["controller"] == c and r["state"] == state]
            if sel:
                out.append([c, state, len(sel), pct(sum(r["reach_hw"] for r in sel), len(sel)),
                            pct(sum(r["reach_c3_soft"] for r in sel), len(sel))])
    MD.append("By the controller whose base states were sampled (both strata):\n")
    table(["base states from", "state", "n", "reachable (hw)", "reachable (C3 soft)"], out)
    bad = [r for r in rows if r["state"] == "normal" and not r["reach_hw"]]
    NUM["A1_unreachable_normal"] = len(bad)
    if bad:
        MD.append("Unreachable targets in NORMAL states (hardware range):\n")
        table(["ctrl", "traj", "B", "v", "seed", "t", "z", "roll", "pitch", "pos resid [mm]", "ori resid [deg]"],
              [[r["controller"], r["trajectory"], r["base_level"], r["ee_speed_mps"], int(r["seed"]),
                f"{r['t']:.2f}", f"{r['base_z']:.3f}", f"{r['roll_deg']:.1f}", f"{r['pitch_deg']:.1f}",
                f"{1000 * r['pos_resid_hw_m']:.1f}", f"{r['ori_resid_hw_deg']:.2f}"] for r in bad[:25]])
    soft_only = [r for r in rows if r["reach_hw"] and not r["reach_c3_soft"]]
    NUM["A1_hw_but_not_soft"] = len(soft_only)
    NUM["A1_hw_but_not_soft_normal"] = sum(r["state"] == "normal" for r in soft_only)
    # E1 attitude log (the 92 %)
    att = rd(os.path.join(ROOT, "protocol", "orientation_attitude_log.csv"))
    fail = [a for a in att if a["unreachable_probe_points"] > 0]
    nf = lambda a: (a["base_z_m"] < NEAR_FALL["z_min"] or abs(a["roll_deg"]) > NEAR_FALL["tilt_max_deg"]
                    or abs(a["pitch_deg"]) > NEAR_FALL["tilt_max_deg"])
    n_norm = sum(not nf(a) for a in att)
    n_norm_fail = sum(not nf(a) for a in fail)
    NUM["A1_E1_attitudes"] = {"n": len(att), "fail": len(fail), "fail_near_fall": len(fail) - n_norm_fail,
                              "normal": n_norm, "normal_fail": n_norm_fail}
    MD.append(f"Selection-time E1 attitude sample: {len(att)} states, {len(fail)} with any unreachable probe point; "
              f"{len(fail) - n_norm_fail} of those are near-fall by the pre-declared rule (height/tilt; the "
              f"pre-termination criterion is not available for that sample). Normal states: {n_norm - n_norm_fail}/"
              f"{n_norm} reach every probe point.\n")
    table(["attitude", "from", "z [m]", "roll", "pitch", "near-fall?", "unreachable probe pts"],
          [[int(a["attitude"]), a["source_controller"], f"{a['base_z_m']:.3f}", f"{a['roll_deg']:+.1f}",
            f"{a['pitch_deg']:+.1f}", "yes" if nf(a) else "no", int(a["unreachable_probe_points"])] for a in fail])
    # figure: reachability by base level, normal vs near-fall
    style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_WIDTH, 2.2), sharey=True)
    for ax, key, ttl in zip(axes, ("reach_hw", "reach_c3_soft"), ("hardware joint range", "C3 IK soft band")):
        for f, mk in zip(FAMS, ("o", "s", "^")):
            ys = []
            for B in BLEV:
                sel = [r[key] for r in rows if r["stratum"] == "random" and r["state"] == "normal"
                       and r["trajectory"] == f and abs(r["base_level"] - B) < 1e-9]
                ys.append(100 * np.mean(sel))
            ax.plot(BLEV, ys, marker=mk, color=style.ORDINAL_BLUE[FAMS.index(f)], label=f"{f} (normal)")
        sel = [r for r in rows if r["state"] == "near-fall"]
        if sel:
            ax.axhline(100 * np.mean([r[key] for r in sel]), color=style.SERIES["C3"], ls="--", lw=1,
                       label=f"near-fall states, all (n={len(sel)})")
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("Base-motion bound")
        ax.set_xticks(BLEV)
    axes[0].set_ylabel("Targets reachable [%]")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    style.save(fig, FIGD, "A1_reachability")


# ======================================================================== A2
def a2():
    MD.append("## A2 Commanded EE speed and IK / actuator saturation\n")
    geo = rd(os.path.join(ROOT, "validation", "trajectory_geometry_check.csv"))
    table(["trajectory", "level [m/s]", "max cmd speed", "mean cmd speed (3–20 s)", "max tang. accel", "off-path max [m]", "ok"],
          [[g["family"], f"{g['speed_level_mps']:.2f}", f"{g['max_speed_mps']:.4f}", f"{g['mean_speed_mps']:.4f}",
            f"{g['max_tangential_accel_mps2']:.3f}", f"{g['max_off_path_m']:.1e}", g["ok"]] for g in geo])
    sat = rd(os.path.join(OUT, "A2_saturation_per_episode.csv"))
    per = {(r["controller"], int(r["base_level"] * 100 + .5), int(r["ee_speed_mps"] * 100 + .5), r["trajectory"],
            int(r["seed"])): r for r in rd(os.path.join(ROOT, "data", "E3v2_per_trial_45000.csv"))}
    for s in sat:
        p = per[(s["controller"], int(BLEV[int(s["bi"])] * 100 + .5), int(VLEV[int(s["vi"])] * 100 + .5),
                 s["family"], int(s["seed"]))]
        s.update({k: p[k] for k in MK}); s["completed"] = p["completed"]
    c3 = [s for s in sat if s["controller"] == "C3"]
    G = np.zeros((5, 5)); G4 = np.zeros((5, 5)); Gany = np.zeros((5, 5))
    for bi in range(5):
        for vi in range(5):
            sel = [s for s in c3 if int(s["bi"]) == bi and int(s["vi"]) == vi]
            G[bi, vi] = 100 * np.nanmean([s["ik_sat_frac"] for s in sel])
            G4[bi, vi] = 100 * np.nanmean([s["ik_sat_frac_from4s"] for s in sel])
            Gany[bi, vi] = 100 * np.mean([s["ik_sat_frac_from4s"] > 0 for s in sel])
    NUM["A2_c3_sat_grid_pct"] = G.round(3).tolist()
    NUM["A2_c3_sat_grid_from4s_pct"] = G4.round(3).tolist()
    MD.append("C3 IK saturation, % of evaluated ticks at the 1.0 rad/s reference cap (pooled over trajectories, "
              "600 episodes per cell); rows = base bound, columns = EE speed. 3–20 s / 4–20 s:\n")
    table(["B \\ v"] + [f"{v:.2f}" for v in VLEV],
          [[f"{B:.2f}"] + [f"{G[i, j]:.2f} / {G4[i, j]:.2f}" for j in range(5)] for i, B in enumerate(BLEV)])
    MD.append("Episodes with any saturated tick after 4 s [% of 600]:\n")
    table(["B \\ v"] + [f"{v:.2f}" for v in VLEV],
          [[f"{B:.2f}"] + [f"{Gany[i, j]:.1f}" for j in range(5)] for i, B in enumerate(BLEV)])
    by_f = {f: 100 * np.nanmean([s["ik_sat_frac"] for s in c3 if s["family"] == f and int(s["vi"]) == 4
                                 and int(s["bi"]) == 4]) for f in FAMS}
    NUM["A2_c3_sat_B060_v020_by_family_pct"] = by_f
    MD.append("At B = 0.60, v = 0.20, by trajectory: " + ", ".join(f"{f} {v:.2f} %" for f, v in by_f.items()) + "\n")
    # correlation with error: within-cell Spearman across seeds (cells with variation)
    rhos = defaultdict(list)
    for bi in range(5):
        for vi in range(5):
            for f in FAMS:
                sel = [s for s in c3 if int(s["bi"]) == bi and int(s["vi"]) == vi and s["family"] == f]
                x = np.array([s["ik_sat_frac"] for s in sel])
                if np.unique(x).size < 5:
                    continue
                for m in ("ee_pos_rmse_m", "ee_ori_rmse_rad"):
                    rhos[m].append(sps.spearmanr(x, [s[m] for s in sel]).statistic)
    corr = {m: {"n_cells": len(v), "median_rho": float(np.median(v)) if v else None,
                "n_positive": int(np.sum(np.array(v) > 0)) if v else 0,
                "sign_test_p": float(sps.binomtest(int(np.sum(np.array(v) > 0)), len(v)).pvalue) if v else None}
            for m, v in rhos.items()}
    NUM["A2_within_cell_spearman"] = corr
    # tick-level: error during saturated vs unsaturated ticks of the same episode
    both = [s for s in c3 if np.isfinite(s.get("ee_pos_err_sat", np.nan)) and np.isfinite(s.get("ee_pos_err_unsat", np.nan))]
    if both:
        dpos = np.array([s["ee_pos_err_sat"] - s["ee_pos_err_unsat"] for s in both])
        dori = np.array([s["ee_ori_err_sat_deg"] - s["ee_ori_err_unsat_deg"] for s in both])
        NUM["A2_tick_level"] = {"n_episodes": len(both), "pos_diff_mean_m": float(dpos.mean()),
                                "pos_diff_ci": stats.bootstrap_ci(dpos), "ori_diff_mean_deg": float(dori.mean()),
                                "ori_diff_ci": stats.bootstrap_ci(dori)}
    MD.append(f"Within-cell Spearman ρ (C3, saturation fraction vs RMSE, across seeds): "
              + "; ".join(f"{m}: median {v['median_rho']:.2f}, {v['n_positive']}/{v['n_cells']} cells positive"
                          for m, v in corr.items() if v["median_rho"] is not None) + "\n")
    if both:
        t = NUM["A2_tick_level"]
        MD.append(f"Tick level (C3 episodes with both saturated and unsaturated ticks, n = {t['n_episodes']}): EE position "
                  f"error is {1000 * t['pos_diff_mean_m']:+.1f} mm [{1000 * t['pos_diff_ci'][0]:+.1f}, "
                  f"{1000 * t['pos_diff_ci'][1]:+.1f}] and orientation error {t['ori_diff_mean_deg']:+.2f}° "
                  f"[{t['ori_diff_ci'][0]:+.2f}, {t['ori_diff_ci'][1]:+.2f}] higher on saturated ticks.\n")
    # actuator (UAN) saturation, all controllers
    U = {c: np.zeros((5, 5)) for c in CTRL}
    for c in CTRL:
        for bi in range(5):
            for vi in range(5):
                U[c][bi, vi] = 100 * np.nanmean([s["uan_sat_frac"] for s in sat if s["controller"] == c
                                                 and int(s["bi"]) == bi and int(s["vi"]) == vi])
    NUM["A2_uan_sat_grid_pct"] = {c: U[c].round(3).tolist() for c in CTRL}
    MD.append("Arm actuator (UAN) saturation, % of evaluated ticks with ≥1 arm action beyond 0.95, pooled; "
              "corners (B, v) = (0,0) / (0,0.20) / (0.60,0) / (0.60,0.20):\n")
    table(["controller", "(0,0)", "(0,0.20)", "(0.60,0)", "(0.60,0.20)", "grid max"],
          [[c, f"{U[c][0, 0]:.2f}", f"{U[c][0, 4]:.2f}", f"{U[c][4, 0]:.2f}", f"{U[c][4, 4]:.2f}",
            f"{U[c].max():.2f}"] for c in CTRL])
    # figure
    style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(style.PAGE_WIDTH, 2.3))
    for ax, Z, ttl in ((axes[0], G, "C3 IK saturation [% ticks], 3–20 s"),
                       (axes[1], G4, "C3 IK saturation [% ticks], 4–20 s")):
        ax.imshow(Z, cmap=style.SEQUENTIAL, origin="lower", aspect="auto", vmin=0, vmax=max(G.max(), 1e-6))
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=5.5,
                        color="white" if Z[i, j] > 0.6 * max(G.max(), 1e-6) else style.TEXT_PRIMARY)
        ax.set_xticks(range(5), [f"{v:.2f}" for v in VLEV], fontsize=6)
        ax.set_yticks(range(5), [f"{B:.2f}" for B in BLEV], fontsize=6)
        ax.set_xlabel("EE speed [m/s]", fontsize=6.5); ax.set_ylabel("Base bound", fontsize=6.5)
        ax.set_title(ttl, fontsize=7); ax.grid(False)
    ax = axes[2]
    for f, mk in zip(FAMS, ("o", "s", "^")):
        xs, ys = [], []
        for bi in range(5):
            for vi in range(5):
                sel = [s for s in c3 if int(s["bi"]) == bi and int(s["vi"]) == vi and s["family"] == f]
                xs.append(100 * np.nanmean([s["ik_sat_frac"] for s in sel]))
                ys.append(np.nanmean([s["ee_pos_rmse_m"] for s in sel]))
        ax.scatter(xs, ys, marker=mk, s=12, color=style.ORDINAL_BLUE[FAMS.index(f)], label=f)
    ax.set_xlabel("C3 IK saturation [% ticks]", fontsize=6.5)
    ax.set_ylabel("C3 EE pos. RMSE [m]", fontsize=6.5)
    ax.set_title("Cell means (25 cells × 3 paths)", fontsize=7)
    ax.legend(fontsize=6)
    fig.tight_layout()
    style.save(fig, FIGD, "A2_c3_ik_saturation")


# ======================================================================== A3
def _verdicts(path):
    return {(r["trajectory"], round(r["base_level"], 2), round(r["ee_speed_mps"], 2), r["metric"], r["comparison"]): r
            for r in rd(path)}


def a3():
    MD.append("## A3 Evaluation-window sensitivity (3–20 s official vs 4–20 s)\n")
    out = []
    flips_all = []
    for scope in ("pooled", "by_trajectory"):
        p = _verdicts(os.path.join(ROOT, "stats", f"E3v2_paired_{scope}.csv"))
        q = _verdicts(os.path.join(ROOT, "stats", f"E3v2_paired_{scope}_sensitivity_from4s.csv"))
        ch = [k for k in p if p[k]["verdict"] != q[k]["verdict"]]
        rev = [k for k in ch if "no significant" not in p[k]["verdict"] and "no significant" not in q[k]["verdict"]]
        out.append([f"controller comparisons, {scope}", len(p), len(ch), len(rev),
                    Counter(f"{p[k]['verdict']} → {q[k]['verdict']}" for k in ch).most_common(4)])
        flips_all += [(scope, k, p[k], q[k]) for k in ch]
    # rankings
    summ = rd(os.path.join(ROOT, "stats", "E3v2_summary.csv"))
    rk = defaultdict(dict)
    for r in summ:
        rk[(r["window"], r["trajectory"], r["base_level"], r["ee_speed_mps"], r["metric"])][r["controller"]] = r["mean"]
    n_rank, n_rank_ch, n_best_ch = 0, 0, 0
    for key, d in rk.items():
        if key[0] != "3-20 s":
            continue
        d2 = rk[("4-20 s (sensitivity)",) + key[1:]]
        r1, r2 = sorted(d, key=d.get), sorted(d2, key=d2.get)
        n_rank += 1
        n_rank_ch += r1 != r2
        n_best_ch += r1[0] != r2[0]
    out.append(["controller ranking (order of means)", n_rank, n_rank_ch, f"best changed: {n_best_ch}", ""])
    # degradation under 4-20 s, recomputed from the arrays with the same functions
    arr = np.load(os.path.join(ROOT, "data", "E3v2_arrays.npz"))
    deg_ch = {}
    for scope in ("pooled", "by_trajectory"):
        prim = {(r["direction"], r["trajectory"], r["controller"], round(r["base_level"], 2),
                 round(r["ee_speed_mps"], 2), r["metric"]): r
                for r in rd(os.path.join(ROOT, "stats", f"E3v2_degradation_{scope}.csv"))}
        primd = {(r["direction"], r["trajectory"], r["comparison"], round(r["base_level"], 2),
                  round(r["ee_speed_mps"], 2), r["metric"]): r
                 for r in rd(os.path.join(ROOT, "stats", f"E3v2_degradation_C1_vs_rivals_{scope}.csv"))}
        fams = [None] if scope == "pooled" else FAMS
        sens, sensd = [], []
        for f in fams:
            for direction, mets, fixed, sweep in (("base_to_ee", A.EE_M, range(5), range(1, 5)),
                                                   ("ee_to_base", A.BASE_M, range(5), range(1, 5))):
                for m in mets:
                    for fx in fixed:
                        for sw in sweep:
                            bi, vi = (sw, fx) if direction == "base_to_ee" else (fx, sw)
                            bi0, vi0 = (0, fx) if direction == "base_to_ee" else (fx, 0)
                            D = {}
                            for c in CTRL:
                                a = arr[f"{c}__{m}_from4s"]
                                x1 = a[bi, vi].mean(0) if f is None else a[bi, vi, FAMS.index(f)]
                                x0 = a[bi0, vi0].mean(0) if f is None else a[bi0, vi0, FAMS.index(f)]
                                D[c] = x1 - x0
                                r = {"key": (direction, "pooled" if f is None else f, c, round(BLEV[bi], 2),
                                             round(VLEV[vi], 2), m)}
                                r.update(A.paired(D[c])); sens.append(r)
                            for rv in ("C3", "C5"):
                                r = {"key": (direction, "pooled" if f is None else f, f"degradation_{rv}_minus_C1",
                                             round(BLEV[bi], 2), round(VLEV[vi], 2), m)}
                                r.update(A.paired(D[rv] - D["C1"])); sensd.append(r)
        A.finish_family(sens, "x", "error increases", "error decreases")
        A.finish_family(sensd, "x", "C1 degrades less", "C1 degrades more")
        c1 = sum(prim[r["key"]]["verdict"] != r["verdict"] for r in sens)
        c2 = sum(primd[r["key"]]["verdict"] != r["verdict"] for r in sensd)
        out.append([f"degradation vs reference, {scope}", len(sens), c1, "", ""])
        out.append([f"degradation C1 vs rivals, {scope}", len(sensd), c2, "", ""])
        deg_ch[scope] = (c1, c2)
    table(["quantity", "n", "verdict changed", "reversed (sig. → opposite sig.)", "most common changes"], out)
    NUM["A3"] = {"rows": [[str(x) for x in r] for r in out]}
    if flips_all:
        MD.append("Controller-comparison verdict changes (pooled scope shown in full, by-trajectory first 12):\n")
        show = [f for f in flips_all if f[0] == "pooled"] + [f for f in flips_all if f[0] != "pooled"][:12]
        table(["scope", "traj", "B", "v", "metric", "comparison", "3–20 s", "4–20 s"],
              [[s, k[0], k[1], k[2], k[3], k[4], f"{p['verdict']} ({p['mean_diff']:+.4f}, p_H={p['p_holm']:.3f})",
                f"{q['verdict']} ({q['mean_diff']:+.4f}, p_H={q['p_holm']:.3f})"] for s, k, p, q in show])
    # figure: paired mean differences, 3-20 vs 4-20 (pooled)
    p = _verdicts(os.path.join(ROOT, "stats", "E3v2_paired_pooled.csv"))
    q = _verdicts(os.path.join(ROOT, "stats", "E3v2_paired_pooled_sensitivity_from4s.csv"))
    style.apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(style.PAGE_WIDTH, 1.9))
    for ax, m in zip(axes, MK):
        for rv, col in (("C3", style.SERIES["C3"]), ("C5", style.SERIES["C5"])):
            ks = [k for k in p if k[3] == m and k[4] == f"{rv}_minus_C1"]
            x = [p[k]["mean_diff"] for k in ks]; y = [q[k]["mean_diff"] for k in ks]
            ax.scatter(x, y, s=8, color=col, marker=style.MARKERS[rv], label=f"{rv} − C1")
            chg = [i for i, k in enumerate(ks) if p[k]["verdict"] != q[k]["verdict"]]
            ax.scatter([x[i] for i in chg], [y[i] for i in chg], s=40, facecolors="none", edgecolors="k", lw=0.6)
        lim = ax.get_xlim()
        ax.plot(lim, lim, color=style.TEXT_MUTED, lw=0.6)
        ax.axhline(0, color=style.GRID, lw=0.5); ax.axvline(0, color=style.GRID, lw=0.5)
        ax.set_title(A.METRICS[MK.index(m)][1], fontsize=7)
        ax.set_xlabel("3–20 s", fontsize=6.5)
    axes[0].set_ylabel("4–20 s", fontsize=6.5)
    axes[0].legend(fontsize=5.5)
    fig.tight_layout()
    style.save(fig, FIGD, "A3_window_sensitivity")


# ======================================================================== A4
def a4():
    MD.append("## A4 Completion and truncated episodes\n")
    per = rd(os.path.join(ROOT, "data", "E3v2_per_trial_45000.csv"))
    out = []
    for c in CTRL:
        sel = [r for r in per if r["controller"] == c]
        term = [r for r in sel if r["completed"] == 0]
        dur = np.array([r["effective_eval_s"] for r in term])
        out.append([c, pct(len(sel) - len(term), len(sel)), len(term),
                    f"{np.median(dur):.2f} [{np.percentile(dur, 25):.2f}, {np.percentile(dur, 75):.2f}]" if len(dur) else "–",
                    f"{dur.min():.2f}" if len(dur) else "–",
                    dict(Counter(r["termination_reason"] for r in term))])
    table(["controller", "completion (all 15,000)", "terminated", "effective eval. [s] median [IQR] (of 17 s)",
           "min", "reasons"], out)
    NUM["A4_completion"] = {c: sum(r["completed"] for r in per if r["controller"] == c) / 15000 for c in CTRL}
    # grid per controller x family: min completion and where
    out = []
    for c in CTRL:
        for f in FAMS:
            grid = np.zeros((5, 5))
            for bi, B in enumerate(BLEV):
                for vi, v in enumerate(VLEV):
                    sel = [r["completed"] for r in per if r["controller"] == c and r["trajectory"] == f
                           and abs(r["base_level"] - B) < 1e-9 and abs(r["ee_speed_mps"] - v) < 1e-9]
                    grid[bi, vi] = 100 * np.mean(sel)
            out.append([c, f] + [" / ".join(f"{grid[bi, vi]:.1f}" for vi in range(5)) for bi in range(5)])
    MD.append("Completion [%] per controller and trajectory; each cell lists v = 0, 0.05, 0.10, 0.15, 0.20 for one base bound:\n")
    table(["ctrl", "traj"] + [f"B={B:.2f}" for B in BLEV], out)
    # truncation effect 1: RMSE of terminated vs completed episodes in the same cell
    diffs = defaultdict(list)
    cells = defaultdict(list)
    for r in per:
        cells[(r["controller"], r["trajectory"], r["base_level"], r["ee_speed_mps"])].append(r)
    for key, rs in cells.items():
        t = [r for r in rs if r["completed"] == 0]
        k = [r for r in rs if r["completed"] == 1]
        if len(t) >= 3 and len(k) >= 3:
            for m in MK:
                diffs[(key[0], m)].append(np.mean([r[m] for r in t]) / np.mean([r[m] for r in k]) - 1)
    out = [[c, m, len(diffs[(c, m)]), f"{100 * np.median(diffs[(c, m)]):+.1f} %"] for c in CTRL for m in MK if diffs[(c, m)]]
    MD.append("RMSE of terminated relative to completed episodes in the same cell (cells with ≥3 of each):\n")
    table(["controller", "metric", "cells", "median (terminated / completed − 1)"], out)
    # truncation effect 2: completed-pairs-only controller comparisons (pooled scope, per family seeds)
    arr = np.load(os.path.join(ROOT, "data", "E3v2_arrays.npz"))
    prim = _verdicts(os.path.join(ROOT, "stats", "E3v2_paired_pooled.csv"))
    rows = []
    for bi in range(5):
        for vi in range(5):
            for m in MK:
                for rv in ("C3", "C5"):
                    a1, a0 = arr[f"{rv}__{m}"][bi, vi], arr[f"C1__{m}"][bi, vi]
                    ok = (arr[f"{rv}__completed"][bi, vi] * arr["C1__completed"][bi, vi]) > 0
                    d = np.where(ok, a1 - a0, np.nan)  # (fam, seed)
                    seeds_ok = ok.all(axis=0)
                    dd = d[:, seeds_ok].mean(axis=0)
                    r = {"key": ("pooled", round(BLEV[bi], 2), round(VLEV[vi], 2), m, f"{rv}_minus_C1"),
                         "n_seeds_all_complete": int(seeds_ok.sum())}
                    r.update(A.paired(dd)); rows.append(r)
    A.finish_family(rows, "completed-only", "C1 better", "C1 worse")
    ch = [r for r in rows if r["verdict"] != prim[r["key"]]["verdict"]]
    NUM["A4_completed_only_changes"] = len(ch)
    MD.append(f"Completed-pairs-only sensitivity (pooled; seeds where both controllers completed all three "
              f"trajectories; conditional on survival, hence selection-biased in favour of the less robust "
              f"controller): {len(ch)} of {len(rows)} verdicts differ from the primary analysis.\n")
    if ch:
        table(["B", "v", "metric", "comparison", "primary", "completed-only", "seeds used"],
              [[r["key"][1], r["key"][2], r["key"][3], r["key"][4], prim[r["key"]]["verdict"], r["verdict"],
                r["n_seeds_all_complete"]] for r in ch])
    # figure: effective duration histogram of terminated episodes
    style.apply_style()
    fig, ax = plt.subplots(figsize=(style.COL_WIDTH, 1.9))
    bins = np.linspace(0, 17, 35)
    for c in CTRL:
        dur = [r["effective_eval_s"] for r in per if r["controller"] == c and r["completed"] == 0]
        if dur:
            ax.hist(dur, bins=bins, histtype="step", color=style.SERIES[c], lw=1.2, label=f"{c} (n={len(dur)})")
    ax.set_xlabel("Effective evaluation time of terminated episodes [s]")
    ax.set_ylabel("Episodes")
    ax.legend(fontsize=6)
    fig.tight_layout()
    style.save(fig, FIGD, "A4_truncated_durations")


# ======================================================================== A5
def a5():
    MD.append("## A5 Cross-task interference and statistical robustness\n")
    pool = rd(os.path.join(ROOT, "stats", "E3v2_paired_pooled.csv"))
    fam = _verdicts(os.path.join(ROOT, "stats", "E3v2_paired_by_trajectory.csv"))
    out = []
    for m in MK:
        for rv in ("C3", "C5"):
            sel = [r for r in pool if r["metric"] == m and r["comparison"] == f"{rv}_minus_C1"]
            cnt = Counter(r["verdict"] for r in sel)
            prac = [r for r in sel if r["significant_holm"] and abs(r["mean_diff"]) >= PRACTICAL_REL * abs(r["C1_mean"])]
            small = [r for r in sel if r["significant_holm"] and abs(r["mean_diff"]) < PRACTICAL_REL * abs(r["C1_mean"])]
            # consistency across families for significant pooled verdicts
            cons = Counter()
            for r in sel:
                if not r["significant_holm"]:
                    continue
                fv = [fam[(f, round(r["base_level"], 2), round(r["ee_speed_mps"], 2), m, r["comparison"])]["verdict"]
                      for f in FAMS]
                cons["all 3 agree" if all(v == r["verdict"] for v in fv) else
                     ("opposite in ≥1" if any(v not in (r["verdict"], "no significant difference") for v in fv)
                      else "n.s. in ≥1")] += 1
            dz = [r["mean_diff"] / r["sd_diff"] for r in sel if r["sd_diff"] > 0]
            out.append([A.METRICS[MK.index(m)][1], f"{rv} vs C1", cnt.get("C1 better", 0), cnt.get("C1 worse", 0),
                        cnt.get("no significant difference", 0), len(prac), len(small), dict(cons),
                        f"{np.median(np.abs(dz)):.2f}"])
    table(["metric", "comparison", "C1 better", "C1 worse", "n.s.", "sig. & ≥10 %", "sig. but <10 %",
           "family consistency of sig. cells", "median |dz|"], out)
    NUM["A5_pooled"] = [[str(x) for x in r] for r in out]
    # degradation
    deg = rd(os.path.join(ROOT, "stats", "E3v2_degradation_pooled.csv"))
    dd = rd(os.path.join(ROOT, "stats", "E3v2_degradation_C1_vs_rivals_pooled.csv"))
    out = []
    for direction in ("base_to_ee", "ee_to_base"):
        for c in CTRL:
            sel = [r for r in deg if r["direction"] == direction and r["controller"] == c]
            cnt = Counter(r["verdict"] for r in sel)
            top = [r for r in sel if (r["base_level"] == 0.6 if direction == "base_to_ee" else r["ee_speed_mps"] == 0.2)]
            out.append([direction, c, len(sel), cnt.get("error increases", 0), cnt.get("error decreases", 0),
                        cnt.get("no significant difference", 0),
                        "; ".join(f"{r['metric']}@{'v' if direction == 'base_to_ee' else 'B'}="
                                  f"{r['ee_speed_mps'] if direction == 'base_to_ee' else r['base_level']:.2f}: "
                                  f"{r['mean_diff']:+.4f}" for r in top if (r["ee_speed_mps"] in (0.0, 0.2)
                                                                            if direction == "base_to_ee"
                                                                            else r["base_level"] in (0.0, 0.6)))])
    MD.append("Degradation relative to the matched reference (pooled; Holm within 240 tests):\n")
    table(["direction", "controller", "n", "sig. increase", "sig. decrease", "n.s.", "top-intensity Δ (extremes of the other factor)"], out)
    out = []
    for direction in ("base_to_ee", "ee_to_base"):
        for rv in ("C3", "C5"):
            sel = [r for r in dd if r["direction"] == direction and r["comparison"] == f"degradation_{rv}_minus_C1"]
            cnt = Counter(r["verdict"] for r in sel)
            out.append([direction, f"{rv} vs C1", len(sel), cnt.get("C1 degrades less", 0),
                        cnt.get("C1 degrades more", 0), cnt.get("no significant difference", 0)])
    MD.append("Does C1 degrade less than the rivals? (pooled; Holm within 160 tests):\n")
    table(["direction", "comparison", "n", "C1 degrades less", "C1 degrades more", "n.s."], out)
    NUM["A5_degradation"] = out
    cur = rd(os.path.join(ROOT, "stats", "E3v2_curvature_pooled.csv"))
    out = []
    for direction in ("base_to_ee", "ee_to_base"):
        for c in CTRL:
            sel = [r for r in cur if r["direction"] == direction and r["controller"] == c]
            cnt = Counter(r["verdict"] for r in sel)
            out.append([direction, c, len(sel), cnt.get("accelerating (convex)", 0),
                        cnt.get("saturating (concave)", 0), cnt.get("no significant difference", 0)])
    MD.append("Shape of the intensity response (per-seed quadratic coefficient; pooled; Holm within 60):\n")
    table(["direction", "controller", "sweeps", "accelerating", "saturating", "no sig. curvature"], out)


def main():
    os.makedirs(FIGD, exist_ok=True)
    for f in (a1, a2, a3, a4, a5):
        f()
        print(f"{f.__name__} done")
    common.write_text(os.path.join(OUT, "audit_tables.md"), "\n".join(MD) + "\n")
    common.write_json(os.path.join(OUT, "audit_numbers.json"), NUM)


if __name__ == "__main__":
    main()
