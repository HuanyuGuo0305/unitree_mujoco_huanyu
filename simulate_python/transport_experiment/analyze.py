"""Statistical analysis of a saved E4 run (no simulation).

    cd unitree_mujoco/simulate_python
    python3 -m transport_experiment.analyze transport_experiment/results/<run_dir> [--force]

Writes <run_dir>/analysis/:
    summary_table.{csv,md,tex}   retention, success-conditional displacement and base tracking
    failure_counts.csv           outcome counts per method
    paired_vs_full.csv           McNemar (exact, Holm) + paired-bootstrap differences vs Full
    matched_all_success.csv      sensitivity: scenarios where all four methods succeeded
    stratified_retention.csv     retention by puck friction / mass tercile (diagnostic)
    diagnostics.csv              failed-episode displacement, failure times, detector cross-check
    fig_e4_transport.{pdf,png}   compact 1x3 figure
    ANALYSIS.md                  human-readable report of all of the above

Population rules
  * Retention denominator = ALL paired scenarios (early failures stay in).
  * Continuous metrics in the main table are CONDITIONAL ON SUCCESS (full 14.4-s active
    window) and always printed with their n and next to retention.
  * Paired continuous comparisons use only scenarios where BOTH methods succeeded
    (matched pairs); unmatched survivor means are never differenced.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_PY = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, SIM_PY)

from transport_experiment.controllers import LABELS, METHODS  # noqa: E402

SHORT = {"no_residual": "No Residual", "pos_only": "EE Pos. Only", "ori_only": "EE Ori. Only", "full": "Full (Ours)"}
OUTCOMES = ("success", "puck_drop", "base_failure", "other_failure")


# ----------------------------------------------------------------------------- stats
def wilson(k, n, conf=0.95):
    if n == 0:
        return float("nan"), float("nan")
    z = sps.norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def boot_ci(x, rng, B, conf, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(B, len(x)))
    s = stat(x[idx], axis=1)
    a = (1 - conf) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


# ----------------------------------------------------------------------------- io
def load_rows(run_dir):
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "episodes.jsonl"))]
    by = {m: {} for m in METHODS}
    for r in rows:
        by[r["method"]][r["scenario_id"]] = r
    sids = sorted(set.intersection(*[set(v) for v in by.values() if v]))
    for m in METHODS:
        if by[m] and set(by[m]) != set(sids):
            raise ValueError(f"{m}: scenario set differs from the other methods (unpaired data)")
    return rows, by, sids


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


# ----------------------------------------------------------------------------- main
def analyze(run_dir, force=False):
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    st = meta["config"].get("statistics", {})
    B = int(st.get("bootstrap_resamples", 10000))
    conf = float(st.get("confidence", 0.95))
    seed = int(st.get("bootstrap_seed", 0))
    out = os.path.join(run_dir, "analysis")
    if os.path.exists(out) and not force:
        raise FileExistsError(f"{out} exists; pass --force to regenerate derived analysis files")
    os.makedirs(out, exist_ok=True)
    rows, by, sids = load_rows(run_dir)
    methods = [m for m in METHODS if by[m]]
    N = len(sids)

    def arr(m, key, only_success=False):
        v = [by[m][s][key] for s in sids if (by[m][s]["success"] or not only_success)]
        return np.asarray(v, float)

    succ = {m: np.array([by[m][s]["success"] for s in sids], bool) for m in methods}

    # ---------------- main table ----------------
    table, fail_rows = [], []
    for m in methods:
        rng = np.random.default_rng(seed)
        k = int(succ[m].sum())
        lo, hi = wilson(k, N, conf)
        disp = arr(m, "mean_puck_xy_displacement", True) * 100.0
        bxy = arr(m, "base_xy_rmse", True)
        yaw = arr(m, "base_yaw_rmse", True)
        d_lo, d_hi = boot_ci(disp, rng, B, conf)
        b_lo, b_hi = boot_ci(bxy, rng, B, conf)
        y_lo, y_hi = boot_ci(yaw, rng, B, conf)
        table.append({
            "method": m, "label": LABELS[m], "n_scenarios": N, "n_success": k,
            "retention_pct": 100.0 * k / N, "retention_ci_lo": 100.0 * lo, "retention_ci_hi": 100.0 * hi,
            "disp_cm_mean": float(disp.mean()) if len(disp) else float("nan"),
            "disp_cm_sd": float(disp.std(ddof=1)) if len(disp) > 1 else float("nan"),
            "disp_cm_ci_lo": d_lo, "disp_cm_ci_hi": d_hi,
            "disp_cm_median": float(np.median(disp)) if len(disp) else float("nan"),
            "base_xy_rmse_mean": float(bxy.mean()) if len(bxy) else float("nan"),
            "base_xy_rmse_sd": float(bxy.std(ddof=1)) if len(bxy) > 1 else float("nan"),
            "base_xy_rmse_ci_lo": b_lo, "base_xy_rmse_ci_hi": b_hi,
            "yaw_rmse_mean": float(yaw.mean()) if len(yaw) else float("nan"),
            "yaw_rmse_sd": float(yaw.std(ddof=1)) if len(yaw) > 1 else float("nan"),
            "yaw_rmse_ci_lo": y_lo, "yaw_rmse_ci_hi": y_hi,
        })
        reasons = [by[m][s]["failure_reason"] for s in sids]
        fail_rows.append({"method": m, "label": LABELS[m], "n": N, **{o: reasons.count(o) for o in OUTCOMES}})
    write_csv(os.path.join(out, "summary_table.csv"), table)
    write_csv(os.path.join(out, "failure_counts.csv"), fail_rows)

    # ---------------- paired comparisons vs Full ----------------
    paired = []
    if "full" in methods:
        others = [m for m in methods if m != "full"]
        pv = []
        for m in others:
            rng = np.random.default_rng(seed + 1)
            f, o = succ["full"], succ[m]
            b = int(np.sum(f & ~o))         # Full succeeds, other fails
            c = int(np.sum(~f & o))         # other succeeds, Full fails
            p = float(sps.binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0
            pv.append(p)
            d = o.astype(float) - f.astype(float)
            r_lo, r_hi = boot_ci(d, rng, B, conf)
            both = f & o
            rec = {"comparison": f"{m} - full", "n_scenarios": N, "full_only_success": b, "other_only_success": c,
                   "mcnemar_exact_p": p, "retention_diff_pct": 100 * d.mean(),
                   "retention_diff_ci_lo": 100 * r_lo, "retention_diff_ci_hi": 100 * r_hi, "n_both_success": int(both.sum())}
            for key, scale, name in (("mean_puck_xy_displacement", 100.0, "disp_cm"),
                                     ("base_xy_rmse", 1.0, "base_xy_rmse"), ("base_yaw_rmse", 1.0, "yaw_rmse")):
                a_ = np.array([by[m][s][key] for s in sids]) * scale
                f_ = np.array([by["full"][s][key] for s in sids]) * scale
                diff = (a_ - f_)[both]
                lo, hi = boot_ci(diff, rng, B, conf)
                rec[f"{name}_diff_mean"] = float(diff.mean()) if len(diff) else float("nan")
                rec[f"{name}_diff_ci_lo"], rec[f"{name}_diff_ci_hi"] = lo, hi
                rec[f"{name}_wilcoxon_p"] = float(sps.wilcoxon(diff).pvalue) if len(diff) > 10 and np.any(diff != 0) else float("nan")
            paired.append(rec)
        for rec, padj in zip(paired, holm(np.array(pv))):
            rec["mcnemar_p_holm"] = float(padj)
        write_csv(os.path.join(out, "paired_vs_full.csv"), paired)

    # ---------------- sensitivity: scenarios where all methods succeeded ----------------
    all_ok = np.all(np.stack([succ[m] for m in methods]), axis=0)
    matched = []
    for m in methods:
        rng = np.random.default_rng(seed + 2)
        disp = np.array([by[m][s]["mean_puck_xy_displacement"] for s in sids])[all_ok] * 100
        bxy = np.array([by[m][s]["base_xy_rmse"] for s in sids])[all_ok]
        matched.append({"method": m, "n_matched": int(all_ok.sum()),
                        "disp_cm_mean": float(disp.mean()) if len(disp) else float("nan"),
                        "disp_cm_ci": boot_ci(disp, rng, B, conf),
                        "base_xy_rmse_mean": float(bxy.mean()) if len(bxy) else float("nan"),
                        "base_xy_rmse_ci": boot_ci(bxy, rng, B, conf)})
    write_csv(os.path.join(out, "matched_all_success.csv"), matched)

    # ---------------- stratified retention (diagnostic) ----------------
    mu = np.array([by[methods[0]][s]["puck_mu"] for s in sids])
    mass = np.array([by[methods[0]][s]["puck_mass"] for s in sids])
    strat = []
    for var, vals in (("puck_mu", mu), ("puck_mass", mass)):
        edges = np.quantile(vals, [0, 1 / 3, 2 / 3, 1])
        for i in range(3):
            sel = (vals >= edges[i]) & (vals <= edges[i + 1] if i == 2 else vals < edges[i + 1])
            rec = {"variable": var, "bin": i, "lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(sel.sum())}
            for m in methods:
                rec[f"retention_{m}"] = 100.0 * float(succ[m][sel].mean())
            strat.append(rec)
    write_csv(os.path.join(out, "stratified_retention.csv"), strat)

    # ---------------- diagnostics ----------------
    diag = []
    for m in methods:
        fails = [by[m][s] for s in sids if not by[m][s]["success"]]
        ft = np.array([r["failure_time"] - 0.5 for r in fails if r["failure_time"] is not None])
        fd = np.array([r["mean_puck_xy_displacement"] for r in fails]) * 100
        allr = [by[m][s] for s in sids]
        diag.append({
            "method": m, "n_fail": len(fails),
            "fail_active_time_s_median": float(np.median(ft)) if len(ft) else float("nan"),
            "fail_active_time_s_iqr": [float(np.quantile(ft, .25)), float(np.quantile(ft, .75))] if len(ft) else None,
            "fail_mean_disp_cm_observed": float(np.nanmean(fd)) if len(fd) else float("nan"),
            "episodes_with_stator_only_contact_steps": int(sum(r["hl_steps_stator_only_contact"] > 0 for r in allr)),
            "mean_abs_executed_action": float(np.mean([r["mean_abs_executed_action"] for r in allr])),
            "max_puck_tilt_deg_success_median": float(np.median([r["max_puck_tilt_deg"] for r in allr if r["success"]]))
            if any(r["success"] for r in allr) else float("nan"),
            "other_failure_errors": sorted({r["error"] for r in allr if r["error"]}),
        })
    write_csv(os.path.join(out, "diagnostics.csv"), diag)

    # ---------------- text tables ----------------
    md = ["| Method | Retention (%) ↑ | Puck displacement (cm) ↓ | Base XY RMSE (m/s) ↓ | Yaw-rate RMSE (rad/s) ↓ |",
          "|---|---:|---:|---:|---:|"]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Dynamic nonprehensile transport in MuJoCo (" + str(N) + r" paired scenarios per method). "
           r"Retention with 95\% Wilson CI over all scenarios; displacement and base-velocity RMSE are means $\pm$ s.d. "
           r"over \emph{successful} episodes (n in parentheses).}",
           r"\label{tab:e4_transport}", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{lccc}", r"\toprule",
           r"Method & Retention (\%) $\uparrow$ & Puck disp. (cm) $\downarrow$ & Base XY RMSE (m/s) $\downarrow$ \\", r"\midrule"]
    for t in table:
        ret = f"{t['retention_pct']:.1f} [{t['retention_ci_lo']:.1f}, {t['retention_ci_hi']:.1f}]"
        dsp = f"{fmt(t['disp_cm_mean'])} ± {fmt(t['disp_cm_sd'])} (n={t['n_success']})"
        bx = f"{fmt(t['base_xy_rmse_mean'], 3)} ± {fmt(t['base_xy_rmse_sd'], 3)}"
        yw = f"{fmt(t['yaw_rmse_mean'], 3)} ± {fmt(t['yaw_rmse_sd'], 3)}"
        md.append(f"| {t['label']} | {ret} | {dsp} | {bx} | {yw} |")
        name = t["label"].replace("(Ours)", r"(Ours)")
        if t["method"] == "full":
            name = r"\textbf{" + name + "}"
        tex.append(f"{name} & {t['retention_pct']:.1f} [{t['retention_ci_lo']:.1f}, {t['retention_ci_hi']:.1f}] & "
                   f"{fmt(t['disp_cm_mean'])} $\\pm$ {fmt(t['disp_cm_sd'])} ({t['n_success']}) & "
                   f"{fmt(t['base_xy_rmse_mean'], 3)} $\\pm$ {fmt(t['base_xy_rmse_sd'], 3)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "summary_table.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(out, "summary_table.tex"), "w").write("\n".join(tex) + "\n")

    report = [f"# E4 analysis — {os.path.basename(os.path.abspath(run_dir))}", "",
              f"{N} paired scenarios × {len(methods)} methods; bootstrap B={B}, {int(conf * 100)}% CIs.", "",
              "## Main table (continuous metrics conditional on success)", "", *md, "",
              "## Outcomes", "", "| Method | " + " | ".join(OUTCOMES) + " |", "|---|" + "---:|" * len(OUTCOMES)]
    for f in fail_rows:
        report.append(f"| {f['label']} | " + " | ".join(str(f[o]) for o in OUTCOMES) + " |")
    if paired:
        report += ["", "## Paired comparisons vs Full Policy (other − Full)", "",
                   "| Comparison | Full-only / other-only successes | McNemar p (Holm) | Δretention (pp) [CI] | n both succ. | Δdisp (cm) [CI] | ΔXY RMSE (m/s) [CI] | Δyaw RMSE (rad/s) [CI] |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for p in paired:
            report.append(
                f"| {p['comparison']} | {p['full_only_success']} / {p['other_only_success']} | {p['mcnemar_exact_p']:.2g} ({p['mcnemar_p_holm']:.2g}) | "
                f"{p['retention_diff_pct']:+.1f} [{p['retention_diff_ci_lo']:+.1f}, {p['retention_diff_ci_hi']:+.1f}] | {p['n_both_success']} | "
                f"{fmt(p['disp_cm_diff_mean'])} [{fmt(p['disp_cm_diff_ci_lo'])}, {fmt(p['disp_cm_diff_ci_hi'])}] | "
                f"{fmt(p['base_xy_rmse_diff_mean'], 4)} [{fmt(p['base_xy_rmse_diff_ci_lo'], 4)}, {fmt(p['base_xy_rmse_diff_ci_hi'], 4)}] | "
                f"{fmt(p['yaw_rmse_diff_mean'], 4)} [{fmt(p['yaw_rmse_diff_ci_lo'], 4)}, {fmt(p['yaw_rmse_diff_ci_hi'], 4)}] |")
    report += ["", f"## Sensitivity: scenarios where all methods succeeded (n={int(all_ok.sum())})", "",
               "| Method | disp (cm) [CI] | XY RMSE (m/s) [CI] |", "|---|---:|---:|"]
    for r in matched:
        report.append(f"| {LABELS[r['method']]} | {fmt(r['disp_cm_mean'])} [{fmt(r['disp_cm_ci'][0])}, {fmt(r['disp_cm_ci'][1])}] | "
                      f"{fmt(r['base_xy_rmse_mean'], 3)} [{fmt(r['base_xy_rmse_ci'][0], 3)}, {fmt(r['base_xy_rmse_ci'][1], 3)}] |")
    report += ["", "## Retention (%) by puck-parameter tercile (diagnostic)", "",
               "| Variable | Range | n | " + " | ".join(SHORT[m] for m in methods) + " |", "|---|---|---:|" + "---:|" * len(methods)]
    for r in strat:
        report.append(f"| {r['variable']} | [{r['lo']:.2f}, {r['hi']:.2f}] | {r['n']} | " + " | ".join(f"{r[f'retention_{m}']:.1f}" for m in methods) + " |")
    report += ["", "## Diagnostics", "", "| Method | failures | median time-to-failure (s, active) | obs. disp. of failed (cm) | episodes w/ stator-only contact | mean abs executed action |",
               "|---|---:|---:|---:|---:|---:|"]
    for dg in diag:
        report.append(f"| {LABELS[dg['method']]} | {dg['n_fail']} | {fmt(dg['fail_active_time_s_median'])} | {fmt(dg['fail_mean_disp_cm_observed'])} | "
                      f"{dg['episodes_with_stator_only_contact_steps']} | {fmt(dg['mean_abs_executed_action'], 3)} |")
    open(os.path.join(out, "ANALYSIS.md"), "w").write("\n".join(report) + "\n")

    make_figure(out, table, by, sids, methods)
    print("\n".join(report))
    return out


def make_figure(out, table, by, sids, methods):
    sys.path.insert(0, SIM_PY)
    from benchmark.plots import style
    import matplotlib.pyplot as plt

    style.apply_style()
    # Fixed categorical slots (dataviz reference palette, never cycled): ours = slot 1.
    color = {"full": style.SERIES["C1"], "no_residual": style.SERIES["C3"],
             "pos_only": style.SERIES["C5"], "ori_only": style.SERIES["C2"]}
    x = np.arange(len(methods))
    tick = {"no_residual": "No\nResidual", "pos_only": "EE Pos.\nOnly", "ori_only": "EE Ori.\nOnly", "full": "Full\n(Ours)"}
    labels = [tick[m] for m in methods]
    fig, axes = plt.subplots(1, 3, figsize=(style.PAGE_WIDTH, 2.05), gridspec_kw={"wspace": 0.45})

    ax = axes[0]
    for i, (m, t) in enumerate(zip(methods, table)):
        ax.bar(i, t["retention_pct"], width=0.62, color=color[m], edgecolor=style.SURFACE, linewidth=1.0, zorder=2)
        ax.errorbar(i, t["retention_pct"], yerr=[[t["retention_pct"] - t["retention_ci_lo"]], [t["retention_ci_hi"] - t["retention_pct"]]],
                    color=style.TEXT_PRIMARY, capsize=2, linewidth=0.8, zorder=3)
        ax.text(i, t["retention_ci_hi"] + 2, f"{t['retention_pct']:.1f}", ha="center", va="bottom",
                fontsize=6.5, color=style.TEXT_SECONDARY)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Payload retention (%)")
    ax.set_title("(a) Retention (all scenarios)", loc="left")

    def box(ax, key, scale, ylabel, title):
        data = [np.array([by[m][s][key] for s in sids if by[m][s]["success"]]) * scale for m in methods]
        bp = ax.boxplot(data, positions=x, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops={"color": style.TEXT_PRIMARY, "linewidth": 1.0},
                        whiskerprops={"color": style.TEXT_SECONDARY, "linewidth": 0.7},
                        capprops={"color": style.TEXT_SECONDARY, "linewidth": 0.7})
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(color[m]); patch.set_alpha(0.85); patch.set_edgecolor(style.SURFACE)
        for i, d in enumerate(data):
            if len(d):
                ax.plot(i, d.mean(), marker="D", markersize=3.2, color=style.SURFACE,
                        markeredgecolor=style.TEXT_PRIMARY, markeredgewidth=0.7, zorder=4)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.1)
        for i, d in enumerate(data):
            ax.text(i, top * 1.04, f"n={len(d)}", ha="center", va="center", fontsize=6, color=style.TEXT_MUTED)

    box(axes[1], "mean_puck_xy_displacement", 100.0, "Mean puck displacement (cm)", "(b) Displacement (successes)")
    box(axes[2], "base_xy_rmse", 1.0, "Base XY vel. RMSE (m/s)", "(c) Base tracking (successes)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.3, linespacing=0.95)
        ax.set_xlim(-0.6, len(methods) - 0.4)
    style.save(fig, out, "fig_e4_transport")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    analyze(a.run_dir, a.force)


if __name__ == "__main__":
    main()
