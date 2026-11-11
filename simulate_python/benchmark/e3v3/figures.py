#!/usr/bin/env python3
"""E3 v3 publication figures (IEEE two-column, 7.16 in), PDF + PNG.

    python3 -m benchmark.e3v3.figures

Reads results/E3_v3/data/E3v3_arrays.npz and stats/*.csv; writes
results/E3_v3/figures/ (main) and figures/supplementary/.
All means are over 100 paired seeds with 95 % percentile-bootstrap CIs
(10,000 resamples, seed 20261106). Palette, markers and line styles are the
project's validated categorical set (benchmark/plots/style.py), so identity is
never carried by colour alone.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark.round3 import common  # noqa: E402
from benchmark.plots import style  # noqa: E402
from benchmark.e3v3.analyze import boot_ci  # noqa: E402

V3 = os.path.join(common.BENCH, "results", "E3_v3")
FIG = os.path.join(V3, "figures")
SUP = os.path.join(FIG, "supplementary")
CTRL = ("C1", "C3", "C5")
PATS = ("line", "circle")
A = np.load(os.path.join(V3, "data", "E3v3_arrays.npz"))
BLEV = A["base_levels"].astype(float)
TLAB = [str(t) for t in A["ee_levels"]]
FREQ = np.array([0.0, 1 / 6, 1 / 4, 1 / 3, 1 / 2])
TT = A["trace_t"]
LAB = {"base_xy_rmse_mps": "Base XY vel. RMSE [m/s]", "yaw_rmse_radps": "Yaw-rate RMSE [rad/s]",
       "ee_pos_rmse_m": "EE position RMSE [m]", "ee_ori_rmse_rad": "EE orientation RMSE [rad]"}
SIG = {"base_xy": "Base XY vel. error [m/s]", "yaw": "Yaw-rate error [rad/s]",
       "ee_pos": "EE position error [m]", "ee_ori": "EE orientation error [rad]"}


def mci(x):
    """mean and bootstrap CI along the last axis (seeds)."""
    x = np.asarray(x, float)
    m = np.nanmean(x, axis=-1)
    lo, hi = np.empty_like(m), np.empty_like(m)
    for i in np.ndindex(m.shape):
        lo[i], hi[i] = boot_ci(x[i])
    return m, lo, hi


def legend(fig, ax, y=-0.01):
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, y))


def fig_base_vs_ee(b_list, name, out):
    """Base tracking vs EE command-update frequency."""
    style.apply_style()
    cols = [(p, bi) for bi in b_list for p in PATS]
    fig, axes = plt.subplots(2, len(cols), figsize=(style.PAGE_WIDTH, 3.6), squeeze=False)
    for j, (p, bi) in enumerate(cols):
        pi = PATS.index(p)
        for r, m in enumerate(("base_xy_rmse_mps", "yaw_rmse_radps")):
            ax = axes[r, j]
            for ci, c in enumerate(CTRL):
                mm, lo, hi = mci(A[m][ci, pi, bi, :, :])
                style.plot_with_ci(ax, FREQ, mm, lo, hi, c, direct_label=False)
            ax.set_xticks(FREQ, ["Fix", "1/6", "1/4", "1/3", "1/2"])
            ax.relim(); ax.autoscale_view(); ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(f"{p}, B = {BLEV[bi]:.2f}", fontsize=8)
            if r == 1:
                ax.set_xlabel("EE update freq. [Hz]")
            if j == 0:
                ax.set_ylabel(LAB[m])
    legend(fig, axes[0, 0])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return style.save(fig, out, name)


def fig_ee_vs_base(t_list, name, out, prefix=""):
    """EE pose tracking vs base-motion intensity."""
    style.apply_style()
    cols = [(p, ti) for ti in t_list for p in PATS]
    fig, axes = plt.subplots(2, len(cols), figsize=(style.PAGE_WIDTH, 3.6), squeeze=False)
    for j, (p, ti) in enumerate(cols):
        pi = PATS.index(p)
        for r, m in enumerate(("ee_pos_rmse_m", "ee_ori_rmse_rad")):
            ax = axes[r, j]
            for ci, c in enumerate(CTRL):
                mm, lo, hi = mci(A[prefix + m][ci, pi, :, ti, :])
                style.plot_with_ci(ax, BLEV, mm, lo, hi, c, direct_label=False)
            ax.set_xticks(BLEV, [f"{b:g}" for b in BLEV])
            ax.relim(); ax.autoscale_view(); ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(f"{p}, EE {'fixed' if TLAB[ti] == 'Fixed' else 'every ' + TLAB[ti]}",
                             fontsize=8)
            if r == 1:
                ax.set_xlabel("Base bound B")
            if j == 0:
                ax.set_ylabel(LAB[m] if not prefix else {"ee_pos_rmse_m": "EE pos. RMSE,\n5–30 s [m]", "ee_ori_rmse_rad": "EE ori. RMSE,\n5–30 s [rad]"}[m])
    legend(fig, axes[0, 0])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return style.save(fig, out, name)


def fig_event(kind, bi, ti, name, out, order):
    """Event-aligned mean error traces (isolated events), rows = patterns."""
    style.apply_style()
    TR = A[f"trace_{kind}"]  # [ctrl, pat, B, T, seed, signal, time]
    sigs = list(A["trace_signals"])
    fig, axes = plt.subplots(2, 4, figsize=(style.PAGE_WIDTH, 3.5), squeeze=False)
    for pi, p in enumerate(PATS):
        for j, s in enumerate(order):
            ax = axes[pi, j]
            si = sigs.index(s)
            for ci, c in enumerate(CTRL):
                x = TR[ci, pi, bi, ti, :, si, :].astype(float)  # [seed, time]
                ok = np.isfinite(x).all(1)
                m, lo, hi = mci(x[ok].T)
                kw = style.series_kwargs(c)
                ax.fill_between(TT, lo, hi, color=kw["color"], alpha=0.15, linewidth=0)
                ax.plot(TT, m, color=kw["color"], linestyle=kw["linestyle"], label=kw["label"],
                        linewidth=1.2)
            ax.axvline(0, color=style.TEXT_MUTED, linewidth=0.7)
            ax.set_ylim(bottom=0)
            ax.set_xlim(TT[0], TT[-1] + 0.02)
            if pi == 0:
                ax.set_title(SIG[s], fontsize=7.5)
            if pi == 1:
                ax.set_xlabel("Time from switch [s]")
            if j == 0:
                ax.set_ylabel(f"{p}\nerror", fontsize=7.5)
    legend(fig, axes[0, 0])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return style.save(fig, out, name)


def fig_degradation(name, out):
    """Directional degradation (pooled) with absolute reference values."""
    import csv
    D = list(csv.DictReader(open(os.path.join(V3, "stats", "E3v3_degradation.csv"))))
    style.apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(style.PAGE_WIDTH, 3.6), squeeze=False)
    # top: Base->EE at EE Fixed and 2 s; bottom: EE->Base at B = 0 and 0.60
    spec = [(0, "ee_pos_rmse_m", "Fixed"), (0, "ee_pos_rmse_m", "2s"),
            (0, "ee_ori_rmse_rad", "Fixed"), (0, "ee_ori_rmse_rad", "2s"),
            (1, "base_xy_rmse_mps", 0.0), (1, "base_xy_rmse_mps", 0.6),
            (1, "yaw_rmse_radps", 0.0), (1, "yaw_rmse_radps", 0.6)]
    for k, (row, m, fixed) in enumerate(spec):
        ax = axes[row, k % 4]
        for c in CTRL:
            if row == 0:
                rs = [r for r in D if r["metric"] == m and r["pattern"] == "pooled"
                      and r["controller"] == c and r["ee_level"] == fixed]
                rs.sort(key=lambda r: float(r["base_bound"]))
                x = [0.0] + [float(r["base_bound"]) for r in rs]
            else:
                rs = [r for r in D if r["metric"] == m and r["pattern"] == "pooled"
                      and r["controller"] == c and abs(float(r["base_bound"]) - fixed) < 1e-9]
                rs.sort(key=lambda r: TLAB.index(r["ee_level"]))
                x = [0.0] + [FREQ[TLAB.index(r["ee_level"])] for r in rs]
            mm = [0.0] + [float(r["mean"]) for r in rs]
            lo = [0.0] + [float(r["ci95_lo"]) for r in rs]
            hi = [0.0] + [float(r["ci95_hi"]) for r in rs]
            style.plot_with_ci(ax, x, mm, lo, hi, c, direct_label=False)
        ax.axhline(0, color=style.TEXT_PRIMARY, linewidth=0.8)
        unit = LAB[m].split("[")[1].rstrip("]")
        short = {"ee_pos_rmse_m": "EE pos.", "ee_ori_rmse_rad": "EE ori.",
                 "base_xy_rmse_mps": "Base XY", "yaw_rmse_radps": "Yaw rate"}[m]
        if row == 0:
            ax.set_title(f"Base→EE: Δ{short} [{unit}]\nEE {'fixed' if fixed == 'Fixed' else 'every ' + fixed}",
                         fontsize=7)
            ax.set_xlabel("Base bound B")
            ax.set_xticks(BLEV, [f"{b:g}" for b in BLEV])
        else:
            ax.set_title(f"EE→Base: Δ{short} [{unit}]\nB = {fixed:.2f}", fontsize=7)
            ax.set_xlabel("EE update freq. [Hz]")
            ax.set_xticks(FREQ, ["Fix", "1/6", "1/4", "1/3", "1/2"])
    legend(fig, axes[0, 0])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return style.save(fig, out, name)


def fig_reach(name, out):
    style.apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(style.PAGE_WIDTH, 3.6), squeeze=False)
    tis = [1, 2, 3, 4]
    x = FREQ[tis]
    cols = [(p, bi) for bi in (0, 4) for p in PATS]
    for j, (p, bi) in enumerate(cols):
        pi = PATS.index(p)
        for r, m in enumerate(("reach_rate_strict", "rmst_strict_s")):
            ax = axes[r, j]
            for ci, c in enumerate(CTRL):
                mm, lo, hi = mci(A[m][ci, pi, bi][tis])
                style.plot_with_ci(ax, x, mm, lo, hi, c, direct_label=False)
            ax.set_xticks(x, ["1/6", "1/4", "1/3", "1/2"])
            if r == 0:
                ax.set_title(f"{p}, B = {BLEV[bi]:.2f}", fontsize=8)
                ax.set_ylim(0, 1.02)
            else:
                ax.set_ylim(bottom=0)
                ax.set_xlabel("EE update freq. [Hz]")
            if j == 0:
                ax.set_ylabel("Targets reached\n(strict tol.) [-]" if r == 0 else
                              "Restricted mean\nresponse time [s]")
    legend(fig, axes[0, 0])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return style.save(fig, out, name)


def main():
    os.makedirs(SUP, exist_ok=True)
    fig_base_vs_ee((0, 4), "E3v3_fig1_base_tracking_vs_ee_frequency", FIG)
    fig_ee_vs_base((0, 4), "E3v3_fig2_ee_tracking_vs_base_intensity", FIG)
    fig_event("ee_iso", 0, 4, "E3v3_fig3_ee_switch_response_B0_T2s", FIG,
              ("ee_pos", "ee_ori", "base_xy", "yaw"))
    fig_event("base_iso", 4, 0, "E3v3_fig4_base_switch_response_B060_fixedEE", FIG,
              ("base_xy", "yaw", "ee_pos", "ee_ori"))
    fig_degradation("E3v3_fig5_directional_degradation", FIG)
    fig_reach("E3v3_fig6_target_reaching", FIG)
    for bi in (1, 2, 3):
        fig_base_vs_ee((bi,), f"E3v3_figS1_base_vs_ee_B{BLEV[bi]:.2f}".replace(".", "p"), SUP)
    for ti in (1, 2, 3):
        fig_ee_vs_base((ti,), f"E3v3_figS2_ee_vs_base_T{TLAB[ti]}", SUP)
    fig_ee_vs_base((0, 4), "E3v3_figS5_ee_vs_base_window5to30s_posthoc", SUP, prefix="s5_")
    for ti in (1, 2, 3):
        fig_event("ee_iso", 0, ti, f"E3v3_figS3_ee_switch_response_B0_T{TLAB[ti]}", SUP,
                  ("ee_pos", "ee_ori", "base_xy", "yaw"))
    for bi in (1, 2, 3):
        fig_event("base_iso", bi, 0,
                  f"E3v3_figS4_base_switch_response_B{BLEV[bi]:.2f}_fixedEE".replace(".", "p"), SUP,
                  ("base_xy", "yaw", "ee_pos", "ee_ori"))


if __name__ == "__main__":
    main()
