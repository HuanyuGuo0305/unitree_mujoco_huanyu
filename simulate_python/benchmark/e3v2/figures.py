#!/usr/bin/env python3
"""E3 v2 publication figures (IEEE two-column width, PDF + PNG). Editable.

Reads only results/E3_v2/stats/*.csv. Main-figure slices are fixed in advance
to the TOP level of the held factor (EE 0.20 m/s for base sweeps, base 0.60 for
EE sweeps), as in E3 v1; every other slice is written as a supplementary figure
with the same layout. Columns: the three trajectories, then the pooled value
(per-seed mean over trajectories); y axes are shared within a row so the
trajectory families are compared on one scale.

Run:  python3 -m benchmark.e3v2.figures
"""
from __future__ import annotations

import csv
import os

import numpy as np

from benchmark.round3 import common  # noqa: F401
from benchmark.plots import style  # noqa: E402
from benchmark.plots.style import plt  # noqa: E402

ROOT = os.path.join(common.BENCH, "results", "E3_v2")
ST = os.path.join(ROOT, "stats")
FIG = os.path.join(ROOT, "figures")
SUPP = os.path.join(FIG, "supplementary")
CTRL = ["C1", "C3", "C5"]
COLS = ["line", "circle", "semicircle", "pooled"]
COL_TITLE = {"line": "Line", "circle": "Circle", "semicircle": "Semicircle", "pooled": "Pooled (3 paths)"}
LABEL = {"base_xy_rmse_mps": ("Base XY RMSE", "m/s"), "yaw_rmse_radps": ("Yaw-rate RMSE", "rad/s"),
         "ee_pos_rmse_m": ("EE pos. RMSE", "m"), "ee_ori_rmse_rad": ("EE ori. RMSE", "rad")}
CTRL_LABEL = {"C1": "C1 Unified RL (ours)", "C3": "C3 RL loco. + arm IK", "C5": "C5 Cooperative dual-PPO"}
BLEV = [0.0, 0.15, 0.30, 0.45, 0.60]
VLEV = [0.0, 0.05, 0.10, 0.15, 0.20]


def read(name):
    with open(os.path.join(ST, name)) as h:
        rows = list(csv.DictReader(h))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def _legend(fig):
    handles = [plt.Line2D([], [], **{**style.series_kwargs(c), "label": CTRL_LABEL[c]}) for c in CTRL]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=7)


def _grid(nrows):
    fig, axes = plt.subplots(nrows, 4, figsize=(style.PAGE_WIDTH, 1.62 * nrows + 0.45),
                             sharey="row", sharex=True, squeeze=False)
    return fig, axes


def line_panels(rows, metrics, xkey, fixkey, fixval, xlabel, title, name, out_dir, ylabel_prefix="",
                zero_line=False):
    fig, axes = _grid(len(metrics))
    xs = BLEV if xkey == "base_level" else VLEV
    for i, m in enumerate(metrics):
        for j, col in enumerate(COLS):
            ax = axes[i, j]
            for c in CTRL:
                sel = {round(r[xkey], 3): r for r in rows if r["controller"] == c and r["trajectory"] == col
                       and r["metric"] == m and abs(r[fixkey] - fixval) < 1e-9}
                xv = [x for x in xs if round(x, 3) in sel]
                if not xv:
                    continue
                mean = [sel[round(x, 3)]["mean"] for x in xv]
                lo = [sel[round(x, 3)]["ci95_lower"] for x in xv]
                hi = [sel[round(x, 3)]["ci95_upper"] for x in xv]
                style.plot_with_ci(ax, xv, mean, lo, hi, c, direct_label=False)
            if zero_line:
                ax.axhline(0.0, color=style.TEXT_MUTED, lw=0.6, zorder=0)
            if i == 0:
                ax.set_title(COL_TITLE[col], fontsize=8)
            if j == 0:
                nm, un = LABEL[m]
                ax.set_ylabel(f"{ylabel_prefix}{nm}\n[{un}]")
            if i == len(metrics) - 1:
                ax.set_xlabel(xlabel)
            ax.set_xticks(xs)
            ax.tick_params(axis="x", labelsize=6)
    _legend(fig)
    fig.text(0.5, -0.01, title, ha="center", fontsize=7, color=style.TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93), h_pad=0.6, w_pad=0.4)
    return style.save(fig, out_dir, name)


def heatmaps(rows, name):
    """Pooled paired difference rival - C1 over the 5x5 grid (positive = C1 lower error)."""
    metrics = list(LABEL)
    fig, axes = plt.subplots(2, 4, figsize=(style.PAGE_WIDTH, 3.9))
    for j, m in enumerate(metrics):
        cells = {(r["comparison"], round(r["base_level"], 2), round(r["ee_speed_mps"], 2)): r
                 for r in rows if r["metric"] == m}
        vmax = max(abs(r["mean_diff"]) for r in cells.values()) or 1.0
        for i, rv in enumerate(["C3", "C5"]):
            ax = axes[i, j]
            Z = np.array([[cells[(f"{rv}_minus_C1", round(B, 2), round(v, 2))]["mean_diff"] for v in VLEV]
                          for B in BLEV])
            S = np.array([[cells[(f"{rv}_minus_C1", round(B, 2), round(v, 2))]["significant_holm"] for v in VLEV]
                          for B in BLEV])
            ax.imshow(Z, cmap=style.DIVERGING, vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
            for a in range(5):
                for b in range(5):
                    val = Z[a, b]
                    txt = f"{val:+.3f}" if abs(val) < 1 else f"{val:+.2f}"
                    dark = abs(val) > 0.55 * vmax
                    ax.text(b, a, txt + ("" if S[a, b] else "\nn.s."), ha="center", va="center",
                            fontsize=4.6, color="white" if dark else style.TEXT_PRIMARY,
                            fontweight="bold" if S[a, b] else "normal")
            ax.set_xticks(range(5), [f"{v:.2f}" for v in VLEV], fontsize=5.5)
            ax.set_yticks(range(5), [f"{B:.2f}" for B in BLEV], fontsize=5.5)
            ax.grid(False)
            if i == 0:
                nm, un = LABEL[m]
                ax.set_title(f"{nm} [{un}]", fontsize=7)
            if i == 1:
                ax.set_xlabel("EE speed [m/s]", fontsize=6.5)
            if j == 0:
                ax.set_ylabel(f"{rv} − C1\nbase bound", fontsize=6.5)
    fig.text(0.5, 0.005, "Pooled paired difference (rival − C1), 200 seeds. Blue: C1 lower error; red: rival "
             "lower error. Bold: Holm-adjusted p < 0.05 (family of 200 pooled tests); n.s.: not significant "
             "(not evidence of equivalence).", ha="center", fontsize=5.6, color=style.TEXT_SECONDARY, wrap=True)
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.8, w_pad=0.5)
    return style.save(fig, FIG, name)


def completion(summary, name):
    fig, axes = plt.subplots(1, 3, figsize=(style.PAGE_WIDTH, 2.25))
    for j, c in enumerate(CTRL):
        cells = {(round(r["base_level"], 2), round(r["ee_speed_mps"], 2)): r for r in summary
                 if r["controller"] == c and r["trajectory"] == "pooled" and r["metric"] == "ee_pos_rmse_m"
                 and r["window"] == "3-20 s"}
        Z = np.array([[100 * cells[(round(B, 2), round(v, 2))]["completion_rate"] for v in VLEV] for B in BLEV])
        ax = axes[j]
        ax.imshow(Z, cmap=style.SEQUENTIAL.reversed(), vmin=50, vmax=100, origin="lower", aspect="auto")
        for a in range(5):
            for b in range(5):
                ax.text(b, a, f"{Z[a, b]:.1f}", ha="center", va="center", fontsize=5.5,
                        color="white" if Z[a, b] < 80 else style.TEXT_PRIMARY)
        ax.set_xticks(range(5), [f"{v:.2f}" for v in VLEV], fontsize=6)
        ax.set_yticks(range(5), [f"{B:.2f}" for B in BLEV], fontsize=6)
        ax.grid(False)
        ax.set_title(CTRL_LABEL[c], fontsize=7)
        ax.set_xlabel("EE speed [m/s]", fontsize=6.5)
        if j == 0:
            ax.set_ylabel("Base bound", fontsize=6.5)
    fig.text(0.5, 0.0, "Completion rate [%] (600 episodes per cell: 3 trajectories × 200 seeds). Colour scale "
             "50–100 %.", ha="center", fontsize=6, color=style.TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return style.save(fig, FIG, name)


def main():
    style.apply_style()
    summ = [r for r in read("E3v2_summary.csv") if r["window"] == "3-20 s"]
    deg = read("E3v2_degradation_pooled.csv") + read("E3v2_degradation_by_trajectory.csv")
    for r in deg:
        r["mean"], r["ci95_lower"], r["ci95_upper"] = r["mean_diff"], r["ci95_lower"], r["ci95_upper"]
    # zero-degradation reference points at the reference level
    refs = []
    for r in deg:
        z = dict(r)
        if r["direction"] == "base_to_ee":
            z["base_level"] = 0.0
        else:
            z["ee_speed_mps"] = 0.0
        z["mean"] = z["ci95_lower"] = z["ci95_upper"] = 0.0
        refs.append(z)
    deg_b2e = [r for r in deg + refs if r["direction"] == "base_to_ee"]
    deg_e2b = [r for r in deg + refs if r["direction"] == "ee_to_base"]

    for v in VLEV:
        main_ = abs(v - 0.20) < 1e-9
        out = FIG if main_ else SUPP
        line_panels(summ, ["ee_pos_rmse_m", "ee_ori_rmse_rad"], "base_level", "ee_speed_mps", v,
                    "Base bound B",
                    f"EE tracking vs base-motion intensity, EE speed {v:.2f} m/s (mean, 95% CI, 200 seeds)",
                    f"E3v2_fig1_ee_vs_base_ee{v:.2f}".replace(".", "p"), out)
        line_panels(deg_b2e, ["ee_pos_rmse_m", "ee_ori_rmse_rad"], "base_level", "ee_speed_mps", v,
                    "Base bound B",
                    f"Base→EE degradation: EE error minus the stationary-base error at EE speed {v:.2f} m/s "
                    "(paired, mean, 95% CI)",
                    f"E3v2_fig3_base_to_ee_degradation_ee{v:.2f}".replace(".", "p"), out, "Δ ", zero_line=True)
    for B in BLEV:
        main_ = abs(B - 0.60) < 1e-9
        out = FIG if main_ else SUPP
        line_panels(summ, ["base_xy_rmse_mps", "yaw_rmse_radps"], "ee_speed_mps", "base_level", B,
                    "EE speed [m/s]",
                    f"Base tracking vs EE-motion intensity, base bound {B:.2f} (mean, 95% CI, 200 seeds)",
                    f"E3v2_fig2_base_vs_ee_base{B:.2f}".replace(".", "p"), out)
        line_panels(deg_e2b, ["base_xy_rmse_mps", "yaw_rmse_radps"], "ee_speed_mps", "base_level", B,
                    "EE speed [m/s]",
                    f"EE→Base degradation: base error minus the fixed-EE-target error at base bound {B:.2f} "
                    "(paired, mean, 95% CI)",
                    f"E3v2_fig4_ee_to_base_degradation_base{B:.2f}".replace(".", "p"), out, "Δ ", zero_line=True)
    heatmaps(read("E3v2_paired_pooled.csv"), "E3v2_fig5_paired_difference_heatmaps")
    completion(read("E3v2_summary.csv"), "E3v2_fig6_completion_rates")


if __name__ == "__main__":
    main()
