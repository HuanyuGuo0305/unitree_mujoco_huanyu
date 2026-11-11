#!/usr/bin/env python3
"""Round-3 publication figures.

Design decisions, all from the project's visualisation rules:

* 2-D panels with 95% confidence bands are the primary form. The coordination
  effects here are of order 0.005-0.03 m against absolute errors near 0.15 m,
  and only a 2-D panel with its interval shows whether a gap is real. No 3-D
  surfaces: at this effect size they compress the interesting variation into a
  nearly flat sheet.
* Axes are never truncated to magnify a small difference. Error axes include
  zero wherever zero is attainable, and every degradation panel draws the zero
  line prominently, because the SIGN of a degradation is the finding.
* Identity is never carried by colour alone: every series has its own marker
  and dash pattern, a legend is always present, and the last point carries a
  direct text label.
* The paired-advantage heatmaps use the validated zero-centred diverging scale
  (blue = C1 ahead, red = rival ahead, neutral grey at exactly zero) with
  symmetric limits, and the signed number is printed in every cell, so polarity
  never depends on hue.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from benchmark.plots import style
from benchmark.round3 import common

LEVELS = [0.0, 0.5, 1.0]
CELLS = [[f"S{i}{j}" for j in range(3)] for i in range(3)]


def _load_e3(out_dir):
    summary = common.read_csv(os.path.join(out_dir, "summary_50seeds.csv"))
    deg = common.read_csv(os.path.join(out_dir, "summary_degradation.csv"))
    paired = common.read_csv(os.path.join(out_dir, "paired_comparisons.csv"))
    pooled = [r for r in summary if r["trajectory_family"] == "all(pooled)"]
    deg_pooled = [r for r in deg if r["trajectory_family"] == "all(pooled)"]
    paired_pooled = [r for r in paired if r["trajectory_family"] == "all(pooled)"]
    return pooled, deg_pooled, paired_pooled


def _series(rows, baseline, metric, sweep, fixed):
    """(x, mean, lo, hi) sorted by the swept alpha, at a fixed other alpha."""
    other = "alpha_ee" if sweep == "alpha_base" else "alpha_base"
    sel = [
        r for r in rows
        if r["baseline"] == baseline and r["metric"] == metric
        and abs(float(r[other]) - fixed) < 1e-9
    ]
    sel.sort(key=lambda r: float(r[sweep]))
    x = np.asarray([float(r[sweep]) for r in sel])
    m = np.asarray([float(r["mean"]) for r in sel])
    lo = np.asarray([float(r["ci95_lower"]) for r in sel])
    hi = np.asarray([float(r["ci95_upper"]) for r in sel])
    return x, m, lo, hi


def _deg_series(rows, baseline, kind, sweep, fixed):
    other = "alpha_ee" if sweep == "alpha_base" else "alpha_base"
    sel = [
        r for r in rows
        if r["baseline"] == baseline and r["degradation_type"] == kind
        and abs(float(r[other]) - fixed) < 1e-9
    ]
    sel.sort(key=lambda r: float(r[sweep]))
    x = np.asarray([float(r[sweep]) for r in sel])
    m = np.asarray([float(r["mean"]) for r in sel])
    lo = np.asarray([float(r["ci95_lower"]) for r in sel])
    hi = np.asarray([float(r["ci95_upper"]) for r in sel])
    return x, m, lo, hi


def _finish(ax, include_zero=True):
    if include_zero:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0.0), hi)


def _legend(fig, axes, ncol=3):
    handles, labels = axes[0].get_legend_handles_labels()
    seen, h, l = set(), [], []
    for handle, label in zip(handles, labels):
        if label in seen or label.startswith("_"):
            continue
        seen.add(label)
        h.append(handle)
        l.append(label)
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.04),
               frameon=False)


# ==========================================================================
# Figure E3-A: absolute tracking RMSE
# ==========================================================================
def e3_absolute_tracking(out_dir, fig_dir, fixed_ae=1.0, fixed_ab=1.0,
                         name="E3_A_absolute_tracking"):
    import matplotlib.pyplot as plt

    pooled, _deg, _paired = _load_e3(out_dir)
    style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_WIDTH, 2.5))

    for b in ("C1", "C3", "C5"):
        x, m, lo, hi = _series(pooled, b, "ee_position_rmse_m", "alpha_base", fixed_ae)
        style.plot_with_ci(axes[0], x, m, lo, hi, b)
    axes[0].set_xlabel(r"Base motion intensity $\alpha_{base}$")
    axes[0].set_ylabel("EE position RMSE (m)")
    axes[0].set_title(rf"(a) EE tracking, $\alpha_{{ee}} = {fixed_ae:g}$", fontsize=8)

    for b in ("C1", "C3", "C5"):
        x, m, lo, hi = _series(pooled, b, "base_vxy_rmse_mps", "alpha_ee", fixed_ab)
        style.plot_with_ci(axes[1], x, m, lo, hi, b)
    axes[1].set_xlabel(r"EE motion intensity $\alpha_{ee}$")
    axes[1].set_ylabel("Base XY velocity RMSE (m/s)")
    axes[1].set_title(rf"(b) Base tracking, $\alpha_{{base}} = {fixed_ab:g}$", fontsize=8)

    for ax in axes:
        ax.set_xticks(LEVELS)
        _finish(ax)
    axes[0].legend(loc="best")
    fig.tight_layout()
    return style.save(fig, fig_dir, name)


def e3_absolute_tracking_all_slices(out_dir, fig_dir,
                                    name="E3_A_absolute_tracking_all_slices"):
    """Every slice of the matrix, so the speed-dependent crossover is visible
    rather than represented by the one slice that favours C1."""
    import matplotlib.pyplot as plt

    pooled, _deg, _paired = _load_e3(out_dir)
    style.apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(style.PAGE_WIDTH, 4.2), sharex="row")

    for col, ae in enumerate(LEVELS):
        ax = axes[0][col]
        for b in ("C1", "C3", "C5"):
            x, m, lo, hi = _series(pooled, b, "ee_position_rmse_m", "alpha_base", ae)
            # No direct end labels on this row: at alpha_ee = 1 the C1 and C5
            # curves nearly coincide and the two labels would overprint. The
            # always-present legend carries identity instead.
            style.plot_with_ci(ax, x, m, lo, hi, b, direct_label=False)
        ax.set_title(rf"$\alpha_{{ee}} = {ae:g}$", fontsize=8)
        ax.set_xlabel(r"$\alpha_{base}$")
        ax.set_xticks(LEVELS)
        _finish(ax)
    axes[0][0].set_ylabel("EE position RMSE (m)")

    for col, ab in enumerate(LEVELS):
        ax = axes[1][col]
        for b in ("C1", "C3", "C5"):
            x, m, lo, hi = _series(pooled, b, "base_vxy_rmse_mps", "alpha_ee", ab)
            style.plot_with_ci(ax, x, m, lo, hi, b, direct_label=(col == 2))
        ax.set_title(rf"$\alpha_{{base}} = {ab:g}$", fontsize=8)
        ax.set_xlabel(r"$\alpha_{ee}$")
        ax.set_xticks(LEVELS)
        _finish(ax)
    axes[1][0].set_ylabel("Base XY vel. RMSE (m/s)")

    # Shared y limits within each row: the comparison across slices is the point.
    for row in axes:
        lo = min(ax.get_ylim()[0] for ax in row)
        hi = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(lo, hi)
    _legend(fig, [axes[0][0]])
    fig.tight_layout()
    return style.save(fig, fig_dir, name)


# ==========================================================================
# Figure E3-B: directional error degradation
# ==========================================================================
def e3_error_degradation(out_dir, fig_dir, fixed_ae=1.0, fixed_ab=1.0,
                         name="E3_B_error_degradation"):
    import matplotlib.pyplot as plt

    _pooled, deg, _paired = _load_e3(out_dir)
    style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_WIDTH, 2.5))

    for b in ("C1", "C3", "C5"):
        x, m, lo, hi = _deg_series(deg, b, "locomotion_induced_ee", "alpha_base", fixed_ae)
        style.plot_with_ci(axes[0], x, m, lo, hi, b)
    axes[0].set_xlabel(r"Base motion intensity $\alpha_{base}$")
    axes[0].set_ylabel(r"$\Delta E_p$: locomotion-induced" "\n" "EE error increase (m)")
    axes[0].set_title(rf"(a) base $\rightarrow$ EE, $\alpha_{{ee}} = {fixed_ae:g}$",
                      fontsize=8)

    for b in ("C1", "C3", "C5"):
        x, m, lo, hi = _deg_series(deg, b, "manipulation_induced_base", "alpha_ee", fixed_ab)
        style.plot_with_ci(axes[1], x, m, lo, hi, b)
    axes[1].set_xlabel(r"EE motion intensity $\alpha_{ee}$")
    axes[1].set_ylabel(r"$\Delta E_v$: manipulation-induced" "\n" "base error increase (m/s)")
    axes[1].set_title(rf"(b) EE $\rightarrow$ base, $\alpha_{{base}} = {fixed_ab:g}$",
                      fontsize=8)

    for ax in axes:
        ax.set_xticks(LEVELS)
        # Zero is the reference the whole panel is about: draw it, keep it in
        # view, and never clip a negative degradation away.
        ax.axhline(0.0, color=style.TEXT_SECONDARY, linewidth=0.8, zorder=1)
        lo, hi = ax.get_ylim()
        pad = 0.08 * (hi - lo)
        ax.set_ylim(min(lo, -pad), max(hi, pad))
        ax.annotate("no degradation", xy=(0.02, 0.0), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points",
                    color=style.TEXT_MUTED, fontsize=6, va="bottom")
        # The swept intensity 0 is the REFERENCE condition: the degradation is
        # zero there by definition, not by measurement. Marked as a hollow
        # point so the empty region is explained and not mistaken for missing
        # data, and visibly distinguished from the measured points.
        ax.plot([0.0], [0.0], marker="o", markersize=4.5, markerfacecolor="none",
                markeredgecolor=style.TEXT_MUTED, markeredgewidth=0.9,
                linestyle="none", zorder=2,
                label="_reference")
        ax.annotate("reference\n($\\Delta \\equiv 0$)", xy=(0.0, 0.0),
                    xytext=(3, -14), textcoords="offset points",
                    color=style.TEXT_MUTED, fontsize=6)
    axes[0].legend(loc="best")
    fig.tight_layout()
    return style.save(fig, fig_dir, name)


# ==========================================================================
# Figure E3-C: paired advantage heatmaps
# ==========================================================================
HEAT_SPEC = [
    ("ee_position_rmse_m", "abs", r"$A_p$: EE position RMSE (m)", 4),
    ("base_vxy_rmse_mps", "abs", r"$A_v$: base XY vel. RMSE (m/s)", 4),
    ("locomotion_induced_ee", "deg", r"$A_{\Delta p}$: loco.$\rightarrow$EE degr. (m)", 4),
    ("manipulation_induced_base", "deg", r"$A_{\Delta v}$: manip.$\rightarrow$base degr. (m/s)", 4),
]


def e3_paired_heatmaps(out_dir, fig_dir, rival="C3", name=None):
    import matplotlib.pyplot as plt

    _pooled, _deg, paired = _load_e3(out_dir)
    name = name or f"E3_C_paired_advantage_{rival}_vs_C1"
    abs_rows = {(r["condition"], r["metric"]): r
                for r in paired if r["rival"] == rival and not r["degradation_type"]}
    deg_rows = {(r["condition"], r["degradation_type"]): r
                for r in paired if r["rival"] == rival and r["degradation_type"]}

    style.apply_style()
    # 2x2 rather than 1x4: at 1x4 each cell is ~0.5 in wide and the signed
    # four-decimal labels collide. The number in the cell has to stay legible,
    # because it, not the hue, is what carries the sign.
    fig, axes = plt.subplots(2, 2, figsize=(style.PAGE_WIDTH, 4.1))
    axes = axes.ravel()
    for ax, (key, kind, title, digits) in zip(axes, HEAT_SPEC):
        grid = np.full((3, 3), np.nan)
        sig = np.zeros((3, 3), dtype=bool)
        for i in range(3):
            for j in range(3):
                r = (abs_rows if kind == "abs" else deg_rows).get((CELLS[i][j], key))
                if r is None:
                    continue
                grid[i, j] = float(r["mean_difference"])
                sig[i, j] = bool(int(r["significant"]))
        # Symmetric limits about zero: the neutral midpoint of the diverging
        # scale must land on exactly 0, or the colour would imply a sign the
        # data does not have.
        vmax = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
        im = ax.imshow(grid, cmap=style.DIVERGING, vmin=-vmax, vmax=vmax,
                       origin="upper")
        ax.set_title(title, fontsize=7.5, pad=6)
        ax.set_xticks(range(3), [f"{v:g}" for v in LEVELS])
        ax.set_yticks(range(3), [f"{v:g}" for v in LEVELS])
        ax.set_xlabel(r"EE motion intensity $\alpha_{ee}$", fontsize=7)
        ax.set_ylabel(r"Base intensity $\alpha_{base}$", fontsize=7)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                if not np.isfinite(grid[i, j]):
                    # The degradation reference cell: the difference is zero by
                    # definition there, so it is marked, not left blank.
                    ax.text(j, i, "ref.", ha="center", va="center",
                            fontsize=6, color=style.TEXT_MUTED, style="italic")
                    continue
                shade = abs(grid[i, j]) / vmax if vmax else 0.0
                colour = "#ffffff" if shade > 0.55 else style.TEXT_PRIMARY
                ax.text(j, i - 0.13, f"{grid[i,j]:+.{digits}f}", ha="center",
                        va="center", fontsize=6.2, color=colour)
                ax.text(j, i + 0.25, "*" if sig[i, j] else "n.s.", ha="center",
                        va="center", fontsize=5.8, color=colour)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_visible(False)
    fig.suptitle(
        f"Paired advantage ({rival} - C1) over 50 seeds, pooled over trajectory "
        f"families.\nBlue = C1 lower error, red = {rival} lower error, "
        f"grey = exactly 0.  * = 95% CI excludes zero.",
        fontsize=7.5, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return style.save(fig, fig_dir, name)


# ==========================================================================
# Selected illustrative cases
# ==========================================================================
def _episode_series(path):
    cmd = common.read_csv(os.path.join(path, "commands.csv"))
    st = common.read_csv(os.path.join(path, "states.csv"))
    keep = [i for i, r in enumerate(cmd) if int(r["evaluated"]) == 1]

    def c(rows, key):
        return np.asarray([float(rows[i][key]) for i in keep], dtype=np.float64)

    return {
        "t": c(cmd, "t"),
        "vxy_cmd": np.hypot(c(cmd, "vx_cmd"), c(cmd, "vy_cmd")),
        "vxy_act": np.hypot(c(st, "vx_act"), c(st, "vy_act")),
        "vxy_err": np.hypot(c(st, "vx_act") - c(cmd, "vx_cmd"),
                            c(st, "vy_act") - c(cmd, "vy_cmd")),
        "wz_cmd": c(cmd, "wz_cmd"), "wz_act": c(st, "wz_act"),
        "ee_pos_err": c(st, "ee_pos_err"), "ee_ori_err": c(st, "ee_ori_err_deg"),
        "ee_cmd_y": c(cmd, "ee_cmd_y_plb"), "ee_act_y": c(st, "ee_act_y_plb"),
        "ee_cmd_z": c(cmd, "ee_cmd_z_plb"), "ee_act_z": c(st, "ee_act_z_plb"),
    }


def _leg_power(path):
    """250 Hz leg mechanical power proxy, validated against metrics.json."""
    from benchmark.round3.verify_power import reconstruct
    import json

    got = reconstruct(path)
    if got is None:
        return None
    with open(os.path.join(path, "metrics.json"), encoding="utf-8") as h:
        want = float(json.load(h)["leg_power"])
    if abs(got["leg_power"] - want) > max(1e-9, 1e-3 * abs(want)):
        raise ValueError(f"leg power reconstruction mismatch for {path}")
    with np.load(os.path.join(path, "physics.npz")) as handle:
        data = handle["data"].astype(np.float64)
    n_total = (data.shape[1] - 1) // 4
    n_leg = n_total - 6
    qvel = data[:, 1 + n_total: 1 + 2 * n_total]
    tau = data[:, 1 + 3 * n_total: 1 + 4 * n_total]
    series = np.sum(np.abs(tau[:, :n_leg] * qvel[:, :n_leg]), axis=1)
    return data[:, 0], series


def _smooth(x, window):
    if window <= 1 or x.size < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def _row_header(axes_row, text):
    axes_row[len(axes_row) // 2].set_title(text, fontsize=7, pad=14)


def selected_cases_figure(experiment, out_dir, fig_dir):
    import matplotlib.pyplot as plt

    rows = common.read_csv(os.path.join(out_dir, "selected_3_cases.csv"))
    if not rows:
        return []
    baselines = {"E1": ["C1", "C3", "C5"], "E2": ["C1", "C2"],
                 "E3": ["C1", "C3", "C5"]}[experiment]
    style.apply_style()
    ncol = 3
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(style.PAGE_WIDTH, 1.85 * len(rows) + 0.9),
                             squeeze=False)

    for r, row in enumerate(rows):
        paths = {}
        for b in baselines:
            seed = int(row["selected_seed"])
            paths[b] = common.episode_dir(experiment, b, row["condition"], seed)
        series = {b: _episode_series(p) for b, p in paths.items()}
        t0 = min(s["t"][0] for s in series.values())

        if experiment == "E2":
            panels = [("vxy_err", "Base XY vel. error (m/s)"),
                      ("legpower", "Leg power proxy (W)"),
                      ("ee_pos_err", "EE position error (m)")]
        else:
            panels = [("vxy_err", "Base XY vel. error (m/s)"),
                      ("ee_pos_err", "EE position error (m)"),
                      ("ee_track", "EE path, PLB $y$--$z$ (m)")]

        for c, (kind, ylabel) in enumerate(panels):
            ax = axes[r][c]
            if kind == "legpower":
                for b in baselines:
                    out = _leg_power(paths[b])
                    if out is None:
                        continue
                    t, p = out
                    kw = style.series_kwargs(b)
                    kw["marker"] = None
                    kw["linewidth"] = 0.9
                    ax.plot(t - t0, _smooth(p, 50), **kw)
                ax.set_xlabel("time (s)" if r == len(rows) - 1 else "")
            elif kind == "ee_track":
                ax.plot(series[baselines[0]]["ee_cmd_y"],
                        series[baselines[0]]["ee_cmd_z"],
                        color=style.TEXT_PRIMARY, linestyle="--", linewidth=0.9,
                        label="command")
                for b in baselines:
                    kw = style.series_kwargs(b)
                    kw["marker"] = None
                    kw["linewidth"] = 0.9
                    ax.plot(series[b]["ee_act_y"], series[b]["ee_act_z"], **kw)
                ax.set_xlabel("PLB $y$ (m)")
                ax.set_aspect("equal", adjustable="datalim")
            else:
                for b in baselines:
                    kw = style.series_kwargs(b)
                    kw["marker"] = None
                    kw["linewidth"] = 0.9
                    ax.plot(series[b]["t"] - t0, series[b][kind], **kw)
                ax.set_xlabel("time (s)" if r == len(rows) - 1 else "")
                ax.set_ylim(0, None)
            ax.set_ylabel(ylabel, fontsize=7)

        tag = (f"{row['case'].replace('_',' ')}  |  seed {row['selected_seed']}, "
               f"{row['condition']}  |  rank {row['rank_among_eligible']}"
               f"/{row['n_eligible']} eligible")
        _row_header(axes[r], tag)

    _legend(fig, [axes[0][2] if experiment != "E2" else axes[0][0]],
            ncol=len(baselines) + 1)
    fig.suptitle(
        f"{experiment}: SELECTED ILLUSTRATIVE TRIALS -- single episodes, not averages. "
        f"The 50-seed tables carry the population claims.",
        fontsize=7, y=1.005,
    )
    # h_pad clears the row header of the panel below from the x-axis label of
    # the panel above; without it the two overprint.
    fig.tight_layout(h_pad=2.4)
    return style.save(fig, fig_dir, f"{experiment}_selected_cases")


# ==========================================================================
# E1 / E2 official summary figures
# ==========================================================================
def e1_summary_figure(out_dir, fig_dir, name="E1_official_50seeds"):
    import matplotlib.pyplot as plt

    summary = common.read_csv(os.path.join(out_dir, "summary_50seeds.csv"))
    idx = {(r["condition"], r["baseline"], r["metric"]): r for r in summary}
    conds = ["T1", "T2", "T3", "T4"]
    metrics = [("base_vxy_rmse_mps", "Base XY vel. RMSE (m/s)"),
               ("base_yaw_rmse_radps", "Base yaw-rate RMSE (rad/s)"),
               ("ee_position_rmse_m", "EE position RMSE (m)"),
               ("ee_orientation_mae_deg", "EE orientation MAE (deg)")]
    style.apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(style.PAGE_WIDTH, 2.3))
    width = 0.26
    for ax, (metric, ylabel) in zip(axes, metrics):
        for k, b in enumerate(("C1", "C3", "C5")):
            x = np.arange(len(conds)) + (k - 1) * width
            m = np.asarray([float(idx[(c, b, metric)]["mean"]) for c in conds])
            lo = np.asarray([float(idx[(c, b, metric)]["ci95_lower"]) for c in conds])
            hi = np.asarray([float(idx[(c, b, metric)]["ci95_upper"]) for c in conds])
            ax.bar(x, m, width * 0.92, color=style.SERIES[b],
                   label=style.LABELS[b], edgecolor=style.SURFACE, linewidth=0.6)
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none",
                        ecolor=style.TEXT_SECONDARY, elinewidth=0.7, capsize=1.6)
        ax.set_xticks(range(len(conds)), conds)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_ylim(0, None)
    _legend(fig, [axes[0]])
    fig.suptitle("E1, 50 official seeds, mean with 95% bootstrap CI "
                 "(base resampling 2--3 s, EE targets replaced every 3 s)",
                 fontsize=7, y=1.02)
    fig.tight_layout()
    return style.save(fig, fig_dir, name)


def e2_summary_figure(out_dir, fig_dir, name="E2_official_50seeds"):
    import matplotlib.pyplot as plt

    summary = common.read_csv(os.path.join(out_dir, "summary_50seeds.csv"))
    idx = {(r["condition"], r["baseline"], r["metric"]): r for r in summary}
    conds = ["Low", "Medium", "High"]
    xs = np.asarray([0.20, 0.40, 0.60])
    metrics = [("base_vxy_rmse_mps", "Base XY vel. RMSE (m/s)"),
               ("total_power_w", "Total mech. power proxy (W)"),
               ("leg_power_w", "Leg mech. power proxy (W)"),
               ("leg_acc_rms_radps2", r"Leg joint acc. RMS (rad/s$^2$)")]
    style.apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(style.PAGE_WIDTH, 2.3))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for b in ("C1", "C2"):
            m = np.asarray([float(idx[(c, b, metric)]["mean"]) for c in conds])
            lo = np.asarray([float(idx[(c, b, metric)]["ci95_lower"]) for c in conds])
            hi = np.asarray([float(idx[(c, b, metric)]["ci95_upper"]) for c in conds])
            style.plot_with_ci(ax, xs, m, lo, hi, b)
        ax.set_xticks(xs, [f"{v:.2f}" for v in xs])
        ax.set_xlabel("Per-axis command\nenvelope (m/s, rad/s)", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_ylim(0, None)
    _legend(fig, [axes[0]], ncol=2)
    fig.suptitle("E2, 50 paired official seeds, mean with 95% bootstrap CI",
                 fontsize=7, y=1.02)
    fig.tight_layout()
    return style.save(fig, fig_dir, name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=common.OUT)
    parser.add_argument("--only", default=None)
    args = parser.parse_args(argv)
    fig_dir = os.path.join(args.out, "figures")
    e3 = os.path.join(args.out, "E3")
    only = args.only.split(",") if args.only else None

    def want(tag):
        return only is None or tag in only

    if want("E1"):
        e1_summary_figure(os.path.join(args.out, "E1"), fig_dir)
    if want("E2"):
        e2_summary_figure(os.path.join(args.out, "E2"), fig_dir)
    if want("E3"):
        e3_absolute_tracking(e3, fig_dir)
        e3_absolute_tracking(e3, fig_dir, fixed_ae=0.5, fixed_ab=0.5,
                             name="E3_A_absolute_tracking_alpha_medium")
        e3_absolute_tracking_all_slices(e3, fig_dir)
        e3_error_degradation(e3, fig_dir)
        e3_paired_heatmaps(e3, fig_dir, rival="C3")
        e3_paired_heatmaps(e3, fig_dir, rival="C5")
    if want("selected"):
        for experiment in ("E1", "E2", "E3"):
            selected_cases_figure(experiment, os.path.join(args.out, experiment), fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
