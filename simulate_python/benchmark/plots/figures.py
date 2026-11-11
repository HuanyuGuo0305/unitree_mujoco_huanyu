"""Figure generators for E1-E4.

E3 deliberately produces BOTH formats requested by the experiment design:

* the primary 2-D performance curves (Panel A: locomotion-induced EE
  degradation; Panel B: manipulation-induced base degradation), with 95%
  bootstrap CIs, one figure per trajectory family plus a faceted version, and
* a secondary 3-D view of the full 3x3 matrix, one surface per baseline.

Neither replaces the other; `plot.py` writes both and the README states which
to prefer after looking at the data.
"""

from __future__ import annotations

import os

import numpy as np

from benchmark.core import stats
from benchmark.plots import style
from benchmark.plots.style import plt


# ==========================================================================
# helpers
# ==========================================================================
def _group(rows, keys):
    out: dict = {}
    for row in rows:
        out.setdefault(tuple(row[k] for k in keys), []).append(row)
    return out


def _matches(row, key, value):
    """Compare a row field to a target, numerically when both are numeric."""
    if key not in row:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return abs(float(row[key]) - float(value)) < 1e-9
        except (TypeError, ValueError):
            return False
    return row[key] == value


def _series(rows, baseline, metric, x_key, x_values, fixed, cfg_stats):
    """Mean and 95% CI of ``metric`` vs ``x_key`` for one baseline."""
    mean, lo, hi, n = [], [], [], []
    for x in x_values:
        items = [
            r for r in rows
            if r["baseline"] == baseline
            and _matches(r, x_key, x)
            and all(_matches(r, k, v) for k, v in fixed.items())
            and metric in r and np.isfinite(r[metric])
        ]
        values = [r[metric] for r in items]
        s = stats.summarize(
            values,
            n_resamples=cfg_stats["bootstrap_resamples"],
            seed=cfg_stats["bootstrap_seed"],
            confidence=cfg_stats["confidence"],
        )
        mean.append(s["mean"])
        lo.append(s["ci_low"])
        hi.append(s["ci_high"])
        n.append(s["n"])
    return np.array(mean), np.array(lo), np.array(hi), np.array(n)


# ==========================================================================
# E1
# ==========================================================================
def e1_table_preview(summary_rows, out_dir: str, baselines=("C1", "C3", "C5")):
    """A rendered preview of the compact four-panel table, as a PDF.

    This is a legibility check for the LaTeX table, not a figure for the paper:
    it shows the four sub-tables at the real IEEE ``table*`` width so the font
    size can be judged before compiling.
    """
    style.apply_style()
    panels = [
        ("T1", "(a) Base still / EE fixed"),
        ("T2", "(b) Base still / EE random"),
        ("T3", "(c) Base moving / EE fixed"),
        ("T4", "(d) Base moving / EE random"),
    ]
    metrics = [
        ("base_vxy_rmse", r"$E_v$ (m/s)", 3),
        ("base_wz_rmse", r"$E_\omega$ (rad/s)", 3),
        ("ee_pos_rmse", r"$E_p$ (m)", 3),
        ("ee_ori_mae_deg", r"$E_R$ (deg)", 1),
    ]
    index = {(r["condition"], r["baseline"]): r for r in summary_rows}

    fig, axes = plt.subplots(2, 2, figsize=(style.PAGE_WIDTH, 2.6))
    for ax, (condition, title) in zip(axes.ravel(), panels):
        ax.axis("off")
        ax.set_title(title, fontsize=8, loc="left", pad=2)
        cells, rows = [], []
        for baseline in baselines:
            rec = index.get((condition, baseline))
            if rec is None:
                continue
            rows.append(style.SHORT[baseline])
            cells.append(
                [f"{rec.get(m + '_mean', float('nan')):.{d}f}" for m, _, d in metrics]
            )
        if not cells:
            continue
        table = ax.table(
            cellText=cells, rowLabels=rows,
            colLabels=[h for _, h, _ in metrics], loc="center", cellLoc="right",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.12)
        for (r, c), cell in table.get_celld().items():
            cell.set_linewidth(0.4)
            cell.set_edgecolor(style.GRID)
            if r == 0:
                cell.set_text_props(color=style.TEXT_SECONDARY)
    fig.suptitle(
        "E1 compact table preview (IEEE table* width) -- legibility check only",
        fontsize=8, color=style.TEXT_SECONDARY, y=1.04,
    )
    return style.save(fig, out_dir, "E1_tracking_table_preview", formats=("pdf", "png"))


def e1_condition_bars(summary_rows, out_dir: str, baselines=("C1", "C3", "C5")):
    """Per-condition EE position RMSE and base XY RMSE with 95% CIs."""
    style.apply_style()
    conditions = ["T1", "T2", "T3", "T4"]
    titles = ["Still/Fixed", "Still/Random", "Moving/Fixed", "Moving/Random"]
    index = {(r["condition"], r["baseline"]): r for r in summary_rows}

    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_WIDTH, 2.2))
    fig.subplots_adjust(wspace=0.32)
    for ax, (metric, label) in zip(
        axes, [("ee_pos_rmse", "EE position RMSE (m)"),
               ("base_vxy_rmse", "Base XY velocity RMSE (m/s)")]
    ):
        x = np.arange(len(conditions), dtype=float)
        width = 0.8 / len(baselines)
        for i, baseline in enumerate(baselines):
            mean, lo, hi = [], [], []
            for condition in conditions:
                rec = index.get((condition, baseline), {})
                mean.append(rec.get(f"{metric}_mean", np.nan))
                lo.append(rec.get(f"{metric}_ci_low", np.nan))
                hi.append(rec.get(f"{metric}_ci_high", np.nan))
            mean = np.asarray(mean, float)
            err = np.vstack([mean - np.asarray(lo, float), np.asarray(hi, float) - mean])
            ax.bar(
                x + (i - (len(baselines) - 1) / 2) * width, mean, width * 0.92,
                color=style.SERIES[baseline], label=style.LABELS[baseline],
                edgecolor=style.SURFACE, linewidth=0.8,
            )
            ax.errorbar(
                x + (i - (len(baselines) - 1) / 2) * width, mean, yerr=np.abs(err),
                fmt="none", ecolor=style.TEXT_SECONDARY, elinewidth=0.7, capsize=1.6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(titles, rotation=12, ha="right")
        ax.set_ylabel(label)
    axes[0].legend(loc="upper left", ncol=1)
    fig.suptitle(
        "E1 tracking error by condition (mean, 95% bootstrap CI over paired seeds)",
        fontsize=8, y=1.03,
    )
    return style.save(fig, out_dir, "E1_tracking_by_condition")


# ==========================================================================
# E2
# ==========================================================================
def e2_intensity_curves(rows, out_dir: str, cfg_stats: dict):
    """C1 vs C2 across the three base-motion intensities."""
    style.apply_style()
    order = ["Low", "Medium", "High"]
    fractions = {"Low": 0.25, "Medium": 0.50, "High": 0.75}
    present = [c for c in order if any(r["condition"] == c for r in rows)]
    x = np.array([fractions[c] for c in present])
    baselines = [b for b in ("C1", "C2") if any(r["baseline"] == b for r in rows)]

    metrics = [
        ("base_vxy_rmse", "Base XY vel. RMSE (m/s)"),
        ("base_wz_rmse", "Base yaw-rate RMSE (rad/s)"),
        ("total_power", "Mech. power proxy (W)"),
        ("leg_acc_rms", r"Leg joint acc. RMS (rad/s$^2$)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(style.PAGE_WIDTH, 2.1))
    fig.subplots_adjust(wspace=0.62)
    for ax, (metric, label) in zip(axes, metrics):
        for baseline in baselines:
            mean, lo, hi, _ = _series(
                rows, baseline, metric, "condition", present, {}, cfg_stats
            )
            style.plot_with_ci(ax, x, mean, lo, hi, baseline, direct_label=False)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.set_xlabel("Base intensity\n(fraction of range)")
        ax.set_ylabel(label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(baselines),
               bbox_to_anchor=(0.5, 1.13))
    return style.save(fig, out_dir, "E2_wheel_actuation_intensity")


# ==========================================================================
# E3
# ==========================================================================
def e3_two_panel(rows, out_dir: str, exp_cfg: dict, cfg_stats: dict,
                 family: str | None = None, baselines=("C1", "C3", "C5"),
                 degradation=False):
    """Panel A: base intensity -> EE error. Panel B: EE speed -> base error."""
    style.apply_style()
    alphas_b = [float(a) for a in exp_cfg["alpha_base"]]
    alphas_e = [float(a) for a in exp_cfg["alpha_ee"]]
    fixed_ae = float(exp_cfg["plots"]["panel_a_fixed_alpha_ee"])
    fixed_ab = float(exp_cfg["plots"]["panel_b_fixed_alpha_base"])
    speed_max = float(exp_cfg["ee"]["speed_max"])
    subset = [r for r in rows if family is None or r["ee_family"] == family]
    baselines = [b for b in baselines if any(r["baseline"] == b for r in subset)]

    if degradation:
        a_metric, a_label = (
            "delta_ee_pos_rmse_from_base",
            r"Locomotion-induced EE error increase" "\n" r"$\Delta E_p$ (m)",
        )
        b_metric, b_label = (
            "delta_base_vxy_rmse_from_ee",
            r"Manipulation-induced base error increase" "\n" r"$\Delta E_v$ (m/s)",
        )
        name = "E3_degradation"
    else:
        a_metric, a_label = "ee_pos_rmse", "EE position RMSE (m)"
        b_metric, b_label = "base_vxy_rmse", "Base XY velocity RMSE (m/s)"
        name = "E3_absolute"

    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_WIDTH, 2.3))
    fig.subplots_adjust(wspace=0.42)
    for baseline in baselines:
        mean, lo, hi, n = _series(
            subset, baseline, a_metric, "alpha_base", alphas_b,
            {"alpha_ee": fixed_ae}, cfg_stats,
        )
        style.plot_with_ci(axes[0], alphas_b, mean, lo, hi, baseline)
    axes[0].set_xlabel(r"Base motion intensity $\alpha_b$")
    axes[0].set_ylabel(a_label)
    axes[0].set_title(
        rf"(a) Locomotion $\rightarrow$ EE  ($\alpha_e={fixed_ae:g}$)", loc="left"
    )
    axes[0].set_xticks(alphas_b)

    for baseline in baselines:
        mean, lo, hi, n = _series(
            subset, baseline, b_metric, "alpha_ee", alphas_e,
            {"alpha_base": fixed_ab}, cfg_stats,
        )
        style.plot_with_ci(
            axes[1], [a * speed_max for a in alphas_e], mean, lo, hi, baseline
        )
    axes[1].set_xlabel("EE motion speed (m/s)")
    axes[1].set_ylabel(b_label)
    axes[1].set_title(
        rf"(b) Manipulation $\rightarrow$ base  ($\alpha_b={fixed_ab:g}$)", loc="left"
    )
    axes[1].set_xticks([a * speed_max for a in alphas_e])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(baselines),
               bbox_to_anchor=(0.5, 1.12))
    suffix = f"_{family}" if family else "_all_families"
    paths = style.save(fig, out_dir, f"{name}{suffix}")

    # The brief names these two files explicitly; emit them for the chosen
    # reference family as well.
    if family == "circle" and not degradation:
        style.apply_style()
        for metric, label, fname, x_key, xs, fixed, xlabel in [
            (a_metric, a_label, "E3_base_to_EE", "alpha_base", alphas_b,
             {"alpha_ee": fixed_ae}, r"Base motion intensity $\alpha_b$"),
            (b_metric, b_label, "E3_EE_to_base", "alpha_ee", alphas_e,
             {"alpha_base": fixed_ab}, "EE motion speed (m/s)"),
        ]:
            f2, ax = plt.subplots(figsize=(style.COL_WIDTH, 2.2))
            for baseline in baselines:
                mean, lo, hi, _ = _series(
                    subset, baseline, metric, x_key, xs, fixed, cfg_stats
                )
                xv = xs if x_key == "alpha_base" else [a * speed_max for a in xs]
                style.plot_with_ci(ax, xv, mean, lo, hi, baseline)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(label)
            ax.set_xticks(xs if x_key == "alpha_base" else [a * speed_max for a in xs])
            ax.legend(loc="upper left")
            paths += style.save(f2, out_dir, fname)
    return paths


def e3_facets(rows, out_dir: str, exp_cfg: dict, cfg_stats: dict,
              baselines=("C1", "C3", "C5"), degradation=False):
    """The same two panels, faceted by trajectory family (3 x 2 grid)."""
    style.apply_style()
    families = [f for f in exp_cfg["ee"]["families"]
                if any(r["ee_family"] == f for r in rows)]
    if not families:
        return []
    alphas_b = [float(a) for a in exp_cfg["alpha_base"]]
    alphas_e = [float(a) for a in exp_cfg["alpha_ee"]]
    fixed_ae = float(exp_cfg["plots"]["panel_a_fixed_alpha_ee"])
    fixed_ab = float(exp_cfg["plots"]["panel_b_fixed_alpha_base"])
    speed_max = float(exp_cfg["ee"]["speed_max"])
    a_metric = "delta_ee_pos_rmse_from_base" if degradation else "ee_pos_rmse"
    b_metric = "delta_base_vxy_rmse_from_ee" if degradation else "base_vxy_rmse"

    fig, axes = plt.subplots(
        2, len(families), figsize=(style.PAGE_WIDTH, 4.0), sharex="row", squeeze=False
    )
    fig.subplots_adjust(hspace=0.62, wspace=0.34)
    for col, family in enumerate(families):
        subset = [r for r in rows if r["ee_family"] == family]
        present = [b for b in baselines if any(r["baseline"] == b for r in subset)]
        for baseline in present:
            mean, lo, hi, _ = _series(
                subset, baseline, a_metric, "alpha_base", alphas_b,
                {"alpha_ee": fixed_ae}, cfg_stats,
            )
            style.plot_with_ci(axes[0][col], alphas_b, mean, lo, hi, baseline,
                               direct_label=False)
            mean, lo, hi, _ = _series(
                subset, baseline, b_metric, "alpha_ee", alphas_e,
                {"alpha_base": fixed_ab}, cfg_stats,
            )
            style.plot_with_ci(
                axes[1][col], [a * speed_max for a in alphas_e], mean, lo, hi,
                baseline, direct_label=False,
            )
        axes[0][col].set_title(family, loc="left")
        axes[0][col].set_xticks(alphas_b)
        axes[1][col].set_xticks([a * speed_max for a in alphas_e])
        axes[1][col].set_xlabel("EE motion speed (m/s)")
        axes[0][col].set_xlabel(r"Base motion intensity $\alpha_b$")
    axes[0][0].set_ylabel(
        (r"$\Delta E_p$ (m)" if degradation else "EE position RMSE (m)")
    )
    axes[1][0].set_ylabel(
        (r"$\Delta E_v$ (m/s)" if degradation else "Base XY vel. RMSE (m/s)")
    )
    # Shared y limits within each row so the facets are comparable.
    for row in axes:
        lims = [ax.get_ylim() for ax in row]
        lo = min(l[0] for l in lims)
        hi = max(l[1] for l in lims)
        for ax in row:
            ax.set_ylim(lo, hi)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    name = "E3_facets_degradation" if degradation else "E3_facets_absolute"
    return style.save(fig, out_dir, name)


def e3_3d_coordination(rows, out_dir: str, exp_cfg: dict, cfg_stats: dict,
                       metric="ee_pos_rmse", family=None,
                       baselines=("C1", "C3", "C5")):
    """Secondary view: one wireframe + measured points per baseline.

    Only nine points are measured per surface, so the surface is drawn as a
    wireframe with the nine measured means plotted as explicit markers. No
    interpolated colour fill is used between them, to avoid implying
    high-resolution measured data that does not exist.
    """
    style.apply_style()
    alphas_b = [float(a) for a in exp_cfg["alpha_base"]]
    alphas_e = [float(a) for a in exp_cfg["alpha_ee"]]
    subset = [r for r in rows if family is None or r["ee_family"] == family]
    present = [b for b in baselines if any(r["baseline"] == b for r in subset)]
    if not present:
        return []

    grids = {}
    for baseline in present:
        Z = np.full((len(alphas_b), len(alphas_e)), np.nan)
        for i, ab in enumerate(alphas_b):
            for j, ae in enumerate(alphas_e):
                values = [
                    r[metric] for r in subset
                    if r["baseline"] == baseline
                    and abs(float(r["alpha_base"]) - ab) < 1e-9
                    and abs(float(r["alpha_ee"]) - ae) < 1e-9
                    and metric in r and np.isfinite(r[metric])
                ]
                if values:
                    Z[i, j] = float(np.mean(values))
        grids[baseline] = Z

    finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    if finite.size == 0:
        return []
    zmin, zmax = float(finite.min()), float(finite.max())
    pad = 0.08 * max(zmax - zmin, 1e-6)

    labels = {
        "ee_pos_rmse": "EE position RMSE (m)",
        "base_vxy_rmse": "Base XY vel. RMSE (m/s)",
        "ee_ori_mae_deg": "EE orientation MAE (deg)",
    }
    panel_tags = "abcdef"
    fig = plt.figure(figsize=(style.PAGE_WIDTH, 2.6))
    fig.subplots_adjust(wspace=0.10, left=0.07, right=0.99)
    A, Eg = np.meshgrid(alphas_b, alphas_e, indexing="ij")
    for k, baseline in enumerate(present):
        ax = fig.add_subplot(1, len(present), k + 1, projection="3d")
        Z = grids[baseline]
        ax.plot_wireframe(
            A, Eg, Z, color=style.SERIES[baseline], linewidth=0.8, rstride=1, cstride=1
        )
        ok = np.isfinite(Z)
        ax.scatter(
            A[ok], Eg[ok], Z[ok], s=14, color=style.SERIES[baseline],
            edgecolor=style.SURFACE, linewidth=0.5, depthshade=False,
        )
        ax.set_xlabel(r"$\alpha_b$", labelpad=-6)
        ax.set_ylabel(r"$\alpha_e$", labelpad=-6)
        ax.set_zlim(zmin - pad, zmax + pad)      # identical z limits across panels
        ax.set_xticks(alphas_b)
        ax.set_yticks(alphas_e)
        ax.tick_params(pad=-2)
        ax.set_title(f"({panel_tags[k]}) {style.LABELS[baseline]}", fontsize=7, loc="left")
        if k == 0:
            ax.set_zlabel(labels.get(metric, metric), labelpad=4, fontsize=7)
        ax.view_init(elev=22, azim=-128)
        ax.grid(True)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_alpha(0.0)
    fig.suptitle(
        f"E3 coordination surfaces -- {labels.get(metric, metric)}"
        + (f", {family} trajectory" if family else "")
        + " (9 measured points per panel, identical z limits)",
        fontsize=7, color=style.TEXT_SECONDARY, y=1.02,
    )
    suffix = f"_{family}" if family else ""
    return style.save(fig, out_dir, f"E3_3D_coordination{suffix}")


# ==========================================================================
# E4
# ==========================================================================
def e4_figures(episode_dir: str, out_dir: str, baseline: str = "C1"):
    """Overlay, error-vs-time and base-motion-vs-drift from one E4 episode."""
    import csv
    import json

    style.apply_style()
    with open(os.path.join(episode_dir, "states.csv"), encoding="utf-8") as handle:
        states = list(csv.DictReader(handle))
    with open(os.path.join(episode_dir, "commands.csv"), encoding="utf-8") as handle:
        commands = list(csv.DictReader(handle))
    with open(os.path.join(episode_dir, "metadata.json"), encoding="utf-8") as handle:
        meta = json.load(handle)

    def col(rows, key):
        return np.array([float(r[key]) for r in rows])

    evaluated = np.array([int(r["evaluated"]) for r in states], dtype=bool)
    t = col(states, "t")
    target = np.asarray(
        meta["scenario"].get("ee_world_target_position", [np.nan] * 3), dtype=float
    )

    paths = []
    # --- (1) world-frame EE trajectory overlay, top view ---
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_WIDTH, 2.4))
    fig.subplots_adjust(wspace=0.32)
    ax = axes[0]
    ax.plot(col(states, "base_x"), col(states, "base_y"), color=style.TEXT_MUTED,
            linewidth=1.0, linestyle=":", label="base path")
    ax.plot(col(states, "ee_act_x_w")[evaluated], col(states, "ee_act_y_w")[evaluated],
            color=style.SERIES[baseline], linewidth=1.2, label="EE actual (world)")
    if np.isfinite(target).all():
        ax.plot(target[0], target[1], marker="*", markersize=9,
                color=style.TEXT_PRIMARY, linestyle="none",
                label="EE commanded (world, fixed)")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("(a) Top view", loc="left")
    ax.legend(loc="best")
    ax.grid(True, axis="both")

    ax = axes[1]
    ax.plot(t[evaluated], col(states, "ee_act_x_w")[evaluated],
            color=style.SERIES[baseline], label="EE x (world)")
    ax.plot(t[evaluated], col(states, "ee_act_z_w")[evaluated],
            color=style.SERIES["C3"], linestyle="--", label="EE z (world)")
    if np.isfinite(target).all():
        ax.axhline(target[0], color=style.TEXT_MUTED, linewidth=0.8, linestyle=":")
        ax.axhline(target[2], color=style.TEXT_MUTED, linewidth=0.8, linestyle=":")
    ax.plot(t[evaluated], col(states, "base_x")[evaluated],
            color=style.TEXT_MUTED, linewidth=1.0, label="base x (world)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("world position (m)")
    ax.set_title("(b) Commanded vs actual over time", loc="left")
    ax.legend(loc="best", ncol=2)
    fig.suptitle(
        f"E4 world-frame EE stabilisation, {style.LABELS[baseline]} "
        "(illustrative / curated example)",
        fontsize=8, y=1.03,
    )
    paths += style.save(fig, out_dir, f"E4_ee_world_overlay_{baseline}")

    # --- (2) error vs time and base motion vs drift ---
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_WIDTH, 2.1))
    fig.subplots_adjust(wspace=0.34)
    ax = axes[0]
    if "ee_world_pos_err" in states[0]:
        ax.plot(t[evaluated], col(states, "ee_world_pos_err")[evaluated],
                color=style.SERIES[baseline], label="position error")
        ax.set_ylabel("world EE position error (m)")
    ax.set_xlabel("time (s)")
    ax.set_title("(a) EE position error vs time", loc="left")
    ax.legend(loc="best")

    ax = axes[1]
    base_speed = np.hypot(col(states, "vx_act"), col(states, "vy_act"))
    err = col(states, "ee_world_pos_err") if "ee_world_pos_err" in states[0] else None
    if err is not None:
        ax.scatter(base_speed[evaluated], err[evaluated], s=3,
                   color=style.SERIES[baseline], alpha=0.5, edgecolor="none")
        ax.set_ylabel("world EE position error (m)")
    ax.set_xlabel("base speed magnitude (m/s)")
    ax.set_title("(b) Base motion vs EE world drift", loc="left")
    paths += style.save(fig, out_dir, f"E4_error_vs_time_{baseline}")
    return paths


def e4_snapshots(baseline: str, seed: int, out_dir: str, exp_cfg: dict,
                 width=560, height=420, azimuth=138.0, elevation=-12.0,
                 distance=2.1):
    """Re-run one E4 episode with an offscreen renderer and save snapshots."""
    import mujoco

    from benchmark.controllers.registry import build_adapter
    from benchmark.core import frames
    from benchmark.scenarios import experiments as E

    style.apply_style()
    scenario = E.build_scenario("E4", exp_cfg, "chicken_head", seed)
    adapter = build_adapter(baseline)
    model = adapter.build_model()
    data = mujoco.MjData(model)
    adapter.attach(model, data)
    adapter.set_initial_state()
    initial_ee = frames.ee_keypoints_plb_from_state(
        data.xpos[adapter.ee_bid], data.xmat[adapter.ee_bid].reshape(3, 3),
        data.qpos[0:3], data.qpos[3:7], adapter.ee_ground_z,
        adapter.ee_kp_dx, adapter.ee_kp_dz,
    )
    scenario.prepare(adapter, initial_ee)
    settle_base, settle_ee = scenario.settle_commands(adapter, initial_ee)
    adapter.reset(settle_base, settle_ee)

    times = list(exp_cfg["outputs"]["snapshot_times_s"])
    settle_ticks = int(round(float(exp_cfg["settle_s"]) / adapter.control_dt))
    n_steps = int(round(float(exp_cfg["duration_s"]) / adapter.simulation_dt))

    renderer = mujoco.Renderer(model, height, width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    camera.distance = float(distance)

    frames_out, labels, sim_time, tick, pending = [], [], 0.0, 0, list(times)
    for counter in range(n_steps):
        adapter.pre_physics(sim_time)
        mujoco.mj_step(model, data)
        sim_time += adapter.simulation_dt
        if counter % adapter.control_decimation == 0:
            k = tick - settle_ticks
            if k < 0:
                base_cmd, ee_cmd = settle_base, settle_ee
            else:
                base_cmd, ee_cmd, _ = scenario.command(k, adapter)
            adapter.policy_tick(base_cmd, ee_cmd, sim_time)
            tick += 1
        if pending and sim_time >= pending[0] - 1e-9:
            # Frame on the torso/arm rather than the base origin.
            camera.lookat[:] = data.qpos[:3]
            camera.lookat[2] = float(data.qpos[2]) + 0.25
            renderer.update_scene(data, camera)
            frames_out.append(renderer.render().copy())
            labels.append(f"t = {pending[0]:.0f} s")
            pending.pop(0)
    renderer.close()

    if not frames_out:
        return []
    n = len(frames_out)
    fig, axes = plt.subplots(
        1, n, figsize=(style.PAGE_WIDTH, style.PAGE_WIDTH / n * (height / width))
    )
    fig.subplots_adjust(wspace=0.04)
    for ax, image, label in zip(np.atleast_1d(axes), frames_out, labels):
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Time labels sit BELOW each frame so they cannot collide with the
        # figure title.
        ax.set_xlabel(label, fontsize=7, color=style.TEXT_SECONDARY, labelpad=2)
    fig.suptitle(
        f"E4 whole-body snapshots, {style.LABELS[baseline]}, seed {seed} "
        "(illustrative / curated example)",
        fontsize=8, y=1.08,
    )
    return style.save(fig, out_dir, f"E4_snapshots_{baseline}_seed{seed:03d}")


def e4_video(baseline: str, seed: int, out_path: str, exp_cfg: dict,
             fps: int = 50, width=640, height=480):
    """Optional MP4 of one E4 episode (needs imageio-ffmpeg)."""
    import mujoco

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("  imageio not installed; skipping E4 video")
        return None

    from benchmark.controllers.registry import build_adapter
    from benchmark.core import frames
    from benchmark.scenarios import experiments as E

    scenario = E.build_scenario("E4", exp_cfg, "chicken_head", seed)
    adapter = build_adapter(baseline)
    model = adapter.build_model()
    data = mujoco.MjData(model)
    adapter.attach(model, data)
    adapter.set_initial_state()
    initial_ee = frames.ee_keypoints_plb_from_state(
        data.xpos[adapter.ee_bid], data.xmat[adapter.ee_bid].reshape(3, 3),
        data.qpos[0:3], data.qpos[3:7], adapter.ee_ground_z,
        adapter.ee_kp_dx, adapter.ee_kp_dz,
    )
    scenario.prepare(adapter, initial_ee)
    settle_base, settle_ee = scenario.settle_commands(adapter, initial_ee)
    adapter.reset(settle_base, settle_ee)

    settle_ticks = int(round(float(exp_cfg["settle_s"]) / adapter.control_dt))
    n_steps = int(round(float(exp_cfg["duration_s"]) / adapter.simulation_dt))
    every = max(1, int(round(1.0 / (fps * adapter.simulation_dt))))

    renderer = mujoco.Renderer(model, height, width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 120, -12, 3.2

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps)
    sim_time, tick = 0.0, 0
    for counter in range(n_steps):
        adapter.pre_physics(sim_time)
        mujoco.mj_step(model, data)
        sim_time += adapter.simulation_dt
        if counter % adapter.control_decimation == 0:
            k = tick - settle_ticks
            if k < 0:
                base_cmd, ee_cmd = settle_base, settle_ee
            else:
                base_cmd, ee_cmd, _ = scenario.command(k, adapter)
            adapter.policy_tick(base_cmd, ee_cmd, sim_time)
            tick += 1
        if counter % every == 0:
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera)
            writer.append_data(renderer.render())
    writer.close()
    renderer.close()
    print(f"  wrote {os.path.relpath(out_path)}")
    return out_path


# ==========================================================================
# side-by-side trajectory comparison for a chosen seed
# ==========================================================================
def side_by_side_trajectories(episode_dirs: dict, out_dir: str, name: str,
                              title: str = ""):
    """Compare one curated seed across baselines: EE error and base tracking."""
    import csv

    style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(style.PAGE_WIDTH, 2.1))
    fig.subplots_adjust(wspace=0.40)
    for baseline, path in sorted(episode_dirs.items()):
        with open(os.path.join(path, "states.csv"), encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with open(os.path.join(path, "commands.csv"), encoding="utf-8") as handle:
            cmds = list(csv.DictReader(handle))
        ev = np.array([int(r["evaluated"]) for r in rows], dtype=bool)
        t = np.array([float(r["t"]) for r in rows])[ev]
        kwargs = style.series_kwargs(baseline)
        kwargs.pop("marker")
        axes[0].plot(t, np.array([float(r["ee_pos_err"]) for r in rows])[ev], **kwargs)
        axes[1].plot(
            t, np.array([float(r["ee_ori_err_deg"]) for r in rows])[ev], **kwargs
        )
        vx = np.array([float(r["vx_act"]) for r in rows])[ev]
        vy = np.array([float(r["vy_act"]) for r in rows])[ev]
        cvx = np.array([float(r["vx_cmd"]) for r in cmds])[ev]
        cvy = np.array([float(r["vy_cmd"]) for r in cmds])[ev]
        axes[2].plot(t, np.hypot(vx - cvx, vy - cvy), **kwargs)
    cmd_kwargs = dict(color=style.TEXT_MUTED, linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("EE position error (m)")
    axes[1].set_ylabel("EE orientation error (deg)")
    axes[2].set_ylabel("Base XY vel. error (m/s)")
    for ax in axes:
        ax.set_xlabel("time (s)")
    axes[0].legend(loc="upper left")
    if title:
        fig.suptitle(title + "  (illustrative / curated example)", fontsize=8, y=1.04)
    return style.save(fig, out_dir, name)
