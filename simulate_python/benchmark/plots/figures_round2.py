"""Round-2 figures: E3 absolute tracking, degradation, and C1-vs-C3 pairing.

Design decisions, so they are not re-litigated per figure:

* **Family pooling uses a cluster bootstrap over seeds.** Where a panel shows
  one curve per baseline rather than one per trajectory family, the three
  families of a given seed are first averaged into ONE number for that seed,
  and the bootstrap then resamples seeds. The three families are three
  different tasks run on the same seeded initial state, so resampling episodes
  would treat them as independent replicates when they are not. Per-family
  figures are emitted alongside for the supplement.
* **Paired panels difference per (seed, family) first**, then average over
  families, then bootstrap seeds -- pairing is never broken by aggregating
  first.
* **Signed quantities get the diverging scale and an explicit zero line.** The
  zero point is drawn, labelled and never clipped; negative values are shown as
  negative.
* Axis limits are the data's own range padded symmetrically about zero for
  signed quantities. No axis is truncated to magnify a small difference.
"""

from __future__ import annotations

import numpy as np

from benchmark.core import stats
from benchmark.plots import style
from benchmark.plots.style import plot_with_ci, save

FAMILIES = ("line", "circle", "semicircle")
ALPHAS = (0.0, 0.5, 1.0)
CELL_NAME = {(0.0, 0.0): "S00", (0.0, 0.5): "S01", (0.0, 1.0): "S02",
             (0.5, 0.0): "S10", (0.5, 0.5): "S11", (0.5, 1.0): "S12",
             (1.0, 0.0): "S20", (1.0, 0.5): "S21", (1.0, 1.0): "S22"}

METRIC_AXIS = {
    "ee_pos_rmse": ("EE position RMSE (m)", "m"),
    "base_vxy_rmse": ("Base XY velocity RMSE (m/s)", "m/s"),
    "ee_ori_mae_deg": ("EE orientation MAE (deg)", "deg"),
    "base_wz_rmse": ("Base yaw-rate RMSE (rad/s)", "rad/s"),
    "delta_ee_pos_rmse_from_base": (r"Locomotion-induced $\Delta$EE RMSE (m)", "m"),
    "delta_base_vxy_rmse_from_ee": (r"Manipulation-induced $\Delta$base RMSE (m/s)", "m/s"),
}


def _cell_rows(rows, ab=None, ae=None, family=None, baseline=None):
    out = []
    for row in rows:
        if "alpha_base" not in row or "alpha_ee" not in row:
            continue
        if ab is not None and abs(float(row["alpha_base"]) - ab) > 1e-9:
            continue
        if ae is not None and abs(float(row["alpha_ee"]) - ae) > 1e-9:
            continue
        if family is not None and row.get("ee_family") != family:
            continue
        if baseline is not None and row.get("baseline") != baseline:
            continue
        out.append(row)
    return out


def _per_seed_pooled(rows, metric):
    """seed -> mean of ``metric`` over the trajectory families present."""
    buckets: dict = {}
    for row in rows:
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            continue
        buckets.setdefault(int(row["seed"]), []).append(float(value))
    return {seed: float(np.mean(values)) for seed, values in buckets.items()}


def _series_pooled(rows, baseline, metric, sweep_key, sweep_values, fixed, cfg_stats):
    """Mean + CI vs the swept intensity, pooled over families by seed."""
    mean, lo, hi, n = [], [], [], []
    for value in sweep_values:
        kwargs = {sweep_key: value}
        kwargs.update(fixed)
        subset = _cell_rows(rows, baseline=baseline, **kwargs)
        per_seed = _per_seed_pooled(subset, metric)
        s = stats.summarize(
            list(per_seed.values()),
            n_resamples=cfg_stats["bootstrap_resamples"],
            seed=cfg_stats["bootstrap_seed"],
            confidence=cfg_stats["confidence"],
        )
        mean.append(s["mean"]); lo.append(s["ci_low"]); hi.append(s["ci_high"]); n.append(s["n"])
    return np.array(mean), np.array(lo), np.array(hi), np.array(n)


def _paired_series_pooled(rows, reference, rival, metric, sweep_key, sweep_values,
                          fixed, cfg_stats):
    """Mean + CI of (rival - reference), paired per (seed, family)."""
    mean, lo, hi, n, sig = [], [], [], [], []
    for value in sweep_values:
        kwargs = {sweep_key: value}
        kwargs.update(fixed)
        subset = _cell_rows(rows, **kwargs)
        ref = {(int(r["seed"]), r["ee_family"]): float(r[metric])
               for r in subset if r["baseline"] == reference
               and r.get(metric) is not None and np.isfinite(r[metric])}
        oth = {(int(r["seed"]), r["ee_family"]): float(r[metric])
               for r in subset if r["baseline"] == rival
               and r.get(metric) is not None and np.isfinite(r[metric])}
        per_seed: dict = {}
        for key in set(ref) & set(oth):
            per_seed.setdefault(key[0], []).append(oth[key] - ref[key])
        diffs = [float(np.mean(v)) for v in per_seed.values()]
        s = stats.summarize(
            diffs,
            n_resamples=cfg_stats["bootstrap_resamples"],
            seed=cfg_stats["bootstrap_seed"],
            confidence=cfg_stats["confidence"],
        )
        mean.append(s["mean"]); lo.append(s["ci_low"]); hi.append(s["ci_high"])
        n.append(s["n"])
        sig.append(bool(np.isfinite(s["ci_low"]) and (s["ci_low"] > 0 or s["ci_high"] < 0)))
    return np.array(mean), np.array(lo), np.array(hi), np.array(n), sig


# ==========================================================================
# Figure E3-A -- absolute tracking
# ==========================================================================
def e3_absolute_tracking(rows, out_dir, exp_cfg, cfg_stats, baselines=("C1", "C3", "C5"),
                         family=None, name=None, fixed_alpha_ee=None,
                         fixed_alpha_base=None):
    """Panel A: EE position RMSE vs base intensity at a fixed nonzero EE speed.
    Panel B: base XY velocity RMSE vs EE speed at a fixed nonzero base intensity.

    ``fixed_alpha_ee`` / ``fixed_alpha_base`` override the config defaults. The
    EE-speed slice matters: which of C1 and C3 has the lower EE error depends
    on it, so the slice where C3 wins is plotted too rather than only the one
    where C1 does.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    fixed_ae = float(exp_cfg["plots"]["panel_a_fixed_alpha_ee"]
                     if fixed_alpha_ee is None else fixed_alpha_ee)
    fixed_ab = float(exp_cfg["plots"]["panel_b_fixed_alpha_base"]
                     if fixed_alpha_base is None else fixed_alpha_base)
    scope = {} if family is None else {"family": family}

    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_WIDTH, 2.5))
    panels = [
        (axes[0], "ee_pos_rmse", "alpha_base", r"Base motion intensity $\alpha_{base}$",
         {"ae": fixed_ae}, f"(a) EE tracking, EE speed $\\alpha_{{ee}}$ = {fixed_ae:g}"),
        (axes[1], "base_vxy_rmse", "alpha_ee", r"EE motion intensity $\alpha_{ee}$",
         {"ab": fixed_ab}, f"(b) Base tracking, base $\\alpha_{{base}}$ = {fixed_ab:g}"),
    ]
    n_report = {}
    for ax, metric, sweep, xlabel, fixed, title in panels:
        fixed = dict(fixed); fixed.update(scope)
        for baseline in baselines:
            mean, lo, hi, n = _series_pooled(
                rows, baseline, metric,
                "ab" if sweep == "alpha_base" else "ae", ALPHAS, fixed, cfg_stats,
            )
            plot_with_ci(ax, ALPHAS, mean, lo, hi, baseline)
            n_report[(metric, baseline)] = n
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRIC_AXIS[metric][0])
        ax.set_title(title, loc="left", color=style.TEXT_SECONDARY)
        ax.set_xticks(ALPHAS)
        ax.set_xlim(-0.06, 1.14)
    axes[0].legend(loc="best", ncol=1)
    sub = ("mean over the three trajectory families" if family is None
           else f"{family} trajectory family")
    fig.text(0.0, -0.09,
             f"{sub}; shaded = 95% bootstrap CI over "
             f"{int(np.max(list(n_report.values())))} seeds",
             fontsize=6.5, color=style.TEXT_MUTED)
    fig.subplots_adjust(wspace=0.34)
    return save(fig, out_dir, name or "E3_absolute_tracking")


# ==========================================================================
# Figure E3-B -- directional degradation
# ==========================================================================
def e3_error_degradation(rows, out_dir, exp_cfg, cfg_stats, baselines=("C1", "C3", "C5"),
                         family=None, name=None):
    import matplotlib.pyplot as plt

    style.apply_style()
    fixed_ae = float(exp_cfg["plots"]["panel_a_fixed_alpha_ee"])
    fixed_ab = float(exp_cfg["plots"]["panel_b_fixed_alpha_base"])
    scope = {} if family is None else {"family": family}

    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_WIDTH, 2.5))
    panels = [
        (axes[0], "delta_ee_pos_rmse_from_base", "ab",
         r"Base motion intensity $\alpha_{base}$", {"ae": fixed_ae},
         f"(a) Locomotion $\\rightarrow$ EE, $\\alpha_{{ee}}$ = {fixed_ae:g}"),
        (axes[1], "delta_base_vxy_rmse_from_ee", "ae",
         r"EE motion intensity $\alpha_{ee}$", {"ab": fixed_ab},
         f"(b) Manipulation $\\rightarrow$ base, $\\alpha_{{base}}$ = {fixed_ab:g}"),
    ]
    for ax, metric, sweep, xlabel, fixed, title in panels:
        fixed = dict(fixed); fixed.update(scope)
        ax.axhline(0.0, color=style.TEXT_SECONDARY, linewidth=0.8, zorder=1)
        ax.annotate("no degradation", xy=(0.02, 0.0), xytext=(0, 3),
                    textcoords="offset points", fontsize=6.5,
                    color=style.TEXT_MUTED, va="bottom")
        for baseline in baselines:
            mean, lo, hi, _ = _series_pooled(
                rows, baseline, metric, sweep, ALPHAS, fixed, cfg_stats)
            plot_with_ci(ax, ALPHAS, mean, lo, hi, baseline)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRIC_AXIS[metric][0])
        ax.set_title(title, loc="left", color=style.TEXT_SECONDARY)
        ax.set_xticks(ALPHAS)
        ax.set_xlim(-0.06, 1.14)
    axes[0].legend(loc="lower left")
    sub = ("mean over the three trajectory families" if family is None
           else f"{family} trajectory family")
    fig.text(0.0, -0.09,
             f"{sub}; positive = error INCREASES when the other subsystem "
             f"moves. Negative values are real and are not clipped.",
             fontsize=6.5, color=style.TEXT_MUTED)
    fig.subplots_adjust(wspace=0.34)
    return save(fig, out_dir, name or "E3_error_degradation")


# ==========================================================================
# Figure E3-C -- direct paired advantage, C1 vs one rival
# ==========================================================================
def e3_paired_advantage(rows, out_dir, exp_cfg, cfg_stats, rival="C3", name=None):
    """Panel A: (rival - C1) EE RMSE vs base intensity, one line per EE speed.
    Panel B: (rival - C1) base RMSE vs EE speed, one line per base intensity.

    Positive = C1 better. The EE-speed family is drawn with the ordinal blue
    ramp because it is an ordered magnitude, not an identity.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_WIDTH, 2.6))
    specs = [
        (axes[0], "ee_pos_rmse", "ab", "ae",
         r"Base motion intensity $\alpha_{base}$",
         f"(a) EE position advantage, {rival} $-$ C1", r"$\alpha_{ee}$"),
        (axes[1], "base_vxy_rmse", "ae", "ab",
         r"EE motion intensity $\alpha_{ee}$",
         f"(b) Base velocity advantage, {rival} $-$ C1", r"$\alpha_{base}$"),
    ]
    for ax, metric, sweep, series_key, xlabel, title, series_label in specs:
        ax.axhline(0.0, color=style.TEXT_PRIMARY, linewidth=0.9, zorder=3)
        for i, level in enumerate(ALPHAS):
            mean, lo, hi, _, sig = _paired_series_pooled(
                rows, "C1", rival, metric, sweep, ALPHAS, {series_key: level}, cfg_stats)
            kwargs = style.ordinal_kwargs(i, len(ALPHAS))
            ax.fill_between(ALPHAS, lo, hi, color=kwargs["color"], alpha=0.13, linewidth=0)
            ax.plot(ALPHAS, mean, label=f"{series_label} = {level:g}", **kwargs)
            # Mark the cells whose paired CI contains zero: not separated.
            for x, m, s in zip(ALPHAS, mean, sig):
                if not s:
                    ax.plot([x], [m], marker="o", markersize=7.5, markerfacecolor="none",
                            markeredgecolor=style.TEXT_SECONDARY, markeredgewidth=0.7,
                            linestyle="none", zorder=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{rival} $-$ C1   ({METRIC_AXIS[metric][1]})")
        ax.set_title(title, loc="left", color=style.TEXT_SECONDARY)
        ax.set_xticks(ALPHAS)
        ax.set_xlim(-0.06, 1.06)
        lim = max(abs(np.array(ax.get_ylim()))) * 1.05
        ax.set_ylim(-lim, lim)
        ax.legend(loc="best")
    fig.text(0.0, -0.10,
             f"Positive = C1 lower error. Paired per (seed, trajectory family), "
             f"averaged over families per seed, 95% bootstrap CI over seeds. "
             f"A hollow ring marks a point whose CI contains zero "
             f"(C1 and {rival} not separated there).",
             fontsize=6.5, color=style.TEXT_MUTED)
    fig.subplots_adjust(wspace=0.34)
    return save(fig, out_dir, name or f"E3_C1_vs_{rival}_advantage")


# ==========================================================================
# Figure E3-C alt -- the full 3x3 matrix as two heatmaps
# ==========================================================================
def e3_paired_heatmaps(rows, out_dir, cfg_stats, rival="C3", name=None):
    import matplotlib.pyplot as plt

    style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_WIDTH, 2.9))
    panels = [
        (axes[0], "ee_pos_rmse", f"(a) EE position RMSE: {rival} $-$ C1 (m)", "%+.3f"),
        (axes[1], "base_vxy_rmse", f"(b) Base XY vel. RMSE: {rival} $-$ C1 (m/s)", "%+.3f"),
    ]
    for ax, metric, title, fmt in panels:
        grid = np.full((len(ALPHAS), len(ALPHAS)), np.nan)
        sig = np.zeros_like(grid, dtype=bool)
        for i, ab in enumerate(ALPHAS):
            for j, ae in enumerate(ALPHAS):
                mean, lo, hi, n, s = _paired_series_pooled(
                    rows, "C1", rival, metric, "ab", [ab], {"ae": ae}, cfg_stats)
                grid[i, j] = mean[0]
                sig[i, j] = s[0]
        lim = float(np.nanmax(np.abs(grid)))
        lim = lim if lim > 0 else 1.0
        im = ax.imshow(grid, cmap=style.DIVERGING, vmin=-lim, vmax=lim, origin="lower")
        ax.set_xticks(range(len(ALPHAS)), [f"{a:g}" for a in ALPHAS])
        ax.set_yticks(range(len(ALPHAS)), [f"{a:g}" for a in ALPHAS])
        ax.set_xlabel(r"EE motion intensity $\alpha_{ee}$")
        ax.set_ylabel(r"Base motion intensity $\alpha_{base}$")
        ax.set_title(title, loc="left", color=style.TEXT_SECONDARY, fontsize=7.5)
        ax.grid(False)
        for i in range(len(ALPHAS)):
            for j in range(len(ALPHAS)):
                value = grid[i, j]
                # Label ink by the cell's own lightness, not by the series hue.
                rgba = style.DIVERGING((value + lim) / (2 * lim))
                lightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ink = "#ffffff" if lightness < 0.55 else style.TEXT_PRIMARY
                text = fmt % value + ("" if sig[i, j] else "\n(ns)")
                ax.text(j, i, text, ha="center", va="center", fontsize=6.2, color=ink,
                        linespacing=1.1)
                ax.text(j, i + 0.33, CELL_NAME[(ALPHAS[i], ALPHAS[j])],
                        ha="center", va="center", fontsize=5.2, color=ink,
                        alpha=0.75)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=6, length=2)
        cbar.set_ticks([-lim, 0.0, lim])
        cbar.set_ticklabels([f"{-lim:+.3f}\n{rival} better", "0", f"{lim:+.3f}\nC1 better"])
    fig.text(0.0, -0.08,
             "Blue = C1 lower error, red = rival lower error, neutral = zero. "
             "Every cell is also printed as a signed number; '(ns)' marks a "
             "paired 95% CI that contains zero. Mean over the three trajectory "
             "families, paired per (seed, family).",
             fontsize=6.5, color=style.TEXT_MUTED)
    fig.subplots_adjust(wspace=0.45)
    return save(fig, out_dir, name or f"E3_C1_vs_{rival}_heatmaps")


# ==========================================================================
# Selected illustrative cases
#
# Every one of these figures is built from a seed chosen by
# benchmark/select_cases.py out of a candidate pool DISJOINT from the official
# statistical pool. The footer of each figure says so, because a reader who
# sees only the figure must not mistake it for average performance.
# ==========================================================================
import csv as _csv  # noqa: E402
import os as _os  # noqa: E402

ILLUSTRATIVE_FOOTER = (
    "Selected illustrative trials from a separately evaluated candidate seed "
    "pool (seeds 100-149), chosen by a pre-declared ranking; see the selection "
    "manifest. NOT benchmark average performance -- the official statistics use "
    "every seed 0-59."
)
COMMAND_INK = style.TEXT_PRIMARY


def _row_header(axes_row, title, metric_summary):
    """One centred two-line header per row, over the middle panel.

    Per-axes left- and right-aligned titles collide once the method summary is
    long enough to be useful, so the header spans the row instead.
    """
    # pad lifts the header clear of the neighbouring panels' y-axis labels,
    # which otherwise sit in the same horizontal band.
    axes_row[len(axes_row) // 2].set_title(
        title + ("\n" + metric_summary if metric_summary else ""),
        loc="center", color=style.TEXT_SECONDARY, fontsize=6.6, linespacing=1.5,
        pad=14,
    )


def _figure_legend(fig, axes_row, footer, n_rows):
    """A single figure-level legend under the panels, plus the footer note.

    Inside a panel the legend lands on top of the traces in at least one row;
    below the figure it never does, and one legend for the whole figure is less
    ink than one per row.
    """
    handles, labels = [], []
    for ax in axes_row:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 0.045 - 0.004 * n_rows),
                   ncol=min(len(handles), 5), fontsize=7, frameon=False)
    fig.text(0.0, -0.025 - 0.012 * n_rows, footer, fontsize=6.5,
             color=style.TEXT_MUTED, wrap=True)


def _load_table(path, filename):
    full = path if _os.path.isabs(path) else _os.path.join(SIM_ROOT_FOR_PATHS, path)
    target = _os.path.join(full, filename)
    if not _os.path.isfile(target):
        return {}
    columns: dict = {}
    with open(target, newline="", encoding="utf-8") as handle:
        reader = _csv.DictReader(handle)
        names = reader.fieldnames or []
        for name in names:
            columns[name] = []
        for row in reader:
            for name in names:
                try:
                    columns[name].append(float(row[name]))
                except (TypeError, ValueError):
                    columns[name].append(np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in columns.items()}


SIM_ROOT_FOR_PATHS = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..")
)


def _evaluated(table):
    if not table:
        return np.zeros(0, dtype=bool)
    return table["evaluated"] > 0.5


def _leg_arm_power(path, tolerance=1e-6):
    """Exact 250 Hz leg/arm |tau * qdot| reconstructed from ``physics.npz``.

    ``physics.npz`` stores one row per EVALUATED physics step (``record_physics``
    returns early while ``sim_time < settle_s``), laid out as

        [t] qpos_leg qpos_arm | qvel_leg qvel_arm | qacc_leg qacc_arm | tau_leg tau_arm

    so each of the three vector blocks is ``n_leg + n_arm`` wide. The time mean
    of the reconstruction must therefore equal ``metrics.json``'s ``leg_power``
    and ``arm_power`` exactly; this function CHECKS that and raises if it does
    not, because a silent off-by-one-block slice reads accelerations as
    velocities and inflates the curve by orders of magnitude without looking
    obviously wrong.
    """
    full = path if _os.path.isabs(path) else _os.path.join(SIM_ROOT_FOR_PATHS, path)
    target = _os.path.join(full, "physics.npz")
    if not _os.path.isfile(target):
        return None
    data = np.load(target)["data"].astype(np.float64)
    n_arm = 6
    n_total = (data.shape[1] - 1) // 4
    n_leg = n_total - n_arm
    if n_leg <= 0 or 1 + 4 * n_total != data.shape[1]:
        raise ValueError(f"unexpected physics.npz layout: {data.shape} columns")
    qvel = data[:, 1 + n_total: 1 + 2 * n_total]
    tau = data[:, 1 + 3 * n_total: 1 + 4 * n_total]
    leg_power = np.sum(np.abs(tau[:, :n_leg] * qvel[:, :n_leg]), axis=1)
    arm_power = np.sum(np.abs(tau[:, n_leg:] * qvel[:, n_leg:]), axis=1)

    metrics_path = _os.path.join(full, "metrics.json")
    if _os.path.isfile(metrics_path):
        import json as _json

        with open(metrics_path, encoding="utf-8") as handle:
            metrics = _json.load(handle)
        for name, series in (("leg_power", leg_power), ("arm_power", arm_power)):
            expected = float(metrics[name])
            got = float(np.mean(series))
            # float32 storage, so compare at a relative tolerance rather than
            # exactly; a wrong block is off by orders of magnitude, not ppm.
            if abs(got - expected) > max(tolerance, 1e-3 * abs(expected)):
                raise ValueError(
                    f"reconstructed {name} = {got:.6g} W does not match "
                    f"metrics.json {expected:.6g} W for {path}; the "
                    f"physics.npz column layout has changed"
                )
    return {
        "t": data[:, 0],
        "leg_power": leg_power,
        "arm_power": arm_power,
        "n_leg": n_leg, "n_arm": n_arm,
    }


def _smooth(x, window):
    if window <= 1 or x.size < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def e1_selected_cases(cases, out_dir, name="E1_selected_cases"):
    """One row per selected trial: base vx, yaw rate, EE position error."""
    import matplotlib.pyplot as plt

    style.apply_style()
    cases = [c for c in cases if c.get("selected")]
    if not cases:
        return []
    fig, axes = plt.subplots(len(cases), 3, figsize=(style.PAGE_WIDTH, 1.75 * len(cases)),
                             squeeze=False)
    for r, case in enumerate(cases):
        sel = case["selected"]
        baselines = list(sel["metrics"].keys())
        tables = {b: (_load_table(sel["paths"][b], "commands.csv"),
                      _load_table(sel["paths"][b], "states.csv")) for b in baselines}
        ref_cmd, _ = tables[baselines[0]]
        mask_c = _evaluated(ref_cmd)
        t0 = ref_cmd["t"][mask_c][0]

        for col, (cmd_key, act_key, ylabel) in enumerate([
            ("vx_cmd", "vx_act", r"Forward velocity $v_x$ (m/s)"),
            ("wz_cmd", "wz_act", r"Yaw rate $\omega_z$ (rad/s)"),
            (None, "ee_pos_err", "EE position error (m)"),
        ]):
            ax = axes[r][col]
            if cmd_key is not None:
                ax.step(ref_cmd["t"][mask_c] - t0, ref_cmd[cmd_key][mask_c],
                        where="post", color=COMMAND_INK, linewidth=1.0,
                        linestyle=(0, (4, 2)), label="command", zorder=5)
            for b in baselines:
                _, states = tables[b]
                mask = _evaluated(states)
                kwargs = style.series_kwargs(b)
                kwargs["marker"] = None
                kwargs["label"] = style.SHORT[b]
                kwargs["linewidth"] = 1.0
                ax.plot(states["t"][mask] - t0, states[act_key][mask], **kwargs)
            if col == 2:
                # Mark where the EE target switched: the transients matter and
                # are inside the evaluated window on purpose.
                ee_cols = [f"ee_cmd_kp{i}" for i in range(9)]
                ee = np.column_stack([ref_cmd[c][mask_c] for c in ee_cols])
                switches = np.nonzero(np.any(np.abs(np.diff(ee, axis=0)) > 1e-9, axis=1))[0]
                for s in switches:
                    ax.axvline(ref_cmd["t"][mask_c][s + 1] - t0, color=style.GRID,
                               linewidth=0.7, zorder=0)
            ax.set_ylabel(ylabel, fontsize=7)
            if r == len(cases) - 1:
                ax.set_xlabel("Time in evaluated window (s)")
        _row_header(
            axes[r],
            f"({chr(97 + r)}) {case['title']} -- seed {sel['seed']}, {sel['condition']}",
            "  ".join(
                f"{b}: $E_v$ {sel['metrics'][b]['base_vxy_rmse']:.3f},  "
                f"$E_p$ {sel['metrics'][b]['ee_pos_rmse']:.3f}"
                for b in baselines),
        )
    _figure_legend(fig, axes[0], ILLUSTRATIVE_FOOTER
                   + " $E_v$ in m/s, $E_p$ in m; every metric for every method "
                     "is in the companion table.", len(cases))
    fig.subplots_adjust(hspace=0.85, wspace=0.34)
    return save(fig, out_dir, name)


def e2_selected_cases(cases, out_dir, name="E2_selected_cases"):
    """One row per selected trial: base vx tracking, leg power, leg acceleration."""
    import matplotlib.pyplot as plt

    style.apply_style()
    cases = [c for c in cases if c.get("selected")]
    if not cases:
        return []
    fig, axes = plt.subplots(len(cases), 3, figsize=(style.PAGE_WIDTH, 1.75 * len(cases)),
                             squeeze=False)
    missing_physics = False
    for r, case in enumerate(cases):
        sel = case["selected"]
        baselines = list(sel["metrics"].keys())
        ref_cmd = _load_table(sel["paths"][baselines[0]], "commands.csv")
        mask_c = _evaluated(ref_cmd)
        t0 = ref_cmd["t"][mask_c][0]

        ax = axes[r][0]
        ax.step(ref_cmd["t"][mask_c] - t0, ref_cmd["vx_cmd"][mask_c], where="post",
                color=COMMAND_INK, linewidth=1.0, linestyle=(0, (4, 2)),
                label="command", zorder=5)
        for b in baselines:
            states = _load_table(sel["paths"][b], "states.csv")
            mask = _evaluated(states)
            kwargs = style.series_kwargs(b); kwargs["marker"] = None
            kwargs["label"] = style.SHORT[b]; kwargs["linewidth"] = 1.0
            ax.plot(states["t"][mask] - t0, states["vx_act"][mask], **kwargs)
        ax.set_ylabel(r"Forward velocity $v_x$ (m/s)", fontsize=7)

        ax = axes[r][1]
        any_power = False
        for b in baselines:
            power = _leg_arm_power(sel["paths"][b])
            if power is None:
                continue
            any_power = True
            kwargs = style.series_kwargs(b); kwargs["marker"] = None
            kwargs["label"] = style.SHORT[b]; kwargs["linewidth"] = 1.0
            # 0.2 s moving average: the raw 250 Hz proxy is spiky at footfall
            # and the comparison of interest is the sustained level.
            ax.plot(power["t"] - t0, _smooth(power["leg_power"], 50), **kwargs)
        if any_power:
            ax.set_ylabel("Leg power proxy (W)", fontsize=7)
            ax.set_xlim(0, None)
        else:
            missing_physics = True
            ax.text(0.5, 0.5, "physics.npz not logged\nfor this episode",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=6.5, color=style.TEXT_MUTED)
            ax.set_axis_off()

        ax = axes[r][2]
        for b in baselines:
            act = _load_table(sel["paths"][b], "actuation.csv")
            if not act:
                continue
            mask = _evaluated(act)
            kwargs = style.series_kwargs(b); kwargs["marker"] = None
            kwargs["label"] = style.SHORT[b]; kwargs["linewidth"] = 1.0
            ax.plot(act["t"][mask] - t0, _smooth(act["leg_acc_abs_mean"][mask], 10), **kwargs)
        ax.set_ylabel(r"Leg joint accel. (rad/s$^2$)", fontsize=7)

        for col in range(3):
            if r == len(cases) - 1:
                axes[r][col].set_xlabel("Time in evaluated window (s)")
        _row_header(
            axes[r],
            f"({chr(97 + r)}) {case['title']} -- seed {sel['seed']}, {sel['condition']}",
            "  ".join(
                f"{b}: $P_{{tot}}$ {sel['metrics'][b]['total_power']:.1f} W,  "
                f"$a_{{leg}}$ {sel['metrics'][b]['leg_acc_rms']:.1f},  "
                f"$E_v$ {sel['metrics'][b]['base_vxy_rmse']:.3f}"
                for b in baselines),
        )
    footer = ILLUSTRATIVE_FOOTER
    if missing_physics:
        footer += (" Leg power panels need the 250 Hz physics log; "
                   "re-run the selected episode with --log-physics.")
    footer += (" Centre panel is the leg mechanical power proxy "
               "sum|tau*qdot| at 250 Hz, 0.2 s moving average -- not battery power.")
    _figure_legend(fig, axes[0], footer, len(cases))
    fig.subplots_adjust(hspace=0.85, wspace=0.38)
    return save(fig, out_dir, name)


def e3_selected_cases(cases, out_dir, name="E3_selected_cases"):
    """One row per selected trial: EE path in the PLB y-z plane, EE error, base vx."""
    import matplotlib.pyplot as plt

    style.apply_style()
    cases = [c for c in cases if c.get("selected")]
    if not cases:
        return []
    fig, axes = plt.subplots(len(cases), 3, figsize=(style.PAGE_WIDTH, 1.95 * len(cases)),
                             squeeze=False)
    for r, case in enumerate(cases):
        sel = case["selected"]
        baselines = list(sel["metrics"].keys())
        ref_cmd = _load_table(sel["paths"][baselines[0]], "commands.csv")
        mask_c = _evaluated(ref_cmd)
        t0 = ref_cmd["t"][mask_c][0]

        ax = axes[r][0]
        ax.plot(ref_cmd["ee_cmd_y_plb"][mask_c], ref_cmd["ee_cmd_z_plb"][mask_c],
                color=COMMAND_INK, linewidth=1.1, linestyle=(0, (4, 2)),
                label="command", zorder=5)
        for b in baselines:
            states = _load_table(sel["paths"][b], "states.csv")
            mask = _evaluated(states)
            kwargs = style.series_kwargs(b); kwargs["marker"] = None
            kwargs["label"] = style.SHORT[b]; kwargs["linewidth"] = 0.9
            ax.plot(states["ee_act_y_plb"][mask], states["ee_act_z_plb"][mask], **kwargs)
        if r == len(cases) - 1:
            ax.set_xlabel("EE $y$ in PLB frame (m)")
        ax.set_ylabel("EE $z$ in PLB frame (m)", fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, axis="both")

        ax = axes[r][1]
        for b in baselines:
            states = _load_table(sel["paths"][b], "states.csv")
            mask = _evaluated(states)
            kwargs = style.series_kwargs(b); kwargs["marker"] = None
            kwargs["label"] = style.SHORT[b]; kwargs["linewidth"] = 1.0
            ax.plot(states["t"][mask] - t0, states["ee_pos_err"][mask], **kwargs)
        ax.set_ylabel("EE position error (m)", fontsize=7)

        ax = axes[r][2]
        ax.step(ref_cmd["t"][mask_c] - t0, ref_cmd["vx_cmd"][mask_c], where="post",
                color=COMMAND_INK, linewidth=1.0, linestyle=(0, (4, 2)),
                label="command", zorder=5)
        for b in baselines:
            states = _load_table(sel["paths"][b], "states.csv")
            mask = _evaluated(states)
            kwargs = style.series_kwargs(b); kwargs["marker"] = None
            kwargs["label"] = style.SHORT[b]; kwargs["linewidth"] = 1.0
            ax.plot(states["t"][mask] - t0, states["vx_act"][mask], **kwargs)
        ax.set_ylabel(r"Forward velocity $v_x$ (m/s)", fontsize=7)

        for col in (1, 2):
            if r == len(cases) - 1:
                axes[r][col].set_xlabel("Time in evaluated window (s)")
        _row_header(
            axes[r],
            f"({chr(97 + r)}) {case['title']} -- seed {sel['seed']}, {sel['condition']}",
            "  ".join(
                f"{b}: $E_p$ {sel['metrics'][b]['ee_pos_rmse']:.3f},  "
                f"$E_v$ {sel['metrics'][b]['base_vxy_rmse']:.3f}"
                for b in baselines),
        )
    _figure_legend(fig, axes[0], ILLUSTRATIVE_FOOTER
                   + " $E_p$ in m, $E_v$ in m/s.", len(cases))
    fig.subplots_adjust(hspace=0.95, wspace=0.40)
    return save(fig, out_dir, name)
