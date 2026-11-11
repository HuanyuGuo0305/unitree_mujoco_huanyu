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
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from benchmark.plots import style
from benchmark.round3 import common


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
    baselines = {"E1": ["C1", "C3", "C5"], "E2": ["C1", "C2"]}[experiment]
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
    only = args.only.split(",") if args.only else None

    def want(tag):
        return only is None or tag in only

    if want("E1"):
        e1_summary_figure(os.path.join(args.out, "E1"), fig_dir)
    if want("E2"):
        e2_summary_figure(os.path.join(args.out, "E2"), fig_dir)
    if want("selected"):
        for experiment in ("E1", "E2"):
            selected_cases_figure(experiment, os.path.join(args.out, experiment), fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
