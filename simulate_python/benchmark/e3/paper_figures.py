#!/usr/bin/env python3
"""E3 paper figures: four single-column (252 pt) IEEE RA-L figures, two stacked
panels each, plus exact plotted-data CSVs.  No new data: reads data/E3_arrays.npz.

    python3 -m benchmark.e3.paper_figures

Fig. 1  T = 2 s: base XY RMSE (a) and EE position RMSE (b) vs base bound B
Fig. 2  B = 0:   base XY RMSE (a) and yaw-rate RMSE (b) vs EE update frequency
Fig. 3  B = 0, T = 2 s, isolated EE switches: signed excess base XY error (a),
        absolute EE position error (b) vs time from the switch
Fig. 4  EE fixed, B = 0.60, isolated base switches: signed excess EE position (a)
        and orientation (b) error vs time from the switch
CIs: percentile bootstrap over the 100 paired seeds (10,000 resamples, seed
20261106), same resampled seeds for all controllers. RMSEs: official 3-30 s.
Event traces (events_matched.py): per seed, mean over the isolated events
present for all three controllers (excess = trace minus
its own mean over the registered [-0.5, 0) s baseline, which equals the mean of
the per-event baseline-subtracted traces), then the seed bootstrap per sample.
No smoothing; negative excess is shown.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark.round3 import common  # noqa: E402
from benchmark.plots import style  # noqa: E402
from benchmark.e3.stats import boot_ci  # noqa: E402

E3 = os.path.join(common.BENCH, "results", "E3")
OUT = os.path.join(E3, "figures", "paper")
A = np.load(os.path.join(E3, "data", "E3_arrays.npz"))
M = np.load(os.path.join(E3, "data", "E3_event_traces_matched.npz"))
CTRL = ("C1", "C3", "C5")
NAME = {"C1": "C1 (Ours)", "C3": "C3 (RL+IK)", "C5": "C5 (Cooperative RL)"}
BLEV = A["base_levels"].astype(float)
FREQ = A["ee_freq_hz"].astype(float)
FREQ_LAB = ["Fixed", "1/6", "1/4", "1/3", "1/2"]
TT = A["trace_t"].astype(float)
SIGS = [str(s) for s in A["trace_signals"]]
PRE = int(np.sum(TT < 0))
WIDTH, HEIGHT = 3.5, 2.7
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
      "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
      "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.titlesize": 7.5,
      "lines.linewidth": 1.1, "lines.markersize": 3.2, "axes.linewidth": 0.6,
      "grid.linewidth": 0.4, "savefig.bbox": None, "xtick.major.size": 2.5,
      "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
      "axes.labelpad": 1.5, "axes.titlepad": 2.0}


def mci(x):
    m = np.mean(x, axis=-1)
    lo, hi = np.empty_like(m), np.empty_like(m)
    for i in np.ndindex(m.shape):
        lo[i], hi[i] = boot_ci(x[i])
    return m, lo, hi


def _series(ax, x, m, lo, hi, c, markers=True):
    kw = style.series_kwargs(c)
    ax.fill_between(x, lo, hi, color=kw["color"], alpha=0.2, linewidth=0)
    ax.plot(x, m, color=kw["color"], linestyle=kw["linestyle"],
            marker=kw["marker"] if markers else None, markeredgecolor="white",
            markeredgewidth=0.4, label=NAME[c])


def _fig():
    style.apply_style()
    plt.rcParams.update(RC)
    return plt.subplots(2, 1, figsize=(WIDTH, HEIGHT), sharex=True)


def _finish(fig, axes, name):
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, handlelength=2.0,
               columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.915), h_pad=0.35, pad=0.25)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=400 if ext == "png" else None)
    plt.close(fig)
    print(f"  {name}.pdf/.png")


def _rows(rows, fig, panel, xname, xs, key, m, lo, hi, n):
    for ci, c in enumerate(CTRL):
        for i, xv in enumerate(xs):
            rows.append({"figure": fig, "panel": panel, "controller": c, xname: round(float(xv), 4),
                         "quantity": key, "mean": float(m[ci, i]), "ci95_lo": float(lo[ci, i]),
                         "ci95_hi": float(hi[ci, i]), "n_seeds": n})


def fig_rmse(no, name, specs, x, xname, xticks, xlabel, sel):
    rows = []
    fig, axes = _fig()
    for ax, (panel, key, ylab, title) in zip(axes, specs):
        data = sel(A[key])                       # [ctrl, level, seed]
        assert np.isfinite(data).all()
        m, lo, hi = mci(data)
        for ci, c in enumerate(CTRL):
            _series(ax, x, m[ci], lo[ci], hi[ci], c)
        _rows(rows, no, panel, xname, x, key, m, lo, hi, data.shape[-1])
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, max(np.max(hi) * 1.12, 1e-9))
    axes[1].set_xticks(x, xticks)
    axes[1].set_xlabel(xlabel)
    axes[1].set_xlim(x[0] - 0.05 * np.ptp(x), x[-1] + 0.05 * np.ptp(x))
    _finish(fig, axes, name)
    common.write_csv(os.path.join(OUT, f"{name}_plotted_values.csv"), rows)


def fig_event(no, name, kind, bi, ti, specs, xlabel):
    """Matched isolated events (events present for all three controllers)."""
    rows = []
    fig, axes = _fig()
    tr = M[f"trace_B{bi}T{ti}"].astype(float)               # [ctrl, seed, sig, time]
    keep = np.isfinite(tr).all(axis=(0, 2, 3))              # seeds with matched events
    tr = tr[:, keep]
    for ax, (panel, sig, mode, ylab, title) in zip(axes, specs):
        x = tr[:, :, SIGS.index(sig), :]
        if mode == "excess":
            x = x - x[:, :, :PRE].mean(axis=2, keepdims=True)
        m, lo, hi = mci(np.moveaxis(x, 1, -1))               # [ctrl, time]
        for ci, c in enumerate(CTRL):
            _series(ax, TT, m[ci], lo[ci], hi[ci], c, markers=False)
        _rows(rows, no, panel, "t_from_switch_s", TT, f"{sig}_{mode}", m, lo, hi, x.shape[1])
        ax.axvline(0.0, color=style.TEXT_MUTED, linewidth=0.6, linestyle=":")
        if mode == "excess":
            ax.axhline(0.0, color=style.TEXT_PRIMARY, linewidth=0.6)
        else:
            ax.set_ylim(0, None)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
    axes[1].set_xlim(TT[0], TT[-1])
    axes[1].set_xlabel(xlabel)
    _finish(fig, axes, name)
    common.write_csv(os.path.join(OUT, f"{name}_plotted_values.csv"), rows)


def main():
    fig_rmse(1, "paper_fig1_concurrent_tracking",
             [("a", "base_xy_rmse_mps", "Base XY vel.\nRMSE (m/s)", "(a) Base tracking"),
              ("b", "ee_pos_rmse_m", "EE position\nRMSE (m)", "(b) EE tracking")],
             BLEV, "base_bound", [f"{b:.2f}" for b in BLEV],
             r"Base-motion bound $B$ (m/s, rad/s); EE target every 2 s",
             lambda a: a[:, :, 4, :])
    fig_rmse(2, "paper_fig2_ee_to_base_frequency",
             [("a", "base_xy_rmse_mps", "Base XY vel.\nRMSE (m/s)", "(a) Base translation"),
              ("b", "yaw_rmse_radps", "Yaw-rate\nRMSE (rad/s)", "(b) Base yaw")],
             FREQ, "ee_update_hz", FREQ_LAB,
             r"EE target update frequency (Hz); stationary base ($B=0$)",
             lambda a: a[:, 0, :, :])
    fig_event(3, "paper_fig3_ee_to_base_response", "ee_iso", 0, 4,
              [("a", "base_xy", "excess", "Excess base XY\nvel. error (m/s)", "(a) Induced base disturbance"),
               ("b", "ee_pos", "absolute", "EE position\nerror (m)", "(b) EE response")],
              r"Time from EE-target switch (s); $B=0$, switch every 2 s")
    fig_event(4, "paper_fig4_base_to_ee_response", "base_iso", 4, 0,
              [("a", "ee_pos", "excess", "Excess EE pos.\nerror (m)", "(a) EE position"),
               ("b", "ee_ori", "excess", "Excess EE ori.\nerror (rad)", "(b) EE orientation")],
              r"Time from base-command switch (s); EE target fixed, $B=0.60$")


if __name__ == "__main__":
    main()
