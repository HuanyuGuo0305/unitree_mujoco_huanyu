#!/usr/bin/env python3
"""Compact single-column RA-L figures for E3 v3 (no new data; reads the
validated analysis arrays).

    python3 -m benchmark.e3v3.paper_figures

Fig. 1  (a) base XY velocity RMSE vs EE update frequency at B = 0.60
        (b) EE position RMSE vs base bound B at T = 2 s
Fig. 2  base XY (a) and yaw-rate (b) tracking error vs time from an isolated
        EE-pose switch, stationary base (B = 0), T = 2 s

Aggregation: for each seed, the line and circle episodes are averaged first
(equal weight); the 95 % CI is a percentile bootstrap over the 100 seeds
(10,000 resamples, seed 20261106), and the same resampled seed indices are
used for every controller and level (paired resampling). RMSEs use the
official 3-30 s window. Event traces: per episode, the mean over its isolated
EE switches, then the pattern average within the seed, then the seed
bootstrap per time sample -- events are never treated as independent trials.
No smoothing. Separate line / circle versions are written to paper/patterns/.
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
OUT = os.path.join(V3, "figures", "paper")
A = np.load(os.path.join(V3, "data", "E3v3_arrays.npz"))
CTRL = ("C1", "C3", "C5")
NAME = {"C1": "C1 (Ours)", "C3": "C3 (RL+IK)", "C5": "C5 (Cooperative RL)"}
BLEV = A["base_levels"].astype(float)
TLAB = [str(t) for t in A["ee_levels"]]
FREQ = np.array([0.0, 1 / 6, 1 / 4, 1 / 3, 1 / 2])
TT = A["trace_t"].astype(float)
SIGS = [str(s) for s in A["trace_signals"]]
WIDTH = 3.5  # in = IEEEtran \columnwidth

RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.titlesize": 8, "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "savefig.bbox": None,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
}


def sel(key, pat):
    """[ctrl, B, T, seed] for 'pooled' (per-seed mean of line and circle) or a pattern."""
    x = A[key]
    return x.mean(axis=1) if pat == "pooled" else x[:, ["line", "circle"].index(pat)]


def mci(x):
    m = np.mean(x, axis=-1)
    lo, hi = np.empty_like(m), np.empty_like(m)
    for i in np.ndindex(m.shape):
        lo[i], hi[i] = boot_ci(x[i])
    return m, lo, hi


def series(ax, x, m, lo, hi, c, markers=True):
    kw = style.series_kwargs(c)
    ax.fill_between(x, lo, hi, color=kw["color"], alpha=0.18, linewidth=0)
    ax.plot(x, m, color=kw["color"], linestyle=kw["linestyle"],
            marker=kw["marker"] if markers else None, markeredgecolor="white",
            markeredgewidth=0.5, label=NAME[c])


def fig1(pat, path, truncate=False):
    with plt.rc_context(style.plt.rcParams.copy()):
        style.apply_style()
        plt.rcParams.update(RC)
        fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 3.2))
        bx = sel("base_xy_rmse_mps", pat)          # [ctrl, B, T, seed]
        for ci, c in enumerate(CTRL):
            series(a, FREQ, *mci(bx[ci, 4]), c)
        a.set_xticks(FREQ, ["Fixed", "1/6", "1/4", "1/3", "1/2"])
        a.set_xlabel("EE command update frequency (Hz)")
        a.set_ylabel("Base XY vel. RMSE (m/s)")
        a.set_xlim(-0.03, 0.53)
        a.set_title("(a) Base tracking, base bound $B=0.60$", loc="left", pad=3)
        ep = sel("ee_pos_rmse_m", pat)
        for ci, c in enumerate(CTRL):
            series(b, BLEV, *mci(ep[ci, :, 4]), c)
        b.set_xticks(BLEV, [f"{v:.2f}" for v in BLEV])
        b.set_xlabel("Base-motion bound $B$ (m/s, rad/s)")
        b.set_ylabel("EE position RMSE (m)")
        b.set_xlim(-0.03, 0.63)
        b.set_title("(b) EE tracking, EE command interval 2 s", loc="left", pad=3)
        if truncate:
            b.set_ylim(0.080, 0.100)
            b.set_ylabel("EE position RMSE (m)\n[axis starts at 0.08]")
        else:
            b.set_ylim(0, 0.115)
        a.set_ylim(0, 0.205)
        h, lab = a.get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, handlelength=2.2,
                   columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.9)
        return _save(fig, path)


def fig2(pat, path, ti=4):
    with plt.rc_context(style.plt.rcParams.copy()):
        style.apply_style()
        plt.rcParams.update(RC)
        fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 3.2), sharex=True)
        tr = A["trace_ee_iso"].astype(float)       # [ctrl, pat, B, T, seed, sig, time]
        tr = tr.mean(axis=1) if pat == "pooled" else tr[:, ["line", "circle"].index(pat)]
        for ax, s, lab in ((a, "base_xy", "Base XY vel. error (m/s)"),
                           (b, "yaw", "Yaw-rate error (rad/s)")):
            si = SIGS.index(s)
            for ci, c in enumerate(CTRL):
                x = tr[ci, 0, ti, :, si, :]           # [seed, time], B = 0
                assert np.isfinite(x).all()
                m, lo, hi = mci(x.T)
                series(ax, TT, m, lo, hi, c, markers=False)
            ax.axvline(0.0, color=style.TEXT_MUTED, linewidth=0.6, linestyle=":")
            ax.set_ylabel(lab)
            ax.set_ylim(bottom=0)
            ax.set_xlim(TT[0], TT[-1])
        b.set_xlabel("Time from EE-pose command switch (s)")
        a.set_title(f"(a) Base XY velocity, stationary base ($B=0$), EE interval {TLAB[ti].replace('s', ' s')}",
                    loc="left", pad=3)
        b.set_title("(b) Yaw rate", loc="left", pad=3)
        h, lab = a.get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, handlelength=2.2,
                   columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=0.6)
        return _save(fig, path)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300 if ext == "png" else None)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  {os.path.relpath(path, common.BENCH)}.pdf/.png  {w:.2f} x {h:.2f} in")
    return path


def numbers():
    """Values plotted, for the caption/text and for checking."""
    rows = []
    bx, ep = sel("base_xy_rmse_mps", "pooled"), sel("ee_pos_rmse_m", "pooled")
    for ci, c in enumerate(CTRL):
        for ti in range(5):
            m, lo, hi = mci(bx[ci, 4, ti][None])
            rows.append({"figure": "1a", "controller": c, "B": 0.6, "ee_level": TLAB[ti],
                         "metric": "base_xy_rmse_mps", "mean": float(m[0]), "ci_lo": float(lo[0]),
                         "ci_hi": float(hi[0]), "n_seeds": bx.shape[-1]})
        for bi in range(5):
            m, lo, hi = mci(ep[ci, bi, 4][None])
            rows.append({"figure": "1b", "controller": c, "B": float(BLEV[bi]), "ee_level": "2s",
                         "metric": "ee_pos_rmse_m", "mean": float(m[0]), "ci_lo": float(lo[0]),
                         "ci_hi": float(hi[0]), "n_seeds": ep.shape[-1]})
    tr = A["trace_ee_iso"].astype(float).mean(axis=1)
    for s in ("base_xy", "yaw"):
        si = SIGS.index(s)
        for ci, c in enumerate(CTRL):
            x = tr[ci, 0, 4, :, si, :]
            m = x.mean(0)
            pre = m[TT < 0].mean()
            k = int(np.argmax(m[TT >= 0]))
            rows.append({"figure": "2", "controller": c, "B": 0.0, "ee_level": "2s",
                         "metric": f"{s}_trace", "mean": float(pre), "ci_lo": float(m[TT >= 0][k]),
                         "ci_hi": float(TT[TT >= 0][k]), "n_seeds": x.shape[0],
                         "note": "mean = pre-switch level; ci_lo = peak of mean trace; ci_hi = peak time [s]"})
    common.write_csv(os.path.join(OUT, "paper_figure_values.csv"), rows)


def main():
    fig1("pooled", os.path.join(OUT, "E3v3_paper_fig1_dynamic_tracking"))
    fig2("pooled", os.path.join(OUT, "E3v3_paper_fig2_ee_switch_base_disturbance"))
    fig1("pooled", os.path.join(OUT, "alternatives", "E3v3_paper_fig1_dynamic_tracking_truncated_axis_b"), truncate=True)
    for pat in ("line", "circle"):
        fig1(pat, os.path.join(OUT, "patterns", f"E3v3_paper_fig1_dynamic_tracking_{pat}"))
        fig2(pat, os.path.join(OUT, "patterns", f"E3v3_paper_fig2_ee_switch_base_disturbance_{pat}"))
    numbers()


if __name__ == "__main__":
    main()
