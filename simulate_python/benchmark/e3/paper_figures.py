#!/usr/bin/env python3
"""E3 paper figure ``wbc_coordination`` (bidirectional 7 x 7 benchmark).

One IEEE RA-L double-column figure (7.16 x 2.95 in): one row of four groups, each with
two stacked subplots. All groups are slices of the single 7 x 7 benchmark
(configs/E3.yaml ``figure_slices``; E.e3_slice):

  (a) base XY RMSE (top) / yaw-rate RMSE (bottom) vs B,  EE update 1/2 Hz (B0T5 ... B6T5)
  (b) EE position RMSE (top) / EE orientation RMSE (bottom), same episodes as (a)
  (c) base XY RMSE (top) / yaw-rate RMSE (bottom) vs EE update frequency, B = 0 (B0T0 ... B0T6)
  (d) EE position RMSE (top) / EE orientation RMSE (bottom), same episodes as (c)

Lines: mean over the 100 paired seeds of the 3-30 s RMSE. Bands: 95 % percentile
bootstrap confidence intervals of the mean over seeds (10,000 resamples, seed 20261106,
the same resampled seeds for every controller) -- the convention of the previous E3
figure. Bands are confidence intervals, not standard deviations. No smoothing.
EE update frequencies are drawn at equally spaced categorical positions.

Reads data/E3_arrays.npz and stats/E3_summary.csv (written by e3/analyze.py), checks that
every plotted value equals the official summary exactly, and writes
figures/paper/wbc_coordination.{pdf,svg,png}, _values.csv and _checks.json.

    python3 -m benchmark.e3.paper_figures
"""

from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

from benchmark.round3 import common  # noqa: E402
from benchmark.plots import style  # noqa: E402
from benchmark.e3.stats import summarise  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

E3 = os.path.join(common.BENCH, "results", "E3")
OUT = os.path.join(E3, "figures", "paper")
NAME = "wbc_coordination"
CTRL = ("C1", "C3", "C5")
LABEL = {"C1": "Unified RL (Ours)", "C3": "RL+IK", "C5": "Cooperative RL"}
COLOR = {"C1": "#1f77b4", "C3": "#d62728", "C5": "#2ca02c"}
LS = {"C1": "-", "C3": "--", "C5": "-."}
MK = {"C1": "o", "C3": "s", "C5": "^"}
SIZE = (7.16, 2.95)
CFG, _ = E.load_config("E3")
BASE_TOP, BASE_BOT = ("base_xy_rmse_mps", "Base XY RMSE (m/s)"), ("yaw_rmse_radps", "Yaw RMSE (rad/s)")
EE_TOP, EE_BOT = ("ee_pos_rmse_m", "EE Pos. RMSE (m)"), ("ee_ori_rmse_rad", "EE Ori. RMSE (rad)")
XB, XF = "Base-motion bound $B$", "EE update frequency (Hz)"
GROUPS = {  # group: (slice, panels, x label)
    "a": ("vs_base_bound", (BASE_TOP, BASE_BOT), XB),
    "b": ("vs_base_bound", (EE_TOP, EE_BOT), XB),
    "c": ("vs_ee_update", (BASE_TOP, BASE_BOT), XF),
    "d": ("vs_ee_update", (EE_TOP, EE_BOT), XF),
}
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
      "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7,
      "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 8,
      "lines.linewidth": 1.2, "lines.markersize": 3.6, "axes.linewidth": 0.6,
      "grid.linewidth": 0.4, "savefig.bbox": None, "xtick.major.size": 2.5,
      "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
      "axes.labelpad": 2.0, "pdf.fonttype": 42, "svg.fonttype": "path"}


def slice_axis(name):
    """-> (condition ids, x positions, x tick labels, x name) of a figure slice."""
    cells = E.e3_slice(CFG, name)
    if name == "vs_base_bound":
        x = np.array([float(CFG["base"]["levels"][int(c[1])]) for c in cells])
        return cells, x, [f"{v:g}" for v in x], "base_bound"
    labs = [str(CFG["ee"]["update_hz_labels"][int(c[3])]) for c in cells]
    return cells, np.arange(len(cells), dtype=float), labs, "ee_update_hz"


def load_summary():
    out = {}
    with open(os.path.join(E3, "stats", "E3_summary.csv")) as h:
        for r in csv.DictReader(h):
            out[(r["controller"], r["condition"], r["metric"])] = r
    return out


def _check_text_layout(fig):
    """Assert every drawn text element lies inside the figure and no two text boxes overlap."""
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    drawn = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis._update_ticks():
                drawn.update((tk.label1, tk.label2))
    ticklabels = {t for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)
                  for tk in axis.get_major_ticks() + axis.get_minor_ticks() for t in (tk.label1, tk.label2)}
    boxes = []
    for t in fig.findobj(matplotlib.text.Text):
        if (t in ticklabels and t not in drawn) or not t.get_visible() or not t.get_text().strip():
            continue
        b = t.get_window_extent(r)
        assert b.x0 >= -0.5 and b.y0 >= -0.5 and b.x1 <= W + 0.5 and b.y1 <= H + 0.5, \
            f"clipped: {t.get_text()!r} {b}"
        boxes.append((t.get_text(), b))
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1:]:
            assert not a.overlaps(b), f"overlap: {ta!r} / {tb!r}"
    return len(boxes)


def build():
    A = np.load(os.path.join(E3, "data", "E3_arrays.npz"))
    assert [str(c) for c in A["controllers"]] == list(CTRL)
    S = load_summary()
    style.apply_style()
    plt.rcParams.update(RC)
    fig = plt.figure(figsize=SIZE)
    gs = GridSpec(2, 4, figure=fig, hspace=0.08, wspace=0.42,
                  left=0.056, right=0.990, top=0.880, bottom=0.128)
    rows, groups, maxdiff = [], [], 0.0
    for col, (g, (sl, panels, xlabel)) in enumerate(GROUPS.items()):
        cells, x, xt, xname = slice_axis(sl)
        top = fig.add_subplot(gs[0, col])
        bot = fig.add_subplot(gs[1, col], sharex=top)
        for ax, (key, ylab) in zip((top, bot), panels):
            hi_max = 0.0
            for ci, c in enumerate(CTRL):
                st = [summarise(A[key][ci, int(cc[1]), int(cc[3])]) for cc in cells]
                m = np.array([s["mean"] for s in st])
                lo = np.array([s["ci95_lo"] for s in st])
                hi = np.array([s["ci95_hi"] for s in st])
                for i, cc in enumerate(cells):    # plotted value == official statistic, exactly
                    off = S[(c, cc, key)]
                    for k, v in (("mean", m[i]), ("ci95_lo", lo[i]), ("ci95_hi", hi[i])):
                        d = abs(float(off[k]) - float(v))
                        maxdiff = max(maxdiff, d)
                        assert d == 0.0, (c, cc, key, k, off[k], v)
                    rows.append({"group": g, "panel": "top" if ax is top else "bottom", "slice": sl,
                                 "metric": key, "controller": LABEL[c], "controller_code": c,
                                 "condition": cc, "x_name": xname, "x_label": xt[i],
                                 "base_bound": float(off["base_bound"]), "ee_update_hz": float(off["ee_update_hz"]),
                                 "mean": m[i], "ci95_lo": lo[i], "ci95_hi": hi[i], "sd": float(off["sd"]),
                                 "n_scored": int(off["n_scored"]), "n_terminated_early": int(off["n_terminated_early"])})
                ax.fill_between(x, lo, hi, color=COLOR[c], alpha=0.16, linewidth=0, zorder=1)
                ax.plot(x, m, color=COLOR[c], linestyle=LS[c], marker=MK[c], markeredgecolor="white",
                        markeredgewidth=0.4, zorder=3 if c == "C1" else 2)
                hi_max = max(hi_max, float(hi.max()))
            ax.set_ylabel(ylab)
            assert "\n" not in ylab
            ax.set_ylim(0, hi_max * 1.10)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        bot.set_xticks(x, xt)
        pad = 0.06 * np.ptp(x)
        bot.set_xlim(x[0] - pad, x[-1] + pad)
        plt.setp(top.get_xticklabels(), visible=False)
        top.tick_params(axis="x", length=0)
        bot.set_xlabel(xlabel)
        fig.align_ylabels([top, bot])
        groups.append((g, top, bot))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for g, top, bot in groups:      # bold group letter only, at the group's upper-left corner
        x0 = min(ax.yaxis.label.get_window_extent(r).x0 for ax in (top, bot))
        fx, fy = inv.transform((x0, top.get_window_extent(r).y1))
        fig.text(fx, fy + 0.012, f"({g})", ha="left", va="bottom", fontsize=8.5, fontweight="bold")
    handles = [Line2D([], [], color=COLOR[c], linestyle=LS[c], linewidth=1.3, marker=MK[c], markersize=3.4,
                      markeredgecolor="white", markeredgewidth=0.4) for c in CTRL]
    fig.legend(handles, [LABEL[c] for c in CTRL], loc="upper center", ncol=3, frameon=False,
               handlelength=2.2, columnspacing=1.8, handletextpad=0.5, borderpad=0.0,
               borderaxespad=0.0, bbox_to_anchor=(0.5, 0.995))
    fig.canvas.draw()
    n_text = _check_text_layout(fig)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUT, f"{NAME}.{ext}"), dpi=600 if ext == "png" else None)
    plt.close(fig)
    common.write_csv(os.path.join(OUT, f"{NAME}_values.csv"), rows)
    checks = {"figure": f"{NAME}.pdf", "size_in": list(SIZE), "n_values": len(rows),
              "slices": {g: E.e3_slice(CFG, sl) for g, (sl, *_ ) in GROUPS.items()},
              "max_abs_diff_vs_E3_summary": maxdiff, "text_elements_checked_no_overlap": n_text,
              "bands": "95 % percentile bootstrap CI of the mean over seeds (10,000 resamples, seed 20261106, "
                       "paired resampling); not standard deviations",
              "lines": "mean over seeds of the 3-30 s RMSE (terminated episodes scored over their valid window)"}
    json.dump(checks, open(os.path.join(OUT, f"{NAME}_checks.json"), "w"), indent=1)
    print(f"  {NAME}.pdf/.svg/.png/_values.csv ({len(rows)} rows); max |plotted - official| = {maxdiff}")
    return rows


def main():
    build()


if __name__ == "__main__":
    main()
