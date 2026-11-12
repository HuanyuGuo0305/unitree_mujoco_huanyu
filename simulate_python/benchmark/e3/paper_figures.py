#!/usr/bin/env python3
"""E3 paper figure: wbc_coordination_1x4_stacked -- one IEEE RA-L double-column
(\\textwidth x 2.95 in) figure: one row of four groups (a)-(d), each with two stacked
subplots (ACTIVE since 2026-11-11). The eight subplots are those of the earlier 2 x 2
layout (GROUPS, below), drawn by the shared _draw_group, so the plotted values are
identical; only layout, group headers and font sizes differ.
Superseded layouts (2 x 2, single-panel 1x4) were removed in the 2026-11-12 cleanup;
they are recoverable from git tag benchmark-pre-cleanup-2026-11-12. Supplementary
diagnostic figures are always written. No new simulation: reads data/E3_arrays.npz, data/E3_event_traces_matched.npz
and data/E3_event_positive_excess_traces.npz (written by e3/positive_excess.py from the
event-level logs).

    python3 -m benchmark.e3.positive_excess     # once, before this script
    python3 -m benchmark.e3.paper_figures       # 1x4 stacked + supplementary

Group contents (top / bottom), shared by the 1x4 stacked and 2 x 2 layouts:
(a) EE tracking under base motion, EE target every 2 s (T4), B = 0 ... 0.60:
    EE position RMSE (top), EE orientation RMSE (bottom).
(b) Base tracking under EE updates, B = 0, EE update Fixed ... 1/2 Hz:
    base XY RMSE (top), base yaw-rate RMSE (bottom).
(c) Isolated EE-target switches, B = 0, T = 2 s (B0T4):
    mean POSITIVE excess base XY error (top; POST-HOC), absolute EE position
    error (bottom).
(d) Isolated base-command switches, EE fixed, B = 0.60 (B4T0):
    absolute base XY error (top), mean POSITIVE excess EE position error
    (bottom; POST-HOC).
Positive excess (both cross-task panels, identical processing): per event,
max(e_i(t) - mean of e_i over the registered [-0.5, 0) s baseline, 0), taken
BEFORE averaging over events; then the mean over the seed's matched events.

RMSEs: official 3-30 s window; 100 paired seeds.
Events: matched isolated events (present for all three controllers;
events_matched.py, a post-hoc restriction), averaged within each seed first;
seeds without a matched event are dropped (B4T0: 90 of 100 seeds).
Signed excess (supplementary only) is the mean of per-event
baseline-subtracted traces. Bands: percentile bootstrap over seeds (10,000
resamples, seed 20261106; boot_ci), with the same resampled seeds for every
controller. No smoothing.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402

from benchmark.round3 import common  # noqa: E402
from benchmark.plots import style  # noqa: E402
from benchmark.e3.stats import boot_ci  # noqa: E402

E3 = os.path.join(common.BENCH, "results", "E3")
OUT = os.path.join(E3, "figures", "paper")
SUPP = os.path.join(E3, "figures", "supplementary")
A = np.load(os.path.join(E3, "data", "E3_arrays.npz"))
M = np.load(os.path.join(E3, "data", "E3_event_traces_matched.npz"))
PX = np.load(os.path.join(E3, "data", "E3_event_positive_excess_traces.npz"))
CTRL = ("C1", "C3", "C5")
LABEL = {"C1": "Unified RL (Ours)", "C3": "RL+IK", "C5": "Cooperative RL"}
# classic blue / red / green (2026-11-12; E3 figures only, shared style.SERIES untouched)
COLOR = {"C1": "#1f77b4", "C3": "#d62728", "C5": "#2ca02c"}
LS = {"C1": "-", "C3": "--", "C5": "-."}
MK = {"C1": "o", "C3": "s", "C5": "^"}
BLEV = A["base_levels"].astype(float)
FREQ = A["ee_freq_hz"].astype(float)
FREQ_LAB = ["Fixed", "1/6", "1/4", "1/3", "1/2"]
TT = M["trace_t"].astype(float)
SIGS = [str(s) for s in M["trace_signals"]]
assert np.allclose(TT, A["trace_t"]) and np.allclose(TT, PX["trace_t"])
assert SIGS == [str(s) for s in A["trace_signals"]]
assert [str(c) for c in M["controllers"]] == list(CTRL) == [str(c) for c in A["controllers"]] \
    == [str(c) for c in PX["controllers"]]
PRE = int(np.sum(TT < 0))
T_IDX = 4                      # EE update every 2 s
B_IDX = 4                      # B = 0.60
WIDTH, HEIGHT = 7.16, 5.2
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
      "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 7.5,
      "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8, "axes.titlesize": 8.5,
      "lines.linewidth": 1.1, "lines.markersize": 3.4, "axes.linewidth": 0.6,
      "grid.linewidth": 0.4, "savefig.bbox": None, "xtick.major.size": 2.5,
      "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
      "axes.labelpad": 2.0, "axes.titlepad": 2.5, "pdf.fonttype": 42}

# panel specs: (source, key/signal, mode, ylabel, y tick step, y format)
GROUPS = {
    "a": dict(title="(a) EE Tracking under Base Motion", kind="rmse", xlabel=r"Base-motion bound $B$",
              x=BLEV, xticks=[f"{b:.2f}" for b in BLEV], xname="base_bound",
              condition="EE target every 2 s (T4); B0T4-B4T4",
              sel=lambda a: a[:, :, T_IDX, :],
              panels=[("ee_pos_rmse_m", "EE position\nRMSE (m)", 0.1, "%.1f"),
                      ("ee_ori_rmse_rad", "EE orientation\nRMSE (rad)", 0.3, "%.1f")]),
    "b": dict(title="(b) Base Tracking under EE Updates", kind="rmse", xlabel="EE update frequency (Hz)",
              x=FREQ, xticks=FREQ_LAB, xname="ee_update_hz",
              condition="stationary base command B = 0; B0T0-B0T4",
              sel=lambda a: a[:, 0, :, :],
              panels=[("base_xy_rmse_mps", "Base XY\nRMSE (m/s)", 0.02, "%.2f"),
                      ("yaw_rmse_radps", "Base yaw\nRMSE (rad/s)", 0.03, "%.2f")]),
    "c": dict(title="(c) EE Command-Switch Response", kind="event", xlabel="Time from EE switch (s)",
              tag="B0T4", condition="isolated EE-target switches, B = 0, T = 2 s (B0T4), matched events",
              panels=[("base_xy", "posexcess", "Base XY error\nincrease (m/s)", 0.01, "%.2f"),
                      ("ee_pos", "absolute", "EE position\nerror (m)", 0.2, "%.1f")]),
    "d": dict(title="(d) Base Command-Switch Response", kind="event", xlabel="Time from base switch (s)",
              tag="B4T0", condition="isolated base-command switches, EE fixed, B = 0.60 (B4T0), matched events",
              panels=[("base_xy", "absolute", "Base XY\nerror (m/s)", 0.2, "%.1f"),
                      ("ee_pos", "posexcess", "EE position error\nincrease (m)", 0.004, "%.3f")]),
}


def mci(x):
    """x[..., seed] -> mean, bootstrap 95 % CI over the last axis (same resampled seeds for all rows)."""
    m = np.mean(x, axis=-1)
    lo, hi = np.empty_like(m), np.empty_like(m)
    for i in np.ndindex(m.shape):
        lo[i], hi[i] = boot_ci(x[i])
    return m, lo, hi


def matched(tag):
    tr = M[f"trace_{tag}"].astype(float)
    keep = np.isfinite(tr).all(axis=(0, 2, 3))
    return tr[:, keep], keep


def event_curves(tag, sig, mode):
    """-> [ctrl, seed, time] seed-level curves (matched events) and n seeds."""
    tr, keep = matched(tag)
    if mode == "posexcess":
        j = [str(g) for g in PX[f"signals_{tag}"]].index(sig)
        p = PX[f"pos_{tag}"][:, :, j]                      # [ctrl, seed, time]
        assert np.array_equal(np.isfinite(p).all(axis=(0, 2)), keep)
        return p[:, keep], int(keep.sum())
    x = tr[:, :, SIGS.index(sig), :]
    if mode == "excess":
        x = x - x[:, :, :PRE].mean(axis=2, keepdims=True)
    return x, int(keep.sum())


def _series(ax, x, m, lo, hi, c, markers):
    ax.fill_between(x, lo, hi, color=COLOR[c], alpha=0.16, linewidth=0, zorder=1)
    ax.plot(x, m, color=COLOR[c], linestyle=LS[c], marker=MK[c] if markers else None,
            markeredgecolor="white", markeredgewidth=0.4, zorder=3 if c == "C1" else 2,
            label=LABEL[c])


def _row(g, panel, quantity, spec, c, xname, xv, m, lo, hi, n, xlab=""):
    return {"group": g, "panel": panel, "group_title": spec["title"], "condition": spec["condition"],
            "quantity": quantity, "controller": LABEL[c], "controller_code": c,
            "x_name": xname, "x": round(float(xv), 4), "x_label": xlab,
            "mean": float(m), "ci95_lo": float(lo), "ci95_hi": float(hi), "n_seeds": n,
            "post_hoc": int(quantity.endswith("_posexcess"))}


def _yaxis(ax, step, fmt, lo=None, hi=None):
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    if lo is not None:
        ax.set_ylim(lo, hi)


def _draw_group(top, bot, g, spec, rows):
    """Draw one stacked group (top/bottom subplots) of GROUPS[g]; append plotted values to rows."""
    if spec["kind"] == "rmse":
        x = spec["x"]
        for ax, (key, ylab, step, fmt) in zip((top, bot), spec["panels"]):
            data = spec["sel"](A[key])                    # [ctrl, level, seed]
            assert np.isfinite(data).all() and data.shape[-1] == 100
            m, lo, hi = mci(data)
            for ci, c in enumerate(CTRL):
                _series(ax, x, m[ci], lo[ci], hi[ci], c, markers=True)
                for i, xv in enumerate(x):
                    rows.append(_row(g, "top" if ax is top else "bottom", key, spec, c,
                                     spec["xname"], xv, m[ci, i], lo[ci, i], hi[ci, i], 100,
                                     xlab=spec["xticks"][i]))
            ax.set_ylabel(ylab)
            _yaxis(ax, step, fmt, 0, float(np.max(hi)) * 1.10)
        bot.set_xticks(x, spec["xticks"])
        pad = 0.05 * np.ptp(x)
        bot.set_xlim(x[0] - pad, x[-1] + pad)
    else:
        for ax, (sig, mode, ylab, step, fmt) in zip((top, bot), spec["panels"]):
            xx, n = event_curves(spec["tag"], sig, mode)   # [ctrl, seed, time]
            m, lo, hi = mci(np.moveaxis(xx, 1, -1))        # [ctrl, time]
            for ci, c in enumerate(CTRL):
                _series(ax, TT, m[ci], lo[ci], hi[ci], c, markers=False)
                for i, t in enumerate(TT):
                    rows.append(_row(g, "top" if ax is top else "bottom", f"{sig}_{mode}", spec,
                                     c, "t_from_switch_s", t, m[ci, i], lo[ci, i], hi[ci, i], n))
            ax.axvline(0.0, color=style.TEXT_MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=0)
            if mode == "excess":
                ax.axhline(0.0, color=style.TEXT_SECONDARY, linewidth=0.6, zorder=0)
                _yaxis(ax, step, fmt)
            else:
                _yaxis(ax, step, fmt, 0, float(np.max(hi)) * 1.08)
            ax.set_ylabel(ylab)
        bot.set_xlim(TT[0], TT[-1])
        bot.set_xticks([-0.5, 0, 0.5, 1.0, 1.5])
        bot.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def unmatched_sensitivity(name):
    """Groups (c)-(d) with ALL isolated events (unmatched; per-controller seed sets)."""
    out = []
    resp = TT >= 0
    for g, arr, bi, ti in (("c", "trace_ee_iso", 0, T_IDX), ("d", "trace_base_iso", B_IDX, 0)):
        spec = GROUPS[g]
        for sig, mode, *_ in spec["panels"]:
            xm, nm = event_curves(spec["tag"], sig, mode)
            xm = xm.mean(axis=1)
            if mode == "posexcess":
                j = [str(g) for g in PX[f"signals_{spec['tag']}"]].index(sig)
                xu_all = PX[f"pos_unmatched_{spec['tag']}"][:, :, j]
            else:
                xu_all = A[arr][:, bi, ti, :, SIGS.index(sig), :].astype(float)
                if mode == "excess":
                    xu_all = xu_all - xu_all[..., :PRE].mean(axis=-1, keepdims=True)
            for ci, c in enumerate(CTRL):
                ok = np.isfinite(xu_all[ci]).all(axis=1)
                xu = xu_all[ci, ok].mean(axis=0)
                out.append({"group": g, "condition": spec["tag"], "quantity": f"{sig}_{mode}",
                            "controller": LABEL[c], "controller_code": c,
                            "n_seeds_matched": nm, "n_seeds_unmatched": int(ok.sum()),
                            "max_abs_diff_curve": float(np.max(np.abs(xu - xm[ci]))),
                            "peak_matched": float(np.max(xm[ci][resp])),
                            "peak_unmatched": float(np.max(xu[resp])),
                            "mean_0_1p5s_matched": float(np.mean(xm[ci][resp])),
                            "mean_0_1p5s_unmatched": float(np.mean(xu[resp]))})
    common.write_csv(os.path.join(OUT, f"{name}_unmatched_sensitivity.csv"), out)
    return out


def supplementary_signed_vs_positive(tag, xlabel, nm):
    """Diagnostic: signed-mean excess (top) vs post-hoc positive excess (bottom), cross-task signals."""
    style.apply_style()
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.0), sharex=True)
    rows = []
    sigs = [str(g) for g in PX[f"signals_{tag}"]]
    unit = {"base_xy": "m/s", "yaw": "rad/s", "ee_pos": "m", "ee_ori": "rad"}
    name = {"base_xy": "base XY", "yaw": "base yaw", "ee_pos": "EE position", "ee_ori": "EE orientation"}
    tr, keep = matched(tag)
    for j, sig in enumerate(sigs):
        signed = PX[f"signed_{tag}"][:, keep, j]
        assert np.allclose(signed, tr[:, :, SIGS.index(sig)] - tr[:, :, SIGS.index(sig), :PRE].mean(-1, keepdims=True))
        posx = PX[f"pos_{tag}"][:, keep, j]
        for i, (x, lab) in enumerate(((signed, "Signed excess"), (posx, "Positive excess"))):
            ax = axes[i, j]
            m, lo, hi = mci(np.moveaxis(x, 1, -1))
            for ci, c in enumerate(CTRL):
                _series(ax, TT, m[ci], lo[ci], hi[ci], c, markers=False)
                for k, t in enumerate(TT):
                    rows.append({"condition": tag, "signal": sig, "quantity": lab, "controller": LABEL[c],
                                 "controller_code": c, "t_from_switch_s": round(float(t), 2),
                                 "mean": float(m[ci, k]), "ci95_lo": float(lo[ci, k]),
                                 "ci95_hi": float(hi[ci, k]), "n_seeds": int(keep.sum())})
            ax.axvline(0.0, color=style.TEXT_MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=0)
            ax.axhline(0.0, color=style.TEXT_SECONDARY, linewidth=0.6, zorder=0)
            ax.set_ylabel(f"{lab}\n{name[sig]} error ({unit[sig]})")
            if i == 1:
                ax.set_xlabel(xlabel)
                ax.set_ylim(0, None)
    fig.legend([Line2D([], [], color=COLOR[c], linestyle=LS[c]) for c in CTRL], [LABEL[c] for c in CTRL],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3, h_pad=0.4)
    os.makedirs(SUPP, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(SUPP, f"{nm}.{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)
    common.write_csv(os.path.join(SUPP, f"{nm}_values.csv"), rows)
    print(f"  supplementary/{nm}.pdf/.png/_values.csv")


STACKED = "wbc_coordination_1x4_stacked"
STACKED_SIZE = (WIDTH, 2.95)          # \\textwidth (7.16 in) x 2.95 in
# single-line y labels (formatting only; quantities as in GROUPS). "Err. Inc." = error increase
# relative to the pre-switch baseline (event-level positive excess).
YLAB_1L = {"a": ("EE Pos. RMSE (m)", "EE Ori. RMSE (rad)"),
           "b": ("Base XY RMSE (m/s)", "Yaw RMSE (rad/s)"),
           "c": ("Base XY Err. Inc. (m/s)", "EE Pos. Error (m)"),
           "d": ("Base XY Error (m/s)", "EE Pos. Err. Inc. (m)")}


def _check_text_layout(fig):
    """Assert every text element lies inside the figure and no two text boxes overlap."""
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    drawn = set()
    for ax in fig.axes:                      # tick labels actually drawn (inside the view limits)
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


def build_stacked():
    """ACTIVE paper figure: one row of four groups (a)-(d), each two stacked subplots
    (the eight panels of GROUPS). Identical data and statistics to the 2x2 layout (shared
    _draw_group); only layout, group letters, single-line y labels and font sizes differ."""
    style.apply_style()
    plt.rcParams.update(RC | {"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
                              "ytick.labelsize": 7, "legend.fontsize": 8, "lines.linewidth": 1.2,
                              "lines.markersize": 3.6, "axes.labelpad": 2.0, "svg.fonttype": "path"})
    fig = plt.figure(figsize=STACKED_SIZE)
    gs = GridSpec(2, 4, figure=fig, hspace=0.08, wspace=0.40,
                  left=0.050, right=0.988, top=0.880, bottom=0.118)
    rows, groups = [], []
    for col, (g, spec) in enumerate(GROUPS.items()):
        top = fig.add_subplot(gs[0, col])
        bot = fig.add_subplot(gs[1, col], sharex=top)
        _draw_group(top, bot, g, spec, rows)
        for ax, lab in zip((top, bot), YLAB_1L[g]):
            assert "\n" not in lab
            ax.set_ylabel(lab)
        plt.setp(top.get_xticklabels(), visible=False)
        top.tick_params(axis="x", length=0)
        bot.set_xlabel(spec["xlabel"])
        fig.align_ylabels([top, bot])
        groups.append((g, top, bot))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for g, top, bot in groups:      # bold group letter at the group's upper-left corner
        x0 = min(ax.yaxis.label.get_window_extent(r).x0 for ax in (top, bot))
        y1 = top.get_window_extent(r).y1
        fx, fy = inv.transform((x0, y1))
        fig.text(fx, fy + 0.012, f"({g})", ha="left", va="bottom", fontsize=8.5, fontweight="bold")
    handles = [Line2D([], [], color=COLOR[c], linestyle=LS[c], linewidth=1.3,
                      marker=MK[c], markersize=3.4, markeredgecolor="white", markeredgewidth=0.4)
               for c in CTRL]
    fig.legend(handles, [LABEL[c] for c in CTRL], loc="upper center", ncol=3, frameon=False,
               handlelength=2.2, columnspacing=1.8, handletextpad=0.5, borderpad=0.0,
               borderaxespad=0.0, bbox_to_anchor=(0.5, 0.995))
    os.makedirs(OUT, exist_ok=True)
    fig.canvas.draw()
    _check_text_layout(fig)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUT, f"{STACKED}.{ext}"), dpi=600 if ext == "png" else None)
    plt.close(fig)
    common.write_csv(os.path.join(OUT, f"{STACKED}_values.csv"), rows)
    print(f"  {STACKED}.pdf/.svg/.png/_values.csv  ({len(rows)} rows)")
    return rows


def main():
    build_stacked()
    unmatched_sensitivity(STACKED)
    supplementary_signed_vs_positive("B4T0", "Time from base switch (s)",
                                     "supp_fig_e3_base_switch_signed_vs_positive_excess")
    supplementary_signed_vs_positive("B0T4", "Time from EE switch (s)",
                                     "supp_fig_e3_ee_switch_signed_vs_positive_excess")


if __name__ == "__main__":
    main()
