#!/usr/bin/env python3
"""E3 paper figure: one IEEE RA-L double-column (\\textwidth) figure,
"Dynamic whole-body coordination", 2 x 2 groups of two stacked panels, plus a
supplementary diagnostic figure. No new simulation: reads data/E3_arrays.npz,
data/E3_event_traces_matched.npz and data/E3_base_event_positive_excess_traces.npz
(written by e3/positive_excess.py from the event-level logs).

    python3 -m benchmark.e3.positive_excess     # once, before this script
    python3 -m benchmark.e3.paper_figures

(a) EE tracking under base motion, EE target every 2 s (T4), B = 0 ... 0.60:
    EE position RMSE (top), EE orientation RMSE (bottom).
(b) Base tracking under EE updates, B = 0, EE update Fixed ... 1/2 Hz:
    base XY RMSE (top), base yaw-rate RMSE (bottom).
(c) Isolated EE-target switches, B = 0, T = 2 s (B0T4):
    absolute EE position error (top), signed excess base XY error (bottom).
(d) Isolated base-command switches, EE fixed, B = 0.60 (B4T0):
    absolute base XY error (top), mean POSITIVE excess EE position error
    (bottom; POST-HOC: per event max(e_i(t) - pre-switch mean, 0), then averaged).

RMSEs: official 3-30 s window; 100 paired seeds.
Events: matched isolated events (present for all three controllers;
events_matched.py, a post-hoc restriction), averaged within each seed first;
seeds without a matched event are dropped (B4T0: 90 of 100 seeds).
Signed excess in (c): the seed-mean trace minus its own mean over the
registered [-0.5, 0) s baseline (equal to the mean of per-event
baseline-subtracted traces). Bands: percentile bootstrap over seeds (10,000
resamples, seed 20261106; boot_ci), with the same resampled seeds for every
controller. No smoothing; negative signed excess is kept.
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
NAME = "paper_fig_e3_dynamic_coordination_combined"
A = np.load(os.path.join(E3, "data", "E3_arrays.npz"))
M = np.load(os.path.join(E3, "data", "E3_event_traces_matched.npz"))
PX = np.load(os.path.join(E3, "data", "E3_base_event_positive_excess_traces.npz"))
CTRL = ("C1", "C3", "C5")
LABEL = {"C1": "Unified RL (Ours)", "C3": "RL+IK", "C5": "Cooperative RL"}
COLOR = {c: style.SERIES[c] for c in CTRL}
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
              panels=[("ee_pos", "absolute", "EE position\nerror (m)", 0.2, "%.1f"),
                      ("base_xy", "excess", "Excess base\nXY error (m/s)", 0.01, "%.2f")]),
    "d": dict(title="(d) Base Command-Switch Response", kind="event", xlabel="Time from base switch (s)",
              tag="B4T0", condition="isolated base-command switches, EE fixed, B = 0.60 (B4T0), matched events",
              panels=[("base_xy", "absolute", "Base XY\nerror (m/s)", 0.2, "%.1f"),
                      ("ee_pos", "posexcess", "Positive excess EE\nposition error (m)", 0.004, "%.3f")]),
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
        assert tag == "B4T0" and sig == "ee_pos"
        p = PX[f"pos_{tag}"][:, :, 0]                      # [ctrl, seed, time], signal 0 = ee_pos
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
            "post_hoc": int(quantity == "ee_pos_posexcess")}


def _yaxis(ax, step, fmt, lo=None, hi=None):
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    if lo is not None:
        ax.set_ylim(lo, hi)


def build():
    style.apply_style()
    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 0.44, 1, 1], hspace=0.11, wspace=0.24,
                  left=0.083, right=0.992, top=0.912, bottom=0.077)
    pos = {"a": (0, 0), "b": (0, 1), "c": (3, 0), "d": (3, 1)}
    rows = []
    for g, spec in GROUPS.items():
        r0, col = pos[g]
        top = fig.add_subplot(gs[r0, col])
        bot = fig.add_subplot(gs[r0 + 1, col], sharex=top)
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
        plt.setp(top.get_xticklabels(), visible=False)
        top.tick_params(axis="x", length=0)
        bot.set_xlabel(spec["xlabel"])
        top.set_title(spec["title"], loc="left", fontweight="bold")
    for col in (0, 1):
        fig.align_ylabels([a for a in fig.axes if a.get_subplotspec().colspan.start == col])
    handles = [Line2D([], [], color=COLOR[c], linestyle=LS[c], linewidth=1.3) for c in CTRL]
    fig.legend(handles, [LABEL[c] for c in CTRL], loc="upper center", ncol=3, frameon=False,
               handlelength=2.6, columnspacing=2.4, handletextpad=0.6, bbox_to_anchor=(0.5, 1.0),
               borderaxespad=0.2)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{NAME}.{ext}"), dpi=400 if ext == "png" else None)
    plt.close(fig)
    common.write_csv(os.path.join(OUT, f"{NAME}_values.csv"), rows)
    print(f"  {NAME}.pdf/.png/_values.csv  ({len(rows)} rows)")
    return rows


def unmatched_sensitivity():
    """Groups (c)-(d) with ALL isolated events (unmatched; per-controller seed sets)."""
    out = []
    resp = TT >= 0
    for g, arr, bi, ti in (("c", "trace_ee_iso", 0, T_IDX), ("d", "trace_base_iso", B_IDX, 0)):
        spec = GROUPS[g]
        for sig, mode, *_ in spec["panels"]:
            xm, nm = event_curves(spec["tag"], sig, mode)
            xm = xm.mean(axis=1)
            if mode == "posexcess":
                xu_all = PX[f"pos_unmatched_{spec['tag']}"][:, :, 0]
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
    common.write_csv(os.path.join(OUT, f"{NAME}_unmatched_sensitivity.csv"), out)
    return out


def supplementary_signed_vs_positive():
    """Diagnostic: the v1 signed-mean excess and the post-hoc positive excess, (d) condition."""
    style.apply_style()
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.0), sharex=True)
    rows = []
    for j, (sig, unit) in enumerate((("ee_pos", "m"), ("ee_ori", "rad"))):
        si = SIGS.index(sig)
        tr, keep = matched("B4T0")
        signed = tr[:, :, si] - tr[:, :, si, :PRE].mean(axis=-1, keepdims=True)
        posx = PX["pos_B4T0"][:, keep, j]
        for i, (x, lab) in enumerate(((signed, "Signed excess"), (posx, "Positive excess"))):
            ax = axes[i, j]
            m, lo, hi = mci(np.moveaxis(x, 1, -1))
            for ci, c in enumerate(CTRL):
                _series(ax, TT, m[ci], lo[ci], hi[ci], c, markers=False)
                for k, t in enumerate(TT):
                    rows.append({"signal": sig, "quantity": lab, "controller": LABEL[c], "controller_code": c,
                                 "t_from_switch_s": round(float(t), 2), "mean": float(m[ci, k]),
                                 "ci95_lo": float(lo[ci, k]), "ci95_hi": float(hi[ci, k]),
                                 "n_seeds": int(keep.sum())})
            ax.axvline(0.0, color=style.TEXT_MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=0)
            ax.axhline(0.0, color=style.TEXT_SECONDARY, linewidth=0.6, zorder=0)
            ax.set_ylabel(f"{lab}\nEE {'position' if sig == 'ee_pos' else 'orientation'} error ({unit})")
            if i == 1:
                ax.set_xlabel("Time from base switch (s)")
                ax.set_ylim(0, None)
    fig.legend([Line2D([], [], color=COLOR[c], linestyle=LS[c]) for c in CTRL], [LABEL[c] for c in CTRL],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3, h_pad=0.4)
    os.makedirs(SUPP, exist_ok=True)
    nm = "supp_fig_e3_base_switch_signed_vs_positive_excess"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(SUPP, f"{nm}.{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)
    common.write_csv(os.path.join(SUPP, f"{nm}_values.csv"), rows)
    print(f"  supplementary/{nm}.pdf/.png/_values.csv")


def main():
    build()
    unmatched_sensitivity()
    supplementary_signed_vs_positive()


if __name__ == "__main__":
    main()
