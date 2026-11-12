#!/usr/bin/env python3
"""PRELIMINARY E2 inspection figure (not the publication figure).

Four panels, one per primary metric: per-seed paired values (thin grey lines),
and the mean with its 95 % bootstrap CI for each mode. Reads only
results/E2/data/E2_per_trial_200.csv and results/E2/stats/E2_summary.csv.

    python3 -m benchmark.e2.figures
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E2 = os.path.join(BENCH, "results", "E2")
OUT = os.path.join(E2, "figures")
NAME = "E2_PRELIMINARY_primary_metrics_paired"
MODES = ("C1", "C2")
LABEL = {"C1": "Wheeled-Legged", "C2": "Legged"}
COLOR = {"C1": "#1f77b4", "C2": "#ff7f0e"}
PANELS = [("power_avg_w", "Avg. mech. power (W)"), ("energy_per_m_jpm", "Energy / progress (J/m)"),
          ("arm_acc_abs_mean", r"Arm |$\ddot q$| (rad/s$^2$)"), ("leg_acc_abs_mean", r"Leg |$\ddot q$| (rad/s$^2$)")]


def build():
    rows = list(csv.DictReader(open(os.path.join(E2, "data", "E2_per_trial_200.csv"))))
    S = {(r["controller"], r["metric"]): r for r in csv.DictReader(open(os.path.join(E2, "stats", "E2_summary.csv")))}
    seeds = sorted({int(r["seed"]) for r in rows})
    idx = {(r["controller"], int(r["seed"])): r for r in rows}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.9))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.86, wspace=0.45)
    for ax, (key, lab) in zip(axes, PANELS):
        a = np.array([float(idx[("C1", s)][key]) for s in seeds])
        b = np.array([float(idx[("C2", s)][key]) for s in seeds])
        for y0, y1 in zip(a, b):
            ax.plot([0, 1], [y0, y1], color="0.6", lw=0.3, alpha=0.4, zorder=1)
        for x, m in enumerate(MODES):
            r = S[(m, key)]
            mu, lo, hi = float(r["mean"]), float(r["ci95_lo"]), float(r["ci95_hi"])
            ax.errorbar([x], [mu], yerr=[[mu - lo], [hi - mu]], fmt="o", color=COLOR[m], ms=4, capsize=2, zorder=3)
        ax.set_xticks([0, 1], [LABEL[m] for m in MODES])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel(lab)
    fig.suptitle("PRELIMINARY - E2, 100 paired seeds (grey: per-seed pairs; dots: mean, 95% CI)", fontsize=7)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{NAME}.{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  figures/{NAME}.pdf/.png (preliminary, not for publication)")


if __name__ == "__main__":
    build()
