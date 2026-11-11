"""Shared plotting style for the RA-L figures.

Colour parameters come from the dataviz skill's validated reference palette
(``references/palette.md``), used unchanged: categorical slots 1-3 in fixed
order for the three baselines, and the single-hue blue ramp for sequential
magnitude. The slots are taken in fixed order and never cycled; a fourth
baseline would take slot 4, not a generated hue.

The palette validator (``scripts/validate_palette.js``) could not be run in
this environment -- no JavaScript runtime is installed -- so the palette is the
skill's pre-validated default instance rather than a new combination.

These are static print figures for an IEEE double-column paper, so there is no
hover layer; identity is carried by a legend that is always present plus direct
end-of-line labels, never by colour alone. Every panel has a single y axis.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots 1, 2, 3 (blue, orange, aqua), light mode, fixed order.
SERIES = {
    "C1": "#2a78d6",
    "C3": "#eb6834",
    "C5": "#1baf7a",
    "C2": "#eda100",   # slot 4, for the E2 pair
    "C4": "#e87ba4",   # slot 5
    "C6": "#008300",   # slot 6
}
MARKERS = {"C1": "o", "C3": "s", "C5": "^", "C2": "D", "C4": "v", "C6": "P"}
# Secondary encoding so identity is never colour-alone in print or for CVD.
LINESTYLES = {"C1": "-", "C3": "--", "C5": "-.", "C2": "--", "C4": ":", "C6": "-."}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8982"
SURFACE = "#ffffff"
GRID = "#dededa"

# Single-hue blue ramp (steps 100 -> 700) for sequential magnitude.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("bench_blue", BLUE_RAMP)

# IEEE double-column: a one-column figure is 3.5 in wide, two columns 7.16 in.
COL_WIDTH = 3.5
PAGE_WIDTH = 7.16

LABELS = {
    "C1": "C1 Unified RL (ours)",
    "C2": "C2 Unified RL, fixed wheel",
    "C3": "C3 RL + IK",
    "C4": "C4 RL + IK, fixed wheel",
    "C5": "C5 Dual-PPO",
    "C6": "C6 Dual-PPO, fixed wheel",
}
SHORT = {"C1": "C1", "C2": "C2", "C3": "C3", "C4": "C4", "C5": "C5", "C6": "C6"}


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.labelcolor": TEXT_PRIMARY,
            "text.color": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            # recessive grid and axes
            "axes.edgecolor": "#b9b8b1",
            "axes.linewidth": 0.6,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def series_kwargs(baseline: str) -> dict:
    return {
        "color": SERIES[baseline],
        "marker": MARKERS[baseline],
        "linestyle": LINESTYLES[baseline],
        "markeredgecolor": SURFACE,
        "markeredgewidth": 0.6,
        "label": LABELS[baseline],
    }


def plot_with_ci(ax, x, mean, ci_low, ci_high, baseline: str, direct_label=True):
    """One series: line + markers + a 95% CI band, with a direct end label."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    kwargs = series_kwargs(baseline)
    ax.fill_between(
        x, np.asarray(ci_low, dtype=float), np.asarray(ci_high, dtype=float),
        color=kwargs["color"], alpha=0.14, linewidth=0,
    )
    ax.plot(x, mean, **kwargs)
    if direct_label and len(x):
        ok = np.isfinite(mean)
        if ok.any():
            i = int(np.nonzero(ok)[0][-1])
            ax.annotate(
                SHORT[baseline],
                xy=(x[i], mean[i]),
                xytext=(4, 0),
                textcoords="offset points",
                color=TEXT_SECONDARY,   # text ink, not the series colour
                fontsize=7,
                va="center",
                clip_on=False,
            )


def save(fig, out_dir: str, name: str, formats=("pdf", "png")):
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in formats:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path)
        paths.append(path)
        print(f"  wrote {os.path.relpath(path)}")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------
# Diverging scale, for signed quantities whose zero point carries meaning
# (a paired advantage, a degradation increase/decrease).
#
# Built to the dataviz skill's diverging rule: the two poles are its diverging
# pair (blue <-> red), the midpoint is its neutral gray #f0efec, and the arms
# have equal step counts. The red arm was DERIVED, not eyeballed: each step
# takes the OKLCH lightness and chroma of the corresponding blue step with the
# hue of the skill's categorical red (#e34948, OKLCH H = 24.9 deg), so the two
# arms are lightness-matched. Verified numerically: lightness is strictly
# monotonic along each arm and the neutral midpoint (L = 0.952) is lighter than
# either arm's first step, which is the stated check for a diverging ramp (the
# categorical validator does not apply to one and FAILs by design).
#
# Blue = positive = the reference method (C1) ahead; red = negative = the rival
# ahead. Sign is also written as a number in every heatmap cell, so polarity is
# never carried by hue alone.
# --------------------------------------------------------------------------
DIVERGING_STEPS = [
    "#621b1a", "#892b2a", "#b13f3c", "#c74845", "#dd716a", "#ea9a93", "#fad6d2",
    "#f0efec",
    "#cde2fb", "#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b",
]
DIVERGING = LinearSegmentedColormap.from_list("bench_div", DIVERGING_STEPS)
NEUTRAL_MID = "#f0efec"

# Sequential steps for an ORDERED set of discrete lines (e.g. the three
# intensity levels). Ordinal use, so the step nearest the light surface must
# still clear 2:1 contrast: start at step 250 (#86b6ef), never at step 100.
ORDINAL_BLUE = ["#86b6ef", "#2a78d6", "#104281"]
ORDINAL_ORANGE = ["#f0a883", "#eb6834", "#8f3517"]


def ordinal_kwargs(index: int, n: int, ramp=None, marker="o", linestyle="-") -> dict:
    ramp = ORDINAL_BLUE if ramp is None else ramp
    step = ramp[min(index, len(ramp) - 1)] if n <= len(ramp) else ramp[-1]
    return {
        "color": step, "marker": marker, "linestyle": linestyle,
        "markeredgecolor": SURFACE, "markeredgewidth": 0.6,
    }
