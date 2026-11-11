"""Round-2 tables: the 60-seed official tables and the illustrative-case tables.

Two tables per experiment, deliberately:

* a **compact** one that fits an IEEE double column and preserves the
  experiment's own structure (E1's 2x2 Still/Moving x Fixed/Random), carrying
  the means and the statistical bolding; and
* an **intervals companion** carrying every 95% CI and both paired differences,
  because those cannot be fitted into the compact cells without making the
  compact table unreadable. Nothing is reported only as a point estimate: the
  companion is a cross-reference away and every number is also in the CSVs.

Condition labels are printed once per group with ``\\multirow``, never repeated
on every line.

Bolding rule, unchanged from round 1: bold marks the SET of methods not
statistically separated from the point-estimate leader by a paired bootstrap
95% CI. A numerically smallest value is never bolded alone when the
uncertainty does not support it.

The illustrative-case tables mark C1's within-trial advantages in bold and say
in the caption that this is a single-trial comparison with no uncertainty
attached -- the three selected seeds are never presented as an average.
"""

from __future__ import annotations

import os

import numpy as np

from benchmark.core import stats
from benchmark.plots.tables import E1_METRICS, E1_PANELS, SHORT_LABEL, _by, _fmt, _write


def _summ(values, cfg_stats):
    return stats.summarize(
        values,
        n_resamples=cfg_stats["bootstrap_resamples"],
        seed=cfg_stats["bootstrap_seed"],
        confidence=cfg_stats["confidence"],
    )


def _paired(a, b, cfg_stats):
    return stats.paired_difference(
        a, b,
        n_resamples=cfg_stats["bootstrap_resamples"],
        seed=cfg_stats["bootstrap_seed"],
        confidence=cfg_stats["confidence"],
    )


def _ci(lo, hi, digits):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "--"
    return f"[{lo:.{digits}f},\\,{hi:.{digits}f}]"


def _signed(value, digits, significant):
    """Signed number in math mode, so a negative reads as a minus not a hyphen."""
    if not np.isfinite(value):
        return "--"
    body = f"{abs(value):.{digits}f}"
    sign = "+" if value >= 0 else "-"
    return "$" + sign + body + (r"^{*}" if significant else "") + "$"


# ==========================================================================
# E1
# ==========================================================================
def e1_official_table(rows, out_dir, cfg_stats, baselines=("C1", "C3", "C5"),
                      name="E1_official_60seeds"):
    baselines = list(baselines)
    n_seeds = len(sorted({r["seed"] for r in rows}))
    cache = {}
    for condition, _, _ in E1_PANELS:
        grouped = _by(rows, condition, baselines)
        entry = {"survival": {}, "cells": {}, "best": {}, "paired": {}}
        for baseline in baselines:
            items = list(grouped[baseline].values())
            entry["survival"][baseline] = (
                sum(int(r["survived"]) for r in items), len(items))
        for metric, _, _ in E1_METRICS:
            per_method = {
                b: {s: r[metric] for s, r in grouped[b].items() if metric in r}
                for b in baselines
            }
            entry["best"][metric] = stats.best_methods(
                per_method, lower_is_better=True,
                n_resamples=cfg_stats["bootstrap_resamples"],
                seed=cfg_stats["bootstrap_seed"])
            for baseline in baselines:
                entry["cells"][(baseline, metric)] = _summ(
                    list(per_method[baseline].values()), cfg_stats)
            for rival in baselines[1:]:
                entry["paired"][(metric, rival)] = _paired(
                    per_method[baselines[0]], per_method[rival], cfg_stats)
        cache[condition] = entry

    # ---------------- compact 2x2 table*
    def sub_table(condition, tag, title):
        entry = cache[condition]
        lines = [
            r"\begin{subtable}{0.48\linewidth}", r"\centering",
            r"\caption{" + title + r"}", r"\label{tab:e1-60-" + tag + r"}",
            r"\begin{tabular}{@{}l" + "r" * len(E1_METRICS) + r"@{}}", r"\toprule",
            "Method & " + " & ".join(h for _, h, _ in E1_METRICS) + r" \\",
            r"\midrule",
        ]
        for baseline in baselines:
            cells = [
                _fmt(entry["cells"][(baseline, metric)]["mean"], digits,
                     bold=baseline in entry["best"][metric])
                for metric, _, digits in E1_METRICS
            ]
            n_surv, n_tot = entry["survival"][baseline]
            marker = "" if n_tot and n_surv == n_tot else r"$^{\ast}$"
            lines.append(SHORT_LABEL[baseline] + marker + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
        for rival in baselines[1:]:
            cells = [
                _signed(entry["paired"][(metric, rival)]["mean"], digits,
                        entry["paired"][(metric, rival)]["significant"])
                for metric, _, digits in E1_METRICS
            ]
            lines.append(rf"$\Delta$ {rival}$-$C1 & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{subtable}"]
        return "\n".join(lines)

    any_fail = any(
        n_surv != n_tot for entry in cache.values()
        for n_surv, n_tot in entry["survival"].values())
    caption = (
        r"Simultaneous base-velocity and end-effector pose tracking (E1), "
        rf"official evaluation over {n_seeds} paired random seeds "
        r"(all seeds analysed, none excluded). "
        r"$30$\,s episodes, the first $3$\,s excluded as the native startup ramp. "
        r"$E_{v}$: base XY linear-velocity RMSE (m/s). "
        r"$E_{\omega}$: base yaw-rate RMSE (rad/s). "
        r"$E_{p}$: end-effector position RMSE (m). "
        r"$E_{R}$: end-effector orientation MAE (deg), geodesic distance on $SO(3)$. "
        r"Errors are computed in the native PLB command frame over every "
        r"evaluated sample, including the transients after command switches. "
        r"Lower is better. \textbf{Bold} marks the methods not statistically "
        r"separated from the leader by a paired bootstrap 95\% CI "
        r"($10^4$ resamples). $\Delta$ rows give the paired difference against "
        r"C1, positive meaning C1 had the smaller error; $^{*}$ marks a paired "
        r"95\% CI that excludes zero. Confidence intervals for every entry are "
        r"in Table~\ref{tab:e1-60-intervals}."
    )
    if any_fail:
        caption += (r" $^{\ast}$at least one episode terminated early; such "
                    r"episodes are retained in every statistic and counted in "
                    r"Table~\ref{tab:e1-survival}.")
    body = "\n".join([
        r"% Generated by benchmark/publish.py. Requires booktabs + subcaption.",
        r"\begin{table*}[t]", r"\centering",
        r"\caption{" + caption + r"}", r"\label{tab:e1-60-tracking}",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.05}",
        r"\small",
        sub_table(*E1_PANELS[0]), r"\hfill", sub_table(*E1_PANELS[1]), r"",
        r"\vspace{0.8em}", r"",
        sub_table(*E1_PANELS[2]), r"\hfill", sub_table(*E1_PANELS[3]),
        r"\end{table*}",
    ])
    _write(os.path.join(out_dir, f"{name}.tex"), body + "\n")

    # ---------------- intervals companion
    lines = [
        r"% Generated by benchmark/publish.py. Requires booktabs + multirow.",
        r"\begin{table*}[t]\centering",
        r"\caption{E1 official evaluation, " + str(n_seeds) + r" paired seeds: "
        r"means with 95\% bootstrap confidence intervals and the paired "
        r"differences against C1. Positive $\Delta$ means C1 had the smaller "
        r"error. $^{*}$: the paired 95\% CI excludes zero. Intervals are "
        r"percentile bootstrap, $10^4$ resamples, fixed resampling seed; the "
        r"experimental unit is the episode, never the time sample.}",
        r"\label{tab:e1-60-intervals}",
        r"\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}\toprule",
        r"Condition & Metric & C1 (ours) & C3 & C5 & $\Delta$ C3$-$C1 & "
        r"$\Delta$ C5$-$C1 \\\midrule",
    ]
    for condition, _, title in E1_PANELS:
        entry = cache[condition]
        for k, (metric, header, digits) in enumerate(E1_METRICS):
            label = (r"\multirow{4}{*}{" + title + r"}") if k == 0 else ""
            cells = []
            for baseline in baselines:
                s = entry["cells"][(baseline, metric)]
                cells.append(
                    _fmt(s["mean"], digits, bold=baseline in entry["best"][metric])
                    + r" {\scriptsize " + _ci(s["ci_low"], s["ci_high"], digits) + "}")
            for rival in baselines[1:]:
                p = entry["paired"][(metric, rival)]
                cells.append(
                    _signed(p["mean"], digits, p["significant"])
                    + r" {\scriptsize " + _ci(p["ci_low"], p["ci_high"], digits) + "}")
            lines.append(f"{label} & {header} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if condition != E1_PANELS[-1][0] else r"\bottomrule")
    lines += [r"\end{tabular}\end{table*}"]
    _write(os.path.join(out_dir, f"{name}_intervals.tex"), "\n".join(lines) + "\n")


# ==========================================================================
# E2
# ==========================================================================
E2_TABLE_METRICS = [
    ("base_vxy_rmse", r"$E_{v}\downarrow$", 3),
    ("base_wz_rmse", r"$E_{\omega}\downarrow$", 3),
    ("total_power", r"$P_{tot}\downarrow$", 1),
    ("leg_power", r"$P_{leg}\downarrow$", 1),
    ("wheel_power", r"$P_{wh}$", 1),
    ("leg_acc_rms", r"$a_{leg}\downarrow$", 1),
    ("arm_acc_rms", r"$a_{arm}\downarrow$", 1),
]
E2_ORDER = ["Low", "Medium", "High"]
E2_INTENSITY = {"Low": r"Low ($0.25\times$)", "Medium": r"Medium ($0.50\times$)",
                "High": r"High ($0.75\times$)"}


def e2_official_table(rows, out_dir, cfg_stats, baselines=("C1", "C2"),
                      name="E2_official_60seeds"):
    baselines = list(baselines)
    n_seeds = len(sorted({r["seed"] for r in rows}))
    conditions = [c for c in E2_ORDER if any(r["condition"] == c for r in rows)]
    cache = {}
    for condition in conditions:
        grouped = _by(rows, condition, baselines)
        entry = {"cells": {}, "best": {}, "paired": {}, "survival": {}}
        for baseline in baselines:
            items = list(grouped[baseline].values())
            entry["survival"][baseline] = (
                sum(int(r["survived"]) for r in items), len(items))
        for metric, _, _ in E2_TABLE_METRICS:
            per_method = {
                b: {s: r[metric] for s, r in grouped[b].items() if metric in r}
                for b in baselines
            }
            # Wheel power is not a "lower is better" objective: C2 has no wheel
            # actuation at all, so bolding it would be meaningless.
            entry["best"][metric] = ([] if metric == "wheel_power" else
                                     stats.best_methods(
                                         per_method, lower_is_better=True,
                                         n_resamples=cfg_stats["bootstrap_resamples"],
                                         seed=cfg_stats["bootstrap_seed"]))
            for baseline in baselines:
                entry["cells"][(baseline, metric)] = _summ(
                    list(per_method[baseline].values()), cfg_stats)
            entry["paired"][metric] = _paired(
                per_method["C1"], per_method["C2"], cfg_stats)
        cache[condition] = entry

    lines = [
        r"% Generated by benchmark/publish.py. Requires booktabs + multirow.",
        r"\begin{table*}[t]\centering",
        r"\caption{Active wheel actuation (E2), official evaluation over "
        + str(n_seeds) + r" paired random seeds per intensity. C1 and C2 were "
        r"trained with the same base-velocity command range and receive the "
        r"identical seeds, command sequences and evaluation window; the only "
        r"differences are the wheeled vs.\ welded-wheel model and C2's smaller "
        r"action space. $E_{v}$ base XY velocity RMSE (m/s); $E_{\omega}$ base "
        r"yaw-rate RMSE (rad/s); $P$ mechanical power proxy "
        r"$\sum_j |\tau_j \dot q_j|$ from actuator-produced torques at "
        r"$250$\,Hz (W) --- \emph{not} electrical battery power; $a$ joint "
        r"acceleration RMS (rad/s$^2$) over the matched $12$ leg and $6$ arm "
        r"DOFs, with no normalisation across the differing action dimensions. "
        r"Entries are mean {\scriptsize[95\% CI]}. \textbf{Bold} marks the "
        r"methods not statistically separated from the leader; wheel power is "
        r"never bolded because C2 has no wheel actuation. $\Delta$ is the "
        r"paired difference C2$-$C1, positive meaning C1 lower; $^{*}$ marks a "
        r"paired 95\% CI excluding zero.}",
        r"\label{tab:e2-60}",
        r"\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}ll" + "r" * len(E2_TABLE_METRICS) + r"@{}}\toprule",
        r"Intensity & Method & " + " & ".join(h for _, h, _ in E2_TABLE_METRICS)
        + r" \\\midrule",
    ]
    for ci, condition in enumerate(conditions):
        entry = cache[condition]
        for k, baseline in enumerate(baselines + ["delta"]):
            label = (r"\multirow{3}{*}{" + E2_INTENSITY.get(condition, condition) + r"}"
                     if k == 0 else "")
            if baseline == "delta":
                cells = [
                    _signed(entry["paired"][metric]["mean"], digits,
                            entry["paired"][metric]["significant"])
                    + r" {\scriptsize " + _ci(entry["paired"][metric]["ci_low"],
                                              entry["paired"][metric]["ci_high"], digits) + "}"
                    for metric, _, digits in E2_TABLE_METRICS
                ]
                row_label = r"$\Delta$ C2$-$C1"
            else:
                cells = []
                for metric, _, digits in E2_TABLE_METRICS:
                    s = entry["cells"][(baseline, metric)]
                    cells.append(
                        _fmt(s["mean"], digits, bold=baseline in entry["best"][metric])
                        + r" {\scriptsize " + _ci(s["ci_low"], s["ci_high"], digits) + "}")
                n_surv, n_tot = entry["survival"][baseline]
                row_label = SHORT_LABEL[baseline] + (
                    "" if n_tot and n_surv == n_tot else r"$^{\ast}$")
            lines.append(f"{label} & {row_label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule" if ci == len(conditions) - 1 else r"\midrule")
    lines += [r"\end{tabular}\end{table*}"]
    _write(os.path.join(out_dir, f"{name}.tex"), "\n".join(lines) + "\n")


# ==========================================================================
# Illustrative-case tables
# ==========================================================================
SELECTED_CAPTION_PREFIX = (
    r"\textbf{Selected illustrative trials from a separately evaluated "
    r"candidate seed pool} (seeds $100$--$149$), disjoint from the official "
    r"statistical pool (seeds $0$--$59$). Each trial was chosen by the "
    r"pre-declared ranking in "
    r"\texttt{benchmark/select\_cases.py}; the full candidate ranking, the "
    r"gates and the scores are in the selection manifest. "
)
SELECTED_CAPTION_SUFFIX = (
    r" These are single episodes, shown to illustrate behaviour. They are "
    r"\emph{not} benchmark average performance and carry no uncertainty "
    r"estimate: \textbf{bold} marks only that C1 had the smaller value "
    r"\emph{in that one trial}. Every measured metric is shown for every "
    r"compared method, including those that do not favour C1."
)


def _selected_rows_table(cases, metrics, compared, caption, label, out_path,
                         bold_reference="C1"):
    lines = [
        r"% Generated by benchmark/publish.py. Requires booktabs + multirow.",
        r"\begin{table*}[t]\centering",
        r"\caption{" + caption + r"}", r"\label{" + label + r"}",
        r"\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lll" + "r" * len(metrics) + r"@{}}\toprule",
        r"Illustrative case & Seed & Method & "
        + " & ".join(h for _, h, _ in metrics) + r" \\\midrule",
    ]
    present = [c for c in cases if c.get("selected")]
    for ci, case in enumerate(present):
        sel = case["selected"]
        rows_here = [b for b in compared if b in sel["metrics"]]
        best = {}
        for metric, _, _ in metrics:
            values = {b: sel["metrics"][b].get(metric) for b in rows_here}
            finite = {b: v for b, v in values.items()
                      if v is not None and np.isfinite(v)}
            if finite and bold_reference in finite:
                best[metric] = (min(finite, key=finite.get) == bold_reference)
            else:
                best[metric] = False
        for k, baseline in enumerate(rows_here):
            case_label = (r"\multirow{" + str(len(rows_here)) + r"}{*}{"
                          + case["title"] + r"}") if k == 0 else ""
            seed_label = (r"\multirow{" + str(len(rows_here)) + r"}{*}{"
                          + str(sel["seed"]) + r"}") if k == 0 else ""
            cells = []
            for metric, _, digits in metrics:
                value = sel["metrics"][baseline].get(metric)
                cells.append(_fmt(value, digits,
                                  bold=(baseline == bold_reference and best[metric])))
            marker = "" if int(sel["survived"][baseline]) else r"$^{\ast}$"
            lines.append(f"{case_label} & {seed_label} & "
                         + SHORT_LABEL[baseline] + marker + " & "
                         + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule" if ci == len(present) - 1 else r"\midrule")
    if not present:
        lines.append(r"\multicolumn{" + str(3 + len(metrics)) + r"}{c}{no candidate "
                     r"satisfied the selection criteria} \\\bottomrule")
    lines += [r"\end{tabular}\end{table*}"]
    _write(out_path, "\n".join(lines) + "\n")


E1_SELECTED_METRICS = [
    ("base_vxy_rmse", r"$E_{v}$ (m/s)", 3),
    ("base_wz_rmse", r"$E_{\omega}$ (rad/s)", 3),
    ("ee_pos_rmse", r"$E_{p}$ (m)", 3),
    ("ee_ori_mae_deg", r"$E_{R}$ (deg)", 1),
]
E2_SELECTED_METRICS = [
    ("base_vxy_rmse", r"$E_{v}$ (m/s)", 3),
    ("base_wz_rmse", r"$E_{\omega}$ (rad/s)", 3),
    ("total_power", r"$P_{tot}$ (W)", 1),
    ("leg_power", r"$P_{leg}$ (W)", 1),
    ("wheel_power", r"$P_{wh}$ (W)", 1),
    ("leg_acc_rms", r"$a_{leg}$", 1),
    ("arm_acc_rms", r"$a_{arm}$", 1),
]
E3_SELECTED_METRICS = [
    ("ee_pos_rmse", r"$E_{p}$ (m)", 3),
    ("ee_ori_mae_deg", r"$E_{R}$ (deg)", 1),
    ("base_vxy_rmse", r"$E_{v}$ (m/s)", 3),
    ("base_wz_rmse", r"$E_{\omega}$ (rad/s)", 3),
    ("delta_ee_pos_rmse_from_base", r"$\Delta E_{p}$ (m)", 4),
    ("delta_base_vxy_rmse_from_ee", r"$\Delta E_{v}$ (m/s)", 4),
]


def e1_selected_table(cases, out_dir, name="E1_selected_3seeds"):
    _selected_rows_table(
        cases, E1_SELECTED_METRICS, ["C1", "C3", "C5"],
        SELECTED_CAPTION_PREFIX
        + r"All three methods ran the identical base-velocity and "
          r"end-effector command sequences in condition T4 (base moving, EE "
          r"random targets) at the stated seed."
        + SELECTED_CAPTION_SUFFIX,
        "tab:e1-selected", os.path.join(out_dir, f"{name}.tex"))


def e2_selected_table(cases, out_dir, name="E2_selected_3seeds"):
    _selected_rows_table(
        cases, E2_SELECTED_METRICS, ["C1", "C2"],
        SELECTED_CAPTION_PREFIX
        + r"C1 and C2 ran the identical base-velocity command sequence at "
          r"Medium intensity ($0.50\times$ the shared supported range) at the "
          r"stated seed. $P$ is the mechanical power proxy "
          r"$\sum_j |\tau_j \dot q_j|$, not battery power."
        + SELECTED_CAPTION_SUFFIX,
        "tab:e2-selected", os.path.join(out_dir, f"{name}.tex"))


def e3_selected_table(cases, out_dir, name="E3_selected_3seeds"):
    _selected_rows_table(
        cases, E3_SELECTED_METRICS, ["C1", "C3", "C5"],
        SELECTED_CAPTION_PREFIX
        + r"All three methods ran the identical base-velocity waveform and "
          r"end-effector path in the stated combined-motion cell. "
          r"$\Delta E_{p}$ is the locomotion-induced EE error increase and "
          r"$\Delta E_{v}$ the manipulation-induced base error increase, both "
          r"computed for this seed against its own $\alpha=0$ reference cells."
        + SELECTED_CAPTION_SUFFIX,
        "tab:e3-selected", os.path.join(out_dir, f"{name}.tex"))
