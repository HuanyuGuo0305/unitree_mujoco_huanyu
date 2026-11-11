"""Publication table generators.

E1's table is the one with the explicit design requirement: not a 12-row list
with repeated condition labels, but ONE IEEE ``table*`` holding four compact
sub-tables laid out 2x2, preserving the 2x2 experimental design:

    (a) Base still / EE fixed      (b) Base still / EE random
    (c) Base moving / EE fixed     (d) Base moving / EE random

Each sub-table is  Method | vxy | wz | EE pos | EE ori  with units in the
shared caption. Bolding marks the set of methods that are not statistically
beaten by the point-estimate leader (paired bootstrap CI containing zero), so a
numerically smallest value is never bolded as conclusively superior when the
uncertainty does not support it.
"""

from __future__ import annotations

import os

import numpy as np

from benchmark.controllers.registry import baseline_label
from benchmark.core import stats

E1_PANELS = [
    ("T1", "a", "Base still / EE fixed"),
    ("T2", "b", "Base still / EE random"),
    ("T3", "c", "Base moving / EE fixed"),
    ("T4", "d", "Base moving / EE random"),
]
E1_METRICS = [
    ("base_vxy_rmse", r"$E_{v}\downarrow$", 3),
    ("base_wz_rmse", r"$E_{\omega}\downarrow$", 3),
    ("ee_pos_rmse", r"$E_{p}\downarrow$", 3),
    ("ee_ori_mae_deg", r"$E_{R}\downarrow$", 1),
]
SHORT_LABEL = {
    "C1": r"C1 Unified RL (ours)",
    "C2": r"C2 Unified RL, fixed wheel",
    "C3": r"C3 RL + IK",
    "C4": r"C4 RL + IK, fixed wheel",
    "C5": r"C5 Dual-PPO",
    "C6": r"C6 Dual-PPO, fixed wheel",
}


def _by(rows, condition, baselines):
    out = {}
    for baseline in baselines:
        out[baseline] = {
            r["seed"]: r for r in rows
            if r["condition"] == condition and r["baseline"] == baseline
        }
    return out


def _fmt(value, digits, bold=False, dagger=False):
    if value is None or not np.isfinite(value):
        return "--"
    text = f"{value:.{digits}f}"
    if bold:
        text = r"\textbf{" + text + "}"
    if dagger:
        text += r"$^{\dagger}$"
    return text


def _write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"  wrote {os.path.relpath(path)}")


# ==========================================================================
# E1
# ==========================================================================
def e1_compact_table(rows, out_dir: str, cfg_stats: dict, baselines=("C1", "C3", "C5")):
    baselines = list(baselines)
    panels = {}
    for condition, _, _ in E1_PANELS:
        grouped = _by(rows, condition, baselines)
        panel = {}
        for metric, _, _ in E1_METRICS:
            per_method = {
                b: {s: r[metric] for s, r in grouped[b].items() if metric in r}
                for b in baselines
            }
            best = stats.best_methods(
                per_method, lower_is_better=True,
                n_resamples=cfg_stats["bootstrap_resamples"],
                seed=cfg_stats["bootstrap_seed"],
            )
            for baseline in baselines:
                values = list(per_method[baseline].values())
                summary = stats.summarize(
                    values,
                    n_resamples=cfg_stats["bootstrap_resamples"],
                    seed=cfg_stats["bootstrap_seed"],
                    confidence=cfg_stats["confidence"],
                )
                panel[(baseline, metric)] = (summary, baseline in best)
        survival = {}
        for baseline in baselines:
            items = list(grouped[baseline].values())
            survival[baseline] = (
                sum(int(r["survived"]) for r in items), len(items)
            )
        panels[condition] = (panel, survival)

    def sub_table(condition, tag, title):
        panel, survival = panels[condition]
        lines = [
            r"\begin{subtable}{0.48\linewidth}",
            r"\centering",
            r"\caption{" + title + r"}",
            r"\label{tab:e1-" + tag + r"}",
            r"\begin{tabular}{@{}l" + "r" * len(E1_METRICS) + r"@{}}",
            r"\toprule",
            "Method & " + " & ".join(h for _, h, _ in E1_METRICS) + r" \\",
            r"\midrule",
        ]
        for baseline in baselines:
            cells = []
            for metric, _, digits in E1_METRICS:
                summary, is_best = panel[(baseline, metric)]
                cells.append(_fmt(summary["mean"], digits, bold=is_best))
            n_surv, n_tot = survival[baseline]
            marker = "" if n_tot and n_surv == n_tot else r"$^{\ast}$"
            lines.append(SHORT_LABEL[baseline] + marker + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{subtable}"]
        return "\n".join(lines)

    n_seeds = len(sorted({r["seed"] for r in rows}))
    any_fail = any(
        n_surv != n_tot
        for _, survival in panels.values()
        for n_surv, n_tot in survival.values()
    )
    caption = (
        r"Simultaneous base-velocity and end-effector pose tracking (E1). "
        rf"Mean over {n_seeds} paired random seeds; "
        r"$30$\,s episodes with the first $3$\,s excluded as the native startup ramp. "
        r"$E_{v}$: base XY linear-velocity RMSE (m/s). "
        r"$E_{\omega}$: base yaw-rate RMSE (rad/s). "
        r"$E_{p}$: end-effector position RMSE (m). "
        r"$E_{R}$: end-effector orientation MAE (deg), geodesic distance on $SO(3)$. "
        r"All errors are computed in the native PLB command frame over every "
        r"evaluated sample, transients after command switches included. "
        r"Lower is better; \textbf{bold} marks methods not statistically "
        r"separated from the leader by a paired bootstrap 95\% CI "
        r"($10^4$ resamples)."
    )
    if any_fail:
        caption += r" $^{\ast}$at least one episode terminated early (see Table~\ref{tab:e1-survival})."

    body = "\n".join(
        [
            r"% Generated by benchmark/aggregate.py --tables. Requires:",
            r"%   \usepackage{booktabs} \usepackage{subcaption}",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{" + caption + r"}",
            r"\label{tab:e1-tracking}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\small",
            sub_table(*E1_PANELS[0]),
            r"\hfill",
            sub_table(*E1_PANELS[1]),
            r"",
            r"\vspace{0.8em}",
            r"",
            sub_table(*E1_PANELS[2]),
            r"\hfill",
            sub_table(*E1_PANELS[3]),
            r"\end{table*}",
        ]
    )
    _write(os.path.join(out_dir, "E1_tracking_compact.tex"), body + "\n")

    # Alternative layout B: one 4x(4 metrics) block table, 2x2 metric columns.
    _write(
        os.path.join(out_dir, "E1_tracking_compact_altB.tex"),
        _e1_alt_single_block(panels, baselines, caption) + "\n",
    )

    # Survival / termination companion table (never folded into the main one).
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{E1 survival and early termination over the official seeds. "
        r"Terminated episodes are retained in every statistic above; this table "
        r"reports how many there were.}",
        r"\label{tab:e1-survival}",
        r"\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}ll" + "r" * 3 + r"@{}}\toprule",
        r"Condition & Method & Episodes & Survived & Rate \\\midrule",
    ]
    for condition, _, title in E1_PANELS:
        _, survival = panels[condition]
        for baseline in baselines:
            n_surv, n_tot = survival[baseline]
            rate = n_surv / n_tot if n_tot else float("nan")
            lines.append(
                f"{title} & {SHORT_LABEL[baseline]} & {n_tot} & {n_surv} & {rate:.2f}" + r" \\"
            )
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    _write(os.path.join(out_dir, "E1_survival.tex"), "\n".join(lines) + "\n")


def _e1_alt_single_block(panels, baselines, caption):
    """Alternative E1 layout: a single block table with the condition as a row
    group. More compact vertically, slightly less legible than four sub-tables.
    """
    lines = [
        r"% Alternative E1 layout (single block, condition row groups).",
        r"\begin{table}[t]\centering",
        r"\caption{" + caption + r"}",
        r"\label{tab:e1-tracking-alt}",
        r"\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}ll" + "r" * len(E1_METRICS) + r"@{}}\toprule",
        r"Condition & Method & " + " & ".join(h for _, h, _ in E1_METRICS) + r" \\\midrule",
    ]
    for i, (condition, _, title) in enumerate(E1_PANELS):
        panel, _ = panels[condition]
        for j, baseline in enumerate(baselines):
            cells = []
            for metric, _, digits in E1_METRICS:
                summary, is_best = panel[(baseline, metric)]
                cells.append(_fmt(summary["mean"], digits, bold=is_best))
            label = r"\multirow{%d}{*}{%s}" % (len(baselines), title) if j == 0 else ""
            lines.append(f"{label} & {SHORT_LABEL[baseline]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if i < len(E1_PANELS) - 1 else "")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines)


# ==========================================================================
# E2
# ==========================================================================
def e2_table(rows, out_dir: str, cfg_stats: dict, primary_condition: str = "Medium"):
    baselines = ["C1", "C2"]
    metrics = [
        ("base_vxy_rmse", r"Base XY RMSE", 3),
        ("base_wz_rmse", r"Yaw RMSE", 3),
        ("total_power", r"Power proxy", 1),
        ("leg_acc_rms", r"Leg acc. RMS", 1),
    ]
    grouped = _by(rows, primary_condition, baselines)
    if not any(grouped.values()):
        print(f"  E2: no episodes for condition {primary_condition}; table skipped")
        return

    per_metric_best = {}
    for metric, _, _ in metrics:
        per_method = {
            b: {s: r[metric] for s, r in grouped[b].items() if metric in r} for b in baselines
        }
        per_metric_best[metric] = stats.best_methods(
            per_method, lower_is_better=True,
            n_resamples=cfg_stats["bootstrap_resamples"],
            seed=cfg_stats["bootstrap_seed"],
        )

    lines = [
        r"% Generated by benchmark/aggregate.py --tables. Requires \usepackage{booktabs}.",
        r"\begin{table}[t]\centering",
        r"\caption{Active wheel actuation (E2) at " + primary_condition.lower() +
        r" base-motion intensity. C1 and C2 share the identical base-velocity "
        r"command range, the identical seeded command sequences and the identical "
        r"fixed EE target; they differ only in whether the four wheels are "
        r"actuated. Power is the mechanical proxy "
        r"$\overline{\sum_j |\tau_j \dot q_j|}$ from the actuator-produced "
        r"torques at 250\,Hz -- not battery power. Joint acceleration RMS uses "
        r"the 12 leg joints only, unfiltered, at 250\,Hz. Mean over the official "
        r"seeds; \textbf{bold} marks methods not statistically separated from the "
        r"leader (paired bootstrap 95\% CI).}",
        r"\label{tab:e2-wheel}",
        r"\small\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}l" + "r" * len(metrics) + r"r@{}}\toprule",
        r"Controller & " + " & ".join(h for _, h, _ in metrics) + r" & Survival \\",
        r" & (m/s) & (rad/s) & (W) & (rad/s$^2$) & \\\midrule",
    ]
    for baseline in baselines:
        items = list(grouped[baseline].values())
        cells = []
        for metric, _, digits in metrics:
            values = [r[metric] for r in items if metric in r]
            summary = stats.summarize(
                values, n_resamples=cfg_stats["bootstrap_resamples"],
                seed=cfg_stats["bootstrap_seed"], confidence=cfg_stats["confidence"],
            )
            cells.append(_fmt(summary["mean"], digits, bold=baseline in per_metric_best[metric]))
        n_surv = sum(int(r["survived"]) for r in items)
        rate = n_surv / len(items) if items else float("nan")
        lines.append(
            SHORT_LABEL[baseline] + " & " + " & ".join(cells)
            + f" & {rate:.2f} ({n_surv}/{len(items)})" + r" \\"
        )
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    _write(os.path.join(out_dir, "E2_wheel_actuation.tex"), "\n".join(lines) + "\n")


# ==========================================================================
# E3
# ==========================================================================
def e3_matrix_table(summary, out_dir: str):
    """Compact 3x3 matrix of EE position RMSE and base XY RMSE per baseline."""
    families = sorted({r["ee_family"] for r in summary})
    baselines = sorted({r["baseline"] for r in summary})
    alphas_b = sorted({float(r["alpha_base"]) for r in summary})
    alphas_e = sorted({float(r["alpha_ee"]) for r in summary})

    lines = [
        r"% Generated by benchmark/aggregate.py --tables. Requires \usepackage{booktabs}.",
        r"\begin{table*}[t]\centering",
        r"\caption{Dynamic whole-body coordination (E3). Each cell is the mean "
        r"over the official paired seeds of the EE position RMSE (m) and, in "
        r"parentheses, the base XY velocity RMSE (m/s), at base-motion intensity "
        r"$\alpha_b$ and EE-speed intensity $\alpha_e$. "
        r"$\alpha_b=1$ corresponds to the full supported base command range "
        r"($\pm0.6$\,m/s, $\pm0.6$\,rad/s); $\alpha_e=1$ to a mean EE path speed "
        r"of $0.40$\,m/s on a fixed-shape, fixed-amplitude, fixed-centre path.}",
        r"\label{tab:e3-matrix}",
        r"\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}ll" + "c" * len(alphas_e) + r"@{}}\toprule",
        r"Trajectory / Method & $\alpha_b$ & "
        + " & ".join(rf"$\alpha_e={a:g}$" for a in alphas_e) + r" \\\midrule",
    ]
    index = {
        (r["ee_family"], r["baseline"], float(r["alpha_base"]), float(r["alpha_ee"])): r
        for r in summary
    }
    for family in families:
        for baseline in baselines:
            for i, ab in enumerate(alphas_b):
                cells = []
                for ae in alphas_e:
                    rec = index.get((family, baseline, ab, ae))
                    if rec is None:
                        cells.append("--")
                    else:
                        cells.append(
                            f"{rec['ee_pos_rmse_mean']:.3f} ({rec['base_vxy_rmse_mean']:.3f})"
                        )
                label = f"{family} / {SHORT_LABEL[baseline]}" if i == 0 else ""
                lines.append(f"{label} & {ab:g} & " + " & ".join(cells) + r" \\")
            lines.append(r"\addlinespace")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}\end{table*}"]
    _write(os.path.join(out_dir, "E3_coordination_matrix.tex"), "\n".join(lines) + "\n")


# ==========================================================================
# E4
# ==========================================================================
def e4_table(rows, out_dir: str, cfg_stats: dict):
    baselines = sorted({r["baseline"] for r in rows})
    metrics = [
        ("ee_world_pos_rmse", r"World EE pos. RMSE (m)", 4),
        ("ee_world_pos_max", r"max (m)", 4),
        ("ee_world_ori_mae_deg", r"World EE ori. MAE (deg)", 2),
        ("base_vxy_rmse", r"Base XY RMSE (m/s)", 3),
    ]
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{World-frame EE stabilisation (E4, ``chicken head''). The "
        r"world-frame EE target is held fixed and transformed into the native "
        r"PLB command frame at every control update from the measured base pose; "
        r"no external stabilisation controller is added and the policy and "
        r"actuation pipeline are unchanged. Errors are world-frame.}",
        r"\label{tab:e4-chicken-head}",
        r"\small\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}l" + "r" * len(metrics) + r"r@{}}\toprule",
        r"Controller & " + " & ".join(h for _, h, _ in metrics) + r" & Survival \\\midrule",
    ]
    for baseline in baselines:
        items = [r for r in rows if r["baseline"] == baseline]
        cells = []
        for metric, _, digits in metrics:
            values = [r[metric] for r in items if metric in r]
            summary = stats.summarize(
                values, n_resamples=cfg_stats["bootstrap_resamples"],
                seed=cfg_stats["bootstrap_seed"], confidence=cfg_stats["confidence"],
            )
            cells.append(_fmt(summary["mean"], digits))
        n_surv = sum(int(r["survived"]) for r in items)
        lines.append(
            SHORT_LABEL[baseline] + " & " + " & ".join(cells)
            + f" & {n_surv}/{len(items)}" + r" \\"
        )
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    _write(os.path.join(out_dir, "E4_stabilization.tex"), "\n".join(lines) + "\n")
