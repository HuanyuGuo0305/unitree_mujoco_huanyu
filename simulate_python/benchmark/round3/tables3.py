#!/usr/bin/env python3
"""Round-3 publication tables (IEEE RA-L, double column).

Built exclusively from the round-3 CSVs in ``final_round3/<E>/``, so a table
cell can always be traced to a summary row and from there to 50 per-seed rows.

Conventions
-----------
* Every table states its seed count and that the population is seeds 0-49.
* **Bold** marks the set of methods not statistically separated from the
  point-estimate leader by a paired bootstrap 95% CI. The numerically smallest
  value is never bolded on its own when the uncertainty does not support it, so
  a tie reads as a tie.
* A paired difference is written as ``rival - C1``; a trailing ``*`` means its
  95% CI excludes zero. Signs are set in math mode so a negative renders as a
  minus, not a hyphen.
* A method whose survival is below 50/50 carries a dagger, and its survival is
  printed in the same table -- its error statistics are conditioned on the
  episodes that survived and must be read that way.
* Compact tables carry means; the ``_intervals`` companion carries every CI and
  both paired differences, because they do not fit in the compact cells without
  making them unreadable.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from benchmark.round3 import common

SHORT = {"C1": r"C1 (ours)", "C2": "C2", "C3": "C3", "C5": "C5"}

E1_PANELS = [
    ("T1", "a", "Base still, EE fixed"),
    ("T2", "b", "Base still, EE random (3 s)"),
    ("T3", "c", "Base moving (2--3 s), EE fixed"),
    ("T4", "d", "Base moving (2--3 s), EE random (3 s)"),
]
E1_COLS = [
    ("base_vxy_rmse_mps", r"$E_v$", 4),
    ("base_yaw_rmse_radps", r"$E_\omega$", 4),
    ("ee_position_rmse_m", r"$E_p$", 4),
    ("ee_orientation_mae_deg", r"$E_R$", 2),
]
E2_COLS = [
    ("base_vxy_rmse_mps", r"$E_v$", 4),
    ("base_yaw_rmse_radps", r"$E_\omega$", 4),
    ("total_power_w", r"$P_{\mathrm{tot}}$", 1),
    ("leg_power_w", r"$P_{\mathrm{leg}}$", 1),
    ("wheel_power_w", r"$P_{\mathrm{wheel}}$", 1),
    ("leg_acc_rms_radps2", r"$a_{\mathrm{leg}}$", 1),
    ("arm_acc_rms_radps2", r"$a_{\mathrm{arm}}$", 2),
]


def _index(rows, keys):
    out = {}
    for r in rows:
        out[tuple(r[k] for k in keys)] = r
    return out


def _fmt(value, digits, bold=False):
    if value is None or not np.isfinite(float(value)):
        return "--"
    text = f"{float(value):.{digits}f}"
    return r"\textbf{" + text + "}" if bold else text


def _signed(value, digits, significant):
    if value is None or not np.isfinite(float(value)):
        return "--"
    value = float(value)
    body = f"{abs(value):.{digits}f}"
    sign = "+" if value >= 0 else "-"
    return "$" + sign + body + (r"^{*}" if significant else "") + "$"


def _ci(lo, hi, digits):
    if lo is None or not (np.isfinite(float(lo)) and np.isfinite(float(hi))):
        return "--"
    return f"$[{float(lo):.{digits}f},\\,{float(hi):.{digits}f}]$"


def _best_set(summary_idx, paired_idx, condition, metric, baselines, family=None):
    """Methods not statistically separated from the point-estimate leader."""
    means = {}
    for b in baselines:
        key = (condition, b, metric) if family is None else (condition, family, b, metric)
        row = summary_idx.get(key)
        if row and np.isfinite(float(row["mean"])):
            means[b] = float(row["mean"])
    if not means:
        return set()
    leader = min(means, key=means.get)
    out = {leader}
    for b in means:
        if b == leader:
            continue
        # The paired file holds rival-vs-C1 only, so a leader that is not C1
        # is compared through C1's own paired row.
        pr = paired_idx.get((condition, b, metric) if family is None
                            else (condition, family, b, metric))
        if pr is None:
            continue
        if leader == "C1":
            if not int(pr["significant"]):
                out.add(b)
        elif b == "C1":
            if not int(pr["significant"]):
                out.add(b)
        else:
            out.add(b)  # cannot be tested from the C1-referenced file; do not claim
    return out


# ==========================================================================
# E1
# ==========================================================================
def e1_tables(out_dir, tables_dir, name="E1_official_50seeds"):
    summary = common.read_csv(os.path.join(out_dir, "summary_50seeds.csv"))
    paired = common.read_csv(os.path.join(out_dir, "paired_comparisons.csv"))
    s_idx = _index(summary, ["condition", "baseline", "metric"])
    p_idx = _index(paired, ["condition", "rival", "metric"])
    baselines = ["C1", "C3", "C5"]
    n_seeds = int(s_idx[("T1", "C1", "base_vxy_rmse_mps")]["n_seeds"])

    def panel(condition, tag, title):
        lines = [
            r"\begin{subtable}{0.49\linewidth}", r"\centering",
            r"\caption{" + title + r"}",
            r"\label{tab:e1-r3-" + tag + r"}",
            r"\begin{tabular}{@{}l" + "r" * len(E1_COLS) + r"@{}}", r"\toprule",
            "Method & " + " & ".join(h for _, h, _ in E1_COLS) + r" \\",
            r"\midrule",
        ]
        for b in baselines:
            cells = []
            for metric, _h, digits in E1_COLS:
                best = _best_set(s_idx, p_idx, condition, metric, baselines)
                cells.append(_fmt(s_idx[(condition, b, metric)]["mean"], digits,
                                  bold=b in best))
            surv = s_idx[(condition, b, "survived")]
            n_surv = int(surv["n_survived"])
            dagger = r"$^{\dagger}$" if n_surv < n_seeds else ""
            lines.append(SHORT[b] + dagger + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
        for rival in ("C3", "C5"):
            cells = []
            for metric, _h, digits in E1_COLS:
                pr = p_idx[(condition, rival, metric)]
                cells.append(_signed(pr["mean_difference"], digits,
                                     int(pr["significant"])))
            lines.append(rf"$\Delta$({rival}$-$C1) & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
        surv_cells = []
        for b in baselines:
            surv = s_idx[(condition, b, "survived")]
            surv_cells.append(f"{int(surv['n_survived'])}/{n_seeds}")
        lines += [
            r"\multicolumn{" + str(1 + len(E1_COLS)) + r"}{@{}l@{}}{Survival: "
            + ", ".join(f"{SHORT[b]} {c}" for b, c in zip(baselines, surv_cells))
            + r"} \\",
            r"\bottomrule", r"\end{tabular}", r"\end{subtable}",
        ]
        return lines

    body = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{E1 base-velocity and end-effector pose tracking, round-3 "
        r"protocol: base commands resampled every 2--3\,s, EE random targets "
        r"replaced every 3\,s with no interpolation. Mean over "
        + str(n_seeds) + r" official seeds (0--" + str(n_seeds - 1) + r"), "
        r"identical seeded command sequences for C1, C3 and C5. "
        r"$E_v$ base XY velocity RMSE (m/s), $E_\omega$ base yaw-rate RMSE "
        r"(rad/s), $E_p$ EE position RMSE (m), $E_R$ EE orientation MAE (deg), "
        r"all over the full $[3,30)$\,s window including every post-switch "
        r"transient. \textbf{Bold}: not statistically separated from the "
        r"point-estimate leader (paired bootstrap 95\% CI). "
        r"$\Delta$ rows give the paired difference rival$-$C1, so a positive "
        r"value favours C1; $^{*}$ marks a 95\% CI excluding zero. "
        r"$^{\dagger}$ the method did not survive all " + str(n_seeds) +
        r" episodes, so its errors are conditioned on the surviving ones; see "
        r"the survival line. Per-seed values: "
        r"\texttt{final\_round3/E1/per\_seed.csv}.}",
        r"\label{tab:e1-round3}",
    ]
    for i, (condition, tag, title) in enumerate(E1_PANELS):
        body += panel(condition, tag, title)
        body.append(r"\hfill" if i % 2 == 0 else r"\\[4pt]")
    body += [r"\end{table*}"]
    path = os.path.join(tables_dir, name + ".tex")
    common.write_text(path, "\n".join(body) + "\n")

    # ---- intervals companion ----
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{E1 round-3 companion: 95\% bootstrap confidence intervals "
        r"(10{,}000 resamples, fixed seed) for every mean in "
        r"Table~\ref{tab:e1-round3}, and both paired differences with their "
        r"intervals and the fraction of the " + str(n_seeds) +
        r" paired seeds on which C1 had the lower error.}",
        r"\label{tab:e1-round3-intervals}",
        r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        r"Cond. & Quantity & " + " & ".join(h for _, h, _ in E1_COLS) + r" \\",
        r"\midrule",
    ]
    for condition, _tag, _title in E1_PANELS:
        for b in baselines:
            cells = [_ci(s_idx[(condition, b, m)]["ci95_lower"],
                         s_idx[(condition, b, m)]["ci95_upper"], d)
                     for m, _h, d in E1_COLS]
            lines.append(f"{condition} & {SHORT[b]} & " + " & ".join(cells) + r" \\")
        for rival in ("C3", "C5"):
            cells = [_ci(p_idx[(condition, rival, m)]["ci95_lower"],
                         p_idx[(condition, rival, m)]["ci95_upper"], d)
                     for m, _h, d in E1_COLS]
            lines.append(rf"{condition} & $\Delta$({rival}$-$C1) & "
                         + " & ".join(cells) + r" \\")
            cells = [f"{100*float(p_idx[(condition, rival, m)]['frac_pairs_c1_better']):.0f}\\%"
                     for m, _h, _d in E1_COLS]
            lines.append(rf"{condition} & C1 better on & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path2 = os.path.join(tables_dir, name + "_intervals.tex")
    common.write_text(path2, "\n".join(lines) + "\n")
    return [path, path2]


# ==========================================================================
# E2
# ==========================================================================
def e2_tables(out_dir, tables_dir, name="E2_official_50seeds"):
    summary = common.read_csv(os.path.join(out_dir, "summary_50seeds.csv"))
    paired = common.read_csv(os.path.join(out_dir, "paired_comparisons.csv"))
    s_idx = _index(summary, ["condition", "baseline", "metric"])
    p_idx = _index(paired, ["condition", "rival", "metric"])
    n_seeds = int(s_idx[("Low", "C1", "base_vxy_rmse_mps")]["n_seeds"])
    envelopes = {"Low": "0.20", "Medium": "0.40", "High": "0.60"}

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{E2 active wheel actuation, round-3 motion intensities. "
        r"Per-axis absolute command envelopes "
        r"$|v_x|,|v_y|\le\alpha$\,m/s and $|\omega_z|\le\alpha$\,rad/s with "
        r"$\alpha = 0.20$ (Low), $0.40$ (Medium), $0.60$ (High); these are "
        r"per-axis limits, not a bound on the XY norm. Mean over " + str(n_seeds) +
        r" paired seeds (0--" + str(n_seeds - 1) + r"); C1 and C2 receive "
        r"bit-identical seeded command sequences and a fixed PLB-frame EE "
        r"target. $P$ values are the mechanical power proxy "
        r"$\sum_j |\tau_j\dot q_j|$ from MuJoCo actuator forces at 250\,Hz --- "
        r"\emph{not} battery power. $P_{\mathrm{wheel}}$ is identically zero "
        r"for C2 by construction (welded wheels), so the $\Delta$ entry for it "
        r"records where the work moved, not a C2 advantage. "
        r"$\Delta$ = C2$-$C1, positive favours C1, $^{*}$ = 95\% CI excludes "
        r"zero. All " + str(n_seeds) + r"$\times$6 episodes survived. "
        r"Per-seed values: \texttt{final\_round3/E2/per\_seed.csv}.}",
        r"\label{tab:e2-round3}",
        r"\begin{tabular}{@{}ll" + "r" * len(E2_COLS) + r"@{}}", r"\toprule",
        r"Intensity & Method & " + " & ".join(h for _, h, _ in E2_COLS) + r" \\",
        r"& & (m/s) & (rad/s) & (W) & (W) & (W) & (rad/s$^2$) & (rad/s$^2$) \\",
        r"\midrule",
    ]
    for condition in ("Low", "Medium", "High"):
        label = rf"\multirow{{3}}{{*}}{{{condition} ($\alpha={envelopes[condition]}$)}}"
        for i, b in enumerate(("C1", "C2")):
            cells = []
            for metric, _h, digits in E2_COLS:
                best = _best_set(s_idx, p_idx, condition, metric, ["C1", "C2"])
                if metric == "wheel_power_w":
                    best = set()  # not a comparison; C2's value is zero by construction
                cells.append(_fmt(s_idx[(condition, b, metric)]["mean"], digits,
                                  bold=b in best))
            prefix = label if i == 0 else ""
            lines.append(f"{prefix} & {SHORT[b]} & " + " & ".join(cells) + r" \\")
        cells = []
        for metric, _h, digits in E2_COLS:
            pr = p_idx[(condition, "C2", metric)]
            cells.append(_signed(pr["mean_difference"], digits, int(pr["significant"])))
        lines.append(r" & $\Delta$(C2$-$C1) & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path = os.path.join(tables_dir, name + ".tex")
    common.write_text(path, "\n".join(lines) + "\n")

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{E2 round-3 companion: 95\% bootstrap confidence intervals "
        r"for every mean in Table~\ref{tab:e2-round3}, the paired C2$-$C1 "
        r"interval, and the fraction of the " + str(n_seeds) + r" paired seeds "
        r"on which C1 was better.}",
        r"\label{tab:e2-round3-intervals}",
        r"\begin{tabular}{@{}ll" + "r" * len(E2_COLS) + r"@{}}", r"\toprule",
        r"Intensity & Quantity & " + " & ".join(h for _, h, _ in E2_COLS) + r" \\",
        r"\midrule",
    ]
    for condition in ("Low", "Medium", "High"):
        for b in ("C1", "C2"):
            cells = [_ci(s_idx[(condition, b, m)]["ci95_lower"],
                         s_idx[(condition, b, m)]["ci95_upper"], d)
                     for m, _h, d in E2_COLS]
            lines.append(f"{condition} & {SHORT[b]} & " + " & ".join(cells) + r" \\")
        cells = [_ci(p_idx[(condition, "C2", m)]["ci95_lower"],
                     p_idx[(condition, "C2", m)]["ci95_upper"], d)
                 for m, _h, d in E2_COLS]
        lines.append(rf"{condition} & $\Delta$(C2$-$C1) & " + " & ".join(cells) + r" \\")
        cells = [f"{100*float(p_idx[(condition,'C2',m)]['frac_pairs_c1_better']):.0f}\\%"
                 for m, _h, _d in E2_COLS]
        lines.append(rf"{condition} & C1 better on & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path2 = os.path.join(tables_dir, name + "_intervals.tex")
    common.write_text(path2, "\n".join(lines) + "\n")
    return [path, path2]


# ==========================================================================
# E3 paired-advantage table
# ==========================================================================
def e3_paired_table(out_dir, tables_dir, name="E3_paired_advantage_50seeds"):
    paired = common.read_csv(os.path.join(out_dir, "paired_comparisons.csv"))
    pooled = [r for r in paired if r["trajectory_family"] == "all(pooled)"]
    abs_rows = {(r["condition"], r["rival"], r["metric"]): r
                for r in pooled if not r["degradation_type"]}
    deg_rows = {(r["condition"], r["rival"], r["degradation_type"]): r
                for r in pooled if r["degradation_type"]}
    cells = ["S00", "S01", "S02", "S10", "S11", "S12", "S20", "S21", "S22"]

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{E3 paired advantages over the $3\times3$ intensity matrix, "
        r"pooled over the three trajectory families by a cluster bootstrap over "
        r"seeds (the three families at one seed share that seed's initial state, "
        r"so they are averaged within the seed before resampling). "
        r"Entries are rival$-$C1 means over 50 paired seeds, so a POSITIVE "
        r"value means C1 had the lower error; $^{*}$ marks a 95\% CI excluding "
        r"zero. $A_{p}$ EE position RMSE (m), $A_{v}$ base XY velocity RMSE "
        r"(m/s), $A_{R}$ EE orientation MAE (deg), $A_{\Delta p}$ "
        r"locomotion-induced EE degradation (m), $A_{\Delta v}$ "
        r"manipulation-induced base degradation (m/s). Cell $S_{ij}$ has "
        r"$\alpha_{base}$ at level $i$ and $\alpha_{ee}$ at level $j$ "
        r"($0$, $0.5$, $1$). Per-seed values: "
        r"\texttt{final\_round3/E3/per\_seed.csv} and "
        r"\texttt{per\_seed\_degradation.csv}.}",
        r"\label{tab:e3-round3-paired}",
        r"\begin{tabular}{@{}ll" + "r" * len(cells) + r"@{}}", r"\toprule",
        r"Rival & Quantity & " + " & ".join(cells) + r" \\",
        r"\midrule",
    ]
    spec = [
        ("ee_position_rmse_m", r"$A_{p}$", 4, "abs"),
        ("base_vxy_rmse_mps", r"$A_{v}$", 4, "abs"),
        ("ee_orientation_mae_deg", r"$A_{R}$", 2, "abs"),
        ("locomotion_induced_ee", r"$A_{\Delta p}$", 4, "deg"),
        ("manipulation_induced_base", r"$A_{\Delta v}$", 4, "deg"),
    ]
    for rival in ("C3", "C5"):
        lines.append(rf"\multirow{{{len(spec)}}}{{*}}{{{rival}$-$C1}}")
        for i, (key, head, digits, kind) in enumerate(spec):
            row_cells = []
            for cell in cells:
                src = abs_rows if kind == "abs" else deg_rows
                r = src.get((cell, rival, key))
                row_cells.append("--" if r is None
                                 else _signed(r["mean_difference"], digits,
                                              int(r["significant"])))
            prefix = "" if i else ""
            lines.append(f"{prefix} & {head} & " + " & ".join(row_cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path = os.path.join(tables_dir, name + ".tex")
    common.write_text(path, "\n".join(lines) + "\n")
    return [path]


# ==========================================================================
# Selected illustrative cases
# ==========================================================================
SELECTED_SPEC = {
    "E1": (["C1", "C3", "C5"],
           [("base_vxy_rmse_mps", r"$E_v$ (m/s)", 4),
            ("base_yaw_rmse_radps", r"$E_\omega$ (rad/s)", 4),
            ("ee_position_rmse_m", r"$E_p$ (m)", 4),
            ("ee_orientation_mae_deg", r"$E_R$ (deg)", 2)]),
    "E2": (["C1", "C2"],
           [("base_vxy_rmse_mps", r"$E_v$ (m/s)", 4),
            ("base_yaw_rmse_radps", r"$E_\omega$ (rad/s)", 4),
            ("total_power_w", r"$P_{\mathrm{tot}}$ (W)", 1),
            ("leg_power_w", r"$P_{\mathrm{leg}}$ (W)", 1),
            ("leg_acc_rms_radps2", r"$a_{\mathrm{leg}}$ (rad/s$^2$)", 1)]),
    "E3": (["C1", "C3", "C5"],
           [("ee_position_rmse_m", r"$E_p$ (m)", 4),
            ("base_vxy_rmse_mps", r"$E_v$ (m/s)", 4),
            ("ee_orientation_mae_deg", r"$E_R$ (deg)", 2),
            ("locomotion_induced_ee", r"$\Delta E_p$ (m)", 4),
            ("manipulation_induced_base", r"$\Delta E_v$ (m/s)", 4)]),
}


def selected_table(experiment, out_dir, tables_dir):
    rows = common.read_csv(os.path.join(out_dir, "selected_3_cases.csv"))
    baselines, metrics = SELECTED_SPEC[experiment]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{Selected illustrative trials} for " + experiment +
        r" --- three single episodes, chosen by the pre-declared criteria in "
        r"\texttt{round3/select3.py} from the official 50-seed pool. "
        r"\textbf{These are individual trials, not averages}: they carry no "
        r"uncertainty and they are \emph{not} evidence about mean performance, "
        r"which is reported in the 50-seed tables. Selecting a seed here does "
        r"not change any aggregate --- all 50 seeds are always used. Each case "
        r"gives its rank among the eligible candidates, so its position in the "
        r"distribution is explicit. Every metric in this table is shown for "
        r"every compared method, including those on which C1 is worse; the "
        r"complete metric set for each trial is in "
        r"\texttt{selected\_3\_cases.csv} and the full ranking of all "
        r"candidates in "
        r"\texttt{showcase/" + experiment + r"\_candidate\_ranking.csv}. "
        r"``--'' marks a quantity that is undefined for the cell (a degradation "
        r"at its own reference condition).}",
        r"\label{tab:" + experiment.lower() + r"-round3-selected}",
        r"\begin{tabular}{@{}ll" + "r" * len(baselines) + r"@{}}", r"\toprule",
        r"Case / seed & Metric & " + " & ".join(SHORT[b] for b in baselines) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        header = (row["case"].replace("_", r"\_") + r", seed "
                  + str(row["selected_seed"]))
        # E3 condition IDs contain an underscore (S02_circle), which is a
        # compile error outside math mode. The structural checker cannot see
        # this, so it is escaped at the source.
        cond = row["condition"].replace("_", r"\_")
        rank = (f"rank {row['rank_among_eligible']}/{row['n_eligible']} eligible "
                f"of {row['n_candidates']}")
        lines.append(
            r"\multicolumn{" + str(2 + len(baselines)) + r"}{@{}l@{}}{\itshape "
            + header + " (" + cond + "; " + rank + r")} \\"
        )
        for metric, head, digits in metrics:
            values = {}
            for b in baselines:
                key = f"{b}_{metric}"
                values[b] = row.get(key, "")
            finite = {b: float(v) for b, v in values.items()
                      if v not in ("", None) and np.isfinite(float(v))}
            best = min(finite, key=finite.get) if finite else None
            cells = [
                _fmt(values[b] if values[b] not in ("", None) else None, digits,
                     bold=(b == best))
                for b in baselines
            ]
            lines.append(f" & {head} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(tables_dir, f"{experiment}_selected_3_cases.tex")
    common.write_text(path, "\n".join(lines) + "\n")
    return [path]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=common.OUT)
    args = parser.parse_args(argv)
    tables_dir = os.path.join(args.out, "tables")
    written = []
    written += e1_tables(os.path.join(args.out, "E1"), tables_dir)
    written += e2_tables(os.path.join(args.out, "E2"), tables_dir)
    written += e3_paired_table(os.path.join(args.out, "E3"), tables_dir)
    for experiment in ("E1", "E2", "E3"):
        written += selected_table(experiment, os.path.join(args.out, experiment),
                                  tables_dir)
    for path in written:
        print(f"  wrote {os.path.relpath(path, common.BENCH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
