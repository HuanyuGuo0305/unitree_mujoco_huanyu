#!/usr/bin/env python3
"""E1 official analysis, 500 paired seeds (0-499): validation, statistics, RA-L table.

All 6,000 episodes are in results/E1/raw/E1. Seeds 0-49 are the round-3 runs and
seeds 50-499 the extension; both were moved there unchanged in the 2026-11-12 cleanup.
Seeds 0-49 are cross-checked against the frozen round-3 v2 reference CSVs in
results/E1/reference_round3_50seed/.

All four metrics are RMSEs over the evaluated ticks (3-30 s window, truncated
at termination), recomputed here from the per-tick logs and cross-checked
against the run-time metrics.json:
  Base XY  sqrt(mean((vx-vx*)^2 + (vy-vy*)^2))   body frame       [m/s]
  Yaw      sqrt(mean((wz-wz*)^2))                body frame       [rad/s]
  EE Pos   sqrt(mean(||p - p*||^2))              PLB frame        [m]
  EE Ori   sqrt(mean(theta^2)), theta = SO(3) geodesic, PLB frame [deg]
Published values follow the v2 table exactly: Base XY / Yaw / EE Pos from
metrics.json (full float64), EE Ori recomputed from the logged keypoints.

Run:  python3 -m benchmark.e1.analyze            (from simulate_python/)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from benchmark.round3 import common  # first: puts the repo root on sys.path
from benchmark.core import frames, stats  # noqa: E402

BENCH = common.BENCH
OUT = os.path.join(BENCH, "results", "E1")
RAW_OLD = RAW_NEW = os.path.join(OUT, "raw", "E1")      # one tree, seeds 0-499
V2_REF = os.path.join(OUT, "reference_round3_50seed")
V2_SUMMARY = os.path.join(V2_REF, "E1_50seed_mean_sd_v2_oriRMSE.csv")
V2_PER_SEED = os.path.join(V2_REF, "E1_per_seed_600rows_v2_oriRMSE.csv")

# Constants formerly imported from round3/paper_tables_e1.py (values unchanged; they
# label the CSV rows and the TeX table exactly as in the published outputs).
CONDITIONS, BASELINES = ["T1", "T2", "T3", "T4"], ["C1", "C3", "C5"]
METHOD_LABEL = {"C1": "C1 Unified RL WBC (Ours)", "C3": "C3 RL Locomotion + IK",
                "C5": "C5 Cooperative Dual-PPO"}
METHOD_SHORT_TEX = {"C1": r"C1 (\textbf{Ours})", "C3": "C3", "C5": "C5"}
N_SEEDS = 500
OLD_SEEDS = range(0, 50)
CONDITION_LABEL = {"T1": "Still / Fixed", "T2": "Still / Random",
                   "T3": "Moving / Fixed", "T4": "Moving / Random"}
# (key, header, unit, mean digits, metrics.json key)
METRICS = [
    ("base_xy_rmse_mps", "Base XY", r"m\,s$^{-1}$", 4, "base_vxy_rmse"),
    ("yaw_rmse_radps", "Yaw", r"rad\,s$^{-1}$", 4, "base_wz_rmse"),
    ("ee_pos_rmse_m", "EE Pos", "m", 4, "ee_pos_rmse"),
    ("ee_ori_rmse_deg", "EE Ori", r"$^{\circ}$", 2, "ee_ori_rmse_deg"),
]
UNITS = {"base_xy_rmse_mps": "m/s", "yaw_rmse_radps": "rad/s",
         "ee_pos_rmse_m": "m", "ee_ori_rmse_deg": "deg", "ee_ori_mae_deg": "deg"}
SD_DIGITS = 3
COMPARISONS = [("C1", "C3"), ("C1", "C5"), ("C3", "C5")]   # (ref, other): other - ref
# metadata keys that legitimately differ by seed or by episode outcome
OUTCOME_KEYS = {"initial_state", "scenario", "seed", "termination_time_s", "survived",
                "termination_reason", "valid_eval_duration_s", "uan_calls",
                "policy_ticks", "physics_steps"}
TOL = {"base_xy_rmse_mps": 1e-6, "yaw_rmse_radps": 1e-6, "ee_pos_rmse_m": 1e-6,
       "ee_ori_rmse_deg": 1e-4}
CAPTION = (r"Base-velocity and EE-pose tracking RMSEs (mean $\pm$ standard "
           r"deviation) over 500 random seeds. Bold values indicate the lowest "
           r"mean errors.")


class ValidationError(SystemExit):
    pass


def ep_dir(baseline, condition, seed):
    root = RAW_OLD if seed in OLD_SEEDS else RAW_NEW
    return os.path.join(root, baseline, condition, f"seed_{seed:03d}")


def _read(path):
    with open(path, newline="", encoding="utf-8") as h:
        rows = list(csv.reader(h))
    data = np.asarray(rows[1:], dtype=np.float64)
    return {k: data[:, i] for i, k in enumerate(rows[0])}


def _sha(path):
    with open(path, "rb") as h:
        return hashlib.sha256(h.read()).hexdigest()


def episode(job):
    """Load one episode, recompute all metrics from the per-tick logs."""
    baseline, condition, seed = job
    d = ep_dir(baseline, condition, seed)
    out = {"baseline": baseline, "condition": condition, "seed": seed,
           "path": os.path.relpath(d, BENCH), "problems": []}
    for f in ("commands.csv", "states.csv", "metrics.json", "metadata.json",
              "scenario_commands.npz"):
        if not os.path.isfile(os.path.join(d, f)):
            out["problems"].append(f"missing {f}")
    if out["problems"]:
        return out
    met = json.load(open(os.path.join(d, "metrics.json")))
    meta = json.load(open(os.path.join(d, "metadata.json")))
    cmd, st = _read(os.path.join(d, "commands.csv")), _read(os.path.join(d, "states.csv"))
    ec, es = cmd["evaluated"] == 1, st["evaluated"] == 1
    if not np.array_equal(cmd["tick"][ec], st["tick"][es]):
        out["problems"].append("evaluated ticks differ between commands and states")
        return out
    n = int(es.sum())
    rec = {}
    if n:
        dvx = st["vx_act"][es] - cmd["vx_cmd"][ec]
        dvy = st["vy_act"][es] - cmd["vy_cmd"][ec]
        dwz = st["wz_act"][es] - cmd["wz_cmd"][ec]
        kc = np.stack([cmd[f"ee_cmd_kp{i}"][ec] for i in range(9)], 1)
        ka = np.stack([st[f"ee_act_kp{i}"][es] for i in range(9)], 1)
        Rc = np.stack([frames.rotation_from_keypoints(k) for k in kc])
        Ra = np.stack([frames.rotation_from_keypoints(k) for k in ka])
        tr = np.einsum("nij,nij->n", Rc, Ra)
        theta = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
        perr = np.linalg.norm(ka[:, :3] - kc[:, :3], axis=1)
        rec = {
            "base_xy_rmse_mps": float(np.sqrt(np.mean(dvx ** 2 + dvy ** 2))),
            "yaw_rmse_radps": float(np.sqrt(np.mean(dwz ** 2))),
            "ee_pos_rmse_m": float(np.sqrt(np.mean(perr ** 2))),
            "ee_ori_rmse_deg": float(np.sqrt(np.mean(theta ** 2))),
            "ee_ori_mae_recomputed": float(np.mean(theta)),
            "theta_vs_logged_max_abs": float(np.max(np.abs(theta - st["ee_ori_err_deg"][es]))),
            "t_first_eval": float(st["t"][es][0]),
        }
    out.update({
        "n_eval_samples": n,
        "metrics_json_n": int(met["n_eval_samples"]),
        "survived": int(met["survived"]),
        "termination_reason": met["termination_reason"],
        "valid_evaluation_time_s": float(met["valid_eval_duration_s"]),
        "recomputed": rec,
        "metrics_json": {k: float(met[mk]) for k, _h, _u, _d, mk in METRICS},
        "ee_ori_mae_deg": float(met["ee_ori_mae_deg"]),
        "meta": {k: v for k, v in meta.items() if k not in OUTCOME_KEYS},
        "scenario": {k: v for k, v in meta["scenario"].items() if k != "seed"},
        "meta_seed": meta["seed"], "scenario_seed": meta["scenario"]["seed"],
        "init_qpos": meta["initial_state"]["qpos"],
        "init_qvel": meta["initial_state"].get("qvel"),
        "scenario_sha": _sha(os.path.join(d, "scenario_commands.npz")),
        "cmd_cols_sha": hashlib.sha256(np.stack(
            [cmd[k][ec] for k in ["vx_cmd", "vy_cmd", "wz_cmd"]
             + [f"ee_cmd_kp{i}" for i in range(9)]], 1).tobytes()).hexdigest(),
    })
    return out


# --------------------------------------------------------------------------
def validate(eps, n_seeds):
    P = []
    seen = {}
    for e in eps:
        key = (e["condition"], e["baseline"], e["seed"])
        tag = "/".join(map(str, key))
        if key in seen:
            P.append(f"duplicate {tag}")
        seen[key] = e
        if e["problems"]:
            P += [f"{tag}: {p}" for p in e["problems"]]
            continue
        if e["meta_seed"] != e["seed"] or e["scenario_seed"] != e["seed"]:
            P.append(f"{tag}: metadata seed {e['meta_seed']}/{e['scenario_seed']}")
        if e["n_eval_samples"] != e["metrics_json_n"]:
            P.append(f"{tag}: {e['n_eval_samples']} evaluated ticks vs metrics.json "
                     f"{e['metrics_json_n']}")
        if e["n_eval_samples"] == 0:
            P.append(f"{tag}: zero evaluated ticks (terminated during settling)")
            continue
        if e["survived"] and e["n_eval_samples"] != 1350:
            P.append(f"{tag}: survived with {e['n_eval_samples']} != 1350 ticks")
        if not e["survived"] and e["n_eval_samples"] >= 1350:
            P.append(f"{tag}: terminated but full window")
        if abs(e["recomputed"]["t_first_eval"] - 3.0) > 0.021:
            P.append(f"{tag}: evaluation starts at {e['recomputed']['t_first_eval']}")
        for k, *_ in METRICS:
            a, b = e["recomputed"][k], e["metrics_json"][k]
            if not np.isfinite(b) or abs(a - b) > TOL[k] * max(1.0, abs(b)):
                P.append(f"{tag}: {k} recomputed {a!r} vs metrics.json {b!r}")
        if e["recomputed"]["theta_vs_logged_max_abs"] > 1e-3:
            P.append(f"{tag}: geodesic differs from logged by "
                     f"{e['recomputed']['theta_vs_logged_max_abs']}")
        if abs(e["recomputed"]["ee_ori_mae_recomputed"] - e["ee_ori_mae_deg"]) > 1e-4:
            P.append(f"{tag}: mean(theta) does not reproduce the MAE")
        if e["recomputed"]["ee_ori_rmse_deg"] < e["recomputed"]["ee_ori_mae_recomputed"] - 1e-12:
            P.append(f"{tag}: RMSE < MAE")
    # counts
    for c in CONDITIONS:
        for b in BASELINES:
            got = sorted(s for (cc, bb, s) in seen if cc == c and bb == b)
            if got != list(range(n_seeds)):
                miss = sorted(set(range(n_seeds)) - set(got))
                P.append(f"{c}/{b}: {len(got)} seeds, missing {miss[:8]}")
    # protocol identity: every episode of a (baseline, condition) cell carries the
    # same non-outcome metadata as its seed-0 episode (the original run)
    for c in CONDITIONS:
        for b in BASELINES:
            ref = seen.get((c, b, 0))
            if ref is None or ref["problems"]:
                continue
            for s in range(1, n_seeds):
                e = seen.get((c, b, s))
                if e is None or e["problems"]:
                    continue
                if e["meta"] != ref["meta"] or e["scenario"] != ref["scenario"]:
                    diff = sorted(k for k in set(e["meta"]) | set(ref["meta"])
                                  if e["meta"].get(k) != ref["meta"].get(k))
                    if e["scenario"] != ref["scenario"]:
                        diff.append("scenario")
                    P.append(f"{c}/{b}/{s}: protocol metadata differs from seed 0: {diff}")
    # pairing: identical command sequence and initial state across controllers
    for c in CONDITIONS:
        for s in range(n_seeds):
            es_ = [seen.get((c, b, s)) for b in BASELINES]
            if any(x is None or x["problems"] for x in es_):
                continue
            if len({x["scenario_sha"] for x in es_}) != 1:
                P.append(f"{c}/seed {s}: scenario_commands.npz differs across controllers")
            if len({json.dumps(x["init_qpos"]) for x in es_}) != 1 or \
               len({json.dumps(x["init_qvel"]) for x in es_}) != 1:
                P.append(f"{c}/seed {s}: initial state differs across controllers")
            # commands actually issued: identical over the ticks all three evaluated
            full = [x for x in es_ if x["survived"]]
            if len({x["cmd_cols_sha"] for x in full}) > 1:
                P.append(f"{c}/seed {s}: issued commands differ across controllers")
    return seen, P


# --------------------------------------------------------------------------
def per_seed_rows(seen, n_seeds):
    rows = []
    for c in CONDITIONS:
        for b in BASELINES:
            for s in range(n_seeds):
                e = seen[(c, b, s)]
                r = {"condition": c, "condition_label": CONDITION_LABEL[c],
                     "baseline": b, "method": METHOD_LABEL[b], "seed": s,
                     "seed_source": "round3_official_0_49" if s in OLD_SEEDS
                     else "E1_500seed_extension"}
                for k, _h, _u, _d, _mk in METRICS:
                    r[k] = e["recomputed"]["ee_ori_rmse_deg"] if k == "ee_ori_rmse_deg" \
                        else e["metrics_json"][k]
                r["ee_ori_mae_deg"] = e["ee_ori_mae_deg"]
                r.update({"survived": e["survived"],
                          "termination_reason": e["termination_reason"],
                          "valid_evaluation_time_s": e["valid_evaluation_time_s"],
                          "n_eval_samples": e["n_eval_samples"], "path": e["path"]})
                rows.append(r)
    return rows


def _vals(rows, c, b, k, seeds):
    return {r["seed"]: r[k] for r in rows
            if r["condition"] == c and r["baseline"] == b and r["seed"] in seeds}


def summary(rows, seeds, cfg):
    out = []
    for c in CONDITIONS:
        for b in BASELINES:
            for k in [m[0] for m in METRICS] + ["ee_ori_mae_deg"]:
                v = np.asarray([x for _s, x in sorted(_vals(rows, c, b, k, seeds).items())])
                lo, hi = stats.bootstrap_ci(v, cfg["bootstrap_resamples"],
                                            cfg["bootstrap_seed"], cfg["confidence"])
                sel = [r for r in rows if r["condition"] == c and r["baseline"] == b
                       and r["seed"] in seeds]
                out.append({
                    "condition": c, "condition_label": CONDITION_LABEL[c],
                    "baseline": b, "method": METHOD_LABEL[b], "metric": k,
                    "unit": UNITS[k], "in_paper_table": int(k != "ee_ori_mae_deg"),
                    "n_seeds": int(v.size), "mean": float(v.mean()),
                    "sd_ddof1": float(v.std(ddof=1)), "ci95_lower": lo, "ci95_upper": hi,
                    "median": float(np.median(v)), "min": float(v.min()),
                    "max": float(v.max()),
                    "n_completed": int(sum(r["survived"] for r in sel)),
                    "bootstrap_resamples": cfg["bootstrap_resamples"],
                    "bootstrap_seed": cfg["bootstrap_seed"],
                })
    return out


def paired(rows, seeds, cfg):
    out = []
    for c in CONDITIONS:
        for k in [m[0] for m in METRICS]:
            for ref, oth in COMPARISONS:
                a = stats.paired_advantage(_vals(rows, c, ref, k, seeds),
                                           _vals(rows, c, oth, k, seeds),
                                           cfg["bootstrap_resamples"],
                                           cfg["bootstrap_seed"], cfg["confidence"])
                out.append({
                    "condition": c, "metric": k, "unit": UNITS[k],
                    "comparison": f"{oth}_minus_{ref}",
                    "sign_convention": f"positive => {ref} has the lower error",
                    "n_pairs": a["n_pairs"], "mean_diff": a["mean"],
                    "median_diff": a["median"], "sd_diff": a["std"],
                    "ci95_lower": a["ci_low"], "ci95_upper": a["ci_high"],
                    "significant": int(a["significant"]),
                    "favours": (ref if a["mean"] > 0 else oth) if a["significant"] else "n.s.",
                    "frac_ref_lower": a["frac_positive"],
                    "bootstrap_resamples": cfg["bootstrap_resamples"],
                    "bootstrap_seed": cfg["bootstrap_seed"],
                })
    return out


def completion(rows, seeds):
    out = []
    for c in CONDITIONS:
        for b in BASELINES:
            sel = [r for r in rows if r["condition"] == c and r["baseline"] == b
                   and r["seed"] in seeds]
            k, n = sum(r["survived"] for r in sel), len(sel)
            lo, hi = stats.wilson_interval(k, n)
            reasons = {}
            for r in sel:
                if not r["survived"]:
                    reasons[r["termination_reason"]] = reasons.get(r["termination_reason"], 0) + 1
            out.append({"condition": c, "baseline": b, "n_episodes": n,
                        "n_completed": k, "n_terminated": n - k,
                        "completion_rate": k / n, "wilson95_lower": lo, "wilson95_upper": hi,
                        "termination_reasons": ";".join(f"{a}:{v}" for a, v in sorted(reasons.items())),
                        "terminated_seeds": " ".join(str(r["seed"]) for r in sel if not r["survived"]),
                        "mean_valid_eval_time_s": float(np.mean([r["valid_evaluation_time_s"] for r in sel])),
                        "episodes_included_in_means": n})
    return out


# --------------------------------------------------------------------------
def build_tex(summ):
    idx = {(e["condition"], e["baseline"], e["metric"]): e for e in summ}
    L = [r"% Requires: booktabs, multirow. No vertical rules by design.",
         r"% E1, 500 paired seeds (0-499). All four metrics are RMSEs; EE Ori is the",
         r"% RMSE of the per-step SO(3) geodesic error. Means at the established",
         r"% precision, SDs at three decimals.",
         r"\begin{table*}[t]", r"\centering",
         r"\caption{" + CAPTION + r"}", r"\label{tab:wbc_tracking}",
         r"\setlength{\tabcolsep}{4pt}", r"\small",
         r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"& & \multicolumn{4}{c}{Tracking RMSE (lower is better)} \\",
         r"\cmidrule(l){3-6}",
         r"Condition (Base / EE) & Method & "
         + " & ".join(m[1] for m in METRICS) + r" \\",
         r" & & " + " & ".join(f"[{m[2]}]" for m in METRICS) + r" \\",
         r"\midrule"]
    for ci, c in enumerate(CONDITIONS):
        best = {k: min(BASELINES, key=lambda b: idx[(c, b, k)]["mean"])
                for k, *_ in METRICS}
        for bi, b in enumerate(BASELINES):
            cells = []
            for k, _h, _u, d, _mk in METRICS:
                e = idx[(c, b, k)]
                m = f"{e['mean']:.{d}f}"
                m = r"\mathbf{" + m + "}" if b == best[k] else m
                cells.append(f"${m} \\pm {e['sd_ddof1']:.{SD_DIGITS}f}$")
            head = (r"\multirow{3}{*}{" + f"{c}: {CONDITION_LABEL[c]}" + "}") if bi == 0 else ""
            L.append(head + " & " + METHOD_SHORT_TEX[b] + " & " + " & ".join(cells) + r" \\")
        if ci < len(CONDITIONS) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def compare_50_500(s50, s500, p50, p500):
    """Rankings and paired conclusions, 50-seed (0-49) vs 500-seed."""
    i50 = {(e["condition"], e["baseline"], e["metric"]): e for e in s50}
    i500 = {(e["condition"], e["baseline"], e["metric"]): e for e in s500}
    rank_rows, concl_rows = [], []
    for c in CONDITIONS:
        for k, *_ in METRICS:
            r50 = sorted(BASELINES, key=lambda b: i50[(c, b, k)]["mean"])
            r500 = sorted(BASELINES, key=lambda b: i500[(c, b, k)]["mean"])
            rank_rows.append({"condition": c, "metric": k,
                              "ranking_50seed": " < ".join(r50),
                              "ranking_500seed": " < ".join(r500),
                              "best_50seed": r50[0], "best_500seed": r500[0],
                              "best_changed": int(r50[0] != r500[0]),
                              "ranking_changed": int(r50 != r500),
                              **{f"mean50_{b}": i50[(c, b, k)]["mean"] for b in BASELINES},
                              **{f"mean500_{b}": i500[(c, b, k)]["mean"] for b in BASELINES}})
    j50 = {(a["condition"], a["metric"], a["comparison"]): a for a in p50}
    for a in p500:
        o = j50[(a["condition"], a["metric"], a["comparison"])]
        concl_rows.append({"condition": a["condition"], "metric": a["metric"],
                           "comparison": a["comparison"],
                           "mean_diff_50": o["mean_diff"], "ci50_lower": o["ci95_lower"],
                           "ci50_upper": o["ci95_upper"], "favours_50": o["favours"],
                           "mean_diff_500": a["mean_diff"], "ci500_lower": a["ci95_lower"],
                           "ci500_upper": a["ci95_upper"], "favours_500": a["favours"],
                           "conclusion_changed": int(o["favours"] != a["favours"]),
                           "ci_width_ratio_500_over_50":
                               (a["ci95_upper"] - a["ci95_lower"]) / (o["ci95_upper"] - o["ci95_lower"])})
    return rank_rows, concl_rows


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=OUT, help="output directory (default results/E1)")
    args = ap.parse_args(argv)
    out_dir = args.out
    n_seeds = args.n_seeds
    cfg = common.stats_config()
    jobs = [(b, c, s) for c in CONDITIONS for b in BASELINES for s in range(n_seeds)]
    print(f"[1] loading {len(jobs)} episodes and recomputing metrics from per-tick logs")
    with ProcessPoolExecutor(args.workers) as ex:
        eps = list(ex.map(episode, jobs, chunksize=20))

    print("[2] validating counts, pairing, protocol identity and metric consistency")
    seen, P = validate(eps, n_seeds)
    if P:
        for p in P[:30]:
            print("  FAIL " + p, file=sys.stderr)
        raise ValidationError(f"{len(P)} validation failure(s)")
    print(f"  {len(seen)} episodes = 4 x 3 x {n_seeds}; seeds paired; protocol metadata "
          f"identical to seed 0 in every cell; all metrics reproduced from per-tick logs")

    rows = per_seed_rows(seen, n_seeds)
    # the 50-seed subset must be exactly the published v2 per-seed values
    v2 = {(r["condition"], r["baseline"], int(r["seed"])): r for r in common.read_csv(V2_PER_SEED)}
    m = {"base_xy_rmse_mps": "base_vxy_rmse_mps", "yaw_rmse_radps": "base_yaw_rmse_radps",
         "ee_pos_rmse_m": "ee_position_rmse_m", "ee_ori_rmse_deg": "ee_orientation_rmse_deg",
         "ee_ori_mae_deg": "ee_orientation_mae_deg"}
    bad = [(r["condition"], r["baseline"], r["seed"], k) for r in rows if r["seed"] < 50
           for k, kk in m.items() if r[k] != float(v2[(r["condition"], r["baseline"], r["seed"])][kk])]
    if bad:
        raise ValidationError(f"{len(bad)} seed 0-49 values differ from v2: {bad[:5]}")
    print("  seeds 0-49: all 600 x 5 values bit-identical to the published v2 per-seed CSV")

    print("[3] statistics")
    s500, p500, comp = summary(rows, range(n_seeds), cfg), paired(rows, range(n_seeds), cfg), \
        completion(rows, range(n_seeds))
    s50, p50 = summary(rows, OLD_SEEDS, cfg), paired(rows, OLD_SEEDS, cfg)
    v2s = {(r["condition"], r["baseline"], r["metric"]): r for r in common.read_csv(V2_SUMMARY)}
    for e in s50:
        kk = m[e["metric"]]
        ref = v2s[(e["condition"], e["baseline"], kk)]
        for a, b in (("mean", "mean"), ("sd_ddof1", "sd_ddof1"),
                     ("ci95_lower", "ci95_lower"), ("ci95_upper", "ci95_upper")):
            if e[a] != float(ref[b]):
                raise ValidationError(f"50-seed {e['condition']}/{e['baseline']}/{kk} {a} "
                                      f"!= published v2 summary")
    print("  50-seed subset summary bit-identical to the published v2 summary")
    rank, concl = compare_50_500(s50, s500, p50, p500)

    print("[4] writing outputs")
    W = [
        common.write_csv(os.path.join(out_dir, "E1_500seed_per_seed_6000rows.csv"), rows),
        common.write_csv(os.path.join(out_dir, "E1_500seed_summary.csv"), s500),
        common.write_csv(os.path.join(out_dir, "E1_500seed_paired_comparisons.csv"), p500),
        common.write_csv(os.path.join(out_dir, "E1_500seed_completion_rates.csv"), comp),
        common.write_csv(os.path.join(out_dir, "E1_50seed_subset_paired_comparisons.csv"), p50),
        common.write_csv(os.path.join(out_dir, "E1_50_vs_500_rankings.csv"), rank),
        common.write_csv(os.path.join(out_dir, "E1_50_vs_500_paired_conclusions.csv"), concl),
        common.write_text(os.path.join(out_dir, "E1_500seed_tracking_rmse.tex"), build_tex(s500)),
    ]
    for w in W:
        print("  " + os.path.relpath(w, BENCH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
