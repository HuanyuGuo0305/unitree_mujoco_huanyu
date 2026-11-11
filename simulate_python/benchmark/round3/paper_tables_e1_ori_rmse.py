#!/usr/bin/env python3
"""E1 tracking table, version 2: EE orientation reported as RMSE, not MAE.

No simulation is run. For every one of the 600 official E1 episodes the EE
orientation RMSE is recomputed from the PER-TIMESTEP log, never from the
episode-level MAE:

    theta_t = arccos(clip((tr(R_cmd,t^T R_ee,t) - 1) / 2, -1, 1))
    RMSE_R  = sqrt(mean_t theta_t^2) * 180/pi

R_cmd,t and R_ee,t are rebuilt from the logged commanded (commands.csv) and
actual (states.csv) 9-D keypoints, both in the PLB frame, with the benchmark's
own ``frames.rotation_from_keypoints`` (a verbatim port of C3's convention).
The timesteps are exactly those flagged ``evaluated == 1`` (the 3-30 s window,
truncated at termination for early-terminated episodes), joined on ``tick`` -
the same samples every other E1 metric uses.

Three independent cross-checks per episode:
  * theta_t vs the geodesic the recorder logged online (``ee_ori_err_deg``);
  * RMSE vs ``ee_ori_rmse_deg`` written to metrics.json at run time;
  * mean(theta_t) vs the published MAE, which proves the sample set matches.

The other three metrics and the MAE are copied unchanged and are asserted to
equal the version-1 published table. Version-1 files are not modified.

Run:  python3 -m benchmark.round3.paper_tables_e1_ori_rmse
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

from benchmark.round3 import common  # first: puts the repo root on sys.path
from benchmark.core import frames, stats  # noqa: E402
from benchmark.round3 import paper_tables_e1 as v1

OUT_DIR = v1.OUT_DIR
TAG = "v2_oriRMSE"
CONDITIONS, BASELINES = v1.CONDITIONS, v1.BASELINES

# (key, header, unit, digits); orientation is now an RMSE.
METRICS = [
    ("base_vxy_rmse_mps", r"Base XY", r"m\,s$^{-1}$", 4),
    ("base_yaw_rmse_radps", r"Yaw", r"rad\,s$^{-1}$", 4),
    ("ee_position_rmse_m", r"EE Pos", "m", 4),
    ("ee_orientation_rmse_deg", r"EE Ori.\ RMSE", r"$^{\circ}$", 2),
]
UNCHANGED = ["base_vxy_rmse_mps", "base_yaw_rmse_radps", "ee_position_rmse_m"]

CAPTION = (r"Base-velocity and EE-pose tracking RMSEs (mean $\pm$ standard "
           r"deviation) over 50 random seeds. Bold values indicate the lowest "
           r"mean errors.")
LABEL = "tab:wbc_tracking"

TOL_THETA_DEG = 1e-3     # per-step: CSV keypoints carry ~9 significant digits
TOL_RMSE_DEG = 1e-4      # episode RMSE vs metrics.json
TOL_MAE_DEG = 1e-4       # mean(theta) vs published MAE


class ValidationError(SystemExit):
    pass


def _read(path):
    with open(path, newline="", encoding="utf-8") as h:
        rows = list(csv.reader(h))
    head = rows[0]
    data = np.asarray(rows[1:], dtype=np.float64)
    return {k: data[:, i] for i, k in enumerate(head)}


def geodesic_deg_batch(R_a, R_b):
    """Vectorised geodesic angle, same formula as frames.geodesic_angle_deg."""
    tr = np.einsum("nij,nij->n", R_a, R_b)          # tr(R_a^T R_b)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def episode_orientation(row) -> dict:
    ep = os.path.join(common.BENCH, row["path"])
    cmd = _read(os.path.join(ep, "commands.csv"))
    st = _read(os.path.join(ep, "states.csv"))
    with open(os.path.join(ep, "metrics.json"), encoding="utf-8") as h:
        met = json.load(h)

    ev_c = cmd["evaluated"] == 1
    ev_s = st["evaluated"] == 1
    tick_c, tick_s = cmd["tick"][ev_c].astype(int), st["tick"][ev_s].astype(int)
    if not np.array_equal(tick_c, tick_s):
        raise ValidationError(f"{row['path']}: evaluated ticks differ between "
                              f"commands.csv and states.csv")
    n = int(tick_s.size)

    kp_c = np.stack([cmd[f"ee_cmd_kp{i}"][ev_c] for i in range(9)], axis=1)
    kp_a = np.stack([st[f"ee_act_kp{i}"][ev_s] for i in range(9)], axis=1)
    R_c = np.stack([frames.rotation_from_keypoints(k) for k in kp_c])
    R_a = np.stack([frames.rotation_from_keypoints(k) for k in kp_a])
    theta = geodesic_deg_batch(R_c, R_a)
    for R in (R_c[:3], R_a[:3]):   # rebuilt matrices really are rotations
        assert np.allclose(R @ np.swapaxes(R, 1, 2), np.eye(3), atol=1e-9)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-9)
    # spot-check the vectorised formula against the scalar benchmark function
    for i in (0, n // 2, n - 1):
        assert abs(theta[i] - frames.geodesic_angle_deg(R_c[i], R_a[i])) < 1e-9

    theta_logged = st["ee_ori_err_deg"][ev_s]
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(theta ** 2))),
        "mae_recomputed": float(np.mean(theta)),
        "rmse_from_logged_theta": float(np.sqrt(np.mean(theta_logged ** 2))),
        "max_abs_theta_diff": float(np.max(np.abs(theta - theta_logged))),
        "theta_max": float(theta.max()),
        "metrics_json_rmse": float(met["ee_ori_rmse_deg"]),
        "metrics_json_mae": float(met["ee_ori_mae_deg"]),
        "metrics_json_n": int(met["n_eval_samples"]),
        "t_first": float(st["t"][ev_s][0]), "t_last": float(st["t"][ev_s][-1]),
    }


def summarize(rows, cfg):
    out = []
    for c in CONDITIONS:
        for b in BASELINES:
            sel = sorted((r for r in rows if r["condition"] == c and r["baseline"] == b),
                         key=lambda r: r["seed"])
            for key, _h, unit, _d in METRICS + [("ee_orientation_mae_deg", "", "deg", 2)]:
                v = np.asarray([r[key] for r in sel], dtype=np.float64)
                lo, hi = stats.bootstrap_ci(v, n_resamples=cfg["bootstrap_resamples"],
                                            seed=cfg["bootstrap_seed"],
                                            confidence=cfg["confidence"])
                out.append({
                    "condition": c, "condition_title": v1.CONDITION_TITLES[c],
                    "baseline": b, "method": v1.METHOD_LABEL[b], "metric": key,
                    "unit": {"base_vxy_rmse_mps": "m/s", "base_yaw_rmse_radps": "rad/s",
                             "ee_position_rmse_m": "m"}.get(key, "deg"),
                    "in_paper_table": int(key != "ee_orientation_mae_deg"),
                    "n_seeds": int(v.size), "mean": float(v.mean()),
                    "sd_ddof1": float(v.std(ddof=1)),
                    "ci95_lower": lo, "ci95_upper": hi,
                    "median": float(np.median(v)), "min": float(v.min()),
                    "max": float(v.max()),
                    "n_survived": int(sum(r["survived"] for r in sel)),
                    "bootstrap_resamples": cfg["bootstrap_resamples"],
                    "bootstrap_seed": cfg["bootstrap_seed"],
                })
    return out


def build_tex(summary):
    idx = {(e["condition"], e["baseline"], e["metric"]): e for e in summary}
    L = [
        r"% Requires: booktabs, multirow. No vertical rules by design.",
        r"% Version 2: EE orientation reported as RMSE of the per-step geodesic",
        r"% error (version 1 reported MAE). Other three metrics unchanged.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{" + CAPTION + r"}",
        r"\label{" + LABEL + r"}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{Tracking RMSE (lower is better)} \\",
        r"\cmidrule(l){3-6}",
        r"Condition & Method & " + " & ".join(h for _k, h, _u, _d in METRICS) + r" \\",
        r" & & " + " & ".join(f"[{u}]" for _k, _h, u, _d in METRICS) + r" \\",
        r"\midrule",
    ]
    for ci, c in enumerate(CONDITIONS):
        best = {k: min(BASELINES, key=lambda b: idx[(c, b, k)]["mean"])
                for k, _h, _u, _d in METRICS}
        for bi, b in enumerate(BASELINES):
            cells = []
            for k, _h, _u, d in METRICS:
                e = idx[(c, b, k)]
                m = f"{e['mean']:.{d}f}"
                m = r"\mathbf{" + m + "}" if b == best[k] else m
                cells.append(f"${m} \\pm {e['sd_ddof1']:.{d}f}$")
            head = (r"\multirow{3}{*}{" + c + r"\,{\scriptsize "
                    + v1.CONDITION_SHORT[c] + r"}}") if bi == 0 else ""
            L.append(head + " & " + v1.METHOD_SHORT_TEX[b] + " & "
                     + " & ".join(cells) + r" \\")
        if ci < len(CONDITIONS) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def main() -> int:
    cfg = common.stats_config()
    print("[1] loading the 600 official E1 records (version-1 validator)")
    rows = v1.load_and_validate(verbose=True)
    if len(rows) != 600:
        raise ValidationError(f"{len(rows)} records, expected 600")

    print("[2] recomputing EE orientation RMSE from per-timestep keypoints")
    problems, diag = [], []
    for r in rows:
        o = episode_orientation(r)
        tag = f"{r['condition']}/{r['baseline']}/seed {r['seed']:02d}"
        n_pub = int(float(r["n_eval_samples"]))
        if o["n"] != n_pub or o["n"] != o["metrics_json_n"]:
            problems.append(f"{tag}: {o['n']} evaluated steps vs published {n_pub}")
        if o["n"] == 0:
            problems.append(f"{tag}: no evaluated timesteps")
        if abs(o["t_first"] - 3.0) > 0.021:
            problems.append(f"{tag}: window starts at {o['t_first']} s, not 3 s")
        if r["survived"] and o["n"] != 1350:
            problems.append(f"{tag}: survived but {o['n']} != 1350 steps")
        if o["max_abs_theta_diff"] > TOL_THETA_DEG:
            problems.append(f"{tag}: per-step theta differs from logged by "
                            f"{o['max_abs_theta_diff']:.2e} deg")
        if abs(o["rmse"] - o["metrics_json_rmse"]) > TOL_RMSE_DEG:
            problems.append(f"{tag}: RMSE {o['rmse']} vs metrics.json "
                            f"{o['metrics_json_rmse']}")
        if abs(o["mae_recomputed"] - r["ee_orientation_mae_deg"]) > TOL_MAE_DEG:
            problems.append(f"{tag}: recomputed MAE {o['mae_recomputed']} vs "
                            f"published {r['ee_orientation_mae_deg']}")
        if o["rmse"] < o["mae_recomputed"] - 1e-12:
            problems.append(f"{tag}: RMSE < MAE on identical samples")
        if not (0.0 <= o["theta_max"] <= 180.0):
            problems.append(f"{tag}: theta outside [0,180] deg")
        r["ee_orientation_rmse_deg"] = o["rmse"]
        diag.append({"condition": r["condition"], "baseline": r["baseline"],
                     "seed": r["seed"], **o,
                     "mae_published": r["ee_orientation_mae_deg"]})
    if problems:
        for p in problems[:20]:
            print("  FAIL " + p, file=sys.stderr)
        raise ValidationError(f"{len(problems)} episode check(s) failed")

    print("[3] summarising and checking the unchanged metrics against version 1")
    summary = summarize(rows, cfg)
    v1sum = {(e["condition"], e["baseline"], e["metric"]): e for e in
             common.read_csv(os.path.join(OUT_DIR, "E1_50seed_mean.csv"))}
    for e in summary:
        key = e["metric"] if e["metric"] != "ee_orientation_mae_deg" else "ee_orientation_mae_deg"
        if e["metric"] in UNCHANGED + ["ee_orientation_mae_deg"]:
            ref = v1sum[(e["condition"], e["baseline"], key)]
            for a, b in (("mean", "mean"), ("sd_ddof1", "sd"),
                         ("ci95_lower", "ci95_lower"), ("ci95_upper", "ci95_upper")):
                if float(ref[b]) != e[a]:
                    problems.append(f"{e['condition']}/{e['baseline']}/{key}: {a} "
                                    f"{e[a]!r} != version-1 {ref[b]!r}")
    if problems:
        for p in problems[:20]:
            print("  FAIL " + p, file=sys.stderr)
        raise ValidationError("unchanged metrics differ from version 1")

    print("[4] writing version-2 outputs (version-1 files untouched)")
    per_seed_cols = ["condition", "condition_title", "baseline", "method", "seed",
                     "base_vxy_rmse_mps", "base_yaw_rmse_radps", "ee_position_rmse_m",
                     "ee_orientation_rmse_deg", "ee_orientation_mae_deg",
                     "survived", "termination_reason", "valid_evaluation_time_s",
                     "n_eval_samples", "path"]
    for r in rows:
        r["condition_title"] = v1.CONDITION_TITLES[r["condition"]]
        r["method"] = v1.METHOD_LABEL[r["baseline"]]
    per_seed = [{k: r[k] for k in per_seed_cols} for r in
                sorted(rows, key=lambda r: (r["condition"], r["baseline"], r["seed"]))]
    written = [
        common.write_csv(os.path.join(OUT_DIR, f"E1_per_seed_600rows_{TAG}.csv"), per_seed),
        common.write_csv(os.path.join(OUT_DIR, f"E1_50seed_mean_sd_{TAG}.csv"), summary),
        common.write_csv(os.path.join(OUT_DIR, f"E1_ori_rmse_episode_checks_{TAG}.csv"),
                         sorted(diag, key=lambda d: (d["condition"], d["baseline"], d["seed"]))),
        common.write_text(os.path.join(OUT_DIR, f"E1_tracking_rmse_{TAG}.tex"),
                          build_tex(summary)),
    ]
    for p in written:
        print("  " + os.path.relpath(p, common.BENCH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
