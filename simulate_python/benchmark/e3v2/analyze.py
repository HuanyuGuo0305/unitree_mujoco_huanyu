#!/usr/bin/env python3
"""E3 v2 analysis: validation, per-trial export, statistics, degradation.

Reads results/E3_v2/raw/E3 (45,000 episodes); writes results/E3_v2/{data,stats,validation}.
No simulation. Run:  python3 -m benchmark.e3v2.analyze

Metrics (identical definitions to the updated E1), over the evaluated ticks
(t in [3, 20) s, truncated at termination), recomputed from the per-tick logs:
  base_xy_rmse_mps  sqrt(mean((vx-vx*)^2 + (vy-vy*)^2))  body frame
  yaw_rmse_radps    sqrt(mean((wz-wz*)^2))               body frame
  ee_pos_rmse_m     sqrt(mean(||p - p*||^2))             PLB frame
  ee_ori_rmse_rad   sqrt(mean(theta^2)), theta = SO(3) geodesic angle  PLB frame

Inference
  * every controller comparison is paired by seed; pooled ("all trajectories")
    values average the three families WITHIN a seed first, so the seed stays
    the resampling unit (the three families share a seed's base command and
    initial state);
  * 95 % CIs: percentile bootstrap of the mean, 10,000 resamples, seed 20261106;
  * p-values: two-sided paired t-test on the per-seed differences (the
    analytic counterpart of the bootstrapped mean), Holm-adjusted within each
    declared family of tests (named in every output); BH q-values also given;
  * a cell is "C1 better" / "C1 worse" only if the Holm-adjusted p < 0.05;
    otherwise "no significant difference" -- never "equivalent".
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
from scipy import stats as sps

from benchmark.round3 import common  # noqa: F401 (sys.path)
from benchmark.core import frames, scenario as sc, stats  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

BENCH = common.BENCH
ROOT = os.path.join(BENCH, "results", "E3_v2")
RAW = os.path.join(ROOT, "raw", "E3")
CFG, GCFG = E.load_config("E3")
assert E.e3_is_v2(CFG)
BASELINES = list(CFG["baselines"])
FAMILIES = list(CFG["ee"]["families"])
BLEV = [float(x) for x in CFG["base"]["levels"]]
VLEV = [float(x) for x in CFG["ee"]["speed_levels_mps"]]
N_SEEDS = int(CFG["official_seeds"]["count"])
DT = 0.02
N_EVAL = E.n_eval_ticks(CFG, DT)
SENS_T0 = 4.0
METRICS = [("base_xy_rmse_mps", "Base XY RMSE", "m/s"),
           ("yaw_rmse_radps", "Yaw-rate RMSE", "rad/s"),
           ("ee_pos_rmse_m", "EE position RMSE", "m"),
           ("ee_ori_rmse_rad", "EE orientation RMSE", "rad")]
MK = [m[0] for m in METRICS]
BASE_M, EE_M = MK[:2], MK[2:]
RIVALS = ["C3", "C5"]
SCFG = GCFG["statistics"]
NB, BSEED = int(SCFG["bootstrap_resamples"]), int(SCFG["bootstrap_seed"])
OUTCOME_KEYS = {"initial_state", "scenario", "seed", "termination_time_s", "survived",
                "termination_reason", "valid_eval_duration_s", "uan_calls",
                "policy_ticks", "physics_steps", "condition"}


class ValidationError(SystemExit):
    pass


def cond_id(bi, vi, fam):
    return f"B{bi}E{vi}_{fam}"


def _read(path):
    with open(path, newline="") as h:
        rows = list(csv.reader(h))
    a = np.asarray(rows[1:], dtype=np.float64)
    return {k: a[:, i] for i, k in enumerate(rows[0])}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def rot_from_kp(kp):
    """Vectorised frames.rotation_from_keypoints (same operations, same order)."""
    kp = kp.reshape(-1, 3, 3)
    x = kp[:, 1] - kp[:, 0]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    z = kp[:, 2] - kp[:, 0]
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    return np.stack([x, y, z], axis=2)


def _rmse_set(st, cmd, es, ec):
    dvx = st["vx_act"][es] - cmd["vx_cmd"][ec]
    dvy = st["vy_act"][es] - cmd["vy_cmd"][ec]
    dwz = st["wz_act"][es] - cmd["wz_cmd"][ec]
    kc = np.stack([cmd[f"ee_cmd_kp{i}"][ec] for i in range(9)], 1)
    ka = np.stack([st[f"ee_act_kp{i}"][es] for i in range(9)], 1)
    tr = np.einsum("nij,nij->n", rot_from_kp(kc), rot_from_kp(ka))
    theta = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    perr = np.linalg.norm(ka[:, :3] - kc[:, :3], axis=1)
    return {"base_xy_rmse_mps": float(np.sqrt(np.mean(dvx ** 2 + dvy ** 2))),
            "yaw_rmse_radps": float(np.sqrt(np.mean(dwz ** 2))),
            "ee_pos_rmse_m": float(np.sqrt(np.mean(perr ** 2))),
            "ee_ori_rmse_rad": float(np.sqrt(np.mean(theta ** 2)))}, theta


def load_episode(job):
    b, bi, vi, fam, seed = job
    cid = cond_id(bi, vi, fam)
    d = os.path.join(RAW, b, cid, f"seed_{seed:03d}")
    out = {"baseline": b, "bi": bi, "vi": vi, "family": fam, "seed": seed, "condition": cid,
           "path": os.path.relpath(d, BENCH), "problems": []}
    need = ("commands.csv", "states.csv", "metrics.json", "metadata.json", "scenario_commands.npz")
    miss = [f for f in need if not os.path.isfile(os.path.join(d, f))]
    if miss:
        out["problems"].append(f"missing {miss}")
        return out
    met = json.load(open(os.path.join(d, "metrics.json")))
    meta = json.load(open(os.path.join(d, "metadata.json")))
    cmd, st = _read(os.path.join(d, "commands.csv")), _read(os.path.join(d, "states.csv"))
    npz = np.load(os.path.join(d, "scenario_commands.npz"))
    ec, es = cmd["evaluated"] == 1, st["evaluated"] == 1
    if not np.array_equal(cmd["tick"][ec], st["tick"][es]):
        out["problems"].append("evaluated ticks differ between commands and states")
        return out
    n = int(es.sum())
    out.update({"n_eval": n, "survived": int(met["survived"]),
                "termination_reason": met["termination_reason"],
                "valid_eval_s": float(met["valid_eval_duration_s"]),
                "metrics_json_n": int(met["n_eval_samples"]),
                "meta_seed": meta["seed"], "scen_seed": meta["scenario"]["seed"],
                "scenario_desc": {k: v for k, v in meta["scenario"].items() if k != "seed"},
                "meta_hash": _sha(json.dumps({k: v for k, v in meta.items() if k not in OUTCOME_KEYS},
                                             sort_keys=True).encode()),
                "init_hash": _sha(json.dumps([meta["initial_state"]["qpos"],
                                              meta["initial_state"].get("randomization")]).encode()),
                "scen_hash": _sha(npz["base_commands"].tobytes() + npz["ee_commands_plb"].tobytes()),
                "base_hash": _sha(npz["base_commands"].tobytes()),
                "ee_hash": _sha(npz["ee_commands_plb"].tobytes()),
                "states_hash": _sha(open(os.path.join(d, "states.csv"), "rb").read())})
    if n == 0:
        return out
    t_ev = st["t"][es]
    rec, theta = _rmse_set(st, cmd, es, ec)
    out["rec"] = rec
    out["json"] = {"base_xy_rmse_mps": float(met["base_vxy_rmse"]),
                   "yaw_rmse_radps": float(met["base_wz_rmse"]),
                   "ee_pos_rmse_m": float(met["ee_pos_rmse"]),
                   "ee_ori_rmse_rad": float(np.radians(met["ee_ori_rmse_deg"]))}
    out["theta_vs_logged"] = float(np.max(np.abs(np.degrees(theta) - st["ee_ori_err_deg"][es])))
    out["t_first"] = float(t_ev[0])
    # sensitivity window: drop the residual start-up transient (t < 4 s)
    es2 = es & (st["t"] >= SENS_T0 - 1e-9)
    ec2 = ec & (cmd["t"] >= SENS_T0 - 1e-9)
    out["rec_from4s"] = _rmse_set(st, cmd, es2, ec2)[0] if es2.sum() > 0 else None
    # commands actually issued
    B = BLEV[bi]
    cmd_b = np.stack([cmd["vx_cmd"][ec], cmd["vy_cmd"][ec], cmd["wz_cmd"][ec]], 1)
    out["base_cmd_absmax"] = np.abs(cmd_b).max(0).tolist()
    out["base_cmd_bound_ok"] = bool(np.all(np.abs(cmd_b) <= B + 1e-6))
    settle = ~ec
    kc_all = np.stack([cmd[f"ee_cmd_kp{i}"] for i in range(9)], 1)
    out["settle_hold_dev"] = float(np.abs(kc_all[settle] - npz["ee_commands_plb"][0][None, :]).max()) \
        if settle.any() else float("nan")
    out["settle_base_zero"] = bool(np.all(cmd["vx_cmd"][settle] == 0) and np.all(cmd["vy_cmd"][settle] == 0)
                                   and np.all(cmd["wz_cmd"][settle] == 0))
    issued = np.concatenate([cmd_b, kc_all[ec]], 1)
    sched = np.concatenate([npz["base_commands"][:n], npz["ee_commands_plb"][:n]], 1)
    out["issued_vs_schedule"] = float(np.abs(issued - sched).max())
    return out


# --------------------------------------------------------------------------
def check_geometry(problems):
    """Trajectory geometry, speed, acceleration and orientation, per condition."""
    table = E.keypoint_table(GCFG)
    R_ref = frames.keypoints_to_pose(table.row(int(CFG["ee"]["orientation_table_row"])))[1]
    c, a, ap = np.asarray(CFG["ee"]["center"]), float(CFG["ee"]["amplitude"]), float(CFG["ee"]["a_peak_mps2"])
    P0 = c - a * np.array([0, 1, 0])
    rows = []
    for vi, v in enumerate(VLEV):
        for fam in FAMILIES:
            npz = np.load(os.path.join(RAW, "C1", cond_id(0, vi, fam), "seed_000", "scenario_commands.npz"))
            kp = npz["ee_commands_plb"].astype(np.float64)
            P = kp[:, 0:3]
            R = rot_from_kp(kp)
            ori_dev = np.degrees(np.arccos(np.clip((np.einsum("ij,nij->n", R_ref, R) - 1) / 2, -1, 1))).max()
            if fam == "line":
                off = np.hypot(P[:, 0] - c[0], P[:, 2] - c[2])
                off = np.maximum(off, np.maximum(np.abs(P[:, 1] - c[1]) - a, 0))
            else:
                r = np.hypot(P[:, 1] - c[1], P[:, 2] - c[2])
                off = np.maximum(np.abs(P[:, 0] - c[0]), np.abs(r - a))
                if fam == "semicircle":
                    off = np.maximum(off, np.maximum(c[2] - P[:, 2] - 1e-6, 0))
            spd = np.linalg.norm(np.diff(P, axis=0), axis=1) / DT
            s_true = sc.e3v2_arclength(np.arange(len(P)) * DT, fam, a, v, ap)
            dspd = np.abs(np.diff(s_true, 2)) / DT ** 2 if v > 0 else np.zeros(1)
            if fam == "circle" and v > 0:
                L = sc.e3v2_path_length(fam, a)
                ds = np.diff(s_true); ds = np.where(ds < -L / 2, ds + L, ds)
                dspd = np.abs(np.diff(ds)) / DT ** 2
            row = {"family": fam, "speed_level_mps": v, "start_point_err_m": float(np.linalg.norm(P[0] - P0)),
                   "max_off_path_m": float(off.max()), "max_speed_mps": float(spd.max()),
                   "max_tangential_accel_mps2": float(dspd.max()), "max_orientation_dev_deg": float(ori_dev),
                   "mean_speed_mps": float(spd.mean()),
                   "y_range_m": [float(P[:, 1].min()), float(P[:, 1].max())],
                   "z_range_m": [float(P[:, 2].min()), float(P[:, 2].max())]}
            ok = (row["start_point_err_m"] < 1e-6 and row["max_off_path_m"] < 1e-6
                  and row["max_speed_mps"] <= v * (1 + 1e-4) + 1e-6
                  and (v == 0 or row["max_speed_mps"] >= 0.999 * v)
                  and row["max_tangential_accel_mps2"] <= ap * 1.001 + 1e-9
                  and row["max_orientation_dev_deg"] < 1e-3)
            row["ok"] = ok
            if not ok:
                problems.append(f"geometry {fam} v={v}: {row}")
            rows.append(row)
    return rows


def validate(eps):
    P = []
    seen = {}
    for e in eps:
        key = (e["baseline"], e["bi"], e["vi"], e["family"], e["seed"])
        tag = f"{e['baseline']}/{e['condition']}/seed {e['seed']}"
        if key in seen:
            P.append(f"duplicate {tag}")
        seen[key] = e
        if e["problems"]:
            P += [f"{tag}: {p}" for p in e["problems"]]
            continue
        if e["meta_seed"] != e["seed"] or e["scen_seed"] != e["seed"]:
            P.append(f"{tag}: seed mismatch in metadata")
        sd = e["scenario_desc"]
        if (abs(sd["base_bound"] - BLEV[e["bi"]]) > 1e-12 or abs(sd["ee_speed_mps"] - VLEV[e["vi"]]) > 1e-12
                or sd["ee_family"] != e["family"] or sd.get("protocol_version") != 2):
            P.append(f"{tag}: scenario description does not match the condition")
        if e["n_eval"] != e["metrics_json_n"]:
            P.append(f"{tag}: {e['n_eval']} evaluated ticks vs metrics.json {e['metrics_json_n']}")
        if e["n_eval"] == 0:
            P.append(f"{tag}: zero evaluated ticks")
            continue
        # survived => the full window; terminated => at most the full window
        # (a fall in the final physics steps after the last policy tick keeps all
        # 850 evaluated ticks and is still a termination -- counted as such).
        if (e["survived"] and e["n_eval"] != N_EVAL) or (not e["survived"] and e["n_eval"] > N_EVAL):
            P.append(f"{tag}: survived={e['survived']} but {e['n_eval']} ticks")
        if not e["survived"] and e["termination_reason"] in ("", "completed"):
            P.append(f"{tag}: terminated without a reason")
        if abs(e["t_first"] - 3.0) > DT + 1e-9:
            P.append(f"{tag}: evaluation starts at {e['t_first']}")
        for k in MK:
            a, b = e["rec"][k], e["json"][k]
            if not np.isfinite(b) or abs(a - b) > 1e-6 * max(1.0, abs(b)) + (2e-6 if k == "ee_ori_rmse_rad" else 0):
                P.append(f"{tag}: {k} recomputed {a!r} vs metrics.json {b!r}")
        if e["theta_vs_logged"] > 1e-3:
            P.append(f"{tag}: geodesic differs from logged by {e['theta_vs_logged']} deg")
        if not e["base_cmd_bound_ok"]:
            P.append(f"{tag}: base command exceeds bound {BLEV[e['bi']]}")
        if e["issued_vs_schedule"] > 1e-6:
            P.append(f"{tag}: issued commands differ from schedule by {e['issued_vs_schedule']}")
        if not (e["settle_hold_dev"] < 1e-6 and e["settle_base_zero"]):
            P.append(f"{tag}: settle hold is not (zero base, trajectory start pose)")
    want = set(range(N_SEEDS))
    for b in BASELINES:
        for bi in range(len(BLEV)):
            for vi in range(len(VLEV)):
                for fam in FAMILIES:
                    got = {s for s in want if (b, bi, vi, fam, s) in seen}
                    if got != want:
                        P.append(f"{b}/{cond_id(bi, vi, fam)}: missing seeds {sorted(want - got)[:8]}")
    ok = {k: e for k, e in seen.items() if not e["problems"]}
    # pairing across controllers; command sequences across conditions
    for bi in range(len(BLEV)):
        for vi in range(len(VLEV)):
            for fam in FAMILIES:
                for s in range(N_SEEDS):
                    es_ = [ok.get((b, bi, vi, fam, s)) for b in BASELINES]
                    if any(x is None for x in es_):
                        continue
                    if len({x["scen_hash"] for x in es_}) != 1:
                        P.append(f"{cond_id(bi, vi, fam)}/seed {s}: commands differ across controllers")
                    if len({x["init_hash"] for x in es_}) != 1:
                        P.append(f"{cond_id(bi, vi, fam)}/seed {s}: initial state differs across controllers")
    for b in BASELINES:
        for s in range(N_SEEDS):
            for bi in range(len(BLEV)):
                hs = {ok[(b, bi, vi, f, s)]["base_hash"] for vi in range(len(VLEV)) for f in FAMILIES
                      if (b, bi, vi, f, s) in ok}
                if len(hs) > 1:
                    P.append(f"{b}/B{bi}/seed {s}: base commands differ across EE conditions")
            inits = {ok[(b, bi, vi, f, s)]["init_hash"] for bi in range(len(BLEV)) for vi in range(len(VLEV))
                     for f in FAMILIES if (b, bi, vi, f, s) in ok}
            if len(inits) > 1:
                P.append(f"{b}/seed {s}: initial state differs across conditions")
        for vi in range(len(VLEV)):
            for fam in FAMILIES:
                hs = {ok[(b, bi, vi, fam, s)]["ee_hash"] for bi in range(len(BLEV)) for s in range(N_SEEDS)
                      if (b, bi, vi, fam, s) in ok}
                if len(hs) > 1:
                    P.append(f"{b}/E{vi}_{fam}: EE commands differ across seeds or base levels")
        # speed 0: the three families are the same fixed-pose scenario -> identical episodes
        for bi in range(len(BLEV)):
            for s in range(N_SEEDS):
                hs = {ok[(b, bi, 0, f, s)]["states_hash"] for f in FAMILIES if (b, bi, 0, f, s) in ok}
                if len(hs) > 1:
                    P.append(f"{b}/B{bi}E0/seed {s}: speed-0 families not identical (determinism)")
    for b in BASELINES:
        hs = {e["meta_hash"] for k, e in ok.items() if k[0] == b}
        if len(hs) != 1:
            P.append(f"{b}: {len(hs)} distinct protocol metadata sets (expected 1)")
    return seen, P


# --------------------------------------------------------------------------
def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    q = np.empty(m)
    prev = 1.0
    for rank, i in zip(range(m, 0, -1), order):
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q


def summarise(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    lo, hi = stats.bootstrap_ci(v, NB, BSEED)
    return {"n": int(v.size), "mean": float(v.mean()), "sd_ddof1": float(v.std(ddof=1)),
            "ci95_lower": lo, "ci95_upper": hi, "median": float(np.median(v))}


def paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    lo, hi = stats.bootstrap_ci(d, NB, BSEED)
    if np.all(d == d[0]):
        p = 1.0 if d[0] == 0 else 0.0
    else:
        p = float(sps.ttest_1samp(d, 0.0).pvalue)
    return {"n_pairs": int(d.size), "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "ci95_lower": lo, "ci95_upper": hi, "p_ttest": p}


def finish_family(rows, family_name, pos_label, neg_label):
    """Holm/BH within a family; classify. Positive mean_diff => pos_label."""
    if not rows:
        return rows
    ph, q = holm([r["p_ttest"] for r in rows]), bh([r["p_ttest"] for r in rows])
    for r, a, b in zip(rows, ph, q):
        r["test_family"] = family_name
        r["family_size"] = len(rows)
        r["p_holm"] = float(a)
        r["q_bh"] = float(b)
        r["significant_holm"] = int(a < 0.05)
        r["ci_excludes_0"] = int(r["ci95_lower"] > 0 or r["ci95_upper"] < 0)
        r["verdict"] = (pos_label if r["mean_diff"] > 0 else neg_label) if a < 0.05 \
            else "no significant difference"
    return rows


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    for sub in ("data", "stats", "validation"):
        os.makedirs(os.path.join(ROOT, sub), exist_ok=True)

    jobs = [(b, bi, vi, f, s) for b in BASELINES for bi in range(len(BLEV)) for vi in range(len(VLEV))
            for f in FAMILIES for s in range(N_SEEDS)]
    print(f"[1] loading {len(jobs)} episodes; recomputing metrics from per-tick logs")
    with ProcessPoolExecutor(args.workers) as ex:
        eps = list(ex.map(load_episode, jobs, chunksize=50))
    # vectorised rotation == benchmark scalar rotation, on real logged keypoints
    st = _read(os.path.join(RAW, "C3", cond_id(4, 4, "circle"), "seed_000", "states.csv"))
    kp = np.stack([st[f"ee_act_kp{i}"] for i in range(9)], 1)[:200]
    assert np.allclose(rot_from_kp(kp), np.stack([frames.rotation_from_keypoints(k) for k in kp]), atol=1e-12)

    print("[2] validation")
    seen, P = validate(eps)
    geo = check_geometry(P)
    common.write_csv(os.path.join(ROOT, "validation", "trajectory_geometry_check.csv"),
                     [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in g.items()} for g in geo])
    if P:
        for p in P[:30]:
            print("  FAIL " + p, file=sys.stderr)
        common.write_text(os.path.join(ROOT, "validation", "FAILURES.txt"), "\n".join(P) + "\n")
        raise ValidationError(f"{len(P)} validation failure(s)")
    stale = os.path.join(ROOT, "validation", "FAILURES.txt")
    if os.path.exists(stale):
        os.remove(stale)   # a clean pass supersedes any earlier failure list
    print(f"  {len(seen)} episodes; all checks passed")

    # ---------------- per-trial export ----------------
    rows = []
    for (b, bi, vi, f, s), e in sorted(seen.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3], kv[0][0], kv[0][4])):
        r = {"controller": b, "trajectory": f, "base_level": BLEV[bi], "ee_speed_mps": VLEV[vi],
             "condition": e["condition"], "seed": s}
        r.update(e["rec"])
        for k in MK:
            r[f"{k}_from4s"] = e["rec_from4s"][k] if e["rec_from4s"] else float("nan")
        r.update({"completed": e["survived"], "termination_reason": e["termination_reason"],
                  "effective_eval_s": e["valid_eval_s"], "n_eval_ticks": e["n_eval"],
                  "cmd_ee_mean_speed_mps": e["scenario_desc"]["ee_mean_speed_eval_mps"],
                  "cmd_base_absmax_vx": e["base_cmd_absmax"][0], "cmd_base_absmax_vy": e["base_cmd_absmax"][1],
                  "cmd_base_absmax_wz": e["base_cmd_absmax"][2], "path": e["path"]})
        rows.append(r)
    common.write_csv(os.path.join(ROOT, "data", "E3v2_per_trial_45000.csv"), rows)

    # arrays: V[b][metric] shape (nB, nV, nF, nS)
    shape = (len(BLEV), len(VLEV), len(FAMILIES), N_SEEDS)
    V = {b: {k: np.full(shape, np.nan) for k in MK + [m + "_from4s" for m in MK] + ["completed", "eval_s"]}
         for b in BASELINES}
    for (b, bi, vi, f, s), e in seen.items():
        fi = FAMILIES.index(f)
        for k in MK:
            V[b][k][bi, vi, fi, s] = e["rec"][k]
            V[b][k + "_from4s"][bi, vi, fi, s] = e["rec_from4s"][k] if e["rec_from4s"] else np.nan
        V[b]["completed"][bi, vi, fi, s] = e["survived"]
        V[b]["eval_s"][bi, vi, fi, s] = e["valid_eval_s"]
    np.savez_compressed(os.path.join(ROOT, "data", "E3v2_arrays.npz"),
                        **{f"{b}__{k}": V[b][k] for b in BASELINES for k in V[b]},
                        base_levels=BLEV, ee_speeds=VLEV, families=FAMILIES)

    def series(b, k, bi, vi, fam):
        """Per-seed values; fam=None -> pooled (mean of the three families within each seed)."""
        a = V[b][k][bi, vi]
        return a.mean(axis=0) if fam is None else a[FAMILIES.index(fam)]

    TRAJ = FAMILIES + [None]
    tname = lambda f: "pooled" if f is None else f

    print("[3] absolute errors, completion")
    summ = []
    for b in BASELINES:
        for f in TRAJ:
            for bi, B in enumerate(BLEV):
                for vi, v in enumerate(VLEV):
                    comp = V[b]["completed"][bi, vi] if f is None else V[b]["completed"][bi, vi, FAMILIES.index(f)]
                    ev = V[b]["eval_s"][bi, vi] if f is None else V[b]["eval_s"][bi, vi, FAMILIES.index(f)]
                    k_, n_ = int(comp.sum()), int(comp.size)
                    wl, wh = stats.wilson_interval(k_, n_)
                    for k in MK:
                        for win in ("", "_from4s"):
                            x = series(b, k + win, bi, vi, f)
                            r = {"controller": b, "trajectory": tname(f), "base_level": B, "ee_speed_mps": v,
                                 "metric": k, "window": "3-20 s" if not win else "4-20 s (sensitivity)"}
                            r.update(summarise(x))
                            r.update({"n_episodes": n_, "n_completed": k_, "completion_rate": k_ / n_,
                                      "completion_wilson95_lower": wl, "completion_wilson95_upper": wh,
                                      "mean_effective_eval_s": float(ev.mean()),
                                      "n_excluded_nan": int(np.sum(~np.isfinite(x)))})
                            summ.append(r)
    common.write_csv(os.path.join(ROOT, "stats", "E3v2_summary.csv"), summ)

    print("[4] paired controller comparisons")
    for win, tagw in (("", ""), ("_from4s", "_sensitivity_from4s")):
        for scope in ("pooled", "by_trajectory"):
            fams = [None] if scope == "pooled" else FAMILIES
            out = []
            for f in fams:
                for bi, B in enumerate(BLEV):
                    for vi, v in enumerate(VLEV):
                        for k in MK:
                            ref = series("C1", k + win, bi, vi, f)
                            for rv in RIVALS:
                                r = {"trajectory": tname(f), "base_level": B, "ee_speed_mps": v, "metric": k,
                                     "comparison": f"{rv}_minus_C1",
                                     "sign_convention": "positive => C1 has the lower error"}
                                d = series(rv, k + win, bi, vi, f) - ref
                                r.update(paired(d))
                                fin = np.isfinite(d)
                                r["frac_seeds_C1_lower"] = float(np.mean(d[fin] > 0))
                                r["rival_mean"] = float(np.nanmean(series(rv, k + win, bi, vi, f)))
                                r["C1_mean"] = float(np.nanmean(ref))
                                out.append(r)
            finish_family(out, f"controller comparisons, {scope}{tagw}", "C1 better", "C1 worse")
            common.write_csv(os.path.join(ROOT, "stats", f"E3v2_paired_{scope}{tagw}.csv"), out)

    print("[5] cross-task degradation")
    for scope in ("pooled", "by_trajectory"):
        fams = [None] if scope == "pooled" else FAMILIES
        deg, ddeg = [], []
        for f in fams:
            for direction, metrics_, fixed_levels, sweep in (
                    ("base_to_ee", EE_M, range(len(VLEV)), range(1, len(BLEV))),
                    ("ee_to_base", BASE_M, range(len(BLEV)), range(1, len(VLEV)))):
                for k in metrics_:
                    for fx in fixed_levels:
                        for sw in sweep:
                            bi, vi = (sw, fx) if direction == "base_to_ee" else (fx, sw)
                            bi0, vi0 = (0, fx) if direction == "base_to_ee" else (fx, 0)
                            D = {}
                            for b in BASELINES:
                                D[b] = series(b, k, bi, vi, f) - series(b, k, bi0, vi0, f)
                                both = (V[b]["completed"][bi, vi] * V[b]["completed"][bi0, vi0])
                                both = both.min(axis=0) if f is None else both[FAMILIES.index(f)]
                                r = {"direction": direction, "trajectory": tname(f), "controller": b,
                                     "base_level": BLEV[bi], "ee_speed_mps": VLEV[vi],
                                     "reference": "same EE speed, stationary base" if direction == "base_to_ee"
                                     else "same base level, fixed EE target",
                                     "metric": k, "abs_error_mean": float(np.nanmean(series(b, k, bi, vi, f))),
                                     "reference_error_mean": float(np.nanmean(series(b, k, bi0, vi0, f)))}
                                r.update(paired(D[b]))
                                dc = D[b][both > 0]
                                r["mean_diff_completed_pairs_only"] = float(dc.mean()) if dc.size else float("nan")
                                r["n_completed_pairs"] = int(dc.size)
                                deg.append(r)
                            for rv in RIVALS:
                                r = {"direction": direction, "trajectory": tname(f), "base_level": BLEV[bi],
                                     "ee_speed_mps": VLEV[vi], "metric": k,
                                     "comparison": f"degradation_{rv}_minus_C1",
                                     "sign_convention": "positive => C1 degrades less"}
                                r.update(paired(D[rv] - D["C1"]))
                                ddeg.append(r)
        finish_family(deg, f"degradation vs reference, {scope}", "error increases", "error decreases")
        finish_family(ddeg, f"degradation C1 vs rival, {scope}", "C1 degrades less", "C1 degrades more")
        common.write_csv(os.path.join(ROOT, "stats", f"E3v2_degradation_{scope}.csv"), deg)
        common.write_csv(os.path.join(ROOT, "stats", f"E3v2_degradation_C1_vs_rivals_{scope}.csv"), ddeg)

    print("[6] curvature of the intensity response (quadratic fit per seed)")
    x = np.linspace(0, 1, 5)
    X = np.stack([np.ones(5), x, x ** 2], 1)
    pinv = np.linalg.pinv(X)
    for scope in ("pooled", "by_trajectory"):
        fams = [None] if scope == "pooled" else FAMILIES
        out = []
        for f in fams:
            for b in BASELINES:
                for direction, metrics_ in (("base_to_ee", EE_M), ("ee_to_base", BASE_M)):
                    for k in metrics_:
                        fixed = range(len(VLEV)) if direction == "base_to_ee" else range(len(BLEV))
                        for fx in fixed:
                            Y = np.stack([series(b, k, i, fx, f) if direction == "base_to_ee"
                                          else series(b, k, fx, i, f) for i in range(5)], 1)  # (seeds, 5)
                            coef = Y @ pinv.T
                            means = np.nanmean(Y, 0)
                            r = {"direction": direction, "trajectory": tname(f), "controller": b, "metric": k,
                                 "fixed_factor": "ee_speed_mps" if direction == "base_to_ee" else "base_level",
                                 "fixed_value": VLEV[fx] if direction == "base_to_ee" else BLEV[fx],
                                 "means_by_level": json.dumps([float(m) for m in means]),
                                 "linear_coef_mean": float(np.nanmean(coef[:, 1]))}
                            r.update(paired(coef[:, 2]))
                            r["quadratic_coef_mean"] = r.pop("mean_diff")
                            r["mean_diff"] = r["quadratic_coef_mean"]
                            inc = np.diff(Y, axis=1)
                            r["n_steps_sig_increase_raw"] = int(sum(
                                (paired(inc[:, j])["ci95_lower"] > 0) for j in range(4)))
                            out.append(r)
        finish_family(out, f"curvature, {scope}", "accelerating (convex)", "saturating (concave)")
        for r in out:
            r.pop("mean_diff", None)
        common.write_csv(os.path.join(ROOT, "stats", f"E3v2_curvature_{scope}.csv"), out)
    print("done")


if __name__ == "__main__":
    main()
