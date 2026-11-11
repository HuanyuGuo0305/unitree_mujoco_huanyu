"""Per-episode extraction for E3 v3 (pre-registered metrics, section 6).

Everything is recomputed from the per-tick logs (commands.csv, states.csv) and
the stored schedule (scenario_commands.npz); metrics.json is used only as a
cross-check and for the termination record.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np

DT = 0.02
PRE, POST, ISO_PRE = 25, 75, 50          # -0.5 s, +1.5 s, -1.0 s (ticks)
DWELL = 10                               # 0.2 s
REC_TOL = {"base_xy": 0.05, "yaw": 0.05, "ee_pos": 0.01, "ee_ori": 0.05}
TOL_PRIMARY = (0.08, 0.35)
TOL_STRICT = (0.04, 0.20)
TRACE_LEN = PRE + POST                   # 100 ticks, t = -0.5 ... +1.48 s
OUTCOME_KEYS = {"wall_time_s", "git_commit", "policy_ticks", "physics_steps", "uan_calls",
                "survived", "termination_reason", "termination_time_s", "valid_eval_duration_s"}


def _read(path):
    with open(path) as h:
        r = csv.reader(h)
        hdr = next(r)
        a = np.asarray([list(map(float, x)) for x in r], dtype=np.float64)
    return {k: a[:, j] for j, k in enumerate(hdr)}


def _sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


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


def tick_errors(cmd, st, ec, es):
    dvx = st["vx_act"][es] - cmd["vx_cmd"][ec]
    dvy = st["vy_act"][es] - cmd["vy_cmd"][ec]
    dwz = st["wz_act"][es] - cmd["wz_cmd"][ec]
    kc = np.stack([cmd[f"ee_cmd_kp{i}"][ec] for i in range(9)], 1)
    ka = np.stack([st[f"ee_act_kp{i}"][es] for i in range(9)], 1)
    tr = np.einsum("nij,nij->n", rot_from_kp(kc), rot_from_kp(ka))
    return {"base_xy": np.sqrt(dvx ** 2 + dvy ** 2), "yaw": np.abs(dwz),
            "ee_pos": np.linalg.norm(ka[:, :3] - kc[:, :3], axis=1),
            "ee_ori": np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))}


def rmse(e, sl=slice(None)):
    return {"base_xy_rmse_mps": float(np.sqrt(np.mean(e["base_xy"][sl] ** 2))),
            "yaw_rmse_radps": float(np.sqrt(np.mean(e["yaw"][sl] ** 2))),
            "ee_pos_rmse_m": float(np.sqrt(np.mean(e["ee_pos"][sl] ** 2))),
            "ee_ori_rmse_rad": float(np.sqrt(np.mean(e["ee_ori"][sl] ** 2)))}


def event_response(x, k, tol):
    """Peak excess, integrated excess and recovery time of signal x at event k."""
    base = float(x[k - PRE:k].mean())
    w = x[k:k + POST] - base
    ipk = int(np.argmax(w))
    within = w <= tol
    rec = POST * DT  # censored
    for j in range(ipk, POST - DWELL + 1):
        if within[j:j + DWELL].all():
            rec = j * DT
            break
    return {"baseline": base, "peak_excess": float(w[ipk]),
            "int_excess": float(w.sum() * DT), "recovery_s": float(rec),
            "recovery_censored": int(rec >= POST * DT)}


def reach(p, o, k, end, tol):
    """First qualifying dwell inside tolerance in [k, end); returns (reached, time)."""
    ok = (p[k:end] <= tol[0]) & (o[k:end] <= tol[1])
    n = len(ok)
    if n >= DWELL:
        run = np.convolve(ok.astype(int), np.ones(DWELL, int), "valid")
        hit = np.nonzero(run == DWELL)[0]
        if hit.size:
            return 1, float(hit[0] * DT)
    return 0, float(n * DT)


def extract(d, base_bound=None):
    """Return a dict of per-episode quantities, traces and integrity hashes."""
    out = {"path": d, "problems": []}
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
    sched_b, sched_e = npz["base_commands"], npz["ee_commands_plb"]
    out.update({
        "n_eval": n, "n_planned": int(len(sched_b)), "survived": int(met["survived"]),
        "termination_reason": met["termination_reason"],
        "valid_eval_s": float(met["valid_eval_duration_s"]),
        "metrics_json_n": int(met["n_eval_samples"]),
        "meta_seed": meta["seed"], "scenario": meta["scenario"],
        "proto_hash": _sha(json.dumps({k: v for k, v in meta.items()
                                      if k not in OUTCOME_KEYS | {"scenario", "condition", "seed",
                                                                  "initial_state"}},
                                     sort_keys=True).encode()),
        "init_hash": _sha(json.dumps([meta["initial_state"]["qpos"],
                                      meta["initial_state"].get("randomization")]).encode()),
        "base_hash": _sha(sched_b.tobytes()), "ee_hash": _sha(sched_e.tobytes()),
        "states_hash": _sha(open(os.path.join(d, "states.csv"), "rb").read()),
    })
    if n == 0:
        return out
    # issued == scheduled
    cb = np.stack([cmd["vx_cmd"][ec], cmd["vy_cmd"][ec], cmd["wz_cmd"][ec]], 1)
    ce = np.stack([cmd[f"ee_cmd_kp{i}"][ec] for i in range(9)], 1)
    out["issued_vs_scheduled"] = float(max(np.abs(cb - sched_b[:n]).max(),
                                           np.abs(ce - sched_e[:n]).max()))
    settle = ~ec
    out["settle_base_zero"] = bool(np.all(cmd["vx_cmd"][settle] == 0) and
                                   np.all(cmd["vy_cmd"][settle] == 0) and
                                   np.all(cmd["wz_cmd"][settle] == 0))
    kc_settle = np.stack([cmd[f"ee_cmd_kp{i}"][settle] for i in range(9)], 1)
    out["settle_ee_constant"] = bool(np.all(kc_settle == kc_settle[0]))
    if base_bound is not None:
        out["base_bound_ok"] = bool(np.all(np.abs(cb) <= base_bound + 1e-6))
    e = tick_errors(cmd, st, ec, es)
    out["rec"] = rmse(e)
    # post-hoc sensitivity: window 5-30 s (drops the first reach from the initial pose)
    out["rec_from5s"] = rmse(e, slice(100, None)) if n > 100 else None
    out["json"] = {"base_xy_rmse_mps": float(met["base_vxy_rmse"]),
                   "yaw_rmse_radps": float(met["base_wz_rmse"]),
                   "ee_pos_rmse_m": float(met["ee_pos_rmse"]),
                   "ee_ori_rmse_rad": float(np.radians(met["ee_ori_rmse_deg"]))}
    out["theta_vs_logged_deg"] = float(np.max(np.abs(np.degrees(e["ee_ori"]) - st["ee_ori_err_deg"][es])))

    # ---------- events ----------
    ee_sw = [int(k) for k in npz["ee_switch_ticks"]]
    b_sw = [int(k) for k in npz["base_switch_ticks"] if int(k) > 0] \
        if np.any(sched_b != 0) else []
    # the t = 3 s step (tick 0) is a command change of both subsystems
    ee_changes = [0] + ee_sw
    b_changes = ([0] if np.any(sched_b[0] != 0) else []) + b_sw

    def isolated(k, others):
        return not any(k - ISO_PRE <= o < k + POST for o in others)

    def ok_window(k):
        return k - PRE >= 0 and k + POST <= n

    ev = {"ee_iso": [], "ee_ovl": [], "base_iso": [], "base_ovl": []}
    tr = {"ee_iso": [], "base_iso": []}
    n_trunc = {"ee": 0, "base": 0}
    for k in ee_sw:
        if k >= n:
            continue
        if not ok_window(k):
            n_trunc["ee"] += 1
            continue
        r = {s: event_response(e[s], k, REC_TOL[s]) for s in ("base_xy", "yaw")}
        key = "ee_iso" if isolated(k, b_changes) else "ee_ovl"
        ev[key].append(r)
        if key == "ee_iso":
            tr["ee_iso"].append(np.stack([e[s][k - PRE:k + POST] for s in
                                          ("base_xy", "yaw", "ee_pos", "ee_ori")]))
    for k in b_sw:
        if k >= n:
            continue
        if not ok_window(k):
            n_trunc["base"] += 1
            continue
        r = {s: event_response(e[s], k, REC_TOL[s]) for s in ("ee_pos", "ee_ori")}
        key = "base_iso" if isolated(k, ee_changes) else "base_ovl"
        ev[key].append(r)
        if key == "base_iso":
            tr["base_iso"].append(np.stack([e[s][k - PRE:k + POST] for s in
                                            ("base_xy", "yaw", "ee_pos", "ee_ori")]))
    agg = {}
    for key, lst in ev.items():
        agg[f"n_{key}"] = len(lst)
        sigs = ("base_xy", "yaw") if key.startswith("ee") else ("ee_pos", "ee_ori")
        for s in sigs:
            for q in ("peak_excess", "int_excess", "recovery_s", "recovery_censored"):
                agg[f"{key}_{s}_{q}"] = (float(np.mean([r[s][q] for r in lst]))
                                         if lst else float("nan"))
    agg["n_ee_events_truncated"] = n_trunc["ee"]
    agg["n_base_events_truncated"] = n_trunc["base"]
    out["events"] = agg
    out["traces"] = {k: (np.mean(v, axis=0).astype(np.float32) if v else None)
                     for k, v in tr.items()}

    # ---------- target reaching (EE switches after the first) ----------
    steps = int(ee_sw[1] - ee_sw[0]) if len(ee_sw) > 1 else (ee_sw[0] if ee_sw else 0)
    targets = []
    for k in ee_sw:
        if k >= n:
            continue  # never issued (terminated before)
        end = min(k + steps, n, len(sched_b))
        rp, tp = reach(e["ee_pos"], e["ee_ori"], k, end, TOL_PRIMARY)
        rs, ts = reach(e["ee_pos"], e["ee_ori"], k, end, TOL_STRICT)
        full = (end - k) == steps
        targets.append((rp, tp, rs, ts, full))
    if targets:
        T = np.asarray(targets, dtype=float)
        out["reach"] = {
            "n_targets": int(len(T)), "n_targets_full_hold": int(T[:, 4].sum()),
            "reach_primary": int(T[:, 0].sum()), "reach_strict": int(T[:, 2].sum()),
            "reach_rate_primary": float(T[:, 0].mean()), "reach_rate_strict": float(T[:, 2].mean()),
            "rmst_primary_s": float(T[:, 1].mean()), "rmst_strict_s": float(T[:, 3].mean()),
            "median_time_reached_primary_s": float(np.median(T[T[:, 0] == 1, 1])) if T[:, 0].any() else float("nan"),
            "hold_s": steps * DT,
        }
    else:
        out["reach"] = None
    return out
