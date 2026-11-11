"""Per-episode extraction for E3 (random full-pose). Definitions are frozen in
results/E3/E3_PROTOCOL.md section 6; everything is recomputed from the per-tick
logs (commands.csv, states.csv) and the stored schedule (scenario_commands.npz).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np

DT = 0.02
PRE, POST, ISO_PRE = 25, 75, 50           # baseline [-0.5,0) s, response [0,1.5) s, isolation [-1.0,+1.5) s
DWELL = 10                                # 0.2 s
BASELINE_ALT = {"b025": 12, "b100": 50}   # sensitivity: [-0.24,0) s and [-1.0,0) s
REC_FRAC = 0.25                           # recovery: excess <= 25 % of peak for 0.2 s
REC_FLOOR = {"base_xy": 0.005, "yaw": 0.005, "ee_pos": 0.002, "ee_ori": 0.01}
TOL_PRIMARY = (0.08, 0.35)
TOL_STRICT = (0.04, 0.20)
SIGS = ("base_xy", "yaw", "ee_pos", "ee_ori")
TRACE_LEN = PRE + POST
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


def geodesic(Ra, Rb):
    tr = np.einsum("nij,nij->n", Ra, Rb)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def tick_errors(cmd, st, ec, es):
    dvx = st["vx_act"][es] - cmd["vx_cmd"][ec]
    dvy = st["vy_act"][es] - cmd["vy_cmd"][ec]
    dwz = st["wz_act"][es] - cmd["wz_cmd"][ec]
    kc = np.stack([cmd[f"ee_cmd_kp{i}"][ec] for i in range(9)], 1)
    ka = np.stack([st[f"ee_act_kp{i}"][es] for i in range(9)], 1)
    return {"base_xy": np.sqrt(dvx ** 2 + dvy ** 2), "yaw": np.abs(dwz),
            "ee_pos": np.linalg.norm(ka[:, :3] - kc[:, :3], axis=1),
            "ee_ori": geodesic(rot_from_kp(kc), rot_from_kp(ka))}


def rmse(e, sl=slice(None)):
    return {"base_xy_rmse_mps": float(np.sqrt(np.mean(e["base_xy"][sl] ** 2))),
            "yaw_rmse_radps": float(np.sqrt(np.mean(e["yaw"][sl] ** 2))),
            "ee_pos_rmse_m": float(np.sqrt(np.mean(e["ee_pos"][sl] ** 2))),
            "ee_ori_rmse_rad": float(np.sqrt(np.mean(e["ee_ori"][sl] ** 2)))}


def event_metrics(x, k, sig):
    """Event quantities for signal x at event tick k (window fully valid)."""
    base = float(x[k - PRE:k].mean())
    w = x[k:k + POST] - base
    ipk = int(np.argmax(w))
    pk = float(w[ipk])
    out = {"baseline_abs": base, "peak_excess": pk, "peak_pos_excess": max(pk, 0.0),
           "int_excess": float(w.sum() * DT), "int_pos_excess": float(np.clip(w, 0, None).sum() * DT),
           "post_abs_mean": float(x[k:k + POST].mean()),
           "residual_abs": float(x[k + 50:k + POST].mean())}
    for tag, n in BASELINE_ALT.items():
        b2 = float(x[k - n:k].mean())
        w2 = x[k:k + POST] - b2
        out[f"peak_pos_excess_{tag}"] = max(float(w2.max()), 0.0)
        out[f"int_pos_excess_{tag}"] = float(np.clip(w2, 0, None).sum() * DT)
    if pk >= REC_FLOOR[sig]:
        rec, cens = POST * DT, 1
        thr = REC_FRAC * pk
        for j in range(ipk, POST - DWELL + 1):
            if np.all(w[j:j + DWELL] <= thr):
                rec, cens = (j - 0) * DT, 0
                break
        out["recovery_s"], out["recovery_censored"], out["recovery_defined"] = rec, cens, 1
    else:
        out["recovery_s"], out["recovery_censored"], out["recovery_defined"] = np.nan, np.nan, 0
    return out


def reach(p, o, k, end, tol):
    ok = (p[k:end] <= tol[0]) & (o[k:end] <= tol[1])
    if ok.size >= DWELL:
        run = np.convolve(ok.astype(int), np.ones(DWELL, int), "valid")
        hit = np.nonzero(run == DWELL)[0]
        if hit.size:
            return 1, float(hit[0] * DT)
    return 0, float(ok.size * DT)


def extract(d, base_bound=None, keep_event_rows=False):
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
    sb, se = npz["base_commands"], npz["ee_commands_plb"]
    out.update({
        "n_eval": n, "survived": int(met["survived"]), "termination_reason": met["termination_reason"],
        "valid_eval_s": float(met["valid_eval_duration_s"]), "metrics_json_n": int(met["n_eval_samples"]),
        "meta_seed": meta["seed"], "scenario": meta["scenario"],
        "proto_hash": _sha(json.dumps({k: v for k, v in meta.items()
                                       if k not in OUTCOME_KEYS | {"scenario", "condition", "seed",
                                                                   "initial_state"}},
                                      sort_keys=True).encode()),
        "init_hash": _sha(json.dumps([meta["initial_state"]["qpos"],
                                      meta["initial_state"].get("randomization")]).encode()),
        "base_hash": _sha(sb.tobytes()), "ee_hash": _sha(se.tobytes()),
    })
    if n == 0:
        return out
    cb = np.stack([cmd["vx_cmd"][ec], cmd["vy_cmd"][ec], cmd["wz_cmd"][ec]], 1)
    ce = np.stack([cmd[f"ee_cmd_kp{i}"][ec] for i in range(9)], 1)
    out["issued_vs_scheduled"] = float(max(np.abs(cb - sb[:n]).max(), np.abs(ce - se[:n]).max()))
    settle = ~ec
    kc_s = np.stack([cmd[f"ee_cmd_kp{i}"][settle] for i in range(9)], 1)
    out["settle_base_zero"] = bool(np.all(cb[:0] == 0) and np.all(cmd["vx_cmd"][settle] == 0)
                                   and np.all(cmd["vy_cmd"][settle] == 0) and np.all(cmd["wz_cmd"][settle] == 0))
    out["settle_ee_is_first_target"] = bool(np.all(np.abs(kc_s - se[0][None]) <= 1e-6))
    if base_bound is not None:
        out["base_bound_ok"] = bool(np.all(np.abs(cb) <= base_bound + 1e-6))
    e = tick_errors(cmd, st, ec, es)
    out["rec"] = rmse(e)
    out["json"] = {"base_xy_rmse_mps": float(met["base_vxy_rmse"]), "yaw_rmse_radps": float(met["base_wz_rmse"]),
                   "ee_pos_rmse_m": float(met["ee_pos_rmse"]),
                   "ee_ori_rmse_rad": float(np.radians(met["ee_ori_rmse_deg"]))}
    out["theta_vs_logged_deg"] = float(np.max(np.abs(np.degrees(e["ee_ori"]) - st["ee_ori_err_deg"][es])))
    out["ee_err_at_t3"] = {"pos": float(e["ee_pos"][0]), "ori": float(e["ee_ori"][0])}

    # ---------- command changes ----------
    ee_sw = [int(k) for k in npz["ee_switch_ticks"]]
    b_sw = [int(k) for k in npz["base_switch_ticks"] if int(k) > 0] if np.any(sb != 0) else []
    b_ch = ([0] if np.any(sb[0] != 0) else []) + b_sw     # t = 3 s step from the zero settle command
    ee_ch = list(ee_sw)                                    # no EE change at t = 3 s (pose 0 held)
    R = rot_from_kp(se.astype(np.float64))

    def isolated(k, others):
        return not any(k - ISO_PRE <= o < k + POST for o in others)

    ev_rows, agg = [], {}
    groups = {"ee_iso": [], "ee_ovl": [], "base_iso": [], "base_ovl": []}
    tr = {"ee_iso": [], "base_iso": []}
    trunc = {"ee": 0, "base": 0}
    for kind, sw, others in (("ee", ee_sw, b_ch), ("base", b_sw, ee_ch)):
        for k in sw:
            if k >= n:
                continue
            if k - PRE < 0 or k + POST > n:
                trunc[kind] += 1
                continue
            iso = isolated(k, others)
            row = {"kind": kind, "tick": k, "isolated": int(iso)}
            if kind == "ee":
                row["jump_pos_m"] = float(np.linalg.norm(se[k, :3] - se[k - 1, :3]))
                row["jump_ori_rad"] = float(geodesic(R[k:k + 1], R[k - 1:k])[0])
            else:
                row["jump_base_cmd"] = float(np.linalg.norm(sb[k] - sb[k - 1]))
            for s in SIGS:
                for q, v in event_metrics(e[s], k, s).items():
                    row[f"{s}_{q}"] = v
            groups[f"{kind}_{'iso' if iso else 'ovl'}"].append(row)
            if iso:
                tr[f"{kind}_iso"].append(np.stack([e[s][k - PRE:k + POST] for s in SIGS]))
            ev_rows.append(row)
    for g, rows in groups.items():
        agg[f"n_{g}"] = len(rows)
        if not rows:
            continue
        for key in rows[0]:
            if key in ("kind", "tick", "isolated"):
                continue
            vals = np.array([r[key] for r in rows], dtype=float)
            agg[f"{g}_{key}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    agg["n_ee_events_truncated"] = trunc["ee"]
    agg["n_base_events_truncated"] = trunc["base"]
    out["events"] = agg
    out["traces"] = {k: (np.mean(v, axis=0).astype(np.float32) if v else None) for k, v in tr.items()}
    if keep_event_rows:
        out["event_rows"] = ev_rows

    # ---------- target reaching (every switch inside the window) ----------
    steps = int(ee_sw[1] - ee_sw[0]) if len(ee_sw) > 1 else (ee_sw[0] if ee_sw else 0)
    T = []
    for k in ee_sw:
        if k >= n:
            continue
        end = min(k + steps, n)
        rp, tp = reach(e["ee_pos"], e["ee_ori"], k, end, TOL_PRIMARY)
        rs, ts = reach(e["ee_pos"], e["ee_ori"], k, end, TOL_STRICT)
        T.append((rp, tp, rs, ts, float(np.linalg.norm(se[k, :3] - se[k - 1, :3])),
                  float(geodesic(R[k:k + 1], R[k - 1:k])[0])))
    if T:
        T = np.asarray(T)
        out["reach"] = {
            "n_targets": int(len(T)), "reach_primary": int(T[:, 0].sum()), "reach_strict": int(T[:, 2].sum()),
            "reach_rate_primary": float(T[:, 0].mean()), "reach_rate_strict": float(T[:, 2].mean()),
            "rmst_primary_s": float(T[:, 1].mean()), "rmst_strict_s": float(T[:, 3].mean()),
            "mean_jump_pos_m": float(T[:, 4].mean()), "mean_jump_ori_rad": float(T[:, 5].mean()),
            "hold_s": steps * DT}
    else:
        out["reach"] = None
    # steady holding of pose 0 (Fixed level and first hold): first evaluated tick
    rp0, _ = reach(e["ee_pos"], e["ee_ori"], 0, min(n, steps if steps else n), TOL_PRIMARY)
    out["pose0_reached_primary_in_first_hold"] = int(rp0)
    return out
