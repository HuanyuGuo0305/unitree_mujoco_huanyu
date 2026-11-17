"""Per-episode extraction for E3 (bidirectional 7 x 7). Definitions: docs/E3_PROTOCOL.md
section 5. Everything is recomputed from the per-tick logs (commands.csv, states.csv)
and checked against the runner's metrics.json and the stored schedule
(scenario_commands.npz). The tracking-error definitions (tick_errors, rmse) are
unchanged from the 5 x 5 E3 and E1.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np

DT = 0.02
SIGS = ("base_xy", "yaw", "ee_pos", "ee_ori")
RMSE_KEYS = ("base_xy_rmse_mps", "yaw_rmse_radps", "ee_pos_rmse_m", "ee_ori_rmse_rad")
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


def extract(d, base_bound=None):
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
        "n_ee_switches_scheduled": int(len(npz["ee_switch_ticks"])),
    })
    if n == 0:
        return out
    cb = np.stack([cmd["vx_cmd"][ec], cmd["vy_cmd"][ec], cmd["wz_cmd"][ec]], 1)
    ce = np.stack([cmd[f"ee_cmd_kp{i}"][ec] for i in range(9)], 1)
    out["issued_vs_scheduled"] = float(max(np.abs(cb - sb[:n]).max(), np.abs(ce - se[:n]).max()))
    settle = ~ec
    kc_s = np.stack([cmd[f"ee_cmd_kp{i}"][settle] for i in range(9)], 1)
    out["settle_base_zero"] = bool(np.all(cmd["vx_cmd"][settle] == 0) and np.all(cmd["vy_cmd"][settle] == 0)
                                   and np.all(cmd["wz_cmd"][settle] == 0))
    out["settle_ee_is_first_target"] = bool(np.all(np.abs(kc_s - se[0][None]) <= 1e-6))
    if base_bound is not None:
        out["base_bound_ok"] = bool(np.all(np.abs(cb) <= base_bound + 1e-6))
    e = tick_errors(cmd, st, ec, es)
    out["rec"] = rmse(e)
    out["json"] = {"base_xy_rmse_mps": float(met["base_vxy_rmse"]), "yaw_rmse_radps": float(met["base_wz_rmse"]),
                   "ee_pos_rmse_m": float(met["ee_pos_rmse"]),
                   "ee_ori_rmse_rad": float(np.radians(met["ee_ori_rmse_deg"]))}
    out["theta_vs_logged_deg"] = float(np.max(np.abs(np.degrees(e["ee_ori"]) - st["ee_ori_err_deg"][es])))
    out["n_ee_switches_evaluated"] = int(np.sum(npz["ee_switch_ticks"] < n))
    return out
