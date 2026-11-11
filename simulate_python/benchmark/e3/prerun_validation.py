#!/usr/bin/env python3
"""E3 pre-run validation (before the official 7,500 episodes).

  V1 pose source         table sha256 == the training file every controller used
  V2 representation      9-D keypoint <-> pose round trip; vectorised rotation == frames
  V3 sample-and-hold     every command is a table row; changes only at switch ticks
  V4 intervals           switches exactly every T from tick T (pose 0 held at t = 3 s)
  V5 base schedule       |cmd| <= B; B = 0 exactly zero; common waveform x B; U(2,4) s
  V6 pairing/independence  EE identical across B; base identical across T; interval
                         levels consume prefixes of one sequence; builder has no
                         controller input
  V7 jump comparability  position/orientation jump distributions per interval level
  V8 metric definitions  run separately on trial episodes (--episodes DIR)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
from scipy import stats as sps

from benchmark.round3 import common
from benchmark.core import frames
from benchmark.e3 import extract as X
from benchmark.scenarios import experiments as E

OUT = os.path.join(common.BENCH, "results", "E3", "protocol")
TRAIN_FILE = "/home/huanyuguo/Workspace_huanyu/IsaacLab_huanyu/scripts/tools/keypoints_plb_evaluation_filtered.npy"


def schedules():
    cfg, g = E.load_config("E3")
    seeds = range(cfg["official_seeds"]["start"], cfg["official_seeds"]["start"] + cfg["official_seeds"]["count"])
    return cfg, g, list(seeds)


def main_schedules():
    cfg, g, seeds = schedules()
    table = E.keypoint_table(g)
    F, res = [], {}
    # V1
    sha_train = hashlib.sha256(open(TRAIN_FILE, "rb").read()).hexdigest() if os.path.isfile(TRAIN_FILE) else None
    res["V1_table_sha256"] = table.sha256
    res["V1_training_file_sha256"] = sha_train
    if sha_train != table.sha256:
        F.append("V1 table differs from the training file")
    # V2
    rt = 0.0
    Rv = X.rot_from_kp(table.table.astype(np.float64))
    for i in range(0, table.n_rows, 7):
        p, R = frames.keypoints_to_pose(table.table[i])
        rt = max(rt, float(np.abs(frames.pose_to_keypoints(p, R) - table.table[i]).max()))
        rt = max(rt, float(np.abs(frames.rotation_from_keypoints(table.table[i]) - Rv[i]).max()))
    res["V2_max_roundtrip_err"] = rt
    if rt > 1e-5:
        F.append(f"V2 representation round trip {rt}")
    jumps = {t: {"pos": [], "ori": []} for t in cfg["ee"]["level_labels"][1:]}
    n_targets = {t: [] for t in cfg["ee"]["level_labels"][1:]}
    for seed in seeds:
        sch = {c: E.build_e3(cfg, c, seed) for c, *_ in E.e3_conditions(cfg)}
        for cid, B, T in E.e3_conditions(cfg):
            s = sch[cid]
            ee = s.ee.commands
            # V3
            rows = s.ee.row_indices
            if not np.all(np.isin(ee.view([("", ee.dtype)] * 9), table.table[rows].view([("", ee.dtype)] * 9))):
                F.append(f"V3 {cid} seed {seed}: command not in the table rows used")
            ch = np.nonzero(np.any(np.diff(ee, axis=0) != 0, axis=1))[0] + 1
            if T is None:
                if len(ch) or len(rows) != 1:
                    F.append(f"V4 {cid} seed {seed}: Fixed level changes")
            else:
                steps = int(round(T / 0.02))
                want = np.arange(steps, len(ee), steps)
                if not set(ch.tolist()) <= set(want.tolist()) or not np.array_equal(s.ee.switch_ticks, want):
                    F.append(f"V4 {cid} seed {seed}: switch ticks")
                if np.any(ee[0] != table.table[rows[0]]):
                    F.append(f"V4 {cid} seed {seed}: pose 0 not first command")
                if B == 0.0:
                    lab = s.description["ee_level"]
                    Rr = X.rot_from_kp(table.table[rows].astype(np.float64))
                    jumps[lab]["pos"] += np.linalg.norm(np.diff(table.table[rows][:, :3], axis=0), axis=1).tolist()
                    jumps[lab]["ori"] += X.geodesic(Rr[1:], Rr[:-1]).tolist()
                    n_targets[lab].append(len(want))
            # V5
            base = s.base.commands
            if B == 0.0 and np.any(base != 0):
                F.append(f"V5 {cid} seed {seed}: B=0 not zero")
            if np.abs(base).max() > B + 1e-6:
                F.append(f"V5 {cid} seed {seed}: bound")
            dur = s.base.segment_durations_s[:-1]
            if len(dur) and (dur.min() < 2 - 1e-9 or dur.max() > 4 + 1e-9):
                F.append(f"V5 {cid} seed {seed}: segment duration")
        # V6
        for ti in range(5):
            ref = sch[f"B0T{ti}"].ee.commands
            for bi in range(1, 5):
                if not np.array_equal(sch[f"B{bi}T{ti}"].ee.commands, ref):
                    F.append(f"V6 seed {seed}: EE differs across base levels")
        for bi in range(5):
            ref = sch[f"B{bi}T0"].base.commands
            for ti in range(1, 5):
                if not np.array_equal(sch[f"B{bi}T{ti}"].base.commands, ref):
                    F.append(f"V6 seed {seed}: base differs across EE levels")
        w = sch["B4T0"].base.waveform
        for bi, B in enumerate(cfg["base"]["levels"][1:], 1):
            if not np.allclose(sch[f"B{bi}T0"].base.commands, w * np.float32(B), atol=1e-6):
                F.append(f"V6 seed {seed}: B{bi} not the common waveform x B")
        longest = sch["B0T4"].ee.row_indices
        for ti in range(5):
            r = sch[f"B0T{ti}"].ee.row_indices
            if not np.array_equal(r, longest[:len(r)]):
                F.append(f"V6 seed {seed}: T{ti} is not a prefix of the common sequence")
    import inspect
    if "baseline" in inspect.signature(E.build_e3).parameters:
        F.append("V6 builder takes a controller argument")
    # V7
    allp = np.concatenate([jumps[k]["pos"] for k in jumps])
    allo = np.concatenate([jumps[k]["ori"] for k in jumps])
    v7 = {}
    for k in jumps:
        p, o = np.asarray(jumps[k]["pos"]), np.asarray(jumps[k]["ori"])
        v7[k] = {"n_jumps": int(p.size), "targets_per_episode": float(np.mean(n_targets[k])),
                 "pos_mean_m": float(p.mean()), "pos_median_m": float(np.median(p)),
                 "pos_p10_p90_m": [float(np.percentile(p, 10)), float(np.percentile(p, 90))],
                 "ori_mean_rad": float(o.mean()), "ori_median_rad": float(np.median(o)),
                 "ori_p10_p90_rad": [float(np.percentile(o, 10)), float(np.percentile(o, 90))],
                 "ks_pos_vs_all_p": float(sps.ks_2samp(p, allp).pvalue),
                 "ks_ori_vs_all_p": float(sps.ks_2samp(o, allo).pvalue)}
    res["V7_jump_distributions_by_level"] = v7
    res["n_seeds"], res["n_conditions"] = len(seeds), len(E.e3_conditions(cfg))
    res["failures"] = F
    res["passed"] = not F
    return res


def v8(ep_dir):
    worst = {"rmse": 0.0, "theta_deg": 0.0, "issued": 0.0}
    n = 0
    for d in sorted(glob.glob(os.path.join(ep_dir, "E3", "*", "*", "seed_*"))):
        o = X.extract(d)
        n += 1
        worst["rmse"] = max(worst["rmse"], max(abs(o["rec"][k] - o["json"][k]) for k in o["rec"]))
        worst["theta_deg"] = max(worst["theta_deg"], o["theta_vs_logged_deg"])
        worst["issued"] = max(worst["issued"], o["issued_vs_scheduled"])
        if not (o["settle_base_zero"] and o["settle_ee_is_first_target"]):
            worst["settle_fail"] = d
    return {"episodes": n, **worst,
            "passed": worst["rmse"] < 1e-6 and worst["theta_deg"] < 1e-3 and worst["issued"] < 1e-6
            and "settle_fail" not in worst}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=None)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "prerun_validation.json")
    res = json.load(open(path)) if os.path.isfile(path) else {}
    if a.episodes:
        res["V8_metric_definitions"] = v8(a.episodes)
        res["passed"] = res.get("passed", False) and res["V8_metric_definitions"]["passed"]
    else:
        res = main_schedules()
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "V7_jump_distributions_by_level"}, indent=1))
    if "V7_jump_distributions_by_level" in res:
        for k, v in res["V7_jump_distributions_by_level"].items():
            print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    return 0 if res.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
