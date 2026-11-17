#!/usr/bin/env python3
"""E3 (bidirectional 7 x 7) pre-run validation and protocol freeze.

    python3 -m benchmark.e3.prerun_validation                 # V1-V7 + V9 on all schedules
    python3 -m benchmark.e3.prerun_validation --freeze        # + write protocol/ snapshot and
                                                              #   the frozen command schedules
    python3 -m benchmark.e3.prerun_validation --episodes DIR  # V8 on smoke-test episodes

  V1 pose source         table sha256 == the training file every controller used;
                         feasible-row set == frozen file (whole table)
  V2 representation      9-D keypoint <-> pose round trip; vectorised rotation == frames
  V3 base levels         the seven B values; |cmd| <= B; B = 0 exactly zero; common
                         waveform x B; segments U(2, 4) s
  V4 EE levels           the seven update frequencies; switches exactly every T from
                         tick T (1 Hz -> every 50 ticks); Fixed never switches; every
                         command is a table row; pose 0 first; no interpolation
  V5 determinism         two independent builds of every schedule are bit-identical
  V6 pairing             EE identical across B; base identical across T; EE levels
                         consume prefixes of one sequence; builder has no controller input
  V7 jump comparability  position/orientation jump distributions per EE level
  V8 episodes            (--episodes) logged commands == frozen schedule; identical initial
                         state and commands across C1/C3/C5; RMSE recomputation == runner
  V9 frozen artefacts    controller checkpoints, deploy configs and the shared benchmark
                         config unchanged vs git HEAD
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy import stats as sps

from benchmark.round3 import common
from benchmark.core import frames
from benchmark.e3 import extract as X
from benchmark.scenarios import experiments as E

E3 = os.path.join(common.BENCH, "results", "E3")
OUT = os.path.join(E3, "protocol")
SCHED = os.path.join(OUT, "E3_command_schedules.npz")
TRAIN_FILE = os.path.join(os.path.dirname(common.REPO_ROOT), "IsaacLab_huanyu", "scripts", "tools",
                          "keypoints_plb_evaluation_filtered.npy")
WANT_B = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
WANT_HZ = [0.0, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1.0]
FROZEN = [  # V9: controller checkpoints + deploy configs + shared benchmark config (repo-relative)
    "exported_policy/b2wz1_wbc_plb_uan/policy_unified_1.onnx",
    "exported_policy/b2wz1_vbc_plb_uan/policy_3.onnx",
    "exported_policy/b2wz1_roboduet_plb_uan/arm_policy.onnx",
    "exported_policy/b2wz1_roboduet_plb_uan/loco_policy.onnx",
    "exported_policy/b2wz1_wbc_plb_uan/model_4000_inference.pt",
    "simulate_python/configs/b2wz1_wbc_plb_uan.yaml",
    "simulate_python/configs/b2wz1_vbc_plb_uan.yaml",
    "simulate_python/configs/b2wz1_roboduet_plb_uan.yaml",
    "simulate_python/benchmark/configs/baselines.yaml",
    "utilities/data/keypoints_plb_evaluation_filtered.npy",
]


def _sha_file(p):
    with open(p, "rb") as h:
        return hashlib.sha256(h.read()).hexdigest()


def _git(*args):
    return subprocess.run(["git", "-C", common.REPO_ROOT, *args], capture_output=True, text=True)


def setup():
    cfg, g = E.load_config("E3")
    s0, n = cfg["official_seeds"]["start"], cfg["official_seeds"]["count"]
    return cfg, g, list(range(s0, s0 + n))


def check_schedules():
    cfg, g, seeds = setup()
    table = E.keypoint_table(g)
    F, res = [], {}
    blev = [float(b) for b in cfg["base"]["levels"]]
    hz = E.e3_update_hz(cfg)
    NB, NT = len(blev), len(hz)
    # V1
    sha_train = _sha_file(TRAIN_FILE) if os.path.isfile(TRAIN_FILE) else None
    feas = E.e3_feasible_rows(cfg)
    res["V1"] = {"table_sha256": table.sha256, "training_file_sha256": sha_train,
                 "feasible_rows": int(feas.size), "feasible_is_whole_table": bool(
                     np.array_equal(feas, np.arange(table.n_rows)))}
    if sha_train != table.sha256:
        F.append("V1 table differs from the training file")
    if not res["V1"]["feasible_is_whole_table"]:
        F.append("V1 feasible set is not the whole table")
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
    # V3/V4 level sets
    res["V3_base_levels"], res["V4_ee_update_hz"] = blev, hz
    if not np.allclose(blev, WANT_B, atol=1e-12):
        F.append(f"V3 base levels {blev}")
    if not np.allclose(hz, WANT_HZ, atol=1e-12):
        F.append(f"V4 EE update frequencies {hz}")
    lab = [str(x) for x in cfg["ee"]["update_hz_labels"]]
    if lab != ["Fixed", "1/6", "1/5", "1/4", "1/3", "1/2", "1"]:
        F.append(f"V4 labels {lab}")
    n_ticks = E.n_eval_ticks(cfg)
    base = np.zeros((NB, len(seeds), n_ticks, 3), np.float32)
    ee = np.zeros((NT, len(seeds), n_ticks, 9), np.float32)
    seqs = np.zeros((len(seeds), E.E3_POSE_SEQUENCE_LENGTH), np.int64)
    jumps = {lab[t]: {"pos": [], "ori": [], "n": []} for t in range(1, NT)}
    switch_steps = {}
    for si, seed in enumerate(seeds):
        sch = {c: E.build_e3(cfg, c, seed) for c, *_ in E.e3_conditions(cfg)}
        seqs[si] = E.sc.e3_pose_sequence(seed, feas, E.E3_POSE_SEQUENCE_LENGTH)
        for cid, B, T in E.e3_conditions(cfg):
            bi, ti = int(cid[1]), int(cid[3])
            s = sch[cid]
            # V5 determinism
            s2 = E.build_e3(cfg, cid, seed)
            a1, a2 = s.arrays(), s2.arrays()
            if any(not np.array_equal(a1[k], a2[k]) for k in a1):
                F.append(f"V5 {cid} seed {seed}: rebuild differs")
            # V4
            c = s.ee.commands
            rows = s.ee.row_indices
            tab = table.table[rows]
            if not all(np.any(np.all(tab == r, axis=1)) for r in np.unique(c, axis=0)):
                F.append(f"V4 {cid} seed {seed}: command not a table row")
            if np.any(c[0] != table.table[rows[0]]) or rows[0] != seqs[si, 0]:
                F.append(f"V4 {cid} seed {seed}: pose 0 not first command")
            ch = np.nonzero(np.any(np.diff(c, axis=0) != 0, axis=1))[0] + 1
            if T is None:
                if len(ch) or len(rows) != 1:
                    F.append(f"V4 {cid} seed {seed}: Fixed level changes")
            else:
                steps = int(round(T / 0.02))
                want = np.arange(steps, n_ticks, steps)
                switch_steps[lab[ti]] = steps
                if not np.array_equal(s.ee.switch_ticks, want) or not set(ch.tolist()) <= set(want.tolist()):
                    F.append(f"V4 {cid} seed {seed}: switch ticks")
                if bi == 0:
                    Rr = X.rot_from_kp(tab.astype(np.float64))
                    jumps[lab[ti]]["pos"] += np.linalg.norm(np.diff(tab[:, :3], axis=0), axis=1).tolist()
                    jumps[lab[ti]]["ori"] += X.geodesic(Rr[1:], Rr[:-1]).tolist()
                    jumps[lab[ti]]["n"].append(len(want))
            # V3
            b = s.base.commands
            if B == 0.0 and np.any(b != 0):
                F.append(f"V3 {cid} seed {seed}: B=0 not zero")
            if np.abs(b).max() > B + 1e-6:
                F.append(f"V3 {cid} seed {seed}: bound")
            dur = s.base.segment_durations_s[:-1]
            if len(dur) and (dur.min() < 2 - 1e-9 or dur.max() > 4 + 1e-9):
                F.append(f"V3 {cid} seed {seed}: segment duration")
            if ti == 0:
                base[bi, si] = b
            if bi == 0:
                ee[ti, si] = c
        # V6
        for cid, B, T in E.e3_conditions(cfg):
            bi, ti = int(cid[1]), int(cid[3])
            if not np.array_equal(sch[cid].ee.commands, ee[ti, si]):
                F.append(f"V6 seed {seed}: EE differs across base levels ({cid})")
            if not np.array_equal(sch[cid].base.commands, base[bi, si]):
                F.append(f"V6 seed {seed}: base differs across EE levels ({cid})")
            r = sch[cid].ee.row_indices
            if not np.array_equal(r, seqs[si, :len(r)]):
                F.append(f"V6 seed {seed}: {cid} rows are not a prefix of the seed sequence")
        w = sch[f"B{NB - 1}T0"].base.waveform
        for bi in range(1, NB):
            if not np.allclose(base[bi, si], w * np.float32(blev[bi]), atol=1e-6):
                F.append(f"V3 seed {seed}: B{bi} not the common waveform x B")
    if "baseline" in inspect.signature(E.build_e3).parameters:
        F.append("V6 builder takes a controller argument")
    res["V4_switch_interval_ticks"] = switch_steps
    if switch_steps.get("1") != 50:
        F.append(f"V4 1 Hz switch interval {switch_steps.get('1')} ticks != 50")
    # V7
    allp = np.concatenate([jumps[k]["pos"] for k in jumps])
    allo = np.concatenate([jumps[k]["ori"] for k in jumps])
    v7 = {}
    for k, v in jumps.items():
        p, o = np.asarray(v["pos"]), np.asarray(v["ori"])
        v7[k] = {"n_jumps": int(p.size), "switches_per_episode": float(np.mean(v["n"])),
                 "pos_mean_m": float(p.mean()), "ori_mean_rad": float(o.mean()),
                 "ks_pos_vs_all_p": float(sps.ks_2samp(p, allp).pvalue),
                 "ks_ori_vs_all_p": float(sps.ks_2samp(o, allo).pvalue)}
    res["V7_jump_distributions_by_level"] = v7
    res["V9_frozen_artefacts"] = check_frozen(F)
    res["n_seeds"], res["n_conditions"] = len(seeds), len(E.e3_conditions(cfg))
    res["failures"], res["passed"] = F, not F
    arrays = {"base_commands_by_B": base, "ee_commands_by_T": ee, "pose_sequences": seqs,
              "seeds": np.asarray(seeds), "base_levels": np.asarray(blev), "ee_update_hz": np.asarray(hz),
              "ee_update_intervals_s": np.asarray([np.nan if t is None else t
                                                   for t in cfg["ee"]["update_intervals_s"]])}
    return res, arrays


def check_frozen(F):
    out = {}
    for rel in FROZEN:
        p = os.path.join(common.REPO_ROOT, rel)
        out[rel] = _sha_file(p)
    diff = _git("diff", "--name-only", "HEAD", "--", *FROZEN).stdout.split()
    untracked = _git("ls-files", "--others", "--", *FROZEN).stdout.split()
    out["_git_diff_vs_HEAD"] = diff
    if diff or untracked:
        F.append(f"V9 frozen artefacts modified: {diff + untracked}")
    return out


def schedule_sha(path):
    return _sha_file(path)


def freeze(res, arrays):
    """Write the protocol snapshot and the frozen command schedules (before the full run)."""
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(SCHED, **arrays)
    sha = schedule_sha(SCHED)
    with open(SCHED + ".sha256", "w") as h:
        h.write(f"{sha}  {os.path.basename(SCHED)}\n")
    shutil.copy(os.path.join(common.BENCH, "configs", "E3.yaml"), os.path.join(OUT, "E3_config.yaml"))
    shutil.copy(os.path.join(common.BENCH, "configs", "baselines.yaml"), os.path.join(OUT, "baselines_snapshot.yaml"))
    with open(os.path.join(OUT, "git_state.txt"), "w") as h:
        h.write(_git("log", "-1", "--format=%H %s").stdout)
        h.write("branch " + _git("rev-parse", "--abbrev-ref", "HEAD").stdout)
        h.write("--- git status --porcelain (benchmark code changes are the uncommitted E3 replacement)\n")
        h.write(_git("status", "--porcelain").stdout)
    files = sorted(p for p in glob.glob(os.path.join(common.BENCH, "**", "*"), recursive=True)
                   if os.path.isfile(p) and "/results/" not in p and "__pycache__" not in p)
    with open(os.path.join(OUT, "code_sha256_at_launch.txt"), "w") as h:
        for p in files:
            h.write(f"{_sha_file(p)}  {os.path.relpath(p, common.SIM_ROOT)}\n")
    with open(os.path.join(OUT, "frozen_artefacts_sha256.txt"), "w") as h:
        for rel in FROZEN:
            h.write(f"{res['V9_frozen_artefacts'][rel]}  {rel}\n")
    with open(os.path.join(OUT, "launch_time.txt"), "w") as h:
        h.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    res["frozen_schedule_sha256"] = sha
    return sha


def v8(ep_dir):
    """Smoke-test episodes: logged schedule == frozen schedule; pairing; metric recomputation."""
    cfg, _, seeds = setup()
    S = np.load(SCHED)
    sidx = {int(s): i for i, s in enumerate(S["seeds"])}
    F, worst, n = [], {"rmse": 0.0, "theta_deg": 0.0, "issued": 0.0, "vs_frozen": 0.0}, 0
    by = {}
    for d in sorted(glob.glob(os.path.join(ep_dir, "E3", "*", "*", "seed_*"))):
        c, cid, sd = d.split(os.sep)[-3:]
        seed = int(sd.split("_")[1])
        bi, ti = int(cid[1]), int(cid[3])
        o = X.extract(d, base_bound=float(cfg["base"]["levels"][bi]))
        if o["problems"]:
            F.append(f"{d}: {o['problems']}")
            continue
        n += 1
        by[(c, cid, seed)] = o
        worst["rmse"] = max(worst["rmse"], max(abs(o["rec"][k] - o["json"][k]) for k in o["rec"]))
        worst["theta_deg"] = max(worst["theta_deg"], o["theta_vs_logged_deg"])
        worst["issued"] = max(worst["issued"], o["issued_vs_scheduled"])
        npz = np.load(os.path.join(d, "scenario_commands.npz"))
        dv = max(np.abs(npz["base_commands"] - S["base_commands_by_B"][bi, sidx[seed]]).max(),
                 np.abs(npz["ee_commands_plb"] - S["ee_commands_by_T"][ti, sidx[seed]]).max())
        worst["vs_frozen"] = max(worst["vs_frozen"], float(dv))
        if not (o["settle_base_zero"] and o["settle_ee_is_first_target"] and o["base_bound_ok"]):
            F.append(f"{d}: settle/bound check")
    pair = 0
    for (c, cid, seed), o in by.items():
        if c != "C1":
            continue
        for r in ("C3", "C5"):
            q = by.get((r, cid, seed))
            if q is None:
                F.append(f"missing {r}/{cid}/seed {seed}")
                continue
            pair += 1
            for k in ("init_hash", "base_hash", "ee_hash"):
                if o[k] != q[k]:
                    F.append(f"pairing {k} {cid} seed {seed} C1 vs {r}")
    if worst["rmse"] > 1e-6 or worst["theta_deg"] > 1e-3 or worst["issued"] > 1e-6 or worst["vs_frozen"] > 0:
        F.append(f"recomputation/schedule mismatch {worst}")
    return {"episodes": n, "pairs_checked": pair, **worst, "failures": F, "passed": not F and n > 0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "prerun_validation.json")
    if a.episodes:
        res = json.load(open(path))
        res["V8_smoke_episodes"] = v8(a.episodes)
        res["V9_frozen_artefacts"] = check_frozen(res.setdefault("failures", []))
        res["passed"] = not res["failures"] and res["V8_smoke_episodes"]["passed"]
    else:
        res, arrays = check_schedules()
        if a.freeze:
            if not res["passed"]:
                print("not freezing: schedule validation failed")
            else:
                freeze(res, arrays)
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("V7_jump_distributions_by_level",
                                                                 "V9_frozen_artefacts")}, indent=1))
    for k, v in res.get("V7_jump_distributions_by_level", {}).items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    return 0 if res.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
