#!/usr/bin/env python3
"""E3 v3 pilot: static/initial pose-tracking screening for all three controllers.

Pre-registered in docs/E3_V3_PREREGISTRATION.md section 4.
  * seeds 9000-9002 (disjoint from official 0-99 and showcase 100-149)
  * base still; 3 s settle on the initial EE pose; waypoints 0..4 held 6 s each
  * per (candidate, controller, waypoint): mean over seeds of the
      initial-response error  (0.5-1.5 s after the switch) and
      steady-state error      (4.0-6.0 s after the switch)
  * rejection P1 (any termination), P2 (worst-controller steady-state >= 0.08 m
    or 0.35 rad), P3 (worst-controller initial >= 0.16 m or 0.70 rad)
  * selection S = 0.4 C_pos + 0.2 C_ori + 0.2 F + 0.2 (1 - W) per pattern
    (W = worst controller/waypoint static error normalised by the tolerances);
    no between-controller difference enters any rule.

    python3 -m benchmark.e3v3.pilot run   [--jobs 16]   # simulate the pilot set
    python3 -m benchmark.e3v3.pilot select              # rejection + selection report
    python3 -m benchmark.e3v3.pilot freeze              # write waypoints into configs/E3.yaml
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "BENCHMARK_ORT_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from benchmark.round3 import common  # noqa: E402
from benchmark.core import scenario as sc  # noqa: E402
from benchmark.core.runner import EpisodeSpec, PrecomputedScenario, run_episode  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

V3 = os.path.join(common.BENCH, "results", "E3_v3")
SCREEN = os.path.join(V3, "screening")
PILOT = os.path.join(V3, "pilot")
CONTROLLERS = ("C1", "C3", "C5")
DT = 0.02
TOL_POS, TOL_ORI = 0.08, 0.35          # primary reaching tolerance (pre-registered)
INIT_WIN, SS_WIN = (0.5, 1.5), (4.0, 6.0)


def _candidates():
    with open(os.path.join(SCREEN, "kinematic_screening_full.json")) as h:
        return {r["id"]: r for r in json.load(h)}


def pilot_scenario(cand, exp_cfg, seed):
    hold = float(exp_cfg["pilot"]["hold_s"])
    n_ticks = int(round((float(exp_cfg["pilot"]["duration_s"]) - float(exp_cfg["settle_s"])) / DT))
    base = E.e3_base_schedule(exp_cfg, 0.0, seed, n_ticks, DT)
    ee = sc.make_ee_waypoint_schedule(n_ticks, DT, cand["waypoints_kp"], cand["pattern"],
                                      hold, 0, 1)
    desc = {"experiment": "E3v3_pilot", "candidate_id": cand["id"], "pattern": cand["pattern"],
            "hold_s": hold, "sequence": ee.row_indices.tolist(), "base": "still",
            "n_eval_ticks": n_ticks, "seed": seed}
    return PrecomputedScenario(base, ee, desc)


def _run(payload):
    import torch

    torch.set_num_threads(1)
    cand, ctrl, seed, exp_cfg = payload
    out = os.path.join(PILOT, "raw", cand["id"], ctrl, f"seed_{seed:04d}")
    if os.path.isfile(os.path.join(out, "metrics.json")):
        return (cand["id"], ctrl, seed, "cached")
    spec = EpisodeSpec(experiment="E3v3_pilot", baseline=ctrl, condition=cand["id"], seed=seed,
                       duration_s=float(exp_cfg["pilot"]["duration_s"]),
                       settle_s=float(exp_cfg["settle_s"]),
                       termination=exp_cfg["termination"],
                       initial_state=exp_cfg.get("initial_state", {}))
    try:
        run_episode(spec, pilot_scenario(cand, exp_cfg, seed), out)
        return (cand["id"], ctrl, seed, None)
    except Exception as exc:  # noqa: BLE001
        return (cand["id"], ctrl, seed, f"{exc}\n{traceback.format_exc()}")


def cmd_run(args):
    exp_cfg, _ = E.load_config("E3")
    cands = _candidates()
    with open(os.path.join(SCREEN, "pilot_set.json")) as h:
        ids = [i for v in json.load(h).values() for i in v]
    if args.extra:
        ids += [i.strip() for i in args.extra.split(",")]
    jobs = [(cands[i], c, s, exp_cfg) for i in ids for c in CONTROLLERS
            for s in exp_cfg["pilot"]["seeds"]]
    print(f"pilot: {len(ids)} candidates x {len(CONTROLLERS)} controllers x "
          f"{len(exp_cfg['pilot']['seeds'])} seeds = {len(jobs)} episodes")
    t0, fails = time.time(), []
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        for i, (cid, ctrl, seed, err) in enumerate(pool.imap_unordered(_run, jobs), 1):
            if err and err != "cached":
                fails.append((cid, ctrl, seed, err))
                print(f"FAIL {cid} {ctrl} {seed}: {err.splitlines()[0]}")
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f} s", flush=True)
    print(f"done, {len(fails)} runner failures")
    return 1 if fails else 0


def _load_states(path):
    with open(path) as h:
        r = csv.reader(h)
        hdr = next(r)
        a = np.asarray([list(map(float, x)) for x in r])
    return {k: a[:, j] for j, k in enumerate(hdr)}


def episode_waypoint_errors(path, hold_s):
    st = _load_states(os.path.join(path, "states.csv"))
    m = st["evaluated"] == 1
    p, o = st["ee_pos_err"][m], np.radians(st["ee_ori_err_deg"][m])
    steps = int(round(hold_s / DT))
    out = []
    for j in range(5):
        a = j * steps
        w = lambda win, x: x[a + int(round(win[0] / DT)): a + int(round(win[1] / DT))]  # noqa: E731
        ip, io, sp, so = w(INIT_WIN, p), w(INIT_WIN, o), w(SS_WIN, p), w(SS_WIN, o)
        full = len(sp) == int(round((SS_WIN[1] - SS_WIN[0]) / DT))
        out.append({"init_pos": float(ip.mean()) if len(ip) else np.nan,
                    "init_ori": float(io.mean()) if len(io) else np.nan,
                    "ss_pos": float(sp.mean()) if full else np.nan,
                    "ss_ori": float(so.mean()) if full else np.nan})
    with open(os.path.join(path, "metrics.json")) as h:
        met = json.load(h)
    return out, int(met["survived"]), met.get("termination_reason", "")


def cmd_select(args):
    exp_cfg, _ = E.load_config("E3")
    cands = _candidates()
    hold = float(exp_cfg["pilot"]["hold_s"])
    seeds = exp_cfg["pilot"]["seeds"]
    rows, wp_rows, piloted = [], [], sorted(os.listdir(os.path.join(PILOT, "raw")))
    for cid in piloted:
        cand = cands[cid]
        per = {c: [] for c in CONTROLLERS}
        terminated = []
        for c in CONTROLLERS:
            for s in seeds:
                path = os.path.join(PILOT, "raw", cid, c, f"seed_{s:04d}")
                errs, surv, why = episode_waypoint_errors(path, hold)
                per[c].append(errs)
                if not surv:
                    terminated.append(f"{c}/seed{s}:{why}")
        agg = {}
        for c in CONTROLLERS:
            for j in range(5):
                vals = {k: float(np.nanmean([per[c][si][j][k] for si in range(len(seeds))]))
                        for k in ("init_pos", "init_ori", "ss_pos", "ss_ori")}
                agg[(c, j)] = vals
                wp_rows.append({"candidate_id": cid, "pattern": cand["pattern"], "controller": c,
                                "waypoint": j, **{k: round(v, 6) for k, v in vals.items()}})
        worst = {k: max(agg[(c, j)][k] for c in CONTROLLERS for j in range(5))
                 for k in ("init_pos", "init_ori", "ss_pos", "ss_ori")}
        reasons = []
        if terminated:
            reasons.append("P1:termination " + ";".join(terminated))
        if worst["ss_pos"] >= TOL_POS or worst["ss_ori"] >= TOL_ORI:
            reasons.append("P2:steady-state error >= tolerance")
        if worst["init_pos"] >= 2 * TOL_POS or worst["init_ori"] >= 2 * TOL_ORI:
            reasons.append("P3:initial-response error >= 2x tolerance")
        W = max(max(agg[(c, j)]["ss_pos"] / TOL_POS, agg[(c, j)]["ss_ori"] / TOL_ORI)
                for c in CONTROLLERS for j in range(5))
        worst_ctrl_ss = {c: max(max(agg[(c, j)]["ss_pos"] / TOL_POS, agg[(c, j)]["ss_ori"] / TOL_ORI)
                                for j in range(5)) for c in CONTROLLERS}
        rows.append({"candidate_id": cid, "pattern": cand["pattern"], "span_m": cand["span_m"],
                     "ori_range_deg": cand["ori_range_deg"], "F": cand["F"],
                     "S_kin": cand["S_kin"],
                     "worst_init_pos_m": worst["init_pos"], "worst_init_ori_rad": worst["init_ori"],
                     "worst_ss_pos_m": worst["ss_pos"], "worst_ss_ori_rad": worst["ss_ori"],
                     "W": W, **{f"W_{c}": v for c, v in worst_ctrl_ss.items()},
                     "n_terminated": len(terminated), "passed": int(not reasons),
                     "rejection_reasons": "|".join(reasons)})
    selection = {}
    for pattern in ("line", "circle"):
        ok = [r for r in rows if r["pattern"] == pattern and r["passed"]]
        if not ok:
            selection[pattern] = None
            continue
        cmax = max(r["span_m"] for r in ok)
        omax = max(r["ori_range_deg"] for r in ok)
        for r in ok:
            r["S"] = (0.4 * r["span_m"] / cmax + 0.2 * r["ori_range_deg"] / omax
                      + 0.2 * r["F"] + 0.2 * (1.0 - r["W"]))
        ok.sort(key=lambda r: (-r["S"], -r["span_m"], r["candidate_id"]))
        for rank, r in enumerate(ok, 1):
            r["rank"] = rank
        selection[pattern] = ok[0]["candidate_id"]
    for r in rows:
        r.setdefault("S", float("nan"))
        r.setdefault("rank", "")
        r["selected"] = int(r["candidate_id"] in selection.values())
    rows.sort(key=lambda r: (r["pattern"], r["rank"] if r["rank"] != "" else 999, r["candidate_id"]))
    common.write_csv(os.path.join(PILOT, "pilot_screening.csv"), rows)
    common.write_csv(os.path.join(PILOT, "pilot_waypoint_errors.csv"), wp_rows)
    with open(os.path.join(PILOT, "selection.json"), "w") as h:
        json.dump({"selected": selection, "rule": "docs/E3_V3_PREREGISTRATION.md section 4",
                   "piloted": piloted, "seeds": seeds}, h, indent=1)
    for pattern in ("line", "circle"):
        rs = [r for r in rows if r["pattern"] == pattern]
        print(f"{pattern}: {sum(r['passed'] for r in rs)}/{len(rs)} pass the pilot; "
              f"selected {selection[pattern]}")
    return 0


def cmd_freeze(args):
    with open(os.path.join(PILOT, "selection.json")) as h:
        sel = json.load(h)["selected"]
    if not all(sel.get(p) for p in ("line", "circle")):
        raise SystemExit(f"selection incomplete: {sel}")
    cands = _candidates()
    from benchmark.core import frames

    path = os.path.join(common.BENCH, "configs", "E3.yaml")
    text = open(path).read()
    if "  waypoints: null\n" not in text:
        raise SystemExit("configs/E3.yaml waypoints already frozen")
    block = ["  waypoints:"]
    for pattern in ("line", "circle"):
        cand = cands[sel[pattern]]
        block.append(f"    {pattern}:")
        block.append(f"      candidate_id: {cand['id']}")
        block.append("      keypoints_plb:   # 9-D: kp0, kp0+R[0.3,0,0], kp0+R[0,0,0.3] (PLB)")
        for kp in cand["waypoints_kp"]:
            block.append("        - [" + ", ".join(f"{v:.7f}" for v in kp) + "]")
        block.append("      position_plb_m:   # informational (= kp0)")
        for kp in cand["waypoints_kp"]:
            p, _ = frames.keypoints_to_pose(np.asarray(kp))
            block.append("        - [" + ", ".join(f"{v:.4f}" for v in p) + "]")
    text = text.replace("  waypoints: null\n", "\n".join(block) + "\n")
    with open(path, "w") as h:
        h.write(text)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(os.path.join(V3, "protocol", "E3_config_sha256_frozen.txt"), "w") as h:
        h.write(f"{sha}  benchmark/configs/E3.yaml\n")
    print(f"frozen: {sel}\nsha256 {sha}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "select", "freeze"])
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--extra", default=None, help="additional candidate ids to pilot")
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(V3, "protocol"), exist_ok=True)
    return {"run": cmd_run, "select": cmd_select, "freeze": cmd_freeze}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
