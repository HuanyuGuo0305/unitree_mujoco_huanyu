#!/usr/bin/env python3
"""E3 v2 protocol audit. Read-only with respect to the canonical dataset.

Reads results/E3_v2/{raw,data,stats,protocol}; writes ONLY results/E3_v2/audit/.
Run after benchmark.e3v2.analyze:  python3 -m benchmark.e3v2.audit

Pre-declared definitions (fixed before any E3 v2 outcome was inspected):
  near-fall state   within 1.0 s before a termination, OR base height < 0.40 m,
                    OR |roll| > 20 deg, OR |pitch| > 20 deg; otherwise "normal"
  reachable         independent DLS IK residual < 1 mm and < 0.5 deg, warm start
                    from the nominal-stance solution + 3 random restarts, inside
                    (a) the hardware joint ranges and (b) C3's IK soft band
                    (90 % of the hardware range, centred, minus 0.01 rad)
  near-singular     smallest singular value of the 6x6 arm Jacobian < 0.05
                    (nominal stance: 0.14)
  C3 IK saturated   ik_delta_max >= 0.1 rad (the configured max_joint_delta;
                    the reference then moves at its 1.0 rad/s cap)
  actuator sat.     UAN arm action |a| > 0.95 on >= 1 of 6 joints (logged)
  practical effect  |paired mean difference| >= 10 % of the C1 mean
"""
from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats as sps

from benchmark.round3 import common  # noqa: F401
from benchmark.core import frames, scenario as sc, stats  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402
from benchmark.e3v2 import analyze as A  # noqa: E402
from benchmark.e3v2 import feasibility as F  # noqa: E402

ROOT = A.ROOT
OUT = os.path.join(ROOT, "audit")
RAW = A.RAW
CFG = A.CFG
BLEV, VLEV, FAMS, CTRL = A.BLEV, A.VLEV, A.FAMILIES, A.BASELINES
NEAR_FALL = {"pre_termination_s": 1.0, "z_min": 0.40, "tilt_max_deg": 20.0}
SING = 0.05
SAT = 0.1 - 1e-9
PRACTICAL_REL = 0.10
N_FEAS_SEEDS, N_FEAS_TICKS = 6, 8


def _read(p):
    return A._read(p)


# ---------------------------------------------------------------- 1 feasibility
def _c3_soft():
    import yaml
    ik = yaml.safe_load(open(os.path.join(os.path.dirname(common.BENCH), "configs",
                                          "b2wz1_vbc_plb_uan.yaml")))["arm_ik"]
    hard = np.asarray(ik["arm_hard_joint_limits"], float)
    mid, half = hard.mean(1), (hard[:, 1] - hard[:, 0]) * ik["soft_joint_pos_limit_factor"] / 2
    return mid - half + ik["joint_limit_margin"], mid + half - ik["joint_limit_margin"]


_IK = None
_NOMQ = {}


def _ik():
    global _IK
    if _IK is None:
        _IK = F.IK()
    return _IK


def _solve(ik, kp, q_warm, lo, hi, rng):
    ik.lo, ik.hi = lo, hi
    best = None
    for k in range(4):
        q0 = np.clip(q_warm, lo, hi) if k == 0 else rng.uniform(lo, hi)
        q, pe, oe, sg, mg = ik.solve(kp, q0)
        if best is None or pe + oe / 100 < best[1] + best[2] / 100:
            best = (q, pe, oe, sg, mg)
        if pe < 1e-3 and oe < 0.5:
            break
    return best


def feas_state(job):
    """IK reachability of the logged commanded target at logged base states.

    stratum "random": N_FEAS_TICKS uniformly random evaluated ticks;
    stratum "pre-termination": 2 random ticks from the final 1.0 s of a
    terminated episode (targeted, so near-fall states are actually covered).
    """
    b, bi, vi, fam, seed, stratum = job
    global _NOMQ
    if not _NOMQ:
        _NOMQ.update(_nominal_q_table())
    nominal_q = _NOMQ
    d = os.path.join(RAW, b, A.cond_id(bi, vi, fam), f"seed_{seed:03d}")
    st, cmd = _read(os.path.join(d, "states.csv")), _read(os.path.join(d, "commands.csv"))
    met = json.load(open(os.path.join(d, "metrics.json")))
    ev = np.nonzero(st["evaluated"] == 1)[0]
    rng = np.random.default_rng(hash((b, bi, vi, fam, seed)) % (2 ** 32))
    t_end = st["t"][ev[-1]]
    if stratum == "random":
        ticks = rng.choice(ev, size=min(N_FEAS_TICKS, ev.size), replace=False)
    else:
        last = ev[st["t"][ev] >= t_end - NEAR_FALL["pre_termination_s"]]
        ticks = rng.choice(last, size=min(2, last.size), replace=False)
    ik = _ik()
    hw = (ik.lo.copy(), ik.hi.copy()) if not hasattr(ik, "_hw") else ik._hw
    ik._hw = hw
    soft = _c3_soft()
    out = []
    for j in ticks:
        pos = [st["base_x"][j], st["base_y"][j], st["base_z"][j]]
        quat = [st[f"base_q{c}"][j] for c in "wxyz"]
        roll, pitch = np.degrees(st["base_roll"][j]), np.degrees(st["base_pitch"][j])
        near = (st["base_z"][j] < NEAR_FALL["z_min"] or abs(roll) > NEAR_FALL["tilt_max_deg"]
                or abs(pitch) > NEAR_FALL["tilt_max_deg"]
                or (not met["survived"] and t_end - st["t"][j] <= NEAR_FALL["pre_termination_s"]))
        k = int(np.argmin(np.abs(cmd["tick"] - st["tick"][j])))
        kp = np.array([cmd[f"ee_cmd_kp{i}"][k] for i in range(9)])
        ik.set_base(pos, quat)
        q_warm = nominal_q["Q"][nominal_q["tree"].query(kp[:3])[1]]
        r_hw = _solve(ik, kp, q_warm, *hw, rng)
        r_sf = _solve(ik, kp, r_hw[0], *soft, rng)
        ik.lo, ik.hi = hw
        out.append({"stratum": stratum, "controller": b, "trajectory": fam, "base_level": BLEV[bi], "ee_speed_mps": VLEV[vi],
                    "seed": seed, "t": float(st["t"][j]), "base_z": float(st["base_z"][j]),
                    "roll_deg": float(roll), "pitch_deg": float(pitch), "state": "near-fall" if near else "normal",
                    "reach_hw": int(r_hw[1] < 1e-3 and r_hw[2] < 0.5),
                    "reach_c3_soft": int(r_sf[1] < 1e-3 and r_sf[2] < 0.5),
                    "pos_resid_hw_m": r_hw[1], "ori_resid_hw_deg": r_hw[2],
                    "sigma_min": r_hw[3], "near_singular": int(r_hw[3] < SING),
                    "joint_margin_hw_rad": r_hw[4]})
    return out


def _nominal_q_table():
    """Nominal-stance IK solution for every distinct commanded target position."""
    st = F._states(os.path.join(F.E1RAW, "C1", "T1", "seed_000", "states.csv"))
    i3 = int(np.argmin(np.abs(st["t"] - 3.0)))
    phys = np.load(os.path.join(F.E1RAW, "C1", "T1", "seed_000", "physics.npz"))["data"]
    q0 = phys[int(np.argmin(np.abs(phys[:, 0] - 3.0))), 13:19]
    ik = F.IK()
    ik.set_base([st["base_x"][i3], st["base_y"][i3], st["base_z"][i3]], [st[f"base_q{c}"][i3] for c in "wxyz"])
    from scipy.spatial import cKDTree
    P, Q = [], []
    for vi in range(len(VLEV)):
        for fam in FAMS:
            sch = E.build_scenario("E3", CFG, A.cond_id(0, vi, fam), 0).ee.commands
            q = q0.copy()
            for kp in sch[::2]:
                q = ik.solve(kp, q)[0]
                P.append(np.asarray(kp[:3], float)); Q.append(q.copy())
    P, Q = np.asarray(P), np.asarray(Q)
    return {"tree": cKDTree(P), "Q": Q}


def audit_feasibility(workers):
    rng = np.random.default_rng(20261110)
    jobs = []
    for b in CTRL:
        for bi in range(5):
            for vi in range(5):
                for fam in FAMS:
                    for s in rng.choice(A.N_SEEDS, N_FEAS_SEEDS, replace=False):
                        jobs.append((b, bi, vi, fam, int(s), "random"))
    # every terminated episode's final second, so near-fall states are covered
    per = list(csv.DictReader(open(os.path.join(ROOT, "data", "E3v2_per_trial_45000.csv"))))
    for r in per:
        if r["completed"] == "0":
            jobs.append((r["controller"], BLEV.index(float(r["base_level"])),
                         VLEV.index(float(r["ee_speed_mps"])), r["trajectory"], int(r["seed"]),
                         "pre-termination"))
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for chunk in ex.map(feas_state, jobs, chunksize=8) for r in chunk]
    common.write_csv(os.path.join(OUT, "A1_feasibility_states.csv"), rows)
    return rows


# ---------------------------------------------------------------- 2 saturation
def sat_episode(job):
    b, bi, vi, fam, s = job
    d = os.path.join(RAW, b, A.cond_id(bi, vi, fam), f"seed_{s:03d}")
    ac, st = _read(os.path.join(d, "actuation.csv")), _read(os.path.join(d, "states.csv"))
    ev = ac["evaluated"] == 1
    evs = st["evaluated"] == 1
    out = {"controller": b, "bi": bi, "vi": vi, "family": fam, "seed": s,
           "uan_sat_frac": float(np.mean(ac["uan_action_saturation"][ev] > 0)) if ev.any() else np.nan}
    if b == "C3" and ev.any():
        satk = ac["ik_delta_max"][ev] >= SAT
        out["ik_sat_frac"] = float(satk.mean())
        pe, oe = st["ee_pos_err"][evs], st["ee_ori_err_deg"][evs]
        n = min(len(pe), len(satk))
        satk, pe, oe = satk[:n], pe[:n], oe[:n]
        out["ee_pos_err_sat"] = float(pe[satk].mean()) if satk.any() else np.nan
        out["ee_pos_err_unsat"] = float(pe[~satk].mean()) if (~satk).any() else np.nan
        out["ee_ori_err_sat_deg"] = float(oe[satk].mean()) if satk.any() else np.nan
        out["ee_ori_err_unsat_deg"] = float(oe[~satk].mean()) if (~satk).any() else np.nan
        # time from 3 s: is saturation only the start-up catch-up?
        t = st["t"][evs][:n]
        out["ik_sat_frac_from4s"] = float(satk[t >= 4.0].mean()) if (t >= 4.0).any() else np.nan
    return out


def audit_saturation(workers):
    jobs = [(b, bi, vi, f, s) for b in CTRL for bi in range(5) for vi in range(5) for f in FAMS
            for s in range(A.N_SEEDS)]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(sat_episode, jobs, chunksize=50))
    common.write_csv(os.path.join(OUT, "A2_saturation_per_episode.csv"), rows)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.only in ("all", "feas"):
        print("[A1] feasibility at logged base states")
        audit_feasibility(args.workers)
    if args.only in ("all", "sat"):
        print("[A2] IK / actuator saturation")
        audit_saturation(args.workers)
    print("raw audit tables written; summarise with benchmark.e3v2.audit_report")


if __name__ == "__main__":
    main()
