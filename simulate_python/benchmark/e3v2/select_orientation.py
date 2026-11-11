#!/usr/bin/env python3
"""E3 v2 pre-registration: choose the held EE orientation by kinematics only.

The v1 rule (orientation of the native table row nearest the path centre)
makes the upper-left arc of the circle and semicircle unreachable for the arm
(independent random-restart DLS IK, see speed_feasibility.json), so the rule is
replaced -- geometry (centre, radius) unchanged -- by this pre-declared one,
run before any E3 v2 episode and blind to every controller's tracking:

  candidates  orientations of native table rows within 0.15 m of the path
              centre (orientations the policies were trained on), de-duplicated
              at 5 deg geodesic distance;
  K1          every probe point of all three paths (13 per family, incl. P0)
              is reached (pos < 1 mm, ori < 0.5 deg, within joint ranges) at
              the nominal stance, best of the warm start + 6 random restarts;
  K2          >= 95 % of 100 base attitudes sampled from E1 moving-base
              episodes (C1/C3/C5 equally) reach every probe point;
  choice      among candidates passing K1 and K2, the smallest geodesic angle
              to the v1 orientation (minimal change). Ties: larger minimum
              joint-limit margin.
  fallback    (applied: no candidate reaches 95 %; best is 92 %, and the
              failing attitudes are mostly near-fall states with base height
              0.35-0.42 m and 12-25 deg pitch, see the attitude log) -- among K1
              passes, maximise the K2 fraction; ties within 0.005 go to the
              smallest angle to v1, then the larger joint-limit margin. Geometry
              (centre, radius, start point) is unchanged from v1.
Writes results/E3_v2/protocol/orientation_selection.{json,csv}.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from benchmark.e3v2 import feasibility as F
from benchmark.core import frames, scenario as sc
from benchmark.scenarios import experiments as E
from benchmark.round3 import common

EE = F.CFG["ee"]
C, A = np.asarray(EE["center"], float), float(EE["amplitude"])
PROBE = np.vstack([sc.e3v2_path_point(f, np.linspace(0, sc.e3v2_path_length(f, A), 13), C, A)
                   for f in sc.E3V2_FAMILIES])


def _geo(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def _setup():
    st = F._states(os.path.join(F.E1RAW, "C1", "T1", "seed_000", "states.csv"))
    i3 = int(np.argmin(np.abs(st["t"] - 3.0)))
    phys = np.load(os.path.join(F.E1RAW, "C1", "T1", "seed_000", "physics.npz"))["data"]
    q0 = phys[int(np.argmin(np.abs(phys[:, 0] - 3.0))), 13:19]
    nominal = ([st["base_x"][i3], st["base_y"][i3], st["base_z"][i3]],
               [st[f"base_q{c}"][i3] for c in "wxyz"])
    rng = np.random.default_rng(20261110)
    att = []
    for b in ("C1", "C3", "C5"):
        for k in range(34 if b != "C5" else 32):
            cond = ("T3", "T4")[k % 2]
            s_ = F._states(os.path.join(F.E1RAW, b, cond, f"seed_{int(rng.integers(0, 50)):03d}", "states.csv"))
            j = int(rng.choice(np.nonzero(s_["evaluated"] == 1)[0]))
            att.append(([s_["base_x"][j], s_["base_y"][j], s_["base_z"][j]],
                        [s_[f"base_q{c}"][j] for c in "wxyz"]))
    return q0, nominal, att


Q0, NOMINAL, ATT = _setup()


def evaluate(args):
    idx, R = args
    ik = F.IK()
    rng = np.random.default_rng(idx)
    ik.set_base(*NOMINAL)
    qs, worst_margin = [], 1e9
    for p in PROBE:
        best = None
        for k in range(7):
            q0 = Q0 if k == 0 else rng.uniform(ik.lo, ik.hi)
            q, pe, oe, sg, mg = ik.solve(frames.pose_to_keypoints(p, R), q0)
            if pe < 1e-3 and oe < 0.5:
                best = (q, mg); break
        if best is None:
            return {"row": idx, "K1": False}
        qs.append(best[0]); worst_margin = min(worst_margin, best[1])
    ok = 0
    for pos, quat in ATT:
        ik.set_base(pos, quat)
        good = True
        for p, qn in zip(PROBE, qs):
            reached = False
            for k in range(4):   # warm start from the nominal solution + 3 restarts
                q0 = qn if k == 0 else rng.uniform(ik.lo, ik.hi)
                _, pe, oe, _, _ = ik.solve(frames.pose_to_keypoints(p, R), q0)
                if pe < 1e-3 and oe < 0.5:
                    reached = True; break
            if not reached:
                good = False; break
        ok += good
    return {"row": idx, "K1": True, "K2_fraction": ok / len(ATT), "min_joint_margin_rad": worst_margin}


def main():
    table = E.keypoint_table(E.load_config("E3")[1])
    v1R = frames.keypoints_to_pose(table.nearest_row(C))[1]
    near = np.nonzero(np.linalg.norm(table.positions - C, axis=1) <= 0.15)[0]
    cands = []
    for i in near:
        R = frames.keypoints_to_pose(table.table[i])[1]
        if all(_geo(R, Rc) >= 5.0 for _, Rc in cands):
            cands.append((int(i), R))
    print(f"{len(near)} table rows within 0.15 m; {len(cands)} distinct orientations (5 deg)")
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(evaluate, cands))
    for r, (_, R) in zip(res, cands):
        r["angle_to_v1_deg"] = _geo(R, v1R)
        r["row_position"] = table.table[r["row"], 0:3].round(4).tolist()
    passing = [r for r in res if r["K1"] and r["K2_fraction"] >= 0.95]
    print(f"K1 pass: {sum(r['K1'] for r in res)}; K1 and K2 pass: {len(passing)}")
    out = os.path.join(common.BENCH, "results", "E3_v2", "protocol")
    common.write_csv(os.path.join(out, "orientation_candidates.csv"),
                     [{k: (str(v) if isinstance(v, list) else v) for k, v in r.items()} for r in
                      sorted(res, key=lambda r: r.get("angle_to_v1_deg", 999))])
    k2 = sorted((r["K2_fraction"] for r in res if r["K1"]), reverse=True)
    print(f"K2 fraction among K1 passes: best {k2[:5]}")
    if passing:
        rule = "K1 and K2 (>= 95 %), closest to v1"
        chosen = min(passing, key=lambda r: (round(r["angle_to_v1_deg"], 6), -r["min_joint_margin_rad"]))
    else:
        rule = "fallback: max K2 among K1 passes (ties 0.005 -> closest to v1 -> margin)"
        k1 = [r for r in res if r["K1"]]
        top = max(r["K2_fraction"] for r in k1)
        tied = [r for r in k1 if r["K2_fraction"] >= top - 0.005]
        chosen = min(tied, key=lambda r: (round(r["angle_to_v1_deg"], 6), -r["min_joint_margin_rad"]))
    # attitude-level log for the chosen orientation: which sampled states fail
    ik = F.IK(); rng = np.random.default_rng(1)
    R = frames.keypoints_to_pose(table.table[chosen["row"]])[1]
    ik.set_base(*NOMINAL)
    qn = [ik.solve(frames.pose_to_keypoints(p, R), Q0)[0] for p in PROBE]
    att_log = []
    for i, (pos, quat) in enumerate(ATT):
        ik.set_base(pos, quat); fails = 0
        for p, q0 in zip(PROBE, qn):
            ok = False
            for k in range(4):
                _, pe, oe, _, _ = ik.solve(frames.pose_to_keypoints(p, R), q0 if k == 0 else rng.uniform(ik.lo, ik.hi))
                if pe < 1e-3 and oe < 0.5:
                    ok = True; break
            fails += not ok
        rpy = np.degrees(frames.roll_pitch_yaw(np.asarray(quat))[:2])
        att_log.append({"attitude": i, "source_controller": ("C1", "C3", "C5")[min(i // 34, 2)],
                        "base_z_m": pos[2], "roll_deg": float(rpy[0]), "pitch_deg": float(rpy[1]),
                        "unreachable_probe_points": fails})
    common.write_csv(os.path.join(out, "orientation_attitude_log.csv"), att_log)
    print("rule:", rule); print("chosen:", chosen)
    common.write_json(os.path.join(out, "orientation_selection.json"),
                      {"rule_text": __doc__, "rule_applied": rule, "n_candidates": len(cands), "n_pass_K1": sum(r["K1"] for r in res),
                       "n_pass_K1_K2": len(passing), "chosen_table_row": chosen["row"],
                       "attitudes_failing": sum(a["unreachable_probe_points"] > 0 for a in att_log),
                       "chosen": chosen, "chosen_keypoints": table.table[chosen["row"]].tolist(),
                       "v1_table_row_nearest_centre_keypoints": table.nearest_row(C).tolist()})


if __name__ == "__main__":
    main()
