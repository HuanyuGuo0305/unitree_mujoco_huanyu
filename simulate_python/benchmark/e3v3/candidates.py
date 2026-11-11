#!/usr/bin/env python3
"""E3 v3 candidate waypoint sets: generation and kinematic screening (K1-K7).

Pre-registered in docs/E3_V3_PREREGISTRATION.md sections 2-3. Controller-blind:
nothing here runs a controller. Output: results/E3_v3/screening/.

    python3 -m benchmark.e3v3.candidates [--jobs 16]
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os

import numpy as np

from benchmark.round3 import common
from benchmark.core import frames
from benchmark.e3v3 import kinematics as K
from benchmark.scenarios import experiments as E

OUT = os.path.join(common.BENCH, "results", "E3_v3", "screening")

CENTRES = [(x, 0.0, z) for x in (0.50, 0.60, 0.70) for z in (0.65, 0.80, 0.95)]
LINE_DIRS = {
    "y": (0, 1, 0), "z": (0, 0, 1), "y+z": (0, 1, 1), "y-z": (0, 1, -1),
    "x": (1, 0, 0), "x+y": (1, 1, 0),
}
LINE_LENGTHS = (0.30, 0.40, 0.50, 0.60)
_s30, _c30 = np.sin(np.radians(30)), np.cos(np.radians(30))
CIRCLE_PLANES = {
    "frontal": ((0, 1, 0), (0, 0, 1)),
    "sagittal": ((1, 0, 0), (0, 0, 1)),
    "horizontal": ((1, 0, 0), (0, 1, 0)),
    "frontal_tilt30": ((0, 1, 0), (_s30, 0, _c30)),
}
CIRCLE_RADII = (0.15, 0.20, 0.25, 0.30)
REF_ORIENTATIONS = ("nearest_row", "local_mean")
SCHEMES = ((0.35, 0.0), (0.5, 0.0), (0.8, 0.0), (0.5, 25.0), (np.inf, 25.0), (0.8, 25.0))
K6_MIN_RANGE_DEG, K6_MAX_STEP_DEG, K7_MIN_SPAN = 20.0, 60.0, 0.30


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot_a_to_b(a, b):
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rx(deg):
    t = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def geodesic_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def reference_orientation(table, centre, kind):
    centre = np.asarray(centre, float)
    if kind == "nearest_row":
        return frames.keypoints_to_pose(table.nearest_row(centre))[1]
    d = np.linalg.norm(table.positions - centre, axis=1)
    rows = table.table[d <= 0.10]
    M = sum(frames.keypoints_to_pose(r)[1] for r in rows)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def waypoint_positions(pattern, centre, geom, size):
    c = np.asarray(centre, float)
    if pattern == "line":
        u = _unit(LINE_DIRS[geom])
        return np.array([c + (-size / 2 + i * size / 4) * u for i in range(5)])
    e1, e2 = (_unit(v) for v in CIRCLE_PLANES[geom])
    phis = np.radians(90.0 + 72.0 * np.arange(5))
    return np.array([c + size * (np.cos(p) * e1 + np.sin(p) * e2) for p in phis])


def roll_profile(pattern):
    if pattern == "line":
        return np.linspace(-1.0, 1.0, 5)
    return np.cos(np.radians(72.0 * np.arange(5)))


def waypoint_orientations(pattern, centre, positions, R_c, D, rho):
    a = R_c[:, 0]
    pivot = np.asarray(centre, float) - (0.0 if np.isinf(D) else D) * a
    out = []
    for p, u in zip(positions, roll_profile(pattern)):
        Rt = np.eye(3) if np.isinf(D) else rot_a_to_b(a, p - pivot)
        out.append(Rt @ R_c @ rx(rho * u))
    return out


def traversal_pairs(pattern):
    pairs = [(i, i + 1) for i in range(4)]
    return pairs + [(4, 0)] if pattern == "circle" else pairs


def enumerate_candidates():
    cands = []
    for pattern, geoms, sizes in (("line", LINE_DIRS, LINE_LENGTHS),
                                  ("circle", CIRCLE_PLANES, CIRCLE_RADII)):
        for (ci, centre), geom, size, ref, (D, rho) in itertools.product(
                enumerate(CENTRES), geoms, sizes, REF_ORIENTATIONS, SCHEMES):
            cid = (f"{pattern[0].upper()}-c{centre[0]*100:.0f}z{centre[2]*100:.0f}-{geom}-"
                   f"{'L' if pattern == 'line' else 'r'}{size*100:.0f}-{ref}-"
                   f"D{'inf' if np.isinf(D) else f'{D*100:.0f}'}-roll{rho:.0f}")
            cands.append({"id": cid, "pattern": pattern, "centre": list(centre),
                          "geometry": geom, "size_m": size, "ref_orientation": ref,
                          "pivot_D_m": float(D), "roll_amp_deg": rho})
    return cands


_W = {}


def _init_worker():
    _W["kin"] = K.Kin()
    _W["stances"] = K.nominal_stances()
    _W["table"] = E.keypoint_table(E.load_config("E1")[1])


def build_poses(cand, table):
    P = waypoint_positions(cand["pattern"], cand["centre"], cand["geometry"], cand["size_m"])
    R_c = reference_orientation(table, cand["centre"], cand["ref_orientation"])
    Rs = waypoint_orientations(cand["pattern"], cand["centre"], P, R_c,
                               cand["pivot_D_m"], cand["roll_amp_deg"])
    return P, Rs


def screen(cand):
    kin, stances, table = _W["kin"], _W["stances"], _W["table"]
    P, Rs = build_poses(cand, table)
    res = dict(cand)
    fails = []
    # K6, K7 (geometry only)
    span = max(np.linalg.norm(P[i] - P[j]) for i in range(5) for j in range(i + 1, 5))
    ori_range = max(geodesic_deg(Rs[i], Rs[j]) for i in range(5) for j in range(i + 1, 5))
    steps = [geodesic_deg(Rs[i], Rs[j]) for i, j in traversal_pairs(cand["pattern"])]
    res.update(span_m=span, ori_range_deg=ori_range, max_step_ori_deg=max(steps),
               mean_step_pos_m=float(np.mean([np.linalg.norm(P[i] - P[j])
                                              for i, j in traversal_pairs(cand["pattern"])])))
    if ori_range < K6_MIN_RANGE_DEG:
        fails.append("K6:orientation range < 20 deg")
    if max(steps) > K6_MAX_STEP_DEG:
        fails.append("K6:consecutive orientation change > 60 deg")
    if span < K7_MIN_SPAN:
        fails.append("K7:span < 0.30 m")
    # K5 support
    sup = [K.table_support(table, p, R) for p, R in zip(P, Rs)]
    res["k5_max_pos_dist_m"] = max(s[0] for s in sup)
    res["k5_max_ori_dist_deg"] = max(s[1] for s in sup)
    if res["k5_max_pos_dist_m"] > K.K5_POS:
        fails.append("K5:position outside table support")
    if res["k5_max_ori_dist_deg"] > K.K5_ORI_DEG:
        fails.append("K5:orientation outside table support")
    # K1-K4 at each stance (skip IK when geometry already fails, to save time)
    worst = {"pos_err_m": 0.0, "ori_err_deg": 0.0, "band_margin_rad": np.inf,
             "sigma_min": np.inf, "clearance_m": np.inf}
    q_sol = {}
    if not fails:
        rng = np.random.default_rng(0)
        for c, st in stances.items():
            kin.set_stance(st)
            q_prev, qs = None, []
            for p, R in zip(P, Rs):
                s = kin.solve(p, R, q_init=q_prev, rng=rng)
                q_prev = s["q"]
                qs.append(s["q"].round(5).tolist())
                for k in ("pos_err_m", "ori_err_deg"):
                    worst[k] = max(worst[k], s[k])
                for k in ("band_margin_rad", "sigma_min", "clearance_m"):
                    worst[k] = min(worst[k], s[k])
            q_sol[c] = qs
        if worst["pos_err_m"] >= K.K1_POS or worst["ori_err_deg"] >= K.K1_ORI_DEG:
            fails.append("K1:IK not converged")
        if worst["band_margin_rad"] < K.K2_MARGIN:
            fails.append("K2:joint margin < 0.10 rad (C3 band)")
        if worst["clearance_m"] < K.K3_CLEAR:
            fails.append("K3:clearance < 0.03 m")
        if worst["sigma_min"] < K.K4_SIGMA:
            fails.append("K4:sigma_min < 0.05")
    res.update({f"k_{k}": v for k, v in worst.items()})
    res["F"] = float(min(1.0, worst["band_margin_rad"] / 0.30, worst["clearance_m"] / 0.10,
                         worst["sigma_min"] / 0.15)) if not fails else float("nan")
    res["passed"] = int(not fails)
    res["rejection_reasons"] = "|".join(fails)
    res["waypoints_kp"] = [frames.pose_to_keypoints(p, R).astype(float).round(7).tolist()
                           for p, R in zip(P, Rs)]
    res["ik_solutions"] = q_sol
    return res


def pilot_set(results, per_pattern=25, offset=0):
    """Top candidates per pattern by S_kin, at most 2 per geometry (pre-registered)."""
    chosen = {}
    for pattern in ("line", "circle"):
        ok = [r for r in results if r["pattern"] == pattern and r["passed"]]
        if not ok:
            chosen[pattern] = []
            continue
        cmax = max(r["span_m"] for r in ok)
        omax = max(r["ori_range_deg"] for r in ok)
        for r in ok:
            r["S_kin"] = 0.4 * r["span_m"] / cmax + 0.2 * r["ori_range_deg"] / omax + 0.2 * r["F"]
        ok.sort(key=lambda r: (-r["S_kin"], r["id"]))
        per_geom, ranked = {}, []
        for r in ok:
            g = (tuple(r["centre"]), r["geometry"], r["size_m"])
            if per_geom.get(g, 0) >= 2:
                continue
            per_geom[g] = per_geom.get(g, 0) + 1
            ranked.append(r)
        chosen[pattern] = ranked[offset:offset + per_pattern]
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    cands = enumerate_candidates()
    print(f"{len(cands)} candidates "
          f"({sum(c['pattern'] == 'line' for c in cands)} line, "
          f"{sum(c['pattern'] == 'circle' for c in cands)} circle)")
    with mp.get_context("fork").Pool(args.jobs, initializer=_init_worker) as pool:
        results = pool.map(screen, cands, chunksize=8)
    chosen = pilot_set(results)
    for r in results:
        r.setdefault("S_kin", float("nan"))
        r["in_pilot_set"] = int(any(r["id"] == c["id"] for v in chosen.values() for c in v))
    cols = ["id", "pattern", "centre", "geometry", "size_m", "ref_orientation", "pivot_D_m",
            "roll_amp_deg", "span_m", "ori_range_deg", "max_step_ori_deg", "mean_step_pos_m",
            "k5_max_pos_dist_m", "k5_max_ori_dist_deg", "k_pos_err_m", "k_ori_err_deg",
            "k_band_margin_rad", "k_sigma_min", "k_clearance_m", "F", "S_kin", "passed",
            "rejection_reasons", "in_pilot_set"]
    common.write_csv(os.path.join(OUT, "kinematic_screening.csv"),
                     [{k: (json.dumps(r[k]) if isinstance(r[k], list) else r[k]) for k in cols}
                      for r in results], cols)
    with open(os.path.join(OUT, "kinematic_screening_full.json"), "w") as h:
        json.dump(results, h, default=float)
    with open(os.path.join(OUT, "pilot_set.json"), "w") as h:
        json.dump({p: [c["id"] for c in v] for p, v in chosen.items()}, h, indent=1)
    from collections import Counter
    for pattern in ("line", "circle"):
        rs = [r for r in results if r["pattern"] == pattern]
        print(f"{pattern}: {sum(r['passed'] for r in rs)}/{len(rs)} pass; pilot set "
              f"{len(chosen[pattern])}")
        cnt = Counter(f for r in rs for f in r["rejection_reasons"].split("|") if f)
        for k, v in cnt.most_common():
            print(f"   {v:5d}  {k}")


if __name__ == "__main__":
    main()
