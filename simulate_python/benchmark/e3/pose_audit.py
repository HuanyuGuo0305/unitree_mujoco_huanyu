#!/usr/bin/env python3
"""E3 random-pose target set: validity screen and kinematic audit (controller-blind).

    python3 -m benchmark.e3.pose_audit [--jobs 16]

1. Validity (the ONLY filter, pre-declared): finite 9-D keypoints whose two
   offset keypoints are 0.30 m from kp0 and orthogonal (|dot| < 1e-3), i.e. a
   proper rigid pose in the documented representation. Rows that pass form the
   sampling set (uniform, as in training); written to configs/E3_feasible_rows.npy.
2. Audit (reported, never used to filter -- filtering on arm-only reachability
   would remove exactly the whole-body targets the WBC policies were trained on):
   a. (targets used by seeds 0-99, plus a seeded random sample of 3,000 table
      rows for context) arm-only IK within HARDWARE joint limits at each controller's nominal
      stance (E1 T1 seed 0, t = 3 s): pos < 1 mm, ori < 0.5 deg; joint margin,
      sigma_min, clearance, best C3 IK-band margin;
   b. rows unreachable at all nominal stances: retried at 24 base postures the
      controllers actually adopted while reaching random table targets in E1 T2;
   c. moving-base diagnostic: the targets actually used by seeds 0-99, at 24
      postures logged in E1 T4 (base moving, random EE targets).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os

import numpy as np

from benchmark.round3 import common
from benchmark.core import frames, scenario as sc
from benchmark.e3 import kinematics as K
from benchmark.scenarios import experiments as E

OUT = os.path.join(common.BENCH, "results", "E3", "protocol", "pose_audit")
E1RAW = os.path.join(common.BENCH, "results", "E1", "raw", "E1")   # moved in the 2026-11-12 cleanup
_W = {}


def logged_postures(cond, n_per_ctrl=8, seed=0):
    """Base pose + leg/arm state at random evaluated ticks of E1 episodes."""
    rng = np.random.default_rng(seed)
    out = []
    for c in ("C1", "C3", "C5"):
        for k in range(n_per_ctrl):
            s = int(rng.integers(0, 50))
            d = os.path.join(E1RAW, c, cond, f"seed_{s:03d}")
            phys = np.load(os.path.join(d, "physics.npz"))["data"]
            with open(os.path.join(d, "states.csv")) as h:
                r = csv.reader(h)
                hdr = next(r)
                st = np.asarray([list(map(float, x)) for x in r])
            i = int(rng.integers(150, len(st)))
            t = st[i, hdr.index("t")]
            row = phys[int(np.argmin(np.abs(phys[:, 0] - t)))]
            out.append({"base_pos": [st[i, hdr.index(k2)] for k2 in ("base_x", "base_y", "base_z")],
                        "base_quat": [st[i, hdr.index(f"base_q{c2}")] for c2 in "wxyz"],
                        "q_leg": row[1:13].tolist(), "q_arm": row[13:19].tolist(),
                        "source": f"{c}/{cond}/seed_{s:03d} t={t:.2f}"})
    return out


def _init():
    _W["kin"] = K.Kin()
    _W["stances"] = K.nominal_stances()
    _W["table"] = E.keypoint_table(E.load_config("E1")[1])
    _W["t2"] = logged_postures("T2", seed=1)


def _audit_row(r):
    kin, table = _W["kin"], _W["table"]
    p, R = frames.keypoints_to_pose(table.table[r])
    out = {"row": int(r), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
    rng = np.random.default_rng(r)
    reach_any = False
    for c, st in _W["stances"].items():
        kin.set_stance(st)
        s = kin.solve_hw(p, R, restarts=3, rng=rng)
        ok = s["pos_err_m"] < K.K1_POS and s["ori_err_deg"] < K.K1_ORI_DEG
        reach_any |= ok
        out.update({f"{c}_reach": int(ok), f"{c}_clear_m": s["clearance_m"],
                    f"{c}_sigma": s["sigma_min"], f"{c}_hw_margin": s["hw_margin_rad"],
                    f"{c}_c3band_margin": s["c3_band_margin_rad"]})
    out["reach_nominal_any"] = int(reach_any)
    out["reach_nominal_all"] = int(all(out[f"{c}_reach"] for c in _W["stances"]))
    out["reach_logged_E1T2_posture"] = -1
    if not reach_any:
        out["reach_logged_E1T2_posture"] = 0
        for st in _W["t2"]:
            kin.set_stance(st)
            s = kin.solve_hw(p, R, restarts=0, rng=rng)
            if s["pos_err_m"] < K.K1_POS and s["ori_err_deg"] < K.K1_ORI_DEG:
                out["reach_logged_E1T2_posture"] = 1
                break
    return out


def _moving(args):
    r, postures = args
    kin, table = _W["kin"], _W["table"]
    p, R = frames.keypoints_to_pose(table.table[r])
    rng = np.random.default_rng(r + 7)
    n = 0
    for st in postures:
        kin.set_stance(st)
        s = kin.solve_hw(p, R, restarts=0, rng=rng)
        n += int(s["pos_err_m"] < K.K1_POS and s["ori_err_deg"] < K.K1_ORI_DEG)
    return int(r), n, len(postures)


def validity(table):
    kp = table.table.astype(np.float64).reshape(-1, 3, 3)
    a, b = kp[:, 1] - kp[:, 0], kp[:, 2] - kp[:, 0]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    dot = np.abs(np.einsum("ij,ij->i", a, b)) / (na * nb)
    ok = np.isfinite(kp).all(axis=(1, 2)) & (np.abs(na - 0.3) < 1e-3) & (np.abs(nb - 0.3) < 1e-3) & (dot < 1e-3)
    return ok, {"max_|kp1-kp0|-0.3": float(np.abs(na - 0.3).max()),
                "max_|kp2-kp0|-0.3": float(np.abs(nb - 0.3).max()), "max_abs_cos": float(dot.max())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    cfg_e1, g = E.load_config("E1")
    table = E.keypoint_table(g)
    ok, vstats = validity(table)
    rows = np.nonzero(ok)[0].astype(np.int32)
    path = os.path.join(common.BENCH, "configs", "E3_feasible_rows.npy")
    np.save(path, rows)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"validity: {ok.sum()}/{ok.size} rows pass; {vstats}; sha256 {sha}")
    with mp.get_context("fork").Pool(args.jobs, initializer=_init) as pool:
        # targets actually used by seeds 0-99 (16 poses per seed)
        used = np.concatenate([sc.e3_pose_sequence(s, rows, E.E3_POSE_SEQUENCE_LENGTH)
                               for s in range(100)])
        uniq = np.unique(used)
        sample = np.random.default_rng(20261111).choice(rows, size=3000, replace=False)
        todo = np.union1d(uniq, sample)
        audit = []
        for i, a in enumerate(pool.imap(_audit_row, todo.tolist(), chunksize=8), 1):
            audit.append(a)
            if i % 500 == 0 or i == len(todo):
                print(f"  audit {i}/{len(todo)}", flush=True)
        postures = logged_postures("T4", seed=2)
        mov = []
        for i, m in enumerate(pool.imap(_moving, [(int(r), postures) for r in uniq], chunksize=4), 1):
            mov.append(m)
            if i % 500 == 0 or i == len(uniq):
                print(f"  moving-base {i}/{len(uniq)}", flush=True)
    common.write_csv(os.path.join(OUT, "table_kinematic_audit.csv"), audit)
    common.write_csv(os.path.join(OUT, "used_targets_moving_base_audit.csv"),
                     [{"row": r, "reachable_postures": n, "of": m} for r, n, m in mov])
    A = {k: np.array([a[k] for a in audit]) for k in audit[0]}
    used_set = set(used.tolist())
    sel = np.array([a["row"] in used_set for a in audit])
    samp = np.isin(A["row"], sample)
    frac = lambda x: float(np.mean(x))  # noqa: E731
    mv = np.array([n / m for _, n, m in mov])
    summary = {
        "table": table.path, "table_sha256": table.sha256, "n_rows": int(table.n_rows),
        "validity": vstats, "valid_rows": int(ok.sum()),
        "sampling_set": "all valid rows, uniform with replacement (as in training)",
        "feasible_rows_file": "configs/E3_feasible_rows.npy", "feasible_rows_sha256": sha,
        "audit_random_table_sample_3000": {
            "reach_nominal_C1": frac(A["C1_reach"][samp]), "reach_nominal_C3": frac(A["C3_reach"][samp]),
            "reach_nominal_C5": frac(A["C5_reach"][samp]), "reach_nominal_any": frac(A["reach_nominal_any"][samp]),
            "unreachable_nominal_but_reachable_at_logged_E1T2_posture":
                frac(A["reach_logged_E1T2_posture"][samp] == 1),
            "unreachable_at_nominal_and_logged": frac(A["reach_logged_E1T2_posture"][samp] == 0),
            "c3_band_reachable_given_reach_C1_stance": frac(A["C1_c3band_margin"][samp & (A["C1_reach"] == 1)] >= 0),
            "sigma_min_below_0.02_given_reach_C1": frac(A["C1_sigma"][samp & (A["C1_reach"] == 1)] < 0.02),
            "clearance_below_0.01_given_reach_C1": frac(A["C1_clear_m"][samp & (A["C1_reach"] == 1)] < 0.01),
        },
        "audit_used_targets_seeds_0_99": {
            "draws": int(used.size), "unique_rows": int(uniq.size),
            "reach_nominal_any": frac(A["reach_nominal_any"][sel]),
            "reach_nominal_all_three": frac(A["reach_nominal_all"][sel]),
            "c3_band_reachable_given_reach_C3_stance": frac(A["C3_c3band_margin"][sel & (A["C3_reach"] == 1)] >= 0),
            "unreachable_at_nominal_and_logged": frac(A["reach_logged_E1T2_posture"][sel] == 0),
            "moving_base_E1T4_postures_mean_fraction_reachable": float(mv.mean()),
            "moving_base_fraction_of_targets_reachable_at_>=1_posture": float(np.mean(mv > 0)),
            "moving_base_fraction_of_targets_reachable_at_all_postures": float(np.mean(mv == 1)),
        },
        "postures_E1T2": [p["source"] for p in logged_postures("T2", seed=1)],
        "postures_E1T4": [p["source"] for p in postures],
    }
    with open(os.path.join(OUT, "pose_audit_summary.json"), "w") as h:
        json.dump(summary, h, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("postures")}, indent=1))


if __name__ == "__main__":
    main()
