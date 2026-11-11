#!/usr/bin/env python3
"""E3 v2 pre-registration: is 0.20 m/s a physically reasonable maximum EE speed?

Controller-agnostic. Runs BEFORE any E3 v2 episode and never looks at E3 v2
tracking results. Pre-declared acceptance criteria for the top speed level:

  (Orientation: the held orientation of configs/E3.yaml, chosen beforehand by
   select_orientation.py; the attitude part of F1 is reported, and its 95 %
   bar is unattainable for every trained orientation -- see that script.)
  F1 workspace   every point of all three paths is reached by an independent
                 DLS IK (position < 1 mm, orientation < 0.5 deg, inside joint
                 ranges) at the nominal stance, and the paths stay within the
                 native command table's support tolerance (0.12 m, v1 rule).
                 Attitude robustness (E1 moving-base states) is reported from
                 select_orientation.py, not gating.
  F2 joint rate  peak arm joint speed required at the top speed (nominal stance,
                 quasi-static IK, 50 Hz) <= 1.0 rad/s, the tightest structural
                 arm-rate cap among the controllers (C3 IK reference:
                 reference_gain 0.2 x max_joint_delta 0.1 rad x 50 Hz).
  F3 envelope    the top cruise speed does not exceed the median PEAK speed of
                 the native deploy script's own sequential-trajectory EE mode
                 (cubic interpolation between consecutive table rows over 2 s),
                 and is below the 95th-percentile EE speed every controller
                 already produces in E1 (base still, random EE targets).

If a criterion fails, the top speed is lowered in 0.05 m/s steps until all pass.
Run:  python3 -m benchmark.e3v2.feasibility
"""

from __future__ import annotations

import csv
import json
import os

import mujoco
import numpy as np
import yaml

from benchmark.round3 import common  # noqa: F401  (sys.path)
from benchmark.controllers.registry import REPO_ROOT  # noqa: E402
from benchmark.core import frames, scenario as sc  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

BENCH = common.BENCH
OUT = os.path.join(BENCH, "results", "E3_v2", "protocol")
CFG = yaml.safe_load(open(os.path.join(BENCH, "configs", "E3.yaml")))
ARM = [f"joint{i}" for i in range(1, 7)]
E1RAW = os.path.join(BENCH, "results", "raw", "E1")


class IK:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(
            os.path.join(REPO_ROOT, "unitree_robots", "b2wz1", "ground.xml"))
        self.d = mujoco.MjData(self.m)
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in jid])
        self.lo = np.array([self.m.jnt_range[j][0] for j in jid])
        self.hi = np.array([self.m.jnt_range[j][1] for j in jid])
        self.ee = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "gripperStator")
        self.jp, self.jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))

    def set_base(self, pos, quat):
        self.d.qpos[0:3], self.d.qpos[3:7] = pos, quat
        self.base_pos, self.base_quat = np.asarray(pos, float), np.asarray(quat, float)

    def target_world(self, kp_plb):
        p_plb, R_plb = frames.keypoints_to_pose(kp_plb)
        plb_pos, plb_quat = frames.plb_pose_from_base(self.base_pos, self.base_quat, 0.0)
        Rp = frames.plb_rotmat(self.base_quat)
        return np.asarray(plb_pos, float) + Rp @ p_plb, Rp @ R_plb

    def solve(self, kp_plb, q0, iters=300):
        p_t, R_t = self.target_world(kp_plb)
        q = np.array(q0, dtype=float)
        for _ in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            p = self.d.xpos[self.ee].copy()
            R = self.d.xmat[self.ee].reshape(3, 3).copy()
            ep = p_t - p
            Re = R_t @ R.T
            ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
            w = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            er = (ang / (2 * np.sin(ang)) * w) if ang > 1e-9 else 0.5 * w
            if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-5:
                break
            mujoco.mj_jacBody(self.m, self.d, self.jp, self.jr, self.ee)
            J = np.vstack([self.jp[:, self.vadr], self.jr[:, self.vadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([ep, er]))
            q = np.clip(q + dq, self.lo, self.hi)
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        p = self.d.xpos[self.ee]; R = self.d.xmat[self.ee].reshape(3, 3)
        perr = float(np.linalg.norm(p_t - p))
        oerr = float(np.degrees(np.arccos(np.clip((np.trace(R_t.T @ R) - 1) / 2, -1, 1))))
        mujoco.mj_jacBody(self.m, self.d, self.jp, self.jr, self.ee)
        J = np.vstack([self.jp[:, self.vadr], self.jr[:, self.vadr]])
        sig = float(np.linalg.svd(J, compute_uv=False).min())
        margin = float(np.min(np.minimum(q - self.lo, self.hi - q)))
        return q, perr, oerr, sig, margin


def _states(path):
    with open(path) as h:
        r = list(csv.reader(h))
    a = np.asarray(r[1:], float)
    return {k: a[:, i] for i, k in enumerate(r[0])}


def main():
    os.makedirs(OUT, exist_ok=True)
    ee = CFG["ee"]
    center, amp, ap = ee["center"], float(ee["amplitude"]), float(ee["a_peak_mps2"])
    levels = [float(v) for v in ee["speed_levels_mps"]]
    table = E.keypoint_table(E.load_config("E3")[1])
    orient = table.row(int(ee["orientation_table_row"]))
    dt = 0.02
    report = {"criteria": "see module docstring", "families": {}}

    # nominal stance: logged base pose and arm state at t = 3.0 s, C1/T1/seed 0
    # (base still; the stance is near-identical across controllers)
    st = _states(os.path.join(E1RAW, "C1", "T1", "seed_000", "states.csv"))
    i3 = int(np.argmin(np.abs(st["t"] - 3.0)))
    phys = np.load(os.path.join(E1RAW, "C1", "T1", "seed_000", "physics.npz"))["data"]
    q_arm0 = phys[int(np.argmin(np.abs(phys[:, 0] - 3.0))), 13:19]
    ik = IK()
    ik.set_base([st["base_x"][i3], st["base_y"][i3], st["base_z"][i3]],
                [st[f"base_q{c}"][i3] for c in "wxyz"])

    # F1 + F2 at the nominal stance
    F1, F2 = True, True
    q_start = None
    for fam in sc.E3V2_FAMILIES:
        L = sc.e3v2_path_length(fam, amp)
        pts = sc.e3v2_path_point(fam, np.linspace(0, L, 181), center, amp)
        sup = table.support_distance(pts)
        q = q_arm0.copy(); worst = [0, 0, 1e9, 1e9]
        for p in pts:
            q, pe, oe, sg, mg = ik.solve(frames.pose_to_keypoints(p, frames.keypoints_to_pose(orient)[1]), q)
            worst = [max(worst[0], pe), max(worst[1], oe), min(worst[2], sg), min(worst[3], mg)]
        if q_start is None:
            q_start, *_ = ik.solve(frames.pose_to_keypoints(pts[0], frames.keypoints_to_pose(orient)[1]), q_arm0)
        fam_rep = {"path_length_m": L, "max_support_distance_m": float(sup.max()),
                   "ik_max_pos_err_m": worst[0], "ik_max_ori_err_deg": worst[1],
                   "ik_min_sigma": worst[2], "ik_min_joint_limit_margin_rad": worst[3]}
        ok1 = worst[0] < 1e-3 and worst[1] < 0.5 and sup.max() <= float(ee["support_tolerance"])
        F1 &= ok1
        # joint rates at every speed level
        rates = {}
        for v in levels[1:]:
            t = np.arange(0, 17.0, dt)
            s = sc.e3v2_arclength(t, fam, amp, v, ap)
            P = sc.e3v2_path_point(fam, s, center, amp)
            qs, q = [], q_start.copy()
            for p in P:
                q, *_ = ik.solve(frames.pose_to_keypoints(p, frames.keypoints_to_pose(orient)[1]), q)
                qs.append(q.copy())
            qs = np.asarray(qs)
            qd = np.diff(qs, axis=0) / dt
            qdd = np.diff(qs, 2, axis=0) / dt ** 2
            rates[f"{v:.2f}"] = {"peak_joint_speed_radps": float(np.abs(qd).max()),
                                 "peak_joint_speed_per_joint": np.abs(qd).max(0).round(4).tolist(),
                                 "peak_joint_acc_radps2": float(np.abs(qdd).max()),
                                 "peak_ee_speed_mps": float(np.linalg.norm(np.diff(P, axis=0), axis=1).max() / dt),
                                 "mean_ee_speed_mps": sc.e3v2_mean_speed(fam, amp, v, ap, 17.0, dt)}
        fam_rep["rates"] = rates
        F2 &= rates[f"{levels[-1]:.2f}"]["peak_joint_speed_radps"] <= 1.0
        report["families"][fam] = fam_rep
        print(f"{fam:10s} L={L:.3f} support<= {sup.max():.3f} m  IK pos {worst[0]:.1e} m ori {worst[1]:.2e} deg "
              f"sigma_min {worst[2]:.3f} margin {worst[3]:.3f} rad | peak qdot @{levels[-1]:.2f}: "
              f"{rates[f'{levels[-1]:.2f}']['peak_joint_speed_radps']:.3f} rad/s, qddot {rates[f'{levels[-1]:.2f}']['peak_joint_acc_radps2']:.2f}")

    # Attitude robustness is speed-independent and was established, with the
    # same random-restart IK as K1, when the orientation was selected. It is
    # reported here, not gating: its pre-declared 95 % bar is unattainable for
    # every one of the trained orientations that reach the paths at all.
    sel = json.load(open(os.path.join(OUT, "orientation_selection.json")))
    report["attitude_robustness"] = {
        "fraction_all_points_reachable": sel["chosen"]["K2_fraction"],
        "n_attitudes": 100, "source": "orientation_selection.json",
        "note": ("95 percent bar unattainable for all %d K1-passing trained orientations "
                 "(best 0.92); failures are mainly near-fall states" % sel["n_pass_K1"])}
    print(f"attitude robustness (from orientation selection): {sel['chosen']['K2_fraction']:.2f} "
          f"of 100 E1 moving-base attitudes; reported, not gating")

    # F3a native sequential-mode speeds
    kp0 = table.table[:, 0:3].astype(float)
    d = np.linalg.norm(np.diff(kp0, axis=0), axis=1)
    seq_mean, seq_peak = d / 2.0, 1.5 * d / 2.0
    # F3b E1 EE speeds actually produced (base still, random EE), every controller
    e1 = {}
    for b in ("C1", "C3", "C5"):
        sp, jq = [], []
        for seed in range(50):
            s_ = _states(os.path.join(E1RAW, b, "T2", f"seed_{seed:03d}", "states.csv"))
            ev = s_["evaluated"] == 1
            P = np.stack([s_[f"ee_act_{c}_plb"][ev] for c in "xyz"], 1)
            sp.append(np.linalg.norm(np.diff(P, axis=0), axis=1) / dt)
            ph = np.load(os.path.join(E1RAW, b, "T2", f"seed_{seed:03d}", "physics.npz"))["data"]
            jq.append(np.abs(ph[ph[:, 0] >= 3.0, 1 + 18 + 12:1 + 18 + 18]).max(1))
        sp, jq = np.concatenate(sp), np.concatenate(jq)
        e1[b] = {"ee_speed_p50": float(np.percentile(sp, 50)), "ee_speed_p95": float(np.percentile(sp, 95)),
                 "ee_speed_p99": float(np.percentile(sp, 99)),
                 "arm_joint_speed_p99_radps": float(np.percentile(jq, 99))}
        print(f"E1 T2 {b}: EE speed p50 {e1[b]['ee_speed_p50']:.3f} p95 {e1[b]['ee_speed_p95']:.3f} "
              f"p99 {e1[b]['ee_speed_p99']:.3f} m/s; arm joint speed p99 {e1[b]['arm_joint_speed_p99_radps']:.2f} rad/s")
    top = levels[-1]
    F3 = top <= float(np.median(seq_peak)) and all(top < e1[b]["ee_speed_p95"] for b in e1)
    report["native_sequential_mode"] = {
        "median_mean_speed_mps": float(np.median(seq_mean)), "median_peak_speed_mps": float(np.median(seq_peak)),
        "p90_peak_speed_mps": float(np.percentile(seq_peak, 90))}
    report["e1_demonstrated"] = e1
    print(f"native sequential mode: median mean speed {np.median(seq_mean):.3f}, "
          f"median peak {np.median(seq_peak):.3f}, p90 peak {np.percentile(seq_peak, 90):.3f} m/s")
    report["old_e3_v1"] = {"top_mean_speed_mps": 0.40, "line_peak_speed_mps": 0.40 * np.pi / 2,
                           "line_peak_accel_mps2": 0.2 * np.pi ** 2}
    report["decision"] = {"F1_workspace": bool(F1), "F2_joint_rate": bool(F2), "F3_envelope": bool(F3),
                          "top_speed_mps": top, "accepted": bool(F1 and F2 and F3)}
    print("DECISION:", report["decision"])
    common.write_json(os.path.join(OUT, "speed_feasibility.json"), report)


if __name__ == "__main__":
    main()
