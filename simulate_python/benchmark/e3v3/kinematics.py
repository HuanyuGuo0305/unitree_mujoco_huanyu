"""Controller-independent kinematic checks for E3 v3 waypoint candidates.

Pre-registered items K1-K5 (docs/E3_V3_PREREGISTRATION.md, section 3):
  K1 reachability   independent DLS IK on unitree_robots/b2wz1/ground.xml,
                    body gripperStator: position < 1 mm, orientation < 0.5 deg
  K2 joint margin   >= 0.10 rad inside C3's IK joint band (the tightest band)
  K3 clearance      >= 0.03 m arm-to-body/legs/ground, arm self-clearance
  K4 manipulability sigma_min(J) >= 0.05
  K5 support        nearest native-table position <= 0.05 m, and a table row
                    within 0.10 m whose orientation is within 20 deg
Each check is evaluated at each controller's own nominal stance (base pose,
leg and arm state logged at t = 3.0 s in E1 T1 seed 0).
"""

from __future__ import annotations

import csv
import os

import mujoco
import numpy as np

from benchmark.round3 import common  # noqa: F401  (sets sys.path)
from benchmark.controllers.base import ARM_ORDER, LEG_ORDER
from benchmark.controllers.registry import REPO_ROOT, load_baseline_config
from benchmark.core import frames

BENCH = common.BENCH
XML = os.path.join(REPO_ROOT, "unitree_robots", "b2wz1", "ground.xml")
E1RAW = os.path.join(BENCH, "results", "raw", "E1")
CONTROLLERS = ("C1", "C3", "C5")

K1_POS, K1_ORI_DEG = 1e-3, 0.5
K2_MARGIN = 0.10
K3_CLEAR = 0.03
K4_SIGMA = 0.05
K5_POS, K5_NEIGH, K5_ORI_DEG = 0.05, 0.10, 20.0

DISTAL = ("link02", "link03", "link04", "link05", "link06", "gripperStator", "gripperMover")
SELF_DISTAL = ("link04", "link05", "link06", "gripperStator", "gripperMover")
SELF_PROX = ("arm_base", "link00")
BODY = ("world", "base_link", "head_Link", "tail_link") + tuple(
    f"{leg}_{part}" for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf", "wheel_link"))


def c3_joint_band():
    cfg, _ = load_baseline_config("C3")
    ik = cfg["arm_ik"]
    hard = np.asarray(ik["arm_hard_joint_limits"], float).reshape(6, 2)
    mid = hard.mean(1)
    half = (hard[:, 1] - hard[:, 0]) * float(ik["soft_joint_pos_limit_factor"]) / 2.0
    m = float(ik["joint_limit_margin"])
    return mid - half + m, mid + half - m


def _states_at(path, t):
    with open(path) as h:
        r = csv.reader(h)
        hdr = next(r)
        rows = [list(map(float, x)) for x in r]
    a = np.asarray(rows)
    i = int(np.argmin(np.abs(a[:, hdr.index("t")] - t)))
    return {k: a[i, j] for j, k in enumerate(hdr)}


def nominal_stances(t=3.0):
    """Per-controller stance: base pose and leg/arm joint positions at t."""
    out = {}
    for c in CONTROLLERS:
        d = os.path.join(E1RAW, c, "T1", "seed_000")
        st = _states_at(os.path.join(d, "states.csv"), t)
        phys = np.load(os.path.join(d, "physics.npz"))["data"]
        row = phys[int(np.argmin(np.abs(phys[:, 0] - t)))]
        out[c] = {
            "base_pos": [st["base_x"], st["base_y"], st["base_z"]],
            "base_quat": [st["base_qw"], st["base_qx"], st["base_qy"], st["base_qz"]],
            "q_leg": row[1:13].tolist(),
            "q_arm": row[13:19].tolist(),
            "source": f"results/raw/E1/{c}/T1/seed_000 at t={t}s",
        }
    return out


class Kin:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        J = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self.aq = np.array([self.m.jnt_qposadr[J(n)] for n in ARM_ORDER])
        self.av = np.array([self.m.jnt_dofadr[J(n)] for n in ARM_ORDER])
        self.lq = np.array([self.m.jnt_qposadr[J(n)] for n in LEG_ORDER])
        self.hw_lo = np.array([self.m.jnt_range[J(n)][0] for n in ARM_ORDER])
        self.hw_hi = np.array([self.m.jnt_range[J(n)][1] for n in ARM_ORDER])
        self.band_lo, self.band_hi = c3_joint_band()
        self.ee = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "gripperStator")
        self.jp, self.jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        bid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        coll = lambda bodies: [g for g in range(self.m.ngeom)  # noqa: E731
                               if self.m.geom_bodyid[g] in {bid(b) for b in bodies}
                               and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
        self.pairs = [(a, b) for a in coll(DISTAL) for b in coll(BODY)]
        self.pairs += [(a, b) for a in coll(SELF_DISTAL) for b in coll(SELF_PROX)]
        self.fromto = np.zeros(6)

    def set_stance(self, stance):
        self.d.qpos[:] = 0.0
        self.d.qpos[0:3] = stance["base_pos"]
        self.d.qpos[3:7] = stance["base_quat"]
        self.d.qpos[self.lq] = stance["q_leg"]
        self.d.qpos[self.aq] = stance["q_arm"]
        self.base_pos = np.asarray(stance["base_pos"], float)
        self.base_quat = np.asarray(stance["base_quat"], float)
        self.q0 = np.asarray(stance["q_arm"], float)

    def target_world(self, p_plb, R_plb):
        origin, _ = frames.plb_pose_from_base(self.base_pos, self.base_quat, 0.0)
        Rp = frames.plb_rotmat(self.base_quat)
        return np.asarray(origin, float) + Rp @ p_plb, Rp @ R_plb

    def _fk(self, q):
        self.d.qpos[self.aq] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.xpos[self.ee].copy(), self.d.xmat[self.ee].reshape(3, 3).copy()

    def _solve_from(self, p_t, R_t, q0, iters=400):
        q = np.array(q0, float)
        for _ in range(iters):
            p, R = self._fk(q)
            ep = p_t - p
            Re = R_t @ R.T
            ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
            w = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            er = (ang / (2 * np.sin(ang)) * w) if ang > 1e-9 else 0.5 * w
            if np.linalg.norm(ep) < 1e-6 and np.linalg.norm(er) < 1e-6:
                break
            mujoco.mj_jacBody(self.m, self.d, self.jp, self.jr, self.ee)
            J = np.vstack([self.jp[:, self.av], self.jr[:, self.av]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([ep, er]))
            n = np.abs(dq).max()
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.hw_lo, self.hw_hi)
        p, R = self._fk(q)
        perr = float(np.linalg.norm(p_t - p))
        oerr = float(np.degrees(np.arccos(np.clip((np.trace(R_t.T @ R) - 1) / 2, -1, 1))))
        return q, perr, oerr

    def solve(self, p_plb, R_plb, q_init=None, restarts=8, rng=None):
        """Best IK solution (inside the C3 band preferred) from several starts."""
        p_t, R_t = self.target_world(p_plb, R_plb)
        rng = rng or np.random.default_rng(0)
        starts = [self.q0 if q_init is None else np.asarray(q_init, float), self.q0]
        starts += [rng.uniform(self.band_lo, self.band_hi) for _ in range(restarts)]
        best = None
        for q0 in starts:
            q, pe, oe = self._solve_from(p_t, R_t, q0)
            ok = pe < K1_POS and oe < K1_ORI_DEG
            margin = float(np.min(np.minimum(q - self.band_lo, self.band_hi - q)))
            key = (ok, margin)
            if best is None or key > best[0]:
                best = (key, q, pe, oe, margin)
            if ok and margin >= K2_MARGIN:
                break
        _, q, pe, oe, margin = best
        self._fk(q)
        mujoco.mj_jacBody(self.m, self.d, self.jp, self.jr, self.ee)
        J = np.vstack([self.jp[:, self.av], self.jr[:, self.av]])
        sigma = float(np.linalg.svd(J, compute_uv=False).min())
        clear = self.clearance()
        return {"q": q, "pos_err_m": pe, "ori_err_deg": oe, "band_margin_rad": margin,
                "sigma_min": sigma, "clearance_m": clear}

    def clearance(self, cap=0.2):
        mujoco.mj_forward(self.m, self.d)
        best = cap
        for a, b in self.pairs:
            dist = mujoco.mj_geomDistance(self.m, self.d, a, b, cap, self.fromto)
            best = min(best, dist)
        return float(best)


def table_support(table, p_plb, R_plb):
    """K5: nearest table position distance, and the smallest orientation
    difference among table rows within K5_NEIGH of the waypoint."""
    from scipy.spatial import cKDTree

    if not hasattr(table, "_v3_tree"):
        table._v3_tree = cKDTree(table.positions.astype(np.float64))
        table._v3_R = np.stack([frames.keypoints_to_pose(r)[1] for r in table.table])
    d_pos, _ = table._v3_tree.query(np.asarray(p_plb, float))
    idx = table._v3_tree.query_ball_point(np.asarray(p_plb, float), K5_NEIGH)
    if not idx:
        return float(d_pos), float("inf")
    Rs = table._v3_R[idx]
    cosang = (np.einsum("nij,ij->n", Rs, R_plb) - 1.0) / 2.0
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return float(d_pos), float(ang.min())
