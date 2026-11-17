"""Physical validation of the E5 MuJoCo cube/gripper model.

    cd unitree_mujoco/simulate_python
    python3 -m retrieval_experiment.validate        # writes docs/PHYSICAL_VALIDATION.json

Pass criteria were fixed before any pilot rollout and are independent of the
high-level policy.

  V1 grasp squeeze + hold for size x density corners (lightest 4.0 cm @ 400, heaviest
     6.0 cm @ 1200 kg/m^3, ...), training min/max jaw friction products and random
     official samples. The robot is rigidly mounted (armature
     1e6 on every robot DOF except jointGripper, so contact forces cannot move it
     and no state is teleported) at the default joint pose with joint6 = 0, where
     the stator jaw is horizontal and faces up. The cube rests on the stator jaw
     at the fingertip pads (stator x = 0.13 m) and the TRAINING gripper actuator
     (DCMotor kp 64, kd 3, 30 N m, 2 rad/s, closed target 0 rad) closes from fully
     open for 1 s. Gravity is then rotated into the jaw plane (stator -y), i.e. the
     roll-90 deg carrying orientation commanded by the policy, where only friction
     holds the cube, for 3 s.
     PASS (grasp held): gripperStator AND gripperMover net contact force > 0.3 N at the
     end (simulator-confirmed grasp detectable and retained), final angle < -5 deg
     (proxy "not fully closed" satisfiable), slip <= 1 mm over the 3-s hold.
     Penetration: peak (incl. the closing impact) and steady state (t >= 0.5 s) are
     reported; steady state > 2 mm is listed as a documented exceedance with its
     cause (see V1_documented_penetration_exceedances) -- contact parameters are
     NOT re-tuned in response.
     Reported, not pass/fail: slip of the cube relative to the stator during the
     friction hold (MuJoCo soft-friction creep; PhysX static friction has none).
     A "slip <= 2 mm / 2.5 s" criterion was considered while designing this check
     and dropped once the creep was characterised; see docs/ALIGNMENT_REPORT.md.
     Reference rows (no pass/fail): MuJoCo default soft contact for the cube, and
     the pre-alignment deploy gripper (target -0.53 rad, ideal PD kd 10) with it.
  V2 cube-ground Coulomb friction: cube on the floor away from the robot, gravity
     tilted by theta. PASS: sticks iff tan(theta) < mu, theta = atan(mu) -/+ 6 deg,
     mu in {0.4, 0.7, 0.9}. (A cube tips over at tan(theta) > 1 before it can
     slide, so the training range mu >= 1 cannot be probed this way; the
     per-contact coefficient readback in tests/test_model.py covers it.)
  V2/V3 are repeated for the lightest, nominal and heaviest cube.
  V4 (OOD extension, 3-7 cm): at 3.0 / 7.0 cm x 400 / 1200 kg/m^3 the implementation checks
     (mass, inertia, geometry, Coulomb stick/slip, drop/rest) must pass; squeeze-and-hold at three
     jaw frictions and a large-cube placement sweep are reported as physical behaviour (no
     pass/fail, no tuning).
  V3 drop 0.10 m onto the floor: PASS rebound <= 1 mm (training restitution
     e_cube * e_ground = e_cube * 0 = 0).
"""
from __future__ import annotations

import json
import math
import os
import sys

import mujoco
import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from retrieval_experiment import scenarios as scen  # noqa: E402
from retrieval_experiment.simulator import RetrievalSim, resolve_repo_path  # noqa: E402


def _max_cube_penetration(sim):
    d = sim.d
    pen = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if sim.cube_gid in (c.geom1, c.geom2):
            pen = max(pen, -float(c.dist))
    return pen


def _cube_in_stator(sim):
    d = sim.d
    R = d.xmat[sim.ee_bid].reshape(3, 3)
    return R.T @ (d.xpos[sim.cube_bid] - d.xpos[sim.ee_bid])


def squeeze_hold(sim, side, density=800.0, target=0.0, pd_kd=None, x_local=0.13, mu=1.0, robot_mu=1.0):
    """V1 with a rigidly mounted robot. Cube-jaw friction = mu * robot_mu (PhysX multiply).
    ``pd_kd`` selects the pre-alignment ideal-PD gripper law."""
    sim.reset(scen.nominal_scenario(sim.robot_shape_names, object_r=2.5, cube_mu=mu,
                                    robot_mu=[robot_mu] * len(sim.robot_shape_names)), side, density)
    m, d = sim.m, sim.d
    gq, gv = 7 + sim.gripper_idx, 6 + sim.gripper_idx
    ca, va = sim.cube_qadr, sim.cube_vadr
    arm0 = m.dof_armature.copy()
    g0 = m.opt.gravity.copy()
    robot_dofs = [i for i in range(6 + sim.nj) if i != gv]
    d.qpos[7 + sim.arm_idx[5]] = 0.0                          # joint6 = 0: stator jaw faces up
    d.qpos[gq] = -math.pi / 2
    mujoco.mj_forward(m, d)
    R = d.xmat[sim.ee_bid].reshape(3, 3).copy()
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    d.qpos[ca:ca + 3] = d.xpos[sim.ee_bid] + R @ np.array([x_local, 0.0, 0.0015 + side / 2])
    d.qpos[ca + 3:ca + 7] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    gi_ctrl = sim.CTRL_JOINT_NAMES.index("jointGripper")
    if pd_kd is None:
        sim.gripper_target = target
        law = sim._gripper_dc_motor_torque
    else:
        law = lambda qq, qd: float(np.clip(64.0 * (target - qq) - pd_kd * qd, -30.0, 30.0))  # noqa: E731
    pen = pen_ss = 0.0
    try:
        m.dof_armature[robot_dofs] = 1.0e6
        for k in range(1000):                               # 1 s close, 3 s friction hold
            if k == 250:
                p1 = _cube_in_stator(sim).copy()
                m.opt.gravity[:] = R @ np.array([0.0, -9.81, 0.0])
            d.ctrl[:] = 0.0
            d.ctrl[gi_ctrl] = law(float(d.qpos[gq]), float(d.qvel[gv]))
            mujoco.mj_step(m, d)
            pk = _max_cube_penetration(sim)
            pen = max(pen, pk)
            if k >= 125:                                    # after the closing impact (0.5 s)
                pen_ss = max(pen_ss, pk)
        mujoco.mj_forward(m, d)
    finally:
        m.dof_armature[:] = arm0
        m.opt.gravity[:] = g0
    thr = float(sim.exp["task"]["gripper_cube_contact_force_threshold"])
    f_st = sim.cube_body_force(sim.w["stator_gids"])
    f_mv = sim.cube_body_force(sim.w["mover_gids"])
    loc = _cube_in_stator(sim)
    return {"side": side, "density": density, "mass_kg": float(m.body_mass[sim.cube_bid]),
            "jaw_mu": mu * robot_mu, "final_angle_deg": math.degrees(float(d.qpos[gq])),
            "f_stator_N": f_st, "f_mover_N": f_mv, "dual_contact": bool(f_st > thr and f_mv > thr),
            "max_penetration_mm": 1000 * pen, "steady_penetration_mm": 1000 * pen_ss, "slip_mm_3s": 1000 * float(np.linalg.norm(loc - p1)),
            "cube_in_stator_frame_mm": (1000 * loc).round(1).tolist(),
            "cube_weight_N": float(m.body_mass[sim.cube_bid]) * 9.81}


def _frozen_steps(sim, frozen, n):
    """Robot kinematically frozen; only the (robot-contact-free) cube evolves."""
    m, d = sim.m, sim.d
    ca, va = sim.cube_qadr, sim.cube_vadr
    for _ in range(n):
        mujoco.mj_step(m, d)
        cq, cv = d.qpos[ca:ca + 7].copy(), d.qvel[va:va + 6].copy()
        d.qpos[:] = frozen
        d.qvel[:] = 0.0
        d.qpos[ca:ca + 7], d.qvel[va:va + 6] = cq, cv
        mujoco.mj_forward(m, d)


def tilt_test(sim, mu, theta_deg, side=0.05, density=800.0):
    sim.reset(scen.nominal_scenario(sim.robot_shape_names, object_r=2.5, cube_mu=mu), side, density)
    m, d = sim.m, sim.d
    _frozen_steps(sim, d.qpos.copy(), 50)
    frozen = d.qpos.copy()
    p0 = d.xpos[sim.cube_bid].copy()
    th = math.radians(theta_deg)
    g0 = m.opt.gravity.copy()
    m.opt.gravity[:] = [9.81 * math.sin(th), 0.0, -9.81 * math.cos(th)]
    try:
        _frozen_steps(sim, frozen, 250)
    finally:
        m.opt.gravity[:] = g0
    return float(np.linalg.norm(d.xpos[sim.cube_bid][:2] - p0[:2]))


def drop_test(sim, side, density=800.0):
    sim.reset(scen.nominal_scenario(sim.robot_shape_names, object_r=2.5), side, density)
    m, d = sim.m, sim.d
    d.qpos[sim.cube_qadr + 2] = side / 2 + 0.10
    mujoco.mj_forward(m, d)
    frozen = d.qpos.copy()
    zs = []
    for _ in range(200):
        _frozen_steps(sim, frozen, 1)
        zs.append(float(d.qpos[sim.cube_qadr + 2]))
    zs = np.array(zs)
    i_land = int(np.argmax(zs < side / 2 + 0.0005))      # first contact
    return {"side": side, "density": density, "rebound_mm": 1000 * float(zs[i_land:].max() - side / 2),
            "rest_offset_mm": 1000 * float(zs[-1] - side / 2)}


def _fmt(r):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}


def main():
    torch.set_num_threads(1)
    exp = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    dep = yaml.safe_load(open(resolve_repo_path(exp["deploy_config"])))
    sim = RetrievalSim(dep, exp, observation_noise="none")
    rlo, rhi = exp["cube"]["density_range"]
    rho0 = float(exp["cube"]["training_density"])
    lightest, nominal, heaviest = (0.040, rlo), (0.050, rho0), (0.060, rhi)
    corners = [lightest, (0.040, rhi), nominal, (0.060, rlo), heaviest]
    bank = scen.generate(2, exp["master_seed"] + 999, exp["distributions"], sim.robot_shape_names)
    randoms = [(scen.side_from_interval(b["cube"]["size_u"], iv), scen.density_from_quantile(b["cube"]["density_u"], (rlo, rhi)))
               for b in bank for iv in exp["size_intervals_cm"]]
    out = {"V1_squeeze_hold": [], "V1_reference_default_contact": [], "V1_reference_predeploy_gripper": [],
           "V2_friction": [], "V3_drop": []}
    ok = True
    cases = [(s, r, 1.0, 1.0) for s, r in corners]
    cases += [(s, r, 1.0, 0.4) for s, r in (lightest, heaviest)]     # training minimum jaw product 1.0 x 0.4
    cases += [(s, r, 2.0, 1.2) for s, r in (lightest, heaviest)]     # training maximum jaw product 2.0 x 1.2
    cases += [(s, r, 1.5, 0.8) for s, r in randoms]                  # random official samples, mean frictions
    exceed = []
    for s, rho, mu, rmu in cases:
        r = squeeze_hold(sim, s, rho, mu=mu, robot_mu=rmu)
        r["grasp_held"] = bool(r["dual_contact"] and r["final_angle_deg"] < -5.0 and r["slip_mm_3s"] <= 1.0)
        r["steady_penetration_ok"] = bool(r["steady_penetration_mm"] <= 2.0)
        r["pass"] = r["grasp_held"] and r["steady_penetration_ok"]
        ok &= r["grasp_held"]
        if not r["steady_penetration_ok"]:
            exceed.append({k: r[k] for k in ("side", "density", "mass_kg", "jaw_mu", "steady_penetration_mm",
                                             "max_penetration_mm", "final_angle_deg")})
        out["V1_squeeze_hold"].append(r)
        print("V1", _fmt(r), flush=True)
    out["V1_documented_penetration_exceedances"] = {
        "cases": exceed,
        "cause": ("MuJoCo's time-constant contact parameterisation scales contact stiffness with the effective mass, "
                  "and at high jaw friction the tangential constraints lock the cube in an unseated multi-contact "
                  "configuration; the ~440 N stall squeeze is then resolved by distributed 1-5 mm overlaps. Present "
                  "already for the training-nominal 5 cm / 800 kg/m^3 cube at jaw mu 2.4 (1.6-3.4 mm over placements) "
                  "and larger for lighter cubes. The grasp is still established and held. Not corrected: removing it "
                  "requires re-parameterising the validated contact stiffness (excluded by the E5 protocol)."),
    }
    dflt = dict(exp)
    dflt["cube"] = dict(exp["cube"], solref=[0.02, 1.0], solimp=[0.9, 0.95, 0.001, 0.5, 2.0])
    sim_d = RetrievalSim(dep, dflt, observation_noise="none")
    for s, rho in (lightest, nominal, heaviest):
        r = squeeze_hold(sim_d, s, rho)
        out["V1_reference_default_contact"].append(r)
        print("V1-ref default contact, training gripper", _fmt(r), flush=True)
        r = squeeze_hold(sim_d, s, rho, target=-0.53, pd_kd=10.0)
        out["V1_reference_predeploy_gripper"].append(r)
        print("V1-ref default contact, pre-alignment gripper (-0.53, PD kd 10)", _fmt(r), flush=True)
    for s, rho in (lightest, nominal, heaviest):
        for mu in (0.4, 0.7, 0.9):
            crit = math.degrees(math.atan(mu))
            for dth, should_slide in ((-6.0, False), (6.0, True)):
                disp = tilt_test(sim, mu, crit + dth, s, rho)
                r = {"side": s, "density": rho, "mu": mu, "theta_deg": crit + dth, "displacement_m": disp,
                     "slid": disp > 0.01, "expected_slide": should_slide}
                r["pass"] = r["slid"] == should_slide
                ok &= r["pass"]
                out["V2_friction"].append(r)
                print("V2", _fmt(r), flush=True)
    for s, rho in (lightest, nominal, heaviest):
        r = drop_test(sim, s, rho)
        r["pass"] = bool(r["rebound_mm"] <= 1.0)
        ok &= r["pass"]
        out["V3_drop"].append(r)
        print("V3", _fmt(r), flush=True)
    # ---- V4: out-of-distribution extremes of the official 3-7 cm evaluation -------------------
    # Implementation checks (V2/V3 + model read-back) must pass; grasp outcomes are REPORTED as
    # physical behaviour of the unchanged model (no pass/fail, no tuning).
    v4 = {"implementation_checks": [], "grasp_behaviour": [], "large_cube_placement_sweep": []}
    for s, rho in ((0.03, rlo), (0.03, rhi), (0.07, rlo), (0.07, rhi)):
        sim.select_cube(s, rho)
        m = sim.m
        mass = rho * s ** 3
        chk = {"side": s, "density": rho, "mass_kg": float(m.body_mass[sim.cube_bid]),
               "mass_ok": abs(float(m.body_mass[sim.cube_bid]) - mass) < 1e-12,
               "inertia_ok": bool(np.allclose(m.body_inertia[sim.cube_bid], mass * s * s / 6.0, rtol=1e-9, atol=0)),
               "geometry_ok": bool(np.allclose(m.geom_size[sim.cube_gid], s / 2, atol=1e-15, rtol=0))}
        chk["coulomb_ok"] = all(
            (tilt_test(sim, mu, math.degrees(math.atan(mu)) - 6.0, s, rho) <= 0.01)
            and (tilt_test(sim, mu, math.degrees(math.atan(mu)) + 6.0, s, rho) > 0.01) for mu in (0.4, 0.9))
        dr = drop_test(sim, s, rho)
        chk["rebound_mm"] = dr["rebound_mm"]
        chk["rest_offset_mm"] = dr["rest_offset_mm"]
        chk["pass"] = bool(chk["mass_ok"] and chk["inertia_ok"] and chk["geometry_ok"] and chk["coulomb_ok"]
                           and dr["rebound_mm"] <= 1.0 and abs(dr["rest_offset_mm"]) < 0.5)
        ok &= chk["pass"]
        v4["implementation_checks"].append(chk)
        print("V4-impl", _fmt(chk), flush=True)
        for mu, rmu in ((1.0, 1.0), (1.0, 0.4), (2.0, 1.2)):
            r = squeeze_hold(sim, s, rho, mu=mu, robot_mu=rmu)
            r["grasp_held"] = bool(r["dual_contact"] and r["final_angle_deg"] < -5.0 and r["slip_mm_3s"] <= 1.0)
            v4["grasp_behaviour"].append(r)
            print("V4-grasp", _fmt({k: r[k] for k in ("side", "density", "jaw_mu", "final_angle_deg", "f_stator_N",
                                                     "max_penetration_mm", "steady_penetration_mm", "slip_mm_3s",
                                                     "dual_contact", "grasp_held")}), flush=True)
    for s in (0.06, 0.065, 0.07):
        for rho in (rlo, rhi):
            for x in (0.115, 0.12, 0.125, 0.13, 0.135):
                r = squeeze_hold(sim, s, rho, mu=1.0, robot_mu=1.0, x_local=x)
                v4["large_cube_placement_sweep"].append({"side": s, "density": rho, "x_local": x,
                                                         "final_angle_deg": r["final_angle_deg"], "dual_contact": r["dual_contact"],
                                                         "slip_mm_3s": r["slip_mm_3s"]})
    v4["interpretation"] = (
        "3 cm cubes are grasped and held at all friction levels (jaw stalls at -17..-24 deg, so the proxy 'not fully "
        "closed' < -5 deg still holds); the documented mass-scaled contact compliance grows for the lightest cubes at "
        "the maximum jaw friction (steady overlap up to ~7.8 mm for 3 cm @ 400 kg/m^3). 6.5-7 cm cubes need a ~55-67 deg "
        "jaw opening: seated deep in the jaws they are held, near the fingertips the non-parallel jaws wedge them out or "
        "let them slide unless friction is high -- a geometric property of the gripper, not an implementation error.")
    out["V4_ood_extremes"] = v4
    out["all_pass"] = bool(ok)                    # grasp/hold, Coulomb friction, drop (+ V4 implementation checks)
    out["all_pass_including_penetration"] = bool(ok and not exceed)
    os.makedirs(os.path.join(HERE, "docs"), exist_ok=True)
    with open(os.path.join(HERE, "docs", "PHYSICAL_VALIDATION.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(("ALL PASS" if ok else "SOME CHECKS FAILED") +
          (f" -- {len(exceed)} documented steady-penetration exceedance(s) > 2 mm: " +
           ", ".join(f"{e['side'] * 100:.1f} cm @ {e['density']:.0f} kg/m^3, jaw mu {e['jaw_mu']:.1f}: "
                     f"{e['steady_penetration_mm']:.2f} mm" for e in exceed) if exceed else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
