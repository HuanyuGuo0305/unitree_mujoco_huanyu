"""Physical sanity checks for the E4 MuJoCo transport model (full robot, frozen WBC + UAN).

    cd unitree_mujoco/simulate_python
    python3 -m transport_experiment.validate            # writes docs/PHYSICAL_VALIDATION.{md,json}

Checks (all controller-agnostic; no method-specific tuning). "Frozen" checks hold the
robot kinematically still (qpos/qvel restored every physics step) so that the REAL
disk geom is a stationary rigid support; only the puck is simulated.
  P1 stationary LEVEL disk (frozen): puck stays supported, drift < 1 mm in 5 s, net force = m g
  P2 disk attachment: collision disk pose == virtual disk frame along a dynamic WBC episode
  P3 puck can slide off and fall under the WBC; drop detector fires after exactly 5 no-contact HL steps
  P4 contact classification: brief lift-off (counter rises then resets, no drop), puck on ground
  P5 friction (frozen disk tilted 12 / 25 deg): Coulomb stick/slip vs atan(mu); mass-independent
  P6 mass: frozen net contact force = m g; under the WBC the arm holding torque grows with mass
  D1 diagnostic (not pass/fail): disk tilt produced by the frozen WBC when holding HIGH with
     zero residual, and its consequence for a low/high-friction puck.
"""
from __future__ import annotations

import json
import os
import sys

import mujoco
import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from transport_experiment.controllers import HLMethod  # noqa: E402
from transport_experiment.sim import TransportSim, resolve_repo_path  # noqa: E402


def scenario(mass=0.5, mu=0.6, target=0, v=(0.0, 0.0, 0.0), xy=(0.0, 0.0)):
    return {
        "scenario_id": -1, "seed": [0],
        "physics": {"puck_mass": mass, "puck_mu": mu, "base_mass_add": 0.0, "base_com_offset": [0, 0, 0]},
        "initial_state": {"base_dz": 0.0, "base_roll": 0.0, "base_pitch": 0.0, "root_vel": [0] * 6,
                          "arm_scale": 1.0, "puck_xy": list(xy)},
        "base_schedule": [{"v": list(v), "steps": 1000}],
        "ee_schedule": [{"target": target, "hold": 1000}],
    }


def settle(sim, meth):
    meth.reset(sim)
    while sim.settling:
        meth.step(sim)
        sim.run_hl_interval()
        sim.post_step_checks()
        sim.advance_references()


def hold_steps(sim, n, rot_offset=None, meth=None, on_step=None):
    """Run n HL steps with zero residual (optionally a fixed EE roll offset)."""
    out = []
    for _ in range(n):
        if rot_offset is None:
            meth.step(sim)
        else:
            rot = np.clip(sim.rot_ref + np.asarray(rot_offset, np.float32), sim.ee_rot_cmd_min, sim.ee_rot_cmd_max)
            sim.set_ee_command(np.clip(sim.ee_kp0_ref, sim.ee_pos_cmd_min, sim.ee_pos_cmd_max), rot)
        sim.run_hl_interval()
        chk = sim.post_step_checks()
        chk["puck_disk"] = sim.get_puck_pos_disk().astype(float).tolist()
        chk["tilt"] = sim.puck_tilt_deg()
        out.append(chk)
        if on_step:
            on_step(chk)
        sim.advance_references()
    return out


def main():
    torch.set_num_threads(1)
    exp = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    dep = yaml.safe_load(open(resolve_repo_path(exp["deploy_config"])))
    sim = TransportSim(dep, low_policy_path=exp["low_policy"]["path"])
    meth = HLMethod("no_residual", None)
    R = {}

    # ---------------- frozen-disk helpers ----------------
    def frozen_disk(mass, mu, tilt_deg, seconds=5.0, xy=(0.0, 0.0)):
        """Real disk geom held still: robot gravity-compensated with huge implicit joint damping."""
        sim.reset(scenario(mass, mu, xy=xy))
        m, d = sim.m, sim.d
        a = sim.puck_qpos_adr
        nrv = sim.puck_dof_adr                                   # robot dofs are 0 .. nrv-1
        saved = (m.body_gravcomp.copy(), m.dof_damping.copy())
        robot_bodies = [b for b in range(1, m.nbody) if b != sim.puck_bid]
        m.body_gravcomp[robot_bodies] = 1.0
        m.dof_damping[:nrv] = 1e9
        try:
            _, x, y, z, _ = sim.get_disk_frame_world()
            z = z.astype(np.float64)
            target = np.array([0.0, 0.0, 1.0])
            axis = np.cross(z, target); s_ = np.linalg.norm(axis)
            q_level = np.array([1.0, 0, 0, 0])
            if s_ > 1e-12:
                ang = np.arctan2(s_, z @ target)
                q_level = np.r_[np.cos(ang / 2), np.sin(ang / 2) * axis / s_]
            R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, q_level)
            x_level = R0.reshape(3, 3) @ x.astype(np.float64)
            q_tilt = np.r_[np.cos(np.radians(tilt_deg) / 2), np.sin(np.radians(tilt_deg) / 2) * x_level]
            q_total = np.zeros(4); mujoco.mju_mulQuat(q_total, q_tilt, q_level)
            qb = np.zeros(4); mujoco.mju_mulQuat(qb, q_total, d.qpos[3:7].copy())
            d.qpos[3:7] = qb; d.qpos[2] += 1.0                   # robot in the air: nothing touches the floor
            mujoco.mj_forward(m, d)
            c, x, y, z, q = sim.get_disk_frame_world()
            d.qpos[a:a + 3] = c + xy[0] * x + xy[1] * y + (0.002 + 0.04 + 0.0005) * z
            d.qpos[a + 3:a + 7] = q
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            disk0 = c.copy()
            p0 = sim.get_puck_pos_disk()[:2].copy()
            forces, contact = [], []
            for k in range(int(seconds / sim.simulation_dt)):
                d.ctrl[:] = 0.0
                mujoco.mj_step(m, d)
                if (k + 1) % 25 == 0:
                    f = sim.puck_disk_force(); forces.append(f); contact.append(f > sim.puck_contact_threshold)
            c1 = sim.get_disk_frame_world()[0]
            tilt_meas = float(np.degrees(np.arccos(np.clip(sim.get_disk_frame_world()[3][2], -1, 1))))
            disp = float(np.linalg.norm(sim.get_puck_pos_disk()[:2] - p0))
            return {"disk_tilt_deg": tilt_meas, "disk_motion_mm": float(np.linalg.norm(c1 - disk0)) * 1e3,
                    "displacement_mm": disp * 1e3, "contact_all_hl_steps": bool(all(contact)),
                    "mean_normal_force_N": float(np.mean(forces[-10:])) if forces else 0.0,
                    "expected_net_force_N": mass * 9.81,     # static puck: net contact force (normal + friction) balances m g
                    "max_puck_tilt_deg": sim.puck_tilt_deg()}
        finally:
            m.body_gravcomp[:] = saved[0]
            m.dof_damping[:] = saved[1]

    # P1 stationary level disk
    p1 = {f"m{mass}_mu{mu}": frozen_disk(mass, mu, 0.0, xy=(0.006, -0.004)) for mass, mu in [(0.5, 0.3), (1.5, 0.3), (1.0, 1.0)]}
    R["P1_frozen_level_disk_5s"] = p1
    R["P1_pass"] = all(v["contact_all_hl_steps"] and v["displacement_mm"] < 1.0 and v["disk_tilt_deg"] < 0.05 and v["disk_motion_mm"] < 0.5
                       and abs(v["mean_normal_force_N"] / v["expected_net_force_N"] - 1) < 0.02 for v in p1.values())

    # P2 attachment along a dynamic episode (base + LOW target)
    sim.reset(scenario(target=1, v=(0.5, -0.4, 0.5)))
    settle(sim, meth)
    errs = []

    def chk_attach(_):
        c, x, y, z, q = sim.get_disk_frame_world()
        g = sim.disk_gid
        errs.append(max(float(np.abs(sim.d.geom_xpos[g] - c).max()),
                        float(np.abs(sim.d.geom_xmat[g].reshape(3, 3)[:, 2] - z).max())))
    hold_steps(sim, 40, meth=meth, on_step=chk_attach)
    R["P2_disk_attachment_max_err"] = max(errs)
    R["P2_pass"] = max(errs) < 1e-5

    # P3 slide-off and drop detection under the WBC: low friction + 45 deg roll tilt
    sim.reset(scenario(mu=0.3))
    settle(sim, meth)
    steps = hold_steps(sim, 25, rot_offset=[0.0, 0.0, -np.pi / 4])
    first_loss = next((i for i, s in enumerate(steps) if not s["puck_disk_contact"]), None)
    first_drop = next((i for i, s in enumerate(steps) if s["puck_drop"]), None)
    R["P3"] = {"first_no_contact_step": first_loss, "first_drop_step": first_drop,
               "drop_delay_steps": None if first_loss is None or first_drop is None else first_drop - first_loss + 1,
               "puck_disk_at_drop": steps[first_drop]["puck_disk"] if first_drop is not None else None,
               "stator_contact_after_loss": any(s["puck_stator_contact"] for s in steps[first_loss:]) if first_loss is not None else None}
    # Detector semantics: drop is raised exactly on the 5th CONSECUTIVE no-contact step
    # (counter 1..5 over the last five steps) and the puck has left the disk support.
    R["P3"]["no_contact_counter_sequence"] = [s["puck_no_contact_steps"] for s in steps]
    R["P3_pass"] = (first_drop is not None and first_drop >= 4
                    and [steps[i]["puck_no_contact_steps"] for i in range(first_drop - 4, first_drop + 1)] == [1, 2, 3, 4, 5]
                    and not any(steps[i]["puck_drop"] for i in range(first_drop))
                    and steps[first_drop]["puck_disk"][2] < 0.0)

    # P4 classification (high friction so the puck itself stays put)
    p4 = {}
    sim.reset(scenario(mu=1.0))
    settle(sim, meth)
    hold_steps(sim, 3, meth=meth)
    c, x, y, z, q = sim.get_disk_frame_world()
    a = sim.puck_qpos_adr
    sim.d.qpos[a:a + 3] = c + 0.0425 * z + 0.10 * z          # 10 cm lift at rest -> ~0.14 s airborne
    sim.d.qpos[a + 3:a + 7] = q
    sim.d.qvel[sim.puck_dof_adr:sim.puck_dof_adr + 6] = 0.0
    mujoco.mj_forward(sim.m, sim.d)
    s = hold_steps(sim, 8, meth=meth)
    p4["brief_lift_10cm"] = {"no_contact_counts": [x["puck_no_contact_steps"] for x in s], "dropped": any(x["puck_drop"] for x in s)}
    sim.d.qpos[a:a + 3] = [1.5, 1.5, 0.041]                    # puck on the ground, away from the robot
    sim.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
    sim.d.qvel[sim.puck_dof_adr:sim.puck_dof_adr + 6] = 0.0
    mujoco.mj_forward(sim.m, sim.d)
    s = hold_steps(sim, 6, meth=meth)
    ground = any(sim.d.contact[i].geom1 == 0 or sim.d.contact[i].geom2 == 0 for i in range(sim.d.ncon)
                 if sim.puck_gid in (sim.d.contact[i].geom1, sim.d.contact[i].geom2))
    p4["on_ground"] = {"disk_contact": any(x["puck_disk_contact"] for x in s),
                       "stator_contact": any(x["puck_stator_contact"] for x in s),
                       "dropped_after_5": s[4]["puck_drop"], "puck_touches_floor": ground}
    R["P4"] = p4
    nc = p4["brief_lift_10cm"]["no_contact_counts"]
    R["P4_pass"] = (not p4["brief_lift_10cm"]["dropped"] and max(nc) >= 1 and nc[-1] == 0
                    and not p4["on_ground"]["disk_contact"] and not p4["on_ground"]["stator_contact"]
                    and p4["on_ground"]["dropped_after_5"])

    # P5 friction on a frozen tilted disk: stick iff tan(tilt) < mu
    p5 = {}
    for tilt in (12.0, 25.0):
        for mu in (0.3, 0.6):
            for mass in (0.5, 1.5):
                p5[f"tilt{tilt:.0f}_mu{mu}_m{mass}"] = frozen_disk(mass, mu, tilt, seconds=2.0)
    R["P5_frozen_tilted_disk_2s"] = p5
    stick = lambda v: v["displacement_mm"] < 5.0 and v["contact_all_hl_steps"]   # noqa: E731
    R["P5_pass"] = all(stick(v) == (np.tan(np.radians(float(k.split("_")[0][4:]))) < float(k.split("_")[1][2:]))
                       for k, v in p5.items())

    # P6 mass: normal force (from P1/P5) and arm holding torque under the WBC
    p6 = {}
    for mass in (0.5, 1.5):
        sim.reset(scenario(mass=mass, mu=1.0))
        settle(sim, meth)
        tq = []
        hold_steps(sim, 10, meth=meth, on_step=lambda _: tq.append(np.abs(sim.d.actuator_force[16:22]).copy()))
        p6[f"m{mass}"] = {"mean_abs_arm_torque_Nm": np.mean(tq, axis=0).round(3).tolist(),
                          "joint2_torque_Nm": float(np.mean(tq, axis=0)[1])}
    R["P6"] = p6
    R["P6_pass"] = (p6["m1.5"]["joint2_torque_Nm"] > p6["m0.5"]["joint2_torque_Nm"]
                    and all(abs(v["mean_normal_force_N"] / v["expected_net_force_N"] - 1) < 0.02
                            for k, v in p5.items() if stick(v)))

    # D1 WBC hold diagnostic (documents a frozen-WBC property; not a pass/fail check)
    d1 = {}
    for mu in (0.3, 1.0):
        sim.reset(scenario(mass=1.0, mu=mu))
        settle(sim, meth)
        tilts = []
        st = hold_steps(sim, 60, meth=meth, on_step=lambda _: tilts.append(
            float(np.degrees(np.arccos(np.clip(sim.get_disk_frame_world()[3][2], -1, 1))))))
        d1[f"mu{mu}"] = {"disk_tilt_deg_at_1s_3s_6s": [tilts[9], tilts[29], tilts[59]],
                         "puck_dropped": any(x["puck_drop"] for x in st)}
    R["D1_wbc_hold_high_zero_residual"] = d1

    R["all_pass"] = all(R[k] for k in R if k.endswith("_pass"))
    out_dir = os.path.join(HERE, "docs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "PHYSICAL_VALIDATION.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(json.dumps(R, indent=1, default=float))
    return R


if __name__ == "__main__":
    main()
