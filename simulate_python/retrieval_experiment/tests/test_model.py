"""Physical model checks for sampled (side, density) cubes: nominal AND effective runtime values.

  * random (size, density) samples from the official intervals and the corners
    (lightest 4.0 cm @ 400, heaviest 6.0 cm @ 1200, nominal 5.0 cm @ 800):
    collision box half extents, mass = density * s^3, inertia = m s^2 / 6 (MuJoCo
    read-back), bounding radius, contact parameters, solver options;
  * spawn height s/2 + 1 mm and stable rest on the floor (|z - s/2| < 0.5 mm, no
    bounce, normal force = m g) for small / nominal / large and lightest / heaviest;
  * mass stays constant over an episode (no runtime change);
  * effective contact friction actually used by MuJoCo (cube-floor = mu_cube,
    cube-gripper = mu_cube * mu_shape via explicit pairs, wheel-floor = mu_shape);
  * base-mass / COM randomization applied and restored; reset state.

    python3 -m unittest retrieval_experiment.tests.test_model -v
"""
from __future__ import annotations

import os
import sys
import unittest

import mujoco
import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../..")))

from retrieval_experiment import scenarios as scen  # noqa: E402
from retrieval_experiment.simulator import RetrievalSim, resolve_repo_path, run_episode  # noqa: E402

EXP = yaml.safe_load(open(os.path.join(HERE, "..", "config.yaml")))
DEP = yaml.safe_load(open(resolve_repo_path(EXP["deploy_config"])))
torch.set_num_threads(1)
SIM = RetrievalSim(DEP, EXP, observation_noise="none")
BANK = scen.generate(6, EXP["master_seed"], EXP["distributions"], SIM.robot_shape_names)
RHO = EXP["cube"]["density_range"]
CORNERS = [(0.040, RHO[0]), (0.050, EXP["cube"]["training_density"]), (0.060, RHO[1]), (0.040, RHO[1]), (0.060, RHO[0])]


def random_samples(n=12):
    out = []
    for sc in scen.generate(n // 4, EXP["master_seed"], EXP["distributions"], SIM.robot_shape_names):
        for iv in EXP["size_intervals_cm"]:
            out.append((scen.side_from_interval(sc["cube"]["size_u"], iv),
                        scen.density_from_quantile(sc["cube"]["density_u"], RHO)))
    return out


class TestPerSample(unittest.TestCase):
    def test_geometry_mass_inertia(self):
        for s, rho in random_samples() + CORNERS:
            SIM.select_cube(s, rho)
            m = SIM.m
            mass = rho * s ** 3
            self.assertTrue(np.allclose(m.geom_size[SIM.cube_gid], [s / 2] * 3, atol=1e-15, rtol=0))
            self.assertEqual(int(m.geom_type[SIM.cube_gid]), int(mujoco.mjtGeom.mjGEOM_BOX))
            self.assertAlmostEqual(float(m.body_mass[SIM.cube_bid]) / mass, 1.0, places=12)
            self.assertTrue(np.allclose(m.body_inertia[SIM.cube_bid], [mass * s * s / 6.0] * 3, rtol=1e-9, atol=0))
            self.assertAlmostEqual(float(m.geom_rbound[SIM.cube_gid]), np.sqrt(3) * s / 2, places=9)
            self.assertTrue(np.allclose(m.body_ipos[SIM.cube_bid], 0.0))
            self.assertEqual(m.npair, 15)
            self.assertTrue(np.allclose(m.pair_solref, EXP["cube"]["solref"]))
            self.assertTrue(np.allclose(m.geom_solref[SIM.cube_gid], EXP["cube"]["solref"]))
            self.assertTrue(np.allclose(m.geom_solimp[SIM.cube_gid], EXP["cube"]["solimp"]))
        # expected extremes of the official parameter space
        self.assertAlmostEqual(RHO[0] * 0.04 ** 3, 0.0256, places=10)
        self.assertAlmostEqual(RHO[1] * 0.06 ** 3, 0.2592, places=10)

    def test_unrounded_side(self):
        s, rho = 0.0412345678912, 777.777777
        SIM.select_cube(s, rho)
        self.assertEqual(float(SIM.m.geom_size[SIM.cube_gid][0]), s / 2)
        self.assertEqual(SIM.side, s)
        self.assertEqual(SIM.density, rho)

    def test_solver_options(self):
        xml = mujoco.MjModel.from_xml_path(SIM.xml_path)
        self.assertEqual(int(xml.opt.noslip_iterations), int(EXP["mujoco_noslip_iterations"]))
        self.assertEqual(int(xml.opt.integrator), int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST))
        for s, rho in CORNERS:
            SIM.select_cube(s, rho)
            self.assertEqual(int(SIM.m.opt.noslip_iterations), int(EXP["mujoco_noslip_iterations"]))
            self.assertAlmostEqual(float(SIM.m.opt.timestep), 0.004)

    def test_spawn_and_rest(self):
        g = 9.81
        for s, rho in CORNERS:
            sc = scen.nominal_scenario(SIM.robot_shape_names, object_r=1.5, object_theta=0.3)
            SIM.reset(sc, s, rho)
            self.assertAlmostEqual(float(SIM.cube_spawn_w[2]), s / 2 + 0.001, places=6)
            m, d = SIM.m, SIM.d
            frozen = d.qpos.copy()
            vz_max = 0.0
            ca, va = SIM.cube_qadr, SIM.cube_vadr
            for k in range(250):                       # 1 s, robot frozen (no robot contact), cube free
                mujoco.mj_step(m, d)
                cq, cv = d.qpos[ca:ca + 7].copy(), d.qvel[va:va + 6].copy()
                d.qpos[:] = frozen
                d.qvel[:] = 0
                d.qpos[ca:ca + 7], d.qvel[va:va + 6] = cq, cv
                mujoco.mj_forward(m, d)
                if k > 10:
                    vz_max = max(vz_max, float(d.qvel[va + 2]))
            z = float(d.qpos[ca + 2])
            fz = 0.0
            f6 = np.zeros(6)
            for i in range(d.ncon):
                c = d.contact[i]
                if SIM.cube_gid in (c.geom1, c.geom2):
                    mujoco.mj_contactForce(m, d, i, f6)
                    fz += abs((c.frame.reshape(3, 3).T @ f6[:3])[2])
            mass = float(m.body_mass[SIM.cube_bid])
            self.assertLess(abs(z - s / 2), 5e-4, f"{s} @ {rho}: rest z {z}")
            self.assertLess(vz_max, 0.05, "cube bounced")
            self.assertLess(abs(fz - mass * g) / (mass * g), 0.02, f"{s} @ {rho}: normal force {fz} vs m g {mass * g}")

    def test_density_fixed_during_episode(self):
        s, rho = 0.0573, 1111.0
        summary, tr = run_episode(SIM, BANK[0], s, rho, record=True)
        self.assertEqual(float(tr["cube_mass"]), float(SIM.m.body_mass[SIM.cube_bid]))
        self.assertAlmostEqual(summary["cube_mass_kg"], rho * s ** 3, places=12)
        self.assertEqual(summary["cube_density"], rho)

    def test_effective_friction(self):
        sc = BANK[1]
        ph = sc["physics"]
        mu_c = ph["cube_mu_static"]
        for s, rho in ((0.040, RHO[0]), (0.060, RHO[1])):
            SIM.reset(sc, s, rho)
            SIM.infer_hl()
            for _ in range(5):                         # 0.5 s under the policy: wheels on the floor
                SIM.run_hl_interval()
                SIM.next_hl_action()
            m, d = SIM.m, SIM.d
            floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            shape_of = {int(g): i for i, g in enumerate(SIM.w["robot_gids"])}
            n_wheel = 0
            for i in range(d.ncon):
                c = d.contact[i]
                pair, mu = {int(c.geom1), int(c.geom2)}, float(c.friction[0])
                if pair == {SIM.cube_gid, floor}:
                    self.assertAlmostEqual(mu, mu_c, places=9)
                elif floor in pair:
                    other = (pair - {floor}).pop()
                    self.assertAlmostEqual(mu, ph["robot_mu_static"][shape_of[other]], places=9)
                    n_wheel += 1
            self.assertGreaterEqual(n_wheel, 4)
            d.qpos[7 + SIM.gripper_idx] = -0.6
            mujoco.mj_forward(m, d)
            R = d.xmat[SIM.ee_bid].reshape(3, 3)
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, R.flatten())
            d.qpos[SIM.cube_qadr:SIM.cube_qadr + 3] = d.xpos[SIM.ee_bid] + R @ np.array([0.12, 0.0, 0.0015 + s / 2])
            d.qpos[SIM.cube_qadr + 3:SIM.cube_qadr + 7] = q
            mujoco.mj_forward(m, d)
            grip = SIM.w["stator_gids"] | SIM.w["mover_gids"]
            seen = 0
            for i in range(d.ncon):
                c = d.contact[i]
                if SIM.cube_gid in (c.geom1, c.geom2):
                    other = int(c.geom2 if c.geom1 == SIM.cube_gid else c.geom1)
                    if other in grip:
                        self.assertAlmostEqual(float(c.friction[0]), mu_c * ph["robot_mu_static"][shape_of[other]], places=9)
                        seen += 1
            self.assertGreater(seen, 0)

    def test_base_randomization_restored(self):
        SIM.reset(BANK[0], 0.05, 800.0)
        w = SIM.w
        ph = BANK[0]["physics"]
        self.assertAlmostEqual(float(SIM.m.body_mass[SIM.base_bid]), w["nom_base_mass"] + ph["base_mass_add"], places=9)
        self.assertTrue(np.allclose(SIM.m.body_ipos[SIM.base_bid], w["nom_base_ipos"] + np.array(ph["base_com_offset"])))
        SIM.reset(scen.nominal_scenario(SIM.robot_shape_names), 0.05, 800.0)
        self.assertAlmostEqual(float(SIM.m.body_mass[SIM.base_bid]), w["nom_base_mass"], places=12)
        self.assertTrue(np.allclose(SIM.m.body_inertia[SIM.base_bid], w["nom_base_inertia"]))

    def test_reset_state(self):
        sc = BANK[2]
        init = sc["initial_state"]
        SIM.reset(sc, 0.05, 800.0)
        d = SIM.d
        self.assertAlmostEqual(float(d.qpos[2]), float(np.float32(DEP["root_pos"][2])) + init["base_dz"], places=6)
        self.assertAlmostEqual(float(d.qpos[7 + SIM.gripper_idx]), -np.pi / 2, places=9)
        arm = d.qpos[7:7 + SIM.nj][SIM.arm_idx]
        self.assertTrue(np.allclose(arm, np.asarray(DEP["default_joint_pos"], np.float32)[SIM.arm_idx] * np.float32(init["arm_scale"]), atol=1e-6))
        self.assertTrue(np.allclose(d.qvel[0:3], init["root_vel"][0:3]))
        self.assertTrue(np.all(SIM.uan_history == 0))
        tg = SIM.retrieval_target_pos_w
        self.assertLessEqual(np.hypot(tg[0] - d.qpos[0], tg[1] - d.qpos[1]), 0.8 + 1e-6)
        self.assertTrue(0.3 <= tg[2] <= 1.0)
        rel = SIM.cube_spawn_w[:2] - d.qpos[0:2]
        self.assertAlmostEqual(np.hypot(rel[0] - 0.8, rel[1]), sc["task"]["object_r"], places=4)


if __name__ == "__main__":
    unittest.main()
