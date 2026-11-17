"""Isaac Lab <-> MuJoCo alignment checks (fast, executable).

    python3 -m unittest transport_experiment.tests.test_alignment -v
"""
from __future__ import annotations

import glob
import importlib.util
import math
import os
import sys
import unittest

import mujoco
import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../..")))

from transport_experiment.controllers import file_sha256  # noqa: E402
from transport_experiment.sim import (  # noqa: E402
    TransportSim, build_keypoints_from_kp0_yaw_pitch_roll_plb, minimum_jerk_blend,
    quat_from_euler_xyz_wxyz, resolve_repo_path,
)

ISAAC = "/home/huanyuguo/Workspace_huanyu/IsaacLab_huanyu"
TASK_DIR = f"{ISAAC}/source/isaaclab_tasks/isaaclab_tasks/direct"
LOGS = f"{ISAAC}/logs/rsl_rl"
EXP = yaml.safe_load(open(os.path.join(HERE, "../config.yaml")))
DEP = yaml.safe_load(open(resolve_repo_path(EXP["deploy_config"])))
RUNS = {
    "full": "unitree_b2wz1_hl_transport_uan/2026-10-27_14-31-10_single_beta_6d_four_pose_pos_rate_0p1",
    "pos_only": "unitree_b2wz1_hl_transport_pos_only_uan/2026-11-13_17-47-25_single_beta_3d_position_only",
    "ori_only": "unitree_b2wz1_hl_transport_ori_only_uan/2026-11-12_19-03-12_single_beta_3d_orientation_only",
}
HAVE_ISAAC = os.path.isdir(LOGS)


class _TolerantLoader(yaml.SafeLoader):
    """Isaac Lab dumps env.yaml with python/tuple etc. tags; read them as plain data."""


_TolerantLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/",
    lambda loader, suffix, node: (loader.construct_sequence(node, deep=True) if isinstance(node, yaml.SequenceNode)
                                  else loader.construct_mapping(node, deep=True) if isinstance(node, yaml.MappingNode)
                                  else loader.construct_scalar(node)),
)


def load_isaac_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=_TolerantLoader)


class TestAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.sim = TransportSim(DEP, low_policy_path=EXP["low_policy"]["path"])

    # ---------------- timing ----------------
    def test_timing(self):
        s = self.sim
        self.assertAlmostEqual(s.simulation_dt, 0.004)
        self.assertAlmostEqual(s.ll_dt, 0.02)
        self.assertAlmostEqual(s.hl_dt, 0.1)
        self.assertEqual(s.settle_steps, 5)
        self.assertEqual(s.max_episode_hl_steps, math.ceil(15.0 / (1.0 / 250.0 * 25)))   # DirectRLEnv
        self.assertEqual(s.max_episode_hl_steps - 1, 149)
        self.assertEqual((s.ee_fast_transition_steps, s.ee_low_transition_steps), (15, 20))
        self.assertAlmostEqual(s.m.opt.timestep, 0.004)

    # ---------------- task constants vs training params/env.yaml ----------------
    @unittest.skipUnless(HAVE_ISAAC, "Isaac Lab logs not available")
    def test_env_params_match_training(self):
        for name, run in RUNS.items():
            p = load_isaac_yaml(f"{LOGS}/{run}/params/env.yaml")["env_params"]
            self.assertEqual(p["ll_load_run"], "2026-10-27_12-47-31_uan_model_4000_multipose20260907")
            np.testing.assert_allclose(DEP["canonical_ee_positions"], p["canonical_ee_positions"], err_msg=name)
            for k in ("kp0_x_range", "kp0_y_range", "kp0_z_range", "ee_yaw_range", "ee_pitch_range", "ee_roll_range",
                      "ee_pos_action_residual_scale", "ee_rot_action_residual_scale", "base_ref_ranges",
                      "disk_center_offset_gripper", "disk_x_axis_gripper", "disk_y_axis_gripper", "disk_z_axis_gripper"):
                np.testing.assert_allclose(DEP[k], p[k], err_msg=f"{name}:{k}")
            for k in ("puck_radius", "puck_height", "disk_thickness", "puck_spawn_xy_randomization",
                      "puck_spawn_clearance", "puck_disk_contact_force_threshold", "puck_drop_no_contact_steps",
                      "reset_settle_time_s", "episode_length_s", "ee_fast_transition_time_s", "ee_low_transition_time_s",
                      "canonical_ee_yaw", "canonical_ee_pitch", "canonical_ee_roll", "kp_dx", "gripper_hold_pos"):
                dk = {"kp_dx": "ee_kp_dx"}.get(k, k)
                self.assertAlmostEqual(float(DEP[dk]), float(p[k]), msg=f"{name}:{k}")
            d = EXP["distributions"]
            np.testing.assert_allclose(d["puck_mass_range"], p["puck_mass_range"])
            np.testing.assert_allclose(d["puck_friction_range"], p["puck_friction_range"])
            np.testing.assert_allclose(d["physical_randomization"]["base_mass_add_range"], p["base_mass_add_range"])
            for ax in "xyz":
                np.testing.assert_allclose(d["physical_randomization"]["base_com_range"][ax], p["base_com_range"][ax])
            rr = d["reset_randomization"]
            np.testing.assert_allclose(rr["base_z"], p["reset_base_pose_range"]["z"])
            np.testing.assert_allclose(rr["roll"], p["reset_base_pose_range"]["roll"])
            np.testing.assert_allclose(rr["arm_joint_position_scale"], p["reset_arm_joint_position_scale_range"])
            for k, v in rr["root_velocity"].items():
                np.testing.assert_allclose(v, p["reset_base_velocity_range"][k])

    # ---------------- assets ----------------
    @unittest.skipUnless(HAVE_ISAAC, "Isaac Lab logs not available")
    def test_policy_identity(self):
        self.assertEqual(file_sha256(resolve_repo_path(EXP["low_policy"]["path"])),
                         file_sha256(f"{LOGS}/unitree_b2wz1_plb_wholebody_UAN_loco_manip/"
                                     "2026-10-27_12-47-31_uan_model_4000_multipose20260907/exported/policy.onnx"))
        for name, run in RUNS.items():
            self.assertEqual(file_sha256(resolve_repo_path(EXP["methods"][name]["policy"])),
                             file_sha256(f"{LOGS}/{run}/exported/policy.onnx"), name)
        uan_isaac = f"{ISAAC}/source/isaaclab_assets/data/Robots/Unitree/B2WZ1_DISK/uan/model_4000_multipose20260907_inference.pt"
        self.assertEqual(file_sha256(self.sim.uan_model_path), file_sha256(uan_isaac))

    @unittest.skipUnless(HAVE_ISAAC, "Isaac Lab logs not available")
    def test_onnx_matches_torchscript(self):
        """ONNX == exported policy.pt (normalizer + Beta mean in graph, output in [-1,1])."""
        import onnxruntime as ort
        rng = np.random.default_rng(0)
        for name, run in RUNS.items():
            sess = ort.InferenceSession(resolve_repo_path(EXP["methods"][name]["policy"]), providers=["CPUExecutionProvider"])
            ts = torch.jit.load(f"{LOGS}/{run}/exported/policy.pt", map_location="cpu").eval()
            n_in = sess.get_inputs()[0].shape[-1]
            for _ in range(32):
                x = rng.normal(0, 1, (1, n_in)).astype(np.float32)
                a = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                with torch.no_grad():
                    b = ts(torch.from_numpy(x)).numpy()
                np.testing.assert_allclose(a, b, atol=1e-4)
                self.assertTrue(np.all(np.abs(a) <= 1.0))

    # ---------------- EE command representation ----------------
    @unittest.skipUnless(HAVE_ISAAC, "Isaac Lab source not available")
    def test_keypoints_match_isaac_adapter(self):
        spec = importlib.util.spec_from_file_location(
            "isaac_cmd_adapter", f"{TASK_DIR}/b2wz1_hl_transport_UAN/modules/command_adapter.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rng = np.random.default_rng(1)
        n = 500
        kp0 = rng.uniform([0.4, -0.4, 0.1], [0.9, 0.4, 1.2], (n, 3)).astype(np.float32)
        ypr = rng.uniform([-1.047, -0.785, 0.785], [1.047, 0.785, 2.356], (n, 3)).astype(np.float32)
        ref = mod.build_keypoints_from_kp0_yaw_pitch_plb(
            torch.from_numpy(kp0), torch.from_numpy(ypr[:, 0]), torch.from_numpy(ypr[:, 1]),
            kp_dx=0.3, kp_dz=0.3, roll=torch.from_numpy(ypr[:, 2])).numpy()
        ours = np.stack([build_keypoints_from_kp0_yaw_pitch_roll_plb(kp0[i], *ypr[i]) for i in range(n)])
        np.testing.assert_allclose(ours, ref, atol=2e-6)

    def test_minimum_jerk(self):
        self.assertEqual(minimum_jerk_blend(0.0), 0.0)
        self.assertEqual(minimum_jerk_blend(1.0), 1.0)
        self.assertAlmostEqual(minimum_jerk_blend(0.5), 0.5)
        for tau in np.linspace(0, 1, 11):
            self.assertAlmostEqual(minimum_jerk_blend(tau), 10 * tau**3 - 15 * tau**4 + 6 * tau**5)

    def test_quat_from_euler_matches_isaac(self):
        # Isaac Lab quat_from_euler_xyz composes Rz(yaw) Ry(pitch) Rx(roll).
        for roll, pitch in [(0.1, -0.07), (-0.1, 0.1), (0.03, 0.0)]:
            q = quat_from_euler_xyz_wxyz(roll, pitch, 0.0)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q)
            cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            np.testing.assert_allclose(R.reshape(3, 3), Ry @ Rx, atol=1e-12)

    # ---------------- disk / puck geometry ----------------
    def test_disk_frame_matches_collision_geom(self):
        s = self.sim
        sc = _nominal_scenario()
        for scale in (1.0, 0.95, 1.05):
            sc["initial_state"]["arm_scale"] = scale
            sc["initial_state"]["base_roll"] = 0.08 * (scale - 1) * 20
            s.reset(sc)
            c, x, y, z, q = s.get_disk_frame_world()
            g = s.disk_gid
            np.testing.assert_allclose(s.d.geom_xpos[g], c, atol=1e-6)
            Rg = s.d.geom_xmat[g].reshape(3, 3)
            np.testing.assert_allclose(Rg[:, 2], z, atol=1e-6)     # disk normal == cylinder axis
            self.assertAlmostEqual(float(s.m.geom_size[g, 0]), 0.100)
            self.assertAlmostEqual(2 * float(s.m.geom_size[g, 1]), 0.004)
            self.assertEqual(int(s.m.geom_bodyid[g]), s.ee_bid)       # rigidly part of gripperStator
            # Puck spawned in disk frame: offset, axis aligned, at rest.
            p = s.get_puck_pos_disk()
            np.testing.assert_allclose(p[:2], sc["initial_state"]["puck_xy"], atol=1e-6)
            self.assertAlmostEqual(float(p[2]), 0.002 + 0.040 + 0.0005, places=6)
            self.assertLess(s.puck_tilt_deg(), 0.05)          # float32 arccos resolution
            self.assertEqual(float(np.abs(s.d.qvel[s.puck_dof_adr:s.puck_dof_adr + 6]).max()), 0.0)

    def test_puck_contact_parameters(self):
        s = self.sim
        sc = _nominal_scenario()
        for mass, mu in [(0.5, 0.3), (1.5, 1.0), (0.93, 0.61)]:
            sc["physics"].update(puck_mass=mass, puck_mu=mu)
            s.reset(sc)
            self.assertAlmostEqual(float(s.m.body_mass[s.puck_bid]), mass)
            I = s.m.body_inertia[s.puck_bid]
            self.assertAlmostEqual(float(I[0]), mass * (3 * 0.05**2 + 0.08**2) / 12)
            self.assertAlmostEqual(float(I[2]), 0.5 * mass * 0.05**2)
            s.run_hl_interval()                                       # 0.1 s under the frozen WBC
            mus = [s.d.contact[i].friction[0] for i in range(s.d.ncon)
                   if {s.d.contact[i].geom1, s.d.contact[i].geom2} == {s.puck_gid, s.disk_gid}]
            self.assertGreaterEqual(len(mus), 3, "multiccd should give a multi-point flat contact")
            np.testing.assert_allclose(mus, mu)                       # effective mu == puck mu (PhysX multiply x 1.0)

    def test_base_randomization_applied(self):
        s = self.sim
        sc = _nominal_scenario()
        sc["physics"].update(base_mass_add=1.5, base_com_offset=[0.01, -0.02, 0.005])
        s.reset(sc)
        self.assertAlmostEqual(float(s.m.body_mass[s.base_bid]), s.nom_base_mass + 1.5)
        np.testing.assert_allclose(s.m.body_inertia[s.base_bid], s.nom_base_inertia * (s.nom_base_mass + 1.5) / s.nom_base_mass)
        np.testing.assert_allclose(s.m.body_ipos[s.base_bid], s.nom_base_ipos + [0.01, -0.02, 0.005])
        s.reset(_nominal_scenario())
        self.assertAlmostEqual(float(s.m.body_mass[s.base_bid]), s.nom_base_mass)


def _nominal_scenario():
    return {
        "scenario_id": 0, "seed": [0],
        "physics": {"puck_mass": 0.5, "puck_mu": 0.6, "base_mass_add": 0.0, "base_com_offset": [0, 0, 0]},
        "initial_state": {"base_dz": 0.0, "base_roll": 0.0, "base_pitch": 0.0, "root_vel": [0] * 6,
                          "arm_scale": 1.0, "puck_xy": [0.004, -0.007]},
        "base_schedule": [{"v": [0.3, -0.2, 0.1], "steps": 200}],
        "ee_schedule": [{"target": 1, "hold": 200}],
    }


if __name__ == "__main__":
    unittest.main()
