"""Static Isaac Lab <-> MuJoCo alignment checks for E5 (checkpoints, constants, interfaces).

A. Checkpoints: SHA-256 of HL / WBC / UAN vs config and vs the Isaac Lab run
   artefacts; ONNX graph dims; HL ONNX == model_8000.pt rebuilt by hand
   (normalizer -> trunk -> heads -> softplus+1+eps -> Beta mean -> 2u-1);
   WBC ONNX == exported/policy.pt; deterministic inference.
B. Constants: every task/distribution/noise constant in config.yaml and the
   control-path constants in the deploy YAML vs the training run's params/env.yaml.
C. Interfaces: 9-D keypoint encoding vs the Isaac ``command_adapter`` (imported
   from the training source), absolute action decoding, binary gripper threshold,
   no force-close.

Isaac-dependent checks are skipped when the Isaac Lab tree is not present.

    python3 -m unittest retrieval_experiment.tests.test_alignment -v
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../..")))

from retrieval_experiment.simulator import RetrievalSim, build_keypoints_from_kp0_yaw_pitch_plb, resolve_repo_path  # noqa: E402

EXP = yaml.safe_load(open(os.path.join(HERE, "..", "config.yaml")))
DEP = yaml.safe_load(open(resolve_repo_path(EXP["deploy_config"])))
ISAAC = EXP["isaac_reference"]["root"]
RUN = os.path.join(ISAAC, EXP["isaac_reference"]["run_dir"])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_env_yaml():
    p = os.path.join(RUN, "params", "env.yaml")
    if not os.path.isfile(p):
        return None

    class L(yaml.SafeLoader):
        pass

    def multi(loader, suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return loader.construct_scalar(node)
    L.add_multi_constructor("", multi)
    return yaml.load(open(p), Loader=L)


ENV = load_env_yaml()


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=0)


class TestCheckpoints(unittest.TestCase):
    def test_hashes_match_config(self):
        for k in ("high_level", "low_level", "uan"):
            p = resolve_repo_path(EXP["policies"][k]["path"])
            self.assertEqual(sha(p), EXP["policies"][k]["sha256"], k)

    def test_hashes_match_isaac_artifacts(self):
        if not os.path.isdir(RUN):
            self.skipTest("Isaac Lab run not available")
        self.assertEqual(sha(os.path.join(RUN, "exported", "policy.onnx")), EXP["policies"]["high_level"]["sha256"])
        wbc = os.path.join(ISAAC, "logs/rsl_rl", EXP["policies"]["low_level"]["training_run"], "exported", "policy.onnx")
        self.assertEqual(sha(wbc), EXP["policies"]["low_level"]["sha256"])
        self.assertEqual(sha(os.path.join(ISAAC, EXP["policies"]["uan"]["isaac_asset"])), EXP["policies"]["uan"]["sha256"])
        # the training env names exactly this WBC run / checkpoint and this UAN file
        ep = ENV["env_params"]
        self.assertEqual(ep["ll_load_run"], EXP["policies"]["low_level"]["training_run"].split("/")[1])
        self.assertEqual(ep["ll_experiment_name"], EXP["policies"]["low_level"]["training_run"].split("/")[0])
        self.assertEqual(ep["ll_load_checkpoint"], "exported/policy.pt")
        net = ENV["robot"]["actuators"]["z1_arm_uan"]["network_file"]
        self.assertTrue(net.endswith("model_4000_multipose20260907_inference.pt"))

    def test_hl_onnx_is_model_8000_beta_mean(self):
        import onnxruntime as ort
        import torch
        import torch.nn.functional as F
        p = resolve_repo_path(EXP["policies"]["high_level"]["path"])
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        self.assertEqual(s.get_inputs()[0].shape, [1, 165])
        self.assertEqual(s.get_outputs()[0].shape, [1, 9])
        x = np.random.default_rng(0).normal(size=(64, 165)).astype(np.float32) * 2
        o = np.concatenate([s.run(None, {"obs": x[i:i + 1]})[0] for i in range(64)])
        o2 = np.concatenate([s.run(None, {"obs": x[i:i + 1]})[0] for i in range(64)])
        self.assertTrue(np.array_equal(o, o2), "ONNX inference is not deterministic")
        self.assertTrue(np.all(np.abs(o) < 1.0))
        ck = os.path.join(RUN, EXP["policies"]["high_level"]["checkpoint"])
        if not os.path.isfile(ck):
            self.skipTest("checkpoint not available")
        sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]

        def mlp(prefix, h, n):
            for i in range(n):
                h = F.linear(h, sd[f"{prefix}.{2 * i}.weight"], sd[f"{prefix}.{2 * i}.bias"])
                if i < n - 1:
                    h = F.elu(h)
            return h
        xt = torch.from_numpy(x)
        h = mlp("actor.shared_trunk", (xt - sd["actor_obs_normalizer._mean"]) / (sd["actor_obs_normalizer._std"] + 1e-2), 3)
        outs = []
        for head, n, dd in (("base_head", 4, 3), ("arm_head", 4, 5), ("gripper_head", 3, 1)):
            z = mlp(f"actor.{head}", h, n)
            a = F.softplus(z[:, :dd]) + 1.0 + 1e-6
            b = F.softplus(z[:, dd:]) + 1.0 + 1e-6
            outs.append(2.0 * (a / (a + b)).clamp(1e-6, 1 - 1e-6) - 1.0)
        ref = torch.cat(outs, -1).numpy()
        self.assertLess(float(np.abs(ref - o).max()), 1e-5)

    def test_wbc_onnx_matches_torchscript(self):
        import onnxruntime as ort
        import torch
        pt = os.path.join(ISAAC, "logs/rsl_rl", EXP["policies"]["low_level"]["training_run"], "exported", "policy.pt")
        if not os.path.isfile(pt):
            self.skipTest("WBC policy.pt not available")
        s = ort.InferenceSession(resolve_repo_path(EXP["policies"]["low_level"]["path"]), providers=["CPUExecutionProvider"])
        self.assertEqual(s.get_inputs()[0].shape[-1], 400)
        self.assertEqual(s.get_outputs()[0].shape[-1], 22)
        m = torch.jit.load(pt, map_location="cpu").eval()
        x = np.random.default_rng(1).normal(size=(32, 400)).astype(np.float32)
        o = np.concatenate([s.run(None, {s.get_inputs()[0].name: x[i:i + 1]})[0] for i in range(32)])
        with torch.no_grad():
            r = m(torch.from_numpy(x)).numpy()
        self.assertLess(float(np.abs(o - r).max()), 1e-4)

    def test_uan_shape(self):
        import torch
        u = torch.jit.load(resolve_repo_path(EXP["policies"]["uan"]["path"]), map_location="cpu").eval()
        with torch.inference_mode():
            out = u(torch.zeros(1, 300))
        self.assertEqual(tuple(out.shape), (1, 6))


@unittest.skipIf(ENV is None, "Isaac Lab params/env.yaml not available")
class TestConstantsVsTraining(unittest.TestCase):
    def test_task_constants(self):
        ep = ENV["env_params"]
        for k, v in EXP["task"].items():
            if k == "stage2_force_gripper_close_enabled":
                continue
            src = "contact_force_threshold" if k == "base_contact_force_threshold" else k
            self.assertTrue(close(ep[src], v), f"{k}: config {v} vs training {ep[src]}")
        # training-only bootstrap is ON in training but expires at iteration 500 of 8000
        self.assertTrue(ep["stage2_force_gripper_close_enabled"])
        self.assertLess(ep["stage2_force_gripper_close_disable_iteration"], 8000)
        self.assertFalse(EXP["task"]["stage2_force_gripper_close_enabled"])
        self.assertFalse(DEP["stage2_force_gripper_close_enabled"])

    def test_distributions(self):
        ep, ev, dist = ENV["env_params"], ENV["events"], EXP["distributions"]
        bm = ev["ball_physics_material"]["params"]
        self.assertTrue(close(bm["static_friction_range"], dist["cube_material"]["static_friction"]))
        self.assertTrue(close(bm["dynamic_friction_range"], dist["cube_material"]["dynamic_friction"]))
        self.assertTrue(close(bm["restitution_range"], dist["cube_material"]["restitution"]))
        self.assertTrue(bm["make_consistent"])
        rm = ev["robot_physics_material"]["params"]
        self.assertTrue(close(rm["static_friction_range"], dist["robot_material"]["static_friction"]))
        self.assertTrue(close(rm["dynamic_friction_range"], dist["robot_material"]["dynamic_friction"]))
        self.assertTrue(close(rm["restitution_range"], dist["robot_material"]["restitution"]))
        self.assertTrue(close(ENV["terrain"]["physics_material"]["static_friction"], dist["ground_friction"]))
        self.assertEqual(ENV["terrain"]["physics_material"]["friction_combine_mode"], "multiply")
        self.assertEqual(ENV["ball"]["spawn"]["physics_material"]["friction_combine_mode"], "multiply")
        self.assertTrue(close(ev["add_base_mass"]["params"]["mass_distribution_params"], dist["base_mass_add"]))
        for a in "xyz":
            self.assertTrue(close(ev["base_com"]["params"]["com_range"][a], dist["base_com"][a]))
        r = dist["reset"]
        self.assertTrue(close(ep["reset_base_pose_range"]["z"], r["base_z"]))
        self.assertTrue(close(ep["reset_base_pose_range"]["roll"], r["roll"]))
        self.assertTrue(close(ep["reset_base_pose_range"]["pitch"], r["pitch"]))
        for k in ("x", "y", "z", "roll", "pitch", "yaw"):
            self.assertTrue(close(ep["reset_base_velocity_range"][k], r["root_velocity"][k]))
        self.assertTrue(close(ep["reset_leg_joint_position_scale_range"], r["leg_joint_position_scale"]))
        self.assertTrue(close(ep["reset_arm_joint_position_scale_range"], r["arm_joint_position_scale"]))
        o = dist["object"]
        self.assertTrue(close([ep["object_sampling_center_x"], ep["object_sampling_center_y"]], o["center_xy"]))
        self.assertTrue(close(ep["object_sampling_r_range"], o["r"]))
        self.assertTrue(close(ep["object_sampling_theta_max"], o["theta_max"]))
        self.assertTrue(close(ep["retrieval_target_radius"], dist["retrieval_target"]["radius"]))
        self.assertTrue(close(ep["retrieval_target_z_range_w"], dist["retrieval_target"]["z_range_w"]))

    def test_cube(self):
        ep = ENV["env_params"]
        lo, hi = ep["object_half_extent_range"]
        ivs = EXP["size_intervals_cm"]
        # the four intervals tile exactly the training size range 2*[0.02, 0.03] m
        self.assertAlmostEqual(ivs[0][0] / 100, 2 * lo, places=12)
        self.assertAlmostEqual(ivs[-1][1] / 100, 2 * hi, places=12)
        for a, b in zip(ivs[:-1], ivs[1:]):
            self.assertEqual(a[1], b[0])
        # training density is fixed; E5's range is an evaluation-time +-50 % perturbation around it
        self.assertTrue(close(ep["object_density"], EXP["cube"]["training_density"]))
        rl, rh = EXP["cube"]["density_range"]
        self.assertTrue(close([rl, rh], [0.5 * ep["object_density"], 1.5 * ep["object_density"]]))
        self.assertTrue(close(ep["object_spawn_z_epsilon"], EXP["cube"]["spawn_z_epsilon"]))
        self.assertTrue(close(ep["object_spawn_z_epsilon"], DEP["object_spawn_z_epsilon"]))

    def test_noise_model(self):
        ep, nm = ENV["env_params"], EXP["observation_noise_model"]
        self.assertTrue(ENV["add_noise"] and ep["obs_noise"])
        self.assertTrue(close(ep["object_pos_detection_noise"], nm["object_center_pos_base"]))
        self.assertTrue(close(ep["gripper_center_pos_noise"], nm["gripper_center_pos_base"]))
        self.assertTrue(close(ep["object_target_error_noise"], nm["retrieval_target_pos_base"]))
        self.assertTrue(close(ep["gripper_orientation_noise_std"], nm["gripper_orientation_std_rad"]))
        # proprioceptive magnitudes are literals in _build_hl_obs_frames
        src = open(os.path.join(ISAAC, EXP["isaac_reference"]["env_source"])).read()
        for term, k in (("root_ang_vel_b, 0.15", "root_ang_vel_b"), ("projected_gravity_b, 0.05", "projected_gravity_b"),
                        ("joint_pos_leg, 0.01", "leg_pos_rel"), ("joint_pos_arm, 0.01", "arm_pos_rel"),
                        ("joint_pos_gripper, 0.01", "gripper_pos_rel"), ("joint_vel_arm, 0.50", "arm_joint_vel")):
            self.assertIn(f"self._add_observation_noise(self._robot.data.{term})"
                          if "root" in term or "gravity" in term else f"self._add_observation_noise({term})", src)
            self.assertTrue(close(float(term.split(",")[1]), nm[k]))

    def test_control_path_constants(self):
        ep = ENV["env_params"]
        for k in ("base_cmd_scale", "kp0_x_range", "kp0_y_range", "kp0_z_range", "ee_yaw_range", "ee_pitch_range",
                  "neutral_kp0", "neutral_ee_yaw", "neutral_ee_pitch", "fixed_ee_roll", "gripper_open_pos",
                  "gripper_close_pos", "gripper_binary_threshold", "gripper_center_offset_local",
                  "object_sampling_center_x", "object_sampling_center_y", "object_sampling_r_range",
                  "object_sampling_theta_max", "episode_length_s", "ground_z"):
            self.assertTrue(close(DEP[k], ep[k], 1e-6), f"{k}: deploy {DEP[k]} vs training {ep[k]}")
        self.assertTrue(close(DEP["ee_kp_dx"], ep["kp_dx"]) and close(DEP["ee_kp_dz"], ep["kp_dz"]))
        self.assertTrue(close(DEP["enlarged_retrieval_target_radius"], ep["retrieval_target_radius"]))
        self.assertTrue(close(DEP["enlarged_retrieval_target_z_range_w"], ep["retrieval_target_z_range_w"], 1e-6))
        self.assertEqual(ep["ll_decimation"], DEP["ll_control_decimation"])
        self.assertEqual(ep["ll_steps_per_hl_step"], DEP["ll_steps_per_hl_step"])
        self.assertTrue(close(ENV["sim"]["dt"], DEP["simulation_dt"]))
        self.assertEqual(ENV["decimation"], 25)
        self.assertEqual(ep["obs_history"], DEP["hl_history_length"])
        self.assertEqual(ep["ll_obs_history"], DEP["ll_history_length"])

    def test_actuators(self):
        act = ENV["robot"]["actuators"]
        g = act["z1_gripper"]
        sim_idx = RetrievalSim.MUJOCO_JOINT_NAMES.index("jointGripper")
        self.assertTrue(g["class_type"].endswith("DCMotor"))
        self.assertEqual(DEP["gripper_mode"], "dc_motor")
        self.assertTrue(close(DEP["kps"][sim_idx], g["stiffness"]) and close(DEP["kds"][sim_idx], g["damping"]))
        self.assertTrue(close(DEP["gripper_torque_limit"], g["effort_limit"]))
        self.assertTrue(close(DEP["gripper_saturation_effort"], g["saturation_effort"]))
        self.assertTrue(close(DEP["gripper_velocity_limit"], g["velocity_limit"]))
        u = act["z1_arm_uan"]
        names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.assertTrue(close(DEP["uan_kp"], [u["kp"][n] for n in names]))
        self.assertTrue(close(DEP["uan_kd"], [u["kd"][n] for n in names]))
        self.assertTrue(close(DEP["uan_nominal_torque_limits"], [u["nominal_effort_limit"][n] for n in names]))
        self.assertTrue(close(DEP["uan_final_torque_limits"], [u["effort_limit"][n] for n in names]))
        for k in ("history_length", "pos_scale", "vel_scale", "action_scale"):
            self.assertTrue(close(DEP[f"uan_{k}"], u[k]), k)
        w = act["wheel"]
        self.assertTrue(close(w["damping"], 10.0) and close(w["effort_limit_sim"], 20.0) and close(w["velocity_limit_sim"], 50.0))
        # default joint positions (WBC zero offsets)
        js = ENV["robot"]["init_state"]["joint_pos"]
        import re
        for i, n in enumerate(RetrievalSim.MUJOCO_JOINT_NAMES):
            isaac_n = n.replace("wheel", "foot")
            vals = [v for pat, v in js.items() if re.fullmatch(pat, isaac_n)]
            self.assertEqual(len(vals), 1, n)
            self.assertTrue(close(DEP["default_joint_pos"][i], vals[0]), n)
        self.assertTrue(close(DEP["root_pos"], ENV["robot"]["init_state"]["pos"]))


class TestInterfaces(unittest.TestCase):
    def _isaac_adapter(self):
        p = os.path.join(ISAAC, "source/isaaclab_tasks/isaaclab_tasks/direct/b2wz1_hl_retrieval_DRC_UAN_ABS/modules/command_adapter.py")
        if not os.path.isfile(p):
            return None
        spec = importlib.util.spec_from_file_location("isaac_command_adapter", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_keypoints_match_isaac_command_adapter(self):
        mod = self._isaac_adapter()
        if mod is None:
            self.skipTest("Isaac command_adapter not available")
        import torch
        rng = np.random.default_rng(0)
        n = 2000
        kp0 = np.stack([rng.uniform(0.5, 0.9, n), rng.uniform(-0.3, 0.3, n), rng.uniform(0.0, 1.15, n)], -1).astype(np.float32)
        yaw = rng.uniform(-math.pi / 3, math.pi / 3, n)
        pitch = rng.uniform(math.radians(-70), 0.0, n)
        roll = np.full(n, math.pi / 2)
        ref = mod.build_keypoints_from_kp0_yaw_pitch_plb(
            torch.from_numpy(kp0), torch.from_numpy(yaw).float(), torch.from_numpy(pitch).float(),
            kp_dx=DEP["ee_kp_dx"], kp_dz=DEP["ee_kp_dz"], roll=torch.from_numpy(roll).float()).numpy()
        ours = np.stack([build_keypoints_from_kp0_yaw_pitch_plb(kp0[i], yaw[i], pitch[i], roll[i], DEP["ee_kp_dx"],
                                                                DEP["ee_kp_dz"], DEP["ee_pitch_to_euler_sign"]) for i in range(n)])
        err = float(np.abs(ref - ours).max())
        print(f"\nkeypoint encoding vs Isaac command_adapter: max |diff| = {err:.2e} over {n} commands")
        self.assertLess(err, 2e-6)

    def test_absolute_decoding_and_gripper(self):
        f = RetrievalSim._action_to_range
        for k, rng in (("kp0_x_range", [0.5, 0.9]), ("kp0_y_range", [-0.3, 0.3]), ("kp0_z_range", [0.0, 1.15]),
                       ("ee_yaw_range", [-math.pi / 3, math.pi / 3]), ("ee_pitch_range", [math.radians(-70), 0.0])):
            self.assertTrue(close(DEP[k], rng, 1e-6), k)
            self.assertTrue(close([f(-1.0, DEP[k]), f(0.0, DEP[k]), f(1.0, DEP[k])],
                                  [rng[0], 0.5 * (rng[0] + rng[1]), rng[1]], 1e-6))
        self.assertTrue(close(DEP["base_cmd_scale"], [0.5, 0.5, 0.5]))
        self.assertEqual(DEP["gripper_binary_threshold"], 0.0)
        self.assertTrue(close(DEP["gripper_close_pos"], 0.0) and close(DEP["gripper_open_pos"], -math.pi / 2))


if __name__ == "__main__":
    unittest.main()
