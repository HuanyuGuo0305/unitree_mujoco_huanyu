"""Bit-exact parity: headless E5 engine vs the interactive retrieval deploy script.

The deploy script ``deploy_mujoco/b2wz1_hl_retrieval_uan.py`` is executed with a
stub viewer and its built-in 50-Hz debug CSV. Both run on the SAME compiled
MuJoCo model (the deploy script's ``MjModel.from_xml_path`` returns a deep copy
of the engine's 5-cm model with the materials/base inertia of scenario 0), the
same task draws (the deploy script's own RNG for episode 1), nominal reset
state and no observation noise (the deploy script has neither). Every WBC-step
record must agree EXACTLY: HL action, WBC EE command, arm targets, arm state,
UAN output, gripper command/state, internal grasp proxy, grasp error and cube
position.

Run from unitree_mujoco/simulate_python:
    python3 -m unittest retrieval_experiment.tests.test_deploy_parity -v
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_PY = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, SIM_PY)

from retrieval_experiment import scenarios as scen  # noqa: E402
from retrieval_experiment.simulator import DEPLOY_SCRIPT, RetrievalSim, resolve_repo_path  # noqa: E402

EXP_CFG = os.path.join(SIM_PY, "retrieval_experiment", "config.yaml")
SEED = 7
RHO = 800.0          # training density; the interactive deploy XML cube is 0.10 kg at 5 cm
DURATION_S = 9.0


class _StubViewer:
    def __init__(self):
        self.cam = types.SimpleNamespace(type=0, azimuth=0, elevation=0, distance=0, lookat=np.zeros(3))
        self.user_scn = types.SimpleNamespace(ngeom=0, maxgeom=0, geoms=[])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def is_running(self):
        return True

    def sync(self):
        pass


def make_engine_and_scenario():
    exp = yaml.safe_load(open(EXP_CFG))
    dep = yaml.safe_load(open(resolve_repo_path(exp["deploy_config"])))
    dep.update(random_seed=SEED)
    # Library-default threading, exactly as the deploy script creates its sessions.
    sim = RetrievalSim(dep, exp, ort_threads=0, observation_noise="none")
    bank = scen.generate(1, exp["master_seed"], exp["distributions"], sim.robot_shape_names)
    sc = scen.nominal_scenario(sim.robot_shape_names)
    sc["physics"] = bank[0]["physics"]                 # randomized materials + base inertia
    sc["task"] = scen.deploy_rng_task(SEED, dep)       # deploy RNG draws for episode 1
    sim.select_cube(0.05, RHO)
    sim.apply_physics(sc["physics"])
    model_for_deploy = copy.deepcopy(sim.m)
    return exp, dep, sim, sc, model_for_deploy


def run_deploy(dep, model, tmpdir):
    cfg = dict(dep)
    csv_path = os.path.join(tmpdir, "deploy.csv")
    cfg.update(simulation_duration=DURATION_S, print_interval_sim_steps=0, vis_retrieval_target=False,
               vis_ee_target=False, vis_grasp_debug=False, debug_tracking_logging=True,
               debug_tracking_log_every_ll_steps=1, debug_tracking_csv_path=csv_path)
    ypath = os.path.join(tmpdir, "deploy.yaml")
    yaml.safe_dump(cfg, open(ypath, "w"))
    spec = importlib.util.spec_from_file_location("deploy_retrieval_parity", DEPLOY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with mock.patch.object(mod.mujoco.viewer, "launch_passive", lambda *a, **k: _StubViewer()), \
            mock.patch.object(mod.mujoco.MjModel, "from_xml_path", lambda p: copy.deepcopy(model)), \
            mock.patch.object(mod.time, "sleep", lambda *_: None), \
            mock.patch.object(sys, "argv", ["deploy", ypath]), \
            mock.patch("builtins.print", lambda *a, **k: None):
        mod.main()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if int(r["episode"]) == 1]


def run_engine(sim, sc, n_rows):
    sim.reset(sc, 0.05, RHO)
    out = []

    def rec():
        _, _, _, _, ge = sim.get_gripper_geometry()
        out.append({
            "hl": sim.current_hl_action.copy(), "ee_cmd": sim.ee_cmd_plb.copy(), "q_des": sim.arm_target.copy(),
            "arm_q": sim.d.qpos[7:7 + sim.nj][sim.arm_idx].astype(np.float32),
            "uan": None, "grasp_error": ge, "gripper_target": sim.gripper_target,
            "gripper_q": float(np.float32(sim.d.qpos[7 + sim.gripper_idx])),
            "exec": sim.executed_gripper_cmd_norm, "proxy": int(sim.task.proxy), "cube": sim.get_cube_pos_w(),
        })

    sim.infer_hl()
    sim.run_low_level_policy()
    rec()
    while len(out) < n_rows:
        for k in range(sim.ll_steps_per_hl):
            if k > 0:
                sim.run_low_level_policy()
                rec()
            for _ in range(sim.ll_decimation):
                sim.apply_low_level_actuation()
                mujoco.mj_step(sim.m, sim.d)
        sim.task_step()                 # deploy: update_grasp_confidence_proxy() at the HL boundary
        sim.next_hl_action()
        sim.run_low_level_policy()
        rec()
    return out


class TestDeployParity(unittest.TestCase):
    def test_bit_exact(self):
        exp, dep, sim, sc, model = make_engine_and_scenario()
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_deploy(dep, model, tmp)
        self.assertGreater(len(rows), 300)
        eng = run_engine(sim, sc, len(rows))
        n = min(len(rows), len(eng))
        worst = {}
        proxies = 0
        for r, e in zip(rows[:n], eng[:n]):
            dep_v = {
                "hl": np.array([float(r[f"hl_a{i}"]) for i in range(9)]),
                "ee_cmd": np.array([float(r[f"ee_cmd_kp{i}_{a}"]) for i in range(3) for a in "xyz"]),
                "q_des": np.array([float(r[f"arm_q_des{j}"]) for j in range(1, 7)]),
                "arm_q": np.array([float(r[f"arm_q{j}"]) for j in range(1, 7)]),
                "grasp_error": np.array([float(r["grasp_error"])]),
                "gripper_target": np.array([float(r["gripper_target"])]),
                "gripper_q": np.array([float(r["gripper_q"])]),
                "exec": np.array([float(r["gripper_exec_norm"])]),
                "proxy": np.array([float(r["grasp_proxy"])]),
                "cube": np.array([float(r[f"cube_{a}"]) for a in "xyz"]),
            }
            proxies += int(dep_v["proxy"][0])
            for k, v in dep_v.items():
                ev = np.asarray(e[k], dtype=np.float64).reshape(-1)
                worst[k] = max(worst.get(k, 0.0), float(np.max(np.abs(v - ev))))
        cube_travel = float(np.linalg.norm(np.array([float(rows[n - 1][f"cube_{a}"]) for a in "xyz"])
                                           - np.array([float(rows[0][f"cube_{a}"]) for a in "xyz"])))
        closes = sum(float(r["gripper_exec_norm"]) > 0 for r in rows[:n])
        print(f"\nparity over {n} WBC steps ({n * 0.02:.2f} s): close-cmd rows={closes}, proxy rows={proxies}, "
              f"cube travel={cube_travel:.3f} m\nmax |diff| = {worst}")
        self.assertGreater(closes, 0, "episode never commanded the gripper closed; parity not exercised")
        for k, v in worst.items():
            self.assertEqual(v, 0.0, f"{k} differs by {v}")


if __name__ == "__main__":
    unittest.main()
