"""Bit-exact parity: headless TransportSim (Full policy) vs the interactive deploy script.

The deploy script is executed unmodified with a stub viewer and its built-in
debug CSV. The engine is given the scenario the deploy script's runtime RNG
produces for episode 1 (spawn offset, base/EE schedules), nominal mass/friction
and no reset randomization. The puck trajectory in the disk frame, the
executed 6-D residual and the WBC EE command must agree exactly.

Run from unitree_mujoco/simulate_python:
    python3 -m unittest transport_experiment.tests.test_deploy_parity -v
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_PY = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, SIM_PY)

from transport_experiment.controllers import HLMethod  # noqa: E402
from transport_experiment.scenarios import deploy_rng_scenario  # noqa: E402
from transport_experiment.sim import TransportSim, resolve_repo_path  # noqa: E402

DEPLOY_SCRIPT = os.path.join(SIM_PY, "deploy_mujoco/b2wz1_hl_transport_uan.py")
DEPLOY_YAML = os.path.join(SIM_PY, "configs/b2wz1_hl_transport_uan.yaml")
SEED = 3
DURATION_S = 6.0


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


def run_deploy(tmpdir):
    cfg = yaml.safe_load(open(DEPLOY_YAML))
    csv_path = os.path.join(tmpdir, "deploy.csv")
    cfg.update(random_seed=SEED, simulation_duration=DURATION_S, print_interval_sim_steps=0,
               vis_task_ee_reference=False, vis_wbc_ee_command=False,
               debug_tracking_logging=True, debug_tracking_log_every_ll_steps=1,
               debug_tracking_csv_path=csv_path)
    ypath = os.path.join(tmpdir, "deploy.yaml")
    yaml.safe_dump(cfg, open(ypath, "w"))
    spec = importlib.util.spec_from_file_location("deploy_transport", DEPLOY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with mock.patch.object(mod.mujoco.viewer, "launch_passive", lambda *a, **k: _StubViewer()), \
            mock.patch.object(mod.time, "sleep", lambda *_: None), \
            mock.patch.object(sys, "argv", ["deploy", ypath]), \
            mock.patch("builtins.print", lambda *a, **k: None):
        mod.main()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return cfg, [r for r in rows if int(r["episode"]) == 1]


def run_engine(cfg, n_ll_rows):
    # Same threading as the deploy script (library defaults) so that floating-point
    # reduction order inside onnxruntime / torch is identical.
    sim = TransportSim(cfg, ort_threads=0)
    meth = HLMethod("full", cfg["high_policy_path"], ort_threads=0)
    sc = deploy_rng_scenario(SEED, cfg, 150, float(cfg["puck_mass"]), float(cfg["puck_friction"]))
    sim.reset(sc)
    meth.reset(sim)
    out = []  # one record per LL boundary, AFTER the boundary's inference (deploy CSV semantics)

    def rec():
        out.append({"puck": sim.get_puck_pos_disk().copy(), "exec": meth.executed.copy(),
                    "kp0_cmd": sim.kp0_cmd.copy(), "kp0_ref": sim.ee_kp0_ref.copy(),
                    "base_ref": sim.base_velocity_ref.copy()})

    meth.step(sim)
    sim.run_low_level_policy()
    rec()                                             # t = 0 row
    while len(out) < n_ll_rows:
        for k in range(sim.ll_steps_per_hl):
            if k > 0:
                sim.run_low_level_policy()
                rec()
            for _ in range(sim.ll_decimation):
                sim.apply_low_level_actuation()
                import mujoco
                mujoco.mj_step(sim.m, sim.d)
        sim.hl_steps_completed += 1
        chk = sim.post_step_checks()
        if chk["puck_drop"] or chk["base_contact"] or chk["timeout"]:
            break
        sim.advance_references()
        meth.step(sim)
        sim.run_low_level_policy()
        rec()
    return out


class TestDeployParity(unittest.TestCase):
    def test_bit_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, rows = run_deploy(tmp)
        self.assertGreater(len(rows), 100)
        eng = run_engine(cfg, len(rows))
        n = min(len(rows), len(eng))
        self.assertGreater(n, 100)
        worst = {"puck": 0.0, "exec": 0.0, "kp0_cmd": 0.0, "kp0_ref": 0.0, "base_ref": 0.0}
        for r, e in zip(rows[:n], eng[:n]):
            dep = {
                "puck": np.array([float(r[f"puck_{a}_D"]) for a in "xyz"]),
                "exec": np.array([float(r[f"hl_exec_{i}"]) for i in range(6)]),
                "kp0_cmd": np.array([float(r[f"kp0_cmd_{a}"]) for a in "xyz"]),
                "kp0_ref": np.array([float(r[f"kp0_ref_{a}"]) for a in "xyz"]),
                "base_ref": np.array([float(r[f"base_ref_{a}"]) for a in ("vx", "vy", "wz")]),
            }
            for k in worst:
                worst[k] = max(worst[k], float(np.max(np.abs(dep[k] - e[k].astype(np.float64)))))
        print(f"\nparity over {n} WBC steps ({n * 0.02:.2f} s): max |diff| = {worst}")
        for k, v in worst.items():
            self.assertEqual(v, 0.0, f"{k} differs by {v}")


if __name__ == "__main__":
    unittest.main()
