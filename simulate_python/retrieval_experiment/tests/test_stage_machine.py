"""Task-state validation for E5.

1. Hand-written sequences: S0->S1, S1->S0, proxy entry/exit, S1->S2 from physical
   dual contact, grasp loss, 3-step retrieval success, no success from proximity
   alone, timeout at 149 HL steps, grasp-error OOB termination, completion time.
2. Cross-implementation check: the Isaac Lab methods ``_update_grasp_stage``,
   ``_update_curriculum_stage``, ``_update_retrieval_success`` and ``_get_dones``
   are extracted from the TRAINING SOURCE FILE and executed (torch, vectorized)
   on random input sequences; the per-step stage / proxy / success / termination
   of ``TaskStateMachine`` must match exactly. Skipped if the Isaac Lab source or
   torch is unavailable.

    python3 -m unittest retrieval_experiment.tests.test_stage_machine -v
"""
from __future__ import annotations

import ast
import math
import os
import sys
import types
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../..")))

from retrieval_experiment.metrics import StepInputs, TaskStateMachine, classify_termination, wilson_ci  # noqa: E402

EXP = yaml.safe_load(open(os.path.join(HERE, "..", "config.yaml")))
OPEN = -math.pi / 2


def tsm():
    return TaskStateMachine(EXP["task"], OPEN, 150)


def inp(err=1.0, q=OPEN, close=False, st=False, mv=False, rerr=1.0, base=False):
    return StepInputs(grasp_error=err, gripper_q=q, close_executed=close, stator_contact=st, mover_contact=mv,
                      retrieval_error=rerr, base_contact=base)


GRASP = dict(err=0.05, q=-0.6, close=True, st=True, mv=True)


class TestSequences(unittest.TestCase):
    def test_s0_to_s1_needs_three_consecutive(self):
        t = tsm()
        for e in (0.1, 0.1, 0.3, 0.1, 0.1):
            ev = t.update(inp(err=e))
            self.assertEqual(ev["stage"], 0)
        ev = t.update(inp(err=0.19))
        self.assertTrue(ev["s0_to_s1"])
        self.assertEqual(ev["stage"], 1)

    def test_s1_to_s0_hysteresis(self):
        t = tsm()
        for _ in range(3):
            t.update(inp(err=0.1))
        self.assertEqual(t.stage, 1)
        for e in (0.22, 0.24, 0.26, 0.26):        # 0.20-0.25 band keeps S1
            ev = t.update(inp(err=e))
            self.assertEqual(ev["stage"], 1)
        ev = t.update(inp(err=0.3))
        self.assertTrue(ev["s1_to_s0"])
        self.assertEqual(ev["stage"], 0)

    def test_proxy_requirements(self):
        base = dict(err=0.05, q=-0.6, close=True)
        for bad in (dict(close=False), dict(err=0.11), dict(q=-0.05)):   # -0.05 rad > -5 deg: fully closed
            t = tsm()
            t.prev_gripper_q = -0.6
            for _ in range(5):
                ev = t.update(inp(**{**base, **bad}))
            self.assertFalse(ev["proxy"], bad)
        # moving gripper (> 3 deg per HL step) is not "holding"
        t = tsm()
        q = -1.2
        for _ in range(5):
            q += 0.06
            ev = t.update(inp(**{**base, "q": q}))
        self.assertFalse(ev["proxy"])
        # three consecutive candidates -> proxy (first step also moves from open: delta > 3 deg)
        t = tsm()
        out = [t.update(inp(**base))["proxy"] for _ in range(4)]
        self.assertEqual(out, [False, False, False, True])

    def test_closure_alone_is_not_a_grasp(self):
        t = tsm()
        for _ in range(10):
            ev = t.update(inp(err=0.05, q=-0.6, close=True, st=False, mv=False))
        self.assertTrue(ev["proxy"])
        self.assertEqual(ev["stage"], 1)
        self.assertFalse(ev["stage2_now"])
        # single-sided contact is not enough either
        for _ in range(5):
            ev = t.update(inp(err=0.05, q=-0.6, close=True, st=True, mv=False))
        self.assertEqual(ev["stage"], 1)

    def test_contact_without_proxy_is_not_a_grasp(self):
        t = tsm()
        for _ in range(10):
            ev = t.update(inp(err=0.05, q=-0.6, close=False, st=True, mv=True))
        self.assertEqual(ev["stage"], 1)

    def test_grasp_confirmation_and_loss(self):
        t = tsm()
        stages = [t.update(inp(**GRASP))["stage"] for _ in range(4)]
        self.assertEqual(stages, [0, 0, 1, 2])   # S0->S1 at 3, proxy at 4 (first step moved from open)
        ev = t.update(inp(**{**GRASP, "st": False}))          # one lost step tolerated
        self.assertFalse(ev["terminated"])
        ev = t.update(inp(**GRASP))
        self.assertEqual(ev["stage"], 2)
        t.update(inp(**{**GRASP, "mv": False}))
        ev = t.update(inp(**{**GRASP, "mv": False}))
        self.assertTrue(ev["s2_lost_terminated"] and ev["terminated"])
        self.assertEqual(classify_termination(ev, True), "grasp_lost_after_confirmation")

    def test_success_requires_stage2_and_three_steps(self):
        t = tsm()
        for _ in range(20):                                    # proximity alone, stage 0/1
            ev = t.update(inp(err=0.05, rerr=0.01))
        self.assertFalse(ev["success"])
        t = tsm()
        for _ in range(4):
            t.update(inp(**GRASP))
        self.assertEqual(t.stage, 2)
        seq = [0.1, 0.1, 0.2, 0.1, 0.1, 0.1]
        flags = [t.update(inp(**GRASP, rerr=r))["newly_successful"] for r in seq]
        self.assertEqual(flags, [False, False, False, False, False, True])
        self.assertEqual(t.hl_steps, 10)                        # completion time = 10 * 0.1 s

    def test_timeout_and_oob(self):
        t = tsm()
        for i in range(1, 150):
            ev = t.update(inp())
            self.assertEqual(ev["timeout"], i >= 149)
        self.assertEqual(classify_termination(ev, False), "grasp_not_achieved")
        self.assertEqual(classify_termination(ev, True), "timeout")
        t = tsm()
        out = [t.update(inp(err=6.0))["oob_terminated"] for _ in range(3)]
        self.assertEqual(out, [False, False, True])

    def test_wilson(self):
        lo, hi = wilson_ci(180, 200)
        self.assertAlmostEqual(lo, 0.8506, places=3)
        self.assertAlmostEqual(hi, 0.9343, places=3)
        self.assertEqual(wilson_ci(0, 10)[0], 0.0)


# ---------------------------------------------------------------------------
# Cross-check against the Isaac Lab training source
# ---------------------------------------------------------------------------
def _isaac_methods():
    try:
        import torch  # noqa: F401
    except Exception:
        return None
    src_path = os.path.join(EXP["isaac_reference"]["root"], EXP["isaac_reference"]["env_source"])
    if not os.path.isfile(src_path):
        return None
    src = open(src_path).read()
    tree = ast.parse(src)
    params = None
    wanted = {"_update_grasp_stage", "_update_curriculum_stage", "_update_retrieval_success", "_get_dones"}
    funcs = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ENV_PARAMS":
            params = eval(compile(ast.Expression(node.value), src_path, "eval"), {"math": math})
        if isinstance(node, ast.ClassDef) and node.name == "B2WZ1HLRetrievalUANABSEnv":
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name in wanted:
                    f.returns = None
                    for a in f.args.args:
                        a.annotation = None
                    mod = ast.Module(body=[f], type_ignores=[])
                    ns = {"torch": __import__("torch")}
                    exec(compile(ast.fix_missing_locations(mod), src_path, "exec"), ns)
                    funcs[f.name] = ns[f.name]
    if params is None or set(funcs) != wanted:
        return None
    return params, funcs


class TestAgainstIsaacSource(unittest.TestCase):
    def test_random_sequences_match_training_code(self):
        got = _isaac_methods()
        if got is None:
            self.skipTest("Isaac Lab source / torch not available")
        import torch
        params, f = got
        self.assertFalse(params["stage2_force_gripper_close_enabled"] is None)
        N, T = 2048, 149
        g = torch.Generator().manual_seed(0)
        env = types.SimpleNamespace()
        env.cfg = types.SimpleNamespace(env_params=params)
        env.num_envs = N
        Z = lambda dt=torch.long: torch.zeros(N, dtype=dt)  # noqa: E731
        for name in ("_grasp_proxy_enter_count", "_grasp_proxy_exit_count", "_curriculum_stage",
                     "_curriculum_s0_to_s1_count", "_curriculum_s1_to_s0_count", "_curriculum_s2_grasp_lost_count",
                     "_retrieval_success_count", "_grasp_error_oob_steps", "episode_length_buf"):
            setattr(env, name, Z())
        for name in ("_grasp_confidence_proxy", "_grasp_proxy_candidate", "_stage2", "_grasp_true_candidate",
                     "_curriculum_s2_grasp_lost_terminated", "_curriculum_s0_to_s1_event", "_curriculum_s1_to_s0_event",
                     "_curriculum_s1_to_s2_event", "_episode_retrieval_success", "_grasp_error_oob_terminated",
                     "_gripper_stator_cube_contact", "_gripper_mover_cube_contact", "_base_contact",
                     "_gripper_not_fully_closed", "_gripper_angle_holding"):
            setattr(env, name, Z(torch.bool))
        env._gripper_angle_delta = Z(torch.float32)
        env._prev_gripper_joint_pos = torch.full((N,), float(params["gripper_open_pos"]))
        env._gripper_cmd_norm = torch.full((N, 1), -1.0)
        env._gripper_cmd_pos = torch.zeros(N, 1)
        env._grasp_error = Z(torch.float32)
        env._object_pos_w = torch.zeros(N, 3)
        env._retrieval_target_pos_w = torch.zeros(N, 3)
        env.max_episode_length = 150
        env.gripper_ids = [0]
        env._robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=torch.zeros(N, 1)))
        env._stage2_force_gripper_close_active = lambda: False
        m = types.MethodType
        upd = [m(f["_update_grasp_stage"], env), m(f["_update_curriculum_stage"], env),
               m(f["_update_retrieval_success"], env)]
        dones = m(f["_get_dones"], env)

        sms = [TaskStateMachine(EXP["task"], float(params["gripper_open_pos"]), 150) for _ in range(N)]
        # Piecewise-constant random regimes so that every transition is exercised.
        def regime(shape, lo, hi):
            v = torch.rand(shape, generator=g) * (hi - lo) + lo
            hold = torch.randint(1, 6, shape, generator=g)
            return v, hold
        err_v = torch.rand(N, generator=g) * 0.4
        q_v = -torch.rand(N, generator=g) * 1.6
        mismatches = 0
        n_s2 = n_succ = n_lost = 0
        for t in range(T):
            jump = torch.rand(N, generator=g) < 0.3
            err_v = torch.where(jump, torch.rand(N, generator=g) * 0.4 * torch.where(torch.rand(N, generator=g) < 0.02, 20.0, 1.0), err_v)
            q_v = torch.where(torch.rand(N, generator=g) < 0.3, -torch.rand(N, generator=g) * 1.6, q_v + 0.01 * torch.randn(N, generator=g))
            close = torch.rand(N, generator=g) < 0.7
            st = torch.rand(N, generator=g) < 0.8
            mv = torch.rand(N, generator=g) < 0.8
            rerr = torch.rand(N, generator=g) * 0.3
            base = torch.rand(N, generator=g) < 0.002
            env._grasp_error[:] = err_v
            env._robot.data.joint_pos[:, 0] = q_v
            env._gripper_cmd_norm[:, 0] = torch.where(close, 1.0, -1.0)
            env._gripper_stator_cube_contact[:] = st
            env._gripper_mover_cube_contact[:] = mv
            env._base_contact[:] = base
            env._object_pos_w[:, 0] = rerr
            env.episode_length_buf += 1
            for u in upd:
                u()
            term, tout = dones()
            for i in range(N):
                ev = sms[i].update(StepInputs(float(err_v[i]), float(q_v[i]), bool(close[i]), bool(st[i]), bool(mv[i]),
                                              float(rerr[i]), bool(base[i])))
                ok = (ev["stage"] == int(env._curriculum_stage[i]) and ev["proxy"] == bool(env._grasp_confidence_proxy[i])
                      and ev["stage2_now"] == bool(env._stage2[i]) and ev["success"] == bool(env._episode_retrieval_success[i])
                      and ev["terminated"] == bool(term[i]) and ev["timeout"] == bool(tout[i]))
                mismatches += int(not ok)
            n_s2 += int((env._curriculum_stage == 2).sum())
            n_succ += int(env._episode_retrieval_success.sum())
            n_lost += int(env._curriculum_s2_grasp_lost_terminated.sum())
        print(f"\nIsaac-source cross-check: {N} envs x {T} steps, stage-2 env-steps={n_s2}, "
              f"success env-steps={n_succ}, grasp-lost flags={n_lost}, mismatches={mismatches}")
        self.assertGreater(n_s2, 1000)
        self.assertGreater(n_succ, 1000)
        self.assertGreater(n_lost, 100)
        self.assertEqual(mismatches, 0)


if __name__ == "__main__":
    unittest.main()
