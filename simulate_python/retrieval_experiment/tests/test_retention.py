"""Physical-retention / One-Shot Success tests (retention.py), on synthetic 250-Hz signals and one engine episode.

    python3 -m unittest retrieval_experiment.tests.test_retention -v
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../..")))

from retrieval_experiment import retention as RT  # noqa: E402

EXP = yaml.safe_load(open(os.path.join(HERE, "..", "config.yaml")))
CFG = EXP["retention"]
DT = 0.004
SIDE = 0.05


class Sig:
    """Piecewise builder: each segment holds forces, ramps the jaw angle and the cube position linearly."""

    def __init__(self, q0=-90.0, rel0=(0.13, 0.0, 0.02), cmd=-1):
        self.fs, self.fm, self.q, self.rel, self.cmd = [], [], [], [], []
        self._q, self._rel, self._cmd = q0, np.array(rel0, float), cmd

    def seg(self, dur, fs=0.0, fm=0.0, q_to=None, drel=(0, 0, 0), cmd=None):
        n = int(round(dur / DT))
        q_to = self._q if q_to is None else q_to
        self._cmd = self._cmd if cmd is None else cmd
        for i in range(1, n + 1):
            u = i / n
            self.fs.append(fs if not callable(fs) else fs(i))
            self.fm.append(fm if not callable(fm) else fm(i))
            self.q.append(self._q + u * (q_to - self._q))
            self.rel.append(self._rel + u * np.asarray(drel, float))
            self.cmd.append(self._cmd)
        self._q, self._rel = q_to, self._rel + np.asarray(drel, float)
        return self

    def approach_and_grasp(self, hold=1.0):
        self.seg(0.3, q_to=-60.0, cmd=+1)                    # closing, no contact
        self.seg(0.1, fs=0.0, fm=0.0, q_to=-41.0)
        self.seg(hold, fs=450.0, fm=450.0, q_to=-41.0)        # settled force closure
        return self

    def ph(self):
        return {"ph_f_stator": np.array(self.fs), "ph_f_mover": np.array(self.fm),
                "ph_q": np.radians(np.array(self.q)), "ph_rel_pos": np.array(self.rel),
                "ph_cmd": np.array(self.cmd, dtype=np.int8), "ph_dt": DT}


def run(sig, success=True, settled=True):
    return RT.analyze_retention(sig.ph(), SIDE, success, CFG, settled)


class TestRetention(unittest.TestCase):
    # ---- the nine required cases -------------------------------------------------
    def test_1_success_without_loss(self):
        r = run(Sig().approach_and_grasp(2.0))
        self.assertTrue(r["acquired"] and r["one_shot_success"])
        self.assertEqual((r["releases"], r["reacquisitions"]), (0, 0))
        self.assertAlmostEqual(r["first_acquisition_s"], 0.3 + 0.1 + 0.1, places=6)   # end of 0.1-s settled window

    def test_2_success_after_contact_flicker(self):
        s = Sig().approach_and_grasp(0.5)
        s.seg(0.012, fs=0.0, fm=60.0, q_to=-40.0, drel=(0.002, 0, 0))      # 12 ms, 2 mm, 1 deg
        s.seg(0.5, fs=450.0, fm=450.0)
        r = run(s)
        self.assertEqual((r["releases"], r["flicker_gaps"]), (0, 1))
        self.assertTrue(r["one_shot_success"])

    def test_3_verified_drop_after_grasp(self):
        s = Sig().approach_and_grasp(0.5)
        s.seg(0.3, q_to=-20.0, drel=(0.05, 0, -0.08))                        # cube falls out, jaw closes
        r = run(s, success=False)
        self.assertEqual(r["releases"], 1)
        self.assertFalse(r["one_shot_success"])

    def test_4_release_within_grasp_region_with_single_jaw_contact_then_regrasp(self):
        s = Sig().approach_and_grasp(0.5)
        # grip lost: one jaw still touches the cube with 5 N (above the 0.3-N sensor threshold),
        # cube slides 30 mm but stays within 0.10 m of the gripper centre
        s.seg(0.3, fs=5.0, fm=0.0, q_to=-35.0, drel=(0.03, 0, 0))
        s.seg(0.4, q_to=-80.0, cmd=-1)                                       # open
        s.seg(0.3, q_to=-45.0, cmd=+1)                                       # close again
        s.seg(1.0, fs=450.0, fm=450.0)                                       # re-acquired
        r = run(s, success=True)
        self.assertEqual((r["releases"], r["reacquisitions"]), (1, 1))
        self.assertTrue(r["retained_at_end"])
        self.assertFalse(r["one_shot_success"])

    def test_5_grasped_never_delivered(self):
        r = run(Sig().approach_and_grasp(3.0), success=False)
        self.assertTrue(r["acquired"] and r["retained_at_end"])
        self.assertFalse(r["one_shot_success"])

    def test_6_target_reached_without_acquisition(self):
        s = Sig().seg(0.3, q_to=-60.0, cmd=+1).seg(2.0, fs=50.0, fm=0.0, q_to=-30.0)   # single-jaw push only
        r = run(s, success=True)
        self.assertFalse(r["acquired"] or r["one_shot_success"])

    def test_7_robot_fall(self):
        r = run(Sig().approach_and_grasp(1.0), success=False)               # base-contact termination
        self.assertFalse(r["one_shot_success"])

    def test_8_timeout_without_grasp(self):
        r = run(Sig().seg(14.9, q_to=-90.0), success=False)
        self.assertFalse(r["acquired"] or r["one_shot_success"])

    def test_9_early_failure_before_grasp(self):
        r = run(Sig().seg(0.3, q_to=-80.0, cmd=+1), success=False)
        self.assertFalse(r["acquired"] or r["one_shot_success"])

    # ---- release criteria individually -------------------------------------------
    def test_release_by_duration(self):
        s = Sig().approach_and_grasp(0.5).seg(0.044, q_to=-40.5).seg(1.0, fs=450.0, fm=450.0)
        r = run(s)
        self.assertEqual(r["releases"], 1)
        self.assertFalse(r["one_shot_success"])

    def test_release_by_displacement(self):
        s = Sig().approach_and_grasp(0.5).seg(0.016, drel=(0.026, 0, 0)).seg(1.0, fs=450.0, fm=450.0)
        self.assertEqual(run(s)["releases"], 1)

    def test_release_by_jaw_closing(self):
        s = Sig().approach_and_grasp(0.5).seg(0.016, q_to=-35.5).seg(1.0, fs=450.0, fm=450.0, q_to=-35.5)
        self.assertEqual(run(s)["releases"], 1)

    def test_weak_contact_chatter_is_not_retention(self):
        # forces chatter around 1-2 N (0.3-N sensor flag toggles) for 64 ms while the cube slides 15 mm
        s = Sig().approach_and_grasp(0.5)
        s.seg(0.064, fs=lambda i: 1.5 if i % 2 else 0.1, fm=lambda i: 1.2 if i % 3 else 0.0, q_to=-36.0,
              drel=(0.015, 0, 0))
        s.seg(1.0, fs=300.0, fm=300.0)
        r = run(s)
        self.assertEqual(r["releases"], 1)
        self.assertFalse(r["one_shot_success"])

    # ---- acquisition ----------------------------------------------------------------
    def test_squeeze_out_pinch_is_an_attempt_not_an_acquisition(self):
        s = Sig().seg(0.3, q_to=-60.0, cmd=+1)
        s.seg(0.3, fs=200.0, fm=200.0, q_to=-50.0, drel=(0.015, 0, 0))     # jaw keeps closing at 33 deg/s
        s.seg(0.2, q_to=0.0)                                                # cube squeezed out, jaw closes fully
        s.seg(0.5, q_to=-90.0, cmd=-1).seg(0.3, q_to=-45.0, cmd=+1)          # reopen, retry
        s.seg(1.5, fs=450.0, fm=450.0)
        r = run(s)
        self.assertTrue(r["one_shot_success"])
        self.assertEqual(r["attempts_before_acquisition"], 1)
        self.assertFalse(run(s, settled=False)["one_shot_success"])          # strict variant counts it

    def test_command_change_while_holding(self):
        s = Sig().approach_and_grasp(0.5)
        s.seg(0.1, fs=450.0, fm=450.0, cmd=-1).seg(0.6, fs=450.0, fm=450.0, cmd=+1)   # open command, jaw stays
        r = run(s)
        self.assertEqual((r["close_to_open_cmd_while_held"], r["close_to_open_cmd_while_held_harmless"]), (1, 1))
        self.assertTrue(r["one_shot_success"])

    def test_invariants_random(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            s = Sig().approach_and_grasp(float(rng.uniform(0.1, 1.0)))
            for _ in range(int(rng.integers(0, 4))):
                d = float(rng.choice([0.008, 0.02, 0.06, 0.3]))
                s.seg(d, fs=float(rng.choice([0.0, 5.0])), q_to=s._q + float(rng.uniform(0, 8)),
                      drel=(float(rng.uniform(0, 0.03)), 0, 0))
                s.seg(float(rng.uniform(0.1, 0.6)), fs=450.0, fm=450.0)
            ok = bool(rng.random() < 0.7)
            r = run(s, success=ok)
            self.assertLessEqual(int(r["one_shot_success"]), int(ok))
            self.assertLessEqual(r["reacquisitions"], r["releases"])
            self.assertLessEqual(int(r["one_shot_success"]), int(r["releases"] == 0))


class TestEngineEpisode(unittest.TestCase):
    def test_logged_episode(self):
        import torch
        from retrieval_experiment import scenarios as scen
        from retrieval_experiment.simulator import RetrievalSim, resolve_repo_path, run_episode
        torch.set_num_threads(1)
        dep = yaml.safe_load(open(resolve_repo_path(EXP["deploy_config"])))
        sim = RetrievalSim(dep, EXP)
        sc = scen.generate(1, EXP["master_seed"], EXP["distributions"], sim.robot_shape_names)[0]
        summary, tr = run_episode(sim, sc, 0.05, 800.0)
        self.assertEqual(len(tr["ph_q"]), 25 * summary["hl_steps"])          # one sample per physics step
        r = RT.from_npz(tr, summary["success"], CFG)
        self.assertTrue(summary["success"] and r["acquired"])
        # physical acquisition precedes (or coincides with) the training confirmation
        self.assertLessEqual(r["first_acquisition_s"], summary["time_to_grasp_s"] + 0.25)
        self.assertTrue(r["one_shot_success"])


if __name__ == "__main__":
    unittest.main()
