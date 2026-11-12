"""E2 tests: the official single dynamic-condition E2 protocol."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for root in (REPO_ROOT, SIM_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

from benchmark.scenarios import experiments as E  # noqa: E402


class TestE2Dynamic(unittest.TestCase):
    """Official E2: one dynamic condition, 100 paired seeds."""

    def setUp(self):
        self.cfg, _ = E.load_config("E2")

    def test_protocol(self):
        self.assertEqual(E.all_conditions("E2", self.cfg), ["dynamic"])
        self.assertEqual(self.cfg["baselines"], ["C1", "C2"])
        self.assertEqual(self.cfg["seeds"], {"start": 0, "count": 100})
        self.assertEqual((self.cfg["duration_s"], self.cfg["settle_s"]), (30.0, 3.0))
        with self.assertRaises(KeyError):
            E.build_e2(self.cfg, "T4", 0)

    def test_deterministic_and_seed_only(self):
        for seed in (0, 13, 99):
            a = E.build_scenario("E2", self.cfg, "dynamic", seed).arrays()
            b = E.build_scenario("E2", self.cfg, "dynamic", seed).arrays()
            for key in a:
                self.assertTrue(np.array_equal(a[key], b[key]), key)

    def test_base_commands_within_training_range_and_2_to_3_s(self):
        lim = np.asarray(self.cfg["base_command_limits"], float)
        for seed in range(100):
            s = E.build_e2(self.cfg, "dynamic", seed)
            self.assertTrue(np.all(np.abs(s.base.commands) <= lim + 1e-6))
            d = s.base.segment_durations_s[:-1]           # last segment truncated by episode end
            self.assertTrue(np.all((d >= 2.0 - 1e-9) & (d <= 3.0 + 1e-9)))
            # piecewise constant: changes only at switch ticks
            ch = np.nonzero(np.any(np.diff(s.base.commands, axis=0) != 0, axis=1))[0] + 1
            self.assertTrue(set(ch.tolist()) <= set(s.base.switch_ticks.tolist()))

    def test_ee_full_pose_rows_held_3_to_5_s(self):
        table = E.keypoint_table(self.cfg["_global"])
        for seed in range(100):
            s = E.build_e2(self.cfg, "dynamic", seed)
            holds = np.diff(np.r_[0, s.ee.switch_ticks]) * 0.02
            self.assertTrue(np.all((holds >= 3.0 - 1e-9) & (holds <= 5.0 + 1e-9)))
            for k in np.r_[0, s.ee.switch_ticks]:
                row = s.ee.row_indices[list(np.r_[0, s.ee.switch_ticks]).index(k)]
                np.testing.assert_array_equal(s.ee.commands[k], table.table[row])
            ch = np.nonzero(np.any(np.diff(s.ee.commands, axis=0) != 0, axis=1))[0] + 1
            self.assertTrue(set(ch.tolist()) <= set(s.ee.switch_ticks.tolist()))

    def test_base_and_ee_streams_independent(self):
        e1, _ = E.load_config("E1")
        a = E.build_e2(self.cfg, "dynamic", 5)
        b = E.build_e1(e1, "T4", 5)
        np.testing.assert_array_equal(a.base.commands, b.base.commands)   # E1 base generator reused
        self.assertFalse(np.array_equal(a.ee.switch_ticks, b.ee.switch_ticks))


if __name__ == "__main__":
    unittest.main()
