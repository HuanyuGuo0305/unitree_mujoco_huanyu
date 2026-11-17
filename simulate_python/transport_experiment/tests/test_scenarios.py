"""Scenario generation: reproducibility, pairing and distribution bounds.

    python3 -m unittest transport_experiment.tests.test_scenarios -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../..")))

from transport_experiment import scenarios as scen  # noqa: E402

EXP = yaml.safe_load(open(os.path.join(HERE, "../config.yaml")))
D = EXP["distributions"]


class TestScenarios(unittest.TestCase):
    def test_reproducible_and_prefix_stable(self):
        a = scen.generate(10, 7, D, 150)
        b = scen.generate(10, 7, D, 150)
        self.assertEqual(json.dumps(a), json.dumps(b))
        c = scen.generate(4, 7, D, 150)
        self.assertEqual(json.dumps(a[:4]), json.dumps(c))       # scenario k independent of n
        self.assertNotEqual(json.dumps(scen.generate(1, 8, D, 150)), json.dumps(a[:1]))

    def test_bounds_and_coverage(self):
        scs = scen.generate(300, 1, D, 150)
        mass = np.array([s["physics"]["puck_mass"] for s in scs])
        mus = np.array([s["physics"]["puck_mu_static"] for s in scs])
        mud = np.array([s["physics"]["puck_mu_dynamic"] for s in scs])
        self.assertTrue(np.all((mass >= 0.5) & (mass <= 1.5)))
        self.assertTrue(np.all((mus >= 0.3) & (mus <= 1.0)) and np.all(mud <= mus) and np.all(mud >= 0.3))
        for s in scs:
            self.assertEqual(s["physics"]["puck_mu"], s["physics"]["puck_mu_" + D["puck_friction_source"]])
            self.assertTrue(np.all(np.abs(s["initial_state"]["puck_xy"]) <= 0.01))
            self.assertTrue(abs(s["physics"]["base_mass_add"]) <= 2.0)
            self.assertTrue(np.all(np.abs(s["physics"]["base_com_offset"]) <= 0.02))
            self.assertTrue(0.95 <= s["initial_state"]["arm_scale"] <= 1.05)
            self.assertTrue(0.0 <= s["initial_state"]["base_dz"] <= 0.1)
            self.assertGreater(sum(e["steps"] for e in s["base_schedule"]), 150)
            self.assertGreater(sum(e["hold"] for e in s["ee_schedule"]), 150)
            for e in s["base_schedule"]:
                self.assertTrue(30 <= e["steps"] <= 50 and np.all(np.abs(e["v"]) <= 0.6))
            for e in s["ee_schedule"]:
                self.assertTrue(40 <= e["hold"] <= 50 and 0 <= e["target"] <= 3)
        targets = np.array([e["target"] for s in scs for e in s["ee_schedule"]])
        self.assertEqual(set(targets.tolist()), {0, 1, 2, 3})
        self.assertTrue(np.all(np.abs(np.bincount(targets) / len(targets) - 0.25) < 0.05))


if __name__ == "__main__":
    unittest.main()
