"""Scenario bank: determinism, prefix stability, ranges, and paired replay across sizes.

    python3 -m unittest retrieval_experiment.tests.test_scenarios -v
"""
from __future__ import annotations

import json
import math
import os
import sys
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../..")))

from retrieval_experiment import scenarios as scen  # noqa: E402

EXP = yaml.safe_load(open(os.path.join(HERE, "..", "config.yaml")))
NAMES = [f"shape{i}" for i in range(48)]


class TestScenarios(unittest.TestCase):
    def test_deterministic_and_prefix_stable(self):
        a = scen.generate(50, EXP["master_seed"], EXP["distributions"], NAMES)
        b = scen.generate(200, EXP["master_seed"], EXP["distributions"], NAMES)
        self.assertEqual(json.dumps(a), json.dumps(b[:50]))
        c = scen.generate(50, EXP["master_seed"] + 1, EXP["distributions"], NAMES)
        self.assertNotEqual(json.dumps(a), json.dumps(c))

    def test_ranges(self):
        d = EXP["distributions"]
        bank = scen.generate(500, EXP["master_seed"], d, NAMES)
        for s in bank:
            ph, i, t = s["physics"], s["initial_state"], s["task"]
            self.assertTrue(1.0 <= ph["cube_mu_static"] <= 2.0)
            self.assertTrue(1.0 <= ph["cube_mu_dynamic"] <= ph["cube_mu_static"])
            self.assertTrue(0.0 <= ph["cube_restitution"] <= 0.05)
            self.assertTrue(all(0.4 <= v <= 1.2 for v in ph["robot_mu_static"]))
            self.assertTrue(all(dd <= ss for dd, ss in zip(ph["robot_mu_dynamic"], ph["robot_mu_static"])))
            self.assertTrue(-2.0 <= ph["base_mass_add"] <= 2.0)
            self.assertTrue(all(abs(v) <= 0.02 for v in ph["base_com_offset"]))
            self.assertTrue(0.0 <= i["base_dz"] <= 0.1 and abs(i["base_roll"]) <= 0.1 and abs(i["base_pitch"]) <= 0.1)
            self.assertTrue(all(abs(v) <= 0.05 for v in i["root_vel"]))
            self.assertEqual(i["leg_scale"], 1.0)
            self.assertTrue(0.95 <= i["arm_scale"] <= 1.05)
            self.assertTrue(0.0 <= t["object_r"] <= 3.0 and abs(t["object_theta"]) <= math.pi / 4)
            self.assertTrue(abs(t["target_theta"]) <= math.pi and 0 <= t["target_u"] <= 1 and 0.3 <= t["target_z"] <= 1.0)
        # target radius is uniform by AREA: P(r < 0.8/sqrt(2)) = 0.5
        rad = np.array([0.8 * math.sqrt(s["task"]["target_u"]) for s in bank])
        self.assertAlmostEqual(float(np.mean(rad < 0.8 / math.sqrt(2))), 0.5, delta=0.07)

    def test_size_independent(self):
        """Scenarios carry only normalized quantiles: the same dict is replayed for every interval."""
        s = scen.generate(1, EXP["master_seed"], EXP["distributions"], NAMES)[0]
        flat = json.dumps(s)
        for k in ("side", "mass", "half_extent", "density\""):
            self.assertNotIn(f'"{k}', flat)
        self.assertEqual(set(s["cube"]), {"size_u", "density_u"})

    def test_size_and_density_sampling(self):
        from scipy.stats import kstest
        ivs = EXP["size_intervals_cm"]
        rho = EXP["cube"]["density_range"]
        self.assertEqual(ivs, [[4.0, 4.5], [4.5, 5.0], [5.0, 5.5], [5.5, 6.0]])
        self.assertEqual(rho, [400.0, 1200.0])
        bank = scen.generate(2000, EXP["master_seed"], EXP["distributions"], NAMES)
        su = np.array([s["cube"]["size_u"] for s in bank])
        du = np.array([s["cube"]["density_u"] for s in bank])
        self.assertGreater(kstest(su, "uniform").pvalue, 0.01)
        self.assertGreater(kstest(du, "uniform").pvalue, 0.01)
        self.assertLess(abs(np.corrcoef(su, du)[0, 1]), 0.1)
        for g, iv in enumerate(ivs):
            sides = np.array([scen.side_from_interval(u, iv) for u in su])
            self.assertTrue(all(scen.interval_index(x, ivs) == g for x in sides))
            self.assertTrue(np.all(sides >= iv[0] / 100) and np.all(sides <= iv[1] / 100))
            # within-interval uniformity and no rounding to a grid (0.1 mm)
            self.assertGreater(kstest((100 * sides - iv[0]) / (iv[1] - iv[0]), "uniform").pvalue, 0.01)
            frac = np.abs(sides * 1e4 - np.round(sides * 1e4))
            self.assertGreater(np.mean(frac > 1e-6), 0.99)
        dens = np.array([scen.density_from_quantile(u, rho) for u in du])
        self.assertTrue(np.all((dens >= 400.0) & (dens <= 1200.0)))
        # boundary classification: half-open except the closed last interval
        self.assertEqual(scen.interval_index(0.045, ivs), 1)
        self.assertEqual(scen.interval_index(0.0449999, ivs), 0)
        self.assertEqual(scen.interval_index(0.060, ivs), 3)
        with self.assertRaises(ValueError):
            scen.interval_index(0.0601, ivs)
        # paired replay: interval g side = lo_g + u*(0.5 cm); offsets across intervals are exact multiples
        for s in bank[:50]:
            sides = [scen.side_from_interval(s["cube"]["size_u"], iv) for iv in ivs]
            self.assertTrue(np.allclose(np.diff(sides), 0.005, atol=1e-15))

    def test_quantiles_reproducible(self):
        a = scen.generate(10, EXP["master_seed"], EXP["distributions"], NAMES)
        b = scen.generate(10, EXP["master_seed"], EXP["distributions"], NAMES)
        self.assertEqual([s["cube"] for s in a], [s["cube"] for s in b])


if __name__ == "__main__":
    unittest.main()
