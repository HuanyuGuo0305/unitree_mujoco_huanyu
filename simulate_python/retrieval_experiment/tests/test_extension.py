"""Tests of the 3-7 cm OOD extension (run.py --ood-extension, combine.py).

    python3 -m unittest retrieval_experiment.tests.test_extension -v

Data-dependent tests use the immutable base run (config.yaml: official.base_run) and the latest
FULL OOD extension run in results/; they are skipped if those directories are absent.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_PY = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, SIM_PY)

from retrieval_experiment import combine as C  # noqa: E402
from retrieval_experiment import scenarios as scen  # noqa: E402
from retrieval_experiment.metrics import mean_std  # noqa: E402
from retrieval_experiment.simulator import resolve_repo_path  # noqa: E402

EXP = yaml.safe_load(open(os.path.join(HERE, "..", "config.yaml")))
OFF = EXP["official"]
BASE = resolve_repo_path(OFF["base_run"])
RHO = EXP["cube"]["density_range"]


def latest_full_extension():
    runs = []
    for p in sorted(glob.glob(os.path.join(HERE, "..", "results", "*", "extension_provenance.json"))):
        if json.load(open(p)).get("full_bank") and os.path.isdir(os.path.join(os.path.dirname(p), "combined")):
            runs.append(os.path.dirname(p))
    return runs[-1] if runs else None


EXT = latest_full_extension()


class TestDesign(unittest.TestCase):
    def test_final_groups_and_strata(self):
        fg = OFF["final_groups"]
        self.assertEqual([g["label"] for g in fg], ["[3.0, 4.0]", "[4.0, 5.0]", "[5.0, 6.0]", "[6.0, 7.0]"])
        self.assertEqual([g["type"] for g in fg], ["ood_small", "in_distribution", "in_distribution", "ood_large"])
        for g in fg:
            (a0, a1), (b0, b1) = g["strata"]
            lo, hi = map(float, g["label"].strip("[]").split(","))
            self.assertEqual((a0, a1, b0, b1), (lo, (lo + hi) / 2, (lo + hi) / 2, hi))     # A lower, B upper half
        ood = [iv for g in fg if g["type"] != "in_distribution" for iv in g["strata"]]
        idd = [iv for g in fg if g["type"] == "in_distribution" for iv in g["strata"]]
        self.assertEqual(ood, OFF["ood_intervals_cm"])
        self.assertEqual(idd, EXP["size_intervals_cm"])              # the base-run protocol
        C.assert_disjoint(idd, ood)
        with self.assertRaises(C.SourceDataError):
            C.assert_disjoint(idd, [[5.8, 6.2]])

    def test_ood_size_and_density_sampling(self):
        from scipy.stats import kstest
        bank = scen.generate(1000, EXP["master_seed"], EXP["distributions"], [f"s{i}" for i in range(48)])
        su = np.array([s["cube"]["size_u"] for s in bank])
        du = np.array([s["cube"]["density_u"] for s in bank])
        ivs = OFF["ood_intervals_cm"]
        for g, iv in enumerate(ivs):
            sides = np.array([scen.side_from_interval(u, iv) for u in su])
            self.assertTrue(all(scen.interval_index(x, ivs) == g for x in sides))
            self.assertGreater(kstest((100 * sides - iv[0]) / (iv[1] - iv[0]), "uniform").pvalue, 0.01)
        dens = np.array([scen.density_from_quantile(u, RHO) for u in du])
        self.assertTrue(np.all((dens >= 400) & (dens <= 1200)))
        # theoretical mass extremes of the 3-7 cm x 400-1200 kg/m^3 domain
        self.assertAlmostEqual(400 * 0.03 ** 3, 0.0108, places=10)
        self.assertAlmostEqual(1200 * 0.07 ** 3, 0.4116, places=10)

    def test_pooled_completion_time(self):
        a, b = [5.0, 6.0, 7.0], [4.0, 9.0]
        m, sd, n = mean_std(a + b)
        self.assertEqual(n, 5)
        self.assertAlmostEqual(m, 6.2)
        self.assertAlmostEqual(sd, float(np.std(a + b, ddof=1)))
        self.assertNotAlmostEqual(m, 0.5 * (np.mean(a) + np.mean(b)))   # not a mean of group means


class TestEngineOOD(unittest.TestCase):
    def test_mass_inertia_geometry_spawn(self):
        import torch
        from retrieval_experiment.simulator import RetrievalSim
        torch.set_num_threads(1)
        dep = yaml.safe_load(open(resolve_repo_path(EXP["deploy_config"])))
        sim = RetrievalSim(dep, EXP, observation_noise="none")
        bank = scen.generate(3, EXP["master_seed"], EXP["distributions"], sim.robot_shape_names)
        cases = [(0.03, 400.0), (0.03, 1200.0), (0.07, 400.0), (0.07, 1200.0)]
        cases += [(scen.side_from_interval(b["cube"]["size_u"], iv), scen.density_from_quantile(b["cube"]["density_u"], RHO))
                  for b in bank for iv in OFF["ood_intervals_cm"]]
        for s, rho in cases:
            sim.reset(scen.nominal_scenario(sim.robot_shape_names, object_r=1.5), s, rho)
            m = sim.m
            mass = rho * s ** 3
            self.assertAlmostEqual(float(m.body_mass[sim.cube_bid]) / mass, 1.0, places=12)
            self.assertTrue(np.allclose(m.body_inertia[sim.cube_bid], mass * s * s / 6.0, rtol=1e-9, atol=0))
            self.assertTrue(np.allclose(m.geom_size[sim.cube_gid], s / 2, atol=1e-15, rtol=0))
            self.assertAlmostEqual(float(sim.cube_spawn_w[2]), s / 2 + 0.001, places=6)
            self.assertEqual(int(m.opt.noslip_iterations), 5)


@unittest.skipUnless(os.path.isdir(BASE), "base run not available")
class TestBaseReuse(unittest.TestCase):
    def test_base_run_verified_and_regression(self):
        info = C.verify_base_run(BASE, EXP)
        self.assertEqual(info["n_scenarios"], 250)
        self.assertEqual(info["intervals_cm"], EXP["size_intervals_cm"])
        ep = list(csv.DictReader(open(os.path.join(BASE, "episodes.csv"))))
        ret = {(r["scenario_id"], r["group"]): r for r in csv.DictReader(open(os.path.join(BASE, "analysis", "episodes_retention.csv")))}
        for lab, gs in (("[4.0, 5.0]", ("1", "2")), ("[5.0, 6.0]", ("3", "4"))):
            rs = [r for r in ep if r["group"] in gs]
            got = {"task_success": sum(r["success"] == "True" for r in rs),
                   "one_shot": sum(ret[(r["scenario_id"], r["group"])]["one_shot_success"] == "True" for r in rs)}
            self.assertEqual(got, OFF["regression"][lab])

    def test_tampered_or_incomplete_base_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "base")
            os.makedirs(dst)
            for f in ("scenarios.json", "episodes.csv", "config.yaml"):
                shutil.copy(os.path.join(BASE, f), dst)
            os.symlink(os.path.join(BASE, "trajectories"), os.path.join(dst, "trajectories"))
            C.verify_base_run(dst, EXP)                                  # pristine copy passes
            rows = open(os.path.join(dst, "episodes.csv")).read().splitlines()
            with open(os.path.join(dst, "episodes.csv"), "w") as f:     # drop one episode
                f.write("\n".join(rows[:-1]) + "\n")
            with self.assertRaises(C.SourceDataError):
                C.verify_base_run(dst, EXP)
            exp2 = json.loads(json.dumps(EXP))
            exp2["official"]["base_run_sha256"]["episodes.csv"] = C.sha256(os.path.join(dst, "episodes.csv"))
            with self.assertRaises(C.SourceDataError):                  # hash ok but incomplete
                C.verify_base_run(dst, exp2)
        with self.assertRaises(C.SourceDataError):
            C.verify_base_run(os.path.join(BASE, "does_not_exist"), EXP)


@unittest.skipUnless(EXT is not None and os.path.isdir(BASE), "no full OOD extension run")
class TestCombinedOfficial(unittest.TestCase):
    def setUp(self):
        self.rows = list(csv.DictReader(open(os.path.join(EXT, "combined", "episodes_combined.csv"))))
        self.summary = list(csv.DictReader(open(os.path.join(EXT, "combined", "summary_table.csv"))))

    def test_counts_pairing_provenance(self):
        self.assertEqual(len(self.rows), 2000)
        keys = [(r["scenario_id"], r["size_group"], r["stratum"]) for r in self.rows]
        self.assertEqual(len(set(keys)), 2000)
        by = {}
        for r in self.rows:
            by.setdefault(r["size_group"], []).append(r)
        for g in OFF["final_groups"]:
            rs = by[g["label"]]
            self.assertEqual(len(rs), 500)
            for st in "AB":
                self.assertEqual(sorted(int(r["scenario_id"]) for r in rs if r["stratum"] == st), list(range(250)))
            src = {r["source_run"] for r in rs}
            want = os.path.relpath(BASE, resolve_repo_path(".")) if g["type"] == "in_distribution" else os.path.relpath(EXT, resolve_repo_path("."))
            self.assertEqual(src, {want})
        for r in self.rows[:: 97]:
            self.assertTrue(os.path.isfile(resolve_repo_path(r["trajectory_file"])))
        # the same scenario has the same density and size quantile in every cell
        q = {}
        for r in self.rows:
            q.setdefault(r["scenario_id"], set()).add((r["size_u"], r["density_u"]))
        self.assertTrue(all(len(v) == 1 for v in q.values()))
        # no OOD episode was simulated for an in-distribution stratum and vice versa
        for r in self.rows:
            lo, hi = map(float, r["stratum_interval"].strip("[]").split(","))
            self.assertTrue(lo <= 100 * float(r["actual_cube_side_m"]) <= hi)
            self.assertAlmostEqual(float(r["mujoco_mass"]), float(r["actual_cube_density"]) * float(r["actual_cube_side_m"]) ** 3, places=12)

    def test_summary_from_raw_episodes(self):
        for s in self.summary:
            rs = [r for r in self.rows if r["size_group"] == s["size_interval_cm"]]
            k = sum(r["task_success"] == "True" for r in rs)
            k1 = sum(r["one_shot_success"] == "True" for r in rs)
            self.assertEqual((k, k1, len(rs)), (int(s["task_successes"]), int(s["one_shot_successes"]), int(s["n_trials"])))
            self.assertLessEqual(k1, k)
            self.assertTrue(all(r["task_success"] == "True" for r in rs if r["one_shot_success"] == "True"))
            m, sd, n = mean_std([float(r["completion_time_s"]) for r in rs if r["task_success"] == "True"])
            self.assertAlmostEqual(m, float(s["completion_time_mean_s"]), places=9)
            self.assertAlmostEqual(sd, float(s["completion_time_sd_s"]), places=9)
            self.assertEqual(n, int(s["completion_time_n"]))
        reg = {s["size_interval_cm"]: (int(s["task_successes"]), int(s["one_shot_successes"])) for s in self.summary}
        for lab, want in OFF["regression"].items():
            self.assertEqual(reg[lab], (want["task_success"], want["one_shot"]))


if __name__ == "__main__":
    unittest.main()
