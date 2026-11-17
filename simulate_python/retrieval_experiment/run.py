"""Run the E5 retrieval benchmark: paired base scenarios x continuous cube-size intervals.

    cd unitree_mujoco/simulate_python
    python3 -m retrieval_experiment.run --trials-per-group 5 --tag pilot5          # 5 x 4 = 20 episodes
    python3 -m retrieval_experiment.run --tag official250                          # config: 250 x 4 = 1000
    python3 -m retrieval_experiment.run --trials-per-group 500 --tag official500
    python3 -m retrieval_experiment.run --scenarios <run>/scenarios.json --tag repeat
    python3 -m retrieval_experiment.run --ood-extension                            # official OOD strata only
    python3 -m retrieval_experiment.run --ood-extension --trials-per-group 5 --tag ood_pilot5

``--ood-extension`` (official revision 4) verifies the immutable base run named in
``config.yaml: official.base_run``, replays ITS scenario bank on the four OOD half-cm strata
(3.0-3.5, 3.5-4.0, 6.0-6.5, 6.5-7.0 cm) only -- the 4-6 cm strata are never re-simulated --
writes extension_provenance.json and then runs combine.py (base + OOD -> four 1-cm groups).

For base scenario i and size interval g = (lo, hi) [cm] the episode uses
    side    = lo + size_u_i * (hi - lo)            [cm, unrounded]
    density = 400 + 800 * density_u_i              [kg/m^3]
    mass    = density * side^3,  inertia = mass * side^2 / 6

Every invocation writes a NEW directory <output-dir>/<UTC timestamp>_<tag>/:
    config.yaml            effective experiment config (incl. CLI overrides) + deploy config
    scenarios.json         paired base scenarios (written before any rollout)
    policy_manifest.json   checkpoint paths, SHA-256, training runs
    model_parameters.json  solver / contact parameters shared by every compiled cube model
    episodes.csv/.jsonl    one row per (scenario, interval) incl. sampled side, density, mass and
                           the MuJoCo read-back mass / inertia; failures included
    trajectories/*.npz     10-Hz traces sufficient to recompute every metric
    metadata.json          git state, software versions, timing
    analysis/              written by analyze.py (also callable stand-alone)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from retrieval_experiment import scenarios as scen  # noqa: E402
from retrieval_experiment.simulator import REPO_ROOT, RetrievalSim, resolve_repo_path, run_episode  # noqa: E402

CSV_FIELDS = [
    "scenario_id", "group", "interval_cm", "size_u", "density_u",
    "cube_side_m", "cube_side_cm", "cube_density", "cube_mass_kg", "mujoco_mass_kg", "mujoco_inertia",
    "expected_inertia", "observation_noise", "noise_seed",
    "cube_mu_static", "cube_mu_dynamic", "cube_mu_used", "cube_restitution",
    "base_mass_add", "base_com_x", "base_com_y", "base_com_z",
    "base_dz", "base_roll", "base_pitch", "arm_scale",
    "object_r", "object_theta", "cube_spawn_x", "cube_spawn_y", "cube_spawn_z", "cube_spawn_dist",
    "target_x", "target_y", "target_z", "target_radius",
    "success", "failure_reason", "hl_steps", "episode_time_s", "completion_time_s",
    "time_to_stage1_s", "time_to_grasp_s", "retrieval_phase_time_s",
    "reached_stage1", "grasp_confirmed", "n_s1_to_s0", "final_stage", "max_stage",
    "final_retrieval_error", "final_cube_z",
    "s2_steps", "s2_dual_contact_frac", "s2_mean_f_stator", "s2_mean_f_mover", "s2_max_penetration_mm",
    "s2_cube_slip_mm", "s2_mean_gripper_angle_deg", "error", "wall_s",
]

_W = {}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def episode_cube(sc: dict, interval_cm, exp: dict) -> tuple[float, float]:
    side = scen.side_from_interval(sc["cube"]["size_u"], interval_cm)
    density = scen.density_from_quantile(sc["cube"]["density_u"], exp["cube"]["density_range"])
    return side, density


def _init_worker(exp, dep, save_traj):
    import torch
    torch.set_num_threads(1)
    for k in ("high_level", "low_level", "uan"):
        p = resolve_repo_path(exp["policies"][k]["path"])
        if file_sha256(p) != exp["policies"][k]["sha256"]:
            raise RuntimeError(f"{k} checkpoint hash mismatch: {p}")
    _W["sim"] = RetrievalSim(dep, exp, ort_threads=1)
    _W["save_traj"] = save_traj
    _W["exp"] = exp


def _task(args):
    sc, g, interval, traj_dir = args
    sim, exp = _W["sim"], _W["exp"]
    side, density = episode_cube(sc, interval, exp)
    if scen.interval_index(side, exp["size_intervals_cm"]) != g:
        raise RuntimeError(f"sampled side {side} not in interval {interval}")
    t0 = time.time()
    summary, tr = run_episode(sim, sc, side, density, record=True)
    if _W["save_traj"]:
        np.savez_compressed(os.path.join(traj_dir, f"s{sc['scenario_id']:04d}_g{g + 1}.npz"), **tr)
    ph, init, task = sc["physics"], sc["initial_state"], sc["task"]
    src = sim.friction_source
    spawn, tgt = tr["cube_spawn_w"], tr["retrieval_target_w"]
    mass = density * side ** 3
    row = {
        "scenario_id": sc["scenario_id"], "group": g + 1, "interval_cm": f"[{interval[0]:.1f}, {interval[1]:.1f}]",
        "size_u": sc["cube"]["size_u"], "density_u": sc["cube"]["density_u"],
        "cube_side_cm": 100.0 * side, "mujoco_mass_kg": float(sim.m.body_mass[sim.cube_bid]),
        "mujoco_inertia": float(sim.m.body_inertia[sim.cube_bid][0]), "expected_inertia": mass * side ** 2 / 6.0,
        "observation_noise": sim.observation_noise, "noise_seed": sc["noise_seed"],
        "cube_mu_static": ph["cube_mu_static"], "cube_mu_dynamic": ph["cube_mu_dynamic"],
        "cube_mu_used": ph[f"cube_mu_{src}"], "cube_restitution": ph["cube_restitution"],
        "base_mass_add": ph["base_mass_add"], "base_com_x": ph["base_com_offset"][0],
        "base_com_y": ph["base_com_offset"][1], "base_com_z": ph["base_com_offset"][2],
        "base_dz": init["base_dz"], "base_roll": init["base_roll"], "base_pitch": init["base_pitch"],
        "arm_scale": init["arm_scale"], "object_r": task["object_r"], "object_theta": task["object_theta"],
        "cube_spawn_x": float(spawn[0]), "cube_spawn_y": float(spawn[1]), "cube_spawn_z": float(spawn[2]),
        "cube_spawn_dist": float(np.hypot(spawn[0], spawn[1])),
        "target_x": float(tgt[0]), "target_y": float(tgt[1]), "target_z": float(tgt[2]),
        "target_radius": float(np.hypot(tgt[0], tgt[1])), "wall_s": time.time() - t0,
    }
    row.update(summary)
    return row


def git_state():
    def sh(*a):
        try:
            return subprocess.check_output(a, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    porcelain = sh("git", "status", "--porcelain") or ""
    return {"commit": sh("git", "rev-parse", "HEAD"), "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len([ln for ln in porcelain.splitlines() if ln.strip()])}


def model_parameters(sim):
    """Parameters shared by every per-episode compiled cube model."""
    import mujoco
    m = sim.m
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return {
        "cube_geom_type": "box", "cube_solref": m.geom_solref[sim.cube_gid].tolist(),
        "cube_solimp": m.geom_solimp[sim.cube_gid].tolist(), "cube_condim": int(m.geom_condim[sim.cube_gid]),
        "cube_priority_runtime": int(sim.cube_cfg["priority"]), "spawn_z": "side/2 + %g" % sim.cube_cfg["spawn_z_epsilon"],
        "mass": "density * side^3", "inertia": "mass * side^2 / 6 (compiler, solid box)",
        "density_range": sim.cube_cfg["density_range"], "training_density": sim.cube_cfg["training_density"],
        "n_cube_gripper_pairs": int(m.npair), "pair_solref": m.pair_solref[0].tolist(),
        "pair_solimp": m.pair_solimp[0].tolist(), "floor_friction_xml": m.geom_friction[floor].tolist(),
        "n_robot_collision_shapes": len(sim.robot_shape_names),
        "timestep": float(m.opt.timestep), "integrator": int(m.opt.integrator), "cone": int(m.opt.cone),
        "solver": int(m.opt.solver), "iterations": int(m.opt.iterations), "impratio": float(m.opt.impratio),
        "noslip_iterations": int(m.opt.noslip_iterations),
        "multiccd": bool(m.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--size-intervals", default=None,
                    help='intervals in cm, e.g. "4.0-4.5,4.5-5.0" (default: config)')
    ap.add_argument("--trials-per-group", type=int, default=None, help="base scenarios replayed per interval")
    ap.add_argument("--seed", type=int, default=None, help="master seed of the scenario bank (default: config)")
    ap.add_argument("--scenarios", default=None, help="reuse an existing scenarios.json")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--observation-noise", choices=["training", "none"], default=None)
    ap.add_argument("--noslip-iterations", type=int, default=None,
                    help="override MuJoCo noslip iterations (sensitivity analysis only; official = config)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--output-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--no-trajectories", action="store_true")
    ap.add_argument("--no-analysis", action="store_true")
    ap.add_argument("--ood-extension", action="store_true",
                    help="simulate only the official OOD strata on the base run's scenario bank, then combine")
    args = ap.parse_args(argv)

    exp = yaml.safe_load(open(args.config))
    if args.size_intervals:
        exp["size_intervals_cm"] = [[float(a) for a in iv.split("-")] for iv in args.size_intervals.split(",")]
    if args.trials_per_group is not None:
        exp["trials_per_group"] = int(args.trials_per_group)
    if args.seed is not None:
        exp["master_seed"] = int(args.seed)
    if args.observation_noise:
        exp["observation_noise"] = args.observation_noise
    if args.workers:
        exp["workers"] = int(args.workers)
    if args.noslip_iterations is not None:
        exp["mujoco_noslip_iterations"] = int(args.noslip_iterations)
    base_info = None
    if args.ood_extension:
        from retrieval_experiment import combine
        if args.size_intervals or args.scenarios or args.seed is not None or args.noslip_iterations is not None \
                or args.observation_noise:
            raise SystemExit("--ood-extension fixes intervals, scenario bank, seed and physics; do not override them")
        base_dir = resolve_repo_path(exp["official"]["base_run"])
        base_info = combine.verify_base_run(base_dir, exp)          # raises on any inconsistency
        combine.assert_disjoint(base_info["intervals_cm"], exp["official"]["ood_intervals_cm"])
        exp["size_intervals_cm"] = [list(map(float, iv)) for iv in exp["official"]["ood_intervals_cm"]]
        args.scenarios = os.path.join(base_dir, "scenarios.json")
        if args.tag == "run":
            args.tag = "ood_extension"
    intervals = [[float(a), float(b)] for a, b in exp["size_intervals_cm"]]
    save_traj = bool(exp.get("save_trajectories", True)) and not args.no_trajectories
    dep = yaml.safe_load(open(resolve_repo_path(exp["deploy_config"])))

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.output_dir, f"{stamp}_{args.tag}")
    os.makedirs(run_dir, exist_ok=False)
    traj_dir = os.path.join(run_dir, "trajectories")
    os.makedirs(traj_dir)

    import torch
    torch.set_num_threads(1)
    probe = RetrievalSim(dep, exp)
    if args.scenarios:
        meta, scs = scen.load(args.scenarios)
        scs = scs[: exp["trials_per_group"]]
        meta = dict(meta, reused_from=os.path.abspath(args.scenarios))
    else:
        scs = scen.generate(exp["trials_per_group"], exp["master_seed"], exp["distributions"], probe.robot_shape_names)
        meta = {"master_seed": exp["master_seed"], "n": len(scs), "distributions": exp["distributions"],
                "size_intervals_cm": intervals, "density_range": exp["cube"]["density_range"],
                "robot_shape_names": probe.robot_shape_names}
    exp["trials_per_group"] = len(scs)
    scen.save(os.path.join(run_dir, "scenarios.json"), scs, meta)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({"experiment_config": exp, "deploy_config_path": exp["deploy_config"],
                        "deploy_config": dep}, f, sort_keys=False)
    manifest = {}
    for k in ("high_level", "low_level", "uan"):
        p = resolve_repo_path(exp["policies"][k]["path"])
        manifest[k] = dict(exp["policies"][k], path=os.path.relpath(p, REPO_ROOT), sha256_verified=file_sha256(p))
    manifest["deploy_config_sha256"] = file_sha256(resolve_repo_path(exp["deploy_config"]))
    manifest["scene_xml"] = os.path.relpath(probe.xml_path, REPO_ROOT)
    manifest["scene_xml_sha256"] = file_sha256(probe.xml_path)
    manifest["robot_xml_sha256"] = file_sha256(os.path.join(os.path.dirname(probe.xml_path), "b2wz1.xml"))
    with open(os.path.join(run_dir, "policy_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(run_dir, "model_parameters.json"), "w") as f:
        json.dump(model_parameters(probe), f, indent=1)

    tasks = [(s, g, iv, traj_dir) for s in scs for g, iv in enumerate(intervals)]
    workers = int(exp.get("workers", 8))
    print(f"[E5] {len(scs)} scenarios x {len(intervals)} size intervals = {len(tasks)} episodes, "
          f"noise={exp['observation_noise']}, noslip={probe.m.opt.noslip_iterations}, {workers} workers -> {run_dir}",
          flush=True)
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(exp, dep, save_traj)) as pool:
        for i, row in enumerate(pool.imap_unordered(_task, tasks, chunksize=1)):
            rows.append(row)
            if (i + 1) % max(1, len(tasks) // 10) == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)} episodes ({time.time() - t0:.0f} s)", flush=True)
    rows.sort(key=lambda r: (r["scenario_id"], r["group"]))
    if len(rows) != len(tasks):
        raise RuntimeError("missing episodes")
    with open(os.path.join(run_dir, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    with open(os.path.join(run_dir, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    import mujoco
    import onnxruntime
    metadata = {
        "experiment": exp["experiment"], "created_utc": stamp, "tag": args.tag, "wall_time_s": time.time() - t0,
        "n_scenarios": len(scs), "size_intervals_cm": intervals, "density_range": exp["cube"]["density_range"],
        "n_episodes": len(rows), "workers": workers, "observation_noise": exp["observation_noise"],
        "mujoco_noslip_iterations": int(probe.m.opt.noslip_iterations), "cli": vars(args),
        "timing": {"physics_dt": probe.simulation_dt, "wbc_dt": probe.ll_dt, "hl_dt": probe.hl_dt,
                   "max_episode_length": probe.max_episode_hl_steps,
                   "timeout_after_hl_steps": probe.max_episode_hl_steps - 1},
        "threads": {"onnxruntime": 1, "torch": 1}, "git": git_state(),
        "software": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                     "onnxruntime": onnxruntime.__version__, "torch": torch.__version__, "numpy": np.__version__,
                     "platform": platform.platform()},
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=1, default=str)
    if base_info is not None:
        with open(os.path.join(run_dir, "extension_provenance.json"), "w") as f:
            json.dump({"role": "ood_extension", "base_run": exp["official"]["base_run"],
                       "base_run_sha256": base_info["sha256"], "base_intervals_cm": base_info["intervals_cm"],
                       "ood_intervals_cm": exp["size_intervals_cm"], "scenario_bank": "identical to base_run/scenarios.json",
                       "n_scenarios": len(scs), "full_bank": len(scs) == base_info["n_scenarios"]}, f, indent=1)
    counts = {}
    for r in rows:
        counts.setdefault(r["interval_cm"], {}).setdefault(r["failure_reason"], 0)
        counts[r["interval_cm"]][r["failure_reason"]] += 1
    print("[E5] outcome counts:", json.dumps(counts), flush=True)
    print(f"[E5] done in {time.time() - t0:.0f} s -> {run_dir}", flush=True)
    if not args.no_analysis:
        from retrieval_experiment import analyze
        analyze.main([run_dir])
        if base_info is not None:
            from retrieval_experiment import combine
            combine.main([run_dir] + ([] if len(scs) == base_info["n_scenarios"] else ["--allow-partial"]))
    return run_dir


if __name__ == "__main__":
    main()
