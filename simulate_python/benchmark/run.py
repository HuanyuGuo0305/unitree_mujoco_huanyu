#!/usr/bin/env python3
"""Benchmark driver.

Examples
--------
    python3 benchmark/run.py --experiment E1 --baselines C1,C3,C5 \
        --seed-start 0 --num-seeds 3 --headless
    python3 benchmark/run.py --experiment E2 --baselines C1,C2 \
        --seed-start 0 --num-seeds 3 --headless
    python3 benchmark/run.py --experiment E3 --baselines C1,C3,C5 \
        --seed-start 0 --num-seeds 3 --headless

Headless is the default; ``--headless`` is accepted for explicitness and
``--viewer`` opens the MuJoCo viewer for a single episode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

# Pinned before torch/onnxruntime are imported anywhere: thread count changes
# the floating-point reduction order and would make episodes irreproducible.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "BENCHMARK_ORT_THREADS"):
    os.environ.setdefault(_var, "1")

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for root in (REPO_ROOT, SIM_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

from benchmark.core.runner import EpisodeSpec, run_episode  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

RESULTS_DIR = os.path.join(SIM_ROOT, "benchmark", "results")   # default: results/<EXP>/raw


def episode_dir(results_root, experiment, baseline, condition, seed):
    return os.path.join(
        results_root, experiment, baseline, condition, f"seed_{seed:03d}"
    )


def is_complete(path: str) -> bool:
    required = ("commands.csv", "states.csv", "actuation.csv", "metrics.json", "metadata.json")
    return all(os.path.isfile(os.path.join(path, name)) for name in required)


def build_job_list(args, exp_cfg):
    conditions = (
        [c.strip() for c in args.conditions.split(",")]
        if args.conditions
        else E.all_conditions(args.experiment, exp_cfg)
    )
    known = set(E.all_conditions(args.experiment, exp_cfg))
    unknown = [c for c in conditions if c not in known]
    if unknown:
        raise SystemExit(
            f"Unknown condition(s) {unknown} for {args.experiment}. Known: {sorted(known)}"
        )
    baselines = (
        [b.strip() for b in args.baselines.split(",")]
        if args.baselines
        else list(exp_cfg["baselines"])
    )
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    jobs = []
    for baseline in baselines:
        for condition in conditions:
            for seed in seeds:
                jobs.append((baseline, condition, seed))
    return jobs, baselines, conditions, seeds


def run_one(payload):
    (
        experiment, baseline, condition, seed, exp_cfg, results_root,
        log_physics, verbose,
    ) = payload
    out_dir = episode_dir(results_root, experiment, baseline, condition, seed)
    scenario = E.build_scenario(experiment, exp_cfg, condition, seed)
    spec = EpisodeSpec(
        experiment=experiment,
        baseline=baseline,
        condition=condition,
        seed=seed,
        duration_s=float(exp_cfg["duration_s"]),
        settle_s=float(exp_cfg["settle_s"]),
        termination=exp_cfg["termination"],
        initial_state=exp_cfg.get("initial_state", {}),
        log_physics=log_physics,
        notes={
            "base_command_limits_source": exp_cfg["_global"]["base_command_limits_source"],
            "settle_s_rationale": (
                "native startup_hold 1.0 s + startup_blend 2.0 s = full policy "
                "authority at t = 3.0 s"
            ),
        },
    )
    try:
        metrics = run_episode(spec, scenario, out_dir, verbose=verbose)
        return (baseline, condition, seed, metrics, None)
    except Exception as exc:  # noqa: BLE001
        return (baseline, condition, seed, None, f"{exc}\n{traceback.format_exc()}")


def _worker(payload):
    import torch

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "BENCHMARK_ORT_THREADS"):
        os.environ[var] = "1"
    torch.set_num_threads(1)
    return run_one(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=["E1", "E2", "E3"])
    parser.add_argument("--config", default=None, help="Override the experiment YAML")
    parser.add_argument("--baselines", default=None, help="Comma-separated, e.g. C1,C3,C5")
    parser.add_argument("--conditions", default=None, help="Comma-separated condition IDs")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--headless", action="store_true", help="Default; kept for explicitness")
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo viewer (1 episode)")
    parser.add_argument("--dry-run", action="store_true", help="Generate and save commands only")
    parser.add_argument("--output-dir", default=None,
                        help="Episode root (default: benchmark/results/<EXPERIMENT>/raw)")
    parser.add_argument("--resume-missing", action="store_true", help="Skip completed episodes")
    parser.add_argument("--showcase-search", action="store_true",
                        help="Use the showcase candidate seed pool from baselines.yaml")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes")
    parser.add_argument("--log-physics", action="store_true", help="Also store 250 Hz arrays")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    exp_cfg, global_cfg = E.load_config(args.experiment, args.config)
    if args.showcase_search:
        pool = global_cfg["showcase_seeds"]
        args.seed_start = int(pool["start"])
        args.num_seeds = int(pool["count"])
        print(f"[showcase] candidate seed pool {args.seed_start}-{args.seed_start+args.num_seeds-1}")

    jobs, baselines, conditions, seeds = build_job_list(args, exp_cfg)
    results_root = os.path.abspath(args.output_dir or os.path.join(RESULTS_DIR, args.experiment, "raw"))
    os.makedirs(results_root, exist_ok=True)

    print("=" * 78)
    print(f"Experiment {args.experiment}  |  baselines {baselines}")
    print(f"Conditions ({len(conditions)}): {conditions}")
    print(f"Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)})   Episodes: {len(jobs)}")
    print(f"Duration {exp_cfg['duration_s']} s, settle {exp_cfg['settle_s']} s "
          f"(evaluated {exp_cfg['duration_s'] - exp_cfg['settle_s']} s)")
    print(f"Base command limits: {exp_cfg['base_command_limits']} "
          f"({global_cfg['base_command_limits_source']})")
    print(f"Output: {results_root}")
    print("=" * 78)

    if args.dry_run:
        return _dry_run(args, exp_cfg, jobs, results_root)

    if args.viewer:
        if len(jobs) != 1:
            raise SystemExit("--viewer requires exactly one episode (1 baseline, 1 condition, 1 seed)")
        return _run_with_viewer(args, exp_cfg, jobs[0], results_root)

    pending = []
    skipped = 0
    for baseline, condition, seed in jobs:
        out_dir = episode_dir(results_root, args.experiment, baseline, condition, seed)
        if args.resume_missing and is_complete(out_dir):
            skipped += 1
            continue
        pending.append(
            (args.experiment, baseline, condition, seed, exp_cfg, results_root,
             args.log_physics, args.verbose)
        )
    if skipped:
        print(f"[resume] skipping {skipped} completed episodes")

    t0 = time.time()
    failures = []
    summary = []

    if args.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker, pending), 1):
                _report(result, i, len(pending), t0, failures, summary)
    else:
        import torch

        torch.set_num_threads(1)
        for i, payload in enumerate(pending, 1):
            _report(run_one(payload), i, len(pending), t0, failures, summary)

    print("=" * 78)
    elapsed = time.time() - t0
    print(f"Completed {len(pending) - len(failures)}/{len(pending)} episodes in {elapsed/60:.1f} min")
    if summary:
        _print_condition_summary(summary)
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for baseline, condition, seed, message in failures[:10]:
            print(f"  {baseline}/{condition}/seed_{seed:03d}: {message.splitlines()[0]}")
        return 1
    return 0


def _report(result, i, total, t0, failures, summary):
    baseline, condition, seed, metrics, error = result
    if error is not None:
        failures.append((baseline, condition, seed, error))
        print(f"[{i:4d}/{total}] {baseline}/{condition}/seed_{seed:03d}  FAILED: "
              f"{error.splitlines()[0]}")
        return
    summary.append((baseline, condition, metrics))
    rate = (time.time() - t0) / max(i, 1)
    eta = rate * (total - i) / 60.0
    print(
        f"[{i:4d}/{total}] {baseline}/{condition}/seed_{seed:03d}  "
        f"vxy={metrics.get('base_vxy_rmse', float('nan')):.4f} "
        f"wz={metrics.get('base_wz_rmse', float('nan')):.4f} "
        f"ee_p={metrics.get('ee_pos_rmse', float('nan')):.4f} "
        f"ee_o={metrics.get('ee_ori_mae_deg', float('nan')):.2f}deg "
        f"surv={metrics['survived']} ({metrics['termination_reason']}) "
        f"{metrics['wall_time_s']:.1f}s  ETA {eta:.1f}min",
        flush=True,
    )


def _print_condition_summary(summary):
    print("\nPer-(baseline, condition) means over the episodes just run:")
    keys = ("base_vxy_rmse", "base_wz_rmse", "ee_pos_rmse", "ee_ori_mae_deg", "survived")
    groups: dict = {}
    for baseline, condition, metrics in summary:
        groups.setdefault((baseline, condition), []).append(metrics)
    header = f"{'baseline':9s} {'condition':20s} " + " ".join(f"{k:>16s}" for k in keys) + "    n"
    print(header)
    print("-" * len(header))
    for (baseline, condition), rows in sorted(groups.items()):
        values = []
        for key in keys:
            vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
            values.append(float(np.mean(vals)) if vals else float("nan"))
        print(
            f"{baseline:9s} {condition:20s} "
            + " ".join(f"{v:16.4f}" for v in values)
            + f" {len(rows):4d}"
        )


def _dry_run(args, exp_cfg, jobs, results_root):
    """Generate and save every command schedule without running physics."""
    out_root = os.path.join(results_root, "_dry_run", args.experiment)
    os.makedirs(out_root, exist_ok=True)
    seen = set()
    for baseline, condition, seed in jobs:
        key = (condition, seed)
        if key in seen:
            continue
        seen.add(key)
        scenario = E.build_scenario(args.experiment, exp_cfg, condition, seed)
        arrays = scenario.arrays()
        path = os.path.join(out_root, f"{condition}_seed_{seed:03d}.npz")
        np.savez_compressed(path, **arrays)
        with open(path.replace(".npz", ".json"), "w", encoding="utf-8") as handle:
            json.dump(scenario.description, handle, indent=2, default=str)
        base = arrays["base_commands"]
        ee = arrays["ee_commands_plb"]
        durations = arrays.get("base_segment_durations_s")
        print(
            f"{condition} seed {seed:3d}: "
            f"base |cmd|max={np.abs(base).max():.3f} "
            f"segments={len(durations) if durations is not None else 0} "
            f"dur=[{durations.min():.2f},{durations.max():.2f}]s "
            f"ee_switches={len(arrays.get('ee_switch_ticks', []))} "
            f"ee_kp0_range=x[{ee[:,0].min():.2f},{ee[:,0].max():.2f}] "
            f"y[{ee[:,1].min():.2f},{ee[:,1].max():.2f}] "
            f"z[{ee[:,2].min():.2f},{ee[:,2].max():.2f}]"
        )
    print(f"\nSaved {len(seen)} command schedules to {out_root}")
    return 0


def _run_with_viewer(args, exp_cfg, job, results_root):
    """Interactive single-episode run, for visual inspection only."""
    import mujoco
    import mujoco.viewer

    from benchmark.controllers.registry import build_adapter
    from benchmark.core import frames

    baseline, condition, seed = job
    scenario = E.build_scenario(args.experiment, exp_cfg, condition, seed)
    adapter = build_adapter(baseline, verbose=True)
    model = adapter.build_model()
    data = mujoco.MjData(model)
    adapter.attach(model, data)
    adapter.set_initial_state()
    initial_ee = frames.ee_keypoints_plb_from_state(
        data.xpos[adapter.ee_bid], data.xmat[adapter.ee_bid].reshape(3, 3),
        data.qpos[0:3], data.qpos[3:7], adapter.ee_ground_z,
        adapter.ee_kp_dx, adapter.ee_kp_dz,
    )
    scenario.prepare(adapter, initial_ee)
    settle_base = np.zeros(3, dtype=np.float32)
    adapter.reset(settle_base, initial_ee)

    settle_ticks = int(round(float(exp_cfg["settle_s"]) / adapter.control_dt))
    n_steps = int(round(float(exp_cfg["duration_s"]) / adapter.simulation_dt))
    sim_time, tick = 0.0, 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 135, -20, 3.0
        for counter in range(n_steps):
            if not viewer.is_running():
                break
            wall = time.time()
            adapter.pre_physics(sim_time)
            mujoco.mj_step(model, data)
            sim_time += adapter.simulation_dt
            if counter % adapter.control_decimation == 0:
                k = tick - settle_ticks
                if k < 0:
                    base_cmd, ee_cmd = settle_base, initial_ee
                else:
                    base_cmd, ee_cmd, _ = scenario.command(k, adapter)
                adapter.policy_tick(base_cmd, ee_cmd, sim_time)
                tick += 1
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            remaining = adapter.simulation_dt - (time.time() - wall)
            if remaining > 0:
                time.sleep(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
