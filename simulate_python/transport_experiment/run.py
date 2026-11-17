"""Run the paired E4 benchmark.

    cd unitree_mujoco/simulate_python
    python3 -m transport_experiment.run --n 5 --tag smoke              # debugging
    python3 -m transport_experiment.run --tag official                 # 200 x 4 episodes
    python3 -m transport_experiment.run --scenarios <run>/scenarios.json --tag repeat   # re-run same scenarios

Every invocation writes a NEW directory results/<UTC timestamp>_<tag>/ (never
overwrites) containing:
    scenarios.json      paired scenario definitions (written before any rollout)
    episodes.csv        one row per (scenario, method), failures included
    episodes.jsonl      same, with nested fields
    timeseries/*.npz    50-Hz / 10-Hz traces sufficient to recompute every metric
    metadata.json       config, asset hashes, git state, software versions, timing
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
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

from transport_experiment import scenarios as scen  # noqa: E402
from transport_experiment.controllers import LABELS, METHODS, build_methods, file_sha256  # noqa: E402
from transport_experiment.episode import run_episode  # noqa: E402
from transport_experiment.sim import REPO_ROOT, TransportSim, resolve_repo_path  # noqa: E402

CSV_FIELDS = [
    "scenario_id", "seed", "method", "method_label", "policy_file", "policy_sha256",
    "puck_mass", "puck_mu", "puck_mu_static", "puck_mu_dynamic", "puck_mu_source",
    "base_mass_add", "base_com_x", "base_com_y", "base_com_z",
    "initial_puck_dx", "initial_puck_dy", "base_dz", "base_roll", "base_pitch", "arm_scale",
    "episode_high_x", "episode_high_y", "episode_high_z",
    "hl_steps", "episode_duration", "active_duration_s", "success", "failure_reason", "failure_time",
    "mean_puck_xy_displacement", "max_puck_xy_displacement", "final_puck_xy_displacement",
    "base_xy_rmse", "base_yaw_rmse",
    "mean_puck_tilt_deg", "max_puck_tilt_deg", "frac_hl_steps_disk_contact", "hl_steps_stator_only_contact",
    "mean_abs_executed_action", "error",
]

_W = {}


def load_configs(config_path):
    exp = yaml.safe_load(open(config_path))
    dep = yaml.safe_load(open(resolve_repo_path(exp["deploy_config"])))
    return exp, dep


def _init_worker(config_path):
    import torch
    torch.set_num_threads(1)
    exp, dep = load_configs(config_path)
    low = resolve_repo_path(exp["low_policy"]["path"])
    if file_sha256(low) != exp["low_policy"]["sha256"]:
        raise RuntimeError("Low-level policy hash mismatch")
    _W["sim"] = TransportSim(dep, low_policy_path=exp["low_policy"]["path"], ort_threads=1)
    if file_sha256(_W["sim"].uan_model_path) != exp["uan_model_sha256"]:
        raise RuntimeError("UAN model hash mismatch")
    _W["methods"] = build_methods(exp, hl_history_length=int(dep["hl_history_length"]))


def _task(args):
    scenario, method_name, ts_dir = args
    meth = _W["methods"][method_name]
    t0 = time.time()
    summary, ts = run_episode(_W["sim"], meth, scenario)
    np.savez_compressed(os.path.join(ts_dir, f"s{scenario['scenario_id']:04d}_{method_name}.npz"), **ts)
    ph, init = scenario["physics"], scenario["initial_state"]
    high = summary.pop("episode_high_kp0") or [float("nan")] * 3
    row = {
        "scenario_id": scenario["scenario_id"], "seed": "-".join(map(str, scenario["seed"])),
        "method": method_name, "method_label": LABELS[method_name],
        "policy_file": os.path.relpath(meth.policy_path, REPO_ROOT) if meth.policy_path else "",
        "policy_sha256": meth.policy_sha256 or "",
        "puck_mass": ph["puck_mass"], "puck_mu": ph["puck_mu"],
        "puck_mu_static": ph.get("puck_mu_static"), "puck_mu_dynamic": ph.get("puck_mu_dynamic"),
        "puck_mu_source": ph.get("puck_mu_source"), "base_mass_add": ph["base_mass_add"],
        "base_com_x": ph["base_com_offset"][0], "base_com_y": ph["base_com_offset"][1], "base_com_z": ph["base_com_offset"][2],
        "initial_puck_dx": init["puck_xy"][0], "initial_puck_dy": init["puck_xy"][1],
        "base_dz": init["base_dz"], "base_roll": init["base_roll"], "base_pitch": init["base_pitch"],
        "arm_scale": init["arm_scale"],
        "episode_high_x": high[0], "episode_high_y": high[1], "episode_high_z": high[2],
        "wall_s": time.time() - t0,
    }
    row.update(summary)
    return row


def git_state():
    def sh(*a):
        try:
            return subprocess.check_output(a, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": sh("git", "rev-parse", "HEAD"), "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": (sh("git", "status", "--porcelain") or "").count("\n") + 1}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--n", type=int, default=None, help="number of scenarios (default: config n_scenarios)")
    ap.add_argument("--scenarios", default=None, help="reuse an existing scenarios.json")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out-root", default=os.path.join(HERE, "results"))
    args = ap.parse_args(argv)

    exp, dep = load_configs(args.config)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.out_root, f"{stamp}_{args.tag}")
    os.makedirs(run_dir, exist_ok=False)
    ts_dir = os.path.join(run_dir, "timeseries")
    os.makedirs(ts_dir)

    probe = TransportSim(dep, low_policy_path=exp["low_policy"]["path"])
    horizon = probe.max_episode_hl_steps
    if args.scenarios:
        meta, scs = scen.load(args.scenarios)
        if args.n is not None:
            scs = scs[: args.n]
        meta = dict(meta, reused_from=os.path.abspath(args.scenarios))
    else:
        n = int(args.n if args.n is not None else exp["n_scenarios"])
        scs = scen.generate(n, int(exp["master_seed"]), exp["distributions"], horizon)
        meta = {"master_seed": exp["master_seed"], "n": n, "distributions": exp["distributions"], "horizon_steps": horizon}
    scen.save(os.path.join(run_dir, "scenarios.json"), scs, meta)

    tasks = [(s, m, ts_dir) for s in scs for m in methods]
    workers = int(args.workers or exp.get("workers", 8))
    print(f"[E4] {len(scs)} scenarios x {len(methods)} methods = {len(tasks)} episodes, {workers} workers -> {run_dir}")
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(args.config,)) as pool:
        for i, row in enumerate(pool.imap_unordered(_task, tasks, chunksize=1)):
            rows.append(row)
            if (i + 1) % max(1, len(tasks) // 20) == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)} episodes  ({time.time() - t0:.0f} s)", flush=True)
    order = {m: i for i, m in enumerate(METHODS)}
    rows.sort(key=lambda r: (r["scenario_id"], order[r["method"]]))

    with open(os.path.join(run_dir, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(run_dir, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    import mujoco
    import onnxruntime
    import torch
    metadata = {
        "experiment": "E4 dynamic nonprehensile transport",
        "created_utc": stamp, "tag": args.tag, "wall_time_s": time.time() - t0,
        "n_scenarios": len(scs), "methods": methods, "workers": workers,
        "config": exp, "deploy_config_path": exp["deploy_config"],
        "deploy_config_sha256": file_sha256(resolve_repo_path(exp["deploy_config"])),
        "xml_path": os.path.relpath(probe.xml_path, REPO_ROOT),
        "assets_sha256": {
            "low_policy": file_sha256(probe.low_policy_path), "uan": file_sha256(probe.uan_model_path),
            **{m: exp["methods"][m].get("sha256") for m in METHODS},
            "xml_scene": file_sha256(probe.xml_path),
            "xml_robot": file_sha256(os.path.join(os.path.dirname(probe.xml_path), "b2wz1_disk.xml")),
        },
        "timing": {"physics_dt": probe.simulation_dt, "wbc_dt": probe.ll_dt, "hl_dt": probe.hl_dt,
                   "max_episode_length": horizon, "timeout_after_hl_steps": horizon - 1,
                   "settle_hl_steps": probe.settle_steps},
        "threads": {"onnxruntime": 1, "torch": 1},
        "git": git_state(),
        "software": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                     "onnxruntime": onnxruntime.__version__, "torch": torch.__version__, "numpy": np.__version__,
                     "platform": platform.platform()},
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=1, default=str)
    counts = {}
    for r in rows:
        counts.setdefault(r["method"], {}).setdefault(r["failure_reason"], 0)
        counts[r["method"]][r["failure_reason"]] += 1
    print("[E4] outcome counts:", json.dumps(counts))
    print(f"[E4] done in {time.time() - t0:.0f} s -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
