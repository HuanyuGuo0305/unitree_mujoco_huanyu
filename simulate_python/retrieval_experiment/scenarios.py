"""Deterministic, size-interval-independent E5 scenario bank.

A *base scenario* fixes every random draw of one episode, including two
normalized quantiles:
  size_u     ~ U(0,1): side_cm = lo + size_u * (hi - lo) inside the evaluated interval
  density_u  ~ U(0,1): density = rho_lo + density_u * (rho_hi - rho_lo)  (400-1200 kg/m^3)
The SAME scenario (same quantiles) is replayed in every size interval (paired
design): within an interval sizes and densities are uniform, across intervals
only the side length -- and with it mass, inertia, spawn height and contact
geometry -- changes. Density is an evaluation-time perturbation (training: 800).

Each scenario has its own ``SeedSequence([master_seed, scenario_id])`` stream,
so scenario k is identical whatever the number of scenarios is. Draw order is
fixed; per-shape robot friction is drawn LAST so the number of robot collision
shapes cannot shift any other draw.

Distributions reproduce the Isaac Lab training environment (EventCfg + _reset_idx):
  physics        cube static/dynamic friction U(1,2) with mu_d = min(mu_s, mu_d)
                 (make_consistent), cube restitution U(0,0.05), base mass +U(-2,2) kg,
                 base COM +U(+-0.02) m, per robot shape static/dynamic friction U(0.4,1.2)
                 (make_consistent) and restitution U(0,0.1);
  initial_state  base z +U(0,0.1), roll/pitch U(+-0.1), 6-D root velocity U(+-0.05),
                 arm joint-position scale U(0.95,1.05), leg scale 1;
  task           cube PLB polar sector r~U(0,3), theta~U(+-pi/4) around (0.8, 0);
                 target theta~U(-pi,pi), radius 0.8*sqrt(U), z~U(0.3,1.0);
  noise_seed     seed of the per-episode actor-observation noise stream;
  cube           size_u, density_u (see above).

Isaac Lab draws material values into 64 shared buckets and assigns buckets to
shapes; sampling each shape directly from the same uniform ranges gives the same
marginal distribution (documented in docs/ALIGNMENT_REPORT.md).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

SCHEMA_VERSION = 2


def _u(rng, lo, hi):
    return float(rng.uniform(float(lo), float(hi)))


def sample_scenario(scenario_id: int, master_seed: int, dist: dict, robot_shape_names: list[str]) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([int(master_seed), int(scenario_id)]))
    cm, rm, rs = dist["cube_material"], dist["robot_material"], dist["reset"]

    # --- cube material (make_consistent: dynamic <= static) ---
    cube_mu_s = _u(rng, *cm["static_friction"])
    cube_mu_d = min(cube_mu_s, _u(rng, *cm["dynamic_friction"]))
    cube_e = _u(rng, *cm["restitution"])
    # --- base inertial randomization ---
    base_mass_add = _u(rng, *dist["base_mass_add"])
    base_com = [_u(rng, *dist["base_com"][k]) for k in ("x", "y", "z")]
    # --- reset state ---
    base_dz = _u(rng, *rs["base_z"])
    roll = _u(rng, *rs["roll"])
    pitch = _u(rng, *rs["pitch"])
    root_vel = [_u(rng, *rs["root_velocity"][k]) for k in ("x", "y", "z", "roll", "pitch", "yaw")]
    leg_scale = _u(rng, *rs["leg_joint_position_scale"])
    arm_scale = _u(rng, *rs["arm_joint_position_scale"])
    # --- task ---
    ob = dist["object"]
    object_r = _u(rng, *ob["r"])
    object_theta = _u(rng, -ob["theta_max"], ob["theta_max"])
    tg = dist["retrieval_target"]
    target_theta = _u(rng, -math.pi, math.pi)
    target_u = _u(rng, 0.0, 1.0)
    target_z = _u(rng, *tg["z_range_w"])
    noise_seed = int(rng.integers(0, 2**31 - 1))
    # --- cube size / density quantiles (shared by all size intervals) ---
    size_u = float(rng.uniform(0.0, 1.0))
    density_u = float(rng.uniform(0.0, 1.0))
    # --- per-shape robot material (last) ---
    n = len(robot_shape_names)
    mu_s = rng.uniform(float(rm["static_friction"][0]), float(rm["static_friction"][1]), size=n)
    mu_d = np.minimum(mu_s, rng.uniform(float(rm["dynamic_friction"][0]), float(rm["dynamic_friction"][1]), size=n))
    e = rng.uniform(float(rm["restitution"][0]), float(rm["restitution"][1]), size=n)

    return {
        "scenario_id": int(scenario_id),
        "seed": [int(master_seed), int(scenario_id)],
        "physics": {
            "cube_mu_static": cube_mu_s,
            "cube_mu_dynamic": cube_mu_d,
            "cube_restitution": cube_e,
            "ground_friction": float(dist.get("ground_friction", 1.0)),
            "robot_shape_names": list(robot_shape_names),
            "robot_mu_static": mu_s.tolist(),
            "robot_mu_dynamic": mu_d.tolist(),
            "robot_restitution": e.tolist(),
            "base_mass_add": base_mass_add,
            "base_com_offset": base_com,
        },
        "initial_state": {
            "base_dz": base_dz, "base_roll": roll, "base_pitch": pitch,
            "root_vel": root_vel, "leg_scale": leg_scale, "arm_scale": arm_scale,
        },
        "task": {
            "object_r": object_r, "object_theta": object_theta,
            "target_theta": target_theta, "target_u": target_u, "target_z": target_z,
        },
        "noise_seed": noise_seed,
        "cube": {"size_u": size_u, "density_u": density_u},
    }


def side_from_interval(size_u: float, interval_cm) -> float:
    """Cube side length in metres for quantile ``size_u`` inside ``interval_cm`` = (lo, hi) [cm]."""
    lo, hi = float(interval_cm[0]), float(interval_cm[1])
    return 0.01 * (lo + float(size_u) * (hi - lo))


def density_from_quantile(density_u: float, density_range) -> float:
    lo, hi = float(density_range[0]), float(density_range[1])
    return lo + float(density_u) * (hi - lo)


def interval_index(side_m: float, intervals_cm) -> int:
    """Group of a side length: [lo, hi) for all but the last interval, [lo, hi] for the last."""
    cm = 100.0 * float(side_m)
    n = len(intervals_cm)
    for i, (lo, hi) in enumerate(intervals_cm):
        if lo <= cm < hi or (i == n - 1 and lo <= cm <= hi):
            return i
    raise ValueError(f"side {cm} cm outside all intervals")


def generate(n: int, master_seed: int, dist: dict, robot_shape_names: list[str]) -> list[dict]:
    return [sample_scenario(i, master_seed, dist, robot_shape_names) for i in range(int(n))]


def nominal_scenario(robot_shape_names: list[str], *, object_r=0.0, object_theta=0.0,
                     target_theta=0.0, target_u=0.25, target_z=0.6, cube_mu=1.0, robot_mu=None) -> dict:
    """No randomization: nominal materials, nominal reset; used by tests/validation."""
    n = len(robot_shape_names)
    rmu = [1.0] * n if robot_mu is None else list(robot_mu)
    return {
        "scenario_id": -1, "seed": [0, -1],
        "physics": {"cube_mu_static": cube_mu, "cube_mu_dynamic": cube_mu, "cube_restitution": 0.0,
                    "ground_friction": 1.0, "robot_shape_names": list(robot_shape_names),
                    "robot_mu_static": rmu, "robot_mu_dynamic": rmu, "robot_restitution": [0.0] * n,
                    "base_mass_add": 0.0, "base_com_offset": [0.0, 0.0, 0.0]},
        "initial_state": {"base_dz": 0.0, "base_roll": 0.0, "base_pitch": 0.0, "root_vel": [0.0] * 6,
                          "leg_scale": 1.0, "arm_scale": 1.0},
        "task": {"object_r": object_r, "object_theta": object_theta, "target_theta": target_theta,
                 "target_u": target_u, "target_z": target_z},
        "noise_seed": 0,
        "cube": {"size_u": 0.5, "density_u": 0.5},
    }


def deploy_rng_task(seed: int, deploy_cfg: dict) -> dict:
    """The task draws the interactive deploy script makes for its FIRST episode.

    Replicates ``sample_task_reset`` (enlarged target mode), including the
    float32 cast of the YAML ranges, so that tests can replay that episode.
    """
    rng = np.random.default_rng(int(seed))
    r_rng = np.asarray(deploy_cfg["object_sampling_r_range"], dtype=np.float32)
    theta_max = float(deploy_cfg["object_sampling_theta_max"])
    z_rng = np.asarray(deploy_cfg["enlarged_retrieval_target_z_range_w"], dtype=np.float32)
    object_r = float(rng.uniform(float(r_rng[0]), float(r_rng[1])))
    object_theta = float(rng.uniform(-theta_max, theta_max))
    target_theta = float(rng.uniform(-math.pi, math.pi))
    target_z = float(rng.uniform(float(z_rng[0]), float(z_rng[1])))
    target_u = float(rng.uniform(0.0, 1.0))
    return {"object_r": object_r, "object_theta": object_theta, "target_theta": target_theta,
            "target_u": target_u, "target_z": target_z}


def save(path: str, scenarios: list[dict], meta: dict):
    with open(path, "w") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "meta": meta, "scenarios": scenarios}, f, indent=1)


def load(path: str):
    with open(path) as f:
        obj = json.load(f)
    if obj.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported scenario schema in {os.path.abspath(path)}")
    return obj["meta"], obj["scenarios"]
