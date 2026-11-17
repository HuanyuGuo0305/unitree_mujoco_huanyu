"""Paired E4 scenarios, generated once and stored independently of any rollout.

Each scenario fixes every random draw an episode needs:
  physics        puck mass, puck friction (Isaac Lab static/dynamic pair and the
                 single MuJoCo coefficient), base-link mass offset, base COM offset;
  initial_state  base z offset, roll, pitch, 6-D root velocity, arm-pose scale,
                 puck in-plane spawn offset in the disk frame;
  base_schedule  [vx, vy, wz] references and their hold lengths (10-Hz steps);
  ee_schedule    canonical target indices (0 HIGH, 1 LOW, 2 LEFT, 3 RIGHT) and
                 their hold lengths (10-Hz steps).

Distributions reproduce the Isaac Lab transport environment
(params/env.yaml of the three training runs). The schedules are consumed in
order by ``TransportSim.advance_references``; they are long enough to cover the
whole 149-step horizon, so their contents never depend on how an episode ends.
Every scenario has its own ``SeedSequence([master_seed, scenario_id])`` stream,
so scenario k is identical whatever ``n_scenarios`` is.

The episode-specific HIGH position is NOT stored: as in training it is the EE
kp0 measured at the end of each episode's 0.5-s settle.
"""
from __future__ import annotations

import json
import os

import numpy as np

SCHEMA_VERSION = 1


def _u(rng, lo, hi, size=None):
    return rng.uniform(float(lo), float(hi), size=size)


def sample_scenario(scenario_id: int, master_seed: int, dist: dict, horizon_steps: int) -> dict:
    """Sample one scenario. ``dist`` is the ``distributions`` block of config.yaml."""
    ss = np.random.SeedSequence([int(master_seed), int(scenario_id)])
    rng = np.random.default_rng(ss)
    dr = dist["physical_randomization"]
    rs = dist["reset_randomization"]

    # --- physics (fixed draw order) ---
    puck_mass = _u(rng, *dist["puck_mass_range"])
    mu_s = _u(rng, *dist["puck_friction_range"])
    mu_d = min(mu_s, _u(rng, *dist["puck_friction_range"]))      # make_consistent=True
    base_mass_add = _u(rng, *dr["base_mass_add_range"])
    base_com = [_u(rng, *dr["base_com_range"][k]) for k in ("x", "y", "z")]
    # --- initial state ---
    base_dz = _u(rng, *rs["base_z"])
    roll = _u(rng, *rs["roll"])
    pitch = _u(rng, *rs["pitch"])
    root_vel = [_u(rng, *rs["root_velocity"][k]) for k in ("x", "y", "z", "roll", "pitch", "yaw")]
    arm_scale = _u(rng, *rs["arm_joint_position_scale"])
    puck_xy = _u(rng, -dist["puck_spawn_xy"], dist["puck_spawn_xy"], size=2).tolist()

    if not dr["enabled"]:
        base_mass_add, base_com = 0.0, [0.0, 0.0, 0.0]
    if not rs["enabled"]:
        base_dz = roll = pitch = 0.0
        root_vel = [0.0] * 6
        arm_scale = 1.0
    source = dist["puck_friction_source"]
    if source not in ("static", "dynamic"):
        raise ValueError("puck_friction_source must be 'static' or 'dynamic'")
    puck_mu = mu_s if source == "static" else mu_d

    # --- reference schedules (10-Hz steps) ---
    b_lo, b_hi = dist["base_hold_steps"]
    e_lo, e_hi = dist["ee_hold_steps"]
    ranges = np.asarray(dist["base_ref_ranges"], dtype=np.float64)
    base_schedule, total = [], 0
    while total <= horizon_steps:
        v = rng.uniform(ranges[:, 0], ranges[:, 1]).tolist()
        n = int(rng.integers(b_lo, b_hi + 1))
        base_schedule.append({"v": v, "steps": n})
        total += n
    ee_schedule, total = [], 0
    while total <= horizon_steps:
        target = int(rng.integers(0, 4))                            # repeats allowed
        n = int(rng.integers(e_lo, e_hi + 1))
        ee_schedule.append({"target": target, "hold": n})
        total += n

    return {
        "scenario_id": int(scenario_id),
        "seed": [int(master_seed), int(scenario_id)],
        "physics": {
            "puck_mass": float(puck_mass),
            "puck_mu_static": float(mu_s),
            "puck_mu_dynamic": float(mu_d),
            "puck_mu": float(puck_mu),
            "puck_mu_source": source,
            "base_mass_add": float(base_mass_add),
            "base_com_offset": [float(x) for x in base_com],
        },
        "initial_state": {
            "base_dz": float(base_dz),
            "base_roll": float(roll),
            "base_pitch": float(pitch),
            "root_vel": [float(x) for x in root_vel],
            "arm_scale": float(arm_scale),
            "puck_xy": [float(x) for x in puck_xy],
        },
        "base_schedule": base_schedule,
        "ee_schedule": ee_schedule,
    }


def generate(n: int, master_seed: int, dist: dict, horizon_steps: int, start_id: int = 0) -> list[dict]:
    return [sample_scenario(start_id + i, master_seed, dist, horizon_steps) for i in range(n)]


def save(path: str, scenarios: list[dict], meta: dict):
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing scenario file: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "meta": meta, "scenarios": scenarios}, f, indent=1)


def load(path: str):
    with open(path) as f:
        blob = json.load(f)
    if blob.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported scenario schema in {path}")
    return blob["meta"], blob["scenarios"]


def deploy_rng_scenario(seed: int, deploy_cfg: dict, horizon_steps: int, puck_mass: float, puck_mu: float) -> dict:
    """Scenario reproducing the deploy script's runtime RNG consumption (episode 1).

    Used ONLY by the deploy-parity test: the interactive script draws
    spawn xy, then at settle end base (uniform x3, integers) and EE
    (integers, integers), then resamples in step order base-before-EE.
    """
    rng = np.random.default_rng(int(seed))
    f32 = lambda k: np.asarray(deploy_cfg[k], dtype=np.float32)  # noqa: E731
    xy = rng.uniform(-deploy_cfg["puck_spawn_xy_randomization"], deploy_cfg["puck_spawn_xy_randomization"], size=2).astype(np.float32)
    hl_dt = 0.1
    b_lo, b_hi = (int(round(x / hl_dt)) for x in deploy_cfg["base_ref_resample_time_range_s"])
    e_lo, e_hi = (int(round(x / hl_dt)) for x in deploy_cfg["ee_ref_resample_time_range_s"])
    br = f32("base_ref_ranges")
    base, ee = [], []

    def nb():
        v = rng.uniform(br[:, 0], br[:, 1]).astype(np.float32)
        base.append({"v": [float(x) for x in v], "steps": int(rng.integers(b_lo, b_hi + 1))})

    def ne():
        idx = int(rng.integers(0, 4))
        ee.append({"target": idx, "hold": int(rng.integers(e_lo, e_hi + 1))})

    nb(); ne()
    b_rem, e_rem = base[-1]["steps"], ee[-1]["hold"]
    for _ in range(horizon_steps + 5):
        b_rem -= 1; e_rem -= 1
        if b_rem <= 0:
            nb(); b_rem = base[-1]["steps"]
        if e_rem <= 0:
            ne(); e_rem = ee[-1]["hold"]
    return {
        "scenario_id": -1, "seed": [int(seed)],
        "physics": {"puck_mass": puck_mass, "puck_mu": puck_mu, "base_mass_add": 0.0, "base_com_offset": [0.0, 0.0, 0.0]},
        "initial_state": {"base_dz": 0.0, "base_roll": 0.0, "base_pitch": 0.0, "root_vel": [0.0] * 6,
                          "arm_scale": 1.0, "puck_xy": [float(x) for x in xy]},
        "base_schedule": base, "ee_schedule": ee,
    }
