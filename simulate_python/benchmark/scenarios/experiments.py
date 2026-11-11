"""Experiment definitions: condition enumeration and scenario construction.

Every experiment builds its command schedules from (seed, condition) alone, so
all compared baselines receive bit-identical commands and a rerun reproduces
them exactly.
"""

from __future__ import annotations

import os

import numpy as np
import yaml

from benchmark.controllers.registry import REPO_ROOT, SIM_ROOT
from benchmark.core import scenario as sc
from benchmark.core.runner import PrecomputedScenario, WorldFrameEEScenario

CONFIG_DIR = os.path.join(SIM_ROOT, "benchmark", "configs")

_TABLE_CACHE: dict = {}


def load_config(experiment: str, override: str | None = None) -> tuple[dict, dict]:
    """Return (experiment config, global config)."""
    with open(os.path.join(CONFIG_DIR, "baselines.yaml"), "r", encoding="utf-8") as handle:
        global_cfg = yaml.safe_load(handle)
    path = override or os.path.join(CONFIG_DIR, f"{experiment}.yaml")
    with open(path, "r", encoding="utf-8") as handle:
        exp_cfg = yaml.safe_load(handle)
    if exp_cfg.get("settle_s") is None:
        exp_cfg["settle_s"] = float(global_cfg["settle_s"])
    exp_cfg["termination"] = dict(global_cfg["termination"])
    exp_cfg["initial_state"] = dict(global_cfg.get("initial_state", {}))
    exp_cfg["base_command_limits"] = list(global_cfg["base_command_limits"])
    exp_cfg["_global"] = global_cfg
    return exp_cfg, global_cfg


def keypoint_table(global_cfg: dict) -> sc.KeypointTable:
    path = os.path.join(REPO_ROOT, global_cfg["ee_command_table"])
    if path not in _TABLE_CACHE:
        _TABLE_CACHE[path] = sc.KeypointTable(path)
    return _TABLE_CACHE[path]


def n_eval_ticks(exp_cfg: dict, control_dt: float = 0.02) -> int:
    return int(round((float(exp_cfg["duration_s"]) - float(exp_cfg["settle_s"])) / control_dt))


# ==========================================================================
# E1
# ==========================================================================
E1_CONDITIONS = ("T1", "T2", "T3", "T4")
E1_CONDITION_TITLES = {
    "T1": "Base still / EE fixed",
    "T2": "Base still / EE random",
    "T3": "Base moving / EE fixed",
    "T4": "Base moving / EE random",
}


def build_e1(exp_cfg: dict, condition: str, seed: int, control_dt: float = 0.02):
    spec = exp_cfg["conditions"][condition]
    n_ticks = n_eval_ticks(exp_cfg, control_dt)
    table = keypoint_table(exp_cfg["_global"])

    base_schedule = sc.make_base_velocity_schedule(
        n_ticks,
        control_dt,
        seed,
        limits=exp_cfg["base_command_limits"],
        intensity=float(exp_cfg["base"]["intensity"]),
        segment_duration_range=tuple(exp_cfg["base"]["segment_duration_range"]),
        zero=(spec["base"] == "still"),
    )
    if spec["ee"] == "fixed":
        ee_schedule = sc.make_ee_fixed_schedule(n_ticks, seed, table)
    else:
        ee_schedule = sc.make_ee_random_schedule(
            n_ticks, control_dt, seed, table,
            cycle_duration_s=float(exp_cfg["ee"]["resample_interval_s"]),
        )

    description = {
        "experiment": "E1",
        "condition": condition,
        "title": E1_CONDITION_TITLES[condition],
        "base_mode": spec["base"],
        "ee_mode": spec["ee"],
        "base_limits": list(exp_cfg["base_command_limits"]),
        "base_segment_duration_range_s": list(exp_cfg["base"]["segment_duration_range"]),
        "ee_resample_interval_s": float(exp_cfg["ee"]["resample_interval_s"]),
        "ee_interpolation": "none",
        "ee_command_table": exp_cfg["_global"]["ee_command_table"],
        "ee_command_table_sha256": table.sha256,
        "n_eval_ticks": n_ticks,
        "seed": seed,
    }
    return PrecomputedScenario(base_schedule, ee_schedule, description)


# ==========================================================================
# E2
# ==========================================================================
E2_CONDITIONS = ("Low", "Medium", "High")


def e2_axis_limits(exp_cfg: dict, condition: str) -> list[float]:
    """Per-axis ABSOLUTE command envelope [vx, vy, wz] for an E2 condition.

    Round 3 states the envelope directly in physical units instead of as a
    fraction of the shared range. The values are validated against
    ``base_command_limits`` so a typo can never command outside the shared
    supported training range of C1 and C2.
    """
    spec = exp_cfg["base"]["motion_intensity"][condition]
    axis = [
        float(spec["max_abs_vx_mps"]),
        float(spec["max_abs_vy_mps"]),
        float(spec["max_abs_yaw_rate_radps"]),
    ]
    shared = [float(v) for v in exp_cfg["base_command_limits"]]
    for name, value, bound in zip(("vx", "vy", "yaw_rate"), axis, shared):
        if value <= 0.0 or value > bound + 1e-12:
            raise ValueError(
                f"E2 {condition}: max_abs_{name} = {value} is outside the shared "
                f"supported command range (0, {bound}]"
            )
    return axis


def build_e2(exp_cfg: dict, condition: str, seed: int, control_dt: float = 0.02):
    n_ticks = n_eval_ticks(exp_cfg, control_dt)
    table = keypoint_table(exp_cfg["_global"])
    axis_limits = e2_axis_limits(exp_cfg, condition)
    shared = [float(v) for v in exp_cfg["base_command_limits"]]
    # Fraction of the shared range, kept for backward-compatible plotting only.
    intensity = float(axis_limits[0] / shared[0])

    base_schedule = _patterned_base_schedule(
        n_ticks,
        control_dt,
        seed,
        # The per-axis absolute envelope IS the limit; the waveform (which is
        # a function of the seed alone) is unchanged, so C1 and C2 at a given
        # seed still receive bit-identical commands and changing the intensity
        # changes nothing but the amplitude.
        limits=axis_limits,
        intensity=1.0,
        segment_duration_range=tuple(exp_cfg["base"]["segment_duration_range"]),
        patterns=tuple(exp_cfg["base"]["pattern_mix"]),
    )
    ee_schedule = sc.make_ee_fixed_schedule(n_ticks, seed, table)

    description = {
        "experiment": "E2",
        "condition": condition,
        "title": (
            f"{condition} motion intensity "
            f"(|vx|,|vy| <= {axis_limits[0]:g} m/s, |wz| <= {axis_limits[2]:g} rad/s)"
        ),
        "base_mode": "patterned",
        "base_intensity": intensity,
        "base_max_abs_vx_mps": axis_limits[0],
        "base_max_abs_vy_mps": axis_limits[1],
        "base_max_abs_yaw_rate_radps": axis_limits[2],
        "base_envelope_is_per_axis": True,
        "base_limits": list(axis_limits),
        "base_shared_supported_limits": shared,
        "base_patterns": list(exp_cfg["base"]["pattern_mix"]),
        "ee_mode": "fixed",
        "ee_command_table_sha256": table.sha256,
        "n_eval_ticks": n_ticks,
        "seed": seed,
    }
    return PrecomputedScenario(base_schedule, ee_schedule, description)


def _patterned_base_schedule(
    n_ticks, control_dt, seed, limits, intensity, segment_duration_range, patterns
):
    """Base schedule that cycles straight / turning / translate+turn segments.

    All three motion patterns appear in every episode, in a seeded order and
    with seeded magnitudes, and the sequence depends only on the seed -- so C1
    and C2 receive identical commands at the same seed.
    """
    limits = np.asarray(limits, dtype=np.float64).reshape(3)
    rng = np.random.default_rng(sc._mix_seed(seed, "base_patterned"))
    order = list(patterns)
    rng.shuffle(order)

    waveform = np.zeros((n_ticks, 3), dtype=np.float32)
    switch_ticks, durations = [], []
    tick = 0
    i = 0
    lo, hi = float(segment_duration_range[0]), float(segment_duration_range[1])
    while tick < n_ticks:
        pattern = order[i % len(order)]
        i += 1
        duration = float(rng.uniform(lo, hi))
        seg = max(1, int(round(duration / control_dt)))
        sign = rng.choice([-1.0, 1.0], size=3)
        mag = rng.uniform(0.5, 1.0, size=3)
        unit = np.zeros(3)
        if pattern == "straight":
            unit[0] = sign[0] * mag[0]
        elif pattern == "turn":
            unit[2] = sign[2] * mag[2]
        elif pattern == "translate_turn":
            unit[0] = sign[0] * mag[0]
            unit[1] = sign[1] * mag[1]
            unit[2] = sign[2] * mag[2]
        else:
            raise ValueError(f"Unknown base motion pattern '{pattern}'")
        end = min(tick + seg, n_ticks)
        waveform[tick:end] = unit.astype(np.float32)
        switch_ticks.append(tick)
        durations.append(seg * control_dt)
        tick = end

    commands = (waveform * (limits * intensity)).astype(np.float32)
    return sc.BaseVelocitySchedule(
        commands=commands,
        switch_ticks=np.asarray(switch_ticks, dtype=np.int64),
        segment_durations_s=np.asarray(durations, dtype=np.float64),
        waveform=waveform,
        intensity=float(intensity),
        limits=tuple(float(v) for v in limits),
    )


# ==========================================================================
# E4
# ==========================================================================
E4_CONDITIONS = ("chicken_head",)


def build_e4(exp_cfg: dict, condition: str, seed: int, control_dt: float = 0.02):
    n_ticks = n_eval_ticks(exp_cfg, control_dt)
    limits = np.asarray(exp_cfg["base_command_limits"], dtype=np.float64)
    amplitude = float(exp_cfg["base"]["amplitude"])
    phase_s = float(exp_cfg["base"]["phase_duration_s"])
    phases = list(exp_cfg["base"]["phases"])

    commands = np.zeros((n_ticks, 3), dtype=np.float32)
    switch_ticks = []
    t = np.arange(n_ticks) * control_dt
    phase_index = np.minimum((t / phase_s).astype(int), len(phases) - 1)
    for p, name in enumerate(phases):
        mask = phase_index == p
        if not mask.any():
            continue
        switch_ticks.append(int(np.nonzero(mask)[0][0]))
        local = t[mask] - p * phase_s
        # Cosine VELOCITY -> sine POSITION: the base returns to its starting
        # pose at the end of every phase, so the world-frame EE target stays
        # inside the reachable region.
        osc = np.cos(2.0 * np.pi * local / phase_s)
        a = amplitude
        if name == "forward_back":
            commands[mask, 0] = a * limits[0] * osc
        elif name == "lateral":
            commands[mask, 1] = a * limits[1] * osc
        elif name == "yaw":
            commands[mask, 2] = a * limits[2] * osc
        elif name == "combined":
            commands[mask, 0] = a * limits[0] * osc
            commands[mask, 2] = a * limits[2] * np.sin(2.0 * np.pi * local / phase_s)
        elif name == "hold":
            pass
        else:
            raise ValueError(f"Unknown E4 base phase '{name}'")

    base_schedule = sc.BaseVelocitySchedule(
        commands=commands,
        switch_ticks=np.asarray(switch_ticks, dtype=np.int64),
        segment_durations_s=np.full(len(switch_ticks), phase_s, dtype=np.float64),
        waveform=commands / np.maximum(limits * amplitude, 1e-9),
        intensity=amplitude,
        limits=tuple(float(v) for v in limits),
    )

    table = keypoint_table(exp_cfg["_global"])
    plb_center = np.asarray(exp_cfg["ee"]["plb_center"], dtype=np.float64)
    target = sc.WorldFrameEETarget(
        position_w=None, rotation_w=None, ground_z=0.0,
        plb_setpoint=table.nearest_row(plb_center),
    )
    description = {
        "experiment": "E4",
        "condition": condition,
        "base_schedule": "chicken_head",
        "base_phases": phases,
        "base_phase_duration_s": phase_s,
        "base_amplitude_fraction": amplitude,
        "base_limits": list(exp_cfg["base_command_limits"]),
        "ee_plb_setpoint_center": plb_center.tolist(),
        "ee_plb_setpoint": target.plb_setpoint.tolist(),
        "ee_world_target": "locked to the measured EE pose at the first evaluated tick",
        "ee_command_transform": "world -> PLB every control update, from the measured base pose",
        "ee_command_table_sha256": table.sha256,
        "n_eval_ticks": n_ticks,
        "seed": seed,
    }
    return WorldFrameEEScenario(
        base_schedule, target, description, table=table,
        support_tolerance=float(exp_cfg["ee"]["support_tolerance"]),
    )


# ==========================================================================
# dispatch
# ==========================================================================
BUILDERS = {"E1": build_e1, "E2": build_e2, "E4": build_e4}


def all_conditions(experiment: str, exp_cfg: dict):
    if experiment == "E1":
        return list(E1_CONDITIONS)
    if experiment == "E2":
        return list(E2_CONDITIONS)
    if experiment == "E4":
        return list(E4_CONDITIONS)
    raise KeyError(experiment)


def build_scenario(experiment: str, exp_cfg: dict, condition: str, seed: int, control_dt=0.02):
    return BUILDERS[experiment](exp_cfg, condition, seed, control_dt)
