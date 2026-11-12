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
from benchmark.core.runner import PrecomputedScenario

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
# E2 (official since 2026-11-12): wheeled-legged (C1) vs legged (C2) actuation,
# ONE dynamic condition: random base velocity AND random full-pose EE targets.
#   base: E1's validated generator (make_base_velocity_schedule): every 2-3 s an
#         independent uniform draw per axis within base_command_limits (the shared
#         WBC training range, baselines.yaml), piecewise constant;
#   EE:   a full 6-DoF pose row drawn uniformly from the native training keypoint
#         table (E1's sampling), held for an independent uniform 3-5 s interval,
#         switched directly (no interpolation). Own RNG stream, independent of base.
# Both streams depend on the seed alone, so both modes replay identical commands.
# ==========================================================================
E2_CONDITIONS = ("dynamic",)


def make_ee_random_interval_schedule(n_ticks: int, control_dt: float, seed: int,
                                     table: sc.KeypointTable, interval_range_s=(3.0, 5.0)):
    """Full-pose EE targets from the table, each held for U(interval_range_s) seconds.

    Segment 0 starts at scenario tick 0 (t = settle_s), like the E1 random schedule;
    durations are rounded to whole control ticks. No interpolation."""
    lo, hi = float(interval_range_s[0]), float(interval_range_s[1])
    if not 0.0 < lo <= hi:
        raise ValueError(f"bad EE interval range {interval_range_s}")
    rng = np.random.default_rng(sc._mix_seed(seed, "e2_ee_random_interval"))
    seg = np.zeros(n_ticks, dtype=np.int64)
    rows, starts, tick = [], [], 0
    while tick < n_ticks:
        duration = float(rng.uniform(lo, hi))
        row = int(rng.integers(0, table.n_rows))
        steps = max(1, int(round(duration / control_dt)))
        seg[tick:tick + steps] = len(rows)
        rows.append(row)
        starts.append(tick)
        tick += steps
    rows = np.asarray(rows, dtype=np.int64)
    return sc.EEKeypointSchedule(
        commands=table.table[rows[seg]].astype(np.float32).copy(),
        switch_ticks=np.asarray(starts[1:], dtype=np.int64),
        row_indices=rows, mode=f"random_interval:{lo:g}-{hi:g}s")


def build_e2(exp_cfg: dict, condition: str, seed: int, control_dt: float = 0.02):
    if condition not in E2_CONDITIONS:
        raise KeyError(f"E2 has one condition {E2_CONDITIONS}, got {condition!r}")
    n_ticks = n_eval_ticks(exp_cfg, control_dt)
    table = keypoint_table(exp_cfg["_global"])
    limits = [float(v) for v in exp_cfg["base_command_limits"]]
    base_schedule = sc.make_base_velocity_schedule(
        n_ticks, control_dt, seed, limits=limits,
        intensity=float(exp_cfg["base"]["intensity"]),
        segment_duration_range=tuple(exp_cfg["base"]["segment_duration_range"]),
        zero=False)
    ee_schedule = make_ee_random_interval_schedule(
        n_ticks, control_dt, seed, table, tuple(exp_cfg["ee"]["interval_range_s"]))
    description = {
        "experiment": "E2", "condition": condition,
        "title": "Dynamic base velocity + dynamic full-pose EE targets",
        "base_mode": "random", "base_limits": limits,
        "base_intensity": float(exp_cfg["base"]["intensity"]),
        "base_segment_duration_range_s": list(exp_cfg["base"]["segment_duration_range"]),
        "base_generator": "core.scenario.make_base_velocity_schedule (E1)",
        "ee_mode": "random_interval",
        "ee_interval_range_s": list(exp_cfg["ee"]["interval_range_s"]),
        "ee_interpolation": "none",
        "ee_command_table": exp_cfg["_global"]["ee_command_table"],
        "ee_command_table_sha256": table.sha256,
        "n_eval_ticks": n_ticks, "seed": seed,
    }
    return PrecomputedScenario(base_schedule, ee_schedule, description)


# ==========================================================================
# E3: random discrete full-pose EE commands x base-motion intensity
# ==========================================================================
E3_POSE_SEQUENCE_LENGTH = 16   # enough for 27 s at the shortest (2 s) interval


def e3_conditions(exp_cfg: dict):
    """(condition_id, base_bound, update_interval_s or None): ``B{i}T{j}``."""
    out = []
    for bi, b in enumerate(exp_cfg["base"]["levels"]):
        for ti, t in enumerate(exp_cfg["ee"]["update_intervals_s"]):
            out.append((f"B{bi}T{ti}", float(b), None if t is None else float(t)))
    return out


def parse_e3_condition(condition: str, exp_cfg: dict):
    if not (len(condition) == 4 and condition[0] == "B" and condition[2] == "T"):
        raise ValueError(f"not an E3 condition id: {condition}")
    bi, ti = int(condition[1]), int(condition[3])
    t = exp_cfg["ee"]["update_intervals_s"][ti]
    return float(exp_cfg["base"]["levels"][bi]), (None if t is None else float(t))


_FEASIBLE_CACHE: dict = {}


def e3_feasible_rows(exp_cfg: dict) -> np.ndarray:
    import hashlib

    path = os.path.join(CONFIG_DIR, exp_cfg["ee"]["feasible_rows_file"])
    if path not in _FEASIBLE_CACHE:
        with open(path, "rb") as h:
            digest = hashlib.sha256(h.read()).hexdigest()
        want = exp_cfg["ee"].get("feasible_rows_sha256")
        if want and digest != want:
            raise ValueError(f"{path}: sha256 {digest} != frozen {want}")
        _FEASIBLE_CACHE[path] = np.load(path).astype(np.int64)
    return _FEASIBLE_CACHE[path]


def e3_base_schedule(exp_cfg: dict, bound: float, seed: int, n_ticks: int, control_dt: float):
    limits = np.asarray(exp_cfg["base_command_limits"], dtype=np.float64)
    if bound < 0.0 or np.any(bound > limits + 1e-12):
        raise ValueError(f"E3 base bound {bound} outside the supported range {limits.tolist()}")
    # limits = the bound itself, intensity 1: commands are exactly waveform * B.
    return sc.make_base_velocity_schedule(
        n_ticks, control_dt, seed,
        limits=(bound, bound, bound) if bound > 0.0 else tuple(limits),
        intensity=1.0,
        segment_duration_range=tuple(exp_cfg["base"]["segment_duration_range"]),
        zero=(bound == 0.0),
    )


class HoldFirstTargetScenario(PrecomputedScenario):
    """Settling-window commands: zero base velocity and the schedule's first EE
    target (the runner's settle hook), so the EE command does not jump at the
    start of the evaluation window."""

    def settle_commands(self, adapter, initial_ee_kp_plb):
        return np.zeros(3, dtype=np.float32), np.asarray(self.ee.commands[0], dtype=np.float32)


def build_e3(exp_cfg: dict, condition: str, seed: int, control_dt: float = 0.02):
    if exp_cfg.get("protocol") != "random_full_pose":
        raise ValueError("configs/E3.yaml is not the random-full-pose E3 protocol")
    bound, interval = parse_e3_condition(condition, exp_cfg)
    n_ticks = n_eval_ticks(exp_cfg, control_dt)
    table = keypoint_table(exp_cfg["_global"])
    rows = sc.e3_pose_sequence(seed, e3_feasible_rows(exp_cfg), E3_POSE_SEQUENCE_LENGTH)
    base_schedule = e3_base_schedule(exp_cfg, bound, seed, n_ticks, control_dt)
    ee_schedule = sc.make_ee_random_pose_schedule(n_ticks, control_dt, table, rows, interval)
    description = {
        "experiment": "E3",
        "protocol": "random_full_pose",
        "condition": condition,
        "base_bound": bound,
        "base_bound_units": "|vx|,|vy| m/s; |wz| rad/s (body frame)",
        "base_segment_duration_range_s": list(exp_cfg["base"]["segment_duration_range"]),
        "ee_update_interval_s": interval,
        "ee_level": "Fixed" if interval is None else f"{interval:g}s",
        "ee_interpolation": "none (sample-and-hold)",
        "ee_pose_table_rows_used": ee_schedule.row_indices.tolist(),
        "ee_settle_hold": "first target of the sequence (no jump at t = 3 s)",
        "ee_command_table_sha256": table.sha256,
        "n_eval_ticks": n_ticks,
        "seed": seed,
    }
    return HoldFirstTargetScenario(base_schedule, ee_schedule, description)


# ==========================================================================
# dispatch
# ==========================================================================
BUILDERS = {"E1": build_e1, "E2": build_e2, "E3": build_e3}


def all_conditions(experiment: str, exp_cfg: dict):
    if experiment == "E1":
        return list(E1_CONDITIONS)
    if experiment == "E2":
        return list(E2_CONDITIONS)
    if experiment == "E3":
        return [c[0] for c in e3_conditions(exp_cfg)]
    raise KeyError(experiment)


def build_scenario(experiment: str, exp_cfg: dict, condition: str, seed: int, control_dt=0.02):
    return BUILDERS[experiment](exp_cfg, condition, seed, control_dt)
