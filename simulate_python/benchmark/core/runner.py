"""Episode runner: drives one (experiment, baseline, condition, seed) episode.

The loop order reproduces each script's own convention exactly:

    for counter in 0, 1, 2, ...:
        adapter.pre_physics(sim_time)      # torques from the held targets
        mujoco.mj_step(m, d)
        sim_time += physics_dt
        recorder.record_physics(sim_time)
        if counter % decimation == 0:
            base_cmd, ee_cmd = scenario(...)   # external injection point
            adapter.policy_tick(base_cmd, ee_cmd, sim_time)
            recorder.record_policy(...)

Commands are injected only as *term values* before the observation vector is
assembled, so nothing is modified after the observation history and the
(ONNX-internal) normalizer have been applied.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from benchmark.controllers.registry import BASELINES, REPO_ROOT, build_adapter
from benchmark.core import frames
from benchmark.core.recorder import EpisodeRecorder


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


@dataclass
class EpisodeSpec:
    experiment: str
    baseline: str
    condition: str
    seed: int
    duration_s: float
    settle_s: float
    termination: dict = field(default_factory=dict)
    initial_state: dict = field(default_factory=dict)
    log_physics: bool = False
    notes: dict = field(default_factory=dict)


class PrecomputedScenario:
    """Base/EE command schedules fully determined before the episode starts."""

    kind = "precomputed"

    def __init__(self, base_schedule, ee_schedule, description: dict):
        self.base = base_schedule
        self.ee = ee_schedule
        self.description = description

    def n_ticks(self) -> int:
        return len(self.base.commands)

    def prepare(self, adapter, initial_ee_kp_plb):
        pass

    def settle_commands(self, adapter, initial_ee_kp_plb):
        """Commands held during the settling window, before scenario tick 0.

        Zero base velocity and the measured initial EE pose, i.e. the native
        ``reset(sample_first=False)`` path. Identical for every baseline
        because the initial state is identical.
        """
        return np.zeros(3, dtype=np.float32), np.asarray(initial_ee_kp_plb, dtype=np.float32)

    def command(self, scenario_tick: int, adapter):
        return self.base.at(scenario_tick), self.ee.at(scenario_tick), None

    def arrays(self) -> dict:
        out = {
            "base_commands": self.base.commands,
            "ee_commands_plb": self.ee.commands,
            "base_switch_ticks": self.base.switch_ticks,
            "base_segment_durations_s": self.base.segment_durations_s,
            "ee_switch_ticks": self.ee.switch_ticks,
            "ee_row_indices": self.ee.row_indices,
        }
        return out


class WorldFrameEEScenario:
    """E4: fixed world-frame EE target, converted to PLB every control update."""

    kind = "world_frame_ee"

    def __init__(self, base_schedule, world_target, description: dict,
                 table=None, support_tolerance: float = 0.12):
        self.base = base_schedule
        self.target = world_target
        self.description = description
        self.table = table
        self.support_tolerance = float(support_tolerance)
        self._issued: list[np.ndarray] = []
        self._support_distance: list[float] = []

    def n_ticks(self) -> int:
        return len(self.base.commands)

    def settle_commands(self, adapter, initial_ee_kp_plb):
        """Hold the PLB setpoint during settling so the arm converges onto it."""
        setpoint = self.target.plb_setpoint
        if setpoint is None:
            setpoint = initial_ee_kp_plb
        return np.zeros(3, dtype=np.float32), np.asarray(setpoint, dtype=np.float32)

    def prepare(self, adapter, initial_ee_kp_plb):
        pass

    def _lock_world_target(self, adapter):
        """Lock the world-frame target to the measured EE pose at the first
        evaluated tick (the arm has converged on the PLB setpoint by then)."""
        d = adapter.data
        ee_pos_w = d.xpos[adapter.ee_bid].copy().astype(np.float64)
        ee_rot_w = d.xmat[adapter.ee_bid].reshape(3, 3).copy().astype(np.float64)
        self.target.position_w = ee_pos_w
        self.target.rotation_w = ee_rot_w
        self.description["ee_world_target_position"] = ee_pos_w.tolist()

    def command(self, scenario_tick: int, adapter):
        d = adapter.data
        if self.target.position_w is None:
            self._lock_world_target(adapter)
        ee_cmd = self.target.command_for_state(d.qpos[0:3], d.qpos[3:7])
        self._issued.append(ee_cmd.copy())
        if self.table is not None:
            self._support_distance.append(
                float(self.table.support_distance(ee_cmd[0:3])[0])
            )
        world_info = np.concatenate(
            [
                np.asarray(self.target.position_w, dtype=np.float64).reshape(3),
                np.asarray(self.target.rotation_w, dtype=np.float64).reshape(9),
            ]
        )
        return self.base.at(scenario_tick), ee_cmd, world_info

    def support_report(self) -> dict:
        """How far the required PLB command strayed from the native command
        support. Reported, never hidden: an unreachable excursion shows up as a
        nonzero ``target_outside_support_fraction``."""
        if not self._support_distance:
            return {}
        d = np.asarray(self._support_distance)
        return {
            "target_support_distance_mean": float(d.mean()),
            "target_support_distance_max": float(d.max()),
            "target_outside_support_fraction": float(
                np.mean(d > self.support_tolerance)
            ),
        }

    def arrays(self) -> dict:
        return {
            "base_commands": self.base.commands,
            "ee_commands_plb": (
                np.stack(self._issued) if self._issued else np.zeros((0, 9), dtype=np.float32)
            ),
            "base_switch_ticks": self.base.switch_ticks,
            "base_segment_durations_s": self.base.segment_durations_s,
            "world_target_position": np.asarray(self.target.position_w, dtype=np.float64),
            "world_target_rotation": np.asarray(self.target.rotation_w, dtype=np.float64),
            "target_support_distance": np.asarray(self._support_distance, dtype=np.float64),
        }


def run_episode(spec: EpisodeSpec, scenario, out_dir: str, verbose: bool = False) -> dict:
    """Run one episode and write all artefacts to ``out_dir``."""
    t_wall = time.time()
    adapter = build_adapter(spec.baseline, verbose=verbose)
    model = adapter.build_model()
    data = mujoco.MjData(model)
    adapter.attach(model, data)
    adapter.set_initial_state()
    initial_noise = adapter.apply_initial_state_noise(spec.seed, spec.initial_state)

    initial_state = {
        "qpos": data.qpos.copy().tolist(),
        "base_z": float(data.qpos[2]),
        "randomization": initial_noise,
    }

    # The EE command held during settling is the measured initial EE pose,
    # matching the native reset(sample_first=False) path. Identical for every
    # baseline because the initial state is identical.
    initial_ee_kp_plb = frames.ee_keypoints_plb_from_state(
        data.xpos[adapter.ee_bid],
        data.xmat[adapter.ee_bid].reshape(3, 3),
        data.qpos[0:3],
        data.qpos[3:7],
        adapter.ee_ground_z,
        adapter.ee_kp_dx,
        adapter.ee_kp_dz,
    )
    scenario.prepare(adapter, initial_ee_kp_plb)

    settle_base, settle_ee = scenario.settle_commands(adapter, initial_ee_kp_plb)
    settle_base = np.asarray(settle_base, dtype=np.float32).reshape(3)
    settle_ee = np.asarray(settle_ee, dtype=np.float32).reshape(9)
    adapter.reset(settle_base, settle_ee)

    recorder = EpisodeRecorder(
        adapter, spec.settle_s, spec.duration_s, log_physics=spec.log_physics
    )

    physics_dt = adapter.simulation_dt
    decimation = adapter.control_decimation
    n_steps = int(round(spec.duration_s / physics_dt))
    settle_ticks = int(round(spec.settle_s / adapter.control_dt))

    sim_time = 0.0
    tick = 0
    for counter in range(n_steps):
        adapter.pre_physics(sim_time)
        mujoco.mj_step(model, data)
        sim_time += physics_dt
        recorder.record_physics(sim_time)

        if counter % decimation == 0:
            scenario_tick = tick - settle_ticks
            if scenario_tick < 0:
                base_cmd, ee_cmd, world_info = settle_base, settle_ee, None
            else:
                base_cmd, ee_cmd, world_info = scenario.command(scenario_tick, adapter)
            adapter.policy_tick(base_cmd, ee_cmd, sim_time)
            recorder.record_policy(
                sim_time, tick, scenario_tick, base_cmd, ee_cmd,
                ee_command_world=world_info, extra=adapter.diagnostics(),
            )
            tick += 1

        reason = recorder.check_termination(sim_time, spec.termination)
        if reason is not None:
            recorder.mark_terminated(reason, sim_time)
            break

    info = BASELINES[spec.baseline]
    metadata = {
        "experiment": spec.experiment,
        "baseline": spec.baseline,
        "baseline_framework": info["framework"],
        "baseline_actuation": info["actuation"],
        "deploy_script": info["deploy_script"],
        "condition": spec.condition,
        "seed": spec.seed,
        "duration_s": spec.duration_s,
        "settle_s": spec.settle_s,
        "eval_window_s": [spec.settle_s, spec.duration_s],
        "physics_dt": physics_dt,
        "control_dt": adapter.control_dt,
        "physics_hz": 1.0 / physics_dt,
        "control_hz": 1.0 / adapter.control_dt,
        "config_path": os.path.relpath(adapter.cfg_path, REPO_ROOT),
        "xml_path": os.path.relpath(adapter.xml_path, REPO_ROOT),
        "asset_sha256": adapter.asset_hashes,
        "policy_paths": {
            k: os.path.relpath(v, REPO_ROOT) for k, v in adapter.asset_paths.items()
        },
        "uan_model_path": os.path.relpath(adapter.uan_model_path, REPO_ROOT),
        "initial_state": initial_state,
        "scenario": scenario.description,
        "scenario_kind": scenario.kind,
        "command_frames": {
            "base_velocity": "base body frame (primary); yaw-only PLB frame also logged",
            "ee_pose": "PLB frame: origin=[base_x, base_y, ee_ground_z], orientation=yaw-only(base_quat)",
            "ee_representation": "9-D keypoints (kp0, kp0+R*[0.3,0,0], kp0+R*[0,0,0.3])",
        },
        "base_command_limits_source": "inferred_hl_transport_frozen_wbc_bounds",
        "termination_config": spec.termination,
        "git_commit": git_commit(),
        "policy_ticks": adapter.policy_ticks,
        "physics_steps": adapter.physics_steps,
        "uan_calls": adapter.uan.n_calls,
        "notes": spec.notes,
    }
    extra_metrics = (
        scenario.support_report() if hasattr(scenario, "support_report") else {}
    )
    metrics = recorder.write(
        out_dir, metadata, scenario.arrays(), extra_metrics=extra_metrics
    )
    metrics["wall_time_s"] = time.time() - t_wall

    # A run is never declared complete just because the process did not crash.
    _validate_episode(adapter, recorder, scenario, spec, settle_ticks)
    return metrics


def _validate_episode(adapter, recorder, scenario, spec, settle_ticks):
    """Post-hoc consistency checks on the artefacts, not just on exit status."""
    expected_uan = adapter.physics_steps
    if adapter.uan.n_calls != expected_uan:
        raise AssertionError(
            f"UAN called {adapter.uan.n_calls} times for {expected_uan} physics steps"
        )
    if not recorder.terminated:
        expected_ticks = int(round(spec.duration_s / adapter.control_dt))
        if abs(adapter.policy_ticks - expected_ticks) > 1:
            raise AssertionError(
                f"{adapter.policy_ticks} policy ticks for {expected_ticks} expected"
            )
    # Commands actually fed to the controller must equal the schedule.
    if scenario.kind == "precomputed":
        for row in recorder.rows:
            k = row["scenario_tick"]
            if k < 0:
                continue
            expected_base = scenario.base.at(k)
            for i, key in enumerate(("vx_cmd", "vy_cmd", "wz_cmd")):
                if abs(row[key] - float(expected_base[i])) > 1e-6:
                    raise AssertionError(
                        f"base command mismatch at scenario tick {k}: "
                        f"{row[key]} vs {expected_base[i]}"
                    )
            expected_ee = scenario.ee.at(k)
            for i in range(9):
                if abs(row[f"ee_cmd_kp{i}"] - float(expected_ee[i])) > 1e-6:
                    raise AssertionError(f"EE command mismatch at scenario tick {k}")
