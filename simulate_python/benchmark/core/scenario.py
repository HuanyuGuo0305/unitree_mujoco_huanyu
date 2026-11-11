"""Deterministic, time-indexed command generation.

A scenario precomputes the *entire* command schedule from a seed before the
simulation starts, so that

  * the identical schedule can be handed to every baseline,
  * the schedule can be written to disk before any physics runs,
  * reruns with the same seed reproduce the targets bit-for-bit.

Tick indexing
-------------
Scenario tick ``k = 0`` is the first *evaluated* policy tick, i.e. the tick at
simulation time ``settle_s``. Before that the runner holds
``base_cmd = 0`` and ``ee_cmd = <measured initial EE keypoints>``, matching the
native ``reset(sample_first=False)`` path.

E1 EE resampling reproduces the native sample-and-hold counter exactly
(see interface_audit.md 4.5): command index ``= (k + 1) // cycle_steps``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from benchmark.core import frames


# ==========================================================================
# base velocity
# ==========================================================================
@dataclass
class BaseVelocitySchedule:
    """Piecewise-constant base velocity command, one row per policy tick."""

    commands: np.ndarray  # (n_ticks, 3) float32  [vx, vy, wz]
    switch_ticks: np.ndarray  # (n_segments,) int, tick at which each segment starts
    segment_durations_s: np.ndarray  # (n_segments,) float
    waveform: np.ndarray  # (n_ticks, 3) float32, the unscaled [-1,1] waveform
    intensity: float = 1.0
    limits: tuple[float, float, float] = (0.6, 0.6, 0.6)

    def at(self, tick: int) -> np.ndarray:
        return self.commands[min(tick, len(self.commands) - 1)]


def make_base_velocity_schedule(
    n_ticks: int,
    control_dt: float,
    seed: int,
    limits=(0.6, 0.6, 0.6),
    intensity: float = 1.0,
    segment_duration_range: tuple[float, float] = (2.0, 4.0),
    zero: bool = False,
) -> BaseVelocitySchedule:
    """Seeded piecewise-random base velocity commands.

    ``intensity`` scales the *same* waveform (for a given seed), so comparing
    intensities changes only the amplitude and nothing else. ``zero=True``
    produces an all-zero command while keeping the waveform for reference.
    """
    limits = np.asarray(limits, dtype=np.float64).reshape(3)
    rng = np.random.default_rng(_mix_seed(seed, "base_velocity"))

    waveform = np.zeros((n_ticks, 3), dtype=np.float32)
    switch_ticks: list[int] = []
    durations: list[float] = []

    tick = 0
    lo, hi = float(segment_duration_range[0]), float(segment_duration_range[1])
    while tick < n_ticks:
        duration = float(rng.uniform(lo, hi))
        seg_ticks = max(1, int(round(duration / control_dt)))
        # Unit-normalised command in [-1, 1]^3; scaled by limits * intensity.
        unit = rng.uniform(-1.0, 1.0, size=3)
        end = min(tick + seg_ticks, n_ticks)
        waveform[tick:end] = unit.astype(np.float32)
        switch_ticks.append(tick)
        durations.append(seg_ticks * control_dt)
        tick = end

    scale = (limits * float(intensity)).astype(np.float32)
    commands = np.zeros_like(waveform) if zero else (waveform * scale).astype(np.float32)

    return BaseVelocitySchedule(
        commands=commands,
        switch_ticks=np.asarray(switch_ticks, dtype=np.int64),
        segment_durations_s=np.asarray(durations, dtype=np.float64),
        waveform=waveform,
        intensity=0.0 if zero else float(intensity),
        limits=tuple(float(v) for v in limits),
    )


# ==========================================================================
# EE command table
# ==========================================================================
class KeypointTable:
    """The native presampled reachable PLB keypoint table."""

    def __init__(self, path: str):
        arr = np.load(path, allow_pickle=False).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9 or not np.isfinite(arr).all():
            raise ValueError(f"Expected finite (N,9) keypoint table, got {arr.shape}")
        self.path = str(path)
        self.table = arr
        self.n_rows = int(arr.shape[0])
        with open(path, "rb") as handle:
            self.sha256 = hashlib.sha256(handle.read()).hexdigest()
        self._tree = None

    def row(self, index: int) -> np.ndarray:
        return self.table[int(index) % self.n_rows].copy()

    @property
    def positions(self) -> np.ndarray:
        return self.table[:, 0:3]

    def support_distance(self, positions: np.ndarray) -> np.ndarray:
        """Distance from each query position to the nearest native command row."""
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.positions.astype(np.float64))
        q = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        return self._tree.query(q)[0]

    def nearest_row(self, position: np.ndarray) -> np.ndarray:
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.positions.astype(np.float64))
        _, idx = self._tree.query(np.asarray(position, dtype=np.float64).reshape(3))
        return self.table[int(idx)].copy()


# ==========================================================================
# E1 EE commands: discrete sample-and-hold, no interpolation
# ==========================================================================
@dataclass
class EEKeypointSchedule:
    commands: np.ndarray  # (n_ticks, 9) float32, PLB keypoints
    switch_ticks: np.ndarray  # ticks at which the command changes
    row_indices: np.ndarray  # table row used by each segment
    mode: str = "random"

    def at(self, tick: int) -> np.ndarray:
        return self.commands[min(tick, len(self.commands) - 1)]


def make_ee_random_schedule(
    n_ticks: int,
    control_dt: float,
    seed: int,
    table: KeypointTable,
    cycle_duration_s: float = 5.0,
) -> EEKeypointSchedule:
    """Discrete 6-D EE pose targets, resampled exactly every ``cycle_duration_s``.

    Reproduces the native ``PresampledKeypointsDirectCommandPLBSim`` counter:
    the command index at tick ``k`` is ``(k + 1) // cycle_steps``. There is no
    interpolation, ramp, smoothing or trajectory generation of any kind -- the
    command is piecewise constant with hard switches, exactly as in training
    and in the validated Sim2Sim scripts.
    """
    if cycle_duration_s <= 0.0:
        raise ValueError(f"cycle_duration_s must be > 0, got {cycle_duration_s}")
    cycle_steps = max(1, int(round(cycle_duration_s / control_dt)))
    rng = np.random.default_rng(_mix_seed(seed, "ee_random"))

    n_segments = (n_ticks + 1) // cycle_steps + 1
    row_indices = rng.integers(0, table.n_rows, size=n_segments)

    ticks = np.arange(n_ticks, dtype=np.int64)
    segment_of_tick = (ticks + 1) // cycle_steps
    commands = table.table[row_indices[segment_of_tick]].copy()

    switch = np.nonzero(np.diff(segment_of_tick) != 0)[0] + 1
    return EEKeypointSchedule(
        commands=commands.astype(np.float32),
        switch_ticks=switch.astype(np.int64),
        row_indices=row_indices.astype(np.int64),
        mode="random",
    )


def make_ee_fixed_schedule(
    n_ticks: int, seed: int, table: KeypointTable
) -> EEKeypointSchedule:
    """One fixed reachable EE pose target, held for the whole episode.

    The target is fixed in the native PLB command frame (not the world frame),
    which is the frame the policies and the IK were built around. The arm is
    NOT frozen: the controller keeps tracking the target while the base moves.
    """
    rng = np.random.default_rng(_mix_seed(seed, "ee_fixed"))
    index = int(rng.integers(0, table.n_rows))
    commands = np.repeat(table.row(index)[None, :], n_ticks, axis=0)
    return EEKeypointSchedule(
        commands=commands.astype(np.float32),
        switch_ticks=np.asarray([], dtype=np.int64),
        row_indices=np.asarray([index], dtype=np.int64),
        mode="fixed",
    )


# ==========================================================================
# E3: random discrete full-pose EE commands (sample-and-hold)
# ==========================================================================
# Each seed owns ONE ordered sequence of poses drawn uniformly, with
# replacement, from the frozen feasible subset of the native training table
# (its own RNG stream, independent of the base-command stream). Every EE
# update-interval level consumes a prefix of that same sequence; the "Fixed"
# level holds pose 0. The command switches directly (no interpolation) every
# ``interval_s`` seconds from scenario tick 0 (t = 3 s): pose 0 is already
# commanded during settling, so the first switch, to pose 1, is at tick
# ``steps``.
def e3_pose_sequence(seed: int, feasible_rows: np.ndarray, length: int) -> np.ndarray:
    rng = np.random.default_rng(_mix_seed(seed, "e3_random_pose"))
    rows = np.asarray(feasible_rows, dtype=np.int64)
    return rows[rng.integers(0, rows.size, size=int(length))]


def make_ee_random_pose_schedule(
    n_ticks: int, control_dt: float, table: KeypointTable, row_sequence: np.ndarray,
    interval_s: float | None,
) -> EEKeypointSchedule:
    seq = np.asarray(row_sequence, dtype=np.int64)
    if interval_s is None:
        idx = np.zeros(n_ticks, dtype=np.int64)
        switch = np.asarray([], dtype=np.int64)
        used = seq[:1]
    else:
        steps = int(round(float(interval_s) / control_dt))
        if steps <= 0 or abs(steps * control_dt - float(interval_s)) > 1e-9:
            raise ValueError(f"interval {interval_s} s is not a whole number of ticks")
        idx = np.arange(n_ticks, dtype=np.int64) // steps
        if idx[-1] >= seq.size:
            raise ValueError("pose sequence too short for this interval")
        switch = np.arange(steps, n_ticks, steps, dtype=np.int64)
        used = seq[: idx[-1] + 1]
    commands = table.table[seq[idx]].copy()
    return EEKeypointSchedule(
        commands=commands.astype(np.float32), switch_ticks=switch,
        row_indices=used.astype(np.int64),
        mode=f"random_pose:{'fixed' if interval_s is None else f'{interval_s:g}s'}",
    )


# ==========================================================================
# E4 world-frame EE stabilisation
# ==========================================================================
@dataclass
class WorldFrameEETarget:
    """Fixed world-frame EE pose, converted to PLB keypoints every tick.

    ``plb_setpoint`` is the native 9-D command held during the settling window
    so the arm converges onto a well-centred, demonstrably reachable pose
    BEFORE the world-frame target is locked. The world target is then taken
    from the measured EE pose at the first evaluated tick, which makes it
    reachable by construction and identical in definition for every baseline.
    """

    position_w: np.ndarray = None
    rotation_w: np.ndarray = None
    ground_z: float = 0.0
    plb_setpoint: np.ndarray = None

    def command_for_state(
        self, base_pos_w: np.ndarray, base_quat_w: np.ndarray
    ) -> np.ndarray:
        return frames.world_pose_to_plb_keypoints(
            self.position_w,
            self.rotation_w,
            base_pos_w,
            base_quat_w,
            ground_z=self.ground_z,
        )


# ==========================================================================
# helpers
# ==========================================================================
def _mix_seed(seed: int, label: str) -> int:
    """Deterministic, collision-resistant sub-seed per command stream.

    Using a separate stream per signal means (for example) changing the EE
    condition never perturbs the base velocity schedule for the same seed,
    which is what makes T1/T3 and T2/T4 paired comparisons valid.
    """
    digest = hashlib.sha256(f"{int(seed)}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")
