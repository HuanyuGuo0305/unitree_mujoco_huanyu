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
from dataclasses import dataclass, field

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
# E3 EE commands: smooth geometric trajectories at controlled speed
# ==========================================================================
@dataclass
class EETrajectorySpec:
    family: str  # "line" | "circle" | "semicircle"
    center: tuple[float, float, float]
    amplitude: float  # [m] half-length (line) or radius (circle/semicircle)
    speed_max: float  # [m/s] mean path speed at alpha_ee = 1
    orientation_keypoints: np.ndarray = field(default=None, repr=False)

    def period_s(self, alpha_ee: float) -> float:
        """Full-cycle period that realises mean path speed ``alpha_ee*speed_max``."""
        if alpha_ee <= 0.0:
            return float("inf")
        v = alpha_ee * self.speed_max
        a = self.amplitude
        if self.family == "line":
            return 4.0 * a / v  # mean speed of a*sin(2pi t/T) is 4a/T
        if self.family in ("circle", "semicircle"):
            return 2.0 * np.pi * a / v  # arc speed a*2pi/T
        raise ValueError(f"Unknown trajectory family: {self.family}")

    def position(self, t: np.ndarray, alpha_ee: float) -> np.ndarray:
        """Path position(s) in the PLB frame.

        All families share centre, amplitude and orientation protocol; only the
        time parameterisation changes with ``alpha_ee``. Velocity is continuous
        everywhere, including at the reversals of the line and semicircle.
        """
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        c = np.asarray(self.center, dtype=np.float64)
        a = float(self.amplitude)

        if alpha_ee <= 0.0:
            phase = np.zeros_like(t)
        else:
            phase = 2.0 * np.pi * t / self.period_s(alpha_ee)

        y_hat = np.array([0.0, 1.0, 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])

        if self.family == "line":
            return c[None, :] + a * np.sin(phase)[:, None] * y_hat[None, :]
        if self.family == "circle":
            return (
                c[None, :]
                + a * np.sin(phase)[:, None] * y_hat[None, :]
                + a * np.cos(phase)[:, None] * z_hat[None, :]
            )
        if self.family == "semicircle":
            theta = (np.pi / 2.0) * np.sin(phase)
            return (
                c[None, :]
                + a * np.sin(theta)[:, None] * y_hat[None, :]
                + a * np.cos(theta)[:, None] * z_hat[None, :]
            )
        raise ValueError(f"Unknown trajectory family: {self.family}")


def make_ee_trajectory_schedule(
    n_ticks: int,
    control_dt: float,
    spec: EETrajectorySpec,
    alpha_ee: float,
    table: KeypointTable | None = None,
    support_tolerance: float = 0.12,
) -> EEKeypointSchedule:
    """Continuous EE position trajectory with a fixed orientation protocol.

    ``alpha_ee = 0`` gives a stationary target at the path's ``t = 0`` point.
    Orientation is held at ``spec.orientation_keypoints``' rotation for the
    whole episode, identical for every family and every intensity, so the only
    variable across the E3 sweep is EE motion speed.
    """
    t = np.arange(n_ticks, dtype=np.float64) * float(control_dt)
    positions = spec.position(t, alpha_ee)

    if spec.orientation_keypoints is None:
        raise ValueError("EETrajectorySpec.orientation_keypoints must be resolved first")
    _, R_cmd = frames.keypoints_to_pose(spec.orientation_keypoints)

    commands = np.empty((n_ticks, 9), dtype=np.float32)
    for i in range(n_ticks):
        commands[i] = frames.pose_to_keypoints(positions[i], R_cmd)

    if table is not None:
        distances = table.support_distance(positions)
        if float(distances.max()) > support_tolerance:
            raise ValueError(
                f"E3 {spec.family} trajectory leaves the native command support: "
                f"max nearest-row distance {distances.max():.3f} m "
                f"> tolerance {support_tolerance} m"
            )

    return EEKeypointSchedule(
        commands=commands,
        switch_ticks=np.asarray([], dtype=np.int64),
        row_indices=np.asarray([], dtype=np.int64),
        mode=f"trajectory:{spec.family}:alpha{alpha_ee:g}",
    )


def resolve_trajectory_orientation(
    spec: EETrajectorySpec, table: KeypointTable
) -> EETrajectorySpec:
    """Pick the orientation protocol from the native table, deterministically.

    The orientation of the native command row whose position is closest to the
    path centre is used. This guarantees the held orientation is one the
    policies were actually trained to reach, and it is identical for every
    family, intensity, seed and baseline.
    """
    row = table.nearest_row(np.asarray(spec.center, dtype=np.float64))
    spec.orientation_keypoints = row
    return spec


# ==========================================================================
# E3 v2 EE commands: arc-length parameterised paths at a controlled speed
# ==========================================================================
# All three families share one start point P0 = c - a*y_hat (PLB), so a
# speed of 0 is the SAME fixed pose for every family. Geometry (PLB, plane x=c_x):
#   line        P0 -> c + a*y_hat, straight, length L = 2a, traversed back and forth
#   semicircle  upper half circle of radius a from P0 over c + a*z_hat to
#               c + a*y_hat, length L = pi*a, traversed back and forth
#   circle      full circle of radius a through P0 and c + a*z_hat, length
#               L = 2*pi*a, traversed in one direction
# Arc length s(t) follows the SAME speed law for every family:
#   * cruise at exactly the level speed v;
#   * start from rest with the half-cosine ramp v*sin(pi*t/tau);
#   * at every reversal of an open path, v*cos(pi*t/tau), i.e. speed
#     v -> 0 -> v with continuous velocity AND acceleration;
#   tau = pi*v/a_peak, so the peak tangential acceleration is a_peak at every
#   speed level and the ramp length is v**2/a_peak.
# The orientation is held constant (one native-table orientation).
E3V2_FAMILIES = ("line", "circle", "semicircle")


def e3v2_path_length(family: str, amplitude: float) -> float:
    if family == "line":
        return 2.0 * amplitude
    if family == "semicircle":
        return np.pi * amplitude
    if family == "circle":
        return 2.0 * np.pi * amplitude
    raise ValueError(f"Unknown E3 v2 trajectory family: {family}")


def e3v2_path_point(family: str, s: np.ndarray, center, amplitude: float) -> np.ndarray:
    """PLB position at arc length ``s`` (metres from P0) along the path."""
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    c = np.asarray(center, dtype=np.float64).reshape(3)
    a = float(amplitude)
    y_hat = np.array([0.0, 1.0, 0.0])
    z_hat = np.array([0.0, 0.0, 1.0])
    if family == "line":
        return c[None, :] + (s - a)[:, None] * y_hat[None, :]
    if family in ("semicircle", "circle"):
        theta = -np.pi / 2.0 + s / a
        return (c[None, :] + a * np.sin(theta)[:, None] * y_hat[None, :]
                + a * np.cos(theta)[:, None] * z_hat[None, :])
    raise ValueError(f"Unknown E3 v2 trajectory family: {family}")


def e3v2_arclength(t: np.ndarray, family: str, amplitude: float, speed: float,
                   a_peak: float) -> np.ndarray:
    """Arc length s(t) >= 0 along the path (open paths: position, not distance)."""
    t = np.asarray(t, dtype=np.float64)
    if speed <= 0.0:
        return np.zeros_like(t)
    v, ap = float(speed), float(a_peak)
    L = e3v2_path_length(family, amplitude)
    tau = np.pi * v / ap
    d_ramp = v * v / ap                     # distance covered by one half-cosine ramp

    def from_rest(u):
        """Distance travelled from rest after time u (start ramp, then cruise)."""
        u = np.asarray(u, dtype=np.float64)
        ramp = (v * tau / np.pi) * (1.0 - np.cos(np.pi * np.minimum(u, tau / 2) / tau))
        return np.where(u < tau / 2, ramp, d_ramp + v * (u - tau / 2))

    if family == "circle":
        return np.mod(from_rest(t), L)

    if 2.0 * d_ramp > L:
        raise ValueError(f"speed {v} too high for a {family} of length {L} at a_peak {ap}")
    # One leg = rest at one end -> rest at the other end.
    t_cruise = (L - 2.0 * d_ramp) / v
    T_leg = tau / 2 + t_cruise + tau / 2

    def leg(u):
        u = np.asarray(u, dtype=np.float64)
        first = from_rest(np.minimum(u, tau / 2 + t_cruise))
        u_dec = np.clip(u - (tau / 2 + t_cruise), 0.0, tau / 2)
        dec = (v * tau / np.pi) * np.sin(np.pi * u_dec / tau)
        return np.where(u <= tau / 2 + t_cruise, first, d_ramp + v * t_cruise + dec)

    k = np.floor(t / T_leg)
    u = t - k * T_leg
    s_leg = leg(u)
    forward = (k.astype(np.int64) % 2) == 0
    return np.where(forward, s_leg, L - s_leg)


def e3v2_mean_speed(family: str, amplitude: float, speed: float, a_peak: float,
                    duration_s: float, control_dt: float) -> float:
    if speed <= 0.0:
        return 0.0
    t = np.arange(int(round(duration_s / control_dt)) + 1) * control_dt
    s = e3v2_arclength(t, family, amplitude, speed, a_peak)
    ds = np.abs(np.diff(s))
    if family == "circle":
        L = e3v2_path_length(family, amplitude)
        ds = np.minimum(ds, L - ds)  # undo the mod-L wrap
    return float(ds.sum() / (t[-1] - t[0]))


def make_ee_arclength_schedule(
    n_ticks: int, control_dt: float, family: str, center, amplitude: float,
    speed: float, a_peak: float, orientation_keypoints: np.ndarray,
    table: KeypointTable | None = None, support_tolerance: float = 0.12,
) -> EEKeypointSchedule:
    """E3 v2 EE schedule: arc-length path, constant orientation, PLB frame."""
    t = np.arange(n_ticks, dtype=np.float64) * float(control_dt)
    s = e3v2_arclength(t, family, amplitude, speed, a_peak)
    positions = e3v2_path_point(family, s, center, amplitude)
    _, R_cmd = frames.keypoints_to_pose(orientation_keypoints)
    commands = np.empty((n_ticks, 9), dtype=np.float32)
    for i in range(n_ticks):
        commands[i] = frames.pose_to_keypoints(positions[i], R_cmd)
    if table is not None:
        L = e3v2_path_length(family, amplitude)
        whole = e3v2_path_point(family, np.linspace(0.0, L, 721), center, amplitude)
        dist = table.support_distance(whole)
        if float(dist.max()) > support_tolerance:
            raise ValueError(
                f"E3 v2 {family} path leaves the native command support: "
                f"{dist.max():.3f} m > {support_tolerance} m")
    return EEKeypointSchedule(
        commands=commands,
        switch_ticks=np.asarray([], dtype=np.int64),
        row_indices=np.asarray([], dtype=np.int64),
        mode=f"arclength:{family}:v{speed:g}",
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
