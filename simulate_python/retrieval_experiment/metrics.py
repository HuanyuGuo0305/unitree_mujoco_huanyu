"""Task state machine, episode classification and statistics for E5.

``TaskStateMachine.update`` is a single-environment port of the Isaac Lab
retrieval environment's end-of-step bookkeeping, in the same order as
``B2WZ1HLRetrievalUANABSEnv.step``:

    _update_task_buffers()      -> inputs of ``update`` (grasp error, contacts, ...)
    _update_grasp_stage()       -> deployable proxy + simulation-confirmed grasp
    _update_curriculum_stage()  -> S0 / S1 / S2 with hysteresis, S2 grasp-loss count
    _update_retrieval_success() -> latched 3-step retrieval success
    _get_dones()                -> grasp-error OOB counter, terminations, timeout

The Stage-2 force-close bootstrap is NOT part of this port: it is a training-only
execution override (disabled after PPO iteration 500) and the evaluation never
re-enables it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FAILURE_REASONS = (
    "success",
    "grasp_not_achieved",            # timeout without ever reaching Stage 2
    "timeout",                       # Stage 2 reached, retrieval not completed in time
    "grasp_lost_after_confirmation", # S2 grasp lost for 2 consecutive HL steps
    "base_contact_failure",          # base_link net contact force > 1 N
    "grasp_error_out_of_bounds",     # grasp error > 5 m for 3 consecutive HL steps
    "numerical_failure",             # non-finite state / MuJoCo instability
)


@dataclass
class StepInputs:
    grasp_error: float          # ||object center - gripper center|| (true, noiseless)
    gripper_q: float            # measured jointGripper angle [rad]
    close_executed: bool        # binary close command executed during this HL step
    stator_contact: bool        # gripperStator <-> cube net force > threshold
    mover_contact: bool         # gripperMover <-> cube net force > threshold
    retrieval_error: float      # ||object center - retrieval target|| (world)
    base_contact: bool          # base_link net contact force > threshold


class TaskStateMachine:
    def __init__(self, task_cfg: dict, gripper_open_pos: float, max_episode_hl_steps: int):
        c = task_cfg
        if bool(c.get("stage2_force_gripper_close_enabled", False)):
            raise ValueError("Stage-2 force-close is a training-only bootstrap; it must be disabled for evaluation.")
        self.s01_thr = float(c["curriculum_s0_to_s1_grasp_error_threshold"])
        self.s01_n = int(c["curriculum_s0_to_s1_consecutive_steps"])
        self.s10_thr = float(c["curriculum_s1_to_s0_grasp_error_threshold"])
        self.s10_n = int(c["curriculum_s1_to_s0_consecutive_steps"])
        self.s2_lost_n = int(c["curriculum_s2_grasp_lost_consecutive_steps"])
        self.proxy_thr = float(c["grasp_proxy_error_threshold"])
        self.not_closed_thr = float(c["gripper_not_fully_closed_angle_threshold"])
        self.hold_thr = float(c["gripper_angle_hold_threshold"])
        self.proxy_enter_n = int(c["grasp_proxy_enter_steps"])
        self.proxy_exit_n = int(c["grasp_proxy_exit_steps"])
        self.succ_thr = float(c["retrieval_success_error_threshold"])
        self.succ_n = int(c["retrieval_success_consecutive_steps"])
        self.oob_thr = float(c["grasp_error_termination_threshold"])
        self.oob_n = int(c["grasp_error_termination_consecutive_steps"])
        self.gripper_open_pos = float(gripper_open_pos)
        self.max_episode_hl_steps = int(max_episode_hl_steps)
        self.reset()

    def reset(self, gripper_q0: float | None = None):
        """_reset_idx: all counters/flags cleared; prev gripper angle = measured reset angle."""
        self.proxy = False
        self.proxy_candidate = False
        self.proxy_enter = 0
        self.proxy_exit = 0
        self.stage2_now = False                 # env._stage2 (proxy & dual contact)
        self.stage = 0                          # env._curriculum_stage
        self.c01 = 0
        self.c10 = 0
        self.s2_lost = 0
        self.s2_lost_terminated = False
        self.succ_count = 0
        self.success = False                    # latched _episode_retrieval_success
        self.oob_steps = 0
        self.oob_terminated = False
        self.prev_gripper_q = self.gripper_open_pos if gripper_q0 is None else float(gripper_q0)
        self.hl_steps = 0                       # env.episode_length_buf
        self.events: dict = {}

    def update(self, x: StepInputs) -> dict:
        self.hl_steps += 1
        ev = {"s0_to_s1": False, "s1_to_s0": False, "s1_to_s2": False, "newly_successful": False}

        # ---------------- _update_grasp_stage ----------------
        dual_contact = bool(x.stator_contact and x.mover_contact)
        angle_delta = abs(float(x.gripper_q) - self.prev_gripper_q)
        not_fully_closed = float(x.gripper_q) < self.not_closed_thr
        holding = angle_delta < self.hold_thr
        cand = bool(x.close_executed and (x.grasp_error < self.proxy_thr) and not_fully_closed and holding)
        self.proxy_candidate = cand
        if not self.proxy:
            self.proxy_enter = self.proxy_enter + 1 if cand else 0
            if self.proxy_enter >= self.proxy_enter_n:
                self.proxy = True
                self.proxy_enter = 0
        else:
            self.proxy_enter = 0
        # NOTE: evaluated on the UPDATED proxy, exactly as the vectorized code
        # (``confirmed = self._grasp_confidence_proxy`` after the enter branch).
        if self.proxy:
            self.proxy_exit = self.proxy_exit + 1 if not cand else 0
            if self.proxy_exit >= self.proxy_exit_n:
                self.proxy = False
                self.proxy_exit = 0
        else:
            self.proxy_exit = 0
        self.stage2_now = bool(self.proxy and dual_contact)
        self.prev_gripper_q = float(x.gripper_q)

        # ---------------- _update_curriculum_stage ----------------
        if self.stage == 0:
            self.c01 = self.c01 + 1 if x.grasp_error < self.s01_thr else 0
            if self.c01 >= self.s01_n:
                self.stage = 1
                ev["s0_to_s1"] = True
                self.c01 = 0
        else:
            self.c01 = 0
        if self.stage == 1:
            self.c10 = self.c10 + 1 if x.grasp_error > self.s10_thr else 0
            if self.c10 >= self.s10_n:
                self.stage = 0
                ev["s1_to_s0"] = True
                self.c10 = 0
        else:
            self.c10 = 0
        if self.stage == 1 and self.stage2_now:
            self.stage = 2
            ev["s1_to_s2"] = True
        if self.stage == 2:
            self.s2_lost = self.s2_lost + 1 if not self.stage2_now else 0
        else:
            self.s2_lost = 0
        self.s2_lost_terminated = self.s2_lost >= self.s2_lost_n

        # ---------------- _update_retrieval_success ----------------
        success_now = (self.stage == 2) and (x.retrieval_error < self.succ_thr)
        if not self.success:
            self.succ_count = self.succ_count + 1 if success_now else 0
            if self.succ_count >= self.succ_n:
                self.success = True
                ev["newly_successful"] = True

        # ---------------- _get_dones ----------------
        self.oob_steps = self.oob_steps + 1 if x.grasp_error > self.oob_thr else 0
        self.oob_terminated = self.oob_steps >= self.oob_n
        terminated = bool(x.base_contact or self.oob_terminated or self.s2_lost_terminated)
        timeout = self.hl_steps >= self.max_episode_hl_steps - 1

        ev.update(
            dual_contact=dual_contact, proxy=self.proxy, proxy_candidate=cand, stage2_now=self.stage2_now,
            stage=self.stage, success_now=success_now, success=self.success,
            terminated=terminated, timeout=timeout, base_contact=bool(x.base_contact),
            oob_terminated=self.oob_terminated, s2_lost_terminated=self.s2_lost_terminated,
            not_fully_closed=not_fully_closed, angle_holding=holding,
        )
        return ev


def classify_termination(ev: dict, ever_stage2: bool) -> str | None:
    """Failure reason for a step on which the episode ends without success, else None."""
    if ev["base_contact"]:
        return "base_contact_failure"
    if ev["s2_lost_terminated"]:
        return "grasp_lost_after_confirmation"
    if ev["oob_terminated"]:
        return "grasp_error_out_of_bounds"
    if ev["timeout"]:
        return "timeout" if ever_stage2 else "grasp_not_achieved"
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (returned as fractions)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_std(x) -> tuple[float, float, int]:
    """Mean and sample standard deviation (ddof=1); NaN when undefined."""
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else float("nan"), int(a.size)
