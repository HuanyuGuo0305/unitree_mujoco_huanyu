"""Physical grasp retention from the 250-Hz log -> One-Shot Success (offline, independent of the
training Stage-2 state machine).

Definition (E5, revision 3)
---------------------------
A one-shot successful retrieval is a task-successful episode in which, after the FIRST physical
acquisition of the cube, the robot continuously retains the cube until task completion, without
releasing and re-grasping it.

Signals (one sample per 4-ms physics step, ``ph_*`` arrays written by simulator.run_episode):
  F_s, F_m   net contact force cube<->gripperStator / cube<->gripperMover (all collision geoms of the body)
  q, qd      jaw angle / velocity;  rel  cube centre in the gripperStator frame
  pinch      load-bearing pinch: min(F_s, F_m) >= f_hold = 10 N. A cube is RETAINED only while both jaws
             carry load: supporting the heaviest E5 cube (0.259 kg) under gravity plus 2 g of extra
             acceleration at the lowest training jaw friction (mu = 0.4) needs <= 9.5 N per jaw, while a
             settled grasp squeezes with ~450 N (p0.1 of held samples: 263 N). Contact above the
             training sensor threshold (0.3 N) but below f_hold -- touching, incidental single-jaw
             contact, a cube sliding loosely along the jaws -- is NOT retention.

Acquisition (settled force closure): the first time a window of ``acquisition_window_s`` (one HL
period, 0.1 s) is entirely pinched, the jaw angle varies by <= ``acquisition_max_jaw_range_deg``
(the jaw is stalled on the cube; an unobstructed jaw closes at up to 2 rad/s = 11 deg per 0.1 s) and
the cube moves by <= ``acquisition_max_rel_disp_m`` relative to the stator (it moves rigidly with the
gripper). The acquisition time is the END of that window. Pinches that do not settle -- the jaw keeps
closing while the cube is squeezed out along the jaws -- are recorded as acquisition attempts, not as
acquisitions.

Release: after an acquisition, every maximal interval without pinch is classified. It is a RELEASE if
  * it lasts >= ``release_min_gap_s`` (40 ms; official run: all interruptions of kept grasps last
    <= 32 ms, all grip losses >= 56 ms; a free cube falls 7.8 mm in 40 ms), or
  * the cube moves relative to the stator by >= ``release_rel_disp_fraction_of_side`` x side (its
    centre has left the seat), measured from the last pinched sample to any sample of the gap or the
    first sample after it, or
  * the unblocked jaw closes by >= ``release_jaw_closing_deg`` during the gap (5 deg = 7.5 mm of pad
    travel at the 86-mm pad radius; kept grasps: <= 4.2 deg).
Otherwise it is contact FLICKER. Single-jaw contact does not count as retention (pinch is required),
and the 0.10-m grasp region of the training detector plays no role.

Re-acquisition: after a release, the next settled force-closure window (same acquisition rule).

one_shot_success := task_success AND acquisition before the success instant AND no release between
the first acquisition and the success instant (success instant = end of the HL step at which the
training retrieval-success condition latched; the log ends there).
A STRICT variant (diagnostic) treats any pinch lasting a full window, settled or not, as an acquisition.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {
    "contact_force_threshold": 0.30,          # training contact sensor (touch diagnostics only)
    "hold_force_threshold": 10.0,             # load-bearing pinch [N]
    "acquisition_window_s": 0.10,
    "acquisition_max_jaw_range_deg": 1.0,
    "acquisition_max_rel_disp_m": 0.002,
    "release_min_gap_s": 0.04,
    "release_rel_disp_fraction_of_side": 0.5,
    "release_jaw_closing_deg": 5.0,
    "attempt_min_pinch_s": 0.02,
}


def _segments(mask: np.ndarray):
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def _window_ok(pinch, q, rel, start, w, max_dq, max_dp, settled=True):
    seg = slice(start, start + w)
    if start + w > len(pinch) or not pinch[seg].all():
        return False
    if not settled:
        return True
    if q[seg].max() - q[seg].min() > max_dq:
        return False
    r = rel[seg]
    return float(np.linalg.norm(r.max(0) - r.min(0))) <= max_dp


def _first_acquisition(pinch, q, rel, start, w, max_dq, max_dp, settled=True):
    """First index i >= start such that window [i, i+w) qualifies; returns window END index or None."""
    for a, b in _segments(pinch[start:]):
        a += start
        b += start
        for i in range(a, b - w + 1):
            if _window_ok(pinch, q, rel, i, w, max_dq, max_dp, settled):
                return i + w
    return None


def analyze_retention(ph: dict, side: float, success: bool, cfg: dict | None = None, settled: bool = True) -> dict:
    """``ph``: dict with ph_f_stator, ph_f_mover, ph_q, ph_rel_pos, ph_cmd, ph_dt (arrays from the npz)."""
    c = dict(DEFAULTS, **(cfg or {}))
    dt = float(ph["ph_dt"])
    fs, fm = np.asarray(ph["ph_f_stator"], dtype=float), np.asarray(ph["ph_f_mover"], dtype=float)
    fh = float(c["hold_force_threshold"])
    pinch = (fs >= fh) & (fm >= fh)
    touch = (fs > float(c["contact_force_threshold"])) | (fm > float(c["contact_force_threshold"]))
    q = np.asarray(ph["ph_q"], dtype=float)
    rel = np.asarray(ph["ph_rel_pos"], dtype=float)
    cmd = np.asarray(ph["ph_cmd"])
    n = len(pinch)
    w = int(round(float(c["acquisition_window_s"]) / dt))
    max_dq = np.radians(float(c["acquisition_max_jaw_range_deg"]))
    max_dp = float(c["acquisition_max_rel_disp_m"])
    gap_min = int(round(float(c["release_min_gap_s"]) / dt))
    disp_rel = float(c["release_rel_disp_fraction_of_side"]) * float(side)
    dq_rel = np.radians(float(c["release_jaw_closing_deg"]))

    acq = _first_acquisition(pinch, q, rel, 0, w, max_dq, max_dp, settled)
    out = {"acquired": acq is not None, "first_acquisition_s": acq * dt if acq is not None else float("nan"),
           "releases": 0, "release_times_s": [], "reacquisitions": 0, "reacquisition_times_s": [],
           "flicker_gaps": 0, "flicker_max_ms": 0.0, "retained_at_end": False,
           "release_gap_ms": [], "release_disp_mm": [], "release_jaw_closing_deg": [],
           "touch_without_pinch_after_release_s": 0.0,
           "acquisition_jaw_angle_deg": float("nan"),
           "gaps": [],                     # diagnostics: every post-acquisition interruption (ms, mm, deg, release?)
           "attempts_before_acquisition": 0, "close_to_open_cmd_after_acquisition": 0,
           "close_to_open_cmd_while_held": 0, "close_to_open_cmd_while_held_harmless": 0}
    att_min = int(round(float(c["attempt_min_pinch_s"]) / dt))
    end_attempts = acq if acq is not None else n
    out["attempts_before_acquisition"] = sum(1 for a, b in _segments(pinch[:end_attempts]) if b - a >= att_min
                                             and not (acq is not None and b == end_attempts))
    if acq is None:
        out["one_shot_success"] = False
        return out
    out["acquisition_jaw_angle_deg"] = float(np.degrees(q[acq - 1]))

    held = True              # state machine over the post-acquisition log
    i = acq
    released_at = []
    while i < n:
        if held:
            if pinch[i]:
                i += 1
                continue
            a = i                                        # gap start
            b = a
            while b < n and not pinch[b]:
                b += 1
            ref = rel[a - 1]
            upto = min(b + 1, n)                         # include the first re-pinched sample
            disp = float(np.max(np.linalg.norm(rel[a:upto] - ref, axis=1)))
            closing = float(q[a:upto].max() - q[a - 1])
            is_release = (b - a) >= gap_min or disp >= disp_rel or closing >= dq_rel
            out["gaps"].append((1000.0 * (b - a) * dt, 1000.0 * disp, float(np.degrees(closing)), bool(is_release), b >= n))
            if is_release:
                out["releases"] += 1
                out["release_times_s"].append(a * dt)
                out["release_gap_ms"].append(1000.0 * (b - a) * dt)
                out["release_disp_mm"].append(1000.0 * disp)
                out["release_jaw_closing_deg"].append(float(np.degrees(closing)))
                released_at.append(a)
                held = False
                i = a
            else:
                out["flicker_gaps"] += 1
                out["flicker_max_ms"] = max(out["flicker_max_ms"], 1000.0 * (b - a) * dt)
                i = b
        else:
            nxt = _first_acquisition(pinch, q, rel, i, w, max_dq, max_dp, settled)
            if nxt is None:
                break
            out["reacquisitions"] += 1
            out["reacquisition_times_s"].append(nxt * dt)
            held = True
            i = nxt
    out["retained_at_end"] = bool(held)
    # Held/released state per sample (for command and touch diagnostics).
    state = np.zeros(n, dtype=bool)
    edges = [acq] + [int(round(t / dt)) for t in out["release_times_s"]]
    reacq = [int(round(t / dt)) for t in out["reacquisition_times_s"]]
    hold_starts = [acq] + reacq
    hold_ends = [int(round(t / dt)) for t in out["release_times_s"]] + [n]
    for a0, b0 in zip(hold_starts, hold_ends):
        state[a0:b0] = True
    del edges
    trans = (np.nonzero((cmd[:-1] > 0) & (cmd[1:] < 0))[0] + 1).tolist()
    while_held = [t for t in trans if t >= acq and state[t - 1]]
    out["close_to_open_cmd_after_acquisition"] = int(sum(1 for t in trans if t >= acq))
    out["close_to_open_cmd_while_held"] = int(len(while_held))
    # harmless: the cube is still held 0.5 s after the command (no release in between)
    out["close_to_open_cmd_while_held_harmless"] = int(sum(1 for t in while_held
                                                            if not any(t <= r <= t + 5 * w for r in released_at)))
    out["touch_without_pinch_after_release_s"] = float(np.sum(touch[acq:] & ~state[acq:]) * dt)
    out["one_shot_success"] = bool(success and out["releases"] == 0 and held)
    return out


def from_npz(z, success: bool, cfg: dict | None = None, settled: bool = True) -> dict:
    ph = {k: z[k] for k in ("ph_f_stator", "ph_f_mover", "ph_q", "ph_rel_pos", "ph_cmd", "ph_dt")}
    return analyze_retention(ph, float(z["cube_side"]), success, cfg, settled)
