"""One E4 episode with the exact Isaac Lab step ordering.

    reset (refs zero / measured HIGH hold, puck spawned on the real disk)
    repeat:
        obs_t(ref_t, a_{t-1}^exec) -> method decodes a_t against ref_t
        5 WBC decisions x 5 physics steps (UAN at 250 Hz)
        contact / base-contact / timeout checks on the completed transition
        terminate, or advance references -> obs_{t+1}

Time series are recorded at the end of every WBC step (50 Hz) and every
high-level step (10 Hz). Nothing is filtered: failed episodes keep their
partial trajectories.
"""
from __future__ import annotations

import numpy as np

from .metrics import episode_metrics

LL_FIELDS = (
    "t", "hl_step", "active", "base_ref", "base_vel_b", "base_pos", "puck_disk",
    "puck_tilt_deg", "kp0_ref", "rot_ref", "kp0_cmd", "rot_cmd", "ee_kp0_actual",
)
HL_FIELDS = (
    "hl_step", "t_end", "active", "ee_target", "raw_action", "executed_action", "diag6_executed",
    "puck_disk_contact", "puck_disk_force", "puck_stator_contact", "puck_no_contact_steps",
    "base_contact",
)


def run_episode(sim, method, scenario: dict):
    sim.reset(scenario)
    method.reset(sim)
    ll = {k: [] for k in LL_FIELDS}
    hl = {k: [] for k in HL_FIELDS}
    status = {"failure_reason": "other_failure", "failure_time": None, "error": None}
    episode_high = None

    try:
        while True:
            active = not sim.settling
            out = method.step(sim)
            hl_step = sim.hl_steps_completed + 1

            def on_ll(k, _active=active, _hl=hl_step):
                ll["t"].append(sim.d.time)
                ll["hl_step"].append(_hl)
                ll["active"].append(_active)
                ll["base_ref"].append(sim.base_velocity_ref.astype(np.float64))
                ll["base_vel_b"].append(sim.base_velocity_body())
                ll["base_pos"].append(sim.d.qpos[0:3].copy())
                ll["puck_disk"].append(sim.get_puck_pos_disk().astype(np.float64))
                ll["puck_tilt_deg"].append(sim.puck_tilt_deg())
                ll["kp0_ref"].append(sim.ee_kp0_ref.astype(np.float64))
                ll["rot_ref"].append(sim.rot_ref.astype(np.float64))
                ll["kp0_cmd"].append(sim.kp0_cmd.astype(np.float64))
                ll["rot_cmd"].append(sim.rot_cmd.astype(np.float64))
                ll["ee_kp0_actual"].append(sim.measure_ee_kp0_plb().astype(np.float64))

            sim.run_hl_interval(on_ll)
            chk = sim.post_step_checks()
            if not np.all(np.isfinite(sim.d.qpos)) or not np.all(np.isfinite(sim.d.qvel)):
                raise FloatingPointError("Non-finite MuJoCo state")

            hl["hl_step"].append(hl_step)
            hl["t_end"].append(sim.d.time)
            hl["active"].append(active)
            hl["ee_target"].append(sim.current_ee_pose_index)
            hl["raw_action"].append(out["raw"])
            hl["executed_action"].append(out["executed"])
            hl["diag6_executed"].append(out["diag6"])
            hl["puck_disk_contact"].append(chk["puck_disk_contact"])
            hl["puck_disk_force"].append(chk["puck_disk_force"])
            hl["puck_stator_contact"].append(chk["puck_stator_contact"])
            hl["puck_no_contact_steps"].append(chk["puck_no_contact_steps"])
            hl["base_contact"].append(chk["base_contact"])

            if chk["base_contact"] or chk["puck_drop"]:
                # Isaac Lab terminates on base_contact | puck_drop. A robot fall is
                # reported as base_failure even if the puck is lost at the same step.
                status["failure_reason"] = "base_failure" if chk["base_contact"] else "puck_drop"
                status["failure_time"] = float(sim.d.time)
                status["puck_drop_flag"] = chk["puck_drop"]
                status["base_contact_flag"] = chk["base_contact"]
                break
            if chk["timeout"]:
                status["failure_reason"] = "success"
                break
            sim.advance_references()
            if episode_high is None and not sim.settling:
                episode_high = sim.episode_high_kp0.astype(np.float64).tolist()
    except Exception as exc:  # numerical blow-up etc. -> other_failure, kept in the denominator
        status["failure_reason"] = "other_failure"
        status["failure_time"] = float(sim.d.time)
        status["error"] = f"{type(exc).__name__}: {exc}"

    ts = {}
    for k, v in ll.items():
        ts["ll_" + k] = np.asarray(v)
    for k, v in hl.items():
        if k in ("raw_action", "executed_action", "diag6_executed"):
            width = max((len(x) for x in v), default=0)
            ts["hl_" + k] = np.asarray(v, dtype=np.float32).reshape(len(v), width)
        else:
            ts["hl_" + k] = np.asarray(v)
    summary = {
        "success": status["failure_reason"] == "success",
        "failure_reason": status["failure_reason"],
        "failure_time": status["failure_time"],
        "error": status["error"],
        "hl_steps": int(sim.hl_steps_completed),
        "episode_duration": float(sim.d.time),
        "episode_high_kp0": episode_high,
    }
    summary.update(episode_metrics(ts, ll_dt=sim.ll_dt))
    return summary, ts
