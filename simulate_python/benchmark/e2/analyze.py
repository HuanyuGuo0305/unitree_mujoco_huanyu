#!/usr/bin/env python3
"""Official E2 analysis: Wheeled-Legged (C1) vs Legged (C2), one dynamic condition,
100 paired seeds (configs/E2.yaml). Reads results/E2/raw/E2/<C>/dynamic/seed_XXX/
(run.py --experiment E2 --log-physics); writes results/E2/{data,stats,validation}/
and E2_summary_table.csv. No simulation.

Evaluation window [3, 30) s. Primary metrics (identical definitions for both modes):
 1 power_avg_w      P_avg = mean over 250 Hz physics steps of sum_j |tau_j qdot_j|,
                    tau = d.actuator_force, qdot = d.qvel; j = 12 leg + 6 arm + 1 gripper
                    (+ 4 wheel for Wheeled-Legged). Online accumulator in core/recorder.py.
                    Mechanical actuation effort, not battery power.
 2 energy_per_m_jpm E / progress. E = P_avg x T_valid (J). progress = sum over evaluated
                    50 Hz ticks with |c_k| > 0 of (v_k . c_k / |c_k|) * dt, where c_k is the
                    commanded planar body velocity (vx, vy) held at tick k and v_k the actual
                    planar body velocity. Signed (motion against the command subtracts);
                    ticks with a zero planar command add nothing; yaw-only effort is charged
                    to the energy but not to the progress. Cross-checked against the
                    yaw-frame velocity and against world-frame position increments
                    projected on the commanded direction.
 3 arm_acc_abs_mean mean over 250 Hz steps and the 6 arm joints of |d.qacc| (physics log).
 4 leg_acc_abs_mean mean over 250 Hz steps and the 12 leg joints (hip/thigh/calf) of
                    |d.qacc|; wheel joints excluded. Finite-difference cross-check.
Statistics: per mode mean, SD, median, 95 % bootstrap CI; paired difference
Legged - Wheeled-Legged per seed with 95 % paired bootstrap CI (10,000 resamples, seed
20261106), paired t-test, Holm over the 4 primary comparisons; Wilcoxon signed-rank
(Holm) as a robustness check. Secondary diagnostics are paired but not corrected.

    python3 -m benchmark.e2.analyze
"""

from __future__ import annotations

import csv
import json
import os
import sys

_SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _root in (os.path.dirname(_SIM), _SIM):          # repo root (utilities/) and simulate_python/
    if _root not in sys.path:
        sys.path.insert(0, _root)

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

from benchmark.e3.stats import holm, paired as paired_fn, summarise, wilson  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E2 = os.path.join(BENCH, "results", "E2")
RAW = os.path.join(E2, "raw", "E2")
REPO = os.path.dirname(_SIM)
COND = "dynamic"
MODES = ("C1", "C2")
LABEL = {"C1": "Wheeled-Legged", "C2": "Legged"}
PHYS_DT, CTRL_DT, EPS = 0.004, 0.02, 1e-9
# physics.npz: t, qpos leg(12), qpos arm(6), qvel leg(12), qvel arm(6), qacc leg(12),
# qacc arm(6), tau leg(12), tau arm(6)
SL = {"qvel_leg": slice(19, 31), "qvel_arm": slice(31, 37), "acc_leg": slice(37, 49),
      "acc_arm": slice(49, 55), "tau_leg": slice(55, 67), "tau_arm": slice(67, 73)}

PRIMARY = [("power_avg_w", "Average mechanical power", "W"),
           ("energy_per_m_jpm", "Energy per metre of realized progress", "J/m"),
           ("arm_acc_abs_mean", "Average arm joint acceleration", "rad/s^2"),
           ("leg_acc_abs_mean", "Average leg joint acceleration", "rad/s^2")]
SECONDARY = [("base_xy_rmse_mps", "Base XY velocity RMSE", "m/s"),
             ("base_yaw_rmse_radps", "Base yaw-rate RMSE", "rad/s"),
             ("ee_pos_rmse_m", "EE position RMSE", "m"),
             ("ee_ori_rmse_deg", "EE orientation RMSE", "deg"),
             ("progress_m", "Realized progress along the command (body-frame velocity)", "m"),
             ("commanded_distance_m", "Commanded planar distance, sum |c| dt", "m"),
             ("progress_ratio", "Progress / commanded distance", "-"),
             ("path_length_m", "Base XY path length (50 Hz positions)", "m"),
             ("net_displacement_m", "Net XY displacement, start to end of window", "m"),
             ("energy_j", "Mechanical energy over the window", "J"),
             ("leg_power_w", "Leg power", "W"), ("arm_power_w", "Arm power", "W"),
             ("wheel_power_w", "Wheel power", "W"), ("gripper_power_w", "Gripper power", "W"),
             ("progress_yawframe_m", "Progress, yaw-frame velocity (check)", "m"),
             ("progress_position_m", "Progress, world position increments (check)", "m"),
             ("energy_per_path_m_jpm", "Energy per metre of path length (diagnostic)", "J/m"),
             ("arm_acc_abs_mean_fd", "Arm acceleration, finite-difference check", "rad/s^2"),
             ("leg_acc_abs_mean_fd", "Leg acceleration, finite-difference check", "rad/s^2")]


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.asarray([float(r[k]) for r in rows]) for k in rows[0]}


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0])
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _ep_dir(mode, seed):
    return os.path.join(RAW, mode, COND, f"seed_{seed:03d}")


def progress_terms(st, cm, ev):
    """Per-tick progress contributions (m) for the three estimators, and the command."""
    c = np.stack([cm["vx_cmd"][ev], cm["vy_cmd"][ev]], 1)
    nc = np.linalg.norm(c, axis=1)
    on = nc > EPS
    u = np.zeros_like(c)
    u[on] = c[on] / nc[on, None]                                   # commanded unit direction (body)
    v_body = np.stack([st["vx_act"][ev], st["vy_act"][ev]], 1)
    v_yaw = np.stack([st["vx_act_yaw"][ev], st["vy_act_yaw"][ev]], 1)
    # world-frame: displacement over [k, k+1] projected on R_yaw(psi_k) u_k
    yaw = st["base_yaw"][ev]
    uw = np.stack([np.cos(yaw) * u[:, 0] - np.sin(yaw) * u[:, 1],
                   np.sin(yaw) * u[:, 0] + np.cos(yaw) * u[:, 1]], 1)
    dp = np.stack([np.diff(st["base_x"][ev]), np.diff(st["base_y"][ev])], 1)
    pos = np.zeros(len(nc))
    pos[:-1] = (dp * uw[:-1]).sum(1)
    return {"body": (v_body * u).sum(1) * CTRL_DT, "yaw": (v_yaw * u).sum(1) * CTRL_DT,
            "pos": pos, "cmd": nc * CTRL_DT, "on": on}


def episode(mode, seed):
    d = _ep_dir(mode, seed)
    meta = json.load(open(os.path.join(d, "metadata.json")))
    m = json.load(open(os.path.join(d, "metrics.json")))
    P = np.load(os.path.join(d, "physics.npz"))["data"].astype(np.float64)
    st, cm = _read_csv(os.path.join(d, "states.csv")), _read_csv(os.path.join(d, "commands.csv"))
    ev = st["evaluated"] > 0.5
    assert np.array_equal(ev, cm["evaluated"] > 0.5)
    assert P.shape == (int(m["n_physics_samples"]), 73)
    acc_l, acc_a = P[:, SL["acc_leg"]], P[:, SL["acc_arm"]]
    qv_l, qv_a = P[:, SL["qvel_leg"]], P[:, SL["qvel_arm"]]
    T = float(m["valid_eval_duration_s"])
    pr = progress_terms(st, cm, ev)
    prog, cmd = float(pr["body"].sum()), float(pr["cmd"].sum())
    bx, by = st["base_x"][ev], st["base_y"][ev]
    path = float(np.sum(np.hypot(np.diff(bx), np.diff(by))))
    energy = float(m["total_power"]) * T
    # transient share: progress in the first 0.5 s after each base-command switch
    sw = np.load(os.path.join(d, "scenario_commands.npz"))["base_switch_ticks"]
    tr = np.zeros(int(ev.sum()), bool)
    for k in sw:
        tr[int(k):int(k) + 25] = True
    return {
        "mode": LABEL[mode], "controller": mode, "condition": COND, "seed": seed,
        "survived": int(m["survived"]), "termination_reason": m["termination_reason"],
        "termination_time_s": m["termination_time_s"], "valid_eval_duration_s": T,
        "n_physics_samples": int(m["n_physics_samples"]),
        "t_first_s": float(P[0, 0]), "t_last_s": float(P[-1, 0]),
        "power_avg_w": float(m["total_power"]),
        "energy_per_m_jpm": energy / prog if prog > 0 else float("nan"),
        "arm_acc_abs_mean": float(np.abs(acc_a).mean()),
        "leg_acc_abs_mean": float(np.abs(acc_l).mean()),
        "base_xy_rmse_mps": float(m["base_vxy_rmse"]), "base_yaw_rmse_radps": float(m["base_wz_rmse"]),
        "ee_pos_rmse_m": float(m["ee_pos_rmse"]), "ee_ori_rmse_deg": float(m["ee_ori_rmse_deg"]),
        "progress_m": prog, "commanded_distance_m": cmd, "progress_ratio": prog / cmd,
        "path_length_m": path, "net_displacement_m": float(np.hypot(bx[-1] - bx[0], by[-1] - by[0])),
        "energy_j": energy,
        "leg_power_w": float(m["leg_power"]), "arm_power_w": float(m["arm_power"]),
        "wheel_power_w": float(m["wheel_power"]), "gripper_power_w": float(m["gripper_power"]),
        "progress_yawframe_m": float(pr["yaw"].sum()), "progress_position_m": float(pr["pos"].sum()),
        "energy_per_path_m_jpm": energy / path,
        "arm_acc_abs_mean_fd": float(np.abs(np.diff(qv_a, axis=0) / PHYS_DT).mean()),
        "leg_acc_abs_mean_fd": float(np.abs(np.diff(qv_l, axis=0) / PHYS_DT).mean()),
        "n_zero_planar_command_ticks": int((~pr["on"]).sum()),
        "frac_ticks_negative_progress": float(np.mean(pr["body"][pr["on"]] < 0)),
        "progress_share_first_0p5s_after_base_switch": float(pr["body"][tr].sum() / prog) if prog else float("nan"),
        "min_planar_command_mps": float(np.min(pr["cmd"] / CTRL_DT)),
        "leg_power_reconstructed_w": float(np.abs(P[:, SL["tau_leg"]] * qv_l).sum(1).mean()),
        "arm_power_reconstructed_w": float(np.abs(P[:, SL["tau_arm"]] * qv_a).sum(1).mean()),
        "leg_acc_rms_online": float(m["leg_acc_rms"]), "arm_acc_rms_online": float(m["arm_acc_rms"]),
        "leg_acc_rms_from_log": float(np.sqrt(np.mean(acc_l ** 2))),
        "arm_acc_rms_from_log": float(np.sqrt(np.mean(acc_a ** 2))),
        "initial_randomization": json.dumps(meta["initial_state"]["randomization"], sort_keys=True),
    }


def wilcoxon_p(d):
    d = d[np.isfinite(d)]
    return float(sps.wilcoxon(d).pvalue) if d.size > 1 and np.any(d != 0) else float("nan")


def stats_block(rows, seeds, keep=None):
    """Per-mode summaries and paired comparisons (optionally restricted to seeds in keep)."""
    idx = {(r["controller"], r["seed"]): r for r in rows}
    use = [s for s in seeds if keep is None or s in keep]

    def vec(mode, key):
        return np.asarray([idx[(mode, s)][key] for s in use], float)

    summary, paired = [], []
    for key, name, unit in PRIMARY + SECONDARY:
        for mode in MODES:
            summary.append({"mode": LABEL[mode], "controller": mode, "metric": key, "metric_name": name,
                            "unit": unit, "primary": int(key in dict((p[0], 1) for p in PRIMARY))}
                           | summarise(vec(mode, key)))
        a, b = vec("C1", key), vec("C2", key)
        r = paired_fn(a, b)
        ok = np.isfinite(a) & np.isfinite(b)
        paired.append({"metric": key, "metric_name": name, "unit": unit,
                       "primary": int(key in dict((p[0], 1) for p in PRIMARY)),
                       "difference": "Legged - Wheeled-Legged", "n_pairs": r["n_pairs"],
                       "wheeled_legged_mean": r.get("ref_mean"), "legged_mean": r.get("rival_mean"),
                       "mean_diff": r["mean_diff"], "sd_diff": r.get("sd_diff"),
                       "ci95_lo": r.get("ci95_lo"), "ci95_hi": r.get("ci95_hi"),
                       "rel_diff_vs_wheeled_legged": r.get("rel_diff"), "cohen_dz": r.get("cohen_dz"),
                       "frac_pairs_legged_lower": r.get("frac_rival_lower"),
                       "p_ttest": r["p"], "p_wilcoxon": wilcoxon_p(b[ok] - a[ok])})
    prim = [r for r in paired if r["primary"]]
    for r, h, hw in zip(prim, holm([r["p_ttest"] for r in prim]), holm([r["p_wilcoxon"] for r in prim])):
        r["p_holm_4"] = float(h)
        r["p_wilcoxon_holm_4"] = float(hw)
        r["verdict"] = ("no significant difference" if h >= 0.05 else
                        "Wheeled-Legged lower" if r["mean_diff"] > 0 else "Legged lower")
    return summary, paired


def main():
    cfg, _ = E.load_config("E2")
    seeds = list(range(cfg["seeds"]["start"], cfg["seeds"]["start"] + cfg["seeds"]["count"]))
    rows = [episode(m, s) for m in MODES for s in seeds]
    _write_csv(os.path.join(E2, "data", f"E2_per_trial_{len(rows)}.csv"), rows)

    summary, paired = stats_block(rows, seeds)
    completion = []
    for mode in MODES:
        k = sum(r["survived"] for r in rows if r["controller"] == mode)
        lo, hi = wilson(k, len(seeds))
        falls = [r["seed"] for r in rows if r["controller"] == mode and not r["survived"]]
        completion.append({"mode": LABEL[mode], "controller": mode, "n_trials": len(seeds), "n_completed": k,
                           "n_failed": len(seeds) - k, "completion_rate": k / len(seeds),
                           "wilson95_lo": lo, "wilson95_hi": hi, "failed_seeds": " ".join(map(str, falls))})
    os.makedirs(os.path.join(E2, "stats"), exist_ok=True)
    _write_csv(os.path.join(E2, "stats", "E2_summary.csv"), summary)
    _write_csv(os.path.join(E2, "stats", "E2_paired_legged_minus_wheeled.csv"), paired)
    _write_csv(os.path.join(E2, "stats", "E2_completion.csv"), completion)

    # sensitivity: drop every seed in which either mode failed (only if there are failures)
    failed = {r["seed"] for r in rows if not r["survived"]}
    if failed:
        _, p2 = stats_block(rows, seeds, keep=set(seeds) - failed)
        _write_csv(os.path.join(E2, "stats", "E2_paired_sensitivity_completed_pairs_only.csv"), p2)

    # summary table: the four primary metrics
    S = {(r["controller"], r["metric"]): r for r in summary}
    Pd = {r["metric"]: r for r in paired}
    table = []
    for key, name, unit in PRIMARY:
        p = Pd[key]
        table.append({"metric": key, "metric_name": name, "unit": unit, "lower_is_better": 1,
                      "n_pairs": p["n_pairs"],
                      "wheeled_legged_mean": S[("C1", key)]["mean"], "wheeled_legged_sd": S[("C1", key)]["sd"],
                      "legged_mean": S[("C2", key)]["mean"], "legged_sd": S[("C2", key)]["sd"],
                      "diff_legged_minus_wheeled": p["mean_diff"], "diff_ci95_lo": p["ci95_lo"],
                      "diff_ci95_hi": p["ci95_hi"], "rel_diff_vs_wheeled_legged": p["rel_diff_vs_wheeled_legged"],
                      "p_ttest": p["p_ttest"], "p_holm_4": p["p_holm_4"],
                      "p_wilcoxon": p["p_wilcoxon"], "p_wilcoxon_holm_4": p["p_wilcoxon_holm_4"],
                      "frac_pairs_legged_lower": p["frac_pairs_legged_lower"], "verdict": p["verdict"],
                      "n_failed_wheeled_legged": completion[0]["n_failed"],
                      "n_failed_legged": completion[1]["n_failed"]})
    _write_csv(os.path.join(E2, "E2_summary_table.csv"), table)
    validate(rows, seeds, cfg)
    print(f"  E2: {len(rows)} episodes, {len(seeds)} pairs; summary/paired/completion written")


def _joint_qpos(xml, qpos):
    """{joint name: qpos value} for the hinge joints, plus the free-joint base pose."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    out = {"base": np.asarray(qpos[0:7])}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            out[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)] = float(qpos[m.jnt_qposadr[j]])
    return out


def validate(rows, seeds, cfg):
    out = {"protocol": "E2 official: one dynamic condition, seeds 0-99, modes C1 (Wheeled-Legged), "
                       "C2 (Legged)"}
    out["n_episodes"], out["n_pairs"] = len(rows), len(seeds)
    out["n_failed"] = int(sum(1 - r["survived"] for r in rows))
    out["failures"] = [(r["controller"], r["seed"], r["termination_reason"], r["termination_time_s"])
                       for r in rows if not r["survived"]]
    surv = [r for r in rows if r["survived"]]
    out["window_s"] = [min(r["t_first_s"] for r in rows), max(r["t_last_s"] for r in surv)]
    out["n_physics_samples_completed"] = sorted({r["n_physics_samples"] for r in surv})
    out["valid_eval_duration_s_completed"] = sorted({r["valid_eval_duration_s"] for r in surv})

    # 1, 6: identical, deterministic commands; equal to the pre-generated protocol file
    pre = np.load(os.path.join(E2, "protocol", "E2_command_schedules_seeds000-099.npz"))
    same_modes = same_pre = same_regen = 0
    lim = np.asarray(cfg["base_command_limits"], float)
    max_cmd = np.zeros(3)
    for s in seeds:
        a = np.load(os.path.join(_ep_dir("C1", s), "scenario_commands.npz"))
        b = np.load(os.path.join(_ep_dir("C2", s), "scenario_commands.npz"))
        regen = E.build_scenario("E2", cfg, COND, s).arrays()
        same_modes += set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
        same_pre += all(np.array_equal(a[k], pre[f"seed{s:03d}_{k}"]) for k in a.files)
        same_regen += all(np.array_equal(a[k], regen[k]) for k in regen)
        max_cmd = np.maximum(max_cmd, np.abs(a["base_commands"]).max(0))
        # the commands actually applied (commands.csv) equal the schedule in both modes
        for mode in MODES:
            cm = _read_csv(os.path.join(_ep_dir(mode, s), "commands.csv"))
            ev = cm["evaluated"] > 0.5
            applied = np.stack([cm["vx_cmd"][ev], cm["vy_cmd"][ev], cm["wz_cmd"][ev]], 1)
            ee = np.stack([cm[f"ee_cmd_kp{i}"][ev] for i in range(9)], 1)
            assert np.allclose(applied, a["base_commands"][: ev.sum()], atol=1e-6), (mode, s)
            assert np.allclose(ee, a["ee_commands_plb"][: ev.sum()], atol=1e-6), (mode, s)
    out["commands_identical_between_modes"] = f"{same_modes}/{len(seeds)}"
    out["commands_equal_pregenerated_protocol_file"] = f"{same_pre}/{len(seeds)}"
    out["commands_regenerated_bit_identical"] = f"{same_regen}/{len(seeds)}"
    out["applied_commands_equal_schedule_both_modes"] = f"{2 * len(seeds)}/{2 * len(seeds)} (asserted)"
    # 3: training range
    out["base_command_max_abs_per_axis"] = max_cmd.tolist()
    out["base_command_limits"] = lim.tolist()
    out["base_commands_within_training_range"] = bool(np.all(max_cmd <= lim + 1e-6))
    table = E.keypoint_table(cfg["_global"])
    out["ee_targets_are_training_table_rows"] = True   # by construction; asserted in tests/test_e2.py
    out["ee_command_table_sha256"] = table.sha256

    # 2: equivalent initial conditions (joint by joint, by name, across the two models)
    xml = {"C1": os.path.join(REPO, "unitree_robots", "b2wz1", "ground.xml"),
           "C2": os.path.join(REPO, "unitree_robots", "b2wz1", "b2wz1_legged.xml")}
    worst, n_joints = 0.0, None
    rand_same = 0
    for s in seeds:
        q = {}
        for mode in MODES:
            meta = json.load(open(os.path.join(_ep_dir(mode, s), "metadata.json")))
            q[mode] = _joint_qpos(xml[mode], meta["initial_state"]["qpos"])
        common = sorted(set(q["C1"]) & set(q["C2"]))
        n_joints = len(common) - 1
        worst = max(worst, max(float(np.max(np.abs(np.asarray(q["C1"][k]) - np.asarray(q["C2"][k]))))
                               for k in common))
        ra = [r["initial_randomization"] for r in rows if r["seed"] == s]
        rand_same += ra[0] == ra[1]
    out["initial_state_common_joints_compared"] = n_joints
    out["initial_state_max_abs_diff_base_pose_and_common_joints"] = worst
    out["initial_randomization_identical"] = f"{rand_same}/{len(seeds)}"

    # 5, 7: window and actuator sets
    rel = [abs(r[a] - r[b]) / max(abs(r[a]), 1e-9) for r in rows
           for a, b in (("leg_power_w", "leg_power_reconstructed_w"), ("arm_power_w", "arm_power_reconstructed_w"))]
    out["power_reconstruction_max_rel_err"] = float(max(rel))
    out["power_components"] = "leg(12) + arm(6) + gripper(1) + wheel(4, Wheeled-Legged only)"
    out["wheel_power_legged_max_w"] = float(max(r["wheel_power_w"] for r in rows if r["controller"] == "C2"))
    out["wheel_power_wheeled_legged_min_w"] = float(min(r["wheel_power_w"] for r in rows if r["controller"] == "C1"))
    out["gripper_power_max_w"] = float(max(r["gripper_power_w"] for r in rows))
    out["qacc_log_vs_online_rms_max_rel_err"] = float(max(
        abs(r[a] - r[b]) / r[a] for r in rows for a, b in
        (("leg_acc_rms_online", "leg_acc_rms_from_log"), ("arm_acc_rms_online", "arm_acc_rms_from_log"))))
    fd = {}
    for mode in MODES:
        for k in ("arm", "leg"):
            x = np.asarray([r[f"{k}_acc_abs_mean"] for r in rows if r["controller"] == mode])
            y = np.asarray([r[f"{k}_acc_abs_mean_fd"] for r in rows if r["controller"] == mode])
            fd[f"{LABEL[mode]}_{k}"] = {"median_rel_diff": float(np.median((y - x) / x)),
                                        "max_abs_rel_diff": float(np.max(np.abs((y - x) / x))),
                                        "pearson_r": float(np.corrcoef(x, y)[0, 1])}
    out["qacc_vs_finite_difference"] = fd

    # progress-definition checks
    pc = {}
    for mode in MODES:
        R = [r for r in rows if r["controller"] == mode]
        body = np.asarray([r["progress_m"] for r in R])
        pc[LABEL[mode]] = {
            "min_progress_m": float(body.min()),
            "all_progress_positive": bool(np.all(body > 0)),
            "max_rel_diff_yawframe_vs_body": float(np.max(np.abs(
                np.asarray([r["progress_yawframe_m"] for r in R]) / body - 1))),
            "max_rel_diff_position_vs_body": float(np.max(np.abs(
                np.asarray([r["progress_position_m"] for r in R]) / body - 1))),
            "median_rel_diff_position_vs_body": float(np.median(
                np.asarray([r["progress_position_m"] for r in R]) / body - 1)),
            "zero_planar_command_ticks_total": int(sum(r["n_zero_planar_command_ticks"] for r in R)),
            "min_planar_command_speed_mps": float(min(r["min_planar_command_mps"] for r in R)),
            "mean_frac_ticks_negative_progress": float(np.mean([r["frac_ticks_negative_progress"] for r in R])),
            "mean_progress_share_first_0p5s_after_base_switch": float(np.mean(
                [r["progress_share_first_0p5s_after_base_switch"] for r in R])),
        }
    pc["first_0p5s_after_switch_time_share"] = "about 0.5 s per 2.5 s segment = ~0.20 of the time"
    out["progress_checks"] = pc
    path = os.path.join(E2, "validation", "E2_validation.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
