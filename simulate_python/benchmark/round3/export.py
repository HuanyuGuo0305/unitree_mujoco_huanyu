#!/usr/bin/env python3
"""Phase D: export every individual seed's metrics BEFORE any averaging.

Three per-seed products, one per experiment, plus E1's per-command-interval
diagnostics and E3's per-seed directional degradation.

Nothing in this module averages over seeds, drops an episode, or rounds a
value for display: floats are written at full ``repr`` precision because these
CSVs are the input to the statistics stage and to any later re-analysis.
Terminated episodes are exported with ``survived = 0`` and their termination
reason, and their shortened ``valid_evaluation_time_s``.

Per-command-interval diagnostics (E1)
-------------------------------------
An interval runs from one command change up to (not including) the next, so it
contains the COMPLETE response to that command change, transient included. Two
interval families are emitted, distinguished by ``interval_type``:

  ``ee_target``    -- between successive EE target replacements
  ``base_segment`` -- between successive base-velocity command segments

Every metric is computed for both families, so an EE interval also carries the
base error observed during it and vice versa.

These are DIAGNOSTICS. The headline per-seed RMSE is always the full-window
RMSE over [settle_s, duration_s), never an unweighted average of the interval
RMSEs. ``check_interval_identity`` verifies the only correct way to recombine
them -- the sample-count-weighted quadratic mean -- reproduces the full-window
value, which is also run as a round-3 validation item.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from benchmark.round3 import common
from benchmark.scenarios import experiments as E

# ---------------------------------------------------------------- column order
# The first columns of each per-seed CSV are exactly the ones the round-3 brief
# names, in the order it names them. Everything measured follows after, so no
# metric is lost.
E1_REQUIRED = [
    "condition", "baseline", "seed",
    "base_vxy_rmse_mps", "base_yaw_rmse_radps",
    "ee_position_rmse_m", "ee_orientation_mae_deg",
    "valid_evaluation_time_s",
]
E1_EXTRA = [
    "experiment", "survived", "termination_reason", "termination_time_s",
    "n_eval_samples", "planned_evaluation_time_s",
    "base_vx_rmse_mps", "base_vy_rmse_mps", "base_vxy_rmse_yawframe_mps",
    "ee_position_mae_m", "ee_position_p95_m", "ee_orientation_rmse_deg",
    "n_base_switches_in_eval", "n_ee_switches_in_eval",
    "ee_pos_err_post_switch_mean_m", "base_vxy_err_post_switch_mean_mps",
    "base_roll_abs_mean_deg", "base_pitch_abs_mean_deg", "base_z_mean_m",
    "base_segment_duration_range_s", "ee_resample_interval_s",
    "duration_s", "settle_s", "path",
]

E2_REQUIRED = [
    "condition", "baseline", "seed",
    "base_vxy_rmse_mps", "base_yaw_rmse_radps",
    "total_power_w", "leg_power_w", "arm_power_w", "wheel_power_w",
    "leg_acc_rms_radps2", "arm_acc_rms_radps2",
    "valid_evaluation_time_s",
]
E2_EXTRA = [
    "experiment", "survived", "termination_reason", "n_eval_samples",
    "max_abs_vx_cmd_mps", "max_abs_vy_cmd_mps", "max_abs_yaw_rate_cmd_radps",
    "gripper_power_w", "leg_power_peak_w", "arm_power_peak_w", "wheel_power_peak_w",
    "leg_acc_rms_fd_radps2", "arm_acc_rms_fd_radps2",
    "leg_tau_abs_mean_nm", "arm_tau_abs_mean_nm", "wheel_tau_abs_mean_nm",
    "ee_position_rmse_m", "ee_orientation_mae_deg",
    "n_physics_samples", "duration_s", "settle_s", "path",
]

E3_REQUIRED = [
    "condition", "baseline", "trajectory_family", "alpha_base", "alpha_ee", "seed",
    "base_vxy_rmse_mps", "base_yaw_rmse_radps",
    "ee_position_rmse_m", "ee_orientation_mae_deg",
    "valid_evaluation_time_s",
]
E3_EXTRA = [
    "experiment", "survived", "termination_reason", "n_eval_samples",
    "ee_mean_speed_mps", "base_max_abs_cmd", "ee_position_mae_m",
    "ee_position_p95_m", "ee_orientation_rmse_deg",
    "base_vxy_rmse_yawframe_mps", "duration_s", "settle_s", "path",
]


def _common_fields(row: dict) -> dict:
    return {
        "experiment": row["experiment"],
        "condition": row["condition"],
        "baseline": row["baseline"],
        "seed": int(row["seed"]),
        "survived": int(row["survived"]),
        "termination_reason": row["termination_reason"],
        "termination_time_s": float(row.get("termination_time_s", float("nan"))),
        "n_eval_samples": int(row["n_eval_samples"]),
        "valid_evaluation_time_s": float(row["valid_eval_duration_s"]),
        "planned_evaluation_time_s": float(row["planned_eval_duration_s"]),
        "duration_s": float(row["duration_s"]),
        "settle_s": float(row["settle_s"]),
        "path": row["path"],
    }


def _tracking_fields(row: dict) -> dict:
    return {
        "base_vxy_rmse_mps": float(row["base_vxy_rmse"]),
        "base_yaw_rmse_radps": float(row["base_wz_rmse"]),
        "ee_position_rmse_m": float(row["ee_pos_rmse"]),
        "ee_orientation_mae_deg": float(row["ee_ori_mae_deg"]),
    }


# ==========================================================================
# E1
# ==========================================================================
def export_e1(out_dir: str) -> tuple[list[dict], list[dict]]:
    exp_cfg, _ = E.load_config("E1")
    baselines = list(exp_cfg["baselines"])
    conditions = list(E.E1_CONDITIONS)
    raw, problems = common.load_official("E1", baselines, conditions)

    rows = []
    for r in raw:
        out = _common_fields(r)
        out.update(_tracking_fields(r))
        out.update(
            {
                "base_vx_rmse_mps": float(r["base_vx_rmse"]),
                "base_vy_rmse_mps": float(r["base_vy_rmse"]),
                "base_vxy_rmse_yawframe_mps": float(r["base_vxy_rmse_yawframe"]),
                "ee_position_mae_m": float(r["ee_pos_mae"]),
                "ee_position_p95_m": float(r["ee_pos_p95"]),
                "ee_orientation_rmse_deg": float(r["ee_ori_rmse_deg"]),
                "n_base_switches_in_eval": int(r["n_base_switches_in_eval"]),
                "n_ee_switches_in_eval": int(r["n_ee_switches_in_eval"]),
                "ee_pos_err_post_switch_mean_m": float(r["ee_pos_err_post_switch_mean"]),
                "base_vxy_err_post_switch_mean_mps": float(r["base_vxy_err_post_switch_mean"]),
                "base_roll_abs_mean_deg": float(r["base_roll_abs_mean_deg"]),
                "base_pitch_abs_mean_deg": float(r["base_pitch_abs_mean_deg"]),
                "base_z_mean_m": float(r["base_z_mean"]),
                "base_segment_duration_range_s": r.get("base_segment_duration_range_s", ""),
                "ee_resample_interval_s": r.get("ee_resample_interval_s", ""),
            }
        )
        rows.append(out)

    rows.sort(key=lambda r: (r["condition"], r["baseline"], r["seed"]))
    common.write_csv(os.path.join(out_dir, "per_seed.csv"), rows, E1_REQUIRED + E1_EXTRA)
    return rows, problems


# ==========================================================================
# E1 per-command-interval diagnostics
# ==========================================================================
INTERVAL_COLUMNS = [
    "experiment", "condition", "baseline", "seed", "interval_type",
    "interval_index", "t_start_s", "t_end_s", "duration_s", "n_valid_samples",
    "starts_at_window_open", "truncated_by_episode_end",
    "ee_position_mean_error_m", "ee_position_rmse_m", "ee_position_max_error_m",
    "ee_orientation_mae_deg", "ee_orientation_rmse_deg",
    "base_vxy_rmse_mps", "base_yaw_rmse_radps",
    "vx_cmd_mps", "vy_cmd_mps", "yaw_rate_cmd_radps", "base_cmd_norm_mps",
    "ee_cmd_x_plb_m", "ee_cmd_y_plb_m", "ee_cmd_z_plb_m",
]


def _episode_arrays(path: str) -> dict:
    cmd = common.read_csv(os.path.join(path, "commands.csv"))
    st = common.read_csv(os.path.join(path, "states.csv"))
    keep = [i for i, r in enumerate(cmd) if int(r["evaluated"]) == 1]
    if not keep:
        return {}

    def c(rows, key):
        return np.asarray([float(rows[i][key]) for i in keep], dtype=np.float64)

    arr = {
        "t": c(cmd, "t"),
        "vx_cmd": c(cmd, "vx_cmd"), "vy_cmd": c(cmd, "vy_cmd"), "wz_cmd": c(cmd, "wz_cmd"),
        "vx_act": c(st, "vx_act"), "vy_act": c(st, "vy_act"), "wz_act": c(st, "wz_act"),
        "ee_pos_err": c(st, "ee_pos_err"), "ee_ori_err_deg": c(st, "ee_ori_err_deg"),
        "ee_cmd_x": c(cmd, "ee_cmd_x_plb"), "ee_cmd_y": c(cmd, "ee_cmd_y_plb"),
        "ee_cmd_z": c(cmd, "ee_cmd_z_plb"),
    }
    arr["ee_cmd_kp"] = np.stack([c(cmd, f"ee_cmd_kp{i}") for i in range(9)], axis=1)
    return arr


def _interval_bounds(arr: dict, kind: str) -> list[tuple[int, int]]:
    """[start, end) index pairs. The first starts at the evaluation window's
    opening sample even when that is not a command change, so no measured
    sample is left out of the diagnostics."""
    n = len(arr["t"])
    if kind == "base_segment":
        signal = np.stack([arr["vx_cmd"], arr["vy_cmd"], arr["wz_cmd"]], axis=1)
        tol = 1e-12
    elif kind == "ee_target":
        signal = arr["ee_cmd_kp"]
        tol = 1e-9
    else:
        raise ValueError(kind)
    changed = np.zeros(n, dtype=bool)
    if n > 1:
        changed[1:] = np.any(np.abs(np.diff(signal, axis=0)) > tol, axis=1)
    starts = [0] + list(np.nonzero(changed)[0])
    return [(s, (starts[i + 1] if i + 1 < len(starts) else n))
            for i, s in enumerate(starts)]


def _interval_row(arr, s, e, dt, kind, index, n, meta) -> dict:
    sl = slice(s, e)
    ee_p = arr["ee_pos_err"][sl]
    ee_o = arr["ee_ori_err_deg"][sl]
    dvx = arr["vx_act"][sl] - arr["vx_cmd"][sl]
    dvy = arr["vy_act"][sl] - arr["vy_cmd"][sl]
    dwz = arr["wz_act"][sl] - arr["wz_cmd"][sl]
    row = dict(meta)
    row.update(
        {
            "interval_type": kind,
            "interval_index": index,
            # Interval [t_start, t_end): t_start is the time of the first sample
            # held under the new command, t_end the time of the first sample
            # under the next one (or the window close).
            "t_start_s": float(arr["t"][s]),
            "t_end_s": float(arr["t"][e - 1] + dt),
            "duration_s": float((e - s) * dt),
            "n_valid_samples": int(e - s),
            "starts_at_window_open": int(s == 0),
            "truncated_by_episode_end": int(e == n),
            "ee_position_mean_error_m": float(np.mean(ee_p)),
            "ee_position_rmse_m": float(np.sqrt(np.mean(ee_p**2))),
            "ee_position_max_error_m": float(np.max(ee_p)),
            "ee_orientation_mae_deg": float(np.mean(ee_o)),
            "ee_orientation_rmse_deg": float(np.sqrt(np.mean(ee_o**2))),
            "base_vxy_rmse_mps": float(np.sqrt(np.mean(dvx**2 + dvy**2))),
            "base_yaw_rmse_radps": float(np.sqrt(np.mean(dwz**2))),
            "vx_cmd_mps": float(arr["vx_cmd"][s]),
            "vy_cmd_mps": float(arr["vy_cmd"][s]),
            "yaw_rate_cmd_radps": float(arr["wz_cmd"][s]),
            "base_cmd_norm_mps": float(np.hypot(arr["vx_cmd"][s], arr["vy_cmd"][s])),
            "ee_cmd_x_plb_m": float(arr["ee_cmd_x"][s]),
            "ee_cmd_y_plb_m": float(arr["ee_cmd_y"][s]),
            "ee_cmd_z_plb_m": float(arr["ee_cmd_z"][s]),
        }
    )
    return row


def export_e1_intervals(out_dir: str, per_seed_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Per-command-interval diagnostics plus the recombination identity check."""
    rows, checks = [], []
    for ps in per_seed_rows:
        path = os.path.join(common.BENCH, ps["path"])
        arr = _episode_arrays(path)
        if not arr:
            continue
        dt = 0.02
        meta = {
            "experiment": "E1", "condition": ps["condition"],
            "baseline": ps["baseline"], "seed": ps["seed"],
        }
        n = len(arr["t"])
        for kind in ("base_segment", "ee_target"):
            bounds = _interval_bounds(arr, kind)
            for index, (s, e) in enumerate(bounds):
                rows.append(_interval_row(arr, s, e, dt, kind, index, n, meta))
            checks.append(_identity_check(arr, bounds, kind, ps, n))
    rows.sort(key=lambda r: (r["condition"], r["baseline"], r["seed"],
                             r["interval_type"], r["interval_index"]))
    common.write_csv(os.path.join(out_dir, "per_command_interval.csv"),
                     rows, INTERVAL_COLUMNS)
    common.write_csv(os.path.join(out_dir, "per_command_interval_identity_check.csv"),
                     checks)
    return rows, checks


def _identity_check(arr, bounds, kind, ps, n) -> dict:
    """Recombine the interval RMSEs the only correct way and compare.

    full-window RMSE = sqrt( sum_i n_i * RMSE_i^2 / sum_i n_i )

    The naive unweighted mean of the interval RMSEs is computed alongside to
    show how far off it is -- which is why the per-seed headline metric is the
    full-window value, not that mean.
    """
    ns, sq_ee, sq_base, rm_ee, rm_base = [], [], [], [], []
    for s, e in bounds:
        sl = slice(s, e)
        ee = arr["ee_pos_err"][sl]
        dvx = arr["vx_act"][sl] - arr["vx_cmd"][sl]
        dvy = arr["vy_act"][sl] - arr["vy_cmd"][sl]
        ns.append(e - s)
        r_ee = float(np.sqrt(np.mean(ee**2)))
        r_b = float(np.sqrt(np.mean(dvx**2 + dvy**2)))
        rm_ee.append(r_ee)
        rm_base.append(r_b)
        sq_ee.append((e - s) * r_ee**2)
        sq_base.append((e - s) * r_b**2)
    total = float(sum(ns))
    weighted_ee = float(np.sqrt(sum(sq_ee) / total))
    weighted_base = float(np.sqrt(sum(sq_base) / total))
    return {
        "experiment": "E1", "condition": ps["condition"], "baseline": ps["baseline"],
        "seed": ps["seed"], "interval_type": kind, "n_intervals": len(bounds),
        "n_samples_total": int(total),
        "n_samples_in_episode": int(n),
        "full_window_ee_position_rmse_m": ps["ee_position_rmse_m"],
        "recombined_weighted_ee_position_rmse_m": weighted_ee,
        "abs_diff_ee_m": abs(weighted_ee - ps["ee_position_rmse_m"]),
        "unweighted_mean_of_interval_ee_rmse_m": float(np.mean(rm_ee)),
        "full_window_base_vxy_rmse_mps": ps["base_vxy_rmse_mps"],
        "recombined_weighted_base_vxy_rmse_mps": weighted_base,
        "abs_diff_base_mps": abs(weighted_base - ps["base_vxy_rmse_mps"]),
        "unweighted_mean_of_interval_base_rmse_mps": float(np.mean(rm_base)),
    }


# ==========================================================================
# E2
# ==========================================================================
def export_e2(out_dir: str) -> tuple[list[dict], list[dict]]:
    exp_cfg, _ = E.load_config("E2")
    baselines = list(exp_cfg["baselines"])
    conditions = list(E.E2_CONDITIONS)
    raw, problems = common.load_official("E2", baselines, conditions)

    rows = []
    for r in raw:
        out = _common_fields(r)
        out.update(_tracking_fields(r))
        out.update(
            {
                "total_power_w": float(r["total_power"]),
                "leg_power_w": float(r["leg_power"]),
                "arm_power_w": float(r["arm_power"]),
                "wheel_power_w": float(r["wheel_power"]),
                "gripper_power_w": float(r["gripper_power"]),
                "leg_power_peak_w": float(r["leg_power_peak"]),
                "arm_power_peak_w": float(r["arm_power_peak"]),
                "wheel_power_peak_w": float(r["wheel_power_peak"]),
                "leg_acc_rms_radps2": float(r["leg_acc_rms"]),
                "arm_acc_rms_radps2": float(r["arm_acc_rms"]),
                "leg_acc_rms_fd_radps2": float(r["leg_acc_rms_fd"]),
                "arm_acc_rms_fd_radps2": float(r["arm_acc_rms_fd"]),
                "leg_tau_abs_mean_nm": float(r["leg_tau_abs_mean"]),
                "arm_tau_abs_mean_nm": float(r["arm_tau_abs_mean"]),
                "wheel_tau_abs_mean_nm": float(r["wheel_tau_abs_mean"]),
                "max_abs_vx_cmd_mps": float(r.get("base_max_abs_vx_mps", float("nan"))),
                "max_abs_vy_cmd_mps": float(r.get("base_max_abs_vx_mps", float("nan"))),
                "max_abs_yaw_rate_cmd_radps": float(
                    r.get("base_max_abs_yaw_rate_radps", float("nan"))
                ),
                "n_physics_samples": int(r["n_physics_samples"]),
            }
        )
        rows.append(out)

    order = {"Low": 0, "Medium": 1, "High": 2}
    rows.sort(key=lambda r: (order[r["condition"]], r["baseline"], r["seed"]))
    common.write_csv(os.path.join(out_dir, "per_seed.csv"), rows, E2_REQUIRED + E2_EXTRA)
    return rows, problems


# ==========================================================================
# E3
# ==========================================================================
def export_e3(out_dir: str) -> tuple[list[dict], list[dict]]:
    exp_cfg, _ = E.load_config("E3")
    baselines = list(exp_cfg["baselines"])
    cells = E.e3_conditions(exp_cfg)
    conditions = [c for c, _ab, _ae, _f in cells]
    meta_by_condition = {c: (ab, ae, f) for c, ab, ae, f in cells}
    raw, problems = common.load_official("E3", baselines, conditions)

    rows = []
    for r in raw:
        ab, ae, family = meta_by_condition[r["condition"]]
        out = _common_fields(r)
        out.update(_tracking_fields(r))
        out.update(
            {
                "trajectory_family": family,
                "alpha_base": float(ab),
                "alpha_ee": float(ae),
                "ee_mean_speed_mps": float(r.get("ee_mean_speed_mps", ae * 0.40)),
                "base_max_abs_cmd": float(ab) * 0.6,
                "ee_position_mae_m": float(r["ee_pos_mae"]),
                "ee_position_p95_m": float(r["ee_pos_p95"]),
                "ee_orientation_rmse_deg": float(r["ee_ori_rmse_deg"]),
                "base_vxy_rmse_yawframe_mps": float(r["base_vxy_rmse_yawframe"]),
            }
        )
        rows.append(out)

    rows.sort(key=lambda r: (r["baseline"], r["trajectory_family"],
                             r["alpha_base"], r["alpha_ee"], r["seed"]))
    common.write_csv(os.path.join(out_dir, "per_seed.csv"), rows, E3_REQUIRED + E3_EXTRA)
    return rows, problems


DEG_COLUMNS = [
    "experiment", "baseline", "trajectory_family", "seed",
    "degradation_type", "condition", "alpha_base", "alpha_ee",
    "reference_condition", "reference_alpha_base", "reference_alpha_ee",
    "metric", "value_at_condition", "value_at_reference", "degradation",
    "condition_survived", "reference_survived",
]


def export_e3_degradation(out_dir: str, per_seed_rows: list[dict]) -> list[dict]:
    """Per-seed directional degradation, with its reference condition named.

    Delta_EE(ab, ae)   = EE_RMSE(ab, ae)      - EE_RMSE(0, ae)
    Delta_Base(ab, ae) = Base_XY_RMSE(ab, ae) - Base_XY_RMSE(ab, 0)

    Each difference is formed within one (baseline, trajectory family, seed),
    so the pairing is exact; a negative value is a genuine measurement (the
    error went DOWN when the other subsystem moved) and is never clipped.
    """
    index = {}
    for r in per_seed_rows:
        key = (r["baseline"], r["trajectory_family"], r["seed"],
               round(float(r["alpha_base"]), 6), round(float(r["alpha_ee"]), 6))
        index[key] = r

    rows = []
    for r in per_seed_rows:
        ab = round(float(r["alpha_base"]), 6)
        ae = round(float(r["alpha_ee"]), 6)
        base_key = (r["baseline"], r["trajectory_family"], r["seed"])
        for kind, metric, ref_ab, ref_ae in (
            ("locomotion_induced_ee", "ee_position_rmse_m", 0.0, ae),
            ("manipulation_induced_base", "base_vxy_rmse_mps", ab, 0.0),
        ):
            if (ab, ae) == (ref_ab, ref_ae):
                continue  # the reference condition itself; degradation is 0 by definition
            ref = index.get(base_key + (round(ref_ab, 6), round(ref_ae, 6)))
            if ref is None:
                continue
            rows.append(
                {
                    "experiment": "E3",
                    "baseline": r["baseline"],
                    "trajectory_family": r["trajectory_family"],
                    "seed": r["seed"],
                    "degradation_type": kind,
                    "condition": r["condition"],
                    "alpha_base": ab, "alpha_ee": ae,
                    "reference_condition": ref["condition"],
                    "reference_alpha_base": ref_ab, "reference_alpha_ee": ref_ae,
                    "metric": metric,
                    "value_at_condition": float(r[metric]),
                    "value_at_reference": float(ref[metric]),
                    "degradation": float(r[metric]) - float(ref[metric]),
                    "condition_survived": int(r["survived"]),
                    "reference_survived": int(ref["survived"]),
                }
            )
    rows.sort(key=lambda r: (r["degradation_type"], r["baseline"],
                             r["trajectory_family"], r["alpha_base"],
                             r["alpha_ee"], r["seed"]))
    common.write_csv(os.path.join(out_dir, "per_seed_degradation.csv"), rows, DEG_COLUMNS)
    return rows


# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", default="E1,E2,E3")
    parser.add_argument("--out", default=common.OUT)
    parser.add_argument("--skip-intervals", action="store_true")
    args = parser.parse_args(argv)
    wanted = [e.strip() for e in args.experiments.split(",")]
    seeds = common.official_seeds()
    all_problems = []

    print("=" * 78)
    print(f"ROUND 3 PER-SEED EXPORT   official seeds {seeds[0]}-{seeds[-1]} "
          f"({len(seeds)} per condition)")
    print("=" * 78)

    if "E1" in wanted:
        out_dir = os.path.join(args.out, "E1")
        rows, problems = export_e1(out_dir)
        all_problems += problems
        _report("E1", rows, problems, seeds)
        if not args.skip_intervals:
            irows, checks = export_e1_intervals(out_dir, rows)
            worst_ee = max((c["abs_diff_ee_m"] for c in checks), default=0.0)
            worst_b = max((c["abs_diff_base_mps"] for c in checks), default=0.0)
            print(f"     per-command-interval rows: {len(irows)} "
                  f"({sum(1 for r in irows if r['interval_type']=='base_segment')} base, "
                  f"{sum(1 for r in irows if r['interval_type']=='ee_target')} EE)")
            print(f"     recombination identity: max |weighted - full window| = "
                  f"{worst_ee:.3e} m (EE), {worst_b:.3e} m/s (base)")

    if "E2" in wanted:
        rows, problems = export_e2(os.path.join(args.out, "E2"))
        all_problems += problems
        _report("E2", rows, problems, seeds)

    if "E3" in wanted:
        out_dir = os.path.join(args.out, "E3")
        rows, problems = export_e3(out_dir)
        all_problems += problems
        _report("E3", rows, problems, seeds)
        deg = export_e3_degradation(out_dir, rows)
        print(f"     per-seed degradation rows: {len(deg)}")

    common.write_csv(os.path.join(args.out, "validation", "incomplete_episodes.csv"),
                     all_problems or [{"note": "none"}])
    print("=" * 78)
    if all_problems:
        print(f"{len(all_problems)} MISSING OR INCOMPLETE EPISODES -- see "
              f"validation/incomplete_episodes.csv")
        return 2
    print("Every official (baseline, condition, seed) episode is present and complete.")
    return 0


def _report(experiment, rows, problems, seeds):
    conds = sorted({r["condition"] for r in rows})
    bases = sorted({r["baseline"] for r in rows})
    counts = {}
    for r in rows:
        counts[(r["condition"], r["baseline"])] = counts.get((r["condition"], r["baseline"]), 0) + 1
    bad = {k: v for k, v in counts.items() if v != len(seeds)}
    n_term = sum(1 for r in rows if not r["survived"])
    print(f"  {experiment}: {len(rows)} per-seed rows   "
          f"{len(bases)} baselines x {len(conds)} conditions x {len(seeds)} seeds "
          f"= {len(bases)*len(conds)*len(seeds)} expected")
    print(f"     early terminations retained: {n_term}")
    if bad:
        print(f"     CELLS WITHOUT {len(seeds)} SEEDS: {bad}")
    if problems:
        print(f"     {len(problems)} missing/incomplete episodes")


if __name__ == "__main__":
    raise SystemExit(main())
