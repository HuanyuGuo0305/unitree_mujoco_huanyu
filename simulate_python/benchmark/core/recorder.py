"""Metrics recorder: reads actual MuJoCo state, never policy outputs alone.

Two sampling rates are handled:

* 250 Hz (every physics step) -- power proxy and joint-acceleration
  accumulators, which must see the real actuator forces and solved
  accelerations. Accumulated online so no multi-GB logs are needed.
* 50 Hz (every policy tick) -- the tracking signals written to
  ``commands.csv`` / ``states.csv`` / ``actuation.csv``.

Only samples with ``t >= settle_s`` enter the primary metrics; the settling
samples are still written to disk and flagged with ``evaluated=0``.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from benchmark.core import frames


class EpisodeRecorder:
    def __init__(self, adapter, settle_s: float, duration_s: float, log_physics: bool = False):
        self.adapter = adapter
        self.settle_s = float(settle_s)
        self.duration_s = float(duration_s)
        self.log_physics = bool(log_physics)

        m = adapter.model
        self.leg_dof = np.asarray(adapter.leg_vidx, dtype=np.int64)
        self.arm_dof = np.asarray(adapter.arm_vidx, dtype=np.int64)
        self.wheel_dof = np.asarray(adapter.wheel_vidx, dtype=np.int64)
        self.grip_dof = int(adapter.gripper_vidx)
        self.leg_act = np.asarray(adapter.leg_act, dtype=np.int64)
        self.arm_act = np.asarray(adapter.arm_act, dtype=np.int64)
        self.wheel_act = np.asarray(adapter.wheel_act, dtype=np.int64)
        self.grip_act = int(adapter.gripper_act)
        self.ee_bid = int(adapter.ee_bid)
        self.ground_z = float(adapter.ee_ground_z)
        self.physics_dt = float(adapter.simulation_dt)
        self.n_bodies = int(m.nbody)

        # --- 250 Hz accumulators (evaluated window only) ---
        self._n_phys = 0
        self._power_sums = {"leg": 0.0, "arm": 0.0, "wheel": 0.0, "gripper": 0.0}
        self._power_peak = {"leg": 0.0, "arm": 0.0, "wheel": 0.0, "gripper": 0.0}
        self._acc_sq = {"leg": 0.0, "arm": 0.0}
        self._acc_sq_fd = {"leg": 0.0, "arm": 0.0}
        self._tau_abs_sum = {"leg": 0.0, "arm": 0.0, "wheel": 0.0}
        self._prev_qvel_leg = None
        self._prev_qvel_arm = None
        self._contact_sum = 0.0

        # --- 50 Hz rows ---
        self.rows: list[dict] = []
        self.physics_rows: list[np.ndarray] = []

        self.terminated = False
        self.termination_reason = "completed"
        self.termination_time_s = float("nan")

    # ------------------------------------------------------------ 250 Hz
    def record_physics(self, sim_time: float):
        d = self.adapter.data
        if sim_time < self.settle_s:
            # Keep the finite-difference seed warm so the first evaluated
            # sample has a valid previous velocity.
            self._prev_qvel_leg = d.qvel[self.leg_dof].copy()
            self._prev_qvel_arm = d.qvel[self.arm_dof].copy()
            return

        qvel_leg = d.qvel[self.leg_dof]
        qvel_arm = d.qvel[self.arm_dof]
        tau_leg = d.actuator_force[self.leg_act]
        tau_arm = d.actuator_force[self.arm_act]

        p_leg = float(np.sum(np.abs(tau_leg * qvel_leg)))
        p_arm = float(np.sum(np.abs(tau_arm * qvel_arm)))
        p_grip = float(abs(d.actuator_force[self.grip_act] * d.qvel[self.grip_dof]))
        if self.wheel_act.size:
            qvel_wheel = d.qvel[self.wheel_dof]
            tau_wheel = d.actuator_force[self.wheel_act]
            p_wheel = float(np.sum(np.abs(tau_wheel * qvel_wheel)))
            self._tau_abs_sum["wheel"] += float(np.sum(np.abs(tau_wheel)))
        else:
            p_wheel = 0.0

        self._power_sums["leg"] += p_leg
        self._power_sums["arm"] += p_arm
        self._power_sums["wheel"] += p_wheel
        self._power_sums["gripper"] += p_grip
        for key, value in (("leg", p_leg), ("arm", p_arm), ("wheel", p_wheel), ("gripper", p_grip)):
            if value > self._power_peak[key]:
                self._power_peak[key] = value

        self._tau_abs_sum["leg"] += float(np.sum(np.abs(tau_leg)))
        self._tau_abs_sum["arm"] += float(np.sum(np.abs(tau_arm)))

        # Joint acceleration: MuJoCo's solved qacc at 250 Hz, unfiltered.
        acc_leg = d.qacc[self.leg_dof]
        acc_arm = d.qacc[self.arm_dof]
        self._acc_sq["leg"] += float(np.sum(acc_leg * acc_leg))
        self._acc_sq["arm"] += float(np.sum(acc_arm * acc_arm))

        # Finite-difference cross-check, same rate, also unfiltered.
        if self._prev_qvel_leg is not None:
            fd_leg = (qvel_leg - self._prev_qvel_leg) / self.physics_dt
            fd_arm = (qvel_arm - self._prev_qvel_arm) / self.physics_dt
            self._acc_sq_fd["leg"] += float(np.sum(fd_leg * fd_leg))
            self._acc_sq_fd["arm"] += float(np.sum(fd_arm * fd_arm))
        self._prev_qvel_leg = qvel_leg.copy()
        self._prev_qvel_arm = qvel_arm.copy()

        self._contact_sum += float(d.ncon)
        self._n_phys += 1

        if self.log_physics:
            self.physics_rows.append(
                np.concatenate(
                    [
                        [sim_time],
                        d.qpos[self.adapter.leg_qidx], d.qpos[self.adapter.arm_qidx],
                        qvel_leg, qvel_arm, acc_leg, acc_arm, tau_leg, tau_arm,
                    ]
                ).astype(np.float32)
            )

    # ------------------------------------------------------------- 50 Hz
    def record_policy(
        self,
        sim_time: float,
        tick: int,
        scenario_tick: int,
        base_command: np.ndarray,
        ee_command_plb: np.ndarray,
        ee_command_world: np.ndarray | None = None,
        extra: dict | None = None,
    ):
        d = self.adapter.data
        base_pos_w = d.qpos[0:3].copy()
        base_quat_w = d.qpos[3:7].copy()
        lin_vel_w = d.qvel[0:3].copy()
        ang_vel_w = d.qvel[3:6].copy()

        v_body, wz_body = frames.base_velocity_body_frame(lin_vel_w, ang_vel_w, base_quat_w)
        v_yaw, wz_yaw = frames.base_velocity_yaw_frame(lin_vel_w, ang_vel_w, base_quat_w)
        roll, pitch, yaw = frames.roll_pitch_yaw(base_quat_w)

        ee_pos_w = d.xpos[self.ee_bid].copy()
        ee_rot_w = d.xmat[self.ee_bid].reshape(3, 3).copy()
        ee_kp_plb = frames.ee_keypoints_plb_from_state(
            ee_pos_w, ee_rot_w, base_pos_w, base_quat_w, self.ground_z,
            self.adapter.ee_kp_dx, self.adapter.ee_kp_dz,
        )

        ee_pos_err, ee_ori_err_deg = frames.keypoint_pose_error(ee_command_plb, ee_kp_plb)
        p_cmd_plb, R_cmd_plb = frames.keypoints_to_pose(ee_command_plb)
        p_act_plb, _ = frames.keypoints_to_pose(ee_kp_plb)

        plb_pos_w, plb_quat_w = frames.plb_pose_from_base(base_pos_w, base_quat_w, self.ground_z)
        cmd_kp_world = frames.keypoints_plb_to_world(ee_command_plb, plb_pos_w, plb_quat_w)

        evaluated = int(sim_time >= self.settle_s)
        row = {
            "t": float(sim_time),
            "tick": int(tick),
            "scenario_tick": int(scenario_tick),
            "evaluated": evaluated,
            # base commands
            "vx_cmd": float(base_command[0]),
            "vy_cmd": float(base_command[1]),
            "wz_cmd": float(base_command[2]),
            # base actual, base body frame (primary)
            "vx_act": float(v_body[0]),
            "vy_act": float(v_body[1]),
            "wz_act": float(wz_body),
            # base actual, yaw-only (PLB) frame (secondary)
            "vx_act_yaw": float(v_yaw[0]),
            "vy_act_yaw": float(v_yaw[1]),
            "wz_act_yaw": float(wz_yaw),
            # base pose
            "base_x": float(base_pos_w[0]),
            "base_y": float(base_pos_w[1]),
            "base_z": float(base_pos_w[2]),
            "base_qw": float(base_quat_w[0]),
            "base_qx": float(base_quat_w[1]),
            "base_qy": float(base_quat_w[2]),
            "base_qz": float(base_quat_w[3]),
            "base_roll": float(roll),
            "base_pitch": float(pitch),
            "base_yaw": float(yaw),
            # EE, PLB frame (native command frame, primary metric frame)
            "ee_cmd_x_plb": float(p_cmd_plb[0]),
            "ee_cmd_y_plb": float(p_cmd_plb[1]),
            "ee_cmd_z_plb": float(p_cmd_plb[2]),
            "ee_act_x_plb": float(p_act_plb[0]),
            "ee_act_y_plb": float(p_act_plb[1]),
            "ee_act_z_plb": float(p_act_plb[2]),
            "ee_pos_err": float(ee_pos_err),
            "ee_ori_err_deg": float(ee_ori_err_deg),
            # EE, world frame
            "ee_act_x_w": float(ee_pos_w[0]),
            "ee_act_y_w": float(ee_pos_w[1]),
            "ee_act_z_w": float(ee_pos_w[2]),
            "ee_cmd_x_w": float(cmd_kp_world[0]),
            "ee_cmd_y_w": float(cmd_kp_world[1]),
            "ee_cmd_z_w": float(cmd_kp_world[2]),
            "n_contacts": int(d.ncon),
        }
        for i in range(9):
            row[f"ee_cmd_kp{i}"] = float(ee_command_plb[i])
            row[f"ee_act_kp{i}"] = float(ee_kp_plb[i])
        if ee_command_world is not None:
            row["ee_cmd_world_x"] = float(ee_command_world[0])
            row["ee_cmd_world_y"] = float(ee_command_world[1])
            row["ee_cmd_world_z"] = float(ee_command_world[2])
            R_tw = np.asarray(ee_command_world[3:12], dtype=np.float64).reshape(3, 3)
            row["ee_world_pos_err"] = float(
                np.linalg.norm(ee_pos_w - np.asarray(ee_command_world[0:3]))
            )
            row["ee_world_ori_err_deg"] = frames.geodesic_angle_deg(R_tw, ee_rot_w)

        # joint/actuator aggregates at the policy tick
        row["leg_tau_abs_mean"] = float(np.mean(np.abs(d.actuator_force[self.leg_act])))
        row["arm_tau_abs_mean"] = float(np.mean(np.abs(d.actuator_force[self.arm_act])))
        row["leg_acc_abs_mean"] = float(np.mean(np.abs(d.qacc[self.leg_dof])))
        row["arm_acc_abs_mean"] = float(np.mean(np.abs(d.qacc[self.arm_dof])))
        if self.wheel_act.size:
            row["wheel_tau_abs_mean"] = float(np.mean(np.abs(d.actuator_force[self.wheel_act])))
            row["wheel_vel_abs_mean"] = float(np.mean(np.abs(d.qvel[self.wheel_dof])))
        else:
            row["wheel_tau_abs_mean"] = 0.0
            row["wheel_vel_abs_mean"] = 0.0
        if extra:
            row.update({k: float(v) for k, v in extra.items()})
        self.rows.append(row)

    # ------------------------------------------------------- termination
    def check_termination(self, sim_time: float, cfg: dict) -> str | None:
        d = self.adapter.data
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            return "non_finite_state"
        z = float(d.qpos[2])
        if z < float(cfg.get("base_height_min", 0.30)):
            return "base_height"
        roll, pitch, _ = frames.roll_pitch_yaw(d.qpos[3:7])
        tilt_max = math.radians(float(cfg.get("base_tilt_max_deg", 50.0)))
        if abs(roll) > tilt_max or abs(pitch) > tilt_max:
            return "base_tilt"
        return None

    def mark_terminated(self, reason: str, sim_time: float):
        self.terminated = True
        self.termination_reason = reason
        self.termination_time_s = float(sim_time)

    # ----------------------------------------------------------- metrics
    def compute_metrics(self) -> dict:
        rows = [r for r in self.rows if r["evaluated"]]
        n = len(rows)
        out: dict = {
            "n_eval_samples": n,
            "n_physics_samples": self._n_phys,
            "survived": int(not self.terminated),
            "termination_reason": self.termination_reason,
            "termination_time_s": self.termination_time_s,
            "valid_eval_duration_s": (
                n * self.adapter.control_dt if n else 0.0
            ),
            "planned_eval_duration_s": self.duration_s - self.settle_s,
        }
        if n == 0:
            return out

        arr = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in rows[0]}

        def rmse(values):
            return float(np.sqrt(np.mean(values**2)))

        dvx = arr["vx_act"] - arr["vx_cmd"]
        dvy = arr["vy_act"] - arr["vy_cmd"]
        dwz = arr["wz_act"] - arr["wz_cmd"]
        out["base_vxy_rmse"] = float(np.sqrt(np.mean(dvx**2 + dvy**2)))
        out["base_wz_rmse"] = rmse(dwz)
        out["base_vx_rmse"] = rmse(dvx)
        out["base_vy_rmse"] = rmse(dvy)

        dvx_y = arr["vx_act_yaw"] - arr["vx_cmd"]
        dvy_y = arr["vy_act_yaw"] - arr["vy_cmd"]
        out["base_vxy_rmse_yawframe"] = float(np.sqrt(np.mean(dvx_y**2 + dvy_y**2)))
        out["base_wz_rmse_yawframe"] = rmse(arr["wz_act_yaw"] - arr["wz_cmd"])

        out["ee_pos_rmse"] = rmse(arr["ee_pos_err"])
        out["ee_ori_mae_deg"] = float(np.mean(arr["ee_ori_err_deg"]))
        out["ee_pos_mae"] = float(np.mean(arr["ee_pos_err"]))
        out["ee_ori_rmse_deg"] = rmse(arr["ee_ori_err_deg"])
        out["ee_pos_p95"] = float(np.percentile(arr["ee_pos_err"], 95))
        if "ee_world_pos_err" in arr:
            out["ee_world_pos_rmse"] = rmse(arr["ee_world_pos_err"])
            out["ee_world_pos_mae"] = float(np.mean(arr["ee_world_pos_err"]))
            out["ee_world_pos_max"] = float(np.max(arr["ee_world_pos_err"]))
            out["ee_world_ori_mae_deg"] = float(np.mean(arr["ee_world_ori_err_deg"]))
            out["ee_world_ori_max_deg"] = float(np.max(arr["ee_world_ori_err_deg"]))

        out["base_roll_abs_mean_deg"] = float(np.mean(np.abs(np.degrees(arr["base_roll"]))))
        out["base_pitch_abs_mean_deg"] = float(np.mean(np.abs(np.degrees(arr["base_pitch"]))))
        out["base_z_mean"] = float(np.mean(arr["base_z"]))

        # Power proxy and acceleration RMS over matched physical DOFs.
        if self._n_phys:
            for key in ("leg", "arm", "wheel", "gripper"):
                out[f"{key}_power"] = self._power_sums[key] / self._n_phys
                out[f"{key}_power_peak"] = self._power_peak[key]
            out["total_power"] = sum(
                out[f"{key}_power"] for key in ("leg", "arm", "wheel", "gripper")
            )
            out["leg_acc_rms"] = float(np.sqrt(self._acc_sq["leg"] / (self._n_phys * 12)))
            out["arm_acc_rms"] = float(np.sqrt(self._acc_sq["arm"] / (self._n_phys * 6)))
            out["leg_acc_rms_fd"] = float(np.sqrt(self._acc_sq_fd["leg"] / (self._n_phys * 12)))
            out["arm_acc_rms_fd"] = float(np.sqrt(self._acc_sq_fd["arm"] / (self._n_phys * 6)))
            out["leg_tau_abs_mean"] = self._tau_abs_sum["leg"] / (self._n_phys * 12)
            out["arm_tau_abs_mean"] = self._tau_abs_sum["arm"] / (self._n_phys * 6)
            out["wheel_tau_abs_mean"] = (
                self._tau_abs_sum["wheel"] / (self._n_phys * 4) if self.wheel_act.size else 0.0
            )
            out["mean_contacts"] = self._contact_sum / self._n_phys

        out.update(self._switch_response_metrics(arr))
        return out

    def _switch_response_metrics(self, arr: dict) -> dict:
        """Secondary diagnostics: error right after a command switch.

        Reported alongside, never subtracted from, the primary metrics. The
        primary RMSE/MAE above include every evaluated sample, transients
        included.
        """
        out: dict = {}
        window_ticks = int(round(0.5 / self.adapter.control_dt))  # 500 ms

        def post_switch_mean(signal_key, switch_mask):
            idx = np.nonzero(switch_mask)[0]
            if idx.size == 0:
                return float("nan")
            values = []
            for i in idx:
                values.append(arr[signal_key][i : i + window_ticks])
            stacked = np.concatenate(values) if values else np.asarray([])
            return float(np.mean(stacked)) if stacked.size else float("nan")

        ee_cmd = np.stack([arr[f"ee_cmd_kp{i}"] for i in range(9)], axis=1)
        ee_switch = np.zeros(ee_cmd.shape[0], dtype=bool)
        ee_switch[1:] = np.any(np.abs(np.diff(ee_cmd, axis=0)) > 1e-6, axis=1)
        out["ee_pos_err_post_switch_mean"] = post_switch_mean("ee_pos_err", ee_switch)
        out["ee_ori_err_post_switch_mean_deg"] = post_switch_mean("ee_ori_err_deg", ee_switch)
        out["n_ee_switches_in_eval"] = int(ee_switch.sum())

        base_cmd = np.stack([arr["vx_cmd"], arr["vy_cmd"], arr["wz_cmd"]], axis=1)
        base_switch = np.zeros(base_cmd.shape[0], dtype=bool)
        base_switch[1:] = np.any(np.abs(np.diff(base_cmd, axis=0)) > 1e-9, axis=1)
        vxy_err = np.sqrt(
            (arr["vx_act"] - arr["vx_cmd"]) ** 2 + (arr["vy_act"] - arr["vy_cmd"]) ** 2
        )
        arr = dict(arr)
        arr["_vxy_err"] = vxy_err
        out["base_vxy_err_post_switch_mean"] = post_switch_mean("_vxy_err", base_switch)
        out["n_base_switches_in_eval"] = int(base_switch.sum())
        return out

    # ------------------------------------------------------------- output
    def write(self, out_dir: str, metadata: dict, commands: dict, extra_metrics: dict | None = None):
        os.makedirs(out_dir, exist_ok=True)
        metrics = self.compute_metrics()
        if extra_metrics:
            metrics.update({k: float(v) for k, v in extra_metrics.items()})

        command_cols = [
            "t", "tick", "scenario_tick", "evaluated",
            "vx_cmd", "vy_cmd", "wz_cmd",
        ] + [f"ee_cmd_kp{i}" for i in range(9)] + [
            "ee_cmd_x_plb", "ee_cmd_y_plb", "ee_cmd_z_plb",
            "ee_cmd_x_w", "ee_cmd_y_w", "ee_cmd_z_w",
        ]
        state_cols = [
            "t", "tick", "evaluated",
            "vx_act", "vy_act", "wz_act", "vx_act_yaw", "vy_act_yaw", "wz_act_yaw",
            "base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz",
            "base_roll", "base_pitch", "base_yaw",
            "ee_act_x_plb", "ee_act_y_plb", "ee_act_z_plb",
            "ee_act_x_w", "ee_act_y_w", "ee_act_z_w",
            "ee_pos_err", "ee_ori_err_deg", "n_contacts",
        ] + [f"ee_act_kp{i}" for i in range(9)]
        if self.rows and "ee_world_pos_err" in self.rows[0]:
            state_cols += ["ee_world_pos_err", "ee_world_ori_err_deg"]
            command_cols += ["ee_cmd_world_x", "ee_cmd_world_y", "ee_cmd_world_z"]
        actuation_cols = [
            "t", "tick", "evaluated",
            "leg_tau_abs_mean", "arm_tau_abs_mean", "wheel_tau_abs_mean",
            "leg_acc_abs_mean", "arm_acc_abs_mean", "wheel_vel_abs_mean",
        ] + [k for k in (self.rows[0] if self.rows else {}) if k.startswith(("uan_", "ik_", "guidance_", "blend"))]

        _write_csv(os.path.join(out_dir, "commands.csv"), self.rows, command_cols)
        _write_csv(os.path.join(out_dir, "states.csv"), self.rows, state_cols)
        _write_csv(os.path.join(out_dir, "actuation.csv"), self.rows, actuation_cols)

        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
        meta = dict(metadata)
        meta.update(
            {
                "termination_reason": self.termination_reason,
                "termination_time_s": self.termination_time_s,
                "valid_eval_duration_s": metrics["valid_eval_duration_s"],
                "survived": metrics["survived"],
            }
        )
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True, default=str)
        np.savez_compressed(os.path.join(out_dir, "scenario_commands.npz"), **commands)
        if self.log_physics and self.physics_rows:
            np.savez_compressed(
                os.path.join(out_dir, "physics.npz"),
                data=np.stack(self.physics_rows),
            )
        return metrics


def _write_csv(path: str, rows: list[dict], columns: list[str]):
    columns = [c for c in dict.fromkeys(columns) if not rows or c in rows[0]]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(
                ",".join(
                    ("%.9g" % row[c]) if isinstance(row[c], float) else str(row[c])
                    for c in columns
                )
                + "\n"
            )
