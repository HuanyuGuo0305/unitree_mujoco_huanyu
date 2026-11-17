"""Per-episode E4 metrics, computed only from the saved time series.

All metrics use the ACTIVE transport window: WBC samples (50 Hz) produced
during high-level steps that started after the 0.5-s reset settle. Reset
settling is never included. For early failures the window ends at termination;
``active_duration_s`` records how much was observed. No penalty is imputed for
the unobserved remainder.

  puck displacement   e(t) = || [x, y] of the puck centre in the moving disk frame ||
                      (exactly the quantity of the training reward)       [m]
  base XY RMSE        sqrt( mean( (vx-vx_ref)^2 + (vy-vy_ref)^2 ) / 2 )   [m/s]
  yaw-rate RMSE       sqrt( mean( (wz-wz_ref)^2 ) )                       [rad/s]

Velocities are the base_link linear/angular velocity in the BODY frame
(Isaac Lab root_lin_vel_b / root_ang_vel_b), the frame of the WBC command.
"""
from __future__ import annotations

import numpy as np


def episode_metrics(ts: dict, ll_dt: float) -> dict:
    act = np.asarray(ts["ll_active"], dtype=bool)
    n = int(act.sum())
    out = {"active_samples": n, "active_duration_s": n * float(ll_dt)}
    nan = float("nan")
    if n == 0:
        out.update(mean_puck_xy_displacement=nan, max_puck_xy_displacement=nan, final_puck_xy_displacement=nan,
                   base_xy_rmse=nan, base_yaw_rmse=nan, mean_puck_tilt_deg=nan, max_puck_tilt_deg=nan)
    else:
        p = np.asarray(ts["ll_puck_disk"])[act]
        rho = np.linalg.norm(p[:, :2], axis=1)
        ref = np.asarray(ts["ll_base_ref"])[act]
        vel = np.asarray(ts["ll_base_vel_b"])[act]
        dxy = vel[:, :2] - ref[:, :2]
        out.update(
            mean_puck_xy_displacement=float(rho.mean()),
            max_puck_xy_displacement=float(rho.max()),
            final_puck_xy_displacement=float(rho[-1]),
            base_xy_rmse=float(np.sqrt(np.mean(np.sum(dxy**2, axis=1)) / 2.0)),
            base_yaw_rmse=float(np.sqrt(np.mean((vel[:, 2] - ref[:, 2]) ** 2))),
            mean_puck_tilt_deg=float(np.mean(np.asarray(ts["ll_puck_tilt_deg"])[act])),
            max_puck_tilt_deg=float(np.max(np.asarray(ts["ll_puck_tilt_deg"])[act])),
        )
    hact = np.asarray(ts["hl_active"], dtype=bool)
    if hact.any():
        disk = np.asarray(ts["hl_puck_disk_contact"], dtype=bool)[hact]
        stator = np.asarray(ts["hl_puck_stator_contact"], dtype=bool)[hact]
        out["active_hl_steps"] = int(hact.sum())
        out["frac_hl_steps_disk_contact"] = float(disk.mean())
        # Steps where an Isaac-Lab-style BODY-level detector (any gripperStator geom)
        # would report contact but the disk-only detector does not.
        out["hl_steps_stator_only_contact"] = int(np.sum(stator & ~disk))
        ex = np.asarray(ts["hl_executed_action"])[hact]
        out["mean_abs_executed_action"] = float(np.abs(ex).mean()) if ex.size else 0.0
    else:
        out.update(active_hl_steps=0, frac_hl_steps_disk_contact=nan, hl_steps_stator_only_contact=0,
                   mean_abs_executed_action=0.0)
    return out
