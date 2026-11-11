#!/usr/bin/env python3
"""Validate E2's power and acceleration metrics against the 250 Hz physics log.

The round-2 build had a real bug here: the ``physics.npz`` velocity block was
sliced one block too far and the ACCELERATION channel was read as velocity,
which inflated the reconstructed leg power by two orders of magnitude. This
script is the permanent guard. It reconstructs, straight from the stored
MuJoCo joint velocities and actuator forces,

    P_group = mean_t  sum_{j in group} | tau_j(t) * qdot_j(t) |
    a_group = sqrt( mean_t mean_j  qacc_j(t)^2 )

and requires each to reproduce the episode's logged ``metrics.json`` value.

``physics.npz`` holds one row per EVALUATED physics step, laid out as

    [t] | qpos_leg qpos_arm | qvel_leg qvel_arm | qacc_leg qacc_arm | tau_leg tau_arm

so each vector block is exactly ``n_leg + n_arm`` wide and the total column
count is ``1 + 4*(n_leg + n_arm)``. The layout is asserted, not assumed: a
column count that does not factor that way is an error, not something to slice
optimistically.

A deliberate negative control is included: the same computation performed with
the acceleration block in the velocity slot must FAIL the comparison. If the
wrong slice ever started passing, the check would be vacuous.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from benchmark.round3 import common

N_ARM = 6
REL_TOL = 1e-3  # physics.npz is float32; a wrong block is off by orders of magnitude


def reconstruct(path: str, wrong_slice: bool = False) -> dict | None:
    target = os.path.join(path, "physics.npz")
    if not os.path.isfile(target):
        return None
    with np.load(target) as handle:
        data = handle["data"].astype(np.float64)
    n_total = (data.shape[1] - 1) // 4
    n_leg = n_total - N_ARM
    if n_leg <= 0 or 1 + 4 * n_total != data.shape[1]:
        raise ValueError(f"unexpected physics.npz layout: {data.shape[1]} columns")

    qvel_block = 2 if wrong_slice else 1   # the round-2 bug, reproduced on demand
    qvel = data[:, 1 + qvel_block * n_total: 1 + (qvel_block + 1) * n_total]
    qacc = data[:, 1 + 2 * n_total: 1 + 3 * n_total]
    tau = data[:, 1 + 3 * n_total: 1 + 4 * n_total]

    return {
        "n_rows": int(data.shape[0]), "n_leg": n_leg, "n_arm": N_ARM,
        "leg_power": float(np.mean(np.sum(np.abs(tau[:, :n_leg] * qvel[:, :n_leg]), axis=1))),
        "arm_power": float(np.mean(np.sum(np.abs(tau[:, n_leg:] * qvel[:, n_leg:]), axis=1))),
        "leg_acc_rms": float(np.sqrt(np.mean(qacc[:, :n_leg] ** 2))),
        "arm_acc_rms": float(np.sqrt(np.mean(qacc[:, n_leg:] ** 2))),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="E2")
    parser.add_argument("--out", default=os.path.join(common.OUT, "validation"))
    args = parser.parse_args(argv)

    seeds = common.official_seeds()
    rows, failures = [], []
    checked_control = False
    control_result = ""

    for baseline in ("C1", "C2"):
        for condition in ("Low", "Medium", "High"):
            for seed in seeds:
                path = common.episode_dir(args.experiment, baseline, condition, seed)
                got = reconstruct(path)
                if got is None:
                    rows.append({"baseline": baseline, "condition": condition,
                                 "seed": seed, "status": "no_physics_log"})
                    continue
                with open(os.path.join(path, "metrics.json"), encoding="utf-8") as h:
                    metrics = json.load(h)
                row = {"baseline": baseline, "condition": condition, "seed": seed,
                       "status": "ok", "n_physics_rows": got["n_rows"],
                       "n_leg_dof": got["n_leg"], "n_arm_dof": got["n_arm"],
                       "n_physics_samples_metrics": int(metrics["n_physics_samples"])}
                if got["n_rows"] != int(metrics["n_physics_samples"]):
                    row["status"] = "row_count_mismatch"
                    failures.append(row)
                for key in ("leg_power", "arm_power", "leg_acc_rms", "arm_acc_rms"):
                    want = float(metrics[key])
                    have = got[key]
                    row[f"{key}_logged"] = want
                    row[f"{key}_reconstructed"] = have
                    row[f"{key}_abs_diff"] = abs(have - want)
                    row[f"{key}_rel_diff"] = (
                        abs(have - want) / abs(want) if want else float("nan")
                    )
                    if abs(have - want) > max(1e-9, REL_TOL * abs(want)):
                        row["status"] = f"mismatch_{key}"
                        failures.append(row)
                # total_power must equal the sum of its parts.
                parts = sum(float(metrics[k]) for k in
                            ("leg_power", "arm_power", "wheel_power", "gripper_power"))
                row["total_power_logged"] = float(metrics["total_power"])
                row["total_power_sum_of_parts"] = parts
                if abs(parts - float(metrics["total_power"])) > 1e-9:
                    row["status"] = "total_power_not_sum_of_parts"
                    failures.append(row)
                # C2's wheels are welded, so its wheel power must be identically 0.
                if baseline == "C2" and float(metrics["wheel_power"]) != 0.0:
                    row["status"] = "fixed_wheel_baseline_has_wheel_power"
                    failures.append(row)
                rows.append(row)

                if not checked_control:
                    checked_control = True
                    bad = reconstruct(path, wrong_slice=True)
                    ratio = bad["leg_power"] / max(float(metrics["leg_power"]), 1e-12)
                    passes = abs(bad["leg_power"] - float(metrics["leg_power"])) <= max(
                        1e-9, REL_TOL * abs(float(metrics["leg_power"]))
                    )
                    control_result = (
                        f"negative control (acceleration block read as velocity): "
                        f"{bad['leg_power']:.6g} W vs logged "
                        f"{float(metrics['leg_power']):.6g} W "
                        f"(x{ratio:.0f}); correctly REJECTED = {not passes}"
                    )
                    if passes:
                        failures.append({"baseline": baseline, "condition": condition,
                                         "seed": seed, "status": "negative_control_passed"})

    common.write_csv(os.path.join(args.out, "E2_power_reconstruction.csv"), rows)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    worst = max((r.get("leg_power_rel_diff", 0.0) for r in rows
                 if r.get("status") == "ok"), default=0.0)
    print("=" * 78)
    print(f"E2 POWER / ACCELERATION RECONSTRUCTION   {n_ok}/{len(rows)} episodes verified")
    print(f"  worst relative difference, leg power: {worst:.3e}")
    print(f"  {control_result}")
    if failures:
        print(f"  {len(failures)} FAILURES:")
        for row in failures[:10]:
            print(f"    {row['baseline']}/{row['condition']}/seed_{row['seed']:03d}: "
                  f"{row['status']}")
    print("=" * 78)
    return 0 if not failures and n_ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
