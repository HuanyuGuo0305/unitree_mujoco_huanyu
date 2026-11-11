#!/usr/bin/env python3
"""Decide, per episode, whether round-2 E3 raw data may be reused in round 3.

Round 3 does not change the E3 scenario, and E3 episodes are deterministic
functions of (baseline, condition, seed). Re-running 1,350 identical episodes
would cost compute and prove nothing -- but "the config looks the same" is not
evidence. This script checks each candidate episode against what the CURRENT
round-3 configuration would produce, and reuses it only if every check passes.

Checks, one per requirement in the round-3 brief:

  C1  presence/completeness  all required artefacts exist and are non-empty
  C2  baseline identity      metadata baseline, deploy script, policy path
  C3  asset hashes           policy ONNX, frozen UAN model and MuJoCo XML
                             SHA-256 equal the files on disk NOW
  C4  control timing         physics_dt, control_dt, duration_s, settle_s,
                             eval window, physics_steps, policy ticks
  C5  scenario definition    the stored scenario description equals the one
                             rebuilt from today's E3.yaml, key by key
  C6  commands bit-exact     the stored scenario_commands.npz equals the
                             freshly generated schedule, bit for bit. This is
                             the real test of trajectory geometry, amplitude,
                             centre, plane, time parameterisation, orientation
                             protocol and motion intensity at once.
  C7  initial state          the recorded qpos equals what the adapter
                             produces for that seed now, and is identical
                             across baselines at the same seed (which is what
                             makes the paired comparison valid)
  C8  recorded samples       evaluated-sample count matches the window, or the
                             episode is marked terminated with a reason
  C9  metrics reconstruction the four headline metrics recomputed from
                             commands.csv + states.csv reproduce metrics.json

Any episode failing any check is listed in ``E3_rerun_required.csv`` and must
be re-run; nothing is silently accepted or silently discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from benchmark.round3 import common
from benchmark.scenarios import experiments as E

TOL_METRIC_REL = 1e-9
TOL_METRIC_ABS = 1e-9


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_asset_hashes(baseline: str, cache: dict) -> dict:
    """SHA-256 of the policy / UAN / XML assets as they exist on disk now."""
    if baseline in cache:
        return cache[baseline]
    from benchmark.controllers.registry import build_adapter

    adapter = build_adapter(baseline, verbose=False)
    cache[baseline] = dict(adapter.asset_hashes)
    return cache[baseline]


def _initial_qpos(baseline: str, seed: int, init_cfg: dict, cache: dict) -> np.ndarray:
    """Recompute the seeded initial qpos exactly as the runner would."""
    import mujoco

    from benchmark.controllers.registry import build_adapter

    if baseline not in cache:
        adapter = build_adapter(baseline, verbose=False)
        model = adapter.build_model()
        data = mujoco.MjData(model)
        adapter.attach(model, data)
        cache[baseline] = adapter
    adapter = cache[baseline]
    adapter.set_initial_state()
    adapter.apply_initial_state_noise(seed, init_cfg)
    return adapter.data.qpos.copy()


def _recompute_metrics(path: str) -> dict:
    """The four headline metrics, recomputed from the raw 50 Hz logs."""
    cmd = common.read_csv(os.path.join(path, "commands.csv"))
    st = common.read_csv(os.path.join(path, "states.csv"))
    if len(cmd) != len(st):
        raise ValueError(f"commands.csv has {len(cmd)} rows, states.csv has {len(st)}")
    keep = [i for i, r in enumerate(cmd) if int(r["evaluated"]) == 1]
    if not keep:
        raise ValueError("no evaluated samples")

    def col(rows, key):
        return np.asarray([float(rows[i][key]) for i in keep], dtype=np.float64)

    dvx = col(st, "vx_act") - col(cmd, "vx_cmd")
    dvy = col(st, "vy_act") - col(cmd, "vy_cmd")
    dwz = col(st, "wz_act") - col(cmd, "wz_cmd")
    ee_p = col(st, "ee_pos_err")
    ee_o = col(st, "ee_ori_err_deg")
    return {
        "base_vxy_rmse": float(np.sqrt(np.mean(dvx**2 + dvy**2))),
        "base_wz_rmse": float(np.sqrt(np.mean(dwz**2))),
        "ee_pos_rmse": float(np.sqrt(np.mean(ee_p**2))),
        "ee_ori_mae_deg": float(np.mean(ee_o)),
        "n_eval_samples": len(keep),
    }


def verify_episode(
    baseline, condition, seed, exp_cfg, fresh, asset_cache, adapter_cache,
    recompute_metrics=True, recompute_initial_state=True,
):
    """Return (ok: bool, failures: list[str], detail: dict)."""
    path = common.episode_dir("E3", baseline, condition, seed)
    fail: list[str] = []
    detail: dict = {"path": os.path.relpath(path, common.BENCH)}

    # --- C1 presence / completeness ---
    status = common.episode_status(path)
    detail["status"] = status
    if status != "ok":
        return False, [f"C1_completeness:{status}"], detail

    with open(os.path.join(path, "metadata.json"), encoding="utf-8") as h:
        meta = json.load(h)
    with open(os.path.join(path, "metrics.json"), encoding="utf-8") as h:
        metrics = json.load(h)

    # --- C2 baseline identity ---
    from benchmark.controllers.registry import BASELINES

    info = BASELINES[baseline]
    if meta.get("baseline") != baseline:
        fail.append(f"C2_baseline:{meta.get('baseline')}!={baseline}")
    if meta.get("deploy_script") != info["deploy_script"]:
        fail.append("C2_deploy_script")
    if int(meta.get("seed", -1)) != int(seed):
        fail.append(f"C2_seed:{meta.get('seed')}!={seed}")
    if meta.get("condition") != condition:
        fail.append("C2_condition")

    # --- C3 asset hashes vs the files on disk now ---
    now = _current_asset_hashes(baseline, asset_cache)
    stored = meta.get("asset_sha256", {})
    detail["asset_sha256_stored"] = stored
    for key, value in now.items():
        if stored.get(key) != value:
            fail.append(f"C3_asset_{key}")

    # --- C4 control timing ---
    timing = {
        "physics_dt": 0.004, "control_dt": 0.02,
        "duration_s": float(exp_cfg["duration_s"]),
        "settle_s": float(exp_cfg["settle_s"]),
    }
    for key, want in timing.items():
        got = float(meta.get(key, float("nan")))
        if abs(got - want) > 1e-12:
            fail.append(f"C4_{key}:{got}!={want}")
    if list(meta.get("eval_window_s", [])) != [timing["settle_s"], timing["duration_s"]]:
        fail.append("C4_eval_window")
    want_steps = int(round(timing["duration_s"] / timing["physics_dt"]))
    if not metrics.get("survived", 1):
        pass  # a terminated episode legitimately has fewer steps
    elif int(meta.get("physics_steps", -1)) != want_steps:
        fail.append(f"C4_physics_steps:{meta.get('physics_steps')}!={want_steps}")

    # --- C5 scenario description vs today's config ---
    want_desc = fresh[condition]["description"]
    got_desc = meta.get("scenario", {})
    for key, want in want_desc.items():
        if key == "seed":
            continue
        got = got_desc.get(key, "<absent>")
        if isinstance(want, float) and isinstance(got, (int, float)):
            same = abs(float(got) - want) <= 1e-12
        elif isinstance(want, list):
            same = (
                isinstance(got, list) and len(got) == len(want)
                and all(
                    abs(float(a) - float(b)) <= 1e-12 if isinstance(b, (int, float)) else a == b
                    for a, b in zip(got, want)
                )
            )
        else:
            same = got == want
        if not same:
            fail.append(f"C5_scenario_{key}")

    # --- C6 commands bit-exact ---
    npz_path = os.path.join(path, "scenario_commands.npz")
    if not os.path.isfile(npz_path):
        fail.append("C6_no_scenario_commands")
    else:
        with np.load(npz_path) as stored_cmd:
            want_cmd = fresh[condition]["arrays"]
            for key, want in want_cmd.items():
                if key not in stored_cmd:
                    fail.append(f"C6_missing_{key}")
                    continue
                got = stored_cmd[key]
                if got.shape != np.asarray(want).shape:
                    fail.append(f"C6_shape_{key}")
                elif not np.array_equal(got, np.asarray(want)):
                    d = float(np.max(np.abs(got.astype(np.float64)
                                            - np.asarray(want, dtype=np.float64))))
                    fail.append(f"C6_values_{key}(max_abs_diff={d:.3e})")
        detail["commands_bitexact"] = not any(f.startswith("C6_") for f in fail)

    # --- C7 initial state ---
    init = meta.get("initial_state", {})
    detail["initial_base_z"] = init.get("base_z")
    detail["initial_qpos_sha256"] = hashlib.sha256(
        np.asarray(init.get("qpos", []), dtype=np.float64).tobytes()
    ).hexdigest()
    if recompute_initial_state:
        want_qpos = _initial_qpos(
            baseline, seed, common.load_global_config().get("initial_state", {}),
            adapter_cache,
        )
        got_qpos = np.asarray(init.get("qpos", []), dtype=np.float64)
        if got_qpos.shape != want_qpos.shape:
            fail.append("C7_qpos_shape")
        elif not np.array_equal(got_qpos, want_qpos):
            fail.append(
                f"C7_qpos(max_abs_diff={float(np.max(np.abs(got_qpos - want_qpos))):.3e})"
            )

    # --- C8 recorded samples ---
    n_eval = int(metrics.get("n_eval_samples", 0))
    want_eval = int(round((timing["duration_s"] - timing["settle_s"]) / timing["control_dt"]))
    detail["n_eval_samples"] = n_eval
    detail["survived"] = int(metrics.get("survived", 0))
    detail["termination_reason"] = metrics.get("termination_reason", "")
    if metrics.get("survived", 0):
        if abs(n_eval - want_eval) > 1:
            fail.append(f"C8_n_eval:{n_eval}!={want_eval}")
    else:
        if n_eval >= want_eval:
            fail.append("C8_terminated_but_full_window")
        if not metrics.get("termination_reason"):
            fail.append("C8_no_termination_reason")

    # --- C9 metrics reconstruction ---
    if recompute_metrics:
        try:
            recomputed = _recompute_metrics(path)
        except Exception as exc:  # noqa: BLE001
            fail.append(f"C9_unreadable({exc})")
        else:
            if recomputed["n_eval_samples"] != n_eval:
                fail.append(
                    f"C9_n_eval:{recomputed['n_eval_samples']}!={n_eval}"
                )
            for key in common.ABS_METRICS:
                want = float(metrics.get(key, float("nan")))
                got = recomputed[key]
                tol = max(TOL_METRIC_ABS, TOL_METRIC_REL * abs(want))
                # commands.csv / states.csv are written with %.9g, so compare
                # at that precision, not at float64 precision.
                tol = max(tol, 5e-9 * max(1.0, abs(want)))
                if not np.isfinite(got) or abs(got - want) > tol:
                    fail.append(f"C9_{key}(stored={want:.12g},recomputed={got:.12g})")
            detail["recomputed"] = recomputed

    return (not fail), fail, detail


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds (default: the official pool)")
    parser.add_argument("--sample-initial-state", type=int, default=0,
                        help="Recompute the seeded initial qpos for the first N "
                             "seeds only (0 = all; it needs a MuJoCo model per baseline)")
    parser.add_argument("--out", default=os.path.join(common.OUT, "E3"))
    args = parser.parse_args(argv)

    exp_cfg, _ = E.load_config("E3")
    baselines = list(exp_cfg["baselines"])
    conditions = [c for c, _ab, _ae, _f in E.e3_conditions(exp_cfg)]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else common.official_seeds())

    print("=" * 78)
    print("E3 REUSE VERIFICATION (round 3)")
    print(f"baselines {baselines}")
    print(f"conditions {len(conditions)}: {conditions[0]} ... {conditions[-1]}")
    print(f"seeds {seeds[0]}-{seeds[-1]} ({len(seeds)})")
    print(f"candidate episodes: {len(baselines)*len(conditions)*len(seeds)}")
    print("=" * 78)

    # Rebuild every condition's schedule once from today's config.
    fresh = {}
    for condition in conditions:
        # Commands depend on the seed, so rebuild per (condition, seed) below;
        # the description does not, except for the seed field.
        fresh[condition] = {"description": None, "arrays": None}

    asset_cache: dict = {}
    adapter_cache: dict = {}
    rows, failures = [], []
    n_initial_checked = 0

    for seed in seeds:
        # Per-seed schedules from today's configuration.
        for condition in conditions:
            scen = E.build_scenario("E3", exp_cfg, condition, seed)
            fresh[condition] = {
                "description": dict(scen.description),
                "arrays": {k: np.asarray(v) for k, v in scen.arrays().items()},
            }
        check_initial = (args.sample_initial_state == 0
                         or n_initial_checked < args.sample_initial_state)
        for baseline in baselines:
            for condition in conditions:
                ok, fail, detail = verify_episode(
                    baseline, condition, seed, exp_cfg, fresh,
                    asset_cache, adapter_cache,
                    recompute_initial_state=check_initial,
                )
                rows.append(
                    {
                        "experiment": "E3", "baseline": baseline,
                        "condition": condition, "seed": seed,
                        "verdict": "reuse" if ok else "rerun_required",
                        "n_failed_checks": len(fail),
                        "failed_checks": "|".join(fail),
                        "initial_state_recomputed": int(check_initial),
                        "status": detail.get("status", ""),
                        "n_eval_samples": detail.get("n_eval_samples", ""),
                        "survived": detail.get("survived", ""),
                        "termination_reason": detail.get("termination_reason", ""),
                        "commands_bitexact": int(bool(detail.get("commands_bitexact", False))),
                        "initial_qpos_sha256": detail.get("initial_qpos_sha256", ""),
                        "path": detail.get("path", ""),
                    }
                )
                if not ok:
                    failures.append(rows[-1])
        if check_initial:
            n_initial_checked += 1
        if (seed + 1) % 10 == 0 or seed == seeds[-1]:
            n_ok = sum(1 for r in rows if r["verdict"] == "reuse")
            print(f"  seeds up to {seed}: {n_ok}/{len(rows)} verified reusable", flush=True)

    # Pairing check: at a given seed every baseline must have started from the
    # identical initial state, or the paired comparison is not paired.
    pairing_problems = []
    by_key: dict = {}
    for row in rows:
        by_key.setdefault((row["condition"], row["seed"]), {})[row["baseline"]] = row[
            "initial_qpos_sha256"
        ]
    for (condition, seed), per_baseline in by_key.items():
        digests = {d for d in per_baseline.values() if d}
        if len(digests) > 1:
            pairing_problems.append(
                {"condition": condition, "seed": seed,
                 "digests": sorted(per_baseline.items())}
            )

    common.write_csv(os.path.join(args.out, "E3_reuse_verification.csv"), rows)
    if failures:
        common.write_csv(os.path.join(args.out, "E3_rerun_required.csv"), failures)

    n_ok = sum(1 for r in rows if r["verdict"] == "reuse")
    summary = {
        "candidate_episodes": len(rows),
        "verified_reusable": n_ok,
        "rerun_required": len(rows) - n_ok,
        "initial_state_recomputed_for_n_seeds": n_initial_checked,
        "initial_state_identical_across_baselines": not pairing_problems,
        "pairing_problems": pairing_problems[:20],
        "checks": [
            "C1 completeness", "C2 baseline identity", "C3 asset SHA-256",
            "C4 control timing", "C5 scenario description", "C6 commands bit-exact",
            "C7 seeded initial state", "C8 recorded samples",
            "C9 metrics reconstruction",
        ],
        "seeds": [seeds[0], seeds[-1]],
        "n_seeds": len(seeds),
    }
    common.write_json(os.path.join(args.out, "E3_reuse_verification.json"), summary)

    print("=" * 78)
    print(f"verified reusable : {n_ok}/{len(rows)}")
    print(f"rerun required    : {len(rows) - n_ok}")
    print(f"initial state recomputed for {n_initial_checked} seed(s) x {len(baselines)} baselines")
    print(f"initial state identical across baselines at every (condition, seed): "
          f"{not pairing_problems}")
    if failures:
        print("\nFirst failures:")
        for row in failures[:10]:
            print(f"  {row['baseline']}/{row['condition']}/seed_{row['seed']:03d}: "
                  f"{row['failed_checks']}")
    print("=" * 78)
    return 0 if n_ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
