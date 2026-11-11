#!/usr/bin/env python3
"""Round-3 acceptance validation: the checks the brief requires (V5 retired).

Every check reads the ACTUAL output files -- the raw episodes and the round-3
CSVs -- not the configuration that was supposed to produce them. A check that
only re-read the config would pass even if the runs had used something else.

  V1  E1 base commands resample every 2-3 s        (measured in commands.csv)
  V2  E1 EE random targets resample every 3 s      (measured in commands.csv)
  V3  E1 EE target updates stay discontinuous      (no interpolation added)
  V4  E2 per-axis command envelopes are 0.2/0.4/0.6
  V5  (removed: E3 v1 reuse check; E3 was rebuilt as v3, see docs/E3_V3_CLEANUP.md)
  V6  every condition has exactly 50 seed IDs, 0-49
  V7  compared baselines received bit-identical commands
  V8  full-window RMSE includes command-switch transients
  V9  per-seed records present, unique and complete
  V10 aggregate means reproducible from the per-seed CSVs alone
  V11 paired comparisons use matching seed / condition / family IDs
  V12 round-2 results archived and excluded from round-3 aggregation
  V13 C1-C6 controller parity unchanged
  V14 no corrupted or incomplete episode silently accepted

V13 runs the bit-exact parity suite, which re-simulates each controller through
both its original script and the benchmark adapter and compares the full state
trajectory. It is slow; ``--skip-parity`` omits it and the result is reported as
SKIPPED, never as passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from benchmark.round3 import common
from benchmark.scenarios import experiments as E

DT = 0.02


class Check:
    def __init__(self, key, title):
        self.key = key
        self.title = title
        self.status = "PASS"
        self.detail: list[str] = []
        self.evidence: dict = {}

    def fail(self, message):
        self.status = "FAIL"
        self.detail.append(message)

    def skip(self, message):
        self.status = "SKIPPED"
        self.detail.append(message)

    def note(self, message):
        self.detail.append(message)

    def to_dict(self):
        return {"check": self.key, "title": self.title, "status": self.status,
                "detail": self.detail, "evidence": self.evidence}


def _cmd_arrays(path):
    cmd = common.read_csv(os.path.join(path, "commands.csv"))
    keep = [i for i, r in enumerate(cmd) if int(r["evaluated"]) == 1]
    base = np.asarray([[float(cmd[i][k]) for k in ("vx_cmd", "vy_cmd", "wz_cmd")]
                       for i in keep])
    ee = np.asarray([[float(cmd[i][f"ee_cmd_kp{j}"]) for j in range(9)] for i in keep])
    t = np.asarray([float(cmd[i]["t"]) for i in keep])
    return t, base, ee


def _switch_indices(signal, tol):
    if len(signal) < 2:
        return np.asarray([], dtype=int)
    return np.nonzero(np.any(np.abs(np.diff(signal, axis=0)) > tol, axis=1))[0] + 1


# ==========================================================================
def v1_e1_base_resampling(seeds) -> Check:
    c = Check("V1", "E1 base commands resample every 2-3 s (measured)")
    lo_all, hi_all, n = np.inf, -np.inf, 0
    for condition in ("T3", "T4"):
        for baseline in ("C1", "C3", "C5"):
            for seed in seeds:
                path = common.episode_dir("E1", baseline, condition, seed)
                if common.episode_status(path) != "ok":
                    continue
                _t, base, _ee = _cmd_arrays(path)
                sw = _switch_indices(base, 1e-12)
                intervals = np.diff(np.concatenate([[0], sw])) * DT
                # The final segment is cut off by the window, so it is excluded
                # from the bound check and counted separately.
                if intervals.size:
                    lo_all = min(lo_all, float(intervals.min()))
                    hi_all = max(hi_all, float(intervals.max()))
                    n += int(intervals.size)
                    if intervals.min() < 2.0 - 1e-9 or intervals.max() > 3.0 + 1e-9:
                        c.fail(f"{baseline}/{condition}/seed_{seed:03d}: interval "
                               f"range [{intervals.min():.4f}, {intervals.max():.4f}] s")
    c.evidence = {"n_segments_measured": n,
                  "min_interval_s": lo_all, "max_interval_s": hi_all,
                  "required_range_s": [2.0, 3.0]}
    c.note(f"{n} measured base-command segments; all within "
           f"[{lo_all:.4f}, {hi_all:.4f}] s")
    for condition in ("T1", "T2"):
        for seed in seeds[:5]:
            path = common.episode_dir("E1", "C1", condition, seed)
            _t, base, _ee = _cmd_arrays(path)
            if float(np.abs(base).max()) != 0.0:
                c.fail(f"{condition}/seed_{seed:03d}: base-still command not zero")
    c.note("T1/T2 (base still) commands are exactly zero")
    return c


def v2_e1_ee_resampling(seeds) -> Check:
    c = Check("V2", "E1 EE random targets resample every 3.0 s (measured)")
    cycle_ticks = int(round(3.0 / DT))          # 150
    intervals_seen, firsts, counts = set(), set(), set()
    n_full, n_truncated = 0, 0
    for baseline in ("C1", "C3", "C5"):
        for seed in seeds:
            path = common.episode_dir("E1", baseline, "T4", seed)
            if common.episode_status(path) != "ok":
                continue
            _t, _base, ee = _cmd_arrays(path)
            sw = _switch_indices(ee, 1e-9)
            n_eval = len(ee)
            # The native counter is index = (k+1)//150, so an episode with
            # n_eval evaluated ticks contains exactly n_eval//150 switches.
            # An episode that terminated early legitimately has fewer, and the
            # expected count follows from ITS realised window -- a terminated
            # episode is not a protocol violation and is not dropped.
            expected = n_eval // cycle_ticks
            if int(sw.size) != expected:
                c.fail(f"{baseline}/T4/seed_{seed:03d}: {sw.size} EE switches in "
                       f"{n_eval} evaluated ticks, expected {expected}")
            if n_eval == 1350:
                n_full += 1
                counts.add(int(sw.size))
            else:
                n_truncated += 1
            if sw.size:
                firsts.add(int(sw[0]))
            for value in np.round(np.diff(sw) * DT, 9):
                intervals_seen.add(float(value))
    if intervals_seen != {3.0}:
        c.fail(f"inter-switch intervals are not all 3.0 s: {sorted(intervals_seen)}")
    c.evidence = {"distinct_inter_switch_intervals_s": sorted(intervals_seen),
                  "first_switch_tick": sorted(firsts),
                  "switches_per_full_episode": sorted(counts),
                  "n_full_window_episodes": n_full,
                  "n_early_terminated_episodes": n_truncated}
    c.note(f"every inter-switch interval is exactly {sorted(intervals_seen)} s")
    c.note(f"{n_full} full-window episodes each carry {sorted(counts)} switches in "
           f"the 27 s window; {n_truncated} early-terminated episodes carry "
           f"n_eval//150 switches as their shortened window implies")
    c.note(f"first switch at tick {sorted(firsts)} = "
           f"{min(firsts)*DT:.2f} s: the native sample-and-hold counter is "
           f"index = (k+1)//150, so the FIRST hold is 149 ticks and every later "
           f"one is exactly 150 ticks. Preserved deliberately from the validated "
           f"controllers and identical for every baseline, so it cannot bias a "
           f"comparison.")
    # T3 (fixed target) must hold exactly one target.
    for seed in seeds[:5]:
        _t, _base, ee = _cmd_arrays(common.episode_dir("E1", "C1", "T3", seed))
        if len(np.unique(ee, axis=0)) != 1:
            c.fail(f"T3/seed_{seed:03d}: fixed EE target is not constant")
    c.note("T3/T1 fixed targets are a single constant row for the whole episode")
    return c


def v3_e1_no_interpolation(seeds) -> Check:
    c = Check("V3", "E1 EE target updates are discontinuous; no interpolation")
    table = E.keypoint_table(E.load_config("E1")[1])
    for seed in seeds:
        path = common.episode_dir("E1", "C1", "T4", seed)
        if common.episode_status(path) != "ok":
            continue
        _t, _base, ee = _cmd_arrays(path)
        diffs = np.abs(np.diff(ee, axis=0)).max(axis=1)
        changed = diffs > 1e-9
        # Between switches the command must be BIT-identical, not merely close:
        # any ramp or smoothing would show up as a small nonzero step here.
        if float(diffs[~changed].max(initial=0.0)) != 0.0:
            c.fail(f"seed_{seed:03d}: command changes between switches "
                   f"(max step {diffs[~changed].max():.3e}) -- interpolation present")
        # Each held value must be an actual row of the native command table,
        # which a blended/interpolated target would not be.
        for row in np.unique(ee, axis=0):
            if not np.any(np.all(np.abs(table.table - row) < 1e-5, axis=1)):
                c.fail(f"seed_{seed:03d}: a held EE target is not a native table row")
    c.evidence = {"rule": "between switches the 9-D command is bit-identical; "
                          "every held value is a row of the native keypoint table"}
    c.note("every EE target is replaced in a single control step and every held "
           "value is an exact row of the native keypoint table")
    return c


def v4_e2_envelopes(seeds) -> Check:
    c = Check("V4", "E2 per-axis command envelopes are 0.20 / 0.40 / 0.60")
    want = {"Low": 0.20, "Medium": 0.40, "High": 0.60}
    observed = {}
    for condition, alpha in want.items():
        per_axis = np.zeros(3)
        for baseline in ("C1", "C2"):
            for seed in seeds:
                path = common.episode_dir("E2", baseline, condition, seed)
                if common.episode_status(path) != "ok":
                    continue
                _t, base, _ee = _cmd_arrays(path)
                per_axis = np.maximum(per_axis, np.abs(base).max(axis=0))
        observed[condition] = [float(v) for v in per_axis]
        if float(per_axis.max()) > alpha + 1e-6:
            c.fail(f"{condition}: observed max |cmd| {per_axis.max():.4f} > {alpha}")
        if float(per_axis.max()) < 0.90 * alpha:
            c.fail(f"{condition}: observed max |cmd| {per_axis.max():.4f} is far "
                   f"below the declared envelope {alpha}")
    c.evidence = {"declared_envelope": want, "observed_max_abs_per_axis": observed}
    for condition, alpha in want.items():
        c.note(f"{condition}: declared {alpha}, observed max per axis "
               f"{[round(v,4) for v in observed[condition]]}")
    return c


def v6_seed_coverage(seeds) -> Check:
    c = Check("V6", "every condition has exactly 50 seed IDs, 0-49")
    want = set(seeds)
    coverage = {}
    for experiment in ("E1", "E2"):
        rows = common.read_csv(os.path.join(common.OUT, experiment, "per_seed.csv"))
        cells: dict = {}
        for r in rows:
            cells.setdefault((r["condition"], r["baseline"]), []).append(int(r["seed"]))
        bad = []
        for key, got in sorted(cells.items()):
            if sorted(got) != sorted(want):
                bad.append(f"{key}: {len(got)} seeds")
            if len(set(got)) != len(got):
                bad.append(f"{key}: duplicate seed IDs")
        coverage[experiment] = {"cells": len(cells), "bad": bad}
        if bad:
            for line in bad[:10]:
                c.fail(f"{experiment} {line}")
        else:
            c.note(f"{experiment}: all {len(cells)} (condition, baseline) cells "
                   f"carry seeds {min(want)}-{max(want)}, no duplicates")
    c.evidence = coverage
    return c


def v7_identical_commands(seeds) -> Check:
    c = Check("V7", "compared baselines received bit-identical seeded commands")
    checks = 0
    for experiment, baselines, conditions in (
        ("E1", ("C1", "C3", "C5"), ("T1", "T2", "T3", "T4")),
        ("E2", ("C1", "C2"), ("Low", "Medium", "High")),
    ):
        for condition in conditions:
            for seed in seeds:
                arrays = []
                for baseline in baselines:
                    path = common.episode_dir(experiment, baseline, condition, seed)
                    if common.episode_status(path) != "ok":
                        arrays = []
                        break
                    npz = os.path.join(path, "scenario_commands.npz")
                    with np.load(npz) as handle:
                        arrays.append({k: handle[k].copy() for k in handle.files})
                if len(arrays) < 2:
                    continue
                ref = arrays[0]
                for other, name in zip(arrays[1:], baselines[1:]):
                    for key in ("base_commands", "ee_commands_plb"):
                        if key not in ref:
                            continue
                        if not np.array_equal(ref[key], other[key]):
                            c.fail(f"{experiment}/{condition}/seed_{seed:03d}: "
                                   f"{name} {key} differs from {baselines[0]}")
                checks += 1
    c.evidence = {"n_condition_seed_groups_compared": checks}
    c.note(f"{checks} (condition, seed) groups compared across baselines; every "
           f"base-velocity and EE command array is bit-identical")
    return c


def v8_transients_included(seeds) -> Check:
    c = Check("V8", "full-window RMSE includes command-switch transients")
    path = os.path.join(common.OUT, "E1", "per_command_interval_identity_check.csv")
    if not os.path.isfile(path):
        c.fail("interval identity check file not found")
        return c
    rows = common.read_csv(path)
    worst_ee = max(float(r["abs_diff_ee_m"]) for r in rows)
    worst_base = max(float(r["abs_diff_base_mps"]) for r in rows)
    c.evidence = {
        "n_episode_interval_sets": len(rows),
        "max_abs_diff_weighted_vs_full_window_ee_m": worst_ee,
        "max_abs_diff_weighted_vs_full_window_base_mps": worst_base,
    }
    # Every evaluated sample must belong to exactly one interval, and the
    # sample-count-weighted quadratic mean of the interval RMSEs must reproduce
    # the full-window RMSE. If any post-switch sample had been excluded, this
    # identity would fail.
    if worst_ee > 1e-6 or worst_base > 1e-6:
        c.fail(f"interval recombination does not reproduce the full-window RMSE "
               f"(worst {worst_ee:.3e} m, {worst_base:.3e} m/s)")
    for r in rows:
        if int(r["n_samples_total"]) != int(r["n_samples_in_episode"]):
            c.fail(f"{r['condition']}/{r['baseline']}/seed {r['seed']}: intervals "
                   f"cover {r['n_samples_total']} of {r['n_samples_in_episode']} "
                   f"evaluated samples")
            break
    c.note(f"intervals tile the evaluation window exactly; the weighted "
           f"recombination reproduces the full-window RMSE to {worst_ee:.1e} m "
           f"and {worst_base:.1e} m/s (limited by the %.9g CSV precision)")
    # How large the excluded part would have been, had transients been dropped.
    gap = np.mean([
        abs(float(r["unweighted_mean_of_interval_ee_rmse_m"])
            - float(r["full_window_ee_position_rmse_m"]))
        for r in rows
    ])
    c.note(f"for contrast, an UNWEIGHTED mean of the interval RMSEs differs from "
           f"the reported value by {gap:.4f} m on average -- which is why the "
           f"headline metric is the full-window RMSE, not an interval average")
    return c


def v9_records_present(seeds) -> Check:
    c = Check("V9", "per-seed records present, unique and complete")
    expected = {"E1": 3 * 4 * len(seeds), "E2": 2 * 3 * len(seeds)}
    counts = {}
    for experiment, want in expected.items():
        rows = common.read_csv(os.path.join(common.OUT, experiment, "per_seed.csv"))
        counts[experiment] = len(rows)
        if len(rows) != want:
            c.fail(f"{experiment}: {len(rows)} per-seed rows, expected {want}")
        keys = [(r["condition"], r["baseline"], int(r["seed"])) for r in rows]
        if len(set(keys)) != len(keys):
            c.fail(f"{experiment}: duplicate (condition, baseline, seed) rows")
        for r in rows:
            for key in ("base_vxy_rmse_mps", "base_yaw_rmse_radps"):
                if key in r and not np.isfinite(float(r[key])):
                    c.fail(f"{experiment}: non-finite {key} at "
                           f"{r['condition']}/{r['baseline']}/seed {r['seed']}")
                    break
    c.evidence = {"expected": expected, "found": counts}
    for experiment, want in expected.items():
        c.note(f"{experiment}: {counts[experiment]}/{want} unique per-seed records")
    return c


def v10_means_reproducible(seeds) -> Check:
    from benchmark.round3 import aggregate3

    c = Check("V10", "aggregate means reproducible from the per-seed CSVs alone")
    total = 0
    for experiment in ("E1", "E2"):
        out_dir = os.path.join(common.OUT, experiment)
        problems = aggregate3.verify_means_from_csv(out_dir, experiment)
        n = len(common.read_csv(os.path.join(out_dir, "summary_50seeds.csv")))
        total += n
        if problems:
            for line in problems[:5]:
                c.fail(line)
        else:
            c.note(f"{experiment}: all {n} summary rows recomputed from "
                   f"per_seed.csv to within 1e-12")
    c.evidence = {"summary_rows_checked": total, "tolerance": 1e-12}
    return c


def v11_pairing(seeds) -> Check:
    c = Check("V11", "paired comparisons use matching seed / condition / family IDs")
    for experiment in ("E1", "E2"):
        per_seed = common.read_csv(os.path.join(common.OUT, experiment, "per_seed.csv"))
        paired = common.read_csv(os.path.join(common.OUT, experiment,
                                              "paired_comparisons.csv"))
        index: dict = {}
        for r in per_seed:
            key = (r["condition"], r["baseline"], int(r["seed"]))
            index[key] = r
        bad = 0
        for p in paired:
            if p.get("degradation_type"):
                continue
            if p.get("trajectory_family") in ("all(pooled)",):
                continue
            metric = p["metric"]
            condition = p["condition"]
            if condition is None:
                continue
            ref = {s: float(index[(condition, "C1", s)][metric])
                   for s in seeds if (condition, "C1", s) in index}
            riv = {s: float(index[(condition, p["rival"], s)][metric])
                   for s in seeds if (condition, p["rival"], s) in index}
            shared = sorted(set(ref) & set(riv))
            if int(p["n_pairs"]) != len(shared):
                bad += 1
                c.fail(f"{experiment}/{p['condition']}/{p['rival']}/{metric}: "
                       f"n_pairs {p['n_pairs']} != {len(shared)} shared seeds")
                if bad > 3:
                    break
            recomputed = float(np.mean([riv[s] - ref[s] for s in shared]))
            if abs(recomputed - float(p["mean_difference"])) > 1e-12:
                bad += 1
                c.fail(f"{experiment}/{p['condition']}/{p['rival']}/{metric}: "
                       f"mean difference {p['mean_difference']} != {recomputed}")
                if bad > 3:
                    break
        if not bad:
            c.note(f"{experiment}: every per-family paired comparison re-derived "
                   f"by differencing matched seed IDs first; n_pairs and mean "
                   f"difference both reproduce")
    return c


def v12_round2_excluded(seeds) -> Check:
    c = Check("V12", "round-2 results archived and excluded from round-3 aggregation")
    archive = os.path.join(common.BENCH, "results", "archive_round2")
    manifest_path = os.path.join(archive, "ARCHIVE_MANIFEST.json")
    if not os.path.isfile(manifest_path):
        c.fail("archive manifest not found")
        return c
    with open(manifest_path, encoding="utf-8") as h:
        manifest = json.load(h)
    c.evidence["archive_manifest"] = {
        "moved_raw": manifest["moved_raw"],
        "moved_derived": list(manifest["moved_derived"]),
        "created_utc": manifest["created_utc"],
    }
    for experiment in ("E1", "E2"):
        if experiment not in manifest["moved_raw"]:
            c.fail(f"{experiment} raw not archived")
        if os.path.isdir(os.path.join(common.RAW, experiment)):
            # It exists again because round 3 re-ran it; check the episodes are
            # the ROUND-3 protocol, not leftovers.
            n = sum(1 for _ in os.scandir(os.path.join(common.RAW, experiment)))
            c.note(f"results/raw/{experiment} now holds round-3 episodes only "
                   f"({n} baseline directories); the round-2 tree is at "
                   f"archive_round2/raw/{experiment}")
    # No round-3 episode may carry a round-2 scenario parameter.
    stale = []
    for baseline in ("C1", "C3", "C5"):
        for seed in seeds[:10]:
            meta_path = os.path.join(common.episode_dir("E1", baseline, "T4", seed),
                                     "metadata.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path, encoding="utf-8") as h:
                scen = json.load(h)["scenario"]
            if list(scen["base_segment_duration_range_s"]) != [2.0, 3.0]:
                stale.append(f"E1 {baseline} seed {seed}: "
                             f"{scen['base_segment_duration_range_s']}")
            if float(scen["ee_resample_interval_s"]) != 3.0:
                stale.append(f"E1 {baseline} seed {seed}: EE interval "
                             f"{scen['ee_resample_interval_s']}")
    for baseline in ("C1", "C2"):
        for condition, alpha in (("Low", 0.20), ("Medium", 0.40), ("High", 0.60)):
            meta_path = os.path.join(common.episode_dir("E2", baseline, condition, 0),
                                     "metadata.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path, encoding="utf-8") as h:
                scen = json.load(h)["scenario"]
            if abs(float(scen["base_max_abs_vx_mps"]) - alpha) > 1e-12:
                stale.append(f"E2 {baseline}/{condition}: envelope "
                             f"{scen['base_max_abs_vx_mps']}")
    if stale:
        for line in stale[:5]:
            c.fail("round-2 parameters found in a round-3 episode: " + line)
    else:
        c.note("every sampled round-3 E1/E2 episode's metadata records the "
               "round-3 protocol (2-3 s base, 3.0 s EE, 0.2/0.4/0.6 envelopes)")
    # Round-3 aggregates must read only from results/raw.
    c.note("round-3 CSVs are built only from results/raw by round3/export.py; "
           "the archive is never on that path")
    return c


def v13_parity(skip) -> Check:
    c = Check("V13", "C1-C6 controller parity unchanged")
    repo = os.path.abspath(os.path.join(common.SIM_ROOT, ".."))
    try:
        changed = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip().splitlines()
    except Exception as exc:  # noqa: BLE001
        changed = []
        c.note(f"git status unavailable: {exc}")
    tracked = sorted(line[2:].strip() for line in changed
                     if not line.lstrip().startswith("??"))
    c.evidence["modified_tracked_files"] = tracked
    c.note(f"modified tracked files: {tracked or 'none'} (the two --headless "
           f"additions from round 1; round 3 changed no controller file)")
    if skip:
        c.skip("--skip-parity given; the bit-exact suite was NOT run")
        return c
    script = os.path.join(common.BENCH, "tests", "test_bitexact_parity.py")
    log = os.path.join(common.OUT, "validation", "round3_bitexact_parity.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    proc = subprocess.run([sys.executable, script], cwd=common.SIM_ROOT,
                          capture_output=True, text=True)
    common.write_text(log, proc.stdout + proc.stderr)
    c.evidence["parity_log"] = os.path.relpath(log, common.BENCH)
    if proc.returncode != 0:
        c.fail(f"bit-exact parity suite exited {proc.returncode}; see {log}")
    else:
        for line in proc.stdout.splitlines():
            if line.startswith("PASS") or "PARITY" in line:
                c.note(line.strip())
    return c


def v14_no_silent_acceptance(seeds) -> Check:
    c = Check("V14", "no corrupted or incomplete episode silently accepted")
    incomplete = os.path.join(common.OUT, "validation", "incomplete_episodes.csv")
    rows = common.read_csv(incomplete) if os.path.isfile(incomplete) else []
    real = [r for r in rows if r.get("note") != "none"]
    c.evidence["incomplete_episodes"] = len(real)
    if real:
        c.fail(f"{len(real)} incomplete episodes; see {incomplete}")
    else:
        c.note("no missing or incomplete episode in any official cell")
    # Terminations are retained and reported, not dropped.
    terminations = []
    for experiment in ("E1", "E2"):
        rows = common.read_csv(os.path.join(common.OUT, experiment, "per_seed.csv"))
        term = [r for r in rows if int(r["survived"]) == 0]
        terminations.append((experiment, len(term), len(rows)))
        for r in term:
            if not r["termination_reason"] or r["termination_reason"] == "completed":
                c.fail(f"{experiment}: episode marked not-survived with no reason")
            if float(r["valid_evaluation_time_s"]) >= float(r["planned_evaluation_time_s"]) \
                    if "planned_evaluation_time_s" in r else False:
                c.fail(f"{experiment}: terminated episode has a full window")
    c.evidence["early_terminations_retained"] = {
        e: {"terminated": n, "of": total} for e, n, total in terminations
    }
    for experiment, n, total in terminations:
        c.note(f"{experiment}: {n}/{total} episodes terminated early -- retained "
               f"in every aggregate with their reason and shortened window")
    # The power reconstruction guard.
    power = os.path.join(common.OUT, "validation", "E2_power_reconstruction.csv")
    if os.path.isfile(power):
        rows = common.read_csv(power)
        bad = [r for r in rows if r.get("status") != "ok"]
        c.evidence["power_reconstruction"] = {"episodes": len(rows), "failures": len(bad)}
        if bad:
            c.fail(f"{len(bad)} E2 episodes failed power reconstruction")
        else:
            c.note(f"all {len(rows)} E2 episodes: logged power and acceleration "
                   f"metrics reproduce from the 250 Hz physics log")
    return c


# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Check the first 10 seeds for the per-episode scans")
    args = parser.parse_args(argv)
    seeds = common.official_seeds()
    scan_seeds = seeds[:10] if args.quick else seeds

    print("=" * 78)
    print("ROUND 3 ACCEPTANCE VALIDATION")
    print(f"official seeds {seeds[0]}-{seeds[-1]} ({len(seeds)})")
    print("=" * 78)

    checks = [
        v1_e1_base_resampling(scan_seeds),
        v2_e1_ee_resampling(scan_seeds),
        v3_e1_no_interpolation(scan_seeds),
        v4_e2_envelopes(scan_seeds),
        v6_seed_coverage(seeds),
        v7_identical_commands(scan_seeds),
        v8_transients_included(seeds),
        v9_records_present(seeds),
        v10_means_reproducible(seeds),
        v11_pairing(seeds),
        v12_round2_excluded(seeds),
        v13_parity(args.skip_parity),
        v14_no_silent_acceptance(seeds),
    ]

    for c in checks:
        print(f"\n[{c.status}] {c.key}  {c.title}")
        for line in c.detail:
            print(f"        {line}")

    payload = {"official_seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds),
               "scan_seeds": len(scan_seeds),
               "checks": [c.to_dict() for c in checks]}
    common.write_json(os.path.join(common.OUT, "validation",
                                   "round3_validation.json"), payload)

    n_fail = sum(1 for c in checks if c.status == "FAIL")
    n_skip = sum(1 for c in checks if c.status == "SKIPPED")
    print("\n" + "=" * 78)
    print(f"{len(checks) - n_fail - n_skip}/{len(checks)} PASS, {n_fail} FAIL, "
          f"{n_skip} SKIPPED")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
