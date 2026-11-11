#!/usr/bin/env python3
"""Archive the round-1/round-2 benchmark results before round 3 replaces them.

Round 3 changes the E1 and E2 command protocols (E1 base resampling 2-4 s ->
2-3 s, E1 EE target resampling 5 s -> 3 s, E2 per-axis command envelope
0.15/0.30/0.45 -> 0.20/0.40/0.60 m/s and rad/s) and fixes the official seed
pool at 0-49. The round-2 E1/E2 raw episodes are therefore measurements of a
DIFFERENT experiment and must not be mixed with round-3 data, and every
round-2 aggregate table or figure that used them is invalidated.

What this script does
---------------------
1. MOVES ``results/raw/E1`` and ``results/raw/E2`` into the archive. A move,
   not a copy: leaving them in place is the only way round-2 episodes could
   silently enter a round-3 aggregate.
2. MOVES the round-2 derived outputs (``tables``, ``figures``, ``summary``,
   ``showcase``, ``validation``) and the two round-1/2 write-ups into the
   archive. They move together, so the relative paths quoted inside
   ``RESULTS_ROUND2.md`` still resolve inside the archive.
3. LEAVES ``results/raw/E3`` in place. E3's protocol is unchanged, its episodes
   are deterministic in (baseline, condition, seed), and round 3 reuses seeds
   0-49 after an explicit configuration and integrity check
   (``round3/verify_e3_reuse.py``). Nothing about them is modified.
4. LEAVES ``results/raw/E4`` in place. E4 is out of scope for round 3.
5. Reconstructs the round-2 ``configs/*.yaml`` values into
   ``archive_round2/configs/`` so the archived raw data stays traceable to the
   configuration that produced it. Each archived episode's own
   ``metadata.json`` already records its full scenario description, which is
   the authoritative record; these files are a convenience copy.
6. Writes ``ARCHIVE_MANIFEST.json`` (episode counts, byte sizes, file counts,
   SHA-256 of every archived config) and ``ARCHIVE_NOTE.md``.

Every move is a rename within one filesystem, so it is atomic per tree and an
interruption cannot leave a half-copied episode. Re-running the script is a
no-op once the archive exists: it refuses to overwrite an existing archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(SIM_ROOT, "benchmark")
RESULTS = os.path.join(BENCH, "results")
ARCHIVE = os.path.join(RESULTS, "archive_round2")

# Round-2 config values, restored exactly. Each entry is
# (path, [(round3_text, round2_text), ...]).
ROUND2_CONFIG_PATCHES = {
    "E1.yaml": [
        (
            """base:
  # ROUND 3: base commands are resampled every 2-3 s (was 2-4 s in rounds 1-2),
  # drawn from a seeded uniform distribution over [2.0, 3.0] s. Nothing else
  # about the sampler changed: same RNG stream label, same unit-cube draw, same
  # piecewise-constant hold, no smoothing or interpolation added.
  segment_duration_range: [2.0, 3.0]""",
            """base:
  # Base moving: deterministic piecewise-random commands, resampled every
  # 2-4 s from a seeded uniform distribution over [2, 4] s.
  segment_duration_range: [2.0, 4.0]""",
        ),
        (
            """  # Random targets: resampled EXACTLY every resample_interval_s, drawn from the native""",
            """  # Random targets: resampled EXACTLY every 5.0 s, drawn from the native""",
        ),
        (
            """  # Training used ~8 s (ee_cycle_duration_s: 8.0). ROUND 3 uses 3.0 s (rounds
  # 1-2 used 5.0 s) for a more demanding dynamic-target evaluation with
  # IDENTICAL command semantics: the native sample-and-hold counter
  # index = (k + 1) // cycle_steps is preserved unchanged, so the target is
  # replaced instantaneously at the scheduled tick with no interpolation,
  # ramp or smoothing. At 50 Hz, cycle_steps = 150 ticks = exactly 3.00 s.
  resample_interval_s: 3.0""",
            """  # Training used ~8 s (ee_cycle_duration_s: 8.0). The benchmark deliberately
  # uses 5 s for a moderately more demanding evaluation with identical command
  # semantics.
  resample_interval_s: 5.0""",
        ),
    ],
    "E2.yaml": [
        (
            """base:
  # Unchanged in round 3: motion patterns, waveform generation, resampling
  # schedule and task structure are all preserved. ONLY the command amplitude
  # changed.
  segment_duration_range: [2.0, 4.0]
  # ROUND 3: per-axis ABSOLUTE command envelopes, in m/s (vx, vy) and rad/s
  # (yaw rate). Rounds 1-2 used fractions 0.25/0.50/0.75 of the 0.6 shared
  # range, i.e. 0.15/0.30/0.45 -- replaced by 0.20/0.40/0.60 below.
  #
  # These are PER-AXIS absolute limits on the commanded value, NOT a bound on
  # the Euclidean norm of the combined XY velocity: a translate+turn segment
  # may command |vx| = |vy| = the limit simultaneously.
  #
  # High = 0.60 equals baselines.yaml base_command_limits exactly, i.e. the
  # full shared supported training range of C1 and C2. It is never exceeded.
  motion_intensity:
    Low:
      max_abs_vx_mps: 0.20
      max_abs_vy_mps: 0.20
      max_abs_yaw_rate_radps: 0.20
    Medium:
      max_abs_vx_mps: 0.40
      max_abs_vy_mps: 0.40
      max_abs_yaw_rate_radps: 0.40
    High:
      max_abs_vx_mps: 0.60
      max_abs_vy_mps: 0.60
      max_abs_yaw_rate_radps: 0.60""",
            """base:
  segment_duration_range: [2.0, 4.0]
  # Intensities are fractions of the shared supported command range
  # (baselines.yaml base_command_limits).
  intensities:
    Low: 0.25
    Medium: 0.50
    High: 0.75""",
        ),
    ],
    "baselines.yaml": [
        (
            """# ROUND 3 (canonical): the official pool is seeds 0-49 inclusive, exactly 50
# seeds per condition, for E1, E2 and E3 alike.
#
# History: round 1 used 0-29 and round 2 extended it to 0-59. Round 3 fixes the
# pool at 50 seeds as specified for the final evaluation. E1 and E2 were re-run
# in full because their command schedules changed (E1 resampling intervals, E2
# command amplitudes). E3's protocol is unchanged, so its deterministic seeds
# 0-49 are reused verbatim after an explicit configuration + integrity check.
#
# E3 seeds 50-59 from round 2 still exist on disk and are deliberately EXCLUDED
# from every round-3 aggregate so that all three experiments report the same
# 50-seed population. They are not deleted, and the effect of excluding them is
# reported in ROUND3_REPORT.md rather than left implicit.
#
# The illustrative candidate pool is DISJOINT from the official pool, so a
# curated figure seed can never also be an official statistical sample.
# (E4 is out of scope for round 3; its round-1 tables are archived unchanged.)
official_seeds:
  start: 0
  count: 50                    # official quantitative seeds 0-49""",
            """# The official pool was extended from 30 to 60 seeds (0-59) in the second
# evaluation round. Seeds 0-29 were NOT re-run: the episodes are deterministic
# functions of (baseline, condition, seed), so the original files are reused
# verbatim and seeds 30-59 were added with --resume-missing.
#
# The illustrative candidate pool is DISJOINT from the official pool, so a
# curated figure seed can never also be an official statistical sample.
# (E4's earlier showcase search used the previous 0-99 pool; E4 is untouched in
# this round and its tables still reflect official seeds 0-29.)
official_seeds:
  start: 0
  count: 60                    # official quantitative seeds 0-59""",
        ),
    ],
}

MOVE_RAW = ["E1", "E2"]
KEEP_RAW = ["E3", "E4"]
MOVE_DERIVED = ["tables", "figures", "summary", "showcase", "validation"]
MOVE_DOCS = ["RESULTS_ROUND2.md", "RESULTS_SUMMARY.md"]


def tree_stats(path: str) -> dict:
    n_files = n_episodes = 0
    n_bytes = 0
    for root, _dirs, files in os.walk(path):
        if "metrics.json" in files:
            n_episodes += 1
        for name in files:
            n_files += 1
            try:
                n_bytes += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return {"files": n_files, "episodes": n_episodes, "bytes": n_bytes}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if os.path.exists(ARCHIVE):
        print(f"Archive already exists: {ARCHIVE}\nRefusing to overwrite it.")
        return 1

    staging = ARCHIVE + ".partial"
    if os.path.exists(staging):
        print(f"Stale staging directory {staging}; remove it first.")
        return 1
    os.makedirs(os.path.join(staging, "raw"))
    os.makedirs(os.path.join(staging, "configs"))

    manifest: dict = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": (
            "Round 3 replaces round 1/2 as the canonical dataset. E1 and E2 raw "
            "episodes measure a different command protocol and are archived; E3 "
            "raw is protocol-identical and stays in place for verified reuse."
        ),
        "moved_raw": {},
        "kept_in_place_raw": {},
        "moved_derived": {},
        "moved_docs": [],
        "round2_configs": {},
    }

    raw = os.path.join(RESULTS, "raw")
    for exp in MOVE_RAW:
        src = os.path.join(raw, exp)
        if not os.path.isdir(src):
            print(f"  [skip] {src} does not exist")
            continue
        info = tree_stats(src)
        dst = os.path.join(staging, "raw", exp)
        shutil.move(src, dst)
        manifest["moved_raw"][exp] = info
        print(f"  moved raw/{exp}: {info['episodes']} episodes, "
              f"{info['files']} files, {info['bytes']/2**30:.2f} GiB")

    for exp in KEEP_RAW:
        src = os.path.join(raw, exp)
        if os.path.isdir(src):
            info = tree_stats(src)
            manifest["kept_in_place_raw"][exp] = info
            print(f"  kept  raw/{exp}: {info['episodes']} episodes "
                  f"(not modified, not aggregated as round 2)")

    for name in MOVE_DERIVED:
        src = os.path.join(RESULTS, name)
        if not os.path.isdir(src):
            continue
        info = tree_stats(src)
        shutil.move(src, os.path.join(staging, name))
        manifest["moved_derived"][name] = info
        print(f"  moved {name}/: {info['files']} files")

    for name in MOVE_DOCS:
        src = os.path.join(RESULTS, name)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(staging, name))
            manifest["moved_docs"].append(name)
            print(f"  moved {name}")

    for name, patches in ROUND2_CONFIG_PATCHES.items():
        current = open(os.path.join(BENCH, "configs", name), encoding="utf-8").read()
        text = current
        for new_text, old_text in patches:
            if new_text not in text:
                raise SystemExit(
                    f"Cannot reconstruct the round-2 {name}: the round-3 text block "
                    f"was not found. Reconstruction aborted rather than guessed."
                )
            text = text.replace(new_text, old_text)
        out = os.path.join(staging, "configs", name)
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(text)
        manifest["round2_configs"][name] = {
            "sha256": sha256_file(out),
            "reconstructed_by_reversing_round3_edit": True,
        }
        print(f"  reconstructed configs/{name} as of round 2")

    for name in ("E3.yaml", "E4.yaml"):
        src = os.path.join(BENCH, "configs", name)
        dst = os.path.join(staging, "configs", name)
        shutil.copy2(src, dst)
        manifest["round2_configs"][name] = {
            "sha256": sha256_file(dst),
            "unchanged_in_round3": True,
        }
        print(f"  copied configs/{name} (unchanged in round 3)")

    with open(os.path.join(staging, "ARCHIVE_MANIFEST.json"), "w", encoding="utf-8") as h:
        json.dump(manifest, h, indent=2, sort_keys=True)
    with open(os.path.join(staging, "ARCHIVE_NOTE.md"), "w", encoding="utf-8") as h:
        h.write(_note(manifest))

    os.rename(staging, ARCHIVE)
    print(f"\nArchive complete: {os.path.relpath(ARCHIVE, SIM_ROOT)}")
    return 0


def _note(manifest: dict) -> str:
    moved = manifest["moved_raw"]
    kept = manifest["kept_in_place_raw"]
    lines = [
        "# Archived round-1 / round-2 results",
        "",
        f"Archived {manifest['created_utc']}.",
        "",
        "**These results are superseded. Do not cite them as the benchmark's",
        "current numbers and do not aggregate them together with round 3.**",
        "",
        "## Why",
        "",
        "Round 3 changed the E1 and E2 command protocols:",
        "",
        "| Setting | Round 1/2 | Round 3 |",
        "|---|---|---|",
        "| E1 base command resampling | 2-4 s | 2-3 s |",
        "| E1 EE random target resampling | 5.0 s | 3.0 s |",
        "| E2 per-axis command envelope | 0.15 / 0.30 / 0.45 | 0.20 / 0.40 / 0.60 |",
        "| Official seed pool | 0-29 (r1), 0-59 (r2) | 0-49 |",
        "",
        "Every E1 and E2 number in this directory was measured under the left-hand",
        "column, so it answers a different question than round 3 does. E3's protocol",
        "did not change.",
        "",
        "## What is here",
        "",
    ]
    for exp, info in sorted(moved.items()):
        lines.append(
            f"* `raw/{exp}/` -- {info['episodes']} episodes "
            f"({info['bytes']/2**30:.2f} GiB), moved out of `results/raw/` so they "
            f"cannot enter a round-3 aggregate."
        )
    for name, info in sorted(manifest["moved_derived"].items()):
        lines.append(f"* `{name}/` -- {info['files']} round-1/2 derived files.")
    for name in manifest["moved_docs"]:
        lines.append(f"* `{name}` -- the round-1/2 write-up, moved with its tables and")
        lines.append("  figures so its relative links still resolve inside this archive.")
    lines += [
        "* `configs/` -- the configuration that produced the archived data.",
        "  `E1.yaml`, `E2.yaml` and `baselines.yaml` are reconstructed by reversing the",
        "  round-3 edit; `E3.yaml` and `E4.yaml` are byte copies (unchanged in round 3).",
        "  The authoritative per-episode record is each episode's own `metadata.json`,",
        "  which stores its complete scenario description.",
        "",
        "## What stayed in place",
        "",
    ]
    for exp, info in sorted(kept.items()):
        why = (
            "protocol-identical in round 3; seeds 0-49 are reused verbatim after an "
            "explicit configuration and integrity check"
            if exp == "E3"
            else "out of scope for round 3 and untouched"
        )
        lines.append(f"* `results/raw/{exp}/` ({info['episodes']} episodes) -- {why}.")
    lines += [
        "",
        "E4's round-1 tables and figures are in this archive (`tables/E4_*`,",
        "`figures/E4_*`). They were not invalidated by a protocol change -- they are",
        "here only because round 3 moved the whole derived-output tree.",
        "",
        "## Where the current results are",
        "",
        "`benchmark/results/final_round3/` and `final_round3/ROUND3_REPORT.md`.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
