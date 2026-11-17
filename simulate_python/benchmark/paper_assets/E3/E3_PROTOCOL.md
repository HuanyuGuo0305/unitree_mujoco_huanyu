# E3 — Bidirectional whole-body tracking benchmark (7 × 7): protocol

**Status: official E3 since 2026-11-13.** It replaces the 5 × 5 random-full-pose E3 (see §9).

**What was frozen before the official run:**
- `configs/E3.yaml`, the scenario builder and the per-episode metric code;
- every command schedule, in `results/E3/protocol/E3_command_schedules.npz` (sha256 `ec089709…b8e5`);
- the code hashes (`protocol/code_sha256_at_launch.txt`).

This document was written while the simulation ran, before any results had been analysed.

## 1. Question

How do base-motion intensity and the EE command-update frequency each affect base tracking and EE tracking? The four comparisons are:

| | Base tracking | EE tracking |
|---|---|---|
| Increasing base-motion bound B | (a) | (b) |
| Increasing EE update frequency | (c) | (d) |

The controllers are C1 Unified RL (Ours), C3 RL+IK and C5 Cooperative RL. Their checkpoints and deploy configurations are used unchanged.

## 2. Design

7 base levels × 7 EE levels × 3 controllers × 100 paired seeds (0–99) = **14,700 episodes**. Conditions are named `B{i}T{j}`.

| Factor | Levels (index i / j = 0…6) |
|---|---|
| Base bound B | 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 |
| EE update frequency | Fixed (0 Hz), 1/6, 1/5, 1/4, 1/3, 1/2, 1 Hz. The corresponding intervals are —, 6, 5, 4, 3, 2 and 1 s. |

**Base commands.** These use the existing generator (`core/scenario.py::make_base_velocity_schedule`, stream `base_velocity`):
- piecewise-constant segments of U(2, 4) s;
- each axis drawn as U(−1, 1) × B in the body frame, so |vx|, |vy| ≤ B m/s and |wz| ≤ B rad/s;
- B = 0 commands exactly zero.

The switch times and the normalised waveform depend only on the seed, so every B level scales the same waveform. There is no interpolation.

**EE commands.**
- **Pose source:** full 6-DoF poses, as 9-D PLB keypoints, from the native training table `utilities/data/keypoints_plb_evaluation_filtered.npy`. Its sha256 is `d822f110…c328`, identical to the IsaacLab training file. The sampling set is the whole table: the pre-declared validity screen keeps all 20,000 rows (`configs/E3_feasible_rows.npy`).
- **Sequence:** per seed, 28 rows are drawn uniformly with replacement (stream `e3_random_pose`, independent of the base stream). Level T commands pose ⌊k / (T/Δt)⌋ at evaluation tick k, so every level uses a prefix of the same sequence. Fixed holds pose 0.
- **Switching:** direct sample-and-hold. The first switch is at t = 3 + T.
- **Switches in the 27 s window:** 0, 4, 5, 6, 8, 13 and 26 per episode, from Fixed to 1 Hz.

**Episodes.**
- **Timing:** 30 s, with physics at 250 Hz and the policy at 50 Hz.
- **Settling (0–3 s):** zero base command; the EE command is pose 0, so there is no command jump at 3 s.
- **Evaluation window:** 3–30 s (1,350 ticks).
- **Termination:** base height < 0.30 m or tilt > 50°.

**Unchanged from the 5 × 5 E3 (and E1).** Everything is the same as in the 5 × 5 E3 except the factor levels and the pose-sequence length:
- generators, streams and pose source;
- settling and the evaluation window;
- termination rule and metrics.

The pose stream draws sequentially, so the first 16 poses of each 28-pose sequence are exactly the 5 × 5 E3 sequence. Conditions present in both designs therefore reproduce the archived episodes bit for bit:
- B = 0 / Fixed;
- B = 0 / 2 s;
- B = 0.6 / 2 s.

This was checked in the smoke test (§7).

## 3. Pairing

- The EE schedule depends only on (T, seed).
- The base schedule depends only on (B, seed).
- The initial state depends only on the seed.
- The builder has no controller argument.

All three controllers therefore receive bit-identical commands and initial states. The analysis checks every episode's stored schedule against the frozen schedule file.

## 4. Figure slices (single source of truth)

All four figure groups are slices of the one 7 × 7 data set. No separate experiment sets are used.
- **(a), (b):** the vs-B slice, B0T5 … B6T5 (EE update 1/2 Hz). The two groups use the same 700 episodes per controller.
- **(c), (d):** the vs-frequency slice, B0T0 … B0T6 (B = 0). The two groups use the same 700 episodes per controller.

## 5. Metrics (identical to E1 and the 5 × 5 E3)

All four are computed over every evaluated tick from 3 to 30 s. Transients after command updates are included.
1. Base XY velocity RMSE [m/s], in the body frame.
2. Base yaw-rate RMSE [rad/s].
3. EE position RMSE [m], in the PLB frame.
4. EE orientation RMSE [rad]: the RMSE of the SO(3) geodesic angle between the commanded and measured frames.

**Terminated episodes** are kept and scored over their valid window. They are counted per condition as `n_terminated_early` and never excluded silently.

## 6. Statistics

**Per controller × condition:**
- the numbers of valid, completed, early-terminated and missing trials;
- mean, SD (ddof = 1) and median over seeds;
- 95 % percentile bootstrap CI of the mean (10,000 resamples, seed 20261106). The same resampled seeds are used for every controller (paired).
- completion, with a Wilson interval.

**Paired comparisons:**
- C3 − C1 and C5 − C1 on seed-level differences, with a bootstrap CI and a paired t-test.
- Holm correction within each metric × rival family over the 49 conditions; BH q-values are also reported.
- Relative difference and d_z. "Practical" means |relative difference| ≥ 10 %.

**Slice trends** (descriptive support for the figure):
- Within each slice, the paired change of each RMSE from the slice's lowest level (B = 0 or Fixed) to every other level, per controller (Holm within slice × metric × controller).
- The rival-minus-C1 difference of that change.

**Sensitivity:** completed-episodes-only re-analysis of the paired comparisons.

**Statistical unit:** paired simulation trials of fixed trained checkpoints, not independent training runs.

**Figure** (`wbc_coordination.pdf`):
- Lines are means over 100 seeds.
- Bands are 95 % bootstrap confidence intervals of the mean, the convention of the previous E3 figure. They are **not** standard deviations; SDs are reported in the tables.
- Frequencies are plotted at equally spaced categorical positions.

## 7. Pre-run validation (`e3/prerun_validation.py`, `protocol/prerun_validation.json`)

- **V1:** the pose table matches the training file, and the feasible set is the whole table.
- **V2:** keypoint ↔ pose round trip.
- **V3:** the seven B values; the bound holds; B = 0 is exactly zero; every level is the common waveform × B; segment durations are U(2, 4) s.
- **V4:** the seven frequencies and labels; switches fall exactly every T (1 Hz is every 50 ticks); Fixed never switches; every command is a table row.
- **V5:** two independent builds of all 4,900 schedules are bit-identical.
- **V6:** EE commands are identical across B and base commands identical across T; each level's EE rows are a prefix of the seed's sequence; the builder takes no controller argument.
- **V7:** jump-size distributions match across levels.
- **V8:** a smoke test of 36 episodes (B0T0, B0T5, B0T6, B3T5, B6T5, B6T6; seeds 0–1; all three controllers). It checks that:
  - logged commands equal the frozen schedule;
  - all controllers share the same initial state and commands;
  - RMSE recomputation matches the runner;
  - the overlapping conditions reproduce the archived 5 × 5 episodes bit for bit.
- **V9:** checkpoints, deploy configs, `baselines.yaml` and the pose table are unchanged against git HEAD (`protocol/frozen_artefacts_sha256.txt`).

## 8. Workflow (from `simulate_python/`)

```
python3 -m benchmark.e3.prerun_validation --freeze
python3 benchmark/run.py --experiment E3 --baselines C1,C3,C5 --conditions B0T0,B0T5,B0T6,B3T5,B6T5,B6T6 \
        --seed-start 0 --num-seeds 2 --headless --jobs 14 --output-dir <scratch>
python3 -m benchmark.e3.prerun_validation --episodes <scratch>
python3 benchmark/run.py --experiment E3 --baselines C1,C3,C5 --seed-start 0 --num-seeds 100 --headless --jobs 14
python3 -m benchmark.e3.analyze
python3 -m benchmark.e3.paper_figures
```

## 9. History

| Version | Design | Status |
|---|---|---|
| v1 / v2 | smooth / arc-length trajectories | removed 2026-11-11 |
| v3 | fixed line/circle waypoints | removed 2026-11-11 (tag `e3v3-pre-random-pose-replacement`) |
| 5 × 5 | random full-pose; B ∈ {0, 0.15, 0.30, 0.45, 0.60}; EE updates Fixed, 6, 4, 3, 2 s; RMSE and event-aligned analyses; 7,500 episodes | superseded 2026-11-13 and archived (see `docs/E3_BIDIRECTIONAL_REPLACEMENT.md`) |
| **7 × 7 (this)** | as 5 × 5, with seven levels per factor (adding 1/5 and 1 Hz); four RMSE comparisons; no event-aligned analysis | **official** |

The 7 × 7 design was chosen after the 5 × 5 results had been examined. It widens the grid and drops the event-aligned panels. The generators, pose source and metric definitions are unchanged, and no threshold was tuned on data.
