# B2W+Z1 Whole-Body Control Benchmark (E1–E4)

A reproducible MuJoCo benchmarking framework for the six validated Sim2Sim
controllers C1–C6, built for the IEEE RA-L submission.

**The six controllers are not retrained, modified or reimplemented.** Each
baseline adapter is a port of its deployment script's inference and actuation
code and reads that script's *own* YAML config, and
`tests/test_bitexact_parity.py` proves that every adapter reproduces its
original script **bit-for-bit** (max |difference| = 0 across `qpos`, `qvel`,
`ctrl`, the raw policy action and the joint targets). Read
[`docs/interface_audit.md`](docs/interface_audit.md) first: it is the Phase-A
audit, and every design decision below traces to a section of it.

> ## E3 was rebuilt from scratch as E3 v3 (2026-11-11)
>
> All earlier E3 code and data (round-2/3 E3 v1 and E3 v2) were removed; see
> [`docs/E3_V3_CLEANUP.md`](docs/E3_V3_CLEANUP.md) (pre-cleanup code: git tag
> `e3-pre-v3-cleanup`). The current E3 is **E3 v3**: discrete full-pose EE
> waypoints, 5 base levels x 5 EE update intervals x 2 patterns x 3 controllers
> x seeds 0-99 (15,000 episodes) in [`results/E3_v3/`](results/E3_v3/), code in
> `e3v3/`, pre-registration in
> [`docs/E3_V3_PREREGISTRATION.md`](docs/E3_V3_PREREGISTRATION.md). Statements
> about E3 elsewhere in this README that are not in section 4 describe the
> removed protocol.
>
> ## Round 3 is the canonical dataset (E1, E2)
>
> **Current results: [`results/final_round3/`](results/final_round3/) and
> [`results/final_round3/ROUND3_REPORT.md`](results/final_round3/ROUND3_REPORT.md).**
> Official seeds are **0–49, exactly 50 per condition**, for E1, E2 and E3.
>
> Round 3 changed the E1 and E2 command protocols (E1 base resampling
> 2–4 s → **2–3 s**, E1 EE target resampling 5 s → **3.0 s**, E2 per-axis command
> envelope 0.15/0.30/0.45 → **0.20/0.40/0.60**). E1 and E2 were therefore re-run
> in full, 900 episodes. E3's protocol is byte-identical, so its 4,050
> deterministic episodes for seeds 0–49 were reused after a nine-point
> per-episode verification (`round3/verify_e3_reuse.py`), which included a
> **bit-exact** comparison of every stored command schedule against one
> regenerated from today's config.
>
> **Round 1/2 results are superseded and archived** at
> [`results/archive_round2/`](results/archive_round2/) — see its
> `ARCHIVE_NOTE.md`. Do not aggregate them together with round 3. E4 is out of
> scope for round 3; its round-1 tables are in the archive, unchanged.
>
> Sections 3–9 below describe the round-1/2 pipeline, which still runs and still
> produces the archived outputs. For round 3 use §3.0.

---

## 1. What was measured, and in which frame

| Quantity | Definition | Frame |
|---|---|---|
| `base_vxy_rmse` | `sqrt(mean ‖v_xy − v_xy_cmd‖²)` | **base body frame** (`R_baseᵀ v_world`); the yaw-only (PLB) projection is logged for every sample as `*_yawframe` |
| `base_wz_rmse` | `sqrt(mean (ω_z − ω_z_cmd)²)` | base body frame (identical to the `imu_gyro` z the policies observe) |
| `ee_pos_rmse` | `sqrt(mean ‖p_ee − p_ee_cmd‖²)` | **PLB frame** — the native command frame |
| `ee_ori_mae_deg` | `mean arccos((tr(R_cmdᵀ R_act) − 1)/2)` | PLB frame; geodesic distance on SO(3), not a quaternion difference |

The EE command is **not** base-frame and **not** a position+quaternion. It is a
9-D keypoint triplet in the PLB frame:

```
PLB frame : origin = [base_x, base_y, 0.0],  orientation = yaw-only(base_quat)
command   : (kp0, kp0 + R_ee·[0.30,0,0], kp0 + R_ee·[0,0,0.30])
```

verified numerically against the native command table
(`|kp1−kp0| = |kp2−kp0| = 0.300`, exactly orthogonal). The benchmark issues the
*same* 9-D vector to every controller, so the physical target is identical, and
reports the error in that same frame for all of them. See audit §4.

**Discovered base-velocity command range** (audit §3.4; the training configs are
not in this repository):

```
vx, vy ∈ [−0.6, +0.6] m/s      wz ∈ [−0.6, +0.6] rad/s
```

inferred from `configs/b2wz1_hl_transport_uan.yaml`'s `base_cmd_scale`, which
`b2wz1_hl_transport_uan.py:286-290` validates against as the *frozen-WBC base
bounds* for the frozen C1 policy. Declared once in
`configs/baselines.yaml:base_command_limits` and used unchanged by E1, E2 and
E3 for every baseline. If the Isaac Lab `UniformVelocityCommandCfg.ranges`
become available, change those three numbers and nothing else.

---

**Round 2 (60 official seeds for E1–E3, plus curated illustrative cases) is
written up in `results/RESULTS_ROUND2.md`, which supersedes
`results/RESULTS_SUMMARY.md` for E1–E3. E4 is unchanged at 30 seeds.**

## 2. Layout

```
benchmark/
├── configs/           baselines.yaml (global) + E1.yaml … E4.yaml
├── controllers/       baseline adapters: base.py, uan.py, c1_wbc.py, c3_vbc.py,
│                      c5_roboduet.py, registry.py
├── core/              frames.py, scenario.py, recorder.py, runner.py, stats.py
├── scenarios/         experiments.py  (condition enumeration + scenario building)
├── metrics/           (metrics live in core/recorder.py and core/stats.py)
├── plots/             style.py, figures.py, tables.py,
│                      figures_round2.py, tables_round2.py
├── round3/            THE CURRENT PIPELINE
│   ├── common.py            seed policy, episode loading, atomic writes
│   ├── export.py            per-seed CSVs + E1 per-command-interval diagnostics
│   ├── aggregate3.py        50-seed summaries and paired comparisons
│   ├── select3.py           illustrative-case selection (pre-declared criteria)
│   ├── tables3.py           IEEE tables
│   ├── figures3.py          publication figures
│   ├── verify_power.py      E2 power/acceleration vs the 250 Hz physics log
│   ├── compare_rounds3.py   round 2 vs round 3
│   └── validate.py          the 14 acceptance checks
├── tests/             test_scenarios.py, test_round3_protocol.py,
│                      test_round2_analysis.py, test_controller_parity.py,
│                      test_bitexact_parity.py, test_latex_structure.py
├── docs/              interface_audit.md
├── results/
│   ├── raw/                 E1, E2 (round-3 runs) + E3, E4 (unchanged)
│   ├── final_round3/        CANONICAL: E1/ E2/ E3/ tables/ figures/ showcase/
│   │                        validation/ ROUND3_REPORT.md
│   └── archive_round2/      superseded round-1/2 raw, tables, figures, configs
├── run.py             episode driver
├── archive_round2.py  one-shot archival of the round-1/2 results
├── aggregate.py       round-1/2 statistics, CSVs, LaTeX tables
├── plot.py            round-1/2 figures
├── e3v3/              E3 v3: screening, pilot, validation, analysis, figures
├── publish.py         the round-2 60-seed tables and figures
├── select_cases.py    round-2 illustrative-case selection
├── compare_rounds.py  round-2: did adding seeds 30-59 change any conclusion?
└── showcase.py        earlier curated-seed utility (kept for E4)
```

All commands below are run from `simulate_python/`.

---

## 3. Running the experiments

### 3.0 Round 3 — the canonical pipeline

Run from `simulate_python/`. Phases A and B (config + smoke) are already applied;
start at C to reproduce the dataset from scratch.

```bash
# Phase A -- archive round 1/2 (refuses to overwrite an existing archive)
python3 benchmark/archive_round2.py

# Phase B -- smoke test, 3 seeds
python3 benchmark/run.py --experiment E1 --baselines C1,C3,C5 \
    --seed-start 0 --num-seeds 3 --headless --jobs 6 --log-physics
python3 benchmark/run.py --experiment E2 --baselines C1,C2 \
    --seed-start 0 --num-seeds 3 --headless --jobs 6 --log-physics

# Phase C -- official evaluation, seeds 0-49 (900 episodes, ~6 min at --jobs 6)
python3 benchmark/run.py --experiment E1 --baselines C1,C3,C5 \
    --seed-start 0 --num-seeds 50 --headless --jobs 6 --log-physics --resume-missing
python3 benchmark/run.py --experiment E2 --baselines C1,C2 \
    --seed-start 0 --num-seeds 50 --headless --jobs 6 --log-physics --resume-missing

# (E3 is no longer part of the round-3 pipeline; see E3 v3 in section 4)

# Phases D-F -- per-seed export, statistics, selection, tables, figures
python3 -m benchmark.round3.export
python3 -m benchmark.round3.aggregate3
python3 -m benchmark.round3.select3
python3 -m benchmark.round3.tables3
python3 -m benchmark.round3.figures3

# Phase G -- validation and round comparison
python3 -m benchmark.round3.verify_power
python3 -m benchmark.round3.compare_rounds3
python3 -m benchmark.round3.validate          # the 14 acceptance checks
```

`validate.py` takes `--skip-parity` (reports V13 as SKIPPED, never as passed)
and `--quick` (scans the first 10 seeds for the per-episode checks).

Round-3 tests:

```bash
python3 benchmark/tests/test_round3_protocol.py     # 37 tests
python3 benchmark/tests/test_scenarios.py           # 59 tests
python3 benchmark/tests/test_latex_structure.py     # structural only; no LaTeX here
```

---

### Round 1/2 pipeline (archived outputs)

### Smoke tests first (3 seeds)

```bash
python3 benchmark/run.py --experiment E1 --baselines C1,C3,C5 --seed-start 0 --num-seeds 3 --headless --jobs 6
python3 benchmark/run.py --experiment E2 --baselines C1,C2   --seed-start 0 --num-seeds 3 --headless --jobs 6
python3 benchmark/run.py --experiment E4 --baselines C1       --headless
```

### Official quantitative evaluation (E1–E3: seeds 0–59; E4: seeds 0–29)

The official pool was extended from 30 to 60 seeds for E1, E2 and E3. Episodes
are deterministic functions of `(baseline, condition, seed)`, so seeds 0–29 are
**reused verbatim** — `--resume-missing` skips every completed episode and only
seeds 30–59 are simulated. E4 was not part of that round and its tables still
report seeds 0–29.

```bash
# the full official pool; already-complete episodes are skipped
python3 benchmark/run.py --experiment E1 --baselines C1,C3,C5 --seed-start 0 --num-seeds 60 --headless --jobs 6 --resume-missing
python3 benchmark/run.py --experiment E2 --baselines C1,C2   --seed-start 0 --num-seeds 60 --headless --jobs 6 --resume-missing
python3 benchmark/run.py --experiment E4 --baselines C1,C3,C5 --seed-start 0 --num-seeds 30 --headless --jobs 6 --resume-missing
```

Episode counts: E1 3×4×60 = 720, E2 2×3×60 = 360, E3 3×9×3×60 = 4860,
E4 3×1×30 = 90. On a 32-core machine with `--jobs 6` the 60-seed set is roughly
50 minutes from scratch, about 25 for the seeds 30–59 increment. **Keep `--jobs`
at 6–8**: each worker loads its own PyTorch and ONNX Runtime, and 24 workers
exhaust 30 GB of RAM (the process gets SIGKILLed).

The seed counts are fixed in advance (0–29, then 30–59). Comparisons are never
re-run until a preferred verdict appears; `compare_rounds.py` reports after the
fact whether any conclusion moved.

### Tables and figures

```bash
# core statistics, CSVs and the round-1 tables
python3 benchmark/aggregate.py --results benchmark/results/raw --tables
python3 benchmark/plot.py      --results benchmark/results/raw

# the 60-seed official tables, the round-2 E3 figures, and the curated-case
# tables and figures (reads the selection manifests)
python3 benchmark/publish.py

# did the extra seeds change anything?
python3 benchmark/compare_rounds.py

# individually
python3 benchmark/aggregate.py --experiments E1 --tables     # E1 compact IEEE table
python3 benchmark/aggregate.py --experiments E2 --tables     # E2 comparison table
python3 benchmark/plot.py      --experiments E4 --seed 0 --baseline C1 --snapshots
python3 benchmark/plot.py      --experiments E4 --seed 0 --baseline C1 --snapshots --video
python3 benchmark/plot.py      --side-by-side E1 --condition T4 --seed 12
```

### Curated illustrative seeds (figures only)

The candidate pool is **seeds 100–149, disjoint from the official pool**, so an
illustrative figure seed can never also be an official statistical sample.
`select_cases.py` raises rather than proceeding if the two pools overlap.

```bash
# simulate the candidate pool (--showcase-search reads the pool from baselines.yaml)
python3 benchmark/run.py --experiment E1 --baselines C1,C3,C5 --showcase-search --headless --jobs 6 --resume-missing
python3 benchmark/run.py --experiment E2 --baselines C1,C2   --showcase-search --headless --jobs 6 --resume-missing

# select, then render
python3 benchmark/select_cases.py --experiment E1 --experiment E2
python3 benchmark/publish.py --only selected
```

`select_cases.py` applies a **pre-declared** procedure — gates, a normalisation
scale taken from the pool itself, a worst-case margin over the scored rivals,
and one scoring rule per criterion — and writes
`results/showcase/<EXP>_selection_manifest.json` containing every candidate
seed, its gate outcome, its normalised margins, its score under every
criterion, the full top-10 ranking and the chosen seed. If no candidate
satisfies a criterion that is recorded as `selected: null` with the reason;
nothing is back-filled with a weaker case. Every table caption and figure
footer built from these seeds says *selected illustrative trials from a
separately evaluated candidate seed pool*. These seeds never enter any
aggregate statistic.

For the E2 leg-power-vs-time panels, re-run just the selected episodes with
`--log-physics` so the 250 Hz torques and velocities are stored; the episodes
are deterministic, so the re-run is identical to the original.

The earlier `showcase.py` utility is kept for E4, which was not re-selected in
this round. Note that it reads the same `showcase_seeds` key, so re-running it
now would report the pool as 100–149; the existing
`results/showcase/selected_seeds.json` was generated under the previous 0–99
pool and its recorded `candidate_seed_pool` reflects that. E4's curated seeds
therefore overlap E4's official pool — a limitation of the round-1 design that
round 2 fixed for E1–E3 and that should be fixed for E4 before E4's figures are
used in the paper.

### Other CLI flags

`--config` (override the experiment YAML) · `--conditions` ·
`--output-dir` · `--dry-run` (generate and save every command schedule without
running physics) · `--resume-missing` · `--log-physics` (also store the 250 Hz
arrays as `physics.npz`) · `--viewer` (one episode in the MuJoCo viewer) ·
`--verbose`.

### Tests

```bash
python3 benchmark/tests/test_scenarios.py                 # 58 unit tests, ~4 s
python3 benchmark/tests/test_round2_analysis.py           # 26 tests: paired signs, selection, pools
python3 benchmark/tests/test_latex_structure.py           # table structure (no LaTeX installed here)
python3 benchmark/tests/test_bitexact_parity.py           # strict parity, all six baselines
python3 benchmark/tests/test_controller_parity.py         # console-log parity
```

`test_latex_structure.py` checks environment nesting, brace and math-delimiter
balance, and that every `tabular` row has as many cells as its column
specification declares. **No LaTeX distribution is installed in this
environment, so the generated tables have not been compiled** — that check is
structural only and does not replace a `pdflatex` run.

---

## 4. Experiment designs

### E1 — tracking (C1, C3, C5)

2 × 2, kept separate everywhere: `T1` still/fixed, `T2` still/random,
`T3` moving/fixed, `T4` moving/random. 30 s episodes.

* **Base moving**: piecewise-constant commands, segment durations drawn from
  `U[2, 4]` s, amplitudes from the shared supported range above. Unit-test
  enforced.
* **Base still**: exactly zero.
* **EE random**: resampled **exactly every 5.000 s** (250 policy ticks),
  drawn from the native presampled reachable PLB keypoint table. **No
  interpolation, ramp, smoothing or trajectory generator** — this is the native
  `ee_command_mode: "direct"` sample-and-hold behaviour, reproduced down to its
  off-by-one counter. Training used ~8 s; 5 s is a deliberately more demanding
  evaluation with identical command semantics.
* **EE fixed**: one row of the same table, held all episode, fixed in the PLB
  command frame (not the world frame). The arm is not frozen.
* Commands are identical for C1/C3/C5 at a given (condition, seed), and T1/T3
  share the EE target while T3/T4 share the base commands, so the conditions
  are paired.
* Transients after command switches are **included** in the primary metrics;
  post-switch error is reported separately as a diagnostic.

### E2 — active wheel actuation (C1 vs C2)

Fixed PLB EE target; three base-motion intensities at 0.25 / 0.50 / 0.75 of the
**shared** command range (C1 and C2 get the identical range, seeds and command
sequences; C2 is never given a wheel action). Each episode mixes straight
driving, turning and simultaneous translation + turning.

* Power proxy `P_abs(t) = Σ_j |τ_j q̇_j|` from `d.actuator_force` (the
  actuator-produced torque, including the wheel velocity servos) and `d.qvel`,
  at 250 Hz, averaged over the evaluated window, split into
  `leg_power` / `arm_power` / `wheel_power` / `gripper` / `total_power`.
  `wheel_power` is identically 0 for C2. **This is a mechanical power proxy,
  not battery power**, and is labelled as such everywhere.
* Joint acceleration RMS from MuJoCo's solved `d.qacc` at 250 Hz, **unfiltered**,
  over **matched physical DOFs only** — the 12 leg joints and the 6 arm joints.
  Wheels excluded; no normalisation across 22 vs 18 action dimensions. A
  finite-difference estimate at the same rate is stored alongside as
  `leg_acc_rms_fd` / `arm_acc_rms_fd` for cross-checking.

### E3 (v3) — dynamic whole-body coordination under discrete full-pose EE commands (C1, C3, C5)

Pre-registered in `docs/E3_V3_PREREGISTRATION.md`; results and reports in
`results/E3_v3/` (`REPRODUCIBILITY_REPORT.md`, `SCREENING_REPORT.md`,
`CONCLUSIONS_RAL.md`, `figures/CAPTIONS.md`).

* 5 base-motion bounds B ∈ {0, 0.15, 0.30, 0.45, 0.60} (per-axis |vx|,|vy| ≤ B
  m/s, |wz| ≤ B rad/s; existing seeded U(2,4) s piecewise-constant generator)
  × 5 EE update intervals {Fixed, 6, 4, 3, 2} s × 2 spatial patterns (line,
  circle) × seeds 0–99. 30 s episodes, 3 s settle on the measured initial pose.
* Each pattern is 5 full 6-DoF waypoints (9-D PLB keypoints) switched directly,
  without interpolation (line ping-pong, circle cyclic); start waypoint and
  direction are counterbalanced over seeds. The waypoint sets were chosen by a
  controller-neutral kinematic screening (`e3v3/candidates.py`) and a pilot on
  seeds 9000–9002 (`e3v3/pilot.py`) and frozen in `configs/E3.yaml`.
* Metrics: E1's four RMSEs (EE orientation as SO(3) geodesic RMSE in rad), the
  two directional degradations (always with the absolute errors),
  switch-aligned event responses (isolated vs overlapping events) and EE
  target reaching with censoring.

```bash
python3 -m benchmark.e3v3.candidates           # kinematic screening (controller-blind)
python3 -m benchmark.e3v3.pilot run|select|freeze
python3 -m benchmark.e3v3.prerun_validation
python3 benchmark/run.py --experiment E3 --baselines C1,C3,C5 --seed-start 0 --num-seeds 100 \
        --headless --jobs 14 --resume-missing --output-dir benchmark/results/E3_v3/raw
python3 -m benchmark.e3v3.analyze              # validation + statistics
python3 -m benchmark.e3v3.figures
```

### E4 — world-frame EE stabilisation (C1; C3, C5 for context)

The arm is first commanded onto a well-centred, demonstrably reachable PLB
setpoint during the settling window; the world-frame target is then **locked to
the measured EE pose at the first evaluated tick** and held. At every control
update that fixed world target is transformed into the native PLB command frame
using the current measured base pose — the same injection point every other
experiment uses. No external EE stabilisation controller is added and the
policy and actuation pipeline are untouched.

Base motion is a bounded **cosine velocity** per phase, so the base *position*
oscillates symmetrically about its start and returns there each period
(forward/back → lateral → yaw → combined → forward/back → hold, 4 s each, at
0.5 of the command range, i.e. ±0.19 m and ±0.19 rad). Reachability is checked
every tick and reported as `target_outside_support_fraction` — measured 0.00 for
every baseline, with a maximum distance to the native command support of 0.04 m.

---

## 5. Statistics

* 30 paired seeds; **per-episode metrics first**, then aggregation across
  seeds. The experimental unit is the independently seeded episode — 50 Hz and
  250 Hz time samples are never treated as independent replicates
  (unit-test enforced).
* 95% **percentile bootstrap** CIs, 10,000 resamples, fixed bootstrap seed
  `20261106`, so reruns give identical intervals.
* Controller differences **preserve pairing**: the difference is taken seed by
  seed and only seeds present for both controllers are used.
* **Bolding rule**: a value is bolded only if it belongs to the set of methods
  *not statistically separated* from the point-estimate leader by a paired
  bootstrap 95% CI. A numerically smallest value is never bolded as
  conclusively superior when the uncertainty does not support it.
* Survival rates use a Wilson interval, not a bootstrap of a 0/1 mean.
* E3 results are reported **per trajectory family**; any aggregation across
  families is stated in the figure or table caption.
* Early-terminated episodes are **retained and reported** (`survival_rate`,
  `n_terminated`, `termination_reasons`, `valid_eval_duration_s`). No run is
  ever dropped for being unfavourable.

### Seeded initial conditions — why they exist

Without them, an episode with zero base command and a deterministic EE
trajectory (E3 at `alpha_base = 0`, and all of E4) is *identical* for every
seed, so 30 "paired seeds" would be 30 copies of one trial and every CI would
collapse to zero width. `configs/baselines.yaml:initial_state` adds a small
seeded perturbation (±0.02 rad joint position, +0–0.01 m base height, ±1° roll
and pitch). It is a function of the seed alone and is written in float64 for
every baseline, so **all six baselines start from exactly the same state at a
given seed** (unit-test enforced). Base yaw is not perturbed: flat ground is
yaw-symmetric, so it would decorrelate nothing. Set `enabled: false` to run from
the nominal posture.

---

## 6. Deliberate deviations from the task brief

Three, all flagged here and in the audit, none baseline-specific:

1. **Settling window is 3.0 s, not 2.0 s** (`configs/baselines.yaml:settle_s`).
   The native startup ramp (`startup_hold_s: 1.0` + `startup_blend_s: 2.0`)
   reaches full policy authority only at t = 3.0 s, and that ramp is part of the
   validated controllers. Evaluating from 2.0 s would include 1 s of partial
   policy authority. Set `settle_s: 2.0` to reproduce the brief literally; the
   value used is written into every `metadata.json`. Audit §10.2.
2. **E4 uses `settle_s: 5.0`** so the arm converges onto the PLB setpoint
   *before* the world-frame target is locked; otherwise the locked target
   inherits the startup transient.
3. **E4's base schedule is a cosine velocity with 4 s phases at 0.5 of the
   range**, rather than larger excursions, because a monotone 5 s velocity phase
   at 0.5 of the range displaces the base ~0.48 m and pushes the fixed
   world-frame target out of the reachable PLB region. Measured world EE
   position RMSE then dropped from 0.435 m to 0.082 m — the first number was
   measuring unreachability, not stabilisation.

Also worth knowing: **single-threaded math is pinned** (`OMP_NUM_THREADS=1`,
`torch.set_num_threads(1)`, `BENCHMARK_ORT_THREADS=1`). Thread count changes the
floating-point reduction order inside TorchScript and ONNX Runtime; without
pinning, episodes are not bit-reproducible. The parity tests pin the standalone
scripts the same way, and under that pinning the adapters are bit-exact.

---

## 7. Logging

```
benchmark/results/raw/E1/C1/T4/seed_000/
├── commands.csv             50 Hz: commanded base velocity + 9-D EE command (PLB and world)
├── states.csv               50 Hz: actual base velocity (body and yaw frame), base pose,
│                            roll/pitch/yaw, EE pose (PLB and world), EE errors, contacts
├── actuation.csv            50 Hz: per-group torque / acceleration / wheel velocity means,
│                            UAN saturation, IK delta, guidance angles, blend
├── metrics.json             every metric for this episode
├── metadata.json            baseline, framework, condition, seed, policy paths + SHA-256,
│                            XML path + SHA-256, UAN path, initial state (incl. the exact
│                            randomisation deltas), physics/control timestep, command
│                            generation parameters, command reference frames, git commit,
│                            termination reason, valid evaluation duration
└── scenario_commands.npz    the full generated command schedule
```

`--log-physics` adds `physics.npz` with the 250 Hz joint positions, velocities,
accelerations and actuator torques. The 250 Hz power and acceleration metrics
are accumulated online, so they are exact without needing that file.

Rerunning the same (experiment, condition, seed) reproduces identical target
trajectories — unit-test enforced.

---

## 8. Acceptance criteria → where they are checked

| # | Criterion | Where |
|---|---|---|
| 1 | Deterministic command generation | `test_scenarios.py::TestDeterminism` |
| 2 | Base velocity resampled every 2–4 s | `TestBaseResampling` |
| 3 | E1 EE resampled exactly every 5 s | `TestE1EECommands::test_e1_ee_switch_interval_exactly_5s` |
| 4 | **No external EE smoothing in E1** | `test_e1_has_no_interpolation_or_smoothing` |
| 5 | Identical commands across baselines | `TestCommandSharing` |
| 6 | PLB / base / world transforms | `TestFrames` (incl. a round-trip against C3's own function) |
| 7 | Quaternion conventions, orientation metric | `TestFrames::test_geodesic_*` |
| 8 | Tracking RMSE implementations | `TestMetricMath` |
| 9 | E3 trajectory geometry and speed control | `TestE3Trajectories` |
| 10 | 3 × 3 intensity matrix | `test_3x3_matrix_generation` |
| 11 | Joint torque and power measurement | `core/recorder.py` (uses `d.actuator_force`), E2 config |
| 12 | Joint acceleration estimation | `core/recorder.py` (`qacc` + finite-difference cross-check) |
| 13 | Early termination handling | `core/recorder.py`, `aggregate.py`, survival tables |
| 14 | Confidence interval calculation | `TestStatistics` |
| 15 | Official vs illustrative seed separation | `TestSeedPolicy`, `showcase.py` |
| 16 | C5/C6 dual-policy inference sequence | `c5_roboduet.py` (asserted single guidance injection per tick) + bit-exact parity |
| 17 | **Preservation of C1–C6 behaviour** | `test_bitexact_parity.py` — max difference 0 for all six |

No run is declared PASS on exit status alone: `core/runner.py::_validate_episode`
checks after every episode that the UAN ran exactly once per physics step, that
the policy tick count matches the episode length, and that the commands actually
fed to the controller equal the generated schedule element-wise.

---

## 9. Changes to existing files

Only one, strictly additive: a `--headless` flag was added to
`deploy_mujoco/b2wz1_wbc_plb_uan.py` (C1) and
`deploy_mujoco/b2wz1_wbc_plb_legged_uan.py` (C2), so the parity tests can run
them without a display. It defaults to off, leaving the standalone interactive
behaviour unchanged. C3–C6 already had the flag and were not modified. No
policy weights, control logic, config value or XML was changed.

The bit-exact parity test writes a temporary instrumented *copy* of each
deployment script into `deploy_mujoco/` and deletes it afterwards; the originals
are never modified.
