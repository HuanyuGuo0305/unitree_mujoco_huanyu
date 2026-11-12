# E2 (official): Wheeled-Legged vs Legged actuation, one dynamic condition (2026-11-12)

## 1. Replacement and archive

This is the official E2. It replaces two earlier versions, which are archived and not mixed with it:
- the E1-matched T1–T4 E2;
- the round-3 E2.

Both were moved out of the benchmark tree in the 2026-11-12 cleanup, to `unitree_mujoco_benchmark_archive_2026-11-12/results/E2_superseded/` next to the repository; each folder has its own README and checksums. Their code (experiment ids `E2_e1protocol` and `E2_round3`) is recoverable from git tag `benchmark-pre-cleanup-2026-11-12` and the code snapshot in that archive. The round-3 E2 code is also at tag `e2-round3-pre-replacement`.

Every file in `results/E2/` refers to this 100-pair experiment only. E1 and E3 scripts, configs, data, results and figures were not modified, rerun or deleted.

## 2. Protocol

**Modes.** Neither was retrained, and no models, gains or physics were changed.
- **Wheeled-Legged** = C1: unified whole-body PPO on `ground.xml`, with 4 wheel velocity servos.
- **Legged** = C2: the same framework, trained separately on `b2wz1_legged.xml`, with the wheels welded to the calves.

**Trials.** 100 paired trials, seeds 0–99, giving **200 episodes**. Each episode is 30 s; 0–3 s is settling (zero base command, measured initial EE pose) and the official window is **3–30 s**.

**Base commands.** E1's validated generator, `core.scenario.make_base_velocity_schedule`.
- Every U(2, 3) s, an independent uniform draw per axis (vx, vy, yaw rate) within the shared WBC training range of ±0.6 m/s, ±0.6 m/s and ±0.6 rad/s (`baselines.yaml`).
- Piecewise constant, no interpolation.
- Observed maxima: 0.5993, 0.5994 and 0.5997. About 10 switches per trial; mean commanded planar speed 0.46 m/s.
- Because this is E1's generator with its RNG stream, the base sequence equals E1's moving-base (T3/T4) sequence at the same seed.

**EE commands.** `make_ee_random_interval_schedule`, new and E2-only.
- A full 6-DoF pose (9-D PLB keypoint row) drawn uniformly from the native training keypoint table (sha256 `d822f110…c328`), using E1's sampling.
- Held for U(3, 5) s and applied directly, with no interpolation, through the existing PLB command pipeline.
- About 6.4 switches per trial; 723 distinct poses across 736 draws.
- It uses its own RNG stream, independent of the base stream.

**Initial state.** The seeded perturbation in `baselines.yaml` (wheel joints excluded) gives an identical base pose and identical positions for all 19 shared joints in both modes.

**Stored.**
- `protocol/E2_command_schedules_seeds000-099.npz`: every seed's base and EE commands, switch ticks (switch time = 3.0 + 0.02 × tick s) and EE table rows, generated before the run.
- `protocol/seeds.json`; config snapshots; `git_state.txt`; `code_sha256_at_launch.txt`.
- Initial states are in each episode's `metadata.json`.

**Run.** `python3 benchmark/run.py --experiment E2 --baselines C1,C2 --seed-start 0 --num-seeds 100 --headless --jobs 14 --log-physics --output-dir benchmark/results/E2/raw`. It took 0.7 min (`run_E2.log`).

## 3. Metrics (identical for both modes)

| # | Metric | Definition |
|---|---|---|
| 1 | **P_avg (W)** ↓ | Mean over 250 Hz physics steps of Σ_j \|τ_j q̇_j\|, with τ = `d.actuator_force` and q̇ = `d.qvel`. j covers 12 leg, 6 arm and 1 gripper actuators, plus 4 wheels for Wheeled-Legged. This is the validated online accumulator in `core/recorder.py`. It measures mechanical actuation effort, not battery power. |
| 2 | **Energy per realized progress (J/m)** ↓ | E / progress, with E = P_avg × T_valid (= ∫P dt over 3–30 s) and progress = Σ_k (v_k · ĉ_k) Δt. See the definition below. |
| 3 | **a_arm (rad/s²)** ↓ | Mean of \|`d.qacc`\| over 250 Hz steps and the 6 arm joints (250 Hz physics log) |
| 4 | **a_leg (rad/s²)** ↓ | The same over the 12 hip, thigh and calf joints. Wheel joints are excluded. |

**Progress (metric 2).** Progress = Σ_k (v_k · ĉ_k) Δt over evaluated 50 Hz ticks, where:
- ĉ_k is the unit vector of the commanded planar body velocity (vx, vy) held at tick k;
- v_k is the actual planar body velocity;
- Δt = 0.02 s.

How it behaves under changing commands:
- **Direction changes:** each tick is projected onto the direction commanded *at that tick*, so progress follows the command as it changes.
- **Motion against the command:** subtracts. Clipping it would reward overshoot and slow transitions.
- **Zero planar command:** adds nothing; yaw-only effort still counts in the energy. No tick in this experiment had a zero planar command (minimum 0.024 m/s).
- **Sway:** fluctuations orthogonal to the command cancel, so sway does not inflate progress. Path length is inflated by sway.
- **Diagnostics recorded:** commanded distance Σ\|c\|Δt, progress ratio, path length and net displacement.

**Verification under time-varying commands** (`validation/E2_validation.json`):
- **Alternative estimators:** progress from yaw-frame velocity, and from world-frame position increments projected on R_yaw ĉ, agree with the official value within 2.4 % in every trial (median +0.7 % to +1.0 %).
- **J/m under each estimator:** the Legged − Wheeled-Legged difference is +5.6 % (official), +5.9 % (yaw-frame) and +5.9 % (position-based).
- **Denominator:** progress is positive in all 200 trials (minimum 7.8 m).
- **Negative ticks:** about 2.1 % of ticks have negative progress in both modes.
- **Transients:** about 16 % of progress accrues in the first 0.5 s after base switches (about 20 % of the time), similar in both modes.

**Limitation of metric 2.** The energy includes yaw-tracking and EE-tracking effort, but the denominator counts planar progress only. It is therefore a whole-task energy per metre, not a pure cost of transport. It is identical for both modes.

**Statistics.**
- Per mode: mean ± SD, median and 95 % bootstrap CI.
- Paired Δ = Legged − Wheeled-Legged per seed, with a 95 % paired-bootstrap CI (10,000 resamples, seed 20261106).
- Paired t-test, Holm-corrected across the 4 primary comparisons. Wilcoxon signed-rank with Holm as a robustness check.
- Secondary diagnostics are paired but not corrected.

## 4. Validation (all passed)

| Check | Result |
|---|---|
| Exactly 100 paired trials | 200/200 episodes; 100 pairs |
| 1. Identical commands | Identical between modes 100/100; equal to the pre-generated file 100/100; regeneration bit-identical 100/100. The commands actually applied (`commands.csv`) equal the schedule in all 200 episodes. |
| 2. Initial conditions | Base pose and all 19 shared joints identical (max diff 0.0); identical seeded randomisation 100/100 |
| 3. Training range | Base commands ≤ 0.6 per axis. EE targets are training-table rows. |
| 4. Identical definitions | One code path for both modes (`e2/analyze.py`) |
| 5. Same interval | 3.000–30.000 s; 6,751 physics samples and 27.0 s in every episode |
| 6. Deterministic | See check 1. Unit tests: `tests/test_e2.py` (92 tests in the three suites pass). |
| 7. Actuator sets | Leg + arm + gripper (+ wheel). Legged wheel power is exactly 0; Wheeled-Legged wheel power ≥ 18.6 W. Power rebuilt from the log matches the online value to 1.4 × 10⁻⁹. |
| 8. Failures | 0 falls or terminations in either mode. Nothing excluded. |
| Acceleration cross-check | Finite difference vs `qacc`. Arm: median ≤ 0.04 %, max 0.28 %. Legs: Wheeled-Legged median −0.7 %, max 1.1 %; Legged 0.015 %. |

## 5. Results (100 paired trials; mean ± SD)

| Metric | Wheeled-Legged | Legged | Δ Legged − Wheeled-Legged [95 % CI] | rel. | Holm p (t) | Holm p (Wilcoxon) | Lower |
|---|---|---|---|---|---|---|---|
| P_avg (W) | 213.8 ± 29.9 | **207.2 ± 25.2** | −6.62 [−9.01, −4.15] | −3.1 % | 1.5 × 10⁻⁶ | 1.8 × 10⁻⁶ | **Legged** (76 % of pairs) |
| Energy / progress (J/m) | **483.6 ± 51.3** | 510.8 ± 39.1 | +27.3 [+20.6, +34.1] | +5.6 % | 1.6 × 10⁻¹¹ | 1.8 × 10⁻¹⁰ | **Wheeled-Legged** (80 %) |
| a_arm (rad/s²) | **7.91 ± 0.99** | 8.57 ± 0.98 | +0.66 [+0.58, +0.75] | +8.4 % | 3.7 × 10⁻²⁶ | 2.3 × 10⁻¹⁶ | **Wheeled-Legged** (90 %) |
| a_leg (rad/s²) | 20.86 ± 2.75 | 21.04 ± 2.25 | +0.19 [−0.12, +0.50] | +0.9 % | 0.25 | 0.32 | No significant difference |

**Leg acceleration, finite-difference check.** The finite-difference estimate gives Δ = +0.34 [+0.03, +0.65] (+1.7 %, uncorrected p = 0.033). Either way the leg difference is below 2 % and not a meaningful effect.

### Diagnostics (paired, uncorrected)

| | Wheeled-Legged | Legged | Δ [95 % CI] |
|---|---|---|---|
| Completion / falls | 100/100 / 0 | 100/100 / 0 | n/a |
| Base XY RMSE (m/s) | 0.152 ± 0.023 | **0.142 ± 0.024** | −0.0104 [−0.0119, −0.0088] (−6.8 %) |
| Yaw-rate RMSE (rad/s) | **0.0764 ± 0.0119** | 0.0782 ± 0.0116 | +0.0019 [+0.0010, +0.0028] (+2.5 %) |
| EE position RMSE (m) | **0.173 ± 0.040** | 0.177 ± 0.041 | +0.0046 [+0.0034, +0.0059] (+2.7 %) |
| EE orientation RMSE (deg) | **29.7 ± 8.8** | 30.1 ± 8.9 | +0.40 [+0.10, +0.69] (+1.3 %) |
| Commanded planar distance (m) | 12.41 | 12.41 | 0 (identical commands) |
| Realized progress (m) | **11.96 ± 1.29** | 10.97 ± 1.23 | −0.99 [−1.04, −0.94] |
| Progress ratio | **0.965 ± 0.025** | 0.884 ± 0.018 | −0.080 [−0.084, −0.076] (lower in 100 % of pairs) |
| Path length (m) | 12.55 | 11.44 | −1.12 (−8.9 %) |
| Net displacement (m) | 3.33 | 3.09 | −0.24 |
| Energy over 27 s (J) | 5772 | 5593 | −179 [−243, −112] |
| Leg power (W) | 178.2 | 201.5 | +23.3 [+20.4, +26.3] (+13.1 %) |
| Arm power (W) | 5.29 | 5.65 | +0.36 [+0.30, +0.42] (+6.9 %) |
| Wheel power (W) | 30.3 | 0 | n/a |
| Gripper power (W) | 1.6 × 10⁻⁵ | 1.4 × 10⁻⁵ | negligible |

## 6. Interpretation (numerical, for checking; not paper text)

**Do the previous E2 findings persist?** All three do, at nearly the same magnitudes as T3/T4 of the archived T1–T4 E2.
- **Average power:** Legged is lower by 3.1 %, against 2.7–3.3 % before.
- **Energy per progress:** Wheeled-Legged is lower by 5.6 %, against 5.3–5.6 % before.
- **Arm acceleration:** Wheeled-Legged is lower by 8.4 %, against 7.0–8.0 % before.
- **Leg acceleration:** no significant difference, as before in the moving conditions.

**Why Legged uses less power.** Legged realizes about 8 % less of the commanded planar motion: progress ratio 0.88 vs 0.96. Its lower average power therefore comes with less executed motion. Per metre actually progressed, Wheeled-Legged needs 5.6 % less mechanical energy. It moves about 23 W of leg power into about 30 W of wheel power, and it covers more ground with it.

**Tracking trade-off.** Legged has lower base-XY velocity RMSE (−6.8 %) despite undershooting the commanded speed: lower velocity fluctuation, larger mean speed deficit. Wheeled-Legged tracks yaw rate and the EE pose slightly better (1.3–2.7 %). Lower arm acceleration is not by itself evidence of better tracking.

## 7. Limitations

- **Policy and training differences.** The modes are separately trained policies on different morphologies, so the comparison includes policy and training differences, not actuation alone. Equal training ranges and rewards cannot be verified from this repository; the ±0.6 range is the documented shared benchmark range (`docs/interface_audit.md` §3).
- **Power is mechanical.** Σ\|τq̇\| includes no motor losses, holding torque or regeneration.
- **Metric 2.** Total energy (base, yaw and EE effort) is divided by planar progress. Its estimator was verified under time-varying commands (§3).
- **Command range.** The ±0.6 range was not narrowed for either mode. Legged executes it without falls but realizes only about 88 % of the commanded planar progress; this is reported, not corrected.
- **Shared base sequence.** The base command sequence at each seed coincides with E1's T3/T4 base sequence, because E1's generator is reused. The EE sequence is new.
- **Acceleration.** `qacc` is unfiltered. Contact impulses enter both modes.
- **Simulation only.**

## 8. Files (`simulate_python/benchmark/results/E2/`)

| Path | Content |
|---|---|
| `E2_summary_table.csv` | **The four primary metrics:** both modes (mean, SD), paired Δ, CI, relative Δ, p, Holm p, Wilcoxon (Holm), verdict, failures |
| `data/E2_per_trial_200.csv` | All 200 episodes: primary metrics, diagnostics, progress checks, power components, initial randomisation |
| `stats/E2_summary.csv` | Per-mode statistics for every metric |
| `stats/E2_paired_legged_minus_wheeled.csv` | Paired comparisons for every metric |
| `stats/E2_completion.csv` | Completion and falls |
| `validation/E2_validation.json` | Fairness and validity checks (§4) |
| `protocol/` | `E2.yaml` and `baselines.yaml` snapshots, `seeds.json`, `E2_command_schedules_seeds000-099.npz`, `git_state.txt`, `code_sha256_at_launch.txt` |
| `raw/E2/{C1,C2}/dynamic/seed_XXX/` | Per-episode logs, `scenario_commands.npz`, 250 Hz `physics.npz` |
| `figures/E2_PRELIMINARY_primary_metrics_paired.{pdf,png}` | **Preliminary** inspection figure only. The publication figure and table are not finalized. |
| `run_E2.log`, `MANIFEST_SHA256.txt`, `RAW_SHA256.txt` | Run log and checksums |

**Code:**
- `benchmark/configs/E2.yaml`;
- `benchmark/scenarios/experiments.py` (`build_e2`, `make_ee_random_interval_schedule`);
- `benchmark/e2/analyze.py`, `benchmark/e2/figures.py`;
- `benchmark/tests/test_e2.py`.

**Regenerate** (from `simulate_python/`): run the command in §2, then `python3 -m benchmark.e2.analyze`, then `python3 -m benchmark.e2.figures`.
