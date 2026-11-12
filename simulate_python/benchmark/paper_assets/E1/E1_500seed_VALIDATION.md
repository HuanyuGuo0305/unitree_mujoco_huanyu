# E1 extended to 500 paired seeds — validation report

**Scope.** E1 only. Policies, checkpoints, the XML, the runner, the command generator, the termination rules and the metric definitions are unchanged. E2 and E3 were not touched.

**Data.** Seeds 0–49 are the existing round-3 episodes in `results/raw/E1/`, reused without re-simulation or copying. Seeds 50–499 are 5,400 new episodes in `results/E1_500seed/raw/E1/`, run with:

`run.py --experiment E1 --baselines C1,C3,C5 --conditions T1,T2,T3,T4 --seed-start 50 --num-seeds 450 --headless --log-physics --jobs 12`

The log is `run_seeds_050_499.log`: 5,400/5,400 episodes, 0 runner failures. No file in the 50-seed tree or in `final_round3/` was written.

**Analysis.** `python3 -m benchmark.round3.e1_500seed`. Statistics follow the existing convention: mean, sample SD (ddof = 1), percentile bootstrap with 10,000 resamples and seed 20261106, paired by seed, and Wilson intervals for completion.

## Checks — all passed

| Check | Result |
|---|---|
| Episode count | 6,000 = 4 conditions × 3 controllers × 500 seeds; no duplicates or missing episodes |
| Seed pairing | for every (condition, seed): `scenario_commands.npz` byte-identical across C1/C3/C5, identical initial qpos/qvel, identical issued commands among full-length episodes |
| Protocol identity | every episode's non-outcome metadata equals its cell's seed-0 (original) episode: duration 30 s, window [3, 30) s, 50/250 Hz, git commit, asset SHA-256s, config, scenario parameters |
| Metric consistency | all four RMSEs recomputed from per-tick logs and matched to the run-time `metrics.json`; geodesic angle matches the online-logged value; mean(θ) reproduces the stored MAE; RMSE ≥ MAE |
| Window | every episode starts evaluation at 3 s; full-length episodes have 1,350 ticks; truncated ones end at termination |
| Continuity with the 50-seed results | seeds 0–49: all 600 × 5 per-seed values and the 50-seed mean, SD and CI are **bit-identical** to the published v2 (orientation RMSE) files |
| Simulation reproducibility | seeds 0–1 (24 episodes) re-simulated at `--jobs 12` are byte-identical to the original `--jobs 6` run; seeds 50, 275 and 499 (36 episodes) re-simulated are byte-identical to the new run. 360 file/array comparisons in total, including 250 Hz physics, 0 differences |
| Analysis reproducibility | rerun gives byte-identical CSV and TeX |
| Negative controls | a tampered command sequence, initial state, duration, git commit or `metrics.json` value, and a missing or duplicate episode, each triggers a failure |
| LaTeX | structurally valid; **not compiled** (no LaTeX installed) |

## Early terminations — included, not excluded

Every terminated episode is in the means, scored over its own valid window (the same rule as the 50-seed table). See `E1_500seed_completion_rates.csv`.

| | C1 | C3 | C5 |
|---|---|---|---|
| T1 | 500/500 | 499/500 | 497/500 |
| T2 | 500/500 | 499/500 | 500/500 |
| T3 | 500/500 | 495/500 | **403/500 (80.6 %)** |
| T4 | 499/500 | 482/500 | **342/500 (68.4 %)** |

Caveat: at T3 and T4, C5's RMSEs average windows that are often shorter, mean 23.4 s and 23.0 s versus 27 s. Its tracking error and its completion rate should be read together.

## 50 → 500 seeds

- **Rankings:** the full ordering of C1, C3 and C5 is unchanged in all 16 condition × metric cells. The bold entries are the same as in the 50-seed table.
- **Paired conclusions:** 9 of 48 changed. All 9 went from *not significant* to *significant* in the direction the 50-seed point estimate already showed. No significant result reversed and no sign flipped.
- **CI width:** median ratio 0.334 between 500 and 50 seeds, against √(50/500) = 0.316 expected.

| Cond. | Metric | Comparison | 50 seeds | 500 seeds |
|---|---|---|---|---|
| T1 | EE Pos | C3 − C1 | n.s. | C3 better, −0.0022 m [−0.0041, −0.0003] |
| T1 | EE Ori | C5 − C3 | n.s. | C3 better, +3.63° |
| T2 | EE Ori | C5 − C1 | n.s. | C1 better, +0.64° [+0.35, +0.92] |
| T3 | EE Pos | C3 − C1 | n.s. | C1 better, +0.0055 m [+0.0039, +0.0072] |
| T3 | EE Ori | C3 − C1 | n.s. | C3 better, −2.12° [−2.73, −1.55] |
| T3 | EE Ori | C5 − C1 | n.s. | C1 better, +2.20° [+1.24, +3.08] |
| T3 | EE Ori | C5 − C3 | n.s. | C3 better, +4.32° |
| T4 | Yaw | C5 − C3 | n.s. | C3 better, +0.0074 rad/s |
| T4 | EE Ori | C5 − C1 | n.s. | C1 better, +1.58° [+1.13, +2.03] |

## Outputs (this directory)

| File | Content |
|---|---|
| `E1_500seed_per_seed_6000rows.csv` | per-episode values: 4 RMSEs, orientation MAE kept for traceability, completion, window, source path |
| `E1_500seed_summary.csv` | mean, SD (ddof = 1), 95 % CI, median, min, max, n completed, per cell |
| `E1_500seed_paired_comparisons.csv` | C3−C1, C5−C1, C5−C3 × 4 conditions × 4 metrics; paired bootstrap CIs |
| `E1_500seed_completion_rates.csv` | completion with Wilson 95 % CI, termination reasons, terminated seed IDs |
| `E1_50seed_subset_paired_comparisons.csv`, `E1_50_vs_500_rankings.csv`, `E1_50_vs_500_paired_conclusions.csv` | the 50 → 500 comparison |
| `E1_500seed_tracking_rmse.tex` | IEEE `table*`, label `tab:wbc_tracking` |
| `raw/E1/` | the 5,400 new episodes (seeds 50–499) |
