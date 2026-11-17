# E3 — Bidirectional whole-body tracking (7 × 7): results report

The protocol is in `docs/E3_PROTOCOL.md`; this report covers execution, data, statistics and findings. All numbers are read from `stats/` (written by `e3/analyze.py`). The tables below were generated from `stats/E3_summary.csv`, not typed in.

## 1. Execution

**Design:** 7 B levels {0, 0.1, …, 0.6} × 7 EE update frequencies {Fixed, 1/6, 1/5, 1/4, 1/3, 1/2, 1 Hz} × 3 controllers × 100 paired seeds (0–99).

**Episodes:** **14,700 / 14,700 simulated and analysed**. The run took 46.2 min at 14 workers (`run_E3.log`, runner exit 0). There were no missing or corrupt episodes.

**Pre-run validation** (`protocol/prerun_validation.json`) passed V1–V9, including a 36-episode smoke test. In the smoke test, the three conditions shared with the 5 × 5 E3 reproduced the archived episodes bit for bit.

**Post-run validation** (`validation/validation_summary.json`) passed:
- all 14,700 stored schedules equal the frozen schedule file (sha256 `ec089709…b8e5`);
- initial state and commands are identical across C1/C3/C5 in every condition × seed;
- one protocol-metadata set per controller;
- the recomputed RMSE matches the runner within 2.4e-8;
- issued commands equal the scheduled ones within 5e-9.

**Frozen artefacts:** checkpoints, deploy configs, `baselines.yaml` and the pose table are unchanged against git HEAD (`protocol/frozen_artefacts_sha256.txt`).

**Pose audit** (descriptive, `protocol/pose_audit/`): the 2,800 pose draws of seeds 0–99 cover 2,619 distinct targets.
- 89.1 % are reachable arm-only at ≥ 1 nominal stance.
- 0.8 % are reachable neither at a nominal stance nor at any logged E1 posture.
- 95.9 % are reachable at ≥ 1 logged moving-base posture.
- The feasible-row file (sha256 `d7887b88…`) is unchanged.

## 2. Completion

| Controller | Episodes | Completed | Terminated early | Missing |
|---|---|---|---|---|
| Unified RL (Ours) | 4,900 | 4,900 | 0 | 0 |
| RL+IK | 4,900 | 4,876 | 24 | 0 |
| Cooperative RL | 4,900 | 4,381 | 519 | 0 |

The 543 terminations (base height < 0.30 m or tilt > 50°) are listed in `validation/validation_summary.json`. These episodes are **kept** and scored over their valid window; none is excluded.

Cooperative RL terminates in 10–33 % of the seeds at B ≥ 0.3, except at 1 Hz (2–6 %). Its RMSEs at those levels therefore describe the episodes up to the fall.

A completed-episodes-only re-analysis of all 392 paired comparisons changes 4 verdicts (`stats/E3_sensitivity_completed_only.csv`). All four are Cooperative-RL EE orientation at 1/3 Hz, B ≥ 0.3, which move from "no significant difference" to "Unified RL better".

## 3. Figure

**Files:** `figures/paper/wbc_coordination.{pdf,svg,png}`. The PDF is 515.5 × 212.4 pt (7.16 × 2.95 in) with embedded STIX TrueType fonts. The plotted values are in `_values.csv` and the checks in `_checks.json`.

**Groups:**
- (a), (b): the vs-B slice, B0T5…B6T5 (1/2 Hz). (a) shows base XY / yaw RMSE; (b) shows EE position / orientation RMSE.
- (c), (d): the vs-frequency slice, B0T0…B0T6 (B = 0). (c) shows base and (d) shows EE tracking.

(a) and (b) share their episodes, as do (c) and (d).

**Lines and bands:**
- Lines are means over 100 seeds.
- **Bands are 95 % percentile bootstrap confidence intervals of the mean** (10,000 resamples, seed 20261106, paired). They are not SDs; SDs are in `stats/E3_summary.csv` and `stats/E3_summary_7x7.csv`.
- Every plotted mean and CI bound equals the official statistic exactly (max |difference| = 0, asserted in the script).

## 4. Findings

**Paired comparisons:** rival − Unified RL per condition, Holm-corrected within each metric × rival family over the 49 conditions (`stats/E3_paired_vs_C1.csv`).

| Metric | RL+IK vs Unified RL (49 conditions) | Cooperative RL vs Unified RL (49 conditions) |
|---|---|---|
| Base XY RMSE | Unified RL better in 49 | Unified RL better in 49 |
| Yaw-rate RMSE | Unified RL better in 49 | Unified RL better in 49 |
| EE position RMSE | Unified RL better in 42; **RL+IK better in 7** (every B with a Fixed EE target, −51 to −83 %) | Unified RL better in 49 |
| EE orientation RMSE | Unified RL better in 42; **RL+IK better in 7** (every B with a Fixed EE target, −55 to −71 %) | Unified RL better in 15; **no significant difference in 34** |

**Base-motion intensity → base tracking (a).** At 1/2 Hz, base XY RMSE grows roughly linearly with B for all controllers:

| Controller | Base XY RMSE, B = 0 → 0.6 (m/s) | Yaw RMSE, B = 0 → 0.6 (rad/s) |
|---|---|---|
| Unified RL | 0.028 → 0.143 | 0.024 → 0.074 |
| RL+IK | 0.041 → 0.157 | 0.044 → 0.092 |
| Cooperative RL | 0.065 → 0.179 | 0.085 → 0.105 |

The absolute increase in base XY error is nearly the same for all three (+0.115 m/s). The rival-minus-Unified-RL difference of that increase is not significant. Unified RL's relative advantage therefore narrows as B grows: RL+IK is +48 % worse at B = 0 and +10 % at B = 0.6, while Cooperative RL is +134 % and +26 % worse.

Cooperative RL's yaw error is high already at B = 0 and rises least (+0.021 rad/s against +0.050 for Unified RL; "Cooperative RL better" on the size of the change). Its absolute yaw RMSE is still higher at every B.

**Base-motion intensity → EE tracking (b).** At 1/2 Hz, EE errors are nearly flat in B. They fall slightly for all controllers between B = 0 and 0.6: Unified RL position −5 % (0.240 → 0.228 m) and orientation −4 % (0.683 → 0.655 rad). Base motion therefore does not degrade EE tracking under frequent EE updates for any controller.

The ordering is constant at every B:
- **Position:** Unified RL < Cooperative RL (+21–26 %) < RL+IK (+39–44 %).
- **Orientation:** Unified RL ≈ Cooperative RL (+3–6 %) < RL+IK (+30–34 %).

Cooperative RL's slightly larger EE-position decrease with B is partly confounded by its falls, which end those episodes' scoring early.

**EE update frequency → base tracking (c).** At B = 0, every EE update disturbs the stationary base, and base error rises monotonically with frequency for all controllers:

| Controller | Base XY RMSE, Fixed → 1 Hz (m/s) | Yaw RMSE, Fixed → 1 Hz (rad/s) |
|---|---|---|
| Unified RL | 0.015 → 0.032 | 0.006 → 0.031 |
| RL+IK | 0.025 → 0.041 | 0.013 → 0.050 |
| Cooperative RL | 0.037 → 0.070 | 0.027 → 0.089 |

Unified RL is lowest at every frequency. The yaw increase is significantly smaller for Unified RL than for both rivals; base XY increases less than Cooperative RL's and about as much as RL+IK's. Unified RL's base XY lead over RL+IK narrows at 1 Hz (+27 %, against +48–68 % at lower frequencies).

**EE update frequency → EE tracking (d).** EE errors rise steeply with update frequency, because every switch adds a reaching transient that is included in the RMSE. Unified RL position goes 0.052 → 0.324 m and orientation 0.239 → 0.871 rad from Fixed to 1 Hz.

- **Fixed target (0 Hz):** RL+IK is clearly the most precise (0.009 m and 0.069 rad, against 0.052 m and 0.239 rad for Unified RL). This holds at every B with a Fixed target, as in the 5 × 5 E3 and E1.
- **Every updating level (1/6–1 Hz):** Unified RL has the lowest EE position RMSE, by +18–47 % over RL+IK and +15–30 % over Cooperative RL. Unified RL's lead over RL+IK in orientation is +16–30 %.
- **Unified RL vs Cooperative RL in orientation:** statistically indistinguishable at B = 0, with the 1/2 Hz cell the only significant one (+3 %).

**Does Unified RL keep its advantage?**
- **Base tracking:** yes, in all 98 base-metric comparisons against each rival (all 49 conditions × 2 metrics).
- **EE position:** yes, whenever the EE target changes. When it is held fixed, RL+IK is better.
- **EE orientation:** yes against RL+IK whenever the target changes. Against Cooperative RL, Unified RL is better in 15 of 49 conditions and indistinguishable in the other 34.
- **Robustness:** Unified RL is also the only controller with no early terminations (0 / 4,900).

**Where the alternatives do better:**
- RL+IK: static EE pose holding (Fixed target, all 7 B levels; 14 of 98 EE comparisons).
- Cooperative RL: EE orientation is on par in 34 of 49 conditions.
- Cooperative RL's yaw error grows less with B, from a higher starting level.

Not every Unified RL advantage is practically large; relative differences are in `rel_diff`.

## 5. Tables (means over 100 seeds; CIs and SDs in `stats/E3_summary_7x7.csv`)

**Base XY velocity RMSE (m/s)**: mean over 100 seeds. Rows are B; columns are the EE update frequency in Hz.

| Controller | B | Fixed | 1/6 | 1/5 | 1/4 | 1/3 | 1/2 | 1 |
|---|---|---|---|---|---|---|---|---|
| Unified RL | 0 | 0.015 | 0.022 | 0.023 | 0.024 | 0.025 | 0.028 | 0.032 |
|  | 0.1 | 0.051 | 0.053 | 0.053 | 0.053 | 0.053 | 0.053 | 0.055 |
|  | 0.2 | 0.061 | 0.062 | 0.063 | 0.063 | 0.063 | 0.064 | 0.066 |
|  | 0.3 | 0.074 | 0.075 | 0.075 | 0.075 | 0.076 | 0.077 | 0.079 |
|  | 0.4 | 0.093 | 0.093 | 0.094 | 0.094 | 0.094 | 0.095 | 0.097 |
|  | 0.5 | 0.115 | 0.115 | 0.115 | 0.115 | 0.116 | 0.117 | 0.119 |
|  | 0.6 | 0.139 | 0.140 | 0.141 | 0.141 | 0.141 | 0.143 | 0.143 |
| RL+IK | 0 | 0.025 | 0.037 | 0.039 | 0.039 | 0.040 | 0.041 | 0.041 |
|  | 0.1 | 0.054 | 0.058 | 0.060 | 0.059 | 0.059 | 0.059 | 0.059 |
|  | 0.2 | 0.073 | 0.077 | 0.077 | 0.078 | 0.078 | 0.077 | 0.076 |
|  | 0.3 | 0.089 | 0.093 | 0.092 | 0.093 | 0.095 | 0.093 | 0.092 |
|  | 0.4 | 0.108 | 0.112 | 0.111 | 0.111 | 0.113 | 0.111 | 0.111 |
|  | 0.5 | 0.128 | 0.133 | 0.133 | 0.132 | 0.133 | 0.133 | 0.132 |
|  | 0.6 | 0.155 | 0.161 | 0.157 | 0.157 | 0.156 | 0.157 | 0.155 |
| Cooperative RL | 0 | 0.037 | 0.054 | 0.056 | 0.058 | 0.060 | 0.065 | 0.070 |
|  | 0.1 | 0.073 | 0.080 | 0.080 | 0.081 | 0.082 | 0.084 | 0.088 |
|  | 0.2 | 0.105 | 0.102 | 0.101 | 0.102 | 0.102 | 0.100 | 0.099 |
|  | 0.3 | 0.115 | 0.115 | 0.113 | 0.114 | 0.113 | 0.112 | 0.110 |
|  | 0.4 | 0.131 | 0.131 | 0.131 | 0.131 | 0.131 | 0.130 | 0.127 |
|  | 0.5 | 0.153 | 0.155 | 0.154 | 0.154 | 0.153 | 0.154 | 0.151 |
|  | 0.6 | 0.181 | 0.180 | 0.180 | 0.180 | 0.180 | 0.179 | 0.178 |

**Base yaw-rate RMSE (rad/s)**: mean over 100 seeds. Rows are B; columns are the EE update frequency in Hz.

| Controller | B | Fixed | 1/6 | 1/5 | 1/4 | 1/3 | 1/2 | 1 |
|---|---|---|---|---|---|---|---|---|
| Unified RL | 0 | 0.006 | 0.015 | 0.016 | 0.018 | 0.020 | 0.024 | 0.031 |
|  | 0.1 | 0.034 | 0.037 | 0.037 | 0.037 | 0.038 | 0.041 | 0.044 |
|  | 0.2 | 0.040 | 0.042 | 0.042 | 0.042 | 0.043 | 0.046 | 0.048 |
|  | 0.3 | 0.045 | 0.047 | 0.048 | 0.048 | 0.049 | 0.051 | 0.054 |
|  | 0.4 | 0.053 | 0.054 | 0.054 | 0.055 | 0.056 | 0.058 | 0.060 |
|  | 0.5 | 0.061 | 0.062 | 0.063 | 0.063 | 0.063 | 0.066 | 0.067 |
|  | 0.6 | 0.070 | 0.071 | 0.071 | 0.071 | 0.072 | 0.074 | 0.076 |
| RL+IK | 0 | 0.013 | 0.029 | 0.031 | 0.035 | 0.038 | 0.044 | 0.050 |
|  | 0.1 | 0.040 | 0.046 | 0.048 | 0.051 | 0.052 | 0.055 | 0.061 |
|  | 0.2 | 0.053 | 0.057 | 0.058 | 0.060 | 0.062 | 0.063 | 0.069 |
|  | 0.3 | 0.061 | 0.066 | 0.066 | 0.067 | 0.070 | 0.071 | 0.074 |
|  | 0.4 | 0.069 | 0.073 | 0.074 | 0.074 | 0.077 | 0.076 | 0.079 |
|  | 0.5 | 0.078 | 0.082 | 0.083 | 0.082 | 0.083 | 0.084 | 0.086 |
|  | 0.6 | 0.087 | 0.093 | 0.091 | 0.091 | 0.091 | 0.092 | 0.093 |
| Cooperative RL | 0 | 0.027 | 0.056 | 0.062 | 0.067 | 0.074 | 0.085 | 0.089 |
|  | 0.1 | 0.055 | 0.070 | 0.073 | 0.075 | 0.081 | 0.088 | 0.094 |
|  | 0.2 | 0.073 | 0.078 | 0.078 | 0.080 | 0.082 | 0.086 | 0.089 |
|  | 0.3 | 0.077 | 0.080 | 0.080 | 0.081 | 0.081 | 0.083 | 0.083 |
|  | 0.4 | 0.084 | 0.087 | 0.087 | 0.088 | 0.087 | 0.088 | 0.084 |
|  | 0.5 | 0.094 | 0.097 | 0.094 | 0.096 | 0.095 | 0.097 | 0.090 |
|  | 0.6 | 0.107 | 0.106 | 0.104 | 0.106 | 0.104 | 0.105 | 0.100 |

**EE position RMSE (m)**: mean over 100 seeds. Rows are B; columns are the EE update frequency in Hz.

| Controller | B | Fixed | 1/6 | 1/5 | 1/4 | 1/3 | 1/2 | 1 |
|---|---|---|---|---|---|---|---|---|
| Unified RL | 0 | 0.052 | 0.141 | 0.156 | 0.170 | 0.195 | 0.240 | 0.324 |
|  | 0.1 | 0.044 | 0.136 | 0.150 | 0.164 | 0.188 | 0.233 | 0.318 |
|  | 0.2 | 0.040 | 0.133 | 0.147 | 0.161 | 0.185 | 0.230 | 0.314 |
|  | 0.3 | 0.039 | 0.132 | 0.147 | 0.160 | 0.185 | 0.229 | 0.313 |
|  | 0.4 | 0.039 | 0.132 | 0.146 | 0.160 | 0.184 | 0.228 | 0.312 |
|  | 0.5 | 0.039 | 0.131 | 0.146 | 0.159 | 0.183 | 0.228 | 0.312 |
|  | 0.6 | 0.039 | 0.131 | 0.146 | 0.159 | 0.183 | 0.228 | 0.311 |
| RL+IK | 0 | 0.009 | 0.203 | 0.226 | 0.249 | 0.284 | 0.333 | 0.383 |
|  | 0.1 | 0.011 | 0.202 | 0.224 | 0.248 | 0.283 | 0.331 | 0.382 |
|  | 0.2 | 0.014 | 0.200 | 0.221 | 0.244 | 0.279 | 0.327 | 0.380 |
|  | 0.3 | 0.016 | 0.199 | 0.220 | 0.243 | 0.278 | 0.327 | 0.380 |
|  | 0.4 | 0.017 | 0.197 | 0.220 | 0.243 | 0.277 | 0.327 | 0.379 |
|  | 0.5 | 0.018 | 0.197 | 0.220 | 0.243 | 0.277 | 0.327 | 0.379 |
|  | 0.6 | 0.019 | 0.197 | 0.221 | 0.243 | 0.278 | 0.328 | 0.379 |
| Cooperative RL | 0 | 0.061 | 0.184 | 0.203 | 0.218 | 0.249 | 0.297 | 0.372 |
|  | 0.1 | 0.060 | 0.181 | 0.198 | 0.213 | 0.244 | 0.293 | 0.369 |
|  | 0.2 | 0.057 | 0.171 | 0.188 | 0.204 | 0.235 | 0.281 | 0.361 |
|  | 0.3 | 0.061 | 0.167 | 0.188 | 0.201 | 0.231 | 0.281 | 0.360 |
|  | 0.4 | 0.062 | 0.169 | 0.188 | 0.199 | 0.228 | 0.280 | 0.360 |
|  | 0.5 | 0.064 | 0.172 | 0.191 | 0.202 | 0.232 | 0.276 | 0.359 |
|  | 0.6 | 0.068 | 0.171 | 0.191 | 0.202 | 0.231 | 0.276 | 0.359 |

**EE orientation RMSE (rad)**: mean over 100 seeds. Rows are B; columns are the EE update frequency in Hz.

| Controller | B | Fixed | 1/6 | 1/5 | 1/4 | 1/3 | 1/2 | 1 |
|---|---|---|---|---|---|---|---|---|
| Unified RL | 0 | 0.24 | 0.48 | 0.51 | 0.53 | 0.58 | 0.68 | 0.87 |
|  | 0.1 | 0.22 | 0.47 | 0.50 | 0.52 | 0.57 | 0.67 | 0.86 |
|  | 0.2 | 0.21 | 0.46 | 0.49 | 0.52 | 0.56 | 0.66 | 0.85 |
|  | 0.3 | 0.20 | 0.45 | 0.48 | 0.51 | 0.56 | 0.66 | 0.85 |
|  | 0.4 | 0.20 | 0.45 | 0.48 | 0.51 | 0.56 | 0.66 | 0.85 |
|  | 0.5 | 0.20 | 0.45 | 0.48 | 0.51 | 0.56 | 0.66 | 0.85 |
|  | 0.6 | 0.20 | 0.45 | 0.48 | 0.51 | 0.56 | 0.66 | 0.85 |
| RL+IK | 0 | 0.07 | 0.58 | 0.63 | 0.69 | 0.76 | 0.89 | 1.01 |
|  | 0.1 | 0.07 | 0.58 | 0.64 | 0.69 | 0.77 | 0.88 | 1.01 |
|  | 0.2 | 0.08 | 0.58 | 0.63 | 0.68 | 0.76 | 0.88 | 1.01 |
|  | 0.3 | 0.08 | 0.58 | 0.63 | 0.68 | 0.75 | 0.88 | 1.01 |
|  | 0.4 | 0.08 | 0.57 | 0.63 | 0.68 | 0.75 | 0.88 | 1.01 |
|  | 0.5 | 0.09 | 0.57 | 0.63 | 0.68 | 0.76 | 0.88 | 1.01 |
|  | 0.6 | 0.09 | 0.56 | 0.63 | 0.68 | 0.76 | 0.88 | 1.01 |
| Cooperative RL | 0 | 0.25 | 0.48 | 0.51 | 0.54 | 0.60 | 0.71 | 0.88 |
|  | 0.1 | 0.25 | 0.48 | 0.51 | 0.54 | 0.60 | 0.70 | 0.88 |
|  | 0.2 | 0.25 | 0.47 | 0.50 | 0.53 | 0.59 | 0.70 | 0.88 |
|  | 0.3 | 0.24 | 0.46 | 0.49 | 0.52 | 0.58 | 0.69 | 0.88 |
|  | 0.4 | 0.24 | 0.45 | 0.49 | 0.52 | 0.57 | 0.69 | 0.88 |
|  | 0.5 | 0.24 | 0.45 | 0.49 | 0.52 | 0.58 | 0.69 | 0.87 |
|  | 0.6 | 0.25 | 0.45 | 0.49 | 0.52 | 0.57 | 0.68 | 0.87 |

**Early terminations** (of 100 seeds per cell). Rows are B; columns are the EE update frequency.

| Controller | B | Fixed | 1/6 | 1/5 | 1/4 | 1/3 | 1/2 | 1 |
|---|---|---|---|---|---|---|---|---|
| RL+IK | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|  | 0.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|  | 0.2 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
|  | 0.3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
|  | 0.4 | 0 | 1 | 0 | 0 | 2 | 0 | 1 |
|  | 0.5 | 0 | 2 | 1 | 0 | 1 | 1 | 1 |
|  | 0.6 | 1 | 4 | 0 | 0 | 1 | 1 | 2 |
| Cooperative RL | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
|  | 0.1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
|  | 0.2 | 2 | 2 | 1 | 1 | 1 | 0 | 0 |
|  | 0.3 | 12 | 12 | 13 | 11 | 10 | 10 | 2 |
|  | 0.4 | 16 | 21 | 22 | 25 | 15 | 13 | 2 |
|  | 0.5 | 19 | 25 | 24 | 28 | 23 | 20 | 5 |
|  | 0.6 | 26 | 31 | 30 | 33 | 32 | 22 | 6 |

## 6. Files

| Path (under `results/E3/`) | Content |
|---|---|
| `raw/E3/<C>/<B?T?>/seed_###/` | Per-episode logs: commands, states, actuation, metrics, metadata, schedule. 14,700 episodes. |
| `data/E3_per_trial_14700.csv` | One row per episode: four RMSEs, termination, schedule and initial-state hashes, pose rows used |
| `data/E3_arrays.npz` | Metric arrays [controller, B, T, seed] |
| `stats/E3_summary.csv` | Per controller × condition × metric: valid / completed / terminated / missing, mean, SD, 95 % CI, median |
| `stats/E3_summary_7x7.csv` | Machine-readable 7 × 7 table, all four metrics, one row per controller × condition |
| `stats/E3_paired_vs_C1.csv`, `E3_slice_trends.csv`, `E3_sensitivity_completed_only.csv`, `E3_completion.csv` | Statistics |
| `figures/paper/` | `wbc_coordination.{pdf,svg,png}`, `_values.csv`, `_checks.json` |
| `protocol/` | Config and baselines snapshots, git state, code hashes at launch, frozen schedules and their sha256, frozen-artefact hashes, pre-run validation, pose audit |
| `validation/validation_summary.json` | Post-run validation and the full list of terminations |
| `MANIFEST_SHA256.txt` | sha256 of every non-raw file; `RAW_SHA256.txt` covers `raw/` |
