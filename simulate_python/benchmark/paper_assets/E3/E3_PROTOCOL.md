# E3 — Random discrete full-pose dynamic whole-body coordination: protocol

**Status: frozen before the official run.** This document and `configs/E3.yaml` are committed to git before any official episode is simulated. Any later change is reported as a deviation in `E3_REPRODUCIBILITY_REPORT.md`.

## 0. History (disclosed)

This is the fourth E3 design. All three earlier designs, and their results, were removed:

| Version | Design | Removed |
|---|---|---|
| v1 | smooth sinusoidal trajectories | 2026-11-11 |
| v2 | arc-length line/circle/semicircle at 0–0.20 m/s | 2026-11-11 |
| v3 | five fixed line/circle full-pose waypoints, 15,000 episodes | 2026-11-11; code preserved at git tag `e3v3-pre-random-pose-replacement`, file hashes in `cleanup/` |

The present design was chosen **after the earlier E3 results had been examined**. Those results showed, among other things:
- C1's base-tracking advantage;
- C3's better static EE precision;
- C1's stationary-base pose bias.

The design replaces hand-selected trajectories with the controllers' own training target distribution, removing any trajectory selection by the experimenters. No rule below was tuned on E3 v3 data. The thresholds that carry over (target-reaching tolerances, event windows) were fixed earlier: from E1 data, and in the v3 pre-registration respectively.

## 1. Question

Does unified whole-body control give coordinated responses to independently changing locomotion and manipulation commands? Three aspects are tested:
- concurrent tracking;
- EE→base coordination;
- base→EE coordination.

The controllers are C1 (Unified RL, ours), C3 (RL locomotion + arm IK) and C5 (cooperative dual-PPO). Existing checkpoints and deploy configurations are used unchanged.

## 2. Design

5 base levels × 5 EE levels × 3 controllers × 100 paired seeds (0–99) = **7,500 episodes**. Conditions are `B{i}T{j}`.

| Factor | Levels |
|---|---|
| Base bound B | 0, 0.15, 0.30, 0.45, 0.60. Per-axis \|vx\|, \|vy\| ≤ B m/s and \|wz\| ≤ B rad/s in the body frame, using the existing piecewise-constant generator: segments of U(2, 4) s, each axis U(−1, 1) × B (stream `base_velocity`). B = 0 commands exactly zero. Switch times and normalised samples depend only on the seed. |
| EE update interval T | Fixed (0 Hz), 6 s, 4 s, 3 s, 2 s. |

**Episodes.**
- **Timing:** 30 s, physics at 250 Hz, policy at 50 Hz.
- **Settling (first 3 s):** zero base command; the EE command is **pose 0** of the seed's sequence. There is therefore no command jump at t = 3 s.
- **Evaluation window:** 3–30 s (1,350 ticks).
- **Termination:** base height < 0.30 m or tilt > 50°. This and every controller setting are unchanged; C3's IK limits are untouched.

## 3. EE targets (controller-independent)

**Source.** `utilities/data/keypoints_plb_evaluation_filtered.npy`, 20,000 rows of 9-D PLB keypoints (kp0, kp0 + R·[0.3, 0, 0], kp0 + R·[0, 0, 0.3]).
- Its sha256 is `d822f110…c328`, identical to `IsaacLab_huanyu/scripts/tools/keypoints_plb_evaluation_filtered.npy`.
- That file is the target table of every controller's training config (`PresampledKeypointsDirectCommandPLBCfg`, `sample_mode = random`).
- Training draws targets uniformly (`torch.randint`, no weights), with direct switching and no interpolation.

**Validity screen.** This is the only filter, and it is pre-declared:
- finite values;
- |kp1 − kp0| = |kp2 − kp0| = 0.30 m (±1e-3);
- orthogonal offsets (|cos| < 1e-3).

All 20,000 rows pass, so the sampling set is the whole table (`configs/E3_feasible_rows.npy`, sha256 `d7887b88…4a80`).

**No reachability filter.** Arm-only reachability at a nominal stance is audited but deliberately not used as a filter. Doing so would remove exactly the targets that need whole-body motion, which all three policies were trained on.

**Sequence.**
- Per seed, 16 table rows are drawn uniformly with replacement (stream `e3_random_pose`, independent of the base stream).
- Level T commands pose ⌊k / (T / Δt)⌋ at tick k, i.e. a prefix of the same sequence.
- Fixed holds pose 0.
- Commands are sample-and-hold with no interpolation; the first switch, to pose 1, is at t = 3 + T.

**Kinematic audit** (`protocol/pose_audit/`, controller-blind). Of the 1,535 distinct targets used by seeds 0–99:

| Check | Share of targets |
|---|---|
| Reachable arm-only at ≥ 1 nominal stance (C1/C3/C5 E1 T1 seed 0) | 88.7 % |
| Reachable at all three nominal stances | 71.2 % |
| Reachable at neither a nominal stance nor any of 24 logged E1 T2 reaching postures | 0.8 % |
| Inside C3's IK joint band, of those reachable at C3's stance | 83.7 % |
| Reachable at ≥ 1 of 24 logged E1 T4 moving-base postures (mean 67 % of postures) | 95.6 % |

**Jump magnitudes.** Position and orientation jumps between consecutive targets have the same distribution at every interval level (V7: mean 0.54 m and 1.33 rad; KS p ≥ 0.93). See `protocol/prerun_validation.json`.

## 4. Pairing

- The EE sequence depends only on the seed.
- The base sequence depends only on (B, seed).
- The initial state depends only on the seed.
- The scenario builder has no controller argument, so all three controllers receive bit-identical commands and initial states in every cell.

## 5. Absolute tracking metrics (identical to E1)

Over all evaluated ticks from 3 to 30 s, switching transients included:
- base XY velocity RMSE [m/s] and yaw-rate RMSE [rad/s], in the body frame;
- EE position RMSE [m] in the PLB frame;
- EE orientation RMSE [rad], the RMSE of the SO(3) geodesic angle.

Also reported:
- completion and effective evaluation time;
- the number of target updates;
- target jump magnitudes;
- the EE error at t = 3 s (pose-0 convergence during settling).

**Target reaching**, per EE switch inside the window:
- **Primary:** ≤ 0.08 m and ≤ 0.35 rad, held for 0.2 s. **Strict:** ≤ 0.04 m and ≤ 0.20 rad. Both tolerances were fixed from E1 data before E3 v3.
- Time-to-target is measured from the switch.
- An unreached target is a failure and is right-censored at the hold length (or at termination).
- Reported: success rate and restricted mean time.

## 6. Event analysis

**Windows and events.**
- The event time is the command change.
- The baseline is the mean error over [−0.5, 0) s. The response window is [0, 1.5) s.
- The event's entire window must lie inside the valid evaluation window. Events that do not are counted as truncated.
- An event is **isolated** if no command change of the other subsystem lies within [−1.0, +1.5) s; otherwise it is **overlapping**. Overlapping events are analysed and reported separately, never discarded silently.
- At B > 0, the base change at t = 3 s (zero settle command → first segment) counts as a base change.

**Event types.**
- **EE events** are EE switches. Signals: base XY and yaw-rate errors (EE→base), and EE position/orientation errors (response).
- **Base events** are base-command changes. Signals: EE position and orientation errors (base→EE), and base errors.

**Per-event quantities**, with δ(t) = e(t_s + t) − mean(e over the baseline window):
- absolute baseline error, absolute mean over the response window, and absolute residual over [1.0, 1.5) s;
- signed peak excess max δ, and **peak positive excess** max(max δ, 0);
- signed integrated excess ∫δ dt, and **integrated positive excess** ∫max(δ, 0) dt;
- **recovery time**: the first time after the peak at which δ ≤ 0.25 · peak for 0.2 s, censored at 1.5 s. It is defined only when the peak ≥ floor (base XY 0.005 m/s, yaw 0.005 rad/s, EE position 0.002 m, EE orientation 0.01 rad); the fraction of defined events is reported. *(The v3 absolute recovery tolerance turned out to be larger than typical disturbances and is not reused.)*
- **Baseline-window sensitivity**: the positive-excess metrics are recomputed with baselines over [−0.24, 0) and [−1.0, 0) s.
- **Jump magnitude** of each EE event: position distance and SO(3) angle between the old and new targets.

**Aggregation.** Events are averaged within each episode. For the main figures, the seed value is the episode value. The seed is the resampling unit.

**Main event conditions (fixed in advance):**
- EE→base: B = 0 and T = 2 s, where every EE event is isolated (12 per episode).
- Base→EE: T = Fixed and B = 0.60, where every base event is isolated. B = 0.15–0.45 are reported as sensitivity.

**Jump-dependence.** EE events are binned into tertiles of position jump (edges 0.432 / 0.640 m) and of orientation jump (edges 0.997 / 1.538 rad). The edges were computed from the frozen sequences (`protocol/analysis_bins.json`). Metrics are compared per bin, and the per-seed slope of peak excess against jump size is reported descriptively, not as a causal claim.

**Target consistency (base→EE).** Seeds are split into tertiles by the height of pose 0 (edges 0.637 / 0.877 m), and the comparisons are repeated per tertile.

## 7. Statistics

**Per controller and condition:**
- mean ± SD (ddof = 1) over the 100 seeds;
- percentile bootstrap 95 % CI (10,000 resamples, seed 20261106), using the same resampled seeds for all controllers (paired);
- completion with a Wilson interval.

**Paired comparisons.**
- C3 − C1 and C5 − C1 are computed on seed-level differences, with a bootstrap CI and a paired t-test.
- Holm correction is applied within each family:
  - RMSEs: metric × comparison, over 25 cells;
  - events and reaching: one family per figure/analysis.
- BH q-values are also reported.
- Effect sizes: relative difference and d_z. "Practical" means |relative difference| ≥ 10 %.
- No equivalence claims are made from non-significance.

**Separated claims.** These are reported separately and never merged into one "coordination" score:
- lower absolute error;
- lower command-induced excess error;
- faster reaching;
- higher strict reaching success;
- robustness to intensity.

**Sensitivity analyses:**
- completed-episode-only analysis;
- baseline windows;
- B and T levels other than the main ones;
- jump tertiles;
- pose-0 height tertiles.

Terminated episodes are kept, scored over their valid window.

**Statistical unit.** The comparison is between the existing trained checkpoints over paired simulation trials. The 7,500 episodes are not independent training runs.

## 8. Paper figures

All are single-column, 252 pt wide, with two stacked panels and paired 95 % CIs.
- **Fig. 1:** T = 2 s; base XY RMSE (a) and EE position RMSE (b) against B.
- **Fig. 2:** B = 0; base XY RMSE (a) and yaw-rate RMSE (b) against EE update frequency.
- **Fig. 3:** B = 0, T = 2 s, isolated EE switches; signed excess base XY error (a) and absolute EE position error (b) against time.
- **Fig. 4:** EE fixed, B = 0.60, isolated base switches; signed excess EE position (a) and orientation (b) error against time.
