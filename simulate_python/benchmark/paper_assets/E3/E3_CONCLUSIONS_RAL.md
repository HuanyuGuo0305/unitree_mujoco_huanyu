# E3 — Conclusions for the RA-L paper

**Conventions.** "Fig. E3(a)–(d)" refers to the four groups of the paper figure (`figures/paper/wbc_coordination_1x4_stacked.pdf`, label `fig:wbc_coordination`). It is one row of groups, each with a top and a bottom subplot:
- (a) EE position RMSE (top) and EE orientation RMSE (bottom) vs B;
- (b) base XY RMSE (top) and base yaw RMSE (bottom) vs EE update frequency;
- (c) at EE switches: base XY error increase (top) and absolute EE position error (bottom);
- (d) at base switches: absolute base XY error (top) and EE position error increase (bottom).

Quantities marked "not plotted" are reported only in the text and statistics. These are the yaw and EE orientation error increases at switches. Numbers are mean over 100 paired seeds; verdicts are paired tests with Holm correction (details in `E3_STATISTICAL_ANALYSIS.md`). The comparison concerns the existing trained checkpoints evaluated over paired simulation trials.

## 1. Concurrent tracking

**Base tracking.** C1 tracks base velocity best in every one of the 25 base × EE intensity cells, against both baselines: base XY by 10–68 % vs C3 and 26–150 % vs C5, and yaw rate likewise. With the EE target switching every 2 s and B = 0.60, base XY RMSE is:
- 0.143 m/s for C1;
- 0.157 m/s for C3;
- 0.179 m/s for C5.

**EE tracking with switching targets.** When the EE target changes (T = 6–2 s), C1 also has the lowest EE position RMSE in all 20 cells:
- by 39–53 % vs C3 and 21–32 % vs C5;
- at T = 2 s, 0.23–0.24 m for C1, against 0.33 m for C3 and 0.28–0.30 m for C5.

C1 also has a lower EE orientation RMSE than C3 (by 21–36 %). It is statistically indistinguishable from C5 in 18 of 25 cells (differences ≤ 5.5 %).

**EE tracking under base motion** (T = 2 s, Fig. E3(a)). As B rises from 0 to 0.60, EE RMSE changes little for every controller and in fact decreases slightly: position by −0.012 m (C1), −0.006 m (C3) and −0.021 m (C5); orientation by −0.027, −0.008 and −0.021 rad. The ranking is set by the absolute levels (C1 0.23–0.24 m, C3 0.33 m, C5 0.28–0.30 m), not by the trend. The slight decrease is not interpreted as disturbance rejection: it is consistent with the stationary-base pose offset seen with fixed targets, and C1's larger decrease is not counted as an advantage.

**Static EE precision.** With a fixed EE target, the IK-based C3 holds the pose far more precisely:
- at B = 0, 0.009 m and 0.07 rad, against 0.052 m and 0.24 rad for C1;
- C3 is better in all 5 Fixed cells, by 51–83 %.

**Target reaching.**
- Under the primary tolerance (0.08 m / 0.35 rad), C1 reaches more targets than C5 in every switching cell and responds fastest: at B = 0, T = 2 s, 74 % within a 1.18 s restricted mean, against 55 % / 1.48 s for C3 and 58 % / 1.34 s for C5.
- Under the strict tolerance (0.04 m / 0.20 rad), **C3 reaches more targets than C1 in 16 of 20 cells**, for example 46 % vs 30 % at B = 0 and T = 2 s.

**Robustness.** C1 completed all 2,500 episodes. C3 fell in 16 (0.6 %). C5 fell in 308 (12.3 %); its completion is 67–78 % at B = 0.60.

## 2. EE→base coordination

**Frequency sensitivity** (stationary base, Fig. E3(b)). More frequent EE commands increase every controller's base error. From a fixed EE to 0.5 Hz:

| Controller | Base XY (m/s) | Yaw rate (rad/s) |
|---|---|---|
| C1 | +0.013 | +0.018 |
| C3 | +0.016 | +0.031 |
| C5 | +0.028 | +0.058 |

- C1 has the lowest absolute base error at every frequency.
- C1 is less sensitive than C5 in base translation.
- **Its base-XY sensitivity is not significantly lower than C3's** in any cell. Its yaw sensitivity is lower than C3's in 7 of 20 cells.
- With a moving base (B > 0), the EE update rate has only a small effect on base error for all controllers. The frequency effect is therefore a stationary-base phenomenon.

**Single EE-target switches** (stationary base, Fig. E3(c); 1,200 matched events).
- **Integrated base disturbance:**
  - Each switch induces about half the integrated base-velocity disturbance in C1 that it does in C5 (0.017 vs 0.036 m), and a statistically indistinguishable amount compared with C3 (0.018 m).
  - The integrated yaw-rate disturbance is lowest for C1: 0.018 rad, against 0.028 for C3 and 0.050 for C5.
- **Peak and recovery:**
  - C1's base-velocity disturbance is brief but its **peak is higher than C3's** (0.039 vs 0.032 m/s).
  - C1 recovers fastest: 0.73 s, against 0.96 and 0.94 s.
- **EE response:** in the same events C1 converges to the new target fastest. Its residual position error 1.0–1.5 s after the switch is 0.076 m, against 0.195 m for C3 and 0.155 m for C5.
- **Figure (post-hoc presentation).** Fig. E3(c), top, shows the event-level mean positive base XY error increase. RL+IK's curve peaks lowest: 0.017 vs 0.027 m/s for C1 and 0.034 for C5. From 0.48 s onwards, C1's curve is the lowest; over 1.0–1.5 s it is 0.0076 vs 0.0111 (RL+IK) and 0.0192 m/s (C5), descriptive. The earlier signed-mean figure showed RL+IK lowest throughout. That reading was an artefact of cancellation: RL+IK's late signed mean is about 0.0014 m/s, but its positive excess is about 0.011 m/s. The integrated increase is not significantly different between C1 and RL+IK. Late values include rectified fluctuation, which is higher for RL+IK and C5 before the switch as well.
- **Interpretation:** C1 combines the fastest manipulation response with a base error increase comparable to C3's in total and smaller than C5's. RL+IK has a smaller, later peak, while C1 has a larger, briefer one. C1's advantage over C3 lies in responsiveness, yaw stability and a shorter-lived increase, not in a smaller peak translational disturbance.

## 3. Base→EE coordination

**Excess error at base-command changes** (EE target fixed, B = 0.60, Fig. E3(d); 652 matched events in 90 seeds). On the formal per-event metrics, C1's EE pose is perturbed least (paired differences with 95 % CIs in `stats/E3_events_matched_paired.csv` and `stats/E3_event_positive_excess.csv`; e.g., peak position excess C3 − C1 = 0.0083 m [0.0070, 0.0097], C5 − C1 = 0.0053 m [0.0040, 0.0066]):

| Metric | C1 | C3 | C5 |
|---|---|---|---|
| Peak excess EE position error (m) | 0.012 | 0.020 | 0.017 |
| Peak excess EE orientation error (rad) | 0.033 | 0.054 | 0.040 |

C1's integrated position excess is also the lowest.

**Reading Fig. E3(d), bottom.** The subplot plots the *mean positive excess* EE position error: for each switch, max(e_i(t) − pre-switch mean, 0), averaged over switches, then over seeds. This presentation is **post-hoc**. It replaced the signed-mean trace, in which Cooperative RL's (C5) increases and decreases cancelled (signed integral −0.0006 m·s), making its curve look near zero. With the positive part taken per event, the curve maxima are 0.0057 m for C1, 0.0106 m for C3 and 0.0079 m for C5 (paired seed-bootstrap CIs of the differences exclude 0). C1's curve is the lowest from 0.16 s to 1.5 s. Over 0.10–0.14 s it is within 0.0002 m of C5's. Over 0–0.02 s it is slightly above C3's, while all curves are still at their pre-switch floor. C5 stays elevated longest: 0.0059 m over 1.0–1.5 s, against 0.0048 m for C3 and 0.0033 m for C1. The curve maximum is not the same quantity as the mean per-event peak in the table: max_t E[δe⁺] ≠ E[max_t δe⁺]. The curve is nonzero before the switch (about 0.001–0.002 m) because fluctuations are rectified. Orientation excess is not plotted in the paper figure; it is in the table and the supplementary figure.

**Absolute EE error around the same switches** (matched events, paired, descriptive for the post-switch mean):

| | C1 | C3 | C5 |
|---|---|---|---|
| Pre-switch, [−0.5, 0) s (m) | 0.034 | **0.013** | 0.051 |
| Post-switch, [0, 1.5) s (m) | 0.035 | **0.018** | 0.050 |

C3 is lower than C1 before and after the switch, by 0.022 m [0.017, 0.026] and 0.017 m [0.012, 0.021]. A smaller induced increment for C1 therefore does not mean better absolute EE accuracy.

**Robustness.** The position results hold at B = 0.15–0.60, across target heights and across baseline windows. The orientation results hold at B = 0.45–0.60. At B = 0.30, orientation integrated excess against C3 is not significant, and at B = 0.15 it favours C3. The positive-excess curve ordering, C1 lowest by the curve maximum, holds at B = 0.15, 0.30 and 0.45 (`stats/E3_event_positive_excess.csv`).

**Absolute EE accuracy** is the opposite ranking against C3. Before those base changes, C3 holds the EE at 0.013 m and 0.076 rad, against 0.034 m and 0.172 rad for C1. Over the whole episode with a fixed target, C3's EE RMSE remains below C1's at every base intensity.

C1's smaller base-induced increment therefore indicates **lower sensitivity of the EE to locomotion commands, not more accurate EE regulation**. C1's EE RMSE decreases with base intensity; that is a stationary-base offset effect and is not interpreted as improved coordination.

## 4. Overall

**Supported.** C1 preserves locomotion tracking under concurrent manipulation better than both baselines, and it is the only controller that never fell. When EE targets change dynamically, it also achieves the lowest EE tracking error, the fastest target convergence and the smallest EE disturbance from base-command changes. These are the properties expected from unified whole-body control.

**Not supported.**
- C1 does not match the IK-based C3 in static EE precision or strict target reaching.
- C1's translational base disturbance per EE switch is not smaller than C3's.
- C1's sensitivity of base XY tracking to EE frequency is not lower than C3's.

## 5. Draft RA-L Experiments subsection

> **E3: Dynamic whole-body coordination.** We evaluate coordination under independently changing locomotion and manipulation commands. EE targets are full 6-DoF poses drawn uniformly from the controllers' common training pose table and commanded by sample-and-hold every T ∈ {∞, 6, 4, 3, 2} s. They are crossed with piecewise-constant base-velocity commands within per-axis bounds B ∈ {0, 0.15, 0.30, 0.45, 0.60} (100 paired trials per cell; 7,500 episodes; 30 s, evaluated over 3–30 s).
>
> *Concurrent tracking.* C1 achieves the lowest base-velocity error in all 25 cells and, whenever the EE target changes, the lowest EE position error (e.g., 0.23 m vs 0.33 m for C3 and 0.28 m for C5 at B = 0.60, T = 2 s). As the base bound grows, EE errors of all controllers stay nearly constant (Fig. E3(a)); the slight decrease is attributed to a stationary-base pose offset rather than to disturbance rejection. It also has the highest primary-tolerance reaching rate and the shortest response time. With a fixed EE target, however, the IK-based C3 is markedly more precise (0.009 m vs 0.052 m at B = 0), and it reaches strict tolerances more often. C1 completed all episodes, whereas C5 failed in 12 % (up to 33 % at B = 0.60).
>
> *EE→base.* With a stationary base, base errors grow with EE update frequency for all controllers (Fig. E3(b)). The increase in yaw-rate error is 0.018 rad/s for C1 versus 0.031 and 0.058 rad/s for C3 and C5, while the increase in base-velocity error is statistically similar for C1 and C3. Around individual EE switches, C1 converges to the new target fastest (Fig. E3(c), bottom). The resulting increase (Fig. E3(c), top) in base-velocity error has a higher, briefer peak than C3's (per switch 0.039 vs 0.032 m/s) and decays below C3's after about 0.5 s. Its integral is statistically equal to C3's and half of C5's, and C1 has the smallest yaw-rate increase.
>
> *Base→EE.* At base-command changes with a fixed EE target, the base-velocity response is similar for all controllers (Fig. E3(d), top; mean excess falls below 10 % of its peak by 0.28–0.32 s), with C1 starting from the lowest pre-switch error; the additional EE position error is smallest for C1, both per switch (mean peak 0.012 m, vs 0.020 m for RL+IK and 0.017 m for Cooperative RL) and in the event-averaged positive-excess trace (Fig. E3(d), bottom; post-hoc presentation). The same ordering holds for orientation (0.033 vs 0.054 / 0.040 rad). C3's absolute EE error nevertheless remains lower (0.013 vs 0.034 m before the switch), so this result reflects lower locomotion-to-EE sensitivity rather than higher EE accuracy.

## 6. Claims not supported by these results

- C1 tracks EE poses more precisely than C3 in general. It does not with fixed targets, nor under strict tolerances.
- C1 produces a smaller translational base disturbance per EE switch than C3.
- C1's base XY tracking is less sensitive to EE frequency than C3's.
- Negative Base→EE degradation of C1 shows better coordination.
- The frequency effect extends to a moving base.
- Equivalence in any non-significant comparison, or generalisation beyond these checkpoints and simulated conditions.

## 7. Limitations

- **Pose-0 convergence.** Pose 0 is not fully reached during the 3 s settling window: at t = 3 s the error is 0.13 m / 0.55 rad for C1 and 0.03 m / 0.10 rad for C3. This raises C1's Fixed-level RMSE.
- **Reachability.** About 11 % of the targets require whole-body posture changes to be reachable, and 0.8 % were not reachable in the audit. They are retained as part of the training distribution.
- **C3's joint band.** C3's IK band excludes about 16 % of the targets that are otherwise reachable at its stance. This is a design limit of the baseline and was not adjusted.
- **C5's falls** reduce its matched base→EE events to 90 seeds and 652 events, and truncate its trials at high B.
- **Overlapping switches.** Isolated events are rare when both subsystems switch (2.3 isolated vs 9.7 overlapping EE events per episode at B > 0, T = 2 s). The event analysis therefore uses the clean conditions; overlapping events are reported separately.
- **History.** The benchmark was redesigned after earlier E3 versions had been examined. The protocol was frozen before this run, and the targets come from the controllers' own training distribution rather than from experimenter selection.
