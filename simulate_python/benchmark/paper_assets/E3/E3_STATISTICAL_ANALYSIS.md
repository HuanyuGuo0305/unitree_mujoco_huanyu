# E3 — Statistical analysis

**Conventions.**
- Values are mean ± SD over 100 paired seeds, with the 95 % percentile-bootstrap CI (10,000 resamples, seed 20261106) in brackets.
- The same resampled seeds are used for all controllers.
- "Rival − C1" > 0 means C1's value is lower.
- Verdicts are paired t-tests with Holm correction within the declared family:
  - RMSE and reaching: metric × comparison over 25 (or 20) cells;
  - events: one family per main condition.
- "Practical" means |relative difference| ≥ 10 %. Non-significance is not equivalence.
- The 7,500 episodes are paired simulation trials of fixed checkpoints, not independent training runs.

## 1. Concurrent tracking (official 3–30 s RMSE)

**Representative cells.** At T = 2 s the EE target switches every 2 s; "Fixed" holds one target.

| Cell | Controller | Base XY (m/s) | Yaw rate (rad/s) | EE pos. (m) | EE ori. (rad) |
|---|---|---|---|---|---|
| B = 0, T = 2 s | C1 | 0.028 ± 0.004 | 0.024 ± 0.003 | 0.240 ± 0.034 | 0.683 ± 0.111 |
| | C3 | 0.041 ± 0.007 | 0.044 ± 0.012 | 0.333 ± 0.047 | 0.885 ± 0.175 |
| | C5 | 0.065 ± 0.007 | 0.085 ± 0.011 | 0.297 ± 0.049 | 0.705 ± 0.113 |
| B = 0.60, T = 2 s | C1 | 0.143 ± 0.023 | 0.074 ± 0.011 | 0.228 ± 0.032 | 0.655 ± 0.108 |
| | C3 | 0.157 ± 0.022 | 0.092 ± 0.015 | 0.328 ± 0.048 | 0.877 ± 0.174 |
| | C5 | 0.179 ± 0.029 | 0.105 ± 0.028 | 0.276 ± 0.058 | 0.685 ± 0.127 |
| B = 0, Fixed | C1 | 0.015 ± 0.015 | 0.006 ± 0.002 | 0.052 ± 0.031 | 0.239 ± 0.187 |
| | C3 | 0.025 ± 0.016 | 0.013 ± 0.007 | **0.009 ± 0.015** | **0.069 ± 0.273** |
| | C5 | 0.037 ± 0.023 | 0.027 ± 0.018 | 0.061 ± 0.051 | 0.255 ± 0.172 |
| B = 0.60, Fixed | C1 | 0.139 ± 0.024 | 0.070 ± 0.014 | 0.039 ± 0.022 | 0.196 ± 0.174 |
| | C3 | 0.155 ± 0.028 | 0.087 ± 0.022 | **0.019 ± 0.017** | **0.088 ± 0.240** |
| | C5 | 0.181 ± 0.039 | 0.107 ± 0.038 | 0.068 ± 0.061 | 0.248 ± 0.157 |

**Verdicts over all 25 cells** (`stats/E3_paired_tracking_and_reaching.csv`):

| Metric | C3 − C1 | C5 − C1 |
|---|---|---|
| Base XY RMSE | C1 lower 25/25 (+10 to +68 %) | C1 lower 25/25 (+26 to +150 %) |
| Yaw-rate RMSE | C1 lower 25/25 (+24 to +111 %) | C1 lower 25/25 (+42 to +327 %) |
| EE position RMSE | C1 lower 20/25 (all switching cells, +39 to +53 %); **C3 lower 5/25 (all Fixed cells, −51 to −83 %)** | C1 lower 25/25 (+18 to +73 %) |
| EE orientation RMSE | C1 lower 20/25 (switching cells, +21 to +36 %); **C3 lower 5/25 (Fixed, −55 to −71 %)** | C1 lower 7/25; **no significant difference 18/25** (differences 0.4–5.5 % in the switching cells) |

**Completion.**

| Controller | Completed | Rate |
|---|---|---|
| C1 | 2,500 / 2,500 | 100 % |
| C3 | 2,484 / 2,500 | 99.4 % |
| C5 | 2,192 / 2,500 | 87.7 % (67–78 % at B = 0.60) |

**Completed-only sensitivity.** 4 of 200 RMSE verdicts change, all C5 vs C1 EE orientation:
- at B2T3, B3T3 and B4T3, "no significant difference" becomes "C1 lower";
- at B4T0, "C1 lower" becomes "no significant difference".

**Target reaching** (switches inside the window; censored at the hold):

| Cell | Measure | C1 | C3 | C5 |
|---|---|---|---|---|
| B = 0, T = 2 s | primary success | **0.74** | 0.55 | 0.58 |
| | strict success | 0.30 | **0.46** | 0.22 |
| | RMST primary (s) | **1.18** | 1.48 | 1.34 |
| | RMST strict (s) | 1.69 | **1.59** | 1.73 |
| B = 0.60, T = 2 s | primary success | **0.81** | 0.58 | 0.65 |
| | strict success | 0.50 | 0.49 | 0.34 |
| | RMST primary (s) | **1.06** | 1.45 | 1.24 |
| | RMST strict (s) | **1.52** | 1.57 | 1.62 |

Over the 20 switching cells:
- **Primary success:** C1 higher than C5 in 20/20; against C3, C1 higher in 9/20 with no significant difference in 11/20.
- **Primary time:** C1 faster than both in 20/20.
- **Strict success:** **C3 higher than C1 in 16/20** (no significant difference in 4); C1 higher than C5 in 20/20.
- **Strict time:** C3 faster in 13/20, C1 faster in 3/20.

**Pose 0 at t = 3 s.** The error on the first target is still 0.13 m / 0.55 rad for C1 and 0.16 m / 0.49 rad for C5, against 0.03 m / 0.10 rad for C3. The settling window includes the policies' native start-up blend, so the Fixed-level RMSEs include a residual initial convergence. This is the same for all base levels.

## 2. EE→base coordination

**Frequency sensitivity, B = 0** (Fig. E3(b)). Change from Fixed to T = 2 s:

| Controller | Base XY (m/s) | Yaw rate (rad/s) |
|---|---|---|
| C1 | +0.013 [0.010, 0.016] | +0.018 [0.017, 0.018] |
| C3 | +0.016 [0.013, 0.019] | +0.031 [0.028, 0.033] |
| C5 | +0.028 [0.023, 0.032] | +0.058 [0.054, 0.062] |

Differences in degradation, over 20 cells:
- **Base XY:** C3 vs C1 is **not significant in 20/20**. C5 vs C1 favours C1 in 4/20, namely the B = 0 cells.
- **Yaw:** C1 less sensitive than C3 in 7/20 and than C5 in 8/20; not significant otherwise.

At B > 0 the EE-frequency effect on base error is small for every controller (|Δ| ≤ 0.010 m/s and ≤ 0.022 rad/s over all cells; mostly ≤ 0.004 m/s) and is not generalised from B = 0.

**Isolated EE-switch events, B = 0, T = 2 s** (Fig. E3(c)). 1,200 matched events, 100 seeds. "Positive excess" is relative to each event's own [−0.5, 0) s baseline.

| Metric | C1 | C3 | C5 | Verdicts (Holm) |
|---|---|---|---|---|
| Base XY peak positive excess (m/s) | 0.039 | **0.032** | 0.071 | **C3 lower than C1**; C1 lower than C5 |
| Base XY integrated positive excess (m·s) | 0.0174 | 0.0179 | 0.0355 | C1 vs C3 not significant; C1 lower than C5 |
| Yaw peak positive excess (rad/s) | 0.077 | 0.083 | 0.101 | C1 vs C3 not significant; C1 lower than C5 |
| Yaw integrated positive excess (rad) | **0.018** | 0.028 | 0.050 | C1 lower than both |
| Base XY recovery (s, 25 % of peak, censored at 1.5) | **0.73** | 0.96 | 0.94 | C1 faster than both (censored fraction 0.20 / 0.39 / 0.32) |
| Absolute pre-switch base XY error (m/s) | 0.017 | 0.032 | 0.044 | descriptive |
| EE position, mean over 0–1.5 s (m) | **0.20** | 0.32 | 0.26 | C1 lower than both |
| EE position residual, 1.0–1.5 s (m) | **0.076** | 0.195 | 0.155 | C1 lower than both |
| EE orientation residual (rad) | **0.33** | 0.51 | 0.35 | C1 lower than both |

**Robustness of these verdicts:**
- **Other EE levels:** at T = 6, 4 and 3 s the peak result is "C3 lower" or not significant, integrated base XY is not significant against C3 in every case, integrated yaw excess is C1 lower against both, and peak yaw against C3 is not significant.
- **Baseline windows** ([−0.24, 0) and [−1.0, 0) s): all directions unchanged.
- **Jump tertiles:** C3's lower base XY peak holds in the middle and large position-jump tertiles; in the smallest tertile the CI spans 0. C1's lower EE residual holds in every tertile.
- **Jump slopes:** within-seed slopes of peak excess against position jump are positive for all controllers. This is descriptive, not causal.

## 3. Base→EE coordination

**Degradation versus B = 0.**
- **Fixed EE:** C1's EE error *decreases* with base motion (0.052 → 0.039 m; 0.239 → 0.196 rad), whereas C3's increases (0.009 → 0.019 m; 0.069 → 0.088 rad).
- **Switching levels:** all controllers' EE RMSE decreases slightly with B.
- C1's negative degradation is **not** evidence of better coordination. Its stationary-base error is higher (pose-0 convergence and static bias), and C3's absolute held-pose error stays lower at every B.
- The C1-vs-rival degradation differences ("C1 better" in 20/20 position cells vs C3) are dominated by this baseline effect and are not used as a claim.

**Isolated base-command events, EE fixed, B = 0.60** (Fig. E3(d)). 652 matched events, 90 seeds.

| Metric | C1 | C3 | C5 | Verdicts (Holm) |
|---|---|---|---|---|
| EE position peak positive excess (m) | **0.012** | 0.020 | 0.017 | C1 lower than both |
| EE position integrated positive excess (m·s) | **0.0058** | 0.0106 | 0.0090 | C1 lower than both |
| EE orientation peak positive excess (rad) | **0.033** | 0.054 | 0.040 | C1 lower than both |
| EE orientation integrated positive excess (rad·s) | **0.020** | 0.028 | 0.023 | C1 lower than C3; vs C5 not significant |
| EE position recovery (s) | 0.85 | 1.02 | 0.92 | C1 faster than C3; vs C5 not significant |
| **Absolute pre-switch EE position error (m)** | 0.034 | **0.013** | 0.051 | **C3 lower than C1**; C1 lower than C5 |
| **Absolute pre-switch EE orientation error (rad)** | 0.172 | **0.076** | 0.210 | **C3 lower than C1**; C1 vs C5 not significant |

**Robustness of these verdicts:**
- **Other base levels:** at B = 0.45, C1 has the lower excess against both rivals in every metric. At B = 0.30 the same holds except orientation integrated excess against C3 (not significant).
- **B = 0.15:** mixed. Position excess is still lower for C1, but orientation integrated excess favours C3, and orientation peaks are not significant.
- **Baseline windows:** directions unchanged. The C5-vs-C1 orientation integrated excess, which is not significant with the registered window, is borderline with [−0.24, 0) s (CI lower bound 0.0000) and spans 0 with [−1.0, 0) s.
- **Pose-0 height tertiles** (unmatched events): C1's lower position excess holds in every tertile against both rivals, and its lower orientation excess against C3 in every tertile. Orientation integrated excess against C5 spans 0 in all tertiles, and the peak spans 0 in the lowest tertile.

**Post-hoc presentation: event-level positive excess for both cross-task panels**

Sources: `e3/positive_excess.py`, `stats/E3_event_positive_excess.csv` and `stats/E3_event_positive_excess_curves_main.csv`.

**Definition.** The same definition is used for Fig. E3(c) (base XY after EE switches) and Fig. E3(d) (EE position after base switches):
- For each matched switch, δe⁺ᵢ(t) = max(eᵢ(t) − ēᵢ,pre, 0), where ēᵢ,pre is that switch's own mean over [−0.5, 0) s.
- The positive part is taken per event, before averaging. Events are averaged within each seed, and seeds are bootstrapped with pairing.
- δe⁺ measures **increases** in tracking error above the pre-switch level. It is not a total disturbance magnitude.
- Because fluctuations are rectified, δe⁺ is nonzero even with no systematic change. A controller whose error fluctuates more after the switch therefore scores higher even if its mean error does not rise.

| Quantity | Status |
|---|---|
| Mean per-event peak E[maxₜ δe⁺] | Official formal metric (`*_peak_pos_excess`); recomputed exactly |
| Integrated positive excess Σₜ δe⁺·dt over [0, 1.5) s | Official formal metric (`*_int_pos_excess`); recomputed exactly |
| Peak of the plotted mean curve maxₜ E[δe⁺] | New, descriptive; paired seed bootstrap of the difference of maxima |
| Sub-window means (0–0.5, 0.5–1.0, 1.0–1.5 s) | New, descriptive, post-hoc; no multiplicity correction |

The two peak quantities differ: maxₜ E[δe⁺] ≠ E[maxₜ δe⁺]. Their difference is reported, never their equality.

**Fig. E3(c): base XY after EE switches** (B = 0, T = 2 s; 1,200 matched switches in 100 seeds)

| | C1 | C3 | C5 | C3 − C1 [95 % CI] | C5 − C1 [95 % CI] |
|---|---|---|---|---|---|
| Mean per-event peak (m/s), formal | 0.0388 | **0.0320** | 0.0706 | −0.0068 [−0.0084, −0.0052], C3 lower (Holm) | 0.0318 [0.0294, 0.0342], C1 lower |
| Integrated (m), formal | **0.0174** | 0.0179 | 0.0355 | 0.0006 [−0.0005, 0.0017], n.s. | 0.0182 [0.0168, 0.0195], C1 lower |
| Peak of mean curve (m/s), descriptive | 0.0269 | **0.0171** | 0.0342 | −0.0098 [−0.0114, −0.0082] | 0.0073 [0.0049, 0.0097] |
| Mean over 0–0.5 s (m/s), descriptive | 0.0177 | **0.0128** | 0.0238 | −0.0049 [−0.0059, −0.0039] | 0.0061 [0.0048, 0.0074] |
| Mean over 0.5–1.0 s (m/s), descriptive | **0.0095** | 0.0121 | 0.0281 | 0.0026 [0.0016, 0.0036] | 0.0186 [0.0172, 0.0200] |
| Mean over 1.0–1.5 s (m/s), descriptive | **0.0076** | 0.0111 | 0.0192 | 0.0035 [0.0027, 0.0042] | 0.0116 [0.0104, 0.0128] |
| Pre-switch level of the curve (m/s), rectified floor | 0.0012 | 0.0023 | 0.0037 | | |

**Fig. E3(d): EE position after base switches** (B = 0.60, EE fixed; 652 matched switches in 90 seeds)

| | C1 | C3 | C5 | C3 − C1 [95 % CI] | C5 − C1 [95 % CI] |
|---|---|---|---|---|---|
| Mean per-event peak (m), formal | **0.0118** | 0.0201 | 0.0171 | 0.0083 [0.0070, 0.0097], C1 lower (Holm) | 0.0053 [0.0040, 0.0066], C1 lower |
| Integrated (m·s), formal | **0.0058** | 0.0106 | 0.0090 | 0.0048 [0.0038, 0.0057], C1 lower | 0.0031 [0.0021, 0.0043], C1 lower |
| Peak of mean curve (m), descriptive | **0.0057** | 0.0106 | 0.0079 | 0.0049 [0.0038, 0.0060] | 0.0022 [0.0009, 0.0036] |

**How the rankings change against the signed-mean presentation**

- **Fig. E3(c), base XY.** The signed-mean curve, used in figure versions v1–v3, showed RL+IK lowest throughout the response: its signed integral is lower than C1's by 0.0052 m [0.0035, 0.0068] (descriptive). With the positive part taken per event, this changes:
  - **The ranking is time-dependent.** RL+IK is lower only in the first 0.5 s, which contains the peak. C1 is lower from 0.48 s onwards.
  - **Integrated:** C1 and RL+IK are not significantly different. The official formal verdict is unchanged.
  - **Peak:** RL+IK still has the lower peak, both per event and on the curve.
  - **Cooperative RL** is highest at almost all times. C1 exceeds it only during 0.06–0.18 s, where C1's earlier rise leads.
  - **Interpretation:** RL+IK's late positive excess (about 0.011 m/s) far exceeds its late signed mean (about 0.0014 m/s). This means its base error rises after some switches and falls after others. The lowest late increase therefore belongs to C1, but part of every late value is the rectified fluctuation (see the pre-switch floors).
- **Fig. E3(d), EE position.** The signed mean made Cooperative RL look near zero. Under positive excess, C1 is lowest and RL+IK highest, consistent with the formal metrics (see §3 robustness).
- **Not plotted:** the yaw and EE-orientation positive excess are in the CSV only. For yaw after EE switches, the peak of Cooperative RL's mean curve is the lowest (0.039 vs 0.062 rad/s for C1), although its mean per-event peak is the highest (0.101 vs 0.077). This is a further example of max E ≠ E max.

**Recomputation checks** (`validation/positive_excess_checks.json`), at all eight event conditions (B0T1–B0T4, B1T0–B4T0):
- the raw traces equal the stored matched traces;
- the signed per-event-baseline means equal the stored excess (≤ 7e-16);
- the per-event peak and integral equal the official metrics (≤ 4e-17).

**Other levels.** By curve maximum:
- base XY at T = 6, 4 and 3 s: RL+IK lower than C1, and C1 lower than Cooperative RL, all with CIs excluding 0;
- EE position at B = 0.15–0.45: C1 lowest, with CIs excluding 0.

**Signed curves** remain available in the supplementary figures and the curves CSV.

## 4. Pre-registered vs post-hoc

**Frozen before the run** (`E3_PROTOCOL.md`, git `30b767a`): every metric, window, bin and main condition above.

**Added after the run** (decided after the RMSE results but before the event results were computed):
- the **matched-event restriction** of the event statistics and Fig. E3(c)–(d), because C5's falls truncate its event sets; the unmatched all-event results are retained as sensitivity;
- the **event-level positive-excess visualisation** in both cross-task panels, now Fig. E3(c) and (d). This includes the curve-maximum and sub-window statistics. It was chosen after inspecting the signed-mean figures, in which opposite-sign responses cancelled. The official metrics are unchanged;
- **membership of the formal Holm families** for events: positive-excess peak/integral and recovery of the disturbed signals, the EE response (mean and residual) for EE→base, and the absolute pre-switch EE error for base→EE. The protocol defined these quantities but did not list the family members.

## 5. Files

| File | Content |
|---|---|
| `stats/E3_summary.csv` | All metrics per controller × cell |
| `stats/E3_paired_tracking_and_reaching.csv` | Paired RMSE and reaching comparisons |
| `stats/E3_degradation*.csv` | Directional degradation and C1-vs-rival differences |
| `stats/E3_events_matched_*.csv` | Formal event statistics |
| `stats/E3_events_main_*.csv`, `stats/E3_events_all_cells.csv`, `stats/E3_events_level_sweeps_paired.csv` | Unmatched event statistics, all cells, level sweeps |
| `stats/E3_events_baseline_window_sensitivity.csv` | Baseline-window sensitivity |
| `stats/E3_events_jump_bins.csv`, `stats/E3_events_jump_slopes.csv` | Jump-magnitude analysis |
| `stats/E3_base_to_ee_by_pose0_height.csv` | Target-consistency analysis |
| `stats/E3_sensitivity_completed_only.csv` | Completed-only sensitivity |
| `stats/E3_completion.csv` | Completion per controller and cell |
| `stats/E3_target_jump_audit.csv` | Target jump audit |
| `stats/E3_event_positive_excess.csv`, `stats/E3_event_positive_excess_curves_main.csv` | Post-hoc event-level positive excess (EE and base switches, all levels), paired; main-condition curves |
