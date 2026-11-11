# Validation of the final combined E3 paper figure (v2)

`paper_fig_e3_dynamic_coordination_combined.pdf/.png`. Generated on 2026-11-11, run from `simulate_python/`:

1. `python3 -m benchmark.e3.positive_excess`
2. `python3 -m benchmark.e3.paper_figures`

No simulation was rerun, and no checkpoint, metric or dataset was changed. The previous version (v1, signed excess in (d)) is preserved in `../archive/combined_v1_signed_excess_2026-11-11/`.

## Changes from v1

- **Group titles:**
  - (a) EE Tracking under Base Motion
  - (b) Base Tracking under EE Updates
  - (c) EE Command-Switch Response (unchanged)
  - (d) Base Command-Switch Response (unchanged)
- **(d) lower panel, post-hoc:** the signed-mean excess EE position error was replaced by the event-level **mean positive excess**:
  - δe⁺ᵢ(t) = max(eᵢ(t) − ēᵢ,pre, 0), using the registered per-event [−0.5, 0) s baseline;
  - the positive part is taken before averaging over events;
  - events are averaged within each seed, then the seeds are bootstrapped.
  - The y axis is nonnegative and starts at 0. The signed curve is kept in `../supplementary/`.
- **Layout and formatting:**
  - Height is now 5.20 in (was 5.40).
  - Group headings are 8.5 pt (labels 7.5 pt, ticks 7 pt, legend 8 pt).
  - Y ticks use fixed steps, with one decimal format per group pair: (a) %.1f; (b) %.2f; (c) top %.1f, bottom %.2f; (d) top %.1f, bottom %.3f. Time ticks use %.1f.
  - The gaps between groups were reduced and the legend moved closer to the plots.

## Format checks

| Check | Result |
|---|---|
| Size | 515.5 × 374.4 pt = 7.16 × 5.20 in (IEEE `\textwidth`; inserted at 100 %). The PNG is 2864 × 2080 px at 400 dpi. |
| Fonts | STIXGeneral Regular, Bold and Italic, all embedded and subset as CID TrueType (`pdffonts`). No Type 3 fonts. |
| Controller names | `pdftotext` finds no "C1", "C3" or "C5" in the PDF; the caption has none either. |
| Visual inspection | Rasterised at 150 dpi (printed width) and inspected. There are no overlapping labels or clipping, and all eight panels plus the legend are fully visible. |
| Encoding | Unified RL (Ours) is blue and solid, RL+IK orange and dashed, Cooperative RL green and dash-dot. Markers appear only on the discrete axes in (a)–(b). |
| Reference lines | A dashed line at t = 0 in (c)–(d). The signed excess panel in (c) has a zero line and allows negative values. |

## Data checks

| Panel | Check | Result |
|---|---|---|
| (a), (b) | Means and 95 % CIs against `stats/E3_summary.csv` | Identical (max difference 0) |
| (a)–(c), (d) top | Against v1 plotted values, which were validated against the official statistics | Identical (max difference 0) |
| (d) bottom | Recomputed independently from the per-tick logs (`e3/positive_excess.py`), using the registered event selection | Equal to the plotted curve (max difference 0) |
| (d) bottom | Consistency: raw seed-mean traces against the stored matched traces | Max difference 0 |
| (d) bottom | Consistency: per-event-baseline signed mean against the stored excess | Max difference 5e-16 |
| (d) bottom | Consistency: mean over events of maxₜ δe⁺ against the official `ee_pos_peak_pos_excess` | Max difference 0 |
| (d) bottom | Consistency: mean of δe⁺ over [0, 1.5) s against the official `int_pos_excess` / 1.5 | Max difference 1e-17 |
| Pairing | `boot_ci` uses one fixed index matrix per sample size (seed 20261106), so every controller, and every point in time, is resampled with the same seeds. Paired differences resample the per-seed differences. | Verified in code (`e3/stats.py:_boot_idx`) |
| Terminations | RMSE uses the official per-episode value over the valid evaluated window. Events need fully valid windows and are matched across controllers, so seeds where Cooperative RL fell before any shared switch are dropped. | (d): 90 of 100 seeds and 652 switches; (c): 100 seeds and 1,200 switches |

## Matched vs unrestricted events

The full comparison is in `*_unmatched_sensitivity.csv`.

| Curve | Matched | Unmatched | Max curve change |
|---|---|---|---|
| (c), all curves | 100 seeds | 100 seeds (the event sets are identical) | 0 |
| (d) top, base XY | 90 seeds | 100 seeds (90 for Cooperative RL) | 0.010 m/s, against a peak of about 0.62 |
| (d) bottom, positive excess | 90 seeds | 100 seeds (90 for Cooperative RL) | 0.0005 m. Curve maxima, matched → unmatched: 0.0057 → 0.0054, 0.0106 → 0.0109, 0.0079 → 0.0079 (unchanged). The ordering is unchanged. |

## Positive excess (d): statistics

EE position, B = 0.60, 90 seeds. The source is `stats/E3_base_event_positive_excess.csv`.

| Quantity | Ours | RL+IK | Cooperative RL | RL+IK − Ours | Coop. − Ours |
|---|---|---|---|---|---|
| E[maxₜ δe⁺], mean per-event peak (m), official, Holm | 0.0118 [0.0108, 0.0129] | 0.0201 [0.0188, 0.0216] | 0.0171 [0.0158, 0.0185] | 0.0083 [0.0070, 0.0097] | 0.0053 [0.0040, 0.0066] |
| Mean δe⁺ over [0, 1.5) s (m), official integral / 1.5 | 0.0039 | 0.0070 | 0.0060 | 0.0032 [0.0025, 0.0038] | 0.0021 [0.0014, 0.0028] |
| maxₜ E[δe⁺], peak of the plotted curve (m), descriptive | 0.0057 [0.0050, 0.0066] | 0.0106 [0.0097, 0.0117] | 0.0079 [0.0068, 0.0093] | 0.0049 [0.0038, 0.0060] | 0.0022 [0.0009, 0.0036] |

**The two peak quantities differ.** maxₜ E[δe⁺] ≠ E[maxₜ δe⁺]. The plotted curve peaks at about half the mean per-event peak, because individual peaks occur at different times.

## Signed vs positive excess

The comparison is shown in `../supplementary/supp_fig_e3_base_switch_signed_vs_positive_excess.pdf`, for position and orientation.

**Signed mean**
- Cooperative RL ≈ 0 throughout: the signed integral is −0.0006 m·s, because increases and decreases cancel.
- Ours peaks at 0.0034 m and RL+IK at 0.0096 m.
- Read on its own, this would wrongly suggest that Cooperative RL is the least perturbed.

**Positive excess**
- The ranking is Ours < Cooperative RL < RL+IK, consistent with the official per-event metrics.
- Cooperative RL stays elevated longest: 0.0059 m over 1.0–1.5 s, against 0.0048 m for RL+IK and 0.0033 m for Ours.
- Before the switch, all curves sit at a rectified-noise floor of 0.0012–0.0017 m.
- Ours is the lowest curve from 0.16 s onwards. It is within 0.0002 m of Cooperative RL over 0.10–0.14 s and slightly above RL+IK at 0–0.02 s, where every curve is still at its floor.

## Evidence not hidden by the figure

- **(c) bottom:** RL+IK has the lowest induced base XY disturbance. Its curve peaks at 0.010 m/s, against 0.024 for Ours; the per-event peaks are 0.032 vs 0.039 m/s. Ours reduces EE error fastest (top panel).
- **(b):** Ours has the lowest absolute base error. Its translational frequency sensitivity is not lower than RL+IK's (no significant paired difference). The caption makes no sensitivity claim.
- **(a):** EE RMSE decreases slightly with B for every controller. This is attributed to stationary-base pose bias, not to disturbance rejection.
- **(d):**
  - The lower-panel excess is an increment, not accuracy. RL+IK's absolute EE position error is lower than Ours before the switch (0.013 vs 0.034 m) and after it (0.018 vs 0.035 m over 0–1.5 s). This is stated in the conclusions and statistics, not in the figure.
  - The top-panel spike is the error with respect to a newly commanded velocity, not an external disturbance.
