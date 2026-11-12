# Validation of the active E3 paper figure: `wbc_coordination_1x4_stacked` (final formatting)

Generated on 2026-11-11, run from `simulate_python/`:

1. `python3 -m benchmark.e3.positive_excess`
2. `python3 -m benchmark.e3.paper_figures`

This version changes formatting only. No simulation was rerun, and no metric, statistic, curve or confidence interval was changed.

The previous version (descriptive group titles, two-line y labels) and earlier layouts were moved out of the benchmark tree in the 2026-11-12 cleanup (see `E3_REPRODUCIBILITY_REPORT.md`), and the full history is in `E3_REPRODUCIBILITY_REPORT.md`.

## Panels

| Group | Top subplot (y label) | Bottom subplot (y label) | Condition |
|---|---|---|---|
| (a) | EE position RMSE vs B (`EE Pos. RMSE (m)`) | EE orientation RMSE vs B (`EE Ori. RMSE (rad)`) | T = 2 s; B0T4–B4T4; 100 seeds |
| (b) | Base XY RMSE vs EE update frequency (`Base XY RMSE (m/s)`) | Base yaw-rate RMSE (`Yaw RMSE (rad/s)`) | B = 0; B0T0–B0T4; 100 seeds |
| (c) | Base XY error increase, positive excess, post-hoc (`Base XY Err. Inc. (m/s)`) | Absolute EE position error (`EE Pos. Error (m)`) | B0T4; 1,200 matched switches in 100 seeds |
| (d) | Absolute base XY error (`Base XY Error (m/s)`) | EE position error increase, positive excess, post-hoc (`EE Pos. Err. Inc. (m)`) | B4T0; 652 matched switches in 90 seeds |

"Err. Inc." is the error increase relative to the pre-switch baseline. The caption defines it.

## Formatting changes from the previous version

| Item | Change |
|---|---|
| Group headers | The descriptive titles are replaced by bold `(a)`–`(d)` (8.5 pt), placed at each group's upper-left corner (the left edge of its y label, just above the top subplot). |
| Legend | One row, frameless, centred at the top. Line samples are 2.2 em long (previously 2.8 em), entries are 1.8 em apart (previously 3.0 em), there is no border padding, and the font is 8 pt (previously 8.5 pt). |
| Y labels | All eight are single strings with no newline, rotated 90° and centred on their axes. They are 7 pt, and each label sits 2.0 pt from its own tick labels. Within each group the top and bottom labels are aligned; where one subplot has narrower tick labels its gap grows, up to 9 pt. The longest label (0.93 in) is shorter than the subplot height (1.08 in). |
| Ticks and x labels | 7 pt. Previously ticks were 7.5 pt and labels 8 pt. |
| Size | 7.16 × 2.95 in. Previously 3.15 in. |

**Colours (2026-11-12):** Unified RL (Ours) #1f77b4 (blue), RL+IK #d62728 (red), Cooperative RL #2ca02c (green). Each band uses its line's colour at alpha 0.16. The previous-palette version was archived (see `E3_REPRODUCIBILITY_REPORT.md`); the plotted values are byte-identical to it.

**Unchanged:** axis limits, tick values, line styles, markers, 95 % bands (alpha 0.16), the dashed t = 0 lines, and the x labels' wording.

## Checks

| # | Check | Result |
|---|---|---|
| 1 | Only formatting changed | `wbc_coordination_1x4_stacked_values.csv` (1,260 rows: means, CIs, n) is byte-identical to the previous version's. It also matches the validated 2×2 v4 values (max difference 0). Each subplot is drawn by the same `_draw_group` code, so the y-limits and ticks are unchanged. |
| 2 | Group headers | The only figure-level texts are `(a)`, `(b)`, `(c)` and `(d)`. `pdftotext` finds no descriptive titles. |
| 3 | Legend | Compact, frameless, centred above the groups. Exact names: Unified RL (Ours), RL+IK, Cooperative RL. No C1, C3 or C5 in the PDF. |
| 4 | Y labels | All eight are single-line and rotated 90° (checked programmatically). |
| 5 | Overlap and clipping | `_check_text_layout` runs at every build. Every drawn text box lies inside the figure and no two overlap. |
| 6 | Printed size | Checked visually at 150 dpi (actual size) and at 600 dpi. Everything is legible; the smallest text is 7 pt. |
| 7 | Export | Vector PDF with STIXGeneral Regular, Bold and Italic embedded as CID TrueType; no Type 3 fonts and no raster images (`pdfimages`). SVG has text converted to paths and contains no `<image>` element. PNG is 4296 × 1770 px at 600 dpi. |
| — | Size | 515.5 × 212.4 pt = 7.16 × 2.95 in (`\textwidth`), inserted at 100 %. |

## Unfavourable results to state in the text

- (c) top: RL+IK has the lowest peak base error increase, although its increase lasts longer.
- (d): RL+IK has the lowest absolute fixed-pose EE error, although its EE error increase is the largest.
- (b): Ours' base XY sensitivity to EE frequency is not significantly lower than RL+IK's.
- (a): the slight decrease with B is not disturbance rejection.
