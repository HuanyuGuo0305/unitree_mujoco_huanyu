# E3 — Reproducibility report

## Configuration

- **Protocol:** `E3_PROTOCOL.md`, with a tracked copy in `benchmark/docs/E3_PROTOCOL.md`.
- **Config:** `benchmark/configs/E3.yaml`; a copy at launch is in `protocol/E3_config.yaml`.
- **Sampling set:** `benchmark/configs/E3_feasible_rows.npy` (all 20,000 rows; sha256 `d7887b88…4a80`).
- **Target table:** `utilities/data/keypoints_plb_evaluation_filtered.npy` (sha256 `d822f110…c328`, identical to the training file).
- **Fixed analysis bins:** `protocol/analysis_bins.json`.
- **Controllers:** C1, C3 and C5 with their existing checkpoints and deploy configs. Nothing was retrained or modified.

## Code and history

| Step | Git |
|---|---|
| Snapshot before the E3 v3 → random-pose replacement | tag `e3v3-pre-random-pose-replacement` (`64a2933`) |
| Protocol, schedule, builder, config, audit, validation (frozen before the run) | `30b767a` on branch `benchmark-e3v3` |
| Analysis, figures and reports | committed after the run (see `git log`) |

**Deleted data.** The previous E3 v3 data was deleted; the per-file SHA-256 is in `cleanup/deleted_E3_v3_sha256.txt.gz`, and the scope is in `benchmark/docs/E3_RANDOM_POSE_REPLACEMENT.md`.

**Code state.** SHA-256 of the 25 simulation-relevant files (runner, core, controllers, scenario builder, configs, deploy configs, target table, sampling set) is in `protocol/code_sha256_at_launch.txt`. All are unchanged after the run.

## Execution

```
python3 benchmark/run.py --experiment E3 --baselines C1,C3,C5 --seed-start 0 --num-seeds 100 \
        --headless --jobs 14 --resume-missing --output-dir benchmark/results/E3/raw
```

- **Timing:** 2026-11-11 15:41:20 → 16:04 CET, 22.8 min. 7,500 / 7,500 episodes, 0 runner failures. Log: `run_E3.log`.
- **Workers:** 14, chosen conservatively for 30 GB RAM. About 4 GB stayed available and swap was flat throughout.

## Regenerating every output

```
python3 -m benchmark.e3.pose_audit          # validity screen + kinematic audit (about 1.5 min)
python3 -m benchmark.e3.prerun_validation   # V1-V7; V8 with --episodes <trial dir>
python3 -m benchmark.e3.analyze             # validation + per-trial + statistics (about 20 s)
python3 -m benchmark.e3.events_matched      # matched event statistics + traces (about 5 s)
python3 -m benchmark.e3.positive_excess     # post-hoc event-level positive excess (base switches) + checks
python3 -m benchmark.e3.paper_figures       # combined paper figure + plotted values + unmatched sensitivity + supplementary
```

**Determinism.** Episodes are deterministic in (controller, condition, seed). 54 re-simulated episodes, including 4 terminations, are byte-identical to the canonical ones.

## Outputs

| Path | Content |
|---|---|
| `raw/E3/<C>/B?T?/seed_XXX/` | Per-episode logs (commands, states, actuation, metrics, metadata, schedule) |
| `data/E3_per_trial_7500.csv` | Every trial: RMSEs, completion, reaching, event aggregates, pose rows used, pose-0 error at t = 3 s |
| `data/E3_arrays.npz` | The same as arrays [controller, B, T, seed], plus event traces |
| `data/E3_event_rows_main_conditions.csv` | Per-event rows at the two main conditions |
| `data/E3_event_traces_matched.npz` | Matched per-seed event traces |
| `stats/` | See `E3_STATISTICAL_ANALYSIS.md` §5 |
| `protocol/pose_audit/` | Target sampling and kinematic audit |
| `protocol/prerun_validation.json` | Includes the jump-magnitude comparison across levels |
| `validation/validation_summary.json` | Post-run validation |
| `figures/paper/paper_fig_e3_dynamic_coordination_combined.pdf/.png` | The single combined paper figure (double column) |
| `figures/paper/paper_fig_e3_dynamic_coordination_combined_values.csv` | Exact plotted values (means and 95 % CIs) |
| `figures/paper/paper_fig_e3_dynamic_coordination_combined_unmatched_sensitivity.csv` | Groups (c)–(d) recomputed with all isolated events |
| `figures/paper/FIGURE_VALIDATION.md` | Checks of the combined figure |
| `figures/archive/superseded_four_figure_set/` | The earlier four single-column figures; superseded, not for use |
| `figures/archive/combined_v1_signed_excess_2026-11-11/` | Combined figure v1 (signed excess in (d)); superseded, not for use |
| `figures/supplementary/supp_fig_e3_base_switch_signed_vs_positive_excess.*` | Diagnostic: signed vs positive excess EE error at base switches |
| `data/E3_base_event_positive_excess_traces.npz`, `validation/positive_excess_checks.json` | Event-level positive-excess traces and their consistency checks |
| `MANIFEST_SHA256.txt` | SHA-256 of the final figure, plotted values, statistics and reports |
| `E3_PAPER_FIGURES_INSERT.tex` | LaTeX `figure*` environment |

## Deviations from the frozen protocol

1. **Event statistics and Fig. E3(c)–(d) use matched events.** These are the events present for all three controllers, a restriction added after the run because C5's falls truncate its event sets. The unmatched results are reported alongside. At the EE→base main condition the two sets are identical.
2. **Formal Holm-family membership for the event metrics** was set in code after the run, before the event results were computed (`E3_STATISTICAL_ANALYSIS.md` §4).
3. **Paper-figure presentation changed after the results** (2026-11-11). The four figures listed in protocol §8 were replaced by one combined figure with a different panel selection: EE orientation RMSE vs B is added, base XY RMSE vs B is dropped, and the event groups now pair the absolute error of the commanded task with the excess error of the other task (absolute base XY error is added in (d); excess EE orientation is dropped from the figure but kept in the statistics). The data, metrics, event sets, baselines and statistics are unchanged; every quantity shared with the earlier figures is identical.
4. **Post-hoc positive-excess panel** (2026-11-11). Fig. E3(d), lower panel, shows the event-level mean positive excess EE position error instead of the signed mean. The change was made after inspecting the signed-mean figure, in which opposite-sign responses cancelled. The official metrics are unchanged; the new curve and its curve-maximum statistic are additional and descriptive. Group titles were also renamed: (a) EE Tracking under Base Motion, (b) Base Tracking under EE Updates.
5. **No other changes:** no episodes were excluded and no controller settings changed.

The manuscript was not available, so no paper was compiled.
