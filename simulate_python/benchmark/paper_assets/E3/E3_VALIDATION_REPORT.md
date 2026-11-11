# E3 — Validation report

All checks below passed unless stated otherwise. Sources:
- `protocol/prerun_validation.json`
- `protocol/pose_audit/`
- `validation/validation_summary.json`
- `stats/E3_events_matched_coverage.csv`

## 1. Before the official run (protocol frozen in git `30b767a`)

| Check | Result |
|---|---|
| Pose source | The target table's sha256 `d822f110…c328` equals the training file used by C1, C3 and C5 (`PresampledKeypointsDirectCommandPLB`, uniform `torch.randint`). |
| Representation | 9-D keypoint → (p, R) → keypoint round trip ≤ 2.4e-7. The vectorised rotation equals `frames.rotation_from_keypoints`. |
| Validity screen | 20,000 / 20,000 rows pass (offset lengths within 2e-7 of 0.30 m; \|cos\| ≤ 8e-7). No other filter is applied. |
| Sample-and-hold | Every EE command is the table row of its segment, and commands change only at the scheduled ticks. Fixed has no change. |
| Intervals | Switches exactly every T from tick T. Pose 0 is held from t = 0, so there is no jump at t = 3 s. |
| Base schedule | \|cmd\| ≤ B; B = 0 is exactly zero; commands = common waveform × B; segments within U(2, 4) s. |
| Pairing and independence | EE identical across base levels; base identical across EE levels; each T level is a prefix of one per-seed sequence; the builder has no controller input. Checked on 2,500 schedules. |
| Jump comparability (V7) | Position jump mean 0.539–0.545 m and orientation jump mean 1.32–1.35 rad at every level; KS p vs pooled ≥ 0.93. 4 / 6 / 8 / 13 switches per episode at T = 6 / 4 / 3 / 2 s. |
| Metric definitions (V8) | 18 trial episodes: recomputed RMSE vs `metrics.json` ≤ 2.6e-9; geodesic vs logged ≤ 3.3e-6°; issued vs scheduled ≤ 5e-9. |

**Kinematic audit.** This is diagnostic only; nothing was filtered. Of the 1,535 distinct targets used by seeds 0–99:

| Check | Share |
|---|---|
| Reachable arm-only at ≥ 1 nominal stance (hardware limits) | 88.7 % |
| Reachable at all three stances | 71.2 % |
| Reachable at neither a nominal stance nor any of 24 logged E1 T2 postures | 0.8 % |
| Inside C3's IK joint band, of those reachable at C3's stance | 83.7 % |
| Reachable at ≥ 1 of 24 logged E1 T4 moving-base postures (mean 67 % of postures) | 95.6 % |

In a random 3,000-row table sample, σ_min < 0.02 occurred in 2.0 % of reachable solutions and clearance < 0.01 m in 4.6 %. These are retained, since they belong to the training distribution.

## 2. After the official run

| Check | Result |
|---|---|
| Episodes | 7,500 / 7,500 = 3 × 25 × 100; no missing, duplicate or unreadable episodes; 0 runner failures. |
| Pairing | For every (condition, seed), the base schedule, EE schedule and initial state are identical across C1/C3/C5. The initial state is identical across all 25 conditions per seed. EE schedules are identical across base levels, base schedules across EE levels, and T levels are prefixes. |
| Protocol metadata | Exactly one set per controller. |
| Commands | Issued vs scheduled ≤ 5.0e-9. During settling: zero base command, and the EE command equals pose 0. Base within bound in every episode. |
| RMSE recomputation | All four metrics recomputed from the per-tick logs vs `metrics.json`: ≤ 2.4e-8. Geodesic vs logged: ≤ 4.0e-6°. |
| Evaluation window | 1,350 ticks in every completed episode; sample counts match. |
| Terminations | 324, each with a reason: C1 0; C3 16 (6 base height, 10 tilt); C5 308 (304 base height, 4 tilt). All retained as not completed. |
| Determinism | 54 episodes re-simulated (seeds 0, 64, 99 × 3 controllers × 6 conditions): 270 / 270 files byte-identical, including 4 terminations at identical times. |
| Code state | All 25 simulation files hashed at launch are unchanged. |
| E1/E2 preservation | The SHA-256 of the 63 canonical files in `final_round3/` and `E1_500seed/` is unchanged, and no file under `raw/E1` or `raw/E2` is newer than the baseline stamp. |

**C5 falls.** The C5 terminations concentrate at B ≥ 0.30. Completion is 88–90 % at B = 0.30, 73–82 % at B = 0.45 and 67–78 % at B = 0.60. C1 completed all 2,500 of its episodes.

## 3. Event-window coverage

**Pre-registered conditions.** At B = 0 and T = 2 s, 12 isolated EE events per episode in all 300 episodes (1,200 per controller; none truncated).

At T = Fixed and B = 0.60 (isolated base events):
- C1 and C3 have 793 events each.
- C5 has 652, because its falls truncate episodes.

**Matched events.** The paper uses events present for all three controllers: 90 seeds and 652 events. The unmatched all-event statistics are kept as a sensitivity check.

**Overlapping events.** When both subsystems switch, events are mostly overlapping; for example, at B > 0 and T = 2 s about 2.3 isolated against 9.7 overlapping EE events per episode. These are reported separately in `stats/E3_events_all_cells.csv` and never pooled with isolated ones.

**Recovery time.** Defined only when the peak excess exceeds the pre-declared floor. The defined fraction and censored fraction are reported per metric; for C1's base XY at B = 0 and T = 2 s, 91 % of events are defined and 20 % of those are censored.

## 4. Issues and how they were handled

- **First pose-audit run.** It was too slow: unpinned BLAS threads and expensive retries on unreachable targets. It was stopped and rerun with one thread per worker on a reduced audit set (all used targets plus a 3,000-row sample). The audit is diagnostic and filters nothing.
- **Unmatched base events.** C5's falls make the all-event base→EE averages at B ≥ 0.30 compare different event sets. The event figures and formal event statistics therefore use matched events (`events_matched.py`), with the unmatched results kept as a sensitivity check.
- **No silently bypassed failures.** Every validation check passed on its first evaluation.
