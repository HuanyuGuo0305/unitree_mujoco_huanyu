# E5 — Dynamic object retrieval across cube size and density (MuJoCo)

Evaluates the learned high-level retrieval policy of the B2W + Z1 wheeled-legged mobile manipulator
(Isaac Lab `B2WZ1HLRetrievalUANABSEnv`, run `2026-09-14_13-37-59`, `model_8000`) with its frozen WBC
(`2026-09-07_20-02-17`) and frozen UAN arm actuator. The robot must approach, physically grasp (through contact,
no attachment) and carry a cube to a random 3-D target.

**Protocol (official, revision 4): four 1-cm groups over 3.0–7.0 cm, 500 episodes each (2000 total).**
Each group is the union of two half-cm strata that replay the SAME 250 base scenarios:

| Group | Evaluation | Stratum A | Stratum B | Source |
|---|---|---|---|---|
| [3.0, 4.0] | OOD, smaller than training | [3.0, 3.5) | [3.5, 4.0) | OOD extension run |
| [4.0, 5.0] | in-distribution | [4.0, 4.5) | [4.5, 5.0) | base run (reused) |
| [5.0, 6.0] | in-distribution | [5.0, 5.5) | [5.5, 6.0] | base run (reused) |
| [6.0, 7.0] | OOD, larger than training | [6.0, 6.5) | [6.5, 7.0] | OOD extension run |

- Every base scenario fixes `size_u, density_u ~ U(0,1)` and all other draws (materials, base inertia, reset state, cube
  position, target, noise seed). A stratum [lo, hi) uses `side = lo + size_u (hi − lo)` (unrounded) and
  `density = 400 + 800 density_u` kg/m³; `mass = density · side³` (0.0108–0.4116 kg over the domain),
  `inertia = mass · side² / 6` (MuJoCo read-back checked for every episode). Pairing key `(scenario_id, stratum)`;
  inference clusters on the 250 scenario ids.
- The 4–6 cm strata are the immutable base run `results/20261116T154934Z_official250_retention` (1000 episodes, SHA-256
  pinned in `config.yaml: official`); only the four OOD strata were simulated (`run.py --ood-extension`), and
  `combine.py` merges both runs with full provenance. The two half-cm strata of a group form a stratified uniform sample
  of the 1-cm interval; group statistics (incl. completion time) are recomputed from the raw episodes.
- **Training used sizes 4–6 cm and a fixed density of 800 kg/m³.** [3, 4] and [6, 7] are size OOD; the density range is an
  evaluation-time ±50 % perturbation in every group. Mass follows volume, so size effects include mass effects.
- Simulator, policy, WBC, UAN, contact model, no-slip 5, noise, horizon and task logic are unchanged across all runs.

Alignment with the training environment: [docs/ALIGNMENT_REPORT.md](docs/ALIGNMENT_REPORT.md).
Physical validation: [docs/PHYSICAL_VALIDATION.json](docs/PHYSICAL_VALIDATION.json).

## Metrics

- **Task success**: curriculum Stage 2 reached and ǀǀcube − targetǀǀ < 0.15 m for 3 consecutive 10-Hz HL steps, before
  the 149-step (14.9 s) timeout and without a simultaneous termination. Rate over **all** trials.
- **One-shot success**: task success AND continuous *physical* retention of the cube from its first physical
  acquisition until the success instant -- no drop and re-grasp. Rate over **all** trials. Evaluated offline by
  `retention.py` from a 250-Hz log (`ph_*` arrays in every trajectory file), independently of the training Stage-2 state:
  - *load-bearing pinch*: both jaws carry >= 10 N (supporting the heaviest cube under 3 g at the lowest training jaw
    friction needs <= 9.5 N; settled grasps squeeze with ~450 N). Touching or single-jaw contact is not retention.
  - *acquisition*: a load-bearing pinch held for one HL period (0.1 s) with the jaw stalled (angle range <= 1 deg) and the
    cube moving rigidly with the gripper (<= 2 mm); pinches that do not settle (jaw keeps closing while the cube squeezes
    out) are acquisition *attempts*.
  - *drop (release)*: after acquisition, an interruption of the load-bearing pinch that lasts >= 40 ms, or during which
    the cube moves >= side/2 relative to the jaws, or the unblocked jaw closes >= 5 deg; shorter, smaller interruptions are
    contact flicker. *Re-grasp*: the next acquisition after a drop.
  - Validation: base run (4–6 cm) — interruptions of kept grasps <= 32 ms, grip losses >= 56 ms; OOD run (3–4, 6–7 cm)
    — flicker <= 36 ms, releases >= 52 ms. One-Shot counts change by at most 3 per group under any single-threshold
    variation (`combined/detector_validation.json`). Thresholds unchanged since revision 3.
- **Relation**: one-shot <= task success per episode. Over all 2000 episodes they differ for 18 of 1856 successes, every
  one a genuine acquire → drop → re-grasp (5 / 4 / 3 / 6 in the four groups). All successful episodes contain a physical
  acquisition; the training Stage-2 flag is not used as retention evidence.
- **Completion time**: 0.1 s × HL step at which task success latches (approach + grasp + transport, simulation time);
  mean ± SD over successful trials only.
- Failure reasons: `grasp_not_achieved`, `timeout` (grasped, not delivered), `grasp_lost_after_confirmation`,
  `base_contact_failure`, `grasp_error_out_of_bounds`, `numerical_failure`.

## Official result — `results/20261116T165310Z_ood_extension/combined/`

| Size (cm) | SR (%) ↑ [95% CI] | One-Shot (%) ↑ [95% CI] | Time (s) ↓ | succ. / one-shot / n |
|---|---:|---:|---:|---:|
| [3.0, 4.0] (OOD) | 96.4 [94.4–97.7] | 95.4 [93.2–96.9] | 6.31 ± 1.96 | 482 / 477 / 500 |
| [4.0, 5.0] | 96.8 [94.9–98.0] | 96.0 [93.9–97.4] | 6.17 ± 1.85 | 484 / 480 / 500 |
| [5.0, 6.0] | 95.4 [93.2–96.9] | 94.8 [92.5–96.4] | 6.20 ± 1.88 | 477 / 474 / 500 |
| [6.0, 7.0] (OOD) | 82.6 [79.0–85.7] | 81.4 [77.8–84.6] | 6.46 ± 2.14 | 413 / 407 / 500 |

(completion time over successful trials; Wilson CIs; 1000 reused + 1000 new episodes; all provenance checks pass)

- **Smaller OOD ≈ in-distribution**: vs [4, 6] cm, Δ SR +0.3 pp [−1.8, +2.3], Δ One-Shot +0.0 pp [−2.1, +2.1]
  (paired cluster bootstrap over scenario ids); completion +0.13 s [+0.02, +0.24].
- **Larger OOD degrades**: Δ SR −13.5 pp [−17.2, −9.9], failure ratio 4.5× [3.2, 6.7], Δ One-Shot −14.0 pp
  [−17.9, −10.2]; worse in [6.5, 7.0] (77.6 % SR) than in [6.0, 6.5] (87.6 %).
- Failures in [6, 7] are post-grasp losses (71 of 87): the cube is physically acquired in 98 % of episodes but slides
  out of the strongly opened, non-parallel jaws (median slip 133 mm in dropped vs 4.7 mm in kept grasps); drop rate rises
  with size (10.7 → 26.0 % over size quartiles) and with mass (6.6 → 22.1 %), not with friction or penetration — consistent
  with the physical wedge behaviour found in validation (V4). Size and mass are coupled; no single cause is claimed.
- Retention detector re-validated on OOD data (flicker ≤ 36 ms, releases ≥ 52 ms; One-Shot changes by ≤ 3 per group under
  any threshold variation). 9 large-cube episodes were "confirmed" by the training proxy while the cube kept sliding out
  (never physically settled) — failures under both metrics, reported as physical findings.
- Files: `combined/summary_table.tex` (paper table), `combined/episodes_combined.csv` (2000 episodes + provenance),
  `combined/COMBINED.md`, `ood_vs_id.json`, `physics_diagnostics.*`, `detector_validation.json`, `provenance.json`.
- Supporting runs: base run (4–6 cm, its own `analysis/`), `20261116T165137Z_ood_pilot5` (OOD pilot, 20 episodes),
  `20261116T155854Z_pilot5` (base-protocol pilot).

Absolute numbers are specific to this MuJoCo model and must not be presented as Isaac Lab performance.

## Layout

| File | Purpose |
|---|---|
| `config.yaml` | checkpoints + hashes, size intervals, density range, trials, seed, cube physics, randomization, noise, task constants |
| `simulator.py` | engine: per-(side, density) MjSpec model, materials, reset, 250/50/10 Hz WBC+UAN+HL loop, contacts, `run_episode` |
| `metrics.py` | task state machine (port of the Isaac Lab stage / success / termination code), Wilson CI |
| `retention.py` | physical acquisition / drop / re-grasp detection from the 250-Hz log → One-Shot Success |
| `scenarios.py` | paired scenario bank (one RNG stream per scenario; size / density quantiles) and interval helpers |
| `run.py` | parallel runner; writes a new results directory, then runs the analysis |
| `combine.py` | official 3–7 cm analysis: verifies and merges the base run and the OOD extension run (provenance, regression checks, cluster bootstrap OOD vs ID, physics and detector diagnostics, paper table) |
| `analyze.py` | tables (CSV / LaTeX), group metrics, physical retention events, one-shot audit and threshold sensitivity, paired tests, density/mass diagnostics, consistency |
| `validate.py` | physical validation over size × density × friction corners → `docs/PHYSICAL_VALIDATION.json` |
| `tests/` | checkpoints/constants vs training, state machine vs training source, physical retention / one-shot cases (`test_retention`), per-sample model, scenarios, bit-exact deploy parity, OOD extension and data reuse (`test_extension`) |

## Commands (from `unitree_mujoco/simulate_python`)

```bash
# 1. tests (~20 s, 62 tests)
for t in test_alignment test_stage_machine test_retention test_scenarios test_model test_deploy_parity test_extension; do
  python3 -m unittest retrieval_experiment.tests.$t -v; done
# 2. physical validation incl. OOD extremes (~2 min)
python3 -m retrieval_experiment.validate
# 3. OOD pilot (first 5 base scenarios x 4 OOD strata = 20 episodes; combined in partial mode)
python3 -m retrieval_experiment.run --ood-extension --trials-per-group 5 --tag ood_pilot5
# 4. official incremental OOD simulation: 250 x 4 OOD strata = 1000 new episodes, then combine
python3 -m retrieval_experiment.run --ood-extension
# 5. regenerate the combined official analysis and LaTeX table (no simulation)
python3 -m retrieval_experiment.combine retrieval_experiment/results/20261116T165310Z_ood_extension
#    (base-protocol run, only if the 4-6 cm base data had to be regenerated -- not needed)
#    python3 -m retrieval_experiment.run --tag official250
```

CLI: `--size-intervals 4.0-4.5,4.5-5.0,...` `--trials-per-group N` `--seed S` `--workers W` `--tag T`
`--output-dir D` `--observation-noise {training,none}` `--noslip-iterations K` (sensitivity only) `--scenarios path`
`--no-trajectories` `--no-analysis` `--ood-extension`.

Each run writes `results/<UTC>_<tag>/`: `config.yaml`, `scenarios.json`, `policy_manifest.json`,
`model_parameters.json`, `episodes.csv/.jsonl`, `trajectories/*.npz` (10-Hz task traces + 250-Hz retention log), `metadata.json`, `analysis/`
(`ANALYSIS.md`, `summary_table.csv/.tex`, `group_metrics.csv`, `episodes_retention.csv`, `oneshot_sensitivity.csv`, `failure_breakdown.csv`, `stage_decomposition.csv`,
`density_mass_bins.csv`, `paired_tests.json`, `oneshot_audit.json`, `consistency.json`).
