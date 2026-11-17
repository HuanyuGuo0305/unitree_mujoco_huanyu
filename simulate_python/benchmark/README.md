# B2W+Z1 whole-body control benchmark (E1–E3)

This is a MuJoCo Sim2Sim benchmark of the existing, validated B2W+Z1 loco-manipulation controllers. No controller is trained or modified here; the benchmark only issues commands, logs and evaluates.

| Exp. | Question | Controllers | Protocol | Episodes |
|---|---|---|---|---|
| **E1** Whole-body tracking ablation | Simultaneous base-velocity and EE-pose tracking accuracy | C1 Unified RL (Ours), C3 RL+IK, C5 Cooperative RL | 4 conditions: T1 Still/Fixed, T2 Still/Random, T3 Moving/Fixed, T4 Moving/Random; seeds 0–499 | 6,000 |
| **E2** Actuation efficiency and smoothness | Wheeled-legged vs legged actuation | C1 Wheeled-Legged, C2 Legged (wheels welded) | 1 dynamic condition; seeds 0–99 | 200 |
| **E3** Dynamic whole-body coordination | Bidirectional tracking: base and EE tracking under increasing base motion and EE update frequency | C1, C3, C5 | 7 base-motion bounds × 7 EE-update frequencies; seeds 0–99 | 14,700 |

**Names.** Paper-facing outputs use the names Unified RL (Ours), RL+IK and Cooperative RL. The code and CSVs keep the internal ids C1/C3/C5 (and C2 for Legged).

## Shared protocol

- **Episodes:** 30 s. Settling 0–3 s, with a zero base command and the measured initial EE pose. Metrics over **3–30 s**; the native startup ramp reaches full policy authority at 3 s (`docs/interface_audit.md` §10.2).
- **Rates:** physics 250 Hz (dt 0.004 s, `implicitfast`); policy 50 Hz.
- **Commands:**
  - Base velocity is piecewise constant within the shared WBC command range of ±0.6 m/s, ±0.6 m/s and ±0.6 rad/s (`configs/baselines.yaml`).
  - EE targets are full 6-DoF poses, given as 9-D PLB keypoints from the native training table `utilities/data/keypoints_plb_evaluation_filtered.npy`, switched directly with no interpolation.
- **Pairing:** every schedule depends only on (experiment, condition, seed), so all compared controllers receive bit-identical commands and identical seeded initial states.
- **Failures:** terminations (base height < 0.30 m or tilt > 50°) are kept and scored over their valid window.
- **Determinism:** single-threaded math is pinned, so episodes are bit-reproducible.
- **Statistics:** per-seed metrics first; 95 % percentile bootstrap with 10,000 resamples and seed 20261106; paired by seed; Holm correction within declared families.
- **Frames:** base velocity errors in the body frame; EE errors in the PLB frame (yaw-only base frame at ground height); orientation as the SO(3) geodesic.

**E1.** Base commands are resampled every U(2, 3) s; T1/T2 have a zero base command. EE targets are fixed (T1/T3) or resampled every 3.0 s (T2/T4). Metrics: base XY / yaw-rate / EE position / EE orientation RMSE. Config `configs/E1.yaml`; builder `build_e1`.

**E2.** Base commands as in E1 (every U(2, 3) s, independent uniform draws per axis). EE poses every U(3, 5) s, using an independent RNG stream. Config `configs/E2.yaml`; builders `build_e2` and `make_ee_random_interval_schedule`. Primary metrics:
- average mechanical power Σ|τ·q̇| over leg, arm, gripper and wheel actuators;
- energy per metre of realized progress along the commanded direction;
- mean |q̈| of the 6 arm joints;
- mean |q̈| of the 12 leg joints.

Definitions and checks are in `results/E2/E2_REPORT.md`.

**E3** (bidirectional 7 × 7, official since 2026-11-13).
- **EE targets:** full-pose targets drawn from the whole training table (`configs/E3_feasible_rows.npy`).
- **EE update frequencies:** Fixed, 1/6, 1/5, 1/4, 1/3, 1/2 or 1 Hz.
- **Base bounds:** symmetric B ∈ {0, 0.1, …, 0.6}, with segments every U(2, 4) s.
- **Metrics:** base XY, yaw-rate, EE position and EE orientation RMSE.
- **Paper figure** (`wbc_coordination`): four groups, all slices of the one 7 × 7 grid.
  - (a), (b): base and EE tracking against B, at 1/2 Hz.
  - (c), (d): base and EE tracking against EE update frequency, at B = 0.

Config `configs/E3.yaml`; protocol `docs/E3_PROTOCOL.md`; builder `build_e3`. The 5 × 5 E3 is archived (`docs/E3_BIDIRECTIONAL_REPLACEMENT.md`).

## Layout

```
benchmark/
├── run.py                 simulation driver (all experiments)
├── configs/               baselines.yaml (shared), E1.yaml, E2.yaml, E3.yaml, E3_feasible_rows.npy
├── scenarios/experiments.py   command schedules per experiment (build_e1 / build_e2 / build_e3)
├── core/                  runner, recorder (metrics, 250 Hz power/acceleration), scenario generators, frames, stats
├── controllers/           adapters around the original deploy scripts (registry, C1/C2, C3/C4, C5/C6, UAN)
├── e1/analyze.py          E1 statistics + LaTeX table from saved episodes
├── e2/analyze.py, e2/figures.py               E2 statistics/validation; preliminary figure
├── e3/                    E3 pose audit, pre-run validation/freeze, extraction, analysis, paper figure
├── plots/style.py         shared plot style
├── round3/common.py       shared I/O helpers (historical package name)
├── tests/                 unit tests + controller parity scripts
├── docs/                  interface audit, E3 protocol and replacement records
├── paper_assets/E1|E2|E3/ tracked copies of the official numbers / paper figure
└── results/E1|E2|E3/      official results (git-ignored; see each folder's README / report)
```

## Requirements

- **Environment:** run from `simulate_python/` in the project environment (`env_unitree_mujoco`). Packages: `mujoco`, `numpy`, `scipy`, `torch`, `onnxruntime`, `pyyaml`, `matplotlib`, `pillow`.
- **Checkpoints and assets** (repository root):

| Controller | Files |
|---|---|
| C1 | `exported_policy/b2wz1_wbc_plb_uan/policy_unified_1.onnx` |
| C2 | `exported_policy/b2wz1_wbc_plb_legged_uan/policy_1.onnx` |
| C3 | `exported_policy/b2wz1_vbc_plb_uan/policy_3.onnx` |
| C5 | `exported_policy/b2wz1_roboduet_plb_uan/{arm_policy,loco_policy}.onnx` |
| All (frozen UAN arm actuator) | `exported_policy/b2wz1_wbc_plb_uan/model_4000_inference.pt` |
| Models | `unitree_robots/b2wz1/ground.xml` (wheeled), `unitree_robots/b2wz1/b2wz1_legged.xml` (C2) |
| EE table | `utilities/data/keypoints_plb_evaluation_filtered.npy` |

Deploy configs are in `simulate_python/configs/b2wz1_*_uan.yaml`.

## Running (from `simulate_python/`)

The default output folder is `benchmark/results/<EXP>/raw`. `--resume-missing` skips completed episodes.

```
# E1 (6,000 episodes)
python3 benchmark/run.py --experiment E1 --baselines C1,C3,C5 --seed-start 0 --num-seeds 500 --headless --jobs 12 --log-physics
python3 -m benchmark.e1.analyze

# E2 (200 episodes, ~1 min at --jobs 14)
python3 benchmark/run.py --experiment E2 --baselines C1,C2 --seed-start 0 --num-seeds 100 --headless --jobs 14 --log-physics
python3 -m benchmark.e2.analyze
python3 -m benchmark.e2.figures          # preliminary inspection figure only

# E3 (14,700 episodes, at --jobs 14; smoke test: docs/E3_PROTOCOL.md section 8)
python3 -m benchmark.e3.prerun_validation --freeze   # schedule checks + frozen schedules/protocol snapshot
python3 benchmark/run.py --experiment E3 --baselines C1,C3,C5 --seed-start 0 --num-seeds 100 --headless --jobs 14
python3 -m benchmark.e3.analyze
python3 -m benchmark.e3.paper_figures    # wbc_coordination.pdf/.svg/.png
```

**Regenerating from saved results.** Every analysis step above reads only the saved episodes; none simulates. To compare with the official outputs without overwriting them:
- **E1:** `python3 -m benchmark.e1.analyze --out <scratch>`.
- **E3:** `python3 -m benchmark.e3.analyze --out <scratch>`; for the figure, set `benchmark.e3.paper_figures.E3` and `.OUT` to that scratch folder before calling `main()`.

## Official results

| | Results (git-ignored) | Tracked copy | Paper item |
|---|---|---|---|
| E1 | `results/E1/` (README, `E1_500seed_*`, `raw/`) | `paper_assets/E1/` | Tracking table `E1_500seed_tracking_rmse.tex` |
| E2 | `results/E2/` (`E2_REPORT.md`, `E2_summary_table.csv`, `raw/`) | `paper_assets/E2/` | Efficiency/smoothness table (layout not finalized) |
| E3 | `results/E3/` (`E3_REPORT.md`, `protocol/`, `data/`, `stats/`, `figures/paper/`, `raw/`) | `paper_assets/E3/` | Figure `wbc_coordination.pdf` (+ `.svg`, `.png`, `_values.csv`) |

Each results folder has a `MANIFEST_SHA256.txt`; E1 and E2 also have `RAW_SHA256.txt`, checked from inside `raw/`. Each `paper_assets/E*` folder has `ASSETS_SHA256.txt`.

## Verifying the installation

```
python3 -m unittest benchmark.tests.test_scenarios benchmark.tests.test_protocol benchmark.tests.test_e2
python3 benchmark/tests/test_latex_structure.py          # structural check of the official .tex (no LaTeX here)
python3 benchmark/tests/test_controller_parity.py         # adapters vs original deploy scripts (slow; runs simulations)
python3 benchmark/tests/test_bitexact_parity.py           # bit-exact version of the above (slow)
(cd benchmark/results/E2 && sha256sum -c --quiet MANIFEST_SHA256.txt)   # likewise for E1, E3, paper_assets
```

## History

The 5 × 5 E3 was superseded on 2026-11-13: code at git tag `e3-5x5-pre-bidirectional-replacement`, results in `unitree_mujoco_benchmark_archive_E3_5x5_2026-11-13/` (next to the repository); see `docs/E3_BIDIRECTIONAL_REPLACEMENT.md`.

Superseded protocols and results were removed from this tree on 2026-11-12:
- round-1/2/3 tooling and tables;
- the E1 50-seed tables;
- the round-3 and E1-matched (T1–T4) E2;
- E4;
- earlier E3 figure versions.

Recover them from:
- **Code:** git tag `benchmark-pre-cleanup-2026-11-12`; older states are at tags `e2-round3-pre-replacement`, `e3v3-pre-random-pose-replacement` and `e3-pre-v3-cleanup`.
- **Results:** `unitree_mujoco_benchmark_archive_2026-11-12/`, in the folder that contains the `unitree_mujoco` repository. It holds the moved result folders and a full code snapshot that includes the then-uncommitted work.
