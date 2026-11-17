# E4 — Dynamic Nonprehensile Transport (MuJoCo)

Paired evaluation of four high-level methods on top of the same frozen WBC + UAN:
**No Residual**, **EE Position Only**, **EE Orientation Only**, **Full Policy (Ours)**.
An unsecured puck rides on a disk rigidly attached to the Z1 gripper while the base and the
EE follow independently sampled references (Isaac Lab transport task, 15-s episodes).

Read `docs/ALIGNMENT_REPORT.md` first: it lists the sim2sim corrections, the verified policy
mapping, and the unresolved Isaac Lab/MuJoCo differences that bound how the numbers may be used.

## Layout

| File | Role |
|---|---|
| `config.yaml` | policy mapping (paths + SHA-256), scenario distributions, n, seeds, statistics |
| `sim.py` | headless engine: line-by-line port of `deploy_mujoco/b2wz1_hl_transport_uan.py` (bit-exact), plus per-scenario randomization |
| `controllers.py` | the four methods: per-method obs layout (64/61-D × 3), own ONNX, Isaac-Lab decoder |
| `scenarios.py` | paired scenario generation (all random draws precomputed, per-scenario seed stream) |
| `episode.py` | one episode in Isaac Lab step order; 50-Hz / 10-Hz logging |
| `metrics.py` | retention outcome, puck displacement (disk frame), base XY / yaw-rate RMSE (body frame) |
| `run.py` | parallel paired run → `results/<UTC>_<tag>/` (never overwrites) |
| `analyze.py` | statistics, tables (CSV/MD/LaTeX), figure — reads saved results only |
| `validate.py` | physical sanity checks P1–P6 + WBC-tilt diagnostic → `docs/PHYSICAL_VALIDATION.json` |
| `tests/` | `unittest` suites: deploy parity, alignment vs Isaac Lab sources, controllers, scenarios |

Robot / WBC / UAN / task constants are read from the deployment YAML
(`simulate_python/configs/b2wz1_hl_transport_uan.yaml`), so the interactive viewer and the
benchmark cannot drift apart.

## Protocol

* 10-Hz HL, 50-Hz WBC, 250-Hz physics/UAN; 0.5-s settle with zero residual, then 144 active
  HL steps (timeout after 149, as `DirectRLEnv`).
* Each scenario fixes puck mass U(0.5, 1.5) kg, puck friction (Isaac static μ ~ U(0.3, 1.0);
  dynamic μ stored), base mass ± 2 kg / COM ± 2 cm, reset base z/roll/pitch/velocity, arm-pose
  scale, puck spawn offset U(±1 cm), and the full base-velocity (U(±0.6), 3–5 s) and EE-target
  (HIGH/LOW/LEFT/RIGHT, 4–5 s, min-jerk 1.5/2.0 s) schedules. HIGH is measured per episode after
  settle, exactly as in training (it is identical across methods because settle is residual-free).
* Failure: `puck_drop` = puck–disk net contact force ≤ 0.05 N for 5 consecutive HL steps
  (active phase only); `base_failure` = base_link contact > 1 N; `other_failure` = numerical
  error. All failures stay in the retention denominator.
* Displacement = mean ‖[x, y]‖ of the puck centre in the moving disk frame over the active window;
  base metrics compare base_link body-frame [vx, vy, ωz] with the reference. Main-table
  continuous metrics are conditional on success (n reported); paired differences use scenarios
  where both methods succeeded.

## Commands (run from `unitree_mujoco/simulate_python`)

```bash
# tests and checks (~10 s)
python3 -m unittest discover -s transport_experiment/tests -t . -v
python3 -m transport_experiment.validate

# smoke test (5 paired scenarios) and a reproducibility re-run on the same scenarios
python3 -m transport_experiment.run --n 5 --tag smoke
python3 -m transport_experiment.run --scenarios transport_experiment/results/<smoke_dir>/scenarios.json --tag smoke_repeat

# official run: 500 paired scenarios x 4 methods (~1.5 min on 24 workers)
python3 -m transport_experiment.run --tag official500 --workers 24

# analysis only (no simulation)
python3 -m transport_experiment.analyze transport_experiment/results/<run_dir>

# interactive viewer (aligned deployment, Full policy)
python3 deploy_mujoco/b2wz1_hl_transport_uan.py configs/b2wz1_hl_transport_uan.yaml
```

Runs are deterministic: one onnxruntime/torch thread per worker; re-running a `scenarios.json`
reproduces every summary field and time series bit-for-bit (checked across worker counts).

## Official result (run `results/20261114T115952Z_official500`, 500 paired scenarios)

| Method | Retention (%) [95% CI] ↑ | Puck displacement (cm) ↓ (n succ.) | Base XY RMSE (m/s) ↓ | Yaw-rate RMSE (rad/s) |
|---|---:|---:|---:|---:|
| No Residual | 4.6 [3.1, 6.8] | 3.03 ± 0.99 (23) | 0.074 ± 0.019 | 0.048 ± 0.009 |
| EE Position Only | 16.8 [13.8, 20.3] | 2.77 ± 1.03 (84) | 0.081 ± 0.020 | 0.052 ± 0.009 |
| EE Orientation Only | 39.4 [35.2, 43.7] | 1.54 ± 1.17 (197) | 0.083 ± 0.019 | 0.051 ± 0.009 |
| **Full Policy (Ours)** | **60.0 [55.6, 64.2]** | **0.95 ± 0.64 (300)** | 0.083 ± 0.019 | 0.053 ± 0.009 |

Full vs each ablation: McNemar exact, Holm-adjusted p ≤ 5e-30. Full-only/other-only successes: 277/0, 216/0, 104/1.
Full vs Ori on the 196 scenarios where both succeed: displacement −0.82 cm [−0.98, −0.67], base XY RMSE difference
0.0003 m/s [−0.0003, 0.0009] (not significant). Details: `results/<run>/analysis/ANALYSIS.md`.
Absolute retention is MuJoCo-specific (see ALIGNMENT_REPORT §5): failures concentrate at low puck friction,
where the frozen WBC's disk tilt in MuJoCo is close to the friction angle.
