# E4 sim2sim alignment report — Isaac Lab (PhysX) vs MuJoCo

Date: 2026-11-14. Reference: `IsaacLab_huanyu/source/isaaclab_tasks/isaaclab_tasks/direct/b2wz1_hl_transport_UAN/b2wz1_hl_env.py`
(and the `_pos_only_` / `_ori_only_` copies), plus `params/env.yaml` saved in each training run.
MuJoCo: `simulate_python/deploy_mujoco/b2wz1_hl_transport_uan.py`, `simulate_python/configs/b2wz1_hl_transport_uan.yaml`,
`unitree_robots/b2wz1/b2wz1_disk_puck.xml` (includes `b2wz1_disk.xml`).

Legend: **E** = verified by executed test/measurement, **C** = verified by code inspection only.

## 1. Corrections made (material discrepancies)

| # | Discrepancy found | Root cause | Correction | Verified |
|---|---|---|---|---|
| 1 | Deployment ran a **different frozen WBC** than the one the transport policies were trained on | YAML `low_policy_path` pointed to `b2wz1_hl_retrieval_uan/low_policy.onnx` (= WBC run 2026-09-07). All three HL runs have `ll_load_run: 2026-10-27_12-47-31_…` | `low_policy_path → exported_policy/b2wz1_wbc_plb_uan/policy_wbc_uan_transport.onnx` (SHA-256 identical to that run's `exported/policy.onnx`) | E (hash, ONNX ≡ policy.pt) |
| 2 | **Puck friction had no effect**; puck–disk μ was always 1.0 | MuJoCo combines friction by element-wise **max**; `disk_collision` has the default μ = 1.0. PhysX uses `multiply` and the disk carries the default material (μ_s = μ_d = 1.0, no material bound in the USD) → μ_eff = μ_puck | `priority="1"` on `transport_puck_geom`: the puck's friction/solref/solimp govern all of its contacts → μ_eff = μ_puck, as in PhysX. Robot contacts unchanged | E (contact `friction[0]` == sampled μ) |
| 3 | **Single-point flat contact**: puck rocked/crept and fell off inclines with tan θ < μ (μ = 0.6 slid off a 25° disk) | Cylinder–cylinder contact goes through the convex collider, which returns one point unless `multiccd` is enabled | `<option><flag multiccd="enable"/></option>` in the scene MJCF (5 contact points). Robot-only rollout is **bit-identical** with/without the flag | E (incline test, robot rollout diff = 0) |
| 4 | Soft-contact **creep** of a statically loaded puck (1–2 mm/s) and, after stiffening, single-step **contact misses** (puck dropped ~1 mm, contact-flag chatter) | Default soft contact (solimp 0.9/0.95); near-zero resting penetration with stiffer impedance | puck geom `solimp="0.99 0.999 0.001"`, `margin="0.001"`, `gap="0.0005"` (contact detected within 1 mm, active within 0.5 mm, cf. PhysX contact/rest offset). Creep ≤ 2 mm per 5 s at 25° with 1.5 kg; 0 missed steps | E (frozen-disk P1/P5) |
| 5 | LEFT/RIGHT canonical targets at x = **0.65** | YAML diverged from training | x = **0.50** (all three `params/env.yaml`) | E (test compares to env.yaml) |
| 6 | Episode horizon 150 HL steps (15.0 s) | DirectRLEnv times out at `episode_length_buf >= max_episode_length - 1` | timeout after **149** HL steps (5 settle + 144 active) | E |
| 7 | Contact quantity = per-point maximum | PhysX `force_matrix_w` / `net_forces_w` are net forces | net world-frame force between the bodies (sign-consistent) for puck–disk and base contact | C + E (P1: net force = m g) |

Changes 1–7 are confined to the transport YAML, the transport scene MJCF and the transport deploy script.

## 2. Verified policy mapping (SHA-256 vs `IsaacLab_huanyu/logs/rsl_rl/<run>/exported/policy.onnx`)

| Method | File (`exported_policy/b2wz1_hl_transport_uan/`) | Training run | in → out |
|---|---|---|---|
| Full | `high_policy_2.onnx` (db783046…) | `unitree_b2wz1_hl_transport_uan/2026-10-27_14-31-10_single_beta_6d_four_pose_pos_rate_0p1` | 192 → 6 |
| EE Position Only | `high_policy_pos.onnx` (5d3b4193…) | `…_pos_only_uan/2026-11-13_17-47-25_single_beta_3d_position_only` | 183 → 3 |
| EE Orientation Only | `high_policy_orn.onnx` (42c74641…) | `…_ori_only_uan/2026-11-12_19-03-12_single_beta_3d_orientation_only` | 183 → 3 |
| (not used) | `high_policy_1.onnx` (1a0d36ea…) | `…/2026-10-26_17-38-10_…` — trained on the **older 2026-09-26 WBC** | 192 → 6 |
| Frozen WBC | `b2wz1_wbc_plb_uan/policy_wbc_uan_transport.onnx` (6b5aa5de…) | `unitree_b2wz1_plb_wholebody_UAN_loco_manip/2026-10-27_12-47-31_…` | 400 → 22 |
| (not used) | `b2wz1_hl_transport_uan/low_policy.onnx` (47f19872…) | WBC run 2026-09-26 (used by `high_policy_1`) | 400 → 22 |
| UAN | `model_4000_inference.pt` (db53ac18…) | byte-identical to Isaac asset `B2WZ1_DISK/uan/model_4000_multipose20260907_inference.pt` | 300 → 6 |

`model_4000_inference.pt` in the transport folder is the **UAN** (same hash), not an HL checkpoint. ONNX ≡ `exported/policy.pt` ≡
the final checkpoint `model_7999.pt` rebuilt by hand (normalizer → MLP → softplus+1+ε → Beta mean → 2u−1), max |Δ| ≤ 2e-5 (E).
Observation normalization and the Beta → [−1, 1] map are **inside** the ONNX graph; no extra transform is applied.
All three ablation envs differ from the full env only in action dim, decoder, previous-action slice and reward masking (`diff`, C).

## 3. Alignment table

| Item | Isaac Lab | MuJoCo | Status |
|---|---|---|---|
| Physics / UAN rate | 250 Hz | 250 Hz (`timestep` overridden to 0.004) | Matched (E) |
| WBC rate, HL rate, hold | 5 phys/WBC, 5 WBC/HL; HL action held 0.1 s | same loop (engine is bit-exact to deploy script) | Matched (E) |
| WBC obs / action scaling | 80-D × 5 feature-major; 0.25 / 0.10 / 4.0 | identical | Matched (C; E for parity) |
| UAN law | clip(clip(PD,nom)+0.2·Kp·a, final), 25-step history | identical | Matched (C) |
| Leg actuator | DCMotor (torque–speed curve) | constant clip (same as E1–E3) | Approximated |
| Wheel actuator | implicit velocity, kd 10, 20 N·m | velocity actuator kv 10, ±20 N·m | Matched (C) |
| Arm joint friction | 0.01 | 0 | Approximated (negligible) |
| Gripper hold | DCMotor kp 64, kd 3 | PD kp 64, kd 10 | Approximated (not in puck path) |
| Disk frame | offset [0.100, 0.045, 0] in gripperStator, R = Rx(−90°) | collision cylinder at same pose; virtual frame == geom pose (err ≤ 1.2e-7 along a dynamic episode) | Matched (E) |
| Disk geometry | r = 0.100, t = 0.004 | r = 0.100, half-length 0.002 | Matched (E) |
| Puck geometry | cylinder r 0.05, h 0.08 | same (asserted at load) | Matched (E) |
| Puck mass / inertia | U(0.5, 1.5) kg, inertia recomputed | per scenario, analytic solid-cylinder inertia | Matched (E) |
| Puck friction | μ_s, μ_d ~ U(0.3, 1.0), μ_d = min(μ_d, μ_s), multiply × 1.0 | single Coulomb μ = sampled μ_s (μ_d stored; `puck_friction_source: dynamic` switch) | **Approximated** (one coefficient) |
| Restitution | 0 | no restitution model; critically damped contact | Approximated |
| Puck spawn | after robot reset, on actual disk, dx, dy ~ U(±0.01), axis ∥ disk normal, v = 0 | identical (E: offset, axis, zero velocity) | Matched (E) |
| Contact detector | gripperStator↔Puck body-level net force > 0.05 N, 5 consecutive HL steps, settle excluded | disk_collision↔puck net force > 0.05 N, same counter | Matched semantics; **disk-only** (see §5) |
| Base termination | base_link net contact > 1 N | same | Matched (C) |
| Base mass / COM DR | add U(±2) kg (inertia scaled), COM ± 2 cm | per scenario, same | Matched (E) |
| Reset DR | z + U(0, 0.1), roll/pitch U(±0.1), 6-D vel U(±0.05), arm scale U(0.95, 1.05) | per scenario, same (ang. vel. world → body) | Matched (C) |
| Wheel/ground material DR | U(0.4, 1.2) × 1.0, restitution U(0, 0.1) | fixed μ_eff = 1.0 (E1–E3 nominal, inside range) | **Not reproduced** |
| Actor obs noise | uniform noise (training) | none (play-mode evaluation) | Intentional |
| HL obs layout | 64-D (full) / 61-D (3-D) × 3 feature-major | same (E: dims; previous-action term = last executed action) | Matched (E) |
| Residual decode | ref + a·[0.1 m, 45°], clip to WBC bounds, settle: zero residual + kp0 hold | identical for all three variants | Matched (E: 900 random cases vs transcribed Isaac decoder) |
| 9-D keypoint encoding | `command_adapter` (PLB, yaw/pitch/roll) | NumPy port | Matched (E: 500 poses vs Isaac torch, ≤ 2e-6) |
| References | base U(±0.6) every 3–5 s; HIGH (measured after settle) / LOW / LEFT / RIGHT every 4–5 s; min-jerk 1.5 / 2.0 s | identical, from a precomputed schedule | Matched (E) |
| Horizon | 149 HL steps | 149 HL steps | Matched (E) |
| Solver | PhysX TGS, 4 position iterations | MuJoCo Newton, implicitfast | Inherent difference |

## 4. Physical sanity checks (executed, `python3 -m transport_experiment.validate`, all PASS)

P1 frozen level disk 5 s: drift 0.03–0.5 mm, contact every step, net force = m g (±0.1%).
P2 disk ≡ gripperStator rigid attachment along a dynamic episode (err 1.2e-7).
P3 under the WBC a 45° roll tilt makes a μ = 0.3 puck slide off and fall below the disk; drop raised exactly on the 5th consecutive no-contact step.
P4 a 10-cm lift-off raises the counter to 1 and resets it on landing (no drop); a puck on the ground has no disk/stator contact and is dropped after 5 steps.
P5 frozen disk at 12° / 25°: sticks iff tan θ < μ for μ ∈ {0.3, 0.6}, m ∈ {0.5, 1.5} (Coulomb, mass-independent).
P6 holding torque on joint2 under the WBC rises 10.1 → 14.5 N·m from 0.5 to 1.5 kg.

## 5. Unresolved differences and their consequence (read before citing numbers)

1. **Frozen-WBC EE-orientation tracking in MuJoCo.** With zero residual, holding the measured HIGH pose, the frozen WBC
   rotates the wrist and the disk tilts to **20–30° within 1–3 s** (D1 in `PHYSICAL_VALIDATION.json`). Every available WBC
   checkpoint behaves this way (13–33° gripper-x error at HIGH), and E1 already reports EE-orientation RMSE of 15–17° for the
   E1–E3 WBC in MuJoCo. This is a property of the frozen controller in this simulator, not of the transport code (the engine
   is bit-exact to the deployment). Whether the WBC tilts the disk equally in Isaac Lab could not be checked here (no Isaac
   Sim rollout); the much higher retention reached in training is consistent with a smaller tilt there, but
   this is an inference, not a measurement.
2. **Consequence: absolute retention is far below training.** Final training statistics (stochastic policy, noisy obs) report
   puck-drop terminations of 0.43 % (Full), 7.9 % (Ori), 28.6 % (Pos). In MuJoCo the Full policy retains 60 %, and failures are
   concentrated in low-friction pucks (Full: 7.8 % retention for μ < 0.54 vs 98.8 % for μ > 0.78). With the Full policy the
   disk still runs at 11–20° tilt (p90), i.e. close to atan(μ) for μ ≈ 0.3–0.5. The **ordering** of the methods matches
   training (Full > Ori > Pos > none) but the **absolute** retention values are MuJoCo-specific and should not be presented
   as equivalent to Isaac Lab.
3. **Single friction coefficient.** MuJoCo uses one μ for stick and slip; we use the Isaac Lab static coefficient
   (U(0.3, 1.0)). PhysX additionally slides with μ_d ≤ μ_s (mean 0.53 vs 0.65), so MuJoCo is, if anything, *more*
   forgiving once slip starts; residual soft-contact creep (≤ 0.5 mm/s in the frozen-disk tests) acts the other way.
4. **Disk-only contact detector.** Isaac Lab counts contact with any gripperStator shape. All other stator/jaw collision
   geoms lie entirely below the disk's lower face, so they can only be touched after the puck has slid off; the engine logs
   such stator-only contact, and it occurred in 0 of 2000 official episodes at an HL check.
5. Not reproduced: wheel/ground material randomization, observation noise, DCMotor torque–speed curve for the legs.
