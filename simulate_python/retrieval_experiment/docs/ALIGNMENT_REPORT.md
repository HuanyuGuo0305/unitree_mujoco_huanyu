# E5 sim2sim alignment report — Isaac Lab (PhysX) vs MuJoCo, dynamic object retrieval

Revision 4 (2026-11-16): official evaluation extended to 3–7 cm (four 1-cm groups; §8). Revision 3: physically defined
One-Shot Success (§4). Revision 2: continuous size intervals and randomized density. Branch `e3_bidirectional`
(uncommitted). All sim2sim corrections of revision 1 are unchanged; no physics or solver setting changed in revisions 2–3.

**Reference (source of truth)**
- Environment: `IsaacLab_huanyu/source/isaaclab_tasks/isaaclab_tasks/direct/b2wz1_hl_retrieval_DRC_UAN_ABS/b2wz1_hl_env.py`
  (`B2WZ1HLRetrievalUANABSEnvCfg`, `B2WZ1HLRetrievalUANABSEnv`; gym id `Isaac-Unitree-B2WZ1-HL-Retrieval-UAN-ABS-Direct-v0`).
- Training run: `logs/rsl_rl/unitree_b2wz1_hl_retrieval_uan_abs/2026-09-14_13-37-59_multi_head_beta_9d_grouped_entropy`
  (`params/env.yaml`, `model_8000.pt`, `exported/`). Source `ENV_PARAMS` == the run's saved `env_params` (0 differing keys).
- Robot: `UNITREE_B2WZ1_UAN_CFG`, as saved in the run's `env.yaml`.

**MuJoCo side**: `deploy_mujoco/b2wz1_hl_retrieval_uan.py`, `configs/b2wz1_hl_retrieval_uan.yaml`,
`unitree_robots/b2wz1/b2wz1_retrieval.xml` (includes `b2wz1.xml`), benchmark engine `retrieval_experiment/simulator.py`.

Legend: **E** = verified by an executed test/measurement, **C** = code inspection. Status: Matched / Corrected / Approximated / Unsupported.

---

## 0. Evaluation protocol vs training distribution (read first)

| Quantity | Training | E5 evaluation | Relation |
|---|---|---|---|
| Cube side | half extent U(0.02, 0.03) m per env | four 1-cm groups [3,4], [4,5], [5,6], [6,7] cm, each two half-cm strata, uniform, unrounded | [4,6] inside the training range; **[3,4] and [6,7] out-of-distribution** (§8) |
| Cube density | **fixed 800 kg/m³** (`_apply_object_mass` overrides the startup `object_mass` event U(0.10, 0.30) kg) | U(400, 1200) kg/m³ per scenario | **evaluation-time ±50 % perturbation, outside training** |
| Cube mass | 800·s³ = 0.051–0.173 kg | density·s³ = 0.011–0.412 kg (3–7 cm) | lighter and heavier than any training cube |
| Inertia | parsed scaled-cube inertia × mass ratio = m s²/6 | compiler solid box m s²/6 | same law |
| Everything else | — | training distributions (§3) | matched |

Pairing: each base scenario draws `size_u` and `density_u` once and replays them in all four intervals (same density,
same relative position inside each interval), with every other random draw shared. The E5 results therefore measure
robustness to size *and* to unseen densities; they must not be described as in-distribution for density.

## 1. Checkpoints (all E, `tests/test_alignment.py`; hashes re-verified for revision 2)

| Component | File | SHA-256 | Identity |
|---|---|---|---|
| HL retrieval policy | `exported_policy/b2wz1_hl_retrieval_uan/high_policy_8.onnx` | `d2320822…57fb92d` | byte-identical to the run's `exported/policy.onnx`; weights == `model_8000.pt`; ONNX == hand rebuild (normalizer → trunk → 3/5/1 heads → softplus+1+1e-6 → Beta **mean** → 2u−1), max ǀΔǀ 1.9e-6; deterministic; 165 → 9 |
| Frozen WBC | `…/low_policy.onnx` | `3b7874b7…cde9` | byte-identical to `unitree_b2wz1_plb_wholebody_UAN_loco_manip/2026-09-07_20-02-17_…/exported/policy.onnx` (= `ll_load_run`); ONNX == `policy.pt`; 400 → 22 |
| UAN | `…/model_4000_inference.pt` | `db53ac18…c4eb` | byte-identical to `B2WZ1_GRIPPER/uan/model_4000_multipose20260907_inference.pt`; 300 → 6 |

## 2. Corrections made to the deployment (revision 1, unchanged)

| # | Discrepancy (before) | Training reference | Correction | Verified |
|---|---|---|---|---|
| 1 | Cube mass 0.20 kg in the XML | 800 kg/m³ → 0.10 kg at 5 cm | XML 0.10 kg (interactive deploy). Benchmark: mass = sampled density · s³ per episode | E (read-back, every episode) |
| 2 | Cube size fixed 5 cm | per-env size | MjSpec rebuild per sampled (side, density): collision box, mass, inertia, bounding radius, spawn z = s/2 + 1 mm | E |
| 3 | Default soft contact: the training gripper crushes 16–31 mm into the cube and loses it | PhysX rigid contacts | cube `solref = (0.008, 1)` (2·dt), `solimp = (0.99, 0.999, 0.001)`, also for the explicit pairs | E (§6) |
| 4 | MuJoCo **max** friction mixing (cube 1.0, wheel–floor 0.8 fixed) | PhysX `multiply`; ground 1.0; cube μ ~ U(1,2); robot shapes U(0.4,1.2) | priorities (cube 2 > robot 1 > floor 0) + 15 explicit cube↔gripper pairs: cube–ground μ_c, cube–jaw μ_c·μ_shape, robot–ground μ_shape | E (every contact type) |
| 5 | Soft-friction creep: held cubes walked 0.11–0.27 m out of the jaws in < 3 s | Coulomb static friction holds | `noslip_iterations = 5` in the retrieval scene | E (§5) |
| 6 | Gripper closed target −0.53 rad | 0.0 rad (open −π/2), close iff raw > 0 | YAML | E |
| 7 | Gripper `pd_position` kd 10; table 76.8 / 4.0 | DCMotor 64 / 3, 30 N·m, 2 rad/s | YAML `dc_motor`, 64 / 3 | E |
| 8 | Target radius 0.50 m, z [0.30, 0.80] | 0.80 m, z [0.30, 1.00] | YAML | E |
| 9 | Timeout 150 HL steps | 149 (`>= max_episode_length − 1`) | deploy + engine | E |
| 10 | No stage machine / success / terminations in the deploy | training methods | `metrics.TaskStateMachine` | E (vs training source, 0 mismatches) |
| 11 | No reset / inertial / friction DR, no obs noise | `_reset_idx`, `EventCfg`, `add_noise` | engine, per scenario | E/C |

Revisions 2–3 change no physical, contact or solver parameter. Revision 2 changed the cube sampling (size intervals,
density); revision 3 adds a read-only 250-Hz retention log and the physical One-Shot evaluation (§4).

## 3. Alignment table

| Component | Isaac Lab (training) | MuJoCo (E5) | Status |
|---|---|---|---|
| HL policy / WBC / UAN | see §1 | same files, deterministic | Matched (E) |
| Rates | 250 / 50 / 10 Hz | same; engine ≡ deploy bit-exact | Matched (E) |
| WBC + HL obs / actions | 80-D×5, 55-D×3 feature-major, ABS EE decode, 9-D keypoints | identical (keypoints vs Isaac `command_adapter` ≤ 1.2e-7) | Matched (E) |
| Gripper | binary, 0 / −π/2, DCMotor 64/3 | same | Corrected |
| Leg / wheel actuators | DCMotor curve / implicit velocity kd 10 | constant clip / velocity actuator kv 10 | Approximated / Matched |
| Arm, gripper joint friction | 0.01 N·m | 0 | Approximated (negligible) |
| Cube geometry | box, 4–6 cm | box, sampled side, unrounded | Matched (E) |
| Cube density / mass | 800 kg/m³ | **U(400, 1200)** (evaluation perturbation) | Deliberately different (§0) |
| Cube inertia | m s²/6 | m s²/6, read back | Matched (E) |
| Cube / robot / ground friction | μ_s, μ_d per shape, multiply | one Coulomb μ (= μ_s), product rule for robot–ground, cube–ground, cube–jaw | Approximated (single coefficient) |
| Restitution | ≈ 0 (× ground 0) | none; drop rebound 0.18 mm | Approximated |
| Contact model | PhysX TGS rigid | MuJoCo Newton, implicitfast, pyramidal, stiff cube contact, noslip 5 | Approximated (§5, §6) |
| Grasp proxy | close ∧ err < 0.10 ∧ q < −5° ∧ ǀΔqǀ < 3°, 3/3 hysteresis; not in actor obs | identical | Matched (E) |
| Sim-confirmed grasp | proxy ∧ stator AND mover net force > 0.3 N | net force cube ↔ all collision geoms of each body > 0.3 N | Matched semantics (E) |
| Stage machine, success, terminations | S0/S1/S2, 3-step success < 0.15 m, S2 lost ×2, base > 1 N, OOB > 5 m ×3, timeout 149 | identical | Matched (E) |
| Stage-2 force-close | training bootstrap, off after iter 500 | disabled, engine refuses otherwise; no weld/attachment/scripted grasp | Matched |
| Reset, base inertial DR, object spawn, retrieval target, obs noise | training distributions | identical, per scenario | Matched (E) |

## 4. One-Shot Success (revision 3: physical retention)

**Definition.** A one-shot successful retrieval is a task-successful episode in which, after the first *physical*
acquisition of the cube, the robot continuously retains it until task completion, without dropping and re-grasping it.
Task success itself is unchanged (training state machine, §3). Denominator: all trials.

**Audit of the revision-2 metric (superseded).** It (i) started at the training S1→S2 confirmation, which needs the
command-based proxy for 3 HL steps, so any acquire–drop–re-grasp before confirmation was invisible; (ii) sampled contacts
only at 10 Hz; (iii) called a drop only if both jaws were contact-free *and* the grasp error exceeded 0.10 m, so slips
staying within 0.10 m or keeping single-jaw contact were "flicker"; (iv) defined "held" through the command-dependent
proxy. It reported one-shot = task success (961/961), which was a property of the metric, not of the robot.

**Data.** The 10-Hz trajectories were insufficient (contact state unknown for 24 of 25 physics steps; no jaw or
relative velocity within an HL period). The frozen official scenario bank was re-executed once with a read-only 250-Hz
log (jaw forces, jaw angle/velocity, cube pose in the stator frame, cube height/speed, executed gripper command); every
logged column of all 1000 episodes reproduced the previous official run exactly (59 columns, 0 differences).

**Physical criteria** (`retention.py`, thresholds in `config.yaml: retention`):
- *Load-bearing pinch*: both jaw net forces >= 10 N. Supporting the heaviest cube (0.259 kg) under gravity plus 2 g at
  the lowest training jaw friction (0.4) needs <= 9.5 N per jaw; settled grasps squeeze with ~450 N (p0.1 of held
  samples 263 N). The training 0.3-N sensor threshold is not a retention criterion: in s247/G1 both jaws chattered at
  1–2 N for 64 ms while the lifted cube slid 15 mm before being re-gripped.
- *Acquisition*: load-bearing pinch for 0.1 s (one HL period) with jaw-angle range <= 1 deg (jaw stalled on the cube;
  an unobstructed jaw closes at up to 2 rad/s) and cube motion relative to the jaws <= 2 mm (rigid). Example of a
  non-acquisition: s53/G1 pinched the cube at 140–220 N for 0.3 s while the jaw kept closing at ~35 deg/s and the cube
  slid out along the jaws (squeeze-out attempt).
- *Drop (release)*: an interruption of the load-bearing pinch after acquisition lasting >= 40 ms, or with cube motion
  relative to the jaws >= side/2, or with the unblocked jaw closing >= 5 deg (7.5 mm of pad travel). Otherwise flicker.
  *Re-grasp*: the next acquisition after a drop.
- Validation on the official run: all 49 interruptions of grasps that were kept lasted <= 32 ms (<= 9.9 mm, <= 4.2 deg);
  all 35 grip losses lasted >= 56 ms; no interruption fell between 32 and 56 ms. Changing any single threshold
  (hold force 0.3–100 N, gap 24–96 ms, jaw 3–8 deg, displacement side/4–1 side, acquisition jaw range 2 deg or 5 mm)
  changes the one-shot total by at most 2 (952–955); a 0.2-s acquisition window gives 959; a strict variant that counts
  any unsettled 0.1-s pinch as an acquisition gives 945.

**Result.** One-shot 954 vs task success 961: 7 successful episodes contain a physical drop and re-grasp —
6 before the training confirmation (s110/G1, s145/G1, s153/G1, s162/G4, s186/G3, s241/G2; gaps 56 ms–2.1 s) and one
after it (s98/G4: 100 ms, 32 mm slip, jaw closed 12 deg, re-gripped). Every training grasp-loss termination (26) contains
a physical drop, every training-confirmed grasp (993) has a physical acquisition, and no gripper close→open command was
issued while the cube was held (the 4 commands after acquisition all followed a drop — re-grasp attempts, not harmless
command changes). One-shot <= task success holds per episode.

## 5. Friction creep and the no-slip solver (revision 1 finding, unchanged)

With soft friction alone, a cube held at ~300 N walked 0.11–0.27 m out of the jaws within 3 s under ~1 N of inertial load
(vibration ratcheting); Coulomb friction with μ_eff ≥ 0.4 holds it. `noslip_iterations = 5` reduces the static creep to
≤ 0.24 mm / 3 s (V1) and the in-episode walk to a few mm; it changes a 10-s open-loop robot replay by ≤ 4.3 cm over 3.6 m.
Disclosure: the issue was found while inspecting pilot failures of the first (fixed-size) protocol; a full sensitivity
run without no-slip on that protocol gave 72.5–83.5 % success vs 89.5–99.5 % with it. Those fixed-size runs were removed
when this revision replaced the protocol; the numbers are kept here as the record of the decision. No-slip 5 is held
fixed in revision 2 and was not re-tuned.

## 6. Physical validation (`python3 -m retrieval_experiment.validate`; `docs/PHYSICAL_VALIDATION.json`)

- **V1 squeeze + friction hold** (robot rigidly mounted, training gripper actuator, gravity in the jaw plane), 17 cases:
  size × density corners (lightest 4.0 cm @ 400 = 0.026 kg, heaviest 6.0 cm @ 1200 = 0.259 kg, 4.0 @ 1200, 6.0 @ 400,
  nominal 5.0 @ 800), lightest / heaviest at the training minimum (0.4) and maximum (2.4) jaw friction products, and 8
  random official samples. **All 17 establish and hold the grasp** (dual contact, angle −27.2 … −52.0°, ≈ 420–476 N per
  jaw, slip ≤ 0.24 mm in 3 s). Steady-state penetration 0.19–1.21 mm in 16 cases; peak including the closing impact
  ≤ 2.9 mm (heaviest cube at μ 0.4: an 8-ms impact transient, 0.19 mm steady).
- **Documented exceedance (not corrected)**: lightest cube (0.026 kg) at the maximum jaw friction 2.4 settles with a
  steady 5.4 mm distributed overlap (14 contacts sharing the stall squeeze, cube tilted 4° and unseated). A placement sweep
  shows the same mechanism already for the training-nominal 5 cm / 800 kg/m³ cube at μ 2.4 (1.6–3.4 mm) and 4 cm @ 800
  (2.4–3.8 mm): MuJoCo's time-constant contact form scales contact stiffness with effective mass, and high friction
  locks the cube before it seats. The grasp is still established and held. Removing it would require re-parameterising
  the validated contact stiffness, which the E5 protocol excludes; in the official run the in-episode p95 of the maximum
  penetration per grasped episode is 2.1–2.5 mm per group.
- **V2 Coulomb friction** (cube–ground, lightest / nominal / heaviest × μ 0.4, 0.7, 0.9, ±6° about atan μ): 18/18 stick
  iff tan θ < μ (mass-independent, as Coulomb requires). μ ≥ 1 cannot be probed on an incline (the cube tips first);
  the per-contact coefficient read-back in `tests/test_model.py` covers it.
- **V3 drop** (lightest / nominal / heaviest): rebound 0.18 mm.
- **Per-sample model checks** (`tests/test_model.py`): random official samples and corners — box half extents, mass =
  ρ s³ (relative error < 1e-12), inertia = m s²/6 (< 1e-9), bounding radius, unrounded side; spawn s/2 + 1 mm, rest at
  s/2 (< 0.5 mm), normal force = m g (< 2 %) for 0.026–0.259 kg; mass constant through an episode; effective friction of
  every contact type; no-slip and integrator settings.
- **In-episode (official)**: jaw normal force 442–512 N, dual-contact fraction 0.982–0.999 of Stage-2 steps, median
  cube slip in the stator frame 3.0–3.5 mm; 250-Hz retention statistics in §4.

## 7. Controller parity (`tests/test_deploy_parity.py`)

The deploy script runs on a deep copy of the engine's compiled model (5 cm, 800 kg/m³) with its own RNG task draws.
Over 451 WBC steps (grasp + 2.5 m carry) HL action, WBC EE command, arm targets/state, gripper target/state/command,
proxy, grasp error and cube position agree with zero difference.

## 8. OOD size extension, 3–7 cm (revision 4)

**Design.** The four OOD half-cm strata ([3.0,3.5), [3.5,4.0), [6.0,6.5), [6.5,7.0]) replay the 250 base scenarios of
the immutable base run (same size quantile within the stratum, same density quantile and all other draws); the 4–6 cm
strata are reused from the base run, which is SHA-256 pinned and was verified byte-identical (1019 files) after the
extension. `run.py --ood-extension` refuses overlapping strata, other scenario banks or physics overrides;
`combine.py` verifies completeness, configuration identity, policy/UAN/XML hashes, the stored One-Shot classification
of every reused episode, and the regression counts ([4,5]: 484 / 480, [5,6]: 477 / 474) before merging.

**Physical validation at the extremes** (`validate.py` V4; 3.0 / 7.0 cm × 400 / 1200 kg/m³):
- Implementation checks pass: mass (0.0108–0.4116 kg), inertia m s²/6, collision box, spawn height, rest at s/2,
  Coulomb stick/slip (μ 0.4, 0.9), drop rebound 0.18 mm, no numerical instability; every OOD episode's mass and inertia
  read back exactly.
- 3 cm cubes are grasped and held at jaw friction 0.4–2.4 (jaw stalls at −17 … −24°, so the proxy's "not fully closed"
  < −5° still applies). The documented mass-scaled contact compliance grows for the lightest cube at maximum jaw
  friction: steady overlap **7.8 mm** (3 cm @ 400 kg/m³, 10.8 g; 3.6 mm at 1200 kg/m³), vs 5.4 mm at 4 cm. Not corrected.
- 6.5–7 cm cubes need a 55–67° jaw opening. Seated deep in the jaws they are held; near the fingertips the strongly
  non-parallel jaws wedge them out or let them slide unless friction is high (7 cm @ 400 kg/m³, jaw μ 1.0: slid 77 mm /
  lost; jaw μ 0.4: ejected; jaw μ 2.4: held). This is the gripper's geometry acting on an oversized object, reproduced by
  a placement sweep, not an implementation error; it is expected to affect OOD-large retention.

**Retention detector on OOD data** (thresholds unchanged): interruptions classified as flicker last <= 36 ms (cube
<= 11.5 mm, jaw <= 4.7°), releases >= 52 ms; four gaps fall in the 32–56 ms band (two 36-ms flickers, two 52-ms releases,
one of which is also a release by the jaw criterion). One-Shot counts change by <= 3 per group under any single-threshold
variation. The 10-N hold force was justified for <= 0.259 kg; the heaviest OOD cube (0.412 kg) needs 5.0 N per jaw against
gravity at μ 0.4 (15 N with 2 g extra), far below the ~400 N squeeze, and 30 N / 100 N thresholds change the large-OOD
count by 1. Every successful episode contains a physical acquisition; all 18 success-but-not-one-shot episodes are genuine
drop + re-grasp sequences.

**Finding (training proxy vs physics).** In 9 large-cube episodes the training proxy (which tolerates 3° of jaw motion per
HL step) confirmed a grasp while the cube was continuously squeezed out of the jaws (jaw closing 7–18°, cube sliding
9–24 mm during the pinch); the cube was never physically settled and was lost. These are failures under both metrics and
are reported as physical findings, not data errors.

**Result.** [3,4]: SR 96.4 %, One-Shot 95.4 %; [4,5]: 96.8 / 96.0; [5,6]: 95.4 / 94.8; [6,7]: 82.6 / 81.4 (500 each).
Smaller OOD is statistically indistinguishable from in-distribution (Δ SR +0.3 pp [−1.8, +2.3]); larger OOD degrades
(Δ SR −13.5 pp [−17.2, −9.9], failure ratio 4.5×), mainly through post-grasp drops that increase with size and mass.

## 9. Remaining differences (not corrected)

1. Contact solver: MuJoCo soft contacts (stiffened) + no-slip vs PhysX TGS rigid contacts; the documented
   light-cube / high-friction overlap (§6). Post-grasp losses grow mildly with size in E5 (1.2 → 4.4 % of confirmed
   grasps from G1 to G4) without a significant effect on overall success; whether PhysX shows the same was not measured.
2. One Coulomb coefficient (static) instead of static + dynamic; robot–robot and cube–non-gripper-robot contacts do not
   use the product rule.
3. Leg DCMotor torque–speed envelope and 0.01 N·m arm/gripper joint friction not modelled (as in E1–E4).
4. Contact flags are evaluated at the HL boundary state (Isaac: last physics substep of the step) — the same instant.
5. Absolute success rates are MuJoCo-specific and should not be presented as Isaac Lab performance.

Note for E4 (not modified): the robot–ground coefficient in all `b2wz1` scenes is max(0.4 wheel, 0.8 floor) = 0.8, not 1.0
as stated in `transport_experiment/docs/ALIGNMENT_REPORT.md`.
