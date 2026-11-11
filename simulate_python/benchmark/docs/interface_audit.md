# B2W+Z1 WBC Baseline Interface Audit (Phase A)

Audit of the six validated MuJoCo Sim2Sim controllers, performed before any
benchmark code was written. Every statement below is traced to a file and line
in this repository. Where the training code is not present in this repository
the finding is marked **[inferred-from-deployment]** and the evidence is given.

Audited sources:

| Baseline | Deployment script | Config |
|---|---|---|
| C1 | `simulate_python/deploy_mujoco/b2wz1_wbc_plb_uan.py` | `configs/b2wz1_wbc_plb_uan.yaml` |
| C2 | `simulate_python/deploy_mujoco/b2wz1_wbc_plb_legged_uan.py` | `configs/b2wz1_wbc_plb_legged_uan.yaml` |
| C3 | `simulate_python/deploy_mujoco/b2wz1_vbc_plb_uan.py` | `configs/b2wz1_vbc_plb_uan.yaml` |
| C4 | `simulate_python/deploy_mujoco/b2wz1_vbc_plb_legged_uan.py` | `configs/b2wz1_vbc_plb_legged_uan.yaml` |
| C5 | `simulate_python/deploy_mujoco/b2wz1_roboduet_plb_uan.py` | `configs/b2wz1_roboduet_plb_uan.yaml` |
| C6 | `simulate_python/deploy_mujoco/b2wz1_roboduet_plb_legged_uan.py` | `configs/b2wz1_roboduet_plb_legged_uan.yaml` |

Supporting assets: `unitree_robots/b2wz1/{b2wz1.xml, ground.xml, b2wz1_legged.xml}`,
`utilities/math.py`, `utilities/mujoco_helper.py`,
`utilities/data/keypoints_plb_evaluation_filtered.npy`,
`exported_policy/*`, and the two hierarchical configs
`configs/b2wz1_hl_{transport,retrieval}_uan.yaml` (only used as evidence for the
frozen-WBC command envelope).

---

## 1. Headline findings that constrain the benchmark design

1. **The EE command is *not* in the base frame.** It is a 9-D keypoint triplet
   expressed in the **PLB (Projected Level Base) frame**. See §4.
2. **The EE command is not a position+quaternion.** It is three 3-D points
   `(kp0, kp1, kp2)` with `kp1 = kp0 + R_ee·[0.30,0,0]` and
   `kp2 = kp0 + R_ee·[0,0,0.30]`. Position *and* orientation are both encoded.
   Verified numerically on the command table (§4.3).
3. **All six baselines share one EE command interface**: the same 9-D PLB
   keypoint vector, the same `gripperStator` EE body, the same
   `ee_kp_dx = ee_kp_dz = 0.30`, the same `ee_ground_z = 0.0`, and the same
   presampled command table. This means a *single* benchmark EE target
   representation is natively correct for every controller — no per-baseline
   conversion or approximation is needed.
4. **The native EE command behaviour is already sample-and-hold with no
   interpolation** (`ee_command_mode: direct`, §4.5). The E1 requirement
   "no smoothing / no interpolation, hard target switch" therefore *is* the
   native semantics; nothing has to be added or removed.
5. **The base velocity command is a constant 3-vector read from YAML**
   (`base_command: [0.5, 0.0, 0.0]`) in every one of the six scripts. There is
   no time-varying base command generator in any deployment script, so the
   benchmark must inject one. This is the one genuinely new command path.
6. **Observation histories are term-major, not step-major**, in all six
   scripts (each observation term keeps its own 5-frame deque and the stacked
   vector concatenates per-term histories). Any benchmark hook must append to
   every term exactly once per policy tick.
7. **No script exposes a reusable controller object.** C1 and C2 are
   `if __name__ == "__main__"` monoliths; C3–C6 have a `main()` whose entire
   control pipeline lives in closures over local variables. See §12 for how the
   benchmark handles this and how behavioural equivalence is verified.

---

## 2. Timing and control frequency

Identical for all six baselines, asserted at startup by each script:

| Quantity | Value | Evidence |
|---|---|---|
| MuJoCo timestep | `0.004 s` (250 Hz) | `simulation_dt: 0.004` in all 6 YAMLs |
| Policy decimation | `5` | `control_decimation: 5` |
| Policy rate | `50 Hz` | `control_dt = 0.004 * 5 = 0.02 s` |
| Frozen-UAN rate | `250 Hz` (every physics step) | C1 L826 `compute_uan_arm_torque` called in the physics loop |
| Observation history | 5 frames at 50 Hz | `history_length: 5` |
| UAN history | 25 frames at 250 Hz (100 ms) | `uan_history_length: 25` |

C1 hard-fails if `simulation_dt != 0.004` or `control_decimation != 5`
(`b2wz1_wbc_plb_uan.py:347-352`); C5/C6 do the same
(`b2wz1_roboduet_plb_uan.py:389-390`). The benchmark therefore uses exactly
these values and never overrides them.

### 2.1 Exact loop order (preserved by the benchmark runner)

C1/C2 (`b2wz1_wbc_plb_uan.py:908-1050`):

```
loop over physics steps, counter = 0,1,2,...
  1. blend  <- f(sim_time)                      # BEFORE the step
  2. leg_tau <- PD(leg_target); arm_tau <- UAN(arm_target); grip_tau; wheel_ctrl
  3. write d.ctrl; mujoco.mj_step(m, d); sim_time += dt
  4. if counter % 5 == 0:
       ee_cmd <- sampler.update()
       obs    <- build_obs_step(ee_cmd); append to 10 term deques
       action <- onnx(obs_stack)
       last_action <- action
       leg_target/arm_target/wheel_cmd <- default + blend * scale * action
  5. counter += 1
```

C3/C4 (`b2wz1_vbc_plb_uan.py:925-1000`) and C5/C6
(`b2wz1_roboduet_plb_uan.py:600-680`) use the same structure with one
difference: `blend` is recomputed **inside** the policy tick from the
already-incremented `sim_time`, so C1/C2 and C3–C6 differ by one physics step in
the blend ramp. The benchmark reproduces each script's own convention
(`BENCH_BLEND_BEFORE_STEP` in the adapters) rather than unifying them.

C5/C6 inference order, which must not be reordered
(`b2wz1_roboduet_plb_uan.py:626-637`):

```
ee_sampler.step()
arm_terms, loco_terms = frame_pair()     # ONE physical state read, before new guidance
arm_hist.append(arm_terms); loco_hist.append(loco_terms)
arm_raw      = ARM_onnx(arm_hist.stack())        # 330 -> 8
new_guidance = posture.inject(arm_raw)           # tanh/clip/slew at 50 Hz
loco_obs     = patch_loco_posture_history(loco_hist.stack(), new_guidance)
loco_raw     = LOCO_onnx(loco_obs)               # 335 -> 16
```

`patch_loco_posture_history` shifts **only** the `base_posture_commands` term
history by one frame and writes the just-computed guidance into the newest slot
(`out[45:53] = src[47:55]; out[53:55] = g`). The posture deque itself is *not*
shifted twice — the patch operates on a copy of the stacked vector. The
benchmark adapter calls `posture.inject()` exactly once per policy tick.

---

## 3. Base velocity command

### 3.1 Variables, order, units

`base_command` is a 3-vector `[vx, vy, yaw_rate]`, in `m/s, m/s, rad/s`.

* C1: `base_command = np.array(cfg["base_command"])`, placed in the observation
  at offset 6..9 (`b2wz1_wbc_plb_uan.py:842`, term `base_command`, dim 3).
* C3: observation term is literally named `velocity_commands`, dim 3
  (`b2wz1_vbc_plb_uan.py:886-893`).
* C5/C6: `velocity_commands`, dim 3, present in **both** the ARM and the LOCO
  observation groups (`b2wz1_roboduet_plb_uan.py:57-68`). Injecting a base
  command therefore changes both policies' inputs, which is correct and
  intended.

### 3.2 Sampling in the native scripts

**There is none.** All six scripts read one constant from YAML and never change
it. `base_command: [0.5, 0.0, 0.0]` in all six configs.

Consequence for the benchmark: injecting a *time-varying* `base_command` is a
new capability. Because the command enters only through the observation term
(and, for C5/C6, through two observation groups), injecting it is a pure data
substitution and does not alter any inference or actuation code path.

### 3.3 Reference frame **[inferred-from-deployment]**

The deployment scripts never compare the command to a measured velocity, so the
frame is not directly observable in this repository. Evidence for the frame:

* The observation group contains `base_ang_vel` from the `imu_gyro` sensor,
  which MuJoCo reports in the **IMU site (base) frame**, and
  `projected_gravity` computed as `R_base^T · [0,0,-1]` — i.e. the whole
  observation group is expressed in the base frame.
* There is no root linear-velocity observation term at all in any of the six
  baselines, which is the standard Isaac Lab `velocity_env_cfg` layout where
  the command is `base_lin_vel_b` / `base_ang_vel_b[2]` and the reward
  (`track_lin_vel_xy_exp`, `track_ang_vel_z_exp`) compares against
  `root_lin_vel_b[:, :2]` and `root_ang_vel_b[:, 2]`.

**Benchmark decision.** The primary base tracking error is computed in the
**base body frame**: `v_cmd` is compared against
`v_b = R_base^T · v_world` (xy components) and `ω_cmd` against
`ω_b,z = R_base^T · ω_world |_z` (identical to the `imu_gyro` z reading the
policies already observe). Because on flat ground roll/pitch are small, the
**yaw-only (PLB) frame** projection is also logged for every episode
(`v_xy_plb`, `wz_plb` columns in `states.csv`) and the metrics module can
recompute all E1/E2/E3 base metrics in that frame via
`metrics.base_frame: "yaw"`. The two differ by <2% of RMSE on flat ground in
the smoke runs; the choice is reported in every `metadata.json`.

### 3.4 Supported command range **[inferred-from-deployment]**

The training ranges are not in this repository. The strongest available
evidence is the hierarchical transport config, whose high-level policy emits a
base-velocity reference *for the frozen C1 WBC policy* and which validates its
own sampling range against what it calls the frozen-WBC bounds
(`b2wz1_hl_transport_uan.py:286-290`):

```python
if np.any(base_ref_ranges[:, 0] < -base_cmd_scale) or np.any(
        base_ref_ranges[:, 1] >  base_cmd_scale):
    raise ValueError("base_ref_ranges must stay inside frozen-WBC base bounds.")
```

with `configs/b2wz1_hl_transport_uan.yaml`:

```yaml
base_cmd_scale:   [0.6, 0.6, 0.6]
base_ref_ranges:  [[-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6]]
```

`configs/b2wz1_hl_retrieval_uan.yaml` uses the more conservative
`base_cmd_scale: [0.5, 0.5, 0.5]` for its own high-level action scaling.

**Benchmark decision.** The shared supported command range used for all of
C1–C6 is

```
vx  in [-0.6, +0.6]  m/s
vy  in [-0.6, +0.6]  m/s
wz  in [-0.6, +0.6]  rad/s
```

declared once in `benchmark/configs/baselines.yaml` as
`base_command_limits` and reused by E1, E2 and E3. It is **not** adjusted per
baseline — C1/C2/C3/C4/C5/C6 all receive the identical range, which is also the
explicit requirement for the C1-vs-C2 comparison in E2. If the user can supply
the Isaac Lab `UniformVelocityCommandCfg.ranges` from the training runs, change
only that one YAML key and every experiment inherits it.

---

## 4. EE pose command — PLB frame

### 4.1 PLB frame definition

Identical in all six scripts. C1 `b2wz1_wbc_plb_uan.py:600-612`:

```python
def get_projected_level_base_pose_world():
    base_pos_w  = d.qpos[0:3]
    base_quat_w = quat_unique_wxyz(d.qpos[3:7])
    _, _, yaw   = euler_xyz_from_quat_wxyz(base_quat_w)
    plb_quat_w  = quat_from_yaw_wxyz(yaw)          # yaw-only, roll=pitch=0
    plb_pos_w   = [base_pos_w[0], base_pos_w[1], ee_ground_z]   # z replaced!
    return plb_pos_w, plb_quat_w
```

So:

* **origin** = `[base_x, base_y, ee_ground_z]` with `ee_ground_z = 0.0`, i.e.
  the base's horizontal position dropped to the ground plane.
* **orientation** = yaw-only rotation of the base (roll and pitch removed).

The PLB frame is therefore *not* the base frame: it tracks base x, y and yaw but
ignores base z, roll and pitch. EE commands given in PLB are height-referenced
to the ground, not to the (varying) torso height.

C3 recomputes the same frame as an explicit matrix when building the IK world
goal (`b2wz1_vbc_plb_uan.py:834-839`); C5/C6 use `yaw_rotation(base_quat)` plus
`ground_z` (`b2wz1_roboduet_plb_uan.py:111-142`). All three are the same frame.

### 4.2 Measuring the actual EE pose in PLB

C1 `b2wz1_wbc_plb_uan.py:614-630`:

```python
ee_pos_w   = d.xpos[ee_bid]                  # body "gripperStator"
ee_quat_w  = quat_from_rotmat_wxyz(d.xmat[ee_bid])
ee_pos_plb  = quat_apply_inverse_wxyz(plb_quat_w, ee_pos_w - plb_pos_w)
ee_quat_plb = quat_unique(quat_mul(conj(plb_quat_w), ee_quat_w))
kp0 = ee_pos_plb
kp1 = ee_pos_plb + R(ee_quat_plb) @ [0.30, 0, 0]
kp2 = ee_pos_plb + R(ee_quat_plb) @ [0, 0, 0.30]
```

`ee_body: "gripperStator"`, `ee_kp_dx: 0.30`, `ee_kp_dz: 0.30` in all six
configs.

### 4.3 Command representation, verified numerically

`utilities/data/keypoints_plb_evaluation_filtered.npy` is `(20000, 9) float32`.
Measured on the whole table:

```
|kp1 - kp0| in [0.29999986, 0.30000013]      (== ee_kp_dx)
|kp2 - kp0| in [0.29999980, 0.30000020]      (== ee_kp_dz)
max |(kp1-kp0)·(kp2-kp0)| = 6.7e-08          (exactly orthogonal)
```

So the 9-D command is exactly the forward map of §4.2 and is losslessly
invertible to a 6-D pose:

```
p_cmd = kp0
x_hat = (kp1 - kp0)/0.30 ,  z_raw = (kp2 - kp0)/0.30
y_hat = normalize(z_raw × x_hat) ,  z_hat = x_hat × y_hat
R_cmd = [x_hat  y_hat  z_hat]        (column-major, PLB frame)
```

This is exactly `_rotation_from_keypoints` in C3
(`b2wz1_vbc_plb_uan.py:241-253`) and `utilities/math.quat_from_keypoints_lb`.
The benchmark reuses `_rotation_from_keypoints`' convention verbatim
(`benchmark/core/frames.py:keypoints_to_pose`) and a unit test asserts the
round trip `pose -> keypoints -> pose` to 1e-6.

### 4.4 Workspace / feasible command region

Measured over the filtered table (PLB metres):

| | 1% | 50% | 99% | min | max |
|---|---|---|---|---|---|
| kp0 x | 0.042 | 0.610 | 0.885 | 0.000 | 0.968 |
| kp0 y | −0.594 | −0.005 | 0.588 | −0.728 | 0.728 |
| kp0 z | 0.199 | 0.810 | 1.273 | 0.115 | 1.300 |

The table is the *native* reachable-command distribution, so the benchmark
samples E1 targets **directly from this table** rather than from an invented
box. That guarantees (a) feasibility, (b) identical semantics to training and
Sim2Sim, (c) a common command region for C1/C3/C5 by construction.

For E3, continuous trajectories are required, so the benchmark instead uses an
analytic path and *verifies* feasibility against the table: a `cKDTree` over the
table's `kp0` confirms the chosen centre `[0.60, 0.00, 0.75]` with path
amplitude `0.20 m` encloses 1756 native command rows within 0.20 m, i.e. the
whole path lies deep inside the native command support
(`benchmark/tests/test_e3_trajectories.py::test_paths_inside_native_support`).

### 4.5 Native sampling behaviour — sample and hold, no interpolation

`ee_command_mode: "direct"` is the setting used by C1–C4
(`configs/b2wz1_*_plb*_uan.yaml`), and C5/C6 have only the direct sampler.

C1 `PresampledKeypointsDirectCommandPLBSim` (`b2wz1_wbc_plb_uan.py:152-224`),
docstring verbatim:

```
1) Randomly sample one row from a presampled reachable PLB keypoint table
2) Hold this sampled keypoint command until next resampling
3) No cubic interpolation
4) No adjacent-target threshold clipping
```

C5 `PresampledDirectPLBCommand` docstring: *"C5 S2 'sample_mode=random': one 9D
row, hold 8 s; no trajectory interpolation."*

Hold period: `ee_cycle_duration_s: 8.0` in all six configs, i.e. 400 policy
ticks. The off-by-one of the native counter is:

```python
def update(self):
    self._step_in_cycle += 1
    if self._step_in_cycle >= self._cycle_steps:   # cycle_steps = round(8.0/0.02) = 400
        self._sample_new_command()                 #   resets counter to 0
    return self.command
```

so with `update()` first called at policy tick `k = 0`, the command index is
`floor((k + 1) / cycle_steps)`: the first target holds for 399 ticks and every
later target for exactly 400 ticks. The benchmark's E1 generator reproduces this
indexing exactly with `cycle_steps = round(5.0/0.02) = 250`, so switch-to-switch
spacing is exactly 250 ticks = 5.000 s
(`benchmark/tests/test_scenarios.py::test_e1_ee_switch_interval_exactly_5s`).

A `SequentialKeypointsTrajectoryCommandPLBSim` with cubic interpolation also
exists in C1–C4 but is **not** the validated evaluation mode
(`ee_command_mode: "direct"`). The benchmark never uses it, so E1 contains no
external EE interpolation of any kind — matching both the requirement and the
native behaviour.

### 4.6 Which frame the benchmark reports EE error in

The command lives in PLB and the PLB frame moves with the robot, so the *same*
9-D command vector is the *same control objective* for every controller, but
corresponds to a different world point once the bases diverge.

**Benchmark decision.**

* **Primary E1/E3 EE metrics are computed in the PLB frame** — the native
  command frame. `E_p = RMSE(kp0_actual_plb − kp0_cmd_plb)` and
  `E_ori = MAE(geodesic(R_actual_plb, R_cmd_plb))`. This is the quantity every
  controller was trained/designed to minimise, and it is frame-consistent and
  identical in definition across baselines.
* The world-frame EE pose and the world-frame command (obtained by pushing the
  PLB command through the *measured* base pose at the same instant) are logged
  for every sample so world-frame errors can be recomputed offline.
* **E4 inverts the direction**: a world-frame target is fixed and converted to
  PLB each tick, and the reported error is world-frame. §9.

---

## 5. Base and EE actual-state measurement APIs

| Signal | Source | Notes |
|---|---|---|
| base position (world) | `d.qpos[0:3]` | free joint |
| base orientation (world) | `d.qpos[3:7]`, wxyz | `quat_unique_wxyz` applied by the scripts |
| base lin. vel (world) | `d.qvel[0:3]` | free joint; also `framelinvel` sensor `frame_vel` at the `imu` site |
| base ang. vel (base frame) | `get_sensor_slice(m,d,"imu_gyro")` | this is what the policies observe |
| base roll/pitch/yaw | `euler_xyz_from_quat_wxyz(quat_unique_wxyz(d.qpos[3:7]))` | used by every script's console log |
| EE position (world) | `d.xpos[ee_bid]`, `ee_bid = body "gripperStator"` | |
| EE orientation (world) | `d.xmat[ee_bid].reshape(3,3)` | |
| joint position | `d.qpos[m.jnt_qposadr[jid]]` | |
| joint velocity | `d.qvel[m.jnt_dofadr[jid]]` | |
| joint acceleration | `d.qacc[m.jnt_dofadr[jid]]` | available, not used by any script |
| applied actuator force | `d.actuator_force[act_id]` | available, not used by any script |
| contacts | `d.ncon`, `d.contact[...]`, `d.cfrc_ext` | available; no script reads them |

Sensors present in `b2wz1.xml:464-542` and `b2wz1_legged.xml:407-473`:
`jointpos`/`jointvel`/`jointactuatorfrc` per joint, `imu_quat` (framequat),
`imu_gyro` (gyro), `imu_acc` (accelerometer), `frame_pos` (framepos),
`frame_vel` (framelinvel). All at the `imu` site.

### 5.1 Available torque and joint-velocity signals (for E2)

`d.actuator_force` gives the **actuator-produced** scalar force for every
actuator, which is exactly what E2's power proxy requires (not the raw policy
action, and not the un-clipped PD torque).

Actuator definitions (`b2wz1.xml:464-490`):

```xml
<motor    class="b2" ctrlrange="-200 200" name="FR_hip"  joint="FR_hip_joint"/>   ... hips/thighs 200 Nm
<motor    class="b2" ctrlrange="-320 320" name="FR_calf" joint="FR_calf_joint"/>  ... calves   320 Nm
<velocity name="FR_wheel" joint="FR_wheel_joint" kv="10" ctrlrange="-50 50" forcerange="-20 20"/>
<motor    class="z1" ctrlrange="-30 30"   name="motor1"  joint="joint1"/>  ... arm, widened at runtime, see §7
```

The four wheels are **velocity servos** (`kv=10`, `forcerange ±20 Nm`), so the
wheel control signal is a velocity setpoint, not a torque; the realised wheel
torque is only available through `d.actuator_force`. This is called out
explicitly in C5 (`b2wz1_roboduet_plb_uan.py:...`: *"C1 `<velocity>` uses
velocity setpoint, NOT torque"*). The benchmark's wheel power therefore uses
`|d.actuator_force[wheel_act] * d.qvel[wheel_dof]|`, which is a true mechanical
power, and C2/C4/C6 have no wheel actuators at all so their `wheel_power` is
identically 0 (logged as 0.0, not as missing).

`b2wz1_legged.xml` is the same robot with the four wheel **joints and
actuators removed** (the wheel links are welded to the calves,
`b2wz1_legged.xml:92-97`): 19 actuators, 19 joints, no wheel DOFs.

---

## 6. Observation construction and normalization

### 6.1 Per-step observation layout

| Term | C1 | C2 | C3 | C4 | C5 ARM | C5 LOCO |
|---|---|---|---|---|---|---|
| `base_ang_vel` | 3 | 3 | 3 | 3 | 3 | 3 |
| `projected_gravity` | 3 | 3 | 3 | 3 | 3 | 3 |
| `velocity_commands` | 3 | 3 | 3 | 3 | 3 | 3 |
| `base_posture_commands` | – | – | – | – | – | 2 |
| `ee_kp_commands` | 9 | 9 | 9 | 9 | 9 | – |
| `joint_pos_leg` | 12 | 12 | 12 | 12 | 12 | 12 |
| `joint_pos_arm` | 6 | 6 | 6 | 6 | 6 | 6 |
| `joint_vel_leg` | 12 | 12 | 12 | 12 | 12 | 12 |
| `joint_vel_arm` | 6 | 6 | 6 | 6 | 6 | 6 |
| `joint_vel_wheel` | 4 | – | 4 | – | 4 | 4 |
| `actions` / `last_arm_joint_action` | 22 | 18 | 16 | 12 | 6 | 16 |
| `previous_guidance` | – | – | – | – | 2 | – |
| **per step** | **80** | **72** | **74** | **66** | **66** | **67** |
| **× history 5** | **400** | **360** | **370** | **330** | **330** | **335** |

C6: ARM `62 × 5 = 310`, LOCO `59 × 5 = 295`.

Asserted in code: C1 `obs_dim_per_step == 80` (L339), C5
`assert sum(d for _,d in LOCO_TERMS) == 67` and `== 66` for ARM
(`b2wz1_roboduet_plb_uan.py:69-70`).

### 6.2 History construction — term-major

All six scripts keep one `deque(maxlen=5)` per term and stack
`concat([hist[term].reshape(-1) for term in terms])`, oldest→newest inside each
term. C1 uses ten explicit deques (`b2wz1_wbc_plb_uan.py:884-905`); C3 uses a
list of deques with `term_names`/`term_dims` (L883-900); C5/C6 use the
`TermMajorHistory` class (L164-193) which raises unless `length == 5`.

At reset, every history is pre-filled with 5 copies of the first observation
(C1 L895-905, C3 L899-902, C5 `TermMajorHistory.seed`).

### 6.3 Normalization

**None in the deployment scripts.** The observation normalizer is baked into the
exported ONNX / TorchScript graphs. Evidence:

* C3 config: *"must be exported to ONNX with the trained actor observation
  normalizer included"*.
* C5 config: *"Both ONNX files contain their trained actor observation
  normalizers."*
* C1 config and script: *"The exported UAN wrapper already contains the actor
  observation normalizer and deterministic Beta-mean inference path."*

Consequence: **the benchmark must inject commands before the observation vector
is assembled** — which is exactly where it does inject them (into the
`velocity_commands` / `ee_kp_commands` term values). Nothing is touched after
the stack is built and handed to ONNX.

### 6.4 Action scaling and `last_action` semantics

The action stored in the observation is the **raw, unscaled** policy output in
every script (C3 comment: *"observation is unscaled raw PPO action"*). Target
construction:

| | C1 | C2 | C3/C4 | C5/C6 |
|---|---|---|---|---|
| legs | `q0 + blend·0.25·a[0:12]` | same | same, `a[0:12]` | `q0 + blend·0.25·loco[0:12]` |
| arm | `q0 + blend·0.10·a[12:18]` | same | **stateful DLS IK**, no RL arm action | `q0 + blend·0.10·arm[0:6]` |
| wheels | `blend·4.0·a[18:22]` | n/a | `blend·4.0·a[12:16]` | `blend·4.0·loco[12:16]` |
| gripper | PD to `default_joint_pos` | same | same | same |

`leg_action_scale: 0.25`, `arm_action_scale: [0.1]*6` (C1/C2) or `0.1` (C5/C6),
`wheel_action_scale: 4.0` (rad/s per unit action).

Startup: `blend = 0` for `t < startup_hold_s = 1.0`, then linear to 1 over
`startup_blend_s = 2.0`, i.e. **full policy authority only from t = 3.0 s**.
This drives the benchmark's settling-window choice (§10.2).

---

## 7. Frozen UAN arm actuator

Identical law in all six scripts (C1 `b2wz1_wbc_plb_uan.py:782-821`, C3
L799-822, C5 `frozen_uan_torque`), run **once per 250 Hz physics step**:

```
pos_err = (q_des - q) * uan_pos_scale            # 1.0
vel     = qd * uan_vel_scale                     # 0.05
history <- shift left; history[:, -1, :] = (pos_err, vel)      # (6, 25, 2) joint-major
action  = uan_torchscript(history.reshape(1, 300))             # Beta-mean in [-1, 1]
tau_pd  = Kp*(q_des - q) - Kd*qd
tau_nom = clip(tau_pd, +-uan_nominal_torque_limits)
dtau    = action * uan_action_scale * Kp         # action_scale = 0.20
tau     = clip(tau_nom + dtau, +-uan_final_torque_limits)
```

with, in every config,

```yaml
uan_model_path: exported_policy/b2wz1_wbc_plb_uan/model_4000_inference.pt
uan_history_length: 25
uan_kp: [64.0, 115.2, 64.0, 64.0, 64.0, 64.0]
uan_kd: [3.0, 4.0, 3.0, 3.0, 3.0, 3.0]
uan_nominal_torque_limits: [30.0, 60.0, 30.0, 30.0, 30.0, 30.0]
uan_final_torque_limits:   [45.0, 90.0, 45.0, 45.0, 45.0, 45.0]
```

All six baselines load the **same** UAN checkpoint file. History is zero-filled
at reset (not pre-filled with the current state) — C1 comment: *"Reset to zero
rather than pre-filling; this preserves the frozen model semantics."* The
benchmark does the same.

`sync_mujoco_arm_limits_to_uan: true` in every config: at startup each script
widens the MuJoCo arm `actuator_ctrlrange`, `actuator_forcerange` and
`jnt_actfrcrange` from the XML's legacy ±30/±60 Nm up to the UAN final envelope
(±45/±90), so MuJoCo cannot clip inside the explicit UAN clamp
(C1 L500-581). The benchmark performs the identical model mutation, with the
same post-condition assertion.

Guards preserved by the benchmark: non-finite UAN action → raise;
`max|action| > 1.001` → raise (C1 L808-812, C3 L813).

---

## 8. Baseline-specific pipelines

### 8.1 C3/C4 — stateful arm DLS IK

`update_stateful_arm_ik` (`b2wz1_vbc_plb_uan.py:827-879`), called once per
50 Hz tick, owns the arm target entirely (no RL arm action):

1. PLB command → world goal: `p_w = plb_pos_w + R_plb·kp0`,
   `R_w = R_plb · rotation_from_keypoints(cmd)`.
2. `pos_error_w = p_w − d.xpos[ee_bid]`;
   `rot_error_w = axis_angle(q_target ⊗ conj(q_current))`.
3. `task_error = [1.0·pos_err, 1.0·rot_err]` (`position_gain`,
   `orientation_gain` = 1.0).
4. `mujoco.mj_jac(m, d, jacp, jacr, actual_pos_w, ee_bid)` at the **body
   origin in world axes**; `J = vstack(jacp[:, arm_vidx], jacr[:, arm_vidx])`
   (6×6).
5. DLS: `dq = Jᵀ(JJᵀ + λ²I)⁻¹ e`, `λ = damping = 0.05`;
   clip `max|dq| ≤ max_joint_delta = 0.10`.
6. **Stateful part** (this is what must not be broken):
   ```python
   if ik_just_reset:                # exactly one tick after reset
       ik_reference = q_current; ik_just_reset = False
   else:
       q_cand = clip(ik_reference + 0.20*dq, ik_lower, ik_upper)
       lead   = clip(q_cand - q_current, -0.50, +0.50)
       ik_reference = q_current + lead
   return ik_reference
   ```
   `reference_gain = 0.20`, `max_reference_lead = 0.50`. `ik_reference`
   persists across ticks — calling the IK twice per tick, or calling it at
   250 Hz, would change the controller. The benchmark calls it exactly once per
   policy tick, after the ONNX inference, matching L996.
7. Soft limits: `mid ± 0.9/2·range ∓ 0.01` from `arm_hard_joint_limits`
   (L414-422). These come from the training URDF, not the MJCF.

### 8.2 C5/C6 — cooperative dual-PPO with posture guidance

`PostureGuidance` (`b2wz1_roboduet_plb_uan.py:195-221`):

```
target  = [tanh(a[6])·radians(25 if >=0 else 5),  tanh(a[7])·radians(3)]
bounded = clip(target, deg2rad([-5,-3]), deg2rad([25,3]))
command += clip(bounded - command, ±slew_rate·dt)      # slew_rate = 0.35 rad/s, dt = 0.02
```

`guidance_rate_rad_s: 0.35` in both configs. The rate limiter is **stateful**;
`inject()` must be called exactly once per policy tick. The benchmark adapter
asserts a single call per tick via an internal tick counter.

ARM policy outputs 8 values: `[0:6]` are arm joint offsets (scaled by 0.1),
`[6:8]` are the raw guidance logits. `last_arm_joint_action` in the ARM
observation stores only `arm_raw[:6]`.

---

## 9. E4 — world-frame target conversion

No existing script holds a world-frame EE target, but the conversion is exactly
the inverse of §4.1/§4.2 and uses only already-measured quantities:

```
plb_pos_w, plb_quat_w = get_projected_level_base_pose_world()   # from d.qpos
kp0_plb = R_plbᵀ (p_target_w - plb_pos_w)
R_plb_cmd = R_plbᵀ R_target_w
cmd = pose_to_keypoints(kp0_plb, R_plb_cmd, dx=0.30, dz=0.30)
```

This is a *command-frame* transformation applied to the benchmark's own target
before it reaches the controller, i.e. it uses the same injection point as every
other experiment. No extra feedback controller, no modification of the policy
output. The benchmark asserts `max|cmd| ` stays inside the native table's
command support each tick and records a `target_left_support` flag if it does
not, so an unreachable excursion is reported rather than hidden.

---

## 10. Benchmark-level decisions forced by this audit

### 10.1 Shared-vs-different simulation settings

Same for all baselines: `simulation_dt`, `control_decimation`, history lengths,
UAN checkpoint and law, EE body, PLB definition, `ee_kp_dx/dz`, `ee_ground_z`,
`default_joint_pos` (per-joint values are identical; only the array *ordering*
differs between configs), leg/arm PD gains, leg torque limits, gripper limit,
ground friction `0.8 0.2 0.1` and `condim=3` (identical in `ground.xml:26` and
`b2wz1_legged.xml:130`), `root_pos: [0,0,0.6017]`, `root_quat: [1,0,0,0]`.

Necessarily different:

* **Model**: C1/C3/C5 use `unitree_robots/b2wz1/ground.xml` (23 joints, 4 wheel
  DOFs, 23 actuators). C2/C4/C6 use `unitree_robots/b2wz1/b2wz1_legged.xml`
  (19 joints, no wheel DOFs, 19 actuators).
* **Action/observation dimensions** (§6.1) and the absence of
  `joint_vel_wheel` for the fixed-wheel variants.
* **Wheel control**: C1/C3/C5 write a velocity setpoint to the four
  `<velocity>` actuators; C2/C4/C6 have none. The benchmark never writes a
  wheel command for a fixed-wheel baseline.

### 10.2 Settling window — deviation from the task brief, stated explicitly

The brief specifies a 2 s settling window. The native startup ramp reaches full
policy authority only at **t = 3.0 s** (`startup_hold_s: 1.0` +
`startup_blend_s: 2.0`, §6.4), and that ramp is part of the validated
controllers, so it must not be shortened.

`benchmark/configs/*.yaml` therefore default to **`settle_s: 3.0`** and the
evaluated window is `[settle_s, duration]`. `settle_s: 2.0` is accepted and
reproduces the brief literally; the value used is written into every
`metadata.json` and printed by `aggregate.py`. Either way the window is
identical for every baseline, so no baseline is advantaged.

External command schedules start at `t = settle_s` with scenario tick 0. During
settling, the base command is exactly zero and the EE command is the measured
initial EE keypoints held constant (equivalent to the native
`reset(sample_first=False)` path), so nothing moves before the evaluation
begins and every target switch falls strictly inside the evaluated window.

### 10.3 Joint acceleration and power (E2)

* Acceleration: `d.qacc[dofadr]` read at **250 Hz**, **unfiltered**. This is
  MuJoCo's own solved acceleration, which is exact for the model, and avoids
  the differentiation noise a finite difference would add. A finite-difference
  estimate `(qvel[t] − qvel[t−1])/dt` is computed in parallel and stored in
  `metrics.json` as `leg_acc_rms_fd` / `arm_acc_rms_fd` so the two can be
  compared; the reported metric is the `qacc` one.
* Power proxy: `P_abs(t) = Σ_j |d.actuator_force[j] · d.qvel[dof(j)]|` at
  250 Hz, accumulated online, reported as a time mean over the evaluated
  window, split into `leg_power` (12), `arm_power` (6), `wheel_power` (4, zero
  for fixed-wheel) and `total_power` (= leg + arm + wheel + gripper).
* Acceleration RMS compares **matched physical DOFs only**: the 12 leg joints
  and the 6 arm joints. Wheels are excluded from both acceleration metrics, and
  no normalisation by action dimension is applied (22 vs 18 action dims are
  never divided out).

### 10.4 Termination / failure criteria

No deployment script has a termination condition — they run for
`simulation_duration` regardless. The benchmark adds (config
`termination:` block, identical for all baselines):

| Flag | Default |
|---|---|
| `base_height_min` | 0.30 m (`d.qpos[2]`) |
| `base_tilt_max_deg` | 50° on \|roll\| or \|pitch\| |
| non-finite state | always terminal |

On termination the episode stops, `termination_reason` and
`valid_eval_duration_s` are recorded, and the metrics are computed over the
samples actually collected inside the evaluated window. Such episodes are
**retained** in the aggregate; `survival_rate` reports them.

---

## 11. Signals requested by the brief that are not available

| Requested | Status |
|---|---|
| "electrical battery power consumption" | Deliberately not reported. Only the mechanical power proxy `Σ|τ q̇|` is, and it is labelled as such everywhere. |
| Collision / contact information | MuJoCo contacts exist (`d.ncon`, `d.contact`, `d.cfrc_ext`) but **no baseline reads them** and no script defines a collision-failure body set. The benchmark logs `n_contacts` and a boolean `non_foot_contact` (any contact involving a body that is not a wheel/calf/floor) as a diagnostic only; it is not used in any primary metric or termination rule. |
| Training command ranges | Not in this repository. Inferred as ±0.6/±0.6/±0.6 with the evidence in §3.4, declared in one place, and flagged in every `metadata.json` as `base_command_limits_source: "inferred_hl_transport_frozen_wbc_bounds"`. |
| Isaac Lab velocity-command frame | Not in this repository. Base frame chosen with the evidence in §3.3; the yaw-frame alternative is logged for every sample so no re-run is needed to switch. |
| C1 headless execution | C1 and C2 had no `--headless` flag. Added as a strictly additive CLI flag (§12). C3–C6 already had one. |

---

## 12. How the benchmark reuses the controllers, and how that is verified

The six scripts expose no importable controller: C1/C2 are top-level
`__main__` monoliths and C3–C6 build their entire pipeline from closures inside
`main()`. Importing them is impossible without executing a full viewer session.

Approach taken:

1. **Adapters** in `benchmark/controllers/` carry the inference and actuation
   code **moved verbatim** from each script (same constants, same operation
   order, same dtypes, same guards), with the only functional change being that
   `base_command` and `ee_kp_commands` come from an argument instead of a YAML
   constant / internal sampler. Each adapter reads the *same* YAML config file
   as its script, so no numeric constant is duplicated into benchmark code.
2. **The original scripts are not modified**, except for one strictly additive
   change: a `--headless` flag on C1 and C2 (default off, so the standalone
   default behaviour is byte-identical). C3–C6 are untouched.
3. **Parity is verified, not assumed.**
   `benchmark/tests/test_controller_parity.py` runs, for each of C1, C2, C3, C5:
   the original script (headless, native constant `base_command`, native
   `ee_command_seed`, native 8 s EE hold) and the adapter driven by a *replay*
   scenario that reproduces the native sampler bit-for-bit, from the same
   initial state, and compares the full `d.qpos`/`d.qvel`/`d.ctrl` trajectory
   and the raw policy actions. Tolerance is 0 for the first policy tick's action
   and `1e-9` on the action sequence over 10 s
   (`--duration 10 --tol 1e-9`). Any divergence fails the test.

A benchmark run is never declared PASS on exit status: `run.py` validates the
written `commands.csv`, `states.csv`, `metrics.json` and the recorded command
schedule against the scenario specification before marking an episode complete,
and `benchmark/tests/` asserts on the generated artefacts.
