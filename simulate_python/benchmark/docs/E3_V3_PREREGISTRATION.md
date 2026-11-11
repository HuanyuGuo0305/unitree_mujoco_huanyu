# E3 v3 — pre-registration

Written and committed before any E3 v3 simulation (pilot or official). Every
threshold below comes either from the brief or from existing E1 data, never from
an E3 v3 result. Changes after this point are logged in the reproducibility
report as deviations.

## 1. Scientific question

How do increasing locomotion intensity and EE-pose command frequency affect:
- base tracking;
- EE tracking;
- mutual task interference;

under discrete full-pose EE commands, as in WBC training.

## 2. Waypoint candidates (controller-blind)

**Frame and representation.** PLB frame. Every waypoint is a full pose (p, R) encoded as the existing 9-D keypoints.

**Line candidates.** 5 equally spaced points on a straight segment:
- centre c ∈ {x 0.50, 0.60, 0.70} × {z 0.65, 0.80, 0.95}, y = 0;
- direction ∈ {y, z, (y+z)/√2, (y−z)/√2, x, (x+y)/√2};
- length ∈ {0.30, 0.40, 0.50, 0.60} m.

**Circle candidates.** 5 equally spaced points c + r(cos φ_i e1 + sin φ_i e2), with φ_i = 90° + 72° i:
- plane (e1, e2) ∈ {frontal (y, z), sagittal (x, z), horizontal (x, y), frontal tilted 30° about y};
- radius ∈ {0.15, 0.20, 0.25, 0.30} m;
- same centres as the lines.

**Reference orientation R_c.** Two variants:
- (a) the orientation of the native-table row nearest to c;
- (b) the chordal mean orientation of all table rows within 0.10 m of c.

**Waypoint orientation scheme (D, ρ).** R_i = Rx_local(roll_i) · Rot(a → d_i) · R_c, where:
- a is R_c's approach axis (keypoint x-axis);
- d_i = normalise(p_i − (c − D a)), i.e. the gripper tilts to point from a pivot a distance D behind the centre towards the waypoint;
- roll_i = ρ · u_i, with u_i = −1…1 linear along the line, or cos(φ_i − 90°) on the circle.

Schemes (D [m], ρ [deg]): (0.35, 0), (0.5, 0), (0.8, 0), (0.5, 25), (∞, 25), (0.8, 25).

## 3. Kinematic screening

A candidate must pass every item for all five poses, at each controller's own nominal stance. The nominal stance is the base pose and arm state logged at t = 3.0 s in E1 T1, seed 0, for C1, C3 and C5.

| Item | Requirement |
|---|---|
| K1 reachability | independent DLS IK (MuJoCo `ground.xml`, body `gripperStator`): position < 1 mm, orientation < 0.5° |
| K2 joint margin | ≥ 0.10 rad inside C3's IK joint band (the tightest band of the three controllers) |
| K3 collision clearance | ≥ 0.03 m between the collidable geoms of link02 … gripperMover and those of the ground, trunk, head, tail, legs and wheels; ≥ 0.03 m between link04 … gripperMover and arm_base/link00 |
| K4 manipulability | σ_min(J_6×6) ≥ 0.05 |
| K5 training support | position ≤ 0.05 m from the nearest native-table row, and some table row within 0.10 m whose orientation is within 20° |
| K6 orientation change | max pairwise orientation difference ≥ 20° (meaningful); every consecutive change in the traversal order ≤ 60° |
| K7 coverage | span (max pairwise waypoint distance) ≥ 0.30 m |

**Kinematic pre-score.** S_kin = 0.4 Ĉ_pos + 0.2 Ĉ_ori + 0.2 F, where:
- Ĉ is normalised by the maximum over the passing candidates of the same pattern;
- C_ori = max pairwise orientation angle;
- F = min(1, joint margin / 0.30 rad, clearance / 0.10 m, σ_min / 0.15).

**Pilot set.** The top 25 per pattern by S_kin, with at most 2 candidates sharing the same (centre, direction/plane, size). If no candidate of a pattern survives the pilot, the next 25 are piloted.

## 4. Pilot (all three controllers, identical conditions)

**Seeds and conditions:**
- Seeds 9000, 9001 and 9002, disjoint from the official seeds 0–99 and the showcase pool 100–149.
- Base still (B = 0).
- 3 s settle holding the initial EE pose, then waypoints 0, 1, 2, 3, 4, each held for 6 s (33 s episode).

**Per controller and waypoint, mean over the 3 seeds:**
- initial-response error: mean over 0.5–1.5 s after the switch;
- steady-state error: mean over 4.0–6.0 s after the switch.

**Rejection rules:**
- P1: any pilot episode terminates.
- P2: the worst controller's steady-state error at any waypoint is ≥ 0.08 m or ≥ 0.35 rad (the primary reaching tolerance, §6).
- P3: the worst controller's initial-response error at any waypoint is ≥ 0.16 m or ≥ 0.70 rad (twice the tolerance).

**Selection score (per pattern, among survivors).** S = 0.4 Ĉ_pos + 0.2 Ĉ_ori + 0.2 F + 0.2 (1 − W), where:
- W = max over controllers and waypoints of max(e_ss,pos / 0.08, e_ss,ori / 0.35);
- the highest S wins; ties go to larger Ĉ_pos, then the lower candidate id.

**Controller neutrality.** No quantity in S, or in P1–P3, involves a difference between controllers or a dynamic-condition result.

The selection is frozen into `configs/E3.yaml`, together with its SHA-256, before the official run.

## 5. Official protocol

**Design.** 5 × 5 factorial × 2 patterns × 3 controllers × seeds 0–99 = 15,000 episodes.
- Base levels B ∈ {0, 0.15, 0.30, 0.45, 0.60}, with per-axis bounds |vx|, |vy| ≤ B m/s and |wz| ≤ B rad/s.
- The base commands use the existing seeded generator (`make_base_velocity_schedule`, U(2, 4) s segments, U(−1, 1)³ × B; B = 0 commands zero).

**EE levels.** The EE update interval is T ∈ {Fixed, 6.0, 4.0, 3.0, 2.0} s.
- Fixed holds the seed's start waypoint for the whole episode.
- Otherwise the command switches every T s starting at t = 3 s, without interpolation.

**Waypoint sequence.**
- Line: ping-pong 0-1-2-3-4-3-2-1-…
- Circle: cyclic 0-1-2-3-4-0-…
- Start waypoint and direction are counterbalanced by seed: start = (seed mod 10) // 2, direction = +1 if seed is even, else −1. At a line end the direction is forced inward.
- Hence for each pattern every start waypoint is used by 20 seeds and each direction by 50 (except the forced line ends).

**Pairing.**
- The EE sequence depends only on (pattern, T, seed) and is identical across base levels and controllers.
- The base sequence depends only on (B, seed) and is identical across EE levels, patterns and controllers.
- The initial state depends only on the seed.

**Episode.** 30 s with a 3 s settle that holds the measured initial EE pose (the runner's default). All controller settings, checkpoints, termination rules (base height < 0.30 m, tilt > 50°) and frequencies (250 Hz physics, 50 Hz policy) are unchanged.

## 6. Metrics

**Tracking RMSEs.** Over every evaluated tick from 3 s to 30 s (1,350 ticks), switching transients included, with the same definitions as E1:
- base XY velocity RMSE [m/s] and yaw-rate RMSE [rad/s], in the body frame;
- EE position RMSE [m] in the PLB frame;
- EE orientation RMSE [rad], the RMSE of the SO(3) geodesic angle.

**Directional degradation** (always reported together with the absolute errors):
- Base→EE: ΔE_ee(B, T) = E_ee(B, T) − E_ee(0, T);
- EE→Base: ΔE_b(B, T) = E_b(B, T) − E_b(B, Fixed).

**Event windows.**
- The event time is the command change.
- The local baseline is the mean error over [−0.5, 0) s.
- The response window is [0, 1.5) s.
- An event is *isolated* if no command change of the other subsystem lies within [−1.0, +1.5) s of it; otherwise it is *overlapping*.
- The first EE command at t = 3 s (the step from the initial pose) is excluded from the event analysis but included in the RMSEs.

**EE→Base**, per EE switch, on the base XY velocity error ‖v − v*‖ and |wz − wz*|:
- peak excess over the local baseline;
- integrated excess ∫(e − baseline) dt over the response window;
- recovery time: the first time after the peak at which the error stays ≤ baseline + 0.05 m/s (0.05 rad/s for yaw) for 0.2 s, censored at 1.5 s.

**Base→EE**, per base command change (B > 0), on the EE position and orientation errors: the same three quantities, with recovery tolerances of 0.01 m and 0.05 rad.

**Target reaching**, per EE target with T ≠ Fixed, excluding the first:
- Reached: position ≤ 0.08 m and orientation ≤ 0.35 rad, held for 0.2 s (primary). The strict tolerance is 0.04 m / 0.20 rad.
- Response time: the time from the switch to the start of the first qualifying dwell.
- An unreached target is a failure, and its time is right-censored at the hold length (or at termination).
- Reported: success rate (Wilson CI) and restricted mean response time (censored times counted at their censoring time).

Per-event quantities are averaged within an episode first. The seed is the statistical unit.

## 7. Statistics

**Per condition × pattern × controller:**
- mean ± SD (ddof = 1) over 100 seeds;
- percentile bootstrap 95 % CI (10,000 resamples, seed 20261106);
- completion rate with Wilson CI;
- mean effective evaluation time.

**Paired comparisons.**
- C3 − C1 and C5 − C1 are compared on seed-level differences, with a paired bootstrap CI and a paired t-test.
- Holm correction is applied within each declared family (metric × comparison type × pattern); BH q-values are also reported.
- Effect size: relative difference and Cohen's d_z.
- A difference is "practically relevant" if |relative difference| ≥ 10 %.
- Non-significance is never reported as equivalence.

Terminated episodes stay in every statistic and are scored over their valid window. A sensitivity analysis restricted to completed episodes is also reported.
