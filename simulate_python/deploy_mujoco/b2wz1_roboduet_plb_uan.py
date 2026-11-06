#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""C5 RoboDuet Stage 2 MuJoCo Sim2Sim, TWO normalized ONNX policies.

TRAINING ABI (not C3/VBC or C1/WBC):
  ARM actor: 330D (5 x 66D, term-major) -> 8 raw actions
             6 Z1 joint offsets + 2 raw posture-guidance logits.
  LOCO actor: 335D (5 x 67D, term-major) -> 16 raw actions
              12 leg offsets + 4 wheel velocity targets.
  MuJoCo: 250 Hz / policy: 50 Hz / Frozen UAN: 250 Hz.

Strict causal order per control tick:
  append physical observation to BOTH histories once
  -> ARM ONNX -> tanh/asymmetric posture scaling -> 0.35 rad/s slew
  -> LOCO posture history SHIFT + current guidance injection
  -> LOCO ONNX -> leg/arm/wheel targets.

Run from unitree_mujoco/simulate_python/:
  python3 deploy_mujoco/b2wz1_roboduet_plb_uan.py configs/b2wz1_roboduet_plb_uan.yaml
  python3 deploy_mujoco/b2wz1_roboduet_plb_uan.py configs/b2wz1_roboduet_plb_uan.yaml --headless

Reuses C1/C3 tested ground.xml, wheels (velocity servos), and frozen Z1 UAN.
The original C1/C3 scripts and robot XMLs are NOT changed.
"""
from __future__ import annotations

import argparse
import csv
from collections import deque
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import yaml

# This file is installed in unitree_mujoco/simulate_python/deploy_mujoco.
SIM_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SIM_ROOT.parent
for import_root in (REPO_ROOT, SIM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

LEG_ORDER_DEFAULT = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
WHEEL_ORDER_DEFAULT = ("FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint")
ARM_ORDER = tuple(f"joint{i}" for i in range(1, 7))

# Terms ordered EXACTLY as C5 S1 policy and C5 S2 arm_policy config.
LOCO_TERMS = (
    ("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3),
    ("base_posture_commands", 2), ("joint_pos_leg", 12), ("joint_pos_arm", 6),
    ("joint_vel_leg", 12), ("joint_vel_arm", 6), ("joint_vel_wheel", 4),
    ("actions", 16),
)
ARM_TERMS = (
    ("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3),
    ("ee_kp_commands", 9), ("joint_pos_leg", 12), ("joint_pos_arm", 6),
    ("joint_vel_leg", 12), ("joint_vel_arm", 6), ("joint_vel_wheel", 4),
    ("last_arm_joint_action", 6), ("previous_guidance", 2),
)
assert sum(d for _, d in LOCO_TERMS) == 67
assert sum(d for _, d in ARM_TERMS) == 66


def resolve_file(value: str, label: str, yaml_path: Path) -> Path:
    """Always prefer canonical repository-root assets, not stale simulate_python copies."""
    path = Path(value).expanduser()
    paths = [path] if path.is_absolute() else [REPO_ROOT / path, SIM_ROOT / path, yaml_path.parent / path]
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    matches = [p for p in paths if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"{label}={value!r} missing. Searched:\n" + "\n".join(str(p) for p in paths))
    chosen = matches[0]
    checksum = hashlib.sha256(chosen.read_bytes()).hexdigest()[:16]
    print(f"[ASSET AUDIT] {label}: {chosen} (sha256={checksum})", flush=True)
    for extra in matches[1:]:
        if hashlib.sha256(extra.read_bytes()).hexdigest()[:16] != checksum:
            print(f"[ASSET WARNING] Ignored DIFFERENT copy of {label}: {extra}", flush=True)
    return chosen


def check_shape(name: str, a, shape: tuple[int, ...]) -> np.ndarray:
    x = np.asarray(a)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f"{name}: expected finite {shape}, got {x.shape}")
    return x


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """MuJoCo free-base quaternion is wxyz. Return world_R_body."""
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        raise ValueError("Invalid root quaternion")
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def yaw_rotation(base_quat: np.ndarray) -> np.ndarray:
    R = quat_to_rotation(base_quat)
    yaw = math.atan2(R[1, 0], R[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def roll_pitch_from_quat(base_quat: np.ndarray) -> tuple[float, float]:
    R = quat_to_rotation(base_quat)
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2]))
    return roll, pitch


def current_ee_keypoints_plb(d, ee_body_id: int, *, ground_z: float, dx: float, dz: float) -> np.ndarray:
    """Match PLB convention of C3: yaw-only base, ground-level origin, gripperStator."""
    root = d.qpos[:3].copy()
    root[2] = ground_z
    R_plb = yaw_rotation(d.qpos[3:7])
    R_ee = d.xmat[ee_body_id].reshape(3, 3)
    p0 = R_plb.T @ (d.xpos[ee_body_id] - root)
    p1 = p0 + R_plb.T @ (R_ee @ np.array([dx, 0.0, 0.0]))
    p2 = p0 + R_plb.T @ (R_ee @ np.array([0.0, 0.0, dz]))
    return np.concatenate((p0, p1, p2)).astype(np.float32)


def plb_keypoints_to_world(d, kp: np.ndarray, *, ground_z: float) -> np.ndarray:
    root = d.qpos[:3].copy()
    root[2] = ground_z
    R = yaw_rotation(d.qpos[3:7])
    return (np.asarray(kp, dtype=np.float64).reshape(3,3) @ R.T + root).astype(np.float64)


class PresampledDirectPLBCommand:
    """C5 S2 'sample_mode=random': one 9D row, hold 8 s; no trajectory interpolation."""
    def __init__(self, filepath: Path, *, control_dt: float, period_s: float, seed: int):
        arr = np.load(filepath, allow_pickle=False).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9 or arr.shape[0] < 1 or not np.isfinite(arr).all():
            raise ValueError(f"PLB goal dataset must be finite [N,9], got {arr.shape}")
        self.table = arr
        self.rng = np.random.default_rng(seed)
        self.cycle_steps = max(1, round(period_s / control_dt))
        self.step_count = 0
        self.command = self.table[int(self.rng.integers(0, len(self.table)))].copy()

    def step(self) -> np.ndarray:
        self.step_count += 1
        if self.step_count >= self.cycle_steps:
            self.command = self.table[int(self.rng.integers(0, len(self.table)))].copy()
            self.step_count = 0
        return self.command.copy()


class TermMajorHistory:
    """Isaac Lab ObsGroup: each TERM independently retains five oldest->newest frames."""
    def __init__(self, schema, length: int):
        self.schema = tuple(schema)
        self.length = int(length)
        if self.length != 5:
            raise ValueError("C5 S2 requires 5-frame history")
        self.hist = {name: deque(maxlen=length) for name, _ in self.schema}

    def seed(self, observation: dict[str, np.ndarray]) -> None:
        self.hist = {name: deque(maxlen=self.length) for name, _ in self.schema}
        for name, dim in self.schema:
            x = check_shape(name, observation[name], (dim,)).astype(np.float32, copy=True)
            for _ in range(self.length):
                self.hist[name].append(x.copy())

    def append(self, observation: dict[str, np.ndarray]) -> None:
        for name, dim in self.schema:
            x = check_shape(name, observation[name], (dim,)).astype(np.float32, copy=True)
            self.hist[name].append(x)

    def stack(self) -> np.ndarray:
        for name, _ in self.schema:
            if len(self.hist[name]) != self.length:
                raise RuntimeError(f"History {name} not initialized")
        return np.concatenate([
            np.asarray(self.hist[name], dtype=np.float32).reshape(-1)
            for name, _ in self.schema
        ]).astype(np.float32)


class PostureGuidance:
    """C5 bridge: raw arm[6:8] -> tanh/angle -> clip -> slew at 50 Hz."""
    def __init__(self, dt: float, slew_rate: float = 0.35):
        self.dt = float(dt)
        self.slew_rate = float(slew_rate)
        self.command = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)

    @staticmethod
    def decode(arm_raw: np.ndarray) -> np.ndarray:
        a = check_shape("ARM policy output", arm_raw, (8,))
        x = np.tanh(a[6:8].astype(np.float64))
        pitch = float(x[0]) * math.radians(25.0 if x[0] >= 0 else 5.0)
        roll = float(x[1]) * math.radians(3.0)
        return np.array([pitch, roll], dtype=np.float32)

    def inject(self, arm_raw: np.ndarray) -> np.ndarray:
        self.target = self.decode(arm_raw)
        lower = np.deg2rad([-5.0, -3.0])
        upper = np.deg2rad([25.0, 3.0])
        bounded = np.clip(self.target.astype(np.float64), lower, upper)
        self.command = (self.command.astype(np.float64) + np.clip(
            bounded - self.command.astype(np.float64),
            -self.slew_rate * self.dt, self.slew_rate * self.dt
        )).astype(np.float32)
        return self.command.copy()


def patch_loco_posture_history(raw_loco: np.ndarray, current_guidance: np.ndarray) -> np.ndarray:
    """MATCH Stage2PolicyBridge.observations_for_loco: shift ONLY posture term."""
    src = check_shape("LOCO raw observation", raw_loco, (335,))
    g = check_shape("executed posture", current_guidance, (2,))
    out = src.copy()
    # Indices from 5*(base_ang3+gravity3+velocity_cmd3) = 45;
    # base_posture commands are 5*2 = [45:55].
    out[45:53] = src[47:55]
    out[53:55] = g
    return out


def load_onnx_session(path: Path, input_dim: int, output_dim: int):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ins, outs = sess.get_inputs(), sess.get_outputs()
    if len(ins) != 1 or len(outs) != 1:
        raise RuntimeError(f"{path.name}: expected exactly one input and output")
    for which, shape, expected in (("input", ins[0].shape, input_dim), ("output", outs[0].shape, output_dim)):
        if len(shape) != 2 or (isinstance(shape[1], int) and shape[1] != expected):
            raise RuntimeError(f"{path.name}: {which} shape {shape}, expected [batch,{expected}]")
    print(f"[ONNX AUDIT] {path.name}: {ins[0].shape} -> {outs[0].shape}", flush=True)
    return sess, ins[0].name, outs[0].name


def onnx_action(session_triplet, obs: np.ndarray, expected_dim: int, label: str) -> np.ndarray:
    sess, inp, out = session_triplet
    raw = sess.run([out], {inp: np.ascontiguousarray(obs.reshape(1, -1), dtype=np.float32)})[0]
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    return check_shape(f"{label} ONNX action", raw, (expected_dim,))


def audit_manifest(path: Path, *, strict: bool) -> None:
    manifest = path.parent / "export_manifest.json"
    if not manifest.is_file():
        if strict:
            raise FileNotFoundError(f"Missing ONNX export manifest next to policy: {manifest}")
        print(f"[ONNX WARNING] manifest missing: {manifest}", flush=True)
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for key, obs, acts in (("arm", 330, 8), ("loco", 335, 16)):
        part = data.get("policies", {}).get(key)
        if not isinstance(part, dict) or part.get("actor_obs") != obs or part.get("action_dim") != acts:
            raise RuntimeError(f"Manifest policy {key} incorrect: {part}")
        if not part.get("includes_trained_actor_normalizer", False):
            raise RuntimeError(f"{key} ONNX manifest reports MISSING trained observation normalizer")
        if not part.get("onnxruntime_verified", False):
            if strict:
                raise RuntimeError(f"{key} export did not pass ONNX Runtime parity check")
            print(f"[ONNX WARNING] {key} ORT parity check was skipped at export", flush=True)
    print(f"[ONNX AUDIT] ARM+LOCO manifest PASS (Stage 2 iteration={data.get('iteration_stage2')})", flush=True)


def audit_scene(m, mujoco_module, *, expect_ground_friction=(0.8,0.2,0.1)) -> None:
    mujoco = mujoco_module
    planes = [i for i in range(m.ngeom) if int(m.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
              and int(m.geom_bodyid[i]) == 0]
    if len(planes) != 1:
        raise RuntimeError(f"Expected exactly one world ground plane; found {len(planes)}")
    gid = planes[0]
    friction = m.geom_friction[gid]
    if int(m.geom_matid[gid]) < 0 or not np.allclose(friction, expect_ground_friction, atol=1e-5):
        raise RuntimeError(f"Wrong C1 ground material/friction: {friction}, material={m.geom_matid[gid]}")
    print(f"[SCENE AUDIT] floor friction={friction.tolist()}, material={int(m.geom_matid[gid])}", flush=True)
    for prefix in ("FL", "FR", "RL", "RR"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_wheel_link")
        if bid < 0:
            raise RuntimeError(f"Missing {prefix}_wheel_link")
        meshes = []
        for i in range(m.ngeom):
            if int(m.geom_bodyid[i]) == bid and int(m.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                meshes.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[i])))
        if meshes != [f"{prefix}_wheel", f"{prefix}_wheel"]:
            raise RuntimeError(f"{prefix} wheel geometry mismatch: {meshes}; use C1 ground.xml")
        print(f"[WHEEL AUDIT] {prefix}: {meshes}")


def audit_actuators(m, mujoco_module, joint_names: tuple[str, ...], wheel_names: tuple[str, ...]):
    mujoco = mujoco_module
    jids, qpos_addr, qvel_addr = {}, {}, {}
    for name in joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or int(m.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise RuntimeError(f"C5 expected hinge joint '{name}'")
        jids[name] = jid
        qpos_addr[name] = int(m.jnt_qposadr[jid])
        qvel_addr[name] = int(m.jnt_dofadr[jid])
    by_joint = {}
    for ai in range(m.nu):
        jid = int(m.actuator_trnid[ai, 0])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in joint_names:
            if name in by_joint:
                raise RuntimeError(f"Duplicate actuator on {name}")
            by_joint[name] = ai
    if set(by_joint) != set(joint_names):
        raise RuntimeError(f"Actuator joint set mismatch: {set(joint_names) ^ set(by_joint)}")
    for name in joint_names:
        ai = by_joint[name]
        if int(m.actuator_dyntype[ai]) != int(mujoco.mjtDyn.mjDYN_NONE) or int(m.actuator_gaintype[ai]) != int(mujoco.mjtGain.mjGAIN_FIXED) or not np.isclose(m.actuator_gear[ai,0], 1.0):
            raise RuntimeError(f"Actuator {name} differs from C1")
        if name not in wheel_names:
            if int(m.actuator_biastype[ai]) != int(mujoco.mjtBias.mjBIAS_NONE) or not np.isclose(m.actuator_gainprm[ai,0], 1.0):
                raise RuntimeError(f"{name}: expected torque motor, not velocity servo")
        else:
            if (int(m.actuator_biastype[ai]) != int(mujoco.mjtBias.mjBIAS_AFFINE)
                    or not np.isclose(m.actuator_gainprm[ai,0],10.0)
                    or not np.allclose(m.actuator_biasprm[ai,:3], [0,0,-10])
                    or not bool(m.actuator_ctrllimited[ai])
                    or not np.allclose(m.actuator_ctrlrange[ai], [-50,50])
                    or not bool(m.actuator_forcelimited[ai])
                    or not np.allclose(m.actuator_forcerange[ai], [-20,20])):
                raise RuntimeError(f"{name}: expected C1 <velocity kv=10> servo (NOT torque motor)")
    print("[ACTUATOR AUDIT] PASS: 19 motors + 4 wheel velocity servos", flush=True)
    return qpos_addr, qvel_addr, jids, by_joint


def audit_uan_arm_limits(m, jids, by_joint, uan_final_limits: np.ndarray, *, sync: bool):
    for k, name in enumerate(ARM_ORDER):
        lim = float(uan_final_limits[k])
        ai = by_joint[name]
        for label, enabled, limits in (("ctrl", m.actuator_ctrllimited, m.actuator_ctrlrange),
                                       ("force", m.actuator_forcelimited, m.actuator_forcerange)):
            if bool(enabled[ai]) and (limits[ai, 0] > -lim+1e-6 or limits[ai, 1] < lim-1e-6):
                if not sync:
                    raise RuntimeError(f"{name} {label} clips Frozen UAN at {limits[ai]}")
                print(f"[UAN LIMIT AUDIT] {name} {label}: {limits[ai].copy()} -> [-{lim},+{lim}]")
                limits[ai] = [-lim, lim]
        jid = jids[name]
        if hasattr(m, "jnt_actfrclimited") and bool(m.jnt_actfrclimited[jid]):
            if m.jnt_actfrcrange[jid, 0] > -lim+1e-6 or m.jnt_actfrcrange[jid, 1] < lim-1e-6:
                if not sync:
                    raise RuntimeError(f"{name} joint actfrcrange clips Frozen UAN")
                m.jnt_actfrcrange[jid] = [-lim, lim]


def visualize_keypoints(viewer, m, d, mujoco, goal: np.ndarray, current: np.ndarray, ground_z: float):
    """Lightweight red-target/blue-current visualization, local to viewer scene."""
    scene = viewer.user_scn
    scene.ngeom = 0
    root = d.qpos[:3].copy()
    root[2] = ground_z
    R = yaw_rotation(d.qpos[3:7])
    for kps, color in ((goal, [1.0,0.1,0.1,0.9]), (current, [0.1,0.2,1.0,0.9])):
        world = plb_keypoints_to_world(d, kps, ground_z=ground_z)
        for i in range(3):
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.array([0.035,0,0],dtype=float), world[i],
                               np.eye(3).reshape(-1), np.array(color,dtype=float))
            scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("yaml_path", type=str)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Load/validate models and assets; do not advance physics")
    ap.add_argument("--metrics-csv", type=str, default="", help="Optional CSV of per-policy-step MuJoCo metrics")
    args = ap.parse_args()
    yaml_path = Path(args.yaml_path).expanduser().resolve()
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError("Expected YAML mapping")

    dt = float(cfg["simulation_dt"])
    decimation = int(cfg["control_decimation"])
    control_dt = dt * decimation
    history_len = int(cfg["history_length"])
    if abs(dt - 0.004) > 1e-10 or decimation != 5 or history_len != 5:
        raise RuntimeError("C5 S2 contract: 250Hz physics, 50Hz control, term-major history=5")
    duration = float(cfg["simulation_duration"])
    hold_s, blend_s = float(cfg["startup_hold_s"]), float(cfg["startup_blend_s"])
    if min(duration, hold_s, blend_s) < 0:
        raise ValueError("Invalid simulation/hold/blend durations")
    velocity_command = check_shape("base_command", cfg["base_command"], (3,)).astype(np.float32)
    root_pos = check_shape("root_pos", cfg["root_pos"], (3,)).astype(np.float64)
    root_quat = check_shape("root_quat_wxyz", cfg["root_quat_wxyz"], (4,)).astype(np.float64)
    names_yaml = list(cfg["config_joint_names"])
    if len(names_yaml) != 23 or len(set(names_yaml)) != 23:
        raise RuntimeError("C5 expected 23 config_joint_names")
    defaults = dict(zip(names_yaml, check_shape("default_joint_pos", cfg["default_joint_pos"], (23,))))
    kp_gain = dict(zip(names_yaml, check_shape("kps", cfg["kps"], (23,))))
    kd_gain = dict(zip(names_yaml, check_shape("kds", cfg["kds"], (23,))))
    leg_names = tuple(cfg["leg_action_joint_names"])
    wheel_names = tuple(cfg["wheel_action_joint_names"])
    arm_names = ARM_ORDER
    gripper_name = "jointGripper"
    if len(leg_names) != 12 or len(wheel_names) != 4 or set(leg_names + wheel_names + arm_names + (gripper_name,)) != set(names_yaml):
        raise RuntimeError("Action mapping invalid; compare against actual Isaac Lab joint_ids")
    if not len(set(leg_names)) == 12 or not len(set(wheel_names)) == 4:
        raise RuntimeError("Duplicate joint in action order")
    if leg_names != LEG_ORDER_DEFAULT or wheel_names != WHEEL_ORDER_DEFAULT:
        print("[JOINT ORDER] Explicit non-C3 ordering selected in YAML; verify against Isaac actor order")
    print("[JOINT ORDER] Leg 12D:", leg_names, flush=True)
    print("[JOINT ORDER] Wheel 4D: Isaac '*_foot_joint' -> MuJoCo '*_wheel_joint'", wheel_names, flush=True)
    print("[JOINT ORDER] Arm 6D:", arm_names, flush=True)
    leg_scale = float(cfg["leg_action_scale"])
    arm_scale = float(cfg["arm_action_scale"])
    wheel_scale = float(cfg["wheel_action_scale"])
    if (leg_scale, arm_scale, wheel_scale) != (0.25, 0.10, 4.0):
        raise RuntimeError("C5 S2 action scales must be 0.25/0.10/4.0, from IsaacLab EnvCfg")
    leg_tau_lim = check_shape("leg_torque_limits", cfg["leg_torque_limits"], (12,)).astype(np.float64)
    wheel_vel_lim = check_shape("wheel_velocity_limits", cfg["wheel_velocity_limits"], (4,)).astype(np.float64)
    grip_tau_lim = float(cfg["gripper_torque_limit"])
    ee_dx, ee_dz, ee_ground_z = float(cfg["ee_kp_dx"]), float(cfg["ee_kp_dz"]), float(cfg["ee_ground_z"])
    command_period = float(cfg["ee_cycle_duration_s"])
    uan_dev = str(cfg["uan_device"])
    uan_hist_len = int(cfg["uan_history_length"])
    uan_pos_scale, uan_vel_scale = float(cfg["uan_pos_scale"]), float(cfg["uan_vel_scale"])
    uan_action_scale = float(cfg["uan_action_scale"])
    uan_kp = check_shape("uan_kp", cfg["uan_kp"], (6,)).astype(np.float32)
    uan_kd = check_shape("uan_kd", cfg["uan_kd"], (6,)).astype(np.float32)
    uan_nom_lim = check_shape("uan_nominal_torque_limits", cfg["uan_nominal_torque_limits"], (6,)).astype(np.float32)
    uan_fin_lim = check_shape("uan_final_torque_limits", cfg["uan_final_torque_limits"], (6,)).astype(np.float32)
    if uan_hist_len != 25 or not np.all(uan_fin_lim >= uan_nom_lim):
        raise RuntimeError("Frozen UAN history/limits incorrect")
    
    arm_path = resolve_file(cfg["arm_policy_path"], "arm_policy_path", yaml_path)
    loco_path = resolve_file(cfg["loco_policy_path"], "loco_policy_path", yaml_path)
    uan_path = resolve_file(cfg["uan_model_path"], "uan_model_path", yaml_path)
    xml_path = resolve_file(cfg["xml_path"], "xml_path", yaml_path)
    goals_path = resolve_file(cfg["ee_command_path"], "ee_command_path", yaml_path)
    if arm_path.parent != loco_path.parent:
        raise RuntimeError("ARM+LOCO ONNX must reside in same export folder")
    audit_manifest(arm_path, strict=bool(cfg.get("require_verified_manifest", False)))

    # Optional runtime imports are intentionally lazy, allowing unit-test of
    # the policy/history/bridge contract without an installed MuJoCo.
    import mujoco
    import mujoco.viewer
    import onnxruntime  # noqa: F401
    import torch
    from utilities.mujoco_helper import get_sensor_slice

    arm_onnx = load_onnx_session(arm_path, 330, 8)
    loco_onnx = load_onnx_session(loco_path, 335, 16)
    if uan_dev.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Frozen UAN configured for CUDA, but CUDA unavailable")
    uan = torch.jit.load(str(uan_path), map_location=uan_dev).eval()
    for p in uan.parameters():
        p.requires_grad_(False)
    with torch.inference_mode():
        test_out = uan(torch.zeros((1, 6*uan_hist_len*2), device=uan_dev))
        if isinstance(test_out, (tuple,list)):
            test_out = test_out[0]
        check_shape("frozen UAN dummy response", test_out.detach().cpu().numpy(), (1,6))

    m = mujoco.MjModel.from_xml_path(str(xml_path))
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    if (m.nq, m.nv, m.nu) != (30,29,23):
        raise RuntimeError(f"Wrong C1 wheeled MJCF: nq/nv/nu={(m.nq,m.nv,m.nu)}, expected (30,29,23)")
    audit_scene(m, mujoco)
    names = leg_names + arm_names + wheel_names + (gripper_name,)
    qpos_addr, qvel_addr, jids, by_joint = audit_actuators(m, mujoco, names, wheel_names)
    audit_uan_arm_limits(m, jids, by_joint, uan_fin_lim, sync=bool(cfg.get("sync_mujoco_arm_limits_to_uan",True)))
    ee_body_name = str(cfg.get("ee_body", "gripperStator"))
    ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    if ee_bid < 0:
        raise RuntimeError(f"EE body missing: {ee_body_name}")

    d.qpos[:] = 0
    d.qvel[:] = 0
    d.ctrl[:] = 0
    d.qpos[:3] = root_pos
    d.qpos[3:7] = root_quat
    for name in names:
        d.qpos[qpos_addr[name]] = defaults[name]
    mujoco.mj_forward(m, d)
    print(f"[C5 S2] Model ready: root_z={d.qpos[2]:.4f}, onnx ARM330->8 LOCO335->16", flush=True)
    if args.dry_run:
        print("[C5 S2] dry-run model/actuator/assets validation PASS (no simulation)")
        return

    leg_qidx = [qpos_addr[n] for n in leg_names]
    leg_vidx = [qvel_addr[n] for n in leg_names]
    arm_qidx = [qpos_addr[n] for n in arm_names]
    arm_vidx = [qvel_addr[n] for n in arm_names]
    wheel_vidx = [qvel_addr[n] for n in wheel_names]
    leg_default = np.array([defaults[n] for n in leg_names], dtype=np.float64)
    arm_default = np.array([defaults[n] for n in arm_names], dtype=np.float64)
    leg_kp = np.array([kp_gain[n] for n in leg_names], dtype=np.float64)
    leg_kd = np.array([kd_gain[n] for n in leg_names], dtype=np.float64)
    grip_kp, grip_kd = float(kp_gain[gripper_name]), float(kd_gain[gripper_name])
    grip_qidx, grip_vidx = qpos_addr[gripper_name], qvel_addr[gripper_name]

    ee_sampler = PresampledDirectPLBCommand(
        goals_path, control_dt=control_dt, period_s=command_period,
        seed=int(cfg.get("ee_command_seed", 0)),
    )
    posture = PostureGuidance(control_dt, slew_rate=float(cfg.get("guidance_rate_rad_s",0.35)))
    arm_hist = TermMajorHistory(ARM_TERMS, history_len)
    loco_hist = TermMajorHistory(LOCO_TERMS, history_len)
    last_arm = np.zeros(6,dtype=np.float32)
    last_loco = np.zeros(16,dtype=np.float32)
    uan_hist = np.zeros((6, uan_hist_len, 2), dtype=np.float32)
    last_uan_action = np.zeros(6,dtype=np.float32)
    last_uan_delta_tau = np.zeros(6,dtype=np.float32)
    leg_target = leg_default.copy()
    arm_target = arm_default.copy()
    wheel_target = np.zeros(4,dtype=np.float64)

    def build_common_obs() -> dict[str,np.ndarray]:
        base_ang_vel = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)
        base_quat = d.qpos[3:7]
        projected_g = (quat_to_rotation(base_quat).T @ np.array([0.0,0.0,-1.0])).astype(np.float32)
        return {
            "base_ang_vel":base_ang_vel,
            "projected_gravity":projected_g,
            "velocity_commands":velocity_command.copy(),
            "joint_pos_leg":(d.qpos[leg_qidx]-leg_default).astype(np.float32),
            "joint_pos_arm":(d.qpos[arm_qidx]-arm_default).astype(np.float32),
            "joint_vel_leg":d.qvel[leg_vidx].astype(np.float32).copy(),
            "joint_vel_arm":d.qvel[arm_vidx].astype(np.float32).copy(),
            "joint_vel_wheel":d.qvel[wheel_vidx].astype(np.float32).copy(),
        }

    def frame_pair() -> tuple[dict[str,np.ndarray],dict[str,np.ndarray]]:
        values = build_common_obs()
        l = dict(values)
        l.update({"base_posture_commands":posture.command.copy(),
                  "actions":last_loco.copy()})
        a = dict(values)
        a.update({"ee_kp_commands":ee_sampler.command.copy(),
                  "last_arm_joint_action":last_arm.copy(),
                  "previous_guidance":posture.command.copy()})
        return a,l

    first_arm, first_loco = frame_pair()
    arm_hist.seed(first_arm)
    loco_hist.seed(first_loco)

    def frozen_uan_torque():
        nonlocal arm_target
        q = d.qpos[arm_qidx].astype(np.float32)
        v = d.qvel[arm_vidx].astype(np.float32)
        err = (arm_target.astype(np.float32)-q)*uan_pos_scale
        uan_hist[:,:-1] = uan_hist[:,1:]
        uan_hist[:,-1,0] = err
        uan_hist[:,-1,1] = v*uan_vel_scale
        with torch.inference_mode():
            action_tensor = uan(torch.from_numpy(np.ascontiguousarray(uan_hist.reshape(1,-1))).to(uan_dev))
            if isinstance(action_tensor, (tuple,list)):
                action_tensor = action_tensor[0]
            out = action_tensor.detach().cpu().numpy().reshape(-1)
        action = check_shape("Frozen UAN output",out,(6,)).astype(np.float32)
        tau_pd = uan_kp*(arm_target.astype(np.float32)-q)-uan_kd*v
        tau_nom = np.clip(tau_pd,-uan_nom_lim,uan_nom_lim)
        delta_tau = action*uan_action_scale*uan_kp
        tau = np.clip(tau_nom+delta_tau,-uan_fin_lim,uan_fin_lim)
        last_uan_action[:] = action
        last_uan_delta_tau[:] = delta_tau
        return tau

    vis = bool(cfg.get("vis_ee_target",True))
    report_each = int(cfg.get("report_every_physics_steps",200))
    csv_file = None
    csv_writer = None
    if args.metrics_csv:
        csv_path = Path(args.metrics_csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "time_s", "base_z_m", "ee_kp0_error_m", "ee_kp1_error_m", "ee_kp2_error_m",
            "ee_kp_mean_error_m", "guidance_req_pitch_deg", "guidance_exec_pitch_deg",
            "guidance_exec_roll_deg", "base_pitch_deg", "base_roll_deg",
            "uan_action_saturation", "uan_delta_tau_abs_mean_nm", "policy_tick",
        ])
        csv_writer.writeheader()
    viewer_cm = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(m,d)
    time_s = 0.0
    count = 0
    policy_ticks = 0
    with viewer_cm as viewer:
        if viewer is not None:
            with viewer.lock():
                for group in (0,1,2):
                    viewer.opt.geomgroup[group] = 1
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        while time_s < duration and (viewer is None or viewer.is_running()):
            start_wall = time.monotonic()
            q = d.qpos[leg_qidx]
            qd = d.qvel[leg_vidx]
            leg_tau = np.clip(leg_kp*(leg_target-q)-leg_kd*qd, -leg_tau_lim,leg_tau_lim)
            arm_tau = frozen_uan_torque()
            grip_tau = float(np.clip(grip_kp*(defaults[gripper_name]-d.qpos[grip_qidx])-grip_kd*d.qvel[grip_vidx],
                                     -grip_tau_lim,grip_tau_lim))
            d.ctrl[:] = 0.0
            for name,tau in zip(leg_names,leg_tau):
                d.ctrl[by_joint[name]] = tau
            for name,tau in zip(arm_names,arm_tau):
                d.ctrl[by_joint[name]] = tau
            for name,velocity in zip(wheel_names,np.clip(wheel_target,-wheel_vel_lim,wheel_vel_lim)):
                d.ctrl[by_joint[name]] = velocity  # C1 <velocity> uses velocity setpoint, NOT torque.
            d.ctrl[by_joint[gripper_name]] = grip_tau
            mujoco.mj_step(m,d)
            time_s += dt
            if count % decimation == 0:
                ee_sampler.step()
                arm_terms, loco_terms = frame_pair()  # Read single physical state, before new guidance.
                arm_hist.append(arm_terms)
                loco_hist.append(loco_terms)
                arm_obs = check_shape("ARM actor observation", arm_hist.stack(), (330,))
                arm_raw = onnx_action(arm_onnx,arm_obs,8,"ARM")
                # Stage2PolicyBridge injects g_now and patches a COPY of loco obs.
                new_guidance = posture.inject(arm_raw)
                loco_obs = patch_loco_posture_history(loco_hist.stack(),new_guidance)
                loco_raw = onnx_action(loco_onnx,loco_obs,16,"LOCO")
                last_arm[:] = arm_raw[:6]
                last_loco[:] = loco_raw
                if time_s < hold_s:
                    blend = 0.0
                elif blend_s > 0:
                    blend = float(np.clip((time_s-hold_s)/blend_s,0,1))
                else:
                    blend = 1.0
                leg_target = leg_default + blend*leg_scale*loco_raw[:12].astype(np.float64)
                arm_target = arm_default + blend*arm_scale*arm_raw[:6].astype(np.float64)
                wheel_target = blend*wheel_scale*loco_raw[12:16].astype(np.float64)
                policy_ticks += 1
                if csv_writer is not None:
                    ee_current = current_ee_keypoints_plb(
                        d, ee_bid, ground_z=ee_ground_z, dx=ee_dx, dz=ee_dz)
                    ee_err = np.linalg.norm((ee_current-ee_sampler.command).reshape(3,3), axis=1)
                    r, p = roll_pitch_from_quat(d.qpos[3:7])
                    csv_writer.writerow({
                        "time_s": round(time_s, 6),
                        "base_z_m": float(d.qpos[2]),
                        "ee_kp0_error_m": float(ee_err[0]),
                        "ee_kp1_error_m": float(ee_err[1]),
                        "ee_kp2_error_m": float(ee_err[2]),
                        "ee_kp_mean_error_m": float(ee_err.mean()),
                        "guidance_req_pitch_deg": math.degrees(float(posture.target[0])),
                        "guidance_exec_pitch_deg": math.degrees(float(posture.command[0])),
                        "guidance_exec_roll_deg": math.degrees(float(posture.command[1])),
                        "base_pitch_deg": math.degrees(p), "base_roll_deg": math.degrees(r),
                        "uan_action_saturation": float(np.mean(np.abs(last_uan_action)>0.95)),
                        "uan_delta_tau_abs_mean_nm": float(np.mean(np.abs(last_uan_delta_tau))),
                        "policy_tick": policy_ticks,
                    })
            if viewer is not None:
                viewer.cam.lookat[:] = d.qpos[:3]
                if vis:
                    ee_current = current_ee_keypoints_plb(d,ee_bid,ground_z=ee_ground_z,dx=ee_dx,dz=ee_dz)
                    visualize_keypoints(viewer,m,d,mujoco,ee_sampler.command,ee_current,ee_ground_z)
                viewer.sync()
                sleep = dt-(time.monotonic()-start_wall)
                if sleep > 0:
                    time.sleep(sleep)
            if report_each > 0 and count % report_each == 0:
                ee_current = current_ee_keypoints_plb(d,ee_bid,ground_z=ee_ground_z,dx=ee_dx,dz=ee_dz)
                kp_errors = np.linalg.norm((ee_current-ee_sampler.command).reshape(3,3),axis=1)
                roll,pitch = roll_pitch_from_quat(d.qpos[3:7])
                print(f"[C5 S2 t={time_s:6.2f}s] z={d.qpos[2]:.3f} "
                      f"kp_errors={np.round(kp_errors,3).tolist()} "
                      f"guide_req_pitch={math.degrees(float(posture.target[0])):+.1f}deg "
                      f"guide_exec_pitch={math.degrees(float(posture.command[0])):+.1f}deg "
                      f"pitch={math.degrees(pitch):+.1f}deg roll={math.degrees(roll):+.1f}deg "
                      f"uan_sat={np.mean(np.abs(last_uan_action)>0.95):.3f} "
                      f"uan_residual={np.mean(np.abs(last_uan_delta_tau)):.2f}Nm "
                      f"ticks={policy_ticks}", flush=True)
            count += 1
    if csv_file is not None:
        csv_file.close()
        print(f"[C5 S2] Wrote benchmark metrics: {csv_path}", flush=True)
    print(f"[C5 S2] Finished {time_s:.2f} s with {policy_ticks} policy ticks", flush=True)


if __name__ == "__main__":
    main()
