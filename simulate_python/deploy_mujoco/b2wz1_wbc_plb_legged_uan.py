"""
C2: fixed-wheel (legged-only) B2W+Z1 RL WBC / frozen-UAN MuJoCo deployment.

Derived from the user-provided, working C1 B2WZ1 WBC-UAN Sim2Sim runner.
This version expects the mechanically locked-wheel MJCF (19 movable joints:
12 legs, 6 arm, 1 gripper). Wheels remain physical collision bodies but have
no revolute joint or actuator. It refuses to run against an unlocked-wheel XML.

50 Hz policy: 72D per-step observation with 5-step term-major history = 360D;
18 actions = 12 position targets (legs) + 6 position targets (Z1 arm).
At 250 Hz, legs use PD and the arm uses the identical frozen UAN law as C1.

Expected directory layout relative to simulate_python/:
  deploy_mujoco/b2wz1_wbc_plb_legged_uan.py
  configs/b2wz1_wbc_plb_legged_uan.yaml
  unitree_robots/b2wz1/b2wz1_legged.xml  (generated from user-supplied b2wz1.xml)

Command examples:
  python3 deploy_mujoco/b2wz1_wbc_plb_legged_uan.py configs/b2wz1_wbc_plb_legged_uan.yaml --mode pd-stand
  python3 deploy_mujoco/b2wz1_wbc_plb_legged_uan.py configs/b2wz1_wbc_plb_legged_uan.yaml --mode lock-arm-policy
  python3 deploy_mujoco/b2wz1_wbc_plb_legged_uan.py configs/b2wz1_wbc_plb_legged_uan.yaml --mode full-policy

Important: C2 policy checkpoint must come from the retrained 19-DOF USD.
The ONNX export must contain the trained actor observation normalizer.
"""

import os
import sys
import time
import argparse
from collections import deque
from contextlib import nullcontext

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import torch
import yaml

# Resolve imports independently of YAML asset paths.  Actual layout:
# unitree_mujoco/{utilities,exported_policy,unitree_robots}/...
# unitree_mujoco/simulate_python/{deploy_mujoco,configs}/...
from pathlib import Path
simulation_root = Path(__file__).resolve().parent.parent  # simulate_python/
repository_root = simulation_root.parent                  # unitree_mujoco/
C2_PATH_FIX_VERSION = "C2-repo-root-xml-audit-v5-20261104"
print(f"[C2 {C2_PATH_FIX_VERSION}] simulate_python={simulation_root}; repository={repository_root}")

# Prefer the real repository utilities/; tolerate a standalone simulate_python
# checkout which happens to contain utilities/ as well.
for _root in (simulation_root, repository_root):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

if not any(
    (_root / "utilities" / "math.py").is_file()
    and (_root / "utilities" / "mujoco_helper.py").is_file()
    for _root in (repository_root, simulation_root)
):
    raise RuntimeError(
        "Cannot find utilities/math.py and utilities/mujoco_helper.py. "
        f"Checked repository root {repository_root} and simulation root {simulation_root}."
    )

from utilities.math import (
    quat_apply_inverse_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_from_rotmat_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_rotate_inverse_numpy,
    quat_unique_wxyz,
    euler_xyz_from_quat_wxyz,
    quat_from_yaw_wxyz,
)
from utilities.mujoco_helper import (
    get_sensor_slice,
    make_arrow_mat,
)


class SequentialKeypointsTrajectoryCommandPLBSim:
    """
    Sequential trajectory sampler in PLB frame:
      - follow rows in npy sequentially
      - cubic interpolation between consecutive rows
      - optional hold at each waypoint
      - loop forever
    """

    def __init__(
        self,
        file_path: str,
        control_dt: float,
        traj_duration_s: float = 2.0,
        hold_duration_s: float = 1.0,
    ):
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(f"Expected npy shape (N,9), got {arr.shape} from '{file_path}'.")

        self.table = arr
        self.N = arr.shape[0]

        self.control_dt = float(control_dt)
        self.traj_duration_s = float(traj_duration_s)
        self.hold_duration_s = float(hold_duration_s)

        self.steps_per_traj = max(1, int(round(self.traj_duration_s / self.control_dt)))
        self.steps_per_hold = max(0, int(round(self.hold_duration_s / self.control_dt)))

        self.idx = 0
        self.step = 0
        self.phase = "move"
        self._has_cmd = False

        self.current = self.table[0].copy()
        self.start = self.table[0].copy()
        self.target = self.table[0].copy()

    @property
    def command(self) -> np.ndarray:
        return self.current.copy()

    @staticmethod
    def cubic(t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        return 3.0 * t * t - 2.0 * t * t * t

    def reset(self, initial_kps_plb: np.ndarray, sample_first: bool = True):
        initial_kps_plb = np.asarray(initial_kps_plb, dtype=np.float32).reshape(9,)

        self.idx = 0
        self.step = 0
        self.phase = "move"
        self._has_cmd = True

        self.current = initial_kps_plb.copy()
        self.start = initial_kps_plb.copy()
        self.target = self.table[0].copy() if sample_first else initial_kps_plb.copy()

    def update(self) -> np.ndarray:
        if not self._has_cmd:
            raise RuntimeError("Command sampler not initialized. Call reset() first.")

        if self.phase == "move":
            tau = self.step / max(1, self.steps_per_traj)
            s = self.cubic(tau)

            self.current = ((1.0 - s) * self.start + s * self.target).astype(np.float32)

            self.step += 1
            if self.step >= self.steps_per_traj:
                self.current = self.target.copy()
                self.phase = "hold"
                self.step = 0

        elif self.phase == "hold":
            self.current = self.target.copy()
            self.step += 1

            if self.step >= self.steps_per_hold:
                self.phase = "move"
                self.step = 0
                self.idx = (self.idx + 1) % self.N
                self.start = self.target.copy()
                self.target = self.table[self.idx].copy()

        return self.current.copy()


class PresampledKeypointsDirectCommandPLBSim:
    """
    Directly sampled EE keypoint command in PLB frame.

    PLB frame:
      - origin = [base_x, base_y, ground_z]
      - orientation = yaw-only(base_quat)

    Behavior:
      1) Randomly sample one row from a presampled reachable PLB keypoint table
      2) Hold this sampled keypoint command until next resampling
      3) No cubic interpolation
      4) No adjacent-target threshold clipping
    """

    def __init__(
        self,
        file_path: str,
        control_dt: float,
        cycle_duration_s: float = 8.0,
        seed: int = 0,
    ):
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(f"Expected npy shape (N,9), got {arr.shape} from '{file_path}'.")

        self._table = arr
        self._num_rows = int(arr.shape[0])
        self._control_dt = float(control_dt)
        self._cycle_duration_s = float(cycle_duration_s)
        if self._cycle_duration_s <= 0.0:
            raise ValueError(f"Invalid cycle_duration_s={self._cycle_duration_s}")

        self._cycle_steps = max(1, int(round(self._cycle_duration_s / self._control_dt)))
        self._rng = np.random.default_rng(seed)

        self._has_cmd = False
        self._step_in_cycle = 0
        self.keypoints_command_plb = np.zeros(9, dtype=np.float32)

    @property
    def command(self) -> np.ndarray:
        return self.keypoints_command_plb.copy()

    def _pick_index(self) -> int:
        return int(self._rng.integers(0, self._num_rows))

    def _sample_new_command(self):
        self.keypoints_command_plb = self._table[self._pick_index()].copy().astype(np.float32)
        self._step_in_cycle = 0
        self._has_cmd = True

    def reset(self, initial_kps_plb: np.ndarray, sample_first: bool = True):
        initial_kps_plb = np.asarray(initial_kps_plb, dtype=np.float32).reshape(9,)
        self._step_in_cycle = 0
        self._has_cmd = True

        if sample_first:
            self._sample_new_command()
        else:
            self.keypoints_command_plb = initial_kps_plb.copy()

    def update(self) -> np.ndarray:
        if not self._has_cmd:
            raise RuntimeError("Command sampler not initialized. Call reset() first.")

        self._step_in_cycle += 1
        if self._step_in_cycle >= self._cycle_steps:
            self._sample_new_command()

        return self.command


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="Path to yaml config")
    parser.add_argument(
        "--mode",
        type=str,
        default="full-policy",
        choices=["pd-stand", "lock-arm-policy", "full-policy"],
        help="Run mode",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without the MuJoCo viewer and without wall-clock pacing. "
             "Default off: the standalone interactive behaviour is unchanged.",
    )
    parser.add_argument(
        "--debug-wheel-frames", action="store_true",
        help="Show colored wheel-link origin and joint-axis markers in the MuJoCo viewer.",
    )
    args = parser.parse_args()
    use_policy = args.mode in ("lock-arm-policy", "full-policy")

    yaml_path = os.path.abspath(args.yaml_path)
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    def resolve_config_file(key: str, *, required: bool = True) -> str:
        """Resolve relative config paths against the repository root, like C1.

        C1 joins every relative YAML path onto unitree_mujoco/, so C2 uses the
        same base for exported_policy/, utilities/ and unitree_robots/ and
        cannot pick up a different copy of the same file.  simulation_root and
        the YAML directory stay as fallbacks for a standalone checkout and for
        custom relative paths.  Absolute paths are used unchanged.
        """
        path = Path(cfg[key]).expanduser()
        if path.is_absolute():
            candidates = [path]
        else:
            candidates = [repository_root / path, simulation_root / path, Path(yaml_path).parent / path]
        candidates = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        if required:
            checked = "\n  ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(f"Config '{key}' does not exist; searched:\n  {checked}")
        # pd-stand intentionally does not need an exported policy.
        return str(candidates[0])

    policy_path = resolve_config_file("policy_path", required=use_policy)
    uan_model_path = resolve_config_file("uan_model_path")
    xml_path = resolve_config_file("xml_path")
    ee_command_path = resolve_config_file("ee_command_path")
    print(f"[C2 XML SELECTED] {xml_path}", flush=True)

    simulation_duration = float(cfg["simulation_duration"])
    simulation_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])
    control_dt = simulation_dt * control_decimation

    history_length = int(cfg["history_length"])
    obs_dim_per_step = int(cfg["obs_dim_per_step"])
    obs_dim = int(cfg["obs_dim"])
    action_dim = int(cfg["action_dim"])

    base_command = np.array(cfg["base_command"], dtype=np.float32)

    root_pos = np.array(cfg["root_pos"], dtype=np.float32)
    root_quat = np.array(cfg["root_quat_wxyz"], dtype=np.float32)

    config_joint_names = list(cfg["config_joint_names"])
    default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)
    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)

    leg_action_scale = float(cfg["leg_action_scale"])
    arm_action_scale = np.array(cfg["arm_action_scale"], dtype=np.float32)

    leg_torque_limits = np.array(cfg["leg_torque_limits"], dtype=np.float32)
    gripper_torque_limit = float(cfg.get("gripper_torque_limit", 30.0))

    # Frozen UAN actuator configuration.  These values must match the
    # Isaac Lab UANActuator used to train the WBC policy.
    uan_device = str(cfg.get("uan_device", "cpu"))
    uan_history_length = int(cfg.get("uan_history_length", 25))
    uan_pos_scale = float(cfg.get("uan_pos_scale", 1.0))
    uan_vel_scale = float(cfg.get("uan_vel_scale", 0.05))
    uan_action_scale = float(cfg.get("uan_action_scale", 0.20))
    uan_kp = np.array(cfg["uan_kp"], dtype=np.float32)
    uan_kd = np.array(cfg["uan_kd"], dtype=np.float32)
    uan_nominal_torque_limits = np.array(cfg["uan_nominal_torque_limits"], dtype=np.float32)
    uan_final_torque_limits = np.array(cfg["uan_final_torque_limits"], dtype=np.float32)
    uan_action_saturation_threshold = float(cfg.get("uan_action_saturation_threshold", 0.95))
    # The MuJoCo XML may still contain the legacy Z1 actuator limits (e.g. +/-30 Nm).
    # In WBC-UAN sim2sim the explicit UAN law owns both nominal and final torque
    # saturation, so the MuJoCo-side arm actuator limits must not clip it again.
    sync_mujoco_arm_limits_to_uan = bool(cfg.get("sync_mujoco_arm_limits_to_uan", True))

    ee_body_name = str(cfg.get("ee_body", "gripperStator"))

    ee_kp_dx = float(cfg.get("ee_kp_dx", 0.30))
    ee_kp_dz = float(cfg.get("ee_kp_dz", 0.30))
    ee_ground_z = float(cfg.get("ee_ground_z", cfg.get("ground_z", 0.0)))
    ee_command_seed = int(cfg.get("ee_command_seed", 0))
    ee_cycle_duration_s = float(cfg.get("ee_cycle_duration_s", 8.0))

    startup_hold_s = float(cfg.get("startup_hold_s", 1.0))
    startup_blend_s = float(cfg.get("startup_blend_s", 2.0))

    vis_ee_target = bool(cfg.get("vis_ee_target", True))
    vis_ee_target_axis_len = float(cfg.get("vis_ee_target_axis_len", 0.20))
    vis_ee_target_axis_radius = float(cfg.get("vis_ee_target_axis_radius", 0.01))
    vis_ee_target_sphere_size = float(cfg.get("vis_ee_target_sphere_size", 0.03))
    vis_ee_target_alpha = float(cfg.get("vis_ee_target_alpha", 0.9))
    vis_ee_target_show_current = bool(cfg.get("vis_ee_target_show_current", True))

    assert history_length == 5, f"C2 requires history_length=5, got {history_length}"
    assert obs_dim_per_step == 72, f"Expected obs_dim_per_step=72, got {obs_dim_per_step}"
    assert obs_dim == obs_dim_per_step * history_length, f"Expected obs_dim={obs_dim_per_step * history_length}, got {obs_dim}"
    assert action_dim == 18, f"Expected action_dim=18, got {action_dim}"
    assert len(config_joint_names) == len(set(config_joint_names)) == 19
    assert len(default_joint_pos) == len(kps) == len(kds) == 19
    assert len(leg_torque_limits) == 12
    assert arm_action_scale.shape == (6,), f"Expected arm_action_scale shape (6,), got {arm_action_scale.shape}"
    assert uan_history_length == 25, f"Expected UAN history_length=25, got {uan_history_length}"
    assert uan_kp.shape == (6,), f"Expected uan_kp shape (6,), got {uan_kp.shape}"
    assert uan_kd.shape == (6,), f"Expected uan_kd shape (6,), got {uan_kd.shape}"
    assert uan_nominal_torque_limits.shape == (6,)
    assert uan_final_torque_limits.shape == (6,)
    if np.any(uan_final_torque_limits < uan_nominal_torque_limits):
        raise ValueError("UAN final torque limits must be >= nominal torque limits.")
    if abs(simulation_dt - 0.004) > 1e-9 or control_decimation != 5:
        raise ValueError(
            f"WBC-UAN timing mismatch: expected simulation_dt=0.004 and control_decimation=5, "
            f"got {simulation_dt} and {control_decimation}."
        )

    ee_command_mode = str(cfg.get("ee_command_mode", "direct"))

    print("=" * 80)
    print("C2 B2WZ1 LEGGED-only WBC-UAN (MuJoCo sim2sim, PLB)")
    print("=" * 80)
    print(f"Mode:            {args.mode}")
    print(f"Policy:          {policy_path}")
    print(f"UAN model:       {uan_model_path}")
    print(f"XML:             {xml_path}")
    print(f"Physics/UAN:     {1.0 / simulation_dt:.1f} Hz")
    print(f"WBC control:     {1.0 / control_dt:.1f} Hz")
    print(f"Obs per step:    {obs_dim_per_step}")
    print(f"Obs stacked dim: {obs_dim}")
    print(f"Action dim:      {action_dim}")
    print(f"Leg scale:       {leg_action_scale}")
    print(f"Arm scales:      {arm_action_scale}")
    print("Arm target RL:   6D arm position offsets in full-policy mode")
    print("Wheel actuation: mechanically locked (no wheel DOFs)")
    print(f"UAN history:     {uan_history_length} x {simulation_dt * 1000.0:.1f} ms = "
          f"{uan_history_length * simulation_dt * 1000.0:.1f} ms")
    print(f"UAN Kp:          {uan_kp}")
    print(f"UAN Kd:          {uan_kd}")
    print(f"UAN nominal cap: {uan_nominal_torque_limits}")
    print(f"UAN final cap:   {uan_final_torque_limits}")
    print(f"EE mode:         {ee_command_mode}")
    print(f"EE frame:        PLB")
    print(f"EE ground_z:     {ee_ground_z:.3f}")
    print(f"EE path:         {ee_command_path}")
    if ee_command_mode == "direct":
        print(f"EE cycle:        {ee_cycle_duration_s:.2f}s")
        print("EE behavior:     direct sample-and-hold")
    elif ee_command_mode == "sequential":
        print(
            f"EE cycle:        "
            f"{float(cfg.get('ee_sequential_traj_duration_s', 2.0)) + float(cfg.get('ee_sequential_hold_duration_s', 1.0)):.2f}s"
        )
        print("EE behavior:     sequential cubic interpolation")
    else:
        raise ValueError(f"Unsupported ee_command_mode: {ee_command_mode}")
    print(f"Vis EE target:   {vis_ee_target}")
    print("=" * 80)

    sess = None
    input_name = None
    output_name = None
    if use_policy:
        sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        print("ONNX loaded:")
        print("  input :", input_name, sess.get_inputs()[0].shape)
        print("  output:", output_name, sess.get_outputs()[0].shape)
        input_shape = sess.get_inputs()[0].shape
        output_shape = sess.get_outputs()[0].shape
        if isinstance(input_shape[-1], int) and input_shape[-1] != obs_dim:
            raise ValueError(f"C2 ONNX expects input {obs_dim}D, got {input_shape}")
        if isinstance(output_shape[-1], int) and output_shape[-1] != action_dim:
            raise ValueError(f"C2 ONNX expects output {action_dim}D, got {output_shape}")
    else:
        print("Policy disabled in pd-stand mode.")

    # The exported UAN wrapper already contains the actor observation
    # normalizer and deterministic Beta-mean inference path.
    if uan_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"UAN device '{uan_device}' requested but CUDA is unavailable.")
    uan_model = torch.jit.load(uan_model_path, map_location=uan_device).eval()
    for p in uan_model.parameters():
        p.requires_grad_(False)
    with torch.inference_mode():
        dummy_u = torch.zeros((1, 6 * uan_history_length * 2), dtype=torch.float32, device=uan_device)
        dummy_out = uan_model(dummy_u)
        if isinstance(dummy_out, (tuple, list)):
            dummy_out = dummy_out[0]
        if tuple(dummy_out.shape) != (1, 6):
            raise ValueError(f"Expected UAN output shape (1,6), got {tuple(dummy_out.shape)}")
        if not torch.isfinite(dummy_out).all():
            raise ValueError("UAN returned non-finite output for zero-history smoke test.")
    print("UAN TorchScript loaded:")
    print(f"  zero-history action: {dummy_out.detach().cpu().numpy()[0]}")
    print("=" * 80)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Report the file the model ACTUALLY loaded.  Do not require an exact
    # ground geom name: older C2 files use "ground" rather than "floor".
    # Paths now resolve against the repository root only, but still list every
    # copy of the C2 XML on disk so a stale duplicate cannot hide.
    import hashlib
    xml_sha256 = hashlib.sha256(Path(xml_path).read_bytes()).hexdigest()
    print(f"[C2 XML AUDIT] loaded={xml_path}", flush=True)
    print(f"[C2 XML AUDIT] sha256={xml_sha256}", flush=True)
    for root in (repository_root, simulation_root):
        candidate = (root / "unitree_robots" / "b2wz1" / "b2wz1_legged.xml").resolve()
        if not candidate.is_file():
            continue
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        tag = "loaded" if str(candidate) == str(Path(xml_path).resolve()) else "UNUSED COPY"
        print(f"[C2 XML AUDIT] {tag}: {candidate} sha256[:16]={digest[:16]}", flush=True)
        if tag != "loaded" and digest != xml_sha256:
            print("[C2 XML WARNING] A different copy of b2wz1_legged.xml exists at "
                  f"{candidate}; it is NOT the file being simulated.", flush=True)
    plane_ids = [
        gid for gid in range(m.ngeom)
        if int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
        and int(m.geom_bodyid[gid]) == 0
    ]
    if len(plane_ids) != 1:
        raise RuntimeError(
            f"Expected exactly one worldbody ground plane; found {len(plane_ids)} "
            f"in loaded XML {xml_path}. Check the *actual* selected file."
        )
    floor_gid = plane_ids[0]
    floor_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, floor_gid)
    floor_friction = m.geom_friction[floor_gid].tolist()
    floor_matid = int(m.geom_matid[floor_gid])
    print(
        "[C2 SCENE AUDIT] "
        f"ground_name={floor_name!r} | floor_id={floor_gid} | "
        f"geomgroup={int(m.geom_group[floor_gid])} | "
        f"friction={floor_friction} | material_id={floor_matid}", flush=True,
    )
    if floor_matid < 0:
        print("[C2 SCENE WARNING] Ground has no checkerboard material. "
              "Likely loaded an old C2 XML revision, not the C1-aligned ground.", flush=True)
    if not np.allclose(floor_friction, [0.8, 0.2, 0.1], atol=1e-5):
        print("[C2 SCENE WARNING] Ground friction differs from C1 [0.8,0.2,0.1]. "
              "Replace the XML at the printed loaded= path.", flush=True)
    # Inspect wheel visual and collision mesh assets, not merely wheel-body
    # origins.  A fixed wheel should still reference the SAME STL mesh as C1.
    for prefix in ("FL", "FR", "RL", "RR"):
        body_name = f"{prefix}_wheel_link"
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise RuntimeError(f"Missing wheel body: {body_name}")
        mesh_names = []
        for gid in range(m.ngeom):
            if int(m.geom_bodyid[gid]) != bid or int(m.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh_id = int(m.geom_dataid[gid])
            mesh_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mesh_id))
        print(f"[C2 WHEEL AUDIT] {body_name} mesh_assets={mesh_names}", flush=True)
        if mesh_names != [f"{prefix}_wheel", f"{prefix}_wheel"]:
            print(f"[C2 WHEEL WARNING] Expected C1 {prefix}_wheel mesh twice "
                  "(visual+collision). A stale foot.obj XML may be loaded.", flush=True)

    ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    if ee_bid < 0:
        raise ValueError(f"Body not found: {ee_body_name}")
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro") < 0:
        raise ValueError("MuJoCo XML is missing required 'imu_gyro' sensor")

    # Isaac Lab C2 play.py, 2026-11-03: 19 articulation joints, no wheel DOFs.
    # ActionTerm group order is legs(12) followed by arm(6); within-term
    # resolved joint ordering is validated against C1's known policy mapping.
    # If play.py prints a different per-term order, edit these lists ONLY.
    leg_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    gripper_joint_name = "jointGripper"

    policy_joint_pos_names = leg_joint_names + arm_joint_names
    policy_joint_vel_names = leg_joint_names + arm_joint_names
    policy_action_names = leg_joint_names + arm_joint_names
    expected_joint_names = set(policy_action_names + [gripper_joint_name])

    if set(config_joint_names) != expected_joint_names:
        raise ValueError(
            f"YAML config_joint_names mismatch: missing={expected_joint_names-set(config_joint_names)}, "
            f"unexpected={set(config_joint_names)-expected_joint_names}"
        )
    param_id_by_name = {name: i for i, name in enumerate(config_joint_names)}
    def config_values(values, names):
        return np.asarray([values[param_id_by_name[n]] for n in names], dtype=np.float32)

    # Use MuJoCo's addresses rather than assuming the original 23-DOF ordering
    # or its hardcoded actuator positions. The free root is checked separately.
    jnt_id_by_name = {}
    for jid in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is not None:
            jnt_id_by_name[name] = jid

    actual_movables = {
        name for name, jid in jnt_id_by_name.items()
        if int(m.jnt_type[jid]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    }
    if actual_movables != expected_joint_names:
        raise RuntimeError(
            "Incorrect C2 MJCF joint set. Wheels must be physically welded, not passive. "
            f"Missing={sorted(expected_joint_names-actual_movables)}, "
            f"extra={sorted(actual_movables-expected_joint_names)}. "
            "Use the fixed-wheel b2wz1_legged.xml supplied with this C2 package."
        )
    if m.nq != 7 + 19 or m.nv != 6 + 19:
        raise RuntimeError(f"Expected C2 free-base nq=26,nv=25, got nq={m.nq},nv={m.nv}")

    qpos_id_by_name = {name: int(m.jnt_qposadr[jnt_id_by_name[name]]) for name in expected_joint_names}
    qvel_id_by_name = {name: int(m.jnt_dofadr[jnt_id_by_name[name]]) for name in expected_joint_names}
    def get_qpos(names):
        return np.asarray([d.qpos[qpos_id_by_name[n]] for n in names], dtype=np.float32)
    def get_qvel(names):
        return np.asarray([d.qvel[qvel_id_by_name[n]] for n in names], dtype=np.float32)

    actuator_index_by_joint_name = {}
    for ai in range(m.nu):
        if int(m.actuator_trntype[ai]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise RuntimeError(f"C2 expects joint-transmission actuators, got index {ai}")
        jid = int(m.actuator_trnid[ai, 0])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in actuator_index_by_joint_name:
            raise RuntimeError(f"More than one actuator controls {name}")
        actuator_index_by_joint_name[name] = ai
    if set(actuator_index_by_joint_name) != expected_joint_names or m.nu != 19:
        raise RuntimeError(
            f"Expected 19 torque actuators for {sorted(expected_joint_names)}; "
            f"found {m.nu}: {sorted(actuator_index_by_joint_name)}"
        )
    # The original C1 runner sends joint torques directly into d.ctrl.
    # Enforce identical motor semantics for the converted C2 model.
    for name, ai in actuator_index_by_joint_name.items():
        if (int(m.actuator_dyntype[ai]) != int(mujoco.mjtDyn.mjDYN_NONE)
                or int(m.actuator_gaintype[ai]) != int(mujoco.mjtGain.mjGAIN_FIXED)
                or int(m.actuator_biastype[ai]) != int(mujoco.mjtBias.mjBIAS_NONE)
                or not np.isclose(m.actuator_gear[ai, 0], 1.0)):
            raise RuntimeError(
                f"Actuator for {name} is not a unit-gear torque motor; "
                "check C1 XML actuator definitions before deployment."
            )

    leg_action_indices = slice(0, 12)
    arm_action_indices = slice(12, 18)

    default_joint_pos_policy_pos = config_values(default_joint_pos, policy_joint_pos_names)
    default_leg_pos = config_values(default_joint_pos, leg_joint_names)
    default_arm_pos = config_values(default_joint_pos, arm_joint_names)
    default_gripper_pos = float(config_values(default_joint_pos, [gripper_joint_name])[0])
    leg_kps = config_values(kps, leg_joint_names)
    leg_kds = config_values(kds, leg_joint_names)
    gripper_kp = float(config_values(kps, [gripper_joint_name])[0])
    gripper_kd = float(config_values(kds, [gripper_joint_name])[0])

    # The XML was originally authored for the old arm PD path and may still
    # carry legacy actuator limits (for example +/-30 Nm on joint1).  That would
    # silently clip the explicit UAN torque after we already applied the exact
    # UAN final clamp, creating a sim2sim mismatch.
    #
    # Match Isaac Lab's transparent explicit actuator behavior: the UAN block
    # owns the torque law and final safety clamp.  Keep MuJoCo's redundant
    # actuator/joint limits synchronized to the same final envelope.
    print("MuJoCo arm torque-limit audit:")
    for j, joint_name in enumerate(arm_joint_names):
        ctrl_i = actuator_index_by_joint_name[joint_name]
        final_lim = float(uan_final_torque_limits[j])

        if bool(m.actuator_ctrllimited[ctrl_i]):
            old_range = m.actuator_ctrlrange[ctrl_i].copy()
            if old_range[0] > -final_lim + 1e-6 or old_range[1] < final_lim - 1e-6:
                if not sync_mujoco_arm_limits_to_uan:
                    raise RuntimeError(
                        f"MuJoCo actuator ctrlrange for {joint_name} is "
                        f"[{old_range[0]}, {old_range[1]}], narrower than "
                        f"UAN final +/-{final_lim} Nm."
                    )
                m.actuator_ctrlrange[ctrl_i, :] = (-final_lim, final_lim)
                print(
                    f"  {joint_name}: ctrlrange "
                    f"[{old_range[0]:.1f}, {old_range[1]:.1f}] -> "
                    f"[-{final_lim:.1f}, +{final_lim:.1f}]"
                )

        if bool(m.actuator_forcelimited[ctrl_i]):
            old_range = m.actuator_forcerange[ctrl_i].copy()
            if old_range[0] > -final_lim + 1e-6 or old_range[1] < final_lim - 1e-6:
                if not sync_mujoco_arm_limits_to_uan:
                    raise RuntimeError(
                        f"MuJoCo actuator forcerange for {joint_name} is "
                        f"[{old_range[0]}, {old_range[1]}], narrower than "
                        f"UAN final +/-{final_lim} Nm."
                    )
                m.actuator_forcerange[ctrl_i, :] = (-final_lim, final_lim)
                print(
                    f"  {joint_name}: forcerange "
                    f"[{old_range[0]:.1f}, {old_range[1]:.1f}] -> "
                    f"[-{final_lim:.1f}, +{final_lim:.1f}]"
                )

        # Some MuJoCo models also impose a joint-level actuator-force limit.
        # Widen that redundant limit as well if present.
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if (
            joint_id >= 0
            and hasattr(m, "jnt_actfrclimited")
            and hasattr(m, "jnt_actfrcrange")
            and bool(m.jnt_actfrclimited[joint_id])
        ):
            old_range = m.jnt_actfrcrange[joint_id].copy()
            if old_range[0] > -final_lim + 1e-6 or old_range[1] < final_lim - 1e-6:
                if not sync_mujoco_arm_limits_to_uan:
                    raise RuntimeError(
                        f"MuJoCo joint actuator-force range for {joint_name} is "
                        f"[{old_range[0]}, {old_range[1]}], narrower than "
                        f"UAN final +/-{final_lim} Nm."
                    )
                m.jnt_actfrcrange[joint_id, :] = (-final_lim, final_lim)
                print(
                    f"  {joint_name}: joint actfrcrange "
                    f"[{old_range[0]:.1f}, {old_range[1]:.1f}] -> "
                    f"[-{final_lim:.1f}, +{final_lim:.1f}]"
                )

    # Final verification: no MuJoCo-side arm limit may be narrower than the
    # explicit UAN final clamp.
    for j, joint_name in enumerate(arm_joint_names):
        ctrl_i = actuator_index_by_joint_name[joint_name]
        final_lim = float(uan_final_torque_limits[j])
        if bool(m.actuator_ctrllimited[ctrl_i]):
            lo, hi = m.actuator_ctrlrange[ctrl_i]
            assert lo <= -final_lim + 1e-6 and hi >= final_lim - 1e-6
        if bool(m.actuator_forcelimited[ctrl_i]):
            lo, hi = m.actuator_forcerange[ctrl_i]
            assert lo <= -final_lim + 1e-6 and hi >= final_lim - 1e-6
        jid = jnt_id_by_name[joint_name]
        if hasattr(m, "jnt_actfrclimited") and bool(m.jnt_actfrclimited[jid]):
            lo, hi = m.jnt_actfrcrange[jid]
            assert lo <= -final_lim + 1e-6 and hi >= final_lim - 1e-6
    print("  PASS: MuJoCo will not clip inside the UAN final torque envelope.")

    print("Joint mapping (C2, fixed wheels):")
    print(f"  YAML parameter order  : {config_joint_names}")
    print(f"  MuJoCo joint addresses: {qpos_id_by_name}")
    print(f"  MuJoCo actuator order : {actuator_index_by_joint_name}")
    print(f"  Policy joint_pos order: {policy_joint_pos_names}")
    print(f"  Policy joint_vel order: {policy_joint_vel_names}")
    print(f"  Policy action order   : {policy_action_names}")
    print("=" * 80)

    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0
    d.qpos[0:3] = root_pos
    d.qpos[3:7] = root_quat
    for name in config_joint_names:
        d.qpos[qpos_id_by_name[name]] = default_joint_pos[param_id_by_name[name]]

    mujoco.mj_forward(m, d)
    print(f"Initialized height z = {d.qpos[2]:.3f} m")
    wheel_body_ids = {}
    for wheel_prefix in ("FL", "FR", "RL", "RR"):
        wheel_name = f"{wheel_prefix}_wheel_link"
        wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, wheel_name)
        if wheel_id < 0:
            raise RuntimeError(f"C2 fixed-wheel model is missing body {wheel_name}")
        wheel_body_ids[wheel_prefix] = wheel_id
        print(
            f"[C2 WHEEL FRAME] {wheel_name}: world origin "
            f"{d.xpos[wheel_id].round(5).tolist()} "
            "(red sphere in --debug-wheel-frames mode)",
            flush=True,
        )

    def get_projected_level_base_pose_world():
        base_pos_w = d.qpos[0:3].copy().astype(np.float32)
        base_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))

        _, _, yaw = euler_xyz_from_quat_wxyz(base_quat_w)
        plb_quat_w = quat_from_yaw_wxyz(yaw)
        plb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(plb_quat_w))

        plb_pos_w = base_pos_w.copy()
        plb_pos_w[2] = ee_ground_z

        return plb_pos_w, plb_quat_w

    def compute_ee_current_kp_plb() -> np.ndarray:
        plb_pos_w, plb_quat_w = get_projected_level_base_pose_world()

        ee_pos_w = d.xpos[ee_bid].copy().astype(np.float32)
        ee_rot_w = d.xmat[ee_bid].reshape(3, 3).copy().astype(np.float32)
        ee_quat_w = quat_from_rotmat_wxyz(ee_rot_w)

        ee_pos_plb = quat_apply_inverse_wxyz(plb_quat_w, ee_pos_w - plb_pos_w)
        ee_quat_plb = quat_mul_wxyz(quat_conjugate_wxyz(plb_quat_w), ee_quat_w)
        ee_quat_plb = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_plb))

        off_x = np.array([ee_kp_dx, 0.0, 0.0], dtype=np.float32)
        off_z = np.array([0.0, 0.0, ee_kp_dz], dtype=np.float32)

        kp0 = ee_pos_plb
        kp1 = ee_pos_plb + quat_apply_wxyz(ee_quat_plb, off_x)
        kp2 = ee_pos_plb + quat_apply_wxyz(ee_quat_plb, off_z)
        return np.concatenate([kp0, kp1, kp2]).astype(np.float32)

    def ee_kp_plb_to_world(ee_kp_plb: np.ndarray):
        plb_pos_w, plb_quat_w = get_projected_level_base_pose_world()

        kp0_plb = ee_kp_plb[0:3].astype(np.float32)
        kp1_plb = ee_kp_plb[3:6].astype(np.float32)
        kp2_plb = ee_kp_plb[6:9].astype(np.float32)

        kp0_w = plb_pos_w + quat_apply_wxyz(plb_quat_w, kp0_plb)
        kp1_w = plb_pos_w + quat_apply_wxyz(plb_quat_w, kp1_plb)
        kp2_w = plb_pos_w + quat_apply_wxyz(plb_quat_w, kp2_plb)

        return kp0_w, kp1_w, kp2_w

    def add_capsule_to_scene(scene, p0, p1, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return

        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        diff = p1 - p0
        length = np.linalg.norm(diff)
        if length < 1e-8:
            return

        pos = 0.5 * (p0 + p1)
        mat = make_arrow_mat(diff)

        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([radius, 0.5 * length, 0.0], dtype=np.float64),
            pos,
            mat.reshape(-1),
            np.array(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    def add_sphere_to_scene(scene, pos, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return

        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0], dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.array(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    def visualize_ee_pose(
        scene,
        ee_kp_plb: np.ndarray,
        axis_len: float,
        axis_radius: float,
        sphere_size: float,
        alpha: float,
        pos_rgba,
        x_rgba,
        z_rgba,
    ):
        kp0_w, kp1_w, kp2_w = ee_kp_plb_to_world(ee_kp_plb)

        x_dir = kp1_w - kp0_w
        z_dir = kp2_w - kp0_w

        x_norm = np.linalg.norm(x_dir)
        z_norm = np.linalg.norm(z_dir)
        if x_norm < 1e-8 or z_norm < 1e-8:
            return

        x_dir = x_dir / x_norm
        z_dir = z_dir / z_norm

        x_end = kp0_w + axis_len * x_dir
        z_end = kp0_w + axis_len * z_dir

        add_sphere_to_scene(scene, kp0_w, sphere_size, [pos_rgba[0], pos_rgba[1], pos_rgba[2], alpha])
        add_capsule_to_scene(scene, kp0_w, x_end, axis_radius, [x_rgba[0], x_rgba[1], x_rgba[2], alpha])
        add_capsule_to_scene(scene, kp0_w, z_end, axis_radius, [z_rgba[0], z_rgba[1], z_rgba[2], alpha])

    def update_custom_visualization(viewer, ee_cmd_plb: np.ndarray):
        viewer.user_scn.ngeom = 0

        if args.debug_wheel_frames:
            # Mark the true wheel-link frame (not the visual mesh's centroid).
            # Red spheres should sit at the physical axle origins; yellow
            # capsules show the local wheel rotation axis (+/- Y).
            for wheel_id in wheel_body_ids.values():
                center = d.xpos[wheel_id].copy()
                axis_y = d.xmat[wheel_id].reshape(3, 3)[:, 1].copy()
                add_sphere_to_scene(viewer.user_scn, center, 0.025, [1., 0.05, 0.05, 1.])
                add_capsule_to_scene(
                    viewer.user_scn, center - 0.11 * axis_y,
                    center + 0.11 * axis_y, 0.007, [1., 0.85, 0.05, 1.],
                )

        if not vis_ee_target:
            return

        visualize_ee_pose(
            viewer.user_scn,
            ee_kp_plb=ee_cmd_plb,
            axis_len=vis_ee_target_axis_len,
            axis_radius=vis_ee_target_axis_radius,
            sphere_size=vis_ee_target_sphere_size,
            alpha=vis_ee_target_alpha,
            pos_rgba=[1.0, 0.0, 0.0, vis_ee_target_alpha],
            x_rgba=[1.0, 0.0, 0.0, vis_ee_target_alpha],
            z_rgba=[1.0, 0.0, 0.0, vis_ee_target_alpha],
        )

        if vis_ee_target_show_current:
            ee_cur_plb = compute_ee_current_kp_plb()
            visualize_ee_pose(
                viewer.user_scn,
                ee_kp_plb=ee_cur_plb,
                axis_len=vis_ee_target_axis_len * 0.85,
                axis_radius=vis_ee_target_axis_radius * 0.8,
                sphere_size=vis_ee_target_sphere_size * 0.85,
                alpha=min(1.0, vis_ee_target_alpha),
                pos_rgba=[0.0, 0.0, 1.0, vis_ee_target_alpha],
                x_rgba=[0.0, 0.0, 1.0, vis_ee_target_alpha],
                z_rgba=[0.0, 0.0, 1.0, vis_ee_target_alpha],
            )

    if ee_command_mode == "sequential":
        ee_cmd_sampler = SequentialKeypointsTrajectoryCommandPLBSim(
            file_path=ee_command_path,
            control_dt=control_dt,
            traj_duration_s=float(cfg.get("ee_sequential_traj_duration_s", 2.0)),
            hold_duration_s=float(cfg.get("ee_sequential_hold_duration_s", 1.0)),
        )
    elif ee_command_mode == "direct":
        ee_cmd_sampler = PresampledKeypointsDirectCommandPLBSim(
            file_path=ee_command_path,
            control_dt=control_dt,
            cycle_duration_s=ee_cycle_duration_s,
            seed=ee_command_seed,
        )
    else:
        raise ValueError(f"Unsupported ee_command_mode: {ee_command_mode}")

    ee_cur_init_plb = compute_ee_current_kp_plb()
    sample_first = False if args.mode == "pd-stand" else True
    ee_cmd_sampler.reset(initial_kps_plb=ee_cur_init_plb, sample_first=sample_first)
    ee_cmd_plb_current = ee_cmd_sampler.command.copy()

    last_action = np.zeros(action_dim, dtype=np.float32)

    # UAN history exactly matches training/UANActuator layout:
    # (joint, time, feature), joint-major, newest sample last.  Reset to zero
    # rather than pre-filling; this preserves the frozen model semantics.
    uan_history = np.zeros((6, uan_history_length, 2), dtype=np.float32)
    last_uan_action = np.zeros(6, dtype=np.float32)
    last_uan_delta_tau = np.zeros(6, dtype=np.float32)
    last_uan_tau_nom = np.zeros(6, dtype=np.float32)
    last_uan_tau_pd = np.zeros(6, dtype=np.float32)

    def compute_uan_arm_torque(q_des: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Run one 250 Hz UAN actuator step and return six arm torques."""
        q_des = np.asarray(q_des, dtype=np.float32).reshape(6,)
        q = np.asarray(q, dtype=np.float32).reshape(6,)
        qd = np.asarray(qd, dtype=np.float32).reshape(6,)

        pos_err = (q_des - q) * uan_pos_scale
        vel = qd * uan_vel_scale

        uan_history[:, :-1, :] = uan_history[:, 1:, :]
        uan_history[:, -1, 0] = pos_err
        uan_history[:, -1, 1] = vel

        obs_np = np.ascontiguousarray(uan_history.reshape(1, -1), dtype=np.float32)
        obs_t = torch.from_numpy(obs_np).to(device=uan_device)
        with torch.inference_mode():
            out = uan_model(obs_t)
            if isinstance(out, (tuple, list)):
                out = out[0]
        action = out.detach().cpu().numpy().reshape(6,).astype(np.float32)

        if not np.all(np.isfinite(action)):
            raise FloatingPointError(f"Non-finite UAN action: {action}")
        if np.max(np.abs(action)) > 1.001:
            raise RuntimeError(
                f"UAN action left expected [-1,1] Beta-mean range: {action}"
            )

        tau_pd = uan_kp * (q_des - q) - uan_kd * qd
        tau_nom = np.clip(tau_pd, -uan_nominal_torque_limits, uan_nominal_torque_limits)
        delta_tau = action * uan_action_scale * uan_kp
        tau = np.clip(
            tau_nom + delta_tau,
            -uan_final_torque_limits,
            uan_final_torque_limits,
        ).astype(np.float32)

        last_uan_action[:] = action
        last_uan_delta_tau[:] = delta_tau
        last_uan_tau_nom[:] = tau_nom
        last_uan_tau_pd[:] = tau_pd
        return tau

    def build_obs_step(ee_cmd_plb: np.ndarray) -> np.ndarray:
        base_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)

        base_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(base_quat_w, gravity_w).astype(np.float32)

        joint_pos_policy = get_qpos(policy_joint_pos_names)
        joint_pos_rel = joint_pos_policy - default_joint_pos_policy_pos
        joint_pos_leg_rel = joint_pos_rel[:12]
        joint_pos_arm_rel = joint_pos_rel[12:18]

        joint_vel_policy = get_qvel(policy_joint_vel_names)
        joint_vel_leg = joint_vel_policy[:12]
        joint_vel_arm = joint_vel_policy[12:18]

        obs = np.concatenate(
            [
                base_ang_vel_b,       # 3
                projected_gravity_b,  # 3
                base_command,         # 3
                ee_cmd_plb,           # 9
                joint_pos_leg_rel,    # 12
                joint_pos_arm_rel,    # 6
                joint_vel_leg,        # 12
                joint_vel_arm,        # 6
                last_action,          # 18
            ],
            dtype=np.float32,
        )
        assert obs.shape[0] == obs_dim_per_step, f"Obs dim mismatch: {obs.shape[0]} vs {obs_dim_per_step}"
        return obs

    leg_target = default_leg_pos.copy()
    arm_target = default_arm_pos.copy()
    gripper_target = default_gripper_pos

    obs0 = build_obs_step(ee_cmd_plb_current)

    i = 0
    obs0_base_ang_vel = obs0[i:i + 3]; i += 3
    obs0_projected_gravity = obs0[i:i + 3]; i += 3
    obs0_base_cmd = obs0[i:i + 3]; i += 3
    obs0_ee_cmd = obs0[i:i + 9]; i += 9
    obs0_joint_pos_leg = obs0[i:i + 12]; i += 12
    obs0_joint_pos_arm = obs0[i:i + 6]; i += 6
    obs0_joint_vel_leg = obs0[i:i + 12]; i += 12
    obs0_joint_vel_arm = obs0[i:i + 6]; i += 6
    obs0_last_action = obs0[i:i + 18]; i += 18

    base_ang_vel_hist = deque(maxlen=history_length)
    projected_gravity_hist = deque(maxlen=history_length)
    base_cmd_hist = deque(maxlen=history_length)
    ee_cmd_hist = deque(maxlen=history_length)
    joint_pos_leg_hist = deque(maxlen=history_length)
    joint_pos_arm_hist = deque(maxlen=history_length)
    joint_vel_leg_hist = deque(maxlen=history_length)
    joint_vel_arm_hist = deque(maxlen=history_length)
    last_action_hist = deque(maxlen=history_length)

    for _ in range(history_length):
        base_ang_vel_hist.append(obs0_base_ang_vel.copy())
        projected_gravity_hist.append(obs0_projected_gravity.copy())
        base_cmd_hist.append(obs0_base_cmd.copy())
        ee_cmd_hist.append(obs0_ee_cmd.copy())
        joint_pos_leg_hist.append(obs0_joint_pos_leg.copy())
        joint_pos_arm_hist.append(obs0_joint_pos_arm.copy())
        joint_vel_leg_hist.append(obs0_joint_vel_leg.copy())
        joint_vel_arm_hist.append(obs0_joint_vel_arm.copy())
        last_action_hist.append(obs0_last_action.copy())

    counter = 0
    policy_tick = 0
    sim_time = 0.0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # MuJoCo viewer can hide geom group 0 independently of physics.
        # Explicitly restore C1/C2 floor (group 0), visual bodies (1, 2).
        with viewer.lock():
            viewer.opt.geomgroup[0] = 1
            viewer.opt.geomgroup[1] = 1
            viewer.opt.geomgroup[2] = 1
        print(
            "[C2 VIEWER AUDIT] geom groups 0/1/2 enabled; "
            "ground plane should be visible if the model and lighting are correct.",
            flush=True,
        )
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = d.qpos[:3]

        while (viewer is None or viewer.is_running()) and sim_time < simulation_duration:
            step_start = time.time()

            if sim_time < startup_hold_s:
                blend = 0.0
            elif sim_time < startup_hold_s + startup_blend_s:
                blend = (sim_time - startup_hold_s) / startup_blend_s
            else:
                blend = 1.0
            blend = float(np.clip(blend, 0.0, 1.0))

            leg_pos = get_qpos(leg_joint_names)
            leg_vel = get_qvel(leg_joint_names)
            arm_pos = get_qpos(arm_joint_names)
            arm_vel = get_qvel(arm_joint_names)
            gripper_pos = float(get_qpos([gripper_joint_name])[0])
            gripper_vel = float(get_qvel([gripper_joint_name])[0])

            leg_tau = leg_kps * (leg_target - leg_pos) - leg_kds * leg_vel
            leg_tau = np.clip(leg_tau, -leg_torque_limits, leg_torque_limits).astype(np.float32)

            # Six-axis Z1 arm uses the frozen UAN actuator at every physics step.
            # q_des is the latest 50 Hz WBC target and is held across five steps.
            arm_tau = compute_uan_arm_torque(arm_target, arm_pos, arm_vel)

            gripper_tau = gripper_kp * (gripper_target - gripper_pos) - gripper_kd * gripper_vel
            gripper_tau = float(np.clip(gripper_tau, -gripper_torque_limit, gripper_torque_limit))

            # d.ctrl actuator addresses need not match the RL action order.
            d.ctrl[:] = 0.0
            for name, value in zip(leg_joint_names, leg_tau):
                d.ctrl[actuator_index_by_joint_name[name]] = float(value)
            for name, value in zip(arm_joint_names, arm_tau):
                d.ctrl[actuator_index_by_joint_name[name]] = float(value)
            d.ctrl[actuator_index_by_joint_name[gripper_joint_name]] = gripper_tau

            mujoco.mj_step(m, d)
            if viewer is not None:
                viewer.cam.lookat[:] = d.qpos[:3]
            sim_time += simulation_dt

            if counter % control_decimation == 0:
                if args.mode == "pd-stand":
                    ee_cmd_plb_current = ee_cmd_sampler.command.copy()
                else:
                    ee_cmd_plb_current = ee_cmd_sampler.update()

                obs_step = build_obs_step(ee_cmd_plb_current)

                i = 0
                curr_base_ang_vel = obs_step[i:i + 3]; i += 3
                curr_projected_gravity = obs_step[i:i + 3]; i += 3
                curr_base_cmd = obs_step[i:i + 3]; i += 3
                curr_ee_cmd = obs_step[i:i + 9]; i += 9
                curr_joint_pos_leg = obs_step[i:i + 12]; i += 12
                curr_joint_pos_arm = obs_step[i:i + 6]; i += 6
                curr_joint_vel_leg = obs_step[i:i + 12]; i += 12
                curr_joint_vel_arm = obs_step[i:i + 6]; i += 6
                curr_last_action = obs_step[i:i + 18]; i += 18

                base_ang_vel_hist.append(curr_base_ang_vel.copy())
                projected_gravity_hist.append(curr_projected_gravity.copy())
                base_cmd_hist.append(curr_base_cmd.copy())
                ee_cmd_hist.append(curr_ee_cmd.copy())
                joint_pos_leg_hist.append(curr_joint_pos_leg.copy())
                joint_pos_arm_hist.append(curr_joint_pos_arm.copy())
                joint_vel_leg_hist.append(curr_joint_vel_leg.copy())
                joint_vel_arm_hist.append(curr_joint_vel_arm.copy())
                last_action_hist.append(curr_last_action.copy())

                obs_stack = np.concatenate(
                    [
                        np.array(base_ang_vel_hist).reshape(-1),
                        np.array(projected_gravity_hist).reshape(-1),
                        np.array(base_cmd_hist).reshape(-1),
                        np.array(ee_cmd_hist).reshape(-1),
                        np.array(joint_pos_leg_hist).reshape(-1),
                        np.array(joint_pos_arm_hist).reshape(-1),
                        np.array(joint_vel_leg_hist).reshape(-1),
                        np.array(joint_vel_arm_hist).reshape(-1),
                        np.array(last_action_hist).reshape(-1),
                    ],
                    dtype=np.float32,
                )

                assert obs_stack.shape[0] == obs_dim, f"obs_stack dim mismatch: {obs_stack.shape[0]} vs {obs_dim}"

                if args.mode == "pd-stand":
                    action = np.zeros(action_dim, dtype=np.float32)
                else:
                    action = sess.run([output_name], {input_name: obs_stack[None, :]})[0][0].astype(np.float32)

                if action.shape != (18,) or not np.all(np.isfinite(action)):
                    raise RuntimeError(f"Invalid C2 action: shape={action.shape}, values={action}")
                last_action[:] = action

                leg_act = action[leg_action_indices].copy()
                arm_act = action[arm_action_indices].copy()

                if args.mode == "pd-stand":
                    leg_target = default_leg_pos.copy()
                    arm_target = default_arm_pos.copy()

                elif args.mode == "lock-arm-policy":
                    leg_target = default_leg_pos + blend * (leg_action_scale * leg_act)
                    arm_target = default_arm_pos.copy()

                elif args.mode == "full-policy":
                    leg_target = default_leg_pos + blend * (leg_action_scale * leg_act)
                    # No target-rate limiter: current WBC training uses ordinary
                    # JointPositionActionCfg.  This target is held for five
                    # 250 Hz UAN/physics steps until the next 50 Hz policy tick.
                    arm_target = default_arm_pos + blend * (arm_action_scale * arm_act)

                else:
                    raise ValueError(f"Unsupported mode: {args.mode}")

                policy_tick += 1

            if viewer is not None:
                update_custom_visualization(viewer, ee_cmd_plb_current)

            if counter % 200 == 0:
                current_leg_pos = get_qpos(leg_joint_names)
                current_arm_pos = get_qpos(arm_joint_names)
                leg_err = leg_target - current_leg_pos
                arm_err = arm_target - current_arm_pos
                ee_err_norm = np.linalg.norm((ee_cmd_plb_current - compute_ee_current_kp_plb()).reshape(3, 3), axis=1)
                log_roll, log_pitch, _ = euler_xyz_from_quat_wxyz(
                    quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
                )

                extra_cmd_info = ""
                if ee_command_mode == "direct":
                    extra_cmd_info = f" | cmd_hold={ee_cycle_duration_s:.2f}s"

                print(
                    f"[{counter:6d}] "
                    f"t={sim_time:6.2f}s | "
                    f"z={d.qpos[2]:.3f} | "
                    f"roll={np.rad2deg(log_roll):+6.2f}deg | "
                    f"pitch={np.rad2deg(log_pitch):+6.2f}deg | "
                    f"blend={blend:.2f}{extra_cmd_info} | "
                    f"leg_act=[{last_action[leg_action_indices].min():+.2f},{last_action[leg_action_indices].max():+.2f}] | "
                    f"arm_act=[{last_action[arm_action_indices].min():+.2f},{last_action[arm_action_indices].max():+.2f}]"
                )
                uan_action_sat = float(np.mean(np.abs(last_uan_action) > uan_action_saturation_threshold))
                uan_nominal_clip = float(np.mean(np.abs(last_uan_tau_pd) > uan_nominal_torque_limits))
                uan_delta_abs = float(np.mean(np.abs(last_uan_delta_tau)))
                uan_pos_err_rms = float(np.sqrt(np.mean(arm_err * arm_err)))

                print(
                    f"           max_leg_err={np.max(np.abs(leg_err)):.4f} | "
                    f"max_arm_err={np.max(np.abs(arm_err)):.4f} | "
                    f"ee_err_plb=[{ee_err_norm[0]:.4f},{ee_err_norm[1]:.4f},{ee_err_norm[2]:.4f}] | "
                    f"max_ctrl={np.max(np.abs(d.ctrl[:])):.2f} | "
                    f"ee_cmd_plb=[{ee_cmd_plb_current[0]:.3f},{ee_cmd_plb_current[1]:.3f},{ee_cmd_plb_current[2]:.3f}]"
                )
                print(
                    f"           UAN sat={uan_action_sat:.3f} | "
                    f"nom_clip={uan_nominal_clip:.3f} | "
                    f"delta_tau_abs={uan_delta_abs:.3f} Nm | "
                    f"pos_err_rms={uan_pos_err_rms:.4f} rad | "
                    f"uan_act=[{last_uan_action.min():+.3f},{last_uan_action.max():+.3f}]"
                )

            counter += 1
            if viewer is not None:
                viewer.sync()

                time_until_next_step = simulation_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
