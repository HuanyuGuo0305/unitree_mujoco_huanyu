"""Headless MuJoCo engine for the E5 dynamic object retrieval benchmark.

Control path
------------
This is a line-by-line port of the (aligned) interactive sim2sim
``deploy_mujoco/b2wz1_hl_retrieval_uan.py``: same float32 arithmetic, same
frozen WBC / UAN / HL ONNX execution order (250 Hz physics+UAN, 50 Hz WBC,
10 Hz HL), and the keypoint/Euler helpers are imported from the deploy module
itself. ``tests/test_deploy_parity.py`` runs the unmodified deploy script on the
same compiled model and requires bit-identical WBC commands, HL actions, arm
targets, gripper commands and cube trajectory.

Added around that path (none of it exists in the interactive script):
  * one compiled MuJoCo model per sampled (side, density) cube (MjSpec, unrounded):
    box half extent, mass = density * side^3 (inertia m s^2 / 6 derived by the
    compiler), cube contact parameters, explicit cube<->gripper contact pairs;
  * per-scenario materials reproducing PhysX "multiply" combination:
    robot-ground mu = mu_shape * 1.0, cube-ground mu = mu_cube * 1.0,
    cube-gripper mu = mu_cube * mu_shape (explicit pairs);
  * base mass / COM and reset randomization (Isaac ``EventCfg`` / ``_reset_idx``);
  * optional training-time actor observation noise;
  * contact sensing (gripperStator/gripperMover<->cube, base_link), the task
    state machine of ``metrics.TaskStateMachine``, and a read-only 250-Hz
    physical-retention log (``ph_*`` arrays) evaluated offline by ``retention.py``.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import mujoco
import numpy as np
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_PY_ROOT = os.path.abspath(os.path.join(HERE, ".."))          # simulate_python/
REPO_ROOT = os.path.abspath(os.path.join(HERE, "../.."))          # unitree_mujoco/
for _p in (REPO_ROOT, SIM_PY_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utilities.math import (  # noqa: E402
    euler_xyz_from_quat_wxyz,
    quat_apply_inverse_wxyz,
    quat_apply_wxyz,
    quat_from_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_rotate_inverse_numpy,
    quat_unique_wxyz,
)
from utilities.mujoco_helper import get_sensor_slice  # noqa: E402

# Proven E4 utilities, reused unchanged (read-only import).
from transport_experiment.sim import (  # noqa: E402
    LL_FEATURE_DIMS,
    FeatureMajorHistory,
    quat_from_euler_xyz_wxyz as isaac_quat_from_euler_xyz,
)

from retrieval_experiment.metrics import StepInputs, TaskStateMachine, classify_termination  # noqa: E402

DEPLOY_SCRIPT = os.path.join(SIM_PY_ROOT, "deploy_mujoco", "b2wz1_hl_retrieval_uan.py")


def _load_deploy_module():
    spec = importlib.util.spec_from_file_location("_e5_deploy_retrieval", DEPLOY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DEPLOY = _load_deploy_module()
build_keypoints_from_kp0_yaw_pitch_plb = _DEPLOY.build_keypoints_from_kp0_yaw_pitch_plb

# HL actor frame groups (Isaac _flatten_actor_history_feature_major).
HL_FEATURE_DIMS = [3, 3, 12, 6, 1, 6, 3, 6, 3, 3, 9]
assert sum(HL_FEATURE_DIMS) == 55 and sum(LL_FEATURE_DIMS) == 80


def resolve_repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(REPO_ROOT, path))


def build_cube_model(xml_path: str, side: float, density: float, cube_cfg: dict,
                     cube_geom: str = "retrieval_cube_geom", gripper_bodies=("gripperStator", "gripperMover")):
    """Compile the retrieval scene with a homogeneous cube of side ``side`` (m) and density (kg/m^3).

    Returns (model, pair_geom_names). Geometry and mass (= density * side^3) are set
    at the spec level so that the compiler derives every dependent quantity
    (solid-box inertia m s^2 / 6, bounding radius/AABB, subtree mass) for the
    actual sample. Side and density are used unrounded.
    """
    spec = mujoco.MjSpec.from_file(xml_path)
    g = spec.geom(cube_geom)
    if g is None:
        raise ValueError(f"cube geom {cube_geom!r} not found in {xml_path}")
    half = 0.5 * float(side)
    g.size = [half, half, half]
    g.mass = float(density) * float(side) ** 3
    g.priority = int(cube_cfg["priority"])
    g.condim = int(cube_cfg["condim"])
    g.solref = list(map(float, cube_cfg["solref"]))
    g.solimp = list(map(float, cube_cfg["solimp"]))
    spin, roll = map(float, cube_cfg["spin_roll_friction"])
    g.friction = [1.0, spin, roll]
    pair_names = []
    for bname in gripper_bodies:
        b = spec.body(bname)
        for i, gg in enumerate(b.geoms):
            if not (gg.contype or gg.conaffinity):
                continue
            if not gg.name:
                gg.name = f"{bname}_col_{i}"
            p = spec.add_pair()
            p.geomname1 = cube_geom
            p.geomname2 = gg.name
            p.condim = int(cube_cfg["condim"])
            p.friction = [1.0, 1.0, spin, roll, roll]
            p.solref = list(map(float, cube_cfg["solref"]))
            p.solimp = list(map(float, cube_cfg["solimp"]))
            pair_names.append(gg.name)
    return spec.compile(), pair_names


class RetrievalSim:
    """MuJoCo world(s) + frozen WBC + frozen UAN + HL policy. Reused across episodes."""

    MUJOCO_JOINT_NAMES = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper",
    ]
    CTRL_JOINT_NAMES = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper",
    ]
    LEG_JOINT_NAMES = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    WHEEL_JOINT_NAMES = ["FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"]

    def __init__(self, deploy_cfg: dict, exp_cfg: dict, ort_threads: int = 1, observation_noise: str | None = None):
        cfg = deploy_cfg
        self.cfg = cfg
        self.exp = exp_cfg
        f32 = lambda k: np.asarray(cfg[k], dtype=np.float32)  # noqa: E731

        # ---------------- timing ----------------
        self.simulation_dt = float(cfg["simulation_dt"])
        self.ll_decimation = int(cfg["ll_control_decimation"])
        self.ll_steps_per_hl = int(cfg["ll_steps_per_hl_step"])
        self.ll_dt = self.simulation_dt * self.ll_decimation
        self.hl_dt = self.ll_dt * self.ll_steps_per_hl
        if abs(self.simulation_dt - 0.004) > 1e-12 or self.ll_decimation != 5 or self.ll_steps_per_hl != 5:
            raise ValueError("E5 requires 250 Hz physics/UAN, 50 Hz WBC, 10 Hz HL.")
        self.episode_length_s = float(cfg["episode_length_s"])
        self.max_episode_hl_steps = int(math.ceil(round(self.episode_length_s / self.hl_dt, 9)))
        assert int(cfg["ll_obs_dim"]) == 400 and int(cfg["ll_action_dim"]) == 22
        assert int(cfg["hl_obs_dim"]) == 165 and int(cfg["hl_action_dim"]) == 9

        # ---------------- robot / actuators ----------------
        self.root_pos_reset = f32("root_pos")
        self.root_quat_reset = f32("root_quat_wxyz")
        self.default_joint_pos = f32("default_joint_pos")
        self.kps = f32("kps")
        self.kds = f32("kds")
        self.leg_action_scale = float(cfg["leg_action_scale"])
        self.arm_action_scale = f32("arm_action_scale")
        self.wheel_action_scale = float(cfg["wheel_action_scale"])
        self.leg_torque_limits = f32("leg_torque_limits")
        self.wheel_velocity_limits = f32("wheel_velocity_limits")
        self.uan_history_length = int(cfg.get("uan_history_length", 25))
        self.uan_pos_scale = float(cfg.get("uan_pos_scale", 1.0))
        self.uan_vel_scale = float(cfg.get("uan_vel_scale", 0.05))
        self.uan_action_scale = float(cfg.get("uan_action_scale", 0.20))
        self.uan_kp = f32("uan_kp")
        self.uan_kd = f32("uan_kd")
        self.uan_nominal_torque_limits = f32("uan_nominal_torque_limits")
        self.uan_final_torque_limits = f32("uan_final_torque_limits")

        # Gripper: the training actuator is an Isaac Lab DCMotor.
        if str(cfg.get("gripper_mode", "dc_motor")).strip().lower() != "dc_motor":
            raise ValueError("E5 requires gripper_mode=dc_motor (training z1_gripper DCMotor).")
        self.gripper_torque_limit = float(cfg["gripper_torque_limit"])
        self.gripper_saturation_effort = float(cfg.get("gripper_saturation_effort", self.gripper_torque_limit))
        self.gripper_velocity_limit = float(cfg.get("gripper_velocity_limit", 2.0))
        self.gripper_open_pos = float(cfg["gripper_open_pos"])
        self.gripper_close_pos = float(cfg["gripper_close_pos"])
        self.gripper_binary_threshold = float(cfg["gripper_binary_threshold"])
        if bool(cfg.get("stage2_force_gripper_close_enabled", True)) or bool(
                exp_cfg["task"].get("stage2_force_gripper_close_enabled", False)):
            raise ValueError("Stage-2 force-close must be disabled for evaluation.")

        # ---------------- HL interface ----------------
        self.base_cmd_scale = f32("base_cmd_scale")
        self.kp0_x_range = f32("kp0_x_range")
        self.kp0_y_range = f32("kp0_y_range")
        self.kp0_z_range = f32("kp0_z_range")
        self.ee_yaw_range = f32("ee_yaw_range")
        self.ee_pitch_range = f32("ee_pitch_range")
        self.ee_pitch_to_euler_sign = float(cfg.get("ee_pitch_to_euler_sign", -1.0))
        self.neutral_kp0 = f32("neutral_kp0")
        self.neutral_ee_yaw = float(cfg["neutral_ee_yaw"])
        self.neutral_ee_pitch = float(cfg["neutral_ee_pitch"])
        self.fixed_ee_roll = float(cfg["fixed_ee_roll"])
        self.ee_kp_dx = float(cfg["ee_kp_dx"])
        self.ee_kp_dz = float(cfg["ee_kp_dz"])
        self.ground_z = float(cfg["ground_z"])
        self.gripper_center_offset_local = f32("gripper_center_offset_local")
        self.neutral_arm_action = np.array(
            [self._range_to_action(self.neutral_kp0[0], self.kp0_x_range),
             self._range_to_action(self.neutral_kp0[1], self.kp0_y_range),
             self._range_to_action(self.neutral_kp0[2], self.kp0_z_range),
             self._range_to_action(self.neutral_ee_yaw, self.ee_yaw_range),
             self._range_to_action(self.neutral_ee_pitch, self.ee_pitch_range)], dtype=np.float32)

        # task reset (deploy YAML is the control-path source; exp config must agree)
        self.object_sampling_center_x = float(cfg["object_sampling_center_x"])
        self.object_sampling_center_y = float(cfg["object_sampling_center_y"])
        self.object_spawn_z_epsilon = float(cfg["object_spawn_z_epsilon"])
        self.target_radius = float(cfg["enlarged_retrieval_target_radius"])
        dist = exp_cfg["distributions"]
        if not bool(cfg.get("enlarged_retrieval_target_sampling_range", False)):
            raise ValueError("Deploy YAML must use the training (enlarged) retrieval-target distribution.")
        if abs(self.target_radius - float(dist["retrieval_target"]["radius"])) > 1e-9:
            raise ValueError("Retrieval target radius differs between deploy YAML and experiment config.")
        if abs(self.object_spawn_z_epsilon - float(exp_cfg["cube"]["spawn_z_epsilon"])) > 1e-12:
            raise ValueError("Cube spawn epsilon differs between deploy YAML and experiment config.")
        task = exp_cfg["task"]
        for k in ("grasp_proxy_error_threshold", "gripper_not_fully_closed_angle_threshold",
                  "gripper_angle_hold_threshold", "grasp_proxy_enter_steps", "grasp_proxy_exit_steps"):
            if abs(float(cfg[k]) - float(task[k])) > 1e-12:
                raise ValueError(f"{k} differs between deploy YAML and experiment config.")

        # observation noise
        self.observation_noise = str(observation_noise or exp_cfg.get("observation_noise", "training"))
        if self.observation_noise not in ("training", "none"):
            raise ValueError("observation_noise must be 'training' or 'none'.")
        self.noise_model = exp_cfg["observation_noise_model"]

        # ---------------- networks ----------------
        so = ort.SessionOptions()
        if ort_threads > 0:          # 0 -> onnxruntime defaults (as in the deploy script)
            so.intra_op_num_threads = ort_threads
            so.inter_op_num_threads = ort_threads
        pol = exp_cfg["policies"]
        self.low_policy_path = resolve_repo_path(pol["low_level"]["path"])
        self.high_policy_path = resolve_repo_path(pol["high_level"]["path"])
        self.uan_model_path = resolve_repo_path(pol["uan"]["path"])
        for k, p in (("low_policy_path", self.low_policy_path), ("high_policy_path", self.high_policy_path),
                     ("uan_model_path", self.uan_model_path)):
            if os.path.abspath(resolve_repo_path(cfg[k])) != os.path.abspath(p):
                raise ValueError(f"{k}: deploy YAML ({cfg[k]}) and experiment config ({p}) disagree.")
        self.low_sess = ort.InferenceSession(self.low_policy_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.high_sess = ort.InferenceSession(self.high_policy_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.low_in, self.low_out = self.low_sess.get_inputs()[0].name, self.low_sess.get_outputs()[0].name
        self.high_in, self.high_out = self.high_sess.get_inputs()[0].name, self.high_sess.get_outputs()[0].name
        assert self.low_sess.get_inputs()[0].shape[-1] == 400 and self.low_sess.get_outputs()[0].shape[-1] == 22
        assert self.high_sess.get_inputs()[0].shape[-1] == 165 and self.high_sess.get_outputs()[0].shape[-1] == 9
        self.uan_model = torch.jit.load(self.uan_model_path, map_location="cpu").eval()
        for p in self.uan_model.parameters():
            p.requires_grad_(False)

        # ---------------- joint index maps ----------------
        idx = self.MUJOCO_JOINT_NAMES.index
        self.nj = len(self.MUJOCO_JOINT_NAMES)
        self.leg_idx = [idx(n) for n in self.LEG_JOINT_NAMES]
        self.arm_idx = [idx(n) for n in self.ARM_JOINT_NAMES]
        self.wheel_idx = [idx(n) for n in self.WHEEL_JOINT_NAMES]
        self.gripper_idx = idx("jointGripper")
        self.policy_pos_idx = [idx(n) for n in self.LEG_JOINT_NAMES + self.ARM_JOINT_NAMES]
        self.policy_vel_idx = [idx(n) for n in self.LEG_JOINT_NAMES + self.ARM_JOINT_NAMES + self.WHEEL_JOINT_NAMES]
        src_names = self.LEG_JOINT_NAMES + self.ARM_JOINT_NAMES + self.WHEEL_JOINT_NAMES + ["jointGripper"]
        src = {n: i for i, n in enumerate(src_names)}
        self.ctrl_src = [src[n] for n in self.CTRL_JOINT_NAMES]
        self.default_policy_pos = self.default_joint_pos[self.policy_pos_idx]
        self.default_leg_pos = self.default_joint_pos[self.leg_idx]
        self.default_arm_pos = self.default_joint_pos[self.arm_idx]
        self.default_gripper_pos = float(self.default_joint_pos[self.gripper_idx])

        # ---------------- MuJoCo models (one per cube size) ----------------
        self.xml_path = resolve_repo_path(cfg["xml_path"])
        self.cube_cfg = exp_cfg["cube"]
        self.friction_source = str(self.cube_cfg.get("friction_source", "static"))
        if self.friction_source not in ("static", "dynamic"):
            raise ValueError("cube.friction_source must be 'static' or 'dynamic'.")
        self._worlds: dict[tuple, dict] = {}          # small LRU of compiled (side, density) models
        self._max_worlds = 2
        self.ll_history = FeatureMajorHistory(LL_FEATURE_DIMS, int(cfg["ll_history_length"]))
        self.hl_history = FeatureMajorHistory(HL_FEATURE_DIMS, int(cfg["hl_history_length"]))
        self.uan_history = np.zeros((6, self.uan_history_length, 2), dtype=np.float32)
        self.task = TaskStateMachine(exp_cfg["task"], self.gripper_open_pos, self.max_episode_hl_steps)
        self.select_cube(0.05, float(self.cube_cfg["training_density"]))

    # ------------------------------------------------------------------
    # Static helpers (identical expressions to the deploy script)
    # ------------------------------------------------------------------
    @staticmethod
    def _action_to_range(x: float, limits) -> float:
        low, high = float(limits[0]), float(limits[1])
        return low + 0.5 * (float(x) + 1.0) * (high - low)

    @staticmethod
    def _range_to_action(value: float, limits) -> float:
        low, high = float(limits[0]), float(limits[1])
        return 2.0 * (float(value) - low) / (high - low) - 1.0

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def _build_world(self, side: float, density: float) -> dict:
        m, pair_names = build_cube_model(self.xml_path, side, density, self.cube_cfg,
                                         cube_geom=str(self.cfg.get("cube_geom", "retrieval_cube_geom")))
        m.opt.timestep = self.simulation_dt
        m.opt.noslip_iterations = int(self.exp.get("mujoco_noslip_iterations", m.opt.noslip_iterations))
        name2id = lambda t, n: mujoco.mj_name2id(m, t, n)  # noqa: E731
        B, J, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM
        w = {"m": m, "side": float(side), "density": float(density)}
        w["ee_bid"] = name2id(B, str(self.cfg.get("ee_body", "gripperStator")))
        w["mover_bid"] = name2id(B, "gripperMover")
        w["base_bid"] = name2id(B, "base_link")
        w["cube_bid"] = name2id(B, str(self.cfg.get("cube_body", "retrieval_cube")))
        w["cube_jid"] = name2id(J, str(self.cfg.get("cube_freejoint", "retrieval_cube_freejoint")))
        w["cube_gid"] = name2id(G, str(self.cfg.get("cube_geom", "retrieval_cube_geom")))
        for k in ("ee_bid", "mover_bid", "base_bid", "cube_bid", "cube_jid", "cube_gid"):
            if w[k] < 0:
                raise ValueError(f"MuJoCo object for {k} not found in {self.xml_path}")
        w["cube_qadr"] = int(m.jnt_qposadr[w["cube_jid"]])
        w["cube_vadr"] = int(m.jnt_dofadr[w["cube_jid"]])
        # robot joints must precede the cube in qpos/qvel (deploy-script layout)
        assert w["cube_qadr"] == 7 + self.nj and w["cube_vadr"] == 6 + self.nj

        # Robot collision shapes = collision geoms in the base_link tree.
        root = int(m.body_rootid[w["base_bid"]])
        names, per_body = [], {}
        robot_gids = []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if int(m.body_rootid[b]) != root or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            k = per_body.get(b, 0)
            per_body[b] = k + 1
            nm = mujoco.mj_id2name(m, G, g) or f"{mujoco.mj_id2name(m, B, b)}#{k}"
            names.append(nm)
            robot_gids.append(g)
        w["robot_shape_names"] = names
        w["robot_gids"] = np.array(robot_gids, dtype=int)
        w["stator_gids"] = frozenset(g for g in robot_gids if m.geom_bodyid[g] == w["ee_bid"])
        w["mover_gids"] = frozenset(g for g in robot_gids if m.geom_bodyid[g] == w["mover_bid"])
        # explicit cube<->gripper pairs: pair index -> robot shape index
        shape_of = {g: i for i, g in enumerate(robot_gids)}
        w["pair_shape_idx"] = []
        pair_others = set()
        for p in range(m.npair):
            g1, g2 = int(m.pair_geom1[p]), int(m.pair_geom2[p])
            other = g2 if g1 == w["cube_gid"] else g1
            assert w["cube_gid"] in (g1, g2) and other in shape_of
            w["pair_shape_idx"].append(shape_of[other])
            pair_others.add(other)
        if pair_others != set(w["stator_gids"] | w["mover_gids"]) or m.npair != len(pair_others):
            raise RuntimeError("Explicit cube pairs must cover exactly the gripperStator/gripperMover collision geoms.")

        # Same ctrlrange/forcerange widening as the deploy script.
        for j, jn in enumerate(self.ARM_JOINT_NAMES):
            ci = self.CTRL_JOINT_NAMES.index(jn)
            lim = float(self.uan_final_torque_limits[j])
            if m.actuator_ctrllimited[ci]:
                lo, hi = m.actuator_ctrlrange[ci]
                if lo > -lim + 1e-6 or hi < lim - 1e-6:
                    m.actuator_ctrlrange[ci, :] = (-lim, lim)
            if m.actuator_forcelimited[ci]:
                lo, hi = m.actuator_forcerange[ci]
                if lo > -lim + 1e-6 or hi < lim - 1e-6:
                    m.actuator_forcerange[ci, :] = (-lim, lim)
            jid = name2id(J, jn)
            if m.jnt_actfrclimited[jid]:
                lo, hi = m.jnt_actfrcrange[jid]
                if lo > -lim + 1e-6 or hi < lim - 1e-6:
                    m.jnt_actfrcrange[jid, :] = (-lim, lim)
        for i, n in enumerate(self.CTRL_JOINT_NAMES):
            jid = int(m.actuator_trnid[i, 0])
            if mujoco.mj_id2name(m, J, jid) != n:
                raise RuntimeError(f"actuator {i} drives {mujoco.mj_id2name(m, J, jid)}, expected {n}")

        w["nom_geom_friction"] = m.geom_friction.copy()
        w["nom_geom_priority"] = m.geom_priority.copy()
        w["nom_pair_friction"] = m.pair_friction.copy()
        w["nom_base_mass"] = float(m.body_mass[w["base_bid"]])
        w["nom_base_inertia"] = m.body_inertia[w["base_bid"]].copy()
        w["nom_base_ipos"] = m.body_ipos[w["base_bid"]].copy()
        w["d"] = mujoco.MjData(m)
        return w

    def select_cube(self, side: float, density: float):
        """Activate the compiled model of this exact (side, density) sample (no rounding)."""
        k = (float(side), float(density))
        if k in self._worlds:
            w = self._worlds.pop(k)
        else:
            w = self._build_world(*k)
        self._worlds[k] = w                                  # most recently used last
        while len(self._worlds) > self._max_worlds:
            self._worlds.pop(next(iter(self._worlds)))
        self.w = w
        self.m, self.d = w["m"], w["d"]
        self.side = w["side"]
        self.density = w["density"]
        self.half_extent = 0.5 * w["side"]
        for key in ("ee_bid", "mover_bid", "base_bid", "cube_bid", "cube_gid", "cube_qadr", "cube_vadr"):
            setattr(self, key, w[key])
        return w

    @property
    def robot_shape_names(self) -> list[str]:
        return list(self.w["robot_shape_names"])

    def apply_physics(self, phys: dict):
        """Per-scenario materials and base inertial randomization (runtime model edits only)."""
        m, w = self.m, self.w
        m.geom_friction[:] = w["nom_geom_friction"]
        m.geom_priority[:] = w["nom_geom_priority"]
        m.pair_friction[:] = w["nom_pair_friction"]
        if list(phys["robot_shape_names"]) != w["robot_shape_names"]:
            raise ValueError("Scenario robot shape list does not match the model.")
        src = "static" if self.friction_source == "static" else "dynamic"
        mu_r = np.asarray(phys[f"robot_mu_{src}"], dtype=np.float64)
        mu_c = float(phys[f"cube_mu_{src}"])
        mu_g = float(phys.get("ground_friction", 1.0))
        # PhysX multiply: robot-ground mu = mu_shape * mu_ground. Priority 1 makes
        # the robot geom's coefficient govern its contacts with the floor (priority 0).
        gids = w["robot_gids"]
        m.geom_priority[gids] = 1
        m.geom_friction[gids, 0] = mu_r * mu_g
        # Cube (priority 2): cube-ground mu = mu_cube * mu_ground.
        m.geom_priority[self.cube_gid] = int(self.cube_cfg["priority"])
        m.geom_friction[self.cube_gid, 0] = mu_c * mu_g
        # Cube-gripper explicit pairs: mu = mu_cube * mu_shape.
        for p, si in enumerate(w["pair_shape_idx"]):
            mu = mu_c * float(mu_r[si])
            m.pair_friction[p, 0] = mu
            m.pair_friction[p, 1] = mu
        # Base inertial randomization (Isaac randomize_rigid_body_mass add + recompute_inertia,
        # randomize_rigid_body_com additive offset in the body frame).
        add = float(phys.get("base_mass_add", 0.0))
        new_mass = w["nom_base_mass"] + add
        m.body_mass[self.base_bid] = new_mass
        m.body_inertia[self.base_bid] = w["nom_base_inertia"] * (new_mass / w["nom_base_mass"])
        m.body_ipos[self.base_bid] = w["nom_base_ipos"] + np.asarray(phys.get("base_com_offset", [0, 0, 0]), dtype=np.float64)
        mujoco.mj_setConst(m, self.d)

    # ------------------------------------------------------------------
    # Kinematics (deploy-script expressions)
    # ------------------------------------------------------------------
    def get_root_pose(self):
        d = self.d
        return d.qpos[0:3].copy().astype(np.float32), quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))

    def get_cube_pos_w(self):
        return self.d.xpos[self.cube_bid].copy().astype(np.float32)

    def get_gripper_geometry(self):
        d = self.d
        root_pos_w, root_quat_w = self.get_root_pose()
        stator_pos_w = d.xpos[self.ee_bid].copy().astype(np.float32)
        stator_rot_w = d.xmat[self.ee_bid].reshape(3, 3).copy().astype(np.float32)
        gripper_center_pos_w = (stator_pos_w + stator_rot_w @ self.gripper_center_offset_local).astype(np.float32)
        object_pos_w = self.get_cube_pos_w()
        object_center_pos_base = quat_apply_inverse_wxyz(root_quat_w, object_pos_w - root_pos_w).astype(np.float32)
        gripper_center_pos_base = quat_apply_inverse_wxyz(root_quat_w, gripper_center_pos_w - root_pos_w).astype(np.float32)
        gx = quat_apply_inverse_wxyz(root_quat_w, stator_rot_w[:, 0].astype(np.float32)).astype(np.float32)
        gy = quat_apply_inverse_wxyz(root_quat_w, stator_rot_w[:, 1].astype(np.float32)).astype(np.float32)
        gripper_orientation_base = np.concatenate([gx, gy]).astype(np.float32)
        grasp_error = float(np.linalg.norm(object_pos_w - gripper_center_pos_w))
        return object_center_pos_base, gripper_orientation_base, gripper_center_pos_base, gripper_center_pos_w, grasp_error

    def root_yaw(self) -> float:
        _, q = self.get_root_pose()
        return float(euler_xyz_from_quat_wxyz(q)[2])

    # ------------------------------------------------------------------
    # Contacts (net world-frame force between bodies, Isaac force_matrix_w / net_forces_w)
    # ------------------------------------------------------------------
    def _net_force(self, select) -> float:
        d, m = self.d, self.m
        f6 = np.zeros(6, dtype=np.float64)
        net = np.zeros(3, dtype=np.float64)
        for ci in range(d.ncon):
            c = d.contact[ci]
            s = select(int(c.geom1), int(c.geom2))
            if s:
                mujoco.mj_contactForce(m, d, ci, f6)
                net += s * (c.frame.reshape(3, 3).T @ f6[:3])
        return float(np.linalg.norm(net))

    def cube_body_force(self, gset) -> float:
        cg = self.cube_gid
        return self._net_force(lambda a, b: (1 if b == cg else -1) if ((a == cg and b in gset) or (b == cg and a in gset)) else 0)

    def cube_max_penetration(self) -> float:
        d = self.d
        pen = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]
            if self.cube_gid in (c.geom1, c.geom2):
                pen = max(pen, -float(c.dist))
        return pen

    def cube_in_stator(self) -> np.ndarray:
        d = self.d
        R = d.xmat[self.ee_bid].reshape(3, 3)
        return R.T @ (d.xpos[self.cube_bid] - d.xpos[self.ee_bid])

    def base_contact_force(self) -> float:
        m, bb = self.m, self.base_bid
        return self._net_force(lambda a, b: (1 if m.geom_bodyid[b] == bb else -1)
                               if (m.geom_bodyid[a] == bb or m.geom_bodyid[b] == bb) else 0)

    # ------------------------------------------------------------------
    # Commands / observations
    # ------------------------------------------------------------------
    def build_neutral_ee_command(self):
        return build_keypoints_from_kp0_yaw_pitch_plb(
            kp0=self.neutral_kp0, yaw=self.neutral_ee_yaw, pitch=self.neutral_ee_pitch, roll=self.fixed_ee_roll,
            kp_dx=self.ee_kp_dx, kp_dz=self.ee_kp_dz, pitch_to_euler_sign=self.ee_pitch_to_euler_sign)

    def decode_hl_action(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.base_command = (action[0:3] * self.base_cmd_scale).astype(np.float32)
        kp0_cmd = np.array([self._action_to_range(action[3], self.kp0_x_range),
                            self._action_to_range(action[4], self.kp0_y_range),
                            self._action_to_range(action[5], self.kp0_z_range)], dtype=np.float32)
        yaw_cmd = self._action_to_range(action[6], self.ee_yaw_range)
        pitch_cmd = self._action_to_range(action[7], self.ee_pitch_range)
        self.ee_cmd_plb = build_keypoints_from_kp0_yaw_pitch_plb(
            kp0=kp0_cmd, yaw=float(yaw_cmd), pitch=float(pitch_cmd), roll=self.fixed_ee_roll,
            kp_dx=self.ee_kp_dx, kp_dz=self.ee_kp_dz, pitch_to_euler_sign=self.ee_pitch_to_euler_sign)
        self.raw_gripper_action = float(action[8])
        close = self.raw_gripper_action > self.gripper_binary_threshold   # no force-close override
        self.executed_gripper_cmd_norm = 1.0 if close else -1.0
        self.gripper_target = self.gripper_close_pos if close else self.gripper_open_pos

    def build_previous_hl_action(self):
        denom = np.maximum(np.abs(self.base_cmd_scale), 1.0e-6)
        eff = np.clip(self.base_command / denom, -1.0, 1.0)
        return np.concatenate([eff, self.current_hl_action[3:8],
                               np.array([self.executed_gripper_cmd_norm], dtype=np.float32)], dtype=np.float32)

    def build_hl_obs_frame(self, rng: np.random.Generator | None = None):
        d, m = self.d, self.m
        root_pos_w, root_quat_w = self.get_root_pose()
        root_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(root_quat_w, gravity_w).astype(np.float32)
        qpos = d.qpos[7:7 + self.nj].copy().astype(np.float32)
        qvel = d.qvel[6:6 + self.nj].copy().astype(np.float32)
        leg_pos_rel = (qpos[self.leg_idx] - self.default_joint_pos[self.leg_idx]).astype(np.float32)
        arm_pos_rel = (qpos[self.arm_idx] - self.default_joint_pos[self.arm_idx]).astype(np.float32)
        gripper_pos_rel = np.array([qpos[self.gripper_idx] - self.default_gripper_pos], dtype=np.float32)
        arm_joint_vel = qvel[self.arm_idx].astype(np.float32)
        obj_b, gori_b, gc_b, _, _ = self.get_gripper_geometry()
        target_b = quat_apply_inverse_wxyz(root_quat_w, self.retrieval_target_pos_w - root_pos_w).astype(np.float32)
        prev = self.build_previous_hl_action()
        if rng is not None:
            nm = self.noise_model

            def un(x, s):
                return (x + rng.uniform(-float(s), float(s), size=x.shape)).astype(np.float32)
            # Isaac order inside _build_hl_obs_frames: object, gripper center, target,
            # orientation rotation, then the proprioceptive terms.
            obj_b = un(obj_b, nm["object_center_pos_base"])
            gc_b = un(gc_b, nm["gripper_center_pos_base"])
            target_b = un(target_b, nm["retrieval_target_pos_base"])
            e = rng.normal(size=3) * float(nm["gripper_orientation_std_rad"])
            qn = isaac_quat_from_euler_xyz(float(e[0]), float(e[1]), float(e[2]))
            gori_b = np.concatenate([quat_apply_wxyz(qn, gori_b[0:3].astype(np.float64)),
                                     quat_apply_wxyz(qn, gori_b[3:6].astype(np.float64))]).astype(np.float32)
            root_ang_vel_b = un(root_ang_vel_b, nm["root_ang_vel_b"])
            projected_gravity_b = un(projected_gravity_b, nm["projected_gravity_b"])
            leg_pos_rel = un(leg_pos_rel, nm["leg_pos_rel"])
            arm_pos_rel = un(arm_pos_rel, nm["arm_pos_rel"])
            gripper_pos_rel = un(gripper_pos_rel, nm["gripper_pos_rel"])
            arm_joint_vel = un(arm_joint_vel, nm["arm_joint_vel"])
        obs = np.concatenate([root_ang_vel_b, projected_gravity_b, leg_pos_rel, arm_pos_rel, gripper_pos_rel,
                              arm_joint_vel, obj_b, gori_b, gc_b, target_b, prev], dtype=np.float32)
        assert obs.shape == (55,)
        return obs

    def ll_obs_frame(self):
        d, m = self.d, self.m
        qpos = d.qpos[7:7 + self.nj].copy().astype(np.float32)
        qvel = d.qvel[6:6 + self.nj].copy().astype(np.float32)
        base_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)
        root_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(root_quat_w, gravity_w).astype(np.float32)
        jp = qpos[self.policy_pos_idx] - self.default_policy_pos
        jv = qvel[self.policy_vel_idx]
        obs = np.concatenate([base_ang_vel_b, projected_gravity_b, self.base_command, self.ee_cmd_plb,
                              jp[:12], jp[12:18], jv[:12], jv[12:18], jv[18:22], self.last_ll_action], dtype=np.float32)
        assert obs.shape == (80,)
        return obs

    # ------------------------------------------------------------------
    # Policies and actuation
    # ------------------------------------------------------------------
    def infer_hl(self) -> np.ndarray:
        x = self.hl_history.flat()
        assert x.shape == (165,)
        a = self.high_sess.run([self.high_out], {self.high_in: x[None, :]})[0][0].astype(np.float32)
        self.current_hl_action = np.clip(a, -1.0, 1.0)
        self.decode_hl_action(self.current_hl_action)
        return self.current_hl_action

    def run_low_level_policy(self):
        self.ll_history.append(self.ll_obs_frame())
        a = self.low_sess.run([self.low_out], {self.low_in: self.ll_history.flat()[None, :]})[0][0].astype(np.float32)
        self.last_ll_action = a
        self.leg_target = self.default_leg_pos + self.leg_action_scale * a[0:12]
        self.arm_target = self.default_arm_pos + self.arm_action_scale * a[12:18]
        self.wheel_cmd = self.wheel_action_scale * a[18:22]

    def _uan_torque(self, q_des, q, qd):
        q_des = np.asarray(q_des, dtype=np.float32).reshape(6,)
        q = np.asarray(q, dtype=np.float32).reshape(6,)
        qd = np.asarray(qd, dtype=np.float32).reshape(6,)
        h = self.uan_history
        h[:, :-1, :] = h[:, 1:, :]
        h[:, -1, 0] = (q_des - q) * self.uan_pos_scale
        h[:, -1, 1] = qd * self.uan_vel_scale
        obs = torch.from_numpy(np.ascontiguousarray(h.reshape(1, -1), dtype=np.float32))
        with torch.inference_mode():
            out = self.uan_model(obs)
            if isinstance(out, (tuple, list)):
                out = out[0]
        a = out.detach().cpu().numpy().reshape(6,).astype(np.float32)
        if not np.all(np.isfinite(a)):
            raise FloatingPointError(f"Non-finite UAN action: {a}")
        tau_pd = self.uan_kp * (q_des - q) - self.uan_kd * qd
        tau_nom = np.clip(tau_pd, -self.uan_nominal_torque_limits, self.uan_nominal_torque_limits)
        return np.clip(tau_nom + a * self.uan_action_scale * self.uan_kp,
                       -self.uan_final_torque_limits, self.uan_final_torque_limits).astype(np.float32)

    def _gripper_dc_motor_torque(self, q: float, qd: float) -> float:
        """Isaac Lab DCMotor: ideal PD clipped by the four-quadrant torque-speed envelope."""
        gi = self.gripper_idx
        tau = self.kps[gi] * (self.gripper_target - q) - self.kds[gi] * qd
        vlim = self.gripper_velocity_limit
        v_eff = vlim * (1.0 + self.gripper_torque_limit / self.gripper_saturation_effort)
        v = float(np.clip(qd, -v_eff, v_eff))
        top = self.gripper_saturation_effort * (1.0 - v / vlim)
        bottom = self.gripper_saturation_effort * (-1.0 - v / vlim)
        return float(np.clip(tau, max(bottom, -self.gripper_torque_limit), min(top, self.gripper_torque_limit)))

    def apply_low_level_actuation(self):
        d = self.d
        qpos = d.qpos[7:7 + self.nj].copy().astype(np.float32)
        qvel = d.qvel[6:6 + self.nj].copy().astype(np.float32)
        li = self.leg_idx
        leg_tau = self.kps[li] * (self.leg_target - qpos[li]) - self.kds[li] * qvel[li]
        leg_tau = np.clip(leg_tau, -self.leg_torque_limits, self.leg_torque_limits).astype(np.float32)
        arm_tau = self._uan_torque(self.arm_target, qpos[self.arm_idx], qvel[self.arm_idx])
        gi = self.gripper_idx
        gripper_tau = self._gripper_dc_motor_torque(float(qpos[gi]), float(qvel[gi]))
        wheel_ctrl = np.clip(self.wheel_cmd, -self.wheel_velocity_limits, self.wheel_velocity_limits).astype(np.float32)
        src = np.concatenate([leg_tau, arm_tau, wheel_ctrl, np.array([gripper_tau], dtype=np.float32)], dtype=np.float32)
        d.ctrl[:] = 0.0
        for ci, si in enumerate(self.ctrl_src):
            d.ctrl[ci] = src[si]

    def run_hl_interval(self, on_physics_step=None):
        """Execute the current HL command: 5 WBC decisions x 5 physics/UAN steps."""
        for _ in range(self.ll_steps_per_hl):
            self.run_low_level_policy()
            for _ in range(self.ll_decimation):
                self.apply_low_level_actuation()
                mujoco.mj_step(self.m, self.d)
                if on_physics_step is not None:
                    on_physics_step()

    # ------------------------------------------------------------------
    # Reset (Isaac _reset_idx for one env, driven by a scenario)
    # ------------------------------------------------------------------
    def reset(self, scenario: dict, side: float, density: float):
        self.select_cube(side, density)
        self.apply_physics(scenario["physics"])
        m, d = self.m, self.d
        init, task = scenario["initial_state"], scenario["task"]
        mujoco.mj_resetData(m, d)
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        dz = float(init.get("base_dz", 0.0))
        if dz != 0.0:
            root_pos = self.root_pos_reset.astype(np.float64).copy()
            root_pos[2] += dz
            d.qpos[0:3] = root_pos
        else:
            d.qpos[0:3] = self.root_pos_reset
        roll, pitch = float(init.get("base_roll", 0.0)), float(init.get("base_pitch", 0.0))
        if roll != 0.0 or pitch != 0.0:
            q = quat_mul_wxyz(isaac_quat_from_euler_xyz(roll, pitch, 0.0), self.root_quat_reset.astype(np.float64))
            d.qpos[3:7] = q / np.linalg.norm(q)
        else:
            d.qpos[3:7] = self.root_quat_reset
        jp = self.default_joint_pos.copy()
        leg_scale, arm_scale = float(init.get("leg_scale", 1.0)), float(init.get("arm_scale", 1.0))
        if leg_scale != 1.0:
            jp[self.leg_idx] = jp[self.leg_idx] * np.float32(leg_scale)
        if arm_scale != 1.0:
            jp[self.arm_idx] = jp[self.arm_idx] * np.float32(arm_scale)
        d.qpos[7:7 + self.nj] = jp
        d.qpos[7 + self.gripper_idx] = self.gripper_open_pos          # Isaac: gripper reset fully open
        root_vel = np.asarray(init.get("root_vel", [0.0] * 6), dtype=np.float64)
        if np.any(root_vel != 0.0):
            d.qvel[0:3] = root_vel[0:3]                                # world-frame linear
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            d.qvel[3:6] = R.reshape(3, 3).T @ root_vel[3:6]          # freejoint angular vel is body-frame
        ca, va = self.cube_qadr, self.cube_vadr
        d.qpos[ca:ca + 3] = np.array([0.8, 0.0, self.half_extent + self.object_spawn_z_epsilon], dtype=np.float32)
        d.qpos[ca + 3:ca + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        d.qvel[va:va + 6] = 0.0
        mujoco.mj_forward(m, d)

        # ---- task reset (deploy sample_task_reset with scenario draws) ----
        root_pos_w, root_quat_w = self.get_root_pose()
        _, _, root_yaw = euler_xyz_from_quat_wxyz(root_quat_w)
        plb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(quat_from_yaw_wxyz(root_yaw)))
        plb_pos_w = root_pos_w.copy()
        plb_pos_w[2] = self.ground_z
        r, theta = float(task["object_r"]), float(task["object_theta"])
        object_rel_plb = np.array([self.object_sampling_center_x + r * math.cos(theta),
                                   self.object_sampling_center_y + r * math.sin(theta),
                                   self.half_extent + self.object_spawn_z_epsilon], dtype=np.float32)
        object_pos_w = (plb_pos_w + quat_apply_wxyz(plb_quat_w, object_rel_plb)).astype(np.float32)
        d.qpos[ca:ca + 3] = object_pos_w
        d.qpos[ca + 3:ca + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        d.qvel[va:va + 6] = 0.0
        tr = self.target_radius * math.sqrt(float(task["target_u"]))
        tt = float(task["target_theta"])
        self.retrieval_target_pos_w = np.array([root_pos_w[0] + tr * math.cos(tt), root_pos_w[1] + tr * math.sin(tt),
                                                float(task["target_z"])], dtype=np.float32)
        mujoco.mj_forward(m, d)
        self.cube_spawn_w = self.get_cube_pos_w()

        # ---- controller state (deploy reset_episode) ----
        self.base_command = np.zeros(3, dtype=np.float32)
        self.ee_cmd_plb = self.build_neutral_ee_command()
        self.last_ll_action = np.zeros(22, dtype=np.float32)
        self.leg_target = self.default_leg_pos.copy()
        self.arm_target = self.default_arm_pos.copy()
        self.wheel_cmd = np.zeros(4, dtype=np.float32)
        self.gripper_target = self.gripper_open_pos
        self.current_hl_action = np.zeros(9, dtype=np.float32)
        self.current_hl_action[3:8] = self.neutral_arm_action
        self.raw_gripper_action = -1.0
        self.executed_gripper_cmd_norm = -1.0
        self.uan_history.fill(0.0)
        self.noise_rng = np.random.default_rng(int(scenario.get("noise_seed", 0))) \
            if self.observation_noise == "training" else None
        self.ll_history.fill(self.ll_obs_frame())
        self.hl_history.fill(self.build_hl_obs_frame(self.noise_rng))
        # Isaac returns the post-reset observation through _get_observations(),
        # which appends one more (independently noised) frame of the same state.
        self.hl_history.append(self.build_hl_obs_frame(self.noise_rng))
        self.task.reset(gripper_q0=float(np.float32(d.qpos[7 + self.gripper_idx])))
        self.hl_steps_completed = 0
        self._warn0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)

    # ------------------------------------------------------------------
    # End-of-HL-step task evaluation
    # ------------------------------------------------------------------
    def numerically_ok(self) -> bool:
        d = self.d
        if int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) != self._warn0:
            return False
        return bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))

    def task_step(self) -> tuple[dict, dict]:
        """Isaac _update_task_buffers + stage/success/dones for the just-completed HL step."""
        self.hl_steps_completed += 1
        thr = float(self.exp["task"]["gripper_cube_contact_force_threshold"])
        _, _, _, gc_w, grasp_error = self.get_gripper_geometry()
        cube_w = self.get_cube_pos_w()
        retrieval_error = float(np.linalg.norm(cube_w - self.retrieval_target_pos_w))
        gq = float(np.float32(self.d.qpos[7 + self.gripper_idx]))
        f_st = self.cube_body_force(self.w["stator_gids"])
        f_mv = self.cube_body_force(self.w["mover_gids"])
        f_base = self.base_contact_force()
        x = StepInputs(grasp_error=grasp_error, gripper_q=gq, close_executed=self.executed_gripper_cmd_norm > 0.0,
                       stator_contact=f_st > thr, mover_contact=f_mv > thr, retrieval_error=retrieval_error,
                       base_contact=f_base > float(self.exp["task"]["base_contact_force_threshold"]))
        ev = self.task.update(x)
        meas = {"grasp_error": grasp_error, "retrieval_error": retrieval_error, "gripper_q": gq,
                "f_stator": f_st, "f_mover": f_mv, "f_base": f_base, "cube_w": cube_w, "gripper_center_w": gc_w}
        return ev, meas

    def next_hl_action(self):
        self.hl_history.append(self.build_hl_obs_frame(self.noise_rng))
        return self.infer_hl()


# ----------------------------------------------------------------------
# One episode
# ----------------------------------------------------------------------
TRAJ_KEYS = ("t", "stage", "proxy", "stage2_now", "stator_contact", "mover_contact", "f_stator", "f_mover",
             "f_base", "grasp_error", "retrieval_error", "gripper_q", "gripper_cmd", "raw_gripper",
             "cube_w", "root_pos_w", "root_yaw", "hl_action", "success_now", "cube_in_stator", "cube_pen")
# 250-Hz physical-retention channels (one sample per physics step, read-only).
PHYS_KEYS = ("ph_f_stator", "ph_f_mover", "ph_q", "ph_qd", "ph_rel_pos", "ph_rel_quat", "ph_cube_z",
             "ph_cube_speed", "ph_cmd")


def run_episode(sim: RetrievalSim, scenario: dict, side: float, density: float, record: bool = True):
    """Run one episode with a cube of ``side`` (m) and ``density`` (kg/m^3).

    Returns (summary dict, 10-Hz trajectory dict of arrays).
    """
    sim.reset(scenario, side, density)
    phys = {k: [] for k in PHYS_KEYS}
    gq, gv = 7 + sim.gripper_idx, 6 + sim.gripper_idx
    st_g, mv_g = sim.w["stator_gids"], sim.w["mover_gids"]
    qbuf = np.zeros(4)

    def log_physics():
        """Read-only sample after every mj_step (MuJoCo data of that step: contacts/poses at its start)."""
        d = sim.d
        Rs = d.xmat[sim.ee_bid].reshape(3, 3)
        mujoco.mju_mat2Quat(qbuf, (Rs.T @ d.xmat[sim.cube_bid].reshape(3, 3)).flatten())
        phys["ph_f_stator"].append(sim.cube_body_force(st_g))
        phys["ph_f_mover"].append(sim.cube_body_force(mv_g))
        phys["ph_q"].append(d.qpos[gq])
        phys["ph_qd"].append(d.qvel[gv])
        phys["ph_rel_pos"].append(Rs.T @ (d.xpos[sim.cube_bid] - d.xpos[sim.ee_bid]))
        phys["ph_rel_quat"].append(qbuf.copy())
        phys["ph_cube_z"].append(d.xpos[sim.cube_bid][2])
        phys["ph_cube_speed"].append(float(np.linalg.norm(d.qvel[sim.cube_vadr:sim.cube_vadr + 3])))
        phys["ph_cmd"].append(sim.executed_gripper_cmd_norm)
    traj = {k: [] for k in TRAJ_KEYS}
    sim.infer_hl()                          # first action from the post-reset observation
    reason, success_step, grasp_step, s1_step = None, None, None, None
    ever_stage2, ever_s1 = False, False
    n_s1_to_s0 = 0
    max_stage = 0
    err = ""
    ev = {"stage": 0}
    try:
        while True:
            hl_action = sim.current_hl_action.copy()
            sim.run_hl_interval(log_physics if record else None)
            if not sim.numerically_ok():
                reason = "numerical_failure"
                break
            ev, meas = sim.task_step()
            k = sim.hl_steps_completed
            if ev["s0_to_s1"] and s1_step is None:
                s1_step = k
            ever_s1 |= ev["stage"] >= 1
            n_s1_to_s0 += int(ev["s1_to_s0"])
            if ev["s1_to_s2"]:
                ever_stage2 = True
                if grasp_step is None:
                    grasp_step = k
            max_stage = max(max_stage, ev["stage"])
            if record:
                rp, _ = sim.get_root_pose()
                for key, val in (("t", k * sim.hl_dt), ("stage", ev["stage"]), ("proxy", ev["proxy"]),
                                 ("stage2_now", ev["stage2_now"]), ("stator_contact", meas["f_stator"] > 0.3),
                                 ("mover_contact", meas["f_mover"] > 0.3), ("f_stator", meas["f_stator"]),
                                 ("f_mover", meas["f_mover"]), ("f_base", meas["f_base"]),
                                 ("grasp_error", meas["grasp_error"]), ("retrieval_error", meas["retrieval_error"]),
                                 ("gripper_q", meas["gripper_q"]), ("gripper_cmd", sim.executed_gripper_cmd_norm),
                                 ("raw_gripper", sim.raw_gripper_action), ("cube_w", meas["cube_w"]),
                                 ("root_pos_w", rp), ("root_yaw", sim.root_yaw()), ("hl_action", hl_action),
                                 ("success_now", ev["success_now"]), ("cube_in_stator", sim.cube_in_stator()),
                                 ("cube_pen", sim.cube_max_penetration())):
                    traj[key].append(val)
            if ev["newly_successful"]:
                if ev["terminated"]:
                    # a simultaneous physical failure invalidates the success instant
                    reason = classify_termination(ev, ever_stage2)
                else:
                    reason, success_step = "success", k
                break
            if ev["terminated"] or ev["timeout"]:
                reason = classify_termination(ev, ever_stage2)
                break
            sim.next_hl_action()
    except FloatingPointError as e:     # numerical blow-ups only; code errors propagate
        reason, err = "numerical_failure", f"{type(e).__name__}: {e}"

    k = sim.hl_steps_completed
    cube_w = sim.get_cube_pos_w()
    summary = {
        "success": reason == "success",
        "failure_reason": reason,
        "hl_steps": k,
        "episode_time_s": k * sim.hl_dt,
        "completion_time_s": success_step * sim.hl_dt if success_step else float("nan"),
        "time_to_grasp_s": grasp_step * sim.hl_dt if grasp_step else float("nan"),
        "time_to_stage1_s": s1_step * sim.hl_dt if s1_step else float("nan"),
        "retrieval_phase_time_s": (success_step - grasp_step) * sim.hl_dt if (success_step and grasp_step) else float("nan"),
        "grasp_confirmed": bool(ever_stage2),
        "reached_stage1": bool(ever_s1),
        "n_s1_to_s0": n_s1_to_s0,
        "final_stage": int(sim.task.stage),
        "max_stage": int(max_stage),
        "final_retrieval_error": float(np.linalg.norm(cube_w - sim.retrieval_target_pos_w)),
        "final_cube_z": float(cube_w[2]),
        "error": err,
        "cube_side_m": float(sim.side),
        "cube_density": float(sim.density),
        "cube_mass_kg": float(sim.m.body_mass[sim.cube_bid]),
    }
    # in-episode grasp physics diagnostics over curriculum Stage-2 HL steps
    if record and traj["stage"]:
        st = np.asarray(traj["stage"])
        s2 = st == 2
        if s2.any():
            cis = np.asarray(traj["cube_in_stator"])[s2]
            summary.update(
                s2_steps=int(s2.sum()),
                s2_dual_contact_frac=float(np.mean(np.asarray(traj["stage2_now"])[s2])),
                s2_mean_f_stator=float(np.mean(np.asarray(traj["f_stator"])[s2])),
                s2_mean_f_mover=float(np.mean(np.asarray(traj["f_mover"])[s2])),
                s2_max_penetration_mm=1000 * float(np.max(np.asarray(traj["cube_pen"])[s2])),
                s2_cube_slip_mm=1000 * float(np.max(np.linalg.norm(cis - cis[0], axis=1))),
                s2_mean_gripper_angle_deg=float(np.degrees(np.mean(np.asarray(traj["gripper_q"])[s2]))),
            )
    tr = {}
    if record:
        for key, vals in traj.items():
            tr[key] = np.asarray(vals)
        tr["retrieval_target_w"] = sim.retrieval_target_pos_w.copy()
        tr["cube_spawn_w"] = sim.cube_spawn_w.copy()
        for key, vals in phys.items():
            tr[key] = np.asarray(vals, dtype=np.int8 if key == "ph_cmd" else np.float32)
        tr["ph_dt"] = np.float64(sim.simulation_dt)
        tr["success_step"] = np.int64(success_step if success_step else -1)
        tr["cube_side"] = np.float64(side)
        tr["cube_density"] = np.float64(density)
        tr["cube_mass"] = np.float64(sim.m.body_mass[sim.cube_bid])
    return summary, tr
