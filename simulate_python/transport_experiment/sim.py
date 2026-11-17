"""Headless MuJoCo engine for the E4 dynamic nonprehensile transport benchmark.

This is a line-by-line port of the (aligned) interactive sim2sim
``deploy_mujoco/b2wz1_hl_transport_uan.py``. The physics, frozen WBC, UAN,
reference process, disk frame, puck spawn and contact logic keep the SAME
float32 arithmetic as that script, so the two are bit-exact on identical inputs
(``tests/test_deploy_parity.py``). The only functional differences are:

  * no viewer; one episode per ``reset()`` / ``run_episode()`` call;
  * the task reference process (base velocity / EE target sequence and hold
    durations) is read from a PRECOMPUTED scenario instead of a runtime RNG, so
    all four methods receive identical schedules regardless of early failure;
  * per-scenario physical randomization (puck mass/friction, base mass/COM) and
    reset randomization (base z/roll/pitch, root velocity, arm-pose scale),
    reproducing the Isaac Lab training distributions (``EventCfg`` and
    ``_reset_idx`` of the transport environment);
  * the high-level policy / decoder is supplied by ``controllers.py``.

Robot, WBC, UAN and task constants are read from the deployment YAML so there
is a single source of truth.
"""
from __future__ import annotations

import math
import os
import sys
from collections import deque

import mujoco
import numpy as np
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_PY_ROOT = os.path.abspath(os.path.join(HERE, ".."))          # simulate_python/
REPO_ROOT = os.path.abspath(os.path.join(HERE, "../.."))          # unitree_mujoco/
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utilities.math import (  # noqa: E402
    euler_xyz_from_quat_wxyz,
    quat_apply_inverse_wxyz,
    quat_from_rotmat_wxyz,
    quat_from_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_rotate_inverse_numpy,
    quat_unique_wxyz,
)
from utilities.mujoco_helper import get_sensor_slice  # noqa: E402


def resolve_repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(REPO_ROOT, path))


def build_keypoints_from_kp0_yaw_pitch_roll_plb(kp0, yaw, pitch, roll, kp_dx=0.30, kp_dz=0.30, eps=1.0e-6):
    """Verbatim NumPy port of the Isaac Lab ``command_adapter`` (see deploy script)."""
    kp0 = np.asarray(kp0, dtype=np.float32).reshape(3,)
    yaw = float(yaw)
    pitch = float(pitch)
    roll = float(roll)
    cp = math.cos(pitch)
    vx = np.array([cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)], dtype=np.float32)
    vx /= max(float(np.linalg.norm(vx)), eps)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    vz0 = up - float(np.dot(up, vx)) * vx
    vz0_norm = float(np.linalg.norm(vz0))
    if vz0_norm > eps:
        vz0 /= vz0_norm
    else:
        vz0 = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)
        vz0 /= max(float(np.linalg.norm(vz0)), eps)
    vy0 = np.cross(vz0, vx).astype(np.float32)
    vy0 /= max(float(np.linalg.norm(vy0)), eps)
    vz = math.cos(roll) * vz0 - math.sin(roll) * vy0
    vz = vz.astype(np.float32)
    vz /= max(float(np.linalg.norm(vz)), eps)
    kp1 = kp0 + float(kp_dx) * vx
    kp2 = kp0 + float(kp_dz) * vz
    return np.concatenate([kp0, kp1, kp2]).astype(np.float32)


def quat_from_euler_xyz_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Isaac Lab ``math_utils.quat_from_euler_xyz`` (q = qz(yaw) qy(pitch) qx(roll))."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return np.array(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ],
        dtype=np.float64,
    )


def minimum_jerk_blend(tau: float) -> float:
    tau = float(np.clip(tau, 0.0, 1.0))
    return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)


# Feature-major LL history groups (identical to Isaac Lab / deploy script).
LL_FEATURE_DIMS = [3, 3, 3, 9, 12, 6, 12, 6, 4, 22]


def split_features(frame, dims):
    out, i = [], 0
    for dim in dims:
        out.append(frame[i : i + dim].copy())
        i += dim
    assert i == frame.shape[0]
    return out


class FeatureMajorHistory:
    """Per-feature deques flattened feature-major, exactly as the deploy script."""

    def __init__(self, dims, length):
        self.dims = list(dims)
        self.length = int(length)
        self.h = [deque(maxlen=self.length) for _ in self.dims]

    def fill(self, frame):
        for hist, feat in zip(self.h, split_features(frame, self.dims)):
            hist.clear()
            for _ in range(self.length):
                hist.append(feat.copy())

    def append(self, frame):
        for hist, feat in zip(self.h, split_features(frame, self.dims)):
            hist.append(feat.copy())

    def flat(self):
        return np.concatenate([np.asarray(h, dtype=np.float32).reshape(-1) for h in self.h], dtype=np.float32)


class TransportSim:
    """One MuJoCo world + frozen WBC + frozen UAN. Reused across episodes."""

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

    def __init__(self, deploy_cfg: dict, low_policy_path: str | None = None, xml_path: str | None = None,
                 ort_threads: int = 1):
        cfg = deploy_cfg
        self.cfg = cfg
        f32 = lambda k: np.asarray(cfg[k], dtype=np.float32)  # noqa: E731

        # ---------------- timing ----------------
        self.simulation_dt = float(cfg["simulation_dt"])
        self.ll_decimation = int(cfg["ll_control_decimation"])
        self.ll_steps_per_hl = int(cfg["ll_steps_per_hl_step"])
        self.ll_dt = self.simulation_dt * self.ll_decimation
        self.hl_dt = self.ll_dt * self.ll_steps_per_hl
        assert abs(self.simulation_dt - 0.004) < 1e-12 and self.ll_decimation == 5 and self.ll_steps_per_hl == 5
        self.ll_history_length = int(cfg["ll_history_length"])
        self.ll_obs_dim = int(cfg["ll_obs_dim"])
        self.ll_action_dim = int(cfg["ll_action_dim"])
        self.hl_history_length = int(cfg["hl_history_length"])
        self.episode_length_s = float(cfg["episode_length_s"])
        # DirectRLEnv: max_episode_length = ceil(T/step_dt); timeout at >= max-1.
        self.max_episode_hl_steps = int(math.ceil(round(self.episode_length_s / self.hl_dt, 9)))

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
        self.gripper_hold_pos = float(cfg["gripper_hold_pos"])
        if str(cfg.get("gripper_mode", "pd_position")).lower() != "pd_position":
            raise ValueError("TransportSim supports gripper_mode=pd_position (deployment default).")
        self.gripper_pd_kp = float(cfg["gripper_pd_kp"])
        self.gripper_pd_kd = float(cfg["gripper_pd_kd"])
        self.gripper_pd_torque_limit = float(cfg["gripper_pd_torque_limit"])

        # ---------------- task / commands ----------------
        self.ground_z = float(cfg["ground_z"])
        self.ee_kp_dx = float(cfg["ee_kp_dx"])
        self.ee_kp_dz = float(cfg["ee_kp_dz"])
        self.ee_pos_scale = f32("ee_pos_action_residual_scale")
        self.ee_rot_scale = f32("ee_rot_action_residual_scale")
        self.ee_pos_cmd_min = np.array([cfg["kp0_x_range"][0], cfg["kp0_y_range"][0], cfg["kp0_z_range"][0]], dtype=np.float32)
        self.ee_pos_cmd_max = np.array([cfg["kp0_x_range"][1], cfg["kp0_y_range"][1], cfg["kp0_z_range"][1]], dtype=np.float32)
        self.ee_rot_cmd_min = np.array([cfg["ee_yaw_range"][0], cfg["ee_pitch_range"][0], cfg["ee_roll_range"][0]], dtype=np.float32)
        self.ee_rot_cmd_max = np.array([cfg["ee_yaw_range"][1], cfg["ee_pitch_range"][1], cfg["ee_roll_range"][1]], dtype=np.float32)
        self.canonical_ee_positions = f32("canonical_ee_positions")
        self.canonical_ee_yaw = float(cfg["canonical_ee_yaw"])
        self.canonical_ee_pitch = float(cfg["canonical_ee_pitch"])
        self.canonical_ee_roll = float(cfg["canonical_ee_roll"])
        self.settle_steps = int(round(float(cfg["reset_settle_time_s"]) / self.hl_dt))
        self.ee_fast_transition_steps = int(round(float(cfg.get("ee_fast_transition_time_s", 1.5)) / self.hl_dt))
        self.ee_low_transition_steps = int(round(float(cfg.get("ee_low_transition_time_s", 2.0)) / self.hl_dt))

        # ---------------- disk / puck ----------------
        self.disk_center_offset_gripper = f32("disk_center_offset_gripper")
        self.disk_x_axis_gripper = f32("disk_x_axis_gripper")
        self.disk_y_axis_gripper = f32("disk_y_axis_gripper")
        self.disk_z_axis_gripper = f32("disk_z_axis_gripper")
        self.disk_thickness = float(cfg["disk_thickness"])
        self.puck_radius = float(cfg["puck_radius"])
        self.puck_height = float(cfg["puck_height"])
        self.puck_spawn_clearance = float(cfg["puck_spawn_clearance"])
        self.puck_contact_threshold = float(cfg["puck_disk_contact_force_threshold"])
        self.puck_drop_no_contact_steps = int(cfg["puck_drop_no_contact_steps"])
        self.base_contact_threshold = float(cfg.get("base_contact_force_threshold", 1.0))

        # ---------------- networks ----------------
        so = ort.SessionOptions()
        if ort_threads > 0:          # 0 -> onnxruntime default (as in the deploy script)
            so.intra_op_num_threads = ort_threads
            so.inter_op_num_threads = ort_threads
        self.low_policy_path = resolve_repo_path(low_policy_path or cfg["low_policy_path"])
        self.low_sess = ort.InferenceSession(self.low_policy_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.low_in = self.low_sess.get_inputs()[0].name
        self.low_out = self.low_sess.get_outputs()[0].name
        assert self.low_sess.get_inputs()[0].shape[-1] == self.ll_obs_dim
        assert self.low_sess.get_outputs()[0].shape[-1] == self.ll_action_dim
        self.uan_model_path = resolve_repo_path(cfg["uan_model_path"])
        self.uan_model = torch.jit.load(self.uan_model_path, map_location="cpu").eval()
        for p in self.uan_model.parameters():
            p.requires_grad_(False)

        # ---------------- MuJoCo ----------------
        self.xml_path = resolve_repo_path(xml_path or cfg["xml_path"])
        m = mujoco.MjModel.from_xml_path(self.xml_path)
        m.opt.timestep = self.simulation_dt
        self.m = m
        name2id = lambda t, n: mujoco.mj_name2id(m, t, n)  # noqa: E731
        B, J, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM
        self.ee_bid = name2id(B, str(cfg.get("ee_body", "gripperStator")))
        self.base_bid = name2id(B, "base_link")
        self.puck_bid = name2id(B, str(cfg.get("puck_body", "transport_puck")))
        self.puck_jid = name2id(J, str(cfg.get("puck_freejoint", "transport_puck_freejoint")))
        self.puck_gid = name2id(G, str(cfg.get("puck_geom", "transport_puck_geom")))
        self.disk_gid = name2id(G, str(cfg.get("disk_collision_geom", "disk_collision")))
        for what, idx in (("ee", self.ee_bid), ("base", self.base_bid), ("puck body", self.puck_bid),
                          ("puck joint", self.puck_jid), ("puck geom", self.puck_gid), ("disk geom", self.disk_gid)):
            if idx < 0:
                raise ValueError(f"MuJoCo {what} not found in {self.xml_path}")
        if abs(float(m.geom_size[self.puck_gid, 0]) - self.puck_radius) > 1e-6 or \
                abs(2.0 * float(m.geom_size[self.puck_gid, 1]) - self.puck_height) > 1e-6:
            raise ValueError("Puck MJCF geometry does not match the configured radius/height.")
        if int(m.geom_priority[self.puck_gid]) <= int(m.geom_priority[self.disk_gid]):
            raise ValueError("Puck geom priority must exceed the disk's so the puck friction is effective.")
        if not (m.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD):
            raise ValueError("multiccd must be enabled for the flat puck/disk contact (see ALIGNMENT_REPORT.md).")

        # Nominal model parameters restored before every scenario.
        self.nom_base_mass = float(m.body_mass[self.base_bid])
        self.nom_base_inertia = m.body_inertia[self.base_bid].copy()
        self.nom_base_ipos = m.body_ipos[self.base_bid].copy()

        self.d = mujoco.MjData(m)
        self.puck_qpos_adr = int(m.jnt_qposadr[self.puck_jid])
        self.puck_dof_adr = int(m.jnt_dofadr[self.puck_jid])
        self.stator_gids = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == self.ee_bid], dtype=int)

        idx = self.MUJOCO_JOINT_NAMES.index
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
        self.nj = len(self.MUJOCO_JOINT_NAMES)

        # Same ctrlrange widening as the deploy script: MuJoCo must not clip
        # inside the UAN final torque envelope (the UAN law owns saturation).
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

        self.ll_history = FeatureMajorHistory(LL_FEATURE_DIMS, self.ll_history_length)
        self.uan_history = np.zeros((6, self.uan_history_length, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Scenario-dependent physical parameters
    # ------------------------------------------------------------------
    def apply_physics(self, phys: dict):
        m = self.m
        mass = float(phys["puck_mass"])
        r, h = self.puck_radius, self.puck_height
        ixy = mass * (3.0 * r**2 + h**2) / 12.0
        iz = 0.5 * mass * r**2
        m.body_mass[self.puck_bid] = mass
        m.body_inertia[self.puck_bid] = np.array([ixy, ixy, iz])
        m.geom_friction[self.puck_gid, 0] = float(phys["puck_mu"])
        # Isaac Lab randomize_rigid_body_mass(operation="add", recompute_inertia=True)
        add = float(phys.get("base_mass_add", 0.0))
        new_mass = self.nom_base_mass + add
        m.body_mass[self.base_bid] = new_mass
        m.body_inertia[self.base_bid] = self.nom_base_inertia * (new_mass / self.nom_base_mass)
        # Isaac Lab randomize_rigid_body_com: additive COM offset in the body frame.
        m.body_ipos[self.base_bid] = self.nom_base_ipos + np.asarray(phys.get("base_com_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        mujoco.mj_setConst(m, self.d)

    # ------------------------------------------------------------------
    # Kinematic helpers (identical expressions to the deploy script)
    # ------------------------------------------------------------------
    def get_root_pose(self):
        d = self.d
        return d.qpos[0:3].copy().astype(np.float32), quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))

    def get_plb_pose_world(self):
        root_pos_w, root_quat_w = self.get_root_pose()
        _, _, yaw = euler_xyz_from_quat_wxyz(root_quat_w)
        plb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(quat_from_yaw_wxyz(yaw)))
        plb_pos_w = root_pos_w.copy()
        plb_pos_w[2] = self.ground_z
        return plb_pos_w, plb_quat_w

    def compute_actual_ee_keypoints_plb(self):
        d = self.d
        plb_pos_w, plb_quat_w = self.get_plb_pose_world()
        ee_pos_w = d.xpos[self.ee_bid].copy().astype(np.float32)
        ee_rot_w = d.xmat[self.ee_bid].reshape(3, 3).copy().astype(np.float32)
        kp1_w = ee_pos_w + ee_rot_w @ np.array([self.ee_kp_dx, 0.0, 0.0], dtype=np.float32)
        kp2_w = ee_pos_w + ee_rot_w @ np.array([0.0, 0.0, self.ee_kp_dz], dtype=np.float32)
        return np.concatenate([
            quat_apply_inverse_wxyz(plb_quat_w, ee_pos_w - plb_pos_w),
            quat_apply_inverse_wxyz(plb_quat_w, kp1_w - plb_pos_w),
            quat_apply_inverse_wxyz(plb_quat_w, kp2_w - plb_pos_w),
        ]).astype(np.float32)

    def measure_ee_kp0_plb(self):
        return self.compute_actual_ee_keypoints_plb()[0:3].copy().astype(np.float32)

    def get_disk_frame_world(self):
        d = self.d
        stator_pos_w = d.xpos[self.ee_bid].copy().astype(np.float32)
        stator_rot_w = d.xmat[self.ee_bid].reshape(3, 3).copy().astype(np.float32)
        center_w = stator_pos_w + stator_rot_w @ self.disk_center_offset_gripper
        x_w = stator_rot_w @ self.disk_x_axis_gripper
        y_w = stator_rot_w @ self.disk_y_axis_gripper
        z_w = stator_rot_w @ self.disk_z_axis_gripper
        R_w_d = np.stack([x_w, y_w, z_w], axis=1).astype(np.float32)
        q_w_d = quat_unique_wxyz(quat_normalize_wxyz(quat_from_rotmat_wxyz(R_w_d)))
        return center_w.astype(np.float32), x_w.astype(np.float32), y_w.astype(np.float32), z_w.astype(np.float32), q_w_d

    def get_puck_pos_disk(self):
        center_w, x_w, y_w, z_w, _ = self.get_disk_frame_world()
        rel = self.d.xpos[self.puck_bid].copy().astype(np.float32) - center_w
        return np.array([np.dot(rel, x_w), np.dot(rel, y_w), np.dot(rel, z_w)], dtype=np.float32)

    def puck_tilt_deg(self):
        """Angle between the puck cylinder axis and the disk normal (diagnostic only)."""
        _, _, _, z_w, _ = self.get_disk_frame_world()
        axis = self.d.xmat[self.puck_bid].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(float(axis @ z_w), -1.0, 1.0))))

    def base_velocity_body(self):
        """[vx, vy, wz] of base_link in the base (body) frame, Isaac Lab root_*_vel_b."""
        _, q = self.get_root_pose()
        v_b = quat_rotate_inverse_numpy(q, self.d.qvel[0:3].astype(np.float32))
        return np.array([v_b[0], v_b[1], self.d.qvel[5]], dtype=np.float64)

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------
    def _net_force_on(self, select) -> float:
        """Norm of the net world-frame contact force on one side of the selected pairs.

        ``select(g1, g2)`` returns +1 if the force on geom2 counts, -1 if the
        force on geom1 counts, 0 to skip. mj_contactForce returns the force on geom2.
        """
        d, m = self.d, self.m
        f6 = np.zeros(6, dtype=np.float64)
        net = np.zeros(3, dtype=np.float64)
        for ci in range(d.ncon):
            c = d.contact[ci]
            s = select(c.geom1, c.geom2)
            if s:
                mujoco.mj_contactForce(m, d, ci, f6)
                net += s * (c.frame.reshape(3, 3).T @ f6[:3])
        return float(np.linalg.norm(net))

    def puck_disk_force(self) -> float:
        pg, dg = self.puck_gid, self.disk_gid
        return self._net_force_on(lambda a, b: (1 if b == pg else -1) if {a, b} == {pg, dg} else 0)

    def puck_stator_any_force(self) -> float:
        """Isaac Lab-style BODY-level gripperStator<->puck force (disk + jaw geoms)."""
        pg, sg = self.puck_gid, set(self.stator_gids.tolist())
        return self._net_force_on(
            lambda a, b: (1 if b == pg else -1) if ((a == pg and b in sg) or (b == pg and a in sg)) else 0
        )

    def base_contact_force(self) -> float:
        m, bb = self.m, self.base_bid
        return self._net_force_on(
            lambda a, b: (1 if m.geom_bodyid[b] == bb else -1)
            if (m.geom_bodyid[a] == bb or m.geom_bodyid[b] == bb) else 0
        )

    # ------------------------------------------------------------------
    # Reference process (precomputed schedule)
    # ------------------------------------------------------------------
    def _sync_ll_commands_from_references(self):
        self.base_command = self.base_velocity_ref.copy()
        self.kp0_cmd = self.ee_kp0_ref.copy()
        self.rot_cmd = np.array([self.ee_yaw_ref, self.ee_pitch_ref, self.ee_roll_ref], dtype=np.float32)
        self.ee_cmd_plb = build_keypoints_from_kp0_yaw_pitch_roll_plb(
            self.kp0_cmd, self.ee_yaw_ref, self.ee_pitch_ref, self.ee_roll_ref, self.ee_kp_dx, self.ee_kp_dz)

    def _set_episode_high_from_measured_ee(self):
        measured = self.measure_ee_kp0_plb()
        self.episode_high_kp0 = measured.copy()
        self.ee_kp0_ref = measured.copy()
        self.current_ee_pose_index = 0
        self.ee_transition_start = measured.copy()
        self.ee_transition_target = measured.copy()
        self.ee_transition_step = 0
        self.ee_transition_total = 1
        self._sync_ll_commands_from_references()

    def _next_base_reference(self):
        if self._base_i >= len(self.schedule_base):
            raise RuntimeError("Scenario base schedule exhausted; regenerate with more entries.")
        entry = self.schedule_base[self._base_i]
        self._base_i += 1
        self.base_velocity_ref = np.asarray(entry["v"], dtype=np.float64).astype(np.float32)
        self.base_ref_steps_remaining = int(entry["steps"])

    def _next_ee_reference(self):
        if self._ee_i >= len(self.schedule_ee):
            raise RuntimeError("Scenario EE schedule exhausted; regenerate with more entries.")
        entry = self.schedule_ee[self._ee_i]
        self._ee_i += 1
        idx = int(entry["target"])
        self.current_ee_pose_index = idx
        target = self.episode_high_kp0.copy() if idx == 0 else self.canonical_ee_positions[idx].copy()
        self.ee_transition_start = self.ee_kp0_ref.copy()
        self.ee_transition_target = target
        self.ee_transition_step = 0
        self.ee_transition_total = self.ee_low_transition_steps if idx == 1 else self.ee_fast_transition_steps
        self.ee_kp0_ref = self.ee_transition_start.copy()
        self.ee_yaw_ref = self.canonical_ee_yaw
        self.ee_pitch_ref = self.canonical_ee_pitch
        self.ee_roll_ref = self.canonical_ee_roll
        self.ee_ref_steps_remaining = int(entry["hold"])

    def _advance_ee_interpolation(self):
        total = max(int(self.ee_transition_total), 1)
        self.ee_transition_step = min(int(self.ee_transition_step) + 1, total)
        blend = minimum_jerk_blend(float(self.ee_transition_step) / float(total))
        self.ee_kp0_ref = (self.ee_transition_start + blend * (self.ee_transition_target - self.ee_transition_start)).astype(np.float32)

    def advance_references(self):
        """Isaac Lab ``_advance_command_references_for_next_step`` for one env."""
        if self.settle_steps_remaining > 0:
            self.settle_steps_remaining -= 1
            if self.settle_steps_remaining <= 0:
                self._set_episode_high_from_measured_ee()
                self._next_base_reference()
                self._next_ee_reference()
            return
        self.base_ref_steps_remaining -= 1
        self.ee_ref_steps_remaining -= 1
        self._advance_ee_interpolation()
        if self.base_ref_steps_remaining <= 0:
            self._next_base_reference()
        if self.ee_ref_steps_remaining <= 0:
            self._next_ee_reference()

    @property
    def settling(self) -> bool:
        return self.settle_steps_remaining > 0

    @property
    def rot_ref(self) -> np.ndarray:
        return np.array([self.ee_yaw_ref, self.ee_pitch_ref, self.ee_roll_ref], dtype=np.float32)

    def set_ee_command(self, kp0_cmd, rot_cmd):
        """Install a decoded EE command; base command is ALWAYS the base reference."""
        self.base_command = self.base_velocity_ref.copy()
        self.kp0_cmd = np.asarray(kp0_cmd, dtype=np.float32)
        self.rot_cmd = np.asarray(rot_cmd, dtype=np.float32)
        yaw, pitch, roll = map(float, self.rot_cmd)
        self.ee_cmd_plb = build_keypoints_from_kp0_yaw_pitch_roll_plb(
            self.kp0_cmd, yaw, pitch, roll, self.ee_kp_dx, self.ee_kp_dz)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def hl_obs_frame(self, prev_executed_action: np.ndarray) -> np.ndarray:
        """Actor frame: 58 shared dims + previous executed action + puck center (3)."""
        d, m = self.d, self.m
        root_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        root_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(root_quat_w, gravity_w).astype(np.float32)
        qpos = d.qpos[7 : 7 + self.nj].copy().astype(np.float32)
        qvel = d.qvel[6 : 6 + self.nj].copy().astype(np.float32)
        leg_pos_rel = (qpos[self.leg_idx] - self.default_joint_pos[self.leg_idx]).astype(np.float32)
        arm_pos_rel = (qpos[self.arm_idx] - self.default_joint_pos[self.arm_idx]).astype(np.float32)
        return np.concatenate([
            root_ang_vel_b, projected_gravity_b, leg_pos_rel, arm_pos_rel,
            qvel[self.leg_idx].astype(np.float32), qvel[self.arm_idx].astype(np.float32),
            qvel[self.wheel_idx].astype(np.float32),
            self.base_velocity_ref, self.ee_kp0_ref, self.rot_ref,
            np.asarray(prev_executed_action, dtype=np.float32),
            self.get_puck_pos_disk(),
        ], dtype=np.float32)

    def ll_obs_frame(self) -> np.ndarray:
        d, m = self.d, self.m
        qpos = d.qpos[7 : 7 + self.nj].copy().astype(np.float32)
        qvel = d.qvel[6 : 6 + self.nj].copy().astype(np.float32)
        base_ang_vel_b = get_sensor_slice(m, d, "imu_gyro").astype(np.float32)
        root_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(root_quat_w, gravity_w).astype(np.float32)
        jp = qpos[self.policy_pos_idx] - self.default_policy_pos
        jv = qvel[self.policy_vel_idx]
        obs = np.concatenate([
            base_ang_vel_b, projected_gravity_b, self.base_command, self.ee_cmd_plb,
            jp[:12], jp[12:18], jv[:12], jv[12:18], jv[18:22], self.last_ll_action,
        ], dtype=np.float32)
        assert obs.shape == (80,)
        return obs

    # ------------------------------------------------------------------
    # Low level + actuation
    # ------------------------------------------------------------------
    def run_low_level_policy(self):
        self.ll_history.append(self.ll_obs_frame())
        action = self.low_sess.run([self.low_out], {self.low_in: self.ll_history.flat()[None, :]})[0][0].astype(np.float32)
        self.last_ll_action = action
        self.leg_target = self.default_leg_pos + self.leg_action_scale * action[0:12]
        self.arm_target = self.default_arm_pos + self.arm_action_scale * action[12:18]
        self.wheel_cmd = self.wheel_action_scale * action[18:22]

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

    def apply_low_level_actuation(self):
        d = self.d
        qpos = d.qpos[7 : 7 + self.nj].copy().astype(np.float32)
        qvel = d.qvel[6 : 6 + self.nj].copy().astype(np.float32)
        li = self.leg_idx
        leg_tau = self.kps[li] * (self.leg_target - qpos[li]) - self.kds[li] * qvel[li]
        leg_tau = np.clip(leg_tau, -self.leg_torque_limits, self.leg_torque_limits).astype(np.float32)
        arm_tau = self._uan_torque(self.arm_target, qpos[self.arm_idx], qvel[self.arm_idx])
        gi = self.gripper_idx
        gripper_tau = float(np.clip(
            self.gripper_pd_kp * (self.gripper_hold_pos - float(qpos[gi])) - self.gripper_pd_kd * float(qvel[gi]),
            -self.gripper_pd_torque_limit, self.gripper_pd_torque_limit))
        wheel_ctrl = np.clip(self.wheel_cmd, -self.wheel_velocity_limits, self.wheel_velocity_limits).astype(np.float32)
        src = np.concatenate([leg_tau, arm_tau, wheel_ctrl, np.array([gripper_tau], dtype=np.float32)], dtype=np.float32)
        d.ctrl[:] = 0.0
        for ci, si in enumerate(self.ctrl_src):
            d.ctrl[ci] = src[si]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, scenario: dict):
        """Isaac Lab ``_reset_idx`` for one env, driven by a precomputed scenario."""
        m, d = self.m, self.d
        self.apply_physics(scenario["physics"])
        init = scenario["initial_state"]
        self.schedule_base = scenario["base_schedule"]
        self.schedule_ee = scenario["ee_schedule"]
        self._base_i = 0
        self._ee_i = 0

        mujoco.mj_resetData(m, d)
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        root_pos = self.root_pos_reset.astype(np.float64).copy()
        root_pos[2] += float(init.get("base_dz", 0.0))
        d.qpos[0:3] = root_pos
        roll, pitch = float(init.get("base_roll", 0.0)), float(init.get("base_pitch", 0.0))
        if roll != 0.0 or pitch != 0.0:
            q = quat_mul_wxyz(quat_from_euler_xyz_wxyz(roll, pitch, 0.0), self.root_quat_reset.astype(np.float64))
            d.qpos[3:7] = q / np.linalg.norm(q)
        else:
            d.qpos[3:7] = self.root_quat_reset
        jp = self.default_joint_pos.copy()
        arm_scale = float(init.get("arm_scale", 1.0))
        if arm_scale != 1.0:
            jp[self.arm_idx] = jp[self.arm_idx] * np.float32(arm_scale)
        d.qpos[7 : 7 + self.nj] = jp
        d.qpos[7 + self.gripper_idx] = self.gripper_hold_pos
        root_vel = np.asarray(init.get("root_vel", [0.0] * 6), dtype=np.float64)
        if np.any(root_vel != 0.0):
            d.qvel[0:3] = root_vel[0:3]                      # world-frame linear
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            d.qvel[3:6] = R.reshape(3, 3).T @ root_vel[3:6]  # freejoint angular vel is body frame
        d.qpos[self.puck_qpos_adr : self.puck_qpos_adr + 3] = np.array([0.0, 0.0, 2.0])
        d.qpos[self.puck_qpos_adr + 3 : self.puck_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        d.qvel[self.puck_dof_adr : self.puck_dof_adr + 6] = 0.0

        # _reset_command_references
        self.base_velocity_ref = np.zeros(3, dtype=np.float32)
        self.episode_high_kp0 = self.canonical_ee_positions[0].copy()
        self.ee_kp0_ref = self.episode_high_kp0.copy()
        self.ee_yaw_ref = self.canonical_ee_yaw
        self.ee_pitch_ref = self.canonical_ee_pitch
        self.ee_roll_ref = self.canonical_ee_roll
        self.current_ee_pose_index = 0
        self.ee_transition_start = self.episode_high_kp0.copy()
        self.ee_transition_target = self.episode_high_kp0.copy()
        self.ee_transition_step = 0
        self.ee_transition_total = 1
        self.settle_steps_remaining = self.settle_steps
        self.base_ref_steps_remaining = 0
        self.ee_ref_steps_remaining = 0
        self._sync_ll_commands_from_references()
        self.last_ll_action = np.zeros(self.ll_action_dim, dtype=np.float32)
        self.leg_target = self.default_leg_pos.copy()
        self.arm_target = self.default_arm_pos.copy()
        self.wheel_cmd = np.zeros(4, dtype=np.float32)
        mujoco.mj_forward(m, d)

        # Measured reset kp0 -> temporary HIGH hold reference during settle.
        self._set_episode_high_from_measured_ee()

        # Spawn the puck on the ACTUAL post-reset disk, offset in disk x/y.
        center_w, x_w, y_w, z_w, q_w_d = self.get_disk_frame_world()
        xy = np.asarray(init["puck_xy"], dtype=np.float64).astype(np.float32)
        normal_offset = 0.5 * self.disk_thickness + 0.5 * self.puck_height + self.puck_spawn_clearance
        d.qpos[self.puck_qpos_adr : self.puck_qpos_adr + 3] = center_w + xy[0] * x_w + xy[1] * y_w + normal_offset * z_w
        d.qpos[self.puck_qpos_adr + 3 : self.puck_qpos_adr + 7] = q_w_d
        d.qvel[self.puck_dof_adr : self.puck_dof_adr + 6] = 0.0
        mujoco.mj_forward(m, d)

        self.uan_history.fill(0.0)
        self.puck_no_contact_steps = 0
        self.ll_history.fill(self.ll_obs_frame())
        self.hl_steps_completed = 0

    def run_hl_interval(self, on_ll_sample=None):
        """Execute the current command for one 10-Hz step: 5 WBC decisions x 5 physics steps."""
        for k in range(self.ll_steps_per_hl):
            self.run_low_level_policy()
            for _ in range(self.ll_decimation):
                self.apply_low_level_actuation()
                mujoco.mj_step(self.m, self.d)
            if on_ll_sample is not None:
                on_ll_sample(k)
        self.hl_steps_completed += 1

    def post_step_checks(self) -> dict:
        """Isaac Lab ``_get_dones`` for the just-completed transition."""
        f_disk = self.puck_disk_force()
        contact = f_disk > self.puck_contact_threshold
        if self.settling or contact:
            self.puck_no_contact_steps = 0
        else:
            self.puck_no_contact_steps += 1
        f_base = self.base_contact_force()
        return {
            "puck_disk_force": f_disk,
            "puck_disk_contact": bool(contact),
            "puck_stator_contact": bool(self.puck_stator_any_force() > self.puck_contact_threshold),
            "puck_no_contact_steps": int(self.puck_no_contact_steps),
            "puck_drop": bool(self.puck_no_contact_steps >= self.puck_drop_no_contact_steps),
            "base_contact": bool(f_base > self.base_contact_threshold),
            "base_contact_force": f_base,
            "timeout": bool(self.hl_steps_completed >= self.max_episode_hl_steps - 1),
        }
