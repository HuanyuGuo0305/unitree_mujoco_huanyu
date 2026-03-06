"""
Sim2sim test for B2WZ1 loco-manipulation ONNX policy in MuJoCo.

Run from simulate_python/:
    python3 deploy_mujoco/b2wz1_loco_manipulation.py configs/b2wz1_loco_manipulation.yaml
"""

import os
import sys
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from utilities.math import (
    quat_angle_wxyz,
    quat_apply_inverse_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_from_keypoints_lb,
    quat_from_rotmat_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_rotate_inverse_numpy,
    quat_slerp_wxyz,
    quat_unique_wxyz,
)
from utilities.mujoco_helper import site_id


class EEKeypointsCommandLB:
    """Single-env version of PresampledKeypointsInterpolateCommandLB."""

    def __init__(
        self,
        table_path: str,
        kp_dx: float,
        kp_dz: float,
        kp0_threshold: float,
        rot_threshold: float,
    ):
        table = np.load(table_path).astype(np.float32)
        assert table.ndim == 2 and table.shape[1] == 9, f"Expected (N,9), got {table.shape}"

        self.table = table
        self.num_rows = table.shape[0]

        self.dx = float(kp_dx)
        self.dz = float(kp_dz)
        self.kp0_th = float(kp0_threshold)
        self.rot_th = float(rot_threshold)

        self.has_cmd = False
        self.cmd_lb = np.zeros(9, dtype=np.float32)

    def _kps_from_pose(self, kp0: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        """Rebuild kp1/kp2 from kp0 and orientation."""
        off_x = np.array([self.dx, 0.0, 0.0], dtype=np.float32)
        off_z = np.array([0.0, 0.0, self.dz], dtype=np.float32)

        kp1 = kp0 + quat_apply_wxyz(quat_wxyz, off_x)
        kp2 = kp0 + quat_apply_wxyz(quat_wxyz, off_z)
        return np.concatenate([kp0, kp1, kp2]).astype(np.float32)

    def resample(self) -> np.ndarray:
        """Sample a new LB keypoint command with thresholded interpolation."""
        kps_s = self.table[np.random.randint(0, self.num_rows)].copy()

        # First command: use raw sample directly
        if not self.has_cmd:
            self.cmd_lb = kps_s
            self.has_cmd = True
            return self.cmd_lb

        kp0_s, kp1_s, kp2_s = kps_s[0:3], kps_s[3:6], kps_s[6:9]
        kp0_p, kp1_p, kp2_p = self.cmd_lb[0:3], self.cmd_lb[3:6], self.cmd_lb[6:9]

        quat_p = quat_from_keypoints_lb(kp0_p, kp1_p, kp2_p, self.dx, self.dz)
        quat_s = quat_from_keypoints_lb(kp0_s, kp1_s, kp2_s, self.dx, self.dz)

        delta = kp0_s - kp0_p
        dist = max(float(np.linalg.norm(delta)), 1e-8)
        alpha_pos = min(1.0, self.kp0_th / dist)

        ang = max(quat_angle_wxyz(quat_p, quat_s), 1e-8)
        alpha_rot = min(1.0, self.rot_th / ang)

        alpha = min(alpha_pos, alpha_rot)

        if (dist <= self.kp0_th) and (ang <= self.rot_th):
            alpha = 1.0

        kp0_new = kp0_p + alpha * delta
        quat_new = quat_slerp_wxyz(quat_p, quat_s, alpha)

        self.cmd_lb = self._kps_from_pose(kp0_new, quat_new)
        return self.cmd_lb


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path to b2wz1_loco_manipulation.yaml")
    args = parser.parse_args()

    # 1) Load config
    yaml_path = os.path.abspath(args.yaml_path)
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    policy_path = cfg["policy_path"]
    xml_path = cfg["xml_path"]
    kp_table_path = cfg["kp_table_path"]

    if not os.path.isabs(policy_path):
        policy_path = os.path.abspath(os.path.join(project_root, policy_path))
    if not os.path.isabs(xml_path):
        xml_path = os.path.abspath(os.path.join(project_root, xml_path))
    if not os.path.isabs(kp_table_path):
        kp_table_path = os.path.abspath(os.path.join(project_root, kp_table_path))

    sim_duration = float(cfg["simulation_duration"])
    dt = float(cfg["simulation_dt"])
    decim = int(cfg["control_decimation"])
    render = bool(cfg.get("render", True))

    history_len = int(cfg["history_length"])
    obs_dim_per_step = int(cfg["obs_dim_per_step"])
    obs_dim = int(cfg["obs_dim"])
    act_dim = int(cfg["action_dim"])

    base_cmd = np.array(cfg["base_command"], dtype=np.float32)

    ee_resample_s = float(cfg["ee_cmd_resampling_time_s"])
    kp_dx = float(cfg["kp_dx"])
    kp_dz = float(cfg["kp_dz"])
    kp0_th = float(cfg["kp0_threshold"])
    rot_th = float(cfg["rot_threshold"])

    root_pos = np.array(cfg["root_pos"], dtype=np.float32)
    root_quat = np.array(cfg["root_quat_wxyz"], dtype=np.float32)

    default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)

    leg_scale = float(cfg["leg_action_scale"])
    arm_scale = float(cfg["arm_action_scale"])
    wheel_scale = float(cfg["wheel_action_scale"])

    leg_kps = np.array(cfg["leg_kps"], dtype=np.float32)
    leg_kds = np.array(cfg["leg_kds"], dtype=np.float32)
    arm_kps = np.array(cfg["arm_kps"], dtype=np.float32)
    arm_kds = np.array(cfg["arm_kds"], dtype=np.float32)

    leg_torque_limits = np.array(cfg["leg_torque_limits"], dtype=np.float32)  # [hip, thigh, calf]
    arm_torque_limit = float(cfg["arm_torque_limit"])
    wheel_vel_limit = float(cfg["wheel_vel_limit"])

    ee_site_name = str(cfg["ee_site"])

    assert obs_dim_per_step == 89 and obs_dim == 89 * history_len, "Obs dims mismatch"
    assert act_dim == 22, "Action dim mismatch"

    print("=" * 70)
    print("B2WZ1 Loco-Manip - ONNX Policy (MuJoCo sim2sim)")
    print("=" * 70)
    print(f"Policy: {policy_path}")
    print(f"XML:    {xml_path}")
    print(f"KP npy: {kp_table_path}")
    print(f"Control freq: {1.0 / (dt * decim):.1f} Hz")
    print("=" * 70)

    # 2) Load ONNX
    sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    print("ONNX loaded:")
    print(" input :", input_name, sess.get_inputs()[0].shape)
    print(" output:", output_name, sess.get_outputs()[0].shape)
    print("=" * 70)

    # 3) Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = dt

    ee_sid = site_id(m, ee_site_name)

    # 4) Joint mapping
    # MuJoCo qpos/qvel order used in XML home layout
    mujoco_joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper",
    ]

    # MuJoCo d.ctrl order follows XML actuator declaration order
    ctrl_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper",
    ]

    # Actual policy observation order after IsaacLab reordering
    policy_joint_pos_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]

    policy_joint_vel_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "joint1",
        "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
        "joint2", "joint3", "joint4", "joint5", "joint6",
    ]

    # Actual policy action order:
    #   12 leg + joint1 + 4 wheel + joint2..joint6
    leg_action_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    arm_action_joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]
    wheel_action_joint_names = [
        "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
    ]

    policy_action_joint_names = (
        leg_action_joint_names
        + ["joint1"]
        + wheel_action_joint_names
        + ["joint2", "joint3", "joint4", "joint5", "joint6"]
    )

    num_leg_joints = len(leg_action_joint_names)
    num_arm_joints = len(arm_action_joint_names)
    num_wheel_joints = len(wheel_action_joint_names)
    assert len(policy_action_joint_names) == act_dim

    # MuJoCo -> policy observation reorder
    mujoco_to_policy_joint_pos_indices = [
        mujoco_joint_names.index(name) for name in policy_joint_pos_names
    ]
    mujoco_to_policy_joint_vel_indices = [
        mujoco_joint_names.index(name) for name in policy_joint_vel_names
    ]

    # MuJoCo -> subsets used by low-level controller
    leg_mujoco_joint_indices = [mujoco_joint_names.index(name) for name in leg_action_joint_names]
    arm_mujoco_joint_indices = [mujoco_joint_names.index(name) for name in arm_action_joint_names]
    wheel_mujoco_joint_indices = [mujoco_joint_names.index(name) for name in wheel_action_joint_names]

    # Policy action -> subsets
    leg_action_policy_indices = [policy_action_joint_names.index(name) for name in leg_action_joint_names]
    arm_action_policy_indices = [policy_action_joint_names.index(name) for name in arm_action_joint_names]
    wheel_action_policy_indices = [policy_action_joint_names.index(name) for name in wheel_action_joint_names]

    # Control source order used when we concatenate [leg_tau, wheel_ctrl, arm_tau]
    control_source_joint_names = (
        leg_action_joint_names
        + wheel_action_joint_names
        + arm_action_joint_names
    )

    # Reorder from control_source_joint_names -> ctrl_joint_names
    control_source_to_ctrl_indices = [
        control_source_joint_names.index(name)
        for name in ctrl_joint_names
        if name in control_source_joint_names
    ]

    assert len(default_joint_pos) == len(mujoco_joint_names), (
        f"default_joint_pos length mismatch: {len(default_joint_pos)} vs {len(mujoco_joint_names)}"
    )

    # Default joint positions in required orders
    default_joint_pos_policy_jointpos = default_joint_pos[mujoco_to_policy_joint_pos_indices]

    default_leg_pos_policy = np.array(
        [default_joint_pos[mujoco_joint_names.index(name)] for name in leg_action_joint_names],
        dtype=np.float32,
    )
    default_arm_pos_policy = np.array(
        [default_joint_pos[mujoco_joint_names.index(name)] for name in arm_action_joint_names],
        dtype=np.float32,
    )
    default_wheel_pos_policy = np.array(
        [default_joint_pos[mujoco_joint_names.index(name)] for name in wheel_action_joint_names],
        dtype=np.float32,
    )

    print("Joint mapping:")
    print(f" MuJoCo joint order         : {mujoco_joint_names}\n")
    print(f" Ctrl joint order           : {ctrl_joint_names}\n")
    print(f" Policy joint_pos order     : {policy_joint_pos_names}\n")
    print(f" Policy joint_vel order     : {policy_joint_vel_names}\n")
    print(f" Policy action order        : {policy_action_joint_names}\n")
    print(f" default_leg_pos_policy     : {default_leg_pos_policy}")
    print(f" default_arm_pos_policy     : {default_arm_pos_policy}")
    print(f" default_wheel_pos_policy   : {default_wheel_pos_policy}")
    print("joint_pos index mapping:")
    for i, idx in enumerate(mujoco_to_policy_joint_pos_indices):
        print(f"  pos[{i:2d}] <- mujoco[{idx:2d}] = {mujoco_joint_names[idx]}")

    print("joint_vel index mapping:")
    for i, idx in enumerate(mujoco_to_policy_joint_vel_indices):
        print(f"  vel[{i:2d}] <- mujoco[{idx:2d}] = {mujoco_joint_names[idx]}")
    print("=" * 70)

    # 5) Initialize robot state
    d.qpos[:] = 0.0
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0

    # Root state
    d.qpos[0:3] = root_pos
    d.qpos[3:7] = root_quat

    # Joint state in MuJoCo joint order
    d.qpos[7:7 + len(mujoco_joint_names)] = default_joint_pos

    mujoco.mj_forward(m, d)
    print(f"Initialized height z = {d.qpos[2]:.3f} m")

    # 6) EE command generator
    cmd_gen = EEKeypointsCommandLB(kp_table_path, kp_dx, kp_dz, kp0_th, rot_th)
    ee_cmd_lb = cmd_gen.resample()
    next_cmd_t = ee_resample_s

    # 7) Helper: EE current keypoints in Level-Base frame
    def compute_ee_current_kp_lb() -> np.ndarray:
        base_pos_w = d.qpos[0:3].copy().astype(np.float32)
        base_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))

        w, x, y, z = base_quat_w
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        lb_quat_w = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float32)
        lb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(lb_quat_w))

        ee_pos_w = d.site_xpos[ee_sid].copy().astype(np.float32)
        ee_rot_w = d.site_xmat[ee_sid].reshape(3, 3).copy().astype(np.float32)
        ee_quat_w = quat_from_rotmat_wxyz(ee_rot_w)

        ee_pos_lb = quat_apply_inverse_wxyz(lb_quat_w, ee_pos_w - base_pos_w)
        ee_quat_lb = quat_mul_wxyz(quat_conjugate_wxyz(lb_quat_w), ee_quat_w)
        ee_quat_lb = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_lb))

        off_x = np.array([kp_dx, 0.0, 0.0], dtype=np.float32)
        off_z = np.array([0.0, 0.0, kp_dz], dtype=np.float32)

        kp0 = ee_pos_lb
        kp1 = ee_pos_lb + quat_apply_wxyz(ee_quat_lb, off_x)
        kp2 = ee_pos_lb + quat_apply_wxyz(ee_quat_lb, off_z)
        return np.concatenate([kp0, kp1, kp2]).astype(np.float32)

    # 8) Build one-step policy observation
    last_action = np.zeros(act_dim, dtype=np.float32)

    def build_obs_step() -> np.ndarray:
        qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy()
        qvel_mujoco = d.qvel[6:6 + len(mujoco_joint_names)].copy()

        base_ang_vel_b = d.qvel[3:6].copy().astype(np.float32)

        base_quat_w = quat_unique_wxyz(d.qpos[3:7].copy().astype(np.float32))
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity_b = quat_rotate_inverse_numpy(base_quat_w, gravity_w)

        ee_cur_lb = compute_ee_current_kp_lb()
        # ee_cur_lb = ee_cmd_lb.copy()  # for debugging

        joint_pos_policy = qpos_mujoco[mujoco_to_policy_joint_pos_indices]
        joint_pos_rel = joint_pos_policy - default_joint_pos_policy_jointpos

        joint_vel_policy = qvel_mujoco[mujoco_to_policy_joint_vel_indices]

        obs = np.concatenate(
            [
                base_ang_vel_b,          # 3
                projected_gravity_b,     # 3
                base_cmd,                # 3
                ee_cmd_lb,               # 9
                ee_cur_lb,               # 9
                joint_pos_rel,           # 18
                joint_vel_policy,        # 22
                last_action,             # 22
            ],
            dtype=np.float32,
        )

        assert obs.shape[0] == obs_dim_per_step, (
            f"Obs dim mismatch: {obs.shape[0]} vs {obs_dim_per_step}"
        )
        return obs

    # 9) Initial targets / history
    leg_target = default_leg_pos_policy.copy()
    arm_target = default_arm_pos_policy.copy()
    wheel_cmd = np.zeros(num_wheel_joints, dtype=np.float32)

    # Build initial one-step observation
    obs0 = build_obs_step()

    i = 0
    obs0_base_ang_vel = obs0[i:i+3].copy(); i += 3
    obs0_projected_gravity = obs0[i:i+3].copy(); i += 3
    obs0_base_cmd = obs0[i:i+3].copy(); i += 3
    obs0_ee_cmd = obs0[i:i+9].copy(); i += 9
    obs0_ee_cur = obs0[i:i+9].copy(); i += 9
    obs0_joint_pos = obs0[i:i+18].copy(); i += 18
    obs0_joint_vel = obs0[i:i+22].copy(); i += 22
    obs0_last_action = obs0[i:i+22].copy(); i += 22

    # History buffers: match IsaacLab / b2w_locomotion layout
    ang_vel_hist = deque(maxlen=history_len)
    gravity_hist = deque(maxlen=history_len)
    cmd_hist = deque(maxlen=history_len)
    ee_cmd_hist = deque(maxlen=history_len)
    ee_cur_hist = deque(maxlen=history_len)
    jpos_hist = deque(maxlen=history_len)
    jvel_hist = deque(maxlen=history_len)
    act_hist = deque(maxlen=history_len)

    for _ in range(history_len):
        ang_vel_hist.append(obs0_base_ang_vel.copy())
        gravity_hist.append(obs0_projected_gravity.copy())
        cmd_hist.append(obs0_base_cmd.copy())
        ee_cmd_hist.append(obs0_ee_cmd.copy())
        ee_cur_hist.append(obs0_ee_cur.copy())
        jpos_hist.append(obs0_joint_pos.copy())
        jvel_hist.append(obs0_joint_vel.copy())
        act_hist.append(obs0_last_action.copy())

    # 10) Simulation loop
    counter = 0
    sim_time = 0.0

    with mujoco.viewer.launch_passive(m, d) as viewer:

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = d.qpos[:3]

        start_time = time.time()

        while viewer.is_running() and sim_time < sim_duration:

            step_start = time.time()

            # Low-level control (PD + wheel velocity)

            qpos_mujoco = d.qpos[7:7 + len(mujoco_joint_names)].copy()
            qvel_mujoco = d.qvel[6:6 + len(mujoco_joint_names)].copy()

            leg_pos = qpos_mujoco[leg_mujoco_joint_indices]  # leg position in policy order
            leg_vel = qvel_mujoco[leg_mujoco_joint_indices]  # leg velocity in policy order

            arm_pos = qpos_mujoco[arm_mujoco_joint_indices]  # arm position in policy order
            arm_vel = qvel_mujoco[arm_mujoco_joint_indices]  # arm velocity in policy order

            # PD torques
            leg_tau = leg_kps * (leg_target - leg_pos) - leg_kds * leg_vel
            arm_tau = arm_kps * (arm_target - arm_pos) - arm_kds * arm_vel

            # clamp torques
            leg_tau = np.clip(
                leg_tau,
                -np.repeat(leg_torque_limits, 4),
                np.repeat(leg_torque_limits, 4),
            )

            arm_tau = np.clip(arm_tau, -arm_torque_limit, arm_torque_limit)

            wheel_ctrl = np.clip(wheel_cmd, -wheel_vel_limit, wheel_vel_limit)

            # Build ctrl vector

            ctrl_source = np.concatenate(
                [
                    leg_tau,
                    wheel_ctrl,
                    arm_tau,
                ],
                dtype=np.float32,
            )  # in [leg, wheel, arm] policy order

            ctrl_ordered = ctrl_source[control_source_to_ctrl_indices]  # from ctrl source order to ctrl order

            d.ctrl[:] = 0.0
            d.ctrl[: len(ctrl_ordered)] = ctrl_ordered

            # Step MuJoCo

            mujoco.mj_step(m, d)
            viewer.cam.lookat[:] = d.qpos[:3]

            sim_time += dt

            # Policy inference

            if counter % decim == 0:

                # EE command resampling
                if sim_time >= next_cmd_t:
                    ee_cmd_lb = cmd_gen.resample()
                    next_cmd_t += ee_resample_s

                obs_step = build_obs_step()

                i = 0
                ang = obs_step[i:i+3]; i += 3
                grav = obs_step[i:i+3]; i += 3
                cmd = obs_step[i:i+3]; i += 3
                ee_cmd = obs_step[i:i+9]; i += 9
                ee_cur = obs_step[i:i+9]; i += 9
                jpos = obs_step[i:i+18]; i += 18
                jvel = obs_step[i:i+22]; i += 22
                act = obs_step[i:i+22]; i += 22

                ang_vel_hist.append(ang.copy())
                gravity_hist.append(grav.copy())
                cmd_hist.append(cmd.copy())
                ee_cmd_hist.append(ee_cmd.copy())
                ee_cur_hist.append(ee_cur.copy())
                jpos_hist.append(jpos.copy())
                jvel_hist.append(jvel.copy())
                act_hist.append(act.copy())

                # Stack history

                obs_stack = np.concatenate(
                    [
                        np.array(ang_vel_hist).reshape(-1),
                        np.array(gravity_hist).reshape(-1),
                        np.array(cmd_hist).reshape(-1),
                        np.array(ee_cmd_hist).reshape(-1),
                        np.array(ee_cur_hist).reshape(-1),
                        np.array(jpos_hist).reshape(-1),
                        np.array(jvel_hist).reshape(-1),
                        np.array(act_hist).reshape(-1),
                    ],
                    dtype=np.float32,
                )

                assert obs_stack.shape[0] == obs_dim

                # Run policy

                action = sess.run(
                    [output_name],
                    {input_name: obs_stack[None, :]},
                )[0][0].astype(np.float32)

                # action = np.zeros_like(action)  # for debugging
                
                last_action[:] = action

                # Split actions

                leg_act = action[leg_action_policy_indices]
                arm_act = action[arm_action_policy_indices]
                wheel_act = action[wheel_action_policy_indices]

                # Convert to targets

                leg_target = default_leg_pos_policy + leg_scale * leg_act
                arm_target = default_arm_pos_policy + arm_scale * arm_act
                wheel_cmd = wheel_scale * wheel_act

            # Logging

            if counter % 200 == 0:

                print(
                    f"[{counter:6d}] "
                    f"t={sim_time:6.2f}s "
                    f"z={d.qpos[2]:.3f} | "
                    f"leg=[{last_action[leg_action_policy_indices].min():+.2f},"
                    f"{last_action[leg_action_policy_indices].max():+.2f}] "
                    f"arm=[{last_action[arm_action_policy_indices].min():+.2f},"
                    f"{last_action[arm_action_policy_indices].max():+.2f}] "
                    f"wheel=[{last_action[wheel_action_policy_indices].min():+.2f},"
                    f"{last_action[wheel_action_policy_indices].max():+.2f}]"
                )

            counter += 1

            viewer.sync()

            # realtime sync
            time_until_next = dt - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)