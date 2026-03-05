import time
import os
import sys
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import yaml
import onnxruntime as ort

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

# Config dir
CONFIG_DATA_DIR = os.path.join(project_root, "simulate_python")

# Math utils (your file)
from utilities.math import (
    quat_rotate_inverse_numpy,
    quat_slerp_wxyz,
    quat_from_keypoints_lb,
)

# -----------------------------
# Helpers: mujoco joint indexing
# -----------------------------
def get_hinge_qpos_qvel_indices(m: mujoco.MjModel, joint_name: str):
    """Return (qpos_index, qvel_index) for a 1-DOF hinge joint by name."""
    j_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if j_id < 0:
        raise ValueError(f"[ERROR] Joint not found in MuJoCo model: {joint_name}")

    qpos_adr = m.jnt_qposadr[j_id]
    dof_adr = m.jnt_dofadr[j_id]
    return int(qpos_adr), int(dof_adr)


def get_actuator_id(m: mujoco.MjModel, actuator_name: str) -> int:
    a_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if a_id < 0:
        raise ValueError(f"[ERROR] Actuator not found in MuJoCo model: {actuator_name}")
    return int(a_id)


def yaw_from_quat_wxyz(q):
    """Extract yaw from quaternion [w,x,y,z]."""
    w, x, y, z = q
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(t3, t4))


def rotz(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def world_to_level_base(base_pos_w, base_quat_wxyz, p_w):
    """
    Level-Base (LB): yaw-only frame at base position.
    p_lb = Rz(yaw)^T * (p_w - base_pos)
    """
    yaw = yaw_from_quat_wxyz(base_quat_wxyz)
    R = rotz(yaw)  # LB->W
    return (R.T @ (p_w - base_pos_w)).astype(np.float32)


def get_body_pose_world(m, d, body_name: str):
    b_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if b_id < 0:
        raise ValueError(f"[ERROR] Body not found: {body_name}")
    p = d.xpos[b_id].copy()
    R = d.xmat[b_id].reshape(3, 3).copy()  # body->world rotation matrix
    return p.astype(np.float32), R.astype(np.float32)


# -----------------------------
# EE keypoints helpers (9D)
# -----------------------------
def split_kps_9(kps_9: np.ndarray):
    kp0 = kps_9[0:3]
    kp1 = kps_9[3:6]
    kp2 = kps_9[6:9]
    return kp0, kp1, kp2


def pack_kps_9(kp0: np.ndarray, kp1: np.ndarray, kp2: np.ndarray):
    return np.concatenate([kp0, kp1, kp2]).astype(np.float32)


def quat_angle_wxyz(q0: np.ndarray, q1: np.ndarray) -> float:
    """Relative rotation angle (rad) between two unit quats (wxyz), shortest path."""
    q0 = q0.astype(np.float32)
    q1 = q1.astype(np.float32)
    dot = float(np.abs(np.dot(q0, q1)))
    dot = float(np.clip(dot, 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def kps_from_pose_lb(kp0: np.ndarray, quat_wxyz: np.ndarray, dx: float, dz: float):
    """
    Given kp0 position and ee orientation (quat in LB), rebuild kp1=kp0+R*x*dx and kp2=kp0+R*z*dz.
    We can reuse quat_apply logic by converting quat to rotation matrix via sampling axes:
    But simplest: build via quat_from_keypoints_lb expects kp1/kp2; here we need kp1/kp2.
    We'll approximate by extracting axes from quat using standard formula by rotating basis vectors.
    """
    # rotate basis vectors using quaternion math (we'll implement minimal apply here)
    w, x, y, z = quat_wxyz
    # rotation matrix (wxyz)
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)

    x_axis = R[:, 0]
    z_axis = R[:, 2]
    kp1 = kp0 + x_axis * float(dx)
    kp2 = kp0 + z_axis * float(dz)
    return kp1.astype(np.float32), kp2.astype(np.float32)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config yaml file name (in simulate_python/configs)")
    parser.add_argument("--no_policy", action="store_true", help="disable ONNX policy, hold default pose")
    args = parser.parse_args()
    config_file = args.config_file

    # Load configs
    config_path = f"{CONFIG_DATA_DIR}/configs/{config_file}"
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = cfg["policy_path"]
    xml_path = cfg["xml_path"]

    simulation_duration = float(cfg["simulation_duration"])
    simulation_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])

    num_actions_total = int(cfg["num_actions_total"])   # 23 (includes gripper)
    num_actions_policy = int(cfg["num_actions_policy"]) # 22 (no gripper)

    default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)  # len 23

    # gains
    leg_kps = np.array(cfg["leg_kps"], dtype=np.float32)   # 12
    leg_kds = np.array(cfg["leg_kds"], dtype=np.float32)   # 12
    wheel_kps = np.array(cfg["wheel_kps"], dtype=np.float32)  # 4 (usually 0)
    wheel_kds = np.array(cfg["wheel_kds"], dtype=np.float32)  # 4

    arm_kps = np.array(cfg["arm_kps"], dtype=np.float32)  # 6
    arm_kds = np.array(cfg["arm_kds"], dtype=np.float32)  # 6
    arm_torque_limits = np.array(cfg["arm_torque_limits"], dtype=np.float32)  # 6

    leg_action_scale = float(cfg["leg_action_scale"])
    arm_action_scale = float(cfg["arm_action_scale"])
    wheel_action_scale = float(cfg["wheel_action_scale"])

    base_command = np.array(cfg["base_command"], dtype=np.float32)  # (3,)

    # obs
    history_length = int(cfg.get("num_history", 5))
    num_obs = int(cfg["num_obs"])  # expected 445

    # ee keypoints
    ee_body_name = cfg["ee_body_name"]  # "gripperStator"
    kp_dx = float(cfg["kp_dx"])
    kp_dz = float(cfg["kp_dz"])
    kp0_threshold = float(cfg.get("kp0_threshold", 0.20))
    rot_threshold = float(cfg.get("rot_threshold", 0.40))
    ee_kp_npy_path = cfg["ee_kp_npy_path"]
    ee_resample_time = float(cfg["ee_resample_time"])

    # gripper
    gripper_hold_torque = float(cfg.get("gripper_hold_torque", 0.0))

    # arm target rate limit (emulate DC motor velocity limit)
    arm_velocity_limit = float(cfg.get("arm_velocity_limit", 3.0))  # rad/s

    debug_every = int(cfg.get("debug_every", 200))

    # Resolve paths
    if not os.path.isabs(policy_path):
        policy_path = os.path.join(project_root, policy_path)
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(project_root, xml_path)
    if not os.path.isabs(ee_kp_npy_path):
        ee_kp_npy_path = os.path.join(project_root, ee_kp_npy_path)

    USE_POLICY = (not args.no_policy)

    print("=" * 80)
    print("B2WZ1 LocoManipulation - MuJoCo sim2sim (Aligned to NEW training obs)")
    print("=" * 80)
    print(f"XML:    {xml_path}")
    print(f"Policy: {policy_path}")
    print(f"USE_POLICY: {USE_POLICY}")
    print(f"dt={simulation_dt}, control_decimation={control_decimation}, control_hz={1.0/(simulation_dt*control_decimation):.1f}")
    print(f"history_length={history_length}, num_obs={num_obs}")
    print("=" * 80)

    # Load reachable kp commands
    reachable_kp = np.load(ee_kp_npy_path).astype(np.float32)
    if reachable_kp.ndim != 2 or reachable_kp.shape[1] != 9:
        raise ValueError(f"[ERROR] ee_kp_npy must be shape (N,9), got {reachable_kp.shape}")
    print(f"[INFO] Loaded reachable EE keypoints: {reachable_kp.shape} from {ee_kp_npy_path}")

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # -----------------------------
    # RL naming -> MuJoCo naming (wheels)
    # -----------------------------
    rl_to_mj_name = {
        "FL_foot_joint": "FL_wheel_joint",
        "FR_foot_joint": "FR_wheel_joint",
        "RL_foot_joint": "RL_wheel_joint",
        "RR_foot_joint": "RR_wheel_joint",
    }

    def mj_name(rl_name: str) -> str:
        return rl_to_mj_name.get(rl_name, rl_name)

    # -----------------------------
    # NEW TRAINING ORDER (you set in cfg):
    #   joint_pos = legs12 + arm6
    #   joint_vel = legs12 + wheels4 + arm6
    # Action order:
    #   legs12 + arm6 + wheel_vel4  (total 22)
    # -----------------------------
    leg_names_12 = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    arm_names_6 = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    wheel_names_4_rl = ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"]
    wheel_names_4_mj = [mj_name(n) for n in wheel_names_4_rl]

    obs_joint_pos_names_18 = leg_names_12 + arm_names_6
    obs_joint_vel_names_22 = leg_names_12 + wheel_names_4_rl + arm_names_6

    # -----------------------------
    # Build qpos/qvel indices
    # -----------------------------
    qpos_idx = {}
    qvel_idx = {}

    all_needed = set(obs_joint_pos_names_18 + obs_joint_vel_names_22 + ["jointGripper"])
    for j in all_needed:
        mj_j = mj_name(j)
        qi, vi = get_hinge_qpos_qvel_indices(m, mj_j)
        qpos_idx[j] = qi
        qvel_idx[j] = vi

    # -----------------------------
    # Actuator IDs (set d.ctrl by actuator id, avoids actuator order bugs)
    # -----------------------------
    actuator_names = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "FR_wheel", "FL_wheel", "RR_wheel", "RL_wheel",
        "motor1", "motor2", "motor3", "motor4", "motor5", "motor6",
        "motorGripper",
    ]
    act_id = {name: get_actuator_id(m, name) for name in actuator_names}

    leg_joint_to_act = {
        "FR_hip_joint": "FR_hip",
        "FR_thigh_joint": "FR_thigh",
        "FR_calf_joint": "FR_calf",
        "FL_hip_joint": "FL_hip",
        "FL_thigh_joint": "FL_thigh",
        "FL_calf_joint": "FL_calf",
        "RR_hip_joint": "RR_hip",
        "RR_thigh_joint": "RR_thigh",
        "RR_calf_joint": "RR_calf",
        "RL_hip_joint": "RL_hip",
        "RL_thigh_joint": "RL_thigh",
        "RL_calf_joint": "RL_calf",
    }
    wheel_joint_to_act = {
        "FL_wheel_joint": "FL_wheel",
        "FR_wheel_joint": "FR_wheel",
        "RL_wheel_joint": "RL_wheel",
        "RR_wheel_joint": "RR_wheel",
    }
    arm_joint_to_act = {
        "joint1": "motor1",
        "joint2": "motor2",
        "joint3": "motor3",
        "joint4": "motor4",
        "joint5": "motor5",
        "joint6": "motor6",
    }
    gripper_act_name = "motorGripper"

    # -----------------------------
    # Init state
    # -----------------------------
    d.qpos[0:3] = [0.0, 0.0, 0.60]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    # YAML default_joint_pos order (MuJoCo order):
    mj_default_order_23 = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        "jointGripper",
    ]
    if len(default_joint_pos) != 23:
        raise ValueError(f"[ERROR] default_joint_pos must be len 23, got {len(default_joint_pos)}")

    for val, jn in zip(default_joint_pos, mj_default_order_23):
        qi, _ = get_hinge_qpos_qvel_indices(m, jn)
        d.qpos[qi] = float(val)

    d.qvel[:6] = np.random.uniform(-0.1, 0.1, size=6).astype(np.float32)
    mujoco.mj_forward(m, d)

    # -----------------------------
    # Load ONNX
    # -----------------------------
    sess = None
    input_name = None
    output_name = None
    if USE_POLICY:
        sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        in_shape = sess.get_inputs()[0].shape
        out_shape = sess.get_outputs()[0].shape
        print(f"[INFO] ONNX loaded. Input: {input_name} {in_shape}, Output: {output_name} {out_shape}")

        if in_shape[-1] != num_obs:
            print(f"[WARN] ONNX expects obs dim {in_shape[-1]} but cfg num_obs={num_obs}")
        if out_shape[-1] != num_actions_policy:
            print(f"[WARN] ONNX outputs {out_shape[-1]} but cfg num_actions_policy={num_actions_policy}")

    # -----------------------------
    # Buffers
    # -----------------------------
    gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # Targets
    leg_default = np.array([d.qpos[qpos_idx[n]] for n in leg_names_12], dtype=np.float32)
    arm_default = np.array([d.qpos[qpos_idx[n]] for n in arm_names_6], dtype=np.float32)

    leg_target = leg_default.copy()
    arm_target = arm_default.copy()
    wheel_v_des = np.zeros(4, dtype=np.float32)

    # arm rate limit per control step
    dt_control = simulation_dt * control_decimation
    arm_max_delta = arm_velocity_limit * dt_control

    actions_22 = np.zeros(num_actions_policy, dtype=np.float32)
    last_actions_22 = np.zeros_like(actions_22)

    # EE kp command state (NO nonlocal)
    ee_kp_cmd = reachable_kp[np.random.randint(0, reachable_kp.shape[0])].copy().astype(np.float32)
    has_cmd = [False]
    last_kp_resample_t = 0.0

    def resample_ee_kp_command():
        """Thresholded interpolation on kp0 + relative rotation, like training."""
        kps_s = reachable_kp[np.random.randint(0, reachable_kp.shape[0])].copy().astype(np.float32)

        if not has_cmd[0]:
            ee_kp_cmd[:] = kps_s
            has_cmd[0] = True
            return

        kp0_s, kp1_s, kp2_s = split_kps_9(kps_s)
        kp0_p, kp1_p, kp2_p = split_kps_9(ee_kp_cmd)

        quat_p = quat_from_keypoints_lb(kp0_p, kp1_p, kp2_p, kp_dx, kp_dz)
        quat_s = quat_from_keypoints_lb(kp0_s, kp1_s, kp2_s, kp_dx, kp_dz)

        delta = (kp0_s - kp0_p).astype(np.float32)
        dist = float(np.linalg.norm(delta) + 1e-8)
        alpha_pos = min(1.0, kp0_threshold / dist)

        ang = quat_angle_wxyz(quat_p, quat_s)
        ang = max(ang, 1e-8)
        alpha_rot = min(1.0, rot_threshold / ang)

        alpha = min(alpha_pos, alpha_rot)

        need = (dist > kp0_threshold) or (ang > rot_threshold)
        alpha_eff = alpha if need else 1.0

        kp0_new = kp0_p + float(alpha_eff) * delta
        quat_new = quat_slerp_wxyz(quat_p, quat_s, float(alpha_eff))
        kp1_new, kp2_new = kps_from_pose_lb(kp0_new, quat_new, kp_dx, kp_dz)

        ee_kp_cmd[:] = pack_kps_9(kp0_new, kp1_new, kp2_new)
        has_cmd[0] = True

    # History buffers
    H = history_length
    base_ang_hist = deque(maxlen=H)  # (3,)
    grav_hist = deque(maxlen=H)      # (3,)
    cmd_hist = deque(maxlen=H)       # (3,)
    ee_cmd_hist = deque(maxlen=H)    # (9,)
    ee_cur_hist = deque(maxlen=H)    # (9,)
    jpos_hist = deque(maxlen=H)      # (18,)
    jvel_hist = deque(maxlen=H)      # (22,)
    act_hist = deque(maxlen=H)       # (22,)

    def get_obs_step_vectors():
        # base
        base_quat = d.qpos[3:7].copy().astype(np.float32)
        base_ang_vel_b = d.qvel[3:6].copy().astype(np.float32)
        grav_b = quat_rotate_inverse_numpy(base_quat, gravity_w)

        # cmd
        cmd = base_command.copy().astype(np.float32)

        # ee current kp in LB
        base_pos_w = d.qpos[0:3].copy().astype(np.float32)
        ee_pos_w, ee_R_w = get_body_pose_world(m, d, ee_body_name)
        ee_x_w = ee_R_w[:, 0]
        ee_z_w = ee_R_w[:, 2]
        kp0_w = ee_pos_w
        kp1_w = ee_pos_w + ee_x_w * kp_dx
        kp2_w = ee_pos_w + ee_z_w * kp_dz
        kp0_lb = world_to_level_base(base_pos_w, base_quat, kp0_w)
        kp1_lb = world_to_level_base(base_pos_w, base_quat, kp1_w)
        kp2_lb = world_to_level_base(base_pos_w, base_quat, kp2_w)
        ee_cur = np.concatenate([kp0_lb, kp1_lb, kp2_lb]).astype(np.float32)  # (9,)

        # joint_pos (18): legs12 + arm6
        jpos_18 = np.array([d.qpos[qpos_idx[n]] for n in obs_joint_pos_names_18], dtype=np.float32)

        # joint_vel (22): legs12 + wheels4 + arm6
        jvel_22 = np.array([d.qvel[qvel_idx[n]] for n in obs_joint_vel_names_22], dtype=np.float32)

        return base_ang_vel_b, grav_b, cmd, ee_kp_cmd.copy(), ee_cur, jpos_18, jvel_22

    # init history
    step_vecs = get_obs_step_vectors()
    for _ in range(H):
        base_ang_hist.append(step_vecs[0].copy())
        grav_hist.append(step_vecs[1].copy())
        cmd_hist.append(step_vecs[2].copy())
        ee_cmd_hist.append(step_vecs[3].copy())
        ee_cur_hist.append(step_vecs[4].copy())
        jpos_hist.append(step_vecs[5].copy())
        jvel_hist.append(step_vecs[6].copy())
        act_hist.append(last_actions_22.copy())

    def build_obs():
        # Expected term shapes with history=5:
        # base_ang_vel: 3*5=15
        # projected_gravity: 3*5=15
        # velocity_commands: 3*5=15
        # ee_kp_commands: 9*5=45
        # ee_current_kp: 9*5=45
        # joint_pos: 18*5=90
        # joint_vel: 22*5=110
        # actions: 22*5=110
        obs = np.concatenate([
            np.array(base_ang_hist, dtype=np.float32).reshape(-1),
            np.array(grav_hist, dtype=np.float32).reshape(-1),
            np.array(cmd_hist, dtype=np.float32).reshape(-1),
            np.array(ee_cmd_hist, dtype=np.float32).reshape(-1),
            np.array(ee_cur_hist, dtype=np.float32).reshape(-1),
            np.array(jpos_hist, dtype=np.float32).reshape(-1),
            np.array(jvel_hist, dtype=np.float32).reshape(-1),
            np.array(act_hist, dtype=np.float32).reshape(-1),
        ]).astype(np.float32)

        if obs.shape[0] != num_obs:
            raise RuntimeError(f"[ERROR] Built obs dim={obs.shape[0]} but expected {num_obs}")
        return obs

    # -----------------------------
    # Simulation loop
    # -----------------------------
    counter = 0
    sim_start_wall = time.time()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = d.qpos[:3]

        while viewer.is_running() and (time.time() - sim_start_wall < simulation_duration):
            step_start = time.time()

            # ---- Control-rate update ----
            if counter % control_decimation == 0:
                sim_t = float(d.time)
                if sim_t - last_kp_resample_t >= ee_resample_time:
                    resample_ee_kp_command()
                    last_kp_resample_t = sim_t

                step_vecs = get_obs_step_vectors()
                base_ang_hist.append(step_vecs[0].copy())
                grav_hist.append(step_vecs[1].copy())
                cmd_hist.append(step_vecs[2].copy())
                ee_cmd_hist.append(step_vecs[3].copy())
                ee_cur_hist.append(step_vecs[4].copy())
                jpos_hist.append(step_vecs[5].copy())
                jvel_hist.append(step_vecs[6].copy())
                act_hist.append(last_actions_22.copy())

                if USE_POLICY:
                    obs = build_obs()
                    actions_22 = sess.run([output_name], {input_name: obs[None, :]})[0][0].astype(np.float32)
                else:
                    actions_22[:] = 0.0

                # action split: legs12 + arm6 + wheel4
                leg_act = actions_22[0:12]
                arm_act = actions_22[12:18]
                wheel_act = actions_22[18:22]

                # targets
                leg_target = leg_default + leg_action_scale * leg_act

                desired_arm_target = arm_default + arm_action_scale * arm_act
                delta = np.clip(desired_arm_target - arm_target, -arm_max_delta, arm_max_delta)
                arm_target = arm_target + delta

                wheel_v_des = wheel_action_scale * wheel_act

                last_actions_22 = actions_22.copy()

            # ---- Low-level control every sim step ----
            # legs: PD position -> torque
            leg_q = np.array([d.qpos[qpos_idx[n]] for n in leg_names_12], dtype=np.float32)
            leg_dq = np.array([d.qvel[qvel_idx[n]] for n in leg_names_12], dtype=np.float32)
            leg_tau = leg_kps * (leg_target - leg_q) - leg_kds * leg_dq
            leg_tau = np.clip(leg_tau, -300.0, 300.0)

            for i, jn in enumerate(leg_names_12):
                act_name = leg_joint_to_act[jn]
                d.ctrl[act_id[act_name]] = float(leg_tau[i])

            # arm: PD position -> torque with limits
            arm_q = np.array([d.qpos[qpos_idx[n]] for n in arm_names_6], dtype=np.float32)
            arm_dq = np.array([d.qvel[qvel_idx[n]] for n in arm_names_6], dtype=np.float32)
            arm_tau = arm_kps * (arm_target - arm_q) - arm_kds * arm_dq
            arm_tau = np.clip(arm_tau, -arm_torque_limits, arm_torque_limits)

            for i, jn in enumerate(arm_names_6):
                act_name = arm_joint_to_act[jn]
                d.ctrl[act_id[act_name]] = float(arm_tau[i])

            # wheels: velocity servo -> torque (Isaac style often uses kd only)
            wheel_dq = np.array([d.qvel[qvel_idx[rln]] for rln in wheel_names_4_rl], dtype=np.float32)
            wheel_tau = wheel_kps * (wheel_v_des - wheel_dq) + wheel_kds * (wheel_v_des - wheel_dq)
            wheel_tau = np.clip(wheel_tau, -20.0, 20.0)

            for i, mj_w in enumerate(wheel_names_4_mj):
                act_name = wheel_joint_to_act[mj_w]
                d.ctrl[act_id[act_name]] = float(wheel_tau[i])

            # gripper hold
            d.ctrl[act_id[gripper_act_name]] = float(np.clip(gripper_hold_torque, -30.0, 30.0))

            # ---- Step simulation ----
            mujoco.mj_step(m, d)
            viewer.cam.lookat[:] = d.qpos[:3]

            # ---- Logging ----
            if counter % debug_every == 0:
                h = float(d.qpos[2])
                max_ctrl = float(np.max(np.abs(d.ctrl[:]))) if m.nu > 0 else 0.0
                print(
                    f"[{counter:6d}] t={d.time:7.3f}s h={h:.3f} | "
                    f"act_leg[{actions_22[:12].min():+.3f},{actions_22[:12].max():+.3f}] "
                    f"act_arm[{actions_22[12:18].min():+.3f},{actions_22[12:18].max():+.3f}] "
                    f"act_whl[{actions_22[18:22].min():+.3f},{actions_22[18:22].max():+.3f}] "
                    f"| max_ctrl={max_ctrl:.2f}"
                )

            counter += 1
            viewer.sync()

            # realtime pacing
            dt_left = m.opt.timestep - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)