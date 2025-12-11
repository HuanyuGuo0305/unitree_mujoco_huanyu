import time
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import onnxruntime as ort
import os
import sys
from collections import deque


# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

# Get config data directory
CONFIG_DATA_DIR = os.path.join(project_root, "simulate_python")

# Flag: whether to use policy
USE_POLICY = True

# Get quaternion rotation function
from utilities.math import quat_rotate_inverse_numpy


if __name__ == "__main__":
    
    # Get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name")
    args = parser.parse_args()
    config_file = args.config_file

    # 1) Load configs
    config_path = f"{CONFIG_DATA_DIR}/configs/{config_file}"
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        # In MuJoCo order
        default_joint_pos = np.array(config["default_joint_pos"], dtype=np.float32)

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        num_actions = config["num_actions"]
        leg_action_scale = float(config["leg_action_scale"])
        wheel_action_scale = float(config["wheel_action_scale"])

        num_obs = config["num_obs"]
        num_commands = config["num_commands"]
        command = np.array(config["command"], dtype=np.float32)

        # If use history
        history_length = config.get("num_history", 5)
    
    print(f"{'='*70}")
    print(f"B2W Locomotion - ONNX Policy (MuJoCo sim2sim)")
    print(f"{'='*70}")

    # Resolve paths
    if not os.path.isabs(policy_path):
        policy_path = os.path.join(project_root, policy_path)
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(project_root, xml_path)
    
    print(f"Policy: {policy_path}")
    print(f"XML:    {xml_path}")
    print(f"Control freq: {1.0/(simulation_dt * control_decimation):.1f} Hz")
    print(f"Leg action scale: {leg_action_scale}")
    print(f"Wheel action scale: {wheel_action_scale}")
    print(f"{'='*70}")

    # 2) Load ONNX policy
    sess = ort.InferenceSession(policy_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    print(f"  ONNX policy loaded")
    print(f"  Input:  {input_name} {sess.get_inputs()[0].shape}")
    print(f"  Output: {output_name} {sess.get_outputs()[0].shape}\n")

    # 3) Joint mapping
    mujoco_joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint"
    ]
    
    mujoco_ctrl_joint_names = [
        # legs (torque control)
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        # wheels (velocity control)
        "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint"
    ]

    policy_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"
    ]
    
    # Leg and wheel joint names in policy order
    leg_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]

    wheel_joint_names = [
        "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"
    ]

    num_leg_joints = len(leg_joint_names)  # 12
    num_wheel_joints = len(wheel_joint_names)  # 4
    assert num_leg_joints + num_wheel_joints == num_actions

    mujoco_to_policy_indices = [mujoco_joint_names.index(name) for name in policy_joint_names]
    policy_to_mujoco_indices = [policy_joint_names.index(name) for name in mujoco_joint_names]
    mujoco_to_ctrl_indices = [mujoco_joint_names.index(name) for name in mujoco_ctrl_joint_names]

    leg_mujoco_indices = [mujoco_joint_names.index(name) for name in leg_joint_names]
    wheel_mujoco_indices = [mujoco_joint_names.index(name) for name in wheel_joint_names]

    leg_policy_indices = [policy_joint_names.index(name) for name in leg_joint_names]
    wheel_policy_indices = [policy_joint_names.index(name) for name in wheel_joint_names]

    # Default joint pos in policy order
    default_joint_pos_policy = default_joint_pos[mujoco_to_policy_indices]
    default_leg_pos_policy = default_joint_pos_policy[leg_policy_indices]

    # Control range
    ctrl_lower = np.array(
        [
            # leg torque limits
            -200.0, -200.0, -320.0, 
            -200.0, -200.0, -320.0,
            -200.0, -200.0, -320.0,
            -200.0, -200.0, -320.0,
            # wheel velocity limits
            -50.0, -50.0, -50.0, -50.0
        ],
        dtype=np.float32,
    )

    ctrl_upper = np.array(
        [
            # leg torque limits
            200.0, 200.0, 320.0, 
            200.0, 200.0, 320.0,
            200.0, 200.0, 320.0,
            200.0, 200.0, 320.0,
            # wheel velocity limits
            50.0, 50.0, 50.0, 50.0
        ],
        dtype=np.float32,
    )

    print("Joint mapping:")
    print(f" Mujoco order: {mujoco_joint_names}\n")
    print(f" Ctrl order:  {mujoco_ctrl_joint_names}\n")
    print(f" Policy order:{policy_joint_names}\n")

    # 4) Define variables
    actions = np.zeros(num_actions, dtype=np.float32)
    last_actions = np.zeros_like(actions)
    commands = command.copy()

    # Base sensors
    ang_vel_b = np.zeros(3, dtype=np.float32)
    quat = np.zeros(4, dtype=np.float32)
    gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_b = np.zeros(3, dtype=np.float32)

    # Joint states in policy order
    joint_pos_policy = np.zeros(num_actions, dtype=np.float32)
    joint_vel_policy = np.zeros(num_actions, dtype=np.float32)

    # Wheel velocity commands (MuJoCo order)
    wheel_vel_cmds = np.zeros(num_wheel_joints, dtype=np.float32)

    # Target joint pos for legs (MuJoCo order)
    target_joint_pos = default_joint_pos.copy()

    # History buffer
    if history_length > 1:
        ang_vel_hist  = deque(maxlen=history_length)  # (3,)
        gravity_hist  = deque(maxlen=history_length)  # (3,)
        commands_hist = deque(maxlen=history_length)  # (3,)
        jpos_hist     = deque(maxlen=history_length)  # (12,) legs only
        jvel_hist     = deque(maxlen=history_length)  # (16,) all joints
        actions_hist  = deque(maxlen=history_length)  # (16,)
    
    counter = 0

    # 5) Load model and init state
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Init robot state
    d.qpos[0:3] = [0.0, 0.0, 0.6]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:7+num_actions] = default_joint_pos
    d.qvel[:3] = np.random.uniform(-0.2, 0.2, size=3)
    d.qvel[3:6] = np.random.uniform(-0.2, 0.2, size=3)

    mujoco.mj_forward(m, d)

    print(f"Initialized at height {d.qpos[2]:.3f} m")
    print(f"Default joint pos (MuJoCo order): {default_joint_pos}\n")

    print(f"{'='*70}")
    print("Starting simulation...")
    print(f"{'='*70}\n")

    # 6) Build initial history
    sim_root_quat = d.qpos[3:7].copy()
    sim_root_ang_vel_b = d.qvel[3:6].copy()

    quat[:] = sim_root_quat
    ang_vel_b[:] = sim_root_ang_vel_b
    gravity_b[:] = quat_rotate_inverse_numpy(quat, gravity_w)

    # Joint states (MuJoCo -> Policy)
    qpos_mujoco = d.qpos[7:7+num_actions].copy()
    qvel_mujoco = d.qvel[6:6+num_actions].copy()

    joint_pos_policy[:] = qpos_mujoco[mujoco_to_policy_indices]
    joint_vel_policy[:] = qvel_mujoco[mujoco_to_policy_indices]

    leg_pos_policy = joint_pos_policy[leg_policy_indices]
    leg_pos_rel_policy = leg_pos_policy - default_leg_pos_policy  # legs only

    base_ang = ang_vel_b.copy()
    base_grav = gravity_b.copy()
    cmd_vec = commands.copy()
    jpos_rel = leg_pos_rel_policy.copy()  # (12,)
    jvel_all = joint_vel_policy.copy()  # (16,)
    last_act = np.zeros_like(actions)  # (16,)

    if history_length > 1:
        for _ in range(history_length):
            ang_vel_hist.append(base_ang.copy())
            gravity_hist.append(base_grav.copy())
            commands_hist.append(cmd_vec.copy())
            jpos_hist.append(jpos_rel.copy())
            jvel_hist.append(jvel_all.copy())
            actions_hist.append(last_act.copy())
    
    # 7) Main simulation loop
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = d.qpos[:3]

        start_time = time.time()

        while viewer.is_running() and (time.time() - start_time < simulation_duration):
            step_start = time.time()

            # 1. Low level control
            qpos_mujoco = d.qpos[7:7+num_actions].copy()
            qvel_mujoco = d.qvel[6:6+num_actions].copy()

            # Leg joints
            leg_pos = qpos_mujoco[leg_mujoco_indices]
            leg_vel = qvel_mujoco[leg_mujoco_indices]
            leg_target_pos = target_joint_pos[leg_mujoco_indices]

            # PD torque
            leg_torques = (
                kps[leg_mujoco_indices] * (leg_target_pos - leg_pos) -
                kds[leg_mujoco_indices] * leg_vel
            )

            ctrl_targets_mujoco = np.zeros(num_actions, dtype=np.float32)
            ctrl_targets_mujoco[leg_mujoco_indices] = leg_torques
            ctrl_targets_mujoco[wheel_mujoco_indices] = wheel_vel_cmds

            ctrl_targets_mujoco = np.clip(ctrl_targets_mujoco, ctrl_lower, ctrl_upper)

            d.ctrl[:] = ctrl_targets_mujoco[mujoco_to_ctrl_indices]

            # 2. Step simulation
            mujoco.mj_step(m, d)
            viewer.cam.lookat[:] = d.qpos[:3]

            # 3. Policy inference
            if USE_POLICY and (counter % control_decimation == 0):
                # Base sensors
                sim_root_quat = d.qpos[3:7].copy()
                sim_root_ang_vel_b = d.qvel[3:6].copy()

                quat[:] = sim_root_quat
                ang_vel_b[:] = sim_root_ang_vel_b
                gravity_b[:] = quat_rotate_inverse_numpy(quat, gravity_w)

                # Joint states (MuJoCo -> Policy)
                qpos_mujoco = d.qpos[7:7+num_actions].copy()
                qvel_mujoco = d.qvel[6:6+num_actions].copy()

                joint_pos_policy[:] = qpos_mujoco[mujoco_to_policy_indices]
                joint_vel_policy[:] = qvel_mujoco[mujoco_to_policy_indices]

                leg_pos_policy = joint_pos_policy[leg_policy_indices]
                leg_pos_rel_policy = leg_pos_policy - default_leg_pos_policy

                curr_ang = ang_vel_b.copy()
                curr_grav = gravity_b.copy()
                curr_cmd = commands.copy()
                curr_jpos_rel = leg_pos_rel_policy.copy()
                curr_jvel_all = joint_vel_policy.copy()
                last_act = actions.copy()

                ang_vel_hist.append(curr_ang.copy())
                gravity_hist.append(curr_grav.copy())
                commands_hist.append(curr_cmd.copy())
                jpos_hist.append(curr_jpos_rel.copy())
                jvel_hist.append(curr_jvel_all.copy())
                actions_hist.append(last_act.copy())

                ang_arr = np.array(ang_vel_hist)  # (H, 3)
                grav_arr = np.array(gravity_hist)
                cmd_arr = np.array(commands_hist)
                jpos_arr = np.array(jpos_hist)
                jvel_arr = np.array(jvel_hist)
                act_arr = np.array(actions_hist)

                obs = np.concatenate(
                    [
                        ang_arr.reshape(-1),
                        grav_arr.reshape(-1),
                        cmd_arr.reshape(-1),
                        jpos_arr.reshape(-1),
                        jvel_arr.reshape(-1),
                        act_arr.reshape(-1),
                    ],
                    dtype = np.float32,
                )

                actions = sess.run([output_name], {input_name: obs[None, :]})[0][0]

                actions_mujoco = actions[policy_to_mujoco_indices]

                # Split leg and wheel actions
                leg_actions_mujoco = actions_mujoco[leg_mujoco_indices]
                wheel_actions_mujoco = actions_mujoco[wheel_mujoco_indices]

                # Leg
                target_joint_pos[leg_mujoco_indices] = (
                    default_joint_pos[leg_mujoco_indices] + leg_action_scale * leg_actions_mujoco
                )

                # Wheel
                wheel_vel_cmds[:] = wheel_action_scale * wheel_actions_mujoco

            # 4. Logging
            if counter % 100 == 0:
                current_leg_pos = d.qpos[7:7+num_actions].copy()[leg_mujoco_indices]
                leg_err = target_joint_pos[leg_mujoco_indices] - current_leg_pos

                print(
                    f"[{counter:5d}] "
                    f"h={d.qpos[2]:.3f}m | "
                    f"leg_action=[{actions[leg_policy_indices].min():6.3f}, {actions[leg_policy_indices].max():6.3f}] | "
                    f"wheel_action=[{actions[wheel_policy_indices].min():6.3f}, {actions[wheel_policy_indices].max():6.3f}] | "
                    f"max_leg_err={np.max(np.abs(leg_err)):.4f}rad | "
                    f"max_ctrl={np.max(np.abs(d.ctrl[:])):.2f}"
                )

            counter += 1
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)