import time
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import onnxruntime as ort
import os
import sys
from collections import deque


# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Get config data directory
CONFIG_DATA_DIR = os.path.join(project_root, 'simulate_python')

# Flag: whether to use policy
USE_POLICY = True

# Get quaternion rotation function
from utilities.math import quat_rotate_inverse_numpy


if __name__ == "__main__":

    # Get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file

    # Load configs
    config_path = f"{CONFIG_DATA_DIR}/configs/{config_file}"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        # In mujoco order
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        num_actions = config["num_actions"]
        action_scale = np.array(config["action_scale"], dtype=np.float32)

        num_obs = config["num_obs"]
        num_commands = config["num_commands"]
        command = config["command"]

        # If use history
        history_length = config.get("num_history", 5)

    print(f"{'='*70}")
    print(f"B2 Flat Locomotion - ONNX Policy")
    print(f"{'='*70}")
    
    # Resolve paths
    if not os.path.isabs(policy_path):
        policy_path = os.path.join(project_root, policy_path)
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(project_root, xml_path)
    
    print(f"Policy: {policy_path}")
    print(f"XML:    {xml_path}")
    print(f"Control freq: {1.0/(simulation_dt * control_decimation):.1f} Hz")
    print(f"Action scale: {action_scale}")
    print(f"{'='*70}\n")

    # Load ONNX policy (normalization is embedded)
    sess = ort.InferenceSession(policy_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    print(f"  ONNX policy loaded")
    print(f"  Input:  {input_name} {sess.get_inputs()[0].shape}")
    print(f"  Output: {output_name} {sess.get_outputs()[0].shape}\n")

    # Define variables
    actions = np.zeros(num_actions, dtype=np.float32)
    commands = np.array(command, dtype=np.float32)

    # Raw sensor values
    ang_vel_b = np.zeros(3, dtype=np.float32)
    quat = np.zeros(4, dtype=np.float32) 
    gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_b = np.zeros(3, dtype=np.float32)
    joint_pos = np.zeros(num_actions, dtype=np.float32)
    joint_vel = np.zeros(num_actions, dtype=np.float32)

    # History buffers (only used if history_length > 1)
    if history_length > 1:
        ang_vel_hist   = deque(maxlen=history_length)  # (3,)
        gravity_hist   = deque(maxlen=history_length)  # (3,)
        commands_hist  = deque(maxlen=history_length)  # (3,)
        joint_pos_hist = deque(maxlen=history_length)  # (12,)
        joint_vel_hist = deque(maxlen=history_length)  # (12,)
        actions_hist   = deque(maxlen=history_length)  # (12,)

    # Other variables
    default_joint_pos = default_angles.copy()
    processed_actions = default_joint_pos.copy()

    # Set joint mapping
    mujoco_joint_names = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                          'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                          'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
                          'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    
    mujoco_ctrl_joint_names = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                               'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                               'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
                               'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
    
    policy_joint_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint',
                          'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint',
                          'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint',
                          'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
    
    # Get permutation indices
    mujoco_to_policy_indices = [mujoco_joint_names.index(name) for name in policy_joint_names]
    mujoco_to_ctrl_indices = [mujoco_joint_names.index(name) for name in mujoco_ctrl_joint_names]
    policy_to_mujoco_indices = [policy_joint_names.index(name) for name in mujoco_joint_names]

    print(f"Joint mapping:")
    print(f"  MuJoCo order: {mujoco_joint_names[:3]}...")
    print(f"  Ctrl order:   {mujoco_ctrl_joint_names[:3]}...")
    print(f"  Policy order: {policy_joint_names[:3]}...")

    counter = 0
    target_joint_pos = default_angles.copy()

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Initialize robot pose
    d.qpos[0:3] = [0.0, 0.0, 0.5]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
    d.qpos[7:] = default_angles
    d.qvel[:3] = np.random.uniform(-0.2, 0.2, size=3)  # Small initial linear velocity
    d.qvel[3:6] = np.random.uniform(-0.2, 0.2, size=3)  # Small initial angular velocity

    print(f"  Initialized at height {d.qpos[2]:.3f}m")
    print(f"  Default angles (MuJoCo order): {default_angles}")

    mujoco.mj_forward(m, d)

    print(f"{'='*70}")
    print(f"Starting simulation...")
    print(f"{'='*70}\n")

    # Build initial history
    sim_root_quat      = d.qpos[3:7].copy()
    sim_root_ang_vel_b = d.qvel[3:6].copy()
    sim_joint_pos      = d.qpos[7:7+num_actions].copy()[mujoco_to_policy_indices]
    sim_joint_vel      = d.qvel[6:6+num_actions].copy()[mujoco_to_policy_indices]

    quat[:]      = sim_root_quat
    ang_vel_b[:] = sim_root_ang_vel_b
    gravity_b    = quat_rotate_inverse_numpy(quat, gravity_w)

    joint_pos[:] = sim_joint_pos
    joint_vel[:] = sim_joint_vel

    base_ang = ang_vel_b.copy()
    base_grav = gravity_b.copy()
    cmd_vec = commands.copy()
    jpos_rel = (joint_pos - default_joint_pos[mujoco_to_policy_indices]).copy()
    jvel_rel = joint_vel.copy()
    last_act = np.zeros_like(actions)

    if history_length > 1:
        for _ in range(history_length):
            ang_vel_hist.append(base_ang.copy())
            gravity_hist.append(base_grav.copy())
            commands_hist.append(cmd_vec.copy())
            joint_pos_hist.append(jpos_rel.copy())
            joint_vel_hist.append(jvel_rel.copy())
            actions_hist.append(last_act.copy())

    # Open viewer and run simulation
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = d.qpos[:3]

        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # 1) PD control using current state
            pos_for_pd = d.qpos[7:7+num_actions].copy()
            vel_for_pd = d.qvel[6:6+num_actions].copy()

            torques = kps * (target_joint_pos - pos_for_pd) - kds * vel_for_pd
            torques = np.clip(
                torques,
                [-200, -200, -320, -200, -200, -320,
                 -200, -200, -320, -200, -200, -320],
                [ 200,  200,  320,  200,  200,  320,
                  200,  200,  320,  200,  200,  320],
            )
            d.ctrl[:] = torques[mujoco_to_ctrl_indices]

            # 2) advance simulation
            mujoco.mj_step(m, d)
            viewer.cam.lookat[:] = d.qpos[:3]

            # 3) policy update at control frequency
            if USE_POLICY and counter % control_decimation == 0:            
                sim_root_quat      = d.qpos[3:7].copy()
                sim_root_ang_vel_b = d.qvel[3:6].copy()
                sim_joint_pos      = d.qpos[7:7+num_actions].copy()[mujoco_to_policy_indices]
                sim_joint_vel      = d.qvel[6:6+num_actions].copy()[mujoco_to_policy_indices]

                quat[:]      = sim_root_quat
                ang_vel_b[:] = sim_root_ang_vel_b
                gravity_b    = quat_rotate_inverse_numpy(quat, gravity_w)

                joint_pos[:] = sim_joint_pos
                joint_vel[:] = sim_joint_vel

                curr_ang_vel   = ang_vel_b.copy()
                curr_gravity   = gravity_b.copy()
                curr_commands  = commands.copy()
                curr_joint_pos = (joint_pos - default_joint_pos[mujoco_to_policy_indices]).copy()
                curr_joint_vel = joint_vel.copy()
                curr_actions   = actions.copy()

                ang_vel_hist.append(curr_ang_vel.copy())
                gravity_hist.append(curr_gravity.copy())
                commands_hist.append(curr_commands.copy())
                joint_pos_hist.append(curr_joint_pos.copy())
                joint_vel_hist.append(curr_joint_vel.copy())
                actions_hist.append(curr_actions.copy())

                ang_arr  = np.array(ang_vel_hist)
                grav_arr = np.array(gravity_hist)
                cmd_arr  = np.array(commands_hist)
                jpos_arr = np.array(joint_pos_hist)
                jvel_arr = np.array(joint_vel_hist)
                act_arr  = np.array(actions_hist)

                obs = np.concatenate(
                    [
                        ang_arr.reshape(-1),
                        grav_arr.reshape(-1),
                        cmd_arr.reshape(-1),
                        jpos_arr.reshape(-1),
                        jvel_arr.reshape(-1),
                        act_arr.reshape(-1),
                    ],
                    dtype=np.float32,
                )

                actions = sess.run([output_name], {input_name: obs[None, :]})[0][0]

                processed_actions = actions[policy_to_mujoco_indices] * action_scale + default_joint_pos
                
                target_joint_pos  = processed_actions.copy()

            if counter % 100 == 0:
                joint_errors = target_joint_pos - d.qpos[7:7+num_actions]
                print(
                    f"[{counter:4d}] "
                    f"h={d.qpos[2]:.3f}m | "
                    f"action=[{actions.min():6.2f}, {actions.max():6.2f}] | "
                    f"max_err={np.max(np.abs(joint_errors)):.4f}rad | "
                    f"max_torque={np.max(np.abs(d.ctrl[:])):.1f}Nm"
                )
                # print(obs)

            counter += 1
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)