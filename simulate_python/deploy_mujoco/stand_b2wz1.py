import time
import numpy as np
import mujoco
import mujoco.viewer


XML_PATH = "../unitree_robots/b2wz1/ground.xml"

# Default joint positions (23)
default_joint_pos = np.array([
    # FL
    0.1, 0.8, -1.5, 0.0,
    # FR
   -0.1, 0.8, -1.5, 0.0,
    # RL
    0.1, 1.0, -1.5, 0.0,
    # RR
   -0.1, 1.0, -1.5, 0.0,
    # arm joint1..6
    0.0, 1.48, -1.0, -0.54, 0.0, 0.0,
    # gripper
    0.0
], dtype=np.float32)

# Gains
LEG_KP = 320.0
LEG_KD = 10.0

ARM_KP = 50.0
ARM_KD = 4.0
ARM_TAU_LIMIT = 30.0

# Joint ordering (qpos after free joint)
MJ_QPOS_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    "jointGripper"
]

mj_index = {name: i for i, name in enumerate(MJ_QPOS_ORDER)}

leg_indices = np.array([
    mj_index["FL_hip_joint"], mj_index["FL_thigh_joint"], mj_index["FL_calf_joint"],
    mj_index["FR_hip_joint"], mj_index["FR_thigh_joint"], mj_index["FR_calf_joint"],
    mj_index["RL_hip_joint"], mj_index["RL_thigh_joint"], mj_index["RL_calf_joint"],
    mj_index["RR_hip_joint"], mj_index["RR_thigh_joint"], mj_index["RR_calf_joint"],
], dtype=np.int32)

wheel_indices = np.array([
    mj_index["FR_wheel_joint"],
    mj_index["FL_wheel_joint"],
    mj_index["RR_wheel_joint"],
    mj_index["RL_wheel_joint"],
], dtype=np.int32)

arm_indices = np.array([
    mj_index["joint1"], mj_index["joint2"], mj_index["joint3"],
    mj_index["joint4"], mj_index["joint5"], mj_index["joint6"],
], dtype=np.int32)

gripper_index = mj_index["jointGripper"]


# Main
m = mujoco.MjModel.from_xml_path(XML_PATH)
d = mujoco.MjData(m)

m.opt.timestep = 0.005

# Initialize base
d.qpos[0:3] = [0.0, 0.0, 0.6]
d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
d.qpos[7:7+23] = default_joint_pos
d.qvel[:] = 0.0

mujoco.mj_forward(m, d)

# Actuator indices
def act_id(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

act_leg = np.array([
    act_id("FR_hip"), act_id("FR_thigh"), act_id("FR_calf"),
    act_id("FL_hip"), act_id("FL_thigh"), act_id("FL_calf"),
    act_id("RR_hip"), act_id("RR_thigh"), act_id("RR_calf"),
    act_id("RL_hip"), act_id("RL_thigh"), act_id("RL_calf"),
], dtype=np.int32)

act_wheel = np.array([
    act_id("FR_wheel"),
    act_id("FL_wheel"),
    act_id("RR_wheel"),
    act_id("RL_wheel"),
], dtype=np.int32)

act_arm = np.array([
    act_id("motor1"), act_id("motor2"), act_id("motor3"),
    act_id("motor4"), act_id("motor5"), act_id("motor6"),
], dtype=np.int32)

act_gripper = act_id("motorGripper")


# Simulation loop
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 3.5
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 135

    while viewer.is_running():
        step_start = time.time()

        qpos_j = d.qpos[7:7+23].copy()
        qvel_j = d.qvel[6:6+23].copy()

        # Leg PD torque
        leg_pos = qpos_j[leg_indices]
        leg_vel = qvel_j[leg_indices]
        leg_tgt = default_joint_pos[leg_indices]

        leg_tau = LEG_KP * (leg_tgt - leg_pos) - LEG_KD * leg_vel

        # actuator order is FR,FL,RR,RL
        leg_tau_act = np.array([
            leg_tau[3], leg_tau[4], leg_tau[5],   # FR
            leg_tau[0], leg_tau[1], leg_tau[2],   # FL
            leg_tau[9], leg_tau[10], leg_tau[11], # RR
            leg_tau[6], leg_tau[7], leg_tau[8],   # RL
        ], dtype=np.float32)

        # Arm PD torque
        arm_pos = qpos_j[arm_indices]
        arm_vel = qvel_j[arm_indices]
        arm_tgt = default_joint_pos[arm_indices]

        arm_tau = ARM_KP * (arm_tgt - arm_pos) - ARM_KD * arm_vel
        arm_tau = np.clip(arm_tau, -ARM_TAU_LIMIT, ARM_TAU_LIMIT)

        # Apply controls
        d.ctrl[act_leg] = leg_tau_act
        d.ctrl[act_arm] = arm_tau
        d.ctrl[act_wheel] = 0.0  # zero velocity
        d.ctrl[act_gripper] = 0.0

        mujoco.mj_step(m, d)

        viewer.sync()

        # real-time pacing
        dt = m.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)