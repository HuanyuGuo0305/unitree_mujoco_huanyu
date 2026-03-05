import mujoco


"""
MuJoCo joint/actuator address helpers
"""
def joint_id(m: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found in model: '{name}'")
    return int(jid)

def act_id(m: mujoco.MjModel, name: str) -> int:
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator not found in model: '{name}'")
    return int(aid)

def qpos_adr(m: mujoco.MjModel, joint_name: str) -> int:
    jid = joint_id(m, joint_name)
    return int(m.jnt_qposadr[jid])

def qvel_adr(m: mujoco.MjModel, joint_name: str) -> int:
    jid = joint_id(m, joint_name)
    return int(m.jnt_dofadr[jid])