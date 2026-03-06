import mujoco


def mj_id(m: mujoco.MjModel, obj_type, name: str) -> int:
    """Return MuJoCo object id from name, or raise if not found."""
    idx = mujoco.mj_name2id(m, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"MuJoCo name not found: {name}")
    return int(idx)


def joint_qposadr(m: mujoco.MjModel, joint_name: str) -> int:
    """Return the starting index of a joint in d.qpos."""
    jid = mj_id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(m.jnt_qposadr[jid])


def joint_dofadr(m: mujoco.MjModel, joint_name: str) -> int:
    """Return the starting index of a joint in d.qvel."""
    jid = mj_id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(m.jnt_dofadr[jid])


def actuator_id(m: mujoco.MjModel, actuator_name: str) -> int:
    """Return actuator id, which is also the index in d.ctrl."""
    return mj_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)


def site_id(m: mujoco.MjModel, site_name: str) -> int:
    """Return site id for accessing d.site_xpos / d.site_xmat."""
    return mj_id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)