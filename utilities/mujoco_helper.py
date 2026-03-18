import mujoco
import numpy as np


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


def get_sensor_slice(model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid < 0:
        raise ValueError(f"Sensor not found: {sensor_name}")
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return data.sensordata[adr: adr + dim].copy()


def make_arrow_mat(direction: np.ndarray, up_hint: np.ndarray = None) -> np.ndarray:
    """
    Build a rotation matrix whose local +Z axis aligns with `direction`.
    This is suitable for mjGEOM_ARROW / mjGEOM_CAPSULE if we place the geom
    at the midpoint and stretch it along local z.
    """
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)

    z_axis = direction / norm

    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Avoid degeneracy when z_axis is close to up_hint
    if abs(np.dot(z_axis, up_hint)) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(up_hint, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis /= x_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-8)

    mat = np.column_stack([x_axis, y_axis, z_axis])
    return mat.astype(np.float64)