import numpy as np


def quat_rotate_inverse_numpy(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Rotate vector by inverse quaternion.

    This computes:
        v' = q^{-1} * v * q

    Args:
        quat: quaternion [w, x, y, z] (unit, wxyz)
        vec:  vector [x, y, z]

    Returns:
        Rotated vector in same frame.
    """
    quat = np.asarray(quat, dtype=np.float32).reshape(4,)
    vec  = np.asarray(vec,  dtype=np.float32).reshape(3,)

    quat = quat_unique_wxyz(quat)
    return quat_apply_inverse_wxyz(quat, vec)

def quat_unique_wxyz(q: np.ndarray) -> np.ndarray:
    """Ensure quaternion has non-negative w for uniqueness."""
    q = q.astype(np.float32)
    return (-q if q[0] < 0 else q)

def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, both wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)

def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    """Conjugate of wxyz quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float32)

def quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (both in wxyz)."""
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul_wxyz(quat_mul_wxyz(q, qv), quat_conj_wxyz(q))[1:4]

def quat_apply_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by inverse(q) (same as applying conjugate for unit q)."""
    return quat_apply_wxyz(quat_conj_wxyz(q), v)

def euler_xyz_from_quat_wxyz(q: np.ndarray):
    """Return roll,pitch,yaw from wxyz quaternion."""
    w, x, y, z = q
    t0 = 2.0*(w*x + y*z)
    t1 = 1.0 - 2.0*(x*x + y*y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0*(w*y - z*x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0*(w*z + x*y)
    t4 = 1.0 - 2.0*(y*y + z*z)
    yaw = np.arctan2(t3, t4)
    return float(roll), float(pitch), float(yaw)

def quat_from_yaw_wxyz(yaw: float) -> np.ndarray:
    """Yaw-only quaternion in wxyz."""
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)

def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return (v * 0.0).astype(np.float32)
    return (v / n).astype(np.float32)

def quat_from_rotmat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to unit quaternion (wxyz)."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    return quat_unique_wxyz(q)

def quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float, eps: float = 1e-8) -> np.ndarray:
    """Slerp between unit quaternions q0->q1."""
    q0 = q0.astype(np.float32)
    q1 = q1.astype(np.float32)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, 0.0, 1.0))

    if dot > 1.0 - 1e-6:
        out = (1.0 - t) * q0 + t * q1
        out = out / (np.linalg.norm(out) + eps)
        return out.astype(np.float32)

    theta_0 = float(np.arccos(dot))
    sin_0 = float(np.sin(theta_0))
    theta = theta_0 * float(t)

    s0 = float(np.sin(theta_0 - theta) / (sin_0 + eps))
    s1 = float(np.sin(theta) / (sin_0 + eps))
    out = s0 * q0 + s1 * q1
    out = out / (np.linalg.norm(out) + eps)
    return out.astype(np.float32)

def quat_from_keypoints_lb(kp0: np.ndarray, kp1: np.ndarray, kp2: np.ndarray, dx: float, dz: float) -> np.ndarray:
    """
    Build an orientation in LB frame from three keypoints:
      - x axis from kp0->kp1 (normalized)
      - z axis from kp0->kp2 (normalized)
      - y = z x x
    """
    x_axis = (kp1 - kp0) / max(dx, 1e-8)
    z_axis = (kp2 - kp0) / max(dz, 1e-8)
    x_axis = normalize(x_axis)
    z_axis = normalize(z_axis)

    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return quat_from_rotmat_wxyz(R)
