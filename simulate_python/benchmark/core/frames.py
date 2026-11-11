"""Frame conventions shared by every benchmark component.

Everything here mirrors the validated Sim2Sim scripts exactly:

PLB (Projected Level Base) frame
    origin      = [base_x, base_y, ee_ground_z]
    orientation = yaw-only(base_quat)

EE command / state representation
    9-D keypoint triplet (kp0, kp1, kp2) with
        kp0 = p_ee
        kp1 = p_ee + R_ee @ [dx, 0, 0]      dx = 0.30
        kp2 = p_ee + R_ee @ [0, 0, dz]      dz = 0.30

See benchmark/docs/interface_audit.md sections 4.1-4.3 for the traced evidence.
"""

from __future__ import annotations

import numpy as np

from utilities.math import (
    euler_xyz_from_quat_wxyz,
    quat_apply_inverse_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_from_rotmat_wxyz,
    quat_from_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_unique_wxyz,
)

EE_KP_DX = 0.30
EE_KP_DZ = 0.30


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8 or not np.isfinite(norm):
        raise ValueError(f"Degenerate keypoints: {label} has norm {norm}")
    return vector / norm


def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    """Yaw angle using the exact helper the deployments use."""
    q = quat_unique_wxyz(np.asarray(quat_wxyz, dtype=np.float32))
    return float(euler_xyz_from_quat_wxyz(q)[2])


def roll_pitch_yaw(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    q = quat_unique_wxyz(np.asarray(quat_wxyz, dtype=np.float32))
    roll, pitch, yaw = euler_xyz_from_quat_wxyz(q)
    return float(roll), float(pitch), float(yaw)


def rotmat_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> 3x3 rotation matrix (world <- local)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------
# PLB frame
# --------------------------------------------------------------------------
def plb_pose_from_base(
    base_pos_w: np.ndarray, base_quat_w: np.ndarray, ground_z: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Replicates ``get_projected_level_base_pose_world`` in every deployment."""
    base_pos_w = np.asarray(base_pos_w, dtype=np.float32)
    base_quat_w = quat_unique_wxyz(np.asarray(base_quat_w, dtype=np.float32))

    yaw = float(euler_xyz_from_quat_wxyz(base_quat_w)[2])
    plb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(quat_from_yaw_wxyz(yaw)))

    plb_pos_w = base_pos_w.copy()
    plb_pos_w[2] = np.float32(ground_z)
    return plb_pos_w, plb_quat_w


def plb_rotmat(base_quat_w: np.ndarray) -> np.ndarray:
    """Yaw-only rotation matrix of the PLB frame (world <- PLB)."""
    yaw = yaw_from_quat(base_quat_w)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# --------------------------------------------------------------------------
# keypoints <-> pose
# --------------------------------------------------------------------------
def pose_to_keypoints(
    position: np.ndarray,
    rotation: np.ndarray,
    dx: float = EE_KP_DX,
    dz: float = EE_KP_DZ,
) -> np.ndarray:
    """(p, R) -> 9-D keypoint command, same convention as the deployments."""
    p = np.asarray(position, dtype=np.float64).reshape(3)
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    kp0 = p
    kp1 = p + R @ np.array([dx, 0.0, 0.0])
    kp2 = p + R @ np.array([0.0, 0.0, dz])
    return np.concatenate([kp0, kp1, kp2]).astype(np.float32)


def rotation_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Verbatim port of C3's ``_rotation_from_keypoints``.

    X from kp1-kp0; Y from cross(Z, X); Z from cross(X, Y). Orthonormalised so
    that float32 rounding in the command table cannot produce a non-rotation.
    """
    kp = np.asarray(keypoints, dtype=np.float64).reshape(3, 3)
    x = _normalize(kp[1] - kp[0], "x-axis")
    z_raw = _normalize(kp[2] - kp[0], "z-axis")
    y = _normalize(np.cross(z_raw, x), "cross(z,x)")
    z = _normalize(np.cross(x, y), "orthogonalized z-axis")
    return np.column_stack((x, y, z))


def keypoints_to_pose(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """9-D keypoints -> (position, rotation matrix)."""
    kp = np.asarray(keypoints, dtype=np.float64).reshape(3, 3)
    return kp[0].copy(), rotation_from_keypoints(keypoints)


# --------------------------------------------------------------------------
# PLB <-> world for keypoints
# --------------------------------------------------------------------------
def keypoints_plb_to_world(
    keypoints_plb: np.ndarray, plb_pos_w: np.ndarray, plb_quat_w: np.ndarray
) -> np.ndarray:
    """Push a 9-D PLB keypoint vector into world coordinates."""
    kp = np.asarray(keypoints_plb, dtype=np.float32).reshape(3, 3)
    out = np.empty((3, 3), dtype=np.float32)
    for i in range(3):
        out[i] = plb_pos_w + quat_apply_wxyz(plb_quat_w, kp[i])
    return out.reshape(9)


def keypoints_world_to_plb(
    keypoints_w: np.ndarray, plb_pos_w: np.ndarray, plb_quat_w: np.ndarray
) -> np.ndarray:
    kp = np.asarray(keypoints_w, dtype=np.float32).reshape(3, 3)
    out = np.empty((3, 3), dtype=np.float32)
    for i in range(3):
        out[i] = quat_apply_inverse_wxyz(plb_quat_w, kp[i] - plb_pos_w)
    return out.reshape(9)


def world_pose_to_plb_keypoints(
    position_w: np.ndarray,
    rotation_w: np.ndarray,
    base_pos_w: np.ndarray,
    base_quat_w: np.ndarray,
    ground_z: float = 0.0,
    dx: float = EE_KP_DX,
    dz: float = EE_KP_DZ,
) -> np.ndarray:
    """World-frame EE pose -> native 9-D PLB keypoint command (used by E4)."""
    plb_pos_w, plb_quat_w = plb_pose_from_base(base_pos_w, base_quat_w, ground_z)
    R_plb = plb_rotmat(base_quat_w)
    p_plb = quat_apply_inverse_wxyz(
        plb_quat_w, np.asarray(position_w, dtype=np.float32) - plb_pos_w
    )
    R_cmd_plb = R_plb.T @ np.asarray(rotation_w, dtype=np.float64).reshape(3, 3)
    return pose_to_keypoints(p_plb, R_cmd_plb, dx=dx, dz=dz)


# --------------------------------------------------------------------------
# measured EE state
# --------------------------------------------------------------------------
def ee_keypoints_plb_from_state(
    ee_pos_w: np.ndarray,
    ee_rotmat_w: np.ndarray,
    base_pos_w: np.ndarray,
    base_quat_w: np.ndarray,
    ground_z: float = 0.0,
    dx: float = EE_KP_DX,
    dz: float = EE_KP_DZ,
) -> np.ndarray:
    """Verbatim port of ``compute_ee_current_kp_plb`` (C1 L614-630).

    Kept bit-compatible with the deployment: the offsets are applied with
    ``quat_apply_wxyz`` on the float32 PLB-relative quaternion, not with a
    float64 matrix, so adapter observations match the originals exactly.
    """
    ee_pos_w = np.asarray(ee_pos_w, dtype=np.float32)
    ee_rot_w = np.asarray(ee_rotmat_w, dtype=np.float32).reshape(3, 3)
    ee_quat_w = quat_from_rotmat_wxyz(ee_rot_w)

    plb_pos_w, plb_quat_w = plb_pose_from_base(base_pos_w, base_quat_w, ground_z)

    ee_pos_plb = quat_apply_inverse_wxyz(plb_quat_w, ee_pos_w - plb_pos_w)
    ee_quat_plb = quat_mul_wxyz(quat_conjugate_wxyz(plb_quat_w), ee_quat_w)
    ee_quat_plb = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_plb))

    off_x = np.array([dx, 0.0, 0.0], dtype=np.float32)
    off_z = np.array([0.0, 0.0, dz], dtype=np.float32)

    kp0 = ee_pos_plb
    kp1 = ee_pos_plb + quat_apply_wxyz(ee_quat_plb, off_x)
    kp2 = ee_pos_plb + quat_apply_wxyz(ee_quat_plb, off_z)
    return np.concatenate([kp0, kp1, kp2]).astype(np.float32)


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------
def geodesic_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic distance on SO(3) in degrees.

    theta = arccos(clip((trace(R_a^T R_b) - 1) / 2, -1, 1))

    This is the rotation angle of the relative rotation; it is NOT a raw
    quaternion subtraction and is invariant to the quaternion sign ambiguity.
    """
    A = np.asarray(R_a, dtype=np.float64).reshape(3, 3)
    B = np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    cos_theta = (np.trace(A.T @ B) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def keypoint_pose_error(
    keypoints_cmd: np.ndarray, keypoints_actual: np.ndarray
) -> tuple[float, float]:
    """(position error [m], orientation geodesic error [deg]) between two 9-D vectors."""
    p_c, R_c = keypoints_to_pose(keypoints_cmd)
    p_a, R_a = keypoints_to_pose(keypoints_actual)
    return float(np.linalg.norm(p_a - p_c)), geodesic_angle_deg(R_c, R_a)


def base_velocity_body_frame(
    lin_vel_w: np.ndarray, ang_vel_w: np.ndarray, base_quat_w: np.ndarray
) -> tuple[np.ndarray, float]:
    """Base-frame linear velocity (3,) and yaw rate (scalar)."""
    R = rotmat_from_quat(base_quat_w)
    v_b = R.T @ np.asarray(lin_vel_w, dtype=np.float64).reshape(3)
    w_b = R.T @ np.asarray(ang_vel_w, dtype=np.float64).reshape(3)
    return v_b, float(w_b[2])


def base_velocity_yaw_frame(
    lin_vel_w: np.ndarray, ang_vel_w: np.ndarray, base_quat_w: np.ndarray
) -> tuple[np.ndarray, float]:
    """Yaw-only (PLB) frame linear velocity (3,) and yaw rate (scalar)."""
    R = plb_rotmat(base_quat_w)
    v = R.T @ np.asarray(lin_vel_w, dtype=np.float64).reshape(3)
    w = R.T @ np.asarray(ang_vel_w, dtype=np.float64).reshape(3)
    return v, float(w[2])
