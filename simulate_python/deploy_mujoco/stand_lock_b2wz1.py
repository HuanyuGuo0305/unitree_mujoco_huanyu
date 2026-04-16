"""
Print ee_current_lb ONCE with all joints strictly locked at default.

Run:

    python3 deploy_mujoco/b2wz1_print_ee_current_lb_once.py configs/b2wz1_loco_manipulation.yaml
"""

import os
import sys
import numpy as np
import mujoco
import yaml

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from utilities.math import (
    quat_apply_inverse_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_from_rotmat_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_unique_wxyz,
    euler_xyz_from_quat_wxyz,
    quat_from_yaw_wxyz,
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("yaml_path", type=str)
args = parser.parse_args()

# ===== load config =====
with open(args.yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

xml_path = cfg["xml_path"]
if not os.path.isabs(xml_path):
    xml_path = os.path.abspath(os.path.join(project_root, xml_path))

root_pos = np.array(cfg["root_pos"], dtype=np.float32)
root_quat = np.array(cfg["root_quat_wxyz"], dtype=np.float32)
default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)

ee_body_name = str(cfg.get("ee_body", "gripperStator"))
ee_kp_dx = float(cfg.get("ee_kp_dx", 0.30))
ee_kp_dz = float(cfg.get("ee_kp_dz", 0.30))

# ===== load mujoco =====
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
if ee_bid < 0:
    raise ValueError(f"Body not found: {ee_body_name}")

# ===== strictly set state =====
d.qpos[:] = 0.0
d.qvel[:] = 0.0

# base
d.qpos[0:3] = root_pos
d.qpos[3:7] = root_quat

# joints (strict lock)
d.qpos[7:7 + len(default_joint_pos)] = default_joint_pos
d.qvel[:] = 0.0

mujoco.mj_forward(m, d)

# ===== compute ee_current_lb =====
def compute_ee_current_lb():
    base_pos_w = d.qpos[0:3].astype(np.float32)
    base_quat_w = quat_unique_wxyz(d.qpos[3:7].astype(np.float32))

    _, _, yaw = euler_xyz_from_quat_wxyz(base_quat_w)
    lb_quat_w = quat_from_yaw_wxyz(yaw)
    lb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(lb_quat_w))

    ee_pos_w = d.xpos[ee_bid].astype(np.float32)
    ee_rot_w = d.xmat[ee_bid].reshape(3, 3).astype(np.float32)
    ee_quat_w = quat_from_rotmat_wxyz(ee_rot_w)

    ee_pos_lb = quat_apply_inverse_wxyz(lb_quat_w, ee_pos_w - base_pos_w)
    ee_quat_lb = quat_mul_wxyz(quat_conjugate_wxyz(lb_quat_w), ee_quat_w)
    ee_quat_lb = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_lb))

    off_x = np.array([ee_kp_dx, 0.0, 0.0], dtype=np.float32)
    off_z = np.array([0.0, 0.0, ee_kp_dz], dtype=np.float32)

    kp0 = ee_pos_lb
    kp1 = ee_pos_lb + quat_apply_wxyz(ee_quat_lb, off_x)
    kp2 = ee_pos_lb + quat_apply_wxyz(ee_quat_lb, off_z)

    return np.concatenate([kp0, kp1, kp2]).astype(np.float32)

ee_current_lb = compute_ee_current_lb()

# ===== print once =====
kp0 = ee_current_lb[0:3]
kp1 = ee_current_lb[3:6]
kp2 = ee_current_lb[6:9]

print("=" * 80)
print("EE CURRENT (LEVEL-BASE FRAME)")
print("=" * 80)
print("ee_current_lb =", np.round(ee_current_lb, 6))
print("kp0 (position) =", np.round(kp0, 6))
print("kp1 (x-axis)   =", np.round(kp1, 6))
print("kp2 (z-axis)   =", np.round(kp2, 6))
print("=" * 80)