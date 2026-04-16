import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import yaml

"""
code to run:
python3 deploy_mujoco/visualize_demo_curve.py configs/b2wz1_loco_manipulation.yaml 
"""

# add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from utilities.math import (
    quat_apply_wxyz,
    quat_unique_wxyz,
    quat_normalize_wxyz,
    euler_xyz_from_quat_wxyz,
    quat_from_yaw_wxyz,
)

from utilities.mujoco_helper import make_arrow_mat

# =========================
# Visualization helpers
# =========================

def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0], dtype=np.float64),
        pos,
        np.eye(3).reshape(-1),
        rgba,
    )
    scene.ngeom += 1


def add_capsule(scene, p0, p1, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return

    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    diff = p1 - p0
    length = np.linalg.norm(diff)
    if length < 1e-6:
        return

    pos = 0.5 * (p0 + p1)
    mat = make_arrow_mat(diff)

    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, 0.5 * length, 0]),
        pos,
        mat.reshape(-1),
        rgba,
    )
    scene.ngeom += 1


# =========================
# main
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("yaml_path", type=str)
args = parser.parse_args()

with open(args.yaml_path, "r") as f:
    cfg = yaml.safe_load(f)

xml_path = cfg["xml_path"]
if not os.path.isabs(xml_path):
    xml_path = os.path.abspath(os.path.join(project_root, xml_path))

traj_path = "/home/huanyuguo/Workspace_huanyu/unitree_mujoco_huanyu/utilities/data/demo_sweep_curve.npy"

root_pos = np.array(cfg["root_pos"], dtype=np.float32)
root_quat = np.array(cfg["root_quat_wxyz"], dtype=np.float32)
default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)

kp_dx = float(cfg.get("ee_kp_dx", 0.30))
kp_dz = float(cfg.get("ee_kp_dz", 0.30))

# load traj
traj = np.load(traj_path).astype(np.float32)
N = traj.shape[0]

print(f"Loaded traj: {traj.shape}")

# load mujoco
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

# init state
d.qpos[:] = 0.0
d.qvel[:] = 0.0

d.qpos[0:3] = root_pos
d.qpos[3:7] = root_quat
d.qpos[7:7 + len(default_joint_pos)] = default_joint_pos

mujoco.mj_forward(m, d)

# =========================
# helper: LB -> world
# =========================

def lb_to_world(kp_lb):
    base_pos = d.qpos[0:3]
    base_quat = quat_unique_wxyz(d.qpos[3:7])

    _, _, yaw = euler_xyz_from_quat_wxyz(base_quat)
    lb_quat = quat_from_yaw_wxyz(yaw)
    lb_quat = quat_unique_wxyz(quat_normalize_wxyz(lb_quat))

    kp0 = kp_lb[0:3]
    kp1 = kp_lb[3:6]
    kp2 = kp_lb[6:9]

    kp0_w = base_pos + quat_apply_wxyz(lb_quat, kp0)
    kp1_w = base_pos + quat_apply_wxyz(lb_quat, kp1)
    kp2_w = base_pos + quat_apply_wxyz(lb_quat, kp2)

    return kp0_w, kp1_w, kp2_w


# =========================
# viewer
# =========================

idx = 0
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.cam.lookat[:] = d.qpos[:3]

    while viewer.is_running():
        step_start = time.time()

        viewer.user_scn.ngeom = 0

        # ===== draw all traj points =====
        for i in range(N):
            kp0_w, _, _ = lb_to_world(traj[i])
            add_sphere(
                viewer.user_scn,
                kp0_w,
                radius=0.01,
                rgba=np.array([1, 0, 0, 0.5]),  # red
            )

        # ===== draw current point =====
        kp0_w, kp1_w, kp2_w = lb_to_world(traj[idx])

        add_sphere(
            viewer.user_scn,
            kp0_w,
            radius=0.02,
            rgba=np.array([0, 0, 1, 1]),  # blue
        )

        # x-axis
        add_capsule(
            viewer.user_scn,
            kp0_w,
            kp1_w,
            0.005,
            np.array([0, 1, 0, 1]),  # green
        )

        # z-axis
        add_capsule(
            viewer.user_scn,
            kp0_w,
            kp2_w,
            0.005,
            np.array([0, 1, 1, 1]),  # cyan
        )

        # ===== step index =====
        idx = (idx + 1) % N

        viewer.sync()

        time.sleep(0.05)