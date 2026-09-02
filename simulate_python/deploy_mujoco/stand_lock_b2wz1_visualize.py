"""
Show a strictly locked B2WZ1 pose in a MuJoCo GUI and print the current
end-effector pose in the PLB frame:

    kp0_xyz, yaw, pitch, roll

PLB definition:
    origin      = [base_x, base_y, ee_ground_z]
    orientation = yaw-only(base_quat)

Run:

    python3 deploy_mujoco/stand_lock_b2wz1_gui_plb_pose.py \
        configs/b2wz1_loco_manipulation.yaml

Optionally override arm joint2/3/4 for pose tuning:

    python3 deploy_mujoco/stand_lock_b2wz1_gui_plb_pose.py \
        configs/b2wz1_loco_manipulation.yaml \
        --joint2 1.48 \
        --joint3 -1.00 \
        --joint4 -0.54

Behavior:
  - Opens a passive MuJoCo viewer.
  - Strictly locks the floating base and all joints.
  - Uses YAML default_joint_pos unless joint2/3/4 are overridden.
  - Prints PLB kp0_xyz and EE yaw/pitch/roll once.
  - Also prints the complete 9-D keypoint representation.
  - Keeps the GUI open until the viewer window is closed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)
sys.path.insert(0, project_root)

from utilities.math import (  # noqa: E402
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


MUJOCO_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    "jointGripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a MuJoCo GUI with B2WZ1 locked at a selected arm pose "
            "and print the EE pose in PLB."
        )
    )
    parser.add_argument("yaml_path", type=str)
    parser.add_argument(
        "--refresh-hz",
        type=float,
        default=60.0,
        help="GUI refresh frequency.",
    )
    parser.add_argument(
        "--joint2",
        type=float,
        default=None,
        help="Optional joint2 override in radians.",
    )
    parser.add_argument(
        "--joint3",
        type=float,
        default=None,
        help="Optional joint3 override in radians.",
    )
    parser.add_argument(
        "--joint4",
        type=float,
        default=None,
        help="Optional joint4 override in radians.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    yaml_path = os.path.abspath(args.yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    xml_path = str(cfg["xml_path"])
    if not os.path.isabs(xml_path):
        xml_path = os.path.abspath(
            os.path.join(project_root, xml_path)
        )

    root_pos = np.asarray(cfg["root_pos"], dtype=np.float64)
    root_quat = np.asarray(
        cfg["root_quat_wxyz"], dtype=np.float64
    )
    default_joint_pos = np.asarray(
        cfg["default_joint_pos"], dtype=np.float64
    ).copy()

    ee_body_name = str(cfg.get("ee_body", "gripperStator"))
    ee_kp_dx = float(cfg.get("ee_kp_dx", 0.30))
    ee_kp_dz = float(cfg.get("ee_kp_dz", 0.30))
    ee_ground_z = float(
        cfg.get("ee_ground_z", cfg.get("ground_z", 0.0))
    )

    refresh_hz = float(args.refresh_hz)
    if refresh_hz <= 0.0:
        raise ValueError(
            f"--refresh-hz must be positive, got {refresh_hz}."
        )
    refresh_dt = 1.0 / refresh_hz

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Base body is used to place the URDF camera frames in world coordinates.
    base_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base_link",
    )
    if base_body_id < 0:
        raise ValueError("Body not found in compiled MuJoCo model: base_link")

    # Exact fixed transforms from the training B2WZ1 URDF:
    #   T_Base<-OC and T_Base<-DC.
    T_BASE_OC_POS = np.array([0.3993, 0.0, -0.01576], dtype=np.float64)
    T_BASE_DC_POS = np.array([0.42161, 0.025, 0.061851], dtype=np.float64)

    def rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr,  cr],
        ], dtype=np.float64)
        ry = np.array([
            [ cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ], dtype=np.float64)
        rz = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # URDF fixed-joint rotation convention.
        return rz @ ry @ rx

    R_BASE_OC = rpy_to_rotmat(-1.5708, 0.0, -1.5708)
    R_BASE_DC = rpy_to_rotmat(-2.3562, 0.0, -1.5708)

    CAMERA_AXIS_LENGTH = 0.16
    CAMERA_AXIS_WIDTH = 0.008

    ee_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        ee_body_name,
    )
    if ee_body_id < 0:
        raise ValueError(f"Body not found: {ee_body_name}")

    expected_joint_qpos = model.nq - 7
    if len(default_joint_pos) != expected_joint_qpos:
        raise ValueError(
            "default_joint_pos length does not match model qpos: "
            f"got {len(default_joint_pos)}, expected {expected_joint_qpos}."
        )
    if len(default_joint_pos) != len(MUJOCO_JOINT_NAMES):
        raise ValueError(
            "This script expects the known B2WZ1 joint ordering with "
            f"{len(MUJOCO_JOINT_NAMES)} joints, but YAML contains "
            f"{len(default_joint_pos)} values."
        )

    # Optional arm pose overrides.
    override_values = {
        "joint2": args.joint2,
        "joint3": args.joint3,
        "joint4": args.joint4,
    }
    for joint_name, value in override_values.items():
        if value is not None:
            joint_index = MUJOCO_JOINT_NAMES.index(joint_name)
            default_joint_pos[joint_index] = float(value)

    locked_qpos = np.zeros(model.nq, dtype=np.float64)
    locked_qpos[0:3] = root_pos
    locked_qpos[3:7] = root_quat
    locked_qpos[7:] = default_joint_pos

    locked_qvel = np.zeros(model.nv, dtype=np.float64)

    def write_locked_state() -> None:
        data.qpos[:] = locked_qpos
        data.qvel[:] = locked_qvel

        if model.na > 0:
            data.act[:] = 0.0
        if model.nu > 0:
            data.ctrl[:] = 0.0

        mujoco.mj_forward(model, data)

    write_locked_state()

    def camera_frame_world(
        p_base_cam: np.ndarray,
        r_base_cam: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return camera origin and orientation in the MuJoCo world frame."""
        p_world_base = data.xpos[base_body_id].copy()
        r_world_base = data.xmat[base_body_id].reshape(3, 3).copy()

        p_world_cam = p_world_base + r_world_base @ p_base_cam
        r_world_cam = r_world_base @ r_base_cam
        return p_world_cam, r_world_cam

    def add_arrow(
        scene,
        index: int,
        p0: np.ndarray,
        p1: np.ndarray,
        rgba: np.ndarray,
    ) -> int:
        if index >= scene.maxgeom:
            raise RuntimeError(
                f"MuJoCo user scene has only {scene.maxgeom} geom slots."
            )

        geom = scene.geoms[index]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([CAMERA_AXIS_WIDTH, CAMERA_AXIS_WIDTH, 0.0]),
            pos=np.zeros(3),
            mat=np.eye(3).reshape(-1),
            rgba=rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            CAMERA_AXIS_WIDTH,
            p0,
            p1,
        )
        return index + 1

    def draw_camera_frames(viewer) -> None:
        """
        Draw the exact URDF f_oc and f_dc local axes in the MuJoCo scene.

        X = red, Y = green, Z = blue.
        Each camera has its own origin sphere plus three arrows.
        """
        viewer.user_scn.ngeom = 0
        idx = 0

        camera_defs = (
            ("f_oc", T_BASE_OC_POS, R_BASE_OC, np.array([0.2, 0.8, 1.0, 1.0])),
            ("f_dc", T_BASE_DC_POS, R_BASE_DC, np.array([1.0, 0.6, 0.2, 1.0])),
        )

        axis_rgba = (
            np.array([1.0, 0.1, 0.1, 1.0]),  # +X red
            np.array([0.1, 1.0, 0.1, 1.0]),  # +Y green
            np.array([0.1, 0.3, 1.0, 1.0]),  # +Z blue
        )

        for _, p_base_cam, r_base_cam, origin_rgba in camera_defs:
            p_world_cam, r_world_cam = camera_frame_world(
                p_base_cam, r_base_cam
            )

            # Origin sphere.
            if idx >= viewer.user_scn.maxgeom:
                raise RuntimeError("MuJoCo user scene geom capacity exhausted.")
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.012, 0.0, 0.0]),
                pos=p_world_cam,
                mat=np.eye(3).reshape(-1),
                rgba=origin_rgba,
            )
            idx += 1

            # Columns of R_world_cam are local +X, +Y, +Z in world.
            for axis_i in range(3):
                p1 = (
                    p_world_cam
                    + CAMERA_AXIS_LENGTH * r_world_cam[:, axis_i]
                )
                idx = add_arrow(
                    viewer.user_scn,
                    idx,
                    p_world_cam,
                    p1,
                    axis_rgba[axis_i],
                )

        viewer.user_scn.ngeom = idx

    # Numerical sanity check before opening the GUI.
    oc_p_w, oc_r_w = camera_frame_world(T_BASE_OC_POS, R_BASE_OC)
    dc_p_w, dc_r_w = camera_frame_world(T_BASE_DC_POS, R_BASE_DC)

    print()
    print("=" * 88)
    print("URDF CAMERA FRAME CHECK")
    print("=" * 88)
    print("f_oc origin world [m] =", np.round(oc_p_w, 6))
    print("f_dc origin world [m] =", np.round(dc_p_w, 6))
    print("f_oc +X world         =", np.round(oc_r_w[:, 0], 6))
    print("f_oc +Y world         =", np.round(oc_r_w[:, 1], 6))
    print("f_oc +Z world         =", np.round(oc_r_w[:, 2], 6))
    print("f_dc +X world         =", np.round(dc_r_w[:, 0], 6))
    print("f_dc +Y world         =", np.round(dc_r_w[:, 1], 6))
    print("f_dc +Z world         =", np.round(dc_r_w[:, 2], 6))
    print()
    print("For identity base attitude, expected approximately:")
    print("  f_oc +Z = [ 1.0000,  0.0000,  0.0000]")
    print("  f_dc +Z = [ 0.7071,  0.0000, -0.7071]")
    print("Axis colors in viewer: X=red, Y=green, Z=blue")
    print("=" * 88)
    print()

    def compute_ee_pose_plb() -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return kp0_xyz, yaw/pitch/roll and full 9-D keypoints in PLB."""
        base_pos_w = data.qpos[0:3].copy().astype(np.float32)
        base_quat_w = quat_unique_wxyz(
            data.qpos[3:7].copy().astype(np.float32)
        )

        # PLB orientation uses only base yaw.
        _, _, base_yaw = euler_xyz_from_quat_wxyz(base_quat_w)
        plb_quat_w = quat_from_yaw_wxyz(base_yaw)
        plb_quat_w = quat_unique_wxyz(
            quat_normalize_wxyz(plb_quat_w)
        )

        # PLB origin uses ground reference for z, not base_pos_z.
        plb_pos_w = base_pos_w.copy()
        plb_pos_w[2] = ee_ground_z

        ee_pos_w = (
            data.xpos[ee_body_id].copy().astype(np.float32)
        )
        ee_rot_w = (
            data.xmat[ee_body_id]
            .reshape(3, 3)
            .copy()
            .astype(np.float32)
        )
        ee_quat_w = quat_from_rotmat_wxyz(ee_rot_w)

        kp0_plb = quat_apply_inverse_wxyz(
            plb_quat_w,
            ee_pos_w - plb_pos_w,
        )

        ee_quat_plb = quat_mul_wxyz(
            quat_conjugate_wxyz(plb_quat_w),
            ee_quat_w,
        )
        ee_quat_plb = quat_unique_wxyz(
            quat_normalize_wxyz(ee_quat_plb)
        )

        # Utility returns XYZ Euler order: roll, pitch, yaw.
        roll, pitch, yaw = euler_xyz_from_quat_wxyz(
            ee_quat_plb
        )
        ypr = np.array(
            [yaw, pitch, roll],
            dtype=np.float32,
        )

        off_x = np.array(
            [ee_kp_dx, 0.0, 0.0],
            dtype=np.float32,
        )
        off_z = np.array(
            [0.0, 0.0, ee_kp_dz],
            dtype=np.float32,
        )

        kp1_plb = kp0_plb + quat_apply_wxyz(
            ee_quat_plb,
            off_x,
        )
        kp2_plb = kp0_plb + quat_apply_wxyz(
            ee_quat_plb,
            off_z,
        )

        keypoints_plb = np.concatenate(
            [kp0_plb, kp1_plb, kp2_plb]
        ).astype(np.float32)

        return (
            kp0_plb.astype(np.float32),
            ypr,
            keypoints_plb,
        )

    kp0_xyz, yaw_pitch_roll, ee_keypoints_plb = (
        compute_ee_pose_plb()
    )

    yaw, pitch, roll = yaw_pitch_roll
    joint2 = default_joint_pos[
        MUJOCO_JOINT_NAMES.index("joint2")
    ]
    joint3 = default_joint_pos[
        MUJOCO_JOINT_NAMES.index("joint3")
    ]
    joint4 = default_joint_pos[
        MUJOCO_JOINT_NAMES.index("joint4")
    ]

    print("=" * 88)
    print("LOCKED B2WZ1 POSE")
    print("=" * 88)
    print("YAML                    =", yaml_path)
    print("XML                     =", xml_path)
    print("EE body                 =", ee_body_name)
    print("PLB origin              =", "[base_x, base_y, ee_ground_z]")
    print("ee_ground_z             =", f"{ee_ground_z:.6f}")
    print()
    print("ARM JOINTS")
    print(
        "joint2/3/4 [rad]        =",
        np.round([joint2, joint3, joint4], 6),
    )
    print(
        "joint2/3/4 [deg]        =",
        np.round(np.degrees([joint2, joint3, joint4]), 3),
    )
    print()
    print("EE POSE IN PLB")
    print("kp0_xyz [m]             =", np.round(kp0_xyz, 6))
    print(
        "yaw/pitch/roll [rad]    =",
        np.round(yaw_pitch_roll, 6),
    )
    print(
        "yaw/pitch/roll [deg]    =",
        np.round(np.degrees(yaw_pitch_roll), 3),
    )
    print()
    print("FULL 9-D KEYPOINT COMMAND IN PLB")
    print("ee_keypoints_plb        =", np.round(ee_keypoints_plb, 6))
    print("kp0                     =", np.round(ee_keypoints_plb[0:3], 6))
    print("kp1 (+EE x axis)        =", np.round(ee_keypoints_plb[3:6], 6))
    print("kp2 (+EE z axis)        =", np.round(ee_keypoints_plb[6:9], 6))
    print("=" * 88)
    print("Close the MuJoCo viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = data.qpos[0:3]

        while viewer.is_running():
            start_time = time.perf_counter()

            # Kinematic visualization only. No mj_step().
            write_locked_state()
            viewer.cam.lookat[:] = data.qpos[0:3]

            # Custom visualization: exact URDF f_oc/f_dc frames.
            draw_camera_frames(viewer)
            viewer.sync()

            remaining = refresh_dt - (
                time.perf_counter() - start_time
            )
            if remaining > 0.0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()