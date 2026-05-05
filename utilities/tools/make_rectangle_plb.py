#!/usr/bin/env python3
import numpy as np


def make_keypoints_from_pos(
    kp0: np.ndarray,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
) -> np.ndarray:
    """
    Minimum version:
    EE orientation aligned with PLB frame.
    kp1 = kp0 + +X * kp_dx
    kp2 = kp0 + +Z * kp_dz
    """
    kp0 = np.asarray(kp0, dtype=np.float32).reshape(3)

    kp1 = kp0 + np.array([kp_dx, 0.0, 0.0], dtype=np.float32)
    kp2 = kp0 + np.array([0.0, 0.0, kp_dz], dtype=np.float32)

    return np.concatenate([kp0, kp1, kp2]).astype(np.float32)


def main():
    # Rectangle in PLB y-z plane
    x = 0.8

    y_left = -0.30
    y_right = 0.30

    z_low = 0.20
    z_high = 1.00

    kp_dx = 0.30
    kp_dz = 0.30

    # 顺序：左下 -> 左上 -> 右上 -> 右下
    corners_kp0 = np.array(
        [
            [x, y_left,  z_low],
            [x, y_left,  z_high],
            [x, y_right, z_high],
            [x, y_right, z_low],
        ],
        dtype=np.float32,
    )

    data = np.stack(
        [
            make_keypoints_from_pos(kp0, kp_dx=kp_dx, kp_dz=kp_dz)
            for kp0 in corners_kp0
        ],
        axis=0,
    )

    assert data.shape == (4, 9)

    out_path = "rectangle.npy"
    np.save(out_path, data)

    print(f"Saved: {out_path}")
    print(f"shape: {data.shape}")
    print("data:")
    print(np.round(data, 4))


if __name__ == "__main__":
    main()