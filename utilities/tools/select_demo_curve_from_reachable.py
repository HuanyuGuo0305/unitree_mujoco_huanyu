import os
import argparse
import numpy as np

"""
Example:

python3 utilities/tools/select_demo_curve_from_reachable.py \
    --input utilities/data/reachable_kp0kp1kp2_lb.npy \
    --output utilities/data/demo_sweep_curve.npy \
    --num-left 6 \
    --num-right 12 
"""


def make_default_pose_row(default_kp0, kp_dx, kp_dz):
    """Create one row from the default EE pose."""
    default_kp0 = np.asarray(default_kp0, dtype=np.float32).reshape(3,)
    kp1 = default_kp0 + np.array([kp_dx, 0.0, 0.0], dtype=np.float32)
    kp2 = default_kp0 + np.array([0.0, 0.0, kp_dz], dtype=np.float32)
    return np.concatenate([default_kp0, kp1, kp2], axis=0).astype(np.float32)


def filter_front_workspace(arr, x_min, x_max, y_min, y_max, z_min, z_max):
    """Keep only poses whose kp0 lies inside the specified front workspace."""
    kp0 = arr[:, 0:3]
    mask = (
        (kp0[:, 0] >= x_min) & (kp0[:, 0] <= x_max) &
        (kp0[:, 1] >= y_min) & (kp0[:, 1] <= y_max) &
        (kp0[:, 2] >= z_min) & (kp0[:, 2] <= z_max)
    )
    return arr[mask]


def find_start_index(arr, default_kp0):
    """Find the pose whose kp0 is closest to the default pose."""
    kp0 = arr[:, 0:3]
    dist = np.linalg.norm(kp0 - default_kp0[None, :], axis=1)
    return int(np.argmin(dist))


def choose_next_in_direction(arr, current_idx, used_mask, max_step, y_direction, x_ref, z_ref):
    """
    Choose the next point with a directional preference along y.

    y_direction:
        -1 : move toward smaller y (go left)
        +1 : move toward larger y (go right)

    Scoring preference:
        1) progress along y in the desired direction
        2) keep x near x_ref
        3) keep z near z_ref
        4) use a reasonable fraction of the step budget
    """
    kp0 = arr[:, 0:3]
    current = kp0[current_idx]

    delta = kp0 - current[None, :]
    dist = np.linalg.norm(delta, axis=1)

    # Only consider unused points within step threshold
    valid = (~used_mask) & (dist > 1e-6) & (dist <= max_step)

    # Enforce directional progress along y
    y_progress = y_direction * (kp0[:, 1] - current[1])
    valid = valid & (y_progress > 1e-4)

    cand_ids = np.where(valid)[0]
    if cand_ids.size == 0:
        return None

    cand = kp0[cand_ids]
    step_len = dist[cand_ids]
    y_gain = y_direction * (cand[:, 1] - current[1])

    # Larger y progress is better
    # Staying near reference x and z is better
    # Slightly prefer longer valid steps to cover more space
    score = (
        4.0 * y_gain
        - 2.0 * np.abs(cand[:, 0] - x_ref)
        - 1.5 * np.abs(cand[:, 2] - z_ref)
        + 0.5 * step_len
    )

    best = int(np.argmax(score))
    return int(cand_ids[best])


def build_sweep_curve(arr, default_kp0, num_left, num_right, max_step):
    """
    Build a sweep curve:
        default -> left -> right
    """
    kp0 = arr[:, 0:3]
    start_idx = find_start_index(arr, default_kp0)

    used_mask = np.zeros(arr.shape[0], dtype=bool)
    curve = [start_idx]
    used_mask[start_idx] = True

    x_ref = float(default_kp0[0])
    z_ref = float(default_kp0[2])

    # Stage 1: move from default toward left (smaller y)
    while len(curve) < 1 + num_left:
        nxt = choose_next_in_direction(
            arr=arr,
            current_idx=curve[-1],
            used_mask=used_mask,
            max_step=max_step,
            y_direction=-1,
            x_ref=x_ref,
            z_ref=z_ref,
        )
        if nxt is None:
            break
        curve.append(nxt)
        used_mask[nxt] = True

    # Stage 2: from current left point, sweep toward right (larger y)
    target_total = 1 + num_left + num_right
    while len(curve) < target_total:
        nxt = choose_next_in_direction(
            arr=arr,
            current_idx=curve[-1],
            used_mask=used_mask,
            max_step=max_step,
            y_direction=+1,
            x_ref=x_ref,
            z_ref=z_ref,
        )
        if nxt is None:
            break
        curve.append(nxt)
        used_mask[nxt] = True

    return arr[curve]


def main():
    parser = argparse.ArgumentParser(description="Select a left-to-right sweep demo curve from reachable keypoint poses")

    parser.add_argument("--input", type=str, required=True, help="Input reachable .npy file")
    parser.add_argument("--output", type=str, required=True, help="Output curve .npy file")

    parser.add_argument("--num-left", type=int, default=6, help="Number of points from default toward left")
    parser.add_argument("--num-right", type=int, default=12, help="Number of points from left toward right")
    parser.add_argument("--max-step", type=float, default=0.20, help="Max adjacent kp0 distance in meters")

    parser.add_argument("--kp-dx", type=float, default=0.30)
    parser.add_argument("--kp-dz", type=float, default=0.30)

    parser.add_argument("--default-kp0-x", type=float, default=0.55)
    parser.add_argument("--default-kp0-y", type=float, default=0.0)
    parser.add_argument("--default-kp0-z", type=float, default=0.51)

    # Front workspace filter
    parser.add_argument("--x-min", type=float, default=0.45)
    parser.add_argument("--x-max", type=float, default=0.70)
    parser.add_argument("--y-min", type=float, default=-0.20)
    parser.add_argument("--y-max", type=float, default=0.20)
    parser.add_argument("--z-min", type=float, default=0.38)
    parser.add_argument("--z-max", type=float, default=0.68)

    args = parser.parse_args()

    arr = np.load(args.input).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(f"Expected input shape (N, 9), got {arr.shape}")

    default_kp0 = np.array(
        [args.default_kp0_x, args.default_kp0_y, args.default_kp0_z],
        dtype=np.float32,
    )

    # Keep only front workspace points
    arr_front = filter_front_workspace(
        arr,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    if arr_front.shape[0] == 0:
        raise ValueError("No reachable poses remain after workspace filtering.")

    traj = build_sweep_curve(
        arr_front,
        default_kp0=default_kp0,
        num_left=args.num_left,
        num_right=args.num_right,
        max_step=args.max_step,
    )

    # Force the first row to be the exact default pose
    traj[0] = make_default_pose_row(default_kp0, args.kp_dx, args.kp_dz)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, traj)

    kp0 = traj[:, 0:3]
    step_dist = np.linalg.norm(kp0[1:] - kp0[:-1], axis=1) if len(kp0) > 1 else np.array([])

    print("=" * 80)
    print("Saved sweep curve:")
    print(out_path)
    print("shape:", traj.shape)
    print("-" * 80)
    print("kp0 range:")
    print("  x:", float(kp0[:, 0].min()), "->", float(kp0[:, 0].max()))
    print("  y:", float(kp0[:, 1].min()), "->", float(kp0[:, 1].max()))
    print("  z:", float(kp0[:, 2].min()), "->", float(kp0[:, 2].max()))
    if step_dist.size > 0:
        print("-" * 80)
        print("adjacent kp0 distance:")
        print("  min:", float(step_dist.min()))
        print("  max:", float(step_dist.max()))
        print("  mean:", float(step_dist.mean()))
    print("-" * 80)
    print("first row:")
    print(np.round(traj[0], 6))
    print("=" * 80)


if __name__ == "__main__":
    main()