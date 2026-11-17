# 5 × 5 E3 → bidirectional 7 × 7 E3 replacement (2026-11-13)

The 5 × 5 random-full-pose E3 is superseded by the bidirectional 7 × 7 benchmark (`docs/E3_PROTOCOL.md`). The work is on branch `e3_bidirectional`; the published `wbc_benchmark` branch is unchanged.

## Archived and recoverable

**Code.**
- Local git tag `e3-5x5-pre-bidirectional-replacement` on commit `0c46a35`, the `wbc_benchmark` HEAD.
- A tarball of the benchmark code tree and per-file hashes (`code_SHA256.txt`).

**Results.** `unitree_mujoco_benchmark_archive_E3_5x5_2026-11-13/`, in the folder that contains the repository. It holds:
- `results/E3_random_pose_5x5/`: the complete former `results/E3/`, moved, not copied. That is 45,061 files and 8.1 GB, including 7,500 raw episodes, data, stats, figures, protocol, validation and reports.
  - Every file's sha256 was taken before the move and verified after it (`ALL_FILES_SHA256.txt`).
  - The original `MANIFEST_SHA256.txt` also verifies.
- `paper_assets/E3_random_pose_5x5/`: the former tracked `paper_assets/E3/`, with `ASSETS_SHA256.txt` verified.
- `docs/`: the 5 × 5 `E3.yaml`, `E3_feasible_rows.npy` and the E3 protocol and history documents.
- `code_snapshot/`: code tarball, git HEAD, status and diff.

## Changed in the active tree

- **Updated:**
  - `configs/E3.yaml`: the 7 × 7 levels and figure slices.
  - `scenarios/experiments.py`: E3 builder (protocol tag, pose-sequence length 16 → 28, slice helper).
  - `e3/extract.py`: RMSE and integrity only.
  - `e3/analyze.py`, `e3/prerun_validation.py`, `e3/paper_figures.py`.
  - `tests/test_scenarios.py` (`TestE3Bidirectional`) and `tests/test_protocol.py`.
  - `docs/E3_PROTOCOL.md`, `README.md`.
  - `paper_assets/E3/`.
- **Removed:** `e3/events_matched.py` and `e3/positive_excess.py` (event-aligned analyses), together with their figure panels and outputs. Nothing outside E3 imported them.
- **Kept unchanged:**
  - `e3/stats.py`, which E2's analysis also imports;
  - `e3/kinematics.py`, `e3/pose_audit.py` and `core/`;
  - the controllers, runner and `baselines.yaml`;
  - all E1 and E2 code, data and assets.
