# E3 v3 → random full-pose E3 replacement (2026-11-11)

The fixed line/circle waypoint benchmark (E3 v3) was replaced by the random discrete full-pose benchmark (`docs/E3_PROTOCOL.md`). The redesign was made after the E3 v3 results had been examined (disclosed in the protocol, section 0).

**Preserved.** The pre-replacement code is git tag `e3v3-pre-random-pose-replacement` (commit `64a2933`), which includes the E3 v3 pre-registration and all v3 analysis and figure scripts.

**Deleted data.** `results/E3_v3/`: 92,837 files, 17.8 GB, including 15,000 raw episodes, statistics, figures, pilot and screening outputs. The SHA-256 of every file is in `results/E3/cleanup/deleted_E3_v3_sha256.txt.gz`. The earlier (v1/v2) deletion manifest is copied alongside.

**Deleted code.**
- The whole `benchmark/e3v3/` package (candidate screening, pilot, extractor, analysis, figures, paper figures).
- The E3 v3 waypoint schedule (`core/scenario.py`).
- The E3 v3 builder (`scenarios/experiments.py`).
- `configs/E3.yaml` (v3).
- The `TestE3v3Waypoints` unit tests.
- `e3v3/kinematics.py` was moved to `benchmark/e3/kinematics.py` and extended.

**Kept as history.** `docs/E3_V3_PREREGISTRATION.md` and `docs/E3_V3_CLEANUP.md`.

**Untouched.** E1, E2 and E4 code and data, controllers, checkpoints, the runner and shared infrastructure. A hash check of the canonical E1/E2/E1-500 outputs is in `results/E3/E3_VALIDATION_REPORT.md`.
