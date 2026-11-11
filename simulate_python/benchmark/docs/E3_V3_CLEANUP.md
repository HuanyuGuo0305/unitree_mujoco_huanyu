# E3 cleanup before the v3 rebuild (2026-11-11)

All earlier E3 implementations and results were removed before E3 v3 was built.

The pre-cleanup code is preserved in git as tag **`e3-pre-v3-cleanup`** (commit `36911c8` on branch `benchmark-e3v3`). That commit contains the complete benchmark code with the E3 v1 and v2 implementations, plus the `--headless` changes in `deploy_mujoco/` that the runner depends on. Restore a file with `git show e3-pre-v3-cleanup:<path>`.

## Deleted data (not in git; results are gitignored)

| What | Files | Notes |
|---|---|---|
| `results/archive_e3_v1/` | 48,091 | Round-2/3 E3 v1 data: 8,010 episodes plus derived files. |
| `results/E3_v2/` | 270,000 raw + 75 derived | The canonical E3 v2 dataset (45,000 episodes), its audit, figures and reports. |
| `results/archive_round2/**/*E3*` | 79 | Round-1/2 E3 figures, tables, summary, showcase, validation log and `configs/E3.yaml`. The E1/E2 contents of the archive were kept. |

The total is 318,255 files, about 41.1 GB. The SHA-256 of every deleted file is in `results/E3_v3/cleanup/deleted_files_sha256.txt.gz`. The v1 archive manifest and note are kept next to it.

## Deleted or edited code

**Deleted files:**
- `benchmark/e3v2/` (whole package)
- `configs/E3_v1.yaml`
- `configs/E3.yaml` (the v2 configuration; replaced by the v3 file)
- `round3/verify_e3_reuse.py`
- `analysis.py` (E3-only round-2 analysis)

**E3-only functions and branches removed from shared modules.** Each listed module's E1/E2/E4 code is unchanged.

| Module | E3 code removed |
|---|---|
| `core/scenario.py` | E3 v1 trajectory spec and schedule; E3 v2 arc-length paths |
| `scenarios/experiments.py` | `build_e3`, `build_e3_v1`, `build_e3_v2`, condition enumeration, `SettleHoldScenario` |
| `round3/export.py`, `aggregate3.py`, `figures3.py`, `tables3.py`, `select3.py`, `compare_rounds3.py` | E3 export, statistics, figures, tables, selection and seed-pool comparison |
| `round3/validate.py` | V5 retired; E3 removed from V6–V14 |
| `aggregate.py`, `plot.py`, `publish.py`, `select_cases.py`, `showcase.py`, `compare_rounds.py` | E3 branches |
| `plots/figures.py`, `plots/figures_round2.py`, `plots/tables.py`, `plots/tables_round2.py` | E3 figure and table generators and their helpers |

**Tests.** E3 v1 unit tests were removed from `tests/test_scenarios.py`, `tests/test_round2_analysis.py` and `tests/test_round3_protocol.py`, including the two tests of the removed E3 family-pooling helper.

**Left in place:**
- `archive_round2.py`: the historical record of an action that was already executed; its E3 mentions describe what it did.
- `results/final_round3/ROUND3_REPORT.md`: its E3 sections describe the removed E3 v1 and are superseded.

## Verification that E1/E2 were preserved

**Pipeline regression.** The E1/E2 parts of the round-3 pipeline (export, aggregate3, select3, tables3, figures3) and of the round-2 `aggregate.py` were run into a scratch directory before and after the edits.
- All 31 comparable outputs are byte-identical. These are CSV, TeX and PNG files; PDFs embed timestamps and were not compared.
- `tables3` previously crashed on the missing E3 data. It now completes, and the E1/E2 selected-case tables it writes are byte-identical to the canonical ones in `final_round3/tables/`.

**Canonical files.** The SHA-256 of all 63 canonical files under `results/final_round3` and `results/E1_500seed` (excluding raw) matched before and after the cleanup. `results/raw/E1` and `results/raw/E2` were not touched.

**Unit tests.** `test_scenarios` (47), `test_round2_analysis` (21) and `test_round3_protocol` (32) pass, and `test_latex_structure` passes (6/6).
