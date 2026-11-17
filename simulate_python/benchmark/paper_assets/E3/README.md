# E3 paper assets (tracked copy) — bidirectional 7 × 7 benchmark

`results/` is excluded from git, so this folder keeps a tracked copy of the official E3 figure and the numbers behind it. The protocol is in `E3_PROTOCOL.md`; results and findings are in `E3_REPORT.md`.

## Contents

- **`figures/`:** the paper figure `wbc_coordination` (PDF, SVG, 600 dpi PNG; 7.16 × 2.95 in), its exact plotted values (`_values.csv`) and the plot checks (`_checks.json`).
- **`stats/`:**
  - `E3_summary.csv`: long format, per controller × condition × metric.
  - `E3_summary_7x7.csv`: machine-readable 7 × 7 table with all four metrics.
  - Paired comparisons, slice trends, completed-only sensitivity and completion.
- **`data/`:** `E3_per_trial_14700.csv` (every episode) and `E3_arrays.npz`, which is enough to regenerate the figure.
- **`protocol/`:** config snapshot; the frozen command schedules and their sha256; frozen-artefact hashes; code hashes and git state at launch; pre-run validation; pose-audit summary.
- **`validation/`:** post-run validation, including the list of all 543 early terminations.
- **`E3_FIGURE_INSERT.tex`:** a draft `figure*` block with caption. It is not inserted into the manuscript.

**Not copied:** `results/E3/raw/`, the 14,700 episode logs (16 GB). They are hashed in `results/E3/RAW_SHA256.txt`. `MANIFEST_SHA256.txt` hashes the other `results/E3/` files, with paths relative to `results/E3/`.

## Regenerating (from `simulate_python/`)

```
python3 -m benchmark.e3.analyze          # needs results/E3/raw
python3 -m benchmark.e3.paper_figures
```

The previous 5 × 5 E3 assets are archived; see `docs/E3_BIDIRECTIONAL_REPLACEMENT.md`.
