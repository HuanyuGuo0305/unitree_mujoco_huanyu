# E3 paper assets (tracked copy)

`results/` is excluded from git, so this folder holds a tracked copy of the final E3 paper figure and the outputs it depends on.

**Contents:**
- `figures/`: the final combined figure (PDF and PNG), its exact plotted values, the unmatched-event sensitivity, and `FIGURE_VALIDATION.md`.
- `figures/supplementary/`: the signed vs positive excess diagnostic figure.
- `stats/`: all E3 statistics CSVs.
- `validation/`: the post-run and positive-excess check JSONs.
- The reports and `E3_PAPER_FIGURES_INSERT.tex`.

**Not copied, to avoid duplicating large datasets:** `results/E3/raw/` (7,500 episode logs), `data/E3_per_trial_7500.csv`, and the `.npz` arrays. Their SHA-256 hashes are in `MANIFEST_SHA256.txt`, which uses paths relative to `results/E3/`.

**Regenerating** (from `simulate_python/`, with the raw logs present):

```
python3 -m benchmark.e3.positive_excess
python3 -m benchmark.e3.paper_figures
```

Then copy the outputs here again.
