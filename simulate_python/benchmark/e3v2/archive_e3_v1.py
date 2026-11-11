#!/usr/bin/env python3
"""Archive the E3 v1 (3x3 alpha grid, 50 seeds) dataset before E3 v2 replaces it.

Moves -- never deletes -- every E3 v1 artefact into results/archive_e3_v1/,
preserving relative paths, after hashing each file; verifies every moved file
against its hash; snapshots the v1 config. Refuses to run twice.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import sys
import time

BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(BENCH, "results")
DEST = os.path.join(RES, "archive_e3_v1")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.exists(DEST):
        sys.exit(f"refusing: {DEST} already exists")
    targets = [os.path.join(RES, "raw", "E3"), os.path.join(RES, "final_round3", "E3")]
    targets += [p for p in glob.glob(os.path.join(RES, "final_round3", "*", "E3_*"))
                if not p.startswith(os.path.join(RES, "final_round3", "E3"))]
    targets += glob.glob(os.path.join(RES, "final_round3", "validation", "*E3*"))
    targets = sorted(set(t for t in targets if os.path.exists(t)))
    files = []
    for t in targets:
        if os.path.isdir(t):
            for root, _d, fs in os.walk(t):
                files += [os.path.join(root, f) for f in fs]
        else:
            files.append(t)
    print(f"hashing {len(files)} files in {len(targets)} targets")
    manifest = {os.path.relpath(f, RES): sha(f) for f in sorted(files)}
    os.makedirs(DEST)
    for t in targets:
        rel = os.path.relpath(t, RES)
        os.makedirs(os.path.dirname(os.path.join(DEST, rel)), exist_ok=True)
        shutil.move(t, os.path.join(DEST, rel))
    bad = [r for r, h in manifest.items() if sha(os.path.join(DEST, r)) != h]
    if bad:
        sys.exit(f"{len(bad)} moved files fail verification: {bad[:3]}")
    os.makedirs(os.path.join(DEST, "configs"), exist_ok=True)
    shutil.copy2(os.path.join(BENCH, "configs", "E3.yaml"), os.path.join(DEST, "configs", "E3_v1.yaml"))
    n_ep = len(glob.glob(os.path.join(DEST, "raw", "E3", "*", "*", "seed_*")))
    with open(os.path.join(DEST, "ARCHIVE_MANIFEST.json"), "w") as f:
        json.dump({"archived_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "reason": "E3 redesigned as v2 (5x5 base-speed x EE-speed grid, arc-length "
                             "trajectories, 200 seeds); v1 superseded, kept for traceability",
                   "n_files": len(manifest), "n_episodes": n_ep,
                   "moved_targets": [os.path.relpath(t, RES) for t in targets],
                   "sha256": manifest}, f, indent=1)
    with open(os.path.join(DEST, "ARCHIVE_NOTE.md"), "w") as f:
        f.write(
            "# E3 v1 — archived, superseded by E3 v2\n\n"
            f"{n_ep} episodes (3 controllers x 3 alpha_base x 3 alpha_ee x 3 families x 50 seeds) "
            "and every derived E3 table, figure, showcase and validation file from round 3, moved "
            "here unchanged (SHA-256 of every file in ARCHIVE_MANIFEST.json, verified after the "
            "move). Config: configs/E3_v1.yaml. The v1 scenario builder is still available in "
            "`benchmark/scenarios/experiments.py` (`build_e3_v1`, selected when the config has no "
            "`protocol_version: 2`), so these episodes remain reproducible.\n\n"
            "The canonical E3 dataset is now `results/E3_v2/`. The round-3 report's E3 sections "
            "describe v1 and are superseded. E1 and E2 are unaffected.\n")
    print(f"archived {len(manifest)} files ({n_ep} episodes) -> {os.path.relpath(DEST, BENCH)}; all verified")


if __name__ == "__main__":
    main()
