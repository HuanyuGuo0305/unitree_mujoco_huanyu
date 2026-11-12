"""Shared I/O helpers for the E1 and E3 analyses (path setup, config, atomic CSV/JSON/text
writers with full float precision). The package name is historical (round-3 tooling);
the round-3 episode loaders that used to live here were removed in the 2026-11-12
cleanup together with that tooling. Output formatting is unchanged.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import yaml

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for _root in (REPO_ROOT, SIM_ROOT):
    if _root not in sys.path:
        sys.path.insert(0, _root)

BENCH = os.path.join(SIM_ROOT, "benchmark")


def load_global_config() -> dict:
    with open(os.path.join(BENCH, "configs", "baselines.yaml"), encoding="utf-8") as h:
        return yaml.safe_load(h)


def stats_config() -> dict:
    return dict(load_global_config()["statistics"])


def write_csv(path: str, rows: list[dict], columns=None) -> str:
    """Atomic CSV write: build a temporary file, fsync, then rename.

    An interrupted run can therefore never leave a half-written results file
    behind for the next stage to read as if it were complete.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in columns})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def write_json(path: str, payload) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def write_text(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def _fmt(value):
    """Full double precision for floats: these CSVs are the input to later
    statistical work, so nothing is rounded for display here."""
    if isinstance(value, float):
        if not np.isfinite(value):
            return "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
        return repr(value)
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return value


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
