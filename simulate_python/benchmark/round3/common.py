"""Shared loading and bookkeeping for the round-3 final evaluation.

Round-3 rules enforced in this module, so that no downstream script can quietly
break them:

* The official population is **exactly** seeds 0-49, read from
  ``configs/baselines.yaml``. A condition that does not have all 50 is reported
  as incomplete rather than aggregated over whatever is present.
* Every episode that completed is used, terminated episodes included. No
  episode is ever dropped for being unfavourable; ``survived`` and
  ``termination_reason`` travel with every row.
* Per-seed values are the unit of analysis. Nothing here averages over seeds.
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
RAW = os.path.join(BENCH, "results", "raw")
OUT = os.path.join(BENCH, "results", "final_round3")

# Metrics every experiment reports, lower-is-better unless noted.
ABS_METRICS = ("base_vxy_rmse", "base_wz_rmse", "ee_pos_rmse", "ee_ori_mae_deg")


def load_global_config() -> dict:
    with open(os.path.join(BENCH, "configs", "baselines.yaml"), encoding="utf-8") as h:
        return yaml.safe_load(h)


def official_seeds() -> list[int]:
    cfg = load_global_config()["official_seeds"]
    return list(range(int(cfg["start"]), int(cfg["start"]) + int(cfg["count"])))


def showcase_seeds() -> list[int]:
    cfg = load_global_config()["showcase_seeds"]
    return list(range(int(cfg["start"]), int(cfg["start"]) + int(cfg["count"])))


def stats_config() -> dict:
    return dict(load_global_config()["statistics"])


def episode_dir(experiment: str, baseline: str, condition: str, seed: int) -> str:
    return os.path.join(RAW, experiment, baseline, condition, f"seed_{seed:03d}")


REQUIRED_FILES = ("commands.csv", "states.csv", "metrics.json", "metadata.json")


def episode_status(path: str) -> str:
    """``ok``, ``missing`` or ``incomplete:<detail>`` -- never silently ok."""
    if not os.path.isdir(path):
        return "missing"
    for name in REQUIRED_FILES:
        f = os.path.join(path, name)
        if not os.path.isfile(f):
            return f"incomplete:no_{name}"
        if os.path.getsize(f) == 0:
            return f"incomplete:empty_{name}"
    try:
        with open(os.path.join(path, "metrics.json"), encoding="utf-8") as h:
            metrics = json.load(h)
    except Exception as exc:  # noqa: BLE001
        return f"incomplete:unreadable_metrics({exc})"
    if int(metrics.get("n_eval_samples", 0)) <= 0:
        return "incomplete:no_evaluated_samples"
    return "ok"


def load_episode(experiment: str, baseline: str, condition: str, seed: int) -> dict | None:
    """One episode as a flat row, or ``None`` if it is missing/incomplete."""
    path = episode_dir(experiment, baseline, condition, seed)
    if episode_status(path) != "ok":
        return None
    with open(os.path.join(path, "metrics.json"), encoding="utf-8") as h:
        metrics = json.load(h)
    with open(os.path.join(path, "metadata.json"), encoding="utf-8") as h:
        meta = json.load(h)
    row = dict(metrics)
    scen = meta.get("scenario", {})
    row.update(
        {
            "experiment": experiment,
            "baseline": baseline,
            "condition": condition,
            "seed": int(seed),
            "settle_s": float(meta["settle_s"]),
            "duration_s": float(meta["duration_s"]),
            "control_dt": float(meta["control_dt"]),
            "physics_dt": float(meta["physics_dt"]),
            "path": os.path.relpath(path, BENCH),
        }
    )
    for key in ("alpha_base", "alpha_ee", "ee_family", "base_intensity",
                "base_max_abs_vx_mps", "base_max_abs_yaw_rate_radps",
                "ee_mean_speed_mps", "base_mode", "ee_mode",
                "base_segment_duration_range_s", "ee_resample_interval_s"):
        if key in scen:
            row[key] = scen[key]
    return row


def load_official(experiment: str, baselines, conditions, seeds=None) -> tuple[list[dict], list[dict]]:
    """Load the full official grid. Returns (rows, problems).

    ``problems`` lists every (baseline, condition, seed) that is not a usable
    episode, with the reason. It is written to disk and reported; it is never
    swallowed.
    """
    seeds = official_seeds() if seeds is None else list(seeds)
    rows, problems = [], []
    for baseline in baselines:
        for condition in conditions:
            for seed in seeds:
                row = load_episode(experiment, baseline, condition, seed)
                if row is None:
                    problems.append(
                        {
                            "experiment": experiment, "baseline": baseline,
                            "condition": condition, "seed": seed,
                            "status": episode_status(
                                episode_dir(experiment, baseline, condition, seed)
                            ),
                        }
                    )
                else:
                    rows.append(row)
    return rows, problems


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


def per_seed_map(rows, baseline, metric, where=None) -> dict:
    """seed -> metric for one baseline, optionally filtered. Used for pairing."""
    out = {}
    for row in rows:
        if row["baseline"] != baseline:
            continue
        if where and any(row.get(k) != v for k, v in where.items()):
            continue
        value = row.get(metric)
        if value is None:
            continue
        out[int(row["seed"])] = float(value)
    return out
