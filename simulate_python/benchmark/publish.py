#!/usr/bin/env python3
"""Generate the round-2 publication tables and figures.

    python3 benchmark/publish.py                      # everything
    python3 benchmark/publish.py --only selected      # just the curated cases

Official artefacts use the official seed pool from ``configs/baselines.yaml``
with no exclusions. Illustrative artefacts are read from the selection
manifests written by ``benchmark/select_cases.py`` and are labelled as curated
examples in both the table captions and the figure footers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIM_ROOT, ".."))
for root in (REPO_ROOT, SIM_ROOT):
    if root not in sys.path:
        sys.path.insert(0, root)

from benchmark import aggregate  # noqa: E402
from benchmark.plots import figures_round2 as figs  # noqa: E402
from benchmark.plots import tables_round2 as tabs  # noqa: E402
from benchmark.scenarios import experiments as E  # noqa: E402

RESULTS_ROOT = os.path.join(SIM_ROOT, "benchmark", "results")


def _official(raw_root, experiment, global_cfg):
    rows = aggregate.load_raw(raw_root, [experiment])
    pool = global_cfg["official_seeds"]
    return aggregate.filter_official(rows, pool["start"], pool["count"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=os.path.join(RESULTS_ROOT, "raw"))
    parser.add_argument("--out", default=RESULTS_ROOT)
    parser.add_argument("--only", action="append",
                        choices=["E1", "E2", "selected"], default=None)
    args = parser.parse_args(argv)
    todo = set(args.only or ["E1", "E2", "selected"])

    _, global_cfg = E.load_config("E1")
    cfg_stats = global_cfg["statistics"]
    tables_dir = os.path.join(args.out, "tables")
    figures_dir = os.path.join(args.out, "figures")
    show_dir = os.path.join(args.out, "showcase")

    if "E1" in todo:
        rows = _official(args.results, "E1", global_cfg)
        seeds = sorted({r["seed"] for r in rows})
        print(f"=== E1 official tables ({len(rows)} episodes, {len(seeds)} seeds "
              f"{seeds[0]}-{seeds[-1]}) ===")
        tabs.e1_official_table(rows, tables_dir, cfg_stats)

    if "E2" in todo:
        rows = _official(args.results, "E2", global_cfg)
        seeds = sorted({r["seed"] for r in rows})
        print(f"=== E2 official tables ({len(rows)} episodes, {len(seeds)} seeds "
              f"{seeds[0]}-{seeds[-1]}) ===")
        tabs.e2_official_table(rows, tables_dir, cfg_stats)

    if "selected" in todo:
        print("=== illustrative cases ===")
        for experiment, table_fn, figure_fn in (
            ("E1", tabs.e1_selected_table, figs.e1_selected_cases),
            ("E2", tabs.e2_selected_table, figs.e2_selected_cases),
        ):
            path = os.path.join(show_dir, f"{experiment}_selection_manifest.json")
            if not os.path.isfile(path):
                print(f"  {experiment}: no manifest at "
                      f"{os.path.relpath(path, SIM_ROOT)}; run select_cases.py first")
                continue
            with open(path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            cases = manifest["cases"]
            n = sum(1 for c in cases if c.get("selected"))
            print(f"  {experiment}: {n}/{len(cases)} criteria had a qualifying case")
            for case in cases:
                if not case.get("selected"):
                    print(f"    [{case['case_id']}] NOT SELECTED -- "
                          f"{case.get('selection_failure_reason', '')}")
            table_fn(cases, tables_dir)
            figure_fn(cases, figures_dir)

    print(f"\nTables -> {tables_dir}\nFigures -> {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
