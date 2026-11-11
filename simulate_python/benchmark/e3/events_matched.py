#!/usr/bin/env python3
"""Matched isolated-event analysis for the paper's event figures (Figs. 3-4).

    python3 -m benchmark.e3.events_matched

Events are recomputed from the per-tick logs with the registered definitions
(extract.py). Because an episode that terminates early loses its later events,
the comparison set per seed is the INTERSECTION of the isolated event ticks
present for all three controllers (same command events, matched across
controllers). Events are averaged within the seed first; seeds without any
matched event are dropped and counted. The unmatched all-events statistics
(analyze.py) are kept as a sensitivity check.
Conditions: EE->base B0T4 (main), B0T1-T3; base->EE B4T0 (main), B1T0-B3T0.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os

import numpy as np

from benchmark.round3 import common
from benchmark.e3 import extract as X
from benchmark.e3.stats import summarise, paired, finish_family

E3 = os.path.join(common.BENCH, "results", "E3")
RAW = os.path.join(E3, "raw", "E3")
CTRL = ("C1", "C3", "C5")
SEEDS = list(range(100))
CONDS = {"ee": [(0, 4), (0, 1), (0, 2), (0, 3)], "base": [(4, 0), (1, 0), (2, 0), (3, 0)]}
METRICS = {"ee": [("base_xy", q) for q in ("peak_pos_excess", "int_pos_excess", "recovery_s")]
           + [("yaw", q) for q in ("peak_pos_excess", "int_pos_excess", "recovery_s")]
           + [("ee_pos", "post_abs_mean"), ("ee_pos", "residual_abs"), ("ee_ori", "post_abs_mean"),
              ("ee_ori", "residual_abs")],
           "base": [("ee_pos", q) for q in ("peak_pos_excess", "int_pos_excess", "recovery_s")]
           + [("ee_ori", q) for q in ("peak_pos_excess", "int_pos_excess", "recovery_s")]
           + [("ee_pos", "baseline_abs"), ("ee_ori", "baseline_abs")]}
DESCR = ["baseline_abs", "peak_excess", "int_excess", "post_abs_mean", "residual_abs",
         "recovery_defined", "recovery_censored", "peak_pos_excess_b025", "int_pos_excess_b025",
         "peak_pos_excess_b100", "int_pos_excess_b100"]


def _episode(job):
    c, bi, ti, s, kind = job
    d = os.path.join(RAW, c, f"B{bi}T{ti}", f"seed_{s:03d}")
    cmd, st = X._read(os.path.join(d, "commands.csv")), X._read(os.path.join(d, "states.csv"))
    ec, es = cmd["evaluated"] == 1, st["evaluated"] == 1
    e = X.tick_errors(cmd, st, ec, es)
    n = int(es.sum())
    npz = np.load(os.path.join(d, "scenario_commands.npz"))
    sb = npz["base_commands"]
    ee_sw = [int(k) for k in npz["ee_switch_ticks"]]
    b_sw = [int(k) for k in npz["base_switch_ticks"] if int(k) > 0] if np.any(sb != 0) else []
    b_ch = ([0] if np.any(sb[0] != 0) else []) + b_sw
    cand, others = (ee_sw, b_ch) if kind == "ee" else (b_sw, ee_sw)
    out = {}
    for k in cand:
        if k - X.PRE < 0 or k + X.POST > n:
            continue
        if any(k - X.ISO_PRE <= o < k + X.POST for o in others):
            continue
        met = {f"{sg}_{q}": v for sg in X.SIGS for q, v in X.event_metrics(e[sg], k, sg).items()}
        tr = np.stack([e[sg][k - X.PRE:k + X.POST] for sg in X.SIGS])
        out[k] = (met, tr)
    return job, out


def run():
    jobs = [(c, bi, ti, s, kind) for kind, cl in CONDS.items() for (bi, ti) in cl for c in CTRL for s in SEEDS]
    with mp.get_context("fork").Pool(16) as pool:
        res = dict(pool.map(_episode, jobs, chunksize=8))
    per_ctrl, pair_rows, cover, traces = [], [], [], {}
    for kind, cl in CONDS.items():
        for (bi, ti) in cl:
            tag = f"B{bi}T{ti}"
            keys = sorted({k for _, _, mets in [(0, 0, METRICS[kind])] for k in
                           [f"{s}_{q}" for s, q in mets] + [f"{s}_{q}" for s in X.SIGS for q in DESCR]})
            V = {k: np.full((3, 100), np.nan) for k in keys}
            TR = np.full((3, 100, 4, X.TRACE_LEN), np.nan)
            n_match, n_all = [], []
            for si, s in enumerate(SEEDS):
                evs = [res[(c, bi, ti, s, kind)] for c in CTRL]
                common_ticks = sorted(set(evs[0]) & set(evs[1]) & set(evs[2]))
                n_match.append(len(common_ticks))
                n_all.append([len(x) for x in evs])
                if not common_ticks:
                    continue
                for ci in range(3):
                    for k in keys:
                        V[k][ci, si] = np.nanmean([evs[ci][t][0][k] for t in common_ticks])
                    TR[ci, si] = np.mean([evs[ci][t][1] for t in common_ticks], axis=0)
            nm = np.array(n_match)
            cover.append({"condition": tag, "event": kind, "seeds_with_matched_events": int((nm > 0).sum()),
                          "matched_events_total": int(nm.sum()), "matched_per_seed_mean": float(nm.mean()),
                          **{f"all_isolated_events_{c}": int(sum(a[i] for a in n_all)) for i, c in enumerate(CTRL)}})
            traces[tag] = TR
            mets = METRICS[kind]
            rows = []
            for k in keys:
                formal = int(any(k == f"{s}_{q}" for s, q in mets))
                for ci, c in enumerate(CTRL):
                    per_ctrl.append({"condition": tag, "event": kind, "metric": k, "controller": c,
                                     "formal": formal} | summarise(V[k][ci]))
                for r in ("C3", "C5"):
                    rows.append({"condition": tag, "event": kind, "metric": k, "comparison": f"{r}-C1",
                                 "rival": r, "formal": formal} | paired(V[k][0], V[k][CTRL.index(r)]))
            finish_family([r for r in rows if r["formal"]])
            for r in rows:
                if not r["formal"]:
                    r["verdict"] = "descriptive (not in the formal family)"
            pair_rows += rows
    S = os.path.join(E3, "stats")
    common.write_csv(os.path.join(S, "E3_events_matched_per_controller.csv"), per_ctrl)
    common.write_csv(os.path.join(S, "E3_events_matched_paired.csv"), pair_rows)
    common.write_csv(os.path.join(S, "E3_events_matched_coverage.csv"), cover)
    np.savez_compressed(os.path.join(E3, "data", "E3_event_traces_matched.npz"),
                        **{f"trace_{k}": v for k, v in traces.items()}, controllers=np.array(CTRL),
                        trace_signals=np.array(X.SIGS), trace_t=np.arange(-X.PRE, X.POST) * X.DT)
    for r in cover:
        print(r)


if __name__ == "__main__":
    run()
