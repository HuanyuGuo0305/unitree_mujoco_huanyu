#!/usr/bin/env python3
"""Post-hoc event-level positive-excess traces for base-command switches (Fig. E3(d)).

    python3 -m benchmark.e3.positive_excess

POST-HOC VISUALISATION (2026-11-11), added after inspecting the signed-mean
trace of the combined figure, in which positive and negative responses cancel
across events. No official metric changes.

For every valid isolated base-command switch i (EE target fixed; registered
event selection, baseline [-0.5, 0) s, response [0, 1.5) s; extract.py):
    de_i(t)   = e_i(t) - mean_{[-0.5,0)} e_i
    de_i^+(t) = max(de_i(t), 0)              (applied per event, BEFORE averaging)
Per seed: the mean over the events present for all three controllers (matched,
as events_matched.py), then the seed bootstrap (paired: same resampled seeds
for every controller). Unmatched (all events per controller) kept as
sensitivity. Conditions B4T0 (main) and B1T0-B3T0 (sensitivity).

Checks against the validated outputs:
 * signed per-event-baseline mean == stored matched trace minus its baseline;
 * per-seed mean of max_t de_i^+(t) == official ee_pos_peak_pos_excess;
 * per-seed mean over [0,1.5) of de_i^+ == official ee_pos_int_pos_excess / 1.5.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os

import numpy as np

from benchmark.round3 import common
from benchmark.e3 import extract as X
from benchmark.e3 import events_matched as EM
from benchmark.e3.stats import boot_ci, summarise, paired, _boot_idx

E3 = os.path.join(common.BENCH, "results", "E3")
CTRL = ("C1", "C3", "C5")
LABEL = {"C1": "Unified RL (Ours)", "C3": "RL+IK", "C5": "Cooperative RL"}
SEEDS = list(range(100))
CONDS = [(4, 0), (1, 0), (2, 0), (3, 0)]
SIG = {"ee_pos": X.SIGS.index("ee_pos"), "ee_ori": X.SIGS.index("ee_ori")}
PRE, POST, DT = X.PRE, X.POST, X.DT
T = np.arange(-PRE, POST) * DT


def _job(job):
    (c, bi, ti, s, kind), evs = EM._episode(job)
    out = {}
    for k, (met, tr) in evs.items():
        tr = tr.astype(float)
        d = tr - tr[:, :PRE].mean(axis=1, keepdims=True)
        out[k] = {"raw": tr, "d": d, "dpos": np.clip(d, 0.0, None),
                  "off_peak_pos": {g: met[f"{g}_peak_pos_excess"] for g in SIG},
                  "off_int_pos": {g: met[f"{g}_int_pos_excess"] for g in SIG}}
    return job, out


def _curve_max_boot(curves):
    """curves[ctrl, seed, time] -> per ctrl (max_t mean, lo, hi) and paired rival-ours diffs."""
    idx = _boot_idx(curves.shape[1])
    resp = T >= 0
    mx = np.stack([curves[ci][idx].mean(axis=1)[:, resp].max(axis=1) for ci in range(3)])  # [ctrl, boot]
    pt = curves.mean(axis=1)[:, resp].max(axis=1)
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))  # noqa: E731
    return pt, [q(mx[ci]) for ci in range(3)], {r: (float(pt[r] - pt[0]), *q(mx[r] - mx[0])) for r in (1, 2)}


def run():
    jobs = [(c, bi, ti, s, "base") for (bi, ti) in CONDS for c in CTRL for s in SEEDS]
    with mp.get_context("fork").Pool(12) as pool:
        res = dict(pool.map(_job, jobs, chunksize=8))
    Mtr = np.load(os.path.join(E3, "data", "E3_event_traces_matched.npz"))
    off = {}
    with open(os.path.join(E3, "stats", "E3_events_matched_per_controller.csv")) as f:
        for r in csv.DictReader(f):
            off[(r["condition"], r["metric"], r["controller"])] = float(r["mean"])
    arrays, stat_rows, curve_rows, checks = {}, [], [], {}
    for (bi, ti) in CONDS:
        tag = f"B{bi}T{ti}"
        P = np.full((3, 100, 2, X.TRACE_LEN), np.nan)          # matched mean positive excess
        S = np.full_like(P, np.nan)                            # matched mean signed excess (per-event baseline)
        R = np.full_like(P, np.nan)                            # matched mean raw (absolute) error
        PU = np.full_like(P, np.nan)                           # unmatched mean positive excess
        PK = np.full((3, 100, 2), np.nan)                      # mean over events of max_t de^+ (recomputed)
        OPK, OINT = np.full_like(PK, np.nan), np.full_like(PK, np.nan)
        nm = np.zeros(100, int)
        for si, s in enumerate(SEEDS):
            ev = [res[(c, bi, ti, s, "base")] for c in CTRL]
            for ci in range(3):
                if ev[ci]:
                    PU[ci, si] = np.mean([e["dpos"][[SIG["ee_pos"], SIG["ee_ori"]]] for e in ev[ci].values()], axis=0)
            ticks = sorted(set(ev[0]) & set(ev[1]) & set(ev[2]))
            nm[si] = len(ticks)
            if not ticks:
                continue
            for ci in range(3):
                sel = [ev[ci][k] for k in ticks]
                gi = [SIG["ee_pos"], SIG["ee_ori"]]
                P[ci, si] = np.mean([e["dpos"][gi] for e in sel], axis=0)
                S[ci, si] = np.mean([e["d"][gi] for e in sel], axis=0)
                R[ci, si] = np.mean([e["raw"][gi] for e in sel], axis=0)
                PK[ci, si] = [np.mean([e["dpos"][SIG[g], PRE:].max() for e in sel]) for g in SIG]
                OPK[ci, si] = [np.mean([e["off_peak_pos"][g] for e in sel]) for g in SIG]
                OINT[ci, si] = [np.mean([e["off_int_pos"][g] for e in sel]) for g in SIG]
        keep = nm > 0
        # ---- consistency checks against validated outputs
        stored = Mtr[f"trace_{tag}"][:, :, [SIG["ee_pos"], SIG["ee_ori"]]]
        stored_ex = stored - stored[..., :PRE].mean(axis=-1, keepdims=True)
        ch = {"matched_seeds": int(keep.sum()), "matched_events": int(nm.sum()),
              "max_abs_diff_raw_vs_stored_trace": float(np.nanmax(np.abs(R[:, keep] - stored[:, keep]))),
              "max_abs_diff_signed_vs_stored_excess": float(np.nanmax(np.abs(S[:, keep] - stored_ex[:, keep]))),
              "max_abs_diff_peak_recomputed_vs_extract": float(np.nanmax(np.abs(PK[:, keep] - OPK[:, keep]))),
              "max_abs_diff_meanpos_vs_int_pos/1.5": float(np.nanmax(np.abs(
                  P[:, keep][..., PRE:].mean(axis=-1) - OINT[:, keep] / (POST * DT))))}
        for gi, g in enumerate(SIG):
            for ci, c in enumerate(CTRL):
                ch[f"official_{g}_peak_pos_excess_{c}"] = off[(tag, f"{g}_peak_pos_excess", c)]
                ch[f"recomputed_{g}_peak_pos_excess_{c}"] = float(np.nanmean(PK[ci, keep, gi]))
        checks[tag] = ch
        arrays |= {f"pos_{tag}": P, f"signed_{tag}": S, f"abs_{tag}": R, f"pos_unmatched_{tag}": PU,
                   f"n_matched_{tag}": nm}
        # ---- statistics (seed level, paired)
        for gi, g in enumerate(SIG):
            Pk = P[:, keep, gi]
            pt, cis, dif = _curve_max_boot(Pk)
            per_peak = PK[:, keep, gi]
            meanpos = Pk[..., PRE:].mean(axis=-1)
            q = {"max_t_of_mean_pos_excess_curve": None,
                 "mean_per_event_peak_pos_excess": per_peak,
                 "mean_pos_excess_0_1p5s": meanpos}
            for qn, v in q.items():
                for ci, c in enumerate(CTRL):
                    row = {"condition": tag, "signal": g, "quantity": qn, "controller": LABEL[c],
                           "controller_code": c, "n_seeds": int(keep.sum())}
                    if v is None:
                        row |= {"mean": float(pt[ci]), "ci95_lo": cis[ci][0], "ci95_hi": cis[ci][1]}
                    else:
                        sm = summarise(v[ci])
                        row |= {"mean": sm["mean"], "ci95_lo": sm["ci95_lo"], "ci95_hi": sm["ci95_hi"]}
                    stat_rows.append(row | {"kind": "per_controller"})
                for r in (1, 2):
                    row = {"condition": tag, "signal": g, "quantity": qn,
                           "controller": f"{LABEL[CTRL[r]]} minus {LABEL['C1']}",
                           "controller_code": f"{CTRL[r]}-C1", "n_seeds": int(keep.sum()), "kind": "paired_diff"}
                    if v is None:
                        row |= {"mean": dif[r][0], "ci95_lo": dif[r][1], "ci95_hi": dif[r][2],
                                "note": "descriptive: paired seed bootstrap of the difference of curve maxima"}
                    else:
                        pr = paired(v[0], v[r])
                        row |= {"mean": pr["mean_diff"], "ci95_lo": pr["ci95_lo"], "ci95_hi": pr["ci95_hi"],
                                "p": pr["p"], "rel_diff": pr["rel_diff"],
                                "note": ("formal metric (official Holm verdict in E3_events_matched_paired.csv: "
                                         f"{g}_{'peak' if 'peak' in qn else 'int'}_pos_excess)")}
                    stat_rows.append(row)
        if tag == "B4T0":
            for gi, g in enumerate(SIG):
                m = P[:, keep, gi].mean(axis=1)
                mu = np.stack([np.nanmean(PU[ci, np.isfinite(PU[ci, :, gi]).all(-1), gi], axis=0) for ci in range(3)])
                for ci, c in enumerate(CTRL):
                    for i, t in enumerate(T):
                        curve_rows.append({"signal": g, "controller": LABEL[c], "controller_code": c,
                                           "t_from_switch_s": round(float(t), 2), "mean_pos_excess_matched": float(m[ci, i]),
                                           "mean_pos_excess_unmatched": float(mu[ci, i]),
                                           "mean_signed_excess_matched": float(S[ci, keep, gi, i].mean())})
    np.savez_compressed(os.path.join(E3, "data", "E3_base_event_positive_excess_traces.npz"), **arrays,
                        trace_t=T, signals=np.array(list(SIG)), controllers=np.array(CTRL))
    common.write_csv(os.path.join(E3, "stats", "E3_base_event_positive_excess.csv"), stat_rows)
    common.write_csv(os.path.join(E3, "stats", "E3_base_event_positive_excess_curves_B4T0.csv"), curve_rows)
    with open(os.path.join(E3, "validation", "positive_excess_checks.json"), "w") as f:
        json.dump(checks, f, indent=1)
    print(json.dumps(checks["B4T0"], indent=1))


if __name__ == "__main__":
    run()
