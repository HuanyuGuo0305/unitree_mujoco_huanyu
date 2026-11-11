#!/usr/bin/env python3
"""E3 analysis: validation, per-trial table, statistics (results/E3/E3_PROTOCOL.md 5-7).

    python3 -m benchmark.e3.analyze [--jobs 12]
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np

from benchmark.round3 import common
from benchmark.e3 import extract as X
from benchmark.e3.stats import boot_ci, wilson, summarise, paired, holm, bh, finish_family
from benchmark.scenarios import experiments as E

E3 = os.path.join(common.BENCH, "results", "E3")
RAW = os.path.join(E3, "raw", "E3")
CTRL = ("C1", "C3", "C5")
CFG, _G = E.load_config("E3")
BLEV = [float(b) for b in CFG["base"]["levels"]]
TLAB = list(CFG["ee"]["level_labels"])
TINT = CFG["ee"]["update_intervals_s"]
FREQ = [0.0] + [1.0 / t for t in TINT[1:]]
SEEDS = list(range(CFG["official_seeds"]["start"], CFG["official_seeds"]["start"] + CFG["official_seeds"]["count"]))
NB, NT, NS = len(BLEV), len(TLAB), len(SEEDS)
N_EVAL = E.n_eval_ticks(CFG)
BINS = json.load(open(os.path.join(E3, "protocol", "analysis_bins.json")))
MAIN_EE, MAIN_BASE = (0, 4), (4, 0)
RMSE = ["base_xy_rmse_mps", "yaw_rmse_radps", "ee_pos_rmse_m", "ee_ori_rmse_rad"]
UNITS = {"base_xy_rmse_mps": "m/s", "yaw_rmse_radps": "rad/s", "ee_pos_rmse_m": "m", "ee_ori_rmse_rad": "rad"}
REACH = ["reach_rate_primary", "reach_rate_strict", "rmst_primary_s", "rmst_strict_s",
         "n_targets", "mean_jump_pos_m", "mean_jump_ori_rad"]
EV_Q = ["baseline_abs", "peak_excess", "peak_pos_excess", "int_excess", "int_pos_excess",
        "post_abs_mean", "residual_abs", "recovery_s", "recovery_censored", "recovery_defined",
        "peak_pos_excess_b025", "int_pos_excess_b025", "peak_pos_excess_b100", "int_pos_excess_b100"]
EVENT_KEYS = [f"{g}_{s}_{q}" for g in ("ee_iso", "ee_ovl", "base_iso", "base_ovl")
              for s in X.SIGS for q in EV_Q] + [f"{g}_jump_pos_m" for g in ("ee_iso", "ee_ovl")] + \
             [f"{g}_jump_ori_rad" for g in ("ee_iso", "ee_ovl")]
COUNTS = ["n_ee_iso", "n_ee_ovl", "n_base_iso", "n_base_ovl", "n_ee_events_truncated", "n_base_events_truncated"]
SCALARS = RMSE + REACH + ["survived", "valid_eval_s", "ee_pos_err_t3", "ee_ori_err_t3",
                          "pose0_reached_primary_in_first_hold"] + EVENT_KEYS + COUNTS


def cid(bi, ti):
    return f"B{bi}T{ti}"


def _job(args):
    c, bi, ti, s = args
    d = os.path.join(RAW, c, cid(bi, ti), f"seed_{s:03d}")
    o = X.extract(d, base_bound=BLEV[bi], keep_event_rows=(bi, ti) in (MAIN_EE, MAIN_BASE))
    o.update(controller=c, bi=bi, ti=ti, seed=s)
    o.pop("traces_full", None)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args(argv)
    for sub in ("data", "stats", "validation"):
        os.makedirs(os.path.join(E3, sub), exist_ok=True)
    jobs = [(c, bi, ti, s) for c in CTRL for bi in range(NB) for ti in range(NT) for s in SEEDS]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        eps = pool.map(_job, jobs, chunksize=10)
    print(f"loaded {len(eps)} episodes")
    F = validate(eps)
    per_trial(eps)
    A, TR = arrays(eps)
    stats_rmse(A)
    stats_events(A, eps)
    stats_misc(A, eps)
    print(f"validation failures: {len(F)}")
    return 1 if F else 0


# ------------------------------------------------------------------ validation
def validate(eps):
    F, chk = [], {}
    probs = [e for e in eps if e["problems"]]
    F += [f"{e['path']}: {e['problems']}" for e in probs[:20]]
    chk["episodes_loaded"], chk["episodes_with_problems"] = len(eps), len(probs)
    ok = [e for e in eps if not e["problems"]]
    keys = {(e["controller"], e["bi"], e["ti"], e["seed"]) for e in eps}
    if len(keys) != 3 * NB * NT * NS or len(eps) != 3 * NB * NT * NS:
        F.append(f"episode count {len(eps)} / unique {len(keys)}")
    by = {(e["controller"], e["bi"], e["ti"], e["seed"]): e for e in ok}
    for bi, ti, s in itertools.product(range(NB), range(NT), SEEDS):
        g = [by.get((c, bi, ti, s)) for c in CTRL]
        if any(x is None for x in g):
            continue
        for k in ("base_hash", "ee_hash", "init_hash"):
            if len({x[k] for x in g}) != 1:
                F.append(f"pairing {k} differs across controllers at {cid(bi, ti)} seed {s}")
    for s in SEEDS:
        if len({e["init_hash"] for e in ok if e["seed"] == s}) != 1:
            F.append(f"initial state differs across conditions, seed {s}")
        for bi in range(NB):
            if len({e["base_hash"] for e in ok if e["seed"] == s and e["bi"] == bi}) != 1:
                F.append(f"base schedule differs across EE levels (B{bi}, seed {s})")
        for ti in range(NT):
            if len({e["ee_hash"] for e in ok if e["seed"] == s and e["ti"] == ti}) != 1:
                F.append(f"EE schedule differs across base levels (T{ti}, seed {s})")
        rows = [by[("C1", 0, ti, s)]["scenario"]["ee_pose_table_rows_used"] for ti in range(NT)]
        longest = max(rows, key=len)
        if any(r != longest[:len(r)] for r in rows):
            F.append(f"EE levels are not prefixes of one sequence, seed {s}")
    for c in CTRL:
        h = {e["proto_hash"] for e in ok if e["controller"] == c}
        chk[f"protocol_metadata_sets_{c}"] = len(h)
        if len(h) != 1:
            F.append(f"{c}: {len(h)} protocol metadata sets")
    w = defaultdict(float)
    for e in ok:
        if e["meta_seed"] != e["seed"]:
            F.append(f"seed mismatch {e['path']}")
        if e["survived"] and e["n_eval"] != N_EVAL:
            F.append(f"completed episode with {e['n_eval']} ticks {e['path']}")
        if not e["survived"] and e["termination_reason"] in ("", "completed"):
            F.append(f"terminated without reason {e['path']}")
        if e["metrics_json_n"] != e["n_eval"]:
            F.append(f"n mismatch {e['path']}")
        if e["n_eval"] == 0:
            continue
        w["rmse_vs_json"] = max(w["rmse_vs_json"], max(abs(e["rec"][k] - e["json"][k]) for k in RMSE))
        w["theta_vs_logged_deg"] = max(w["theta_vs_logged_deg"], e["theta_vs_logged_deg"])
        w["issued_vs_scheduled"] = max(w["issued_vs_scheduled"], e["issued_vs_scheduled"])
        if not (e["settle_base_zero"] and e["settle_ee_is_first_target"]):
            F.append(f"settle commands wrong {e['path']}")
        if not e["base_bound_ok"]:
            F.append(f"base bound {e['path']}")
    chk.update({f"max_{k}": v for k, v in w.items()})
    if w["rmse_vs_json"] > 1e-6 or w["theta_vs_logged_deg"] > 1e-3 or w["issued_vs_scheduled"] > 1e-6:
        F.append(f"metric/command recomputation mismatch {dict(w)}")
    term = [e for e in ok if not e["survived"]]
    chk["terminated"] = len(term)
    chk["terminations"] = sorted(f"{e['controller']}/{cid(e['bi'], e['ti'])}/seed_{e['seed']:03d}:"
                                 f"{e['termination_reason']}@{e['valid_eval_s']:.2f}s" for e in term)
    # event coverage at the main conditions
    for name, (bi, ti), g in (("ee_main", MAIN_EE, "n_ee_iso"), ("base_main", MAIN_BASE, "n_base_iso")):
        n = [by[(c, bi, ti, s)]["events"].get(g, 0) for c in CTRL for s in SEEDS]
        chk[f"{name}_isolated_events_per_episode_min_mean"] = [int(min(n)), float(np.mean(n))]
    # matched event ticks across controllers (completed episodes)
    mism = 0
    for (bi, ti) in (MAIN_EE, MAIN_BASE):
        for s in SEEDS:
            g = [by[(c, bi, ti, s)] for c in CTRL]
            if all(x["survived"] for x in g):
                t = [[(r["kind"], r["tick"], r["isolated"]) for r in x["event_rows"]] for x in g]
                mism += int(not (t[0] == t[1] == t[2]))
    chk["main_condition_event_tick_mismatches"] = mism
    if mism:
        F.append(f"{mism} main-condition episodes with different event ticks across controllers")
    chk["failures"], chk["passed"] = F, not F
    json.dump(chk, open(os.path.join(E3, "validation", "validation_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in chk.items() if k not in ("failures", "terminations")}, indent=1))
    print("terminations:", chk["terminations"])
    for f in F[:20]:
        print("FAIL", f)
    return F


def _flat(e):
    f = {}
    if e["problems"] or e["n_eval"] == 0:
        return f
    f.update(e["rec"])
    f.update(e["events"])
    f.update(e["reach"] or {})
    f.update(survived=e["survived"], valid_eval_s=e["valid_eval_s"], ee_pos_err_t3=e["ee_err_at_t3"]["pos"],
             ee_ori_err_t3=e["ee_err_at_t3"]["ori"],
             pose0_reached_primary_in_first_hold=e["pose0_reached_primary_in_first_hold"])
    return f


def per_trial(eps):
    rows = []
    for e in sorted(eps, key=lambda e: (e["controller"], e["bi"], e["ti"], e["seed"])):
        r = {"controller": e["controller"], "condition": cid(e["bi"], e["ti"]), "base_bound": BLEV[e["bi"]],
             "ee_level": TLAB[e["ti"]], "ee_update_hz": FREQ[e["ti"]], "seed": e["seed"]}
        if e["problems"]:
            r["problems"] = ";".join(e["problems"])
        else:
            r.update(termination_reason=e["termination_reason"], n_eval_ticks=e["n_eval"],
                     pose_rows_used=" ".join(map(str, e["scenario"]["ee_pose_table_rows_used"])))
            r.update(_flat(e))
        rows.append(r)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    common.write_csv(os.path.join(E3, "data", "E3_per_trial_7500.csv"), rows, cols)


def arrays(eps):
    A = {k: np.full((3, NB, NT, NS), np.nan) for k in SCALARS}
    TR = {k: np.full((3, NB, NT, NS, 4, X.TRACE_LEN), np.nan, dtype=np.float32) for k in ("ee_iso", "base_iso")}
    for e in eps:
        i = (CTRL.index(e["controller"]), e["bi"], e["ti"], SEEDS.index(e["seed"]))
        for k, v in _flat(e).items():
            if k in A and v is not None:
                A[k][i] = v
        if not e["problems"] and e["n_eval"]:
            for k in TR:
                if e["traces"][k] is not None:
                    TR[k][i] = e["traces"][k]
    np.savez_compressed(os.path.join(E3, "data", "E3_arrays.npz"), **A,
                        **{f"trace_{k}": v for k, v in TR.items()}, controllers=np.array(CTRL),
                        base_levels=np.array(BLEV), ee_levels=np.array(TLAB), ee_freq_hz=np.array(FREQ),
                        seeds=np.array(SEEDS), trace_signals=np.array(X.SIGS),
                        trace_t=np.arange(-X.PRE, X.POST) * X.DT)
    ev = []
    for e in eps:
        for r in e.get("event_rows", []):
            ev.append({"controller": e["controller"], "condition": cid(e["bi"], e["ti"]), "seed": e["seed"]} | r)
    common.write_csv(os.path.join(E3, "data", "E3_event_rows_main_conditions.csv"), ev)
    return A, TR


# ------------------------------------------------------------------ statistics
def stats_rmse(A):
    S = os.path.join(E3, "stats")
    rows = []
    summ_metrics = RMSE + REACH + ["ee_pos_err_t3", "ee_ori_err_t3", "pose0_reached_primary_in_first_hold"]
    for ci, c in enumerate(CTRL):
        for bi, ti in itertools.product(range(NB), range(NT)):
            surv = A["survived"][ci, bi, ti]
            k, n = int(np.nansum(surv)), int(np.isfinite(surv).sum())
            wl, wh = wilson(k, n)
            base = {"controller": c, "condition": cid(bi, ti), "base_bound": BLEV[bi], "ee_level": TLAB[ti],
                    "ee_update_hz": FREQ[ti], "completion_rate": k / n, "completion_wilson_lo": wl,
                    "completion_wilson_hi": wh, "n_episodes": n,
                    "mean_effective_eval_s": float(np.nanmean(A["valid_eval_s"][ci, bi, ti]))}
            for m in summ_metrics:
                x = A[m][ci, bi, ti]
                if np.isfinite(x).any():
                    rows.append(base | {"metric": m, "unit": UNITS.get(m, "")} | summarise(x))
    common.write_csv(os.path.join(S, "E3_summary.csv"), rows)
    P = []
    for m, riv in itertools.product(RMSE + ["reach_rate_primary", "reach_rate_strict", "rmst_primary_s",
                                            "rmst_strict_s"], ("C3", "C5")):
        fam = []
        for bi, ti in itertools.product(range(NB), range(NT)):
            a, b = A[m][0, bi, ti], A[m][CTRL.index(riv), bi, ti]
            if np.isfinite(a).sum() < 10:
                continue
            fam.append({"metric": m, "comparison": f"{riv}-C1", "rival": riv, "condition": cid(bi, ti),
                        "base_bound": BLEV[bi], "ee_level": TLAB[ti], "family": f"{m}|{riv}-C1"} | paired(a, b))
        P += finish_family(fam, lower_is_better=not m.startswith("reach_rate"))
    common.write_csv(os.path.join(S, "E3_paired_tracking_and_reaching.csv"), P)
    D, DC = [], []
    for m in RMSE:
        is_ee = m.startswith("ee_")
        famc = {r: [] for r in ("C3", "C5")}
        for bi, ti in itertools.product(range(NB), range(NT)):
            if (is_ee and bi == 0) or (not is_ee and ti == 0):
                continue
            rb, rt = (0, ti) if is_ee else (bi, 0)
            deg = {}
            for ci, c in enumerate(CTRL):
                x = A[m][ci]
                deg[c] = x[bi, ti] - x[rb, rt]
                pr = paired(x[rb, rt], x[bi, ti])
                D.append({"direction": "base_to_ee" if is_ee else "ee_to_base", "metric": m, "controller": c,
                          "condition": cid(bi, ti), "base_bound": BLEV[bi], "ee_level": TLAB[ti],
                          "reference": cid(rb, rt), "abs_at_condition": float(np.nanmean(x[bi, ti])),
                          "abs_at_reference": float(np.nanmean(x[rb, rt])), "rel_change": pr.get("rel_diff"),
                          "p": pr.get("p")} | summarise(deg[c]))
            for r in ("C3", "C5"):
                famc[r].append({"direction": "base_to_ee" if is_ee else "ee_to_base", "metric": m,
                                "comparison": f"deg({r})-deg(C1)", "rival": r, "condition": cid(bi, ti),
                                "base_bound": BLEV[bi], "ee_level": TLAB[ti]} | paired(deg["C1"], deg[r]))
        for r in famc:
            DC += finish_family(famc[r])
    grp = defaultdict(list)
    for r in D:
        grp[(r["metric"], r["controller"])].append(r)
    for g in grp.values():
        for r, h in zip(g, holm([r["p"] for r in g])):
            r["p_holm"], r["significant_holm"] = float(h), int(np.isfinite(h) and h < 0.05)
    common.write_csv(os.path.join(S, "E3_degradation.csv"), D)
    common.write_csv(os.path.join(S, "E3_degradation_C1_vs_rivals.csv"), DC)
    # completed-only sensitivity
    rows, ch = [], 0
    for m, riv in itertools.product(RMSE, ("C3", "C5")):
        fa, fb = [], []
        for bi, ti in itertools.product(range(NB), range(NT)):
            surv = np.all(A["survived"][:, bi, ti] == 1, axis=0)
            a, b = A[m][0, bi, ti], A[m][CTRL.index(riv), bi, ti]
            base = {"metric": m, "comparison": f"{riv}-C1", "rival": riv, "condition": cid(bi, ti)}
            fa.append(base | paired(a, b))
            fb.append(base | {"n_completed_seeds": int(surv.sum())} |
                      paired(np.where(surv, a, np.nan), np.where(surv, b, np.nan)))
        finish_family(fa)
        finish_family(fb)
        for x, y in zip(fa, fb):
            ch += x["verdict"] != y["verdict"]
            rows.append(y | {"verdict_all": x["verdict"], "verdict_completed_only": y["verdict"],
                             "changed": int(x["verdict"] != y["verdict"])})
    common.write_csv(os.path.join(S, "E3_sensitivity_completed_only.csv"), rows)
    print(f"RMSE/reach paired {len(P)}, degradation {len(D)}/{len(DC)}, completed-only changes {ch}/{len(rows)}")


EV_MAIN = {
    "ee_to_base_B0_T2s": ("ee_iso", MAIN_EE, [("base_xy", q) for q in ("peak_pos_excess", "int_pos_excess", "recovery_s")]
                          + [("yaw", q) for q in ("peak_pos_excess", "int_pos_excess", "recovery_s")]
                          + [("ee_pos", "post_abs_mean"), ("ee_pos", "residual_abs"),
                             ("ee_ori", "post_abs_mean"), ("ee_ori", "residual_abs")]),
    "base_to_ee_Fixed_B060": ("base_iso", MAIN_BASE, [("ee_pos", q) for q in ("peak_pos_excess", "int_pos_excess", "recovery_s")]
                              + [("ee_ori", q) for q in ("peak_pos_excess", "int_pos_excess", "recovery_s")]
                              + [("ee_pos", "baseline_abs"), ("ee_ori", "baseline_abs")]),
}
DESCR = ["baseline_abs", "peak_excess", "int_excess", "recovery_censored", "recovery_defined",
         "peak_pos_excess_b025", "int_pos_excess_b025", "peak_pos_excess_b100", "int_pos_excess_b100"]


def stats_events(A, eps):
    S = os.path.join(E3, "stats")
    pc, pp = [], []
    for fam, (g, (bi, ti), mets) in EV_MAIN.items():
        sigs = sorted({s for s, _ in mets})
        allm = list(dict.fromkeys(mets + [(s, q) for s in sigs for q in DESCR]))
        rows = []
        for s, q in allm:
            key = f"{g}_{s}_{q}"
            x = A[key][:, bi, ti]
            for ci, c in enumerate(CTRL):
                pc.append({"family": fam, "event": g, "condition": cid(bi, ti), "metric": f"{s}_{q}",
                           "controller": c, "formal": int((s, q) in mets)} | summarise(x[ci]))
            for r in ("C3", "C5"):
                row = {"family": fam, "event": g, "condition": cid(bi, ti), "metric": f"{s}_{q}",
                       "comparison": f"{r}-C1", "rival": r, "formal": int((s, q) in mets)} | \
                      paired(x[0], x[CTRL.index(r)])
                rows.append(row)
        formal = [r for r in rows if r["formal"]]
        finish_family(formal)
        for r in rows:
            if not r["formal"]:
                r["verdict"] = "descriptive (not in the formal family)"
        pp += rows
    common.write_csv(os.path.join(S, "E3_events_main_per_controller.csv"), pc)
    common.write_csv(os.path.join(S, "E3_events_main_paired.csv"), pp)
    # every cell: isolated and overlapping (descriptive) + isolated paired sweeps
    rows, sweep = [], []
    for g, s, q in itertools.product(("ee_iso", "ee_ovl", "base_iso", "base_ovl"), X.SIGS,
                                     ("peak_pos_excess", "int_pos_excess", "baseline_abs", "post_abs_mean")):
        key = f"{g}_{s}_{q}"
        for ci, c in enumerate(CTRL):
            for bi, ti in itertools.product(range(NB), range(NT)):
                x = A[key][ci, bi, ti]
                if np.isfinite(x).sum() >= 1:
                    cnt = A[f"n_{g}"][ci, bi, ti]
                    rows.append({"event": g, "signal": s, "metric": q, "controller": c, "condition": cid(bi, ti),
                                 "base_bound": BLEV[bi], "ee_level": TLAB[ti],
                                 "episodes_with_events": int(np.isfinite(x).sum()),
                                 "events_per_episode_mean": float(np.nanmean(cnt))} | summarise(x))
    common.write_csv(os.path.join(S, "E3_events_all_cells.csv"), rows)
    for (g, sigs, cells, tag) in (("ee_iso", ("base_xy", "yaw"), [(0, ti) for ti in range(1, NT)], "ee_to_base_B0_by_T"),
                                  ("base_iso", ("ee_pos", "ee_ori"), [(bi, 0) for bi in range(1, NB)], "base_to_ee_Fixed_by_B")):
        for s, r in itertools.product(sigs, ("C3", "C5")):
            fam = []
            for bi, ti in cells:
                for q in ("peak_pos_excess", "int_pos_excess"):
                    x = A[f"{g}_{s}_{q}"][:, bi, ti]
                    fam.append({"sweep": tag, "metric": f"{s}_{q}", "comparison": f"{r}-C1", "rival": r,
                                "condition": cid(bi, ti)} | paired(x[0], x[CTRL.index(r)]))
            sweep += finish_family(fam)
    common.write_csv(os.path.join(S, "E3_events_level_sweeps_paired.csv"), sweep)
    # baseline-window sensitivity at the main conditions
    bs = []
    for fam, (g, (bi, ti), mets) in EV_MAIN.items():
        sigs = ("base_xy", "yaw") if g == "ee_iso" else ("ee_pos", "ee_ori")
        for s, q, w, r in itertools.product(sigs, ("peak_pos_excess", "int_pos_excess"), ("", "_b025", "_b100"),
                                            ("C3", "C5")):
            x = A[f"{g}_{s}_{q}{w}"][:, bi, ti]
            bs.append({"family": fam, "metric": f"{s}_{q}", "baseline_window": {"": "[-0.5,0) s (registered)",
                       "_b025": "[-0.24,0) s", "_b100": "[-1.0,0) s"}[w], "comparison": f"{r}-C1"}
                      | paired(x[0], x[CTRL.index(r)]))
    common.write_csv(os.path.join(S, "E3_events_baseline_window_sensitivity.csv"), bs)
    # jump-magnitude bins (EE events at the main EE condition)
    ev = [r for e in eps for r in e.get("event_rows", []) if r["kind"] == "ee" and (e["bi"], e["ti"]) == MAIN_EE
          for r in [dict(r, controller=e["controller"], seed=e["seed"])]]
    jb = []
    for jk, edges in (("jump_pos_m", BINS["pos_jump_tertile_edges_m"]), ("jump_ori_rad", BINS["ori_jump_tertile_edges_rad"])):
        for b in range(3):
            lo = -np.inf if b == 0 else edges[b - 1]
            hi = np.inf if b == 2 else edges[b]
            for m in ("base_xy_peak_pos_excess", "base_xy_int_pos_excess", "yaw_peak_pos_excess",
                      "yaw_int_pos_excess", "ee_pos_residual_abs", "ee_ori_residual_abs"):
                # per-seed means
                acc = defaultdict(list)
                for r in ev:
                    if r["isolated"] and lo <= r[jk] < hi:
                        acc[(r["controller"], r["seed"])].append(r[m])
                X3 = np.full((3, NS), np.nan)
                for (c, s), v in acc.items():
                    X3[CTRL.index(c), SEEDS.index(s)] = np.mean(v)
                for ci, c in enumerate(CTRL):
                    jb.append({"jump": jk, "bin": b, "bin_range": f"[{lo:.3g}, {hi:.3g})", "metric": m,
                               "controller": c, "comparison": ""} | summarise(X3[ci]))
                for rv in ("C3", "C5"):
                    jb.append({"jump": jk, "bin": b, "bin_range": f"[{lo:.3g}, {hi:.3g})", "metric": m,
                               "controller": "", "comparison": f"{rv}-C1"} | paired(X3[0], X3[CTRL.index(rv)]))
    common.write_csv(os.path.join(S, "E3_events_jump_bins.csv"), jb)
    # within-seed slope of peak excess vs jump (descriptive)
    sl = []
    for c in CTRL:
        for jk in ("jump_pos_m", "jump_ori_rad"):
            for m in ("base_xy_peak_pos_excess", "yaw_peak_pos_excess"):
                slopes = []
                for s in SEEDS:
                    rr = [r for r in ev if r["controller"] == c and r["seed"] == s and r["isolated"]]
                    if len(rr) >= 4:
                        x = np.array([r[jk] for r in rr])
                        y = np.array([r[m] for r in rr])
                        if np.ptp(x) > 0:
                            slopes.append(np.polyfit(x, y, 1)[0])
                lo_, hi_ = boot_ci(np.array(slopes))
                sl.append({"controller": c, "jump": jk, "metric": m, "n_seeds": len(slopes),
                           "mean_slope": float(np.mean(slopes)), "ci95_lo": lo_, "ci95_hi": hi_})
    common.write_csv(os.path.join(S, "E3_events_jump_slopes.csv"), sl)
    print(f"events: main {len(pp)}, cells {len(rows)}, sweeps {len(sweep)}, baseline {len(bs)}, jump bins {len(jb)}")


def stats_misc(A, eps):
    S = os.path.join(E3, "stats")
    # pose-0 height tertiles: base->EE consistency across sampled targets
    tab = E.keypoint_table(_G)
    z0 = np.array([tab.table[E.build_e3(CFG, "B0T0", s).ee.row_indices[0], 2] for s in SEEDS])
    ed = BINS["pose0_height_tertile_edges_m"]
    grp = np.digitize(z0, ed)
    rows = []
    bi, ti = MAIN_BASE
    for m in ("base_iso_ee_pos_peak_pos_excess", "base_iso_ee_pos_int_pos_excess", "base_iso_ee_ori_peak_pos_excess",
              "base_iso_ee_ori_int_pos_excess", "base_iso_ee_pos_baseline_abs", "base_iso_ee_ori_baseline_abs",
              "ee_pos_rmse_m", "ee_ori_rmse_rad"):
        for gi in range(3):
            sel = grp == gi
            x = A[m][:, bi, ti][:, sel]
            for r in ("C3", "C5"):
                rows.append({"metric": m, "pose0_height_tertile": gi, "n_seeds": int(sel.sum()),
                             "comparison": f"{r}-C1"} | paired(x[0], x[CTRL.index(r)]))
    common.write_csv(os.path.join(S, "E3_base_to_ee_by_pose0_height.csv"), rows)
    # completion and failure summary
    comp = []
    for ci, c in enumerate(CTRL):
        s = A["survived"][ci]
        comp.append({"controller": c, "episodes": int(np.isfinite(s).sum()), "completed": int(np.nansum(s)),
                     "terminated": int(np.isfinite(s).sum() - np.nansum(s))} |
                    {f"completion_{cid(bi, ti)}": float(np.nanmean(s[bi, ti])) for bi in range(NB) for ti in range(NT)})
    common.write_csv(os.path.join(S, "E3_completion.csv"), comp)
    # jump audit from the logged commands (C1 episodes; commands identical across controllers)
    ja = []
    for ti in range(1, NT):
        p = A["mean_jump_pos_m"][0, 0, ti]
        o = A["mean_jump_ori_rad"][0, 0, ti]
        n = A["n_targets"][0, 0, ti]
        ja.append({"ee_level": TLAB[ti], "targets_per_episode_mean": float(np.nanmean(n)),
                   "episode_mean_jump_pos_m_mean": float(np.nanmean(p)),
                   "episode_mean_jump_ori_rad_mean": float(np.nanmean(o))})
    common.write_csv(os.path.join(S, "E3_target_jump_audit.csv"), ja)


if __name__ == "__main__":
    raise SystemExit(main())
