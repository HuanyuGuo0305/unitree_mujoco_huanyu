#!/usr/bin/env python3
"""E3 v3 analysis: validation, per-trial table, statistics (pre-registered, section 6-7).

    python3 -m benchmark.e3v3.analyze [--jobs 12]

Outputs under results/E3_v3/: data/, stats/, validation/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np
from scipy import stats as sps

from benchmark.round3 import common
from benchmark.core import scenario as sc
from benchmark.e3v3 import extract as X
from benchmark.scenarios import experiments as E

V3 = os.path.join(common.BENCH, "results", "E3_v3")
RAW = os.path.join(V3, "raw", "E3")
CTRL = ("C1", "C3", "C5")
PATTERNS = ("line", "circle")
PAT_ALL = ("line", "circle", "pooled")
CFG, _G = E.load_config("E3")
BLEV = [float(b) for b in CFG["base"]["levels"]]
TLAB = list(CFG["ee"]["level_labels"])
TINT = CFG["ee"]["update_intervals_s"]
SEEDS = list(range(CFG["official_seeds"]["start"], CFG["official_seeds"]["start"] + CFG["official_seeds"]["count"]))
N_EVAL = E.n_eval_ticks(CFG)
NB, NT, NS = len(BLEV), len(TLAB), len(SEEDS)
BOOT_N, BOOT_SEED, CONF = 10_000, 20261106, 0.95
PRACTICAL = 0.10

RMSE = ["base_xy_rmse_mps", "yaw_rmse_radps", "ee_pos_rmse_m", "ee_ori_rmse_rad"]
UNITS = {"base_xy_rmse_mps": "m/s", "yaw_rmse_radps": "rad/s", "ee_pos_rmse_m": "m",
         "ee_ori_rmse_rad": "rad"}
EVENT = [f"ee_iso_{s}_{q}" for s in ("base_xy", "yaw")
         for q in ("peak_excess", "int_excess", "recovery_s", "recovery_censored")] + \
        [f"base_iso_{s}_{q}" for s in ("ee_pos", "ee_ori")
         for q in ("peak_excess", "int_excess", "recovery_s", "recovery_censored")] + \
        [f"ee_ovl_{s}_{q}" for s in ("base_xy", "yaw") for q in ("peak_excess", "int_excess")] + \
        [f"base_ovl_{s}_{q}" for s in ("ee_pos", "ee_ori") for q in ("peak_excess", "int_excess")]
COUNTS = ["n_ee_iso", "n_ee_ovl", "n_base_iso", "n_base_ovl", "n_ee_events_truncated",
          "n_base_events_truncated"]
REACH = ["reach_rate_primary", "reach_rate_strict", "rmst_primary_s", "rmst_strict_s",
         "median_time_reached_primary_s", "n_targets", "reach_primary", "reach_strict"]
S5 = [f"s5_{m}" for m in RMSE]
SCALARS = RMSE + S5 + EVENT + REACH + ["survived", "valid_eval_s"]


def cond_id(bi, ti, pat):
    return f"B{bi}T{ti}_{pat}"


def _job(args):
    c, bi, ti, pat, seed = args
    d = os.path.join(RAW, c, cond_id(bi, ti, pat), f"seed_{seed:03d}")
    o = X.extract(d, base_bound=BLEV[bi])
    o.update(controller=c, bi=bi, ti=ti, pattern=pat, seed=seed)
    return o


# ------------------------------------------------------------------ statistics
_IDX = {}


def _boot_idx(n):
    if n not in _IDX:
        _IDX[n] = np.random.default_rng(BOOT_SEED).integers(0, n, size=(BOOT_N, n))
    return _IDX[n]


def boot_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), float("nan")
    m = x[_boot_idx(x.size)].mean(1)
    a = (1 - CONF) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def wilson(k, n, conf=CONF):
    if n == 0:
        return float("nan"), float("nan")
    z = sps.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def summarise(x):
    x = np.asarray(x, float)
    f = x[np.isfinite(x)]
    lo, hi = boot_ci(f)
    return {"n": int(f.size), "mean": float(f.mean()) if f.size else float("nan"),
            "sd": float(f.std(ddof=1)) if f.size > 1 else float("nan"),
            "ci95_lo": lo, "ci95_hi": hi,
            "median": float(np.median(f)) if f.size else float("nan")}


def paired(ref, riv):
    ref, riv = np.asarray(ref, float), np.asarray(riv, float)
    ok = np.isfinite(ref) & np.isfinite(riv)
    d = riv[ok] - ref[ok]
    n = int(d.size)
    out = {"n_pairs": n}
    if n < 2:
        return out | {"mean_diff": float("nan"), "p": float("nan")}
    lo, hi = boot_ci(d)
    sd = float(d.std(ddof=1))
    rm = float(ref[ok].mean())
    p = float(sps.ttest_1samp(d, 0.0).pvalue) if sd > 0 else (1.0 if d.mean() == 0 else 0.0)
    return out | {"ref_mean": rm, "rival_mean": float(riv[ok].mean()), "mean_diff": float(d.mean()),
                  "sd_diff": sd, "ci95_lo": lo, "ci95_hi": hi, "p": p,
                  "rel_diff": float(d.mean() / rm) if rm != 0 else float("nan"),
                  "cohen_dz": float(d.mean() / sd) if sd > 0 else float("nan"),
                  "frac_rival_lower": float(np.mean(d < 0))}


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    m = q.size
    adj = np.maximum.accumulate((m - np.arange(m)) * q[order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    m = q.size
    order = np.argsort(q)
    adj = q[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def finish_family(rows, lower_is_better=True):
    p = [r.get("p", float("nan")) for r in rows]
    h, q = holm(p), bh(p)
    for r, hh, qq in zip(rows, h, q):
        r["p_holm"], r["q_bh"] = float(hh), float(qq)
        sig = np.isfinite(hh) and hh < 0.05
        r["significant_holm"] = int(sig)
        r["practical"] = int(np.isfinite(r.get("rel_diff", np.nan)) and abs(r["rel_diff"]) >= PRACTICAL)
        if not sig:
            r["verdict"] = "no significant difference"
        else:
            rival_lower = r["mean_diff"] < 0
            better = rival_lower if lower_is_better else not rival_lower
            r["verdict"] = f"{r['rival']} better" if better else "C1 better"
    return rows


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args(argv)
    for sub in ("data", "stats", "validation"):
        os.makedirs(os.path.join(V3, sub), exist_ok=True)
    jobs = [(c, bi, ti, pat, s) for c in CTRL for bi in range(NB) for ti in range(NT)
            for pat in PATTERNS for s in SEEDS]
    with mp.get_context("fork").Pool(args.jobs) as pool:
        eps = pool.map(_job, jobs, chunksize=20)
    print(f"loaded {len(eps)} episodes")
    fails = validate(eps)
    trials = per_trial(eps)
    A = arrays(eps)
    stats_all(A)
    sensitivity_completed(A)
    sensitivity_window(A)
    print(f"validation failures: {len(fails)}")
    return 1 if fails else 0


# ------------------------------------------------------------------ validation
def validate(eps):
    F = []
    checks = {}
    probs = [e for e in eps if e["problems"]]
    for e in probs[:20]:
        F.append(f"{e['path']}: {e['problems']}")
    checks["episodes_loaded"] = len(eps)
    checks["episodes_with_problems"] = len(probs)
    ok = [e for e in eps if not e["problems"]]
    keys = {(e["controller"], e["bi"], e["ti"], e["pattern"], e["seed"]) for e in eps}
    if len(keys) != len(eps) or len(eps) != 3 * NB * NT * 2 * NS:
        F.append(f"episode count {len(eps)} / unique {len(keys)} != {3 * NB * NT * 2 * NS}")
    by = {(e["controller"], e["bi"], e["ti"], e["pattern"], e["seed"]): e for e in ok}
    # pairing across controllers; initial state per seed
    n_pair = 0
    for bi, ti, pat, s in itertools.product(range(NB), range(NT), PATTERNS, SEEDS):
        grp = [by.get((c, bi, ti, pat, s)) for c in CTRL]
        if any(g is None for g in grp):
            continue
        n_pair += 1
        for k in ("base_hash", "ee_hash", "init_hash"):
            if len({g[k] for g in grp}) != 1:
                F.append(f"pairing {k} differs across controllers at {cond_id(bi, ti, pat)} seed {s}")
    for s in SEEDS:
        if len({e["init_hash"] for e in ok if e["seed"] == s}) != 1:
            F.append(f"initial state differs across conditions for seed {s}")
        for bi in range(NB):
            if len({e["base_hash"] for e in ok if e["seed"] == s and e["bi"] == bi}) != 1:
                F.append(f"base schedule differs across EE levels/patterns (B{bi}, seed {s})")
        for ti, pat in itertools.product(range(NT), PATTERNS):
            if len({e["ee_hash"] for e in ok if e["seed"] == s and e["ti"] == ti
                    and e["pattern"] == pat}) != 1:
                F.append(f"EE schedule differs across base levels (T{ti} {pat}, seed {s})")
    checks["paired_groups_checked"] = n_pair
    # protocol metadata per controller
    for c in CTRL:
        h = {e["proto_hash"] for e in ok if e["controller"] == c}
        checks[f"protocol_metadata_sets_{c}"] = len(h)
        if len(h) != 1:
            F.append(f"{c}: {len(h)} distinct protocol metadata sets")
    # per-episode integrity
    worst = defaultdict(float)
    for e in ok:
        if e["meta_seed"] != e["seed"]:
            F.append(f"seed mismatch {e['path']}")
        st, d = sc.e3v3_start_direction(e["seed"])
        scn = e["scenario"]
        if scn["ee_start_waypoint"] != st or scn["ee_direction"] != d:
            F.append(f"start/direction mismatch {e['path']}")
        if e["survived"] and e["n_eval"] != N_EVAL:
            F.append(f"completed episode with {e['n_eval']} ticks: {e['path']}")
        if (not e["survived"]) and (not e["termination_reason"] or e["termination_reason"] == "completed"):
            F.append(f"terminated without reason: {e['path']}")
        if e["metrics_json_n"] != e["n_eval"]:
            F.append(f"metrics.json n mismatch {e['path']}")
        if e["n_eval"] == 0:
            continue
        for k in RMSE:
            worst["rmse_vs_json"] = max(worst["rmse_vs_json"], abs(e["rec"][k] - e["json"][k]))
        worst["theta_vs_logged_deg"] = max(worst["theta_vs_logged_deg"], e["theta_vs_logged_deg"])
        worst["issued_vs_scheduled"] = max(worst["issued_vs_scheduled"], e["issued_vs_scheduled"])
        if not (e["settle_base_zero"] and e["settle_ee_constant"]):
            F.append(f"settle commands not held: {e['path']}")
        if not e["base_bound_ok"]:
            F.append(f"base command outside bound: {e['path']}")
    checks.update({f"max_{k}": v for k, v in worst.items()})
    if worst["rmse_vs_json"] > 1e-6:
        F.append(f"RMSE recomputation differs by {worst['rmse_vs_json']}")
    if worst["theta_vs_logged_deg"] > 1e-3:
        F.append("geodesic angle differs from logged value")
    if worst["issued_vs_scheduled"] > 1e-6:
        F.append("issued commands differ from schedule")
    term = [e for e in ok if not e["survived"]]
    checks["terminated_episodes"] = len(term)
    checks["termination_reasons"] = dict(sorted(
        {r: sum(1 for e in term if e["termination_reason"] == r)
         for r in {e["termination_reason"] for e in term}}.items()))
    checks["failures"] = F
    checks["passed"] = not F
    with open(os.path.join(V3, "validation", "validation_summary.json"), "w") as h:
        json.dump(checks, h, indent=1)
    print(json.dumps({k: v for k, v in checks.items() if k != "failures"}, indent=1))
    for f in F[:20]:
        print("FAIL", f)
    return F


# ------------------------------------------------------------------ tables
def per_trial(eps):
    rows = []
    for e in sorted(eps, key=lambda e: (e["controller"], e["pattern"], e["bi"], e["ti"], e["seed"])):
        r = {"controller": e["controller"], "pattern": e["pattern"], "condition":
             cond_id(e["bi"], e["ti"], e["pattern"]), "base_level_index": e["bi"],
             "base_bound": BLEV[e["bi"]], "ee_level_index": e["ti"], "ee_level": TLAB[e["ti"]],
             "ee_update_interval_s": TINT[e["ti"]] if TINT[e["ti"]] is not None else "",
             "seed": e["seed"]}
        if e["problems"]:
            r["problems"] = ";".join(e["problems"])
            rows.append(r)
            continue
        scn = e["scenario"]
        r.update({"ee_start_waypoint": scn["ee_start_waypoint"], "ee_direction": scn["ee_direction"],
                  "survived": e["survived"], "termination_reason": e["termination_reason"],
                  "valid_eval_s": e["valid_eval_s"], "n_eval_ticks": e["n_eval"]})
        r.update(e.get("rec", {}))
        r.update({f"s5_{k}": v for k, v in (e.get("rec_from5s") or {}).items()})
        r.update(e.get("events", {}))
        r.update(e.get("reach") or {})
        rows.append(r)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    common.write_csv(os.path.join(V3, "data", "E3v3_per_trial_15000.csv"), rows, cols)
    return rows


def arrays(eps):
    """metric -> array [ctrl, pattern(line, circle), B, T, seed] (NaN if absent)."""
    A = {k: np.full((3, 2, NB, NT, NS), np.nan) for k in SCALARS + COUNTS}
    TR = {k: np.full((3, 2, NB, NT, NS, 4, X.TRACE_LEN), np.nan, dtype=np.float32)
          for k in ("ee_iso", "base_iso")}
    for e in eps:
        if e["problems"] or e["n_eval"] == 0:
            continue
        i = (CTRL.index(e["controller"]), PATTERNS.index(e["pattern"]), e["bi"], e["ti"],
             SEEDS.index(e["seed"]))
        flat = dict(e["rec"]) | {f"s5_{k}": v for k, v in (e.get("rec_from5s") or {}).items()} | dict(e["events"]) | dict(e["reach"] or {}) | {
            "survived": e["survived"], "valid_eval_s": e["valid_eval_s"]}
        for k in A:
            if k in flat and flat[k] is not None:
                A[k][i] = flat[k]
        for k in TR:
            if e["traces"][k] is not None:
                TR[k][i] = e["traces"][k]
    np.savez_compressed(os.path.join(V3, "data", "E3v3_arrays.npz"),
                        **A, **{f"trace_{k}": v for k, v in TR.items()},
                        controllers=np.array(CTRL), patterns=np.array(PATTERNS),
                        base_levels=np.array(BLEV), ee_levels=np.array(TLAB), seeds=np.array(SEEDS),
                        trace_signals=np.array(["base_xy", "yaw", "ee_pos", "ee_ori"]),
                        trace_t=np.arange(-X.PRE, X.POST) * X.DT)
    return A


def _pat(A, k, ci, pat):
    """[B, T, seed] for a pattern, or the per-seed mean of line and circle."""
    if pat == "pooled":
        return np.mean(A[k][ci], axis=0)  # NaN if either is NaN (pooled requires both)
    return A[k][ci, PATTERNS.index(pat)]


def stats_all(A):
    S = os.path.join(V3, "stats")
    metrics = RMSE + S5 + EVENT + ["reach_rate_primary", "reach_rate_strict", "rmst_primary_s",
                              "rmst_strict_s", "median_time_reached_primary_s"]
    # ---- summary
    rows = []
    for ci, c in enumerate(CTRL):
        for pat in PAT_ALL:
            for bi, ti in itertools.product(range(NB), range(NT)):
                surv = _pat(A, "survived", ci, pat)[bi, ti] if pat != "pooled" else \
                    A["survived"][ci, :, bi, ti].reshape(-1)
                k = int(np.nansum(surv))
                n = int(np.isfinite(surv).sum())
                wl, wh = wilson(k, n)
                ve = _pat(A, "valid_eval_s", ci, pat)[bi, ti]
                base = {"controller": c, "pattern": pat, "base_bound": BLEV[bi],
                        "ee_level": TLAB[ti], "condition": f"B{bi}T{ti}",
                        "completion_rate": k / n if n else float("nan"),
                        "completion_wilson_lo": wl, "completion_wilson_hi": wh,
                        "n_episodes": n, "mean_effective_eval_s": float(np.nanmean(ve))}
                for m in metrics:
                    x = _pat(A, m, ci, pat)[bi, ti]
                    if not np.isfinite(x).any():
                        continue
                    rows.append(base | {"metric": m, "unit": UNITS.get(m, "")} | summarise(x))
    common.write_csv(os.path.join(S, "E3v3_summary.csv"), rows)
    print(f"summary rows {len(rows)}")
    # ---- paired comparisons: RMSE metrics, families = metric x comparison x pattern
    P = []
    for m, pat, riv in itertools.product(RMSE, PAT_ALL, ("C3", "C5")):
        fam = []
        for bi, ti in itertools.product(range(NB), range(NT)):
            r = paired(_pat(A, m, 0, pat)[bi, ti], _pat(A, m, CTRL.index(riv), pat)[bi, ti])
            fam.append({"metric": m, "pattern": pat, "comparison": f"{riv}-C1", "rival": riv,
                        "base_bound": BLEV[bi], "ee_level": TLAB[ti], "condition": f"B{bi}T{ti}",
                        "family": f"{m}|{riv}-C1|{pat}"} | r)
        P += finish_family(fam)
    common.write_csv(os.path.join(S, "E3v3_paired_rmse.csv"), P)
    # ---- degradation per controller and C1-vs-rival on degradation
    D, DC = [], []
    for m, pat in itertools.product(RMSE, PAT_ALL):
        is_ee = m.startswith("ee_")
        kind = "base_to_ee" if is_ee else "ee_to_base"
        famc = {riv: [] for riv in ("C3", "C5")}
        for bi, ti in itertools.product(range(NB), range(NT)):
            if (is_ee and bi == 0) or ((not is_ee) and ti == 0):
                continue
            ref_b, ref_t = (0, ti) if is_ee else (bi, 0)
            deg = {}
            for ci, c in enumerate(CTRL):
                x = _pat(A, m, ci, pat)
                dd = x[bi, ti] - x[ref_b, ref_t]
                deg[c] = dd
                refm = float(np.nanmean(x[ref_b, ref_t]))
                pr = paired(x[ref_b, ref_t], x[bi, ti])
                D.append({"direction": kind, "metric": m, "pattern": pat, "controller": c,
                          "base_bound": BLEV[bi], "ee_level": TLAB[ti],
                          "reference": f"B{ref_b}T{ref_t}", "abs_at_condition": float(np.nanmean(x[bi, ti])),
                          "abs_at_reference": refm,
                          "rel_degradation": pr.get("rel_diff", float("nan")),
                          "p": pr.get("p", float("nan"))} | summarise(dd))
            for riv in ("C3", "C5"):
                r = paired(deg["C1"], deg[riv])
                famc[riv].append({"direction": kind, "metric": m, "pattern": pat,
                                  "comparison": f"deg({riv})-deg(C1)", "rival": riv,
                                  "base_bound": BLEV[bi], "ee_level": TLAB[ti]} | r)
        for riv in famc:
            DC += finish_family(famc[riv])
    # Holm within (direction, metric, pattern, controller) for per-controller degradation tests
    grp = defaultdict(list)
    for r in D:
        grp[(r["metric"], r["pattern"], r["controller"])].append(r)
    for g in grp.values():
        for r, h in zip(g, holm([r["p"] for r in g])):
            r["p_holm"] = float(h)
            r["significant_holm"] = int(np.isfinite(h) and h < 0.05)
    common.write_csv(os.path.join(S, "E3v3_degradation.csv"), D)
    common.write_csv(os.path.join(S, "E3v3_degradation_C1_vs_rivals.csv"), DC)
    # ---- curvature / shape: per-seed quadratic fit of RMSE vs level (pooled)
    CV = []
    for m, ci in itertools.product(RMSE, range(3)):
        x = _pat(A, m, ci, "pooled")
        for sweep in ("base", "ee"):
            for fixed in range(NT if sweep == "base" else NB):
                Y = x[:, fixed, :] if sweep == "base" else x[fixed, :, :]
                lv = np.array(BLEV) if sweep == "base" else np.array(
                    [0.0] + [1.0 / t for t in TINT[1:]])  # EE frequency [Hz]
                ok = np.isfinite(Y).all(0)
                co = np.polyfit(lv, Y[:, ok], 2)
                lin = np.polyfit(lv, Y[:, ok], 1)
                r2 = {"quadratic": co[0], "linear_slope": lin[0]}
                CV.append({"metric": m, "controller": CTRL[ci], "sweep": sweep,
                           "fixed_level": (TLAB[fixed] if sweep == "base" else BLEV[fixed]),
                           "x_unit": "B" if sweep == "base" else "Hz",
                           **{f"{k}_{q}": v for k, arr in r2.items()
                              for q, v in zip(("mean", "ci95_lo", "ci95_hi"),
                                              [float(arr.mean()), *boot_ci(arr)])}})
    common.write_csv(os.path.join(S, "E3v3_curvature.csv"), CV)
    # ---- event and reach paired comparisons (main slices + all cells)
    EV = []
    ev_metrics = [m for m in EVENT if "recovery_censored" not in m] + [
        "reach_rate_primary", "reach_rate_strict", "rmst_primary_s", "rmst_strict_s"]
    for m, pat, riv in itertools.product(ev_metrics, PAT_ALL, ("C3", "C5")):
        fam = []
        for bi, ti in itertools.product(range(NB), range(NT)):
            a, b = _pat(A, m, 0, pat)[bi, ti], _pat(A, m, CTRL.index(riv), pat)[bi, ti]
            if np.isfinite(a).sum() < 10:
                continue
            r = paired(a, b)
            fam.append({"metric": m, "pattern": pat, "comparison": f"{riv}-C1", "rival": riv,
                        "base_bound": BLEV[bi], "ee_level": TLAB[ti]} | r)
        if fam:
            P2 = finish_family(fam, lower_is_better=not m.startswith("reach_rate"))
            EV += P2
    common.write_csv(os.path.join(S, "E3v3_paired_events_reach.csv"), EV)
    print(f"paired rmse {len(P)}, degradation {len(D)}, deg-vs-rival {len(DC)}, events {len(EV)}")


def sensitivity_completed(A):
    """Re-run the pooled RMSE comparisons on seeds where all three controllers
    completed both patterns of the cell; count verdict changes."""
    rows, changes = [], 0
    for m, riv in itertools.product(RMSE, ("C3", "C5")):
        fam_a, fam_b = [], []
        for bi, ti in itertools.product(range(NB), range(NT)):
            surv = np.all(A["survived"][:, :, bi, ti, :] == 1, axis=(0, 1))
            ref, rv = _pat(A, m, 0, "pooled")[bi, ti], _pat(A, m, CTRL.index(riv), "pooled")[bi, ti]
            base = {"metric": m, "comparison": f"{riv}-C1", "rival": riv,
                    "base_bound": BLEV[bi], "ee_level": TLAB[ti]}
            fam_a.append(base | paired(ref, rv))
            fam_b.append(base | {"n_completed_seeds": int(surv.sum())} |
                         paired(np.where(surv, ref, np.nan), np.where(surv, rv, np.nan)))
        finish_family(fam_a)
        finish_family(fam_b)
        for a, b in zip(fam_a, fam_b):
            ch = a["verdict"] != b["verdict"]
            changes += ch
            rows.append(b | {"verdict_all": a["verdict"], "verdict_completed_only": b["verdict"],
                             "changed": int(ch)})
    common.write_csv(os.path.join(V3, "stats", "E3v3_sensitivity_completed_only.csv"), rows)
    print(f"completed-only sensitivity: {changes}/{len(rows)} verdict changes")


def sensitivity_window(A):
    """Post-hoc: pooled RMSE verdicts on the 5-30 s window vs the official 3-30 s."""
    rows, changes, reversals = [], 0, 0
    for m, riv in itertools.product(RMSE, ("C3", "C5")):
        fa, fb = [], []
        for bi, ti in itertools.product(range(NB), range(NT)):
            base = {"metric": m, "comparison": f"{riv}-C1", "rival": riv,
                    "base_bound": BLEV[bi], "ee_level": TLAB[ti]}
            fa.append(base | paired(_pat(A, m, 0, "pooled")[bi, ti],
                                    _pat(A, m, CTRL.index(riv), "pooled")[bi, ti]))
            fb.append(base | paired(_pat(A, f"s5_{m}", 0, "pooled")[bi, ti],
                                    _pat(A, f"s5_{m}", CTRL.index(riv), "pooled")[bi, ti]))
        finish_family(fa)
        finish_family(fb)
        for a, b in zip(fa, fb):
            ch = a["verdict"] != b["verdict"]
            rev = ch and "better" in a["verdict"] and "better" in b["verdict"]
            changes += ch
            reversals += rev
            rows.append(b | {"verdict_3_30s": a["verdict"], "verdict_5_30s": b["verdict"],
                             "rel_diff_3_30s": a.get("rel_diff"), "changed": int(ch),
                             "reversed": int(rev)})
    common.write_csv(os.path.join(V3, "stats", "E3v3_sensitivity_window_5_30s_posthoc.csv"), rows)
    print(f"window 5-30 s (post hoc): {changes}/{len(rows)} verdict changes, {reversals} reversals")


if __name__ == "__main__":
    raise SystemExit(main())
