#!/usr/bin/env python3
"""E3 (bidirectional 7 x 7) analysis: validation, per-trial table, statistics.
Definitions: docs/E3_PROTOCOL.md sections 5-6. Reads only saved episodes.

    python3 -m benchmark.e3.analyze [--jobs 12] [--raw DIR] [--out DIR]

Outputs (default results/E3/):
  data/E3_per_trial_14700.csv      one row per episode (all 49 conditions x 3 controllers x 100 seeds)
  data/E3_arrays.npz               metric arrays [controller, B, T, seed]
  stats/E3_summary.csv             per controller x condition x metric: n valid / completed /
                                   terminated / missing, mean, SD, 95 % bootstrap CI, median
  stats/E3_summary_7x7.csv         machine-readable 7 x 7 table, all four metrics, one row per
                                   controller x condition
  stats/E3_paired_vs_C1.csv        C3-C1 and C5-C1 per condition; Holm/BH within metric x rival
  stats/E3_slice_trends.csv        per figure slice: change from the lowest to each level, per
                                   controller, and rival-minus-C1 differences of that change
  stats/E3_sensitivity_completed_only.csv
  stats/E3_completion.csv
  validation/validation_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np

from benchmark.round3 import common
from benchmark.e3 import extract as X
from benchmark.e3.stats import wilson, summarise, paired, holm, finish_family
from benchmark.scenarios import experiments as E

E3 = os.path.join(common.BENCH, "results", "E3")
SCHED = os.path.join(E3, "protocol", "E3_command_schedules.npz")
CTRL = ("C1", "C3", "C5")
CFG, _G = E.load_config("E3")
BLEV = [float(b) for b in CFG["base"]["levels"]]
HZ = E.e3_update_hz(CFG)
HZLAB = [str(x) for x in CFG["ee"]["update_hz_labels"]]
SEEDS = list(range(CFG["official_seeds"]["start"], CFG["official_seeds"]["start"] + CFG["official_seeds"]["count"]))
NB, NT, NS = len(BLEV), len(HZ), len(SEEDS)
N_EP = len(CTRL) * NB * NT * NS
N_EVAL = E.n_eval_ticks(CFG)
RMSE = list(X.RMSE_KEYS)
UNITS = {"base_xy_rmse_mps": "m/s", "yaw_rmse_radps": "rad/s", "ee_pos_rmse_m": "m", "ee_ori_rmse_rad": "rad"}
SCALARS = RMSE + ["survived", "valid_eval_s", "n_ee_switches_evaluated"]
SLICES = {"vs_base_bound": E.e3_slice(CFG, "vs_base_bound"), "vs_ee_update": E.e3_slice(CFG, "vs_ee_update")}
RAW = os.path.join(E3, "raw", "E3")


def cid(bi, ti):
    return f"B{bi}T{ti}"


def idx(c):
    return int(c[1]), int(c[3])


def _job(args):
    raw, c, bi, ti, s = args
    d = os.path.join(raw, c, cid(bi, ti), f"seed_{s:03d}")
    o = X.extract(d, base_bound=BLEV[bi])
    o.update(controller=c, bi=bi, ti=ti, seed=s)
    return o


def main(argv=None):
    global E3
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--raw", default=RAW, help="episode root containing C1/C3/C5")
    ap.add_argument("--out", default=E3, help="output root (data/, stats/, validation/)")
    ap.add_argument("--seeds", default=None, help="'a-b' subset (smoke tests only)")
    a = ap.parse_args(argv)
    E3 = a.out
    seeds = SEEDS if a.seeds is None else list(range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1))
    for sub in ("data", "stats", "validation"):
        os.makedirs(os.path.join(E3, sub), exist_ok=True)
    jobs = [(a.raw, c, bi, ti, s) for c in CTRL for bi in range(NB) for ti in range(NT) for s in seeds]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        eps = pool.map(_job, jobs, chunksize=10)
    print(f"loaded {len(eps)} episodes")
    F = validate(eps, seeds)
    per_trial(eps)
    A = arrays(eps)
    summary(A)
    stats_paired(A)
    slice_trends(A)
    completion(A)
    print(f"validation failures: {len(F)}")
    return 1 if F else 0


# ------------------------------------------------------------------ validation
def _sha16(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def validate(eps, seeds):
    F, chk = [], {}
    probs = [e for e in eps if e["problems"]]
    F += [f"{e['path']}: {e['problems']}" for e in probs[:20]]
    chk["episodes_expected"] = len(CTRL) * NB * NT * len(seeds)
    chk["episodes_loaded"], chk["episodes_with_problems"] = len(eps), len(probs)
    ok = [e for e in eps if not e["problems"]]
    keys = {(e["controller"], e["bi"], e["ti"], e["seed"]) for e in eps}
    if len(keys) != chk["episodes_expected"] or len(eps) != chk["episodes_expected"]:
        F.append(f"episode count {len(eps)} / unique {len(keys)}")
    # frozen schedules (written before the run)
    S = np.load(SCHED)
    with open(SCHED + ".sha256") as h:
        want = h.read().split()[0]
    with open(SCHED, "rb") as h:
        got = hashlib.sha256(h.read()).hexdigest()
    chk["frozen_schedule_sha256"], chk["frozen_schedule_sha256_ok"] = got, got == want
    if got != want:
        F.append("frozen schedule file changed since the freeze")
    sidx = {int(s): i for i, s in enumerate(S["seeds"])}
    hb = {(bi, s): _sha16(S["base_commands_by_B"][bi, sidx[s]]) for bi in range(NB) for s in seeds}
    he = {(ti, s): _sha16(S["ee_commands_by_T"][ti, sidx[s]]) for ti in range(NT) for s in seeds}
    nfz = 0
    for e in ok:
        if e["base_hash"] != hb[(e["bi"], e["seed"])] or e["ee_hash"] != he[(e["ti"], e["seed"])]:
            nfz += 1
    chk["episodes_with_commands_differing_from_frozen_schedule"] = nfz
    if nfz:
        F.append(f"{nfz} episodes whose stored schedule differs from the frozen schedule")
    by = {(e["controller"], e["bi"], e["ti"], e["seed"]): e for e in ok}
    for bi, ti, s in itertools.product(range(NB), range(NT), seeds):
        g = [by.get((c, bi, ti, s)) for c in CTRL]
        if any(x is None for x in g):
            continue
        for k in ("base_hash", "ee_hash", "init_hash"):
            if len({x[k] for x in g}) != 1:
                F.append(f"pairing {k} differs across controllers at {cid(bi, ti)} seed {s}")
    for s in seeds:
        if len({e["init_hash"] for e in ok if e["seed"] == s}) != 1:
            F.append(f"initial state differs across conditions, seed {s}")
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
        if e["scenario"].get("protocol") != E.E3_PROTOCOL:
            F.append(f"protocol tag {e['path']}")
    chk.update({f"max_{k}": v for k, v in w.items()})
    if w["rmse_vs_json"] > 1e-6 or w["theta_vs_logged_deg"] > 1e-3 or w["issued_vs_scheduled"] > 1e-6:
        F.append(f"metric/command recomputation mismatch {dict(w)}")
    term = [e for e in ok if not e["survived"]]
    chk["terminated"] = len(term)
    chk["terminations"] = sorted(f"{e['controller']}/{cid(e['bi'], e['ti'])}/seed_{e['seed']:03d}:"
                                 f"{e['termination_reason']}@{e['valid_eval_s']:.2f}s" for e in term)
    chk["failures"], chk["passed"] = F, not F
    json.dump(chk, open(os.path.join(E3, "validation", "validation_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in chk.items() if k not in ("failures", "terminations")}, indent=1))
    print("terminations:", chk["terminations"][:50])
    for f in F[:20]:
        print("FAIL", f)
    return F


# ------------------------------------------------------------------ tables
def _flat(e):
    if e["problems"] or e["n_eval"] == 0:
        return {}
    return e["rec"] | {"survived": e["survived"], "valid_eval_s": e["valid_eval_s"],
                       "n_ee_switches_evaluated": e["n_ee_switches_evaluated"]}


def per_trial(eps):
    rows = []
    for e in sorted(eps, key=lambda e: (e["controller"], e["bi"], e["ti"], e["seed"])):
        r = {"controller": e["controller"], "condition": cid(e["bi"], e["ti"]), "base_bound": BLEV[e["bi"]],
             "ee_update_hz": HZ[e["ti"]], "ee_update_label": HZLAB[e["ti"]], "seed": e["seed"]}
        if e["problems"]:
            r["problems"] = ";".join(e["problems"])
        else:
            r.update(termination_reason=e["termination_reason"], n_eval_ticks=e["n_eval"],
                     pose_rows_used=" ".join(map(str, e["scenario"]["ee_pose_table_rows_used"])),
                     base_schedule_sha16=e["base_hash"], ee_schedule_sha16=e["ee_hash"],
                     initial_state_sha16=e["init_hash"])
            r.update(_flat(e))
        rows.append(r)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    common.write_csv(os.path.join(E3, "data", f"E3_per_trial_{len(rows)}.csv"), rows, cols)


def arrays(eps):
    A = {k: np.full((len(CTRL), NB, NT, NS), np.nan) for k in SCALARS}
    for e in eps:
        i = (CTRL.index(e["controller"]), e["bi"], e["ti"], SEEDS.index(e["seed"]))
        for k, v in _flat(e).items():
            A[k][i] = v
    np.savez_compressed(os.path.join(E3, "data", "E3_arrays.npz"), **A, controllers=np.array(CTRL),
                        base_levels=np.array(BLEV), ee_update_hz=np.array(HZ), ee_update_labels=np.array(HZLAB),
                        seeds=np.array(SEEDS))
    return A


def _counts(A, ci, bi, ti):
    surv = A["survived"][ci, bi, ti]
    valid = int(np.isfinite(surv).sum())
    comp = int(np.nansum(surv))
    return {"n_trials": NS, "n_valid": valid, "n_completed": comp, "n_terminated_early": valid - comp,
            "n_missing_or_corrupt": NS - valid}


def summary(A):
    rows, wide = [], []
    for ci, c in enumerate(CTRL):
        for bi, ti in itertools.product(range(NB), range(NT)):
            n = _counts(A, ci, bi, ti)
            wl, wh = wilson(n["n_completed"], n["n_valid"])
            base = {"controller": c, "condition": cid(bi, ti), "base_bound": BLEV[bi], "ee_update_hz": HZ[ti],
                    "ee_update_label": HZLAB[ti]} | n | {
                "completion_rate": n["n_completed"] / n["n_valid"] if n["n_valid"] else float("nan"),
                "completion_wilson_lo": wl, "completion_wilson_hi": wh,
                "mean_effective_eval_s": float(np.nanmean(A["valid_eval_s"][ci, bi, ti]))}
            wr = dict(base)
            for m in RMSE:
                s = summarise(A[m][ci, bi, ti])
                rows.append(base | {"metric": m, "unit": UNITS[m], "n_scored": s.pop("n")} | s)
                wr.update({f"{m}_{k}": v for k, v in s.items() if k != "median"})
            wide.append(wr)
    common.write_csv(os.path.join(E3, "stats", "E3_summary.csv"), rows)
    common.write_csv(os.path.join(E3, "stats", "E3_summary_7x7.csv"), wide)


def stats_paired(A):
    P, sens, ch = [], [], 0
    for m, riv in itertools.product(RMSE, ("C3", "C5")):
        fa, fb = [], []
        for bi, ti in itertools.product(range(NB), range(NT)):
            a, b = A[m][0, bi, ti], A[m][CTRL.index(riv), bi, ti]
            base = {"metric": m, "comparison": f"{riv}-C1", "rival": riv, "condition": cid(bi, ti),
                    "base_bound": BLEV[bi], "ee_update_hz": HZ[ti], "ee_update_label": HZLAB[ti],
                    "family": f"{m}|{riv}-C1 (49 conditions)"}
            fa.append(base | paired(a, b))
            surv = np.all(A["survived"][:, bi, ti] == 1, axis=0)
            fb.append(base | {"n_completed_seeds": int(surv.sum())} |
                      paired(np.where(surv, a, np.nan), np.where(surv, b, np.nan)))
        P += finish_family(fa)
        finish_family(fb)
        for x, y in zip(fa, fb):
            ch += x["verdict"] != y["verdict"]
            sens.append(y | {"verdict_all": x["verdict"], "verdict_completed_only": y["verdict"],
                             "changed": int(x["verdict"] != y["verdict"])})
    common.write_csv(os.path.join(E3, "stats", "E3_paired_vs_C1.csv"), P)
    common.write_csv(os.path.join(E3, "stats", "E3_sensitivity_completed_only.csv"), sens)
    print(f"paired {len(P)}, completed-only verdict changes {ch}/{len(sens)}")


def slice_trends(A):
    """Within each figure slice: change of each RMSE from the slice's lowest level (B = 0 or
    Fixed) to every other level, per controller (paired by seed; Holm within slice x metric x
    controller), and the rival-minus-C1 difference of that change (Holm within slice x metric x
    rival)."""
    rows = []
    for name, cells in SLICES.items():
        ref = idx(cells[0])
        for m in RMSE:
            per_c = {c: [] for c in CTRL}
            dd = {r: [] for r in ("C3", "C5")}
            for cell in cells[1:]:
                bi, ti = idx(cell)
                deg = {}
                for ci, c in enumerate(CTRL):
                    x = A[m][ci]
                    deg[c] = x[bi, ti] - x[ref]
                    per_c[c].append({"slice": name, "metric": m, "kind": "change_from_reference",
                                     "controller": c, "condition": cell, "reference": cells[0],
                                     "level": BLEV[bi] if name == "vs_base_bound" else HZLAB[ti]}
                                    | paired(x[ref], x[bi, ti]))
                for r in dd:
                    dd[r].append({"slice": name, "metric": m, "kind": "rival_change_minus_C1_change",
                                  "controller": f"{r}-C1", "rival": r, "condition": cell, "reference": cells[0],
                                  "level": BLEV[bi] if name == "vs_base_bound" else HZLAB[ti]}
                                 | paired(deg["C1"], deg[r]))
            for g in per_c.values():
                for r, h in zip(g, holm([r["p"] for r in g])):
                    r["p_holm"], r["significant_holm"] = float(h), int(np.isfinite(h) and h < 0.05)
                rows += g
            for r in dd:
                rows += finish_family(dd[r])
    common.write_csv(os.path.join(E3, "stats", "E3_slice_trends.csv"), rows)


def completion(A):
    comp = []
    for ci, c in enumerate(CTRL):
        s = A["survived"][ci]
        comp.append({"controller": c, "episodes_expected": NB * NT * NS, "episodes_valid": int(np.isfinite(s).sum()),
                     "completed": int(np.nansum(s)), "terminated_early": int(np.isfinite(s).sum() - np.nansum(s)),
                     "missing_or_corrupt": int(NB * NT * NS - np.isfinite(s).sum())}
                    | {f"completion_{cid(bi, ti)}": float(np.nanmean(s[bi, ti])) if np.isfinite(s[bi, ti]).any()
                       else float("nan") for bi in range(NB) for ti in range(NT)})
    common.write_csv(os.path.join(E3, "stats", "E3_completion.csv"), comp)


if __name__ == "__main__":
    raise SystemExit(main())
