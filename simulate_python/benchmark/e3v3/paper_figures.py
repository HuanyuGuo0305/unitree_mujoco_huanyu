#!/usr/bin/env python3
"""E3 v3 paper figures (four single-column RA-L figures) and their statistics.

    python3 -m benchmark.e3v3.paper_figures

Reads only existing data: the validated analysis arrays (official 3-30 s RMSEs)
and, for the event figures, the per-tick episode logs, from which every event
quantity is recomputed. No simulation is run and no official result is changed.

Fig. 1  concurrent tracking: base XY RMSE (a) and EE position RMSE (b) vs base
        bound B, EE command interval 2 s
Fig. 2  EE->base, frequency: base XY RMSE (a) and yaw-rate RMSE (b) vs EE update
        frequency, stationary base (B = 0)
Fig. 3  EE->base, events: signed excess base XY error (a) and absolute EE position
        error (b) around isolated EE-pose switches (B = 0, T = 2 s)
Fig. 4  base->EE, events: signed excess EE position (a) and orientation (b) error
        around isolated base-command changes (EE fixed, B = 0.60)

Aggregation (all figures): per seed, line and circle are averaged with equal
weight (event quantities: first averaged over the episode's events); 95 % CIs
are percentile bootstraps over the 100 seeds (10,000 resamples, seed 20261106)
with the same resampled seeds for every controller (paired). Event definitions
(baseline [-0.5, 0) s, response [0, 1.5) s, isolation, peak/integrated excess,
recovery) are the pre-registered ones; positive-part metrics are post hoc.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

from benchmark.round3 import common  # noqa: E402
from benchmark.plots import style  # noqa: E402
from benchmark.e3v3 import extract as X  # noqa: E402
from benchmark.e3v3.analyze import boot_ci, holm, bh  # noqa: E402

V3 = os.path.join(common.BENCH, "results", "E3_v3")
RAW = os.path.join(V3, "raw", "E3")
OUT = os.path.join(V3, "figures", "paper")
DATA = os.path.join(OUT, "data")
A = np.load(os.path.join(V3, "data", "E3v3_arrays.npz"))
CTRL = ("C1", "C3", "C5")
PATS = ("line", "circle")
NAME = {"C1": "C1 (Ours)", "C3": "C3 (RL+IK)", "C5": "C5 (Cooperative RL)"}
BLEV = A["base_levels"].astype(float)
TLAB = [str(t) for t in A["ee_levels"]]
FREQ = np.array([0.0, 1 / 6, 1 / 4, 1 / 3, 1 / 2])
FREQ_LAB = ["Fixed", "1/6", "1/4", "1/3", "1/2"]
SEEDS = [int(s) for s in A["seeds"]]
DT, PRE, POST = X.DT, X.PRE, X.POST
TT = np.arange(-PRE, POST) * DT
SIGS = ("base_xy", "yaw", "ee_pos", "ee_ori")
EE_EVENT = (0, 4)      # (B index, T index): B = 0, T = 2 s
BASE_EVENT = (4, 0)    # B = 0.60, EE fixed
BASE_SENS = [(1, 0), (2, 0), (3, 0)]
WIDTH, HEIGHT = 3.5, 2.7
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
      "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
      "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.titlesize": 7.5,
      "lines.linewidth": 1.1, "lines.markersize": 3.2, "axes.linewidth": 0.6,
      "grid.linewidth": 0.4, "savefig.bbox": None, "xtick.major.size": 2.5,
      "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
      "axes.labelpad": 1.5, "axes.titlepad": 2.0}


# ============================================================ event extraction
def _episode(job):
    """Recompute isolated-event quantities for one episode from its logs."""
    c, bi, ti, pat, seed, kind = job
    d = os.path.join(RAW, c, f"B{bi}T{ti}_{pat}", f"seed_{seed:03d}")
    cmd, st = X._read(os.path.join(d, "commands.csv")), X._read(os.path.join(d, "states.csv"))
    ec, es = cmd["evaluated"] == 1, st["evaluated"] == 1
    assert np.array_equal(cmd["tick"][ec], st["tick"][es])
    e = X.tick_errors(cmd, st, ec, es)
    n = int(es.sum())
    npz = np.load(os.path.join(d, "scenario_commands.npz"))
    sb = npz["base_commands"]
    ee_sw = [int(k) for k in npz["ee_switch_ticks"]]
    b_sw = [int(k) for k in npz["base_switch_ticks"] if int(k) > 0] if np.any(sb != 0) else []
    ee_ch = [0] + ee_sw
    b_ch = ([0] if np.any(sb[0] != 0) else []) + b_sw
    cand, others = (ee_sw, b_ch) if kind == "ee" else (b_sw, ee_ch)
    iso = [k for k in cand if k - PRE >= 0 and k + POST <= n
           and not any(k - X.ISO_PRE <= o < k + POST for o in others)]
    tol = X.REC_TOL
    raw = np.stack([np.stack([e[s][k - PRE:k + POST] for s in SIGS]) for k in iso])  # [E, 4, 100]
    base = raw[:, :, :PRE].mean(axis=2, keepdims=True)                              # [E, 4, 1]
    exc = raw - base
    w = exc[:, :, PRE:]
    met = {}
    for si, s in enumerate(SIGS):
        pk = w[:, si].max(axis=1)
        met[f"{s}_baseline_abs"] = base[:, si, 0]
        met[f"{s}_peak_excess"] = pk                                    # pre-registered
        met[f"{s}_int_excess"] = w[:, si].sum(axis=1) * DT              # pre-registered
        met[f"{s}_peak_pos_excess"] = np.maximum(pk, 0.0)               # post hoc
        met[f"{s}_int_pos_excess"] = np.clip(w[:, si], 0, None).sum(axis=1) * DT  # post hoc
        rec = np.array([X.event_response(e[s], k, tol[s]) for k in iso], dtype=object)
        met[f"{s}_recovery_s"] = np.array([r["recovery_s"] for r in rec])        # pre-registered
        met[f"{s}_recovery_censored"] = np.array([r["recovery_censored"] for r in rec])
        met[f"{s}_residual_abs"] = raw[:, si, PRE + 50:PRE + 75].mean(axis=1)    # 1.0-1.5 s
        met[f"{s}_peak_abs"] = raw[:, si, PRE:].max(axis=1)
    return {"job": job, "ticks": iso, "n_events": len(iso),
            "raw_mean": raw.mean(0), "exc_mean": exc.mean(0),
            "met": {k: float(np.mean(v)) for k, v in met.items()},
            "survived": int(json_load(os.path.join(d, "metrics.json"))["survived"])}


def json_load(p):
    import json
    with open(p) as h:
        return json.load(h)


def extract_events(jobs_n=16):
    jobs = []
    for c, pat, s in itertools.product(CTRL, PATS, SEEDS):
        jobs.append((c, *EE_EVENT, pat, s, "ee"))
        for bt in [BASE_EVENT] + BASE_SENS:
            jobs.append((c, *bt, pat, s, "base"))
    with mp.get_context("fork").Pool(jobs_n) as pool:
        res = pool.map(_episode, jobs, chunksize=10)
    return {r["job"]: r for r in res}


# ============================================================ helpers
def seed_level(R, bi, ti, kind, key, pat="pooled", trace=None):
    """[ctrl, seed] (or [ctrl, seed, sig, time] for traces); pooled = mean of patterns."""
    pats = PATS if pat == "pooled" else (pat,)
    out = []
    for c in CTRL:
        per = []
        for p in pats:
            if trace:
                per.append(np.stack([R[(c, bi, ti, p, s, kind)][trace] for s in SEEDS]))
            else:
                per.append(np.array([R[(c, bi, ti, p, s, kind)]["met"][key] for s in SEEDS]))
        out.append(np.mean(per, axis=0))
    return np.stack(out)


def official(key, pat="pooled"):
    """[ctrl, B, T, seed] official 3-30 s values (pattern mean within seed)."""
    x = A[key]
    return x.mean(axis=1) if pat == "pooled" else x[:, PATS.index(pat)]


def mci(x):
    m = np.mean(x, axis=-1)
    lo, hi = np.empty_like(m), np.empty_like(m)
    for i in np.ndindex(m.shape):
        lo[i], hi[i] = boot_ci(x[i])
    return m, lo, hi


def paired(ref, riv):
    d = riv - ref
    lo, hi = boot_ci(d)
    sd = d.std(ddof=1)
    p = float(sps.ttest_1samp(d, 0.0).pvalue) if sd > 0 else float("nan")
    return {"mean_diff": float(d.mean()), "ci95_lo": lo, "ci95_hi": hi, "p": p,
            "ref_mean": float(ref.mean()), "rival_mean": float(riv.mean()),
            "rel_diff": float(d.mean() / ref.mean()) if ref.mean() != 0 else float("nan"),
            "cohen_dz": float(d.mean() / sd) if sd > 0 else float("nan"), "n_seeds": int(d.size)}


def verdict(r, lower_better=True):
    if not (np.isfinite(r["p_holm"]) and r["p_holm"] < 0.05):
        return "no significant difference"
    rival_lower = r["mean_diff"] < 0
    return f"{r['rival']} lower" if rival_lower else "C1 lower"


# ============================================================ plotting
def _series(ax, x, m, lo, hi, c, markers=True):
    kw = style.series_kwargs(c)
    ax.fill_between(x, lo, hi, color=kw["color"], alpha=0.2, linewidth=0)
    ax.plot(x, m, color=kw["color"], linestyle=kw["linestyle"],
            marker=kw["marker"] if markers else None, markeredgecolor="white",
            markeredgewidth=0.4, label=NAME[c])


def _fig(nrows=2):
    style.apply_style()
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(nrows, 1, figsize=(WIDTH, HEIGHT), sharex=True)
    return fig, axes


def _finish(fig, axes, path, top=0.915):
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, handlelength=2.0,
               columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=0.35, pad=0.25)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)
    print(f"  {os.path.relpath(path, OUT)}.pdf/.png")


def fig1(pat, path, rows):
    fig, (a, b) = _fig()
    for ax, key, lab in ((a, "base_xy_rmse_mps", "Base XY vel.\nRMSE (m/s)"),
                         (b, "ee_pos_rmse_m", "EE position\nRMSE (m)")):
        x = official(key, pat)[:, :, 4, :]                       # [ctrl, B, seed], T = 2 s
        for ci, c in enumerate(CTRL):
            m, lo, hi = mci(x[ci])
            _series(ax, BLEV, m, lo, hi, c)
            rows += [{"figure": 1, "panel": "a" if ax is a else "b", "pattern": pat,
                      "controller": c, "x_base_bound": float(BLEV[i]), "ee_interval": "2s",
                      "metric": key, "mean": float(m[i]), "ci95_lo": float(lo[i]),
                      "ci95_hi": float(hi[i]), "n_seeds": x.shape[-1]} for i in range(5)]
        ax.set_ylabel(lab)
        ax.set_ylim(0, None)
    a.set_ylim(0, 0.2)
    b.set_ylim(0, 0.11)
    a.set_title("(a) Base tracking", loc="left")
    b.set_title("(b) EE tracking", loc="left")
    b.set_xticks(BLEV, [f"{v:.2f}" for v in BLEV])
    b.set_xlabel(r"Base-motion bound $B$ (m/s, rad/s); EE command interval 2 s")
    b.set_xlim(-0.03, 0.63)
    _finish(fig, (a, b), path)


def fig2(pat, path, rows):
    fig, (a, b) = _fig()
    for ax, key, lab in ((a, "base_xy_rmse_mps", "Base XY vel.\nRMSE (m/s)"),
                         (b, "yaw_rmse_radps", "Yaw-rate\nRMSE (rad/s)")):
        x = official(key, pat)[:, 0, :, :]                       # [ctrl, T, seed], B = 0
        for ci, c in enumerate(CTRL):
            m, lo, hi = mci(x[ci])
            _series(ax, FREQ, m, lo, hi, c)
            rows += [{"figure": 2, "panel": "a" if ax is a else "b", "pattern": pat,
                      "controller": c, "x_ee_update_hz": float(FREQ[i]), "ee_level": TLAB[i],
                      "base_bound": 0.0, "metric": key, "mean": float(m[i]),
                      "ci95_lo": float(lo[i]), "ci95_hi": float(hi[i]),
                      "n_seeds": x.shape[-1]} for i in range(5)]
        ax.set_ylabel(lab)
        ax.set_ylim(0, None)
    a.set_title("(a) Base translation", loc="left")
    b.set_title("(b) Base yaw", loc="left")
    b.set_xticks(FREQ, FREQ_LAB)
    b.set_xlim(-0.025, 0.525)
    b.set_xlabel(r"EE command update frequency (Hz); stationary base ($B=0$)")
    _finish(fig, (a, b), path)


def _trace_rows(rows, fig_no, panel, pat, key, x):
    m, lo, hi = mci(np.moveaxis(x, 1, -1))     # x [ctrl, seed, time] -> [ctrl, time, seed]
    for ci, c in enumerate(CTRL):
        rows += [{"figure": fig_no, "panel": panel, "pattern": pat, "controller": c,
                  "t_from_switch_s": round(float(TT[i]), 3), "quantity": key,
                  "mean": float(m[ci, i]), "ci95_lo": float(lo[ci, i]),
                  "ci95_hi": float(hi[ci, i]), "n_seeds": x.shape[1]} for i in range(len(TT))]
    return m, lo, hi


def fig_event(R, kind, bt, panels, pat, path, rows, fig_no, xlabel):
    fig, axes = _fig()
    for ax, (tag, title, sig, mode, ylab) in zip(axes, panels):
        si = SIGS.index(sig)
        x = seed_level(R, *bt, kind, None, pat, trace="exc_mean" if mode == "exc" else "raw_mean")
        x = x[:, :, si, :]                                   # [ctrl, seed, time]
        m, lo, hi = _trace_rows(rows, fig_no, tag, pat, f"{sig}_{mode}", x)
        for ci, c in enumerate(CTRL):
            _series(ax, TT, m[ci], lo[ci], hi[ci], c, markers=False)
        ax.axvline(0.0, color=style.TEXT_MUTED, linewidth=0.6, linestyle=":")
        if mode == "exc":
            ax.axhline(0.0, color=style.TEXT_PRIMARY, linewidth=0.6)
        else:
            ax.set_ylim(0, None)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
    axes[-1].set_xlim(TT[0], TT[-1])
    axes[-1].set_xlabel(xlabel)
    _finish(fig, axes, path)


# ============================================================ statistics
EV_METRICS = {
    "fig3": ("ee", EE_EVENT, ["base_xy", "yaw"]),
    "fig4": ("base", BASE_EVENT, ["ee_pos", "ee_ori"]),
}
PRE_REG = ["peak_excess", "int_excess", "recovery_s"]
POST_HOC = ["peak_pos_excess", "int_pos_excess"]


def event_statistics(R):
    per_ctrl, pairs = [], {"pre-registered": [], "post-hoc": []}
    for fig, (kind, bt, sigs) in EV_METRICS.items():
        extra = ["ee_pos_baseline_abs", "ee_pos_peak_abs", "ee_pos_residual_abs",
                 "ee_ori_baseline_abs", "ee_ori_residual_abs",
                 "base_xy_baseline_abs", "yaw_baseline_abs"]
        keys = [f"{s}_{m}" for s in sigs for m in PRE_REG + POST_HOC + ["recovery_censored"]] + extra
        for key in dict.fromkeys(keys):
            x = seed_level(R, *bt, kind, key)
            for ci, c in enumerate(CTRL):
                m, lo, hi = mci(x[ci][None])
                per_ctrl.append({"figure": fig, "event": kind, "base_bound": float(BLEV[bt[0]]),
                                 "ee_level": TLAB[bt[1]], "metric": key, "controller": c,
                                 "mean": float(m[0]), "ci95_lo": float(lo[0]), "ci95_hi": float(hi[0]),
                                 "sd": float(x[ci].std(ddof=1)), "n_seeds": x.shape[1],
                                 "status": ("pre-registered" if any(key.endswith(p) for p in PRE_REG)
                                            else "post-hoc" if any(key.endswith(p) for p in POST_HOC)
                                            else "descriptive")})
            fam = ("pre-registered" if any(key.endswith(p) for p in PRE_REG)
                   else "post-hoc" if any(key.endswith(p) for p in POST_HOC) else None)
            if fam:
                for riv in ("C3", "C5"):
                    r = paired(x[0], x[CTRL.index(riv)])
                    pairs[fam].append({"figure": fig, "event": kind, "metric": key,
                                       "comparison": f"{riv}-C1", "rival": riv,
                                       "base_bound": float(BLEV[bt[0]]), "ee_level": TLAB[bt[1]],
                                       "family": fam} | r)
    out = []
    for fam, rows in pairs.items():
        for r, h, q in zip(rows, holm([r["p"] for r in rows]), bh([r["p"] for r in rows])):
            r["p_holm"], r["q_bh"] = float(h), float(q)
            r["verdict"] = verdict(r)
            r["holm_family_size"] = len(rows)
        out += rows
    common.write_csv(os.path.join(DATA, "event_statistics_per_controller.csv"), per_ctrl)
    common.write_csv(os.path.join(DATA, "event_statistics_paired.csv"), out)
    return per_ctrl, out


def base_event_sensitivity(R):
    """Fig. 4 metrics at every base level with a fixed EE target (B = 0.15 ... 0.60)."""
    rows = []
    for bt in BASE_SENS + [BASE_EVENT]:
        for key in ("ee_pos_peak_pos_excess", "ee_pos_int_pos_excess", "ee_ori_peak_pos_excess",
                    "ee_ori_int_pos_excess", "ee_pos_peak_excess", "ee_ori_peak_excess",
                    "ee_pos_baseline_abs", "ee_ori_baseline_abs"):
            x = seed_level(R, *bt, "base", key)
            for riv in ("C3", "C5"):
                rows.append({"base_bound": float(BLEV[bt[0]]), "metric": key,
                             "comparison": f"{riv}-C1", "rival": riv,
                             "n_events_per_episode_C1": float(np.mean(
                                 [R[("C1", *bt, p, s, "base")]["n_events"] for p in PATS for s in SEEDS]))}
                            | paired(x[0], x[CTRL.index(riv)]))
    for r, h in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm_within_this_table"] = float(h)
    common.write_csv(os.path.join(DATA, "fig4_base_level_sensitivity.csv"), rows)
    return rows


def frequency_statistics():
    """Fig. 2 support: absolute RMSE and change vs Fixed (B = 0), per controller,
    and C1-vs-rival differences in that change (official pre-registered family
    from stats/E3v3_degradation_C1_vs_rivals.csv)."""
    rows = []
    for key in ("base_xy_rmse_mps", "yaw_rmse_radps"):
        x = official(key)[:, 0]                     # [ctrl, T, seed]
        for ci, c in enumerate(CTRL):
            for ti in range(5):
                d = x[ci, ti] - x[ci, 0]
                lo, hi = boot_ci(d) if ti else (0.0, 0.0)
                rows.append({"metric": key, "controller": c, "ee_level": TLAB[ti],
                             "abs_mean": float(x[ci, ti].mean()),
                             "delta_vs_fixed_mean": float(d.mean()), "delta_ci95_lo": lo,
                             "delta_ci95_hi": hi,
                             "delta_rel_pct": float(100 * d.mean() / x[ci, 0].mean())})
    common.write_csv(os.path.join(DATA, "fig2_change_vs_fixed.csv"), rows)
    import csv
    off = [r for r in csv.DictReader(open(os.path.join(V3, "stats",
                                                        "E3v3_degradation_C1_vs_rivals.csv")))
           if r["direction"] == "ee_to_base" and r["pattern"] == "pooled"
           and float(r["base_bound"]) in (0.0, 0.6)]
    common.write_csv(os.path.join(DATA, "fig2_official_degradation_C1_vs_rivals_B0_B060.csv"), off)
    return rows, off


def pattern_consistency(R):
    """Sign of rival - C1 per pattern for every plotted comparison."""
    rows = []

    def add(fig, label, ref_l, riv_l, ref_c, riv_c):
        for p, (ref, riv) in (("line", (ref_l, riv_l)), ("circle", (ref_c, riv_c))):
            r = paired(ref, riv)
            rows.append({"figure": fig, "quantity": label, "pattern": p} | r)

    for key, f in (("base_xy_rmse_mps", 1), ("ee_pos_rmse_m", 1)):
        for bi in range(5):
            for riv in ("C3", "C5"):
                L, C = official(key, "line"), official(key, "circle")
                ri = CTRL.index(riv)
                add(f, f"{key} B={BLEV[bi]:.2f} T=2s {riv}-C1", L[0, bi, 4], L[ri, bi, 4],
                    C[0, bi, 4], C[ri, bi, 4])
    for key in ("base_xy_rmse_mps", "yaw_rmse_radps"):
        for ti in range(5):
            for riv in ("C3", "C5"):
                L, C = official(key, "line"), official(key, "circle")
                ri = CTRL.index(riv)
                add(2, f"{key} B=0 T={TLAB[ti]} {riv}-C1", L[0, 0, ti], L[ri, 0, ti],
                    C[0, 0, ti], C[ri, 0, ti])
    for fig, (kind, bt, sigs) in EV_METRICS.items():
        for s, m, riv in itertools.product(sigs, PRE_REG + POST_HOC, ("C3", "C5")):
            key = f"{s}_{m}"
            L, C = seed_level(R, *bt, kind, key, "line"), seed_level(R, *bt, kind, key, "circle")
            ri = CTRL.index(riv)
            add(fig, f"{key} {riv}-C1", L[0], L[ri], C[0], C[ri])
    # flag opposite signs
    by = {}
    for r in rows:
        by.setdefault((r["figure"], r["quantity"]), {})[r["pattern"]] = r
    flags = []
    for k, v in by.items():
        l, c = v["line"], v["circle"]
        opp = np.sign(l["mean_diff"]) != np.sign(c["mean_diff"])
        both_sig = (l["ci95_lo"] > 0 or l["ci95_hi"] < 0) and (c["ci95_lo"] > 0 or c["ci95_hi"] < 0)
        flags.append({"figure": k[0], "quantity": k[1], "line_mean_diff": l["mean_diff"],
                      "line_ci": f"[{l['ci95_lo']:.4g}, {l['ci95_hi']:.4g}]",
                      "circle_mean_diff": c["mean_diff"],
                      "circle_ci": f"[{c['ci95_lo']:.4g}, {c['ci95_hi']:.4g}]",
                      "opposite_sign": int(opp), "opposite_and_both_CIs_exclude_0": int(opp and both_sig)})
    common.write_csv(os.path.join(DATA, "pattern_consistency.csv"), flags)
    return flags


def verify(R, plotted):
    """Plotted values vs official summary; recomputed events vs analysis arrays."""
    import csv
    S = {(r["controller"], float(r["base_bound"]), r["ee_level"], r["metric"]): r
         for r in csv.DictReader(open(os.path.join(V3, "stats", "E3v3_summary.csv")))
         if r["pattern"] == "pooled"}
    w_off = 0.0
    for r in plotted:
        if r["figure"] in (1, 2) and r["pattern"] == "pooled":
            b = r.get("x_base_bound", r.get("base_bound"))
            t = r.get("ee_level", "2s") if r["figure"] == 2 else "2s"
            s = S[(r["controller"], float(b), t, r["metric"])]
            w_off = max(w_off, abs(r["mean"] - float(s["mean"])), abs(r["ci95_lo"] - float(s["ci95_lo"])),
                        abs(r["ci95_hi"] - float(s["ci95_hi"])))
    w_tr, w_cnt, w_met, same_events, surv = 0.0, 0, 0.0, True, True
    for (c, bi, ti, p, s, kind), r in R.items():
        ci, pi, si = CTRL.index(c), PATS.index(p), SEEDS.index(s)
        arr = A[f"trace_{kind}_iso"][ci, pi, bi, ti, si].astype(float)
        w_tr = max(w_tr, float(np.abs(arr - r["raw_mean"]).max()))
        w_cnt += int(A[f"n_{kind}_iso"][ci, pi, bi, ti, si] != r["n_events"])
        pref = "ee_iso" if kind == "ee" else "base_iso"
        for sig in (("base_xy", "yaw") if kind == "ee" else ("ee_pos", "ee_ori")):
            for m in ("peak_excess", "int_excess", "recovery_s"):
                w_met = max(w_met, abs(float(A[f"{pref}_{sig}_{m}"][ci, pi, bi, ti, si])
                                       - r["met"][f"{sig}_{m}"]))
        if R[("C1", bi, ti, p, s, kind)]["ticks"] != r["ticks"]:
            same_events = False
        surv &= bool(r["survived"])
    res = {"max_abs_plotted_minus_official_summary_figs1_2": w_off,
           "max_abs_recomputed_trace_minus_analysis_arrays": w_tr,
           "event_count_mismatches_vs_analysis": w_cnt,
           "max_abs_prereg_event_metric_minus_analysis_arrays": w_met,
           "identical_event_ticks_across_controllers": same_events,
           "all_event_episodes_completed": surv,
           "episodes_used_for_events": len(R)}
    common.write_csv(os.path.join(DATA, "verification.csv"), [res])
    print(res)
    return res


# ============================================================ main
def main():
    os.makedirs(DATA, exist_ok=True)
    R = extract_events()
    for pat in ("pooled", "line", "circle"):
        sub = "" if pat == "pooled" else os.path.join("patterns", "")
        suf = "" if pat == "pooled" else f"_{pat}"
        rows1, rows2, rows3, rows4 = [], [], [], []
        fig1(pat, os.path.join(OUT, sub, f"E3v3_paper_fig1_concurrent_tracking{suf}"), rows1)
        fig2(pat, os.path.join(OUT, sub, f"E3v3_paper_fig2_ee_to_base_frequency{suf}"), rows2)
        fig_event(R, "ee", EE_EVENT,
                  [("a", "(a) Base disturbance (signed excess)", "base_xy", "exc",
                    "Excess base XY\nvel. error (m/s)"),
                   ("b", "(b) EE response (absolute)", "ee_pos", "raw", "EE position\nerror (m)")],
                  pat, os.path.join(OUT, sub, f"E3v3_paper_fig3_ee_to_base_response{suf}"), rows3, 3,
                  r"Time from EE-pose command switch (s); $B=0$, interval 2 s")
        fig_event(R, "base", BASE_EVENT,
                  [("a", "(a) EE position (signed excess)", "ee_pos", "exc", "Excess EE pos.\nerror (m)"),
                   ("b", "(b) EE orientation (signed excess)", "ee_ori", "exc",
                    "Excess EE ori.\nerror (rad)")],
                  pat, os.path.join(OUT, sub, f"E3v3_paper_fig4_base_to_ee_response{suf}"), rows4, 4,
                  r"Time from base-command switch (s); EE fixed, $B=0.60$")
        tag = "" if pat == "pooled" else f"_{pat}"
        for no, rr in ((1, rows1), (2, rows2), (3, rows3), (4, rows4)):
            common.write_csv(os.path.join(DATA, f"fig{no}_plotted_values{tag}.csv"), rr)
        if pat == "pooled":
            plotted = rows1 + rows2
    event_statistics(R)
    base_event_sensitivity(R)
    frequency_statistics()
    pattern_consistency(R)
    verify(R, plotted)


if __name__ == "__main__":
    main()
