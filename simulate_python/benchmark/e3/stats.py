"""Statistics helpers for E3 (percentile bootstrap, paired tests, Holm/BH).
Identical procedures to the earlier E3 analyses: 10,000 resamples, seed
20261106, the same resampled seed indices for every series of equal length
(paired resampling), Holm within declared families, |rel| >= 10 % practical."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

BOOT_N, BOOT_SEED, CONF = 10_000, 20261106, 0.95
PRACTICAL = 0.10

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


