"""Statistics for the official quantitative evaluation.

Rules enforced here:

* The experimental unit is the independently seeded **episode**. The 50 Hz and
  250 Hz time samples inside an episode are never treated as independent
  replicates: every function takes one number per episode.
* 95% confidence intervals come from the percentile bootstrap with 10,000
  resamples and a fixed bootstrap seed, so a rerun gives identical intervals.
* Controller differences preserve pairing: the difference is taken per seed and
  only seeds present for both controllers are used.
* ``significant`` means the paired-difference CI excludes zero. Nothing is
  bolded as "best" on a point estimate alone.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261106
DEFAULT_CONFIDENCE = 0.95


def _finite(values) -> np.ndarray:
    a = np.asarray(list(values), dtype=np.float64)
    return a[np.isfinite(a)]


def bootstrap_ci(
    values,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
    statistic=np.mean,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``statistic`` over per-episode values."""
    a = _finite(values)
    if a.size == 0:
        return (float("nan"), float("nan"))
    if a.size == 1:
        return (float(a[0]), float(a[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(int(n_resamples), a.size))
    draws = statistic(a[idx], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1.0 - alpha)),
    )


def summarize(
    values,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    a = _finite(values)
    if a.size == 0:
        return {
            "n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"), "sem": float("nan"),
        }
    lo, hi = bootstrap_ci(a, n_resamples, seed, confidence)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "ci_low": lo,
        "ci_high": hi,
        "sem": float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0,
    }


def paired_difference(
    values_a: dict,
    values_b: dict,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """Bootstrap CI of ``mean(b - a)`` over the seeds present in BOTH dicts.

    ``values_*`` map seed -> per-episode metric. Pairing is preserved: the
    difference is formed seed by seed before any resampling, and the bootstrap
    resamples whole pairs.
    """
    seeds = sorted(set(values_a) & set(values_b))
    diffs = []
    for s in seeds:
        x, y = float(values_a[s]), float(values_b[s])
        if np.isfinite(x) and np.isfinite(y):
            diffs.append(y - x)
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        return {
            "n_pairs": 0, "mean": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "significant": False,
        }
    lo, hi = bootstrap_ci(d, n_resamples, seed, confidence)
    return {
        "n_pairs": int(d.size),
        "mean": float(np.mean(d)),
        "std": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "significant": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)),
    }


def best_methods(
    per_method: dict,
    lower_is_better: bool = True,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list:
    """Methods that are not statistically beaten by the point-estimate leader.

    Returns every method whose paired difference against the leader has a CI
    containing zero, plus the leader itself. Bolding this set, instead of the
    single smallest number, is what keeps the tables honest when the
    uncertainty does not support a clear winner.
    """
    means = {
        m: float(np.mean(_finite(v.values()))) if len(_finite(v.values())) else float("nan")
        for m, v in per_method.items()
    }
    valid = {m: v for m, v in means.items() if np.isfinite(v)}
    if not valid:
        return []
    leader = min(valid, key=valid.get) if lower_is_better else max(valid, key=valid.get)
    out = [leader]
    for method in valid:
        if method == leader:
            continue
        res = paired_difference(
            per_method[leader], per_method[method], n_resamples=n_resamples, seed=seed
        )
        if not res["significant"]:
            out.append(method)
    return sorted(out)


def survival_rate(survived) -> float:
    a = np.asarray(list(survived), dtype=np.float64)
    return float(np.mean(a)) if a.size else float("nan")


def wilson_interval(successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE):
    """CI for a survival rate. A proportion needs a proportion interval, not a
    bootstrap of a 0/1 mean, when n is small."""
    if n == 0:
        return (float("nan"), float("nan"))
    from math import sqrt

    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else _z_for(confidence)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _z_for(confidence: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(0.5 + confidence / 2.0))


def paired_advantage(
    values_ref: dict,
    values_other: dict,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """Per-seed advantage of ``ref`` over ``other`` for a lower-is-better metric.

    ``advantage = other - ref``, so a POSITIVE value means ``ref`` had the
    smaller error on that seed. Reported as mean, median and the fraction of
    paired trials in which ``ref`` won, because a mean advantage can be carried
    by a few seeds while the median and the win fraction cannot.

    ``significant`` is still the mean's bootstrap CI excluding zero; the win
    fraction is descriptive and is never used to declare significance.
    """
    seeds = sorted(set(values_ref) & set(values_other))
    pairs = []
    for s in seeds:
        x, y = float(values_ref[s]), float(values_other[s])
        if np.isfinite(x) and np.isfinite(y):
            pairs.append((s, y - x))
    if not pairs:
        return {
            "n_pairs": 0, "mean": float("nan"), "median": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
            "median_ci_low": float("nan"), "median_ci_high": float("nan"),
            "frac_positive": float("nan"), "n_positive": 0, "significant": False,
        }
    d = np.asarray([v for _, v in pairs], dtype=np.float64)
    lo, hi = bootstrap_ci(d, n_resamples, seed, confidence)
    mlo, mhi = bootstrap_ci(d, n_resamples, seed, confidence, statistic=np.median)
    n_pos = int(np.sum(d > 0.0))
    return {
        "n_pairs": int(d.size),
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "std": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "median_ci_low": mlo,
        "median_ci_high": mhi,
        "frac_positive": float(n_pos / d.size),
        "n_positive": n_pos,
        "significant": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)),
        "seeds": [s for s, _ in pairs],
    }
