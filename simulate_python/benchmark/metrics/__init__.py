"""Metric definitions.

The per-episode metric computation lives in ``benchmark.core.recorder``
(:class:`EpisodeRecorder.compute_metrics`, which also owns the online 250 Hz
power and joint-acceleration accumulators), and the across-episode statistics
live in ``benchmark.core.stats``. They are re-exported here so
``benchmark.metrics`` is a usable entry point.
"""

from benchmark.core.recorder import EpisodeRecorder  # noqa: F401
from benchmark.core.stats import (  # noqa: F401
    best_methods,
    bootstrap_ci,
    paired_difference,
    summarize,
    survival_rate,
    wilson_interval,
)
