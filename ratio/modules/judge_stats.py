"""Interval estimates for the Judicial History Tracker, in the standard library only.

A judge's rate is shown with a Wilson score interval. Whether it differs from the baseline is
decided by Newcombe's hybrid score interval for the difference of two proportions (Newcombe 1998,
method 10), at a confidence level split across the indicators compared (Bonferroni), so a page
with many indicators does not fire by chance.
"""

from __future__ import annotations

import math
from statistics import NormalDist


def z_value(confidence: float) -> float:
    """Two-sided normal quantile: 1.96 for 0.95."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def family_confidence(alpha: float, compared: int) -> float:
    """Per-indicator confidence that keeps the family-wise error at ``alpha`` over ``compared`` indicators."""
    if compared < 1:
        raise ValueError("at least one indicator must be compared")
    return 1.0 - alpha / compared


def wilson(k: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"impossible counts: {k} of {n}")
    z = z_value(confidence)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    low = 0.0 if k == 0 else max(0.0, centre - half)  # exact at the extremes, without rounding error
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def newcombe(k1: int, n1: int, k2: int, n2: int, confidence: float) -> tuple[float, float]:
    """Interval for p1 - p2 built from the two Wilson intervals (Newcombe's method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    low1, high1 = wilson(k1, n1, confidence)
    low2, high2 = wilson(k2, n2, confidence)
    difference = p1 - p2
    low = difference - math.sqrt((p1 - low1) ** 2 + (high2 - p2) ** 2)
    high = difference + math.sqrt((high1 - p1) ** 2 + (p2 - low2) ** 2)
    return low, high
