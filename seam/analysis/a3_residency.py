"""A3 residency sweep analysis: OLS regressions, extremes ratio, block-position drift."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from typing import Any

__all__ = [
    "block_position_slope",
    "classify_mechanism_line",
    "ols_regression",
    "page_read_distribution",
    "phase3_fork_from_top_blocks",
    "regression_with_bootstrap_extremes",
]


def page_read_distribution(rates: Sequence[float | None], *, threshold: float) -> dict[str, Any]:
    numeric = [float(rate) for rate in rates if rate is not None]
    if not numeric:
        return {
            "samples": [],
            "median": None,
            "p95": None,
            "max": None,
            "fraction_above_threshold": None,
            "n_samples": 0,
        }
    ordered = sorted(numeric)
    n = len(ordered)
    p95_index = min(n - 1, max(0, round(0.95 * (n - 1))))
    above = sum(1 for rate in numeric if rate > threshold)
    return {
        "samples": list(numeric),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[p95_index]),
        "max": float(ordered[-1]),
        "fraction_above_threshold": above / n,
        "n_samples": n,
    }


def ols_regression(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    """Ordinary least squares: y ~ a + b * x. Returns slope, intercept, R², n."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have equal length")
    n = len(xs)
    if n < 2:
        return {
            "n": n,
            "slope": math.nan,
            "intercept": math.nan,
            "r_squared": math.nan,
            "note": "need >=2 points",
        }
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    if sxx == 0:
        return {
            "n": n,
            "slope": math.nan,
            "intercept": mean_y,
            "r_squared": math.nan,
            "note": "zero variance in x",
        }
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True)]
    ss_res = sum(r * r for r in residuals)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r_squared = (1.0 - ss_res / ss_tot) if ss_tot else math.nan
    return {
        "n": n,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "fitted_at_min_x": float(intercept + slope * min(xs)),
        "fitted_at_max_x": float(intercept + slope * max(xs)),
        "x_min": float(min(xs)),
        "x_max": float(max(xs)),
    }


def _bootstrap_fitted_extremes(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    n = len(xs)
    if n < 2:
        return {
            "fitted_high": {"point": math.nan, "lo": math.nan, "hi": math.nan},
            "fitted_low": {"point": math.nan, "lo": math.nan, "hi": math.nan},
            "extremes_ratio": {"point": math.nan, "lo": math.nan, "hi": math.nan},
            "resamples": 0,
        }
    x_min, x_max = min(xs), max(xs)
    point = ols_regression(xs, ys)
    rng = random.Random(seed)
    high_draws: list[float] = []
    low_draws: list[float] = []
    ratio_draws: list[float] = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idx]
        by = [ys[i] for i in idx]
        fit = ols_regression(bx, by)
        if math.isnan(fit["slope"]):
            continue
        high = fit["intercept"] + fit["slope"] * x_max
        low = fit["intercept"] + fit["slope"] * x_min
        high_draws.append(high)
        low_draws.append(low)
        if low != 0:
            ratio_draws.append(high / low)

    def _ci(draws: list[float], point_value: float) -> dict[str, float]:
        if not draws:
            return {"point": point_value, "lo": math.nan, "hi": math.nan}
        ordered = sorted(draws)
        alpha = (1.0 - confidence) / 2.0
        lo_idx = max(0, int(alpha * len(ordered)) - 1)
        hi_idx = min(len(ordered) - 1, int((1.0 - alpha) * len(ordered)))
        return {
            "point": float(point_value),
            "lo": float(ordered[lo_idx]),
            "hi": float(ordered[hi_idx]),
        }

    high_point = point["fitted_at_max_x"]
    low_point = point["fitted_at_min_x"]
    if low_point not in (0, None) and not math.isnan(low_point):
        ratio_point = high_point / low_point
    else:
        ratio_point = math.nan
    return {
        "fitted_high": _ci(high_draws, high_point),
        "fitted_low": _ci(low_draws, low_point),
        "extremes_ratio": _ci(ratio_draws, ratio_point),
        "resamples": len(high_draws),
        "x_high": float(x_max),
        "x_low": float(x_min),
    }


def regression_with_bootstrap_extremes(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260804,
    confidence: float = 0.95,
    label: str,
) -> dict[str, Any]:
    fit = ols_regression(xs, ys)
    extremes = _bootstrap_fitted_extremes(
        xs, ys, resamples=resamples, seed=seed, confidence=confidence
    )
    return {"label": label, "ols": fit, "bootstrap_extremes": extremes}


def block_position_slope(block_positions: Sequence[int], values: Sequence[float]) -> dict[str, Any]:
    """Drift check: value ~ block_position."""
    fit = ols_regression([float(p) for p in block_positions], list(values))
    return {"ols": fit, "interpretation": "non-zero slope suggests thermal/position drift"}


def phase3_fork_from_top_blocks(
    top_blocks: Sequence[dict[str, Any]],
    *,
    threshold: float,
    sustained_consecutive_samples: int,
) -> dict[str, Any]:
    """True when top-of-ladder blocks still show sustained page reads above threshold."""
    any_sustained = False
    details: list[dict[str, Any]] = []
    for block in top_blocks:
        rates = block.get("hard_page_reads_per_s", {}).get("samples") or []
        longest = 0
        current = 0
        for rate in rates:
            if rate is not None and float(rate) > threshold:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        sustained = longest >= sustained_consecutive_samples
        any_sustained = any_sustained or sustained
        details.append(
            {
                "block_position": block.get("block_position"),
                "longest_consecutive_excess": longest,
                "sustained": sustained,
            }
        )
    return {
        "phase3_indicated": any_sustained,
        "threshold": threshold,
        "sustained_consecutive_samples": sustained_consecutive_samples,
        "top_block_details": details,
        "criterion": ("top-of-ladder blocks show sustained hard page reads above threshold"),
    }


def classify_mechanism_line(
    *,
    r_decode_slope: float,
    page_read_slope: float,
    extremes_ratio: float,
    target_ratio: float,
) -> str:
    """One-line mechanism classification for the report."""
    decode_rises = r_decode_slope > 0
    pages_fall = page_read_slope < 0
    pages_vary = abs(page_read_slope) > 1e-9
    decode_varies = abs(r_decode_slope) > 1e-9
    if decode_rises and pages_fall:
        return "paging"
    if decode_varies and pages_vary:
        return "both"
    if (not decode_varies) and pages_vary:
        return "placement"
    if decode_varies and not pages_vary:
        return "unexplained"
    # Extremes-ratio vs target_ratio is reported separately; unused here by design.
    _ = (extremes_ratio, target_ratio)
    return "unexplained"
