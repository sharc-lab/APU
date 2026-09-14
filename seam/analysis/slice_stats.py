"""Statistics for M-SLICE: bootstrap intervals, noise floor, and the null rule.

Two rules from the brief are encoded here rather than applied by hand at reporting time, because a
rule applied by hand is a rule that gets bent once the numbers are visible.

**Any effect smaller than twice the measured CV is reported as NULL.** :func:`compare_against_noise`
returns that verdict, and the caller cannot get a "real" verdict without supplying a noise floor.

**A/A must pass before any comparison is believed.** :func:`aa_verdict` runs the *same* pipeline on
two labels of one condition; a significant difference there means the harness manufactures effects,
and the correct response is to stop rather than to interpret the main run.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

__all__ = [
    "BootstrapCI",
    "ComparisonVerdict",
    "NoiseFloor",
    "aa_verdict",
    "bootstrap_ci",
    "coefficient_of_variation",
    "compare_against_noise",
]


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Point estimate with a percentile bootstrap interval."""

    point: float
    lo: float
    hi: float
    n: int
    resamples: int
    statistic: str = "mean"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260802,
    label: str = "mean",
) -> BootstrapCI:
    """Percentile bootstrap interval.

    Seeded so the interval is reproducible: an unseeded bootstrap makes a figure that cannot be
    regenerated from raw, which blueprint G5 forbids.
    """
    data = list(values)
    if not data:
        return BootstrapCI(
            point=math.nan, lo=math.nan, hi=math.nan, n=0, resamples=0, statistic=label
        )
    if len(data) == 1:
        only = float(data[0])
        return BootstrapCI(point=only, lo=only, hi=only, n=1, resamples=0, statistic=label)

    rng = random.Random(seed)
    n = len(data)
    draws = [statistic([data[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)]
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(alpha * resamples) - 1)
    hi_idx = min(resamples - 1, int((1.0 - alpha) * resamples))
    return BootstrapCI(
        point=statistic(data),
        lo=draws[lo_idx],
        hi=draws[hi_idx],
        n=n,
        resamples=resamples,
        statistic=label,
    )


def coefficient_of_variation(values: Sequence[float]) -> float:
    """CV = sd / |mean|.

    Returns ``nan`` for a zero mean rather than infinity: a metric that is identically zero across
    repeats (for example a zero escalation rate under a generous deadline) has no meaningful
    relative spread, and reporting ``inf`` would falsely imply enormous noise.
    """
    data = [float(v) for v in values]
    if len(data) < 2:
        return math.nan
    mean = statistics.fmean(data)
    if mean == 0.0:
        return math.nan
    return statistics.stdev(data) / abs(mean)


@dataclass(frozen=True, slots=True)
class NoiseFloor:
    """Measured repeat-to-repeat variability on one fixed configuration."""

    metric: str
    n_repeats: int
    mean: float
    sd: float
    cv: float
    ci: BootstrapCI

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ci"] = self.ci.to_dict()
        return payload


def noise_floor(metric: str, values: Sequence[float], *, seed: int = 20260802) -> NoiseFloor:
    data = [float(v) for v in values]
    return NoiseFloor(
        metric=metric,
        n_repeats=len(data),
        mean=statistics.fmean(data) if data else math.nan,
        sd=statistics.stdev(data) if len(data) > 1 else math.nan,
        cv=coefficient_of_variation(data),
        ci=bootstrap_ci(data, seed=seed, label=f"{metric}_mean"),
    )


@dataclass(frozen=True, slots=True)
class ComparisonVerdict:
    """Difference between two conditions, judged against the noise floor."""

    metric: str
    a_mean: float
    b_mean: float
    absolute_difference: float
    relative_difference: float
    threshold_relative: float
    cv: float
    verdict: Literal["REAL", "NULL", "UNDETERMINED"]
    reason: str
    a_ci: BootstrapCI
    b_ci: BootstrapCI
    ci_separated: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["a_ci"] = self.a_ci.to_dict()
        payload["b_ci"] = self.b_ci.to_dict()
        return payload


def compare_against_noise(
    *,
    metric: str,
    a_values: Sequence[float],
    b_values: Sequence[float],
    cv: float,
    null_threshold_cv_multiple: float = 2.0,
    resamples: int = 10_000,
    seed: int = 20260802,
) -> ComparisonVerdict:
    """Judge a difference against the measured noise floor.

    The threshold is ``null_threshold_cv_multiple x cv`` in **relative** terms. An effect below it
    is NULL - not "suggestive", not "trending". That is the whole point of measuring a noise floor
    before collecting the comparison.
    """
    a_ci = bootstrap_ci(a_values, resamples=resamples, seed=seed, label=f"{metric}_a_mean")
    b_ci = bootstrap_ci(b_values, resamples=resamples, seed=seed + 1, label=f"{metric}_b_mean")

    a_mean, b_mean = a_ci.point, b_ci.point
    absolute = b_mean - a_mean
    denominator = abs(a_mean)
    relative = absolute / denominator if denominator > 0 else math.nan
    threshold = null_threshold_cv_multiple * cv
    ci_separated = a_ci.hi < b_ci.lo or b_ci.hi < a_ci.lo

    if math.isnan(cv):
        verdict: Literal["REAL", "NULL", "UNDETERMINED"] = "UNDETERMINED"
        reason = (
            "no usable noise floor for this metric (CV undefined, typically a metric that is "
            "identically zero across repeats); the difference cannot be judged against noise"
        )
    elif math.isnan(relative):
        verdict = "UNDETERMINED"
        reason = "baseline mean is zero; relative effect size is undefined"
    elif abs(relative) < threshold:
        verdict = "NULL"
        reason = (
            f"|relative difference| {abs(relative):.4f} < {null_threshold_cv_multiple}x CV "
            f"{threshold:.4f}; reported as NULL"
        )
    elif not ci_separated:
        verdict = "NULL"
        reason = (
            f"|relative difference| {abs(relative):.4f} exceeds {null_threshold_cv_multiple}x CV "
            f"{threshold:.4f}, but the bootstrap intervals overlap; reported as NULL"
        )
    else:
        verdict = "REAL"
        reason = (
            f"|relative difference| {abs(relative):.4f} >= {null_threshold_cv_multiple}x CV "
            f"{threshold:.4f} and bootstrap intervals are separated"
        )

    return ComparisonVerdict(
        metric=metric,
        a_mean=a_mean,
        b_mean=b_mean,
        absolute_difference=absolute,
        relative_difference=relative,
        threshold_relative=threshold,
        cv=cv,
        verdict=verdict,
        reason=reason,
        a_ci=a_ci,
        b_ci=b_ci,
        ci_separated=ci_separated,
    )


def aa_verdict(
    *,
    metrics: dict[str, tuple[Sequence[float], Sequence[float]]],
    cvs: dict[str, float],
    null_threshold_cv_multiple: float = 2.0,
    resamples: int = 10_000,
    seed: int = 20260802,
) -> dict[str, Any]:
    """Run the full comparison pipeline on two labels of the SAME condition.

    A ``REAL`` verdict anywhere means the pipeline reports differences that cannot exist, so no
    comparison it produces is trustworthy. The correct response is to stop and fix the harness.
    """
    comparisons = {
        metric: compare_against_noise(
            metric=metric,
            a_values=a,
            b_values=b,
            cv=cvs.get(metric, math.nan),
            null_threshold_cv_multiple=null_threshold_cv_multiple,
            resamples=resamples,
            seed=seed,
        )
        for metric, (a, b) in metrics.items()
    }
    false_positives = [m for m, c in comparisons.items() if c.verdict == "REAL"]
    return {
        "passed": not false_positives,
        "false_positive_metrics": false_positives,
        "comparisons": {m: c.to_dict() for m, c in comparisons.items()},
        "interpretation": (
            "A/A PASSED: the pipeline reports no significant difference between two labels of the "
            "same condition."
            if not false_positives
            else (
                "A/A FAILED: the pipeline reported a difference between two labels of the SAME "
                f"condition on {false_positives}. The harness manufactures effects; no comparison "
                "is valid until this is fixed. STOP."
            )
        ),
    }
