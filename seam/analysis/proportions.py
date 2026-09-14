"""Binomial proportion confidence intervals - project-wide convention.

**Chosen method: Wilson score interval (continuity-uncorrected).**

Recorded 2026-08-02 after Phase E. The E2 failure-rate interval was initially mislabeled
Agresti-Coull; the numbers matched Wilson ([0, 27.75%] for 0/10), not Agresti-Coull
([0, 32.09%]). Rule of three for 0/n is 3/n (= 30% at n=10).

Wilson is the single method for every binomial proportion in this project. Do not mix
Agresti-Coull, Clopper-Pearson, or rule-of-three into the same table without renaming.
Rule of three may be cited as a *bound* in prose when x=0, but the reported interval is Wilson.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Final

__all__ = [
    "PROPORTION_CI_METHOD",
    "ProportionCI",
    "wilson_ci",
]

#: Project-wide label. Cite this string in every artifact that reports a proportion CI.
PROPORTION_CI_METHOD: Final = "wilson_score"


@dataclass(frozen=True, slots=True)
class ProportionCI:
    """A binomial proportion with a Wilson score interval."""

    successes: int
    trials: int
    proportion: float
    lo: float
    hi: float
    confidence: float
    method: str = PROPORTION_CI_METHOD

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wilson_ci(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
    z: float = 1.96,
) -> ProportionCI:
    """Wilson score interval for a binomial proportion.

    For ``successes=0, trials=10, confidence=0.95`` this returns approximately
    ``[0.0, 0.2775]`` - matching the Phase E E2 report numbers that were mislabeled
    Agresti-Coull.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid successes/trials: {successes}/{trials}")
    if trials == 0:
        return ProportionCI(
            successes=0,
            trials=0,
            proportion=math.nan,
            lo=math.nan,
            hi=math.nan,
            confidence=confidence,
        )
    phat = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (phat + z2 / (2.0 * trials)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) / trials) + z2 / (4.0 * trials * trials))) / denom
    return ProportionCI(
        successes=successes,
        trials=trials,
        proportion=phat,
        lo=max(0.0, center - half),
        hi=min(1.0, center + half),
        confidence=confidence,
    )
