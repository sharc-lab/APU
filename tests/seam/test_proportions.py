"""Project-wide proportion CI is Wilson score - not Agresti-Coull."""

from __future__ import annotations

import pytest

from seam.analysis.proportions import PROPORTION_CI_METHOD, wilson_ci


def test_method_label_is_wilson() -> None:
    assert PROPORTION_CI_METHOD == "wilson_score"
    assert wilson_ci(0, 10).method == "wilson_score"


def test_phase_e_e2_zero_of_ten_matches_reported_interval() -> None:
    """The E2 report [0, 0.2775] is Wilson; Agresti-Coull would be ~0.3209."""
    ci = wilson_ci(0, 10)
    assert ci.lo == pytest.approx(0.0)
    assert ci.hi == pytest.approx(0.2775401687666166)
    # Agresti-Coull upper bound for the same data is higher - pin the distinction.
    assert ci.hi < 0.30
