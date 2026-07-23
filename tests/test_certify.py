"""Tests for sampled certification helpers."""

from evaluation.certify import _wilson_interval


def test_wilson_interval_bounds_and_order():
    low, high = _wilson_interval(7, 10)
    assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_empty_samples():
    low, high = _wilson_interval(0, 0)
    assert low == 0.0
    assert high == 0.0
