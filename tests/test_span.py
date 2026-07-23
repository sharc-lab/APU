"""Tests for Span accumulation and merging."""

import pytest
from harness.instrumentation import Span


def test_span_record():
    """Span.record() should accumulate values."""
    span = Span()
    span.record(cpu=1000, wall=2000, b_in=100, b_out=50)
    span.record(cpu=500, wall=1000, b_in=50, b_out=25)

    assert span.cpu_ns == 1500
    assert span.wall_ns == 3000
    assert span.bytes_in == 150
    assert span.bytes_out == 75
    assert span.count == 2


def test_span_merge():
    """Span.merge() should combine two spans."""
    s1 = Span()
    s1.record(cpu=1000, wall=2000, b_in=100, b_out=50)

    s2 = Span()
    s2.record(cpu=500, wall=1000, b_in=50, b_out=25)

    s1.merge(s2)

    assert s1.cpu_ns == 1500
    assert s1.wall_ns == 3000
    assert s1.bytes_in == 150
    assert s1.bytes_out == 75
    assert s1.count == 2


def test_span_to_dict():
    """Span.to_dict() should produce correct CategoryMetrics format."""
    span = Span()
    span.record(cpu=1000, wall=2000, b_in=100, b_out=50)

    d = span.to_dict()
    assert d["cpu_ns"] == 1000
    assert d["wall_ns"] == 2000
    assert d["bytes_in"] == 100
    assert d["bytes_out"] == 50
    assert d["count"] == 1


def test_span_initial_values():
    """Span should initialize to all zeros."""
    span = Span()
    assert span.cpu_ns == 0
    assert span.wall_ns == 0
    assert span.bytes_in == 0
    assert span.bytes_out == 0
    assert span.count == 0
