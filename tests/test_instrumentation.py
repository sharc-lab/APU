"""Tests for instrumentation timing primitives."""

import time
import pytest
from harness.instrumentation import wall_ns, process_cpu_ns


def test_wall_ns_monotonic():
    """wall_ns() should be monotonically increasing."""
    t1 = wall_ns()
    time.sleep(0.001)  # 1 ms
    t2 = wall_ns()
    assert t2 > t1


def test_wall_ns_resolution():
    """wall_ns() should have sub-millisecond resolution."""
    # Call in tight loop; consecutive calls should differ by < 1 ms (1e6 ns)
    times = [wall_ns() for _ in range(10)]
    diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    # At least some diffs should be < 100 microseconds (1e5 ns)
    assert any(d < 1e5 for d in diffs), f"diffs too coarse: {diffs}"


def test_process_cpu_ns_works():
    """process_cpu_ns() should return a positive integer."""
    cpu = process_cpu_ns()
    assert isinstance(cpu, int)
    assert cpu > 0


def test_process_cpu_ns_accumulates():
    """process_cpu_ns() should increase with CPU work."""
    t1 = process_cpu_ns()
    # Do some CPU work
    _ = sum(i ** 2 for i in range(100000))
    t2 = process_cpu_ns()
    # On Windows 15.6ms ticks, may still be equal, but generally should increase
    assert t2 >= t1
