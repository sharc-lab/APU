"""High-resolution timing primitives for instrumentation."""

import time


def wall_ns() -> int:
    """High-resolution monotonic wall clock — ~100 ns on Windows."""
    return time.perf_counter_ns()


def process_cpu_ns() -> int:
    """Process CPU time. 15.6 ms resolution on Windows — session-level only.

    Used only for session-level measurements where coarse granularity is acceptable.
    Per-span timing uses wall_ns() as a CPU proxy for CPU-bound work.
    """
    return time.process_time_ns()
