"""Process RSS high-water sampling for one agent step.



The analytic KV figure (``seam.kvmath``) excludes weights, activations, allocator slack, and any

runtime-internal copies. Observed RSS includes all of them. Recording both per step is what makes

"analytic" and "observed" comparable instead of interchangeable.



Peak RSS is sampled rather than read at step end: the high-water mark during prefill is the number

a capacity budget has to cover, and it is gone by the time the step returns.



C2g also samples host free memory (``virtual_memory().available``) on the same cadence so the

per-task free-memory minimum can be attributed to a step and context length.

"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

__all__ = [
    "RssSampler",
    "RssWindow",
    "available_mb_now",
    "commit_bytes_now",
    "free_memory_mb_now",
    "free_physical_mb_now",
    "rss_bytes_now",
]


def available_mb_now() -> float | None:
    """Host ``\\Memory\\Available MBytes`` (PDH-preferred); see host_environment."""

    from seam.telemetry.host_environment import available_mb_now as _available

    value, _method = _available()
    return value


def rss_bytes_now() -> int:
    """Current process resident set size in bytes."""

    import psutil

    return int(psutil.Process().memory_info().rss)


def commit_bytes_now() -> int:
    """Current process commit charge in bytes.

    On Windows ``vms`` is the pagefile / commit usage for the process; elsewhere it is the
    address-space size. Either way it is the additive peak-commit metric the acceptance gate
    records alongside peak working set.
    """

    import psutil

    return int(psutil.Process().memory_info().vms)


def free_memory_mb_now() -> float:
    """Host available memory in MiB (psutil ``virtual_memory().available``)."""

    import psutil

    return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)


def free_physical_mb_now() -> float:
    """Host free physical memory in MiB (psutil ``virtual_memory().free``)."""

    import psutil

    return float(psutil.virtual_memory().free) / (1024.0 * 1024.0)


@dataclass(frozen=True, slots=True)
class RssWindow:
    """RSS / commit / free-memory statistics over one sampling window."""

    peak_bytes: int

    start_bytes: int

    end_bytes: int

    n_samples: int

    interval_s: float

    sampler_overhead_s: float

    free_memory_mb_min: float | None = None

    free_memory_mb_start: float | None = None

    free_memory_mb_end: float | None = None

    peak_commit_bytes: int | None = None

    commit_bytes_start: int | None = None

    commit_bytes_end: int | None = None

    free_physical_mb_start: float | None = None

    free_physical_mb_at_peak: float | None = None

    free_physical_mb_min: float | None = None

    available_mb_start: float | None = None

    available_mb_at_peak: float | None = None

    available_mb_min: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "peak_rss_bytes": self.peak_bytes,
            "rss_bytes_start": self.start_bytes,
            "rss_bytes_end": self.end_bytes,
            "n_rss_samples": self.n_samples,
            "rss_interval_s": self.interval_s,
            "rss_sampler_overhead_s": self.sampler_overhead_s,
            "free_memory_mb_min": self.free_memory_mb_min,
            "free_memory_mb_start": self.free_memory_mb_start,
            "free_memory_mb_end": self.free_memory_mb_end,
            "peak_commit_bytes": self.peak_commit_bytes,
            "commit_bytes_start": self.commit_bytes_start,
            "commit_bytes_end": self.commit_bytes_end,
            "free_physical_mb_start": self.free_physical_mb_start,
            "free_physical_mb_at_peak": self.free_physical_mb_at_peak,
            "free_physical_mb_min": self.free_physical_mb_min,
            "available_mb_start": self.available_mb_start,
            "available_mb_at_peak": self.available_mb_at_peak,
            "available_mb_min": self.available_mb_min,
        }


class RssSampler:
    """Sample process RSS (and host free memory) on a background thread.



    A window always contains at least the two endpoint samples, so ``peak_bytes`` is never zero for

    a live process: a zero would be indistinguishable from "sampling failed", and the step record

    has to be able to tell those apart.

    """

    def __init__(self, *, interval_s: float = 0.05) -> None:
        self._interval_s = float(interval_s)

        self._stop = threading.Event()

        self._thread: threading.Thread | None = None

        self._peak = 0

        self._start = 0

        self._n = 0

        self._overhead_ns = 0

        self._free_start: float | None = None

        self._free_min: float | None = None

        self._free_end: float | None = None

        self._peak_commit = 0

        self._commit_start = 0

        self._free_phys_start: float | None = None

        self._free_phys_min: float | None = None

        self._free_phys_at_peak: float | None = None

        self._available_start: float | None = None

        self._available_min: float | None = None

        self._available_at_peak: float | None = None

    def start(self) -> None:
        self._stop.clear()

        self._start = rss_bytes_now()

        self._peak = self._start

        self._commit_start = commit_bytes_now()

        self._peak_commit = self._commit_start

        self._n = 1

        self._overhead_ns = 0

        free0 = free_memory_mb_now()

        self._free_start = free0

        self._free_min = free0

        self._free_end = free0

        phys0 = free_physical_mb_now()

        self._free_phys_start = phys0

        self._free_phys_min = phys0

        self._free_phys_at_peak = phys0

        avail0 = available_mb_now()

        self._available_start = avail0

        self._available_min = avail0

        self._available_at_peak = avail0

        self._thread = threading.Thread(target=self._run, name="rss-sample", daemon=True)

        self._thread.start()

    def stop(self) -> RssWindow:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

            self._thread = None

        end = rss_bytes_now()

        commit_end = commit_bytes_now()

        if end >= self._peak:
            self._peak = end

            self._free_phys_at_peak = free_physical_mb_now()

            self._available_at_peak = available_mb_now()

        self._peak_commit = max(self._peak_commit, commit_end)

        self._n += 1

        free_end = free_memory_mb_now()

        self._free_end = free_end

        if self._free_min is None:
            self._free_min = free_end

        else:
            self._free_min = min(self._free_min, free_end)

        phys_end = free_physical_mb_now()

        if self._free_phys_min is None:
            self._free_phys_min = phys_end

        else:
            self._free_phys_min = min(self._free_phys_min, phys_end)

        avail_end = available_mb_now()

        if avail_end is not None:
            if self._available_min is None:
                self._available_min = avail_end
            else:
                self._available_min = min(self._available_min, avail_end)

        return RssWindow(
            peak_bytes=self._peak,
            start_bytes=self._start,
            end_bytes=end,
            n_samples=self._n,
            interval_s=self._interval_s,
            sampler_overhead_s=self._overhead_ns / 1e9,
            free_memory_mb_min=self._free_min,
            free_memory_mb_start=self._free_start,
            free_memory_mb_end=self._free_end,
            peak_commit_bytes=self._peak_commit,
            commit_bytes_start=self._commit_start,
            commit_bytes_end=commit_end,
            free_physical_mb_start=self._free_phys_start,
            free_physical_mb_at_peak=self._free_phys_at_peak,
            free_physical_mb_min=self._free_phys_min,
            available_mb_start=self._available_start,
            available_mb_at_peak=self._available_at_peak,
            available_mb_min=self._available_min,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter_ns()

            try:
                value = rss_bytes_now()

                commit = commit_bytes_now()

                free_mb = free_memory_mb_now()

                free_phys = free_physical_mb_now()

                avail = available_mb_now()

            except Exception:  # a dead/denied handle ends sampling; the window still reports n
                return

            self._overhead_ns += time.perf_counter_ns() - t0

            if value >= self._peak:
                self._peak = value

                self._free_phys_at_peak = free_phys

                self._available_at_peak = avail

            self._peak_commit = max(self._peak_commit, commit)

            self._n += 1

            if self._free_min is None:
                self._free_min = free_mb

            else:
                self._free_min = min(self._free_min, free_mb)

            if self._free_phys_min is None:
                self._free_phys_min = free_phys

            else:
                self._free_phys_min = min(self._free_phys_min, free_phys)

            if avail is not None:
                if self._available_min is None:
                    self._available_min = avail
                else:
                    self._available_min = min(self._available_min, avail)

            self._stop.wait(self._interval_s)
