"""Per-logical-CPU frequency sampling via Windows PDH.

Throttle detection for the confinement matrix uses frequency, not package temperature:
temperature is a proxy; frequency is the mechanism by which thermal state affects the
measurement (authorized 2026-08-02). Sampled at >=1 Hz alongside utilization.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

__all__ = ["FrequencySample", "FrequencySampler", "sample_frequencies_once"]


@dataclass(frozen=True, slots=True)
class FrequencySample:
    """One PDH snapshot of per-CPU frequency."""

    t_ns: int
    mhz_per_cpu: list[float | None]
    pct_of_max_per_cpu: list[float | None]
    method: str


def sample_frequencies_once(*, n_cpus: int = 8) -> FrequencySample:
    """Read ``\\Processor Information(*)\\Processor Frequency`` and ``% of Maximum Frequency``."""
    t0 = time.perf_counter_ns()
    mhz: list[float | None] = [None] * n_cpus
    pct: list[float | None] = [None] * n_cpus
    method = "unavailable"
    try:
        import win32pdh  # type: ignore[import-untyped]

        for counter_name, dest in (
            (r"\Processor Information(*)\Processor Frequency", mhz),
            (r"\Processor Information(*)\% of Maximum Frequency", pct),
        ):
            paths = win32pdh.ExpandCounterPath(counter_name)
            # Paths look like \\...\Processor Information(0,0)\... - map by first index.
            query = win32pdh.OpenQuery()
            handles: list[tuple[int, int]] = []
            try:
                for path in paths:
                    # Skip _Total aggregates.
                    if "_Total" in path:
                        continue
                    cpu_idx = _cpu_index_from_path(path)
                    if cpu_idx is None or cpu_idx >= n_cpus:
                        continue
                    h = win32pdh.AddCounter(query, path)
                    handles.append((cpu_idx, h))
                win32pdh.CollectQueryData(query)
                time.sleep(0.05)
                win32pdh.CollectQueryData(query)
                for cpu_idx, h in handles:
                    _typ, val = win32pdh.GetFormattedCounterValue(h, win32pdh.PDH_FMT_DOUBLE)
                    dest[cpu_idx] = float(val)
                method = "win32pdh"
            finally:
                win32pdh.CloseQuery(query)
    except Exception as exc:  # recorded by caller; never silent in the matrix report
        method = f"unavailable:{type(exc).__name__}"
    return FrequencySample(t_ns=t0, mhz_per_cpu=mhz, pct_of_max_per_cpu=pct, method=method)


def _cpu_index_from_path(path: str) -> int | None:
    """Extract logical CPU index from a Processor Information instance string."""
    # Typical: ...\Processor Information(0,0)\Processor Frequency  or (0,1)
    try:
        start = path.index("(") + 1
        end = path.index(")", start)
        instance = path[start:end]
        parts = instance.split(",")
        if len(parts) == 2:
            # (group, cpu) - on this platform group is usually 0 and cpu is logical index.
            return int(parts[1])
        return int(parts[0])
    except (ValueError, IndexError):
        return None


class FrequencySampler:
    """Background sampler collecting frequency at ``interval_s`` until stopped."""

    def __init__(self, *, interval_s: float = 1.0, n_cpus: int = 8) -> None:
        self._interval_s = interval_s
        self._n_cpus = n_cpus
        self._stop = threading.Event()
        self._samples: list[FrequencySample] = []
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._run, name="freq-sample", daemon=True)
        self._thread.start()

    def stop(self) -> list[FrequencySample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return list(self._samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._samples.append(sample_frequencies_once(n_cpus=self._n_cpus))
            self._stop.wait(self._interval_s)

    def summary(self) -> dict[str, Any]:
        if not self._samples:
            return {"n": 0, "method": "none"}
        methods = {s.method for s in self._samples}
        # Retain both PDH fields.  On Platform A they were shown inert/nominal by the recorded
        # idle-versus-burn check, so neither field is interpreted as an actual clock measurement.
        n = self._n_cpus
        pct_means: list[float | None] = []
        pct_mins: list[float | None] = []
        mhz_means: list[float | None] = []
        mhz_mins: list[float | None] = []
        for cpu in range(n):
            pct_vals = [
                float(v) for s in self._samples if (v := s.pct_of_max_per_cpu[cpu]) is not None
            ]
            mhz_vals = [float(v) for s in self._samples if (v := s.mhz_per_cpu[cpu]) is not None]
            pct_means.append(sum(pct_vals) / len(pct_vals) if pct_vals else None)
            pct_mins.append(min(pct_vals) if pct_vals else None)
            mhz_means.append(sum(mhz_vals) / len(mhz_vals) if mhz_vals else None)
            mhz_mins.append(min(mhz_vals) if mhz_vals else None)
        return {
            "n": len(self._samples),
            "methods": sorted(methods),
            "mean_pct_of_max_per_cpu": pct_means,
            "min_pct_of_max_per_cpu": pct_mins,
            "mean_mhz_per_cpu": mhz_means,
            "min_mhz_per_cpu": mhz_mins,
            "interpretation": (
                "PDH frequency counters are inert/nominal on Platform A under the recorded "
                "idle-versus-burn evidence; retained for audit, not actual-clock inference"
            ),
        }
