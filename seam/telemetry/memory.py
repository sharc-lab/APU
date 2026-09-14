"""Available-memory, hard-page-read, and CPU telemetry for timed blocks."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class MemoryPressureSample:
    timestamp_utc: str
    available_memory_mb: float
    hard_page_reads_per_s: float | None
    hard_page_reads_method: str
    cpu_pct_total: float
    cpu_pct_per_core: list[float]


def _hard_page_reads_per_s() -> tuple[float | None, str]:
    """Read the Windows PDH Memory/Page Reads/sec counter."""
    try:
        import win32pdh  # type: ignore[import-untyped]

        query = win32pdh.OpenQuery()
        try:
            counter = win32pdh.AddCounter(query, r"\Memory\Page Reads/sec")
            win32pdh.CollectQueryData(query)
            time.sleep(0.05)
            win32pdh.CollectQueryData(query)
            _kind, value = win32pdh.GetFormattedCounterValue(counter, win32pdh.PDH_FMT_DOUBLE)
            return float(value), "win32pdh:Memory/Page Reads/sec"
        finally:
            win32pdh.CloseQuery(query)
    except Exception as exc:
        return None, f"unavailable:{type(exc).__name__}"


def sample_memory_pressure_once() -> MemoryPressureSample:
    import psutil

    per_core = [float(value) for value in psutil.cpu_percent(interval=None, percpu=True)]
    available = float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    page_reads, method = _hard_page_reads_per_s()
    return MemoryPressureSample(
        timestamp_utc=datetime.now(UTC).isoformat(),
        available_memory_mb=available,
        hard_page_reads_per_s=page_reads,
        hard_page_reads_method=method,
        cpu_pct_total=sum(per_core) / len(per_core) if per_core else 0.0,
        cpu_pct_per_core=per_core,
    )


def summarize_memory_pressure(
    samples: list[MemoryPressureSample],
    *,
    available_memory_min_mb: float,
    sustained_nonzero_samples: int,
    page_read_threshold: float = 0.0,
    gate_mode: str = "absolute_zero",
) -> dict[str, Any]:
    if sustained_nonzero_samples < 2:
        raise ValueError("sustained_nonzero_samples must be at least two")
    if page_read_threshold < 0.0:
        raise ValueError("page_read_threshold must be non-negative")
    if not samples:
        return {
            "samples": [],
            "valid": False,
            "invalid_reasons": ["memory-pressure sampler produced no samples"],
        }
    rates = [sample.hard_page_reads_per_s for sample in samples]
    longest_excess_run = 0
    current_run = 0
    for rate in rates:
        current_run = current_run + 1 if rate is not None and rate > page_read_threshold else 0
        longest_excess_run = max(longest_excess_run, current_run)
    minimum_memory = min(sample.available_memory_mb for sample in samples)
    invalid_reasons: list[str] = []
    if minimum_memory < available_memory_min_mb:
        invalid_reasons.append(
            f"available memory {minimum_memory:.3f} MiB < {available_memory_min_mb:.3f} MiB"
        )
    if any(rate is None for rate in rates):
        invalid_reasons.append("Windows hard page-read telemetry unavailable")
    if longest_excess_run >= sustained_nonzero_samples:
        invalid_reasons.append(
            "sustained hard page reads: "
            f"{longest_excess_run} consecutive samples > {page_read_threshold} "
            f"(gate_mode={gate_mode}) >= {sustained_nonzero_samples}"
        )
    return {
        "samples": [asdict(sample) for sample in samples],
        "sample_count": len(samples),
        "available_memory_mb_before": samples[0].available_memory_mb,
        "available_memory_mb_after": samples[-1].available_memory_mb,
        "available_memory_mb_min": minimum_memory,
        "hard_page_reads_per_s": rates,
        "hard_page_reads_method": sorted({sample.hard_page_reads_method for sample in samples}),
        "longest_consecutive_excess_page_read_samples": longest_excess_run,
        # Alias retained so existing readers of the absolute-zero wording still resolve.
        "longest_consecutive_nonzero_page_read_samples": longest_excess_run,
        "page_read_threshold_per_s": page_read_threshold,
        "gate_mode": gate_mode,
        "sustained_definition": (
            f">= {sustained_nonzero_samples} consecutive samples with "
            f"hard_page_reads_per_s > {page_read_threshold} (gate_mode={gate_mode})"
        ),
        "available_memory_min_mb": available_memory_min_mb,
        "cpu_pct_total": [sample.cpu_pct_total for sample in samples],
        "cpu_pct_per_core": [sample.cpu_pct_per_core for sample in samples],
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
    }


class MemoryPressureSampler:
    def __init__(
        self,
        *,
        interval_s: float,
        available_memory_min_mb: float,
        sustained_nonzero_samples: int,
        page_read_threshold: float = 0.0,
        gate_mode: str = "absolute_zero",
    ) -> None:
        self._interval_s = interval_s
        self._available_memory_min_mb = available_memory_min_mb
        self._sustained_nonzero_samples = sustained_nonzero_samples
        self._page_read_threshold = page_read_threshold
        self._gate_mode = gate_mode
        self._samples: list[MemoryPressureSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._samples.clear()
        self._stop.clear()
        # Synchronous bookend so a short block still has an explicit "before" sample.
        self._samples.append(sample_memory_pressure_once())
        self._thread = threading.Thread(
            target=self._run, name="memory-pressure-sample", daemon=True
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # Synchronous bookend for an explicit "after" sample.
        self._samples.append(sample_memory_pressure_once())
        return summarize_memory_pressure(
            self._samples,
            available_memory_min_mb=self._available_memory_min_mb,
            sustained_nonzero_samples=self._sustained_nonzero_samples,
            page_read_threshold=self._page_read_threshold,
            gate_mode=self._gate_mode,
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._samples.append(sample_memory_pressure_once())
