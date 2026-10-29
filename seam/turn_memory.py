"""Per-turn memory telemetry for hybrid local turns (DEMO-2, Task AS part 6).

TurnMemorySampler samples this process while a local turn runs:
- peak resident set (psutil rss) seen by a background thread at
  ``interval_s``, plus the value at turn end;
- the process peak working set (Windows ``peak_wset``; lifetime, monotonic);
- the lowest system available memory seen during the turn.

With ``gpu_counters=True`` on Windows it also reads the
``\\GPU Process Memory(pid_<pid>*)\\Shared Usage`` and ``Dedicated Usage``
counters once at turn end (one PowerShell Get-Counter call, outside the timed
generate). Any read that fails is recorded as an error string, never raised.

Off by default: the runner builds a sampler only with --mem-telemetry.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from typing import Any

MB = 1024 * 1024
GPU_COUNTERS = ("Shared Usage", "Dedicated Usage")


def _ps_counter_script(pid: int) -> str:
    paths = ",".join(f"'\\GPU Process Memory(pid_{pid}_*)\\{c}'" for c in GPU_COUNTERS)
    return (
        f"$s = Get-Counter -Counter @({paths}) -ErrorAction SilentlyContinue; "
        "if ($s) { $s.CounterSamples | ForEach-Object { $_.Path + '=' + [string]$_.CookedValue } }"
    )


def parse_gpu_counter_lines(text: str) -> dict[str, float]:
    """Sum CookedValue per counter name (one GPU process can have several instances)."""
    out: dict[str, float] = {}
    for line in (text or "").splitlines():
        if "=" not in line:
            continue
        path, _, raw = line.rpartition("=")
        name = next((c for c in GPU_COUNTERS if path.lower().endswith(c.lower())), None)
        if name is None:
            continue
        try:
            out[name] = out.get(name, 0.0) + float(raw)
        except ValueError:
            continue
    return {f"{k.lower().replace(' ', '_')}_mb": round(v / MB, 1) for k, v in out.items()}


def read_gpu_counters(pid: int, timeout_s: float = 15.0) -> dict[str, Any]:
    if sys.platform != "win32":
        return {"error": "not windows"}
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _ps_counter_script(pid)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    parsed = parse_gpu_counter_lines(proc.stdout)
    if not parsed:
        return {"error": "no GPU Process Memory samples", "returncode": proc.returncode}
    return parsed


class TurnMemorySampler:
    def __init__(self, *, interval_s: float = 0.1, gpu_counters: bool = False) -> None:
        import psutil

        self._psutil = psutil
        self._proc = psutil.Process(os.getpid())
        self.interval_s = float(interval_s)
        self.gpu_counters = bool(gpu_counters)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = 0
        self._min_avail: int | None = None
        self._t0 = 0.0

    def _sample(self) -> None:
        with contextlib.suppress(Exception):
            rss = int(self._proc.memory_info().rss)
            self._peak_rss = max(self._peak_rss, rss)
        with contextlib.suppress(Exception):
            avail = int(self._psutil.virtual_memory().available)
            self._min_avail = avail if self._min_avail is None else min(self._min_avail, avail)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:  # a previous turn raised before stop()
            self._stop.set()
            self._thread.join(timeout=5.0)
            self._thread = None
        self._stop.clear()
        self._peak_rss = 0
        self._min_avail = None
        self._t0 = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="turn-mem", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._sample()
        rec: dict[str, Any] = {
            "peak_rss_mb": round(self._peak_rss / MB, 1),
            "min_available_mb": None if self._min_avail is None else round(self._min_avail / MB, 1),
            "interval_s": self.interval_s,
            "sampled_s": round(time.perf_counter() - self._t0, 3),
        }
        try:
            mi = self._proc.memory_info()
            rec["end_rss_mb"] = round(int(mi.rss) / MB, 1)
            peak_wset = getattr(mi, "peak_wset", None)
            if peak_wset is not None:
                rec["process_peak_wset_mb"] = round(int(peak_wset) / MB, 1)
        except Exception as exc:
            rec["memory_info_error"] = f"{type(exc).__name__}: {exc}"
        if self.gpu_counters:
            rec["gpu_process_memory"] = read_gpu_counters(self._proc.pid)
        return rec
