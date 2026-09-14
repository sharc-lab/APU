"""Host memory / process environment snapshot for cell records and pre-run gates.

``available_mb`` is Windows ``\\Memory\\Available MBytes`` (PDH), not Win32
``FreePhysicalMemory``. A 2026-08-09 pair of gpu_only n=12000 sessions showed a
3.1 GB Available gap (4401 vs 7477 MB) moving prefill ~2.5x and turning a flat
series into monotonic degradation - while FreePhysical alone did not govern.

On non-Windows hosts the PDH path is unavailable; fields are null with
``probe_error`` set. Platform A measurement is Windows-only.
"""

from __future__ import annotations

import sys
import time
from typing import Any

__all__ = [
    "available_mb_now",
    "capture_host_environment",
]


def available_mb_now() -> tuple[float | None, str]:
    """Return (``\\Memory\\Available MBytes``, method). Prefer PDH; fall back to psutil."""
    if sys.platform == "win32":
        try:
            import win32pdh  # type: ignore[import-untyped]

            query = win32pdh.OpenQuery()
            try:
                counter = win32pdh.AddCounter(query, r"\Memory\Available MBytes")
                win32pdh.CollectQueryData(query)
                _kind, value = win32pdh.GetFormattedCounterValue(counter, win32pdh.PDH_FMT_DOUBLE)
                return float(value), "win32pdh:Memory/Available MBytes"
            finally:
                win32pdh.CloseQuery(query)
        except Exception as exc:
            pdh_err = f"win32pdh_failed:{type(exc).__name__}"
    else:
        pdh_err = "non_windows"

    try:
        import psutil

        # On Windows, ullAvailPhys ≈ Available MBytes; recorded as fallback only.
        mb = float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
        return mb, f"psutil.virtual_memory.available({pdh_err})"
    except Exception as exc:
        return None, f"unavailable:{pdh_err}:{type(exc).__name__}"


def _pdh_byte_counters_mb() -> dict[str, Any]:
    """Pool / committed bytes via PDH, converted to MiB."""
    out: dict[str, Any] = {
        "pool_nonpaged_mb": None,
        "pool_paged_mb": None,
        "committed_mb": None,
        "pdh_method": None,
        "pdh_error": None,
    }
    if sys.platform != "win32":
        out["pdh_error"] = "non_windows"
        return out
    try:
        import win32pdh  # type: ignore[import-untyped]

        paths = {
            "pool_nonpaged_mb": r"\Memory\Pool Nonpaged Bytes",
            "pool_paged_mb": r"\Memory\Pool Paged Bytes",
            "committed_mb": r"\Memory\Committed Bytes",
        }
        query = win32pdh.OpenQuery()
        try:
            handles: dict[str, Any] = {}
            for key, path in paths.items():
                handles[key] = win32pdh.AddCounter(query, path)
            win32pdh.CollectQueryData(query)
            for key, handle in handles.items():
                _kind, value = win32pdh.GetFormattedCounterValue(handle, win32pdh.PDH_FMT_DOUBLE)
                out[key] = float(value) / (1024.0 * 1024.0)
            out["pdh_method"] = "win32pdh:Memory/{Pool Nonpaged,Pool Paged,Committed} Bytes"
        finally:
            win32pdh.CloseQuery(query)
    except Exception as exc:
        out["pdh_error"] = f"{type(exc).__name__}:{exc}"
    return out


def _process_totals_and_tier2() -> (
    tuple[int | None, float | None, list[dict[str, Any]], str | None]
):
    """(process_count, sum_private_mb, tier2_processes, error).

    ``tier2_processes`` lists auto-respawning shell/vendor agents (record only; never refuse).
    See ``seam.isolation.TIER2_CONTENDING_PROCESS_NAMES``.
    """
    from seam.isolation import TIER2_CONTENDING_PROCESS_NAMES, _normalize_process_name

    try:
        import psutil
    except ImportError as exc:
        return None, None, [], f"psutil unavailable: {exc}"

    count = 0
    private_sum = 0.0
    tier2: list[dict[str, Any]] = []
    try:
        for process in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                count += 1
                info = process.info.get("memory_info")
                private_bytes: int | None = None
                if info is not None:
                    private = getattr(info, "private", None)
                    if private is None:
                        private = getattr(info, "rss", None)
                    if private is not None:
                        private_bytes = int(private)
                        private_sum += float(private_bytes) / (1024.0 * 1024.0)
                name = str(process.info.get("name") or "")
                if _normalize_process_name(name) in TIER2_CONTENDING_PROCESS_NAMES:
                    tier2.append(
                        {
                            "name": name,
                            "pid": int(process.info["pid"]),
                            "private_working_set_bytes": private_bytes,
                        }
                    )
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                count += 1
                continue
    except Exception as exc:  # pragma: no cover - platform-dependent
        return None, None, [], f"process enumeration failed: {exc}"
    tier2.sort(key=lambda row: (row["name"], row["pid"]))
    return count, private_sum, tier2, None


def capture_host_environment() -> dict[str, Any]:
    """One-shot host snapshot for ``environment_start`` / ``environment_peak``."""
    available, available_method = available_mb_now()
    pools = _pdh_byte_counters_mb()
    process_count, sum_private_mb, tier2_processes, proc_err = _process_totals_and_tier2()

    uptime_s: float | None = None
    uptime_error: str | None = None
    try:
        import psutil

        uptime_s = float(time.time() - psutil.boot_time())
    except Exception as exc:
        uptime_error = f"{type(exc).__name__}:{exc}"

    probe_parts = [p for p in (pools.get("pdh_error"), proc_err, uptime_error) if p]
    if available is None:
        probe_parts.insert(0, available_method)

    tier2_private_mb = (
        sum(
            (row.get("private_working_set_bytes") or 0) / (1024.0 * 1024.0)
            for row in tier2_processes
        )
        if tier2_processes
        else 0.0
    )

    return {
        "available_mb": available,
        "available_mb_method": available_method,
        "process_count": process_count,
        "sum_private_mb": sum_private_mb,
        "pool_nonpaged_mb": pools["pool_nonpaged_mb"],
        "pool_paged_mb": pools["pool_paged_mb"],
        "committed_mb": pools["committed_mb"],
        "uptime_s": uptime_s,
        # Tier-2 shell/vendor agents: record presence + private WS; never refuse on these.
        "tier2_processes": tier2_processes,
        "tier2_private_mb": tier2_private_mb,
        "probe_error": "; ".join(probe_parts) if probe_parts else None,
    }
