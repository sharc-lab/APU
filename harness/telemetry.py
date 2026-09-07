"""Per-call telemetry: wall time, TTFT, token counts, RSS, GPU/unified memory."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, fields


@dataclass
class Telemetry:
    latency_ms: float
    ttft_ms: float
    tokens_in: int
    tokens_out: int
    mem_rss_mb: float
    gpu_mem_mb: float | None = None
    gpu_mem_method: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Telemetry":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def _read_nvidia_smi() -> tuple[float | None, str | None]:
    """Query nvidia-smi for discrete GPU memory usage.

    Returns (used_mb, "nvidia_smi") on success.
    Returns (None, "unavailable:<reason>") on any failure so that a missing
    nvidia-smi is never reported as 0 MB used.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip().split("\n")[0]), "nvidia_smi"
    except FileNotFoundError:
        return None, "unavailable:nvidia_smi_not_found"
    except Exception as exc:
        return None, f"unavailable:{type(exc).__name__}"


def _read_unified_pool() -> tuple[float | None, str | None]:
    """Read system-wide used memory from /proc/meminfo for unified-memory hosts.

    On unified-memory systems (e.g. AMD Strix Halo LPDDR5X) the GPU and CPU
    share one physical pool; there is no separate VRAM counter.  /proc/meminfo
    MemTotal - MemAvailable is the in-use share of the shared pool.  This is
    not GPU-only memory — it includes OS, all processes, and GPU allocations.
    rocm-smi VRAM reporting for Strix Halo is not yet reliable (as of
    2026-09); this is the best available proxy.  The method field labels the
    measurement so analysis code can distinguish it from discrete-GPU readings.

    Returns (None, "unavailable:<reason>") on any failure.
    """
    try:
        info: dict[str, str] = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = v.strip()
        total_kb = int(info["MemTotal"].split()[0])
        avail_kb = int(info["MemAvailable"].split()[0])
        used_mb = (total_kb - avail_kb) / 1024
        return round(used_mb, 1), "unified_sys_pool"
    except FileNotFoundError:
        return None, "unavailable:proc_meminfo_not_found"
    except Exception as exc:
        return None, f"unavailable:{type(exc).__name__}"


def gpu_mem_mb(memory_architecture: str = "discrete") -> tuple[float | None, str | None]:
    """Sample memory usage, returning (used_mb, method).

    Returns (None, reason) when a measurement is unavailable; never returns
    0.0 as a proxy for unavailability.

    memory_architecture:
      "discrete" — query nvidia-smi for dedicated VRAM.
      "unified"  — read the whole-system used pool from /proc/meminfo.
                   Reports shared GPU+CPU pool, not GPU-only memory.
      anything else — returns (None, "unavailable:unknown_architecture").
    """
    if memory_architecture == "unified":
        return _read_unified_pool()
    if memory_architecture == "discrete":
        return _read_nvidia_smi()
    return None, f"unavailable:unknown_architecture:{memory_architecture}"
