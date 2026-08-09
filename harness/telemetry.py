"""Per-call telemetry: wall time, TTFT, token counts, RSS, GPU memory."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class Telemetry:
    latency_ms: float
    ttft_ms: float
    tokens_in: int
    tokens_out: int
    mem_rss_mb: float
    gpu_mem_mb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Telemetry":
        return cls(**d)


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def gpu_mem_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return 0.0
