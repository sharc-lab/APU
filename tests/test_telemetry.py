"""Tests for harness/telemetry.py — None sentinel and unified-memory support."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from harness import telemetry
from harness.telemetry import Telemetry, _read_nvidia_smi, _read_unified_pool, gpu_mem_mb


# ── None-on-failure: no 0.0 as a proxy for unavailability ────────────────────

def test_nvidia_smi_not_found_returns_none_not_zero() -> None:
    """FileNotFoundError for nvidia-smi must yield None, never 0.0."""
    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        val, method = _read_nvidia_smi()
    assert val is None, f"Expected None, got {val!r}"
    assert method is not None and "nvidia_smi_not_found" in method


def test_nvidia_smi_generic_error_returns_none_not_zero() -> None:
    """Any other subprocess error must yield None, never 0.0."""
    with patch("subprocess.check_output", side_effect=RuntimeError("timeout")):
        val, method = _read_nvidia_smi()
    assert val is None
    assert method is not None and method.startswith("unavailable:")


def test_gpu_mem_mb_discrete_missing_smi_returns_none(monkeypatch) -> None:
    """gpu_mem_mb('discrete') with absent nvidia-smi returns (None, reason)."""
    monkeypatch.setattr(
        "subprocess.check_output", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError())
    )
    val, method = gpu_mem_mb("discrete")
    assert val is None
    assert isinstance(method, str) and method.startswith("unavailable:")


def test_gpu_mem_mb_unknown_architecture_returns_none() -> None:
    """An unknown memory_architecture must return (None, reason), not 0.0."""
    val, method = gpu_mem_mb("magnetic_tape")
    assert val is None
    assert method is not None and "unknown_architecture" in method


# ── unified-memory path ───────────────────────────────────────────────────────

def test_unified_pool_reads_proc_meminfo(tmp_path: Path) -> None:
    """_read_unified_pool parses /proc/meminfo and labels the method."""
    fake_meminfo = "MemTotal:       32768000 kB\nMemAvailable:   16384000 kB\n"
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = lambda s, *a: False
        mock_open.return_value.__iter__ = lambda s: iter(fake_meminfo.splitlines(keepends=True))
        val, method = _read_unified_pool()
    assert method == "unified_sys_pool"
    # (32768000 - 16384000) / 1024 = 16000.0
    assert val == pytest.approx(16000.0, abs=1.0)


def test_unified_pool_missing_proc_returns_none() -> None:
    """Missing /proc/meminfo returns (None, reason), not 0.0."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        val, method = _read_unified_pool()
    assert val is None
    assert method is not None and "proc_meminfo_not_found" in method


# ── Telemetry dataclass ───────────────────────────────────────────────────────

def test_telemetry_from_dict_tolerates_missing_gpu_method() -> None:
    """Old cached rows without gpu_mem_method deserialise without error."""
    d = {
        "latency_ms": 100.0, "ttft_ms": 50.0,
        "tokens_in": 10, "tokens_out": 5,
        "mem_rss_mb": 200.0, "gpu_mem_mb": 0.0,
        # gpu_mem_method absent — old cached row
    }
    t = Telemetry.from_dict(d)
    assert t.gpu_mem_mb == 0.0
    assert t.gpu_mem_method is None


def test_telemetry_gpu_mem_mb_default_is_none() -> None:
    """The dataclass default for gpu_mem_mb is None, not 0.0."""
    t = Telemetry(latency_ms=1.0, ttft_ms=0.5, tokens_in=1, tokens_out=1, mem_rss_mb=0.0)
    assert t.gpu_mem_mb is None
    assert t.gpu_mem_method is None
