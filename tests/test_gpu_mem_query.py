"""Unit tests for harness/telemetry.py — vendor-aware GPU memory cascade.

Verifies:
  - fallback order: nvidia-smi → rocm-smi → intel-level-zero → intel-sysfs
    → unified-psutil (unified only) → (None, "unavailable")
  - total failure yields (None, "unavailable"), never (0.0, anything)
  - psutil fallback only fires on "unified" architecture, not "discrete"
  - each source literal is exactly one of the specified strings
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from harness.telemetry import (
    _query_intel_level_zero,
    _query_intel_sysfs,
    _query_nvidia_smi,
    _query_rocm_smi,
    _query_unified_psutil,
    gpu_mem_mb,
)

_VALID_SOURCES = {
    "nvidia-smi",
    "rocm-smi",
    "intel-level-zero",
    "intel-sysfs",
    "unified-psutil",
    "unavailable",
}


# ── Per-backend smoke tests ───────────────────────────────────────────────────

def test_query_nvidia_smi_success(monkeypatch) -> None:
    """nvidia-smi path returns (float, 'nvidia-smi') on success."""
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **kw: b"4096\n",
    )
    val, src = _query_nvidia_smi()
    assert val == pytest.approx(4096.0)
    assert src == "nvidia-smi"


def test_query_nvidia_smi_absent(monkeypatch) -> None:
    """Missing nvidia-smi returns (None, 'unavailable'), never 0.0."""
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()),
    )
    val, src = _query_nvidia_smi()
    assert val is None
    assert val != 0.0
    assert src == "unavailable"


def test_query_rocm_smi_success(monkeypatch) -> None:
    """rocm-smi path parses VRAM Total Used Memory line and returns MiB."""
    fake_out = (
        b"GPU[0]          : VRAM Total Memory (B): 137438953472\n"
        b"GPU[0]          : VRAM Total Used Memory (B): 536870912\n"
    )
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: fake_out)
    val, src = _query_rocm_smi()
    # 536870912 / 1024 / 1024 = 512.0 MiB
    assert val == pytest.approx(512.0, abs=0.5)
    assert src == "rocm-smi"


def test_query_rocm_smi_absent(monkeypatch) -> None:
    """Missing rocm-smi returns (None, 'unavailable'), never 0.0."""
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()),
    )
    val, src = _query_rocm_smi()
    assert val is None
    assert val != 0.0
    assert src == "unavailable"


def test_query_intel_level_zero_load_failure() -> None:
    """Level Zero falls through (unavailable) when ze_loader cannot be loaded."""
    import ctypes
    orig_cdll = ctypes.CDLL
    orig_windll = getattr(ctypes, "WinDLL", None)

    def _raise(*a, **kw):
        raise OSError("no such library")

    with patch.object(ctypes, "CDLL", side_effect=_raise):
        if orig_windll is not None:
            with patch.object(ctypes, "WinDLL", side_effect=_raise):
                val, src = _query_intel_level_zero()
        else:
            val, src = _query_intel_level_zero()

    assert val is None
    assert src == "unavailable"


def test_query_intel_sysfs_no_drm(monkeypatch) -> None:
    """Intel sysfs returns unavailable when DRM device path is absent."""
    import glob
    monkeypatch.setattr(glob, "glob", lambda *a, **kw: [])
    val, src = _query_intel_sysfs()
    assert val is None
    assert src == "unavailable"


def test_query_unified_psutil_success() -> None:
    """psutil path returns (float, 'unified-psutil') when psutil is available."""
    import psutil

    class _FakeVM:
        total = 64 * 1024 ** 3
        available = 32 * 1024 ** 3

    with patch.object(psutil, "virtual_memory", return_value=_FakeVM()):
        val, src = _query_unified_psutil()

    assert val == pytest.approx(32 * 1024.0, abs=1.0)  # 32 GiB in MiB
    assert src == "unified-psutil"


def test_query_unified_psutil_import_error(monkeypatch) -> None:
    """psutil unavailable returns (None, 'unavailable'), never 0.0."""
    import builtins
    real_import = builtins.__import__

    def _block(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)
    val, src = _query_unified_psutil()
    assert val is None
    assert val != 0.0
    assert src == "unavailable"


# ── Cascade order and fallback ────────────────────────────────────────────────

def _patch_all_fail(monkeypatch) -> None:
    """Patch every backend to fail."""
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()),
    )
    import ctypes, glob as _glob
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **kw: (_ for _ in ()).throw(OSError()))
    if hasattr(ctypes, "WinDLL"):
        monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(_glob, "glob", lambda *a, **kw: [])


def test_total_failure_discrete_returns_none_unavailable(monkeypatch) -> None:
    """When every backend fails on 'discrete', result is (None, 'unavailable'), never 0.0."""
    _patch_all_fail(monkeypatch)
    val, src = gpu_mem_mb("discrete")
    assert val is None
    assert val != 0.0
    assert src == "unavailable"


def test_total_failure_unified_with_psutil_blocked(monkeypatch) -> None:
    """When every backend including psutil fails on 'unified', result is (None, 'unavailable')."""
    _patch_all_fail(monkeypatch)
    import builtins
    real_import = builtins.__import__

    def _block(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)
    val, src = gpu_mem_mb("unified")
    assert val is None
    assert val != 0.0
    assert src == "unavailable"


def test_psutil_fallback_only_on_unified(monkeypatch) -> None:
    """psutil is tried on 'unified' but NOT on 'discrete' when earlier backends fail."""
    _patch_all_fail(monkeypatch)
    import psutil

    class _FakeVM:
        total = 64 * 1024 ** 3
        available = 32 * 1024 ** 3

    with patch.object(psutil, "virtual_memory", return_value=_FakeVM()):
        val_uni, src_uni = gpu_mem_mb("unified")
        val_dis, src_dis = gpu_mem_mb("discrete")

    assert val_uni is not None and src_uni == "unified-psutil"
    assert val_dis is None and src_dis == "unavailable"


def test_nvidia_smi_wins_when_present(monkeypatch) -> None:
    """If nvidia-smi succeeds, cascade stops at 'nvidia-smi'; later backends not reached."""
    monkeypatch.setattr("subprocess.check_output", lambda *a, **kw: b"3072\n")

    rocm_called = []

    def _fake_rocm() -> tuple[float | None, str]:
        rocm_called.append(True)
        return 9999.0, "rocm-smi"

    monkeypatch.setattr("harness.telemetry._query_rocm_smi", _fake_rocm)

    val, src = gpu_mem_mb("discrete")
    assert src == "nvidia-smi"
    assert val == pytest.approx(3072.0)
    assert not rocm_called, "rocm-smi must not be called when nvidia-smi succeeds"


def test_rocm_smi_wins_when_nvidia_absent(monkeypatch) -> None:
    """nvidia-smi absent → rocm-smi is tried and wins if it succeeds."""
    fake_rocm_out = (
        b"GPU[0]          : VRAM Total Used Memory (B): 1073741824\n"
    )

    call_count = [0]

    def _fake_subprocess(*a, **kw):
        call_count[0] += 1
        cmd = a[0]
        if "nvidia-smi" in cmd[0]:
            raise FileNotFoundError
        if "rocm-smi" in cmd[0]:
            return fake_rocm_out
        raise FileNotFoundError

    monkeypatch.setattr("subprocess.check_output", _fake_subprocess)
    val, src = gpu_mem_mb("discrete")
    assert src == "rocm-smi"
    assert val == pytest.approx(1024.0, abs=0.5)  # 1 GiB in MiB


# ── Source literal whitelist ──────────────────────────────────────────────────

@pytest.mark.parametrize("arch", ["discrete", "unified"])
def test_source_literal_is_valid(monkeypatch, arch: str) -> None:
    """gpu_mem_mb always returns a source literal from the defined set."""
    _patch_all_fail(monkeypatch)
    _, src = gpu_mem_mb(arch)
    assert src in _VALID_SOURCES or src.startswith("unavailable:"), (
        f"Unexpected source literal: {src!r}"
    )
