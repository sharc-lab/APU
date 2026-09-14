"""Host environment capture + two-tier contending list for the Available-MBytes gate."""

from __future__ import annotations

import sys

import pytest

from seam.isolation import (
    CONTENDING_PROCESS_NAMES,
    TIER1_CONTENDING_PROCESS_NAMES,
    TIER2_CONTENDING_PROCESS_NAMES,
    _normalize_process_name,
)
from seam.telemetry import host_environment

REQUIRED_ENV_KEYS = {
    "available_mb",
    "available_mb_method",
    "process_count",
    "sum_private_mb",
    "pool_nonpaged_mb",
    "pool_paged_mb",
    "committed_mb",
    "uptime_s",
    "tier2_processes",
    "tier2_private_mb",
    "probe_error",
}


def test_normalize_process_name_adds_exe_and_lowercases() -> None:
    assert _normalize_process_name("Cursor") == "cursor.exe"

    assert _normalize_process_name("msedgewebview2.EXE") == "msedgewebview2.exe"

    assert _normalize_process_name("WorkloadsSessionHost") == "workloadssessionhost.exe"

    assert _normalize_process_name("SearchHost") == "searchhost.exe"


def test_tier_split_20260809() -> None:
    """Tier 1 refuses; tier 2 records. Union is CONTENDING_PROCESS_NAMES."""

    for name in (
        "cursor.exe",
        "code.exe",
        "chrome.exe",
        "msedge.exe",
        "vmmem.exe",
        "claude.exe",
    ):
        assert name in TIER1_CONTENDING_PROCESS_NAMES

        assert name not in TIER2_CONTENDING_PROCESS_NAMES

    for name in (
        "msedgewebview2.exe",
        "searchhost.exe",
        "widgets.exe",
        "workloadssessionhost.exe",
        "delloptimizer.systray.exe",
        "supportassistagent.exe",
        "icps.exe",
    ):
        assert name in TIER2_CONTENDING_PROCESS_NAMES

        assert name not in TIER1_CONTENDING_PROCESS_NAMES

        assert name in CONTENDING_PROCESS_NAMES

    assert TIER1_CONTENDING_PROCESS_NAMES.isdisjoint(TIER2_CONTENDING_PROCESS_NAMES)

    assert CONTENDING_PROCESS_NAMES == (
        TIER1_CONTENDING_PROCESS_NAMES | TIER2_CONTENDING_PROCESS_NAMES
    )


def test_capture_host_environment_shape() -> None:
    snap = host_environment.capture_host_environment()

    assert set(snap) >= REQUIRED_ENV_KEYS

    assert isinstance(snap["tier2_processes"], list)

    assert isinstance(snap["tier2_private_mb"], float)

    for row in snap["tier2_processes"]:
        assert "name" in row and "pid" in row and "private_working_set_bytes" in row

    if sys.platform == "win32":
        assert snap["available_mb"] is not None

        assert snap["available_mb"] > 0

        assert "Available MBytes" in snap["available_mb_method"] or snap[
            "available_mb_method"
        ].startswith("psutil")

        assert snap["process_count"] is not None and snap["process_count"] > 0

        assert snap["uptime_s"] is not None and snap["uptime_s"] > 0

    else:
        # Platform A measurement is Windows; Mac/Linux get a structured null probe.

        assert snap["available_mb"] is None or isinstance(snap["available_mb"], float)


def test_available_mb_now_returns_method() -> None:
    value, method = host_environment.available_mb_now()

    assert isinstance(method, str)

    if sys.platform == "win32":
        assert value is not None

        assert value > 0


def test_report_settle_adequacy_uses_available_mb() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools" / "report_gpu_only_matrix.py"

    spec = importlib.util.spec_from_file_location("report_gpu_only_matrix", path)

    assert spec and spec.loader

    mod = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(mod)

    cells = [
        {
            "arm": "A",
            "n_tokens": 12000,
            "record": {
                "classification": "OK",
                "available_mb_start": 7200.0,
                "free_physical_mb_start": 100.0,
            },
        },
        {
            "arm": "gpu_only",
            "n_tokens": 12000,
            "record": {
                "classification": "OK",
                "environment_start": {"available_mb": 7400.0},
                "free_physical_mb_start": 5000.0,
            },
        },
    ]

    result = mod.settle_adequacy(cells, n=12000, threshold_mb=512.0, provisional=True)

    assert result["evaluated"] is True

    assert result["metric"] == "available_mb"

    assert result["spread_mb"] == pytest.approx(200.0)

    # free_physical spread would be ~4900 MB and would fail; available spread is 200.

    assert result["pass"] is True
