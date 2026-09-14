"""Working-set lock ctypes path - must exercise GetProcessWorkingSetSizeEx on Windows.

The acceptance abort bd0261cd failed because ``read_working_set_limits`` called
``GetProcessWorkingSetSizeEx`` without ``argtypes`` while ``GetCurrentProcess`` returned a
64-bit pseudo-handle. The suite previously never called this path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from seam.tools.delta_n import _ceiling_refusal_message

_ROOT = Path(__file__).resolve().parents[1]
_DELTA_N = yaml.safe_load((_ROOT / "configs" / "delta_n.yaml").read_text(encoding="utf-8"))
_WS = _DELTA_N["working_set_lock"]
_MIN = int(_WS["minimum_bytes"])
_MAX = int(_WS["maximum_bytes"])

_windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows working-set APIs only")


@_windows_only
def test_read_working_set_limits_survives_pseudo_handle() -> None:
    """GetCurrentProcess returns INVALID_HANDLE_VALUE-shaped HANDLE; argtypes must accept it."""
    from seam.tools._winmem import read_working_set_limits

    limits = read_working_set_limits()
    assert limits.get("attempted") is not False  # not the non-Windows stub
    assert "reason" not in limits or "not Windows" not in str(limits.get("reason"))
    assert limits["read"] is True
    assert isinstance(limits["minimum_bytes"], int)
    assert isinstance(limits["maximum_bytes"], int)
    assert isinstance(limits["flags"], int)


@_windows_only
def test_lock_working_set_with_delta_n_bytes() -> None:
    """Exact configs/delta_n.yaml byte values - the path acceptance children take."""
    from seam.tools._winmem import lock_working_set

    assert _MIN == 4294967296
    assert _MAX == 12884901888
    record = lock_working_set(minimum_bytes=_MIN, maximum_bytes=_MAX)
    assert record["requested"] is True
    assert record["call"] == "SetProcessWorkingSetSizeEx"
    assert record["requested_minimum_bytes"] == _MIN
    assert record["requested_maximum_bytes"] == _MAX
    assert isinstance(record["before"], dict)
    assert record["before"].get("read") is True
    assert isinstance(record["after"], dict)
    # Grant depends on privilege; either outcome is a platform fact, not a test failure.
    assert "granted" in record
    assert "last_error" in record


def test_ceiling_refusal_distinguishes_deterministic_load_failure() -> None:
    msg = _ceiling_refusal_message(
        "armA.n2048.r0",
        [
            {
                "reason": "load_failure",
                "detail": "exception:ArgumentError",
                "exception_message": "argument 1: OverflowError: int too long to convert",
            },
            {
                "reason": "load_failure",
                "detail": "exception:ArgumentError",
                "exception_message": "argument 1: OverflowError: int too long to convert",
            },
            {
                "reason": "load_failure",
                "detail": "exception:ArgumentError",
                "exception_message": "argument 1: OverflowError: int too long to convert",
            },
        ],
        3,
    )
    assert "deterministic load_failure/exception" in msg
    assert "will not hold still" not in msg
    assert "OverflowError" in msg


def test_ceiling_refusal_keeps_quiescence_language_when_blocks_dominate() -> None:
    msg = _ceiling_refusal_message(
        "armA.n2048.r0",
        [
            {"reason": "quiescence_refusal", "detail": "busy"},
            {"reason": "quiescence_refusal", "detail": "busy"},
            {"reason": "load_failure", "detail": "exception:ArgumentError"},
        ],
        3,
    )
    assert "will not hold still" in msg
    assert "deterministic load_failure/exception" not in msg
