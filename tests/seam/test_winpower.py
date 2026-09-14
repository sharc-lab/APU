"""PowerCreateRequest / PowerSetRequest path and modern-standby admissibility."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from seam.tools.delta_n import (
    _ceiling_refusal_message,
    _modern_standby_hits,
    _reject_kernel_power_ids,
)

_ROOT = Path(__file__).resolve().parents[1]
_DELTA_N = yaml.safe_load((_ROOT / "configs" / "delta_n.yaml").read_text(encoding="utf-8"))

_windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows power APIs only")


def test_delta_n_preregisters_modern_standby_reject_ids() -> None:
    adm = _DELTA_N["admissibility"]
    assert adm["max_block_retries"] == 3
    assert adm["modern_standby_inadmissible"] is True
    assert list(adm["reject_kernel_power_ids"]) == [506, 507]


def test_reject_kernel_power_ids_from_config() -> None:
    assert _reject_kernel_power_ids(_DELTA_N) == {506, 507}
    assert (
        _reject_kernel_power_ids({"admissibility": {"modern_standby_inadmissible": False}}) == set()
    )
    assert _reject_kernel_power_ids({"admissibility": {}}) == set()


def test_modern_standby_hits_filters_reject_ids() -> None:
    cfg = {
        "admissibility": {
            "modern_standby_inadmissible": True,
            "reject_kernel_power_ids": [506, 507],
        }
    }
    fake = {
        "events": [
            {"id": 506, "time_utc": "2026-08-06T21:11:08.244Z", "message_head": "enter"},
            {"id": 566, "time_utc": "2026-08-06T21:11:08.300Z", "message_head": "session"},
            {"id": 507, "time_utc": "2026-08-06T21:15:16.974Z", "message_head": "exit"},
        ]
    }
    with patch(
        "seam.tools.acceptance_instrumentation.collect_kernel_power_events",
        return_value=fake,
    ):
        hits = _modern_standby_hits(
            cfg=cfg,
            started_utc="2026-08-06T21:10:00+00:00",
            ended_utc="2026-08-06T21:16:00+00:00",
        )
    assert [int(h["id"]) for h in hits] == [506, 507]


def test_ceiling_refusal_names_modern_standby() -> None:
    msg = _ceiling_refusal_message(
        "fixed-throughput/2048/2",
        [
            {"reason": "modern_standby_in_window", "detail": "506"},
            {"reason": "modern_standby_in_window", "detail": "506"},
            {"reason": "quiescence_refusal", "detail": "x"},
        ],
        3,
    )
    assert "modern_standby_in_window" in msg
    assert "platform fact" in msg


@_windows_only
def test_power_request_create_set_clear_roundtrip() -> None:
    from seam.tools._winpower import assert_system_required, system_required

    req = assert_system_required(reason="SEAM test_winpower roundtrip", role="unit_test")
    try:
        assert req.record["created"] is True
        assert req.record["set"] is True
        assert req.record["succeeded"] is True
        assert req.record["mechanism"] == "PowerCreateRequest/PowerSetRequest"
        assert req.record["request_type"] == "PowerRequestSystemRequired"
        assert req.record["error"] is None
    finally:
        released = req.release()
    assert released["cleared"] is True
    assert released["closed"] is True

    with system_required(reason="SEAM test_winpower cm", role="unit_test_cm") as record:
        assert record["succeeded"] is True


def test_stats_excludes_none_only_present_values() -> None:
    """Admissible-only stats: None rates (missing) stay out of the CV sample."""
    from seam.tools.fixed_throughput import _stats

    stats = _stats([10.0, None, 12.0])
    assert stats["n"] == 2
    assert stats["n_missing"] == 1
    assert stats["median"] == 11.0
