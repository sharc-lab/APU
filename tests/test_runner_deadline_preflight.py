"""Tests for runner.py deadline and preflight features."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from harness.runner import (
    _parse_duration,
    _deadline_exceeded,
    _run_preflight,
)


# ── _parse_duration ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("s,expected", [
    ("6h30m",  6 * 3600 + 30 * 60),
    ("2h",     2 * 3600),
    ("45m",    45 * 60),
    ("90s",    90),
    ("1h0m0s", 3600),
])
def test_parse_duration_valid(s: str, expected: float) -> None:
    assert _parse_duration(s) == pytest.approx(expected)


@pytest.mark.parametrize("s", ["", "abc", "6", "6h30", "-1h"])
def test_parse_duration_invalid(s: str) -> None:
    with pytest.raises(ValueError):
        _parse_duration(s)


def test_parse_duration_zero_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        _parse_duration("0h0m0s")


# ── _deadline_exceeded ────────────────────────────────────────────────────────

def test_deadline_none_never_fires() -> None:
    assert _deadline_exceeded(None, time.monotonic()) is False


def test_deadline_not_yet_exceeded() -> None:
    run_start = time.monotonic()
    assert _deadline_exceeded(3600.0, run_start) is False


def test_deadline_exceeded() -> None:
    # Pretend run started 10 s ago with a 1 s deadline.
    run_start = time.monotonic() - 10.0
    assert _deadline_exceeded(1.0, run_start) is True


def test_deadline_exactly_at_boundary() -> None:
    # At exactly deadline_s elapsed, should be True (>=).
    run_start = time.monotonic() - 60.0
    assert _deadline_exceeded(60.0, run_start) is True


# ── _run_preflight ────────────────────────────────────────────────────────────

def _good_probe() -> dict:
    return {"id": "rea_01", "scorer_type": "exact", "prompt": "q", "expected": "a"}


def test_preflight_passes_when_everything_ok(tmp_path: Path) -> None:
    """All checks pass when Ollama is reachable, disk/memory plentiful, probes present."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "qwen3:4b-instruct"}]}

    with patch("harness.runner.shutil.disk_usage") as mock_du, \
         patch("harness.runner.httpx.get", return_value=mock_resp):
        mock_du.return_value = MagicMock(free=100 * 1024 ** 3)  # 100 GB
        try:
            import psutil
            with patch.object(psutil, "virtual_memory") as mock_vm:
                mock_vm.return_value = MagicMock(available=32 * 1024 ** 3)
                failures = _run_preflight(
                    host="http://localhost:11434",
                    model="qwen3:4b-instruct",
                    evalset_path=tmp_path / "prompts.jsonl",
                    probes=[_good_probe()],
                    min_disk_gb=1.0,
                    min_mem_gb=2.0,
                )
        except ImportError:
            failures = _run_preflight(
                host="http://localhost:11434",
                model="qwen3:4b-instruct",
                evalset_path=tmp_path / "prompts.jsonl",
                probes=[_good_probe()],
                min_disk_gb=1.0,
                min_mem_gb=2.0,
            )

    assert failures == []


def test_preflight_fails_when_ollama_unreachable(tmp_path: Path) -> None:
    """Server unreachable produces a failure message."""
    with patch("harness.runner.httpx.get", side_effect=Exception("connection refused")), \
         patch("harness.runner.shutil.disk_usage") as mock_du:
        mock_du.return_value = MagicMock(free=100 * 1024 ** 3)
        failures = _run_preflight(
            host="http://localhost:11434",
            model="qwen3:4b-instruct",
            evalset_path=tmp_path / "prompts.jsonl",
            probes=[_good_probe()],
            min_disk_gb=1.0,
            min_mem_gb=2.0,
        )

    assert any("not reachable" in f or "Ollama" in f for f in failures)


def test_preflight_fails_when_model_not_in_tags(tmp_path: Path) -> None:
    """Model absent from /api/tags produces a failure message."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "some_other_model"}]}

    with patch("harness.runner.httpx.get", return_value=mock_resp), \
         patch("harness.runner.shutil.disk_usage") as mock_du:
        mock_du.return_value = MagicMock(free=100 * 1024 ** 3)
        failures = _run_preflight(
            host="http://localhost:11434",
            model="qwen3:4b-instruct",
            evalset_path=tmp_path / "prompts.jsonl",
            probes=[_good_probe()],
            min_disk_gb=1.0,
            min_mem_gb=2.0,
        )

    assert any("not found" in f for f in failures)


def test_preflight_fails_when_disk_too_low(tmp_path: Path) -> None:
    """Insufficient disk space produces a failure message."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "qwen3:4b-instruct"}]}

    with patch("harness.runner.httpx.get", return_value=mock_resp), \
         patch("harness.runner.shutil.disk_usage") as mock_du:
        mock_du.return_value = MagicMock(free=50 * 1024 ** 2)  # 50 MB — below 1 GB threshold
        failures = _run_preflight(
            host="http://localhost:11434",
            model="qwen3:4b-instruct",
            evalset_path=tmp_path / "prompts.jsonl",
            probes=[_good_probe()],
            min_disk_gb=1.0,
            min_mem_gb=2.0,
        )

    assert any("disk" in f.lower() for f in failures)


def test_preflight_fails_when_probes_empty(tmp_path: Path) -> None:
    """Zero probes after filtering produces a failure message."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "qwen3:4b-instruct"}]}

    with patch("harness.runner.httpx.get", return_value=mock_resp), \
         patch("harness.runner.shutil.disk_usage") as mock_du:
        mock_du.return_value = MagicMock(free=100 * 1024 ** 3)
        failures = _run_preflight(
            host="http://localhost:11434",
            model="qwen3:4b-instruct",
            evalset_path=tmp_path / "prompts.jsonl",
            probes=[],              # empty after filter
            min_disk_gb=1.0,
            min_mem_gb=2.0,
        )

    assert any("zero probes" in f for f in failures)


def test_preflight_multiple_failures_reported(tmp_path: Path) -> None:
    """When multiple checks fail, all failures are returned."""
    with patch("harness.runner.httpx.get", side_effect=Exception("refused")), \
         patch("harness.runner.shutil.disk_usage") as mock_du:
        mock_du.return_value = MagicMock(free=0)  # 0 bytes free
        failures = _run_preflight(
            host="http://localhost:11434",
            model="qwen3:4b-instruct",
            evalset_path=tmp_path / "prompts.jsonl",
            probes=[],
            min_disk_gb=1.0,
            min_mem_gb=2.0,
        )

    assert len(failures) >= 2  # server + disk + probes
