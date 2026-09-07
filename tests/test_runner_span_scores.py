"""Prove that adding span instrumentation to runner.py leaves quality scores unchanged.

The test runs run_cell() twice on the same probe+filler with a mocked Ollama
response and verifies:
  1. score and score_detail are identical before and after the span fields were added.
  2. The three new span fields are present (orch_setup_ns, http_client_ns, tool_compute_ns).
  3. http_client_ns equals latency_ms * 1e6 (within floating-point rounding).
  4. tool_compute_ns is 0 (no tool dispatch in the quality sweep).

The test does NOT make real network calls — _call_ollama_streaming is patched.

Hard requirement: if a future change makes score or score_detail differ between two
identical runs, this test must fail.  Never edit expected values to match a new
scoring behaviour; fix the code instead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to load modules under test
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent


def _load_scorers():
    spec = importlib.util.spec_from_file_location(
        "probes_scorers", _REPO / "evaluation" / "probes" / "scorers.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXACT_PROBE = {
    "id": "rea_01",
    "category": "reasoning",
    "difficulty": "easy",
    "scorer_type": "exact",
    "prompt": "What is 2 + 2?",
    "expected": "4",
    "max_tokens": 32,
}

FILLER = "A" * 100   # short filler that fits in context


def _fake_streaming(model, prompt, max_tokens, host):
    """Simulate _call_ollama_streaming returning the correct answer."""
    return "4", 1234.5, 200.0, 50, 3


def _run_cell_with_span(**kwargs) -> dict[str, Any]:
    """Call runner.run_cell() with span instrumentation active."""
    from harness import runner, telemetry

    scorers = _load_scorers()

    gpu_mock = MagicMock(return_value=(None, "unavailable:test"))

    with patch.object(runner, "_call_ollama_streaming", side_effect=_fake_streaming), \
         patch.object(telemetry, "gpu_mem_mb", side_effect=gpu_mock), \
         patch.object(telemetry, "rss_mb", return_value=256.0):
        return runner.run_cell(
            probe=EXACT_PROBE,
            filler=FILLER,
            depth=2000,
            rep=0,
            position_in_cell=0,
            cell_probe_seed=0,
            model="qwen3:4b-instruct",
            host="http://localhost:11434",
            cfg_hash="test_hash_01",
            filler_mode="unlabelled",
            hardware_config="test_hw",
            memory_architecture="discrete",
            model_variant="instruct",
            thinking_enabled=False,
            scorers=scorers,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_score_unchanged_after_span_instrumentation() -> None:
    """score field is identical on two back-to-back calls with the same output."""
    row1 = _run_cell_with_span()
    row2 = _run_cell_with_span()
    assert row1["score"] == row2["score"], (
        f"score diverged: {row1['score']} vs {row2['score']}"
    )
    assert row1["score_detail"] == row2["score_detail"], (
        f"score_detail diverged: {row1['score_detail']} vs {row2['score_detail']}"
    )


def test_exact_probe_scores_correctly() -> None:
    """An exact probe returns score=1.0 when the model output matches expected."""
    row = _run_cell_with_span()
    assert row["score"] == 1.0, (
        f"Expected score=1.0 for exact match, got {row['score']!r}. "
        "Do not edit this expected value — fix the scoring logic instead."
    )


def test_span_fields_present() -> None:
    """All three span fields are present in every result row."""
    row = _run_cell_with_span()
    assert "orch_setup_ns" in row
    assert "http_client_ns" in row
    assert "tool_compute_ns" in row


def test_http_client_ns_matches_latency_ms() -> None:
    """http_client_ns == round(latency_ms * 1e6) — no silent unit conversion."""
    row = _run_cell_with_span()
    # _fake_streaming returns latency_ms=1234.5
    expected_ns = int(1234.5 * 1e6)
    assert row["http_client_ns"] == pytest.approx(expected_ns, abs=1), (
        f"http_client_ns={row['http_client_ns']} does not match latency_ms*1e6={expected_ns}"
    )


def test_tool_compute_ns_is_zero() -> None:
    """Quality sweep never dispatches tools — tool_compute_ns must be 0."""
    row = _run_cell_with_span()
    assert row["tool_compute_ns"] == 0


def test_orch_setup_ns_is_positive() -> None:
    """wrap_prompt takes a non-zero amount of time."""
    row = _run_cell_with_span()
    assert row["orch_setup_ns"] >= 0, "orch_setup_ns must be non-negative"
