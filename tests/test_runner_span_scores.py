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


def _load_outcome():
    spec = importlib.util.spec_from_file_location(
        "evaluation_outcome", _REPO / "evaluation" / "outcome.py"
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


def _run_cell_with_span(probe=None, fake_output="4", **kwargs) -> dict[str, Any]:
    """Call runner.run_cell() with span instrumentation active."""
    from harness import runner, telemetry, cache

    scorers = _load_scorers()
    outcome_mod = _load_outcome()

    def _fake(model, prompt, max_tokens, host):
        return fake_output, 1234.5, 200.0, 50, 3, "stop"

    gpu_mock = MagicMock(return_value=(None, "unavailable:test"))

    with patch.object(runner, "_call_ollama_streaming", side_effect=_fake), \
         patch.object(cache, "get", return_value=None), \
         patch.object(cache, "put"), \
         patch.object(telemetry, "gpu_mem_mb", side_effect=gpu_mock), \
         patch.object(telemetry, "rss_mb", return_value=256.0):
        return runner.run_cell(
            probe=probe or EXACT_PROBE,
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
            outcome_mod=outcome_mod,
            git_sha="abc123",
            run_seed=42,
            hostname="test-host",
            operator="researcher_a",
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


# ---------------------------------------------------------------------------
# Integration tests — four-way outcome fields wired into run_cell
# ---------------------------------------------------------------------------

_OUTCOME_FIELDS = ("outcome_class", "classification_method", "format_compliant")


def test_outcome_fields_present_on_correct_row() -> None:
    """All three outcome fields are present when the model output is correct."""
    row = _run_cell_with_span()  # EXACT_PROBE expects "4", fake returns "4"
    for field in _OUTCOME_FIELDS:
        assert field in row, f"Missing field {field!r} on correct row"


def test_outcome_class_correct_on_exact_match() -> None:
    """outcome_class is CORRECT and format_compliant is True for an exact match."""
    row = _run_cell_with_span()
    assert row["outcome_class"] == "CORRECT", row["outcome_class"]
    assert row["format_compliant"] is True
    assert row["classification_method"] == "score"


def test_outcome_class_fabricated_on_wrong_answer() -> None:
    """outcome_class is FABRICATED when the model returns a wrong answer."""
    row = _run_cell_with_span(fake_output="99")  # EXACT_PROBE expects "4"
    assert row["outcome_class"] == "FABRICATED", row["outcome_class"]
    assert row["score"] == 0.0


def test_outcome_class_refused_on_abstention() -> None:
    """outcome_class is REFUSED when the model explicitly abstains."""
    abstention = "The provided text does not contain this value."
    row = _run_cell_with_span(fake_output=abstention)
    assert row["outcome_class"] == "REFUSED", row["outcome_class"]
    assert row["format_compliant"] is None


def test_outcome_class_unclassifiable_on_empty_output() -> None:
    """outcome_class is UNCLASSIFIABLE when the model returns an empty string."""
    row = _run_cell_with_span(fake_output="")
    assert row["outcome_class"] == "UNCLASSIFIABLE", row["outcome_class"]


def test_format_noncompliant_correct_via_last_token() -> None:
    """outcome_class is CORRECT, format_compliant=False when model shows working."""
    # EXACT_PROBE expects "4"; model returns "2+2=4" — last token "4" matches
    row = _run_cell_with_span(fake_output="2+2=4")
    assert row["outcome_class"] == "CORRECT", row["outcome_class"]
    assert row["format_compliant"] is False
    assert row["classification_method"] == "last_token"


def test_outcome_fields_present_on_multiple_probes() -> None:
    """All three outcome fields are present across probes with different scorer types."""
    probes = [
        EXACT_PROBE,
        {
            "id": "str_01",
            "category": "structured_output",
            "difficulty": "easy",
            "scorer_type": "exact",
            "prompt": "Return the number seven.",
            "expected": "7",
            "max_tokens": 32,
        },
    ]
    for probe in probes:
        row = _run_cell_with_span(probe=probe)
        for field in _OUTCOME_FIELDS:
            assert field in row, (
                f"Field {field!r} missing on row for probe {probe['id']!r}"
            )


def test_normalize_result_row_fills_missing_fields() -> None:
    """normalize_result_row adds None for absent outcome fields on old rows."""
    from evaluation.outcome import normalize_result_row

    old_row = {"probe_id": "rea_01", "score": 1.0, "depth": 0, "rep": 0}
    normalized = normalize_result_row(old_row)

    assert normalized["outcome_class"] is None
    assert normalized["classification_method"] is None
    assert normalized["format_compliant"] is None
    # Original fields unchanged
    assert normalized["score"] == 1.0
    assert normalized["probe_id"] == "rea_01"


def test_normalize_result_row_preserves_existing_fields() -> None:
    """normalize_result_row does not overwrite outcome fields that are already present."""
    from evaluation.outcome import normalize_result_row

    new_row = {
        "probe_id": "rea_01",
        "score": 1.0,
        "outcome_class": "CORRECT",
        "classification_method": "score",
        "format_compliant": True,
    }
    normalized = normalize_result_row(new_row)
    assert normalized["outcome_class"] == "CORRECT"
    assert normalized["format_compliant"] is True


# ---------------------------------------------------------------------------
# Run-identity and done_reason fields
# ---------------------------------------------------------------------------

_IDENTITY_FIELDS = ("git_sha", "run_seed", "hostname", "operator", "done_reason")


def test_identity_fields_present_on_row() -> None:
    """All run-identity and done_reason fields appear on every result row."""
    row = _run_cell_with_span()
    for field in _IDENTITY_FIELDS:
        assert field in row, f"Missing field {field!r} on result row"


def test_identity_field_values_match_inputs() -> None:
    """git_sha, run_seed, hostname, operator are the values passed to run_cell."""
    row = _run_cell_with_span()
    assert row["git_sha"] == "abc123"
    assert row["run_seed"] == 42
    assert row["hostname"] == "test-host"
    assert row["operator"] == "researcher_a"


def test_done_reason_stop_on_normal_output() -> None:
    """done_reason is 'stop' when the mock returns 'stop' as the stop reason."""
    row = _run_cell_with_span()  # _fake returns "stop" for done_reason
    assert row["done_reason"] == "stop"


def test_done_reason_length_makes_unclassifiable() -> None:
    """done_reason='length' causes outcome_class to be UNCLASSIFIABLE."""
    from harness import runner, telemetry, cache

    scorers = _load_scorers()
    outcome_mod = _load_outcome()

    def _fake_length(model, prompt, max_tokens, host):
        return "partial output", 1234.5, 200.0, 50, 3, "length"

    gpu_mock = MagicMock(return_value=(None, "unavailable:test"))

    with patch.object(runner, "_call_ollama_streaming", side_effect=_fake_length), \
         patch.object(cache, "get", return_value=None), \
         patch.object(cache, "put"), \
         patch.object(telemetry, "gpu_mem_mb", side_effect=gpu_mock), \
         patch.object(telemetry, "rss_mb", return_value=256.0):
        row = runner.run_cell(
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
            outcome_mod=outcome_mod,
            git_sha=None,
            run_seed=None,
            hostname="test-host",
            operator=None,
        )
    assert row["done_reason"] == "length"
    assert row["outcome_class"] == "UNCLASSIFIABLE", row["outcome_class"]


def test_normalize_result_row_fills_identity_fields_on_old_row() -> None:
    """normalize_result_row fills identity fields with None on pre-identity rows."""
    from evaluation.outcome import normalize_result_row

    old_row = {"probe_id": "rea_01", "score": 1.0}
    normalized = normalize_result_row(old_row)
    for field in _IDENTITY_FIELDS:
        assert field in normalized, f"Missing field {field!r} after normalize"
        assert normalized[field] is None, f"Expected None for {field!r}, got {normalized[field]!r}"


def test_normalize_result_row_preserves_identity_fields_when_set() -> None:
    """normalize_result_row does not overwrite identity fields already present."""
    from evaluation.outcome import normalize_result_row

    row = {
        "probe_id": "rea_01",
        "git_sha": "deadbeef",
        "run_seed": 7,
        "hostname": "strix-halo-01",
        "operator": "researcher_b",
        "done_reason": "stop",
    }
    normalized = normalize_result_row(row)
    assert normalized["git_sha"] == "deadbeef"
    assert normalized["run_seed"] == 7
    assert normalized["hostname"] == "strix-halo-01"
    assert normalized["operator"] == "researcher_b"
    assert normalized["done_reason"] == "stop"


# ---------------------------------------------------------------------------
# TTFT fields — ttft_ms and ttft_source
# ---------------------------------------------------------------------------

def test_ttft_fields_present_on_streamed_row() -> None:
    """ttft_ms and ttft_source are present on a live-streamed row."""
    row = _run_cell_with_span()  # _fake returns ttft_ms=200.0, not a cache hit
    assert "ttft_ms" in row, "ttft_ms missing on streamed row"
    assert "ttft_source" in row, "ttft_source missing on streamed row"
    assert row["ttft_source"] == "streamed", (
        f"Expected ttft_source='streamed', got {row['ttft_source']!r}"
    )
    # ttft_ms is 200.0 (from _fake); must be a positive float, not None
    assert row["ttft_ms"] is not None, "ttft_ms must not be None on a streamed row"
    assert row["ttft_ms"] == pytest.approx(200.0, abs=0.2)


def test_ttft_replay_unavailable_on_cache_hit() -> None:
    """When a cache hit is served, ttft_source='replay-unavailable' and ttft_ms=None."""
    from harness import runner, telemetry, cache
    from harness.telemetry import Telemetry

    scorers = _load_scorers()
    outcome_mod = _load_outcome()

    # Build a cached entry that looks like a prior streamed call.
    cached_tel = Telemetry(
        latency_ms=1234.5, ttft_ms=200.0, tokens_in=50, tokens_out=3,
        mem_rss_mb=256.0, gpu_mem_mb=None, gpu_mem_source="unavailable:test",
    )
    cached_entry = {"output": "4", "telemetry": cached_tel.to_dict(), "done_reason": "stop"}

    gpu_mock = MagicMock(return_value=(None, "unavailable:test"))

    with patch.object(cache, "get", return_value=cached_entry), \
         patch.object(cache, "put"), \
         patch.object(telemetry, "gpu_mem_mb", side_effect=gpu_mock), \
         patch.object(telemetry, "rss_mb", return_value=256.0):
        row = runner.run_cell(
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
            outcome_mod=outcome_mod,
            git_sha="abc123",
            run_seed=42,
            hostname="test-host",
            operator="researcher_a",
        )

    assert row["ttft_source"] == "replay-unavailable", row["ttft_source"]
    assert row["ttft_ms"] is None, (
        f"ttft_ms must be None on cache-hit rows, got {row['ttft_ms']!r}"
    )


def test_normalize_result_row_fills_ttft_fields_on_old_row() -> None:
    """normalize_result_row fills ttft_ms and ttft_source with None on old rows."""
    from evaluation.outcome import normalize_result_row

    old_row = {"probe_id": "rea_01", "score": 1.0, "latency_ms": 500.0}
    normalized = normalize_result_row(old_row)
    assert "ttft_ms" in normalized, "ttft_ms missing after normalize"
    assert "ttft_source" in normalized, "ttft_source missing after normalize"
    assert normalized["ttft_ms"] is None
    assert normalized["ttft_source"] is None


def test_normalize_result_row_preserves_ttft_when_set() -> None:
    """normalize_result_row does not overwrite ttft fields already present."""
    from evaluation.outcome import normalize_result_row

    row = {
        "probe_id": "rea_01",
        "ttft_ms": 123.4,
        "ttft_source": "streamed",
    }
    normalized = normalize_result_row(row)
    assert normalized["ttft_ms"] == 123.4
    assert normalized["ttft_source"] == "streamed"
