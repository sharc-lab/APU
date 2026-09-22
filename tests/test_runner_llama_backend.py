"""Tests for the llama-server backend wiring in harness/runner.py.

Covers:
- Session path: run_cell produces rows with all session metadata fields populated.
- Ollama path: run_cell produces rows with session metadata as None.
- ContextSizeError: caught inside run_cell, produces outcome_class="REJECTED" row.
- normalize_result_row: backward compat for old rows missing the new fields.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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


EXACT_PROBE = {
    "id": "rea_01",
    "category": "reasoning",
    "difficulty": "easy",
    "scorer_type": "exact",
    "prompt": "What is 2 + 2?",
    "expected": "4",
    "max_tokens": 32,
}

FILLER = "A" * 100

_COMMON_KWARGS: dict[str, Any] = dict(
    filler=FILLER,
    depth=0,
    rep=0,
    position_in_cell=0,
    cell_probe_seed=0,
    model="qwen3:4b-instruct",
    host="http://localhost:11434",
    cfg_hash="test_cfg_01",
    filler_mode="unlabelled",
    hardware_config="test_hw",
    memory_architecture="discrete",
    model_variant="instruct",
    thinking_enabled=False,
    git_sha="abc123",
    run_seed=42,
    hostname="test-host",
    operator="researcher_a",
    count_method="ollama_prompt_eval",
)


def _fake_session(output="4", latency=500.0, ttft=120.0, tokens_in=10, tokens_out=3,
                  done_reason="stop", thinking_chars=0):
    """Return a mock LlamaServerSession whose call() returns the standard 7-tuple."""
    session = MagicMock()
    session.call.return_value = (output, latency, ttft, tokens_in, tokens_out, done_reason, thinking_chars)
    session.row_metadata.return_value = {
        "server_session_id": "sess-uuid-1234",
        "n_ctx_slot": 8192,
        "build_id": "b10970",
        "backend": "vulkan",
        "platform": "evo-t2s",
        "context_shift_probe_result": "skipped",
    }
    return session


def _run_cell_ollama(probe=None, fake_output="4") -> dict[str, Any]:
    from harness import runner, telemetry, cache
    scorers = _load_scorers()
    outcome_mod = _load_outcome()

    def _fake(model, prompt, max_tokens, host, suppress_thinking=True):
        return fake_output, 1000.0, 200.0, 50, 3, "stop", 0

    with patch.object(runner, "_call_ollama_streaming", side_effect=_fake), \
         patch.object(cache, "get", return_value=None), \
         patch.object(cache, "put"), \
         patch.object(telemetry, "gpu_mem_mb", return_value=(None, "unavailable:test")), \
         patch.object(telemetry, "rss_mb", return_value=100.0):
        return runner.run_cell(
            probe=probe or EXACT_PROBE,
            scorers=scorers,
            outcome_mod=outcome_mod,
            **_COMMON_KWARGS,
        )


def _run_cell_session(probe=None, session=None, fake_output="4") -> dict[str, Any]:
    from harness import runner, telemetry
    scorers = _load_scorers()
    outcome_mod = _load_outcome()

    if session is None:
        session = _fake_session(output=fake_output)

    with patch.object(telemetry, "gpu_mem_mb", return_value=(None, "unavailable:test")), \
         patch.object(telemetry, "rss_mb", return_value=100.0):
        return runner.run_cell(
            probe=probe or EXACT_PROBE,
            scorers=scorers,
            outcome_mod=outcome_mod,
            session=session,
            **_COMMON_KWARGS,
        )


# ── Ollama path: session metadata is None ─────────────────────────────────────

class TestOllamaRowShape:
    def test_session_metadata_fields_are_none(self):
        row = _run_cell_ollama()
        for field in ("server_session_id", "n_ctx_slot", "build_id",
                      "backend", "platform", "context_shift_probe_result"):
            assert row[field] is None, f"expected None for {field!r}, got {row[field]!r}"

    def test_replayed_false_on_live_call(self):
        row = _run_cell_ollama()
        assert row["replayed"] is False

    def test_replayed_true_on_cache_hit(self):
        from harness import runner, telemetry, cache
        scorers = _load_scorers()
        outcome_mod = _load_outcome()

        fake_tel = {"latency_ms": 500.0, "ttft_ms": None, "tokens_in": 10, "tokens_out": 3,
                    "mem_rss_mb": 100.0, "gpu_mem_mb": None, "gpu_mem_source": "unavailable:test"}
        cached = {"output": "4", "telemetry": fake_tel, "done_reason": "stop", "thinking_chars": 0}

        with patch.object(cache, "get", return_value=cached), \
             patch.object(telemetry, "rss_mb", return_value=100.0):
            row = runner.run_cell(
                probe=EXACT_PROBE,
                scorers=scorers,
                outcome_mod=outcome_mod,
                **_COMMON_KWARGS,
            )
        assert row["replayed"] is True
        assert row["ttft_source"] == "replay-unavailable"

    def test_context_overflow_fields_false_on_normal_row(self):
        row = _run_cell_ollama()
        assert row["context_size_exceeded"] is False
        assert row["n_prompt_tokens"] is None
        assert row["n_ctx"] is None


# ── llama-server path: session metadata populated ─────────────────────────────

class TestSessionRowShape:
    def test_session_metadata_populated(self):
        row = _run_cell_session()
        assert row["server_session_id"] == "sess-uuid-1234"
        assert row["n_ctx_slot"] == 8192
        assert row["build_id"] == "b10970"
        assert row["backend"] == "vulkan"
        assert row["platform"] == "evo-t2s"
        assert row["context_shift_probe_result"] == "skipped"

    def test_replayed_always_false(self):
        row = _run_cell_session()
        assert row["replayed"] is False

    def test_ttft_source_streamed(self):
        row = _run_cell_session()
        assert row["ttft_source"] == "streamed"

    def test_score_computed(self):
        row = _run_cell_session(fake_output="4")
        assert row["score"] == 1.0

    def test_session_call_invoked(self):
        session = _fake_session(output="4")
        _run_cell_session(session=session)
        session.call.assert_called_once()

    def test_no_cache_interaction(self):
        """llama-server path must not call cache.get or cache.put."""
        from harness import runner, telemetry, cache
        scorers = _load_scorers()
        outcome_mod = _load_outcome()
        session = _fake_session(output="4")

        with patch.object(cache, "get") as mock_get, \
             patch.object(cache, "put") as mock_put, \
             patch.object(telemetry, "gpu_mem_mb", return_value=(None, "unavailable:test")), \
             patch.object(telemetry, "rss_mb", return_value=100.0):
            runner.run_cell(
                probe=EXACT_PROBE,
                scorers=scorers,
                outcome_mod=outcome_mod,
                session=session,
                **_COMMON_KWARGS,
            )

        mock_get.assert_not_called()
        mock_put.assert_not_called()


# ── ContextSizeError: REJECTED row ────────────────────────────────────────────

class TestContextSizeErrorRow:
    def _make_rejected_row(self):
        from harness import runner, telemetry
        from harness.llama_server import ContextSizeError

        scorers = _load_scorers()
        outcome_mod = _load_outcome()
        session = MagicMock()
        session.call.side_effect = ContextSizeError(
            n_prompt_tokens=9000, n_ctx=8192, raw='{"error":{"type":"exceed_context_size_error"}}'
        )
        session.row_metadata.return_value = {
            "server_session_id": "sess-uuid-5678",
            "n_ctx_slot": 8192,
            "build_id": "b10970",
            "backend": "vulkan",
            "platform": "evo-t2s",
            "context_shift_probe_result": "skipped",
        }

        with patch.object(telemetry, "gpu_mem_mb", return_value=(None, "unavailable:test")), \
             patch.object(telemetry, "rss_mb", return_value=100.0):
            return runner.run_cell(
                probe=EXACT_PROBE,
                scorers=scorers,
                outcome_mod=outcome_mod,
                session=session,
                **_COMMON_KWARGS,
            )

    def test_does_not_propagate(self):
        """ContextSizeError is caught; run_cell returns a dict, not raises."""
        row = self._make_rejected_row()
        assert isinstance(row, dict)

    def test_outcome_class_rejected(self):
        row = self._make_rejected_row()
        assert row["outcome_class"] == "REJECTED"

    def test_classification_method(self):
        row = self._make_rejected_row()
        assert row["classification_method"] == "context_size_exceeded"

    def test_context_size_exceeded_flag(self):
        row = self._make_rejected_row()
        assert row["context_size_exceeded"] is True

    def test_token_counts(self):
        row = self._make_rejected_row()
        assert row["n_prompt_tokens"] == 9000
        assert row["n_ctx"] == 8192

    def test_score_is_none(self):
        row = self._make_rejected_row()
        assert row["score"] is None

    def test_session_metadata_present(self):
        row = self._make_rejected_row()
        assert row["server_session_id"] == "sess-uuid-5678"
        assert row["n_ctx_slot"] == 8192

    def test_replayed_false(self):
        row = self._make_rejected_row()
        assert row["replayed"] is False

    def test_required_row_fields_present(self):
        from harness.runner import REQUIRED_ROW_FIELDS
        row = self._make_rejected_row()
        missing = REQUIRED_ROW_FIELDS - row.keys()
        assert not missing, f"REJECTED row missing required fields: {sorted(missing)}"


# ── normalize_result_row: backward compat ─────────────────────────────────────

class TestNormalizeResultRow:
    def test_old_row_gets_session_fields_as_none(self):
        from evaluation.outcome import normalize_result_row
        old_row = {"probe_id": "x", "score": 1.0, "outcome_class": "CORRECT"}
        out = normalize_result_row(old_row)
        for field in ("server_session_id", "n_ctx_slot", "build_id",
                      "backend", "platform", "context_shift_probe_result"):
            assert field in out, f"missing field {field!r}"
            assert out[field] is None, f"expected None for {field!r}"

    def test_old_row_gets_replayed_as_none(self):
        from evaluation.outcome import normalize_result_row
        old_row = {"probe_id": "x", "score": 1.0}
        out = normalize_result_row(old_row)
        assert "replayed" in out
        assert out["replayed"] is None

    def test_old_row_gets_context_overflow_fields_as_none(self):
        from evaluation.outcome import normalize_result_row
        old_row = {"probe_id": "x", "score": 1.0}
        out = normalize_result_row(old_row)
        for field in ("context_size_exceeded", "n_prompt_tokens", "n_ctx"):
            assert field in out
            assert out[field] is None

    def test_existing_values_preserved(self):
        from evaluation.outcome import normalize_result_row
        row = {"probe_id": "x", "server_session_id": "abc", "replayed": False}
        out = normalize_result_row(row)
        assert out["server_session_id"] == "abc"
        assert out["replayed"] is False

    def test_all_prior_fields_still_normalized(self):
        from evaluation.outcome import normalize_result_row
        old_row = {"probe_id": "x"}
        out = normalize_result_row(old_row)
        for field in ("outcome_class", "classification_method", "format_compliant",
                      "git_sha", "run_seed", "hostname", "operator", "done_reason",
                      "ttft_ms", "ttft_source", "thinking_chars"):
            assert field in out
            assert out[field] is None

    def test_old_row_gets_count_method_as_none(self):
        from evaluation.outcome import normalize_result_row
        old_row = {"probe_id": "x", "score": 1.0}
        out = normalize_result_row(old_row)
        assert "count_method" in out
        assert out["count_method"] is None

    def test_existing_count_method_preserved(self):
        from evaluation.outcome import normalize_result_row
        row = {"probe_id": "x", "count_method": "llamaserver_tokenize"}
        out = normalize_result_row(row)
        assert out["count_method"] == "llamaserver_tokenize"


# ── count_method field on all row types ───────────────────────────────────────

class TestCountMethodField:
    def test_ollama_row_carries_count_method(self):
        row = _run_cell_ollama()
        assert row["count_method"] == "ollama_prompt_eval"

    def test_session_row_carries_count_method(self):
        row = _run_cell_session()
        assert row["count_method"] == "ollama_prompt_eval"  # default in _COMMON_KWARGS

    def test_session_row_llamaserver_count_method(self):
        """count_method=llamaserver_tokenize is passed through to the row."""
        from harness import runner, telemetry
        scorers = _load_scorers()
        outcome_mod = _load_outcome()
        session = _fake_session(output="4")
        kwargs = dict(_COMMON_KWARGS)
        kwargs["count_method"] = "llamaserver_tokenize"

        with patch.object(telemetry, "gpu_mem_mb", return_value=(None, "unavailable:test")), \
             patch.object(telemetry, "rss_mb", return_value=100.0):
            row = runner.run_cell(
                probe=EXACT_PROBE,
                scorers=scorers,
                outcome_mod=outcome_mod,
                session=session,
                **kwargs,
            )
        assert row["count_method"] == "llamaserver_tokenize"

    def test_rejected_row_carries_count_method(self):
        from harness import runner, telemetry
        from harness.llama_server import ContextSizeError
        scorers = _load_scorers()
        outcome_mod = _load_outcome()
        session = MagicMock()
        session.call.side_effect = ContextSizeError(9000, 8192, "{}")
        session.row_metadata.return_value = {
            "server_session_id": "x", "n_ctx_slot": 8192,
            "build_id": "b10970", "backend": "vulkan",
            "platform": "evo-t2s", "context_shift_probe_result": "skipped",
        }
        kwargs = dict(_COMMON_KWARGS)
        kwargs["count_method"] = "llamaserver_tokenize"

        with patch.object(telemetry, "gpu_mem_mb", return_value=(None, "unavailable:test")), \
             patch.object(telemetry, "rss_mb", return_value=100.0):
            row = runner.run_cell(
                probe=EXACT_PROBE, scorers=scorers, outcome_mod=outcome_mod,
                session=session, **kwargs,
            )
        assert row["count_method"] == "llamaserver_tokenize"

    def test_count_method_in_required_fields(self):
        from harness.runner import REQUIRED_ROW_FIELDS
        assert "count_method" in REQUIRED_ROW_FIELDS
