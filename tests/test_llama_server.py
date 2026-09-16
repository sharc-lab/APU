"""Tests for harness/llama_server.py.

Mock-based tests
----------------
All tests in this file run without a real server.  They cover:
  - _parse_n_ctx_slot: log parsing for the n_ctx_slot ground-truth value.
  - _raise_if_context_size_error: HTTP 400 body parsing, including the Ollama
    wrapper format.
  - ContextSizeError attributes.
  - _stream_chat: 7-tuple structure and ContextSizeError path, via httpx mock.
  - LlamaServerSession.start(): process launch, log parsing, probe wiring, and
    row_metadata(), via Popen and httpx mocks.
  - LlamaServerSession.call(): delegates to _stream_chat; raises when not started.
  - _run_overflow_probe: shift-detected path via log mock and httpx mock.

Real-server tests (skipped by default)
---------------------------------------
Marked ``@pytest.mark.llama_server``.  Set the env var
``APU_LLAMA_SERVER_EXE`` and ``APU_LLAMA_SERVER_MODEL`` to enable them.
These verify actual server behaviour against evo-t2s or Blade 14:
  - n_ctx_slot correctly read from live startup log.
  - ContextSizeError raised on a real oversized prompt.
  - server_session_id is a valid UUID4.
  - thinking_chars == 0 when reasoning_budget=0 (only verifiable on real server).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call as mock_call

import pytest

from harness.llama_server import (
    ContextSizeError,
    LlamaServerConfig,
    LlamaServerSession,
    _log_has_context_shift,
    _parse_n_ctx_slot,
    _raise_if_context_size_error,
    _run_overflow_probe,
    _stream_chat,
)


# ── _parse_n_ctx_slot ──────────────────────────────────────────────────────────

class TestParseNCtxSlot:
    def test_canonical_line(self):
        log = (
            "0.00.975.759 I srv    load_model: initializing, "
            "n_slots = 1, n_ctx_slot = 256, kv_unified = 'false'"
        )
        assert _parse_n_ctx_slot(log) == 256

    def test_large_context(self):
        log = "load_model: initializing, n_slots = 1, n_ctx_slot = 32768, kv_unified = 'false'"
        assert _parse_n_ctx_slot(log) == 32768

    def test_8192_context(self):
        log = "load_model: initializing, n_slots = 1, n_ctx_slot = 8192, kv_unified = 'false'"
        assert _parse_n_ctx_slot(log) == 8192

    def test_multiline_log_returns_first_match(self):
        log = (
            "some startup preamble\n"
            "load_model: initializing, n_slots = 1, n_ctx_slot = 512, kv_unified = 'false'\n"
            "more log lines\n"
        )
        assert _parse_n_ctx_slot(log) == 512

    def test_absent_returns_none(self):
        assert _parse_n_ctx_slot("no relevant lines here") is None

    def test_empty_log_returns_none(self):
        assert _parse_n_ctx_slot("") is None


# ── _log_has_context_shift ─────────────────────────────────────────────────────

class TestLogHasContextShift:
    def test_detects_shift_warning(self):
        log = (
            "0.07.935.631 W slot   operator(): id  0 | task 0 | "
            "slot context shift, n_keep = 0, n_left = 255, n_discard = 127"
        )
        assert _log_has_context_shift(log) is True

    def test_absent(self):
        assert _log_has_context_shift("normal log with no shift event") is False

    def test_empty(self):
        assert _log_has_context_shift("") is False


# ── _raise_if_context_size_error ───────────────────────────────────────────────

class TestRaiseIfContextSizeError:
    def test_direct_llama_server_body(self):
        body = json.dumps({
            "error": {
                "code": 400,
                "message": "request (8614 tokens) exceeds the available context size (8192 tokens)",
                "type": "exceed_context_size_error",
                "n_prompt_tokens": 8614,
                "n_ctx": 8192,
            }
        })
        with pytest.raises(ContextSizeError) as exc_info:
            _raise_if_context_size_error(body)
        err = exc_info.value
        assert err.n_prompt_tokens == 8614
        assert err.n_ctx == 8192
        assert err.raw == body

    def test_ollama_wrapped_body(self):
        """Ollama wraps the inner llama.cpp error as a JSON string in the outer error field."""
        inner = json.dumps({
            "error": {
                "code": 400,
                "message": "request (403 tokens) exceeds the available context size (256 tokens)",
                "type": "exceed_context_size_error",
                "n_prompt_tokens": 403,
                "n_ctx": 256,
            }
        })
        body = json.dumps({"error": inner})
        with pytest.raises(ContextSizeError) as exc_info:
            _raise_if_context_size_error(body)
        err = exc_info.value
        assert err.n_prompt_tokens == 403
        assert err.n_ctx == 256

    def test_other_400_does_not_raise(self):
        body = json.dumps({"error": {"code": 400, "message": "some other error", "type": "other_error"}})
        _raise_if_context_size_error(body)  # must not raise

    def test_malformed_json_does_not_raise(self):
        _raise_if_context_size_error("not json at all")  # must not raise

    def test_empty_body_does_not_raise(self):
        _raise_if_context_size_error("")


# ── ContextSizeError ───────────────────────────────────────────────────────────

class TestContextSizeError:
    def test_attributes(self):
        err = ContextSizeError(n_prompt_tokens=1000, n_ctx=512, raw='{"error":{}}')
        assert err.n_prompt_tokens == 1000
        assert err.n_ctx == 512
        assert err.raw == '{"error":{}}'

    def test_is_exception(self):
        err = ContextSizeError(1000, 512, "")
        assert isinstance(err, Exception)

    def test_str_contains_counts(self):
        err = ContextSizeError(1000, 512, "")
        s = str(err)
        assert "1000" in s
        assert "512" in s


# ── _stream_chat (mocked httpx) ────────────────────────────────────────────────

def _make_sse_lines(chunks: list[dict]) -> list[str]:
    """Build SSE lines from a list of delta dicts."""
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
    lines.append("data: [DONE]")
    return lines


def _mock_stream_response(
    status_code: int = 200,
    sse_lines: list[str] | None = None,
    body_bytes: bytes = b"",
) -> MagicMock:
    """Return a mock httpx response suitable for use as a context manager."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.iter_lines.return_value = iter(sse_lines or [])
    resp.read.return_value = body_bytes
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _content_chunk(text: str, finish: str | None = None) -> dict:
    c: dict[str, Any] = {
        "choices": [{"delta": {"content": text}, "finish_reason": finish, "index": 0}]
    }
    return c


def _reasoning_chunk(text: str) -> dict:
    return {
        "choices": [{"delta": {"reasoning_content": text}, "finish_reason": None, "index": 0}]
    }


def _usage_chunk(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "choices": [{"delta": {}, "finish_reason": None, "index": 0}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class TestStreamChat:
    def _patch_stream(self, mock_resp):
        return patch("harness.llama_server.httpx.stream", return_value=mock_resp)

    def test_returns_7_tuple(self):
        lines = _make_sse_lines([
            _content_chunk("hello ", None),
            _content_chunk("world", "stop"),
            _usage_chunk(10, 5),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with self._patch_stream(resp):
            result = _stream_chat(port=8181, prompt="hi", max_tokens=50)
        assert len(result) == 7

    def test_text_assembled_correctly(self):
        lines = _make_sse_lines([
            _content_chunk("foo "),
            _content_chunk("bar", "stop"),
            _usage_chunk(5, 2),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with self._patch_stream(resp):
            text, *_ = _stream_chat(port=8181, prompt="x", max_tokens=10)
        assert text == "foo bar"

    def test_ttft_set_on_first_content_token(self):
        lines = _make_sse_lines([
            _content_chunk("tok", "stop"),
            _usage_chunk(3, 1),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with self._patch_stream(resp):
            _, _, ttft_ms, *_ = _stream_chat(port=8181, prompt="x", max_tokens=10)
        assert ttft_ms is not None
        assert ttft_ms >= 0.0

    def test_ttft_none_when_no_content_token(self):
        """Empty response: no content tokens, ttft_ms must be None."""
        lines = _make_sse_lines([
            {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]},
            _usage_chunk(5, 0),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with self._patch_stream(resp):
            _, _, ttft_ms, *_ = _stream_chat(port=8181, prompt="x", max_tokens=10)
        assert ttft_ms is None

    def test_thinking_chars_counted_from_reasoning_content(self):
        lines = _make_sse_lines([
            _reasoning_chunk("think step 1 "),
            _reasoning_chunk("think step 2"),
            _content_chunk("answer", "stop"),
            _usage_chunk(8, 3),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with self._patch_stream(resp):
            *_, thinking_chars = _stream_chat(port=8181, prompt="x", max_tokens=50)
        assert thinking_chars == len("think step 1 ") + len("think step 2")

    def test_thinking_chars_zero_when_suppressed(self):
        lines = _make_sse_lines([
            _content_chunk("4", "stop"),
            _usage_chunk(8, 1),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with self._patch_stream(resp):
            *_, thinking_chars = _stream_chat(port=8181, prompt="2+2?", max_tokens=10)
        assert thinking_chars == 0

    def test_token_counts_from_usage_chunk(self):
        lines = _make_sse_lines([
            _content_chunk("ok", "stop"),
            _usage_chunk(prompt_tokens=42, completion_tokens=7),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with self._patch_stream(resp):
            _, _, _, tokens_in, tokens_out, _, _ = _stream_chat(
                port=8181, prompt="x", max_tokens=10
            )
        assert tokens_in == 42
        assert tokens_out == 7

    def test_done_reason_captured(self):
        lines = _make_sse_lines([
            _content_chunk("out", "length"),
            _usage_chunk(5, 3),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with self._patch_stream(resp):
            _, _, _, _, _, done_reason, _ = _stream_chat(
                port=8181, prompt="x", max_tokens=3
            )
        assert done_reason == "length"

    def test_raises_context_size_error_on_400(self):
        body_400 = json.dumps({
            "error": {
                "code": 400,
                "message": "request (500 tokens) exceeds the available context size (256 tokens)",
                "type": "exceed_context_size_error",
                "n_prompt_tokens": 500,
                "n_ctx": 256,
            }
        }).encode()
        resp = _mock_stream_response(status_code=400, body_bytes=body_400)
        with self._patch_stream(resp):
            with pytest.raises(ContextSizeError) as exc_info:
                _stream_chat(port=8181, prompt="x" * 1000, max_tokens=10)
        assert exc_info.value.n_prompt_tokens == 500
        assert exc_info.value.n_ctx == 256

    def test_latency_ms_is_positive(self):
        lines = _make_sse_lines([
            _content_chunk("x", "stop"),
            _usage_chunk(2, 1),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with self._patch_stream(resp):
            _, latency_ms, *_ = _stream_chat(port=8181, prompt="x", max_tokens=5)
        assert latency_ms >= 0.0

    def test_skips_done_sentinel_line(self):
        """data: [DONE] must not be parsed as JSON."""
        lines = [
            f"data: {json.dumps(_content_chunk('hi', 'stop'))}",
            f"data: {json.dumps(_usage_chunk(3, 1))}",
            "data: [DONE]",
        ]
        resp = _mock_stream_response(sse_lines=lines)
        with self._patch_stream(resp):
            text, *_ = _stream_chat(port=8181, prompt="x", max_tokens=5)
        assert text == "hi"

    def test_malformed_sse_lines_skipped(self):
        """Lines that are not valid JSON after stripping 'data: ' are ignored."""
        lines = [
            "data: NOT_JSON",
            f"data: {json.dumps(_content_chunk('ok', 'stop'))}",
            f"data: {json.dumps(_usage_chunk(2, 1))}",
        ]
        resp = _mock_stream_response(sse_lines=lines)
        with self._patch_stream(resp):
            text, *_ = _stream_chat(port=8181, prompt="x", max_tokens=5)
        assert text == "ok"


# ── LlamaServerSession.start() mocked ─────────────────────────────────────────

_FAKE_LOG_TEXT = (
    "0.00.800.000 I srv    llama server listening at http://0.0.0.0:8181\n"
    "0.00.975.759 I srv    load_model: initializing, "
    "n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'\n"
)


def _make_path_mock(log_text: str = _FAKE_LOG_TEXT):
    """Return (mock_path_cls, context_manager_patch) for Path in llama_server."""
    mock_path_instance = MagicMock()
    mock_path_instance.read_text.return_value = log_text
    return MagicMock(return_value=mock_path_instance)


def _standard_start_patches(proc, health_resp, log_text=_FAKE_LOG_TEXT, probe_result="skipped"):
    """Return a list of patch context managers for LlamaServerSession.start() mocking."""
    from contextlib import ExitStack
    return (
        patch("harness.llama_server.subprocess.Popen", return_value=proc),
        patch("harness.llama_server.httpx.get", return_value=health_resp),
        patch("harness.llama_server.Path", _make_path_mock(log_text)),
        patch("harness.llama_server.open", create=True),
        patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
        patch("harness.llama_server.os.close"),
        patch("harness.llama_server._run_overflow_probe", return_value=probe_result),
    )

_FAKE_CFG = LlamaServerConfig(
    exe=r"C:\apu\bin\llama-b10970\llama-server.exe",
    model=r"C:\apu\models\Qwen3-4B-Q4_K_M.gguf",
    ctx_size=4096,
    port=8181,
    context_shift=False,
    reasoning_budget=None,
    platform="evo-t2s",
)


def _make_session_start_mocks(log_text: str = _FAKE_LOG_TEXT):
    """Return (popen_mock, health_mock, log_mock) for patching LlamaServerSession.start()."""
    proc = MagicMock()
    proc.kill = MagicMock()
    proc.wait = MagicMock(return_value=0)
    return proc


class TestLlamaServerSessionStart:
    def _start_with_mocks(
        self,
        cfg: LlamaServerConfig = _FAKE_CFG,
        log_text: str = _FAKE_LOG_TEXT,
        probe_result: str = "skipped",
    ) -> LlamaServerSession:
        proc = MagicMock()
        health_resp = MagicMock()
        health_resp.status_code = 200

        # Patch Path so that Path(any_path).read_text(...) returns log_text.
        mock_path_instance = MagicMock()
        mock_path_instance.read_text.return_value = log_text
        mock_path_cls = MagicMock(return_value=mock_path_instance)

        with (
            patch("harness.llama_server.subprocess.Popen", return_value=proc),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", mock_path_cls),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/test.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value=probe_result),
        ):
            session = LlamaServerSession(cfg)
            session.start()
        return session

    def test_n_ctx_slot_parsed_from_log(self):
        session = self._start_with_mocks()
        assert session.n_ctx_slot == 4096

    def test_server_session_id_is_uuid4(self):
        session = self._start_with_mocks()
        parsed = uuid.UUID(session.server_session_id, version=4)
        assert str(parsed) == session.server_session_id

    def test_server_session_id_unique_per_session(self):
        s1 = self._start_with_mocks()
        s2 = self._start_with_mocks()
        assert s1.server_session_id != s2.server_session_id

    def test_context_shift_probe_skipped_when_disabled(self):
        session = self._start_with_mocks(cfg=_FAKE_CFG)
        assert session.context_shift_probe_result == "skipped"

    def test_context_shift_probe_runs_when_enabled(self):
        cfg = LlamaServerConfig(
            exe=_FAKE_CFG.exe, model=_FAKE_CFG.model,
            ctx_size=256, port=8181, context_shift=True,
        )
        session = self._start_with_mocks(cfg=cfg, probe_result="passed")
        assert session.context_shift_probe_result == "passed"

    def test_row_metadata_contains_required_fields(self):
        session = self._start_with_mocks()
        meta = session.row_metadata()
        for key in ("server_session_id", "n_ctx_slot", "build_id", "backend",
                    "platform", "context_shift_probe_result"):
            assert key in meta, f"Missing key: {key}"

    def test_row_metadata_values_match_config(self):
        session = self._start_with_mocks()
        meta = session.row_metadata()
        assert meta["n_ctx_slot"] == 4096
        assert meta["build_id"] == "b10970"
        assert meta["backend"] == "vulkan"
        assert meta["platform"] == "evo-t2s"
        assert meta["server_session_id"] == session.server_session_id

    def test_raises_if_n_ctx_slot_missing_from_log(self):
        with pytest.raises(RuntimeError, match="n_ctx_slot"):
            self._start_with_mocks(log_text="no useful lines here")

    def test_double_start_raises(self):
        proc = MagicMock()
        health_resp = MagicMock(status_code=200)
        with (
            patch("harness.llama_server.subprocess.Popen", return_value=proc),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="skipped"),
        ):
            session = LlamaServerSession(_FAKE_CFG)
            session.start()
            with pytest.raises(RuntimeError, match="already started"):
                session.start()

    def test_call_raises_when_not_started(self):
        session = LlamaServerSession(_FAKE_CFG)
        with pytest.raises(RuntimeError, match="not started"):
            session.call("hello", 10)

    def test_command_includes_no_context_shift_by_default(self):
        """When context_shift=False, --no-context-shift must appear in the command."""
        proc = MagicMock()
        health_resp = MagicMock(status_code=200)
        captured_cmd: list[str] = []

        def _fake_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return proc

        with (
            patch("harness.llama_server.subprocess.Popen", side_effect=_fake_popen),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="skipped"),
        ):
            LlamaServerSession(_FAKE_CFG).start()

        assert "--no-context-shift" in captured_cmd
        assert "--context-shift" not in captured_cmd

    def test_command_includes_context_shift_when_enabled(self):
        cfg = LlamaServerConfig(
            exe=_FAKE_CFG.exe, model=_FAKE_CFG.model,
            ctx_size=512, port=8181, context_shift=True,
        )
        proc = MagicMock()
        health_resp = MagicMock(status_code=200)
        captured_cmd: list[str] = []

        def _fake_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return proc

        with (
            patch("harness.llama_server.subprocess.Popen", side_effect=_fake_popen),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="passed"),
        ):
            LlamaServerSession(cfg).start()

        assert "--context-shift" in captured_cmd
        assert "--no-context-shift" not in captured_cmd

    def test_command_includes_reasoning_budget_0(self):
        cfg = LlamaServerConfig(
            exe=_FAKE_CFG.exe, model=_FAKE_CFG.model,
            ctx_size=4096, port=8181, reasoning_budget=0,
        )
        proc = MagicMock()
        health_resp = MagicMock(status_code=200)
        captured_cmd: list[str] = []

        def _fake_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return proc

        with (
            patch("harness.llama_server.subprocess.Popen", side_effect=_fake_popen),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="skipped"),
            patch.object(LlamaServerSession, "_verify_thinking_suppressed", return_value=True),
        ):
            LlamaServerSession(cfg).start()

        assert "--reasoning-budget" in captured_cmd
        idx = captured_cmd.index("--reasoning-budget")
        assert captured_cmd[idx + 1] == "0"

    def test_command_includes_reasoning_format_deepseek_by_default(self):
        proc = MagicMock()
        health_resp = MagicMock(status_code=200)
        captured_cmd: list[str] = []

        def _fake_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return proc

        with (
            patch("harness.llama_server.subprocess.Popen", side_effect=_fake_popen),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="skipped"),
        ):
            LlamaServerSession(_FAKE_CFG).start()

        assert "--reasoning-format" in captured_cmd
        idx = captured_cmd.index("--reasoning-format")
        assert captured_cmd[idx + 1] == "deepseek"

    def test_command_omits_reasoning_budget_when_none(self):
        proc = MagicMock()
        health_resp = MagicMock(status_code=200)
        captured_cmd: list[str] = []

        def _fake_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return proc

        with (
            patch("harness.llama_server.subprocess.Popen", side_effect=_fake_popen),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="skipped"),
        ):
            LlamaServerSession(_FAKE_CFG).start()  # reasoning_budget=None by default

        assert "--reasoning-budget" not in captured_cmd

    def test_context_manager_stops_on_exit(self):
        proc = MagicMock()
        health_resp = MagicMock(status_code=200)
        with (
            patch("harness.llama_server.subprocess.Popen", return_value=proc),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="skipped"),
        ):
            with LlamaServerSession(_FAKE_CFG):
                pass
        proc.kill.assert_called_once()


# ── LlamaServerSession.call() ─────────────────────────────────────────────────

class TestLlamaServerSessionCall:
    def _started_session(self) -> LlamaServerSession:
        """Return a session that has been started via mocks."""
        proc = MagicMock()
        health_resp = MagicMock(status_code=200)
        with (
            patch("harness.llama_server.subprocess.Popen", return_value=proc),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="skipped"),
        ):
            session = LlamaServerSession(_FAKE_CFG)
            session.start()
        return session

    def test_call_returns_7_tuple(self):
        session = self._started_session()
        lines = _make_sse_lines([
            _content_chunk("answer", "stop"),
            _usage_chunk(10, 5),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with patch("harness.llama_server.httpx.stream", return_value=resp):
            result = session.call("prompt text", max_tokens=50)
        assert len(result) == 7

    def test_call_propagates_context_size_error(self):
        session = self._started_session()
        body_400 = json.dumps({
            "error": {
                "code": 400,
                "message": "request (5000 tokens) exceeds the available context size (4096 tokens)",
                "type": "exceed_context_size_error",
                "n_prompt_tokens": 5000,
                "n_ctx": 4096,
            }
        }).encode()
        resp = _mock_stream_response(status_code=400, body_bytes=body_400)
        with patch("harness.llama_server.httpx.stream", return_value=resp):
            with pytest.raises(ContextSizeError) as exc_info:
                session.call("x" * 10000, max_tokens=10)
        assert exc_info.value.n_prompt_tokens == 5000
        assert exc_info.value.n_ctx == 4096


# ── LlamaServerSession.is_alive() / restart() ─────────────────────────────────

class TestIsAliveAndRestart:
    def test_is_alive_false_before_start(self):
        session = LlamaServerSession(_FAKE_CFG)
        assert session.is_alive() is False

    def test_is_alive_true_after_start(self):
        session = self._start_session()
        assert session.is_alive() is True

    def test_is_alive_false_after_stop(self):
        session = self._start_session()
        session.stop()
        assert session.is_alive() is False

    def test_restart_generates_new_session_id(self):
        with self._all_patches():
            session = LlamaServerSession(_FAKE_CFG).start()
            old_id = session.server_session_id
            session.restart()
            assert session.server_session_id != old_id

    def test_restart_applies_new_config(self):
        new_cfg = LlamaServerConfig(
            exe=_FAKE_CFG.exe, model=_FAKE_CFG.model,
            ctx_size=1024, port=_FAKE_CFG.port, platform="blade14",
        )
        with self._all_patches():
            session = LlamaServerSession(_FAKE_CFG).start()
            session.restart(new_config=new_cfg)
            assert session.cfg.ctx_size == 1024
            assert session.cfg.platform == "blade14"

    def test_restart_resets_state_fields(self):
        with self._all_patches():
            session = LlamaServerSession(_FAKE_CFG).start()
            session.restart()
            assert session.n_ctx_slot is not None  # re-parsed after restart
            assert session.context_shift_probe_result is not None
            assert session.thinking_suppression_verified is None  # no budget=0

    def _all_patches(self):
        from contextlib import ExitStack
        proc = MagicMock()
        proc.poll.return_value = None
        health_resp = MagicMock(status_code=200)
        stack = ExitStack()
        stack.enter_context(patch("harness.llama_server.subprocess.Popen", return_value=proc))
        stack.enter_context(patch("harness.llama_server.httpx.get", return_value=health_resp))
        stack.enter_context(patch("harness.llama_server.Path", _make_path_mock()))
        stack.enter_context(patch("harness.llama_server.open", create=True))
        stack.enter_context(patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")))
        stack.enter_context(patch("harness.llama_server.os.close"))
        stack.enter_context(patch("harness.llama_server._run_overflow_probe", return_value="skipped"))
        return stack

    def _start_session(self) -> LlamaServerSession:
        proc = MagicMock()
        proc.poll.return_value = None  # alive
        health_resp = MagicMock(status_code=200)
        with (
            patch("harness.llama_server.subprocess.Popen", return_value=proc),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="skipped"),
        ):
            return LlamaServerSession(_FAKE_CFG).start()


# ── _verify_thinking_suppressed tri-state ─────────────────────────────────────

class TestVerifyThinkingSuppressed:
    def test_returns_true_when_no_thinking_chars(self):
        proc = MagicMock()
        health_resp = MagicMock(status_code=200)
        cfg = LlamaServerConfig(
            exe=_FAKE_CFG.exe, model=_FAKE_CFG.model,
            ctx_size=512, port=8181, reasoning_budget=0,
        )
        lines = _make_sse_lines([_content_chunk("1/3.", "stop"), _usage_chunk(10, 5)])
        resp = _mock_stream_response(sse_lines=lines)
        with (
            patch("harness.llama_server.subprocess.Popen", return_value=proc),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="skipped"),
            patch("harness.llama_server.httpx.stream", return_value=resp),
        ):
            session = LlamaServerSession(cfg).start()
        assert session.thinking_suppression_verified is True

    def test_returns_false_when_thinking_chars_present(self):
        proc = MagicMock()
        health_resp = MagicMock(status_code=200)
        cfg = LlamaServerConfig(
            exe=_FAKE_CFG.exe, model=_FAKE_CFG.model,
            ctx_size=512, port=8181, reasoning_budget=0,
        )
        lines = _make_sse_lines([
            _reasoning_chunk("some thinking"),
            _content_chunk("1/3.", "stop"),
            _usage_chunk(10, 5),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with (
            patch("harness.llama_server.subprocess.Popen", return_value=proc),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="skipped"),
            patch("harness.llama_server.httpx.stream", return_value=resp),
        ):
            session = LlamaServerSession(cfg).start()
        assert session.thinking_suppression_verified is False

    def test_returns_none_on_exception(self):
        """Exception in verify_thinking_suppressed must not be misread as False."""
        proc = MagicMock()
        health_resp = MagicMock(status_code=200)
        cfg = LlamaServerConfig(
            exe=_FAKE_CFG.exe, model=_FAKE_CFG.model,
            ctx_size=512, port=8181, reasoning_budget=0,
        )
        with (
            patch("harness.llama_server.subprocess.Popen", return_value=proc),
            patch("harness.llama_server.httpx.get", return_value=health_resp),
            patch("harness.llama_server.Path", _make_path_mock()),
            patch("harness.llama_server.open", create=True),
            patch("harness.llama_server.tempfile.mkstemp", return_value=(0, "/tmp/t.log")),
            patch("harness.llama_server.os.close"),
            patch("harness.llama_server._run_overflow_probe", return_value="skipped"),
            patch("harness.llama_server.httpx.stream", side_effect=OSError("conn failed")),
        ):
            session = LlamaServerSession(cfg).start()
        assert session.thinking_suppression_verified is None


# ── _run_overflow_probe ────────────────────────────────────────────────────────

class TestRunOverflowProbe:
    def test_returns_passed_when_shift_in_log(self, tmp_path):
        log_file = tmp_path / "srv.log"
        log_file.write_text(
            "0.07.935 W slot operator(): slot context shift, n_keep = 0, "
            "n_left = 255, n_discard = 127\n",
            encoding="utf-8",
        )
        lines = _make_sse_lines([_content_chunk("1\n2\n", "length"), _usage_chunk(5, 2)])
        resp = _mock_stream_response(sse_lines=lines)
        with patch("harness.llama_server.httpx.stream", return_value=resp):
            result = _run_overflow_probe(
                port=8181,
                n_ctx_slot=256,
                log_path=str(log_file),
            )
        assert result == "passed"

    def test_returns_failed_when_no_shift(self, tmp_path):
        log_file = tmp_path / "srv.log"
        log_file.write_text("no shift events here\n", encoding="utf-8")
        lines = _make_sse_lines([_content_chunk("short answer", "stop"), _usage_chunk(5, 3)])
        resp = _mock_stream_response(sse_lines=lines)
        with patch("harness.llama_server.httpx.stream", return_value=resp):
            result = _run_overflow_probe(
                port=8181,
                n_ctx_slot=256,
                log_path=str(log_file),
            )
        assert result == "failed"

    def test_returns_failed_on_non_200(self, tmp_path):
        log_file = tmp_path / "srv.log"
        log_file.write_text("", encoding="utf-8")
        resp = _mock_stream_response(status_code=500)
        with patch("harness.llama_server.httpx.stream", return_value=resp):
            result = _run_overflow_probe(
                port=8181,
                n_ctx_slot=256,
                log_path=str(log_file),
            )
        assert result == "failed"


# ── Signature compatibility ────────────────────────────────────────────────────

class TestSignatureCompatibility:
    """Verify the 7-tuple matches _call_ollama_streaming's positional contract."""

    def test_7_tuple_positional_unpacking(self):
        """The 7-tuple must unpack correctly into the names run_cell uses."""
        lines = _make_sse_lines([
            _reasoning_chunk("think"),
            _content_chunk("answer text", "stop"),
            _usage_chunk(12, 4),
        ])
        resp = _mock_stream_response(sse_lines=lines)
        with patch("harness.llama_server.httpx.stream", return_value=resp):
            (
                output,          # [0] text
                latency_ms,      # [1] float
                ttft_ms,         # [2] float | None
                tokens_in,       # [3] int
                tokens_out,      # [4] int
                done_reason,     # [5] str | None
                thinking_chars,  # [6] int
            ) = _stream_chat(port=8181, prompt="x", max_tokens=20)

        assert isinstance(output, str)
        assert isinstance(latency_ms, float)
        # ttft_ms is float (content token arrived) or None
        assert ttft_ms is None or isinstance(ttft_ms, float)
        assert isinstance(tokens_in, int)
        assert isinstance(tokens_out, int)
        assert done_reason is None or isinstance(done_reason, str)
        assert isinstance(thinking_chars, int)
        assert thinking_chars == len("think")
        assert output == "answer text"
        assert tokens_in == 12
        assert tokens_out == 4


# ── Real-server tests (skipped unless env vars are set) ───────────────────────

_REAL_EXE = os.environ.get("APU_LLAMA_SERVER_EXE")
_REAL_MODEL = os.environ.get("APU_LLAMA_SERVER_MODEL")
_REAL_PORT = int(os.environ.get("APU_LLAMA_SERVER_PORT", "8282"))
_REAL_PLATFORM = os.environ.get("APU_LLAMA_SERVER_PLATFORM", "unknown")

_needs_real_server = pytest.mark.skipif(
    not (_REAL_EXE and _REAL_MODEL),
    reason=(
        "Real-server tests require APU_LLAMA_SERVER_EXE and "
        "APU_LLAMA_SERVER_MODEL env vars"
    ),
)


@_needs_real_server
class TestRealServer:
    """Integration tests against a live llama-server process.

    These verify behaviour that mocks cannot reach:
    - n_ctx_slot actually parsed from the live startup log.
    - ContextSizeError raised on a real oversized prompt at scale.
    - thinking_chars == 0 when --reasoning-budget 0 is active.

    Run with::

        APU_LLAMA_SERVER_EXE="C:/apu/bin/llama-b10970/llama-server.exe" \\
        APU_LLAMA_SERVER_MODEL="C:/apu/models/Qwen3-4B-Q4_K_M.gguf" \\
        APU_LLAMA_SERVER_PORT=8282 \\
        APU_LLAMA_SERVER_PLATFORM=evo-t2s \\
        py -3.12 -m pytest tests/test_llama_server.py -k TestRealServer -v
    """

    @pytest.fixture(scope="class")
    @classmethod
    def session(cls):
        cfg = LlamaServerConfig(
            exe=_REAL_EXE,
            model=_REAL_MODEL,
            ctx_size=512,
            port=_REAL_PORT,
            context_shift=False,
            reasoning_budget=0,
            platform=_REAL_PLATFORM,
        )
        with LlamaServerSession(cfg) as s:
            yield s

    def test_n_ctx_slot_is_int(self, session):
        assert isinstance(session.n_ctx_slot, int)
        assert session.n_ctx_slot >= 256  # enforced minimum

    def test_n_ctx_slot_matches_or_exceeds_requested(self, session):
        # ctx_size=512, so n_ctx_slot must be 512 (above the 256 floor)
        assert session.n_ctx_slot == 512

    def test_server_session_id_is_uuid4(self, session):
        parsed = uuid.UUID(session.server_session_id, version=4)
        assert str(parsed) == session.server_session_id

    def test_row_metadata_has_all_fields(self, session):
        meta = session.row_metadata()
        for key in ("server_session_id", "n_ctx_slot", "build_id",
                    "backend", "platform", "context_shift_probe_result"):
            assert key in meta

    def test_normal_call_returns_7_tuple_with_correct_types(self, session):
        text, lat, ttft, tin, tout, reason, think = session.call("Say hello.", 20)
        assert isinstance(text, str)
        assert isinstance(lat, float) and lat > 0
        assert ttft is None or (isinstance(ttft, float) and ttft > 0)
        assert isinstance(tin, int) and tin > 0
        assert isinstance(tout, int) and tout >= 0
        assert reason in ("stop", "length", None)
        assert isinstance(think, int) and think >= 0

    def test_thinking_chars_zero_with_reasoning_budget_0(self, session):
        # reasoning_budget=0 was set at server start; all calls should produce 0 thinking chars
        _, _, _, _, _, _, thinking_chars = session.call(
            "What is the exact value of the definite integral of x^3 from 0 to 2?",
            max_tokens=60,
        )
        assert thinking_chars == 0, (
            f"Expected 0 thinking chars with --reasoning-budget 0, got {thinking_chars}"
        )

    def test_oversized_prompt_raises_context_size_error(self, session):
        # session has n_ctx_slot=512.  Use "a " * 1000 (~1000 tokens) to
        # reliably exceed any n_ctx_slot value in [256, 512].  The exact
        # tokenisation of the repeated phrase doesn't matter; 1000 single-
        # character tokens are always > 512.
        filler = "a " * 1000
        prompt = f"Count to three. {filler}"
        with pytest.raises(ContextSizeError) as exc_info:
            session.call(prompt, max_tokens=10)
        err = exc_info.value
        assert err.n_ctx == session.n_ctx_slot
        assert err.n_prompt_tokens > session.n_ctx_slot

    def test_ttft_source_semantics(self, session):
        """ttft_ms should be non-None and positive for a normal call."""
        _, _, ttft_ms, _, _, _, _ = session.call("Say one word.", max_tokens=5)
        assert ttft_ms is not None
        assert ttft_ms > 0
        # The caller tags ttft_source="streamed" when ttft_ms is not None

    def test_ttft_meaningfully_below_latency(self, session):
        """ttft_ms should be materially less than latency_ms for a multi-token response."""
        _, latency_ms, ttft_ms, _, _, _, _ = session.call(
            "List the first five prime numbers.", max_tokens=40
        )
        assert ttft_ms is not None
        assert ttft_ms < latency_ms  # first token always before last token

    @pytest.fixture(scope="class")
    @classmethod
    def session_no_suppression(cls):
        """A second session with reasoning enabled (no --reasoning-budget)."""
        cfg = LlamaServerConfig(
            exe=_REAL_EXE,
            model=_REAL_MODEL,
            ctx_size=2048,
            port=_REAL_PORT + 1,
            context_shift=False,
            reasoning_budget=None,  # no suppression
            platform=_REAL_PLATFORM,
        )
        with LlamaServerSession(cfg) as s:
            yield s

    def test_thinking_chars_positive_without_suppression(self, session_no_suppression):
        """With reasoning enabled, a think-heavy prompt should produce thinking_chars > 0."""
        _, _, _, _, _, _, thinking_chars = session_no_suppression.call(
            "A bat and a ball cost $1.10 in total. "
            "The bat costs $1.00 more than the ball. "
            "How much does the ball cost? Show all reasoning.",
            max_tokens=200,
        )
        assert thinking_chars > 0, (
            f"Expected thinking_chars > 0 with reasoning enabled, got {thinking_chars}. "
            "Check that --reasoning-format deepseek is working and the model is thinking."
        )

    def test_ttft_and_latency_live_numbers(self, session):
        """Print actual ttft_ms and latency_ms so sweep cost can be estimated."""
        _, latency_ms, ttft_ms, _, _, _, _ = session.call(
            "List the first five prime numbers.", max_tokens=40
        )
        print(
            f"\n  [LIVE NUMBERS] ttft_ms={ttft_ms:.1f}  latency_ms={latency_ms:.1f}  "
            f"ratio={ttft_ms/latency_ms:.2f}",
            flush=True,
        )
        assert ttft_ms is not None and ttft_ms > 0
        assert latency_ms > ttft_ms

    @pytest.fixture(scope="class")
    @classmethod
    def session_context_shift(cls):
        """Session with context_shift=True to exercise the overflow probe live."""
        cfg = LlamaServerConfig(
            exe=_REAL_EXE,
            model=_REAL_MODEL,
            ctx_size=512,
            port=_REAL_PORT + 2,
            context_shift=True,
            reasoning_budget=0,
            platform=_REAL_PLATFORM,
        )
        import time as _time
        t0 = _time.monotonic()
        with LlamaServerSession(cfg) as s:
            probe_elapsed_s = _time.monotonic() - t0
            s._probe_elapsed_s = probe_elapsed_s  # stash for the test
            yield s

    def test_context_shift_probe_passes_live(self, session_context_shift):
        """The overflow probe must detect an actual context shift on a live server.

        This is the R1 positive control for --context-shift — the only way
        to verify the flag took effect, since it leaves no startup-log evidence.
        Also prints probe elapsed time so sweep startup cost is known.
        """
        result = session_context_shift.context_shift_probe_result
        elapsed = getattr(session_context_shift, "_probe_elapsed_s", None)
        print(
            f"\n  [CONTEXT SHIFT PROBE] result={result!r}  "
            f"probe_elapsed_s={elapsed:.1f}",
            flush=True,
        )
        assert result == "passed", (
            f"context_shift_probe_result={result!r}; expected 'passed'. "
            "--context-shift may not have taken effect or the probe timed out."
        )
