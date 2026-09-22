"""Tests for LlamaServerSession.tokenize() and make_count_fn().

Covers:
- /tokenize response-shape parsing: {"tokens": [int, ...]} → len()
- BOS handling: add_special=False is always passed regardless of model
- make_count_fn() fail-loud: raises RuntimeError when /tokenize is unreachable
  rather than silently falling back to the character heuristic
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import httpx
import pytest

from harness.llama_server import LlamaServerConfig, LlamaServerSession


def _make_session(port: int = 8181) -> LlamaServerSession:
    """Return a LlamaServerSession with a mock _proc so it appears started."""
    cfg = LlamaServerConfig(
        exe="/fake/llama-server",
        model="/fake/model.gguf",
        ctx_size=8192,
        port=port,
        platform="test",
        build_id="b10970",
    )
    session = LlamaServerSession(cfg)
    # Simulate a started session without spawning a real process.
    session._proc = MagicMock()
    session.n_ctx_slot = 8192
    session.context_shift_probe_result = "skipped"
    return session


# ── Response-shape parsing ────────────────────────────────────────────────────

class TestTokenizeResponseParsing:
    def test_five_token_response(self):
        """len(tokens) returned for a 5-token response."""
        session = _make_session()
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"tokens": [785, 3974, 13876, 38835, 34208]}
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_resp) as mock_post:
            count = session.tokenize("The quick brown fox jumps")

        assert count == 5

    def test_single_token_response(self):
        session = _make_session()
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"tokens": [12345]}
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_resp):
            count = session.tokenize("x")

        assert count == 1

    def test_empty_token_list(self):
        """Empty content tokenizes to 0 tokens."""
        session = _make_session()
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"tokens": []}
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_resp):
            count = session.tokenize("")

        assert count == 0

    def test_url_uses_configured_port(self):
        session = _make_session(port=9999)
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"tokens": [1, 2, 3]}
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_resp) as mock_post:
            session.tokenize("hello")

        url_called = mock_post.call_args[0][0]
        assert "9999" in url_called


# ── BOS handling: add_special=False always passed ────────────────────────────

class TestBOSHandling:
    """Verify add_special=False is always sent in the POST payload.

    Background:
    - Qwen3: no BOS configured in tokenizer; add_special has no effect,
      but passing False documents intent and guards against future change.
    - Llama 3.1: BOS token 128000 (<|begin_of_text|>) IS configured.
      Default or add_special=True prepends it, over-counting filler by 1.
      add_special=False prevents this; REQUIRED for correct filler calibration.
    """

    def test_tokenize_sends_add_special_false(self):
        session = _make_session()
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"tokens": [1, 2, 3]}
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_resp) as mock_post:
            session.tokenize("some filler text")

        payload_sent = mock_post.call_args.kwargs["json"]
        assert payload_sent.get("add_special") is False

    def test_tokenize_never_sends_add_special_true(self):
        """No code path in tokenize() should set add_special=True."""
        session = _make_session()
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"tokens": [1]}
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_resp) as mock_post:
            session.tokenize("text")

        payload_sent = mock_post.call_args.kwargs["json"]
        assert payload_sent.get("add_special") is not True

    def test_make_count_fn_callable_sends_add_special_false(self):
        """The callable returned by make_count_fn() also sends add_special=False."""
        session = _make_session()
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"tokens": [10, 20, 30]}
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_resp) as mock_post:
            count_fn = session.make_count_fn()
            # make_count_fn does a probe call; reset and call via count_fn
            mock_post.reset_mock()
            mock_post.return_value = fake_resp
            result = count_fn("filler text")

        payload_sent = mock_post.call_args.kwargs["json"]
        assert payload_sent.get("add_special") is False
        assert result == 3


# ── Fail-loud: unreachable server ─────────────────────────────────────────────

class TestMakeCountFnFailLoud:
    """make_count_fn() must raise RuntimeError if /tokenize is unreachable.

    This prevents silent fallback to the character heuristic: a
    misconfigured filler-calibration path must fail before the sweep begins,
    not silently produce underestimated fillers.
    """

    def test_raises_runtimeerror_on_connection_refused(self):
        session = _make_session()
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(RuntimeError, match=r"/tokenize not reachable"):
                session.make_count_fn()

    def test_raises_runtimeerror_on_timeout(self):
        session = _make_session()
        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(RuntimeError, match=r"/tokenize not reachable"):
                session.make_count_fn()

    def test_raises_runtimeerror_on_http_500(self):
        session = _make_session()
        fake_resp = MagicMock()
        fake_resp.status_code = 500
        fake_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=fake_resp
        )
        with patch("httpx.post", return_value=fake_resp):
            with pytest.raises(RuntimeError, match=r"/tokenize not reachable"):
                session.make_count_fn()

    def test_does_not_catch_and_swallow_error(self):
        """Verify the error propagates as RuntimeError, not None or 0."""
        session = _make_session()
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            result = None
            try:
                count_fn = session.make_count_fn()
                result = count_fn("test")
            except RuntimeError:
                result = "raised"
            except Exception:
                result = "other"
        assert result == "raised"

    def test_raises_runtimeerror_if_session_not_started(self):
        """tokenize() and make_count_fn() require an active session."""
        cfg = LlamaServerConfig(
            exe="/fake/llama-server", model="/fake/model.gguf",
            ctx_size=8192, port=8181,
        )
        session = LlamaServerSession(cfg)
        # _proc is None — session not started

        with pytest.raises(RuntimeError, match="not started"):
            session.tokenize("text")

        with pytest.raises(RuntimeError, match="not started"):
            session.make_count_fn()

    def test_success_returns_callable(self):
        """When /tokenize responds correctly, make_count_fn returns a callable."""
        session = _make_session()
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"tokens": [1, 2]}
        fake_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=fake_resp):
            count_fn = session.make_count_fn()

        assert callable(count_fn)
