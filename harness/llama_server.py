"""llama-server b10970 backend for the context-degradation sweep.

Architecture
------------
One ``LlamaServerSession`` manages one persistent llama-server process. Start it
once at the beginning of a sweep configuration block, use ``session.call()`` for
every probe, then ``session.stop()`` (or use it as a context manager).

Interface contract
------------------
``LlamaServerSession.call()`` returns the same 7-tuple as
``_call_ollama_streaming`` in harness/runner.py::

    (text, latency_ms, ttft_ms, tokens_in, tokens_out, done_reason, thinking_chars)

This means ``run_cell`` can swap in ``session.call`` for ``_call_ollama_streaming``
without structural change. Additional per-session metadata (``n_ctx_slot``,
``server_session_id``, etc.) is available as attributes and collected via
``session.row_metadata()``.

Thinking suppression
--------------------
Uses ``--reasoning-budget 0`` as a server launch flag, not the ``/no_think``
prompt injection. Rationale: ``/no_think`` is in-band — it occupies context
tokens and introduces an instruction-following signal that could confound quality
measurements. ``--reasoning-budget 0`` is enforced by the inference engine before
any prompt content is evaluated, so the test prompt is unmodified and the context
window is uncontaminated.

Positive control
----------------
When ``context_shift=True``, ``start()`` runs a generation-overflow probe that
asserts the ``slot context shift`` warning appears in the server log. The probe
streams a generation request and polls the log in a background thread; it breaks
out of the stream as soon as the shift is confirmed, so startup overhead is
bounded (~1 shift event worth of generation rather than ``max_tokens`` tokens).

Context-size rejection
----------------------
HTTP 400 ``exceed_context_size_error`` is raised as ``ContextSizeError``, a
distinct exception type callers can catch to record the prompt as
``context_size_exceeded=True`` rather than treating it as an infrastructure
failure.

Verified facts this module is built against
-------------------------------------------
- Reject behaviour holds at ctx 8192 and 32768, on both /v1/chat/completions
  and /completion.  Verified on evo-t2s, llama-server b10970 Vulkan.
- n_ctx_slot from the startup log is ground truth; server silently raises
  ctx_size below 256 to 256.
- context-shift applies only to generation overflow, not to oversized prompts.
- Eviction: n_left = n_ctx_slot - 1, n_discard = (n_ctx_slot - 1) // 2,
  oldest-first, nothing pinned at n_keep = 0.
- --context-shift emits no startup log evidence; positive control is empirical.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

_LOG = logging.getLogger(__name__)

# ── Exceptions ─────────────────────────────────────────────────────────────────


class ContextSizeError(Exception):
    """HTTP 400 exceed_context_size_error from llama-server.

    Distinct from a generic HTTP error: the prompt was too long for the
    configured context window.  Callers should catch this and record
    ``context_size_exceeded=True`` as the row outcome rather than treating it
    as an infrastructure failure.

    Attributes
    ----------
    n_prompt_tokens : int
        Token count reported by the server for the rejected prompt.
    n_ctx : int
        Effective context size (== ``n_ctx_slot``) at the time of rejection.
    raw : str
        Full response body from the server.
    """

    def __init__(self, n_prompt_tokens: int, n_ctx: int, raw: str) -> None:
        self.n_prompt_tokens = n_prompt_tokens
        self.n_ctx = n_ctx
        self.raw = raw
        super().__init__(
            f"request ({n_prompt_tokens} tokens) exceeds context size ({n_ctx} tokens)"
        )


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LlamaServerConfig:
    """All parameters that define one persistent llama-server session.

    ``n_ctx_slot`` is NOT set here; it is read from the startup log after the
    server starts, because the server silently raises ctx_size below 256.
    """

    exe: str
    """Absolute path to llama-server.exe."""

    model: str
    """Absolute path to .gguf model file."""

    ctx_size: int
    """Requested context size passed as --ctx-size. n_ctx_slot in the log may differ."""

    port: int
    """TCP port the server will listen on."""

    n_gpu_layers: int = 99
    """Passed as --n-gpu-layers. 99 = offload everything available."""

    context_shift: bool = False
    """Whether to pass --context-shift. Default off (server default in b10970)."""

    reasoning_budget: int | None = None
    """Passed as --reasoning-budget when not None. Use 0 to suppress thinking."""

    reasoning_format: str = "deepseek"
    """Passed as --reasoning-format. 'deepseek' ensures thinking always goes into
    delta.reasoning_content (never inline <think> tags in content). Verified
    empirically on b10970: default ('auto') also uses reasoning_content, but
    setting this explicitly makes the field location deterministic. Use 'none'
    only if you want <think> tags in content — that is NOT suppression."""

    platform: str = "unknown"
    """Machine identifier recorded on every row. E.g. "evo-t2s", "blade14"."""

    build_id: str = "b10970"
    """llama-server build identifier recorded on every row."""

    backend: str = "vulkan"
    """Inference backend recorded on every row. "vulkan" or "cuda"."""

    startup_timeout_s: float = 120.0
    """Seconds to wait for the server to pass /health before giving up."""

    log_path: str | None = None
    """Where to write the server log. If None, a temp file is created."""

    model_alias: str = "local"
    """Value sent as 'model' in API payloads. llama-server ignores it."""


# ── Log parsing ────────────────────────────────────────────────────────────────

_N_CTX_SLOT_RE = re.compile(r"n_ctx_slot\s*=\s*(\d+)")


def _parse_n_ctx_slot(log_text: str) -> int | None:
    """Return n_ctx_slot from a llama-server startup log, or None if absent.

    Looks for the canonical line::

        I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 256, ...
    """
    m = _N_CTX_SLOT_RE.search(log_text)
    return int(m.group(1)) if m else None


def _log_has_context_shift(log_text: str) -> bool:
    """Return True if the log contains a 'slot context shift' warning."""
    return "slot context shift" in log_text


# ── HTTP error parsing ─────────────────────────────────────────────────────────


def _raise_if_context_size_error(body: str) -> None:
    """Parse a 400 response body and raise ContextSizeError if appropriate.

    Handles both direct llama-server responses and Ollama-wrapped responses
    (where the inner JSON is encoded as a string in the outer ``error`` field).
    """
    try:
        outer = json.loads(body)
        err = outer.get("error", {})
        # Ollama wraps the inner JSON as a string; unwrap it.
        if isinstance(err, str):
            try:
                err = json.loads(err).get("error", {})
            except (json.JSONDecodeError, AttributeError):
                return
        if isinstance(err, dict) and err.get("type") == "exceed_context_size_error":
            raise ContextSizeError(
                n_prompt_tokens=int(err.get("n_prompt_tokens", 0)),
                n_ctx=int(err.get("n_ctx", 0)),
                raw=body,
            )
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass


# ── Server health poll ─────────────────────────────────────────────────────────


def _wait_for_server(port: int, timeout_s: float) -> bool:
    """Poll GET /health until 200 or timeout. Returns True if ready."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


# ── Streaming call ─────────────────────────────────────────────────────────────

_STREAM_DONE_SENTINEL = "data: [DONE]"


def _stream_chat(
    port: int,
    prompt: str,
    max_tokens: int,
    model_alias: str = "local",
    timeout_s: float = 300.0,
) -> tuple[str, float, float | None, int, int, str | None, int]:
    """POST to /v1/chat/completions with stream=True and return the 7-tuple.

    Return signature matches ``_call_ollama_streaming`` in harness/runner.py::

        (text, latency_ms, ttft_ms, tokens_in, tokens_out, done_reason, thinking_chars)

    - ``ttft_ms``: wall time to first content token (``delta.content``), or None
      if no content token arrived.
    - ``thinking_chars``: accumulated length of ``delta.reasoning_content`` across
      all chunks.  Zero when ``--reasoning-budget 0`` suppresses thinking.
      Empirically verified on b10970: with ``--reasoning-format deepseek`` (the
      default in LlamaServerConfig) thinking tokens always appear in
      ``reasoning_content``, never as ``<think>`` tags in ``content``.
    - ``ttft_source`` is always ``"streamed"`` for non-None ``ttft_ms`` (set by the
      caller when recording into the result row).

    Raises
    ------
    ContextSizeError
        HTTP 400 exceed_context_size_error.
    httpx.HTTPStatusError
        Any other non-200 response.
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model_alias,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    ttft_ms: float | None = None
    full_text = ""
    thinking_chars = 0
    tokens_in = 0
    tokens_out = 0
    done_reason: str | None = None

    with httpx.stream("POST", url, json=payload, timeout=timeout_s) as resp:
        if resp.status_code == 400:
            body = resp.read().decode("utf-8", errors="replace")
            _raise_if_context_size_error(body)
            resp.raise_for_status()
        elif resp.status_code != 200:
            resp.raise_for_status()

        for raw in resp.iter_lines():
            raw = raw.strip()
            if not raw or raw == _STREAM_DONE_SENTINEL:
                continue
            if raw.startswith("data: "):
                raw = raw[6:]
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}

            content_tok: str = delta.get("content") or ""
            reasoning_tok: str = delta.get("reasoning_content") or ""
            thinking_chars += len(reasoning_tok)

            if content_tok and ttft_ms is None:
                ttft_ms = (time.perf_counter() - t_start) * 1000
            full_text += content_tok

            if choices:
                finish = choices[0].get("finish_reason")
                if finish:
                    done_reason = finish

            usage = chunk.get("usage")
            if usage:
                tokens_in = int(usage.get("prompt_tokens") or 0)
                tokens_out = int(usage.get("completion_tokens") or 0)

    latency_ms = (time.perf_counter() - t_start) * 1000
    return full_text, latency_ms, ttft_ms, tokens_in, tokens_out, done_reason, thinking_chars


# ── Context-shift positive control ────────────────────────────────────────────

_OVERFLOW_PROBE_PROMPT = (
    "Count every integer from 1 upward, one per line, with no extra text. "
    "Start with 1 and continue until told to stop."
)


def _run_overflow_probe(
    port: int,
    n_ctx_slot: int,
    log_path: str,
    model_alias: str = "local",
    probe_timeout_s: float = 300.0,
) -> str:
    """Run a generation-overflow probe and return the probe result string.

    Sends a short prompt requesting ``n_ctx_slot * 2`` tokens; the generation
    must overflow the context, which should trigger the ``slot context shift``
    warning in the log.

    Uses a background polling thread to detect the shift event and break out of
    the HTTP stream early — the request is aborted as soon as the shift is
    confirmed rather than waiting for ``max_tokens`` tokens to be generated.

    Returns
    -------
    "passed"
        "slot context shift" was seen in the log within ``probe_timeout_s``.
    "failed"
        The probe completed without observing a shift, or the request failed.
    "timeout"
        The probe did not complete within ``probe_timeout_s``.
    """
    shift_seen = threading.Event()
    stop_poll = threading.Event()

    def _poll() -> None:
        while not stop_poll.wait(1.5):
            try:
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
                if _log_has_context_shift(text):
                    shift_seen.set()
                    return
            except Exception:
                pass

    poll_thread = threading.Thread(target=_poll, daemon=True, name="shift-probe-poll")
    poll_thread.start()

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model_alias,
        "messages": [{"role": "user", "content": _OVERFLOW_PROBE_PROMPT}],
        "stream": True,
        "max_tokens": n_ctx_slot * 2,
        "temperature": 0,
    }

    result = "failed"
    try:
        with httpx.stream("POST", url, json=payload, timeout=probe_timeout_s) as resp:
            if resp.status_code != 200:
                return "failed"
            for _ in resp.iter_lines():
                if shift_seen.is_set():
                    result = "passed"
                    break
        if not shift_seen.is_set():
            # Stream exhausted; check log one final time
            try:
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
                if _log_has_context_shift(text):
                    result = "passed"
            except Exception:
                pass
    except httpx.ReadTimeout:
        result = "timeout"
    except ContextSizeError:
        # Probe prompt should fit; this is unexpected but not a shift failure.
        result = "failed"
    except Exception as exc:
        _LOG.warning("Overflow probe raised: %s", exc)
        result = "failed"
    finally:
        stop_poll.set()
        poll_thread.join(timeout=5)

    return result


# ── Session ────────────────────────────────────────────────────────────────────


class LlamaServerSession:
    """Manages one persistent llama-server process.

    Usage::

        cfg = LlamaServerConfig(exe=..., model=..., ctx_size=8192, port=8181,
                                context_shift=False, reasoning_budget=0,
                                platform="evo-t2s")
        with LlamaServerSession(cfg) as session:
            for prompt in prompts:
                try:
                    text, lat, ttft, tin, tout, reason, think = session.call(prompt, 256)
                    row = {"output": text, "latency_ms": lat, ...}
                    row.update(session.row_metadata())
                except ContextSizeError as e:
                    row = {"context_size_exceeded": True, "n_prompt_tokens": e.n_prompt_tokens}
                    row.update(session.row_metadata())

    Attributes available after ``start()``
    ---------------------------------------
    n_ctx_slot : int
        Effective context size read from the server startup log.  Ground truth;
        may differ from ``cfg.ctx_size`` when ctx_size < 256.
    server_session_id : str
        UUID4 generated at session creation.  Stamped on every row to link
        measurements to the session whose R1 overflow probe verified
        ``--context-shift``.
    context_shift_probe_result : str
        "passed" | "failed" | "timeout" | "skipped".  "skipped" when
        ``cfg.context_shift`` is False.
    thinking_suppression_verified : bool | None
        True if a test prompt produced thinking_chars == 0 after
        ``--reasoning-budget 0``. None when not tested.
    """

    def __init__(self, cfg: LlamaServerConfig) -> None:
        self.cfg = cfg
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_path: str | None = None
        self._log_fh: Any = None

        # Populated in start()
        self.server_session_id: str = str(uuid.uuid4())
        self.n_ctx_slot: int | None = None
        self.context_shift_probe_result: str | None = None
        self.thinking_suppression_verified: bool | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "LlamaServerSession":
        """Launch the server process, parse n_ctx_slot, run startup probes.

        Returns self for chaining / context-manager use.

        Raises
        ------
        RuntimeError
            If already started, server did not become ready within
            ``cfg.startup_timeout_s``, or n_ctx_slot could not be parsed.
        """
        if self._proc is not None:
            raise RuntimeError("Session already started")

        if self.cfg.log_path:
            self._log_path = self.cfg.log_path
        else:
            fd, self._log_path = tempfile.mkstemp(
                prefix=f"llamasrv_{self.cfg.port}_",
                suffix=".log",
            )
            os.close(fd)

        cmd = self._build_cmd()
        _LOG.info("Starting llama-server: %s", " ".join(cmd))

        self._log_fh = open(self._log_path, "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_fh,
            stderr=self._log_fh,
        )

        if not _wait_for_server(self.cfg.port, self.cfg.startup_timeout_s):
            self.stop()
            raise RuntimeError(
                f"llama-server port {self.cfg.port} not ready within "
                f"{self.cfg.startup_timeout_s}s. Log: {self._log_path}"
            )

        log_text = Path(self._log_path).read_text(encoding="utf-8", errors="replace")
        self.n_ctx_slot = _parse_n_ctx_slot(log_text)
        if self.n_ctx_slot is None:
            self.stop()
            raise RuntimeError(
                f"Could not parse n_ctx_slot from startup log. Log: {self._log_path}"
            )

        # Startup probes
        if self.cfg.context_shift:
            self.context_shift_probe_result = _run_overflow_probe(
                port=self.cfg.port,
                n_ctx_slot=self.n_ctx_slot,
                log_path=self._log_path,
                model_alias=self.cfg.model_alias,
            )
        else:
            self.context_shift_probe_result = "skipped"

        if self.cfg.reasoning_budget == 0:
            self.thinking_suppression_verified = self._verify_thinking_suppressed()

        return self

    def is_alive(self) -> bool:
        """Return True if the server process is running."""
        return self._proc is not None and self._proc.poll() is None

    def restart(self, new_config: "LlamaServerConfig | None" = None) -> "LlamaServerSession":
        """Stop, optionally apply a new config, and start again.

        Generates a fresh ``server_session_id`` so rows recorded before and
        after the restart are distinguishable in results.

        Parameters
        ----------
        new_config:
            If provided, replaces ``self.cfg`` before restarting.  If None,
            the existing config is reused.

        Returns
        -------
        self
            For chaining.
        """
        self.stop()
        if new_config is not None:
            self.cfg = new_config
        self.server_session_id = str(uuid.uuid4())
        self.n_ctx_slot = None
        self.context_shift_probe_result = None
        self.thinking_suppression_verified = None
        self._proc = None
        return self.start()

    def stop(self) -> None:
        """Kill the server process and close the log file."""
        if self._proc is not None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            self._proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def __enter__(self) -> "LlamaServerSession":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ── Model call ─────────────────────────────────────────────────────────────

    def call(
        self,
        prompt: str,
        max_tokens: int,
        timeout_s: float = 300.0,
    ) -> tuple[str, float, float | None, int, int, str | None, int]:
        """Call the server and return the standard 7-tuple.

        Returns
        -------
        tuple
            ``(text, latency_ms, ttft_ms, tokens_in, tokens_out, done_reason,
            thinking_chars)``

            Matches the return signature of ``_call_ollama_streaming`` in
            ``harness/runner.py`` so ``run_cell`` can use it without structural
            change.

        Raises
        ------
        ContextSizeError
            Prompt exceeds ``n_ctx_slot``.  Catch to record as an outcome row.
        RuntimeError
            Session not started.
        """
        if self._proc is None:
            raise RuntimeError("Session not started; call start() or use as context manager")
        return _stream_chat(
            port=self.cfg.port,
            prompt=prompt,
            max_tokens=max_tokens,
            model_alias=self.cfg.model_alias,
            timeout_s=timeout_s,
        )

    # ── Row metadata ───────────────────────────────────────────────────────────

    def row_metadata(self) -> dict[str, Any]:
        """Return fields to merge into every result row produced by this session.

        Fields
        ------
        server_session_id : str
            Links rows to the session that ran the startup probes.
        n_ctx_slot : int | None
            Effective context size from the startup log.
        build_id : str
            llama-server build identifier (e.g. "b10970").
        backend : str
            Inference backend (e.g. "vulkan", "cuda").
        platform : str
            Hardware identifier (e.g. "evo-t2s", "blade14").
        context_shift_probe_result : str | None
            Startup probe outcome: "passed" | "failed" | "timeout" | "skipped".
        """
        return {
            "server_session_id": self.server_session_id,
            "n_ctx_slot": self.n_ctx_slot,
            "build_id": self.cfg.build_id,
            "backend": self.cfg.backend,
            "platform": self.cfg.platform,
            "context_shift_probe_result": self.context_shift_probe_result,
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.cfg.exe,
            "-m", self.cfg.model,
            "--ctx-size", str(self.cfg.ctx_size),
            "--n-gpu-layers", str(self.cfg.n_gpu_layers),
            "--port", str(self.cfg.port),
            "--log-file", self._log_path,
            "--log-verbosity", "3",
            "-np", "1",
        ]
        cmd.append("--context-shift" if self.cfg.context_shift else "--no-context-shift")
        cmd += ["--reasoning-format", self.cfg.reasoning_format]
        if self.cfg.reasoning_budget is not None:
            cmd += ["--reasoning-budget", str(self.cfg.reasoning_budget)]
        return cmd

    def _verify_thinking_suppressed(self) -> bool | None:
        """Send a reasoning-heavy prompt and return whether thinking_chars == 0.

        Returns
        -------
        True
            ``thinking_chars == 0``: suppression confirmed.
        False
            ``thinking_chars > 0``: thinking is NOT suppressed despite
            ``--reasoning-budget 0``.
        None
            The check itself raised an exception (infrastructure failure,
            timeout, etc.).  Cannot distinguish from a suppression failure;
            do not treat as confirmation in either direction.
        """
        probe = "What is the exact value of the integral from 0 to 1 of x^2 dx?"
        try:
            _, _, _, _, _, _, thinking_chars = _stream_chat(
                port=self.cfg.port,
                prompt=probe,
                max_tokens=60,
                model_alias=self.cfg.model_alias,
                timeout_s=30.0,
            )
            return thinking_chars == 0
        except Exception as exc:
            _LOG.warning("Thinking suppression check raised: %s", exc)
            return None
