"""Anthropic cloud backend with budget guard and prompt caching (minimal M3.2).

Three properties matter more than convenience here.

**No call is issued without authorization.** :meth:`AnthropicBackend.generate` asks the
:class:`~seam.budget.BudgetGuard` first, and the guard refuses unless the remaining budget covers
the call's worst case. The ceiling is therefore structural, not observational.

**Cost comes from the response, not from an estimate.** Anthropic returns ``usage`` with separate
``cache_read_input_tokens`` and ``cache_creation_input_tokens``; conflating those with plain input
tokens misprices a cached run by roughly 10x in the read direction. The local estimate is used for
the pre-flight projection only, and projected-vs-actual goes into the ledger so the projection
self-calibrates.

**Retries are visible.** Spec §9.6 forbids silent retries. Exactly one retry is permitted; after
that the harness falls back to local and **marks** the step, because a fallback perturbs the
realized partition and must be visible in analysis rather than absorbed into it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Final

from seam.backends.base import (
    GenerationRequest,
    GenerationResult,
    PreflightVerdict,
    ToolCall,
    ToolSpec,
)
from seam.budget import BudgetGuard, CallUsage
from seam.credentials import require_api_key
from seam.errors import BackendError
from seam.jsonlog import log_event

__all__ = ["AnthropicBackend", "is_retryable_protocol_error"]

_API_KEY_ENV: Final = "ANTHROPIC_API_KEY"

#: Substrings that mark a protocol-level disconnect as retryable. Auth/budget errors are NOT
#: retryable - retrying them burns money without a chance of success.
_RETRYABLE_MARKERS: Final = (
    "connection",
    "timeout",
    "temporarily",
    "tls",
    "ssl",
    "reset",
    "broken pipe",
    "remote end closed",
    "server disconnected",
    "api connection",
)


def is_retryable_protocol_error(exc: BaseException) -> bool:
    """True when the failure is a protocol-level disconnect, not an auth or schema error.

    Rationale (recorded, not assumed): ``api.anthropic.com`` probed 8/8 healthy in the
    contemporaneous TLS session, but with n=8 and zero failures the rule of three puts the 95%
    upper bound on the failure rate near 31%, *not* near zero. The probe establishes "not in the
    badly-degraded class", not "clean". H1 makes hundreds of calls; a 5% rate is invisible to that
    probe and very visible in collection. Protocol disconnects are therefore treated as retryable
    (exactly once) and logged as events.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    blob = f"{name} {text}"
    if any(tok in blob for tok in ("unauthorized", "authentication", "invalid api", "permission")):
        return False
    return any(tok in blob for tok in _RETRYABLE_MARKERS)


class AnthropicBackend:
    """Cloud execution against a pinned Anthropic snapshot."""

    def __init__(
        self,
        *,
        model_id: str,
        guard: BudgetGuard,
        max_tokens: int,
        temperature: float = 0.0,
        prompt_caching: bool = True,
        cache_ttl: str = "5m",
        max_retries: int = 1,
        run_id: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._guard = guard
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._prompt_caching = prompt_caching
        self._cache_ttl = cache_ttl
        self._max_retries = max_retries
        self._run_id = run_id
        self._client: Any = None
        #: After the first successful call with prompt caching, later estimates may use warm rates.
        self._cache_prefix_written: bool = False

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model_ref(self) -> str:
        return self._model_id

    def set_run_id(self, run_id: str) -> None:
        self._run_id = run_id

    # ---------------------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        # Read from the environment (optionally seeded from a gitignored .env). Refuses rather
        # than degrading; the key is never logged, echoed, or written to a manifest.
        api_key = require_api_key(_API_KEY_ENV)
        import anthropic

        # Retries are the harness's business, not the SDK's: an SDK-internal retry would spend
        # money without appearing in the ledger or the event log.
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0)
        return self._client

    def preflight(self) -> PreflightVerdict:
        try:
            self._ensure_client()
        except BackendError as exc:
            return PreflightVerdict("UNSUPPORTED", reason=str(exc))
        except Exception as exc:  # reported as a verdict, never swallowed
            return PreflightVerdict("ERROR", reason=f"{type(exc).__name__}: {exc}")
        return PreflightVerdict(
            "OK",
            detail={
                "model_id": self._model_id,
                "pin_convention": self._guard.pricing.pin_convention_by_model.get(
                    self._model_id, "unspecified"
                ),
                "pricing_version": self._guard.pricing.version,
            },
        )

    # ---------------------------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------------------------

    def _render_tools(self, tools: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        # One breakpoint at the end of the tool block. The tool definitions are byte-identical
        # across every task in the slice, so this prefix is the largest reliably cacheable span;
        # a 5m cache pays for itself after a single read.
        if rendered and self._prompt_caching:
            rendered[-1] = {
                **rendered[-1],
                "cache_control": {"type": "ephemeral", "ttl": self._cache_ttl},
            }
        return rendered

    def generate(self, request: GenerationRequest) -> GenerationResult:
        client = self._ensure_client()
        tools = self._render_tools(request.tools)

        # Pre-flight projection. Authorization always uses the cold/write bound. The logged
        # estimate uses cache state when known so warm-prefix calls are not 67% over-projected
        # (Phase E 0.3). Synthetic 1-tok/byte payloads must NOT be used to recalibrate the
        # byte→token slope (Phase E 0.2 / AUDIT_LOG).
        est_prompt_tokens = _estimate_prompt_tokens(request, self._guard)
        cache_state = "warm" if (tools and self._prompt_caching) else "cold"
        # First call after a process start cannot assume a warm cache; subsequent calls that
        # share the tool/system prefix can. The backend tracks whether a cacheable prefix was
        # already written in this process.
        if not getattr(self, "_cache_prefix_written", False):
            cache_state = "cold"
        projected = self._guard.projected_call_usd(
            model=self._model_id,
            prompt_tokens=est_prompt_tokens,
            max_tokens=self._max_tokens,
            cache_state=cache_state,  # type: ignore[arg-type]
        )
        # Refuses unless the remaining budget covers the WORST case, so no call can overshoot.
        self._guard.authorize_call(
            model=self._model_id,
            prompt_tokens=est_prompt_tokens,
            max_tokens=self._max_tokens,
        )

        system_block: list[dict[str, Any]] = [{"type": "text", "text": request.system}]
        attempts = self._max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            t0 = time.perf_counter_ns()
            try:
                # claude-sonnet-5 rejects `temperature` (400: deprecated for this model).
                # Observed 2026-08-02 on Phase E validation; omit rather than send a no-op.
                create_kwargs: dict[str, Any] = {
                    "model": self._model_id,
                    "max_tokens": self._max_tokens,
                    "system": system_block,
                    "tools": tools,
                    "messages": request.messages,
                }
                if self._model_id != "claude-sonnet-5":
                    create_kwargs["temperature"] = self._temperature
                response = client.messages.create(**create_kwargs)
            except Exception as exc:  # logged as an event, then re-raised or retried
                last_error = exc
                retryable = is_retryable_protocol_error(exc)
                will_retry = retryable and attempt < attempts - 1
                log_event(
                    "backend.cloud.call_failed",
                    severity="warning" if will_retry else "error",
                    message=f"attempt {attempt + 1}/{attempts}: {type(exc).__name__}: {exc}",
                    attempt=attempt + 1,
                    max_attempts=attempts,
                    model=self._model_id,
                    run_id=self._run_id,
                    retryable=retryable,
                    retried=will_retry,
                )
                if not will_retry:
                    break
                continue

            wall_ns = time.perf_counter_ns() - t0
            usage = _usage_from_response(response)
            actual = self._guard.record(
                model=self._model_id,
                usage=usage,
                projected_usd=projected,
                run_id=self._run_id,
            )
            if self._prompt_caching and (
                usage.cache_creation_input_tokens > 0 or usage.cache_read_input_tokens > 0
            ):
                self._cache_prefix_written = True

            text, calls = _content_to_text_and_calls(response)
            return GenerationResult(
                text=text,
                tool_calls=calls,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                completion_chars=len(text),
                completion_bytes=len(text.encode("utf-8")),
                wall_ns=wall_ns,
                ttft_ns=None,
                usd_cost=actual,
                backend=self.name,
                model_ref=self._model_id,
                # Echoed back by the API. A mismatch means the pin did not hold (AM-020).
                reported_model=str(getattr(response, "model", "") or "") or None,
                retried=attempt > 0,
                extra={
                    "stop_reason": getattr(response, "stop_reason", None),
                    "projected_usd": projected,
                    # Anthropic returns cache_read_input_tokens / cache_creation_input_tokens, so a
                    # zero here IS a measurement of no reuse - unlike the local path.
                    "cache_instrumented": True,
                },
            )

        raise BackendError(
            f"cloud call failed after {attempts} attempt(s): {type(last_error).__name__}: "
            f"{last_error}. The harness falls back to local and MARKS the step; the fallback is "
            f"logged as an event because it perturbs the realized partition."
        )


def _estimate_prompt_tokens(request: GenerationRequest, guard: BudgetGuard) -> int:
    """Rough local prompt-token estimate for the pre-flight projection only.

    Four characters per token is the usual English approximation. It is wrong in a known
    direction for JSON-heavy tool schemas, which is precisely why the ledger tracks
    projected-vs-actual instead of trusting this number.
    """
    chars = len(request.system)
    for message in request.messages:
        content = message.get("content")
        chars += len(content) if isinstance(content, str) else len(str(content))
    for tool in request.tools:
        chars += len(tool.name) + len(tool.description) + len(str(tool.input_schema))
    overhead = guard.pricing.tool_use_overhead_tokens.get("tool_choice_auto", 0)
    return chars // 4 + overhead


def _usage_from_response(response: Any) -> CallUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        raise BackendError(
            "cloud response carried no usage block; cost cannot be computed from the provider's "
            "own accounting and a local estimate is not an acceptable substitute for billing."
        )
    return CallUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
    )


def _content_to_text_and_calls(response: Any) -> tuple[str, tuple[ToolCall, ...]]:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in getattr(response, "content", []) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            texts.append(str(getattr(block, "text", "")))
        elif kind == "tool_use":
            raw_args = getattr(block, "input", {})
            calls.append(
                ToolCall(
                    name=str(getattr(block, "name", "")),
                    arguments=raw_args if isinstance(raw_args, dict) else {},
                    call_id=str(getattr(block, "id", "") or "") or None,
                )
            )
    return "\n".join(texts), tuple(calls)


def load_pricing_path(config_root: Path, relative: str) -> Path:
    return (config_root / relative).resolve()
