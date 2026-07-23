"""Backend base class that routes model calls through ReplayCache."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from harness.replay import ReplayCache, ReplayMode


@dataclass(frozen=True)
class ModelCallResult:
    """Normalized model call result for all backends."""

    response_json: dict[str, Any]
    token_counts: dict[str, int]
    recorded_latency_ms: float
    replay_latency_ms: float
    replayed: bool
    cache_key: str


class BackendBase(ABC):
    """Base backend wrapper ensuring every model call passes through ReplayCache."""

    def __init__(self, replay_cache: ReplayCache | None = None) -> None:
        self.replay_cache = replay_cache or ReplayCache(mode=ReplayMode.AUTO)

    def model_call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> ModelCallResult:
        """Execute or replay a model call with cache metadata attached."""

        def _api_call() -> tuple[dict[str, Any], dict[str, int], float]:
            t0 = time.perf_counter_ns()
            response_obj = self._call_model_api(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                seed=seed,
                **kwargs,
            )
            recorded_latency_ms = (time.perf_counter_ns() - t0) / 1e6
            response_json = self._to_json(response_obj)
            token_counts = self._extract_token_counts(response_json)
            return response_json, token_counts, recorded_latency_ms

        replay_result = self.replay_cache.execute(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            seed=seed,
            api_call=_api_call,
        )

        return ModelCallResult(
            response_json=replay_result.response_json,
            token_counts=replay_result.token_counts,
            recorded_latency_ms=replay_result.recorded_latency_ms,
            replay_latency_ms=replay_result.replay_latency_ms,
            replayed=replay_result.replayed,
            cache_key=replay_result.key,
        )

    @staticmethod
    def _to_json(response_obj: Any) -> dict[str, Any]:
        if isinstance(response_obj, dict):
            return response_obj
        if hasattr(response_obj, "model_dump"):
            return response_obj.model_dump(exclude_unset=False)
        raise TypeError(f"Unsupported response object type: {type(response_obj)!r}")

    @staticmethod
    def _extract_token_counts(response_json: dict[str, Any]) -> dict[str, int]:
        usage = response_json.get("usage") or {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    @abstractmethod
    def _call_model_api(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        seed: int | None,
        **kwargs: Any,
    ) -> Any:
        """Backend-specific raw API invocation."""
        raise NotImplementedError
