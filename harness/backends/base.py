"""Backend interface with per-turn instrumentation and replay integration."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from harness.instrumentation import Category
from harness.replay import ReplayCache, ReplayMode

CATEGORY_FIELDS = ("cpu_ns", "wall_ns", "bytes_in", "bytes_out", "count")


@dataclass(frozen=True)
class ModelCallResult:
    """Normalized model call output for all backends."""

    backend: str
    model: str
    response_json: dict[str, Any]
    token_counts: dict[str, int]
    recorded_latency_ms: float
    replay_latency_ms: float
    replayed: bool
    cache_key: str
    per_turn_categories: dict[str, dict[str, int]]


class Backend(ABC):
    """Abstract backend for model calls with replay and instrumentation."""

    def __init__(
        self,
        *,
        name: str,
        default_model: str,
        is_cloud: bool,
        replay_mode: ReplayMode | str = ReplayMode.AUTO,
        traces_root=None,
    ) -> None:
        self.name = name
        self.default_model = default_model
        self.is_cloud = is_cloud
        self.replay_cache = ReplayCache(mode=replay_mode, traces_root=traces_root)

    def model_call(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> ModelCallResult:
        """Execute model call through replay cache and emit per-turn category metrics."""
        chosen_model = model or self.default_model
        categories = self._blank_categories()

        t_orch = time.perf_counter_ns()
        prompt_payload = json.dumps(messages, sort_keys=True)
        prompt_bytes = len(prompt_payload.encode("utf-8"))
        self._record(categories, Category.ORCH_SETUP.value, cpu_ns=time.perf_counter_ns() - t_orch)

        t_tok = time.perf_counter_ns()
        _ = prompt_payload.encode("utf-8")
        self._record(
            categories,
            Category.TOKENIZATION.value,
            cpu_ns=time.perf_counter_ns() - t_tok,
            bytes_in=prompt_bytes,
        )

        def _api_call() -> tuple[dict[str, Any], dict[str, int], float]:
            t0 = time.perf_counter_ns()
            response_json = self._provider_call(
                model=chosen_model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                seed=seed,
                **kwargs,
            )
            recorded_latency_ms = (time.perf_counter_ns() - t0) / 1e6
            token_counts = self._extract_token_counts(response_json)
            return response_json, token_counts, recorded_latency_ms

        replay_result = self.replay_cache.execute(
            model=chosen_model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            seed=seed,
            api_call=_api_call,
        )

        response_bytes = len(json.dumps(replay_result.response_json, sort_keys=True).encode("utf-8"))
        http_wall_ns = int(replay_result.recorded_latency_ms * 1e6)

        self._record(categories, Category.HTTP_CLIENT.value, wall_ns=http_wall_ns, bytes_out=response_bytes)
        self._record(
            categories,
            Category.CLIENT_HTTP.value,
            wall_ns=http_wall_ns,
            bytes_in=prompt_bytes,
            bytes_out=response_bytes,
        )

        t_fw = time.perf_counter_ns()
        # Parse+copy output as framework handling overhead.
        _ = replay_result.response_json.get("choices", [])
        self._record(categories, Category.FRAMEWORK.value, cpu_ns=time.perf_counter_ns() - t_fw)

        return ModelCallResult(
            backend=self.name,
            model=chosen_model,
            response_json=replay_result.response_json,
            token_counts=self._extract_token_counts(replay_result.response_json),
            recorded_latency_ms=replay_result.recorded_latency_ms,
            replay_latency_ms=replay_result.replay_latency_ms,
            replayed=replay_result.replayed,
            cache_key=replay_result.key,
            per_turn_categories=categories,
        )

    @staticmethod
    def _extract_token_counts(response_json: dict[str, Any]) -> dict[str, int]:
        usage = response_json.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _blank_categories() -> dict[str, dict[str, int]]:
        return {
            cat.value: {field: 0 for field in CATEGORY_FIELDS}
            for cat in Category
        }

    @staticmethod
    def _record(
        categories: dict[str, dict[str, int]],
        category: str,
        *,
        cpu_ns: int = 0,
        wall_ns: int = 0,
        bytes_in: int = 0,
        bytes_out: int = 0,
    ) -> None:
        row = categories[category]
        row["cpu_ns"] += int(cpu_ns)
        row["wall_ns"] += int(wall_ns if wall_ns else cpu_ns)
        row["bytes_in"] += int(bytes_in)
        row["bytes_out"] += int(bytes_out)
        row["count"] += 1

    @abstractmethod
    def _provider_call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        seed: int | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute backend-specific provider call and return normalized JSON."""
        raise NotImplementedError
