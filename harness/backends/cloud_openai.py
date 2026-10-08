"""OpenAI cloud backend implementation routed through ReplayCache and the capped cloud client.

Every live request goes through src/cloud/client.py's CloudClient in real mode (USD 50 total hard cap,
ledger at results/cloud_ledger.jsonl, alerts at 50/75/90%). The key is the explicit api_key argument,
else OPENAI_API_KEY, else CLOUD_API_KEY; with none set, construction raises MissingApiKeyError (as the
bare OpenAI SDK constructor used here before 2026-10-08 also raised without a key). The cap check runs
before Backend.model_call's timer starts and the ledger write after it stops, so recorded_latency_ms
covers the same work as before (the SDK request plus model_dump).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harness.backends.base import Backend
from harness.replay import ReplayMode
from src.cloud.client import DEFAULT_LEDGER_PATH, CloudClient


class CloudOpenAIBackend(Backend):
    """Cloud backend using OpenAI chat completions via the capped CloudClient."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        replay_mode: ReplayMode | str | None = None,
        traces_root: Path | None = None,
        cloud_client: CloudClient | None = None,
        ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    ) -> None:
        mode = replay_mode or os.environ.get("APU_REPLAY_MODE", "AUTO")
        root = traces_root or Path("analysis") / "traces"
        super().__init__(
            name="cloud_openai",
            default_model=model,
            is_cloud=True,
            replay_mode=mode,
            traces_root=root,
        )
        self.cloud = cloud_client or CloudClient.real(api_key=api_key, ledger_path=ledger_path)

    @staticmethod
    def _payload(tools, temperature, seed, kwargs: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = dict(kwargs)
        if tools is not None:
            payload["tools"] = tools
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed
        return payload

    def _prepare_capped_call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        seed: int | None,
        **kwargs: Any,
    ):
        return self.cloud.prepare(model_id=model, messages=messages,
                                  **self._payload(tools, temperature, seed, kwargs))

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
        """Untimed one-shot path (model_call uses _prepare_capped_call instead). Still capped."""
        result = self.cloud.call(model_id=model, messages=messages,
                                 **self._payload(tools, temperature, seed, kwargs))
        return result.raw
