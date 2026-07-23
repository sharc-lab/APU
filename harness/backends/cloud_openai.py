"""OpenAI cloud backend implementation routed through ReplayCache."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from harness.backends.base import Backend
from harness.replay import ReplayMode


class CloudOpenAIBackend(Backend):
    """Cloud backend using OpenAI chat completions."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        replay_mode: ReplayMode | str | None = None,
        traces_root: Path | None = None,
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
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

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
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **kwargs,
        }
        if tools is not None:
            payload["tools"] = tools
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed
        response = self.client.chat.completions.create(**payload)
        return response.model_dump(exclude_unset=False)
