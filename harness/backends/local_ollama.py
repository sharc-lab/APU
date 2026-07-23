"""Ollama local backend implementation routed through ReplayCache."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from harness.backends.base import Backend
from harness.replay import ReplayMode


class LocalOllamaBackend(Backend):
    """Local backend using Ollama HTTP API."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        replay_mode: ReplayMode | str | None = None,
        traces_root: Path | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        mode = replay_mode or os.environ.get("APU_REPLAY_MODE", "AUTO")
        root = traces_root or Path("analysis") / "traces_ollama"
        super().__init__(
            name="local_ollama",
            default_model=model or os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b"),
            is_cloud=False,
            replay_mode=mode,
            traces_root=root,
        )
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)

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
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if seed is not None:
            options["seed"] = seed

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if tools:
            payload["tools"] = tools

        response = self.client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        raw = response.json()

        prompt_tokens = int(raw.get("prompt_eval_count") or 0)
        completion_tokens = int(raw.get("eval_count") or 0)
        finish_reason = raw.get("done_reason") or "stop"

        return {
            "id": raw.get("id") or "ollama-chat",
            "object": "chat.completion",
            "model": raw.get("model") or model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": raw.get("message") or {"role": "assistant", "content": ""},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            },
        }
