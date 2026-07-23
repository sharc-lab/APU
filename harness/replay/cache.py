"""Replay cache for deterministic model-call record/replay flows."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
import re
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class ReplayMode(str, Enum):
    """Modes that control cache behavior for model calls."""

    RECORD = "RECORD"
    REPLAY = "REPLAY"
    AUTO = "AUTO"


class ReplayCacheMissError(FileNotFoundError):
    """Raised when REPLAY mode is active and an entry is missing."""


@dataclass(frozen=True)
class ReplayResult:
    """Result returned by ReplayCache for each model invocation."""

    key: str
    response_json: dict[str, Any]
    token_counts: dict[str, int]
    recorded_latency_ms: float
    replay_latency_ms: float
    replayed: bool


class ReplayCache:
    """Stores and reuses model responses keyed by normalized request content."""

    def __init__(self, mode: ReplayMode | str = ReplayMode.AUTO, traces_root: Path | None = None) -> None:
        if isinstance(mode, str):
            mode = ReplayMode(mode.upper())
        self.mode = mode
        self.traces_root = traces_root or (Path(__file__).resolve().parents[2] / "analysis" / "traces")

    @staticmethod
    def make_key(
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        seed: int | None,
    ) -> str:
        """Build the deterministic SHA256 key for a model invocation."""
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools or [],
            "temperature": temperature,
            "seed": seed,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _sanitize_model_name(model: str) -> str:
        sanitized = model.replace("/", "__")
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", sanitized)
        return sanitized or "unknown_model"

    def _entry_path(self, *, model: str, key: str) -> Path:
        return self.traces_root / self._sanitize_model_name(model) / f"{key}.json"

    def _read_entry(self, *, model: str, key: str) -> ReplayResult:
        t0 = time.perf_counter_ns()
        path = self._entry_path(model=model, key=key)
        if not path.exists():
            raise ReplayCacheMissError(f"Replay key missing: {key} ({path})")
        entry = json.loads(path.read_text(encoding="utf-8"))
        replay_latency_ms = (time.perf_counter_ns() - t0) / 1e6
        return ReplayResult(
            key=key,
            response_json=entry["response"],
            token_counts=entry.get("token_counts", {}),
            recorded_latency_ms=float(entry.get("recorded_latency_ms", 0.0)),
            replay_latency_ms=replay_latency_ms,
            replayed=True,
        )

    def _write_entry(
        self,
        *,
        model: str,
        key: str,
        request: dict[str, Any],
        response_json: dict[str, Any],
        token_counts: dict[str, int],
        recorded_latency_ms: float,
    ) -> None:
        path = self._entry_path(model=model, key=key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key,
            "mode": "recorded",
            "request": request,
            "response": response_json,
            "token_counts": token_counts,
            "recorded_latency_ms": float(recorded_latency_ms),
        }
        path.write_text(json.dumps(entry, indent=2), encoding="utf-8")

    def resolve(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        seed: int | None,
    ) -> tuple[str, Path, dict[str, Any]]:
        """Resolve key, file path, and canonical request payload."""
        request = {
            "model": model,
            "messages": messages,
            "tools": tools or [],
            "temperature": temperature,
            "seed": seed,
        }
        key = self.make_key(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            seed=seed,
        )
        return key, self._entry_path(model=model, key=key), request

    def execute(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        seed: int | None,
        api_call: Callable[[], tuple[dict[str, Any], dict[str, int], float]],
    ) -> ReplayResult:
        """Run a model call according to replay mode and return normalized metadata."""
        key, path, request = self.resolve(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            seed=seed,
        )

        if self.mode is ReplayMode.REPLAY:
            return self._read_entry(model=model, key=key)

        if self.mode is ReplayMode.AUTO and path.exists():
            return self._read_entry(model=model, key=key)

        response_json, token_counts, recorded_latency_ms = api_call()
        self._write_entry(
            model=model,
            key=key,
            request=request,
            response_json=response_json,
            token_counts=token_counts,
            recorded_latency_ms=recorded_latency_ms,
        )
        return ReplayResult(
            key=key,
            response_json=response_json,
            token_counts=token_counts,
            recorded_latency_ms=recorded_latency_ms,
            replay_latency_ms=recorded_latency_ms,
            replayed=False,
        )
