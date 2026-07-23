"""Tests for ReplayCache and backend-level model call wrapping."""

import json

import pytest

from harness.adapters.base import BackendBase
from harness.replay import ReplayCache, ReplayCacheMissError, ReplayMode


class FakeBackend(BackendBase):
    """Simple backend that returns deterministic fake responses."""

    def __init__(self, replay_cache: ReplayCache):
        super().__init__(replay_cache=replay_cache)
        self.calls = 0

    def _call_model_api(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float | None,
        seed: int | None,
        **kwargs,
    ) -> dict:
        self.calls += 1
        return {
            "id": f"fake-{self.calls}",
            "model": model,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
        }


REQUEST = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "hello"}],
    "tools": [{"type": "function", "function": {"name": "search"}}],
    "temperature": 0.0,
    "seed": 42,
}


def _call(backend: FakeBackend):
    return backend.model_call(
        model=REQUEST["model"],
        messages=REQUEST["messages"],
        tools=REQUEST["tools"],
        temperature=REQUEST["temperature"],
        seed=REQUEST["seed"],
    )


def test_record_mode_writes_trace(tmp_path):
    traces_root = tmp_path / "analysis" / "traces"
    cache = ReplayCache(mode=ReplayMode.RECORD, traces_root=traces_root)
    backend = FakeBackend(replay_cache=cache)

    result = _call(backend)

    key = ReplayCache.make_key(**REQUEST)
    entry_path = traces_root / "gpt-4o-mini" / f"{key}.json"

    assert backend.calls == 1
    assert result.cache_key == key
    assert result.replayed is False
    assert result.token_counts["total_tokens"] == 14
    assert result.recorded_latency_ms >= 0.0
    assert entry_path.exists()

    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    assert entry["recorded_latency_ms"] >= 0.0
    assert entry["token_counts"]["prompt_tokens"] == 11
    assert entry["response"]["choices"][0]["message"]["content"] == "ok"


def test_replay_mode_raises_when_missing(tmp_path):
    traces_root = tmp_path / "analysis" / "traces"
    cache = ReplayCache(mode=ReplayMode.REPLAY, traces_root=traces_root)
    backend = FakeBackend(replay_cache=cache)

    with pytest.raises(ReplayCacheMissError):
        _call(backend)

    assert backend.calls == 0


def test_auto_mode_replays_existing_entry(tmp_path):
    traces_root = tmp_path / "analysis" / "traces"
    cache = ReplayCache(mode=ReplayMode.AUTO, traces_root=traces_root)
    backend = FakeBackend(replay_cache=cache)

    first = _call(backend)
    second = _call(backend)

    assert backend.calls == 1
    assert first.replayed is False
    assert second.replayed is True
    assert second.cache_key == first.cache_key
    assert second.token_counts == first.token_counts
    assert second.recorded_latency_ms == first.recorded_latency_ms
    assert second.replay_latency_ms >= 0.0


def test_cache_key_changes_with_seed_and_temperature():
    base = ReplayCache.make_key(**REQUEST)
    changed_seed = ReplayCache.make_key(
        model=REQUEST["model"],
        messages=REQUEST["messages"],
        tools=REQUEST["tools"],
        temperature=REQUEST["temperature"],
        seed=43,
    )
    changed_temp = ReplayCache.make_key(
        model=REQUEST["model"],
        messages=REQUEST["messages"],
        tools=REQUEST["tools"],
        temperature=0.2,
        seed=REQUEST["seed"],
    )

    assert base != changed_seed
    assert base != changed_temp
