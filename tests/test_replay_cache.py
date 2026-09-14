"""Tests for ReplayCache and backend-level model call wrapping."""

from __future__ import annotations

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


OLLAMA_REQUEST = {
    "model": "qwen3:4b-instruct",
    "messages": [{"role": "user", "content": "hello"}],
    "tools": [{"type": "function", "function": {"name": "search"}}],
    "temperature": 0.0,
    "seed": 42,
}


def _call_ollama(backend: FakeBackend):
    return backend.model_call(
        model=OLLAMA_REQUEST["model"],
        messages=OLLAMA_REQUEST["messages"],
        tools=OLLAMA_REQUEST["tools"],
        temperature=OLLAMA_REQUEST["temperature"],
        seed=OLLAMA_REQUEST["seed"],
    )


def test_ollama_record_and_replay(tmp_path):
    """Ollama model calls record and replay correctly via ReplayCache."""
    traces_root = tmp_path / "analysis" / "traces"
    cache = ReplayCache(mode=ReplayMode.AUTO, traces_root=traces_root)
    backend = FakeBackend(replay_cache=cache)

    first = _call_ollama(backend)
    second = _call_ollama(backend)

    assert backend.calls == 1, "second call should have replayed, not hit the model"
    assert first.replayed is False
    assert second.replayed is True
    assert second.recorded_latency_ms == first.recorded_latency_ms
    assert second.token_counts == first.token_counts


def test_ollama_cache_dir_no_collision_with_openai(tmp_path):
    """qwen3:4b-instruct and gpt-4o-mini must use separate cache directories.

    _sanitize_model_name("qwen3:4b-instruct") → "qwen3_4b-instruct" which is
    distinct from "gpt-4o-mini", so their trace files never collide.
    """
    traces_root = tmp_path / "analysis" / "traces"
    cache = ReplayCache(mode=ReplayMode.RECORD, traces_root=traces_root)

    # Build both cache keys
    openai_key = ReplayCache.make_key(**REQUEST)
    ollama_key  = ReplayCache.make_key(**OLLAMA_REQUEST)

    # Keys differ (different model names) — but let's also check directory paths
    assert openai_key != ollama_key, "different model+prompt → different cache keys"

    # Simulate recording for both — check that the directories are distinct.
    backend_openai = FakeBackend(replay_cache=ReplayCache(mode=ReplayMode.RECORD, traces_root=traces_root))
    backend_ollama = FakeBackend(replay_cache=ReplayCache(mode=ReplayMode.RECORD, traces_root=traces_root))

    _call(backend_openai)
    _call_ollama(backend_ollama)

    # Each model should have its own subdirectory under traces_root
    dirs = [p for p in traces_root.iterdir() if p.is_dir()]
    assert len(dirs) == 2, f"expected 2 model dirs under traces_root, got {[d.name for d in dirs]}"
    dir_names = {d.name for d in dirs}
    assert "gpt-4o-mini" in dir_names, f"gpt-4o-mini dir not found; dirs={dir_names}"
    # Ollama model name after sanitization: "qwen3:4b-instruct" → "qwen3_4b-instruct"
    ollama_dir = next((n for n in dir_names if "qwen3" in n), None)
    assert ollama_dir is not None, f"no qwen3 dir found; dirs={dir_names}"
    assert ":" not in ollama_dir, f"colon in dir name {ollama_dir!r} — sanitization failed"
