"""Tests for speculative routing policy behavior."""

from __future__ import annotations

from pathlib import Path

from harness.backends.base import Backend, ModelCallResult
from harness.replay import ReplayMode
from routing.policies.speculative import SpeculativePolicy


class DummyBackend(Backend):
    def __init__(self, *, name: str, is_cloud: bool, traces_root: Path):
        super().__init__(
            name=name,
            default_model="dummy-model",
            is_cloud=is_cloud,
            replay_mode=ReplayMode.AUTO,
            traces_root=traces_root / name,
        )

    def _provider_call(self, *, model, messages, tools, temperature, seed, **kwargs):
        return {
            "id": self.name,
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }


def _result(backend: Backend, text: str, tool_calls=None):
    msg = {"role": "assistant", "content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return ModelCallResult(
        backend=backend.name,
        model=backend.default_model,
        response_json={
            "choices": [{"message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
        token_counts={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        recorded_latency_ms=1.0,
        replay_latency_ms=1.0,
        replayed=False,
        cache_key="k",
        per_turn_categories={},
    )


def test_speculative_agreement_memory_disables_future_cloud_charge(tmp_path):
    cloud = DummyBackend(name="cloud", is_cloud=True, traces_root=tmp_path)
    local = DummyBackend(name="local", is_cloud=False, traces_root=tmp_path)
    policy = SpeculativePolicy(cloud_backend=cloud, local_backend=local, text_similarity_threshold=0.8)

    local_result = _result(local, "Answer with useful detail")
    cloud_result = _result(cloud, "Answer with useful detail")

    first = policy.decide(
        task_id="SH-01",
        category="search_hybrid",
        prompt="p",
        local_result=local_result,
        cloud_result=cloud_result,
    )
    second = policy.decide(
        task_id="SH-01",
        category="search_hybrid",
        prompt="p",
        local_result=local_result,
        cloud_result=cloud_result,
    )

    assert first["agreed"] is True
    assert first["charge_cloud_tokens"] is True
    assert second["agreed"] is True
    assert second["charge_cloud_tokens"] is False


def test_speculative_disagreement_rolls_back_to_cloud(tmp_path):
    cloud = DummyBackend(name="cloud", is_cloud=True, traces_root=tmp_path)
    local = DummyBackend(name="local", is_cloud=False, traces_root=tmp_path)
    policy = SpeculativePolicy(cloud_backend=cloud, local_backend=local, text_similarity_threshold=0.95)

    local_result = _result(local, "short uncertain output")
    cloud_result = _result(cloud, "completely different detailed answer")

    decision = policy.decide(
        task_id="SH-01",
        category="search_hybrid",
        prompt="p",
        local_result=local_result,
        cloud_result=cloud_result,
    )

    assert decision["agreed"] is False
    assert decision["rollback"] is True
    assert decision["chosen_backend"] is cloud
    assert decision["charge_cloud_tokens"] is True
