"""Tests for cascade and budget-aware cascade routing policies."""

from __future__ import annotations

from pathlib import Path

from harness.backends.base import Backend
from harness.replay import ReplayMode
from routing.policies.budget_aware_cascade import BudgetAwareCascadePolicy
from routing.policies.cascade import CascadePolicy


class DummyBackend(Backend):
    def __init__(self, *, name: str, is_cloud: bool, traces_root: Path):
        super().__init__(
            name=name,
            default_model="dummy",
            is_cloud=is_cloud,
            replay_mode=ReplayMode.AUTO,
            traces_root=traces_root / name,
        )

    def _provider_call(self, *, model, messages, tools, temperature, seed, **kwargs):
        return {
            "id": self.name,
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


def test_cascade_stays_local_on_high_confidence(tmp_path):
    cloud = DummyBackend(name="cloud", is_cloud=True, traces_root=tmp_path)
    local = DummyBackend(name="local", is_cloud=False, traces_root=tmp_path)
    policy = CascadePolicy(cloud_backend=cloud, local_backend=local, escalation_threshold=0.55)

    ctx = {
        "local_output_text": "This is a complete and confident answer with enough detail.",
        "local_output_len": 60,
        "malformed_tool_call": False,
        "retry_count": 0,
    }
    chosen = policy.route("SH-01", "search_hybrid", ctx)

    assert chosen is local
    assert ctx["theta"] == 0.55
    assert ctx["confidence"] >= ctx["theta"]
    assert ctx["escalate"] is False


def test_cascade_escalates_on_low_confidence(tmp_path):
    cloud = DummyBackend(name="cloud", is_cloud=True, traces_root=tmp_path)
    local = DummyBackend(name="local", is_cloud=False, traces_root=tmp_path)
    policy = CascadePolicy(cloud_backend=cloud, local_backend=local, escalation_threshold=0.60)

    ctx = {
        "local_output_text": "not sure",
        "local_output_len": 8,
        "malformed_tool_call": True,
        "retry_count": 2,
    }
    chosen = policy.route("SH-01", "search_hybrid", ctx)

    assert chosen is cloud
    assert ctx["confidence"] < ctx["theta"]
    assert ctx["escalate"] is True


def test_budget_aware_theta_increases_as_budget_depletes(tmp_path):
    cloud = DummyBackend(name="cloud", is_cloud=True, traces_root=tmp_path)
    local = DummyBackend(name="local", is_cloud=False, traces_root=tmp_path)
    policy = BudgetAwareCascadePolicy(
        cloud_backend=cloud,
        local_backend=local,
        theta_min=0.30,
        theta_max=0.80,
    )

    ctx_full = {
        "remaining_budget_fraction": 1.0,
        "local_output_text": "adequate answer content",
        "local_output_len": 40,
        "malformed_tool_call": False,
        "retry_count": 0,
    }
    policy.route("SH-01", "search_hybrid", ctx_full)

    ctx_empty = {
        "remaining_budget_fraction": 0.0,
        "local_output_text": "adequate answer content",
        "local_output_len": 40,
        "malformed_tool_call": False,
        "retry_count": 0,
    }
    policy.route("SH-01", "search_hybrid", ctx_empty)

    assert ctx_full["theta"] == 0.30
    assert ctx_empty["theta"] == 0.80
    assert ctx_empty["theta"] > ctx_full["theta"]
