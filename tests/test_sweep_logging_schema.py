"""Validation tests for sweep routing-decision logging schema."""

from __future__ import annotations

from pathlib import Path

from evaluation.sweep import _run_task_step
from harness.backends.base import Backend
from harness.replay import ReplayMode
from routing.budget import BudgetTracker
from routing.policies.all_local import AllLocalPolicy


class FakeBackend(Backend):
    def __init__(self, *, name: str, is_cloud: bool, traces_root: Path):
        super().__init__(
            name=name,
            default_model="fake-model",
            is_cloud=is_cloud,
            replay_mode=ReplayMode.AUTO,
            traces_root=traces_root / name,
        )

    def _provider_call(self, *, model, messages, tools, temperature, seed, **kwargs):
        return {
            "id": f"{self.name}-1",
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": self.name}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


class FakeEvaluator:
    def score_task(self, *, task_id, task_prompt, response_json):
        return {"score": 5.0, "method": "deterministic"}


def test_routing_decision_schema_fields_present(tmp_path):
    local = FakeBackend(name="local", is_cloud=False, traces_root=tmp_path)
    cloud = FakeBackend(name="cloud", is_cloud=True, traces_root=tmp_path)
    policy = AllLocalPolicy(local)
    budget = BudgetTracker(cloud_token_cap=0)

    row = _run_task_step(
        task_id="SH-01",
        task={"category": "search_hybrid", "prompt": "test prompt"},
        seed=0,
        policy_name="all_local",
        policy=policy,
        budget=budget,
        cloud_backend=cloud,
        local_backend=local,
        evaluator=FakeEvaluator(),
    )

    decision = row["routing_decisions"][0]
    required = {
        "step_index",
        "step_type",
        "category",
        "backend_chosen",
        "remaining_budget",
        "local_output_hash",
        "cloud_output_hash",
    }
    assert required.issubset(decision.keys())
