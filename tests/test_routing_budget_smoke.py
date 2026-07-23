"""Smoke tests for routing policies and budget fallback behavior."""

from __future__ import annotations

from pathlib import Path

from harness.backends.base import Backend
from harness.replay import ReplayMode
from routing.budget import BudgetTracker
from routing.policies.all_cloud import AllCloudPolicy
from routing.policies.all_local import AllLocalPolicy
from routing.policies.static_category import StaticCategoryPolicy


TASKS = {
    "SH-01": {
        "category": "search_hybrid",
        "prompt": (
            "Search for AI research breakthroughs from the past year. "
            "Then search for practical applications and summarize two findings."
        ),
    }
}


class FakeCloudBackend(Backend):
    def __init__(self, traces_root: Path):
        super().__init__(
            name="fake_cloud",
            default_model="gpt-4o-mini",
            is_cloud=True,
            replay_mode=ReplayMode.AUTO,
            traces_root=traces_root / "cloud_traces",
        )
        self.calls = 0

    def _provider_call(self, *, model, messages, tools, temperature, seed, **kwargs):
        self.calls += 1
        return {
            "id": f"cloud-{self.calls}",
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "cloud"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }


class FakeLocalBackend(Backend):
    def __init__(self, traces_root: Path):
        super().__init__(
            name="fake_local",
            default_model="qwen2.5:1.5b",
            is_cloud=False,
            replay_mode=ReplayMode.AUTO,
            traces_root=traces_root / "local_traces",
        )
        self.calls = 0

    def _provider_call(self, *, model, messages, tools, temperature, seed, **kwargs):
        self.calls += 1
        return {
            "id": f"local-{self.calls}",
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "local"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


def _run_single(policy, budget: BudgetTracker, cloud_backend: Backend, local_backend: Backend):
    task_id = "SH-01"
    category = TASKS[task_id]["category"]
    prompt = TASKS[task_id]["prompt"]

    requested = policy.route(task_id, category, {"step": 0})
    selected = budget.enforce(
        task_id=task_id,
        category=category,
        requested_backend=requested,
        local_backend=local_backend,
        step_context={"policy": type(policy).__name__, "task_id": task_id},
    )

    result = selected.model_call(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        temperature=0.0,
        seed=123,
    )
    budget.record_usage(selected, result.token_counts)
    return selected.name, result


def test_smoke_sh01_each_policy(tmp_path):
    cloud = FakeCloudBackend(tmp_path)
    local = FakeLocalBackend(tmp_path)
    static_map = tmp_path / "static_category.yaml"
    static_map.write_text("category_to_backend:\n  search_hybrid: cloud\n", encoding="utf-8")

    all_cloud = AllCloudPolicy(cloud)
    all_local = AllLocalPolicy(local)
    static = StaticCategoryPolicy(
        cloud_backend=cloud,
        local_backend=local,
        mapping_path=static_map,
        default_backend="local",
    )

    budget = BudgetTracker(cloud_token_cap=100)

    selected_cloud, cloud_result = _run_single(all_cloud, budget, cloud, local)
    selected_local, local_result = _run_single(all_local, budget, cloud, local)
    selected_static, static_result = _run_single(static, budget, cloud, local)

    assert selected_cloud == "fake_cloud"
    assert selected_local == "fake_local"
    assert selected_static == "fake_cloud"

    assert cloud_result.per_turn_categories["HTTP_CLIENT"]["count"] == 1
    assert local_result.per_turn_categories["HTTP_CLIENT"]["count"] == 1
    assert static_result.per_turn_categories["HTTP_CLIENT"]["count"] == 1

    artifact = budget.to_artifact()
    assert artifact["cloud_total_tokens"] == 14
    assert len(artifact["routing_decisions"]) == 3


def test_budget_forces_local_fallback_when_exhausted(tmp_path):
    cloud = FakeCloudBackend(tmp_path)
    local = FakeLocalBackend(tmp_path)
    budget = BudgetTracker(cloud_token_cap=0)
    policy = AllCloudPolicy(cloud)

    selected_name, result = _run_single(policy, budget, cloud, local)

    assert selected_name == "fake_local"
    assert result.token_counts["total_tokens"] == 0
    assert cloud.calls == 0
    assert local.calls == 1

    decision = budget.to_artifact()["routing_decisions"][0]
    assert decision["forced_local"] is True
    assert decision["selected_backend"] == "fake_local"
    assert decision["reason"] == "cloud_budget_exhausted"
