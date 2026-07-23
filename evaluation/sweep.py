"""Budget-constrained policy sweep across the 14-task suite."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.quality import QualityEvaluator
from harness.adapters.sdk_direct import TASKS
from harness.backends.cloud_openai import CloudOpenAIBackend
from harness.backends.local_ollama import LocalOllamaBackend
from routing.budget import BudgetTracker
from routing.policies.all_cloud import AllCloudPolicy
from routing.policies.all_local import AllLocalPolicy
from routing.policies.budget_aware_cascade import BudgetAwareCascadePolicy
from routing.policies.cascade import CascadePolicy
from routing.policies.learned_router import LearnedRouterPolicy
from routing.policies.speculative import SpeculativePolicy
from routing.policies.static_category import StaticCategoryPolicy


BUDGET_LEVELS = [0.0, 0.10, 0.25, 0.50, 1.00]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
OUTPUT_PATH = Path("results") / "pareto_results.json"


@dataclass
class SweepConfig:
    seeds: list[int]
    budget_levels: list[float]
    tasks: list[str]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _output_hash(response_json: dict[str, Any]) -> str:
    content = ""
    choices = response_json.get("choices", [])
    if choices:
        content = str(choices[0].get("message", {}).get("content", ""))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _build_policies(cloud_backend: CloudOpenAIBackend, local_backend: LocalOllamaBackend):
    static_mapping_path = Path("routing") / "policies" / "static_category.yaml"
    policies = {
        "all_cloud": AllCloudPolicy(cloud_backend),
        "all_local": AllLocalPolicy(local_backend),
        "static_category": StaticCategoryPolicy(
            cloud_backend=cloud_backend,
            local_backend=local_backend,
            mapping_path=static_mapping_path,
            default_backend="cloud",
        ),
        "cascade": CascadePolicy(
            cloud_backend=cloud_backend,
            local_backend=local_backend,
            escalation_threshold=float(os.environ.get("APU_CASCADE_THETA", "0.55")),
        ),
        "budget_aware_cascade": BudgetAwareCascadePolicy(
            cloud_backend=cloud_backend,
            local_backend=local_backend,
            theta_min=float(os.environ.get("APU_BUDGET_CASCADE_THETA_MIN", "0.35")),
            theta_max=float(os.environ.get("APU_BUDGET_CASCADE_THETA_MAX", "0.80")),
        ),
        "speculative": SpeculativePolicy(
            cloud_backend=cloud_backend,
            local_backend=local_backend,
            text_similarity_threshold=float(os.environ.get("APU_SPEC_SIM_THRESHOLD", "0.88")),
        ),
    }

    learned_model_path = os.environ.get("APU_LEARNED_ROUTER_MODEL")
    if learned_model_path and Path(learned_model_path).exists():
        policies["learned_router"] = LearnedRouterPolicy(
            cloud_backend=cloud_backend,
            local_backend=local_backend,
            model_path=learned_model_path,
            threshold=float(os.environ.get("APU_LEARNED_ROUTER_THRESHOLD", "0.5")),
        )

    return policies


def _extract_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _has_malformed_tool_call(response_json: dict[str, Any]) -> bool:
    choices = response_json.get("choices", [])
    if not choices:
        return False
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return False
    if not isinstance(tool_calls, list):
        return True
    for call in tool_calls:
        if not isinstance(call, dict):
            return True
        function = call.get("function")
        if not isinstance(function, dict):
            return True
        if "name" not in function or "arguments" not in function:
            return True
        raw_args = function.get("arguments")
        if isinstance(raw_args, str):
            try:
                json.loads(raw_args)
            except Exception:
                return True
    return False


def _trajectory_bucket(task_index: int, task_count: int) -> str:
    if task_count <= 0:
        return "early"
    third = max(1, task_count // 3)
    if task_index < third:
        return "early"
    if task_index < min(task_count, third * 2):
        return "mid"
    return "late"


def _compute_speculative_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spec_rows = [r for r in rows if r.get("policy") == "speculative"]
    if not spec_rows:
        return {}

    by_category: dict[str, dict[str, int]] = {}
    on_latency: list[float] = []
    off_latency: list[float] = []
    rollback_latency_total_ms = 0.0
    rollback_cloud_tokens = 0

    for row in spec_rows:
        category = row.get("category")
        if category not in by_category:
            by_category[category] = {"total": 0, "agreed": 0}
        by_category[category]["total"] += 1
        if row.get("speculative_agreed"):
            by_category[category]["agreed"] += 1

        on_latency.append(float(row.get("spec_on_latency_ms", 0.0)))
        off_latency.append(float(row.get("spec_off_latency_ms", 0.0)))
        rollback_latency_total_ms += float(row.get("spec_rollback_latency_ms", 0.0))
        rollback_cloud_tokens += int(row.get("rollback_cloud_tokens", 0))

    agreement_rate_per_category = {
        cat: (vals["agreed"] / vals["total"] if vals["total"] else 0.0)
        for cat, vals in by_category.items()
    }

    return {
        "agreement_rate_per_category": agreement_rate_per_category,
        "rollback_cost": {
            "cloud_tokens": rollback_cloud_tokens,
            "latency_ms": rollback_latency_total_ms,
        },
        "latency_speculation_on_ms": {
            "p50": _percentile(on_latency, 0.50),
            "p95": _percentile(on_latency, 0.95),
            "p99": _percentile(on_latency, 0.99),
        },
        "latency_speculation_off_ms": {
            "p50": _percentile(off_latency, 0.50),
            "p95": _percentile(off_latency, 0.95),
            "p99": _percentile(off_latency, 0.99),
        },
    }


def _run_task_step(
    *,
    task_id: str,
    task: dict[str, Any],
    seed: int,
    policy_name: str,
    policy,
    budget: BudgetTracker,
    cloud_backend: CloudOpenAIBackend,
    local_backend: LocalOllamaBackend,
    evaluator: QualityEvaluator,
    task_index: int = 0,
    task_count: int = 1,
) -> dict[str, Any]:
    category = task["category"]
    prompt = task["prompt"]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    remaining_fraction = 1.0
    if budget.cloud_token_cap > 0:
        remaining_fraction = budget.remaining_cloud_tokens / budget.cloud_token_cap
    step_context: dict[str, Any] = {
        "policy": policy_name,
        "seed": seed,
        "step_index": 0,
        "remaining_budget_fraction": remaining_fraction,
        "retry_count": 0,
    }

    local_probe = None
    cloud_probe = None
    spec_result = None
    spec_rollback_latency_ms = 0.0
    spec_on_latency_ms = None
    spec_off_latency_ms = None
    if getattr(policy, "requires_local_attempt", False):
        local_probe = local_backend.model_call(
            messages=messages,
            tools=None,
            temperature=0.0,
            seed=seed,
        )
        local_text = _extract_text(local_probe.response_json)
        step_context["local_output_text"] = local_text
        step_context["local_output_len"] = len(local_text)
        step_context["malformed_tool_call"] = _has_malformed_tool_call(local_probe.response_json)

    if getattr(policy, "requires_speculative_dual", False):
        local_probe = local_backend.model_call(
            messages=messages,
            tools=None,
            temperature=0.0,
            seed=seed,
        )
        cloud_probe = cloud_backend.model_call(
            messages=messages,
            tools=None,
            temperature=0.0,
            seed=seed,
        )
        spec_result = policy.decide(
            task_id=task_id,
            category=category,
            prompt=prompt,
            local_result=local_probe,
            cloud_result=cloud_probe,
        )
        step_context["speculative"] = {
            "agreed": spec_result["agreed"],
            "rollback": spec_result["rollback"],
            "comparator": spec_result["comparator"],
            "text_similarity": spec_result.get("text_similarity"),
            "known_agreement": spec_result.get("known_agreement"),
        }
        spec_off_latency_ms = cloud_probe.recorded_latency_ms
        spec_on_latency_ms = max(local_probe.recorded_latency_ms, cloud_probe.recorded_latency_ms)
        if spec_result["rollback"]:
            t_rb = time.perf_counter_ns()
            _ = spec_result["cloud_result"].response_json
            spec_rollback_latency_ms = (time.perf_counter_ns() - t_rb) / 1e6
            spec_on_latency_ms += spec_rollback_latency_ms

    requested = policy.route(task_id, category, step_context)
    chosen = budget.enforce(
        task_id=task_id,
        category=category,
        requested_backend=requested,
        local_backend=local_backend,
        step_context=step_context,
    )

    if spec_result is not None:
        chosen = spec_result["chosen_backend"]
        chosen_result = spec_result["committed_result"]
        if spec_result["charge_cloud_tokens"]:
            budget.record_usage(cloud_backend, cloud_probe.token_counts)
    else:
        if chosen is local_backend and local_probe is not None:
            chosen_result = local_probe
        else:
            chosen_result = chosen.model_call(
                messages=messages,
                tools=None,
                temperature=0.0,
                seed=seed,
            )

    cloud_result = None
    local_result = None
    if local_probe is not None:
        local_result = local_probe
    if cloud_probe is not None:
        cloud_result = cloud_probe
    if chosen.is_cloud and cloud_result is None:
        cloud_result = chosen_result
    elif not chosen.is_cloud:
        local_result = chosen_result

    # Optional dual-run for training logs. Default off to avoid extra provider usage.
    run_dual = os.environ.get("APU_SWEEP_RUN_BOTH", "0") == "1"
    if run_dual:
        if cloud_result is None:
            cloud_result = cloud_backend.model_call(messages=messages, tools=None, temperature=0.0, seed=seed)
        if local_result is None:
            local_result = local_backend.model_call(messages=messages, tools=None, temperature=0.0, seed=seed)

    if spec_result is None:
        budget.record_usage(chosen, chosen_result.token_counts)

    quality_info = evaluator.score_task(task_id=task_id, task_prompt=prompt, response_json=chosen_result.response_json)

    latencies = [chosen_result.recorded_latency_ms]
    routing_decision = {
        "step_index": 0,
        "step_type": "llm_call",
        "category": category,
        "backend_chosen": chosen.name,
        "remaining_budget": budget.remaining_cloud_tokens,
        "theta": step_context.get("theta"),
        "confidence": step_context.get("confidence"),
        "escalate": step_context.get("escalate"),
        "spec_agreed": spec_result.get("agreed") if spec_result else None,
        "spec_comparator": spec_result.get("comparator") if spec_result else None,
        "spec_text_similarity": spec_result.get("text_similarity") if spec_result else None,
        "spec_known_agreement": spec_result.get("known_agreement") if spec_result else None,
        "spec_charge_cloud_tokens": spec_result.get("charge_cloud_tokens") if spec_result else None,
        "local_output_hash": _output_hash(local_result.response_json) if local_result else None,
        "cloud_output_hash": _output_hash(cloud_result.response_json) if cloud_result else None,
    }

    local_tokens = int(local_result.token_counts.get("total_tokens", 0)) if local_result else 0

    cloud_tokens = 0
    if spec_result is not None:
        if spec_result["charge_cloud_tokens"] and cloud_result is not None:
            cloud_tokens = int(cloud_result.token_counts.get("total_tokens", 0))
    elif chosen.is_cloud:
        cloud_tokens = int(chosen_result.token_counts.get("total_tokens", 0))

    category_metrics = {}
    if spec_result and spec_result["rollback"]:
        category_metrics["SPEC_ROLLBACK"] = {
            "cpu_ns": int(spec_rollback_latency_ms * 1e6),
            "wall_ns": int(spec_rollback_latency_ms * 1e6),
            "bytes_in": 0,
            "bytes_out": 0,
            "count": 1,
        }

    return {
        "task_id": task_id,
        "seed": seed,
        "quality": quality_info["score"],
        "quality_method": quality_info["method"],
        "category": category,
        "cloud_tokens_spent": cloud_tokens,
        "local_tokens": local_tokens,
        "p50_turn_latency_ms": _percentile(latencies, 0.50),
        "p95_turn_latency_ms": _percentile(latencies, 0.95),
        "p99_turn_latency_ms": _percentile(latencies, 0.99),
        "trajectory_bucket": _trajectory_bucket(task_index, task_count),
        "speculative_agreed": spec_result.get("agreed") if spec_result else None,
        "rollback": spec_result.get("rollback") if spec_result else False,
        "rollback_cloud_tokens": cloud_tokens if (spec_result and spec_result.get("rollback")) else 0,
        "spec_rollback_latency_ms": spec_rollback_latency_ms,
        "spec_on_latency_ms": spec_on_latency_ms,
        "spec_off_latency_ms": spec_off_latency_ms,
        "category_metrics": category_metrics,
        "routing_decisions": [routing_decision],
    }


def _baseline_cloud_tokens_by_seed(
    *,
    config: SweepConfig,
    cloud_backend: CloudOpenAIBackend,
    local_backend: LocalOllamaBackend,
    evaluator: QualityEvaluator,
) -> dict[int, int]:
    policies = _build_policies(cloud_backend, local_backend)
    baseline_policy = policies["all_cloud"]
    out: dict[int, int] = {}

    for seed in config.seeds:
        budget = BudgetTracker(cloud_token_cap=10**12)
        total = 0
        for idx, task_id in enumerate(config.tasks):
            row = _run_task_step(
                task_id=task_id,
                task=TASKS[task_id],
                seed=seed,
                policy_name="all_cloud",
                policy=baseline_policy,
                budget=budget,
                cloud_backend=cloud_backend,
                local_backend=local_backend,
                evaluator=evaluator,
                task_index=idx,
                task_count=len(config.tasks),
            )
            total += int(row["cloud_tokens_spent"])
        out[seed] = total
    return out


def _trajectory_histogram(rows: list[dict[str, Any]], policy_names: list[str], budget_levels: list[float]) -> dict[str, Any]:
    hist = {
        policy: {
            str(level): {"early": 0, "mid": 0, "late": 0}
            for level in budget_levels
        }
        for policy in policy_names
    }
    for row in rows:
        policy = row["policy"]
        level = str(row["budget_level"])
        if policy not in hist or level not in hist[policy]:
            continue
        bucket = row.get("trajectory_bucket", "early")
        hist[policy][level][bucket] += int(row.get("cloud_tokens_spent", 0))
    return hist


def run_sweep(comparison_mode: bool = False) -> dict[str, Any]:
    seeds = DEFAULT_SEEDS
    task_ids = list(TASKS.keys())
    config = SweepConfig(seeds=seeds, budget_levels=BUDGET_LEVELS, tasks=task_ids)

    replay_mode = os.environ.get("APU_REPLAY_MODE", "AUTO")
    cloud_backend = CloudOpenAIBackend(replay_mode=replay_mode)
    local_backend = LocalOllamaBackend(replay_mode=replay_mode)
    evaluator = QualityEvaluator(replay_mode=replay_mode)

    policies = _build_policies(cloud_backend, local_backend)
    selected_policy_names = ["cascade", "budget_aware_cascade"] if comparison_mode else list(policies.keys())
    selected_policies = {name: policies[name] for name in selected_policy_names}
    baseline_by_seed = _baseline_cloud_tokens_by_seed(
        config=config,
        cloud_backend=cloud_backend,
        local_backend=local_backend,
        evaluator=evaluator,
    )

    records: list[dict[str, Any]] = []

    for policy_name, policy in selected_policies.items():
        for budget_level in config.budget_levels:
            for seed in config.seeds:
                cap = int(round(baseline_by_seed[seed] * budget_level))
                budget = BudgetTracker(cloud_token_cap=cap)

                for idx, task_id in enumerate(config.tasks):
                    row = _run_task_step(
                        task_id=task_id,
                        task=TASKS[task_id],
                        seed=seed,
                        policy_name=policy_name,
                        policy=policy,
                        budget=budget,
                        cloud_backend=cloud_backend,
                        local_backend=local_backend,
                        evaluator=evaluator,
                        task_index=idx,
                        task_count=len(config.tasks),
                    )
                    row["policy"] = policy_name
                    row["budget_level"] = budget_level
                    row["budget_cap_tokens"] = cap
                    records.append(row)

    comparison = None
    if comparison_mode:
        comparison = {
            "mode": "cascade_vs_budget_aware_cascade",
            "trajectory_spend_histogram": _trajectory_histogram(
                records,
                selected_policy_names,
                config.budget_levels,
            ),
        }

    speculative_report = _compute_speculative_report(records)

    artifact = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seeds": config.seeds,
            "budget_levels": config.budget_levels,
            "tasks": config.tasks,
            "baseline_policy": "all_cloud",
            "baseline_cloud_tokens_by_seed": baseline_by_seed,
            "judge_model": evaluator.judge_model,
            "judge_prompt_version": evaluator.judge_prompt_version,
            "comparison_mode": comparison_mode,
            "policies": selected_policy_names,
        },
        "results": records,
    }
    if comparison is not None:
        artifact["comparison"] = comparison
    if speculative_report:
        artifact["speculative_report"] = speculative_report

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    mode = os.environ.get("APU_SWEEP_MODE", "full").lower()
    comparison_mode = mode in {"cascade_compare", "comparison", "cascade"}
    artifact = run_sweep(comparison_mode=comparison_mode)
    rows = artifact["results"]
    quality_values = [r["quality"] for r in rows]
    print(f"Sweep rows: {len(rows)}")
    print(f"Mean quality: {statistics.mean(quality_values):.3f}" if quality_values else "Mean quality: n/a")
    print(f"Comparison mode: {comparison_mode}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
