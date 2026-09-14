"""C2c: inter-task teardown recovery, double after-canary, context_cap projection, memory series."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from seam import measurement as measurement_mod
from seam.agent.harness import run_task
from seam.agent.policy import DEADLINE_DISABLED_S, ThroughputModel
from seam.agent.tools import TaskSpec, load_workload
from seam.backends.base import GenerationRequest, GenerationResult, ToolCall
from seam.backends.refusing_cloud import RefusingCloudBackend
from seam.gitinfo import repo_root
from seam.measurement import machine_measurement
from seam.tools.efilter_run import (
    memory_series_summary,
    wait_free_memory_recovery,
)

_ROOT = repo_root(Path(__file__).parent)
_THROUGHPUT = ThroughputModel(
    target="cpu-p",
    r_prefill_tok_s=100.0,
    r_decode_tok_s=10.0,
    measured_by_run_id="test",
)
_N_OUT = {"tool_call_synthesis": 32, "answer_synthesis": 32}


# ==================================================================================================
# Teardown recovery wait
# ==================================================================================================


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.t += float(seconds)


def test_wait_free_memory_recovery_succeeds_when_threshold_met() -> None:
    clock = _FakeClock()
    readings = iter([100.0, 120.0, 200.0])

    result = wait_free_memory_recovery(
        threshold_mb=180.0,
        timeout_s=30.0,
        poll_interval_s=2.0,
        clock=clock,
        free_memory_fn=lambda: next(readings),
    )
    assert result["recovered"] is True
    assert result["free_memory_mb"] == pytest.approx(200.0)
    assert result["threshold_mb"] == pytest.approx(180.0)
    assert len(clock.sleeps) == 2


def test_wait_free_memory_recovery_times_out() -> None:
    clock = _FakeClock()

    result = wait_free_memory_recovery(
        threshold_mb=500.0,
        timeout_s=5.0,
        poll_interval_s=2.0,
        clock=clock,
        free_memory_fn=lambda: 100.0,
    )
    assert result["recovered"] is False
    assert result["free_memory_mb"] == pytest.approx(100.0)
    assert result["elapsed_s"] >= 5.0


# ==================================================================================================
# Double after-canary
# ==================================================================================================


def _minimal_measurement_config(*, settle_s: float | None = None) -> dict[str, Any]:
    canary: dict[str, Any] = {
        "iterations_per_cpu": 1,
        "seed": 1,
        "max_relative_drift": 0.15,
    }
    if settle_s is not None:
        canary["settle_s"] = settle_s
        canary["double_after"] = True
    return {
        "locking": {"wait_timeout_s": 1, "poll_interval_s": 0.01},
        "quiescence": {
            "window_s": 1,
            "sample_interval_s": 1,
            "total_cpu_max_pct": 20,
            "p_core_cpu_max_pct": 30,
            "available_memory_min_mb": 2048,
        },
        "canary": canary,
        "paging": {
            "sample_interval_s": 0.5,
            "available_memory_min_mb": 500,
            "sustained_nonzero_consecutive_samples": 2,
            "exclude_on_failure": True,
            "gate_mode": "absolute_zero",
            "hard_page_reads_threshold_per_s": 0.0,
        },
    }


def test_double_canary_settled_authoritative_transient_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pre = {"affinity_passed": True, "workers": [{"runtime_ns": 1000.0}]}
    immediate = {"affinity_passed": True, "workers": [{"runtime_ns": 1300.0}]}  # 0.30 drift
    settled = {"affinity_passed": True, "workers": [{"runtime_ns": 1050.0}]}  # 0.05 drift
    calls = {"n": 0}

    def _canary(**_kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            return pre
        if calls["n"] == 2:
            return immediate
        return settled

    monkeypatch.setattr(measurement_mod, "run_compute_canary", _canary)
    monkeypatch.setattr(
        measurement_mod,
        "measure_quiescence",
        lambda **_k: {"passed": True, "failures": []},
    )
    sleeps: list[float] = []
    monkeypatch.setattr(measurement_mod.time, "sleep", lambda s: sleeps.append(float(s)))

    # MemoryPressureSampler still runs - stub stop to a clean record.
    class _FakePressure:
        def start(self) -> None:
            return None

        def stop(self) -> dict[str, Any]:
            return {
                "available_memory_mb_before": 4000.0,
                "available_memory_mb_after": 4000.0,
                "hard_page_reads_per_s": [],
                "invalid_reasons": [],
                "cpu_pct_total": [],
                "cpu_pct_per_core": [],
            }

    monkeypatch.setattr(
        "seam.telemetry.memory.MemoryPressureSampler",
        lambda **_k: _FakePressure(),
    )

    with machine_measurement(
        repo_root=tmp_path,
        label="double-canary",
        config=_minimal_measurement_config(settle_s=7.0),
        p_cpus=[0],
        prevalidated_quiescence={"passed": True, "source": "test"},
    ) as block:
        pass

    assert block.record["valid"] is True
    assert block.record["canary_double_after"] is True
    assert block.record["canary_drift_authoritative"] == "settled"
    assert block.record["canary_relative_drift_immediate"] == pytest.approx(0.30)
    assert block.record["canary_relative_drift"] == pytest.approx(0.05)
    assert block.record["canary_immediate_drifted"] is True
    assert block.record["canary_settled_drifted"] is False
    assert "canary_transient_drift_evidence" in block.record
    assert 7.0 in sleeps
    assert block.record["canary_drift_threshold"] == pytest.approx(0.15)


def test_double_canary_both_drifted_invalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pre = {"affinity_passed": True, "workers": [{"runtime_ns": 1000.0}]}
    drifted = {"affinity_passed": True, "workers": [{"runtime_ns": 1400.0}]}
    seq = iter([pre, drifted, drifted])
    monkeypatch.setattr(measurement_mod, "run_compute_canary", lambda **_k: next(seq))
    monkeypatch.setattr(
        measurement_mod,
        "measure_quiescence",
        lambda **_k: {"passed": True, "failures": []},
    )
    monkeypatch.setattr(measurement_mod.time, "sleep", lambda _s: None)

    class _FakePressure:
        def start(self) -> None:
            return None

        def stop(self) -> dict[str, Any]:
            return {
                "available_memory_mb_before": 4000.0,
                "available_memory_mb_after": 4000.0,
                "hard_page_reads_per_s": [],
                "invalid_reasons": [],
                "cpu_pct_total": [],
                "cpu_pct_per_core": [],
            }

    monkeypatch.setattr(
        "seam.telemetry.memory.MemoryPressureSampler",
        lambda **_k: _FakePressure(),
    )

    with machine_measurement(
        repo_root=tmp_path,
        label="persistent-drift",
        config=_minimal_measurement_config(settle_s=5.0),
        p_cpus=[0],
        prevalidated_quiescence={"passed": True, "source": "test"},
    ) as block:
        pass

    assert block.record["valid"] is False
    assert any("persistent" in r for r in block.record["invalid_reasons"])


def test_settle_s_zero_keeps_single_after_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pre = {"affinity_passed": True, "workers": [{"runtime_ns": 1000.0}]}
    post = {"affinity_passed": True, "workers": [{"runtime_ns": 1010.0}]}
    seq = iter([pre, post])
    monkeypatch.setattr(measurement_mod, "run_compute_canary", lambda **_k: next(seq))
    monkeypatch.setattr(
        measurement_mod,
        "measure_quiescence",
        lambda **_k: {"passed": True, "failures": []},
    )
    sleeps: list[float] = []
    monkeypatch.setattr(measurement_mod.time, "sleep", lambda s: sleeps.append(float(s)))

    class _FakePressure:
        def start(self) -> None:
            return None

        def stop(self) -> dict[str, Any]:
            return {
                "available_memory_mb_before": 4000.0,
                "available_memory_mb_after": 4000.0,
                "hard_page_reads_per_s": [],
                "invalid_reasons": [],
                "cpu_pct_total": [],
                "cpu_pct_per_core": [],
            }

    monkeypatch.setattr(
        "seam.telemetry.memory.MemoryPressureSampler",
        lambda **_k: _FakePressure(),
    )

    with machine_measurement(
        repo_root=tmp_path,
        label="single-canary",
        config=_minimal_measurement_config(settle_s=None),
        p_cpus=[0],
        prevalidated_quiescence={"passed": True, "source": "test"},
    ) as block:
        pass

    assert block.record["canary_double_after"] is False
    assert block.record["canary_drift_authoritative"] == "immediate"
    assert "canary_post_immediate" not in block.record
    assert sleeps == []


# ==================================================================================================
# C2T20 context_cap projection
# ==================================================================================================


class _ProjectionStub:
    name = "stub_local"
    model_ref = "stub"

    def __init__(self, projections: list[int]) -> None:
        self.projections = list(projections)
        self.calls = 0
        self.logged_projections: list[int] = []

    def preflight(self) -> Any:
        return type("V", (), {"status": "OK", "reason": ""})()

    def estimate_context_tokens(self, request: GenerationRequest) -> int:
        # Match C2T20: under cap for first two generates, then jump past the context cap.
        idx = min(self.calls, len(self.projections) - 1)
        projected = int(self.projections[idx])
        self.logged_projections.append(projected)
        return projected

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        if self.calls == 1:
            text = (
                '<tool_call>{"name": "list_files", '
                '"arguments": {"directory": "/docs"}}</tool_call>'
            )
            calls: tuple[ToolCall, ...] = (
                ToolCall(name="list_files", arguments={"directory": "/docs"}),
            )
            prompt_tokens = 829
        else:
            text = (
                '<tool_call>{"name": "retrieve_documents", '
                '"arguments": {"query": "latency envelope"}}</tool_call>'
            )
            calls = (ToolCall(name="retrieve_documents", arguments={"query": "latency envelope"}),)
            prompt_tokens = 1203
        return GenerationResult(
            text=text,
            tool_calls=calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=20,
            completion_chars=len(text),
            completion_bytes=len(text.encode()),
            wall_ns=1_000_000,
            ttft_ns=500_000,
            backend=self.name,
            model_ref=self.model_ref,
            extra={"cache_instrumented": False},
        )


def test_context_cap_logs_projected_next_not_peak_completed() -> None:
    """C2T20 shape: peak completed 1203, but projected next ~7461 trips the 5000 cap."""
    stub = _ProjectionStub([829, 1203, 7461])
    # World must accept retrieve_documents - use the C2 workload world.
    workload = load_workload(_ROOT / "configs" / "tasks" / "bfcl_slice_c2.json")
    task = next(t for t in workload.tasks if t.task_id == "C2T20")
    # Use a tiny prompt task with the same call pattern via stub projections.
    result = run_task(
        task=TaskSpec(
            task_id="C2T20",
            prompt=task.prompt,
            expected=task.expected,
            min_tool_calls=1,
        ),
        world=workload.world,
        local_backend=stub,
        cloud_backend=RefusingCloudBackend(run_id="test"),
        throughput=_THROUGHPUT,
        deadline_s=DEADLINE_DISABLED_S,
        n_out_pred_tokens=_N_OUT,
        max_steps=8,
        max_tokens=128,
        run_id="test",
        program_id="test/C2T20",
        context_cap_tokens=5000,
        constrain_tool_calls=True,
        prompt_token_scaffold_tokens=621,
    )
    assert result.terminated_reason == "context_cap"
    assert result.realized_steps == 2
    assert max(s.context_tokens_total for s in result.steps) == 1203
    assert result.projected_next_context_tokens == 7461
    assert result.projected_next_context_tokens > 5000


def test_c2t20_pilot_payload_projects_above_cap() -> None:
    """Reproduce 1f92fc4a C2T20: retrieve 'latency envelope' pushes HF projection above cap."""
    from transformers import AutoTokenizer

    from seam.agent.tools import SYSTEM_PROMPT, TOOL_SPECS
    from seam.backends.local_openvino import _tool_to_openai_schema

    workload = load_workload(_ROOT / "configs" / "tasks" / "bfcl_slice_c2.json")
    task = next(t for t in workload.tasks if t.task_id == "C2T20")
    messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
    text0, _ = workload.world.execute("list_files", {"directory": "/corpus"})
    messages.append(
        {
            "role": "assistant",
            "content": (
                '<tool_call>{"name": "list_files", '
                '"arguments": {"directory": "/corpus"}}</tool_call>'
            ),
        }
    )
    messages.append({"role": "user", "content": f"Tool list_files returned: {text0}"})
    text1, _ = workload.world.execute("retrieve_documents", {"query": "latency envelope"})
    assert len(text1) == 20273  # byte-identical to sealed 1f92fc4a tool result_bytes
    messages.append(
        {
            "role": "assistant",
            "content": (
                '<tool_call>{"name": "retrieve_documents", '
                '"arguments": {"query": "latency envelope"}}</tool_call>'
            ),
        }
    )
    messages.append({"role": "user", "content": f"Tool retrieve_documents returned: {text1}"})

    spec = yaml.safe_load(
        (_ROOT / "configs" / "models" / "Qwen3-4B-int4-ov.yaml").read_text(encoding="utf-8")
    )
    tok = AutoTokenizer.from_pretrained(spec["ir_dir"])
    kwargs: dict[str, Any] = {}
    if isinstance(tok.chat_template, str) and "enable_thinking" in tok.chat_template:
        kwargs["enable_thinking"] = False
    rendered = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        tools=[_tool_to_openai_schema(t) for t in TOOL_SPECS],
        add_generation_prompt=True,
        tokenize=False,
        **kwargs,
    )
    projected = len(tok(rendered)["input_ids"])
    assert projected > 5000
    # Historical uncapped projection under the 7000-era cap; still trips C2g's 5000.
    assert projected == 7461


# ==================================================================================================
# Memory series fields
# ==================================================================================================


def test_memory_series_summary_flags_monotonic_decline() -> None:
    per_task = [
        {
            "free_memory_mb_before_task": 8000.0,
            "free_memory_mb_after_task": 7000.0,
            "free_memory_mb_after_teardown": 7800.0,
            "process_rss_before": 1_000_000_000,
            "process_rss_after": 1_200_000_000,
        },
        {
            "free_memory_mb_before_task": 7500.0,
            "free_memory_mb_after_task": 6500.0,
            "free_memory_mb_after_teardown": 7400.0,
            "process_rss_before": 1_200_000_000,
            "process_rss_after": 1_400_000_000,
        },
        {
            "free_memory_mb_before_task": 7000.0,
            "free_memory_mb_after_task": 6000.0,
            "free_memory_mb_after_teardown": None,
            "process_rss_before": 1_400_000_000,
            "process_rss_after": 1_600_000_000,
        },
    ]
    summary = memory_series_summary(per_task)
    assert summary["monotonic_decline_free_memory_mb_before_task"] is True
    assert summary["monotonic_decline_free_memory_mb_after_task"] is True
    assert summary["monotonic_increase_process_rss_after"] is True
    assert len(summary["free_memory_mb_before_task"]) == 3


def test_efilter_yaml_c2c_pins() -> None:
    cfg = yaml.safe_load((_ROOT / "configs" / "efilter.yaml").read_text(encoding="utf-8"))
    assert cfg["workload"]["context_cap_tokens"] == 5000
    assert cfg["workload"]["max_tokens"] == 128
    assert cfg["workload"]["pilot_min_median_context_ratio"] == 3.0
    assert cfg["canary"]["settle_s"] == 10
    assert cfg["canary"]["double_after"] is True
    assert cfg["teardown"]["free_memory_recovery_fraction"] == pytest.approx(0.90)
    assert cfg["workload"]["require_pilot_context_clearance"] is True
    assert cfg["policy"]["prompt_token_scaffold_tokens"] == 621
    assert cfg["c2c_amendment"] == "docs/CURSOR_PROMPT_C2c.md"


def test_backend_close_reload_methods_exist() -> None:
    from seam.backends.local_openvino import LocalOpenVinoBackend

    assert callable(LocalOpenVinoBackend.close)
    assert callable(LocalOpenVinoBackend.reload)
