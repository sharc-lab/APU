"""C2b: context-ratio gate, context cap, max_tokens, constrained tool path, 20k gate removed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from seam.agent.harness import run_task
from seam.agent.policy import DEADLINE_DISABLED_S, ThroughputModel
from seam.agent.tools import TOOL_SPECS, TaskSpec, ToolWorld
from seam.analysis.efilter import per_task_context_ceiling
from seam.backends.base import GenerationRequest, GenerationResult, ToolCall
from seam.backends.local_openvino import (
    CONSTRAINED_DECODING_MECHANISM,
    build_tool_call_structured_output_config,
)
from seam.backends.refusing_cloud import RefusingCloudBackend
from seam.errors import ConfigError
from seam.gitinfo import repo_root
from seam.tools.efilter_run import (
    evaluate_pilot_context_gate,
    median_context_ratio,
    per_task_context_ratio,
)

_ROOT = repo_root(Path(__file__).parent)
_THROUGHPUT = ThroughputModel(
    target="cpu-p",
    r_prefill_tok_s=100.0,
    r_decode_tok_s=10.0,
    measured_by_run_id="test",
)
_N_OUT = {"tool_call_synthesis": 32, "answer_synthesis": 32}
_WORLD = ToolWorld({"directories": {"/docs": ["a.txt", "b.txt", "c.txt"]}})
_TASK = TaskSpec(
    task_id="T01", prompt="How many files are in /docs?", expected="3", min_tool_calls=1
)


class _GrowingStub:
    name = "stub_local"
    model_ref = "stub"

    def __init__(self, *, base_prompt: int = 100, growth: int = 500) -> None:
        self.base_prompt = base_prompt
        self.growth = growth
        self.calls = 0
        self.last_request: GenerationRequest | None = None

    def preflight(self) -> Any:
        return type("V", (), {"status": "OK", "reason": ""})()

    def estimate_context_tokens(self, request: GenerationRequest) -> int:
        # Projection grows with transcript length so the cap can trip before generate.
        chars = sum(len(str(m.get("content", ""))) for m in request.messages)
        return self.base_prompt + chars // 4

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.last_request = request
        self.calls += 1
        prompt_tokens = self.base_prompt + self.growth * (self.calls - 1)
        if self.calls == 1:
            text = (
                '<tool_call>{"name": "list_files", '
                '"arguments": {"directory": "/docs"}}</tool_call>'
            )
            calls: tuple[ToolCall, ...] = (
                ToolCall(name="list_files", arguments={"directory": "/docs"}),
            )
        else:
            text = (
                '<tool_call>{"name": "submit_answer", ' '"arguments": {"answer": "3"}}</tool_call>'
            )
            calls = (ToolCall(name="submit_answer", arguments={"answer": "3"}),)
        return GenerationResult(
            text=text,
            tool_calls=calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=24,
            completion_chars=len(text),
            completion_bytes=len(text.encode()),
            wall_ns=1_000_000,
            ttft_ns=500_000,
            backend=self.name,
            model_ref=self.model_ref,
            extra={
                "cache_instrumented": False,
                "constrained_decoding": bool(request.expect_tool_call),
                "constrained_decoding_mechanism": (
                    CONSTRAINED_DECODING_MECHANISM if request.expect_tool_call else None
                ),
            },
        )


def test_per_task_context_ratio_uses_min_max_not_order() -> None:
    # Non-monotonic series still uses min/max (documented C2b definition).
    assert per_task_context_ratio([100, 300, 200]) == pytest.approx(3.0)
    assert per_task_context_ratio([650, 9114]) == pytest.approx(9114 / 650)
    assert per_task_context_ratio([]) is None
    assert median_context_ratio([9.0, 7.9, 14.0]) == pytest.approx(9.0)


def test_pilot_ratio_gate_replaces_20k_peak() -> None:
    growth = {"max_context_tokens_observed": 5000, "per_task_ratios": [9.0, 7.9, 14.0, 4.0, 3.5]}
    per_task = [{"context_ratio_cmax_over_cmin": r} for r in growth["per_task_ratios"]]
    ok = evaluate_pilot_context_gate(
        growth, per_task=per_task, min_median_context_ratio=3.0, memory_or_canary_invalidations=0
    )
    assert ok["cleared"] is True
    assert ok["max_context_tokens_is_gate"] is False
    assert ok["gate"] == "median_context_ratio_cmax_over_cmin"
    assert ok["median_context_ratio"] == pytest.approx(7.9)

    fail_ratio = evaluate_pilot_context_gate(
        {"max_context_tokens_observed": 50_000, "per_task_ratios": [1.2, 1.3, 1.4]},
        min_median_context_ratio=3.0,
    )
    assert fail_ratio["cleared"] is False
    assert fail_ratio["stop_message"] and "STOP" in fail_ratio["stop_message"]
    assert "20000" not in (fail_ratio["stop_message"] or "")

    fail_inv = evaluate_pilot_context_gate(
        growth, per_task=per_task, memory_or_canary_invalidations=1
    )
    assert fail_inv["cleared"] is False
    assert "invalidation" in (fail_inv["stop_message"] or "").lower()

    # C2e: paging-only invalidations do not refuse clearance.
    paging_ok = evaluate_pilot_context_gate(
        growth,
        per_task=per_task,
        memory_or_canary_invalidations=0,
        paging_invalidations=2,
    )
    assert paging_ok["cleared"] is True
    assert paging_ok["paging_invalidations_run_fatal"] is False

    with pytest.raises(ConfigError, match="min_peak_context_tokens"):
        evaluate_pilot_context_gate(growth, min_peak_context_tokens=20_000)


def test_context_cap_terminates_without_generate() -> None:
    backend = _GrowingStub(base_prompt=100, growth=100)
    # Force projection above cap on the first step via a huge user message.
    task = TaskSpec(
        task_id="CAP",
        prompt="x" * 40_000,
        expected="3",
        min_tool_calls=0,
    )
    result = run_task(
        task=task,
        world=_WORLD,
        local_backend=backend,
        cloud_backend=RefusingCloudBackend(run_id="test"),
        throughput=_THROUGHPUT,
        deadline_s=DEADLINE_DISABLED_S,
        n_out_pred_tokens=_N_OUT,
        max_steps=4,
        max_tokens=128,
        run_id="test",
        program_id="test/CAP",
        context_cap_tokens=5000,
        constrain_tool_calls=True,
    )
    assert result.terminated_reason == "context_cap"
    assert result.realized_steps == 0
    assert backend.calls == 0


def test_context_cap_allows_steps_under_cap_then_stops() -> None:
    backend = _GrowingStub(base_prompt=100, growth=50)

    class _CapStub(_GrowingStub):
        def estimate_context_tokens(self, request: GenerationRequest) -> int:
            # After first generate appends tool output, projection jumps over the cap.
            if self.calls == 0:
                return 500
            return 8000

    stub = _CapStub()
    result = run_task(
        task=_TASK,
        world=_WORLD,
        local_backend=stub,
        cloud_backend=RefusingCloudBackend(run_id="test"),
        throughput=_THROUGHPUT,
        deadline_s=DEADLINE_DISABLED_S,
        n_out_pred_tokens=_N_OUT,
        max_steps=4,
        max_tokens=128,
        run_id="test",
        program_id="test/T01",
        context_cap_tokens=5000,
        constrain_tool_calls=True,
    )
    assert result.terminated_reason == "context_cap"
    assert result.realized_steps == 1
    assert stub.calls == 1
    assert stub.last_request is not None
    assert stub.last_request.expect_tool_call is True
    assert stub.last_request.max_tokens == 128


def test_efilter_yaml_c2b_pins() -> None:
    cfg = yaml.safe_load((_ROOT / "configs" / "efilter.yaml").read_text(encoding="utf-8"))
    assert cfg["workload"]["max_tokens"] == 128
    assert cfg["workload"]["context_cap_tokens"] == 5000
    assert cfg["workload"]["n_tasks_pilot"] == 5
    assert cfg["workload"]["pilot_min_median_context_ratio"] == 3.0
    assert "pilot_min_peak_context_tokens" not in cfg["workload"]
    assert cfg["constrained_decoding"]["enabled"] is True
    assert "structured_output" in cfg["constrained_decoding"]["mechanism"]
    n_out = cfg["policy"]["n_out_pred_tokens"]
    assert n_out["tool_call_synthesis"] == n_out["answer_synthesis"]
    pilot = cfg["policy"]["n_out_pred_tokens_pilot"]
    assert pilot["tool_call_synthesis"] == pilot["answer_synthesis"]
    assert cfg["policy"]["prompt_token_scaffold_tokens"] == 621
    assert cfg["replay"]["p95_step_target_s"] is None
    assert "withdrawn_AM-032" in cfg["replay"]["p95_step_target_status"]


def test_build_tool_call_structured_output_config_smoke() -> None:
    import openvino_genai as ov_genai

    config = build_tool_call_structured_output_config(ov_genai, TOOL_SPECS)
    assert config.compound_grammar is not None
    assert CONSTRAINED_DECODING_MECHANISM.startswith("openvino_genai_structured_output")


def test_analysis_context_ceiling_and_p6_no_longer_requires_20k() -> None:
    from seam.analysis.efilter import _STAGE1_P6_BASELINE, StepView, _predictions

    def _sv(task_id: str, step_idx: int, context: int) -> StepView:
        return StepView(
            task_id=task_id,
            step_idx=step_idx,
            step_type="tool_call_synthesis",
            router_step_type="tool_call_synthesis",
            prompt_tokens_proxy=context,
            prompt_tokens_native=context,
            context_tokens_total=context,
            prompt_tokens_new=context,
            completion_tokens=32,
            kv_bytes_resident=(context + 32) * 100,
            peak_rss_bytes=1_000_000,
            cache_instrumented=True,
            cache_evicted=step_idx > 0,
            actual_wall_s=1.0 + step_idx,
            logged_t_pred_s=1.0 + step_idx,
            logged_deadline_s=DEADLINE_DISABLED_S,
            logged_target="local",
            logged_t_pred_prefill_s=context / 100.0,
            logged_t_pred_decode_s=3.2,
        )

    steps = [_sv("A", 0, 650), _sv("A", 1, 2600)]
    ceiling = per_task_context_ceiling(steps)
    assert ceiling["per_task"]["A"] == pytest.approx(2600 / 650)
    assert ceiling["max_context_tokens_is_gate"] is False

    # High ceiling + OP inside Stage-1 CI → FALSIFIED (C2b), even with peak << 20k.
    headline = {
        "deadline_s": 10.0,
        "over_provisioning": {
            "peak_kv_bytes_resident": {"point": 1.243, "lo": 1.12, "hi": 1.35},
            "peak_context_tokens": {"point": 1.2, "lo": 1.0, "hi": 1.4},
            "max_prompt_tokens_new": {"point": 1.2, "lo": 1.0, "hi": 1.4},
            "p95_required_decode_rate_tok_s": {"point": 1.2, "lo": 1.0, "hi": 1.4},
            "peak_rss_bytes": {"point": 1.2, "lo": 1.0, "hi": 1.4},
            "mean_arithmetic_intensity": {"point": 1.1, "lo": 1.0, "hi": 1.2},
        },
        "definition": "test",
        "distribution_units": {},
    }
    preds = _predictions(
        curve=[headline],
        headline=headline,
        feasibility={},
        materiality_ratio=1.2,
        stratified={"by_step_idx": {}},
        peak_context_tokens=5000,
        context_ceiling={"median": 4.0, "mean": 4.0, "definition": ceiling["definition"]},
    )
    assert "FALSIFIED" in preds["P6"]["verdict"]
    assert "20000" not in preds["P6"]["verdict"]
    assert preds["P6"]["c2_peak_context_tokens_is_gate"] is False
    assert _STAGE1_P6_BASELINE["context_ceiling_mean"] == pytest.approx(1.456)


def test_am032_and_scaffold_still_present() -> None:
    text = (_ROOT / "AMENDMENTS.md").read_text(encoding="utf-8")
    assert "## AM-032" in text
    assert "Finding note - C9 practical context ceiling" in text
    note = (_ROOT / "derived" / "efilter" / "c9_practical_ceiling_note.json").read_text(
        encoding="utf-8"
    )
    assert "0fe5e4c7-bb38-4666-826b-2c512b17a969" in note
