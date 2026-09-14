"""C2g: memory instrumentation, cap 5000, chunked-prefill audit, clearance gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from seam.agent.harness import StepRecord, TaskResult, run_task
from seam.agent.policy import DEADLINE_DISABLED_S, ThroughputModel
from seam.agent.steplog import step_to_record
from seam.agent.tools import TaskSpec, ToolWorld
from seam.analysis.c2g_memory import (
    fit_memory_vs_context,
    memory_vs_context_note_template,
    write_memory_vs_context_note,
)
from seam.backends.base import GenerationRequest, GenerationResult, ToolCall
from seam.backends.refusing_cloud import RefusingCloudBackend
from seam.gitinfo import repo_root
from seam.tools.efilter_run import (
    build_efilter_summary,
    evaluate_pilot_context_gate,
    nested_key_set_structure,
    stub_efilter_summary_kwargs,
    task_memory_instrumentation,
)

_ROOT = repo_root(Path(__file__).parent)
_THROUGHPUT = ThroughputModel(
    target="cpu-p", r_prefill_tok_s=100.0, r_decode_tok_s=10.0, measured_by_run_id="test"
)
_N_OUT = {"tool_call_synthesis": 64, "answer_synthesis": 64}
_WORLD = ToolWorld({"directories": {"/docs": ["a.txt", "b.txt", "c.txt"]}})


class _RssStub:
    name = "stub_local"
    model_ref = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def preflight(self) -> Any:
        return type("V", (), {"status": "OK", "reason": ""})()

    def estimate_context_tokens(self, request: GenerationRequest) -> int:
        return 200 + self.calls * 50

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        text = (
            '<tool_call>{"name": "list_files", ' '"arguments": {"directory": "/docs"}}</tool_call>'
        )
        return GenerationResult(
            text=text,
            tool_calls=(ToolCall(name="list_files", arguments={"directory": "/docs"}),),
            prompt_tokens=200 + (self.calls - 1) * 50,
            completion_tokens=20,
            completion_chars=len(text),
            completion_bytes=len(text.encode()),
            wall_ns=2_000_000,
            ttft_ns=500_000,
            backend=self.name,
            model_ref=self.model_ref,
            extra={"cache_instrumented": False},
        )


def test_efilter_yaml_pins_c2g_cap_5000() -> None:
    cfg = yaml.safe_load((_ROOT / "configs" / "efilter.yaml").read_text(encoding="utf-8"))
    assert cfg["workload"]["context_cap_tokens"] == 5000
    assert cfg.get("c2g_amendment") == "docs/CURSOR_PROMPT_C2g.md"
    assert cfg["chunked_prefill"]["enabled"] is False
    assert cfg["chunked_prefill"]["available_on_cpu_stateful_pipeline"] is False
    assert cfg["workload"]["pilot_min_median_context_ratio"] == 3.0
    assert cfg["workload"]["discard_first_task_warmup"] is True


def test_chunked_prefill_cpu_audit_artifact_exists() -> None:
    path = _ROOT / "derived" / "efilter" / "c2g_chunked_prefill_cpu_audit.json"
    assert path.is_file()
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["available"] is False
    assert audit["verdict"] == "NOT_AVAILABLE_ON_CPU_STATEFUL_PATH"
    assert isinstance(audit["property_names_tried"], list)
    assert len(audit["property_names_tried"]) >= 3
    assert "NPUW_LLM_PREFILL_CHUNK_SIZE" in {row["name"] for row in audit["property_names_tried"]}


def test_memory_vs_context_note_template_exists() -> None:
    path = _ROOT / "derived" / "efilter" / "c2g_memory_vs_context_note.json"
    assert path.is_file()
    note = json.loads(path.read_text(encoding="utf-8"))
    assert note["note_id"] == "c2g_memory_vs_context"
    assert note["status"] in {"TEMPLATE", "FILLED"}


def test_c9_prefill_activation_bound_note_exists() -> None:
    path = _ROOT / "derived" / "efilter" / "c9_prefill_activation_bound_note.json"
    assert path.is_file()
    note = json.loads(path.read_text(encoding="utf-8"))
    assert note["ledger"] == "C9"
    assert note["chunked_prefill_cpu"]["available"] is False


def test_task_memory_instrumentation_fields() -> None:
    steps = [
        StepRecord(
            run_id="r",
            program_id="p",
            step_idx=0,
            step_type="tool_call_synthesis",
            assigned_target="local",
            model_ref="m",
            t_start_ns=0,
            t_end_ns=1,
            prompt_tokens=100,
            completion_tokens=10,
            cached_prompt_tokens=0,
            completion_chars=10,
            completion_bytes=10,
            tool=None,
            usd_cost=0.0,
            privacy_class="synthetic_benchmark",
            terminated=False,
            retry_of=None,
            routing={"t_pred_s": 1.0, "deadline_s": DEADLINE_DISABLED_S},
            deadline_overrun=False,
            actual_wall_s=1.0,
            context_tokens_total=100,
            rss_before_generate=1_000,
            rss_peak_during_generate=2_000,
            rss_after_generate=1_500,
            peak_rss_bytes=2_000,
            free_memory_mb_min_during_generate=700.0,
            free_memory_mb_before_generate=800.0,
            free_memory_mb_after_generate=750.0,
        ),
        StepRecord(
            run_id="r",
            program_id="p",
            step_idx=1,
            step_type="tool_call_synthesis",
            assigned_target="local",
            model_ref="m",
            t_start_ns=2,
            t_end_ns=3,
            prompt_tokens=400,
            completion_tokens=10,
            cached_prompt_tokens=0,
            completion_chars=10,
            completion_bytes=10,
            tool=None,
            usd_cost=0.0,
            privacy_class="synthetic_benchmark",
            terminated=False,
            retry_of=None,
            routing={"t_pred_s": 1.0, "deadline_s": DEADLINE_DISABLED_S},
            deadline_overrun=False,
            actual_wall_s=1.0,
            context_tokens_total=400,
            rss_before_generate=1_500,
            rss_peak_during_generate=5_000,
            rss_after_generate=2_000,
            peak_rss_bytes=5_000,
            free_memory_mb_min_during_generate=200.0,
            free_memory_mb_before_generate=500.0,
            free_memory_mb_after_generate=300.0,
        ),
    ]
    result = TaskResult(
        task_id="T",
        program_id="p",
        success=False,
        submitted_answer=None,
        expected="x",
        realized_steps=2,
        escalated_steps=0,
        local_steps=2,
        jct_s=1.0,
        local_tokens=20,
        cloud_tokens=0,
        local_completion_chars=20,
        cloud_completion_chars=0,
        usd_cost=0.0,
        deadline_overruns=0,
        steps=steps,
    )
    mem = task_memory_instrumentation(
        result,
        free_before=900.0,
        free_after=850.0,
        rss_before=900,
        rss_after=2_100,
    )
    assert mem["free_memory_mb_min"] == pytest.approx(200.0)
    assert mem["free_memory_mb_min_step_idx"] == 1
    assert mem["free_memory_mb_min_context_tokens"] == 400
    assert mem["process_rss_peak"] == 5_000
    assert mem["process_rss_peak_step_idx"] == 1
    assert mem["free_memory_mb_at_task_start"] == pytest.approx(900.0)
    assert mem["free_memory_mb_at_task_end"] == pytest.approx(850.0)
    assert mem["free_memory_mb_before_task"] == pytest.approx(900.0)
    assert mem["free_memory_mb_after_task"] == pytest.approx(850.0)
    assert len(mem["step_memory"]) == 2
    assert mem["step_memory"][1]["rss_peak_during_generate"] == 5_000


def test_step_record_exposes_c2g_rss_fields() -> None:
    stub = _RssStub()
    with patch("seam.agent.harness.RssSampler") as sampler_cls:
        window = type(
            "W",
            (),
            {
                "peak_bytes": 9_000,
                "start_bytes": 4_000,
                "end_bytes": 5_000,
                "free_memory_mb_min": 321.0,
                "free_memory_mb_start": 400.0,
                "free_memory_mb_end": 350.0,
            },
        )()
        inst = sampler_cls.return_value
        inst.start.return_value = None
        inst.stop.return_value = window
        result = run_task(
            task=TaskSpec(task_id="M", prompt="list files", expected="1", min_tool_calls=0),
            world=_WORLD,
            local_backend=stub,
            cloud_backend=RefusingCloudBackend(run_id="test"),
            throughput=_THROUGHPUT,
            deadline_s=DEADLINE_DISABLED_S,
            n_out_pred_tokens=_N_OUT,
            max_steps=1,
            max_tokens=128,
            run_id="test",
            program_id="test/M",
            rss_sample_interval_s=0.05,
        )
    assert result.realized_steps == 1
    step = result.steps[0]
    rec = step_to_record(step)
    assert rec["rss_before_generate"] == 4_000
    assert rec["rss_peak_during_generate"] == 9_000
    assert rec["rss_after_generate"] == 5_000
    assert rec["peak_rss_bytes"] == 9_000
    assert rec["free_memory_mb_min_during_generate"] == pytest.approx(321.0)


def test_memory_floor_still_clearance_fatal_paging_canary_not() -> None:
    growth = {
        "max_context_tokens_observed": 4000,
        "per_task_ratios": [7.0, 8.0, 6.0, 5.0, 9.0],
    }
    per_task = [{"context_ratio_cmax_over_cmin": r} for r in growth["per_task_ratios"]]
    ok = evaluate_pilot_context_gate(
        growth,
        per_task=per_task,
        min_median_context_ratio=3.0,
        paging_invalidations=2,
        canary_invalidations=3,
        memory_floor_invalidations=0,
    )
    assert ok["cleared"] is True
    assert ok["paging_invalidations_run_fatal"] is False
    assert ok["canary_invalidations_run_fatal"] is False
    assert ok["memory_floor_invalidations_run_fatal"] is True

    blocked = evaluate_pilot_context_gate(
        growth,
        per_task=per_task,
        paging_invalidations=0,
        canary_invalidations=0,
        memory_floor_invalidations=1,
    )
    assert blocked["cleared"] is False


def test_fit_memory_vs_context_linear_and_superlinear() -> None:
    linear_tasks = [
        {
            "discarded_warmup": False,
            "task_id": "L",
            "step_memory": [
                {
                    "step_idx": i,
                    "context_tokens_total": 100 * (i + 1),
                    "rss_peak_during_generate": 1_000_000 + 1000 * 100 * (i + 1),
                    "free_memory_mb_before_generate": 8000.0,
                    "free_memory_mb_min_during_generate": 8000.0 - 0.01 * 100 * (i + 1),
                }
                for i in range(5)
            ],
        }
    ]
    lin = fit_memory_vs_context(linear_tasks)
    assert lin["n_points"] == 5
    assert lin["rss_peak_vs_context"]["linear"]["r_squared"] == pytest.approx(1.0, abs=1e-6)

    super_tasks = [
        {
            "discarded_warmup": False,
            "task_id": "S",
            "step_memory": [
                {
                    "step_idx": i,
                    "context_tokens_total": ctx,
                    "rss_peak_during_generate": int(1e-2 * ctx * ctx),
                    "free_memory_mb_before_generate": 8000.0,
                    "free_memory_mb_min_during_generate": 8000.0 - 1e-6 * ctx * ctx,
                }
                for i, ctx in enumerate([100, 200, 400, 800, 1600])
            ],
        }
    ]
    sup = fit_memory_vs_context(super_tasks)
    assert sup["overall_verdict"] == "superlinear"
    assert (sup["rss_peak_vs_context"]["quadratic"]["r_squared"] or 0) > (
        (sup["rss_peak_vs_context"]["linear"]["r_squared"] or 0) + 0.02
    )


def test_write_memory_vs_context_note_fills_from_tasks(tmp_path: Path) -> None:
    path = tmp_path / "note.json"
    per_task = [
        {
            "discarded_warmup": False,
            "task_id": "T0",
            "step_memory": [
                {
                    "step_idx": 0,
                    "context_tokens_total": 100,
                    "rss_peak_during_generate": 1000,
                    "free_memory_mb_before_generate": 100.0,
                    "free_memory_mb_min_during_generate": 90.0,
                },
                {
                    "step_idx": 1,
                    "context_tokens_total": 200,
                    "rss_peak_during_generate": 2000,
                    "free_memory_mb_before_generate": 90.0,
                    "free_memory_mb_min_during_generate": 70.0,
                },
                {
                    "step_idx": 2,
                    "context_tokens_total": 300,
                    "rss_peak_during_generate": 3000,
                    "free_memory_mb_before_generate": 70.0,
                    "free_memory_mb_min_during_generate": 40.0,
                },
            ],
        }
    ]
    payload = write_memory_vs_context_note(path, per_task=per_task, run_id="test-run")
    assert path.is_file()
    assert payload["status"] == "FILLED"
    assert payload["run_id"] == "test-run"
    assert payload["analysis"]["n_points"] == 3


def test_builder_dry_run_includes_c2g_memory_fields() -> None:
    cfg = yaml.safe_load((_ROOT / "configs" / "efilter.yaml").read_text(encoding="utf-8"))
    a = build_efilter_summary(**stub_efilter_summary_kwargs(cfg, mode="pilot"))
    b = build_efilter_summary(**stub_efilter_summary_kwargs(cfg, mode="pilot"))
    assert nested_key_set_structure(a) == nested_key_set_structure(b)
    row = a["per_task"][0]
    for key in (
        "free_memory_mb_min",
        "free_memory_mb_min_step_idx",
        "free_memory_mb_min_context_tokens",
        "process_rss_peak",
        "process_rss_peak_step_idx",
        "free_memory_mb_at_task_start",
        "free_memory_mb_at_task_end",
        "step_memory",
    ):
        assert key in row, key
    assert "rss_before_generate" in row["step_memory"][0]
    assert "memory_vs_context" in a
    assert a["workload_memory_controls"]["context_cap_tokens"] == 5000


def test_launch_scripts_exist_ascii() -> None:
    pilot = _ROOT / "tools" / "launch_efilter_c2g_pilot.ps1"
    full = _ROOT / "tools" / "launch_efilter_c2g_full.ps1"
    assert pilot.is_file()
    assert full.is_file()
    for path in (pilot, full):
        text = path.read_text(encoding="utf-8")
        assert "≥" not in text
        assert "-" not in text
        assert "-" not in text
    assert "READY_FOR_PILOT" in pilot.read_text(encoding="utf-8")


def test_memory_vs_context_template_shape() -> None:
    tmpl = memory_vs_context_note_template()
    assert tmpl["status"] == "TEMPLATE"
    assert tmpl["run_id"] is None
