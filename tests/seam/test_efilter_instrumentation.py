"""Phase-1 instrumentation for E-FILTER Stage 1.

Covers the pieces whose silent failure would invalidate the study rather than break it: the
escalation-disabled sentinel, the KV constant, the extended step record, and the NDJSON writer's
refusal to emit non-finite JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from seam.agent.harness import StepRecord, run_task
from seam.agent.policy import (
    DEADLINE_DISABLED_MIN_HEADROOM,
    DEADLINE_DISABLED_S,
    ThroughputModel,
    assert_escalation_disabled,
    decide,
)
from seam.agent.steplog import StepLogWriter, read_steps, step_to_record
from seam.agent.tools import TaskSpec, ToolWorld
from seam.backends.base import GenerationRequest, GenerationResult, ToolCall
from seam.backends.refusing_cloud import RefusingCloudBackend
from seam.errors import ConfigError, EscalationRefusedError, SeamError
from seam.kvmath import BYTES_PER_ELEMENT, KV_FORMULA, KvGeometry, load_kv_geometry

_THROUGHPUT = ThroughputModel(
    target="cpu-p", r_prefill_tok_s=100.0, r_decode_tok_s=10.0, measured_by_run_id="test"
)
_N_OUT = {"tool_call_synthesis": 64, "answer_synthesis": 16}


# ==================================================================================================
# The sentinel
# ==================================================================================================


def test_sentinel_is_finite_and_json_serializable() -> None:
    assert math.isfinite(DEADLINE_DISABLED_S)
    assert "Infinity" not in json.dumps({"deadline_s": DEADLINE_DISABLED_S}, allow_nan=False)


def test_sentinel_keeps_a_realistic_step_local() -> None:
    decision = decide(
        throughput=_THROUGHPUT,
        deadline_s=DEADLINE_DISABLED_S,
        prompt_tokens=4_000,
        step_type="tool_call_synthesis",
        n_out_pred_tokens=_N_OUT,
    )
    assert decision.assigned_target == "local"
    assert_escalation_disabled(deadline_s=DEADLINE_DISABLED_S, t_pred_s=decision.t_pred_s)


def test_decide_refuses_non_finite_deadline() -> None:
    with pytest.raises(SeamError, match="non-finite"):
        decide(
            throughput=_THROUGHPUT,
            deadline_s=float("inf"),
            prompt_tokens=10,
            step_type="answer_synthesis",
            n_out_pred_tokens=_N_OUT,
        )


def test_assert_escalation_disabled_rejects_a_merely_large_deadline() -> None:
    with pytest.raises(SeamError, match="DEADLINE_DISABLED_S"):
        assert_escalation_disabled(deadline_s=1e6, t_pred_s=1.0)


def test_assert_escalation_disabled_rejects_insufficient_headroom() -> None:
    t_pred = DEADLINE_DISABLED_S / DEADLINE_DISABLED_MIN_HEADROOM * 2
    with pytest.raises(SeamError, match="within"):
        assert_escalation_disabled(deadline_s=DEADLINE_DISABLED_S, t_pred_s=t_pred)


# ==================================================================================================
# KV geometry
# ==================================================================================================

_QWEN3_4B_CONFIG = {
    "model_type": "qwen3",
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_attention_heads": 32,
    "hidden_size": 2560,
}


def _write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model_dir


def test_kv_geometry_matches_the_project_formula(tmp_path: Path) -> None:
    model_dir = _write_config(tmp_path, _QWEN3_4B_CONFIG)
    geometry = load_kv_geometry(model_dir, assumed_kv_dtype="f16")

    # The formula, not a dtype assumption: the KV precision is a runtime property and this platform
    # reports u8 rather than the f16 an offline capacity model would have assumed.
    expected = 2 * 36 * 8 * 128 * geometry.kv_dtype_bytes
    assert geometry.bytes_per_token == expected
    # Same formula as censor/kv_math.kv_bytes_per_token, which the earlier capacity analysis used.
    assert KV_FORMULA == "2 * n_layers * n_kv_heads * head_dim * dtype_bytes"
    assert geometry.bytes_for(2048) == expected * 2048
    assert geometry.to_record()["kv_bytes_per_token"] == expected


def test_kv_geometry_agrees_with_the_earlier_capacity_model(tmp_path: Path) -> None:
    """The seam constant must equal censor/kv_math's for the same architecture and dtype."""
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_censor_kv_math", root / "censor" / "kv_math.py")
    assert spec is not None and spec.loader is not None
    censor_kv = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = censor_kv
    spec.loader.exec_module(censor_kv)

    model_dir = _write_config(tmp_path, _QWEN3_4B_CONFIG)
    geometry = load_kv_geometry(model_dir, device="NO_SUCH_DEVICE", assumed_kv_dtype="f16")
    reference_spec = censor_kv.ModelSpec(
        name="qwen3-4b",
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        d_model=2560,
        params_total=4.0e9,
        params_active=4.0e9,
        attention="GQA",
        source="test fixture",
    )
    assert geometry.bytes_per_token == censor_kv.kv_bytes_per_token(reference_spec, "fp16")


def test_kv_geometry_refuses_to_derive_head_dim(tmp_path: Path) -> None:
    config = {k: v for k, v in _QWEN3_4B_CONFIG.items() if k != "head_dim"}
    model_dir = _write_config(tmp_path, config)
    with pytest.raises(ConfigError, match="head_dim"):
        load_kv_geometry(model_dir)


def test_kv_dtype_source_records_an_assumption(tmp_path: Path) -> None:
    model_dir = _write_config(tmp_path, _QWEN3_4B_CONFIG)
    geometry = load_kv_geometry(model_dir, device="NO_SUCH_DEVICE", assumed_kv_dtype="f16")
    assert geometry.kv_dtype_source.startswith("assumed:")
    assert BYTES_PER_ELEMENT[geometry.kv_dtype] == geometry.kv_dtype_bytes


# ==================================================================================================
# Step log
# ==================================================================================================


def _step(**overrides: Any) -> StepRecord:
    base: dict[str, Any] = {
        "run_id": "r",
        "program_id": "p",
        "step_idx": 0,
        "step_type": "tool_call_synthesis",
        "assigned_target": "local",
        "model_ref": "m",
        "t_start_ns": 0,
        "t_end_ns": 1,
        "prompt_tokens": 800,
        "completion_tokens": 40,
        "cached_prompt_tokens": 0,
        "completion_chars": 100,
        "completion_bytes": 100,
        "tool": None,
        "usd_cost": 0.0,
        "privacy_class": "synthetic_benchmark",
        "terminated": False,
        "retry_of": None,
        "routing": {"t_pred_s": 1.0, "deadline_s": DEADLINE_DISABLED_S},
        "deadline_overrun": False,
        "actual_wall_s": 1.0,
    }
    base.update(overrides)
    return StepRecord(**base)


def test_step_record_carries_the_efilter_fields() -> None:
    record = step_to_record(
        _step(
            prompt_tokens_proxy=750,
            context_tokens_total=800,
            prompt_tokens_new=800,
            kv_bytes_per_token=147456,
            kv_bytes_resident_before=800 * 147456,
            kv_bytes_resident=840 * 147456,
            peak_rss_bytes=5_000_000_000,
            cache_instrumented=False,
        )
    )
    for field in (
        "prompt_tokens_proxy",
        "context_tokens_total",
        "prompt_tokens_new",
        "kv_bytes_per_token",
        "kv_bytes_resident",
        "kv_bytes_resident_before",
        "peak_rss_bytes",
        "rss_before_generate",
        "rss_peak_during_generate",
        "rss_after_generate",
        "cache_evicted",
        "evicted_bytes",
        "cache_instrumented",
    ):
        assert field in record, field
    assert record["kv_bytes_resident"] == 840 * 147456


def test_kv_fields_default_to_none_not_zero() -> None:
    record = step_to_record(_step())
    assert record["kv_bytes_per_token"] is None
    assert record["kv_bytes_resident"] is None


def test_step_writer_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "steps.ndjson"
    with StepLogWriter(path) as writer:
        writer.write(_step(step_idx=0))
        writer.write(_step(step_idx=1))
        assert writer.n_written == 2
    records = read_steps(path)
    assert [r["step_idx"] for r in records] == [0, 1]


def test_step_writer_refuses_non_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "steps.ndjson"
    writer = StepLogWriter(path)
    with pytest.raises(ValueError):
        writer.write(_step(routing={"deadline_s": float("inf")}))


def test_read_steps_refuses_an_infinity_token(tmp_path: Path) -> None:
    path = tmp_path / "steps.ndjson"
    path.write_text('{"step_idx": 0, "deadline_s": Infinity}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        read_steps(path)


# ==================================================================================================
# The refusing cloud backend, in place
# ==================================================================================================


class _StubLocal:
    """Local backend that emits one tool call, then a final answer."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub_local"

    @property
    def model_ref(self) -> str:
        return "stub"

    def preflight(self) -> Any:  # pragma: no cover - not exercised
        raise NotImplementedError

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        prompt_chars = sum(len(str(m.get("content", ""))) for m in request.messages)
        if self.calls == 1:
            text = (
                '<tool_call>{"name": "list_files", "arguments": {"directory": "/docs"}}</tool_call>'
            )
            calls: tuple[ToolCall, ...] = (
                ToolCall(name="list_files", arguments={"directory": "/docs"}),
            )
        else:
            text = '<tool_call>{"name": "submit_answer", "arguments": {"answer": "3"}}</tool_call>'
            calls = (ToolCall(name="submit_answer", arguments={"answer": "3"}),)
        return GenerationResult(
            text=text,
            tool_calls=calls,
            prompt_tokens=prompt_chars // 3,
            completion_tokens=len(text) // 4,
            completion_chars=len(text),
            completion_bytes=len(text.encode()),
            wall_ns=1_000_000,
            ttft_ns=500_000,
            backend=self.name,
            model_ref=self.model_ref,
            extra={"cache_instrumented": False},
        )


_WORLD = ToolWorld({"directories": {"/docs": ["a.txt", "b.txt", "c.txt"]}})
_TASK = TaskSpec(
    task_id="T01", prompt="How many files are in /docs?", expected="3", min_tool_calls=1
)


def test_arm_l_run_stays_local_and_populates_kv_fields() -> None:
    sink: list[StepRecord] = []
    result = run_task(
        task=_TASK,
        world=_WORLD,
        local_backend=_StubLocal(),
        cloud_backend=RefusingCloudBackend(run_id="test"),
        throughput=_THROUGHPUT,
        deadline_s=DEADLINE_DISABLED_S,
        n_out_pred_tokens=_N_OUT,
        max_steps=4,
        max_tokens=128,
        run_id="test",
        program_id="test/T01",
        kv_bytes_per_token=147456,
        step_sink=sink.append,
    )

    assert result.escalated_steps == 0
    assert len(sink) == result.realized_steps
    for step in result.steps:
        assert step.assigned_target == "local"
        assert step.kv_bytes_per_token == 147456
        assert (
            step.kv_bytes_resident == (step.context_tokens_total + step.completion_tokens) * 147456
        )
        assert step.kv_bytes_resident_before == step.context_tokens_total * 147456
        assert step.prompt_tokens_new == step.context_tokens_total
        assert step.cache_instrumented is False
        assert step.prompt_tokens_proxy > 0
        assert_escalation_disabled(
            deadline_s=DEADLINE_DISABLED_S, t_pred_s=step.routing["t_pred_s"]
        )
    # Context grows: the transcript accumulates.
    assert result.steps[-1].context_tokens_total > result.steps[0].context_tokens_total
    # Step 0 has no predecessor to evict; later steps re-prefill the whole transcript.
    assert result.steps[0].cache_evicted is False
    assert result.steps[1].cache_evicted is True
    assert result.steps[1].evicted_bytes == result.steps[0].kv_bytes_resident


def test_refusing_cloud_backend_raises_and_is_not_a_backend_error() -> None:
    backend = RefusingCloudBackend()
    assert backend.preflight().status == "UNSUPPORTED"
    with pytest.raises(EscalationRefusedError):
        backend.generate(GenerationRequest(messages=[], system="", tools=(), max_tokens=1))
    assert backend.call_attempts == 1


def test_escalation_in_arm_l_kills_the_run() -> None:
    """A step routed to cloud must abort, not fall back to local and look like Arm L."""
    tight = 1e-9
    with pytest.raises(EscalationRefusedError):
        run_task(
            task=_TASK,
            world=_WORLD,
            local_backend=_StubLocal(),
            cloud_backend=RefusingCloudBackend(),
            throughput=_THROUGHPUT,
            deadline_s=tight,
            n_out_pred_tokens=_N_OUT,
            max_steps=4,
            max_tokens=128,
            run_id="test",
            program_id="test/T01",
        )


def test_kv_geometry_dataclass_is_frozen() -> None:
    geometry = KvGeometry(
        model_name="m",
        n_layers=36,
        n_kv_heads=8,
        head_dim=128,
        n_attention_heads=32,
        kv_dtype="f16",
        kv_dtype_bytes=2,
        kv_dtype_source="device_readback",
        config_path="c",
        config_sha256="0" * 64,
    )
    with pytest.raises(AttributeError):
        geometry.n_layers = 1  # type: ignore[misc]
