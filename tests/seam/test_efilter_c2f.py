"""C2f: canary baseline, per-endpoint canary admissibility, discard-first warmup."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml

from seam.agent.policy import ThroughputModel
from seam.analysis.efilter import EfilterInputs, StepView, _curve_point, _with_admissible_basis
from seam.gitinfo import repo_root
from seam.tools.canary_baseline import (
    DEFAULT_MARGIN,
    canary_pair_drift_stats,
    derive_canary_gate,
)
from seam.tools.efilter_run import (
    build_efilter_summary,
    canary_admissibility,
    classify_invalidation_reason,
    evaluate_pilot_context_gate,
    nested_key_set_structure,
    stub_efilter_summary_kwargs,
    timed_block_invalidation_split,
)

_ROOT = repo_root(Path(__file__).parent)


def test_wallclock_timeout_audit_artifact_none() -> None:
    path = _ROOT / "derived" / "efilter" / "c2f_wallclock_timeout_audit.json"
    assert path.is_file()
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["verdict"] == "NONE"
    assert audit["section_3_stands"] is True
    assert audit["per_endpoint_canary_scoping"] == "implemented"


def test_harness_and_backend_have_no_generation_wallclock_timeout() -> None:
    """Static scan: generation path must not set a wall-clock timeout that truncates tokens."""
    harness = (_ROOT / "seam" / "agent" / "harness.py").read_text(encoding="utf-8")
    backend = (_ROOT / "seam" / "backends" / "local_openvino.py").read_text(encoding="utf-8")
    # No timeout= on generate / GenerationConfig path.
    assert "timeout" not in harness.lower()
    assert re.search(r"GenerationConfig\s*\(", backend)
    assert "max_new_tokens" in backend
    # Forbid common wall-clock truncation knobs on the GenAI config assignment block.
    gen_block = backend[
        backend.index("GenerationConfig()") : backend.index("result = self._pipe.generate")
    ]
    for forbidden in ("timeout", "max_duration", "max_time", "deadline"):
        assert forbidden not in gen_block.lower()

    tree = ast.parse(harness)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "timeout":
                    pytest.fail(f"harness.py uses timeout= kwarg at line {node.lineno}")


def test_canary_baseline_threshold_is_p95_plus_margin() -> None:
    drifts = [0.01, 0.02, 0.03, 0.04, 0.05] + [0.01] * 25
    dist = canary_pair_drift_stats(drifts)
    assert dist["n_pairs"] == 30
    gate = derive_canary_gate(
        dist,
        margin=DEFAULT_MARGIN,
        baseline_run_id="00000000-0000-0000-0000-000000000001",
        spacing_s=10.0,
        n_samples=31,
    )
    assert gate["threshold"] == pytest.approx(dist["p95"] + DEFAULT_MARGIN)
    assert gate["idle_p95"] == dist["p95"]
    assert gate["margin"] == DEFAULT_MARGIN
    assert gate["baseline_run_id"]
    assert "not chosen to make a pilot pass" in gate["justification"]


def test_canary_admissibility_tagging() -> None:
    ok = canary_admissibility(
        {
            "canary_gate": {"verdict": "pass", "reasons": []},
            "canary_relative_drift": 0.01,
            "canary_relative_drift_immediate": 0.02,
            "canary_drift_authoritative": "settled",
            "invalid_reasons": [],
        }
    )
    assert ok["canary_admissible"] is True
    fail = canary_admissibility(
        {
            "canary_gate": {
                "verdict": "fail",
                "reasons": ["compute canary settled drift 0.23 > 0.15"],
            },
            "canary_relative_drift": 0.23,
            "canary_relative_drift_immediate": 0.20,
            "canary_drift_authoritative": "settled",
            "invalid_reasons": ["compute canary settled drift 0.23 > 0.15"],
        }
    )
    assert fail["canary_admissible"] is False
    assert fail["canary_admissible_reasons"]


def test_classify_invalidation_splits_canary_from_memory() -> None:
    assert classify_invalidation_reason("sustained hard page reads") == "paging"
    assert classify_invalidation_reason("compute canary settled drift 0.2 > 0.15") == "canary"
    assert classify_invalidation_reason("available memory 400.0 MiB < 500.0 MiB") == "memory_floor"


def test_pilot_clears_with_canary_failures_if_seal_ratio_ok() -> None:
    growth = {
        "max_context_tokens_observed": 5000,
        "per_task_ratios": [7.0, 8.0, 6.0, 5.0, 9.0],
    }
    per_task = [{"context_ratio_cmax_over_cmin": r} for r in growth["per_task_ratios"]]
    gate = evaluate_pilot_context_gate(
        growth,
        per_task=per_task,
        min_median_context_ratio=3.0,
        memory_or_canary_invalidations=2,
        paging_invalidations=1,
        canary_invalidations=2,
        memory_floor_invalidations=0,
    )
    assert gate["cleared"] is True
    assert gate["canary_invalidations_run_fatal"] is False
    assert gate["paging_invalidations_run_fatal"] is False
    assert gate["wallclock_timeout_audit_verdict"] == "NONE"

    gate_mem = evaluate_pilot_context_gate(
        growth,
        per_task=per_task,
        canary_invalidations=0,
        memory_floor_invalidations=1,
        paging_invalidations=0,
    )
    assert gate_mem["cleared"] is False


def test_timed_split_counts_canary_separately() -> None:
    per_task = [
        {
            "paging_admissible": True,
            "canary_admissible": False,
            "machine_measurement_invalid_reasons": ["compute canary settled drift 0.2 > 0.15"],
        },
        {
            "paging_admissible": False,
            "canary_admissible": True,
            "machine_measurement_invalid_reasons": ["sustained hard page reads above threshold"],
        },
        {
            "paging_admissible": True,
            "canary_admissible": True,
            "machine_measurement_invalid_reasons": ["available memory 100.0 MiB < 500.0 MiB"],
        },
    ]
    split = timed_block_invalidation_split(per_task)
    assert split["canary_invalidations_inside_timed_blocks"] == 1
    assert split["paging_invalidations_inside_timed_blocks"] == 1
    assert split["memory_floor_invalidations_inside_timed_blocks"] == 1
    assert split["clearance_fatal_invalidations_inside_timed_blocks"] == 1


def test_discard_first_warmup_in_stub_summary() -> None:
    cfg = yaml.safe_load((_ROOT / "configs" / "efilter.yaml").read_text(encoding="utf-8"))
    assert cfg["workload"]["discard_first_task_warmup"] is True
    assert cfg.get("c2f_amendment") == "docs/CURSOR_PROMPT_C2f.md"
    kwargs = stub_efilter_summary_kwargs(cfg, mode="pilot")
    summary = build_efilter_summary(**kwargs)
    assert summary["discard_first_task_warmup"] is True
    assert summary["discarded_warmup_task"]["execution_position"] == 0
    assert summary["discarded_warmup_task"]["discarded_warmup"] is True
    positions = [row["execution_position"] for row in summary["canary_drift_by_execution_position"]]
    assert 0 in positions
    # Timed tasks only in per_task / ratio - discarded warmup excluded.
    assert all(not row.get("discarded_warmup") for row in summary["per_task"])
    assert all(row.get("execution_position", 1) >= 1 for row in summary["per_task"])


def test_builder_dry_run_key_parity_includes_c2f_fields() -> None:
    cfg = yaml.safe_load((_ROOT / "configs" / "efilter.yaml").read_text(encoding="utf-8"))
    a = build_efilter_summary(**stub_efilter_summary_kwargs(cfg, mode="pilot"))
    b = build_efilter_summary(**stub_efilter_summary_kwargs(cfg, mode="pilot"))
    assert nested_key_set_structure(a) == nested_key_set_structure(b)
    assert "canary_admissible" in a["per_task"][0]
    assert "canary_gate" in a
    assert a["wallclock_timeout_audit"]["verdict"] == "NONE"


def test_analysis_timing_uses_canary_paging_intersection() -> None:
    steps = [
        StepView(
            task_id="T0",
            step_idx=0,
            step_type="tool_call_synthesis",
            router_step_type="tool_call_synthesis",
            prompt_tokens_proxy=100,
            prompt_tokens_native=100,
            context_tokens_total=100,
            prompt_tokens_new=100,
            completion_tokens=10,
            kv_bytes_resident=1000,
            peak_rss_bytes=None,
            cache_instrumented=False,
            cache_evicted=False,
            actual_wall_s=1.0,
            logged_t_pred_s=0.5,
            logged_deadline_s=1e9,
            logged_target="local",
            deadline_overrun=False,
        ),
        StepView(
            task_id="T1",
            step_idx=0,
            step_type="tool_call_synthesis",
            router_step_type="tool_call_synthesis",
            prompt_tokens_proxy=200,
            prompt_tokens_native=200,
            context_tokens_total=200,
            prompt_tokens_new=200,
            completion_tokens=10,
            kv_bytes_resident=2000,
            peak_rss_bytes=None,
            cache_instrumented=False,
            cache_evicted=False,
            actual_wall_s=10.0,
            logged_t_pred_s=0.5,
            logged_deadline_s=1e9,
            logged_target="local",
            deadline_overrun=True,
        ),
    ]
    inputs = EfilterInputs(
        run_id="test-c2f",
        steps=steps,
        throughput=ThroughputModel(
            target="cpu-p",
            r_prefill_tok_s=100.0,
            r_decode_tok_s=50.0,
            measured_by_run_id="stub",
        ),
        n_out_pred_tokens={"tool_call_synthesis": 10, "answer_synthesis": 10},
        kv_bytes_per_token=10,
        cache_instrumented=False,
        cache_probe={},
        summary={
            "per_task": [
                {
                    "task_id": "T0",
                    "paging_admissible": True,
                    "canary_admissible": True,
                    "jct_s": 1.0,
                },
                {
                    "task_id": "T1",
                    "paging_admissible": True,
                    "canary_admissible": False,
                    "canary_admissible_reasons": ["canary drift"],
                    "jct_s": 10.0,
                },
            ]
        },
        integrity_verified=True,
        paging_admissible_by_task={"T0": True, "T1": True},
        canary_admissible_by_task={"T0": True, "T1": False},
    )
    frac = inputs.admissible_fraction()
    assert frac["canary_admissible_task_fraction"] == pytest.approx(0.5)
    assert frac["timing_admissible_task_fraction"] == pytest.approx(0.5)

    point = _curve_point(
        inputs,
        deadline_s=1e9,
        r_prefill_tok_s=100.0,
        resamples=10,
        seed=1,
        materiality=1.2,
        use_native_tokens=False,
    )
    assert point["unfiltered_envelope"]["basis"] == "all_blocks"
    # Timing excludes canary-inadmissible T1 wall=10s.
    assert point["p95_actual_wall_s_surviving"] == pytest.approx(1.0)
    assert point["p95_actual_wall_s_surviving_meta"]["basis"] == "timing_admissible_only"
    assert point["p95_actual_wall_s_surviving_meta"]["canary_admissible_task_fraction"] == (
        pytest.approx(0.5)
    )
    # Envelope still sees both tasks' KV.
    assert point["unfiltered_envelope"]["peak_kv_bytes_resident"]["per_task_max"] == (
        pytest.approx(2000.0)
    )
    meta = _with_admissible_basis(1.0, fraction=frac, basis="timing_admissible_only")
    assert meta["canary_admissible_task_fraction"] == pytest.approx(0.5)


def test_measurement_yaml_has_canary_gate_placeholder() -> None:
    cfg = yaml.safe_load((_ROOT / "configs" / "measurement.yaml").read_text(encoding="utf-8"))
    canary = cfg["canary"]
    assert canary["gate_mode"] == "asserted_pending_baseline"
    assert canary["baseline_run_id"] is None
    assert canary["margin_above_idle_p95"] == pytest.approx(0.05)
    assert "PLACEHOLDER" in canary["threshold_justification"]


def test_launch_scripts_exist() -> None:
    assert (_ROOT / "tools" / "launch_canary_baseline.ps1").is_file()
    assert (_ROOT / "tools" / "launch_efilter_c2f_pilot.ps1").is_file()
    assert (_ROOT / "tools" / "launch_efilter_c2f_full.ps1").is_file()
