"""C2e: builder-based dry-run, outputs.tasks schema, per-endpoint paging admissibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from seam.agent.policy import ThroughputModel
from seam.analysis.efilter import (
    EfilterInputs,
    StepView,
    _curve_point,
    _with_admissible_basis,
)
from seam.gitinfo import repo_root
from seam.manifest import validate_manifest
from seam.tools.efilter_run import (
    build_efilter_summary,
    efilter_output_paths,
    evaluate_pilot_context_gate,
    nested_key_set_structure,
    paging_admissibility,
    split_invalidation_reasons,
    stub_efilter_summary_kwargs,
    timed_block_invalidation_split,
)

_ROOT = repo_root(Path(__file__).parent)


def _load_efilter_cfg() -> dict[str, Any]:
    from seam.tools.efilter_run import _load_cfg

    cfg, _ = _load_cfg(_ROOT)
    return cfg


def test_schema_accepts_outputs_tasks() -> None:
    schema = json.loads(
        (_ROOT / "seam" / "schemas" / "run_manifest.schema.json").read_text(encoding="utf-8")
    )
    outputs = schema["properties"]["outputs"]
    assert "tasks" in outputs["properties"]
    assert outputs["additionalProperties"] is False


def test_efilter_output_paths_include_tasks() -> None:
    paths = efilter_output_paths()
    assert paths["steps"] == "steps.ndjson"
    assert paths["tasks"] == "tasks.jsonl"


def test_manifest_outputs_tasks_validates(
    fake_config: Any,
    fake_repo: Path,
    clean_git_state: Any,
    minimal_workload: dict[str, Any],
) -> None:
    from tests.test_manifest import VERIFIED_TOPOLOGY, _build

    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        topology_override=VERIFIED_TOPOLOGY,
    )
    manifest["outputs"]["tasks"] = "tasks.jsonl"
    validate_manifest(manifest)  # must not raise


def test_builder_dry_run_and_real_key_sets_identical() -> None:
    cfg = _load_efilter_cfg()
    a = build_efilter_summary(**stub_efilter_summary_kwargs(cfg, mode="pilot"))
    b_kwargs = stub_efilter_summary_kwargs(cfg, mode="pilot")
    b_kwargs["run_wall_s"] = 99.0
    b_kwargs["n_steps"] = 8
    b_kwargs["per_task"] = list(b_kwargs["per_task"]) + [
        {**b_kwargs["per_task"][0], "task_id": "C2T_STUB_EXTRA", "order_index": 2}
    ]
    b = build_efilter_summary(**b_kwargs)
    assert nested_key_set_structure(a) == nested_key_set_structure(b)
    assert "per_task" in a
    assert "paging_admissible" in a["per_task"][0]
    assert a["lifecycle"]["summary_builder"] == "build_efilter_summary"
    assert (
        "e66701aa-37b9-4b00-b86b-7ffb477f4e66" in a["c9_output_path_defects_note"]["citing_run_ids"]
    )


def test_startup_dry_run_uses_builder_and_outputs_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from seam.stdio_utf8 import NON_ASCII_PROBE
    from seam.tools import efilter_run as er

    printed: list[str] = []
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "builtins.print", lambda *a, **_k: printed.append(" ".join(str(x) for x in a))
    )
    monkeypatch.setattr(er, "log_event", lambda *_a, **_k: None)

    class _Handle:
        run_id = "dry-run-c2e"
        run_dir = tmp_path / "raw" / "dry-run-c2e"

        def __init__(self) -> None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            # Minimal sealed artifacts so dry-run post-checks can read them.
            (self.run_dir / "summary.json").write_text("{}", encoding="utf-8")
            (self.run_dir / "manifest.json").write_text(
                json.dumps({"outputs": {"tasks": "tasks.jsonl", "steps": "steps.ndjson"}}),
                encoding="utf-8",
            )

    def _fake_emit(**kwargs: Any) -> _Handle:
        captured.update(kwargs)
        handle = _Handle()
        # Persist the summary the dry-run built so post-checks see real keys.
        (handle.run_dir / "summary.json").write_text(
            json.dumps(kwargs["summary"], sort_keys=True), encoding="utf-8"
        )
        (handle.run_dir / "manifest.json").write_text(
            json.dumps({"outputs": kwargs.get("outputs") or {}}), encoding="utf-8"
        )
        if kwargs.get("before_integrity_hash"):
            kwargs["before_integrity_hash"](
                type("RD", (), {"path": handle.run_dir, "append_ndjson": None})()
            )
        return handle

    monkeypatch.setattr(er, "_prepare_isolated_emit_root", lambda *_a, **_k: {"platform": "test"})
    monkeypatch.setattr("seam.manifest.emit", _fake_emit)
    monkeypatch.setattr("seam.rawstore.verify_sealed", lambda _d: True)
    monkeypatch.setattr(
        er,
        "capture_power_state",
        lambda: type(
            "P",
            (),
            {
                "on_battery": False,
                "battery_pct": 100.0,
                "charging": False,
                "battery_saver": False,
            },
        )(),
    )
    monkeypatch.setattr(er, "manifest_power_state", lambda *_a, **_k: {})
    monkeypatch.setattr(er, "assert_acyclic", lambda *_a, **_k: None)

    cfg = _load_efilter_cfg()
    result = er._efilter_startup_dry_run(root=_ROOT, cfg=cfg, allow_dirty=True)
    assert result["summary_builder"] == "build_efilter_summary"
    assert result["outputs_catalog"]["tasks"] == "tasks.jsonl"
    assert captured["outputs"]["tasks"] == "tasks.jsonl"
    assert captured["summary"]["lifecycle"]["summary_builder"] == "build_efilter_summary"
    assert "per_task" in captured["summary"]
    assert any(NON_ASCII_PROBE in p for p in printed)


def test_paging_admissibility_tagging() -> None:
    ok = paging_admissibility(
        {"paging_gate": {"verdict": "pass", "reasons": []}, "invalid_reasons": []}
    )
    assert ok["paging_admissible"] is True
    fail = paging_admissibility(
        {
            "paging_gate": {
                "verdict": "fail",
                "reasons": ["sustained hard page reads above threshold"],
            },
            "invalid_reasons": ["sustained hard page reads above threshold"],
        }
    )
    assert fail["paging_admissible"] is False
    assert fail["paging_admissible_reasons"]
    warmup = paging_admissibility(
        {"paging_gate": {"verdict": "not_sampled", "reasons": []}, "invalid_reasons": []}
    )
    assert warmup["paging_admissible"] is True


def test_pilot_clears_with_paging_failures_if_ratio_ok() -> None:
    growth = {"max_context_tokens_observed": 5000, "per_task_ratios": [7.0, 8.0, 6.0, 5.0, 9.0]}
    per_task = [{"context_ratio_cmax_over_cmin": r} for r in growth["per_task_ratios"]]
    gate = evaluate_pilot_context_gate(
        growth,
        per_task=per_task,
        min_median_context_ratio=3.0,
        memory_or_canary_invalidations=0,
        paging_invalidations=2,
    )
    assert gate["cleared"] is True
    assert gate["paging_invalidations"] == 2
    assert gate["paging_invalidations_run_fatal"] is False

    # C2f: canary alone is not clearance-fatal (timeout audit NONE); memory floor still is.
    gate_canary = evaluate_pilot_context_gate(
        growth,
        per_task=per_task,
        memory_or_canary_invalidations=1,
        paging_invalidations=0,
        canary_invalidations=1,
        memory_floor_invalidations=0,
    )
    assert gate_canary["cleared"] is True
    assert gate_canary["canary_invalidations_run_fatal"] is False

    gate_memory = evaluate_pilot_context_gate(
        growth,
        per_task=per_task,
        memory_or_canary_invalidations=1,
        paging_invalidations=0,
        canary_invalidations=0,
        memory_floor_invalidations=1,
    )
    assert gate_memory["cleared"] is False


def test_timed_block_invalidation_split_separates_paging() -> None:
    per_task = [
        {
            "paging_admissible": True,
            "machine_measurement_invalid_reasons": [],
        },
        {
            "paging_admissible": False,
            "machine_measurement_invalid_reasons": ["sustained hard page reads above threshold"],
        },
        {
            "paging_admissible": True,
            "machine_measurement_invalid_reasons": ["compute canary settled drift 0.2 > 0.15"],
        },
    ]
    split = timed_block_invalidation_split(per_task)
    assert split["paging_invalidations_inside_timed_blocks"] == 1
    assert split["non_paging_invalidations_inside_timed_blocks"] == 1
    assert split_invalidation_reasons(["sustained hard page reads", "compute canary drift"])[
        "paging"
    ]


def test_analysis_all_vs_admissible_split() -> None:
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
        run_id="test",
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
                {"task_id": "T0", "paging_admissible": True, "jct_s": 1.0},
                {
                    "task_id": "T1",
                    "paging_admissible": False,
                    "paging_admissible_reasons": ["page reads"],
                    "jct_s": 10.0,
                },
            ]
        },
        integrity_verified=True,
        paging_admissible_by_task={"T0": True, "T1": False},
        paging_admissible_reasons_by_task={"T1": ["page reads"]},
    )
    frac = inputs.admissible_fraction()
    assert frac["n_paging_admissible_tasks"] == 1
    assert frac["paging_admissible_task_fraction"] == pytest.approx(0.5)

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
    assert "paging_admissible_subset" in point["unfiltered_envelope"]
    # Timing excludes the inadmissible T1 wall=10s.
    assert point["p95_actual_wall_s_surviving"] == pytest.approx(1.0)
    assert point["p95_actual_wall_s_surviving_meta"]["basis"] == "timing_admissible_only"
    assert point["max_actual_wall_s_surviving"] == pytest.approx(1.0)
    # Envelope still sees both tasks' KV.
    assert point["unfiltered_envelope"]["peak_kv_bytes_resident"]["per_task_max"] == pytest.approx(
        2000.0
    )

    meta = _with_admissible_basis(1.0, fraction=frac, basis="timing_admissible_only")
    assert meta["paging_admissible_task_fraction"] == pytest.approx(0.5)


def test_c9_note_cites_three_write_path_deaths() -> None:
    note = json.loads(
        (_ROOT / "derived" / "efilter" / "c9_output_path_defects_note.json").read_text(
            encoding="utf-8"
        )
    )
    ids = note["citing_run_ids"]
    assert "cb0ed2e3-ed70-4627-b231-51016d2b255b" in ids
    assert "a161f89e-fcc2-4cb8-9b04-c215358a2f88" in ids
    assert "e66701aa-37b9-4b00-b86b-7ffb477f4e66" in ids
    assert note["outputs_tasks_decision"]["wrong_side"] == "schema"


def test_efilter_yaml_pins_c2e() -> None:
    cfg = yaml.safe_load((_ROOT / "configs" / "efilter.yaml").read_text(encoding="utf-8"))
    assert cfg.get("c2e_amendment") == "docs/CURSOR_PROMPT_C2e.md"
