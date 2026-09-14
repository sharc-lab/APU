"""Offline replay for E-FILTER Stage 1 (:mod:`seam.analysis.efilter`).

The replay produces the study's headline, so the tests here target the ways it could be wrong
while still looking right: replaying a rule that is not the harness's rule, treating an
uninstrumented cache counter as a measurement, and letting a task with no surviving steps enter
the envelope as a zero.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from seam.agent.policy import DEADLINE_DISABLED_S, ThroughputModel
from seam.analysis.efilter import (
    EfilterInputs,
    StepView,
    analyze,
    deadline_grid,
    envelope_by_task,
    load_run,
    proxy_error_regression,
    self_check_replay,
    strict_json_payload,
    survivors,
    tail_latency_replay,
)
from seam.errors import SeamError
from seam.rawstore import create_run_dir, make_writable_for_test

_THROUGHPUT = ThroughputModel(
    target="cpu-p", r_prefill_tok_s=100.0, r_decode_tok_s=10.0, measured_by_run_id="test"
)
_N_OUT = {"tool_call_synthesis": 64, "answer_synthesis": 16}
_KV_PER_TOKEN = 147_456

_REPLAY_CFG: dict[str, Any] = {
    "n_deadlines": 12,
    "grid_quantile_lo": 0.0,
    "grid_quantile_hi": 1.0,
    "grid_pad_factor": 1.25,
    "bootstrap_resamples": 200,
    "bootstrap_seed": 7,
    "materiality_ratio": 1.2,
    "proxy_bias_interval_threshold": 0.15,
    # C2 / AM-032: absolute 8 s headline withdrawn; P1-P3 use the material grid point.
    "p95_step_target_s": None,
    "p95_step_target_status": "withdrawn_AM-032",
}


def _step(
    task_id: str,
    step_idx: int,
    *,
    context: int,
    completion: int = 40,
    step_type: str = "tool_call_synthesis",
    wall_s: float | None = None,
) -> StepView:
    proxy = int(context * 0.9)
    t_pred = _THROUGHPUT.predict_s(prompt_tokens=proxy, n_out_pred=_N_OUT["tool_call_synthesis"])
    return StepView(
        task_id=task_id,
        step_idx=step_idx,
        # The realized label may be rewritten by the harness after the decision; the router's
        # label is what priced the step and is what the replay must consume.
        step_type=step_type,  # type: ignore[arg-type]
        router_step_type="tool_call_synthesis",
        prompt_tokens_proxy=proxy,
        prompt_tokens_native=context,
        context_tokens_total=context,
        prompt_tokens_new=context,
        completion_tokens=completion,
        kv_bytes_resident=(context + completion) * _KV_PER_TOKEN,
        peak_rss_bytes=5_000_000_000 + context,
        cache_instrumented=False,
        cache_evicted=step_idx > 0,
        actual_wall_s=t_pred if wall_s is None else wall_s,
        logged_t_pred_s=t_pred,
        logged_deadline_s=DEADLINE_DISABLED_S,
        logged_target="local",
        logged_t_pred_prefill_s=proxy / _THROUGHPUT.r_prefill_tok_s,
        logged_t_pred_decode_s=(_N_OUT["tool_call_synthesis"] / _THROUGHPUT.r_decode_tok_s),
    )


def _inputs(
    n_tasks: int = 6,
    n_steps: int = 5,
    *,
    wall_divisor: float | None = None,
    context_step: int = 500,
    task_offset: int = 50,
) -> EfilterInputs:
    def _context(task: int, idx: int) -> int:
        return context_step * (idx + 1) + task_offset * task

    steps = [
        _step(
            f"T{task:02d}",
            idx,
            context=_context(task, idx),
            wall_s=(None if wall_divisor is None else _context(task, idx) / wall_divisor),
        )
        for task in range(n_tasks)
        for idx in range(n_steps)
    ]
    return EfilterInputs(
        run_id="test-run",
        steps=steps,
        throughput=_THROUGHPUT,
        n_out_pred_tokens=_N_OUT,
        kv_bytes_per_token=_KV_PER_TOKEN,
        cache_instrumented=False,
        cache_probe={"verdict": "no cross-call prefix reuse; full re-prefill"},
        summary={},
        integrity_verified=True,
    )


# ==================================================================================================
# The rule is the harness's rule
# ==================================================================================================


def test_self_check_passes_on_consistent_inputs() -> None:
    check = self_check_replay(_inputs())
    assert check["passed"] is True
    assert check["n_steps_checked"] == 30


def test_self_check_catches_a_throughput_that_does_not_match_the_run() -> None:
    """The failure this guard exists for: t_pred reconstructed from the wrong rate."""
    base = _inputs()
    wrong = EfilterInputs(
        run_id=base.run_id,
        steps=base.steps,
        throughput=ThroughputModel(
            target="cpu-p", r_prefill_tok_s=50.0, r_decode_tok_s=10.0, measured_by_run_id="wrong"
        ),
        n_out_pred_tokens=base.n_out_pred_tokens,
        kv_bytes_per_token=base.kv_bytes_per_token,
        cache_instrumented=base.cache_instrumented,
        cache_probe=base.cache_probe,
        summary={},
        integrity_verified=True,
    )
    check = self_check_replay(wrong)
    assert check["passed"] is False
    with pytest.raises(SeamError, match="self-check failed"):
        analyze(wrong, replay_cfg=_REPLAY_CFG)


def test_survivors_shrink_monotonically_as_the_deadline_tightens() -> None:
    inputs = _inputs()
    counts = [
        len(survivors(inputs, deadline_s=d)) for d in (0.5, 2.0, 8.0, 60.0, DEADLINE_DISABLED_S)
    ]
    assert counts == sorted(counts)
    assert counts[-1] == len(inputs.steps)
    assert counts[0] < counts[-1]


def test_the_filter_removes_the_large_steps_first() -> None:
    inputs = _inputs()
    kept = survivors(inputs, deadline_s=20.0)
    dropped = [s for s in inputs.steps if s not in kept]
    assert dropped, "the test deadline must actually filter something"
    assert max(s.context_tokens_total for s in kept) < min(s.context_tokens_total for s in dropped)


# ==================================================================================================
# Envelope
# ==================================================================================================


def test_envelope_omits_tasks_with_no_surviving_steps_rather_than_zeroing_them() -> None:
    inputs = _inputs()
    kept = survivors(inputs, deadline_s=6.0)
    envelope = envelope_by_task(kept, deadline_s=6.0, r_prefill_tok_s=100.0)
    assert set(envelope) == {s.task_id for s in kept}
    assert all(block["n_steps"] > 0 for block in envelope.values())
    assert len(envelope) < len(inputs.task_ids)


def test_envelope_peaks_are_the_maxima_of_the_supplied_steps() -> None:
    inputs = _inputs(n_tasks=1, n_steps=3)
    envelope = envelope_by_task(inputs.steps, deadline_s=100.0, r_prefill_tok_s=100.0)["T00"]
    assert envelope["peak_context_tokens"] == max(s.context_tokens_total for s in inputs.steps)
    assert envelope["peak_kv_bytes_resident"] == max(s.kv_bytes_resident for s in inputs.steps)


def test_arithmetic_intensity_is_one_for_a_pure_decode_step() -> None:
    step = _step("T00", 0, context=0, completion=64)
    assert step.arithmetic_intensity() == pytest.approx(1.0)
    prefill_heavy = _step("T00", 0, context=4000, completion=1)
    assert prefill_heavy.arithmetic_intensity() > step.arithmetic_intensity()


def test_deadline_grid_spans_the_observed_t_pred_range() -> None:
    t_preds = [1.0, 2.0, 10.0]
    grid = deadline_grid(t_preds, n=8, quantile_lo=0.0, quantile_hi=1.0, pad_factor=2.0)
    assert len(grid) == 8
    assert grid[0] == pytest.approx(0.5)
    assert grid[-1] == pytest.approx(20.0)
    assert grid == sorted(grid)


# ==================================================================================================
# The result
# ==================================================================================================


def test_analyze_produces_a_monotone_over_provisioning_curve() -> None:
    result = analyze(_inputs(), replay_cfg=_REPLAY_CFG)
    assert result["replay_self_check"]["passed"] is True

    escalation = [point["escalation_rate"] for point in result["curve"]]
    assert escalation == sorted(escalation, reverse=True)
    assert escalation[0] == pytest.approx(1.0)
    assert escalation[-1] == pytest.approx(0.0)

    ratios = [
        point["over_provisioning"]["peak_kv_bytes_resident"]["point"]
        for point in result["curve"]
        if point["n_tasks_with_survivors"] > 0
    ]
    assert all(r >= 1.0 - 1e-9 for r in ratios), "filtering can only lower a peak"
    assert ratios[-1] == pytest.approx(1.0), "an unreachable deadline filters nothing"


def test_result_is_strict_json_and_carries_the_cache_caveat() -> None:
    result = analyze(_inputs(), replay_cfg=_REPLAY_CFG)
    assert result["cache"]["instrumented"] is False
    assert "UPPER BOUNDS" in result["cache"]["consequence"]
    assert result["cache"]["caching_fork_verdict"].startswith("UNDETERMINED")
    # NaN is permitted in the emitted payload (undefined envelopes), but nothing may be infinite.
    text = json.dumps(result, allow_nan=True)
    assert "Infinity" not in text


def test_disabled_deadline_required_decode_rate_is_null_with_reason() -> None:
    result = analyze(_inputs(), replay_cfg=_REPLAY_CFG)
    unfiltered = result["unfiltered_envelope"]
    assert unfiltered["aggregate"]["p95_required_decode_rate_tok_s"] is None
    assert (
        unfiltered["undefined_metrics"]["p95_required_decode_rate_tok_s"]["reason_code"]
        == "deadline_disabled_no_required_decode_rate"
    )
    payload = strict_json_payload(result)
    assert all(
        task["p95_required_decode_rate_tok_s"] is None
        for task in payload["unfiltered_envelope"]["per_task"].values()
    )
    # A real finite deadline still has a numeric rate and its aggregate statistics retain shape.
    finite = result["curve"][-1]["unfiltered_envelope"]["p95_required_decode_rate_tok_s"]
    assert isinstance(finite["point"], float)
    assert finite["point"] > 0


def test_proxy_regression_recovers_a_known_slope() -> None:
    inputs = _inputs()
    regression = proxy_error_regression(inputs.steps)
    # Fixture proxy is exactly 0.9 x native, so native = proxy / 0.9.
    assert regression["slope"] == pytest.approx(1 / 0.9, rel=1e-6)
    assert regression["mean_relative_bias_proxy_vs_native"] == pytest.approx(-0.1, abs=2e-3)


def test_a_biased_proxy_makes_the_boundary_an_interval() -> None:
    result = analyze(_inputs(), replay_cfg=_REPLAY_CFG)
    # The fixture's proxy is 10% low, under the 15% threshold: a line, not an interval.
    assert result["proxy_error"]["boundary_reported_as_interval"] is False
    assert result["native_token_curve"] == []

    strict = {**_REPLAY_CFG, "proxy_bias_interval_threshold": 0.05}
    widened = analyze(_inputs(), replay_cfg=strict)
    assert widened["proxy_error"]["boundary_reported_as_interval"] is True
    assert len(widened["native_token_curve"]) == _REPLAY_CFG["n_deadlines"]


def test_stratification_reports_step_idx_and_the_step_type_limitation() -> None:
    result = analyze(_inputs(), replay_cfg=_REPLAY_CFG)
    strata = result["stratified"]
    assert set(strata["by_step_idx"]) == {"0", "1", "2", "3", "4"}
    assert "hardcoded" in strata["step_type_limitation"]
    assert "P4" in result["predictions"]


# ==================================================================================================
# Loading a sealed run
# ==================================================================================================


def _seal_a_run(tmp_path: Path, *, cache_instrumented: bool) -> tuple[str, Path]:
    run_id = "0000ffff-0000-0000-0000-000000000001"
    (tmp_path / "raw").mkdir(exist_ok=True)
    run_dir = create_run_dir(run_id, repo_root=tmp_path)
    inputs = _inputs(n_tasks=2, n_steps=3)
    lines = []
    for step in inputs.steps:
        lines.append(
            json.dumps(
                {
                    "program_id": f"efilter/full/{step.task_id}",
                    "step_idx": step.step_idx,
                    "step_type": step.step_type,
                    "assigned_target": step.logged_target,
                    "prompt_tokens": step.prompt_tokens_native,
                    "prompt_tokens_proxy": step.prompt_tokens_proxy,
                    "context_tokens_total": step.context_tokens_total,
                    "prompt_tokens_new": step.prompt_tokens_new,
                    "completion_tokens": step.completion_tokens,
                    "kv_bytes_per_token": _KV_PER_TOKEN,
                    "kv_bytes_resident": step.kv_bytes_resident,
                    "peak_rss_bytes": step.peak_rss_bytes,
                    "cache_instrumented": cache_instrumented,
                    "cache_evicted": step.cache_evicted,
                    "actual_wall_s": step.actual_wall_s,
                    "routing": {
                        "t_pred_s": step.logged_t_pred_s,
                        "deadline_s": step.logged_deadline_s,
                    },
                },
                sort_keys=True,
            )
        )
    run_dir.write_text("steps.ndjson", "\n".join(lines) + "\n")
    run_dir.write_json(
        "summary.json",
        {
            "cache_instrumented": cache_instrumented,
            "cache_probe": {"verdict": "probe", "ttft_ratio_second_over_first": 0.98},
            "kv_geometry": {"kv_bytes_per_token": _KV_PER_TOKEN},
            "policy": {"n_out_pred_tokens_used": _N_OUT},
            "throughput": {
                "target": "cpu-p",
                "r_prefill_tok_s": 100.0,
                "r_decode_tok_s": 10.0,
                "measured_by_run_id": "inline_warmup",
            },
        },
    )
    run_dir.seal()
    return run_id, tmp_path


def test_load_run_refuses_an_uninstrumented_cache_without_acknowledgement(tmp_path: Path) -> None:
    run_id, root = _seal_a_run(tmp_path, cache_instrumented=False)
    try:
        with pytest.raises(SeamError, match="STRUCTURAL zero"):
            load_run(run_id, root=root)
        loaded = load_run(run_id, root=root, accept_uninstrumented_cache=True)
        assert loaded.cache_instrumented is False
        assert len(loaded.steps) == 6
        assert loaded.integrity_verified is True
    finally:
        make_writable_for_test(root / "raw")


def test_load_run_refuses_an_unsealed_run(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    run_dir = create_run_dir("0000ffff-0000-0000-0000-000000000002", repo_root=tmp_path)
    run_dir.write_text("steps.ndjson", "")
    with pytest.raises(SeamError, match="not sealed"):
        load_run(run_dir.run_id, root=tmp_path, accept_uninstrumented_cache=True)


def test_load_run_refuses_steps_without_the_kv_constant(tmp_path: Path) -> None:
    run_id = "0000ffff-0000-0000-0000-000000000003"
    (tmp_path / "raw").mkdir()
    run_dir = create_run_dir(run_id, repo_root=tmp_path)
    run_dir.write_text(
        "steps.ndjson",
        json.dumps(
            {
                "program_id": "efilter/full/T00",
                "step_idx": 0,
                "step_type": "tool_call_synthesis",
                "assigned_target": "local",
                "prompt_tokens": 100,
                "prompt_tokens_proxy": 90,
                "context_tokens_total": 100,
                "prompt_tokens_new": 100,
                "completion_tokens": 10,
                "kv_bytes_per_token": None,
                "kv_bytes_resident": None,
                "peak_rss_bytes": None,
                "cache_instrumented": True,
                "cache_evicted": False,
                "actual_wall_s": 1.0,
                "routing": {"t_pred_s": 1.0, "deadline_s": DEADLINE_DISABLED_S},
            }
        )
        + "\n",
    )
    run_dir.write_json(
        "summary.json",
        {
            "cache_instrumented": True,
            "kv_geometry": {"kv_bytes_per_token": _KV_PER_TOKEN},
            "policy": {"n_out_pred_tokens_used": _N_OUT},
            "throughput": {
                "target": "cpu-p",
                "r_prefill_tok_s": 100.0,
                "r_decode_tok_s": 10.0,
                "measured_by_run_id": "x",
            },
        },
    )
    run_dir.seal()
    try:
        with pytest.raises(SeamError, match="kv_bytes_per_token"):
            load_run(run_id, root=tmp_path)
    finally:
        make_writable_for_test(tmp_path / "raw")


def test_headline_is_the_material_grid_deadline_not_a_fixed_wall_clock_target() -> None:
    result = analyze(_inputs(), replay_cfg=_REPLAY_CFG)
    headline = result["headline"]
    assert headline["p95_step_target_status"] == "withdrawn_AM-032"
    assert headline["absolute_wall_clock_claim"] is False
    assert headline["deadline_s"] in result["deadline_grid_s"]
    assert "distribution_units" in headline
    assert result["predictions"]["P6"]["stage1_baseline"]["run_id"].startswith("1a0166b9")


def test_latency_feasibility_records_withdrawn_absolute_target() -> None:
    result = analyze(_inputs(), replay_cfg=_REPLAY_CFG)
    feasibility = result["latency_feasibility"]
    assert feasibility["target_s"] is None
    assert feasibility["target_status"] == "withdrawn_AM-032"
    assert feasibility["target_reachable"] is None
    assert feasibility["absolute_wall_clock_claim"] is False
    p2 = result["predictions"]["P2"]
    assert p2["withdrawn_wall_clock_target"]["status"] == "withdrawn_AM-032"


def test_the_replay_prices_a_terminal_step_with_the_routers_label_not_the_realized_one() -> None:
    """The harness rewrites step_type AFTER deciding; using the realized label reprices the step."""
    terminal = _step("T00", 1, context=1000, step_type="answer_synthesis")
    assert terminal.step_type == "answer_synthesis"
    assert terminal.router_step_type == "tool_call_synthesis"
    inputs = EfilterInputs(
        run_id="terminal",
        steps=[_step("T00", 0, context=500), terminal],
        throughput=_THROUGHPUT,
        n_out_pred_tokens=_N_OUT,
        kv_bytes_per_token=_KV_PER_TOKEN,
        cache_instrumented=False,
        cache_probe={},
        summary={},
        integrity_verified=True,
    )
    # n_out differs 4x between the two labels, so a self-check that passes proves the replay used
    # the router's label: pricing the terminal step as answer_synthesis would move its t_pred.
    assert self_check_replay(inputs)["passed"] is True


def test_self_check_catches_a_t_pred_decomposition_that_does_not_sum() -> None:
    good = _step("T00", 0, context=500)
    broken = StepView(**{**dataclasses.asdict(good), "logged_t_pred_prefill_s": 99.0})
    inputs = EfilterInputs(
        run_id="broken",
        steps=[broken],
        throughput=_THROUGHPUT,
        n_out_pred_tokens=_N_OUT,
        kv_bytes_per_token=_KV_PER_TOKEN,
        cache_instrumented=False,
        cache_probe={},
        summary={},
        integrity_verified=True,
    )
    check = self_check_replay(inputs)
    assert check["passed"] is False
    assert check["n_decomposition_mismatches"] == 1
    # The decision itself still replays: only the decomposition disagrees, and that alone must stop
    # the analysis because the caching fork is read off those two terms.
    assert check["n_mismatches"] == 0
    with pytest.raises(SeamError, match="self-check failed"):
        analyze(inputs, replay_cfg=_REPLAY_CFG)


def test_predictor_decomposition_states_when_prefill_can_dominate() -> None:
    result = analyze(_inputs(), replay_cfg=_REPLAY_CFG)
    decomposition = result["predictor_decomposition"]
    # Crossover context is (R_prefill / R_decode) * n_out: 10 x 64 = 640 tokens for this fixture.
    assert decomposition["r_prefill_over_r_decode"] == pytest.approx(10.0)
    assert decomposition["context_tokens_for_prefill_to_dominate"][
        "tool_call_synthesis"
    ] == pytest.approx(640.0)
    assert decomposition["prefill_dominant_regime_reached"]["tool_call_synthesis"] is True
    assert 0.0 <= decomposition["prefill_share_of_t_pred"]["mean"] <= 1.0


def test_p1_does_not_let_arithmetic_intensity_satisfy_a_resource_claim() -> None:
    # Contexts span the data-derived grid so the material headline has both survivors and
    # escalations and every ratio is defined.
    result = analyze(
        _inputs(n_tasks=4, n_steps=3, context_step=90, task_offset=5), replay_cfg=_REPLAY_CFG
    )
    assert 0 < result["headline"]["n_surviving_steps"] < result["n_steps"]
    p1 = result["predictions"]["P1"]
    assert "mean_arithmetic_intensity" in p1["dimensions_excluded"]
    assert p1["best_dimension"] in p1["dimensions_scanned"]
    # The alternative verdict is stated rather than hidden, so the choice is auditable.
    assert "verdict" in p1["verdict_if_arithmetic_intensity_admitted"]


def test_uninstrumented_cache_labels_every_derived_metric_as_a_bound() -> None:
    labels = analyze(_inputs(), replay_cfg=_REPLAY_CFG)["bound_labels"]
    assert labels["cache_instrumented"] is False
    assert "max_prompt_tokens_new" in labels["upper_bound_metrics"]
    assert "mean_arithmetic_intensity" in labels["upper_bound_metrics"]
    # The primary endpoint is analytic and must NOT be labelled a bound.
    assert "peak_kv_bytes_resident" in labels["not_bounds"]


def test_emitted_payload_is_strict_json_with_no_bare_nan() -> None:
    result = analyze(_inputs(), replay_cfg=_REPLAY_CFG)
    payload = strict_json_payload(result)
    text = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert "NaN" not in text
    assert "Infinity" not in text
    # A strict reader must accept it back.
    assert json.loads(text, parse_constant=lambda token: pytest.fail(f"non-finite {token}"))


def test_prompt_b_tail_replay_reports_quantiles_histogram_and_material_point() -> None:
    result = tail_latency_replay(_inputs(), replay_cfg=_REPLAY_CFG)
    distribution = result["completion_length_distribution"]
    assert distribution["n"] == 30
    assert distribution["median_tokens"] == 40
    assert distribution["p90_tokens"] == 40
    assert distribution["p99_tokens"] == 40
    assert distribution["histogram"]["count_check"] == 30
    assert set(result["replays"]) == {"baseline_median", "p90", "p99"}
    for replay in result["replays"].values():
        assert replay["n_out_pred_tokens"] == 40
        assert replay["best_realized_tail_across_grid"]["unfiltered_p95_actual_wall_s"] > 0
        assert replay["material_deadline"] is not None
        assert (
            replay["material_deadline"]["kv_over_provisioning_ratio"]
            >= _REPLAY_CFG["materiality_ratio"]
        )
