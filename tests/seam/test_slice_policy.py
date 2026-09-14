"""Pre-registered routing policy (AMENDMENTS.md AM-022) and the isolation invariant.

The isolation tests matter more than they look. If the two arms differ in anything besides
throughput, the experiment stops measuring silicon and starts measuring whatever else drifted -
and the resulting number is indistinguishable from a real result by inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seam.agent.policy import ArmConfig, ThroughputModel, assert_isolated, decide
from seam.agent.tools import TOOL_SPECS, load_workload, normalize_answer
from seam.errors import IsolationViolationError

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO_ROOT / "configs" / "tasks" / "bfcl_slice_v1.json"

FAST = ThroughputModel(
    target="cpu-p", r_prefill_tok_s=200.0, r_decode_tok_s=20.0, measured_by_run_id="test"
)
SLOW = ThroughputModel(
    target="cpu-lpe", r_prefill_tok_s=100.0, r_decode_tok_s=10.0, measured_by_run_id="test"
)
N_OUT = {"tool_call_synthesis": 100, "answer_synthesis": 20}


# --------------------------------------------------------------------------------------------
# The predictor
# --------------------------------------------------------------------------------------------


def test_prediction_includes_prefill() -> None:
    """Prefill is real local work and must be priced.

    Omitting it would make the escalation rate independent of transcript length, erasing the
    within-trajectory growth the slice deliberately wants to observe.
    """
    decision = decide(
        throughput=FAST,
        deadline_s=100.0,
        prompt_tokens=2000,
        step_type="tool_call_synthesis",
        n_out_pred_tokens=N_OUT,
    )
    # 2000/200 prefill + 100/20 decode = 10 + 5
    assert decision.t_pred_s == pytest.approx(15.0)


def test_slower_target_escalates_at_a_deadline_the_faster_one_clears() -> None:
    """The causal chain under test, in one assertion."""
    kwargs = {
        "deadline_s": 8.0,
        "prompt_tokens": 400,
        "step_type": "tool_call_synthesis",
        "n_out_pred_tokens": N_OUT,
    }
    fast = decide(throughput=FAST, **kwargs)  # type: ignore[arg-type]
    slow = decide(throughput=SLOW, **kwargs)  # type: ignore[arg-type]
    assert fast.assigned_target == "local"
    assert slow.assigned_target == "cloud"


def test_longer_transcripts_escalate_more_often_on_the_same_target() -> None:
    short = decide(
        throughput=FAST,
        deadline_s=6.0,
        prompt_tokens=100,
        step_type="tool_call_synthesis",
        n_out_pred_tokens=N_OUT,
    )
    long = decide(
        throughput=FAST,
        deadline_s=6.0,
        prompt_tokens=4000,
        step_type="tool_call_synthesis",
        n_out_pred_tokens=N_OUT,
    )
    assert short.assigned_target == "local"
    assert long.assigned_target == "cloud"


def test_pre_registered_horizontal_rescaling_holds_for_the_predictor() -> None:
    """AM-022 item 4: the two curves should be one curve, rescaled by the throughput ratio.

    Exact here because prefill and decode are scaled by the same factor. In the real data they
    need not be, and the analysis reports it when they are not.
    """
    ratio = FAST.r_decode_tok_s / SLOW.r_decode_tok_s
    assert FAST.r_prefill_tok_s / SLOW.r_prefill_tok_s == pytest.approx(ratio)

    for deadline in (2.0, 5.0, 12.0):
        fast = decide(
            throughput=FAST,
            deadline_s=deadline,
            prompt_tokens=600,
            step_type="tool_call_synthesis",
            n_out_pred_tokens=N_OUT,
        )
        slow = decide(
            throughput=SLOW,
            deadline_s=deadline * ratio,
            prompt_tokens=600,
            step_type="tool_call_synthesis",
            n_out_pred_tokens=N_OUT,
        )
        assert fast.assigned_target == slow.assigned_target


def test_zero_throughput_refuses_rather_than_escalating_everything() -> None:
    """A zero would make every step escalate, which looks exactly like a real silicon effect."""
    broken = ThroughputModel(
        target="cpu-lpe", r_prefill_tok_s=0.0, r_decode_tok_s=10.0, measured_by_run_id="test"
    )
    with pytest.raises(ValueError, match="non-positive throughput"):
        decide(
            throughput=broken,
            deadline_s=1.0,
            prompt_tokens=10,
            step_type="tool_call_synthesis",
            n_out_pred_tokens=N_OUT,
        )


def test_decision_cost_is_recorded() -> None:
    """H3's routing-amortization bound needs this baseline."""
    decision = decide(
        throughput=FAST,
        deadline_s=10.0,
        prompt_tokens=100,
        step_type="tool_call_synthesis",
        n_out_pred_tokens=N_OUT,
    )
    assert decision.decision_ns > 0
    assert "decision_ns" in decision.to_record()


# --------------------------------------------------------------------------------------------
# Isolation invariant
# --------------------------------------------------------------------------------------------


def _arm(target: str, throughput: ThroughputModel, **overrides: object) -> ArmConfig:
    fields: dict[str, object] = {
        "task_ids": ["T01", "T02"],
        "seed": 20260802,
        "system_prompt_sha256": "aaa",
        "tool_specs_sha256": "bbb",
        "task_list_sha256": "ccc",
        "n_out_pred_tokens": N_OUT,
        "deadline_s": 5.0,
        "max_steps": 8,
        "max_tokens": 512,
        "temperature": 0.0,
        "cloud_model_id": "claude-sonnet-5",
        "local_model_ref": "qwen3-4b-int4",
        "confinement_mechanism": "process_affinity",
        "reasoning_mode": "thinking_off",
    }
    fields.update(overrides)
    return ArmConfig(target=target, throughput=throughput, fields=fields)


def test_arms_differing_only_in_throughput_are_isolated() -> None:
    assert_isolated(_arm("cpu-p", FAST), _arm("cpu-lpe", SLOW))


def test_differing_deadline_between_arms_is_caught() -> None:
    with pytest.raises(IsolationViolationError, match="deadline_s"):
        assert_isolated(_arm("cpu-p", FAST), _arm("cpu-lpe", SLOW, deadline_s=9.0))


def test_differing_task_set_between_arms_is_caught() -> None:
    with pytest.raises(IsolationViolationError, match="task_ids"):
        assert_isolated(_arm("cpu-p", FAST), _arm("cpu-lpe", SLOW, task_ids=["T01"]))


def test_differing_n_out_pred_between_arms_is_caught() -> None:
    """n_out_pred must be frozen and shared; a per-arm value would be an oracle."""
    with pytest.raises(IsolationViolationError, match="n_out_pred_tokens"):
        assert_isolated(
            _arm("cpu-p", FAST),
            _arm("cpu-lpe", SLOW, n_out_pred_tokens={"tool_call_synthesis": 50}),
        )


def test_differing_prompt_between_arms_is_caught() -> None:
    """Per-model prompt tailoring invalidates H1 (spec §7 M3.1)."""
    with pytest.raises(IsolationViolationError, match="system_prompt_sha256"):
        assert_isolated(_arm("cpu-p", FAST), _arm("cpu-lpe", SLOW, system_prompt_sha256="zzz"))


def test_aa_control_with_identical_targets_passes_isolation() -> None:
    """The A/A control runs one target twice under two labels; that must not itself trip."""
    assert_isolated(_arm("cpu-p", FAST), _arm("cpu-p", FAST))


def test_mixed_confinement_mechanism_between_arms_is_caught() -> None:
    """Symmetry is binding.

    Confining cpu-p by process affinity while confining cpu-lpe by SCHEDULING_CORE_TYPE would make
    the arms differ in TBB pool construction and thread placement as well as in silicon, so any
    measured difference would no longer be attributable to the processor cluster.
    """
    with pytest.raises(IsolationViolationError, match="confinement_mechanism"):
        assert_isolated(
            _arm("cpu-p", FAST, confinement_mechanism="process_affinity"),
            _arm("cpu-lpe", SLOW, confinement_mechanism="openvino_scheduling_core_type"),
        )


def test_mismatched_reasoning_mode_between_arms_is_caught() -> None:
    """AM-024: reasoning mode is matched per arm, never inherited from a default on one side."""
    with pytest.raises(IsolationViolationError, match="reasoning_mode"):
        assert_isolated(
            _arm("cpu-p", FAST, reasoning_mode="thinking_off"),
            _arm("cpu-lpe", SLOW, reasoning_mode="thinking_on"),
        )


# --------------------------------------------------------------------------------------------
# Frozen workload
# --------------------------------------------------------------------------------------------


def test_task_list_is_loadable_and_has_at_least_twenty_tasks() -> None:
    workload = load_workload(TASKS_PATH)
    assert len(workload.tasks) >= 20
    assert len({t.task_id for t in workload.tasks}) == len(workload.tasks)


def test_every_task_is_solvable_in_the_frozen_world() -> None:
    """A task whose answer the tools cannot produce would score as a model failure forever."""
    workload = load_workload(TASKS_PATH)
    for task in workload.tasks:
        assert task.expected.strip(), f"{task.task_id} has no expected answer"
        assert task.min_tool_calls >= 1


def test_tool_specs_are_shared_and_include_a_terminal_action() -> None:
    names = {t.name for t in TOOL_SPECS}
    assert "submit_answer" in names
    assert "calculator" in names


def test_calculator_refuses_non_arithmetic_input() -> None:
    """Model output reaching eval() would be a remote-code-execution hole."""
    workload = load_workload(TASKS_PATH)
    text, error = workload.world.execute("calculator", {"expression": "__import__('os')"})
    assert error is None
    assert text.startswith("error:")


def test_calculator_evaluates_arithmetic() -> None:
    workload = load_workload(TASKS_PATH)
    text, error = workload.world.execute("calculator", {"expression": "1200 + 1850 + 2300 + 1400"})
    assert error is None
    assert text == "6750"


def test_answer_normalization_is_narrow() -> None:
    """Generous grading would score the local model's failures as successes."""
    assert normalize_answer(" 6,750. ") == "6750"
    assert normalize_answer("Portland") == "portland"
    assert normalize_answer("14.0") == "14"
    # Semantically close but wrong stays wrong.
    assert normalize_answer("about 6750") != "6750"


def test_expected_answers_match_what_the_tools_actually_return() -> None:
    """Guards against a typo in the frozen list silently capping success at less than 100%."""
    raw = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    workload = load_workload(TASKS_PATH)
    by_id = {t.task_id: t for t in workload.tasks}

    listing, _ = workload.world.execute("list_files", {"directory": "/docs"})
    assert normalize_answer(str(len(json.loads(listing)))) == normalize_answer(
        by_id["T01"].expected
    )

    budget, _ = workload.world.execute("read_file", {"path": "/docs/budget.txt"})
    total = sum(int(line.split(":")[1]) for line in budget.splitlines())
    assert normalize_answer(str(total)) == normalize_answer(by_id["T03"].expected)

    employee, _ = workload.world.execute("lookup_employee", {"name": "Dana Whitfield"})
    assert normalize_answer(json.loads(employee)["city"]) == normalize_answer(by_id["T04"].expected)

    weather, _ = workload.world.execute("get_weather", {"city": "Austin"})
    assert normalize_answer(str(json.loads(weather)["temp_c"])) == normalize_answer(
        by_id["T05"].expected
    )

    assert raw["benchmark"] == workload.benchmark
