"""B3 v2 for llama3.3:70b at the default window (docs/FINDINGS.md 2026-10-22, a larger local option for the
route-at-start comparison): --plan tasks_v2_llama33_70b with --preflight v2_strict (no waiver, every gate passing,
llama3.3:70b only), its queue entries, and the pre-registered route reading. Committed v2 tasks validation file with the
model id rewritten; no network, no process launches, no queue."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "analysis"))

import x2_queue_fill as qf  # noqa: E402
import x2_r2_evalset as ev  # noqa: E402
from tests.test_x2_evalset_tasks_real_v2_qwen3_8b import rows_as  # noqa: E402

L70 = "llama3.3:70b"
REAL = ["--design", "v2", "--mode", "real", "--workload", "evalset", "--guardrails", "off", "--models", L70,
        "--plan", "tasks_v2_llama33_70b", "--preflight", "v2_strict", "--rules-from", "v.jsonl"]


def test_plan_is_default_window_only_three_seeds_thirty_turns():
    assert ev.STRICT_PLAN_MODELS[ev.TASKS_V2_L70_PLAN] == (L70,)
    plan = ev.tasks_v2_l70_plan()
    assert len(plan) == 1
    arm, seeds, turns = plan[0]
    assert arm.startswith("ollama_default") and list(seeds) == [20260901, 20260902, 20260903] and turns == 30
    assert ev.ag.ARMS[arm]["num_ctx"] is None          # no num_ctx sent at the default
    args = ev.build_arg_parser().parse_args(REAL + ["--out", "o.jsonl"])
    assert ev.plan_for(args, None) == plan


def test_check_plan_three_sessions_of_the_tasks_v2_task_set(tmp_path):
    args = ev.build_arg_parser().parse_args(REAL + ["--out", str(tmp_path / "o.jsonl")])
    c = ev.check_plan(args)
    assert len(c["sessions"]) == 3 and all(s["turns"] == 30 for s in c["sessions"])
    assert all(s["kinds"]["task"] == 30 for s in c["sessions"])


def test_strict_refuses_on_a_failed_gate_and_admits_when_all_pass():
    pf = ev.strict_preflight_v2(rows_as("qwen3:14b", L70), [L70], allowed_models=(L70,))
    assert not pf["ok"] and L70 in pf["refused"] and pf["waived"] == {}
    ok = ev.strict_preflight_v2(rows_as("qwen3:14b", L70, fix_end_state=True), [L70], allowed_models=(L70,))
    assert ok["ok"] and ok["models_ok"] == [L70] and all(ok["strict_gates"][L70].values())


def test_strict_refuses_any_other_model_under_the_llama_plan():
    pf = ev.strict_preflight_v2(rows_as("qwen3:14b", "qwen3:8b", fix_end_state=True), ["qwen3:8b"],
                                allowed_models=ev.STRICT_PLAN_MODELS[ev.TASKS_V2_L70_PLAN])
    assert not pf["ok"]


@pytest.mark.parametrize("extra", [["--tiers", "4096"], ["--seeds", "20260904"]])
def test_plan_fixes_tiers_and_seeds(extra, tmp_path):
    with pytest.raises(SystemExit):
        ev.main(REAL + ["--out", str(tmp_path / "o.jsonl")] + extra, advance=False)


def test_queue_entries_gate_and_command():
    v, r = qf.entry("x2_evalset_tasks_val_v2_llama33_70b"), qf.entry("x2_evalset_tasks_real_v2_llama33_70b")
    assert r["gate"] == {"requires_done": "x2_evalset_tasks_val_v2_llama33_70b"}
    assert v["cmd"][1] == r["cmd"][1] == "evalv2l70\\x2_r2_evalset.py"
    j = " ".join(r["cmd"])
    for part in ("--mode real", "--models llama3.3:70b", "--plan tasks_v2_llama33_70b", "--preflight v2_strict",
                 "x2_evalset_tasks_val_v2_llama33_70b.jsonl"):
        assert part in j
    assert "--tiers" not in j and "--seeds" not in j
    assert "--mode validation" in " ".join(v["cmd"]) and "--models llama3.3:70b" in " ".join(v["cmd"])
    with pytest.raises(ValueError):
        qf.append_entries([], ["x2_evalset_tasks_real_v2_llama33_70b"])


def test_route_reading_is_the_preregistered_rule():
    import numbers_register as nr
    assert nr.l70_route_reading(87, 90, 87, 90)["reading"] == "llama3.3:70b routes local at matched quality"
    r = nr.l70_route_reading(84, 90, 87, 90)
    assert r["routes"] == {0: "cloud", 5: "local", 10: "local"}
    assert r["reading"] == "cloud at matched quality; routes local at tolerance >= 3.3"
    assert nr.l70_route_reading(70, 90, 90, 90)["reading"] == "cloud at every tolerance shown"


def test_real_outcome_validity_on_default_only_shape():
    o = ev.tasks_real_v2_outcome([{"record": "run_end"}], tiers=ev.TASKS_V2_L70_TIERS, seeds=ev.TASKS_V2_L70_SEEDS,
                                 models=ev.TASKS_V2_L70_MODELS)
    d = o["models"][L70]
    assert d["valid"] is False and any("expected [20260901, 20260902, 20260903]" in p for p in d["problems"])
