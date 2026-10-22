"""B3 v2 extended to qwen3:8b (docs/FINDINGS.md 2026-10-21, "PRE-REGISTRATION: protocol consistency extended to
qwen3:8b"): --plan tasks_v2_qwen3_8b and --preflight v2_strict (no waiver, every gate passing). Uses the committed
v2 tasks validation file with the model id rewritten; no network, no process launches, no queue."""
import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import x2_r2_evalset as ev  # noqa: E402

VAL = REPO / "results" / "x2_evalset_tasks_val_v2.jsonl"
Q8 = "qwen3:8b"


def rows_as(model_from, model_to, fix_end_state=False):
    out = []
    for r in ev.ag.read_rows(VAL):
        r = copy.deepcopy(r)
        if r.get("model_id") == model_from:
            r["model_id"] = model_to
            if fix_end_state and r.get("record") == "r2a_turn" and r.get("item_kind") == "task":
                r["end_state_ok"] = True
        elif r.get("model_id") is not None and r.get("record") in ("r2a_turn", "r2a_session", "heartbeat"):
            continue   # drop the other model so only model_to is in the file
        out.append(r)
    return out


def test_plan_is_the_tasks_v2_plan():
    assert ev.tasks_v2_q8_plan() == ev.tasks_v2_plan()
    assert len(ev.tasks_v2_q8_plan()) == 5
    assert all(len(seeds) == 5 and turns == 30 for _, seeds, turns in ev.tasks_v2_q8_plan())


def test_strict_refuses_a_failed_task_end_state_with_no_waiver():
    # qwen3:14b's real 80/90 under the qwen3:8b id: the waiver must not extend to it
    pf = ev.strict_preflight_v2(rows_as("qwen3:14b", Q8), [Q8])
    assert not pf["ok"] and Q8 in pf["refused"]
    assert any("task end-state gate failed" in r for r in pf["refused"][Q8])
    assert pf["strict_gates"][Q8]["task_end_state"] is False
    assert pf["waived"] == {}


def test_strict_refuses_qwen3_14b_even_at_its_waived_count():
    pf = ev.strict_preflight_v2(ev.ag.read_rows(VAL), ["qwen3:14b"], allowed_models=("qwen3:14b",))
    assert not pf["ok"] and "qwen3:14b" in pf["refused"]


def test_strict_admits_when_every_gate_passes():
    pf = ev.strict_preflight_v2(rows_as("qwen3:14b", Q8, fix_end_state=True), [Q8])
    assert pf["ok"], pf["refused"]
    assert pf["models_ok"] == [Q8]
    assert all(pf["strict_gates"][Q8].values())
    assert pf["rules_in_use"][Q8]


def test_strict_refuses_any_other_model():
    pf = ev.strict_preflight_v2(rows_as("qwen3:14b", "qwen3:14b", fix_end_state=True), ["qwen3:14b"])
    assert not pf["ok"]
    assert any("not in this plan's model list" in r for r in pf["refused"]["qwen3:14b"])


def test_strict_refuses_a_failed_baseline_gate():
    rows = rows_as("llama3.1:8b", Q8, fix_end_state=True)   # llama's baseline and negative control failed
    pf = ev.strict_preflight_v2(rows, [Q8])
    assert not pf["ok"]
    assert pf["strict_gates"][Q8]["baseline"] is False


def test_waiver_table_unchanged():
    assert ev.TE_DEVIATION == {"qwen3:14b": (80, 90)}


@pytest.mark.parametrize("argv", [
    ["--mode", "real", "--workload", "evalset", "--guardrails", "off", "--plan", "tasks_v2_qwen3_8b",
     "--rules-from", "x", "--out", "y"],                                       # plan without v2_strict
    ["--mode", "real", "--workload", "evalset", "--guardrails", "off", "--plan", "tasks_v2",
     "--preflight", "v2_strict", "--rules-from", "x", "--out", "y"],           # v2_strict with the 14b plan
    ["--mode", "real", "--workload", "evalset", "--guardrails", "on", "--plan", "tasks_v2_qwen3_8b",
     "--preflight", "v2_strict", "--rules-from", "x", "--out", "y"],           # guardrails on
    ["--mode", "real", "--workload", "evalset", "--guardrails", "off", "--plan", "tasks_v2_qwen3_8b",
     "--preflight", "v2_strict", "--seeds", "1", "--rules-from", "x", "--out", "y"],
])
def test_bad_argument_combinations_are_errors(argv):
    with pytest.raises(SystemExit):
        ev.main(argv, advance=False)


def test_check_plan_for_the_real_run(capsys):
    ev.main(["--design", "v2", "--mode", "real", "--workload", "evalset", "--guardrails", "off", "--models", Q8,
             "--plan", "tasks_v2_qwen3_8b", "--preflight", "v2_strict", "--rules-from", "x", "--out", "y",
             "--check-plan"], advance=False)
    assert "ollama_ctx_4096_numkeep" in capsys.readouterr().out
