"""B3 v2 extended to mistral-small3.2:24b (docs/FINDINGS.md 2026-10-22): --plan tasks_v2_mistral_small32 with
--preflight v2_strict (no waiver, every gate passing, mistral only). Committed v2 tasks validation file with the model
id rewritten; no network, no process launches, no queue."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import x2_r2_evalset as ev  # noqa: E402
from tests.test_x2_evalset_tasks_real_v2_qwen3_8b import rows_as  # noqa: E402

MS = "mistral-small3.2:24b"


def test_plan_and_model_list():
    assert ev.STRICT_PLAN_MODELS[ev.TASKS_V2_MS_PLAN] == (MS,)
    args = ev.build_arg_parser().parse_args(["--design", "v2", "--mode", "real", "--workload", "evalset",
                                             "--guardrails", "off", "--models", MS, "--plan", ev.TASKS_V2_MS_PLAN,
                                             "--preflight", "v2_strict", "--rules-from", "v.jsonl", "--out", "o.jsonl"])
    assert ev.plan_for(args, None) == ev.tasks_v2_plan()


def test_strict_refuses_mistral_on_a_failed_gate_and_admits_when_all_pass():
    pf = ev.strict_preflight_v2(rows_as("qwen3:14b", MS), [MS], allowed_models=(MS,))
    assert not pf["ok"] and MS in pf["refused"] and pf["waived"] == {}
    ok = ev.strict_preflight_v2(rows_as("qwen3:14b", MS, fix_end_state=True), [MS], allowed_models=(MS,))
    assert ok["ok"] and ok["models_ok"] == [MS] and all(ok["strict_gates"][MS].values())


def test_strict_refuses_any_other_model_under_the_mistral_plan():
    pf = ev.strict_preflight_v2(rows_as("qwen3:14b", "qwen3:8b", fix_end_state=True), ["qwen3:8b"],
                                allowed_models=(MS,))
    assert not pf["ok"]


@pytest.mark.parametrize("argv", [
    ["--plan", "tasks_v2_mistral_small32"],                       # strict plan without the strict preflight
    ["--preflight", "v2_strict"],                                 # strict preflight without a strict plan
    ["--plan", "tasks_v2_mistral_small32", "--preflight", "v2_strict", "--tiers", "4096"],
])
def test_argument_pairing_is_enforced(argv, tmp_path):
    base = ["--design", "v2", "--mode", "real", "--workload", "evalset", "--guardrails", "off", "--models", MS,
            "--rules-from", str(tmp_path / "v.jsonl"), "--out", str(tmp_path / "o.jsonl")]
    with pytest.raises(SystemExit):
        ev.main(base + argv, advance=False)


def test_check_plan_for_the_mistral_real_run(tmp_path):
    args = ev.build_arg_parser().parse_args(["--design", "v2", "--mode", "real", "--workload", "evalset",
                                             "--guardrails", "off", "--models", MS, "--plan", ev.TASKS_V2_MS_PLAN,
                                             "--preflight", "v2_strict", "--rules-from", "v.jsonl", "--out",
                                             str(tmp_path / "o.jsonl")])
    c = ev.check_plan(args)
    assert len(c["sessions"]) == 25 and all(s["turns"] == 30 for s in c["sessions"])
