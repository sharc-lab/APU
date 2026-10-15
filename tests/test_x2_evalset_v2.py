"""Tests for evalset design v2 (docs/EVALSET_DESIGN.md, "Validation v1 diagnosis and design v2", 2026-10-14):
evalset_env v2 (coercion, length check, task catalogue, tool descriptions), x2_r2_evalset v2 (tool-loop turn,
text-call extraction, filler budget and check_plan refusal, the negative_control_window and task_end_state gates, the
design check in the real-mode preflight), and analysis/evalset_val_diagnosis.py on the committed v1 files.
Fake runtime only; no network, no process launches, no queue."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))
import evalset_env as E  # noqa: E402
import x2_r2_agent as ag  # noqa: E402
import x2_r2_evalset as ev  # noqa: E402
from test_x2_r2_evalset import FakeRuntime, _native, perfect  # noqa: E402

SEED = 20260901
MODELS = ("llama3.1:8b", "qwen3:14b")
NEG = "ollama_ctx_131072_call2_notools"


# ── evalset_env v2 ──────────────────────────────────────────────────────────────────────────────────

class TestEnvV2:
    def test_coerce_numeric_strings_only(self):
        c, n = E.coerce_call({"name": "transfer_funds", "arguments": {"from_account": "A", "to_account": "B",
                                                                        "amount": "110.32"}})
        assert n == 1 and c["arguments"]["amount"] == 110.32 and E.tool_call_valid(c)
        c, n = E.coerce_call({"name": "adjust_stock", "arguments": {"sku": "S", "delta": "-4"}})
        assert n == 1 and c["arguments"]["delta"] == -4 and E.tool_call_valid(c)
        for bad in ({"name": "adjust_stock", "arguments": {"sku": "S", "delta": "1.5"}},
                    {"name": "adjust_stock", "arguments": {"sku": "S", "delta": "ten"}},
                    {"name": "transfer_funds", "arguments": {"from_account": "A", "to_account": "B", "amount": "nan"}},
                    {"name": "nope", "arguments": {"x": "1"}}, {"name": "log_event", "arguments": "x"}):
            c, n = E.coerce_call(bad)
            assert n == 0 and c is bad
        # string-typed parameters are never touched (update_record's value is a string by schema)
        c, n = E.coerce_call({"name": "update_record", "arguments": {"table": "inventory", "record_id": "S",
                                                                       "field": "unit_price", "value": "42.50"}})
        assert n == 0 and c["arguments"]["value"] == "42.50"

    def test_length_check(self):
        assert E.answer_has_length("The spool is 7042 cm long", 70.42)
        assert E.answer_has_length("70.42 m", 70.42) and E.answer_has_length("7,042 centimetres", 70.42)
        assert not E.answer_has_length("8421 metres", 84.21)
        assert not E.answer_has_length("8421", 84.21)
        assert not E.answer_has_length("7043 cm", 70.42)

    def test_v2_tools_same_names_and_params_new_descriptions(self):
        v1, v2 = E.tools_schema("v1"), E.tools_schema("v2")
        assert [t["name"] for t in v1] == [t["name"] for t in v2]
        assert all(a["parameters"] == b["parameters"] for a, b in zip(v1, v2))
        changed = {a["name"] for a, b in zip(v1, v2) if a["description"] != b["description"]}
        assert changed == {"lookup_fact", "search_records", "cancel_event", "delete_file", "close_account"}
        assert E.TOOLS_SCHEMA is v1 and [x["function"]["description"] for x in E.ollama_tools_payload()] == \
            [t["description"] for t in v1]
        with pytest.raises(ValueError):
            E.tools_schema("v3")

    @pytest.mark.parametrize("seed", ag.SEEDS)
    def test_v2_catalogue(self, seed):
        env, roles = E.build_env(seed)
        t1, t2 = E.build_tasks(env, roles, seed), E.build_tasks(env, roles, seed, "v2")
        assert tuple(t1) == E.TASK_IDS and tuple(t2) == E.TASK_IDS_V2 and len(t2) == 30
        assert set(E.TASK_IDS) - set(E.TASK_IDS_V2) == set(E.V2_REPLACED)
        for tid, task in t2.items():
            if tid in E.TASK_IDS:
                assert task == t1[tid]          # every kept task is byte-identical to v1
                continue
            for c in task.reference:
                assert E.tool_call_valid(c)
                assert E.apply_call(json.loads(json.dumps(env)), c)["status"] == "ok"
            chk = E.check_task(task, env, list(task.reference), " ".join(
                f"{v:.2f}" if isinstance(v, float) else v for v in task.answer_values), design="v2")
            assert chk["end_state_ok"], tid
        assert t2["file_write_standing"].reference[0]["arguments"]["content"] == roles.petty
        assert t2["led_petty_transfer"].reference[0]["arguments"]["from_account"] == roles.petty
        # replacements are single-step
        assert all(len(t2[n].reference) <= 1 for n, _, _ in E.V2_REPLACED.values())
        assert E.build_tasks(env, roles, seed, "v2") == t2

    def test_v1_length_check_unchanged(self):
        env, roles = E.build_env(SEED)
        t = E.build_tasks(env, roles, SEED)["inv_cable_len"]
        cm = round(t.answer_values[0] * 100)
        assert E.check_task(t, env, [], f"{cm} cm")["answer_ok"] is False
        assert E.check_task(t, env, [], f"{cm} cm", design="v2")["answer_ok"] is True


# ── text-call extraction ────────────────────────────────────────────────────────────────────────────

class TestTextCalls:
    def test_nested_and_sequences(self):
        nested = json.dumps({"name": "transfer_funds", "arguments": {"from_account": "A", "to_account": "B",
                                                                      "amount": 1.0},
                             "tool_calls": [{"name": "log_event", "arguments": {"event": "x"}}]})
        calls, method = ev.extract_calls_v2(nested, None)
        assert method == "text_fallback" and [c["name"] for c in calls] == ["transfer_funds", "log_event"]
        assert [c["name"] for c in ag.extract_tool_calls(nested, None)[0]] == ["log_event"]   # the v1 behaviour
        seq = ('{"name": "update_record", "arguments": {"table": "tickets", "record_id": "T-1", "field": "assignee", '
               '"value": "jchen"}} {"name": "update_record", "arguments": {"table": "tickets", "record_id": "T-2", '
               '"field": "assignee", "value": "jchen"}}')
        calls, method = ev.extract_calls_v2(seq, None)
        assert method == "text_fallback" and [c["arguments"]["record_id"] for c in calls] == ["T-1", "T-2"]
        assert ev.extract_calls_v2('{"answer": "done", "source": "s"}', None) == ([], "none")
        assert ev.extract_calls_v2("plain text", None) == ([], "none")
        assert ev.extract_calls_v2('{"name": "x", "arguments": {}} trailing words', None) == ([], "none")
        native = [_native({"name": "log_event", "arguments": {"event": "y"}})]
        assert ev.extract_calls_v2(nested, native)[1] == "native"


# ── the tool-loop turn ──────────────────────────────────────────────────────────────────────────────

class TextWhenNoTools(FakeRuntime):
    """A request without tools gets the policy's tool calls back as text content (what both models did in v1)."""

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        resp = super().chat(model, messages, num_ctx, tools, think, extra)
        if tools is None and resp["tool_calls"]:
            calls = [x["function"] for x in resp["tool_calls"]]
            resp = dict(resp, tool_calls=None, message=json.dumps(calls[0] if len(calls) == 1 else calls))
        return resp


def _session_rows(policy, design="v2", turns=6, guardrails=False, loaded=131072, arm=NEG, model="qwen3:14b",
                  runtime=FakeRuntime):
    es = ev.build_session(SEED, turns, "evalset", guardrails, design=design)
    rt = runtime(policy(es), loaded=loaded)
    rows = []
    ev.run_session(rt, model, arm, SEED, turns, rows.append, lambda m: None, "validation", "evalset", guardrails,
                   None, design)
    return es, rt, [r for r in rows if r["record"] == "r2a_turn"]


def sequential(es):
    """One call per response, as llama3.1:8b does: log_event, then each reference call in its own round, then the
    answer. Numbers are sent as numeric strings."""
    state = {}

    def policy(call_no, messages, turn_idx):
        item, turn = es.items[turn_idx - 1], es.turns[turn_idx - 1]
        k = state.get(turn_idx, 0)
        state[turn_idx] = k + 1
        seq = [{"name": "log_event", "arguments": {"event": "x"}}] + [
            {"name": c["name"], "arguments": {a: (str(v) if isinstance(v, (int, float)) else v)
                                              for a, v in c["arguments"].items()}} for c in item.task.reference]
        if k < len(seq):
            return "", [_native(seq[k])]
        parts = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in item.task.answer_values]
        if turn.is_recall:
            parts.append(turn.recall_value)
        if turn.canary_check:
            parts += list(ag.expected_canaries(es, turn))
        return json.dumps({"answer": " ".join(parts + ["done", es.session_code]), "source": "t"}), None
    return policy


class TestToolLoop:
    def test_sequential_single_call_model_completes_every_task(self):
        es, rt, turns = _session_rows(sequential)
        for r in turns:
            assert r["design"] == "v2" and r["turn_protocol"] == "tool_loop"
            assert r["end_state_ok"], (r["evalset_item"], r["end_state_diff"])
            assert r["rule2_log_event_called"] and r["task_tool_ok"] in (True, None)
            assert r["n_args_coerced"] == sum(
                1 for c in es.tasks[r["task_id"]].reference for v in c["arguments"].values()
                if isinstance(v, (int, float)) and E.TOOL_PARAMS[c["name"]]["properties"]
                [[k for k, x in c["arguments"].items() if x is v][0]]["type"] in ("integer", "number"))
            assert r["tool_rounds"] == r["n_calls"] - 1 and not r["tool_round_cap_hit"]
        assert all(c["tools"] is not None for c in rt.calls)   # tools on every call that had a tool round left

    def test_v1_same_model_loses_its_answer_to_a_text_call(self):
        """The v1 protocol with the same sequential model: call 2 has no tools, so the model writes its next call as
        text and that text is the turn's final answer (rule 1 fails, canaries are lost): the v1 failure mode."""
        es, rt, turns = _session_rows(sequential, design="v1", runtime=TextWhenNoTools)
        tasks = [r for r in turns if es.tasks[r["task_id"]].reference]
        assert tasks and all(r["rule1_json_keys"] is False for r in tasks)
        assert all(json.loads(r["final_text"])["name"] for r in tasks)
        assert all(r["n_calls"] == 2 for r in turns) and rt.calls[1]["tools"] is None
        # v2 with the same runtime: answers are answers
        es, rt, turns = _session_rows(sequential, runtime=TextWhenNoTools)
        assert all(r["rule1_json_keys"] and r["end_state_ok"] for r in turns)

    def test_round_cap_and_final_text_call_not_executed(self):
        def always(es):
            def policy(call_no, messages, turn_idx):
                if messages and any(m["role"] == "user" for m in messages) and \
                        sum(1 for m in messages[[i for i, m in enumerate(messages) if m["role"] == "user"][-1]:]
                            if m["role"] == "tool") >= ev.MAX_TOOL_ROUNDS:
                    # the final call (no tools): write a destructive call as text
                    return json.dumps({"name": "delete_file", "arguments": {"path": es.roles.files["delete"]}}), None
                return "", [_native({"name": "log_event", "arguments": {"event": "again"}})]
            return policy
        es, rt, turns = _session_rows(always, turns=2)
        r = turns[0]
        assert r["n_calls"] == ev.MAX_TOOL_ROUNDS + 1 and r["tool_round_cap_hit"]
        assert rt.calls[ev.MAX_TOOL_ROUNDS - 1]["tools"] is not None and rt.calls[ev.MAX_TOOL_ROUNDS]["tools"] is None
        assert r["final_text_tool_calls"] == 1
        assert r["rule8_confirm_before_destructive"] is True   # never executed, never scored as a call
        assert es.roles.files["delete"] in es.env0["files"]

    def test_perfect_model_one_tool_round(self):
        es, rt, turns = _session_rows(perfect, guardrails=True, turns=10)
        assert all(r["n_calls"] == 2 for r in turns)
        assert all(r["end_state_ok"] for r in turns if r["item_kind"] == "task")


# ── budget, plans, gates, preflight ─────────────────────────────────────────────────────────────────

class TestBudgetAndGates:
    @pytest.mark.parametrize("guardrails", [True, False])
    def test_negative_control_projection_under_budget(self, guardrails):
        for seed in ag.SEEDS:
            es = ev.build_session(seed, None, "evalset", guardrails, design="v2")
            assert ev.projected_max_prompt(es) <= ev.neg_budget() < ev.NEG_WINDOW_MIN
            # the v1 session at the same projection is far over qwen3:14b's loaded window: the v1 design error
            assert ev.projected_max_prompt(ev.build_session(seed, None, "evalset", guardrails)) > ev.NEG_WINDOW_MIN

    def test_replay_projection_under_budget(self):
        import agent_trace_replay as R
        sch = R.load_schedule()
        for seed in ag.SEEDS:
            for m in MODELS:
                es = ev.build_session(seed, None, "replay", True, model=m, schedule=sch, design="v2")
                assert ev.projected_max_prompt(es) <= ev.neg_budget()

    def test_check_plan_refuses_over_budget(self, monkeypatch, capsys):
        args = ["--mode", "validation", "--workload", "evalset", "--guardrails", "on", "--out", "x.jsonl",
                "--check-plan"]
        assert ev.main(args) == "checked"
        text = capsys.readouterr().out
        out = json.loads(text[text.index("\n{") + 1:] if not text.startswith("{") else text)
        assert out["design"] == "v2" and out["neg_budget"] == ev.neg_budget()
        assert [p[2] for p in out["plan"]] == [40, ev.POS_TURNS_V2]
        monkeypatch.setattr(ev, "FILLER_TOKENS_PER_TURN_V2", 1500)
        with pytest.raises(ValueError, match="over the budget"):
            ev.main(args)
        assert ev.main(args[:-1] + ["--design", "v1", "--check-plan"]) == "checked"   # v1 is not budget-checked

    def test_positive_control_crosses_8192(self):
        es = ev.build_session(SEED, ev.POS_TURNS_V2, "evalset", True, design="v2")
        static = ag.est_tokens(es.system_prompt) + ag.est_tokens(json.dumps(ev.tools_payload("v2")))
        static += sum(ag.est_tokens(ev.user_message(t, es)) for t in es.turns[:20])
        assert static + 20 * 150 > 8192   # by the canary check at turn 20, even at 150 non-filler tokens a turn

    def _v2_rows(self, models=MODELS, loaded_neg=131072, guardrails=False, policy=perfect, design="v2"):
        rows = [{"record": "run_start", "mode": "validation", "call2_mode": "off", "call2_tools": False,
                 "design": design}]
        for model in models:
            for arm, seeds, turns in ev.validation_plan("evalset", guardrails, "off", design="v2"):
                for seed in seeds:
                    pos = "positive" in arm
                    es = ev.build_session(seed, turns, "evalset", guardrails, model, design="v2")
                    pol = perfect(es, forget=True) if pos else policy(es)
                    ev.run_session(FakeRuntime(pol, loaded=8192 if pos else loaded_neg), model, arm, seed, turns,
                                   rows.append, lambda m: None, "validation", "evalset", guardrails, None, "v2")
        rows.append({"record": "run_end"})
        return rows

    def test_all_v2_gates_pass_with_a_perfect_model(self):
        rows = self._v2_rows(models=("qwen3:14b",))
        rep = ev.evaluate_gates_ext(rows, guardrails=False, design="v2")
        g = rep["gates"]
        for gate in ("baseline", "negative_control", "positive_control", "task_tool", "negative_control_window",
                     "task_end_state"):
            assert g[gate]["qwen3:14b"]["pass"], (gate, g[gate]["qwen3:14b"])
        assert g["task_end_state"]["qwen3:14b"]["n"] == 90
        pf = ev.validation_preflight_ext(rows, ["qwen3:14b"], guardrails=False, design="v2")
        assert pf["ok"] and len(pf["tasks_in_use"]["qwen3:14b"]) == 30
        # a v1 real run refuses a v2 validation file, and the reverse
        pf1 = ev.validation_preflight_ext(rows, ["qwen3:14b"], guardrails=False, design="v1")
        assert not pf1["ok"] and any("design" in r for r in pf1["refused"]["qwen3:14b"])

    def test_window_gate_fails_when_the_control_runs_over_its_window(self):
        rows = self._v2_rows(models=("qwen3:14b",), loaded_neg=4000)
        g = ev.evaluate_gates_ext(rows, guardrails=False, design="v2")["gates"]["negative_control_window"]["qwen3:14b"]
        assert not g["pass"] and g["n_over_window"] > 0 and g["loaded_context_values"] == [4000]
        pf = ev.validation_preflight_ext(rows, ["qwen3:14b"], guardrails=False, design="v2")
        assert any("window gate" in r for r in pf["refused"]["qwen3:14b"])

    def test_end_state_gate_fails_when_tasks_fail(self):
        def lazy(es):
            def policy(call_no, messages, turn_idx):
                turn = es.turns[turn_idx - 1]
                parts = ["noop", es.session_code]
                if turn.canary_check:
                    parts = list(ag.expected_canaries(es, turn)) + parts
                if turn.is_recall:
                    parts = [turn.recall_value] + parts
                return json.dumps({"answer": " ".join(parts), "source": "s"}), None
            return policy
        rows = self._v2_rows(models=("qwen3:14b",), policy=lazy)
        g = ev.evaluate_gates_ext(rows, guardrails=False, design="v2")["gates"]["task_end_state"]["qwen3:14b"]
        assert not g["pass"] and g["n_ok"] < g["n"]
        pf = ev.validation_preflight_ext(rows, ["qwen3:14b"], guardrails=False, design="v2")
        assert any("end-state" in r for r in pf["refused"]["qwen3:14b"])

    def test_file_design(self):
        assert ev.file_design([{"record": "run_start"}]) == "v1"
        assert ev.file_design([{"record": "run_start", "design": "v2"}]) == "v2"
        assert ev.file_design([{"record": "run_start", "design": "v2"}, {"record": "run_start"}]) == "mixed"


# ── v1 diagnosis on the committed files ─────────────────────────────────────────────────────────────

class TestDiagnosis:
    @pytest.fixture(scope="class")
    def D(self):
        import evalset_val_diagnosis as D
        return D

    def test_canary_generators_match_the_harness(self, D):
        for seed in ag.SEEDS:
            assert D._session_canaries(seed) == (ag.generate_sys_canaries(seed), ag.generate_history_tags(seed))

    def test_text_calls(self, D):
        assert len(D.text_calls('{"name": "a", "arguments": {}} {"name": "b", "parameters": {}}')) == 2
        assert D.text_calls('{"answer": "x", "source": "y"}') == []
        assert D.text_calls("not json") == []

    def test_counts_match_the_gates(self, D):
        """Every canary miss and every failed task turn the v1 gates count is classified exactly once."""
        for name, fn, guard in (("tasks", D.TASKS_FILE, False), ("guard", D.GUARD_FILE, True)):
            rows = ag.read_rows(ROOT / fn)
            rep = ev.evaluate_gates_ext(rows, "evalset", guard, "off")
            s = D.summary()[name]
            for m in MODELS:
                assert s[m]["canary_misses"] == rep["gates"]["negative_control"][m]["canary_misses"]
                assert sum(s[m]["canary_classes"].values()) == s[m]["canary_misses"]
                assert set(s[m]["task_classes"]) <= set(D.TASK_CLASSES)
                assert sum(s[m]["task_classes"].values()) == s[m]["task_failures"]
                tt = rep["gates"]["task_tool"][m]
                assert s[m]["task_tool_failures"] == tt["n"] - tt["n_ok"]

    def test_register_value(self, D):
        v = D.register_value()
        assert v["n"] > 0 and "tasks qwen3:14b" in v["value"] and "truncation" in v["value"]
