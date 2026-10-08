"""Tests for harness/x2_r2_agent.py (the two-step R2 design, docs/R2_DESIGN.md). Mock runtime only; no network,
no process launches."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import x2_r2_agent as ag  # noqa: E402
import t2s_r2_session_growth as r2  # noqa: E402


SEED = 20260901


def _session(turns=10):
    return ag.build_agent_session(SEED, turns)


def _turn(ags, idx):
    return ags.spec.turns[idx - 1]


def _answer(text, code, **extra):
    return json.dumps({"answer": f"{text} {code}", "source": "tool", **extra})


def _native(name, **args):
    return {"function": {"name": name, "arguments": args}}


# ── session construction ────────────────────────────────────────────────────────────────────────────

class TestSession:
    def test_canaries_deterministic_distinct_and_unique(self):
        assert ag.generate_sys_canaries(1) == ag.generate_sys_canaries(1)
        assert ag.generate_sys_canaries(1) != ag.generate_sys_canaries(2)
        allc = ag.generate_sys_canaries(1) + ag.generate_history_tags(1)
        assert len(set(allc)) == 2 * ag.N_CANARIES
        assert r2.generate_canary(1) not in allc

    def test_history_tags_only_in_first_user_message(self):
        ags = _session()
        first = ag.user_message(_turn(ags, 1), ags)
        for h in ags.hist_tags:
            assert h in first and h not in ags.system_prompt
            assert h not in ag.user_message(_turn(ags, 2), ags)

    def test_system_prompt_has_numbered_canaries_not_r2_single(self):
        ags = _session()
        for k, c in enumerate(ags.sys_canaries, 1):
            assert f"C{k}: {c}" in ags.system_prompt
        assert ags.spec.canary not in ags.system_prompt
        assert ags.spec.session_code in ags.system_prompt and "ZEBRA-7" in ags.system_prompt

    def test_canary_turn_asks_for_its_own_pair_only(self):
        ags = _session(40)
        msg5 = ag.user_message(_turn(ags, 5), ags)
        assert "canary phrase C1" in msg5 and "history tag H1" in msg5
        assert "this session's canary phrase exactly" not in msg5  # r2's single-canary ask removed
        msg40 = ag.user_message(_turn(ags, 40), ags)
        assert "C8" in msg40 and "H8" in msg40
        assert ag.expected_canaries(ags, _turn(ags, 10)) == (ags.sys_canaries[1], ags.hist_tags[1])
        assert ag.expected_canaries(ags, _turn(ags, 11)) == (None, None)

    def test_too_many_checks_raises(self):
        ags = _session(45)
        with pytest.raises(ValueError):
            ag.user_message(_turn(ags, 45), ags)

    def test_fact_store_excludes_recall_facts(self):
        ags = _session()
        assert all(k.startswith("REC-") for k in ags.store)
        for key, _ in ags.spec.facts:
            assert key not in ags.store
        assert len(ags.store) == 5  # odd turns 1..9


# ── tool extraction / validity / args ───────────────────────────────────────────────────────────────

class TestToolScoring:
    def test_native_preferred(self):
        calls, method = ag.extract_tool_calls('{"name": "lookup_fact", "arguments": {"key": "x"}}',
                                              [_native("log_event", event="e")])
        assert method == "native" and calls == [{"name": "log_event", "arguments": {"event": "e"}}]

    def test_text_fallback_bare_and_list_and_parameters(self):
        c, m = ag.extract_tool_calls('{"name": "log_event", "parameters": {"event": "e"}}', None)
        assert m == "text_fallback" and c[0]["arguments"] == {"event": "e"}
        c, m = ag.extract_tool_calls('```json\n[{"name": "log_event", "arguments": {"event": "e"}}]\n```', [])
        assert m == "text_fallback" and len(c) == 1

    def test_answer_shaped_json_is_not_a_call(self):
        txt = json.dumps({"answer": "a", "source": "s", "tool_calls": [{"name": "log_event", "arguments": {"event": "e"}}]})
        assert ag.extract_tool_calls(txt, None) == ([], "none")

    def test_string_arguments_are_decoded(self):
        c, _ = ag.extract_tool_calls(None, [{"function": {"name": "lookup_fact", "arguments": '{"key": "REC-0001"}'}}])
        assert c[0]["arguments"] == {"key": "REC-0001"}

    def test_validity(self):
        assert ag.tool_call_valid({"name": "log_event", "arguments": {"event": "x"}})
        assert not ag.tool_call_valid({"name": "delete_db", "arguments": {"event": "x"}})
        assert not ag.tool_call_valid({"name": "log_event", "arguments": {"evt": "x"}})
        assert not ag.tool_call_valid({"name": "log_event", "arguments": {"event": "x", "extra": "y"}})
        assert not ag.tool_call_valid({"name": "lookup_fact", "arguments": {"key": 5}})
        assert not ag.tool_call_valid({"name": "lookup_fact", "arguments": "REC-0001"})

    def test_args_lookup(self):
        ags = _session()
        t1 = _turn(ags, 1)
        assert ag.tool_args_correct({"name": "lookup_fact", "arguments": {"key": "REC-0001"}}, t1)
        assert not ag.tool_args_correct({"name": "lookup_fact", "arguments": {"key": "REC-0003"}}, t1)
        # lookup on a turn whose task is not a lookup is a wrong-argument call
        assert not ag.tool_args_correct({"name": "lookup_fact", "arguments": {"key": "ALPHA-CACHE"}}, _turn(ags, 4))

    def test_args_log_event_length_turn(self):
        ags = _session()
        t2 = _turn(ags, 2)
        rack, cm = ag._LEN_EVENT.fullmatch(t2.tool_args["event"]).groups()
        assert ag.tool_args_correct({"name": "log_event", "arguments": {"event": f"cable in rack {rack}: {cm} cm"}}, t2)
        assert ag.tool_args_correct({"name": "log_event",
                                     "arguments": {"event": f"{rack} {int(cm) / 100:.2f} m"}}, t2)
        assert not ag.tool_args_correct({"name": "log_event", "arguments": {"event": "something happened"}}, t2)
        # on a lookup turn, rule 2's log_event only needs a non-empty event
        assert ag.tool_args_correct({"name": "log_event", "arguments": {"event": "audit"}}, _turn(ags, 1))
        assert not ag.tool_args_correct({"name": "log_event", "arguments": {"event": "  "}}, _turn(ags, 1))

    def test_simulated_results(self):
        ags = _session()
        ok = json.loads(ag.simulate_tool_result({"name": "lookup_fact", "arguments": {"key": "REC-0001"}}, ags.store))
        assert ok["value"] == ags.store["REC-0001"]
        nf = json.loads(ag.simulate_tool_result({"name": "lookup_fact", "arguments": {"key": "ALPHA-CACHE"}}, ags.store))
        assert nf["value"] is None and nf["message"] == ag.NOT_FOUND
        assert json.loads(ag.simulate_tool_result({"name": "log_event", "arguments": {"event": "e"}}, ags.store))["status"] == "ok"
        assert json.loads(ag.simulate_tool_result({"name": "x", "arguments": {}}, ags.store))["status"] == "error"


# ── final-answer rules ──────────────────────────────────────────────────────────────────────────────

class TestFinalAnswer:
    def _score(self, turn_idx, text):
        ags = _session()
        sc, hc = ag.expected_canaries(ags, _turn(ags, turn_idx))
        return ag.score_final_answer(_turn(ags, turn_idx), text, ags.spec.session_code, sc, hc), ags

    def test_compliant(self):
        ags = _session()
        s = ag.score_final_answer(_turn(ags, 2), _answer("13.5 metres", ags.spec.session_code),
                                  ags.spec.session_code, None, None)
        assert s["rule1_json_keys"] and s["rule3_no_zebra"] and s["rule4_metres"] and s["rule5_session_code"]
        assert s["canary_sys_ok"] is None and s["recall_ok"] is None

    def test_rule4_only_on_length_turns(self):
        s, _ = self._score(1, "anything")
        assert s["rule4_metres"] is None

    def test_rule4_independent_of_rule1(self):
        s, ags = self._score(2, "It is 40 feet long")
        assert s["rule1_json_keys"] is False and s["rule4_metres"] is False
        s, ags = self._score(2, "It is 12 metres long")
        assert s["rule1_json_keys"] is False and s["rule4_metres"] is True

    def test_rule1_extra_key_and_trailing_text(self):
        ags = _session()
        code = ags.spec.session_code
        t = _turn(ags, 1)
        bad_key = json.dumps({"answer": f"x {code}", "source": "s", "note": "n"})
        assert not ag.score_final_answer(t, bad_key, code, "", "")["rule1_json_keys"]
        trailing = json.dumps({"answer": "x", "source": "s"}) + code
        s = ag.score_final_answer(t, trailing, code, "", "")
        assert not s["rule1_json_keys"] and not s["rule5_session_code"]
        fenced = "```json\n" + _answer("x", code) + "\n```"
        assert ag.score_final_answer(t, fenced, code, "", "")["rule1_json_keys"]

    def test_zebra(self):
        s, _ = self._score(1, '{"answer": "ZEBRA-7", "source": "s"}')
        assert s["rule3_no_zebra"] is False

    def test_canaries_scored_separately(self):
        ags = _session()
        t5 = _turn(ags, 5)
        code = ags.spec.session_code
        sc, hc = ag.expected_canaries(ags, t5)
        both = _answer(f"{sc} {hc}", code)
        s = ag.score_final_answer(t5, both, code, sc, hc)
        assert s["canary_sys_ok"] and s["canary_hist_ok"]
        only_sys = _answer(sc, code)
        s = ag.score_final_answer(t5, only_sys, code, sc, hc)
        assert s["canary_sys_ok"] is True and s["canary_hist_ok"] is False

    def test_recall(self):
        ags = _session()
        t4 = _turn(ags, 4)
        code = ags.spec.session_code
        assert ag.score_final_answer(t4, _answer(t4.recall_value, code), code, "", "")["recall_ok"] is True
        assert ag.score_final_answer(t4, _answer("VAL-WRONG", code), code, "", "")["recall_ok"] is False

    def test_empty(self):
        s, _ = self._score(1, "")
        assert s["final_content_empty"] and not s["rule1_json_keys"]


# ── fake runtime and full turn/session flow ─────────────────────────────────────────────────────────

class FakeRuntime:
    """Scripted runtime. policy(call_no_in_turn, messages, turn_idx) -> (content, native_tool_calls)."""

    def __init__(self, policy, loaded=131072, prompt_eval_first=None):
        self.policy = policy
        self.loaded = loaded
        self.calls = []
        self.unloads = []
        self.prompt_eval_first = prompt_eval_first

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        snapshot = [dict(m) for m in messages]
        self.calls.append({"model": model, "messages": snapshot, "num_ctx": num_ctx, "tools": tools, "think": think,
                           "extra": extra})
        turn_idx = sum(1 for m in messages if m["role"] == "user")
        call_no = 1 if messages[-1]["role"] == "user" else 2
        content, native = self.policy(call_no, messages, turn_idx)
        pec = self.prompt_eval_first if len(self.calls) == 1 else 7  # 7: deliberately nonsense after call 1
        return {"outcome": "ok", "status": 200, "message": content, "tool_calls": native,
                "prompt_eval_count": pec, "eval_count": 20, "duration_s": 1.5,
                "raw": {"done_reason": "stop", "message": {"content": content}}}

    def loaded_context(self, model):
        return self.loaded

    def unload(self, model):
        self.unloads.append(model)
        return True

    def recover(self):
        return True


def good_policy(ags):
    code = ags.spec.session_code

    def policy(call_no, messages, turn_idx):
        turn = ags.spec.turns[turn_idx - 1]
        if call_no == 1:
            calls = [_native("log_event", event=f"audit turn {turn_idx}")]
            if turn.tool_name == "lookup_fact":
                calls.append(_native("lookup_fact", key=turn.tool_args["key"]))
            else:
                calls = [_native("log_event", event=turn.tool_args["event"])]
            return "", calls
        parts = ["done"]
        if turn.is_recall:
            parts.append(turn.recall_value)
        if turn.canary_check:
            parts += list(ag.expected_canaries(ags, turn))
        return _answer(" ".join(parts), code), None
    return policy


def _run(policy_factory, turns=10, loaded=131072, arm="ollama_ctx_131072", prompt_eval_first=None, model="qwen3:14b"):
    ags = _session(turns)
    rt = FakeRuntime(policy_factory(ags), loaded=loaded, prompt_eval_first=prompt_eval_first)
    rows = []
    summary = ag.run_session(rt, model, arm, SEED, turns, rows.append, lambda m: None, "validation")
    return rt, rows, summary, ags


class TestTurnFlow:
    def test_two_calls_with_tool_results_in_between(self):
        rt, rows, summary, ags = _run(good_policy, turns=1)
        assert len(rt.calls) == 2
        second = rt.calls[1]["messages"]
        assert second[-1]["role"] == "tool" and second[-2]["role"] == "tool"
        assert second[-3]["role"] == "assistant" and second[-3]["tool_calls"]
        looked_up = json.loads(second[-1]["content"])
        assert looked_up["value"] == ags.store["REC-0001"]
        assert rt.calls[0]["tools"] and rt.calls[1]["tools"]  # tools available on both calls
        assert rt.calls[0]["think"] is False  # qwen3: think false sent and recorded
        assert rows[0]["think"] is False

    def test_good_session_all_pass(self):
        rt, rows, summary, ags = _run(good_policy)
        turn_rows = [r for r in rows if r["record"] == "r2a_turn"]
        assert len(turn_rows) == 10
        for r in turn_rows:
            assert r["n_calls"] == 2
            assert r["rule1_json_keys"] and r["rule2_log_event_called"] and r["rule5_session_code"]
            assert r["tool_validity_all"] and r["tool_args_all"] and r["task_tool_ok"]
        assert summary["first_failure_turn"] is None
        assert summary["canary_sys_misses"] == 0 and summary["canary_hist_misses"] == 0
        assert summary["truncation_detected_turn"] is None
        assert rt.unloads == ["qwen3:14b", "qwen3:14b"]  # fresh load before, unload after

    def test_rule2_satisfied_by_call2_only(self):
        def factory(ags):
            code = ags.spec.session_code

            def policy(call_no, messages, turn_idx):
                turn = ags.spec.turns[turn_idx - 1]
                if call_no == 1:
                    return "", [_native("lookup_fact", key=turn.tool_args.get("key", "REC-0001"))]
                if messages[-1]["role"] == "tool" and messages[-2]["role"] == "assistant":
                    # call 2: log_event natively, with a text answer alongside
                    return _answer("x", code), [_native("log_event", event="late audit")]
                return _answer("x", code), None
            return policy
        rt, rows, summary, ags = _run(factory, turns=1)
        r = rows[0]
        assert r["n_calls"] == 2 and r["n_tool_calls_call2"] == 1
        assert r["rule2_log_event_called"] is True  # turn-level OR across both calls
        assert len(rt.calls) == 2  # no third call even though call 2 called a tool

    def test_rule2_fails_when_never_called(self):
        def factory(ags):
            code = ags.spec.session_code

            def policy(call_no, messages, turn_idx):
                if call_no == 1:
                    return "", [_native("lookup_fact", key="REC-0001")]
                return _answer("x", code), None
            return policy
        _, rows, summary, _ = _run(factory, turns=1)
        assert rows[0]["rule2_log_event_called"] is False
        assert summary["first_failure_turn"] == 1 and "rule2_log_event_called" in summary["first_failure_reasons"]

    def test_text_rules_scored_on_final_answer_not_call1(self):
        def factory(ags):
            code = ags.spec.session_code

            def policy(call_no, messages, turn_idx):
                if call_no == 1:
                    # call 1 content is junk prose (would fail rule 1) plus a native call
                    return "let me call the tool ZEBRA-7", [_native("log_event", event="e"),
                                                             _native("lookup_fact", key="REC-0001")]
                return _answer("fine", code), None
            return policy
        _, rows, _, _ = _run(factory, turns=1)
        r = rows[0]
        assert r["rule1_json_keys"] and r["rule5_session_code"] and r["rule3_no_zebra"]

    def test_no_tool_call_means_single_call_and_call1_is_final(self):
        def factory(ags):
            code = ags.spec.session_code
            return lambda call_no, messages, turn_idx: (_answer("x", code), None)
        rt, rows, _, _ = _run(factory, turns=1)
        assert len(rt.calls) == 1 and rows[0]["n_calls"] == 1
        assert rows[0]["rule1_json_keys"] and rows[0]["rule2_log_event_called"] is False

    def test_invalid_call_counts_against_validity_not_args(self):
        def factory(ags):
            code = ags.spec.session_code

            def policy(call_no, messages, turn_idx):
                if call_no == 1:
                    return "", [_native("log_event", evt="bad"), _native("lookup_fact", key="REC-0009")]
                return _answer("x", code), None
            return policy
        _, rows, _, _ = _run(factory, turns=1)
        r = rows[0]
        assert r["n_tool_calls"] == 2 and r["n_valid_tool_calls"] == 1 and r["n_args_correct"] == 0
        assert r["tool_validity_all"] is False and r["tool_args_all"] is False
        assert r["rule2_log_event_called"] is True  # called by name, even though its args were malformed
        assert r["task_tool_ok"] is False


class TestTokens:
    def test_both_calls_counted_and_transcript_grows_with_round_trip(self):
        rt, rows, _, _ = _run(good_policy, turns=2)
        r1 = rows[0]
        assert len(r1["calls"]) == 2
        expected = sum(c["prompt_tokens_est"] + c["completion_tokens"] for c in r1["calls"])
        assert r1["turn_tokens_both_calls"] == expected
        assert r1["calls"][1]["prompt_tokens_est"] > r1["calls"][0]["prompt_tokens_est"]
        assert rows[1]["session_tokens_billed_cumulative"] == r1["turn_tokens_both_calls"] + rows[1]["turn_tokens_both_calls"]
        # turn 2's first call carries turn 1's whole tool round trip in its prompt
        msgs = rt.calls[2]["messages"]
        assert [m["role"] for m in msgs[:6]] == ["system", "user", "assistant", "tool", "tool", "assistant"]

    def test_calibration_uses_first_call_only_and_bounds(self):
        _, rows, _, _ = _run(good_policy, turns=2, prompt_eval_first=None)
        assert rows[0]["token_calib_ratio"] is None
        ags = _session(2)
        est = ag.prompt_tokens_est([{"role": "system", "content": ags.system_prompt},
                                    {"role": "user", "content": ag.user_message(_turn(ags, 1), ags)}],
                                   r2.ollama_tools_payload())
        _, rows, _, _ = _run(good_policy, turns=2, prompt_eval_first=int(est * 1.2))
        assert rows[0]["token_calib_ratio"] == pytest.approx(1.2, rel=0.01)
        assert rows[1]["token_calib_ratio"] == rows[0]["token_calib_ratio"]  # later prompt_eval_counts ignored
        _, rows, _, _ = _run(good_policy, turns=1, prompt_eval_first=int(est * 10))
        assert rows[0]["token_calib_ratio"] is None  # out of bounds -> not trusted


class TestTruncation:
    def _forgetful_after(self, n_turn):
        """Model that reproduces the history tag only while turn <= n_turn (simulating loss of the first user
        message), but always reproduces the system canary (Ollama keeps system messages)."""
        def factory(ags):
            base = good_policy(ags)
            code = ags.spec.session_code

            def policy(call_no, messages, turn_idx):
                content, native = base(call_no, messages, turn_idx)
                if call_no == 2 and turn_idx > n_turn and ags.spec.turns[turn_idx - 1].canary_check:
                    return _answer(f"done {ag.expected_canaries(ags, ags.spec.turns[turn_idx - 1])[0]}", code), None
                return content, native
            return factory_policy(policy)
        return factory

    def test_positive_control_fires_on_miss_over_window(self):
        _, rows, summary, _ = _run(self._forgetful_after(0), loaded=8192, arm="ollama_ctx_8192_positive_control")
        assert summary["canary_hist_misses"] == 2 and summary["canary_sys_misses"] == 0
        over = [r for r in rows if r["record"] == "r2a_turn" and r["over_loaded_window"]]
        assert over, "a 10-turn session with ~1.5k-token filler must exceed an 8192 window"
        assert summary["truncation_detected_turn"] in (5, 10)
        assert summary["truncation_detected_turn"] >= summary["first_over_window_turn"]

    def test_miss_within_window_is_not_truncation(self):
        _, rows, summary, _ = _run(self._forgetful_after(0), loaded=131072)
        assert summary["canary_hist_misses"] == 2
        assert summary["truncation_detected_turn"] is None
        assert summary["canary_misses_within_window"] == 2

    def test_over_window_without_miss_is_not_truncation(self):
        _, _, summary, _ = _run(good_policy, loaded=4096)
        assert summary["first_over_window_turn"] is not None
        assert summary["truncation_detected_turn"] is None

    def test_prompt_eval_count_never_drives_truncation(self):
        # prompt_eval_count is 7 on every call after the first (nonsense); truncation must not change
        _, rows, summary, _ = _run(good_policy, loaded=131072, prompt_eval_first=None)
        assert all(r["over_loaded_window"] is False for r in rows if r["record"] == "r2a_turn")
        assert summary["truncation_detected_turn"] is None

    def test_unknown_loaded_context_never_flags(self):
        _, rows, summary, _ = _run(self._forgetful_after(0), loaded=None)
        assert summary["truncation_detected_turn"] is None


def factory_policy(policy):
    return policy


# ── kill criterion, baseline table, gates ───────────────────────────────────────────────────────────

def _sess(arm, first_failure, first_error, attempt="a"):
    return {"arm_id": arm, "attempt_id": attempt, "first_failure_turn": first_failure,
            "first_error_turn": first_error,
            "error_surfaced_before_failure": (None if first_failure is None
                                              else (first_error is not None and first_error < first_failure))}


class TestKillCriterion:
    def test_killed_when_every_failure_follows_an_error(self):
        res = ag.kill_criterion([_sess("a", 5, 2), _sess("b", 7, 1), _sess("b", None, None)])
        assert res["killed"] is True

    def test_alive_with_one_silent_failure(self):
        res = ag.kill_criterion([_sess("a", 5, 2), _sess("b", 7, None)])
        assert res["killed"] is False and res["per_arm"]["b"]["n_silent_failures"] == 1

    def test_rules_in_use_resummarizes(self):
        _, rows, summary, _ = _run(lambda ags: _rule5_breaker(ags), turns=3)
        summary = dict(summary)
        assert summary["first_failure_turn"] == 1
        res_all = ag.kill_criterion([summary], ag.RULE_IDS, rows)
        assert res_all["killed"] is False  # silent rule-5 failure keeps the claim alive
        res_dropped = ag.kill_criterion([summary], tuple(r for r in ag.RULE_IDS if r != "rule5_session_code"), rows)
        assert res_dropped["per_arm"]["ollama_ctx_131072"]["n_silent_failures"] == 0

    def test_http_error_before_failure(self):
        rows = [{"turn_idx": 1, "any_http_error": True},
                {"turn_idx": 2, "any_http_error": False, "rule1_json_keys": False}]
        s = ag.summarize_session(rows)
        assert s["first_error_turn"] == 1 and s["first_failure_turn"] == 2
        assert s["error_surfaced_before_failure"] is True


def _rule5_breaker(ags):
    base = good_policy(ags)

    def policy(call_no, messages, turn_idx):
        content, native = base(call_no, messages, turn_idx)
        if call_no == 2:
            obj = json.loads(content)
            obj["answer"] = obj["answer"].replace(ags.spec.session_code, "").strip()
            return json.dumps(obj), None
        return content, native
    return policy


class TestGates:
    def _validation_rows(self, neg_factory=good_policy, pos_factory=None):
        rows = []
        for model in ("llama3.1:8b", "qwen3:14b"):
            for seed in ag.SEEDS:
                ags = ag.build_agent_session(seed, 10)
                rt = FakeRuntime(neg_factory(ags), loaded=131072)
                ag.run_session(rt, model, "ollama_ctx_131072", seed, 10, rows.append, lambda m: None, "validation")
            ags = ag.build_agent_session(ag.SEEDS[0], 10)
            pf = pos_factory or TestTruncation()._forgetful_after(0)
            rt = FakeRuntime(pf(ags), loaded=8192)
            ag.run_session(rt, model, "ollama_ctx_8192_positive_control", ag.SEEDS[0], 10, rows.append,
                           lambda m: None, "validation")
        return rows

    def test_all_gates_pass(self):
        rep = ag.evaluate_gates(self._validation_rows())
        g = rep["gates"]
        for model in ("llama3.1:8b", "qwen3:14b"):
            assert g["baseline"][model]["pass"]
            assert g["negative_control"][model]["pass"]
            assert g["positive_control"][model]["pass"]
            assert g["content_nonempty"][model]["pass"]
            t = rep["baseline_table"][model]
            assert t["n_sessions"] == 3 and t["n_turns"] == 30
            assert t["rules"]["rule4_metres"]["n"] == 15  # length turns only
        md = ag.format_baseline_markdown(rep)
        assert "rule5_session_code" in md and "positive control" in md

    def test_rule_below_threshold_fails_baseline(self):
        rep = ag.evaluate_gates(self._validation_rows(neg_factory=_rule5_breaker))
        for model in ("llama3.1:8b", "qwen3:14b"):
            b = rep["gates"]["baseline"][model]
            assert not b["pass"] and b["failing_rules"] == ["rule5_session_code"]
            assert "rule5_session_code" not in b["rules_in_use"]

    def test_negative_control_fails_on_any_miss(self):
        rep = ag.evaluate_gates(self._validation_rows(neg_factory=TestTruncation()._forgetful_after(5)))
        assert not rep["gates"]["negative_control"]["qwen3:14b"]["pass"]

    def test_positive_control_fails_if_no_miss(self):
        rep = ag.evaluate_gates(self._validation_rows(pos_factory=good_policy))
        assert not rep["gates"]["positive_control"]["llama3.1:8b"]["pass"]

    def test_not_exercised(self):
        rows = [r for r in self._validation_rows()]
        for r in rows:
            if r.get("record") == "r2a_turn":
                r["rule4_metres"] = None
        rep = ag.evaluate_gates(rows)
        assert rep["baseline_table"]["qwen3:14b"]["rules"]["rule4_metres"]["status"] == "not_exercised"
        assert rep["gates"]["baseline"]["qwen3:14b"]["pass"]

    def test_incomplete_session_excluded(self):
        rows = self._validation_rows()
        rows.append({**[r for r in rows if r["record"] == "r2a_turn"][0], "attempt_id": "orphan",
                     "rule1_json_keys": False})
        rep = ag.evaluate_gates(rows)
        assert rep["baseline_table"]["llama3.1:8b"]["rules"]["rule1_json_keys"]["rate"] == 1.0


class TestResume:
    def test_run_plan_skips_completed_sessions(self, tmp_path):
        out = tmp_path / "r.jsonl"
        plan = [("ollama_ctx_131072", ag.SEEDS[:2], 2)]

        class RT(FakeRuntime):
            def __init__(self):
                super().__init__(lambda *a: ("", None))

        def make_rt():
            rt = RT()
            rt.policy = lambda call_no, messages, turn_idx: (_answer("x", "SC"), None)
            return rt
        rt = make_rt()
        ag.run_plan(rt, plan, ["llama3.1:8b"], "validation", out, log=lambda m: None)
        assert len(rt.calls) == 4
        rt2 = make_rt()
        ag.run_plan(rt2, plan, ["llama3.1:8b"], "validation", out, log=lambda m: None)
        assert rt2.calls == []
        assert len(ag.completed_sessions(ag.read_rows(out))) == 2

    def test_llama_has_no_think_field(self):
        rt, rows, _, _ = _run(good_policy, turns=1, model="llama3.1:8b")
        assert rt.calls[0]["think"] is None and rows[0]["think"] is None


class TestCall2ToolsVariant:
    def test_notools_arm_withholds_tools_on_call2_only(self):
        rt, rows, _, _ = _run(good_policy, turns=1, arm="ollama_ctx_131072" + ag.NOTOOLS_SUFFIX)
        assert rt.calls[0]["tools"] and rt.calls[1]["tools"] is None
        assert rt.calls[0]["num_ctx"] == 131072 and rows[0]["call2_tools"] is False

    def test_real_plan_variants(self):
        assert [a for a, _, _ in ag.real_plan(True)] == ["ollama_default", "ollama_ctx_32768", "ollama_ctx_4096"]
        assert all(a.endswith(ag.NOTOOLS_SUFFIX) for a, _, _ in ag.real_plan(False))
        assert ag.ARMS["ollama_default"]["num_ctx"] is None

    def test_diagnostic_table_separate_from_gates(self):
        rows = TestGates()._validation_rows()
        for seed in ag.SEEDS:
            ags = ag.build_agent_session(seed, 10)
            rt = FakeRuntime(_rule5_breaker(ags))
            ag.run_session(rt, "qwen3:14b", "ollama_ctx_131072" + ag.NOTOOLS_SUFFIX, seed, 10, rows.append,
                           lambda m: None, "validation")
        rep = ag.evaluate_gates(rows)
        assert rep["gates"]["baseline"]["qwen3:14b"]["pass"]
        assert rep["diagnostic_call2_notools_table"]["qwen3:14b"]["rules"]["rule5_session_code"]["status"] == "fail"


class TestPerCheckCanaries:
    def test_echo_of_earlier_canary_does_not_count(self):
        """A model that copies the canaries it gave at turn 5 when asked at turn 10 (the echo seen in the first
        validation run) must score a miss at turn 10, and over the window that is truncation."""
        def factory(ags):
            base = good_policy(ags)
            code = ags.spec.session_code

            def policy(call_no, messages, turn_idx):
                content, native = base(call_no, messages, turn_idx)
                if call_no == 2 and turn_idx == 10:
                    sc, hc = ag.expected_canaries(ags, ags.spec.turns[4])  # the turn-5 pair
                    return _answer(f"done {sc} {hc}", code), None
                return content, native
            return policy
        _, rows, summary, _ = _run(factory, loaded=8192, arm="ollama_ctx_8192_positive_control")
        t10 = [r for r in rows if r.get("turn_idx") == 10 and r["record"] == "r2a_turn"][0]
        assert t10["canary_sys_ok"] is False and t10["canary_hist_ok"] is False
        assert t10["other_canaries_in_answer"] == 2 and t10["canary_k"] == 2
        assert summary["truncation_detected_turn"] == 10

    def test_validation_plan_variants(self):
        p = ag.validation_plan(False)
        assert p[0][0] == "ollama_ctx_131072" + ag.NOTOOLS_SUFFIX and p[1][2] == 15
        assert p[2][0] == "ollama_ctx_131072"
        assert len(ag.validation_plan(True, with_diagnostic=False)) == 2

    def test_gates_on_notools_variant_and_rules_in_use_per_model(self):
        rows = []
        for model, fac in (("llama3.1:8b", _rule5_breaker), ("qwen3:14b", good_policy)):
            for seed in ag.SEEDS:
                ags = ag.build_agent_session(seed, 10)
                ag.run_session(FakeRuntime(fac(ags)), model, "ollama_ctx_131072" + ag.NOTOOLS_SUFFIX, seed, 10,
                               rows.append, lambda m: None, "validation")
        rep = ag.evaluate_gates(rows, call2_tools=False)
        assert rep["gates"]["baseline"]["qwen3:14b"]["pass"]
        assert not rep["gates"]["baseline"]["llama3.1:8b"]["pass"]
        riu = ag.rules_in_use_from(rows, call2_tools=False)
        assert "rule5_session_code" not in riu["llama3.1:8b"] and "rule5_session_code" in riu["qwen3:14b"]
        sessions = ag.completed_sessions(rows)
        kc = ag.kill_criterion(sessions, riu, rows)
        # llama's only failures are rule 5, which is not in use for llama -> no silent failure from llama
        assert kc["per_arm"]["ollama_ctx_131072" + ag.NOTOOLS_SUFFIX]["n_silent_failures"] == 0


# ── call-2 protocol v3: forced_none (identical prompt and tools, tool_choice none on call 2) ─────────

FN = "ollama_ctx_131072" + ag.MODE_SUFFIX["forced_none"]


def _call2_tool_caller(ags):
    """Answers call 2 with another tool call (what llama3.1:8b did on 20 of 30 turns in the first validation)."""
    base = good_policy(ags)

    def policy(call_no, messages, turn_idx):
        if call_no == 2:
            return "", [_native("log_event", event="again")]
        return base(call_no, messages, turn_idx)
    return policy


class TestForcedNone:
    def test_identical_prompt_and_tools_and_tool_choice_on_call2_only(self):
        rt, rows, _, _ = _run(good_policy, turns=3, arm=FN)
        assert len(rt.calls) == 6
        assert all(c["tools"] for c in rt.calls)  # tools on every call
        for i, c in enumerate(rt.calls):
            assert c["extra"] == ({"tool_choice": "none"} if i % 2 == 1 else None)
        turn_rows = [r for r in rows if r["record"] == "r2a_turn"]
        for r in turn_rows:
            c1, c2 = r["calls"]
            assert c1["system_prompt_sha256"] == c2["system_prompt_sha256"] is not None
            assert c1["tools_sha256"] == c2["tools_sha256"] is not None
            assert c1["tool_choice_sent"] is None and c2["tool_choice_sent"] == "none"
            assert r["system_prompt_identical_across_calls"] and r["tools_identical_across_calls"]
            assert r["call2_mode"] == "forced_none" and r["call2_tool_choice_sent"] == "none"
            assert r["call2_protocol_violation"] is False
        # identical across the whole session too, not only within a turn
        assert len({c["system_prompt_sha256"] for r in turn_rows for c in r["calls"]}) == 1
        assert len({c["tools_sha256"] for r in turn_rows for c in r["calls"]}) == 1
        assert turn_rows[0]["call2_tools"] is True

    def test_hashes_match_what_is_sent(self):
        rt, rows, _, ags = _run(good_policy, turns=1, arm=FN)
        c2 = rows[0]["calls"][1]
        assert c2["tools_sha256"] == ag.sha256_text(json.dumps(rt.calls[1]["tools"]))
        assert c2["system_prompt_sha256"] == ag.sha256_text(ags.system_prompt)

    def test_call2_tool_call_is_protocol_violation(self):
        _, rows, summary, _ = _run(_call2_tool_caller, turns=2, arm=FN)
        r = rows[0]
        assert r["n_tool_calls_call2"] == 1 and r["call2_protocol_violation"] is True
        assert "call2_protocol_violation" in ag.turn_failures(r)
        assert summary["call2_protocol_violations"] == 2
        assert summary["first_failure_turn"] == 1
        assert "call2_protocol_violation" in summary["first_failure_reasons"]

    def test_violation_not_scored_in_other_modes(self):
        _, rows, _, _ = _run(_call2_tool_caller, turns=1)  # mode "on"
        assert rows[0]["call2_protocol_violation"] is None
        assert "call2_protocol_violation" not in ag.turn_failures(rows[0])
        rt, rows, _, _ = _run(good_policy, turns=1, arm="ollama_ctx_131072" + ag.NOTOOLS_SUFFIX)
        assert rows[0]["tools_identical_across_calls"] is False and rows[0]["calls"][1]["tools_sha256"] is None
        assert rt.calls[1]["extra"] is None

    def test_format_mode_sends_schema_on_call2_only(self):
        rt, rows, _, _ = _run(good_policy, turns=1, arm="ollama_ctx_131072_call2_forcednone_format")
        assert rt.calls[0]["extra"] is None
        assert rt.calls[1]["extra"] == {"tool_choice": "none", "format": ag.ANSWER_FORMAT_SCHEMA}
        assert rows[0]["calls"][1]["format_sent_sha256"] and rows[0]["tools_identical_across_calls"]

    def test_plans(self):
        p = ag.validation_plan("forced_none")
        assert [a for a, _, _ in p] == [FN, "ollama_ctx_8192_positive_control_call2_forcednone", "ollama_ctx_131072"]
        assert p[0][1] == ag.SEEDS and p[0][2] == 10 and p[1][2] == 15
        assert [a for a, _, _ in ag.real_plan("forced_none")] == [
            "ollama_default_call2_forcednone", "ollama_ctx_32768_call2_forcednone", "ollama_ctx_4096_call2_forcednone"]
        assert ag.ARMS["ollama_default_call2_forcednone"]["num_ctx"] is None
        # old bool spellings unchanged
        assert ag.validation_plan(False) == ag.validation_plan("off")
        assert ag.real_plan(True) == ag.real_plan("on")

    def test_native_body_matches_client_plus_extra(self):
        b = ag.native_chat_body("m", [{"role": "user", "content": "x"}], 4096, [{"t": 1}], False,
                                {"tool_choice": "none"})
        assert b["options"] == {"num_predict": ag.MAX_TOKENS_PER_CALL, "temperature": 0, "seed": 42, "num_ctx": 4096}
        assert b["tool_choice"] == "none" and b["tools"] == [{"t": 1}] and b["think"] is False
        assert b["keep_alive"] == ag.KEEP_ALIVE and b["stream"] is False
        b2 = ag.native_chat_body("m", [], None, None, None)
        assert "num_ctx" not in b2["options"] and "tools" not in b2 and "think" not in b2


def _validation_rows_mode(mode, neg_factory=good_policy, pos_factory=None, run_end=True, models=None):
    rows = [{"record": "run_start", "mode": "validation", "call2_mode": mode, "call2_tools": mode != "off"}]
    for model in models or ("llama3.1:8b", "qwen3:14b"):
        for arm, seeds, turns in ag.validation_plan(mode, with_diagnostic=False):
            for seed in seeds:
                ags = ag.build_agent_session(seed, turns)
                if "positive" in arm:
                    fac, loaded = (pos_factory or TestTruncation()._forgetful_after(0)), 8192
                else:
                    fac, loaded = neg_factory, 131072
                ag.run_session(FakeRuntime(fac(ags), loaded=loaded), model, arm, seed, turns, rows.append,
                               lambda m: None, "validation")
    if run_end:
        rows.append({"record": "run_end"})
    return rows


class TestValidationPreflight:
    MODELS = ["llama3.1:8b", "qwen3:14b"]

    def test_passes_when_all_gates_pass(self):
        pf = ag.validation_preflight(_validation_rows_mode("forced_none"), "forced_none", self.MODELS)
        assert pf["ok"], pf["reasons"]
        assert set(pf["rules_in_use"]["qwen3:14b"]) == set(ag.RULE_IDS)

    def test_refuses_on_negative_control_miss(self):
        rows = _validation_rows_mode("forced_none", neg_factory=TestTruncation()._forgetful_after(5))
        pf = ag.validation_preflight(rows, "forced_none", self.MODELS)
        assert not pf["ok"] and any("negative control" in r for r in pf["reasons"])

    def test_refuses_when_positive_control_does_not_fire(self):
        rows = _validation_rows_mode("forced_none", pos_factory=good_policy)
        pf = ag.validation_preflight(rows, "forced_none", self.MODELS)
        assert not pf["ok"] and any("positive control" in r for r in pf["reasons"])

    def test_refuses_when_no_rule_passes(self):
        def junk(ags):
            def policy(call_no, messages, turn_idx):
                if call_no == 2:
                    return "ZEBRA-7 not json, 3 feet", None
                return "", [_native("lookup_fact", key="REC-0001")]  # never log_event: rule 2 fails too
            return policy
        pf = ag.validation_preflight(_validation_rows_mode("forced_none", neg_factory=junk), "forced_none",
                                     self.MODELS)
        assert not pf["ok"] and any("no rule reached" in r for r in pf["reasons"])

    def test_dropped_rule_does_not_refuse_but_is_not_in_use(self):
        pf = ag.validation_preflight(_validation_rows_mode("forced_none", neg_factory=_rule5_breaker),
                                     "forced_none", self.MODELS)
        assert pf["ok"], pf["reasons"]
        assert "rule5_session_code" not in pf["rules_in_use"]["llama3.1:8b"]

    def test_refuses_unfinished_or_wrong_mode_or_missing_model(self):
        pf = ag.validation_preflight(_validation_rows_mode("forced_none", run_end=False), "forced_none", self.MODELS)
        assert not pf["ok"] and any("run_end" in r for r in pf["reasons"])
        pf = ag.validation_preflight(_validation_rows_mode("off"), "forced_none", self.MODELS)
        assert not pf["ok"] and any("call-2 mode" in r for r in pf["reasons"])
        pf = ag.validation_preflight(_validation_rows_mode("forced_none", models=["qwen3:14b"]), "forced_none",
                                     self.MODELS)
        assert not pf["ok"] and any(r.startswith("llama3.1:8b") for r in pf["reasons"])

    def test_main_refuses_without_starting_ollama(self, tmp_path, monkeypatch):
        vfile = tmp_path / "v3.jsonl"
        rows = _validation_rows_mode("forced_none", pos_factory=good_policy)
        vfile.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
        import socket
        import types
        monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
        started, notes = [], []
        monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
            start_ollama_server=lambda: started.append(1), stop_ollama_server=lambda: None,
            wait_for_ollama_ready=lambda timeout_s=0: True))
        monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))
        out = tmp_path / "real.jsonl"
        ag.main(["--mode", "real", "--call2-tools", "forced_none", "--rules-from", str(vfile),
                 "--require-validation-gates", "--out", str(out)])
        assert started == []
        assert len(notes) == 1 and notes[0].startswith("stopped: refused to start") and "STOP" not in notes[0]
        recs = ag.read_rows(out)
        assert [r["record"] for r in recs] == ["run_start", "refused"]
        assert recs[0]["call2_mode"] == "forced_none" and recs[0]["validation_preflight"]["ok"] is False


class TestGatesForced:
    def test_protocol_violations_in_table(self):
        rows = _validation_rows_mode("forced_none", neg_factory=_call2_tool_caller)
        rep = ag.evaluate_gates(rows, call2_tools="forced_none")
        t = rep["baseline_table"]["llama3.1:8b"]
        assert t["call2_protocol_violations"] == 30 and t["call2_tool_call_turns"] == 30
        assert t["prompt_or_tools_changed_within_turn"] == 0
        assert rep["call2_mode"] == "forced_none" and rep["diagnostic_mode"] == "on"
        assert "protocol violations" in ag.format_baseline_markdown(rep)


# ── tool_choice check (mock runtime that ignores tool_choice, as Ollama 0.34.4 does) ──────────────

class FakeTCRuntime(FakeRuntime):
    def render_only(self, model, messages, num_ctx, tools, think, extra=None):
        # the template never sees tool_choice: identical text with or without it
        return json.dumps([messages, tools]), None

    def chat_openai(self, model, messages, tools, think, tool_choice=None):
        self.calls.append({"openai": True, "messages": messages, "tool_choice": tool_choice})
        return {"outcome": "ok", "status": 200, "message": "", "tool_calls": [_native("log_event", event="x")]}


class TestToolChoiceCheck:
    def test_ignored_tool_choice_gives_no_effect_verdict(self):
        rows = []
        ags = ag.build_agent_session(ag.SEEDS[0], ag.TC_TURNS)
        rt = FakeTCRuntime(_call2_tool_caller(ags))
        ag.run_toolchoice_check(rt, ["llama3.1:8b"], rows.append, lambda m: None)
        s = ag.summarize_toolchoice_check(rows)["llama3.1:8b"]
        n = ag.TC_TURNS
        assert s["rendered_prompt_pairs"] == n and s["rendered_prompt_identical"] == n
        assert s["variants"]["on"]["tool_call_turns"] == n and s["variants"]["tool_choice_none"]["tool_call_turns"] == n
        assert s["identical_on_vs_tool_choice_none"] == {"n": n, "identical": n}
        assert s["verdict_api_chat_tool_choice_none"] == "no_effect_observed"
        assert s["verdict_v1_tool_choice_none"] == "no_effect_observed"
        # tools sent on every native call, tool_choice only on the variants that ask for it
        native = [c for c in rt.calls if not c.get("openai")]
        assert all(c["tools"] for c in native)
        assert sum(1 for c in native if (c["extra"] or {}).get("tool_choice") == "none") == 2 * n
        assert sum(1 for c in rt.calls if c.get("openai") and c["tool_choice"] == "none") == n

    def test_effect_detected_when_runtime_honours_it(self):
        class Honouring(FakeTCRuntime):
            def chat(self, model, messages, num_ctx, tools, think, extra=None):
                r = super().chat(model, messages, num_ctx, tools, think, extra)
                if extra and extra.get("tool_choice") == "none":
                    r = {**r, "message": _answer("ok", "SC"), "tool_calls": None}
                return r
        rows = []
        ags = ag.build_agent_session(ag.SEEDS[0], ag.TC_TURNS)
        ag.run_toolchoice_check(Honouring(_call2_tool_caller(ags)), ["llama3.1:8b"], rows.append, lambda m: None)
        s = ag.summarize_toolchoice_check(rows)["llama3.1:8b"]
        assert s["verdict_api_chat_tool_choice_none"] == "effect_observed"
        assert s["variants"]["tool_choice_none"]["tool_call_turns"] == 0

    def test_openai_message_conversion(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "", "tool_calls": [_native("log_event", event="e"),
                                                                    _native("lookup_fact", key="K")]},
                {"role": "tool", "content": "r1", "tool_name": "log_event"},
                {"role": "tool", "content": "r2", "tool_name": "lookup_fact"}]
        o = ag.to_openai_messages(msgs)
        tcs = o[2]["tool_calls"]
        assert [t["type"] for t in tcs] == ["function", "function"]
        assert json.loads(tcs[1]["function"]["arguments"]) == {"key": "K"}
        assert o[3]["tool_call_id"] == tcs[0]["id"] and o[4]["tool_call_id"] == tcs[1]["id"]

    def test_rendered_template_lookup(self):
        assert ag.rendered_template_of({"_debug_info": {"rendered_template": "abc"}}) == "abc"
        assert ag.rendered_template_of({"message": {}}) is None


# ── strengthened run (2026-10-08): thinking verification, plans, resume from v1, per-model preflight ────────────

class TestThinking:
    def _thinking_factory(self, kind):
        def factory(ags):
            base = good_policy(ags)

            def policy(call_no, messages, turn_idx):
                content, native = base(call_no, messages, turn_idx)
                if call_no == 2 and turn_idx == 2 and kind == "tag":
                    content = "<think>hmm</think>" + content
                return content, native
            return policy
        return factory

    def test_think_tag_in_content_fails_turn_loudly(self):
        logs = []
        ags = _session(3)
        rt = FakeRuntime(self._thinking_factory("tag")(ags))
        rows = []
        s = ag.run_session(rt, "qwen3:8b", "ollama_ctx_131072_call2_notools", SEED, 3, rows.append, logs.append,
                           "validation")
        turns = [r for r in rows if r["record"] == "r2a_turn"]
        assert [r["thinking_present"] for r in turns] == [False, True, False]
        assert "thinking_present" in ag.turn_failures(turns[1])
        assert turns[1]["calls"][1]["think_tag_in_content"] is True
        assert s["thinking_present_turns"] == 1 and any("THINKING PRESENT" in m for m in logs)
        assert rt.calls[0]["think"] is False  # think false sent for qwen3:8b

    def test_message_thinking_field_fails_turn(self):
        class ThinkingRT(FakeRuntime):
            def chat(self, *a, **k):
                r = super().chat(*a, **k)
                r["raw"] = {**r["raw"], "message": {"content": r["message"], "thinking": "let me think"}}
                return r
        ags = _session(1)
        rows = []
        ag.run_session(ThinkingRT(good_policy(ags)), "qwen3-4b-2507", "ollama_ctx_131072_call2_notools", SEED, 1,
                       rows.append, lambda m: None, "validation")
        t = next(r for r in rows if r["record"] == "r2a_turn")
        assert t["thinking_present"] is True and t["calls"][0]["thinking_present"] is True

    def test_thinking_gate_and_preflight(self):
        rows = _validation_rows_mode("off")
        for r in rows:
            if r.get("record") == "r2a_turn" and r["model_id"] == "qwen3:14b":
                r["calls"][0]["think_tag_in_content"] = True
                break
        rep = ag.evaluate_gates(rows, call2_tools="off")
        assert not rep["gates"]["thinking_off"]["qwen3:14b"]["pass"]
        assert rep["gates"]["thinking_off"]["llama3.1:8b"]["pass"]
        pf = ag.validation_preflight(rows, "off", ["llama3.1:8b", "qwen3:14b"])
        assert not pf["ok"] and any(r.startswith("qwen3:14b: thinking present") for r in pf["reasons"])

    def test_new_models_send_think_false(self):
        assert ag.MODELS["qwen3:8b"] is False and ag.MODELS["qwen3-4b-2507"] is False
        assert ag.MODELS["llama3.1:8b"] is None


class TestTags:
    def test_latest_normalization(self):
        assert ag.same_tag("qwen3-4b-2507", "qwen3-4b-2507:latest")
        assert ag.same_tag("qwen3:8b", "qwen3:8b") and not ag.same_tag("qwen3:8b", "qwen3:14b")
        assert not ag.same_tag(None, "x")


class TestStrongPlan:
    def test_plan_shape(self):
        plan = ag.strong_plan("off")
        assert [a for a, _, _ in plan] == ["ollama_default_call2_notools", "ollama_ctx_32768_call2_notools",
                                           "ollama_ctx_16384_call2_notools", "ollama_ctx_8192_call2_notools",
                                           "ollama_ctx_4096_call2_notools"]
        assert all(s == ag.SEEDS_STRONG and t == 40 for _, s, t in plan)
        assert ag.SEEDS_STRONG[:3] == tuple(ag.SEEDS) and len(set(ag.SEEDS_STRONG)) == 5
        assert all(ag.ARMS[a]["call2_mode"] == "off" for a, _, _ in plan)
        assert ag.CALL2_MODE_BY_RUNTIME == {"ollama": "off", "llama_server": "off", "openai": "forced_none"}

    def test_new_seeds_build_40_turn_sessions(self):
        for seed in ag.SEEDS_STRONG[3:]:
            ags = ag.build_agent_session(seed, 40)
            assert len(ags.spec.turns) == 40 and ag.canary_index(ags.spec.turns[39]) == 8

    def test_seed_major_order(self):
        plan = [("A", (1, 2), 3), ("B", (1, 2), 3)]
        got = ag.session_order(plan, ["m1", "m2"], "seed_major")
        assert got[:4] == [("m1", "A", 1, 3), ("m1", "B", 1, 3), ("m2", "A", 1, 3), ("m2", "B", 1, 3)]
        assert ag.session_order(plan, ["m1"], "model_major") == [("m1", "A", 1, 3), ("m1", "A", 2, 3),
                                                                 ("m1", "B", 1, 3), ("m1", "B", 2, 3)]

    def test_skip_done_from_v1_cells(self, tmp_path):
        v1 = []
        ags = _session(2)
        ag.run_session(FakeRuntime(good_policy(ags)), "llama3.1:8b", "ollama_ctx_4096_call2_notools", ag.SEEDS[0], 2,
                       v1.append, lambda m: None, "real")
        skip = ag.done_keys_from(v1, mode="real")
        assert skip == {("llama3.1:8b", "ollama_ctx_4096_call2_notools", ag.SEEDS[0])}
        out = tmp_path / "v1b.jsonl"
        rt = FakeRuntime(lambda c, m, t: (_answer("x", "SC"), None))
        plan = [("ollama_ctx_4096_call2_notools", ag.SEEDS[:2], 2)]
        ag.run_plan(rt, plan, ["llama3.1:8b"], "real", out, log=lambda m: None, skip_done=skip)
        done = ag.completed_sessions(ag.read_rows(out))
        assert [(s["seed"]) for s in done] == [ag.SEEDS[1]]
        hb = [r for r in ag.read_rows(out) if r["record"] == "heartbeat"]
        assert hb and all(r["call2_mode"] == "off" for r in hb)


def _vfile(tmp_path, name, models, **kw):
    rows = _validation_rows_mode("off", models=models, **kw)
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
    return p, rows


class TestPerModelPreflight:
    def test_each_model_checked_against_its_own_file(self, tmp_path):
        _, v2 = _vfile(tmp_path, "v2.jsonl", ["llama3.1:8b", "qwen3:14b"])
        _, v2b = _vfile(tmp_path, "v2b.jsonl", ["qwen3-4b-2507-tools", "qwen3:8b"])
        pf = ag.validation_preflight_per_model([("v2.jsonl", v2), ("v2b.jsonl", v2b)], "off", list(ag.STRONG_MODELS))
        assert pf["ok"] and pf["models_ok"] == list(ag.STRONG_MODELS) and pf["refused"] == {}
        assert pf["source"] == {"llama3.1:8b": "v2.jsonl", "qwen3:14b": "v2.jsonl", "qwen3-4b-2507-tools": "v2b.jsonl",
                                "qwen3:8b": "v2b.jsonl"}

    def test_failing_new_model_refused_alone(self, tmp_path):
        _, v2 = _vfile(tmp_path, "v2.jsonl", ["llama3.1:8b", "qwen3:14b"])
        rows = _validation_rows_mode("off", models=["qwen3-4b-2507-tools"])
        rows = rows[:-1] + _validation_rows_mode("off", models=["qwen3:8b"], pos_factory=good_policy)[1:]
        pf = ag.validation_preflight_per_model([("v2.jsonl", v2), ("v2b.jsonl", rows)], "off",
                                               list(ag.STRONG_MODELS))
        assert pf["ok"] and "qwen3:8b" not in pf["models_ok"] and "qwen3-4b-2507-tools" in pf["models_ok"]
        assert any("positive control" in r for r in pf["refused"]["qwen3:8b"])
        assert "qwen3:8b" not in pf["rules_in_use"]

    def test_missing_validation_refuses_model(self, tmp_path):
        _, v2 = _vfile(tmp_path, "v2.jsonl", ["llama3.1:8b", "qwen3:14b"])
        pf = ag.validation_preflight_per_model([("v2.jsonl", v2)], "off", ["llama3.1:8b", "qwen3:8b"])
        assert pf["models_ok"] == ["llama3.1:8b"] and "no validation rows" in pf["refused"]["qwen3:8b"][0]

    def test_unfinished_file_refuses_its_models_only(self, tmp_path):
        _, v2 = _vfile(tmp_path, "v2.jsonl", ["llama3.1:8b"])
        v2b = _validation_rows_mode("off", models=["qwen3:8b"], run_end=False)
        pf = ag.validation_preflight_per_model([("v2.jsonl", v2), ("v2b.jsonl", v2b)], "off",
                                               ["llama3.1:8b", "qwen3:8b"])
        assert pf["models_ok"] == ["llama3.1:8b"] and any("run_end" in r for r in pf["refused"]["qwen3:8b"])


def _patch_real_main(monkeypatch, plan):
    import socket
    import types
    monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
    state = {"started": 0, "notes": []}
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
        start_ollama_server=lambda: state.__setitem__("started", state["started"] + 1),
        stop_ollama_server=lambda: None, wait_for_ollama_ready=lambda timeout_s=0: True))
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=state["notes"].append))

    class RT(FakeRuntime):
        def __init__(self):
            super().__init__(lambda c, m, t: ("", [_native("log_event", event="e")]) if c == 1
                             else (_answer("ok", "SC"), None))
            self.server_env = self.server_log = None

        def available_models(self):
            return ["llama3.1:8b", "qwen3:14b", "qwen3-4b-2507-tools:latest", "qwen3:8b"]

        def version(self):
            return "0.34.4"
    monkeypatch.setattr(ag, "OllamaRuntime", RT)
    monkeypatch.setattr(ag, "strong_plan", lambda c2: plan)
    return state


class TestStrongMain:
    def test_refuses_failing_model_runs_rest_and_skips_v1(self, tmp_path, monkeypatch):
        v2p, _ = _vfile(tmp_path, "v2.jsonl", ["llama3.1:8b", "qwen3:14b"])
        rows = _validation_rows_mode("off", models=["qwen3-4b-2507-tools"])
        rows = rows[:-1] + _validation_rows_mode("off", models=["qwen3:8b"], pos_factory=good_policy)[1:]
        v2bp = tmp_path / "v2b.jsonl"
        v2bp.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
        plan = [("ollama_ctx_4096_call2_notools", ag.SEEDS_STRONG[:1] + ag.SEEDS_STRONG[3:4], 2)]
        v1 = tmp_path / "v1.jsonl"
        v1rows = [{"record": "run_start", "mode": "real", "call2_tools": False}]
        ags = _session(2)
        ag.run_session(FakeRuntime(good_policy(ags)), "llama3.1:8b", plan[0][0], ag.SEEDS_STRONG[0], 2,
                       v1rows.append, lambda m: None, "real")
        v1.write_text("\n".join(json.dumps(r, default=str) for r in v1rows) + "\n", encoding="utf-8")
        state = _patch_real_main(monkeypatch, plan)
        out = tmp_path / "v1b.jsonl"
        ag.main(["--mode", "real", "--plan", "strong", "--call2-tools", "off", "--rules-from", f"{v2p},{v2bp}",
                 "--require-validation-gates", "--per-model-refusal", "--skip-done-from", str(v1),
                 "--order", "seed_major", "--out", str(out)])
        assert state["started"] == 1
        assert len(state["notes"]) == 1 and state["notes"][0].startswith("completed; refused models")
        recs = ag.read_rows(out)
        assert recs[0]["record"] == "run_start" and recs[0]["models"] == ["llama3.1:8b", "qwen3:14b", "qwen3-4b-2507-tools"]
        assert [r["model_id"] for r in recs if r["record"] == "refused_model"] == ["qwen3:8b"]
        done = {(s["model_id"], s["seed"]) for s in ag.completed_sessions(recs)}
        assert ("llama3.1:8b", ag.SEEDS_STRONG[0]) not in done  # v1 cell reused, not rerun
        assert done == {("llama3.1:8b", ag.SEEDS_STRONG[3]), ("qwen3:14b", ag.SEEDS_STRONG[0]),
                        ("qwen3:14b", ag.SEEDS_STRONG[3]), ("qwen3-4b-2507-tools", ag.SEEDS_STRONG[0]),
                        ("qwen3-4b-2507-tools", ag.SEEDS_STRONG[3])}
        assert recs[-1]["record"] == "run_end"
        assert all(r.get("call2_mode") == "off" for r in recs if r["record"] not in ("run_start",))
        rep = json.loads(Path(str(out) + ".report.json").read_text(encoding="utf-8"))
        assert set(rep["rules_in_use"]) == {"llama3.1:8b", "qwen3:14b", "qwen3-4b-2507-tools"}

    def test_refuses_whole_job_when_no_model_passes(self, tmp_path, monkeypatch):
        rows = _validation_rows_mode("off", models=["qwen3:8b"], pos_factory=good_policy)
        p = tmp_path / "v2b.jsonl"
        p.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
        state = _patch_real_main(monkeypatch, [("ollama_ctx_4096_call2_notools", (1,), 2)])
        out = tmp_path / "v1b.jsonl"
        ag.main(["--mode", "real", "--plan", "strong", "--call2-tools", "off", "--models", "qwen3:8b",
                 "--rules-from", str(p), "--require-validation-gates", "--per-model-refusal", "--out", str(out)])
        assert state["started"] == 0 and state["notes"][0].startswith("stopped: refused to start")
        assert [r["record"] for r in ag.read_rows(out)] == ["run_start", "refused_model", "refused"]


class TestCall2ModeEverywhere:
    def test_toolchoice_rows_carry_call2_mode(self):
        rows = []
        ags = ag.build_agent_session(ag.SEEDS[0], ag.TC_TURNS)
        ag.run_toolchoice_check(FakeTCRuntime(_call2_tool_caller(ags)), ["llama3.1:8b"], rows.append, lambda m: None)
        assert rows and all(r.get("call2_mode") in ag.CALL2_MODES for r in rows)


# ── task-tool gate (2026-10-08) ──────────────────────────────────────────────────────────────────────

def _break_task_tool(rows, model, n_bad):
    """Mark the first n_bad baseline turns of `model` as task tool not called correctly."""
    k = 0
    for r in rows:
        if (r.get("record") == "r2a_turn" and r["model_id"] == model and r["arm_id"] == "ollama_ctx_131072_call2_notools"
                and k < n_bad):
            r["task_tool_ok"] = False
            k += 1
    return rows


class TestTaskToolGate:
    def test_exactly_90_percent_passes(self):
        rows = _break_task_tool(_validation_rows_mode("off"), "llama3.1:8b", 3)   # 27/30
        g = ag.evaluate_gates(rows, call2_tools="off")["gates"]["task_tool"]["llama3.1:8b"]
        assert g["pass"] and g["n_ok"] == 27 and g["n"] == 30
        assert ag.validation_preflight(rows, "off", ["llama3.1:8b", "qwen3:14b"])["ok"]

    def test_below_90_percent_refuses_model(self):
        rows = _break_task_tool(_validation_rows_mode("off"), "qwen3:14b", 4)     # 26/30
        g = ag.evaluate_gates(rows, call2_tools="off")["gates"]["task_tool"]["qwen3:14b"]
        assert not g["pass"] and g["n_ok"] == 26
        pf = ag.validation_preflight(rows, "off", ["llama3.1:8b", "qwen3:14b"])
        assert not pf["ok"] and pf["reasons"] == ["qwen3:14b: task tool called correctly on 26/30 baseline turns "
                                                  "(< 90%, task-tool gate)"]

    def test_per_model_refusal_drops_only_that_model(self, tmp_path):
        _, v2 = _vfile(tmp_path, "v2.jsonl", ["llama3.1:8b", "qwen3:14b"])
        v2c = _break_task_tool(_validation_rows_mode("off", models=["qwen3-4b-2507-tools"]), "qwen3-4b-2507-tools", 25)
        pf = ag.validation_preflight_per_model([("v2.jsonl", v2), ("v2c.jsonl", v2c)], "off",
                                               ["llama3.1:8b", "qwen3:14b", "qwen3-4b-2507-tools"])
        assert pf["models_ok"] == ["llama3.1:8b", "qwen3:14b"]
        assert "task-tool gate" in pf["refused"]["qwen3-4b-2507-tools"][0]

    @pytest.mark.parametrize("fname,model,n_ok,passes", [
        ("x2_r2_validation_v2.jsonl", "llama3.1:8b", 27, True),
        ("x2_r2_validation_v2.jsonl", "qwen3:14b", 30, True),
        ("x2_r2_validation_v2b.jsonl", "qwen3:8b", 30, True),
        ("x2_r2_validation_v2b.jsonl", "qwen3-4b-2507", 5, False),
    ])
    def test_real_validation_files(self, fname, model, n_ok, passes):
        p = Path(__file__).resolve().parents[1] / "results" / fname
        if not p.exists():
            pytest.skip(f"{fname} not present")
        rows = ag.read_rows(p)
        g = ag.evaluate_gates(rows, call2_tools="off")["gates"]["task_tool"][model]
        assert (g["n_ok"], g["n"], g["pass"]) == (n_ok, 30, passes)
        # the gate does not change the verdict for the models that passed before it existed
        if passes:
            assert ag.validation_preflight(rows, "off", [model])["ok"]

    def test_main_returns_note_and_can_skip_advance(self, tmp_path, monkeypatch):
        state = _patch_real_main(monkeypatch, [("ollama_ctx_4096_call2_notools", (1,), 2)])
        rows = _validation_rows_mode("off", models=["qwen3:8b"], pos_factory=good_policy)
        p = tmp_path / "v.jsonl"
        p.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
        note = ag.main(["--mode", "real", "--plan", "strong", "--call2-tools", "off", "--models", "qwen3:8b",
                        "--rules-from", str(p), "--require-validation-gates", "--per-model-refusal",
                        "--out", str(tmp_path / "o.jsonl")], advance=False)
        assert note.startswith("stopped: refused to start") and state["notes"] == []

    def test_strong_models_use_tools_tag(self):
        assert "qwen3-4b-2507-tools" in ag.STRONG_MODELS and "qwen3-4b-2507" not in ag.STRONG_MODELS
        assert ag.MODELS["qwen3-4b-2507-tools"] is False


def test_refusal_category_missing_errored_failed():
    """2026-10-08: the real run records why a model was refused: missing, errored or failed."""
    f = ag.refusal_category
    assert f(["m: no validation rows in any of ['v2c']"], {"v2c": False}, {"v2c": False}) == "missing"
    assert f(["m: no validation rows in any of ['v2c']"], {"v2c": True}, {"v2c": False}) == "errored"
    assert f(["validation file has no run_end record (validation did not finish)"], {"v2c": True}, {"v2c": False}) == "errored"
    assert f(["m: rule2 below 90%"], {"v2c": True}, {"v2c": True}) == "failed"
