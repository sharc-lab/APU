"""Tests for harness/t2s_r2_session_growth.py: turn generator, scorer, arm config-application,
memory-condition wiring, and a stub end-to-end dry run against fakes (no real network/process
calls anywhere in this file).

Fake-object conventions here follow tests/test_k1_ollama.py exactly (FakeOllama/FakeServer, a
make_lab(tmp_path, host, resume) helper building a real Lab subclass with an injected host_cfg and
args.Namespace), since harness/t2s_r2_session_growth.py now reuses t2s_k1_ollama.K1Lab and
t2s_k1_ollama.OllamaClient directly rather than a separate hand-rolled Lab/client pair."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import t2s_r2_session_growth as r2  # noqa: E402
import llama_server as ls  # noqa: E402


def word_count_fn(text):
    """Deterministic, fast token-count stand-in for tests: whitespace tokens."""
    return max(1, len(text.split()))


# ── R2.1: session generator ─────────────────────────────────────────────────────────────────────


class TestSessionCodeAndFacts:
    def test_session_code_deterministic_per_seed(self):
        assert r2.generate_session_code(1) == r2.generate_session_code(1)
        assert r2.generate_session_code(1) != r2.generate_session_code(2)

    def test_session_code_format(self):
        code = r2.generate_session_code(42)
        assert code.startswith("SC-")
        assert len(code) == len("SC-") + 8

    def test_facts_deterministic_and_distinct(self):
        facts_a = r2.pick_facts(7)
        facts_b = r2.pick_facts(7)
        assert facts_a == facts_b
        keys = [k for k, _ in facts_a]
        assert keys == list(r2.FACT_KEYS)
        values = [v for _, v in facts_a]
        assert len(set(values)) == 3

    def test_facts_differ_across_seeds(self):
        assert r2.pick_facts(1) != r2.pick_facts(2)


class TestSystemPrompt:
    def test_contains_all_5_rules_verbatim_markers(self):
        text, code, facts, canary = r2.build_system_prompt(5)
        assert "log_event" in text and "lookup_fact" in text
        assert "ZEBRA-7" in text
        assert "metres" in text
        assert code in text
        assert canary in text
        for key, value in facts:
            assert key in text and value in text

    def test_pure_function_of_seed(self):
        a = r2.build_system_prompt(11)
        b = r2.build_system_prompt(11)
        assert a == b

    def test_canary_distinct_from_session_code_and_facts(self):
        text, code, facts, canary = r2.build_system_prompt(5)
        assert canary != code
        assert canary not in (v for _, v in facts)
        assert canary.startswith("CANARY-")


class TestCanary:
    def test_deterministic_per_seed(self):
        assert r2.generate_canary(1) == r2.generate_canary(1)
        assert r2.generate_canary(1) != r2.generate_canary(2)

    def test_distinct_from_session_code(self):
        assert r2.generate_canary(1) != r2.generate_session_code(1)


class TestGenerateTurn:
    def test_odd_turn_is_lookup_task(self):
        turn = r2.generate_turn(1, seed=1, facts=r2.pick_facts(1), count_fn=word_count_fn)
        assert turn.tool_name == "lookup_fact"
        assert turn.tool_args["key"] in turn.user_text
        assert not turn.requires_length

    def test_even_turn_is_length_task(self):
        turn = r2.generate_turn(2, seed=1, facts=r2.pick_facts(1), count_fn=word_count_fn)
        assert turn.tool_name == "log_event"
        assert turn.requires_length

    def test_every_4th_turn_is_a_recall_turn_cycling_facts(self):
        facts = r2.pick_facts(1)
        t4 = r2.generate_turn(4, seed=1, facts=facts, count_fn=word_count_fn)
        t8 = r2.generate_turn(8, seed=1, facts=facts, count_fn=word_count_fn)
        t12 = r2.generate_turn(12, seed=1, facts=facts, count_fn=word_count_fn)
        assert t4.is_recall and t8.is_recall and t12.is_recall
        assert {t4.recall_key, t8.recall_key, t12.recall_key} == set(r2.FACT_KEYS)

    def test_non_4th_turn_is_not_recall(self):
        turn = r2.generate_turn(5, seed=1, facts=r2.pick_facts(1), count_fn=word_count_fn)
        assert not turn.is_recall
        assert turn.recall_key is None

    def test_canary_none_skips_canary_schedule_entirely(self):
        turn = r2.generate_turn(5, seed=1, facts=r2.pick_facts(1), count_fn=word_count_fn)
        assert not turn.canary_check
        assert turn.canary_expected is None

    def test_every_5th_turn_is_a_canary_check_when_canary_given(self):
        canary = r2.generate_canary(1)
        facts = r2.pick_facts(1)
        t5 = r2.generate_turn(5, seed=1, facts=facts, canary=canary, count_fn=word_count_fn)
        t10 = r2.generate_turn(10, seed=1, facts=facts, canary=canary, count_fn=word_count_fn)
        t6 = r2.generate_turn(6, seed=1, facts=facts, canary=canary, count_fn=word_count_fn)
        assert t5.canary_check and t5.canary_expected == canary
        assert t10.canary_check and t10.canary_expected == canary
        assert not t6.canary_check
        assert t6.canary_expected is None

    def test_canary_schedule_is_independent_of_recall_schedule(self):
        # turn 20 is both a multiple of 4 (recall) and of 5 (canary) -- both must fire together.
        canary = r2.generate_canary(1)
        facts = r2.pick_facts(1)
        t20 = r2.generate_turn(20, seed=1, facts=facts, canary=canary, count_fn=word_count_fn)
        assert t20.is_recall
        assert t20.canary_check

    def test_pure_function_of_idx_seed_facts(self):
        facts = r2.pick_facts(3)
        a = r2.generate_turn(6, seed=3, facts=facts, count_fn=word_count_fn)
        b = r2.generate_turn(6, seed=3, facts=facts, count_fn=word_count_fn)
        assert a == b

    def test_args_derivable_from_own_text_not_earlier_turns(self):
        facts = r2.pick_facts(9)
        t1 = r2.generate_turn(1, seed=9, facts=facts, count_fn=word_count_fn)
        t3 = r2.generate_turn(3, seed=9, facts=facts, count_fn=word_count_fn)
        # each lookup task's key is unique to that turn's own text
        assert t1.tool_args["key"] != t3.tool_args["key"]
        assert t1.tool_args["key"] in t1.user_text
        assert t3.tool_args["key"] in t3.user_text


class TestGenerateSession:
    def test_pure_function_of_seed(self):
        a = r2.generate_session(1, count_fn=word_count_fn, max_turns=10)
        b = r2.generate_session(1, count_fn=word_count_fn, max_turns=10)
        assert a == b

    def test_stops_at_max_turns(self):
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=10,
                                       max_prompt_tokens=10**9)
        assert len(session.turns) == 10
        assert [t.idx for t in session.turns] == list(range(1, 11))

    def test_stops_at_token_budget(self):
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=1000,
                                       max_prompt_tokens=500)
        assert len(session.turns) < 1000
        assert len(session.turns) >= 1

    def test_no_live_model_or_server_needed(self):
        # count_fn omitted entirely -> falls back to _approx_token_count, no tokenizer required
        session = r2.generate_session(1, max_turns=5)
        assert len(session.turns) == 5

    def test_default_generator_reaches_full_80_turns_or_160k_budget(self):
        session = r2.generate_session(1, count_fn=word_count_fn)
        assert len(session.turns) <= r2.MAX_TURNS
        assert len(session.turns) > 0


# ── R2.2: scorer ────────────────────────────────────────────────────────────────────────────────


def make_session_and_turn(idx=1, requires_length=False, is_recall=False, seed=1):
    session = r2.generate_session(seed, count_fn=word_count_fn, max_turns=max(idx, 12))
    turn = next(t for t in session.turns if t.idx == idx)
    return session, turn


class TestScoreTurn:
    def test_all_rules_pass_on_a_fully_compliant_output(self):
        session, turn = make_session_and_turn(idx=1)
        output = {
            "tool_calls": [{"name": "log_event", "arguments": {"event": "x"}},
                            {"name": turn.tool_name, "arguments": turn.tool_args}],
            "answer": f"looked it up{session.session_code}",
            "source": "lookup_fact",
        }
        score = r2.score_turn(turn, session.session_code, json.dumps(output), sent_tokens=100,
                               processed_tokens=100)
        assert score.rule1_json_keys
        assert score.rule2_log_event_called
        assert score.rule3_no_zebra
        assert score.rule4_metres
        assert score.rule5_session_code
        assert score.tool_args_match
        assert not score.any_rule_failed
        assert not score.silent_truncation

    def test_rule1_fails_on_non_json_output(self):
        session, turn = make_session_and_turn(idx=1)
        score = r2.score_turn(turn, session.session_code, "not json at all", 100, 100)
        assert not score.rule1_json_keys
        assert score.any_rule_failed

    def test_rule1_fails_missing_keys(self):
        session, turn = make_session_and_turn(idx=1)
        score = r2.score_turn(turn, session.session_code, json.dumps({"foo": "bar"}), 100, 100)
        assert not score.rule1_json_keys

    def test_rule2_fails_when_log_event_not_called(self):
        session, turn = make_session_and_turn(idx=1)
        output = {"tool_calls": [{"name": turn.tool_name, "arguments": turn.tool_args}],
                  "answer": f"x{session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert not score.rule2_log_event_called

    def test_rule3_fails_on_zebra_literal(self):
        session, turn = make_session_and_turn(idx=1)
        output = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}}],
                  "answer": f"ZEBRA-7{session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert not score.rule3_no_zebra

    def test_rule4_fails_on_forbidden_unit_for_length_turn(self):
        session, turn = make_session_and_turn(idx=2)  # even turn -> requires_length
        assert turn.requires_length
        output = {"tool_calls": [{"name": "log_event", "arguments": turn.tool_args}],
                  "answer": f"3 feet {session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert not score.rule4_metres

    def test_rule4_passes_for_non_length_turn_regardless_of_units(self):
        session, turn = make_session_and_turn(idx=1)  # odd turn -> not requires_length
        output = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}}],
                  "answer": f"3 feet {session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert score.rule4_metres

    def test_rule5_fails_when_session_code_missing(self):
        session, turn = make_session_and_turn(idx=1)
        output = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}}],
                  "answer": "no code here", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert not score.rule5_session_code

    def test_tool_args_mismatch_detected(self):
        session, turn = make_session_and_turn(idx=1)
        output = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}},
                                  {"name": turn.tool_name, "arguments": {"key": "WRONG"}}],
                  "answer": f"x{session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert not score.tool_args_match

    def test_fact_recall_exact_match(self):
        session, turn = make_session_and_turn(idx=4)
        assert turn.is_recall
        output = {"tool_calls": [{"name": "log_event", "arguments": turn.tool_args},
                                  {"name": turn.tool_name, "arguments": turn.tool_args}],
                  "answer": f"{turn.recall_value} {session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert score.fact_recall_ok is True

    def test_fact_recall_wrong_value(self):
        session, turn = make_session_and_turn(idx=4)
        output = {"tool_calls": [{"name": "log_event", "arguments": turn.tool_args}],
                  "answer": "WRONG-VALUE", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert score.fact_recall_ok is False

    def test_fact_recall_none_on_non_recall_turn(self):
        session, turn = make_session_and_turn(idx=1)
        output = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}}],
                  "answer": f"x{session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert score.fact_recall_ok is None

    def test_silent_truncation_signal(self):
        session, turn = make_session_and_turn(idx=1)
        score = r2.score_turn(turn, session.session_code, "not json", sent_tokens=5000,
                               processed_tokens=4000)
        assert score.silent_truncation

    def test_no_truncation_when_processed_gte_sent(self):
        session, turn = make_session_and_turn(idx=1)
        score = r2.score_turn(turn, session.session_code, "not json", sent_tokens=100,
                               processed_tokens=100)
        assert not score.silent_truncation

    def test_http_status_and_error_captured_verbatim(self):
        session, turn = make_session_and_turn(idx=1)
        score = r2.score_turn(turn, session.session_code, "x", 10, 10, http_status=500,
                               error_text="boom")
        assert score.http_status == 500
        assert score.error_text == "boom"


class TestScoreSession:
    def test_first_failure_and_error_ordering(self):
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=6)
        outputs = []
        for t in session.turns:
            if t.idx < 3:
                outputs.append({"output_text": "not json", "sent_tokens": 10,
                                 "processed_tokens": 10, "http_status": 200, "error_text": None})
            elif t.idx == 3:
                outputs.append({"output_text": "not json", "sent_tokens": 10,
                                 "processed_tokens": 10, "http_status": 500,
                                 "error_text": "server error"})
            else:
                outputs.append({"output_text": "not json", "sent_tokens": 10,
                                 "processed_tokens": 10, "http_status": 200, "error_text": None})
        result = r2.score_session(session, outputs)
        assert result["first_rule_violation_turn"] == 1
        assert result["first_error_turn"] == 3
        assert result["first_failure_turn"] == 1
        # failure (turn 1) happened before the error (turn 3) -> NOT preceded by an error
        assert result["error_surfaced_before_failure"] is False

    def test_error_before_failure_when_failure_is_later(self):
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=6)
        outputs = []
        for t in session.turns:
            answer = f"{t.recall_value} {session.session_code}" if t.is_recall else f"x{session.session_code}"
            good = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}},
                                     {"name": t.tool_name, "arguments": t.tool_args}],
                    "answer": answer, "source": "s"}
            if t.idx == 2:
                outputs.append({"output_text": json.dumps(good), "sent_tokens": 10,
                                 "processed_tokens": 10, "http_status": 500, "error_text": "boom"})
            elif t.idx == 5:
                outputs.append({"output_text": "not json", "sent_tokens": 10,
                                 "processed_tokens": 10, "http_status": 200, "error_text": None})
            else:
                outputs.append({"output_text": json.dumps(good), "sent_tokens": 10,
                                 "processed_tokens": 10, "http_status": 200, "error_text": None})
        result = r2.score_session(session, outputs)
        assert result["first_error_turn"] == 2
        assert result["first_failure_turn"] == 5
        assert result["error_surfaced_before_failure"] is True

    def test_survival_curve_shape(self):
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=4)
        clean_outputs = []
        for t in session.turns:
            answer = f"{t.recall_value} {session.session_code}" if t.is_recall else f"x{session.session_code}"
            good = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}},
                                     {"name": t.tool_name, "arguments": t.tool_args}],
                    "answer": answer, "source": "s"}
            clean_outputs.append({"output_text": json.dumps(good), "sent_tokens": 10,
                                   "processed_tokens": 10, "http_status": 200, "error_text": None})
        clean_result = r2.score_session(session, clean_outputs)
        curve = r2.survival_curve([clean_result], max_turns=4)
        assert curve == [1.0, 1.0, 1.0, 1.0]

    def test_survival_curve_empty_is_all_nan(self):
        curve = r2.survival_curve([], max_turns=3)
        assert len(curve) == 3
        assert all(c != c for c in curve)  # NaN != NaN


def _compliant_output(session, turn, *, log_event=True, canary_text=None):
    """Build a fully rule-1/3/4/5-compliant JSON answer string for `turn`, optionally including a
    verbatim canary reproduction and/or a log_event tool call. Shared helper for the new baseline-
    compliance / canary tests below.

    NOTE: log_event=False only has an effect on rule 2 for ODD-idx turns (lookup_fact tasks) --
    even-idx turns' own required tool (turn.tool_name) already IS "log_event" (see _length_task),
    so it is unconditionally present in tool_calls regardless of this flag."""
    answer = ""
    if canary_text is not None:
        answer += f"{canary_text} "
    if turn.is_recall:
        answer += f"{turn.recall_value} "
    answer += session.session_code
    tool_calls = [{"name": turn.tool_name, "arguments": turn.tool_args}]
    if log_event:
        tool_calls.append({"name": "log_event", "arguments": {"event": "e"}})
    return json.dumps({"tool_calls": tool_calls, "answer": answer, "source": "s"})


def _score_session_with(seed, max_turns, *, log_event_missing_turns=(), loaded_context_tokens=None,
                        cumulative_tokens_by_turn=None, canary_included_turns=None):
    """Build a real generated session and score it with per-turn overrides, for the baseline-
    compliance and canary-truncation tests: log_event_missing_turns makes rule 2 fail on those turn
    indices (everywhere else it passes); canary_included_turns, if given, restricts which canary-
    check turns actually reproduce the canary (default: all of them do)."""
    session = r2.generate_session(seed, count_fn=word_count_fn, max_turns=max_turns)
    turn_outputs = []
    for t in session.turns:
        include_canary = None
        if t.canary_check:
            include_canary = (session.canary if canary_included_turns is None
                               or t.idx in canary_included_turns else None)
        log_event = t.idx not in log_event_missing_turns
        text = _compliant_output(session, t, log_event=log_event, canary_text=include_canary)
        cum = (cumulative_tokens_by_turn or {}).get(t.idx, 100 * t.idx)
        turn_outputs.append({"output_text": text, "sent_tokens": 100, "processed_tokens": 100,
                              "http_status": 200, "error_text": None, "cumulative_tokens": cum})
    return r2.score_session(session, turn_outputs, loaded_context_tokens=loaded_context_tokens)


class TestRuleBaselineCompliance:
    def test_rule_passes_bar_when_compliant_every_turn(self):
        scored = _score_session_with(1, 10)
        compliance = r2.rule_baseline_compliance([scored])
        assert compliance == {r: True for r in r2.RULE_IDS}

    def test_rule2_fails_bar_when_missing_every_turn(self):
        scored = _score_session_with(1, 10, log_event_missing_turns=range(1, 11))
        compliance = r2.rule_baseline_compliance([scored])
        assert compliance["rule2_log_event_called"] is False
        assert compliance["rule1_json_keys"] is True

    def test_turn1_failure_in_control_arm_specifically_fails_the_bar(self):
        """The exact live scenario this problem exists for: rule 2 fails at turn 1 (and turn 3, to
        push the rate below the 90% bar -- turn 2 cannot be used for this since even turns'
        required tool IS log_event, so the "missing" flag has no effect there; see
        _compliant_output's docstring-equivalent note in its own body). 2 misses out of 10 -> 80%,
        under the bar."""
        scored = _score_session_with(1, 10, log_event_missing_turns=(1, 3))
        compliance = r2.rule_baseline_compliance([scored])
        assert compliance["rule2_log_event_called"] is False

    def test_canary_turns_excluded_from_rule2_denominator(self):
        # 2026-10-01 fix: turns 5 and 10 (CANARY_CHECK_EVERY=5) withhold tools entirely in the real
        # harness, so rule2 cannot be meaningfully graded there -- they must not count in its
        # denominator. With log_event missing only at turn 1, the window 1-10 has 8 gradable turns
        # (3,4,6,7,8,9 plus 1,2 -- 5 and 10 excluded), 7 of which pass: 7/8 = 87.5%, below the 90% bar.
        scored = _score_session_with(1, 20, log_event_missing_turns=(1,))
        compliance = r2.rule_baseline_compliance([scored])
        assert compliance["rule2_log_event_called"] is False

    def test_single_turn1_miss_over_a_wider_window_can_still_pass(self):
        # Window 1-14 has two canary turns (5, 10) excluded -> 12 gradable turns. One miss at turn 1:
        # 11/12 = 91.7%, clears the 90% bar -- confirms the exclusion only removes canary turns from
        # the denominator, it does not otherwise change how the bar is applied.
        scored = _score_session_with(1, 20, log_event_missing_turns=(1,))
        compliance = r2.rule_baseline_compliance([scored], min_turn=1, max_turn=14)
        assert compliance["rule2_log_event_called"] is True

    def test_report_gives_pass_rate_and_turn1_examples_for_failing_rules_only(self):
        # odd turns (1,3,5,7,9) are the only ones where "missing log_event" is actually effective
        # (see the previous test's note) -- all 5 missing -> exactly 50% overall pass rate.
        odd_missing = (1, 3, 5, 7, 9)
        scored_a = _score_session_with(1, 10, log_event_missing_turns=odd_missing)
        scored_b = _score_session_with(2, 10, log_event_missing_turns=odd_missing)
        report = r2.rule_baseline_compliance_report([scored_a, scored_b])
        assert set(report.keys()) == {"rule2_log_event_called"}
        entry = report["rule2_log_event_called"]
        assert entry["baseline_pass_rate"] == 0.5
        assert len(entry["example_turn1_outputs"]) == 2
        assert all(isinstance(o, str) and o for o in entry["example_turn1_outputs"])

    def test_report_empty_when_every_rule_clears_the_bar(self):
        scored = _score_session_with(1, 10)
        report = r2.rule_baseline_compliance_report([scored])
        assert report == {}


class TestNativeToolCallDetection:
    def test_native_tool_calls_used_when_detection_method_is_native(self):
        session, turn = make_session_and_turn(idx=1)
        # text-based "tool_calls" field deliberately omits log_event (would fail if graded as text)
        output = {"answer": f"x{session.session_code}", "source": "s"}
        native_calls = [{"function": {"name": "log_event", "arguments": {"event": "e"}}},
                         {"function": {"name": turn.tool_name, "arguments": turn.tool_args}}]
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100,
                               native_tool_calls=native_calls, tool_detection_method="native")
        assert score.rule2_log_event_called is True
        assert score.tool_args_match is True
        assert score.tool_call_detection_method == "native"

    def test_text_fallback_used_by_default(self):
        session, turn = make_session_and_turn(idx=1)
        output = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}},
                                  {"name": turn.tool_name, "arguments": turn.tool_args}],
                  "answer": f"x{session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert score.tool_call_detection_method == "text_fallback"
        assert score.rule2_log_event_called is True

    def test_native_empty_tool_calls_fails_rule2_even_with_compliant_text(self):
        # the model made no real tool call at all; a text "tool_calls" field, if present, must NOT
        # be used to paper over that once native detection is in effect.
        session, turn = make_session_and_turn(idx=1)
        output = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}}],
                  "answer": f"x{session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100,
                               native_tool_calls=[], tool_detection_method="native")
        assert score.rule2_log_event_called is False

    def test_ollama_tools_payload_shape(self):
        payload = r2.ollama_tools_payload()
        names = {p["function"]["name"] for p in payload}
        assert names == {"log_event", "lookup_fact"}
        assert all(p["type"] == "function" for p in payload)

    def test_run_turn_ollama_records_native_method_and_sends_tools(self, tmp_path):
        lab = make_lab(tmp_path)
        client = FakeOllama(tool_calls=[{"function": {"name": "log_event", "arguments": {"event": "e"}}}])
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=1)
        turn = session.turns[0]
        messages = [{"role": "system", "content": session.system_prompt},
                    {"role": "user", "content": r2.turn_message_content(turn)}]
        row = lab.run_turn_ollama("qwen3-4b-2507", "ollama_default", "as_is", session, turn, messages,
                                   ollama=client, count_fn=word_count_fn)
        assert row["tool_detection_method"] == "native"
        assert row["native_tool_calls"] == client.tool_calls
        assert client.calls[0]["tools"] is not None

    def test_run_turn_ollama_sends_explicit_keep_alive(self, tmp_path):
        """Found live 2026-10-01 (R2 positive control on evo-x2): without an explicit keep_alive,
        OllamaClient.chat omits the field and the model falls through to the server's own default --
        on evo-x2, hc.start_ollama_server() sets OLLAND_KEEP_ALIVE=0 (deliberately, for K1's clean-
        reload tier probes), which unloaded the model after every single R2 turn, breaking the
        cross-turn KV-cache reuse the whole module's design assumes and making
        get_loaded_context_ollama's /api/ps check always see nothing loaded. run_turn_ollama must
        always pass an explicit, long-lived keep_alive regardless of the server's own default."""
        lab = make_lab(tmp_path)
        client = FakeOllama()
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=1)
        turn = session.turns[0]
        messages = [{"role": "system", "content": session.system_prompt},
                    {"role": "user", "content": r2.turn_message_content(turn)}]
        lab.run_turn_ollama("qwen3-4b-2507", "ollama_default", "as_is", session, turn, messages,
                            ollama=client, count_fn=word_count_fn)
        assert client.calls[0]["keep_alive"] is not None
        assert client.calls[0]["keep_alive"] != 0

    def test_run_turn_ollama_withholds_tools_on_canary_check_turns(self, tmp_path):
        """2026-10-01 fix (R2 validity item 2): a canary-check turn must not also be offered tools --
        a model that calls a tool typically returns empty message.content in the same response
        (confirmed live: 3/3 raw turn-5 outputs on evo-x2 were a lone lookup_fact tool call with
        content=""), which makes canary_reproduced fail regardless of whether the model actually still
        remembers the canary. Withholding tools forces a text response so the canary check is not
        corrupted by this collision."""
        lab = make_lab(tmp_path)
        client = FakeOllama()
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=5)
        turn5 = next(t for t in session.turns if t.idx == 5)
        assert turn5.canary_check is True
        messages = [{"role": "system", "content": session.system_prompt},
                    {"role": "user", "content": r2.turn_message_content(turn5)}]
        lab.run_turn_ollama("qwen3-4b-2507", "ollama_default", "as_is", session, turn5, messages,
                            ollama=client, count_fn=word_count_fn)
        assert client.calls[0]["tools"] is None

    def test_run_turn_ollama_still_sends_tools_on_non_canary_turns(self, tmp_path):
        lab = make_lab(tmp_path)
        client = FakeOllama()
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=5)
        turn1 = session.turns[0]
        assert turn1.canary_check is False
        messages = [{"role": "system", "content": session.system_prompt},
                    {"role": "user", "content": r2.turn_message_content(turn1)}]
        lab.run_turn_ollama("qwen3-4b-2507", "ollama_default", "as_is", session, turn1, messages,
                            ollama=client, count_fn=word_count_fn)
        assert client.calls[0]["tools"] is not None

    def test_run_turn_ollama_makes_a_second_call_after_a_tool_call(self, tmp_path):
        """2026-10-01 redesign: a non-canary turn that returns a tool call in its first response
        must get a second call (tools withheld) to produce the final text answer, rather than
        scoring the empty-content tool-call response as the turn's output."""
        lab = make_lab(tmp_path)
        tool_call = {"function": {"name": "log_event", "arguments": {"event": "e"}}}
        client = FakeOllama(responses=[
            {"outcome": "ok", "status": 200, "message": "", "prompt_eval_count": 50,
             "error": None, "tool_calls": [tool_call]},
            {"outcome": "ok", "status": 200, "message": '{"answer": "final SC-X", "source": "s"}',
             "prompt_eval_count": 80, "error": None, "tool_calls": None},
        ])
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=1)
        turn = session.turns[0]
        messages = [{"role": "system", "content": session.system_prompt},
                    {"role": "user", "content": r2.turn_message_content(turn)}]
        row = lab.run_turn_ollama("qwen3-4b-2507", "ollama_default", "as_is", session, turn, messages,
                                   ollama=client, count_fn=word_count_fn)
        assert len(client.calls) == 2
        # call 2 must withhold tools and must include the assistant tool-call message plus a
        # synthetic tool-result message appended after the original 2 messages
        assert client.calls[1]["tools"] is None
        assert len(client.calls[1]["messages"]) == 4
        assert client.calls[1]["messages"][2]["role"] == "assistant"
        assert client.calls[1]["messages"][2]["tool_calls"] == [tool_call]
        assert client.calls[1]["messages"][3]["role"] == "tool"
        # the turn's final output_text is call 2's content, not call 1's empty content
        assert row["output_text"] == '{"answer": "final SC-X", "source": "s"}'
        # rule 2 is graded on call 1's tool_calls regardless of the second call
        assert row["native_tool_calls"] == [tool_call]
        assert row["processed_tokens"] == 80
        # sent_tokens counts every message across both calls, including the synthetic round trip
        assert row["sent_tokens"] == sum(word_count_fn(m["content"]) for m in client.calls[1]["messages"])

    def test_run_turn_ollama_skips_second_call_with_no_tool_call(self, tmp_path):
        lab = make_lab(tmp_path)
        client = FakeOllama(responses=[
            {"outcome": "ok", "status": 200, "message": '{"answer": "direct SC-X", "source": "s"}',
             "prompt_eval_count": 60, "error": None, "tool_calls": None},
        ])
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=1)
        turn = session.turns[0]
        messages = [{"role": "system", "content": session.system_prompt},
                    {"role": "user", "content": r2.turn_message_content(turn)}]
        row = lab.run_turn_ollama("qwen3-4b-2507", "ollama_default", "as_is", session, turn, messages,
                                   ollama=client, count_fn=word_count_fn)
        assert len(client.calls) == 1
        assert row["output_text"] == '{"answer": "direct SC-X", "source": "s"}'
        assert row["native_tool_calls"] is None

    def test_run_turn_llama_server_always_text_fallback(self, tmp_path):
        lab = make_lab(tmp_path)
        client = FakeLlamaServerSession()
        session = r2.generate_session(1, count_fn=word_count_fn, max_turns=1)
        turn = session.turns[0]
        row = lab.run_turn_llama_server(client, "qwen3-4b-2507", "llama_server_default_fit", "as_is",
                                         session, turn, log_text_fn=lambda: client.log_text)
        assert row["tool_detection_method"] == "text_fallback"
        assert row["native_tool_calls"] is None


class TestCanaryTruncationDetection:
    def test_canary_hit_no_truncation(self):
        scored = _score_session_with(1, 5, loaded_context_tokens=1_000_000)
        assert scored["truncation_detected_turn"] is None

    def test_canary_miss_without_context_exceeded_does_not_count(self):
        scored = _score_session_with(1, 5, canary_included_turns=set(),
                                      loaded_context_tokens=1_000_000,
                                      cumulative_tokens_by_turn={5: 10})
        assert scored["truncation_detected_turn"] is None

    def test_context_exceeded_without_canary_miss_does_not_count(self):
        scored = _score_session_with(1, 5, loaded_context_tokens=50,
                                      cumulative_tokens_by_turn={5: 10_000})
        assert scored["truncation_detected_turn"] is None

    def test_both_conditions_together_is_the_only_thing_that_fires(self):
        scored = _score_session_with(1, 5, canary_included_turns=set(), loaded_context_tokens=50,
                                      cumulative_tokens_by_turn={5: 10_000})
        assert scored["truncation_detected_turn"] == 5

    def test_no_loaded_context_tokens_means_no_verdict_ever(self):
        scored = _score_session_with(1, 5, canary_included_turns=set(),
                                      cumulative_tokens_by_turn={5: 10_000})
        assert scored["loaded_context_tokens"] is None
        assert scored["truncation_detected_turn"] is None


class TestLoadedContextHelpers:
    def test_get_loaded_context_ollama_matches_by_name(self):
        client = FakeOllama(context_length=32768)
        client.chat("qwen3:4b-instruct-2507", "")  # seed .calls so get_ps can report a name
        assert r2.get_loaded_context_ollama(client, "qwen3:4b-instruct-2507") == 32768

    def test_get_loaded_context_ollama_none_when_not_loaded(self):
        client = FakeOllama()
        client.chat("qwen3:4b-instruct-2507", "")
        assert r2.get_loaded_context_ollama(client, "qwen3:4b-instruct-2507") is None

    def test_get_loaded_context_llama_server_parses_n_ctx_slot(self):
        log_text = "I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 8192, n_ctx = 8192"
        assert r2.get_loaded_context_llama_server(log_text) == 8192

    def test_get_loaded_context_llama_server_none_when_absent(self):
        assert r2.get_loaded_context_llama_server("no such line here") is None


class TestKillCriterion:
    def test_killed_true_when_every_arm_shows_error_before_failure(self):
        result = {"first_failure_turn": 5, "error_surfaced_before_failure": True}
        out = r2.evaluate_kill_criterion({"arm_a": [result], "arm_b": [result]})
        assert out["killed"] is True

    def test_killed_false_when_any_arm_shows_silent_failure(self):
        silent = {"first_failure_turn": 5, "error_surfaced_before_failure": False}
        errored = {"first_failure_turn": 5, "error_surfaced_before_failure": True}
        out = r2.evaluate_kill_criterion({"arm_a": [errored], "arm_b": [silent]})
        assert out["killed"] is False
        assert out["per_arm"]["arm_b"]["shows_silent_failure"] is True
        assert out["per_arm"]["arm_a"]["shows_silent_failure"] is False

    def test_no_failure_sessions_do_not_count_as_silent(self):
        no_failure = {"first_failure_turn": None, "error_surfaced_before_failure": None}
        out = r2.evaluate_kill_criterion({"arm_a": [no_failure]})
        assert out["killed"] is True


# ── R2.3: arms ──────────────────────────────────────────────────────────────────────────────────


class TestArms:
    def test_applicable_arms_evo_t2s_excludes_arm_c(self):
        arms = r2.applicable_arms("evo-t2s")
        assert "ollama_ctx_32768_x2" not in arms
        assert "ollama_default" in arms
        assert "llama_server_default_fit" in arms

    def test_applicable_arms_evo_x2_includes_arm_c(self):
        arms = r2.applicable_arms("evo-x2")
        assert "ollama_ctx_32768_x2" in arms

    def test_apply_ollama_arm_default_has_no_num_ctx(self):
        options = r2.apply_ollama_arm("ollama_default", {"temperature": 0})
        assert "num_ctx" not in options
        assert options["temperature"] == 0

    def test_apply_ollama_arm_131072_sets_num_ctx(self):
        options = r2.apply_ollama_arm("ollama_ctx_131072")
        assert options["num_ctx"] == 131072

    def test_apply_ollama_arm_32768_x2_sets_num_ctx(self):
        options = r2.apply_ollama_arm("ollama_ctx_32768_x2")
        assert options["num_ctx"] == 32768

    def test_apply_ollama_arm_rejects_llama_server_arm(self):
        import pytest
        with pytest.raises(ValueError):
            r2.apply_ollama_arm("llama_server_c_131072")

    def test_build_llama_server_config_c_131072(self):
        cfg = r2.build_llama_server_config("llama_server_c_131072", exe="llama-server.exe",
                                            model_path="m.gguf", port=8181)
        assert cfg.ctx_size == 131072

    def test_build_llama_server_config_default_fit_uses_sentinel(self):
        cfg = r2.build_llama_server_config("llama_server_default_fit", exe="llama-server.exe",
                                            model_path="m.gguf", port=8181)
        assert cfg.ctx_size == r2.DEFAULT_FIT_CTX_SENTINEL

    def test_build_llama_server_cmd_omits_ctx_size_for_default_fit(self):
        cfg = r2.build_llama_server_config("llama_server_default_fit", exe="llama-server.exe",
                                            model_path="m.gguf", port=8181)
        cmd = r2.build_llama_server_cmd(cfg, "log.txt")
        assert "--ctx-size" not in cmd

    def test_build_llama_server_cmd_includes_ctx_size_for_fixed_arm(self):
        cfg = r2.build_llama_server_config("llama_server_c_131072", exe="llama-server.exe",
                                            model_path="m.gguf", port=8181)
        cmd = r2.build_llama_server_cmd(cfg, "log.txt")
        assert "--ctx-size" in cmd
        assert cmd[cmd.index("--ctx-size") + 1] == "131072"


class TestClassifyContextOverflow:
    def test_hard_error_on_exceed_context_size_error_body(self):
        body = '{"error": {"type": "exceed_context_size_error", "n_prompt_tokens": 9000, "n_ctx": 8192}}'
        result = r2.classify_context_overflow(log_text="", sent_tokens=9000, processed_tokens=9000,
                                               http_status=400, error_body=body)
        assert result == "hard_error"

    def test_context_shift_when_log_has_shift_line(self):
        result = r2.classify_context_overflow(log_text="... slot context shift ...",
                                               sent_tokens=9000, processed_tokens=9000)
        assert result == "context_shift"

    def test_silent_truncation_when_processed_lt_sent_and_no_error_or_shift(self):
        result = r2.classify_context_overflow(log_text="ordinary log", sent_tokens=9000,
                                               processed_tokens=8000)
        assert result == "silent_truncation"

    def test_ok_when_nothing_unusual(self):
        result = r2.classify_context_overflow(log_text="ordinary log", sent_tokens=100,
                                               processed_tokens=100)
        assert result == "ok"

    def test_reuses_real_llama_server_helpers(self):
        # Confirms this isn't a reimplementation: the same exception type flows through.
        with_shift = "0.00 W srv slot context shift, n_keep = 0"
        assert ls._log_has_context_shift(with_shift)
        assert r2.classify_context_overflow(log_text=with_shift, sent_tokens=1,
                                             processed_tokens=1) == "context_shift"


# ── R2 host wiring: reuses t2s_k1_ollama.require_host / harness/host_config.py ─────────────────────


class TestHostWiring:
    def test_require_host_is_the_real_k1_wrapper(self):
        cfg = r2.k1.require_host("evo-t2s")
        assert cfg["gpu_vendor"] == "intel"
        assert cfg["backend"] == "vulkan"

    def test_require_host_unknown_raises(self):
        try:
            r2.k1.require_host("not-a-real-host")
            assert False, "expected SystemExit"
        except SystemExit:
            pass

    def test_build_arg_parser_host_choices_match_k1(self):
        ap = r2.build_arg_parser()
        assert set(ap._option_string_actions["--host"].choices) == set(r2.k1.K1_HOST_EXTRAS)


# ── R2.4: memory-condition wiring (Occupier mirrors phase_memory_pressure's confirm pattern) ──────


class FakeServer:
    """Same shape as tests/test_k1_ollama.py's FakeServer: .start() -> {"ok", ...}, .stop()."""

    def __init__(self, ok=True, load_s=5.0):
        self.ok, self.load_s = ok, load_s
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return {"ok": self.ok, "load_s": self.load_s, "error": None if self.ok else "boom"}

    def stop(self):
        self.stopped = True


def make_lab(tmp_path, host="evo-x2", resume=None):
    host_cfg = r2.k1.require_host(host)
    args = argparse.Namespace(resume=resume, out_dir=str(tmp_path), ollama_port=11434)
    prov = {"git_head": "deadbeef", "committed": True, "problems": [], "files": []}
    return r2.R2SessionLab(args, host_cfg, prov)


class TestOccupier:
    def test_start_confirms_held_gb_within_tolerance(self, tmp_path):
        lab = make_lab(tmp_path)
        fake_srv = FakeServer(ok=True)
        avails = iter([100000.0, 100000.0 - 40 * 1024])  # 40 GB drop
        occ = r2.Occupier(lab, occupier_mi=object(), occupier_n_ctx=32768,
                           server_factory=lambda mi, n_ctx, tag: fake_srv,
                           avail_mb_fn=lambda: next(avails))
        result = occ.start()
        assert result["ok"] is True
        assert result["held_confirmed"] is True
        assert abs(result["held_gb"] - 40.0) < 0.01
        occ.stop()
        assert fake_srv.stopped is True

    def test_start_flags_unconfirmed_when_drop_too_small(self, tmp_path):
        lab = make_lab(tmp_path)
        fake_srv = FakeServer(ok=True)
        avails = iter([100000.0, 100000.0 - 5 * 1024])  # only 5 GB drop against a 40 GB target
        occ = r2.Occupier(lab, occupier_mi=object(), occupier_n_ctx=32768,
                           server_factory=lambda mi, n_ctx, tag: fake_srv,
                           avail_mb_fn=lambda: next(avails))
        result = occ.start()
        assert result["held_confirmed"] is False

    def test_start_reports_failure_without_raising(self, tmp_path):
        lab = make_lab(tmp_path)
        fake_srv = FakeServer(ok=False)
        occ = r2.Occupier(lab, occupier_mi=object(), occupier_n_ctx=32768,
                           server_factory=lambda mi, n_ctx, tag: fake_srv,
                           avail_mb_fn=lambda: 100000.0)
        result = occ.start()
        assert result["ok"] is False
        assert result["held_gb"] is None


class TestMemoryConditionWiring:
    def test_as_is_returns_none_and_starts_nothing(self):
        occ, result = r2.wire_memory_condition("as_is")
        assert occ is None
        assert result is None

    def test_occupied_40gb_starts_the_occupier(self, tmp_path):
        lab = make_lab(tmp_path)
        fake_srv = FakeServer(ok=True)
        occ, result = r2.wire_memory_condition(
            "occupied_40gb",
            occupier_factory=lambda: r2.Occupier(lab, occupier_mi=object(), occupier_n_ctx=32768,
                                                  server_factory=lambda mi, n_ctx, tag: fake_srv,
                                                  avail_mb_fn=lambda: 100000.0))
        assert fake_srv.started
        assert result["ok"] is True
        occ.stop()
        assert fake_srv.stopped

    def test_occupied_40gb_without_factory_raises(self):
        import pytest
        with pytest.raises(ValueError):
            r2.wire_memory_condition("occupied_40gb")

    def test_unknown_condition_raises(self):
        import pytest
        with pytest.raises(ValueError):
            r2.wire_memory_condition("not_a_real_condition")


# ── R2.8: hour estimate ─────────────────────────────────────────────────────────────────────────


class TestEstimateHours:
    def test_evo_t2s_excludes_arm_c_from_cell_count(self):
        est_t2s = r2.estimate_hours(machine="evo-t2s")
        est_x2 = r2.estimate_hours(machine="evo-x2")
        assert est_x2["n_cells"] > est_t2s["n_cells"]
        assert "ollama_ctx_32768_x2" not in est_t2s["applicable_arms"]

    def test_memory_pressure_condition_has_its_own_count(self):
        est = r2.estimate_hours(machine="evo-x2")
        assert set(est["by_memory_condition"]) == set(r2.MEMORY_CONDITIONS)
        assert est["by_memory_condition"]["as_is"]["cells"] > 0
        assert est["by_memory_condition"]["occupied_40gb"]["cells"] > 0

    def test_total_hours_positive_and_matches_cells_x_turns_x_seconds(self):
        est = r2.estimate_hours(machine="evo-x2", max_turns=80, seconds_per_call=30.0)
        expected = est["n_cells"] * 80 * 30.0 / 3600
        assert abs(est["total_hours"] - expected) < 1e-9


# ── Full stub dry run: R2SessionLab.phase_run_session against fakes, no real network/process ──────


class FakeOllama:
    """Same shape as tests/test_k1_ollama.py's FakeOllama, matching the real
    t2s_k1_ollama.OllamaClient.chat(model, prompt, num_ctx=None, max_tokens=64, keep_alive=None,
    messages=None) signature (including the messages= keyword this rebuild added for R2). Always
    answers compliantly except it never calls log_event, so rule 2 always fails from turn 1 -- lets
    the dry run exercise the scorer's failure paths too."""

    def __init__(self, tool_calls=None, context_length=None, responses=None):
        self.calls = []
        self.tool_calls = tool_calls  # injected native message.tool_calls for rule-2 native tests
        self.context_length = context_length  # injected /api/ps context_length for truncation tests
        # a queue of distinct canned responses, one per successive chat() call (2026-10-01, the
        # two-call agent-step redesign: a single tool_calls value can't model "call 1 returns a tool
        # call, call 2 returns the final text" -- responses, if given, pops one dict per call in
        # order; falls back to the single-tool_calls shape below once (or if never) exhausted.
        self._responses = list(responses) if responses is not None else None

    def chat(self, model, prompt, num_ctx=None, max_tokens=64, keep_alive=None, messages=None,
             tools=None):
        # snapshot messages (list(...)) since the caller keeps mutating the same list object turn
        # to turn -- storing the reference itself would make every recorded call alias the final,
        # fully-grown history instead of what was actually sent at that point in time.
        self.calls.append({"model": model, "num_ctx": num_ctx, "messages": list(messages or []),
                            "tools": tools, "keep_alive": keep_alive})
        if self._responses:
            return self._responses.pop(0)
        return {"outcome": "ok", "status": 200, "message": "not valid json output",
                "prompt_eval_count": 50, "error": None, "tool_calls": self.tool_calls}

    def get_ps(self):
        if self.context_length is None:
            return {"outcome": "ok", "models": []}
        # matches whatever model id the most recent chat() call actually used, so a test need not
        # guess the exact Ollama tag spelling a given model_id aliases to.
        name = self.calls[-1]["model"] if self.calls else "unknown"
        return {"outcome": "ok", "models": [{"name": name, "context_length": self.context_length}]}

    def unload(self, model):
        """Matches the real OllamaClient.unload: a chat() call with keep_alive=0, max_tokens=1 --
        found missing live 2026-10-01 when the pre-occupier unload fix called this on a FakeOllama
        that didn't have it, silently swallowed by the production code's own except Exception."""
        return self.chat(model, "", num_ctx=None, max_tokens=1, keep_alive=0)


class FakeLlamaServerSession:
    """Deterministic fake standing in for LlamaServerSession: raises ContextSizeError once prompts
    exceed a small fake context, otherwise echoes a compliant-looking (but still non-JSON, for
    simplicity) response."""

    def __init__(self, fake_ctx=200):
        self.fake_ctx = fake_ctx
        self.n_calls = 0
        self.n_tokenize_calls = 0
        self.log_text = "ordinary startup log, n_ctx_slot = 200"

    def tokenize(self, text):
        self.n_tokenize_calls += 1
        return len(text.split())

    def call(self, prompt, max_tokens):
        self.n_calls += 1
        n_tokens = len(prompt.split())
        if n_tokens > self.fake_ctx:
            raw = json.dumps({"error": {"type": "exceed_context_size_error",
                                         "n_prompt_tokens": n_tokens, "n_ctx": self.fake_ctx}})
            raise ls.ContextSizeError(n_prompt_tokens=n_tokens, n_ctx=self.fake_ctx, raw=raw)
        return ("not valid json output", 10.0, 5.0, n_tokens, 5, "stop", 0)


class TestStubDryRun:
    """End-to-end stub dry run: 8 turns, 1 seed, 1 model, all 5 arms, all against fakes."""

    def test_dry_run_all_5_arms_call_shape(self, tmp_path):
        lab = make_lab(tmp_path)
        model_id = "qwen3-4b-2507"
        seed = 20260901
        max_turns = 8
        call_counts = {}
        for arm_id in r2.ARM_ORDER:
            runtime = r2.ARMS[arm_id]["runtime"]
            client = FakeOllama() if runtime == "ollama" else FakeLlamaServerSession()
            kwargs = {"ollama": client} if runtime == "ollama" else {"llama_session": client}
            result = lab.phase_run_session(model_id=model_id, arm_id=arm_id,
                                            condition_id="as_is", seed=seed,
                                            count_fn=word_count_fn, max_turns=max_turns,
                                            log_text_fn=(lambda c=client: getattr(c, "log_text", "")),
                                            **kwargs)
            n_turns = len(result["rows"])
            n_calls = len(client.calls) if runtime == "ollama" else client.n_calls
            call_counts[arm_id] = {"chat_or_call_invocations": n_calls, "turns_run": n_turns}
            assert n_turns == max_turns
            assert n_calls == max_turns

        # Exact call-shape breakdown by arm for 8 turns x 1 seed x 1 model x 5 arms.
        assert call_counts == {
            "ollama_default": {"chat_or_call_invocations": 8, "turns_run": 8},
            "ollama_ctx_131072": {"chat_or_call_invocations": 8, "turns_run": 8},
            "ollama_ctx_32768_x2": {"chat_or_call_invocations": 8, "turns_run": 8},
            "llama_server_default_fit": {"chat_or_call_invocations": 8, "turns_run": 8},
            "llama_server_c_131072": {"chat_or_call_invocations": 8, "turns_run": 8},
        }
        total_calls = sum(v["chat_or_call_invocations"] for v in call_counts.values())
        assert total_calls == 40  # 8 turns x 5 arms

    def test_dry_run_writes_one_jsonl_row_per_turn(self, tmp_path):
        """5 r2_turn rows (one per turn) plus 6 item_done rows (one per turn, for resumability, plus
        one for the whole cell) -- see build_item_id/R2SessionLab.item_done."""
        lab = make_lab(tmp_path)
        client = FakeOllama()
        lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="ollama_default",
                               condition_id="as_is", seed=1, ollama=client,
                               count_fn=word_count_fn, max_turns=5)
        lines = Path(lab.rows_path).read_text(encoding="utf-8").strip().splitlines()
        rows = [json.loads(line) for line in lines]
        turn_rows = [r for r in rows if r["record"] == "r2_turn"]
        done_rows = [r for r in rows if r["record"] == "item_done"]
        assert len(turn_rows) == 5
        assert len(done_rows) == 6  # 5 per-turn markers + 1 whole-cell marker
        assert len(rows) == 11

    def test_dry_run_ollama_arm_sends_growing_message_history(self, tmp_path):
        """The reason OllamaClient.chat needed a messages= keyword at all: turn N's call must carry
        every earlier turn's user/assistant messages, not just that turn's own prompt."""
        lab = make_lab(tmp_path)
        client = FakeOllama()
        lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="ollama_default",
                               condition_id="as_is", seed=1, ollama=client,
                               count_fn=word_count_fn, max_turns=4)
        msg_counts = [len(c["messages"]) for c in client.calls]
        assert msg_counts == [2, 4, 6, 8]  # system+user, then +assistant+user each turn

    def test_dry_run_with_occupied_40gb_condition_starts_and_stops_occupier(self, tmp_path):
        lab = make_lab(tmp_path)
        fake_srv = FakeServer(ok=True)
        client = FakeOllama()
        lab.phase_run_session(
            model_id="qwen3-4b-2507", arm_id="ollama_default", condition_id="occupied_40gb",
            seed=1, ollama=client, count_fn=word_count_fn, max_turns=3,
            occupier_factory=lambda: r2.Occupier(lab, occupier_mi=object(), occupier_n_ctx=32768,
                                                  server_factory=lambda mi, n_ctx, tag: fake_srv,
                                                  avail_mb_fn=lambda: 100000.0))
        assert fake_srv.started
        assert fake_srv.stopped

    def test_occupied_40gb_unloads_ollama_before_starting_the_occupier(self, tmp_path):
        """Found live 2026-10-01 (evo-x2): the occupier starts its own llama-server-shaped process via
        t2s_lab.Server.start(), which refuses unconditionally if ANY process named llama-server.exe is
        already running -- including Ollama's own internal engine, still alive via run_turn_ollama's
        keep_alive=30m from an earlier cell in the same run. phase_run_session must unload any Ollama
        model (via self.ollama, not the possibly-None ollama= param) before the occupier ever starts."""
        lab = make_lab(tmp_path)
        lab.ollama = FakeOllama()  # self.ollama, not the client passed as ollama= below
        fake_srv = FakeServer(ok=True)
        client = FakeOllama()
        lab.phase_run_session(
            model_id="qwen3-4b-2507", arm_id="ollama_default", condition_id="occupied_40gb",
            seed=1, ollama=client, count_fn=word_count_fn, max_turns=3,
            occupier_factory=lambda: r2.Occupier(lab, occupier_mi=object(), occupier_n_ctx=32768,
                                                  server_factory=lambda mi, n_ctx, tag: fake_srv,
                                                  avail_mb_fn=lambda: 100000.0))
        unload_calls = [c for c in lab.ollama.calls if c.get("keep_alive") == 0]
        assert unload_calls, "expected an unload() call (keep_alive=0) on lab.ollama before the occupier started"

    def test_as_is_condition_does_not_unload_ollama(self, tmp_path):
        """The unload is specific to occupied_40gb -- an as_is cell must not pay for an unnecessary
        unload/reload cycle."""
        lab = make_lab(tmp_path)
        lab.ollama = FakeOllama()
        client = FakeOllama()
        lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="ollama_default", condition_id="as_is",
                              seed=1, ollama=client, count_fn=word_count_fn, max_turns=1)
        assert lab.ollama.calls == []

    def test_dry_run_llama_server_arm_classifies_hard_error_on_overflow(self, tmp_path):
        lab = make_lab(tmp_path)
        client = FakeLlamaServerSession(fake_ctx=1)  # tiny -> every turn overflows
        result = lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="llama_server_c_131072",
                                        condition_id="as_is", seed=1, llama_session=client,
                                        count_fn=word_count_fn, max_turns=3,
                                        log_text_fn=lambda: client.log_text)
        classifications = {r["overflow_classification"] for r in result["rows"]}
        assert classifications == {"hard_error"}
        assert all(r["http_status"] == 400 for r in result["rows"])


class FakeSilentTruncationSession:
    """A stub server that never errors (HTTP 200 always) but silently returns fewer processed tokens
    than were sent once a fake threshold is crossed -- the exact "no error, but data vanished" case
    item 2's own instruction calls for, distinct from FakeLlamaServerSession's hard_error case."""

    def __init__(self, fake_ctx=200):
        self.fake_ctx = fake_ctx
        self.n_calls = 0
        self.log_text = "ordinary startup log, no context shift line"

    def tokenize(self, text):
        return len(text.split())

    def call(self, prompt, max_tokens):
        self.n_calls += 1
        n_tokens = len(prompt.split())
        tin = min(n_tokens, self.fake_ctx)  # silently clamps instead of raising
        return ("truncated but still 200", 10.0, 5.0, tin, 5, "stop", 0)


class TestSilentTruncationClassification:
    def test_silent_truncation_classified_correctly_and_no_error_surfaced(self, tmp_path):
        lab = make_lab(tmp_path)
        client = FakeSilentTruncationSession(fake_ctx=1)
        result = lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="llama_server_default_fit",
                                        condition_id="as_is", seed=1, llama_session=client,
                                        count_fn=word_count_fn, max_turns=6,
                                        log_text_fn=lambda: client.log_text)
        rows = result["rows"]
        assert any(r["overflow_classification"] == "silent_truncation" for r in rows)
        assert all(r["http_status"] == 200 for r in rows)
        assert all(r["error_text"] is None for r in rows)


# ── Resumability: per-turn and per-cell item_done tracking ─────────────────────────────────────────


class TestResumability:
    def test_cell_already_done_is_skipped_entirely(self, tmp_path):
        lab = make_lab(tmp_path)
        client = FakeOllama()
        lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="ollama_default",
                               condition_id="as_is", seed=1, ollama=client,
                               count_fn=word_count_fn, max_turns=3)
        n_calls_first_pass = len(client.calls)
        result = lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="ollama_default",
                                        condition_id="as_is", seed=1, ollama=client,
                                        count_fn=word_count_fn, max_turns=3)
        assert result == {"skipped": True, "reason": "cell already done (resume)"}
        assert len(client.calls) == n_calls_first_pass  # no new calls made

    def test_resumed_mid_session_only_calls_for_remaining_turns(self, tmp_path):
        """Simulates a crash after turn 2 of a 5-turn session (item_done written per turn, but the
        whole-cell marker never got written) by manually replaying what a first, truncated attempt
        would have left on disk, then resuming: the resumed run must not re-call the model for the
        turns already marked done, and must replay their rows (verbatim) into the final result."""
        lab = make_lab(tmp_path)
        client = FakeOllama()

        # First "attempt": manually stop after 2 turns by calling with max_turns=2 (this writes the
        # per-turn item_done markers for turns 1-2, and also the whole-cell marker for THIS 2-turn
        # session -- to simulate a genuine crash mid-way through a longer session instead, delete the
        # cell-level marker's row after the fact and keep only the per-turn ones).
        lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="ollama_default",
                               condition_id="as_is", seed=1, ollama=client,
                               count_fn=word_count_fn, max_turns=2)
        n_calls_after_partial = len(client.calls)
        assert n_calls_after_partial == 2

        # Simulate the crash: rebuild a fresh lab pointed at the same stem/resume, but first strip the
        # whole-cell item_done record out of self.done (the per-turn ones are real progress; the
        # cell-level one is what a genuine crash would never have reached).
        cell_id = r2.build_item_id("qwen3-4b-2507", "ollama_default", "as_is", 1)
        lab.done.discard(cell_id)

        result = lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="ollama_default",
                                        condition_id="as_is", seed=1, ollama=client,
                                        count_fn=word_count_fn, max_turns=5)
        assert result["skipped"] is False
        assert len(result["rows"]) == 5  # 2 replayed + 3 freshly run
        assert len(client.calls) == n_calls_after_partial + 3  # only the 3 new turns called the model

    def test_missing_replay_row_raises_rather_than_silently_redoing_or_skipping(self, tmp_path):
        """If a turn's item_done marker exists but its r2_turn row is somehow missing (a corrupted or
        hand-edited jsonl), this must fail loudly rather than guess -- silently redoing the turn could
        double-count a call, and silently skipping it would produce a session with a hole in it."""
        lab = make_lab(tmp_path)
        turn_id = r2.build_item_id("qwen3-4b-2507", "ollama_default", "as_is", 1, turn_idx=1)
        lab.item_done(turn_id)  # marker written, but no matching r2_turn row exists
        client = FakeOllama()
        import pytest
        with pytest.raises(RuntimeError, match="resume state is inconsistent"):
            lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="ollama_default",
                                   condition_id="as_is", seed=1, ollama=client,
                                   count_fn=word_count_fn, max_turns=3)


# ── Stale-server guard: checked before every llama-server turn ─────────────────────────────────────


class FakeGuard:
    def __init__(self, ok=True, port=8385, pid=1234):
        self.ok, self.port, self.pid = ok, port, pid
        self.n_checks = 0

    def check(self):
        self.n_checks += 1
        return self.ok, [self.pid] if self.ok else [9999]


class TestStaleServerGuard:
    def test_guard_checked_before_every_turn(self, tmp_path):
        lab = make_lab(tmp_path)
        client = FakeLlamaServerSession()
        guard = FakeGuard(ok=True)
        lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="llama_server_default_fit",
                               condition_id="as_is", seed=1, llama_session=client,
                               count_fn=word_count_fn, max_turns=4,
                               log_text_fn=lambda: client.log_text, guard=guard)
        assert guard.n_checks == 4

    def test_guard_failure_stops_the_session_immediately(self, tmp_path):
        lab = make_lab(tmp_path)
        client = FakeLlamaServerSession()
        guard = FakeGuard(ok=False)
        import pytest
        with pytest.raises(RuntimeError, match="stale-server guard failed"):
            lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="llama_server_default_fit",
                                   condition_id="as_is", seed=1, llama_session=client,
                                   count_fn=word_count_fn, max_turns=4,
                                   log_text_fn=lambda: client.log_text, guard=guard)
        assert client.n_calls == 0  # the guard fires before the call, not after


# ── Problem 3: positive-control stubs (fake runtimes that silently drop content past a fake 8192- ──
# ── token boundary) confirming the new detector actually fires, and at the right turn ───────────────


class FakePositiveControlOllama:
    """A fake Ollama client that answers compliantly (including the canary, when asked) as long as
    the growing conversation's cumulative token count stays within `fake_ctx`; once it is exceeded,
    it silently drops the canary from its answer (everything else about the answer stays compliant)
    -- simulating exactly the "no error, but the model can no longer see its own system prompt"
    failure mode problem 2's detector exists to catch. session_code/canary are the real session's own
    values (closed over by the test), so _compliant_output-style grading still works."""

    def __init__(self, session_code, canary, fake_ctx=8192):
        self.session_code, self.canary, self.fake_ctx = session_code, canary, fake_ctx
        self.calls = []

    def chat(self, model, prompt, num_ctx=None, max_tokens=64, keep_alive=None, messages=None,
             tools=None):
        self.calls.append({"model": model, "messages": list(messages or [])})
        sent_tokens = sum(len(m["content"].split()) for m in (messages or []))
        last_user = next((m["content"] for m in reversed(messages or []) if m["role"] == "user"), "")
        asked_canary = "canary phrase" in last_user
        include_canary = asked_canary and sent_tokens <= self.fake_ctx
        answer = (f"{self.canary} " if include_canary else "") + f"ok {self.session_code}"
        body = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}}],
                "answer": answer, "source": "s"}
        return {"outcome": "ok", "status": 200, "message": json.dumps(body),
                "prompt_eval_count": 50, "error": None, "tool_calls": None}

    def get_ps(self):
        name = self.calls[-1]["model"] if self.calls else "unknown"
        return {"outcome": "ok", "models": [{"name": name, "context_length": self.fake_ctx}]}


class FakePositiveControlLlamaServer:
    """llama-server equivalent of FakePositiveControlOllama: no growing message history (matches
    run_turn_llama_server's real send shape), so it tracks its own running cumulative token count
    across calls the same way phase_run_session's cumulative_tokens_before/row["cumulative_tokens"]
    does, and compares that running total (not just this call's own prompt) against fake_ctx."""

    def __init__(self, session_code, canary, fake_ctx=8192):
        self.session_code, self.canary, self.fake_ctx = session_code, canary, fake_ctx
        self.log_text = f"ordinary startup log, n_ctx_slot = {fake_ctx}"
        self.cumulative = 0
        self.n_calls = 0

    def tokenize(self, text):
        return len(text.split())

    def call(self, prompt, max_tokens):
        self.n_calls += 1
        n_tokens = len(prompt.split())
        self.cumulative += n_tokens
        asked_canary = "canary phrase" in prompt
        include_canary = asked_canary and self.cumulative <= self.fake_ctx
        answer = (f"{self.canary} " if include_canary else "") + f"ok {self.session_code}"
        body = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}}],
                "answer": answer, "source": "s"}
        return (json.dumps(body), 10.0, 5.0, n_tokens, 5, "stop", 0)


class TestPositiveControlStubs:
    """Problem 3's two required positive controls, run against fakes (no real process): confirms the
    loaded-context-vs-cumulative-tokens + canary-miss detector fires once cumulative tokens exceed a
    fake 8192-token boundary, and reports the exact turn."""

    def test_ollama_positive_control_fires_once_past_8192_tokens(self, tmp_path):
        lab = make_lab(tmp_path)
        session = r2.generate_session(20260901, count_fn=word_count_fn, max_turns=10)
        client = FakePositiveControlOllama(session.session_code, session.canary, fake_ctx=8192)
        result = lab.phase_run_session(model_id="qwen3-4b-2507",
                                        arm_id="ollama_ctx_8192_positive_control",
                                        condition_id="as_is", seed=20260901, ollama=client,
                                        count_fn=word_count_fn, max_turns=10)
        scored = result["scored"]
        assert scored["loaded_context_tokens"] == 8192
        canary_rows = [r for r in result["rows"] if r["turn_idx"] % r2.CANARY_CHECK_EVERY == 0]
        expected_turn = next((r["turn_idx"] for r in canary_rows if r["cumulative_tokens"] > 8192), None)
        assert expected_turn is not None  # the filler growth must actually cross 8192 within 10 turns
        assert scored["truncation_detected_turn"] == expected_turn

    def test_llama_server_positive_control_fires_once_past_8192_tokens(self, tmp_path):
        lab = make_lab(tmp_path)
        session = r2.generate_session(20260901, count_fn=word_count_fn, max_turns=10)
        client = FakePositiveControlLlamaServer(session.session_code, session.canary, fake_ctx=8192)
        result = lab.phase_run_session(model_id="qwen3-4b-2507",
                                        arm_id="llama_server_c_8192_positive_control",
                                        condition_id="as_is", seed=20260901, llama_session=client,
                                        count_fn=word_count_fn, max_turns=10,
                                        log_text_fn=lambda: client.log_text)
        scored = result["scored"]
        assert scored["loaded_context_tokens"] == 8192
        canary_rows = [r for r in result["rows"] if r["turn_idx"] % r2.CANARY_CHECK_EVERY == 0]
        expected_turn = next((r["turn_idx"] for r in canary_rows if r["cumulative_tokens"] > 8192), None)
        assert expected_turn is not None
        assert scored["truncation_detected_turn"] == expected_turn

    def test_positive_control_arms_are_not_in_normal_enumeration(self):
        assert "ollama_ctx_8192_positive_control" not in r2.ARM_ORDER
        assert "llama_server_c_8192_positive_control" not in r2.ARM_ORDER
        assert "ollama_ctx_8192_positive_control" not in r2.applicable_arms("evo-x2")


# ── CLI: --smoke dry run reports the call-shape count without starting any real process ────────────


class TestSmokeCLI:
    def test_smoke_reports_call_shape_and_writes_no_rows(self, tmp_path, capsys):
        shape = r2.main(["--host", "evo-x2", "--smoke", "--models", "qwen3-4b-2507",
                          "--arms", "ollama_default", "--seeds", "20260901", "--max-turns", "6",
                          "--memory-conditions", "as_is", "--out-dir", str(tmp_path)])
        assert shape["total_calls"] == 6  # 1 model x 1 arm x 1 condition x 1 seed x 6 turns
        assert shape["by_arm"]["ollama_default"] == {"cells": 1, "calls": 6}
        assert list(tmp_path.glob("*.jsonl")) == []  # smoke writes no rows at all

    def test_smoke_rejects_an_inapplicable_arm_for_the_host(self, tmp_path):
        import pytest
        with pytest.raises(SystemExit, match="not applicable"):
            r2.main(["--host", "evo-t2s", "--smoke", "--arms", "ollama_ctx_32768_x2",
                     "--out-dir", str(tmp_path)])
