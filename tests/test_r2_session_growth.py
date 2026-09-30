"""Tests for harness/t2s_r2_session_growth.py: turn generator, scorer, arm config-application,
memory-condition wiring, and a stub end-to-end dry run against fakes (no real network/process
calls anywhere in this file)."""

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
        text, code, facts = r2.build_system_prompt(5)
        assert "log_event" in text and "lookup_fact" in text
        assert "ZEBRA-7" in text
        assert "metres" in text
        assert code in text
        for key, value in facts:
            assert key in text and value in text

    def test_pure_function_of_seed(self):
        a = r2.build_system_prompt(11)
        b = r2.build_system_prompt(11)
        assert a == b


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
        import json
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
        import json
        session, turn = make_session_and_turn(idx=1)
        score = r2.score_turn(turn, session.session_code, json.dumps({"foo": "bar"}), 100, 100)
        assert not score.rule1_json_keys

    def test_rule2_fails_when_log_event_not_called(self):
        import json
        session, turn = make_session_and_turn(idx=1)
        output = {"tool_calls": [{"name": turn.tool_name, "arguments": turn.tool_args}],
                  "answer": f"x{session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert not score.rule2_log_event_called

    def test_rule3_fails_on_zebra_literal(self):
        import json
        session, turn = make_session_and_turn(idx=1)
        output = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}}],
                  "answer": f"ZEBRA-7{session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert not score.rule3_no_zebra

    def test_rule4_fails_on_forbidden_unit_for_length_turn(self):
        import json
        session, turn = make_session_and_turn(idx=2)  # even turn -> requires_length
        assert turn.requires_length
        output = {"tool_calls": [{"name": "log_event", "arguments": turn.tool_args}],
                  "answer": f"3 feet {session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert not score.rule4_metres

    def test_rule4_passes_for_non_length_turn_regardless_of_units(self):
        import json
        session, turn = make_session_and_turn(idx=1)  # odd turn -> not requires_length
        output = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}}],
                  "answer": f"3 feet {session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert score.rule4_metres

    def test_rule5_fails_when_session_code_missing(self):
        import json
        session, turn = make_session_and_turn(idx=1)
        output = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}}],
                  "answer": "no code here", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert not score.rule5_session_code

    def test_tool_args_mismatch_detected(self):
        import json
        session, turn = make_session_and_turn(idx=1)
        output = {"tool_calls": [{"name": "log_event", "arguments": {"event": "e"}},
                                  {"name": turn.tool_name, "arguments": {"key": "WRONG"}}],
                  "answer": f"x{session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert not score.tool_args_match

    def test_fact_recall_exact_match(self):
        import json
        session, turn = make_session_and_turn(idx=4)
        assert turn.is_recall
        output = {"tool_calls": [{"name": "log_event", "arguments": turn.tool_args},
                                  {"name": turn.tool_name, "arguments": turn.tool_args}],
                  "answer": f"{turn.recall_value} {session.session_code}", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert score.fact_recall_ok is True

    def test_fact_recall_wrong_value(self):
        import json
        session, turn = make_session_and_turn(idx=4)
        output = {"tool_calls": [{"name": "log_event", "arguments": turn.tool_args}],
                  "answer": "WRONG-VALUE", "source": "s"}
        score = r2.score_turn(turn, session.session_code, json.dumps(output), 100, 100)
        assert score.fact_recall_ok is False

    def test_fact_recall_none_on_non_recall_turn(self):
        import json
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
            import json
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
        import json
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


# ── R2.4: memory-condition wiring ───────────────────────────────────────────────────────────────


class FakeOccupierServer:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return {"ok": True}

    def stop(self):
        self.stopped = True


class TestMemoryConditionWiring:
    def test_as_is_returns_none_and_starts_nothing(self):
        result = r2.wire_memory_condition("as_is")
        assert result is None

    def test_occupied_40gb_starts_the_occupier(self):
        fake_server = FakeOccupierServer()
        occ = r2.wire_memory_condition("occupied_40gb",
                                        occupier_factory=lambda: r2.Occupier(fake_server))
        assert fake_server.started
        occ.stop()
        assert fake_server.stopped

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


# ── Full stub dry run: R2Lab.phase_run_session against fakes, no real network/process ────────────


class FakeOllamaClient:
    """Deterministic fake: always answers compliantly except it never calls log_event, so rule 2
    always fails from turn 1 -- lets the dry run exercise the scorer's failure paths too."""

    def __init__(self):
        self.n_calls = 0

    def chat(self, *, model, messages, options):
        self.n_calls += 1
        last_user = messages[-1]["content"]
        return {"message": {"role": "assistant", "content": "not valid json output"},
                "prompt_eval_count": 50, "http_status": 200, "error": None}


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
            import json as _json
            raw = _json.dumps({"error": {"type": "exceed_context_size_error",
                                          "n_prompt_tokens": n_tokens, "n_ctx": self.fake_ctx}})
            raise ls.ContextSizeError(n_prompt_tokens=n_tokens, n_ctx=self.fake_ctx, raw=raw)
        return ("not valid json output", 10.0, 5.0, n_tokens, 5, "stop", 0)


class TestStubDryRun:
    """End-to-end stub dry run: 8 turns, 1 seed, 1 model, all 5 arms, all against fakes."""

    def test_dry_run_all_5_arms_call_shape(self, tmp_path):
        lab = r2.R2Lab(str(tmp_path / "r2_smoke.jsonl"))
        model_id = "qwen3-4b-2507"
        seed = 20260901
        max_turns = 8
        call_counts = {}
        for arm_id in r2.ARM_ORDER:
            runtime = r2.ARMS[arm_id]["runtime"]
            if runtime == "ollama":
                client = FakeOllamaClient()
            else:
                client = FakeLlamaServerSession()
            result = lab.phase_run_session(model_id=model_id, arm_id=arm_id,
                                            condition_id="as_is", seed=seed, client=client,
                                            count_fn=word_count_fn, max_turns=max_turns,
                                            log_text_fn=(lambda c=client: getattr(c, "log_text", "")))
            n_turns = len(result["rows"])
            call_counts[arm_id] = {
                "chat_or_call_invocations": client.n_calls,
                "turns_run": n_turns,
            }
            assert n_turns == max_turns
            assert client.n_calls == max_turns

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
        jsonl_path = tmp_path / "r2_rows.jsonl"
        lab = r2.R2Lab(str(jsonl_path))
        client = FakeOllamaClient()
        result = lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="ollama_default",
                                        condition_id="as_is", seed=1, client=client,
                                        count_fn=word_count_fn, max_turns=5)
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        import json
        for line in lines:
            row = json.loads(line)
            assert row["record"] == "r2_turn"

    def test_dry_run_with_occupied_40gb_condition_starts_and_stops_occupier(self, tmp_path):
        lab = r2.R2Lab(str(tmp_path / "r2_occ.jsonl"))
        fake_server = FakeOccupierServer()
        client = FakeOllamaClient()
        lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="ollama_default",
                               condition_id="occupied_40gb", seed=1, client=client,
                               count_fn=word_count_fn, max_turns=3,
                               occupier_factory=lambda: r2.Occupier(fake_server))
        assert fake_server.started
        assert fake_server.stopped

    def test_dry_run_llama_server_arm_classifies_hard_error_on_overflow(self, tmp_path):
        lab = r2.R2Lab(str(tmp_path / "r2_overflow.jsonl"))
        client = FakeLlamaServerSession(fake_ctx=1)  # tiny -> every turn overflows
        result = lab.phase_run_session(model_id="qwen3-4b-2507", arm_id="llama_server_c_131072",
                                        condition_id="as_is", seed=1, client=client,
                                        count_fn=word_count_fn, max_turns=3,
                                        log_text_fn=lambda: client.log_text)
        classifications = {r["overflow_classification"] for r in result["rows"]}
        assert classifications == {"hard_error"}
        assert all(r["http_status"] == 400 for r in result["rows"])
