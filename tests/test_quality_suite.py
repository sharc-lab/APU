"""Tests for harness/quality_suite.py.

All tests run against a stub Server (crude length-based tokenize, chat via a rigged callable) --
no real llama-server process, same convention as the mock-only tests in tests/test_llama_server.py
and the sys.path bootstrap used in tests/test_server_guard.py for harness's flat-import modules.

The "report both" real-model control described in the task (does a real qwen3-8b show the expected
degradation pattern) is out of scope here and needs a live run; these tests only check that the
harness's own scoring/truncation/dispatch code paths behave as designed, using a fully deterministic
stub chat function.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import quality_suite as qs  # noqa: E402


class StubServer:
    """Duck-types harness/t2s_lab.py's Server: .tokenize(text) -> int, .chat(prompt, max_tokens,
    ignore_eos) -> dict with outcome/output. chat_fn(prompt, max_tokens, ignore_eos) returns the
    output string, or None to simulate a call that errored."""

    def __init__(self, chat_fn):
        self.chat_fn = chat_fn
        self.calls = []

    def tokenize(self, text):
        return max(len(text) // 4, 1)  # deterministic, monotonic in len(text); no real tokenizer needed

    def chat(self, prompt, max_tokens, ignore_eos):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "ignore_eos": ignore_eos})
        out = self.chat_fn(prompt, max_tokens, ignore_eos)
        if out is None:
            return {"outcome": "error", "output": None, "error": "stub error"}
        return {"outcome": "ok", "output": out}


TARGET_TOKENS = 2000


# ---------------------------------------------------------------- build_task generation

class TestBuildTaskAtTargetLength:
    @pytest.mark.parametrize("task_type", qs.TASK_TYPES)
    def test_generatable_at_2k_tokens(self, task_type):
        task = qs.build_task(task_type, TARGET_TOKENS, seed=42)
        assert task.task_type == task_type
        assert isinstance(task.prompt, str) and task.prompt
        assert task.scorer in ("exact", "set_f1", "json_keys", "tool_call")
        # sanity check the prompt is in the right ballpark, not off by an order of magnitude
        approx_tokens = len(task.prompt) / 5.03
        assert approx_tokens > TARGET_TOKENS * 0.4

    def test_deterministic_given_same_seed(self):
        t1 = qs.build_task("niah_multikey", TARGET_TOKENS, seed=7)
        t2 = qs.build_task("niah_multikey", TARGET_TOKENS, seed=7)
        assert t1.prompt == t2.prompt
        assert t1.expected == t2.expected

    def test_different_seeds_differ(self):
        t1 = qs.build_task("niah_multikey", TARGET_TOKENS, seed=1)
        t2 = qs.build_task("niah_multikey", TARGET_TOKENS, seed=2)
        assert t1.prompt != t2.prompt

    def test_unknown_task_type_raises(self):
        with pytest.raises(ValueError):
            qs.build_task("not_a_real_task", TARGET_TOKENS, seed=1)

    def test_niah_multikey_needle_present_in_prompt(self):
        task = qs.build_task("niah_multikey", TARGET_TOKENS, seed=3)
        needle = f"{task.meta['chosen_key']} is {task.expected}"
        assert needle in task.prompt

    def test_variable_tracking_chain_resolves_to_expected(self):
        task = qs.build_task("variable_tracking", TARGET_TOKENS, seed=9)
        assert task.expected == str(task.meta["final_value"])

    def test_common_words_top_words_exactly_ten(self):
        task = qs.build_task("common_words_extraction", TARGET_TOKENS, seed=11)
        assert len(task.expected) == 10

    def test_system_rule_prompt_starts_with_rule(self):
        task = qs.build_task("system_rule_compliance", TARGET_TOKENS, seed=4)
        assert task.prompt.startswith(qs.SYSTEM_RULE)

    def test_tool_call_expected_matches_request_fields(self):
        task = qs.build_task("tool_call_correctness", TARGET_TOKENS, seed=6)
        args = task.expected["arguments"]
        assert args["title"] in task.meta["request"]
        assert str(args["duration_minutes"]) in task.meta["request"]


# ---------------------------------------------------------------- exact-match scorer

class TestScoreExact:
    def test_correct(self):
        score, _ = qs.score_exact_local("48291", "48291")
        assert score == 1.0

    def test_incorrect(self):
        score, _ = qs.score_exact_local("00000", "48291")
        assert score == 0.0

    def test_ignores_preamble_on_last_line(self):
        score, _ = qs.score_exact_local("Sure, here it is:\n48291", "48291")
        assert score == 1.0

    def test_substring_fallback(self):
        score, _ = qs.score_exact_local("The code you want is 48291, hope that helps.", "48291")
        assert score == 1.0


# ---------------------------------------------------------------- set-F1 scorer

class TestScoreSetF1:
    def test_perfect(self):
        score, _ = qs.score_set_f1("alpha, beta, gamma", {"alpha", "beta", "gamma"})
        assert score == 1.0

    def test_partial_credit_between_zero_and_one(self):
        score, _ = qs.score_set_f1("alpha, beta, delta", {"alpha", "beta", "gamma"})
        assert 0.0 < score < 1.0

    def test_completely_wrong(self):
        score, _ = qs.score_set_f1("zeta, eta", {"alpha", "beta", "gamma"})
        assert score == 0.0

    def test_empty_output_and_empty_expected_is_perfect(self):
        score, _ = qs.score_set_f1("", set())
        assert score == 1.0

    def test_order_independent(self):
        s1, _ = qs.score_set_f1("gamma, alpha, beta", {"alpha", "beta", "gamma"})
        s2, _ = qs.score_set_f1("alpha, beta, gamma", {"alpha", "beta", "gamma"})
        assert s1 == s2 == 1.0


# ---------------------------------------------------------------- json_keys / tool_call scorers

class TestScoreJsonKeys:
    def test_valid_exact_keys(self):
        score, _ = qs.score_json_keys('{"answer": "x", "source": "y"}', ["answer", "source"])
        assert score == 1.0

    def test_extra_key_fails(self):
        score, _ = qs.score_json_keys('{"answer": "x", "source": "y", "extra": 1}', ["answer", "source"])
        assert score == 0.0

    def test_missing_key_fails(self):
        score, _ = qs.score_json_keys('{"answer": "x"}', ["answer", "source"])
        assert score == 0.0

    def test_invalid_json_fails(self):
        score, _ = qs.score_json_keys("not json", ["answer", "source"])
        assert score == 0.0

    def test_tolerates_prose_and_fences(self):
        score, _ = qs.score_json_keys('Sure:\n```json\n{"answer": "x", "source": "y"}\n```', ["answer", "source"])
        assert score == 1.0


class TestScoreToolCall:
    def test_exact_match(self):
        call = {"name": "f", "arguments": {"a": 1}}
        score, _ = qs.score_tool_call('{"name": "f", "arguments": {"a": 1}}', call)
        assert score == 1.0

    def test_wrong_argument_value(self):
        call = {"name": "f", "arguments": {"a": 1}}
        score, _ = qs.score_tool_call('{"name": "f", "arguments": {"a": 2}}', call)
        assert score == 0.0

    def test_missing_argument(self):
        call = {"name": "f", "arguments": {"a": 1, "b": 2}}
        score, _ = qs.score_tool_call('{"name": "f", "arguments": {"a": 1}}', call)
        assert score == 0.0

    def test_invalid_json_fails(self):
        score, _ = qs.score_tool_call("not json", {"name": "f", "arguments": {}})
        assert score == 0.0


# ---------------------------------------------------------------- fabrication vs refusal

class TestClassifyFabricationOrRefusal:
    def test_refusal_phrase_detected(self):
        assert qs.classify_fabrication_or_refusal("I don't know, it isn't mentioned anywhere.") == "refusal"

    def test_confident_wrong_answer_is_fabrication(self):
        assert qs.classify_fabrication_or_refusal("The value is 91205.") == "fabrication"

    def test_empty_output_is_refusal(self):
        assert qs.classify_fabrication_or_refusal("") == "refusal"

    def test_none_output_is_refusal(self):
        assert qs.classify_fabrication_or_refusal(None) == "refusal"

    def test_not_provided_phrase(self):
        assert qs.classify_fabrication_or_refusal("That information is not provided in the text.") == "refusal"


# ---------------------------------------------------------------- 50%-front-truncation control

class TestTruncationControl:
    """Exercises the truncation/scoring code path itself, not a real model. A stub chat function is
    rigged to answer correctly ONLY when the literal needle sentence is present in what it receives;
    dropping the front half of the prompt (where the needle lives here) must then make run_task
    report a lower score. This is the harness-side half of the control the task asks for; the "report
    both" half (a real qwen3-8b run showing the same pattern) is out of scope for this module.
    """

    def test_front_truncation_drops_score_when_needle_is_removed(self):
        needle_sentence = "The registration code associated with KEY-ABCD is 918273."
        prompt = needle_sentence + " " + ("buffer word " * 2000)
        expected = "918273"

        def rigged_chat(prompt_seen, max_tokens, ignore_eos):
            return expected if needle_sentence in prompt_seen else "unknown"

        task = qs.Task(task_type="niah_multikey", prompt=prompt, expected=expected, scorer="exact",
                        max_tokens=32, meta={})

        full_row = qs.run_task(StubServer(rigged_chat), task, TARGET_TOKENS, seed=1)
        assert full_row["score"] == 1.0

        truncated = dataclasses.replace(task, prompt=qs.truncate_front(task.prompt, 0.5))
        trunc_row = qs.run_task(StubServer(rigged_chat), truncated, TARGET_TOKENS, seed=1)

        assert trunc_row["score"] < full_row["score"]
        assert trunc_row["score"] == 0.0

    def test_truncate_front_is_a_pure_character_drop(self):
        assert qs.truncate_front("abcdefghij", 0.5) == "fghij"
        assert qs.truncate_front("abcdefghij", 0.0) == "abcdefghij"


# ---------------------------------------------------------------- dry run: every public call site

class TestDryRunPublicAPI:
    """Exercises every public call site with realistic arguments, the way a t2s_night2-style phase
    would call this module. Guards against the kind of duplicate-keyword-argument bug that otherwise
    only surfaces at 2am on real hardware (see tests/test_llama_server.py's equivalent mock-only
    coverage for harness/llama_server.py)."""

    def test_run_suite_end_to_end_with_stub_server_that_always_refuses(self):
        srv = StubServer(lambda prompt, max_tokens, ignore_eos: "I don't know, it isn't mentioned.")
        rows = qs.run_suite(srv, target_tokens=TARGET_TOKENS, seed=42)

        seen_task_types = {r["task_type"] for r in rows}
        assert seen_task_types == set(qs.TASK_TYPES) | {"art_probe"}
        for row in rows:
            assert row["target_tokens"] == TARGET_TOKENS
            assert row["seed"] == 42
            assert "score" in row and "fabrication_or_refusal" in row and "meta" in row

        wrong = [r for r in rows if r["score"] is not None and r["score"] < 1.0]
        assert wrong  # a fixed refusal string should not accidentally satisfy any task
        assert all(r["fabrication_or_refusal"] == "refusal" for r in wrong)

    def test_run_suite_without_art_probes(self):
        srv = StubServer(lambda prompt, max_tokens, ignore_eos: "42")
        rows = qs.run_suite(srv, target_tokens=1500, seed=1, include_art_probes=False)
        assert {r["task_type"] for r in rows} == set(qs.TASK_TYPES)

    def test_run_suite_with_art_probes_only(self):
        srv = StubServer(lambda prompt, max_tokens, ignore_eos: "no information provided")
        rows = qs.run_suite(srv, target_tokens=1200, seed=2)
        art_rows = [r for r in rows if r["task_type"] == "art_probe"]
        assert len(art_rows) == 5
        assert all(r["fabrication_or_refusal"] in ("refusal", None) for r in art_rows)

    def test_build_task_and_run_task_every_type(self):
        srv = StubServer(lambda prompt, max_tokens, ignore_eos: "some answer")
        for task_type in qs.TASK_TYPES:
            task = qs.build_task(task_type, TARGET_TOKENS, seed=5)
            row = qs.run_task(srv, task, TARGET_TOKENS, seed=5)
            assert row["task_type"] == task_type
            assert row["outcome"] == "ok"

    def test_run_task_handles_server_error_outcome(self):
        srv = StubServer(lambda prompt, max_tokens, ignore_eos: None)
        task = qs.build_task("niah_multikey", TARGET_TOKENS, seed=8)
        row = qs.run_task(srv, task, TARGET_TOKENS, seed=8)
        assert row["outcome"] == "error"
        assert row["score"] is None
        assert row["fabrication_or_refusal"] is None

    def test_load_probes_returns_five_art_probes(self):
        mod, probes = qs.load_probes()
        assert len(probes) == 5
        assert all(p["id"].startswith("art_") for p in probes)
        assert hasattr(mod, "score")

    def test_build_probe_prompts_one_per_probe(self):
        srv = StubServer(lambda prompt, max_tokens, ignore_eos: "x")
        _, probes = qs.load_probes()
        out = qs.build_probe_prompts(srv, 1200, probes, seed=42)
        assert len(out) == len(probes)
        for probe, prompt, n_tok in out:
            assert probe["artifact"].strip() in prompt
            assert isinstance(n_tok, int) and n_tok > 0
