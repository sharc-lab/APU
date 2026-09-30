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

    def test_common_words_prompt_length_does_not_blow_past_target(self):
        """Regression: the original padding scheme (synthetic f"n{seed}{i:06d}" tokens, each intended as ~1 token)
        actually tokenized far more expensively than context.py's 5.03 chars/token calibration assumes, so a
        2000-token-intended prompt came out at 12,657 real tokens on evo-x2 (2026-09-29), exceeding an 8192-token
        server context and failing every call outright. The estimated token count (char length / _CHARS_PER_TOKEN)
        must now land close to target_tokens at every scale K1/A70 actually use, not several times over."""
        for target in (2000, 8000, 24000, 96000):
            task = qs.build_task("common_words_extraction", target, seed=42)
            est_tokens = len(task.prompt) / qs.ctx_mod._CHARS_PER_TOKEN
            assert 0.8 * target <= est_tokens <= 1.3 * target, (
                f"target={target} estimated_tokens={est_tokens:.0f} ratio={est_tokens / target:.2f}")

    def test_common_words_padding_repeat_count_stays_below_vocab_minimum(self):
        """No padding word's exact repeat count may reach the vocabulary's own smallest designed count -- otherwise
        the "top 10 most frequent" answer could legitimately include a padding word instead of a designed one."""
        for target in (500, 2000, 24000, 96000):
            task = qs.build_task("common_words_extraction", target, seed=7)
            min_vocab_count = min(task.meta["counts"].values())
            word_counts: dict[str, int] = {}
            for w in task.prompt.split():
                word_counts[w] = word_counts.get(w, 0) + 1
            for pad_word in qs._PAD_WORD_POOL:
                assert word_counts.get(pad_word, 0) < min_vocab_count, (
                    f"target={target} pad word {pad_word!r} count {word_counts.get(pad_word)} "
                    f">= vocab minimum {min_vocab_count}")

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


# ---------------------------------------------------------------------------------------------------- build_task count_fn
def test_build_task_count_fn_none_is_byte_identical_to_old_behavior():
    """count_fn=None must produce exactly the same prompt as before this parameter existed, for every task type,
    so no already-recorded result depends on a prompt that silently changed underneath it."""
    for task_type in qs.TASK_TYPES:
        a = qs.build_task(task_type, 3000, seed=42)
        b = qs.build_task(task_type, 3000, seed=42, count_fn=None)
        assert a.prompt == b.prompt


def test_build_task_count_fn_corrects_a_biased_tokenizer():
    """The 2026-09-29 bug this exists for: a tokenizer that needs more chars/token than context._CHARS_PER_TOKEN
    assumes under-fills without count_fn (real case: Llama 3.1/3.3 landed around 0.81 actual/target on niah_*).
    A count_fn that reports fewer tokens than the naive char-based estimate must pull the ratio back up."""
    def stingy_count_fn(text):
        return int(len(text) / 6.2)  # needs ~6.2 chars/token, vs context._CHARS_PER_TOKEN=5.03's assumption

    target = 4000
    without = qs.build_task("niah_multikey", target, seed=42)
    with_fn = qs.build_task("niah_multikey", target, seed=42, count_fn=stingy_count_fn)
    ratio_without = stingy_count_fn(without.prompt) / target
    ratio_with = stingy_count_fn(with_fn.prompt) / target
    assert abs(ratio_with - 1.0) < abs(ratio_without - 1.0)
    assert 0.9 < ratio_with < 1.1


def test_build_task_common_words_count_fn_none_is_byte_identical_to_old_behavior():
    """common_words_extraction has its own (non-build_filler) padding scheme; same backward-compat guarantee."""
    a = qs.build_task("common_words_extraction", 3000, seed=42)
    b = qs.build_task("common_words_extraction", 3000, seed=42, count_fn=None)
    assert a.prompt == b.prompt


def test_build_task_common_words_count_fn_corrects_a_biased_tokenizer():
    def stingy_count_fn(text):
        return int(len(text) / 6.2)

    target = 4000
    without = qs.build_task("common_words_extraction", target, seed=42)
    with_fn = qs.build_task("common_words_extraction", target, seed=42, count_fn=stingy_count_fn)
    ratio_without = stingy_count_fn(without.prompt) / target
    ratio_with = stingy_count_fn(with_fn.prompt) / target
    assert abs(ratio_with - 1.0) < abs(ratio_without - 1.0)


# ---------------------------------------------------------------------------------------------------- load_calibration_pass_set
# The 2026-09-29 gate fix: the calibration gate must check the calibration run's own per-task results, not whole-job
# queue status. A q0_token_calibration.py run writes one q0_calibration_summary record with a per-task_type
# {"all_ok": bool} map; downstream phases (K1 curves, K2) must run only the task_types that actually passed.

def _write_jsonl(path, records):
    import json as _json
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(_json.dumps(r) + "\n")


def test_load_calibration_pass_set_mixed_pass(tmp_path):
    p = tmp_path / "calib.jsonl"
    _write_jsonl(p, [
        {"record": "some_other_row", "x": 1},
        {"record": "q0_calibration_summary", "summary": {
            "niah_multikey": {"all_ok": True},
            "common_words_extraction": {"all_ok": False},
            "rag_lookup": {"all_ok": True},
        }},
    ])
    result = qs.load_calibration_pass_set(str(p))
    assert result == {"niah_multikey", "rag_lookup"}


def test_load_calibration_pass_set_all_pass(tmp_path):
    p = tmp_path / "calib.jsonl"
    _write_jsonl(p, [
        {"record": "q0_calibration_summary", "summary": {
            "niah_multikey": {"all_ok": True},
            "rag_lookup": {"all_ok": True},
        }},
    ])
    assert qs.load_calibration_pass_set(str(p)) == {"niah_multikey", "rag_lookup"}


def test_load_calibration_pass_set_all_fail(tmp_path):
    """Empty-but-not-None matters: an empty set means 'everything failed, filter it all out', which is different
    from None ('could not determine, do not filter')."""
    p = tmp_path / "calib.jsonl"
    _write_jsonl(p, [
        {"record": "q0_calibration_summary", "summary": {
            "niah_multikey": {"all_ok": False},
        }},
    ])
    assert qs.load_calibration_pass_set(str(p)) == set()


def test_load_calibration_pass_set_missing_file_returns_none(tmp_path):
    assert qs.load_calibration_pass_set(str(tmp_path / "does_not_exist.jsonl")) is None


def test_load_calibration_pass_set_no_summary_record_returns_none(tmp_path):
    p = tmp_path / "calib.jsonl"
    _write_jsonl(p, [{"record": "item_done", "task_type": "niah_multikey"}])
    assert qs.load_calibration_pass_set(str(p)) is None


def test_load_calibration_pass_set_malformed_json_lines_are_skipped(tmp_path):
    p = tmp_path / "calib.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write("not json at all\n")
        f.write('{"record": "q0_calibration_summary", "summary": {"niah_multikey": {"all_ok": true}}}\n')
    assert qs.load_calibration_pass_set(str(p)) == {"niah_multikey"}


def test_load_calibration_pass_set_uses_last_summary_record_if_multiple(tmp_path):
    p = tmp_path / "calib.jsonl"
    _write_jsonl(p, [
        {"record": "q0_calibration_summary", "summary": {"niah_multikey": {"all_ok": True}}},
        {"record": "q0_calibration_summary", "summary": {"niah_multikey": {"all_ok": False}, "rag_lookup": {"all_ok": True}}},
    ])
    assert qs.load_calibration_pass_set(str(p)) == {"rag_lookup"}
