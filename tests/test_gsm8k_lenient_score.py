"""GSM8K scoring in both outcome-table harnesses (2026-10-07): the strict `#### N` score (the 1349e6b fix, ported to
t2s_outcome_table.py), the separate lenient last-sentence score, format_ok, parity between the two harnesses, and
resume/cache backward compatibility for rows written before the lenient fields existed. No live calls."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import t2s_outcome_table as t2s  # noqa: E402
import x2_outcome_table as x2  # noqa: E402

MODULES = [x2, t2s]
GSM = {"grading": {"method": "final_number_match"}, "oracle_answer": "18"}


@pytest.mark.parametrize("mod", MODULES, ids=["x2", "t2s"])
def test_strict_extracts_number_after_last_hashes(mod):
    g = mod.load_graders()
    assert mod.score_response(g, GSM, "16 - 3 - 4 = 9 eggs, 9 * 2 = 18.\n#### 18") == 1.0
    assert mod.score_response(g, GSM, "#### 17\nrecheck\n#### 18") == 1.0
    assert mod.score_response(g, GSM, "**#### 18**") == 1.0
    assert mod.score_response(g, GSM, "The answer is 18.") == 0.0
    big = {"grading": {"method": "final_number_match"}, "oracle_answer": "1200"}
    assert mod.score_response(g, big, "#### 1,200") == 1.0


@pytest.mark.parametrize("mod", MODULES, ids=["x2", "t2s"])
def test_last_sentence_definition(mod):
    assert mod.gsm8k_last_sentence("A is 3. B is 4. So the total is 7.") == "So the total is 7."
    assert mod.gsm8k_last_sentence("Work.\n\n#### 7\n") == "#### 7"
    assert mod.gsm8k_last_sentence("It costs 1.5 dollars") == "It costs 1.5 dollars"  # "1.5" is not a boundary
    assert mod.gsm8k_last_sentence("Done!  Really?  Yes") == "Yes"
    assert mod.gsm8k_last_sentence("<think>x. y.</think>The total is 9.") == "The total is 9."
    assert mod.gsm8k_last_sentence("   ") == ""


@pytest.mark.parametrize("mod", MODULES, ids=["x2", "t2s"])
@pytest.mark.parametrize("text,want", [
    ("So she makes $18 every day.", 1.0),          # $ and trailing period
    ("The answer is 18", 1.0),
    ("Total: 18.00 dollars.", 1.0),                 # numeric equality
    ("Step 1. 9 * 2 = 18. So the answer is 19.", 0.0),  # 18 is not in the LAST sentence
    ("#### 18\n\nI hope this helps!", 0.0),         # strict 1, lenient 0 by definition
    ("The answer is 3-18 or so", 1.0),              # "-18" after a digit is not negative
    ("It is -18 degrees", 0.0),                     # a real negative is not 18
    ("", 0.0),
])
def test_lenient_score(mod, text, want):
    assert mod.gsm8k_lenient_score("18", text) == want


@pytest.mark.parametrize("mod", MODULES, ids=["x2", "t2s"])
def test_lenient_commas_and_grouping(mod):
    assert mod.gsm8k_lenient_score("1200", "She earns $1,200 per month.") == 1.0
    assert mod.gsm8k_lenient_score("1,200", "She earns 1200 per month.") == 1.0
    assert mod.gsm8k_lenient_score("34", "The values are 3,4 here.") == 0.0  # "3,4" is not a grouped number
    assert mod.gsm8k_lenient_score("-5", "The change is -5.") == 1.0


@pytest.mark.parametrize("mod", MODULES, ids=["x2", "t2s"])
def test_score_fields_columns(mod):
    g = mod.load_graders()
    f = mod.gsm8k_score_fields(g, GSM, "work\n#### 18")
    assert f == {"score_strict": 1.0, "format_ok": True, "score_lenient": 1.0}
    f = mod.gsm8k_score_fields(g, GSM, "work. The answer is 18.")
    assert f == {"score_strict": 0.0, "format_ok": False, "score_lenient": 1.0}
    f = mod.gsm8k_score_fields(g, GSM, "#### 17")
    assert f == {"score_strict": 0.0, "format_ok": True, "score_lenient": 0.0}
    other = {"grading": {"method": "exact_substring"}, "oracle_answer": "x"}
    assert mod.gsm8k_score_fields(g, other, "x") == {}


def test_harness_parity_on_mixed_inputs():
    g = x2.load_graders()
    texts = ["#### 18", "a. b 18.", "$18!", "18\n\nThanks", "x 1,800 y", "<think>18</think>no", "-18", "3-18"]
    for txt in texts:
        assert x2.gsm8k_score_fields(g, GSM, txt) == t2s.gsm8k_score_fields(g, GSM, txt), txt


def test_t2s_scored_fields_include_text_and_version():
    f = t2s.scored_fields(GSM, "x" * 600 + "\n#### 18")
    assert f["score"] == 1.0 and f["score_strict"] == 1.0 and f["scorer_version"] == t2s.SCORER_VERSION
    assert len(f["output_text"]) == 500 and f["output_tail"].endswith("#### 18")


def test_t2s_resume_skips_pre_fix_gsm8k_rows_only():
    assert t2s.row_is_reusable({"family": "gsm8k", "http_status": 200, "score": 0.0}) is False
    assert t2s.row_is_reusable({"family": "gsm8k", "http_status": 200, "scorer_version": 2}) is True
    assert t2s.row_is_reusable({"family": "gsm8k", "http_status": 500}) is True
    assert t2s.row_is_reusable({"family": "longdoc_qa", "http_status": 200, "score": 1.0}) is True


def test_x2_rows_without_lenient_fields_remain_valid_cache(tmp_path):
    """The running v3 job's existing rows (scorer_version 2, no score_lenient/format_ok/output_tail) are reused."""
    out = tmp_path / "v3.jsonl"
    old = {"record": "outcome_row", "item_id": "gsm8k_0", "family": "gsm8k", "config": "llama_server",
           "model_id": "llama3.1:8b", "http_status": 200, "score": 1.0, "scorer_version": 2,
           "output_text": "#### 18"}
    x2.emit(out, old)
    assert x2.row_is_valid(old) is True
    assert ("gsm8k_0", "llama3.1:8b", "llama_server") in x2.already_done_keys(out)
    assert x2.SCORER_VERSION == 2  # bumping it would invalidate the running job's cache


def test_x2_run_one_ollama_records_strict_lenient_and_tail(monkeypatch, tmp_path):
    import host_config as hc
    monkeypatch.setattr(hc, "start_ollama_server", lambda: None)
    monkeypatch.setattr(hc, "stop_ollama_server", lambda: None)
    monkeypatch.setattr(hc, "wait_for_ollama_ready", lambda timeout_s=60: True)
    reply = "x" * 600 + "\nSo the answer is $18."

    class FakeClient:
        def chat(self, model, prompt, num_ctx=None, messages=None, max_tokens=256, think=False, keep_alive=None,
                 timeout=None):
            return {"status": 200, "outcome": "ok", "message": reply, "prompt_eval_count": 5, "error": None}

    monkeypatch.setattr(x2.k1, "OllamaClient", FakeClient)
    item = dict(GSM, item_id="gsm8k_t", family="gsm8k", prompt="p", prompt_tokens=5)
    row = x2.run_one_ollama(item, "qwen3-8b", "qwen3:8b", tmp_path / "out.jsonl", call_timeout_s=30)
    assert row["score"] == 0.0 and row["score_strict"] == 0.0  # no "#### N"
    assert row["format_ok"] is False and row["score_lenient"] == 1.0
    assert row["output_tail"].endswith("$18.") and len(row["output_text"]) == 500
    assert x2.row_is_valid(row) is True


def test_non_gsm8k_or_malformed_items_get_no_extra_columns():
    g = x2.load_graders()
    for mod in MODULES:
        assert mod.gsm8k_score_fields(g, {"item_id": "x"}, "#### 18") == {}
