"""CUAD contract-review subset (harness/cuad_review.py): segmenter, gold pairs, the deterministic exact-set grader, the
answer shape check, subset selection, and that the committed manifest carries no contract or clause text."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
import cuad_review as cr  # noqa: E402

CTX = "AGREEMENT\n\nThis Agreement is governed by the laws of Delaware.\nTerm: five years.\n\nEither party may assign."


def test_segment_skips_blank_lines():
    segs = cr.segment(CTX)
    assert [CTX[s:e] for s, e in segs] == ["AGREEMENT", "This Agreement is governed by the laws of Delaware.",
                                          "Term: five years.", "Either party may assign."]


def test_gold_pairs_overlap():
    start = CTX.index("laws of Delaware")
    ans = {"Governing Law": [(start, "laws of Delaware")], "Anti-Assignment": [(CTX.index("assign"), "assign")],
           "Parties": [(0, "AGREEMENT")]}   # not one of the 10 categories: ignored
    assert cr.gold_pairs(CTX, ans) == {("Governing Law", 2), ("Anti-Assignment", 4)}


def test_gold_span_across_segments():
    s = CTX.index("Delaware")
    ans = {"Expiration Date": [(s, CTX[s:CTX.index("years") + 5])]}
    assert cr.gold_pairs(CTX, ans) == {("Expiration Date", 2), ("Expiration Date", 3)}


def test_grade_exact_set():
    gold = {("Governing Law", 2), ("Anti-Assignment", 4)}
    assert cr.grade(set(gold), gold)["correct"]
    g = cr.grade({("Governing Law", 2)}, gold)
    assert not g["correct"] and g["tp"] == 1 and g["fn"] == 1 and g["fp"] == 0
    assert not cr.grade(gold | {("Exclusivity", 1)}, gold)["correct"]
    assert cr.grade(set(), set())["correct"]          # empty gold needs an empty answer
    assert not cr.grade(None, set())["correct"]       # failed shape check is wrong


def test_parse_answer_shape_check():
    ok, why = cr.parse_answer('Here: [{"category": "Governing Law", "segment": 2}]', 4)
    assert ok == {("Governing Law", 2)} and why is None
    assert cr.parse_answer('[{"category": "Nope", "segment": 2}]', 4)[0] is None
    assert cr.parse_answer('[{"category": "Governing Law", "segment": 9}]', 4)[0] is None
    assert cr.parse_answer('[{"category": "Governing Law", "segment": true}]', 4)[0] is None
    assert cr.parse_answer("no list", 4)[0] is None
    assert cr.parse_answer("[]", 4)[0] == set()


def test_select_subset_strata_and_order():
    small = [{"title": f"t{i}", "context": "x" * 400, "answers": {}} for i in range(60)]
    big = [{"title": f"b{i}", "context": "y\n" * 100000, "answers": {}} for i in range(3)]
    sub = cr.select_subset(small + big)
    assert sum(1 for c in sub if c["stratum"] == "fits") == cr.N_FIT
    assert sum(1 for c in sub if c["stratum"] == "over_window") == 3
    fits = [c["title"] for c in sub if c["stratum"] == "fits"]
    assert fits == sorted(fits, key=cr.order_key)


def test_committed_manifest_has_no_task_text():
    m = json.loads((REPO / "results" / "cuad_manifest_v1.json").read_text(encoding="utf-8"))
    assert m["categories"] == list(cr.CATEGORIES)
    allowed = {"title", "stratum", "context_sha256", "n_chars", "n_segments", "est_prompt_tokens", "gold_pairs",
               "gold_empty"}
    for c in m["contracts"]:
        assert set(c) == allowed
        assert len(c["context_sha256"]) == 64
    assert len(m["contracts"]) == sum(1 for c in m["contracts"] if c["stratum"] == "fits") + m["n_over_window_in_split"]
    assert len(json.dumps(m)) < 50000   # titles and hashes only


def test_estimate_pure():
    man = {"window": 100, "contracts": [{"est_prompt_tokens": 50, "stratum": "fits"},
                                        {"est_prompt_tokens": 200, "stratum": "over_window"}]}
    rates = {"short": {"input": 1.0, "output": 5.0}, "long": {"input": 1.0, "output": 5.0}, "long_context_above": None}
    e = cr.estimate(man, rates, prefill_tok_s=10.0, decode_tok_s=1.0)
    p = e["point"]
    assert e["n_over"] == 1
    assert abs(p["cloud_arm_usd"] - (250 + 2 * 5 * cr.OUTPUT_TOKENS["point"]) / 1e6) < 1e-12
    assert abs(p["pipeline_routed_usd"] - (200 + 5 * cr.OUTPUT_TOKENS["point"]) / 1e6) < 1e-12
    assert abs(p["local_arm_h"] * 3600 - (5 + 10 + 2 * cr.OUTPUT_TOKENS["point"])) < 1e-9
