"""Deterministic grader + self-check for results/workload_pack/.

No model is called anywhere in this script. For every item we construct the
"perfect" response implied by the item's own oracle_answer and grading rule,
score it with that rule, and assert the score is exactly 1.0. An item whose
own oracle answer does not score 1.0 against its own grading rule is a broken
item (bad generator logic) and must be fixed, not shipped.

Run with: python grade.py   (repo's main Python 3.12 is fine here -- no
transformers/datasets import needed to grade, only to build.)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parent
ITEMS_DIR = PACK_ROOT / "items"

FAMILY_FILES = [
    "a_longdoc_qa.jsonl",
    "b_function_calling.jsonl",
    "c_gsm8k.jsonl",
    "d_r2_sessions.jsonl",
    "e_trace_mix.jsonl",
]


def load_all() -> list[dict]:
    items = []
    for fname in FAMILY_FILES:
        path = ITEMS_DIR / fname
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Grading rules (one per "method" used across items)
# ---------------------------------------------------------------------------

def grade_exact_substring(oracle: str, response: str, case_sensitive: bool = True) -> float:
    o = oracle if case_sensitive else oracle.lower()
    r = response if case_sensitive else response.lower()
    return 1.0 if o in r else 0.0


def _canon(obj):
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in sorted(obj.items())}
    return obj


def grade_exact_dict_match(oracle: dict, response: dict) -> float:
    return 1.0 if _canon(oracle) == _canon(response) else 0.0


def grade_final_number_match(oracle: str, response: str) -> float:
    def norm(x: str) -> float:
        return float(str(x).replace(",", "").strip())

    try:
        return 1.0 if norm(oracle) == norm(response) else 0.0
    except ValueError:
        return 0.0


def grade_session_rule_recall(oracle: dict, response: dict) -> float:
    expected_calls = oracle["expected_tool_calls"]
    got_calls = response.get("expected_tool_calls", response.get("tool_calls"))
    calls_ok = got_calls == expected_calls
    recall_ok = response.get("expected_recall", response.get("recall")) == oracle["expected_recall"]
    if calls_ok and recall_ok:
        return 1.0
    if calls_ok or recall_ok:
        return 0.5
    return 0.0


GRADERS = {
    "exact_substring": grade_exact_substring,
    "exact_dict_match": grade_exact_dict_match,
    "final_number_match": grade_final_number_match,
    "session_rule_recall": grade_session_rule_recall,
}


def perfect_response_for(item: dict):
    """Construct the response a perfect model would give, straight from the
    item's own oracle_answer, so grading the oracle against its own rule is
    the self-check."""
    method = item["grading"]["method"]
    oracle = item["oracle_answer"]
    if method == "exact_substring":
        return oracle
    if method == "exact_dict_match":
        return oracle
    if method == "final_number_match":
        return oracle
    if method == "session_rule_recall":
        return oracle
    raise ValueError(f"unknown grading method: {method}")


def score_item(item: dict) -> float:
    method = item["grading"]["method"]
    oracle = item["oracle_answer"]
    response = perfect_response_for(item)
    fn = GRADERS[method]
    if method == "exact_substring":
        return fn(oracle, response, case_sensitive=item["grading"].get("case_sensitive", True))
    return fn(oracle, response)


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def main() -> int:
    items = load_all()
    total = len(items)
    passed = 0
    failures = []
    by_family: dict[str, list[dict]] = {}
    for item in items:
        by_family.setdefault(item["family"], []).append(item)
        score = score_item(item)
        if score == 1.0:
            passed += 1
        else:
            failures.append((item["item_id"], score))

    print(f"TOTAL ITEMS: {total}")
    for fam, rows in by_family.items():
        print(f"  {fam}: {len(rows)} items")
    print(f"GRADE-CHECK PASS: {passed}/{total}")
    if failures:
        print("FAILURES:")
        for item_id, score in failures:
            print(f"  {item_id}: score={score}")

    all_tokens = [it["prompt_tokens"] for it in items]
    print("\nPrompt-token length distribution (full pack, {} items):".format(total))
    print(f"  p50={pct(all_tokens, 0.50):.0f}  p90={pct(all_tokens, 0.90):.0f}  p99={pct(all_tokens, 0.99):.0f}")
    print(f"  min={min(all_tokens)}  max={max(all_tokens)}  mean={statistics.mean(all_tokens):.0f}")

    for fam, rows in by_family.items():
        toks = [r["prompt_tokens"] for r in rows]
        print(
            f"  [{fam}] p50={pct(toks,0.50):.0f} p90={pct(toks,0.90):.0f} "
            f"p99={pct(toks,0.99):.0f} min={min(toks)} max={max(toks)}"
        )

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
