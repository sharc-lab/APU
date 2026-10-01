"""Family (c): GSM8K subset, pulled from the real Hugging Face dataset.

Real openai/gsm8k ("main" config, "test" split) is reachable from this
machine over the datasets library with streaming=True (verified live before
writing this file). We use the dataset's own final-answer field (the number
after "####" in the "answer" column) as the oracle for deterministic grading.
License: openai/gsm8k is MIT-licensed (see README.md for the full citation).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ITEMS_DIR, count_tokens, write_jsonl  # noqa: E402

FAMILY = "gsm8k"
N_ITEMS = 80

FINAL_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def difficulty_for(question: str, n_steps: int) -> str:
    if n_steps <= 2:
        return "easy"
    if n_steps <= 4:
        return "medium"
    return "hard"


def build_items() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    items = []
    for i, row in enumerate(ds):
        if i >= N_ITEMS:
            break
        question = row["question"]
        answer_full = row["answer"]
        m = FINAL_RE.search(answer_full)
        if not m:
            # Skip any row without a clean final-answer marker; GSM8K's own
            # format always has one, so this should not trigger in practice.
            continue
        final_answer = m.group(1).replace(",", "")
        n_steps = answer_full.count("<<")
        prompt = (
            f"{question}\n\n"
            "Solve step by step, then give the final numeric answer on its own "
            'line in the form "#### <number>".'
        )
        item_id = f"gsm8k_{i:03d}"
        items.append(
            {
                "item_id": item_id,
                "family": FAMILY,
                "task_type": "math_word_problem",
                "difficulty": difficulty_for(question, n_steps),
                "prompt_tokens": count_tokens(prompt),
                "prompt": prompt,
                "oracle_answer": final_answer,
                "reference_solution": answer_full,
                "grading": {"method": "final_number_match"},
                "source": "openai/gsm8k (main, test split), real data, not a substitute",
            }
        )
    return items


def main():
    items = build_items()
    write_jsonl(ITEMS_DIR / "c_gsm8k.jsonl", items)
    print(f"[c] wrote {len(items)} real GSM8K items")


if __name__ == "__main__":
    main()
