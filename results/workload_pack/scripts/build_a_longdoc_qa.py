"""Family (a): long-document QA with a buried, deterministically-gradable answer.

Prior-art note (required by the task spec): the task description assumes this
repo already has a Q0 / quality_suite long-document generator in
harness/t2s_k1_ollama.py and asks us to reuse it. Before writing this file we
searched the whole repository for "qs_build_task", "Q0", "quality_suite",
"t2s_k1_ollama" (file name and string), and "CWE":
  - No file named harness/t2s_k1_ollama.py exists.
  - No string "qs_build_task" exists anywhere in the repo.
  - No string "Q0" or "quality_suite" exists in harness/ or evaluation/.
  - No string "CWE" exists anywhere in the repo (checked docs/ and *.py).
So there is no real long-document QA generator and no CWE exclusion
convention to reuse here; this is new code, and no CWE exclusion was applied
because the term does not appear anywhere in this repository.

Design: classic needle-in-haystack long-document QA. A filler document of
realistic SME (subject-matter-expert-ish) prose is built out of a fixed
sentence pool, a single needle sentence carrying a random verification code
is inserted at a controlled relative depth, and the question asks for that
code. Grading is exact string match of the code in the response -- fully
deterministic, no LLM judge.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ITEMS_DIR, count_tokens, rng_for, write_jsonl  # noqa: E402

FAMILY = "longdoc_qa"

FILLER_SENTENCES = [
    "The quarterly maintenance log shows routine inspection of the cooling loop with no anomalies recorded.",
    "Field engineers rotated the backup compressor into primary service ahead of the scheduled downtime window.",
    "The procurement team finalized vendor contracts for replacement filtration cartridges this cycle.",
    "Average throughput across the three production lines held within two percent of the baseline target.",
    "A minor calibration drift was observed on sensor bank C and corrected during the next shift changeover.",
    "The safety committee reviewed incident reports and found no reportable events in the period.",
    "Warehouse inventory counts reconciled against the ERP system with a variance under half a percent.",
    "The training department completed refresher certification for all forklift operators on site.",
    "Energy consumption per unit produced decreased slightly following the insulation retrofit in bay four.",
    "Customer satisfaction scores for the regional support desk remained stable quarter over quarter.",
    "The audit team sampled thirty transactions and found all supporting documentation properly filed.",
    "Scheduled firmware updates for the fleet telematics units were deployed without incident overnight.",
    "The facilities group replaced two aging HVAC units ahead of the summer peak demand period.",
    "Supplier lead times for the critical fastener component lengthened by roughly one week this quarter.",
    "The data governance board approved a revised retention schedule for operational log files.",
    "Night shift headcount was temporarily increased to cover seasonal order volume.",
    "The pilot program for predictive maintenance sensors expanded to two additional production cells.",
    "Regulatory filings for the annual environmental report were submitted ahead of the statutory deadline.",
    "The logistics team renegotiated freight contracts, trimming average transit time by half a day.",
    "An internal review of the escalation process recommended consolidating two overlapping ticket queues.",
    "The finance group closed the month with no material reconciling items outstanding.",
    "A new intake form reduced average onboarding time for contractor badges by roughly a day.",
    "The quality lab added a second reference standard to improve traceability on tensile testing.",
    "Routine backups of the configuration management database completed successfully across all nodes.",
    "The site held its semiannual fire drill with a full evacuation completed within the target window.",
]

BUCKETS = [2000, 4000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 80000, 96000, 120000]
ITEMS_PER_BUCKET = 10


def difficulty_for(tokens: int) -> str:
    if tokens < 8000:
        return "short"
    if tokens < 32000:
        return "medium"
    if tokens < 80000:
        return "long"
    return "very_long"


def build_document(rng, target_tokens: int, needle_code: str) -> tuple[str, int]:
    header = (
        "INTERNAL OPERATIONS DIGEST\n\n"
        "The following document is a compiled summary of recent operational notes. "
        "Read carefully; one sentence contains a verification code you will be asked to report.\n\n"
    )
    needle_sentence = (
        f"NOTE: the secret verification code for this document is {needle_code}. "
        "Remember this code; it will be asked about later."
    )
    question = (
        "\n\nQUESTION: What is the secret verification code stated in the document above? "
        "Answer with only the code."
    )

    depth = rng.uniform(0.10, 0.90)

    def assemble(sentences: list[str]) -> str:
        insert_at = int(len(sentences) * depth)
        body_sentences = sentences[:insert_at] + [needle_sentence] + sentences[insert_at:]
        body = ""
        for i in range(0, len(body_sentences), 5):
            body += " ".join(body_sentences[i : i + 5]) + "\n\n"
        return header + body + question

    # Calibrate tokens-per-sentence once, then jump close to the target and
    # fine-tune with a small number of tokenizer calls instead of one call
    # per sentence (which is O(n^2) and far too slow at 120K tokens).
    calib_n = 200
    calib_sentences = [rng.choice(FILLER_SENTENCES) for _ in range(calib_n)]
    calib_tokens = count_tokens(assemble(calib_sentences))
    per_sentence = max(calib_tokens / calib_n, 1.0)

    sentences = calib_sentences
    text = assemble(sentences)
    n = count_tokens(text)
    while n < target_tokens:
        remaining = target_tokens - n
        add = max(int(remaining / per_sentence), 1)
        sentences.extend(rng.choice(FILLER_SENTENCES) for _ in range(add))
        text = assemble(sentences)
        n = count_tokens(text)
        if add == 1 and n >= target_tokens:
            break
    return text, n


def build_items() -> list[dict]:
    items = []
    for bucket in BUCKETS:
        for i in range(ITEMS_PER_BUCKET):
            item_id = f"longdoc_{bucket}_{i:02d}"
            rng = rng_for(item_id)
            needle_code = f"{rng.randrange(10**8, 10**9)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
            prompt, n_tokens = build_document(rng, bucket, needle_code)
            items.append(
                {
                    "item_id": item_id,
                    "family": FAMILY,
                    "task_type": "needle_qa",
                    "difficulty": difficulty_for(n_tokens),
                    "target_prompt_tokens": bucket,
                    "prompt_tokens": n_tokens,
                    "prompt": prompt,
                    "oracle_answer": needle_code,
                    "grading": {"method": "exact_substring", "case_sensitive": True},
                }
            )
    return items


def main():
    items = build_items()
    write_jsonl(ITEMS_DIR / "a_longdoc_qa.jsonl", items)
    print(f"[a] wrote {len(items)} long-doc QA items")


if __name__ == "__main__":
    main()
