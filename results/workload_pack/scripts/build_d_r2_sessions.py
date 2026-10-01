"""Family (d): multi-turn agent sessions (rule-compliance + tool-call + recall).

Prior-art note: the task asks us to reuse harness/t2s_r2_session_growth.py's
own session generator. We searched the repo (file name, and the strings
"generate_session", "r2_session", "session_growth") before writing this file:
no file named harness/t2s_r2_session_growth.py exists, and none of those
strings appear anywhere in the repository. There is no multi-turn session
generator to reuse, so this is new code.

Design: each item is a simulated multi-turn session transcript. Turn 0
establishes a standing rule (always call a specific tool before answering)
and a fact the user states once. Middle turns are ordinary requests that each
require a tool call. The final turn asks the model to recall the turn-0 fact.
The oracle answer records (a) the expected tool-call name for every turn that
requires one and (b) the expected recalled fact. Grading checks both: full
tool-call-sequence compliance and exact recall match.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ITEMS_DIR, count_tokens, rng_for, write_jsonl  # noqa: E402

FAMILY = "r2_sessions"
N_ITEMS = 60

TOOL_NAME = "log_action"
FACT_KINDS = [
    ("account ID", lambda rng: f"ACC-{rng.randrange(100000,999999)}"),
    ("ticket number", lambda rng: f"TCK-{rng.randrange(1000,9999)}"),
    ("confirmation code", lambda rng: f"{rng.choice('ABCDEFGH')}{rng.randrange(1000,9999)}"),
    ("project codename", lambda rng: rng.choice(["Project Falcon", "Project Harbor", "Project Lumen", "Project Atlas"])),
]
REQUEST_TOPICS = [
    "check the status of the latest deployment",
    "summarize the last incident report",
    "look up the current queue depth",
    "list open tickets assigned to the team",
    "check whether the nightly job completed",
    "summarize this week's error rate",
    "look up the on-call engineer for today",
    "check disk usage on the primary server",
]


def build_item(idx: int) -> dict:
    item_id = f"r2_session_{idx:03d}"
    rng = rng_for(item_id)
    n_turns = rng.randrange(4, 11)
    fact_label, fact_fn = rng.choice(FACT_KINDS)
    fact_value = fact_fn(rng)

    turns = []
    turns.append(
        {
            "role": "system",
            "content": (
                f"Standing rule: before answering any user request, you must call the "
                f"`{TOOL_NAME}(action: string)` tool summarizing what you are about to do."
            ),
        }
    )
    turns.append(
        {
            "role": "user",
            "content": f"For reference, my {fact_label} is {fact_value}. Please keep that noted.",
        }
    )

    expected_tool_calls = []
    for t in range(n_turns):
        topic = rng.choice(REQUEST_TOPICS)
        turns.append({"role": "user", "content": f"Please {topic}."})
        expected_tool_calls.append({"turn": len(turns) - 1, "name": TOOL_NAME, "argument_contains": topic})

    final_turn_index = len(turns)
    turns.append({"role": "user", "content": f"Quick check: what {fact_label} did I give you earlier?"})

    transcript = "\n".join(f"[{t['role']}] {t['content']}" for t in turns)
    prompt = (
        f"{transcript}\n\n"
        f"Continue the conversation: respond to the final user turn, following the standing "
        f"rule (call `{TOOL_NAME}` before answering) and correctly recalling the {fact_label} "
        "given earlier."
    )

    oracle_answer = {
        "expected_tool_calls": expected_tool_calls,
        "final_turn_index": final_turn_index,
        "expected_recall": fact_value,
    }

    n_turns_total = len(turns) + 1
    difficulty = "easy" if n_turns_total <= 6 else "medium" if n_turns_total <= 9 else "hard"

    return {
        "item_id": item_id,
        "family": FAMILY,
        "task_type": "multiturn_rule_recall",
        "difficulty": difficulty,
        "n_turns": n_turns_total,
        "prompt_tokens": count_tokens(prompt),
        "prompt": prompt,
        "oracle_answer": oracle_answer,
        "grading": {"method": "session_rule_recall"},
    }


def build_items() -> list[dict]:
    return [build_item(i) for i in range(N_ITEMS)]


def main():
    items = build_items()
    write_jsonl(ITEMS_DIR / "d_r2_sessions.jsonl", items)
    print(f"[d] wrote {len(items)} multi-turn session items")


if __name__ == "__main__":
    main()
