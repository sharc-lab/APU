"""The naive hybrid baseline for the side-by-side scenario: a RouteLLM-style, hardware-blind router.

What it does, per step:
  1. difficulty = words in the turn's task instruction / DIFFICULTY_WORDS, clamped to [0, 1]. Like RouteLLM's
     routers it scores the query, not the attached context (the R2 turn's log filler is context). Above
     DIFFICULTY_THRESHOLD the step goes to the cloud.
  2. length check against the model's ADVERTISED context: the context Ollama loads for that model by default
     (read from the same result file's ollama_default sessions, `loaded_context_values`), not the num_ctx the
     deployment actually runs with. It never reads num_ctx, so on a 4096 deployment it sends a 30K-token
     conversation to the local model and gets HTTP 200 back.

The outcome it shows for a local step is the recorded outcome of that same turn in the replayed session (the
session ran with exactly this routing: every turn local at num_ctx 4096). A cloud step has no recorded outcome;
its cost is a STUB estimate from src/cloud/client.py's pricing table.
"""
from __future__ import annotations

DIFFICULTY_WORDS = 150
DIFFICULTY_THRESHOLD = 0.5
CLOUD_MODEL = "gpt-4o-mini"


def difficulty(task_words: int) -> float:
    return max(0.0, min(1.0, task_words / DIFFICULTY_WORDS))


def decide(step: dict, advertised_ctx: int | None) -> dict:
    d = difficulty(int(step.get("task_words") or 0))
    tokens = int(step.get("transcript_tokens_calibrated") or 0)
    if d > DIFFICULTY_THRESHOLD:
        target, why = "cloud", f"difficulty {d:.2f} > {DIFFICULTY_THRESHOLD}: cloud"
    elif advertised_ctx is not None and tokens > advertised_ctx:
        target, why = "cloud", f"{tokens} tokens > advertised context {advertised_ctx}: cloud"
    else:
        target = "local"
        why = (f"difficulty {d:.2f} <= {DIFFICULTY_THRESHOLD} and {tokens} tokens <= advertised context "
               f"{advertised_ctx}: local (num_ctx not checked)")
    cost = 0.0
    if target == "cloud":
        from src.cloud.client import estimate_cost_usd
        cost = estimate_cost_usd(CLOUD_MODEL, tokens, int(step.get("answer_budget_tokens") or 0))
    return {"target": target, "difficulty": d, "reason": why, "est_cost_usd": cost, "stub": target == "cloud"}
