"""Verification tier (docs/VERIFICATION_TIER_PROPOSAL.md, operator decisions 2026-11-03): the payload an independent
cloud verifier sees after the post-run check (L1 to L3) has accepted a LOCAL result. Arm (a), approved: a
claude-haiku-5-5 accept/reject verdict. Arm (c), not approved for the run, priced as a comparison only: a
claude-haiku-5-5 re-derivation of the task from the pre-turn state.

What this module never sees: the grader (evalset_env.check_task), the task's reference calls, its expected answer
values, the expected end state, or any graded field of a result row (end_state_ok, state_ok, answer_ok,
end_state_diff, ...). Like src/dse/l3_invariants.py it imports nothing from the harness and takes plain strings,
lists and dicts, so it cannot reach them. The caller builds the inputs from what the client holds at the end of the
turn (analysis/verification_tier_plan.py::client_view): the task text, the standing instructions, the rules, the
pre-turn state, the turn's tool calls with their results, the final answer and the records the turn changed (before vs
after, NOT expected vs actual). tests/test_verification_tier.py checks this statically and at run time.

Operator decisions 2026-11-03 built in here: the recall-probe suffix ("Also, recall the exact value ...", appended to
one turn in four and not graded by end_state_ok) is STRIPPED from the task text the verifier sees; only local results
L1 to L3 accept are verified (cloud-escalated results are not, for now). The prompt texts below are frozen for Run V1:
a change to either is an amendment and must be filed in docs/EXPERIMENTS.md before any call.
"""
from __future__ import annotations

import json

from src.cloud.prompt_cache import with_rolling_breakpoints

VERIFIER_MODEL = "claude-haiku-5-5"
COMPARISON_MODEL = "claude-sonnet-5-5"     # priced for comparison only; no Sonnet call in Run V1
THINKING = {"type": "disabled"}
MAX_TOKENS_VERDICT = 256
MAX_TOKENS_REDERIVE = 1024
PER_TASK_CAP_USD = 0.005                  # per verified task, separate from the USD 0.25 escalation cap
JOB_CAP_USD = 2.00                        # billed, client spend cap = ledger total + 2.00

# The recall probe appended by harness/x2_r2_evalset.py::build_session to every fourth turn (that module's
# _RECALL_SUFFIX; tests/test_verification_tier.py asserts the two strings are identical).
RECALL_SUFFIX = (" Also, recall the exact value that was recorded for key '{key}' at the start of this session, "
                 "without looking it up.")
REDACTED = "[REDACTED]"

VERIFIER_INSTRUCTION = (
    "You are a verifier. An AI assistant has just carried out one request from a user in a small business workspace "
    "(inventory, ledger accounts, payees, calendar, files, tickets, customers) by calling tools. You do not carry out "
    "the request yourself. You decide whether the assistant's result does what the request asks.\n"
    "\n"
    "The user message gives you: the request; the workspace state before the turn (secret values are shown as "
    "[REDACTED]); every tool call the assistant made in this turn, in order, with its result; the assistant's final "
    "answer; and every record the turn changed, before and after. The standing instructions and session rules the "
    "assistant was given are below; a request may refer to them.\n"
    "\n"
    "Reject when the result does not do what the request asks, read together with the standing instructions: a change "
    "the request asks for is missing, incomplete, or made with a different value; a record the request does not ask to "
    "change was changed; or, for a question, the final answer does not give the value that the state before the turn "
    "holds. Otherwise accept. Do not judge the wording or format of the final answer, only whether its values are "
    "right.\n"
    "\n"
    "Reply with one JSON object and nothing else: {\"verdict\": \"accept\" or \"reject\", \"reason\": \"<one sentence "
    "naming the record or value>\"}."
)

REDERIVE_INSTRUCTION = (
    "You are given one request from a user in a small business workspace (inventory, ledger accounts, payees, "
    "calendar, files, tickets, customers), the workspace state before the request (secret values are shown as "
    "[REDACTED]), and the standing instructions and session rules below; a request may refer to them. Do not carry "
    "out the request and do not call any tool. Work out what a correct result is.\n"
    "\n"
    "Reply with one JSON object and nothing else: {\"calls\": [{\"name\": \"<tool>\", \"arguments\": {...}}, ...], "
    "\"answer\": \"<the value asked for, or empty when the request asks for no value>\"}. List only the calls that "
    "change the workspace (update_record, adjust_stock, transfer_funds, create_event, write_file, send_email, "
    "cancel_event, delete_file, close_account), in order, with exact arguments."
)


def strip_recall_suffix(task_text: str, recall_key: str | None) -> str:
    """The task text without the recall probe. recall_key None: the turn carries no probe, text returned unchanged.
    Raises ValueError when a probe is expected but the text does not end with it exactly."""
    if recall_key is None:
        return task_text
    suffix = RECALL_SUFFIX.format(key=recall_key)
    if not task_text.endswith(suffix):
        raise ValueError(f"task text does not end with the recall probe for key {recall_key!r}")
    return task_text[: -len(suffix)]


def redact(text: str, secrets) -> str:
    for s in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(s, REDACTED)
    return text


def _static(instruction: str, standing: str, rules: str) -> str:
    return (instruction + "\n\nStanding instructions given to the assistant:\n" + standing.strip() + "\n\n"
            "Session rules given to the assistant:\n" + rules.strip() + "\n")


def build_verdict_payload(task_text: str, recall_key: str | None, standing: str, rules: str, state_before: str,
                          trace: list, final_text: str | None, changes: list, secrets) -> dict:
    """Arm (a). Returns {"system", "user"}: system = the frozen instruction, the standing instructions and the rules
    (the same for every call of one session, so it is the cached prefix); user = everything specific to the turn.
    trace: [{"call": {"name", "arguments"}, "result": {...}}] in order; changes: [{"record", "before", "after"}] and
    optionally {"outbox_added": [...]}. Every secret value is replaced by [REDACTED] in the user part."""
    request = strip_recall_suffix(task_text, recall_key)
    calls = "\n".join(f"{i}. {json.dumps(t['call'], sort_keys=True)} -> {json.dumps(t['result'], sort_keys=True)}"
                      for i, t in enumerate(trace, 1)) or "(no tool calls)"
    user = ("Request:\n" + request.strip() + "\n\n"
            "Workspace state before this turn:\n" + state_before.rstrip() + "\n\n"
            "Tool calls in this turn, in order (call -> result):\n" + calls + "\n\n"
            "Final answer:\n" + (final_text or "(empty)").strip() + "\n\n"
            "Records changed by this turn (before -> after):\n"
            + ("\n".join(json.dumps(c, sort_keys=True) for c in changes) or "(no record changed)") + "\n")
    return {"system": _static(VERIFIER_INSTRUCTION, standing, rules), "user": redact(user, secrets)}


def build_rederive_payload(task_text: str, recall_key: str | None, standing: str, rules: str, state_before: str,
                           secrets) -> dict:
    """Arm (c) (comparison, not approved for the run): request and pre-turn state only; never the local calls, answer
    or change set."""
    request = strip_recall_suffix(task_text, recall_key)
    user = "Request:\n" + request.strip() + "\n\nWorkspace state before the request:\n" + state_before.rstrip() + "\n"
    return {"system": _static(REDERIVE_INSTRUCTION, standing, rules), "user": redact(user, secrets)}


def messages(payload: dict) -> list:
    """OpenAI-shape messages (litellm) with the live client's rolling cache breakpoints (src/cloud/prompt_cache.py):
    one on the system prompt (read from cache after a session's first verifier call), one on the user message
    (written; each turn's user message differs, so it is never read back)."""
    return with_rolling_breakpoints([{"role": "system", "content": payload["system"]},
                                     {"role": "user", "content": payload["user"]}])
