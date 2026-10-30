"""Cloud redo of every wrong task in the B3 v2 task real run: the measured cost of a silent wrong answer.

Source: results/x2_evalset_tasks_real_v2.jsonl (qwen3:14b, five tiers x five seeds x 30 task turns; see
harness/x2_r2_evalset.py::tasks_real_v2_outcome and register row x2-evalset-tasks-real-v2-wrong-rate). Every task turn
with end_state_ok false in a completed real session is one "wrong task". Each is re-run on the cloud escalation model
(results/cloud_prices_anthropic_20261014.json "selection" -> "escalation_mid_tier") through src/cloud/client.py, with the
USD 50 hard cap of that module in force (this script never raises or bypasses it).

Task re-run (same definition, environment and grader): the session is rebuilt with x2_r2_evalset.build_session (same
seed, design, workload and guardrails as the local run); the transcript before the wrong turn is rebuilt from the local
run's own turn rows (each prior user message rebuilt exactly; each prior call's assistant content and parsed tool calls
from the row; each tool result recomputed with evalset_env.apply_call on the environment as the local model left it).
env_before is that environment. The cloud model then runs the turn with the same tool loop as the local run (tools on
every call, at most MAX_TOOL_ROUNDS tool-bearing calls, then one final call) and the turn is graded by
evalset_env.check_task, the same end-state checker. As a parity check, the local turn's own calls are regraded on the
rebuilt env_before (local_regrade_end_state_ok); it must reproduce end_state_ok false.

Modes:
  stub (default): "priced, not executed". No network call and no key read. Each task is costed from the shared price
    table using the task's real token counts: input = the prompt tokens of each call of the turn as the local run sent
    them (calls[].prompt_tokens_est, the harness's client-side estimate of the whole prompt; not
    prompt_eval_count, which is capped by the local window at small tiers); output = the local run's completion_tokens
    per call, used as a STAND-IN for the cloud model's output (not measured). Seconds and cloud correctness are NOT
    MEASURED in this mode (null in every row).
  live (--live, and ANTHROPIC_API_KEY must be set; refused otherwise): "executed". One warm-up call (excluded from
    stats), then each wrong task once, in (tier, seed, turn) order. Seconds = wall time of the provider requests only
    (PreparedCall.send), summed over the turn's calls. Tokens from the provider's usage.

Prompt caching (priced, both modes): on. Pricing model in stub mode, per task with a cold cache (no hit carried across
tasks, so an upper side): call 1 writes its whole prompt (5-minute cache write rate); call k reads call k-1's prompt
(cache-hit rate) and writes the rest. A prompt below the model's min_cacheable_tokens is charged at the input rate. Each
row also carries usd_all_uncached (every input token at the input rate, the basis client.py's cap check uses).

Token basis: counts are the local harness's estimate, not Anthropic's tokenizer (the price file notes Claude 4.7 and
later produce about 30% more tokens for the same text).

Operator command once the key is set (from the repo root, on the controller; the key is never printed):
    $env:CLOUD_CLIENT_HOST_ROLE = "controller"; py -3.12 scripts/cloud_redo.py --live
(litellm must be installed: client.py sends Anthropic calls through it.)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "harness", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import evalset_env as E  # noqa: E402
import x2_r2_agent as ag  # noqa: E402
import x2_r2_evalset as EV  # noqa: E402
from cloud import client as cc  # noqa: E402
from cloud import console_gate  # noqa: E402
from cloud import task_budget as TB  # noqa: E402

SOURCE = REPO / "results" / "x2_evalset_tasks_real_v2.jsonl"
OUT = REPO / "results" / "cloud_redo_v1.jsonl"
OUT_FULL_CONTEXT = REPO / "results" / "cloud_redo_full_context_v1.jsonl"
OUT_ROLLBACK = REPO / "results" / "own_rollback_redo_v1.jsonl"
V1_EXECUTED = OUT
# --thinking-off (2026-10-21, after analysis/cloud_redo_diagnosis.py): v1 sent no thinking parameter and max_tokens 384;
# both models think by default and thinking counts against max_tokens, so 109 of v1's 118 failures were empty replies at
# the cap. Same values as scripts/all_cloud_b3.py THINKING_OFF: Haiku 5.5 accepts "disabled"; Sonnet 5.5 rejects it and
# turns thinking off with "between_tools". Temperature is never sent (provider default; temperature 0 is rejected).
THINKING_OFF = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}
PRICE_FILE = REPO / "results" / "cloud_prices_anthropic_20261014.json"
HARNESS = "cloud_redo"
MODE_STUB = "priced_not_executed"
MODE_LIVE = "executed"
KEY_ENV = cc.ENV_VAR_ANTHROPIC_API_KEY
NOT_MEASURED_STUB = ("seconds", "cloud_end_state_ok")
MAX_TOOL_ROUNDS = EV.MAX_TOOL_ROUNDS
TRUNC_CHARS = 4000  # x2_r2_agent call records keep content[:4000]


# ── provenance ──────────────────────────────────────────────────────────────────────────────────────

def git_blob(path: Path) -> str:
    """git hash-object of the file as committed (CRLF normalised to LF, as the repo stores text)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── source rows ─────────────────────────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def tier_of(arm_id: str, call2_mode: str = "off") -> str:
    sfx = ag.MODE_SUFFIX[ag.as_mode(call2_mode)]
    return arm_id[: -len(sfx)] if sfx and arm_id.endswith(sfx) else arm_id


def wrong_tasks(rows: list[dict]) -> list[dict]:
    """Task turns with end_state_ok false in completed real sessions (the same filter as tasks_real_v2_outcome),
    sorted by (tier order, seed, turn)."""
    sessions = [s for s in ag.completed_sessions(rows) if s.get("mode") == "real"]
    att = {s["attempt_id"] for s in sessions}
    order = {t: i for i, t in enumerate(EV.TASKS_V2_TIERS)}
    out = [r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in att
           and r.get("item_kind") == "task" and r.get("end_state_ok") is False]
    return sorted(out, key=lambda r: (order.get(tier_of(r["arm_id"], r.get("call2_mode", "off")), 99),
                                      r["seed"], r["turn_idx"]))


def session_rows(rows: list[dict], attempt_id: str) -> list[dict]:
    return sorted((r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") == attempt_id),
                  key=lambda r: r["turn_idx"])


# ── prices ──────────────────────────────────────────────────────────────────────────────────────────

def load_price_table(path: Path = PRICE_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def escalation_model(table: dict) -> str:
    return table["selection"]["escalation_mid_tier"]


def model_rates(table: dict, model_id: str) -> dict:
    m = table["models"][model_id]
    return {"short": m["standard"]["short"], "long": m["standard"]["long"],
            "long_context_above": m.get("long_context_above"), "min_cacheable": int(m.get("min_cacheable_tokens") or 0)}


def _rate(rates: dict, prompt_tokens: int) -> dict:
    above = rates.get("long_context_above")
    return rates["long"] if above is not None and prompt_tokens > above else rates["short"]


def call_cost(rates: dict, uncached_in: int, write: int, read: int, out: int, prompt_total: int) -> float:
    r = _rate(rates, prompt_total)
    return (uncached_in * r["input"] + write * r["cache_write"] + read * r["cached_input"] + out * r["output"]) / 1e6


def price_turn(calls: list[tuple[int, int]], rates: dict) -> dict:
    """Stub pricing of one task turn. calls: (prompt_tokens, output_tokens) per call, in order. Cold cache per task:
    call 1 writes its prompt; call k reads call k-1's prompt and writes the rest."""
    prev = 0
    tot = {"input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0,
           "uncached_input_tokens": 0, "usd": 0.0, "usd_all_uncached": 0.0}
    for p, o in calls:
        if p >= rates["min_cacheable"]:
            read = min(prev, p)
            write, unc = p - read, 0
        else:
            read, write, unc = 0, 0, p
        tot["input_tokens"] += p
        tot["output_tokens"] += o
        tot["cache_read_tokens"] += read
        tot["cache_write_tokens"] += write
        tot["uncached_input_tokens"] += unc
        tot["usd"] += call_cost(rates, unc, write, read, o, p)
        tot["usd_all_uncached"] += call_cost(rates, p, 0, 0, o, p)
        prev = p
    return tot


def local_call_tokens(row: dict) -> list[tuple[int, int]]:
    return [(int(c.get("prompt_tokens_est") or 0), int(c.get("completion_tokens") or 0)) for c in row.get("calls") or []]


# ── task re-run (same definition, environment, grader) ─────────────────────────────────────────────

def _oa_tool_calls(calls: list[dict], prefix: str) -> list[dict]:
    return [{"id": f"{prefix}_{j}", "type": "function",
             "function": {"name": c.get("name") or "unknown", "arguments": json.dumps(c.get("arguments") or {})}}
            for j, c in enumerate(calls)]


def _append_call(messages: list, env: dict, content: str, calls: list[dict], prefix: str) -> None:
    """Assistant message (OpenAI shape, as litellm takes it) plus one tool message per call, results applied to env."""
    if calls:
        tcs = _oa_tool_calls(calls, prefix)
        messages.append({"role": "assistant", "content": content or "", "tool_calls": tcs})
        for tc, c in zip(tcs, calls):
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(E.apply_call(env, c))})
    else:
        messages.append({"role": "assistant", "content": content or ""})


def build_session_for(row: dict, cache: dict) -> EV.EvalSession:
    key = (row["seed"], row.get("design", "v2"), bool(row.get("guardrails")), row.get("workload", "evalset"),
           row["model_id"])
    if key not in cache:
        cache[key] = EV.build_session(row["seed"], None, key[3], key[2], row["model_id"], None, key[1])
    return cache[key]


def rebuild(es: EV.EvalSession, prior: list[dict], turn_idx: int) -> dict:
    """Transcript and env_before for turn turn_idx, rebuilt from the local run's prior turn rows."""
    by_idx = {t.idx: (t, it) for t, it in zip(es.turns, es.items)}
    messages = [{"role": "system", "content": es.system_prompt}]
    env = copy.deepcopy(es.env0)
    truncated = 0
    for r in prior:
        t, _it = by_idx[r["turn_idx"]]
        messages.append({"role": "user", "content": EV.user_message(t, es)})
        for k, c in enumerate(r.get("calls") or []):
            content = c.get("content") or ""
            truncated += len(content) >= TRUNC_CHARS
            _append_call(messages, env, content, c.get("tool_calls_parsed") or [], f"h{r['turn_idx']}_{k}")
    turn, item = by_idx[turn_idx]
    messages.append({"role": "user", "content": EV.user_message(turn, es)})
    return {"messages": messages, "env_before": copy.deepcopy(env), "turn": turn, "item": item,
            "history_truncated_messages": truncated}


def rollback_env(es: EV.EvalSession, prior: list[dict], failed: dict) -> dict:
    """Own-rollback (2026-10-23): the local session replayed on an environment with transactional snapshots
    (evalset_env.SnapshotLog: a snapshot before every state-changing call, plus a mark at the start of each turn), the
    failed turn's own local calls applied too (the damage a real escalation would find), then the PRE-TASK mark
    restored. Returns {"env", "n_snapshots", "undone_calls"}. By construction this equals rebuild()'s env_before (the
    full-history redo already started from the pre-task state); run() records the comparison per task."""
    env = copy.deepcopy(es.env0)
    log = E.SnapshotLog(env)
    log.mark("session_start")
    for r in prior:
        log.mark(f"pre_turn_{r['turn_idx']}")
        for c in r.get("calls") or []:
            for call in c.get("tool_calls_parsed") or []:
                log.apply(call)
    log.mark("pre_task")
    n0 = len(log.snaps)
    for c in failed.get("calls") or []:
        for call in c.get("tool_calls_parsed") or []:
            log.apply(call)
    undone = len(log.snaps) - n0
    log.restore_mark("pre_task")
    return {"env": copy.deepcopy(env), "n_snapshots": len(log.snaps), "undone_calls": undone}


def stand_in_final_text(es: EV.EvalSession, task) -> str:
    """The final assistant message of an earlier wrong task turn in the cascaded transcript (a STAND-IN for the reply of
    a successful escalation; not a recorded cloud reply): the answer values for an information task, the session code
    (rule 5), JSON as rule 1 asks."""
    vals = " ".join(str(v) for v in (task.answer_values or ())) if task.is_info else ""
    ans = " ".join(x for x in ("Done.", vals, es.session_code) if x)
    return json.dumps({"answer": ans, "source": "tool results"})


def rebuild_cascaded(es: EV.EvalSession, prior: list[dict], turn_idx: int) -> dict:
    """Cascaded rollback (operator decision 2026-10-23): what a rollback hybrid with the oracle signal leaves before turn
    turn_idx. The local session is replayed on evalset_env.SnapshotLog; at every earlier task turn that ended wrong
    (end_state_ok false) the turn's local calls are applied, the environment is restored to that turn's pre-turn mark,
    and the task's REFERENCE calls are applied instead: a STAND-IN for a successful escalation of that earlier task.
    Transcript: correct earlier turns and guardrail turns as the local model ran them; an earlier wrong turn as its user
    message, one assistant message carrying the reference calls (their results recomputed on the environment), and
    stand_in_final_text. The grader is unchanged (expected = the reference applied to this env_before)."""
    by_idx = {t.idx: (t, it) for t, it in zip(es.turns, es.items)}
    messages = [{"role": "system", "content": es.system_prompt}]
    env = copy.deepcopy(es.env0)
    log = E.SnapshotLog(env)
    log.mark("session_start")
    truncated, replaced, undone = 0, 0, 0
    for r in prior:
        t, it = by_idx[r["turn_idx"]]
        log.mark(f"pre_turn_{r['turn_idx']}")
        messages.append({"role": "user", "content": EV.user_message(t, es)})
        if it.task is not None and r.get("end_state_ok") is False:
            n0 = len(log.snaps)
            for c in r.get("calls") or []:
                for call in c.get("tool_calls_parsed") or []:
                    log.apply(call)
            undone += len(log.snaps) - n0
            log.restore_mark(f"pre_turn_{r['turn_idx']}")
            ref = [dict(c) for c in it.task.reference]
            prefix = f"s{r['turn_idx']}"
            if ref:
                tcs = _oa_tool_calls(ref, prefix)
                messages.append({"role": "assistant", "content": "", "tool_calls": tcs})
                for tc, c in zip(tcs, ref):
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": json.dumps(log.apply(c))})
            messages.append({"role": "assistant", "content": stand_in_final_text(es, it.task)})
            replaced += 1
            continue
        for k, c in enumerate(r.get("calls") or []):
            content = c.get("content") or ""
            truncated += len(content) >= TRUNC_CHARS
            calls = c.get("tool_calls_parsed") or []
            if calls:
                tcs = _oa_tool_calls(calls, f"h{r['turn_idx']}_{k}")
                messages.append({"role": "assistant", "content": content, "tool_calls": tcs})
                for tc, call in zip(tcs, calls):
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(log.apply(call))})
            else:
                messages.append({"role": "assistant", "content": content})
    turn, item = by_idx[turn_idx]
    messages.append({"role": "user", "content": EV.user_message(turn, es)})
    return {"messages": messages, "env_before": copy.deepcopy(env), "turn": turn, "item": item,
            "history_truncated_messages": truncated, "earlier_wrong_replaced": replaced,
            "earlier_wrong_calls_undone": undone}


def repair_keys(repo: Path = REPO) -> set:
    """(attempt_id, turn_idx) of the full-history redo's earlier_state_repair failures (analysis/cloud_redo_diagnosis.py
    over cloud_redo_full_context_v1 and its parts: the classes of row cloud-redo-full-context)."""
    import contextlib
    import importlib.util
    import io
    spec = importlib.util.spec_from_file_location("cloud_redo_diagnosis", repo / "analysis" / "cloud_redo_diagnosis.py")
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)
    paths = part_files(repo / "results" / OUT_FULL_CONTEXT.name)
    with contextlib.redirect_stdout(io.StringIO()):
        d = cd.compute(repo, paths[0].relative_to(repo).as_posix(),
                       tuple(q.relative_to(repo).as_posix() for q in paths[1:]))
    want = {(p["tier"], p["seed"], p["turn_idx"]) for p in d["per_task"] if p["cls"] == "earlier_state_repair"}
    return {(r["attempt_id"], r["turn_idx"]) for r in merged_rows(paths)
            if (r["tier"], r["seed"], r["turn_idx"]) in want}


def grade(es: EV.EvalSession, item, env_before: dict, flat_calls: list[dict], final_text: str) -> dict:
    obj = ag.parse_json_strict(final_text or "")
    answer_field = obj.get("answer") if isinstance(obj, dict) else None
    chk = E.check_task(item.task, env_before, flat_calls, final_text, answer_field, design=es.design)
    return {k: chk[k] for k in ("end_state_ok", "state_ok", "answer_ok", "end_state_diff")}


def local_regrade(es, item, env_before, row) -> bool:
    flat = [c for rec in row.get("calls") or [] for c in rec.get("tool_calls_parsed") or []]
    return grade(es, item, env_before, flat, row.get("final_text") or "")["end_state_ok"]


# ── live path ───────────────────────────────────────────────────────────────────────────────────────

def with_cache_control(messages: list[dict]) -> list[dict]:
    """Prompt caching on: cache breakpoints on the system prompt and on the last user message."""
    out = copy.deepcopy(messages)
    last_user = max(i for i, m in enumerate(out) if m["role"] == "user")
    for i in (0, last_user):
        if isinstance(out[i]["content"], str):
            out[i]["content"] = [{"type": "text", "text": out[i]["content"], "cache_control": {"type": "ephemeral"}}]
    return out


def _usage(resp: dict) -> dict:
    u = resp.get("usage") or {}
    read = int(u.get("cache_read_input_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    write = int(u.get("cache_creation_input_tokens") or 0)
    prompt = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    # litellm's prompt_tokens includes the cache tokens; Anthropic's raw input_tokens does not
    unc = prompt - read - write if prompt >= read + write and u.get("prompt_tokens") is not None else prompt
    return {"input_tokens": unc + read + write, "uncached_input_tokens": unc, "cache_read_tokens": read,
            "cache_write_tokens": write, "output_tokens": int(u.get("completion_tokens") or u.get("output_tokens") or 0)}


def _parse_reply(resp: dict) -> tuple[str, list[dict]]:
    msg = ((resp.get("choices") or [{}])[0] or {}).get("message") or {}
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = ag._norm_args(fn.get("arguments"))
        c2, _n = E.coerce_call({"name": fn.get("name"), "arguments": args if isinstance(args, dict) else {}})
        calls.append(c2)
    return msg.get("content") or "", calls


# Bounded retry for transient provider errors only (2026-10-22: part 2 crashed on "Server disconnected without sending a
# response"). A failed PreparedCall.send() releases its reservation and writes no ledger row (client.py), so only the
# billed call counts toward the cap; each retry re-prepares (a fresh cap check).
RETRY_BACKOFF_S = (10.0, 30.0, 90.0)
_TRANSIENT_NAMES = {"InternalServerError", "ServiceUnavailableError", "RateLimitError", "APIConnectionError",
                    "Timeout", "APITimeoutError", "ConnectionError", "ConnectionResetError",
                    "RemoteProtocolError", "ServerDisconnectedError"}
_TRANSIENT_TEXT = ("server disconnected", "connection reset", "connection aborted", "overloaded", "overloaded_error",
                   "rate limit", "rate_limit", " 429", " 529", " 503", " 502", "status code 5")
_sleep = time.sleep


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, cc.SpendCapExceeded):
        return False
    names = {t.__name__ for t in type(exc).__mro__}
    if names & _TRANSIENT_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504, 529):
        return True
    text = f" {exc}".lower()
    return any(t in text for t in _TRANSIENT_TEXT)


def send_with_retry(client, model_id: str, messages: list[dict], kw: dict):
    """(prepared, raw, seconds of the successful send, retries). Transient errors: up to len(RETRY_BACKOFF_S) retries
    with that backoff; anything else, or a transient error after the last retry, is raised."""
    retries = 0
    while True:
        prepared = client.prepare(model_id=model_id, messages=messages, **kw)
        t0 = time.perf_counter()
        try:
            raw = prepared.send()
        except Exception as e:   # send() has already released the reservation; no ledger row was written
            if is_transient(e) and retries < len(RETRY_BACKOFF_S):
                print(f"  transient provider error ({type(e).__name__}); retry {retries + 1} in "
                      f"{RETRY_BACKOFF_S[retries]:.0f} s", flush=True)
                _sleep(RETRY_BACKOFF_S[retries])
                retries += 1
                continue
            raise
        return prepared, raw, time.perf_counter() - t0, retries


def empty_at_cap(call_rec: dict) -> bool:
    """A call that stopped at max_tokens with no visible content and no tool call (output spent where the harness cannot
    see it, e.g. thinking). Same test as scripts/all_cloud_b3.py::empty_at_cap; the run stops on it."""
    return call_rec.get("finish_reason") in ("length", "max_tokens") and not call_rec.get("content_chars") \
        and not call_rec.get("n_tool_calls")


def live_turn(client, model_id: str, es, built: dict, rates: dict, max_tokens: int, thinking: dict | None = None) -> dict:
    tools = EV.tools_payload(es.design)
    messages = copy.deepcopy(built["messages"])
    live_env = copy.deepcopy(built["env_before"])
    flat, final_text, secs = [], "", 0.0
    tot = {"input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0,
           "uncached_input_tokens": 0, "usd": 0.0, "usd_all_uncached": 0.0, "usd_ledger": 0.0}
    n_calls, call_recs = 0, []
    for i in range(MAX_TOOL_ROUNDS + 1):
        offer = i < MAX_TOOL_ROUNDS
        # Anthropic needs tools defined when the transcript holds tool calls: the final call keeps them, choice none
        kw = {"tools": tools, "max_tokens": max_tokens, "custom_llm_provider": "anthropic"}
        if thinking is not None:
            kw["thinking"] = thinking
        if not offer:
            kw["tool_choice"] = "none"
        prepared, raw, dt, retries = send_with_retry(client, model_id, with_cache_control(messages), kw)
        secs += dt
        res = prepared.settle(raw)
        n_calls += 1
        resp = res.raw
        u = _usage(resp)
        for k in ("input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens", "uncached_input_tokens"):
            tot[k] += u[k]
        tot["usd"] += call_cost(rates, u["uncached_input_tokens"], u["cache_write_tokens"], u["cache_read_tokens"],
                                u["output_tokens"], u["input_tokens"])
        tot["usd_all_uncached"] += call_cost(rates, u["input_tokens"], 0, 0, u["output_tokens"], u["input_tokens"])
        tot["usd_ledger"] += res.cost_usd
        content, calls = _parse_reply(resp)
        if not offer:
            calls = []
        call_recs.append({"finish_reason": ((resp.get("choices") or [{}])[0] or {}).get("finish_reason"),
                          "output_tokens": u["output_tokens"], "input_tokens": u["input_tokens"],
                          "content_chars": len(content), "n_tool_calls": len(calls),
                          "tool_choice": kw.get("tool_choice"), "retries": retries})
        if empty_at_cap(call_recs[-1]):
            final_text = content
            break
        if calls:
            flat.extend(calls)
            _append_call(messages, live_env, content, calls, f"c{i}")
            continue
        final_text = content
        break
    g = grade(es, built["item"], built["env_before"], flat, final_text)
    return {**tot, "seconds": secs, "n_calls": n_calls, "cloud_end_state_ok": g["end_state_ok"],
            "cloud_state_ok": g["state_ok"], "cloud_answer_ok": g["answer_ok"], "cloud_end_state_diff": g["end_state_diff"],
            "cloud_final_text": final_text[:TRUNC_CHARS], "cloud_n_tool_calls": len(flat), "cloud_calls": call_recs,
            "retries": sum(c.get("retries", 0) for c in call_recs),
            "empty_at_cap": any(empty_at_cap(c) for c in call_recs)}


def warmup(client, model_id: str, thinking: dict | None = None) -> dict:
    t0 = time.perf_counter()
    kw = {"thinking": thinking} if thinking is not None else {}
    res = client.call(model_id=model_id, messages=[{"role": "user", "content": "Reply with the single word OK."}],
                      max_tokens=8, custom_llm_provider="anthropic", **kw)
    return {"record": "warmup", "excluded_from_stats": True, "seconds": time.perf_counter() - t0,
            "usd_ledger": res.cost_usd, "usd_ledger_upper_bound": res.cost_usd_upper_bound,
            "input_tokens": res.input_tokens, "output_tokens": res.output_tokens}


# ── summary ─────────────────────────────────────────────────────────────────────────────────────────

def _stats(vals: list[float]) -> dict:
    return {"n": len(vals), "total_usd": sum(vals), "mean_usd": statistics.fmean(vals) if vals else None,
            "median_usd": statistics.median(vals) if vals else None}


def summarize(rows: list[dict]) -> dict:
    tasks = [r for r in rows if r.get("record") == "cloud_redo_task"]
    modes = sorted({r["mode"] for r in tasks})
    out = {"mode": modes[0] if len(modes) == 1 else modes, **_stats([r["usd"] for r in tasks]),
           "total_usd_all_uncached": sum(r["usd_all_uncached"] for r in tasks), "per_tier": {}}
    for t in EV.TASKS_V2_TIERS:
        vals = [r["usd"] for r in tasks if r["tier"] == t]
        if vals:
            out["per_tier"][t] = _stats(vals)
    sec = [r["seconds"] for r in tasks if r.get("seconds") is not None]
    ok = [r["cloud_end_state_ok"] for r in tasks if r.get("cloud_end_state_ok") is not None]
    out["seconds_mean"] = statistics.fmean(sec) if sec else None
    out["cloud_correct"] = sum(ok) if ok else None
    out["cloud_graded"] = len(ok)
    out["local_regrade_false"] = sum(1 for r in tasks if r.get("local_regrade_end_state_ok") is False)
    return out



# ── projection ──────────────────────────────────────────────────────────────────────────────────────

def projection_from_executed(wrong: list[dict], executed_rows: list[dict]) -> dict | None:
    """Ledger-basis (every input token uncached, as client.py's cap check charges) projection for a rerun, from an
    executed run of the same tasks: per task, its measured ledger USD per call times max(its call count, the mean call
    count of the tasks that run got right), since a turn that ended early on an empty reply will now run its tool loop.
    Also the same with the cache-priced USD (usd). None when no executed row matches."""
    ex = {(r["attempt_id"], r["turn_idx"]): r for r in executed_rows
          if r.get("record") == "cloud_redo_task" and r.get("mode") == MODE_LIVE}
    hit = [ex[k] for k in ((r["attempt_id"], r["turn_idx"]) for r in wrong) if k in ex]
    if not hit:
        return None
    ok = [r["n_calls"] for r in hit if r.get("cloud_end_state_ok")]
    mc = statistics.fmean(ok) if ok else float(MAX_TOOL_ROUNDS + 1)
    led = sum(r["usd_ledger"] / r["n_calls"] * max(r["n_calls"], mc) for r in hit)
    cached = sum(r["usd"] / r["n_calls"] * max(r["n_calls"], mc) for r in hit)
    worst = sum(r["usd_ledger"] / r["n_calls"] * (MAX_TOOL_ROUNDS + 1) for r in hit)
    return {"n_matched": len(hit), "n_tasks": len(wrong), "mean_calls_correct": mc, "ledger_usd": led,
            "cached_usd": cached, "ledger_usd_worst_all_rounds": worst}


# ── resume (2026-10-22: the full-context run stopped at 41/197 on an empty reply at max_tokens 384) ──

OUT_FULL_CONTEXT_PART2 = REPO / "results" / "cloud_redo_full_context_v1_part2.jsonl"
LOCAL_MAX_TOKENS = ag.MAX_TOKENS_PER_CALL


def hit_limit(row: dict, old_limit: int = LOCAL_MAX_TOKENS) -> bool:
    """A finished task row whose reply hit the old output limit or came back empty: any call with finish_reason
    length / max_tokens, output_tokens >= old_limit, or empty content with no tool call. Rows without per-call records
    (v1 shape) count as hit when the final text is empty."""
    calls = row.get("cloud_calls")
    if calls is None:
        return not (row.get("cloud_final_text") or "")
    return any(c.get("finish_reason") in ("length", "max_tokens") or int(c.get("output_tokens") or 0) >= old_limit
               or (not c.get("content_chars") and not c.get("n_tool_calls")) for c in calls)


def resume_plan(prior_rows: list[dict], old_limit: int = LOCAL_MAX_TOKENS) -> tuple[set, list[dict]]:
    """(keys to skip, supersede records) from a previous executed run. A task row that hit the limit is rerun and its
    old row is superseded (kept in its file, flagged by a supersede record in the continuation file)."""
    done = [r for r in prior_rows if r.get("record") == "cloud_redo_task" and r.get("mode") == MODE_LIVE]
    skip, sup = set(), []
    for r in done:
        k = (r["attempt_id"], r["turn_idx"])
        if hit_limit(r, old_limit):
            sup.append({"record": "supersede", "attempt_id": r["attempt_id"], "turn_idx": r["turn_idx"],
                        "task_id": r["task_id"], "seed": r["seed"], "tier": r["tier"], "old_ts_utc": r.get("ts_utc"),
                        "reason": f"a call hit the {old_limit}-token limit or came back empty; rerun",
                        "old_calls": [{"finish_reason": c.get("finish_reason"), "output_tokens": c.get("output_tokens")}
                                      for c in r.get("cloud_calls") or []]})
        else:
            skip.add(k)
    return skip, sup


def part_files(first: Path) -> list[Path]:
    """The run's first file and every existing continuation file <stem>_partN.jsonl, in part order."""
    first = Path(first)
    parts = []
    for q in first.parent.glob(f"{first.stem}_part*.jsonl"):
        tail = q.stem[len(first.stem) + len("_part"):]
        if tail.isdigit():
            parts.append((int(tail), q))
    return [first] + [q for _n, q in sorted(parts)]


def next_part_path(first: Path) -> Path:
    first = Path(first)
    ns = [int(q.stem.rsplit("_part", 1)[1]) for q in part_files(first)[1:]]
    return first.parent / f"{first.stem}_part{max(ns) + 1 if ns else 2}.jsonl"


def _superseded(rows: list[dict]) -> set:
    return {(r["attempt_id"], r["turn_idx"], r.get("old_ts_utc")) for r in rows if r.get("record") == "supersede"}


def resume_plan_parts(paths: list[Path], max_tokens: int) -> tuple[set, list[dict], list[dict]]:
    """(keys to skip, new supersede records, kept rows) over a run's parts. Per task the latest row that no supersede
    record names counts. It is rerun (and a supersede record written) only if its part ran with a smaller max_tokens
    than this run and it hit that part's limit (hit_limit); a task whose rows are all superseded is rerun without a
    new record. A part without run_end is fine: rows are written whole, one per finished task."""
    per_part = [(load_rows(Path(q))) for q in paths]
    sup_seen = _superseded([r for rows in per_part for r in rows])
    latest, superseded_keys = {}, set()
    for rows in per_part:
        st = next((r for r in rows if r.get("record") == "run_start"), {})
        limit = int(st.get("max_tokens") or LOCAL_MAX_TOKENS)
        for r in rows:
            if r.get("record") != "cloud_redo_task" or r.get("mode") != MODE_LIVE:
                continue
            k = (r["attempt_id"], r["turn_idx"])
            if (k[0], k[1], r.get("ts_utc")) in sup_seen:
                superseded_keys.add(k)
                continue
            latest[k] = (r, limit)
    skip, sup, kept = set(), [], []
    for k, (r, limit) in latest.items():
        if limit < max_tokens and hit_limit(r, limit):
            sup.append({"record": "supersede", "attempt_id": r["attempt_id"], "turn_idx": r["turn_idx"],
                        "task_id": r["task_id"], "seed": r["seed"], "tier": r["tier"], "old_ts_utc": r.get("ts_utc"),
                        "reason": f"a call hit the {limit}-token limit or came back empty; rerun",
                        "old_calls": [{"finish_reason": c.get("finish_reason"), "output_tokens": c.get("output_tokens")}
                                      for c in r.get("cloud_calls") or []]})
        else:
            skip.add(k)
            kept.append(r)
    return skip, sup, kept


def merged_rows(paths: list[Path]) -> list[dict]:
    """Task rows of a run and all its continuation parts: rows named by a supersede record (attempt, turn, ts) are
    excluded; otherwise the last row per task wins."""
    paths = [Path(q) for q in paths if Path(q).exists()]
    per_part = [load_rows(q) for q in paths]
    sup = _superseded([r for rows in per_part for r in rows])
    seen = {}
    for rows in per_part:
        for r in rows:
            if r.get("record") != "cloud_redo_task":
                continue
            if (r["attempt_id"], r["turn_idx"], r.get("ts_utc")) in sup:
                continue
            seen[(r["attempt_id"], r["turn_idx"])] = r
    return list(seen.values())


def calibration(kept_rows: list[dict], executed_rows: list[dict]) -> float:
    """Measured ledger USD of kept new-protocol rows over the v1-based projection for the same tasks (>= 1 used)."""
    if not kept_rows:
        return 1.0
    p = projection_from_executed(kept_rows, executed_rows)
    if not p or not p["ledger_usd"]:
        return 1.0
    return sum(r["usd_ledger"] for r in kept_rows) / p["ledger_usd"]


# ── main ────────────────────────────────────────────────────────────────────────────────────────────

def _rel(q: Path) -> str:
    q = Path(q).resolve()
    return q.relative_to(REPO).as_posix() if q.is_relative_to(REPO) else q.name


def run(live: bool, source: Path = SOURCE, out: Path = OUT, price_file: Path = PRICE_FILE, limit: int | None = None,
        client=None, max_tokens: int = ag.MAX_TOKENS_PER_CALL, emit_print=print, thinking_off: bool = False,
        job_cap: float | None = None, ledger_total_at_start: float | None = None,
        executed_file: Path | None = None, resume_from: Path | None = None,
        old_limit: int = LOCAL_MAX_TOKENS, rollback: str | None = None, only_keys: set | None = None,
        projection_rows: list[dict] | None = None, pre_job_gate: bool = False,
        console_freshness_h: float | None = None, client_cap_total: float | None = None,
        operator_approval: str | None = None, task_cap: float | None = None) -> list[dict]:
    """task_cap: per-task USD cap (src/cloud/task_budget.py, operator item C): checked before every cloud call on
    the call's upper-bound reservation; on hit no further call or retry for that task, its row has outcome
    budget_exhausted (graded not fixed) and the run continues with the next task. Never loosens the client cap or
    the Console gate. thinking_off: send THINKING_OFF[model] on every call and stop the run at the first call that ends at max_tokens
    with nothing visible. job_cap (live): refuse up front when the projection (from executed_file if it holds an
    executed run of these tasks, else from the local token counts; ledger basis) exceeds it. Rows are written as they
    are made, so a stop or a cap leaves every finished task on disk.
    rollback "pre_task" (own-rollback, 2026-10-23): env_before comes from rollback_env (snapshots, failed turn undone)
    and each row records whether it equals rebuild()'s env_before. only_keys: run only these (attempt_id, turn_idx).
    projection_rows: executed rows of the same tasks; the projection is then their measured cache-priced USD (the basis
    the ledger records since 2026-10-22) with the all-uncached upper bound recorded beside it.
    pre_job_gate (live, set by main): before anything is sent, src/cloud/console_gate.py::pre_job must pass: the
    job's estimate is at most USD 2 or operator_approval (main's --operator-approval, this job's command line only)
    names this job (its output file stem) and an amount at or above it; then the Console check (fresh reading that
    agrees with the billed ledgers; billed cap headroom covers the projection); then the approval is written to the
    ledger and echoed in run_start. ESTIMATE: the projection; with --job-cap, the larger of the job cap and the
    projection; with --client-cap-total, the larger of (client cap total - ledger total at start) and the projection.
    Tests that drive run() with a fake real-mode client leave it off; stub mode is never gated."""
    if rollback not in (None, "pre_task", "cascaded"):
        raise SystemExit(f"refused: unknown rollback {rollback!r}")
    table = load_price_table(price_file)
    model_id = escalation_model(table)
    rates = model_rates(table, model_id)
    thinking = THINKING_OFF.get(model_id) if thinking_off else None
    if thinking_off and thinking is None:
        raise SystemExit(f"refused: no thinking-off setting known for {model_id}")
    src = load_rows(source)
    wrong = wrong_tasks(src)
    if only_keys is not None:
        wrong = [r for r in wrong if (r["attempt_id"], r["turn_idx"]) in only_keys]
    if limit is not None:
        wrong = wrong[:limit]
    n_all = len(wrong)
    sup, calib, kept = [], 1.0, []
    if resume_from is not None:
        rpaths = [Path(q) for q in resume_from] if isinstance(resume_from, (list, tuple)) else [Path(resume_from)]
        skip, sup, kept = resume_plan_parts(rpaths, max_tokens)
        wrong = [r for r in wrong if (r["attempt_id"], r["turn_idx"]) not in skip]
        if executed_file is not None and Path(executed_file).exists():
            calib = calibration(kept, load_rows(Path(executed_file)))
    mode = MODE_LIVE if live else MODE_STUB
    if client is None:
        client = cc.CloudClient(mode="real") if live else cc.CloudClient(mode="stub")
    if live and client.stub_mode:
        raise SystemExit("refused: live mode needs a real-mode client")
    cc.print_stub_banner_if_needed(client)
    per_tier = {}
    for r in wrong:
        t = tier_of(r["arm_id"], r.get("call2_mode", "off"))
        per_tier[t] = per_tier.get(t, 0) + 1
    proj = None
    if executed_file is not None and Path(executed_file).exists():
        proj = projection_from_executed(wrong, load_rows(Path(executed_file)))
    local_proj = sum(price_turn(local_call_tokens(r), rates)["usd_all_uncached"] for r in wrong)
    start = {
        "record": "run_start", "harness": HARNESS, "ts_utc": utc_iso(), "mode": mode,
        "script": "scripts/cloud_redo.py", "script_git_blob": git_blob(Path(__file__)),
        "price_file": price_file.relative_to(REPO).as_posix() if price_file.is_relative_to(REPO) else price_file.name,
        "price_file_git_blob": git_blob(price_file), "price_file_fetched_utc": table.get("fetched_utc"),
        "model_id": model_id, "source": source.relative_to(REPO).as_posix() if source.is_relative_to(REPO)
        else source.name, "source_git_blob": git_blob(source), "n_wrong_tasks": n_all,
        "n_tasks_this_run": len(wrong),
        "wrong_tasks_per_tier": per_tier, "prompt_caching": True, "spend_cap_usd": client.spend_cap_usd,
        "hard_spend_cap_usd": cc.HARD_SPEND_CAP_USD,
        "pricing_model": ("cache on, cold per task: call 1 writes its prompt (5m write rate), call k reads call k-1's "
                          "prompt (hit rate) and writes the rest; usd_all_uncached = every input token at input rate"),
        "token_basis": "calls[].prompt_tokens_est (harness estimate of the prompt sent), not Anthropic's tokenizer",
        "output_basis": ("local completion_tokens used as a stand-in for the cloud output" if not live
                         else "provider usage"),
        "not_measured": list(NOT_MEASURED_STUB) if not live else [],
        "context": "full client-side history (rebuild), no trimming",
        "max_tokens": max_tokens, "thinking_sent": thinking, "temperature_sent": None,
        "stop_on_empty_at_cap": bool(thinking_off),
        "job_cap_usd": job_cap, "ledger_total_at_start_usd": ledger_total_at_start,
        "projection_local_tokens_ledger_usd": local_proj, "projection_from_executed": proj,
        "protocol_difference": (None if max_tokens == LOCAL_MAX_TOKENS else
                                f"max_tokens {max_tokens} per cloud call vs the local run's {LOCAL_MAX_TOKENS}"),
        "resume_from": ([_rel(q) for q in rpaths] if resume_from is not None else None),
        "resume_kept_tasks": len(kept), "resume_superseded_tasks": len(sup), "resume_old_limit": old_limit,
        "projection_calibration": calib,
        "rollback": rollback, "only_keys_n": None if only_keys is None else len(only_keys),
        "task_cap_usd": task_cap,
    }
    meas = None
    if projection_rows is not None:
        ex = {(x["attempt_id"], x["turn_idx"]): x for x in projection_rows if x.get("record") == "cloud_redo_task"}
        hit = [ex[(r["attempt_id"], r["turn_idx"])] for r in wrong if (r["attempt_id"], r["turn_idx"]) in ex]
        meas = {"n_matched": len(hit), "n_tasks": len(wrong), "billed_usd": sum(x["usd"] for x in hit),
                "upper_bound_usd_all_uncached": sum(x["usd_all_uncached"] for x in hit)}
        start["projection_measured_same_tasks"] = meas
    rows = [start]
    if live and meas is not None:
        if meas["n_matched"] != len(wrong):
            raise SystemExit("refused: projection rows do not cover every task; nothing was sent")
        projected = meas["billed_usd"]
        start["projected_ledger_usd_used"] = projected
        projected_upper = meas["upper_bound_usd_all_uncached"]
        if job_cap is not None and projected > job_cap:
            raise SystemExit(f"refused: projected ${projected:.2f} billed (upper bound ${projected_upper:.2f}), "
                             f"measured on the same tasks, exceeds the job cap ${job_cap:.2f}; nothing was sent")
        if projected > client.remaining_budget_usd:
            raise SystemExit(f"refused: projected ${projected:.2f} billed (upper bound ${projected_upper:.2f}) "
                             f"exceeds the remaining budget (billed headroom); {client.spend_summary()}; nothing was sent")
    elif live:
        projected = proj["ledger_usd"] if proj and proj["n_matched"] == len(wrong) else local_proj
        projected *= max(1.0, calib)
        start["projected_ledger_usd_used"] = projected
        projected_upper = projected   # all input uncached: the projection is itself an upper bound
        if job_cap is not None and projected > job_cap:
            raise SystemExit(f"refused: projected ${projected:.2f} (upper bound, all input uncached) exceeds the job "
                             f"cap ${job_cap:.2f}; nothing was sent")
        if projected > client.remaining_budget_usd:
            raise SystemExit(f"refused: projected ${projected:.2f} (upper bound, all input uncached) exceeds the "
                             f"remaining budget (billed headroom); {client.spend_summary()}; nothing was sent")
    if live:
        start["spend_at_start"] = {"cap_usd": client.spend_cap_usd, "billed_usd": client.running_total_usd,
                                   "upper_bound_usd": client.running_total_usd_upper_bound,
                                   "cap_basis": getattr(client, "cap_basis", None)}
        start["projected_upper_bound_usd"] = projected_upper
        emit_print(f"projection: billed ${projected:.4f} (upper bound ${projected_upper:.4f}); "
                   f"{client.spend_summary()}")
        if pre_job_gate:
            room, basis = None, "projection"
            if job_cap is not None:
                room, basis = job_cap, "max(job cap, projection)"
            elif client_cap_total is not None and ledger_total_at_start is not None:
                room = client_cap_total - ledger_total_at_start
                basis = "max(client cap total - ledger total, projection)"
            estimate = projected if room is None else max(room, projected)
            try:
                gate = console_gate.pre_job(client, job_id=Path(out).stem, estimate_usd=estimate,
                                            estimate_basis=basis, projection_billed_usd=projected,
                                            projection_upper_usd=projected_upper, approval=operator_approval,
                                            freshness_h=console_freshness_h)
            except console_gate.ConsoleCheckRefused as e:
                raise SystemExit(str(e)) from None
            start["console_check"] = gate
            start["operator_approval"] = gate.get("approval")
            emit_print(console_gate.format_report(gate))
    out.parent.mkdir(parents=True, exist_ok=True)
    stopped = None
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        def emit(row):
            rows.append(row) if row is not start else None
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()

        emit(start)
        for x in sup:
            emit(x)
        try:
            if live:
                emit(warmup(client, model_id, thinking))
            lane = TB.BudgetedClient(client, task_cap) if (live and task_cap is not None) else None
            cache: dict = {}
            for r in wrong:
                es = build_session_for(r, cache)
                prior = [x for x in session_rows(src, r["attempt_id"]) if x["turn_idx"] < r["turn_idx"]]
                built = rebuild(es, prior, r["turn_idx"])
                if built["item"].task is None or built["item"].task.task_id != r["task_id"]:
                    raise RuntimeError(f"rebuilt item {built['item'].item_id} does not match row task {r['task_id']}")
                base = {"record": "cloud_redo_task", "mode": mode, "model_id": model_id,
                        "tier": tier_of(r["arm_id"], r.get("call2_mode", "off")), "arm_id": r["arm_id"],
                        "seed": r["seed"], "turn_idx": r["turn_idx"], "task_id": r["task_id"],
                        "attempt_id": r["attempt_id"], "local_model_id": r["model_id"],
                        "local_n_calls": len(r.get("calls") or []), "local_any_http_error": r.get("any_http_error"),
                        "local_regrade_end_state_ok": local_regrade(es, built["item"], built["env_before"], r),
                        "history_truncated_messages": built["history_truncated_messages"],
                        "rebuilt_messages": len(built["messages"])}
                if rollback == "pre_task":
                    rb = rollback_env(es, prior, r)
                    base.update({"rollback": rollback, "rollback_env_equals_rebuild": rb["env"] == built["env_before"],
                                 "rollback_undone_calls": rb["undone_calls"], "rollback_snapshots": rb["n_snapshots"]})
                    built["env_before"] = rb["env"]
                elif rollback == "cascaded":
                    cb = rebuild_cascaded(es, prior, r["turn_idx"])
                    base.update({"rollback": rollback, "earlier_wrong_replaced": cb["earlier_wrong_replaced"],
                                 "earlier_wrong_calls_undone": cb["earlier_wrong_calls_undone"],
                                 "env_differs_from_pre_task": cb["env_before"] != built["env_before"],
                                 "local_regrade_cascaded_end_state_ok": local_regrade(es, cb["item"],
                                                                                      cb["env_before"], r),
                                 "history_truncated_messages": cb["history_truncated_messages"],
                                 "rebuilt_messages": len(cb["messages"])})
                    built = cb
                if live and task_cap is not None:
                    lane.start_task(f"{base['tier']}:{r['seed']}:{r['turn_idx']}:{r['task_id']}")
                    try:
                        res = live_turn(lane, model_id, es, built, rates, max_tokens, thinking)
                    except TB.TaskBudgetExhausted as e:
                        spent = lane.task_spent_usd   # calls made before the refused one, counted at upper bound
                        res = {"outcome": TB.OUTCOME_EXHAUSTED, "cloud_end_state_ok": False, "usd": spent,
                               "usd_all_uncached": spent, "usd_ledger": spent, "seconds": None,
                               "usd_basis": "per-task counted spend (upper bound of the calls made)",
                               "task_budget": e.record}
                    res = {**res, **lane.task_state()}
                elif live:
                    res = live_turn(client, model_id, es, built, rates, max_tokens, thinking)
                else:
                    res = {**price_turn(local_call_tokens(r), rates), "n_calls": len(r.get("calls") or []),
                           "seconds": None, "cloud_end_state_ok": None, "not_measured": list(NOT_MEASURED_STUB)}
                emit({**base, **res, "ts_utc": utc_iso()})
                if thinking_off and res.get("empty_at_cap"):
                    stopped = (f"call ended at max_tokens with nothing visible ({r['task_id']} seed {r['seed']} turn "
                               f"{r['turn_idx']}); run stopped")
                    break
        except cc.SpendCapExceeded as e:
            stopped = f"spend cap: {e}"
        except Exception as e:   # a non-transient provider error, or a transient one past the last retry
            stopped = f"error, run stopped: {type(e).__name__}: {str(e)[:300]}"
        tasks = [x for x in rows if x.get("record") == "cloud_redo_task"]
        summary = summarize(rows) if tasks else {"n": 0}
        end = {"record": "run_end", "harness": HARNESS, "ts_utc": utc_iso(), "stopped": stopped, "summary": summary}
        if live:
            end["ledger_total_usd"] = getattr(client, "running_total_usd", None)
        emit(end)
    s = rows[-1]["summary"]
    if s.get("n"):
        emit_print(f"{mode}: n={s['n']} total ${s['total_usd']:.4f} mean ${s['mean_usd']:.5f} "
                   f"median ${s['median_usd']:.5f} -> {out.name}" + (f"; STOPPED: {stopped}" if stopped else ""))
    else:
        emit_print(f"{mode}: no task row -> {out.name}" + (f"; STOPPED: {stopped}" if stopped else ""))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true", help="execute on the cloud (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default results/cloud_redo_v1.jsonl, or cloud_redo_full_context_v1.jsonl with --thinking-off")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thinking-off", action="store_true",
                    help="send THINKING_OFF[model] and stop at the first empty reply at max_tokens (the full-context "
                         "rerun of 2026-10-21)")
    ap.add_argument("--ledger", type=Path, default=cc.DEFAULT_LEDGER_PATH,
                    help="the shared ledger (pass the main checkout's results/cloud_ledger.jsonl)")
    ap.add_argument("--job-cap", type=float, default=None,
                    help="this job's own cap in USD: client cap = ledger total at start + job cap; refuse up front if "
                         "the projection exceeds it")
    ap.add_argument("--max-tokens", type=int, default=LOCAL_MAX_TOKENS,
                    help=f"max_tokens per cloud call (default {LOCAL_MAX_TOKENS}, the local num_predict, for v1 "
                         "reproducibility); any other value is recorded in run_start as a protocol difference")
    ap.add_argument("--resume", action="store_true",
                    help="continue the full-context run across parts: read results/cloud_redo_full_context_v1.jsonl "
                         "and every existing <stem>_partN.jsonl, skip tasks finished in any part except rows that hit "
                         "a smaller earlier limit or came back empty (rerun, old row superseded), write the next part")
    ap.add_argument("--resume-from", type=Path, default=OUT_FULL_CONTEXT)
    ap.add_argument("--client-cap-total", type=float, default=None,
                    help="absolute client cap in USD on the ledger total (e.g. 43.1913 = original start 21.1913 + "
                         "job cap 22); refuse up front if the projected ledger total would pass it")
    ap.add_argument("--rollback", choices=("pre_task", "cascaded"), default=None,
                    help="own-rollback (2026-10-23): only the full-history redo's earlier_state_repair tasks. pre_task: "
                         "env_before from transactional snapshots with the failed turn undone (rollback_env); "
                         "cascaded: every earlier wrong task turn also rolled back and replaced by its reference calls "
                         "(rebuild_cascaded, the stand-in for a successful escalation). Default out "
                         "results/own_rollback_redo_v1.jsonl; projection = the same tasks' measured billed USD")
    ap.add_argument("--task-cap-usd", type=float, default=None,
                    help="per-task USD cap (src/cloud/task_budget.py): a task whose next call's upper-bound "
                         "reservation would pass it stops (outcome budget_exhausted); under the job cap, never above")
    ap.add_argument("--console-freshness-hours", type=float, default=None,
                    help=f"console check: a SHORTER freshness window than the confirmed "
                         f"{console_gate.FRESHNESS_HOURS_DEFAULT:g} h (a longer one is refused)")
    ap.add_argument(console_gate.APPROVAL_FLAG, dest="operator_approval", default=None,
                    help="operator only: approve this one job above the USD 2 per-job limit, e.g. "
                         "\"approve <output stem> up to USD <n>: <reason>\"; recorded in the ledger before the first "
                         "call; never raises a cap or skips the Console check")
    a = ap.parse_args(argv)
    if a.resume:
        out = a.out or next_part_path(a.resume_from)
        if Path(out).resolve() in {q.resolve() for q in part_files(a.resume_from)}:
            print("refused: --resume writes a continuation file, not the run it resumes; nothing was sent",
                  file=sys.stderr)
            return 2
        if Path(out).exists():
            print(f"refused: {Path(out).name} exists (would be overwritten); nothing was sent", file=sys.stderr)
            return 2
        if not Path(a.resume_from).exists():
            print(f"refused: {Path(a.resume_from).name} not found; nothing was sent", file=sys.stderr)
            return 2
    elif a.rollback:
        if not a.thinking_off:
            print("refused: --rollback runs the full-context protocol and needs --thinking-off; nothing was sent",
                  file=sys.stderr)
            return 2
        out = a.out or OUT_ROLLBACK
        if a.live and Path(out).exists():
            print(f"refused: {Path(out).name} exists (would be overwritten); nothing was sent", file=sys.stderr)
            return 2
    else:
        out = a.out or (OUT_FULL_CONTEXT if a.thinking_off else OUT)
    if a.client_cap_total is not None and a.job_cap is not None:
        print("refused: pass --job-cap or --client-cap-total, not both; nothing was sent", file=sys.stderr)
        return 2
    if a.client_cap_total is not None and a.client_cap_total > cc.HARD_SPEND_CAP_USD:
        print(f"refused: --client-cap-total above the ${cc.HARD_SPEND_CAP_USD:.2f} hard cap; nothing was sent",
              file=sys.stderr)
        return 2
    if a.live and not os.environ.get(KEY_ENV):
        print(f"refused: --live needs {KEY_ENV} set in the environment; nothing was sent", file=sys.stderr)
        return 2
    client, total0 = None, None
    if a.live:
        total0 = cc.CloudClient(mode="stub", ledger_path=a.ledger)._read_running_total()
        if a.job_cap is not None:
            cap = total0 + a.job_cap
        elif a.client_cap_total is not None:
            cap = a.client_cap_total
        else:
            cap = cc.HARD_SPEND_CAP_USD
        client = cc.CloudClient(mode="real", spend_cap_usd=cap, ledger_path=a.ledger)
        print(f"ledger total at start ${total0:.4f} billed; client cap ${client.spend_cap_usd:.4f} (the projection "
              "line below prints billed and upper bound side by side)")
    rb = {}
    if a.rollback:
        rb = {"rollback": a.rollback, "only_keys": repair_keys(REPO),
              "projection_rows": merged_rows(part_files(OUT_FULL_CONTEXT))}
    rows = run(a.live, a.source, out, limit=a.limit, client=client, thinking_off=a.thinking_off, job_cap=a.job_cap,
               ledger_total_at_start=total0, executed_file=V1_EXECUTED if a.thinking_off and not a.rollback else None,
               max_tokens=a.max_tokens, resume_from=part_files(a.resume_from) if a.resume else None,
               pre_job_gate=a.live, console_freshness_h=a.console_freshness_hours,
               client_cap_total=a.client_cap_total, operator_approval=a.operator_approval,
               task_cap=a.task_cap_usd, **rb)
    return 0 if rows[-1].get("stopped") is None else 4


if __name__ == "__main__":
    raise SystemExit(main())
