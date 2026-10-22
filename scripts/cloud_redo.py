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

SOURCE = REPO / "results" / "x2_evalset_tasks_real_v2.jsonl"
OUT = REPO / "results" / "cloud_redo_v1.jsonl"
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


def live_turn(client, model_id: str, es, built: dict, rates: dict, max_tokens: int) -> dict:
    tools = EV.tools_payload(es.design)
    messages = copy.deepcopy(built["messages"])
    live_env = copy.deepcopy(built["env_before"])
    flat, final_text, secs = [], "", 0.0
    tot = {"input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0,
           "uncached_input_tokens": 0, "usd": 0.0, "usd_all_uncached": 0.0, "usd_ledger": 0.0}
    n_calls = 0
    for i in range(MAX_TOOL_ROUNDS + 1):
        offer = i < MAX_TOOL_ROUNDS
        # Anthropic needs tools defined when the transcript holds tool calls: the final call keeps them, choice none
        kw = {"tools": tools, "max_tokens": max_tokens, "custom_llm_provider": "anthropic"}
        if not offer:
            kw["tool_choice"] = "none"
        prepared = client.prepare(model_id=model_id, messages=with_cache_control(messages), **kw)
        t0 = time.perf_counter()
        raw = prepared.send()
        secs += time.perf_counter() - t0
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
        if calls:
            flat.extend(calls)
            _append_call(messages, live_env, content, calls, f"c{i}")
            continue
        final_text = content
        break
    g = grade(es, built["item"], built["env_before"], flat, final_text)
    return {**tot, "seconds": secs, "n_calls": n_calls, "cloud_end_state_ok": g["end_state_ok"],
            "cloud_state_ok": g["state_ok"], "cloud_answer_ok": g["answer_ok"], "cloud_end_state_diff": g["end_state_diff"],
            "cloud_final_text": final_text[:TRUNC_CHARS], "cloud_n_tool_calls": len(flat)}


def warmup(client, model_id: str) -> dict:
    t0 = time.perf_counter()
    res = client.call(model_id=model_id, messages=[{"role": "user", "content": "Reply with the single word OK."}],
                      max_tokens=8, custom_llm_provider="anthropic")
    return {"record": "warmup", "excluded_from_stats": True, "seconds": time.perf_counter() - t0,
            "usd_ledger": res.cost_usd, "input_tokens": res.input_tokens, "output_tokens": res.output_tokens}


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


# ── main ────────────────────────────────────────────────────────────────────────────────────────────

def run(live: bool, source: Path = SOURCE, out: Path = OUT, price_file: Path = PRICE_FILE, limit: int | None = None,
        client=None, max_tokens: int = ag.MAX_TOKENS_PER_CALL, emit_print=print) -> list[dict]:
    table = load_price_table(price_file)
    model_id = escalation_model(table)
    rates = model_rates(table, model_id)
    src = load_rows(source)
    wrong = wrong_tasks(src)
    if limit is not None:
        wrong = wrong[:limit]
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
    rows = [{
        "record": "run_start", "harness": HARNESS, "ts_utc": utc_iso(), "mode": mode,
        "script": "scripts/cloud_redo.py", "script_git_blob": git_blob(Path(__file__)),
        "price_file": price_file.relative_to(REPO).as_posix() if price_file.is_relative_to(REPO) else price_file.name,
        "price_file_git_blob": git_blob(price_file), "price_file_fetched_utc": table.get("fetched_utc"),
        "model_id": model_id, "source": source.relative_to(REPO).as_posix() if source.is_relative_to(REPO)
        else source.name, "source_git_blob": git_blob(source), "n_wrong_tasks": len(wrong),
        "wrong_tasks_per_tier": per_tier, "prompt_caching": True, "spend_cap_usd": client.spend_cap_usd,
        "hard_spend_cap_usd": cc.HARD_SPEND_CAP_USD,
        "pricing_model": ("cache on, cold per task: call 1 writes its prompt (5m write rate), call k reads call k-1's "
                          "prompt (hit rate) and writes the rest; usd_all_uncached = every input token at input rate"),
        "token_basis": "calls[].prompt_tokens_est (harness estimate of the prompt sent), not Anthropic's tokenizer",
        "output_basis": ("local completion_tokens used as a stand-in for the cloud output" if not live
                         else "provider usage"),
        "not_measured": list(NOT_MEASURED_STUB) if not live else [],
    }]
    if live:
        projected = sum(price_turn(local_call_tokens(r), rates)["usd_all_uncached"] for r in wrong)
        if projected > client.remaining_budget_usd:
            raise SystemExit(f"refused: projected ${projected:.2f} (all input uncached) exceeds the remaining budget "
                             f"${client.remaining_budget_usd:.2f} under the ${client.spend_cap_usd:.2f} cap")
        rows.append(warmup(client, model_id))
    cache: dict = {}
    for r in wrong:
        es = build_session_for(r, cache)
        prior = [x for x in session_rows(src, r["attempt_id"]) if x["turn_idx"] < r["turn_idx"]]
        built = rebuild(es, prior, r["turn_idx"])
        if built["item"].task is None or built["item"].task.task_id != r["task_id"]:
            raise RuntimeError(f"rebuilt item {built['item'].item_id} does not match row task {r['task_id']}")
        base = {"record": "cloud_redo_task", "mode": mode, "model_id": model_id,
                "tier": tier_of(r["arm_id"], r.get("call2_mode", "off")), "arm_id": r["arm_id"], "seed": r["seed"],
                "turn_idx": r["turn_idx"], "task_id": r["task_id"], "attempt_id": r["attempt_id"],
                "local_model_id": r["model_id"], "local_n_calls": len(r.get("calls") or []),
                "local_any_http_error": r.get("any_http_error"),
                "local_regrade_end_state_ok": local_regrade(es, built["item"], built["env_before"], r),
                "history_truncated_messages": built["history_truncated_messages"],
                "rebuilt_messages": len(built["messages"])}
        if live:
            res = live_turn(client, model_id, es, built, rates, max_tokens)
        else:
            res = {**price_turn(local_call_tokens(r), rates), "n_calls": len(r.get("calls") or []),
                   "seconds": None, "cloud_end_state_ok": None, "not_measured": list(NOT_MEASURED_STUB)}
        row = {**base, **res, "ts_utc": utc_iso()}
        rows.append(row)
    rows.append({"record": "run_end", "harness": HARNESS, "ts_utc": utc_iso(), "summary": summarize(rows)})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    s = rows[-1]["summary"]
    emit_print(f"{mode}: n={s['n']} total ${s['total_usd']:.4f} mean ${s['mean_usd']:.5f} "
               f"median ${s['median_usd']:.5f} -> {out.name}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true", help="execute on the cloud (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.live and not os.environ.get(KEY_ENV):
        print(f"refused: --live needs {KEY_ENV} set in the environment; nothing was sent", file=sys.stderr)
        return 2
    run(a.live, a.source, a.out, limit=a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
