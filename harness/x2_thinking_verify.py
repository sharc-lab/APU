"""Live verification of which thinking-disable mechanism actually works on evo-x2's deployed runtimes
(llama-server b10970, Ollama), with the raw evidence saved.

Why: the weekend X2 outcome table scored qwen3-8b/14b/32b at 0.05 to 0.22 on llama_server against 0.84 to
1.00 for the non-thinking models. The weekend rows stored no output text, only scores; the llama-server
per-item logs show "chat template, thinking = 1" and n_gen = 256 (the whole max_tokens budget) on most
qwen3-8b/14b/32b calls. This module reproduces those calls live with the weekend settings (baseline) and
then tests each candidate mechanism separately, so the fix is chosen from evidence rather than assumed.

Mechanisms (llama-server):
  baseline            no thinking control at all (the weekend setting)
  request_kwarg       per-request "chat_template_kwargs": {"enable_thinking": false}
  no_think_prompt     " /no_think" appended to the user message (Qwen3 soft switch)
  server_budget0      server flag --reasoning-budget 0
  server_kwarg        server flag --chat-template-kwargs {"enable_thinking":false}
  production          server_budget0 + request_kwarg (what harness/x2_outcome_table.py applies)
Mechanisms (Ollama, qwen3:8b): think omitted (default) vs "think": false.

Per call it records the full request body, the raw response JSON, content vs reasoning_content lengths,
finish_reason, completion tokens and the graded score. A mechanism "works" for a model when every prompt
returns reasoning_chars == 0, non-empty content without a <think> tag, and finish_reason "stop".

It starts and stops its own servers, so it must only run as (part of) a queue job, never alongside a
measurement. harness/x2_outcome_table.py --verify-thinking calls ensure_verified() at startup; standalone:
  py -3.12 x2_thinking_verify.py --out C:\\apu\\ovn\\results\\x2_thinking_verify.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

import x2_outcome_table as x2  # noqa: E402

THINKING_MODELS = ["qwen3-8b", "qwen3-14b", "qwen3-32b"]
N_PROMPTS = 3
VERIFY_N_CTX = 2048


def verify_prompts(items, n=N_PROMPTS):
    return sorted((it for it in items if it["family"] == "gsm8k"), key=lambda it: it["prompt_tokens"])[:n]


def mechanism_works(rows):
    """Thinking is suppressed: no reasoning text and a non-empty answer without a <think> tag, on every prompt.
    finish_reason is NOT part of the test (first version required "stop"): a non-thinking answer to a
    "solve step by step" prompt can legitimately use the whole 256-token budget, which says nothing about
    thinking (live 2026-10-07: qwen3-14b/32b production and Ollama think=false each had one such answer with
    reasoning_chars 0)."""
    return bool(rows) and all(r.get("http_status") == 200 and r.get("reasoning_chars") == 0 and r.get("content_chars", 0) > 0
                              and not r.get("think_tag_in_content") for r in rows)


def works_from_calls(rows):
    """Recompute per-mechanism verdicts from the stored call records with the current mechanism_works."""
    groups = {}
    for r in rows:
        if r.get("record") == "thinking_verify_call":
            groups.setdefault(f"{r['runtime']}/{r['model_id']}/{r['mechanism']}", []).append(r)
    return {k: mechanism_works(v) for k, v in groups.items()}


def _llama_rows(out_path, model_key, mechanism, server_args, request_extra, prompt_suffix, prompts, grade_module):
    _tag, gguf = x2.MODEL_MAP[model_key]
    log_path = str(x2.LLAMA_LOG_DIR / f"x2_thinking_verify_{x2.safe_name(model_key)}_{mechanism}.log")
    srv = x2.LlamaServer(gguf, VERIFY_N_CTX, log_path, extra_args=server_args)
    rows = []
    try:
        err = srv.start(call_timeout_s=900)
        for it in prompts:
            rec = {"record": "thinking_verify_call", "ts_utc": x2.utc_iso(), "runtime": "llama_server",
                   "model_id": model_key, "mechanism": mechanism, "server_args": server_args, "item_id": it["item_id"]}
            if err:
                rec.update({"http_status": None, "error": err})
            else:
                body = {"model": "x", "messages": [{"role": "user", "content": it["prompt"] + prompt_suffix}],
                        "max_tokens": x2.MAX_TOKENS, "temperature": 0}
                body.update(request_extra)
                rec["request_body"] = body
                t0 = time.monotonic()
                try:
                    data = srv.chat(body, timeout=900)
                    content, reasoning, finish, usage = x2.llama_message_fields(data)
                    rec.update({"http_status": 200, "latency_s": time.monotonic() - t0, "raw_response": data,
                                "content_chars": len(content), "reasoning_chars": len(reasoning),
                                "think_tag_in_content": "<think>" in content, "finish_reason": finish,
                                "completion_tokens": usage.get("completion_tokens"),
                                "score": x2.score_response(grade_module, it, x2.strip_thinking(content))})
                except Exception as e:
                    rec.update({"http_status": getattr(e, "code", None), "error": str(e)[:400]})
            x2.emit(out_path, rec)
            rows.append(rec)
    finally:
        srv.stop()
    return rows


def _ollama_rows(out_path, model_key, mechanism, think, prompts, grade_module):
    import host_config as hc
    import t2s_k1_ollama as k1
    tag, _gguf = x2.MODEL_MAP[model_key]
    rows = []
    hc.start_ollama_server()
    try:
        ready = hc.wait_for_ollama_ready(timeout_s=90)
        for it in prompts:
            rec = {"record": "thinking_verify_call", "ts_utc": x2.utc_iso(), "runtime": "ollama",
                   "model_id": model_key, "mechanism": mechanism, "think": think, "item_id": it["item_id"]}
            if not ready:
                rec.update({"http_status": None, "error": "ollama server did not become ready within 90s"})
            else:
                resp = k1.OllamaClient().chat(tag, it["prompt"], max_tokens=x2.MAX_TOKENS, think=think,
                                              keep_alive="2m", timeout=900)
                raw = resp.get("raw") or {}
                content = resp.get("message") or ""
                reasoning = ((raw.get("message") or {}).get("thinking")) or ""
                rec.update({"http_status": resp.get("status"), "error": resp.get("error"), "raw_response": raw,
                            "request_think_field": think, "content_chars": len(content),
                            "reasoning_chars": len(reasoning), "think_tag_in_content": "<think>" in content,
                            "finish_reason": raw.get("done_reason"), "completion_tokens": resp.get("eval_count"),
                            "score": x2.score_response(grade_module, it, x2.strip_thinking(content))})
            x2.emit(out_path, rec)
            rows.append(rec)
    finally:
        hc.stop_ollama_server()
    return rows


def run_verification(out_path, items, models=None):
    models = [m for m in (models or THINKING_MODELS) if m in THINKING_MODELS] or THINKING_MODELS
    prompts = verify_prompts(items)
    g = x2.load_graders()
    kw = {"chat_template_kwargs": {"enable_thinking": False}}
    plan = []
    first = models[0]
    for m in models:
        plan.append((m, "baseline", [], {}, ""))
    plan += [
        (first, "request_kwarg", [], kw, ""),
        (first, "no_think_prompt", [], {}, " /no_think"),
        (first, "server_budget0", ["--reasoning-budget", "0"], {}, ""),
        (first, "server_kwarg", ["--chat-template-kwargs", json.dumps({"enable_thinking": False})], {}, ""),
    ]
    for m in models:
        plan.append((m, "production", list(x2.LLAMA_SERVER_THINKING_ARGS), dict(x2.LLAMA_REQUEST_THINKING_FIELDS), ""))
    results = {}
    for m, mech, sargs, extra, suffix in plan:
        rows = _llama_rows(out_path, m, mech, sargs, extra, suffix, prompts, g)
        results[f"llama_server/{m}/{mech}"] = rows
        print(f"llama_server {m} {mech}: works={mechanism_works(rows)} "
              f"reasoning={[r.get('reasoning_chars') for r in rows]} content={[r.get('content_chars') for r in rows]} "
              f"finish={[r.get('finish_reason') for r in rows]}", flush=True)
    for think, mech in ((None, "ollama_default_think"), (False, "ollama_think_false")):
        rows = _ollama_rows(out_path, first, mech, think, prompts, g)
        results[f"ollama/{first}/{mech}"] = rows
        print(f"ollama {first} {mech}: works={mechanism_works(rows)} reasoning={[r.get('reasoning_chars') for r in rows]}",
              flush=True)
    summary = {"record": "thinking_verify_summary", "ts_utc": x2.utc_iso(),
               "llama_server_exe": x2.LLAMA_SERVER_EXE, "prompts": [p["item_id"] for p in prompts],
               "works": {k: mechanism_works(v) for k, v in results.items()},
               "mean_score": {k: (sum((r.get("score") or 0.0) for r in v) / len(v) if v else None) for k, v in results.items()},
               "reasoning_chars": {k: [r.get("reasoning_chars") for r in v] for k, v in results.items()},
               "content_chars": {k: [r.get("content_chars") for r in v] for k, v in results.items()},
               "finish_reason": {k: [r.get("finish_reason") for r in v] for k, v in results.items()},
               "production_setting_llama": x2.THINKING_SETTING_LLAMA,
               "production_setting_ollama": x2.THINKING_SETTING_OLLAMA}
    x2.emit(out_path, summary)
    return summary


def ensure_verified(out_path, items, models=None):
    """Runs the verification once per evidence file (skipped when a summary record already exists). If the
    production setting fails for any thinking model, writes an ALERT (the canary gate then also halts it)."""
    out_path = Path(out_path)
    rows = x2.read_rows(out_path)
    existing = [r for r in rows if r.get("record") == "thinking_verify_summary"]
    summary = existing[-1] if existing else run_verification(out_path, items, models)
    works = works_from_calls(x2.read_rows(out_path))
    if works != summary.get("works") and not any(r.get("record") == "thinking_verify_reassessed" for r in rows):
        x2.emit(out_path, {"record": "thinking_verify_reassessed", "ts_utc": x2.utc_iso(), "works": works,
                           "note": "verdicts recomputed from the stored calls without the finish_reason=='stop' "
                                   "requirement; supersedes the summary's 'works' and its alert"})
    bad =[k for k, v in works.items() if k.endswith("/production") and not v]
    if not works.get(f"ollama/{THINKING_MODELS[0]}/ollama_think_false", True):
        bad.append("ollama think=false")
    if any(r.get("record") == "alert" and r.get("source") == "x2_thinking_verify" and r.get("failed") == bad
           for r in rows):
        return summary  # already alerted for exactly this, do not repeat on every resume
    if bad:
        x2.emit(out_path, {"record": "alert", "source": "x2_thinking_verify", "ts_utc": x2.utc_iso(),
                           "reason": "production thinking-disable setting failed live verification", "failed": bad})
        print(f"ALERT: production thinking setting failed for {bad}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    items = x2.load_items_trace_weighted(x2.REPO)
    print(json.dumps(run_verification(Path(args.out), items)["works"], indent=1))


if __name__ == "__main__":
    main()
