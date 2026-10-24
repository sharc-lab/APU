"""Overflow detector for the live demo (2026-10-22 operator brief, item 2): after every local call, did the model see
the whole prompt, or was the conversation cut off?

The naive check, "prompt_eval_count < the prompt's own token count", is unsafe: the runner reuses the KV cache of the
longest common prefix with the previous request and evaluates only the rest. On a long session at a window that holds
everything, the newly evaluated part is routinely a small fraction of the prompt
(results/x2_r2_mechanism_logs/llama3.1_8b_ollama_ctx_16384_call2_notools_20260901.ollama_log.txt, task 181:
task.n_tokens = 4065, cached n_tokens = 2981, "prompt eval time = ... / 1084 tokens"), with nothing cut, and any
runtime version whose prompt_eval_count reports only that part would make the naive check fire.

What this module compares instead (the server runs with OLLAMA_DEBUG=1, its own log file, one lane per server, so a
call's log slice is the byte range written between the request and the response):

  used tokens   the total prompt the model was given, cached prefix included: llama-server's own
                "new prompt, n_ctx_slot = W, n_keep = K, task.n_tokens = N" line (N). The first "cached n_tokens = C"
                line after it (the reused prefix) is recorded. If the new-prompt line is missing, only the direct cut
                lines and "own > loaded window" are used (method "log_lines_only").
                Observed, not assumed: in the R2 mechanism run (results/x2_r2_mechanism.jsonl, Ollama 0.34.4) the
                API's prompt_eval_count equalled task.n_tokens in 400 of 400 calls, i.e. this version already reports
                the cached prefix in it; other versions count only the newly evaluated tokens. The detector reads the
                server total either way and records prompt_eval_count_is_total per call.
  own tokens    the prompt's own size before any cut: the harness's chars/4 estimate of the messages and tools sent
                (x2_r2_agent.prompt_tokens_est) times the B3 v2 tokenizer ratio (src/dse/route_profiles.json
                "calibration").
  direct cuts   Ollama's message drop ("truncating input messages which exceed context length", DEBUG; logged on
                every call with the count of messages kept, so a drop is kept < messages sent), its token
                cut ("truncating input prompt", WARN), llama.cpp's context shift ("slot context shift"), or an
                exceed-context error. Patterns are x2_r2_mechanism._PATTERNS (from Ollama 0.34.4's source).

  cut (v1) = any direct cut, or used < own * (1 - COUNT_TOLERANCE).
  cut_v2 (proposed 2026-10-23) = any direct cut only. Why: the v1 count comparison fired at the default window on calls
  with no cut (row detector-precision-recall): our chars/4-based size overstated these long prompts by 8 to 12% for
  qwen3:14b and is off by more for llama3.1:8b; the server's own lines report the cut directly.
  Without a log slice (no server log), the fallback is "own > loaded window" (method "window_only"), never
  prompt_eval_count alone.

Pure functions; unit-tested in tests/test_demo_overflow.py on log lines copied from committed mechanism logs.
"""
from __future__ import annotations

import re

import x2_r2_mechanism as mech

COUNT_TOLERANCE = 0.10
_CACHED_RE = re.compile(r"cached n_tokens = (\d+), memory_seq_rm")
_NEW_PROMPT_RE = re.compile(r"new prompt, n_ctx_slot = (\d+), n_keep = (-?\d+), task\.n_tokens = (\d+)")


def used_from_log(text: str) -> dict:
    """Per call slice: the last new-prompt line's task.n_tokens and slot size, the first cached n_tokens after it."""
    last = None
    for line in (text or "").splitlines():
        m = _NEW_PROMPT_RE.search(line)
        if m:
            last = {"n_ctx_slot": int(m.group(1)), "n_keep": int(m.group(2)), "n_tokens": int(m.group(3)),
                    "cached": None}
            continue
        c = _CACHED_RE.search(line)
        if c and last is not None and last["cached"] is None:
            last["cached"] = int(c.group(1))
    return last or {}


def detect(own_tokens: int, prompt_eval_count: int | None, log_text: str | None, loaded_window: int | None,
           n_messages: int | None = None, tolerance: float = COUNT_TOLERANCE) -> dict:
    """The verdict for one call. own_tokens: calibrated estimate of the whole prompt as sent; n_messages: how many
    messages the request carried (Ollama logs its truncation line on every call, with the number of messages KEPT on
    the Go-template path or the index of the first kept one on the native path, so a drop is kept < n_messages or
    index > 0; mech.first_kept_index)."""
    out = {"own_tokens": own_tokens, "prompt_eval_count": prompt_eval_count, "loaded_window": loaded_window,
           "used_tokens": None, "cached_tokens": None, "evidence": [], "method": None, "cut": False}
    if log_text is None:
        out["method"] = "window_only"
        if loaded_window is not None and own_tokens > loaded_window:
            out["cut"] = True
            out["evidence"].append(f"prompt about {own_tokens} tokens, window {loaded_window}")
        return out
    ev = mech.parse_call_log(log_text)
    idx = mech.first_kept_index(ev, n_messages) if n_messages is not None else None
    if idx is not None and idx > 0:
        out["evidence"].append(f"older messages dropped (first kept: message {idx} of {n_messages})")
    if ev.get("token_cut"):
        out["evidence"].append("prompt tokens cut ({prompt} -> {new})".format(**ev["token_cut"]))
    if ev.get("context_shifts"):
        out["evidence"].append(f"context shift x{len(ev['context_shifts'])}")
    if ev.get("exceed_context_error"):
        out["evidence"].append("exceeds the context size")
    u = used_from_log(log_text)
    if u.get("n_tokens") is not None:
        out["used_tokens"], out["cached_tokens"], out["method"] = u["n_tokens"], u.get("cached"), "server_total"
        out["prompt_eval_count_is_total"] = (prompt_eval_count == u["n_tokens"]
                                             if prompt_eval_count is not None else None)
    else:
        out["method"] = "log_lines_only"
        if loaded_window is not None and own_tokens > loaded_window:
            out["evidence"].append(f"prompt about {own_tokens} tokens, window {loaded_window}")
    lines = [e for e in out["evidence"] if not e.startswith("prompt about")]
    if out["used_tokens"] is not None and out["used_tokens"] < own_tokens * (1 - tolerance):
        out["evidence"].append(f"model saw {out['used_tokens']} of about {own_tokens} tokens")
    out["cut"] = bool(out["evidence"])
    # v2 (2026-10-23, docs/EXPERIMENTS.md "detector v2 re-check"; proposed, re-checked on evo-x2 before the screen uses
    # it): the server's own cut lines only. The token-count comparison above is v1 and stays the recorded `cut`.
    out["cut_v2"] = bool(lines) if log_text is not None else out["cut"]
    out["evidence_v2"] = lines
    return out


def verdict(v: dict, version: str = "v1") -> bool:
    """The detector's answer for a version: v1 (as pre-registered for the 2026-10-23 runs) or v2 (cut lines only)."""
    if version not in ("v1", "v2"):
        raise ValueError(version)
    return bool(v["cut"] if version == "v1" else v["cut_v2"])


def naive_detect(own_tokens: int, prompt_eval_count: int | None, tolerance: float = COUNT_TOLERANCE) -> bool:
    """The check the brief warns about (kept only so the tests can show it misfires under prefix caching)."""
    return prompt_eval_count is not None and prompt_eval_count < own_tokens * (1 - tolerance)
