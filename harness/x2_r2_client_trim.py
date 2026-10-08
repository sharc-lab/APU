"""R2 mitigation: client-side history trimming (x2_r2_agent.py --mode mitigation --client-trim margin=0.05,
results/x2_r2_mitigation_v1.jsonl). Pre-registered in docs/FINDINGS.md ("PRE-REGISTRATION: R2 mitigation
x2_r2_mitigation_v1", 2026-10-08).

Why: the mechanism run (x2_r2_mechanism.py, register rows R2-mechanism-verdict / R2-mechanism-lowlevel) found that at
num_ctx 4096 Ollama keeps the system message when it drops whole messages but leaves too little room for the answer,
so llama.cpp's context shift (n_keep 5) and Ollama's token-level cut (keep 4) discard the start of the prompt, system
prompt included. This module keeps every request inside the window from the client side, so neither can fire.

The rule, applied before EVERY call (call 1 and call 2 of a turn):
  kept prompt tokens <= floor(num_ctx * (1 - margin)) - num_predict
  - num_predict is the harness's per-call answer cap (x2_r2_agent.MAX_TOKENS_PER_CALL), which every request already
    sends as options.num_predict; it is recorded per call;
  - the system prompt (every leading system message) is always kept;
  - whole turns are dropped from the oldest end; a turn is a user message and everything after it up to the next user
    message, so a tool result is never sent without its assistant tool call and no assistant message without its user
    message;
  - the current turn (the last one) is always kept; if system + current turn alone exceed the budget the call is still
    sent and flagged over_budget.
The harness's own transcript is never modified; only the list sent on this call is trimmed.

Token counting: render and count. The candidate request is rendered by Ollama itself (_debug_render_only, the same
template, tools and truncation code path as the real call) and the rendered text is tokenized with the model's own
GGUF tokenizer (prompt_token_check.count_tokens, llama-tokenize). The mechanism run validated exactly this count against
fresh uncached prompt_eval_count on every check in all five tiers (R2-mechanism-verdict). A render that Ollama itself
had to trim (some kept user message or the system prompt missing from the rendered text) means the candidate was over
num_ctx: it counts as over budget. The chars/4 estimate, calibrated by the last exact count of this session, only picks
the first candidate cut. If rendering or tokenizing raises on a call, that call falls back to the calibrated estimate
times FALLBACK_SAFETY and records count_method "estimate_fallback".

After each call the wrapper reads the mechanism wrapper's r2m_call row for that call (Ollama message drops, token
cuts, context shifts, exceed-context errors, llama-server's own task.n_tokens) and attaches it, with the trim record,
to the response as resp["client_trim"]; x2_r2_agent._call_record copies it into the turn row's call record.

Pure functions (split_turns, kept_messages, choose_trim, prompt_budget) are unit-tested with fake counters in
tests/test_x2_r2_client_trim.py; ClientTrimRuntime is driven through a real x2_r2_agent session with a fake runtime.
"""
from __future__ import annotations

import json
import math
import time

DEFAULT_MARGIN = 0.05
FALLBACK_SAFETY = 1.25
MAX_COUNTS_PER_CALL = 8
COUNT_RENDER = "render_tokenize"
COUNT_FALLBACK = "estimate_fallback"


def parse_client_trim(spec: str | None) -> dict | None:
    """'margin=0.05' -> {"margin": 0.05}; '' or None -> None. Only the margin key is accepted."""
    if not spec:
        return None
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        if k != "margin" or not v:
            raise ValueError(f"--client-trim: unknown or empty setting {part!r} (expected margin=<fraction>)")
        m = float(v)
        if not 0.0 <= m < 0.5:
            raise ValueError(f"--client-trim margin {m} out of range [0, 0.5)")
        out["margin"] = m
    return out or {"margin": DEFAULT_MARGIN}


def prompt_budget(num_ctx: int, margin: float, num_predict: int) -> int:
    """Largest prompt (tokens) such that prompt + num_predict <= num_ctx minus the margin."""
    return int(math.floor(num_ctx * (1.0 - margin))) - int(num_predict)


def split_turns(messages: list[dict]) -> tuple[list[int], list[list[int]]]:
    """(indices of the leading system messages, [indices of each turn]). A turn starts at a user message and runs to
    the message before the next user message. Non-system messages before the first user message (none in R2) form a
    turn of their own so nothing is silently lost."""
    head, i = [], 0
    while i < len(messages) and messages[i].get("role") == "system":
        head.append(i)
        i += 1
    turns: list[list[int]] = []
    for j in range(i, len(messages)):
        if messages[j].get("role") == "user" or not turns:
            turns.append([j])
        else:
            turns[-1].append(j)
    return head, turns


def kept_messages(messages, head, turns, s) -> list[dict]:
    """System messages plus turns[s:], in order."""
    idx = list(head) + [j for t in turns[s:] for j in t]
    return [messages[j] for j in idx]


def session_turn_numbers(messages, turns) -> list[int]:
    """1-based session turn number of each turn group: the count of user messages up to and including its first
    message (x2_r2_agent sends exactly one user message per turn, so this is the session's turn index)."""
    out, n = [], 0
    starts = {t[0] for t in turns}
    for j, m in enumerate(messages):
        if m.get("role") == "user":
            n += 1
        if j in starts:
            out.append(n)
    return out


def choose_trim(messages, budget: int, count_fn, est_fn, ratio: float | None = None,
                max_counts: int = MAX_COUNTS_PER_CALL) -> dict:
    """Pick s (the number of oldest turns dropped) as the smallest s whose exact count fits the budget.

    count_fn(kept_messages) -> exact token count (int), None when that candidate is known to be over num_ctx (Ollama
    had to trim its render), or raises on a counting failure. est_fn(kept_messages) -> chars-based estimate.
    ratio: exact/estimate ratio from the last exact count of this session (None = 1.0), used only for the first
    candidate. Returns the trim record (without the messages)."""
    head, turns = split_turns(messages)
    last = max(len(turns) - 1, 0)
    r = ratio if ratio else 1.0
    est_all = est_fn(messages)
    counts: dict[int, int | None] = {}
    method = COUNT_RENDER
    error = None

    def est_s(s):
        return est_fn(kept_messages(messages, head, turns, s))

    s0 = next((s for s in range(0, last + 1) if est_s(s) * r <= budget), last)

    def fits(s):
        if s not in counts:
            if len(counts) >= max_counts:
                raise _CountCap()
            counts[s] = count_fn(kept_messages(messages, head, turns, s))
        n = counts[s]
        return n is not None and n <= budget

    chosen = None
    try:
        if not turns:
            chosen = 0
        elif fits(s0):
            chosen = s0
            while chosen > 0 and fits(chosen - 1):
                chosen -= 1
        else:
            s = s0
            while s < last and not fits(s + 1):
                s += 1
            chosen = s + 1 if s < last else last
    except _CountCap:
        # keep the smallest s known to fit; else the largest one tried (+1, bounded)
        ok = [s for s, n in counts.items() if n is not None and n <= budget]
        chosen = min(ok) if ok else min(max(counts) + 1, last)
        error = f"count cap {max_counts} reached"
    except Exception as e:  # rendering or tokenizing failed: calibrated estimate with a safety factor
        method = COUNT_FALLBACK
        error = f"{e!r}"[:300]
        safety = r * FALLBACK_SAFETY
        chosen = next((s for s in range(0, last + 1) if est_s(s) * safety <= budget), last)
    kept_n = counts.get(chosen) if method == COUNT_RENDER else None
    est_kept = est_s(chosen) if turns else est_fn(messages)
    if method == COUNT_RENDER and kept_n is None and chosen in counts:
        over = True               # even the current turn alone renders past num_ctx
    elif method == COUNT_RENDER and kept_n is None:
        over = None               # count cap hit before this candidate was counted
    elif method == COUNT_RENDER:
        over = kept_n > budget
    else:
        over = est_kept * r * FALLBACK_SAFETY > budget
    nums = session_turn_numbers(messages, turns)
    return {
        "budget_prompt_tokens": budget, "count_method": method, "count_error": error,
        "n_counts": len(counts), "counts_by_dropped": {str(k): v for k, v in sorted(counts.items())},
        "client_est_full_prompt_tokens": round(est_all * r), "client_est_full_raw": est_all,
        "client_est_kept_prompt_tokens": round(est_kept * r), "calib_ratio_used": ratio,
        "kept_prompt_tokens_exact": kept_n, "over_budget": over,
        "n_turns_total": len(turns), "n_turns_dropped": chosen,
        "dropped_turns": nums[:chosen], "kept_turns": nums[chosen:],
        "n_messages_full": len(messages),
        "n_messages_sent": len(head) + sum(len(t) for t in turns[chosen:]),
        "_chosen": chosen,
    }


class _CountCap(Exception):
    pass


def render_was_trimmed(rendered: str, messages: list[dict]) -> bool:
    """Did Ollama drop messages from this candidate while rendering it (the candidate was over num_ctx)? Ollama drops
    from the front only (x2_r2_mechanism docstring), so the candidate's first user message is missing from the render
    exactly when it trimmed. Raises if no user message of the candidate appears verbatim at all: then the template does
    not insert content verbatim and the check cannot be made (the caller falls back to the estimate, visibly)."""
    users = [m.get("content") or "" for m in messages if m.get("role") == "user"]
    if not users:
        return False
    present = [bool(u) and u in rendered for u in users]
    if not any(present):
        raise RuntimeError("no user message of the candidate found verbatim in the render")
    return not present[0]


class RenderTokenCounter:
    """Exact prompt tokens of a candidate request: Ollama _debug_render_only (base.render_only) + the model's own GGUF
    tokenizer (prompt_token_check.count_tokens). Returns None when the render shows Ollama had to trim the candidate
    (it was over num_ctx), raises on any render/tokenize failure."""

    def __init__(self, base, models_dir=None, exe=None, run=None, tmp_dir=None):
        import prompt_token_check as ptc
        self.ptc = ptc
        self.base = base
        self.models_dir, self.exe, self.run, self.tmp_dir = models_dir, exe or ptc.TOKENIZE_EXE, run, tmp_dir
        self._blobs = {}

    def blob(self, model):
        if model not in self._blobs:
            self._blobs[model] = self.ptc.resolve_model_blob(self.models_dir or self.ptc.default_models_dir(), model)
        return self._blobs[model]

    def __call__(self, model, messages, num_ctx, tools, think, extra=None):
        rendered, err = self.base.render_only(model, [dict(m) for m in messages], num_ctx, tools, think, extra)
        if rendered is None:
            raise RuntimeError(f"render-only failed: {err}")
        if render_was_trimmed(rendered, messages):
            return None
        return self.ptc.count_tokens(rendered, self.blob(model), self.exe, self.run, self.tmp_dir)["n_tokens"]

    def self_check(self, models) -> dict:
        """Startup check, recorded in the run: each model's GGUF blob resolves and a known short string tokenizes."""
        out = {}
        for m in models:
            try:
                n = self.ptc.count_tokens("Hello world", self.blob(m), self.exe, self.run, self.tmp_dir)["n_tokens"]
                out[m] = {"ok": isinstance(n, int) and n > 0, "n_tokens_hello_world": n, "gguf": str(self.blob(m))}
            except Exception as e:
                out[m] = {"ok": False, "error": f"{e!r}"[:300]}
        return out


_OBSERVED_KEYS = ("message_level_truncation", "message_truncation_in_log", "token_level_cut", "token_cut",
                  "context_shift_fired", "exceed_context_error", "log_slice_complete", "dropped_turns",
                  "system_kept", "render_ok")


class ClientTrimRuntime:
    """Wraps a runtime (normally x2_r2_mechanism.MechanismRuntime around OllamaRuntime). chat() trims what it sends
    per choose_trim, forwards, then attaches the trim record and the observed log events for that call to the
    response as resp["client_trim"]. counter(model, messages, num_ctx, tools, think, extra) -> exact count."""

    def __init__(self, inner, counter, margin=DEFAULT_MARGIN, num_predict=None, est_fn=None, clock=time.monotonic,
                 mech=None):
        import x2_r2_agent as ag
        self.inner, self.counter, self.margin = inner, counter, margin
        self.num_predict = ag.MAX_TOKENS_PER_CALL if num_predict is None else num_predict
        self.est_fn = est_fn or (lambda msgs: ag.transcript_tokens(msgs))
        self._tools_est = lambda tools: ag.est_tokens(json.dumps(tools)) if tools else 0
        self.clock = clock
        self.mech = mech if mech is not None else (inner if hasattr(inner, "session_calls") else None)
        self.ratio = None
        self.records = []

    # delegation
    def loaded_context(self, model):
        return self.inner.loaded_context(model)

    def unload(self, model):
        return self.inner.unload(model)

    def recover(self):
        return self.inner.recover()

    def begin_session(self):
        self.ratio = None
        self.records = []

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        t0 = self.clock()
        if num_ctx is None:
            rec = {"applied": False, "reason": "num_ctx not set (runtime default): nothing to trim against"}
            sent = messages
            chosen = 0
        else:
            budget = prompt_budget(num_ctx, self.margin, self.num_predict)

            def count(msgs):
                return self.counter(model, msgs, num_ctx, tools, think, extra)

            def est(msgs):
                return self.est_fn(msgs) + self._tools_est(tools)

            rec = choose_trim(messages, budget, count, est, self.ratio)
            chosen = rec.pop("_chosen")
            head, turns = split_turns(messages)
            sent = kept_messages(messages, head, turns, chosen)
            if rec["kept_prompt_tokens_exact"]:
                raw = est(sent)
                if raw:
                    self.ratio = rec["kept_prompt_tokens_exact"] / raw
            rec.update({"applied": True, "num_ctx": num_ctx, "margin": self.margin})
        rec["num_predict"] = self.num_predict
        rec["trim_wall_s"] = self.clock() - t0
        n_users = sum(1 for m in messages if m.get("role") == "user")
        if self.mech is not None:
            self.mech.call_meta = {"turn_idx": n_users, "client_trim_turns_dropped": chosen,
                                   "client_trim_kept_turns": rec.get("kept_turns")}
        resp = (self.inner.chat(model, sent, num_ctx, tools, think, extra=extra) if extra
                else self.inner.chat(model, sent, num_ctx, tools, think))
        if self.mech is not None:
            self.mech.call_meta = None
            row = self.mech.session_calls[-1] if self.mech.session_calls else {}
            np_ = row.get("new_prompt") or []
            obs = {k: row.get(k) for k in _OBSERVED_KEYS}
            obs.update({"context_shifts": len(row.get("context_shifts") or []),
                        "context_shift_events": row.get("context_shifts") or [],
                        "server_prompt_tokens": np_[-1]["n_tokens"] if np_ else None,
                        "server_n_ctx_slot": np_[-1]["n_ctx_slot"] if np_ else None,
                        "prompt_eval_count": resp.get("prompt_eval_count")})
            ex = rec.get("kept_prompt_tokens_exact")
            obs["client_exact_vs_server_diff"] = (ex - obs["server_prompt_tokens"]
                                                  if isinstance(ex, int) and isinstance(obs["server_prompt_tokens"], int)
                                                  else None)
            rec["observed"] = obs
        resp = dict(resp)
        resp["client_trim"] = rec
        self.records.append(rec)
        return resp


def call_trim_records(rows: list[dict]):
    """(turn row, call index 1/2, client_trim record) for every call of every mitigation turn row."""
    for r in rows:
        if r.get("record") != "r2a_turn":
            continue
        for i, c in enumerate(r.get("calls") or [], 1):
            ct = c.get("client_trim")
            if ct is not None:
                yield r, i, ct


def summarize_calls(recs: list[dict]) -> dict:
    """Counts over client_trim records (one per call)."""
    obs = [c.get("observed") or {} for c in recs]
    diffs = [o.get("client_exact_vs_server_diff") for o in obs if o.get("client_exact_vs_server_diff") is not None]
    exact = [c["kept_prompt_tokens_exact"] for c in recs if isinstance(c.get("kept_prompt_tokens_exact"), int)]
    return {
        "n_calls": len(recs),
        "n_calls_trimmed": sum(1 for c in recs if c.get("n_turns_dropped")),
        "count_method": {m: sum(1 for c in recs if c.get("count_method") == m) for m in (COUNT_RENDER, COUNT_FALLBACK)},
        "over_budget_calls": sum(1 for c in recs if c.get("over_budget")),
        "max_kept_prompt_tokens_exact": max(exact) if exact else None,
        "budget_prompt_tokens": sorted({c.get("budget_prompt_tokens") for c in recs if c.get("budget_prompt_tokens")}),
        "num_predict": sorted({c.get("num_predict") for c in recs if c.get("num_predict") is not None}),
        "max_turns_kept": max((len(c.get("kept_turns") or []) for c in recs), default=None),
        "min_turns_kept": min((len(c.get("kept_turns") or []) for c in recs), default=None),
        "observed_calls": sum(1 for o in obs if o),
        "ollama_message_drop_calls": sum(1 for o in obs if o.get("message_level_truncation")),
        "token_cut_calls": sum(1 for o in obs if o.get("token_level_cut")),
        "context_shift_events": sum(o.get("context_shifts") or 0 for o in obs),
        "context_shift_calls": sum(1 for o in obs if o.get("context_shifts")),
        "exceed_context_error_calls": sum(1 for o in obs if o.get("exceed_context_error")),
        "log_slice_incomplete_calls": sum(1 for o in obs if o and o.get("log_slice_complete") is False),
        "client_vs_server_tokens_compared": len(diffs),
        "client_vs_server_tokens_equal": sum(1 for d in diffs if d == 0),
        "client_vs_server_max_abs_diff": max((abs(d) for d in diffs), default=None),
    }


def summarize(rows: list[dict]) -> dict:
    """Per "model|arm": summarize_calls over that cell's mitigation calls (the job's .report.json)."""
    by = {}
    for r, _i, ct in call_trim_records(rows):
        by.setdefault(f"{r.get('model_id')}|{r.get('arm_id')}", []).append(ct)
    return {k: summarize_calls(v) for k, v in sorted(by.items())}
