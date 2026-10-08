"""R2 overflow mechanism, observed directly (x2_r2_agent.py --mode mechanism, results/x2_r2_mechanism.jsonl).

Question: when an R2 chat transcript outgrows the loaded context, what does Ollama 0.34.4 actually send to the model?
"Drops whole old turns, keeps the system prompt" (docs/R2_DESIGN.md, from Ollama's chat truncation) or the
single-prompt behaviour in docs/FINDINGS.md (ollama-overflow-keeps-half: the tail is kept, a fixed half window)?

What Ollama 0.34.4 does and logs, from its source at tag v0.34.4 (llama.cpp b11081 per its LLAMA_CPP_VERSION):
  1. Message-level truncation, before rendering the final prompt. server/prompt.go chatPrompt (models served with an
     Ollama Go template, e.g. the library llama3.1:8b, qwen3:8b, qwen3:14b) and server/routes.go
     truncateNativeChatMessages (models without one, served through llama-server's jinja template, e.g. the
     GGUF-created qwen3-4b-2507): drop messages from the front until the rendered prompt's token count fits num_ctx
     (the runner's context length, optionsForPrompt), always keeping every system message among the dropped ones
     and the last message. Logged at DEBUG only:
       msg="truncating input messages which exceed context length" truncated=<number of messages KEPT>
       msg="truncating native chat messages which exceed context length" truncated=<index of first kept message>
  2. Token-level cut, only on the Go-template path, only if the rendered prompt still has more than num_ctx - 1
     tokens: llm/llama_server.go completionPromptForRequest keeps the first num_keep tokens (default 4, +1 for BOS)
     and the tail, limit = num_ctx - (num_ctx - keep) / 2. This is the single-prompt half-window rule. Logged at
     WARN (visible without debug):
       msg="truncating input prompt" limit=<n> prompt=<n> keep=<n> new=<n>
  3. llama.cpp context shift during generation: Ollama launches llama-server with --context-shift (and --keep
     <num_keep> when > 0; llm/llama_server.go appendContextShiftArgs) and --log-verbosity 4, and forwards its
     output into the Ollama server log. tools/server/server-context.cpp pre_decode logs at WARN:
       slot context shift, n_keep = <n>, n_left = <n>, n_discard = <n>
     and each new task at TRACE (shown at verbosity 4): new prompt, n_ctx_slot = <n>, n_keep = <n>,
     task.n_tokens = <n>. A prompt at or over the slot context is refused with "request (<n> tokens) exceeds the
     available context size" (ERROR_TYPE_EXCEED_CONTEXT_SIZE, HTTP 400) instead.
  At DEBUG Ollama also logs per call "llama-server completion request" media=<n> prompt_len=<chars> (Go-template
  path) or "llama-server chat request" messages=<n> tools=<n> (native path). It never logs the rendered prompt text.

So the rendered prompt is taken from Ollama itself: before every real call the same request is sent once with
"_debug_render_only": true, which runs the same truncation (chatPrompt / truncateNativeChatMessages run before the
DebugRenderOnly return in ChatHandler / handleNativeChat) and returns the rendered prompt without generating. Each
of the session's user messages is then looked up in that text (whole-content substring), which gives which turns
were dropped, whether the system prompt is in it, and whether the kept messages are a contiguous tail. The server
log slice of the real call (byte offsets before and after it) gives the message count Ollama kept, any token-level
cut and any context shift. Both go into one structured row per call (record r2m_call); the session's raw log slice
is written next to the output file (<out stem>_logs/<model>_<arm>_<seed>.ollama_log.txt), as is the full rendered
prompt of the first truncated call of each session.

Validity check (operator requirement; harness/prompt_token_check.py has the exact commands): on a schedule of calls
per session (the first pre-overflow call and every 10th one after it, up to 5; the first 3 truncated calls and every
4th one after, up to 8) the render-only text is tokenized with the model's own tokenizer (llama-tokenize on the GGUF
blob from the Ollama manifest) and compared with prompt_eval_count of a fresh request (model unloaded first, so no
cache) with the same messages and options and num_predict 1. Each check is its own record r2m_prompt_check, sent
after the real call, so the real call's row, log slice and timing are untouched; the next real call starts on a
freshly loaded model instead of the previous call's KV cache. A tier is mechanism_citable only if it has at least
5 checks and all match within 1%; otherwise the report states "render does not show what the model saw; mechanism
result not citable" for it, and the per-tier hypothesis is reported but not citable.

Pure functions (parse_call_log, analyse_render, summarize_*) are unit-tested on synthetic log text; MechanismRuntime
wraps any runtime with chat/render_only and works with the tests' fake runtime and a temp log file.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import prompt_token_check as ptc

OLLAMA_DEBUG_ENV = {"OLLAMA_DEBUG": "1"}
MECH_SERVE_LOG = r"C:\apu\ovn\ollama_serve_mechanism.log"
EXCERPT_MAX_LINES = 60
EXCERPT_LINE_CHARS = 500
GIN_WAIT_S = 3.0
CHECK_PRE_EVERY, CHECK_PRE_MAX = 10, 5         # pre-overflow calls checked: index 0, 10, 20, ... up to 5
CHECK_TRUNC_FIRST, CHECK_TRUNC_EVERY, CHECK_TRUNC_MAX = 3, 4, 8   # truncated: 0, 1, 2, then 4, 8, ... up to 8

_PATTERNS = {
    "trunc_rendered": re.compile(r'msg="truncating input messages which exceed context length" truncated=(\d+)'),
    "trunc_native": re.compile(r'msg="truncating native chat messages which exceed context length" truncated=(\d+)'),
    "completion_request": re.compile(r'msg="llama-server completion request" media=(\d+) prompt_len=(\d+)'),
    "chat_request": re.compile(r'msg="llama-server chat request" messages=(\d+) tools=(\d+)'),
    "token_cut": re.compile(r'msg="truncating input prompt" limit=(\d+) prompt=(\d+) keep=(\d+) new=(\d+)'),
    "new_prompt": re.compile(r"new prompt, n_ctx_slot = (\d+), n_keep = (-?\d+), task\.n_tokens = (\d+)"),
    "context_shift": re.compile(r"context shift, n_keep = (-?\d+), n_left = (-?\d+), n_discard = (-?\d+)"),
    "exceed_context": re.compile(r"(exceeds the available context size|is larger than the max context size)"),
    "stop_processing": re.compile(r"stop processing: n_tokens = (\d+), truncated = (\d+)"),
    "gin": re.compile(r"\[GIN\].*?\|\s*(\d{3})\s*\|.*?\"(/api/[\w/]+)\""),
}
_ALSO_EXCERPT = re.compile(r"level=(WARN|ERROR)|starting llama-server|server config")


def sha256_text(s):
    return None if s is None else hashlib.sha256(s.encode("utf-8")).hexdigest()


# ── log file access ─────────────────────────────────────────────────────────────────────────────────

def log_size(path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def read_log(path, start: int, end: int | None = None) -> str:
    """Bytes [start, end) of the server log as text (errors replaced). The server holds the file open for writing;
    reading it is the same thing t2s_k2_pressure does with its own serve log."""
    try:
        with open(path, "rb") as f:
            f.seek(max(start, 0))
            data = f.read() if end is None else f.read(max(end - start, 0))
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def debug_enabled(log_text: str) -> bool:
    """True if the server's own startup line shows debug level (server/routes.go logs "server config" with
    envconfig.Values(); OLLAMA_DEBUG=1 renders as OLLAMA_DEBUG:DEBUG) or any DEBUG-level line was written."""
    return "OLLAMA_DEBUG:DEBUG" in log_text or "level=DEBUG" in log_text


# ── parsing (pure) ──────────────────────────────────────────────────────────────────────────────────

def _ints(m):
    return [int(x) for x in m.groups()]


def parse_call_log(text: str) -> dict:
    """Structured events from one call's slice of the server log (see the module docstring for each line)."""
    ev = {"chat_path": None, "msg_truncation_value": None, "completion_prompt_len": None,
          "chat_request_messages": None, "chat_request_tools": None, "token_cut": None, "new_prompt": [],
          "context_shifts": [], "exceed_context_error": None, "stop_processing_truncated": None,
          "gin_status": [], "excerpt": []}
    for line in (text or "").splitlines():
        hit = False
        for name, pat in _PATTERNS.items():
            m = pat.search(line)
            if not m:
                continue
            hit = True
            if name == "trunc_rendered":
                ev["chat_path"], ev["msg_truncation_value"] = "rendered", int(m.group(1))
            elif name == "trunc_native":
                ev["chat_path"], ev["msg_truncation_value"] = "native", int(m.group(1))
            elif name == "completion_request":
                ev["chat_path"] = ev["chat_path"] or "rendered"
                ev["completion_prompt_len"] = int(m.group(2))
            elif name == "chat_request":
                ev["chat_path"] = ev["chat_path"] or "native"
                ev["chat_request_messages"], ev["chat_request_tools"] = _ints(m)
            elif name == "token_cut":
                ev["token_cut"] = dict(zip(("limit", "prompt", "keep", "new"), _ints(m)))
            elif name == "new_prompt":
                ev["new_prompt"].append(dict(zip(("n_ctx_slot", "n_keep", "n_tokens"), _ints(m))))
            elif name == "context_shift":
                ev["context_shifts"].append(dict(zip(("n_keep", "n_left", "n_discard"), _ints(m))))
            elif name == "exceed_context":
                ev["exceed_context_error"] = line.strip()[:EXCERPT_LINE_CHARS]
            elif name == "stop_processing":
                ev["stop_processing_truncated"] = int(m.group(2))
            elif name == "gin":
                ev["gin_status"].append([int(m.group(1)), m.group(2)])
        if (hit or _ALSO_EXCERPT.search(line)) and len(ev["excerpt"]) < EXCERPT_MAX_LINES:
            ev["excerpt"].append(line.strip()[:EXCERPT_LINE_CHARS])
    return ev


def message_turns(messages: list[dict]) -> list[int]:
    """Turn index of each message: system messages before the first user message are turn 0, and each user
    message starts the next turn (x2_r2_agent sends exactly one user message per turn)."""
    out, t = [], 0
    for m in messages:
        if m.get("role") == "user":
            t += 1
        out.append(t)
    return out


def analyse_render(rendered: str | None, messages: list[dict]) -> dict:
    """What the rendered (post-truncation) prompt contains, judged by whole-content substring matches of the
    system message and of each user message (unique per turn: task text plus that turn's filler)."""
    if rendered is None:
        return {"render_ok": False}
    turns = message_turns(messages)
    sys_contents = [m.get("content") or "" for m in messages if m.get("role") == "system"]
    users = [(turns[i], m.get("content") or "") for i, m in enumerate(messages) if m.get("role") == "user"]
    present = [t for t, c in users if c and c in rendered]
    all_turns = [t for t, _ in users]
    first_kept = min(present) if present else None
    contiguous = bool(present) and present == list(range(first_kept, all_turns[-1] + 1))
    unique = len({c for _, c in users}) == len(users) and not any(
        a != b and a in b for (_, a) in users for (_, b) in users)
    return {"render_ok": True, "rendered_chars": len(rendered), "rendered_sha256": sha256_text(rendered),
            "system_kept": bool(sys_contents) and all(c in rendered for c in sys_contents),
            "n_user_messages_sent": len(users), "user_turns_kept": present,
            "dropped_turns": [t for t in all_turns if t not in present],
            "first_kept_turn": first_kept, "kept_is_contiguous_tail": contiguous,
            "current_user_message_kept": bool(users) and users[-1][1] in rendered,
            "user_messages_unique": unique}


def first_kept_index(ev: dict, n_messages: int) -> int | None:
    """Index (into the messages sent) of the first non-system message Ollama kept, from its debug line: the
    Go-template path logs the number of messages kept, the native path the index of the first kept one. None if
    no truncation line was logged (nothing dropped)."""
    v = ev.get("msg_truncation_value")
    if v is None:
        return None
    return n_messages - v if ev.get("chat_path") == "rendered" else v


def call_verdict(render: dict, ev: dict, messages: list[dict]) -> dict:
    """Per-call answers to the mechanism question, from the render analysis and the log events together."""
    n = len(messages)
    idx = first_kept_index(ev, n)
    turns = message_turns(messages)
    msg_trunc_log = idx is not None and idx > 0
    msg_trunc_render = bool(render.get("render_ok")) and bool(render.get("dropped_turns"))
    out = {
        "message_level_truncation": msg_trunc_log or msg_trunc_render,
        "message_truncation_in_log": msg_trunc_log,
        "first_kept_message_index_log": idx,
        "n_messages_dropped_log": (idx - sum(1 for m in messages[:idx] if m.get("role") == "system")
                                   if msg_trunc_log else 0),
        "first_kept_message_role_log": messages[idx]["role"] if msg_trunc_log and idx < n else None,
        "first_kept_message_turn_log": turns[idx] if msg_trunc_log and idx < n else None,
        "cut_at_turn_boundary_log": (messages[idx]["role"] == "user") if msg_trunc_log and idx < n else None,
        "system_kept": render.get("system_kept"),
        "token_level_cut": ev.get("token_cut") is not None,
        "context_shift_fired": bool(ev.get("context_shifts")),
        "exceed_context_error": ev.get("exceed_context_error") is not None,
    }
    if msg_trunc_log and idx < n and render.get("render_ok"):
        # the render's first kept user turn must agree with the log's cut position (a cut that lands on a tool or
        # assistant message keeps the rest of that turn, so its user message is gone and the next turn is first)
        exp = turns[idx] if messages[idx]["role"] == "user" else turns[idx] + 1
        out["render_and_log_agree"] = render.get("first_kept_turn") == exp
    else:
        out["render_and_log_agree"] = None
    return out


def summarize_session(call_rows: list[dict]) -> dict:
    """One session's mechanism summary from its r2m_call rows (real calls only)."""
    rows = sorted(call_rows, key=lambda r: (r["turn_idx"], r["call_idx"]))
    trunc = [r for r in rows if r.get("message_level_truncation")]
    first = trunc[0] if trunc else None
    return {
        "n_calls": len(rows),
        "chat_paths": sorted({r["chat_path"] for r in rows if r.get("chat_path")}),
        "n_calls_message_truncated": len(trunc),
        "first_truncated_turn": first["turn_idx"] if first else None,
        "first_truncated_call": [first["turn_idx"], first["call_idx"]] if first else None,
        "dropped_turns_at_first_truncation": first.get("dropped_turns") if first else None,
        "dropped_turns_at_last_call": rows[-1].get("dropped_turns") if rows else None,
        "system_kept_on_every_truncated_call": all(r.get("system_kept") is True for r in trunc) if trunc else None,
        "kept_contiguous_tail_on_every_truncated_call": (all(r.get("kept_is_contiguous_tail") is True for r in trunc)
                                                         if trunc else None),
        "current_message_kept_on_every_call": all(r.get("current_user_message_kept") is not False for r in rows),
        "cut_at_turn_boundary_calls": sum(1 for r in trunc if r.get("cut_at_turn_boundary_log") is True),
        "cut_mid_turn_calls": sum(1 for r in trunc if r.get("cut_at_turn_boundary_log") is False),
        "token_level_cut_calls": sum(1 for r in rows if r.get("token_level_cut")),
        "context_shift_calls": sum(1 for r in rows if r.get("context_shift_fired")),
        "context_shift_events": sum(len(r.get("context_shifts") or []) for r in rows),
        "exceed_context_error_calls": sum(1 for r in rows if r.get("exceed_context_error")),
        "render_log_disagreements": sum(1 for r in rows if r.get("render_and_log_agree") is False),
        "render_failures": sum(1 for r in rows if not r.get("render_ok")),
        "log_slices_incomplete": sum(1 for r in rows if r.get("log_slice_complete") is False),
        "max_new_prompt_tokens": max((p["n_tokens"] for r in rows for p in r.get("new_prompt") or []), default=None),
        "n_ctx_slot_values": sorted({p["n_ctx_slot"] for r in rows for p in r.get("new_prompt") or []}),
    }


def verdict(session_summaries: list[dict]) -> dict:
    """Across sessions: does "drops whole old turns, keeps the system prompt" hold, and is the single-prompt
    tail-keeping cut (token level) seen at all?
      confirmed     : truncation was observed, and on every truncated call the system prompt was in the rendered
                      prompt, the kept user messages were a contiguous tail, and no token-level cut was logged;
      refuted       : some truncated call lost the system prompt, kept a non-contiguous set, or was token-cut;
      not_observed  : no call was truncated.
    Context shift is reported separately (it acts on the KV cache during generation, after the prompt is set)."""
    s = [x for x in session_summaries if x.get("n_calls")]
    trunc = [x for x in s if x["n_calls_message_truncated"]]
    if not trunc:
        hyp = "not_observed"
    elif all(x["system_kept_on_every_truncated_call"] and x["kept_contiguous_tail_on_every_truncated_call"]
             and x["token_level_cut_calls"] == 0 for x in trunc):
        hyp = "confirmed"
    else:
        hyp = "refuted"
    return {"drop_old_turns_keep_system": hyp,
            "sessions_with_truncation": len(trunc), "sessions": len(s),
            "token_level_cut_seen": any(x["token_level_cut_calls"] for x in s),
            "context_shift_seen": any(x["context_shift_calls"] for x in s),
            "exceed_context_error_seen": any(x["exceed_context_error_calls"] for x in s),
            "render_log_disagreements": sum(x["render_log_disagreements"] for x in s),
            "log_slices_incomplete": sum(x["log_slices_incomplete"] for x in s)}


# ── runtime wrapper ─────────────────────────────────────────────────────────────────────────────────

def check_due(post_overflow: bool, k: int, n_done: int) -> bool:
    """Is the k-th (0-based) pre-overflow / truncated call of a session checked, given n_done checks of that kind."""
    if post_overflow:
        return n_done < CHECK_TRUNC_MAX and (k < CHECK_TRUNC_FIRST or k % CHECK_TRUNC_EVERY == 0)
    return n_done < CHECK_PRE_MAX and k % CHECK_PRE_EVERY == 0


def default_checker(inner):
    """A PromptChecker for a real runtime (one with _post and unload); None for runtimes without them."""
    import sys
    if not (hasattr(inner, "_post") and hasattr(inner, "unload")):
        return None
    mod = sys.modules.get(type(inner).__module__)
    body_fn = getattr(mod, "native_chat_body", None)
    return ptc.PromptChecker(inner, body_fn) if body_fn else None


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


class MechanismRuntime:
    """Wraps a runtime (chat, render_only, loaded_context, unload, recover). Around every chat call: a render-only
    request with identical arguments, the server-log slice of each, and one r2m_call row. begin_session /
    end_session bracket a session (raw log slice and an r2m_session row)."""

    def __init__(self, inner, log_path, emit, raw_dir: Path, sleep=time.sleep, clock=time.monotonic,
                 gin_wait_s=GIN_WAIT_S, checker="default"):
        self.inner, self.log_path, self.emit, self.raw_dir = inner, log_path, emit, Path(raw_dir)
        self.checker = default_checker(inner) if checker == "default" else checker
        self.sleep, self.clock, self.gin_wait_s = sleep, clock, gin_wait_s
        self.ctx = {}
        self.session_calls = []
        self.session_start = 0
        self._saved_rendered = False
        # set per call by a wrapper (x2_r2_client_trim.ClientTrimRuntime): fields merged into this call's r2m_call row
        # last, e.g. the session turn index when the messages sent were trimmed by the client
        self.call_meta = None
        self._reset_checks()

    def _reset_checks(self):
        self.session_checks = []
        self._k = {True: 0, False: 0}
        self._n_checked = {True: 0, False: 0}

    # delegation
    def loaded_context(self, model):
        return self.inner.loaded_context(model)

    def unload(self, model):
        return self.inner.unload(model)

    def recover(self):
        return self.inner.recover()

    def begin_session(self, model, arm, seed, mode="mechanism", call2_mode=None):
        self.ctx = {"mode": mode, "model_id": model, "arm_id": arm, "seed": seed, "call2_mode": call2_mode}
        self.session_calls = []
        self.session_start = log_size(self.log_path)
        self._saved_rendered = False
        self._reset_checks()

    def _wait_for_gin(self, start):
        """Position after this request's GIN access line (Gin logs it after the handler returns, which can be a
        moment after the client has the response). (pos, complete)."""
        deadline = self.clock() + self.gin_wait_s
        while True:
            pos = log_size(self.log_path)
            if pos < start:  # server restarted (recover): the log was truncated
                return pos, False
            if _PATTERNS["gin"].search(read_log(self.log_path, start, pos)):
                return pos, True
            if self.clock() >= deadline:
                return pos, False
            self.sleep(0.1)

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        snap = [dict(m) for m in messages]
        p0 = log_size(self.log_path)
        rendered, rerr = self.inner.render_only(model, snap, num_ctx, tools, think, extra)
        p1, _ = self._wait_for_gin(p0)
        resp = (self.inner.chat(model, messages, num_ctx, tools, think, extra=extra) if extra
                else self.inner.chat(model, messages, num_ctx, tools, think))
        p2, complete = self._wait_for_gin(p1)
        call_log = read_log(self.log_path, p1, p2) if p2 >= p1 else ""
        render_log = read_log(self.log_path, p0, p1) if p1 >= p0 else ""
        ev = parse_call_log(call_log)
        rev = parse_call_log(render_log)
        ra = analyse_render(rendered, snap)
        turn_idx = sum(1 for m in snap if m.get("role") == "user")
        call_idx = 1 if snap and snap[-1].get("role") == "user" else 2
        row = {"record": "r2m_call", **self.ctx, "num_ctx_requested": num_ctx, "think": think,
               "turn_idx": turn_idx, "call_idx": call_idx, "n_messages_sent": len(snap),
               "tools_sent": tools is not None, "render_error": rerr,
               "http_status": resp.get("status"), "outcome": resp.get("outcome"),
               "prompt_eval_count": resp.get("prompt_eval_count"),
               "done_reason": (resp.get("raw") or {}).get("done_reason"),
               "log_bytes": [p1, p2], "log_slice_complete": complete,
               "render_log_msg_truncation_value": rev.get("msg_truncation_value"), "ts_utc": _utc()}
        row.update(ra)
        row.update({k: v for k, v in ev.items() if k != "excerpt"})
        row.update(call_verdict(ra, ev, snap))
        row["log_excerpt"] = ev["excerpt"]
        if self.call_meta:
            row.update(self.call_meta)
            turn_idx = row["turn_idx"]
        if row["message_level_truncation"] and rendered is not None and not self._saved_rendered:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            p = self.raw_dir / (f"{_safe(self.ctx.get('model_id', model))}_{_safe(str(self.ctx.get('arm_id')))}_"
                                f"{self.ctx.get('seed')}_turn{turn_idx}_call{call_idx}.rendered.txt")
            p.write_text(rendered, encoding="utf-8")
            row["rendered_prompt_file"] = p.name
            self._saved_rendered = True
        self.emit(row)
        self.session_calls.append(row)
        post = bool(row["message_level_truncation"])
        k = self._k[post]
        self._k[post] += 1
        if self.checker is not None and check_due(post, k, self._n_checked[post]):
            self._n_checked[post] += 1
            self._prompt_check(model, snap, num_ctx, tools, think, extra, rendered, row)
        return resp

    def _prompt_check(self, model, snap, num_ctx, tools, think, extra, rendered, call_row):
        """After the real call: fresh-load prompt_eval_count vs the render's own-tokenizer count (record
        r2m_prompt_check). Any exception becomes an error row, never a skipped check."""
        try:
            res = self.checker.check(model, snap, num_ctx, tools, think, extra, rendered)
        except Exception as e:
            res = {"error": f"check: {e!r}"[:400], "match": False, "diff": None, "rel_diff": None}
        row = {"record": "r2m_prompt_check", **self.ctx, "turn_idx": call_row["turn_idx"],
               "call_idx": call_row["call_idx"], "post_overflow": bool(call_row["message_level_truncation"]),
               "num_ctx_requested": num_ctx, "rendered_sha256": call_row.get("rendered_sha256"),
               "real_call_prompt_eval_count": call_row.get("prompt_eval_count"), **res, "ts_utc": _utc()}
        self.emit(row)
        self.session_checks.append(row)

    def end_session(self) -> dict:
        end = log_size(self.log_path)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        name = (f"{_safe(self.ctx.get('model_id', ''))}_{_safe(str(self.ctx.get('arm_id')))}_"
                f"{self.ctx.get('seed')}.ollama_log.txt")
        start = self.session_start if end >= self.session_start else 0
        (self.raw_dir / name).write_text(read_log(self.log_path, start, end), encoding="utf-8")
        summ = summarize_session(self.session_calls)
        pc = ptc.tier_summary(self.session_checks).get(self.ctx.get("arm_id"))
        row = {"record": "r2m_session", **self.ctx, **summ, "prompt_check": pc, "raw_log_file": name, "log_bytes": [start, end],
               "ts_utc": _utc()}
        self.emit(row)
        return row


def _utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def report(rows: list[dict]) -> dict:
    """The job's .report.json: per session summary (latest per (model, arm, seed)), the prompt check per tier, and
    the verdict. verdict["per_tier"][tier] carries the tier's own hypothesis result, its prompt-check match rate and
    mechanism_citable; a tier that is not citable gets the explicit not-citable statement, and
    verdict["mechanism_citable"] is true only when every tier is."""
    latest = {}
    for r in rows:
        if r.get("record") == "r2m_session":
            latest[(r["model_id"], r["arm_id"], r["seed"])] = r
    sessions = list(latest.values())
    checks = ptc.tier_summary([r for r in rows if r.get("record") == "r2m_prompt_check"])
    v = verdict(sessions)
    per_tier = {}
    for tier in sorted({s["arm_id"] for s in sessions} | set(checks), key=str):
        pc = checks.get(tier) or {"n_checks": 0, "n_match": 0, "match_rate": None, "mechanism_citable": False,
                                  "statement": ptc.NOT_CITABLE + " (no prompt checks)"}
        hyp = verdict([s for s in sessions if s["arm_id"] == tier])["drop_old_turns_keep_system"]
        per_tier[tier] = {"hypothesis": hyp, "match_rate": pc["match_rate"], "n_checks": pc["n_checks"],
                          "n_match": pc["n_match"], "mechanism_citable": pc["mechanism_citable"],
                          "statement": pc["statement"]}
    v["per_tier"] = per_tier
    v["mechanism_citable"] = bool(per_tier) and all(t["mechanism_citable"] for t in per_tier.values())
    v["not_citable_statements"] = [f"{t}: {p['statement']}" for t, p in per_tier.items() if not p["mechanism_citable"]]
    return {"sessions": sessions, "prompt_check": checks, "verdict": v}


def write_report(out_path: Path, rows: list[dict]) -> dict:
    rep = report(rows)
    Path(str(out_path) + ".report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return rep
