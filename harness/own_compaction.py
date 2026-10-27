"""own-compaction (operator, 2026-10-23): a client-side context manager for the B3 v2 sessions, run through
harness/x2_r2_evalset.py --plan own_compaction_v1 (pre-registered in docs/EXPERIMENTS.md "own-compaction"), and the
per-call cut-off detector v2 used to count every failure silent vs flagged.

Why: client trimming (harness/x2_r2_client_trim.py) keeps every request inside the window by dropping whole old turns,
and on qwen3:14b at 4096 that removed the in-context examples of the two-call pattern (log_event plus the task tool),
so the model stopped calling log_event (register rows R2-qwen3-14b-4096-reconciliation, R2-mitigation-v1-verdict).
This manager keeps three things and compresses the rest with the local model itself:

  (a) pinned rules: the system message (rules, canaries, recorded facts, workspace directory) is always sent; the
      session's standing instructions (first user message, EvalSession.standing) are re-sent at the top of the
      summary message once turn 1 is no longer sent verbatim;
  (b) one tool-call example: the earliest completed turn whose tool calls include log_event and at least one other
      tool (the pattern trimming removed), kept verbatim and never folded;
  (c) a running summary of every other old turn, written by the same local model (think off, no tools, the run's
      num_predict) and placed as a user message plus a one-word assistant acknowledgement after the example turn.

Trigger rule (checked before EVERY model call of the session, task calls only; W = the arm's num_ctx):
  projected next prompt = round((chars/4 estimate of the messages that would be sent + tools) x r), where r is the
  calibration ratio: llama-server's own total prompt tokens (task.n_tokens, from the call's server-log slice) over
  the same estimate for the previous call of this session (falls back to prompt_eval_count, bounded to
  x2_r2_agent.CALIB_RATIO_BOUNDS; PROJ_CALIB 1.20 before the first call).
  If projected > floor(TRIGGER_FRAC x W) (0.85): fold the oldest foldable turns (not the example, not the current
  turn) into the summary until the projection with a full-size summary reserve is <= floor(TARGET_FRAC x W) (0.60),
  or nothing is left to fold. If the floor (system + current turn) alone is already over the trigger, nothing is
  folded (a summary could not be sent) and the call goes out as system + current turn ("floor_over_trigger").
  After folding, if still over the trigger: the example is left out of this call, then the summary.
The summarizer call is itself kept inside the window: the turns to fold are split into chunks whose summarizer
prompt projects to <= W - num_predict - SUMMARY_INPUT_MARGIN.

The harness transcript is never modified (scoring reads the full transcript); only the list sent on a call changes.

Detector v2 (the flag used to count failures, as harness/demo_overflow.py cut_v2 on the demo branch, 2026-10-23):
a call is flagged when the Ollama server's own log slice for it shows a cut: a message drop (truncation line with
first kept message index > 0, x2_r2_mechanism.first_kept_index), a token-level cut ("truncating input prompt"), a
llama.cpp context shift, or an exceeds-context error. The job's own server runs with OLLAMA_DEBUG=1 and its own log
file. Never the token-count comparison (v1), which raised false alarms at the default window.

Pure functions are unit-tested in tests/test_own_compaction.py with fake runtimes; nothing here starts a server.
"""
from __future__ import annotations

import json
import math
import time

import x2_r2_client_trim as ct
import x2_r2_mechanism as mech

TRIGGER_FRAC = 0.85
TARGET_FRAC = 0.60
SUMMARY_INPUT_MARGIN = 256
SUMMARY_MAX_CHARS = 2400
USER_CHARS = 2000
ASSISTANT_CHARS = 1200
TOOL_RESULT_CHARS = 600
INITIAL_CALIB = 1.20      # x2_r2_evalset.PROJ_CALIB
SUMMARY_HEADER = ("[Summary of the earlier part of this session, written to save space. Older turns were compacted; "
                  "everything listed here really happened.]")
STANDING_HEADER = "[Standing instructions from the start of this session, still in force:]"
SUMMARY_ACK = "Noted."
SUMMARIZER_SYSTEM = (
    "You keep a compact running record of an assistant's work session so the assistant can continue without the full "
    "transcript. Record only facts that matter later: every record the assistant created, changed, moved, closed or "
    "deleted (with ids, fields and new values), payments and transfers, files written, emails sent, events booked or "
    "cancelled, confirmations given or still pending, values the user stated, and mistakes or failed tool calls. "
    "Ignore the context filler text (administrative log entries that are not part of any task). Plain text, short "
    "lines, at most 150 words. Do not call tools.")


# ── pure helpers ─────────────────────────────────────────────────────────────────────────────────────

def turn_tool_names(messages: list[dict], idxs: list[int]) -> list[str]:
    """Tool names called in one turn: tool result messages (tool_name, set for native and text-parsed calls alike)
    and native tool_calls on assistant messages."""
    out = []
    for j in idxs:
        m = messages[j]
        if m.get("role") == "tool" and m.get("tool_name"):
            out.append(m["tool_name"])
        elif m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                fn = (c or {}).get("function") or {}
                if fn.get("name"):
                    out.append(fn["name"])
    return out


def is_example_turn(names: list[str]) -> bool:
    return "log_event" in names and any(n != "log_event" for n in names)


def choose_example(messages: list[dict], turns: list[list[int]]) -> int | None:
    """Index (into turns) of the earliest turn showing the log_event + task-tool pattern, or None."""
    for i, t in enumerate(turns):
        if is_example_turn(turn_tool_names(messages, t)):
            return i
    return None


def render_turn_text(messages: list[dict], idxs: list[int], turn_no: int) -> str:
    lines = [f"[turn {turn_no}]"]
    for j in idxs:
        m = messages[j]
        role, content = m.get("role"), (m.get("content") or "")
        if role == "user":
            lines.append("USER: " + content[:USER_CHARS])
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                lines.append("ASSISTANT TOOL CALLS: " + json.dumps(
                    [((c or {}).get("function") or {}) for c in calls], default=str)[:ASSISTANT_CHARS])
            if content.strip():
                lines.append("ASSISTANT: " + content[:ASSISTANT_CHARS])
        elif role == "tool":
            lines.append(f"TOOL RESULT ({m.get('tool_name') or 'unknown'}): " + content[:TOOL_RESULT_CHARS])
    return "\n".join(lines)


def summarizer_messages(prev_summary: str | None, turn_texts: list[str]) -> list[dict]:
    body = ("Record so far:\n" + (prev_summary or "(none yet)") + "\n\nNew turns to add to the record:\n\n"
            + "\n\n".join(turn_texts) + "\n\nWrite the complete updated record now.")
    return [{"role": "system", "content": SUMMARIZER_SYSTEM}, {"role": "user", "content": body}]


def summary_pair(summary: str | None, standing: str | None) -> list[dict]:
    """The user + assistant messages carrying the pinned standing instructions and/or the summary ([] if neither)."""
    parts = []
    if standing:
        parts.append(STANDING_HEADER + "\n" + standing.strip())
    if summary:
        parts.append(SUMMARY_HEADER + "\n" + summary.strip())
    if not parts:
        return []
    return [{"role": "user", "content": "\n\n".join(parts)}, {"role": "assistant", "content": SUMMARY_ACK}]


def assemble(messages, head, turns, cur, example, folded, summary, standing, include_example=True,
             include_summary=True, past=True) -> tuple[list[dict], dict]:
    """The message list to send and what it contains. Order: system, example turn, summary pair, unfolded past turns,
    current turn. Standing instructions go in the summary pair only once turn 0 is not sent verbatim."""
    folded = set(folded)
    sent = [messages[j] for j in head]
    ex_sent = include_example and example is not None and example != cur and past
    rest = [i for i in range(cur) if past and i not in folded and i != example]
    kept = sorted(set(rest) | ({example} if ex_sent else set()))
    turn0_verbatim = 0 in kept or cur == 0
    pair = summary_pair(summary if include_summary else None,
                        standing if (include_summary and not turn0_verbatim) else None)
    # chronological order; the summary pair goes after any kept turn older than the newest folded turn (the example)
    cut = max(folded) if folded else -1
    for i in kept:
        if i < cut:
            sent += [messages[j] for j in turns[i]]
    sent += pair
    for i in kept:
        if i >= cut:
            sent += [messages[j] for j in turns[i]]
    sent += [messages[j] for j in turns[cur]]
    return sent, {"example_sent": ex_sent, "summary_sent": bool(pair) and bool(summary) and include_summary,
                  "standing_pinned": bool(pair) and not turn0_verbatim and bool(standing) and include_summary,
                  "past_turns_sent": rest}


def cut_v2_verdict(log_text: str | None, n_messages: int) -> dict:
    """Detector v2 for one call from its server-log slice: Ollama's own cut lines only."""
    if log_text is None:
        return {"method": "no_log", "flagged": None, "evidence": [], "used_tokens": None}
    ev = mech.parse_call_log(log_text)
    evidence = []
    idx = mech.first_kept_index(ev, n_messages)
    if idx is not None and idx > 0:
        evidence.append(f"older messages dropped (first kept: message {idx} of {n_messages})")
    if ev.get("token_cut"):
        evidence.append("prompt tokens cut ({prompt} -> {new})".format(**ev["token_cut"]))
    if ev.get("context_shifts"):
        evidence.append(f"context shift x{len(ev['context_shifts'])}")
    if ev.get("exceed_context_error"):
        evidence.append("exceeds the context size")
    np_ = ev.get("new_prompt") or []
    return {"method": "server_log", "flagged": bool(evidence), "evidence": evidence,
            "used_tokens": np_[-1]["n_tokens"] if np_ else None,
            "n_ctx_slot": np_[-1]["n_ctx_slot"] if np_ else None,
            "msg_truncation_value": ev.get("msg_truncation_value"), "chat_path": ev.get("chat_path")}


# ── runtimes ─────────────────────────────────────────────────────────────────────────────────────────

class CutLogRuntime:
    """Wraps the Ollama runtime: around every chat call, the server-log slice of that call and its detector-v2
    verdict, attached to the response as resp["cut_v2"]. No render-only request, no prompt check, no unload (unlike
    x2_r2_mechanism.MechanismRuntime), so timing and the KV cache are those of a plain call."""

    def __init__(self, inner, log_path, sleep=time.sleep, clock=time.monotonic, gin_wait_s=mech.GIN_WAIT_S):
        self.inner, self.log_path = inner, log_path
        self.sleep, self.clock, self.gin_wait_s = sleep, clock, gin_wait_s

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _wait_for_gin(self, start):
        deadline = self.clock() + self.gin_wait_s
        while True:
            pos = mech.log_size(self.log_path)
            if pos < start:
                return pos, False
            if mech._PATTERNS["gin"].search(mech.read_log(self.log_path, start, pos)):
                return pos, True
            if self.clock() >= deadline:
                return pos, False
            self.sleep(0.1)

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        p0 = mech.log_size(self.log_path)
        resp = (self.inner.chat(model, messages, num_ctx, tools, think, extra=extra) if extra
                else self.inner.chat(model, messages, num_ctx, tools, think))
        p1, complete = self._wait_for_gin(p0)
        text = mech.read_log(self.log_path, p0, p1) if p1 >= p0 else None
        v = cut_v2_verdict(text, len(messages))
        v["log_slice_complete"] = complete
        v["log_bytes"] = [p0, p1]
        resp = dict(resp)
        resp["cut_v2"] = v
        return resp


class OwnCompactionRuntime:
    """The context manager (module docstring). chat() builds the list to send, runs any summarizer calls through the
    same inner runtime, forwards, and attaches its record to the response as resp["own_compaction"]."""

    def __init__(self, inner, standing=None, num_predict=None, est_fn=None, tools_est_fn=None, clock=time.monotonic,
                 calib_bounds=None):
        import x2_r2_agent as ag
        self.inner, self.standing = inner, standing
        self.num_predict = ag.MAX_TOKENS_PER_CALL if num_predict is None else num_predict
        self.est_fn = est_fn or ag.transcript_tokens
        self.tools_est_fn = tools_est_fn or (lambda tools: ag.est_tokens(json.dumps(tools)) if tools else 0)
        self.calib_bounds = calib_bounds or ag.CALIB_RATIO_BOUNDS
        self.clock = clock
        self.reset()

    def reset(self):
        self.summary, self.folded, self.example, self.ratio = None, [], None, INITIAL_CALIB
        self.n_compactions = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _proj(self, msgs, tools, extra_est=0):
        return round((self.est_fn(msgs) + self.tools_est_fn(tools) + extra_est) * self.ratio)

    def _update_ratio(self, resp, sent, tools):
        used = (resp.get("cut_v2") or {}).get("used_tokens") or resp.get("prompt_eval_count")
        est = self.est_fn(sent) + self.tools_est_fn(tools)
        if used and est:
            r = used / est
            if self.calib_bounds[0] <= r <= self.calib_bounds[1]:
                self.ratio = r

    def _summarize(self, model, num_ctx, think, messages, turns, nums, to_fold) -> list[dict]:
        """Fold turns (indices into turns) into self.summary, chunked so each summarizer prompt fits the window."""
        budget = num_ctx - self.num_predict - SUMMARY_INPUT_MARGIN
        calls, chunk = [], []
        texts = {i: render_turn_text(messages, turns[i], nums[i]) for i in to_fold}

        def fits(ch):
            return self._proj(summarizer_messages(self.summary, [texts[i] for i in ch]), None) <= budget

        def run(ch):
            smsgs = summarizer_messages(self.summary, [texts[i] for i in ch])
            if not fits(ch):   # one turn alone too long: cut its text to the budget
                over = self._proj(smsgs, None) - budget
                keep = max(200, len(texts[ch[0]]) - int(math.ceil(over / self.ratio)) * 4)
                smsgs = summarizer_messages(self.summary, [texts[ch[0]][:keep]])
            t0 = self.clock()
            resp = self.inner.chat(model, smsgs, num_ctx, None, think)
            if resp.get("outcome") == "error" and resp.get("status") is None:   # server gone: session incomplete
                import x2_r2_agent as ag
                raise ag.SessionIncomplete(resp.get("error"), {"kind": "summary", "error": resp.get("error"),
                                                               "turns_folded": [nums[i] for i in ch]})
            text = (resp.get("message") or "").strip()
            ok = resp.get("outcome", "ok") == "ok" and bool(text)
            if ok:
                self.summary = text[:SUMMARY_MAX_CHARS]
            else:
                lost = ", ".join(str(nums[i]) for i in ch)
                self.summary = ((self.summary or "") + f"\n(turns {lost}: summary failed, details lost)").strip()
            self.folded = sorted(set(self.folded) | set(ch))
            calls.append({"kind": "summary", "turns_folded": [nums[i] for i in ch], "ok": ok,
                          "prompt_proj": self._proj(smsgs, None), "prompt_eval_count": resp.get("prompt_eval_count"),
                          "eval_count": resp.get("eval_count"), "duration_s": resp.get("duration_s"),
                          "wall_s": self.clock() - t0, "summary_chars": len(self.summary or ""),
                          "error": resp.get("error"), "cut_v2": resp.get("cut_v2")})

        for i in to_fold:
            if chunk and not fits(chunk + [i]):
                run(chunk)
                chunk = []
            chunk.append(i)
        if chunk:
            run(chunk)
        return calls

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        t0 = self.clock()
        if num_ctx is None:
            resp = (self.inner.chat(model, messages, num_ctx, tools, think, extra=extra) if extra
                    else self.inner.chat(model, messages, num_ctx, tools, think))
            resp = dict(resp)
            resp["own_compaction"] = {"applied": False, "reason": "num_ctx not set"}
            return resp
        head, turns = ct.split_turns(messages)
        nums = ct.session_turn_numbers(messages, turns)
        cur = len(turns) - 1
        if self.example is None and cur > 0:
            self.example = choose_example(messages, turns[:cur])
        trig, target = int(math.floor(TRIGGER_FRAC * num_ctx)), int(math.floor(TARGET_FRAC * num_ctx))
        ratio_used = self.ratio
        a = dict(messages=messages, head=head, turns=turns, cur=cur, example=self.example)
        sent, info = assemble(**a, folded=self.folded, summary=self.summary, standing=self.standing)
        proj_before = self._proj(sent, tools)
        rec = {"applied": True, "num_ctx": num_ctx, "trigger_tokens": trig, "target_tokens": target,
               "ratio_used": ratio_used, "proj_full": self._proj(messages, tools), "proj_before": proj_before,
               "example_turn": nums[self.example] if self.example is not None else None, "triggered": False,
               "floor_over_trigger": False, "summary_calls": [], "dropped_for_fit": []}
        if proj_before > trig and cur >= 0:
            rec["triggered"] = True
            floor, _ = assemble(**a, folded=self.folded, summary=None, standing=None, include_example=False,
                                include_summary=False, past=False)
            rec["proj_floor"] = self._proj(floor, tools)
            if rec["proj_floor"] > trig:
                rec["floor_over_trigger"] = True
                sent, info = floor, {"example_sent": False, "summary_sent": False, "standing_pinned": False,
                                     "past_turns_sent": []}
            else:
                foldable = [i for i in range(cur) if i not in set(self.folded) and i != self.example]
                reserve = self.num_predict + 64
                k = len(foldable)
                for n in range(0, len(foldable) + 1):
                    cand, _ = assemble(**a, folded=list(self.folded) + foldable[:n], summary=None,
                                       standing=self.standing)
                    if self._proj(cand, tools, extra_est=reserve) <= target:
                        k = n
                        break
                if k:
                    rec["summary_calls"] = self._summarize(model, num_ctx, think, messages, turns, nums, foldable[:k])
                    self.n_compactions += 1
                sent, info = assemble(**a, folded=self.folded, summary=self.summary, standing=self.standing)
                if self._proj(sent, tools) > trig and info["example_sent"]:
                    sent, info = assemble(**a, folded=self.folded, summary=self.summary, standing=self.standing,
                                          include_example=False)
                    rec["dropped_for_fit"].append("example")
                if self._proj(sent, tools) > trig and info["summary_sent"]:
                    sent, info = assemble(**a, folded=self.folded, summary=None, standing=None,
                                          include_example=False, include_summary=False)
                    rec["dropped_for_fit"].append("summary")
        rec.update({"proj_sent": self._proj(sent, tools), "over_trigger_sent": self._proj(sent, tools) > trig,
                    "n_messages_full": len(messages), "n_messages_sent": len(sent),
                    "folded_turns": [nums[i] for i in self.folded], "n_compactions": self.n_compactions,
                    "past_turns_sent": [nums[i] for i in info["past_turns_sent"]],
                    **{k: info[k] for k in ("example_sent", "summary_sent", "standing_pinned")},
                    "summary_chars": len(self.summary or "")})
        rec["manage_wall_s"] = self.clock() - t0
        resp = (self.inner.chat(model, sent, num_ctx, tools, think, extra=extra) if extra
                else self.inner.chat(model, sent, num_ctx, tools, think))
        self._update_ratio(resp, sent, tools)
        resp = dict(resp)
        resp["own_compaction"] = rec
        return resp


class PolicySessionRuntime:
    """Outermost wrapper for --plan own_compaction_v1 / own_config_probe_v1: picks the per-session context policy by
    arm (own compaction, client trim, or none), keeps the detector-v2 log runtime underneath every policy, and buffers
    one record per task call for x2_r2_evalset._run_session_body (drain_turn)."""

    def __init__(self, base, log_path, emit, policy_of, counter_factory=None, sleep=time.sleep):
        self.base, self.emit, self.policy_of = base, emit, policy_of
        self.cut = CutLogRuntime(base, log_path, sleep=sleep)
        self.counter_factory = counter_factory
        self._counter = None
        self.active, self.policy, self.buf = self.cut, None, []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def begin_session(self, model, arm, seed, es):
        self.policy = self.policy_of(arm)
        self.buf = []
        if self.policy == "own_compaction":
            self.active = OwnCompactionRuntime(self.cut, standing=es.standing)
        elif self.policy == "client_trim":
            if self._counter is None:
                self._counter = (self.counter_factory or (lambda b: ct.RenderTokenCounter(b)))(self.base)
            self.active = ct.ClientTrimRuntime(self.cut, self._counter, margin=ct.DEFAULT_MARGIN)
            self.active.begin_session()
        else:
            self.active = self.cut
        self.emit({"record": "ctx_policy_session", "model_id": model, "arm_id": arm, "seed": seed,
                   "policy": self.policy, "trigger_frac": TRIGGER_FRAC, "target_frac": TARGET_FRAC,
                   "trim_margin": ct.DEFAULT_MARGIN if self.policy == "client_trim" else None,
                   "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return None

    def end_session(self):
        self.active, self.policy = self.cut, None

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        resp = (self.active.chat(model, messages, num_ctx, tools, think, extra=extra) if extra
                else self.active.chat(model, messages, num_ctx, tools, think))
        oc, tr = resp.get("own_compaction"), resp.get("client_trim")
        self.buf.append({"kind": "task", "policy": self.policy, "cut_v2": resp.get("cut_v2"),
                         "own_compaction": oc,
                         "client_trim_sent": ({k: tr.get(k) for k in ("kept_prompt_tokens_exact", "over_budget",
                                                                     "n_turns_dropped", "budget_prompt_tokens")}
                                              if tr else None)})
        return resp

    def drain_turn(self) -> list[dict]:
        out, self.buf = self.buf, []
        return out


def turn_fields(recs: list[dict]) -> dict:
    """Per-turn fields from the drained call records: detector-v2 flag on any task call (and on any summarizer call,
    kept apart), the policy, the projected prompt actually sent on the final call."""
    task = [r for r in recs if r.get("kind") == "task"]
    summ = [s for r in task for s in ((r.get("own_compaction") or {}).get("summary_calls") or [])]

    def flagged(c):
        return bool((c.get("cut_v2") or {}).get("flagged"))
    last = task[-1] if task else {}
    oc = last.get("own_compaction") or {}
    trs = last.get("client_trim_sent") or {}
    return {"ctx_policy": last.get("policy"), "cut_v2_flagged": any(flagged(r) for r in task),
            "cut_v2_evidence": sorted({e for r in task for e in ((r.get("cut_v2") or {}).get("evidence") or [])}),
            "cut_v2_no_log_calls": sum(1 for r in task if (r.get("cut_v2") or {}).get("flagged") is None),
            "cut_v2_flagged_summary_calls": sum(1 for s in summ if flagged(s)),
            "n_summary_calls": len(summ),
            "sent_final_prompt_proj": oc.get("proj_sent") if oc.get("applied") else trs.get("kept_prompt_tokens_exact"),
            "ctx_policy_calls": recs}


# ── outcome (register rows own-compaction-4096 / -8192) ──────────────────────────────────────────────

ARM_POLICY = {"own_compaction_4096": ("own_compaction", 4096), "own_compaction_8192": ("own_compaction", 8192),
              "client_trim_4096": ("client_trim", 4096), "client_trim_8192": ("client_trim", 8192)}
TARGETS = {4096: 0.20, 8192: 0.15}
BASELINE_WRONG = {4096: (63, 150), 8192: (53, 150)}   # x2-evalset-tasks-real-v2-wrong-rate (plain num_ctx tiers)


def outcome(rows: list[dict], read_rows_fn=None) -> dict:
    """Per arm: wrong task turns over completed sessions, split flagged (detector v2 on any task call of the turn) vs
    silent, plus summary-call counts and compaction counts. Rows of completed sessions only (r2a_session present)."""
    done = {(r["arm_id"], r["seed"], r.get("attempt_id")) for r in rows if r.get("record") == "r2a_session"}
    out = {}
    for r in rows:
        if r.get("record") != "r2a_turn" or r.get("item_kind") != "task":
            continue
        if (r["arm_id"], r["seed"], r.get("attempt_id")) not in done:
            continue
        base = r["arm_id"].replace("_call2_notools", "")
        a = out.setdefault(base, {"n": 0, "wrong": 0, "flagged_wrong": 0, "silent_wrong": 0, "flagged_turns": 0,
                                  "no_log_turns": 0, "seeds": set(), "summary_calls": 0, "summary_flagged": 0,
                                  "loaded": set(), "example_sent_calls": 0, "floor_over_trigger_calls": 0,
                                  "task_calls": 0})
        a["n"] += 1
        a["seeds"].add(r["seed"])
        a["loaded"].add(r.get("loaded_context"))
        f = bool(r.get("cut_v2_flagged"))
        a["flagged_turns"] += f
        a["no_log_turns"] += bool(r.get("cut_v2_no_log_calls"))
        a["summary_calls"] += r.get("n_summary_calls") or 0
        a["summary_flagged"] += r.get("cut_v2_flagged_summary_calls") or 0
        for c in r.get("ctx_policy_calls") or []:
            oc = c.get("own_compaction") or {}
            a["task_calls"] += 1
            a["example_sent_calls"] += bool(oc.get("example_sent"))
            a["floor_over_trigger_calls"] += bool(oc.get("floor_over_trigger"))
        if r.get("end_state_ok") is not True:
            a["wrong"] += 1
            a["flagged_wrong" if f else "silent_wrong"] += 1
    for a in out.values():
        a["seeds"] = sorted(a["seeds"])
        a["loaded"] = sorted(x for x in a["loaded"] if x is not None)
        a["wrong_rate"] = a["wrong"] / a["n"] if a["n"] else None
    return out


def verdict(out: dict, numkeep_wrong=(45, 150)) -> dict:
    """Pre-registered criteria: own_compaction at 4096 wrong < 20% and at 8192 < 15%; plus the plain comparisons
    with num_keep (4096) and client trim at the same window (lower wrong rate = beats; equal = does not beat)."""
    res = {}
    for w in (4096, 8192):
        c = out.get(f"own_compaction_{w}")
        t = out.get(f"client_trim_{w}")
        if not c or not c["n"]:
            res[w] = {"status": "missing"}
            continue
        r = {"wrong_rate": c["wrong_rate"], "target": TARGETS[w], "meets_target": c["wrong_rate"] < TARGETS[w],
             "baseline_plain": BASELINE_WRONG[w], "beats_plain": c["wrong_rate"] < BASELINE_WRONG[w][0] / BASELINE_WRONG[w][1]}
        if t and t["n"]:
            r["trim_wrong_rate"] = t["wrong_rate"]
            r["beats_trim"] = c["wrong_rate"] < t["wrong_rate"]
        if w == 4096:
            r["numkeep_wrong"] = numkeep_wrong
            r["beats_numkeep"] = c["wrong_rate"] < numkeep_wrong[0] / numkeep_wrong[1]
        res[w] = r
    return res


# ── hours ESTIMATE (register row own-compaction-hours) ───────────────────────────────────────────────

SUMMARY_S = {"point": 16.0, "high": 27.0}           # one summarizer call; basis in the docstring below
COMPACTIONS_PER_SESSION = {"point": 5, "high": 15}   # dry run on the fake runtime: 3 per 8192 session with a one-word summary


def per_task_seconds(rows: list[dict]) -> dict:
    """{base arm: {"mean": s, "max_seed": s, "calls_per_task": c}} from B3 v2 task turns (turn_wall_s)."""
    acc = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and r.get("item_kind") == "task":
            a = acc.setdefault(r["arm_id"].replace("_call2_notools", ""), {})
            s = a.setdefault(r["seed"], [0.0, 0, 0])
            s[0] += r["turn_wall_s"]
            s[1] += 1
            s[2] += r.get("n_calls") or 0
    out = {}
    for a, seeds in acc.items():
        per = [v[0] / v[1] for v in seeds.values()]
        out[a] = {"mean": sum(per) / len(per), "max_seed": max(per),
                  "calls_per_task": sum(v[2] for v in seeds.values()) / sum(v[1] for v in seeds.values())}
    return out


def trim_overhead_s(mit_rows: list[dict], model="qwen3:14b") -> dict:
    """{num_ctx: (mean, p90) trim_wall_s per call} from the R2 mitigation run (render + tokenize counts)."""
    by = {}
    for r in mit_rows:
        if r.get("record") == "r2a_turn" and r.get("model_id") == model:
            for c in r.get("calls") or []:
                t = c.get("client_trim")
                if t and t.get("trim_wall_s") is not None:
                    by.setdefault(t.get("num_ctx"), []).append(t["trim_wall_s"])
    out = {}
    for k, v in by.items():
        v = sorted(v)
        out[k] = (sum(v) / len(v), v[min(len(v) - 1, int(0.9 * len(v)))])
    return out


def hours_estimate(b3v2_rows, mit_rows, seeds=5, tasks=30) -> dict:
    """ESTIMATE per arm (hours) for seeds x tasks task turns. Bases, all from committed files:
      client_trim_W: point = plain num_ctx W per-task mean + trim overhead mean x calls per task; high = the slowest
        seed's per-task time + p90 overhead x calls (trimming changes the prefix each call, as the plain tier's
        truncation does, so the plain tier is the base).
      own_compaction_8192: point = default-window per-task mean (prefix stable between compactions) + COMPACTIONS x
        SUMMARY_S per session; high = plain 8192 slowest seed + the high compaction count x high summary time.
      own_compaction_4096: point = num_keep tier per-task mean (system prefix cached, history not sent) and no summary
        (floor over trigger); high = plain 4096 slowest seed + high compactions x high summary time.
      SUMMARY_S: prompt eval of a ~3500-token summarizer prompt at the 8192 tier's median rate plus 200 (point) or 384
      (high) generated tokens at the measured 17.9 tok/s (B3 v2 call records)."""
    pt = per_task_seconds(b3v2_rows)
    ov = trim_overhead_s(mit_rows)
    n = seeds * tasks
    out = {}
    for w in (4096, 8192):
        plain = pt[f"ollama_ctx_{w}"]
        o_mean, o_p90 = ov.get(w, (0.0, 0.0))
        out[f"client_trim_{w}"] = {
            "point_h": n * (plain["mean"] + o_mean * plain["calls_per_task"]) / 3600,
            "high_h": n * (plain["max_seed"] + o_p90 * plain["calls_per_task"]) / 3600}
    dflt, nk = pt["ollama_default"], pt["ollama_ctx_4096_numkeep"]
    sess_pt = COMPACTIONS_PER_SESSION["point"] * SUMMARY_S["point"]
    sess_hi = COMPACTIONS_PER_SESSION["high"] * SUMMARY_S["high"]
    out["own_compaction_8192"] = {"point_h": (n * dflt["mean"] + seeds * sess_pt) / 3600,
                                  "high_h": (n * pt["ollama_ctx_8192"]["max_seed"] + seeds * sess_hi) / 3600}
    out["own_compaction_4096"] = {"point_h": n * nk["mean"] / 3600,
                                  "high_h": (n * pt["ollama_ctx_4096"]["max_seed"] + seeds * sess_hi) / 3600}
    # config probes (--plan own_config_probe_v1, 3 seeds): point = the 16384 tier's per-task mean (in-window prefix
    # reuse); high = the plain 8192 tier's slowest seed (truncation-heavy calls); KV q8_0 / q4_0 speed on ROCm for
    # B3 v2 is not measured, so neither case assumes the B5 speedup
    n3 = 3 * tasks
    for k in ("probe_q8_15360", "probe_q4_28928"):
        out[k] = {"point_h": n3 * pt["ollama_ctx_16384"]["mean"] / 3600,
                  "high_h": n3 * pt["ollama_ctx_8192"]["max_seed"] / 3600}
    out["part_a"] = {x: out["own_compaction_8192"][x] + out["own_compaction_4096"][x] for x in ("point_h", "high_h")}
    out["part_b"] = {x: out["client_trim_4096"][x] + out["client_trim_8192"][x] for x in ("point_h", "high_h")}
    out["_per_task"] = pt
    out["_trim_overhead"] = ov
    return out
