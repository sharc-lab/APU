"""R2 two-step agent-session harness (evo-x2, Ollama). docs/R2_DESIGN.md is the spec; read it first.

Each agent turn is two model calls:
  call 1: the turn's user task with the 2-tool schema available -> the model emits tool call(s);
  the harness simulates each tool result (log_event -> ok, lookup_fact -> the seeded value for the key, or a
  deterministic "fact not found");
  call 2: the same conversation plus the call-1 assistant message and the simulated tool results -> the
  model's final answer for the turn.

Scoring (pure functions below, unit-tested with a fake runtime in tests/test_x2_r2_agent.py):
  - tool validity (known tool, exact parameter keys, string values) and tool arguments (values right for this
    turn's task) are scored per call, separately;
  - text rules 1/3/4/5, the canaries and fact recall are scored on the FINAL answer only;
  - rule 2 (log_event) is a turn-level OR: satisfied if log_event was called in call 1 or call 2;
  - token accounting counts both calls: the transcript keeps the full tool round trip (call-1 tool-call message,
    tool results, call-2 answer), and per-call prompt+completion tokens are also summed for cost;
  - truncation = a canary miss on a canary turn WHILE the transcript's token count exceeds the context Ollama
    actually has loaded (GET /api/ps context_length, read every turn). prompt_eval_count is NEVER the truncation
    signal; it is recorded per call for information only, and used once per session (turn 1 call 1, right after
    a fresh model load, so nothing is cached and nothing is truncated) as a tokenizer calibration ratio for the
    chars/4 estimate.

Canaries: two kinds, and a fresh pair per check (extensions of R2_DESIGN.md, recorded there).
  - Ollama's /api/chat history truncation keeps every system message and drops the oldest non-system messages
    first, so a canary living only in the system prompt survives a real truncation. Numbered canaries C1..C8 sit in
    the system prompt (replacing r2's single-canary paragraph) AND numbered history tags H1..H8 sit in the first
    USER message, which Ollama drops first. Both are scored separately; a miss of either while over the loaded
    window is truncation, and which kind survives is recorded.
  - Turn 5k asks only for Ck and Hk. The first validation run (2026-10-07) reused one canary at every check, and
    qwen3:14b's positive control copied it at turn 10 from its own turn-5 answer, which was still inside the
    retained tail: with a canary asked exactly once, no earlier answer can carry it forward.

Reused unchanged from harness/t2s_r2_session_growth.py: generate_session (session code, facts, system-prompt canary,
rule text, turn tasks and filler), TOOLS_SCHEMA / ollama_tools_payload, CANARY_CHECK_EVERY, the forbidden-length
unit regex, _approx_token_count, evaluate_kill_criterion.

Modes (--call2-tools on = the spec, tools available on call 2; off = withheld on call 2 only, the "_call2_notools"
arms; chosen for the real run after llama3.1:8b left 20 of 30 final answers empty with tools on call 2, see
R2_DESIGN.md and FINDINGS):
  --mode validation : arm b (num_ctx 131072) 3 seeds x 10 turns, both models (negative control + baseline table),
                      positive control (num_ctx 8192, 1 seed x 15 turns), and the other call-2 variant's arm b as
                      a diagnostic.
  --mode real       : Ollama default vs num_ctx 32768 vs num_ctx 4096, 3 seeds x 40 turns, both models;
                      --rules-from <validation jsonl> sets each model's rules in use for the kill criterion.

Queue job: calls t2s_queue.advance() exactly once on exit (finally). Never run it as a bare process on evo-x2 while
another queued job is running (see the WARNING at the top of t2s_r2_session_growth.py; advance() also refuses to act
for a caller that the queue did not launch).

Usage: py -3.12 x2_r2_agent.py --mode validation --out results/x2_r2_validation.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import re
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if (_here.parent / "harness").is_dir():
    sys.path.insert(0, str(_here.parent / "harness"))

import t2s_r2_session_growth as r2  # noqa: E402

# ── configuration ───────────────────────────────────────────────────────────────────────────────────

# Ollama tag -> the "think" value sent on every call (None = field omitted). qwen3 is a hybrid-reasoning model
# whose thinking default leaves message.content empty (see OllamaClient.chat's think= docstring); llama3.1 has no
# thinking mode, so the field is omitted for it rather than sent.
MODELS = {"llama3.1:8b": None, "qwen3:14b": False}

_BASE_ARMS = {
    "ollama_ctx_131072": 131072,                 # arm b, the no-truncation control (validation, negative control)
    "ollama_ctx_8192_positive_control": 8192,    # validation positive control
    "ollama_default": None,                      # real run: no num_ctx sent, the runtime picks
    "ollama_ctx_32768": 32768,
    "ollama_ctx_4096": 4096,
}
# arm id -> {"num_ctx", "call2_tools"}. The spec (R2_DESIGN.md step 4) keeps tools available on call 2; every base
# arm also has a "_call2_notools" variant that withholds them on call 2 only. Validation runs that variant on arm b
# as a DIAGNOSTIC alongside the spec arm (gates are computed on the spec arm only), so that if a model loops on tool
# calls in call 2 instead of answering, the alternative's baseline is already measured in the same job.
NOTOOLS_SUFFIX = "_call2_notools"
ARMS = {}
for _a, _n in _BASE_ARMS.items():
    ARMS[_a] = {"num_ctx": _n, "call2_tools": True}
    ARMS[_a + NOTOOLS_SUFFIX] = {"num_ctx": _n, "call2_tools": False}

SEEDS = r2.SEEDS
VALIDATION_PLAN = [  # (arm, seeds, turns) -- the first validation run (2026-10-07, file x2_r2_validation.jsonl)
    ("ollama_ctx_131072", SEEDS, 10),
    ("ollama_ctx_8192_positive_control", SEEDS[:1], 10),
    ("ollama_ctx_131072" + NOTOOLS_SUFFIX, SEEDS, 10),
]
NEG_ARM, POS_ARM = "ollama_ctx_131072", "ollama_ctx_8192_positive_control"


def validation_plan(call2_tools: bool, with_diagnostic: bool = True):
    """Validation for the chosen call-2 variant: negative control/baseline (3 seeds x 10 turns) and positive
    control (1 seed x 15 turns, so two canary checks fall past the 8192 window). The opposite variant's arm b is
    added as a diagnostic if with_diagnostic."""
    sfx = "" if call2_tools else NOTOOLS_SUFFIX
    plan = [(NEG_ARM + sfx, SEEDS, 10), (POS_ARM + sfx, SEEDS[:1], 15)]
    if with_diagnostic:
        plan.append((NEG_ARM + (NOTOOLS_SUFFIX if call2_tools else ""), SEEDS, 10))
    return plan
REAL_PLAN = [
    ("ollama_default", SEEDS, 40),
    ("ollama_ctx_32768", SEEDS, 40),
    ("ollama_ctx_4096", SEEDS, 40),
]


def real_plan(call2_tools: bool):
    return [(a if call2_tools else a + NOTOOLS_SUFFIX, s, t) for a, s, t in REAL_PLAN]

MAX_TOKENS_PER_CALL = 384
KEEP_ALIVE = "30m"          # hc.start_ollama_server sets OLLAMA_KEEP_ALIVE=0; keep the model (and its KV cache)
                            # loaded across a session's calls, then unload explicitly at the session boundary
CALL_TIMEOUT_S = 900
BASELINE_THRESHOLD = 0.90
CALIB_RATIO_BOUNDS = (0.5, 2.5)
PER_MESSAGE_OVERHEAD_TOKENS = 4

RULE_IDS = ("rule1_json_keys", "rule2_log_event_called", "rule3_no_zebra", "rule4_metres", "rule5_session_code")
ANSWER_KEYS_REQUIRED = {"answer", "source"}
ANSWER_KEYS_ALLOWED = {"answer", "source", "tool_calls"}
TOOL_PARAMS = {t["name"]: t["parameters"] for t in r2.TOOLS_SCHEMA}

NOT_FOUND = "fact not found"


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def est_tokens(text) -> int:
    """chars/4 estimate (r2._approx_token_count); 0 for empty/None."""
    if not text:
        return 0
    return r2._approx_token_count(text if isinstance(text, str) else json.dumps(text))


# ── session construction (pure) ─────────────────────────────────────────────────────────────────────

N_CANARIES = 8  # one fresh canary pair per check; 8 checks covers 40 turns (CANARY_CHECK_EVERY = 5)
_R2_CANARY_PARAGRAPH = "This session also has a unique canary phrase"
_R2_CANARY_ASK = (" Also, repeat this session's canary phrase exactly, verbatim, in your answer "
                  "-- it was given to you at the start of this session, do not look it up.")


def generate_sys_canaries(seed: int, n: int = N_CANARIES) -> list[str]:
    """Numbered system-prompt canaries C1..Cn. Distinct constants from generate_canary/session code/facts."""
    out = []
    for k in range(1, n + 1):
        rng = random.Random(seed * 999331 + 17 + 7919 * k)
        out.append("CANARY-" + "".join(rng.choice(r2._CODE_ALPHABET) for _ in range(10)))
    return out


def generate_history_tags(seed: int, n: int = N_CANARIES) -> list[str]:
    """Numbered history tags H1..Hn, placed in the first USER message (what Ollama's truncation drops first)."""
    out = []
    for k in range(1, n + 1):
        rng = random.Random(seed * 7919 + 29 + 104729 * k)
        out.append("HTAG-" + "".join(rng.choice(r2._CODE_ALPHABET) for _ in range(10)))
    return out


def canary_index(turn) -> int | None:
    """1-based check number k for a canary turn (turn 5k asks for Ck and Hk), else None."""
    if not turn.canary_check:
        return None
    k = turn.idx // r2.CANARY_CHECK_EVERY
    if k > N_CANARIES:
        raise ValueError(f"turn {turn.idx} needs canary #{k} but only {N_CANARIES} exist")
    return k


def build_system_prompt(spec, sys_canaries: list[str]) -> str:
    """r2's system prompt (rules, tools, facts) with its single-canary paragraph replaced by the numbered list.
    A single canary asked at every check is echoed by the model's own earlier canary answers, which then keep it
    alive inside the retained tail after a truncation (observed in the first validation run, positive control,
    qwen3:14b: both canaries reproduced at turn 10 with the window long exceeded). Each Ck is asked exactly once."""
    base = spec.system_prompt
    cut = base.index(_R2_CANARY_PARAGRAPH)
    lines = "\n".join(f"  C{k}: {c}" for k, c in enumerate(sys_canaries, 1))
    return (base[:cut] + f"This session also has {len(sys_canaries)} numbered canary phrases, separate from the "
            f"facts above and from the session code:\n{lines}\nStarting at turn 5 and every 5th turn after that, "
            f"you will be asked to repeat one of these canary phrases, by its number, verbatim in your \"answer\" "
            f"field, without looking it up -- this checks whether you can still see this message later in a long "
            f"session.\n")


def build_fact_store(session) -> dict[str, str]:
    """The simulated lookup_fact backend: one deterministic value per lookup key a turn asks for. The three
    system-prompt recall facts are deliberately NOT in the store (looking one up returns "fact not found"), so a
    recall answer can only come from the model's own context, which is what recall measures."""
    store = {}
    for t in session.turns:
        if t.tool_name == "lookup_fact":
            rng = random.Random(session.seed * 31337 + t.idx)
            store[t.tool_args["key"]] = "RV-" + "".join(rng.choice(r2._CODE_ALPHABET) for _ in range(6))
    return store


@dataclasses.dataclass(frozen=True)
class AgentSession:
    spec: object            # r2.SessionSpec (turns, session code, facts)
    system_prompt: str
    sys_canaries: list
    hist_tags: list
    store: dict


def build_agent_session(seed: int, max_turns: int) -> AgentSession:
    spec = r2.generate_session(seed, count_fn=r2._approx_token_count, max_turns=max_turns)
    sys_c = generate_sys_canaries(seed)
    return AgentSession(spec=spec, system_prompt=build_system_prompt(spec, sys_c), sys_canaries=sys_c,
                        hist_tags=generate_history_tags(seed), store=build_fact_store(spec))


def expected_canaries(ags: AgentSession, turn) -> tuple[str | None, str | None]:
    k = canary_index(turn)
    return (None, None) if k is None else (ags.sys_canaries[k - 1], ags.hist_tags[k - 1])


def user_message(turn, ags: AgentSession) -> str:
    """The text sent for a turn: r2.turn_message_content (task + filler), with r2's single-canary request replaced
    by a numbered one, and the history-tag list prefixed to turn 1."""
    text = turn.user_text
    k = canary_index(turn)
    if k is not None:
        text = text.replace(_R2_CANARY_ASK, "")
        text += (f" Also, repeat canary phrase C{k} (given at the start of this session) and history tag H{k} "
                 f"(given in the first user message of this session) exactly, verbatim, in your answer; do not "
                 f"look them up.")
    msg = r2.turn_message_content(dataclasses.replace(turn, user_text=text))
    if turn.idx == 1:
        tags = "\n".join(f"  H{i}: {h}" for i, h in enumerate(ags.hist_tags, 1))
        msg = ("Session history tags (remember them; later in this session you will be asked to repeat one of "
               f"them, by its number, verbatim):\n{tags}\n\n" + msg)
    return msg


# ── tool-call extraction and scoring (pure) ─────────────────────────────────────────────────────────

def _strip_fence(text: str) -> str:
    """Strip one surrounding markdown code fence (```json ... ```) and outer whitespace, nothing else."""
    s = (text or "").strip()
    m = re.fullmatch(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```", s, flags=re.DOTALL)
    return m.group(1).strip() if m else s


def parse_json_strict(text: str):
    """json.loads of the whole content after stripping at most one code fence. None on any failure."""
    try:
        return json.loads(_strip_fence(text))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _norm_args(args):
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return args
    return args


def _as_call(obj):
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function")
    if isinstance(fn, dict) and "name" in fn:
        return {"name": fn.get("name"), "arguments": _norm_args(fn.get("arguments"))}
    if "name" in obj and ("arguments" in obj or "parameters" in obj):
        return {"name": obj.get("name"),
                "arguments": _norm_args(obj.get("arguments", obj.get("parameters")))}
    return None


def extract_tool_calls(content: str | None, native) -> tuple[list[dict], str]:
    """(calls, method). Native message.tool_calls first; only if that is empty, a text-embedded call: the whole
    content (fence-stripped) is a bare call object {"name", "arguments"|"parameters"}, a list of them, or a
    {"tool_calls": [...]} object WITHOUT an "answer" key. An answer-shaped JSON (has "answer") is a final answer,
    and its optional "tool_calls" key is a report, never counted as a call."""
    if isinstance(native, list) and native:
        calls = [c for c in (_as_call(x) for x in native) if c is not None]
        # a native entry we cannot parse is still a call attempt; keep it as an invalid call
        calls += [{"name": None, "arguments": None} for x in native if _as_call(x) is None]
        return calls, "native"
    obj = parse_json_strict(content or "")
    cands = None
    if isinstance(obj, list):
        cands = obj
    elif isinstance(obj, dict):
        if "answer" in obj:
            cands = None
        elif isinstance(obj.get("tool_calls"), list):
            cands = obj["tool_calls"]
        elif _as_call(obj) is not None:
            cands = [obj]
    if cands:
        calls = [c for c in (_as_call(x) for x in cands) if c is not None]
        if calls and len(calls) == len(cands):
            return calls, "text_fallback"
    return [], "none"


def tool_call_valid(call: dict) -> bool:
    """Known tool, arguments a dict with exactly the schema's keys, every value a string."""
    params = TOOL_PARAMS.get(call.get("name"))
    if params is None:
        return False
    args = call.get("arguments")
    if not isinstance(args, dict):
        return False
    if set(args) != set(params["properties"]):
        return False
    return all(isinstance(args[k], str) for k in args)


_LEN_EVENT = re.compile(r"rack (R\d{3}) cable length (\d+) cm")


def tool_args_correct(call: dict, turn) -> bool:
    """For a VALID call: are the argument values right for this turn's task?
    lookup_fact: only correct on a lookup turn, with exactly that turn's key (whitespace-stripped).
    log_event on a length turn: the event mentions this turn's rack id and its length (the cm figure, or the same
    length in metres). log_event on any other turn (rule 2's mandatory call): any non-empty event."""
    name, args = call["name"], call["arguments"]
    if name == "lookup_fact":
        return turn.tool_name == "lookup_fact" and args["key"].strip() == turn.tool_args["key"]
    event = args["event"].strip()
    if not event:
        return False
    if turn.tool_name == "log_event":
        m = _LEN_EVENT.fullmatch(turn.tool_args["event"])
        if m:
            rack, cm = m.group(1), int(m.group(2))
            metres = {f"{cm / 100:g}", f"{cm / 100:.2f}", f"{cm / 100:.1f}"}
            return rack in event and (str(cm) in event or any(v in event for v in metres))
    return True


def simulate_tool_result(call: dict, store: dict) -> str:
    if not tool_call_valid(call):
        return json.dumps({"status": "error", "error": f"invalid tool call: {call.get('name')!r}"})
    if call["name"] == "log_event":
        return json.dumps({"status": "ok", "tool": "log_event", "logged": True})
    key = call["arguments"]["key"].strip()
    if key in store:
        return json.dumps({"status": "ok", "tool": "lookup_fact", "key": key, "value": store[key]})
    return json.dumps({"status": "ok", "tool": "lookup_fact", "key": key, "value": None, "message": NOT_FOUND})


def score_final_answer(turn, text: str | None, session_code: str, sys_canary: str, hist_canary: str) -> dict:
    """Rules 1/3/4/5, the two canaries and fact recall, on the FINAL answer text only.
    rule 1: whole content (one code fence allowed) is a JSON object with keys answer (string) and source, nothing
            beyond answer/source/tool_calls.
    rule 3: "ZEBRA-7" nowhere in the raw text.
    rule 4: length turns only (None elsewhere, excluded from its denominator): no forbidden unit (feet, inches,
            yards, km, miles) in the answer field (raw text if the JSON did not parse; scored independently of
            rule 1, so a JSON failure does not cascade into a metres failure).
    rule 5: the answer field's string ends with the session code (needs a parsed answer field by definition).
    canaries / recall: verbatim substring of the raw text (independent of rule 1); None on turns that do not ask."""
    raw = text or ""
    obj = parse_json_strict(raw)
    answer = obj.get("answer") if isinstance(obj, dict) else None
    rule1 = (isinstance(obj, dict) and isinstance(answer, str) and ANSWER_KEYS_REQUIRED <= set(obj)
             and set(obj) <= ANSWER_KEYS_ALLOWED)
    rule3 = r2._ZEBRA not in raw
    rule4 = None
    if turn.requires_length:
        scope = answer if isinstance(answer, str) else raw
        rule4 = not r2._FORBIDDEN_LENGTH_UNITS.search(scope)
    rule5 = isinstance(answer, str) and answer.rstrip().endswith(session_code)
    out = {"rule1_json_keys": bool(rule1), "rule3_no_zebra": rule3, "rule4_metres": rule4,
           "rule5_session_code": bool(rule5), "final_content_empty": not raw.strip(),
           "canary_sys_ok": None, "canary_hist_ok": None, "recall_ok": None}
    if turn.canary_check:
        out["canary_sys_ok"] = sys_canary in raw
        out["canary_hist_ok"] = hist_canary in raw
    if turn.is_recall:
        out["recall_ok"] = turn.recall_value in raw
    return out


def score_calls(turn, calls_by_call: list[list[dict]]) -> dict:
    """Per-call tool validity/arguments plus the turn-level rule 2 OR and task-tool check.
    calls_by_call: [calls from call 1, calls from call 2 (or [] if no call 2)]."""
    flat = [c for calls in calls_by_call for c in calls]
    valid = [tool_call_valid(c) for c in flat]
    args_ok = [tool_args_correct(c, turn) if v else None for c, v in zip(flat, valid)]
    return {
        "n_tool_calls": len(flat),
        "n_tool_calls_call1": len(calls_by_call[0]) if calls_by_call else 0,
        "n_tool_calls_call2": len(calls_by_call[1]) if len(calls_by_call) > 1 else 0,
        "n_valid_tool_calls": sum(valid),
        "n_args_correct": sum(1 for a in args_ok if a),
        "tool_validity_all": all(valid) if flat else None,
        "tool_args_all": all(a for a in args_ok if a is not None) if any(valid) else None,
        "rule2_log_event_called": any(c.get("name") == "log_event" for c in flat),
        "task_tool_ok": any(v and a and c["name"] == turn.tool_name for c, v, a in zip(flat, valid, args_ok)),
        "recall_key_looked_up": (turn.is_recall and any(
            v and c["name"] == "lookup_fact" and c["arguments"]["key"].strip() == turn.recall_key
            for c, v in zip(flat, valid))),
    }


def transcript_tokens(messages: list[dict]) -> int:
    """Estimated tokens of the whole transcript (each message once, both calls' content included)."""
    total = 0
    for m in messages:
        total += est_tokens(m.get("content")) + PER_MESSAGE_OVERHEAD_TOKENS
        if m.get("tool_calls"):
            total += est_tokens(json.dumps(m["tool_calls"]))
    return total


def prompt_tokens_est(messages: list[dict], tools) -> int:
    return transcript_tokens(messages) + (est_tokens(json.dumps(tools)) if tools else 0)


# ── runtime wrapper (real: Ollama over HTTP; tests inject a fake with the same methods) ────────────

class OllamaRuntime:
    def __init__(self, base="http://127.0.0.1:11434"):
        import t2s_k1_ollama as k1
        self.base = base
        self.client = k1.OllamaClient(timeout=CALL_TIMEOUT_S)

    def chat(self, model, messages, num_ctx, tools, think):
        return self.client.chat(model, "", num_ctx=num_ctx, max_tokens=MAX_TOKENS_PER_CALL, keep_alive=KEEP_ALIVE,
                                messages=messages, tools=tools, think=think, timeout=CALL_TIMEOUT_S)

    def loaded_context(self, model):
        ps = self.client.get_ps()
        for m in ps.get("models", []):
            if (m.get("name") or m.get("model")) == model:
                ctx = m.get("context_length")
                return int(ctx) if ctx is not None else None
        return None

    def unload(self, model, wait_s=30):
        body = json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(f"{self.base}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                r.read()
        except Exception:
            pass
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            names = [(m.get("name") or m.get("model")) for m in self.client.get_ps().get("models", [])]
            if model not in names:
                return True
            time.sleep(0.5)
        return False

    def available_models(self):
        try:
            with urllib.request.urlopen(f"{self.base}/api/tags", timeout=30) as r:
                return [m.get("name") for m in json.loads(r.read()).get("models", [])]
        except Exception:
            return []

    def recover(self):
        """A connection-level failure (no HTTP status): wait for the server, restart it once if needed."""
        import host_config as hc
        if hc.wait_for_ollama_ready(timeout_s=60):
            return True
        hc.stop_ollama_server()
        hc.start_ollama_server()
        return hc.wait_for_ollama_ready(timeout_s=90)


# ── driver ──────────────────────────────────────────────────────────────────────────────────────────

def _call_record(resp: dict, messages, tools) -> dict:
    raw = resp.get("raw") or {}
    msg = raw.get("message") or {}
    return {
        "outcome": resp.get("outcome"), "http_status": resp.get("status"), "error": resp.get("error"),
        "content": (resp.get("message") or "")[:4000], "native_tool_calls": resp.get("tool_calls"),
        "prompt_tokens_est": prompt_tokens_est(messages, tools),
        "completion_tokens": resp.get("eval_count"),
        "prompt_eval_count_info_only": resp.get("prompt_eval_count"),
        "duration_s": resp.get("duration_s"), "done_reason": raw.get("done_reason"),
        "thinking_present": bool(msg.get("thinking")),
        "load_duration_s": (raw.get("load_duration") or 0) / 1e9 if raw.get("load_duration") else None,
        "prompt_eval_duration_s": (raw.get("prompt_eval_duration") or 0) / 1e9 if raw.get("prompt_eval_duration") else None,
        "eval_duration_s": (raw.get("eval_duration") or 0) / 1e9 if raw.get("eval_duration") else None,
    }


def _call_tokens(rec: dict) -> int:
    comp = rec["completion_tokens"] if rec["completion_tokens"] is not None else est_tokens(rec["content"])
    return rec["prompt_tokens_est"] + comp


def _do_call(runtime, model, messages, num_ctx, tools, think, log):
    resp = runtime.chat(model, messages, num_ctx, tools, think)
    retried = False
    if resp.get("outcome") == "error" and resp.get("status") is None:
        log(f"  connection-level error ({resp.get('error')!r}); recovering and retrying once")
        retried = True
        if runtime.recover():
            resp = runtime.chat(model, messages, num_ctx, tools, think)
    rec = _call_record(resp, messages, tools)
    rec["retried_after_connection_error"] = retried
    return resp, rec


def _assistant_msg(resp: dict, method: str) -> dict:
    m = {"role": "assistant", "content": resp.get("message") or ""}
    if method == "native" and resp.get("tool_calls"):
        m["tool_calls"] = resp["tool_calls"]
    return m


def _tool_msgs(calls, store):
    return [{"role": "tool", "content": simulate_tool_result(c, store), "tool_name": c.get("name") or "unknown"}
            for c in calls]


def run_turn(runtime, model, num_ctx, think, ags: AgentSession, turn, messages: list, log,
             call2_tools: bool = True) -> dict:
    """One two-step turn. Mutates `messages` (the session transcript) and returns the scored turn row."""
    tools = r2.ollama_tools_payload()
    messages.append({"role": "user", "content": user_message(turn, ags)})
    resp1, rec1 = _do_call(runtime, model, messages, num_ctx, tools, think, log)
    calls1, method1 = extract_tool_calls(resp1.get("message"), resp1.get("tool_calls"))
    rec1.update({"tool_calls_parsed": calls1, "tool_detection_method": method1})
    calls_by_call = [calls1]
    recs = [rec1]
    if calls1:
        messages.append(_assistant_msg(resp1, method1))
        messages.extend(_tool_msgs(calls1, ags.store))
        resp2, rec2 = _do_call(runtime, model, messages, num_ctx, tools if call2_tools else None, think, log)
        calls2, method2 = extract_tool_calls(resp2.get("message"), resp2.get("tool_calls"))
        rec2.update({"tool_calls_parsed": calls2, "tool_detection_method": method2})
        recs.append(rec2)
        calls_by_call.append(calls2)
        messages.append(_assistant_msg(resp2, method2))
        if calls2:  # keep the transcript well-formed; a third call is not made (R2_DESIGN.md step 4)
            messages.extend(_tool_msgs(calls2, ags.store))
        final_text = resp2.get("message") or ""
    else:
        messages.append(_assistant_msg(resp1, "none"))
        final_text = resp1.get("message") or ""
    spec = ags.spec
    row = {"turn_idx": turn.idx, "tool_task": turn.tool_name, "requires_length": turn.requires_length,
           "is_recall": turn.is_recall, "canary_check": turn.canary_check, "n_calls": len(recs),
           "calls": recs, "final_text": final_text[:4000]}
    row.update(score_calls(turn, calls_by_call))
    exp_sys, exp_hist = expected_canaries(ags, turn)
    row.update(score_final_answer(turn, final_text, spec.session_code, exp_sys, exp_hist))
    row["canary_k"] = canary_index(turn)
    row["other_canaries_in_answer"] = sum(1 for c in ags.sys_canaries + ags.hist_tags
                                          if c in final_text and c not in (exp_sys, exp_hist))
    row["lookup_result_used"] = (turn.tool_name == "lookup_fact"
                                 and ags.store.get(turn.tool_args["key"], "\0") in final_text)
    row["turn_tokens_both_calls"] = sum(_call_tokens(r) for r in recs)
    row["transcript_tokens_est"] = transcript_tokens(messages)
    row["any_http_error"] = any((r["http_status"] not in (200, None)) or r["outcome"] != "ok" or r["error"]
                                for r in recs)
    return row


def run_session(runtime, model, arm, seed, max_turns, emit, log, mode) -> dict:
    """Run one full session; emits r2a_turn rows and one r2a_session row (the completion marker)."""
    num_ctx = ARMS[arm]["num_ctx"]
    call2_tools = ARMS[arm]["call2_tools"]
    think = MODELS.get(model)
    ags = build_agent_session(seed, max_turns)
    attempt = uuid.uuid4().hex[:12]
    base = {"mode": mode, "model_id": model, "arm_id": arm, "num_ctx_requested": num_ctx, "seed": seed,
            "think": think, "call2_tools": call2_tools, "attempt_id": attempt,
            "session_code": ags.spec.session_code}
    unloaded = runtime.unload(model)
    messages = [{"role": "system", "content": ags.system_prompt}]
    calib = None
    cumulative_billed = 0
    rows = []
    for turn in ags.spec.turns:
        t0 = time.monotonic()
        row = run_turn(runtime, model, num_ctx, think, ags, turn, messages, log, call2_tools=call2_tools)
        loaded = runtime.loaded_context(model)
        if turn.idx == 1:
            c1 = row["calls"][0]
            pec = c1.get("prompt_eval_count_info_only")
            if pec and c1["prompt_tokens_est"]:
                ratio = pec / c1["prompt_tokens_est"]
                if CALIB_RATIO_BOUNDS[0] <= ratio <= CALIB_RATIO_BOUNDS[1]:
                    calib = ratio
        cumulative_billed += row["turn_tokens_both_calls"]
        ratio = calib if calib is not None else 1.0
        row.update(base)
        row.update({
            "record": "r2a_turn", "ts_utc": utc_iso(), "turn_wall_s": time.monotonic() - t0,
            "loaded_context": loaded, "token_calib_ratio": calib,
            "transcript_tokens_calibrated": round(row["transcript_tokens_est"] * ratio),
            "final_call_prompt_tokens_calibrated": round(row["calls"][-1]["prompt_tokens_est"] * ratio),
            "session_tokens_billed_cumulative": cumulative_billed,
            "model_unloaded_before_session": unloaded,
        })
        row["over_loaded_window"] = (loaded is not None
                                     and row["final_call_prompt_tokens_calibrated"] > loaded)
        emit(row)
        rows.append(row)
        log(f"  {model} {arm} seed={seed} turn {turn.idx}/{len(ags.spec.turns)} calls={row['n_calls']} "
            f"r1={row['rule1_json_keys']} r2={row['rule2_log_event_called']} r5={row['rule5_session_code']} "
            f"ctx~{row['final_call_prompt_tokens_calibrated']}/{loaded} {row['turn_wall_s']:.1f}s")
    summary = summarize_session(rows)
    summary.update(base)
    summary.update({"record": "r2a_session", "ts_utc": utc_iso(), "n_turns": len(rows)})
    emit(summary)
    try:
        runtime.unload(model)
    except Exception:
        pass
    return summary


# ── session / kill-criterion / baseline analysis (pure, operates on rows) ───────────────────────────

def turn_failures(row: dict, rules_in_use=RULE_IDS) -> list[str]:
    if isinstance(rules_in_use, dict):
        rules_in_use = rules_in_use.get(row.get("model_id"), RULE_IDS)
    fails = [r for r in rules_in_use if row.get(r) is False]
    if row.get("tool_validity_all") is False:
        fails.append("tool_validity")
    if row.get("tool_args_all") is False:
        fails.append("tool_args")
    if row.get("recall_ok") is False:
        fails.append("recall")
    return fails


def truncation_turn(rows: list[dict]):
    """First canary turn with a miss of either canary while the final call's prompt exceeded the loaded window."""
    for r in rows:
        if r.get("canary_check") and r.get("over_loaded_window") and (
                r.get("canary_sys_ok") is False or r.get("canary_hist_ok") is False):
            return r["turn_idx"]
    return None


def summarize_session(rows: list[dict], rules_in_use=RULE_IDS) -> dict:
    rows = sorted(rows, key=lambda r: r["turn_idx"])

    def first(pred):
        return next((r["turn_idx"] for r in rows if pred(r)), None)

    first_failure = first(lambda r: bool(turn_failures(r, rules_in_use)))
    first_error = first(lambda r: r.get("any_http_error"))
    canary_rows = [r for r in rows if r.get("canary_check")]
    return {
        "first_failure_turn": first_failure,
        "first_failure_reasons": next((turn_failures(r, rules_in_use) for r in rows
                                       if r["turn_idx"] == first_failure), None),
        "first_error_turn": first_error,
        "error_surfaced_before_failure": (None if first_failure is None
                                          else (first_error is not None and first_error < first_failure)),
        "truncation_detected_turn": truncation_turn(rows),
        "first_over_window_turn": first(lambda r: r.get("over_loaded_window")),
        "canary_sys_misses": sum(1 for r in canary_rows if r.get("canary_sys_ok") is False),
        "canary_hist_misses": sum(1 for r in canary_rows if r.get("canary_hist_ok") is False),
        "canary_checks": len(canary_rows),
        "canary_misses_within_window": sum(1 for r in canary_rows if not r.get("over_loaded_window") and (
            r.get("canary_sys_ok") is False or r.get("canary_hist_ok") is False)),
        "loaded_context_values": sorted({r.get("loaded_context") for r in rows if r.get("loaded_context")}),
        "token_calib_ratio": rows[0].get("token_calib_ratio") if rows else None,
        "max_transcript_tokens_calibrated": max((r.get("transcript_tokens_calibrated") or 0) for r in rows) if rows else 0,
        "session_tokens_billed": rows[-1].get("session_tokens_billed_cumulative") if rows else 0,
    }


def kill_criterion(session_summaries: list[dict], rules_in_use=RULE_IDS, turn_rows=None) -> dict:
    """The pre-registered R2 kill criterion (docs/FINDINGS.md, 2026-09-29), via r2.evaluate_kill_criterion.
    If turn_rows is given, sessions are re-summarized with rules_in_use (so only gate-passing rules count)."""
    by_arm: dict[str, list[dict]] = {}
    for s in session_summaries:
        if turn_rows is not None:
            rows = [r for r in turn_rows if r.get("record") == "r2a_turn" and r.get("attempt_id") == s["attempt_id"]]
            s = {**s, **summarize_session(rows, rules_in_use)}
        by_arm.setdefault(s["arm_id"], []).append(s)
    return r2.evaluate_kill_criterion(by_arm)


def completed_sessions(rows: list[dict]) -> list[dict]:
    """The r2a_session rows (one per completed session; latest wins per (mode, model, arm, seed))."""
    latest = {}
    for r in rows:
        if r.get("record") == "r2a_session":
            latest[(r["mode"], r["model_id"], r["arm_id"], r["seed"])] = r
    return list(latest.values())


def _rate(vals):
    vals = [v for v in vals if v is not None]
    return (sum(1 for v in vals if v) / len(vals), len(vals)) if vals else (None, 0)


def baseline_table(rows: list[dict], arm="ollama_ctx_131072", threshold=BASELINE_THRESHOLD) -> dict:
    """Per model x rule compliance on the control arm, from completed sessions' turn rows only. A rule with no
    applicable turn is "not_exercised", never counted as 100%."""
    sessions = [s for s in completed_sessions(rows) if s["arm_id"] == arm]
    attempts = {s["attempt_id"] for s in sessions}
    turns = [r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in attempts]
    out = {}
    for model in sorted({r["model_id"] for r in turns}):
        mt = [r for r in turns if r["model_id"] == model]
        calls = [c for r in mt for c in r["calls"]]
        entry = {"n_sessions": len({r["attempt_id"] for r in mt}), "n_turns": len(mt), "n_calls": len(calls),
                 "rules": {}}
        for rule in RULE_IDS:
            rate, n = _rate([r.get(rule) for r in mt])
            entry["rules"][rule] = {"rate": rate, "n": n,
                                    "status": ("not_exercised" if n == 0 else
                                               "pass" if rate >= threshold else "fail")}
        n_tc = sum(r["n_tool_calls"] for r in mt)
        n_valid = sum(r["n_valid_tool_calls"] for r in mt)
        entry["tool_validity"] = {"rate": n_valid / n_tc if n_tc else None, "n_calls": n_tc}
        entry["tool_args"] = {"rate": sum(r["n_args_correct"] for r in mt) / n_valid if n_valid else None,
                              "n_valid_calls": n_valid}
        entry["task_tool_ok"] = dict(zip(("rate", "n"), _rate([r.get("task_tool_ok") for r in mt])))
        entry["recall"] = dict(zip(("rate", "n"), _rate([r.get("recall_ok") for r in mt])))
        entry["recall_key_looked_up"] = sum(1 for r in mt if r.get("recall_key_looked_up"))
        entry["canary_sys"] = dict(zip(("rate", "n"), _rate([r.get("canary_sys_ok") for r in mt])))
        entry["canary_hist"] = dict(zip(("rate", "n"), _rate([r.get("canary_hist_ok") for r in mt])))
        entry["two_call_turn_rate"] = sum(1 for r in mt if r["n_calls"] == 2) / len(mt) if mt else None
        entry["final_content_empty"] = sum(1 for r in mt if r.get("final_content_empty"))
        entry["any_content_empty_calls"] = sum(1 for c in calls if not (c.get("content") or "").strip()
                                               and not c.get("tool_calls_parsed"))
        entry["thinking_present_calls"] = sum(1 for c in calls if c.get("thinking_present"))
        entry["text_fallback_calls"] = sum(1 for c in calls if c.get("tool_detection_method") == "text_fallback")
        entry["http_error_turns"] = sum(1 for r in mt if r.get("any_http_error"))
        entry["think"] = sorted({str(r.get("think")) for r in mt})
        durs = [c["duration_s"] for c in calls if c.get("duration_s") is not None]
        entry["mean_call_s"] = sum(durs) / len(durs) if durs else None
        out[model] = entry
    return out


def control_results(rows: list[dict], sfx: str = "") -> dict:
    sessions = completed_sessions(rows)
    neg = [s for s in sessions if s["arm_id"] == NEG_ARM + sfx]
    pos = [s for s in sessions if s["arm_id"] == POS_ARM + sfx]
    res = {"negative": {}, "positive": {}}
    for model in sorted({s["model_id"] for s in neg + pos}):
        n = [s for s in neg if s["model_id"] == model]
        p = [s for s in pos if s["model_id"] == model]
        res["negative"][model] = {
            "n_sessions": len(n), "canary_checks": sum(s["canary_checks"] for s in n),
            "canary_misses": sum(s["canary_sys_misses"] + s["canary_hist_misses"] for s in n),
            "loaded_context_values": sorted({v for s in n for v in s["loaded_context_values"]}),
            "max_transcript_tokens_calibrated": max((s["max_transcript_tokens_calibrated"] for s in n), default=0),
        }
        res["positive"][model] = {
            "n_sessions": len(p),
            "truncation_detected_turns": [s["truncation_detected_turn"] for s in p],
            "canary_sys_misses": sum(s["canary_sys_misses"] for s in p),
            "canary_hist_misses": sum(s["canary_hist_misses"] for s in p),
            "first_over_window_turns": [s["first_over_window_turn"] for s in p],
            "loaded_context_values": sorted({v for s in p for v in s["loaded_context_values"]}),
        }
    return res


def evaluate_gates(rows: list[dict], threshold=BASELINE_THRESHOLD, call2_tools: bool = True) -> dict:
    """Gates on the chosen call-2 variant's arms; the other variant's arm b (if present) as a diagnostic table."""
    sfx = "" if call2_tools else NOTOOLS_SUFFIX
    table = baseline_table(rows, arm=NEG_ARM + sfx, threshold=threshold)
    ctrl = control_results(rows, sfx)
    gates = {"baseline": {}, "negative_control": {}, "positive_control": {}, "content_nonempty": {}}
    for model, e in table.items():
        failing = [r for r, v in e["rules"].items() if v["status"] == "fail"]
        gates["baseline"][model] = {"pass": not failing, "failing_rules": failing,
                                    "rules_in_use": [r for r, v in e["rules"].items() if v["status"] == "pass"]}
        gates["content_nonempty"][model] = {"pass": e["final_content_empty"] == 0,
                                            "final_content_empty": e["final_content_empty"]}
    for model, n in ctrl["negative"].items():
        gates["negative_control"][model] = {"pass": n["n_sessions"] > 0 and n["canary_misses"] == 0, **n}
    for model, p in ctrl["positive"].items():
        gates["positive_control"][model] = {
            "pass": p["n_sessions"] > 0 and all(t is not None for t in p["truncation_detected_turns"]), **p}
    diag = baseline_table(rows, arm=NEG_ARM + (NOTOOLS_SUFFIX if call2_tools else ""), threshold=threshold)
    return {"call2_tools": call2_tools, "baseline_table": table, "controls": ctrl, "gates": gates,
            "diagnostic_other_call2_variant_table": diag,
            "diagnostic_call2_notools_table": diag if call2_tools else None}


def rules_in_use_from(validation_rows: list[dict], call2_tools: bool) -> dict:
    """Per-model gate-passing rules from a validation file: {model: [rule ids]}."""
    gates = evaluate_gates(validation_rows, call2_tools=call2_tools)["gates"]["baseline"]
    return {m: g["rules_in_use"] for m, g in gates.items()}


def format_baseline_markdown(report: dict) -> str:
    """The checkpoint table as markdown (numbers straight from evaluate_gates)."""
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"
    t = report["baseline_table"]
    models = sorted(t)
    lines = ["| metric | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for rule in RULE_IDS:
        cells = []
        for m in models:
            v = t[m]["rules"][rule]
            cells.append("not exercised" if v["status"] == "not_exercised"
                         else f"{pct(v['rate'])} (n={v['n']}, {v['status']})")
        lines.append(f"| {rule} | " + " | ".join(cells) + " |")
    for key, label, nkey in (("tool_validity", "tool validity (per call)", "n_calls"),
                             ("tool_args", "tool arguments (per valid call)", "n_valid_calls"),
                             ("task_tool_ok", "task tool called correctly (per turn)", "n"),
                             ("recall", "fact recall (recall turns)", "n"),
                             ("canary_sys", "system-prompt canary (canary turns)", "n"),
                             ("canary_hist", "first-user-message canary (canary turns)", "n")):
        lines.append(f"| {label} | " + " | ".join(f"{pct(t[m][key]['rate'])} (n={t[m][key][nkey]})"
                                                 for m in models) + " |")
    lines.append("| two-call turns | " + " | ".join(pct(t[m]["two_call_turn_rate"]) for m in models) + " |")
    lines.append("| empty final answers | " + " | ".join(str(t[m]["final_content_empty"]) for m in models) + " |")
    lines.append("| think sent | " + " | ".join(",".join(t[m]["think"]) for m in models) + " |")
    lines.append("| mean s per call | " + " | ".join("n/a" if t[m]["mean_call_s"] is None
                                                     else f"{t[m]['mean_call_s']:.1f}" for m in models) + " |")
    c = report["controls"]
    lines.append("| negative control canary misses (131072) | " + " | ".join(
        f"{c['negative'][m]['canary_misses']}/{2 * c['negative'][m]['canary_checks']}" for m in models) + " |")
    lines.append("| positive control truncation turn (8192) | " + " | ".join(
        ",".join(str(x) for x in c['positive'].get(m, {}).get('truncation_detected_turns', [])) or "none"
        for m in models) + " |")
    return "\n".join(lines)


# ── CLI / queue entry ───────────────────────────────────────────────────────────────────────────────

def read_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def run_plan(runtime, plan, models, mode, out_path: Path, log=print, emit=None):
    if emit is None:
        def emit(row):
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
    done = {(s["mode"], s["model_id"], s["arm_id"], s["seed"]) for s in completed_sessions(read_rows(out_path))}
    for model in models:
        for arm, seeds, turns in plan:
            for seed in seeds:
                if (mode, model, arm, seed) in done:
                    log(f"skip (done): {model} {arm} seed={seed}")
                    continue
                emit({"record": "heartbeat", "mode": mode, "model_id": model, "arm_id": arm, "seed": seed,
                      "ts_utc": utc_iso()})
                log(f"session: {model} {arm} seed={seed} turns={turns}")
                s = run_session(runtime, model, arm, seed, turns, emit, log, mode)
                log(f"session done: first_failure={s['first_failure_turn']} {s['first_failure_reasons']} "
                    f"truncation={s['truncation_detected_turn']} canary_misses="
                    f"{s['canary_sys_misses']}+{s['canary_hist_misses']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("validation", "real"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--call2-tools", choices=("on", "off"), default="on",
                    help="real mode: keep tools available on call 2 (spec) or withhold them (the _call2_notools arms)")
    ap.add_argument("--rules-from", default=None,
                    help="real mode: a validation JSONL; per-model rules in use = that model's gate-passing rules")
    ap.add_argument("--rules-in-use", default=",".join(RULE_IDS),
                    help="real mode: rules that count toward first-failure / kill criterion (gate-passing rules)")
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{utc_iso()}] {msg}", flush=True)

    note = "completed"
    import host_config as hc
    started = False
    try:
        import socket
        if socket.gethostname().upper() != "EVO-X2":
            raise RuntimeError(f"x2_r2_agent runs on EVO-X2 only, this is {socket.gethostname()!r}")
        models = [m for m in args.models.split(",") if m]
        c2 = args.call2_tools == "on"
        plan = validation_plan(c2) if args.mode == "validation" else real_plan(c2)
        rules_in_use = tuple(args.rules_in_use.split(","))
        if args.rules_from:
            rules_in_use = rules_in_use_from(read_rows(Path(args.rules_from)), c2)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"record": "run_start", "mode": args.mode, "models": models, "plan": plan,
                                "rules_in_use": rules_in_use, "call2_tools": c2, "ts_utc": utc_iso(),
                                "max_tokens_per_call": MAX_TOKENS_PER_CALL, "keep_alive": KEEP_ALIVE},
                               default=str) + "\n")
        hc.start_ollama_server()
        started = True
        if not hc.wait_for_ollama_ready(timeout_s=60):
            hc.stop_ollama_server()
            hc.start_ollama_server()
            if not hc.wait_for_ollama_ready(timeout_s=90):
                raise RuntimeError("ollama did not become ready")
        runtime = OllamaRuntime()
        avail = runtime.available_models()
        missing = [m for m in models if m not in avail]
        if missing:
            raise RuntimeError(f"models not present in this Ollama store: {missing} (have {avail})")
        log(f"x2_r2_agent mode={args.mode} models={models} out={out_path}")
        run_plan(runtime, plan, models, args.mode, out_path, log=log)
        rows = read_rows(out_path)
        if args.mode == "validation":
            report = evaluate_gates(rows, call2_tools=c2)
        else:
            sessions = [s for s in completed_sessions(rows) if s["mode"] == "real"]
            report = {"rules_in_use": rules_in_use, "kill_criterion": kill_criterion(sessions, rules_in_use, rows)}
        Path(str(out_path) + ".report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"record": "run_end", "ts_utc": utc_iso()}) + "\n")
        if args.mode == "validation":
            log("\n" + format_baseline_markdown(report))
            log(json.dumps(report["gates"], default=str))
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        log(note)
        traceback.print_exc()
    finally:
        if started:
            try:
                hc.stop_ollama_server()
            except Exception as e:
                log(f"ollama stop failed: {e!r}")
        try:
            import t2s_queue as tq
            tq.advance(note)
        except Exception as e:
            log(f"queue advance failed: {e!r}")


if __name__ == "__main__":
    main()
