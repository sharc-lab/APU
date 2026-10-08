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

Call-2 modes (--call2-tools, see CALL2_MODES): on = tools available on call 2, nothing forced (the first spec);
off = tools withheld on call 2 only, the "_call2_notools" arms (x2_r2_validation_v2 / x2_r2_real_v1, kept to reproduce
them); forced_none = the protocol for every R2 run from 2026-10-07 on: identical system prompt and identical tool
definitions on every call, "tool_choice": "none" sent on call 2, a call-2 tool call scored as a protocol violation
(Ollama 0.34.4 ignores tool_choice; see R2_DESIGN.md "Call-2 protocol v3"); forced_none_format = forced_none plus an
answer JSON-schema "format" on call 2 (an operator option, not the default). Every call records the SHA-256 of its
system prompt and of its tools JSON and the tool_choice it sent, so "identical on every call" is checkable from rows.
  --mode validation       : arm b (num_ctx 131072) 3 seeds x 10 turns, both models (negative control + baseline
                            table), positive control (num_ctx 8192, 1 seed x 15 turns), and DIAGNOSTIC_MODE's arm b.
  --mode real             : Ollama default vs num_ctx 32768 vs num_ctx 4096, 3 seeds x 40 turns, both models;
                            --rules-from <validation jsonl> sets each model's rules in use for the kill criterion;
                            --require-validation-gates makes the job refuse to start unless validation_preflight
                            passes on that file.
  --mode toolchoice_check : short live check of what forces a text answer on call 2 (run_toolchoice_check).
  --mode real --plan strong : the strengthened run (2026-10-08): STRONG_TIERS (default, 32768, 16384, 8192, 4096) x
                            SEEDS_STRONG (5) x 40 turns x STRONG_MODELS, v1 call-2 mode; --skip-done-from
                            results/x2_r2_real_v1.jsonl keeps v1's completed cells; --rules-from takes the v2 and v2b
                            validation files and --per-model-refusal refuses only a model whose own controls failed.
  --mode mechanism        : one 40-turn session per STRONG_TIERS tier of MECH_MODEL with the Ollama server started
                            under OLLAMA_DEBUG=1, its log captured per call and per session, and a render-only
                            request before every call (harness/x2_r2_mechanism.py has the source evidence).

Thinking: "think": false is sent for every qwen3 model (MODELS); every call records thinking_present (non-empty
message.thinking) and think_tag_in_content, and a turn with either fails ("thinking_present" in turn_failures,
logged with "!!! THINKING PRESENT"). Validation has a thinking_off gate, and the real-run preflight refuses on it.

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
import urllib.error
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
# qwen3-4b-2507 (added 2026-10-08) is the Instruct-2507 release, non-thinking, created on evo-x2 from a bare GGUF
# (no Ollama template, so Ollama 0.34.4 serves it through llama-server's own jinja chat template). "think": false is
# still sent: Ollama 0.34.4's ChatHandler (server/routes.go) turns an omitted think into true for any model whose
# capabilities include thinking, which for a template-less GGUF are read from the GGUF chat template's text
# (server/images.go chatTemplateCapabilities), and only think=true is rejected for a model without the capability.
# Every row records thinking_present / think_tag_in_content and a thinking answer fails its turn (see run_turn).
# qwen3-4b-2507-tools (added 2026-10-08, created by the queued job x2_r2_4b_tools_validate,
# harness/x2_r2_4b_tools_validate.py): the SAME GGUF blob as qwen3-4b-2507 with the Ollama library Qwen3 TEMPLATE
# (copied verbatim from qwen3:8b) and its stop parameters. qwen3-4b-2507 wrote its tool calls as text inside its
# JSON answer in x2_r2_validation_v2b (register R2-validation-v2b-runinfo); this tag isolates the install path.
MODELS = {"llama3.1:8b": None, "qwen3:14b": False, "qwen3:8b": False, "qwen3-4b-2507": False,
          "qwen3-4b-2507-tools": False}

_BASE_ARMS = {
    "ollama_ctx_131072": 131072,                 # arm b, the no-truncation control (validation, negative control)
    "ollama_ctx_8192_positive_control": 8192,    # validation positive control
    "ollama_default": None,                      # real run: no num_ctx sent, the runtime picks
    "ollama_ctx_32768": 32768,
    "ollama_ctx_16384": 16384,                   # strengthened run (2026-10-08) and mechanism job
    "ollama_ctx_8192": 8192,                     # strengthened run (2026-10-08) and mechanism job
    "ollama_ctx_4096": 4096,
}
# Call-2 modes (what a turn's second call sends; call 1 is the same in every mode):
#   "on"                 : tools, nothing else (R2_DESIGN.md step 4 as first written). Arms: no suffix.
#   "off"                : tools withheld on call 2 only (x2_r2_validation_v2 / x2_r2_real_v1). Arms: _call2_notools.
#   "forced_none"        : the operator rule from 2026-10-07: identical system prompt and identical tool definitions
#                          on every call, and "tool_choice": "none" sent on call 2. Ollama 0.34.4 (installed on
#                          evo-x2) has no tool_choice field in api.ChatRequest and silently drops unknown JSON keys, so
#                          on that runtime this request is byte-identical in effect to "on"; the field is sent and
#                          recorded so the data says what was asked for, and a call-2 tool call is scored as a
#                          protocol violation. See R2_DESIGN.md "Call-2 protocol v3". Arms: _call2_forcednone.
#   "forced_none_format" : forced_none plus a JSON-schema "format" (the answer object) on call 2, the one mechanism in
#                          Ollama 0.34.4 that does constrain call 2 to a text answer while tools stay defined. Not the
#                          default: it makes rule 1 (JSON keys) close to automatic. An operator choice, see the doc.
#                          Arms: _call2_forcednone_format.
CALL2_MODES = ("on", "off", "forced_none", "forced_none_format")
NOTOOLS_SUFFIX = "_call2_notools"
MODE_SUFFIX = {"on": "", "off": NOTOOLS_SUFFIX, "forced_none": "_call2_forcednone",
               "forced_none_format": "_call2_forcednone_format"}
FORCED_MODES = ("forced_none", "forced_none_format")
ANSWER_FORMAT_SCHEMA = {"type": "object",
                        "properties": {"answer": {"type": "string"}, "source": {"type": "string"}},
                        "required": ["answer", "source"]}


def call2_extra(mode: str) -> dict:
    """Extra top-level request fields sent on call 2 only (never on call 1)."""
    if mode == "forced_none":
        return {"tool_choice": "none"}
    if mode == "forced_none_format":
        return {"tool_choice": "none", "format": ANSWER_FORMAT_SCHEMA}
    return {}


def as_mode(call2) -> str:
    """Back-compat: True -> "on", False -> "off", a mode string -> itself."""
    if call2 is True:
        return "on"
    if call2 is False:
        return "off"
    if call2 not in CALL2_MODES:
        raise ValueError(f"unknown call-2 mode {call2!r}")
    return call2


ARMS = {}
for _a, _n in _BASE_ARMS.items():
    for _m, _s in MODE_SUFFIX.items():
        ARMS[_a + _s] = {"num_ctx": _n, "call2_tools": _m != "off", "call2_mode": _m}

SEEDS = r2.SEEDS
VALIDATION_PLAN = [  # (arm, seeds, turns) -- the first validation run (2026-10-07, file x2_r2_validation.jsonl)
    ("ollama_ctx_131072", SEEDS, 10),
    ("ollama_ctx_8192_positive_control", SEEDS[:1], 10),
    ("ollama_ctx_131072" + NOTOOLS_SUFFIX, SEEDS, 10),
]
NEG_ARM, POS_ARM = "ollama_ctx_131072", "ollama_ctx_8192_positive_control"
NEG_NUM_CTX = _BASE_ARMS[NEG_ARM]
# the diagnostic arm-b variant run next to each mode's validation (gates never use it)
DIAGNOSTIC_MODE = {"on": "off", "off": "on", "forced_none": "on", "forced_none_format": "forced_none"}


def validation_plan(call2_tools, with_diagnostic: bool = True):
    """Validation for the chosen call-2 mode: negative control/baseline (3 seeds x 10 turns) and positive
    control (1 seed x 15 turns, so two canary checks fall past the 8192 window). DIAGNOSTIC_MODE's arm b is
    added as a diagnostic if with_diagnostic (for forced_none that is "on": the same request minus tool_choice)."""
    mode = as_mode(call2_tools)
    sfx = MODE_SUFFIX[mode]
    plan = [(NEG_ARM + sfx, SEEDS, 10), (POS_ARM + sfx, SEEDS[:1], 15)]
    if with_diagnostic:
        plan.append((NEG_ARM + MODE_SUFFIX[DIAGNOSTIC_MODE[mode]], SEEDS, 10))
    return plan
REAL_PLAN = [
    ("ollama_default", SEEDS, 40),
    ("ollama_ctx_32768", SEEDS, 40),
    ("ollama_ctx_4096", SEEDS, 40),
]


def real_plan(call2_tools):
    sfx = MODE_SUFFIX[as_mode(call2_tools)]
    return [(a + sfx, s, t) for a, s, t in REAL_PLAN]


# Strengthened real run (2026-10-08, results/x2_r2_real_v1b.jsonl): 5 seeds (v1's 3 plus 2 new), 5 tiers, 40 turns,
# v1 call-2 mode. The v1 cells (3 seeds x {default, 32768, 4096} x {llama3.1:8b, qwen3:14b}) are not rerun: main()'s
# --skip-done-from treats sessions completed in results/x2_r2_real_v1.jsonl as done, keyed by (model, arm, seed).
SEEDS_STRONG = tuple(SEEDS) + (20260904, 20260905)
STRONG_TIERS = ("ollama_default", "ollama_ctx_32768", "ollama_ctx_16384", "ollama_ctx_8192", "ollama_ctx_4096")
STRONG_TURNS = 40
# 2026-10-08 (operator): the 4B runs as qwen3-4b-2507-tools (library template), validated in
# x2_r2_validation_v2c.jsonl; the bare-GGUF qwen3-4b-2507 failed the task-tool gate in v2b (tool calls as text).
STRONG_MODELS = ("llama3.1:8b", "qwen3:14b", "qwen3-4b-2507-tools", "qwen3:8b")
# Operator decision 2026-10-08: every Ollama and llama-server R2 run uses call-2 mode "off" (v1, tools withheld on a
# turn's second call); a cloud (OpenAI) R2 session, if one runs, uses "forced_none" (identical tools, tool_choice
# "none" on call 2), which that API enforces. See R2_DESIGN.md "Call-2 mode per runtime".
CALL2_MODE_BY_RUNTIME = {"ollama": "off", "llama_server": "off", "openai": "forced_none"}


def strong_plan(call2_tools="off", tiers=STRONG_TIERS, seeds=SEEDS_STRONG, turns=STRONG_TURNS):
    sfx = MODE_SUFFIX[as_mode(call2_tools)]
    return [(a + sfx, tuple(seeds), turns) for a in tiers]


# Mechanism job (2026-10-08, results/x2_r2_mechanism.jsonl): one 40-turn session per tier, one model, with the Ollama
# server started under OLLAMA_DEBUG=1 and its log captured per call. MECH_MODEL is llama3.1:8b: at the four
# num_ctx tiers it exceeds the window by turn 3 (4096) to turn 23 (32768) in x2_r2_real_v1, its 40-turn sessions are
# the shortest measured (7 to 43 min), and it is the model the single-prompt half-window finding was measured with.
# At Ollama default it loaded 131072 and never exceeded it in v1, so that session is the no-overflow reference.
MECH_MODEL = "llama3.1:8b"
MECH_TURNS = 40


def mechanism_plan(call2_tools="off", tiers=STRONG_TIERS, seed=SEEDS[0], turns=MECH_TURNS):
    sfx = MODE_SUFFIX[as_mode(call2_tools)]
    return [(a + sfx, (seed,), turns) for a in tiers]

MAX_TOKENS_PER_CALL = 384
KEEP_ALIVE = "30m"          # hc.start_ollama_server sets OLLAMA_KEEP_ALIVE=0; keep the model (and its KV cache)
                            # loaded across a session's calls, then unload explicitly at the session boundary
CALL_TIMEOUT_S = 900
BASELINE_THRESHOLD = 0.90
# Task-tool gate (2026-10-08): the turn's own task tool (lookup_fact with the right key, or log_event with the right
# rack/length) called validly with correct arguments on at least this share of baseline-arm turns. Uniform for every
# model; a model below it is refused for the real run (validation_preflight). Added after qwen3-4b-2507 passed the
# rule gates in x2_r2_validation_v2b with the task tool correct on only 5 of 30 baseline turns.
TASK_TOOL_THRESHOLD = BASELINE_THRESHOLD
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


# ── request construction and protocol bookkeeping (pure) ───────────────────────────────────────────

def native_chat_body(model, messages, num_ctx, tools, think, extra=None) -> dict:
    """The /api/chat body, field for field what OllamaClient.chat sends, plus `extra` top-level fields."""
    options = {"num_predict": MAX_TOKENS_PER_CALL, "temperature": 0, "seed": 42}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    body = {"model": model, "messages": messages, "stream": False, "options": options, "keep_alive": KEEP_ALIVE}
    if tools is not None:
        body["tools"] = tools
    if think is not None:
        body["think"] = think
    body.update(extra or {})
    return body


def sha256_text(s) -> str | None:
    import hashlib
    if s is None:
        return None
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def tools_sha256(tools) -> str | None:
    """Hash of the tools JSON exactly as serialized into the request (json.dumps defaults, as the clients do)."""
    return None if tools is None else sha256_text(json.dumps(tools))


def system_prompt_sha256(messages) -> str | None:
    sys_msgs = [m.get("content") or "" for m in messages if m.get("role") == "system"]
    return sha256_text("\n\0\n".join(sys_msgs)) if sys_msgs else None


def rendered_template_of(data) -> str | None:
    """The rendered prompt from a _debug_render_only response (api.DebugInfo.RenderedTemplate); the JSON key name is
    looked up permissively (any top-level key containing "debug", then any key containing "render")."""
    if not isinstance(data, dict):
        return None
    for k, v in data.items():
        if "debug" in k.lower() and isinstance(v, dict):
            for k2, v2 in v.items():
                if "render" in k2.lower() and isinstance(v2, str):
                    return v2
    return None


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Ollama-native transcript -> OpenAI chat format: assistant tool_calls get ids, type and string arguments; each
    tool message gets the tool_call_id of the call it answers (matched in order)."""
    out, pending = [], []
    n = 0
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                n += 1
                cid = tc.get("id") or f"call_{n}"
                tcs.append({"id": cid, "type": "function",
                            "function": {"name": fn.get("name"),
                                         "arguments": args if isinstance(args, str) else json.dumps(args)}})
                pending.append(cid)
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
        elif m["role"] == "tool":
            msg = {"role": "tool", "content": m.get("content") or ""}
            if pending:
                msg["tool_call_id"] = pending.pop(0)
            if m.get("tool_name"):
                msg["name"] = m["tool_name"]
            out.append(msg)
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


# ── runtime wrapper (real: Ollama over HTTP; tests inject a fake with the same methods) ────────────

class OllamaRuntime:
    def __init__(self, base="http://127.0.0.1:11434"):
        import t2s_k1_ollama as k1
        self.base = base
        self.client = k1.OllamaClient(timeout=CALL_TIMEOUT_S)

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        """extra: top-level request fields added on call 2 by the forced modes (call2_extra). Without it, the
        request goes through OllamaClient.chat exactly as in x2_r2_validation_v2 / x2_r2_real_v1."""
        if not extra:
            return self.client.chat(model, "", num_ctx=num_ctx, max_tokens=MAX_TOKENS_PER_CALL,
                                    keep_alive=KEEP_ALIVE, messages=messages, tools=tools, think=think,
                                    timeout=CALL_TIMEOUT_S)
        body = native_chat_body(model, messages, num_ctx, tools, think, extra)
        return self._post_native(body)

    def _post(self, path, body):
        req = urllib.request.Request(f"{self.base}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
                return r.status, json.loads(r.read()), None, time.monotonic() - t0
        except urllib.error.HTTPError as e:
            return e.code, None, e.read().decode(errors="replace")[:400], time.monotonic() - t0
        except Exception as e:
            return None, None, str(e)[:400], time.monotonic() - t0

    def _post_native(self, body):
        """Same return shape as OllamaClient.chat."""
        status, data, err, dt = self._post("/api/chat", body)
        if data is None:
            return {"outcome": "http_error" if status else "error", "status": status, "error": err,
                    "duration_s": dt, "num_ctx_requested": body.get("options", {}).get("num_ctx")}
        msg = data.get("message") or {}
        return {"outcome": "ok", "status": status, "message": msg.get("content"), "tool_calls": msg.get("tool_calls"),
                "prompt_eval_count": data.get("prompt_eval_count"), "eval_count": data.get("eval_count"),
                "duration_s": dt, "num_ctx_requested": body.get("options", {}).get("num_ctx"), "raw": data}

    def render_only(self, model, messages, num_ctx, tools, think, extra=None):
        """/api/chat with _debug_render_only: the prompt text the template renders (no generation). None on error."""
        body = native_chat_body(model, messages, num_ctx, tools, think, extra)
        body["_debug_render_only"] = True
        status, data, err, _ = self._post("/api/chat", body)
        return rendered_template_of(data), (None if data is not None else f"{status} {err}")

    def chat_openai(self, model, messages, tools, think, tool_choice=None):
        """/v1/chat/completions (OpenAI-compatible). num_ctx cannot be set on this endpoint in 0.34.4."""
        body = {"model": model, "messages": to_openai_messages(messages), "stream": False,
                "max_tokens": MAX_TOKENS_PER_CALL, "temperature": 0, "seed": 42}
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if think is False:
            body["reasoning_effort"] = "none"   # openai.ThinkingFromReasoningEffort("none") -> think false
        status, data, err, dt = self._post("/v1/chat/completions", body)
        if data is None:
            return {"outcome": "error", "status": status, "error": err, "duration_s": dt}
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        return {"outcome": "ok", "status": status, "message": msg.get("content"), "tool_calls": msg.get("tool_calls"),
                "duration_s": dt, "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason")}

    def version(self):
        try:
            with urllib.request.urlopen(f"{self.base}/api/version", timeout=30) as r:
                return json.loads(r.read()).get("version")
        except Exception:
            return None

    def loaded_context(self, model):
        ps = self.client.get_ps()
        for m in ps.get("models", []):
            if same_tag(m.get("name") or m.get("model"), model):
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
            if not any(same_tag(n, model) for n in names):
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
        """A connection-level failure (no HTTP status): wait for the server, restart it once if needed (with the
        same env and log path the job started it with)."""
        import host_config as hc
        if hc.wait_for_ollama_ready(timeout_s=60):
            return True
        hc.stop_ollama_server()
        _start_server(hc, self.server_env, self.server_log)
        return hc.wait_for_ollama_ready(timeout_s=90)

    server_env = None   # set by main() for the mechanism job (OLLAMA_DEBUG)
    server_log = None


def norm_tag(tag) -> str | None:
    """Ollama reports a tagless model with an explicit ":latest" (/api/tags and /api/ps list "qwen3-4b-2507" as
    "qwen3-4b-2507:latest", see x2_model_pulls._normalize_tag); requests and rows use the bare form."""
    if not tag:
        return tag
    return tag if ":" in tag else tag + ":latest"


def same_tag(a, b) -> bool:
    return a is not None and b is not None and norm_tag(a) == norm_tag(b)


def _start_server(hc, env=None, log_path=None):
    """hc.start_ollama_server with env/log_path only when set, so the plain call (and test fakes without those
    parameters) are unchanged for every job except the mechanism job."""
    if env or log_path:
        return hc.start_ollama_server(env=env, log_path=log_path)
    return hc.start_ollama_server()


# ── driver ──────────────────────────────────────────────────────────────────────────────────────────

def _call_record(resp: dict, messages, tools, extra=None) -> dict:
    raw = resp.get("raw") or {}
    msg = raw.get("message") or {}
    extra = extra or {}
    return {
        # what this call sent, so "identical system prompt and tools on every call" is checkable from the data
        "system_prompt_sha256": system_prompt_sha256(messages), "tools_sha256": tools_sha256(tools),
        "tools_sent": tools is not None, "tool_choice_sent": extra.get("tool_choice"),
        "format_sent_sha256": (sha256_text(json.dumps(extra["format"], sort_keys=True))
                               if extra.get("format") is not None else None),
        "outcome": resp.get("outcome"), "http_status": resp.get("status"), "error": resp.get("error"),
        "content": (resp.get("message") or "")[:4000], "native_tool_calls": resp.get("tool_calls"),
        "prompt_tokens_est": prompt_tokens_est(messages, tools),
        "completion_tokens": resp.get("eval_count"),
        "prompt_eval_count_info_only": resp.get("prompt_eval_count"),
        "duration_s": resp.get("duration_s"), "done_reason": raw.get("done_reason"),
        "thinking_present": bool(msg.get("thinking")),
        "think_tag_in_content": has_think_tag(resp.get("message")),
        "load_duration_s": (raw.get("load_duration") or 0) / 1e9 if raw.get("load_duration") else None,
        "prompt_eval_duration_s": (raw.get("prompt_eval_duration") or 0) / 1e9 if raw.get("prompt_eval_duration") else None,
        "eval_duration_s": (raw.get("eval_duration") or 0) / 1e9 if raw.get("eval_duration") else None,
    }


_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


def has_think_tag(text) -> bool:
    """A <think> or </think> tag anywhere in a call's content (thinking that leaked into the answer text)."""
    return bool(text) and bool(_THINK_TAG_RE.search(text))


def _call_tokens(rec: dict) -> int:
    comp = rec["completion_tokens"] if rec["completion_tokens"] is not None else est_tokens(rec["content"])
    return rec["prompt_tokens_est"] + comp


def _do_call(runtime, model, messages, num_ctx, tools, think, log, extra=None):
    def go():
        if extra:
            return runtime.chat(model, messages, num_ctx, tools, think, extra=extra)
        return runtime.chat(model, messages, num_ctx, tools, think)
    resp = go()
    retried = False
    if resp.get("outcome") == "error" and resp.get("status") is None:
        log(f"  connection-level error ({resp.get('error')!r}); recovering and retrying once")
        retried = True
        if runtime.recover():
            resp = go()
    rec = _call_record(resp, messages, tools, extra)
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
             call2_tools=True) -> dict:
    """One two-step turn. Mutates `messages` (the session transcript) and returns the scored turn row.
    call2_tools: a call-2 mode (CALL2_MODES) or the old bool (True = "on", False = "off")."""
    mode = as_mode(call2_tools)
    tools = r2.ollama_tools_payload()
    messages.append({"role": "user", "content": user_message(turn, ags)})
    resp1, rec1 = _do_call(runtime, model, messages, num_ctx, tools, think, log)
    calls1, method1 = extract_tool_calls(resp1.get("message"), resp1.get("tool_calls"))
    rec1.update({"tool_calls_parsed": calls1, "tool_detection_method": method1})
    calls_by_call = [calls1]
    recs = [rec1]
    extra2 = call2_extra(mode)
    if calls1:
        messages.append(_assistant_msg(resp1, method1))
        messages.extend(_tool_msgs(calls1, ags.store))
        resp2, rec2 = _do_call(runtime, model, messages, num_ctx, None if mode == "off" else tools, think, log,
                               extra=extra2)
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
    two = len(recs) == 2
    row.update({
        "call2_mode": mode,
        "call2_tool_choice_sent": recs[1]["tool_choice_sent"] if two else None,
        "system_prompt_identical_across_calls": (recs[0]["system_prompt_sha256"] == recs[1]["system_prompt_sha256"]
                                                 if two else None),
        "tools_identical_across_calls": recs[0]["tools_sha256"] == recs[1]["tools_sha256"] if two else None,
        # forced modes: a tool call in call 2 after tool_choice "none" was sent breaks the protocol; it is still
        # scored for validity/arguments/rule 2 like any call-2 tool call, and also counts as a turn failure
        "call2_protocol_violation": (bool(calls_by_call[1]) if two and mode in FORCED_MODES else None),
    })
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
    # thinking must be off on every call (think false sent where the model supports it): a non-empty
    # message.thinking or a <think> tag in any call's content fails the turn (turn_failures "thinking_present")
    row["thinking_present"] = any(r.get("thinking_present") or r.get("think_tag_in_content") for r in recs)
    if row["thinking_present"]:
        log(f"  !!! THINKING PRESENT: {model} turn {turn.idx} (think sent: {think!r}); turn scored as failed")
    return row


def run_session(runtime, model, arm, seed, max_turns, emit, log, mode) -> dict:
    """Run one full session; emits r2a_turn rows and one r2a_session row (the completion marker)."""
    num_ctx = ARMS[arm]["num_ctx"]
    call2_tools = ARMS[arm]["call2_tools"]
    call2_mode = ARMS[arm]["call2_mode"]
    think = MODELS.get(model)
    ags = build_agent_session(seed, max_turns)
    attempt = uuid.uuid4().hex[:12]
    base = {"mode": mode, "model_id": model, "arm_id": arm, "num_ctx_requested": num_ctx, "seed": seed,
            "think": think, "call2_tools": call2_tools, "call2_mode": call2_mode, "attempt_id": attempt,
            "session_code": ags.spec.session_code}
    unloaded = runtime.unload(model)
    messages = [{"role": "system", "content": ags.system_prompt}]
    calib = None
    cumulative_billed = 0
    rows = []
    for turn in ags.spec.turns:
        t0 = time.monotonic()
        row = run_turn(runtime, model, num_ctx, think, ags, turn, messages, log, call2_tools=call2_mode)
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
    if row.get("call2_protocol_violation") is True:
        fails.append("call2_protocol_violation")
    if row.get("thinking_present") is True:
        fails.append("thinking_present")
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
        "call2_protocol_violations": sum(1 for r in rows if r.get("call2_protocol_violation") is True),
        "thinking_present_turns": sum(1 for r in rows if r.get("thinking_present") is True),
        "prompt_or_tools_changed_within_turn": sum(
            1 for r in rows if r.get("system_prompt_identical_across_calls") is False
            or r.get("tools_identical_across_calls") is False),
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
        entry["thinking_present_calls"] = sum(1 for c in calls if c.get("thinking_present")
                                              or c.get("think_tag_in_content"))
        entry["text_fallback_calls"] = sum(1 for c in calls if c.get("tool_detection_method") == "text_fallback")
        entry["http_error_turns"] = sum(1 for r in mt if r.get("any_http_error"))
        entry["call2_modes"] = sorted({str(r.get("call2_mode", "on" if r.get("call2_tools", True) else "off"))
                                       for r in mt})
        entry["two_call_turns"] = sum(1 for r in mt if r["n_calls"] == 2)
        entry["call2_tool_call_turns"] = sum(1 for r in mt if r.get("n_tool_calls_call2"))
        entry["call2_protocol_violations"] = sum(1 for r in mt if r.get("call2_protocol_violation") is True)
        entry["prompt_or_tools_changed_within_turn"] = sum(
            1 for r in mt if r.get("system_prompt_identical_across_calls") is False
            or r.get("tools_identical_across_calls") is False)
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


def evaluate_gates(rows: list[dict], threshold=BASELINE_THRESHOLD, call2_tools=True) -> dict:
    """Gates on the chosen call-2 mode's arms; DIAGNOSTIC_MODE's arm b (if present) as a diagnostic table.
    call2_tools: a call-2 mode or the old bool (True = "on", False = "off")."""
    mode = as_mode(call2_tools)
    sfx = MODE_SUFFIX[mode]
    table = baseline_table(rows, arm=NEG_ARM + sfx, threshold=threshold)
    ctrl = control_results(rows, sfx)
    gates = {"baseline": {}, "negative_control": {}, "positive_control": {}, "content_nonempty": {},
             "thinking_off": {}, "task_tool": {}}
    for model, e in table.items():
        tt_rate, tt_n = e["task_tool_ok"]["rate"], e["task_tool_ok"]["n"]
        tt_ok = round(tt_rate * tt_n) if tt_rate is not None else 0
        # integer comparison (27/30 at a 0.90 threshold passes exactly, no float edge)
        gates["task_tool"][model] = {"pass": tt_n > 0 and tt_ok * 100 >= round(TASK_TOOL_THRESHOLD * 100) * tt_n,
                                     "rate": tt_rate, "n": tt_n, "n_ok": tt_ok, "threshold": TASK_TOOL_THRESHOLD}
        failing = [r for r, v in e["rules"].items() if v["status"] == "fail"]
        gates["baseline"][model] = {"pass": not failing, "failing_rules": failing,
                                    "rules_in_use": [r for r, v in e["rules"].items() if v["status"] == "pass"]}
        gates["content_nonempty"][model] = {"pass": e["final_content_empty"] == 0,
                                            "final_content_empty": e["final_content_empty"]}
        gates["thinking_off"][model] = {"pass": e["thinking_present_calls"] == 0,
                                        "thinking_present_calls": e["thinking_present_calls"]}
    for model, n in ctrl["negative"].items():
        gates["negative_control"][model] = {"pass": n["n_sessions"] > 0 and n["canary_misses"] == 0, **n}
    for model, p in ctrl["positive"].items():
        gates["positive_control"][model] = {
            "pass": p["n_sessions"] > 0 and all(t is not None for t in p["truncation_detected_turns"]), **p}
    diag_mode = DIAGNOSTIC_MODE[mode]
    diag = baseline_table(rows, arm=NEG_ARM + MODE_SUFFIX[diag_mode], threshold=threshold)
    return {"call2_tools": mode != "off", "call2_mode": mode, "baseline_table": table, "controls": ctrl,
            "gates": gates, "diagnostic_mode": diag_mode,
            "diagnostic_other_call2_variant_table": diag,
            "diagnostic_call2_notools_table": diag if diag_mode == "off" else None}


def rules_in_use_from(validation_rows: list[dict], call2_tools) -> dict:
    """Per-model gate-passing rules from a validation file: {model: [rule ids]}."""
    gates = evaluate_gates(validation_rows, call2_tools=call2_tools)["gates"]["baseline"]
    return {m: g["rules_in_use"] for m, g in gates.items()}


def validation_preflight(validation_rows: list[dict], call2_tools, models, threshold=BASELINE_THRESHOLD) -> dict:
    """The real run's own start check against its validation file (x2_r2_real_v2 refuses to start unless this
    passes; no manual step). For every model: the validation file finished (run_end) under the same call-2 mode, the
    plan's negative-control (3 sessions) and positive-control (1 session) arms completed, the model has at least one
    rule in use and every rule in use is at or above the threshold, the negative control had no canary miss, and
    the positive control fired on every session, no thinking, and (2026-10-08) the task-tool gate: the turn's task
    tool called correctly on >= TASK_TOOL_THRESHOLD of baseline turns. Returns {"ok", "reasons", "rules_in_use", "gates"}."""
    mode = as_mode(call2_tools)
    reasons = []
    if not any(r.get("record") == "run_end" for r in validation_rows):
        reasons.append("validation file has no run_end record (validation did not finish)")
    starts = [r for r in validation_rows if r.get("record") == "run_start" and r.get("mode") == "validation"]
    start_modes = {str(r.get("call2_mode", "on" if r.get("call2_tools", True) else "off")) for r in starts}
    if start_modes != {mode}:
        reasons.append(f"validation file call-2 mode {sorted(start_modes)} != real run's {mode!r}")
    rep = evaluate_gates(validation_rows, threshold=threshold, call2_tools=mode)
    g, t = rep["gates"], rep["baseline_table"]
    need = {a: len(s) for a, s, _ in validation_plan(mode, with_diagnostic=False)}
    need_neg, need_pos = need[NEG_ARM + MODE_SUFFIX[mode]], need[POS_ARM + MODE_SUFFIX[mode]]
    riu = {}
    for m in models:
        e = t.get(m)
        if e is None or e["n_sessions"] < need_neg:
            reasons.append(f"{m}: negative-control arm incomplete "
                           f"({0 if e is None else e['n_sessions']} of {need_neg} sessions)")
            continue
        riu[m] = list(g["baseline"][m]["rules_in_use"])
        if not riu[m]:
            reasons.append(f"{m}: no rule reached {threshold:.0%} on the validation baseline")
        for rule in riu[m]:
            rate = e["rules"][rule]["rate"]
            if rate is None or rate < threshold:
                reasons.append(f"{m}: rule in use {rule} at {rate} < {threshold}")
        neg = g["negative_control"].get(m)
        if not neg or not neg["pass"]:
            reasons.append(f"{m}: negative control failed (canary misses "
                           f"{neg['canary_misses'] if neg else 'n/a'} at num_ctx 131072)")
        pos = g["positive_control"].get(m)
        if not pos or pos["n_sessions"] < need_pos or not pos["pass"]:
            reasons.append(f"{m}: positive control did not fire (truncation turns "
                           f"{pos['truncation_detected_turns'] if pos else 'n/a'} at num_ctx 8192)")
        th = g.get("thinking_off", {}).get(m)
        if th is not None and not th["pass"]:
            reasons.append(f"{m}: thinking present on {th['thinking_present_calls']} validation calls")
        tt = g.get("task_tool", {}).get(m)
        if tt is not None and not tt["pass"]:
            reasons.append(f"{m}: task tool called correctly on {tt['n_ok']}/{tt['n']} baseline turns "
                           f"(< {tt['threshold']:.0%}, task-tool gate)")
    return {"ok": not reasons, "reasons": reasons, "rules_in_use": riu, "gates": g}


def validation_preflight_per_model(files: list[tuple[str, list[dict]]], call2_tools, models,
                                   threshold=BASELINE_THRESHOLD) -> dict:
    """The strengthened real run's start check (2026-10-08). files: [(name, rows)], e.g. x2_r2_validation_v2.jsonl
    for llama3.1:8b and qwen3:14b and x2_r2_validation_v2b.jsonl for the new models. Each model is checked against
    the LAST file that holds validation turn rows for it, with validation_preflight's full set of checks (file
    finished, same call-2 mode, both control arms complete, rules in use at or above the threshold, negative
    control clean, positive control fired, no thinking). A model that fails is refused alone; the others run.
    Returns {"ok" (at least one model passes), "models_ok", "refused": {model: reasons}, "rules_in_use",
    "source": {model: file name}, "per_file": {name: validation_preflight result}}."""
    source = {}
    for name, rows in files:
        have = {r.get("model_id") for r in rows if r.get("record") == "r2a_turn" and r.get("mode") == "validation"}
        for m in models:
            if m in have:
                source[m] = name
    refused, riu, per_file = {}, {}, {}
    for m in models:
        if m not in source:
            refused[m] = [f"{m}: no validation rows in any of {[n for n, _ in files]}"]
    for name, rows in files:
        ms = [m for m in models if source.get(m) == name]
        if not ms:
            continue
        pf = validation_preflight(rows, call2_tools, ms, threshold=threshold)
        per_file[name] = {k: pf[k] for k in ("ok", "reasons", "rules_in_use")}
        file_level = [r for r in pf["reasons"] if not any(r.startswith(f"{m}:") for m in ms)]
        for m in ms:
            mine = file_level + [r for r in pf["reasons"] if r.startswith(f"{m}:")]
            if mine:
                refused[m] = mine
            else:
                riu[m] = pf["rules_in_use"][m]
    models_ok = [m for m in models if m not in refused]
    return {"ok": bool(models_ok), "models_ok": models_ok, "refused": refused, "rules_in_use": riu,
            "source": source, "per_file": per_file}


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
    lines.append("| call-2 tool-call turns / two-call turns | " + " | ".join(
        f"{t[m].get('call2_tool_call_turns', 'n/a')}/{t[m].get('two_call_turns', 'n/a')}" for m in models) + " |")
    lines.append("| call-2 protocol violations (forced modes) | " + " | ".join(
        str(t[m].get("call2_protocol_violations", "n/a")) for m in models) + " |")
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


# ── tool_choice check (--mode toolchoice_check, queue id x2_r2_toolchoice_check) ─────────────────────
#
# Does anything on this Ollama force a text answer on call 2 while the tools stay defined? Source reading for
# 0.34.4 (R2_DESIGN.md "Call-2 protocol v3") says: /api/chat has no tool_choice field (unknown JSON keys are
# dropped), /v1/chat/completions lists tool_choice as unsupported and binds with ShouldBindJSON (also drops it), and
# a JSON-schema "format" constrains generation. This job checks that live, per model, on the real call-2 contexts of
# a short forced_none session (arm b num_ctx, seed SEEDS[0], TC_TURNS turns):
#   on, on_repeat         : /api/chat, tools, nothing else (on_repeat measures run-to-run nondeterminism)
#   tool_choice_none      : /api/chat, tools, "tool_choice": "none" (this is the call that continues the transcript)
#   format_schema         : /api/chat, tools, tool_choice none, format = ANSWER_FORMAT_SCHEMA
#   render_on / render_tool_choice_none : _debug_render_only; the rendered prompt text, compared byte for byte
#   v1_on / v1_tool_choice_none          : /v1/chat/completions on the same contexts (phase 2, after the session;
#                                          that endpoint cannot set num_ctx, so the model reloads at its default)

TC_TURNS = 6
TC_NATIVE_VARIANTS = ("on", "on_repeat", "tool_choice_none", "format_schema")
TC_V1_VARIANTS = ("v1_on", "v1_tool_choice_none")
TC_EXTRA = {"on": {}, "on_repeat": {}, "tool_choice_none": {"tool_choice": "none"},
            "format_schema": {"tool_choice": "none", "format": ANSWER_FORMAT_SCHEMA}}


# the call-2 mode each toolchoice-check variant corresponds to (every row carries call2_mode); call 1 and the
# transcript continuation belong to the forced_none session the check is built on
TC_VARIANT_MODE = {"call1": "forced_none", "on": "on", "on_repeat": "on", "tool_choice_none": "forced_none",
                   "format_schema": "forced_none_format", "render_on": "on", "render_tool_choice_none": "forced_none",
                   "v1_on": "on", "v1_tool_choice_none": "forced_none"}


def _tc_row(model, turn_idx, variant, endpoint, resp, extra_fields=None):
    calls, method = extract_tool_calls(resp.get("message"), resp.get("tool_calls"))
    row = {"record": "r2tc_call", "model_id": model, "turn_idx": turn_idx, "variant": variant, "endpoint": endpoint,
           "call2_mode": TC_VARIANT_MODE.get(variant),
           "outcome": resp.get("outcome"), "http_status": resp.get("status"), "error": resp.get("error"),
           "content": (resp.get("message") or "")[:4000], "native_tool_calls": resp.get("tool_calls"),
           "n_tool_calls": len(calls), "tool_detection_method": method,
           "content_empty": not (resp.get("message") or "").strip(),
           "answer_json_ok": isinstance(parse_json_strict(resp.get("message") or ""), dict),
           "duration_s": resp.get("duration_s"), "ts_utc": utc_iso()}
    row.update(extra_fields or {})
    return row


def run_toolchoice_check(runtime, models, emit, log, turns=TC_TURNS):
    tools = r2.ollama_tools_payload()
    for model in models:
        think = MODELS.get(model)
        ags = build_agent_session(SEEDS[0], turns)
        runtime.unload(model)
        messages = [{"role": "system", "content": ags.system_prompt}]
        contexts = []
        for turn in ags.spec.turns:
            messages.append({"role": "user", "content": user_message(turn, ags)})
            resp1 = runtime.chat(model, messages, NEG_NUM_CTX, tools, think)
            calls1, method1 = extract_tool_calls(resp1.get("message"), resp1.get("tool_calls"))
            emit(_tc_row(model, turn.idx, "call1", "api_chat", resp1,
                         {"system_prompt_sha256": system_prompt_sha256(messages), "tools_sha256": tools_sha256(tools)}))
            if not calls1:
                messages.append(_assistant_msg(resp1, "none"))
                log(f"  {model} turn {turn.idx}: no call-1 tool call, no call-2 context")
                continue
            messages.append(_assistant_msg(resp1, method1))
            messages.extend(_tool_msgs(calls1, ags.store))
            ctx = [dict(m) for m in messages]
            contexts.append((turn.idx, ctx))
            renders = {}
            for v, ex in (("render_on", {}), ("render_tool_choice_none", {"tool_choice": "none"})):
                text, err = runtime.render_only(model, ctx, NEG_NUM_CTX, tools, think, ex)
                renders[v] = text
                emit({"record": "r2tc_render", "model_id": model, "turn_idx": turn.idx, "variant": v,
                      "call2_mode": TC_VARIANT_MODE[v],
                      "rendered_sha256": sha256_text(text), "rendered_len": len(text) if text else None,
                      "error": err, "ts_utc": utc_iso()})
            cont = None
            for v in TC_NATIVE_VARIANTS:
                ex = TC_EXTRA[v]
                resp = runtime.chat(model, ctx, NEG_NUM_CTX, tools, think, extra=ex) if ex else \
                    runtime.chat(model, ctx, NEG_NUM_CTX, tools, think)
                emit(_tc_row(model, turn.idx, v, "api_chat", resp,
                             {"tool_choice_sent": ex.get("tool_choice"), "format_sent": "format" in ex,
                              "system_prompt_sha256": system_prompt_sha256(ctx), "tools_sha256": tools_sha256(tools)}))
                if v == "tool_choice_none":
                    cont = resp
            calls2, method2 = extract_tool_calls(cont.get("message"), cont.get("tool_calls"))
            messages.append(_assistant_msg(cont, method2))
            if calls2:
                messages.extend(_tool_msgs(calls2, ags.store))
            log(f"  {model} turn {turn.idx}: renders identical="
                f"{renders['render_on'] is not None and renders['render_on'] == renders['render_tool_choice_none']}")
        runtime.unload(model)
        for turn_idx, ctx in contexts:
            for v in TC_V1_VARIANTS:
                tc = "none" if v == "v1_tool_choice_none" else None
                resp = runtime.chat_openai(model, ctx, tools, think, tool_choice=tc)
                emit(_tc_row(model, turn_idx, v, "v1_chat_completions", resp, {"tool_choice_sent": tc}))
        runtime.unload(model)


def summarize_toolchoice_check(rows: list[dict]) -> dict:
    """Per model: per-variant call-2 tool-call / empty / answer-JSON counts, pairwise identical-output counts, and
    rendered-prompt identity with and without tool_choice. Verdict per mechanism: "no_effect_observed" when every
    rendered prompt is identical and tool_choice none produced as many call-2 tool calls as plain "on" (or more),
    "effect_observed" otherwise."""
    out = {}
    for model in sorted({r["model_id"] for r in rows if r.get("record") in ("r2tc_call", "r2tc_render")}):
        calls = [r for r in rows if r.get("record") == "r2tc_call" and r["model_id"] == model]
        by = {}
        for r in calls:
            by.setdefault(r["variant"], {})[r["turn_idx"]] = r
        variants = {}
        for v, d in by.items():
            if v == "call1":
                continue
            variants[v] = {"n": len(d), "tool_call_turns": sum(1 for r in d.values() if r["n_tool_calls"]),
                           "content_empty": sum(1 for r in d.values() if r["content_empty"]),
                           "answer_json_ok": sum(1 for r in d.values() if r["answer_json_ok"]),
                           "errors": sum(1 for r in d.values() if r["outcome"] != "ok")}

        def same(a, b):
            ta = by.get(a, {})
            tb = by.get(b, {})
            ks = sorted(set(ta) & set(tb))
            return {"n": len(ks), "identical": sum(
                1 for k in ks if ta[k]["content"] == tb[k]["content"]
                and json.dumps(ta[k]["native_tool_calls"], sort_keys=True)
                == json.dumps(tb[k]["native_tool_calls"], sort_keys=True))}
        rend = {}
        for r in rows:
            if r.get("record") == "r2tc_render" and r["model_id"] == model:
                rend.setdefault(r["turn_idx"], {})[r["variant"]] = r["rendered_sha256"]
        rend_pairs = [d for d in rend.values() if d.get("render_on") and d.get("render_tool_choice_none")]
        n_rend_same = sum(1 for d in rend_pairs if d["render_on"] == d["render_tool_choice_none"])

        def tc(v):
            return variants.get(v, {}).get("tool_call_turns")
        native_ok = (bool(rend_pairs) and n_rend_same == len(rend_pairs)
                     and tc("on") is not None and tc("tool_choice_none") is not None
                     and tc("tool_choice_none") >= tc("on"))
        v1_ok = tc("v1_on") is not None and tc("v1_tool_choice_none") is not None and \
            tc("v1_tool_choice_none") >= tc("v1_on")
        out[model] = {
            "variants": variants,
            "identical_on_vs_on_repeat": same("on", "on_repeat"),
            "identical_on_vs_tool_choice_none": same("on", "tool_choice_none"),
            "identical_v1_on_vs_v1_tool_choice_none": same("v1_on", "v1_tool_choice_none"),
            "rendered_prompt_pairs": len(rend_pairs), "rendered_prompt_identical": n_rend_same,
            "verdict_api_chat_tool_choice_none": "no_effect_observed" if native_ok else "effect_observed",
            "verdict_v1_tool_choice_none": "no_effect_observed" if v1_ok else "effect_observed",
            "format_schema_tool_call_turns": tc("format_schema"),
        }
    return out


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


def done_keys_from(rows: list[dict], mode: str | None = None) -> set:
    """(model, arm, seed) of every session completed in rows (optionally only rows of one mode). Used by
    --skip-done-from so the strengthened run never reruns a cell already completed in x2_r2_real_v1.jsonl."""
    return {(s["model_id"], s["arm_id"], s["seed"]) for s in completed_sessions(rows)
            if mode is None or s.get("mode") == mode}


def session_order(plan, models, order="model_major"):
    """[(model, arm, seed, turns)] in run order. model_major: model, then arm, then seed (the original order).
    seed_major: seed position first, then model, then arm, so an interrupted run leaves every (model, tier) cell
    with as many seeds as possible rather than some cells complete and others empty."""
    items = [(m, a, s, t, i, mi, ai) for mi, m in enumerate(models) for ai, (a, seeds, t) in enumerate(plan)
             for i, s in enumerate(seeds)]
    if order == "seed_major":
        items.sort(key=lambda x: (x[4], x[5], x[6]))
    elif order != "model_major":
        raise ValueError(f"unknown order {order!r}")
    return [(m, a, s, t) for m, a, s, t, *_ in items]


def run_plan(runtime, plan, models, mode, out_path: Path, log=print, emit=None, skip_done=(), order="model_major",
             on_session_start=None, on_session_end=None):
    if emit is None:
        def emit(row):
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
    done = {(s["mode"], s["model_id"], s["arm_id"], s["seed"]) for s in completed_sessions(read_rows(out_path))}
    skip_done = set(skip_done)
    for model, arm, seed, turns in session_order(plan, models, order):
        if (mode, model, arm, seed) in done:
            log(f"skip (done): {model} {arm} seed={seed}")
            continue
        if (model, arm, seed) in skip_done:
            log(f"skip (done in --skip-done-from): {model} {arm} seed={seed}")
            continue
        emit({"record": "heartbeat", "mode": mode, "model_id": model, "arm_id": arm, "seed": seed,
              "call2_mode": ARMS[arm]["call2_mode"], "ts_utc": utc_iso()})
        log(f"session: {model} {arm} seed={seed} turns={turns}")
        if on_session_start:
            on_session_start(model, arm, seed)
        s = run_session(runtime, model, arm, seed, turns, emit, log, mode)
        if on_session_end:
            on_session_end()
        log(f"session done: first_failure={s['first_failure_turn']} {s['first_failure_reasons']} "
            f"truncation={s['truncation_detected_turn']} canary_misses="
            f"{s['canary_sys_misses']}+{s['canary_hist_misses']}")


DEFAULT_MODELS = ("llama3.1:8b", "qwen3:14b")   # validation / real / toolchoice_check unless --models or --plan


def _default_models(mode, plan_name):
    if mode == "mechanism":
        return (MECH_MODEL,)
    if mode == "real" and plan_name == "strong":
        return STRONG_MODELS
    return DEFAULT_MODELS


def main(argv=None, advance=True):
    """advance=False: do not call t2s_queue.advance() on exit (a wrapper job such as x2_r2_4b_tools_validate calls
    it once itself); the exit note is returned either way."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("validation", "real", "toolchoice_check", "mechanism"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None,
                    help="comma-separated Ollama tags (default: llama3.1:8b,qwen3:14b; --plan strong: STRONG_MODELS; "
                         "--mode mechanism: MECH_MODEL)")
    ap.add_argument("--plan", choices=("default", "strong"), default="default",
                    help="real mode: default = the v1 plan (3 tiers x 3 seeds); strong = STRONG_TIERS x SEEDS_STRONG "
                         "x 40 turns (2026-10-08)")
    ap.add_argument("--call2-tools", choices=CALL2_MODES, default="on",
                    help="call-2 mode: on (tools, nothing forced), off (tools withheld on call 2, v1/v2), "
                         "forced_none (tools kept, tool_choice none on call 2), forced_none_format (plus an answer "
                         "JSON-schema format on call 2)")
    ap.add_argument("--rules-from", default=None,
                    help="real mode: validation JSONL file(s), comma-separated; per-model rules in use = that "
                         "model's gate-passing rules in the last file holding validation rows for it")
    ap.add_argument("--rules-in-use", default=",".join(RULE_IDS),
                    help="real mode: rules that count toward first-failure / kill criterion (gate-passing rules)")
    ap.add_argument("--require-validation-gates", action="store_true",
                    help="real mode: refuse to start (no Ollama call, queue advanced with a note) unless "
                         "validation_preflight passes on --rules-from")
    ap.add_argument("--per-model-refusal", action="store_true",
                    help="with --require-validation-gates: check each model against its own validation file "
                         "(validation_preflight_per_model), refuse only the models that fail, run the rest; the job "
                         "refuses as a whole only if no model passes")
    ap.add_argument("--skip-done-from", default=None,
                    help="real mode: comma-separated result files; a (model, arm, seed) session completed in any of "
                         "them is not run again (the strengthened run reuses x2_r2_real_v1.jsonl's cells)")
    ap.add_argument("--order", choices=("model_major", "seed_major"), default="model_major")
    args = ap.parse_args(argv)
    if args.require_validation_gates and not args.rules_from:
        ap.error("--require-validation-gates needs --rules-from")
    rule_files = [p for p in (args.rules_from or "").split(",") if p]
    if len(rule_files) > 1 and args.require_validation_gates and not args.per_model_refusal:
        ap.error("several --rules-from files need --per-model-refusal")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{utc_iso()}] {msg}", flush=True)

    def emit(row):
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    note = "completed"
    import host_config as hc
    started = False
    try:
        import socket
        if socket.gethostname().upper() != "EVO-X2":
            raise RuntimeError(f"x2_r2_agent runs on EVO-X2 only, this is {socket.gethostname()!r}")
        models = [m for m in (args.models.split(",") if args.models else _default_models(args.mode, args.plan)) if m]
        models_requested = list(models)
        c2 = args.call2_tools
        if args.mode == "validation":
            plan = validation_plan(c2)
        elif args.mode == "real":
            plan = strong_plan(c2) if args.plan == "strong" else real_plan(c2)
        elif args.mode == "mechanism":
            plan = mechanism_plan(c2)
        else:
            plan = []
        rules_in_use = tuple(args.rules_in_use.split(","))
        preflight = None
        refused_models = {}
        if rule_files:
            vfiles = [(Path(p).name, read_rows(Path(p))) for p in rule_files]
            if args.require_validation_gates and args.per_model_refusal:
                preflight = validation_preflight_per_model(vfiles, c2, models)
                rules_in_use = preflight["rules_in_use"]
                refused_models = preflight["refused"]
                models = list(preflight["models_ok"])
            else:
                merged = {}
                for _name, vrows in vfiles:
                    merged.update(rules_in_use_from(vrows, c2))
                rules_in_use = merged
                if args.require_validation_gates:
                    preflight = validation_preflight(vfiles[-1][1], c2, models)
                    rules_in_use = preflight["rules_in_use"]
        skip_done = set()
        for p in [p for p in (args.skip_done_from or "").split(",") if p]:
            skip_done |= done_keys_from(read_rows(Path(p)), mode="real")
        emit({"record": "run_start", "mode": args.mode, "plan_name": args.plan, "models": models,
              "models_requested": models_requested, "plan": plan, "order": args.order,
              "rules_in_use": rules_in_use, "call2_tools": c2 != "off", "call2_mode": c2,
              "call2_extra": call2_extra(c2), "rules_from": args.rules_from,
              "skip_done_from": args.skip_done_from, "n_skip_done": len(skip_done),
              "validation_preflight": preflight, "ts_utc": utc_iso(),
              "max_tokens_per_call": MAX_TOKENS_PER_CALL, "keep_alive": KEEP_ALIVE})
        for m, reasons in refused_models.items():
            emit({"record": "refused_model", "model_id": m, "reasons": reasons, "call2_mode": c2,
                  "ts_utc": utc_iso()})
            log(f"refused model {m}: {'; '.join(reasons)}")
        if preflight is not None and not preflight["ok"]:
            reasons = (preflight["reasons"] if "reasons" in preflight
                       else [r for rs in preflight["refused"].values() for r in rs])
            emit({"record": "refused", "reasons": reasons, "call2_mode": c2, "ts_utc": utc_iso()})
            note = ("stopped: refused to start, validation gates failed in "
                    f"{','.join(Path(p).name for p in rule_files)}: " + "; ".join(reasons))[:400]
            log(note)
            return note
        server_env = server_log = None
        if args.mode == "mechanism":
            import x2_r2_mechanism as mech
            server_env, server_log = dict(mech.OLLAMA_DEBUG_ENV), mech.MECH_SERVE_LOG
            hc.stop_ollama_server()   # start_ollama_server is a no-op while any Ollama runs; env must take effect
        _start_server(hc, server_env, server_log)
        started = True
        if not hc.wait_for_ollama_ready(timeout_s=60):
            hc.stop_ollama_server()
            _start_server(hc, server_env, server_log)
            if not hc.wait_for_ollama_ready(timeout_s=90):
                raise RuntimeError("ollama did not become ready")
        runtime = OllamaRuntime()
        runtime.server_env, runtime.server_log = server_env, server_log
        avail = runtime.available_models()
        missing = [m for m in models if not any(same_tag(m, a) for a in avail)]
        if missing:
            raise RuntimeError(f"models not present in this Ollama store: {missing} (have {avail})")
        version = runtime.version()
        emit({"record": "runtime", "ollama_version": version, "call2_mode": c2, "server_env": server_env,
              "server_log": server_log, "ts_utc": utc_iso()})
        log(f"x2_r2_agent mode={args.mode} plan={args.plan} call2={c2} models={models} ollama={version} "
            f"out={out_path}")
        if args.mode == "toolchoice_check":
            run_toolchoice_check(runtime, models, emit, log)
            rows = read_rows(out_path)
            report = {"ollama_version": version, "per_model": summarize_toolchoice_check(rows)}
            Path(str(out_path) + ".report.json").write_text(json.dumps(report, indent=1, default=str),
                                                            encoding="utf-8")
            emit({"record": "run_end", "call2_mode": c2, "ts_utc": utc_iso()})
            log(json.dumps(report, default=str))
            return note
        if args.mode == "mechanism":
            head = mech.read_log(server_log, 0)
            if not mech.debug_enabled(head):
                raise RuntimeError(f"OLLAMA_DEBUG not active: no OLLAMA_DEBUG:DEBUG / level=DEBUG in {server_log}")
            mrt = mech.MechanismRuntime(runtime, server_log, emit, out_path.parent / (out_path.stem + "_logs"))
            run_plan(mrt, plan, models, "mechanism", out_path, log=log, emit=emit,
                     on_session_start=lambda m, a, s: mrt.begin_session(m, a, s,
                                                                        call2_mode=ARMS[a]["call2_mode"]),
                     on_session_end=mrt.end_session)
            rep = mech.write_report(out_path, read_rows(out_path))
            emit({"record": "run_end", "call2_mode": c2, "ts_utc": utc_iso()})
            log(json.dumps(rep["verdict"], default=str))
            return note
        run_plan(runtime, plan, models, args.mode, out_path, log=log, emit=emit, skip_done=skip_done,
                 order=args.order)
        rows = read_rows(out_path)
        if args.mode == "validation":
            report = evaluate_gates(rows, call2_tools=c2)
        else:
            sessions = [s for s in completed_sessions(rows) if s["mode"] == "real"]
            report = {"rules_in_use": rules_in_use, "refused_models": refused_models,
                      "kill_criterion": kill_criterion(sessions, rules_in_use, rows)}
        Path(str(out_path) + ".report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        emit({"record": "run_end", "call2_mode": c2, "ts_utc": utc_iso()})
        if args.mode == "validation":
            log("\n" + format_baseline_markdown(report))
            log(json.dumps(report["gates"], default=str))
        if refused_models:
            note = f"completed; refused models (validation gates): {sorted(refused_models)}"
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
        if advance:
            try:
                import t2s_queue as tq
                tq.advance(note)
            except Exception as e:
                log(f"queue advance failed: {e!r}")
    return note


if __name__ == "__main__":
    main()
