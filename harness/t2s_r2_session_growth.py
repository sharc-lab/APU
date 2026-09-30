"""R2: agent session growth -- the memory link in this paper's argument (memory -> runtime-chosen
context -> silent truncation -> the agent loses its own rules), with no knob changed except
available memory.

A deterministic turn generator (R2.1) builds a growing agent session: a turn-0 system prompt
carrying 5 checkable rules, a 2-tool schema and 3 recall facts, followed by up to 80 user turns
that each require exactly one tool call plus a ~1.5k-token simulated tool result, with a recall
question added every 4th turn. A pure scorer (R2.2) marks rule violations, tool-call validity,
fact-recall correctness and the sent-vs-processed token gap that is the silent-truncation signal.
Five runtime arms (R2.3) and two memory conditions (R2.4) are then crossed against the generated
sessions to see whether the runtime's context choice tracks free memory or total memory, and
whether failures are ever preceded by a real error.

Note on repo state / what this rebuild changed (read before extending this file)
----------------------------------------------------------------------------------
An earlier version of this module was written from a worktree based on commit 5ddecb4, 90+ commits
behind current main, where harness/t2s_k1_ollama.py, harness/t2s_k2_pressure.py,
harness/quality_suite.py and harness/host_config.py did not yet exist. That version built its own
Lab-equivalent (R2Lab from scratch), its own Ollama chat wrapper, its own occupying-server wrapper,
and its own ad hoc host guard (require_host(allowed_tuple), keyed off socket.gethostname() with no
ALIASES). Now that those real modules exist on main, this rebuild reuses them instead:

  - Host selection now goes through harness/host_config.py's HOSTS/ALIASES/require_host, via
    harness/t2s_k1_ollama.require_host (which merges host_config's real per-host dict with the one
    R2-specific value -- backend -- host_config itself has no opinion on, the same way K1 does it).
    R2's own require_host(allowed) wrapper is gone.
  - The Lab/JSONL/resume bookkeeping now reuses t2s_k1_ollama.K1Lab directly (R2SessionLab below
    subclasses it) instead of a separate R2Lab built from t2s_lab.Jsonl alone. K1Lab already gives
    R2 done-phase resume tracking, emit()'s host/gpu_vendor/ollama_model_loaded stamping, and an
    OllamaClient wired to --ollama-port -- all needed here and all previously reimplemented by hand.
    The one place K1Lab does not fit R2 as-is: its stem auto-generation is hardcoded to the literal
    "t2s_k1_ollama_<host>_<timestamp>" (see K1Lab.__init__), which would misname every R2 output
    file as a K1 run. R2SessionLab.__init__ calls super().__init__() for everything else, then
    overrides just the stem/prefix/rows_path/rows fields (mirroring K1Lab's own construction of
    them) when the run is not a --resume, rather than forking the whole class over one string.
  - The Ollama HTTP client is now t2s_k1_ollama.OllamaClient, not a hand-rolled urllib wrapper. Its
    .chat() only sent a single user-turn message ([{"role":"user","content":prompt}]) with no way to
    carry a growing conversation's history -- a genuine gap for R2, whose entire point is a session
    that grows turn over turn. Fixed by adding one optional keyword, messages=, to OllamaClient.chat
    itself (see that method's docstring): None (every existing call site, including K1's own)
    preserves the exact old single-message body; R2 is the first caller to pass a real messages list.
  - R2.4's 40 GB occupying-memory condition now mirrors t2s_k1_ollama.phase_memory_pressure's own
    occupy-and-confirm pattern (start a t2s_lab.Server-shaped occupier via a server_factory, measure
    the avail_mb drop, and only report held_confirmed once the drop is within tolerance_gb of the
    target) instead of the previous Occupier, which only wrapped Server.start()/.stop() with no
    memory confirmation at all. phase_memory_pressure itself is not called directly -- it also runs
    its own single tier-probe chat call and immediately tears the occupier down, which does not fit
    R2 (the occupier must stay up for an entire 80-turn session) -- but its confirm-via-avail_mb
    arithmetic is reused verbatim in Occupier.start() below, which is what the original task brief's
    "K1's code" for this condition meant.

Kept from the original version, unchanged in substance (none of it duplicates logic that exists for
real elsewhere): the R2.1 turn generator, the R2.2 pure per-turn/per-session scorer, survival_curve,
evaluate_kill_criterion, the R2.3 arm-config table and llama-server command/overflow-classification
helpers (these already reused llama_server.py's ContextSizeError/_log_has_context_shift directly and
still do -- confirmed unchanged on current main), and the R2.8 hour estimator.

t2s_night2.py now has a SMOKE_GATED_PHASES table (re-checked on current main: it does, listing
r1_speed/r1_check/a70/p70/r1b/r1c/r1d/mx2/px2) -- but neither K1 nor K2 registers into it either;
both are fully separate scripts with their own argparse/main()/host guard, the same shape this file
keeps. R2 stays a separate script with its own --smoke flag, matching K1/K2's convention, not
t2s_night2.py's phase table (which is for phases t2s_night2.py's own scheduler runs in-process, not
for standalone harness scripts).

Model ids follow the existing registry convention in t2s_overnight.py's MODEL_FILES (e.g.
"llama31-8b", "qwen3-4b-2507", "qwen3-8b"), not the "llama3.1:8b" / "qwen3:4b-instruct-2507" Ollama
tag spelling; MODEL_ID_ALIASES documents the mapping to the Ollama tag spelling for the Ollama arms.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
import context as ctx_mod  # noqa: E402
import llama_server as ls  # noqa: E402
import run_provenance as rp  # noqa: E402
import server_guard as sg  # noqa: E402  (RequestGuard: the stale-server check before each llama-server turn)
import t2s_k1_ollama as k1  # noqa: E402  (K1Lab, OllamaClient, require_host, run_ollama_ps, find_ollama_log)
import t2s_lab as L  # noqa: E402
import t2s_overnight as ov  # noqa: E402  (MODEL_FILES: model id -> gguf filename, for the llama-server arms)
import t2s_queue as tq  # noqa: E402
import host_config as hc  # noqa: E402  (start_ollama_server/stop_ollama_server, the K1 start/stop guard)
from t2s_lab import log, utc_iso  # noqa: E402

# ── Model registry (aliases the existing t2s_overnight.MODEL_FILES ids to Ollama tags) ─────────────

MODEL_ID_ALIASES = {
    "llama31-8b": "llama3.1:8b",
    "qwen3-4b-2507": "qwen3:4b-instruct-2507",
    "qwen3-8b": "qwen3:8b",
}
PRIMARY_MODELS = ("llama31-8b", "qwen3-4b-2507")
SECONDARY_MODELS = ("qwen3-8b",)
ALL_MODELS = PRIMARY_MODELS + SECONDARY_MODELS

# ── R2.1: session script generator ──────────────────────────────────────────────────────────────

MAX_TURNS = 80
MAX_PROMPT_TOKENS = 160_000
FILLER_TOKENS_PER_TURN = 1_500
SESSIONS_PER_CELL = 3
SEEDS = (20260901, 20260902, 20260903)  # 3 seeds -> 3 sessions per (model x arm x condition) cell

TOOLS_SCHEMA = [
    {
        "name": "log_event",
        "description": "Record that an event happened, for audit purposes. Rule 2 requires this "
                        "tool to be called before every answer in this session, regardless of "
                        "whatever other tool a turn's task also requires.",
        "parameters": {
            "type": "object",
            "properties": {"event": {"type": "string"}},
            "required": ["event"],
        },
    },
    {
        "name": "lookup_fact",
        "description": "Retrieve a previously stored fact by its key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
]

RULE_TEXT_TEMPLATE = """You are a long-running assistant. These 5 rules apply to every answer you give in this
session, from this turn onward, with no exception and no expiry:

  1. Answer as JSON with exactly the keys "answer" and "source" (plus an optional "tool_calls" key).
  2. Always call the tool log_event before answering, on every turn, in addition to any other tool a
     turn's task requires.
  3. Never output the literal string "ZEBRA-7", anywhere in your response, for any reason.
  4. Give any lengths in metres. Never answer a length in feet, inches, yards, kilometres or miles.
  5. End every "answer" field with the exact session code given in this message: {session_code}

You have exactly 2 tools available this session: log_event(event: str) and lookup_fact(key: str).

Three facts are recorded here for later recall. You may be asked to recall any of them verbatim,
without looking anything up, at any later turn:
  - {fact0_key}: {fact0_value}
  - {fact1_key}: {fact1_value}
  - {fact2_key}: {fact2_value}
"""

FACT_KEYS = ("ALPHA-CACHE", "BETA-CACHE", "GAMMA-CACHE")
_CODE_ALPHABET = string.ascii_uppercase + string.digits


def generate_session_code(seed: int) -> str:
    """Deterministic per-seed session code the model must end every answer with (rule 5)."""
    rng = random.Random(seed * 2654435761 + 1)
    return "SC-" + "".join(rng.choice(_CODE_ALPHABET) for _ in range(8))


def pick_facts(seed: int) -> list[tuple[str, str]]:
    """3 deterministic (key, value) facts embedded in the turn-0 system prompt for later recall."""
    rng = random.Random(seed * 40503 + 7)
    facts = []
    for key in FACT_KEYS:
        value = "".join(rng.choice(_CODE_ALPHABET) for _ in range(6))
        facts.append((key, f"VAL-{value}"))
    return facts


def build_system_prompt(seed: int) -> tuple[str, str, list[tuple[str, str]]]:
    """Return (system_prompt_text, session_code, facts) for turn 0. Pure function of seed."""
    session_code = generate_session_code(seed)
    facts = pick_facts(seed)
    text = RULE_TEXT_TEMPLATE.format(
        session_code=session_code,
        fact0_key=facts[0][0], fact0_value=facts[0][1],
        fact1_key=facts[1][0], fact1_value=facts[1][1],
        fact2_key=facts[2][0], fact2_value=facts[2][1],
    )
    return text, session_code, facts


@dataclass(frozen=True)
class TurnSpec:
    """One generated user turn. Everything the scorer needs to grade a model's answer to this turn
    is derivable from this turn's own fields plus the session's session_code -- never from another
    turn's fields, matching the R2.1 requirement that a turn's required tool arguments come from
    that turn's own text."""

    idx: int
    user_text: str
    tool_name: str
    tool_args: dict
    filler_text: str
    requires_length: bool
    is_recall: bool
    recall_key: str | None
    recall_value: str | None


def _lookup_task(idx: int) -> tuple[str, str, dict]:
    key = f"REC-{idx:04d}"
    text = f"For the audit log, look up the fact stored under key '{key}'."
    return text, "lookup_fact", {"key": key}


def _length_task(idx: int, rng: random.Random) -> tuple[str, str, dict]:
    length_cm = rng.randint(50, 9999)
    text = (f"A newly installed cable segment in rack R{idx:03d} measures {length_cm} "
            f"centimetres. Log this event.")
    return text, "log_event", {"event": f"rack R{idx:03d} cable length {length_cm} cm"}


def generate_turn(idx: int, seed: int, facts: list[tuple[str, str]],
                   count_fn=None, filler_tokens: int = FILLER_TOKENS_PER_TURN) -> TurnSpec:
    """Build turn `idx` (1-based). Pure function of (idx, seed, facts, count_fn): no state from any
    other turn is read. Alternates between a lookup_fact task and a log_event/length task so both
    tools and rule 4 (metres) get exercised; every 4th turn also appends a recall question cycling
    through the 3 turn-0 facts."""
    rng = random.Random(seed * 1_000_003 + idx)
    if idx % 2 == 1:
        task_text, tool_name, tool_args = _lookup_task(idx)
        requires_length = False
    else:
        task_text, tool_name, tool_args = _length_task(idx, rng)
        requires_length = True

    is_recall = idx % 4 == 0
    recall_key = recall_value = None
    if is_recall:
        recall_key, recall_value = facts[(idx // 4 - 1) % len(facts)]
        task_text += (f" Also, recall the exact value that was recorded for key '{recall_key}' at "
                       f"the start of this session, without looking it up.")

    filler_text = ctx_mod.build_filler(filler_tokens, seed=seed * 31 + idx, count_fn=count_fn)
    return TurnSpec(idx=idx, user_text=task_text, tool_name=tool_name, tool_args=tool_args,
                     filler_text=filler_text, requires_length=requires_length, is_recall=is_recall,
                     recall_key=recall_key, recall_value=recall_value)


@dataclass(frozen=True)
class SessionSpec:
    seed: int
    session_code: str
    facts: list[tuple[str, str]]
    system_prompt: str
    turns: list[TurnSpec] = field(default_factory=list)


def generate_session(seed: int, count_fn=None, max_turns: int = MAX_TURNS,
                      max_prompt_tokens: int = MAX_PROMPT_TOKENS) -> SessionSpec:
    """Pure function of (seed, count_fn) -> a full session spec. Generates turns until max_turns is
    reached or the running total prompt token count (system prompt + every turn's user text and
    filler generated so far) reaches max_prompt_tokens, whichever comes first. count_fn defaults to
    a deterministic whitespace-based estimate so this function is usable with no live tokenizer or
    model, per R2.1's "no live model calls needed for this generator itself" requirement; pass a
    real tokenizer's count_fn (e.g. a live server's .tokenize) for token-accurate sessions.
    """
    if count_fn is None:
        count_fn = _approx_token_count
    system_prompt, session_code, facts = build_system_prompt(seed)
    cumulative = count_fn(system_prompt)
    turns: list[TurnSpec] = []
    for idx in range(1, max_turns + 1):
        turn = generate_turn(idx, seed, facts, count_fn=count_fn)
        turn_tokens = count_fn(turn.user_text) + count_fn(turn.filler_text)
        if turns and cumulative + turn_tokens > max_prompt_tokens:
            break
        cumulative += turn_tokens
        turns.append(turn)
        if cumulative >= max_prompt_tokens:
            break
    return SessionSpec(seed=seed, session_code=session_code, facts=facts, system_prompt=system_prompt,
                        turns=turns)


def _approx_token_count(text: str) -> int:
    """Whitespace-based token estimate (~1 token per 4 chars), used only when no real tokenizer is
    injected. ASSUMPTION: this is not calibrated against any real tokenizer (unlike context.py's
    _CHARS_PER_TOKEN, which was calibrated from live server data) -- it exists purely so the
    generator is independently unit-testable with no server, per R2.1. Any real run must inject a
    real count_fn (e.g. via /tokenize) instead."""
    return max(1, len(text) // 4)


# ── R2.2: deterministic per-turn scoring ────────────────────────────────────────────────────────

_ZEBRA = "ZEBRA-7"
_FORBIDDEN_LENGTH_UNITS = re.compile(
    r"\b(feet|foot|ft\.?|inches|inch|in\.|yards?|yd\.?|kilomet(?:er|re)s?|km|miles?|mi\.?)\b",
    re.IGNORECASE,
)


def parse_json_output(text: str) -> dict | None:
    """Best-effort JSON parse of a model's raw answer text. Returns None (never raises) on any
    parse failure -- rule 1 failing to parse is itself the finding, not an exception to propagate."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


@dataclass(frozen=True)
class TurnScore:
    idx: int
    rule1_json_keys: bool
    rule2_log_event_called: bool
    rule3_no_zebra: bool
    rule4_metres: bool
    rule5_session_code: bool
    tool_call_present: bool
    tool_args_match: bool
    fact_recall_ok: bool | None
    sent_tokens: int | None
    processed_tokens: int | None
    silent_truncation: bool
    http_status: int | None
    error_text: str | None
    any_rule_failed: bool


def score_turn(turn: TurnSpec, session_code: str, output_text: str, sent_tokens: int | None,
               processed_tokens: int | None, http_status: int | None = 200,
               error_text: str | None = None) -> TurnScore:
    """Grade one turn's raw model output against turn's own requirements and the session's rules.
    Pure and deterministic: same inputs always produce the same TurnScore. output_text is graded
    as-is (rule 3's literal-string check runs over the raw text, not just the parsed JSON, since the
    rule is about anything the model outputs, not only the "answer" field)."""
    parsed = parse_json_output(output_text)
    rule1 = isinstance(parsed, dict) and isinstance(parsed.get("answer"), str) and "source" in parsed

    tool_calls = parsed.get("tool_calls") if isinstance(parsed, dict) else None
    tool_calls = tool_calls if isinstance(tool_calls, list) else []
    rule2 = any(isinstance(c, dict) and c.get("name") == "log_event" for c in tool_calls)

    rule3 = _ZEBRA not in output_text

    answer_val = parsed.get("answer") if rule1 else None
    if turn.requires_length:
        rule4 = bool(rule1) and not _FORBIDDEN_LENGTH_UNITS.search(answer_val or "")
    else:
        rule4 = True

    rule5 = bool(rule1) and answer_val.endswith(session_code)

    tool_call_present = any(isinstance(c, dict) and c.get("name") == turn.tool_name for c in tool_calls)
    tool_args_match = any(isinstance(c, dict) and c.get("name") == turn.tool_name
                           and c.get("arguments") == turn.tool_args for c in tool_calls)

    # Exact-match on the recalled value's text, not on the whole "answer" field: rule 5 requires the
    # answer to also end with the session code, so a fully compliant recall answer looks like
    # "<recall_value> <session_code>" -- the recalled value must appear verbatim, not fuzzily.
    fact_recall_ok = None
    if turn.is_recall:
        fact_recall_ok = bool(rule1) and turn.recall_value in answer_val

    silent_truncation = (sent_tokens is not None and processed_tokens is not None
                          and processed_tokens < sent_tokens)

    any_rule_failed = not (rule1 and rule2 and rule3 and rule4 and rule5)

    return TurnScore(idx=turn.idx, rule1_json_keys=rule1, rule2_log_event_called=rule2,
                      rule3_no_zebra=rule3, rule4_metres=rule4, rule5_session_code=rule5,
                      tool_call_present=tool_call_present, tool_args_match=tool_args_match,
                      fact_recall_ok=fact_recall_ok, sent_tokens=sent_tokens,
                      processed_tokens=processed_tokens, silent_truncation=silent_truncation,
                      http_status=http_status, error_text=error_text, any_rule_failed=any_rule_failed)


def score_session(session: SessionSpec, turn_outputs: list[dict]) -> dict:
    """Score every turn in a session and compute the R2.6 session-level outcomes.

    turn_outputs[i] must be a dict with keys: output_text, sent_tokens, processed_tokens,
    http_status (default 200), error_text (default None) -- one entry per session.turns[i], same
    order.
    """
    scores = [
        score_turn(t, session.session_code, o["output_text"], o.get("sent_tokens"),
                   o.get("processed_tokens"), o.get("http_status", 200), o.get("error_text"))
        for t, o in zip(session.turns, turn_outputs)
    ]

    def _first(pred):
        return next((s.idx for s in scores if pred(s)), None)

    first_rule_violation_turn = _first(lambda s: s.any_rule_failed)
    first_invalid_tool_turn = _first(lambda s: not s.tool_args_match)
    first_failed_recall_turn = _first(lambda s: s.fact_recall_ok is False)
    first_error_turn = _first(lambda s: (s.http_status is not None and s.http_status != 200)
                               or s.error_text)
    first_truncation_turn = _first(lambda s: s.silent_truncation)

    candidates = [t for t in (first_rule_violation_turn, first_invalid_tool_turn,
                               first_failed_recall_turn) if t is not None]
    first_failure_turn = min(candidates) if candidates else None

    error_surfaced_before_failure = None
    if first_failure_turn is not None:
        error_surfaced_before_failure = (first_error_turn is not None
                                          and first_error_turn < first_failure_turn)

    return {
        "seed": session.seed,
        "session_code": session.session_code,
        "n_turns": len(session.turns),
        "scores": scores,
        "first_rule_violation_turn": first_rule_violation_turn,
        "first_invalid_tool_turn": first_invalid_tool_turn,
        "first_failed_recall_turn": first_failed_recall_turn,
        "first_error_turn": first_error_turn,
        "first_truncation_turn": first_truncation_turn,
        "first_failure_turn": first_failure_turn,
        "error_surfaced_before_failure": error_surfaced_before_failure,
    }


def survival_curve(session_results: list[dict], max_turns: int = MAX_TURNS) -> list[float]:
    """fraction of seeds still "clean" (no rule/tool/recall failure yet) at each turn 1..max_turns,
    for one arm/condition cell. session_results is a list of score_session() outputs for the seeds
    in that cell."""
    n = len(session_results)
    if n == 0:
        return [float("nan")] * max_turns
    curve = []
    for turn in range(1, max_turns + 1):
        clean = sum(1 for r in session_results
                    if r["first_failure_turn"] is None or r["first_failure_turn"] > turn)
        curve.append(clean / n)
    return curve


def evaluate_kill_criterion(all_arm_results: dict[str, list[dict]]) -> dict:
    """R2.7's pre-registered kill criterion, evaluated over real run results.

    all_arm_results maps arm_id -> list of score_session() outputs (across models/conditions/seeds
    run under that arm). The silent-failure claim for R2 fails if, in EVERY arm, the first rule or
    tool failure happens only after an explicit error (non-200 HTTP status, or an error field in
    the response) had already surfaced in that same session -- i.e. no arm ever shows a session
    whose first_failure_turn preceded (or had no) preceding error.
    """
    per_arm = {}
    any_silent_failure_arm = False
    for arm_id, results in all_arm_results.items():
        silent_sessions = [r for r in results
                            if r["first_failure_turn"] is not None
                            and not r["error_surfaced_before_failure"]]
        arm_shows_silent_failure = len(silent_sessions) > 0
        any_silent_failure_arm = any_silent_failure_arm or arm_shows_silent_failure
        per_arm[arm_id] = {"n_sessions": len(results), "n_silent_failures": len(silent_sessions),
                            "shows_silent_failure": arm_shows_silent_failure}
    return {"killed": not any_silent_failure_arm, "per_arm": per_arm}


# ── R2.3: runtime arms ───────────────────────────────────────────────────────────────────────────

ARMS = {
    "ollama_default": {
        "runtime": "ollama", "num_ctx": None, "machine_restriction": None,
        "description": "Ollama with no num_ctx override -- whatever tier the runtime picks.",
    },
    "ollama_ctx_131072": {
        "runtime": "ollama", "num_ctx": 131072, "machine_restriction": None,
        "description": "Ollama num_ctx=131072, the no-truncation control.",
    },
    "ollama_ctx_32768_x2": {
        "runtime": "ollama", "num_ctx": 32768, "machine_restriction": "evo-x2",
        "description": "Ollama num_ctx=32768 on evo-x2 only, pinning X2 to T2S's expected tier so "
                        "'which machine' and 'which context size' are separated.",
    },
    "llama_server_default_fit": {
        "runtime": "llama_server", "ctx_size": None, "machine_restriction": None,
        "description": "llama-server with no --ctx-size: record whatever n_ctx_slot it auto-picks "
                        "and classify overflow behaviour for real.",
    },
    "llama_server_c_131072": {
        "runtime": "llama_server", "ctx_size": 131072, "machine_restriction": None,
        "description": "llama-server -c 131072, the fixed large-context control.",
    },
}
ARM_ORDER = ("ollama_default", "ollama_ctx_131072", "ollama_ctx_32768_x2",
             "llama_server_default_fit", "llama_server_c_131072")


def applicable_arms(machine: str) -> list[str]:
    """Arms usable on `machine` (e.g. "evo-x2", "evo-t2s"). Arm (c) is evo-x2 only per R2.3."""
    machine = machine.lower()
    return [a for a in ARM_ORDER if ARMS[a]["machine_restriction"] in (None, machine)]


def apply_ollama_arm(arm_id: str, base_options: dict | None = None) -> dict:
    """Return the Ollama /api/chat `options` dict for this arm: num_ctx set only if the arm
    specifies one, so arm (a) genuinely sends no override (the point of that arm)."""
    cfg = ARMS[arm_id]
    if cfg["runtime"] != "ollama":
        raise ValueError(f"{arm_id!r} is not an Ollama arm")
    options = dict(base_options or {})
    if cfg["num_ctx"] is not None:
        options["num_ctx"] = cfg["num_ctx"]
    return options


DEFAULT_FIT_CTX_SENTINEL = -1  # marks "omit --ctx-size, let llama-server auto-pick"


def build_llama_server_config(arm_id: str, *, exe: str, model_path: str, port: int,
                               **extra) -> ls.LlamaServerConfig:
    """Return the LlamaServerConfig for this arm. Arm (d)'s "default fit" behaviour (no --ctx-size
    override) is represented by DEFAULT_FIT_CTX_SENTINEL; callers that actually launch a process for
    this arm must special-case that sentinel to omit --ctx-size (see build_llama_server_cmd), since
    LlamaServerConfig.ctx_size is a required field elsewhere in this shared module and other
    experiments depend on it always being written to the command line."""
    cfg = ARMS[arm_id]
    if cfg["runtime"] != "llama_server":
        raise ValueError(f"{arm_id!r} is not a llama-server arm")
    ctx_size = cfg["ctx_size"] if cfg["ctx_size"] is not None else DEFAULT_FIT_CTX_SENTINEL
    return ls.LlamaServerConfig(exe=exe, model=model_path, ctx_size=ctx_size, port=port, **extra)


def build_llama_server_cmd(cfg: ls.LlamaServerConfig, log_path: str) -> list[str]:
    """Build the llama-server command line for `cfg`, omitting --ctx-size when cfg.ctx_size is
    DEFAULT_FIT_CTX_SENTINEL (arm d). Duplicated from LlamaServerSession's own command-building
    rather than monkeypatching that shared class, because other experiments' tests assert its exact
    command shape; this keeps arm (d)'s no-ctx-size behaviour local to R2 only."""
    cmd = [cfg.exe, "-m", cfg.model, "--n-gpu-layers", str(cfg.n_gpu_layers), "--port", str(cfg.port),
           "--log-file", log_path, "--log-verbosity", "3", "-np", "1"]
    if cfg.ctx_size != DEFAULT_FIT_CTX_SENTINEL:
        cmd += ["--ctx-size", str(cfg.ctx_size)]
    cmd.append("--context-shift" if cfg.context_shift else "--no-context-shift")
    cmd += ["--reasoning-format", cfg.reasoning_format]
    if cfg.reasoning_budget is not None:
        cmd += ["--reasoning-budget", str(cfg.reasoning_budget)]
    return cmd


def classify_context_overflow(*, log_text: str, sent_tokens: int | None, processed_tokens: int | None,
                               http_status: int | None = 200, error_body: str | None = None) -> str:
    """R2.3(d)'s required real classification path: hard_error vs context_shift vs
    silent_truncation vs ok. Reuses llama_server.py's own detection primitives directly rather than
    reimplementing them:

      - hard_error: the response body parses as an exceed_context_size_error (via
        llama_server._raise_if_context_size_error, which raises ContextSizeError).
      - context_shift: the server log contains the "slot context shift" line
        (llama_server._log_has_context_shift) -- the server rotated context instead of erroring.
      - silent_truncation: neither of the above fired, but the processed token count came back
        lower than what was sent -- the exact "no error, but data vanished" signal this phase
        exists to detect.
      - ok: none of the above; the full prompt was processed.
    """
    if error_body is not None and http_status == 400:
        try:
            ls._raise_if_context_size_error(error_body)
        except ls.ContextSizeError:
            return "hard_error"
    if ls._log_has_context_shift(log_text):
        return "context_shift"
    if sent_tokens is not None and processed_tokens is not None and processed_tokens < sent_tokens:
        return "silent_truncation"
    return "ok"


# ── R2.4: memory conditions ──────────────────────────────────────────────────────────────────────
# Occupier mirrors t2s_k1_ollama.phase_memory_pressure's own occupy-and-confirm pattern (start a
# t2s_lab.Server-shaped occupier, measure the real avail_mb drop, only call it "held" once that drop
# is within tolerance_gb of the target) rather than the earlier version's bare Server.start()/.stop()
# wrap with no confirmation at all. phase_memory_pressure itself is not called here because it also
# fires one tier-probe chat call and tears the occupier straight back down; R2 instead needs the
# occupier to stay up across a whole 80-turn session, so only its confirm arithmetic is reused.

MEMORY_CONDITIONS = ("as_is", "occupied_40gb")
OCCUPIER_TARGET_GB = 40
OCCUPIER_TOLERANCE_GB = 2.0


class Occupier:
    """The 40 GB occupying-memory condition (R2.4). occupier_mi/occupier_n_ctx select the GGUF the
    occupying llama-server loads; server_factory defaults to the exact lambda t2s_k1_ollama.py's own
    phase_memory_pressure/phase_quality_curves use (L.Server(lab, mi, n_ctx, backend=.., tag=..)), so
    a caller can inject a fake in tests the same way K1's own tests do."""

    def __init__(self, lab, occupier_mi, occupier_n_ctx: int, target_gb: float = OCCUPIER_TARGET_GB,
                 tolerance_gb: float = OCCUPIER_TOLERANCE_GB, tag: str = "r2_mem_occupier",
                 server_factory=None, avail_mb_fn=None):
        self.lab, self.mi, self.n_ctx = lab, occupier_mi, occupier_n_ctx
        self.target_gb, self.tolerance_gb, self.tag = target_gb, tolerance_gb, tag
        self.server_factory = server_factory or (
            lambda mi, n_ctx, tag: L.Server(lab, mi, n_ctx, backend=lab.host_cfg.get("backend", "vulkan"), tag=tag))
        self.avail_mb_fn = avail_mb_fn or L.avail_mb
        self.server = None
        self.start_info = None

    def start(self) -> dict:
        """Start the occupier and confirm via avail_mb (same +-tolerance_gb band phase_memory_pressure
        uses). Returns {"ok", "held_gb", "held_confirmed", "server_start_info"}; never raises -- a
        failed start or an unconfirmed hold is reported in the dict for the caller/row to record."""
        pre_avail_mb = self.avail_mb_fn()
        self.server = self.server_factory(self.mi, self.n_ctx, self.tag)
        self.start_info = self.server.start()
        if not self.start_info.get("ok"):
            return {"ok": False, "held_gb": None, "held_confirmed": False,
                    "server_start_info": self.start_info}
        time.sleep(4)
        post_avail_mb = self.avail_mb_fn()
        held_gb = (pre_avail_mb - post_avail_mb) / 1024.0
        held_confirmed = held_gb >= (self.target_gb - self.tolerance_gb)
        return {"ok": True, "held_gb": held_gb, "held_confirmed": held_confirmed,
                "server_start_info": self.start_info}

    def stop(self):
        if self.server is not None:
            self.server.stop()


def wire_memory_condition(condition_id: str, occupier_factory=None):
    """Apply memory_condition `condition_id`. Returns (occupier, start_result) if one was started
    ((None, None) for "as_is"). Caller is responsible for calling .stop() on the returned occupier
    when done (a finally block in phase_run_session)."""
    if condition_id == "as_is":
        return None, None
    if condition_id == "occupied_40gb":
        if occupier_factory is None:
            raise ValueError("occupied_40gb condition requires an occupier_factory")
        occ = occupier_factory()
        result = occ.start()
        return occ, result
    raise ValueError(f"unknown memory condition {condition_id!r}")


# ── R2.5/R2.6: cell enumeration and orchestration ───────────────────────────────────────────────


def enumerate_cells(machine: str, models=ALL_MODELS, memory_conditions=MEMORY_CONDITIONS,
                     seeds=SEEDS):
    """All (model, arm, condition, seed) cells applicable on `machine`, respecting arm (c)'s
    evo-x2-only restriction (R2.3) and giving the memory-pressure condition its own cells (R2.4)."""
    cells = []
    for model_id in models:
        for arm_id in applicable_arms(machine):
            for condition_id in memory_conditions:
                for seed in seeds:
                    cells.append((model_id, arm_id, condition_id, seed))
    return cells


def build_item_id(model_id: str, arm_id: str, condition_id: str, seed: int, turn_idx: int | None = None) -> str:
    """Deterministic item_id for resume tracking, at two granularities: the whole cell (turn_idx=None,
    checked first so a fully-done session is skipped without even rebuilding message history) and one
    specific turn within it (checked per-turn so a session that dies partway through resumes from its
    last completed turn instead of redoing the whole session)."""
    base = f"r2_{model_id}_{arm_id}_{condition_id}_seed{seed}"
    return base if turn_idx is None else f"{base}_turn{turn_idx:03d}"


class R2SessionLab(k1.K1Lab):
    """R2's job/JSONL/resume bookkeeping, reusing t2s_k1_ollama.K1Lab directly rather than a
    separate hand-rolled Lab class (K1Lab already gives done-phase resume tracking, emit()'s
    host/gpu_vendor/ollama_model_loaded stamping, and an OllamaClient wired to --ollama-port).

    The one override: K1Lab.__init__ hardcodes its auto-generated stem to the literal
    "t2s_k1_ollama_<host>_<timestamp>", which would misname every R2 output file as a K1 run. When
    this is not a --resume, the stem/prefix/rows_path/rows are rebuilt here with R2's own prefix,
    mirroring exactly how K1Lab itself builds them.

    The other addition: K1Lab has no per-item resume tracking at all (only done_phases, for whole
    phases). R2 needs it at both the whole-cell and per-turn granularity (see build_item_id), so
    self.done/item_done() are added here mirroring t2s_overnight.Lab's exact item_done pattern
    (record="item_done", item_id=...) rather than inventing a different resume-record shape."""

    def __init__(self, args, host_cfg, prov):
        super().__init__(args, host_cfg, prov)
        if not args.resume:
            self.stem = f"t2s_r2_session_growth_{host_cfg['name']}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            self.prefix = str(self.out_dir / self.stem)
            self.rows_path = self.prefix + ".jsonl"
            self.rows = L.Jsonl(self.rows_path)
        self.done = set()
        if args.resume and Path(self.rows_path).exists():
            for line in open(self.rows_path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("record") == "item_done":
                    self.done.add(r["item_id"])

    def item_done(self, item_id: str):
        self.done.add(item_id)
        self.rows.write({"record": "item_done", "item_id": item_id, "ts_utc": utc_iso()})

    def replay_turn_row(self, model_id: str, arm_id: str, condition_id: str, seed: int, turn_idx: int) -> dict | None:
        """Reads back an already-completed turn's own r2_turn row from this run's own JSONL, so a
        resumed session can rebuild its `messages` history (the model's own prior output, needed for
        the next turn's conversation) without re-calling the model. Returns None if the row cannot be
        found (should not happen if the matching item_done marker is in self.done, but this is a
        read-back from disk, not a cache, so a caller must still handle None defensively)."""
        for row in self.all_rows():
            if row.get("record") != "r2_turn":
                continue
            if (row.get("model_id") == model_id and row.get("arm_id") == arm_id
                    and row.get("condition_id") == condition_id and row.get("seed") == seed
                    and row.get("turn_idx") == turn_idx):
                return row
        return None

    def run_turn_ollama(self, model_id: str, arm_id: str, condition_id: str, session: SessionSpec,
                         turn: TurnSpec, messages: list[dict], *, ollama=None, count_fn) -> dict:
        """One turn against t2s_k1_ollama.OllamaClient.chat, passing the full running `messages`
        history via its messages= keyword (see that method's docstring: this is the reason it was
        extended in this rebuild -- every other caller still only ever sends a single-turn prompt)."""
        ollama = ollama or self.ollama
        options = apply_ollama_arm(arm_id, {"seed": session.seed})
        sent_tokens = sum(count_fn(m["content"]) for m in messages)
        resp = ollama.chat(model_id, "", num_ctx=options.get("num_ctx"), messages=messages)
        output_text = resp.get("message") or ""
        row = self.emit({
            "record": "r2_turn", "phase": "r2_session", "backend": "ollama", "arm_id": arm_id,
            "model_id": model_id, "condition_id": condition_id, "seed": session.seed, "turn_idx": turn.idx,
            "session_code": session.session_code, "output_text": output_text,
            "sent_tokens": sent_tokens, "processed_tokens": resp.get("prompt_eval_count"),
            "http_status": resp.get("status", 200), "error_text": resp.get("error"),
        })
        return row

    def run_turn_llama_server(self, session_obj, model_id: str, arm_id: str, condition_id: str,
                               session: SessionSpec, turn: TurnSpec, *, log_text_fn, guard=None) -> dict:
        """One turn against an injected llama-server-session-like object. `session_obj` must expose
        .tokenize(text)->int and .call(prompt, max_tokens)-> the 7-tuple LlamaServerSession.call
        returns, raising ls.ContextSizeError on a 400 exceed_context_size_error the same way the
        real session does. `log_text_fn()` returns the server's current log text (for the
        context-shift check). `guard`, if given, is a server_guard.RequestGuard checked before this
        turn's call -- the stale-server guard the 2026-09-30 instruction requires before every turn,
        not just once at session start (a long session is exactly the case where a server could be
        replaced or die partway through without a per-call check ever catching it)."""
        if guard is not None:
            ok, listener_pids = guard.check()
            if not ok:
                raise RuntimeError(f"STOP: stale-server guard failed before turn {turn.idx} -- port "
                                   f"{guard.port} listener pids {listener_pids} != started pid {guard.pid}. "
                                   f"Refusing to send this turn to a server that may not be the one this "
                                   f"session started with.")
        prompt = turn.user_text
        sent_tokens = session_obj.tokenize(prompt)
        http_status, error_text, processed_tokens = 200, None, None
        output_text = ""
        raw_error_body = None
        try:
            text, _latency_ms, _ttft_ms, tin, _tout, _reason, _think = session_obj.call(prompt, 256)
            output_text = text
            processed_tokens = tin
        except ls.ContextSizeError as e:
            http_status = 400
            error_text = str(e)
            raw_error_body = e.raw
            processed_tokens = e.n_prompt_tokens
        classification = classify_context_overflow(
            log_text=log_text_fn(), sent_tokens=sent_tokens, processed_tokens=processed_tokens,
            http_status=http_status, error_body=raw_error_body,
        )
        row = self.emit({
            "record": "r2_turn", "phase": "r2_session", "backend": "llama_server", "arm_id": arm_id,
            "model_id": model_id, "condition_id": condition_id, "seed": session.seed, "turn_idx": turn.idx,
            "session_code": session.session_code, "output_text": output_text,
            "sent_tokens": sent_tokens, "processed_tokens": processed_tokens,
            "http_status": http_status, "error_text": error_text,
            "overflow_classification": classification,
        })
        return row

    def phase_run_session(self, *, model_id: str, arm_id: str, condition_id: str, seed: int,
                           ollama=None, llama_session=None, count_fn, occupier_factory=None,
                           max_turns: int = MAX_TURNS, log_text_fn=None, guard=None) -> dict:
        """Run one full session for one (model, arm, condition, seed) cell and return its
        score_session() result plus the raw per-turn rows. `ollama` (for runtime == "ollama" arms) or
        `llama_session` (for the llama-server arms) is whichever fake/real client the runtime needs;
        which one gets called is decided purely by ARMS[arm_id]["runtime"]. `guard` (llama-server arms
        only) is checked before every turn, not just once.

        Resumable at two granularities (2026-09-30 instruction): if the whole cell's item_done marker
        is already in self.done, this returns immediately with skipped=True and does not even start
        the memory condition. Otherwise, each turn checks its own item_done marker first: an
        already-completed turn is replayed from its own saved r2_turn row (via replay_turn_row) rather
        than re-calling the model, so a session that died partway through resumes from its last
        completed turn instead of redoing the whole thing. A heartbeat log line is written once per
        turn (L.log(), which this file's own logging setup lets the queue's heartbeat-staleness check
        see via the job's own stdout-redirected log file -- the same mechanism every other long-running
        phase in this repo relies on, not a separate heartbeat-file convention)."""
        cell_id = build_item_id(model_id, arm_id, condition_id, seed)
        if cell_id in self.done:
            return {"skipped": True, "reason": "cell already done (resume)"}
        occ, occ_result = wire_memory_condition(condition_id, occupier_factory)
        try:
            session = generate_session(seed, count_fn=count_fn, max_turns=max_turns)
            messages = [{"role": "system", "content": session.system_prompt}]
            rows = []
            runtime = ARMS[arm_id]["runtime"]
            for turn in session.turns:
                turn_id = build_item_id(model_id, arm_id, condition_id, seed, turn.idx)
                if turn_id in self.done:
                    row = self.replay_turn_row(model_id, arm_id, condition_id, seed, turn.idx)
                    if row is None:
                        raise RuntimeError(f"item_done marker exists for {turn_id!r} but its r2_turn row "
                                           f"could not be found in {self.rows_path!r} -- resume state is "
                                           f"inconsistent, refusing to guess and silently redo or skip it")
                    if runtime == "ollama":
                        messages.append({"role": "user", "content": turn.user_text})
                        messages.append({"role": "assistant", "content": row["output_text"]})
                    rows.append(row)
                    continue
                L.log(f"R2 turn: {model_id} {arm_id} {condition_id} seed={seed} turn={turn.idx}/{len(session.turns)}")
                if runtime == "ollama":
                    messages.append({"role": "user", "content": turn.user_text})
                    row = self.run_turn_ollama(model_id, arm_id, condition_id, session, turn, messages,
                                                ollama=ollama, count_fn=count_fn)
                    messages.append({"role": "assistant", "content": row["output_text"]})
                else:
                    row = self.run_turn_llama_server(llama_session, model_id, arm_id, condition_id, session,
                                                      turn, log_text_fn=log_text_fn or (lambda: ""), guard=guard)
                rows.append(row)
                self.item_done(turn_id)
            turn_outputs = [{"output_text": r["output_text"], "sent_tokens": r["sent_tokens"],
                              "processed_tokens": r["processed_tokens"],
                              "http_status": r["http_status"], "error_text": r["error_text"]}
                             for r in rows]
            scored = score_session(session, turn_outputs)
            scored.update({"model_id": model_id, "arm_id": arm_id, "condition_id": condition_id,
                            "memory_condition_result": occ_result})
            self.item_done(cell_id)
            return {"skipped": False, "session": session, "rows": rows, "scored": scored}
        finally:
            if occ is not None:
                occ.stop()


# ── R2.8: hour estimate ──────────────────────────────────────────────────────────────────────────

SECONDS_PER_CALL_8B_CLASS = 25.0  # ASSUMPTION: midpoint of the ~15-40s/call range for 8b-class
                                  # models seen elsewhere in this repo's JSONL data (t2s_overnight,
                                  # t2s_night2 decode/TTFT figures); real per-arm/model timing should
                                  # replace this once Section-0-style smoke timings exist for R2.


def estimate_hours(machine: str = "evo-x2", models=ALL_MODELS, memory_conditions=MEMORY_CONDITIONS,
                    seeds=SEEDS, max_turns: int = MAX_TURNS,
                    seconds_per_call: float = SECONDS_PER_CALL_8B_CLASS) -> dict:
    """Hour estimate for a full real run on `machine`: total call count is models x applicable arms
    x memory conditions x seeds x turns, then x seconds_per_call / 3600. The memory-pressure
    condition gets its own count (it is a separate axis from arms per R2.4, not a multiplier that
    only applies to some arms) and arm (c) is only counted on evo-x2 (R2.3/R2.7)."""
    n_cells = len(enumerate_cells(machine, models=models, memory_conditions=memory_conditions,
                                   seeds=seeds))
    total_calls = n_cells * max_turns
    total_hours = total_calls * seconds_per_call / 3600
    by_condition = {}
    for cond in memory_conditions:
        n = len(enumerate_cells(machine, models=models, memory_conditions=(cond,), seeds=seeds))
        by_condition[cond] = {"cells": n, "calls": n * max_turns,
                               "hours": n * max_turns * seconds_per_call / 3600}
    return {
        "machine": machine, "n_cells": n_cells, "max_turns_per_session": max_turns,
        "total_calls": total_calls, "seconds_per_call_assumption": seconds_per_call,
        "total_hours": total_hours, "by_memory_condition": by_condition,
        "applicable_arms": applicable_arms(machine),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, choices=sorted(k1.K1_HOST_EXTRAS),
                    help="which local machine's config to use (never used to open an SSH connection); "
                         "see harness/host_config.py HOSTS/ALIASES, wrapped by t2s_k1_ollama.require_host")
    ap.add_argument("--ollama-port", type=int, default=k1.DEFAULT_OLLAMA_PORT)
    ap.add_argument("--out-dir", default=str(DEPLOY.parent / "results"))
    ap.add_argument("--resume", default=None, help="prior run stem to resume/extend")
    ap.add_argument("--require-committed", action="store_true",
                    help="refuse to run unless this script is committed (standing rule; off by "
                         "default while this file is new/under review)")
    ap.add_argument("--smoke", action="store_true",
                    help="stub dry run against fake clients; no real Ollama server, no real "
                         "llama-server process (t2s_night2.py's SMOKE_GATED_PHASES table does not "
                         "apply here -- like K1/K2, R2 is a fully separate script with its own flag). "
                         "Prints the exact call-shape count and exits; writes no rows.")
    ap.add_argument("--models", default=None,
                    help="comma list overriding ALL_MODELS for this run (e.g. for a small live "
                         "smoke test: --models llama31-8b)")
    ap.add_argument("--arms", default=None,
                    help="comma list overriding applicable_arms(host) for this run (e.g. --arms "
                         "ollama_default for a small live smoke test using only arm (a))")
    ap.add_argument("--seeds", default=None,
                    help="comma list of ints overriding SEEDS for this run (e.g. --seeds 20260901 "
                         "for a single-session live smoke test)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="overrides MAX_TURNS for this run (e.g. --max-turns 6 for a live smoke test)")
    ap.add_argument("--memory-conditions", default=None,
                    help="comma list overriding MEMORY_CONDITIONS for this run")
    ap.add_argument("--gguf-dir", default=L.MODELS_DIR,
                    help="directory the llama-server arms resolve model gguf filenames against "
                         "(t2s_overnight.MODEL_FILES gives the filename per model id)")
    return ap


def _resolve_llama_server_exe(host_cfg) -> str:
    return L.BINARIES[host_cfg.get("backend", "vulkan")]


def _resolve_gguf_path(model_id: str, gguf_dir: str) -> str:
    fn = ov.MODEL_FILES[model_id][0]
    return str(Path(gguf_dir) / fn)


def _dry_run_call_shape(models, arms, memory_conditions, seeds, max_turns) -> dict:
    """The stub call-shape count --smoke reports: one call per turn per cell, broken down by arm so a
    reviewer can see exactly how many Ollama vs llama-server calls a real run of this scope would
    make, without starting any real process."""
    by_arm = {}
    total = 0
    for arm_id in arms:
        n_cells = sum(1 for _m in models for _c in memory_conditions for _s in seeds)
        n_calls = n_cells * max_turns
        by_arm[arm_id] = {"cells": n_cells, "calls": n_calls}
        total += n_calls
    return {"by_arm": by_arm, "total_calls": total, "models": list(models),
            "memory_conditions": list(memory_conditions), "seeds": list(seeds), "max_turns": max_turns}


def main(argv=None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    host_cfg = k1.require_host(args.host)

    models = tuple(args.models.split(",")) if args.models else ALL_MODELS
    arms = tuple(args.arms.split(",")) if args.arms else tuple(applicable_arms(host_cfg["name"]))
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else SEEDS
    max_turns = args.max_turns if args.max_turns is not None else MAX_TURNS
    memory_conditions = (tuple(args.memory_conditions.split(","))
                         if args.memory_conditions else MEMORY_CONDITIONS)

    # arm (c) is evo-x2 only (R2.3/R2.7); an explicit --arms that names it on evo-t2s is a usage error,
    # not something to silently drop, since a silently-dropped arm on the T2S side is exactly the kind
    # of mistake this whole phase exists to catch elsewhere.
    for arm_id in arms:
        if arm_id not in applicable_arms(host_cfg["name"]):
            raise SystemExit(f"arm {arm_id!r} is not applicable on {host_cfg['name']!r} "
                             f"(machine_restriction={ARMS[arm_id]['machine_restriction']!r})")

    est = estimate_hours(machine=host_cfg["hw_id"], models=models, memory_conditions=memory_conditions,
                         seeds=seeds, max_turns=max_turns)
    log(f"R2 estimated hours on {host_cfg['hw_id']}: {est['total_hours']:.1f} h over {est['n_cells']} cells "
        f"({est['total_calls']} calls, {est['seconds_per_call_assumption']}s/call assumed)")

    if args.smoke:
        shape = _dry_run_call_shape(models, arms, memory_conditions, seeds, max_turns)
        log(f"R2 smoke dry run (no real process started): {json.dumps(shape)}")
        return shape

    prov = rp.script_provenance([__file__], require_committed=args.require_committed)
    lab = R2SessionLab(args, host_cfg, prov)
    Path(lab.prefix + "_manifest.json").write_text(json.dumps({
        "launch_utc": utc_iso(), "host": host_cfg, "args": vars(args), "script_provenance": prov,
        "estimate": est, "models": models, "arms": arms, "seeds": seeds, "max_turns": max_turns,
        "memory_conditions": memory_conditions,
    }, indent=1, default=str), encoding="utf-8")
    log(f"R2 real run writing to {lab.rows_path}")

    def occupier_factory_for(host_cfg, occupier_mi, occupier_n_ctx):
        return lambda: Occupier(lab, occupier_mi, occupier_n_ctx)

    note = "completed"
    ollama_started = False
    try:
        needs_ollama = any(ARMS[a]["runtime"] == "ollama" for a in arms)
        if needs_ollama:
            # the K1 start/stop guard: start Ollama fresh for this run rather than assume one is
            # already resident in the background, and always stop it on the way out (main()'s own
            # try/finally, same shape as t2s_k1_ollama.main()).
            start_result = hc.start_ollama_server()
            ollama_started = bool(start_result and start_result.get("ok", True))
            log(f"R2: Ollama start result: {start_result}")

        occupier_mi = None
        if "occupied_40gb" in memory_conditions:
            # the occupying model: reuse whichever gguf t2s_k1_ollama's own phase_memory_pressure
            # tests use elsewhere (a mid-size model is enough to hold ~40GB at a large n_ctx) -- here,
            # the first model in this run's own model list, so no extra download is required.
            occupier_mi = L.ModelInfo(models[0], _resolve_gguf_path(models[0], args.gguf_dir),
                                      None, False, 32768, 1)

        for model_id in models:
            for arm_id in arms:
                runtime = ARMS[arm_id]["runtime"]
                ollama_tag = MODEL_ID_ALIASES.get(model_id, model_id)
                for condition_id in memory_conditions:
                    for seed in seeds:
                        cell_id = build_item_id(model_id, arm_id, condition_id, seed)
                        if cell_id in lab.done:
                            log(f"R2: skipping {cell_id} (already done, resume)")
                            continue
                        if runtime == "ollama":
                            result = lab.phase_run_session(
                                model_id=ollama_tag, arm_id=arm_id, condition_id=condition_id, seed=seed,
                                ollama=lab.ollama, count_fn=_approx_token_count,
                                occupier_factory=(occupier_factory_for(host_cfg, occupier_mi, 32768)
                                                  if condition_id == "occupied_40gb" else None),
                                max_turns=max_turns)
                        else:
                            exe = _resolve_llama_server_exe(host_cfg)
                            model_path = _resolve_gguf_path(model_id, args.gguf_dir)
                            port = args.ollama_port + 1  # distinct from Ollama's own port
                            cfg = build_llama_server_config(arm_id, exe=exe, model_path=model_path, port=port,
                                                            n_gpu_layers=99, context_shift=False,
                                                            reasoning_budget=0, reasoning_format="none",
                                                            platform=host_cfg["hw_id"])
                            session_obj = ls.LlamaServerSession(cfg)
                            session_obj.start()
                            guard = sg.RequestGuard(port=port, pid=session_obj._proc.pid)
                            try:
                                log_path = session_obj._log_path
                                result = lab.phase_run_session(
                                    model_id=model_id, arm_id=arm_id, condition_id=condition_id, seed=seed,
                                    llama_session=session_obj, count_fn=session_obj.tokenize,
                                    occupier_factory=(occupier_factory_for(host_cfg, occupier_mi, 32768)
                                                      if condition_id == "occupied_40gb" else None),
                                    max_turns=max_turns, guard=guard,
                                    log_text_fn=lambda: Path(log_path).read_text(encoding="utf-8", errors="replace"))
                            finally:
                                session_obj.stop()
                        if result.get("skipped"):
                            continue
                        scored = result["scored"]
                        log(f"R2 cell done: {cell_id} first_failure_turn={scored['first_failure_turn']} "
                            f"first_error_turn={scored['first_error_turn']} "
                            f"error_surfaced_before_failure={scored['error_surfaced_before_failure']}")
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
        log(note)
        raise
    finally:
        if ollama_started:
            try:
                hc.stop_ollama_server()
            except Exception as e:
                log(f"R2: Ollama stop failed: {e!r}")
        try:
            tq.advance(note)
        except Exception as e:
            log(f"queue advance failed: {e!r}")
    return lab, est


if __name__ == "__main__":
    main()
