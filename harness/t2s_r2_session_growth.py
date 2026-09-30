"""R2: agent session growth -- the memory link in this paper's argument (memory -> runtime-chosen
context -> silent truncation -> the agent loses its own rules), with no knob changed except
available memory.

A deterministic turn generator (R2.1) builds a growing agent session: a turn-0 system prompt
carrying 5 checkable rules, a 2-tool schema and 3 recall facts, followed by up to 80 user turns
that each require exactly one tool call plus a ~1.5k-token simulated tool result, with a recall
question added every 4th turn. A pure scorer (R2.2) marks rule violations, tool-call validity,
fact-recall correctness and the sent-vs-processed token gap that is the silent-truncation signal.
Five runtime arms (R2.3) and three memory conditions (R2.4) are then crossed against the generated
sessions to see whether the runtime's context choice tracks free memory or total memory, and
whether failures are ever preceded by a real error.

Note on repo state (read before extending this file)
------------------------------------------------------
The task brief that produced this module named harness/t2s_k1_ollama.py, harness/t2s_k2_pressure.py,
harness/quality_suite.py, harness/host_config.py, tests/test_k1_ollama.py and tests/test_k2_pressure.py
as the conventions to imitate, plus a "K1Lab phase_memory_pressure" occupying-server helper. As of
commit 5ddecb4 (this module's base), none of those files exist in this repository. This module
instead follows the closest real analogs that do exist:

  - t2s_lab.py / t2s_night2.py / t2s_overnight.py for the Lab/JSONL/CLI/host-guard/provenance
    conventions (Jsonl row writer, Server/Balloon lifecycle, script_provenance/verify_deployed_blobs,
    the `--smoke` flag, the socket.gethostname() host guard).
  - context.py's build_filler for the ~1.5k-token simulated tool-result filler (same count_fn
    injection pattern already used by t2s_lab.py's build_probe_prompts).
  - llama_server.py's ContextSizeError, _parse_n_ctx_slot and _log_has_context_shift for the
    llama-server context-overflow classification in R2.3(d) -- reused directly via
    classify_context_overflow() below rather than reimplemented.
  - t2s_lab.py's Server class (already used for the memory-lock experiments in t2s_night2's C1
    phase) for the 40 GB occupying process in R2.4, wrapped by Occupier below rather than
    reimplemented.

This is a fully separate script, the same way a hypothetical K1/K2 harness would be: t2s_night2.py
has no SMOKE_GATED_PHASES table to register into (checked -- absent from this repo), so R2 is not
wired into t2s_night2.py's PHASE_ORDER. It gets its own --smoke flag instead, following
t2s_overnight.py's --smoke convention.

Model ids follow the existing registry convention in t2s_overnight.py's MODEL_FILES (e.g.
"llama31-8b", "qwen3-4b-2507", "qwen3-8b"), not the "llama3.1:8b" / "qwen3:4b-instruct-2507" Ollama
tag spelling from the task brief -- the two are aliases for the same GGUF files; MODEL_ID_ALIASES
below documents the mapping to the Ollama tag spelling for the Ollama arms.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import socket
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
import t2s_lab as L  # noqa: E402
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
    real tokenizer's count_fn (e.g. LlamaServerSession.make_count_fn()) for token-accurate sessions.
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


def build_llama_server_config(arm_id: str, *, exe: str, model_path: str, port: int,
                               **extra) -> ls.LlamaServerConfig:
    """Return the LlamaServerConfig for this arm. Arm (d)'s "default fit" behaviour (no --ctx-size
    override) is represented by DEFAULT_FIT_CTX_SENTINEL; callers that actually launch a process for
    this arm must special-case that sentinel to omit --ctx-size (see start_llama_server_for_arm),
    since LlamaServerConfig.ctx_size is a required field elsewhere in this shared module and other
    experiments depend on it always being written to the command line."""
    cfg = ARMS[arm_id]
    if cfg["runtime"] != "llama_server":
        raise ValueError(f"{arm_id!r} is not a llama-server arm")
    ctx_size = cfg["ctx_size"] if cfg["ctx_size"] is not None else DEFAULT_FIT_CTX_SENTINEL
    return ls.LlamaServerConfig(exe=exe, model=model_path, ctx_size=ctx_size, port=port, **extra)


DEFAULT_FIT_CTX_SENTINEL = -1  # marks "omit --ctx-size, let llama-server auto-pick"


def build_llama_server_cmd(cfg: ls.LlamaServerConfig, log_path: str) -> list[str]:
    """Build the llama-server command line for `cfg`, omitting --ctx-size when cfg.ctx_size is
    DEFAULT_FIT_CTX_SENTINEL (arm d). Duplicated from LlamaServerSession._build_cmd rather than
    monkeypatching that shared class, because other experiments' tests assert its exact command
    shape; this keeps arm (d)'s no-ctx-size behaviour local to R2 only."""
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

MEMORY_CONDITIONS = ("as_is", "occupied_40gb")
OCCUPIER_TARGET_GB = 40


class Occupier:
    """The 40 GB occupying-memory condition. Wraps t2s_lab.Server (already used for the memory-lock
    experiments in t2s_night2's C1 phase) rather than reimplementing process/memory management --
    the "K1Lab phase_memory_pressure" helper named in the original task brief does not exist in this
    repository (see module docstring); Server is the real analog. In a real run this would be
    constructed with a Server sized (model + -c) so its resident memory approaches
    OCCUPIER_TARGET_GB; the exact sizing must be verified against Server.start()'s reported
    private_mib on the live machine before a run is treated as a true "40 GB held" condition.
    """

    def __init__(self, server):
        self.server = server
        self.start_info = None

    def start(self):
        self.start_info = self.server.start()
        return self.start_info

    def stop(self):
        return self.server.stop()


def wire_memory_condition(condition_id: str, occupier_factory=None):
    """Apply memory_condition `condition_id`. Returns the Occupier instance if one was started, or
    None for "as_is". Caller is responsible for calling .stop() on the returned occupier when done
    (a finally block in the real phase runner)."""
    if condition_id == "as_is":
        return None
    if condition_id == "occupied_40gb":
        if occupier_factory is None:
            raise ValueError("occupied_40gb condition requires an occupier_factory")
        occ = occupier_factory()
        occ.start()
        return occ
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


class R2Lab:
    """Orchestrates one (model, arm, condition, seed) session end to end and writes one JSONL row
    per turn, the same way t2s_lab.py's Jsonl-backed rows are written elsewhere in this repo.
    Every external dependency (the Ollama client, the llama-server session, the occupier) is
    injected, so this class runs against fakes in tests with no real network or process calls."""

    def __init__(self, jsonl_path):
        self.jsonl = L.Jsonl(jsonl_path)

    def run_turn_ollama(self, client, model_id: str, arm_id: str, session: SessionSpec,
                         turn: TurnSpec, history: list[dict], *, count_fn) -> dict:
        """One turn against an injected Ollama-like client. `client.chat(model, messages, options)`
        must return a dict with keys: message (={"role":.., "content":..}), prompt_eval_count,
        http_status (default 200), error (default None)."""
        options = apply_ollama_arm(arm_id, {"temperature": 0, "seed": session.seed})
        messages = history + [{"role": "user", "content": turn.user_text}]
        sent_tokens = count_fn(turn.user_text) + sum(count_fn(m["content"]) for m in history)
        resp = client.chat(model=model_id, messages=messages, options=options)
        output_text = (resp.get("message") or {}).get("content", "")
        row = {
            "record": "r2_turn", "backend": "ollama", "arm_id": arm_id, "model_id": model_id,
            "seed": session.seed, "turn_idx": turn.idx, "session_code": session.session_code,
            "output_text": output_text, "sent_tokens": sent_tokens,
            "processed_tokens": resp.get("prompt_eval_count"),
            "http_status": resp.get("http_status", 200), "error_text": resp.get("error"),
            "ts_utc": utc_iso(),
        }
        self.jsonl.write(row)
        return row

    def run_turn_llama_server(self, session_obj, model_id: str, arm_id: str, session: SessionSpec,
                               turn: TurnSpec, *, log_text_fn) -> dict:
        """One turn against an injected llama-server-session-like object. `session_obj` must expose
        .tokenize(text)->int and .call(prompt, max_tokens)-> the 7-tuple LlamaServerSession.call
        returns, raising ls.ContextSizeError on a 400 exceed_context_size_error the same way the
        real session does. `log_text_fn()` returns the server's current log text (for the
        context-shift check)."""
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
        row = {
            "record": "r2_turn", "backend": "llama_server", "arm_id": arm_id, "model_id": model_id,
            "seed": session.seed, "turn_idx": turn.idx, "session_code": session.session_code,
            "output_text": output_text, "sent_tokens": sent_tokens,
            "processed_tokens": processed_tokens, "http_status": http_status,
            "error_text": error_text, "overflow_classification": classification, "ts_utc": utc_iso(),
        }
        self.jsonl.write(row)
        return row

    def phase_run_session(self, *, model_id: str, arm_id: str, condition_id: str, seed: int,
                           client, count_fn, occupier_factory=None, max_turns: int = MAX_TURNS,
                           log_text_fn=None) -> dict:
        """Run one full session for one (model, arm, condition, seed) cell and return its
        score_session() result plus the raw per-turn rows. `client` is either an Ollama-like fake
        (for `runtime == "ollama"` arms) or a llama-server-session-like fake (for the other arms);
        which methods get called is decided purely by ARMS[arm_id]["runtime"]."""
        occ = wire_memory_condition(condition_id, occupier_factory)
        try:
            session = generate_session(seed, count_fn=count_fn, max_turns=max_turns)
            history = [{"role": "system", "content": session.system_prompt}]
            rows = []
            runtime = ARMS[arm_id]["runtime"]
            for turn in session.turns:
                if runtime == "ollama":
                    row = self.run_turn_ollama(client, model_id, arm_id, session, turn, history,
                                                count_fn=count_fn)
                else:
                    row = self.run_turn_llama_server(client, model_id, arm_id, session, turn,
                                                      log_text_fn=log_text_fn or (lambda: ""))
                rows.append(row)
                history.append({"role": "user", "content": turn.user_text})
                history.append({"role": "assistant", "content": row["output_text"]})
            turn_outputs = [{"output_text": r["output_text"], "sent_tokens": r["sent_tokens"],
                              "processed_tokens": r["processed_tokens"],
                              "http_status": r["http_status"], "error_text": r["error_text"]}
                             for r in rows]
            scored = score_session(session, turn_outputs)
            scored.update({"model_id": model_id, "arm_id": arm_id, "condition_id": condition_id})
            return {"session": session, "rows": rows, "scored": scored}
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


def require_host(allowed: tuple[str, ...]) -> str:
    """Host guard following t2s_night2.py's `socket.gethostname().upper() != "EVO-T2S"` pattern,
    generalised to the set of machines R2 is allowed to run on (evo-x2 and evo-t2s)."""
    name = socket.gethostname().upper()
    if name not in allowed:
        raise SystemExit(f"R2 may only run on {allowed}, this host is {name!r}")
    return name


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-blobs", default=None,
                     help="run_provenance.verify_deployed_blobs manifest; required unless --smoke")
    ap.add_argument("--out", default=None, help="output JSONL path prefix")
    ap.add_argument("--smoke", action="store_true",
                     help="stub dry run against fake clients; no real host guard, no real processes")
    ap.add_argument("--machine", default=None, help="override machine id for --smoke runs")
    args = ap.parse_args(argv)

    if args.smoke:
        machine = args.machine or "evo-x2"
    else:
        machine = require_host(("EVO-X2", "EVO-T2S")).lower()
        if not args.expect_blobs:
            raise SystemExit("--expect-blobs is required for a non-smoke run")
        rp.verify_deployed_blobs(DEPLOY, args.expect_blobs)

    est = estimate_hours(machine=machine)
    log(f"R2 estimated hours on {machine}: {est['total_hours']:.1f} h over {est['n_cells']} cells "
        f"({est['total_calls']} calls, {est['seconds_per_call_assumption']}s/call assumed)")

    if not args.smoke:
        raise SystemExit("real (non-smoke) execution requires live Ollama/llama-server wiring not "
                          "built in this pass -- see module docstring; run with --smoke for the "
                          "stub dry run, or use R2Lab.phase_run_session directly with real clients")

    out = args.out or str(DEPLOY / f"t2s_r2_smoke_{int(time.time())}")
    lab = R2Lab(out + ".jsonl")
    log(f"R2 smoke dry run writing to {out}.jsonl")
    return lab, est


if __name__ == "__main__":
    main()
