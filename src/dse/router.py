"""Routers for the DSE demo. Two classes live here:

  Router (the demo router, 2026-10-08; the dashboard builds against its API, keep it stable)
      Per agent step: local vs local_trimmed vs cloud, from the measured envelope in src/dse/envelope_data.json,
      the context-overflow guard from the R2 mechanism result, and remaining cloud budget. See the "Router" section
      further down for the full contract.
  EnvelopeRouter (B6, single-task, older)
      Kept unchanged for analysis/pareto.py and src/dse/live_validation_stub.py; its decision type is
      EnvelopeRouteDecision. Described in the rest of this docstring.

EnvelopeRouter: envelope feasibility + effective context (analysis/envelope_model.py), predicted latency with
the real per-machine co-runner term (B5's 2026-10-01 refit), a simple difficulty estimate, live remaining-budget
state from CloudClient, and a hard rule against ever routing into a silent-failure configuration.

Every EnvelopeRouteDecision carries a human-readable `reason` string -- this is a design requirement (B6b), not an
afterthought: a routing decision that cannot explain itself in one sentence is not auditable.

Silent-failure rule (reused, not reinvented)
---------------------------------------------
"Never route into a silent-failure configuration" is checked two ways, both reusing detection logic/thresholds
that already exist in this repo rather than inventing a new one:
  1. analysis/envelope_model.predict_failure_silence(machine, runtime_policy) -- reuses
     analysis/make_failure_map.EVIDENCE directly. Any outcome in SILENT_FAILURE_OUTCOMES below blocks local
     routing for that (machine, runtime_policy) cell.
  2. analysis/envelope_model.predict_quality_regime(...) -- when the predicted regime is "truncation_cliff" and
     the predicted score falls below QUALITY_FLOOR, that is itself a silent failure in the harness's own sense
     (harness/t2s_r2_session_growth.py's `silent_truncation` classification: "processed_tokens < sent_tokens"
     with no error and no context-shift log line -- the server returns 200 with truncated output, not an error).
     The router treats "predicted truncation-cliff regime below the quality floor" as the same class of event
     and blocks local routing for it too, using the exact R1b-measured score table
     (R1B_BASELINE_SCORE_BY_RATIO) envelope_model already carries, not a new threshold invented here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import envelope_model as em  # noqa: E402
from src.cloud.client import CloudClient, CLOUD_MODELS, estimate_cost_usd  # noqa: E402

# make_failure_map's own outcome vocabulary (docs/FAILURE_MAP.md / analysis/make_failure_map.py EVIDENCE values).
# Any of these for a (machine, runtime_policy) cell means: do not route a real task into that local configuration.
SILENT_FAILURE_OUTCOMES = {"SILENT_SPILL", "SILENT_SPILL, then CRASH one step higher", "CRASH", "HANG",
                            "HARD_FAIL"}  # HARD_FAIL is loud, not silent, but still means "will not run" locally

DEFAULT_QUALITY_FLOOR = 0.5  # predicted_score below this in a truncation_cliff regime is treated as unacceptable


@dataclass
class EnvelopeRouteDecision:
    task_id: str
    target: str  # "local" | "cloud_cheap" | "cloud_strong" | "compact_then_local" | "refuse"
    reason: str
    prediction: dict[str, Any] = field(default_factory=dict)
    blocked_silent_failure: bool = False
    estimated_cost_usd: float = 0.0


def estimate_difficulty(prompt: str) -> float:
    """Simple difficulty proxy in [0, 1]: word count relative to a 1500-word cap, clamped.

    This is a deliberately simple heuristic, not a calibrated difficulty model -- no labeled difficulty dataset
    exists anywhere in this repo to fit one against. It is good enough to separate "short/simple" from
    "long/complex" prompts for the demo's tier choice, and is stated as a heuristic, not a measurement.
    """
    n_words = len(prompt.split())
    return max(0.0, min(1.0, n_words / 1500.0))


class EnvelopeRouter:
    """Routes one task at a time, carrying live remaining-budget state via the CloudClient it is given."""

    def __init__(self, client: CloudClient, machine: str = "evo-t2s", runtime_policy: str = "llama_ngl99",
                 quality_floor: float = DEFAULT_QUALITY_FLOOR, co_runner_present: bool = False,
                 co_runner_kind: str = "bandwidth"):
        self.client = client
        self.machine = machine
        self.runtime_policy = runtime_policy
        self.quality_floor = quality_floor
        self.co_runner_present = co_runner_present
        self.co_runner_kind = co_runner_kind

    def route(self, task: dict[str, Any]) -> EnvelopeRouteDecision:
        task_id = task.get("task_id", "unknown")
        prompt = task.get("prompt", "")
        model_id = task.get("model_id", "qwen3-8b")
        context_length = task.get("context_length", max(1, len(prompt.split()) * 2))
        prompt_tokens = task.get("prompt_tokens", context_length)
        full_prompt_tokens = task.get("full_prompt_tokens", prompt_tokens)

        prediction = em.predict(
            self.machine, self.runtime_policy, model_id, context_length,
            prompt_tokens=prompt_tokens, full_prompt_tokens=full_prompt_tokens,
            co_runner=self.co_runner_present, co_runner_kind=self.co_runner_kind,
        )
        pred_dict = {
            "feasibility": prediction.feasibility, "latency": prediction.latency,
            "effective_context": prediction.effective_context, "quality_regime": prediction.quality_regime,
            "failure_silence": prediction.failure_silence,
        }

        # --- Rule 1: memory-budget feasibility at THIS context length (A-24). predict_feasibility already folds
        # in the make_failure_map/EVIDENCE outcome label whenever the margin is negative for this specific
        # (machine, model, context_length) -- that is the per-task signal, not the generic per-cell one.
        feas = prediction.feasibility
        qual = prediction.quality_regime
        fail_outcome = prediction.failure_silence.get("outcome")

        if feas["status"] == "NOT_MEASURED":
            # No A-24-equivalent bisection exists for this machine/model -- fall back to the generic per-cell
            # EVIDENCE outcome for (machine, runtime_policy) as the only available silent/hard-failure signal.
            if fail_outcome in SILENT_FAILURE_OUTCOMES:
                return self._route_away_from_local(
                    task, task_id, pred_dict,
                    reason=f"blocked local: feasibility is NOT_MEASURED for this (machine, model, context), and "
                           f"(machine={self.machine}, runtime_policy={self.runtime_policy}) is a known "
                           f"{fail_outcome} cell in analysis/make_failure_map.EVIDENCE "
                           f"({prediction.failure_silence.get('detail')}) -- unverified at this exact context, "
                           f"but this router never assumes success when the only evidence available says failure",
                    blocked=True,
                )
        elif feas["status"] != "FITS":
            return self._route_away_from_local(
                task, task_id, pred_dict,
                reason=f"blocked local: predict_feasibility returned status={feas['status']!r} "
                       f"(margin_mib={feas.get('margin_mib')}) for this exact context_length -- a confirmed "
                       f"{feas['status']} outcome from the real A-24 budget-boundary fit, not a generic cell label",
                blocked=True,
            )

        # --- Rule 2: never route into a predicted silent-truncation-style quality cliff below the floor, even
        # when the memory-feasibility check above passed cleanly (truncation is an orthogonal failure mode).
        if qual["regime"] == "truncation_cliff" and qual["predicted_score"] < self.quality_floor:
            return self._route_away_from_local(
                task, task_id, pred_dict,
                reason=f"blocked local: predicted quality regime is truncation_cliff with predicted_score="
                       f"{qual['predicted_score']} (ratio={qual['ratio']}), below quality_floor="
                       f"{self.quality_floor} -- this is the harness's own 'processed_tokens < sent_tokens, no "
                       f"error' silent_truncation signature (harness/t2s_r2_session_growth.py), not a new rule",
                blocked=True,
            )

        # --- Local is feasible and not a predicted silent failure: route locally.
        return EnvelopeRouteDecision(
            task_id=task_id, target="local",
            reason=f"routed local: predict_feasibility={feas['status']} (margin_mib={feas['margin_mib']}), quality_regime="
                   f"{qual['regime']} (predicted_score={qual['predicted_score']}) >= quality_floor="
                   f"{self.quality_floor}, failure_silence outcome={fail_outcome!r} is not in "
                   f"SILENT_FAILURE_OUTCOMES, predicted ttft_s={prediction.latency['ttft_s']}",
            prediction=pred_dict, blocked_silent_failure=False, estimated_cost_usd=0.0,
        )

    def _route_away_from_local(self, task: dict[str, Any], task_id: str, pred_dict: dict,
                               reason: str, blocked: bool) -> EnvelopeRouteDecision:
        """Shared tail for every "don't run this locally" path: try cloud within budget, else compaction if
        the quality floor allows, else refuse -- never silently fall back to the blocked local configuration."""
        prompt = task.get("prompt", "")
        difficulty = estimate_difficulty(prompt)
        tier = "strong" if difficulty > 0.5 else "cheap"
        model_id = CLOUD_MODELS[tier]
        input_tokens = max(1, len(prompt.split()))
        output_tokens = task.get("expected_output_tokens", 256)
        projected_cost = estimate_cost_usd(model_id, input_tokens, output_tokens)

        if projected_cost <= self.client.remaining_budget_usd:
            return EnvelopeRouteDecision(
                task_id=task_id, target=f"cloud_{tier}",
                reason=f"{reason}; routed to cloud_{tier} ({model_id}): difficulty={difficulty:.2f} -> {tier} "
                       f"tier, projected_cost_usd={projected_cost:.4f} <= remaining_budget_usd="
                       f"{self.client.remaining_budget_usd:.4f}",
                prediction=pred_dict, blocked_silent_failure=blocked, estimated_cost_usd=projected_cost,
            )

        # Over budget for cloud. If the quality floor can tolerate a compacted (shorter) local prompt that would
        # bring it back under the effective context (i.e. the ONLY reason this was blocked was feasibility/
        # truncation at the full prompt length, not a hard failure outcome), offer compaction; otherwise refuse.
        qual = pred_dict["quality_regime"]
        feas = pred_dict["feasibility"]
        fail_outcome = pred_dict["failure_silence"].get("outcome")
        if fail_outcome not in SILENT_FAILURE_OUTCOMES and feas["status"] != "HARD_FAIL" and \
                qual["regime"] == "truncation_cliff":
            return EnvelopeRouteDecision(
                task_id=task_id, target="compact_then_local",
                reason=f"{reason}; over cloud budget (projected_cost_usd={projected_cost:.4f} > "
                       f"remaining_budget_usd={self.client.remaining_budget_usd:.4f}) and the block is a "
                       f"truncation_cliff (not a hard/silent failure outcome) -- compaction to fit effective "
                       f"context is offered instead of a cloud call this router cannot afford",
                prediction=pred_dict, blocked_silent_failure=blocked, estimated_cost_usd=0.0,
            )

        return EnvelopeRouteDecision(
            task_id=task_id, target="refuse",
            reason=f"{reason}; refused: over cloud budget (projected_cost_usd={projected_cost:.4f} > "
                   f"remaining_budget_usd={self.client.remaining_budget_usd:.4f}) and local is blocked by a "
                   f"hard/silent failure outcome ({fail_outcome!r}), so compaction cannot rescue this task -- "
                   f"this router never falls back to running it in the blocked local configuration",
            prediction=pred_dict, blocked_silent_failure=blocked, estimated_cost_usd=0.0,
        )


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Router: the demo router (2026-10-08). Public API, the dashboard builds against exactly this:
#
#   Router(envelope, budget_usd, quality_floor, latency_target_ms, hardware, runtime, model, cloud_client=None)
#   Router.decide(step) -> RouteDecision
#   Router.replay_session(rows) -> list[RouteDecision]     (one per R2 turn row; replay_calls: one per model call)
#   load_envelope() -> Envelope                             (the default envelope; Router(None, ...) loads it too)
#
# step = {"messages": [openai-style dicts], "num_predict": int,
#         "needs_turns": optional list[int], "token_counts": optional list[int] (one per message),
#         "tools": optional tool schema list (counted with the system prompt),
#         "num_ctx": optional int (Ollama's per-request num_ctx option; overrides the Ollama runtime's context for
#                    this step, ignored for llama-server whose -c is fixed at start),
#         "co_runner": optional bool (overrides the router's default for this step)}
#
# What a decision is built from
# -----------------------------
# (a) The envelope: src/dse/envelope_data.json, written by analysis/build_envelope_data.py from numbers-register rows
#     only (each block names its claim_id). It gives, per machine x model: the effective context of the runtime
#     (Ollama default ctx per machine: evo-t2s 4096 for every measured model, evo-x2 per model; an explicit num_ctx or
#     llama-server -c is taken as given), the memory budget (evo-t2s llama-server: the last n_ctx that started in the
#     A-24 bisection; anything else is "not measured", never assumed to fail), and predicted latency
#     = TTFT fit (a*n + b*n^2, fresh prompt) x co-runner TTFT ratio + num_predict / (decode tok/s x co-runner decode
#     ratio). The co-runner term is per machine: evo-t2s B3 nonp12 (TTFT only), evo-x2 PX2 B4 (TTFT and decode).
#     The TTFT fit is for a fresh prompt; a runtime that reuses a cached prefix (Ollama within a session) is faster,
#     so for multi-turn sessions the figure is an upper estimate.
# (b) The context guard, from the R2 mechanism result (FINDINGS "R2 mechanism, evo-x2", 2026-10-08): when system
#     prompt + history + answer budget exceeds num_ctx, Ollama drops whole messages but keeps the system message, and
#     then llama.cpp's context shift and Ollama's token cut keep only the first 4-5 tokens and discard what follows,
#     system prompt included (at 4096), all with HTTP 200. So the router never sends a local request whose
#     system + kept history + num_predict is over the usable window, usable = floor(num_ctx x (1 - 5% margin)). The
#     margin applies to the trigger as well as to the trim target, because the token count is an estimate. When the
#     step is over, the router trims whole turns, oldest first: a turn is a user message plus every assistant and
#     tool message after it up to the next user message, so a tool result is never orphaned and an assistant message
#     never loses its user message. System messages are always kept; the current (last) turn is always kept.
#     If the trimmed request still does not fit (system + current turn alone are over), local is not possible.
# (c) Cloud budget as state: remaining = min(budget_usd - projected spend of this router's cloud decisions so far,
#     cloud_client.remaining_budget_usd) (the client's hard cap over the whole ledger). The router never calls the
#     cloud; it projects cost with the client's published-price estimate (input = the untrimmed request,
#     output = num_predict). The default client is CloudClient(mode="stub"): stub unless a caller passes a real one.
#
# Quality floor: when does a step "need" a dropped turn?
# -------------------------------------------------------
# A step needs turn k if (1) k is in step["needs_turns"] (the caller knows, e.g. R2 replay marks canary turns as
# needing turn 1, where the history tags live), or (2) the current turn's user text contains an identifier
# (pattern ID_PATTERN, e.g. "REC-0001", "HTAG-...") that occurs in turn k's text. If trimming would drop a needed
# turn and quality_floor > 0, a trimmed local answer cannot meet the floor (with the fact gone the answer is wrong,
# not refused: register self-report-truncation-awareness and R2-real-v1-gated-kill), so the step goes to cloud.
# require_full_history=True is the stricter policy flag: any trim at all goes to cloud. quality_floor == 0 means
# the caller accepts trimmed history. If cloud is needed but the budget cannot cover it, the router still returns
# local_trimmed (never an overflowing local call) and says in the reason that the floor is not met
# (quality_floor_met=False). target "refuse" is returned only when local is impossible (memory budget exceeded,
# or system + current turn do not fit) and the cloud budget cannot cover the step.
#
# Token counting (documented, the R2 harness's calibrated estimate)
# ------------------------------------------------------------------
# Default: harness/x2_r2_agent.py's transcript_tokens per message: len(text) // 4 (min 1 for non-empty text) + 4
# per-message overhead (+ the same estimate of json.dumps(tool_calls) for an assistant tool-call message), times the
# model's calibration ratio (register R2-real-v1-token-calib: turn-1 prompt_eval_count / estimate, median over
# sessions; 1.0 for a model without a measured ratio), rounded up per message. Tools: the estimate of
# json.dumps(tools), same ratio. Override per step with "token_counts", or per router with token_counter=callable
# (message dict -> int), e.g. a real tokenizer.
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════

import json as _json
import math as _math
import re as _re

ENVELOPE_DATA_PATH = Path(__file__).resolve().parent / "envelope_data.json"
CONTEXT_MARGIN = 0.05
PER_MESSAGE_OVERHEAD_TOKENS = 4  # harness/x2_r2_agent.py PER_MESSAGE_OVERHEAD_TOKENS
ID_PATTERN = _re.compile(r"\b[A-Z]{2,}-[A-Z0-9]{4,}\b")
TARGETS = ("local", "local_trimmed", "cloud", "refuse")
DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "results" / "cloud_ledger.jsonl"


def norm_model(name: str) -> str:
    """One key for both model spellings in the register: Ollama tags and llama-server ids (llama3.1:8b == llama31-8b)."""
    return name.strip().lower().replace(".", "").replace(":", "-")


def _chars4(text) -> int:
    if not text:
        return 0
    if not isinstance(text, str):
        text = _json.dumps(text)
    return max(1, len(text) // 4)


def r2_message_tokens_raw(message: dict) -> int:
    """harness/x2_r2_agent.py transcript_tokens for one message, before calibration."""
    n = _chars4(message.get("content")) + PER_MESSAGE_OVERHEAD_TOKENS
    if message.get("tool_calls"):
        n += _chars4(_json.dumps(message["tool_calls"]))
    return n


def _fmt_int(n) -> str:
    return f"{int(round(n)):,}"


def _fmt_turns(turns: list[int]) -> str:
    """[1,2,3,5] -> '1-3, 5'."""
    if not turns:
        return "none"
    out, start, prev = [], turns[0], turns[0]
    for t in turns[1:] + [None]:
        if t is not None and t == prev + 1:
            prev = t
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        if t is not None:
            start = prev = t
    return ", ".join(out)


class Envelope:
    """Read-only view of src/dse/envelope_data.json (see analysis/build_envelope_data.py)."""

    def __init__(self, data: dict):
        self.data = data

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Envelope":
        p = Path(path) if path else ENVELOPE_DATA_PATH
        return cls(_json.loads(p.read_text(encoding="utf-8")))

    @classmethod
    def coerce(cls, envelope) -> "Envelope":
        if envelope is None:
            return cls.load()
        if isinstance(envelope, Envelope):
            return envelope
        if isinstance(envelope, dict):
            return cls(envelope)
        return cls.load(envelope)

    def machine(self, hw: str) -> dict:
        try:
            return self.data["machines"][hw]
        except KeyError:
            raise ValueError(f"hardware {hw!r} not in the envelope; known: {sorted(self.data['machines'])}") from None

    def ollama_default_ctx(self, hw: str, model: str) -> tuple[int, list[str]]:
        table = self.machine(hw)["ollama_default_ctx"]
        key = norm_model(model)
        if key not in table:
            raise ValueError(f"no measured Ollama default context for {model} on {hw} (measured: {sorted(table)}); "
                             f"pass an explicit runtime such as 'ollama_ctx_4096'")
        return table[key]["ctx"], table[key]["claim_ids"]

    def memory_check(self, hw: str, runtime_kind: str, model: str, num_ctx: int) -> tuple[str, str]:
        """('FITS' | 'OVER_BUDGET' | 'NOT_MEASURED', short text)."""
        mem = self.machine(hw).get("llama_server_memory") if runtime_kind == "llama_server" else None
        if not mem or norm_model(model) not in mem["last_ok_n_ctx"]:
            return "NOT_MEASURED", "memory budget not measured for this machine/runtime/model"
        last_ok = mem["last_ok_n_ctx"][norm_model(model)]
        if num_ctx <= last_ok:
            return "FITS", f"-c {_fmt_int(num_ctx)} <= last n_ctx that started {_fmt_int(last_ok)} ({mem['claim_id']})"
        return "OVER_BUDGET", (f"-c {_fmt_int(num_ctx)} > last n_ctx that started {_fmt_int(last_ok)} on {hw} "
                               f"({mem['claim_id']})")

    def token_calib_ratio(self, model: str) -> float | None:
        return self.data.get("token_calib_ratio", {}).get(norm_model(model))

    def predict_latency_ms(self, hw: str, model: str, prompt_tokens: int, num_predict: int,
                           co_runner: bool = False) -> tuple[float | None, dict]:
        m = self.machine(hw)
        key = norm_model(model)
        fit = m.get("ttft_fit", {}).get(key)
        dec = m.get("decode_tok_s", {}).get(key)
        if fit is None or not dec:
            return None, {"basis": "not measured"}
        n = float(prompt_tokens)
        ttft_s = max(0.0, fit["a"] * n + fit["b"] * n * n)
        ttft_ratio = decode_ratio = 1.0
        if co_runner:
            cr = m.get("co_runner") or {}
            ttft_ratio = cr.get("ttft_ratio", {}).get(key, 1.0)
            decode_ratio = cr.get("decode_ratio", {}).get(key, 1.0)
        decode_s = num_predict / (dec * decode_ratio)
        total_ms = (ttft_s * ttft_ratio + decode_s) * 1000.0
        return total_ms, {"basis": "ttft-physical-fit-per-machine + decode-rate-per-machine",
                          "ttft_s": ttft_s * ttft_ratio, "decode_s": decode_s, "ttft_fit_r2": fit["r2"],
                          "co_runner": bool(co_runner), "co_runner_ttft_ratio": ttft_ratio,
                          "co_runner_decode_ratio": decode_ratio}


def parse_runtime(runtime: str) -> tuple[str, int | None]:
    """'ollama' / 'ollama_default[...]' -> ('ollama', None); 'ollama_ctx_4096[...]' / 'ollama:4096' ->
    ('ollama', 4096); 'llama_server:32768' / 'llama_server_ctx_32768' -> ('llama_server', 32768). R2 arm ids
    (e.g. 'ollama_ctx_4096_call2_notools') parse too. llama-server always needs an explicit -c."""
    s = runtime.strip().lower().replace("-", "_")
    if s.startswith("llama_server"):
        m = _re.match(r"llama_server(?:_ctx_|_c_|:|_)(\d+)", s)
        if not m:
            raise ValueError(f"llama-server runtime needs an explicit -c, e.g. 'llama_server:32768' (got {runtime!r})")
        return "llama_server", int(m.group(1))
    if s.startswith("ollama"):
        m = _re.match(r"ollama(?:_ctx_|:|_num_ctx_)(\d+)", s)
        if m:
            return "ollama", int(m.group(1))
        if s == "ollama" or s.startswith("ollama_default"):
            return "ollama", None
    raise ValueError(f"unknown runtime {runtime!r}: use 'ollama', 'ollama_ctx_<N>' or 'llama_server:<N>'")


@dataclass
class RouteDecision:
    """One routing decision. target: 'local' | 'local_trimmed' | 'cloud' (| 'refuse', see the Router section).
    For a cloud decision machine='cloud', runtime='stub' or 'real', model = the cloud model id, and num_ctx is the
    local window that was considered. tokens: system (incl. tools), kept_history, answer_budget, total, plus
    full_history / full_total (the untrimmed request). kept_turns / dropped_turns are counts (what the dashboard
    shows); kept_turn_ids / dropped_turn_ids are the turn numbers, 1.. by user message."""
    target: str
    machine: str
    runtime: str
    model: str
    num_ctx: int
    tokens: dict
    kept_turns: int
    dropped_turns: int
    predicted_latency_ms: Optional[float]
    est_cost_usd: float
    budget_remaining_usd: float
    stub: bool
    reason: str
    kept_turn_ids: list = field(default_factory=list)
    dropped_turn_ids: list = field(default_factory=list)
    needs_turns: list = field(default_factory=list)
    quality_floor_met: bool = True
    memory_status: str = "NOT_MEASURED"
    latency_target_met: Optional[bool] = None
    turn_idx: Optional[int] = None
    call_idx: Optional[int] = None
    observed: Optional[dict] = None

    def to_dict(self) -> dict:
        import dataclasses as _dc
        return _dc.asdict(self)


def split_turns(messages: list[dict]) -> tuple[list[int], list[list[int]]]:
    """(indices of system messages, turns as lists of message indices). A turn starts at each user message and runs
    to the next one; non-system messages before the first user message belong to the first turn."""
    system, turns = [], []
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            system.append(i)
        elif m.get("role") == "user" or not turns:
            turns.append([i])
        else:
            turns[-1].append(i)
    return system, turns


def _msg_text(m: dict) -> str:
    c = m.get("content")
    text = c if isinstance(c, str) else (_json.dumps(c) if c else "")
    if m.get("tool_calls"):
        text += _json.dumps(m["tool_calls"])
    return text


class Router:
    """Per-step local / local_trimmed / cloud router. See the Router section comment above for the full contract."""

    def __init__(self, envelope, budget_usd: float, quality_floor: float, latency_target_ms: float | None,
                 hardware: str, runtime: str, model: str, cloud_client: CloudClient | None = None, *,
                 cloud_model: str = CLOUD_MODELS["cheap"], co_runner: bool = False,
                 require_full_history: bool = False, token_counter=None, token_calib_ratio: float | None = None,
                 margin: float = CONTEXT_MARGIN):
        self.envelope = Envelope.coerce(envelope)
        self.budget_usd = float(budget_usd)
        self.quality_floor = float(quality_floor)
        self.latency_target_ms = latency_target_ms
        self.hardware = hardware
        self.runtime = runtime
        self.model = model
        self.client = cloud_client if cloud_client is not None else CloudClient(mode="stub",
                                                                               ledger_path=DEFAULT_LEDGER)
        self.cloud_model = cloud_model
        self.co_runner = co_runner
        self.require_full_history = require_full_history
        self.token_counter = token_counter
        self.margin = margin
        self.committed_usd = 0.0

        self.envelope.machine(hardware)  # validates the machine
        self.runtime_kind, explicit_ctx = parse_runtime(runtime)
        if explicit_ctx is None:
            self.num_ctx, self.ctx_claims = self.envelope.ollama_default_ctx(hardware, model)
            self.runtime_label = f"Ollama default ctx {_fmt_int(self.num_ctx)}"
        else:
            self.num_ctx, self.ctx_claims = explicit_ctx, []
            self.runtime_label = (f"Ollama num_ctx {_fmt_int(self.num_ctx)}" if self.runtime_kind == "ollama"
                                  else f"llama-server -c {_fmt_int(self.num_ctx)}")
        if token_calib_ratio is None:
            token_calib_ratio = self.envelope.token_calib_ratio(model) or 1.0
        self.token_calib_ratio = float(token_calib_ratio)
        self.usable_ctx = int(_math.floor(self.num_ctx * (1.0 - margin)))

    # ── state
    @property
    def stub(self) -> bool:
        return bool(getattr(self.client, "stub_mode", True))

    def budget_remaining_usd(self) -> float:
        return max(0.0, min(self.budget_usd - self.committed_usd, self.client.remaining_budget_usd))

    # ── token counting
    def _count(self, step: dict) -> tuple[list[int], int]:
        msgs = step["messages"]
        if step.get("token_counts") is not None:
            counts = [int(_math.ceil(c)) for c in step["token_counts"]]
            if len(counts) != len(msgs):
                raise ValueError(f"token_counts has {len(counts)} entries for {len(msgs)} messages")
        elif self.token_counter is not None:
            counts = [int(self.token_counter(m)) for m in msgs]
        else:
            counts = [int(_math.ceil(r2_message_tokens_raw(m) * self.token_calib_ratio)) for m in msgs]
        tools = step.get("tools")
        if step.get("tools_tokens") is not None:
            tools_tokens = int(step["tools_tokens"])
        elif tools:
            tools_tokens = int(_math.ceil(_chars4(_json.dumps(tools)) * self.token_calib_ratio))
        else:
            tools_tokens = 0
        return counts, tools_tokens

    # ── the decision
    def decide(self, step: dict) -> RouteDecision:
        msgs = step["messages"]
        num_predict = int(step["num_predict"])
        co_runner = step.get("co_runner", self.co_runner)
        counts, tools_tokens = self._count(step)
        sys_idx, turns = split_turns(msgs)
        n_turns = len(turns)
        turn_tok = [sum(counts[i] for i in t) for t in turns]
        system_tok = sum(counts[i] for i in sys_idx) + tools_tokens
        full_hist = sum(turn_tok)
        full_total = system_tok + full_hist + num_predict
        num_ctx, runtime_label = self.num_ctx, self.runtime_label
        if step.get("num_ctx") and self.runtime_kind == "ollama":  # Ollama's per-request num_ctx option
            num_ctx = int(step["num_ctx"])
            runtime_label = f"Ollama num_ctx {_fmt_int(num_ctx)}"
        limit = int(_math.floor(num_ctx * (1.0 - self.margin)))
        all_turns = list(range(1, n_turns + 1))

        # trim oldest whole turns (never the current one) until the request fits the usable window
        dropped: list[int] = []
        kept_hist = full_hist
        k = 0
        while system_tok + kept_hist + num_predict > limit and k < n_turns - 1:
            kept_hist -= turn_tok[k]
            dropped.append(k + 1)
            k += 1
        fits_local = system_tok + kept_hist + num_predict <= limit
        kept = [t for t in all_turns if t not in dropped]

        # which turns does this step need?
        needs = set(int(t) for t in (step.get("needs_turns") or []))
        if turns and dropped:
            cur_text = " ".join(_msg_text(msgs[i]) for i in turns[-1] if msgs[i].get("role") == "user")
            ids = set(ID_PATTERN.findall(cur_text))
            if ids:
                # an identifier still present in what is kept (system prompt, kept earlier turns) is not lost
                kept_text = " ".join(_msg_text(msgs[i]) for i in sys_idx) + " " + " ".join(
                    _msg_text(msgs[i]) for tn, t in enumerate(turns[:-1], start=1) if tn not in dropped for i in t)
                ids = {x for x in ids if x not in kept_text}
                for tn in dropped:
                    text = " ".join(_msg_text(msgs[i]) for i in turns[tn - 1])
                    if any(x in text for x in ids):
                        needs.add(tn)
        needs_list = sorted(t for t in needs if 1 <= t <= n_turns)
        needed_dropped = [t for t in needs_list if t in dropped]
        quality_needs_cloud = bool(dropped) and (
            (self.quality_floor > 0 and bool(needed_dropped)) or self.require_full_history)

        mem_status, mem_text = self.envelope.memory_check(self.hardware, self.runtime_kind, self.model, num_ctx)
        local_prompt = system_tok + (kept_hist if fits_local else full_hist)
        lat_ms, _lat = self.envelope.predict_latency_ms(self.hardware, self.model, local_prompt, num_predict,
                                                        co_runner)
        lat_ok = None if (self.latency_target_ms is None or lat_ms is None) else lat_ms <= self.latency_target_ms

        cost = self.client.estimate_cost(self.cloud_model, system_tok + full_hist, num_predict)
        remaining = self.budget_remaining_usd()
        affordable = cost <= remaining + 1e-12

        where = f"{self.hardware} {runtime_label}"
        over_txt = (f"{_fmt_int(full_total)} tok > {_fmt_int(limit)} usable of {_fmt_int(num_ctx)} ctx on "
                    f"{where}")
        cloud_txt = f"{self.cloud_model} est ${cost:.4f}"

        def make(target, reason, kept_turns, dropped_turns, q_met=True):
            if target == "cloud":
                self.committed_usd += cost
                tokens = {"system": system_tok, "kept_history": full_hist, "answer_budget": num_predict,
                          "total": full_total}
                machine, runtime, model = "cloud", ("stub" if self.stub else "real"), self.cloud_model
                lat, est, lat_met = None, cost, None
            elif target == "refuse":
                tokens = {"system": system_tok, "kept_history": 0, "answer_budget": num_predict,
                          "total": system_tok + num_predict}
                machine, runtime, model = self.hardware, self.runtime, self.model
                lat, est, lat_met = None, 0.0, None
            else:
                hist = kept_hist if target == "local_trimmed" else full_hist
                tokens = {"system": system_tok, "kept_history": hist, "answer_budget": num_predict,
                          "total": system_tok + hist + num_predict}
                machine, runtime, model = self.hardware, self.runtime, self.model
                lat, est, lat_met = lat_ms, 0.0, lat_ok
            tokens["full_history"] = full_hist
            tokens["full_total"] = full_total
            left = self.budget_remaining_usd()
            if target == "cloud":
                reason = f"{reason}; {cloud_txt}, ${left:.4f} left" + (" (stub)" if self.stub else "")
            return RouteDecision(
                target=target, machine=machine, runtime=runtime, model=model, num_ctx=num_ctx, tokens=tokens,
                kept_turns=len(kept_turns), dropped_turns=len(dropped_turns), kept_turn_ids=list(kept_turns),
                dropped_turn_ids=list(dropped_turns),
                predicted_latency_ms=None if lat is None else round(lat, 1), est_cost_usd=est,
                budget_remaining_usd=left, stub=self.stub, reason=reason, needs_turns=needs_list,
                quality_floor_met=q_met, memory_status=mem_status, latency_target_met=lat_met,
                turn_idx=step.get("turn_idx"), call_idx=step.get("call_idx"), observed=step.get("observed"))

        no_budget = f"cloud {cloud_txt} > ${remaining:.4f} budget left"

        # 1. memory budget: a measured boundary says this local configuration will not start
        if mem_status == "OVER_BUDGET":
            if affordable:
                return make("cloud", f"cloud: {mem_text}", all_turns, [])
            return make("refuse", f"refuse: {mem_text}; {no_budget}", [], all_turns, q_met=False)

        # 2. system prompt + current turn alone do not fit the usable window
        if not fits_local:
            cur = system_tok + turn_tok[-1] + num_predict if turns else system_tok + num_predict
            base = (f"system + current turn + answer = {_fmt_int(cur)} tok > {_fmt_int(limit)} usable of "
                    f"{_fmt_int(num_ctx)} ctx on {where}, trimming cannot fit it")
            if affordable:
                return make("cloud", f"cloud: {base}", all_turns, [])
            return make("refuse", f"refuse: {base}; {no_budget}", [], all_turns, q_met=False)

        lat_txt = "" if lat_ms is None else f"; predicted {_fmt_int(lat_ms)} ms"
        lat_over = lat_ok is False

        # 3. fits untrimmed
        if not dropped:
            if lat_over and affordable:
                return make("cloud", f"cloud: {_fmt_int(full_total)} tok fits {where} but predicted "
                                     f"{_fmt_int(lat_ms)} ms > {_fmt_int(self.latency_target_ms)} ms target", all_turns, [])
            mem = "" if mem_status == "NOT_MEASURED" else f"; memory {mem_status.lower()}"
            return make("local", f"local: {_fmt_int(full_total)} tok <= {_fmt_int(limit)} usable of "
                                 f"{_fmt_int(num_ctx)} ctx on {where}{mem}{lat_txt}", all_turns, [])

        # 4. over the window: trim, unless the quality floor needs what trimming drops
        trim_txt = f"dropped turn{'s' if len(dropped) != 1 else ''} {_fmt_turns(dropped)}, kept system prompt"
        if quality_needs_cloud:
            why = (f"trimming would drop turn{'s' if len(needed_dropped) != 1 else ''} {_fmt_turns(needed_dropped)}, "
                   f"which this step needs (quality floor {self.quality_floor:g})" if needed_dropped else
                   f"trimming would drop turn{'s' if len(dropped) != 1 else ''} {_fmt_turns(dropped)} and "
                   f"require_full_history is set")
            if affordable:
                return make("cloud", f"cloud: {over_txt}; {why}", all_turns, [])
            return make("local_trimmed", f"local_trimmed: {over_txt}; {trim_txt}; {why} but {no_budget}, "
                                         f"quality floor NOT met", kept, dropped, q_met=False)
        if lat_over and affordable:
            return make("cloud", f"cloud: {over_txt}; trimmed request predicted {_fmt_int(lat_ms)} ms > "
                                 f"{_fmt_int(self.latency_target_ms)} ms target", all_turns, [])
        floor_txt = ("quality floor does not need dropped history" if self.quality_floor > 0
                     else "quality floor 0 accepts trimmed history")
        return make("local_trimmed", f"local_trimmed: {over_txt}; {trim_txt}; {floor_txt}{lat_txt}", kept, dropped)

    # ── R2 replay
    def replay_calls(self, rows: list[dict]) -> list[RouteDecision]:
        """Replays R2 session rows (results/x2_r2_real_v1.jsonl format, r2a_turn records; other records are
        ignored) through this router, one decision per model call (call 1 and call 2 of each turn). The requests
        are rebuilt as the harness built them (src/dse/r2_replay.py); each step carries the session's own num_ctx
        (Ollama's per-request option, so it overrides the runtime default when the router runs Ollama) and the real
        run's observation in decision.observed. Counterfactual: the real run's later turns are what the model said
        with the untrimmed history; the router's trims are not fed back into them."""
        from src.dse import r2_replay  # noqa: PLC0415  (imports the R2 harness, only needed here)
        return [self.decide(step) for step in r2_replay.r2_steps(rows)]

    def replay_session(self, rows: list[dict]) -> list[RouteDecision]:
        """One decision per r2a_turn row, in row order (sessions grouped as r2_replay.r2_sessions does). Every call
        of the turn is decided (replay_calls); the turn's decision is its final call's (the answer request), with
        est_cost_usd summed over the turn's calls, budget_remaining_usd after them, and observed["call_targets"]
        listing each call's target. If an earlier call of the turn got a different target, the reason says so."""
        out: list[RouteDecision] = []
        by_turn: dict = {}
        order: list = []
        for d in self.replay_calls(rows):
            o = d.observed or {}
            key = (o.get("model_id"), o.get("arm_id"), o.get("seed"), d.turn_idx)
            if key not in by_turn:
                order.append(key)
            by_turn.setdefault(key, []).append(d)
        for key in order:
            calls = by_turn[key]
            final = calls[-1]
            final.est_cost_usd = sum(c.est_cost_usd for c in calls)
            final.observed = dict(final.observed or {}, call_targets=[c.target for c in calls])
            others = [f"call {c.call_idx}: {c.target}" for c in calls[:-1] if c.target != final.target]
            if others:
                final.reason = f"{final.reason}; {', '.join(others)}"
            out.append(final)
        return out


def load_envelope(path: str | Path | None = None) -> Envelope:
    """The envelope Router uses by default (src/dse/envelope_data.json)."""
    return Envelope.load(path)


if __name__ == "__main__":
    import tempfile

    from src.cloud.client import print_stub_banner_if_needed

    demo_tasks = [
        # over native-context -> blocked (truncation_cliff below floor) -> cloud, NOT a memory-budget failure.
        {"task_id": "over_native_ctx_truncation", "model_id": "qwen3-32b", "context_length": 100000,
         "prompt": "short prompt"},
        # over the A-24 memory budget boundary -> HARD_FAIL -> cloud.
        {"task_id": "over_budget_hard_fail", "model_id": "qwen3-32b", "context_length": 120000,
         "prompt": "short prompt"},
        # fits comfortably under both the native-context cap and the memory budget -> routed local.
        {"task_id": "fits_local", "model_id": "qwen3-8b", "context_length": 300,
         "prompt_tokens": 294, "full_prompt_tokens": 300, "prompt": "x " * 50},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        client = CloudClient(api_key=None, ledger_path=Path(tmp) / "ledger.jsonl")
        print_stub_banner_if_needed(client)
        router = EnvelopeRouter(client, machine="evo-t2s", runtime_policy="llama_ngl99")
        for t in demo_tasks:
            d = router.route(t)
            print(f"\n{d.task_id}: target={d.target} blocked_silent_failure={d.blocked_silent_failure}")
            print(f"  reason: {d.reason}")
