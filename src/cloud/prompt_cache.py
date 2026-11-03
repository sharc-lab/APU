"""Anthropic prompt caching: where the cache breakpoints go, and what that placement costs.

One module for both sides, so the live client and the pricing model cannot drift apart:

  with_rolling_breakpoints(messages)   the live message builder (scripts/cloud_redo.py::with_cache_control, used by
                                       scripts/cloud_redo.py and scripts/all_cloud_b3.py): one breakpoint on the system
                                       prompt and one ROLLING breakpoint on the latest message with text, whatever its
                                       role (user, tool result, assistant).
  rolling_split(prompts, min_cacheable, first=None)
                                       the pricing model of that placement: per call, (read, write, uncached) tokens.
  reprice_recorded_calls(calls, ...)   the same model applied to the per-call usage of a recorded run that used the
                                       earlier placement (offline repricing, no call is made).

Why (2026-11-02, external review): the earlier builder put the second breakpoint on the last USER message. In a tool
loop that message is the request near the start of the turn, so every tool round after it was sent at the uncached
input rate on the next call and never written to the cache (docs/EXPERIMENTS.md, the long-task note under
"billed / priced-realistic"). Anthropic's rule (platform.claude.com/docs/en/build-with-claude/prompt-caching, quoted
in results/cloud_prices_anthropic_20261014.json "caching_rules_quoted"): a breakpoint caches the whole prefix up to and
including its block; a later request reads the longest cached prefix found within 20 blocks before one of its
breakpoints; reads are billed at the cache-hit rate, the newly cached part at the 5-minute cache-write rate; at most
4 breakpoints per request; the default lifetime is 5 minutes, refreshed on every read.

The rolling model, per call k of one conversation (prompt P_k tokens, in order):
  k = first call: as given by `first` (read, write, uncached) when known (a recorded first call keeps its recorded
      split, e.g. a system prompt already cached by a warm-up), else write P_k (cold cache) when P_k reaches the
      model's minimum cacheable prefix, else all uncached;
  k > first: read P_{k-1} when P_{k-1} reaches the minimum and P_k >= P_{k-1} (the conversation only grew, so the
      previous call's cached prompt is a prefix of this one), write P_k - read, uncached 0. When P_k < P_{k-1} the
      history was rewritten (condensed or truncated): no prefix match is assumed beyond `fallback_read` (default 0),
      the rest is written.
ASSUMPTIONS stated with every figure priced from it: consecutive calls of one conversation are under 5 minutes apart
(the cache lifetime), the prefix is byte-identical between calls (same tools, same tool_choice, same system prompt),
and each call adds at most 20 blocks (one assistant message and its tool results) so the lookback finds the previous
breakpoint.
"""
from __future__ import annotations

import copy
from typing import Iterable

EPHEMERAL = {"type": "ephemeral"}
MAX_BREAKPOINTS = 4     # Anthropic: at most 4 cache_control breakpoints per request
ROLLING_LABEL = ("rolling breakpoint (system prompt + latest message): each call reads the previous call's whole "
                 "prompt from cache and writes only the new tokens")


# ── live message builder ─────────────────────────────────────────────────────────────────────────────

def _n_breakpoints(m: dict) -> int:
    n = 1 if isinstance(m.get("cache_control"), dict) else 0
    c = m.get("content")
    if isinstance(c, list):
        n += sum(1 for b in c if isinstance(b, dict) and isinstance(b.get("cache_control"), dict))
    return n


def count_breakpoints(messages: Iterable[dict]) -> int:
    return sum(_n_breakpoints(m) for m in messages)


def _has_text(m: dict) -> bool:
    c = m.get("content")
    if isinstance(c, str):
        return bool(c)
    if isinstance(c, list):
        return any(isinstance(b, dict) and b.get("type") == "text" and b.get("text") for b in c)
    return False


def _mark(m: dict) -> None:
    """Put the breakpoint on the message's last non-empty text block (Anthropic rejects an empty text block)."""
    c = m["content"]
    if isinstance(c, str):
        m["content"] = [{"type": "text", "text": c, "cache_control": dict(EPHEMERAL)}]
        return
    for b in reversed(c):
        if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
            b["cache_control"] = dict(EPHEMERAL)
            return


def latest_text_index(messages: list[dict]) -> int | None:
    """Index of the latest message with non-empty text (the rolling breakpoint's position), None when none has text."""
    return next((i for i in range(len(messages) - 1, -1, -1) if _has_text(messages[i])), None)


def with_rolling_breakpoints(messages: list[dict]) -> list[dict]:
    """A copy of OpenAI-shape messages (litellm) with cache breakpoints on the system prompt (when the first message is
    one) and on the latest message with text (the rolling breakpoint: user, tool result or assistant). Breakpoints
    already present are kept and counted; the total never exceeds MAX_BREAKPOINTS (the system prompt first, then the
    latest message, so a caller that pre-marks 4 blocks gets no new ones)."""
    out = copy.deepcopy(messages)
    used = count_breakpoints(out)
    targets = []
    if out and out[0].get("role") == "system" and _has_text(out[0]):
        targets.append(0)
    last = latest_text_index(out)
    if last is not None and last not in targets:
        targets.append(last)
    for i in targets:
        if used >= MAX_BREAKPOINTS:
            break
        if _n_breakpoints(out[i]):
            continue
        _mark(out[i])
        used += 1
    return out


# ── pricing model of the placement ─────────────────────────────────────────────────────────────────────

def rolling_split(prompts: list[int], min_cacheable: int, first: tuple | None = None,
                  fallback_read: float = 0) -> list[tuple]:
    """Per call of one conversation, (read, write, uncached) input tokens under the rolling breakpoint (module
    docstring). prompts: total input tokens per call in order; first: the first call's (read, write, uncached) when
    known (recorded), else a cold cache."""
    out, prev = [], None
    for k, p in enumerate(prompts):
        if k == 0:
            if first is not None:
                r, w, u = first
                if abs((r + w + u) - p) > 1e-6 * max(p, 1):
                    raise ValueError(f"first-call split {first} does not add up to its prompt {p}")
                out.append((r, w, u))
            elif p >= min_cacheable:
                out.append((0, p, 0))
            else:
                out.append((0, 0, p))
        elif p < min_cacheable:
            out.append((0, 0, p))
        elif prev >= min_cacheable and p >= prev:
            out.append((prev, p - prev, 0))
        else:
            r = min(fallback_read, p)
            out.append((r, p - r, 0))
        prev = p
    return out


def recorded_first_split(call: dict) -> tuple:
    return (int(call.get("cache_read_tokens") or 0), int(call.get("cache_write_tokens") or 0),
            int(call.get("uncached_input_tokens") or 0))


def first_split_from_totals(prompts: list[int], read_total: int, write_total: int) -> tuple:
    """A recorded run that kept only per-task totals (scripts/cloud_redo.py rows: per-call input tokens, task-level
    read / write / uncached) under the EARLIER placement (system + last user message, the last user message being the
    request before the first call): every call after the first reads exactly what the first call cached (R1 + W1) and
    writes nothing, so write_total = W1 and read_total = R1 + (n - 1)(R1 + W1). Returns the first call's
    (R1, W1, U1); raises when the totals do not fit that placement."""
    n = len(prompts)
    w1 = write_total
    r1 = (read_total - (n - 1) * w1) / n
    u1 = prompts[0] - r1 - w1
    if r1 < -0.5 or u1 < -0.5:
        raise ValueError(f"totals read {read_total} write {write_total} do not fit the earlier placement for {prompts}")
    return (max(r1, 0.0), float(w1), max(u1, 0.0))


def split_cost(rates: dict, split: Iterable[tuple], prompts: list[int], outputs: list[int]) -> float:
    """USD of a split at rates {"input", "cached_input", "cache_write", "output"} per 1M tokens, or a rates dict with
    "short"/"long"/"long_context_above" (scripts/cloud_redo.py::model_rates shape: the long rates apply per call above
    the boundary)."""
    total = 0.0
    for (r, w, u), p, o in zip(split, prompts, outputs):
        above = rates.get("long_context_above")
        rt = rates if "input" in rates else (rates["long"] if above is not None and p > above else rates["short"])
        total += (u * rt["input"] + w * rt["cache_write"] + r * rt["cached_input"] + o * rt["output"]) / 1e6
    return total


def reprice_recorded_calls(calls: list[dict], rates: dict, min_cacheable: int, first: tuple | None = None) -> dict:
    """Reprice one recorded conversation's calls ([{input_tokens, output_tokens, (cache_* fields)}] in order) under
    the rolling breakpoint. first: the first call's split (default: the first call's recorded split when it carries
    the cache fields, else a cold cache). Returns {usd, read, write, uncached, split}."""
    prompts = [int(c["input_tokens"]) for c in calls]
    outputs = [int(c.get("output_tokens") or 0) for c in calls]
    if first is None and calls and "cache_read_tokens" in calls[0]:
        first = recorded_first_split(calls[0])
    split = rolling_split(prompts, min_cacheable, first)
    return {"usd": split_cost(rates, split, prompts, outputs), "read": sum(s[0] for s in split),
            "write": sum(s[1] for s in split), "uncached": sum(s[2] for s in split), "split": split}
