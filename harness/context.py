"""Synthetic filler context builder.

Generates semantically inert prose (numbered administrative log entries) to
reach a target token depth without influencing probe answers.

Filler mode:
  unlabelled (default) — filler is prepended with no framing, banner, or
      instruction referring to it. The model receives filler as though it were
      ordinary accumulated context, which is the condition the degradation study
      intends to measure. Labelled filler would instead measure whether the
      model follows an instruction to ignore a block, conflating instruction-
      following with context degradation.
  labelled — filler is wrapped in <background_context> tags with an explicit
      "this is irrelevant" instruction. Retained as a comparison arm only.

All callers must pass filler_mode explicitly or accept the default (unlabelled).
"""

from __future__ import annotations

import random
from typing import Callable

FILLER_MODES = ("unlabelled", "labelled")
DEFAULT_FILLER_MODE = "unlabelled"

# Empirically calibrated from smoke-test data (2026-08-13):
# At target=4096, api/chat prompt_eval_count yielded 3256 filler tokens from
# 16384 chars (4*4096), giving 16384/3256 = 5.03 chars/token for this template.
# The old value of 4.0 caused a 20% undershoot.
_CHARS_PER_TOKEN = 5.03

_TEMPLATE = (
    "Administrative log entry {n}: The oversight committee reviewed all "
    "submitted documentation for compliance period {n} and confirmed that "
    "operational metrics remained within established baseline parameters. "
    "No anomalies were recorded in district {n} during the reference interval. "
    "Budget allocations for cycle {n} were processed according to standing "
    "procedure without escalation. Routine maintenance of infrastructure "
    "segment {n} was completed on schedule and filed under reference {n}. "
)


def build_filler(
    target_tokens: int,
    seed: int = 42,
    count_fn: Callable[[str], int] | None = None,
) -> str:
    """Return filler text targeting target_tokens.

    If count_fn is provided (a callable that returns the actual token count of
    a string), the result is iteratively adjusted until within 2% of
    target_tokens or until 4 iterations are exhausted, whichever comes first.
    Without count_fn the static _CHARS_PER_TOKEN estimate is used directly.
    """
    if target_tokens <= 0:
        return ""

    rng = random.Random(seed)
    start = rng.randint(10_000, 99_999)

    # Pre-generate 150% of estimated chars so trim/extend doesn't need
    # a second generation pass.
    needed_chars = int(target_tokens * _CHARS_PER_TOKEN * 1.5)
    chunks: list[str] = []
    total = 0
    i = start
    while total < needed_chars:
        chunk = _TEMPLATE.format(n=i)
        chunks.append(chunk)
        total += len(chunk)
        i += 1
    raw = "".join(chunks)

    target_chars = min(int(target_tokens * _CHARS_PER_TOKEN), len(raw))
    filler = raw[:target_chars].strip()

    if count_fn is None:
        return filler

    # Iterative refinement toward 2% tolerance.
    for _ in range(4):
        actual = count_fn(filler)
        if actual <= 0:
            break
        if abs(actual - target_tokens) / target_tokens <= 0.02:
            break
        new_chars = min(int(len(filler) * target_tokens / actual), len(raw))
        filler = raw[:new_chars].strip()

    return filler


def wrap_prompt(
    filler: str,
    probe_prompt: str,
    filler_mode: str = DEFAULT_FILLER_MODE,
) -> str:
    """Return the full prompt for a given filler and mode.

    unlabelled: filler + blank line + probe_prompt. No framing.
    labelled:   filler in <background_context> tags with an ignore instruction.
    """
    if filler_mode not in FILLER_MODES:
        raise ValueError(f"filler_mode must be one of {FILLER_MODES}, got {filler_mode!r}")
    if not filler:
        return probe_prompt
    if filler_mode == "unlabelled":
        return f"{filler}\n\n{probe_prompt}"
    # labelled
    return (
        "<background_context>\n"
        f"{filler}\n"
        "</background_context>\n\n"
        "The background context above is administrative filler unrelated to the "
        "question below. Answer using only your own knowledge.\n\n"
        f"{probe_prompt}"
    )
