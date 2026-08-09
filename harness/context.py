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

FILLER_MODES = ("unlabelled", "labelled")
DEFAULT_FILLER_MODE = "unlabelled"

# Rough estimate for the prose below: ~4 chars per token for English
_CHARS_PER_TOKEN = 4

_TEMPLATE = (
    "Administrative log entry {n}: The oversight committee reviewed all "
    "submitted documentation for compliance period {n} and confirmed that "
    "operational metrics remained within established baseline parameters. "
    "No anomalies were recorded in district {n} during the reference interval. "
    "Budget allocations for cycle {n} were processed according to standing "
    "procedure without escalation. Routine maintenance of infrastructure "
    "segment {n} was completed on schedule and filed under reference {n}. "
)


def build_filler(target_tokens: int, seed: int = 42) -> str:
    """Return ~target_tokens tokens of semantically neutral prose."""
    if target_tokens <= 0:
        return ""
    target_chars = target_tokens * _CHARS_PER_TOKEN
    rng = random.Random(seed)
    start = rng.randint(10_000, 99_999)
    chunks: list[str] = []
    total = 0
    i = start
    while total < target_chars:
        chunk = _TEMPLATE.format(n=i)
        chunks.append(chunk)
        total += len(chunk)
        i += 1
    return "".join(chunks)[:target_chars].strip()


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
