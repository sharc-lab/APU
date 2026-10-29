"""Session = list of turns; synthetic generator only (no trace rows)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    prompt_tokens: int  # full transcript the call sends (Qwen3 tokens)
    new_tokens: int  # growth since the previous turn (C1 on the first turn)
    output_tokens: int
    t_start_s: float
    n_tool_blocks: int = 1  # cache block positions this turn appends
    tools_hash: str = "tools-v1"


Session = list[Turn]


def synthetic(
    c1: int,
    delta: int,
    n: int,
    out: int,
    gaps_s: float | Sequence[float] = 0.0,
    n_tool_blocks: int = 1,
) -> Session:
    """Prompt of turn t (1-based) = c1 + delta*(t-1); gaps_s is the idle gap before each turn after
    the first (a float applies to every gap)."""
    gaps = [float(gaps_s)] * max(n - 1, 0) if isinstance(gaps_s, (int, float)) else list(gaps_s)
    if len(gaps) != max(n - 1, 0):
        raise ValueError(f"need {n - 1} gaps, got {len(gaps)}")
    turns: Session = []
    t = 0.0
    for i in range(n):
        if i > 0:
            t += gaps[i - 1]
        turns.append(
            Turn(
                prompt_tokens=c1 + delta * i,
                new_tokens=c1 if i == 0 else delta,
                output_tokens=out,
                t_start_s=t,
                n_tool_blocks=n_tool_blocks,
            )
        )
    return turns
