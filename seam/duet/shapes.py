"""Synthetic session shapes U, H, T (ASSUMED) and a big-turn-share generator. No trace data."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DTurn:
    new: int  # user/agent text tokens (includes the previous reply)
    tool: int  # tool-output tokens appended this turn
    out: int


def _shape(c1: int, n: int, out: int, new: int, tool: int, every: int, at: int) -> list[DTurn]:
    turns = []
    for t in range(n):
        if t == 0:
            turns.append(DTurn(c1, 0, out))
        else:
            turns.append(DTurn(new, tool if (every and t % every == at) else 0, out))
    return turns


def shape_u() -> list[DTurn]:
    return _shape(4000, 30, 300, 1500, 0, 0, 0)


def shape_h() -> list[DTurn]:
    return _shape(4000, 30, 300, 700, 4000, 5, 3)


def shape_t() -> list[DTurn]:
    return _shape(8000, 60, 400, 1000, 8000, 4, 2)


def big_share(
    n: int, share: float, seed: int, c1: int = 4000, new: int = 700, tool: int = 4000
) -> list[DTurn]:
    rng = random.Random(seed)
    return [DTurn(c1, 0, 300)] + [
        DTurn(new, tool if rng.random() < share else 0, 300) for _ in range(n - 1)
    ]
