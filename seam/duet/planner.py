"""DUET loop A (session planner) and loop B (per-turn step). Fixed, non-learned, feasibility only.

Loop A enumerates config x payload x migration turn over three growth scenarios (P10/P50/P90) with prefix
sums. Loop B picks the cheapest admissible action for one turn, with hysteresis on side changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from seam.duet import params as prm
from seam.duet.device import cold_mem_ok, session_cap
from seam.duet.ttft import cold, ext

SCENARIOS: tuple[tuple[float, float], ...] = (
    (0.8, 0.25),
    (1.0, 0.5),
    (1.25, 0.25),
)  # (scale, weight)
PAYLOADS: tuple[str, ...] = ("full", "compact")


@dataclass(frozen=True)
class PlanChoice:
    prec: str
    payload: str
    migrate_turn: int  # 1-based first cloud turn
    expected_usd: float


def _cloud_segment(prompts: Sequence[int], outs: Sequence[int], start: int, price: prm.Price,
                   payload: str) -> float:  # fmt: skip
    """$ of turns start.. in the cloud: cold write at start, then read prior + write delta + output."""
    if start >= len(prompts):
        return 0.0
    first = prompts[start]
    usd = 0.0
    if payload == "compact":
        prev = prompts[start - 1] if start > 0 else 0
        kept = int(prm.COMPACT_KEEP * prev)
        usd += (prev * prm.EXECUTOR.input + kept * prm.EXECUTOR.output) / 1e6
        shift = prev - kept
    else:
        shift = 0
    usd += ((first - shift) * price.write_5m + outs[start] * price.output) / 1e6
    for t in range(start + 1, len(prompts)):
        usd += ((prompts[t - 1] - shift) * price.read + (prompts[t] - prompts[t - 1]) * price.write_5m
                + outs[t] * price.output) / 1e6  # fmt: skip
    return usd


def last_admissible(prompts: Sequence[int], deltas: Sequence[int], prec: str, slo: bool) -> int:
    """Number of leading turns the device can serve (memory with margin, window, and the SLO)."""
    n = 0
    for t, (p, d) in enumerate(zip(prompts, deltas, strict=True)):
        if p > session_cap(prec, d, prm.MARGIN_M):
            break
        if slo:
            tt = cold(p) if t == 0 else ext(d, p)
            if cold_mem_ok(p, prec):
                tt = min(tt, cold(p))
            if tt > prm.SLO_S:
                break
        n = t + 1
    return n


def plan(c1: int, delta: float, n_turns: int, out: int, price: prm.Price, slo: bool,
         precs: Sequence[str] = ("i8", "u4")) -> PlanChoice:  # fmt: skip
    """Loop A: minimize expected $ over config x payload x migration turn (migrate no later than the
    last admissible turn in each scenario)."""
    best: PlanChoice | None = None
    for prec in precs:
        for payload in PAYLOADS:
            for tm in range(1, n_turns + 2):
                exp_usd = 0.0
                feasible = True
                for scale, w in SCENARIOS:
                    d = max(int(delta * scale), 1)
                    prompts = [c1 + d * i for i in range(n_turns)]
                    deltas = [c1] + [d] * (n_turns - 1)
                    if tm - 1 > last_admissible(prompts, deltas, prec, slo):
                        feasible = False
                        break
                    exp_usd += w * _cloud_segment(prompts, [out] * n_turns, tm - 1, price, payload)
                if feasible and (best is None or exp_usd < best.expected_usd - 1e-12):
                    best = PlanChoice(prec, payload, tm, exp_usd)
    assert best is not None
    return best


ACTIONS = ("EXTEND", "REPREFILL", "EXCURSION_STUB", "MIGRATE")


def step(n_new: int, tool: int, pd: int, prec: str, price: prm.Price, slo: bool,
         side_local: bool = True, hyst_usd: float = 0.0) -> str:  # fmt: skip
    """Loop B: cheapest admissible action for one turn (cloud $ + 0 for local), with hysteresis."""
    p = pd + n_new + tool
    cap = session_cap(prec, n_new + tool, prm.MARGIN_M)
    adm: list[tuple[float, str]] = []
    if p <= cap:
        if not slo or ext(n_new + tool, p) <= prm.SLO_S:
            adm.append((0.0, "EXTEND"))
        if cold_mem_ok(p, prec) and (not slo or cold(p) <= prm.SLO_S):
            adm.append((0.0, "REPREFILL"))
    if tool > 0:
        pstub = pd + n_new + prm.STUB_TOKENS
        if pstub <= session_cap(prec, prm.SMALL_TURN, prm.MARGIN_M) and (
            not slo or ext(prm.SMALL_TURN, pstub + prm.SMALL_TURN) <= prm.SLO_S
        ):
            adm.append(((n_new + tool) * price.write_5m / 1e6, "EXCURSION_STUB"))
    adm.append((p * price.write_5m / 1e6, "MIGRATE"))
    cost, act = min(adm)
    if side_local and act == "MIGRATE" and len(adm) > 1 and cost - adm[0][0] < hyst_usd:
        act = adm[0][1]
    return act
