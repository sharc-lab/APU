"""ROUTE-1 planner: config choice at session start + planned migration turn + delta-hat update."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from seam.route1.device import CONFIG_SET, MARGIN_M, WINDOW_HEADROOM, Config, Constants, ttft_s
from seam.route1.guard import SLO_S


def predicted_cap(cfg: Config, c: Constants, n_new: int, m: float = MARGIN_M) -> float:
    """Largest admissible n_ctx: memory (margin on resident configs only) and the exact window cap."""
    k = c.k[cfg.prec]
    if cfg.resident:
        step = n_new if cfg.chunk is None else min(n_new, cfg.chunk)
        mem = (c.B * (1.0 - m) - c.a * step) / k
    elif cfg.chunk is None:
        mem = c.B / (k + c.a)
    else:
        mem = (c.B - c.a * cfg.chunk) / k
    return min(mem, float(c.W - WINDOW_HEADROOM))


@dataclass
class Plan:
    config: Config
    cap: float
    delta_hat: float
    migrate_turn: int  # 1-based; first turn predicted to exceed the cap


def predicted_local_turns(c1: int, delta_hat: float, cap: float) -> int:
    if c1 > cap:
        return 0
    return int((cap - c1) // max(delta_hat, 1.0)) + 1


def choose(
    c1: int,
    delta_hat: float,
    c: Constants,
    params: dict[str, float],
    configs: Sequence[Config] = CONFIG_SET,
    m: float = MARGIN_M,
) -> Plan:
    """Maximize predicted local turns; ties go to the earlier (default-first) config."""
    best: Plan | None = None
    best_n = -1
    for cfg in configs:
        if not cfg.passed_xq:
            continue
        cap = predicted_cap(cfg, c, int(delta_hat), m)
        n = predicted_local_turns(c1, delta_hat, cap)
        # SLO: drop turns whose predicted TTFT exceeds 10 s
        n_ok = 0
        for i in range(n):
            if ttft_s(cfg, int(c1 + delta_hat * i), int(delta_hat), i == 0, params) > SLO_S:
                break
            n_ok += 1
        if n_ok > best_n:
            best_n = n_ok
            best = Plan(cfg, cap, delta_hat, n_ok + 1)
    if best is None:
        raise ValueError("no config passed X-Q")
    return best


def update(
    plan: Plan, c1: int, observed_prompts: Sequence[int], c: Constants, m: float = MARGIN_M
) -> Plan:
    """Re-estimate delta-hat from observed growth and re-plan the migration turn."""
    if len(observed_prompts) >= 2:
        plan.delta_hat = (observed_prompts[-1] - observed_prompts[0]) / (len(observed_prompts) - 1)
    plan.cap = predicted_cap(plan.config, c, int(plan.delta_hat), m)
    plan.migrate_turn = predicted_local_turns(c1, plan.delta_hat, plan.cap) + 1
    return plan
