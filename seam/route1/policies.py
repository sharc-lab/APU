"""Controls, ROUTE-1, oracle and ablations on one synthetic session.

Every policy returns a Result. Cloud calls go through one CloudCache ledger per session, so prices and
cache state are identical across policies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from seam.route1.cache_sim import CloudCache
from seam.route1.device import (
    I8_DEFAULT,
    TTFT_PARAMS,
    Config,
    Constants,
    true_ceiling,
    ttft_s,
)
from seam.route1.guard import SLO_S, admit
from seam.route1.planner import choose, update
from seam.route1.prices import PriceSheet
from seam.route1.session import Session

DECLARED_WINDOW = 40_960


@dataclass
class Result:
    policy: str
    usd: float = 0.0
    local_turns: int = 0
    first_cloud_turn: int | None = None  # 1-based
    failed_attempts: int = 0
    timeouts: int = 0
    silent_overruns: int = 0
    wasted_s: float = 0.0
    local_j: float = 0.0
    wait_s: float = 0.0
    tokens: dict[str, int] = field(
        default_factory=lambda: {"uncached": 0, "cache_write": 0, "cache_read": 0, "output": 0}
    )
    config: str = ""


class _Run:
    def __init__(self, name: str, session: Session, sheet: PriceSheet, p: dict[str, float]) -> None:
        self.r = Result(name)
        self.s = session
        self.sheet = sheet
        self.cache = CloudCache()
        self.p = p

    def cloud(self, t: int) -> None:
        turn = self.s[t]
        row = self.cache.call(
            self.sheet, turn.prompt_tokens, turn.output_tokens, turn.t_start_s, turn.tools_hash,
            turn.n_tool_blocks,
        )  # fmt: skip
        self.r.usd += row.usd
        for k, v in (("uncached", row.uncached), ("cache_write", row.cache_write),
                     ("cache_read", row.cache_read), ("output", row.output)):  # fmt: skip
            self.r.tokens[k] += v
        if self.r.first_cloud_turn is None:
            self.r.first_cloud_turn = t + 1
        self.r.wait_s += self.p["cloud_ttft_s"] + turn.output_tokens / self.p["cloud_tok_s"]

    def local(self, t: int, ttft: float, c: Constants) -> None:
        turn = self.s[t]
        secs = ttft + turn.output_tokens / self.p["decode_tok_s"]
        self.r.local_turns += 1
        self.r.wait_s += secs
        self.r.local_j += secs * self.p["package_w"]
        if turn.prompt_tokens + turn.output_tokens > c.W:
            self.r.silent_overruns += 1

    def fail(self) -> None:
        self.r.failed_attempts += 1
        self.r.wasted_s += self.p["time_to_fail_s"]
        self.r.wait_s += self.p["time_to_fail_s"]

    def timeout(self) -> None:
        self.r.timeouts += 1
        self.r.wasted_s += SLO_S
        self.r.wait_s += SLO_S


def all_cloud(session: Session, sheet: PriceSheet, p: dict[str, float] = TTFT_PARAMS) -> Result:
    run = _Run("all-cloud", session, sheet, p)
    for t in range(len(session)):
        run.cloud(t)
    return run.r


def d1(
    session: Session,
    sheet: PriceSheet,
    c: Constants,
    *,
    cfg: Config = I8_DEFAULT,
    timeout_s: float | None = SLO_S,
    sticky: bool = True,
    retries: int = 0,
    precheck_window: int | None = None,
    name: str = "D1",
    p: dict[str, float] = TTFT_PARAMS,
) -> Result:
    """Local-first default. Residency comes from cfg (M1 decides which D1 is the control).

    precheck_window set = D2 (declared-window pre-check, sticky)."""
    run = _Run(name, session, sheet, p)
    run.r.config = cfg.name
    in_cloud = False
    for t, turn in enumerate(session):
        if in_cloud:
            run.cloud(t)
            continue
        if (
            precheck_window is not None
            and turn.prompt_tokens + turn.output_tokens > precheck_window
        ):
            in_cloud = True
            run.cloud(t)
            continue
        ttft = ttft_s(cfg, turn.prompt_tokens, turn.new_tokens, t == 0, p)
        ok = False
        for _ in range(retries + 1):
            if timeout_s is not None and ttft > timeout_s:
                run.timeout()
                break
            if turn.prompt_tokens > true_ceiling(c, cfg, turn.new_tokens):
                run.fail()
                continue
            ok = True
            break
        if ok:
            run.local(t, ttft, c)
        else:
            run.cloud(t)
            in_cloud = sticky
    return run.r


def d2(
    session: Session, sheet: PriceSheet, c: Constants, margin: float = 0.0, **kw: object
) -> Result:
    window = int(DECLARED_WINDOW * (1.0 - margin))
    return d1(session, sheet, c, precheck_window=window, name="D2", **kw)  # type: ignore[arg-type]


def guard_policy(
    session: Session,
    sheet: PriceSheet,
    c: Constants,
    cfg: Config = I8_DEFAULT,
    name: str = "A0",
    slo: bool = False,
    p: dict[str, float] = TTFT_PARAMS,
) -> Result:
    """A0 = HYB-1 P2-U: the ceiling guard on a fixed config, sticky on the first reject.

    HYB-1 P2-U is a feasibility (memory + window) guard, so the SLO term is off by default."""
    run = _Run(name, session, sheet, p)
    run.r.config = cfg.name
    in_cloud = False
    for t, turn in enumerate(session):
        ttft = ttft_s(cfg, turn.prompt_tokens, turn.new_tokens, t == 0, p)
        if not in_cloud:
            ok, _, _ = admit(turn.prompt_tokens, turn.new_tokens, 0, cfg, c, ttft if slo else 0.0)
            if ok and turn.prompt_tokens <= true_ceiling(c, cfg, turn.new_tokens):
                run.local(t, ttft, c)
                continue
            if ok:
                run.fail()
            in_cloud = True
        run.cloud(t)
    return run.r


def route1(
    session: Session,
    sheet: PriceSheet,
    c: Constants,
    configs: tuple[Config, ...] | None = None,
    plan_c: Constants | None = None,
    delta_prior: float | None = None,
    sticky_on_fail: bool = False,
    slo: bool = True,
    name: str = "ROUTE-1",
    p: dict[str, float] = TTFT_PARAMS,
) -> Result:
    """Config at session start, planned migration, per-call guard backstop.

    plan_c lets an ablation plan with wrong constants (A2: a = 0) while the device truth stays c.
    slo=False evaluates memory and window only (A2, the KV-only memory ablation)."""
    pp = p if slo else {**p, "cold_s_per_token": 0.0, "resident_turn_s": 0.0}
    pc = plan_c or c
    c1 = session[0].prompt_tokens
    dh = (
        delta_prior
        if delta_prior is not None
        else float(session[1].new_tokens if len(session) > 1 else 0)
    )
    plan = choose(c1, dh, pc, pp, configs) if configs is not None else choose(c1, dh, pc, pp)
    run = _Run(name, session, sheet, p)
    run.r.config = plan.config.name
    cfg = plan.config
    migrated = False
    seen: list[int] = []
    for t, turn in enumerate(session):
        seen.append(turn.prompt_tokens)
        if t > 0:
            update(plan, c1, seen[:-1] if len(seen) > 2 else seen, pc)
        if migrated:
            run.cloud(t)
            continue
        ttft = ttft_s(cfg, turn.prompt_tokens, turn.new_tokens, t == 0, p)
        ok, _, _ = admit(turn.prompt_tokens, turn.new_tokens, 0, cfg, pc, ttft if slo else 0.0)
        if t + 1 >= plan.migrate_turn or not ok:
            migrated = True
            run.cloud(t)
            continue
        if turn.prompt_tokens > true_ceiling(c, cfg, turn.new_tokens):
            run.fail()
            run.cloud(t)
            migrated = sticky_on_fail
            continue
        run.local(t, ttft, c)
    return run.r


def static_threshold(
    session: Session, sheet: PriceSheet, threshold: int, p: dict[str, float] = TTFT_PARAMS
) -> Result:
    run = _Run(f"B-{threshold}", session, sheet, p)
    in_cloud = False
    for t, turn in enumerate(session):
        in_cloud = in_cloud or turn.prompt_tokens > threshold
        if in_cloud:
            run.cloud(t)
        else:
            run.local(t, 0.0, Constants())
    return run.r


def random_router(
    session: Session, sheet: PriceSheet, cloud_share: float, seed: int = 20261031,
    p: dict[str, float] = TTFT_PARAMS,
) -> Result:  # fmt: skip
    rng = random.Random(seed)
    run = _Run("B-rand", session, sheet, p)
    for t in range(len(session)):
        if rng.random() < cloud_share:
            run.cloud(t)
        else:
            run.local(t, 0.0, Constants())
    return run.r


def oracle(
    session: Session,
    sheet: PriceSheet,
    c: Constants,
    configs: tuple[Config, ...] | None = None,
    p: dict[str, float] = TTFT_PARAMS,
) -> Result:
    """Perfect per-turn feasibility; minimum $ over configs and migration turns, no failure, no SLO miss."""
    from seam.route1.device import CONFIG_SET

    best: Result | None = None
    for cfg in configs or CONFIG_SET:
        last_ok = 0
        for t, turn in enumerate(session):
            ttft = ttft_s(cfg, turn.prompt_tokens, turn.new_tokens, t == 0, p)
            feasible = (
                turn.prompt_tokens <= true_ceiling(c, cfg, turn.new_tokens)
                and turn.prompt_tokens + turn.output_tokens <= c.W
                and ttft <= SLO_S
            )
            if not feasible:
                break
            last_ok = t + 1
        for tm in range(0, last_ok + 1):  # migrate before turn tm+1 (0 = all-cloud)
            run = _Run("Oracle", session, sheet, p)
            run.r.config = cfg.name
            for t, turn in enumerate(session):
                if t < tm:
                    run.local(t, ttft_s(cfg, turn.prompt_tokens, turn.new_tokens, t == 0, p), c)
                else:
                    run.cloud(t)
            if best is None or run.r.usd < best.usd - 1e-12:
                best = run.r
    assert best is not None
    return best
