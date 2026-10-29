"""Run every policy on every session; per-session outputs and $ per 1,000 sessions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from seam.route1 import policies as pol
from seam.route1.device import I8_DEFAULT, I8_RESIDENT, U4_C256_RESIDENT, Constants
from seam.route1.prices import PriceSheet
from seam.route1.session import Session

PolicyFn = Callable[[Session, PriceSheet, Constants], pol.Result]


def default_policies() -> dict[str, PolicyFn]:
    return {
        "all-cloud": lambda s, sh, _c: pol.all_cloud(s, sh),
        "D1-stateless": lambda s, sh, c: pol.d1(s, sh, c, name="D1-stateless"),
        "D1-hardfail": lambda s, sh, c: pol.d1(s, sh, c, timeout_s=None, name="D1-hardfail"),
        "D1-resident": lambda s, sh, c: pol.d1(s, sh, c, cfg=I8_RESIDENT, name="D1-resident"),
        "D1-nb": lambda s, sh, c: pol.d1(s, sh, c, timeout_s=None, sticky=False, name="D1-nb"),
        "D2": lambda s, sh, c: pol.d2(s, sh, c, timeout_s=None),
        "A0": lambda s, sh, c: pol.guard_policy(s, sh, c),
        "ROUTE-1": lambda s, sh, c: pol.route1(s, sh, c),
        "A1": lambda s, sh, c: pol.d1(s, sh, c, cfg=U4_C256_RESIDENT, name="A1"),
        "A2": lambda s, sh, c: pol.route1(
            s,
            sh,
            c,
            configs=(I8_DEFAULT,),
            plan_c=Constants(B=c.B, a=0, k=c.k, W=c.W),
            slo=False,
            name="A2",
        ),
        "B-4K": lambda s, sh, _c: pol.static_threshold(s, sh, 4096),
        "B-8K": lambda s, sh, _c: pol.static_threshold(s, sh, 8192),
        "Oracle": lambda s, sh, c: pol.oracle(s, sh, c),
    }


def run_all(
    sessions: Sequence[Session],
    sheets: Sequence[PriceSheet],
    c: Constants,
    policies: dict[str, PolicyFn] | None = None,
) -> dict[str, dict[str, list[pol.Result]]]:
    """{sheet name: {policy: [Result per session]}}."""
    pols = policies or default_policies()
    return {sh.name: {k: [f(s, sh, c) for s in sessions] for k, f in pols.items()} for sh in sheets}


def usd_per_1000(results: Sequence[pol.Result]) -> float:
    return 1000.0 * sum(r.usd for r in results) / max(len(results), 1)
