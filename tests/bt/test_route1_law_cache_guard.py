"""ROUTE-1 device law, cache rules and guard (synthetic only)."""

from __future__ import annotations

import pytest

from seam.route1 import device as dev
from seam.route1.cache_sim import CloudCache
from seam.route1.guard import admit
from seam.route1.prices import HAIKU, SONNET, reingest_penalty
from seam.route1.session import synthetic

C = dev.Constants(B=3.27e9)


def test_law_and_kv_only_phantom() -> None:
    assert round(dev.l_unchunked(C, "i8")) == 19_054
    assert round(dev.l_kv_only(C, "i8")) == 38_884
    assert dev.l_kv_only(C, "i8") / dev.l_unchunked(C, "i8") == pytest.approx(2.04, abs=0.01)
    assert (round(dev.l_unchunked(C, "f16")), round(dev.l_kv_only(C, "f16"))) == (13_916, 22_176)
    assert (round(dev.l_unchunked(C, "u4")), round(dev.l_kv_only(C, "u4"))) == (24_267, 69_233)
    assert round(dev.l_session_total(C, "i8", 1500, None)) == 37_323


@pytest.mark.parametrize(("tokens", "usd"), [(20_500, 0.0492), (47_500, 0.114), (119_000, 0.2856)])
def test_reingest_penalty(tokens: int, usd: float) -> None:
    assert reingest_penalty(SONNET, tokens) == pytest.approx(usd, abs=1e-4)


def _warm_after(gap_s: float, **kw: object) -> int:
    cache = CloudCache()
    cache.call(SONNET, 20_000, 300, 0.0)
    row = cache.call(SONNET, 21_500, 300, gap_s, **kw)  # type: ignore[arg-type]
    return row.cache_read


def test_ttl_301_cold_299_warm() -> None:
    assert _warm_after(301.0) == 0
    assert _warm_after(299.0) == 20_000


def test_tools_change_and_model_switch_are_cold() -> None:
    assert _warm_after(1.0, tools_hash="tools-v2") == 0
    cache = CloudCache()
    cache.call(SONNET, 20_000, 300, 0.0)
    assert cache.call(HAIKU, 21_500, 300, 1.0).cache_read == 0


def test_short_prompt_uncached() -> None:
    row = CloudCache().call(SONNET, 400, 10, 0.0)
    assert (row.uncached, row.cache_write, row.cache_read) == (400, 0, 0)


def test_lookback_20_blocks() -> None:
    assert _warm_after(1.0, new_blocks=20) == 20_000
    assert _warm_after(1.0, new_blocks=21) == 0


def test_hit_refreshes_ttl() -> None:
    cache = CloudCache()
    cache.call(SONNET, 20_000, 0, 0.0)
    assert cache.call(SONNET, 20_500, 0, 250.0).cache_read == 20_000
    assert cache.call(SONNET, 21_000, 0, 500.0).cache_read == 20_500  # 250 s after the refresh


def test_haiku_100k_first_turn_66() -> None:
    s = synthetic(4000, 1500, 70, 300)
    first = next(i + 1 for i, t in enumerate(s) if t.prompt_tokens > 100_000)
    assert first == 66
    assert HAIKU.multiplier(100_001) == 5.0 and HAIKU.multiplier(100_000) == 1.0


def test_guard() -> None:
    ok, why, _ = admit(20_500, 20_500, 0, dev.I8_DEFAULT, C, 0.0)
    assert not ok and why == "memory"
    ok, _, _ = admit(20_500, 1500, 0, dev.U4_C256_RESIDENT, C, 0.0)
    assert ok
    ok, why, _ = admit(40_000, 1500, 512, dev.U4_C256_RESIDENT, C, 0.0)
    assert not ok and why == "window"
