"""analysis/cloud_task_cost.py: cloud USD per real agent task, three caching cases, two price tiers."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import cloud_task_cost as ct  # noqa: E402

PRICES = {"models": {"m": {"min_cacheable_tokens": 100, "standard": {
    "short": {"input": 1.0, "cached_input": 0.1, "cache_write": 1.25, "output": 5.0},
    "long": {"input": 1.0, "cached_input": 0.1, "cache_write": 1.25, "output": 5.0}}}}}


def test_none_bills_every_resent_prompt_uncached():
    usd = ct.task_cost(PRICES, "m", [1000, 1500, 2000], 10, "none")
    assert usd == pytest.approx(((1000 + 1500 + 2000) * 1.0 + 3 * 10 * 5.0) / 1e6)


def test_best_reads_resent_prefix_and_writes_only_new_tokens():
    usd = ct.task_cost(PRICES, "m", [1000, 1500], 0, "best")
    # step 1: 1000 written; step 2: 1000 read, 500 written
    assert usd == pytest.approx((1000 * 1.25 + 1000 * 0.1 + 500 * 1.25) / 1e6)


def test_realistic_hit_rate_splits_resent_prefix():
    usd = ct.task_cost(PRICES, "m", [1000, 1500], 0, "realistic", hit_rate=0.7)
    assert usd == pytest.approx((1000 * 1.25 + 700 * 0.1 + (500 + 300) * 1.25) / 1e6)
    assert ct.task_cost(PRICES, "m", [1000, 1500], 0, "realistic", hit_rate=1.0) == pytest.approx(
        ct.task_cost(PRICES, "m", [1000, 1500], 0, "best"))


def test_prompt_under_min_cacheable_is_plain_input():
    assert ct.task_cost(PRICES, "m", [50, 80], 0, "best") == pytest.approx(130 * 1.0 / 1e6)


def test_case_ordering_and_summary_on_real_files():
    res = ct.compute(REPO)
    assert set(res["sets"]) == {"nebius-rebench-openhands", "SWE-Gym"}
    for e in res["sets"].values():
        for m in ct.TIERS:
            u = e["usd"]
            assert u[(m, "best")]["median"] <= u[(m, "realistic")]["median"] <= u[(m, "none")]["median"]
            assert u[(m, "none")]["median"] <= u[(m, "none")]["p90"]
        assert e["usd"][("claude-haiku-5-5", "none")]["median"] < e["usd"][("claude-sonnet-5-5", "none")]["median"]
    txt = ct.value_text(res)
    assert "STAND-IN" in txt and "ASSUMPTION" in txt and "priced, not executed" in txt
