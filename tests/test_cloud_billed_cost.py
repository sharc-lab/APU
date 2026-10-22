"""Billed cost (from the provider usage fields) vs the uncached upper bound in src/cloud/client.py (operator
2026-10-22): the ledger's cost_usd and running_total_usd are BILLED, cost_usd_upper_bound and
running_total_usd_upper_bound keep the old every-input-token-uncached figure, the USD 50 cap and the spend alerts
use the billed total, the pre-call reservation stays on the upper bound.

Only a fake provider (an injected completion_fn) is used: no real API, no network. Rates come from the shared price
files (results/cloud_prices_anthropic_20261014.json), never typed here."""
from __future__ import annotations

import json

import pytest

from src.cloud import client as cc
from src.cloud.client import CloudClient, SpendCapExceeded

MODEL = "claude-sonnet-5-5"
R = cc.MODEL_PRICING[MODEL]["rates"]["short"]


def _fake(usage, seen=None, fail=False):
    def fn(*, model_id, messages, api_key, **kwargs):
        if seen is not None:
            seen.append(kwargs)
        if fail:
            raise RuntimeError("provider down")
        return {"usage": usage, "choices": [{"message": {"content": "ok"}}]}
    return fn


def _client(ledger, fn, cap=None):
    kw = {} if cap is None else {"spend_cap_usd": cap}
    return CloudClient(api_key="sk-fake", ledger_path=ledger, role_override="controller", completion_fn=fn, **kw)


def _rows(ledger):
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("{")]


def _write(ledger, rows):
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


LITELLM_USAGE = {"prompt_tokens": 10_000, "completion_tokens": 200, "cache_read_input_tokens": 6_000,
                 "cache_creation_input_tokens": 3_000, "prompt_tokens_details": {"cached_tokens": 6_000}}
WANT_BILLED = (1_000 * R["input"] + 6_000 * R["cached_input"] + 3_000 * R["cache_write"] + 200 * R["output"]) / 1e6
WANT_UPPER = (10_000 * R["input"] + 200 * R["output"]) / 1e6


def test_billed_cost_from_usage_with_cache_read_and_write():
    u = cc.usage_breakdown(LITELLM_USAGE)
    assert (u["input_tokens"], u["uncached_input_tokens"], u["cache_read_tokens"], u["cache_write_tokens"]) == \
        (10_000, 1_000, 6_000, 3_000)
    assert cc.billed_cost_usd(MODEL, u) == pytest.approx(WANT_BILLED)
    # Anthropic's own shape (input_tokens excludes the cache tokens) gives the same breakdown and cost
    raw = cc.usage_breakdown({"input_tokens": 1_000, "output_tokens": 200, "cache_read_input_tokens": 6_000,
                              "cache_creation_input_tokens": 3_000})
    assert raw == u
    assert WANT_BILLED < WANT_UPPER


def test_one_hour_writes_at_the_one_hour_rate():
    split = cc.usage_breakdown({"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 3_000,
                                "cache_creation": {"ephemeral_5m_input_tokens": 1_000,
                                                   "ephemeral_1h_input_tokens": 2_000}})
    assert cc.billed_cost_usd(MODEL, split) == pytest.approx((1_000 * R["cache_write"] + 2_000 * R["cache_write_1h"]) / 1e6)
    plain = cc.usage_breakdown({"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 3_000})
    assert cc.billed_cost_usd(MODEL, plain, "1h") == pytest.approx(3_000 * R["cache_write_1h"] / 1e6)
    msgs = [{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]
    assert cc.request_cache_ttl(msgs) == "1h" and cc.request_cache_ttl([{"role": "user", "content": "x"}]) == "5m"


def test_ledger_row_has_billed_and_upper_bound_columns(tmp_path):
    ledger = tmp_path / "l.jsonl"
    c = _client(ledger, _fake(LITELLM_USAGE))
    res = c.call(model_id=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=200)
    assert res.cost_usd == pytest.approx(WANT_BILLED) and res.cost_usd_upper_bound == pytest.approx(WANT_UPPER)
    row = _rows(ledger)[-1]
    assert row["cost_usd"] == pytest.approx(WANT_BILLED)
    assert row["cost_usd_upper_bound"] == pytest.approx(WANT_UPPER)
    assert row["running_total_usd"] == pytest.approx(WANT_BILLED)
    assert row["running_total_usd_upper_bound"] == pytest.approx(WANT_UPPER)
    assert (row["cache_read_tokens"], row["cache_write_tokens"], row["uncached_input_tokens"]) == (6_000, 3_000, 1_000)
    assert row["cost_basis"] == "billed_from_usage" and row["input_tokens"] == 10_000
    assert c.running_total_usd == pytest.approx(WANT_BILLED)
    assert c.running_total_usd_upper_bound == pytest.approx(WANT_UPPER)
    # a second client reads both totals back
    c2 = _client(ledger, _fake(LITELLM_USAGE))
    assert (c2.running_total_usd, c2.running_total_usd_upper_bound) == pytest.approx((WANT_BILLED, WANT_UPPER))


def test_no_usage_falls_back_to_upper_bound(tmp_path):
    ledger = tmp_path / "l.jsonl"
    res = _client(ledger, _fake(None)).call(model_id=MODEL, messages=[{"role": "user", "content": "a b"}],
                                            expected_output_tokens=10)
    row = _rows(ledger)[-1]
    assert row["cost_basis"] == "upper_bound_no_usage"
    assert row["cost_usd"] == pytest.approx(row["cost_usd_upper_bound"]) == pytest.approx(res.cost_usd)


def test_cap_refuses_on_billed_running_total(tmp_path):
    # upper-bound total far past the cap, billed total below it: calls are still allowed
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [{"model_id": MODEL, "cost_usd": 10.0, "cost_usd_upper_bound": 80.0}])
    c = _client(ledger, _fake(LITELLM_USAGE))
    assert c.running_total_usd == pytest.approx(10.0) and c.running_total_usd_upper_bound == pytest.approx(80.0)
    c.call(model_id=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=200)
    # billed total at the cap: refused before any request is sent
    ledger2 = tmp_path / "l2.jsonl"
    _write(ledger2, [{"model_id": MODEL, "cost_usd": 49.999, "cost_usd_upper_bound": 49.999}])
    seen = []
    c2 = _client(ledger2, _fake(LITELLM_USAGE, seen))
    with pytest.raises(SpendCapExceeded, match="billed"):
        c2.call(model_id=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=200)
    assert seen == [] and len(_rows(ledger2)) == 1


def test_mixed_old_and_new_rows(tmp_path):
    ledger = tmp_path / "l.jsonl"
    _write(ledger, [
        {"model_id": MODEL, "input_tokens": 10, "output_tokens": 1, "cost_usd": 1.0, "running_total_usd": 1.0},
        {"record": "alert", "source": "cloud_client", "threshold_fraction": 0.5, "running_total_usd": 1.0},
        {"model_id": MODEL, "cost_usd": 0.2, "cost_usd_upper_bound": 0.5, "running_total_usd": 1.2,
         "running_total_usd_upper_bound": 1.5},
    ])
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
    c = _client(ledger, _fake(LITELLM_USAGE))
    assert c.running_total_usd == pytest.approx(1.2)          # old row's cost_usd counts as billed (conservative)
    assert c.running_total_usd_upper_bound == pytest.approx(1.5)
    assert c._read_running_total() == pytest.approx(1.2)
    c.call(model_id=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=200)
    row = _rows(ledger)[-1]
    assert row["running_total_usd"] == pytest.approx(1.2 + WANT_BILLED)
    assert row["running_total_usd_upper_bound"] == pytest.approx(1.5 + WANT_UPPER)


def test_reservation_released_on_failure(tmp_path):
    ledger = tmp_path / "l.jsonl"
    c = _client(ledger, _fake(LITELLM_USAGE, fail=True), cap=0.05)
    for _ in range(5):
        with pytest.raises(RuntimeError, match="provider down"):
            c.call(model_id=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=2000)
        assert c._reserved_usd == 0.0
    assert not ledger.exists()
    # a leaked reservation (5 x the projection) would now block this call; it goes through
    c._completion_fn = _fake(LITELLM_USAGE)
    c.call(model_id=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=2000)
    assert len(_rows(ledger)) == 1 and c._reserved_usd == 0.0


def test_reservation_is_upper_bound_and_covers_cache_writes(tmp_path):
    plain = cc.reservation_cost_usd(MODEL, 1_000, 100)
    assert plain == pytest.approx(cc.estimate_cost_usd(MODEL, 1_000, 100))
    cached = cc.reservation_cost_usd(MODEL, 1_000, 100, "5m")
    assert cached == pytest.approx((1_000 * R["cache_write"] + 100 * R["output"]) / 1e6) and cached > plain
    # the reservation actually held while a cache_control request is in flight
    p = _client(tmp_path / "l.jsonl", _fake(LITELLM_USAGE)).prepare(
        model_id=MODEL, expected_output_tokens=100,
        messages=[{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}])
    assert p.reserved_usd == pytest.approx(cc.reservation_cost_usd(MODEL, p.input_tokens_est, 100, "5m"))
    p.release()


def test_alerts_on_billed_total_and_fired_alerts_stay_fired(tmp_path, capsys):
    ledger = tmp_path / "l.jsonl"
    # billed 24.99 (below 50% of 50), upper 45 (past 75% and 90%): only billed counts
    _write(ledger, [{"model_id": MODEL, "cost_usd": 24.99, "cost_usd_upper_bound": 45.0}])
    c = _client(ledger, _fake({"prompt_tokens": 1, "completion_tokens": 2_000}))
    c.call(model_id=MODEL, messages=[{"role": "user", "content": "x"}], expected_output_tokens=2_000)
    alerts = [r for r in _rows(ledger) if r.get("record") == "alert"]
    assert [a["threshold_fraction"] for a in alerts] == [0.5]
    assert alerts[0]["basis"] == "billed" and alerts[0]["running_total_usd"] == pytest.approx(c.running_total_usd)
    # a new client on the same ledger does not fire 50% again
    c2 = _client(ledger, _fake({"prompt_tokens": 1, "completion_tokens": 1}))
    c2.call(model_id=MODEL, messages=[{"role": "user", "content": "x"}], expected_output_tokens=1)
    assert [r["threshold_fraction"] for r in _rows(ledger) if r.get("record") == "alert"] == [0.5]
