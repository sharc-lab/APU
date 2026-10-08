"""Hard USD 50 total spend cap and the 50% / 75% / 90% spend alerts in src/cloud/client.py (2026-10-07).

Only a fake provider (an injected completion_fn) is used: no real API, no litellm, no network. Costs are steered
exactly by the fake's reported usage: claude-sonnet-4-5 output tokens at $15.00 / 1M, with a 1-word message
(1 input token, $0.000003) so each call costs 15e-6 * out + 3e-6 USD."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.cloud.client import (
    DEFAULT_SPEND_CAP_USD,
    HARD_SPEND_CAP_USD,
    SPEND_ALERT_FRACTIONS,
    CloudClient,
    SpendCapExceeded,
    estimate_cost_usd,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import results_digest  # noqa: E402

MODEL = "claude-sonnet-4-5"


def _out_tokens_for(usd):
    """Output tokens whose call cost (with the 1 input token) is as close to usd as possible from below."""
    return int((usd - 3e-6) / 15e-6)


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def __call__(self, *, model_id, messages, api_key, **kwargs):
        self.calls += 1
        return {"usage": {"prompt_tokens": 1, "completion_tokens": kwargs.get("fake_out", 0)},
                "choices": [{"message": {"content": "ok"}}]}


def _client(ledger, fake, cap=None):
    kw = {} if cap is None else {"spend_cap_usd": cap}
    return CloudClient(api_key="sk-fake", ledger_path=ledger, role_override="controller", completion_fn=fake, **kw)


def _spend(client, usd):
    out = _out_tokens_for(usd)
    return client.call(model_id=MODEL, messages=[{"role": "user", "content": "x"}], expected_output_tokens=out,
                       fake_out=out)


def _alerts(ledger):
    return [json.loads(line) for line in ledger.read_text().splitlines() if json.loads(line).get("record") == "alert"]


def test_cap_is_usd_50_total_and_cannot_be_raised_in_real_mode(tmp_path):
    assert HARD_SPEND_CAP_USD == 50.0 and DEFAULT_SPEND_CAP_USD == 50.0
    assert SPEND_ALERT_FRACTIONS == (0.50, 0.75, 0.90)
    assert _client(tmp_path / "l.jsonl", FakeProvider()).spend_cap_usd == 50.0
    with pytest.raises(ValueError, match="hard cap"):
        _client(tmp_path / "l.jsonl", FakeProvider(), cap=200.0)
    assert _client(tmp_path / "l.jsonl", FakeProvider(), cap=10.0).spend_cap_usd == 10.0  # lowering is allowed
    # stub mode spends nothing, so a sweep may still pass any cap
    assert CloudClient(api_key=None, spend_cap_usd=200.0, ledger_path=tmp_path / "s.jsonl").stub_mode


def test_request_that_would_exceed_the_cap_is_refused_before_sending(tmp_path):
    ledger, fake = tmp_path / "ledger.jsonl", FakeProvider()
    c = _client(ledger, fake)
    _spend(c, 49.0)
    assert fake.calls == 1
    with pytest.raises(SpendCapExceeded):
        _spend(c, 1.5)  # 49 + 1.5 > 50
    assert fake.calls == 1  # the provider was never called for the refused request
    assert c.running_total_usd == pytest.approx(49.0, abs=1e-4)
    _spend(c, 0.99)  # still under the cap: allowed
    assert fake.calls == 2


def test_each_alert_fires_exactly_once_at_its_threshold(tmp_path, capsys):
    ledger, fake = tmp_path / "ledger.jsonl", FakeProvider()
    c = _client(ledger, fake)
    _spend(c, 24.9)  # 49.8%
    assert _alerts(ledger) == []
    _spend(c, 0.2)  # 50.2% -> 50% alert
    assert [a["threshold_fraction"] for a in _alerts(ledger)] == [0.5]
    _spend(c, 1.0)  # still between 50% and 75%: nothing new
    assert len(_alerts(ledger)) == 1
    _spend(c, 11.5)  # 37.6 = 75.2% -> 75% alert
    assert [a["threshold_fraction"] for a in _alerts(ledger)] == [0.5, 0.75]
    _spend(c, 7.3)  # 44.9 = 89.8%: nothing new
    assert len(_alerts(ledger)) == 2
    _spend(c, 0.2)  # 45.1 = 90.2% -> 90% alert
    _spend(c, 1.0)  # above 90%: nothing new
    alerts = _alerts(ledger)
    assert [a["threshold_fraction"] for a in alerts] == [0.5, 0.75, 0.9]
    for a in alerts:
        assert a["source"] == "cloud_client" and a["cap_usd"] == 50.0
        assert a["running_total_usd"] >= a["threshold_usd"]
    out = capsys.readouterr().out
    assert out.count("ALERT: cloud spend") == 3
    assert "50%" in out and "75%" in out and "90%" in out


def test_alerts_do_not_change_running_total_and_do_not_refire_after_restart(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    c = _client(ledger, FakeProvider())
    _spend(c, 30.0)  # 60%
    total = c.running_total_usd
    c2 = _client(ledger, FakeProvider())  # a new process on the same ledger
    assert c2.running_total_usd == pytest.approx(total)  # alert rows carry no cost
    _spend(c2, 1.0)
    assert [a["threshold_fraction"] for a in _alerts(ledger)] == [0.5]


def test_one_call_crossing_several_thresholds_fires_each_once(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    c = _client(ledger, FakeProvider())
    _spend(c, 46.0)  # 92%: crosses 50, 75 and 90 in one call
    assert [a["threshold_fraction"] for a in _alerts(ledger)] == [0.5, 0.75, 0.9]
    _spend(c, 1.0)
    assert len(_alerts(ledger)) == 3


def test_alerts_surface_in_results_digest(tmp_path):
    ledger = tmp_path / "cloud_ledger.jsonl"
    c = _client(ledger, FakeProvider())
    _spend(c, 40.0)
    shown, total = results_digest.collect_alerts([ledger])
    assert total == 2
    md = results_digest.render_markdown("controller", {}, harness_alerts=(shown, total))
    assert "cloud_client" in md and "75%" in md and "cap_usd=50.0" in md


def test_cost_helper_matches_fake_steering():
    out = _out_tokens_for(10.0)
    assert estimate_cost_usd(MODEL, 1, out) == pytest.approx(10.0, abs=2e-5)
