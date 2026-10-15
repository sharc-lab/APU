"""src/cloud/prices.py and src/cloud/client.py on the shared price files (2026-10-14).

The client has no hard-coded price table: MODEL_PRICING and CLOUD_MODELS come from results/cloud_prices_*.json, the
latest file per provider. The USD 50 hard cap is unchanged and still refuses a call that would exceed it. Fakes only:
no real API, no network, no paid call.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import src.cloud.client as cc
from src.cloud import prices as cp

REPO = Path(__file__).resolve().parents[1]
OPENAI_FILE = "results/cloud_prices_20261008.json"


def _anthropic_table(fetched="2026-10-14T00:00Z", mid_in=3.0):
    return {"provider": "anthropic", "source_url": "https://claude.com/pricing", "fetched_utc": fetched,
            "selection": {"cheap_small": "claude-cheap", "mid_tier": "claude-mid"},
            "models": {"claude-cheap": {"standard": {"input": 1.0, "cache_read": 0.1, "output": 5.0}},
                       "claude-mid": {"standard": {"short": {"input": mid_in, "cached_input": 0.3, "output": 15.0}}}}}


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / OPENAI_FILE).write_text((REPO / OPENAI_FILE).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def test_client_pricing_matches_the_shared_price_file():
    raw = json.loads((REPO / OPENAI_FILE).read_text(encoding="utf-8"))
    for model, entry in raw["models"].items():
        short = entry["standard"]["short"]
        assert cc.MODEL_PRICING[model]["input_per_1m_usd"] == short["input"]
        assert cc.MODEL_PRICING[model]["output_per_1m_usd"] == short["output"]
        assert cc.MODEL_PRICING[model]["source_url"] == raw["source_url"]
    inp, cached, out = raw["other_rows_short_context"]["standard"]["gpt-4o-mini"]
    assert (cc.MODEL_PRICING["gpt-4o-mini"]["input_per_1m_usd"], cc.MODEL_PRICING["gpt-4o-mini"]["output_per_1m_usd"]) \
        == (inp, out)
    files = cp.price_files(REPO)
    if "anthropic" not in files:   # before the Anthropic price file lands, OpenAI's selection is the tier map
        assert cc.CLOUD_MODELS == {"cheap": raw["selection"]["cheap_small"], "strong": raw["selection"]["mid_tier"]}
    for model in cc.CLOUD_MODELS.values():
        assert cc.MODEL_PRICING[model]["price_file"].startswith("results/cloud_prices_")


def test_client_source_has_no_hard_coded_price_table():
    src = (REPO / "src/cloud/client.py").read_text(encoding="utf-8")
    assert not re.search(r"_per_1m_usd\"\s*:\s*\d", src), "a literal per-token price in src/cloud/client.py"
    assert "claude-sonnet-4-5" not in src and "gpt-4o-mini\": {" not in src


def test_latest_file_per_provider_and_anthropic_becomes_the_tier_provider(repo):
    (repo / "results/cloud_prices_anthropic_20261001.json").write_text(json.dumps(_anthropic_table("old", 99.0)),
                                                                      encoding="utf-8")
    (repo / "results/cloud_prices_anthropic_20261014.json").write_text(json.dumps(_anthropic_table()),
                                                                      encoding="utf-8")
    (repo / "results/cloud_prices_notes.json").write_text("{}", encoding="utf-8")   # unparseable name: ignored
    files = cp.price_files(repo)
    assert {p: f.name for p, f in files.items()} == {"anthropic": "cloud_prices_anthropic_20261014.json",
                                                     "openai": "cloud_prices_20261008.json"}
    pricing, tiers = cc.load_model_pricing(repo)
    assert tiers == {"cheap": "claude-cheap", "strong": "claude-mid"}
    assert pricing["claude-mid"]["input_per_1m_usd"] == 3.0 and pricing["claude-mid"]["provider"] == "anthropic"
    assert pricing["claude-cheap"]["cached_input_per_1m_usd"] == 0.1          # "cache_read" spelling accepted
    assert pricing["gpt-6.1-sol"]["provider"] == "openai"                      # both providers stay priced


def test_no_price_file_fails_closed(tmp_path):
    (tmp_path / "results").mkdir()
    assert cc.load_model_pricing(tmp_path) == ({}, {})


def test_long_context_rate_applies_above_the_boundary():
    r = cc.MODEL_PRICING["gpt-6.1-sol"]["rates"]
    assert cp.rate_for(r, 272_000) == r["short"] and cp.rate_for(r, 272_001) == r["long"]
    assert cc.estimate_cost_usd("gpt-6.1-sol", 300_000, 0) == pytest.approx(0.3 * r["long"]["input"])


def test_bad_rates_raise():
    with pytest.raises(cp.PriceTableError):
        cp.model_rates({"models": {"m": {"standard": {"short": {"input": 1.0}}}}})


def test_hard_cap_unchanged_and_refuses_a_call_that_would_exceed_it(tmp_path):
    """USD 50 total, from the shared-price model: 49 spent, then a call projected past 50 is refused before
    anything is sent and nothing is written; a call that fits is still allowed."""
    assert cc.HARD_SPEND_CAP_USD == 50.0 and cc.DEFAULT_SPEND_CAP_USD == 50.0
    model = cc.CLOUD_MODELS["strong"]
    out_rate = cc.MODEL_PRICING[model]["output_per_1m_usd"] / 1e6
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"model_id": model, "cost_usd": 49.0}) + "\n", encoding="utf-8")
    sent = []

    def fake(**kw):
        sent.append(kw)
        return {"usage": {"prompt_tokens": 1, "completion_tokens": kw.get("fake_out", 0)},
                "choices": [{"message": {"content": "ok"}}]}

    client = cc.CloudClient(api_key="sk-fake", ledger_path=ledger, role_override="controller", completion_fn=fake)
    assert client.spend_cap_usd == 50.0 and client.running_total_usd == pytest.approx(49.0)
    over = int(1.5 / out_rate)   # about USD 1.50 of output: 49 + 1.5 > 50
    with pytest.raises(cc.SpendCapExceeded, match="exceeding the \\$50.00 cap"):
        client.call(model_id=model, messages=[{"role": "user", "content": "x"}], expected_output_tokens=over,
                    fake_out=over)
    assert sent == [] and len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    fits = int(0.5 / out_rate)
    res = client.call(model_id=model, messages=[{"role": "user", "content": "x"}], expected_output_tokens=fits,
                      fake_out=fits)
    assert res.stub is False and len(sent) == 1 and client.running_total_usd <= 50.0
    with pytest.raises(ValueError, match="hard cap"):
        cc.CloudClient(api_key="sk-fake", spend_cap_usd=50.01, ledger_path=ledger)


def test_unpriced_model_is_refused_in_real_mode(tmp_path):
    client = cc.CloudClient(api_key="sk-fake", ledger_path=tmp_path / "l.jsonl", role_override="controller",
                            completion_fn=lambda **kw: pytest.fail("must not send"))
    with pytest.raises(cc.UnknownModelError, match="shared price files"):
        client.call(model_id="claude-sonnet-4-5", messages=[{"role": "user", "content": "x"}])


def test_stub_path_never_reads_key_env(monkeypatch, tmp_path):
    """mode='auto' with no api_key must not touch os.environ for a key, even with every key exported."""
    for var in cc.REAL_MODE_KEY_ENV_VARS:
        monkeypatch.setenv(var, "sk-exported")
    read = []
    real_get = cc.os.environ.get

    def spy(key, default=None):
        read.append(key)
        return real_get(key, default)

    monkeypatch.setattr(cc.os.environ, "get", spy)
    client = cc.CloudClient(api_key=None, ledger_path=tmp_path / "l.jsonl")
    res = client.call(model_id=cc.CLOUD_MODELS["cheap"], messages=[{"role": "user", "content": "x"}])
    assert client.stub_mode and res.stub and client.api_key is None
    assert not set(read) & set(cc.REAL_MODE_KEY_ENV_VARS)
