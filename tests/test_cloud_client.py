"""Unit tests for src/cloud/client.py.

Covers: stub fallback, spend-cap enforcement (including the exact boundary),
ledger writing, and the host-restriction check. No real network calls; the
real-call path is exercised via a fake completion_fn, never litellm or a
real key.
"""

from __future__ import annotations

import json

import pytest

from src.cloud.client import (
    CLOUD_MODELS,
    MODEL_PRICING,
    CallResult,
    CloudClient,
    HostRestrictionError,
    SpendCapExceeded,
    UnknownModelError,
    check_host_allowed,
    estimate_cost_usd,
    print_stub_banner_if_needed,
    redact_key,
)


# ---------------------------------------------------------------------------
# redact_key
# ---------------------------------------------------------------------------


def test_redact_key_none_or_empty():
    assert redact_key(None) == "<none>"
    assert redact_key("") == "<none>"


def test_redact_key_short_key_fully_masked():
    assert redact_key("ab") == "**"
    assert redact_key("abcd") == "****"


def test_redact_key_long_key_shows_only_first_and_last_two():
    redacted = redact_key("sk-abcdefghijklmnop")
    assert redacted.startswith("sk")
    assert redacted.endswith("op")
    assert "abcdefghijklmn" not in redacted
    # never more than 4 real characters exposed
    real_chars = redacted.replace("*", "")
    assert len(real_chars) == 4


# ---------------------------------------------------------------------------
# host restriction
# ---------------------------------------------------------------------------


def test_host_restriction_hard_denies_evo_t2s_regardless_of_allowlist(monkeypatch):
    monkeypatch.setenv("CLOUD_ALLOWED_HOSTS", "evo-t2s,anything")  # even if misconfigured
    with pytest.raises(HostRestrictionError, match="hard-denied"):
        check_host_allowed(hostname="EVO-T2S")


def test_host_restriction_hard_deny_beats_role_override():
    with pytest.raises(HostRestrictionError, match="hard-denied"):
        check_host_allowed(hostname="evo-t2s-remote", role_override="controller")


def test_host_restriction_allows_evo_x2_by_default(monkeypatch):
    monkeypatch.delenv("CLOUD_ALLOWED_HOSTS", raising=False)
    assert check_host_allowed(hostname="EVO-X2") is True


def test_host_restriction_allows_explicit_controller_role():
    assert check_host_allowed(hostname="some-laptop", role_override="controller") is True


def test_host_restriction_allows_controller_role_via_env(monkeypatch):
    monkeypatch.setenv("CLOUD_CLIENT_HOST_ROLE", "controller")
    assert check_host_allowed(hostname="some-laptop") is True


def test_host_restriction_denies_unknown_host(monkeypatch):
    monkeypatch.delenv("CLOUD_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("CLOUD_CLIENT_HOST_ROLE", raising=False)
    with pytest.raises(HostRestrictionError, match="not allowed"):
        check_host_allowed(hostname="random-laptop")


def test_host_restriction_custom_allowlist_via_env(monkeypatch):
    monkeypatch.setenv("CLOUD_ALLOWED_HOSTS", "my-controller-box")
    assert check_host_allowed(hostname="MY-CONTROLLER-BOX") is True
    with pytest.raises(HostRestrictionError):
        check_host_allowed(hostname="evo-x2")  # default no longer applies once env var set


# ---------------------------------------------------------------------------
# stub fallback
# ---------------------------------------------------------------------------


def test_stub_mode_when_no_api_key(tmp_path):
    client = CloudClient(api_key=None, ledger_path=tmp_path / "ledger.jsonl")
    assert client.stub_mode is True


def test_stub_mode_when_api_key_present_is_false(tmp_path):
    client = CloudClient(api_key="sk-real-looking-key", ledger_path=tmp_path / "ledger.jsonl")
    assert client.stub_mode is False


def test_stub_call_never_raises_on_disallowed_host(tmp_path):
    # stub mode must work even on a hostname that would hard-deny real calls
    client = CloudClient(api_key=None, ledger_path=tmp_path / "ledger.jsonl", hostname="evo-t2s")
    result = client.call(model_id="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    assert result.stub is True
    assert result.model_id == "stub-gpt-4o-mini"


def test_stub_call_labels_every_row_as_stub(tmp_path):
    client = CloudClient(api_key=None, ledger_path=tmp_path / "ledger.jsonl")
    for model_id in CLOUD_MODELS.values():
        result = client.call(model_id=model_id, messages=[{"role": "user", "content": "hello there"}])
        assert result.stub is True
        assert result.model_id.startswith("stub-")
        assert result.cost_usd == 0.0
        assert "stub" in result.content.lower()


def test_stub_mode_never_writes_ledger(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    client = CloudClient(api_key=None, ledger_path=ledger_path)
    client.call(model_id="gpt-4o-mini", messages=[{"role": "user", "content": "hello"}])
    assert not ledger_path.exists()


def test_print_stub_banner_if_needed(tmp_path, capsys):
    client = CloudClient(api_key=None, ledger_path=tmp_path / "ledger.jsonl")
    print_stub_banner_if_needed(client)
    captured = capsys.readouterr()
    assert captured.out.strip() == "BLOCKED: cloud key"


def test_print_stub_banner_silent_when_not_stub(tmp_path, capsys):
    client = CloudClient(api_key="sk-real", ledger_path=tmp_path / "ledger.jsonl")
    print_stub_banner_if_needed(client)
    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------
# real-call path (fake completion_fn, no litellm, no network)
# ---------------------------------------------------------------------------


def _fake_completion_fn(**kwargs):
    return {
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "choices": [{"message": {"content": "4 business hours"}}],
    }


def test_real_call_requires_allowed_host(tmp_path):
    client = CloudClient(
        api_key="sk-real",
        ledger_path=tmp_path / "ledger.jsonl",
        hostname="evo-t2s",
        completion_fn=_fake_completion_fn,
    )
    with pytest.raises(HostRestrictionError):
        client.call(model_id="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])


def test_real_call_succeeds_on_allowed_host_and_writes_ledger(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    client = CloudClient(
        api_key="sk-real",
        ledger_path=ledger_path,
        role_override="controller",
        completion_fn=_fake_completion_fn,
    )
    result = client.call(model_id="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    assert result.stub is False
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    expected_cost = estimate_cost_usd("gpt-4o-mini", 100, 50)
    assert result.cost_usd == pytest.approx(expected_cost)

    assert ledger_path.exists()
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["model_id"] == "gpt-4o-mini"
    assert rows[0]["input_tokens"] == 100
    assert rows[0]["output_tokens"] == 50
    assert rows[0]["cost_usd"] == pytest.approx(expected_cost)
    assert rows[0]["running_total_usd"] == pytest.approx(expected_cost)
    assert "timestamp" in rows[0]


def test_real_call_unknown_model_raises(tmp_path):
    client = CloudClient(
        api_key="sk-real",
        ledger_path=tmp_path / "ledger.jsonl",
        role_override="controller",
        completion_fn=_fake_completion_fn,
    )
    with pytest.raises(UnknownModelError):
        client.call(model_id="not-a-real-model", messages=[{"role": "user", "content": "hi"}])


def test_running_total_is_read_back_from_existing_ledger(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        json.dumps({"cost_usd": 12.5}) + "\n" + json.dumps({"cost_usd": 2.5}) + "\n"
    )
    client = CloudClient(api_key="sk-real", ledger_path=ledger_path, role_override="controller")
    assert client.running_total_usd == pytest.approx(15.0)
    assert client.remaining_budget_usd == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# spend cap enforcement, including exactly at the boundary
# ---------------------------------------------------------------------------


def test_spend_cap_refuses_call_that_would_exceed_cap(tmp_path):
    def expensive_completion_fn(**kwargs):
        return {"usage": {"prompt_tokens": 10_000_000, "completion_tokens": 10_000_000}}

    client = CloudClient(
        api_key="sk-real",
        ledger_path=tmp_path / "ledger.jsonl",
        role_override="controller",
        spend_cap_usd=1.00,
        completion_fn=expensive_completion_fn,
    )
    with pytest.raises(SpendCapExceeded, match="exceeding"):
        client.call(
            model_id="gpt-4o-mini",
            messages=[{"role": "user", "content": "x" * 100}],
            expected_output_tokens=10_000_000,
        )


def test_spend_cap_allows_call_landing_exactly_on_cap(tmp_path):
    # gpt-4o-mini: $0.15/1M in, $0.60/1M out. Pick tokens so the cost is exactly $1.00.
    # 1,000,000 input tokens -> $0.15; remaining $0.85 of output budget at $0.60/1M
    # -> 1,416,666.67 output tokens. Use the estimator itself to build an exact-boundary case.
    cap = 1.00
    input_tokens = 1_000_000
    input_cost = estimate_cost_usd("gpt-4o-mini", input_tokens, 0)
    remaining = cap - input_cost
    output_tokens = int(remaining / (MODEL_PRICING["gpt-4o-mini"]["output_per_1m_usd"] / 1_000_000))
    projected = estimate_cost_usd("gpt-4o-mini", input_tokens, output_tokens)
    assert projected <= cap  # sanity: our chosen token counts land at or under the cap

    def completion_fn(**kwargs):
        return {"usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}}

    client = CloudClient(
        api_key="sk-real",
        ledger_path=tmp_path / "ledger.jsonl",
        role_override="controller",
        spend_cap_usd=cap,
        completion_fn=completion_fn,
    )
    result = client.call(
        model_id="gpt-4o-mini",
        messages=[{"role": "user", "content": "x"}],
        expected_output_tokens=output_tokens,
    )
    assert result.cost_usd <= cap


def test_spend_cap_refuses_call_one_cent_over_cap(tmp_path):
    # The pre-call projection is computed from the *caller's* expected_output_tokens
    # and the approximate token count of the actual messages passed (negligible here,
    # a one-word message) -- not from whatever a completion_fn might later return.
    # So we build expected_output_tokens directly from the estimator to land $0.01
    # over the cap, independent of the (irrelevant, pre-call) completion_fn return.
    cap = 1.00
    target_projected_cost = cap + 0.01
    output_tokens = int(
        target_projected_cost / (MODEL_PRICING["gpt-4o-mini"]["output_per_1m_usd"] / 1_000_000)
    ) + 1  # +1 to guard against int() truncation landing exactly on/under cap
    projected = estimate_cost_usd("gpt-4o-mini", 1, output_tokens)
    assert projected > cap  # sanity: this case must be over the cap

    def completion_fn(**kwargs):
        raise AssertionError("completion_fn must not be called once the spend cap refuses the request")

    client = CloudClient(
        api_key="sk-real",
        ledger_path=tmp_path / "ledger.jsonl",
        role_override="controller",
        spend_cap_usd=cap,
        completion_fn=completion_fn,
    )
    with pytest.raises(SpendCapExceeded):
        client.call(
            model_id="gpt-4o-mini",
            messages=[{"role": "user", "content": "x"}],
            expected_output_tokens=output_tokens,
        )


def test_spend_cap_refusal_never_writes_ledger(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"

    def expensive_completion_fn(**kwargs):
        return {"usage": {"prompt_tokens": 10_000_000, "completion_tokens": 10_000_000}}

    client = CloudClient(
        api_key="sk-real",
        ledger_path=ledger_path,
        role_override="controller",
        spend_cap_usd=0.01,
        completion_fn=expensive_completion_fn,
    )
    with pytest.raises(SpendCapExceeded):
        client.call(
            model_id="gpt-4o-mini",
            messages=[{"role": "user", "content": "x"}],
            expected_output_tokens=10_000_000,
        )
    assert not ledger_path.exists()


def test_spend_cap_accumulates_across_calls(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    # 250,000 output tokens at $0.60/1M = $0.15 per call (input negligible: 1-word messages).
    # Both the pre-call projection (expected_output_tokens) and the post-call actual
    # usage returned by completion_fn use the same figure so the running total the
    # cap is checked against matches what gets written to the ledger.
    output_tokens_per_call = 250_000

    def completion_fn(**kwargs):
        return {"usage": {"prompt_tokens": 1, "completion_tokens": output_tokens_per_call}}

    client = CloudClient(
        api_key="sk-real",
        ledger_path=ledger_path,
        role_override="controller",
        spend_cap_usd=0.40,
        completion_fn=completion_fn,
    )
    client.call(
        model_id="gpt-4o-mini",
        messages=[{"role": "user", "content": "a"}],
        expected_output_tokens=output_tokens_per_call,
    )
    client.call(
        model_id="gpt-4o-mini",
        messages=[{"role": "user", "content": "b"}],
        expected_output_tokens=output_tokens_per_call,
    )
    # two calls at ~$0.15 = ~$0.30; a third would bring total to ~$0.45 > $0.40 cap
    with pytest.raises(SpendCapExceeded):
        client.call(
            model_id="gpt-4o-mini",
            messages=[{"role": "user", "content": "c"}],
            expected_output_tokens=output_tokens_per_call,
        )
    rows = [json.loads(r) for r in ledger_path.read_text().splitlines()]
    calls = [r for r in rows if r.get("record") != "alert"]
    assert len(calls) == 2  # refused third call never appended
    # ~$0.30 of a $0.40 cap crossed 50% and 75% (spend alerts, 2026-10-07); 90% was never reached
    assert sorted(r["threshold_fraction"] for r in rows if r.get("record") == "alert") == [0.5, 0.75]


# ---------------------------------------------------------------------------
# estimate_cost_usd
# ---------------------------------------------------------------------------


def test_estimate_cost_usd_known_model():
    cost = estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.15 + 0.60)


def test_estimate_cost_usd_unknown_model_raises():
    with pytest.raises(UnknownModelError):
        estimate_cost_usd("not-a-model", 1, 1)


def test_model_pricing_table_has_both_tiers():
    assert set(CLOUD_MODELS) == {"cheap", "strong"}
    for model_id in CLOUD_MODELS.values():
        assert model_id in MODEL_PRICING
        assert "fetch_date" in MODEL_PRICING[model_id]
        assert "source_url" in MODEL_PRICING[model_id]


def test_call_result_is_frozen_dataclass():
    result = CallResult(
        model_id="stub-gpt-4o-mini",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        stub=True,
        content="x",
    )
    with pytest.raises(Exception):
        result.cost_usd = 5.0  # type: ignore[misc]
