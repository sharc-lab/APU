"""Key/mode rule of src/cloud/client.py and the routing of every OpenAI call site through it (2026-10-08).

Fakes only: an injected completion_fn stands in for the provider. No real API, no network, no paid call.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from src.cloud.client import (
    CloudClient,
    MissingApiKeyError,
    SpendCapExceeded,
    local_openai_compatible_client,
)

MSG = [{"role": "user", "content": "hello"}]


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("OPENAI_API_KEY", "CLOUD_API_KEY", "CLOUD_ALLOWED_HOSTS", "CLOUD_CLIENT_HOST_ROLE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


class FakeProvider:
    def __init__(self, out_tokens=10, delay_s=0.0):
        self.calls = []
        self.out_tokens = out_tokens
        self.delay_s = delay_s

    def __call__(self, *, model_id, messages, api_key, **kwargs):
        self.calls.append({"model_id": model_id, "messages": messages, "api_key": api_key, **kwargs})
        if self.delay_s:
            time.sleep(self.delay_s)
        return {"usage": {"prompt_tokens": 5, "completion_tokens": self.out_tokens, "total_tokens": 5 + self.out_tokens},
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


# ---------------------------------------------------------------------------
# key / mode rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("var", ["OPENAI_API_KEY", "CLOUD_API_KEY"])
def test_default_mode_ignores_env_keys_so_stub_callers_stay_stubs(clean_env, tmp_path, var):
    """CloudClient(api_key=None) is how pareto.py and src/dse build a stub; an exported key must not flip it."""
    clean_env.setenv(var, "sk-env-key")
    client = CloudClient(api_key=None, ledger_path=tmp_path / "l.jsonl")
    assert client.stub_mode is True
    res = client.call(model_id="gpt-4o-mini", messages=MSG)
    assert res.stub and res.cost_usd == 0.0
    assert not (tmp_path / "l.jsonl").exists()


def test_stub_mode_wins_over_an_explicit_key(clean_env, tmp_path):
    assert CloudClient(mode="stub", api_key="sk-x", ledger_path=tmp_path / "l.jsonl").stub_mode is True


def test_real_mode_prefers_openai_key_over_cloud_key(clean_env, tmp_path):
    clean_env.setenv("OPENAI_API_KEY", "sk-openai")
    clean_env.setenv("CLOUD_API_KEY", "sk-cloud")
    client = CloudClient.real(ledger_path=tmp_path / "l.jsonl")
    assert client.stub_mode is False
    assert client.api_key == "sk-openai" and client.api_key_source == "OPENAI_API_KEY"


def test_real_mode_falls_back_to_cloud_key(clean_env, tmp_path):
    clean_env.setenv("CLOUD_API_KEY", "sk-cloud")
    client = CloudClient(mode="real", ledger_path=tmp_path / "l.jsonl")
    assert client.api_key_source == "CLOUD_API_KEY"


def test_real_mode_explicit_argument_beats_env(clean_env, tmp_path):
    clean_env.setenv("OPENAI_API_KEY", "sk-openai")
    client = CloudClient(mode="real", api_key="sk-arg", ledger_path=tmp_path / "l.jsonl")
    assert client.api_key == "sk-arg" and client.api_key_source == "argument"


def test_real_mode_without_any_key_refuses_instead_of_stubbing(clean_env, tmp_path):
    with pytest.raises(MissingApiKeyError):
        CloudClient(mode="real", ledger_path=tmp_path / "l.jsonl")
    clean_env.setenv("OPENAI_API_KEY", "")
    with pytest.raises(MissingApiKeyError):
        CloudClient.real(ledger_path=tmp_path / "l.jsonl")


def test_bad_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        CloudClient(mode="live", ledger_path=tmp_path / "l.jsonl")


def test_openai_key_is_not_sent_to_another_provider(clean_env, tmp_path):
    clean_env.setenv("OPENAI_API_KEY", "sk-openai")
    fake = FakeProvider()
    client = CloudClient.real(ledger_path=tmp_path / "l.jsonl", role_override="controller", completion_fn=fake)
    with pytest.raises(ValueError, match="anthropic"):
        client.call(model_id="claude-sonnet-4-5", messages=MSG)
    assert fake.calls == []


def test_real_mode_still_cannot_raise_the_cap(clean_env, tmp_path):
    clean_env.setenv("OPENAI_API_KEY", "sk-openai")
    with pytest.raises(ValueError):
        CloudClient.real(spend_cap_usd=51.0, ledger_path=tmp_path / "l.jsonl")


# ---------------------------------------------------------------------------
# tools / tool_choice pass-through (R2 call-2 rule: identical tools + tool_choice "none")
# ---------------------------------------------------------------------------


def test_tools_and_tool_choice_reach_the_provider_unchanged_and_tool_choice_is_ledgered(tmp_path):
    fake = FakeProvider()
    ledger = tmp_path / "l.jsonl"
    client = CloudClient(api_key="sk-x", ledger_path=ledger, role_override="controller", completion_fn=fake)
    tools = [{"type": "function", "function": {"name": "calc", "description": "d", "parameters": {"type": "object"}}}]
    client.call(model_id="gpt-4o-mini", messages=MSG, tools=tools, tool_choice="auto")
    client.call(model_id="gpt-4o-mini", messages=MSG, tools=tools, tool_choice="none")
    assert [c["tool_choice"] for c in fake.calls] == ["auto", "none"]
    assert fake.calls[0]["tools"] is tools and fake.calls[1]["tools"] is tools
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r["tool_choice"] for r in rows] == ["auto", "none"]


# ---------------------------------------------------------------------------
# prepare / send / settle
# ---------------------------------------------------------------------------


def test_in_flight_reservations_count_against_the_cap(tmp_path):
    fake = FakeProvider()
    # gpt-4o-mini: 1 input token + 1_000_000 output tokens projects to ~$0.60
    client = CloudClient(api_key="sk-x", spend_cap_usd=1.0, ledger_path=tmp_path / "l.jsonl",
                         role_override="controller", completion_fn=fake)
    first = client.prepare(model_id="gpt-4o-mini", messages=[{"role": "user", "content": "x"}],
                           expected_output_tokens=1_000_000)
    with pytest.raises(SpendCapExceeded, match="reserved by in-flight"):
        client.prepare(model_id="gpt-4o-mini", messages=[{"role": "user", "content": "x"}],
                       expected_output_tokens=1_000_000)
    first.release()
    client.prepare(model_id="gpt-4o-mini", messages=[{"role": "user", "content": "x"}],
                   expected_output_tokens=1_000_000)


def test_failed_send_releases_reservation_and_writes_nothing(tmp_path):
    def boom(**kwargs):
        raise RuntimeError("network down")

    ledger = tmp_path / "l.jsonl"
    client = CloudClient(api_key="sk-x", ledger_path=ledger, role_override="controller", completion_fn=boom)
    prepared = client.prepare(model_id="gpt-4o-mini", messages=MSG)
    assert client._reserved_usd > 0
    with pytest.raises(RuntimeError):
        prepared.send()
    assert client._reserved_usd == 0
    assert not ledger.exists()


def test_max_tokens_drives_the_projection_when_no_expected_output_given(tmp_path):
    client = CloudClient(api_key="sk-x", ledger_path=tmp_path / "l.jsonl", role_override="controller",
                         completion_fn=FakeProvider())
    assert client.prepare(model_id="gpt-4o-mini", messages=MSG, max_tokens=1024).expected_output_tokens == 1024
    assert client.prepare(model_id="gpt-4o-mini", messages=MSG).expected_output_tokens == 256


def test_concurrent_calls_are_all_ledgered(tmp_path):
    fake = FakeProvider(delay_s=0.01)
    ledger = tmp_path / "l.jsonl"
    client = CloudClient(api_key="sk-x", ledger_path=ledger, role_override="controller", completion_fn=fake)
    threads = [threading.Thread(target=client.call, kwargs={"model_id": "gpt-4o-mini", "messages": MSG})
               for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 6
    assert client._reserved_usd == 0
    assert rows[-1]["running_total_usd"] == pytest.approx(client.running_total_usd)


def test_local_compat_client_refuses_openai_hosts():
    for url in ("https://api.openai.com/v1", "https://openai.com/v1", "http://API.OPENAI.COM/v1"):
        with pytest.raises(ValueError):
            local_openai_compatible_client(url)
    with pytest.raises(ValueError):
        local_openai_compatible_client("not-a-url")
    client = local_openai_compatible_client("http://localhost:11434/v1")
    assert "localhost" in str(client.base_url)


# ---------------------------------------------------------------------------
# routed call sites: cap path is outside the timed region
# ---------------------------------------------------------------------------


def _slow_bookkeeping(monkeypatch, client, delay_s):
    """Make prepare() and settle() slow so a timer that wrongly wraps them would show it."""
    real_prepare, real_settle = client.prepare, client._settle

    def slow_prepare(**kw):
        time.sleep(delay_s)
        return real_prepare(**kw)

    def slow_settle(*a, **kw):
        time.sleep(delay_s)
        return real_settle(*a, **kw)

    monkeypatch.setattr(client, "prepare", slow_prepare)
    monkeypatch.setattr(client, "_settle", slow_settle)


def test_sdk_direct_backend_routes_through_cap_and_times_only_the_send(monkeypatch, tmp_path):
    from harness.adapters.sdk_direct import TOOL_DEFINITIONS, OpenAIChatBackend
    from harness.replay import ReplayCache

    fake = FakeProvider()
    ledger = tmp_path / "l.jsonl"
    cloud = CloudClient(api_key="sk-x", ledger_path=ledger, role_override="controller", completion_fn=fake)
    _slow_bookkeeping(monkeypatch, cloud, 0.3)
    backend = OpenAIChatBackend(cloud=cloud, replay_cache=ReplayCache(mode="RECORD", traces_root=tmp_path / "t"))
    res = backend.model_call(model="gpt-4o-mini", messages=MSG, tools=TOOL_DEFINITIONS, temperature=None,
                             seed=None, tool_choice="none", max_tokens=1024)
    assert res.recorded_latency_ms < 250, res.recorded_latency_ms
    assert fake.calls[0]["tool_choice"] == "none" and fake.calls[0]["tools"] == TOOL_DEFINITIONS
    assert fake.calls[0]["max_tokens"] == 1024
    assert len(ledger.read_text().splitlines()) == 1


def test_sdk_direct_backend_requires_exactly_one_client():
    from harness.adapters.sdk_direct import OpenAIChatBackend

    with pytest.raises(ValueError):
        OpenAIChatBackend()


def test_cloud_openai_backend_routes_through_cap_and_times_only_the_send(monkeypatch, tmp_path):
    from harness.backends.cloud_openai import CloudOpenAIBackend

    fake = FakeProvider()
    ledger = tmp_path / "l.jsonl"
    cloud = CloudClient(api_key="sk-x", ledger_path=ledger, role_override="controller", completion_fn=fake)
    _slow_bookkeeping(monkeypatch, cloud, 0.3)
    backend = CloudOpenAIBackend(cloud_client=cloud, replay_mode="RECORD", traces_root=tmp_path / "t")
    res = backend.model_call(messages=MSG, seed=7)
    assert res.recorded_latency_ms < 250, res.recorded_latency_ms
    assert fake.calls[0]["seed"] == 7
    assert len(ledger.read_text().splitlines()) == 1


def test_cloud_openai_backend_without_key_refuses(clean_env, tmp_path):
    from harness.backends.cloud_openai import CloudOpenAIBackend

    with pytest.raises(MissingApiKeyError):
        CloudOpenAIBackend(replay_mode="RECORD", traces_root=tmp_path / "t", ledger_path=tmp_path / "l.jsonl")


def test_cap_refusal_reaches_backend_caller_before_any_send(tmp_path):
    from harness.backends.cloud_openai import CloudOpenAIBackend

    fake = FakeProvider()
    cloud = CloudClient(api_key="sk-x", spend_cap_usd=0.0, ledger_path=tmp_path / "l.jsonl",
                        role_override="controller", completion_fn=fake)
    backend = CloudOpenAIBackend(cloud_client=cloud, replay_mode="RECORD", traces_root=tmp_path / "t")
    with pytest.raises(SpendCapExceeded):
        backend.model_call(messages=MSG)
    assert fake.calls == []


def test_default_openai_path_uses_one_sdk_client_and_passes_kwargs(monkeypatch, tmp_path):
    """No completion_fn injected: the OpenAI SDK path is used (with a fake SDK object, never the network)."""
    import src.cloud.client as cc

    made = []

    class _Resp:
        def model_dump(self, exclude_unset=False):
            return {"usage": {"prompt_tokens": 3, "completion_tokens": 4}, "choices": [{"message": {"content": "hi"}}]}

    class _Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kw):
            self.calls.append(kw)
            return _Resp()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _SDK:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = _Chat()

    def fake_make(api_key):
        made.append(_SDK(api_key))
        return made[-1]

    monkeypatch.setattr(cc, "_make_openai_sdk_client", fake_make)
    ledger = tmp_path / "l.jsonl"
    client = CloudClient(api_key="sk-x", ledger_path=ledger, role_override="controller")
    r1 = client.call(model_id="gpt-4o-mini", messages=MSG, tool_choice="none", tools=[])
    client.call(model_id="gpt-4o-mini", messages=MSG)
    assert len(made) == 1 and made[0].api_key == "sk-x"
    assert made[0].chat.completions.calls[0] == {"model": "gpt-4o-mini", "messages": MSG, "tool_choice": "none",
                                                 "tools": []}
    assert r1.content == "hi" and r1.input_tokens == 3 and r1.output_tokens == 4
    assert len(ledger.read_text().splitlines()) == 2
