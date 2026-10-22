"""scripts/all_cloud_b3.py with a fake provider (no network, no key): transcript conversion, the runtime shape, the
session loop and its rows, the projection stop rule, the spend cap, the refusals."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("all_cloud_b3", REPO / "scripts" / "all_cloud_b3.py")
AC = importlib.util.module_from_spec(_spec)
sys.modules["all_cloud_b3"] = AC
_spec.loader.exec_module(AC)
cc = AC.cc


def fake_provider(calls_seen):
    """First call of a turn (last message is the user's): a native log_event call. Otherwise a JSON answer.
    Usage: prompt tokens from the text length, half of them cache reads after the first call."""
    state = {"n": 0}

    def fn(*, model_id, messages, api_key, **kw):
        calls_seen.append({"model_id": model_id, "messages": messages, "kw": kw})
        state["n"] += 1
        prompt = max(600, len(json.dumps(messages)) // 4)
        read = prompt // 2 if state["n"] > 1 else 0
        write = prompt - read
        if messages[-1]["role"] == "user":
            msg = {"role": "assistant", "content": "", "tool_calls": [
                {"id": "x", "type": "function", "function": {"name": "log_event",
                                                             "arguments": json.dumps({"event": "turn"})}}]}
        else:
            msg = {"role": "assistant", "content": json.dumps({"answer": "done"})}
        return {"choices": [{"message": msg, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": 12, "cache_read_input_tokens": read,
                          "cache_creation_input_tokens": write}}
    return fn


def make_client(tmp_path, fn, cap=50.0):
    return cc.CloudClient(mode="real", api_key="fake-not-a-key", completion_fn=fn, role_override="controller",
                          ledger_path=tmp_path / "ledger.jsonl", spend_cap_usd=cap)


def test_models_from_shared_price_file():
    t = AC.CR.load_price_table(AC.PRICE_FILE)
    assert AC.models_from(t) == ("claude-haiku-5-5", "claude-sonnet-5-5")


def test_to_provider_messages_ids_and_text_fallback():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "log_event",
                                                                              "arguments": {"event": "e"}}}]},
            {"role": "tool", "content": "{\"ok\": true}", "tool_name": "log_event"},
            {"role": "assistant", "content": "{\"name\": \"get_record\", \"arguments\": {}}"},
            {"role": "tool", "content": "{\"r\": 1}", "tool_name": "get_record"}]
    out = AC.to_provider_messages(msgs)
    assert out[2]["tool_calls"][0]["id"] == "call_2_0"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"event": "e"}
    assert out[3] == {"role": "tool", "tool_call_id": "call_2_0", "content": "{\"ok\": true}"}
    assert out[4] == {"role": "assistant", "content": "{\"name\": \"get_record\", \"arguments\": {}}"}
    assert out[5]["role"] == "user" and "get_record" in out[5]["content"]
    assert AC.to_provider_messages(msgs) == out   # deterministic (stable cache prefix)


def test_runtime_final_call_keeps_tools_with_choice_none(tmp_path):
    seen = []
    client = make_client(tmp_path, fake_provider(seen))
    rt = AC.CloudRuntime(client, "claude-haiku-5-5", AC.CR.model_rates(AC.CR.load_price_table(), "claude-haiku-5-5"))
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "log_event",
                                                                              "arguments": {"event": "e"}}}]},
            {"role": "tool", "content": "{}", "tool_name": "log_event"}]
    resp = rt.chat("m", msgs, None, None, None)
    assert resp["outcome"] == "ok"
    assert seen[-1]["kw"]["tool_choice"] == "none" and seen[-1]["kw"]["tools"]
    assert seen[-1]["kw"]["custom_llm_provider"] == "anthropic" and seen[-1]["kw"]["max_tokens"] == 384
    assert seen[-1]["kw"]["temperature"] == 0.0
    # cache breakpoints on system and last user message
    sent = seen[-1]["messages"]
    assert sent[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    c = rt.drain()[0]
    assert c["cache_read_tokens"] + c["cache_write_tokens"] + c["uncached_input_tokens"] == c["input_tokens"]
    assert c["usd_ledger"] == pytest.approx(cc.estimate_cost_usd("claude-haiku-5-5", c["input_tokens"], 12))


def test_runtime_returns_native_calls_in_harness_shape(tmp_path):
    client = make_client(tmp_path, fake_provider([]))
    rt = AC.CloudRuntime(client, "claude-haiku-5-5", AC.CR.model_rates(AC.CR.load_price_table(), "claude-haiku-5-5"))
    resp = rt.chat("m", [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], None,
                   AC.EV.tools_payload("v2"), None)
    calls, method = AC.ag.extract_tool_calls(resp["message"], resp["tool_calls"])
    assert method == "native" and calls == [{"name": "log_event", "arguments": {"event": "turn"}}]


def test_session_rows_and_spend(tmp_path):
    client = make_client(tmp_path, fake_provider([]))
    rt = AC.CloudRuntime(client, "claude-sonnet-5-5", AC.CR.model_rates(AC.CR.load_price_table(), "claude-sonnet-5-5"))
    rows, spend = [], {"usd": 0.0, "usd_ledger": 0.0}
    s = AC.run_cloud_session(rt, 20260901, rows.append, lambda *_: None, max_turns=3, spend=spend)
    turns = [r for r in rows if r["record"] == "r2a_turn"]
    assert len(turns) == 3 and s["n_turns"] == 3
    for r in turns:
        assert r["n_calls"] == 2 and r["end_state_ok"] in (True, False)
        assert r["model_id"] == "claude-sonnet-5-5" and r["design"] == "v2" and r["turn_protocol"] == "tool_loop"
        for k in AC.COST_KEYS:
            assert k in r
    assert spend["usd_ledger"] == pytest.approx(sum(r["usd_ledger"] for r in turns))
    assert turns[-1]["ledger_total_usd"] == pytest.approx(client.running_total_usd)
    assert sum(1 for _ in open(tmp_path / "ledger.jsonl")) == 6


def _local_file(tmp_path, seeds=AC.SEEDS):
    rows = []
    for s in seeds:
        att = f"a{s}"
        rows.append({"record": "r2a_turn", "mode": "real", "arm_id": AC.LOCAL_REF_ARM, "seed": s, "attempt_id": att,
                     "turn_idx": 1, "calls": [{"prompt_tokens_est": 5000, "completion_tokens": 100}] * 2})
        rows.append({"record": "r2a_session", "mode": "real", "arm_id": AC.LOCAL_REF_ARM, "seed": s,
                     "attempt_id": att, "n_turns": 1, "model_id": "qwen3:14b"})
    p = tmp_path / "local.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_project_ledger_basis():
    local = {1: [(1000, 10)], 2: [(2000, 20)]}
    got = AC.project("claude-sonnet-5-5", 1.3, local, margin=1.5)
    want = 1.5 * (cc.estimate_cost_usd("claude-sonnet-5-5", 1300, 10) + cc.estimate_cost_usd("claude-sonnet-5-5", 2600, 20))
    assert got == pytest.approx(want)


def test_run_full_small(tmp_path, monkeypatch):
    if not hasattr(AC.ag, "completed_sessions"):
        pytest.skip("no completed_sessions")
    local = _local_file(tmp_path)
    monkeypatch.setattr(AC.ag, "completed_sessions", lambda rows: [r for r in rows if r.get("record") == "r2a_session"])
    client = make_client(tmp_path, fake_provider([]))
    out = tmp_path / "o.jsonl"
    res = AC.run(client, out, local_source=local, max_turns=2, log=lambda *_: None, ledger_total_at_start=0.0)
    assert res["decision"] == "run" and res["stopped"] is None
    rows = [json.loads(x) for x in open(out, encoding="utf-8")]
    kinds = [r["record"] for r in rows]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("warmup") == 2 and kinds.count("pilot_turn") == 2 and kinds.count("projection") == 1
    assert kinds.count("r2a_turn") == 2 * 3 * 2
    st = rows[0]
    assert st["models"] == ["claude-haiku-5-5", "claude-sonnet-5-5"] and st["seeds"] == list(AC.SEEDS)
    assert len(st["script_git_blob"]) == 40 and st["price_file"] == "results/cloud_prices_anthropic_20261014.json"
    summ = rows[-1]["summary"]
    for m in st["models"]:
        assert summ[m]["graded"] == 6 and summ[m]["usd"]["n"] == 6 and summ[m]["usd"]["p90"] is not None
    assert rows[-1]["ledger_total_usd"] == pytest.approx(client.running_total_usd)


def test_run_stops_when_projection_over_cap(tmp_path, monkeypatch):
    local = _local_file(tmp_path)
    monkeypatch.setattr(AC.ag, "completed_sessions", lambda rows: [r for r in rows if r.get("record") == "r2a_session"])
    client = make_client(tmp_path, fake_provider([]))
    out = tmp_path / "o.jsonl"
    res = AC.run(client, out, local_source=local, max_turns=2, log=lambda *_: None, job_cap=1e-6)
    assert res["decision"] == "stop" and "projection" in res["stopped"]
    rows = [json.loads(x) for x in open(out, encoding="utf-8")]
    assert not any(r["record"] == "r2a_turn" for r in rows)


def test_spend_cap_stops_job(tmp_path, monkeypatch):
    local = _local_file(tmp_path)
    monkeypatch.setattr(AC.ag, "completed_sessions", lambda rows: [r for r in rows if r.get("record") == "r2a_session"])
    client = make_client(tmp_path, fake_provider([]), cap=0.01)
    res = AC.run(client, tmp_path / "o.jsonl", local_source=local, max_turns=30, log=lambda *_: None, job_cap=100.0)
    assert res["stopped"] and res["stopped"].startswith("SpendCapExceeded")
    assert client.running_total_usd <= 0.01 + 1e-9


def test_main_refusals(tmp_path, monkeypatch):
    monkeypatch.delenv(AC.KEY_ENV, raising=False)
    assert AC.main(["--out", str(tmp_path / "o.jsonl")]) == 2
    assert AC.main(["--live", "--out", str(tmp_path / "o.jsonl")]) == 2
    monkeypatch.setenv(AC.KEY_ENV, "fake-not-a-key")
    stub_redo = tmp_path / "redo.jsonl"
    stub_redo.write_text(json.dumps({"record": "run_start", "mode": "priced_not_executed"}) + "\n"
                         + json.dumps({"record": "run_end"}) + "\n", encoding="utf-8")
    assert AC.main(["--live", "--out", str(tmp_path / "o.jsonl"), "--redo-file", str(stub_redo),
                    "--ledger", str(tmp_path / "l.jsonl")]) == 3
    assert not (tmp_path / "o.jsonl").exists()


def test_redo_finished(tmp_path):
    p = tmp_path / "r.jsonl"
    assert not AC.redo_finished(p)
    p.write_text(json.dumps({"record": "run_start", "mode": "executed"}) + "\n" + json.dumps({"record": "run_end"})
                 + "\n", encoding="utf-8")
    assert AC.redo_finished(p)


class UnsupportedParamsError(Exception):
    pass


def test_temperature_rejected_falls_back_to_provider_default(tmp_path, monkeypatch):
    local = _local_file(tmp_path)
    monkeypatch.setattr(AC.ag, "completed_sessions", lambda rows: [r for r in rows if r.get("record") == "r2a_session"])
    inner = fake_provider([])

    def fn(**kw):
        if "temperature" in kw:
            raise UnsupportedParamsError("claude-haiku-5-5 does not support temperature=0.0")
        return inner(**kw)
    client = make_client(tmp_path, fn)
    out = tmp_path / "o.jsonl"
    res = AC.run(client, out, local_source=local, max_turns=1, log=lambda *_: None)
    assert res["decision"] == "run" and res["stopped"] is None
    rows = [json.loads(x) for x in open(out, encoding="utf-8")]
    assert rows[0]["temperature_per_model"] == {"claude-haiku-5-5": None, "claude-sonnet-5-5": None}
    w = [r for r in rows if r["record"] == "warmup"]
    assert all(r["temperature_sent"] is None and "UnsupportedParamsError" in r["temperature_rejected"] for r in w)
    assert not any(r.get("any_http_error") for r in rows if r["record"] in ("r2a_turn", "pilot_turn"))
    # no ledger row for the rejected attempts: 2 warm-ups + 2 pilot turns x 2 calls + 6 turns x 2 calls
    assert sum(1 for _ in open(tmp_path / "ledger.jsonl")) == 2 + 4 + 12


def test_project_nan_ratio_is_infinite():
    assert AC.project("claude-haiku-5-5", float("nan"), {1: [(10, 1)]}) == float("inf")
