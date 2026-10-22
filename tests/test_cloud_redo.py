"""scripts/cloud_redo.py: stub pricing, the live refusal without a key, the live path with a fake provider, the cap."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cloud_redo", REPO / "scripts" / "cloud_redo.py")
CR = importlib.util.module_from_spec(_spec)
sys.modules["cloud_redo"] = CR
_spec.loader.exec_module(CR)
cc = CR.cc

RATES = {"short": {"input": 2.0, "cache_write": 2.5, "cached_input": 0.1, "output": 10.0},
         "long": {"input": 2.0, "cache_write": 2.5, "cached_input": 0.1, "output": 10.0},
         "long_context_above": None, "min_cacheable": 512}


def test_price_turn_cold_cache_model():
    t = CR.price_turn([(10_000, 100), (12_000, 50)], RATES)
    assert t["cache_write_tokens"] == 10_000 + 2_000 and t["cache_read_tokens"] == 10_000
    want = (12_000 * 2.5 + 10_000 * 0.1 + 150 * 10.0) / 1e6
    assert t["usd"] == pytest.approx(want)
    assert t["usd_all_uncached"] == pytest.approx((22_000 * 2.0 + 150 * 10.0) / 1e6)


def test_price_turn_below_min_cacheable_is_uncached():
    t = CR.price_turn([(400, 10)], RATES)
    assert t["uncached_input_tokens"] == 400 and t["cache_write_tokens"] == 0
    assert t["usd"] == pytest.approx(t["usd_all_uncached"])


def test_escalation_model_from_shared_price_file():
    table = CR.load_price_table()
    assert CR.escalation_model(table) == table["selection"]["escalation_mid_tier"]
    assert CR.escalation_model(table) in cc.MODEL_PRICING


def test_wrong_task_counts_match_register_tiers():
    wrong = CR.wrong_tasks(CR.load_rows(CR.SOURCE))
    per = {}
    for r in wrong:
        per[CR.tier_of(r["arm_id"])] = per.get(CR.tier_of(r["arm_id"]), 0) + 1
    assert per == {"ollama_default": 17, "ollama_ctx_16384": 19, "ollama_ctx_8192": 53, "ollama_ctx_4096": 63,
                   "ollama_ctx_4096_numkeep": 45}


def test_main_refuses_live_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv(CR.KEY_ENV, raising=False)
    out = tmp_path / "o.jsonl"
    assert CR.main(["--live", "--out", str(out), "--limit", "1"]) == 2
    assert not out.exists()


def test_stub_run_rows(tmp_path, capsys):
    out = tmp_path / "o.jsonl"
    rows = CR.run(False, out=out, limit=2)
    assert capsys.readouterr().out.splitlines()[0] == "BLOCKED: cloud key"
    start = rows[0]
    assert start["record"] == "run_start" and start["mode"] == CR.MODE_STUB
    assert start["script_git_blob"] and start["price_file_git_blob"] and start["model_id"]
    tasks = [r for r in rows if r["record"] == "cloud_redo_task"]
    assert len(tasks) == 2
    for r in tasks:
        assert r["seconds"] is None and r["cloud_end_state_ok"] is None
        assert r["local_regrade_end_state_ok"] is False
        assert r["usd"] > 0
    assert rows[-1]["record"] == "run_end" and rows[-1]["summary"]["n"] == 2


def _resp(content="", calls=(), usage=(1000, 20, 0, 0)):
    p, o, read, write = usage
    tcs = [{"id": f"t{i}", "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for i, c in enumerate(calls)]
    return {"choices": [{"message": {"content": content, "tool_calls": tcs or None}}],
            "usage": {"prompt_tokens": p, "completion_tokens": o, "cache_read_input_tokens": read,
                      "cache_creation_input_tokens": write}}


def _client(tmp_path, fn, cap=cc.HARD_SPEND_CAP_USD):
    return cc.CloudClient(mode="real", api_key="test-not-a-key", role_override="controller", completion_fn=fn,
                          ledger_path=tmp_path / "ledger.jsonl", spend_cap_usd=cap)


def _first_built():
    src = CR.load_rows(CR.SOURCE)
    r = CR.wrong_tasks(src)[0]
    es = CR.build_session_for(r, {})
    prior = [x for x in CR.session_rows(src, r["attempt_id"]) if x["turn_idx"] < r["turn_idx"]]
    return es, CR.rebuild(es, prior, r["turn_idx"])


def test_live_turn_reference_calls_grade_correct(tmp_path):
    es, built = _first_built()
    ref = [dict(c) for c in built["item"].task.reference]
    answer = json.dumps({"answer": " ".join(str(v) for v in built["item"].task.answer_values)})
    seen = []

    def fake(*, model_id, messages, api_key, **kw):
        seen.append(kw)
        return _resp(calls=ref, usage=(9000, 30, 0, 9000)) if len(seen) == 1 else _resp(answer, usage=(9100, 20, 9000, 100))

    rates = CR.model_rates(CR.load_price_table(), CR.escalation_model(CR.load_price_table()))
    res = CR.live_turn(_client(tmp_path, fake), CR.escalation_model(CR.load_price_table()), es, built, rates, 384)
    assert res["cloud_end_state_ok"] is True and res["n_calls"] == 2
    assert res["cache_read_tokens"] == 9000 and res["cache_write_tokens"] == 9100
    assert res["seconds"] is not None and res["usd"] > 0
    assert seen[0]["tools"] and seen[0]["max_tokens"] == 384
    # caching on: system prompt carries a cache breakpoint
    led = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(led) == 2


def test_live_run_with_fake_client(tmp_path):
    calls = []

    def fake(*, model_id, messages, api_key, **kw):
        calls.append(messages)
        return _resp('{"answer": "no idea"}')

    out = tmp_path / "o.jsonl"
    rows = CR.run(True, out=out, limit=2, client=_client(tmp_path, fake))
    assert rows[0]["mode"] == CR.MODE_LIVE and rows[0]["not_measured"] == []
    assert rows[1]["record"] == "warmup" and rows[1]["excluded_from_stats"] is True
    tasks = [r for r in rows if r["record"] == "cloud_redo_task"]
    assert len(tasks) == 2 and all(isinstance(r["cloud_end_state_ok"], bool) for r in tasks)
    assert all(r["seconds"] is not None for r in tasks)
    assert len(calls) == 3   # warm-up + one call per task (no tool calls returned)
    sysmsg = calls[1][0]
    assert sysmsg["role"] == "system" and sysmsg["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert rows[-1]["summary"]["n"] == 2


def test_live_turn_cap_refuses(tmp_path):
    es, built = _first_built()

    def fake(**kw):
        raise AssertionError("must not be sent")

    rates = CR.model_rates(CR.load_price_table(), CR.escalation_model(CR.load_price_table()))
    with pytest.raises(cc.SpendCapExceeded):
        CR.live_turn(_client(tmp_path, fake, cap=0.0001), CR.escalation_model(CR.load_price_table()), es, built,
                     rates, 384)


def test_live_run_refuses_projection_over_budget(tmp_path):
    def fake(**kw):
        raise AssertionError("must not be sent")

    with pytest.raises(SystemExit):
        CR.run(True, out=tmp_path / "o.jsonl", limit=3, client=_client(tmp_path, fake, cap=0.001))
    assert not (tmp_path / "ledger.jsonl").exists()
