"""scripts/longtask_cloud.py with a fake provider (no network, no key): rows, billed vs upper bound vs priced, the
projection stop rule, the Sonnet decision, the refusals."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("longtask_cloud", REPO / "scripts" / "longtask_cloud.py")
LC = importlib.util.module_from_spec(_spec)
sys.modules["longtask_cloud"] = LC
_spec.loader.exec_module(LC)
cc = LC.cc
L = LC.L


def oracle_provider(seen, seed=LC.SEED, tids=LC.CLOUD_TASKS):
    """Answers each task with its oracle calls as native tool calls, then a text summary. Usage from text length,
    most of the prompt a cache read after the first call of a task."""
    by_path = {}
    for t, e in L.build_tasks(seed, tids):
        msgs, _ = L.oracle_transcript(t, e)
        by_path[t.path] = [m["tool_calls"][0]["function"] for m in msgs if m.get("tool_calls")]

    def fn(*, model_id, messages, api_key, **kw):
        seen.append({"model_id": model_id, "kw": kw, "n": len(messages)})
        prompt = max(600, len(json.dumps(messages)) // 4)
        n_assist = sum(1 for m in messages if m.get("role") == "assistant")
        read = int(prompt * 0.8) if n_assist else 0
        usage = {"prompt_tokens": prompt, "completion_tokens": 40, "cache_read_input_tokens": read,
                 "cache_creation_input_tokens": prompt - read}
        path = next((p for p in by_path if len(messages) > 1 and p in str(messages[1].get("content"))), None)
        calls = by_path.get(path, [])
        if path and n_assist < len(calls) and kw.get("tool_choice") != "none" and kw.get("tools"):
            c = calls[n_assist]
            msg = {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"t{n_assist}", "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}]}
        else:
            msg = {"role": "assistant", "content": "Done."}
        return {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": usage}
    return fn


def make_client(tmp_path, fn, cap=50.0):
    return cc.CloudClient(mode="real", api_key="fake-not-a-key", completion_fn=fn, role_override="controller",
                          ledger_path=tmp_path / "ledger.jsonl", spend_cap_usd=cap)


def test_run_haiku_tasks_rows_and_costs(tmp_path):
    seen = []
    client = make_client(tmp_path, oracle_provider(seen))
    res = LC.run(client, tmp_path / "out.jsonl", log=lambda *_: None, retry_wait_s=0, job_cap=6.0)
    rows = res["rows"]
    tr = [r for r in rows if r["record"] == "longtask_cloud_task"]
    assert res["decision"] == "run" and res["stopped"] is None
    haiku = [r for r in tr if r["model_id"] == LC.HAIKU]
    assert [r["task_id"] for r in haiku] == list(LC.CLOUD_TASKS)
    assert all(r["correct"] for r in haiku)
    for r in haiku:
        assert r["usd_billed"] <= r["usd_upper_bound"] + 1e-12
        pe = r["priced_estimate_usd"]
        assert set(pe) == {"none", "realistic", "best"} and pe["best"] <= pe["realistic"] <= pe["none"] + 1e-12
        assert r["input_tokens"] > 0 and r["seconds"] >= 0 and r["n_calls"] == r["ref_steps"] + 1
    # thinking disabled for haiku as all_cloud_b3
    assert all(s["kw"].get("thinking") == {"type": "disabled"} for s in seen if s["model_id"] == LC.HAIKU)
    sd = [r for r in rows if r["record"] == "sonnet_decision"]
    assert len(sd) == 1 and rows[-1]["record"] == "run_end"
    if sd[0]["run"]:
        assert any(r["model_id"] == LC.SONNET for r in tr)
    s = res["summary"][LC.HAIKU]
    assert s["n_tasks"] == 5 and s["correct"] == 5


def test_projection_over_cap_stops_before_any_task(tmp_path):
    client = make_client(tmp_path, oracle_provider([]))
    res = LC.run(client, tmp_path / "out.jsonl", log=lambda *_: None, retry_wait_s=0, job_cap=0.00001)
    assert res["decision"] == "stop" and "projection" in res["stopped"]
    assert not [r for r in res["rows"] if r["record"] == "longtask_cloud_task"]


def test_sonnet_skipped_when_estimate_over_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(LC, "SONNET_MAX_USD", 0.0)
    client = make_client(tmp_path, oracle_provider([]))
    res = LC.run(client, tmp_path / "out.jsonl", log=lambda *_: None, retry_wait_s=0)
    sd = [r for r in res["rows"] if r["record"] == "sonnet_decision"][0]
    assert sd["run"] is False
    assert not [r for r in res["rows"] if r.get("model_id") == LC.SONNET and r["record"] == "longtask_cloud_task"]


def test_main_refusals(tmp_path, monkeypatch):
    assert LC.main(["--ledger", str(tmp_path / "l.jsonl")]) == 2
    monkeypatch.delenv(LC.ACB.KEY_ENV, raising=False)
    assert LC.main(["--live", "--ledger", str(tmp_path / "l.jsonl")]) == 2
    monkeypatch.setenv(LC.ACB.KEY_ENV, "fake")
    assert LC.main(["--live", "--job-cap", "7", "--ledger", str(tmp_path / "l.jsonl")]) == 2


def test_reprice_and_priced_estimate():
    table = LC.CR.load_price_table(LC.PRICE_FILE)
    calls = [{"input_tokens": 1000, "uncached_input_tokens": 1000, "cache_write_tokens": 0, "cache_read_tokens": 0,
              "output_tokens": 100}]
    h = LC.reprice(calls, LC.CR.model_rates(table, LC.HAIKU))
    s = LC.reprice(calls, LC.CR.model_rates(table, LC.SONNET))
    assert s > h > 0
    prices = json.loads(LC.PRICE_FILE.read_text(encoding="utf-8"))
    pe = LC.priced_estimate(prices, LC.HAIKU, calls)
    assert abs(pe["none"] - (1000 * 0.1 + 100 * 0.5) / 1e6) < 1e-12
