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


# ── --thinking-off (full-context rerun of 2026-10-21) ───────────────────────────────────────────────

def _resp_fr(content="", calls=(), usage=(1000, 20, 0, 0), finish="stop"):
    r = _resp(content, calls, usage)
    r["choices"][0]["finish_reason"] = finish
    return r


def test_thinking_off_sends_setting_no_temperature_and_records_finish(tmp_path):
    seen = []

    def fake(*, model_id, messages, api_key, **kw):
        seen.append(kw)
        return _resp_fr('{"answer": "no idea"}')

    rows = CR.run(True, out=tmp_path / "o.jsonl", limit=2, client=_client(tmp_path, fake), thinking_off=True)
    assert rows[0]["thinking_sent"] == {"type": "between_tools"} and rows[0]["temperature_sent"] is None
    assert all(kw.get("thinking") == {"type": "between_tools"} and "temperature" not in kw for kw in seen)
    tasks = [r for r in rows if r["record"] == "cloud_redo_task"]
    assert len(tasks) == 2 and tasks[0]["cloud_calls"][0]["finish_reason"] == "stop"
    assert rows[-1]["stopped"] is None
    # written incrementally, same rows on disk
    assert len((tmp_path / "o.jsonl").read_text().splitlines()) == len(rows)


def test_thinking_off_stops_on_empty_reply_at_cap(tmp_path):
    def fake(*, model_id, messages, api_key, **kw):
        if kw["max_tokens"] == 8:
            return _resp_fr("OK")
        return _resp_fr("", usage=(1000, 384, 0, 0), finish="length")

    rows = CR.run(True, out=tmp_path / "o.jsonl", limit=3, client=_client(tmp_path, fake), thinking_off=True)
    tasks = [r for r in rows if r["record"] == "cloud_redo_task"]
    assert len(tasks) == 1 and tasks[0]["empty_at_cap"] is True and tasks[0]["n_calls"] == 1
    assert rows[-1]["record"] == "run_end" and "max_tokens" in rows[-1]["stopped"]


def test_job_cap_refuses_on_executed_projection(tmp_path):
    def fake(**kw):
        raise AssertionError("must not be sent")

    src = CR.load_rows(CR.SOURCE)
    w = CR.wrong_tasks(src)[:2]
    ex = tmp_path / "ex.jsonl"
    ex.write_text("\n".join(json.dumps({"record": "cloud_redo_task", "mode": CR.MODE_LIVE, "attempt_id": r["attempt_id"],
                                        "turn_idx": r["turn_idx"], "n_calls": 1, "usd_ledger": 3.0, "usd": 1.0,
                                        "cloud_end_state_ok": i == 0}) for i, r in enumerate(w)))
    proj = CR.projection_from_executed(w, CR.load_rows(ex))
    assert proj["n_matched"] == 2 and proj["ledger_usd"] == 6.0
    with pytest.raises(SystemExit, match="job cap"):
        CR.run(True, out=tmp_path / "o.jsonl", limit=2, client=_client(tmp_path, fake), thinking_off=True,
               job_cap=5.0, executed_file=ex)
    assert not (tmp_path / "ledger.jsonl").exists()


def test_main_thinking_off_default_output_and_job_cap_client(monkeypatch, tmp_path):
    got = {}
    monkeypatch.setenv(CR.KEY_ENV, "test-not-a-key")
    led = tmp_path / "ledger.jsonl"

    def fake_run(live, source, out, **kw):
        got.update(kw, out=out)
        return [{"record": "run_end", "stopped": None}]

    class FakeClient:
        def __init__(self, mode, spend_cap_usd=None, ledger_path=None, **kw):
            self.mode, self.spend_cap_usd, self.ledger_path = mode, spend_cap_usd, ledger_path

        def _read_running_total(self):
            return 21.0

    monkeypatch.setattr(CR, "run", fake_run)
    monkeypatch.setattr(CR.cc, "CloudClient", FakeClient)
    assert CR.main(["--live", "--thinking-off", "--job-cap", "10", "--ledger", str(led)]) == 0
    assert got["out"] == CR.OUT_FULL_CONTEXT and got["thinking_off"] is True and got["job_cap"] == 10.0
    assert got["client"].spend_cap_usd == 31.0 and got["client"].ledger_path == led
    assert got["ledger_total_at_start"] == 21.0 and got["executed_file"] == CR.V1_EXECUTED


# ── --max-tokens / --resume / --client-cap-total (2026-10-22) ───────────────────────────────────────

def _call(fr="stop", out=100, chars=10, ntc=0):
    return {"finish_reason": fr, "output_tokens": out, "content_chars": chars, "n_tool_calls": ntc}


def test_hit_limit_rules():
    assert CR.hit_limit({"cloud_calls": [_call("length", 384, 200, 2)]})
    assert CR.hit_limit({"cloud_calls": [_call("tool_calls", 384, 50, 1)]})      # output at the old limit
    assert CR.hit_limit({"cloud_calls": [_call("tool_calls", 100, 50, 1), _call("stop", 10, 0, 0)]})   # empty reply
    assert not CR.hit_limit({"cloud_calls": [_call("tool_calls", 200, 50, 1), _call("stop", 150, 300, 0)]})


def _prior_file(tmp_path, wrong, hit_idx):
    rows = [{"record": "run_start", "mode": CR.MODE_LIVE}]
    for i, r in enumerate(wrong):
        calls = [_call("length", 384, 0, 0)] if i in hit_idx else [_call("stop", 120, 40, 0)]
        rows.append({"record": "cloud_redo_task", "mode": CR.MODE_LIVE, "attempt_id": r["attempt_id"],
                     "turn_idx": r["turn_idx"], "task_id": r["task_id"], "seed": r["seed"], "tier": "t",
                     "cloud_calls": calls, "cloud_end_state_ok": i not in hit_idx, "usd_ledger": 0.05, "usd": 0.02,
                     "n_calls": 1, "seconds": 1.0})
    rows.append({"record": "run_end", "stopped": "x"})
    p = tmp_path / "prior.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_resume_skips_done_reruns_hit_and_records_supersede(tmp_path):
    w = CR.wrong_tasks(CR.load_rows(CR.SOURCE))[:4]
    prior = _prior_file(tmp_path, w[:3], hit_idx={1})
    seen = []

    def fake(*, model_id, messages, api_key, **kw):
        seen.append(kw)
        return _resp_fr('{"answer": "no idea"}', usage=(1000, 30, 0, 0))

    out = tmp_path / "part2.jsonl"
    rows = CR.run(True, out=out, limit=4, client=_client(tmp_path, fake), thinking_off=True, max_tokens=2048,
                  resume_from=prior)
    start = rows[0]
    assert start["max_tokens"] == 2048 and "2048" in start["protocol_difference"]
    assert start["n_wrong_tasks"] == 4 and start["n_tasks_this_run"] == 2
    assert start["resume_kept_tasks"] == 2 and start["resume_superseded_tasks"] == 1
    sup = [r for r in rows if r["record"] == "supersede"]
    assert len(sup) == 1 and (sup[0]["attempt_id"], sup[0]["turn_idx"]) == (w[1]["attempt_id"], w[1]["turn_idx"])
    tasks = [r for r in rows if r["record"] == "cloud_redo_task"]
    assert [(r["attempt_id"], r["turn_idx"]) for r in tasks] == [(x["attempt_id"], x["turn_idx"]) for x in (w[1], w[3])]
    assert all(kw["max_tokens"] == 2048 for kw in seen if kw["max_tokens"] != 8)
    assert tasks[0]["cloud_calls"][0]["finish_reason"] == "stop" and tasks[0]["cloud_calls"][0]["output_tokens"] == 30
    merged = CR.merged_rows([prior, out])
    keys = {(r["attempt_id"], r["turn_idx"]): r for r in merged}
    assert len(merged) == 4 and keys[(w[1]["attempt_id"], w[1]["turn_idx"])]["mode"] == CR.MODE_LIVE
    assert keys[(w[1]["attempt_id"], w[1]["turn_idx"])].get("cloud_n_tool_calls") is not None   # the new row
    # the original file is untouched
    assert len(prior.read_text().splitlines()) == 5


def test_resume_stop_on_empty_at_2048_stays_on(tmp_path):
    w = CR.wrong_tasks(CR.load_rows(CR.SOURCE))[:3]
    prior = _prior_file(tmp_path, w[:1], hit_idx=set())

    def fake(*, model_id, messages, api_key, **kw):
        if kw["max_tokens"] == 8:
            return _resp_fr("OK")
        return _resp_fr("", usage=(1000, 2048, 0, 0), finish="length")

    rows = CR.run(True, out=tmp_path / "p2.jsonl", limit=3, client=_client(tmp_path, fake), thinking_off=True,
                  max_tokens=2048, resume_from=prior)
    assert len([r for r in rows if r["record"] == "cloud_redo_task"]) == 1 and rows[-1]["stopped"]


def test_resume_refuses_when_projection_passes_client_cap(tmp_path):
    def fake(**kw):
        raise AssertionError("must not be sent")

    w = CR.wrong_tasks(CR.load_rows(CR.SOURCE))[:3]
    prior = _prior_file(tmp_path, w[:1], hit_idx=set())
    with pytest.raises(SystemExit, match="remaining budget"):
        CR.run(True, out=tmp_path / "p2.jsonl", limit=3, client=_client(tmp_path, fake, cap=0.0001), thinking_off=True,
               max_tokens=2048, resume_from=prior, executed_file=CR.V1_EXECUTED)
    assert not (tmp_path / "ledger.jsonl").exists()


def test_main_resume_client_cap_total(monkeypatch, tmp_path):
    got = {}
    monkeypatch.setenv(CR.KEY_ENV, "test-not-a-key")
    prior = tmp_path / "prior.jsonl"
    prior.write_text("{}\n")

    def fake_run(live, source, out, **kw):
        got.update(kw, out=out)
        return [{"record": "run_end", "stopped": None}]

    class FakeClient:
        def __init__(self, mode, spend_cap_usd=None, ledger_path=None, **kw):
            self.spend_cap_usd, self.ledger_path = spend_cap_usd, ledger_path

        def _read_running_total(self):
            return 25.0

    monkeypatch.setattr(CR, "run", fake_run)
    monkeypatch.setattr(CR.cc, "CloudClient", FakeClient)
    out = tmp_path / "p2.jsonl"
    assert CR.main(["--live", "--thinking-off", "--resume", "--resume-from", str(prior), "--out", str(out),
                    "--max-tokens", "2048", "--client-cap-total", "43.1913", "--ledger", str(tmp_path / "l")]) == 0
    assert got["client"].spend_cap_usd == 43.1913 and got["max_tokens"] == 2048 and got["resume_from"] == [prior]
    assert got["out"] == out
    # refusals: above the hard cap, both caps, writing over the resumed file, an existing continuation file
    assert CR.main(["--live", "--resume", "--resume-from", str(prior), "--out", str(tmp_path / "x.jsonl"),
                    "--client-cap-total", "51"]) == 2
    assert CR.main(["--live", "--resume", "--resume-from", str(prior), "--out", str(tmp_path / "y.jsonl"),
                    "--client-cap-total", "40", "--job-cap", "5"]) == 2
    assert CR.main(["--live", "--resume", "--resume-from", str(prior), "--out", str(prior)]) == 2
    out.write_text("x\n")
    assert CR.main(["--live", "--resume", "--resume-from", str(prior), "--out", str(out)]) == 2


# ── transient retry and multi-part resume (2026-10-22) ──────────────────────────────────────────────

class InternalServerError(Exception):   # same class name as litellm's
    pass


class BadRequestError(Exception):
    pass


def test_transient_error_retried_with_backoff_and_no_ledger_row(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(CR, "_sleep", waits.append)
    state = {"n": 0}

    def fake(*, model_id, messages, api_key, **kw):
        state["n"] += 1
        if kw["max_tokens"] != 8 and state["n"] in (2, 3):
            raise InternalServerError("AnthropicException - Server disconnected without sending a response")
        return _resp_fr('{"answer": "no idea"}')

    client = _client(tmp_path, fake)
    rows = CR.run(True, out=tmp_path / "o.jsonl", limit=1, client=client, thinking_off=True, max_tokens=2048)
    task = [r for r in rows if r["record"] == "cloud_redo_task"][0]
    assert waits == [10.0, 30.0] and task["retries"] == 2 and task["cloud_calls"][0]["retries"] == 2
    assert rows[-1]["stopped"] is None
    led = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(led) == 2           # warm-up + the one billed task call; failed sends write nothing
    assert client._reserved_usd == 0


def test_transient_error_past_last_retry_stops_with_run_end(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(CR, "_sleep", waits.append)

    def fake(*, model_id, messages, api_key, **kw):
        if kw["max_tokens"] == 8:
            return _resp_fr("OK")
        raise InternalServerError("overloaded")

    client = _client(tmp_path, fake)
    rows = CR.run(True, out=tmp_path / "o.jsonl", limit=2, client=client, thinking_off=True, max_tokens=2048)
    assert waits == [10.0, 30.0, 90.0]
    assert rows[-1]["record"] == "run_end" and "InternalServerError" in rows[-1]["stopped"]
    assert not [r for r in rows if r["record"] == "cloud_redo_task"] and client._reserved_usd == 0


def test_non_transient_error_stops_without_retry(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(CR, "_sleep", waits.append)

    def fake(*, model_id, messages, api_key, **kw):
        if kw["max_tokens"] == 8:
            return _resp_fr("OK")
        raise BadRequestError("invalid request: tools malformed")

    rows = CR.run(True, out=tmp_path / "o.jsonl", limit=2, client=_client(tmp_path, fake), thinking_off=True)
    assert waits == [] and "BadRequestError" in rows[-1]["stopped"]
    assert (tmp_path / "o.jsonl").read_text().splitlines()[-1].startswith('{"harness"')   # run_end on disk


def test_is_transient_classes():
    assert CR.is_transient(InternalServerError("x"))
    assert CR.is_transient(ConnectionResetError("reset"))
    assert CR.is_transient(Exception("Error code: 529 overloaded_error"))
    assert not CR.is_transient(BadRequestError("bad"))
    assert not CR.is_transient(cc.SpendCapExceeded("cap"))


def test_multi_part_resume_and_merge(tmp_path):
    w = CR.wrong_tasks(CR.load_rows(CR.SOURCE))[:5]
    first = tmp_path / "run.jsonl"
    key = lambda r: (r["attempt_id"], r["turn_idx"])

    def task(r, calls, ts, ok=True):
        return {"record": "cloud_redo_task", "mode": CR.MODE_LIVE, "attempt_id": r["attempt_id"],
                "turn_idx": r["turn_idx"], "task_id": r["task_id"], "seed": r["seed"], "tier": "t", "ts_utc": ts,
                "cloud_calls": calls, "cloud_end_state_ok": ok, "usd_ledger": 0.05, "usd": 0.02, "n_calls": 1,
                "seconds": 1.0}

    # part 1 (384): w0 fine, w1 hit, w2 hit
    p1 = [{"record": "run_start", "max_tokens": 384}, task(w[0], [_call()], "a0"),
          task(w[1], [_call("length", 384, 0, 0)], "a1", False), task(w[2], [_call("length", 384, 5, 1)], "a2", False)]
    first.write_text("\n".join(json.dumps(r) for r in p1) + "\n")
    # part 2 (2048), crashed (no run_end): supersedes w1 and w2, reran w1 (output 600, fine at 2048), did w3
    sup = lambda r, ts: {"record": "supersede", "attempt_id": r["attempt_id"], "turn_idx": r["turn_idx"],
                         "old_ts_utc": ts}
    p2 = [{"record": "run_start", "max_tokens": 2048}, sup(w[1], "a1"), sup(w[2], "a2"),
          task(w[1], [_call("stop", 600, 50, 0)], "b1"), task(w[3], [_call()], "b3")]
    (tmp_path / "run_part2.jsonl").write_text("\n".join(json.dumps(r) for r in p2) + "\n")
    parts = CR.part_files(first)
    assert parts == [first, tmp_path / "run_part2.jsonl"] and CR.next_part_path(first) == tmp_path / "run_part3.jsonl"
    skip, new_sup, kept = CR.resume_plan_parts(parts, 2048)
    assert skip == {key(w[0]), key(w[1]), key(w[3])} and new_sup == [] and len(kept) == 3
    merged = {key(r): r for r in CR.merged_rows(parts)}
    assert set(merged) == {key(w[0]), key(w[1]), key(w[3])} and merged[key(w[1])]["ts_utc"] == "b1"

    def fake(*, model_id, messages, api_key, **kw):
        return _resp_fr('{"answer": "no idea"}')

    out = CR.next_part_path(first)
    rows = CR.run(True, out=out, limit=5, client=_client(tmp_path, fake), thinking_off=True, max_tokens=2048,
                  resume_from=parts)
    done = [key(r) for r in rows if r["record"] == "cloud_redo_task"]
    assert done == [key(w[2]), key(w[4])] and rows[0]["resume_kept_tasks"] == 3
    assert not [r for r in rows if r["record"] == "supersede"]     # w2 was already superseded in part 2
    merged = CR.merged_rows(CR.part_files(first))
    assert len(merged) == 5
