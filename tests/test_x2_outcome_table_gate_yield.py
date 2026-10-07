"""2026-10-07 X2 outcome-table restart: canary validity gate in the run loop, rolling-error ALERT, cached-row
reuse, the item-boundary yield contract (for the R2 harness), the qwen3-32b subset, and --reuse-from seeding.
No live calls: the runtimes are injected fakes."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import x2_outcome_table as x2  # noqa: E402


def _items():
    its = [{"item_id": f"gsm8k_{i}", "family": "gsm8k", "prompt_tokens": 50 + i, "prompt": "p"} for i in range(5)]
    its += [{"item_id": f"fc_{i}", "family": "function_calling", "prompt_tokens": 80 + i, "prompt": "p"} for i in range(5)]
    its += [{"item_id": f"long_{i}", "family": "longdoc_qa", "prompt_tokens": 9000 + i, "prompt": "p"} for i in range(4)]
    return its


def _weights(items):
    return {it["item_id"]: 1.0 / len(items) for it in items}


class FakeRunner:
    """Records calls, emits a row like the real runners. behavior(item, model, config) -> row fields."""

    def __init__(self, config, behavior):
        self.config, self.behavior, self.calls = config, behavior, []

    def __call__(self, item, model_key, target, out_path, call_timeout_s, canary=False):
        self.calls.append((item["item_id"], model_key, canary))
        row = {"record": "outcome_row", "item_id": item["item_id"], "family": item["family"], "config": self.config,
               "model_id": model_key, "ts_utc": x2.utc_iso(), "canary": canary, "scorer_version": 2}
        row.update(self.behavior(item, model_key, self.config))
        x2.emit(out_path, row)
        return row


def _ok(*_):
    return {"http_status": 200, "score": 1.0}


def _rows(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _run(tmp_path, behaviors, models=("llama3.1:8b",), yield_hook=lambda: None, items=None):
    items = items or _items()
    runners = {c: FakeRunner(c, behaviors.get(c, _ok)) for c in x2.CONFIGS}
    out = tmp_path / "out.jsonl"
    status = x2.run(out, list(models), runners=runners, yield_hook=yield_hook, items=items, weights=_weights(items))
    return out, runners, status


# ------------------------------------------------------------------------------------------------- canary gate
def test_canaries_run_first_for_every_model_config_and_pass(tmp_path):
    out, runners, status = _run(tmp_path, {})
    for c in x2.CONFIGS:
        first10 = runners[c].calls[:10]
        assert all(canary for _, _, canary in first10)
        assert {iid for iid, _, _ in first10} == {f"gsm8k_{i}" for i in range(5)} | {f"fc_{i}" for i in range(5)}
        # canaries are cached rows: the main loop never re-runs them, so 10 canaries + 4 longdoc = 14 calls
        assert len(runners[c].calls) == 14
    gates = [r for r in _rows(out) if r["record"] == "canary_gate"]
    assert len(gates) == 2 and all(g["passed"] for g in gates)
    assert status.startswith("completed")


def test_canary_gate_failure_halts_only_that_model_config_and_writes_alert(tmp_path):
    def broken(item, model, config):
        return {"http_status": 200, "score": 0.1}  # no errors, but mean score < 0.5
    out, runners, _ = _run(tmp_path, {"llama_server": broken})
    assert len(runners["llama_server"].calls) == 10  # canaries only, then halted
    assert len(runners["ollama_default"].calls) == 14  # unaffected
    alerts = [r for r in _rows(out) if r["record"] == "alert"]
    assert len(alerts) == 1 and alerts[0]["reason"] == "canary_gate_failed" and alerts[0]["config"] == "llama_server"


def test_canary_gate_fails_on_error_rate_above_10_percent(tmp_path):
    def flaky(item, model, config):
        if item["item_id"] in ("gsm8k_0", "gsm8k_1"):
            return {"http_status": None, "error": "timed out", "score": 0.0}
        return {"http_status": 200, "score": 1.0}
    out, runners, _ = _run(tmp_path, {"ollama_default": flaky})
    gate = [r for r in _rows(out) if r["record"] == "canary_gate" and r["config"] == "ollama_default"][0]
    assert gate["passed"] is False and abs(gate["error_rate"] - 0.2) < 1e-9 and gate["causes"]["timeout"] == 2


def test_canary_gate_counts_thinking_leak_as_error():
    rows = [{"http_status": 200, "score": 1.0, "thinking_leak": True} for _ in range(2)]
    rows += [{"http_status": 200, "score": 1.0} for _ in range(8)]
    passed, err, _ = x2.canary_gate_check(rows)
    assert passed is False and abs(err - 0.2) < 1e-9


def test_rolling_error_alert_fires_once_per_crossing():
    mon = x2.RollingErrorMonitor(threshold=0.10, min_n=10)
    alerts = []
    for i in range(30):
        row = {"http_status": None, "error": "timed out"} if i % 4 == 0 else {"http_status": 200}
        a = mon.add(("m", "c"), row)
        if a:
            alerts.append(a)
    assert len(alerts) == 1 and alerts[0]["rate"] > 0.10


def test_rolling_error_ignores_context_overflow():
    mon = x2.RollingErrorMonitor(threshold=0.10, min_n=10)
    for _ in range(20):
        assert mon.add(("m", "c"), {"http_status": 400, "error": "exceeds the available context size"}) is None


def test_rolling_alert_record_written_in_run(tmp_path):
    def bad_long(item, model, config):
        if item["family"] == "longdoc_qa":
            return {"http_status": None, "error": "some 500", "score": 0.0}
        return {"http_status": 200, "score": 1.0}
    items = _items() + [{"item_id": f"long_x{i}", "family": "longdoc_qa", "prompt_tokens": 9100 + i, "prompt": "p"}
                        for i in range(8)]
    out, _, _ = _run(tmp_path, {"ollama_default": bad_long}, items=items)
    alerts = [r for r in _rows(out) if r["record"] == "alert" and r["reason"] == "rolling_error_rate"]
    assert len(alerts) == 1 and alerts[0]["config"] == "ollama_default"


# ------------------------------------------------------------------------------------------------- cache reuse
def test_resume_reuses_valid_cached_rows_and_retries_connection_rows(tmp_path):
    out = tmp_path / "out.jsonl"
    for it in _items():
        x2.emit(out, {"record": "outcome_row", "item_id": it["item_id"], "family": it["family"], "config": "llama_server",
                      "model_id": "llama3.1:8b", "http_status": 200, "score": 1.0, "scorer_version": 2})
    x2.emit(out, {"record": "outcome_row", "item_id": "long_0", "family": "longdoc_qa", "config": "ollama_default",
                  "model_id": "llama3.1:8b", "http_status": None, "chat_outcome": "infra_not_ready", "score": 0.0})
    items = _items()
    runners = {c: FakeRunner(c, _ok) for c in x2.CONFIGS}
    x2.run(out, ["llama3.1:8b"], runners=runners, yield_hook=lambda: None, items=items, weights=_weights(items))
    assert runners["llama_server"].calls == []  # everything cached, canaries included
    assert ("long_0", "llama3.1:8b", False) in runners["ollama_default"].calls  # connection row retried


# ------------------------------------------------------------------------------------------------- yield contract
def _queue():
    return [
        {"id": "a_done", "cmd": ["x"], "status": "done"},
        {"id": "x2_outcome_table_v3", "cmd": ["py", "x2_outcome_table.py", "--out", "o.jsonl"], "status": "running"},
        {"id": "other_pending", "cmd": ["y"], "status": "pending"},
        {"id": "r2_validation", "cmd": ["z"], "status": "pending"},
    ]


def test_build_yield_queue_inserts_resume_after_first_following_pending():
    new, resume = x2.build_yield_queue(_queue(), "x2_outcome_table_v3")
    assert [it["id"] for it in new] == ["a_done", "x2_outcome_table_v3", "other_pending",
                                        "x2_outcome_table_v3_resume1", "r2_validation"]
    assert resume["status"] == "pending" and resume["cmd"] == _queue()[1]["cmd"]
    # nothing else touched
    assert [dict(it) for it in new if it["id"] != resume["id"]] == _queue()


def test_build_yield_queue_numbering_and_no_following_pending():
    q = [{"id": "t", "cmd": ["c"], "status": "done"}, {"id": "t_resume1", "cmd": ["c"], "status": "running"}]
    new, resume = x2.build_yield_queue(q, "t_resume1")
    assert resume["id"] == "t_resume2"
    assert [it["id"] for it in new] == ["t", "t_resume1", "t_resume2"]


class FakeQueue:
    def __init__(self, items):
        self.items = items
        self.writes = 0

    def read_queue(self):
        return [dict(it) for it in self.items]

    def write_queue(self, items):
        self.items = items
        self.writes += 1


def test_check_and_yield_deletes_flag_and_writes_queue(tmp_path):
    flag = tmp_path / "yield_x2_outcome_table"
    fq = FakeQueue(_queue())
    assert x2.check_and_yield(flag, fq, "x2_outcome_table_v3") is None  # no flag, no-op
    flag.write_text("r2")
    resume = x2.check_and_yield(flag, fq, "x2_outcome_table_v3")
    assert resume["id"] == "x2_outcome_table_v3_resume1"
    assert not flag.exists() and fq.writes == 1


def test_check_and_yield_noop_outside_queue_launch(tmp_path, monkeypatch):
    monkeypatch.delenv("APU_QUEUE_JOB_ID", raising=False)
    flag = tmp_path / "yield_x2_outcome_table"
    flag.write_text("r2")
    assert x2.check_and_yield(flag, FakeQueue(_queue())) is None
    assert flag.exists()  # left for the real job


def test_run_yields_at_item_boundary_and_resume_reuses_cache(tmp_path):
    out = tmp_path / "out.jsonl"
    items = _items()
    state = {"n": 0}

    def hook():
        state["n"] += 1
        return {"id": "x_resume1"} if state["n"] == 3 else None  # yield before the 3rd item

    runners = {c: FakeRunner(c, _ok) for c in x2.CONFIGS}
    status = x2.run(out, ["llama3.1:8b"], runners=runners, yield_hook=hook, items=items, weights=_weights(items))
    assert status == "yielded at item boundary"
    assert any(r["record"] == "yield" for r in _rows(out))
    n_first = sum(len(r.calls) for r in runners.values())
    runners2 = {c: FakeRunner(c, _ok) for c in x2.CONFIGS}
    x2.run(out, ["llama3.1:8b"], runners=runners2, yield_hook=lambda: None, items=items, weights=_weights(items))
    seen1 = {(c, iid) for c, r in runners.items() for iid, _, _ in r.calls}
    seen2 = {(c, iid) for c, r in runners2.items() for iid, _, _ in r.calls}
    assert not (seen1 & seen2)  # nothing re-run after resume
    assert n_first + sum(len(r.calls) for r in runners2.values()) == 2 * 14


# ------------------------------------------------------------------------------------------------- 32b subset
def test_qwen32b_subset_deterministic_weighted_and_sized():
    items = [{"item_id": f"i{k}"} for k in range(300)]
    w = {f"i{k}": (0.0 if k < 50 else 1.0 + (k % 7)) for k in range(300)}
    a = x2.qwen32b_subset(items, w, n=100, seed=1)
    b = x2.qwen32b_subset(list(reversed(items)), w, n=100, seed=1)
    assert a == b and len(a) == 100
    assert not any(int(i[1:]) < 50 for i in a)  # zero-weight items never drawn


def test_run_restricts_qwen32b_to_subset_and_records_ids(tmp_path):
    items = _items()
    runners = {c: FakeRunner(c, _ok) for c in x2.CONFIGS}
    out = tmp_path / "out.jsonl"
    old = x2.QWEN32B_SUBSET_N
    try:
        x2.QWEN32B_SUBSET_N = 2
        subset = x2.qwen32b_subset(items, _weights(items), n=2)
        x2.run(out, ["qwen3-32b"], runners=runners, yield_hook=lambda: None, items=items, weights=_weights(items))
    finally:
        x2.QWEN32B_SUBSET_N = old
    rec = [r for r in _rows(out) if r["record"] == "qwen32b_subset"]
    assert len(rec) == 1
    non_canary = {iid for iid, _, canary in runners["llama_server"].calls if not canary}
    assert non_canary <= subset


# ------------------------------------------------------------------------------------------------- seeding
def test_seed_from_previous_run_copies_only_valid_rows_once(tmp_path):
    prev = tmp_path / "weekend.jsonl"
    rows = [
        {"record": "outcome_row", "item_id": "a", "config": "ollama_default", "model_id": "qwen3-8b",
         "error": "<urlopen error [WinError 10061] ...>", "http_status": None, "ts_utc": "2026-10-05T20:00:00+00:00"},
        {"record": "outcome_row", "item_id": "a", "config": "llama_server", "model_id": "qwen3-8b", "http_status": 200},
        {"record": "outcome_row", "item_id": "a", "config": "llama_server", "model_id": "llama3.1:8b", "http_status": 200},
        {"record": "outcome_row", "item_id": "b", "config": "ollama_default", "model_id": "llama3.1:8b",
         "http_status": 500, "error": '{"error":"cudaMalloc failed: out of memory"}'},
    ]
    prev.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    rec = x2.seed_from_previous_run(out, prev)
    assert rec["n_copied"] == 1
    assert rec["tagged"] == {"invalid_race": 1, "invalid_thinking": 1, "invalid_infra_oom": 1}
    tagged = _rows(prev)
    assert tagged[0]["invalid_race_cause"] == "system_ollama_exe_resolution"
    assert x2.seed_from_previous_run(out, prev) is None  # idempotent
    copied = [r for r in _rows(out) if r["record"] == "outcome_row"]
    assert len(copied) == 1 and copied[0]["model_id"] == "llama3.1:8b" and copied[0]["reused_from"] == "weekend.jsonl"


def test_final_number_match_extracts_number_after_last_hashes():
    g = x2.load_graders()
    item = {"grading": {"method": "final_number_match"}, "oracle_answer": "3"}
    assert x2.score_response(g, item, "Half of 2 is 1.\n### Final Answer:\n#### 3") == 1.0
    assert x2.score_response(g, item, "#### 2\nwait, recheck\n#### 3") == 1.0
    assert x2.score_response(g, item, "**#### 3**") == 1.0
    assert x2.score_response(g, item, "the answer is 3") == 0.0  # format the prompt asks for is required
    big = {"grading": {"method": "final_number_match"}, "oracle_answer": "1200"}
    assert x2.score_response(g, big, "#### 1,200") == 1.0


def test_pre_fix_gsm8k_rows_are_not_valid_cache():
    assert x2.row_is_valid({"family": "gsm8k", "http_status": 200, "score": 0.0}) is False
    assert x2.row_is_valid({"family": "gsm8k", "http_status": 200, "score": 1.0, "scorer_version": 2}) is True
    assert x2.row_is_valid({"family": "longdoc_qa", "http_status": 200, "score": 1.0}) is True


def test_thinking_verify_verdict_ignores_finish_reason():
    import x2_thinking_verify as tv
    ok = {"record": "thinking_verify_call", "runtime": "llama_server", "model_id": "m", "mechanism": "production",
          "http_status": 200, "reasoning_chars": 0, "content_chars": 600, "finish_reason": "length"}
    leak = dict(ok, mechanism="baseline", reasoning_chars=900, content_chars=0)
    w = tv.works_from_calls([ok, ok, leak])
    assert w == {"llama_server/m/production": True, "llama_server/m/baseline": False}


def test_safe_name_strips_colon():
    assert ":" not in x2.safe_name("llama3.1:8b")
