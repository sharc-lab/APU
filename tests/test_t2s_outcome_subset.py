"""harness/t2s_outcome_table.py's T2S subset path (docs/T2S_WEEK_PLAN.md step 5): subset parity with
x2_outcome_table.qwen32b_subset, the canary gate, item-major multi-model order, resume, the queue exit, and that the
queued job's rows go through the fixed GSM8K scorer (#### extraction, SCORER_VERSION 2) and the lenient column."""
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "scripts"))

import t2s_outcome_table as ot  # noqa: E402
import x2_outcome_table as x2ot  # noqa: E402


def _items(n=30):
    fams = ["gsm8k", "function_calling", "longdoc_qa"]
    return [{"item_id": f"i{k:03d}", "family": fams[k % 3], "prompt": f"p{k}", "prompt_tokens": 100 + k,
             "grading": {"method": "exact_substring"}, "oracle_answer": "x"} for k in range(n)]


def _weights(items):
    return {it["item_id"]: (0.0 if k % 7 == 0 else 1.0 + (k % 5)) for k, it in enumerate(items)}


def test_subset_draw_is_identical_to_x2_qwen32b_subset():
    items = _items(60)
    w = _weights(items)
    for n, seed in ((10, 1), (25, x2ot.QWEN32B_SUBSET_SEED), (60, 7)):
        assert ot.trace_weighted_subset(items, w, n, seed) == x2ot.qwen32b_subset(items, w, n=n, seed=seed)
    assert ot.SUBSET_SEED_DEFAULT == x2ot.QWEN32B_SUBSET_SEED


def test_subset_on_the_real_pack_matches_the_recorded_x2_subset():
    """The x2 file's own qwen32b_subset record (results/x2_outcome_table_v3.jsonl) is reproduced by the T2S draw."""
    rec = next(json.loads(l) for l in (REPO / "results" / "x2_outcome_table_v3.jsonl").read_text().splitlines()
               if '"qwen32b_subset"' in l)
    try:
        items = ot.load_items_trace_weighted(REPO)
    except Exception as e:  # pack loader dependency missing on this machine
        pytest.skip(f"pack loader unavailable: {e!r}")
    w = json.loads((REPO / "results" / "workload_pack" / "item_weights_trace_weighted.json").read_text())
    assert sorted(ot.trace_weighted_subset(items, w, 100, rec["seed"])) == rec["item_ids"]


def test_canary_helpers_match_x2():
    items = _items(30)
    assert ot.canary_items(items) == x2ot.canary_items(items)
    for rows in ([{"http_status": 200, "score": 1.0}] * 10,
                 [{"http_status": 500, "error": "alloc_tensor_range: failed", "score": 0.0}] * 2
                 + [{"http_status": 200, "score": 1.0}] * 8,
                 [{"http_status": 200, "score": 0.2}] * 10):
        assert ot.canary_gate_check(rows) == x2ot.canary_gate_check(rows)
        for r in rows:
            assert ot.classify_error_cause(r) == x2ot.classify_error_cause(r)
    assert ot.LLAMA_SERVER_THINKING_ARGS == x2ot.LLAMA_SERVER_THINKING_ARGS
    assert ot.LLAMA_REQUEST_THINKING_FIELDS == x2ot.LLAMA_REQUEST_THINKING_FIELDS


def _fake_runners(log, fail=()):
    def make(config):
        def fn(item, mk, canary):
            log.append((item["item_id"], mk, config, canary))
            bad = (mk, config) in fail
            row = {"record": "outcome_row", "item_id": item["item_id"], "family": item["family"], "config": config,
                   "model_key": mk, "http_status": 500 if bad else 200, "error": "boom" if bad else None,
                   "score": 0.0 if bad else 1.0, "canary": canary}
            return row
        return fn
    return {c: make(c) for c in ot.CONFIGS}


def test_run_records_the_subset_gates_each_model_config_and_runs_item_major(tmp_path):
    items = _items(30)
    w = {it["item_id"]: 1.0 for it in items}
    log = []
    out = tmp_path / "t2s_outcome_subset.jsonl"
    note = ot.run(out, models=["llama3.1:8b", "qwen3-8b"], subset_n=5, subset_seed=3, canary_gate=True,
                  runners=_fake_runners(log, fail={("qwen3-8b", "ollama_igpu_enable")}), items=items, weights=w)
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    sub = [r for r in recs if r["record"] == "t2s_subset"]
    assert len(sub) == 1 and sub[0]["n"] == 5 and sub[0]["seed"] == 3
    gates = [r for r in recs if r["record"] == "canary_gate"]
    assert len(gates) == 2 * 3
    assert [r for r in recs if r["record"] == "ALERT"][0]["config"] == "ollama_igpu_enable"
    measured = [x for x in log if not x[3]]
    assert ("qwen3-8b", "ollama_igpu_enable") not in {(m, c) for _, m, c, _ in measured}
    first_item = measured[0][0]
    assert [(m, c) for i, m, c, _ in measured if i == first_item][:3] == [("llama3.1:8b", c) for c in ot.CONFIGS]
    assert set(i for i, *_ in measured) <= set(sub[0]["item_ids"])
    assert "canary gate halted" in note


def test_run_reuses_cached_rows_on_resume(tmp_path):
    items = _items(12)
    w = {it["item_id"]: 1.0 for it in items}
    out = tmp_path / "t2s_outcome_subset.jsonl"
    log = []
    runners = _fake_runners(log)

    def persist(fn):
        def inner(item, mk, canary):
            r = fn(item, mk, canary)
            ot.emit(out, r)
            return r
        return inner
    runners = {c: persist(f) for c, f in runners.items()}
    ot.run(out, models=["llama3.1:8b"], subset_n=4, subset_seed=1, runners=runners, items=items, weights=w)
    n_first = len(log)
    ot.run(out, models=["llama3.1:8b"], subset_n=4, subset_seed=1, runners=runners, items=items, weights=w)
    assert len(log) == n_first == 4 * 3
    assert sum(1 for x in out.read_text().splitlines() if '"t2s_subset"' in x) == 1


def test_queued_outcome_job_rows_use_the_fixed_gsm8k_scorer_and_lenient_column(monkeypatch, tmp_path):
    """Runs the real run_one_item_ollama for the queued job's models (model_key path) with a fake Ollama: a
    '#### 42' answer must score 1.0 strict (SCORER_VERSION 2), and the lenient column must be present."""
    item = {"item_id": "gsm_t", "family": "gsm8k", "prompt": "q", "prompt_tokens": 40,
            "grading": {"method": "final_number_match"}, "oracle_answer": "42"}

    class FakeClient:
        def chat(self, *a, **kw):
            assert kw.get("think") is False
            return {"message": "Adding them gives 42.\n#### 42", "status": 200, "prompt_eval_count": 40}

    fake_hc = types.SimpleNamespace(_resolve_ollama_exe_for_serve=lambda: "ollama", start_ollama_server=lambda: None,
                                    wait_for_ollama_ready=lambda timeout_s=60: True, stop_ollama_server=lambda: None)
    monkeypatch.setattr(ot, "hc", fake_hc)
    monkeypatch.setattr(ot.k1, "OllamaClient", FakeClient)
    monkeypatch.setattr(ot, "get_ollama_ps_context_length", lambda tag: 4096)
    monkeypatch.setattr(ot, "chat_template_fields", lambda **kw: {"chat_template_source": "ollama_library",
                                                                   "chat_template_sha256": "x"})
    out = tmp_path / "t2s_o.jsonl"
    for mk in ("llama3.1:8b", "qwen3-8b"):
        row = ot.run_one_item_ollama(item, "ollama_default", False, out, 60, model_key=mk, canary=False)
        assert row["score"] == 1.0 and row["score_strict"] == 1.0 and row["score_lenient"] == 1.0
        assert row["scorer_version"] == ot.SCORER_VERSION == 2 and row["format_ok"] is True
        assert row["model_id"] == ot.MODEL_MAP[mk][0] and row["model_key"] == mk
        assert row["chat_template_source"] == "ollama_library" and row["thinking_leak"] is False


def test_queue_entry_argv_reaches_the_model_key_path(monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    import t2s_week_queue as wq
    cmd = next(it["cmd"] for it in wq.build_queue() if it["id"] == "t2s_wk_outcome_subset")
    seen = {}
    monkeypatch.setattr(ot, "run", lambda out, **kw: seen.update(kw) or "completed 0 items")
    monkeypatch.setattr(ot, "hc", types.SimpleNamespace(require_host=lambda h: {"interactive_guard": True},
                                                        enforce_or_record_interactive_session=lambda c: {}))
    monkeypatch.delenv("APU_QUEUE_JOB_ID", raising=False)
    ot.main(cmd[2:])
    assert seen["models"] == ["llama3.1:8b", "qwen3-8b"] and seen["subset_n"] == 100 and seen["canary_gate"]


def test_main_advances_once_when_queued(monkeypatch):
    notes = []
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))
    monkeypatch.setattr(ot, "run", lambda out, **kw: "completed 3 items")
    monkeypatch.setenv("APU_QUEUE_JOB_ID", "t2s_wk_outcome_subset")
    assert ot.main(["--out", "x.jsonl"]) == "completed 3 items" and notes == ["completed 3 items"]
    notes.clear()

    def boom(out, **kw):
        raise RuntimeError("gpu gone")
    monkeypatch.setattr(ot, "run", boom)
    note = ot.main(["--out", "x.jsonl"])
    assert note.startswith("stopped:") and notes == [note]
    notes.clear()
    monkeypatch.delenv("APU_QUEUE_JOB_ID")
    monkeypatch.setattr(ot, "run", lambda out, **kw: "completed")
    ot.main(["--out", "x.jsonl"])
    assert notes == []


def test_legacy_invocation_keeps_its_row_shape(monkeypatch, tmp_path):
    """No --models: rows carry no model_key (the original single-model path) and resume keys default to llama."""
    out = tmp_path / "t2s_outcome_table_x.jsonl"
    ot.emit(out, {"record": "outcome_row", "item_id": "a", "config": "ollama_default", "family": "longdoc_qa",
                  "http_status": 200})
    assert ot.already_done_keys(out) == {("a", "ollama_default", "llama3.1:8b")}
