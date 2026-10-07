"""2026-10-06 validity overhaul: thinking-mode fix, invalid-row tagging, the canary validity gate, and
per-cause error classification, for harness/x2_outcome_table.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import x2_outcome_table as x2  # noqa: E402


# --------------------------------------------------------------------------------------------- classify_error_cause
def test_classify_error_cause_none_for_a_clean_row():
    assert x2.classify_error_cause({"http_status": 200}) == "none"


def test_classify_error_cause_context_overflow():
    row = {"http_status": 400, "error": '{"error":{"type":"exceed_context_size_error"}}'}
    assert x2.classify_error_cause(row) == "context_overflow"


def test_classify_error_cause_connection_from_error_text():
    row = {"http_status": None, "error": "<urlopen error ... actively refused it>"}
    assert x2.classify_error_cause(row) == "connection"


def test_classify_error_cause_connection_from_infra_not_ready():
    row = {"http_status": None, "chat_outcome": "infra_not_ready", "error": "..."}
    assert x2.classify_error_cause(row) == "connection"


def test_classify_error_cause_timeout():
    row = {"http_status": None, "error": "call timed out after 900s"}
    assert x2.classify_error_cause(row) == "timeout"


def test_classify_error_cause_other_for_anything_unrecognized():
    row = {"http_status": 500, "error": "ROCm error: something unexpected"}
    assert x2.classify_error_cause(row) == "other"


def test_classify_error_cause_ignores_invalid_tags_and_reports_the_raw_cause():
    """Validity (invalid_* tags) is a separate axis from cause: a tagged race row still reports connection."""
    row = {"http_status": None, "error": "<urlopen error [WinError 10061] ...>", "invalid_race": True}
    assert x2.classify_error_cause(row) == "connection"
    assert x2.row_is_valid(row) is False


def test_row_is_valid_rejects_thinking_tag_and_connection_rows_but_keeps_real_errors():
    assert x2.row_is_valid({"http_status": 200, "invalid_thinking": True}) is False
    assert x2.row_is_valid({"http_status": None, "chat_outcome": "infra_not_ready"}) is False
    assert x2.row_is_valid({"http_status": 200, "score": 1.0}) is True
    assert x2.row_is_valid({"http_status": None, "error": "timed out"}) is True  # a real measured outcome
    assert x2.row_is_valid({"http_status": 400, "error": "exceeds the available context size"}) is True


def test_error_subcause_oom_and_server_not_ready():
    assert x2.error_subcause({"http_status": 500, "error": '{"error":"cudaMalloc failed: out of memory"}'}) == "oom"
    t2s_alloc = {"http_status": 500, "error": '{"error":"llama-server process has terminated: exit status 1: '
                                              'alloc_tensor_range: failed to allocate Vulkan0 buffer"}'}
    assert x2.error_subcause(t2s_alloc) == "oom"
    assert x2.classify_error_cause({"http_status": None, "error": "[WinError 10054] An existing connection was "
                                    "forcibly closed by the remote host"}) == "connection"
    row = {"http_status": None, "error": "server did not open its port in time"}
    assert x2.error_subcause(row) == "llama_server_not_ready"
    assert x2.classify_error_cause(row) == "connection"

# --------------------------------------------------------------------------------------------------- canary_items
def test_canary_items_picks_5_shortest_gsm8k_and_5_shortest_function_calling():
    items = (
        [{"item_id": f"gsm8k_{i}", "family": "gsm8k", "prompt_tokens": 100 - i} for i in range(8)]
        + [{"item_id": f"fcall_{i}", "family": "function_calling", "prompt_tokens": 200 + i} for i in range(8)]
        + [{"item_id": "longdoc_0", "family": "longdoc_qa", "prompt_tokens": 50000}]
    )
    c = x2.canary_items(items)
    assert len(c) == 10
    gsm = [it for it in c if it["family"] == "gsm8k"]
    fc = [it for it in c if it["family"] == "function_calling"]
    assert len(gsm) == 5 and len(fc) == 5
    assert gsm == sorted(gsm, key=lambda it: it["prompt_tokens"])
    all_gsm_sorted = sorted((it for it in items if it["family"] == "gsm8k"), key=lambda it: it["prompt_tokens"])
    assert [it["item_id"] for it in gsm] == [it["item_id"] for it in all_gsm_sorted[:5]]


# --------------------------------------------------------------------------------------------- canary_gate_check
def test_canary_gate_passes_clean_rows():
    rows = [{"http_status": 200, "score": 1.0} for _ in range(10)]
    passed, err, mean = x2.canary_gate_check(rows)
    assert passed is True
    assert err == 0.0
    assert mean == 1.0


def test_canary_gate_fails_on_high_error_rate():
    rows = [{"http_status": None, "error": "connection refused", "score": 0.0} for _ in range(3)]
    rows += [{"http_status": 200, "score": 1.0} for _ in range(7)]
    passed, err, mean = x2.canary_gate_check(rows)
    assert passed is False
    assert err == 0.3


def test_canary_gate_fails_on_low_mean_score_even_with_no_errors():
    rows = [{"http_status": 200, "score": 0.1} for _ in range(10)]
    passed, err, mean = x2.canary_gate_check(rows)
    assert passed is False
    assert err == 0.0
    assert mean == 0.1


def test_canary_gate_context_overflow_does_not_count_as_a_gate_error():
    rows = [{"http_status": 400, "error": "exceed_context_size_error", "score": 0.0} for _ in range(3)]
    rows += [{"http_status": 200, "score": 1.0} for _ in range(7)]
    passed, err, mean = x2.canary_gate_check(rows)
    assert err == 0.0  # overflow excluded from the error count
    # mean score still reflects the 3 zero-scored overflow rows
    assert abs(mean - 0.7) < 1e-9


def test_canary_gate_empty_canaries_passes_trivially():
    assert x2.canary_gate_check([]) == (True, 0.0, 1.0)


# --------------------------------------------------------------------------------------------------- tag_invalid_rows
def test_tag_invalid_rows_tags_race_and_thinking_without_touching_other_fields(tmp_path):
    path = tmp_path / "out.jsonl"
    rows = [
        {"record": "outcome_row", "config": "ollama_default", "model_id": "qwen3-8b",
         "error": "<urlopen error ... actively refused it>", "score": 0.0, "latency_s": 2.0},
        {"record": "outcome_row", "config": "llama_server", "model_id": "qwen3-8b",
         "score": 0.06, "http_status": 200, "latency_s": 5.0},  # pre-fix, no thinking_disabled marker
        {"record": "outcome_row", "config": "llama_server", "model_id": "qwen3-8b",
         "score": 1.0, "http_status": 200, "thinking_disabled": True, "latency_s": 5.0},  # post-fix, clean
        {"record": "outcome_row", "config": "llama_server", "model_id": "llama3.1:8b",
         "score": 0.9, "http_status": 200, "latency_s": 5.0},  # not a thinking model, must not be tagged
        {"record": "heartbeat", "item_id": "x"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    n_race, n_thinking, n_oom = x2.tag_invalid_rows(path)
    assert n_race == 1
    assert n_thinking == 1
    assert n_oom == 0

    updated = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert updated[0]["invalid_race"] is True
    assert updated[1]["invalid_thinking"] is True
    assert "invalid_thinking" not in updated[2]
    assert "invalid_race" not in updated[2]
    assert "invalid_thinking" not in updated[3]  # llama3.1:8b never tagged
    assert updated[1]["score"] == 0.06  # raw fields untouched


def test_tag_invalid_rows_is_idempotent(tmp_path):
    path = tmp_path / "out.jsonl"
    row = {"record": "outcome_row", "config": "ollama_default", "model_id": "qwen3-8b",
          "error": "actively refused", "score": 0.0}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    x2.tag_invalid_rows(path)
    once = path.read_text(encoding="utf-8")
    x2.tag_invalid_rows(path)
    twice = path.read_text(encoding="utf-8")
    assert once == twice


# --------------------------------------------------------------------------------------------------- reasoning budget
def test_llama_server_cmd_always_includes_reasoning_budget_zero():
    assert x2.LLAMA_SERVER_REASONING_BUDGET_ARGS == ["--reasoning-budget", "0"]


def test_run_one_llama_server_cmd_includes_reasoning_budget_flag(monkeypatch, tmp_path):
    captured = {}

    class FakeProc:
        returncode = 0
        def terminate(self): pass
        def wait(self, timeout=None): pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(x2.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(x2, "REPO", tmp_path)
    import socket as _socket
    monkeypatch.setattr(_socket, "create_connection", lambda *a, **kw: (_ for _ in ()).throw(OSError()))
    times = iter([0, 0, 200])  # jump straight past the 180s "server never opened its port" deadline
    monkeypatch.setattr(x2.time, "monotonic", lambda: next(times, 200))
    monkeypatch.setattr(x2.time, "sleep", lambda s: None)
    # the production log path is a hardcoded C:\apu\ovn\results location (deployment-specific, not
    # REPO-relative) -- create it so this test does not depend on that directory already existing
    Path(r"C:\apu\ovn\results").mkdir(parents=True, exist_ok=True)

    item = {"item_id": "x", "family": "f", "prompt": "hi", "prompt_tokens": 10}
    row = x2.run_one_llama_server(item, "qwen3-8b", r"C:\apu\models\x.gguf", tmp_path / "out.jsonl", call_timeout_s=5)
    assert "--reasoning-budget" in captured["cmd"]
    assert row["http_status"] is None  # never got ready in this stub -- just checking the cmd shape
