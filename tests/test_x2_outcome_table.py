"""2026-10-02: "qwen3:30b-a3b-instruct-2507" does not exist on the Ollama registry (confirmed live against
the registry's own manifest endpoint: 404) -- every ollama_default call for this model 404'd for the whole
first night of the X2 weekend outcome-table run, silently producing zero real data for that model's ollama
leg across all 340 items. The real tag (also confirmed live: 200) is plain "qwen3:30b-a3b". This test locks
that mapping down so a future edit cannot silently reintroduce the bad tag."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import x2_outcome_table as x2  # noqa: E402


def test_qwen3_30b_a3b_ollama_tag_is_the_real_registry_tag():
    ollama_tag, _gguf = x2.MODEL_MAP["qwen3-30b-a3b"]
    assert ollama_tag == "qwen3:30b-a3b"
    assert "instruct-2507" not in ollama_tag  # the confirmed-404 tag


def test_every_model_map_entry_has_an_ollama_tag_and_a_gguf_path():
    for model_key, (ollama_tag, gguf_path) in x2.MODEL_MAP.items():
        assert ollama_tag, f"{model_key} has an empty ollama tag"
        assert gguf_path.lower().endswith(".gguf"), f"{model_key}'s path is not a .gguf: {gguf_path}"


# -------------------------------------------------------------------------------------------- run_one_ollama
def test_run_one_ollama_records_infra_not_ready_when_server_never_comes_up(monkeypatch, tmp_path):
    """2026-10-06 bug found live: wait_for_ollama_ready's own return value used to be discarded, so a server
    that never came up (409 of ~650 real calls over the weekend hit this) still got a real chat() attempt,
    which just failed with a generic connection-refused error indistinguishable from a model problem."""
    import host_config as hc
    monkeypatch.setattr(hc, "start_ollama_server", lambda: None)
    monkeypatch.setattr(hc, "stop_ollama_server", lambda: None)
    monkeypatch.setattr(hc, "wait_for_ollama_ready", lambda timeout_s=60: False)
    item = {"item_id": "x", "family": "f", "prompt": "hi", "prompt_tokens": 10}
    row = x2.run_one_ollama(item, "qwen3-8b", "qwen3:8b", tmp_path / "out.jsonl", call_timeout_s=30)
    assert row["chat_outcome"] == "infra_not_ready"
    assert row["score"] == 0.0


def test_run_one_ollama_retries_once_then_succeeds(monkeypatch, tmp_path):
    import host_config as hc
    calls = {"wait_n": 0, "start_n": 0}
    monkeypatch.setattr(hc, "start_ollama_server", lambda: calls.__setitem__("start_n", calls["start_n"] + 1))
    monkeypatch.setattr(hc, "stop_ollama_server", lambda: None)

    def fake_wait(timeout_s=60):
        calls["wait_n"] += 1
        return calls["wait_n"] >= 2  # fails the first time, succeeds on the retry

    monkeypatch.setattr(hc, "wait_for_ollama_ready", fake_wait)

    class FakeClient:
        def chat(self, model, prompt, num_ctx=None, messages=None, max_tokens=256, think=False, keep_alive=None, timeout=None):
            return {"status": 200, "outcome": "ok", "message": "hi", "prompt_eval_count": 5, "error": None}

    monkeypatch.setattr(x2.k1, "OllamaClient", FakeClient)
    monkeypatch.setattr(x2, "load_graders", lambda: object())
    monkeypatch.setattr(x2, "score_response", lambda grade_module, item, output_text: 1.0)

    item = {"item_id": "x", "family": "f", "prompt": "hi", "prompt_tokens": 5}
    row = x2.run_one_ollama(item, "qwen3-8b", "qwen3:8b", tmp_path / "out.jsonl", call_timeout_s=30)
    assert calls["wait_n"] == 2  # one retry happened
    assert row["chat_outcome"] == "ok"
    assert row["score"] == 1.0
