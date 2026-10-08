"""2026-10-02: "qwen3:30b-a3b-instruct-2507" does not exist on the Ollama registry (confirmed live against
the registry's own manifest endpoint: 404) -- every ollama_default call for this model 404'd for the whole
first night of the X2 weekend outcome-table run, silently producing zero real data for that model's ollama
leg across all 340 items. The real tag (also confirmed live: 200) is plain "qwen3:30b-a3b". This test locks
that mapping down so a future edit cannot silently reintroduce the bad tag."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import x2_outcome_table as x2  # noqa: E402


def test_qwen3_30b_a3b_ollama_leg_uses_a_model_created_from_the_llama_server_gguf():
    """2026-10-07: the registry tag qwen3:30b-a3b always thinks (template opens <think>) and is not the
    Instruct-2507 model the llama_server leg runs; the Ollama leg is created from the same GGUF instead."""
    ollama_tag, gguf = x2.MODEL_MAP["qwen3-30b-a3b"]
    assert ollama_tag == "qwen3-30b-a3b-2507"
    assert x2.OLLAMA_CREATE_FROM_GGUF[ollama_tag] == gguf
    assert "instruct-2507" not in ollama_tag  # the confirmed-404 registry tag


def test_rows_measured_with_the_old_30b_registry_tag_are_not_valid_cache():
    old = {"config": "ollama_default", "model_id": "qwen3-30b-a3b", "http_status": 200, "score": 1.0}
    assert x2.row_is_valid(old) is False
    new = dict(old, ollama_tag="qwen3-30b-a3b-2507")
    assert x2.row_is_valid(new) is True
    other = {"config": "ollama_default", "model_id": "qwen3-8b", "http_status": 200, "score": 1.0}
    assert x2.row_is_valid(other) is True  # legacy rows of unchanged mappings stay valid


def test_ensure_ollama_custom_models_creates_only_missing_tags(monkeypatch, tmp_path):
    import host_config as hc
    import json as _json
    monkeypatch.setattr(hc, "start_ollama_server", lambda: None)
    monkeypatch.setattr(hc, "stop_ollama_server", lambda: None)
    monkeypatch.setattr(hc, "wait_for_ollama_ready", lambda timeout_s=90: True)
    calls = []

    class R:
        returncode, stdout, stderr = 0, "success", ""

    out = tmp_path / "out.jsonl"
    x2.ensure_ollama_custom_models(["qwen3-30b-a3b", "qwen3-8b"], out, run_fn=lambda a: calls.append(a) or R(),
                                   models_dir=tmp_path)
    assert len(calls) == 1 and calls[0][1:3] == ["create", "qwen3-30b-a3b-2507"]
    rec = _json.loads(out.read_text().splitlines()[0])
    assert rec["record"] == "ollama_create" and rec["outcome"] == "ok"
    m = x2.ollama_manifest_path("qwen3-30b-a3b-2507", tmp_path)
    m.parent.mkdir(parents=True)
    m.write_text("{}")
    x2.ensure_ollama_custom_models(["qwen3-30b-a3b"], out, run_fn=lambda a: calls.append(a) or R(), models_dir=tmp_path)
    assert len(calls) == 1  # already present, not recreated


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


# -------------------------------------------------------------------------------------------- chat template source
def test_ollama_row_records_chat_template_source(monkeypatch, tmp_path):
    import chat_template_source as cts
    import host_config as hc
    monkeypatch.setattr(hc, "start_ollama_server", lambda: None)
    monkeypatch.setattr(hc, "stop_ollama_server", lambda: None)
    monkeypatch.setattr(hc, "wait_for_ollama_ready", lambda timeout_s=60: False)
    monkeypatch.setattr(cts, "ollama_tag_template", lambda tag: {"chat_template_source": "bare_gguf_ollama_create",
                                                                 "chat_template_sha256": "abc"})
    item = {"item_id": "x", "family": "f", "prompt": "hi", "prompt_tokens": 10}
    row = x2.run_one_ollama(item, "qwen3-4b-2507", "qwen3-4b-2507", tmp_path / "out.jsonl", call_timeout_s=30)
    assert row["chat_template_source"] == "bare_gguf_ollama_create" and row["chat_template_sha256"] == "abc"


def test_chat_template_fields_never_raise(monkeypatch):
    import chat_template_source as cts

    def boom(*a):
        raise RuntimeError("no store")
    monkeypatch.setattr(cts, "ollama_tag_template", boom)
    f = x2.chat_template_fields("ollama_default", "qwen3:8b")
    assert f["chat_template_source"] is None and "no store" in f["chat_template_error"]
    g = x2.chat_template_fields("llama_server", r"C:\does\not\exist.gguf")
    assert g == {"chat_template_source": "gguf_embedded_llama_server", "chat_template_sha256": None}


def test_rows_without_chat_template_fields_stay_valid_cache():
    """Backward compatibility with the running v3 resume: the new fields are informational only."""
    old = {"record": "outcome_row", "config": "ollama_default", "model_id": "qwen3-4b-2507", "ollama_tag": "qwen3-4b-2507",
           "http_status": 200, "score": 1.0, "family": "function_calling", "scorer_version": x2.SCORER_VERSION}
    assert x2.row_is_valid(old) and x2.SCORER_VERSION == 2


def test_t2s_rows_record_chat_template_source(monkeypatch):
    import chat_template_source as cts
    import t2s_outcome_table as t2s
    monkeypatch.setattr(cts, "ollama_tag_template", lambda tag: {"chat_template_source": "ollama_library",
                                                                 "chat_template_sha256": "x"})
    assert t2s.chat_template_fields(ollama_tag="llama3.1:8b")["chat_template_source"] == "ollama_library"
    assert t2s.chat_template_fields(gguf=t2s.GGUF_PATH)["chat_template_source"] == "gguf_embedded_llama_server"
