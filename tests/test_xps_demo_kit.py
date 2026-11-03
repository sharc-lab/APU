"""XPS demo kit, local step (harness/xps_demo_kit.py): fakes only, no network, no runtime."""
import argparse
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "harness"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import xps_demo_kit as K  # noqa: E402


def _ns(tmp_path, **kw):
    base = dict(arms="a,b,c", seeds=[K.SEEDS[0]], max_turns=5, runtime="ollama", base_url="fake", model="qwen3:4b",
                allow_nondefault=None, pipeline_window=None, out=str(tmp_path / "local.jsonl"), note=None,
                runtime_exe=None, runtime_config_file=None, declared_default_window=None)
    base.update(kw)
    return argparse.Namespace(**base)


CLEAN = {"runtime": "ollama", "overrides": [], "unverified": [], "default_window": None}


def _rows(path):
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


# ── overrides ──────────────────────────────────────────────────────────────────────────────────────────

def test_env_overrides_flags_runtime_settings_and_keeps_address_vars_benign():
    over, benign = K.env_overrides({"process": {"OLLAMA_CONTEXT_LENGTH": "8192", "OLLAMA_HOST": "0.0.0.0",
                                                "PATH": "x", "GGML_SYCL_DEVICE": "0"},
                                    "user": {"OV_CACHE_DIR": "c:/x", "LLAMA_ARG_CTX_SIZE": "4096"}})
    keys = {(o["source"], o["key"]) for o in over}
    assert keys == {("env:process", "OLLAMA_CONTEXT_LENGTH"), ("env:process", "GGML_SYCL_DEVICE"),
                    ("env:user", "OV_CACHE_DIR"), ("env:user", "LLAMA_ARG_CTX_SIZE")}
    assert benign == {"process:OLLAMA_HOST": "0.0.0.0"}


def test_llamacpp_flags_model_and_address_only_is_default():
    assert K.llamacpp_flag_overrides('"C:\\l\\llama-server.exe" -m "C:\\m\\q.gguf" --port 8080 --jinja') == []
    got = K.llamacpp_flag_overrides("llama-server.exe -m q.gguf -c 8192 --cache-type-k q8_0 -fa on --jinja")
    assert [(o["key"], o["value"]) for o in got] == [("-c", "8192"), ("--cache-type-k", "q8_0"), ("-fa", "on")]
    assert K.llamacpp_flag_overrides(None) is None


def test_ovms_config_and_modelfile_overrides():
    txt = json.dumps({"plugin_config": {"KV_CACHE_PRECISION": "u8"}, "device": "GPU", "cache_size": 2})
    keys = {o["key"] for o in K.ovms_config_overrides(txt)}
    assert {"KV_CACHE_PRECISION", "plugin_config", "device", "cache_size"} <= keys
    assert K.ovms_config_overrides(json.dumps({"model_path": "x"})) == []
    mf = K.ollama_modelfile_overrides({"parameters": "num_ctx 8192\nstop \"<|im_end|>\""})
    assert [o["key"] for o in mf] == ["num_ctx", "stop"]
    assert K.ollama_modelfile_overrides({}) == []


def test_assert_default_refuses_arm_a_only_and_records_reason_path():
    cfg = {"overrides": [{"source": "env:process", "key": "OLLAMA_KV_CACHE_TYPE", "value": "q8_0"}], "unverified": []}
    with pytest.raises(SystemExit, match="OLLAMA_KV_CACHE_TYPE"):
        K.assert_default(cfg, "a", None)
    K.assert_default(cfg, "a", "operator: q8_0 is how this machine is set up")
    K.assert_default(cfg, "b", None)
    with pytest.raises(SystemExit, match="not verifiable"):
        K.assert_default({"overrides": [], "unverified": ["llama-server command line could not be read"]}, "a", "  ")


def _fake_get(table):
    def gj(url, timeout=30.0, body=None):
        for k, v in table.items():
            if url.endswith(k):
                return v
        return None
    return gj


def test_probe_ollama_records_version_modelfile_and_env():
    cfg = K.probe_runtime("ollama", "http://h:11434", "qwen3:4b", environ={"OLLAMA_FLASH_ATTENTION": "1"},
                          reg_env={"user": {}, "machine": {}},
                          get_json=_fake_get({"/api/version": {"version": "0.12.3"},
                                              "/api/show": {"parameters": "num_ctx 16384",
                                                            "model_info": {"qwen3.context_length": 40960}}}))
    assert cfg["version"] == "0.12.3" and cfg["model_info_context_length"] == 40960
    assert {o["key"] for o in cfg["overrides"]} == {"OLLAMA_FLASH_ATTENTION", "num_ctx"}
    assert cfg["overrides_present"] and cfg["default_window"] is None


def test_probe_llamacpp_reads_props_and_cmdline():
    gj = _fake_get({"/props": {"build_info": "b6500-abc", "default_generation_settings": {"n_ctx": 40960}}})
    cfg = K.probe_runtime("llamacpp", "http://127.0.0.1:8080", "q", environ={}, reg_env={}, get_json=gj,
                          cmdline_fn=lambda port: "llama-server.exe -m q.gguf --port 8080 --jinja")
    assert cfg["default_window"] == 40960 and cfg["version"] == "b6500-abc"
    assert cfg["overrides"] == [] and cfg["unverified"] == []
    cfg2 = K.probe_runtime("llamacpp", "http://127.0.0.1:8080", "q", environ={}, reg_env={}, get_json=gj,
                           cmdline_fn=lambda port: None)
    assert cfg2["unverified"]


def test_probe_openvino_needs_declared_window_and_config(tmp_path):
    conf = tmp_path / "graph.json"
    conf.write_text(json.dumps({"device": "GPU", "plugin_config": {"KV_CACHE_PRECISION": "u8"}}), encoding="utf-8")
    cfg = K.probe_runtime("openvino", "http://127.0.0.1:8000", "Qwen3-4B", config_file=str(conf),
                          declared_default_window=None, environ={}, reg_env={}, get_json=_fake_get({}),
                          version_fn=lambda argv: "OpenVINO Model Server 2025.3")
    assert any("not declared" in u for u in cfg["unverified"])
    assert {"device", "KV_CACHE_PRECISION"} <= {o["key"] for o in cfg["overrides"]}
    cfg2 = K.probe_runtime("openvino", "http://127.0.0.1:8000", "Qwen3-4B", runtime_exe="ovms.exe", config_file=None,
                           declared_default_window=32768, environ={}, reg_env={}, get_json=_fake_get({}),
                           version_fn=lambda argv: "OpenVINO Model Server 2025.3")
    assert cfg2["default_window"] == 32768 and cfg2["version"].startswith("OpenVINO Model Server")


# ── arm logic ─────────────────────────────────────────────────────────────────────────────────────────

def test_standard_route():
    assert K.standard_route(9000, 8192) == "cloud_over_window"
    assert K.standard_route(8192, 8192) == "local"
    assert K.standard_route(100, None) == "local"
    assert K.standard_route(100, 8192, {"outcome": "ok", "status": 200}) == "local"
    assert K.standard_route(100, 8192, {"outcome": "http_error", "status": 400, "error": "ctx"}) == "cloud_hard_error"
    assert K.standard_route(100, 8192, {"outcome": "error", "status": None, "error": "refused"}) == "cloud_hard_error"


def test_standard_hybrid_runtime_defers_before_call_and_on_hard_error():
    inner = K.FakeRuntime(error_after=0)
    rt = K.StandardHybridRuntime(inner, window=10)
    with pytest.raises(K.Defer) as e:
        rt.chat("m", [{"role": "user", "content": "x" * 400}], None, None, False)
    assert e.value.reason == "over_window" and inner.n == 0      # no local call was made
    rt = K.StandardHybridRuntime(inner, window=10 ** 6)
    with pytest.raises(K.Defer) as e:
        rt.chat("m", [{"role": "user", "content": "hi"}], None, None, False)
    assert e.value.reason == "hard_error" and e.value.detail["status"] == 400


def test_route_at_start_first_k_local_then_projection():
    ft = [(1, 3000), (2, 3200), (3, 3400)]
    assert not K.route_at_start(ft, 3, 4000)["route"]
    assert not K.route_at_start(ft, 5, 4000)["route"]          # 3400 + 2 x 200 = 3800 <= 4000
    r = K.route_at_start(ft, 10, 4000)
    assert r["route"] and r["projected_tokens"] == 4800
    assert not K.route_at_start(ft, 10, None)["route"]


def test_post_run_check_never_reads_the_grader():
    import x2_r2_evalset as EV
    es = EV.build_session(K.SEEDS[0], 2, K.WORKLOAD, K.GUARDRAILS, "qwen3:4b", None, K.DESIGN)
    row = {"turn_idx": 1, "seed": K.SEEDS[0], "calls": [], "final_text": '{"answer": "x", "source": "y"}',
           "over_loaded_window": False}
    a = K.post_run_check({**row, "end_state_ok": True, "answer_ok": True}, es, "ollama")
    b = K.post_run_check({**row, "end_state_ok": False, "answer_ok": False}, es, "ollama")
    assert a["decision"] == b["decision"] and a["fired"] == b["fired"]
    c = K.post_run_check({**row, "over_loaded_window": True}, es, "ollama")
    assert "L1_proxy" in c["fired"] and c["decision"] == "escalate"


def test_openai_compat_runtime_maps_tool_calls_and_l1_class():
    data = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "log_event", "arguments": "{\"event\": \"x\"}"}}]},
        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 100, "completion_tokens": 9}}
    sent = {}

    def post(url, body):
        sent["url"], sent["body"] = url, body
        return 200, data, None, 0.1
    rt = K.OpenAICompatRuntime("http://127.0.0.1:8080", "llamacpp", "qwen3-4b", n_ctx=4096, post=post)
    r = rt.chat("qwen3-4b", [{"role": "user", "content": "hi"}], None, [{"type": "function"}], False)
    assert sent["url"].endswith("/v1/chat/completions")
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert r["tool_calls"][0]["function"]["arguments"] == {"event": "x"} and r["prompt_eval_count"] == 100
    ev = rt.pop_turn_events()
    assert ev[0]["class"] == "none" and rt.pop_turn_events() == []
    ov = K.OpenAICompatRuntime("http://127.0.0.1:8000", "openvino", "Qwen3-4B", post=post)
    ov.chat("Qwen3-4B", [{"role": "user", "content": "hi"}], None, None, False)
    assert sent["url"].endswith("/v3/chat/completions") and sent["body"]["model"] == "Qwen3-4B"


# ── the driver end to end (fake runtime) ──────────────────────────────────────────────────────────────

def test_run_writes_all_record_kinds_and_defers(tmp_path):
    res = K.run(_ns(tmp_path), runtime=K.FakeRuntime(window=3500), cfg=dict(CLEAN), log=lambda m: None)
    rows = _rows(res["out"])
    st = rows[0]
    assert st["record"] == "run_start" and st["window_default"] == 3500
    assert st["runtime_config"]["default_window"] == 3500
    assert rows[1]["record"] == "warmup" and rows[1]["excluded_from_stats"]
    assert rows[-1]["record"] == "run_end" and rows[-1]["stopped"] is None
    a = [r for r in rows if r.get("arm") == "a" and r["record"] == "r2a_turn"]
    assert len(a) == 5 and all(r["num_ctx_requested"] is None for r in a)
    b_def = [r for r in rows if r.get("arm") == "b" and r["record"] == K.DEFER_RECORD]
    assert b_def and all(r["reason"] == "over_window" for r in b_def)
    c = [r for r in rows if r.get("arm") == "c"]
    assert all("prc_check" in r for r in c if r["record"] == "r2a_turn")
    assert [r["turn_idx"] for r in c if r["record"] == K.DEFER_RECORD and r["reason"] == "route_at_start"]
    sessions = [r for r in rows if r["record"] == "r2a_session"]
    assert len(sessions) == 3 and all(r["mode"] == "real" for r in sessions)


def test_run_refuses_override_for_arm_a_before_writing(tmp_path):
    cfg = dict(CLEAN, overrides=[{"source": "env:process", "key": "OLLAMA_CONTEXT_LENGTH", "value": "8192"}])
    with pytest.raises(SystemExit, match="refused"):
        K.run(_ns(tmp_path), runtime=K.FakeRuntime(), cfg=cfg, log=lambda m: None)
    assert not (tmp_path / "local.jsonl").exists()
    res = K.run(_ns(tmp_path, arms="b"), runtime=K.FakeRuntime(), cfg=cfg, log=lambda m: None)
    assert Path(res["out"]).exists()                          # arms b and c are not refused for an override


def test_run_never_overwrites_and_rejects_arm_d(tmp_path):
    (tmp_path / "local.jsonl").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="exists"):
        K.run(_ns(tmp_path), runtime=K.FakeRuntime(), cfg=dict(CLEAN), log=lambda m: None)
    with pytest.raises(SystemExit, match="unknown arm"):
        K.run(_ns(tmp_path, arms="d", out=str(tmp_path / "o2.jsonl")), runtime=K.FakeRuntime(), cfg=dict(CLEAN),
              log=lambda m: None)


def test_hard_error_in_arm_a_is_recorded_not_deferred(tmp_path):
    res = K.run(_ns(tmp_path, arms="a,b", max_turns=3), runtime=K.FakeRuntime(window=10 ** 6, error_after=2),
                cfg=dict(CLEAN), log=lambda m: None)
    rows = _rows(res["out"])
    a = [r for r in rows if r.get("arm") == "a" and r["record"] == "r2a_turn"]
    assert len(a) == 3 and a[-1]["any_http_error"] and a[-1]["end_state_ok"] is False
    b = [r for r in rows if r.get("arm") == "b" and r["record"] == K.DEFER_RECORD]
    assert b and all(r["reason"] == "hard_error" for r in b)


def test_selftest_passes(tmp_path, capsys):
    assert K.selftest(tmp_path) == 0
    assert "SELFTEST OK" in capsys.readouterr().out
