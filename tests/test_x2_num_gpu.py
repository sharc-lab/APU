"""harness/x2_num_gpu.py and x2_r2_agent --num-gpu N (x2_r2_llama_val_32k_offload, pre-registered in docs/FINDINGS.md
2026-10-14): num_gpu on every request body, the offload lines per session from Ollama-shaped log lines, the split check,
main() with fakes for host_config, the queue and the runtime, and the register row's pre-registered verdict. No
network, no process launches, no queue."""
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "tests"))

import x2_num_gpu as ng  # noqa: E402
import x2_r2_agent as ag  # noqa: E402
from test_x2_r2_agent import FakeRuntime, good_policy, _answer, _native  # noqa: E402

L_OFF = 'time=2026-10-14T22:00:00.000-07:00 level=INFO source=ggml.go:482 msg="offloaded {}/{} layers to GPU"'
L_LIB = ('time=2026-10-14T22:00:00.000-07:00 level=INFO source=types.go:32 msg="inference compute" id=0 '
         'library=ROCm compute=gfx1151 name=ROCm0 total="96.0 GiB"')
NEG = "ollama_ctx_32768_negative_control"
NEG_ARM = NEG + "_call2_notools"
SEEDS5 = (20260901, 20260902, 20260903, 20260904, 20260905)
JOB_ARGV = ["--mode", "validation", "--call2-tools", "off", "--models", "llama3.1:8b", "--neg-ctx", "32768",
            "--neg-sessions", "5", "--num-gpu", "22", "--out", "C:\\apu\\ovn\\results\\x2_r2_llama_val_32k_offload.jsonl"]


# ── pure parts ──────────────────────────────────────────────────────────────────────────────────────

def test_parse_offload_and_library_lines():
    text = "\n".join([L_LIB, L_OFF.format(22, 33), "noise", L_OFF.format(33, 33)])
    assert ng.parse_offloaded(text) == [[22, 33], [33, 33]]
    assert ng.parse_libraries(text) == ["ROCm"]
    assert ng.parse_offloaded("") == [] and ng.parse_libraries(None) == []


def test_tray_app_running_reads_the_process_count():
    assert ng.tray_app_running(run=lambda *a, **k: types.SimpleNamespace(stdout="1\n")) is True
    assert ng.tray_app_running(run=lambda *a, **k: types.SimpleNamespace(stdout="0\n")) is False
    assert ng.tray_app_running(run=lambda *a, **k: (_ for _ in ()).throw(OSError("x"))) is False


def test_server_log_sits_next_to_out():
    assert ng.server_log_for(r"C:\apu\ovn\results\x2_r2_llama_val_32k_offload.jsonl").endswith(
        "x2_r2_llama_val_32k_offload_serve.log")


def test_read_from_follows_a_truncated_log(tmp_path):
    p = tmp_path / "s.log"
    p.write_text("a" * 100, encoding="utf-8")
    text, off = ng.read_from(p, 0)
    assert len(text) == 100 and off == 100
    p.write_bytes(b"restart\n")   # cmd.exe '>' truncates at a server restart
    text, off = ng.read_from(p, off)
    assert text == "restart\n" and off == 8
    assert ng.read_from(tmp_path / "missing.log", 5) == ("", 5)


@pytest.mark.parametrize("offloaded,ok", [([[22, 33]], True), ([[20, 33], [24, 33]], True), ([[25, 33]], False),
                                          ([[19, 33]], False), ([[22, 37]], False), ([], False),
                                          ([[22, 33], [33, 33]], False)])
def test_split_as_intended(offloaded, ok):
    assert ng.split_as_intended(offloaded, 22, 33, 2) is ok


def test_num_gpu_goes_into_every_request_body(monkeypatch):
    inner = ag.OllamaRuntime()
    bodies = []
    monkeypatch.setattr(inner, "_post", lambda path, body: (bodies.append(body) or (200, {"message": {"content": "x"}},
                                                                                   None, 0.1)))
    rt = ng.NumGpuRuntime(inner, 22)
    assert inner.options_extra == {"num_gpu": 22}
    rt.chat("llama3.1:8b", [{"role": "user", "content": "hi"}], 32768, None, None)
    rt.chat("llama3.1:8b", [{"role": "user", "content": "hi"}], 32768, None, None, extra={"tool_choice": "none"})
    assert [b["options"]["num_gpu"] for b in bodies] == [22, 22]
    assert bodies[0]["options"]["num_ctx"] == 32768 and bodies[0]["options"]["temperature"] == 0
    assert bodies[0]["options"]["seed"] == 42 and bodies[0]["options"]["num_predict"] == ag.MAX_TOKENS_PER_CALL
    assert rt.n_chat == 2 and rt.n_chat_num_gpu == 2
    with pytest.raises(ValueError):
        ng.NumGpuRuntime(types.SimpleNamespace(), -1)


def test_without_the_flag_requests_are_unchanged():
    assert "num_gpu" not in ag.native_chat_body("m", [], 32768, None, None)["options"]
    assert ag.OllamaRuntime.options_extra is None


class _PS:
    def __init__(self, size=9518773696, vram=6248820899):
        self.size, self.vram = size, vram

    def get_ps(self):
        return {"models": [{"name": "llama3.1:8b", "size": self.size, "size_vram": self.vram,
                            "context_length": 32768}]}


def test_recorder_emits_one_row_per_session(tmp_path):
    log = tmp_path / "s.log"
    log.write_text(L_LIB + "\n" + L_OFF.format(33, 33) + "\n", encoding="utf-8")   # before the run: not counted
    inner = FakeRuntime(lambda c, m, t: ("", None))
    inner.client = _PS()
    rt = ng.NumGpuRuntime(inner, 22)
    rows = []
    rec = ng.OffloadRecorder(rt, log, rows.append, lambda m: None)
    rec.start("llama3.1:8b", NEG_ARM, 20260901)
    with open(log, "a", encoding="utf-8") as f:
        f.write(L_OFF.format(22, 33) + "\n")
    rt.chat("llama3.1:8b", [{"role": "user", "content": "x"}], 32768, None, None)
    rt.loaded_context("llama3.1:8b")
    r = rec.end()
    assert rows == [r] and r["record"] == "num_gpu_offload" and r["offloaded"] == [[22, 33]]
    assert r["seed"] == 20260901 and r["arm_id"] == NEG_ARM and r["num_gpu_sent"] == 22
    assert r["n_chat_calls"] == 1 and r["n_chat_calls_num_gpu"] == 1
    assert abs(r["ps_vram_share"] - 6248820899 / 9518773696) < 1e-12


# ── CLI and main() ──────────────────────────────────────────────────────────────────────────────────

def test_job_argv_parses_to_the_preregistered_plan():
    a = ag.build_arg_parser().parse_args(JOB_ARGV)
    assert a.num_gpu == 22 and a.neg_ctx == 32768 and a.neg_sessions == 5 and a.call2_tools == "off"
    plan, neg = ag.validation_plan_neg(a.call2_tools, a.neg_ctx, a.neg_sessions)
    assert neg == NEG and plan[0] == (NEG_ARM, SEEDS5, 10)
    assert plan[1] == ("ollama_ctx_8192_positive_control_call2_notools", (20260901,), 15)
    assert plan[2] == (NEG, tuple(ag.SEEDS), 10)


def _patch_main(monkeypatch, tmp_path, tray=False, offloaded=22):
    import socket
    calls = {"start": [], "stop": 0, "notes": []}

    def start(env=None, log_path=None):
        calls["start"].append({"env": env, "log_path": log_path})
        Path(log_path).write_text(L_LIB + "\n", encoding="utf-8")

    def stop():
        calls["stop"] += 1

    monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
        start_ollama_server=start, stop_ollama_server=stop, wait_for_ollama_ready=lambda timeout_s=0: True,
        require_host=lambda h: {"hw_id": "evo-x2", "interactive_guard": False},
        enforce_or_record_interactive_session=lambda cfg: {}))
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=calls["notes"].append))
    monkeypatch.setattr(ng, "tray_app_running", lambda run=None: tray)

    class RT(FakeRuntime):
        def __init__(self):
            super().__init__(lambda c, m, t: ("", [_native("log_event", event="e")]) if c == 1
                             else (_answer("ok", "SC"), None), loaded=32768)
            self.server_env = self.server_log = None
            self.options_extra = None
            self.client = _PS()
            self.loaded_model = False

        def chat(self, model, messages, num_ctx, tools, think, extra=None):
            if not self.loaded_model:   # a load logs the offload line, as Ollama does
                with open(self.server_log, "a", encoding="utf-8") as f:
                    f.write(L_OFF.format(offloaded, 33) + "\n")
                self.loaded_model = True
            calls.setdefault("options", []).append(dict(self.options_extra or {}))
            return super().chat(model, messages, num_ctx, tools, think, extra)

        def unload(self, model):
            self.loaded_model = False
            return super().unload(model)

        def available_models(self):
            return ["llama3.1:8b"]

        def version(self):
            return "0.34.4"

    monkeypatch.setattr(ag, "OllamaRuntime", RT)
    ag.neg_arm_for_ctx(32768)
    monkeypatch.setattr(ag, "validation_plan_neg",
                        lambda c2, a, b: ([(NEG_ARM, (20260901, 20260902), 2)], NEG))
    return calls


def test_main_num_gpu_runs_its_own_server_and_records_the_offload(tmp_path, monkeypatch):
    calls = _patch_main(monkeypatch, tmp_path)
    out = tmp_path / "off.jsonl"
    note = ag.main(["--mode", "validation", "--call2-tools", "off", "--models", "llama3.1:8b", "--neg-ctx", "32768",
                    "--neg-sessions", "5", "--num-gpu", "22", "--out", str(out)])
    assert note == "completed" and calls["notes"] == ["completed"]
    assert calls["stop"] >= 1 and calls["start"][0]["log_path"] == ng.server_log_for(out)
    assert calls["start"][0]["env"] is None
    assert calls["options"] and all(o == {"num_gpu": 22} for o in calls["options"])
    rows = ag.read_rows(out)
    assert rows[0]["record"] == "run_start" and rows[0]["num_gpu"] == 22 and rows[-1]["record"] == "run_end"
    srv = next(r for r in rows if r["record"] == "num_gpu_server")
    assert srv["libraries"] == ["ROCm"]
    off = [r for r in rows if r["record"] == "num_gpu_offload"]
    assert [r["seed"] for r in off] == [20260901, 20260902]
    assert all(r["offloaded"] == [[22, 33]] and r["n_chat_calls"] == r["n_chat_calls_num_gpu"] == 4 for r in off)


def test_main_num_gpu_refuses_while_the_tray_app_runs(tmp_path, monkeypatch):
    calls = _patch_main(monkeypatch, tmp_path, tray=True)
    note = ag.main(["--mode", "validation", "--call2-tools", "off", "--models", "llama3.1:8b", "--num-gpu", "22",
                    "--out", str(tmp_path / "o.jsonl")])
    assert note.startswith("stopped:") and "tray app" in note
    assert calls["start"] == [] and calls["stop"] == 0 and calls["notes"] == [note]


def test_num_gpu_argument_errors(tmp_path, monkeypatch):
    _patch_main(monkeypatch, tmp_path)
    for argv in (["--mode", "mechanism", "--num-gpu", "22"], ["--mode", "validation", "--num-gpu", "-1"],
                 ["--mode", "real", "--num-gpu", "22", "--num-keep", "system"],
                 ["--mode", "toolchoice_check", "--num-gpu", "22"]):
        with pytest.raises(SystemExit):
            ag.main(argv + ["--out", str(tmp_path / "x.jsonl")])


# ── register row: the pre-registered verdict ────────────────────────────────────────────────────────

def _text_tool_call_at_5(ags):
    base = good_policy(ags)

    def policy(call_no, messages, turn_idx):
        if call_no == 2 and turn_idx == 5:
            return '{"name": "lookup_fact", "parameters": {"key": "REC-0005"}}', None
        return base(call_no, messages, turn_idx)
    return policy


def _offload_rows(policy_factory=good_policy, offloaded=((22, 33),), drop_seed=None, n_gpu_calls=None):
    ag.neg_arm_for_ctx(32768)
    rows = [{"record": "run_start", "mode": "validation", "num_gpu": 22},
            {"record": "num_gpu_server", "libraries": ["ROCm"]}]
    for s in SEEDS5:
        ags = ag.build_agent_session(s, 10)
        rt = FakeRuntime(policy_factory(ags), loaded=32768)
        ag.run_session(rt, "llama3.1:8b", NEG_ARM, s, 10, rows.append, lambda m: None, "validation")
        if s != drop_seed:
            rows.append({"record": "num_gpu_offload", "model_id": "llama3.1:8b", "arm_id": NEG_ARM, "seed": s,
                         "num_gpu_sent": 22, "offloaded": [list(x) for x in offloaded], "n_chat_calls": 20,
                         "n_chat_calls_num_gpu": 20 if n_gpu_calls is None else n_gpu_calls,
                         "ps_size": 100, "ps_size_vram": 66, "ps_vram_share": 0.66})
    rows.append({"record": "run_end"})
    return rows


@pytest.fixture(scope="module")
def nr():
    import numbers_register
    return numbers_register


def test_verdict_backend_when_no_miss_and_split_holds(nr):
    v = nr.x2_r2_llama_val_32k_offload_verdict(_offload_rows(), ag)
    assert v["valid"] and v["verdict"] == "backend effect (CUDA vs evo-x2's backend)"
    assert v["n_missed_canaries"] == 0 and not v["split_problems"] and not v["split_unverified"]


def test_verdict_offload_when_every_miss_is_a_turn5_text_tool_call(nr):
    v = nr.x2_r2_llama_val_32k_offload_verdict(_offload_rows(_text_tool_call_at_5), ag)
    assert v["verdict"] == "offload effect" and v["misses"] and all(m["turn"] == 5 for m in v["misses"])


def test_verdict_flagged_when_the_split_differs_or_is_not_seen(nr):
    v = nr.x2_r2_llama_val_32k_offload_verdict(_offload_rows(offloaded=((33, 33),)), ag)
    assert v["verdict"].startswith("inconclusive (offload split not as intended")
    v = nr.x2_r2_llama_val_32k_offload_verdict(_offload_rows(offloaded=()), ag)
    assert v["verdict"].startswith("inconclusive (offload split not verified")
    v = nr.x2_r2_llama_val_32k_offload_verdict(_offload_rows(drop_seed=20260903), ag)
    assert "20260903" in v["verdict"] and v["verdict"].startswith("inconclusive")
    v = nr.x2_r2_llama_val_32k_offload_verdict(_offload_rows(n_gpu_calls=19), ag)
    assert v["verdict"].startswith("inconclusive (offload split not as intended")


def test_register_row_pending_then_computes(nr, tmp_path):
    entry = next(e for e in nr.NUMBER_ENTRIES if e["claim_id"] == "x2-r2-llama-val-32k-offload")
    assert entry["pending_ok"] and entry["data_files"] == [nr.X2_R2_LLAMA_VAL_32K_OFFLOAD]
    f = tmp_path / nr.X2_R2_LLAMA_VAL_32K_OFFLOAD
    f.parent.mkdir(parents=True)
    f.write_text("\n".join(json.dumps(r, default=str) for r in _offload_rows()) + "\n", encoding="utf-8")
    r = nr.compute_x2_r2_llama_val_32k_offload(tmp_path)
    assert r["value"].endswith("verdict: backend effect (CUDA vs evo-x2's backend)")
    assert "backend library ['ROCm']" in r["value"] and "20260901: [[22, 33]] VRAM share 0.660" in r["value"]
    with pytest.raises(FileNotFoundError):
        nr.compute_x2_r2_llama_val_32k_offload(tmp_path / "nothing")
