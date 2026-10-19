"""harness/x2_backend.py and x2_r2_agent --backend vulkan (x2_r2_llama_val_32k_vulkan, pre-registered in
docs/FINDINGS.md 2026-10-19): the run-level and per-session backend checks on Ollama-shaped log lines, main() with fakes
for host_config, the queue and the runtime (own server env and log, refusal when Vulkan is not loaded, refusal while
the tray app runs), the sidecar import check, and the register rows' pre-registered verdict. No network, no process
launches, no queue."""
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "tests"))

import x2_backend as xb  # noqa: E402
import x2_r2_agent as ag  # noqa: E402
from test_x2_r2_agent import FakeRuntime, good_policy, _answer, _native  # noqa: E402

T = "time=2026-10-19T22:00:00.000-07:00 level=INFO "
L_VK = (T + 'source=types.go:32 msg="inference compute" id=0 filter_id=0 library=Vulkan compute=0.0 name=Vulkan0 '
        'description="AMD Radeon(TM) 8060S Graphics" libdirs=ollama,vulkan driver=0.0 pci_id="" type=iGPU '
        'total="96.0 GiB" available="95.5 GiB"')
L_ROCM = (T + 'source=types.go:32 msg="inference compute" id=0 filter_id=0 library=ROCm compute=gfx1151 name=ROCm0 '
          'description="AMD Radeon(TM) 8060S Graphics" libdirs=ollama,rocm_v7_1 driver=0.0 pci_id=0000:c5:00.0 '
          'type=iGPU total="99.7 GiB" available="99.6 GiB"')
L_DROP = (T + 'source=runner.go:405 msg="dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1" id=0 '
          'library=Vulkan compute=0.0 name=Vulkan0 description="AMD Radeon(TM) 8060S Graphics" pci_id=""')
L_CFG = (T + 'source=routes.go:2005 msg="server config" env="map[CUDA_VISIBLE_DEVICES: GGML_VK_VISIBLE_DEVICES: '
         'HIP_VISIBLE_DEVICES:-1 OLLAMA_IGPU_ENABLE:true OLLAMA_LLM_LIBRARY:vulkan OLLAMA_VULKAN:true '
         'ROCR_VISIBLE_DEVICES:]"')
L_MEM = T + 'source=sched.go:625 msg="gpu memory" id=0 library={} available="95.0 GiB" free="95.5 GiB"'
L_OFF = "load_tensors: offloaded {}/{} layers to GPU"
L_BUF = "load_tensors: {} model buffer size =  4403.49 MiB"
L_DEV = "ggml_vulkan: 0 = AMD Radeon(TM) 8060S Graphics (AMD proprietary driver) | uma: 1 | fp16: 1 | bf16: 1"
NEG = "ollama_ctx_32768_negative_control"
NEG_ARM = NEG + "_call2_notools"
SEEDS5 = (20260901, 20260902, 20260903, 20260904, 20260905)
OUT = "C:\\apu\\ovn\\results\\x2_r2_llama_val_32k_vulkan.jsonl"
JOB_ARGV = ["--mode", "validation", "--call2-tools", "off", "--models", "llama3.1:8b", "--neg-ctx", "32768",
            "--neg-sessions", "5", "--backend", "vulkan", "--out", OUT]


def _vk_session_log(n=33):
    return "\n".join([L_MEM.format("Vulkan"), L_DEV, L_OFF.format(n, 33), L_BUF.format("Vulkan0"),
                      L_BUF.format("CPU_Mapped")]) + "\n"


# ── pure parts ──────────────────────────────────────────────────────────────────────────────────────

def test_backend_env_is_process_scoped_and_preregistered():
    assert xb.BACKEND_ENVS["vulkan"] == {"OLLAMA_IGPU_ENABLE": "1", "OLLAMA_LLM_LIBRARY": "vulkan",
                                         "HIP_VISIBLE_DEVICES": "-1"}
    import host_config as hc
    s = hc._extra_env_sets(xb.BACKEND_ENVS["vulkan"])   # passes start_ollama_server's own value check
    assert 'set "OLLAMA_LLM_LIBRARY=vulkan"&& ' in s and 'set "HIP_VISIBLE_DEVICES=-1"&& ' in s
    assert xb.server_log_for(OUT).endswith("x2_r2_llama_val_32k_vulkan_serve.log")


def test_parse_inference_compute():
    lines = xb.parse_inference_compute("\n".join([L_CFG, L_DROP, L_VK]))
    assert lines == [{"library": "Vulkan", "name": "Vulkan0", "description": "AMD Radeon(TM) 8060S Graphics",
                      "compute": "0.0", "type": "iGPU", "total": "96.0 GiB"}]


@pytest.mark.parametrize("text,ok,why", [
    ("\n".join([L_CFG, L_VK]), True, None),
    ("\n".join([L_CFG, L_DROP, L_ROCM]), False, "['ROCm']"),          # today's default on evo-x2
    ("\n".join([L_CFG, L_VK, L_ROCM]), False, "['ROCm']"),
    (L_CFG + "\n" + T + 'msg="no compatible GPUs were discovered"', False, "no GPU"),
    (L_VK.replace("8060S", "780M"), False, "names 8060S"),
    ("", False, "no inference compute"),
])
def test_check_server_backend(text, ok, why):
    r = xb.check_server_backend(text, "vulkan")
    assert r["ok"] is ok
    if why:
        assert any(why in x for x in r["reasons"])


def test_wait_for_inference_compute_polls_until_discovery(tmp_path):
    p = tmp_path / "s.log"
    p.write_text(L_CFG + "\n", encoding="utf-8")
    t = {"now": 0.0}

    def sleep(dt):
        t["now"] += dt
        if t["now"] >= 3:
            p.write_text(L_CFG + "\n" + L_VK + "\n", encoding="utf-8")
    text = xb.wait_for_inference_compute(p, timeout_s=60, sleep=sleep, clock=lambda: t["now"])
    assert "inference compute" in text and t["now"] == 3
    p.write_text(L_CFG + "\n", encoding="utf-8")
    t["now"] = 0.0
    text = xb.wait_for_inference_compute(p, timeout_s=5, sleep=lambda dt: t.__setitem__("now", t["now"] + dt),
                                         clock=lambda: t["now"])
    assert "inference compute" not in text and t["now"] == 5


def test_session_evidence_and_verified():
    ev = xb.session_evidence(_vk_session_log())
    assert ev["offloaded"] == [[33, 33]] and ev["gpu_memory_libraries"] == ["Vulkan"]
    assert ev["buffer_devices"] == ["CPU_Mapped", "Vulkan0"] and ev["forbidden_lines"] == []
    assert ev["vulkan_devices"] == ["0 = AMD Radeon(TM) 8060S Graphics (AMD proprietary driver)"]
    assert xb.session_verified(ev)
    assert not xb.session_verified(xb.session_evidence(_vk_session_log(22)))           # partial offload
    assert not xb.session_verified(xb.session_evidence(L_OFF.format(33, 33)))          # nothing names Vulkan
    assert not xb.session_verified(xb.session_evidence(L_MEM.format("Vulkan")))        # no offload line
    rocm = "\n".join([L_MEM.format("ROCm"), L_OFF.format(33, 33), L_BUF.format("ROCm0")])
    ev = xb.session_evidence(rocm)
    assert ev["forbidden_lines"] and not xb.session_verified(ev)
    mixed = _vk_session_log() + L_BUF.format("ROCm0") + "\n"
    assert not xb.session_verified(xb.session_evidence(mixed))
    # the server config line's CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES names are not ROCm/CUDA device lines
    assert xb.session_evidence(L_CFG)["forbidden_lines"] == []


class _PS:
    def get_ps(self):
        return {"models": [{"name": "llama3.1:8b", "size": 9518773696, "size_vram": 9518773696,
                            "context_length": 32768}]}


def test_recorder_emits_one_row_per_session(tmp_path):
    log = tmp_path / "s.log"
    log.write_text(L_CFG + "\n" + L_VK + "\n", encoding="utf-8")   # before the run: not counted
    inner = FakeRuntime(lambda c, m, t: ("", None))
    inner.client = _PS()
    rt = xb.BackendRuntime(inner, "vulkan")
    rows = []
    rec = xb.BackendRecorder(rt, log, rows.append, lambda m: None)
    rec.start("llama3.1:8b", NEG_ARM, 20260901)
    with open(log, "a", encoding="utf-8") as f:
        f.write(_vk_session_log())
    rt.loaded_context("llama3.1:8b")
    r = rec.end()
    assert rows == [r] and r["record"] == "backend_session" and r["verified"] is True
    assert r["seed"] == 20260901 and r["arm_id"] == NEG_ARM and r["backend"] == "vulkan"
    assert r["offloaded"] == [[33, 33]] and r["ps_size_vram"] == 9518773696
    with pytest.raises(ValueError):
        xb.BackendRuntime(inner, "cuda")


# ── CLI and main() ──────────────────────────────────────────────────────────────────────────────────

def test_job_argv_parses_to_the_preregistered_plan():
    a = ag.build_arg_parser().parse_args(JOB_ARGV)
    assert a.backend == "vulkan" and a.num_gpu is None and a.neg_ctx == 32768 and a.neg_sessions == 5
    plan, neg = ag.validation_plan_neg(a.call2_tools, a.neg_ctx, a.neg_sessions)
    assert neg == NEG and plan[0] == (NEG_ARM, SEEDS5, 10)
    assert plan[1] == ("ollama_ctx_8192_positive_control_call2_notools", (20260901,), 15)
    assert plan[2] == (NEG, tuple(ag.SEEDS), 10)
    assert ag.build_arg_parser().parse_args(["--mode", "validation", "--out", OUT]).backend is None


def _patch_main(monkeypatch, tray=False, server_head=None, session_log=None):
    import socket
    calls = {"start": [], "stop": 0, "notes": [], "chats": 0}
    head = server_head if server_head is not None else L_CFG + "\n" + L_VK + "\n"

    def start(env=None, log_path=None):
        calls["start"].append({"env": env, "log_path": log_path})
        Path(log_path).write_text(head, encoding="utf-8")

    def stop():
        calls["stop"] += 1

    monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
        start_ollama_server=start, stop_ollama_server=stop, wait_for_ollama_ready=lambda timeout_s=0: True,
        require_host=lambda h: {"hw_id": "evo-x2", "interactive_guard": False},
        enforce_or_record_interactive_session=lambda cfg: {}))
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=calls["notes"].append))
    monkeypatch.setattr(xb, "tray_app_running", lambda run=None: tray)
    monkeypatch.setattr(xb, "wait_for_inference_compute", lambda path, **k: xb.read_from(path, 0)[0])

    class RT(FakeRuntime):
        def __init__(self):
            super().__init__(lambda c, m, t: ("", [_native("log_event", event="e")]) if c == 1
                             else (_answer("ok", "SC"), None), loaded=32768)
            self.server_env = self.server_log = None
            self.options_extra = None
            self.client = _PS()
            self.loaded_model = False

        def chat(self, model, messages, num_ctx, tools, think, extra=None):
            calls["chats"] += 1
            if not self.loaded_model:   # a load logs the backend lines, as Ollama does
                with open(self.server_log, "a", encoding="utf-8") as f:
                    f.write(session_log if session_log is not None else _vk_session_log())
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


def _argv(out):
    return JOB_ARGV[:-1] + [str(out)]


def test_main_backend_runs_its_own_vulkan_server_and_records_each_session(tmp_path, monkeypatch):
    calls = _patch_main(monkeypatch)
    out = tmp_path / "vk.jsonl"
    note = ag.main(_argv(out))
    assert note == "completed" and calls["notes"] == ["completed"]
    assert calls["stop"] >= 2   # before its own start, and in finally
    assert calls["start"][0] == {"env": xb.BACKEND_ENVS["vulkan"], "log_path": xb.server_log_for(out)}
    assert calls["options"] and all(o == {} for o in calls["options"])   # request bodies unchanged
    rows = ag.read_rows(out)
    assert rows[0]["record"] == "run_start" and rows[0]["backend"] == "vulkan"
    assert rows[0]["backend_env"] == xb.BACKEND_ENVS["vulkan"] and rows[-1]["record"] == "run_end"
    srv = next(r for r in rows if r["record"] == "backend_server")
    assert srv["ok"] and srv["inference_compute"][0]["library"] == "Vulkan"
    rt = next(r for r in rows if r["record"] == "runtime")
    assert rt["server_env"] == xb.BACKEND_ENVS["vulkan"] and rt["server_log"] == xb.server_log_for(out)
    ses = [r for r in rows if r["record"] == "backend_session"]
    assert [r["seed"] for r in ses] == [20260901, 20260902] and all(r["verified"] for r in ses)
    assert not any(r["record"] == "backend_refused" for r in rows)


def test_main_backend_refuses_when_vulkan_did_not_load(tmp_path, monkeypatch):
    calls = _patch_main(monkeypatch, server_head=L_CFG + "\n" + L_DROP + "\n" + L_ROCM + "\n")
    out = tmp_path / "vk.jsonl"
    note = ag.main(_argv(out))
    assert note.startswith("stopped: refused, backend vulkan not loaded (no result claimed)") and "ROCm" in note
    assert calls["notes"] == [note] and calls["chats"] == 0 and calls["stop"] >= 2   # own server stopped in finally
    rows = ag.read_rows(out)
    ref = [r for r in rows if r["record"] == "backend_refused"]
    assert len(ref) == 1 and ref[0]["inference_compute"][0]["library"] == "ROCm"
    assert not any(r["record"] in ("r2a_turn", "r2a_session", "run_end") for r in rows)


def test_main_backend_refuses_while_the_tray_app_runs(tmp_path, monkeypatch):
    calls = _patch_main(monkeypatch, tray=True)
    note = ag.main(_argv(tmp_path / "o.jsonl"))
    assert note.startswith("stopped:") and "tray app" in note
    assert calls["start"] == [] and calls["stop"] == 0 and calls["notes"] == [note]


def test_main_backend_records_an_unverified_session(tmp_path, monkeypatch):
    calls = _patch_main(monkeypatch, session_log=L_MEM.format("ROCm") + "\n" + L_OFF.format(33, 33) + "\n")
    out = tmp_path / "vk.jsonl"
    assert ag.main(_argv(out)) == "completed" and calls["notes"] == ["completed"]
    ses = [r for r in ag.read_rows(out) if r["record"] == "backend_session"]
    assert ses and not any(r["verified"] for r in ses)


def test_backend_argument_errors(tmp_path, monkeypatch):
    _patch_main(monkeypatch)
    for argv in (["--mode", "mechanism", "--backend", "vulkan"], ["--mode", "validation", "--backend", "cuda"],
                 ["--mode", "validation", "--backend", "vulkan", "--num-gpu", "22"],
                 ["--mode", "real", "--backend", "vulkan", "--num-keep", "system"],
                 ["--mode", "validation", "--backend", "vulkan", "--server-env", "OLLAMA_LLM_LIBRARY=rocm_v7_1"]):
        with pytest.raises(SystemExit):
            ag.main(argv + ["--out", str(tmp_path / "x.jsonl")])


def test_sidecar_import_check_reports_backend_env(monkeypatch):
    import x2_r2_sidecar as sc
    monkeypatch.setattr(xb, "tray_app_running", lambda run=None: False)
    saved = list(sys.path)
    try:
        r = sc.import_check(JOB_ARGV, here=REPO / "harness")
    finally:
        sys.path[:] = saved
    assert r["args"]["backend"] == "vulkan" and r["backend_env"] == xb.BACKEND_ENVS["vulkan"]
    assert r["server_log"].endswith("x2_r2_llama_val_32k_vulkan_serve.log") and r["tray_app_running"] is False
    assert r["modules"]["x2_backend"].endswith("x2_backend.py") and r["plan"][0][0] == NEG_ARM


# ── register rows: the pre-registered verdict and the hours estimate ────────────────────────────────

def _text_tool_call_at(turn):
    def factory(ags):
        base = good_policy(ags)

        def policy(call_no, messages, turn_idx):
            if call_no == 2 and turn_idx == turn:
                return '{"name": "lookup_fact", "parameters": {"key": "REC-0005"}}', None
            return base(call_no, messages, turn_idx)
        return policy
    return factory


def _plain_wrong_at_5(ags):
    base = good_policy(ags)

    def policy(call_no, messages, turn_idx):
        if call_no == 2 and turn_idx == 5:
            return _answer("I do not know", "nope"), None
        return base(call_no, messages, turn_idx)
    return policy


def _vk_rows(policy_factory=good_policy, verified=True, drop_seed=None, refused=False):
    ag.neg_arm_for_ctx(32768)
    rows = [{"record": "run_start", "mode": "validation", "backend": "vulkan"},
            {"record": "backend_server", "backend": "vulkan", "ok": not refused,
             "inference_compute": [{"library": "ROCm" if refused else "Vulkan", "description": "8060S"}]}]
    if refused:
        return rows + [{"record": "backend_refused", "backend": "vulkan", "reasons": ["inference compute on ROCm"]}]
    for s in SEEDS5:
        ags = ag.build_agent_session(s, 10)
        rt = FakeRuntime(policy_factory(ags), loaded=32768)
        ag.run_session(rt, "llama3.1:8b", NEG_ARM, s, 10, rows.append, lambda m: None, "validation")
        if s != drop_seed:
            ev = xb.session_evidence(_vk_session_log() if verified else _vk_session_log(22))
            rows.append({"record": "backend_session", "model_id": "llama3.1:8b", "arm_id": NEG_ARM, "seed": s,
                         "backend": "vulkan", **ev, "verified": xb.session_verified(ev)})
    rows.append({"record": "run_end"})
    return rows


@pytest.fixture(scope="module")
def nr():
    import numbers_register
    return numbers_register


def test_verdict_vendor_when_no_miss_on_verified_vulkan(nr):
    v = nr.x2_r2_llama_val_32k_vulkan_verdict(_vk_rows(), ag)
    assert v["valid"] and v["verdict"] == "follows the vendor (0 misses on Vulkan)"
    assert v["n_missed_canaries"] == 0 and not v["backend_unverified"]


@pytest.mark.parametrize("turn", [5, 10])
def test_verdict_backend_implementation_on_text_tool_call_misses(nr, turn):
    v = nr.x2_r2_llama_val_32k_vulkan_verdict(_vk_rows(_text_tool_call_at(turn)), ag)
    assert v["verdict"] == "backend implementation, not the vendor (text-tool-call misses on Vulkan)"
    assert v["misses"] and all(m["text_tool_call"] and m["turn"] == turn for m in v["misses"])


def test_verdict_inconclusive_cases(nr):
    v = nr.x2_r2_llama_val_32k_vulkan_verdict(_vk_rows(_plain_wrong_at_5), ag)
    assert v["verdict"].startswith("inconclusive") and v["misses"]
    v = nr.x2_r2_llama_val_32k_vulkan_verdict(_vk_rows(verified=False), ag)
    assert v["verdict"].startswith("inconclusive (backend not verified as Vulkan")
    v = nr.x2_r2_llama_val_32k_vulkan_verdict(_vk_rows(drop_seed=20260903), ag)
    assert v["verdict"].startswith("inconclusive (backend not verified as Vulkan") and "20260903" in v["verdict"]
    rows = [r for r in _vk_rows() if r.get("record") != "run_end"]
    assert nr.x2_r2_llama_val_32k_vulkan_verdict(rows, ag)["verdict"].startswith("inconclusive (incomplete")
    v = nr.x2_r2_llama_val_32k_vulkan_verdict(_vk_rows(refused=True), ag)
    assert v["verdict"] == "refused (Vulkan not loaded)" and not v["valid"]


def test_register_rows_pending_then_compute(nr, tmp_path):
    entry = next(e for e in nr.NUMBER_ENTRIES if e["claim_id"] == "x2-r2-llama-val-32k-vulkan")
    assert entry["pending_ok"] and entry["data_files"] == [nr.X2_R2_LLAMA_VAL_32K_VULKAN]
    f = tmp_path / nr.X2_R2_LLAMA_VAL_32K_VULKAN
    f.parent.mkdir(parents=True)
    f.write_text("\n".join(json.dumps(r, default=str) for r in _vk_rows()) + "\n", encoding="utf-8")
    r = nr.compute_x2_r2_llama_val_32k_vulkan(tmp_path)
    assert r["value"].endswith("verdict: follows the vendor (0 misses on Vulkan)")
    assert "backend verified per negative-control session 5/5" in r["value"]
    with pytest.raises(FileNotFoundError):
        nr.compute_x2_r2_llama_val_32k_vulkan(tmp_path / "nothing")


def test_hours_row_from_the_measured_siblings(nr, tmp_path):
    entry = next(e for e in nr.NUMBER_ENTRIES if e["claim_id"] == "x2-r2-llama-val-32k-vulkan-hours")
    assert nr.X2_R2_LLAMA_VAL_32K in entry["data_files"]
    d = tmp_path / "results"
    d.mkdir()
    (d / "x2_r2_llama_val_32k.jsonl").write_text(
        json.dumps({"record": "run_start", "ts_utc": "2026-10-11T23:00:00+00:00"}) + "\n"
        + json.dumps({"record": "run_end", "ts_utc": "2026-10-11T23:12:00+00:00"}) + "\n", encoding="utf-8")
    r = nr.compute_x2_r2_llama_val_32k_vulkan_hours(tmp_path)
    assert "x2_r2_llama_val_32k 0.20 h (measured" in r["value"] and "offload not synced" in r["value"]
    assert "ESTIMATE 0.20 to 0.60 h" in r["value"]
    (d / "x2_r2_llama_val_32k_offload.jsonl").write_text(
        json.dumps({"record": "run_start", "ts_utc": "2026-10-14T23:00:00+00:00"}) + "\n"
        + json.dumps({"record": "run_end", "ts_utc": "2026-10-14T23:18:00+00:00"}) + "\n", encoding="utf-8")
    r = nr.compute_x2_r2_llama_val_32k_vulkan_hours(tmp_path)
    assert "x2_r2_llama_val_32k_offload 0.30 h (measured" in r["value"] and "ESTIMATE 0.20 to 0.90 h" in r["value"]
