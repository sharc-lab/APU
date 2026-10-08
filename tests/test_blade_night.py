"""Blade night wrapper, queue, C3 gate and job helpers, all with fakes (no powercfg, nvidia-smi, model or server)."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "scripts"))

import blade_common as bc  # noqa: E402
import blade_queue as bq  # noqa: E402
import blade_night as bn  # noqa: E402
import blade_c3  # noqa: E402
import blade_k1  # noqa: E402
import blade_r2  # noqa: E402


def ok_paused():
    return {"ok": True, "reasons": [], "heavy": []}


def make_power(events, values=None):
    fake = bn.FakePowercfg(values)

    def run(argv):
        if argv[1] in ("/setacvalueindex", "/setactive"):
            events.append(("power", argv[1], tuple(argv[3:6])))
        return fake(argv)
    return bn.Power(run=run, log=lambda m: None), fake


def night(tmp_path, events, runner, paused=ok_paused, flag=True, ac=True, values=None, n=1):
    if flag:
        (tmp_path / f"START_BLADE_NIGHT_{n}.flag").write_text("go\n")
    power, fake = make_power(events, values)
    summ = bn.run_night(n, power=power, ac_fn=lambda: ac, paused_fn=paused, runner=runner, blade_dir=tmp_path,
                        results_dir=tmp_path / "results", state_path=tmp_path / "state.json", log=lambda m: None)
    return summ, fake


def recording_runner(events, rc=0):
    def runner(job, log_path):
        events.append(("job", job["id"]))
        return rc(job) if callable(rc) else rc
    return runner


# ── wrapper ─────────────────────────────────────────────────────────────────────────────────────────

def test_refuses_without_start_flag(tmp_path):
    ev = []
    summ, fake = night(tmp_path, ev, recording_runner(ev), flag=False)
    assert "no start flag" in summ["refused"]
    assert ev == [] and fake.calls == []


def test_refuses_on_battery_and_when_not_paused(tmp_path):
    ev = []
    summ, fake = night(tmp_path, ev, recording_runner(ev), ac=False)
    assert summ["refused"] == "not on AC power" and fake.calls == []
    summ, fake = night(tmp_path, ev, recording_runner(ev),
                       paused=lambda: {"ok": False, "reasons": ["python.exe(12)"], "heavy": []})
    assert "paused condition" in summ["refused"] and ev == [] and fake.calls == []


def test_set_before_jobs_restore_after_and_values_back(tmp_path):
    ev = []
    summ, fake = night(tmp_path, ev, recording_runner(ev))
    kinds = [e[0] for e in ev]
    first_job, last_job = kinds.index("job"), len(kinds) - 1 - kinds[::-1].index("job")
    assert all(k == "power" for k in kinds[:first_job]) and first_job == 4      # 3 sets + setactive
    assert [e[2][2] for e in ev[:3]] == ["0", "0", "0"]
    assert kinds[last_job + 1:] == ["power"] * 4
    assert fake.values == bn.FakePowercfg().values
    assert summ["power"]["restored"] is True
    assert [j["id"] for j in summ["queue"]["jobs"]] == [j["id"] for j in bq.NIGHTS[1]]
    log = json.loads(Path(summ["power"]["log"]).read_text())
    assert log["original"]["SUB_SLEEP/STANDBYIDLE"]["ac"] == 1800 and log["restored"] is True
    assert Path(summ["summary_file"]).name.startswith("blade_night1_summary_")
    assert not (tmp_path / "START_BLADE_NIGHT_1.flag").exists()


def test_restore_on_failure(tmp_path, monkeypatch):
    ev = []

    def boom(*a, **k):
        raise RuntimeError("queue crashed")
    monkeypatch.setattr(bq, "run_night", boom)
    (tmp_path / "START_BLADE_NIGHT_1.flag").write_text("go\n")
    power, fake = make_power(ev)
    with pytest.raises(RuntimeError):
        bn.run_night(1, power=power, ac_fn=lambda: True, paused_fn=ok_paused, runner=None, blade_dir=tmp_path,
                     results_dir=tmp_path / "r", log=lambda m: None)
    assert fake.values == bn.FakePowercfg().values
    logs = list((tmp_path / "logs").glob("night1_power_*.json"))
    assert json.loads(logs[0].read_text())["restored"] is True


def test_restore_when_a_job_raises_keyboard_interrupt(tmp_path):
    ev = []

    def runner(job, log_path):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        night(tmp_path, ev, runner)
    logs = list((tmp_path / "logs").glob("night1_power_*.json"))
    assert json.loads(logs[0].read_text())["restored"] is True


def test_unrestored_earlier_log_supplies_originals(tmp_path):
    (tmp_path / "logs").mkdir()
    prior = {"original": {"SUB_SLEEP/STANDBYIDLE": {"ac": 900}, "SUB_SLEEP/HIBERNATEIDLE": {"ac": 3600},
                          "SUB_BUTTONS/LIDACTION": {"ac": 1}}, "restored": False}
    (tmp_path / "logs" / "night1_power_20261001T000000Z.json").write_text(json.dumps(prior))
    ev = []
    summ, fake = night(tmp_path, ev, recording_runner(ev),
                       values={"SUB_SLEEP/STANDBYIDLE": 0, "SUB_SLEEP/HIBERNATEIDLE": 0, "SUB_BUTTONS/LIDACTION": 0})
    assert fake.values == {"SUB_SLEEP/STANDBYIDLE": 900, "SUB_SLEEP/HIBERNATEIDLE": 3600, "SUB_BUTTONS/LIDACTION": 1}


def test_stub_night_passes_both_nights():
    for n in (1, 2):
        rep = bn.stub_night(n, log=lambda m: None)
        assert rep["set_before_first_job"] and rep["restore_after_last_job"]
        assert rep["final_values_equal_original"] and rep["all_outputs_prefixed_blade"]
        assert len(rep["jobs"]) == len(bq.NIGHTS[n])


# ── queue ───────────────────────────────────────────────────────────────────────────────────────────

def test_queue_resume_and_statuses(tmp_path):
    ev = []
    codes = {"blade_r2_real_v1": 2, "blade_r2_mitigation_v1": 3}
    s1 = bq.run_night(1, runner=recording_runner(ev, lambda j: codes.get(j["id"], 0)), state_path=tmp_path / "s.json",
                      paused_fn=ok_paused, log=lambda m: None)
    assert [j["status"] for j in s1["jobs"]] == ["done", "done", "error", "blocked_dependency"]
    ev.clear()
    bq.run_night(1, runner=recording_runner(ev), state_path=tmp_path / "s.json", paused_fn=ok_paused, log=lambda m: None)
    assert ev == [("job", "blade_r2_real_v1")]    # only the errored job reruns


def test_queue_reruns_a_job_left_running(tmp_path):
    st = bq.init_night({"nights": {}}, 2)
    st["nights"]["2"]["jobs"]["blade_c3_sysmem_fallback_v1"]["status"] = "running"
    st["nights"]["2"]["jobs"]["blade_c3_sysmem_fallback_v1"]["attempts"] = 1
    bq.save_state(st, tmp_path / "s.json")
    ev = []
    bq.run_night(2, runner=recording_runner(ev), state_path=tmp_path / "s.json", paused_fn=ok_paused, log=lambda m: None)
    assert ev[0] == ("job", "blade_c3_sysmem_fallback_v1")
    assert bq.load_state(tmp_path / "s.json")["nights"]["2"]["jobs"]["blade_c3_sysmem_fallback_v1"]["attempts"] == 2


def test_queue_stops_when_paused_flag_removed(tmp_path):
    ev, calls = [], []

    def paused():
        calls.append(1)
        return {"ok": len(calls) < 2, "reasons": ["flag removed"]}
    s = bq.run_night(1, runner=recording_runner(ev), state_path=tmp_path / "s.json", paused_fn=paused, log=lambda m: None)
    assert ev == [("job", "blade_k1_v1")] and s["stopped_early"]["before"] == "blade_r2_validation_v1"


def test_heavy_processes_excludes_own_tree_and_ollama():
    procs = [{"pid": 1, "ppid": 0, "name": "explorer.exe"},
             {"pid": 10, "ppid": 1, "name": "py.exe"}, {"pid": 11, "ppid": 10, "name": "python.exe"},
             {"pid": 12, "ppid": 11, "name": "python.exe"},                          # our job child
             {"pid": 20, "ppid": 1, "name": "python.exe", "cmd": "pytest -q"},
             {"pid": 21, "ppid": 1, "name": "git.exe"},
             {"pid": 30, "ppid": 1, "name": "ollama.exe"},
             {"pid": 31, "ppid": 30, "name": "llama-server.exe", "path": r"C:\x\Ollama\lib\llama-server.exe"},
             {"pid": 40, "ppid": 1, "name": "llama-server.exe", "path": r"C:\apu\bin\llama-server.exe"}]
    heavy = {h["pid"] for h in bq.heavy_processes(procs, 11)}
    assert heavy == {20, 21, 40}


def test_paused_condition_needs_flag(tmp_path):
    c = bq.paused_condition(flag=tmp_path / "nope.flag", procs_fn=lambda: [], my_pid=1)
    assert not c["ok"] and "nope.flag" in c["reasons"][0]
    (tmp_path / "p.flag").write_text("")
    assert bq.paused_condition(flag=tmp_path / "p.flag", procs_fn=lambda: [], my_pid=1)["ok"]


def test_every_queued_output_is_prefixed_and_argv_parses():
    import importlib
    for n, jobs in bq.NIGHTS.items():
        for j in jobs:
            assert all(Path(o).name.startswith("blade_") for o in j["outputs"])
            mod = importlib.import_module(Path(j["argv"][0]).stem)
            args = mod.build_arg_parser().parse_args(j["argv"][1:])
            assert Path(args.out).name.startswith("blade_")


# ── C3 gate ─────────────────────────────────────────────────────────────────────────────────────────

class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def test_c3_gate_waits_for_exact_text_newer_than_wait(tmp_path):
    rows = []
    half = blade_c3.HALVES[0]
    flag = tmp_path / "c3_confirm_A.flag"
    flag.write_text(half["policy"] + "\n")                 # left from an earlier run: must not count
    clock = Clock()
    step = {"n": 0}

    def sleep(_):
        step["n"] += 1
        clock.t += 10
        if step["n"] == 2:
            flag.write_text("Driver Default\n")              # wrong value
        if step["n"] == 4:
            flag.write_text(half["policy"] + "\nok\n")
        import os
        if flag.exists():
            os.utime(flag, (clock.t, clock.t))

    row = blade_c3.wait_for_confirmation(half, tmp_path, lambda r: rows.append(r) or r, lambda m: None,
                                         sleep=sleep, now=clock, poll_s=10, max_polls=50)
    recs = [r["record"] for r in rows]
    assert recs[0] == "c3_gate_stale_flag_moved" and "c3_gate_rejected" in recs
    assert row["record"] == "c3_gate_confirmed" and row["policy"] == half["policy"]
    assert not (tmp_path / "c3_WAITING_A.txt").exists()


def test_c3_gate_has_no_timeout_of_its_own(tmp_path):
    clock = Clock()

    def sleep(_):
        clock.t += 3600
    with pytest.raises(TimeoutError, match="test-only"):
        blade_c3.wait_for_confirmation(blade_c3.HALVES[1], tmp_path, lambda r: r, lambda m: None, sleep=sleep,
                                       now=clock, max_polls=48)      # 48 simulated hours, still waiting


def test_c3_halves_design():
    a, b = blade_c3.HALVES
    assert a["policy"] == "Prefer No Sysmem Fallback" and sorted(a["ctxs"]) == [36864, 38912, 40960, 43008]
    assert b["policy"] == "Driver Default" and b["ctxs"] == (40960, 43008) and b["n_measured"] == 3
    assert sorted(blade_c3.ordered_ctxs(a)) == sorted(a["ctxs"])
    assert blade_c3.halves_done([{"record": "c3_half_done", "half": "A"}]) == {"A"}


# ── R2 / K1 helpers ─────────────────────────────────────────────────────────────────────────────────

def test_r2_tiers_and_plans():
    assert blade_r2.tier_arm("default") == "ollama_default_call2_notools"
    assert blade_r2.tier_arm("4096") == "ollama_ctx_4096_call2_notools"
    real = blade_r2.plan_for("real", ["default", "4096", "32768"])
    assert [a for a, _, _ in real] == ["ollama_default_call2_notools", "ollama_ctx_4096_call2_notools",
                                       "ollama_ctx_32768_call2_notools"]
    assert all(len(s) == 5 and t == 40 for _, s, t in real)
    mech = blade_r2.plan_for("mechanism", ["default", "4096"])
    assert all(len(s) == 1 for _, s, _ in mech)
    val = blade_r2.plan_for("validation", [])
    assert val[0][0] == "ollama_ctx_131072_call2_notools"


def test_r2_fit_decision_and_default_tier(tmp_path):
    p = tmp_path / "k1.summary.json"
    p.write_text(json.dumps({"models": {"qwen3:8b": {"default_ctx": 4096, "fits_8gb_at_default": True, "size": 1,
                                                     "size_vram": 1},
                                        "llama3.1:8b": {"default_ctx": 4096, "fits_8gb_at_default": True}}}))
    assert blade_r2.fit_decision(["qwen3:8b"], p)["qwen3:8b"]["run"] is True
    assert blade_r2.fit_decision(["qwen3:8b"], tmp_path / "none.json")["qwen3:8b"]["run"] is False
    tiers, note = blade_r2.resolve_default_tier(["default", "4096", "8192"], p, "llama3.1:8b")
    assert tiers == ["4096", "8192"] and "4096" in note
    tiers, note = blade_r2.resolve_default_tier(["default", "4096"], None, "llama3.1:8b")
    assert tiers == ["4096"] and "dropped" in note


def _mitigation_job_args():
    job = next(j for j in bq.NIGHTS[1] if j["id"] == "blade_r2_mitigation_v1")
    return blade_r2.build_arg_parser().parse_args(job["argv"][1:])


def test_mitigation_job_argv_parses_with_x2_r2_agent_parser():
    args = _mitigation_job_args()
    argv, spec = blade_r2.check_mitigation_args(args, ["llama3.1:8b"])
    parsed = blade_r2.agent.build_arg_parser().parse_args(argv)
    assert parsed.mode == "mitigation" and parsed.client_trim == "margin=0.05" and parsed.call2_tools == "off"
    assert spec == {"margin": 0.05}
    assert args.tiers == "default,4096,8192"


def test_mitigation_requires_client_trim_and_rejects_bad_spec():
    args = _mitigation_job_args()
    args.client_trim = None
    with pytest.raises(SystemExit):
        blade_r2.check_mitigation_args(args, ["llama3.1:8b"])
    args.client_trim = "margin=0.9"
    with pytest.raises(ValueError):
        blade_r2.check_mitigation_args(args, ["llama3.1:8b"])


def test_mitigation_plan_is_x2_design_on_blade_tiers(tmp_path):
    p = tmp_path / "k1.summary.json"
    p.write_text(json.dumps({"models": {"llama3.1:8b": {"default_ctx": 40960}}}))
    tiers, _ = blade_r2.resolve_default_tier(["default", "4096", "8192"], p, "llama3.1:8b")
    plan = blade_r2.plan_for("mitigation", tiers)
    assert [a for a, _, _ in plan] == ["ollama_ctx_40960_call2_notools", "ollama_ctx_4096_call2_notools",
                                       "ollama_ctx_8192_call2_notools"]
    assert all(s == tuple(blade_r2.agent.SEEDS) and t == 40 for _, s, t in plan)
    assert blade_r2.agent.ARMS["ollama_ctx_40960_call2_notools"] == {"num_ctx": 40960, "call2_tools": False,
                                                                    "call2_mode": "off"}
    x2 = blade_r2.agent.mitigation_plan("off")
    assert [a for a, _, _ in x2] == [a for a, _, _ in plan[1:]]   # same arms as evo-x2's run at 4096/8192


def test_mitigation_calls_x2_run_mitigation_with_blade_server(monkeypatch):
    out = bc.DRYRUN_DIR / "blade_dryrun_test_mitigation_wiring.jsonl"
    calls = {}
    monkeypatch.setattr(bc, "require_blade", lambda *a: {})
    monkeypatch.setattr(bc, "versions_record", lambda *a, **k: {})
    monkeypatch.setattr(bc, "version_problems", lambda rec: [])
    monkeypatch.setattr(bc, "gpu_memory", lambda *a: {})
    monkeypatch.setattr(bc.LocalOllama, "start", lambda self: calls.setdefault("env", dict(self.env)))
    monkeypatch.setattr(bc.LocalOllama, "wait_ready", lambda self, **k: True)
    monkeypatch.setattr(bc.LocalOllama, "stop", lambda self, **k: [])
    monkeypatch.setattr(blade_r2.BladeOllamaRuntime, "available_models", lambda self: ["llama3.1:8b"])
    monkeypatch.setattr(blade_r2.BladeOllamaRuntime, "version", lambda self: "0.34.4")

    def fake_run_mitigation(runtime, server_log, plan, models, out_path, emit, log, order, client_trim, rules):
        calls.update(server_log=server_log, plan=plan, models=models, client_trim=client_trim, order=order)
        return {"trim_summary": {}}
    monkeypatch.setattr(blade_r2.agent, "run_mitigation", fake_run_mitigation)
    try:
        rc = blade_r2.main(["--mode", "mitigation", "--out", str(out), "--tiers", "4096,8192",
                            "--client-trim", "margin=0.05", "--dry-run-seconds", "60"])
        assert rc == 0
        assert calls["env"] == {"OLLAMA_DEBUG": "1"} and calls["client_trim"] == {"margin": 0.05}
        assert calls["server_log"].endswith(".ollama_serve.log") and calls["models"] == ["llama3.1:8b"]
        start = json.loads(out.read_text().splitlines()[0])
        assert start["x2_r2_agent_equivalent_argv"][:4] == ["--mode", "mitigation", "--client-trim", "margin=0.05"]
    finally:
        out.unlink(missing_ok=True)


def test_k1_log_parsers_and_summary():
    log = ('time=x level=INFO source=types.go:42 msg="inference compute" id=GPU-1 library=CUDA compute=8.9 '
           'name=CUDA0 description="NVIDIA GeForce RTX 4070 Laptop GPU" total="8.0 GiB" available="7.1 GiB"\n'
           'time=x level=INFO source=routes.go:1 msg="vram-based default context" total_vram="8.0 GiB" '
           'default_num_ctx=4096\n')
    d = blade_k1.parse_ollama_server_log(log)
    assert d["vram_default_ctx"] == 4096 and d["libraries"] == ["CUDA"] and d["device_guess"] == "gpu"
    ll = blade_k1.parse_llama_server_log("load_tensors: offloaded 33/33 layers to GPU\nllama_context: n_ctx = 131072\n"
                                         "llama_kv_cache: CUDA0 KV buffer size = 16384.00 MiB\n")
    assert ll["offloaded"] == [33, 33] and ll["n_ctx"] == 131072 and ll["kv_buffers_mib"]["CUDA0"] == 16384.0
    s = blade_k1.summarize([{"record": "blade_k1_default_ctx", "model_tag": "a", "context_length": 4096,
                             "size": 100, "size_vram": 100, "fully_on_gpu": True},
                            {"record": "blade_k1_default_ctx", "model_tag": "b", "context_length": 4096,
                             "size": 100, "size_vram": 60, "fully_on_gpu": False},
                            {"record": "blade_k1_model_missing", "model_tag": "c"}])
    assert s["models"]["a"]["fits_8gb_at_default"] and not s["models"]["b"]["fits_8gb_at_default"]
    assert not s["models"]["c"]["fits_8gb_at_default"]
    assert blade_k1.half_window(4096) == 2050
    assert blade_k1.fully_on_gpu({"size": 100, "size_vram": 99}) and not blade_k1.fully_on_gpu({"size": 100,
                                                                                                "size_vram": 50})


# ── common ──────────────────────────────────────────────────────────────────────────────────────────

def test_version_parsing_and_pins():
    assert bc.parse_ollama_version("ollama version is 0.34.1\n") == "0.34.1"
    assert bc.parse_llama_version("version: 0.4.1-dev (build 10970, commit bfdc32183)")["build"] == "10970"
    assert bc.parse_nvidia_smi_header("| NVIDIA-SMI 610.88   KMD Version: 610.88   CUDA UMD Version: 13.3 |")[
        "cuda_driver_api"] == "13.3"
    good = {"ollama_version": "0.34.4", "llama_server_version": {"build": "10970", "commit": "bfdc32183"},
            "nvidia_driver": "610.88", "llama_cuda_runtime_dlls": ["cudart64_12.dll"]}
    assert bc.version_problems(good) == []
    assert len(bc.version_problems({**good, "ollama_version": "0.34.1"})) == 1


def test_dry_run_clock_needs_a_row_before_stopping():
    t = {"v": 0.0}
    c = bc.DryRunClock(60, clock=lambda: t["v"])
    t["v"] = 120
    c.check()                       # no row yet: keeps going
    with pytest.raises(bc.DryRunTimeUp):
        c.wrote()
    bc.DryRunClock(None).wrote()    # a real run never stops


def test_out_path_rules():
    with pytest.raises(SystemExit):
        bc.out_path_ok("results/k1.jsonl", False)
    with pytest.raises(SystemExit):
        bc.out_path_ok("results/blade_k1.jsonl", True)
    assert bc.out_path_ok(bc.DRYRUN_DIR / "blade_dryrun_k1.jsonl", True)


def test_local_ollama_stop_kills_server_tray_and_runners_only(monkeypatch):
    procs = [{"pid": 1, "name": "ollama.exe", "path": ""}, {"pid": 2, "name": "ollama app.exe", "path": ""},
             {"pid": 3, "name": "llama-server.exe", "path": r"D:\apps\Ollama\lib\ollama\llama-server.exe"},
             {"pid": 4, "name": "llama-server.exe", "path": r"C:\apu\bin\llama-b10970-cuda\llama-server.exe"}]
    killed = []
    monkeypatch.setattr(bc, "kill_pid", lambda pid, run=None: killed.append(pid))
    state = {"n": 0}

    def procs_fn():
        state["n"] += 1
        return procs if state["n"] == 1 else [procs[3]]
    srv = bc.LocalOllama(exe="x", procs_fn=procs_fn)
    srv.stop(wait_s=5)
    assert sorted(killed) == [1, 2, 3]


def test_blade_host_entry():
    import host_config as hc
    h = hc.HOSTS["RITZLAPTOP"]
    assert h["hw_id"] == "blade" and h["gpu_vendor"] == "nvidia" and h["ssh_host"] is None
    assert h["models_dir"] == "C:\\apu\\models"
    assert "blade" not in hc.ALIASES       # never an SSH target for deploy/sync
