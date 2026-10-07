"""P4 (2026-10-06): llama-3.3-70b edge reps. Covers the parsing/classification this job adds -- llama-server log
stage/timestamp/signature parsing, outcome classification, exit-code naming, Get-Counter sample parsing, the
page-cache residency summary, the ABAB arm interleave, and the parent orchestrator's resume / child-failure /
STOP handling -- with every real subprocess, counter, and GPU call replaced by fakes."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import x2_70b_edge_reps as er  # noqa: E402

E = "\x1b[34m"
R = "\x1b[0m"
G = "\x1b[32m"

# Shape of the real by-hand 70B re-test log (rep 3, 2026-10-06): ANSI colour, llama.cpp m.ss.mmm.uuu prefix,
# last line the Vulkan_Host model buffer size at ~3.7s, then silence.
STALL_LOG = "\n".join([
    f"{E}0.00.512.100{R} {G}I {R}llama_model_loader: loaded meta data with 36 key-value pairs",
    f"{E}0.01.071.017{R} {G}I {R}print_info: max token length      = 256",
    f"{E}0.01.073.207{R} {G}I {R}load_tensors: loading model tensors, this can take a while... (load_mode = none)",
    f"{E}0.03.725.353{R} {G}I {R}load_tensors:      Vulkan0 model buffer size = 39979.48 MiB",
    f"{E}0.03.725.361{R} {G}I {R}load_tensors:  Vulkan_Host model buffer size =   563.63 MiB",
    "",
])

# The real qwen3-32b HARD_FAIL signature from the same correction section.
ALLOC_FAIL_LOG = "\n".join([
    "0.00.400.000 I llama_model_loader: loaded meta data with 30 key-value pairs",
    "0.02.000.000 I load_tensors: loading model tensors, this can take a while...",
    "0.02.100.000 I load_tensors:      Vulkan0 model buffer size = 18000.00 MiB",
    "0.20.000.000 I llama_context: constructing llama_context",
    "0.21.000.000 I llama_kv_cache:    Vulkan0 KV buffer size = 90000.00 MiB",
    "0.32.700.000 E ggml_gallocr_reserve_n_impl: failed to allocate Vulkan0 buffer of size 571222020",
    "0.32.700.100 E graph_reserve: failed to allocate compute buffers",
    "0.32.700.200 E llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers",
])

STARTED_LOG = "\n".join([
    "0.00.400.000 I llama_model_loader: loaded meta data",
    "0.03.000.000 I load_tensors: loading model tensors",
    "0.03.100.000 I load_tensors: Vulkan0 model buffer size = 39979.48 MiB",
    "1.10.000.000 I llama_context: constructing llama_context",
    "1.11.000.000 I llama_kv_cache: Vulkan0 KV buffer size = 69280.00 MiB",
    "1.12.000.000 I llama_context: Vulkan0 compute buffer size = 1104.00 MiB",
    "1.20.000.000 I common_init_from_params: warming up the model with an empty run",
    "1.21.500.000 I main: server is listening on http://127.0.0.1:58699",
])


# ------------------------------------------------------------------------------------------------ parse_log
def test_parse_log_stall_shape_strips_ansi_and_finds_stage_and_timestamp():
    p = er.parse_log(STALL_LOG)
    assert p["last_stage"] == "model_buffers_sized"
    assert p["last_log_ts_s"] == pytest.approx(3.725361)
    assert p["fail_signatures"] == []
    assert "Vulkan_Host model buffer size" in p["log_tail"][-1]
    assert "\x1b" not in "".join(p["log_tail"])


def test_parse_log_minutes_field_counts_as_sixty_seconds():
    p = er.parse_log(STARTED_LOG)
    assert p["last_stage"] == "listening"
    assert p["last_log_ts_s"] == pytest.approx(81.5)


def test_parse_log_alloc_failure_signatures():
    p = er.parse_log(ALLOC_FAIL_LOG)
    assert p["last_stage"] == "kv_cache"
    assert {"vk_buffer_alloc_fail", "compute_buffer_alloc_fail", "context_init_fail"} <= set(p["fail_signatures"])
    assert any("failed to allocate Vulkan0 buffer" in l for l in p["fail_lines"])


def test_parse_log_empty():
    p = er.parse_log("")
    assert p["last_stage"] is None and p["last_log_ts_s"] is None and p["n_log_lines"] == 0


# ----------------------------------------------------------------------------------------- classify_outcome
def test_classify_stall_reports_stage_and_idle_seconds():
    c = er.classify_outcome(False, None, True, er.parse_log(STALL_LOG), 600.0)
    assert c["outcome"] == "STALL"
    assert c["stall_stage"] == "model_buffers_sized"
    assert c["log_idle_s"] == pytest.approx(600.0 - 3.725361)


def test_classify_crash_with_real_nonzero_code():
    c = er.classify_outcome(False, 3221226505, False, er.parse_log(STALL_LOG), 347.2)
    assert c["outcome"] == "CRASH" and c["crash_stage"] == "model_buffers_sized" and c["log_idle_s"] is None


def test_classify_started_wins_and_exit_zero_without_start_is_distinct():
    assert er.classify_outcome(True, None, False, er.parse_log(STARTED_LOG), 82.0)["outcome"] == "STARTED"
    assert er.classify_outcome(False, 0, False, {}, 5.0)["outcome"] == "EXIT_CLEAN_NO_START"
    assert er.classify_outcome(False, None, False, None, 0.0, launch_error="boom")["outcome"] == "LAUNCH_ERROR"


def test_exit_code_info_names_the_two_real_codes():
    a = er.exit_code_info(3221226505)
    assert a["exit_code_hex"] == "0xC0000409" and a["exit_code_name"] == "STATUS_STACK_BUFFER_OVERRUN"
    b = er.exit_code_info(3221225477)
    assert b["exit_code_hex"] == "0xC0000005" and b["exit_code_name"] == "STATUS_ACCESS_VIOLATION"
    assert er.exit_code_info(-1073740791)["exit_code_hex"] == "0xC0000409"  # signed form
    assert er.exit_code_info(None)["exit_code"] is None
    assert er.exit_code_info(12345)["exit_code_name"] == "UNKNOWN"


# ------------------------------------------------------------------------------------ counter sample parsing
SAMPLES = [
    {"Path": r"\\evo-x2\memory\available bytes", "InstanceName": "", "CookedValue": 59826614272},
    {"Path": r"\\evo-x2\memory\standby cache normal priority bytes", "InstanceName": "", "CookedValue": 49333170176},
    {"Path": r"\\evo-x2\memory\standby cache reserve bytes", "InstanceName": "", "CookedValue": 2112675840},
    {"Path": r"\\evo-x2\memory\standby cache core bytes", "InstanceName": "", "CookedValue": 127377408},
    {"Path": r"\\evo-x2\memory\modified page list bytes", "InstanceName": "", "CookedValue": 217964544},
    {"Path": r"\\evo-x2\memory\free & zero page list bytes", "InstanceName": "", "CookedValue": 8253390848},
    {"Path": r"\\evo-x2\gpu adapter memory(luid_0x00000000_0x0085359a_phys_0)\dedicated usage",
     "InstanceName": "luid_0x00000000_0x0085359a_phys_0", "CookedValue": 64891183104},
    {"Path": r"\\evo-x2\gpu adapter memory(luid_0x00000000_0x0001054c_phys_0)\dedicated usage",
     "InstanceName": "luid_0x00000000_0x0001054c_phys_0", "CookedValue": 0},
    {"Path": r"\\evo-x2\gpu adapter memory(luid_0x00000000_0x0085359a_phys_0)\shared usage",
     "InstanceName": "luid_0x00000000_0x0085359a_phys_0", "CookedValue": 1197944832},
    {"Path": r"\\evo-x2\gpu process memory(pid_2472_luid_0x00000000_0x0085359a_phys_0)\shared usage",
     "InstanceName": "pid_2472_luid_0x00000000_0x0085359a_phys_0", "CookedValue": 477261824},
    {"Path": r"\\evo-x2\gpu process memory(pid_5920_luid_0x00000000_0x0085359a_phys_0)\dedicated usage",
     "InstanceName": "pid_5920_luid_0x00000000_0x0085359a_phys_0", "CookedValue": 64000000000},
    {"Path": r"\\evo-x2\gpu process memory(pid_5920_luid_0x00000000_0x0085359a_phys_0)\shared usage",
     "InstanceName": "pid_5920_luid_0x00000000_0x0085359a_phys_0", "CookedValue": 693882880},
    {"Path": r"\\evo-x2\gpu process memory(pid_9_luid_0x00000000_0x0085359a_phys_0)\shared usage",
     "InstanceName": "pid_9_luid_0x00000000_0x0085359a_phys_0", "CookedValue": 0},
]


def test_parse_counter_samples_memory_adapters_and_processes():
    p = er.parse_counter_samples(SAMPLES)
    assert p["memory"]["available_bytes"] == 59826614272
    assert p["memory"]["free_zero_page_list_bytes"] == 8253390848
    assert er.standby_total(p) == 49333170176 + 2112675840 + 127377408
    ad = p["gpu_adapters"]["luid_0x00000000_0x0085359a_phys_0"]
    assert ad["dedicated_usage"] == 64891183104 and ad["shared_usage"] == 1197944832
    assert er.adapter_dedicated_total(p) == 64891183104
    pids = [x["pid"] for x in p["gpu_processes"]]
    assert pids == [5920, 2472]  # sorted by total, pid 9 (all zero) dropped
    assert p["gpu_processes"][0]["dedicated_bytes"] == 64000000000
    assert p["gpu_processes"][0]["luid"] == "luid_0x00000000_0x0085359a"


def test_parse_counter_samples_tolerates_single_dict_and_empty():
    assert er.parse_counter_samples(SAMPLES[0])["memory"]["available_bytes"] == 59826614272
    p = er.parse_counter_samples([])
    assert p["gpu_processes"] == [] and er.adapter_dedicated_total(p) is None and er.standby_total(p) is None


def test_read_counters_with_fake_powershell():
    calls = []

    def fake_ps(cmd, timeout):
        calls.append(cmd)
        return json.dumps(SAMPLES)
    p = er.read_counters(ps_fn=fake_ps)
    assert p["counter_read_ok"] is True and er.adapter_dedicated_total(p) == 64891183104
    assert "Free & Zero Page List Bytes" in calls[0] and "GPU Process Memory(*)" in calls[0]
    bad = er.read_counters(ps_fn=lambda c, t: "not json")
    assert bad["counter_read_ok"] is False


# ------------------------------------------------------------------------------------------------ residency
def test_summarize_residency_fraction_and_quantiles():
    lat = [20e-6] * 75 + [400e-6] * 25
    s = er.summarize_residency(lat)
    assert s["n"] == 100 and s["cached_fraction_est"] == 0.75
    assert s["p10_s"] == 20e-6 and s["p90_s"] == 400e-6
    assert er.summarize_residency([])["cached_fraction_est"] is None


def test_residency_probe_reads_a_real_small_file(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"\0" * (4 * 2 ** 20))
    s = er.residency_probe(path=str(f), n=16, chunk=4096)
    assert s["n"] == 16 and 0.0 <= s["cached_fraction_est"] <= 1.0
    assert "error" in er.residency_probe(path=str(tmp_path / "missing.gguf"))


# ------------------------------------------------------------------------------------------ arms / summary
def test_arms_are_interleaved_abab_and_balanced():
    arms = [er.arm_for_rep(i) for i in range(er.DEFAULT_REPS)]
    assert arms[:4] == ["warm_read", "cold_purge", "warm_read", "cold_purge"]
    assert arms.count("warm_read") == arms.count("cold_purge") == 5


def test_summarize_counts_outcomes_per_arm():
    rows = [{"arm": "warm_read", "outcome": "CRASH"}, {"arm": "warm_read", "outcome": "CRASH"},
            {"arm": "cold_purge", "outcome": "STALL"}]
    assert er.summarize(rows) == {"warm_read": {"CRASH": 2}, "cold_purge": {"STALL": 1}}


def test_server_cmd_matches_the_recheck_flags():
    cmd = er.server_cmd("x.log")
    s = " ".join(cmd)
    for flag in ("-c 221696", "-ctk f16", "-ctv f16", "-fa on", "-np 1", "-t 4", "--no-context-shift", "-ngl 99",
                 f"--port {er.PORT}", "--log-verbosity 4"):
        assert flag in s
    assert er.START_BUDGET_S == 600


# ------------------------------------------------------------------------------------- parent orchestrator
def _fake_spawn_factory(out_path, outcomes, seen):
    def spawn(rep, arm):
        seen.append((rep, arm))
        o = outcomes.get(rep, "CRASH")
        if o is None:
            return 1  # child died without writing a row
        row = {"record": "x2_70b_edge_rep", "rep": rep, "arm": arm, "outcome": o}
        if o == "NOT_RUN":
            row["error"] = "STOP: a llama-server process we did not start is running: [...]"
        er.emit(out_path, row)
        return 0
    return spawn


def test_run_spawns_one_child_per_rep_in_abab_order(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "kill_our_servers", lambda: [])
    out = tmp_path / "o.jsonl"
    seen = []
    er.run(out, 4, log=lambda m: None, spawn=_fake_spawn_factory(out, {1: "STALL"}, seen))
    assert seen == [(0, "warm_read"), (1, "cold_purge"), (2, "warm_read"), (3, "cold_purge")]
    rows = er.read_rows(out)
    summ = next(r for r in rows if r["record"] == "summary")
    assert summ["outcomes_by_arm"] == {"warm_read": {"CRASH": 2}, "cold_purge": {"CRASH": 1, "STALL": 1}}
    assert rows[-1]["record"] == "run_end"


def test_run_records_child_failed_and_resume_retries_only_that_rep(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "kill_our_servers", lambda: [4242])
    out = tmp_path / "o.jsonl"
    seen = []
    er.run(out, 3, log=lambda m: None, spawn=_fake_spawn_factory(out, {1: None}, seen))
    failed = [r for r in er.read_rows(out) if r.get("outcome") == "CHILD_FAILED"]
    assert len(failed) == 1 and failed[0]["rep"] == 1 and failed[0]["killed_orphan_servers"] == [4242]
    seen2 = []
    er.run(out, 3, log=lambda m: None, spawn=_fake_spawn_factory(out, {}, seen2))
    assert seen2 == [(1, "cold_purge")]


def test_run_raises_stop_when_child_reports_foreign_server(tmp_path, monkeypatch):
    monkeypatch.setattr(er, "kill_our_servers", lambda: [])
    out = tmp_path / "o.jsonl"
    seen = []
    with pytest.raises(RuntimeError, match="STOP:"):
        er.run(out, 3, log=lambda m: None, spawn=_fake_spawn_factory(out, {0: "NOT_RUN"}, seen))
    assert seen == [(0, "warm_read")]


def test_main_parent_advances_queue_exactly_once_and_child_never(tmp_path, monkeypatch):
    import types
    calls = []
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=lambda note: calls.append(note)))
    monkeypatch.setattr(er, "run", lambda out, reps, deadline_h=3.0: None)
    er.main(["--out", str(tmp_path / "o.jsonl")])
    assert calls == ["completed"]
    monkeypatch.setattr(er, "run_single_rep", lambda rep, arm, out: None)
    er.main(["--out", str(tmp_path / "o.jsonl"), "--single-rep", "3"])
    assert calls == ["completed"]

    def boom(out, reps, deadline_h=3.0):
        raise RuntimeError("STOP: foreign server")
    monkeypatch.setattr(er, "run", boom)
    er.main(["--out", str(tmp_path / "o.jsonl")])
    assert len(calls) == 2 and calls[1].startswith("stopped:") and "STOP" in calls[1]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows token APIs")
def test_purge_dry_run_never_raises_and_never_purges(monkeypatch):
    """dry_run stops after the privilege step whether or not this account holds it (a non-elevated test account
    does not); the counters are faked so no PowerShell is spawned."""
    monkeypatch.setattr(er, "read_counters", lambda include_gpu=True, ps_fn=None: er.parse_counter_samples(SAMPLES))
    r = er.purge_standby_list(dry_run=True)
    assert r["ok"] is False and r["ntstatus"] is None
    assert r["standby_before_bytes"] == er.standby_total(er.parse_counter_samples(SAMPLES))
    assert r.get("dry_run") is True or r["privilege_ok"] is False
