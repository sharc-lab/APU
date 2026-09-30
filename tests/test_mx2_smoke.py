"""Dry runs (stub lab, no real server, no vulkaninfo, no registry) for phase_mx2's two-line memory budget: the
baseline readers, the FITS/SILENT_SPILL/HARD_FAIL/HANG regime labelling, Mx2HeapProber's two criteria over one shared
raw-result cache, and the phase's own orchestration. Same shape as tests/test_a70_p70_smoke.py: am.Prober.find/probe
and am.start_and_record are patched so the real (already-relied-on) exponential-search/bisection algorithm is not what
these tests exercise -- phase_mx2's own orchestration is.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2s_night2 as n2  # noqa: E402
from stub_lab import StubLab, StubModelInfo, StubServer  # noqa: E402

# One DEVICE_LOCAL heap plus one host-visible heap, in vulkaninfo's own layout. Sizes are small and round so the
# derived lines are obvious: 1024 MiB device-local, 3072 MiB total.
VK_TEXT = """
VkPhysicalDeviceMemoryProperties:
memoryHeaps[0]:
        size   = 2147483648 (0x80000000)
        budget = 2147483648 (0x80000000)
        usage  = 0 (0x00000000)
        flags:
                None
memoryHeaps[1]:
        size   = 1073741824 (0x40000000)
        budget = 1073741824 (0x40000000)
        usage  = 0 (0x00000000)
        flags:
                MEMORY_HEAP_DEVICE_LOCAL_BIT
"""

LINES = {"device_local_mib": 1024.0, "vulkan_total_mib": 3072.0}


def _mi70():
    mi = StubModelInfo("llama-3.3-70b")
    mi.kv_bpt_meta = 2.0  # bytes/token; tiny so am.STEP-rounded bisection math stays sane in a test
    mi.yarn_factor = 1
    return mi


def _mi32():
    mi = StubModelInfo("qwen3-32b", max_ctx_native=32768)
    mi.kv_bpt_meta = 2.0
    mi.yarn_factor = 4
    return mi


def _fake_run(stdout, returncode=0):
    return lambda *a, **kw: SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def _info(ok=True, exit_code=None, log=None, error=None):
    return {"ok": ok, "error": error if error is not None else (None if ok else "boom"), "exit_code": exit_code,
            "log": log or {}, "load_s": 3.0, "pid": 4242, "build": "b10970-stub", "t_start": 0.0, "t_end": 3.0}


# ---------------------------------------------------------------- baseline readers
def test_parse_heaps_reads_sizes_and_device_local_flag():
    heaps = n2.mx2_parse_heaps(VK_TEXT)
    assert [h["index"] for h in heaps] == [0, 1]
    assert heaps[0]["size_mib"] == 2048.0 and heaps[0]["device_local"] is False
    assert heaps[1]["size_mib"] == 1024.0 and heaps[1]["device_local"] is True


def test_lines_from_heaps_sums_device_local_and_total():
    assert n2.mx2_lines_from_heaps(n2.mx2_parse_heaps(VK_TEXT)) == LINES


def test_lines_from_heaps_device_local_is_none_when_no_heap_advertises_it():
    """None, not 0 -- a zero line would read as a real budget of zero and label every context a spill."""
    heaps = [{"index": 0, "size_mib": 100.0, "budget_mib": 100.0, "usage_mib": 0.0, "flags": "None",
              "device_local": False}]
    assert n2.mx2_lines_from_heaps(heaps) == {"device_local_mib": None, "vulkan_total_mib": 100.0}


def test_lines_from_heaps_empty_is_all_none():
    assert n2.mx2_lines_from_heaps([]) == {"device_local_mib": None, "vulkan_total_mib": None}


def test_read_list_devices_parses_total_and_free():
    txt = "load_backend: loaded Vulkan backend\nVulkan0: Radeon 8060S (114326 MiB, 108610 MiB free)\n"
    got = n2.mx2_read_list_devices(run=_fake_run(txt))
    assert got["total_mib"] == 114326.0 and got["free_mib"] == 108610.0


def test_read_list_devices_survives_a_failing_binary():
    def boom(*a, **kw):
        raise OSError("no such file")
    assert n2.mx2_read_list_devices(run=boom) is None


def test_read_registry_dedicated_parses_qw_memory_size():
    got = n2.mx2_read_registry_dedicated(ps_fn=lambda cmd, t: "qwMemorySize=68543315968\n")
    assert got["qw_memory_size_bytes"] == 68543315968
    assert round(got["qw_memory_size_mib"]) == 65368


def test_read_registry_dedicated_handles_unavailable_key():
    got = n2.mx2_read_registry_dedicated(ps_fn=lambda cmd, t: "qwMemorySize=unavailable\n")
    assert got["qw_memory_size_bytes"] is None and got["qw_memory_size_mib"] is None


def test_record_baseline_measures_lines_and_records_caveat():
    lab = StubLab()
    lines = n2.mx2_record_baseline(lab, run=_fake_run(VK_TEXT), ps_fn=lambda cmd, t: "qwMemorySize=1\n")
    assert lines == LINES
    rec = [r for r in lab.rows if r.get("record") == "mx2_baseline"][0]
    assert rec["lines_measured"] is True
    assert rec["fallback_lines"] is None
    assert len(rec["heaps"]) == 2
    assert "not fully explained" in rec["baseline_caveat"] or "not explained" in rec["baseline_caveat"]
    assert "quality out of scope" in rec["quality_scope"]


def test_record_baseline_falls_back_and_flags_it_when_vulkaninfo_is_unreadable():
    lab = StubLab()

    def boom(*a, **kw):
        raise OSError("vulkaninfo not found")
    lines = n2.mx2_record_baseline(lab, run=boom, ps_fn=lambda cmd, t: "")
    assert lines == n2.MX2_FALLBACK_LINES
    rec = [r for r in lab.rows if r.get("record") == "mx2_baseline"][0]
    assert rec["lines_measured"] is False
    assert rec["fallback_lines"] == n2.MX2_FALLBACK_LINES


# ---------------------------------------------------------------- regime labelling
def test_classify_fits_when_logged_memory_stays_inside_device_local_line():
    srv = StubServer(StubLab(), _mi70(), 4096)
    info = _info(ok=True, log={"model_buffer_mib": 200.0, "kv_buffer_mib": 100.0, "compute_buffer_mib": 50.0})
    regime, cap, ev = n2.mx2_classify(info, srv, {"projected_mib": 400.0}, LINES)
    assert regime == "FITS"
    assert ev["spill_evidence"] == []


def test_classify_silent_spill_when_logged_buffers_cross_the_device_local_line():
    srv = StubServer(StubLab(), _mi70(), 4096)
    info = _info(ok=True, log={"model_buffer_mib": 800.0, "kv_buffer_mib": 400.0, "compute_buffer_mib": 50.0})
    regime, cap, ev = n2.mx2_classify(info, srv, {"projected_mib": 1250.0}, LINES)
    assert regime == "SILENT_SPILL"
    assert "logged_buffers_over_device_local_line" in ev["spill_evidence"]
    assert "llama_cpp_projection_over_device_local_line" in ev["spill_evidence"]


def test_classify_silent_spill_on_projection_alone():
    """llama.cpp's own projection over the line counts even when the logged buffers do not, since the projection is
    what the fit policy acted on."""
    srv = StubServer(StubLab(), _mi70(), 4096)
    info = _info(ok=True, log={"model_buffer_mib": 100.0})
    regime, cap, ev = n2.mx2_classify(info, srv, {"projected_mib": 2000.0}, LINES)
    assert regime == "SILENT_SPILL"
    assert ev["spill_evidence"] == ["llama_cpp_projection_over_device_local_line"]


def test_classify_hard_fail_when_server_exits_with_a_code():
    srv = StubServer(StubLab(), _mi70(), 4096)
    with mock.patch.object(n2.am, "read_logs", lambda s: ""):
        regime, cap, ev = n2.mx2_classify(_info(ok=False, exit_code=1), srv, {}, LINES)
    assert regime == "HARD_FAIL"
    assert ev["exit_code"] == 1


def test_classify_hang_when_server_never_exits():
    srv = StubServer(StubLab(), _mi70(), 4096)
    with mock.patch.object(n2.am, "read_logs", lambda s: ""):
        regime, cap, ev = n2.mx2_classify(_info(ok=False, exit_code=None), srv, {}, LINES)
    assert regime == "HANG"


def test_classify_treats_a_benign_yarn_props_cap_as_a_start_not_a_failure():
    """The bug that bit the A-mech bisection and map phases twice: a YaRN /props n_ctx cap is a start. mx2_classify
    must inherit am.classify_start's handling of it rather than re-deciding."""
    srv = StubServer(StubLab(), _mi32(), 131072)
    info = _info(ok=False, exit_code=None,
                 error="guard: server does not match intended config: n_ctx 32768 != 131072")
    with mock.patch.object(n2.am, "read_logs", lambda s: "llama_server: listening on 127.0.0.1:8385"):
        regime, cap, ev = n2.mx2_classify(info, srv, {"projected_mib": 100.0}, LINES)
    assert regime == "FITS"
    assert cap == 32768


def test_classify_never_reports_spill_when_the_device_local_line_is_unknown():
    srv = StubServer(StubLab(), _mi70(), 4096)
    info = _info(ok=True, log={"model_buffer_mib": 99999.0})
    regime, cap, ev = n2.mx2_classify(info, srv, {"projected_mib": 99999.0},
                                      {"device_local_mib": None, "vulkan_total_mib": 3072.0})
    assert regime == "FITS"
    assert ev["spill_evidence"] == []


# ---------------------------------------------------------------- Mx2HeapProber
def _regime_prober(lab, mi, criterion, regime_fn, raw=None):
    pr = n2.Mx2HeapProber(lab, mi, [], "t", LINES, criterion, raw)
    pr._probe_raw = lambda n_ctx: {"ok": None, "regime": regime_fn(n_ctx), "props_cap": None, "projected_mib": 1.0,
                                   "logged_mib": 1.0, "error": None, "vk": None, "alloc_failed": None,
                                   "spill_evidence": [], "exit_code": None}
    return pr


def _ladder(n_ctx):
    if n_ctx <= 4096:
        return "FITS"
    if n_ctx <= 8192:
        return "SILENT_SPILL"
    return "HARD_FAIL"


def test_prober_rejects_an_unknown_criterion():
    try:
        n2.Mx2HeapProber(StubLab(), _mi70(), [], "t", LINES, "nonsense")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "criterion" in str(e)


def test_no_spill_criterion_treats_silent_spill_as_not_ok():
    pr = _regime_prober(StubLab(), _mi70(), "no_spill", _ladder)
    assert pr.probe(4096)["ok"] is True
    assert pr.probe(8192)["ok"] is False   # started, but spilled: not ok for line 1
    assert pr.probe(8192)["regime"] == "SILENT_SPILL"


def test_start_criterion_treats_silent_spill_as_ok():
    pr = _regime_prober(StubLab(), _mi70(), "start", _ladder)
    assert pr.probe(8192)["ok"] is True    # a spilling start is still a start: ok for line 2
    assert pr.probe(16384)["ok"] is False


def test_both_criteria_share_one_raw_cache_so_no_server_start_is_repeated():
    lab, mi, raw = StubLab(), _mi70(), {}
    starts = []

    def counting(n_ctx):
        starts.append(n_ctx)
        return _ladder(n_ctx)

    pr1 = _regime_prober(lab, mi, "no_spill", counting, raw)
    pr2 = _regime_prober(lab, mi, "start", counting, raw)
    pr1.probe(4096)
    pr1.probe(8192)
    pr2.probe(8192)   # already in the shared raw cache: must not start a server again
    pr2.probe(4096)
    assert starts == [4096, 8192]
    # ...but each prober still applies its OWN criterion to the shared result
    assert pr1.probe(8192)["ok"] is False and pr2.probe(8192)["ok"] is True


def test_prober_populates_cache_in_the_shape_boundary_record_reads():
    lab, mi = StubLab(), _mi70()
    pr = _regime_prober(lab, mi, "no_spill", _ladder)
    pr.probe(4096)
    pr.probe(8192)
    rec = n2.am.boundary_record(lab, mi, "mx2_t", [], pr, 4096, 8192, target_note="line 1")
    assert rec["last_ok_n_ctx"] == 4096 and rec["first_fail_n_ctx"] == 8192
    assert rec["projected_mib_last_ok"] == 1.0  # read straight out of pr.cache


def test_prober_find_bisects_line1_with_the_inherited_algorithm():
    """No patching of find(): the real t2s_amech exponential-search-then-bisect runs against a synthetic ladder, to
    confirm the subclass's ok/regime plumbing is what find() expects."""
    pr = _regime_prober(StubLab(), _mi70(), "no_spill", lambda n: "FITS" if n <= 20000 else "SILENT_SPILL")
    lo, hi = pr.find(n2.MX2_START_CTX, n2.MX2_START_STEP)
    assert hi - lo == n2.am.STEP
    assert lo <= 20000 < hi


def test_prober_find_bisects_line2_higher_than_line1():
    regime = lambda n: "FITS" if n <= 20000 else ("SILENT_SPILL" if n <= 50000 else "HARD_FAIL")
    raw = {}
    lo1, hi1 = _regime_prober(StubLab(), _mi70(), "no_spill", regime, raw).find(n2.MX2_START_CTX, n2.MX2_START_STEP)
    lo2, hi2 = _regime_prober(StubLab(), _mi70(), "start", regime, raw).find(hi1, n2.MX2_START_STEP)
    assert lo1 <= 20000 < hi1
    assert lo2 <= 50000 < hi2
    assert lo2 > lo1  # line 2 is strictly above line 1: that gap is the spill window


def test_real_probe_raw_emits_one_mx2_probe_row_and_caches_it():
    """The real _probe_raw (not the stub used by the criterion tests above): one row per distinct context, carrying
    both lines and the regime, and a repeat probe of the same context must not start a second server."""
    lab, mi = StubLab(), _mi70()
    pr = n2.Mx2HeapProber(lab, mi, [], "t", LINES, "no_spill")
    with mock.patch.object(n2.am, "start_and_record",
                           _fake_start_and_record(lambda n: "FITS" if n <= 4096 else "SILENT_SPILL")), \
         mock.patch.object(n2.am, "read_logs", lambda s: ""):
        a = pr.probe(4096)
        pr.probe(4096)   # cached: no second server start, no second row
        b = pr.probe(8192)
    rows = [r for r in lab.rows if r.get("record") == "mx2_probe"]
    assert [r["n_ctx"] for r in rows] == [4096, 8192]
    assert [r["regime"] for r in rows] == ["FITS", "SILENT_SPILL"]
    assert rows[0]["device_local_line_mib"] == LINES["device_local_mib"]
    assert rows[0]["vulkan_total_line_mib"] == LINES["vulkan_total_mib"]
    assert a["ok"] is True and b["ok"] is False
    assert pr.n == 2  # exactly two server starts for three probe() calls


def test_probe_rounds_context_to_the_amech_step():
    lab, mi = StubLab(), _mi70()
    pr = n2.Mx2HeapProber(lab, mi, [], "t", LINES, "no_spill")
    with mock.patch.object(n2.am, "start_and_record", _fake_start_and_record(lambda n: "FITS")), \
         mock.patch.object(n2.am, "read_logs", lambda s: ""):
        pr.probe(4100)
        pr.probe(4096)  # same context after rounding: served from the cache
    assert pr.n == 1
    assert [r["n_ctx"] for r in lab.rows if r.get("record") == "mx2_probe"] == [4096]


# ---------------------------------------------------------------- phase_mx2 dry runs
def _fake_start_and_record(regime_fn):
    """A fake am.start_and_record driving mx2_classify through info/px, so the regime is decided by the real
    classifier rather than injected. Returns a real StubServer for the arm path's tokenize/measured_sequence."""
    def _fake(lab, mi, n_ctx, tag, item_id, phase, extra_flags, ngl=99, fit=None, **kw):
        srv = StubServer(lab, mi, n_ctx)
        regime = regime_fn(n_ctx)
        if regime == "FITS":
            info = _info(ok=True, log={"model_buffer_mib": 100.0, "kv_buffer_mib": 100.0})
            px = {"projected_mib": 200.0, "vk_errors": None, "alloc_failed": None, "error_lines": None}
        elif regime == "SILENT_SPILL":
            info = _info(ok=True, log={"model_buffer_mib": 900.0, "kv_buffer_mib": 400.0})
            px = {"projected_mib": 1300.0, "vk_errors": None, "alloc_failed": None, "error_lines": None}
        elif regime == "HANG":
            info = _info(ok=False, exit_code=None)
            px = {"projected_mib": None, "vk_errors": None, "alloc_failed": None, "error_lines": []}
        else:
            info = _info(ok=False, exit_code=1)
            px = {"projected_mib": None, "vk_errors": ["vk::OutOfDeviceMemory"], "alloc_failed": [], "error_lines": []}
        return srv, info, px
    return _fake


def _patched_phase(lab, regime_fn, run=None, ps_fn=None):
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.am.L, "Server", StubServer), \
         mock.patch.object(n2.hc, "ollama_process_running", lambda: False), \
         mock.patch.object(n2.am, "read_logs", lambda s: ""), \
         mock.patch.object(n2.am, "start_and_record", _fake_start_and_record(regime_fn)), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_mx2(lab, run=run or _fake_run(VK_TEXT), ps_fn=ps_fn or (lambda cmd, t: "qwMemorySize=1\n"))


def test_phase_mx2_dry_run_finds_both_lines_and_runs_both_arms():
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    _patched_phase(lab, lambda n: "FITS" if n <= 20000 else ("SILENT_SPILL" if n <= 50000 else "HARD_FAIL"))

    res = [r for r in lab.rows if r.get("record") == "mx2_two_line_result"]
    assert len(res) == 1
    r = res[0]
    assert r["device_local_last_fit_n_ctx"] <= 20000 < r["device_local_first_spill_n_ctx"]
    assert r["vulkan_total_last_start_n_ctx"] <= 50000 < r["vulkan_total_first_fail_n_ctx"]
    assert r["spill_window_tokens"] > 0
    assert r["lines"] == LINES
    assert set(r["regime_counts"]) <= {"FITS", "SILENT_SPILL", "HARD_FAIL", "HANG"}
    assert "quality out of scope" in r["quality_scope"]
    # both boundary records, via the real am.boundary_record
    labels = {b["label"] for b in lab.rows if b.get("record") == "bisect_result"}
    assert labels == {"mx2_llama-3.3-70b_device_local", "mx2_llama-3.3-70b_vulkan_total"}
    # two arms: one below line 1, one inside the spill window
    arms = [a for a in lab.rows if a.get("record") == "mx2_arm_result"]
    assert {a["arm"] for a in arms} == {"below_line1", "in_spill_window"}
    assert all(a["started"] for a in arms)
    assert [a for a in arms if a["arm"] == "below_line1"][0]["regime"] == "FITS"
    assert [a for a in arms if a["arm"] == "in_spill_window"][0]["regime"] == "SILENT_SPILL"
    assert {"MX2_llama-3.3-70b_arm_below_line1", "MX2_llama-3.3-70b_arm_in_spill_window",
            "MX2_llama-3.3-70b_lines"} <= lab.done


def test_phase_mx2_arms_run_one_warmup_plus_three_measured_calls_and_no_q0():
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    _patched_phase(lab, lambda n: "FITS" if n <= 20000 else ("SILENT_SPILL" if n <= 50000 else "HARD_FAIL"))
    calls = [c for c in lab.rows if c.get("kind") == "call"]
    measured = [c for c in calls if not c.get("warmup")]
    assert len(measured) == 2 * n2.MX2_ARM_CALLS
    assert len(calls) - len(measured) == 2  # exactly one warm-up per arm
    assert not [r for r in lab.rows if str(r.get("record", "")).startswith("mx2_q0")]


def test_phase_mx2_records_hang_distinctly_from_hard_fail():
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    _patched_phase(lab, lambda n: "FITS" if n <= 20000 else ("SILENT_SPILL" if n <= 50000 else "HANG"))
    r = [x for x in lab.rows if x.get("record") == "mx2_two_line_result"][0]
    assert "HANG" in r["regime_counts"]
    assert "HARD_FAIL" not in r["regime_counts"]


def test_phase_mx2_flags_when_there_is_no_spill_window():
    """A model whose spill onset and allocation failure are within one probe step has no context that both starts and
    spills, so there is nothing to measure the cost of -- that must be recorded, not silently produce one arm."""
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    _patched_phase(lab, lambda n: "FITS" if n <= 20000 else "HARD_FAIL")
    assert [r for r in lab.rows if r.get("record") == "mx2_no_spill_window"]
    arms = [a for a in lab.rows if a.get("record") == "mx2_arm_result"]
    assert {a["arm"] for a in arms} == {"below_line1"}
    r = [x for x in lab.rows if x.get("record") == "mx2_two_line_result"][0]
    assert r["spill_window_tokens"] == 0


def test_phase_mx2_records_no_fitting_context_when_nothing_fits_line1():
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    _patched_phase(lab, lambda n: "SILENT_SPILL")
    assert [r for r in lab.rows if r.get("record") == "mx2_no_fitting_context"]
    assert not [r for r in lab.rows if r.get("record") == "mx2_two_line_result"]
    assert "MX2_llama-3.3-70b_lines" in lab.done  # still marked done, so a resume does not retry it forever


def test_phase_mx2_skips_models_that_are_not_loaded():
    lab = StubLab(models={})
    _patched_phase(lab, lambda n: "FITS")
    skipped = {r["model_id"] for r in lab.rows if r.get("record") == "mx2_skipped"}
    assert skipped == set(n2.MX2_MODELS)
    assert not [r for r in lab.rows if r.get("record") == "mx2_two_line_result"]


def test_phase_mx2_resumes_past_a_completed_model():
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    lab.item_done("MX2_llama-3.3-70b_lines")
    _patched_phase(lab, lambda n: "FITS" if n <= 20000 else "HARD_FAIL")
    assert not [r for r in lab.rows if r.get("record") == "mx2_probe"]
    assert [r for r in lab.rows if r.get("record") == "mx2_baseline"]  # baseline is still re-recorded per run


def test_phase_mx2_uses_yarn_flags_for_qwen3_32b_and_none_for_the_70b():
    """The 32B's real ov.MODEL_FILES entry is native 32768 with yarn_factor 4, so it is in ov.YARN_MODELS and must be
    probed with the same YaRN flags the rest of the harness uses; the 70B is native 131072 with yarn_factor 1 and
    must be probed with no rope flags at all."""
    assert ov_native("qwen3-32b") == (32768, 4)
    assert ov_native("llama-3.3-70b") == (131072, 1)
    assert "qwen3-32b" in n2.ov.YARN_MODELS
    assert "llama-3.3-70b" not in n2.ov.YARN_MODELS
    lab = StubLab(models={"llama-3.3-70b": _mi70(), "qwen3-32b": _mi32()})
    seen = {}

    def recording(lab_, mi, n_ctx, tag, item_id, phase, extra_flags, ngl=99, fit=None, **kw):
        seen.setdefault(mi.model_id, set()).add(tuple(extra_flags or ()))
        return _fake_start_and_record(lambda n: "FITS" if n <= 20000 else "HARD_FAIL")(
            lab_, mi, n_ctx, tag, item_id, phase, extra_flags, ngl=ngl, fit=fit, **kw)

    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.am.L, "Server", StubServer), \
         mock.patch.object(n2.hc, "ollama_process_running", lambda: False), \
         mock.patch.object(n2.am, "read_logs", lambda s: ""), \
         mock.patch.object(n2.am, "start_and_record", recording), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_mx2(lab, run=_fake_run(VK_TEXT), ps_fn=lambda cmd, t: "")

    assert seen["llama-3.3-70b"] == {()}
    assert seen["qwen3-32b"] == {tuple(n2.am.YARN)}


def ov_native(model_id):
    """(max_ctx_native, yarn_factor) straight out of the real ov.MODEL_FILES, so these tests fail loudly if that
    table changes under them rather than carrying a stale copy of the numbers."""
    fn, hyb, mx, yf = n2.ov.MODEL_FILES[model_id]
    return mx, yf


def test_mx2_model_files_entries_are_the_real_ones():
    fn70 = n2.ov.MODEL_FILES["llama-3.3-70b"][0]
    assert fn70 == "Llama-3.3-70B-Instruct-Q4_K_M.gguf"
    assert n2.ov.MODEL_FILES["qwen3-32b"][0] == "Qwen3-32B-Q4_K_M.gguf"
    assert n2.MX2_MODELS == ["llama-3.3-70b", "qwen3-32b"]
    # MX2 reuses A70_MODEL's entry rather than carrying its own filename constant
    assert n2.A70_MODEL in n2.MX2_MODELS


def test_phase_mx2_aborts_if_ollama_running(monkeypatch):
    lab = StubLab(models={"llama-3.3-70b": _mi70()})
    monkeypatch.setattr(n2.hc, "ollama_process_running", lambda: True)
    try:
        n2.phase_mx2(lab, run=_fake_run(VK_TEXT), ps_fn=lambda cmd, t: "")
        assert False, "expected SmokeFailure"
    except n2.SmokeFailure as e:
        assert "STOP" in str(e) and "ollama" in str(e)
    assert lab.rows == []  # aborted before even the baseline was read


# ---------------------------------------------------------------- registration
def test_mx2_is_registered_in_prio_and_phase_fn():
    assert "mx2" in n2.PRIO
    assert n2.PHASE_FN["mx2"] is n2.phase_mx2


def test_mx2_is_smoke_gated_like_a70_and_p70():
    """A70 and P70 are smoke-gated, and MX2 is the same shape of phase (expensive, big model, first run on a machine),
    so it must be gated too."""
    assert {"a70", "p70"} <= n2.SMOKE_GATED_PHASES
    assert "mx2" in n2.SMOKE_GATED_PHASES


def test_estimate_hours_includes_mx2_only_for_models_with_timing():
    lab = StubLab()
    lab.table = {"qwen3-32b": {"ok": True, "load_s": 20.0}}
    est = n2.estimate_hours(lab, {"load_s": {"qwen3-32b": 19.9}, "call_e2e_s": {"qwen3-32b": 111.7},
                                  "gate_wait_s": 90.0})
    assert est["mx2"] > 0
    est_none = n2.estimate_hours(StubLab(), {"load_s": {}, "call_e2e_s": {}, "gate_wait_s": 90.0})
    assert est_none["mx2"] == 0.0  # no model on this machine has timing yet: nothing to estimate from
