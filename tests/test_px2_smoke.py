"""Dry runs and unit tests for the PX2 power-versus-bandwidth co-runner phase (evo-x2). Uses the real
tests/stub_lab.py fixtures (StubLab/StubModelInfo/StubServer) and the same mocking shape as
tests/test_a70_p70_smoke.py's phase_p70 dry run: no real server, no real hog, no real affinity call.

Topology is injected as a recorded win_cpu_topology.read_cache_groups() document rather than read from the machine, so
these tests describe evo-x2's 2 x 8 x 2 layout regardless of what CPU actually runs them.
"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bw_hog  # noqa: E402
import t2s_night2 as n2  # noqa: E402
from stub_lab import StubLab, StubModelInfo, StubServer  # noqa: E402


def _l2(cpus):
    return {"level": 2, "logical_cpus": list(cpus), "cache_size_kb": 1024, "type": 0, "associativity": 16,
            "line_size": 64, "group_count": 1}


def _l3(cpus, kb=32768):
    return {"level": 3, "logical_cpus": list(cpus), "cache_size_kb": kb, "type": 0, "associativity": 16,
            "line_size": 64, "group_count": 1}


def x2_topo_doc():
    """evo-x2 as win_cpu_topology.py really reports it: 2 CCDs, logical 0-15 on CCD0 and 16-31 on CCD1, 32 MiB L3
    each, and a private 1 MiB L2 per physical core shared by that core's two SMT threads."""
    return {"l2_groups": [_l2((i, i + 1)) for i in range(0, 32, 2)],
            "l3_groups": [_l3(range(0, 16)), _l3(range(16, 32))],
            "buffer_bytes": 4096}


def t2s_topo_doc():
    """evo-t2s: P-cores 0-7 with a private L2 per core, then two 4-wide shared-L2 E-core clusters and one LP-E
    cluster, all under a single L3. A shared-L2 cluster is exactly the case px2_topology() must refuse."""
    return {"l2_groups": [_l2((0, 1)), _l2((2, 3)), _l2((4, 5)), _l2((6, 7)),
                          _l2((8, 9, 10, 11)), _l2((12, 13, 14, 15))],
            "l3_groups": [_l3(range(0, 16), kb=18432)], "buffer_bytes": 2048}


def laptop_topo_doc():
    """The single-CCD development laptop, as really measured while writing PX2: 8 physical cores, one 16 MiB L3."""
    return {"l2_groups": [_l2((i, i + 1)) for i in range(0, 16, 2)],
            "l3_groups": [_l3(range(0, 16), kb=16384)], "buffer_bytes": 1024}


class _FakeHog:
    def __init__(self):
        self.pid = 9999


# ------------------------------------------------------------------ topology
def test_px2_topology_reads_ccds_and_physical_cores_from_real_cache_groups():
    topo = n2.px2_topology(x2_topo_doc())
    assert topo["n_ccd"] == 2
    assert topo["n_physical"] == 16
    assert topo["n_logical"] == 32
    assert topo["l3_kb"] == [32768, 32768]
    assert topo["l2_kb"] == [1024]
    # CCD0 holds physical cores made of logical 0-15, CCD1 of 16-31, each core being its own SMT pair.
    assert topo["ccds"][0][0] == [0, 1]
    assert topo["ccds"][1][0] == [16, 17]
    assert all(len(core) == 2 for ccd in topo["ccds"] for core in ccd)


def test_px2_topology_refuses_a_shared_l2_cluster_part():
    """evo-t2s's E-core clusters share one L2 across 4 logical CPUs, so a physical core cannot be read off L2 there.
    Refusing is the point: a guess would put hog threads on SMT siblings of the server's own cores."""
    try:
        n2.px2_topology(t2s_topo_doc())
        assert False, "expected a RuntimeError for a shared-L2 cluster part"
    except RuntimeError as e:
        assert "shared-L2" in str(e)


def test_px2_topology_surfaces_the_readers_own_error():
    try:
        n2.px2_topology({"error": "GetLogicalProcessorInformationEx failed, GetLastError=87"})
        assert False, "expected a RuntimeError"
    except RuntimeError as e:
        assert "GetLastError=87" in str(e)


def test_px2_condition_cpus_needs_two_ccds():
    topo = n2.px2_topology(laptop_topo_doc())
    assert topo["n_ccd"] == 1
    try:
        n2.px2_condition_cpus(topo)
        assert False, "expected a RuntimeError on a single-CCD part"
    except RuntimeError as e:
        assert "2 CCDs" in str(e)


# ------------------------------------------------------------------ core sets
def test_px2_server_cpus_are_both_smt_threads_of_the_first_two_ccd0_cores():
    assert n2.px2_server_cpus(n2.px2_topology(x2_topo_doc())) == [0, 1, 2, 3]


def test_px2_condition_cpus_sizes_and_placement():
    topo = n2.px2_topology(x2_topo_doc())
    srv = n2.px2_server_cpus(topo)
    cond = n2.px2_condition_cpus(topo, srv)

    assert cond["N0"] == [] and cond["N1"] == []
    assert len(cond["S2"]) == 2 and len(cond["S4"]) == 4 and len(cond["S8"]) == 8
    assert len(cond["S8x"]) == 8
    assert len(cond["S14"]) == 14  # CCD1's 8 free cores + CCD0's 6 free cores (2 are the server's)
    assert len(cond["S28"]) == 28  # every free logical CPU, both SMT threads
    # S2/S4/S8 live entirely on the hog CCD (logical 16-31); S8x straddles both.
    assert all(c >= 16 for c in cond["S2"] + cond["S4"] + cond["S8"])
    assert sum(1 for c in cond["S8x"] if c >= 16) == 4 and sum(1 for c in cond["S8x"] if c < 16) == 4
    # B4 is the SAME four cores as S4: that matched-core-count pair is the power-vs-bandwidth contrast.
    assert cond["B4"] == cond["S4"]


def test_px2_conditions_never_touch_the_servers_cores_or_their_smt_siblings():
    topo = n2.px2_topology(x2_topo_doc())
    srv = n2.px2_server_cpus(topo)
    cond = n2.px2_condition_cpus(topo, srv)
    siblings = {l for core in topo["ccds"][0] for l in core if any(s in core for s in srv)}
    assert siblings == {0, 1, 2, 3}
    for name, cpus in cond.items():
        assert not (set(cpus) & siblings), f"{name} lands on the server's cores or their SMT siblings: {cpus}"


def test_px2_physical_conditions_load_one_thread_per_core_only_s28_uses_smt():
    topo = n2.px2_topology(x2_topo_doc())
    cond = n2.px2_condition_cpus(topo, n2.px2_server_cpus(topo))
    pair_of = {l: tuple(core) for ccd in topo["ccds"] for core in ccd for l in core}
    for name in ("S2", "S4", "S8", "S8x", "S14", "B4"):
        used = cond[name]
        assert len({pair_of[c] for c in used}) == len(used), f"{name} uses two SMT siblings of one core"
    assert len({pair_of[c] for c in cond["S28"]}) == 14  # both siblings of all 14 free cores


def test_px2_condition_cpus_raises_rather_than_shrinking_an_undersized_condition():
    """A 2-CCD part with only 4 physical cores per CCD cannot fill S8; a quietly undersized S8 would look real."""
    doc = {"l2_groups": [_l2((i, i + 1)) for i in range(0, 16, 2)],
           "l3_groups": [_l3(range(0, 8)), _l3(range(8, 16))], "buffer_bytes": 512}
    topo = n2.px2_topology(doc)
    try:
        n2.px2_condition_cpus(topo)
        assert False, "expected a RuntimeError for an unfillable S8"
    except RuntimeError as e:
        assert "S8" in str(e) and "free physical cores" in str(e)


def test_px2_mask_is_a_bitmask_over_logical_cpus():
    assert n2.px2_mask([0, 1, 2, 3]) == 0xF
    assert n2.px2_mask([16, 18, 20, 22]) == 0x00550000
    assert n2.px2_mask([]) == 0


# ------------------------------------------------------------------ order / drift
def test_px2_order_keeps_n0_first_and_n1_last_and_is_reproducible_from_the_logged_seed():
    order, seed = n2.px2_order("qwen3-8b")
    assert order[0] == "N0" and order[-1] == "N1"
    assert sorted(order[1:-1]) == sorted(n2.PX2_RANDOMISED)
    assert n2.px2_order("qwen3-8b") == (order, seed)  # same seed -> same order
    import random
    replay = list(n2.PX2_RANDOMISED)
    random.Random(seed).shuffle(replay)
    assert ["N0"] + replay + ["N1"] == order  # the logged seed alone reproduces the order


def test_px2_order_differs_between_models():
    orders = {m: tuple(n2.px2_order(m)[0]) for m in n2.PX2_MODELS}
    assert len(set(orders.values())) > 1


def test_px2_drift_flags_and_clears():
    assert n2.px2_drift({"N0": [1.0, 1.0], "N1": [1.01, 1.01]})["drift_flag"] is False
    d = n2.px2_drift({"N0": [1.0, 1.0], "N1": [1.2, 1.2]})
    assert d["drift_flag"] is True and abs(d["drift_pct"] - 20.0) < 1e-9
    assert n2.px2_drift({"N0": [1.0], "N1": [0.9]})["drift_flag"] is True  # drift in either direction
    assert n2.px2_drift({"N0": [], "N1": [1.0]})["drift_flag"] is None


# ------------------------------------------------------------------ dry run
def _dry_run(lab, topo_doc=None, util=None):
    util = util or {"per_cpu_pct": {}, "min_pct": 99.0, "n_read": 32, "all_at_95": True}
    started = []

    def fake_start_hog(lab_, item, cpus, kind):
        started.append((item, tuple(cpus), kind))
        return _FakeHog(), f"report_{item}.txt", f"aff_{item}.json", n2.px2_mask(cpus)

    with mock.patch.object(n2.L, "Server", StubServer), \
         mock.patch.object(n2.wct, "read_cache_groups", lambda: topo_doc or x2_topo_doc()), \
         mock.patch.object(n2, "px2_start_hog", fake_start_hog), \
         mock.patch.object(n2, "px2_pin_server", lambda pid, cpus: {"pinned": True, "mask": hex(n2.px2_mask(cpus)),
                                                                   "read_back": hex(n2.px2_mask(cpus)),
                                                                   "server_cpus": list(cpus)}), \
         mock.patch.object(n2, "px2_per_core_util", lambda cpus: util), \
         mock.patch.object(n2.L.m3, "read_ips", lambda report: 120.0), \
         mock.patch.object(n2.L.m3, "kill_tree", lambda pid: None), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_px2(lab)
    return started


def test_phase_px2_dry_run_all_nine_conditions_per_model():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    started = _dry_run(lab)

    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    assert {r["co_runner"] for r in call_rows} == set(n2.PX2_CONDITIONS)
    assert len(n2.PX2_CONDITIONS) == 9
    # 9 conditions x (1 warm-up + 5 measured) for a non-cut model
    assert len(call_rows) == 9 * (1 + n2.PX2_N_MEASURED)
    assert {f"PX2_qwen3-8b_{c}" for c in n2.PX2_CONDITIONS} <= lab.done
    # a hog was started for every non-baseline condition and for none of the baselines
    hog_items = {item for item, _, _ in started if not item.startswith("PX2_bw_calibration")}
    assert hog_items == {f"PX2_qwen3-8b_{c}" for c in n2.PX2_RANDOMISED}


def test_phase_px2_dry_run_b4_is_the_bandwidth_hog_and_s4_the_spin_hog_on_the_same_cores():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    started = _dry_run(lab)
    by_item = {item: (cpus, kind) for item, cpus, kind in started}
    s4_cpus, s4_kind = by_item["PX2_qwen3-8b_S4"]
    b4_cpus, b4_kind = by_item["PX2_qwen3-8b_B4"]
    assert s4_kind == "spin" and b4_kind == "bw"
    assert s4_cpus == b4_cpus, "the power-vs-bandwidth contrast must be at a matched core set"


def test_phase_px2_dry_run_records_topology_and_the_bandwidth_calibration():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    _dry_run(lab)
    topo_rec = [r for r in lab.rows if r.get("record") == "px2_topology"]
    assert len(topo_rec) == 1
    assert topo_rec[0]["server_cpus"] == [0, 1, 2, 3]
    assert topo_rec[0]["condition_masks"]["S4"] == hex(n2.px2_mask(topo_rec[0]["condition_cpus"]["S4"]))
    assert topo_rec[0]["hog_kind_by_cond"]["B4"] == "bw"
    cal = [r for r in lab.rows if r.get("record") == "px2_bw_calibration"]
    assert len(cal) == 1 and cal[0]["gbps_solo"] == 120.0
    assert "PX2_bw_calibration" in lab.done


def test_phase_px2_dry_run_rows_carry_the_condition_seed_and_saturation_fields():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    _dry_run(lab)
    s4 = [r for r in lab.rows if r.get("kind") == "call" and r.get("co_runner") == "S4"]
    assert s4
    r = s4[0]
    assert r["hog_kind"] == "spin"
    assert r["cond"] == "S4"
    assert r["server_pinned"] is True
    assert r["server_affinity_mask"] == "0xf"
    assert r["hog_cpus_all_at_95"] is True
    assert r["hog_min_core_pct"] == 99.0
    assert r["px2_seed"] == n2.px2_order("qwen3-8b")[1]
    assert r["bw_gbps_solo"] == 120.0
    # the pre-condition gate is recorded under px2_gate_*, never under do_call's reserved thermal_* names
    assert "px2_gate_released_by" in r and "px2_gate_wait_s" in r
    n0 = [x for x in lab.rows if x.get("kind") == "call" and x.get("co_runner") == "N0"][0]
    assert n0["hog_kind"] == "none" and n0["hog_cpus"] == [] and n0["cpu_mask"] is None


def test_phase_px2_dry_run_emits_a_drift_check_per_model():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    _dry_run(lab)
    s = [r for r in lab.rows if r.get("record") == "px2_model_summary"]
    assert len(s) == 1
    # StubServer.chat returns a constant TTFT, so N1 matches N0 exactly and drift must not be flagged
    assert s[0]["drift_flag"] is False
    assert set(s[0]["ttft_median_by_cond"]) == set(n2.PX2_CONDITIONS)


def test_phase_px2_dry_run_applies_the_cut_rule_to_the_70b():
    lab = StubLab(models={"llama-3.3-70b": StubModelInfo("llama-3.3-70b")})
    _dry_run(lab)
    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    assert len(call_rows) == 9 * (1 + n2.PX2_N_MEASURED_CUT)
    assert all(r["n_reduced_cut_rule"] is True for r in call_rows)
    assert all(r["n_measured_planned"] == n2.PX2_N_MEASURED_CUT for r in call_rows)


def test_phase_px2_dry_run_resumes_past_conditions_already_marked_done():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    lab.item_done("PX2_bw_calibration")
    for c in ("N0", "S2", "S4"):
        lab.item_done(f"PX2_qwen3-8b_{c}")
    _dry_run(lab)
    done_conds = {r["co_runner"] for r in lab.rows if r.get("kind") == "call"}
    assert not (done_conds & {"N0", "S2", "S4"})
    assert not [r for r in lab.rows if r.get("record") == "px2_bw_calibration"]


def test_phase_px2_disables_itself_with_a_reason_on_a_shared_l2_part():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    with mock.patch.object(n2.wct, "read_cache_groups", lambda: t2s_topo_doc()):
        n2.phase_px2(lab)
    rec = [r for r in lab.rows if r.get("record") == "px2_disabled"]
    assert len(rec) == 1 and "shared-L2" in rec[0]["reason"]
    assert not [r for r in lab.rows if r.get("kind") == "call"]


def test_phase_px2_disables_itself_on_a_single_ccd_part():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    with mock.patch.object(n2.wct, "read_cache_groups", lambda: laptop_topo_doc()):
        n2.phase_px2(lab)
    rec = [r for r in lab.rows if r.get("record") == "px2_disabled"]
    assert len(rec) == 1 and "2 CCDs" in rec[0]["reason"]


def test_phase_px2_skips_a_model_that_load_models_did_not_populate():
    lab = StubLab(models={})
    started = _dry_run(lab)
    assert not [r for r in lab.rows if r.get("kind") == "call"]
    assert [r for r in lab.rows if r.get("record") == "px2_topology"]
    assert [item for item, _, _ in started] == ["PX2_bw_calibration"]


def test_phase_px2_kills_the_hog_even_when_a_condition_raises():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    killed = []

    def boom(*a, **kw):
        raise RuntimeError("call blew up")

    with mock.patch.object(n2.L, "Server", StubServer), \
         mock.patch.object(n2.wct, "read_cache_groups", x2_topo_doc), \
         mock.patch.object(n2, "px2_start_hog", lambda l, i, c, k: (_FakeHog(), "r", "a", n2.px2_mask(c))), \
         mock.patch.object(n2, "px2_pin_server", lambda pid, cpus: {"pinned": True}), \
         mock.patch.object(n2, "px2_per_core_util", lambda cpus: {"all_at_95": True, "min_pct": 99.0}), \
         mock.patch.object(n2, "measured_with_extra", boom), \
         mock.patch.object(n2.L.m3, "read_ips", lambda report: 1.0), \
         mock.patch.object(n2.L.m3, "kill_tree", lambda pid: killed.append(pid)), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        try:
            n2.phase_px2(lab)
            assert False, "expected the condition's exception to propagate to main()'s per-phase handler"
        except RuntimeError as e:
            assert "call blew up" in str(e)
    assert killed, "the hog must be killed in the finally block before the exception leaves the phase"


# ------------------------------------------------------------------ registration / estimate
def test_px2_is_registered_but_not_in_the_default_phase_order():
    assert n2.PHASE_FN["px2"] is n2.phase_px2
    assert "px2" in n2.PRIO
    assert "px2" not in n2.PHASE_ORDER.split(",")
    assert "px2" in n2.SMOKE_GATED_PHASES  # a brand-new phase must pass the 1-item live smoke first


def test_px2_models_all_have_a_real_model_files_entry():
    """llama-3.3-70b included: ov.MODEL_FILES really does define it, so phase_px2 must not invent a 'missing model'
    record for it -- a miss can only mean downloads.jsonl has no verified hash on this machine yet."""
    import t2s_overnight as ov
    for mid in n2.PX2_MODELS:
        assert mid in ov.MODEL_FILES, mid
    assert ov.MODEL_FILES["llama-3.3-70b"][0] == "Llama-3.3-70B-Instruct-Q4_K_M.gguf"


def test_estimate_hours_includes_px2_and_scales_with_the_cut_rule():
    lab = StubLab()
    lab.table = {}
    base = n2.estimate_hours(lab, {"load_s": {}, "call_e2e_s": {}, "gate_wait_s": 20.0})
    assert base["px2"] > 0
    with mock.patch.object(n2, "PX2_CUT_MODELS", tuple(n2.PX2_MODELS)):
        cut = n2.estimate_hours(lab, {"load_s": {}, "call_e2e_s": {}, "gate_wait_s": 20.0})
    assert cut["px2"] < base["px2"]


# ------------------------------------------------------------------ bw_hog
def test_bw_hog_self_test_passes():
    assert bw_hog.self_test() == 0


def test_bw_hog_uses_the_stream_triad_convention():
    # 3 arrays x 1e9 bytes x 2 iterations / 2 s = 3 GB/s
    assert abs(bw_hog.gbps(1_000_000_000, 2, 2.0) - 3.0) < 1e-9
    assert bw_hog.gbps(1_000_000_000, 2, 0) is None  # honest about having no measurement yet
    assert bw_hog.ARRAYS_PER_ITER == 3


def test_bw_hog_array_sizing_far_exceeds_one_strix_halo_l3_slice():
    """320 MiB per array against a 32 MiB per-CCD L3 slice -- exactly 10x, and three such arrays per worker, so the
    working set cannot be cached. That is the whole point of the bandwidth arm."""
    l3_slice = 32 * 2 ** 20
    array_bytes = bw_hog.elems_for_mib(n2.PX2_BW_ARRAY_MIB) * bw_hog.BYTES_PER_ELEM
    assert array_bytes == 10 * l3_slice
    assert bw_hog.ARRAYS_PER_ITER * array_bytes >= 30 * l3_slice
    assert bw_hog.elems_for_mib(0) == 1  # never a zero-length array
