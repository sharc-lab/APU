"""Dry runs for PX2I, the evo-t2s bandwidth-hog phase in harness/t2s_night2.py (docs/T2S_WEEK_PLAN.md step 4). Same
stub lab and mocking shape as tests/test_px2_smoke.py: no real server, no real hog, no affinity call."""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2s_night2 as n2  # noqa: E402
from stub_lab import StubLab, StubModelInfo, StubServer  # noqa: E402


class _FakeHog:
    pid = 4242


def _dry_run(lab):
    started, pinned = [], []

    def fake_start_hog(lab_, item, cpus, kind):
        started.append((item, tuple(cpus), kind))
        return _FakeHog(), f"report_{item}.txt", f"aff_{item}.json", n2.px2_mask(cpus)

    def fake_pin(pid, cpus):
        pinned.append(list(cpus))
        return {"pinned": True, "server_cpus": list(cpus)}

    with mock.patch.object(n2.L, "Server", StubServer), \
         mock.patch.object(n2, "px2_start_hog", fake_start_hog), \
         mock.patch.object(n2, "px2_pin_server", fake_pin), \
         mock.patch.object(n2, "px2_per_core_util", lambda cpus: {"all_at_95": True, "min_pct": 99.0}), \
         mock.patch.object(n2.L.m3, "read_ips", lambda report: 50.0), \
         mock.patch.object(n2.L.m3, "kill_tree", lambda pid: None), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_px2i(lab)
    return started, pinned


def test_px2i_runs_the_four_conditions_for_both_models():
    lab = StubLab(models={m: StubModelInfo(m) for m in n2.PX2I_MODELS})
    started, pinned = _dry_run(lab)
    calls = [r for r in lab.rows if r.get("kind") == "call"]
    assert n2.PX2I_MODELS == ["qwen3-8b", "qwen3-14b"]
    assert n2.PX2I_CONDITIONS == ["N0", "B4", "e8", "N1"]
    for m in n2.PX2I_MODELS:
        mine = [r for r in calls if r.get("model_id") == m]
        assert {r["co_runner"] for r in mine} == set(n2.PX2I_CONDITIONS)
        assert len(mine) == 4 * (1 + n2.PX2_N_MEASURED)
    assert pinned == [[0, 1, 2, 3], [0, 1, 2, 3]]


def test_px2i_b4_is_bandwidth_on_e_cluster_a_and_e8_spin_on_all_e_cores():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    started, _ = _dry_run(lab)
    by_item = {item: (cpus, kind) for item, cpus, kind in started}
    assert by_item["PX2I_qwen3-8b_B4"] == ((4, 5, 6, 7), "bw")
    assert by_item["PX2I_qwen3-8b_e8"] == (tuple(range(4, 12)), "spin")
    assert "PX2I_qwen3-8b_N0" not in by_item and "PX2I_qwen3-8b_N1" not in by_item
    assert by_item["PX2_bw_calibration"] == ((4, 5, 6, 7), "bw")  # px2_bw_calibration's own item id
    assert "PX2I_bw_calibration" in lab.done
    for cpus, _ in by_item.values():
        assert not set(cpus) & set(n2.PX2I_SERVER_CPUS)


def test_px2i_order_keeps_baselines_outside_and_is_reproducible():
    for m in n2.PX2I_MODELS:
        order, seed = n2.px2i_order(m)
        assert order[0] == "N0" and order[-1] == "N1" and set(order[1:3]) == {"B4", "e8"}
        assert n2.px2i_order(m) == (order, seed)


def test_px2i_emits_level_zero_summary_and_drift_check():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    _dry_run(lab)
    lz = [r for r in lab.rows if r.get("record") == "px2i_level_zero"]
    assert [r["cond"] for r in lz] == n2.px2i_order("qwen3-8b")[0]
    assert all("igpu_mhz_median" in r and "igpu_throttle_bits" in r and "sysman_available" in r for r in lz)
    summ = [r for r in lab.rows if r.get("record") == "px2i_model_summary"]
    assert len(summ) == 1 and "drift_flag" in summ[0]


def test_px2i_level_zero_summary_uses_the_condition_rows():
    rows = [{"igpu_mhz": 2000, "pkg_power_w": 30, "igpu_throttle_bits": [0]},
            {"igpu_mhz": 1600, "pkg_power_w": 40, "igpu_throttle_bits": [2]}, None]
    s = n2.px2i_level_zero_summary(rows)
    assert s["igpu_mhz_median"] == 1800 and s["pkg_power_w_median"] == 35 and s["igpu_throttle_bits"] == [0, 2]


def test_px2i_is_opt_in_smoke_gated_and_estimated():
    assert n2.PHASE_FN["px2i"] is n2.phase_px2i
    assert "px2i" not in n2.PHASE_ORDER.split(",")
    assert "px2i" in n2.SMOKE_GATED_PHASES
    lab = StubLab()
    lab.table = {}
    est = n2.estimate_hours(lab, {"load_s": {}, "call_e2e_s": {}, "gate_wait_s": 20.0})
    assert est["px2i"] > 0
