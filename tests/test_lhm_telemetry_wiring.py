"""On AMD, iGPU clock/power and CPU/iGPU temperature are fed into Telemetry.sys_ring as synthetic Sysman-shaped rows
by the LHM feeder, so metrics()/temp_now() need no AMD-specific branch. This tests that translation without starting
any real process (no PawnIO, no LHM dll needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import t2s_lab as L  # noqa: E402


def make_tele():
    t = L.Telemetry.__new__(L.Telemetry)
    t.sys_ring, t.win_ring, t.gpu_ring, t.avail_ring = L.Ring(), L.Ring(), L.Ring(), L.Ring()
    return t


def test_temp_now_uses_domain_0_only():
    t = make_tele()
    now = 1000.0
    t.sys_ring.add({"kind": "temp", "domain": 0, "temp_c": 55.0, "t": now})
    t.sys_ring.add({"kind": "temp", "domain": 1, "temp_c": 90.0, "t": now})  # GPU, must not leak into temp_now()
    import time
    orig = time.time
    time.time = lambda: now
    try:
        assert t.temp_now() == 55.0
    finally:
        time.time = orig


def test_metrics_reports_igpu_power_and_temp_separately_from_cpu():
    t = make_tele()
    t.sys_ring.add({"kind": "freq", "domain": 0, "actual_mhz": 2600.0, "throttle_reasons": 0, "t": 10.0})
    t.sys_ring.add({"kind": "power", "domain": 0, "power_w": 32.0, "source": "lhm_gpu", "t": 10.0})
    t.sys_ring.add({"kind": "temp", "domain": 0, "temp_c": 59.0, "t": 10.0})  # CPU
    t.sys_ring.add({"kind": "temp", "domain": 1, "temp_c": 54.5, "t": 10.0})  # iGPU
    m = t.metrics(0.0, 20.0)
    assert m["igpu_mhz"] == 2600.0
    assert m["igpu_power_w"] == 32.0
    assert m["igpu_temp_c_max"] == 54.5
    assert m["temp_c_max"] == 59.0  # includes both domains, same as before this change


def test_metrics_without_lhm_rows_keeps_old_none_behavior():
    t = make_tele()
    m = t.metrics(0.0, 20.0)
    assert m["igpu_power_w"] is None
    assert m["igpu_temp_c_max"] is None
