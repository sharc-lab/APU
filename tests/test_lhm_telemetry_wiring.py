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


# ---------------------------------------------------------------------------------------------------- CPU package power
def test_metrics_pkg_power_w_falls_back_to_lhm_cpu_when_no_rapl(monkeypatch):
    """No Windows RAPL Energy Meter instance exists on AMD, so pkg_power_w must come from the LHM-fed 'Package'
    sensor instead of staying None (found missing 2026-09-29: PX2's power-effect criterion had no CPU power
    signal at all on evo-x2 even though LHM does expose one)."""
    t = make_tele()
    t.sys_ring.add({"kind": "power", "domain": 0, "power_w": 21.5, "source": "lhm_cpu", "t": 10.0})
    t.sys_ring.add({"kind": "power", "domain": 0, "power_w": 85.0, "source": "lhm_cpu", "t": 12.0})
    m = t.metrics(0.0, 20.0)
    assert m["pkg_power_w"] == 53.25  # median of the two lhm_cpu samples


def test_metrics_pkg_power_w_prefers_rapl_over_lhm_cpu_when_both_present():
    """On Intel (or if a future AMD platform exposes a real RAPL instance), the existing RAPL-based reading must
    still win -- lhm_cpu is a fallback, not a replacement."""
    t = make_tele()
    t.win_ring.add({"rapl_pkg_mw": 45000, "t": 10.0})  # 45 W via RAPL
    t.sys_ring.add({"kind": "power", "domain": 0, "power_w": 999.0, "source": "lhm_cpu", "t": 10.0})
    m = t.metrics(0.0, 20.0)
    assert m["pkg_power_w"] == 45.0


def test_metrics_pkg_power_w_none_when_neither_source_present():
    t = make_tele()
    m = t.metrics(0.0, 20.0)
    assert m["pkg_power_w"] is None


def test_lhm_loop_maps_ryzen_package_power_key():
    """Sensor-key matching logic (mirrors t2s_lab.Telemetry._start_lhm_feeder's lhm_loop closure): any 'Power |'
    key containing both 'ryzen' and 'package' (case-insensitive) is the CPU package power, confirmed live against
    evo-x2's real LHM sensor dump 2026-09-29 ('Power | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | Package')."""
    sensors = {
        "Power | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | Package": 13.834,
        "Power | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S | Core #7 (SMU)": 1.586,
        "Power | AMD Radeon(TM) 8060S Graphics | GPU Core": 0,
    }
    cpu_power = next((v for k, v in sensors.items() if k.startswith("Power |") and "ryzen" in k.lower() and "package" in k.lower()), None)
    assert cpu_power == 13.834
