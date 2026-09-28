"""Co-runner calls must use a fixed 20 s settle, not the package-power proxy (which cannot release while the hog runs)."""

import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import t2s_lab as L  # noqa: E402


def test_corunner_active_uses_fixed_settle_not_proxy():
    tele = L.Telemetry.__new__(L.Telemetry)
    tele.sys_ring = L.Ring()
    with patch.object(time, "sleep"), patch.object(tele, "pkg_now", return_value=44.9):
        g = tele.thermal_gate(idle_temp=None, idle_pkg=22.8, corunner_active=True)
    assert g["gate_released_by"] == "corunner_fixed_settle"
    assert g["thermal_wait_s"] == 20.0
    assert g["pkg_w_at_release"] == 44.9


def test_no_corunner_keeps_the_proxy_gate():
    tele = L.Telemetry.__new__(L.Telemetry)
    tele.sys_ring = L.Ring()
    with patch.object(time, "sleep"), patch.object(tele, "pkg_now", return_value=22.9):
        g = tele.thermal_gate(idle_temp=None, idle_pkg=22.8, corunner_active=False)
    assert g["gate_released_by"] == "pkg_power_proxy"
