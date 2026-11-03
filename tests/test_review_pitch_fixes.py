"""External review pitch fixes (2026-11-02): the default-window router row, the operating point's escalation-set
comparison, and the rolling-cache repriced rows, each computed from the committed result files (replay only)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "analysis", REPO / "harness", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hybrid_full_pipeline as H  # noqa: E402

HAVE = (REPO / "results/x2_own_config_probe_q4_28928.jsonl").exists() and (REPO / "results/all_cloud_b3_v1.jsonl").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="result files not synced")


class _Esc:
    source = "test stand-in (escalate the listed tasks)"

    def __init__(self, ids):
        self.ids = set(ids)

    def check(self, rec):
        return {"escalate": (rec["seed"], rec["task_id"]) in self.ids, "reason": "listed"}


class _Allow:
    source = "test stand-in (allow all)"

    def allow(self, k, usd):
        return {"allow": True, "cap_usd": 1.0}


def _entry(cid):
    import numbers_register as nr
    return next(e for e in nr.NUMBER_ENTRIES if e["claim_id"] == cid)


def test_escalation_set_comparison_with_a_stand_in_check():
    ids = [(20260901, "cust_phone_info"), (20260902, "inv_cable_len")]
    d = H.escalation_set_comparison(REPO, check=_Esc(ids), budget=_Allow())
    assert d["status"] == "VERIFIED" and d["b"]["ceiling"] == 40960 and d["a"]["ceiling"] == 28928
    assert d["a"]["routed"] == 0 and d["b"]["routed"] == 0
    assert d["same"] and d["both"] == sorted(ids) and not d["only_a"] and not d["only_b"]
    assert "f16 default" not in H.CONFIGS        # the comparison config is not a cell of hybrid-full-pipeline-v1


def test_operating_point_attribution_row_compares_the_two_sets():
    v = _entry("operating-point-check-attribution")["compute"](REPO)["value"]
    assert "q4_0 28928 (loaded window 28928)" in v and "f16 default (loaded window 40960)" in v
    assert ("SAME SET" in v) or ("DIFFERENT" in v)
    op = _entry("operating-point-q4-28928")["compute"](REPO)["value"]
    assert "replay, escalations imputed" in op and "post-run check" in op
    assert " / " not in op.split("cloud USD", 1)[1].split("; routed", 1)[0]   # the frontier parser splits on " / "


def test_default_window_router_row():
    import impact_model as im
    d = im.route_default_window(REPO)
    assert d["status"] == "VERIFIED" and d["window"] == 40960
    for c in d["cells"]:
        assert c["router"]["n"] == c["naive"]["n"] == 90
        assert c["router_8192"]["n"] == 90
    v = _entry("route-at-start-default-window")["compute"](REPO)["value"]
    assert "8,192 window (memory-constrained)" in v and "DEFAULT window (40960)" in v


def test_rolling_rows_show_old_and_new():
    for cid in ("all-cloud-b3-cost-rolling-cache", "hybrid-route-at-start-rolling-cache",
                "operating-point-q4-28928-rolling-cache", "cloud-cost-per-real-task-rolling-cache",
                "pitch-allcloud-bill-year-rolling-cache"):
        v = _entry(cid)["compute"](REPO)["value"]
        assert "OLD" in v and "NEW" in v, cid
    p = _entry("pitch-allcloud-bill-year-rolling-cache")["compute"](REPO)["value"]
    assert "USD 75,921.17 per 30-day month" in p and "USD 911K" in p      # the old row's figures stay visible


def test_old_rows_keep_the_assumed_hit_rate():
    import cloud_task_cost as ct
    assert ct.HIT_RATE == 0.7 and ct.CASES == ("none", "realistic", "best")
    import impact_model as im
    assert im.CURVE_CASE == "realistic"
