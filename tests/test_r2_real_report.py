"""analysis/r2_real_report: first-event and gated silent-failure logic on synthetic turns."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import r2_real_report as rr  # noqa: E402


def _t(i, **kw):
    base = {"turn_idx": i, "canary_check": False, "any_http_error": False, "over_loaded_window": False,
            "rule1_json_keys": True, "tool_validity_all": True, "tool_args_all": True, "recall_ok": None}
    base.update(kw)
    return base


def test_session_events_first_turns():
    turns = [_t(1), _t(2, over_loaded_window=True), _t(3, rule1_json_keys=False, over_loaded_window=True),
             _t(5, canary_check=True, canary_sys_ok=True, canary_hist_ok=False, over_loaded_window=True)]
    ev = rr.session_events(turns, ["rule1_json_keys"])
    assert ev["first_fail_rule1_json_keys"] == 3 and ev["first_canary_miss"] == 5
    assert ev["first_canary_miss_over_window"] is True and ev["first_over_window"] == 2
    assert ev["first_any_failure"] == 3 and ev["error_before_first_failure"] is False


def test_gated_session_ignores_excluded_metric_and_flags_after_window():
    turns = [_t(1, tool_args_all=False), _t(2, over_loaded_window=True), _t(4, rule1_json_keys=False, over_loaded_window=True)]
    g = rr.gated_session(turns, ["rule1_json_keys", "tool_validity_all"])  # tool_args excluded
    assert g == {"first_gated_failure": 4, "silent": True, "after_window_exceeded": True}


def test_gated_session_error_before_failure_is_not_silent():
    turns = [_t(1, any_http_error=True), _t(2, rule1_json_keys=False)]
    assert rr.gated_session(turns, ["rule1_json_keys"])["silent"] is False
