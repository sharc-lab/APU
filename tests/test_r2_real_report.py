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


# ── combined (v1 + v1b) report on synthetic files ───────────────────────────────────────────────────

import json  # noqa: E402

RULES = {"llama3.1:8b": ["rule1_json_keys"], "qwen3:8b": ["rule1_json_keys", "rule5_session_code"]}


def _session_rows(model, arm, seed, n_turns=40, events=None, call2_tools=False, call2_mode=None, attempt=None):
    """r2a_turn rows plus the r2a_session completion marker; events: {turn: {field: value}}."""
    attempt = attempt or f"{model}-{arm}-{seed}"
    rows = []
    for i in range(1, n_turns + 1):
        t = _t(i, canary_check=(i % 5 == 0), canary_sys_ok=True if i % 5 == 0 else None,
               canary_hist_ok=True if i % 5 == 0 else None, rule5_session_code=True)
        t.update((events or {}).get(i, {}))
        t.update({"record": "r2a_turn", "mode": "real", "model_id": model, "arm_id": arm, "seed": seed,
                  "attempt_id": attempt, "call2_tools": call2_tools})
        if call2_mode:
            t["call2_mode"] = call2_mode
        rows.append(t)
    s = {"record": "r2a_session", "mode": "real", "model_id": model, "arm_id": arm, "seed": seed,
         "attempt_id": attempt, "call2_tools": call2_tools, "first_failure_turn": None,
         "error_surfaced_before_failure": None}
    if call2_mode:
        s["call2_mode"] = call2_mode
    return rows + [s]


def _write(path, rows, rules):
    path.write_text("\n".join(json.dumps(r) for r in [{"record": "run_start", "rules_in_use": rules}] + rows),
                    encoding="utf-8")


def _files(tmp_path):
    v1 = []
    for seed in (1, 2, 3):
        v1 += _session_rows("llama3.1:8b", "ollama_ctx_4096_call2_notools", seed,
                            events={3: {"over_loaded_window": True}, 5: {"canary_hist_ok": False,
                                                                         "over_loaded_window": True}})
    v1b = []
    for seed in (4, 5):
        v1b += _session_rows("llama3.1:8b", "ollama_ctx_4096_call2_notools", seed, call2_mode="off",
                             events={3: {"over_loaded_window": True}, 12: {"rule1_json_keys": False}})
    for seed in (1, 2, 3, 4, 5):
        ev = {22: {"over_loaded_window": True}, 25: {"canary_hist_ok": False}}
        if seed == 5:
            ev[18] = {"any_http_error": True}
            ev[30] = {"rule5_session_code": False}
        v1b += _session_rows("qwen3:8b", "ollama_ctx_16384_call2_notools", seed, call2_mode="off", events=ev)
    # a duplicate of a v1 cell and a session of another call-2 mode: neither may be pooled
    v1b += _session_rows("llama3.1:8b", "ollama_ctx_4096_call2_notools", 1, call2_mode="off", attempt="dup",
                         events={1: {"rule1_json_keys": False}})
    v1b += _session_rows("llama3.1:8b", "ollama_ctx_4096_call2_forcednone", 4, call2_mode="forced_none")
    _write(tmp_path / "v1.jsonl", v1, {"llama3.1:8b": RULES["llama3.1:8b"]})
    _write(tmp_path / "v1b.jsonl", v1b, RULES)
    return [tmp_path / "v1.jsonl", tmp_path / "v1b.jsonl"]


def test_tier_of_and_call2_mode_of():
    assert rr.tier_of("ollama_ctx_16384_call2_notools") == "num_ctx_16384"
    assert rr.tier_of("ollama_default_call2_notools") == "ollama_default"
    assert rr.tier_of("ollama_ctx_8192_call2_forcednone_format") == "num_ctx_8192"
    assert rr.call2_mode_of({"call2_tools": False}) == "off" and rr.call2_mode_of({"call2_mode": "forced_none"}) == \
        "forced_none"


def test_combined_build(tmp_path):
    rep = rr.build_combined(_files(tmp_path), validation_paths=[tmp_path / "none.jsonl"])
    assert list(rep["table"]) == ["num_ctx_16384|qwen3:8b", "num_ctx_4096|llama3.1:8b"]  # TIER_ORDER
    ll = rep["table"]["num_ctx_4096|llama3.1:8b"]
    assert ll["seeds"] == [1, 2, 3, 4, 5] and ll["n"] == 5
    assert [e["first_over_window"] for e in ll["sessions"]] == [3, 3, 3, 3, 3]
    assert [e["first_canary_miss"] for e in ll["sessions"]] == [5, 5, 5, None, None]
    assert [e["first_fail_rule1_json_keys"] for e in ll["sessions"]] == [None, None, None, 12, 12]
    assert ll["survival"] == {5: 2, 10: 2, 15: 0, 20: 0, 25: 0, 30: 0, 35: 0, 40: 0}
    q = rep["table"]["num_ctx_16384|qwen3:8b"]
    s5 = q["sessions"][4]
    assert s5["first_error"] == 18 and s5["error_before_canary_miss"] is True
    assert s5["error_before_over_window"] is True and s5["error_before_fail_rule5_session_code"] is True
    assert q["sessions"][0]["error_before_over_window"] is False
    assert s5["error_before_first_failure"] is True and q["sessions"][0]["error_before_first_failure"] is False
    assert q["survival"][20] == 5 and q["survival"][25] == 0
    iss = rep["issues"]
    assert len(iss["duplicates"]) == 1 and iss["duplicates"][0]["kept_from"] == "v1.jsonl"
    assert len(iss["excluded_other_call2_mode"]) == 1
    assert rep["n_sessions"] == 10 and rep["n_turn_rows"] == 400
    # pre-registered kill: silent failures in both tiers (canary misses are not in the pre-registered rule; rule
    # failures are), gated variant per tier
    assert rep["kill"]["killed"] is False
    assert rep["gated_kill"]["num_ctx_4096"] == {"n_sessions": 5, "silent": 2, "silent_after_window_exceeded": 2}
    assert rep["gated_kill"]["num_ctx_16384"]["silent"] == 0 and rep["gated_killed"] is False
    md = rr.format_markdown(rep)
    assert "5/10/15/20/25/30/35/40" in md and "2/2/0/0/0/0/0/0 of 5" in md


def test_gated_metrics_uses_each_models_validation_file(monkeypatch):
    monkeypatch.setattr(rr, "_baseline_tables", lambda paths: {
        "llama3.1:8b": {"tool_validity": {"rate": 1.0}, "tool_args": {"rate": 0.95}, "recall": {"rate": 0.5}},
        "qwen3:8b": {"tool_validity": {"rate": 0.8}, "tool_args": {"rate": 1.0}, "recall": {"rate": 1.0}}})
    gm = rr.gated_metrics(RULES, validation_paths=["a", "b"])
    assert gm["llama3.1:8b"]["fields"] == ["rule1_json_keys", "tool_validity_all", "tool_args_all"]
    assert gm["llama3.1:8b"]["excluded"] == ["recall"]
    assert gm["qwen3:8b"]["excluded"] == ["tool_validity"]


def test_v1_build_keeps_its_checkpoints_and_order(tmp_path):
    files = _files(tmp_path)
    rep = rr.build(files[0])
    assert rep["checkpoints"] == [5, 10, 20, 30, 40]
    assert list(rep["table"]) == ["num_ctx_4096|llama3.1:8b"]
