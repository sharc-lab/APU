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
    assert g == {"first_gated_failure": 4, "silent": True, "first_gated_failure_fields": ["rule1_json_keys"],
                 "after_window_exceeded": True}


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


# ── per tier x model gated table, baseline-type failures, excess over default (2026-10-14) ──────────────────

def _excess_files(tmp_path):
    """Default tier: llama fails rule1 at turn 4 with no window ever exceeded (gated, baseline-type) in seeds 1-2;
    qwen3:8b fails tool args (not a gated field without a validation baseline) at turn 4, before the first over-window
    turn 25, in every seed (kill-criterion silent, baseline-type). 4096 tier: llama fails rule1 at turn 6 after the
    window is exceeded at turn 3 in seeds 1-4 and not at all in seed 5; qwen3:8b fails rule1 at turn 2 (before turn 3,
    baseline-type) in seed 1 only."""
    rows = []
    for seed in (1, 2, 3, 4, 5):
        rows += _session_rows("llama3.1:8b", "ollama_default_call2_notools", seed, call2_mode="off",
                              events={4: {"rule1_json_keys": False}} if seed <= 2 else {})
        rows += _session_rows("qwen3:8b", "ollama_default_call2_notools", seed, call2_mode="off",
                              events={4: {"tool_args_all": False}, 25: {"over_loaded_window": True}})
        ev = {i: {"over_loaded_window": True} for i in range(3, 41)}
        if seed <= 4:
            ev[6] = {"over_loaded_window": True, "rule1_json_keys": False}
        rows += _session_rows("llama3.1:8b", "ollama_ctx_4096_call2_notools", seed, call2_mode="off", events=ev)
        ev = {i: {"over_loaded_window": True} for i in range(3, 41)}
        if seed == 1:
            ev[2] = {"rule1_json_keys": False}
        rows += _session_rows("qwen3:8b", "ollama_ctx_4096_call2_notools", seed, call2_mode="off", events=ev)
    _write(tmp_path / "x.jsonl", rows, RULES)
    return rr.build_combined([tmp_path / "x.jsonl"], validation_paths=[tmp_path / "none.jsonl"])


def test_gated_per_model_and_baseline_type(tmp_path):
    rep = _excess_files(tmp_path)
    g = rr.gated_per_model(rep)
    assert g["ollama_default|llama3.1:8b"]["silent"] == 2
    assert g["ollama_default|llama3.1:8b"]["silent_after_window_exceeded"] == 0
    assert g["ollama_default|llama3.1:8b"]["silent_baseline_type"] == 2
    assert g["ollama_default|qwen3:8b"]["silent"] == 0          # tool args is not gated here
    ll = g["num_ctx_4096|llama3.1:8b"]
    assert (ll["n"], ll["silent"], ll["silent_after_window_exceeded"], ll["silent_baseline_type"]) == (5, 4, 4, 0)
    assert g["num_ctx_4096|qwen3:8b"]["silent"] == 1 and g["num_ctx_4096|qwen3:8b"]["silent_baseline_type"] == 1
    assert g["num_ctx_4096|qwen3:8b"]["silent_after_window_exceeded"] == 0
    d = rr.silent_session_detail(rep, "ollama_default")
    assert [(x["model"], x["seed"], x["first_failure"], x["what_failed"], x["first_over_window"]) for x in d] == \
        [("llama3.1:8b", 1, 4, ["rule1_json_keys"], None), ("llama3.1:8b", 2, 4, ["rule1_json_keys"], None)]
    k = rr.kill_baseline_type(rep, "ollama_default")
    assert k["n_sessions"] == 10 and k["n_silent"] == 7 and k["n_baseline_type"] == 7 and k["n_after_window"] == 0
    assert k["n_never_over_window"] == 2 and k["n_before_first_over_window"] == 5
    assert {x["model"]: x["what_failed"] for x in k["sessions"]}["qwen3:8b"] == ["tool_args"]
    # the kill-criterion silent count per tier agrees with the pre-registered evaluation
    assert rep["kill"]["per_arm"]["ollama_default_call2_notools"]["n_silent_failures"] == 7


def test_excess_over_default_point_and_bootstrap(tmp_path):
    rep = _excess_files(tmp_path)
    x = rr.excess_over_default(rep, n_resamples=500, seed=7)
    assert x["unit"] == "session" and x["n_resamples"] == 500 and x["default_tier"] == "ollama_default"
    t = x["per_tier"]["num_ctx_4096"]
    assert "ollama_default" not in x["per_tier"]
    assert t["overall"]["rate"] == 0.4 and t["overall"]["default_rate"] == 0.0 and t["overall"]["excess"] == 0.4
    assert t["llama3.1:8b"]["excess"] == 0.8 and t["qwen3:8b"]["excess"] == 0.0   # baseline-type is not after-window
    lo, hi = t["overall"]["ci95"]
    assert 0.0 <= lo <= 0.4 <= hi <= 0.8
    assert t["qwen3:8b"]["ci95"] == (0.0, 0.0)                                   # 0/n cell: no bootstrap variance
    # seeded: the same seed gives the same CI
    assert rr.excess_over_default(rep, n_resamples=500, seed=7)["per_tier"] == x["per_tier"]


def test_excess_over_default_nonzero_default_rate(tmp_path):
    """If the default tier itself has after-window failures, they are subtracted."""
    rows = []
    for seed in (1, 2, 3, 4):
        ev = {i: {"over_loaded_window": True} for i in range(10, 41)}
        if seed == 1:
            ev[12] = {"over_loaded_window": True, "rule1_json_keys": False}
        rows += _session_rows("llama3.1:8b", "ollama_default_call2_notools", seed, call2_mode="off", events=ev)
        ev = {i: {"over_loaded_window": True} for i in range(3, 41)}
        if seed <= 3:
            ev[5] = {"over_loaded_window": True, "rule1_json_keys": False}
        rows += _session_rows("llama3.1:8b", "ollama_ctx_4096_call2_notools", seed, call2_mode="off", events=ev)
    _write(tmp_path / "y.jsonl", rows, RULES)
    rep = rr.build_combined([tmp_path / "y.jsonl"], validation_paths=[tmp_path / "none.jsonl"])
    v = rr.excess_over_default(rep, n_resamples=200, seed=1)["per_tier"]["num_ctx_4096"]["llama3.1:8b"]
    assert v["rate"] == 0.75 and v["default_rate"] == 0.25 and v["excess"] == 0.5
    assert -0.25 <= v["ci95"][0] <= 0.5 <= v["ci95"][1] <= 1.0


def test_percentile_interpolates():
    assert rr._percentile([0.0, 1.0], 0.5) == 0.5 and rr._percentile([3.0], 0.975) == 3.0
    assert rr._percentile([], 0.5) is None
