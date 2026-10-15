"""analysis/impact_model.py on synthetic inputs: formulas, not-measured propagation, ranking, totals, the R2 helpers,
and the committed profile's provenance fields."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import impact_model as im  # noqa: E402


def _profile(**over):
    a = {"team_size": 2, "agent_sessions_per_dev_per_day": 5, "turns_per_session": 10,
         "single_shot_calls_per_dev_per_day": 3, "workdays_per_month": 20, "co_runner_share": 0.5,
         "cloud_session_share": 0.5, "device": "evo-x2", "model": "m:8b", "runtime": "ollama_default",
         "scenario_runtime": "ollama_ctx_4096", "cloud_model": "gpt-6.1-sol", "cloud_tier": "standard"}
    a.update(over)
    return {"hour_value": {"usd_per_hour": 100.0}, "assumptions": a}


# ── pure formulas ──

def test_cost_p1_formula_zero_rate_and_not_measured():
    assert im.cost_p1(10, 0.5, 2.0, 100.0) == pytest.approx(1000.0)
    assert im.cost_p1(10, 0.0, None, 100.0) == 0.0          # a measured zero rate needs no per-event cost
    assert im.cost_p1(10, 0.5, None, 100.0) is None         # nonzero rate, unmeasured cost: not a number
    assert im.cost_p1(10, None, 2.0, 100.0) is None


def test_cost_p2_clamps_negative_delta_and_scales_by_turns():
    assert im.cost_p2(10, 1.8, 40, 100.0) == pytest.approx(10 * 1.8 * 40 / 3600 * 100)
    assert im.cost_p2(10, -3.0, 40, 100.0) == 0.0
    assert im.cost_p2(10, None, 40, 100.0) is None


def test_cost_p3_p4_p5_p6_p7():
    assert im.cost_p3(10, 0.5, 2.0) == pytest.approx(10.0)
    assert im.cost_p3(10, 0.5, None) is None
    # 3600 tokens at 10 tok/s vs best 20 tok/s: 180 s extra per session
    assert im.cost_p4(1, 3600, 10.0, 20.0, 3600.0) == pytest.approx(180.0)
    assert im.cost_p4(1, 3600, 20.0, 20.0, 3600.0) == 0.0
    assert im.cost_p4(1, None, 10.0, 20.0, 3600.0) is None
    # prefill 100 s x 0.5 + decode 100 s x (1/0.8 - 1) = 50 + 25 s
    assert im.cost_p5(1, 1.0, 100.0, 100.0, 1.5, 0.8, 3600.0) == pytest.approx(75.0)
    assert im.cost_p5(1, 1.0, None, 100.0, 1.5, 0.8, 3600.0) is None
    assert im.cost_p6(10, 0.1, 900, 0.0, None, 3600.0) == pytest.approx(900.0)
    assert im.cost_p6(10, None, None, None, None, 3600.0) is None
    assert im.cost_p7(10, 0.0, None) == 0.0
    assert im.cost_p7(10, 0.4, None) is None


def test_rank_puts_not_measured_last_and_total_excludes_it():
    rows = {"P1": {"usd_day": None}, "P2": {"usd_day": 5.0}, "P3": {"usd_day": 9.0}, "P4": {"usd_day": 0.0}}
    assert im.rank(rows) == ["P3", "P2", "P4", "P1"]
    t = im.total(rows)
    assert t["usd_day"] == 14.0 and t["excluded_not_measured"] == ["P1"]


def test_tier_of_matches_r2_report_labels():
    assert im.tier_of("ollama_default") == "ollama_default"
    assert im.tier_of("ollama_ctx_4096") == "num_ctx_4096"
    assert im.tier_of("ollama_ctx_32768_call2_notools") == "num_ctx_32768"
    assert im.tier_ctx("num_ctx_8192") == 8192 and im.tier_ctx("ollama_default") is None


# ── rows on injected inputs ──

def test_p1_and_p7_rows_on_synthetic_rates():
    prof = _profile()
    rates = {("m:8b", "ollama_default"): {"n": 4, "silent_ctx": 0, "rule_after_window": 0},
             ("m:8b", "num_ctx_4096"): {"n": 4, "silent_ctx": 2, "rule_after_window": 1}}
    wall = {("evo-x2", "m:8b", "ollama_default"): {"median_wall_s": 3600.0}}
    b3 = {"files": [], "windows": [], "met": False}
    r = im.p1(REPO, prof, inputs={"rates": rates, "wall": wall, "b3": b3})
    assert r["usd_day"] is None                       # P1 not measured until B3 task runs exist
    assert r["cell"]["scenario_usd_day"] == 0.0
    r4 = im.p1(REPO, prof, runtime="ollama_ctx_4096", inputs={"rates": rates, "wall": wall, "b3": b3})
    assert r4["usd_day"] is None
    assert r4["cell"]["scenario_usd_day"] == pytest.approx(10 * 0.5 * 1.0 * 100.0)   # ASSUMED, scenario only
    au = im.audit_row("P1", r4, prof)
    assert au["status"] == "assumed" and au["measured"] is None and au["assumed"] == pytest.approx(500.0)
    assert im.p7(REPO, prof, inputs={"rates": rates})["usd_day"] == 0.0
    assert im.p7(REPO, prof, runtime="ollama_ctx_4096", inputs={"rates": rates})["usd_day"] is None


def test_p1_is_not_measured_off_the_measured_device():
    prof = _profile(device="evo-t2s")
    rates = {("m:8b", "ollama_default"): {"n": 4, "silent_ctx": 1, "rule_after_window": 0}}
    r = im.p1(REPO, prof, inputs={"rates": rates, "wall": {}})
    assert r["usd_day"] is None


def test_p2_p3_p6_rows_on_synthetic_inputs():
    prof = _profile()
    d = {("m:8b", "ollama_default"): {"vs": "num_ctx_32768", "median_delta_s": 3.6, "n_turns": 9}}
    assert im.p2(REPO, prof, inputs={"deltas": d})["usd_day"] == pytest.approx(10 * 3.6 * 10 / 3600 * 100)
    over = {"m:8b": {"uncached": 3.0, "cached": 1.0, "overspend": 2.0, "resent_cached": 0.1, "resent_uncached": 2.0,
                     "resent_tokens": 1000, "n": 2}}
    r3 = im.p3(REPO, prof, inputs={"overspend": over})
    assert r3["usd_day"] == pytest.approx(10 * 0.5 * 0.1)          # caching on: main case
    assert r3["usd_day_upper"] == pytest.approx(10 * 0.5 * 2.0)    # no caching: upper bound only
    au = im.audit_row("P3", r3, prof)
    assert au["status"] == "measured" and au["low"] == pytest.approx(0.5) and au["high"] == pytest.approx(10.0)
    ss = {("evo-x2", "m-8b", "ollama_default"): {"p_timeout": 0.1, "timeout_s": 900, "p_fail": 0.0, "fail_s": None,
                                                 "n": 10, "n_events": 1, "escalation_usd_per_call": 0.01}}
    r = im.p6(REPO, prof, inputs={"single_shot": ss})
    assert r["usd_day"] == pytest.approx(6 * 0.1 * 900 / 3600 * 100)
    au = im.audit_row("P6", r, prof)        # time: 0 (nobody waits) to the full hour value
    assert au["status"] == "mixed" and au["measured"] == 0.0 and au["assumed"] == pytest.approx(r["usd_day"])
    assert r["cell"]["escalation_usd_day"] == pytest.approx(0.06)
    assert im.p6(REPO, prof, runtime="ollama_ctx_4096", inputs={"single_shot": ss})["usd_day"] is None


def test_p4_and_p5_rows_on_synthetic_inputs():
    prof = _profile()
    wall = {("evo-x2", "m:8b", "ollama_default"): {"median_wall_s": 100.0, "mean_prefill_s": 360.0,
                                                   "mean_decode_s": 36.0, "mean_output_tokens": 360.0, "n": 2}}
    rates = {"evo-x2": {"m-8b": 10.0}, "evo-t2s": {"m-8b": 5.0}}
    assert im.p4(REPO, prof, inputs={"rates": rates, "wall": wall})["usd_day"] == 0.0
    t2s = im.p4(REPO, _profile(device="evo-t2s"), inputs={"rates": rates, "wall": wall})
    assert t2s["usd_day"] == pytest.approx(10 * 360 * (1 / 5 - 1 / 10) / 3600 * 100)
    ratios = {"evo-x2": {"claim_id": "X", "ttft": {"m-8b": 1.5}, "decode": {"m-8b": 0.5}},
              "evo-t2s": {"claim_id": "Y", "ttft": {"m-8b": 1.4}, "decode": {}}}
    r = im.p5(REPO, prof, inputs={"ratios": ratios, "wall": wall})
    assert r["usd_day"] == pytest.approx(10 * 0.5 * (360 * 0.5 + 36 * 1.0) / 3600 * 100)
    assert im.p5(REPO, _profile(device="evo-t2s"), inputs={"ratios": ratios, "wall": wall})["usd_day"] is None


def test_model_rows_and_render_block_print_not_measured(monkeypatch):
    prof = _profile()
    stub = {"id": "PX", "usd_day": None, "cell": None, "grid": {}, "config": "c"}
    for k in im.ROW_FUNCS:
        monkeypatch.setitem(im.ROW_FUNCS, k, lambda repo, profile, runtime=None, inputs=None, k=k: dict(stub, id=k))
    rows = im.model_rows(REPO, prof)
    assert all(r["usd_day"] is None and r["usd_month"] is None for r in rows.values())
    audits = {k: r["audit"] for k, r in rows.items()}
    res = {"profile": prof, "base": rows, "scenario": rows, "total": im.total(rows), "scenario_total": im.total(rows),
           "audit_total": im.audit_total(rows, audits), "scenario_audit_total": im.audit_total(rows, audits),
           "ranking": im.rank(rows), "previous_basis": None}
    block = im.render_block(res, {})
    assert block.startswith(im.BEGIN) and block.endswith(im.END)
    assert "| P1 | silent wrong results from context loss | not measured | not measured | not measured |" in block
    assert chr(0x2014) not in block  # no em dashes in generated docs


# ── R2 helpers on synthetic sessions ──

def _turn(t, wall, over=False):
    return {"turn_idx": t, "turn_wall_s": wall, "over_loaded_window": over,
            "calls": [{"prompt_eval_duration_s": wall / 2, "eval_duration_s": wall / 4, "completion_tokens": 10}]}


def test_oversize_turn_delta_pairs_only_turns_that_fit_both(monkeypatch):
    sess = [
        {"device": "evo-x2", "model": "m", "tier": "ollama_default", "seed": 1,
         "turns": [_turn(1, 5.0), _turn(2, 6.0), _turn(3, 7.0)]},
        {"device": "evo-x2", "model": "m", "tier": "num_ctx_4096", "seed": 1,
         "turns": [_turn(1, 4.0), _turn(2, 4.0), _turn(3, 1.0, over=True)]},
    ]
    monkeypatch.setattr(im, "r2_data", lambda repo=None: {"evo-x2": {"sessions": sess, "report": None}})
    d = im.oversize_turn_delta(REPO)
    assert d[("m", "ollama_default")] == {"vs": "num_ctx_4096", "median_delta_s": 1.5, "n_turns": 2}
    assert d[("m", "num_ctx_4096")]["vs"] is None
    w = im.session_wallclock(REPO)
    assert w[("evo-x2", "m", "ollama_default")]["median_wall_s"] == 18.0
    assert w[("evo-x2", "m", "ollama_default")]["mean_output_tokens"] == 30


def test_context_loss_rates_counts_only_failures_after_the_window(monkeypatch):
    rep = {"table": {"num_ctx_4096|m": {"n": 3, "rules_in_use": ["rule1"], "sessions": [
        {"silent": True, "after_window_exceeded": True, "first_over_window": 3, "first_fail_rule1": 4},
        {"silent": True, "after_window_exceeded": False, "first_over_window": 3, "first_fail_rule1": 2},
        {"silent": False, "after_window_exceeded": False, "first_over_window": None, "first_fail_rule1": None}]}}}
    monkeypatch.setattr(im, "r2_data", lambda repo=None: {"evo-x2": {"sessions": [], "report": rep}})
    assert im.context_loss_rates(REPO) == {("m", "num_ctx_4096"): {"n": 3, "silent_ctx": 1, "rule_after_window": 1}}


# ── committed inputs ──

def test_profile_hour_value_is_cited_and_other_fields_are_assumptions():
    prof = im.load_profile(REPO)
    hv = prof["hour_value"]
    for k in ("usd_per_hour", "source_url", "table", "reference_period", "fetched_utc", "figure_quoted"):
        assert hv.get(k), k
    assert hv["source_url"].startswith("https://www.bls.gov/")
    assert str(hv["usd_per_hour"]) in hv["figure_quoted"].replace("$", "")
    assert "ASSUMPTION" in prof["purpose"]
    for k in ("team_size", "agent_sessions_per_dev_per_day", "device", "model", "runtime", "cloud_model"):
        assert k in prof["assumptions"]


def test_escalation_ids_and_price_file_are_shared():
    assert list(im.ESCALATIONS) == ["E1", "E2", "E3", "E4"]
    prof = im.load_profile(REPO)
    assert im.price_file(prof) == im.PRICES_FILE == "results/cloud_prices_anthropic_20261014.json"
    prices = im.prices(REPO, prof)
    assert prof["assumptions"]["cloud_model"] in prices["models"]
    assert prices["source_url"].startswith("https://platform.claude.com/") and prices["fetched_utc"]
    s = prices["models"]["claude-sonnet-5-5"]["standard"]["short"]
    assert (s["input"], s["cache_write"], s["cached_input"], s["output"]) == (2.00, 2.50, 0.10, 10.00)


def test_resent_history_and_escalation_caching_rules():
    import cloud_budget_forecast as cbf
    pr = {"models": {"m": {"min_cacheable_tokens": 512, "standard": {
        "short": {"input": 2.0, "cached_input": 0.1, "cache_write": 2.5, "output": 10.0}}}}}
    calls = [(600, 10), (1000, 10), (400, 10)]
    assert cbf.resent_tokens(calls, 512) == [0, 600, 400]
    rh = im.resent_history_cost(calls, "m", "standard", pr)
    assert rh["tokens"] == 1000 and rh["cached"] == pytest.approx(1000 * 0.1 / 1e6)
    assert rh["uncached"] == pytest.approx(1000 * 2.0 / 1e6)
    # second step within the cache lifetime hits, third one (400 s later) does not
    e = im.escalated_cost([(600, 0, 0.0), (1000, 0, 100.0), (1200, 0, 500.0)], "m", "standard", pr)
    assert e["hits"] == 1
    on = (600 * 2.5 + (600 * 0.1 + 400 * 2.5) + 1200 * 2.5) / 1e6
    assert e["cached"] == pytest.approx(on) and e["uncached"] == pytest.approx(2800 * 2.0 / 1e6)


def test_b3_gate_ignores_validation_files(tmp_path):
    (tmp_path / "results").mkdir()
    row = {"record": "r2a_turn", "arm_id": "ollama_ctx_8192_call2_notools", "end_state_ok": True}
    (tmp_path / "results" / "x2_evalset_tasks_val_v1.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert im.b3_task_windows(tmp_path)["met"] is False
    rows = [row, dict(row, arm_id="ollama_ctx_4096_call2_notools")]
    (tmp_path / "results" / "x2_evalset_tasks_real_v1.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    g = im.b3_task_windows(tmp_path)
    assert g["met"] is True and g["windows"] == ["num_ctx_4096", "num_ctx_8192"]


def test_audit_total_adds_measured_only_and_lists_assumed():
    audits = {"P1": {"measured": None, "assumed": 9.0, "status": "assumed", "low": None, "high": None},
              "P2": {"measured": 0.0, "assumed": 5.0, "status": "mixed", "low": 0.0, "high": 5.0},
              "P3": {"measured": 1.0, "assumed": 0.0, "status": "measured", "low": 1.0, "high": 20.0}}
    t = im.audit_total({}, audits)
    assert t["measured_usd_day"] == 1.0 and t["low"] == 1.0 and t["high"] == 25.0
    assert t["excluded_assumed"] == ["P1"]


def test_register_has_impact_rows():
    import numbers_register as nr
    ids = {e["claim_id"] for e in nr.NUMBER_ENTRIES}
    assert {"impact-hour-value", "impact-total"} | {f"impact-P{i}" for i in range(1, 8)} <= ids
    assert not ({"zach-kv-residency-xps", "zach-kv-residency-t2s", "zach-npu-export"} & ids)  # data not found


def test_hour_value_row_on_a_synthetic_profile(tmp_path):
    import numbers_register as nr
    (tmp_path / "results").mkdir()
    p = _profile()
    p["hour_value"] = {"usd_per_hour": 12.5, "figure_quoted": "$12.50 per hour", "table": "T",
                       "reference_period": "May 2099 (x)", "source_url": "https://example.org", "fetched_utc": "2099-01-01"}
    (tmp_path / im.PROFILE_FILE).write_text(json.dumps(p), encoding="utf-8")
    v = nr.compute_impact_hour_value(tmp_path)["value"]
    assert v.startswith("USD 12.50/h") and "May 2099;" in v
