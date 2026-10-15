"""analysis/business_metrics.py: the pure arithmetic on synthetic inputs (policy totals, the router's P1/P7 credit
rule, today's mix, steps kept local) and the committed profile file's provenance and assumption labels."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import business_metrics as bm  # noqa: E402


def test_policy_total_lists_not_measured_and_never_fills_it_in():
    t = bm.policy_total({"P1": None, "P2": 2.0, "P4": 1.0, "P5": None, "P7": 0.0}, cloud_usd_day=3.0)
    assert t["usd_day"] == pytest.approx(6.0)
    assert t["time_usd_day"] == pytest.approx(3.0) and t["cloud_usd_day"] == 3.0
    assert t["not_measured"] == ["P1", "P5"]


def test_router_rows_no_assumption_when_naive_p1_p7_are_measured_zero():
    rows, used = bm.router_rows({"P1": 0.0, "P2": 5.0, "P7": 0.0}, None)
    assert rows == {"P1": 0.0, "P2": 5.0, "P7": 0.0} and used is False


def test_router_rows_credit_only_where_the_mitigation_recovered():
    naive = {"P1": 100.0, "P2": 5.0, "P7": None}
    rows, used = bm.router_rows(naive, "recovered")
    assert rows["P1"] == 0.0 and rows["P7"] == 0.0 and rows["P2"] == 5.0 and used is True
    for verdict in ("not_recovered", "invalid", None):
        rows, used = bm.router_rows(naive, verdict)
        assert rows == naive and used is False      # no credit: naive P1/P7 kept


def test_mix_and_local_share():
    assert bm.mix_usd_day(0.25, 400.0, 100.0) == pytest.approx(175.0)
    assert bm.mix_usd_day(0.0, 400.0, 100.0) == pytest.approx(100.0)
    assert bm.steps_local_share({"cloud_steps_per_session": 16, "steps_per_session": 80}) == pytest.approx(0.8)
    assert bm.steps_local_share({"cloud_steps_per_session": None, "steps_per_session": None}) is None


def test_time_range_and_steps_local_trace():
    assert bm.time_range({"P1": None, "P2": 2.0, "P4": 1.0, "P5": None, "P7": 0.0}) == (0.0, 3.0)
    assert bm.time_range({"P2": None, "P4": None, "P5": None}) == (None, None)
    esc = {"n": 5, "cloud_steps_values": [16], "steps_values": [80], "cloud_steps_per_session": 16,
           "steps_per_session": 80}
    assert "16 of 80 steps escalated in every one of 5" in bm.steps_local_trace(esc)
    assert "no step escalated" in bm.steps_local_trace(dict(esc, cloud_steps_values=[0]))


def test_profiles_drop_the_laptop_and_use_the_shared_anthropic_table():
    doc = bm.load_profiles()
    assert "small_team" not in doc["profiles"] and "small_team" in doc["dropped_profiles"]
    for p in doc["profiles"].values():
        a = p["assumptions"]
        assert a["device"] == "evo-x2"
        assert a["cloud_price_file"] == "results/cloud_prices_anthropic_20261014.json"
        assert a["cloud_model"] == "claude-sonnet-5-5"


def test_formatting_prints_not_measured_for_none():
    assert bm.usd(None) == bm.NOT_MEASURED
    assert bm.rng(1.0, None) == bm.NOT_MEASURED
    assert bm.rng(-5.0, 10.0) == "-5 to 10"


def test_committed_profiles_label_assumptions_and_cite_the_wage_source():
    doc = json.loads((REPO / bm.PROFILES_FILE).read_text(encoding="utf-8"))
    assert "ASSUMPTION" in doc["purpose"]
    r = doc["hour_value"]["range"]
    assert r["source_url"].startswith("https://www.bls.gov/") and r["fetched_utc"]
    assert any("82,460" in q and "214,670" in q for q in r["quoted"])
    assert len(doc["profiles"]) in (2, 3)
    for p in doc["profiles"].values():
        a = p["assumptions"]
        assert a["default_tier"] in a["router_tiers"]
        for k in ("team_size", "agent_sessions_per_dev_per_day", "workdays_per_month", "cloud_session_share"):
            assert k in a


def test_hour_values_use_the_page_implied_hours_per_year():
    doc = bm.load_profiles()
    h = bm.hour_values(doc)
    assert h["hours_per_year"] == pytest.approx(134040 / 64.44)
    assert h["p10"] < h["median"] < h["p90"]
    assert h["p10"] == pytest.approx(82460 / (134040 / 64.44))
