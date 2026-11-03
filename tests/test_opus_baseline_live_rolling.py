"""Rolling-breakpoint estimate of the live post-run check's cloud phase (actual escalation set of the finished local
phase) and the Opus all-cloud baseline beside Sonnet (priced from Sonnet token counts, not run on Opus). Offline: no
network, no client, no call."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "analysis", REPO / "harness", REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import post_run_check_tuning as prt  # noqa: E402

HAVE_LOCAL = (REPO / prt.LIVE_LOCAL_FILE).exists()
HAVE_B3 = (REPO / "results/all_cloud_b3_v1.jsonl").exists()


def _entry(cid):
    import numbers_register as nr
    return next(e for e in nr.NUMBER_ENTRIES if e["claim_id"] == cid)


# ── pure estimate ───────────────────────────────────────────────────────────────────────────────

def test_estimate_rolling_measured_imputed_and_cap():
    T = "ollama_ctx_8192"
    redo = {(T, 5, "a"): {"usd_new": 0.02, "usd_recorded": 0.03},
            (T, 1, "b"): {"usd_new": 0.04, "usd_recorded": 0.05},
            (T, 5, "big"): {"usd_new": 0.20, "usd_recorded": 0.21},
            ("other", 5, "c"): {"usd_new": 9.0, "usd_recorded": 9.0}}
    uppers = {(T, 5, "a"): 0.05, (T, 1, "b"): 0.07, (T, 5, "big"): 0.40, ("other", 5, "c"): 9.0}
    secs = {(T, 5, "a"): 4.0, (T, 1, "b"): 6.0, (T, 5, "big"): 8.0}
    esc = [{"seed": 5, "task_id": "a", "local_ok": False}, {"seed": 5, "task_id": "new", "local_ok": True},
           {"seed": 5, "task_id": "big", "local_ok": False}]
    e = prt.estimate_rolling(esc, T, redo, {}, uppers, secs, cap=0.25)
    mean = (0.02 + 0.04 + 0.20) / 3
    assert e["measured"] == ["a", "big"] and e["imputed"] == ["new"]
    per = {p["task_id"]: p for p in e["per_task"]}
    assert per["a"]["usd"] == pytest.approx(0.02) and per["a"]["upper"] == pytest.approx(0.05)
    assert per["new"]["usd"] == pytest.approx(mean) and per["new"]["src"] == "imputed_tier_mean"
    assert per["big"]["budget_exhausted"] and per["big"]["upper"] == 0.25 and per["big"]["usd"] == pytest.approx(0.20)
    assert e["point_billed"] == pytest.approx(0.02 + mean + 0.20)
    assert e["high_billed"] == pytest.approx(3 * 0.20)
    assert e["upper_bound"] == pytest.approx(3 * 0.25)       # max upper 0.40 capped at 0.25 (other tier ignored)
    assert e["cap_ceiling"] == pytest.approx(0.75)
    assert e["redo_s"] == pytest.approx(6.0) and e["cloud_hours"] == pytest.approx(3 * 6.0 / 3600)
    old = prt.estimate_rolling(esc, T, redo, {}, uppers, secs, cap=0.25, col="usd_recorded")
    assert old["point_billed"] > e["point_billed"]


# ── on the committed files ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAVE_LOCAL, reason="local phase file not synced")
def test_live_escalations_actual_set():
    le = prt.live_escalations(REPO)
    esc = [t for t in le["tasks"] if t["decision"] == "escalate"]
    assert len(le["tasks"]) == 30 and len(esc) == 20
    assert sum(not t["local_ok"] for t in esc) == 9 and sum(t["local_ok"] for t in esc) == 11
    assert all(t["local_ok"] for t in le["tasks"] if t["decision"] != "escalate")


@pytest.mark.skipif(not (HAVE_LOCAL and HAVE_B3), reason="result files not synced")
def test_live_8192_estimate_rolling_row():
    e = prt.live_estimate_rolling(REPO)
    n = e["new"]
    assert n["n_escalated"] == 20 and len(n["measured"]) + len(n["imputed"]) == 20
    assert n["point_billed"] <= n["high_billed"] <= n["upper_bound"] + 1e-12 <= n["cap_ceiling"] + 1e-12
    assert n["upper_bound"] <= 20 * 0.25 + 1e-12 and e["job_cap"] == 2.00
    txt = _entry("post-run-check-live-8192-estimate-rolling")["compute"](REPO)["value"]
    assert txt.startswith("ESTIMATE, not measured") and "20 of 30 escalated" in txt
    assert "MEASURED for 9 tasks" in txt and "IMPUTED for 11 tasks" in txt and chr(0x2014) not in txt
    # the old estimate row is kept, unchanged in method
    assert _entry("post-run-check-live-8192-estimate")["script_function"].endswith("::live_estimate")


# ── Opus priced from Sonnet tokens ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAVE_B3, reason="result files not synced")
def test_opus_priced_from_sonnet_tokens_is_twice_sonnet():
    import rolling_cache_reprice as rr
    son = rr.all_cloud_rolling(REPO)["claude-sonnet-5-5"]
    op = rr.all_cloud_rolling(REPO, price_model="claude-opus-5-5", source_model="claude-sonnet-5-5")
    assert list(op) == ["claude-opus-5-5"]
    op = op["claude-opus-5-5"]
    assert set(op) == set(son)
    for k in son:   # Opus list prices are 2x Sonnet's on every component, same minimum cacheable prefix
        assert op[k]["usd_new"] == pytest.approx(2 * son[k]["usd_new"])
        assert op[k]["ok"] == son[k]["ok"] and op[k]["read_new"] == son[k]["read_new"]
        assert op[k]["usd_recorded"] == op[k]["usd_old"]     # the billed Sonnet USD is not carried over
    rd = rr.redo_rolling(REPO, price_model="claude-opus-5-5")
    rs = rr.redo_rolling(REPO)
    assert rd["model"] == "claude-opus-5-5" and rd["recorded_model"] == "claude-sonnet-5-5"
    k = next(iter(rs["redo"]))
    assert rd["redo"][k]["usd_new"] == pytest.approx(2 * rs["redo"][k]["usd_new"])


@pytest.mark.skipif(not HAVE_B3, reason="result files not synced")
def test_opus_rows_say_not_run_and_name_price_source():
    for cid in ("all-cloud-b3-cost-opus-rolling-cache", "hybrid-route-at-start-opus-rolling-cache"):
        v = _entry(cid)["compute"](REPO)["value"]
        assert "PRICED FROM SONNET TOKEN COUNTS, NOT RUN ON OPUS" in v and "Opus accuracy is NOT MEASURED" in v
        assert "claude-opus-5-5" in v and "claude-sonnet-5-5" in v
        assert "results/cloud_prices_anthropic_20261014.json" in v and "fetched 2026-10-14" in v
        assert "https://platform.claude.com/docs/en/about-claude/pricing" in v and chr(0x2014) not in v


@pytest.mark.skipif(not (REPO / "results/traces/agent_step_lengths.parquet").exists(), reason="traces not synced")
def test_allcloud_baseline_opus_row():
    v = _entry("allcloud-baseline-opus-rolling-cache")["compute"](REPO)["value"]
    assert v.startswith("PRICED, NOT RUN") and "claude-opus-5-5: USD" in v and "claude-sonnet-5-5: USD" in v
    assert "per 30-day month" in v and "12 months" in v and "Opus accuracy not measured" in v
    son = _entry("pitch-allcloud-bill-year-rolling-cache")["compute"](REPO)["value"]
    import re
    t = re.search(r"claude-sonnet-5-5: USD ([0-9.]+)/task", v).group(1)
    assert f"USD {t}/task" in son.split("NEW (rolling")[1]     # the Sonnet figure matches the pitch rolling row
