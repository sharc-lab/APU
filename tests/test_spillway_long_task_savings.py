"""Row spillway-long-task-savings-rolling: Spillway vs all-cloud on the pitch workload (the same per-task basis as row
pitch-allcloud-bill-year-rolling-cache's NEW bill). Offline: replay and pricing of committed files, no call."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numbers_register as nr  # noqa: E402

HAVE = all((REPO / f).exists() for f in nr.SLT_FILES)
pytestmark = pytest.mark.skipif(not HAVE, reason="trace, long-task or price files not present")


def _entry(cid):
    return next(e for e in nr.NUMBER_ENTRIES if e["claim_id"] == cid)


@pytest.fixture(scope="module")
def data():
    return nr.spillway_long_task_data(REPO)


def test_registered_with_its_files():
    e = _entry("spillway-long-task-savings-rolling")
    assert e["compute"] is nr.compute_spillway_long_task_savings_rolling
    for f in ("results/x2_longtask_local_v1.jsonl", "results/longtask_cloud_v1.jsonl",
              "results/traces/agent_step_lengths.parquet", "results/cloud_prices_anthropic_20261014.json"):
        assert f in e["data_files"]


def test_same_workload_and_bill_as_the_pitch_row(data):
    pitch = _entry("pitch-allcloud-bill-year-rolling-cache")["compute"](REPO)["value"]
    assert f"USD {data['per_task']:.5f}/task, USD {data['bill']:,.2f} per {data['days']}-day month" in pitch
    assert data["model"] == nr.PITCH_YEAR_MODEL and data["volume"] == nr.PITCH_YEAR_VOLUME
    assert data["bill"] == pytest.approx(data["per_task"] * data["volume"] * data["days"])


def test_cases_follow_the_measured_rows(data):
    s = data["strata"][nr.SLT_STRATUM]
    cases = {c["label"].split(" (")[0].split(",")[0]: c for c in data["cases"]}
    for c in data["cases"]:
        assert c["bill"] == pytest.approx(data["bill"] * c["usd_share"])
        assert c["saving"] == pytest.approx(1 - c["usd_share"])
        assert 0.0 <= c["saving"] <= 1.0
        for k, (a, n) in data["cloud_acc"].items():
            assert c["acc"][k] == pytest.approx(c["kept_ok"] + c["esc_tasks"] * a / n)
    # perfect check escalates exactly the locally wrong tasks of the matched stratum
    assert cases["perfect check"]["esc_tasks"] == pytest.approx(1 - s["ok"] / s["n"])
    # the post-run check's escalation share is its measured firing on the recorded long-task rows
    pc = next(c for c in data["cases"] if c["label"].startswith("post-run check"))
    assert pc["esc_tasks"] == pytest.approx(s["esc"] / s["n"])
    assert pc["kept_ok"] == pytest.approx(s["kept_ok"] / s["n"])
    # no check keeps everything local: no cloud bill, local accuracy only
    nc = cases["no check"]
    assert nc["bill"] == 0.0 and nc["acc"]["all"] == pytest.approx(s["ok"] / s["n"])
    # the projected window flag selects tasks by size, so its share is a USD share
    wp = next(c for c in data["cases"] if "projected on the matched traces" in c["label"])
    assert wp["usd_share"] == pytest.approx(data["flag_share"])
    assert wp["esc_tasks"] == pytest.approx(data["n_flag"] / data["n_traces"])


def test_value_labels_measured_projected_and_stratum(data):
    v = _entry("spillway-long-task-savings-rolling")["compute"](REPO)["value"]
    for needle in ("STRATUM BASIS", "MEASURED", "PROJECTED", "NOT SUPPORTED", "rollback", "L3",
                   f"{nr.SLT_STRATUM} stratum", "budget_exhausted", "NOT RUN"):
        assert needle in v
    assert "—" not in v
