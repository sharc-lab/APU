"""src/dse/route_at_start.py with the measured evo-x2 and evo-t2s profiles (register rows), and the committed profile
file matching what the register and the B3 v2 file give."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.dse import route_at_start as R

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prof():
    text = (REPO / "docs" / "NUMBERS_REGISTER.md").read_text(encoding="utf-8")
    with open(REPO / "results" / "x2_evalset_tasks_real_v2.jsonl", encoding="utf-8") as fh:
        b3 = [json.loads(line) for line in fh if line.strip()]
    return R.build_profiles(text, b3)


def test_profiles_from_the_register(prof):
    x2, t2s = prof["devices"]["evo-x2"], prof["devices"]["evo-t2s"]
    assert x2["default_window"] == 40960 and x2["settings"][0]["id"] == "kv_q8_0"
    assert x2["settings"][0]["window"] == 40960
    assert t2s["default_window"] == 4096
    assert [s["id"] for s in t2s["settings"]] == ["igpu", "default"] and t2s["settings"][0]["window"] == 32768
    assert not any(s["id"] == "kv_q8_0" for s in t2s["settings"])      # not measured there
    c = prof["calibration"]
    assert 1.19 < c["ratio"] < 1.21 and c["growth_p90"] > c["growth_median"] > 0
    assert prof["cloud"]["usd_per_task_ledger"] == pytest.approx(0.3285 / 90, abs=1e-6)


def test_committed_profile_file_is_current(prof):
    assert R.load_profiles() == prof


def test_short_task_runs_locally_on_both(prof):
    for dev in ("evo-x2", "evo-t2s"):
        d = R.route(prof, dev, est_now=2000, expected_turns=1, cloud_budget_usd=10)
        assert d["route"] == "local", d
    assert R.route(prof, "evo-t2s", 2000, 1, 10)["setting"] == "igpu"
    assert R.route(prof, "evo-x2", 2000, 1, 10)["setting"] == "kv_q8_0"


def test_long_history_on_the_intel_laptop_needs_the_igpu_window(prof):
    d = R.route(prof, "evo-t2s", est_now=9000, expected_turns=1, cloud_budget_usd=10)
    assert d["route"] == "local" and d["setting"] == "igpu" and d["need_tokens"] > 4096
    assert "4,096 would cut it off" in d["reason"]
    assert R.route(prof, "evo-x2", 9000, 1, 10)["setting"] == "kv_q8_0"


def test_over_every_window_goes_to_the_cloud_or_says_cut_off(prof):
    d = R.route(prof, "evo-t2s", est_now=40000, expected_turns=1, cloud_budget_usd=10)
    assert d["route"] == "cloud" and d["model"] == "claude-haiku-5-5"
    assert R.route(prof, "evo-x2", 40000, 1, 10)["route"] == "cloud"
    broke = R.route(prof, "evo-t2s", 40000, 1, cloud_budget_usd=0.0)
    assert broke["route"] == "local" and broke.get("will_be_cut_off") and "cut off" in broke["reason"]


def test_need_and_margin(prof):
    c = prof["calibration"]
    need = R.context_need(1000, 2, c)
    assert need == math.ceil(c["ratio"] * (1000 + 2 * c["growth_p90"])) + 384
    assert R.fits(3891, 4096) and not R.fits(3892, 4096)
