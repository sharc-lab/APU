"""Unit tests for src/dse/baselines.py, against the stub CloudClient only."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cloud.client import CloudClient, CLOUD_MODELS  # noqa: E402
from src.dse.baselines import (  # noqa: E402
    all_cloud_cheap, all_cloud_strong, all_local, calibrate_routellm_threshold, routellm_bert_baseline,
)

TASKS = [{"task_id": "t1", "prompt": "short prompt"}, {"task_id": "t2", "prompt": "another short prompt"}]


def _client(tmp_path):
    return CloudClient(api_key=None, ledger_path=Path(tmp_path) / "ledger.jsonl")


def test_all_local_never_calls_cloud(tmp_path):
    results = all_local(TASKS)
    assert all(r.target == "local" for r in results)
    assert all(r.cost_usd == 0.0 for r in results)
    assert [r.task_id for r in results] == ["t1", "t2"]


def test_all_cloud_cheap_uses_cheap_model(tmp_path):
    client = _client(tmp_path)
    results = all_cloud_cheap(TASKS, client)
    assert all(r.target == "cloud_cheap" for r in results)
    assert all(r.stub is True for r in results)  # api_key=None, default mode -> stub mode
    assert all(r.cost_usd == 0.0 for r in results)  # stub calls never cost anything


def test_all_cloud_strong_uses_strong_model(tmp_path):
    client = _client(tmp_path)
    results = all_cloud_strong(TASKS, client)
    assert all(r.target == "cloud_strong" for r in results)
    assert all(r.stub is True for r in results)


def test_routellm_bert_skips_cleanly_when_package_absent(tmp_path):
    # routellm is not installed in this environment (checked 2026-09-30) -- every task must come back
    # target="skipped" with a clear, correct reason, never a crash and never a silent fake routing decision.
    client = _client(tmp_path)
    results = routellm_bert_baseline(TASKS, client)
    assert all(r.target == "skipped" for r in results)
    assert all("routellm is not installed" in r.reason for r in results)
    assert all("OPENAI_API_KEY" in r.reason or "OpenAI embeddings" in r.reason for r in results)


def test_calibrate_routellm_threshold_stays_within_budget(tmp_path):
    client = _client(tmp_path)
    win_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    calib = calibrate_routellm_threshold(win_rates, client)
    thr = calib["threshold"]
    frac_strong = sum(1 for w in win_rates if w >= thr) / len(win_rates)
    expected_cost = (frac_strong * calib["strong_cost_usd_per_task"]
                      + (1 - frac_strong) * calib["cheap_cost_usd_per_task"])
    assert expected_cost <= calib["budget_per_task_usd"] + 1e-9
    assert calib["spend_cap_usd"] == client.spend_cap_usd
    assert "NOT a live traffic calibration" in calib["note"]


def test_calibrate_routellm_threshold_empty_sample_is_safe(tmp_path):
    client = _client(tmp_path)
    calib = calibrate_routellm_threshold([], client)
    assert calib["threshold"] == 1.0  # default: never route to strong when there's nothing to calibrate on
    assert calib["n_sample"] == 0
