"""Tests for sweep comparison trajectory histogram utilities."""

from evaluation.sweep import _trajectory_histogram


def test_trajectory_histogram_accumulates_cloud_spend():
    rows = [
        {"policy": "cascade", "budget_level": 0.1, "trajectory_bucket": "early", "cloud_tokens_spent": 10},
        {"policy": "cascade", "budget_level": 0.1, "trajectory_bucket": "mid", "cloud_tokens_spent": 20},
        {
            "policy": "budget_aware_cascade",
            "budget_level": 0.1,
            "trajectory_bucket": "late",
            "cloud_tokens_spent": 7,
        },
    ]

    hist = _trajectory_histogram(rows, ["cascade", "budget_aware_cascade"], [0.1])

    assert hist["cascade"]["0.1"]["early"] == 10
    assert hist["cascade"]["0.1"]["mid"] == 20
    assert hist["cascade"]["0.1"]["late"] == 0
    assert hist["budget_aware_cascade"]["0.1"]["late"] == 7
