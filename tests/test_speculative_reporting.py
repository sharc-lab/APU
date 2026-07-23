"""Tests for speculative reporting aggregation in sweep artifacts."""

from evaluation.sweep import _compute_speculative_report


def test_speculative_report_contains_requested_metrics():
    rows = [
        {
            "policy": "speculative",
            "category": "search_hybrid",
            "speculative_agreed": True,
            "rollback": False,
            "rollback_cloud_tokens": 0,
            "spec_rollback_latency_ms": 0.0,
            "spec_on_latency_ms": 20.0,
            "spec_off_latency_ms": 30.0,
        },
        {
            "policy": "speculative",
            "category": "search_hybrid",
            "speculative_agreed": False,
            "rollback": True,
            "rollback_cloud_tokens": 25,
            "spec_rollback_latency_ms": 4.0,
            "spec_on_latency_ms": 50.0,
            "spec_off_latency_ms": 35.0,
        },
    ]

    report = _compute_speculative_report(rows)

    assert report["agreement_rate_per_category"]["search_hybrid"] == 0.5
    assert report["rollback_cost"]["cloud_tokens"] == 25
    assert report["rollback_cost"]["latency_ms"] == 4.0
    assert report["latency_speculation_on_ms"]["p50"] > 0
    assert report["latency_speculation_off_ms"]["p95"] > 0
