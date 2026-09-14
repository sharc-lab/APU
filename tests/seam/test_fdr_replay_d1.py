"""Tests for D-1 / D-1b / D-1c fdr_replay."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fdr_replay import (  # noqa: E402
    assert_all_tagged,
    assert_no_nan,
    fit_cloud_cost_predictor,
    load_models,
    reconcile_cloud_only_20,
    run,
    x2_per_turn_decomposition,
    x2_replay_check,
)


def test_x2_replay_prints_errors(capsys=None) -> None:
    models = load_models()
    results = x2_replay_check(models)
    assert len(results) == 4
    cpu = [r for r in results if r["placement"] == "cpu-p"]
    assert cpu
    assert "ASSUMED" in cpu[0]["cpu_p_decode_source"] or "LOST" in cpu[0]["cpu_p_decode_source"]
    assert cpu[0]["cancellation"]["note"]
    print("X2_REPLAY_ERRORS:")
    for r in results:
        print(
            f"  {r['session_id'][:8]} {r['placement']} {r['residency']}: "
            f"pred={r['predicted_session_time_s']:.1f} sealed={r['sealed_session_time_s']:.1f} "
            f"err={r['error_s']:.1f}s ({r['error_pct']:.1f}%)"
        )
        assert r["error_s"] is not None
        assert not math.isnan(r["error_s"])


def test_cloud_reconciliation_exact_lookup_test() -> None:
    recon = reconcile_cloud_only_20()
    assert recon["kind"] == "exact_lookup_test"
    assert recon["n_entries"] == 20
    assert abs(recon["predicted_total_usd"] - recon["measured_total_usd"]) < 1e-9
    assert recon["max_abs_error_usd"] < 1e-9


def test_cloud_cost_fit_linear() -> None:
    fit = fit_cloud_cost_predictor()
    assert fit.n >= 20
    assert fit.r2 > 0.5  # C-3 linear confirm on per-entry from_turn points
    assert "ASSUMED(fit=20" in fit.tag
    r0 = fit.r0_scale_200()
    assert r0["value"] > 40.0  # ~0.27 * 200
    assert r0["hi"] > r0["value"] > r0["lo"]


def test_x2_decomposition_holdout() -> None:
    models = load_models()
    decomp = x2_per_turn_decomposition(models)
    assert decomp["holdout"] is True
    assert decomp["no_fit_to_x2"] is True
    assert len(decomp["cells"]) == 4
    assert decomp["resident_miss_verdict"]["finding"]


def test_source_tags_present() -> None:
    art = run(write_outputs=False)
    for row in art["configs"]["rows"]:
        assert_all_tagged(row["objectives"], path=row["config"]["id"])
        if row["config"]["placement"] == "cpu-p":
            tag = row["objectives"]["decode_tok_s_at_2048"]["tag"]
            assert tag.startswith("ASSUMED(from=gpu")
    for d in art["configs"]["decode_residuals"]:
        assert d.get("tag")
    assert art["cloud_reconciliation"]["n_entries"] == 20
    assert art["x2_decomposition"]["holdout"] is True
    h1_rows = None
    # H1 only written when write_outputs; check fit on artifact
    assert art["cloud_fit"].r2 > 0.5


def test_no_nan_in_48_row_table() -> None:
    art = run(write_outputs=False)
    assert art["configs"]["n_configs"] == 48
    assert len(art["configs"]["rows"]) == 48
    assert_no_nan(art["configs"]["rows"])


def test_h1_no_zero_cloud_placeholders() -> None:
    art = run(write_outputs=True)
    rows = {r["label"]: r for r in art["h1"]["rows"]}
    assert rows["R0"]["trace_n"] == 200
    assert rows["R0"]["cloud_usd_total"] > 40.0
    assert rows["R2b"]["cloud_usd_total"] > 1.0  # 45 escalated via fit, not $0.45
    assert rows["R2b"]["cloud_billing"]["n_escalated"] == 45
    for label in ("R0", "R1", "R2a", "R2b"):
        assert "cloud_usd_lo" in rows[label]
        assert "cloud_usd_hi" in rows[label]


if __name__ == "__main__":
    test_x2_replay_prints_errors()
    test_cloud_reconciliation_exact_lookup_test()
    test_cloud_cost_fit_linear()
    test_x2_decomposition_holdout()
    test_source_tags_present()
    test_no_nan_in_48_row_table()
    test_h1_no_zero_cloud_placeholders()
    print("PASS tests/test_fdr_replay_d1.py")
