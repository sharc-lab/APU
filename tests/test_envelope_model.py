"""Unit tests for analysis/envelope_model.py, checked against real numbers pulled from the claims/results this
model is fit on (see envelope_model.py's own constant docstrings for each number's exact source)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import envelope_model as em  # noqa: E402


def test_feasibility_fits_below_budget_qwen3_32b():
    # Real bisect point: (111104, 46494.0) MiB, well below the 47,865 MiB budget -- must return FITS with a
    # positive margin close to 47865 - 46494 = 1371 MiB (the fitted line passes very close to this real point,
    # R2=1.0 on 7 points since the relation is genuinely linear in llama.cpp's own projection).
    result = em.predict_feasibility("evo-t2s", "qwen3-32b", 111104, runtime_policy="llama_ngl99")
    assert result["status"] == "FITS"
    assert result["measured"] is True
    assert abs(result["margin_mib"] - (em.T2S_VULKAN_BUDGET_MIB - 46494.0)) < 5.0


def test_feasibility_hard_fail_above_budget_qwen3_32b():
    # Real bisect point: (117248, 48036.0) MiB, above the 47,865 MiB budget -- claim A-24 says this is a hard
    # vkAllocateMemory failure under -ngl 99, which make_failure_map's EVIDENCE table labels HARD_FAIL.
    result = em.predict_feasibility("evo-t2s", "qwen3-32b", 117248, runtime_policy="llama_ngl99")
    assert result["status"] == "HARD_FAIL"
    assert result["margin_mib"] < 0


def test_feasibility_default_fit_is_silent_spill_then_crash():
    # Same over-budget context, but under default-fit runtime_policy claim A-25 says the outcome is a silent
    # partial CPU offload, not an immediate hard failure -- make_failure_map's label for this cell.
    result = em.predict_feasibility("evo-t2s", "qwen3-32b", 117248, runtime_policy="llama_default_fit")
    assert "SILENT_SPILL" in result["status"]


def test_feasibility_x2_not_measured():
    # No A-24-equivalent bisection exists for evo-x2 anywhere in this repo.
    result = em.predict_feasibility("evo-x2", "qwen3-8b", 8192)
    assert result["status"] == "NOT_MEASURED"
    assert result["measured"] is False


def test_feasibility_unknown_model_not_measured():
    result = em.predict_feasibility("evo-t2s", "not-a-real-model", 8192)
    assert result["status"] == "NOT_MEASURED"


def test_latency_t2s_measured_flag_true():
    result = em.predict_latency("evo-t2s", "llama_ngl99", 12802)
    assert result["measured"] is True
    # Real measured TTFT at prompt_tokens=12802 (near-budget n_ctx) clusters 209.97-229.02s; the fitted line
    # must land inside that real range, not wildly outside it.
    assert 190 <= result["ttft_s"] <= 240


def test_latency_x2_measured_flag_false():
    result = em.predict_latency("evo-x2", "ollama_default", 12802)
    assert result["measured"] is False
    assert "cross-machine extrapolation" in result["note"]


def test_latency_co_runner_multiplier_applied():
    base = em.predict_latency("evo-t2s", "llama_ngl99", 2000, co_runner=False)
    with_corunner = em.predict_latency("evo-t2s", "llama_ngl99", 2000, co_runner=True)
    ratio = with_corunner["ttft_s"] / base["ttft_s"]
    # H2_CORUNNER_TTFT_MULTIPLIER point estimate is 1.42 (A-23, real measured evo-t2s CPU co-runner TTFT ratio)
    assert abs(ratio - em.H2_CORUNNER_TTFT_MULTIPLIER["point_estimate"]) < 1e-3
    assert with_corunner["co_runner_multiplier_source"].startswith("A-23")


def test_effective_context_flag_is_assumed_qwen3_8b():
    # No K1/R2 JSONL present locally -- every effective-context return must be flagged ASSUMED.
    result = em.predict_effective_context("evo-t2s", "ollama_default", "qwen3-8b")
    assert result["flag"] == "ASSUMED"
    assert result["measured"] is False
    # 64 GiB evo-t2s falls in the 48GiB+ Ollama tier (262,144) but qwen3-8b's own native cap (40,960, claim
    # A-28, MEASURED) is the binding constraint -- min() must pick the smaller one.
    assert result["effective_context"] == 40960
    assert "MEASURED" in result["native_ctx_source"]


def test_effective_context_ollama_tier_picks_smallest_tier_for_low_vram():
    # Directly exercise the tier table: a model with a very large native ctx on a machine reported at < 24 GiB
    # (blade_rtx4070, 8 GiB) must be capped at the 4,096 tier, not its native context.
    result = em.predict_effective_context("blade_rtx4070", "ollama_default", "llama33-70b")
    assert result["effective_context"] == 4096


def test_quality_regime_full_when_within_effective_context():
    result = em.predict_quality_regime(1000, 1000, effective_context=8192)
    assert result["regime"] == "full"
    assert result["predicted_score"] == 1.0


def test_quality_regime_truncation_cliff_matches_r1b_ratio_098():
    # Real R1b baseline-arm number at ratio 0.98: 3/10 probes correct (art_08/09/10 only) = 0.3, both machines
    # (docs/FINDINGS.md R1b evaluation audit section, transcribed into R1B_BASELINE_SCORE_BY_RATIO).
    result = em.predict_quality_regime(980, 1000, effective_context=500)
    assert result["regime"] == "truncation_cliff"
    assert result["nearest_measured_ratio"] == 0.98
    assert result["predicted_score"] == 0.3


def test_quality_regime_truncation_cliff_low_ratio_is_zero():
    result = em.predict_quality_regime(400, 1000, effective_context=100)
    assert result["predicted_score"] == 0.0


def test_failure_silence_reuses_make_failure_map_evidence_directly():
    from make_failure_map import EVIDENCE
    result = em.predict_failure_silence("evo-t2s", "llama_ngl99")
    expected_outcome = EVIDENCE[("T2S (Intel, Vulkan)", "llama.cpp -ngl 99")][0]
    assert result["outcome"] == expected_outcome == "HARD_FAIL"


def test_failure_silence_blade_silent_spill():
    result = em.predict_failure_silence("blade_rtx4070", "llama_ngl99")
    assert result["outcome"] == "SILENT_SPILL"


def test_failure_silence_unrecognized_cell_not_measured():
    result = em.predict_failure_silence("evo-x2", "llama_fit_off")
    assert result["outcome"] == "NOT_MEASURED"


def test_predict_combines_all_five():
    p = em.predict("evo-t2s", "llama_ngl99", "qwen3-32b", 111104)
    assert p.feasibility["status"] == "FITS"
    assert p.latency["measured"] is True
    assert p.effective_context["flag"] == "ASSUMED"
    assert p.quality_regime["regime"] in ("full", "truncation_cliff")
    assert p.failure_silence["outcome"] == "HARD_FAIL"  # cell outcome label regardless of this call's own fit


def test_budget_boundary_fit_r2_near_one_on_real_points():
    # Sanity: the A-24 bisect points are (by claim A-24's own text) extremely linear near the boundary --
    # the fit's R2 per model should be very high (>0.999), confirming the linear form is a defensible choice,
    # not an arbitrary one.
    for model_id, fit in em.T2S_BUDGET_FITS.items():
        assert fit["r2"] > 0.999, f"{model_id} r2={fit['r2']}"
