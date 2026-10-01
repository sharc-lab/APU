"""Unit tests for src/dse/router.py's EnvelopeRouter, against the stub CloudClient and the real envelope_model
predictions (no mocking of envelope_model -- these exercise the real fits/constants from B5)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from src.cloud.client import CloudClient  # noqa: E402
from src.dse.router import EnvelopeRouter, SILENT_FAILURE_OUTCOMES, estimate_difficulty  # noqa: E402


def _client(tmp_path, spend_cap_usd=50.0):
    return CloudClient(api_key=None, spend_cap_usd=spend_cap_usd, ledger_path=Path(tmp_path) / "ledger.jsonl")


def test_fits_local_routes_local(tmp_path):
    router = EnvelopeRouter(_client(tmp_path), machine="evo-t2s", runtime_policy="llama_ngl99")
    # Real A-24 bisect point well under budget, and well under qwen3-8b's native context -> should fit.
    decision = router.route({"task_id": "ok", "model_id": "qwen3-8b", "context_length": 4096,
                              "prompt_tokens": 300, "full_prompt_tokens": 300, "prompt": "x"})
    assert decision.target == "local"
    assert decision.blocked_silent_failure is False


def test_hard_fail_over_budget_routes_to_cloud_not_local(tmp_path):
    router = EnvelopeRouter(_client(tmp_path), machine="evo-t2s", runtime_policy="llama_ngl99")
    # Real A-24 bisect point above the 47,865 MiB budget -- predict_feasibility must say HARD_FAIL.
    decision = router.route({"task_id": "hf", "model_id": "qwen3-32b", "context_length": 117248,
                              "prompt": "short"})
    assert decision.target != "local"
    assert decision.blocked_silent_failure is True
    assert "HARD_FAIL" in decision.reason


def test_never_routes_local_into_a_silent_failure_outcome(tmp_path):
    # Blade RTX4070 / llama.cpp -ngl 99 is a known SILENT_SPILL cell in make_failure_map.EVIDENCE, and blade
    # has no A-24-equivalent bisection, so feasibility is NOT_MEASURED and the generic cell label must gate it.
    router = EnvelopeRouter(_client(tmp_path), machine="blade_rtx4070", runtime_policy="llama_ngl99")
    decision = router.route({"task_id": "spill", "model_id": "qwen3-8b", "context_length": 40000,
                              "prompt": "x"})
    assert decision.target != "local"
    assert decision.blocked_silent_failure is True
    assert decision.target not in SILENT_FAILURE_OUTCOMES  # target is a route, not an outcome label


def test_truncation_cliff_below_floor_blocks_local(tmp_path):
    # evo-x2 has a real measured K1 effective context of 40,960 for qwen3-8b; requesting a prompt well beyond
    # that, with a low real R1b ratio, must land in truncation_cliff below the default 0.5 floor.
    router = EnvelopeRouter(_client(tmp_path), machine="evo-x2", runtime_policy="ollama_default",
                             quality_floor=0.5)
    decision = router.route({"task_id": "trunc", "model_id": "qwen3-8b", "context_length": 120000,
                              "prompt_tokens": 48000, "full_prompt_tokens": 120000, "prompt": "x"})
    assert decision.target != "local"
    assert decision.blocked_silent_failure is True
    assert "truncation_cliff" in decision.reason


def test_blocked_task_over_cloud_budget_refuses_when_hard_blocked(tmp_path):
    # Spend cap of 0.0 means no cloud call can ever be afforded -- a HARD_FAIL-blocked task must be refused,
    # never silently run in the blocked local configuration.
    router = EnvelopeRouter(_client(tmp_path, spend_cap_usd=0.0), machine="evo-t2s", runtime_policy="llama_ngl99")
    decision = router.route({"task_id": "hf0", "model_id": "qwen3-32b", "context_length": 117248,
                              "prompt": "short"})
    assert decision.target == "refuse"
    assert decision.blocked_silent_failure is True


def test_every_decision_has_a_nonempty_reason(tmp_path):
    router = EnvelopeRouter(_client(tmp_path), machine="evo-t2s", runtime_policy="llama_ngl99")
    tasks = [
        {"task_id": "a", "model_id": "qwen3-8b", "context_length": 4096, "prompt": "x"},
        {"task_id": "b", "model_id": "qwen3-32b", "context_length": 120000, "prompt": "x"},
    ]
    for t in tasks:
        decision = router.route(t)
        assert isinstance(decision.reason, str) and len(decision.reason) > 10


def test_estimate_difficulty_is_bounded_and_monotone():
    short = estimate_difficulty("a short prompt")
    long = estimate_difficulty(" ".join(["word"] * 2000))
    assert 0.0 <= short <= 1.0
    assert 0.0 <= long <= 1.0
    assert long >= short


def test_corunner_present_x2_uses_decode_multiplier_in_latency(tmp_path):
    router_no_hog = EnvelopeRouter(_client(tmp_path), machine="evo-x2", runtime_policy="ollama_default",
                                   co_runner_present=False)
    router_hog = EnvelopeRouter(_client(tmp_path), machine="evo-x2", runtime_policy="ollama_default",
                                co_runner_present=True, co_runner_kind="bandwidth")
    task = {"task_id": "c", "model_id": "qwen3-8b", "context_length": 2000, "prompt": "x"}
    d_no = router_no_hog.route(task)
    d_hog = router_hog.route(task)
    assert d_hog.prediction["latency"]["decode_tok_s"] < d_no.prediction["latency"]["decode_tok_s"]
