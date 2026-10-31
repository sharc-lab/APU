"""fit-predictor-v1: the session-start predictor (src/dse/fits_locally.SessionGrowthPredictor) and its scoring
(analysis/fit_predictor.py), on synthetic inputs, plus the register row computing from the committed files."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "analysis", REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.dse import fits_locally as F  # noqa: E402
import fit_predictor as fp  # noqa: E402


def test_projection_rule_is_linear_from_first_k():
    assert F.project_growth([(0, 1000), (1, 1300), (2, 1600)], 10) == 1600 + 300 * 8
    assert F.project_growth([(1, 1000), (2, 900), (3, 800)], 20) == 800      # no negative growth
    assert F.project_growth([(0, 1000), (1, 1300), (2, 1600), (3, 99999)], 2) == 1600   # only the first k read
    with pytest.raises(ValueError):
        F.project_growth([(0, 1), (1, 2)], 5)


def test_ceiling_prefers_measured_then_window():
    p = F.SessionGrowthPredictor()
    assert p.ceiling({"window": 8192}) == 8192
    assert p.ceiling({"window": 8192, "measured_ceiling": 7000}) == 7000
    ctx = {"first_turns": [(0, 6000), (1, 6500), (2, 7000)], "target_turn": 4}
    assert p.predict(ctx, {"window": 8192})["fits"] is True            # 8000 <= 8192
    assert p.predict(ctx, {"window": 8192, "measured_ceiling": 7500})["fits"] is False
    assert isinstance(p, F.FitsLocallyPredictor)


def test_ceiling_model_slot_and_zach_stub():
    class Fixed:
        def ceiling(self, prof):
            return 100

    p = F.SessionGrowthPredictor(ceiling_model=Fixed())
    assert p.predict({"projected_tokens": 101}, {"window": 10 ** 6})["fits"] is False
    with pytest.raises(NotImplementedError):
        F.SessionGrowthPredictor(ceiling_model=F.ZachCeilingPredictor()).predict({"projected_tokens": 1}, {"window": 2})


def test_ground_truth_order():
    assert fp.ground_truth({"context_event": "flagged_reject", "over_loaded_window": False}) == (True, "v2")
    assert fp.ground_truth({"context_event": "none", "over_loaded_window": True}) == (False, "v2")
    assert fp.ground_truth({"context_event": "error_other"})[0] is None
    assert fp.ground_truth({"cut_v2_flagged": False, "over_loaded_window": True}) == (False, "v2")
    assert fp.ground_truth({"cut_v2_flagged": None, "over_loaded_window": True}) == (True, "v1")
    assert fp.ground_truth({})[0] is None


def test_session_units_skip_first_k_and_score():
    def turn(i, p, over):
        return {"record": "r2a_turn", "arm_id": "a", "seed": 1, "attempt_id": "x", "turn_idx": i,
                "final_call_prompt_tokens_calibrated": p, "loaded_context": 1000, "over_loaded_window": over,
                "item_kind": None}
    rows = [turn(i, 500 + 100 * i, 500 + 100 * i > 1000) for i in range(8)]
    rows.append({"record": "r2a_session", "arm_id": "a", "seed": 1, "attempt_id": "x"})
    ex = {}
    us = fp.session_units(REPO, "results/x2_fake.jsonl", rows, {}, ex)
    assert [u["turn_idx"] for u in us] == [3, 4, 5, 6, 7] and not ex
    d = fp.score(us)
    c = d["groups"]["pooled"]["pred"]
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 0, 0, 3)
    assert d["target_met"] is True and d["by_device"]["evo-x2"]["base"] is None


@pytest.mark.skipif(not (REPO / "results/x2_evalset_tasks_real_v2.jsonl").exists(), reason="result files not synced")
def test_register_row_computes():
    import numbers_register as nr
    v = nr.compute_fit_predictor_v1(nr.REPO)["value"]
    # operator decision 2026-10-30: detector-v2 result first, verdict MISSED, pooled target labelled agent-proposed
    assert v.startswith("HEADLINE (operator 2026-10-30): against the runtime log")
    assert "verdict MISSED" in v and "(proposed by the agent, not the operator)" in v
    assert v.index("ground truth detector v2") < v.index("pooled (secondary)")
    assert "baseline (current per-task estimate" in v and "evo-t2s" in v
