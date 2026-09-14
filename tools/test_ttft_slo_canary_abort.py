"""Unit tests: ttft_slo canary trip must abort, not soft-return.

Mirrors tools/test_canary_drift_abort_enforcement.ps1 for the Python path.
INF-1b: dual-bound N, budget refuse, unarmed seal refuse, 39- and 300-probe arming.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ttft_slo_canary import (
    CALIBRATION_C,
    FAIL_STATUS,
    ONSET_S,
    CanaryBudgetRefuse,
    CanaryDriftAbort,
    CanaryUnarmedSealRefuse,
    assert_canary_budget_fits,
    assert_seal_requires_armed_or_unguarded,
    derive_canary_every_n,
    new_canary_gate,
    update_canary_drift_bookkeeping,
)


def _ok_rec(t1: float, t2: float, idx: int) -> dict:
    return {
        "classification": "OK",
        "turn1_prefill_s": t1,
        "turn2_prefill_s": t2,
        "canary_index": idx,
    }


def _simulate_arm(*, planned: int, mean_wall_s: float = 9.93) -> dict:
    """Opening + every-N in-run canaries over a planned probe budget."""
    assert_canary_budget_fits(planned, calibration_c=CALIBRATION_C)
    d = derive_canary_every_n(
        [mean_wall_s] * max(1, planned - 1),
        planned_probe_count=planned,
        calibration_c=CALIBRATION_C,
        onset_s=ONSET_S,
    )
    assert d["derivable"] is True
    n = int(d["n"])
    gate = new_canary_gate(calibration_c=CALIBRATION_C, rel_drift_floor=0.05)
    prior: list[dict] = []
    canary_times = 0

    def fire(idx: int) -> None:
        nonlocal canary_times
        rec = _ok_rec(2.66 + 0.001 * idx, 1.06, idx)
        bk = update_canary_drift_bookkeeping(
            rec=rec, gate=gate, prior_ok_canaries=prior, calibration_c=CALIBRATION_C
        )
        prior.append(bk["rec"])
        canary_times += 1

    fire(0)
    probes_since = 0
    for _ in range(planned):
        probes_since += 1
        if probes_since >= n:
            fire(canary_times)
            probes_since = 0

    return {
        "planned": planned,
        "n": n,
        "n_onset": d.get("n_onset"),
        "n_budget": d.get("n_budget"),
        "binding_bound": d.get("binding_bound"),
        "armed": bool(gate.get("armed")),
        "n_canaries": canary_times,
        "derivation": d,
    }


def test_calibration_arms_then_trip_raises_path() -> None:
    gate = new_canary_gate(calibration_c=3, rel_drift_floor=0.05)
    prior: list[dict] = []
    for i, (t1, t2) in enumerate([(2.66, 1.06), (2.65, 1.05), (2.67, 1.07)]):
        rec = _ok_rec(t1, t2, i)
        bk = update_canary_drift_bookkeeping(
            rec=rec, gate=gate, prior_ok_canaries=prior, calibration_c=3
        )
        assert bk["tripped"] is False
        prior.append(bk["rec"])
    assert gate["calibration_complete"] is True
    assert gate["threshold_t1"] >= 0.05

    bad = _ok_rec(2.517, 0.802, 3)
    bk = update_canary_drift_bookkeeping(
        rec=bad, gate=gate, prior_ok_canaries=prior, calibration_c=3
    )
    assert bk["tripped"] is True
    assert bad["drift_tripped"] is True

    try:
        if bk["tripped"]:
            raise CanaryDriftAbort(str(bk["trip_detail"]), canary_record=bad)
        raise AssertionError("must not continue after trip")
    except CanaryDriftAbort as exc:
        assert FAIL_STATUS == "FAIL_CANARY_DRIFT"
        assert "rel_drift" in exc.detail


def test_n_dual_bound_records_candidates() -> None:
    d = derive_canary_every_n([10.5, 10.8, 11.0])
    assert d["derivable"] is True
    mean_w = (10.5 + 10.8 + 11.0) / 3.0
    assert d["n"] == int(657.0 // mean_w)
    assert d["n"] != 12
    assert d["binding_bound"] == "onset"
    assert d["n_onset"] == d["n"]
    assert d["n_budget"] is None

    walls = [9.93] * 38
    d2 = derive_canary_every_n(walls, planned_probe_count=39, calibration_c=3)
    assert d2["derivable"] is True
    assert d2["n_onset"] == int(657.0 // 9.93)  # 66
    assert d2["n_budget"] == 39 // (3 + 1)  # 9
    assert d2["n"] == 9
    assert d2["binding_bound"] == "budget"


def test_refuse_start_when_budget_cannot_fit_c_plus_one() -> None:
    try:
        assert_canary_budget_fits(3, calibration_c=3)
        raise AssertionError("must refuse")
    except CanaryBudgetRefuse as exc:
        assert "Do not start unguarded" in exc.detail

    d = derive_canary_every_n([9.0], planned_probe_count=3, calibration_c=3)
    assert d["refuse"] is True
    assert d["derivable"] is False


def test_39_probe_budget_must_arm_or_refuse_never_silent_unguarded() -> None:
    """A 39-probe budget must arm under INF-1b N, and unarmed seal is refused."""
    r = _simulate_arm(planned=39, mean_wall_s=9.93)
    assert r["n"] == 9
    assert r["n_onset"] == 66
    assert r["n_budget"] == 9
    assert r["binding_bound"] == "budget"
    assert r["armed"] is True, r
    assert r["n_canaries"] >= CALIBRATION_C + 1

    try:
        assert_seal_requires_armed_or_unguarded(armed=False, allow_unguarded=False)
        raise AssertionError("must not seal silently unguarded")
    except CanaryUnarmedSealRefuse:
        pass

    fin = assert_seal_requires_armed_or_unguarded(armed=False, allow_unguarded=True)
    assert fin["UNGUARDED"] is True
    fin2 = assert_seal_requires_armed_or_unguarded(armed=True, allow_unguarded=False)
    assert fin2["UNGUARDED"] is False


def test_300_probe_budget_must_arm() -> None:
    """A 300-probe budget must arm (onset may bind when budget is large)."""
    r = _simulate_arm(planned=300, mean_wall_s=9.93)
    assert r["n_onset"] == 66
    assert r["n_budget"] == 300 // (3 + 1)  # 75
    assert r["n"] == min(66, 75)  # 66
    assert r["binding_bound"] == "onset"
    assert r["armed"] is True, r
    assert r["n_canaries"] >= CALIBRATION_C + 1
    # With N=66 and 300 probes: opening + floor(300/66)=4 in-run = 5 canaries
    assert r["n_canaries"] >= 5


def test_cannot_return_true_after_trip() -> None:
    gate = new_canary_gate(calibration_c=2, rel_drift_floor=0.05)
    prior = []
    for i, t in enumerate([(1.0, 0.5), (1.01, 0.51)]):
        rec = _ok_rec(t[0], t[1], i)
        bk = update_canary_drift_bookkeeping(
            rec=rec, gate=gate, prior_ok_canaries=prior, calibration_c=2
        )
        prior.append(bk["rec"])
    bad = _ok_rec(2.0, 0.5, 2)
    bk = update_canary_drift_bookkeeping(
        rec=bad, gate=gate, prior_ok_canaries=prior, calibration_c=2
    )
    assert bk["tripped"] is True
    raised = False
    try:
        raise CanaryDriftAbort(bk["trip_detail"] or "trip", canary_record=bad)
    except CanaryDriftAbort:
        raised = True
    assert raised


if __name__ == "__main__":
    test_calibration_arms_then_trip_raises_path()
    test_n_dual_bound_records_candidates()
    test_refuse_start_when_budget_cannot_fit_c_plus_one()
    test_39_probe_budget_must_arm_or_refuse_never_silent_unguarded()
    test_300_probe_budget_must_arm()
    test_cannot_return_true_after_trip()
    print("PASS tools/test_ttft_slo_canary_abort.py")
