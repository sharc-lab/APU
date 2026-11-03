"""pitch1 (analysis/pitch1_standard_hybrid.py): the standard hybrid's window check and hard-error fallback on small
fixtures, the target logic, the L3 config pass-through of the post-run check adapter, and the register rows."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "analysis", REPO / "harness"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pitch1_standard_hybrid as P  # noqa: E402


def _row(ok, ests, wall=10.0, calib=1.25, http=False, outcomes=None, empty=False):
    outcomes = outcomes or ["ok"] * len(ests)
    return {"end_state_ok": ok, "turn_wall_s": wall, "token_calib_ratio": calib, "any_http_error": http,
            "final_content_empty": empty,
            "calls": [{"prompt_tokens_est": e, "outcome": o} for e, o in zip(ests, outcomes)]}


def test_window_check_uses_chars4_first_call_by_default():
    r = _row(True, [7000, 8500])
    assert not P.over_window(r, 8192)                       # first call 7000 fits
    assert P.over_window(r, 8192, every_call=True)          # second call 8500 does not
    assert P.over_window(r, 8192, est="tokenizer")          # 7000 x 1.25 = 8750 > 8192
    assert not P.over_window(_row(True, [8192]), 8192)      # equal to the window fits


def test_hard_error_only_on_failures_not_wrong_answers():
    assert not P.hard_error(_row(False, [100]))             # wrong but well-formed: accepted
    assert P.hard_error(_row(True, [100], http=True))
    assert P.hard_error(_row(True, [100, 100], outcomes=["ok", "timeout"]))
    assert P.hard_error(_row(True, [100], empty=True))


def test_standard_hybrid_counts_and_sources():
    tasks = [(1, "a"), (1, "b"), (1, "c"), (1, "d")]
    local = {(1, "a"): _row(True, [1000], wall=5.0), (1, "b"): _row(False, [2000], wall=6.0),
             (1, "c"): _row(False, [9000], wall=7.0), (1, "d"): _row(False, [3000], wall=8.0, http=True)}
    cloud = {k: {"ok": True, "usd": 0.01, "seconds": 4.0} for k in tasks}
    esc = {k: {"ok": True, "usd": 0.03, "seconds": 20.0, "src": "imputed"} for k in tasks}
    r = P.standard_hybrid(tasks, local, 8192, cloud, esc)
    assert (r["kept_local"], r["window_sent"], r["fallback"], r["sent"]) == (2, 1, 1, 2)
    assert r["correct"] == 3 and r["wrong_kept_local"] == 1 and r["wrong"] == 1
    assert r["imputed"] == 1
    assert r["usd_job"] == pytest.approx((0.01 + 0.03) / 4)
    assert r["seconds"] == {(1, "a"): 5.0, (1, "b"): 6.0, (1, "c"): 4.0, (1, "d"): 28.0}
    assert r["median_s"] == pytest.approx(5.5)


def test_targets_need_both_accuracy_and_cost():
    std = {"correct": 80, "usd_job": 0.001}
    assert P.targets({"correct": 81, "usd_job": 0.001}, std) == {"T1": True, "T2": True, "T3": True}
    assert P.targets({"correct": 86, "usd_job": 0.004}, std) == {"T1": False, "T2": True, "T3": False}
    assert P.targets({"correct": 80, "usd_job": 0.0}, std) == {"T1": False, "T2": False, "T3": True}


def test_extra_usd_per_extra_correct():
    assert P.extra_usd_per_extra_correct({"correct": 86, "usd": 0.4}, {"correct": 80, "usd": 0.1}) == pytest.approx(0.05)
    assert P.extra_usd_per_extra_correct({"correct": 80, "usd": 0.4}, {"correct": 80, "usd": 0.1}) is None


def test_adapter_passes_the_l3_config():
    # load the analysis adapter by path: another test may already hold src/dse/post_run_check.py under the bare name
    import importlib.util
    spec = importlib.util.spec_from_file_location("analysis_post_run_check", REPO / "analysis" / "post_run_check.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    from src.dse import l3_invariants as L3
    assert A.make_post_run_check(REPO).config is None              # every existing row: CONFIG_V1
    assert A.make_post_run_check(REPO, L3.CONFIG_TUNED).config is L3.CONFIG_TUNED


def test_register_rows_are_listed():
    import numbers_register as NR
    ids = {e["claim_id"] for e in NR.NUMBER_ENTRIES}
    assert {"pitch1-standard-hybrid-default", "pitch1-standard-hybrid-8192"} <= ids


@pytest.mark.skipif(not (REPO / "results" / "x2_evalset_tasks_real_v2.jsonl").exists(), reason="data not synced")
def test_real_replay_shape_and_no_company_name():
    d = P.compare(REPO)
    for wname in P.WINDOWS:
        w = d["windows"][wname]
        assert w["n"] == 90
        assert w["all_cloud"]["sent"] == 90 and w["local_only"]["usd"] == 0.0
        for k in ("standard_chars4_first", "pipeline_tuned"):
            assert 0 <= w[k]["correct"] <= 90 and w[k]["imputed"] <= w[k]["sent"]
        t = P.window_text(d, wname)
        assert "Spillway" not in t and "—" not in t
        assert f"T1-{wname}" in t
