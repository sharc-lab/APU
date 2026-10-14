"""Tests for harness/agent_trace_replay.py (docs/EVALSET_DESIGN.md part (c)): the exported schedule matches the parquet's
uncensored sources, the sampler is seeded and deterministic, filler sizing, the positive-control seed, fidelity."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
import agent_trace_replay as R  # noqa: E402

PARQUET = REPO / "results" / "traces" / "agent_step_lengths.parquet"


@pytest.fixture(scope="module")
def schedule():
    return R.load_schedule()


def test_datasets_match_trace_weighted_pack():
    import trace_weighted_pack as twp   # analysis/ is on pytest's pythonpath
    assert tuple(R.UNCENSORED_DATASETS) == tuple(twp.UNCENSORED_DATASETS)


def test_json_equals_parquet_export(schedule):
    pytest.importorskip("pandas")
    if not PARQUET.exists():
        pytest.skip("parquet not present")
    assert R.build_schedule_from_parquet(PARQUET) == schedule


def test_only_uncensored_sources(schedule):
    assert {t["dataset"] for t in schedule["trajectories"]} <= set(R.UNCENSORED_DATASETS)
    assert schedule["n_trajectories"] == len(schedule["trajectories"])
    for t in schedule["trajectories"]:
        assert len(t["tokens_qwen"]) == len(t["tokens_llama"]) >= 1
        assert t["tokens_qwen"] == sorted(t["tokens_qwen"])   # cumulative prompt lengths never shrink in this data


def test_sampler_seeded_deterministic_and_eligible(schedule):
    a = R.sample_trajectory(schedule, 20260901)
    assert a == R.sample_trajectory(schedule, 20260901)
    ids = {R.sample_trajectory(schedule, s)["id"] for s in range(20260901, 20260921)}
    assert len(ids) > 5
    assert all(len(R.sample_trajectory(schedule, s)["tokens_qwen"]) >= R.MIN_STEPS for s in range(100))
    pool = R.eligible(schedule)
    assert [t["id"] for t in pool] == sorted(t["id"] for t in pool)
    with pytest.raises(ValueError):
        R.sample_trajectory(schedule, 1, min_steps=10 ** 6)


def test_order_of_input_does_not_change_sample(schedule):
    rev = dict(schedule, trajectories=list(reversed(schedule["trajectories"])))
    assert R.sample_trajectory(rev, 7)["id"] == R.sample_trajectory(schedule, 7)["id"]


def test_targets_per_tokenizer(schedule):
    t = R.sample_trajectory(schedule, 20260901)
    assert R.targets_for(t, "llama3.1:8b") == t["tokens_llama"][:40]
    assert R.targets_for(t, "qwen3:14b") == t["tokens_qwen"][:40]
    assert len(R.targets_for(t, "qwen3:14b", max_turns=5)) == 5
    assert R.token_column("qwen3-4b-2507-tools") == "tokens_qwen"


def test_negative_control_window_never_reached(schedule):
    peak = max(max(t["tokens_qwen"][:R.MAX_TURNS] + t["tokens_llama"][:R.MAX_TURNS]) for t in R.eligible(schedule))
    assert peak < 0.8 * 131072


def test_positive_seed(schedule):
    s = R.positive_seed(schedule)
    t = R.sample_trajectory(schedule, s)
    for m in ("llama3.1:8b", "qwen3:14b"):
        assert R.targets_for(t, m)[R.POSITIVE_BY_TURN - 1] > R.POSITIVE_MARGIN * R.POSITIVE_WINDOW
    assert s == R.positive_seed(schedule)
    for earlier in range(R.SEED_SEARCH_START, s):
        e = R.sample_trajectory(schedule, earlier)
        assert not all(R.targets_for(e, m)[R.POSITIVE_BY_TURN - 1] > R.POSITIVE_MARGIN * R.POSITIVE_WINDOW
                       for m in ("llama", "qwen"))


def test_filler_sizing():
    assert R.filler_tokens_for_turn(10000, 3000, 200, None) == 10000 - 3000 - 200 - 8
    assert R.filler_tokens_for_turn(10000, 3000, 200, 1.25) == 8000 - 3000 - 200 - 8
    assert R.filler_tokens_for_turn(1000, 3000, 200, 1.0) == 0


def test_fidelity():
    rows = [{"replay_target_tokens": 1000, "replay_reachable": True, "replay_prompt_calibrated": 1050},
            {"replay_target_tokens": 1000, "replay_reachable": True, "replay_prompt_calibrated": 1200},
            {"replay_target_tokens": 500, "replay_reachable": False, "replay_prompt_calibrated": 3000},
            {"turn_idx": 1}]
    f = R.fidelity(rows)
    assert f == {"n_reachable": 2, "n_within": 1, "rate": 0.5, "tol": 0.10}
    assert R.fidelity([])["rate"] is None


def test_cli_without_export_does_nothing(capsys):
    assert R.main([]) == 2
