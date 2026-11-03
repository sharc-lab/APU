"""Unit tests for analysis/session_exact_tests.py (one-sided Fisher exact tests on sessions)."""
import sys
from math import comb
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import session_exact_tests as se  # noqa: E402


def test_all_versus_none_is_one_over_binomial():
    assert se.fisher_one_sided_greater(5, 5, 0, 5) == pytest.approx(1 / 252)
    assert se.fisher_one_sided_greater(10, 10, 0, 10) == pytest.approx(1 / comb(20, 10))


def test_matches_a_hand_computed_tail():
    # 4/5 vs 0/5: margins 4 events in 10; P(X >= 4) = C(5,4)C(5,0)/C(10,4) = 5/210.
    assert se.fisher_one_sided_greater(4, 5, 0, 5) == pytest.approx(5 / 210)
    # equal cells: the tail includes the whole distribution's upper half and more.
    assert se.fisher_one_sided_greater(0, 5, 0, 5) == pytest.approx(1.0)


def test_matches_scipy_when_available():
    stats = pytest.importorskip("scipy.stats")
    for a, n1, b, n0 in [(5, 5, 0, 5), (3, 5, 1, 5), (8, 10, 2, 10), (2, 15, 0, 15)]:
        _odds, p = stats.fisher_exact([[a, n1 - a], [b, n0 - b]], alternative="greater")
        assert se.fisher_one_sided_greater(a, n1, b, n0) == pytest.approx(p)


def test_bad_counts_raise():
    with pytest.raises(ValueError):
        se.fisher_one_sided_greater(6, 5, 0, 5)


def test_degenerate():
    assert se.is_degenerate(5, 5, 0, 5)
    assert se.is_degenerate(0, 5, 0, 5)
    assert not se.is_degenerate(4, 5, 0, 5)


def _rep(cells):
    """cells: {"tier|model": list of after_window_exceeded flags}."""
    return {"table": {k: {"sessions": [{"after_window_exceeded": f} for f in v]} for k, v in cells.items()}}


def test_exact_tests_picks_only_degenerate_nonzero_cells():
    rep = _rep({"ollama_default|m": [False] * 5, "num_ctx_4096|m": [True] * 5,
                "num_ctx_8192|m": [True] * 4 + [False], "num_ctx_32768|m": [False] * 5})
    t = se.exact_tests(rep)
    assert [(d["tier"], d["a"], d["n1"], d["b"], d["n0"]) for d in t] == [("num_ctx_4096", 5, 5, 0, 5)]
    assert t[0]["p"] == pytest.approx(1 / 252)
    assert len(se.exact_tests(rep, degenerate_only=False)) == 3


def test_format():
    rep = _rep({"ollama_default|m": [False] * 5, "num_ctx_4096|m": [True] * 5})
    s = se.format_tests("run", se.exact_tests(rep))
    assert s == "run: m num_ctx_4096 5/5 vs default 0/5 sessions, excess +1.000, p = 0.00397"
    assert se.format_tests("run", []) == "run: no degenerate cell"
