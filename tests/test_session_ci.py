"""analysis/session_ci.py: the session-level bootstrap used by rows headline-ci-session-rescue,
x2-evalset-tasks-real-v2-silent-total and headline-ci-session-hybrid (pure parts only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import session_ci as sc  # noqa: E402


def test_point_estimate_is_pooled_and_ci_brackets_it():
    cells = {"a": [(9, 10), (5, 10), (10, 10)], "b": [(1, 2), (2, 2)]}
    b = sc.session_bootstrap_prop(cells, n_boot=2000)
    assert (b["k"], b["n"]) == (27, 34)
    assert b["n_sessions"] == 5 and b["n_strata"] == 2
    lo, hi = b["ci95"]
    assert lo <= b["p"] <= hi


def test_all_or_nothing_has_no_bootstrap_variance():
    assert sc.session_bootstrap_prop({"a": [(0, 5), (0, 7)]}, n_boot=500)["ci95"] == (0.0, 0.0)
    assert sc.session_bootstrap_prop({"a": [(5, 5), (7, 7)]}, n_boot=500)["ci95"] == (1.0, 1.0)


def test_seeded_and_reproducible():
    cells = {"a": [(3, 10), (6, 10), (8, 10)]}
    assert sc.session_bootstrap_prop(cells, n_boot=300) == sc.session_bootstrap_prop(cells, n_boot=300)


def test_strata_are_resampled_within_themselves():
    # stratum a is all success, b all failure: each resample keeps one session per stratum, so the pooled rate is fixed
    b = sc.session_bootstrap_prop({"a": [(4, 4)], "b": [(0, 4)]}, n_boot=200)
    assert b["ci95"] == (0.5, 0.5)


def test_cells_from_units_groups_by_tier_and_seed():
    units = [{"tier": "t1", "seed": 1, "ok": True}, {"tier": "t1", "seed": 1, "ok": False},
             {"tier": "t1", "seed": 2, "ok": True}, {"tier": "t2", "seed": 1, "ok": False}]
    assert sc.cells_from_units(units, lambda u: u["ok"]) == {"t1": [(1, 2), (1, 1)], "t2": [(0, 1)]}


def test_hybrid_bootstrap_paired_and_ratio_undefined_counted():
    sess = [{"seed": 1, "n": 30, "hyb_ok": 30, "hyb_usd": 0.24, "ac_ok": 30, "ac_usd": 0.30, "nv_ok": 17},
            {"seed": 2, "n": 30, "hyb_ok": 28, "hyb_usd": 0.25, "ac_ok": 30, "ac_usd": 0.30, "nv_ok": 20}]
    b = sc.hybrid_bootstrap(sess, n_boot=1000)
    assert b["hyb_ok"] == 58 and b["nv_ok"] == 37 and b["n"] == 60
    assert abs(b["point"]["ratio"] - 23 / 2) < 1e-9
    assert abs(b["point"]["saving"] - (1 - 0.49 / 0.60)) < 1e-9
    assert 0 < b["ratio_undefined"] < 1000  # resamples drawing seed 1 twice have 0 hybrid wrong
    assert b["ci95"]["ac"] == (1.0, 1.0)
