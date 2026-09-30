"""Dry run of phase_r1c (truncation cliff under position pressure) against a stub lab -- real, read-only probe data
from this repo's evaluation/probes/segments.jsonl (never written to), stub server/lab for everything else. R1c is the
art_* truncation-cliff probe set run through the position-pressure machinery, so it reuses _phase_position_pressure
with R1b's probe id list and nothing else changed; these tests also pin that phase_r1_check (R1a) and phase_r1d keep
their exact existing call shapes, which is the load-bearing check that adding R1c did not touch the shared function.
"""
import inspect
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2s_night2 as n2  # noqa: E402
from stub_lab import StubLab, StubModelInfo, StubServer  # noqa: E402

REPO_PROBES_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "probes"


def test_r1c_reuses_r1b_probe_ids_and_the_position_pressure_ratios_arms():
    """R1c invents no probes and no constants of its own: it is exactly R1b's art_* id list crossed with R1a/R1d's
    ratio/arm/rep sweep."""
    assert n2.R1C_PROBE_IDS == n2.R1B_PROBE_IDS
    assert len(n2.R1C_PROBE_IDS) == 10
    assert all(pid.startswith("art_") for pid in n2.R1C_PROBE_IDS)
    assert not set(n2.R1C_PROBE_IDS) & set(n2.R1D_PROBE_IDS)  # a different probe family from R1a/R1d
    assert n2.R1C_MODELS == n2.R1B_MODELS == n2.R1D_MODELS == ["qwen3-8b", "qwen3-14b"]


def test_r1c_is_registered_like_every_other_phase():
    assert n2.PHASE_FN["r1c"] is n2.phase_r1c
    assert "r1c" in n2.PRIO
    # R1a/R1b/R1d are all smoke-gated, so R1c must be too (it starts a real server and runs real calls)
    assert {"r1_check", "r1b", "r1d"} <= n2.SMOKE_GATED_PHASES
    assert "r1c" in n2.SMOKE_GATED_PHASES


def test_load_r1c_probes_finds_all_ten():
    with mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR):
        probes = n2.load_position_pressure_probes(n2.R1C_PROBE_IDS)
    assert [p["id"] for p in probes] == n2.R1C_PROBE_IDS


def test_phase_r1c_small_stub_shape_call_count():
    """Explicit models x probes x ratios x arms x reps arithmetic on a deliberately small stub shape, so the total is
    checkable by hand: 1 model x 2 probes x 3 ratios x 2 arms x 2 reps = 24 calls. The real run is the full shape,
    2 models x 10 art_* probes x 6 ratios x 2 arms x 3 reps = 720 calls (360 per model); see
    test_phase_r1c_dry_run_full_sweep_shape below, which asserts that real total."""
    models = ["qwen3-8b"]
    probe_ids = ["art_01", "art_02"]
    ratios = [1.20, 0.85, 0.40]
    arms = ["LATE", "EARLY"]
    reps = 2
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in models})
    with mock.patch.object(n2, "R1C_MODELS", models), mock.patch.object(n2, "R1C_PROBE_IDS", probe_ids), \
         mock.patch.object(n2, "R1_CHECK_RATIOS", ratios), mock.patch.object(n2, "R1_CHECK_ARMS", arms), \
         mock.patch.object(n2, "R1_CHECK_REPS", reps), \
         mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1c(lab)
    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    expected_n = len(models) * len(probe_ids) * len(ratios) * len(arms) * reps
    assert len(call_rows) == expected_n == 24
    assert {r["arm"] for r in call_rows} == set(arms)
    assert {r["budget_ratio"] for r in call_rows} == set(ratios)
    assert all(r["item_id"].startswith("R1c_") for r in call_rows)


def test_phase_r1c_dry_run_full_sweep_shape():
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1C_MODELS})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1c(lab)

    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    start_rows = [r for r in lab.rows if r.get("kind") == "start"]
    # 2 models x 10 art_* probes x 6 ratios x 2 position arms x 3 reps = 720 calls
    expected_n = len(n2.R1C_MODELS) * len(n2.R1C_PROBE_IDS) * len(n2.R1_CHECK_RATIOS) * len(n2.R1_CHECK_ARMS) * n2.R1_CHECK_REPS
    assert len(call_rows) == expected_n == 720
    assert all(r.get("axis") == "quality" for r in call_rows)
    assert all(r.get("axis") == "quality" for r in start_rows)
    assert all(r["item_id"].startswith("R1c_") for r in call_rows)
    assert {r["arm"] for r in call_rows} == set(n2.R1_CHECK_ARMS)
    assert {r["budget_ratio"] for r in call_rows} == set(n2.R1_CHECK_RATIOS)
    # ratio 1.20 (>= 1.0) never truncates and always retains the whole artifact, in either arm
    r120 = [r for r in call_rows if r["budget_ratio"] == 1.20]
    assert all(r["truncating"] is False and r["artifact_fraction_retained"] == 1.0 for r in r120)
    # the point of the phase: at a truncating ratio the LATE arm holds the artifact past the dropped span while the
    # EARLY arm (R1b's own layout) loses it, so the two arms must not report the same retention
    deep = [r for r in call_rows if r["budget_ratio"] == 0.40]
    late_frac = {r["artifact_fraction_retained"] for r in deep if r["arm"] == "LATE"}
    early_frac = {r["artifact_fraction_retained"] for r in deep if r["arm"] == "EARLY"}
    assert late_frac == {1.0}
    assert early_frac == {0.0}
    assert all("score" in r for r in call_rows)


def test_phase_r1c_dry_run_no_models_present_does_not_crash():
    lab = StubLab(models={})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1c(lab)
    assert lab.rows == []


# ---------------------------------------------------------------- regression: R1a/R1d are untouched
def test_shared_function_signature_unchanged_by_r1c():
    """R1c added no parameter to _phase_position_pressure: its signature is still exactly the one R1a and R1d call."""
    assert list(inspect.signature(n2._phase_position_pressure).parameters) == ["lab", "models", "probe_ids", "section"]
    assert all(p.default is inspect.Parameter.empty
               for p in inspect.signature(n2._phase_position_pressure).parameters.values())


def test_phase_r1_check_shape_unchanged_by_r1c():
    """Same assertion tests/test_r1_check_phase.py and tests/test_r1d_phase.py already make for R1a: 2 models x 5
    probes x 6 ratios x 2 arms x 3 reps = 360 calls, item ids still R1check_*."""
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1_CHECK_MODELS})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1_check(lab)
    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    expected_n = len(n2.R1_CHECK_MODELS) * len(n2.R1_CHECK_PROBE_IDS) * len(n2.R1_CHECK_RATIOS) * len(n2.R1_CHECK_ARMS) * n2.R1_CHECK_REPS
    assert len(call_rows) == expected_n == 360
    assert all(r["item_id"].startswith("R1check_") for r in call_rows)


def test_phase_r1d_shape_unchanged_by_r1c():
    """Same assertion tests/test_r1d_phase.py already makes: 2 models x 11 probes x 6 ratios x 2 arms x 3 reps = 792
    calls, item ids still R1d_*."""
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1D_MODELS})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1d(lab)
    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    expected_n = len(n2.R1D_MODELS) * len(n2.R1D_PROBE_IDS) * len(n2.R1_CHECK_RATIOS) * len(n2.R1_CHECK_ARMS) * n2.R1_CHECK_REPS
    assert len(call_rows) == expected_n == 792
    assert all(r["item_id"].startswith("R1d_") for r in call_rows)


def test_phase_r1b_shape_unchanged_by_r1c():
    """R1c aliases R1B_PROBE_IDS, so R1b's own 480-call shape must be unaffected by that sharing."""
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1B_MODELS})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1b(lab)
    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    expected_n = len(n2.R1B_MODELS) * len(n2.R1B_PROBE_IDS) * len(n2.R1B_RATIOS) * len(n2.R1B_ARM_SUFFIXES) * n2.R1B_REPS
    assert len(call_rows) == expected_n == 480
    assert all(r["item_id"].startswith("R1b_") for r in call_rows)
    assert {r["arm"] for r in call_rows} == set(n2.R1B_ARM_SUFFIXES)
