"""Dry run of phase_r1_check (Fig 6.1 quality-side check set) against a stub lab -- real, read-only probe data from
this repo's evaluation/probes/segments.jsonl (never written to), stub server/lab for everything else."""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2s_night2 as n2  # noqa: E402
from stub_lab import StubLab, StubModelInfo, StubServer  # noqa: E402

REPO_PROBES_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "probes"


def test_load_r1_check_probes_finds_all_five():
    with mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR):
        probes = n2.load_r1_check_probes()
    assert [p["id"] for p in probes] == n2.R1_CHECK_PROBE_IDS


def test_load_r1_check_probes_raises_on_missing_id():
    with mock.patch.object(n2, "R1_CHECK_PROBE_IDS", ["rag_01", "not_a_real_probe_id"]), \
         mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR):
        try:
            n2.load_r1_check_probes()
            assert False, "expected RuntimeError for a missing probe id"
        except RuntimeError as e:
            assert "not_a_real_probe_id" in str(e)


def test_phase_r1_check_dry_run_full_sweep_shape():
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1_CHECK_MODELS})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1_check(lab)

    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    start_rows = [r for r in lab.rows if r.get("kind") == "start"]
    # 2 models x 5 probes x 6 ratios x 2 arms x 3 reps = 360 calls
    expected_n = len(n2.R1_CHECK_MODELS) * len(n2.R1_CHECK_PROBE_IDS) * len(n2.R1_CHECK_RATIOS) * len(n2.R1_CHECK_ARMS) * n2.R1_CHECK_REPS
    assert len(call_rows) == expected_n == 360
    assert all(r.get("axis") == "quality" for r in call_rows)
    assert all(r.get("axis") == "quality" for r in start_rows)
    assert {r["arm"] for r in call_rows} == set(n2.R1_CHECK_ARMS)
    assert {r["budget_ratio"] for r in call_rows} == set(n2.R1_CHECK_RATIOS)
    # ratio 1.20 (>= 1.0, never truncates) must have full artifact retention and no truncation
    r120 = [r for r in call_rows if r["budget_ratio"] == 1.20]
    assert all(r["truncating"] is False and r["artifact_fraction_retained"] == 1.0 for r in r120)
    # every row got a score (StubServer always returns ok output, scorers.score must not have crashed)
    assert all("score" in r for r in call_rows)


def test_phase_r1_check_dry_run_no_models_present_does_not_crash():
    lab = StubLab(models={})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1_check(lab)
    assert lab.rows == []
