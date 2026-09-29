"""Dry run of phase_r1d (position pressure, full 11-probe spec) against a stub lab -- real, read-only probe data
from this repo's evaluation/probes/segments.jsonl, stub server/lab for everything else. Also checks the
_phase_position_pressure refactor did not change phase_r1_check's (R1a) own behavior."""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2s_night2 as n2  # noqa: E402
from stub_lab import StubLab, StubModelInfo, StubServer  # noqa: E402

REPO_PROBES_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "probes"


def test_r1d_probe_ids_are_the_11_rag_sea_probes_and_superset_of_r1a():
    assert len(n2.R1D_PROBE_IDS) == 11
    assert set(n2.R1_CHECK_PROBE_IDS) <= set(n2.R1D_PROBE_IDS)
    assert all(pid.startswith(("rag_", "sea_")) for pid in n2.R1D_PROBE_IDS)


def test_load_position_pressure_probes_finds_all_eleven():
    with mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR):
        probes = n2.load_position_pressure_probes(n2.R1D_PROBE_IDS)
    assert [p["id"] for p in probes] == n2.R1D_PROBE_IDS


def test_phase_r1d_dry_run_full_sweep_shape():
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1D_MODELS})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1d(lab)

    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    start_rows = [r for r in lab.rows if r.get("kind") == "start"]
    # 2 models x 11 probes x 6 ratios x 2 arms x 3 reps = 792 calls
    expected_n = len(n2.R1D_MODELS) * len(n2.R1D_PROBE_IDS) * len(n2.R1_CHECK_RATIOS) * len(n2.R1_CHECK_ARMS) * n2.R1_CHECK_REPS
    assert len(call_rows) == expected_n == 792
    assert all(r.get("axis") == "quality" for r in call_rows)
    assert all(r.get("axis") == "quality" for r in start_rows)
    assert all(r["item_id"].startswith("R1d_") for r in call_rows)


def test_phase_r1d_dry_run_no_models_present_does_not_crash():
    lab = StubLab(models={})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR):
        n2.phase_r1d(lab)
    assert lab.rows == []


def test_phase_r1_check_still_works_after_the_shared_refactor():
    """Regression: phase_r1_check (R1a) must still produce exactly its original 180-call shape after being rewritten
    to call the new shared _phase_position_pressure helper."""
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1_CHECK_MODELS})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1_check(lab)
    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    expected_n = len(n2.R1_CHECK_MODELS) * len(n2.R1_CHECK_PROBE_IDS) * len(n2.R1_CHECK_RATIOS) * len(n2.R1_CHECK_ARMS) * n2.R1_CHECK_REPS
    assert len(call_rows) == expected_n == 360  # 2 models x 180 calls/model
    assert all(r["item_id"].startswith("R1check_") for r in call_rows)
