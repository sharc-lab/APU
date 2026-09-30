"""Unit tests for r1b_controls (needle_deleted / filler_only_cut), the control phase that isolates whether R1b's
truncation-cliff quality drop is caused by the needle leaving the context or by truncation damaging the prompt
regardless. Dry run against a stub lab -- real, read-only probe data from this repo's evaluation/probes/segments.jsonl
(never written to), stub server/lab for everything else, same pattern as tests/test_r1b_phase.py.

No live model or server calls anywhere in this file.
"""
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2s_night2 as n2  # noqa: E402
from stub_lab import StubLab, StubModelInfo, StubServer  # noqa: E402

REPO_PROBES_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "probes"


@pytest.fixture(scope="module")
def probes():
    with mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR):
        return {p["id"]: p for p in n2.load_r1b_probes()}


# ---------------------------------------------------------------- needle_values

def test_needle_values_scalar_expected(probes):
    assert n2._r1bc_needle_values(probes["art_01"]) == ["51847"]


def test_needle_values_span_match_expected(probes):
    assert n2._r1bc_needle_values(probes["art_05"]) == ["47.3", "12.8"]


# ---------------------------------------------------------------- delete_needle

@pytest.mark.parametrize("pid", [f"art_{i:02d}" for i in range(1, 11)])
def test_delete_needle_removes_answer_line_and_nothing_else(pid, probes):
    seg = probes[pid]
    artifact = seg["artifact"].strip()
    result = n2._r1bc_delete_needle(seg)

    orig_lines = artifact.split("\n")
    new_lines = result["artifact"].split("\n")
    idx = result["needle_line_idx"]

    assert len(new_lines) == len(orig_lines) - 1
    assert result["deleted_line"] == orig_lines[idx]
    # every other line survives byte-for-byte, in original order
    assert new_lines == orig_lines[:idx] + orig_lines[idx + 1:]


def test_delete_needle_answer_gone_from_artifact(probes):
    seg = probes["art_01"]
    result = n2._r1bc_delete_needle(seg)
    assert "51847" not in result["artifact"]


def test_delete_needle_disambiguates_recurring_surname(probes):
    """art_06's answer ('Blum') also appears in the attendee list and in an unrelated vote tally line. Only the line
    that actually seconds the motion (what the question asks about) may be deleted; the attendee list and the vote
    tallies must survive untouched."""
    seg = probes["art_06"]
    result = n2._r1bc_delete_needle(seg)
    assert result["deleted_line"] == "  Motion: V. Herrera. Second: T. Blum."
    assert "Attendees: D. Okonkwo (chair), S. Nakagawa, V. Herrera, R. Mäkinen, T. Blum" in result["artifact"]
    assert "Votes" in result["artifact"] and "Blum: yes" in result["artifact"]
    assert "Second: T. Blum" not in result["artifact"]


def test_delete_needle_raises_if_needle_not_found():
    fake_probe = {
        "id": "fake_01",
        "artifact": "line one\nline two\nline three",
        "question": "what is the value?",
        "expected": "nowhere-to-be-found",
    }
    with pytest.raises(ValueError):
        n2._r1bc_delete_needle(fake_probe)


# ---------------------------------------------------------------- filler_only_cut

def test_filler_only_cut_matches_ratio_098_ordinary_cut(probes):
    seg = probes["art_01"]
    artifact, question = seg["artifact"].strip(), seg["question"].strip()
    srv = StubServer(lab=None, mi=None, n_ctx=8192)
    filler = n2.L.ctx_mod.build_filler(n2.R1B_CONTROLS_FILLER, seed=42, count_fn=srv.tokenize)
    base_prompt = f"{artifact}\n\n{filler}\n\n{question}"
    full_tokens = srv.tokenize(base_prompt)

    target_tokens = round(full_tokens * n2.R1B_CONTROLS_CUT_RATIO)
    ordinary_truncated = n2.sc_left_truncate(base_prompt, full_tokens, target_tokens)
    expected_cut = len(base_prompt) - len(ordinary_truncated)

    result = n2._r1bc_filler_only_cut(artifact, filler, question, full_tokens)
    assert result["chars_cut"] == expected_cut
    assert result["chars_cut"] > 0


def test_filler_only_cut_preserves_needle_verbatim_and_position(probes):
    seg = probes["art_05"]
    artifact, question = seg["artifact"].strip(), seg["question"].strip()
    srv = StubServer(lab=None, mi=None, n_ctx=8192)
    filler = n2.L.ctx_mod.build_filler(n2.R1B_CONTROLS_FILLER, seed=42, count_fn=srv.tokenize)
    base_prompt = f"{artifact}\n\n{filler}\n\n{question}"
    full_tokens = srv.tokenize(base_prompt)

    result = n2._r1bc_filler_only_cut(artifact, filler, question, full_tokens)

    for v in n2._r1bc_needle_values(seg):
        assert v in result["prompt"]
    assert result["prompt"].startswith(artifact)
    assert result["prompt"].endswith(question)
    assert artifact in result["prompt"]


def test_filler_only_cut_raises_if_filler_too_small():
    # A large artifact makes the ratio-0.98 char cut far bigger than a near-empty filler can absorb --
    # filler_only_cut must refuse rather than silently eating into the artifact.
    artifact = "A" * 2000
    tiny_filler = "x" * 5
    question = "What is the value?"
    srv = StubServer(lab=None, mi=None, n_ctx=8192)
    base_prompt = f"{artifact}\n\n{tiny_filler}\n\n{question}"
    full_tokens = srv.tokenize(base_prompt)
    with pytest.raises(ValueError):
        n2._r1bc_filler_only_cut(artifact, tiny_filler, question, full_tokens)


# ---------------------------------------------------------------- phase dry run: cell/rep wiring

def test_phase_r1b_controls_dry_run_full_sweep_shape():
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1B_CONTROLS_MODELS})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1b_controls(lab)

    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    start_rows = [r for r in lab.rows if r.get("kind") == "start"]

    # 2 models x 10 probes x 2 conditions x 30 reps = 1200 calls, no more, no less
    expected_n = (len(n2.R1B_CONTROLS_MODELS) * len(n2.R1B_CONTROLS_PROBE_IDS)
                  * len(n2.R1B_CONTROLS_CONDITIONS) * n2.R1B_CONTROLS_REPS)
    assert expected_n == 1200
    assert len(call_rows) == expected_n
    assert len(start_rows) == len(n2.R1B_CONTROLS_MODELS)

    assert all(r.get("axis") == "quality" for r in call_rows)
    assert {r["condition"] for r in call_rows} == set(n2.R1B_CONTROLS_CONDITIONS)
    assert {r["model_id"] for r in call_rows} == set(n2.R1B_CONTROLS_MODELS)
    assert {r["probe_id"] for r in call_rows} == set(n2.R1B_CONTROLS_PROBE_IDS)
    assert all("score" in r for r in call_rows)

    # exactly 30 reps per (model, probe, condition) cell
    from collections import Counter
    cell_counts = Counter((r["model_id"], r["probe_id"], r["condition"]) for r in call_rows)
    assert len(cell_counts) == len(n2.R1B_CONTROLS_MODELS) * len(n2.R1B_CONTROLS_PROBE_IDS) * len(n2.R1B_CONTROLS_CONDITIONS)
    assert set(cell_counts.values()) == {30}

    needle_deleted_rows = [r for r in call_rows if r["condition"] == "needle_deleted"]
    filler_only_rows = [r for r in call_rows if r["condition"] == "filler_only_cut"]
    assert all(r["truncating"] is True and r["chars_cut"] is None and r["deleted_line"] for r in needle_deleted_rows)
    assert all(r["truncating"] is False and r["chars_cut"] is not None and r["deleted_line"] is None
               for r in filler_only_rows)


def test_phase_r1b_controls_dry_run_no_models_present_does_not_crash():
    lab = StubLab(models={})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1b_controls(lab)
    assert lab.rows == []


def test_phase_r1b_controls_resume_skips_done_items():
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1B_CONTROLS_MODELS})
    # pre-mark every rep for one (model, probe, condition) cell as already done
    mid0 = n2.R1B_CONTROLS_MODELS[0]
    pid0 = n2.R1B_CONTROLS_PROBE_IDS[0]
    cond0 = n2.R1B_CONTROLS_CONDITIONS[0]
    for rep in range(n2.R1B_CONTROLS_REPS):
        lab.done.add(f"R1bControls_{mid0}_{pid0}_{cond0}_{rep}")

    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1b_controls(lab)

    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    expected_n = (len(n2.R1B_CONTROLS_MODELS) * len(n2.R1B_CONTROLS_PROBE_IDS)
                  * len(n2.R1B_CONTROLS_CONDITIONS) * n2.R1B_CONTROLS_REPS) - n2.R1B_CONTROLS_REPS
    assert len(call_rows) == expected_n


# ---------------------------------------------------------------- dry-run call-count accounting (stub, no server)

def test_dry_run_reports_exact_total_call_count():
    """Same dry run as above, but the deliverable is the raw call count a real run would make: no live model or
    server calls anywhere in this file, only the StubServer/StubLab fakes."""
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1B_CONTROLS_MODELS})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.L, "PROBES_DIR", REPO_PROBES_DIR), \
         mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1b_controls(lab)
    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    assert len(call_rows) == 1200


# ---------------------------------------------------------------- hour estimate wiring

def test_estimate_hours_includes_r1b_controls():
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1B_CONTROLS_MODELS})
    est = n2.estimate_hours(lab)
    assert "r1b_controls" in est
    assert est["r1b_controls"] > 0
