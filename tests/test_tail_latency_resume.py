"""
Byte-identical resume test for tail_latency_instrument._run_all.

Verifies that interrupting a sweep partway through and resuming produces
a _compute_tail_records() output that is JSON-identical to a complete
uninterrupted run with the same deterministic probe stub.

The stub returns a value that is a fixed function of (task_id, condition,
sample_index), so each sample is uniquely determined regardless of call order.
This makes the byte-identity and ordering tests meaningful: dropped, duplicated,
or reordered samples produce different percentile arrays and thus different JSON.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import harness.tail_latency_instrument as m


# ── Small test config ───────────────────────────────────────────────────────

SMALL_TASKS = list(m.TASKS.keys())[:3]   # "CH-01", "CH-02", "CN-01"
SMALL_CONDITIONS = ["single", "chained"]
SMALL_MIN_SAMPLES = 5


def _cfg_hash(tasks: list[str], conditions: list[str], min_samples: int) -> str:
    src = json.dumps(
        {"model": m.MODEL, "min_samples": min_samples,
         "tasks": sorted(tasks), "conditions": conditions},
        sort_keys=True,
    )
    return hashlib.sha256(src.encode()).hexdigest()[:12]


# ── Deterministic probe stub ─────────────────────────────────────────────────

_TASK_INDEX = {t: i for i, t in enumerate(SMALL_TASKS)}

_BASE_MCP_MS = {"single": 48.0, "chained": 143.0, "fan_out": 81.0}

def _make_fake_probes() -> dict:
    """Return _PROBES-shaped dict with deterministic per-(task, condition, sample_index) values.

    Each probe returns a distinct dict for each (task_id, sample_index) pair, so
    reordered or duplicated samples produce different aggregate JSON.  Real probes
    ignore sample_index; the stub uses it to return a reproducible value.
    """
    def _make(condition: str):
        base = _BASE_MCP_MS[condition]
        def probe(backend, task_id: str, sample_index: int = 0) -> dict:
            task_offset = _TASK_INDEX.get(task_id, 0) * 11.3
            si_offset   = sample_index * 1.7
            return {
                "mcp_roundtrip_ms":    base + task_offset + si_offset,
                "replay_roundtrip_ms": 0.0,
                "tool_dispatch_ms":    3.1 + task_offset / 20 + si_offset * 0.1,
                "turn_total_ms":       base + task_offset + si_offset + 6.0,
            }
        return probe

    return {cond: _make(cond) for cond in SMALL_CONDITIONS}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_resume_byte_identical(tmp_path: Path, monkeypatch) -> None:
    cfg_hash = _cfg_hash(SMALL_TASKS, SMALL_CONDITIONS, SMALL_MIN_SAMPLES)

    monkeypatch.setattr(m, "CHARACTERIZE_TASKS", SMALL_TASKS)
    monkeypatch.setattr(m, "CONDITIONS",         SMALL_CONDITIONS)
    monkeypatch.setattr(m, "MIN_SAMPLES",        SMALL_MIN_SAMPLES)
    monkeypatch.setattr(m, "CONFIG_HASH",        cfg_hash)
    monkeypatch.setattr(m, "_PROBES",            _make_fake_probes())

    total_rows = len(SMALL_TASKS) * len(SMALL_CONDITIONS) * SMALL_MIN_SAMPLES

    # ── Full run ──────────────────────────────────────────────────────────────
    jsonl_full = tmp_path / "full.jsonl"
    all_samples_full = m._run_all(
        backend=None,
        out_jsonl=jsonl_full,
        completed=set(),
        prior_samples={},
        no_fsync=True,
    )
    records_full = m._compute_tail_records(all_samples_full)
    full_json = json.dumps(records_full)

    lines = jsonl_full.read_text(encoding="utf-8").splitlines()
    assert len(lines) == total_rows, (
        f"expected {total_rows} rows, got {len(lines)}"
    )

    # ── Interrupted + resumed run ─────────────────────────────────────────────
    # Interrupt halfway — the cut point deliberately falls mid-task so the
    # resume has to handle both a partially-completed and a not-yet-started task.
    cut = total_rows // 2
    assert 0 < cut < total_rows, "cut must leave work on both sides"

    jsonl_resume = tmp_path / "resume.jsonl"
    jsonl_resume.write_text("\n".join(lines[:cut]) + "\n", encoding="utf-8")

    completed2, prior2 = m._load_jsonl_samples(jsonl_resume, cfg_hash)
    assert len(completed2) == cut, (
        f"expected {cut} completed samples after truncation, got {len(completed2)}"
    )

    all_samples_resume = m._run_all(
        backend=None,
        out_jsonl=jsonl_resume,
        completed=completed2,
        prior_samples=prior2,
        no_fsync=True,
    )
    records_resume = m._compute_tail_records(all_samples_resume)
    resume_json = json.dumps(records_resume)

    assert full_json == resume_json, (
        "Resumed run produced different aggregate JSON.\n"
        f"Full   ({len(records_full)} records): {full_json[:300]}\n"
        f"Resume ({len(records_resume)} records): {resume_json[:300]}"
    )


def test_resume_jsonl_row_count(tmp_path: Path, monkeypatch) -> None:
    """After a resume, the JSONL contains exactly total_rows lines."""
    cfg_hash = _cfg_hash(SMALL_TASKS, SMALL_CONDITIONS, SMALL_MIN_SAMPLES)

    monkeypatch.setattr(m, "CHARACTERIZE_TASKS", SMALL_TASKS)
    monkeypatch.setattr(m, "CONDITIONS",         SMALL_CONDITIONS)
    monkeypatch.setattr(m, "MIN_SAMPLES",        SMALL_MIN_SAMPLES)
    monkeypatch.setattr(m, "CONFIG_HASH",        cfg_hash)
    monkeypatch.setattr(m, "_PROBES",            _make_fake_probes())

    total_rows = len(SMALL_TASKS) * len(SMALL_CONDITIONS) * SMALL_MIN_SAMPLES

    # Full run
    jsonl_full = tmp_path / "full.jsonl"
    m._run_all(
        backend=None, out_jsonl=jsonl_full,
        completed=set(), prior_samples={}, no_fsync=True,
    )
    lines = jsonl_full.read_text(encoding="utf-8").splitlines()

    # Truncate to 1/3
    cut = total_rows // 3
    jsonl_resume = tmp_path / "resume.jsonl"
    jsonl_resume.write_text("\n".join(lines[:cut]) + "\n", encoding="utf-8")

    completed2, prior2 = m._load_jsonl_samples(jsonl_resume, cfg_hash)
    m._run_all(
        backend=None, out_jsonl=jsonl_resume,
        completed=completed2, prior_samples=prior2, no_fsync=True,
    )

    resume_lines = [
        ln for ln in jsonl_resume.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(resume_lines) == total_rows, (
        f"expected {total_rows} rows after resume, got {len(resume_lines)}"
    )


def test_resume_no_reorder(tmp_path: Path, monkeypatch) -> None:
    """Samples within a cell must appear in the same order as a full run.

    The cut is placed within the first cell (after 2 of 5 samples), so the
    resume must interleave prior [0,1] with new [2,3,4].  With
    sample-index-distinct probe values, any reordering produces a different
    samples list and therefore different JSON.
    """
    cfg_hash = _cfg_hash(SMALL_TASKS, SMALL_CONDITIONS, SMALL_MIN_SAMPLES)

    monkeypatch.setattr(m, "CHARACTERIZE_TASKS", SMALL_TASKS)
    monkeypatch.setattr(m, "CONDITIONS",         SMALL_CONDITIONS)
    monkeypatch.setattr(m, "MIN_SAMPLES",        SMALL_MIN_SAMPLES)
    monkeypatch.setattr(m, "CONFIG_HASH",        cfg_hash)
    monkeypatch.setattr(m, "_PROBES",            _make_fake_probes())

    # Full run
    jsonl_full = tmp_path / "full.jsonl"
    all_samples_full = m._run_all(
        backend=None, out_jsonl=jsonl_full,
        completed=set(), prior_samples={}, no_fsync=True,
    )
    records_full = m._compute_tail_records(all_samples_full)
    full_json = json.dumps(records_full)

    lines = jsonl_full.read_text(encoding="utf-8").splitlines()

    # Cut within first cell: first 2 samples of (SMALL_TASKS[0], SMALL_CONDITIONS[0])
    cut = 2
    jsonl_resume = tmp_path / "resume.jsonl"
    jsonl_resume.write_text("\n".join(lines[:cut]) + "\n", encoding="utf-8")

    completed2, prior2 = m._load_jsonl_samples(jsonl_resume, cfg_hash)
    all_samples_resume = m._run_all(
        backend=None, out_jsonl=jsonl_resume,
        completed=completed2, prior_samples=prior2, no_fsync=True,
    )
    records_resume = m._compute_tail_records(all_samples_resume)
    resume_json = json.dumps(records_resume)

    assert full_json == resume_json, (
        "Resumed run produced different JSON — samples may have been reordered.\n"
        f"Full   first-record samples: {records_full[0]['samples']}\n"
        f"Resume first-record samples: {records_resume[0]['samples']}"
    )


def test_resume_no_duplication(tmp_path: Path, monkeypatch) -> None:
    """A sample_index that appears twice in the JSONL must not inflate n.

    _load_jsonl_samples uses a set for completed (so no duplicate keys), but
    appends to prior_samples as a list.  If the list is not deduplicated,
    _run_all will carry the extra entry into all_samples and n will exceed
    MIN_SAMPLES.  This test fails if that deduplication is absent.
    """
    cfg_hash = _cfg_hash(SMALL_TASKS, SMALL_CONDITIONS, SMALL_MIN_SAMPLES)

    monkeypatch.setattr(m, "CHARACTERIZE_TASKS", SMALL_TASKS)
    monkeypatch.setattr(m, "CONDITIONS",         SMALL_CONDITIONS)
    monkeypatch.setattr(m, "MIN_SAMPLES",        SMALL_MIN_SAMPLES)
    monkeypatch.setattr(m, "CONFIG_HASH",        cfg_hash)
    monkeypatch.setattr(m, "_PROBES",            _make_fake_probes())

    # Full run
    jsonl_full = tmp_path / "full.jsonl"
    all_samples_full = m._run_all(
        backend=None, out_jsonl=jsonl_full,
        completed=set(), prior_samples={}, no_fsync=True,
    )
    records_full = m._compute_tail_records(all_samples_full)
    full_json = json.dumps(records_full)

    lines = jsonl_full.read_text(encoding="utf-8").splitlines()
    total_rows = len(SMALL_TASKS) * len(SMALL_CONDITIONS) * SMALL_MIN_SAMPLES
    assert len(lines) == total_rows

    # Inject duplicates: repeat the first two lines of the JSONL at position 2-3.
    # sample_index 0 and 1 of the first cell now appear twice each.
    dup_lines = lines[:2] + lines[:2] + lines[2:]
    jsonl_dup = tmp_path / "dup.jsonl"
    jsonl_dup.write_text("\n".join(dup_lines) + "\n", encoding="utf-8")

    completed_dup, prior_dup = m._load_jsonl_samples(jsonl_dup, cfg_hash)

    # completed set must have exactly total_rows distinct entries (set deduplicates).
    assert len(completed_dup) == total_rows, (
        f"completed set should have {total_rows} entries after dedup, "
        f"got {len(completed_dup)}"
    )

    all_samples_dup = m._run_all(
        backend=None, out_jsonl=jsonl_dup,
        completed=completed_dup, prior_samples=prior_dup, no_fsync=True,
    )
    records_dup = m._compute_tail_records(all_samples_dup)
    dup_json = json.dumps(records_dup)

    assert full_json == dup_json, (
        "Duplicate JSONL rows inflated the aggregate — prior_samples carries "
        "duplicates that were not filtered out.\n"
        f"Full n={records_full[0]['n']}, dup n={records_dup[0]['n']}\n"
        f"Full first-record samples: {records_full[0]['samples']}\n"
        f"Dup  first-record samples: {records_dup[0]['samples']}"
    )
