"""
Byte-identical resume test for tail_latency_instrument._run_all.

Verifies that interrupting a sweep partway through and resuming produces
a _compute_tail_records() output that is JSON-identical to a complete
uninterrupted run with the same deterministic probe stub.

The stub returns a fixed value per (task_id, condition) — the only viable
approach because _run_all does not pass sample_index to the probe function.
Any call-order-varying stub would diverge on resume (counter resets), so
byte-identity would correctly fail. This test proves the JSONL round-trip
and resume bookkeeping are correct.
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
    """Return _PROBES-shaped dict with deterministic per-(task, condition) values.

    Each probe returns the same dict on every call for the same task_id so that
    JSONL-loaded samples (from a prior run) match freshly probed ones exactly.
    """
    def _make(condition: str):
        base = _BASE_MCP_MS[condition]
        def probe(backend, task_id: str) -> dict:
            offset = _TASK_INDEX.get(task_id, 0) * 11.3
            return {
                "mcp_roundtrip_ms":    base + offset,
                "replay_roundtrip_ms": 0.0,
                "tool_dispatch_ms":    3.1 + offset / 20,
                "turn_total_ms":       base + offset + 6.0,
            }
        return probe

    return {cond: _make(cond) for cond in SMALL_CONDITIONS}


# ── Test ─────────────────────────────────────────────────────────────────────

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
