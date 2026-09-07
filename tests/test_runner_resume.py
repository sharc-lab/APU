"""Resume tolerance and durability tests for harness/runner.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.runner import _load_completed

_HASH = "abc123def456"


def _row(depth: int, rep: int, probe_id: str) -> dict:
    return {"depth": depth, "rep": rep, "probe_id": probe_id,
            "config_hash": _HASH, "score": 1.0}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ── normal resume ─────────────────────────────────────────────────────────────

def test_well_formed_file_recovers_all_rows(tmp_path: Path) -> None:
    """Happy path: a clean JSONL with no truncation returns every completed cell."""
    rows = [_row(0, 0, "rea_01"), _row(0, 1, "rea_02"), _row(2000, 0, "rea_03")]
    path = tmp_path / "resume.jsonl"
    _write_jsonl(path, rows)

    completed = _load_completed(path, _HASH)

    assert completed == {(0, 0, "rea_01"), (0, 1, "rea_02"), (2000, 0, "rea_03")}


# ── truncated final line ───────────────────────────────────────────────────────

def test_truncated_final_line_is_discarded(tmp_path: Path) -> None:
    """Intact rows are recovered; a truncated final line is discarded, not raised."""
    rows = [_row(0, 0, "rea_01"), _row(0, 1, "rea_02")]
    path = tmp_path / "resume.jsonl"
    _write_jsonl(path, rows)
    # Simulate power-loss mid-write: partial object, no closing brace or newline.
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"depth": 0, "rep": 2, "probe_id": "rea_03", "config_hash": "' + _HASH[:8])

    completed = _load_completed(path, _HASH)

    assert completed == {(0, 0, "rea_01"), (0, 1, "rea_02")}


# ── mid-file corruption ────────────────────────────────────────────────────────

def test_malformed_mid_file_line_raises(tmp_path: Path) -> None:
    """A malformed line that is not the last raises ValueError (real corruption)."""
    path = tmp_path / "resume.jsonl"
    path.write_text(
        json.dumps(_row(0, 0, "rea_01")) + "\n"
        + "NOT_JSON\n"
        + json.dumps(_row(0, 1, "rea_02")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Malformed JSON"):
        _load_completed(path, _HASH)


# ── config hash guard ─────────────────────────────────────────────────────────

def test_config_hash_mismatch_raises(tmp_path: Path) -> None:
    """A row with a different config_hash raises after the refactor."""
    path = tmp_path / "resume.jsonl"
    other_hash = "zzz999fff000"
    row = {"depth": 0, "rep": 0, "probe_id": "rea_01",
           "config_hash": other_hash, "score": 1.0}
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match="Config hash mismatch"):
        _load_completed(path, _HASH)
