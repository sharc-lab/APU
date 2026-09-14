"""Explicit complete-versus-partial reader for Prompt A2 lifecycle artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from seam.errors import SeamError
from seam.rawstore import open_run_dir, verify_sealed

BLOCKS_FILENAME = "blocks.jsonl"
IN_PROGRESS_FILENAME = "in_progress.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SeamError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def read_prompt_a_run(root: Path, run_id: str, *, allow_partial: bool = False) -> dict[str, Any]:
    """Read a sealed run, or an explicitly labelled partial run when authorized."""
    run_dir = open_run_dir(run_id, repo_root=root)
    blocks = _read_jsonl(run_dir.path / BLOCKS_FILENAME)
    if not run_dir.is_sealed():
        if not allow_partial:
            raise SeamError(
                f"run {run_id} is unsealed PARTIAL/INCOMPLETE; pass allow_partial=True "
                "for diagnostic-only block recovery"
            )
        marker_path = run_dir.path / IN_PROGRESS_FILENAME
        marker = (
            json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.is_file() else None
        )
        return {
            "run_id": run_id,
            "lifecycle": "PARTIAL/INCOMPLETE",
            "sealed": False,
            "usable_for_final_analysis": False,
            "blocks": blocks,
            "in_progress": marker,
            "files": sorted(path.name for path in run_dir.path.iterdir() if path.is_file()),
        }
    if not verify_sealed(run_dir):
        raise SeamError(f"sealed run {run_id} failed integrity verification")
    summary_path = run_dir.path / "summary.json"
    return {
        "run_id": run_id,
        "lifecycle": "COMPLETE/SEALED",
        "sealed": True,
        "usable_for_final_analysis": True,
        "blocks": blocks,
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }
