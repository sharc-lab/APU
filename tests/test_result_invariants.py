"""Invariant checks on completed result files.

Runs over every results/run_*.jsonl that exists at test time. Skips files that
pre-date the model_variant field (no rows carry it). Fails if a file mixes
model_variant values — reasoning and instruct runs must never be pooled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _result_files():
    return sorted(RESULTS_DIR.glob("run_*.jsonl"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.parametrize("fpath", _result_files(), ids=lambda p: p.name)
def test_single_model_variant(fpath):
    rows = _rows(fpath)
    variants = {r["model_variant"] for r in rows if "model_variant" in r}
    if not variants:
        pytest.skip("pre-dates model_variant field")
    assert len(variants) == 1, (
        f"{fpath.name} mixes model_variants {variants}. "
        "Reasoning and instruct runs must never be pooled in one file."
    )


@pytest.mark.parametrize("fpath", _result_files(), ids=lambda p: p.name)
def test_thinking_enabled_consistent_with_variant(fpath):
    rows = _rows(fpath)
    typed = [r for r in rows if "model_variant" in r and "thinking_enabled" in r]
    if not typed:
        pytest.skip("pre-dates model_variant / thinking_enabled fields")
    for r in typed:
        if r["model_variant"] == "instruct":
            assert r["thinking_enabled"] is False, (
                f"{fpath.name}: row for {r.get('probe_id')} has "
                f"model_variant=instruct but thinking_enabled={r['thinking_enabled']}"
            )
