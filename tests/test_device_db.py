"""G1 device database (analysis/device_db.py): one schema, every non-null value sourced, the committed JSON equals a
rebuild from the result files, and the markdown export is current."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import device_db as ddb  # noqa: E402


@pytest.fixture(scope="module")
def db():
    return ddb.build(REPO)


def test_schema(db):
    assert db["schema"] == list(ddb.FIELDS)
    for e in db["entries"]:
        for f in ddb.FIELDS:
            assert f in e


def test_every_value_sourced(db):
    for e in db["entries"]:
        for f in ("version", "window", "kv", "backend", "measured_ceiling", "wrong_rate", "s_per_task"):
            if e[f] is not None:
                assert e["sources"].get(f), (e["device"], e["model"], e.get("setting"), f)


def test_r2_never_carries_a_wrong_rate(db):
    for e in db["entries"]:
        if e.get("workload", "").startswith("R2"):
            assert e["wrong_rate"] is None


def test_known_cells(db):
    b3 = {e["setting"]: e for e in db["entries"] if e.get("workload") == "B3 v2 tasks, 5 seeds" and e["runtime"] == "ollama"}
    assert (b3["ollama_ctx_4096"]["wrong_rate"]["k"], b3["ollama_ctx_4096"]["wrong_rate"]["n"]) == (63, 150)
    assert (b3["ollama_default"]["wrong_rate"]["k"], b3["ollama_default"]["window"]) == (17, 40960)
    devs = {e["device"] for e in db["entries"]}
    assert {"evo-x2", "evo-t2s Intel CPU", "evo-t2s Intel iGPU"} <= devs
    assert not any("Blade" in d for d in devs)


def test_committed_json_matches_rebuild(db):
    committed = json.loads((REPO / ddb.OUT_JSON).read_text(encoding="utf-8"))
    assert json.loads(json.dumps(db, default=str)) == committed


def test_markdown_current(db, tmp_path):
    p = tmp_path / "DEVICE_DB.md"
    ddb.write_md(db, p)
    assert p.read_text(encoding="utf-8") == (REPO / ddb.OUT_MD).read_text(encoding="utf-8")
