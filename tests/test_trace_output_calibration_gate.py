"""scripts/trace_output_calibration.py --count: the pre-job gate runs before the first count_tokens call (operator
2026-10-24). count_tokens is not billed, so its estimate is USD 0 and the USD 2 limit cannot trip; the Console check
still must pass. Fakes only: no network, no key."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("trace_output_calibration", REPO / "scripts" / "trace_output_calibration.py")
TOC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TOC)


def test_count_refused_by_the_console_check_before_any_call(monkeypatch, tmp_path):
    from src.cloud import client as cc
    from src.cloud import console_gate as g
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "s1.json").write_text(json.dumps({"trajectory_id": "t1", "text": "x"}), encoding="utf-8")
    calib = tmp_path / "calib.jsonl"
    monkeypatch.setattr(TOC, "CALIB_FILE", calib)
    monkeypatch.setattr(g, "default_check_path", lambda: tmp_path / "missing_console.json")
    counted = []
    real_cc = cc.CloudClient

    def client(*a, **kw):
        kw["ledger_path"] = tmp_path / "results" / "cloud_ledger.jsonl"
        c = real_cc(*a, api_key="test-not-a-key", **kw)
        c.count_tokens = lambda **k: counted.append(k) or 1
        return c

    monkeypatch.setattr(cc, "CloudClient", client)
    with pytest.raises(SystemExit, match="refused .console check."):
        TOC.count(scratch, role_override="controller")
    assert counted == [] and not calib.exists() and not (tmp_path / "results" / "cloud_ledger.jsonl").exists()
