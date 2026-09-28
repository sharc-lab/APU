"""Dry run of phase_r1_speed against a stub lab (no real server, no real co-runner) -- required before any new or
changed phase is queued on a live machine, per the standing rule added after the two duplicate-keyword crashes."""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t2s_night2 as n2  # noqa: E402
from stub_lab import StubLab, StubModelInfo, StubServer  # noqa: E402


def test_phase_r1_speed_dry_run_every_model_every_length():
    lab = StubLab(models={mid: StubModelInfo(mid) for mid in n2.R1_SPEED_MODELS if mid != "llama-3.3-70b"})
    # llama-3.3-70b intentionally left absent from lab.models, mirroring load_models() skipping it before its
    # download/verify lands -- the phase must not crash on a missing model, just skip it.
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1_speed(lab)

    start_rows = [r for r in lab.rows if r.get("kind") == "start"]
    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    assert {r["model_id"] for r in start_rows} == {"qwen3-8b", "qwen3-14b", "qwen3-32b"}
    assert all(r.get("axis") == "latency" for r in start_rows)
    assert all(r.get("axis") == "latency" for r in call_rows)
    # 3 models x 5 lengths x (1 warm-up + 3 measured) = 60 call rows
    assert len(call_rows) == 3 * len(n2.R1_SPEED_LENGTHS) * 4
    for mid in ("qwen3-8b", "qwen3-14b", "qwen3-32b"):
        got = {r["target_prompt_tokens"] for r in call_rows if r["model_id"] == mid and not r["warmup"]}
        assert got == set(n2.R1_SPEED_LENGTHS)
    # every item_id got marked done exactly once, no duplicate-keyword or double-run
    expected_items = {f"R1speed_{mid}_{tgt}" for mid in ("qwen3-8b", "qwen3-14b", "qwen3-32b") for tgt in n2.R1_SPEED_LENGTHS}
    assert expected_items <= lab.done


def test_phase_r1_speed_dry_run_no_models_present_does_not_crash():
    lab = StubLab(models={})
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1_speed(lab)
    assert lab.rows == []


def test_phase_r1_speed_dry_run_resumes_skipping_done_items():
    lab = StubLab(models={"qwen3-8b": StubModelInfo("qwen3-8b")})
    lab.done.add("R1speed_qwen3-8b_1024")
    with mock.patch.object(n2.L, "Server", StubServer), mock.patch.object(n2.time, "sleep", lambda *a: None):
        n2.phase_r1_speed(lab)
    call_rows = [r for r in lab.rows if r.get("kind") == "call"]
    got = {r["target_prompt_tokens"] for r in call_rows}
    assert 1024 not in got
    assert got == set(n2.R1_SPEED_LENGTHS) - {1024}
