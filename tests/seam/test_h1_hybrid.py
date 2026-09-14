"""Tests for tools/run_h1_hybrid.py (D-2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase_timers import phases_sum_to_wall  # noqa: E402
from tools.run_h1_hybrid import (  # noqa: E402
    CostCapExceeded,
    CostGuard,
    StubCloudBackend,
    StubLocalBackend,
    assert_seal_allowed,
    cloud_usd,
    decide_emission_escalate,
    decide_slo_escalate,
    run_hybrid_entry,
    run_session,
)

FIXTURE = ROOT / "tests" / "fixtures" / "h1_hybrid_3entries.json"


def _load_fix() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_blinding_runner_source_has_no_prediction_paths() -> None:
    src = (ROOT / "tools" / "run_h1_hybrid.py").read_text(encoding="utf-8")
    # Exact prediction artifact names must never appear (read or cited).
    forbidden = (
        "H1_PREDICTIONS.md",
        "h1_predictions.json",
        "derived/d1_replay/H1_PREDICTIONS",
        "derived\\d1_replay\\H1_PREDICTIONS",
    )
    for tok in forbidden:
        assert tok not in src, f"blinding violated: {tok!r} appears in run_h1_hybrid.py"


def test_phase_timers_sum_on_hybrid_ledger(tmp_path: Path) -> None:
    fix = _load_fix()
    local = StubLocalBackend(script=fix["local_script"])
    cloud = StubCloudBackend(tokens_in=100, tokens_out=20)
    cost = CostGuard(max_usd=100.0)
    er = run_hybrid_entry(
        fix["entries"][2],
        policy="slo_escalate",
        local=local,
        cloud=cloud,
        cost=cost,
        model="stub-4B",
    )
    assert er.turns
    for t in er.turns:
        d = t.as_dict()
        assert phases_sum_to_wall(d, tol_s=1e-3)


def test_policy_dispatch_slo_and_emission_on_3entry_fixture(tmp_path: Path) -> None:
    fix = _load_fix()
    local = StubLocalBackend(script=fix["local_script"])
    cloud = StubCloudBackend(
        tokens_in=fix["cloud_tokens_in"],
        tokens_out=fix["cloud_tokens_out"],
    )

    # slo_escalate: entry_0 turn1 has n_ctx=12000 > 10000 → escalate
    er0 = run_hybrid_entry(
        fix["entries"][0],
        policy="slo_escalate",
        local=local,
        cloud=cloud,
        cost=CostGuard(max_usd=100.0),
        model="stub-4B",
    )
    assert er0.turns[0].placement == "local"
    assert er0.turns[0].escalated is False
    assert er0.turns[1].escalated is True
    assert er0.turns[1].placement == "cloud"
    assert "ctx>" in (er0.turns[1].escalate_reason or "")
    assert er0.turns[2].placement == "cloud"
    assert er0.turns[2].escalate_reason == "stay_cloud"

    # emission_escalate: entry_1 turn0 has no parseable tool call
    er1 = run_hybrid_entry(
        fix["entries"][1],
        policy="emission_escalate",
        local=local,
        cloud=cloud,
        cost=CostGuard(max_usd=100.0),
        model="stub-4B",
    )
    assert er1.turns[0].escalated is True
    assert er1.turns[0].escalate_reason == "no_parseable_tool_call"
    assert er1.turns[1].placement == "cloud"

    # cloud_only: every turn cloud
    er2 = run_hybrid_entry(
        fix["entries"][2],
        policy="cloud_only",
        local=local,
        cloud=cloud,
        cost=CostGuard(max_usd=100.0),
        model="stub-4B",
    )
    assert all(t.placement == "cloud" and t.escalated for t in er2.turns)


def test_cost_guard_trips(tmp_path: Path) -> None:
    fix = _load_fix()
    local = StubLocalBackend(script=fix["local_script"])
    # One cloud turn ≈ cloud_usd(1000,200) = 0.006
    per = cloud_usd(1000, 200)
    cloud = StubCloudBackend(tokens_in=1000, tokens_out=200)
    with pytest.raises(CostCapExceeded) as ei:
        run_hybrid_entry(
            fix["entries"][2],
            policy="cloud_only",
            local=local,
            cloud=cloud,
            cost=CostGuard(max_usd=per * 0.5),
            model="stub-4B",
        )
    assert ei.value.partial is not None
    assert ei.value.partial.status == "aborted_cap"

    summary = run_session(
        policy="cloud_only",
        entries=fix["entries"],
        out_dir=tmp_path / "cap",
        max_usd=per * 0.5,
        local=local,
        cloud=cloud,
        skip_entry_assert=True,
    )
    assert summary["status"] == "aborted_cap"
    assert (tmp_path / "cap" / "checkpoint.json").is_file()
    assert (tmp_path / "cap" / ".sealed").is_file()


def test_resume_skips_completed_entries(tmp_path: Path) -> None:
    fix = _load_fix()
    local = StubLocalBackend(script=fix["local_script"])
    calls = {"n": 0}

    def _count() -> None:
        calls["n"] += 1

    cloud = StubCloudBackend(tokens_in=10, tokens_out=5, on_call=_count)
    out = tmp_path / "resume"
    # First run completes all under large cap
    s1 = run_session(
        policy="cloud_only",
        entries=fix["entries"],
        out_dir=out,
        max_usd=100.0,
        local=local,
        cloud=cloud,
        run_id="resume-test",
        skip_entry_assert=True,
        seal=False,
    )
    assert s1["status"] == "complete"
    n_first = calls["n"]
    assert n_first > 0

    # Second run with same out_dir must skip all entries (no new cloud calls)
    calls["n"] = 0
    s2 = run_session(
        policy="cloud_only",
        entries=fix["entries"],
        out_dir=out,
        max_usd=100.0,
        local=local,
        cloud=cloud,
        run_id="resume-test",
        skip_entry_assert=True,
        seal=True,
    )
    assert s2["status"] == "complete"
    assert calls["n"] == 0
    ckpt = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    assert set(ckpt["completed_entry_ids"]) == {e["id"] for e in fix["entries"]}


def test_decide_helpers() -> None:
    esc, reason = decide_slo_escalate(
        already_on_cloud=False, ttft_s=11.0, decode_tok_s=10.0, n_ctx=100
    )
    assert esc and "ttft>" in (reason or "")
    esc, reason = decide_slo_escalate(
        already_on_cloud=False, ttft_s=1.0, decode_tok_s=3.0, n_ctx=100
    )
    assert esc and "decode<" in (reason or "")
    esc, _ = decide_emission_escalate(already_on_cloud=False, emitted_parseable_tool_call=True)
    assert not esc


def test_agnostic_default_refused_live(tmp_path: Path) -> None:
    fix = _load_fix()
    with pytest.raises(SystemExit, match="derive-r1"):
        run_session(
            policy="agnostic_default",
            entries=fix["entries"],
            out_dir=tmp_path / "r1",
            max_usd=0.0,
            local=StubLocalBackend(script=fix["local_script"]),
            cloud=StubCloudBackend(),
            skip_entry_assert=True,
            seal=False,
        )


def test_stub_cannot_seal_hybrid(tmp_path: Path) -> None:
    fix = _load_fix()
    local = StubLocalBackend(script=fix["local_script"])
    cloud = StubCloudBackend()
    with pytest.raises(SystemExit, match="OpenVinoLocalBackend"):
        assert_seal_allowed(seal=True, local=local, policy="slo_escalate")
    with pytest.raises(SystemExit, match="OpenVinoLocalBackend"):
        run_session(
            policy="slo_escalate",
            entries=fix["entries"][:1],
            out_dir=tmp_path / "nosseal",
            max_usd=100.0,
            local=local,
            cloud=cloud,
            skip_entry_assert=True,
            seal=True,
        )


def test_fixture_cli_refuses_seal(tmp_path: Path) -> None:
    from tools import run_h1_hybrid as mod

    with pytest.raises(SystemExit, match="incompatible with --fixture"):
        mod.main(
            [
                "--policy",
                "cloud_only",
                "--max-usd",
                "1",
                "--out",
                str(tmp_path / "fx"),
                "--fixture",
                str(FIXTURE),
                "--seal",
            ]
        )


if __name__ == "__main__":
    import tempfile

    test_blinding_runner_source_has_no_prediction_paths()
    test_decide_helpers()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_phase_timers_sum_on_hybrid_ledger(p)
        test_policy_dispatch_slo_and_emission_on_3entry_fixture(p)
        test_cost_guard_trips(p)
        test_resume_skips_completed_entries(p)
        test_agnostic_default_refused_live(p)
        test_stub_cannot_seal_hybrid(p)
        test_fixture_cli_refuses_seal(p)
    print("PASS tests/test_h1_hybrid.py")
