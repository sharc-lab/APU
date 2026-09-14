"""Smoke: W-3 ledger records per-turn emitted_parseable_tool_call."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_w3_bfcl_quality import _entry_ledger_row  # noqa: E402


def test_emitted_parseable_tool_call_present_and_boolean() -> None:
    entry = {
        "id": "smoke_entry",
        "score": {"valid": False, "error_type": None},
        "force_quit": False,
        "wall_s": 1.0,
        "n_completed_turns": 2,
        "n_user_turns": 2,
        "turn_metrics": [
            {
                "turn": 0,
                "n_generations": 1,
                "calls_emitted_per_generation": [2],
                "last_wall_s": 0.5,
                "ttft_s": 0.1,
                "decode_tok_s": 10.0,
                "slo_ok": True,
                "per_turn_accuracy": {"correct": True},
            },
            {
                "turn": 1,
                "n_generations": 1,
                "calls_emitted_per_generation": [0],
                "last_wall_s": 0.4,
                "ttft_s": 0.1,
                "decode_tok_s": 10.0,
                "slo_ok": True,
                "per_turn_accuracy": {"correct": False},
            },
        ],
    }
    row = _entry_ledger_row(
        entry_result=entry,
        model_spec=Path("models/specs/smoke.yaml"),
        ir_sha256="0" * 64,
    )
    assert len(row["per_turn"]) == 2
    for t in row["per_turn"]:
        assert "emitted_parseable_tool_call" in t
        assert isinstance(t["emitted_parseable_tool_call"], bool)
    assert row["per_turn"][0]["emitted_parseable_tool_call"] is True
    assert row["per_turn"][1]["emitted_parseable_tool_call"] is False


if __name__ == "__main__":
    test_emitted_parseable_tool_call_present_and_boolean()
    print("PASS tests/test_w3_ledger_emission.py")
