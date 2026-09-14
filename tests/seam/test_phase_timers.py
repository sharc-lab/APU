"""Smoke: phase timers sum to turn_wall within 1 ms (stubbed generate)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.phase_timers import finalize_phase_timers, phases_sum_to_wall  # noqa: E402


def _stub_generate(duration_s: float = 0.01) -> float:
    t0 = time.perf_counter()
    time.sleep(duration_s)
    return time.perf_counter() - t0


def test_phase_timers_sum_to_wall_within_1ms_stubbed_generate() -> None:
    t_turn0 = time.perf_counter()
    t_template_build = 0.002
    time.sleep(t_template_build)
    t_tokenize = 0.001
    time.sleep(t_tokenize)
    t_generate = _stub_generate(0.01)
    t_tool_exec = 0.003
    time.sleep(t_tool_exec)
    # leftover harness work lands in t_other
    time.sleep(0.002)
    turn_wall = time.perf_counter() - t_turn0
    phases = finalize_phase_timers(
        turn_wall_s=turn_wall,
        t_tool_exec=t_tool_exec,
        t_template_build=t_template_build,
        t_tokenize=t_tokenize,
        t_generate=t_generate,
    )
    assert phases_sum_to_wall(phases, tol_s=1e-3)
    assert (
        abs(
            phases["t_tool_exec"]
            + phases["t_template_build"]
            + phases["t_tokenize"]
            + phases["t_generate"]
            + phases["t_other"]
            - phases["turn_wall_s"]
        )
        <= 1e-3
    )
    assert phases["t_generate"] >= 0.009
    assert phases["t_other"] >= 0.0


if __name__ == "__main__":
    test_phase_timers_sum_to_wall_within_1ms_stubbed_generate()
    print("PASS tests/test_phase_timers.py")
