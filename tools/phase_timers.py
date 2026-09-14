"""Per-turn phase timers for the session_residency harness (additive ledger fields).

Sealed runs are untouched. Replay does not consume these until a sealed run
measures them.
"""

from __future__ import annotations

from typing import Any

PHASE_KEYS = (
    "t_tool_exec",
    "t_template_build",
    "t_tokenize",
    "t_generate",
)


def finalize_phase_timers(
    *,
    turn_wall_s: float,
    t_tool_exec: float = 0.0,
    t_template_build: float = 0.0,
    t_tokenize: float = 0.0,
    t_generate: float = 0.0,
) -> dict[str, float]:
    """Close a turn's phase budget. t_other = turn_wall - sum(known)."""
    known = float(t_tool_exec) + float(t_template_build) + float(t_tokenize) + float(t_generate)
    turn_wall = float(turn_wall_s)
    return {
        "turn_wall_s": turn_wall,
        "t_tool_exec": float(t_tool_exec),
        "t_template_build": float(t_template_build),
        "t_tokenize": float(t_tokenize),
        "t_generate": float(t_generate),
        "t_other": turn_wall - known,
    }


def phases_sum_to_wall(phases: dict[str, Any], *, tol_s: float = 1e-3) -> bool:
    """True iff known phases + t_other equal turn_wall within tol_s (default 1 ms)."""
    wall = float(phases["turn_wall_s"])
    total = (
        float(phases["t_tool_exec"])
        + float(phases["t_template_build"])
        + float(phases["t_tokenize"])
        + float(phases["t_generate"])
        + float(phases["t_other"])
    )
    return abs(total - wall) <= tol_s
