"""Inline C-2 prediction block for plan.json.

The runner does not open a preregistration, amendment, or predictions file.
Those documents are attached by the scoring tool after the run.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def resolve_ttft_slo_plan_predictions(
    *,
    model_spec: Path,
    arm_ids: list[str],
    slo_s: float,
    repeats: int,
    default_c2_predictions: Callable[..., dict[str, Any]],
    repo_root: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    """Return the inline C-2 block. ``model_spec`` is recorded, not used to open a file."""
    del repo_root, predictions_path
    inline = default_c2_predictions(slo_s=slo_s, repeats=repeats)
    return {
        "source": "c2_am038_inline",
        "predictions_path": None,
        "arm_key": None,
        "arms": list(arm_ids),
        "model_spec": str(model_spec),
        "note": (
            "Inline C-2 / AM-038 turn-1 KV agreement prediction. "
            "Prediction files are attached at scoring time, not by the runner."
        ),
        "c2_am038_not_applicable": False,
        **inline,
    }
