from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from seam.bench.attrib import (
    _assert_sweep_authorized,
    _interaction_cells,
    _runtime_projection,
)
from seam.errors import SeamError

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs" / "attrib.yaml").read_text(encoding="utf-8"))


def _wall(prompt: float, n_out: float) -> float:
    return 0.1 + prompt / 1000.0 + n_out * 0.01 + prompt * n_out * 1e-6


def test_runtime_projection_includes_grid_heldout_warmup_and_cooldown() -> None:
    cfg = _config()
    records = [
        {
            **corner,
            "wall_s": _wall(corner["total_prompt_tokens"], corner["n_out"]),
        }
        for _repeat in range(cfg["runtime_pilot"]["repeats"])
        for corner in cfg["runtime_pilot"]["corners"]
    ]

    result = _runtime_projection(records, attrib=cfg)

    cells = _interaction_cells(cfg, quantization="int4")
    one_pass = sum(_wall(cell["total_prompt_tokens"], cell["n_out"]) for cell in cells)
    repeats = cfg["runtime_pilot"]["projected_measured_repeats"]
    expected = (
        repeats * one_pass
        + one_pass
        + repeats * cfg["thermal"]["warmup_s"]
        + (repeats - 1) * cfg["thermal"]["cooldown_between_blocks_s"]
    )
    assert result["n_grid_cells"] == 35
    assert result["n_held_out_cells"] == 6
    assert result["projected_total_s"] == pytest.approx(expected)


def test_full_sweep_refuses_before_runtime_pilot_authorization() -> None:
    with pytest.raises(SeamError, match="has not been human-authorized"):
        _assert_sweep_authorized(root=ROOT, attrib=_config())
