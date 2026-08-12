"""Tests for the BOM frontier model.

The frontier drives which cells get spent on model calls, so an error here wastes
an overnight run or, worse, silently omits the cells that carry the finding.
"""

import pytest

import bom_sweep as B
from harness.runner import validate_result_row, REQUIRED_ROW_FIELDS


# ------------------------------------------------------------------ KV arithmetic

def test_kv_footprint_matches_measured_baseline():
    """
    The measured figure is 73,728 B/token for Qwen3-4B. Deriving it from the
    architecture must reproduce it exactly, at int8 KV. If this fails, either the
    architecture constants are wrong or the measured figure means something else,
    and every downstream number is suspect.
    """
    m = next(m for m in B.MODELS if m.name == "qwen3-4b")
    assert m.kv_bytes_per_token(8) == 73_728


def test_kv_quantization_halves_footprint():
    m = next(m for m in B.MODELS if m.name == "qwen3-4b")
    assert m.kv_bytes_per_token(4) * 2 == m.kv_bytes_per_token(8)


def test_tokens_per_gb_reproduces_reported_figure():
    """The reported figure is ~13,500 tokens/GB. Decimal GB must reproduce it; this pins
    the unit convention so decimal and binary GB never get mixed."""
    m = next(m for m in B.MODELS if m.name == "qwen3-4b")
    per_gb = B.GB // m.kv_bytes_per_token(8)
    assert 13_400 <= per_gb <= 13_700, per_gb


# ------------------------------------------------------------------ feasibility

def test_weights_larger_than_memory_is_infeasible():
    big = B.Model("huge", 200.0, 80, 8, 128, 32_768)
    cell = B.evaluate(big, B.LADDER[0], kv_bits=8, weight_bits=4)
    assert cell.binding == "infeasible"
    assert not cell.fits
    assert cell.resident_tokens == 0


def test_infeasible_cells_are_never_planned():
    """Spending calls on a config that cannot load the model is pure waste."""
    cells = B.frontier()
    for c in B.informative_cells(cells):
        assert c.fits, f"{c.model} on {c.config} cannot load weights"


def test_reserved_memory_reduces_usable():
    a = B.BomConfig("x", 16, 0, memory_architecture="unified", reserved_gb=2.0)
    b = B.BomConfig("x", 16, 0, memory_architecture="unified", reserved_gb=8.0)
    assert a.usable_bytes > b.usable_bytes


def test_reserved_memory_cannot_go_negative():
    c = B.BomConfig("tiny", 2, 0, memory_architecture="unified", reserved_gb=8.0)
    assert c.usable_bytes == 0


# ------------------------------------------------------------------ memory_architecture field

def test_memory_architecture_required_no_default():
    """BomConfig must not silently default memory_architecture."""
    with pytest.raises(TypeError):
        B.BomConfig("x", 16, 0)  # missing memory_architecture


def test_memory_architecture_rejects_invalid_value():
    with pytest.raises(ValueError, match="unified.*discrete"):
        B.BomConfig("x", 16, 0, memory_architecture="shared")


def test_all_ladder_configs_are_unified():
    for cfg in B.LADDER:
        assert cfg.memory_architecture == "unified", (
            f"LADDER entry {cfg.name!r} has memory_architecture={cfg.memory_architecture!r}; "
            "all LADDER entries must be 'unified' (they model AI PCs, not measurement hosts)"
        )


def test_blade_host_is_discrete():
    assert B.BLADE_HOST.memory_architecture == "discrete"
    assert B.BLADE_HOST.bom_cost_usd == 0


def test_cell_carries_memory_architecture():
    cell = B.evaluate(B.MODELS[0], B.LADDER[0], kv_bits=8, weight_bits=4)
    assert cell.memory_architecture == "unified"
    blade_cell = B.evaluate(B.MODELS[0], B.BLADE_HOST, kv_bits=8, weight_bits=4)
    assert blade_cell.memory_architecture == "discrete"


# ------------------------------------------------------------------ discrete exclusion

def test_informative_cells_excludes_discrete():
    """Discrete measurement hosts must never appear in the informative set."""
    blade_cells = [
        B.evaluate(m, B.BLADE_HOST, kb, wb)
        for m in B.MODELS
        for kb in B.KV_BITS
        for wb in B.WEIGHT_BITS
    ]
    all_cells = B.frontier() + blade_cells
    inf = B.informative_cells(all_cells)
    discrete_in_inf = [c for c in inf if c.memory_architecture == "discrete"]
    assert not discrete_in_inf, (
        f"Discrete config(s) leaked into informative set: "
        f"{[c.config for c in discrete_in_inf]}"
    )


def test_cheapest_config_per_model_excludes_discrete():
    blade_cells = [
        B.evaluate(m, B.BLADE_HOST, kb, wb)
        for m in B.MODELS
        for kb in B.KV_BITS
        for wb in B.WEIGHT_BITS
    ]
    all_cells = B.frontier() + blade_cells
    cheapest = B.cheapest_config_per_model(all_cells)
    for model, cell in cheapest.items():
        if cell is not None:
            assert cell.memory_architecture == "unified", (
                f"cheapest_config_per_model returned a discrete cell for {model}"
            )


# ------------------------------------------------------------------ binding classification

def test_binding_is_memory_when_residency_below_context_window():
    cells = B.frontier()
    for c in cells:
        if c.binding == "memory":
            assert c.resident_tokens < c.native_ctx
        elif c.binding == "context_window":
            assert c.resident_tokens >= c.native_ctx


def test_every_cell_classified():
    for c in B.frontier():
        assert c.binding in ("memory", "context_window", "infeasible")


# ------------------------------------------------------------------ informative selection

def test_informative_includes_every_memory_bound_cell():
    cells = B.frontier()
    inf = set(id(c) for c in B.informative_cells(cells))
    for c in cells:
        if c.binding == "memory":
            assert id(c) in inf


def test_informative_includes_a_control_per_model():
    """
    Without a non-binding control, a flat quality curve cannot be distinguished
    from a broken harness. Every model with any feasible config needs one.
    """
    cells = B.frontier()
    inf = B.informative_cells(cells)
    for model in {c.model for c in cells if c.fits}:
        controls = [c for c in inf if c.model == model
                    and c.binding == "context_window"]
        assert controls, f"no control cell for {model}"


def test_informative_is_a_strict_subset():
    cells = B.frontier()
    assert len(B.informative_cells(cells)) < len(cells)


# ------------------------------------------------------------------ depth grid

def test_depth_grid_starts_at_zero():
    for c in B.informative_cells(B.frontier()):
        assert B.depth_grid(c)[0] == 0


def test_depth_grid_respects_binding_ceiling():
    for c in B.informative_cells(B.frontier()):
        ceiling = min(c.resident_tokens, c.native_ctx)
        for d in B.depth_grid(c):
            assert d <= ceiling, f"{c.model}/{c.config}: depth {d} exceeds {ceiling}"


def test_depth_grid_is_sorted_and_unique():
    for c in B.informative_cells(B.frontier()):
        g = B.depth_grid(c)
        assert g == sorted(g)
        assert len(g) == len(set(g))


def test_depth_grid_aligned_to_1024():
    """Alignment keeps depths comparable across configs and reproducible."""
    for c in B.informative_cells(B.frontier()):
        for d in B.depth_grid(c):
            assert d % 1024 == 0


# ------------------------------------------------------------------ planning

def test_plan_call_count_is_consistent():
    cells = B.informative_cells(B.frontier())
    p = B.plan(cells, n_probes=10, reps=2)
    assert p["total_calls"] == sum(r["calls"] for r in p["cells"])
    for r, _c in zip(p["cells"], cells, strict=True):
        assert r["calls"] == len(r["depths"]) * 10 * 2


def test_plan_records_unit_convention():
    p = B.plan(B.informative_cells(B.frontier()), 5, 1)
    assert "decimal" in p["units"]


# ------------------------------------------------------------------ robustness of the headline

@pytest.mark.parametrize("reserved", [2.0, 3.0, 4.0, 5.0, 6.0])
def test_16gb_verdict_is_stable_across_reserved_assumption(reserved):
    """
    The headline claim -- a 16 GB device runs 4B/8B/14B but never 32B -- must not
    depend on the reserved-memory guess, which is an estimate rather than a
    measurement. If this test starts failing, the claim needs hedging.
    """
    ladder = tuple(
        B.BomConfig(c.name, c.memory_gb, c.bom_cost_usd,
                    memory_architecture=c.memory_architecture,
                    has_npu=c.has_npu, npu_tops=c.npu_tops,
                    reserved_gb=reserved, notes=c.notes)
        for c in B.LADDER
    )
    cells = B.frontier(configs=ladder)
    runs = {c.model for c in cells if c.config == "16gb" and c.fits}
    assert "qwen3-32b" not in runs
    assert {"qwen3-4b", "qwen3-8b", "qwen3-14b"} <= runs


# ------------------------------------------------------------------ result row validation

def test_validate_result_row_passes_complete_row():
    row = {f: "x" for f in REQUIRED_ROW_FIELDS}
    validate_result_row(row)  # must not raise


def test_validate_result_row_fails_missing_memory_architecture():
    row = {f: "x" for f in REQUIRED_ROW_FIELDS}
    del row["memory_architecture"]
    with pytest.raises(ValueError, match="memory_architecture"):
        validate_result_row(row)


def test_validate_result_row_fails_missing_hardware_config():
    row = {f: "x" for f in REQUIRED_ROW_FIELDS}
    del row["hardware_config"]
    with pytest.raises(ValueError, match="hardware_config"):
        validate_result_row(row)


def test_validate_result_row_fails_on_empty_row():
    with pytest.raises(ValueError):
        validate_result_row({})
