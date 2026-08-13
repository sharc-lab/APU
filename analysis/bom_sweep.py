"""
BOM feasibility model and sweep planner.

WHY THIS IS NOT JUST A MEMORY SWEEP
-----------------------------------
The obvious experiment is "vary memory, measure quality." Running the arithmetic
first shows that experiment is not informative on its own:

    Qwen3-4B, KV at 73,728 B/token, ~4 GB reserved for OS and framework
      8 GB  -> tens of thousands of tokens of resident session
     16 GB  -> more still
     32 GB  -> far more

Qwen3-4B's native context window is 262,144 tokens (confirmed 2026-08-13 from
ollama show; the qwen3-2507 release extended all variants to 256k). At this
window size, memory is the binding constraint for 8 GB and 16 GB unified
configs: neither can hold enough KV tokens to fill the context window. Memory
does NOT bind for 32 GB+, where residency exceeds the context window ceiling.
A 4B sweep therefore DOES carry information across the 8–16 GB range; it is
NOT a flat line. The earlier conclusion that "16 GB is generous for small
models" was based on the wrong 32,768 value and is inverted by the true window.

Capacity binds only when the model is large enough that weights plus KV crowd
the budget. So the BOM question is two-dimensional: (model size x memory), and
the deliverable is the FRONTIER between the region where memory binds and the
region where it does not. That frontier is what an OEM actually needs, and it is
what Paper 1 should produce.

This module computes the frontier analytically, then plans the empirical sweep
only over cells where it predicts something happens. That keeps an overnight run
to the cells that carry information.

UNITS
-----
The reported "~13,500 tokens per GB" implies decimal GB (1e9). Binary GiB gives
~14,563 for the same footprint, a 7% difference. Mixing the two across a paper
produces numbers that do not reconcile. This module uses DECIMAL GB throughout
to match his figures and says so in every emitted artifact.

KV QUANTIZATION
---------------
73,728 B/token corresponds to int8 KV for this architecture
(2 x 36 layers x 8 KV heads x 128 head_dim x 1 byte). The "int4" in the reported
description refers to WEIGHT quantization; KV quantization is a separate knob,
consistent with him later listing a "u4 KV cache" as its own lever. That lever
halves the footprint to 36,864 B/token and doubles residency per GB, so it
belongs in the sweep as a first-class axis rather than a footnote.

MEMORY ARCHITECTURE
-------------------
BomConfig carries a required memory_architecture field: "unified" or "discrete".

  unified  — weights and KV share one physical memory pool (LPDDR5x on AMD Strix
             Halo, Apple Silicon, etc.). reserved_gb covers OS + framework overhead
             against that unified pool. This is the target AI PC architecture.

  discrete — GPU has its own VRAM (GDDR6, GDDR6X). Weights and KV live in VRAM;
             system RAM is a separate pool and irrelevant to the inference budget.
             reserved_gb on a discrete config means VRAM driver/framework overhead,
             NOT system RAM. Discrete configs are valid measurement hosts for
             quality-vs-depth experiments (the curve transfers across memory
             architectures) but cannot stand in for unified BOM points: bandwidth,
             memory pressure dynamics, and feasibility numbers all differ.

Rule: informative_cells() and cheapest_config_per_model() exclude discrete
configs. A discrete result row is labelled with its host but never merged into
unified BOM analysis.

Architecture constants below are approximate and MUST be confirmed against the
actual model configs before publication; they drive every number here.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import pathlib

GB = 1_000_000_000  # decimal, to match the reported figures


# ------------------------------------------------------------------ models

@dataclasses.dataclass(frozen=True)
class Model:
    name: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    native_ctx: int

    def kv_bytes_per_token(self, kv_bits: int) -> int:
        return int(2 * self.layers * self.kv_heads * self.head_dim * (kv_bits / 8))

    def weight_bytes(self, weight_bits: int, overhead: float = 1.15) -> int:
        """Params at the given quantization, plus embedding/activation overhead."""
        return int(self.params_b * 1e9 * (weight_bits / 8) * overhead)


MODELS: tuple[Model, ...] = (
    # native_ctx confirmed 2026-08-13: ollama show qwen3:4b-instruct and qwen3:4b
    # both report 262144 (qwen3-2507 release). Previous value of 32768 was wrong
    # and inverted the binding conclusion for 16 GB configs.
    Model("qwen3-4b",  4.0,  36,  8, 128, 262_144),
    Model("qwen3-8b",  8.2,  36,  8, 128, 262_144),
    Model("qwen3-14b", 14.8, 40,  8, 128, 262_144),
    Model("qwen3-32b", 32.8, 64,  8, 128, 262_144),
)


# ------------------------------------------------------------------ configs

@dataclasses.dataclass(frozen=True)
class BomConfig:
    name: str
    memory_gb: int
    bom_cost_usd: int
    # Required: "unified" or "discrete". No default — must be explicit.
    # See module docstring for the distinction and why it matters.
    memory_architecture: str
    has_npu: bool = False
    npu_tops: int = 0
    # unified configs: OS + framework overhead against the shared pool.
    # discrete configs: VRAM driver/framework overhead (NOT system RAM).
    reserved_gb: float = 4.0
    notes: str = ""

    def __post_init__(self):
        if self.memory_architecture not in ("unified", "discrete"):
            raise ValueError(
                f"memory_architecture must be 'unified' or 'discrete', "
                f"got {self.memory_architecture!r}"
            )

    @property
    def usable_bytes(self) -> int:
        return int(max(0.0, self.memory_gb - self.reserved_gb) * GB)


# BOM candidate ladder — unified memory AI PC configurations.
LADDER: tuple[BomConfig, ...] = (
    BomConfig("8gb",  8,  549, memory_architecture="unified",
              notes="below Copilot+ floor"),
    BomConfig("16gb", 16, 799, memory_architecture="unified",
              notes="Copilot+ floor; on-silicon anchor available (XPS 16, Core Ultra 5)"),
    BomConfig("24gb", 24, 999, memory_architecture="unified",
              notes="common OEM upsell"),
    BomConfig("32gb", 32, 1399, memory_architecture="unified",
              has_npu=True, npu_tops=45, notes="premium AI PC"),
    BomConfig("64gb", 64, 1899, memory_architecture="unified",
              has_npu=True, npu_tops=45, notes="upper bound"),
)

# Measurement host — discrete GPU, NOT a BOM candidate.
# reserved_gb here is VRAM driver/framework overhead, not system RAM.
BLADE_HOST = BomConfig(
    "blade_rtx4070",
    memory_gb=8,
    bom_cost_usd=0,
    memory_architecture="discrete",
    reserved_gb=0.82,  # measured: 783 MiB framework workspace above model weights
    notes=(
        "measurement host, RTX 4070 Laptop 8188 MiB VRAM; "
        "NOT a BOM candidate; discrete GDDR6 memory, "
        "bandwidth and pressure dynamics unlike unified LPDDR5x targets"
    ),
)

KV_BITS = (8, 4)      # int8 KV (measured baseline) and the u4 lever
WEIGHT_BITS = (4,)    # int4 weights throughout


# ------------------------------------------------------------------ frontier

@dataclasses.dataclass(frozen=True)
class Cell:
    model: str
    config: str
    memory_gb: int
    bom_cost_usd: int
    memory_architecture: str
    kv_bits: int
    weight_bits: int
    weight_gb: float
    kv_bytes_per_token: int
    resident_tokens: int
    native_ctx: int
    fits: bool          # weights alone fit in usable memory
    binding: str        # "memory" | "context_window" | "infeasible"

    def to_row(self) -> dict:
        return dataclasses.asdict(self)


def evaluate(model: Model, cfg: BomConfig, kv_bits: int, weight_bits: int) -> Cell:
    w = model.weight_bytes(weight_bits)
    kv_per_tok = model.kv_bytes_per_token(kv_bits)
    left = cfg.usable_bytes - w

    if left <= 0:
        return Cell(model.name, cfg.name, cfg.memory_gb, cfg.bom_cost_usd,
                    cfg.memory_architecture,
                    kv_bits, weight_bits, round(w / GB, 2), kv_per_tok,
                    0, model.native_ctx, False, "infeasible")

    resident = left // kv_per_tok
    binding = "memory" if resident < model.native_ctx else "context_window"
    return Cell(model.name, cfg.name, cfg.memory_gb, cfg.bom_cost_usd,
                cfg.memory_architecture,
                kv_bits, weight_bits, round(w / GB, 2), kv_per_tok,
                int(resident), model.native_ctx, True, binding)


def frontier(models=MODELS, configs=LADDER,
             kv_bits=KV_BITS, weight_bits=WEIGHT_BITS) -> list[Cell]:
    return [evaluate(m, c, kb, wb)
            for m, c, kb, wb in itertools.product(models, configs, kv_bits, weight_bits)]


def informative_cells(cells: list[Cell]) -> list[Cell]:
    """
    Cells worth spending model calls on.

    Discrete-memory configs are excluded: they cannot stand in for unified BOM
    points because bandwidth, memory pressure, and feasibility numbers differ.

    A cell where the context window binds shows quality flat against memory by
    construction; measuring it repeatedly buys nothing. The informative cells are
    those where memory binds (quality should degrade as session is truncated),
    plus, for each model, the single cheapest unified config where it does NOT
    bind — that one is the control proving the effect is memory and not the model.
    """
    unified = [c for c in cells if c.memory_architecture == "unified"]
    out = [c for c in unified if c.binding == "memory"]
    for model in {c.model for c in unified}:
        controls = sorted(
            (c for c in unified if c.model == model and c.binding == "context_window"),
            key=lambda c: c.bom_cost_usd,
        )
        if controls:
            out.append(controls[0])
    return sorted(out, key=lambda c: (c.model, c.memory_gb, c.kv_bits))


def cheapest_config_per_model(cells: list[Cell]) -> dict[str, Cell | None]:
    """For each model, the cheapest unified config where it fits.

    Discrete configs are excluded — they are measurement hosts, not BOM points.
    Returns None for a model if no unified config can load its weights.
    """
    result: dict[str, Cell | None] = {}
    for model in {c.model for c in cells}:
        feasible = sorted(
            (c for c in cells
             if c.model == model and c.fits and c.memory_architecture == "unified"),
            key=lambda c: c.bom_cost_usd,
        )
        result[model] = feasible[0] if feasible else None
    return result


# ------------------------------------------------------------------ planning

def depth_grid(cell: Cell, points: int = 5) -> list[int]:
    """Context depths for a cell, capped by whichever constraint binds."""
    ceiling = min(cell.resident_tokens, cell.native_ctx)
    if ceiling < 1024:
        return [0]
    step = ceiling // points
    grid = [0]
    for i in range(1, points + 1):
        d = (i * step) // 1024 * 1024
        if d and d not in grid:
            grid.append(d)
    return grid


def plan(cells: list[Cell], n_probes: int, reps: int) -> dict:
    rows, total = [], 0
    for c in cells:
        d = depth_grid(c)
        n = len(d) * n_probes * reps
        total += n
        rows.append({**c.to_row(), "depths": d, "calls": n})
    return {"cells": rows, "total_calls": total,
            "units": "decimal GB (1e9)", "n_probes": n_probes, "reps": reps}


# ------------------------------------------------------------------ cli

def _fmt(cells: list[Cell]) -> str:
    hdr = (f"{'model':<11}{'cfg':<7}{'kv':>4}{'wt GB':>7}{'B/tok':>8}"
           f"{'resident':>11}{'binds on':>17}")
    lines = [hdr, "-" * len(hdr)]
    for c in cells:
        lines.append(f"{c.model:<11}{c.config:<7}{'u' + str(c.kv_bits):>4}"
                     f"{c.weight_gb:>7}{c.kv_bytes_per_token:>8}"
                     f"{c.resident_tokens:>11,}{c.binding:>17}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--all", action="store_true",
                    help="plan every cell, not just the informative ones")
    ap.add_argument("--emit", default=None, help="write plan JSON here")
    a = ap.parse_args()

    cells = frontier()
    print("FULL GRID (unified BOM configs)")
    print(_fmt(cells))

    # Also show the discrete measurement host for reference
    blade_cells = [evaluate(m, BLADE_HOST, kb, wb)
                   for m, kb, wb in itertools.product(MODELS, KV_BITS, WEIGHT_BITS)]
    print(f"\nMEASUREMENT HOST: {BLADE_HOST.name} (discrete — excluded from BOM analysis)")
    print(_fmt(blade_cells))

    infeasible = [c for c in cells if c.binding == "infeasible"]
    if infeasible:
        print(f"\n{len(infeasible)} cells cannot load weights at all:")
        for c in sorted({(c.model, c.config, c.weight_gb) for c in infeasible}):
            print(f"  {c[0]} on {c[1]}: weights alone are {c[2]} GB")

    inf = informative_cells(cells)
    print(f"\nINFORMATIVE CELLS ({len(inf)} of {len(cells)}, unified only)")
    print(_fmt(inf))

    p = plan(cells if a.all else inf, a.probes, a.reps)
    print(f"\nplanned calls: {p['total_calls']:,}  "
          f"({a.probes} probes x {a.reps} reps, units: {p['units']})")

    if a.emit:
        pathlib.Path(a.emit).write_text(json.dumps(p, indent=2))
        print("wrote", a.emit)
