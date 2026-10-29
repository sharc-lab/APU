"""A32 failure taxonomy (Task BM): class a CC probe failure from its child record.

Classes (amendment A32):
- budget_check: GenAI's construction-time assertion "Requested cache size is larger than
  available memory" / "budget_in_bytes <= total_available_memory" (runs only with an explicit
  cache_size, openvino.genai 2026.2.1.0 cache_orchestrator.hpp);
- device_alloc: CL_OUT_OF_RESOURCES, CL_MEM_OBJECT_ALLOCATION_FAILURE, an allocation or
  out-of-memory message, or clWaitForEvents errors after an allocation failure;
- port_input: input/port/shape errors;
- kv_mismatch: the A31 rejection;
- timeout: the child timed out with no error text (censored, A4);
- other.
Phase: construction (failed before "resident"), prefill (turn1 / grow / conc), decode (turn2).
"""

from __future__ import annotations

import re
from typing import Any

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "budget_check",
        re.compile(r"budget_in_bytes|Requested cache (size|blocks) (is|require)", re.I),
    ),
    (
        "device_alloc",
        re.compile(
            r"CL_OUT_OF_RESOURCES|CL_MEM_OBJECT_ALLOCATION_FAILURE|out of memory|"
            r"failed to allocate|allocation failed|CL_EXEC_STATUS_ERROR",
            re.I,
        ),
    ),
    (
        "kv_mismatch",
        re.compile(r"KvCompiledMismatch|KvCachePrecisionMismatch|kv_compiled_mismatch", re.I),
    ),
    ("port_input", re.compile(r"\bport\b|input shape|Shape mismatch|incompatible shape", re.I)),
)


def classify(child: dict[str, Any]) -> dict[str, Any]:
    """child: a probe's child result (failure_mode, exception, phase fields)."""
    mode = str(child.get("failure_mode") or "")
    exc = child.get("exception") or {}
    text = " ".join([mode, str(exc.get("type") or ""), str(exc.get("message") or "")])
    if mode.startswith(("turn1", "grow_turn", "conc_")):
        phase = "prefill"
    elif mode.startswith("turn2"):
        phase = "decode"
    elif mode.startswith("timeout"):
        phase = None  # the child record does not say which turn timed out
    elif mode:
        phase = "construction"
    else:
        phase = None
    cls = "other"
    for name, rx in _RULES:
        if rx.search(text):
            cls = name
            break
    else:
        if mode.startswith("timeout") or "timeout" in mode.lower():
            cls = "timeout"
    if not mode and not exc:
        cls = "none"
    return {"class": cls, "phase": phase, "censored": cls == "timeout"}
