"""Execution backends for the agent harness.

One protocol (:mod:`seam.backends.base`) spans local and cloud execution so the harness can swap
the model endpoint and change nothing else. Spec §7 M3.1 requires exactly that: identical prompts,
tools, stopping criteria, and retry logic across conditions, with **only** the model/target
assignment varying. Any per-model prompt tailoring invalidates H1 and is forbidden.
"""

from seam.backends.base import (
    Backend,
    GenerationRequest,
    GenerationResult,
    PreflightVerdict,
    ToolCall,
    ToolSpec,
)

__all__ = [
    "Backend",
    "GenerationRequest",
    "GenerationResult",
    "PreflightVerdict",
    "ToolCall",
    "ToolSpec",
]
