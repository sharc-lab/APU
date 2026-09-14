"""Backend protocol shared by local and cloud execution.

The point of a single protocol is that the harness cannot accidentally treat the two differently.
Spec §7 M3.1: only the model/target assignment varies between conditions.

**Token counts are not the behavioral currency here.** :class:`GenerationResult` carries
``completion_chars`` and ``completion_bytes`` alongside token counts because Claude 4.7+ uses a
tokenizer that emits roughly 30% more tokens for identical text than earlier models, and local
models use their own. A cross-backend token delta therefore measures tokenizer disagreement plus
behavior and cannot separate them - see ``AMENDMENTS.md`` AM-021. Native token counts are retained
for **cost** only, where they are exactly right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "Backend",
    "GenerationRequest",
    "GenerationResult",
    "PreflightVerdict",
    "ToolCall",
    "ToolSpec",
]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool offered to the model.

    Identical across every backend and every condition. Rendered into each provider's **native**
    tool-calling mechanism rather than re-described in prose per model: using a provider's own
    protocol is not prompt tailoring, whereas writing a different tool description per model would
    be.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation emitted by the model."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One model call.

    ``messages`` is a provider-neutral transcript of ``{"role", "content"}`` entries. Each backend
    renders it through its own chat template or API; the *content* is byte-identical across
    backends.
    """

    messages: list[dict[str, Any]]
    system: str
    tools: tuple[ToolSpec, ...]
    max_tokens: int
    temperature: float = 0.0
    #: When True, backends that support it must constrain generation to a well-formed tool call
    #: (C2b: OpenVINO GenAI StructuredOutputConfig Tag+JSONSchema). Default False preserves
    #: pre-C2b callers.
    expect_tool_call: bool = False


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Outcome of one model call, with everything the ledger and the analysis need."""

    text: str
    tool_calls: tuple[ToolCall, ...]

    #: Native token counts. COST accounting only (AM-021).
    prompt_tokens: int
    completion_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    #: Tokenizer-independent output volume. These are the behavioral currency.
    completion_chars: int = 0
    completion_bytes: int = 0

    wall_ns: int = 0
    #: Time to first token. Separates prefill from decode for the throughput baseline.
    ttft_ns: int | None = None
    usd_cost: float = 0.0

    backend: str = ""
    model_ref: str = ""
    #: Identifier the provider echoed back, so a silent substitution is visible (AM-020).
    reported_model: str | None = None

    #: Populated when the call failed and the harness fell back. Never silently swallowed.
    error: str | None = None
    retried: bool = False
    fell_back_to_local: bool = False

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreflightVerdict:
    """Spec §5.1: a backend reports a definite verdict rather than crashing.

    ``UNSUPPORTED`` with a documented reason is a legitimate, citable outcome - a negative result
    is a data point.
    """

    status: Literal["OK", "UNSUPPORTED", "ERROR"]
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Uniform interface over local and cloud execution."""

    @property
    def name(self) -> str:
        """Stable backend identifier, e.g. ``openvino_cpu`` or ``anthropic``."""
        ...

    @property
    def model_ref(self) -> str:
        """Pinned model identifier as sent on the wire or loaded from disk."""
        ...

    def preflight(self) -> PreflightVerdict:
        """Report whether this backend can run, without crashing."""
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute one model call."""
        ...
