"""Local CPU backend via OpenVINO GenAI (minimal M4).

Provides the ``cpu-p`` / ``cpu-lpe`` contrast that M-SLICE varies. One runtime spans CPU, iGPU, and
NPU on Intel, so this abstraction carries forward to full M4 unchanged.

CORE SELECTION IS NOT SELF-VERIFYING
------------------------------------
Core placement is requested through OpenVINO's own hybrid awareness (``SCHEDULING_CORE_TYPE``),
because that is the mechanism that survives into the iGPU/NPU targets. But Panther Lake's
low-power die has 4 P-cores and 4 **LP-E** cores and **no standard E-cores**, so ``ECORE_ONLY`` may
map onto LP-E, or may match nothing and silently fall back to every core.

**Setting the property is not evidence that it took effect.** If affinity leaks, both arms run on
the same cores, the experiment returns a null for reasons having nothing to do with the science,
and the null looks exactly like a real one. ``seam/tools/verify_core_affinity.py`` therefore
confirms per-core utilization against the M1-committed mapping before any measurement, and
:class:`LocalOpenVinoBackend` supports a process-affinity fallback via
:func:`seam.topology.affinity_for` when OpenVINO's mapping disagrees.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from seam.backends.base import (
    GenerationRequest,
    GenerationResult,
    PreflightVerdict,
    ToolCall,
    ToolSpec,
)
from seam.errors import BackendError
from seam.jsonlog import log_event
from seam.measurement import capture_process_affinity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from seam.topology import CpuTarget

__all__ = [
    "CONSTRAINED_DECODING_MECHANISM",
    "LocalOpenVinoBackend",
    "OpenVinoRuntimeInfo",
    "build_tool_call_structured_output_config",
    "parse_tool_calls",
    "resolve_ttft_ns",
    "runtime_info",
]

#: Manifest / report identifier for the C2b constrained-decoding path on this backend.
CONSTRAINED_DECODING_MECHANISM: Final = "openvino_genai_structured_output_config_tag_jsonschema"

#: Qwen-family tool-call envelope. The model emits native ``<tool_call>`` blocks from its own chat
#: template; we parse that rather than instructing it into a bespoke format, which would be
#: per-model prompt tailoring.
_TOOL_CALL_RE: Final = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

#: Qwen3 reasoning envelope. Presence/absence is the empirical check that ``enable_thinking``
#: actually took effect (AM-024) - the flag being set proves nothing on its own.
_THINK_RE: Final = re.compile(r"<think>", re.IGNORECASE)

_NS_PER_MS: Final = 1_000_000


def _template_accepts_thinking(tokenizer: Any) -> bool:
    """Does this model's chat template gate reasoning behind ``enable_thinking``?"""
    template = getattr(tokenizer, "chat_template", None)
    return isinstance(template, str) and "enable_thinking" in template


@dataclass(frozen=True, slots=True)
class OpenVinoRuntimeInfo:
    """Versions recorded into every manifest's ``drivers`` block (spec §5.2)."""

    openvino: str
    genai: str


def runtime_info() -> OpenVinoRuntimeInfo:
    import openvino
    import openvino_genai

    return OpenVinoRuntimeInfo(
        openvino=str(openvino.__version__),
        genai=str(getattr(openvino_genai, "__version__", "unknown")),
    )


class LocalOpenVinoBackend:
    """INT4 OpenVINO IR on CPU, pinned to one core cluster.

    The pipeline is constructed once and reused: compilation is expensive and would otherwise be
    charged to the first step of every run, inflating that step's measured latency and biasing the
    escalation decision that depends on it.
    """

    def __init__(
        self,
        *,
        model_dir: Path,
        target: CpuTarget,
        scheduling_core_type: str | None,
        inference_num_threads: int,
        enable_cpu_pinning: bool | None = True,
        model_ref: str = "",
        enable_thinking: bool = False,
        affinity_cpus: list[int] | None = None,
    ) -> None:
        self._model_dir = model_dir
        self._target = target
        self._scheduling_core_type = scheduling_core_type
        self._inference_num_threads = inference_num_threads
        self._enable_cpu_pinning = enable_cpu_pinning
        self._model_ref = model_ref or model_dir.name
        # AM-024: reasoning mode is an experimental axis, declared per arm and recorded in the
        # manifest. It is applied identically to both targets, so it never differs between arms
        # of the silicon contrast.
        self._enable_thinking = enable_thinking
        # Process-affinity fallback (spec §4). Used only when SCHEDULING_CORE_TYPE is shown NOT to
        # confine work to the M1-committed cluster; applied before compilation so the pipeline's
        # worker threads inherit the mask.
        self._affinity_cpus = list(affinity_cpus) if affinity_cpus else None
        self._process_affinity_before_load: list[int] | None = None
        self._process_affinity_after_apply: list[int] | None = None
        self._process_affinity_after_load: list[int] | None = None
        self._pipe: Any = None
        self._tokenizer: Any = None

    @property
    def enable_thinking(self) -> bool:
        return self._enable_thinking

    @property
    def reasoning_mode(self) -> str:
        return "thinking_on" if self._enable_thinking else "thinking_off"

    def _apply_process_affinity(self) -> None:
        if not self._affinity_cpus:
            return
        import psutil

        proc = psutil.Process()
        proc.cpu_affinity(self._affinity_cpus)
        log_event(
            "backend.local.process_affinity_applied",
            severity="warning",
            message=(
                f"{self._target}: OpenVINO core selection was insufficient; pinned the process "
                f"to cpus {self._affinity_cpus} via topology.affinity_for()"
            ),
            target=self._target,
            affinity_cpus=self._affinity_cpus,
        )

    @property
    def name(self) -> str:
        return "openvino_cpu"

    @property
    def model_ref(self) -> str:
        return self._model_ref

    @property
    def target(self) -> CpuTarget:
        return self._target

    @property
    def tokenizer(self) -> Any:
        """Loaded HF tokenizer (after :meth:`load`)."""
        self.load()
        return self._tokenizer

    def properties(self) -> dict[str, Any]:
        """Exact OpenVINO properties applied. Recorded in the manifest.

        ``None`` means *do not set the property at all*, which is a materially different
        condition from setting it to a default-looking value: the confinement matrix has to be
        able to distinguish "pinning left at OpenVINO's default" from "pinning explicitly
        requested".
        """
        props: dict[str, Any] = {}
        if self._scheduling_core_type is not None:
            props["SCHEDULING_CORE_TYPE"] = self._scheduling_core_type
        if self._enable_cpu_pinning is not None:
            props["ENABLE_CPU_PINNING"] = self._enable_cpu_pinning
        props["INFERENCE_NUM_THREADS"] = self._inference_num_threads
        return props

    def config_record(self) -> dict[str, Any]:
        """Everything about this backend that a manifest must carry."""
        return {
            "properties": self.properties(),
            "reasoning_mode": self.reasoning_mode,
            "enable_thinking": self._enable_thinking,
            "affinity_cpus": self._affinity_cpus,
            "affinity_mechanism": (
                "process_affinity" if self._affinity_cpus else "openvino_scheduling_core_type"
            ),
            "process_affinity_readback": {
                "before_load": self._process_affinity_before_load,
                "after_apply": self._process_affinity_after_apply,
                "after_load": self._process_affinity_after_load,
            },
            "model_dir": str(self._model_dir),
        }

    # ---------------------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------------------

    def close(self) -> None:
        """Release the compiled pipeline and tokenizer so runtime caches can free (C2c).

        Idempotent. Does not clear affinity configuration; the next :meth:`load` reapplies it.
        Callers that need the allocator to return pages should follow with ``gc.collect()``.
        """
        had_pipe = self._pipe is not None
        self._pipe = None
        self._tokenizer = None
        if had_pipe:
            log_event(
                "backend.local.closed",
                message=f"released pipeline for {self._model_ref} ({self._target})",
                target=self._target,
            )

    def reload(self) -> None:
        """Destroy any loaded pipeline and recompile from the IR directory (C2c)."""
        self.close()
        self.load()

    def load(self) -> None:
        """Compile the pipeline. Idempotent."""
        if self._pipe is not None:
            return
        import openvino_genai as ov_genai
        from transformers import AutoTokenizer

        if not self._model_dir.exists():
            raise BackendError(
                f"OpenVINO IR not found at {self._model_dir}. Export it with "
                f"seam/tools/export_model.py - models are never hand-converted (spec §5.3)."
            )

        self._process_affinity_before_load = capture_process_affinity()
        self._apply_process_affinity()
        self._process_affinity_after_apply = capture_process_affinity()

        t0 = time.perf_counter_ns()
        self._pipe = ov_genai.LLMPipeline(str(self._model_dir), "CPU", **self.properties())
        self._process_affinity_after_load = capture_process_affinity()
        # The HF tokenizer renders the model's OWN chat template, including its native `tools`
        # support. Using the model's template is not tailoring; hand-writing a tool format per
        # model would be.
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
        compile_ns = time.perf_counter_ns() - t0

        log_event(
            "backend.local.loaded",
            message=f"compiled {self._model_ref} for {self._target} in {compile_ns / 1e9:.1f}s",
            target=self._target,
            compile_ns=compile_ns,
            process_affinity_before_load=self._process_affinity_before_load,
            process_affinity_after_apply=self._process_affinity_after_apply,
            process_affinity_after_load=self._process_affinity_after_load,
            **self.properties(),
        )

    def preflight(self) -> PreflightVerdict:
        try:
            self.load()
        except BackendError as exc:
            return PreflightVerdict("UNSUPPORTED", reason=str(exc))
        except Exception as exc:  # reported as a verdict, never swallowed
            log_event(
                "backend.local.preflight_error",
                severity="error",
                message=f"{type(exc).__name__}: {exc}",
                target=self._target,
            )
            return PreflightVerdict("ERROR", reason=f"{type(exc).__name__}: {exc}")
        return PreflightVerdict(
            "OK",
            detail={"properties": self.properties(), "model_dir": str(self._model_dir)},
        )

    # ---------------------------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------------------------

    def render_prompt(self, request: GenerationRequest) -> str:
        """Apply the model's native chat template, including tool definitions."""
        if self._tokenizer is None:
            self.load()
        messages = [{"role": "system", "content": request.system}, *request.messages]
        kwargs: dict[str, Any] = {}
        # Qwen3 is a hybrid-reasoning family whose template gates thinking behind this flag and
        # defaults it ON. Passing it explicitly is a template setting applied identically to both
        # targets, not per-model prompt tailoring. Templates that do not accept the kwarg are left
        # alone rather than being force-fed an argument they would reject.
        if _template_accepts_thinking(self._tokenizer):
            kwargs["enable_thinking"] = self._enable_thinking
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tools=[_tool_to_openai_schema(t) for t in request.tools],
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )
        return str(rendered)

    def estimate_context_tokens(self, request: GenerationRequest) -> int:
        """Native tokenizer count of the rendered prompt (C2b context-cap projection)."""
        self.load()
        prompt = self.render_prompt(request)
        return len(self._tokenizer(prompt)["input_ids"])

    def generate(self, request: GenerationRequest, *, ignore_eos: bool = False) -> GenerationResult:
        import openvino_genai as ov_genai

        self.load()
        prompt = self.render_prompt(request)

        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = request.max_tokens
        # Greedy. Sampling noise would land inside the very variance the noise floor is meant to
        # measure, and would differ between arms by luck rather than by silicon.
        cfg.do_sample = False
        if ignore_eos and hasattr(cfg, "ignore_eos"):
            cfg.ignore_eos = True

        constrained = False
        constrained_mechanism: str | None = None
        if request.expect_tool_call:
            cfg.structured_output_config = build_tool_call_structured_output_config(
                ov_genai, request.tools
            )
            # Stop when the forced envelope closes; keep the stop string in the text so
            # parse_tool_calls still sees a complete ``</tool_call>`` close tag.
            if hasattr(cfg, "stop_strings"):
                cfg.stop_strings = {"</tool_call>"}
            if hasattr(cfg, "include_stop_str_in_output"):
                cfg.include_stop_str_in_output = True
            constrained = True
            constrained_mechanism = CONSTRAINED_DECODING_MECHANISM

        # GenAI 2026.2.1 returns a bare ``str`` (no ``perf_metrics``) for a single string
        # prompt. A one-element list yields ``DecodedResults`` with TTFT/token counts.
        # Keep a first-token streamer as wall-clock fallback if metrics are absent.
        streamer = _make_ttft_streamer(ov_genai)
        t0 = time.perf_counter_ns()
        streamer.t0_ns = t0
        result = self._pipe.generate([prompt], cfg, streamer)
        wall_ns = time.perf_counter_ns() - t0

        text = _text_from_generate_result(result)
        metrics = getattr(result, "perf_metrics", None)
        metrics_ttft_ns, prompt_tokens, completion_tokens = _extract_metrics(metrics)
        ttft_ns, ttft_source = resolve_ttft_ns(metrics_ttft_ns, streamer.ttft_ns)
        if ttft_ns is None:
            raise BackendError(
                "OpenVINO GenAI returned no TTFT (perf_metrics absent and streamer saw no "
                "first token). Prefill/decode split would be fabricated; refusing to continue."
            )
        # Always record both candidates - short perf_metrics TTFT vs streamer wall must be
        # auditable when phase-window sample counts look sparse.
        if prompt_tokens == 0:
            prompt_tokens = len(self._tokenizer(prompt)["input_ids"])
        if completion_tokens == 0:
            completion_tokens = len(self._tokenizer(text)["input_ids"])

        return GenerationResult(
            text=text,
            tool_calls=parse_tool_calls(text),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_chars=len(text),
            completion_bytes=len(text.encode("utf-8")),
            wall_ns=wall_ns,
            ttft_ns=ttft_ns,
            usd_cost=0.0,
            backend=self.name,
            model_ref=self._model_ref,
            reported_model=self._model_ref,
            extra={
                "target": self._target,
                "properties": self.properties(),
                "reasoning_mode": self.reasoning_mode,
                # AM-024 requires empirical confirmation that the flag took effect, in BOTH
                # directions. Setting it is not evidence.
                "think_block_present": _THINK_RE.search(text) is not None,
                # OpenVINO GenAI's LLMPipeline.generate() takes a fully rendered prompt and reports
                # no cache-reuse counters, so `cache_read_input_tokens` is structurally 0 here. That
                # is NOT the same claim as "no reuse occurred", and the E-FILTER caching fork turns
                # on the difference - see docs/EXPERIMENT_escalation_filter.md §6.
                "cache_instrumented": False,
                "cache_instrumentation_note": (
                    "openvino_genai LLMPipeline exposes no cache_read/cache_creation counters; "
                    "cached_prompt_tokens is a structural zero, not a measurement"
                ),
                "ttft_source": ttft_source,
                "ttft_ns_perf_metrics": metrics_ttft_ns,
                "ttft_ns_streamer": streamer.ttft_ns,
                "constrained_decoding": constrained,
                "constrained_decoding_mechanism": constrained_mechanism,
            },
        )


def resolve_ttft_ns(
    metrics_ttft_ns: int | None, streamer_ttft_ns: int | None
) -> tuple[int | None, str]:
    """Prefer GenAI perf_metrics TTFT; fall back to first-token streamer timestamp."""
    if metrics_ttft_ns is not None and metrics_ttft_ns > 0:
        return metrics_ttft_ns, "perf_metrics"
    if streamer_ttft_ns is not None and streamer_ttft_ns > 0:
        return streamer_ttft_ns, "streamer_first_token"
    return None, "unavailable"


def _text_from_generate_result(result: Any) -> str:
    """Extract completion text from ``DecodedResults``, ``EncodedResults``, or ``str``."""
    texts = getattr(result, "texts", None)
    if texts:
        return str(texts[0])
    return str(result)


def _make_ttft_streamer(ov_genai: Any) -> Any:
    """Build a ``StreamerBase`` that records wall time to the first generated token."""

    class _TtftStreamer(ov_genai.StreamerBase):
        def __init__(self) -> None:
            super().__init__()
            self.t0_ns: int | None = None
            self.ttft_ns: int | None = None

        def write(self, _token: Any) -> Any:
            if self.ttft_ns is None and self.t0_ns is not None:
                self.ttft_ns = time.perf_counter_ns() - self.t0_ns
            return ov_genai.StreamingStatus.RUNNING

        def end(self) -> None:
            return None

    return _TtftStreamer()


def _extract_metrics(metrics: Any) -> tuple[int | None, int, int]:
    """Pull TTFT and token counts out of GenAI perf metrics, tolerating API drift.

    Returns ``(ttft_ns, prompt_tokens, completion_tokens)``; unavailable values come back as
    ``None`` / ``0`` so the caller can fall back to tokenizing, rather than a wrong number being
    invented here. Callers must not treat missing TTFT as zero.
    """
    if metrics is None:
        return None, 0, 0
    ttft_ns: int | None = None
    try:
        ttft_ms = float(metrics.get_ttft().mean)
        ttft_ns = int(ttft_ms * _NS_PER_MS) if ttft_ms > 0.0 else None
    except Exception:  # optional metric; absence is not a failure
        ttft_ns = None
    prompt_tokens = 0
    completion_tokens = 0
    try:
        prompt_tokens = int(metrics.get_num_input_tokens())
    except Exception:
        prompt_tokens = int(getattr(metrics, "num_input_tokens", 0) or 0)
    try:
        completion_tokens = int(metrics.get_num_generated_tokens())
    except Exception:
        completion_tokens = int(getattr(metrics, "num_generated_tokens", 0) or 0)
    return ttft_ns, prompt_tokens, completion_tokens


def _tool_to_openai_schema(tool: ToolSpec) -> dict[str, Any]:
    """Render a :class:`ToolSpec` into the OpenAI-style shape HF chat templates expect."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def build_tool_call_structured_output_config(ov_genai: Any, tools: tuple[ToolSpec, ...]) -> Any:
    """Force a Qwen ``<tool_call>{...}</tool_call>`` envelope via GenAI structured output.

    Mechanism (C2b): ``StructuredOutputConfig.compound_grammar`` =
    ``Tag("<tool_call>", JSONSchema(name enum + arguments object), "</tool_call>")``.
    Not post-hoc repair - malformed JSON is never patched into a tool call.
    """
    names = [t.name for t in tools]
    if not names:
        raise BackendError(
            "expect_tool_call=True but GenerationRequest.tools is empty; cannot build "
            "structured-output schema"
        )
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": names},
            "arguments": {"type": "object"},
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }
    tag = ov_genai.StructuredOutputConfig.Tag(
        "<tool_call>",
        ov_genai.StructuredOutputConfig.JSONSchema(json.dumps(schema, sort_keys=True)),
        "</tool_call>",
    )
    config = ov_genai.StructuredOutputConfig()
    config.compound_grammar = tag
    return config


def parse_tool_calls(text: str) -> tuple[ToolCall, ...]:
    """Extract tool calls from a Qwen-style ``<tool_call>`` envelope.

    A malformed call is **dropped and logged**, not repaired. Silently fixing broken JSON would
    inflate the local model's apparent capability, which is exactly the quantity the unconstrained
    success-rate check exists to measure.
    """
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_RE.finditer(text):
        blob = match.group(1)
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError as exc:
            log_event(
                "backend.tool_call_unparseable",
                severity="warning",
                message=f"dropping malformed tool call: {exc}",
                raw=blob[:400],
            )
            continue
        name = payload.get("name")
        if not isinstance(name, str):
            log_event(
                "backend.tool_call_unnamed",
                severity="warning",
                message="dropping tool call with no name",
                raw=blob[:400],
            )
            continue
        args = payload.get("arguments")
        calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {}))
    return tuple(calls)
