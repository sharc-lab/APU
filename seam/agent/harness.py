"""Fixed agent scaffold (minimal M3.1).

One loop, one system prompt, one tool set, one stopping rule. **Only the model endpoint swaps.**
Spec §7 M3.1 makes that a hard requirement: any per-model prompt tailoring invalidates H1, so the
prompt, the tools, and the termination criteria live here as constants rather than as per-backend
options.

Each step emits a record conforming to spec §6.2, extended with the fields AM-021 and AM-022
require: tokenizer-independent output volume, the routing decision and its inputs, the overrun
flag, and the cost of evaluating the predictor.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from seam.agent.policy import StepType, ThroughputModel, decide
from seam.agent.tools import SYSTEM_PROMPT, TOOL_SPECS, TaskSpec, ToolWorld, normalize_answer
from seam.backends.base import Backend, GenerationRequest, GenerationResult, ToolCall
from seam.errors import BackendError
from seam.jsonlog import log_event
from seam.telemetry.rss import RssSampler

__all__ = ["StepRecord", "TaskResult", "run_task"]


@dataclass(slots=True)
class StepRecord:
    """Spec §6.2 step record, extended for M-SLICE."""

    run_id: str
    program_id: str
    step_idx: int
    step_type: StepType
    assigned_target: str
    model_ref: str
    t_start_ns: int
    t_end_ns: int

    # Native token counts - COST accounting only (AM-021).
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int

    # Tokenizer-independent behavioral currency (AM-021).
    completion_chars: int
    completion_bytes: int

    tool: dict[str, Any] | None
    usd_cost: float
    privacy_class: str
    terminated: bool
    retry_of: int | None

    # AM-022 routing telemetry.
    routing: dict[str, Any]
    #: True when a locally-executed step took longer than the deadline. Expected roughly half the
    #: time because n_out_pred is a median - this is the predictor's error rate, not a bug.
    deadline_overrun: bool
    actual_wall_s: float
    escalation_fallback: bool = False
    cloud_retried: bool = False
    error: str | None = None

    # ----------------------------------------------------------------------------------------
    # E-FILTER instrumentation (docs/EXPERIMENT_escalation_filter.md §3).
    #
    # Defaults keep every pre-E-FILTER caller behaviourally unchanged, but a default is not a
    # measurement: `kv_bytes_per_token=None` means "not supplied", and the E-FILTER analysis
    # refuses to run on records where it is, rather than treating a zero as a KV figure.
    # ----------------------------------------------------------------------------------------

    #: Transcript-length proxy the ROUTER consumed (``chars // 4 + scaffold`` under C2). Logged
    #: separately from the native ``prompt_tokens`` because the counterfactual escalation set is
    #: built on the proxy, so the proxy's error is part of the result and cannot be recovered
    #: after the fact.
    prompt_tokens_proxy: int = 0
    #: Full context presented this step, native tokenizer count. Grows through the trajectory.
    context_tokens_total: int = 0
    #: Tokens actually processed this step, post-cache: ``prompt_tokens - cached_prompt_tokens``.
    #: When ``cache_instrumented`` is False this is an UPPER BOUND, not a measurement.
    prompt_tokens_new: int = 0
    #: False when the runtime does not report cache reuse at all. Distinguishes "no caching
    #: occurred" from "not instrumented" - zero ``cached_prompt_tokens`` alone cannot.
    cache_instrumented: bool = False
    cache_evicted: bool = False
    evicted_bytes: int = 0
    #: Analytic, from the model config: see :mod:`seam.kvmath`. None means not supplied.
    kv_bytes_per_token: int | None = None
    #: Analytic KV bytes after prefill (``context_tokens_total * kv_bytes_per_token``).
    kv_bytes_resident_before: int | None = None
    #: Analytic KV bytes at the step's peak, i.e. including the tokens generated this step.
    kv_bytes_resident: int | None = None
    #: Observed process RSS high-water for the step. Includes weights and activations, so it is
    #: never the same quantity as ``kv_bytes_resident`` and is not a check on it.
    #: Alias of ``rss_peak_during_generate`` (retained for pre-C2g readers).
    peak_rss_bytes: int | None = None
    rss_bytes_end: int | None = None
    #: C2g: RSS at generate entry / peak during generate (background sampler) / generate return.
    rss_before_generate: int | None = None
    rss_peak_during_generate: int | None = None
    rss_after_generate: int | None = None
    #: C2g: host free-memory minimum (MiB) sampled during this generate window.
    free_memory_mb_min_during_generate: float | None = None
    free_memory_mb_before_generate: float | None = None
    free_memory_mb_after_generate: float | None = None
    #: Predictor decomposition (pre-registration §3). ``t_pred_total_s`` equals
    #: ``routing["t_pred_s"]``.
    t_pred_prefill_s: float | None = None
    t_pred_decode_s: float | None = None
    t_pred_total_s: float | None = None
    #: Time to first token; splits the step's realized prefill from its realized decode.
    ttft_ns: int | None = None


@dataclass(slots=True)
class TaskResult:
    """Outcome of one task trajectory."""

    task_id: str
    program_id: str
    success: bool
    submitted_answer: str | None
    expected: str
    realized_steps: int
    escalated_steps: int
    local_steps: int
    jct_s: float
    local_tokens: int
    cloud_tokens: int
    local_completion_chars: int
    cloud_completion_chars: int
    usd_cost: float
    deadline_overruns: int
    steps: list[StepRecord] = field(default_factory=list)
    terminated_reason: str = ""
    #: Set when ``terminated_reason == "context_cap"``: the projected next-step context that
    #: tripped the cap (C2c). Distinct from peak completed-step ``context_tokens_total``.
    projected_next_context_tokens: int | None = None

    @property
    def escalation_rate(self) -> float:
        return self.escalated_steps / self.realized_steps if self.realized_steps else 0.0

    def to_summary(self) -> dict[str, Any]:
        payload = {k: v for k, v in asdict(self).items() if k != "steps"}
        payload["escalation_rate"] = self.escalation_rate
        return payload


def run_task(
    *,
    task: TaskSpec,
    world: ToolWorld,
    local_backend: Backend,
    cloud_backend: Backend,
    throughput: ThroughputModel,
    deadline_s: float,
    n_out_pred_tokens: dict[str, int],
    max_steps: int,
    max_tokens: int,
    run_id: str,
    program_id: str,
    kv_bytes_per_token: int | None = None,
    rss_sample_interval_s: float | None = None,
    step_sink: Callable[[StepRecord], None] | None = None,
    prompt_token_scaffold_tokens: int = 0,
    context_cap_tokens: int | None = None,
    constrain_tool_calls: bool = False,
) -> TaskResult:
    """Run one task under the deadline-aware local-first policy.

    The transcript is provider-neutral and identical across arms; each backend renders it through
    its own native chat template and tool protocol.

    Args:
        kv_bytes_per_token: Analytic KV constant from :mod:`seam.kvmath`. When None the KV fields of
            each step record stay None rather than zero, so an unsupplied constant cannot be read
            downstream as a measured one.
        rss_sample_interval_s: Sampling interval for the per-step RSS high-water mark. None
            disables sampling, which leaves the timing of pre-E-FILTER callers untouched.
        step_sink: Called with each completed record. Used to stream ``steps.ndjson`` as the run
            proceeds, so a crash mid-trajectory does not lose the steps already measured.
        prompt_token_scaffold_tokens: Fixed chat-template scaffold added to the ``chars // 4``
            proxy (C2.3). Per-model and per-template; default 0 preserves pre-C2 callers.
        context_cap_tokens: When set, project the next step's context before generate and terminate
            with ``terminated_reason="context_cap"`` if it would exceed the cap (C2b).
        constrain_tool_calls: When True, set ``GenerationRequest.expect_tool_call`` so backends
            that support it apply constrained tool-call decoding (C2b).
    """
    world.begin_task()
    messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
    steps: list[StepRecord] = []
    submitted: str | None = None
    terminated_reason = "max_steps"
    projected_next_context_tokens: int | None = None
    prev_kv_peak: int | None = None
    t_job0 = time.perf_counter_ns()

    for step_idx in range(max_steps):
        # C2b: n_out_pred is held constant across step types (context-selectivity). The taxonomy
        # label remains tool_call_synthesis for routing until a terminal submit_answer rewrites it
        # after the fact for logging only.
        step_type: StepType = "tool_call_synthesis"

        prompt_tokens_est = _estimate_prompt_tokens(
            messages, scaffold_tokens=prompt_token_scaffold_tokens
        )
        decision = decide(
            throughput=throughput,
            deadline_s=deadline_s,
            prompt_tokens=prompt_tokens_est,
            step_type=step_type,
            n_out_pred_tokens=n_out_pred_tokens,
        )

        backend: Backend = cloud_backend if decision.assigned_target == "cloud" else local_backend
        request = GenerationRequest(
            messages=list(messages),
            system=SYSTEM_PROMPT,
            tools=TOOL_SPECS,
            max_tokens=max_tokens,
            temperature=0.0,
            expect_tool_call=constrain_tool_calls,
        )

        projected = _project_context_tokens(
            backend,
            request,
            messages=messages,
            scaffold_tokens=prompt_token_scaffold_tokens,
        )
        if context_cap_tokens is not None and projected > int(context_cap_tokens):
            terminated_reason = "context_cap"
            projected_next_context_tokens = int(projected)
            peak_completed = max((s.context_tokens_total for s in steps), default=0)
            log_event(
                "harness.context_cap",
                message=(
                    f"step {step_idx}: projected_next_context_tokens={projected} > "
                    f"cap {context_cap_tokens} (peak completed context={peak_completed}); "
                    f"terminating without generate"
                ),
                run_id=run_id,
                program_id=program_id,
                step_idx=step_idx,
                projected_next_context_tokens=int(projected),
                projected_context_tokens=int(projected),  # alias retained for older readers
                peak_completed_context_tokens=peak_completed,
                context_cap_tokens=int(context_cap_tokens),
            )
            break

        rss = RssSampler(interval_s=rss_sample_interval_s) if rss_sample_interval_s else None
        if rss is not None:
            rss.start()
        t0 = time.perf_counter_ns()
        fell_back = False
        error: str | None = None
        try:
            result = backend.generate(request)
        except BackendError as exc:
            # AM-022 item 5: after one retry inside the backend, fall back to local and MARK the
            # step. The fallback perturbs the realized partition, so it is never silent.
            if decision.assigned_target != "cloud":
                raise
            log_event(
                "harness.cloud_fallback_to_local",
                severity="warning",
                message=f"step {step_idx}: cloud unavailable, falling back to local: {exc}",
                run_id=run_id,
                program_id=program_id,
                step_idx=step_idx,
            )
            fell_back = True
            error = str(exc)
            result = local_backend.generate(request)
        t1 = time.perf_counter_ns()
        rss_window = rss.stop() if rss is not None else None

        actual_wall_s = (t1 - t0) / 1e9
        executed_on = "local" if (fell_back or decision.assigned_target == "local") else "cloud"
        overrun = executed_on == "local" and actual_wall_s > deadline_s

        tool_record, tool_result_text, terminal_answer = _apply_tool_calls(result, world)
        if terminal_answer is not None:
            submitted = terminal_answer
            step_type = "answer_synthesis"
            terminated_reason = "submitted"

        # A runtime that does not report cache reuse must say so: a zero cached-token count is
        # otherwise indistinguishable from "no reuse happened", and the two support opposite
        # readings of the caching fork in the pre-registration §6.
        cache_instrumented = bool(result.extra.get("cache_instrumented", False))
        context_tokens_total = int(result.prompt_tokens)
        prompt_tokens_new = max(context_tokens_total - int(result.cache_read_input_tokens), 0)
        kv_before = (
            context_tokens_total * kv_bytes_per_token if kv_bytes_per_token is not None else None
        )
        kv_peak = (
            (context_tokens_total + int(result.completion_tokens)) * kv_bytes_per_token
            if kv_bytes_per_token is not None
            else None
        )
        # Eviction is inferred from the transcript, not from a runtime flag: if the whole context
        # was re-processed on a step that had a predecessor, nothing survived from that predecessor.
        evicted = step_idx > 0 and prompt_tokens_new >= context_tokens_total
        rss_before = rss_window.start_bytes if rss_window else None
        rss_peak = rss_window.peak_bytes if rss_window else None
        rss_after = rss_window.end_bytes if rss_window else None
        steps.append(
            StepRecord(
                run_id=run_id,
                program_id=program_id,
                step_idx=step_idx,
                step_type=step_type,
                assigned_target=executed_on,
                model_ref=result.model_ref,
                t_start_ns=t0,
                t_end_ns=t1,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cached_prompt_tokens=result.cache_read_input_tokens,
                completion_chars=result.completion_chars,
                completion_bytes=result.completion_bytes,
                tool=tool_record,
                usd_cost=result.usd_cost,
                privacy_class="synthetic_benchmark",
                terminated=terminal_answer is not None,
                retry_of=None,
                routing=decision.to_record(),
                deadline_overrun=overrun,
                actual_wall_s=actual_wall_s,
                escalation_fallback=fell_back,
                cloud_retried=result.retried,
                error=error,
                prompt_tokens_proxy=prompt_tokens_est,
                context_tokens_total=context_tokens_total,
                prompt_tokens_new=prompt_tokens_new,
                cache_instrumented=cache_instrumented,
                cache_evicted=evicted,
                evicted_bytes=prev_kv_peak if evicted and prev_kv_peak else 0,
                kv_bytes_per_token=kv_bytes_per_token,
                kv_bytes_resident_before=kv_before,
                kv_bytes_resident=kv_peak,
                peak_rss_bytes=rss_peak,
                rss_bytes_end=rss_after,
                rss_before_generate=rss_before,
                rss_peak_during_generate=rss_peak,
                rss_after_generate=rss_after,
                free_memory_mb_min_during_generate=(
                    rss_window.free_memory_mb_min if rss_window else None
                ),
                free_memory_mb_before_generate=(
                    rss_window.free_memory_mb_start if rss_window else None
                ),
                free_memory_mb_after_generate=(
                    rss_window.free_memory_mb_end if rss_window else None
                ),
                t_pred_prefill_s=prompt_tokens_est / throughput.r_prefill_tok_s,
                t_pred_decode_s=decision.n_out_pred_tokens / throughput.r_decode_tok_s,
                t_pred_total_s=decision.t_pred_s,
                ttft_ns=result.ttft_ns,
            )
        )
        prev_kv_peak = kv_peak
        if step_sink is not None:
            step_sink(steps[-1])

        if terminal_answer is not None:
            break

        messages.append({"role": "assistant", "content": result.text or "(no output)"})
        if tool_result_text is None:
            # No parseable tool call and no answer. Nudge with a fixed, model-agnostic message -
            # identical in every condition, so it cannot function as per-model tailoring.
            messages.append(
                {
                    "role": "user",
                    "content": "Call a tool, or call submit_answer with the final answer.",
                }
            )
        else:
            messages.append({"role": "user", "content": tool_result_text})

    jct_s = (time.perf_counter_ns() - t_job0) / 1e9
    local_steps = sum(1 for s in steps if s.assigned_target == "local")
    cloud_steps = sum(1 for s in steps if s.assigned_target == "cloud")

    return TaskResult(
        task_id=task.task_id,
        program_id=program_id,
        success=submitted is not None
        and normalize_answer(submitted) == normalize_answer(task.expected),
        submitted_answer=submitted,
        expected=task.expected,
        realized_steps=len(steps),
        escalated_steps=cloud_steps,
        local_steps=local_steps,
        jct_s=jct_s,
        local_tokens=sum(s.completion_tokens for s in steps if s.assigned_target == "local"),
        cloud_tokens=sum(s.completion_tokens for s in steps if s.assigned_target == "cloud"),
        local_completion_chars=sum(
            s.completion_chars for s in steps if s.assigned_target == "local"
        ),
        cloud_completion_chars=sum(
            s.completion_chars for s in steps if s.assigned_target == "cloud"
        ),
        usd_cost=sum(s.usd_cost for s in steps),
        deadline_overruns=sum(1 for s in steps if s.deadline_overrun),
        steps=steps,
        terminated_reason=terminated_reason,
        projected_next_context_tokens=projected_next_context_tokens,
    )


def _apply_tool_calls(
    result: GenerationResult, world: ToolWorld
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Execute the first tool call, if any.

    Returns ``(tool_record, tool_result_text, terminal_answer)``. Only the **first** call is
    executed: a model that emits several in one step is exhibiting behavior worth recording, and
    executing all of them would change the realized step count in a way that differs by model.
    """
    if not result.tool_calls:
        return None, None, None

    call: ToolCall = result.tool_calls[0]
    if call.name == "submit_answer":
        answer = str(call.arguments.get("answer", ""))
        return (
            {"name": call.name, "duration_ns": 0, "result_bytes": len(answer), "error": None},
            None,
            answer,
        )

    t0 = time.perf_counter_ns()
    text, error = world.execute(call.name, call.arguments)
    duration_ns = time.perf_counter_ns() - t0
    payload = text if error is None else f"error: {error}"
    return (
        {
            "name": call.name,
            "duration_ns": duration_ns,
            "result_bytes": len(payload.encode("utf-8")),
            "error": error,
        },
        f"Tool {call.name} returned: {payload}",
        None,
    )


def _estimate_prompt_tokens(messages: list[dict[str, Any]], *, scaffold_tokens: int = 0) -> int:
    """Cheap transcript-length proxy used by the router.

    The router must price a step **before** it runs, so it cannot use the true tokenizer count for
    a prompt it has not yet rendered. This proxy is identical in both arms, which is what the
    isolation invariant requires: it may be biased, but it is biased identically, so it cannot
    manufacture a difference between targets.

    C2.3 lands ``chars // 4 + scaffold``. The scaffold is the fixed chat-template / tool-schema
    overhead the content-only proxy never sees; it is per-model and per-template and must be
    re-measured when either changes (see ``template_scaffold_analysis``).
    """
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return chars // 4 + max(0, int(scaffold_tokens))


def _project_context_tokens(
    backend: Backend,
    request: GenerationRequest,
    *,
    messages: list[dict[str, Any]],
    scaffold_tokens: int,
) -> int:
    """Project the next step's context tokens before ``generate`` (C2b context cap).

    Prefer the backend's native tokenizer count of the rendered prompt when available; fall back
    to the router proxy so stubs and non-OpenVINO backends still enforce the cap.
    """
    estimator = getattr(backend, "estimate_context_tokens", None)
    if callable(estimator):
        return int(estimator(request))
    return _estimate_prompt_tokens(messages, scaffold_tokens=scaffold_tokens)
