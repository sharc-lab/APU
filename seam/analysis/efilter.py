"""E-FILTER Stage 1: the counterfactual escalation filter, replayed offline.

Pre-registration: ``docs/EXPERIMENT_escalation_filter.md`` §2 (Stage 1), §4 (envelope metrics),
§5 (analysis), §6 (the caching fork), §7 (predictions).

The collection run (:mod:`seam.tools.efilter_run`) executes every step locally with escalation
disabled and logs the inputs the router consumed. This module replays the **real** routing rule
over those logged inputs at every deadline in a grid, and reports the local resource envelope of
the steps that survive as a function of deadline.

Three properties are structural rather than conventional:

**The rule is imported, never re-implemented.** :func:`seam.agent.policy.decide` is called with the
logged inputs and a different deadline. A second implementation would let the replay pass while
testing a rule the harness does not run - the failure would be invisible in every test that used
the replay's own arithmetic as its oracle. The replay additionally self-checks by re-deciding at
the run's own deadline and requiring the logged decision back, byte for byte.

**A single max is never the headline.** Peaks are extreme-value statistics. Everything is reported
as the distribution of per-task peaks with a bootstrap CI over tasks, alongside P95 of per-step
context as the stable companion (pre-registration §4).

**An uninstrumented cache stops the analysis.** ``cached_prompt_tokens == 0`` from a runtime with
no cache counter is not a measurement of "no reuse", and §6 turns entirely on the difference. The
analysis refuses to run unless the caller explicitly acknowledges the bound, and every metric
derived from ``prompt_tokens_new`` is then labelled an upper bound rather than a value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, cast

import yaml

from seam.agent.policy import DEADLINE_DISABLED_S, StepType, ThroughputModel, decide
from seam.agent.steplog import STEPS_FILENAME, read_steps
from seam.analysis.slice_stats import bootstrap_ci
from seam.errors import SeamError
from seam.gitinfo import repo_root
from seam.rawstore import open_run_dir, verify_sealed

__all__ = [
    "EfilterInputs",
    "StepView",
    "analyze",
    "deadline_distribution_units",
    "deadline_grid",
    "envelope_by_task",
    "load_run",
    "main",
    "material_deadline_point",
    "per_task_context_ceiling",
    "prompt_b_offline_analysis",
    "proxy_error_regression",
    "self_check_replay",
    "strict_json_payload",
    "survivors",
    "tail_latency_replay",
    "template_scaffold_analysis",
    "write_figure",
]

#: Relative tolerance when re-deciding a logged step at its own logged deadline. The replay is the
#: same arithmetic on the same floats, so this is tight on purpose: anything looser would hide a
#: throughput or n_out_pred that was reconstructed wrongly from the summary.
_SELF_CHECK_RTOL: Final = 1e-9

#: Nothing survives a deadline below this in the required-decode-rate term. Guards the division
#: when the deadline is entirely consumed by prefill.
_EPS_S: Final = 1e-9


# ==================================================================================================
# Loading
# ==================================================================================================


@dataclass(frozen=True, slots=True)
class StepView:
    """One step, reduced to the fields the replay and the envelope need."""

    task_id: str
    step_idx: int
    #: The step's REALIZED label, used for stratification (AM-025). The harness rewrites it to
    #: ``answer_synthesis`` after a terminal answer is seen, i.e. **after** the routing decision was
    #: taken, so it is not what the router priced. Never feed it to the replay.
    step_type: StepType
    #: The step_type the ROUTER consumed, recovered from the logged routing record. This is the
    #: replay's input; using the realized label instead silently reprices every terminal step.
    router_step_type: StepType
    prompt_tokens_proxy: int
    prompt_tokens_native: int
    context_tokens_total: int
    prompt_tokens_new: int
    completion_tokens: int
    kv_bytes_resident: int
    peak_rss_bytes: int | None
    cache_instrumented: bool
    cache_evicted: bool
    actual_wall_s: float
    logged_t_pred_s: float
    logged_deadline_s: float
    logged_target: str
    #: Logged decomposition of ``t_pred`` (pre-registration §3). Carried so the replay can check
    #: that the two terms sum to the total, and so the caching fork can be answered in the terms
    #: §6 states it in - which term dominates - rather than inferred from context length alone.
    logged_t_pred_prefill_s: float | None = None
    logged_t_pred_decode_s: float | None = None
    deadline_overrun: bool = False

    def prefill_share_of_t_pred(self) -> float:
        """Fraction of the predicted latency the router attributed to prefill.

        This is the quantity the §6 caching fork turns on. Escalation selects on **context** only
        to the extent that prefill dominates ``t_pred``; if the decode term dominates, escalation
        selects on predicted output length and the memory claim weakens regardless of whether the
        cache was reused. Reported per step so the balance can be tracked along the trajectory.
        """
        if self.logged_t_pred_prefill_s is None or self.logged_t_pred_s <= 0:
            return math.nan
        return self.logged_t_pred_prefill_s / self.logged_t_pred_s

    def arithmetic_intensity(self) -> float:
        """Unitless compute-per-weight-load proxy for the step.

        A prefill pass computes ``prompt_tokens_new`` tokens against one pass over the weights;
        each decode step computes one token against another full pass. So

        .. code-block:: text

            AI_proxy = (prompt_tokens_new + completion_tokens) / (passes)
            passes   = (1 if prompt_tokens_new else 0) + completion_tokens

        is tokens computed per weight load. It is 1.0 for a pure-decode step and rises with the
        prefill share. It is a **proxy**: it counts weight traffic only, ignoring KV traffic and
        activations, so it ranks steps rather than predicting FLOP/byte.
        """
        passes = (1 if self.prompt_tokens_new > 0 else 0) + self.completion_tokens
        return (self.prompt_tokens_new + self.completion_tokens) / max(passes, 1)

    def required_decode_rate_tok_s(self, *, deadline_s: float, r_prefill_tok_s: float) -> float:
        """Decode rate this step would need to meet ``deadline_s`` after paying its prefill."""
        budget = deadline_s - self.prompt_tokens_new / r_prefill_tok_s
        return self.completion_tokens / max(budget, _EPS_S)


@dataclass(frozen=True, slots=True)
class EfilterInputs:
    """A sealed E-FILTER run, reduced to what the analysis consumes."""

    run_id: str
    steps: list[StepView]
    throughput: ThroughputModel
    n_out_pred_tokens: dict[str, int]
    kv_bytes_per_token: int
    cache_instrumented: bool
    cache_probe: dict[str, Any]
    summary: dict[str, Any]
    integrity_verified: bool
    #: Per-task paging admissibility (C2e). Missing keys default to True for pre-C2e seals.
    paging_admissible_by_task: dict[str, bool] = field(default_factory=dict)
    paging_admissible_reasons_by_task: dict[str, list[str]] = field(default_factory=dict)
    #: Per-task canary admissibility (C2f). Missing keys default to True for pre-C2f seals.
    canary_admissible_by_task: dict[str, bool] = field(default_factory=dict)
    canary_admissible_reasons_by_task: dict[str, list[str]] = field(default_factory=dict)

    @property
    def task_ids(self) -> list[str]:
        seen: list[str] = []
        for step in self.steps:
            if step.task_id not in seen:
                seen.append(step.task_id)
        return seen

    def task_paging_admissible(self, task_id: str) -> bool:
        """Default True when the sealed run predates C2e tagging (no record => treat as admissible)."""
        return bool(self.paging_admissible_by_task.get(task_id, True))

    def task_canary_admissible(self, task_id: str) -> bool:
        """Default True when the sealed run predates C2f tagging."""
        return bool(self.canary_admissible_by_task.get(task_id, True))

    def task_timing_admissible(self, task_id: str) -> bool:
        """Timing endpoints: canary_admissible ∩ paging_admissible (C2f)."""
        return self.task_paging_admissible(task_id) and self.task_canary_admissible(task_id)

    def paging_admissible_steps(self) -> list[StepView]:
        return [s for s in self.steps if self.task_paging_admissible(s.task_id)]

    def timing_admissible_steps(self) -> list[StepView]:
        return [s for s in self.steps if self.task_timing_admissible(s.task_id)]

    def admissible_fraction(self) -> dict[str, Any]:
        n_tasks = len(self.task_ids)
        n_paging = sum(1 for t in self.task_ids if self.task_paging_admissible(t))
        n_canary = sum(1 for t in self.task_ids if self.task_canary_admissible(t))
        n_timing = sum(1 for t in self.task_ids if self.task_timing_admissible(t))
        n_steps = len(self.steps)
        n_paging_steps = len(self.paging_admissible_steps())
        n_timing_steps = len(self.timing_admissible_steps())
        return {
            "n_tasks": n_tasks,
            "n_paging_admissible_tasks": n_paging,
            "paging_admissible_task_fraction": (n_paging / n_tasks if n_tasks else None),
            "n_canary_admissible_tasks": n_canary,
            "canary_admissible_task_fraction": (n_canary / n_tasks if n_tasks else None),
            "n_timing_admissible_tasks": n_timing,
            "timing_admissible_task_fraction": (n_timing / n_tasks if n_tasks else None),
            "n_steps": n_steps,
            "n_paging_admissible_steps": n_paging_steps,
            "paging_admissible_step_fraction": (n_paging_steps / n_steps if n_steps else None),
            "n_timing_admissible_steps": n_timing_steps,
            "timing_admissible_step_fraction": (n_timing_steps / n_steps if n_steps else None),
            "timing_rule": "canary_admissible AND paging_admissible",
        }


def load_run(
    run_id: str, *, root: Path, accept_uninstrumented_cache: bool = False
) -> EfilterInputs:
    """Load a **sealed** E-FILTER run and refuse anything the analysis cannot interpret.

    Raises:
        SeamError: If the run is unsealed, if its integrity hash does not recompute, if the KV
            constant is missing, or if the cache is uninstrumented and the caller has not
            acknowledged that ``prompt_tokens_new`` is then an upper bound rather than a value.
    """
    run_dir = open_run_dir(run_id, repo_root=root)
    if not run_dir.is_sealed():
        raise SeamError(
            f"run {run_id} is not sealed. Nothing is analyzed from an unsealed run: its files can "
            f"still change, so a number taken from it does not trace to a fixed artifact."
        )
    integrity_ok = verify_sealed(run_dir)
    if not integrity_ok:
        raise SeamError(
            f"run {run_id} failed its raw integrity check: the recomputed tree hash does not match "
            f"the one recorded at seal time. The run has been modified since sealing."
        )

    summary = json.loads((run_dir.path / "summary.json").read_text(encoding="utf-8"))
    records = read_steps(run_dir.path / STEPS_FILENAME)
    if not records:
        raise SeamError(f"run {run_id} has no step records")

    cache_instrumented = bool(summary.get("cache_instrumented", False))
    if not cache_instrumented and not accept_uninstrumented_cache:
        probe = summary.get("cache_probe", {})
        raise SeamError(
            f"run {run_id} reports cache_instrumented=false: the local runtime exposes no "
            f"cache-reuse counter, so cached_prompt_tokens=0 is a STRUCTURAL zero and not a "
            f"measurement of 'no reuse'. prompt_tokens_new is therefore an UPPER BOUND on the "
            f"tokens actually prefilled, and the caching fork (pre-registration §6) cannot be "
            f"settled from counters. The run's independent TTFT probe says: "
            f"{probe.get('verdict')!r} (ttft ratio {probe.get('ttft_ratio_second_over_first')!r}). "
            f"Re-run with accept_uninstrumented_cache=True to proceed with every "
            f"prompt_tokens_new-derived quantity labelled as a bound."
        )

    throughput_block = summary["throughput"]
    throughput = ThroughputModel(
        target=str(throughput_block["target"]),
        r_prefill_tok_s=float(throughput_block["r_prefill_tok_s"]),
        r_decode_tok_s=float(throughput_block["r_decode_tok_s"]),
        measured_by_run_id=str(throughput_block["measured_by_run_id"]),
    )
    n_out_pred = {str(k): int(v) for k, v in summary["policy"]["n_out_pred_tokens_used"].items()}
    kv_bytes_per_token = int(summary["kv_geometry"]["kv_bytes_per_token"])

    steps: list[StepView] = []
    for record in records:
        if record.get("kv_bytes_per_token") is None:
            raise SeamError(
                f"run {run_id}: step {record.get('program_id')}/{record.get('step_idx')} has no "
                f"kv_bytes_per_token. The KV envelope is analytic and cannot be reconstructed "
                f"after the fact; a zero here would be read as a measured value."
            )
        if int(record["kv_bytes_per_token"]) != kv_bytes_per_token:
            raise SeamError(
                f"run {run_id}: step KV constant {record['kv_bytes_per_token']} disagrees with the "
                f"manifest constant {kv_bytes_per_token}"
            )
        steps.append(
            StepView(
                # program_id is "<label>/<task_id>"; the task is the bootstrap's resampling unit.
                task_id=str(record["program_id"]).split("/")[-1],
                step_idx=int(record["step_idx"]),
                step_type=str(record["step_type"]),  # type: ignore[arg-type]
                router_step_type=str(  # type: ignore[arg-type]
                    record["routing"].get("step_type", record["step_type"])
                ),
                prompt_tokens_proxy=int(record["prompt_tokens_proxy"]),
                prompt_tokens_native=int(record["prompt_tokens"]),
                context_tokens_total=int(record["context_tokens_total"]),
                prompt_tokens_new=int(record["prompt_tokens_new"]),
                completion_tokens=int(record["completion_tokens"]),
                kv_bytes_resident=int(record["kv_bytes_resident"]),
                peak_rss_bytes=(
                    int(record["peak_rss_bytes"]) if record.get("peak_rss_bytes") else None
                ),
                cache_instrumented=bool(record["cache_instrumented"]),
                cache_evicted=bool(record["cache_evicted"]),
                actual_wall_s=float(record["actual_wall_s"]),
                logged_t_pred_s=float(record["routing"]["t_pred_s"]),
                logged_deadline_s=float(record["routing"]["deadline_s"]),
                logged_target=str(record["assigned_target"]),
                logged_t_pred_prefill_s=(
                    float(record["t_pred_prefill_s"])
                    if record.get("t_pred_prefill_s") is not None
                    else None
                ),
                logged_t_pred_decode_s=(
                    float(record["t_pred_decode_s"])
                    if record.get("t_pred_decode_s") is not None
                    else None
                ),
                deadline_overrun=bool(record.get("deadline_overrun", False)),
            )
        )

    paging_by_task: dict[str, bool] = {}
    paging_reasons_by_task: dict[str, list[str]] = {}
    canary_by_task: dict[str, bool] = {}
    canary_reasons_by_task: dict[str, list[str]] = {}
    for row in summary.get("per_task") or []:
        tid = str(row.get("task_id") or "")
        if not tid:
            continue
        if "paging_admissible" in row:
            paging_by_task[tid] = bool(row["paging_admissible"])
            paging_reasons_by_task[tid] = [
                str(r) for r in (row.get("paging_admissible_reasons") or [])
            ]
        if "canary_admissible" in row:
            canary_by_task[tid] = bool(row["canary_admissible"])
            canary_reasons_by_task[tid] = [
                str(r) for r in (row.get("canary_admissible_reasons") or [])
            ]

    return EfilterInputs(
        run_id=run_id,
        steps=steps,
        throughput=throughput,
        n_out_pred_tokens=n_out_pred,
        kv_bytes_per_token=kv_bytes_per_token,
        cache_instrumented=cache_instrumented,
        cache_probe=dict(summary.get("cache_probe", {})),
        summary=summary,
        integrity_verified=integrity_ok,
        paging_admissible_by_task=paging_by_task,
        paging_admissible_reasons_by_task=paging_reasons_by_task,
        canary_admissible_by_task=canary_by_task,
        canary_admissible_reasons_by_task=canary_reasons_by_task,
    )


def _with_admissible_basis(value: Any, *, fraction: dict[str, Any], basis: str) -> dict[str, Any]:
    """Attach paging/canary/timing-admissible fractions next to a reported number (C2e/C2f)."""
    return {
        "value": value,
        "basis": basis,
        "paging_admissible_task_fraction": fraction.get("paging_admissible_task_fraction"),
        "paging_admissible_step_fraction": fraction.get("paging_admissible_step_fraction"),
        "canary_admissible_task_fraction": fraction.get("canary_admissible_task_fraction"),
        "timing_admissible_task_fraction": fraction.get("timing_admissible_task_fraction"),
        "timing_admissible_step_fraction": fraction.get("timing_admissible_step_fraction"),
        "n_paging_admissible_tasks": fraction.get("n_paging_admissible_tasks"),
        "n_canary_admissible_tasks": fraction.get("n_canary_admissible_tasks"),
        "n_timing_admissible_tasks": fraction.get("n_timing_admissible_tasks"),
        "n_tasks": fraction.get("n_tasks"),
        "n_paging_admissible_steps": fraction.get("n_paging_admissible_steps"),
        "n_timing_admissible_steps": fraction.get("n_timing_admissible_steps"),
        "n_steps": fraction.get("n_steps"),
        "timing_rule": fraction.get("timing_rule"),
    }


# ==================================================================================================
# Replay
# ==================================================================================================


def self_check_replay(inputs: EfilterInputs) -> dict[str, Any]:
    """Re-decide every step at its OWN logged deadline and require the logged decision back.

    This is what makes the replay's other deadlines believable. If the throughput or ``n_out_pred``
    reconstructed from the summary differs from what the harness held, ``t_pred`` moves, and every
    counterfactual built on it moves with it - silently, because the replay would still be
    internally consistent.
    """
    mismatches: list[dict[str, Any]] = []
    decomposition_mismatches: list[dict[str, Any]] = []
    for step in inputs.steps:
        decision = decide(
            throughput=inputs.throughput,
            deadline_s=step.logged_deadline_s,
            prompt_tokens=step.prompt_tokens_proxy,
            step_type=step.router_step_type,
            n_out_pred_tokens=inputs.n_out_pred_tokens,
        )
        t_pred_ok = math.isclose(
            decision.t_pred_s, step.logged_t_pred_s, rel_tol=_SELF_CHECK_RTOL, abs_tol=0.0
        )
        if not t_pred_ok or decision.assigned_target != step.logged_target:
            mismatches.append(
                {
                    "task_id": step.task_id,
                    "step_idx": step.step_idx,
                    "logged_t_pred_s": step.logged_t_pred_s,
                    "replayed_t_pred_s": decision.t_pred_s,
                    "logged_target": step.logged_target,
                    "replayed_target": decision.assigned_target,
                }
            )
        # The decomposition is logged separately from the total, so it can disagree with it. The
        # caching fork is read off the two terms, so a decomposition that does not sum to the
        # total would answer §6 from arithmetic the router never did.
        if step.logged_t_pred_prefill_s is not None and step.logged_t_pred_decode_s is not None:
            parts = step.logged_t_pred_prefill_s + step.logged_t_pred_decode_s
            if not math.isclose(parts, step.logged_t_pred_s, rel_tol=_SELF_CHECK_RTOL):
                decomposition_mismatches.append(
                    {
                        "task_id": step.task_id,
                        "step_idx": step.step_idx,
                        "prefill_plus_decode_s": parts,
                        "logged_t_pred_s": step.logged_t_pred_s,
                    }
                )

    passed = not mismatches and not decomposition_mismatches
    return {
        "passed": passed,
        "n_steps_checked": len(inputs.steps),
        "rel_tol": _SELF_CHECK_RTOL,
        "mismatches": mismatches[:20],
        "n_mismatches": len(mismatches),
        "decomposition_mismatches": decomposition_mismatches[:20],
        "n_decomposition_mismatches": len(decomposition_mismatches),
        "n_steps_with_decomposition": sum(
            1 for s in inputs.steps if s.logged_t_pred_prefill_s is not None
        ),
        "interpretation": (
            "replaying each step at its own logged deadline reproduces the logged decision and "
            "t_pred, and the logged prefill/decode terms sum to the logged total, so the "
            "counterfactual deadlines are evaluated by the same rule the run used"
            if passed
            else "REPLAY SELF-CHECK FAILED: the replay does not reproduce the logged decisions; "
            "every counterfactual below is invalid. STOP."
        ),
    }


def deadline_grid(
    t_preds: Sequence[float],
    *,
    n: int,
    quantile_lo: float,
    quantile_hi: float,
    pad_factor: float,
) -> list[float]:
    """Log-spaced grid spanning the observed ``t_pred`` distribution, padded at both ends.

    Endpoints come from the data rather than from a guessed range, so the curve covers the
    transition region where the filter actually changes what survives. The padding puts a
    fully-unfiltered and a fully-filtered point on the curve, which is what makes the plateaus at
    either end visible instead of implied.

    C2 requires ``n >= 8`` so the curve resolves the transition rather than collapsing to a
    handful of guessed points (docs/CURSOR_PROMPT_C2_efilter.md C2.2).
    """
    values = sorted(float(t) for t in t_preds if t > 0)
    if not values:
        raise SeamError("no positive t_pred values; the deadline grid cannot be built")
    lo = _quantile(values, quantile_lo) / pad_factor
    hi = _quantile(values, quantile_hi) * pad_factor
    if n < 8 or lo <= 0 or hi <= lo:
        raise SeamError(
            f"degenerate or under-resolved deadline grid: n={n} (C2 requires >=8), lo={lo}, hi={hi}"
        )
    step = (math.log(hi) - math.log(lo)) / (n - 1)
    return [math.exp(math.log(lo) + i * step) for i in range(n)]


#: Stage-1 baseline for P6 (peak context 1826; mean per-task ceiling 1.456).
_STAGE1_P6_BASELINE: Final[dict[str, Any]] = {
    "over_provisioning_ratio": 1.243,
    "ci": [1.115, 1.409],
    "peak_context_tokens": 1826,
    "context_ceiling_mean": 1.456,
    "run_id": "1a0166b9-cbaf-43f4-8d76-bcd7c01841e0",
}


def per_task_context_ceiling(steps: Sequence[StepView]) -> dict[str, Any]:
    """Compute ``C_max/C_min`` per task and aggregate medians/means for interpretation.

    Ratio definition (C2b): ``max(context_tokens_total)/min(context_tokens_total)`` within each
    task's steps - min/max of the trajectory, not first/last.
    """
    by_task: dict[str, list[int]] = {}
    for step in steps:
        by_task.setdefault(step.task_id, []).append(int(step.context_tokens_total))
    per_task: dict[str, float] = {}
    for task_id, values in by_task.items():
        c_min, c_max = min(values), max(values)
        if c_min > 0:
            per_task[task_id] = float(c_max) / float(c_min)
    ratios = list(per_task.values())
    return {
        "definition": (
            "C_max/C_min = max/min of context_tokens_total per task (min/max, not first/last)"
        ),
        "per_task": per_task,
        "median": float(statistics.median(ratios)) if ratios else None,
        "mean": float(statistics.fmean(ratios)) if ratios else None,
        "n_tasks": len(ratios),
        "max_context_tokens": max((s.context_tokens_total for s in steps), default=0),
        "max_context_tokens_is_gate": False,
    }


def material_deadline_point(
    curve: Sequence[dict[str, Any]], *, materiality_ratio: float
) -> dict[str, Any] | None:
    """Loosest grid point whose KV over-provisioning point estimate is >= materiality."""
    finite = _finite_curve_points(curve)
    material = [
        point
        for point in finite
        if float(point["over_provisioning"]["peak_kv_bytes_resident"].get("point", math.nan))
        >= materiality_ratio
    ]
    if not material:
        return None
    return max(material, key=lambda point: point["deadline_s"])


def deadline_distribution_units(deadline_s: float, t_preds: Sequence[float]) -> dict[str, Any]:
    """Locate a deadline in the observed ``t_pred`` distribution (primary reporting units)."""
    values = sorted(float(t) for t in t_preds if t > 0)
    if not values:
        return {
            "deadline_s": deadline_s,
            "empirical_cdf": None,
            "fraction_of_range": None,
            "seconds_caveat": "R unverified pending A4",
        }
    lo, hi = values[0], values[-1]
    n_le = sum(1 for t in values if t <= deadline_s)
    return {
        "deadline_s": deadline_s,
        "empirical_cdf": n_le / len(values),
        "fraction_of_observed_t_pred_range": ((deadline_s - lo) / (hi - lo) if hi > lo else 0.0),
        "t_pred_min_s": lo,
        "t_pred_max_s": hi,
        "seconds_caveat": "R unverified pending A4 - seconds are secondary; do not claim absolute wall-clock targets",
    }


def survivors(
    inputs: EfilterInputs, *, deadline_s: float, use_native_tokens: bool = False
) -> list[StepView]:
    """Steps that would NOT have escalated at ``deadline_s``, by the real routing rule.

    Args:
        use_native_tokens: Feed the router the tokenizer's count instead of the ``chars // 4``
            proxy it actually consumed. Used to bound the proxy's effect on the filter boundary,
            never as the headline: the counterfactual has to be built on what the router saw.
    """
    kept: list[StepView] = []
    for step in inputs.steps:
        prompt_tokens = step.prompt_tokens_native if use_native_tokens else step.prompt_tokens_proxy
        decision = decide(
            throughput=inputs.throughput,
            deadline_s=deadline_s,
            prompt_tokens=prompt_tokens,
            step_type=step.router_step_type,
            n_out_pred_tokens=inputs.n_out_pred_tokens,
        )
        if decision.assigned_target == "local":
            kept.append(step)
    return kept


# ==================================================================================================
# Envelope
# ==================================================================================================

#: Envelope dimensions, pre-registration §4. Each is a per-task peak or per-task P95, aggregated
#: across tasks - never a single global max.
_ENVELOPE_METRICS: Final[tuple[str, ...]] = (
    "peak_kv_bytes_resident",
    "peak_context_tokens",
    "max_prompt_tokens_new",
    "p95_required_decode_rate_tok_s",
    "peak_rss_bytes",
    "mean_arithmetic_intensity",
)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile on already-sorted values."""
    if not sorted_values:
        return math.nan
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    rank = max(1, math.ceil(q * len(sorted_values)))
    return float(sorted_values[rank - 1])


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Mean over the defined values, or None if none are defined."""
    usable = [float(v) for v in values if v is not None and not math.isnan(v)]
    return statistics.fmean(usable) if usable else None


def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Least-squares slope, or None when it is not identifiable."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def p95(values: Sequence[float]) -> float:
    return _quantile(sorted(float(v) for v in values), 0.95)


def envelope_by_task(
    steps: Sequence[StepView], *, deadline_s: float, r_prefill_tok_s: float
) -> dict[str, dict[str, float]]:
    """Per-task envelope over the supplied (surviving) steps.

    Returned as ``{task_id: {metric: value}}``. Tasks with no surviving steps are absent rather
    than zero: an envelope over an empty set is undefined, and a zero there would silently drag
    every mean down and inflate the over-provisioning ratio.
    """
    by_task: dict[str, list[StepView]] = {}
    for step in steps:
        by_task.setdefault(step.task_id, []).append(step)

    out: dict[str, dict[str, float]] = {}
    for task_id, task_steps in by_task.items():
        rss = [s.peak_rss_bytes for s in task_steps if s.peak_rss_bytes]
        out[task_id] = {
            "peak_kv_bytes_resident": float(max(s.kv_bytes_resident for s in task_steps)),
            "peak_context_tokens": float(max(s.context_tokens_total for s in task_steps)),
            "max_prompt_tokens_new": float(max(s.prompt_tokens_new for s in task_steps)),
            "p95_required_decode_rate_tok_s": p95(
                [
                    s.required_decode_rate_tok_s(
                        deadline_s=deadline_s, r_prefill_tok_s=r_prefill_tok_s
                    )
                    for s in task_steps
                ]
            ),
            "peak_rss_bytes": float(max(rss)) if rss else math.nan,
            "mean_arithmetic_intensity": statistics.fmean(
                s.arithmetic_intensity() for s in task_steps
            ),
            "n_steps": float(len(task_steps)),
        }
    return out


def _paired_ratio_bootstrap(
    unfiltered: dict[str, float],
    filtered: dict[str, float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap ``mean(unfiltered)/mean(filtered)`` by resampling **tasks**, paired.

    Paired because the two envelopes are computed over the same tasks: resampling them
    independently would add between-task variance that the ratio does not contain, widening the
    interval for a reason that is an artifact of the procedure.
    """
    common = sorted(set(unfiltered) & set(filtered))
    if not common:
        return {
            "point": math.nan,
            "lo": math.nan,
            "hi": math.nan,
            "n_tasks": 0,
            "note": "no task retained a surviving step at this deadline",
        }
    num = [unfiltered[t] for t in common]
    den = [filtered[t] for t in common]
    if any(math.isnan(v) for v in num + den):
        pairs = [
            (n, d) for n, d in zip(num, den, strict=True) if not (math.isnan(n) or math.isnan(d))
        ]
        num = [n for n, _ in pairs]
        den = [d for _, d in pairs]
    if not num or statistics.fmean(den) == 0:
        return {
            "point": math.nan,
            "lo": math.nan,
            "hi": math.nan,
            "n_tasks": len(num),
            "note": "denominator envelope is zero or undefined",
        }

    point = statistics.fmean(num) / statistics.fmean(den)
    rng = random.Random(seed)
    n = len(num)
    draws: list[float] = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        d_mean = statistics.fmean(den[i] for i in idx)
        if d_mean == 0:
            continue
        draws.append(statistics.fmean(num[i] for i in idx) / d_mean)
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = draws[max(0, int(alpha * len(draws)) - 1)] if draws else math.nan
    hi = draws[min(len(draws) - 1, int((1.0 - alpha) * len(draws)))] if draws else math.nan
    return {"point": point, "lo": lo, "hi": hi, "n_tasks": n, "resamples": len(draws)}


# ==================================================================================================
# The proxy the router actually consumed
# ==================================================================================================


def proxy_error_regression(steps: Sequence[StepView]) -> dict[str, Any]:
    """Regress the router's ``chars // 4`` proxy against the tokenizer's count.

    The counterfactual escalation set is built on the proxy, so the proxy's error is part of the
    result. Reported as slope/intercept/residual spread plus the relative bias that decides whether
    the filter boundary is a line or an interval.
    """
    x = [float(s.prompt_tokens_proxy) for s in steps]
    y = [float(s.prompt_tokens_native) for s in steps]
    n = len(x)
    if n < 3:
        return {"n": n, "note": "too few steps to regress"}

    mean_x, mean_y = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
    slope = sxy / sxx if sxx else math.nan
    intercept = mean_y - slope * mean_x if sxx else math.nan
    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y, strict=True)]
    ss_res = sum(r * r for r in residuals)
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    relative = [(xi - yi) / yi for xi, yi in zip(x, y, strict=True) if yi > 0]

    return {
        "n": n,
        "model": "native_prompt_tokens ~ a + b * prompt_tokens_proxy",
        "slope": slope,
        "intercept": intercept,
        "r_squared": (1 - ss_res / ss_tot) if ss_tot else math.nan,
        "residual_sd_tokens": statistics.stdev(residuals) if n > 2 else math.nan,
        "residual_max_abs_tokens": max(abs(r) for r in residuals),
        "mean_relative_bias_proxy_vs_native": statistics.fmean(relative) if relative else math.nan,
        "median_relative_bias_proxy_vs_native": (
            statistics.median(relative) if relative else math.nan
        ),
        "relative_bias_definition": "(proxy - native) / native, per step, then averaged",
        "proxy_mean_tokens": mean_x,
        "native_mean_tokens": mean_y,
    }


# ==================================================================================================
# Stratification and the pre-registered predictions
# ==================================================================================================


def _stratified(inputs: EfilterInputs, *, deadline_s: float) -> dict[str, Any]:
    """Escalation selectivity by ``step_idx`` and by ``step_type`` (AM-025).

    ``step_type`` is hardcoded in the harness, so its stratum is reported for completeness and is
    NOT a taxonomy result; ``step_idx`` carries the pre-registered P4 test.
    """
    kept = {(s.task_id, s.step_idx) for s in survivors(inputs, deadline_s=deadline_s)}

    def _block(steps: Sequence[StepView]) -> dict[str, Any]:
        if not steps:
            return {"n": 0}
        escalated = [s for s in steps if (s.task_id, s.step_idx) not in kept]
        surviving = [s for s in steps if (s.task_id, s.step_idx) in kept]
        return {
            "n": len(steps),
            "n_escalated": len(escalated),
            "escalation_rate": len(escalated) / len(steps),
            "mean_context_tokens_escalated": (
                statistics.fmean(s.context_tokens_total for s in escalated) if escalated else None
            ),
            "mean_context_tokens_surviving": (
                statistics.fmean(s.context_tokens_total for s in surviving) if surviving else None
            ),
            "mean_completion_tokens_escalated": (
                statistics.fmean(s.completion_tokens for s in escalated) if escalated else None
            ),
            "mean_completion_tokens_surviving": (
                statistics.fmean(s.completion_tokens for s in surviving) if surviving else None
            ),
            "mean_arithmetic_intensity_surviving": (
                statistics.fmean(s.arithmetic_intensity() for s in surviving) if surviving else None
            ),
            # Deadline-independent: a property of how the router priced this stratum. P4 is a
            # claim about the BASIS of selection, and this is that basis in one number - the
            # share of t_pred that context (prefill) contributes rather than output length.
            "mean_prefill_share_of_t_pred": _mean_or_none(
                [s.prefill_share_of_t_pred() for s in steps]
            ),
            "mean_t_pred_s": statistics.fmean(s.logged_t_pred_s for s in steps),
            "mean_context_tokens": statistics.fmean(s.context_tokens_total for s in steps),
        }

    by_idx: dict[int, list[StepView]] = {}
    by_type: dict[str, list[StepView]] = {}
    for step in inputs.steps:
        by_idx.setdefault(step.step_idx, []).append(step)
        by_type.setdefault(step.step_type, []).append(step)

    return {
        "deadline_s": deadline_s,
        "by_step_idx": {str(k): _block(v) for k, v in sorted(by_idx.items())},
        "by_step_type": {k: _block(v) for k, v in sorted(by_type.items())},
        "step_type_limitation": (
            "step_type is hardcoded to tool_call_synthesis except on the terminal step, so its "
            "strata are not a taxonomy finding. The pre-registered P4 test uses step_idx."
        ),
    }


#: Dimensions P1 is evaluated on. ``mean_arithmetic_intensity`` is deliberately excluded: it is a
#: *balance* statistic and it is P3's own subject, so letting it satisfy P1 would let a shift in
#: workload character be reported as a reduction in the resource envelope. Its ratio is still
#: reported in full, and P1 additionally records what its verdict would have been had that
#: dimension been admitted, so the choice is visible rather than buried.
_P1_RESOURCE_DIMENSIONS: Final[tuple[str, ...]] = (
    "peak_kv_bytes_resident",
    "peak_context_tokens",
    "max_prompt_tokens_new",
    "p95_required_decode_rate_tok_s",
    "peak_rss_bytes",
)


def _best_ratio(ratios: dict[str, Any], metrics: Sequence[str]) -> tuple[str | None, float]:
    best_dimension: str | None = None
    best = 0.0
    for metric in metrics:
        point = ratios.get(metric, {}).get("point")
        if point is not None and not math.isnan(point) and point > best:
            best, best_dimension = point, metric
    return best_dimension, best


def _ratio_verdict(ratio: float, *, materiality_ratio: float) -> str:
    """Three-way verdict against the frozen materiality threshold (pre-registration §7)."""
    if math.isnan(ratio) or ratio == 0.0:
        return "UNDETERMINED"
    if ratio >= materiality_ratio:
        return "SUPPORTED"
    if ratio <= 1.0:
        return "FALSIFIED"
    return "NULL"


def _predictions(
    *,
    curve: list[dict[str, Any]],
    headline: dict[str, Any],
    feasibility: dict[str, Any],
    materiality_ratio: float,
    stratified: dict[str, Any],
    peak_context_tokens: int,
    context_ceiling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate P1-P6. P1-P3 at the material grid deadline; 8 s wall-clock target withdrawn (AM-032)."""
    del feasibility  # retained in analyze() report; not a P1-P3 success/fail input under C2
    ratios_at_headline = headline.get("over_provisioning", {})
    best_dimension, best_ratio = _best_ratio(ratios_at_headline, _P1_RESOURCE_DIMENSIONS)
    with_ai_dimension, with_ai_ratio = _best_ratio(ratios_at_headline, _ENVELOPE_METRICS)

    kv = ratios_at_headline.get("peak_kv_bytes_resident", {})
    kv_point = kv.get("point")
    kv_lo, kv_hi = kv.get("lo"), kv.get("hi")
    kv_value = float(kv_point) if kv_point is not None else math.nan
    hi_value = float(kv_hi) if kv_hi is not None else math.nan
    if math.isnan(kv_value):
        p2_verdict = "UNDETERMINED: no paired KV envelope at the material deadline"
    elif kv_value >= 2.0:
        p2_verdict = "SUPPORTED"
    elif kv_value < materiality_ratio and not math.isnan(hi_value) and hi_value < 2.0:
        p2_verdict = "FALSIFIED: below materiality with the CI excluding 2x"
    elif kv_value < materiality_ratio:
        p2_verdict = "NULL: below the 1.2x materiality threshold, CI does not exclude 2x"
    else:
        p2_verdict = "MATERIAL_BUT_BELOW_2X"

    # P3 - arithmetic intensity. Direction matters: the prediction is a DROP, and a rise falsifies
    # it. Evaluated on the same paired per-task ratio as the envelope dimensions so it carries a CI
    # rather than being a bare mean comparison.
    ai = ratios_at_headline.get("mean_arithmetic_intensity", {})
    ai_point = ai.get("point")
    ai_ratio = float(ai_point) if ai_point is not None else math.nan
    p3_verdict = _ratio_verdict(ai_ratio, materiality_ratio=materiality_ratio)
    if p3_verdict == "SUPPORTED":
        p3_detail = (
            "surviving steps are materially less prefill-heavy: the local mix shifts toward decode"
        )
    elif p3_verdict == "FALSIFIED":
        p3_detail = (
            "arithmetic intensity does not fall: filtering leaves the mix as prefill-heavy or more"
        )
    elif p3_verdict == "NULL":
        p3_detail = "a drop below the materiality threshold, reported as null"
    else:
        p3_detail = "no paired arithmetic-intensity ratio at the material deadline"

    # P4 - the BASIS of selection, not its magnitude. Two independent series, because either alone
    # is ambiguous: the escalation rate can rise with step_idx while selection stays entirely
    # output-length driven, and the prefill share can rise without any step being filtered.
    strata = [
        (int(idx), block) for idx, block in stratified["by_step_idx"].items() if block.get("n")
    ]
    strata.sort()
    idxs = [float(i) for i, _ in strata]
    escalation_slope = _ols_slope(idxs, [float(block["escalation_rate"]) for _, block in strata])
    shares = [(i, b.get("mean_prefill_share_of_t_pred")) for i, b in strata]
    defined_shares = [(i, s) for i, s in shares if s is not None]
    share_slope = (
        _ols_slope([float(i) for i, _ in defined_shares], [float(s) for _, s in defined_shares])
        if len(defined_shares) >= 3
        else None
    )
    share_first = defined_shares[0][1] if defined_shares else None
    share_last = defined_shares[-1][1] if defined_shares else None
    if share_slope is None:
        p4_verdict = (
            "UNDETERMINED: the t_pred decomposition is absent, so the basis of selection is not "
            "observable"
        )
    elif share_slope > 0 and escalation_slope is not None and escalation_slope > 0:
        p4_verdict = "SUPPORTED"
    elif share_slope > 0:
        p4_verdict = (
            "PARTIAL: the prefill share of t_pred rises with step_idx, but the escalation rate "
            "does not, so the shift in basis does not translate into a shift in what is filtered"
        )
    else:
        p4_verdict = "FALSIFIED: selectivity does not become more context-driven as context grows"

    # P6 - over-provisioning rises with the trajectory context ceiling vs Stage-1.
    # C2b: falsified if OP lands inside Stage-1 CI despite a much higher ceiling; peak tokens
    # are never a P6 gate (the 20k absolute peak gate is withdrawn).
    baseline = _STAGE1_P6_BASELINE
    ci_lo, ci_hi = float(baseline["ci"][0]), float(baseline["ci"][1])
    ceiling = context_ceiling or {}
    ceiling_median = ceiling.get("median")
    ceiling_mean = ceiling.get("mean")
    stage1_ceiling = float(baseline["context_ceiling_mean"])
    ceiling_for_gate = (
        float(ceiling_median)
        if ceiling_median is not None
        else (float(ceiling_mean) if ceiling_mean is not None else math.nan)
    )
    # "Much higher" than Stage-1's 1.456 mean: median ratio ≥ 3.0 (C2b pilot gate) or ≥ 2x Stage-1.
    ceiling_much_higher = not math.isnan(ceiling_for_gate) and (
        ceiling_for_gate >= 3.0 or ceiling_for_gate >= 2.0 * stage1_ceiling
    )
    if math.isnan(ceiling_for_gate):
        p6_verdict = "UNDETERMINED: context ceiling C_max/C_min not computable on this run"
    elif not ceiling_much_higher:
        p6_verdict = (
            f"UNDETERMINED: context ceiling {ceiling_for_gate:.3f} is not much higher than "
            f"Stage-1's {stage1_ceiling} (need median ≥ 3.0 or ≥ 2x Stage-1); peak context "
            f"{peak_context_tokens} is reported only"
        )
    elif math.isnan(kv_value):
        p6_verdict = "UNDETERMINED: no KV over-provisioning ratio at the material deadline"
    elif ci_lo <= kv_value <= ci_hi:
        p6_verdict = (
            "FALSIFIED: over-provisioning lands inside Stage-1 CI "
            f"[{ci_lo}, {ci_hi}] despite much higher context ceiling {ceiling_for_gate:.3f} "
            f"(Stage-1 OP {baseline['over_provisioning_ratio']}x vs ceiling "
            f"{stage1_ceiling}, run_id={baseline['run_id']})"
        )
    elif kv_value > ci_hi:
        if kv_value >= baseline["over_provisioning_ratio"] * materiality_ratio:
            p6_verdict = "SUPPORTED"
        else:
            p6_verdict = (
                "MATERIAL_DIRECTION: above Stage-1 CI but below baseline*materiality; report as "
                "directionally higher without claiming the full material rise"
            )
    else:
        p6_verdict = (
            "FALSIFIED: over-provisioning is below the Stage-1 CI - opposite of the predicted rise"
        )

    return {
        "P1": {
            "statement": (
                "at the material deadline on the derived grid, filtered envelope is lower than "
                "unfiltered on >=1 primary dimension"
            ),
            "dimensions_scanned": list(_P1_RESOURCE_DIMENSIONS),
            "dimensions_excluded": ["mean_arithmetic_intensity"],
            "exclusion_reason": (
                "arithmetic intensity is a balance statistic and is P3's own subject; admitting it "
                "would let a change in workload character satisfy a claim about the resource "
                "envelope. Its ratio is reported in full and the alternative verdict is stated."
            ),
            "evaluation_deadline": "material grid point (AM-032; not a fixed wall-clock target)",
            "headline_deadline_s": headline.get("deadline_s"),
            "headline_deadline_distribution_units": headline.get("distribution_units"),
            "best_dimension": best_dimension,
            "best_ratio": best_ratio,
            "materiality_ratio": materiality_ratio,
            "verdict": _ratio_verdict(best_ratio, materiality_ratio=materiality_ratio),
            "verdict_if_arithmetic_intensity_admitted": {
                "best_dimension": with_ai_dimension,
                "best_ratio": with_ai_ratio,
                "verdict": _ratio_verdict(with_ai_ratio, materiality_ratio=materiality_ratio),
            },
        },
        "P2": {
            "statement": (
                "at the material deadline on the derived grid, peak local KV falls by >= 2x"
            ),
            "headline_deadline_s": headline.get("deadline_s"),
            "headline_definition": headline.get("definition"),
            "headline_deadline_distribution_units": headline.get("distribution_units"),
            "kv_ratio": kv_point,
            "ci": [kv_lo, kv_hi],
            "threshold": 2.0,
            "verdict": p2_verdict,
            "withdrawn_wall_clock_target": {
                "p95_step_target_s": None,
                "status": "withdrawn_AM-032",
                "reason": (
                    "pre-registered 8 s sat below the achievable floor and made P1-P3 vacuous; "
                    "PRE-DATA w.r.t. C2"
                ),
            },
            "absolute_wall_clock_claim": False,
            "seconds_caveat": "R unverified pending A4",
        },
        "P3": {
            "statement": (
                "at the material deadline on the derived grid, surviving steps have lower mean "
                "arithmetic intensity"
            ),
            "arithmetic_intensity_ratio_unfiltered_over_filtered": ai_point,
            "ci": [ai.get("lo"), ai.get("hi")],
            "mean_ai_surviving_by_step_idx": [
                block.get("mean_arithmetic_intensity_surviving") for _, block in strata
            ],
            "verdict": p3_verdict,
            "detail": p3_detail,
        },
        "P4": {
            "statement": (
                "filter selectivity shifts from output-length driven to context driven as context "
                "grows"
            ),
            "operationalization": (
                "two series across step_idx: the share of t_pred contributed by prefill (the basis "
                "of selection) and the counterfactual escalation rate (its consequence). The "
                "router's decode term is a constant per step_type, so a shift in basis can only "
                "appear as a rising prefill share."
            ),
            "prefill_share_of_t_pred_by_step_idx": [
                {"step_idx": i, "mean_prefill_share": s} for i, s in shares
            ],
            "prefill_share_slope_per_step": share_slope,
            "prefill_share_first_stratum": share_first,
            "prefill_share_last_stratum": share_last,
            "escalation_rate_vs_step_idx_slope": escalation_slope,
            "verdict": p4_verdict,
        },
        "P5": {
            "statement": "Stage 1 overestimates the Stage 2 reduction",
            "verdict": "NOT_EVALUABLE: requires Stage 2, which is out of scope for this dispatch",
        },
        "P6": {
            "statement": (
                "over-provisioning rises with the trajectory context ceiling (C_max/C_min) "
                "relative to Stage-1 (OP 1.243, ceiling 1.456, run_id=1a0166b9…)"
            ),
            "stage1_baseline": dict(baseline),
            "c2_peak_context_tokens": peak_context_tokens,
            "c2_peak_context_tokens_is_gate": False,
            "c2_context_ceiling": {
                "median": ceiling_median,
                "mean": ceiling_mean,
                "definition": ceiling.get("definition"),
            },
            "c2_kv_over_provisioning_at_material_deadline": kv_point,
            "c2_ci": [kv_lo, kv_hi],
            "falsification_rule": (
                "FALSIFIED if OP lands inside Stage-1 CI "
                f"[{ci_lo}, {ci_hi}] despite a much higher ceiling (median ≥ 3.0 or ≥ 2x "
                f"Stage-1's {stage1_ceiling}); SUPPORTED if above CI and "
                f">= baseline * {materiality_ratio}. Absolute peak context is never a gate."
            ),
            "verdict": p6_verdict,
        },
        "curve_points": len(curve),
        "verdict_vocabulary": {
            "SUPPORTED": "meets the pre-registered threshold",
            "NULL": "an effect in the predicted direction but below the 1.2x materiality threshold",
            "FALSIFIED": "no effect, or an effect in the opposite direction",
            "UNDETERMINED": "the statistic is not defined on this run",
        },
    }


# ==================================================================================================
# Top level
# ==================================================================================================


def analyze(inputs: EfilterInputs, *, replay_cfg: dict[str, Any]) -> dict[str, Any]:
    """Produce the deadline curve, the over-provisioning ratios, and the guard reports."""
    check = self_check_replay(inputs)
    if not check["passed"]:
        first = (check["mismatches"] or check["decomposition_mismatches"])[0]
        raise SeamError(
            f"replay self-check failed on {check['n_mismatches']} decision(s) and "
            f"{check['n_decomposition_mismatches']} decomposition(s): the replayed rule does not "
            f"reproduce the logged decisions, so no counterfactual from it is valid. "
            f"First mismatch: {first}"
        )

    resamples = int(replay_cfg["bootstrap_resamples"])
    seed = int(replay_cfg["bootstrap_seed"])
    materiality = float(replay_cfg["materiality_ratio"])
    raw_target = replay_cfg.get("p95_step_target_s")
    target_withdrawn = raw_target is None or str(
        replay_cfg.get("p95_step_target_status", "")
    ).startswith("withdrawn")
    target_s = None if target_withdrawn else float(raw_target)
    r_prefill = inputs.throughput.r_prefill_tok_s
    t_preds = [s.logged_t_pred_s for s in inputs.steps]
    peak_context = max((s.context_tokens_total for s in inputs.steps), default=0)
    context_ceiling = per_task_context_ceiling(inputs.steps)

    grid = deadline_grid(
        t_preds,
        n=int(replay_cfg["n_deadlines"]),
        quantile_lo=float(replay_cfg["grid_quantile_lo"]),
        quantile_hi=float(replay_cfg["grid_quantile_hi"]),
        pad_factor=float(replay_cfg["grid_pad_factor"]),
    )

    curve: list[dict[str, Any]] = []
    for deadline_s in grid:
        curve.append(
            _curve_point(
                inputs,
                deadline_s=deadline_s,
                r_prefill_tok_s=r_prefill,
                resamples=resamples,
                seed=seed,
                materiality=materiality,
                use_native_tokens=False,
            )
        )

    # C2 / AM-032: headline is the material deadline on the derived grid, not a fixed wall-clock
    # target. The pre-registered 8 s point is withdrawn (PRE-DATA w.r.t. C2). If no grid point
    # reaches materiality, fall back to the point with the largest KV over-provisioning ratio and
    # mark the selection as non-material.
    material_point = material_deadline_point(curve, materiality_ratio=materiality)
    if material_point is not None:
        headline = {
            **material_point,
            "definition": (
                "loosest grid deadline whose point-estimate KV over-provisioning ratio is >= the "
                f"pre-registered {materiality}x materiality threshold"
            ),
            "selection": "material grid point (C2 / AM-032); not a fixed wall-clock target",
            "material_deadline_reached": True,
        }
    else:
        finite = _finite_curve_points(curve)
        if not finite:
            raise SeamError(
                "deadline curve has no finite surviving points; cannot select a headline"
            )
        fallback = max(
            finite,
            key=lambda point: float(
                point["over_provisioning"]["peak_kv_bytes_resident"].get("point") or 0.0
            ),
        )
        headline = {
            **fallback,
            "definition": (
                "no grid point reached materiality; headline is the grid point with the largest "
                "KV over-provisioning ratio (non-material)"
            ),
            "selection": "max-KV-ratio grid point fallback; material threshold not met",
            "material_deadline_reached": False,
        }
    headline["distribution_units"] = deadline_distribution_units(
        float(headline["deadline_s"]), t_preds
    )
    headline["p95_step_target_s"] = target_s
    headline["p95_step_target_status"] = (
        "withdrawn_AM-032" if target_withdrawn else "legacy_absolute_target"
    )
    headline["absolute_wall_clock_claim"] = False
    headline["seconds_caveat"] = "R unverified pending A4"
    feasibility = _latency_feasibility(curve, target_s=target_s)
    proxy = proxy_error_regression(inputs.steps)
    bias = abs(float(proxy.get("mean_relative_bias_proxy_vs_native", math.nan)))
    threshold = float(replay_cfg["proxy_bias_interval_threshold"])
    boundary_is_interval = not math.isnan(bias) and bias > threshold

    native_curve: list[dict[str, Any]] = []
    if boundary_is_interval:
        # The proxy is biased enough that the filter boundary cannot honestly be drawn as a line.
        # The second curve is the same replay fed the tokenizer's count, and the pair brackets it.
        native_curve = [
            _curve_point(
                inputs,
                deadline_s=deadline_s,
                r_prefill_tok_s=r_prefill,
                resamples=resamples,
                seed=seed,
                materiality=materiality,
                use_native_tokens=True,
            )
            for deadline_s in grid
        ]

    unfiltered = envelope_by_task(
        inputs.steps, deadline_s=DEADLINE_DISABLED_S, r_prefill_tok_s=r_prefill
    )
    adm_frac = inputs.admissible_fraction()
    adm_steps = inputs.timing_admissible_steps()
    unfiltered_adm = envelope_by_task(
        adm_steps, deadline_s=DEADLINE_DISABLED_S, r_prefill_tok_s=r_prefill
    )
    unfiltered_aggregate = _aggregate(unfiltered, resamples=resamples, seed=seed)
    unfiltered_aggregate_adm = _aggregate(unfiltered_adm, resamples=resamples, seed=seed)
    # A disabled deadline imposes no finite decode-rate requirement. Evaluating the formula at the
    # 1e9 sentinel yields a tiny positive number that is arithmetically correct but semantically
    # meaningless and dangerously quotable. Null the statistic at every top-level unfiltered
    # location while retaining numeric values for every real finite deadline in ``curve``.
    undefined_decode_rate = {
        "reason_code": "deadline_disabled_no_required_decode_rate",
        "reason": (
            "DEADLINE_DISABLED_S disables escalation; it is not a service deadline, so a required "
            "decode rate is undefined"
        ),
        "deadline_s": DEADLINE_DISABLED_S,
    }
    for task_envelope in unfiltered.values():
        task_envelope["p95_required_decode_rate_tok_s"] = math.nan
    for task_envelope in unfiltered_adm.values():
        task_envelope["p95_required_decode_rate_tok_s"] = math.nan
    unfiltered_aggregate["p95_required_decode_rate_tok_s"] = None
    unfiltered_aggregate_adm["p95_required_decode_rate_tok_s"] = None
    stratified = _stratified(inputs, deadline_s=headline["deadline_s"])

    # Timing aggregates over canary∩paging-admissible tasks/steps only (C2f).
    adm_task_rows = [
        row
        for row in (inputs.summary.get("per_task") or [])
        if inputs.task_timing_admissible(str(row.get("task_id") or ""))
    ]
    jct_values = [float(row["jct_s"]) for row in adm_task_rows if row.get("jct_s") is not None]
    overrun_steps = [s for s in adm_steps if s.deadline_overrun]
    timing_block = {
        "basis": "timing_admissible_only",
        "timing_rule": "canary_admissible AND paging_admissible",
        "paging_admissible_task_fraction": adm_frac["paging_admissible_task_fraction"],
        "canary_admissible_task_fraction": adm_frac["canary_admissible_task_fraction"],
        "timing_admissible_task_fraction": adm_frac["timing_admissible_task_fraction"],
        "timing_admissible_step_fraction": adm_frac["timing_admissible_step_fraction"],
        "mean_jct_s": _with_admissible_basis(
            statistics.fmean(jct_values) if jct_values else None,
            fraction=adm_frac,
            basis="timing_admissible_only",
        ),
        "p95_jct_s": _with_admissible_basis(
            p95(jct_values) if jct_values else math.nan,
            fraction=adm_frac,
            basis="timing_admissible_only",
        ),
        "deadline_overrun_rate": _with_admissible_basis(
            (len(overrun_steps) / len(adm_steps)) if adm_steps else None,
            fraction=adm_frac,
            basis="timing_admissible_only",
        ),
        "p95_actual_wall_s": _with_admissible_basis(
            p95([s.actual_wall_s for s in adm_steps]) if adm_steps else math.nan,
            fraction=adm_frac,
            basis="timing_admissible_only",
        ),
    }

    return {
        "run_id": inputs.run_id,
        "pre_registration": "docs/EXPERIMENT_escalation_filter.md",
        "stage": 1,
        "integrity_verified": inputs.integrity_verified,
        "replay_self_check": check,
        "rule_source": "seam.agent.policy.decide (imported, not re-implemented)",
        "router_input_field": (
            "prompt_tokens_proxy (chars_div_4_plus_scaffold) - what the router consumed"
        ),
        "deadline_reporting": {
            "primary_units": "distribution (empirical CDF / fraction of observed t_pred range)",
            "seconds_secondary": True,
            "seconds_caveat": "R unverified pending A4",
            "absolute_wall_clock_claims": False,
            "p95_step_target_s": target_s,
            "p95_step_target_status": headline["p95_step_target_status"],
        },
        "peak_context_tokens": peak_context,
        "peak_context_tokens_is_gate": False,
        "context_ceiling": context_ceiling,
        "over_provisioning_vs_ceiling_note": (
            "Read measured over_provisioning alongside context_ceiling: Stage-1 OP 1.243 sat at "
            "85% of its mean ceiling 1.456 (run_id=1a0166b9…). With median ceiling ≥ 3.0 the "
            "achievable OP range widens; P6 falsifies if OP still lands inside Stage-1 CI."
        ),
        "router_step_type_counts": _counts(s.router_step_type for s in inputs.steps),
        "realized_step_type_counts": _counts(s.step_type for s in inputs.steps),
        "router_step_type_note": (
            "the harness rewrites step_type to answer_synthesis AFTER the routing decision, so the "
            "realized label on a terminal step is not what the router priced. The replay uses "
            "routing.step_type; using the realized label would reprice every terminal step and "
            "the counterfactual would silently differ from the rule the run applied."
        ),
        "throughput": {
            "target": inputs.throughput.target,
            "r_prefill_tok_s": inputs.throughput.r_prefill_tok_s,
            "r_decode_tok_s": inputs.throughput.r_decode_tok_s,
            "measured_by_run_id": inputs.throughput.measured_by_run_id,
        },
        "n_out_pred_tokens": inputs.n_out_pred_tokens,
        "kv_bytes_per_token": inputs.kv_bytes_per_token,
        "n_tasks": len(inputs.task_ids),
        "n_steps": len(inputs.steps),
        "cache": {
            "instrumented": inputs.cache_instrumented,
            "probe": inputs.cache_probe,
            "consequence": (
                "prompt_tokens_new and every metric derived from it are UPPER BOUNDS, not "
                "measurements: the runtime reports no cache-reuse counter, so a zero "
                "cached_prompt_tokens cannot distinguish 'no reuse' from 'not instrumented'"
                if not inputs.cache_instrumented
                else "cached_prompt_tokens is a measurement; prompt_tokens_new is a value"
            ),
            "caching_fork_verdict": (
                "UNDETERMINED_FROM_COUNTERS; the run's independent TTFT probe is the only evidence"
                if not inputs.cache_instrumented
                else "determinable from counters"
            ),
            "fraction_steps_cache_evicted": (
                statistics.fmean(1.0 if s.cache_evicted else 0.0 for s in inputs.steps)
                if inputs.steps
                else None
            ),
        },
        "paging_admissibility": {
            **adm_frac,
            "by_task": dict(inputs.paging_admissible_by_task),
            "reasons_by_task": dict(inputs.paging_admissible_reasons_by_task),
            "note": (
                "Envelope endpoints use all blocks; timing endpoints use "
                "canary_admissible ∩ paging_admissible. Blocks are never dropped (C2e/C2f)."
            ),
        },
        "canary_admissibility": {
            "by_task": dict(inputs.canary_admissible_by_task),
            "reasons_by_task": dict(inputs.canary_admissible_reasons_by_task),
            "canary_admissible_task_fraction": adm_frac["canary_admissible_task_fraction"],
            "note": (
                "Canary drift contaminates timing only (wall-clock timeout audit NONE). "
                "Envelope primary unaffected (C2f)."
            ),
        },
        "timing_endpoints": timing_block,
        "unfiltered_envelope": {
            "per_task": unfiltered,
            "aggregate": {
                **unfiltered_aggregate,
                "basis": "all_blocks",
                "paging_admissible_task_fraction": adm_frac["paging_admissible_task_fraction"],
                "canary_admissible_task_fraction": adm_frac["canary_admissible_task_fraction"],
            },
            "aggregate_timing_admissible_subset": {
                **unfiltered_aggregate_adm,
                "basis": "timing_admissible_subset",
                "timing_admissible_task_fraction": adm_frac["timing_admissible_task_fraction"],
                "canary_admissible_task_fraction": adm_frac["canary_admissible_task_fraction"],
                "paging_admissible_task_fraction": adm_frac["paging_admissible_task_fraction"],
            },
            # Alias retained for C2e readers.
            "aggregate_paging_admissible_subset": {
                **unfiltered_aggregate_adm,
                "basis": "timing_admissible_subset",
                "paging_admissible_task_fraction": adm_frac["paging_admissible_task_fraction"],
                "canary_admissible_task_fraction": adm_frac["canary_admissible_task_fraction"],
            },
            "undefined_metrics": {
                "p95_required_decode_rate_tok_s": undefined_decode_rate,
            },
            "p95_per_step_context_tokens": p95(
                [float(s.context_tokens_total) for s in inputs.steps]
            ),
            "p95_per_step_context_tokens_meta": _with_admissible_basis(
                p95([float(s.context_tokens_total) for s in inputs.steps]),
                fraction=adm_frac,
                basis="all_blocks",
            ),
            "note": (
                "computed at DEADLINE_DISABLED_S, i.e. the local-only benchmark envelope; "
                "primary aggregate is over all blocks with timing-admissible subset alongside"
            ),
        },
        "predictor_decomposition": _predictor_decomposition(inputs),
        "bound_labels": _bound_labels(inputs),
        "deadline_grid_s": grid,
        "curve": curve,
        "headline": headline,
        "latency_feasibility": feasibility,
        "proxy_error": {
            **proxy,
            "interval_threshold": threshold,
            "boundary_reported_as_interval": boundary_is_interval,
            "interpretation": (
                f"proxy bias {bias:.3f} exceeds {threshold}: the filter boundary is reported as an "
                f"interval between the proxy-driven and tokenizer-driven replays"
                if boundary_is_interval
                else f"proxy bias {bias:.3f} is within {threshold}: the boundary is a line"
            ),
        },
        "native_token_curve": native_curve,
        "stratified": stratified,
        "predictions": _predictions(
            curve=curve,
            headline=headline,
            feasibility=feasibility,
            materiality_ratio=materiality,
            stratified=stratified,
            peak_context_tokens=peak_context,
            context_ceiling=context_ceiling,
        ),
        "materiality_ratio": materiality,
        "scaffold_tokens_remeasure": {
            "hook": "seam.analysis.efilter.template_scaffold_analysis",
            "note": (
                "Re-measure when model or chat template / tool schema changes; record "
                "fixed_scaffold_tokens and scaffold_tokens_source in the analysis report."
            ),
        },
        "reporting_rule": (
            "no single max is reported as a headline: every envelope figure is the mean of "
            "per-task peaks with a bootstrap CI over tasks, with P95 of per-step context as the "
            "stable companion (pre-registration §4). Deadlines are reported in distribution units; "
            "absolute wall-clock targets are not claimed (AM-032)."
        ),
    }


def _predictor_decomposition(inputs: EfilterInputs) -> dict[str, Any]:
    """Which term of ``t_pred`` the filter is actually selecting on (pre-registration §6).

    §6 frames the caching fork as a question about caching, but the fork is decided by which term
    dominates ``t_pred``, and caching is only one of the two things that decides that. The other is
    the throughput ratio. Prefill overtakes decode only once

    .. code-block:: text

        context_tokens > (R_prefill / R_decode) * n_out_pred

    so on hardware whose prefill rate is two orders of magnitude above its decode rate, a fully
    re-prefilled context can still be decode-dominated. Reported explicitly, because "the cache is
    evicted" is not by itself sufficient for the memory-selective regime §6 predicts.
    """
    shares = [s.prefill_share_of_t_pred() for s in inputs.steps]
    defined = [s for s in shares if not math.isnan(s)]
    ratio = (
        inputs.throughput.r_prefill_tok_s / inputs.throughput.r_decode_tok_s
        if inputs.throughput.r_decode_tok_s > 0
        else math.nan
    )
    crossover = {step_type: ratio * n_out for step_type, n_out in inputs.n_out_pred_tokens.items()}
    router_types = sorted({s.router_step_type for s in inputs.steps})
    observed_peak_context = max((s.context_tokens_total for s in inputs.steps), default=0)
    reached = {
        step_type: observed_peak_context > threshold for step_type, threshold in crossover.items()
    }
    return {
        "formula": "t_pred = prompt_tokens_proxy / R_prefill + n_out_pred / R_decode",
        "r_prefill_over_r_decode": ratio,
        "prefill_share_of_t_pred": {
            "mean": _mean_or_none(defined),
            "min": min(defined) if defined else None,
            "max": max(defined) if defined else None,
            "n_steps_with_decomposition": len(defined),
        },
        "decode_term_s_by_router_step_type": {
            step_type: inputs.n_out_pred_tokens[step_type] / inputs.throughput.r_decode_tok_s
            for step_type in router_types
            if step_type in inputs.n_out_pred_tokens
        },
        "decode_term_is_constant_within_a_step_type": True,
        "context_tokens_for_prefill_to_dominate": crossover,
        "observed_peak_context_tokens": observed_peak_context,
        "prefill_dominant_regime_reached": reached,
        "interpretation": (
            "the filter selects on context only to the extent that prefill dominates t_pred. Where "
            "the decode term dominates, n_out_pred is a constant per step_type, so t_pred varies "
            "across steps ONLY through context - the filter still orders steps by context, but the "
            "deadline band over which it discriminates is narrow and sits above the constant "
            "decode floor."
        ),
    }


def _bound_labels(inputs: EfilterInputs) -> dict[str, Any]:
    """Mark every emitted quantity that is a bound rather than a measurement.

    With no cache-reuse counter, ``prompt_tokens_new`` is an upper bound on the tokens actually
    prefilled, and so is everything computed from it. Labelling them individually keeps a reader
    from lifting one number out of the payload without its caveat.
    """
    if inputs.cache_instrumented:
        return {
            "cache_instrumented": True,
            "note": "cached_prompt_tokens is a measurement; no metric here is a bound",
        }
    return {
        "cache_instrumented": False,
        "upper_bound_metrics": {
            "max_prompt_tokens_new": "derived from prompt_tokens_new",
            "p95_required_decode_rate_tok_s": (
                "the prefill share of the deadline is computed from prompt_tokens_new, so the "
                "residual decode budget is a lower bound and this rate an upper bound"
            ),
            "mean_arithmetic_intensity": "prompt_tokens_new appears in the numerator",
            "cache_evicted / fraction_steps_cache_evicted": (
                "inferred from prompt_tokens_new >= context_tokens_total, which is trivially true "
                "when no reuse can be reported; it is not an observation of an eviction event"
            ),
        },
        "not_bounds": {
            "peak_kv_bytes_resident": (
                "analytic from context_tokens_total and the KV constant; independent of the cache "
                "counter, which is why the primary endpoint survives the instrumentation gap"
            ),
            "peak_context_tokens": "native tokenizer count from the runtime",
            "peak_rss_bytes": "observed process high-water",
        },
        "consequence_for_the_caching_fork": (
            "the fork cannot be settled from counters. The run's TTFT probe is the only evidence, "
            "and it is a behavioural probe, not a counter."
        ),
    }


def _curve_point(
    inputs: EfilterInputs,
    *,
    deadline_s: float,
    r_prefill_tok_s: float,
    resamples: int,
    seed: int,
    materiality: float,
    use_native_tokens: bool,
) -> dict[str, Any]:
    kept = survivors(inputs, deadline_s=deadline_s, use_native_tokens=use_native_tokens)
    # C2f: envelope over ALL blocks; timing over canary∩paging-admissible only.
    adm_frac = inputs.admissible_fraction()
    kept_timing = [s for s in kept if inputs.task_timing_admissible(s.task_id)]
    adm_steps = inputs.timing_admissible_steps()
    filtered = envelope_by_task(kept, deadline_s=deadline_s, r_prefill_tok_s=r_prefill_tok_s)
    unfiltered = envelope_by_task(
        inputs.steps, deadline_s=deadline_s, r_prefill_tok_s=r_prefill_tok_s
    )
    filtered_adm = envelope_by_task(
        kept_timing, deadline_s=deadline_s, r_prefill_tok_s=r_prefill_tok_s
    )
    unfiltered_adm = envelope_by_task(
        adm_steps, deadline_s=deadline_s, r_prefill_tok_s=r_prefill_tok_s
    )

    n_dropped = len(inputs.task_ids) - len(filtered)
    ratios: dict[str, Any] = {}
    for metric in _ENVELOPE_METRICS:
        block = _paired_ratio_bootstrap(
            {t: v[metric] for t, v in unfiltered.items()},
            {t: v[metric] for t, v in filtered.items()},
            resamples=resamples,
            seed=seed,
        )
        point = block.get("point")
        block["material"] = bool(
            point is not None and not math.isnan(point) and point >= materiality
        )
        block["basis"] = "all_blocks"
        block["paging_admissible_task_fraction"] = adm_frac["paging_admissible_task_fraction"]
        block["canary_admissible_task_fraction"] = adm_frac["canary_admissible_task_fraction"]
        if n_dropped:
            # A task whose every step escalates has no local envelope to compare, so it leaves
            # BOTH sides of the paired ratio. That is the only defensible pairing, but it means
            # the ratio at tight deadlines describes the tasks that still run locally, not all of
            # them - recorded here because it biases the ratio toward 1.
            block["tasks_excluded_no_survivors"] = n_dropped
            block["pairing_note"] = (
                f"{n_dropped} of {len(inputs.task_ids)} tasks lost every step at this deadline and "
                f"are excluded from both sides of the paired ratio; the ratio describes the tasks "
                f"that retain local work"
            )
        # Robustness: same ratio on the timing-admissible subset only.
        adm_block = _paired_ratio_bootstrap(
            {t: v[metric] for t, v in unfiltered_adm.items()},
            {t: v[metric] for t, v in filtered_adm.items()},
            resamples=resamples,
            seed=seed,
        )
        adm_block["basis"] = "timing_admissible_subset"
        adm_block["timing_admissible_task_fraction"] = adm_frac["timing_admissible_task_fraction"]
        adm_block["canary_admissible_task_fraction"] = adm_frac["canary_admissible_task_fraction"]
        adm_block["paging_admissible_task_fraction"] = adm_frac["paging_admissible_task_fraction"]
        block["timing_admissible_subset"] = adm_block
        block["paging_admissible_subset"] = adm_block  # C2e alias
        ratios[metric] = block

    wall_values = [s.actual_wall_s for s in kept_timing]
    # Scalars stay numeric for existing consumers; basis/fraction sit beside them (C2e/C2f).
    p95_wall = p95(wall_values) if wall_values else math.nan
    max_wall = max(wall_values) if wall_values else math.nan
    p95_ctx = p95([float(s.context_tokens_total) for s in kept])
    return {
        "deadline_s": deadline_s,
        "n_surviving_steps": len(kept),
        "n_steps": len(inputs.steps),
        "escalation_rate": 1 - len(kept) / len(inputs.steps) if inputs.steps else math.nan,
        "n_tasks_with_survivors": len(filtered),
        "n_tasks": len(inputs.task_ids),
        "n_tasks_without_survivors": n_dropped,
        "filtered_envelope": {
            **_aggregate(filtered, resamples=resamples, seed=seed),
            "basis": "all_blocks",
            "paging_admissible_task_fraction": adm_frac["paging_admissible_task_fraction"],
            "canary_admissible_task_fraction": adm_frac["canary_admissible_task_fraction"],
            "paging_admissible_subset": {
                **_aggregate(filtered_adm, resamples=resamples, seed=seed),
                "basis": "timing_admissible_subset",
                "timing_admissible_task_fraction": adm_frac["timing_admissible_task_fraction"],
                "canary_admissible_task_fraction": adm_frac["canary_admissible_task_fraction"],
                "paging_admissible_task_fraction": adm_frac["paging_admissible_task_fraction"],
            },
        },
        "unfiltered_envelope": {
            **_aggregate(unfiltered, resamples=resamples, seed=seed),
            "basis": "all_blocks",
            "paging_admissible_task_fraction": adm_frac["paging_admissible_task_fraction"],
            "canary_admissible_task_fraction": adm_frac["canary_admissible_task_fraction"],
            "paging_admissible_subset": {
                **_aggregate(unfiltered_adm, resamples=resamples, seed=seed),
                "basis": "timing_admissible_subset",
                "timing_admissible_task_fraction": adm_frac["timing_admissible_task_fraction"],
                "canary_admissible_task_fraction": adm_frac["canary_admissible_task_fraction"],
                "paging_admissible_task_fraction": adm_frac["paging_admissible_task_fraction"],
            },
        },
        "p95_per_step_context_tokens_surviving": p95_ctx,
        "p95_per_step_context_tokens_surviving_meta": _with_admissible_basis(
            p95_ctx, fraction=adm_frac, basis="all_blocks"
        ),
        # Timing endpoints: timing-admissible survivors only (C2f). Envelope stays on all blocks.
        "p95_actual_wall_s_surviving": p95_wall,
        "p95_actual_wall_s_surviving_meta": _with_admissible_basis(
            p95_wall, fraction=adm_frac, basis="timing_admissible_only"
        ),
        "max_actual_wall_s_surviving": max_wall,
        "max_actual_wall_s_surviving_meta": _with_admissible_basis(
            max_wall, fraction=adm_frac, basis="timing_admissible_only"
        ),
        "n_surviving_steps_timing_basis": len(kept_timing),
        "mean_prefill_share_of_t_pred_surviving": (
            statistics.fmean(
                [
                    share
                    for share in (s.prefill_share_of_t_pred() for s in kept)
                    if not math.isnan(share)
                ]
                or [math.nan]
            )
        ),
        "over_provisioning": ratios,
        "router_input": "native" if use_native_tokens else "proxy",
        "paging_admissibility": adm_frac,
    }


def _aggregate(
    per_task: dict[str, dict[str, float]], *, resamples: int, seed: int
) -> dict[str, Any]:
    """Mean of per-task peaks with a bootstrap CI over tasks, per envelope dimension."""
    out: dict[str, Any] = {"n_tasks": len(per_task)}
    for metric in _ENVELOPE_METRICS:
        values = [v[metric] for v in per_task.values() if not math.isnan(v.get(metric, math.nan))]
        ci = bootstrap_ci(values, resamples=resamples, seed=seed, label=f"{metric}_mean")
        out[metric] = {
            **ci.to_dict(),
            "per_task_min": min(values) if values else math.nan,
            "per_task_max": max(values) if values else math.nan,
            "per_task_median": statistics.median(values) if values else math.nan,
        }
    return out


def _latency_feasibility(curve: list[dict[str, Any]], *, target_s: float | None) -> dict[str, Any]:
    """Secondary diagnostic: realized p95 wall vs a wall-clock target, if any.

    Under C2 / AM-032 the pre-registered 8 s target is withdrawn. When ``target_s`` is None this
    records the withdrawal and does not claim absolute wall-clock success or failure. Seconds, when
    shown, carry the caveat that R is unverified pending A4.
    """
    loosest_p95 = max(curve, key=lambda point: point["deadline_s"])["p95_actual_wall_s_surviving"]
    if target_s is None:
        return {
            "target_s": None,
            "target_status": "withdrawn_AM-032",
            "target_reachable": None,
            "deadline_s": None,
            "unfiltered_p95_actual_wall_s": loosest_p95,
            "absolute_wall_clock_claim": False,
            "seconds_caveat": "R unverified pending A4",
            "definition": (
                "no absolute p95 step-latency target; headline is the material grid deadline"
            ),
            "interpretation": (
                "AM-032 withdrew the 8 s headline. Do not treat realized wall seconds as a "
                "success/fail criterion for P1-P3."
            ),
        }
    meeting = [
        point
        for point in curve
        if point["n_surviving_steps"] > 0
        and not math.isnan(point["p95_actual_wall_s_surviving"])
        and point["p95_actual_wall_s_surviving"] <= target_s
    ]
    if meeting:
        # The loosest such deadline: the least escalation that buys the target, so the filter is
        # not credited with a reduction the target did not require.
        best = max(meeting, key=lambda point: point["deadline_s"])
        return {
            "target_s": target_s,
            "target_status": "legacy_absolute_target",
            "target_reachable": True,
            "deadline_s": best["deadline_s"],
            "p95_actual_wall_s_surviving": best["p95_actual_wall_s_surviving"],
            "escalation_rate": best["escalation_rate"],
            "n_surviving_steps": best["n_surviving_steps"],
            "unfiltered_p95_actual_wall_s": loosest_p95,
            "absolute_wall_clock_claim": False,
            "seconds_caveat": "R unverified pending A4",
            "definition": (
                "loosest grid deadline whose surviving steps have p95 realized wall time <= the "
                "stated target"
            ),
            "over_provisioning": best["over_provisioning"],
        }
    tightest = min(curve, key=lambda point: point["deadline_s"])
    with_survivors = [point for point in curve if point["n_surviving_steps"] > 0]
    best_effort = (
        min(with_survivors, key=lambda point: point["p95_actual_wall_s_surviving"])
        if with_survivors
        else tightest
    )
    return {
        "target_s": target_s,
        "target_status": "legacy_absolute_target",
        "target_reachable": False,
        "deadline_s": None,
        "unfiltered_p95_actual_wall_s": loosest_p95,
        "absolute_wall_clock_claim": False,
        "seconds_caveat": "R unverified pending A4",
        "best_achievable_p95_actual_wall_s": best_effort["p95_actual_wall_s_surviving"],
        "best_achievable_at_deadline_s": best_effort["deadline_s"],
        "escalation_rate_there": best_effort["escalation_rate"],
        "definition": "p95(actual_wall_s of surviving steps) <= target_s",
        "interpretation": (
            "NO deadline on the grid delivers the target. Escalation is triggered on PREDICTED "
            "latency, and the predictor's output-length term is a per-step-type median, so the "
            "filter does not order steps by their realized latency. Where realized latency is "
            "dominated by output length rather than by context, tightening the deadline removes "
            "the large-context steps without removing the slow ones. The premise of P2 is "
            "therefore not satisfiable on this workload, which is a finding about the escalation "
            "mechanism and is reported as one."
        ),
    }


# ==================================================================================================
# Prompt B: sealed-data-only forensics and counterfactuals
# ==================================================================================================


def _completion_distribution(steps: Sequence[StepView]) -> dict[str, Any]:
    """Descriptive distribution and an explicit fixed-width histogram over completion lengths."""
    values = sorted(int(step.completion_tokens) for step in steps)
    if not values:
        raise SeamError("completion-length distribution is empty")

    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
    width = 64
    upper = max(512, math.ceil(max(values) / width) * width)
    edges = list(range(0, upper + width, width))
    bins: list[dict[str, Any]] = []
    for lo, hi in pairwise(edges):
        bins.append(
            {
                "left_inclusive": lo,
                "right_exclusive": hi,
                "count": sum(lo <= value < hi for value in values),
            }
        )
    bins.append(
        {
            "left_inclusive": edges[-1],
            "right_exclusive": None,
            "count": sum(value >= edges[-1] for value in values),
        }
    )

    return {
        "n": len(values),
        "min_tokens": values[0],
        "mean_tokens": mean,
        "median_tokens": _quantile(values, 0.5),
        "p90_tokens": _quantile(values, 0.9),
        "p99_tokens": _quantile(values, 0.99),
        "max_tokens": values[-1],
        "sample_sd_tokens": sample_sd,
        "cv_sample_sd_over_mean": sample_sd / mean if mean else math.nan,
        "quantile_convention": (
            "nearest-rank: sort ascending and select rank ceil(q*n), one-indexed; q=0.5 is the "
            "nearest-rank median (n=51 is odd)"
        ),
        "sd_convention": "sample standard deviation (n-1 denominator)",
        "histogram": {
            "bin_width_tokens": width,
            "interval_convention": "[left, right), final bin [left, +infinity)",
            "bins": bins,
            "count_check": sum(block["count"] for block in bins),
        },
    }


def _counterfactual_t_preds(
    inputs: EfilterInputs, *, n_out_pred_tokens: dict[str, int]
) -> list[float]:
    """Reprice logged pre-execution inputs with the imported production policy."""
    return [
        decide(
            throughput=inputs.throughput,
            deadline_s=DEADLINE_DISABLED_S,
            prompt_tokens=step.prompt_tokens_proxy,
            step_type=step.router_step_type,
            n_out_pred_tokens=n_out_pred_tokens,
        ).t_pred_s
        for step in inputs.steps
    ]


def _finite_curve_points(curve: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        point
        for point in curve
        if point["n_surviving_steps"] > 0
        and math.isfinite(float(point["p95_actual_wall_s_surviving"]))
    ]


def tail_latency_replay(inputs: EfilterInputs, *, replay_cfg: dict[str, Any]) -> dict[str, Any]:
    """Replay median/p90/p99 realized-length estimators over one sealed local-only trajectory.

    The logged replay is verified first with the original estimator. The three estimator
    counterfactuals then change only ``n_out_pred`` and re-call :func:`policy.decide` on each
    step's logged pre-execution inputs. Realized wall times are never predicted or synthesized:
    surviving steps retain their observed local-only ``actual_wall_s``.
    """
    check = self_check_replay(inputs)
    if not check["passed"]:
        raise SeamError("Prompt B tail replay refused: the logged-policy self-check failed")

    distribution = _completion_distribution(inputs.steps)
    distribution["run_id"] = inputs.run_id
    sorted_lengths = sorted(float(step.completion_tokens) for step in inputs.steps)
    quantiles = {
        "baseline_median": (0.5, int(_quantile(sorted_lengths, 0.5))),
        "p90": (0.9, int(_quantile(sorted_lengths, 0.9))),
        "p99": (0.99, int(_quantile(sorted_lengths, 0.99))),
    }
    r_prefill = inputs.throughput.r_prefill_tok_s
    resamples = int(replay_cfg["bootstrap_resamples"])
    seed = int(replay_cfg["bootstrap_seed"])
    materiality = float(replay_cfg["materiality_ratio"])
    raw_target = replay_cfg.get("p95_step_target_s")
    target_s = None if raw_target is None else float(raw_target)
    unfiltered_p95 = p95([step.actual_wall_s for step in inputs.steps])

    replays: dict[str, Any] = {}
    for label, (quantile, n_out_pred) in quantiles.items():
        counterfactual_n_out = dict.fromkeys(inputs.n_out_pred_tokens, n_out_pred)
        counterfactual_inputs = replace(
            inputs,
            n_out_pred_tokens=counterfactual_n_out,
        )
        t_preds = _counterfactual_t_preds(inputs, n_out_pred_tokens=counterfactual_n_out)
        grid = deadline_grid(
            t_preds,
            n=int(replay_cfg["n_deadlines"]),
            quantile_lo=float(replay_cfg["grid_quantile_lo"]),
            quantile_hi=float(replay_cfg["grid_quantile_hi"]),
            pad_factor=float(replay_cfg["grid_pad_factor"]),
        )
        curve = [
            _curve_point(
                counterfactual_inputs,
                deadline_s=deadline_s,
                r_prefill_tok_s=r_prefill,
                resamples=resamples,
                seed=seed,
                materiality=materiality,
                use_native_tokens=False,
            )
            for deadline_s in grid
        ]
        finite = _finite_curve_points(curve)
        best = min(finite, key=lambda point: point["p95_actual_wall_s_surviving"])
        material = [
            point
            for point in finite
            if float(point["over_provisioning"]["peak_kv_bytes_resident"].get("point", math.nan))
            >= materiality
        ]
        # The loosest qualifying deadline is the minimum escalation that reaches materiality.
        material_point = max(material, key=lambda point: point["deadline_s"]) if material else None
        tail_meeting = (
            [
                point
                for point in finite
                if target_s is not None and point["p95_actual_wall_s_surviving"] <= target_s
            ]
            if target_s is not None
            else []
        )
        tail_point = (
            max(tail_meeting, key=lambda point: point["deadline_s"]) if tail_meeting else None
        )

        def _point_summary(point: dict[str, Any] | None) -> dict[str, Any] | None:
            if point is None:
                return None
            kv = point["over_provisioning"]["peak_kv_bytes_resident"]
            return {
                "deadline_s": point["deadline_s"],
                "p95_actual_wall_s_surviving": point["p95_actual_wall_s_surviving"],
                "escalation_rate": point["escalation_rate"],
                "n_surviving_steps": point["n_surviving_steps"],
                "kv_over_provisioning_ratio": kv["point"],
                "kv_over_provisioning_bootstrap_ci95": [kv["lo"], kv["hi"]],
                "kv_over_provisioning_bootstrap_n_tasks": kv["n_tasks"],
                "bootstrap_resamples": kv.get("resamples"),
            }

        replays[label] = {
            "run_id": inputs.run_id,
            "realized_completion_quantile": quantile,
            "n_out_pred_tokens": n_out_pred,
            "effective_n_out_pred_by_router_step_type": counterfactual_n_out,
            "deadline_grid_s": grid,
            "best_realized_tail_across_grid": {
                **(_point_summary(best) or {}),
                "unfiltered_p95_actual_wall_s": unfiltered_p95,
                "ratio_best_over_unfiltered": (
                    best["p95_actual_wall_s_surviving"] / unfiltered_p95
                ),
            },
            "material_deadline": _point_summary(material_point),
            "material_deadline_definition": (
                "loosest grid deadline whose point-estimate KV over-provisioning ratio is >= the "
                f"pre-registered {materiality}x materiality threshold"
            ),
            "tail_target_s": target_s,
            "any_grid_point_meets_tail_target": bool(tail_meeting),
            "least_escalation_tail_target_point": _point_summary(tail_point),
            "curve": curve,
        }

    return {
        "run_id": inputs.run_id,
        "replay_self_check": check,
        "policy_function": "seam.agent.policy.decide",
        "logged_inputs": [
            "prompt_tokens_proxy",
            "routing.step_type",
            "throughput.r_prefill_tok_s",
            "throughput.r_decode_tok_s",
        ],
        "original_logged_n_out_pred_tokens": inputs.n_out_pred_tokens,
        "completion_length_distribution": distribution,
        "unfiltered_p95_actual_wall_s": unfiltered_p95,
        "replays": replays,
        "realized_latency_counterfactual": (
            "At each counterfactual deadline, policy.decide selects survivors from the sealed "
            "pre-execution router inputs. A survivor keeps its observed local-only actual_wall_s; "
            "an escalated step is omitted. No cloud latency or replacement output is imputed."
        ),
        "limitations": [
            "Trace replay holds the local-only trajectory fixed after a counterfactual escalation; "
            "a realized hybrid trajectory would fork.",
            "The estimator changes only selection. It does not predict a new realized latency for "
            "any step.",
            f"The finite {int(replay_cfg['n_deadlines'])}-point data-derived grid can establish "
            "whether a sampled grid point meets the target, not prove that no deadline between "
            "points would.",
            "Every routing.step_type in this run is tool_call_synthesis, so the realized-length "
            "quantiles are pooled over all 51 steps rather than estimated per router stratum.",
            "Cache counters are uninstrumented; this does not alter actual_wall_s, but "
            "prompt_tokens_new-derived envelope quantities remain bounds.",
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def template_scaffold_analysis(inputs: EfilterInputs, *, root: Path) -> dict[str, Any]:
    """Measure the fixed scaffold through the exact tokenizer/template path used by the run."""
    from transformers import AutoTokenizer

    from seam.agent.tools import SYSTEM_PROMPT, TOOL_SPECS
    from seam.backends.local_openvino import _tool_to_openai_schema

    recorded_dir = Path(str(inputs.summary["backend_config"]["model_dir"]))
    model_dir = root / "models" / recorded_dir.name
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    tools = cast(
        list[dict[Any, Any] | Callable[..., Any]],
        [_tool_to_openai_schema(tool) for tool in TOOL_SPECS],
    )
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    scaffold_tokens = len(tokenizer(str(rendered))["input_ids"])

    baseline = proxy_error_regression(inputs.steps)
    corrected_steps = [
        replace(
            step,
            prompt_tokens_proxy=step.prompt_tokens_proxy + scaffold_tokens,
        )
        for step in inputs.steps
    ]
    corrected = proxy_error_regression(corrected_steps)
    corrected["model"] = (
        "native_prompt_tokens ~ a + b * (prompt_tokens_proxy + fixed_template_tokens)"
    )
    intercept = float(baseline["intercept"])

    tokenizer_path = model_dir / "tokenizer.json"
    template_path = model_dir / "chat_template.jinja"
    manifest = json.loads(
        (root / "raw" / inputs.run_id / "manifest.json").read_text(encoding="utf-8")
    )
    return {
        "run_id": inputs.run_id,
        "tokenizer_path": str(tokenizer_path.relative_to(root)),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "chat_template_path": str(template_path.relative_to(root)),
        "chat_template_sha256": _sha256(template_path),
        "model_revision": manifest["model"]["revision"],
        "render_path": (
            "AutoTokenizer.from_pretrained(local_files_only=True) -> apply_chat_template with the "
            "same system prompt, six TOOL_SPECS, add_generation_prompt=True, tokenize=False, "
            "enable_thinking=False -> tokenizer(rendered)['input_ids']"
        ),
        "empty_conversation_definition": (
            "no user/assistant/tool-result messages; retain the fixed harness system prompt, tool "
            "schemas, template wrappers, and generation prompt because the router proxy sees none "
            "of them"
        ),
        "fixed_scaffold_tokens": scaffold_tokens,
        "rendered_scaffold_chars": len(str(rendered)),
        "baseline_regression": baseline,
        "intercept_decomposition": {
            "fitted_intercept_tokens": intercept,
            "measured_fixed_scaffold_tokens": scaffold_tokens,
            "scaffold_fraction_of_intercept": scaffold_tokens / intercept,
            "residual_intercept_tokens_direct_subtraction": intercept - scaffold_tokens,
            "residual_fraction_of_intercept": (intercept - scaffold_tokens) / intercept,
            "note": (
                "Direct decomposition subtracts token counts. The corrected OLS intercept differs "
                "because shifting x by 621 changes the intercept by slope*621."
            ),
        },
        "proposed_router_formula": ("sum(message content characters) // 4 + fixed_template_tokens"),
        "corrected_fixed_offset_regression": corrected,
        "sealed_transcript_sufficiency": {
            "fixed_offset_correction_applied_to_all_steps": True,
            "n_steps": len(inputs.steps),
            "full_dynamic_template_correction_possible": False,
            "reason": (
                "steps.ndjson seals proxy and native counts but not complete assistant text. The "
                "fixed scaffold can be added offline, but per-step role wrappers and serialization "
                "cannot be re-rendered exactly for all steps."
            ),
        },
        "scaffold_tokens_source": (
            f"template_scaffold_analysis on run_id={inputs.run_id}; per-model/per-template - "
            "re-measure when either changes"
        ),
        "router_lands_scaffold": True,
        "router_formula_landed_in_harness": "chars_div_4_plus_scaffold",
    }


def _sealed_manifest_inventory(root: Path) -> dict[str, Any]:
    run_ids: list[str] = []
    quantizations: dict[str, int] = {}
    int8_run_ids: list[str] = []
    for run_dir in sorted((root / "raw").iterdir()):
        manifest_path = run_dir / "manifest.json"
        if not run_dir.is_dir() or not (run_dir / ".sealed").exists() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest.get("run_id", run_dir.name))
        run_ids.append(run_id)
        quantization = str((manifest.get("model") or {}).get("quantization"))
        quantizations[quantization] = quantizations.get(quantization, 0) + 1
        if "int8" in quantization.lower():
            int8_run_ids.append(run_id)
    return {
        "sealed_run_ids_searched": run_ids,
        "n_sealed_manifests_searched": len(run_ids),
        "quantization_counts": quantizations,
        "int8_run_ids": int8_run_ids,
    }


def _b1_provenance_forensics(root: Path, *, candidate_run_id: str) -> dict[str, Any]:
    run_dir = open_run_dir(candidate_run_id, repo_root=root)
    if not run_dir.is_sealed() or not verify_sealed(run_dir):
        raise SeamError(f"B1 candidate {candidate_run_id} is not an intact sealed run")
    manifest = json.loads((run_dir.path / "manifest.json").read_text(encoding="utf-8"))
    matrix = json.loads((run_dir.path / "affinity_matrix.json").read_text(encoding="utf-8"))
    inventory = _sealed_manifest_inventory(root)

    candidate_arms: dict[str, Any] = {}
    for arm_id in ("A5", "A6"):
        arm = matrix["summary"][arm_id]
        placements = [
            {
                "block": block,
                "position_zero_indexed": schedule.index(arm_id),
            }
            for block, schedule in enumerate(matrix["schedule"])
        ]
        candidate_arms[arm_id] = {
            "run_id": candidate_run_id,
            "description": matrix["runs"][arm_id][0]["description"],
            "target": matrix["runs"][arm_id][0]["target"],
            "quantization": manifest["model"]["quantization"],
            "decode_tok_s_mean": arm["r_decode_tok_s_mean"],
            "decode_tok_s_bootstrap_ci95": [
                arm["r_decode_tok_s_ci"]["lo"],
                arm["r_decode_tok_s_ci"]["hi"],
            ],
            "complete_scored_decode_series_tok_s": arm["r_decode_tok_s_all"],
            "schedule_positions": placements,
            "warmup_throughput_series": None,
            "warmup_series_reason": (
                "the sealed matrix records two scored generations per block but does not record "
                "backend warmup generations or their throughput"
            ),
        }

    a5 = float(candidate_arms["A5"]["decode_tok_s_mean"])
    a6 = float(candidate_arms["A6"]["decode_tok_s_mean"])
    all_interleaved = all("A5" in schedule and "A6" in schedule for schedule in matrix["schedule"])
    return {
        "claimed_comparison": {
            "numerator": {"label": "INT4", "decode_tok_s": 15.1},
            "denominator": {"label": "INT8", "decode_tok_s": 7.4},
            "claimed_ratio": 2.04,
            "status": "claim under provenance audit, not emitted as a measurement",
        },
        "exact_sealed_provenance": None,
        "one_or_two_runs": "UNKNOWN",
        "run_ids": [],
        "timestamps_utc": [],
        "manifest_diff_if_separate": None,
        "inventory": inventory,
        "nearest_sealed_candidate": {
            "run_id": candidate_run_id,
            "timestamp_utc": manifest["timestamp_utc"],
            "integrity_self_check": manifest["integrity"]["self_check"],
            "model_name": manifest["model"]["name"],
            "model_quantization": manifest["model"]["quantization"],
            "arms": candidate_arms,
            "a5_over_a6_ratio": a5 / a6,
            "same_run": True,
            "interleaved_within_each_block": all_interleaved,
            "randomized": matrix.get("shuffle_seed") is not None,
            "shuffle_seed": matrix.get("shuffle_seed"),
            "complete_block_schedule": matrix["schedule"],
            "matches_claim": False,
            "mismatch_reason": (
                "both arms use the same INT4 model and their sealed means are "
                f"{a5:.6f} and {a6:.6f} tok/s, not 15.1 and 7.4"
            ),
            "canary": {
                "calibrated_canary_present": False,
                "reason": "no calibrated canary is recorded in the sealed run",
            },
            "within_run_stability_checks": matrix.get("throttle_detectors"),
            "verification_gates": matrix.get("verification_gates"),
        },
        "verdict": "UNKNOWN",
        "verdict_reason": (
            "No intact sealed manifest contains an INT8 arm, and the nearest sealed one-run "
            "candidate is an interleaved P-core/LP-E comparison of the same INT4 model with "
            "different means. Provenance is insufficient to assign 15.1 and 7.4 to INT4/INT8."
        ),
        "bandwidth_ceiling_fraction": {
            "claimed_fraction": 0.29,
            "verdict": "UNKNOWN",
            "reason": (
                "the 15.1 tok/s numerator has no exact sealed provenance under the stated "
                "comparison, and the bandwidth denominator is a derived peak, not a measured "
                "Platform-A bandwidth result; platform memory.bandwidth_gbps_measured is null"
            ),
            "measured_bandwidth_run_id": None,
            "prohibited_interpretation": (
                "approximately 120 GB/s is derived from LPDDR5X-7467 and a 128-bit bus, never "
                "measured bandwidth"
            ),
        },
    }


def prompt_b_offline_analysis(
    inputs: EfilterInputs,
    *,
    replay_cfg: dict[str, Any],
    root: Path,
    b1_candidate_run_id: str,
) -> dict[str, Any]:
    """Assemble Prompt B from sealed records only; performs no timed generation or external call."""
    return {
        "analysis": "Prompt B sealed-data offline analysis",
        "source_run_ids": [b1_candidate_run_id, inputs.run_id],
        "B1": _b1_provenance_forensics(root, candidate_run_id=b1_candidate_run_id),
        "B2": tail_latency_replay(inputs, replay_cfg=replay_cfg),
        "B3": template_scaffold_analysis(inputs, root=root),
        "B4": {
            "run_id": inputs.run_id,
            "unfiltered_p95_required_decode_rate_tok_s": None,
            "reason_code": "deadline_disabled_no_required_decode_rate",
            "real_finite_deadline_values_remain_numeric": True,
        },
        "scope": {
            "sealed_data_only": True,
            "timed_performance_run": False,
            "cloud_call": False,
            "secret_loaded": False,
            "raw_mutated": False,
            "router_modified": False,
        },
    }


# ==================================================================================================
# Figure
# ==================================================================================================


def strict_json_payload(value: Any) -> Any:
    """Replace every non-finite float with ``None`` so the payload is strict JSON.

    An undefined envelope is a real outcome - a task can lose every step at a tight deadline - so
    NaN occurs legitimately in this analysis. It must not reach the file. ``json.dumps`` writes a
    bare ``NaN`` token, which is not valid JSON: a strict reader rejects the deliverable and a lax
    one silently yields a float that compares false against everything, including itself. ``null``
    is the representation JSON has for "no value", so that is what is written.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: strict_json_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strict_json_payload(v) for v in value]
    return value


def write_figure(result: dict[str, Any], path: Path) -> Path | None:
    """Envelope and over-provisioning versus deadline. Returns None if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    curve = result["curve"]
    deadlines = [point["deadline_s"] for point in curve]
    filtered = [
        point["filtered_envelope"]["peak_kv_bytes_resident"]["point"] / 1e9 for point in curve
    ]
    lo = [point["filtered_envelope"]["peak_kv_bytes_resident"]["lo"] / 1e9 for point in curve]
    hi = [point["filtered_envelope"]["peak_kv_bytes_resident"]["hi"] / 1e9 for point in curve]
    unfiltered = result["unfiltered_envelope"]["aggregate"]["peak_kv_bytes_resident"]["point"] / 1e9
    ratio = [point["over_provisioning"]["peak_kv_bytes_resident"]["point"] for point in curve]
    ratio_lo = [point["over_provisioning"]["peak_kv_bytes_resident"]["lo"] for point in curve]
    ratio_hi = [point["over_provisioning"]["peak_kv_bytes_resident"]["hi"] for point in curve]
    escalation = [point["escalation_rate"] for point in curve]

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10.5), sharex=True)

    axes[0].plot(deadlines, filtered, marker="o", ms=3, label="filtered (surviving steps)")
    axes[0].fill_between(deadlines, lo, hi, alpha=0.2)
    axes[0].axhline(unfiltered, ls="--", color="k", label="unfiltered (local-only benchmark)")
    axes[0].set_ylabel("mean per-task peak KV (GB)")
    axes[0].set_title(
        f"E-FILTER Stage 1 - local envelope vs escalation deadline\nrun {result['run_id']}",
        fontsize=10,
    )
    axes[0].legend(fontsize=8)

    axes[1].plot(deadlines, ratio, marker="o", ms=3, color="tab:red")
    axes[1].fill_between(deadlines, ratio_lo, ratio_hi, alpha=0.2, color="tab:red")
    axes[1].axhline(result["materiality_ratio"], ls=":", color="k", label="materiality (1.2x)")
    axes[1].set_ylabel("over-provisioning\n(unfiltered / filtered)")
    axes[1].legend(fontsize=8)

    axes[2].plot(deadlines, escalation, marker="o", ms=3, color="tab:green")
    axes[2].set_ylabel("counterfactual escalation rate")
    axes[2].set_xlabel("deadline D (s, log scale)")
    axes[2].set_xscale("log")
    for axis in axes:
        axis.grid(alpha=0.3)
    axes[0].axvline(result["headline"]["deadline_s"], ls="-.", color="tab:blue", lw=1)
    axes[1].axvline(result["headline"]["deadline_s"], ls="-.", color="tab:blue", lw=1)
    axes[2].axvline(result["headline"]["deadline_s"], ls="-.", color="tab:blue", lw=1)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ==================================================================================================
# CLI
# ==================================================================================================


def _fmt(value: Any) -> str:
    """Format a possibly-undefined statistic without pretending it has a value."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "undefined"
    return f"{float(value):.4g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="sealed E-FILTER Stage 1 run")
    parser.add_argument(
        "--out-name",
        default="envelope_vs_deadline.json",
        help=(
            "output filename. The default is the study deliverable and belongs to the sealed full "
            "run; a pilot must be written under its own name so it cannot be mistaken for it."
        ),
    )
    parser.add_argument(
        "--accept-uninstrumented-cache",
        action="store_true",
        help=(
            "proceed when the runtime reports no cache-reuse counter; every prompt_tokens_new "
            "derived quantity is then labelled an upper bound"
        ),
    )
    parser.add_argument(
        "--prompt-b-out-name",
        default=None,
        help="also emit the Prompt-B sealed-data analysis under this filename",
    )
    parser.add_argument(
        "--b1-candidate-run-id",
        default=None,
        help="sealed affinity-matrix candidate inspected by Prompt B1",
    )
    args = parser.parse_args(argv)
    if bool(args.prompt_b_out_name) != bool(args.b1_candidate_run_id):
        parser.error("--prompt-b-out-name and --b1-candidate-run-id must be supplied together")

    root = repo_root(Path(__file__).parent)
    cfg = yaml.safe_load((root / "configs" / "efilter.yaml").read_text(encoding="utf-8"))
    replay_cfg = cfg["replay"]

    inputs = load_run(
        args.run_id,
        root=root,
        accept_uninstrumented_cache=args.accept_uninstrumented_cache,
    )
    result = analyze(inputs, replay_cfg=replay_cfg)

    out_dir = root / replay_cfg["outputs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    out_path.write_text(
        json.dumps(strict_json_payload(result), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    figure = write_figure(result, out_path.with_suffix(".png"))
    prompt_b_path: Path | None = None
    if args.prompt_b_out_name:
        prompt_b = prompt_b_offline_analysis(
            inputs,
            replay_cfg=replay_cfg,
            root=root,
            b1_candidate_run_id=str(args.b1_candidate_run_id),
        )
        prompt_b_path = out_dir / str(args.prompt_b_out_name)
        prompt_b_path.write_text(
            json.dumps(
                strict_json_payload(prompt_b),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    headline = result["headline"]
    kv = headline["over_provisioning"]["peak_kv_bytes_resident"]
    print(f"run {result['run_id']}: {result['n_steps']} steps over {result['n_tasks']} tasks")
    print(f"replay self-check: {'PASS' if result['replay_self_check']['passed'] else 'FAIL'}")
    print(f"cache instrumented: {result['cache']['instrumented']}")
    print(
        f"proxy bias: {result['proxy_error'].get('mean_relative_bias_proxy_vs_native')}, "
        f"boundary as interval: {result['proxy_error']['boundary_reported_as_interval']}"
    )
    print(
        f"headline D={headline['deadline_s']:.3f}s: "
        f"escalation {headline['escalation_rate']:.2%}, "
        f"p95 realized wall {_fmt(headline['p95_actual_wall_s_surviving'])}s, "
        f"KV over-provisioning {_fmt(kv['point'])} "
        f"[{_fmt(kv['lo'])}, {_fmt(kv['hi'])}]"
    )
    feasibility = result["latency_feasibility"]
    print(
        f"p95 {feasibility['target_s']}s target reachable: "
        f"{feasibility['target_reachable']} "
        f"(unfiltered p95 realized {_fmt(feasibility['unfiltered_p95_actual_wall_s'])}s)"
    )
    for key in ("P1", "P2", "P3", "P4", "P5", "P6"):
        print(f"  {key}: {result['predictions'][key]['verdict']}")
    print(f"wrote {out_path}")
    if prompt_b_path:
        print(f"wrote {prompt_b_path}")
    if figure:
        print(f"wrote {figure}")
    else:
        print("matplotlib unavailable; figure not written")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
