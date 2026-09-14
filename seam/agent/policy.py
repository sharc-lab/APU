"""Deadline-aware local-first routing policy (AMENDMENTS.md AM-022).

The mechanism under test. Each agent step is priced against a per-step latency deadline using
**measured** per-target throughput; a step predicted to overrun is sent to the cloud instead:

.. code-block:: text

    t_pred(s,T) = prompt_tokens(s) / R_prefill(T) + n_out_pred(step_type(s)) / R_decode(T)
    escalate   iff  t_pred(s,T) > D

Slower local silicon raises ``t_pred``, so it escalates more often. That is the causal chain the
slice exists to test.

Three properties are load-bearing and are enforced here rather than left to discipline.

**Prefill is included.** It is real local work and it grows through a trajectory as the transcript
accumulates, so later steps escalate more often than earlier ones. That is realistic and wanted.

**The deadline is ADVISORY.** ``n_out_pred`` is a *median*, so roughly half of locally-executed
steps overrun the deadline. That is the predictor's error rate, not a defect, and
:class:`RoutingDecision` records it so it can be reported alongside the escalation curve.

**The isolation invariant is asserted, not assumed.** Between the ``cpu-p`` and ``cpu-lpe`` arms
the only permitted difference is throughput. :func:`assert_isolated` fails loudly if anything else
differs, because an unnoticed second difference converts the experiment into a comparison of
something nobody chose to study.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Literal

from seam.errors import IsolationViolationError, SeamError

__all__ = [
    "DEADLINE_DISABLED_MIN_HEADROOM",
    "DEADLINE_DISABLED_S",
    "RoutingDecision",
    "StepType",
    "ThroughputModel",
    "assert_escalation_disabled",
    "assert_identical_deadline_grid",
    "assert_isolated",
    "decide",
]

#: Deadline sentinel for an escalation-disabled arm (E-FILTER Arm L). Escalation is
#: ``t_pred > deadline_s``, so a deadline far above any achievable ``t_pred`` keeps every step local
#: through exactly the same code path a hybrid run uses - no second code path, no separate policy.
#:
#: **Deliberately not** ``float("inf")``. ``json.dumps`` emits a bare ``Infinity`` token, which is
#: not valid strict JSON; every downstream reader that uses a strict parser would then either fail
#: or silently coerce. A finite sentinel is recorded in the manifest so a reader can distinguish
#: "escalation disabled" from "the deadline happened to be loose".
DEADLINE_DISABLED_S: Final[float] = 1e9

#: Required ratio between :data:`DEADLINE_DISABLED_S` and any observed ``t_pred``. If a predicted
#: step latency ever came within this factor of the sentinel, the sentinel would no longer be
#: distinguishable from a loose-but-real deadline and the arm's "no step could escalate" claim
#: would rest on arithmetic nobody checked.
DEADLINE_DISABLED_MIN_HEADROOM: Final[float] = 1e6

#: Step taxonomy for this slice. Deliberately coarse - the full §7.1 taxonomy and its kappa study
#: belong to G1, not here - but typed, because ``n_out_pred`` is per step type.
StepType = Literal["tool_call_synthesis", "answer_synthesis"]


@dataclass(frozen=True, slots=True)
class ThroughputModel:
    """Measured per-target throughput. The ONLY term permitted to differ between arms."""

    target: str
    #: Tokens/second during prompt ingestion.
    r_prefill_tok_s: float
    #: Tokens/second during generation.
    r_decode_tok_s: float
    #: run_id of the step-1 baseline this came from. No number without a run_id (spec §9.2).
    measured_by_run_id: str

    def predict_s(self, *, prompt_tokens: int, n_out_pred: int) -> float:
        if self.r_prefill_tok_s <= 0 or self.r_decode_tok_s <= 0:
            raise ValueError(
                f"non-positive throughput for {self.target}: prefill={self.r_prefill_tok_s}, "
                f"decode={self.r_decode_tok_s}. A zero would make every step escalate and would "
                f"look like a real silicon effect."
            )
        return prompt_tokens / self.r_prefill_tok_s + n_out_pred / self.r_decode_tok_s


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """One routing decision, fully logged (AM-022 item 7)."""

    assigned_target: Literal["local", "cloud"]
    reason: str
    t_pred_s: float
    deadline_s: float
    prompt_tokens: int
    n_out_pred_tokens: int
    step_type: StepType
    #: Wall-clock cost of evaluating the predictor. Not expected to bind at this scale, but it is
    #: exactly the quantity H3's routing-amortization bound concerns, so the slice baselines it.
    decision_ns: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def decide(
    *,
    throughput: ThroughputModel,
    deadline_s: float,
    prompt_tokens: int,
    step_type: StepType,
    n_out_pred_tokens: dict[str, int],
) -> RoutingDecision:
    """Route one step. Pure function of measured throughput, the deadline, and the prompt."""
    if not math.isfinite(deadline_s):
        raise SeamError(
            f"non-finite deadline_s={deadline_s!r} is refused. json.dumps would write a bare "
            f"Infinity/NaN token, which strict JSON readers reject, so the routing decision would "
            f"not survive being logged. Use DEADLINE_DISABLED_S to disable escalation."
        )
    t0 = time.perf_counter_ns()
    n_out = n_out_pred_tokens[step_type]
    t_pred = throughput.predict_s(prompt_tokens=prompt_tokens, n_out_pred=n_out)
    escalate = t_pred > deadline_s
    decision_ns = time.perf_counter_ns() - t0

    return RoutingDecision(
        assigned_target="cloud" if escalate else "local",
        reason=(
            f"t_pred={t_pred:.3f}s > deadline={deadline_s:.3f}s"
            if escalate
            else f"t_pred={t_pred:.3f}s <= deadline={deadline_s:.3f}s"
        ),
        t_pred_s=t_pred,
        deadline_s=deadline_s,
        prompt_tokens=prompt_tokens,
        n_out_pred_tokens=n_out,
        step_type=step_type,
        decision_ns=decision_ns,
    )


def assert_escalation_disabled(*, deadline_s: float, t_pred_s: float) -> None:
    """Verify by readback that escalation really is disabled for this step.

    Two things are checked, because setting the sentinel is not evidence it took effect:

    1. The deadline in force **is** :data:`DEADLINE_DISABLED_S`, not merely a large number.
    2. The observed ``t_pred`` sits at least :data:`DEADLINE_DISABLED_MIN_HEADROOM` below it, so
       "no step escalated" is a structural property of the arm rather than a lucky margin.

    Raises:
        SeamError: If either check fails. An Arm L run that could have escalated is not Arm L, and
            the resulting trajectory would be a hybrid one wearing a local-only label.
    """
    if deadline_s != DEADLINE_DISABLED_S:
        raise SeamError(
            f"escalation-disabled arm requires deadline_s == DEADLINE_DISABLED_S "
            f"({DEADLINE_DISABLED_S}); got {deadline_s!r}"
        )
    if not math.isfinite(t_pred_s) or t_pred_s < 0:
        raise SeamError(f"t_pred_s must be a finite non-negative number; got {t_pred_s!r}")
    if t_pred_s * DEADLINE_DISABLED_MIN_HEADROOM > deadline_s:
        raise SeamError(
            f"t_pred={t_pred_s}s is within {DEADLINE_DISABLED_MIN_HEADROOM:g}x of the "
            f"escalation-disabled sentinel {deadline_s}s. The sentinel is no longer "
            f"distinguishable from a loose real deadline, so this run cannot be labelled "
            f"escalation-disabled."
        )


# ==================================================================================================
# Isolation invariant
# ==================================================================================================

#: Everything that must be byte-identical between the two arms. Throughput is excluded because it
#: IS the variable under test; target and label are excluded because they name the arm.
_ISOLATED_KEYS: tuple[str, ...] = (
    "task_ids",
    "seed",
    "system_prompt_sha256",
    "tool_specs_sha256",
    "task_list_sha256",
    "n_out_pred_tokens",
    "deadline_s",
    "max_steps",
    "max_tokens",
    "temperature",
    "cloud_model_id",
    "local_model_ref",
    # Confinement mechanism must be SYMMETRIC across arms. Process affinity and
    # SCHEDULING_CORE_TYPE differ in how the TBB pool is built and how threads are placed, so
    # mixing them would add a second difference between the arms and conflate silicon with
    # confinement method. Measured on this platform: see derived/mslice/affinity_matrix.json.
    "confinement_mechanism",
    # AM-024: reasoning mode is declared per arm and matched on both the local and cloud sides.
    # One arm inheriting a default the other does not share is a cross-boundary confound.
    "reasoning_mode",
)


@dataclass(slots=True)
class ArmConfig:
    """The inputs to one arm of the comparison, for isolation checking."""

    target: str
    throughput: ThroughputModel
    fields: dict[str, Any] = field(default_factory=dict)


def assert_identical_deadline_grid(grids: dict[str, list[float]]) -> None:
    """Refuse per-target quantile deadline grids (AM-024 / blueprint §7.4).

    Deadlines must be identical across targets within a reasoning arm. Setting them at each
    target's own quantiles gives prettier coverage and destroys the rescaling test.
    """
    if len(grids) < 2:
        return
    items = list(grids.items())
    ref_target, ref_grid = items[0]
    for target, grid in items[1:]:
        if list(grid) != list(ref_grid):
            raise SeamError(
                f"per-target deadline grids are refused: {ref_target}={list(ref_grid)!r} vs "
                f"{target}={list(grid)!r}. Within a reasoning arm the absolute D values must be "
                f"identical across targets so the rescaling prediction is testable (AM-024)."
            )


def assert_isolated(arm_a: ArmConfig, arm_b: ArmConfig) -> None:
    """Fail unless the two arms differ **only** in measured throughput.

    AM-022 item 2. If affinity leaks or a config drifts, both arms silently become the same
    experiment and the resulting null is uninterpretable - but looks exactly like a real one. This
    is the check that makes that failure mode loud.
    """
    # Equal targets are legitimate: the A/A negative control deliberately runs one target twice
    # under two labels. Every other field must still match, which is what this checks.
    differences: list[str] = []
    for key in _ISOLATED_KEYS:
        if key not in arm_a.fields and key not in arm_b.fields:
            differences.append(f"{key}: absent from both arms (cannot verify isolation)")
            continue
        a_value = arm_a.fields.get(key)
        b_value = arm_b.fields.get(key)
        if a_value != b_value:
            differences.append(
                f"{key}: {a_value!r} (arm {arm_a.target}) != {b_value!r} (arm {arm_b.target})"
            )

    if differences:
        raise IsolationViolationError(
            f"arms {arm_a.target!r} and {arm_b.target!r} differ in something other than measured "
            f"throughput, so any difference between them is not attributable to silicon "
            f"(AM-022): " + "; ".join(differences)
        )
