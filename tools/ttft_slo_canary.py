"""Drift canary for tools/run_c1_ceiling.py --criterion ttft_slo.

Reimplements docs/CANARY_PROTOCOL.md for the Python TTFT-SLO worker.
Fixed cell matches the matrix runner defaults (gpu_only_f16, n_cached=4000,
delta=400, RESIDENT) on the session's pinned model.

Interval N (INF-1b) is the tighter of two bounds:
  N = min(floor(onset_s / mean_probe_wall_s),
          floor(planned_probe_count / (C + 1)))
with ONSET_S = 657 from session 7f569929 (protocol). The budget bound ensures
C calibration canaries plus one armed check can fire inside the planned probe
count (c647f0c7 failure mode: onset-only N=66 against a 39-probe run).

A trip raises CanaryDriftAbort. It must not return a soft status the caller
can ignore (see tools/test_canary_drift_abort_enforcement.ps1 / Invoke-DriftCanary).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.canary_power_gate import (
    assert_no_ac_transition,
    capture_canary_power_snapshot,
)

# Shortest observed degradation onset (docs/CANARY_PROTOCOL.md / 7f569929).
ONSET_S = 657.0
CALIBRATION_C = 3
# Current-rule pool: pre-WARM_KV_AMEND_2 sessions had no warm-up discard, so each
# session's first canary is excluded. The 0.05 floor is replaced.
# Series: derived/c2_ttft/analysis/canary_threshold_pool.json
HEALTHY_FLOOR_T1 = 0.07709453214145699
HEALTHY_FLOOR_T2 = 0.1642942216508742
HEALTHY_FLOOR_P95_T1 = 0.062365838022024195
HEALTHY_FLOOR_P95_T2 = 0.10864235515205196
PREVIOUS_HEALTHY_FLOOR_T1 = 0.07709453214145699
PREVIOUS_HEALTHY_FLOOR_T2 = 0.5655217613849336
HEALTHY_FLOOR_ANALYSIS = "derived/c2_ttft/analysis/canary_threshold_pool.json"
POOL_RULE = (
    "Current calibration discards one warm-up canary, then calibrates on the next three. "
    "Sessions before WARM_KV_AMEND_2 had no warm-up discard, so each of those sessions "
    "contributes every canary except its first. The current-rule pool is the derivation."
)
WSH_POLICY = (
    "Kill WorkloadsSessionHost and refuse if it remains. Kill again in every cell preamble; "
    "it respawns about 4 min after a kill. Record instance_count, pids, WS_MB, and CPU_s "
    "at every gate and every canary. A failed kill records the exception text in the refusal. "
    "Record-only would break parity with sealed runs measured with the host cleared."
)
HEALTHY_FLOOR_SESSIONS = (
    "322b2f86-9571-46ae-be6a-ba1cec44e018",
    "69234ed5-a716-4274-b851-33509bdb9972",
    "7f232f86-2525-481b-a000-c6b8491bc224",
    "a427233b-df1c-4c4a-87b9-c368bb270b31",
    "c2246b1f-c588-4998-838a-5507da87e9ee",
    "d3dcbd3b-5107-4b5e-a0dd-402f99f6f90c",
    "fea55e0c-b9f8-4ba9-b5ea-558401910d74",
    "e9264a0a-d2f0-46dd-99af-8e4d9e36ad44",
)
CANARY_ARM = "gpu_only_f16"
CANARY_N_CACHED = 4000
CANARY_DELTA = 400
CANARY_MODE = "RESIDENT"
FAIL_STATUS = "FAIL_CANARY_DRIFT"
UNARMED_REFUSE = "REFUSED_UNARMED_CANARY"


def threshold_derivation() -> dict[str, Any]:
    """Pooled healthy-session floors recorded on every canary plan."""
    return {
        "formula": "threshold = max(2 * early_max_rel, healthy_floor)",
        "healthy_floor_t1": HEALTHY_FLOOR_T1,
        "healthy_floor_t2": HEALTHY_FLOOR_T2,
        "pooled_max_rel_t1": HEALTHY_FLOOR_T1,
        "pooled_max_rel_t2": HEALTHY_FLOOR_T2,
        "pooled_p95_rel_t1": HEALTHY_FLOOR_P95_T1,
        "pooled_p95_rel_t2": HEALTHY_FLOOR_P95_T2,
        "p95_method": "statistics.quantiles(data, n=100, method=inclusive)[94]",
        "pool_rule": POOL_RULE,
        "previous_healthy_floor_t1": PREVIOUS_HEALTHY_FLOOR_T1,
        "previous_healthy_floor_t2": PREVIOUS_HEALTHY_FLOOR_T2,
        "n_canaries_current_rule": 31,
        "replaced_floor": 0.05,
        "cell": "gpu_only_f16 RESIDENT n_cached=4000 delta=400",
        "host": "<xps-hostname>",
        "platform_id": "aipc-c1",
        "sessions": list(HEALTHY_FLOOR_SESSIONS),
        "analysis": HEALTHY_FLOOR_ANALYSIS,
        "rationale": (
            "healthy_floor is the largest relative drift, abs(t - session_ref) / session_ref, "
            "in the current-rule pool: sealed guarded XPS sessions on this cell that finished "
            "without a trip, excluding each pre-WARM_KV_AMEND_2 session's first canary. "
            "No accepted session exceeded it, so a larger relative drift is evidence of degradation. "
            "The pool rule is not chosen by whether a later check would trip."
        ),
    }


def drift_threshold(early_max_rel: float, healthy_floor: float) -> float:
    return max(2.0 * float(early_max_rel), float(healthy_floor))


def _active_floors(
    gate: dict[str, Any],
    *,
    rel_drift_floor: float | None,
    rel_drift_floor_t1: float | None,
    rel_drift_floor_t2: float | None,
) -> tuple[float, float]:
    """A single override applies to both turns. Otherwise use the gate floors."""
    if rel_drift_floor is not None:
        floor = float(rel_drift_floor)
        return floor, floor
    t1 = rel_drift_floor_t1
    t2 = rel_drift_floor_t2
    if t1 is None:
        t1 = gate.get("rel_drift_floor_t1", HEALTHY_FLOOR_T1)
    if t2 is None:
        t2 = gate.get("rel_drift_floor_t2", HEALTHY_FLOOR_T2)
    return float(t1), float(t2)


class CanaryDriftAbort(Exception):
    """Session must stop; status = FAIL_CANARY_DRIFT."""

    def __init__(self, detail: str, *, canary_record: dict[str, Any] | None = None):
        super().__init__(detail)
        self.detail = detail
        self.canary_record = canary_record or {}


class CanaryBudgetRefuse(Exception):
    """Session must not start; planned probe budget cannot arm the guard."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class CanaryUnarmedSealRefuse(Exception):
    """Seal/finalize refused: gate never armed and AllowUnguarded was not set."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _rel_drift(value: float, ref: float) -> float | None:
    if ref == 0.0:
        return None
    return abs(float(value) - float(ref)) / abs(float(ref))


def estimate_bisect_planned_probes(
    *,
    n_arms: int,
    low: int,
    high: int,
    resolution: int,
    repeats: int,
) -> int:
    """Upper bound on probes_log rows (one per repeat) for a C-2-style bisect.

    Each arm: low + high + up to ceil(log2((high-low)/resolution)) mid probes,
    each multiplied by repeats. Matches after_probe accounting in run_c1_ceiling.
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    if repeats < 1 or n_arms < 1:
        raise ValueError("n_arms and repeats must be >= 1")
    ratio = max(1.0, float(high - low) / float(resolution))
    mid_probes = int(math.ceil(math.log2(ratio)))
    points_per_arm = 2 + mid_probes
    return int(n_arms) * points_per_arm * int(repeats)


def _n_budget(planned: int, c: int, calibrated_before_probes: bool) -> int:
    """floor(planned / (C + 1)); at least 1 when calibration runs before the first probe.

    CC fix 2026-10-23e: with the C calibration canaries run up front, only one armed
    check has to fit inside the planned probes, so a run of 1-3 probes keeps N = 1.
    """
    n = planned // (c + 1)
    if calibrated_before_probes and planned >= 1:
        return max(1, n)
    return n


def assert_canary_budget_fits(
    planned_probe_count: int,
    *,
    calibration_c: int = CALIBRATION_C,
    calibrated_before_probes: bool = False,
) -> dict[str, Any]:
    """Refuse before start if C calibration + one armed check cannot fit.

    Requires floor(planned / (C + 1)) >= 1 so the budget-bound N is at least 1.
    With calibrated_before_probes (CC fix 2026-10-23e) the C canaries run before
    any probe, so the budget needs only one armed check: planned >= 1.
    """
    planned = int(planned_probe_count)
    c = int(calibration_c)
    if c < 2:
        raise ValueError("calibration_c must be >= 2")
    n_budget = _n_budget(planned, c, calibrated_before_probes)
    if n_budget < 1:
        raise CanaryBudgetRefuse(
            f"canary budget cannot fit C={c} calibration canaries + 1 armed check: "
            f"planned_probe_count={planned} < C+1={c + 1} "
            f"(floor(planned/(C+1))={n_budget}). Do not start unguarded."
        )
    out: dict[str, Any] = {
        "planned_probe_count": planned,
        "calibration_c": c,
        "n_budget": n_budget,
        "min_canaries_required": c + 1,
        "ok": True,
    }
    if calibrated_before_probes:
        out["budget_rule"] = "calibrated_before_probes: C canaries up front + 1 armed check"
    return out


def derive_canary_every_n(
    probe_wall_s: list[float],
    *,
    onset_s: float = ONSET_S,
    planned_probe_count: int | None = None,
    calibration_c: int = CALIBRATION_C,
    calibrated_before_probes: bool = False,
) -> dict[str, Any]:
    """N = min(onset bound, budget bound). Records both candidates and which bound."""
    c = int(calibration_c)
    n_budget: int | None = None
    if planned_probe_count is not None:
        n_budget = _n_budget(int(planned_probe_count), c, calibrated_before_probes)

    walls = [float(w) for w in probe_wall_s if w is not None and float(w) > 0]
    mean_w: float | None = None
    n_onset: int | None = None
    if walls:
        mean_w = float(statistics.mean(walls))
        n_onset = max(1, int(onset_s // mean_w))

    if planned_probe_count is not None and (n_budget is None or n_budget < 1):
        return {
            "n": None,
            "n_onset": n_onset,
            "n_budget": n_budget if n_budget is not None else 0,
            "binding_bound": None,
            "mean_probe_wall_s": mean_w,
            "onset_s": onset_s,
            "planned_probe_count": int(planned_probe_count),
            "calibration_c": c,
            "n_probes_in_mean": len(walls),
            "derivable": False,
            "refuse": True,
            "note": (
                f"budget cannot fit C={c}+1 canaries "
                f"(planned={planned_probe_count}, n_budget={n_budget})"
            ),
        }

    if n_onset is None and n_budget is None:
        return {
            "n": None,
            "n_onset": None,
            "n_budget": None,
            "binding_bound": None,
            "mean_probe_wall_s": None,
            "onset_s": onset_s,
            "planned_probe_count": planned_probe_count,
            "calibration_c": c,
            "derivable": False,
            "note": "waiting for planned_probe_count and/or at least one positive probe wall_s",
        }

    if n_onset is not None and n_budget is not None:
        n = min(n_onset, n_budget)
        binding = "onset" if n_onset <= n_budget else "budget"
    elif n_onset is not None:
        n = n_onset
        binding = "onset"
    else:
        n = int(n_budget)  # type: ignore[arg-type]
        binding = "budget"

    return {
        "n": n,
        "n_onset": n_onset,
        "n_budget": n_budget,
        "binding_bound": binding,
        "mean_probe_wall_s": mean_w,
        "onset_s": onset_s,
        "planned_probe_count": planned_probe_count,
        "calibration_c": c,
        "n_probes_in_mean": len(walls),
        "derivable": True,
        "refuse": False,
        "derivation": (
            f"N=min(floor(onset_s/mean_probe_wall_s), floor(planned/(C+1)))="
            f"min({n_onset if n_onset is not None else 'None'},"
            f"{n_budget if n_budget is not None else 'None'})={n} "
            f"(binding={binding}); onset_s={onset_s}; "
            f"mean_probe_wall_s="
            f"{f'{mean_w:.4f}' if mean_w is not None else 'None'} "
            f"from THIS run ({len(walls)} probes); "
            f"planned_probe_count={planned_probe_count}; C={c}."
        ),
    }


def assert_seal_requires_armed_or_unguarded(
    *,
    armed: bool,
    allow_unguarded: bool,
    unguarded_already: bool = False,
) -> dict[str, Any]:
    """Refuse seal/finalize when armed is false unless AllowUnguarded was set."""
    if armed:
        return {"ok": True, "UNGUARDED": False, "armed": True}
    if allow_unguarded or unguarded_already:
        return {
            "ok": True,
            "UNGUARDED": True,
            "armed": False,
            "note": "armed==false; AllowUnguarded. Summary and seal must record UNGUARDED",
        }
    raise CanaryUnarmedSealRefuse(
        "canary gate armed==false; refuse to seal without -AllowUnguarded / "
        "--allow-unguarded (would write UNGUARDED into summary and seal)"
    )


def update_canary_drift_bookkeeping(
    *,
    rec: dict[str, Any],
    gate: dict[str, Any],
    prior_ok_canaries: list[dict[str, Any]],
    calibration_c: int = CALIBRATION_C,
    rel_drift_floor: float | None = None,
    rel_drift_floor_t1: float | None = None,
    rel_drift_floor_t2: float | None = None,
) -> dict[str, Any]:
    """Python port of tools/SeamPsCommon.ps1 Update-CanaryDriftBookkeeping."""
    if calibration_c < 2:
        raise ValueError("calibration_c must be >= 2")
    was_complete = bool(gate.get("calibration_complete"))
    rel_t1 = None
    rel_t2 = None
    tripped = False
    trip_detail = None

    t1 = rec.get("turn1_prefill_s")
    t2 = rec.get("turn2_prefill_s")
    ok = rec.get("classification") == "OK" and t1 is not None and t2 is not None

    if rec.get("warmup"):
        rec["discarded_from_calibration"] = True
        rec["counts_toward_calibration"] = False
        rec["drift_tripped"] = False
        rec["trip_detail"] = None
        rec["rel_drift_t1"] = None
        rec["rel_drift_t2"] = None
        rec["gate_armed"] = bool(gate.get("armed"))
        rec["threshold_t1"] = gate.get("threshold_t1")
        rec["threshold_t2"] = gate.get("threshold_t2")
        return {
            "rec": rec,
            "tripped": False,
            "trip_detail": None,
            "just_armed": False,
            "derivation_applied": gate.get("derivation_applied"),
        }
    rec["discarded_from_calibration"] = False

    if ok:
        if not gate.get("calibration_complete"):
            ok_cal = [
                c
                for c in prior_ok_canaries
                if not c.get("warmup")
                and c.get("classification") == "OK"
                and c.get("turn1_prefill_s") is not None
                and c.get("turn2_prefill_s") is not None
            ] + [rec]
            if len(ok_cal) >= calibration_c:
                t1s = [float(c["turn1_prefill_s"]) for c in ok_cal]
                t2s = [float(c["turn2_prefill_s"]) for c in ok_cal]
                ref1 = float(statistics.median(t1s))
                ref2 = float(statistics.median(t2s))
                early1 = [_rel_drift(v, ref1) or 0.0 for v in t1s]
                early2 = [_rel_drift(v, ref2) or 0.0 for v in t2s]
                em1 = max(early1)
                em2 = max(early2)
                floor1, floor2 = _active_floors(
                    gate,
                    rel_drift_floor=rel_drift_floor,
                    rel_drift_floor_t1=rel_drift_floor_t1,
                    rel_drift_floor_t2=rel_drift_floor_t2,
                )
                th1 = drift_threshold(em1, floor1)
                th2 = drift_threshold(em2, floor2)
                gate["armed"] = True
                gate["calibration_complete"] = True
                gate["ref_turn1_prefill_s"] = ref1
                gate["ref_turn2_prefill_s"] = ref2
                gate["early_max_rel_t1"] = em1
                gate["early_max_rel_t2"] = em2
                gate["threshold_t1"] = th1
                gate["threshold_t2"] = th2
                gate["calibration_turn1_prefill_s"] = t1s
                gate["calibration_turn2_prefill_s"] = t2s
                gate["derivation_applied"] = (
                    f"calib_n={calibration_c}; ref_t1={ref1:.6f} ref_t2={ref2:.6f}; "
                    f"early_max_rel_t1={em1:.6f} early_max_rel_t2={em2:.6f}; "
                    f"threshold_t1=max(2*early_max_t1,healthy_floor_t1={floor1})={th1:.6f}; "
                    f"threshold_t2=max(2*early_max_t2,healthy_floor_t2={floor2})={th2:.6f}"
                )
        else:
            rel_t1 = _rel_drift(float(t1), float(gate["ref_turn1_prefill_s"]))
            rel_t2 = _rel_drift(float(t2), float(gate["ref_turn2_prefill_s"]))
            if rel_t1 is not None and rel_t1 > float(gate["threshold_t1"]):
                tripped = True
                trip_detail = (
                    f"turn1 rel_drift={rel_t1:.6f} > threshold_t1={gate['threshold_t1']:.6f} "
                    f"(t1={t1} ref={gate['ref_turn1_prefill_s']})"
                )
            elif rel_t2 is not None and rel_t2 > float(gate["threshold_t2"]):
                tripped = True
                trip_detail = (
                    f"turn2 rel_drift={rel_t2:.6f} > threshold_t2={gate['threshold_t2']:.6f} "
                    f"(t2={t2} ref={gate['ref_turn2_prefill_s']})"
                )
    elif gate.get("calibration_complete"):
        tripped = True
        trip_detail = (
            f"canary classification={rec.get('classification')!r} "
            "(no usable turn1/turn2 after gate armed)"
        )

    rec["rel_drift_t1"] = rel_t1
    rec["rel_drift_t2"] = rel_t2
    rec["gate_armed"] = bool(gate.get("armed"))
    rec["threshold_t1"] = gate.get("threshold_t1")
    rec["threshold_t2"] = gate.get("threshold_t2")
    rec["drift_tripped"] = tripped
    rec["trip_detail"] = trip_detail

    return {
        "rec": rec,
        "tripped": tripped,
        "trip_detail": trip_detail,
        "just_armed": (not was_complete) and bool(gate.get("calibration_complete")),
        "derivation_applied": gate.get("derivation_applied"),
    }


def new_canary_gate(
    *,
    calibration_c: int = CALIBRATION_C,
    rel_drift_floor: float | None = None,
    rel_drift_floor_t1: float | None = None,
    rel_drift_floor_t2: float | None = None,
) -> dict[str, Any]:
    if rel_drift_floor is not None:
        if rel_drift_floor_t1 is None:
            rel_drift_floor_t1 = rel_drift_floor
        if rel_drift_floor_t2 is None:
            rel_drift_floor_t2 = rel_drift_floor
    if rel_drift_floor_t1 is None:
        rel_drift_floor_t1 = HEALTHY_FLOOR_T1
    if rel_drift_floor_t2 is None:
        rel_drift_floor_t2 = HEALTHY_FLOOR_T2
    return {
        "armed": False,
        "calibration_complete": False,
        "calibration_c": calibration_c,
        "rel_drift_floor_t1": rel_drift_floor_t1,
        "rel_drift_floor_t2": rel_drift_floor_t2,
        "threshold_derivation": threshold_derivation(),
        "ref_turn1_prefill_s": None,
        "ref_turn2_prefill_s": None,
        "early_max_rel_t1": None,
        "early_max_rel_t2": None,
        "threshold_t1": None,
        "threshold_t2": None,
        "derivation_applied": None,
        "fixed_cell": {
            "arm": CANARY_ARM,
            "n_cached": CANARY_N_CACHED,
            "delta": CANARY_DELTA,
            "mode": CANARY_MODE,
        },
        "onset_s": ONSET_S,
        "protocol": "docs/CANARY_PROTOCOL.md",
    }


@dataclass
class TtftSloCanaryGuard:
    """Interleaves fixed delta-prefill canaries into a ttft_slo ceiling run."""

    root: Path
    model_spec: Path
    work_dir: Path
    plan_path: Path | None = None
    planned_probe_count: int = 0
    allow_unguarded: bool = False
    discard_warmup: bool = False
    # NPU amendment 2026-10-08: run the C calibration canaries before any probe.
    arm_before_probes: bool = False
    # CC fix 2026-10-23e: with arm_before_probes, budget for one armed check only.
    up_front_budget: bool = False
    enforce_probe_budget: bool = True
    calibration_c: int = CALIBRATION_C
    rel_drift_floor: float | None = None
    rel_drift_floor_t1: float | None = None
    rel_drift_floor_t2: float | None = None
    onset_s: float = ONSET_S
    gate: dict[str, Any] = field(default_factory=new_canary_gate)
    canaries: list[dict[str, Any]] = field(default_factory=list)
    n_every: int | None = None
    n_derivation: dict[str, Any] = field(default_factory=dict)
    probes_since_canary: int = 0
    opening_done: bool = False
    budget_preflight: dict[str, Any] = field(default_factory=dict)
    last_power_snapshot: dict[str, Any] | None = None
    # NPU amendment 2026-10-14: optional per-canary host telemetry (record only).
    telemetry: Any = None
    # NPU amendment 2026-10-14b: run each canary cell in a fresh child process.
    child_process: bool = False
    child_timeout_s: float = 180.0
    # Callable (result_path, model_spec) -> argv; None = tools.canary_child.
    child_command: Any = None
    # 2026-10-14b transition record: also run in-process (record only) at the
    # record-only point and the first post-arm point.
    inprocess_transition: bool = False
    _transition_post_done: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.up_front_budget and not self.arm_before_probes:
            raise ValueError("up_front_budget needs arm_before_probes")
        self.gate = new_canary_gate(
            calibration_c=self.calibration_c,
            rel_drift_floor=self.rel_drift_floor,
            rel_drift_floor_t1=self.rel_drift_floor_t1,
            rel_drift_floor_t2=self.rel_drift_floor_t2,
        )
        self.gate["onset_s"] = self.onset_s
        if self.enforce_probe_budget:
            self.budget_preflight = assert_canary_budget_fits(
                int(self.planned_probe_count),
                calibration_c=self.calibration_c,
                calibrated_before_probes=self.up_front_budget,
            )
        else:
            self.budget_preflight = {
                "ok": True,
                "skipped": True,
                "reason": "calibrate-only rehearsal measures no probes",
                "planned_probe_count": int(self.planned_probe_count),
                "calibration_c": int(self.calibration_c),
            }
        self.refresh_interval([])

    def plan_fragment(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "criterion_path": "ttft_slo",
            "fixed_cell": self.gate["fixed_cell"],
            "model_spec": str(self.model_spec),
            "calibration_c": self.calibration_c,
            "rel_drift_floor_t1": self.gate["rel_drift_floor_t1"],
            "rel_drift_floor_t2": self.gate["rel_drift_floor_t2"],
            "threshold_derivation": self.gate["threshold_derivation"],
            "onset_s": self.onset_s,
            "planned_probe_count": self.planned_probe_count,
            "budget_preflight": self.budget_preflight,
            "allow_unguarded": bool(self.allow_unguarded),
            "canary_every_n": self.n_every,
            **({"schedule": "arm_before_probes"} if self.arm_before_probes else {}),
            **({"budget_rule": "up_front (CC fix 2026-10-23e)"} if self.up_front_budget else {}),
            "n_derivation": self.n_derivation,
            "canary_gate": self.gate,
            "n_canaries": len(self.canaries),
            "workloads_session_host_policy": WSH_POLICY,
            "workloads_session_host_at_canaries": [
                {
                    "canary_index": c.get("canary_index"),
                    "workloads_session_host": c.get("workloads_session_host"),
                }
                for c in self.canaries
            ],
            "abort_on_trip": FAIL_STATUS,
            "power_transition_refuse": (
                "INF-6: capture Win32_Battery BatteryStatus + EstimatedChargeRemaining "
                "at every canary; refuse on AC on_ac transition. "
                "No-battery hosts pass trivially."
            ),
            "enforcement": (
                "CanaryDriftAbort exception; session status FAIL_CANARY_DRIFT. "
                "Does not return a soft bool that a caller can ignore "
                "(Invoke-DriftCanary Write-Output bug class). "
                "INF-1b: refuse start if budget < C+1 canaries; refuse seal "
                "with armed==false unless AllowUnguarded writes UNGUARDED."
            ),
        }

    def refresh_interval(self, probe_wall_s: list[float]) -> None:
        if not self.enforce_probe_budget:
            self.n_derivation = {
                "n": None,
                "derivable": False,
                "refuse": False,
                "planned_probe_count": int(self.planned_probe_count),
                "calibration_c": int(self.calibration_c),
                "note": "calibrate-only rehearsal measures no probes",
            }
            return
        self.n_derivation = derive_canary_every_n(
            probe_wall_s,
            onset_s=self.onset_s,
            planned_probe_count=int(self.planned_probe_count),
            calibration_c=self.calibration_c,
            calibrated_before_probes=self.up_front_budget,
        )
        if self.n_derivation.get("refuse"):
            raise CanaryBudgetRefuse(self.n_derivation.get("note") or "canary budget refuse")
        if self.n_derivation.get("derivable"):
            self.n_every = int(self.n_derivation["n"])

    def finalize_or_refuse(self) -> dict[str, Any]:
        """Call before writing a sealable complete summary."""
        return assert_seal_requires_armed_or_unguarded(
            armed=bool(self.gate.get("armed")),
            allow_unguarded=bool(self.allow_unguarded),
        )

    def _run_cell(self) -> dict[str, Any]:
        from tools.canary_child import cell_inprocess, child_command, run_child

        self.work_dir.mkdir(parents=True, exist_ok=True)
        child: dict[str, Any] | None = None
        if self.child_process:
            out = self.work_dir / f"child.{len(self.canaries)}.json"
            build = self.child_command or child_command
            raw, child = run_child(
                list(build(out, self.model_spec)), out, timeout_s=float(self.child_timeout_s)
            )
        else:
            raw = cell_inprocess(self.model_spec)
        t1 = (raw.get("turn1") or {}).get("prefill_s")
        t2 = (raw.get("turn2") or {}).get("prefill_s")
        classification = raw.get("classification") or "OTHER"
        if classification == "OK" or (raw.get("execute_ok") and t1 is not None and t2 is not None):
            classification = "OK"
        rec = {
            "is_canary": True,
            "canary_index": len(self.canaries),
            "after_probe_count": None,
            "utc": _utc(),
            "arm_id": CANARY_ARM,
            "n_cached": CANARY_N_CACHED,
            "delta": CANARY_DELTA,
            "mode": CANARY_MODE,
            "classification": classification,
            "turn1_prefill_s": t1,
            "turn2_prefill_s": t2,
            "available_mb_start": (raw.get("environment_start") or {}).get("available_mb"),
            "raw_classification": raw.get("classification"),
            "execute_ok": raw.get("execute_ok"),
            "execute_error": raw.get("execute_error"),
        }
        if child is not None:
            rec["child"] = child
        out = self.work_dir / f"canary.{rec['canary_index']}.json"
        out.write_text(
            json.dumps({"record": rec, "cell": raw}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return rec

    def _inprocess_record(self, point: str) -> dict[str, Any]:
        """2026-10-14b transition record: the cell in-process, after the gating run."""
        from tools.canary_child import cell_inprocess

        try:
            raw = cell_inprocess(self.model_spec)
        except Exception as exc:
            return {"point": point, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "point": point,
            "classification": raw.get("classification"),
            "turn1_prefill_s": (raw.get("turn1") or {}).get("prefill_s"),
            "turn2_prefill_s": (raw.get("turn2") or {}).get("prefill_s"),
            "execute_error": raw.get("execute_error"),
            "gates": False,
        }

    def _write_canary_record(self, rec: dict[str, Any]) -> None:
        path = self.work_dir / f"canary.{rec['canary_index']}.json"
        if not path.is_file():
            return
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["record"] = rec
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _stamp_plan(self) -> None:
        if self.plan_path is None or not self.plan_path.is_file():
            return
        doc = json.loads(self.plan_path.read_text(encoding="utf-8"))
        doc["canary"] = self.plan_fragment()
        self.plan_path.write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def run_canary(self, *, after_probe_count: int, warmup: bool = False) -> dict[str, Any]:
        """Run one canary. Raises CanaryDriftAbort / CanaryPowerTransitionAbort."""
        print(
            f"[c2_canary] index={len(self.canaries)} after_probes={after_probe_count} "
            f"armed={self.gate.get('armed')} every_n={self.n_every} "
            f"bound={(self.n_derivation or {}).get('binding_bound')}",
            flush=True,
        )
        # INF-6: record BatteryStatus + charge; refuse on AC transition.
        power = capture_canary_power_snapshot()
        transition = assert_no_ac_transition(previous=self.last_power_snapshot, current=power)
        self.last_power_snapshot = power
        from seam.run_environment import snapshot_workloads_session_host

        wsh = snapshot_workloads_session_host()
        # Taken before the canary cell so its own child process is not counted.
        tele: dict[str, Any] | None = None
        if self.telemetry is not None:
            try:
                tele = self.telemetry()
            except Exception as exc:
                tele = {"error": f"{type(exc).__name__}: {exc}"}
        rec = self._run_cell()
        rec["power"] = power
        rec["power_transition"] = transition
        rec["workloads_session_host"] = wsh
        rec["after_probe_count"] = after_probe_count
        rec["warmup"] = bool(warmup)
        if tele is not None:
            rec["telemetry"] = tele
        point = None
        if self.inprocess_transition:
            if warmup:
                point = "record_only"
            elif after_probe_count >= 0 and not self._transition_post_done:
                point = "first_post_arm"
                self._transition_post_done = True
        if point is not None:
            rec["inprocess_transition"] = self._inprocess_record(point)
        prior_ok = [
            c
            for c in self.canaries
            if not c.get("warmup")
            and c.get("classification") == "OK"
            and c.get("turn1_prefill_s") is not None
            and c.get("turn2_prefill_s") is not None
        ]
        bk = update_canary_drift_bookkeeping(
            rec=rec,
            gate=self.gate,
            prior_ok_canaries=prior_ok,
            calibration_c=self.calibration_c,
            rel_drift_floor=self.rel_drift_floor,
        )
        rec = bk["rec"]
        self.canaries.append(rec)
        self._write_canary_record(rec)
        self._stamp_plan()
        self.probes_since_canary = 0
        if bk.get("just_armed"):
            print(f"[c2_canary] gate ARMED: {bk.get('derivation_applied')}", flush=True)
        if bk["tripped"] or rec.get("drift_tripped"):
            detail = bk.get("trip_detail") or rec.get("trip_detail") or "drift tripped"
            print(f"REFUSED -- {FAIL_STATUS}: {detail}", flush=True)
            raise CanaryDriftAbort(str(detail), canary_record=rec)
        return rec

    def opening(self) -> None:
        if self.opening_done:
            return
        if self.discard_warmup:
            self.run_canary(after_probe_count=-1, warmup=True)
        for _ in range(int(self.calibration_c) if self.arm_before_probes else 1):
            self.run_canary(after_probe_count=-1)
        self.opening_done = True

    def closing(self, probes_log: list[dict[str, Any]]) -> None:
        """One armed check after the last probe when probes ran since the last canary."""
        if self.arm_before_probes and probes_log and self.probes_since_canary > 0:
            self.run_canary(after_probe_count=len(probes_log) - 1)

    def after_probe(self, probes_log: list[dict[str, Any]]) -> None:
        """Call once after each successful probe append."""
        walls = [
            float(p["wall_s"])
            for p in probes_log
            if p.get("wall_s") is not None and float(p["wall_s"]) > 0
        ]
        self.refresh_interval(walls)
        self.probes_since_canary += 1
        if self.n_every is None:
            return
        if self.probes_since_canary >= self.n_every:
            self.run_canary(after_probe_count=len(probes_log) - 1)
