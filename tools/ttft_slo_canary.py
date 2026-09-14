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

A trip raises CanaryDriftAbort — it must not return a soft status the caller
can ignore (see tools/test_canary_drift_abort_enforcement.ps1 / Invoke-DriftCanary).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Shortest observed degradation onset (docs/CANARY_PROTOCOL.md / 7f569929).
ONSET_S = 657.0
CALIBRATION_C = 3
REL_DRIFT_FLOOR = 0.05
CANARY_ARM = "gpu_only_f16"
CANARY_N_CACHED = 4000
CANARY_DELTA = 400
CANARY_MODE = "RESIDENT"
FAIL_STATUS = "FAIL_CANARY_DRIFT"
UNARMED_REFUSE = "REFUSED_UNARMED_CANARY"


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


def assert_canary_budget_fits(
    planned_probe_count: int,
    *,
    calibration_c: int = CALIBRATION_C,
) -> dict[str, Any]:
    """Refuse before start if C calibration + one armed check cannot fit.

    Requires floor(planned / (C + 1)) >= 1 so the budget-bound N is at least 1.
    """
    planned = int(planned_probe_count)
    c = int(calibration_c)
    if c < 2:
        raise ValueError("calibration_c must be >= 2")
    n_budget = planned // (c + 1)
    if n_budget < 1:
        raise CanaryBudgetRefuse(
            f"canary budget cannot fit C={c} calibration canaries + 1 armed check: "
            f"planned_probe_count={planned} < C+1={c + 1} "
            f"(floor(planned/(C+1))={n_budget}). Do not start unguarded."
        )
    return {
        "planned_probe_count": planned,
        "calibration_c": c,
        "n_budget": n_budget,
        "min_canaries_required": c + 1,
        "ok": True,
    }


def derive_canary_every_n(
    probe_wall_s: list[float],
    *,
    onset_s: float = ONSET_S,
    planned_probe_count: int | None = None,
    calibration_c: int = CALIBRATION_C,
) -> dict[str, Any]:
    """N = min(onset bound, budget bound). Records both candidates and which bound."""
    c = int(calibration_c)
    n_budget: int | None = None
    if planned_probe_count is not None:
        n_budget = int(planned_probe_count) // (c + 1)

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
            "note": "armed==false; AllowUnguarded — summary/seal must record UNGUARDED",
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
    rel_drift_floor: float = REL_DRIFT_FLOOR,
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

    if ok:
        if not gate.get("calibration_complete"):
            ok_cal = [
                c
                for c in prior_ok_canaries
                if c.get("classification") == "OK"
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
                th1 = max(2.0 * em1, float(rel_drift_floor))
                th2 = max(2.0 * em2, float(rel_drift_floor))
                gate["armed"] = True
                gate["calibration_complete"] = True
                gate["ref_turn1_prefill_s"] = ref1
                gate["ref_turn2_prefill_s"] = ref2
                gate["early_max_rel_t1"] = em1
                gate["early_max_rel_t2"] = em2
                gate["threshold_t1"] = th1
                gate["threshold_t2"] = th2
                gate["derivation_applied"] = (
                    f"calib_n={calibration_c}; ref_t1={ref1:.6f} ref_t2={ref2:.6f}; "
                    f"early_max_rel_t1={em1:.6f} early_max_rel_t2={em2:.6f}; "
                    f"threshold_t1=max(2*early_max_t1,floor={rel_drift_floor})={th1:.6f}; "
                    f"threshold_t2={th2:.6f}"
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
    rel_drift_floor: float = REL_DRIFT_FLOOR,
) -> dict[str, Any]:
    return {
        "armed": False,
        "calibration_complete": False,
        "calibration_c": calibration_c,
        "rel_drift_floor": rel_drift_floor,
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
    calibration_c: int = CALIBRATION_C
    rel_drift_floor: float = REL_DRIFT_FLOOR
    onset_s: float = ONSET_S
    gate: dict[str, Any] = field(default_factory=new_canary_gate)
    canaries: list[dict[str, Any]] = field(default_factory=list)
    n_every: int | None = None
    n_derivation: dict[str, Any] = field(default_factory=dict)
    probes_since_canary: int = 0
    opening_done: bool = False
    budget_preflight: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.gate = new_canary_gate(
            calibration_c=self.calibration_c,
            rel_drift_floor=self.rel_drift_floor,
        )
        self.gate["onset_s"] = self.onset_s
        self.budget_preflight = assert_canary_budget_fits(
            int(self.planned_probe_count),
            calibration_c=self.calibration_c,
        )
        self.refresh_interval([])

    def plan_fragment(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "criterion_path": "ttft_slo",
            "fixed_cell": self.gate["fixed_cell"],
            "model_spec": str(self.model_spec),
            "calibration_c": self.calibration_c,
            "rel_drift_floor": self.rel_drift_floor,
            "onset_s": self.onset_s,
            "planned_probe_count": self.planned_probe_count,
            "budget_preflight": self.budget_preflight,
            "allow_unguarded": bool(self.allow_unguarded),
            "canary_every_n": self.n_every,
            "n_derivation": self.n_derivation,
            "canary_gate": self.gate,
            "n_canaries": len(self.canaries),
            "abort_on_trip": FAIL_STATUS,
            "enforcement": (
                "CanaryDriftAbort exception; session status FAIL_CANARY_DRIFT. "
                "Does not return a soft bool that a caller can ignore "
                "(Invoke-DriftCanary Write-Output bug class). "
                "INF-1b: refuse start if budget < C+1 canaries; refuse seal "
                "with armed==false unless AllowUnguarded writes UNGUARDED."
            ),
        }

    def refresh_interval(self, probe_wall_s: list[float]) -> None:
        self.n_derivation = derive_canary_every_n(
            probe_wall_s,
            onset_s=self.onset_s,
            planned_probe_count=int(self.planned_probe_count),
            calibration_c=self.calibration_c,
        )
        if self.n_derivation.get("refuse"):
            raise CanaryBudgetRefuse(
                self.n_derivation.get("note") or "canary budget refuse"
            )
        if self.n_derivation.get("derivable"):
            self.n_every = int(self.n_derivation["n"])

    def finalize_or_refuse(self) -> dict[str, Any]:
        """Call before writing a sealable complete summary."""
        return assert_seal_requires_armed_or_unguarded(
            armed=bool(self.gate.get("armed")),
            allow_unguarded=bool(self.allow_unguarded),
        )

    def _run_cell(self) -> dict[str, Any]:
        from tools.smoke_delta_prefill import run_cell

        self.work_dir.mkdir(parents=True, exist_ok=True)
        raw = run_cell(
            "ttft_slo_canary",
            arm_id=CANARY_ARM,
            mode=CANARY_MODE,
            n_cached=CANARY_N_CACHED,
            delta=CANARY_DELTA,
            model_spec=self.model_spec,
            pipeline_type="llm",
        )
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
        out = self.work_dir / f"canary.{rec['canary_index']}.json"
        out.write_text(
            __import__("json").dumps({"record": rec, "cell": raw}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return rec

    def run_canary(self, *, after_probe_count: int) -> dict[str, Any]:
        """Run one canary. Raises CanaryDriftAbort on trip — never soft-fails."""
        print(
            f"[c2_canary] index={len(self.canaries)} after_probes={after_probe_count} "
            f"armed={self.gate.get('armed')} every_n={self.n_every} "
            f"bound={(self.n_derivation or {}).get('binding_bound')}",
            flush=True,
        )
        rec = self._run_cell()
        rec["after_probe_count"] = after_probe_count
        prior_ok = [
            c
            for c in self.canaries
            if c.get("classification") == "OK"
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
        self.run_canary(after_probe_count=-1)
        self.opening_done = True

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
