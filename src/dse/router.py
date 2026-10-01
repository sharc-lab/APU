"""Our own router: envelope feasibility + effective context (analysis/envelope_model.py), predicted latency with
the real per-machine co-runner term (B5's 2026-10-01 refit), a simple difficulty estimate, live remaining-budget
state from CloudClient, and a hard rule against ever routing into a silent-failure configuration.

Every RouteDecision carries a human-readable `reason` string -- this is a design requirement (B6b), not an
afterthought: a routing decision that cannot explain itself in one sentence is not auditable.

Silent-failure rule (reused, not reinvented)
---------------------------------------------
"Never route into a silent-failure configuration" is checked two ways, both reusing detection logic/thresholds
that already exist in this repo rather than inventing a new one:
  1. analysis/envelope_model.predict_failure_silence(machine, runtime_policy) -- reuses
     analysis/make_failure_map.EVIDENCE directly. Any outcome in SILENT_FAILURE_OUTCOMES below blocks local
     routing for that (machine, runtime_policy) cell.
  2. analysis/envelope_model.predict_quality_regime(...) -- when the predicted regime is "truncation_cliff" and
     the predicted score falls below QUALITY_FLOOR, that is itself a silent failure in the harness's own sense
     (harness/t2s_r2_session_growth.py's `silent_truncation` classification: "processed_tokens < sent_tokens"
     with no error and no context-shift log line -- the server returns 200 with truncated output, not an error).
     The router treats "predicted truncation-cliff regime below the quality floor" as the same class of event
     and blocks local routing for it too, using the exact R1b-measured score table
     (R1B_BASELINE_SCORE_BY_RATIO) envelope_model already carries, not a new threshold invented here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import envelope_model as em  # noqa: E402
from src.cloud.client import CloudClient, CLOUD_MODELS, estimate_cost_usd  # noqa: E402

# make_failure_map's own outcome vocabulary (docs/FAILURE_MAP.md / analysis/make_failure_map.py EVIDENCE values).
# Any of these for a (machine, runtime_policy) cell means: do not route a real task into that local configuration.
SILENT_FAILURE_OUTCOMES = {"SILENT_SPILL", "SILENT_SPILL, then CRASH one step higher", "CRASH", "HANG",
                            "HARD_FAIL"}  # HARD_FAIL is loud, not silent, but still means "will not run" locally

DEFAULT_QUALITY_FLOOR = 0.5  # predicted_score below this in a truncation_cliff regime is treated as unacceptable


@dataclass
class RouteDecision:
    task_id: str
    target: str  # "local" | "cloud_cheap" | "cloud_strong" | "compact_then_local" | "refuse"
    reason: str
    prediction: dict[str, Any] = field(default_factory=dict)
    blocked_silent_failure: bool = False
    estimated_cost_usd: float = 0.0


def estimate_difficulty(prompt: str) -> float:
    """Simple difficulty proxy in [0, 1]: word count relative to a 1500-word cap, clamped.

    This is a deliberately simple heuristic, not a calibrated difficulty model -- no labeled difficulty dataset
    exists anywhere in this repo to fit one against. It is good enough to separate "short/simple" from
    "long/complex" prompts for the demo's tier choice, and is stated as a heuristic, not a measurement.
    """
    n_words = len(prompt.split())
    return max(0.0, min(1.0, n_words / 1500.0))


class EnvelopeRouter:
    """Routes one task at a time, carrying live remaining-budget state via the CloudClient it is given."""

    def __init__(self, client: CloudClient, machine: str = "evo-t2s", runtime_policy: str = "llama_ngl99",
                 quality_floor: float = DEFAULT_QUALITY_FLOOR, co_runner_present: bool = False,
                 co_runner_kind: str = "bandwidth"):
        self.client = client
        self.machine = machine
        self.runtime_policy = runtime_policy
        self.quality_floor = quality_floor
        self.co_runner_present = co_runner_present
        self.co_runner_kind = co_runner_kind

    def route(self, task: dict[str, Any]) -> RouteDecision:
        task_id = task.get("task_id", "unknown")
        prompt = task.get("prompt", "")
        model_id = task.get("model_id", "qwen3-8b")
        context_length = task.get("context_length", max(1, len(prompt.split()) * 2))
        prompt_tokens = task.get("prompt_tokens", context_length)
        full_prompt_tokens = task.get("full_prompt_tokens", prompt_tokens)

        prediction = em.predict(
            self.machine, self.runtime_policy, model_id, context_length,
            prompt_tokens=prompt_tokens, full_prompt_tokens=full_prompt_tokens,
            co_runner=self.co_runner_present, co_runner_kind=self.co_runner_kind,
        )
        pred_dict = {
            "feasibility": prediction.feasibility, "latency": prediction.latency,
            "effective_context": prediction.effective_context, "quality_regime": prediction.quality_regime,
            "failure_silence": prediction.failure_silence,
        }

        # --- Rule 1: memory-budget feasibility at THIS context length (A-24). predict_feasibility already folds
        # in the make_failure_map/EVIDENCE outcome label whenever the margin is negative for this specific
        # (machine, model, context_length) -- that is the per-task signal, not the generic per-cell one.
        feas = prediction.feasibility
        qual = prediction.quality_regime
        fail_outcome = prediction.failure_silence.get("outcome")

        if feas["status"] == "NOT_MEASURED":
            # No A-24-equivalent bisection exists for this machine/model -- fall back to the generic per-cell
            # EVIDENCE outcome for (machine, runtime_policy) as the only available silent/hard-failure signal.
            if fail_outcome in SILENT_FAILURE_OUTCOMES:
                return self._route_away_from_local(
                    task, task_id, pred_dict,
                    reason=f"blocked local: feasibility is NOT_MEASURED for this (machine, model, context), and "
                           f"(machine={self.machine}, runtime_policy={self.runtime_policy}) is a known "
                           f"{fail_outcome} cell in analysis/make_failure_map.EVIDENCE "
                           f"({prediction.failure_silence.get('detail')}) -- unverified at this exact context, "
                           f"but this router never assumes success when the only evidence available says failure",
                    blocked=True,
                )
        elif feas["status"] != "FITS":
            return self._route_away_from_local(
                task, task_id, pred_dict,
                reason=f"blocked local: predict_feasibility returned status={feas['status']!r} "
                       f"(margin_mib={feas.get('margin_mib')}) for this exact context_length -- a confirmed "
                       f"{feas['status']} outcome from the real A-24 budget-boundary fit, not a generic cell label",
                blocked=True,
            )

        # --- Rule 2: never route into a predicted silent-truncation-style quality cliff below the floor, even
        # when the memory-feasibility check above passed cleanly (truncation is an orthogonal failure mode).
        if qual["regime"] == "truncation_cliff" and qual["predicted_score"] < self.quality_floor:
            return self._route_away_from_local(
                task, task_id, pred_dict,
                reason=f"blocked local: predicted quality regime is truncation_cliff with predicted_score="
                       f"{qual['predicted_score']} (ratio={qual['ratio']}), below quality_floor="
                       f"{self.quality_floor} -- this is the harness's own 'processed_tokens < sent_tokens, no "
                       f"error' silent_truncation signature (harness/t2s_r2_session_growth.py), not a new rule",
                blocked=True,
            )

        # --- Local is feasible and not a predicted silent failure: route locally.
        return RouteDecision(
            task_id=task_id, target="local",
            reason=f"routed local: predict_feasibility={feas['status']} (margin_mib={feas['margin_mib']}), quality_regime="
                   f"{qual['regime']} (predicted_score={qual['predicted_score']}) >= quality_floor="
                   f"{self.quality_floor}, failure_silence outcome={fail_outcome!r} is not in "
                   f"SILENT_FAILURE_OUTCOMES, predicted ttft_s={prediction.latency['ttft_s']}",
            prediction=pred_dict, blocked_silent_failure=False, estimated_cost_usd=0.0,
        )

    def _route_away_from_local(self, task: dict[str, Any], task_id: str, pred_dict: dict,
                               reason: str, blocked: bool) -> RouteDecision:
        """Shared tail for every "don't run this locally" path: try cloud within budget, else compaction if
        the quality floor allows, else refuse -- never silently fall back to the blocked local configuration."""
        prompt = task.get("prompt", "")
        difficulty = estimate_difficulty(prompt)
        tier = "strong" if difficulty > 0.5 else "cheap"
        model_id = CLOUD_MODELS[tier]
        input_tokens = max(1, len(prompt.split()))
        output_tokens = task.get("expected_output_tokens", 256)
        projected_cost = estimate_cost_usd(model_id, input_tokens, output_tokens)

        if projected_cost <= self.client.remaining_budget_usd:
            return RouteDecision(
                task_id=task_id, target=f"cloud_{tier}",
                reason=f"{reason}; routed to cloud_{tier} ({model_id}): difficulty={difficulty:.2f} -> {tier} "
                       f"tier, projected_cost_usd={projected_cost:.4f} <= remaining_budget_usd="
                       f"{self.client.remaining_budget_usd:.4f}",
                prediction=pred_dict, blocked_silent_failure=blocked, estimated_cost_usd=projected_cost,
            )

        # Over budget for cloud. If the quality floor can tolerate a compacted (shorter) local prompt that would
        # bring it back under the effective context (i.e. the ONLY reason this was blocked was feasibility/
        # truncation at the full prompt length, not a hard failure outcome), offer compaction; otherwise refuse.
        qual = pred_dict["quality_regime"]
        feas = pred_dict["feasibility"]
        fail_outcome = pred_dict["failure_silence"].get("outcome")
        if fail_outcome not in SILENT_FAILURE_OUTCOMES and feas["status"] != "HARD_FAIL" and \
                qual["regime"] == "truncation_cliff":
            return RouteDecision(
                task_id=task_id, target="compact_then_local",
                reason=f"{reason}; over cloud budget (projected_cost_usd={projected_cost:.4f} > "
                       f"remaining_budget_usd={self.client.remaining_budget_usd:.4f}) and the block is a "
                       f"truncation_cliff (not a hard/silent failure outcome) -- compaction to fit effective "
                       f"context is offered instead of a cloud call this router cannot afford",
                prediction=pred_dict, blocked_silent_failure=blocked, estimated_cost_usd=0.0,
            )

        return RouteDecision(
            task_id=task_id, target="refuse",
            reason=f"{reason}; refused: over cloud budget (projected_cost_usd={projected_cost:.4f} > "
                   f"remaining_budget_usd={self.client.remaining_budget_usd:.4f}) and local is blocked by a "
                   f"hard/silent failure outcome ({fail_outcome!r}), so compaction cannot rescue this task -- "
                   f"this router never falls back to running it in the blocked local configuration",
            prediction=pred_dict, blocked_silent_failure=blocked, estimated_cost_usd=0.0,
        )


if __name__ == "__main__":
    import tempfile

    from src.cloud.client import print_stub_banner_if_needed

    demo_tasks = [
        # over native-context -> blocked (truncation_cliff below floor) -> cloud, NOT a memory-budget failure.
        {"task_id": "over_native_ctx_truncation", "model_id": "qwen3-32b", "context_length": 100000,
         "prompt": "short prompt"},
        # over the A-24 memory budget boundary -> HARD_FAIL -> cloud.
        {"task_id": "over_budget_hard_fail", "model_id": "qwen3-32b", "context_length": 120000,
         "prompt": "short prompt"},
        # fits comfortably under both the native-context cap and the memory budget -> routed local.
        {"task_id": "fits_local", "model_id": "qwen3-8b", "context_length": 300,
         "prompt_tokens": 294, "full_prompt_tokens": 300, "prompt": "x " * 50},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        client = CloudClient(api_key=None, ledger_path=Path(tmp) / "ledger.jsonl")
        print_stub_banner_if_needed(client)
        router = EnvelopeRouter(client, machine="evo-t2s", runtime_policy="llama_ngl99")
        for t in demo_tasks:
            d = router.route(t)
            print(f"\n{d.task_id}: target={d.target} blocked_silent_failure={d.blocked_silent_failure}")
            print(f"  reason: {d.reason}")
