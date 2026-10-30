"""The "fits locally?" predictor interface (2026-10-26 operator brief, item 3: ceiling-aware routing).

Every route-at-start decision in this repo asks one question: will this task's prompt fit in the window the local
runtime has loaded? Before this module the answer was written inline in three places (src/dse/route_at_start.fits,
analysis/impact_model.route_measured_policy, src/dse/dse_v1.build_long_candidates_pure). They now all call a
FitsLocallyPredictor, so a better ceiling (Zach's formula, once his files arrive) can replace the baseline without
touching the routers.

Interface:
  predict(task_context, device_profile) -> FitsResult {fits, projected_tokens, ceiling_tokens, source}
    task_context    a dict describing the task. The baseline reads "projected_tokens" (already in model tokens), or
                    "est_tokens" (the harness's chars/4 estimate, x2_r2_agent.est_tokens) times "calib" (the tokenizer
                    calibration, e.g. x2_r2_evalset.PROJ_CALIB or the B3 v2 token_calib_ratio), rounded.
    device_profile  a dict describing the local setting. The baseline reads "window" (the loaded num_ctx) and an
                    optional "margin" (share of the window held back; 0 when absent).

ProjectionBaseline is the current estimate, unchanged: ceiling = floor(window x (1 - margin)), fits iff projected <=
ceiling. With margin 0 that is "projected <= window", the rule of hybrid-route-at-start-measured and own-dse-v1; with
route_at_start.MARGIN it is route_at_start.fits. tests/test_fits_locally.py checks both register rows compute
identically through the interface.

SessionGrowthPredictor (row fit-predictor-v1, docs/EXPERIMENTS.md "PRE-REGISTRATION (2026-10-29): fit-predictor-v1")
is the session-start form: from the first k turns' prompt sizes it projects the history's growth to a later turn and
compares that with the device's ceiling, which is the measured ceiling where the device profile has one
("measured_ceiling", e.g. llama-server's n_ctx) and the loaded window otherwise. A ceiling model (Zach's) plugs in as
ceiling_model: anything with ceiling(device_profile) -> int.
  task_context    "first_turns": [(turn_idx, prompt_tokens), ...] (the first k recorded turns, in order) and
                  "target_turn": the turn index to project to; or "projected_tokens" when there is nothing left to grow.

ZachCeilingPredictor is a placeholder only: it raises NotImplementedError until Zach's files are in the repo. Do not
implement his formula without them (operator, 2026-10-26).
"""
from __future__ import annotations

import math
from typing import Protocol, TypedDict, runtime_checkable

BASELINE_SOURCE = "projection-baseline"


class FitsResult(TypedDict):
    fits: bool
    projected_tokens: int
    ceiling_tokens: int
    source: str


@runtime_checkable
class FitsLocallyPredictor(Protocol):
    """Anything with predict(task_context, device_profile) -> FitsResult."""

    def predict(self, task_context: dict, device_profile: dict) -> FitsResult: ...


class ProjectionBaseline:
    """The current estimate: projected prompt (chars/4 x calibration, or a precomputed projection) vs the loaded
    window less an optional margin."""

    source = BASELINE_SOURCE

    def __init__(self, margin: float | None = None):
        self.margin = margin  # overrides device_profile["margin"] when set

    @staticmethod
    def projected(task_context: dict) -> int:
        if task_context.get("projected_tokens") is not None:
            return int(task_context["projected_tokens"])
        if task_context.get("est_tokens") is not None and task_context.get("calib") is not None:
            return int(round(task_context["est_tokens"] * task_context["calib"]))
        raise ValueError("task_context needs projected_tokens, or est_tokens and calib")

    def ceiling(self, device_profile: dict) -> int:
        if device_profile.get("window") is None:
            raise ValueError("device_profile needs window (the loaded num_ctx)")
        margin = self.margin if self.margin is not None else float(device_profile.get("margin") or 0.0)
        return math.floor(int(device_profile["window"]) * (1 - margin))

    def predict(self, task_context: dict, device_profile: dict) -> FitsResult:
        p, c = self.projected(task_context), self.ceiling(device_profile)
        return {"fits": p <= c, "projected_tokens": p, "ceiling_tokens": c, "source": self.source}


GROWTH_SOURCE = "session-growth-k3"
GROWTH_K = 3


def device_ceiling(device_profile: dict, margin: float = 0.0) -> int:
    """The measured ceiling where the profile records one, else the loaded window; less an optional margin."""
    c = device_profile.get("measured_ceiling")
    if c is None:
        c = device_profile.get("window")
    if c is None:
        raise ValueError("device_profile needs window (the loaded num_ctx) or measured_ceiling")
    return math.floor(int(c) * (1 - margin))


def project_growth(first_turns, target_turn: int, k: int = GROWTH_K) -> int:
    """Linear projection from the first k turns only: g = max(0, (p_last - p_first) / (t_last - t_first)),
    projected = p_last + g x (target_turn - t_last). Fewer than k turns: ValueError."""
    ft = list(first_turns)[:k]
    if len(ft) < k:
        raise ValueError(f"need the first {k} turns, got {len(ft)}")
    (t0, p0), (t1, p1) = ft[0], ft[-1]
    g = max(0.0, (p1 - p0) / (t1 - t0)) if t1 != t0 else 0.0
    return int(round(p1 + g * max(0, target_turn - t1)))


class SessionGrowthPredictor:
    """Session-start "will this task overflow": the first k turns' prompts projected to the task's turn, against the
    device's measured ceiling (else the loaded window). ceiling_model, when given, supplies the ceiling instead."""

    source = GROWTH_SOURCE

    def __init__(self, k: int = GROWTH_K, margin: float = 0.0, ceiling_model=None):
        self.k, self.margin, self.ceiling_model = k, margin, ceiling_model

    def projected(self, task_context: dict) -> int:
        if task_context.get("first_turns") is not None:
            return project_growth(task_context["first_turns"], int(task_context["target_turn"]), self.k)
        if task_context.get("projected_tokens") is not None:
            return int(task_context["projected_tokens"])
        raise ValueError("task_context needs first_turns and target_turn, or projected_tokens")

    def ceiling(self, device_profile: dict) -> int:
        if self.ceiling_model is not None:
            return int(self.ceiling_model.ceiling(device_profile))
        return device_ceiling(device_profile, self.margin)

    def predict(self, task_context: dict, device_profile: dict) -> FitsResult:
        p, c = self.projected(task_context), self.ceiling(device_profile)
        return {"fits": p <= c, "projected_tokens": p, "ceiling_tokens": c, "source": self.source}


class ZachCeilingPredictor:
    """Placeholder for Zach's ceiling formula. Not implemented: awaiting his files (operator, 2026-10-26). Usable as
    SessionGrowthPredictor(ceiling_model=...) once ceiling() exists."""

    source = "zach-ceiling"

    def ceiling(self, device_profile: dict) -> int:
        raise NotImplementedError("awaiting Zach's files")

    def predict(self, task_context: dict, device_profile: dict) -> FitsResult:
        raise NotImplementedError("awaiting Zach's files")


DEFAULT_PREDICTOR = ProjectionBaseline()


def fits_locally(projected_tokens: int, window: int, margin: float = 0.0,
                 predictor: FitsLocallyPredictor | None = None) -> FitsResult:
    """Convenience call for the routers: one projected prompt against one window."""
    p = predictor or DEFAULT_PREDICTOR
    return p.predict({"projected_tokens": projected_tokens}, {"window": window, "margin": margin})
