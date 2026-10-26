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


class ZachCeilingPredictor:
    """Placeholder for Zach's ceiling formula. Not implemented: awaiting his files (operator, 2026-10-26)."""

    source = "zach-ceiling"

    def predict(self, task_context: dict, device_profile: dict) -> FitsResult:
        raise NotImplementedError("awaiting Zach's files")


DEFAULT_PREDICTOR = ProjectionBaseline()


def fits_locally(projected_tokens: int, window: int, margin: float = 0.0,
                 predictor: FitsLocallyPredictor | None = None) -> FitsResult:
    """Convenience call for the routers: one projected prompt against one window."""
    p = predictor or DEFAULT_PREDICTOR
    return p.predict({"projected_tokens": projected_tokens}, {"window": window, "margin": margin})
