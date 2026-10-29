"""Pre-generate guard: the three ROUTE-1 inequalities (cloud branch stubbed)."""

from __future__ import annotations

from seam.route1.device import MARGIN_M, WINDOW_HEADROOM, Config, Constants

SLO_S = 10.0


def admit(
    n_ctx: int,
    n_new: int,
    max_new: int,
    cfg: Config,
    c: Constants,
    ttft_pred_s: float,
    m: float = MARGIN_M,
) -> tuple[bool, str | None, str]:
    """Return (admit, binding_ceiling, reason).

    Memory: k*n_ctx + a*min(n_new, chunk) <= B*(1-m). A non-resident call prefills the whole prompt
    (n_new = n_ctx). The margin applies to resident configs (session-law error band) only."""
    k = c.k[cfg.prec]
    step = n_ctx if not cfg.resident else n_new
    if cfg.chunk is not None:
        step = min(step, cfg.chunk)
    budget = c.B * (1.0 - m) if cfg.resident else c.B
    if k * n_ctx + c.a * step > budget:
        return False, "memory", f"k*n_ctx + a*step = {k * n_ctx + c.a * step:.0f} > {budget:.0f}"
    if n_ctx + max_new > c.W - WINDOW_HEADROOM:
        return False, "window", f"n_ctx + max_new = {n_ctx + max_new} > {c.W - WINDOW_HEADROOM}"
    if ttft_pred_s > SLO_S:
        return False, "ttft", f"predicted TTFT {ttft_pred_s:.2f} s > {SLO_S} s"
    return True, None, "admit"


def cloud_branch(n_ctx: int) -> str:
    """Stub: the cloud call is made by the caller (no network here)."""
    return f"CLOUD({n_ctx})"
