"""Device admission: session cap and cold-prefill memory check."""

from __future__ import annotations

from seam.duet import params as prm


def session_cap(prec: str, n_new: float, m: float) -> float:
    return min(
        (prm.B * (1.0 - m) - prm.A[prec] * n_new) / prm.K[prec], float(prm.W - prm.WINDOW_HEADROOM)
    )


def cold_mem_ok(p: float, prec: str) -> bool:
    return p * (prm.K[prec] + prm.A[prec]) <= prm.B * (1.0 - prm.MARGIN_M)
