"""Measured-model TTFT (seconds). Inputs: n = new tokens, P = prompt tokens including them."""

from __future__ import annotations

from seam.duet import params as prm


def cold(p: float) -> float:
    return prm.COLD_C0 + prm.COLD_C1 * p + prm.COLD_C2 * p * p


def ext(n: float, p: float) -> float:
    return float(p * (prm.ALPHA + prm.BETA * n) * (p / prm.P_REF) ** prm.GAMMA)


def reprefill_wins(n: float, p: float) -> bool:
    return cold(p) < ext(n, p)


def cold_ns_per_pair(p: float) -> float:
    """Quadratic cold term per causal attention pair (~n^2/2 pairs): 2*C2, in ns."""
    del p
    return 2.0 * prm.COLD_C2 * 1e9


def ext_ns_per_pair(n: float, p: float) -> float:
    return ext(n, p) / (n * p) * 1e9
