"""INF-6: criterion-aware probe timeout derivation.

Against a TTFT SLO, a probe still running at a small multiple of the SLO has
definitively failed the criterion. Waiting for ``generation.timeout_s=1800``
from ``configs/delta_n.yaml`` burns wall clock without adding information.
"""

from __future__ import annotations

from typing import Any

# A probe still running at 3x the TTFT SLO has failed the criterion.
# For slo_s=10 -> timeout_s=30. Recorded in plan.json as probe_timeout_derivation.
TTFT_SLO_TIMEOUT_MULTIPLIER: float = 3.0

CRITERION_TTFT_SLO = "ttft_slo"
CRITERION_COMPLETION = "completion"


def derive_probe_timeout_s(
    *,
    criterion: str,
    slo_s: float | None,
    config_timeout_s: float,
) -> dict[str, Any]:
    """Derive the parent ``run_child`` timeout from the active criterion.

    Returns a record suitable for ``plan.json`` under ``probe_timeout_derivation``.
    ``timeout_s`` is the value to write into ``cfg["generation"]["timeout_s"]``.
    """
    criterion = str(criterion)
    config_timeout_s = float(config_timeout_s)
    if config_timeout_s <= 0:
        raise ValueError(f"config_timeout_s must be > 0; got {config_timeout_s}")

    if criterion == CRITERION_TTFT_SLO:
        if slo_s is None:
            raise ValueError("ttft_slo criterion requires slo_s")
        slo = float(slo_s)
        if slo <= 0:
            raise ValueError(f"slo_s must be > 0; got {slo}")
        timeout_s = slo * TTFT_SLO_TIMEOUT_MULTIPLIER
        return {
            "criterion": criterion,
            "timeout_s": timeout_s,
            "slo_s": slo,
            "multiplier": TTFT_SLO_TIMEOUT_MULTIPLIER,
            "formula": "timeout_s = slo_s * 3",
            "derivation": (
                f"criterion=ttft_slo; timeout_s = slo_s * {TTFT_SLO_TIMEOUT_MULTIPLIER:g} "
                f"= {slo:g} * {TTFT_SLO_TIMEOUT_MULTIPLIER:g} = {timeout_s:g}. "
                "A probe still running at 3x the TTFT SLO has definitively failed "
                "the criterion; configs/delta_n.yaml generation.timeout_s is not used."
            ),
            "config_timeout_s_not_used": config_timeout_s,
            "basis": (
                "INF-6: against a 10 s TTFT SLO a probe still running at 30 s has "
                "failed; 1800 s parent timeout cost >1 h across CAP-3 arm-1 sessions."
            ),
        }

    return {
        "criterion": criterion,
        "timeout_s": config_timeout_s,
        "slo_s": None,
        "multiplier": None,
        "formula": "timeout_s = configs/delta_n.yaml generation.timeout_s",
        "derivation": (
            f"criterion={criterion}; using config generation.timeout_s="
            f"{config_timeout_s:g} (completion / memory-wall path)."
        ),
        "config_timeout_s_not_used": None,
        "basis": "Non-ttft_slo criteria retain the delta_n generation timeout.",
    }
