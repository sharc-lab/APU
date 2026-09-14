"""Provenance corrections for sealed manifests (AM-036).

``raw/`` is write-once. When a sealed manifest records incorrect metadata, the sealed bytes stay
intact and a correction registry under ``derived/manifest_corrections/`` supplies explicit
amendments for known digests. The **default** loader path is structural: detect promote-time
power leaked into measurement ``power_state`` on post-hoc / retro-sealed runs, without growing a
run_id list.

Authoritative view: :func:`seam.manifest.load_run_manifest`.
Do not backfill measurement environment from memory or from promote-time samples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

__all__ = [
    "CORRECTIONS_DIR_NAME",
    "NULL_MEASUREMENT_POWER_STATE",
    "apply_manifest_corrections",
    "detect_promote_time_power_leak",
    "load_correction_registry",
]

CORRECTIONS_DIR_NAME: Final = "manifest_corrections"
_REGISTRY_FILENAME: Final = "registry.json"
_SUMMARY_FILENAME: Final = "summary.json"

NULL_MEASUREMENT_POWER_STATE: Final[dict[str, Any]] = {
    "on_battery": None,
    "battery_pct_start": None,
    "battery_pct_end": None,
    "soc_at_start": None,
    "soc_at_end": None,
    "charging": None,
    "power_plan": None,
    "display_brightness": None,
    "defender_realtime": None,
    "windows_update_paused": None,
    "pinned_profile": None,
    "ac_disconnected_for_s": None,
    "discharge_rate_stable": None,
    "background_quiesced": None,
    "wifi_state": None,
    "design_capacity_mwh": None,
    "full_charge_capacity_mwh": None,
    "power_source": None,
}

_MEASUREMENT_POWER_SAMPLE_KEYS: Final = frozenset(NULL_MEASUREMENT_POWER_STATE)

_POWER_LEAK_NOTE: Final = (
    "This run predates measurement-time powerstate capture; the true measurement "
    "power state is unrecorded. Values previously written into power_state were "
    "promote-time host samples (AM-036 / AF-036) and must not be read as run environment."
)

_STRUCTURAL_CORRECTION_ID: Final = "structural-promote-time-power-leak"


def load_correction_registry(*, repo_root: Path) -> dict[str, Any]:
    """Load the correction registry, or an empty registry if absent."""
    path = repo_root / "derived" / CORRECTIONS_DIR_NAME / _REGISTRY_FILENAME
    if not path.is_file():
        return {
            "schema": "seam.manifest_corrections/v1",
            "corrections": {},
        }
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
    return parsed


def _load_run_summary(run_id: str, *, repo_root: Path) -> dict[str, Any] | None:
    path = repo_root / "raw" / run_id / _SUMMARY_FILENAME
    if not path.is_file():
        return None
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
    return parsed


def _measurement_power_has_sample(power_state: dict[str, Any] | None) -> bool:
    if not power_state:
        return False
    return any(power_state.get(key) is not None for key in _MEASUREMENT_POWER_SAMPLE_KEYS)


def _promote_time_present(manifest: dict[str, Any]) -> bool:
    promote = manifest.get("promote_time_power_state")
    if not isinstance(promote, dict) or not promote:
        return False
    return any(promote.get(key) is not None for key in promote)


def _is_post_hoc_promote_summary(summary: dict[str, Any] | None) -> bool:
    """True when companion summary marks a post-hoc / derived-diagnostic promote."""
    if not summary:
        return False
    promotion = summary.get("promotion")
    return isinstance(promotion, str) and promotion.startswith("post_hoc")


def detect_promote_time_power_leak(
    manifest: dict[str, Any],
    *,
    summary: dict[str, Any] | None = None,
) -> bool:
    """Return True when measurement ``power_state`` looks like a promote-time leak (AM-036).

    Structural rule (not a run_id list):

    1. Run is marked retro-seal / post-hoc promote - either ``manifest.retro_seal is True``
       or ``summary.promotion`` starts with ``post_hoc``.
    2. ``promote_time_power_state`` is absent / empty (leak was written into measurement fields).
    3. Measurement ``power_state`` carries non-null sample fields.
    4. ``measurement_power_from_records`` is not True (record-backed measurement stays trusted).

    Self-sealed runs (no retro_seal, no post_hoc promotion marker) are never matched - e.g.
    ``693b44d2`` keeps its AC measurement power. Correct modern retro-seals that already null
    measurement power and populate ``promote_time_power_state`` are also unmatched.
    """
    if manifest.get("measurement_power_from_records") is True:
        return False
    retro = manifest.get("retro_seal") is True or _is_post_hoc_promote_summary(summary)
    if not retro:
        return False
    if _promote_time_present(manifest):
        return False
    return _measurement_power_has_sample(manifest.get("power_state"))


def _null_measurement_promote_leak_view(
    manifest: dict[str, Any],
    *,
    correction_meta: dict[str, Any],
    power_state_note: str | None = None,
    promote_time_power_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(manifest)
    sealed_power = dict(manifest.get("power_state") or {})
    promote = promote_time_power_state
    if promote is None and any(sealed_power.get(k) is not None for k in sealed_power):
        promote = {
            key: sealed_power.get(key)
            for key in NULL_MEASUREMENT_POWER_STATE
            if key in sealed_power
        }
        promote["captured_at_utc"] = manifest.get("timestamp_utc")
    out["power_state"] = dict(NULL_MEASUREMENT_POWER_STATE)
    out["promote_time_power_state"] = promote
    out["power_state_note"] = power_state_note or _POWER_LEAK_NOTE
    out["_manifest_correction"] = {
        **correction_meta,
        "kind": "null_measurement_power_promote_leak",
        "raw_bytes_unmutated": True,
    }
    return out


def apply_manifest_corrections(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a shallow-copied manifest with provenance corrections applied.

    Order:

    1. Explicit ``derived/manifest_corrections/registry.json`` entry for ``run_id`` (known
       digests / amendments).
    2. Else structural promote-time power leak detection (see
       :func:`detect_promote_time_power_leak`). Future leaks must not require a new run_id
       registry row - emit refuse is the source fix; this is the safety net.
    """
    run_id = str(manifest.get("run_id") or "")
    registry = load_correction_registry(repo_root=repo_root)
    entry = (registry.get("corrections") or {}).get(run_id)
    if entry:
        kind = entry.get("kind")
        if kind != "null_measurement_power_promote_leak":
            raise ValueError(f"unknown manifest correction kind for {run_id}: {kind!r}")
        return _null_measurement_promote_leak_view(
            manifest,
            correction_meta={
                "id": entry.get("id"),
                "source": "registry",
                "audit_ref": entry.get("audit_ref"),
            },
            power_state_note=entry.get("power_state_note"),
            promote_time_power_state=entry.get("promote_time_power_state"),
        )

    resolved_summary = summary
    if resolved_summary is None and run_id:
        resolved_summary = _load_run_summary(run_id, repo_root=repo_root)

    if detect_promote_time_power_leak(manifest, summary=resolved_summary):
        return _null_measurement_promote_leak_view(
            manifest,
            correction_meta={
                "id": _STRUCTURAL_CORRECTION_ID,
                "source": "structural",
                "audit_ref": "AMENDMENTS.md AM-036",
            },
        )

    return manifest
