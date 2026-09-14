"""Run-manifest emission and validation (blueprint §5.2, spec §6.1).

Every experimental run emits exactly one manifest. **No manifest, no data** - and conversely, no
number may be reported that does not trace to a ``run_id`` recorded here (spec §9.2).

What :func:`emit` guarantees, in order:

1. Git state is captured, and a dirty tree is **refused** unless ``allow_dirty`` is passed - in
   which case that fact is written into the manifest.
2. The configuration is fully resolved and hashed, so the manifest pins the exact configuration the
   run behaved according to.
3. Provenance artifacts are hashed **from bytes on disk at emit time** (blueprint AF-002), never
   transcribed from a document.
4. The manifest validates against ``seam/schemas/run_manifest.schema.json`` **before** it is
   written. An invalid manifest never reaches ``raw/``.
5. The run directory is sealed write-once.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from seam import SPEC_VERSION
from seam.blinding import blinded_label_for, get_or_create_salt, record_unblind_entry
from seam.config import ResolvedConfig
from seam.errors import ManifestValidationError, ProvenanceError, RawStoreError
from seam.gitinfo import GitState, assert_clean_or_allowed, capture_git_state
from seam.hashing import sha256_file, sha256_tree
from seam.isolation import resolve_isolation_mode
from seam.jsonlog import add_json_sink, log_event, remove_json_sink, utc_now_iso
from seam.launch_context import resolve_launch_context
from seam.locks import exclusive
from seam.rawstore import RunDir, create_run_dir

__all__ = [
    "SCHEMA_PATH",
    "RunHandle",
    "build_manifest",
    "emit",
    "is_elevated",
    "load_run_manifest",
    "load_schema",
    "validate_manifest",
]

# Measurement power_state fields that must never be filled from a promote-time host sample
# (AM-036). A retro-seal may leave them null or supply values derived from measurement records.
_MEASUREMENT_POWER_SAMPLE_KEYS: Final = frozenset(
    {
        "on_battery",
        "battery_pct_start",
        "battery_pct_end",
        "soc_at_start",
        "soc_at_end",
        "charging",
        "power_plan",
        "display_brightness",
        "defender_realtime",
        "windows_update_paused",
        "pinned_profile",
        "ac_disconnected_for_s",
        "discharge_rate_stable",
        "background_quiesced",
        "wifi_state",
        "design_capacity_mwh",
        "full_charge_capacity_mwh",
        "power_source",
    }
)

SCHEMA_PATH: Final = Path(__file__).parent / "schemas" / "run_manifest.schema.json"

_MANIFEST_FILENAME: Final = "manifest.json"
_SUMMARY_FILENAME: Final = "summary.json"
_EVENTS_FILENAME: Final = "events.ndjson"


@dataclass(frozen=True, slots=True)
class RunHandle:
    """The result of a successful :func:`emit`."""

    run_id: str
    manifest: dict[str, Any]
    run_dir: RunDir
    raw_sha256: str


# ==================================================================================================
# Elevation
# ==================================================================================================


def is_elevated() -> bool:
    """Return whether the process has Administrator (or root) privileges.

    Recorded in every manifest because RAPL MSR access (spec §3.1 signal S2) requires elevation, so
    a non-elevated run cannot have produced RAPL data. Returns a definite boolean and logs rather
    than raising, since elevation is a recorded property of a run, not a precondition for emitting a
    manifest.
    """
    if sys.platform == "win32":
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
        except (AttributeError, OSError) as exc:
            log_event(
                "manifest.elevation_check_failed",
                severity="warning",
                message="could not determine elevation; recording elevated=False",
                error=str(exc),
            )
            return False
    return hasattr(os, "geteuid") and os.geteuid() == 0


# ==================================================================================================
# Schema validation
# ==================================================================================================


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Load and cache the run-manifest JSON Schema."""
    if not SCHEMA_PATH.is_file():
        raise ManifestValidationError(f"manifest schema not found at {SCHEMA_PATH}")
    parsed: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return parsed


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate a manifest against the schema.

    Every error is collected and reported together, rather than only the first, because fixing
    manifests one field per run is needlessly slow.

    Raises:
        ManifestValidationError: If the manifest is invalid, or if ``jsonschema`` is unavailable.
            A missing validator is a hard failure: an unvalidated manifest would defeat the entire
            point of having a schema, so the run stops rather than proceeding unchecked.
    """
    try:
        import jsonschema
    except ImportError as exc:
        raise ManifestValidationError(
            "jsonschema is required to validate run manifests (`pip install jsonschema`). "
            "Refusing to emit an unvalidated manifest."
        ) from exc

    validator_cls = jsonschema.validators.validator_for(load_schema())
    validator = validator_cls(load_schema(), format_checker=validator_cls.FORMAT_CHECKER)

    errors = sorted(validator.iter_errors(manifest), key=lambda e: list(e.absolute_path))
    if errors:
        rendered = "\n".join(
            f"  - {'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
            for err in errors
        )
        raise ManifestValidationError(
            f"manifest failed schema validation with {len(errors)} error(s):\n{rendered}"
        )


def _measurement_power_has_sample(power_state: dict[str, Any] | None) -> bool:
    """Return True if ``power_state`` carries any non-null measurement sample field."""
    if not power_state:
        return False
    return any(power_state.get(key) is not None for key in _MEASUREMENT_POWER_SAMPLE_KEYS)


def _refuse_retro_seal_measurement_power(
    *,
    retro_seal: bool,
    power_state: dict[str, Any] | None,
    measurement_power_from_records: bool,
) -> None:
    """Refuse promote-time leakage into measurement ``power_state`` (AM-036).

    Retro-seal / raw-promote paths must leave measurement environment null, or supply values
    derived from measurement records with ``measurement_power_from_records=True``. A live
    ``capture_power_state()`` at promote time is never a legal source for ``power_state``.
    """
    if not retro_seal:
        return
    if power_state is None or not _measurement_power_has_sample(power_state):
        return
    if measurement_power_from_records:
        return
    raise ProvenanceError(
        "retro_seal=True refuses to populate measurement power_state from promote-time "
        "sampling (AM-036). Pass power_state=None (null measurement environment), or pass "
        "measurement-record-derived fields with measurement_power_from_records=True. "
        "Optional promote-time samples belong in promote_time_power_state only."
    )


def load_run_manifest(run_id: str, *, repo_root: Path) -> dict[str, Any]:
    """Load ``raw/<run_id>/manifest.json`` and apply provenance corrections (AM-036).

    Sealed bytes under ``raw/`` are write-once and are not mutated. Corrections come from:

    1. Explicit ``derived/manifest_corrections/registry.json`` entries for known digests.
    2. A **structural** promote-time power leak rule (``retro_seal`` / summary
       ``promotion: post_hoc*`` + populated measurement ``power_state`` + absent
       ``promote_time_power_state``) - not a growing run_id list.
    """
    from seam.manifest_corrections import apply_manifest_corrections

    path = repo_root / "raw" / run_id / _MANIFEST_FILENAME
    if not path.is_file():
        raise RawStoreError(f"no manifest for run {run_id}: {path}")
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
    return apply_manifest_corrections(manifest, repo_root=repo_root)


# ==================================================================================================
# Manifest construction
# ==================================================================================================


def _hash_provenance_artifacts(config: ResolvedConfig, repo_root: Path) -> list[dict[str, str]]:
    """Hash every declared provenance artifact at emit time.

    Raises:
        ProvenanceError: If the list is empty or an artifact is missing. Blueprint §5.2 requires
            each run to pin the provenance snapshot it relied on, so a missing artifact invalidates
            the run rather than degrading it to a null hash.
    """
    declared = config.get("provenance_artifacts") or []
    if not declared:
        raise ProvenanceError(
            "platform config declares no provenance_artifacts; blueprint §5.2 requires every run "
            "to pin the provenance snapshot it relied on"
        )

    artifacts: list[dict[str, str]] = []
    for relative in declared:
        path = repo_root / relative
        if not path.is_file():
            raise ProvenanceError(
                f"declared provenance artifact is missing: {relative} (looked in {path}). "
                f"Refusing to emit a manifest with an unpinned provenance chain."
            )
        artifacts.append({"path": str(relative), "sha256": sha256_file(path)})
    return artifacts


def build_manifest(
    *,
    run_id: str,
    config: ResolvedConfig,
    git_state: GitState,
    allow_dirty: bool,
    target: str,
    workload: dict[str, Any],
    condition_label: str,
    blinded_label: str,
    repo_root: Path,
    model: dict[str, Any] | None = None,
    drivers: dict[str, Any] | None = None,
    power_state: dict[str, Any] | None = None,
    thermal: dict[str, Any] | None = None,
    topology_override: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    raw_sha256: str | None = None,
    self_check: str = "pass",
    isolation_mode: str | None = None,
    isolation_evidence: dict[str, Any] | None = None,
    launch_context: str | None = None,
    session_id: int | None = None,
    window_station: str | None = None,
    require_launch_context: bool = False,
    retro_seal: bool = False,
    measurement_power_from_records: bool = False,
    promote_time_power_state: dict[str, Any] | None = None,
    power_state_note: str | None = None,
) -> dict[str, Any]:
    """Assemble a manifest dict. Does not validate or write it.

    Split out from :func:`emit` so tests can build manifests without touching git, the filesystem,
    or hardware.

    ``topology_override`` exists for one legitimate case: the topology-verification run itself,
    which *produces* the mapping and therefore cannot read it from a config that does not yet
    contain it.

    ``isolation_mode`` is resolved here when not supplied, which means it is resolved for every
    caller rather than only the ones that remembered. Resolution has no default and refuses a
    declaration the machine contradicts, so an undeclared or false mode stops the run instead of
    producing a manifest that is quietly wrong about the conditions it was measured under.

    ``launch_context`` / ``session_id`` / ``window_station`` record how and where the process was
    launched. When not supplied they are resolved from ``SEAM_LAUNCH_CONTEXT`` and a process
    placement probe. Older call sites may leave them null; acceptance and ΔN pass
    ``require_launch_context=True`` so a mislabelled Cursor-session run cannot seal as detached.

    ``retro_seal`` marks a post-hoc promote into ``raw/``. Measurement ``power_state`` must then
    come from measurement records (``measurement_power_from_records=True``) or stay null;
    promote-time host samples go only in ``promote_time_power_state`` (AM-036). Both flags are
    written into the manifest so loaders can detect leakage structurally.
    """
    _refuse_retro_seal_measurement_power(
        retro_seal=retro_seal,
        power_state=power_state,
        measurement_power_from_records=measurement_power_from_records,
    )
    if isolation_mode is None:
        isolation_mode, resolved_evidence = resolve_isolation_mode()
    else:
        resolved_evidence = (
            dict(isolation_evidence)
            if isolation_evidence is not None
            else resolve_isolation_mode(isolation_mode)[1]
        )
    resolved_launch, placement = resolve_launch_context(
        launch_context, required=require_launch_context
    )
    if session_id is None:
        session_id = placement.get("session_id")
    if window_station is None:
        window_station = placement.get("window_station")
    launch_context = resolved_launch
    topology = topology_override or {
        "p_cpus": config.get("topology.p_cpus"),
        "lpe_cpus": config.get("topology.lpe_cpus"),
        "verified": bool(config.get("topology.verified", False)),
    }

    # Thermal constants are null until M2.3 freezes them (AMENDMENTS.md AM-006). The illustrative
    # values in spec §6.1 are schema examples and are deliberately not used as defaults.
    thermal_block: dict[str, Any] = {
        "ambient_c": None,
        "warmup_s": config.get("thermal.warmup_s"),
        "cooldown_ceiling_c": config.get("thermal.cooldown_ceiling_c"),
        "throttle_residency_pct": None,
        "throttle_threshold_pct": config.get("thermal.throttle_threshold_pct"),
        "excluded": False,
        # Blueprint §6.4 / H8. Null until the run declares confound vs axis.
        "regime": None,
    }
    if thermal:
        thermal_block.update(thermal)

    npu_config = config.get("npu_config") or {}

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "spec_version": SPEC_VERSION,
        "timestamp_utc": utc_now_iso(),
        "git_sha": git_state.sha,
        "git_dirty": git_state.dirty,
        "allow_dirty": allow_dirty,
        "git_branch": git_state.branch,
        "git_dirty_files": list(git_state.dirty_files) if git_state.dirty else None,
        "config_hash": config.config_hash,
        "config_sources": list(config.sources),
        "elevated": is_elevated(),
        "isolation_mode": isolation_mode,
        "isolation_evidence": resolved_evidence,
        "launch_context": launch_context,
        "session_id": session_id,
        "window_station": window_station,
        "platform": {
            "id": config.require("platform_id"),
            "cpu": config.require("identity.cpu"),
            "family": config.require("identity.family"),
            "topology": topology,
            "os_build": str(config.require("identity.os_build")),
            "provenance_artifacts": _hash_provenance_artifacts(config, repo_root),
            # Blueprint §5.2 fields absent from spec §6.1 (AM-003). Null until captured; M2 fills
            # them. Never defaulted to a plausible-looking value.
            "microcode": None,
            "bios_version": None,
            "kernel": None,
            "power_cap_w": None,
            "cpu_governor": None,
        },
        "drivers": {
            "npu": None,
            "igpu": None,
            "openvino": None,
            "genai": None,
            "lhm_bridge": None,
        },
        "power_state": {
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
        },
        "promote_time_power_state": None,
        "power_state_note": None,
        "retro_seal": bool(retro_seal),
        "measurement_power_from_records": (True if measurement_power_from_records else None),
        "thermal": thermal_block,
        "confinement_mechanism": None,
        "target": target,
        "model": {
            "name": None,
            "revision": None,
            "quantization": None,
            "ir_sha256": None,
            "runtime": None,
            "precision": None,
            "reasoning_mode": None,
            # AM-023: discriminant is required so a flattened ModelSpec/FetchedModelSpec
            # cannot reach disk. kind=absent is the only legal null-path for non-local runs.
            "provenance": {
                "kind": "absent",
                "self_converted": None,
                "source_repo": None,
                "download_method": None,
                "export_command": None,
                "quantization_config": None,
                "ladder_position": None,
                "spec_path": None,
                "spec_sha256": None,
                "file_verification": None,
            },
            "cloud": None,
        },
        "npu_config": {
            "MAX_PROMPT_LEN": npu_config.get("MAX_PROMPT_LEN"),
            "NPUW_LLM_PREFILL_CHUNK_SIZE": npu_config.get("NPUW_LLM_PREFILL_CHUNK_SIZE"),
        },
        "workload": {
            "kind": workload["kind"],
            "benchmark": workload["benchmark"],
            "task_ids": list(workload.get("task_ids") or []),
            "seed": workload.get("seed"),
            "n_repeats": workload.get("n_repeats"),
            "concurrency": workload.get("concurrency"),
        },
        "policy": {
            "name": None,
            "version": None,
            "params": None,
            "escalation_semantics": None,
            "kv_residency": None,
        },
        "condition_label": condition_label,
        "blinded_label": blinded_label,
        "outputs": {
            "samples": None,
            "steps": None,
            "summary": _SUMMARY_FILENAME,
            "events_path": _EVENTS_FILENAME,
        },
        "integrity": {
            "self_check": self_check,
            "raw_sha256": raw_sha256,
        },
    }

    if drivers:
        manifest["drivers"].update(drivers)
    if power_state:
        manifest["power_state"].update(power_state)
    if promote_time_power_state is not None:
        manifest["promote_time_power_state"] = dict(promote_time_power_state)
    if power_state_note is not None:
        manifest["power_state_note"] = power_state_note
    if model:
        manifest["model"].update(model)
    if outputs:
        manifest["outputs"].update(outputs)

    return manifest


# ==================================================================================================
# Emission
# ==================================================================================================


def _emit_locked_body(
    *,
    config: ResolvedConfig,
    target: str,
    workload: dict[str, Any],
    condition_label: str,
    repo_root: Path,
    run_id: str | None = None,
    existing_run_dir: RunDir | None = None,
    allow_dirty: bool = False,
    summary: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    drivers: dict[str, Any] | None = None,
    power_state: dict[str, Any] | None = None,
    thermal: dict[str, Any] | None = None,
    topology_override: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    self_check: str = "pass",
    seal: bool = True,
    isolation_mode: str | None = None,
    launch_context: str | None = None,
    session_id: int | None = None,
    window_station: str | None = None,
    require_launch_context: bool = False,
    before_integrity_hash: Callable[[RunDir], dict[str, Any] | None] | None = None,
    retro_seal: bool = False,
    measurement_power_from_records: bool = False,
    promote_time_power_state: dict[str, Any] | None = None,
    power_state_note: str | None = None,
) -> RunHandle:
    """Emit a schema-valid manifest into a fresh, sealed ``raw/<run_id>/`` directory.

    Args:
        config: Fully resolved configuration; its hash is recorded.
        target: One of ``cpu-p``, ``cpu-lpe``, ``igpu``, ``npu``, ``cloud``.
        workload: Must carry ``kind`` and ``benchmark``; ``task_ids``, ``seed``, ``n_repeats``
            optional.
        condition_label: The true condition. Written to the manifest but **never read by
            analysis**, which consumes ``blinded_label`` only.
        repo_root: Repository root.
        allow_dirty: Permit a dirty git tree. Recorded in the manifest when used.
        summary: Written to ``summary.json`` before the manifest, so it is covered by
            ``integrity.raw_sha256``.
        seal: Seal the run directory on success. Only a run that writes more data afterwards should
            pass False, and it must seal explicitly.
        before_integrity_hash: Optional callback invoked after the run directory and event sink
            exist and **before** ``summary.json`` / ``raw_sha256`` are written. Use it to place
            data outputs (e.g. ``samples.ndjson``) so they are covered by the integrity hash.
            May return a dict merged into ``summary``.
        retro_seal: Post-hoc promote into ``raw/``. Refuses measurement ``power_state`` from
            promote-time sampling unless ``measurement_power_from_records=True`` (AM-036).

    Returns:
        A :class:`RunHandle`.

    Raises:
        DirtyTreeError: Dirty tree without ``allow_dirty``.
        ProvenanceError: A declared provenance artifact is missing, or retro-seal power leak.
        ManifestValidationError: The assembled manifest is not schema-valid.
    """
    _refuse_retro_seal_measurement_power(
        retro_seal=retro_seal,
        power_state=power_state,
        measurement_power_from_records=measurement_power_from_records,
    )
    # Resolved before anything is created on disk. An undeclared or contradicted mode should stop
    # the run outright, not leave a half-built run directory behind for someone to puzzle over.
    resolved_mode, isolation_evidence = resolve_isolation_mode(isolation_mode)
    resolved_launch, placement = resolve_launch_context(
        launch_context, required=require_launch_context
    )
    if session_id is None:
        session_id = placement.get("session_id")
    if window_station is None:
        window_station = placement.get("window_station")
    launch_context = resolved_launch

    git_state = capture_git_state(cwd=repo_root)
    assert_clean_or_allowed(git_state, allow_dirty=allow_dirty)

    salt = get_or_create_salt(repo_root)
    blinded_label = blinded_label_for(condition_label, salt=salt)

    if existing_run_dir is None:
        run_id = run_id or str(uuid.uuid4())
        run_dir = create_run_dir(run_id, repo_root=repo_root)
    else:
        if run_id is not None and run_id != existing_run_dir.run_id:
            raise RawStoreError(
                f"run_id {run_id} does not match existing run {existing_run_dir.run_id}"
            )
        run_id = existing_run_dir.run_id
        run_dir = existing_run_dir
        if run_dir.is_sealed():
            raise RawStoreError(f"cannot emit into sealed run {run_id}")
        for final_name in (_SUMMARY_FILENAME, _MANIFEST_FILENAME):
            if (run_dir.path / final_name).exists():
                raise RawStoreError(f"cannot finalize run {run_id}: {final_name} already exists")

    # Route structured events into the run directory from here on, so the log of what happened is
    # part of the sealed artifact rather than lost to the console.
    add_json_sink(run_dir.path / _EVENTS_FILENAME)

    log_event(
        "manifest.emit_started",
        message=f"emitting manifest for run {run_id}",
        run_id=run_id,
        target=target,
        workload_kind=workload.get("kind"),
        benchmark=workload.get("benchmark"),
        blinded_label=blinded_label,
        git_sha=git_state.sha,
        git_dirty=git_state.dirty,
        allow_dirty=allow_dirty,
        config_hash=config.config_hash,
    )

    summary_obj: dict[str, Any] = dict(summary) if summary is not None else {}
    if before_integrity_hash is not None:
        extra = before_integrity_hash(run_dir)
        if extra:
            summary_obj.update(extra)

    run_dir.write_json(_SUMMARY_FILENAME, summary_obj)

    # `integrity.raw_sha256` covers the run's DATA outputs only. Two files are necessarily excluded:
    #   - manifest.json, because a hash stored inside a file cannot cover that file;
    #   - events.ndjson, because emitting this manifest is itself a logged event, so the log keeps
    #     growing after the hash is taken.
    # The seal marker separately records a tree hash over everything except itself, which is what
    # verify_sealed() checks. So the data and the full run are both checksummed - by two different
    # values, for two different purposes.
    raw_sha256 = sha256_tree(run_dir.path, exclude={_MANIFEST_FILENAME, _EVENTS_FILENAME})

    manifest = build_manifest(
        run_id=run_id,
        config=config,
        git_state=git_state,
        allow_dirty=allow_dirty,
        target=target,
        workload=workload,
        condition_label=condition_label,
        blinded_label=blinded_label,
        repo_root=repo_root,
        model=model,
        drivers=drivers,
        power_state=power_state,
        thermal=thermal,
        topology_override=topology_override,
        outputs=outputs,
        raw_sha256=raw_sha256,
        self_check=self_check,
        isolation_mode=resolved_mode,
        isolation_evidence=isolation_evidence,
        launch_context=launch_context,
        session_id=session_id,
        window_station=window_station,
        require_launch_context=require_launch_context,
        retro_seal=retro_seal,
        measurement_power_from_records=measurement_power_from_records,
        promote_time_power_state=promote_time_power_state,
        power_state_note=power_state_note,
    )

    # Validate before writing. An invalid manifest must never reach raw/, because raw/ is
    # write-once and a bad manifest there could not be corrected.
    validate_manifest(manifest)

    run_dir.write_json(_MANIFEST_FILENAME, manifest)
    record_unblind_entry(condition_label, blinded_label, repo_root=repo_root)

    log_event(
        "manifest.emitted",
        message=f"manifest written for run {run_id}",
        run_id=run_id,
        raw_sha256=raw_sha256,
        path=str(run_dir.path / _MANIFEST_FILENAME),
    )

    if seal:
        # Deregister the sink first: raw/ is write-once, so an event appended after the seal would
        # invalidate the recorded tree hash and hit a read-only file.
        remove_json_sink(run_dir.path / _EVENTS_FILENAME)
        run_dir.seal(self_check=self_check)

    return RunHandle(
        run_id=run_id,
        manifest=manifest,
        run_dir=run_dir,
        raw_sha256=raw_sha256,
    )


def emit(
    *,
    config: ResolvedConfig,
    target: str,
    workload: dict[str, Any],
    condition_label: str,
    repo_root: Path,
    run_id: str | None = None,
    existing_run_dir: RunDir | None = None,
    allow_dirty: bool = False,
    summary: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    drivers: dict[str, Any] | None = None,
    power_state: dict[str, Any] | None = None,
    thermal: dict[str, Any] | None = None,
    topology_override: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    self_check: str = "pass",
    seal: bool = True,
    isolation_mode: str | None = None,
    launch_context: str | None = None,
    session_id: int | None = None,
    window_station: str | None = None,
    require_launch_context: bool = False,
    before_integrity_hash: Callable[[RunDir], dict[str, Any] | None] | None = None,
    retro_seal: bool = False,
    measurement_power_from_records: bool = False,
    promote_time_power_state: dict[str, Any] | None = None,
    power_state_note: str | None = None,
) -> RunHandle:
    """Emit under the shared raw-store lock.

    The run directory is unique, but creation, unblind-ledger update, and sealing touch shared
    repository paths.  One lock covers the complete transaction so no sealed run can be observed
    half-written.
    """
    with exclusive(repo_root / "raw"):
        return _emit_locked_body(
            config=config,
            target=target,
            workload=workload,
            condition_label=condition_label,
            repo_root=repo_root,
            run_id=run_id,
            existing_run_dir=existing_run_dir,
            allow_dirty=allow_dirty,
            summary=summary,
            model=model,
            drivers=drivers,
            power_state=power_state,
            thermal=thermal,
            topology_override=topology_override,
            outputs=outputs,
            self_check=self_check,
            seal=seal,
            isolation_mode=isolation_mode,
            launch_context=launch_context,
            session_id=session_id,
            window_station=window_station,
            require_launch_context=require_launch_context,
            before_integrity_hash=before_integrity_hash,
            retro_seal=retro_seal,
            measurement_power_from_records=measurement_power_from_records,
            promote_time_power_state=promote_time_power_state,
            power_state_note=power_state_note,
        )
