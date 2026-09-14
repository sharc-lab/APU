"""Manifest emission and schema-validation tests (spec §7/M1 acceptance).

Covers the two M1 criteria that concern the manifest:

* ``seam.manifest.emit()`` produces a schema-valid manifest, enforced by a JSON-Schema test.
* A dirty git tree is refused unless ``--allow-dirty`` is passed, and passing it is recorded.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from seam.config import ResolvedConfig
from seam.errors import DirtyTreeError, ManifestValidationError, ProvenanceError
from seam.gitinfo import GitState
from seam.manifest import (
    SCHEMA_PATH,
    build_manifest,
    emit,
    load_schema,
    validate_manifest,
)

VERIFIED_TOPOLOGY = {"p_cpus": [0, 1, 2, 3], "lpe_cpus": [4, 5, 6, 7], "verified": True}


def _build(
    config: ResolvedConfig,
    repo_root: Path,
    git_state: GitState,
    workload: dict[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
    """Build a manifest with sensible test defaults."""
    kwargs: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "config": config,
        "git_state": git_state,
        "allow_dirty": git_state.dirty,
        "target": "npu",
        "workload": workload,
        "condition_label": "A",
        "blinded_label": "cond_0123456789ab",
        "repo_root": repo_root,
    }
    kwargs.update(overrides)
    return build_manifest(**kwargs)


# ==================================================================================================
# Schema hygiene
# ==================================================================================================


def test_schema_file_exists_and_is_valid_json_schema() -> None:
    assert SCHEMA_PATH.is_file(), f"manifest schema missing at {SCHEMA_PATH}"
    schema = load_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["spec_version"]["const"] == "1.0"


def test_schema_requires_every_field_named_in_the_spec() -> None:
    """Guard against a field being quietly dropped from the required set.

    The list is spec §6.1's schema plus ``allow_dirty``, which M1 requires to be recorded.
    """
    expected_required = {
        "run_id",
        "spec_version",
        "timestamp_utc",
        "git_sha",
        "git_dirty",
        "allow_dirty",
        "config_hash",
        "elevated",
        "platform",
        "drivers",
        "power_state",
        "thermal",
        "target",
        "model",
        "npu_config",
        "workload",
        "condition_label",
        "blinded_label",
        "outputs",
        "integrity",
    }
    assert expected_required <= set(load_schema()["required"])


@pytest.mark.parametrize(
    ("block", "fields"),
    [
        ("platform", {"id", "cpu", "family", "topology", "os_build", "provenance_artifacts"}),
        ("drivers", {"npu", "igpu", "openvino", "genai", "lhm_bridge"}),
        (
            "power_state",
            {
                "on_battery",
                "battery_pct_start",
                "battery_pct_end",
                "charging",
                "power_plan",
                "display_brightness",
                "defender_realtime",
                "windows_update_paused",
            },
        ),
        (
            "thermal",
            {
                "ambient_c",
                "warmup_s",
                "cooldown_ceiling_c",
                "throttle_residency_pct",
                "throttle_threshold_pct",
                "excluded",
            },
        ),
        ("model", {"name", "revision", "quantization", "ir_sha256", "provenance"}),
        ("npu_config", {"MAX_PROMPT_LEN", "NPUW_LLM_PREFILL_CHUNK_SIZE"}),
        ("workload", {"kind", "benchmark", "task_ids", "seed", "n_repeats"}),
        ("outputs", {"samples", "steps", "summary"}),
        ("integrity", {"self_check", "raw_sha256"}),
    ],
)
def test_schema_nested_blocks_require_their_spec_fields(block: str, fields: set[str]) -> None:
    assert fields <= set(load_schema()["properties"][block]["required"])


def test_schema_rejects_unknown_top_level_field(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    """``additionalProperties: false`` means a field cannot appear without a schema change."""
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        topology_override=VERIFIED_TOPOLOGY,
    )
    manifest["undeclared_field"] = "sneaked in"
    with pytest.raises(ManifestValidationError, match="undeclared_field"):
        validate_manifest(manifest)


# ==================================================================================================
# build_manifest
# ==================================================================================================


def test_built_manifest_is_schema_valid(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        topology_override=VERIFIED_TOPOLOGY,
    )
    validate_manifest(manifest)


def test_manifest_hashes_provenance_artifacts_at_build_time(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    """Blueprint AF-002: hashes come from bytes on disk, not from a transcribed constant."""
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        topology_override=VERIFIED_TOPOLOGY,
    )
    artifacts = manifest["platform"]["provenance_artifacts"]

    assert artifacts, "expected at least one provenance artifact"
    assert all(len(entry["sha256"]) == 64 for entry in artifacts)

    # Changing an artifact's bytes must change its recorded hash.
    target = fake_repo / artifacts[0]["path"]
    before = artifacts[0]["sha256"]
    target.write_text("mutated content\n", encoding="utf-8")
    after = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        topology_override=VERIFIED_TOPOLOGY,
    )["platform"]["provenance_artifacts"][0]["sha256"]
    assert after != before


def test_missing_provenance_artifact_is_a_hard_failure(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    """A missing artifact must invalidate the run, not degrade to a null hash."""
    declared = fake_config.get("provenance_artifacts")
    (fake_repo / declared[0]).unlink()
    with pytest.raises(ProvenanceError, match="missing"):
        _build(
            fake_config,
            fake_repo,
            clean_git_state,
            minimal_workload,
            topology_override=VERIFIED_TOPOLOGY,
        )


def test_thermal_constants_are_null_not_defaulted(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    """AMENDMENTS.md AM-006: spec §6.1's thermal numbers are examples, not defaults.

    A manifest that reported ``warmup_s: 120`` before M2.3 measured it would be an untraceable
    number, which spec §9.2 forbids.
    """
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        topology_override=VERIFIED_TOPOLOGY,
    )
    thermal = manifest["thermal"]
    assert thermal["warmup_s"] is None
    assert thermal["cooldown_ceiling_c"] is None
    assert thermal["throttle_threshold_pct"] is None
    assert thermal["excluded"] is False


def test_cpu_target_requires_verified_topology(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    """A cpu-p run is meaningless without a verified core mapping, so the schema forbids it."""
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        target="cpu-p",
        topology_override={"p_cpus": None, "lpe_cpus": None, "verified": False},
    )
    with pytest.raises(ManifestValidationError):
        validate_manifest(manifest)


def test_topology_verify_may_run_before_the_topology_is_verified(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
) -> None:
    """AF-006 / AM-010: the run that PRODUCES the mapping is exempt from requiring one.

    Without this, a refused verification could not emit a manifest at all, and its per-CPU scores
    would be unciteable - the exact collision with spec §9.2 that AF-006 records.
    """
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        {
            "kind": "topology_verify",
            "benchmark": "python_intfp_v1",
            "task_ids": [],
            "seed": None,
            "n_repeats": 7,
        },
        target="cpu-p",
        topology_override={"p_cpus": None, "lpe_cpus": None, "verified": False},
        self_check="fail",
    )
    validate_manifest(manifest)


def test_refused_topology_verify_must_record_a_failing_verdict(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
) -> None:
    """The exemption must not let a refusal be recorded as a success.

    A manifest that says "topology not verified" while reporting ``self_check: pass`` would read as
    a successful verification to anyone scanning ``raw/``.
    """
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        {
            "kind": "topology_verify",
            "benchmark": "python_intfp_v1",
            "task_ids": [],
            "seed": None,
            "n_repeats": 7,
        },
        target="cpu-p",
        topology_override={"p_cpus": None, "lpe_cpus": None, "verified": False},
        self_check="pass",
    )
    with pytest.raises(ManifestValidationError, match="self_check"):
        validate_manifest(manifest)


def test_the_exemption_is_keyed_on_workload_kind_not_on_target(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    """A measurement workload must not inherit the topology-verification exemption."""
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        target="cpu-lpe",
        topology_override={"p_cpus": None, "lpe_cpus": None, "verified": False},
        self_check="fail",
    )
    with pytest.raises(ManifestValidationError):
        validate_manifest(manifest)


@pytest.mark.parametrize("kind", ["microbench", "aa", "h1_pilot", "topology_verify"])
def test_schema_accepts_every_declared_workload_kind(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    kind: str,
) -> None:
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        {"kind": kind, "benchmark": "b", "task_ids": [], "seed": None, "n_repeats": 1},
        topology_override=VERIFIED_TOPOLOGY,
    )
    validate_manifest(manifest)


def test_schema_rejects_an_unregistered_workload_kind(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
) -> None:
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        {"kind": "adhoc", "benchmark": "b", "task_ids": [], "seed": None, "n_repeats": 1},
        topology_override=VERIFIED_TOPOLOGY,
    )
    with pytest.raises(ManifestValidationError, match="kind"):
        validate_manifest(manifest)


def test_verified_topology_must_name_its_cpus(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        topology_override={"p_cpus": None, "lpe_cpus": None, "verified": True},
    )
    with pytest.raises(ManifestValidationError):
        validate_manifest(manifest)


def test_dirty_tree_manifest_must_record_allow_dirty(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    dirty_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    """It must be structurally impossible to emit a manifest that hides a dirty tree."""
    manifest = _build(
        fake_config,
        fake_repo,
        dirty_git_state,
        minimal_workload,
        allow_dirty=False,
        topology_override=VERIFIED_TOPOLOGY,
    )
    assert manifest["git_dirty"] is True
    with pytest.raises(ManifestValidationError):
        validate_manifest(manifest)


@pytest.mark.parametrize("bad_target", ["dgpu", "gpu", "cpu", ""])
def test_schema_rejects_targets_outside_the_five(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
    bad_target: str,
) -> None:
    """Platform A has no dGPU; the target set is exactly four local plus cloud."""
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        target=bad_target,
        topology_override=VERIFIED_TOPOLOGY,
    )
    with pytest.raises(ManifestValidationError):
        validate_manifest(manifest)


def test_schema_rejects_non_uuid4_run_id(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        run_id="not-a-uuid",
        topology_override=VERIFIED_TOPOLOGY,
    )
    with pytest.raises(ManifestValidationError, match="run_id"):
        validate_manifest(manifest)


# ==================================================================================================
# emit()
# ==================================================================================================


def test_emit_writes_schema_valid_manifest_and_seals_the_run(
    verified_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
    patch_git: Any,
) -> None:
    patch_git(clean_git_state)

    handle = emit(
        config=verified_config,
        target="cpu-p",
        workload=minimal_workload,
        condition_label="A",
        repo_root=fake_repo,
        summary={"note": "unit test"},
    )

    manifest_path = handle.run_dir.path / "manifest.json"
    assert manifest_path.is_file()

    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(on_disk)
    assert on_disk["run_id"] == handle.run_id
    assert on_disk["git_sha"] == clean_git_state.sha
    assert on_disk["git_dirty"] is False
    assert on_disk["allow_dirty"] is False
    assert on_disk["config_hash"] == verified_config.config_hash
    assert on_disk["integrity"]["raw_sha256"] == handle.raw_sha256

    assert handle.run_dir.is_sealed()
    assert (handle.run_dir.path / "summary.json").is_file()


def test_emit_refuses_dirty_tree_without_allow_dirty(
    verified_config: ResolvedConfig,
    fake_repo: Path,
    dirty_git_state: GitState,
    minimal_workload: dict[str, Any],
    patch_git: Any,
) -> None:
    """M1 acceptance: dirty-tree refusal works."""
    patch_git(dirty_git_state)

    with pytest.raises(DirtyTreeError, match="uncommitted"):
        emit(
            config=verified_config,
            target="cpu-p",
            workload=minimal_workload,
            condition_label="A",
            repo_root=fake_repo,
        )

    # The refusal must happen before any run directory is created.
    raw = fake_repo / "raw"
    run_dirs = (
        [p for p in raw.iterdir() if p.is_dir() and not p.name.startswith("_")]
        if raw.is_dir()
        else []
    )
    assert run_dirs == []


def test_emit_records_allow_dirty_when_passed(
    verified_config: ResolvedConfig,
    fake_repo: Path,
    dirty_git_state: GitState,
    minimal_workload: dict[str, Any],
    patch_git: Any,
) -> None:
    """M1 acceptance: ``--allow-dirty`` is recorded, including *what* was uncommitted."""
    patch_git(dirty_git_state)

    handle = emit(
        config=verified_config,
        target="cpu-p",
        workload=minimal_workload,
        condition_label="A",
        repo_root=fake_repo,
        allow_dirty=True,
    )

    assert handle.manifest["git_dirty"] is True
    assert handle.manifest["allow_dirty"] is True
    assert handle.manifest["git_dirty_files"] == list(dirty_git_state.dirty_files)
    validate_manifest(handle.manifest)


def test_emit_produces_distinct_run_ids(
    verified_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
    patch_git: Any,
) -> None:
    patch_git(clean_git_state)
    ids = {
        emit(
            config=verified_config,
            target="cpu-p",
            workload=minimal_workload,
            condition_label="A",
            repo_root=fake_repo,
        ).run_id
        for _ in range(3)
    }
    assert len(ids) == 3


def test_emit_uses_preallocated_run_id(
    verified_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
    patch_git: Any,
) -> None:
    patch_git(clean_git_state)
    allocated = "2024c246-95fa-4f4b-905d-c4da0a28d23c"

    handle = emit(
        config=verified_config,
        target="cpu-p",
        workload=minimal_workload,
        condition_label="A",
        repo_root=fake_repo,
        run_id=allocated,
    )

    assert handle.run_id == allocated
    assert handle.manifest["run_id"] == allocated
    assert handle.run_dir.path.name == allocated


def test_emit_blinds_the_condition_label(
    verified_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
    patch_git: Any,
) -> None:
    """``blinded_label`` must not leak the condition, and must be stable for a given condition."""
    patch_git(clean_git_state)

    first = emit(
        config=verified_config,
        target="cpu-p",
        workload=minimal_workload,
        condition_label="treatment_gpt5",
        repo_root=fake_repo,
    ).manifest
    second = emit(
        config=verified_config,
        target="cpu-p",
        workload=minimal_workload,
        condition_label="treatment_gpt5",
        repo_root=fake_repo,
    ).manifest
    other = emit(
        config=verified_config,
        target="cpu-p",
        workload=minimal_workload,
        condition_label="reference",
        repo_root=fake_repo,
    ).manifest

    assert first["blinded_label"] == second["blinded_label"]
    assert first["blinded_label"] != other["blinded_label"]
    assert "gpt5" not in first["blinded_label"]
    assert first["blinded_label"].startswith("cond_")
