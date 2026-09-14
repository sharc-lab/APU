"""AM-036: measurement vs promote-time power_state; retro-seal refuse; corrections."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from seam.config import ResolvedConfig
from seam.errors import ProvenanceError
from seam.gitinfo import GitState
from seam.manifest import (
    build_manifest,
    emit,
    load_run_manifest,
    load_schema,
    validate_manifest,
)
from seam.manifest_corrections import (
    NULL_MEASUREMENT_POWER_STATE,
    detect_promote_time_power_leak,
)

VERIFIED_TOPOLOGY = {"p_cpus": [0, 1, 2, 3], "lpe_cpus": [4, 5, 6, 7], "verified": True}

LEAKED_RUNS = (
    "b5ce21e5-9f29-46f4-8319-f74adcdeb628",
    "64e525e7-37df-4a7a-91e5-21419acf5dd2",
    "404dc3d0-1760-41a8-b9c5-d0d439b1a1fe",
)
SELF_SEALED_OK = "693b44d2-8234-453c-bc8d-107a9ff259a0"
DELTA_NULL_RUNS = (
    "d5c98342-a0b2-41a9-b6e2-93ac7a39c3ba",
    "9f38eb15-6fe6-40b4-871b-a02ec5629bb1",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build(
    config: ResolvedConfig,
    repo_root: Path,
    git_state: GitState,
    workload: dict[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
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
        "topology_override": VERIFIED_TOPOLOGY,
    }
    kwargs.update(overrides)
    return build_manifest(**kwargs)


def test_schema_separates_promote_time_power_state() -> None:
    load_schema.cache_clear()
    props = load_schema()["properties"]
    assert "promote_time_power_state" in props
    assert "power_state_note" in props
    assert "retro_seal" in props
    assert "measurement_power_from_records" in props
    assert "MEASUREMENT-TIME" in props["power_state"]["description"]
    assert "MUST NOT" in props["promote_time_power_state"]["description"]


def test_retro_seal_refuses_promote_time_sample_as_measurement(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    with pytest.raises(ProvenanceError, match="retro_seal=True refuses"):
        _build(
            fake_config,
            fake_repo,
            clean_git_state,
            minimal_workload,
            retro_seal=True,
            power_state={"on_battery": True, "battery_pct_start": 88.0},
        )


def test_retro_seal_allows_null_measurement_and_promote_time_block(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    load_schema.cache_clear()
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        retro_seal=True,
        power_state=None,
        promote_time_power_state={
            "on_battery": True,
            "battery_pct_start": 88.0,
            "captured_at_utc": "2026-08-10T13:05:32.514660+00:00",
        },
        power_state_note="unrecorded measurement power",
    )
    validate_manifest(manifest)
    assert manifest["power_state"]["on_battery"] is None
    assert manifest["power_state"]["battery_pct_start"] is None
    assert manifest["promote_time_power_state"]["on_battery"] is True
    assert manifest["power_state_note"] == "unrecorded measurement power"
    assert manifest["retro_seal"] is True
    assert manifest["measurement_power_from_records"] is None


def test_retro_seal_allows_measurement_from_records_flag(
    fake_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
) -> None:
    load_schema.cache_clear()
    manifest = _build(
        fake_config,
        fake_repo,
        clean_git_state,
        minimal_workload,
        retro_seal=True,
        measurement_power_from_records=True,
        power_state={"on_battery": False, "battery_pct_start": 100.0},
    )
    validate_manifest(manifest)
    assert manifest["power_state"]["on_battery"] is False
    assert manifest["measurement_power_from_records"] is True
    assert not detect_promote_time_power_leak(manifest, summary={"promotion": "post_hoc_x"})


def test_emit_retro_seal_nulls_measurement_and_keeps_promote_time(
    verified_config: ResolvedConfig,
    fake_repo: Path,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
    patch_git: Any,
) -> None:
    """Full emit path: promote-only / retro_seal cannot leak live samples into power_state."""
    load_schema.cache_clear()
    patch_git(clean_git_state)
    promote = {
        "on_battery": True,
        "battery_pct_start": 88.0,
        "captured_at_utc": "2026-08-10T13:05:32.514660+00:00",
    }
    handle = emit(
        config=verified_config,
        target="cpu-p",
        workload=minimal_workload,
        condition_label="retro_seal_unit",
        repo_root=fake_repo,
        summary={"promotion": "post_hoc_from_derived_diagnostic", "note": "synthetic"},
        power_state=None,
        promote_time_power_state=promote,
        power_state_note="synthetic retro-seal; measurement power unrecorded",
        retro_seal=True,
        isolation_mode="local",
    )
    on_disk = json.loads((handle.run_dir.path / "manifest.json").read_text(encoding="utf-8-sig"))
    validate_manifest(on_disk)
    assert on_disk["retro_seal"] is True
    assert on_disk["power_state"]["on_battery"] is None
    assert on_disk["power_state"]["battery_pct_start"] is None
    assert on_disk["promote_time_power_state"]["on_battery"] is True
    assert on_disk["promote_time_power_state"]["battery_pct_start"] == 88.0

    with pytest.raises(ProvenanceError, match="retro_seal=True refuses"):
        emit(
            config=verified_config,
            target="cpu-p",
            workload=minimal_workload,
            condition_label="retro_seal_leak_attempt",
            repo_root=fake_repo,
            power_state={"on_battery": True, "battery_pct_start": 88.0},
            retro_seal=True,
            isolation_mode="local",
        )


@pytest.mark.parametrize("run_id", LEAKED_RUNS)
def test_loader_nulls_leaked_ceiling_a_measurement_power(run_id: str) -> None:
    sealed = json.loads(
        (REPO_ROOT / "raw" / run_id / "manifest.json").read_text(encoding="utf-8-sig")
    )
    # Sealed bytes still carry the leak (write-once; not mutated).
    assert sealed["power_state"]["on_battery"] is True
    assert sealed["power_state"]["battery_pct_start"] == 88.0

    loaded = load_run_manifest(run_id, repo_root=REPO_ROOT)
    for key, value in NULL_MEASUREMENT_POWER_STATE.items():
        assert loaded["power_state"][key] is value
    assert loaded["promote_time_power_state"]["on_battery"] is True
    assert loaded["promote_time_power_state"]["battery_pct_start"] == 88.0
    assert "predates measurement-time powerstate capture" in loaded["power_state_note"]
    assert loaded["_manifest_correction"]["raw_bytes_unmutated"] is True


def test_structural_rule_corrects_without_registry_entry(tmp_path: Path) -> None:
    """Default path is structural - a future leak must not require adding a run_id."""
    run_id = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    raw_dir = tmp_path / "raw" / run_id
    raw_dir.mkdir(parents=True)
    # Minimal sealed-shaped artifacts: leak in power_state, post_hoc summary, no promote_time.
    donor = REPO_ROOT / "raw" / LEAKED_RUNS[0] / "manifest.json"
    manifest = json.loads(donor.read_text(encoding="utf-8-sig"))
    manifest["run_id"] = run_id
    manifest["promote_time_power_state"] = None
    manifest["power_state_note"] = None
    manifest.pop("retro_seal", None)
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (raw_dir / "summary.json").write_text(
        json.dumps({"promotion": "post_hoc_from_derived_diagnostic_partial"}),
        encoding="utf-8",
    )
    # No registry under tmp_path/derived - structural path only.
    assert detect_promote_time_power_leak(
        manifest, summary={"promotion": "post_hoc_from_derived_diagnostic_partial"}
    )
    loaded = load_run_manifest(run_id, repo_root=tmp_path)
    assert loaded["power_state"]["on_battery"] is None
    assert loaded["promote_time_power_state"]["on_battery"] is True
    assert loaded["_manifest_correction"]["source"] == "structural"


def test_self_sealed_693b44d2_remains_correct() -> None:
    loaded = load_run_manifest(SELF_SEALED_OK, repo_root=REPO_ROOT)
    assert loaded["power_state"]["on_battery"] is False
    assert loaded["power_state"]["battery_pct_start"] == 100.0
    assert "_manifest_correction" not in loaded
    sealed = json.loads(
        (REPO_ROOT / "raw" / SELF_SEALED_OK / "manifest.json").read_text(encoding="utf-8-sig")
    )
    assert not detect_promote_time_power_leak(
        sealed,
        summary=json.loads(
            (REPO_ROOT / "raw" / SELF_SEALED_OK / "summary.json").read_text(encoding="utf-8-sig")
        ),
    )


@pytest.mark.parametrize("run_id", DELTA_NULL_RUNS)
def test_delta_prefill_promotes_stay_null_measurement_power(run_id: str) -> None:
    loaded = load_run_manifest(run_id, repo_root=REPO_ROOT)
    assert loaded["power_state"]["on_battery"] is None
    assert loaded["power_state"]["battery_pct_start"] is None
    assert "_manifest_correction" not in loaded


def test_registry_lists_all_three_af036_digests() -> None:
    registry = json.loads(
        (REPO_ROOT / "derived" / "manifest_corrections" / "registry.json").read_text(
            encoding="utf-8-sig"
        )
    )
    corrections = registry["corrections"]
    for run_id in LEAKED_RUNS:
        assert run_id in corrections
        assert corrections[run_id]["kind"] == "null_measurement_power_promote_leak"
