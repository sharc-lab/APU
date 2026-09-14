"""isolation_mode: mandatory declaration, contradiction refusal, and the never-pool rule.

These cover the field that exists because two identically-configured runs differed by 1.98x with
nothing in either manifest recording that one was measured on a busy machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from seam.config import ResolvedConfig
from seam.gitinfo import GitState
from seam.isolation import (
    ENV_VAR,
    IsolationModeError,
    assert_poolable,
    harness_termination_record,
    resolve_isolation_mode,
)
from seam.manifest import build_manifest, validate_manifest

QUIET: dict[str, Any] = {
    "contending_processes": [],
    "tier2_recorded_processes": [],
    "sshd_session_count": 1,
    "probe_error": None,
}
BUSY: dict[str, Any] = {
    "contending_processes": [
        {"name": "Cursor.exe", "pid": 4242, "working_set_bytes": 3_000_000_000}
    ],
    "tier2_recorded_processes": [],
    "sshd_session_count": 0,
    "probe_error": None,
}
TIER2_ONLY: dict[str, Any] = {
    "contending_processes": [],
    "tier2_recorded_processes": [
        {
            "name": "msedgewebview2.exe",
            "pid": 5555,
            "working_set_bytes": 50_000_000,
            "private_working_set_bytes": 40_000_000,
        }
    ],
    "sshd_session_count": 1,
    "probe_error": None,
}


def test_undeclared_mode_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """No default. An undeclared mode stops the run rather than guessing one."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(IsolationModeError, match="not declared"):
        resolve_isolation_mode(evidence=QUIET)


def test_environment_variable_is_a_valid_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    """How a detached launcher declares the mode for a process it cannot otherwise talk to."""
    monkeypatch.setenv(ENV_VAR, "remote")
    mode, evidence = resolve_isolation_mode(evidence=QUIET)
    assert mode == "remote"
    assert evidence["consistent_with_declaration"] is True


def test_unknown_mode_refuses() -> None:
    with pytest.raises(IsolationModeError, match="must be one of"):
        resolve_isolation_mode("quiet-ish", evidence=QUIET)


def test_remote_declaration_contradicted_by_a_resident_editor_refuses() -> None:
    """The claim is false, not merely mislabelled, so it is refused rather than recorded."""
    with pytest.raises(IsolationModeError) as excinfo:
        resolve_isolation_mode("remote", evidence=BUSY)
    # The refusal must name what to close; a bare rejection sends the operator hunting.
    assert "Cursor.exe" in str(excinfo.value)
    assert "4242" in str(excinfo.value)
    assert "tier-1" in str(excinfo.value)


def test_remote_accepts_tier2_only_residents() -> None:
    """Tier-2 shell/vendor agents are recorded; they must not refuse remote mode."""
    mode, evidence = resolve_isolation_mode("remote", evidence=TIER2_ONLY)
    assert mode == "remote"
    assert evidence["consistent_with_declaration"] is True
    assert evidence["contending_processes"] == []
    assert len(evidence["tier2_recorded_processes"]) == 1
    assert evidence["tier2_recorded_processes"][0]["name"] == "msedgewebview2.exe"


def test_local_declaration_accepts_a_busy_machine() -> None:
    """`local` is the honest label for a machine in use, so it is never contradicted."""
    mode, evidence = resolve_isolation_mode("local", evidence=BUSY)
    assert mode == "local"
    assert evidence["consistent_with_declaration"] is True


def test_pooling_across_modes_refuses() -> None:
    with pytest.raises(IsolationModeError, match="different isolation modes"):
        assert_poolable(
            [
                {"run_id": "run-a", "isolation_mode": "remote"},
                {"run_id": "run-b", "isolation_mode": "local"},
            ]
        )


def test_pooling_refuses_a_run_that_predates_the_field() -> None:
    """An unknown machine state is exactly what cannot be pooled; absence is not agreement."""
    with pytest.raises(IsolationModeError, match="records no isolation_mode"):
        assert_poolable([{"run_id": "run-a", "isolation_mode": "remote"}, {"run_id": "run-old"}])


def test_pooling_within_one_mode_returns_it() -> None:
    assert (
        assert_poolable(
            [
                {"run_id": "run-a", "isolation_mode": "remote"},
                {"run_id": "run-b", "isolation_mode": "remote"},
            ]
        )
        == "remote"
    )


def test_harness_termination_record_is_refuse_only() -> None:
    """Remote mode refuses contending software; it does not invent kill timestamps."""
    record = harness_termination_record()
    assert record["harness_did_not_terminate"] is True
    assert record["operator_kill_unobservable"] is True
    assert record["seconds_since_harness_interactive_termination"] is None


def _manifest(
    config: ResolvedConfig,
    git_state: GitState,
    workload: dict[str, Any],
    repo_root: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return build_manifest(
        run_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        config=config,
        git_state=git_state,
        allow_dirty=False,
        target="cloud",
        workload=workload,
        condition_label="unit-test",
        blinded_label="cond_abcd",
        repo_root=repo_root,
        **kwargs,
    )


def test_manifest_records_the_declared_mode(
    fake_config: ResolvedConfig,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
    fake_repo: Path,
) -> None:
    manifest = _manifest(
        fake_config,
        clean_git_state,
        minimal_workload,
        fake_repo,
        isolation_mode="remote",
        isolation_evidence=QUIET,
    )
    assert manifest["isolation_mode"] == "remote"
    assert manifest["isolation_evidence"]["contending_processes"] == []
    validate_manifest(manifest)


def test_schema_rejects_remote_with_contending_processes(
    fake_config: ResolvedConfig,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
    fake_repo: Path,
) -> None:
    """Enforced in the schema as well as the emitter, so raw/ cannot hold the contradiction."""
    manifest = _manifest(
        fake_config,
        clean_git_state,
        minimal_workload,
        fake_repo,
        isolation_mode="local",
        isolation_evidence=BUSY,
    )
    manifest["isolation_mode"] = "remote"
    with pytest.raises(Exception, match="expected to be empty"):
        validate_manifest(manifest)


def test_schema_requires_the_field(
    fake_config: ResolvedConfig,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
    fake_repo: Path,
) -> None:
    manifest = _manifest(
        fake_config,
        clean_git_state,
        minimal_workload,
        fake_repo,
        isolation_mode="local",
        isolation_evidence=BUSY,
    )
    del manifest["isolation_mode"]
    with pytest.raises(Exception, match="isolation_mode"):
        validate_manifest(manifest)
