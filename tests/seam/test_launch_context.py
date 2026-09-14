"""launch_context / session_id / window_station on the run manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from seam.config import ResolvedConfig
from seam.gitinfo import GitState
from seam.launch_context import (
    LaunchContextError,
    assert_same_launch_context,
    resolve_launch_context,
)
from seam.manifest import build_manifest, validate_manifest


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
        isolation_mode="local",
        isolation_evidence={
            "contending_processes": [],
            "tier2_recorded_processes": [],
            "sshd_session_count": 0,
            "consistent_with_declaration": True,
            "probe_error": None,
        },
        **kwargs,
    )


def test_resolve_requires_declaration_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEAM_LAUNCH_CONTEXT", raising=False)
    with pytest.raises(LaunchContextError, match="not declared"):
        resolve_launch_context(required=True)


def test_resolve_accepts_ssh_detached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEAM_LAUNCH_CONTEXT", "ssh_detached")
    context, placement = resolve_launch_context(required=True)
    assert context == "ssh_detached"
    assert "session_id" in placement
    assert "window_station" in placement


def test_manifest_records_launch_fields(
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
        launch_context="ssh_detached",
        session_id=0,
        window_station="Service-0x0-test$",
        require_launch_context=True,
    )
    assert manifest["launch_context"] == "ssh_detached"
    assert manifest["session_id"] == 0
    assert manifest["window_station"] == "Service-0x0-test$"
    validate_manifest(manifest)


def test_manifest_allows_null_launch_fields_for_legacy(
    fake_config: ResolvedConfig,
    clean_git_state: GitState,
    minimal_workload: dict[str, Any],
    fake_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEAM_LAUNCH_CONTEXT", raising=False)
    manifest = _manifest(
        fake_config,
        clean_git_state,
        minimal_workload,
        fake_repo,
    )
    assert manifest["launch_context"] is None
    validate_manifest(manifest)


def test_assert_same_launch_context_refuses_span() -> None:
    with pytest.raises(LaunchContextError, match="different launch_context"):
        assert_same_launch_context(
            [
                {"run_id": "a", "launch_context": "ssh_detached"},
                {"run_id": "b", "launch_context": "local_console"},
            ]
        )


def test_assert_same_launch_context_allows_shared() -> None:
    assert (
        assert_same_launch_context(
            [
                {"run_id": "a", "launch_context": "ssh_detached"},
                {"run_id": "b", "launch_context": "ssh_detached"},
            ]
        )
        == "ssh_detached"
    )
