"""Git-state capture tests (spec §7/M1: dirty-tree refusal)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from seam.errors import DirtyTreeError, GitError
from seam.gitinfo import GitState, assert_clean_or_allowed, capture_git_state


def test_clean_tree_is_allowed(clean_git_state: GitState) -> None:
    assert_clean_or_allowed(clean_git_state, allow_dirty=False)  # must not raise


def test_dirty_tree_is_refused_by_default(dirty_git_state: GitState) -> None:
    with pytest.raises(DirtyTreeError, match="uncommitted"):
        assert_clean_or_allowed(dirty_git_state, allow_dirty=False)


def test_dirty_tree_error_names_the_offending_files(dirty_git_state: GitState) -> None:
    """The operator needs to know what to commit, not merely that something is uncommitted."""
    with pytest.raises(DirtyTreeError) as excinfo:
        assert_clean_or_allowed(dirty_git_state, allow_dirty=False)
    assert "seam/manifest.py" in str(excinfo.value)


def test_dirty_tree_is_allowed_with_the_flag(dirty_git_state: GitState) -> None:
    assert_clean_or_allowed(dirty_git_state, allow_dirty=True)  # must not raise


def test_allowing_a_dirty_tree_emits_an_auditable_event(
    dirty_git_state: GitState, tmp_path: Path
) -> None:
    """Spec §9.6 forbids a silent fallback, so the waiver must be recorded."""
    import json

    from seam.jsonlog import add_json_sink

    sink = tmp_path / "events.ndjson"
    add_json_sink(sink)

    assert_clean_or_allowed(dirty_git_state, allow_dirty=True)

    events = [json.loads(line) for line in sink.read_text(encoding="utf-8").splitlines()]
    dirty_events = [e for e in events if e["event"] == "git.dirty_tree_allowed"]
    assert len(dirty_events) == 1
    assert dirty_events[0]["severity"] == "warning"
    assert dirty_events[0]["n_dirty_files"] == 2
    assert dirty_events[0]["git_sha"] == dirty_git_state.sha


def test_truncated_file_list_is_marked(clean_git_state: GitState) -> None:
    """With many dirty files the message truncates; it must say so rather than mislead."""
    many = GitState(
        sha="c" * 40,
        dirty=True,
        branch="main",
        dirty_files=tuple(f" M file{i}.py" for i in range(25)),
    )
    with pytest.raises(DirtyTreeError) as excinfo:
        assert_clean_or_allowed(many, allow_dirty=False)
    assert "..." in str(excinfo.value)
    assert "25 uncommitted" in str(excinfo.value)


# ==================================================================================================
# Real git interaction
# ==================================================================================================


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


requires_git = pytest.mark.skipif(not _git_available(), reason="git not available on PATH")


@requires_git
def test_capture_git_state_on_a_real_repository(tmp_path: Path) -> None:
    """End-to-end against a throwaway repository, so the parsing is not only tested via mocks."""
    run = lambda *args: subprocess.run(  # noqa: E731
        ["git", *args], cwd=tmp_path, capture_output=True, check=True, text=True
    )
    run("init", "-q")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "SEAM test")
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    run("add", "a.txt")
    run("commit", "-q", "-m", "initial")

    clean = capture_git_state(cwd=tmp_path)
    assert len(clean.sha) == 40
    assert clean.dirty is False
    assert clean.dirty_files == ()

    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    dirty = capture_git_state(cwd=tmp_path)
    assert dirty.dirty is True
    assert any("b.txt" in entry for entry in dirty.dirty_files)
    assert dirty.sha == clean.sha, "an untracked file does not change HEAD"


@requires_git
def test_capture_git_state_fails_outside_a_repository(tmp_path: Path) -> None:
    """An unknown git state is a hard failure, never a null field in a manifest."""
    with pytest.raises(GitError):
        capture_git_state(cwd=tmp_path)
