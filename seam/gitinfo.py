"""Git state capture for run manifests.

``git_sha`` is the field that makes a run reproducible, so this module fails loudly rather than
degrading. There is no "unknown" git state: if git cannot be interrogated, the run does not
happen.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from seam.errors import DirtyTreeError, GitError
from seam.jsonlog import log_event

__all__ = ["GitState", "capture_git_state", "repo_root"]

#: Git can hang on a lock or a credential prompt. Bound it so a run fails instead of stalling.
_TIMEOUT_S: Final = 30


@dataclass(frozen=True, slots=True)
class GitState:
    """Immutable snapshot of the working tree's git state."""

    sha: str
    dirty: bool
    branch: str | None
    #: ``git status --porcelain`` output when dirty. Recorded so a ``--allow-dirty`` run states
    #: exactly *what* was uncommitted, rather than only that something was.
    dirty_files: tuple[str, ...]


def _run_git(args: list[str], cwd: Path) -> str:
    """Run a git command and return its stripped stdout.

    Raises:
        GitError: If git is missing, times out, or exits non-zero. Wrapped rather than propagated
            so callers have a single named failure to handle, and never a bare ``OSError``.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args)} timed out after {_TIMEOUT_S}s") from exc

    if completed.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def repo_root(start: Path | None = None) -> Path:
    """Return the repository root containing ``start``."""
    cwd = (start or Path.cwd()).resolve()
    return Path(_run_git(["rev-parse", "--show-toplevel"], cwd=cwd)).resolve()


def capture_git_state(*, cwd: Path | None = None) -> GitState:
    """Capture the current git SHA, dirty flag, branch, and uncommitted paths."""
    root = cwd.resolve() if cwd is not None else Path.cwd().resolve()
    sha = _run_git(["rev-parse", "HEAD"], cwd=root)
    porcelain = _run_git(["status", "--porcelain"], cwd=root)
    dirty_files = tuple(line for line in porcelain.splitlines() if line.strip())

    try:
        branch: str | None = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    except GitError:
        # Detached HEAD is a legitimate state, not an error; the SHA is what matters for
        # reproducibility. Recorded as an event so the null branch in the manifest is explained
        # rather than mysterious.
        log_event(
            "git.branch_unavailable",
            severity="warning",
            message="could not resolve branch name; recording null (detached HEAD?)",
            git_sha=sha,
        )
        branch = None

    return GitState(sha=sha, dirty=bool(dirty_files), branch=branch, dirty_files=dirty_files)


def assert_clean_or_allowed(state: GitState, *, allow_dirty: bool) -> None:
    """Enforce the clean-tree requirement (spec §7/M1).

    A dirty tree means ``git_sha`` does not describe the code that ran, so the run is not
    reproducible. ``allow_dirty`` permits it for development, but the fact is recorded in the
    manifest and logged, so it can never be an invisible choice.

    Raises:
        DirtyTreeError: If the tree is dirty and ``allow_dirty`` is False.
    """
    if not state.dirty:
        return

    if not allow_dirty:
        raise DirtyTreeError(
            f"working tree has {len(state.dirty_files)} uncommitted change(s); "
            f"commit them or pass --allow-dirty (which is recorded in the manifest). "
            f"Files: {', '.join(state.dirty_files[:10])}"
            + (" ..." if len(state.dirty_files) > 10 else "")
        )

    log_event(
        "git.dirty_tree_allowed",
        severity="warning",
        message=(
            "running on a dirty tree with --allow-dirty; this run is NOT reproducible from "
            "git_sha alone and is recorded as such in the manifest"
        ),
        git_sha=state.sha,
        n_dirty_files=len(state.dirty_files),
        dirty_files=list(state.dirty_files),
    )
