"""Pre-commit hook: sealed run directories are write-once once tracked.

Fails if a staged change modifies or deletes a file that already exists under
a sealed evidence directory in HEAD. New sealed run directories may be added.

Sealed roots:
  - raw/<run_id>/
  - derived/**/sealed_<run_id>/

Exit 0 = clean, 1 = blocked.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = re.compile(r"^raw/[^/]+/.+")
SEALED_DERIVED = re.compile(r"^derived/.*/sealed_[^/]+/.+")


def _norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _in_head(rel: str) -> bool:
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def _is_sealed_path(rel: str) -> bool:
    if RAW_FILE.match(rel):
        # raw/_blinding/ is excluded from commits; still treat as protected if present.
        return True
    return bool(SEALED_DERIVED.match(rel))


def _staged_paths() -> list[str]:
    """Paths in the index that differ from HEAD (includes deletes)."""
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return []
    return [p for p in proc.stdout.decode("utf-8", errors="replace").split("\0") if p]


def scan(paths: list[str]) -> int:
    # No HEAD yet (orphan / empty) — nothing previously sealed in git.
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if head.returncode != 0:
        return 0

    # Prefer explicit paths (pre-commit pass_filenames), but always fall back to
    # the full staged set so deletions are caught when argv is empty or when
    # pre-commit omits a deleted path.
    candidates = [_norm(p) for p in paths] if paths else []
    staged = [_norm(p) for p in _staged_paths()]
    if not candidates:
        candidates = staged
    else:
        # Union so a filename-filtered run still sees sealed deletes in the index.
        seen = set(candidates)
        for p in staged:
            if p not in seen and _is_sealed_path(p):
                candidates.append(p)
                seen.add(p)

    violations: list[str] = []
    for rel in candidates:
        if not _is_sealed_path(rel):
            continue
        if not _in_head(rel):
            # New evidence file — allowed.
            continue
        # Exists in HEAD: any staged change is a sealed-tree mutation.
        violations.append(rel)

    if violations:
        sys.stderr.write("\nBLOCKED — sealed evidence must not be modified after seal:\n")
        for v in violations:
            sys.stderr.write(f"  {v}\n")
        sys.stderr.write(
            "\nraw/<run_id>/ and derived/**/sealed_*/ are write-once once tracked.\n"
            "Add a new run_id / sealed_* tree instead of editing an existing one.\n\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(scan(sys.argv[1:]))
