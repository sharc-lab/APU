"""Session-wide pytest configuration.

Fails immediately when the interpreter is below the declared Python floor
(requires-python = ">=3.12" in pyproject.toml).  A clear error here is
preferable to cryptic collection failures on unsupported syntax.
"""

from __future__ import annotations

import sys


def pytest_configure(config):
    if sys.version_info < (3, 12):
        import pytest
        pytest.exit(
            reason=(
                f"Python {sys.version_info.major}.{sys.version_info.minor} is below the "
                f"required floor (>=3.12).\n"
                "Install Python 3.12+ and re-run: py -3.12 -m pytest\n"
                f"Detected: {sys.version}"
            ),
            returncode=2,
        )


# Committed result files that tests must never write (operator 2026-10-31): hashed at session start, compared at
# session end; a change fails the run.
_GUARDED = ("results/per_task_budget_check_v1.jsonl",)
_GUARD_HASH: dict = {}


def _hash(rel):
    import hashlib
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def pytest_sessionstart(session):
    for rel in _GUARDED:
        _GUARD_HASH[rel] = _hash(rel)


def pytest_sessionfinish(session, exitstatus):
    changed = [rel for rel in _GUARDED if _hash(rel) != _GUARD_HASH.get(rel)]
    if changed:
        print(f"\nERROR: committed result file(s) modified by the test session: {changed}", file=sys.stderr)
        session.exitstatus = 1
