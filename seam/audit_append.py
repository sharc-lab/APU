"""Append-only writes to ``AUDIT_LOG.md`` under §6.6 mutual exclusion."""

from __future__ import annotations

from pathlib import Path

from seam.gitinfo import repo_root
from seam.locks import exclusive

__all__ = ["append_audit"]


def append_audit(text: str, *, path: Path | None = None) -> None:
    """Append ``text`` to the audit log, refusing if another writer holds the lock."""
    root = repo_root(Path(__file__).parent)
    target = path or (root / "AUDIT_LOG.md")
    with exclusive(target), target.open("a", encoding="utf-8", newline="\n") as handle:
        if not text.endswith("\n"):
            text += "\n"
        handle.write(text)
