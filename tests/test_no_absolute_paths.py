"""Guard against absolute user-specific paths in committed source files.

The repo is public. Any path of the form C:\\Users\\<name>, /home/<name>,
/Users/<name>, or /c/Users/<name> in a committed source file leaks the
developer's username and cannot be recalled once pushed.

This test fails on the specific patterns that have appeared in this repo's
history, so fixing forward is caught immediately rather than silently
re-introduced.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Patterns that indicate a hard-coded user path.
# Anchored so that environment variable references like $HOME or %USERPROFILE%
# are not matched, only literal paths.
_PATTERNS = [
    re.compile(r"C:[/\\]Users[/\\]", re.IGNORECASE),
    re.compile(r"/c/Users/", re.IGNORECASE),
    re.compile(r"/home/[a-z_][a-z0-9_]{0,30}/", re.IGNORECASE),
    re.compile(r"/Users/[a-z_][a-z0-9_]{0,30}/", re.IGNORECASE),
]

# File extensions to check. Data files (*.json, *.jsonl) are excluded —
# result files may legitimately record provenance paths from the machine
# that generated them and cannot be changed without invalidating checksums.
_CHECKED_EXTENSIONS = {".py", ".yaml", ".yml", ".sh", ".toml", ".cfg", ".ini", ".md", ".txt"}

# Paths prefixed with these strings are skipped even when committed.
# .venv/ is gitignored but we skip it defensively.
# This file itself is excluded: it intentionally contains the patterns it
# searches for (in docstrings and regex literals) and cannot be self-checked.
_SKIP_PREFIXES = (".venv/", ".venv\\", "results/", "results\\")
_SKIP_EXACT = {
    "tests/test_no_absolute_paths.py",  "tests\\test_no_absolute_paths.py",
    # RESULT_PROVENANCE.md intentionally records the old paths (Before column)
    # in the path-redaction table so git history and audit trail are readable.
    # The redacted results/*.json files are covered by test_results_no_username.
    "docs/RESULT_PROVENANCE.md",        "docs\\RESULT_PROVENANCE.md",
}


def _committed_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    return result.stdout.splitlines()


def test_results_no_username() -> None:
    """results/*.json files must not contain the developer's literal username.

    Absolute paths in these files have been redacted to use %LOCALAPPDATA% and
    %USERPROFILE% tokens.  This test catches regressions where a result file is
    regenerated and the raw path is committed again.

    Only the username segment "rithw" is checked (not the general path patterns)
    because result JSON files may legitimately contain relative path fragments.
    The sha256 model digest must be preserved — this test does not match it.
    """
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results"
    if not results_dir.is_dir():
        return  # nothing to check

    username_pat = re.compile(r"[Uu]sers[/\\]rithw[/\\]")
    violations: list[str] = []

    for json_file in sorted(results_dir.glob("*.json")):
        try:
            text = json_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if username_pat.search(line):
                violations.append(f"results/{json_file.name}:{i}: {line.strip()[:120]}")

    assert not violations, (
        f"Username 'rithw' found in {len(violations)} result JSON location(s) "
        "(should be redacted to %LOCALAPPDATA% / %USERPROFILE%):\n"
        + "\n".join(violations)
    )


def test_no_absolute_user_paths_in_source() -> None:
    """No committed source file contains a hard-coded absolute user path."""
    repo_root = Path(__file__).parent.parent
    violations: list[str] = []

    for rel in _committed_files():
        if any(rel.startswith(pfx) for pfx in _SKIP_PREFIXES):
            continue
        if rel in _SKIP_EXACT:
            continue
        suffix = Path(rel).suffix.lower()
        if suffix not in _CHECKED_EXTENSIONS:
            continue
        path = repo_root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat in _PATTERNS:
                if pat.search(line):
                    violations.append(f"{rel}:{i}: {line.strip()[:100]}")
                    break  # one violation per line is enough

    assert not violations, (
        f"Absolute user paths found in {len(violations)} location(s):\n"
        + "\n".join(violations)
    )
