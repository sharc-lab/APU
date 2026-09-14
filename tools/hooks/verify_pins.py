"""Pre-commit hook: verify governing-document SHA-256 pins (AM-009).

Governing documents are hash-pinned so that drift is detectable rather than
silent. This hook fails the commit when a pinned document's hash no longer
matches ``GOVERNING_DOCS.sha256``.

Distinguishing the two cases matters, and the hook cannot do it for you:

  known supersession  — the document changed via a logged amendment. Update the
                        pin, archive the superseded hash, commit both together.
  unexplained drift   — the document changed with no amendment. Stop. Find out
                        why before committing anything.

A CRLF save is the most common innocent cause; ``.gitattributes`` sets -text on
SEAM paths (AF-007) and ``.vscode/settings.json`` pins ``files.eol`` to "\\n",
but an external editor can still bypass both.

Exit 0 = pins match or no pin file exists, 1 = mismatch.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIN_FILE = REPO_ROOT / "GOVERNING_DOCS.sha256"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not PIN_FILE.is_file():
        # Absent pin file is not a failure: the pin record may not be
        # established yet. Say so rather than silently passing.
        sys.stderr.write(
            f"note: {PIN_FILE.name} absent — governing-document pins are not "
            "being verified. Create it to enable this check (AM-009).\n"
        )
        return 0

    mismatches: list[tuple[str, str, str]] = []
    missing: list[str] = []
    checked = 0

    for line in PIN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected, rel_path = parts[0], parts[1].strip()

        target = REPO_ROOT / rel_path
        if not target.is_file():
            missing.append(rel_path)
            continue

        actual = sha256_of(target)
        checked += 1
        if actual != expected:
            mismatches.append((rel_path, expected, actual))

    if missing:
        sys.stderr.write("\nPINNED DOCUMENT MISSING:\n")
        for rel_path in missing:
            sys.stderr.write(f"  {rel_path}\n")

    if mismatches:
        sys.stderr.write("\nBLOCKED — governing-document pin mismatch (AM-009):\n")
        for rel_path, expected, actual in mismatches:
            sys.stderr.write(f"  {rel_path}\n")
            sys.stderr.write(f"    expected {expected}\n")
            sys.stderr.write(f"    actual   {actual}\n")
        sys.stderr.write(
            "\nDecide which case this is before proceeding:\n"
            "  KNOWN SUPERSESSION — the doc changed via a logged amendment. Update the\n"
            "    pin, archive the superseded hash, and commit both in the same change.\n"
            "  UNEXPLAINED DRIFT — no amendment covers this. Stop and investigate.\n"
            "  ENCODING — a CRLF save or a changed trailing newline. Restore LF and a\n"
            "    single trailing newline; do not edit the pin to match.\n\n"
            "Never edit the pin list to silence a mismatch you do not understand.\n\n"
        )
        return 1

    if missing:
        return 1

    sys.stderr.write(f"governing-document pins verified ({checked} file(s))\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
