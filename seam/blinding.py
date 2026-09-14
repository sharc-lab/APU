"""Condition blinding (spec §8, blueprint §5).

**ANALYSIS CODE MUST NOT IMPORT THIS MODULE.**

The rule from spec §8: *"analysis consumes ``blinded_label``; a separate explicit ``unblind`` step
joins labels. Analysis code must not import the condition mapping."* This module **is** the
condition mapping. Anything under ``seam/analysis/`` that imports it has broken the blind, so
``tests/test_blinding_boundary.py`` fails the build if that happens.

Why bother, given the same repository contains both sides: blinding is not protection against a
malicious analyst, it is protection against an honest one. Analysis choices made while knowing
which arm is which drift toward the expected answer without anyone intending it, and that drift is
the "analysis drift" threat in blueprint §5.1. Making the mapping an *explicit import* turns
breaking the blind into a visible, reviewable act rather than an accident.

The salt lives under ``raw/_blinding/`` and is gitignored. Without a salt, ``blinded_label`` would
be a plain hash of a tiny label space (``"A"``, ``"B"``, ``"reference"``…) and trivially invertible
by anyone who guessed the labels, which would make the blind decorative.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Final

from seam.errors import SeamError
from seam.hashing import sha256_bytes
from seam.jsonlog import log_event, utc_now_iso

__all__ = [
    "BLINDING_DIR",
    "blinded_label_for",
    "get_or_create_salt",
    "record_unblind_entry",
]

BLINDING_DIR: Final = Path("raw") / "_blinding"
_SALT_FILE: Final = "salt.txt"
_UNBLIND_FILE: Final = "unblind_map.json"

#: Hex characters retained from the digest. 12 hex = 48 bits: collision-free at this scale while
#: staying short enough to read in a filename or a log line.
_LABEL_HEX_CHARS: Final = 12

#: 32 bytes of CSPRNG output.
_SALT_BYTES: Final = 32


def get_or_create_salt(repo_root: Path) -> str:
    """Return the project blinding salt, creating it on first use.

    The salt is written once and never rotated: rotating it would change every previously emitted
    ``blinded_label``, severing already-collected runs from their conditions.
    """
    salt_path = repo_root / BLINDING_DIR / _SALT_FILE
    if salt_path.is_file():
        salt = salt_path.read_text(encoding="utf-8").strip()
        if not salt:
            raise SeamError(
                f"blinding salt at {salt_path} is empty. Refusing to regenerate it, because a new "
                f"salt would orphan every blinded_label already emitted. Restore it from backup."
            )
        return salt

    salt_path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(_SALT_BYTES)
    salt_path.write_text(salt + "\n", encoding="utf-8")
    log_event(
        "blinding.salt_created",
        message=f"created project blinding salt at {salt_path} (gitignored; back it up)",
        path=str(salt_path),
    )
    return salt


def blinded_label_for(condition_label: str, *, salt: str) -> str:
    """Map a condition label to its blinded form.

    Deterministic given the salt, so the same condition always yields the same blinded label across
    runs - which is what lets analysis group runs by arm without knowing which arm is which.
    """
    if not condition_label:
        raise ValueError("condition_label must be non-empty")
    digest = sha256_bytes(f"{salt}\x00{condition_label}".encode())
    return f"cond_{digest[:_LABEL_HEX_CHARS]}"


def record_unblind_entry(condition_label: str, blinded_label: str, *, repo_root: Path) -> None:
    """Append a condition-to-blinded mapping to the unblind map.

    The map is the input to the explicit unblinding step. It is append-only and lives outside any
    individual run directory, so sealing a run does not prevent later runs from registering their
    conditions.
    """
    path = repo_root / BLINDING_DIR / _UNBLIND_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, str] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8")).get("map", {})

    previous = existing.get(blinded_label)
    if previous is not None and previous != condition_label:
        raise SeamError(
            f"blinded label {blinded_label} already maps to condition {previous!r}, cannot also "
            f"map to {condition_label!r}. This is a hash collision or a salt change; either way "
            f"the blind is compromised and must be investigated, not worked around."
        )

    existing[blinded_label] = condition_label
    document = {"updated_utc": utc_now_iso(), "map": existing}
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
