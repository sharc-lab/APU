"""Content hashing for provenance and integrity.

Every hash in a manifest is computed here, at emit time, from bytes on disk. No hash is ever
transcribed from a document by hand - a hand-copied hash is indistinguishable from a fabricated
one and goes stale silently (blueprint AF-002).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from pathlib import Path
from typing import Any, Final

__all__ = [
    "canonical_json",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_tree",
]

#: Read files in fixed-size chunks so a large artifact never has to be held in memory.
_CHUNK_BYTES: Final = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 of a file's bytes.

    Hashes raw bytes, never decoded text, so line-ending normalisation or encoding assumptions
    cannot change the result.

    Raises:
        FileNotFoundError: If the path does not exist. Not defaulted to a sentinel - an absent
            provenance artifact must fail the run, not produce a placeholder hash.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Serialise ``obj`` to a canonical byte form suitable for hashing.

    Canonical means: keys sorted, no insignificant whitespace, UTF-8, non-ASCII preserved. Two
    configs that differ only in key order or formatting must hash identically, otherwise
    ``config_hash`` would report spurious condition changes on a reformat.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(obj: Any) -> str:
    """Return the SHA-256 of ``obj`` in canonical JSON form."""
    return sha256_bytes(canonical_json(obj))


def sha256_tree(root: Path, *, exclude: Collection[str] = ()) -> str:
    """Return a single SHA-256 covering every file under ``root``.

    One value that changes if any byte of any included file changes.

    The digest covers each file's ``root``-relative POSIX path as well as its content, so renaming a
    file changes the tree hash. Paths are sorted for determinism, and POSIX separators are used so a
    hash computed on Windows matches one computed on Linux.

    Args:
        root: Directory to hash.
        exclude: ``root``-relative POSIX paths to skip. Needed for two cases where a file cannot be
            covered by a hash it contains or accompanies: a seal marker that records the tree hash,
            and a manifest that embeds it.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")

    excluded = frozenset(exclude)
    digest = hashlib.sha256()
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = file_path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()
