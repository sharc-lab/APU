"""Verify a sealed run directory, and write void notices outside the run.

New seals store ``tree_sha256`` only in ``.sealed``. That file is excluded from
the tree hash by name. Older H1 seals appended the same hash to ``summary.json``
after hashing. ``verify_seal`` accepts that pattern as ``MATCH_LEGACY_SELF_REF``
without rewriting the file.

Manifest-format seals store the word ``sealed`` in ``.sealed`` and a
``manifest.sha256.json`` of relative path to sha256. ``MATCH_MANIFEST`` requires
every other file to be listed, every listed file to exist, and every hash to
match. Any miss is ``MANIFEST_COVERAGE_GAP`` and names the files. Neither
verdict writes into the tree.

Void notices go to ``derived/VOIDS/<run_id>.json``. Nothing in this module
writes ``VOID.json`` inside a run directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SealStatus = Literal[
    "MATCH",
    "MATCH_LEGACY_SELF_REF",
    "MATCH_MANIFEST",
    "MANIFEST_COVERAGE_GAP",
    "MISMATCH",
    "UNSEALED",
]

_SEAL_EXCLUDE = frozenset({".sealed"})
_MANIFEST_NAME = "manifest.sha256.json"
_TREE_LINE = re.compile(rb'^(\s*)"tree_sha256"\s*:')


@dataclass(frozen=True)
class SealVerdict:
    """Verifier result. ``files`` is empty except for ``MANIFEST_COVERAGE_GAP``."""

    status: SealStatus
    files: tuple[str, ...] = ()


def tree_sha256(
    root: Path,
    *,
    file_bytes: dict[str, bytes] | None = None,
    skip_rel: frozenset[str] | None = None,
) -> str:
    """Hash a run tree the way the sealers do. Files named ``.sealed`` are excluded.

    ``file_bytes`` substitutes in-memory contents by relative posix path. The
    files on disk are not modified. ``skip_rel`` drops additional relative
    posix paths from the hash, also without modifying the tree.
    """
    root = root.resolve()
    skipped = skip_rel or frozenset()
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in _SEAL_EXCLUDE),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel in skipped:
            continue
        if file_bytes is not None and rel in file_bytes:
            blob = file_bytes[rel]
        else:
            blob = path.read_bytes()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(blob)
        digest.update(b"\0")
    return digest.hexdigest()


def strip_legacy_tree_sha256_line(data: bytes) -> bytes:
    """Drop a ``tree_sha256`` JSON line and the comma it left on the previous key.

    The old sealer rewrote ``summary.json`` with ``json.dumps`` after the hash,
    so the only byte difference is that one key. Re-serializing the JSON does
    not restore the original bytes (float spelling). This is a line edit.
    """
    newline = b"\r\n" if b"\r\n" in data else b"\n"
    raw_lines = data.splitlines(keepends=True)
    kept: list[bytes] = []
    removed = False
    for line in raw_lines:
        body = line.rstrip(b"\r\n")
        if _TREE_LINE.match(body):
            removed = True
            continue
        kept.append(line)
    if not removed:
        return data
    repaired: list[bytes] = []
    for index, line in enumerate(kept):
        body = line.rstrip(b"\r\n")
        ending = line[len(body) :]
        if not ending and index != len(kept) - 1:
            ending = newline
        nxt = b""
        for later in kept[index + 1 :]:
            nxt = later.strip()
            if nxt:
                break
        if body.endswith(b",") and nxt.startswith(b"}"):
            body = body[:-1]
        repaired.append(body + ending)
    return b"".join(repaired)


def _legacy_overlay(root: Path) -> dict[str, bytes]:
    overlay: dict[str, bytes] = {}
    for path in root.rglob("summary.json"):
        if not path.is_file():
            continue
        original = path.read_bytes()
        stripped = strip_legacy_tree_sha256_line(original)
        if stripped != original:
            overlay[path.relative_to(root.resolve()).as_posix()] = stripped
    return overlay


def _recorded_tree_sha256(root: Path) -> str | None:
    marker = root / ".sealed"
    if not marker.is_file():
        return None
    try:
        doc = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    value = doc.get("tree_sha256")
    if not isinstance(value, str) or not value:
        return None
    return value


def seal_verdict(directory: Path) -> SealVerdict:
    """Classify ``directory``. Does not write into it."""
    root = directory.resolve()
    recorded = _recorded_tree_sha256(root)
    if recorded is not None:
        if tree_sha256(root) == recorded:
            return SealVerdict("MATCH")
        overlay = _legacy_overlay(root)
        if overlay and tree_sha256(root, file_bytes=overlay) == recorded:
            return SealVerdict("MATCH_LEGACY_SELF_REF")
        return SealVerdict("MISMATCH")
    if _is_word_sealed(root) and (root / _MANIFEST_NAME).is_file():
        return _manifest_verdict(root)
    return SealVerdict("UNSEALED")


def verify_seal(directory: Path) -> SealStatus:
    """Return the seal status. Does not write into ``directory``.

    Tree-hash seals: MATCH, MATCH_LEGACY_SELF_REF, MISMATCH, or UNSEALED.
    Manifest-format seals: MATCH_MANIFEST or MANIFEST_COVERAGE_GAP.
    """
    return seal_verdict(directory).status


def _is_word_sealed(root: Path) -> bool:
    marker = root / ".sealed"
    if not marker.is_file():
        return False
    try:
        text = marker.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    return text.strip() == "sealed"


def _manifest_verdict(root: Path) -> SealVerdict:
    """MATCH_MANIFEST, or MANIFEST_COVERAGE_GAP naming every problem path."""
    manifest_path = root / _MANIFEST_NAME
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return SealVerdict("MANIFEST_COVERAGE_GAP", (_MANIFEST_NAME,))
    if not isinstance(doc, dict):
        return SealVerdict("MANIFEST_COVERAGE_GAP", (_MANIFEST_NAME,))

    listed: dict[str, object] = {str(key).replace("\\", "/"): value for key, value in doc.items()}
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in _SEAL_EXCLUDE
        and path.resolve() != manifest_path.resolve()
    }
    unlisted = tuple(sorted(present - set(listed)))
    missing: list[str] = []
    hash_mismatch: list[str] = []
    for rel, expect in sorted(listed.items()):
        path = _path_inside(root, rel)
        if path is None or not path.is_file():
            missing.append(rel)
            continue
        if not isinstance(expect, str) or hashlib.sha256(path.read_bytes()).hexdigest() != expect:
            hash_mismatch.append(rel)
    files = unlisted + tuple(missing) + tuple(hash_mismatch)
    if files:
        return SealVerdict("MANIFEST_COVERAGE_GAP", files)
    return SealVerdict("MATCH_MANIFEST")


def _path_inside(root: Path, rel: str) -> Path | None:
    """Resolve ``rel`` only when it stays inside ``root``."""
    if not rel or rel.startswith(("/", "\\")):
        return None
    parts = Path(rel).parts
    if ".." in parts:
        return None
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path


def void_notice_path(run_id: str, *, root: Path) -> Path:
    """Path of the out-of-tree void notice for ``run_id``."""
    _require_run_id(run_id)
    return (root / "derived" / "VOIDS" / f"{run_id}.json").resolve()


def write_void_notice(run_id: str, notice: Mapping[str, Any], *, root: Path) -> Path:
    """Write ``derived/VOIDS/<run_id>.json``. Never writes inside a run directory."""
    _require_run_id(run_id)
    dest = void_notice_path(run_id, root=root)
    voids = (root.resolve() / "derived" / "VOIDS").resolve()
    if dest.parent != voids:
        raise ValueError(f"void notice path escaped derived/VOIDS: {dest}")
    if dest.name == "VOID.json":
        raise ValueError("refusing in-tree VOID.json name")
    body = dict(notice)
    cited = body.get("run_id")
    if cited is not None and cited != run_id:
        raise ValueError(f"notice run_id {cited!r} does not match {run_id!r}")
    body["run_id"] = run_id
    body["void_notice_path"] = f"derived/VOIDS/{run_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def _require_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError(f"invalid run_id for void notice: {run_id!r}")
