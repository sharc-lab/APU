"""Write-once ``raw/`` store (spec §9.1, blueprint §5.3).

``raw/`` is immutable. A run directory is *open* while its run is in progress and *sealed* once it
completes; after sealing, every mutating path in this module refuses, and the on-disk files are
additionally marked read-only.

The refusal happens on **attempt**, before any bytes are written, so a caller cannot partially
corrupt a sealed run and then discover the error.

Two layers, deliberately:

* **Advisory** - a ``.sealed`` marker holding the tree hash. This is what code checks.
* **Best-effort OS enforcement** - the read-only file attribute, which stops a careless editor or
  script that never asks this module for permission.

Neither layer defends against a determined attacker, and neither is meant to. They defend against
the realistic failure: a well-intentioned later script appending "just one fix" to finished data.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

from seam.errors import RawStoreError, RunSealedError
from seam.hashing import sha256_tree
from seam.jsonlog import log_event, utc_now_iso

__all__ = ["RAW_DIR_NAME", "RunDir", "create_run_dir", "is_sealed", "open_run_dir", "verify_sealed"]

RAW_DIR_NAME: Final = "raw"
_SEAL_MARKER: Final = ".sealed"


@dataclass(frozen=True, slots=True)
class RunDir:
    """A handle to one ``raw/<run_id>/`` directory."""

    run_id: str
    path: Path

    # ---------------------------------------------------------------------------------- state ---

    @property
    def seal_marker(self) -> Path:
        return self.path / _SEAL_MARKER

    def is_sealed(self) -> bool:
        return self.seal_marker.is_file()

    def _assert_open(self, operation: str, target: str | None = None) -> None:
        """Refuse ``operation`` if the run is already sealed.

        Raises:
            RunSealedError: Always, if sealed. This is the write-once guard.
        """
        if not self.is_sealed():
            return

        detail = f" (target: {target})" if target else ""
        log_event(
            "rawstore.write_after_seal_refused",
            severity="error",
            message=f"refused {operation} on sealed run {self.run_id}{detail}",
            run_id=self.run_id,
            operation=operation,
            target=target,
            path=str(self.path),
        )
        raise RunSealedError(
            f"run {self.run_id} is sealed; refusing {operation}{detail}. "
            f"raw/ is write-once (spec §9.1, blueprint §5.3): completed raw data is never "
            f"modified. To correct an error, emit a NEW run and record the supersession in "
            f"AUDIT_LOG.md."
        )

    # ---------------------------------------------------------------------------------- writes ---

    def write_text(self, name: str, content: str) -> Path:
        """Write a text file into the run directory.

        Raises:
            RunSealedError: If the run is sealed.
            RawStoreError: If ``name`` escapes the run directory, or the file already exists.
        """
        target = self._resolve_child(name)
        self._assert_open("write_text", name)
        if target.exists():
            raise RawStoreError(
                f"{name} already exists in run {self.run_id}; raw files are written once, "
                f"not overwritten"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def write_json(self, name: str, obj: Any) -> Path:
        """Write a pretty-printed JSON file into the run directory."""
        return self.write_text(name, json.dumps(obj, indent=2, sort_keys=True) + "\n")

    def append_ndjson(self, name: str, record: dict[str, Any]) -> Path:
        """Append one newline-delimited JSON record.

        Appending is legitimate *while the run is open* - that is how ``samples.ndjson`` and
        ``steps.ndjson` are produced. It becomes illegal the moment the run seals.

        Raises:
            RunSealedError: If the run is sealed.
        """
        target = self._resolve_child(name)
        self._assert_open("append_ndjson", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return target

    def open_write(self, name: str, *, binary: bool = False) -> IO[Any]:
        """Open a file inside the run directory for writing.

        Raises:
            RunSealedError: If the run is sealed. Checked before the handle is created, so a
                sealed run cannot be truncated by the act of opening it.
        """
        target = self._resolve_child(name)
        self._assert_open("open_write", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target.open("wb" if binary else "w", encoding=None if binary else "utf-8")

    def _resolve_child(self, name: str) -> Path:
        """Resolve ``name`` inside the run directory, rejecting traversal.

        Raises:
            RawStoreError: If ``name`` is absolute or escapes the run directory. A run must not be
                able to write outside its own directory, or the tree hash would not cover its
                output.
        """
        candidate = Path(name)
        if candidate.is_absolute():
            raise RawStoreError(f"raw file names must be relative, got {name!r}")

        resolved = (self.path / candidate).resolve()
        root = self.path.resolve()
        if resolved != root and root not in resolved.parents:
            raise RawStoreError(
                f"{name!r} resolves outside run directory {self.path}; refusing to write"
            )
        return resolved

    # ----------------------------------------------------------------------------------- seal ---

    def seal(self, *, self_check: str = "pass") -> str:
        """Seal the run directory and return its tree hash.

        Computes the hash over every file *except* the marker itself (which cannot contain its own
        hash), writes the marker, then marks all files read-only.

        Raises:
            RunSealedError: If already sealed. Re-sealing would overwrite the recorded hash, which
                is exactly the tampering this guard exists to prevent.
            RawStoreError: If the directory contains no files.
        """
        if self.is_sealed():
            raise RunSealedError(
                f"run {self.run_id} is already sealed; refusing to re-seal and overwrite the "
                f"recorded raw_sha256"
            )

        files = [p for p in self.path.rglob("*") if p.is_file()]
        if not files:
            raise RawStoreError(f"refusing to seal empty run directory {self.path}")

        # The marker is excluded so that seal-time and verify-time hashes are computed over exactly
        # the same file set; a marker cannot contain its own hash.
        tree_hash = sha256_tree(self.path, exclude={_SEAL_MARKER})

        marker = {
            "run_id": self.run_id,
            "sealed_at_utc": utc_now_iso(),
            "raw_sha256": tree_hash,
            "self_check": self_check,
            "n_files": len(files),
            "files": sorted(p.relative_to(self.path).as_posix() for p in files),
        }
        self.seal_marker.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        self._set_read_only([*files, self.seal_marker])

        log_event(
            "rawstore.sealed",
            message=f"sealed run {self.run_id} ({len(files)} file(s))",
            run_id=self.run_id,
            path=str(self.path),
            raw_sha256=tree_hash,
            n_files=len(files),
            self_check=self_check,
        )
        return tree_hash

    def _set_read_only(self, paths: list[Path]) -> None:
        """Best-effort OS-level read-only marking.

        Failure is logged, never swallowed (spec §9.6), and does not abort the seal: the advisory
        marker is the authoritative layer, and losing the belt does not invalidate the braces.
        """
        for path in paths:
            try:
                path.chmod(stat.S_IREAD)
            except OSError as exc:
                log_event(
                    "rawstore.read_only_failed",
                    severity="warning",
                    message="could not set read-only attribute; advisory seal marker still applies",
                    run_id=self.run_id,
                    path=str(path),
                    error=str(exc),
                )

    def read_seal(self) -> dict[str, Any]:
        """Return the seal marker contents.

        Raises:
            RawStoreError: If the run is not sealed.
        """
        if not self.is_sealed():
            raise RawStoreError(f"run {self.run_id} is not sealed")
        parsed: dict[str, Any] = json.loads(self.seal_marker.read_text(encoding="utf-8"))
        return parsed


def create_run_dir(run_id: str, *, repo_root: Path) -> RunDir:
    """Create ``raw/<run_id>/``.

    Raises:
        RawStoreError: If the directory already exists. Reusing a run directory would conflate two
            runs under one identifier, breaking the "every number traces to a run_id" invariant.
    """
    path = repo_root / RAW_DIR_NAME / run_id
    if path.exists():
        raise RawStoreError(
            f"run directory already exists: {path}. Run IDs are uuid4 and never reused; "
            f"a collision means the caller passed an existing id."
        )
    path.mkdir(parents=True)
    log_event(
        "rawstore.run_created",
        message=f"created run directory for {run_id}",
        run_id=run_id,
        path=str(path),
    )
    return RunDir(run_id=run_id, path=path)


def open_run_dir(run_id: str, *, repo_root: Path) -> RunDir:
    """Open an existing run directory.

    Raises:
        RawStoreError: If it does not exist.
    """
    path = repo_root / RAW_DIR_NAME / run_id
    if not path.is_dir():
        raise RawStoreError(f"no such run directory: {path}")
    return RunDir(run_id=run_id, path=path)


def is_sealed(run_id: str, *, repo_root: Path) -> bool:
    """Return whether a run directory is sealed."""
    return (repo_root / RAW_DIR_NAME / run_id / _SEAL_MARKER).is_file()


def verify_sealed(run_dir: RunDir) -> bool:
    """Recompute a sealed run's tree hash and compare it against the recorded one.

    This is the integrity check behind blueprint §5.6 item 3 ("verify raw checksums").

    Returns:
        True if the recomputed hash matches.

    Raises:
        RawStoreError: If the run is not sealed.
    """
    marker = run_dir.read_seal()
    recorded: str = marker["raw_sha256"]

    # Recomputed over the same exclusion set used at seal time. Nothing is moved or modified:
    # verification is strictly read-only, so a failed verification cannot itself damage the run.
    actual = sha256_tree(run_dir.path, exclude={_SEAL_MARKER})

    matched = actual == recorded
    log_event(
        "rawstore.integrity_check",
        severity="info" if matched else "critical",
        message=(f"raw integrity {'OK' if matched else 'FAILED'} for run {run_dir.run_id}"),
        run_id=run_dir.run_id,
        recorded_sha256=recorded,
        recomputed_sha256=actual,
        matched=matched,
    )
    return matched


def make_writable_for_test(path: Path) -> None:
    """Clear the read-only attribute on every file under ``path``.

    Sealed runs are read-only, which makes a pytest ``tmp_path`` impossible to clean up on Windows.
    Exported for test teardown only. It is deliberately not used anywhere in the harness - that
    would be a hole straight through the write-once guard.
    """
    for child in path.rglob("*"):
        if child.is_file():
            with_write = child.stat().st_mode | stat.S_IWRITE
            child.chmod(with_write)
