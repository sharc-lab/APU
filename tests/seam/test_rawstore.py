"""Write-once ``raw/`` guard tests (spec §7/M1: "raw/ write-once enforced by a guard that fails on
modification attempts").

The central test is :func:`test_every_write_path_fails_after_seal`: it enumerates every mutating
entry point and asserts each one raises. A guard that covers only the path someone remembered to
check is not a guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seam.errors import RawStoreError, RunSealedError
from seam.rawstore import (
    RunDir,
    create_run_dir,
    is_sealed,
    open_run_dir,
    verify_sealed,
)

RUN_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def open_run(tmp_path: Path) -> RunDir:
    """A run directory with one file written, not yet sealed."""
    run_dir = create_run_dir(RUN_ID, repo_root=tmp_path)
    run_dir.write_json("summary.json", {"stage": "open"})
    return run_dir


@pytest.fixture
def sealed_run(open_run: RunDir) -> RunDir:
    open_run.seal()
    return open_run


# ==================================================================================================
# Lifecycle
# ==================================================================================================


def test_create_run_dir_creates_the_directory(tmp_path: Path) -> None:
    run_dir = create_run_dir(RUN_ID, repo_root=tmp_path)
    assert run_dir.path == tmp_path / "raw" / RUN_ID
    assert run_dir.path.is_dir()
    assert not run_dir.is_sealed()


def test_create_run_dir_refuses_to_reuse_an_existing_id(tmp_path: Path) -> None:
    """Reusing a run directory would conflate two runs under one identifier."""
    create_run_dir(RUN_ID, repo_root=tmp_path)
    with pytest.raises(RawStoreError, match="already exists"):
        create_run_dir(RUN_ID, repo_root=tmp_path)


def test_writes_succeed_while_open(open_run: RunDir) -> None:
    open_run.write_text("notes.txt", "hello\n")
    open_run.append_ndjson("samples.ndjson", {"t_ns": 0})
    open_run.append_ndjson("samples.ndjson", {"t_ns": 1})

    assert (open_run.path / "notes.txt").read_text(encoding="utf-8") == "hello\n"
    lines = (open_run.path / "samples.ndjson").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_write_text_refuses_to_overwrite_an_existing_file(open_run: RunDir) -> None:
    """Raw files are written once, not overwritten - even before the run seals."""
    open_run.write_text("notes.txt", "first\n")
    with pytest.raises(RawStoreError, match="already exists"):
        open_run.write_text("notes.txt", "second\n")


def test_seal_records_hash_and_file_list(open_run: RunDir) -> None:
    tree_hash = open_run.seal()

    assert open_run.is_sealed()
    assert len(tree_hash) == 64
    marker = open_run.read_seal()
    assert marker["run_id"] == RUN_ID
    assert marker["raw_sha256"] == tree_hash
    assert marker["self_check"] == "pass"
    assert "summary.json" in marker["files"]


def test_seal_refuses_an_empty_run_directory(tmp_path: Path) -> None:
    run_dir = create_run_dir(RUN_ID, repo_root=tmp_path)
    with pytest.raises(RawStoreError, match="empty"):
        run_dir.seal()


# ==================================================================================================
# The write-once guard
# ==================================================================================================


def test_every_write_path_fails_after_seal(sealed_run: RunDir) -> None:
    """M1 acceptance: every mutating entry point must refuse once the run is sealed."""
    with pytest.raises(RunSealedError, match="write-once"):
        sealed_run.write_text("new.txt", "nope")

    with pytest.raises(RunSealedError, match="write-once"):
        sealed_run.write_json("new.json", {"nope": True})

    with pytest.raises(RunSealedError, match="write-once"):
        sealed_run.append_ndjson("samples.ndjson", {"t_ns": 99})

    with pytest.raises(RunSealedError, match="write-once"):
        sealed_run.open_write("new.bin", binary=True)


def test_append_to_an_existing_file_fails_after_seal(sealed_run: RunDir) -> None:
    """The realistic failure mode: a later script appending "just one fix" to finished data."""
    with pytest.raises(RunSealedError):
        sealed_run.append_ndjson("summary.json", {"appended": True})


def test_guard_refuses_before_writing_any_bytes(sealed_run: RunDir) -> None:
    """The refusal must precede the write, so a sealed run cannot be partially corrupted."""
    before = json.loads((sealed_run.path / "summary.json").read_text(encoding="utf-8"))

    with pytest.raises(RunSealedError):
        sealed_run.open_write("summary.json")

    after = json.loads((sealed_run.path / "summary.json").read_text(encoding="utf-8"))
    assert after == before, "sealed file was modified by a refused open_write"


def test_reseal_is_refused(sealed_run: RunDir) -> None:
    """Re-sealing would overwrite the recorded hash - precisely the tampering this prevents."""
    with pytest.raises(RunSealedError, match="already sealed"):
        sealed_run.seal()


def test_sealed_files_are_marked_read_only(sealed_run: RunDir) -> None:
    """Best-effort OS enforcement, for code that never asks this module for permission."""
    with pytest.raises(PermissionError):
        (sealed_run.path / "summary.json").open("a", encoding="utf-8")


def test_guard_survives_reopening_the_directory(tmp_path: Path, sealed_run: RunDir) -> None:
    """A fresh handle to a sealed run must be just as locked as the original."""
    reopened = open_run_dir(RUN_ID, repo_root=tmp_path)
    assert reopened.is_sealed()
    with pytest.raises(RunSealedError):
        reopened.write_text("sneaky.txt", "nope")


def test_is_sealed_helper_reflects_state(tmp_path: Path, open_run: RunDir) -> None:
    assert is_sealed(RUN_ID, repo_root=tmp_path) is False
    open_run.seal()
    assert is_sealed(RUN_ID, repo_root=tmp_path) is True


# ==================================================================================================
# Path traversal
# ==================================================================================================


@pytest.mark.parametrize("bad_name", ["../escape.txt", "sub/../../escape.txt"])
def test_writes_cannot_escape_the_run_directory(open_run: RunDir, bad_name: str) -> None:
    """A run must not write outside its own directory, or the tree hash would not cover its output."""
    with pytest.raises(RawStoreError, match="outside"):
        open_run.write_text(bad_name, "nope")


def test_absolute_paths_are_rejected(open_run: RunDir, tmp_path: Path) -> None:
    with pytest.raises(RawStoreError, match="relative"):
        open_run.write_text(str(tmp_path / "absolute.txt"), "nope")


def test_nested_relative_paths_are_allowed(open_run: RunDir) -> None:
    open_run.write_text("sub/dir/file.txt", "ok\n")
    assert (open_run.path / "sub" / "dir" / "file.txt").is_file()


# ==================================================================================================
# Integrity verification
# ==================================================================================================


def test_verify_sealed_passes_on_untouched_run(sealed_run: RunDir) -> None:
    assert verify_sealed(sealed_run) is True


def test_verify_sealed_detects_tampering(sealed_run: RunDir) -> None:
    """Blueprint §5.6 item 3: verifying raw checksums must actually catch a modification.

    The read-only attribute is cleared first, simulating an actor that bypassed this module
    entirely - which is the only way a sealed file gets modified in practice.
    """
    import stat

    victim = sealed_run.path / "summary.json"
    victim.chmod(stat.S_IWRITE | stat.S_IREAD)
    victim.write_text(json.dumps({"stage": "tampered"}) + "\n", encoding="utf-8")

    assert verify_sealed(sealed_run) is False


def test_verify_sealed_is_read_only(sealed_run: RunDir) -> None:
    """Verification must not modify the run, so a failed check cannot itself damage the data."""
    before = {
        path.relative_to(sealed_run.path).as_posix(): path.read_bytes()
        for path in sealed_run.path.rglob("*")
        if path.is_file()
    }

    verify_sealed(sealed_run)

    after = {
        path.relative_to(sealed_run.path).as_posix(): path.read_bytes()
        for path in sealed_run.path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert sealed_run.seal_marker.is_file()


def test_verify_sealed_requires_a_sealed_run(open_run: RunDir) -> None:
    with pytest.raises(RawStoreError, match="not sealed"):
        verify_sealed(open_run)
