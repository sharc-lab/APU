"""Content verification in the model fetcher.

The load-bearing test here is :func:`test_same_size_wrong_content_is_refused`. Size checking
alone catches an overlap-append, because overlapping writes make a file grow. It cannot catch a
resume that splices at a misaligned offset and lands on the correct total length - which is the
residual failure mode on a link that drops most of its TLS handshakes. A size-only fetcher passes
every other test in this file and fails that one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from seam.errors import SeamError
from seam.tools import fetch_model
from seam.tools.fetch_model import RemoteFile, git_blob_sha1, verify_file


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _lfs(payload: bytes, path: str = "openvino_model.bin") -> RemoteFile:
    return RemoteFile(
        path=path,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        git_blob_sha1=None,
    )


class TestVerifyFile:
    def test_intact_lfs_file_verifies_by_sha256(self, tmp_path: Path) -> None:
        payload = b"weights" * 1000
        record = verify_file(_write(tmp_path / "m.bin", payload), _lfs(payload))
        assert record["ok"] is True
        assert record["method"] == "sha256"
        assert record["hash_ok"] is True

    def test_same_size_wrong_content_is_refused(self, tmp_path: Path) -> None:
        """The misaligned-splice case: right length, wrong bytes."""
        good = b"A" * 4096
        spliced = b"A" * 2048 + b"B" * 2048
        assert len(spliced) == len(good)

        record = verify_file(_write(tmp_path / "m.bin", spliced), _lfs(good))

        assert record["size_ok"] is True, "the corruption is invisible to a size check"
        assert record["hash_ok"] is False
        assert record["ok"] is False

    def test_wrong_size_is_refused_without_hashing(self, tmp_path: Path) -> None:
        payload = b"A" * 4096
        record = verify_file(_write(tmp_path / "m.bin", payload + b"extra"), _lfs(payload))
        assert record["ok"] is False
        assert record["size_ok"] is False
        assert record["actual_hash"] is None

    def test_missing_file_is_refused(self, tmp_path: Path) -> None:
        record = verify_file(tmp_path / "absent.bin", _lfs(b"payload"))
        assert record["ok"] is False
        assert record["actual_size"] == -1

    def test_non_lfs_file_verifies_by_git_blob_sha1(self, tmp_path: Path) -> None:
        payload = b'{"model_type": "qwen3"}'
        local = _write(tmp_path / "config.json", payload)
        remote = RemoteFile(
            path="config.json",
            size=len(payload),
            sha256=None,
            git_blob_sha1=git_blob_sha1(local),
        )
        record = verify_file(local, remote)
        assert record["ok"] is True
        assert record["method"] == "git-blob-sha1"
        assert record["hash_ok"] is True

    def test_git_blob_sha1_matches_gits_own_definition(self, tmp_path: Path) -> None:
        """Guards against comparing a plain SHA-1 against git's length-prefixed one."""
        payload = b"hello world"
        expected = hashlib.sha1(b"blob 11\0" + payload).hexdigest()
        assert git_blob_sha1(_write(tmp_path / "f", payload)) == expected
        assert git_blob_sha1(tmp_path / "f") != hashlib.sha1(payload).hexdigest()

    def test_tampered_non_lfs_file_is_refused(self, tmp_path: Path) -> None:
        original = b'{"model_type": "qwen3"}'
        tampered = b'{"model_type": "qwen2"}'
        assert len(tampered) == len(original)
        remote = RemoteFile(
            path="config.json",
            size=len(original),
            sha256=None,
            git_blob_sha1=git_blob_sha1(_write(tmp_path / "orig.json", original)),
        )
        record = verify_file(_write(tmp_path / "config.json", tampered), remote)
        assert record["ok"] is False
        assert record["hash_ok"] is False

    def test_size_only_is_recorded_as_weaker_provenance(self, tmp_path: Path) -> None:
        """A file the Hub gave us no hash for must not look equivalent to a hashed one."""
        payload = b"unhashed"
        remote = RemoteFile(path="f", size=len(payload), sha256=None, git_blob_sha1=None)
        record = verify_file(_write(tmp_path / "f", payload), remote)
        assert record["ok"] is True
        assert record["method"] == "size-only"
        assert record["hash_ok"] is None, "no hash was checked; do not claim one passed"


class TestFetchRefusal:
    """``fetch`` must not emit a provenance claim for bytes it could not verify."""

    @staticmethod
    def _patch_hub(
        monkeypatch: pytest.MonkeyPatch, *, entries: list[dict[str, Any]], content: bytes
    ) -> None:
        def fake_curl_json(url: str) -> Any:
            if "/revision/" in url:
                return {"sha": "deadbeef", "siblings": [{"rfilename": "openvino_model.bin"}]}
            return entries

        def fake_download(
            repo: str, revision: str, name: str, dest: Path, remote: RemoteFile | None = None
        ) -> tuple[int, dict[str, Any]]:
            # Stands in for a transport that returns the wrong bytes without erroring.
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            return 1, {"method": "sha256", "ok": True}

        monkeypatch.setattr(fetch_model, "_curl_json", fake_curl_json)
        monkeypatch.setattr(fetch_model, "_download", fake_download)

    def test_no_spec_is_written_for_same_size_wrong_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good = b"A" * 4096
        spliced = b"A" * 2048 + b"B" * 2048
        self._patch_hub(
            monkeypatch,
            entries=[
                {
                    "type": "file",
                    "path": "openvino_model.bin",
                    "size": len(good),
                    "lfs": {"oid": hashlib.sha256(good).hexdigest(), "size": len(good)},
                }
            ],
            content=spliced,
        )
        spec_path = tmp_path / "spec.yaml"

        with pytest.raises(SeamError, match="refusing to write a FetchedModelSpec"):
            fetch_model.fetch(
                repo="OpenVINO/Qwen3-4B-int4-ov",
                revision="main",
                ladder_position="4b",
                out_dir=tmp_path / "ir",
                spec_path=spec_path,
            )

        assert not spec_path.exists(), "a corrupt IR must not be certified"

    def test_spec_records_per_file_verification_method(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = b"A" * 4096
        self._patch_hub(
            monkeypatch,
            entries=[
                {
                    "type": "file",
                    "path": "openvino_model.bin",
                    "size": len(payload),
                    "lfs": {"oid": hashlib.sha256(payload).hexdigest(), "size": len(payload)},
                }
            ],
            content=payload,
        )
        spec_path = tmp_path / "spec.yaml"

        spec = fetch_model.fetch(
            repo="OpenVINO/Qwen3-4B-int4-ov",
            revision="main",
            ladder_position="4b",
            out_dir=tmp_path / "ir",
            spec_path=spec_path,
        )

        assert spec.verification["openvino_model.bin"]["method"] == "sha256"
        assert spec.verification["openvino_model.bin"]["hash_ok"] is True
        written = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        assert written["verification"]["openvino_model.bin"]["method"] == "sha256"


class TestRemoteFileParsing:
    """``lfs.oid`` is a SHA-256; the top-level ``oid`` is a git blob SHA-1. Never conflate them."""

    def test_lfs_entry_takes_sha256_from_lfs_oid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            fetch_model,
            "_curl_json",
            lambda _url: [
                {
                    "type": "file",
                    "path": "openvino_model.bin",
                    "oid": "1111111111111111111111111111111111111111",
                    "size": 135,
                    "lfs": {"oid": "b" * 64, "size": 2263625445, "pointerSize": 135},
                }
            ],
        )
        files = fetch_model._remote_files("r", "rev")
        entry = files["openvino_model.bin"]
        assert entry.sha256 == "b" * 64
        assert entry.git_blob_sha1 is None, "the pointer's git oid is not the payload's hash"
        assert entry.size == 2263625445, "LFS size is the payload, not the pointer"
        assert entry.verification_method == "sha256"

    def test_non_lfs_entry_takes_git_blob_sha1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            fetch_model,
            "_curl_json",
            lambda _url: [
                {"type": "file", "path": "config.json", "oid": "a" * 40, "size": 700},
            ],
        )
        entry = fetch_model._remote_files("r", "rev")["config.json"]
        assert entry.sha256 is None
        assert entry.git_blob_sha1 == "a" * 40
        assert entry.verification_method == "git-blob-sha1"

    def test_directories_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            fetch_model,
            "_curl_json",
            lambda _url: [
                {"type": "directory", "path": "sub", "oid": "c" * 40},
                {"type": "file", "path": "sub/f.json", "oid": "d" * 40, "size": 10},
            ],
        )
        assert set(fetch_model._remote_files("r", "rev")) == {"sub/f.json"}


class TestHashIrDir:
    def test_only_listed_files_are_hashed(self, tmp_path: Path) -> None:
        listed = tmp_path / "openvino_model.bin"
        listed.write_bytes(b"weights")
        junk = tmp_path / ".cache" / "huggingface" / "junk"
        junk.parent.mkdir(parents=True)
        junk.write_bytes(b"ephemeral")

        aggregate, per_file, total = fetch_model.hash_ir_dir(tmp_path, ["openvino_model.bin"])

        assert set(per_file) == {"openvino_model.bin"}
        assert total == len(b"weights")
        assert len(aggregate) == 64

    def test_missing_declared_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SeamError, match="declared IR file missing"):
            fetch_model.hash_ir_dir(tmp_path, ["openvino_model.bin"])

    def test_include_list_is_sorted_and_deduplicated(self, tmp_path: Path) -> None:
        (tmp_path / "a.bin").write_bytes(b"a")
        (tmp_path / "b.bin").write_bytes(b"b")
        _aggregate, per_file, total = fetch_model.hash_ir_dir(tmp_path, ["b.bin", "a.bin", "b.bin"])
        assert list(per_file.keys()) == ["a.bin", "b.bin"]
        assert total == 2
