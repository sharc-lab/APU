"""Fetch a pre-converted OpenVINO IR from the Hub, with provenance (AM-023).

Spec §5.3 requires conversion to be scripted and reproducible. Obtaining a *pre-converted* IR
is a documented weakening of that rule: the quantization parameters were chosen by the
publisher, not by us. This tool exists to make the weakening auditable rather than invisible -
it records the source repo, the pinned revision SHA, the compression metadata that the IR
itself carries in its ``rt_info`` block, and our own computed SHA-256 over every file.

It also works around a measured network fault on this host (see AUDIT_LOG, 2026-08-02): TLS
handshakes to ``huggingface.co`` fail roughly two times in three, at the transport layer rather
than with an HTTP status. ``huggingface_hub`` retries HTTP statuses but treats a protocol-level
disconnect as fatal after a few attempts, and a snapshot download opens enough connections that
it effectively never completes. ``curl --retry-all-errors`` does retry that class of failure, so
every transfer here goes through curl: one connection at a time, resumable via range requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from seam.errors import SeamError
from seam.gitinfo import repo_root
from seam.hashing import sha256_file
from seam.locks import exclusive
from seam.model_provenance import publisher_quantization_from_readme

__all__ = ["FetchedModelSpec", "fetch", "hash_ir_dir"]

_HF_API = "https://huggingface.co/api/models"
_HF_RESOLVE = "https://huggingface.co"

# The transport fault is bursty, so a generous retry count matters more than a long delay: each
# individual attempt fails within a few hundred milliseconds when it fails at all.
_CURL_RETRY = ["--retry", "40", "--retry-all-errors", "--retry-delay", "1", "--max-time", "3600"]


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """What the Hub says a file should be.

    ``sha256`` is populated only for LFS entries, where ``lfs.oid`` *is* the SHA-256 of the
    content. For small non-LFS files the Hub reports a **git blob SHA-1**
    (``sha1(b"blob <size>\\0" + content)``), which is a different function over a different
    preimage and must never be compared against a SHA-256.
    """

    path: str
    size: int
    sha256: str | None
    git_blob_sha1: str | None

    @property
    def verification_method(self) -> str:
        if self.sha256:
            return "sha256"
        if self.git_blob_sha1:
            return "git-blob-sha1"
        return "size-only"


@dataclass(slots=True)
class FetchedModelSpec:
    """Provenance for an IR we did not convert ourselves.

    ``ir_sha256`` is a **self-constructed** aggregate digest over the per-file SHA-256 values
    recorded in ``ir_files``. It is *not* a publisher-verifiable value - the Hub never publishes
    that aggregate. The externally checkable hashes are the per-file entries in ``verification``
    (``lfs.oid`` SHA-256 for LFS files; git blob SHA-1 for non-LFS files).
    """

    name: str
    revision: str
    ladder_position: str
    source: str
    source_repo: str
    download_method: str
    self_converted: bool
    publisher_quantization: dict[str, Any]
    ir_dir: str
    ir_sha256: str
    ir_files: dict[str, str]
    ir_bytes: int
    fetched_utc: str
    fetch_duration_s: float
    transfer_attempts: dict[str, int]
    #: Per file: how its content was verified against the Hub. A file checked by length alone is
    #: weaker provenance than one checked by hash, and must be visibly so rather than silently
    #: equivalent.
    verification: dict[str, dict[str, Any]] = field(default_factory=dict)


def _curl_json(url: str) -> Any:
    completed = subprocess.run(
        ["curl.exe", "-sL", *_CURL_RETRY, url], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise SeamError(
            f"curl failed (exit {completed.returncode}) for {url}. This host has a measured "
            f"TLS handshake fault; if retries are exhausted the network is worse than when it "
            f"was characterized. Re-run the connectivity probe before assuming a Hub outage."
        )
    return json.loads(completed.stdout)


def _download(
    repo: str, revision: str, name: str, dest: Path, remote: RemoteFile | None = None
) -> tuple[int, dict[str, Any]]:
    """Download one file, resuming if a partial exists, then verify size AND content hash.

    Returns ``(attempts, verification_record)``.

    Verification is not optional and not advisory. A file that fails it is **deleted and
    re-fetched from zero**, because a corrupt IR loads happily and poisons every measurement
    taken with it - silently, and in a way that looks like a result.
    """
    url = f"{_HF_RESOLVE}/{repo}/resolve/{revision}/{name}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # An oversized leftover can never be repaired by resuming: curl would ask for bytes beyond the
    # end of the source. Discard it and start clean.
    if remote is not None and dest.exists() and dest.stat().st_size > remote.size:
        dest.unlink()

    record: dict[str, Any] = {"method": "unverified", "ok": False}
    for attempt in range(1, 6):
        completed = subprocess.run(
            ["curl.exe", "-sL", "-C", "-", *_CURL_RETRY, "-o", str(dest), url],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            if remote is None:
                return attempt, {"method": "none", "ok": True}
            record = verify_file(dest, remote)
            if record["ok"]:
                return attempt, record
            # Correct length but wrong content means a misaligned splice, which no resume can
            # repair - the bad bytes are already inside the file. Start over from zero.
            dest.unlink(missing_ok=True)
            continue
        # curl exit 33 means the server refused a range request, which leaves the partial file
        # unusable for resume. Start it over rather than retrying the same doomed range.
        if completed.returncode == 33 and dest.exists():
            dest.unlink()

    raise SeamError(
        f"could not obtain a verified copy of {name} from {repo}@{revision} after 5 attempts. "
        f"Last verification: {record}"
    )


def hash_ir_dir(path: Path, include: Iterable[str]) -> tuple[str, dict[str, str], int]:
    """SHA-256 per declared IR file from a positive include-list.

    Only paths declared by the Hub tree / siblings API are hashed. Ephemeral cache junk under
    ``.cache/`` never enters the aggregate digest. A declared path that is missing on disk is a
    hard refusal - emitting a spec for an incomplete IR would certify corruption.

    Hashing streams in chunks via :func:`seam.hashing.sha256_file`. Reading a multi-gigabyte
    weight file with ``read_bytes()`` fails on Windows with ``OSError: [Errno 22]``.
    """
    import hashlib

    per_file: dict[str, str] = {}
    total = 0
    for name in sorted(set(include)):
        file = path / name
        if not file.is_file():
            raise SeamError(
                f"declared IR file missing on disk: {name} under {path}. "
                f"Refusing to hash a partial tree."
            )
        per_file[name] = sha256_file(file)
        total += file.stat().st_size
    aggregate = hashlib.sha256(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(per_file.items())).encode("utf-8")
    ).hexdigest()
    return aggregate, per_file, total


def _remote_files(repo: str, revision: str) -> dict[str, RemoteFile]:
    """Authoritative per-file size and content hash from the Hub tree API.

    Size alone is not sufficient. It catches an overlap-append, because overlapping writes make
    the file grow - but it cannot catch a resume that splices at a misaligned offset and still
    lands on the correct total length. Over a link that drops roughly two thirds of TLS
    handshakes, misaligned splices are the residual failure mode, and they produce a same-size
    corrupt file that loads and generates perfectly plausible garbage.
    """
    files: dict[str, RemoteFile] = {}
    entries = _curl_json(f"{_HF_API}/{repo}/tree/{revision}?recursive=true")
    for entry in entries:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else None
        size = int((lfs or {}).get("size") or entry.get("size") or 0)
        path = str(entry["path"])
        files[path] = RemoteFile(
            path=path,
            size=size,
            sha256=str(lfs["oid"]) if lfs and lfs.get("oid") else None,
            git_blob_sha1=None if lfs else (str(entry["oid"]) if entry.get("oid") else None),
        )
    return files


def git_blob_sha1(path: Path) -> str:
    """Git's object id for a blob: ``sha1(b"blob <size>\\0" + content)``.

    Git hashes a length-prefixed header along with the content, so this is not the SHA-1 of the
    file's bytes and cannot be compared against one.
    """
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(local: Path, remote: RemoteFile) -> dict[str, Any]:
    """Check one downloaded file against the Hub's record of it."""
    actual_size = local.stat().st_size if local.exists() else -1
    record: dict[str, Any] = {
        "method": remote.verification_method,
        "expected_size": remote.size,
        "actual_size": actual_size,
        "size_ok": actual_size == remote.size,
        "hash_ok": None,
        "expected_hash": remote.sha256 or remote.git_blob_sha1,
        "actual_hash": None,
    }
    if actual_size != remote.size:
        record["ok"] = False
        return record

    if remote.sha256:
        record["actual_hash"] = sha256_file(local)
        record["hash_ok"] = record["actual_hash"] == remote.sha256.lower()
    elif remote.git_blob_sha1:
        record["actual_hash"] = git_blob_sha1(local)
        record["hash_ok"] = record["actual_hash"] == remote.git_blob_sha1.lower()

    record["ok"] = record["size_ok"] and record["hash_ok"] is not False
    return record


def _publisher_quantization(ir_dir: Path) -> dict[str, Any]:
    """Prefer artifact-embedded ``rt_info``; fall back to the publisher's README.

    Hub-published OpenVINO IRs frequently omit ``rt_info`` compression metadata even when the
    card documents NNCF parameters. Recording "available: false" while the README states
    INT4_ASYM / ratio 0.8 / group 128 would understate what we know, so the README is a
    labelled fallback (``source: readme``), never silently preferred over the artifact.
    """
    xml_path = ir_dir / "openvino_model.xml"
    if xml_path.exists():
        found: dict[str, Any] = {}
        for _event, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag == "rt_info":
                for child in elem.iter():
                    value = child.attrib.get("value")
                    if value is not None and child.tag != "rt_info":
                        found[child.tag] = value
                elem.clear()
                break
            elem.clear()
        if found:
            found["available"] = True
            found["source"] = "rt_info"
            return found
    return publisher_quantization_from_readme(ir_dir)


def fetch(
    *,
    repo: str,
    revision: str,
    out_dir: Path,
    ladder_position: str,
    allow: tuple[str, ...] = (),
    spec_path: Path | None = None,
    verify_only: bool = False,
) -> FetchedModelSpec:
    """Fetch every file of ``repo`` at ``revision`` into ``out_dir`` and record provenance.

    With ``verify_only``, nothing is transferred: files already on disk are checked against the
    Hub and a spec is written only if every one of them passes. This is the path for certifying a
    download performed by an earlier build of this module, whose spec would record weaker
    (size-only) provenance.
    """
    started = time.time()
    # §6.6: two concurrent fetches to the same IR directory already produced a same-path corrupt
    # .bin that still loaded. The second writer is refused, not raced. Lock lives *beside* the
    # IR dir so it cannot enter hash_ir_dir's aggregate.
    with exclusive(out_dir.parent / f".{out_dir.name}.fetch.lock"):
        info = _curl_json(f"{_HF_API}/{repo}/revision/{revision}")
        resolved = str(info.get("sha") or revision)
        names = [str(entry["rfilename"]) for entry in info["siblings"]]
        if allow:
            names = [n for n in names if n.endswith(allow)]

        remote = _remote_files(repo, resolved)
        attempts: dict[str, int] = {}
        verification: dict[str, dict[str, Any]] = {}
        for name in sorted(names):
            if verify_only:
                attempts[name] = 0
                continue
            attempts[name], verification[name] = _download(
                repo, resolved, name, out_dir / name, remote.get(name)
            )

        # Re-verify every file at the end, not just at download time: a file downloaded early could
        # have been disturbed while later ones transferred.
        for name in sorted(names):
            if name in remote:
                verification[name] = verify_file(out_dir / name, remote[name])

        failed = {name: rec for name, rec in verification.items() if not rec.get("ok")}
        if failed:
            raise SeamError(
                f"refusing to write a FetchedModelSpec for an unverified IR: {failed}. "
                f"A ModelSpec is a provenance claim; emitting one for bytes that do not match the "
                f"publisher would certify corruption. Re-fetch with exactly ONE fetch process."
            )

        aggregate, per_file, total_bytes = hash_ir_dir(out_dir, sorted(names))
        spec = FetchedModelSpec(
            name=repo.split("/")[-1],
            revision=resolved,
            ladder_position=ladder_position,
            source="pre-converted",
            source_repo=repo,
            download_method="curl --retry-all-errors -C - (single connection, resumable)",
            self_converted=False,
            publisher_quantization=_publisher_quantization(out_dir),
            ir_dir=str(out_dir),
            ir_sha256=aggregate,
            ir_files=per_file,
            ir_bytes=total_bytes,
            fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            fetch_duration_s=time.time() - started,
            transfer_attempts=attempts,
            verification=verification,
        )

        if spec_path is not None:
            with exclusive(spec_path):
                spec_path.parent.mkdir(parents=True, exist_ok=True)
                spec_path.write_text(
                    yaml.safe_dump(asdict(spec), sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
        return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--ladder-position", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify files already on disk against the Hub; transfer nothing",
    )
    args = parser.parse_args(argv)

    root = repo_root(Path(__file__).parent)
    slug = args.repo.split("/")[-1]
    out_dir = args.out or (root / "models" / slug)
    spec_path = args.spec or (root / "configs" / "models" / f"{slug}.yaml")

    spec = fetch(
        repo=args.repo,
        revision=args.revision,
        out_dir=out_dir,
        ladder_position=args.ladder_position,
        spec_path=spec_path,
        verify_only=args.verify_only,
    )
    summary = {k: v for k, v in asdict(spec).items() if k not in {"ir_files", "transfer_attempts"}}
    print(json.dumps(summary, indent=2))
    print(f"\nModelSpec written to {spec_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
