"""Pin an already-downloaded OpenVINO IR as a FetchedModelSpec YAML.

The 4B pin was produced by ``seam.tools.fetch_model`` at download time. Later rungs (8B, 14B)
arrive by ``hf download`` (or any other method) and still need the same provenance schema:
per-file hashes, the self-constructed ``ir_sha256`` roll-up, and Hub verification
(git-blob-sha1 for non-LFS, sha256 / ``lfs.oid`` for ``.bin`` and other LFS files).

This tool hashes what is on disk and checks it against the Hub at a resolved revision. It does
not transfer bytes. ``--validate-against`` is the gate on the roll-up rule: run it over the 4B
IR and refuse to emit a new pin if ``ir_sha256`` or any per-file hash disagrees with the
known-good yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seam.errors import SeamError  # noqa: E402
from seam.locks import exclusive  # noqa: E402
from seam.tools.fetch_model import (  # noqa: E402
    FetchedModelSpec,
    _curl_json,
    _publisher_quantization,
    _remote_files,
    hash_ir_dir,
    verify_file,
)

_HF_API = "https://huggingface.co/api/models"

_SPEC_FIELD_ORDER = (
    "name",
    "revision",
    "ladder_position",
    "source",
    "source_repo",
    "download_method",
    "self_converted",
    "publisher_quantization",
    "note",
    "ir_dir",
    "ir_sha256",
    "ir_files",
    "ir_bytes",
    "fetched_utc",
    "fetch_duration_s",
    "transfer_attempts",
    "verification",
)

_HASH_COMPARE_KEYS = ("ir_sha256", "ir_files", "ir_bytes", "verification")

# 8B INT4_ASYM vs 4B INT4_SYM. Required on the 8B pin; harmless to pass for other rungs.
_SIZE_VS_RECIPE_NOTE = (
    "The 4B is INT4_SYM; this is INT4_ASYM with scale_estimation on wikitext2. Any "
    "4B-vs-8B comparison confounds model size with quantization recipe. Acceptable for "
    "latency and footprint; NOT acceptable for a quality comparison without either "
    "matching recipes or stating the confound."
)


def _ordered_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key in _SPEC_FIELD_ORDER:
        if key in payload and payload[key] is not None:
            ordered[key] = payload[key]
    for key, value in payload.items():
        if key not in ordered and value is not None:
            ordered[key] = value
    return ordered


def dump_spec_yaml(payload: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        _ordered_spec(payload),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def resolve_revision_sha(repo: str, revision: str) -> tuple[str, list[str]]:
    info = _curl_json(f"{_HF_API}/{repo}/revision/{revision}")
    if not isinstance(info, dict):
        raise SeamError(f"Hub revision lookup for {repo}@{revision} was not a mapping")
    sha = info.get("sha")
    if not sha:
        raise SeamError(f"Hub revision lookup for {repo}@{revision} returned no sha")
    siblings = info.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise SeamError(f"Hub revision lookup for {repo}@{revision} returned no siblings")
    names = [str(entry["rfilename"]) for entry in siblings if isinstance(entry, dict)]
    if not names:
        raise SeamError(f"Hub revision lookup for {repo}@{revision} had empty rfilename list")
    return str(sha), names


def pin_ir(
    *,
    ir_dir: Path,
    source_repo: str,
    revision: str,
    ladder_position: str,
    download_method: str,
    source: str = "pre-converted",
    self_converted: bool = False,
    publisher_quantization: dict[str, Any] | None = None,
    note: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Hash ``ir_dir`` against Hub siblings at ``revision`` and return a spec mapping."""
    started = time.time()
    ir_dir = ir_dir.resolve()
    if not ir_dir.is_dir():
        raise SeamError(f"IR directory missing: {ir_dir}")

    resolved, names = resolve_revision_sha(source_repo, revision)
    remote = _remote_files(source_repo, resolved)

    missing = [n for n in names if not (ir_dir / n).is_file()]
    if missing:
        raise SeamError(
            f"refusing to pin an incomplete IR under {ir_dir}: missing {missing}. "
            "Hub siblings are the include-list; extra local files (.cache) are ignored."
        )

    aggregate, per_file, total_bytes = hash_ir_dir(ir_dir, names)

    attempts: dict[str, int] = dict.fromkeys(sorted(names), 0)
    verification: dict[str, dict[str, Any]] = {}
    for name_i in sorted(names):
        if name_i not in remote:
            raise SeamError(
                f"Hub tree has no entry for sibling {name_i!r} at {source_repo}@{resolved}"
            )
        verification[name_i] = verify_file(ir_dir / name_i, remote[name_i])

    failed = {n: rec for n, rec in verification.items() if not rec.get("ok")}
    if failed:
        raise SeamError(
            f"refusing to write a FetchedModelSpec for an unverified IR: {failed}. "
            "A ModelSpec is a provenance claim; emitting one for bytes that do not match "
            "the publisher would certify corruption."
        )

    quant = publisher_quantization
    if quant is None:
        quant = _publisher_quantization(ir_dir)

    spec = FetchedModelSpec(
        name=name or source_repo.split("/")[-1],
        revision=resolved,
        ladder_position=ladder_position,
        source=source,
        source_repo=source_repo,
        download_method=download_method,
        self_converted=self_converted,
        publisher_quantization=quant,
        ir_dir=str(ir_dir),
        ir_sha256=aggregate,
        ir_files=per_file,
        ir_bytes=total_bytes,
        fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        fetch_duration_s=time.time() - started,
        transfer_attempts=attempts,
        verification=verification,
    )
    payload = asdict(spec)
    if note:
        payload["note"] = note
    return _ordered_spec(payload)


def compare_hash_fields(generated: Mapping[str, Any], known: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the load-bearing pin fields. Timestamps and download_method are excluded."""
    mismatches: dict[str, Any] = {}
    if generated.get("ir_sha256") != known.get("ir_sha256"):
        mismatches["ir_sha256"] = {
            "generated": generated.get("ir_sha256"),
            "known": known.get("ir_sha256"),
        }
    if generated.get("ir_bytes") != known.get("ir_bytes"):
        mismatches["ir_bytes"] = {
            "generated": generated.get("ir_bytes"),
            "known": known.get("ir_bytes"),
        }
    gen_files = dict(generated.get("ir_files") or {})
    known_files = dict(known.get("ir_files") or {})
    if gen_files != known_files:
        only_gen = sorted(set(gen_files) - set(known_files))
        only_known = sorted(set(known_files) - set(gen_files))
        value_mismatch = sorted(
            k for k in set(gen_files) & set(known_files) if gen_files[k] != known_files[k]
        )
        mismatches["ir_files"] = {
            "only_generated": only_gen,
            "only_known": only_known,
            "hash_mismatch": value_mismatch,
        }
    gen_ver = dict(generated.get("verification") or {})
    known_ver = dict(known.get("verification") or {})
    ver_mismatch: dict[str, Any] = {}
    for name in sorted(set(gen_ver) | set(known_ver)):
        if gen_ver.get(name) != known_ver.get(name):
            ver_mismatch[name] = {"generated": gen_ver.get(name), "known": known_ver.get(name)}
    if ver_mismatch:
        mismatches["verification"] = ver_mismatch
    return {
        "identical": not mismatches,
        "mismatches": mismatches,
        "generated_ir_sha256": generated.get("ir_sha256"),
        "known_ir_sha256": known.get("ir_sha256"),
        "n_ir_files": len(gen_files),
    }


_UNSET = object()


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {value!r}")


def _parse_tri_bool(value: str) -> bool | None:
    """true/false, or null/none for an explicit publisher-null (not the same as unset)."""
    lowered = value.strip().lower()
    if lowered in {"null", "none"}:
        return None
    return _parse_bool(value)


def _parse_optional_float_or_null(value: str) -> float | None:
    lowered = value.strip().lower()
    if lowered in {"null", "none"}:
        return None
    return float(value)


def _parse_optional_int_or_null(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"null", "none"}:
        return None
    return int(value)


def _parse_optional_str_or_null(value: str) -> str | None:
    lowered = value.strip().lower()
    if lowered in {"null", "none"}:
        return None
    return value


def _quant_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    """Build publisher_quantization.

    When any quant flag is set, ratio/group_size/scale_estimation/dataset are always
    present (value or explicit null). Absent means the pin was written without looking;
    null means the publisher does not use the field. ``awq`` is tri-state: omitted when
    unset, present when ``--quant-awq`` was passed (true/false/null).
    """
    awq = getattr(args, "quant_awq", _UNSET)
    quant_mode = getattr(args, "quant_mode", _UNSET)
    any_set = any(
        [
            quant_mode is not _UNSET,
            args.quant_ratio is not _UNSET,
            args.quant_group_size is not _UNSET,
            args.quant_scale_estimation is not _UNSET,
            args.quant_dataset is not _UNSET,
            awq is not _UNSET,
        ]
    )
    if not any_set:
        return None
    block: dict[str, Any] = {
        "available": True,
        "source": args.quant_source,
    }
    # mode may be an explicit null (uncompressed / no publisher quant recipe).
    if quant_mode is not _UNSET:
        block["mode"] = quant_mode
    # Always emit these four when a recipe block is being written.
    block["ratio"] = None if args.quant_ratio is _UNSET else args.quant_ratio
    block["group_size"] = None if args.quant_group_size is _UNSET else args.quant_group_size
    block["scale_estimation"] = (
        None if args.quant_scale_estimation is _UNSET else args.quant_scale_estimation
    )
    block["dataset"] = None if args.quant_dataset is _UNSET else args.quant_dataset
    if awq is not _UNSET:
        block["awq"] = awq
    return block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument(
        "--revision",
        default=None,
        help="Hub revision (SHA or branch). Default: main, or the yaml under --validate-against.",
    )
    parser.add_argument("--ladder-position", required=True)
    parser.add_argument(
        "--download-method",
        default=None,
        help="How these bytes arrived. Required when writing a spec; copied from yaml on validate.",
    )
    parser.add_argument("--source", default="pre-converted")
    parser.add_argument("--name", default=None)
    parser.add_argument("--note", default=None)
    parser.add_argument(
        "--size-vs-recipe-note",
        nargs="?",
        const=_SIZE_VS_RECIPE_NOTE,
        default=None,
        help=(
            "Attach a size/recipe confound note. With no value, uses the 4B-SYM vs 8B-ASYM "
            "default. Pass a string to use a custom note (e.g. int4-vs-int8)."
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="Write spec YAML here")
    parser.add_argument(
        "--validate-against",
        type=Path,
        default=None,
        help="Known-good FetchedModelSpec YAML. Does not write. Refuses --out.",
    )
    parser.add_argument(
        "--quant-mode",
        type=_parse_optional_str_or_null,
        default=_UNSET,
        help="Publisher mode string, or the literal 'null' (e.g. uncompressed fp16).",
    )
    parser.add_argument(
        "--quant-ratio",
        type=_parse_optional_float_or_null,
        default=_UNSET,
        help="Float, or the literal 'null' for an explicit null.",
    )
    parser.add_argument(
        "--quant-group-size",
        type=_parse_optional_int_or_null,
        default=_UNSET,
        help="Int, or the literal 'null' for an explicit null.",
    )
    parser.add_argument(
        "--quant-scale-estimation",
        type=_parse_tri_bool,
        default=_UNSET,
        help="true/false, or null for an explicit null.",
    )
    parser.add_argument(
        "--quant-dataset",
        type=_parse_optional_str_or_null,
        default=_UNSET,
        help="Dataset name, or the literal 'null' for an explicit null.",
    )
    parser.add_argument(
        "--quant-awq",
        type=_parse_tri_bool,
        default=_UNSET,
        help=(
            "Tri-state AWQ flag: true/false, or null. Omit the flag entirely to leave "
            "awq absent (nobody looked)."
        ),
    )
    parser.add_argument("--quant-source", default="readme")
    args = parser.parse_args(argv)

    if args.out is not None and args.validate_against is not None:
        parser.error("--validate-against never writes; do not pass --out (protects the 4B yaml)")

    known: dict[str, Any] | None = None
    if args.validate_against is not None:
        known_path = args.validate_against
        if not known_path.is_file():
            known_path = ROOT / known_path
        known = yaml.safe_load(known_path.read_text(encoding="utf-8"))
        if not isinstance(known, dict):
            raise SeamError(f"--validate-against {known_path} is not a mapping")

    revision = args.revision
    if revision is None:
        revision = str(known["revision"]) if known and known.get("revision") else "main"

    download_method = args.download_method
    if download_method is None:
        if known and known.get("download_method"):
            download_method = str(known["download_method"])
        elif args.out is not None:
            parser.error("--download-method is required when writing a spec")
        else:
            download_method = "unspecified (validate/hash-only)"

    note = args.note
    if args.size_vs_recipe_note is not None:
        note = args.size_vs_recipe_note if note is None else f"{note}\n{args.size_vs_recipe_note}"

    payload = pin_ir(
        ir_dir=args.ir_dir if args.ir_dir.is_absolute() else ROOT / args.ir_dir,
        source_repo=args.source_repo,
        revision=revision,
        ladder_position=args.ladder_position,
        download_method=download_method,
        source=args.source,
        publisher_quantization=_quant_from_args(args),
        note=note,
        name=args.name,
    )

    if known is not None:
        report = compare_hash_fields(payload, known)
        print(json.dumps(report, indent=2, sort_keys=True))
        # Canonical hash-field dump so a reader can see identity without timestamps.
        gen_hash = {k: payload[k] for k in _HASH_COMPARE_KEYS}
        known_hash = {k: known[k] for k in _HASH_COMPARE_KEYS if k in known}
        print("--- known hash fields ---")
        print(yaml.safe_dump(known_hash, sort_keys=True, allow_unicode=True), end="")
        print("--- generated hash fields ---")
        print(yaml.safe_dump(gen_hash, sort_keys=True, allow_unicode=True), end="")
        if not report["identical"]:
            print("FATAL: roll-up or per-file hash does not match the known-good pin", flush=True)
            return 2
        extra = {
            k: {"generated": payload.get(k), "known": known.get(k)}
            for k in (
                "name",
                "revision",
                "ladder_position",
                "source",
                "source_repo",
                "download_method",
                "self_converted",
                "publisher_quantization",
                "ir_dir",
            )
            if payload.get(k) != known.get(k)
        }
        print(
            json.dumps(
                {
                    "hash_identity": "PASS",
                    "non_hash_field_diffs": extra,
                    "note": (
                        "fetched_utc / fetch_duration_s are pin-time, not download-time; "
                        "they are expected to differ from a historical yaml."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    rendered = dump_spec_yaml(payload)
    if args.out is not None:
        out_path = args.out if args.out.is_absolute() else ROOT / args.out
        with exclusive(out_path):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(rendered, encoding="utf-8")
        print(f"wrote {out_path}", flush=True)
    print(rendered, end="")
    summary = {
        "name": payload["name"],
        "revision": payload["revision"],
        "ir_sha256": payload["ir_sha256"],
        "ir_bytes": payload["ir_bytes"],
        "n_files": len(payload["ir_files"]),
        "ir_dir": payload["ir_dir"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
