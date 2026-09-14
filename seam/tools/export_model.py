"""Scripted INT4 OpenVINO IR export (spec §5.3).

Spec §5.3: "Conversion is scripted and reproducible; never hand-converted." A hand-run
``optimum-cli`` invocation leaves no record of which flags produced the artifact, and quantization
flags change model behavior - so an unrecorded export makes every downstream number
unreproducible.

This tool runs the export and writes a ``ModelSpec`` capturing the model name, the pinned revision
SHA, the quantization configuration, the SHA-256 of every IR file, and the exact command line, so
the artifact can be regenerated or falsified later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from seam.gitinfo import repo_root

__all__ = ["ModelSpec", "export"]


@dataclass(slots=True)
class ModelSpec:
    """Everything needed to reproduce or verify the exported IR."""

    name: str
    revision: str
    task: str
    quantization: str
    quantization_config: dict[str, Any]
    export_command: list[str]
    ir_dir: str
    ir_sha256: str
    ir_files: dict[str, str]
    ir_bytes: int
    exported_utc: str
    openvino_version: str
    optimum_intel_version: str
    export_duration_s: float


def _versions() -> tuple[str, str]:
    import importlib.metadata as md

    def _get(pkg: str) -> str:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "absent"

    return _get("openvino"), _get("optimum-intel")


def _hash_dir(path: Path) -> tuple[str, dict[str, str], int]:
    """SHA-256 per file plus a stable aggregate over the sorted (name, hash) pairs."""
    per_file: dict[str, str] = {}
    total = 0
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        per_file[file.relative_to(path).as_posix()] = digest
        total += file.stat().st_size
    aggregate = hashlib.sha256(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(per_file.items())).encode("utf-8")
    ).hexdigest()
    return aggregate, per_file, total


def export(
    *,
    model_id: str,
    revision: str,
    out_dir: Path,
    weight_format: str = "int4",
    task: str = "text-generation-with-past",
    group_size: int = 128,
    ratio: float = 1.0,
    spec_path: Path | None = None,
) -> ModelSpec:
    """Export ``model_id`` to INT4 OpenVINO IR and write its :class:`ModelSpec`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "optimum.commands.optimum_cli",
        "export",
        "openvino",
        "--model",
        model_id,
        "--revision",
        revision,
        "--task",
        task,
        "--weight-format",
        weight_format,
        "--group-size",
        str(group_size),
        "--ratio",
        str(ratio),
        str(out_dir),
    ]

    started = time.time()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    duration = time.time() - started
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-4000:]
        raise RuntimeError(
            f"optimum export failed (exit {completed.returncode}). Last output:\n{tail}"
        )

    aggregate, per_file, total_bytes = _hash_dir(out_dir)
    openvino_version, optimum_version = _versions()

    spec = ModelSpec(
        name=model_id,
        revision=revision,
        task=task,
        quantization=weight_format,
        quantization_config={
            "weight_format": weight_format,
            "group_size": group_size,
            "ratio": ratio,
        },
        export_command=command,
        ir_dir=str(out_dir),
        ir_sha256=aggregate,
        ir_files=per_file,
        ir_bytes=total_bytes,
        exported_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        openvino_version=openvino_version,
        optimum_intel_version=optimum_version,
        export_duration_s=duration,
    )

    if spec_path is not None:
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(
            yaml.safe_dump(asdict(spec), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--weight-format", default="int4")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--spec", type=Path, default=None)
    args = parser.parse_args(argv)

    root = repo_root(Path(__file__).parent)
    slug = args.model.split("/")[-1].replace(".", "_").replace("-", "_").lower()
    out_dir = args.out or (root / "models" / f"{slug}_{args.weight_format}_ov")
    spec_path = args.spec or (root / "configs" / "models" / f"{slug}_{args.weight_format}.yaml")

    spec = export(
        model_id=args.model,
        revision=args.revision,
        out_dir=out_dir,
        weight_format=args.weight_format,
        group_size=args.group_size,
        ratio=args.ratio,
        spec_path=spec_path,
    )
    print(json.dumps({k: v for k, v in asdict(spec).items() if k != "ir_files"}, indent=2))
    print(f"\nModelSpec written to {spec_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
