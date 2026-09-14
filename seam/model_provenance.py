"""Local-model provenance for run manifests (AM-023).

Spec §5.3 requires scripted, reproducible conversion. Obtaining a *pre-converted* OpenVINO IR
is a documented weakening of that rule: the quantization parameters were chosen by the publisher,
not by us. The run-manifest ``model`` block therefore carries an explicit ``provenance`` object
that discriminates the two paths - flattening them into ``name/revision/quantization/ir_sha256``
alone would make the weakening invisible to analysis.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml

from seam.errors import ConfigError
from seam.hashing import sha256_file

__all__ = [
    "ProvenanceKind",
    "load_local_spec",
    "manifest_model_block",
    "quantization_summary",
]

ProvenanceKind = Literal["self_exported", "pre_converted"]

_README_MODE = re.compile(r"mode:\s*\*\*(?P<mode>[A-Z0-9_]+)\*\*", re.IGNORECASE)
_README_RATIO = re.compile(r"ratio:\s*\*\*(?P<ratio>[0-9.]+)\*\*", re.IGNORECASE)
_README_GROUP = re.compile(r"group_size:\s*\*\*(?P<group>\d+)\*\*", re.IGNORECASE)


def load_local_spec(path: Path) -> dict[str, Any]:
    """Load a ModelSpec or FetchedModelSpec YAML and refuse an untyped blob."""
    if not path.exists():
        raise ConfigError(f"model spec absent: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"model spec {path} is not a mapping")
    kind = _kind_of(data)
    required = {"name", "revision", "ir_dir", "ir_sha256"}
    missing = required - set(data)
    if missing:
        raise ConfigError(f"model spec {path} missing required fields: {sorted(missing)}")
    if kind == "pre_converted" and data.get("self_converted") is not False:
        raise ConfigError(
            f"model spec {path} claims source=pre-converted but self_converted is not false"
        )
    if kind == "self_exported" and "export_command" not in data:
        raise ConfigError(f"self-exported model spec {path} has no export_command")
    return data


def _kind_of(spec: dict[str, Any]) -> ProvenanceKind:
    source = str(spec.get("source") or "").lower()
    if source in {"pre-converted", "pre_converted"} or spec.get("self_converted") is False:
        return "pre_converted"
    if "export_command" in spec or spec.get("self_converted") is True:
        return "self_exported"
    raise ConfigError(
        "model spec does not discriminate provenance: need either export_command "
        "(self_exported ModelSpec) or source/self_converted=false (FetchedModelSpec). "
        "Flattening the two paths would hide the §5.3 weakening (AM-023)."
    )


def quantization_summary(spec: dict[str, Any]) -> str:
    """Short label for the top-level ``model.quantization`` field."""
    if isinstance(spec.get("quantization"), str) and spec["quantization"]:
        return str(spec["quantization"])
    pub = spec.get("publisher_quantization") or {}
    if isinstance(pub, dict):
        mode = pub.get("mode") or pub.get("weight_format")
        if mode:
            return str(mode).lower()
    qcfg = spec.get("quantization_config") or {}
    if isinstance(qcfg, dict) and qcfg.get("weight_format"):
        return str(qcfg["weight_format"]).lower()
    # Last resort: the IR directory / repo name almost always carries the format.
    name = str(spec.get("name") or "")
    for token in ("int4", "int8", "fp16", "bf16"):
        if token in name.lower():
            return token
    return "unknown"


def publisher_quantization_from_readme(ir_dir: Path) -> dict[str, Any]:
    """Parse the publisher's stated NNCF parameters out of the model-card README.

    Used when the IR itself carries no ``rt_info`` compression block (common for Hub-published
    OpenVINO IRs). The result is labelled ``source: readme`` so it is never mistaken for
    artifact-embedded metadata.
    """
    readme = ir_dir / "README.md"
    if not readme.exists():
        return {"available": False, "reason": "README.md absent", "source": None}
    text = readme.read_text(encoding="utf-8", errors="replace")
    mode = _README_MODE.search(text)
    ratio = _README_RATIO.search(text)
    group = _README_GROUP.search(text)
    if not (mode or ratio or group):
        return {"available": False, "reason": "no NNCF parameters in README", "source": "readme"}
    return {
        "available": True,
        "source": "readme",
        "mode": mode.group("mode") if mode else None,
        "ratio": float(ratio.group("ratio")) if ratio else None,
        "group_size": int(group.group("group")) if group else None,
    }


def manifest_model_block(
    *,
    spec: dict[str, Any],
    spec_path: Path,
    runtime: str | None = "openvino-genai",
    reasoning_mode: str | None = None,
    cloud: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``model`` block for :func:`seam.manifest.emit`.

    The ``provenance.kind`` discriminator is the load-bearing field: analysis must be able to
    tell a self-exported IR from a pre-converted one without parsing the ModelSpec offline.
    """
    kind = _kind_of(spec)
    quant_config: dict[str, Any]
    if kind == "self_exported":
        quant_config = dict(spec.get("quantization_config") or {})
    else:
        quant_config = dict(spec.get("publisher_quantization") or {})

    provenance: dict[str, Any] = {
        "kind": kind,
        "self_converted": kind == "self_exported",
        "source_repo": spec.get("source_repo") if kind == "pre_converted" else spec.get("name"),
        "download_method": spec.get("download_method") if kind == "pre_converted" else None,
        "export_command": list(spec["export_command"]) if kind == "self_exported" else None,
        "quantization_config": quant_config or None,
        "ladder_position": spec.get("ladder_position"),
        "spec_path": spec_path.as_posix(),
        "spec_sha256": sha256_file(spec_path),
    }

    return {
        "name": spec.get("name"),
        "revision": spec.get("revision"),
        "quantization": quantization_summary(spec),
        "ir_sha256": spec.get("ir_sha256"),
        "runtime": runtime,
        "precision": quantization_summary(spec),
        "reasoning_mode": reasoning_mode,
        "provenance": provenance,
        "cloud": cloud,
    }
