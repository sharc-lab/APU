"""AM-023: ModelSpec vs FetchedModelSpec must remain distinguishable in manifests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from seam.errors import ConfigError
from seam.model_provenance import (
    load_local_spec,
    manifest_model_block,
    publisher_quantization_from_readme,
    quantization_summary,
)


def _write_spec(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_fetched_and_exported_specs_produce_distinct_provenance_kinds(tmp_path: Path) -> None:
    exported = _write_spec(
        tmp_path / "exported.yaml",
        {
            "name": "Qwen/Qwen3-4B",
            "revision": "a" * 40,
            "quantization": "int4",
            "quantization_config": {"weight_format": "int4", "group_size": 128},
            "export_command": ["optimum-cli", "export", "openvino"],
            "ir_dir": str(tmp_path / "ir"),
            "ir_sha256": "b" * 64,
            "self_converted": True,
        },
    )
    fetched = _write_spec(
        tmp_path / "fetched.yaml",
        {
            "name": "Qwen3-4B-int4-ov",
            "revision": "c" * 40,
            "source": "pre-converted",
            "source_repo": "OpenVINO/Qwen3-4B-int4-ov",
            "download_method": "curl",
            "self_converted": False,
            "publisher_quantization": {"mode": "INT4_ASYM", "group_size": 128},
            "ir_dir": str(tmp_path / "ir2"),
            "ir_sha256": "d" * 64,
            "ladder_position": "slice-4b",
        },
    )

    exported_block = manifest_model_block(spec=load_local_spec(exported), spec_path=exported)
    fetched_block = manifest_model_block(spec=load_local_spec(fetched), spec_path=fetched)

    assert exported_block["provenance"]["kind"] == "self_exported"
    assert exported_block["provenance"]["self_converted"] is True
    assert exported_block["provenance"]["export_command"] == [
        "optimum-cli",
        "export",
        "openvino",
    ]
    assert fetched_block["provenance"]["kind"] == "pre_converted"
    assert fetched_block["provenance"]["self_converted"] is False
    assert fetched_block["provenance"]["source_repo"] == "OpenVINO/Qwen3-4B-int4-ov"
    assert fetched_block["provenance"]["export_command"] is None
    # The two paths must not collapse to identical provenance objects.
    assert exported_block["provenance"] != fetched_block["provenance"]


def test_untyped_spec_is_refused(tmp_path: Path) -> None:
    path = _write_spec(
        tmp_path / "flat.yaml",
        {
            "name": "mystery",
            "revision": "a" * 40,
            "ir_dir": str(tmp_path),
            "ir_sha256": "b" * 64,
            "quantization": "int4",
        },
    )
    with pytest.raises(ConfigError, match="does not discriminate provenance"):
        load_local_spec(path)


def test_readme_quantization_parser(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "* mode: **INT4_ASYM**\n* ratio: **0.8**\n* group_size: **128**\n",
        encoding="utf-8",
    )
    parsed = publisher_quantization_from_readme(tmp_path)
    assert parsed["available"] is True
    assert parsed["mode"] == "INT4_ASYM"
    assert parsed["ratio"] == 0.8
    assert parsed["group_size"] == 128
    assert parsed["source"] == "readme"


def test_quantization_summary_prefers_publisher_mode() -> None:
    assert (
        quantization_summary({"publisher_quantization": {"mode": "INT4_ASYM"}, "name": "x-int4-ov"})
        == "int4_asym"
    )
