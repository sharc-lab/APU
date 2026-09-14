"""AM-014: raw/ payload ceiling is enforced mechanically."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from seam.raw_retention import (
    RawRetentionExceededError,
    assert_raw_under_ceiling,
    measure_raw_payload_bytes,
)


def test_repo_config_declares_100mb_ceiling() -> None:
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "configs" / "repo.yaml").read_text(encoding="utf-8"))
    assert data["raw_retention"]["ceiling_mb"] == 100


def test_current_raw_payload_is_under_ceiling() -> None:
    """Mechanical AM-014 check against the real repo ``raw/`` tree."""
    root = Path(__file__).resolve().parents[1]
    total = assert_raw_under_ceiling(root)
    # M2.1 adds a ~2.4 MB S1 samples.ndjson; still well under the 100 MB AM-014 ceiling.
    assert total < 100_000_000


def test_blinding_dir_is_excluded_from_size(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "run_a").mkdir(parents=True)
    (raw / "run_a" / "manifest.json").write_bytes(b"x" * 1000)
    blind = raw / "_blinding"
    blind.mkdir()
    (blind / "unblind_map.json").write_bytes(b"y" * 50_000)
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "repo.yaml").write_text(
        yaml.dump(
            {
                "raw_retention": {
                    "ceiling_mb": 100,
                    "exclude_globs": ["_blinding/**"],
                }
            }
        ),
        encoding="utf-8",
    )
    total = measure_raw_payload_bytes(tmp_path, exclude_globs=["_blinding/**"])
    assert total == 1000


def test_exceeding_ceiling_raises(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "big"
    raw.mkdir(parents=True)
    # 2 MB payload against a 1 MB ceiling (ceiling_mb uses decimal MB = 1e6 bytes).
    (raw / "blob.bin").write_bytes(b"z" * 2_000_000)
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "repo.yaml").write_text(
        yaml.dump({"raw_retention": {"ceiling_mb": 1, "exclude_globs": []}}),
        encoding="utf-8",
    )
    with pytest.raises(RawRetentionExceededError, match="AM-014"):
        assert_raw_under_ceiling(tmp_path)


def test_assert_returns_bytes_when_under(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "run"
    raw.mkdir(parents=True)
    (raw / "manifest.json").write_bytes(b"{}" * 10)
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "repo.yaml").write_text(
        yaml.dump({"raw_retention": {"ceiling_mb": 100, "exclude_globs": []}}),
        encoding="utf-8",
    )
    total = assert_raw_under_ceiling(tmp_path)
    assert total == 20
