"""AM-014: threshold-based ``raw/`` retention policy.

While total payload size is under the configured ceiling, sealed runs are committed
to git. Above the ceiling, payloads move off-repo and ``raw/MANIFEST.sha256`` is the
in-repo audit index. The ceiling is enforced mechanically here so the transition is
not remembered ad hoc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from seam.errors import SeamError

DEFAULT_REPO_CONFIG = Path("configs/repo.yaml")


class RawRetentionExceededError(SeamError):
    """Total ``raw/`` payload size exceeds the AM-014 ceiling."""


def load_raw_retention_policy(
    repo_root: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    path = (repo_root / DEFAULT_REPO_CONFIG) if config_path is None else config_path
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "raw_retention" not in data:
        msg = f"raw_retention policy missing from {path}"
        raise SeamError(msg)
    policy = data["raw_retention"]
    if not isinstance(policy, dict):
        msg = f"raw_retention must be a mapping in {path}"
        raise SeamError(msg)
    return policy


def _excluded(rel: Path, exclude_globs: list[str]) -> bool:
    text = rel.as_posix()
    for pattern in exclude_globs:
        if rel.match(pattern) or Path(text).match(pattern):
            return True
        # Path.match is relative to pattern shape; also accept prefix-style globs.
        if pattern.endswith("/**"):
            prefix = pattern[: -len("/**")]
            if text == prefix or text.startswith(prefix + "/"):
                return True
    return False


def measure_raw_payload_bytes(
    repo_root: Path,
    *,
    exclude_globs: list[str] | None = None,
) -> int:
    """Sum file sizes under ``raw/``, excluding blinding secrets."""
    raw = repo_root / "raw"
    if not raw.is_dir():
        return 0
    globs = exclude_globs if exclude_globs is not None else ["_blinding/**"]
    total = 0
    for path in raw.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(raw)
        if _excluded(rel, globs):
            continue
        total += path.stat().st_size
    return total


def assert_raw_under_ceiling(repo_root: Path, config_path: Path | None = None) -> int:
    """Return payload bytes if under ceiling; raise if over.

    This is the mechanical AM-014 gate. Call from CI / pre-commit / tests.
    """
    policy = load_raw_retention_policy(repo_root, config_path=config_path)
    ceiling_mb = policy["ceiling_mb"]
    if not isinstance(ceiling_mb, int | float) or ceiling_mb <= 0:
        msg = f"raw_retention.ceiling_mb must be a positive number, got {ceiling_mb!r}"
        raise SeamError(msg)
    exclude = policy.get("exclude_globs", ["_blinding/**"])
    if not isinstance(exclude, list):
        msg = "raw_retention.exclude_globs must be a list"
        raise SeamError(msg)
    total = measure_raw_payload_bytes(repo_root, exclude_globs=[str(g) for g in exclude])
    ceiling_bytes = int(float(ceiling_mb) * 1_000_000)
    if total > ceiling_bytes:
        raise RawRetentionExceededError(
            f"raw/ payload is {total} bytes ({total / 1_000_000:.3f} MB), "
            f"exceeding AM-014 ceiling of {ceiling_mb} MB ({ceiling_bytes} bytes). "
            "Move payloads to an external checksummed archive and retain "
            "raw/MANIFEST.sha256 as the in-repo audit index."
        )
    return total
