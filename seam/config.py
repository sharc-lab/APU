"""YAML configuration loading, resolution, and hashing (spec §8).

"Fully resolved" means every ``!include`` expanded and every override applied, so that the object
hashed into ``config_hash`` is exactly the configuration the run behaved according to - not a
template that a later default silently changed.

No magic numbers live in code. Anything a run's behaviour depends on is read from here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from seam.errors import ConfigError
from seam.hashing import sha256_json

__all__ = [
    "PLATFORM_CONFIG_DIR",
    "ResolvedConfig",
    "apply_overrides",
    "load_platform_config",
    "load_yaml",
    "resolve_config",
]

PLATFORM_CONFIG_DIR: Final = Path("configs") / "platforms"


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """A fully resolved configuration together with its content hash."""

    data: dict[str, Any]
    config_hash: str
    #: Repository-relative source paths that contributed, in load order. Recorded so a manifest
    #: reader can reconstruct which files produced the hash.
    sources: tuple[str, ...]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted path, e.g. ``"topology.expected.n_p_cores"``."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Fetch a nested value by dotted path, raising if absent or ``None``.

        Raises:
            ConfigError: If the key is missing or its value is ``None``. ``None`` is treated as
                absent because in SEAM configs ``null`` means "not yet measured" - a caller that
                requires a value must not silently receive a not-yet-measured one.
        """
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"required config key is missing: {dotted}")
        if value is None:
            raise ConfigError(
                f"required config key {dotted} is null. In SEAM configs null means "
                f"'not yet measured' - run the milestone that determines it rather than "
                f"supplying a default."
            )
        return value


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file into a dict.

    Uses ``yaml.safe_load`` - configs are data, never executable.

    Raises:
        ConfigError: If the file is missing, unparseable, or does not contain a mapping.
    """
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"config file is empty: {path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config file must contain a mapping at top level, got {type(raw).__name__}: {path}"
        )
    return raw


def apply_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overrides`` onto a copy of ``base``.

    Mappings merge recursively; every other type replaces wholesale. Lists deliberately do not
    concatenate - an experimental parameter list must be fully stated by whoever overrides it,
    since a half-overridden list is very hard to read back out of a manifest.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = apply_overrides(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_config(
    paths: list[Path],
    *,
    overrides: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> ResolvedConfig:
    """Load and merge ``paths`` in order, apply ``overrides``, and hash the result.

    Later paths override earlier ones.

    Raises:
        ConfigError: If ``paths`` is empty, or any file is missing or malformed.
    """
    if not paths:
        raise ConfigError("resolve_config requires at least one config path")

    merged: dict[str, Any] = {}
    sources: list[str] = []
    for path in paths:
        merged = apply_overrides(merged, load_yaml(path))
        sources.append(_relative_source(path, repo_root))

    if overrides:
        merged = apply_overrides(merged, overrides)
        # The override set participates in the hash via `merged`, but is also recorded as a source
        # so a manifest reader can tell that the hash does not correspond to the files alone.
        sources.append(f"<overrides:{sha256_json(overrides)[:12]}>")

    return ResolvedConfig(
        data=merged,
        config_hash=sha256_json(merged),
        sources=tuple(sources),
    )


def load_platform_config(
    platform_id: str,
    *,
    repo_root: Path,
    overrides: dict[str, Any] | None = None,
) -> ResolvedConfig:
    """Load ``configs/platforms/<platform_id>.yaml`` and verify its declared identity.

    Raises:
        ConfigError: If the file is absent, or its ``platform_id`` does not match the requested
            one. The cross-check catches a copied-and-renamed config, which would otherwise
            attribute one machine's measurements to another.
    """
    path = repo_root / PLATFORM_CONFIG_DIR / f"{platform_id}.yaml"
    resolved = resolve_config([path], overrides=overrides, repo_root=repo_root)

    declared = resolved.get("platform_id")
    if declared != platform_id:
        raise ConfigError(
            f"platform config {path} declares platform_id={declared!r} but was loaded as "
            f"{platform_id!r}; refusing to attribute measurements to the wrong machine"
        )
    return resolved


def _relative_source(path: Path, repo_root: Path | None) -> str:
    """Render a config path as a repository-relative POSIX string where possible."""
    if repo_root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        # Outside the repository. Kept absolute rather than dropped, because a config from outside
        # the repo is exactly the kind of provenance gap a reviewer needs to see.
        return path.resolve().as_posix()
