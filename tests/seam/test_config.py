"""Config resolution and hashing tests (spec §8: configs are YAML, fully resolved and hashed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from seam.config import (
    ResolvedConfig,
    apply_overrides,
    load_platform_config,
    load_yaml,
    resolve_config,
)
from seam.errors import ConfigError
from seam.hashing import canonical_json, sha256_file, sha256_json, sha256_tree

# ==================================================================================================
# Loading
# ==================================================================================================


def test_load_yaml_reads_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\nb: {c: 2}\n", encoding="utf-8")
    assert load_yaml(path) == {"a": 1, "b": {"c": 2}}


def test_load_yaml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(tmp_path / "absent.yaml")


def test_load_yaml_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_yaml(path)


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(path)


def test_load_yaml_rejects_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse"):
        load_yaml(path)


# ==================================================================================================
# Overrides
# ==================================================================================================


def test_overrides_merge_mappings_recursively() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    assert apply_overrides(base, {"a": {"c": 99}}) == {"a": {"b": 1, "c": 99}, "d": 3}


def test_overrides_replace_lists_wholesale() -> None:
    """Lists must not concatenate: a half-overridden parameter list is unreadable in a manifest."""
    assert apply_overrides({"x": [1, 2, 3]}, {"x": [9]}) == {"x": [9]}


def test_overrides_do_not_mutate_the_base() -> None:
    base = {"a": {"b": 1}}
    apply_overrides(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


def test_later_paths_override_earlier(tmp_path: Path) -> None:
    first, second = tmp_path / "1.yaml", tmp_path / "2.yaml"
    first.write_text("a: 1\nb: 1\n", encoding="utf-8")
    second.write_text("b: 2\n", encoding="utf-8")

    resolved = resolve_config([first, second], repo_root=tmp_path)
    assert resolved.data == {"a": 1, "b": 2}
    assert resolved.sources == ("1.yaml", "2.yaml")


def test_resolve_config_requires_at_least_one_path() -> None:
    with pytest.raises(ConfigError, match="at least one"):
        resolve_config([])


# ==================================================================================================
# Hashing
# ==================================================================================================


def test_config_hash_is_stable_across_key_order(tmp_path: Path) -> None:
    """Reformatting a config must not report a spurious condition change."""
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("x: 1\ny: 2\n", encoding="utf-8")
    b.write_text("y: 2\nx: 1\n", encoding="utf-8")

    assert (
        resolve_config([a], repo_root=tmp_path).config_hash
        == resolve_config([b], repo_root=tmp_path).config_hash
    )


def test_config_hash_changes_with_content(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("x: 1\n", encoding="utf-8")
    b.write_text("x: 2\n", encoding="utf-8")

    assert (
        resolve_config([a], repo_root=tmp_path).config_hash
        != resolve_config([b], repo_root=tmp_path).config_hash
    )


def test_overrides_change_the_hash_and_are_recorded(tmp_path: Path) -> None:
    """An override must be visible in the manifest, not silently folded into the file hash."""
    path = tmp_path / "c.yaml"
    path.write_text("x: 1\n", encoding="utf-8")

    plain = resolve_config([path], repo_root=tmp_path)
    overridden = resolve_config([path], overrides={"x": 2}, repo_root=tmp_path)

    assert overridden.config_hash != plain.config_hash
    assert any(source.startswith("<overrides:") for source in overridden.sources)


def test_canonical_json_rejects_nan() -> None:
    """NaN is not valid JSON; allowing it would produce manifests other tools cannot read."""
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_sha256_json_is_order_independent() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_sha256_file_hashes_bytes(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello")
    # SHA-256 of b"hello".
    assert sha256_file(path) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_file_raises_on_missing(tmp_path: Path) -> None:
    """Never a sentinel: an absent provenance artifact must fail the run."""
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "absent")


def test_sha256_tree_covers_content_and_paths(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("b", encoding="utf-8")

    baseline = sha256_tree(root)
    assert sha256_tree(root) == baseline, "tree hash must be deterministic"

    (root / "a.txt").write_text("changed", encoding="utf-8")
    content_changed = sha256_tree(root)
    assert content_changed != baseline

    (root / "a.txt").rename(root / "renamed.txt")
    assert sha256_tree(root) != content_changed, "renaming a file must change the tree hash"


# ==================================================================================================
# Dotted access
# ==================================================================================================


def test_get_and_require_traverse_dotted_paths(fake_config: ResolvedConfig) -> None:
    assert fake_config.get("topology.expected.n_p_cores") == 4
    assert fake_config.require("platform_id") == "aipc-c1"
    assert fake_config.get("does.not.exist") is None
    assert fake_config.get("does.not.exist", "fallback") == "fallback"


def test_require_rejects_missing_key(fake_config: ResolvedConfig) -> None:
    with pytest.raises(ConfigError, match="missing"):
        fake_config.require("no.such.key")


def test_require_rejects_null_because_null_means_not_yet_measured(
    fake_config: ResolvedConfig,
) -> None:
    """``null`` is "not yet measured", so a caller must never silently receive one."""
    with pytest.raises(ConfigError, match="not yet measured"):
        fake_config.require("thermal.warmup_s")


# ==================================================================================================
# Platform config
# ==================================================================================================


def test_load_platform_config_reads_the_committed_file(fake_repo: Path) -> None:
    config = load_platform_config("aipc-c1", repo_root=fake_repo)
    assert config.require("platform_id") == "aipc-c1"
    assert config.require("identity.family") == "Panther Lake"


def test_load_platform_config_rejects_identity_mismatch(fake_repo: Path) -> None:
    """Catches a copied-and-renamed config, which would misattribute one machine's measurements."""
    copied = fake_repo / "configs" / "platforms" / "other-box.yaml"
    copied.write_text(
        (fake_repo / "configs" / "platforms" / "aipc-c1.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="refusing to attribute"):
        load_platform_config("other-box", repo_root=fake_repo)


def test_load_platform_config_rejects_missing_platform(fake_repo: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_platform_config("no-such-platform", repo_root=fake_repo)


# ==================================================================================================
# Prohibited numbers (spec §9.4)
# ==================================================================================================


def test_bandwidth_is_not_stated_anywhere_in_platform_config(
    real_platform_config: ResolvedConfig,
) -> None:
    """Spec §9.4 / blueprint §16.3(4): no document may state a bandwidth number before M2.4."""
    assert real_platform_config.get("memory.bandwidth_gbps_measured") is None
    assert real_platform_config.get("memory.bandwidth_measured_by_run_id") is None

    # The derived ~120 GB/s figure must not appear as a readable value anywhere in the config.
    flat = canonical_json(real_platform_config.data).decode("utf-8")
    assert "120" not in flat.replace("26200", ""), (
        "a bandwidth-like number appeared in the platform config; the derived ~120 GB/s figure is "
        "withheld by design until M2.4 measures it"
    )


def test_npu_tops_is_marked_peak_and_achieved_is_null(
    real_platform_config: ResolvedConfig,
) -> None:
    """Spec §9.4: 50 TOPS is peak INT8 and must never be reported as achieved."""
    npu = real_platform_config.get("accelerators.npu")
    assert npu["peak_int8_tops"] == 50
    assert npu["achieved_tops_measured"] is None
    assert npu["_provenance"]["peak_int8_tops"] == "vendor_stated_peak_not_achieved"


def test_platform_has_four_local_targets_and_no_dgpu(
    real_platform_config: ResolvedConfig,
) -> None:
    local = real_platform_config.get("targets.local")
    assert local == ["cpu-p", "cpu-lpe", "igpu", "npu"]
    assert "dgpu" not in local
