"""Analysis-blinding boundary tests (spec §8, blueprint §5.1 "analysis drift").

The rule: *analysis consumes ``blinded_label``; analysis code must not import the condition
mapping.* Analysis does not exist yet - it arrives with M3/M6 - so these tests are the boundary's
tripwire, installed now so that the first analysis module written cannot quietly cross it.

The static-import checks pass trivially while ``seam/analysis/`` is empty. That is intentional and
is the point: the check must already be in the suite when the directory appears.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from seam.blinding import blinded_label_for, get_or_create_salt, record_unblind_entry
from seam.errors import SeamError

SEAM_PACKAGE = Path(__file__).resolve().parents[1] / "seam"
ANALYSIS_DIR = SEAM_PACKAGE / "analysis"

#: Modules that constitute the condition mapping. Analysis may not import these.
FORBIDDEN_FOR_ANALYSIS = {"seam.blinding", "blinding"}


def _imported_modules(path: Path) -> set[str]:
    """Return every module name imported by a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _analysis_modules() -> list[Path]:
    return sorted(ANALYSIS_DIR.rglob("*.py")) if ANALYSIS_DIR.is_dir() else []


# ==================================================================================================
# The boundary
# ==================================================================================================


def test_analysis_does_not_import_the_condition_mapping() -> None:
    """Spec §8: "Analysis code must not import the condition mapping.\" """
    offenders: list[str] = []
    for module in _analysis_modules():
        forbidden = _imported_modules(module) & FORBIDDEN_FOR_ANALYSIS
        if forbidden:
            offenders.append(
                f"{module.relative_to(SEAM_PACKAGE.parent)} imports {sorted(forbidden)}"
            )

    assert not offenders, (
        "analysis code must not import the condition mapping; a separate explicit unblind step "
        "joins labels:\n" + "\n".join(offenders)
    )


def test_analysis_does_not_reference_condition_label() -> None:
    """Analysis must consume ``blinded_label`` only, never ``condition_label``."""
    offenders = [
        str(module.relative_to(SEAM_PACKAGE.parent))
        for module in _analysis_modules()
        if "condition_label" in module.read_text(encoding="utf-8")
    ]
    assert (
        not offenders
    ), f"analysis code referenced condition_label; it must consume blinded_label only: {offenders}"


def test_blinding_module_documents_the_prohibition() -> None:
    """The docstring is load-bearing: it is where a future author learns the rule."""
    source = (SEAM_PACKAGE / "blinding.py").read_text(encoding="utf-8")
    assert "ANALYSIS CODE MUST NOT IMPORT THIS MODULE" in source


# ==================================================================================================
# Label derivation
# ==================================================================================================


def test_blinded_label_is_deterministic_for_a_given_salt() -> None:
    """Stable labels are what let analysis group runs by arm without knowing which arm is which."""
    salt = "0" * 64
    assert blinded_label_for("A", salt=salt) == blinded_label_for("A", salt=salt)


def test_blinded_label_differs_between_conditions() -> None:
    salt = "0" * 64
    assert blinded_label_for("A", salt=salt) != blinded_label_for("B", salt=salt)


def test_blinded_label_changes_with_salt() -> None:
    """Without a salt, a tiny label space would be trivially invertible."""
    assert blinded_label_for("A", salt="0" * 64) != blinded_label_for("A", salt="f" * 64)


def test_blinded_label_does_not_leak_the_condition() -> None:
    label = blinded_label_for("reference_gpt5_high", salt="0" * 64)
    assert "gpt5" not in label
    assert "reference" not in label
    assert label.startswith("cond_")


def test_blinded_label_matches_the_schema_pattern() -> None:
    import re

    assert re.fullmatch(r"cond_[0-9a-f]{4,}", blinded_label_for("A", salt="0" * 64))


def test_empty_condition_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        blinded_label_for("", salt="0" * 64)


# ==================================================================================================
# Salt and unblind map
# ==================================================================================================


def test_salt_is_created_once_and_reused(tmp_path: Path) -> None:
    """Rotating the salt would orphan every previously emitted blinded_label."""
    first = get_or_create_salt(tmp_path)
    assert get_or_create_salt(tmp_path) == first
    assert (tmp_path / "raw" / "_blinding" / "salt.txt").is_file()


def test_empty_salt_file_is_a_hard_failure(tmp_path: Path) -> None:
    salt_path = tmp_path / "raw" / "_blinding" / "salt.txt"
    salt_path.parent.mkdir(parents=True)
    salt_path.write_text("\n", encoding="utf-8")

    with pytest.raises(SeamError, match="orphan"):
        get_or_create_salt(tmp_path)


def test_unblind_map_accumulates_entries(tmp_path: Path) -> None:
    import json

    record_unblind_entry("A", "cond_aaaa", repo_root=tmp_path)
    record_unblind_entry("B", "cond_bbbb", repo_root=tmp_path)

    data = json.loads(
        (tmp_path / "raw" / "_blinding" / "unblind_map.json").read_text(encoding="utf-8")
    )
    assert data["map"] == {"cond_aaaa": "A", "cond_bbbb": "B"}


def test_unblind_map_is_idempotent_for_the_same_pair(tmp_path: Path) -> None:
    record_unblind_entry("A", "cond_aaaa", repo_root=tmp_path)
    record_unblind_entry("A", "cond_aaaa", repo_root=tmp_path)  # must not raise


def test_conflicting_mapping_is_a_hard_failure(tmp_path: Path) -> None:
    """A collision or salt change compromises the blind and must be investigated, not absorbed."""
    record_unblind_entry("A", "cond_aaaa", repo_root=tmp_path)
    with pytest.raises(SeamError, match="compromised"):
        record_unblind_entry("B", "cond_aaaa", repo_root=tmp_path)
