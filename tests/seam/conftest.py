"""Shared fixtures for the SEAM test suite.

Every hardware and git interaction is mocked (spec §8: "pytest for all non-hardware logic with
hardware calls mocked"), so the suite runs on any machine and does not require Windows, elevation,
or a specific CPU.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from seam import jsonlog
from seam.config import ResolvedConfig, resolve_config
from seam.gitinfo import GitState
from seam.rawstore import make_writable_for_test

#: Repository root, resolved from this file's location rather than the working directory so the
#: suite behaves identically under `pytest`, `pytest tests/`, and an IDE runner.
REPO_ROOT = Path(__file__).resolve().parents[1]

PLATFORM_CONFIG = REPO_ROOT / "configs" / "platforms" / "aipc-c1.yaml"


@pytest.fixture(autouse=True)
def _isolate_json_sinks() -> Iterator[None]:
    """Prevent structured-log sinks leaking between tests."""
    jsonlog._clear_json_sinks()
    yield
    jsonlog._clear_json_sinks()


@pytest.fixture(autouse=True)
def _declare_isolation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declare ``local`` for the suite, since a test is never a measurement.

    :func:`seam.manifest.emit` has no default for ``isolation_mode`` and refuses to emit without
    one, which is the behaviour that keeps contended runs out of quiet comparisons. Tests emit
    manifests into throwaway directories, so the declaration is made once here rather than in
    every test -- and ``local`` is the honest value: the suite runs on a machine in normal use.
    Tests that exercise the refusal itself delete the variable.
    """
    monkeypatch.setenv("SEAM_ISOLATION_MODE", "local")


@pytest.fixture
def real_platform_config() -> ResolvedConfig:
    """The committed Platform A config, loaded as-is.

    Used by tests that assert on the *committed* state, e.g. that topology ships unverified.
    """
    return resolve_config([PLATFORM_CONFIG], repo_root=REPO_ROOT)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Iterator[Path]:
    """A throwaway repository root containing a real platform config and stub probe artifacts.

    Sealed run directories are read-only, which on Windows blocks pytest's ``tmp_path`` cleanup, so
    write permission is restored on teardown.
    """
    (tmp_path / "configs" / "platforms").mkdir(parents=True)
    shutil.copy(PLATFORM_CONFIG, tmp_path / "configs" / "platforms" / "aipc-c1.yaml")

    config = resolve_config(
        [tmp_path / "configs" / "platforms" / "aipc-c1.yaml"], repo_root=tmp_path
    )
    for relative in config.get("provenance_artifacts") or []:
        artifact = tmp_path / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"stub provenance artifact for tests: {relative}\n", encoding="utf-8")

    yield tmp_path

    make_writable_for_test(tmp_path)


@pytest.fixture
def fake_config(fake_repo: Path) -> ResolvedConfig:
    """Platform config resolved against the throwaway repository."""
    return resolve_config(
        [fake_repo / "configs" / "platforms" / "aipc-c1.yaml"], repo_root=fake_repo
    )


@pytest.fixture
def verified_config(fake_repo: Path) -> ResolvedConfig:
    """Platform config with a verified 4/4 topology, for tests that need one.

    The mapping is supplied as an explicit test override rather than read from the real config, so a
    test's expectations stay independent of whatever the committed measurement currently says.
    """
    return resolve_config(
        [fake_repo / "configs" / "platforms" / "aipc-c1.yaml"],
        overrides={
            "topology": {
                "verified": True,
                "p_cpus": [0, 1, 2, 3],
                "lpe_cpus": [4, 5, 6, 7],
                "measured": {"run_id": "00000000-0000-4000-8000-000000000000"},
            }
        },
        repo_root=fake_repo,
    )


@pytest.fixture
def unverified_config(fake_repo: Path) -> ResolvedConfig:
    """Platform config with **no** verified mapping.

    Since M1 the committed config carries a measured mapping, so the unverified state has to be
    constructed explicitly. It still has to be tested: refusing to guess the P/LP-E split is the
    behaviour spec §4 requires, and it would otherwise lose its coverage the moment the real config
    stopped supplying it by accident.
    """
    return resolve_config(
        [fake_repo / "configs" / "platforms" / "aipc-c1.yaml"],
        overrides={
            "topology": {
                "verified": False,
                "p_cpus": None,
                "lpe_cpus": None,
                "measured": {"run_id": None},
            }
        },
        repo_root=fake_repo,
    )


@pytest.fixture
def clean_git_state() -> GitState:
    return GitState(sha="a" * 40, dirty=False, branch="main", dirty_files=())


@pytest.fixture
def dirty_git_state() -> GitState:
    return GitState(
        sha="b" * 40,
        dirty=True,
        branch="main",
        dirty_files=(" M seam/manifest.py", "?? scratch.txt"),
    )


@pytest.fixture
def patch_git(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return a helper that patches git capture in :mod:`seam.manifest` to a fixed state."""

    def _patch(state: GitState) -> None:
        # Accepts and discards whatever keyword arguments the caller passes (currently `cwd`), so
        # the stub does not have to track the real signature.
        monkeypatch.setattr("seam.manifest.capture_git_state", lambda **_kwargs: state)

    return _patch


@pytest.fixture
def minimal_workload() -> dict[str, Any]:
    return {
        "kind": "microbench",
        "benchmark": "unit_test",
        "task_ids": [],
        "seed": 1234,
        "n_repeats": 1,
    }
