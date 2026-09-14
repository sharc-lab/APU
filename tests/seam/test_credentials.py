"""Credential handling and leak detection.

Cheap insurance against the expensive mistake: a key captured in a log, a trace, or a manifest and
then committed. The scan runs on every suite invocation so a leak is caught at the commit that
introduces it rather than after the repo is public.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from seam.credentials import (
    ANTHROPIC_API_KEY_ENV,
    CredentialStatus,
    credential_status,
    load_dotenv_once,
)
from seam.errors import BackendError

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories that end up committed or shared. A credential must never appear in any of them.
_SCANNED_DIRS = ("raw", "derived", "figures", "configs")

#: Anthropic key prefix. Matching on the prefix catches a leak even when the environment does not
#: currently hold the key that leaked.
_KEY_PREFIX = "sk-" + "ant"

_SKIP_SUFFIXES = {".bin", ".xml", ".safetensors", ".onnx", ".png", ".pdf", ".zip"}


def _scannable_files() -> list[Path]:
    files: list[Path] = []
    for name in _SCANNED_DIRS:
        directory = REPO_ROOT / name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() not in _SKIP_SUFFIXES:
                files.append(path)
    return files


def test_dotenv_is_effectively_gitignored() -> None:
    """If .env is committable, a live key can be committed. That must fail loudly."""
    result = subprocess.run(
        ["git", "check-ignore", "-v", ".env"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        ".env is NOT gitignored. A key placed there would be committable. "
        "Add it to .gitignore before storing any credential."
    )
    assert ".gitignore" in result.stdout


def test_no_key_prefix_appears_in_shared_artifacts() -> None:
    offenders: list[str] = []
    for path in _scannable_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _KEY_PREFIX in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"possible API key found in shared artifacts: {offenders}"


def test_live_key_value_never_appears_in_shared_artifacts() -> None:
    """Catches a leak of the key actually in use, not just one matching the known prefix.

    ``.env`` is loaded first. Reading only ``os.environ`` made this skip on exactly the machine
    that holds the key - the one machine where the scan is worth running.
    """
    load_dotenv_once()
    key = os.environ.get(ANTHROPIC_API_KEY_ENV, "")
    if not key:
        pytest.skip(f"{ANTHROPIC_API_KEY_ENV} not set; nothing to scan for")
    offenders: list[str] = []
    for path in _scannable_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if key in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"LIVE API KEY leaked into: {offenders}"


def test_credential_status_exposes_no_value_fragment() -> None:
    """Presence and length only: a prefix or hash is still a credential-linked identifier."""
    load_dotenv_once()
    status = credential_status(ANTHROPIC_API_KEY_ENV)
    assert isinstance(status, CredentialStatus)
    payload = status.to_dict()
    assert set(payload) == {"env_var", "present", "char_count", "source"}

    key = os.environ.get(ANTHROPIC_API_KEY_ENV, "")
    if key:
        serialized = str(payload)
        assert key not in serialized
        assert key[:8] not in serialized


def test_absent_key_refuses_rather_than_degrading(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing key must stop a paid run, not silently reroute it."""
    monkeypatch.delenv(ANTHROPIC_API_KEY_ENV, raising=False)
    monkeypatch.setattr("seam.credentials._dotenv_loaded", True)
    from seam.credentials import require_api_key

    with pytest.raises(BackendError, match="is not set"):
        require_api_key(ANTHROPIC_API_KEY_ENV)
