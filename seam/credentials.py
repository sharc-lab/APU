"""Credential loading. The key never leaves this module's return value.

Rules, enforced here and tested in ``tests/test_credentials.py``:

* The key is read from the process environment, optionally seeded from a **gitignored** ``.env``
  at repo root. It is never accepted from a command line, a config file, or a chat message.
* Nothing derived from the key - not the value, not a prefix, not a hash - is ever logged or
  written into a manifest. A hash is still a credential-linked identifier and would let a leaked
  artifact be matched against a candidate key.
* Presence checks report a boolean and a character count. Never a fragment.
* An absent key means paid runs **refuse to start** and emit a refusal manifest, rather than
  degrading to some cheaper path that quietly changes the experiment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from seam.errors import BackendError
from seam.jsonlog import log_event

__all__ = ["ANTHROPIC_API_KEY_ENV", "CredentialStatus", "load_dotenv_once", "require_api_key"]

ANTHROPIC_API_KEY_ENV: Final = "ANTHROPIC_API_KEY"

_dotenv_loaded = False


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    """Safe-to-log description of a credential.

    Deliberately carries no value-derived field beyond a length.
    """

    env_var: str
    present: bool
    char_count: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_var": self.env_var,
            "present": self.present,
            "char_count": self.char_count,
            "source": self.source,
        }


def load_dotenv_once(repo_root: Path | None = None) -> str:
    """Seed the environment from a gitignored ``.env`` at repo root, if present.

    ``override=False``: a variable already exported in the shell wins. Silently overriding an
    explicitly exported key with a stale file would be an unpleasant way to bill the wrong
    account.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return "already_loaded"
    _dotenv_loaded = True

    if repo_root is None:
        from seam.gitinfo import repo_root as find_root

        repo_root = find_root(Path(__file__).parent)

    env_path = repo_root / ".env"
    if not env_path.exists():
        return "no_dotenv_file"

    try:
        from dotenv import load_dotenv
    except ImportError:
        log_event(
            "credentials.dotenv_unavailable",
            severity="warning",
            message="python-dotenv is not installed; .env not loaded",
        )
        return "dotenv_unavailable"

    load_dotenv(env_path, override=False)
    log_event(
        "credentials.dotenv_loaded",
        message=f"seeded environment from {env_path.name} (values not recorded)",
        path=str(env_path),
    )
    return "dotenv"


def credential_status(env_var: str = ANTHROPIC_API_KEY_ENV) -> CredentialStatus:
    """Presence and length only. Never a fragment of the value."""
    source = load_dotenv_once()
    value = os.environ.get(env_var, "")
    return CredentialStatus(
        env_var=env_var,
        present=bool(value),
        char_count=len(value),
        source=source if value else "absent",
    )


def require_api_key(env_var: str = ANTHROPIC_API_KEY_ENV) -> str:
    """Return the key, or refuse.

    The returned value is passed straight to the SDK client and must not be stored, formatted
    into a message, or included in any structure that gets serialized.
    """
    status = credential_status(env_var)
    if not status.present:
        log_event(
            "credentials.absent",
            severity="error",
            message=f"{env_var} is not set; refusing to start a paid run",
            **status.to_dict(),
        )
        raise BackendError(
            f"{env_var} is not set. Put it in a gitignored .env at repo root, or export it at "
            f"user scope and restart the application so a new process inherits it. Never paste a "
            f"key into a config, a command line, or a chat message."
        )
    log_event(
        "credentials.present",
        message=f"{env_var} present ({status.char_count} chars, source {status.source})",
        **status.to_dict(),
    )
    return os.environ[env_var]
