"""Pre-commit hook: block API key material from entering the repository.

Agent harnesses log request payloads by default, and authorization headers are
trivially easy to capture without noticing. This hook is cheap insurance against
a live credential reaching git history, where removing it requires a rewrite.

Checks staged files for:
  1. Known provider key prefixes (Anthropic, OpenAI, Google, HuggingFace, AWS).
  2. The literal value of any *_API_KEY / *_TOKEN found in the environment or in
     a repo-root .env, so a key is caught even if its prefix is unrecognised.
  3. Accidental staging of .env itself.

Exit 0 = clean, 1 = blocked.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Provider key shapes. Deliberately conservative: a false positive costs a
# minute, a false negative costs a credential rotation and a history rewrite.
PREFIX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Anthropic", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    ("OpenAI", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{32,}")),
    ("Google", re.compile(r"AIza[A-Za-z0-9_\-]{30,}")),
    ("HuggingFace", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
]

# Files that legitimately discuss key handling and must not trip the scanner on
# documentation text alone. Literal secret values are still checked everywhere.
DOC_SUFFIXES = {".md", ".mdc"}

MIN_SECRET_LEN = 20


def literal_secrets() -> list[str]:
    """Collect concrete secret values to search for, from env and .env."""
    values: set[str] = set()

    for name, value in os.environ.items():
        if not value or len(value) < MIN_SECRET_LEN:
            continue
        if name.endswith(("_API_KEY", "_TOKEN", "_SECRET")):
            values.add(value)

    dotenv = REPO_ROOT / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            _, _, value = line.partition("=")
            value = value.strip().strip("'\"")
            if len(value) >= MIN_SECRET_LEN:
                values.add(value)

    return sorted(values)


def scan(paths: list[str]) -> int:
    secrets = literal_secrets()
    violations: list[str] = []

    for raw_path in paths:
        path = Path(raw_path)

        if path.name == ".env" or path.suffix == ".env":
            violations.append(f"{path}: .env must never be staged")
            continue

        if not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Literal values are fatal anywhere, documentation included.
        for secret in secrets:
            if secret in text:
                violations.append(f"{path}: contains a live secret value from env/.env")
                break

        if path.suffix in DOC_SUFFIXES:
            continue

        for provider, pattern in PREFIX_PATTERNS:
            match = pattern.search(text)
            if match:
                violations.append(
                    f"{path}: looks like a {provider} key "
                    f"({match.group()[:10]}... , {len(match.group())} chars)"
                )

    if violations:
        sys.stderr.write("\nBLOCKED — possible key material in staged files:\n")
        for violation in violations:
            sys.stderr.write(f"  {violation}\n")
        sys.stderr.write(
            "\nRemove the material and re-stage. If a key was already committed,\n"
            "rotate it in the provider console before doing anything else.\n\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(scan(sys.argv[1:]))
