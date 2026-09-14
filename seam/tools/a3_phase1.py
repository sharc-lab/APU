"""DEPRECATED - A3 Phase 1 (8192 MB headroom re-baseline) is deleted.

Superseded by ``seam.tools.a3_residency`` (A3 v2 continuous residency sweep).
The Phase-1 fork and the 8,192 MB headroom refusal no longer exist; top-of-ladder
blocks of the residency sweep answer the same question without a launch gate.
"""

from __future__ import annotations

import argparse

from seam.errors import SeamError

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    raise SeamError(
        "A3 Phase 1 is deleted (no 8192 MB headroom gate). "
        "Use: python -m seam.tools.a3_residency --allow-dirty "
        "(detached bare PowerShell; see docs/CURSOR_PROMPT_A3_residency_v2.md)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
