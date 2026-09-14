"""Force UTF-8 on stdout/stderr so non-ASCII log/print paths cannot raise under cp1252.

C2d / C9: two completed efilter runs were lost to write-path defects discovered only after
measurement finished (circular JSON; UnicodeEncodeError on ``→``). Call
:func:`configure_utf8_stdio` at process start, before logging is configured.
"""

from __future__ import annotations

import sys
from typing import Any, Final

__all__ = ["NON_ASCII_PROBE", "configure_utf8_stdio"]

#: Synthetic marker used by startup dry-runs and encoding stress tests.
NON_ASCII_PROBE: Final = "C2d encoding probe: arrow=\u2192 alpha=\u03b1"


def configure_utf8_stdio() -> dict[str, Any]:
    """Reconfigure ``sys.stdout`` / ``sys.stderr`` to UTF-8 with ``errors='replace'``.

    Returns a small record of what was applied (for dry-run / tests). Safe to call more than once.
    Streams without ``reconfigure`` (e.g. some redirected ``TextIO`` doubles) are left unchanged
    when the attribute is missing; callers that need a hard guarantee should replace the stream.
    """
    applied: dict[str, Any] = {
        "stdout": False,
        "stderr": False,
        "encoding": "utf-8",
        "errors": "replace",
    }
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
            applied[name] = True
    return applied
