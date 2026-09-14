"""Structured JSON-lines logging alongside human-readable console output (spec §8).

Two sinks, one call site:

* **stderr** - human-readable, for the operator watching a run.
* **JSON lines** - machine-readable, one object per line, for the audit trail.

The JSON sink is what makes spec §9.6 enforceable. "No silent fallbacks" is only meaningful if
there is somewhere for a fallback to be recorded, so :func:`log_event` is the mechanism by which
a fallback, retry, or waiver becomes an auditable event instead of a shrug.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

__all__ = [
    "EventSeverity",
    "add_json_sink",
    "get_logger",
    "log_event",
    "remove_json_sink",
    "utc_now_iso",
]

EventSeverity = Literal["debug", "info", "warning", "error", "critical"]

_LOGGER_NAME: Final = "seam"
_lock: Final = threading.Lock()
_json_sinks: Final[list[Path]] = []
_console_configured = False

_SEVERITY_TO_LEVEL: Final[dict[str, int]] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with explicit offset.

    Used for cross-signal correlation only. Never for measuring durations - spec §3.5 requires
    :func:`time.perf_counter_ns` for those, because wall-clock time is not monotonic.
    """
    return datetime.now(UTC).isoformat()


def get_logger() -> logging.Logger:
    """Return the SEAM console logger, configuring it once."""
    global _console_configured
    logger = logging.getLogger(_LOGGER_NAME)
    with _lock:
        if not _console_configured:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
            _console_configured = True
    return logger


def add_json_sink(path: Path) -> None:
    """Register a JSON-lines sink. Events are appended to every registered sink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        if path not in _json_sinks:
            _json_sinks.append(path)


def remove_json_sink(path: Path) -> None:
    """Deregister a JSON-lines sink.

    Required before a run directory is sealed: ``raw/`` is write-once, so an event emitted after
    the seal would both invalidate the recorded tree hash and hit a read-only file.
    """
    with _lock:
        if path in _json_sinks:
            _json_sinks.remove(path)


def _clear_json_sinks() -> None:
    """Drop all registered sinks. For test isolation only."""
    with _lock:
        _json_sinks.clear()


def log_event(
    event: str,
    *,
    severity: EventSeverity = "info",
    message: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Emit one structured event to stderr and to every registered JSON-lines sink.

    Args:
        event: Stable machine-readable event name, e.g. ``"topology.verified"``. Grep-able, so it
            should not contain free text.
        severity: Console log level.
        message: Optional human-readable sentence for the console.
        **fields: Arbitrary JSON-serialisable context.

    Returns:
        The event record, so callers can attach it to a manifest or assert on it in a test.

    Raises:
        TypeError: If a field is not JSON-serialisable. Deliberately not coerced with
            ``default=str``: a field that cannot be serialised is a bug at the call site, and
            silently stringifying it would hide the bug in the audit trail.
    """
    record: dict[str, Any] = {"ts": utc_now_iso(), "event": event, "severity": severity}
    if message is not None:
        record["message"] = message
    record.update(fields)

    line = json.dumps(record, sort_keys=True)

    logger = get_logger()
    logger.log(_SEVERITY_TO_LEVEL[severity], "%s%s", event, f" - {message}" if message else "")

    with _lock:
        sinks = list(_json_sinks)
    for sink in sinks:
        with sink.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return record
