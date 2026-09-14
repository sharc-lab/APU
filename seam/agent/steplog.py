"""``steps.ndjson`` writer and reader (spec §6.2).

One JSON object per agent step, one line each. Two properties are enforced here rather than left to
the caller:

**Strict JSON only.** ``json.dumps`` is called with ``allow_nan=False``, so a non-finite value
raises at write time instead of producing a bare ``Infinity`` or ``NaN`` token that a strict reader
would reject - which is exactly how an infinite deadline sentinel would have poisoned every
downstream consumer of this file.

**Append-only while the run is open.** The writer never rewrites a line. A crash mid-trajectory
leaves the steps already measured on disk and valid.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Any

from seam.agent.harness import StepRecord

__all__ = ["STEPS_FILENAME", "StepLogWriter", "read_steps", "step_to_record"]

STEPS_FILENAME = "steps.ndjson"


def step_to_record(step: StepRecord) -> dict[str, Any]:
    """Render one :class:`~seam.agent.harness.StepRecord` as a JSON-ready dict."""
    return asdict(step)


class StepLogWriter:
    """Append :class:`~seam.agent.harness.StepRecord` objects to an NDJSON file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._n = 0

    @property
    def n_written(self) -> int:
        return self._n

    def write(self, step: StepRecord) -> None:
        """Append one record, flushing so a later crash cannot lose it.

        Raises:
            ValueError: If any value is non-finite. Strict JSON has no representation for it, so the
                failure belongs at the write, not at the read.
        """
        line = json.dumps(step_to_record(step), sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._n += 1

    def __enter__(self) -> StepLogWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def read_steps(path: Path) -> list[dict[str, Any]]:
    """Read an NDJSON step log with a strict parser.

    Raises:
        ValueError: On a non-finite token. ``json.loads`` accepts ``Infinity`` by default, which
            would let a poisoned file load silently; ``parse_constant`` refuses it here.
        TypeError: If a line is valid JSON but not an object.
    """

    def _refuse(token: str) -> Any:
        raise ValueError(
            f"{path}: non-finite JSON token {token!r}. Strict JSON has no such value; the file was "
            f"written by something that did not enforce allow_nan=False."
        )

    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text, parse_constant=_refuse)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise TypeError(f"{path}:{lineno}: expected a JSON object, got {type(parsed)}")
            records.append(parsed)
    return records
