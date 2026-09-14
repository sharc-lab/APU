"""Observe an in-flight model fetch by sampling bytes on disk.

``fetch_model`` runs ``curl -sL``, which is silent by construction, and prints only once the
whole transfer has finished. During a multi-hour download over a degraded link that leaves no
way to distinguish "slow" from "stalled" from "dead" - the failure mode this exists to remove.

This watcher never touches the download. It samples file sizes and appends one JSON line per
sample, so the log is both human-tailable and machine-readable after the fact.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from seam.gitinfo import repo_root

__all__ = ["Sample", "sample_once", "watch"]

#: Rate is averaged over a trailing window rather than since start, so a stall shows up promptly
#: instead of being masked by earlier healthy throughput.
_RATE_WINDOW = 10


class Sample:
    """One observation of a file's size, with a windowed rate estimate."""

    __slots__ = ("bytes_now", "elapsed_s", "expected_bytes", "path", "rate_bps", "utc")

    def __init__(
        self,
        *,
        path: str,
        bytes_now: int,
        expected_bytes: int | None,
        rate_bps: float | None,
        elapsed_s: float,
        utc: str,
    ) -> None:
        self.path = path
        self.bytes_now = bytes_now
        self.expected_bytes = expected_bytes
        self.rate_bps = rate_bps
        self.elapsed_s = elapsed_s
        self.utc = utc

    @property
    def pct(self) -> float | None:
        if not self.expected_bytes:
            return None
        return 100.0 * self.bytes_now / self.expected_bytes

    @property
    def eta_s(self) -> float | None:
        """Seconds remaining at the windowed rate, or ``None`` if stalled or size unknown."""
        if not self.expected_bytes or not self.rate_bps or self.rate_bps <= 0:
            return None
        return max(0.0, (self.expected_bytes - self.bytes_now) / self.rate_bps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "utc": self.utc,
            "path": self.path,
            "bytes": self.bytes_now,
            "expected_bytes": self.expected_bytes,
            "pct": None if self.pct is None else round(self.pct, 3),
            "rate_bytes_per_s": None if self.rate_bps is None else round(self.rate_bps, 1),
            "rate_mb_per_min": (
                None if self.rate_bps is None else round(self.rate_bps * 60 / 1e6, 2)
            ),
            "eta_s": None if self.eta_s is None else round(self.eta_s, 1),
            "elapsed_s": round(self.elapsed_s, 1),
            "stalled": self.rate_bps is not None and self.rate_bps <= 0,
        }

    def render(self) -> str:
        pct = "  ? " if self.pct is None else f"{self.pct:5.1f}%"
        rate = "    ?" if self.rate_bps is None else f"{self.rate_bps * 60 / 1e6:6.2f} MB/min"
        eta = "?" if self.eta_s is None else f"{self.eta_s / 60:.0f} min"
        flag = "  STALLED" if (self.rate_bps is not None and self.rate_bps <= 0) else ""
        return f"{self.utc}  {self.path}  {self.bytes_now:>13,} B  {pct}  {rate}  ETA {eta}{flag}"


def sample_once(
    path: Path, *, expected_bytes: int | None, history: deque[tuple[float, int]], started: float
) -> Sample:
    now = time.time()
    size = path.stat().st_size if path.exists() else 0
    history.append((now, size))

    rate: float | None = None
    if len(history) >= 2:
        t0, b0 = history[0]
        dt = now - t0
        if dt > 0:
            rate = (size - b0) / dt

    return Sample(
        path=path.name,
        bytes_now=size,
        expected_bytes=expected_bytes,
        rate_bps=rate,
        elapsed_s=now - started,
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
    )


def watch(
    targets: dict[Path, int | None],
    *,
    log_path: Path,
    interval_s: float = 30.0,
    max_samples: int | None = None,
) -> int:
    """Sample ``targets`` until each reaches its expected size. Returns samples taken."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    histories: dict[Path, deque[tuple[float, int]]] = {
        p: deque(maxlen=_RATE_WINDOW) for p in targets
    }

    taken = 0
    with log_path.open("a", encoding="utf-8") as log:
        while max_samples is None or taken < max_samples:
            done = True
            for path, expected in targets.items():
                s = sample_once(
                    path, expected_bytes=expected, history=histories[path], started=started
                )
                log.write(json.dumps(s.to_dict()) + "\n")
                log.flush()
                print(s.render(), flush=True)
                if expected is None or s.bytes_now < expected:
                    done = False
            taken += 1
            if done:
                print("all targets reached expected size", flush=True)
                break
            time.sleep(interval_s)
    return taken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        action="append",
        required=True,
        metavar="PATH[:BYTES]",
        help="file to watch, optionally with its expected final size",
    )
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)

    targets: dict[Path, int | None] = {}
    for spec in args.file:
        # rsplit, because a Windows path contains a drive-letter colon.
        head, sep, tail = spec.rpartition(":")
        if sep and tail.isdigit():
            targets[Path(head)] = int(tail)
        else:
            targets[Path(spec)] = None

    log_path = args.log or (repo_root(Path(__file__).parent) / "derived" / "mslice" / "watch.jsonl")
    watch(targets, log_path=log_path, interval_s=args.interval_s, max_samples=args.max_samples)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
