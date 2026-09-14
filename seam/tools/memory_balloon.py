"""Resident memory balloon: allocate and touch pages in a child process.

Reserved-but-untouched memory does not create pressure. The child therefore writes every
page before reporting ``balloon_bytes_touched``.
"""

from __future__ import annotations

import contextlib
import gc
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

__all__ = ["BalloonResult", "ResidentBalloon", "touch_allocate"]

_PAGE = 4096


def touch_allocate(n_bytes: int) -> tuple[bytearray, int]:
    """Allocate ``n_bytes`` and touch every page; return (buffer, bytes_touched)."""
    if n_bytes < 0:
        raise ValueError("n_bytes must be non-negative")
    if n_bytes == 0:
        return bytearray(), 0
    buf = bytearray(n_bytes)
    touched = 0
    for offset in range(0, n_bytes, _PAGE):
        buf[offset] = 1
        touched += min(_PAGE, n_bytes - offset)
    # Touch the final byte so a partial trailing page is resident too.
    if n_bytes > 0:
        buf[n_bytes - 1] = 1
    return buf, touched


def _balloon_child(conn: Any) -> None:
    held: list[bytearray] = []
    try:
        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")
            if cmd == "exit":
                held.clear()
                gc.collect()
                conn.send({"ok": True})
                break
            if cmd == "resize":
                requested = int(msg["bytes"])
                held.clear()
                gc.collect()
                if requested <= 0:
                    conn.send(
                        {
                            "ok": True,
                            "balloon_bytes_requested": 0,
                            "balloon_bytes_touched": 0,
                        }
                    )
                    continue
                buf, touched = touch_allocate(requested)
                held.append(buf)
                conn.send(
                    {
                        "ok": True,
                        "balloon_bytes_requested": requested,
                        "balloon_bytes_touched": touched,
                    }
                )
            else:
                conn.send({"ok": False, "error": f"unknown cmd {cmd!r}"})
    finally:
        held.clear()
        gc.collect()


@dataclass(slots=True)
class BalloonResult:
    free_memory_mb_target: float
    free_memory_mb_achieved_before: float
    free_memory_mb_achieved_after: float
    balloon_bytes_requested: int
    balloon_bytes_touched: int


class ResidentBalloon:
    """Child-process resident balloon driven by free-memory targets."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        self._proc = self._ctx.Process(
            target=_balloon_child, args=(child_conn,), name="seam-resident-balloon", daemon=True
        )
        self._proc.start()
        child_conn.close()
        self._current_bytes = 0

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.is_alive():
                self._parent_conn.send({"cmd": "exit"})
                with contextlib.suppress(EOFError, OSError):
                    self._parent_conn.recv()
                self._proc.join(timeout=10)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=5)
        finally:
            self._parent_conn.close()
            self._proc = None
            self._current_bytes = 0

    def __enter__(self) -> ResidentBalloon:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def resize_bytes(self, n_bytes: int) -> dict[str, int]:
        if self._proc is None or not self._proc.is_alive():
            raise RuntimeError("balloon process is not running")
        requested = max(0, int(n_bytes))
        self._parent_conn.send({"cmd": "resize", "bytes": requested})
        reply = self._parent_conn.recv()
        if not reply.get("ok"):
            raise RuntimeError(f"balloon resize failed: {reply!r}")
        self._current_bytes = int(reply["balloon_bytes_requested"])
        return {
            "balloon_bytes_requested": int(reply["balloon_bytes_requested"]),
            "balloon_bytes_touched": int(reply["balloon_bytes_touched"]),
        }

    def set_free_memory_mb(self, target_free_mb: float) -> BalloonResult:
        """Resize so available memory approaches ``target_free_mb``; never refuse on miss."""
        import psutil

        before = float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
        # Current free includes nothing from a zero balloon; when we already hold pages,
        # releasing them would raise free. Work from the non-balloon free estimate:
        # non_balloon_free ≈ before + current_balloon_mb
        current_balloon_mb = self._current_bytes / (1024.0 * 1024.0)
        non_balloon_free = before + current_balloon_mb
        need_balloon_mb = max(0.0, non_balloon_free - float(target_free_mb))
        need_bytes = int(need_balloon_mb * 1024.0 * 1024.0)
        sized = self.resize_bytes(need_bytes)
        after = float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
        return BalloonResult(
            free_memory_mb_target=float(target_free_mb),
            free_memory_mb_achieved_before=before,
            free_memory_mb_achieved_after=after,
            balloon_bytes_requested=sized["balloon_bytes_requested"],
            balloon_bytes_touched=sized["balloon_bytes_touched"],
        )
