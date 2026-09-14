"""Mutual exclusion for shared-path writers (blueprint §6.6).

Two concurrent writers have already produced a corrupt 2.26 GB IR that still loaded. A second
writer to ``AUDIT_LOG.md`` is the worst merge conflict this repo admits. This module makes a
second writer a hard refusal rather than a race.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from seam.errors import SeamError
from seam.jsonlog import log_event

__all__ = ["ExclusiveLock", "exclusive", "read_lock_record"]


def _current_boot_time() -> str:
    import psutil

    # psutil's Windows estimate can jitter by fractions of a second across calls. Canonicalize to
    # whole seconds so two processes on the same boot cannot mistake estimator jitter for reboot.
    return datetime.fromtimestamp(int(psutil.boot_time()), UTC).isoformat()


def _current_hostname() -> str:
    return socket.gethostname()


def _pid_alive(pid: int) -> bool:
    import psutil

    return bool(psutil.pid_exists(pid))


def read_lock_record(lock_path: Path) -> dict[str, Any]:
    """Read a current JSON lock record or identify a legacy ``pid=`` record."""
    text = lock_path.read_text(encoding="utf-8").strip()
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        if text.startswith("pid="):
            try:
                return {
                    "format": "legacy_pid_only",
                    "pid": int(text.removeprefix("pid=")),
                    "boot_time": None,
                    "hostname": None,
                }
            except ValueError:
                pass
        return {
            "format": "unparseable",
            "pid": None,
            "boot_time": None,
            "hostname": None,
        }
    if not isinstance(record, dict):
        return {
            "format": "unparseable",
            "pid": None,
            "boot_time": None,
            "hostname": None,
        }
    return {
        "format": "json_v2",
        "pid": record.get("pid"),
        "boot_time": record.get("boot_time"),
        "hostname": record.get("hostname"),
    }


class ExclusiveLock:
    """Exclusive lock held for the lifetime of a shared-path write.

    Uses ``O_CREAT | O_EXCL`` so acquisition is atomic on every platform this project runs on.
    Cross-boot records are definitionally stale and reclaimed automatically. Same-boot dead owners
    require explicit recovery; PID liveness alone is never enough for the default path.
    """

    __slots__ = ("_fd", "_recover_dead_owner", "lock_path", "resource")

    def __init__(self, resource: Path, *, recover_dead_owner: bool = False) -> None:
        self.resource = resource
        self.lock_path = resource if resource.suffix == ".lock" else Path(str(resource) + ".lock")
        self._fd: int | None = None
        self._recover_dead_owner = recover_dead_owner

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        current_boot = _current_boot_time()
        current_hostname = _current_hostname()
        try:
            self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            owner = read_lock_record(self.lock_path)
            recorded_hostname = owner.get("hostname")
            recorded_boot = owner.get("boot_time")
            owner_pid = owner.get("pid")
            same_host = recorded_hostname == current_hostname
            cross_boot = (
                same_host and isinstance(recorded_boot, str) and recorded_boot != current_boot
            )
            explicit_dead_recovery = (
                same_host
                and recorded_boot == current_boot
                and isinstance(owner_pid, int)
                and not _pid_alive(owner_pid)
                and self._recover_dead_owner
            )
            if cross_boot:
                log_event(
                    "lock.cross_boot_reclaimed",
                    severity="warning",
                    message=f"reclaiming cross-boot stale lock at {self.lock_path}",
                    lock_path=str(self.lock_path),
                    recorded_pid=owner_pid,
                    recorded_boot_time=recorded_boot,
                    current_boot_time=current_boot,
                    recorded_hostname=recorded_hostname,
                    current_hostname=current_hostname,
                )
            elif explicit_dead_recovery:
                log_event(
                    "lock.same_boot_dead_owner_reclaimed",
                    severity="warning",
                    message=f"explicitly reclaiming same-boot dead-owner lock at {self.lock_path}",
                    lock_path=str(self.lock_path),
                    recorded_pid=owner_pid,
                    recorded_boot_time=recorded_boot,
                    current_boot_time=current_boot,
                    recorded_hostname=recorded_hostname,
                    current_hostname=current_hostname,
                    explicit_recovery=True,
                )
            else:
                reason = "owner record is not safely reclaimable"
                if recorded_hostname not in (None, current_hostname):
                    reason = "lock belongs to a different hostname"
                elif recorded_boot is None:
                    reason = "legacy/unparseable record has no boot identity"
                elif recorded_boot == current_boot and isinstance(owner_pid, int):
                    reason = (
                        "same-boot owner is live"
                        if _pid_alive(owner_pid)
                        else "same-boot owner PID is dead but explicit recovery was not requested"
                    )
                raise SeamError(
                    f"refusing second writer on {self.resource}: lock held at {self.lock_path}; "
                    f"{reason}. Blueprint §6.6 mutual exclusion forbids silent lock theft."
                ) from exc
            try:
                self.lock_path.unlink()
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except (FileNotFoundError, FileExistsError, OSError) as reclaim_exc:
                raise SeamError(
                    f"stale-lock recovery raced at {self.lock_path}; refusing acquisition"
                ) from reclaim_exc
        record = {
            "pid": os.getpid(),
            "boot_time": current_boot,
            "hostname": current_hostname,
        }
        os.write(self._fd, (json.dumps(record, sort_keys=True) + "\n").encode())

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self.lock_path.exists():
            self.lock_path.unlink()

    def __enter__(self) -> ExclusiveLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


@contextmanager
def exclusive(resource: Path) -> Iterator[ExclusiveLock]:
    """Context-manager form of :class:`ExclusiveLock`."""
    lock = ExclusiveLock(resource)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()
