from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import seam.locks as locks
from seam.errors import SeamError


def _lock_path(resource: Path) -> Path:
    return Path(str(resource) + ".lock")


def _patch_identity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    boot_time: str = "2026-08-04T01:20:59.500000+00:00",
    hostname: str = "platform-a",
    live_pids: set[int] | None = None,
) -> None:
    live = live_pids or set()
    monkeypatch.setattr(locks, "_current_boot_time", lambda: boot_time)
    monkeypatch.setattr(locks, "_current_hostname", lambda: hostname)
    monkeypatch.setattr(locks, "_pid_alive", lambda pid: pid in live)


def _write_record(
    resource: Path,
    *,
    pid: int,
    boot_time: str,
    hostname: str,
) -> None:
    _lock_path(resource).write_text(
        json.dumps({"pid": pid, "boot_time": boot_time, "hostname": hostname}) + "\n",
        encoding="utf-8",
    )


def test_old_format_record_refuses_without_boot_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = tmp_path / "machine"
    _lock_path(resource).write_text("pid=12345\n", encoding="utf-8")
    _patch_identity(monkeypatch)

    with pytest.raises(SeamError, match="no boot identity"):
        locks.ExclusiveLock(resource).acquire()


def test_cross_boot_record_is_automatically_reclaimed_and_logs_both_boots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = tmp_path / "machine"
    current_boot = "2026-08-04T01:20:59.500000+00:00"
    old_boot = "2026-08-03T01:20:59.500000+00:00"
    _write_record(resource, pid=12345, boot_time=old_boot, hostname="platform-a")
    _patch_identity(monkeypatch, boot_time=current_boot)
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        locks,
        "log_event",
        lambda event, **fields: events.append((event, fields)) or {"event": event, **fields},
    )

    lock = locks.ExclusiveLock(resource)
    lock.acquire()
    try:
        record = locks.read_lock_record(_lock_path(resource))
        assert record["pid"] == os.getpid()
        assert events[0][0] == "lock.cross_boot_reclaimed"
        assert events[0][1]["recorded_boot_time"] == old_boot
        assert events[0][1]["current_boot_time"] == current_boot
    finally:
        lock.release()


def test_same_boot_dead_pid_refuses_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = tmp_path / "machine"
    boot = "2026-08-04T01:20:59.500000+00:00"
    _write_record(resource, pid=12345, boot_time=boot, hostname="platform-a")
    _patch_identity(monkeypatch, boot_time=boot)

    with pytest.raises(SeamError, match="explicit recovery was not requested"):
        locks.ExclusiveLock(resource).acquire()


def test_explicit_same_boot_dead_pid_recovery_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = tmp_path / "machine"
    boot = "2026-08-04T01:20:59.500000+00:00"
    _write_record(resource, pid=12345, boot_time=boot, hostname="platform-a")
    _patch_identity(monkeypatch, boot_time=boot)
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        locks,
        "log_event",
        lambda event, **fields: events.append((event, fields)) or {"event": event, **fields},
    )

    lock = locks.ExclusiveLock(resource, recover_dead_owner=True)
    lock.acquire()
    try:
        assert events[0][0] == "lock.same_boot_dead_owner_reclaimed"
        assert events[0][1]["explicit_recovery"] is True
    finally:
        lock.release()


def test_live_owner_refuses_even_with_explicit_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = tmp_path / "machine"
    boot = "2026-08-04T01:20:59.500000+00:00"
    _write_record(resource, pid=12345, boot_time=boot, hostname="platform-a")
    _patch_identity(monkeypatch, boot_time=boot, live_pids={12345})

    with pytest.raises(SeamError, match="owner is live"):
        locks.ExclusiveLock(resource, recover_dead_owner=True).acquire()


def test_different_hostname_never_reclaims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resource = tmp_path / "machine"
    _write_record(
        resource,
        pid=12345,
        boot_time="2026-08-03T01:20:59.500000+00:00",
        hostname="other-host",
    )
    _patch_identity(monkeypatch, hostname="platform-a")

    with pytest.raises(SeamError, match="different hostname"):
        locks.ExclusiveLock(resource, recover_dead_owner=True).acquire()


def test_exclusivity_and_record_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resource = tmp_path / "machine"
    boot = "2026-08-04T01:20:59.500000+00:00"
    _patch_identity(monkeypatch, boot_time=boot, live_pids={os.getpid()})

    with locks.exclusive(resource):
        record = locks.read_lock_record(_lock_path(resource))
        assert record == {
            "format": "json_v2",
            "pid": os.getpid(),
            "boot_time": boot,
            "hostname": "platform-a",
        }
        with pytest.raises(SeamError, match="refusing second writer"):
            locks.ExclusiveLock(resource).acquire()
