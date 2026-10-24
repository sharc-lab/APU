"""harness/x2_demo_hold.py: waits until the release time with heartbeats, releases at once when started late, refuses
a naive time or a release more than 7 days ahead; its queue entry states the release time and sets no pause."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import x2_demo_hold as H  # noqa: E402
import x2_queue_fill as qf  # noqa: E402

T0 = datetime(2026, 10, 29, 20, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, t):
        self.t = t
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += timedelta(seconds=s)


def test_waits_until_release_with_heartbeats():
    c, rows = Clock(T0), []
    end = H.hold(T0 + timedelta(minutes=12), rows.append, clock=c, sleep=c.sleep)
    assert c.t == T0 + timedelta(minutes=12) and max(c.slept) <= H.HEARTBEAT_S
    assert rows[0]["record"] == "run_start" and rows[-1]["record"] == "run_end"
    assert end["started_after_release"] is False


def test_started_after_release_releases_at_once():
    c, rows = Clock(T0 + timedelta(hours=2)), []
    end = H.hold(T0, rows.append, clock=c, sleep=c.sleep)
    assert c.slept == [] and end["started_after_release"] is True


def test_guards():
    with pytest.raises(ValueError):
        H.parse_release("2026-10-29T20:00:00")
    with pytest.raises(ValueError):
        H.plan(T0 + timedelta(hours=200), T0)
    with pytest.raises(ValueError):
        H.plan(T0, T0 - timedelta(hours=1), start=T0 + timedelta(hours=1))
    assert H.plan(T0 + timedelta(hours=10), T0 + timedelta(hours=1), start=T0)["launched_after_window_start"]
    soon = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert H.main(["--release-utc", soon, "--dry-run"]) == 0


def test_queue_entry_states_the_release_time():
    # the operator's fallback: 08:00 to 18:00 PT on 2026-10-29 (PDT, UTC-7)
    e = qf.hold_entry("2026-10-29T18:00:00-07:00", "2026-10-29T08:00:00-07:00")
    assert "2026-10-30T01:00:00+00:00" in e["note"] and "2026-10-30T01:00:00+00:00" in e["cmd"]
    assert "2026-10-29T15:00:00+00:00" in e["note"] and "--start-utc" in e["cmd"]
    assert "tq.advance()" in e["note"] and "Sets no pause" in e["note"] and "--queued" in e["cmd"]
    assert qf.HOLD_ID not in qf.ENTRIES        # only defined when the operator's times are given
    with pytest.raises(ValueError):
        qf.hold_entry("2026-10-29T20:00:00", "2026-10-29T10:00:00+00:00")
    with pytest.raises(ValueError):
        qf.hold_entry("2026-10-29T10:00:00+00:00", "2026-10-29T20:00:00+00:00")
