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


# ── rolling hold (Tue 2026-10-27 showing) ──
START = datetime(2026, 10, 27, 17, 0, tzinfo=timezone.utc)      # showing time
HOLD_START = START - timedelta(minutes=30)                      # warm-up: machine free 30 min before
END = datetime(2026, 10, 27, 19, 0, tzinfo=timezone.utc)


def _job(jid, proj, high=None):
    note = f"some job. ESTIMATE {proj} h" + (f" (high {high} h; register row x)" if high is not None else "")
    return {"id": jid, "status": "pending", "projected_duration_h": proj, "note": note, "cmd": ["py", jid]}


def test_rolling_entry_states_window_warmup_and_release():
    e = qf.hold_entry("2026-10-27T12:00:00-07:00", "2026-10-27T10:00:00-07:00", 30, True)
    assert "--rolling" in e["cmd"] and e["cmd"][e["cmd"].index("--warmup-min") + 1] == "30"
    assert "machine free from 2026-10-27T16:30:00+00:00" in e["note"] and "never interrupted" in e["note"]
    assert "releases itself via tq.advance() at 2026-10-27T19:00:00+00:00" in e["note"]
    assert H.high_estimate_h(e) is None          # the hold's own note states no duration estimate


def test_high_estimate_from_note_or_factor():
    assert H.high_estimate_h(_job("a", 2.0, 3.5)) == 3.5
    assert H.high_estimate_h(_job("b", 2.0)) == 3.0
    assert H.high_estimate_h({"id": "c", "note": ""}) is None


def test_decide_defers_only_when_the_next_job_fits_by_its_high_estimate():
    now = HOLD_START - timedelta(hours=4)
    assert H.decide(now, HOLD_START, END, _job("j", 1.0, 3.9))["action"] == "defer"
    d = H.decide(now, HOLD_START, END, _job("j", 1.0, 4.1))     # projected fits, high does not: hold
    assert d["action"] == "hold" and "high estimate 4.10 h" in d["why"]
    assert H.decide(now, HOLD_START, END, None)["action"] == "hold"
    assert H.decide(now, HOLD_START, END, {"id": "x", "note": "", "projected_duration_h": None})["action"] == "hold"
    assert H.decide(END, HOLD_START, END, _job("j", 0.1, 0.2))["action"] == "release"
    assert H.decide(START, HOLD_START, END, _job("j", 0.1, 0.2))["action"] == "hold"   # inside the window


def test_deferred_items_reinserts_a_pending_copy_after_the_next_job():
    hold = {"id": "x2_demo_hold_tue", "status": "running", "cmd": ["py", "demo\\x2_demo_hold.py"], "note": "n",
            "projected_duration_h": None}
    q = [{"id": "done", "status": "done"}, hold, _job("j1", 1.0, 1.5), _job("j2", 5.0, 8.0)]
    out = H.deferred_items(q, "x2_demo_hold_tue", "j1")
    assert [i["id"] for i in out] == ["done", "x2_demo_hold_tue", "j1", "x2_demo_hold_tue_r1", "j2"]
    assert out[1]["status"] == "running" and out[3]["status"] == "pending" and out[3]["cmd"] == hold["cmd"]
    out2 = H.deferred_items([dict(out[3], status="running")] + out[4:], "x2_demo_hold_tue_r1", "j2")
    assert [i["id"] for i in out2] == ["x2_demo_hold_tue_r1", "j2", "x2_demo_hold_tue_r2"]


class FakeQueue:
    def __init__(self, items):
        self.items, self.paused, self.pause_reasons, self.advanced = items, False, [], []

    def read_queue(self):
        return [dict(i) for i in self.items]

    def write_queue(self, items):
        assert self.paused
        self.items = items

    def set_pause(self, reason):
        self.paused = True
        self.pause_reasons.append(reason)

    def clear_pause(self):
        self.paused = False

    def advance(self, note):
        self.advanced.append(note)


def test_rolling_step_with_a_fake_clock_walks_the_queue_until_the_window():
    hold = {"id": "x2_demo_hold_tue", "status": "running", "cmd": ["py", "demo\\x2_demo_hold.py", "--rolling"],
            "note": "hold", "projected_duration_h": None}
    q = FakeQueue([hold, _job("short", 0.5, 1.0), _job("long", 20.0, 30.0)])
    rows = []
    t = HOLD_START - timedelta(hours=3)
    d = H.rolling_step(q, HOLD_START, END, rows.append, clock=lambda: t)
    assert d["action"] == "defer" and d["next"] == "short" and not q.paused
    assert "released by clear_pause()" in q.pause_reasons[0]
    assert [i["id"] for i in q.items] == ["x2_demo_hold_tue", "short", "x2_demo_hold_tue_r1", "long"]
    # the queue advances: the hold is done, short runs and ends; the copy launches and must hold before long
    q.items = [dict(q.items[2], status="running"), q.items[3]]
    d2 = H.rolling_step(q, HOLD_START, END, rows.append, clock=lambda: t + timedelta(hours=1))
    assert d2["action"] == "hold" and d2["next"] == "long" and len(q.pause_reasons) == 1
    assert [r["action"] for r in rows] == ["defer", "hold"]


def test_simulate_projects_with_high_estimates_and_warns_on_a_running_overrun():
    now = HOLD_START - timedelta(hours=10)
    run = dict(_job("running", 4.0, 6.0), status="running", started_ts=(now - timedelta(hours=1)).timestamp())
    q = [run, _job("a", 1.0, 2.0), _job("b", 0.5, 1.0), _job("c", 3.0, 4.0)]
    s = H.simulate(q, now, HOLD_START, END)
    # running ends by high at now+5 h; a ends +2 h (now+7), b +1 h (now+8); c (+4 h) would end now+12 > window: hold
    assert [j["id"] for j in s["jobs_run_before_window"]] == ["a", "b"] and s["warning"] is None
    assert s["hold_engages_by_utc"] == (now + timedelta(hours=8)).isoformat()
    late = dict(run, started_ts=(HOLD_START - timedelta(hours=2)).timestamp())
    assert "could still be running" in H.simulate([late], now, HOLD_START, END)["warning"]


def test_main_rolling_hold_then_release_with_one_advance(monkeypatch, tmp_path):
    hold = {"id": "x2_demo_hold_tue", "status": "running", "cmd": ["py", "demo\\x2_demo_hold.py", "--rolling"],
            "note": "hold", "projected_duration_h": None}
    q = FakeQueue([hold, _job("long", 20.0, 30.0)])
    monkeypatch.setattr(H, "_tq", lambda: q)
    slept = []
    monkeypatch.setattr(H.time, "sleep", lambda s: slept.append(s))
    soon = datetime.now(timezone.utc) + timedelta(seconds=1)
    H.main(["--start-utc", (soon - timedelta(minutes=1)).isoformat(), "--release-utc", soon.isoformat(),
            "--warmup-min", "30", "--rolling", "--queued", "--out", str(tmp_path / "h.jsonl")])
    assert len(q.advanced) == 1 and q.advanced[0].startswith("completed: demo hold released at")
    assert "STOP" not in q.advanced[0]           # a STOP note would halt the queue
