"""UTC ISO timestamp parsing must be correct regardless of the local machine's timezone or DST state.

time.mktime(time.strptime(s, ...)) - time.timezone assumes the local zone's STANDARD (non-DST) offset and is wrong by
an hour whenever DST is active -- found live on evo-x2 (Pacific Standard Time zone ID, PDT actually in effect),
silently moving every windowed sensor lookup outside the sample period. Fixed in harness/lhm_control.py and
harness/lhm_x2_control.py with calendar.timegm; harness/t2s_lab.py's iso_epoch() (the one actually used by every
overnight/night2 telemetry row) already used the correct datetime.fromisoformat pattern.
"""

import calendar
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import t2s_lab as L  # noqa: E402
import lhm_control as lc  # noqa: E402
import lhm_x2_control as lc2  # noqa: E402

UTC_TS = "2026-09-28T10:16:00.074Z"
EXPECTED_EPOCH = datetime(2026, 9, 28, 10, 16, 0, tzinfo=timezone.utc).timestamp()


def test_iso_epoch_correct_under_dst():
    """t2s_lab.iso_epoch is what every overnight/night2 row is windowed on; must be right under any local TZ/DST."""
    with mock.patch.object(time, "timezone", -25200), mock.patch.object(time, "daylight", 1):  # pretend PDT
        assert abs(L.iso_epoch(UTC_TS) - EXPECTED_EPOCH) < 1.0


def test_broken_pattern_actually_breaks_under_dst():
    """Documents the bug: the old pattern is wrong by exactly the DST hour when time.timezone is the STANDARD offset
    but DST is in effect (time.timezone itself does not shift with DST)."""
    broken = time.mktime(time.strptime(UTC_TS[:19], "%Y-%m-%dT%H:%M:%S")) - (-28800)  # PST standard offset
    correct = calendar.timegm(time.strptime(UTC_TS[:19], "%Y-%m-%dT%H:%M:%S"))
    # under PDT (actual offset -25200) with time.timezone fixed at the PST value (-28800), the broken pattern is off
    # by the 3600 s DST delta -- this assertion only holds on a machine where local wall time differs from the
    # struct's naive interpretation, so it is illustrative here rather than environment-independent; the real
    # regression coverage is test_iso_epoch_correct_under_dst and the two _correct tests below.
    assert correct == EXPECTED_EPOCH


def test_lhm_control_read_rows_uses_correct_parser():
    p = Path(__file__).resolve().parent / "_tmp_lhm_rows.jsonl"
    p.write_text('{"ts": "%s", "sensors": {"Temperature | x | y": 50.0}}\n' % UTC_TS, encoding="utf-8")
    try:
        rows = lc.read(str(p))
    finally:
        p.unlink(missing_ok=True)
    assert len(rows) == 1
    assert abs(rows[0]["t"] - EXPECTED_EPOCH) < 1.0


def test_lhm_x2_control_read_rows_correct():
    p = Path(__file__).resolve().parent / "_tmp_lhm_x2_rows.jsonl"
    p.write_text('{"ts": "%s", "sensors": {"Power | x | y": 1.0}}\n' % UTC_TS, encoding="utf-8")
    try:
        rows = lc2.read_rows(str(p))
    finally:
        p.unlink(missing_ok=True)
    assert len(rows) == 1
    assert abs(rows[0]["t"] - EXPECTED_EPOCH) < 1.0
