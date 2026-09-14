"""Observability of an in-flight fetch.

The property that matters is that a stalled transfer is *visibly* stalled. A watcher that
silently reports the last good rate forever would be worse than no watcher, because it would
manufacture confidence that the download is progressing.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from seam.tools.fetch_progress import sample_once, watch


def _history() -> deque[tuple[float, int]]:
    return deque(maxlen=10)


class TestSampleOnce:
    def test_reports_size_and_percentage(self, tmp_path: Path) -> None:
        target = tmp_path / "m.bin"
        target.write_bytes(b"x" * 500)
        s = sample_once(target, expected_bytes=1000, history=_history(), started=0.0)
        assert s.bytes_now == 500
        assert s.pct == 50.0

    def test_absent_file_reads_as_zero_not_an_error(self, tmp_path: Path) -> None:
        """A fetch that has not created the file yet is at zero, not broken."""
        s = sample_once(
            tmp_path / "absent.bin", expected_bytes=1000, history=_history(), started=0.0
        )
        assert s.bytes_now == 0
        assert s.pct == 0.0

    def test_first_sample_has_no_rate(self, tmp_path: Path) -> None:
        target = tmp_path / "m.bin"
        target.write_bytes(b"x" * 10)
        s = sample_once(target, expected_bytes=100, history=_history(), started=0.0)
        assert s.rate_bps is None
        assert s.eta_s is None, "an ETA from a single sample would be invented"

    def test_rate_and_eta_are_computed_across_samples(self, tmp_path: Path) -> None:
        target = tmp_path / "m.bin"
        history = _history()
        # Two samples with a synthetic gap: the deque holds (time, bytes) pairs directly.
        history.append((1000.0, 0))
        target.write_bytes(b"x" * 100)
        s = sample_once(target, expected_bytes=200, history=history, started=1000.0)
        assert s.rate_bps is not None
        assert s.rate_bps > 0
        assert s.eta_s is not None

    def test_stalled_transfer_is_flagged(self, tmp_path: Path) -> None:
        """No growth between samples must surface as stalled, not as a stale rate."""
        target = tmp_path / "m.bin"
        target.write_bytes(b"x" * 100)
        history = _history()
        history.append((1000.0, 100))
        s = sample_once(target, expected_bytes=200, history=history, started=1000.0)
        assert s.rate_bps == 0.0
        assert s.to_dict()["stalled"] is True
        assert s.eta_s is None, "a stalled transfer has no meaningful ETA"

    def test_unknown_expected_size_yields_no_pct_or_eta(self, tmp_path: Path) -> None:
        target = tmp_path / "m.bin"
        target.write_bytes(b"x" * 100)
        s = sample_once(target, expected_bytes=None, history=_history(), started=0.0)
        assert s.pct is None
        assert s.eta_s is None


class TestWatch:
    def test_stops_once_target_reaches_expected_size(self, tmp_path: Path) -> None:
        target = tmp_path / "m.bin"
        target.write_bytes(b"x" * 64)
        log = tmp_path / "progress.jsonl"

        taken = watch({target: 64}, log_path=log, interval_s=0.0, max_samples=10)

        assert taken == 1, "a complete file needs exactly one confirming sample"

    def test_writes_one_parseable_json_line_per_sample(self, tmp_path: Path) -> None:
        target = tmp_path / "m.bin"
        target.write_bytes(b"x" * 10)
        log = tmp_path / "progress.jsonl"

        watch({target: 1000}, log_path=log, interval_s=0.0, max_samples=3)

        lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 3
        assert all(entry["bytes"] == 10 for entry in lines)
        assert all("utc" in entry and "eta_s" in entry for entry in lines)

    def test_log_is_appended_not_truncated(self, tmp_path: Path) -> None:
        """Restarting the watcher must not destroy the history of a long download."""
        target = tmp_path / "m.bin"
        target.write_bytes(b"x" * 10)
        log = tmp_path / "progress.jsonl"

        watch({target: 1000}, log_path=log, interval_s=0.0, max_samples=2)
        watch({target: 1000}, log_path=log, interval_s=0.0, max_samples=2)

        assert len(log.read_text(encoding="utf-8").splitlines()) == 4

    def test_creates_the_log_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "m.bin"
        target.write_bytes(b"x" * 8)
        log = tmp_path / "nested" / "deeper" / "progress.jsonl"

        watch({target: 8}, log_path=log, interval_s=0.0, max_samples=1)

        assert log.exists()
