"""C1's in-harness responsiveness sampler: times a trivial local subprocess every 30 s (monotonic clock), start/stop
cleanly, and summarizes to resp_median_s/resp_max_s/resp_n."""

import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import t2s_night2 as n2  # noqa: E402


def test_sampler_collects_and_summarizes():
    r = n2.ResponsivenessSampler()
    with mock.patch.object(n2, "st") as st_mock:
        st_mock.median.side_effect = lambda xs: sorted(xs)[len(xs) // 2]
        r.samples = [0.05, 0.06, 0.20]  # simulate collected samples without waiting 90 s for real ones
        summary = r.summary()
    assert summary["resp_n"] == 3
    assert summary["resp_max_s"] == 0.20
    assert summary["resp_median_s"] == 0.06


def test_sampler_empty_summary_is_none_not_a_crash():
    r = n2.ResponsivenessSampler()
    assert r.summary() == {"resp_median_s": None, "resp_max_s": None, "resp_n": 0}


def test_sampler_start_stop_collects_at_least_one_real_sample():
    r = n2.ResponsivenessSampler()
    r.start()
    time.sleep(0.5)
    r.stop()
    assert r.samples, "expected at least one real subprocess timing in half a second"
    assert all(s >= 0 for s in r.samples)
