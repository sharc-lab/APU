"""Planner backfill: when work finishes ahead of its estimates, trimmed items are pulled back in by priority."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import t2s_overnight as ov  # noqa: E402


def item(i, sec, prio, est):
    return {"item_id": i, "section": sec, "prio": prio, "est_s": est, "ord": prio}


def test_backfill_pulls_by_priority_when_ahead_of_estimate():
    kept = [item("A1", "A", 1, 3600), item("A2", "A", 2, 3600)]
    dropped = [item("C1", "C", 30, 3600), item("B1", "B", 20, 3600), item("C2", "C", 31, 3600)]
    budget = 2.5 * 3600
    on_time = ov.simulate_backfill(kept, dropped, budget, 1.0)
    assert on_time["events"] == [] and on_time["items_run"] == 2 and on_time["still_trimmed"] == 3
    fast = ov.simulate_backfill(kept, dropped, budget, 0.5)
    pulled = [p for e in fast["events"] for p in e["pulled"]]
    assert pulled == ["B1", "C1"]  # highest priority first, and only what still fits
    assert fast["finished_h"] <= fast["budget_h"]


def test_pick_backfill_respects_remaining_queue_and_sycl_gate():
    queue = [item("A2", "A", 2, 3600)]
    dropped = [item("B1", "B", 20, 3600)]
    assert ov.pick_backfill(queue, dropped, 3600) == []  # queued work already uses the time
    assert [p["item_id"] for p in ov.pick_backfill(queue, dropped, 2 * 3600)] == ["B1"]
    cell = {"item_id": "D_c", "section": "D", "prio": 91, "est_s": 60, "kind": "cell", "backend": "sycl", "ord": 91}
    assert ov.pick_backfill([], [cell], 3600) == []  # SYCL cell needs D_prepare first
    assert [p["item_id"] for p in ov.pick_backfill([], [cell], 3600, done_ids=["D_prepare"])] == ["D_c"]
