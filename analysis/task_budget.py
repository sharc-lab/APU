"""Adapter: the per-task cloud budget (C, src/cloud/task_budget.py) behind analysis/hybrid_full_pipeline.py's TaskBudget
interface (make_task_budget(repo).allow(task_key, usd_estimate) -> {"allow", "cap_usd"}).

The replay has no live client, so it applies C's rule to the estimate: a cloud use is allowed while the task's
spending stays within the per-task cap. The cap is the same USD 0.25 the post-run check replay and the demo cloud lane
use (analysis/post_run_check_replay.TASK_CAP_USD); the live path is src/cloud/task_budget.BudgetedClient, which
checks the per-call upper-bound reservation before each call.
"""
from __future__ import annotations

import sys
from pathlib import Path


class _ReplayTaskBudget:
    def __init__(self, cap_usd: float):
        if cap_usd is None or float(cap_usd) <= 0:
            raise ValueError("per-task cap must be a positive USD amount")
        self.cap_usd, self.spent = float(cap_usd), {}

    def allow(self, task_key, usd_estimate: float) -> dict:
        spent = self.spent.get(task_key, 0.0)
        ok = spent + float(usd_estimate) <= self.cap_usd
        if ok:
            self.spent[task_key] = spent + float(usd_estimate)
        return {"allow": ok, "cap_usd": self.cap_usd, "task_spent_usd": self.spent.get(task_key, 0.0)}


def make_task_budget(repo):
    sys.path.insert(0, str(Path(repo) / "analysis"))
    import post_run_check_replay as replay  # noqa: PLC0415
    return _ReplayTaskBudget(replay.TASK_CAP_USD)
