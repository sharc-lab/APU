"""Per-task cloud budget (operator item C, 2026-10-29): "hard USD cap per task inside the cloud lane; on hit, stop
retries and fail cleanly or hand off. Test with a forced retry loop." Pre-registered in docs/EXPERIMENTS.md
"PRE-REGISTRATION (2026-10-29): post-run check and per-task cloud budget".

BudgetedClient wraps a CloudClient (real or stub) for one cloud lane (scripts/cloud_redo.py, the local-first hybrid,
demo/live/cloud_lane.py):

- start_task(key) opens a task; every prepare()/call() after it belongs to that task until the next start_task.
- Before EVERY call: the call's upper-bound reservation (CloudClient.projected_cost_usd, the same figure the client
  reserves against the job cap) plus the task's spend so far must be <= the per-task cap. If not, no call is made: the
  task is marked exhausted, a `task_budget_exhausted` record is appended to the ledger beside the client's own rows,
  and TaskBudgetExhausted is raised. Every later prepare() for that task raises again without another ledger record,
  so a retry loop stops at the first refused call.
- A settled call counts at max(billed, upper bound, reservation): the counted spend is never below what the call can
  have cost, so the billed total of a task can never pass its cap.
- A send that fails (the client releases its reservation and writes no ledger row) counts 0, as in the client.

It composes with the existing limits and loosens neither: the wrapped client's own spend cap (the USD 2 per-job cap
callers set via spend_cap_usd = ledger total + job cap) and the Console gate (src/cloud/console_gate.pre_job, run by the
caller before the first call) are untouched; the per-task check runs first and can only refuse more.

Stub clients: the projection needs a priced model id; a stub call costs 0, so a stub task is counted at its reservation
(the conservative figure) and still exercises the cap.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUTCOME_EXHAUSTED = "budget_exhausted"
LEDGER_RECORD = "task_budget_exhausted"


class TaskBudgetExhausted(Exception):
    """The per-task cap would be passed by the next call; nothing was sent."""

    def __init__(self, message: str, record: dict):
        super().__init__(message)
        self.record = record


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class _BudgetedPrepared:
    """A PreparedCall whose settle() adds the call's cost to the task's spend."""

    def __init__(self, owner: "BudgetedClient", inner: Any, reserved: float):
        self._owner, self._inner, self._reserved = owner, inner, reserved

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def send(self):
        return self._inner.send()

    def settle(self, response):
        res = self._inner.settle(response)
        self._owner._count(res, self._reserved)
        return res

    def release(self):
        return self._inner.release()


class BudgetedClient:
    def __init__(self, inner: Any, cap_usd: float, ledger_path: str | Path | None = None):
        if cap_usd is None or float(cap_usd) <= 0:
            raise ValueError("per-task cap must be a positive USD amount")
        self.inner = inner
        self.cap_usd = float(cap_usd)
        led = ledger_path if ledger_path is not None else getattr(inner, "ledger_path", None)
        if led is None:
            raise ValueError("no ledger path for the per-task budget records")
        self.ledger_path = Path(led)
        self.task_key: str | None = None
        self.task_spent_usd = 0.0
        self.task_calls = 0
        self.exhausted = False
        self.records: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    # ── task scope ─────────────────────────────────────────────────────────────────────────────
    def start_task(self, key: str) -> None:
        self.task_key, self.task_spent_usd, self.task_calls, self.exhausted = str(key), 0.0, 0, False

    def task_state(self) -> dict:
        return {"task_key": self.task_key, "task_cap_usd": self.cap_usd, "task_spent_usd": self.task_spent_usd,
                "task_calls": self.task_calls, "task_budget_exhausted": self.exhausted}

    def _count(self, res: Any, reserved: float) -> None:
        billed = float(getattr(res, "cost_usd", 0.0) or 0.0)
        upper = getattr(res, "cost_usd_upper_bound", None)
        self.task_spent_usd += max(billed, float(upper or 0.0), float(reserved))
        self.task_calls += 1

    def _refuse(self, model_id: str, projected: float) -> None:
        msg = (f"per-task cap: task {self.task_key} has counted ${self.task_spent_usd:.4f} over {self.task_calls} "
               f"call(s); the next call's upper-bound reservation ${projected:.4f} would bring it to "
               f"${self.task_spent_usd + projected:.4f}, over the ${self.cap_usd:.4f} per-task cap; nothing sent, "
               f"outcome {OUTCOME_EXHAUSTED}")
        rec = {"record": LEDGER_RECORD, "ts_utc": _utc(), "task_key": self.task_key, "model_id": model_id,
               "task_cap_usd": self.cap_usd, "task_spent_usd": self.task_spent_usd, "task_calls": self.task_calls,
               "next_call_reservation_usd": projected, "outcome": OUTCOME_EXHAUSTED, "cost_usd": 0.0}
        if not self.exhausted:
            self.exhausted = True
            self.records.append(rec)
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        raise TaskBudgetExhausted(msg, rec)

    # ── calls ──────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def projected_cost_usd(*, model_id: str, messages: list[dict], expected_output_tokens: int | None = None,
                           **kwargs) -> float:
        """The upper-bound reservation CloudClient.prepare() makes for this call (same function, same rates),
        computed here so the per-task check never depends on the wrapped object."""
        from src.cloud import client as cc  # noqa: PLC0415
        return cc.CloudClient.projected_cost_usd(None, model_id=model_id, messages=messages,
                                                 expected_output_tokens=expected_output_tokens, **kwargs)

    def prepare(self, *, model_id: str, messages: list[dict], expected_output_tokens: int | None = None, **kwargs):
        if self.task_key is None:
            raise RuntimeError("BudgetedClient.prepare() before start_task()")
        projected = self.projected_cost_usd(model_id=model_id, messages=messages,
                                            expected_output_tokens=expected_output_tokens, **kwargs)
        if self.exhausted or self.task_spent_usd + projected > self.cap_usd + 1e-12:
            self._refuse(model_id, projected)
        p = self.inner.prepare(model_id=model_id, messages=messages, expected_output_tokens=expected_output_tokens,
                               **kwargs)
        return _BudgetedPrepared(self, p, projected)

    def call(self, *, model_id: str, messages: list[dict], expected_output_tokens: int | None = None, **kwargs):
        p = self.prepare(model_id=model_id, messages=messages, expected_output_tokens=expected_output_tokens, **kwargs)
        return p.settle(p.send())
