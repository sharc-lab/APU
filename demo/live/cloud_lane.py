"""Cloud parts of the live demo, on the controller (the only machine with the key; it is read from this process's
environment by src/cloud/client.py and never printed, logged or written here).

Every call goes through CloudClient(mode="real", spend_cap_usd <= 10, ledger demo/ledger/cloud_ledger_demo.jsonl),
the demo's own ledger. Before each task the projected ledger cost is checked against the cap and against the
session limit (USD 3, operator rule: stop and report before anything that could exceed USD 3 in one session); if a
live call is not possible (live calls off, no key, cap or session limit, provider error) the lane shows the recorded
all-cloud result of the same task and model (results/all_cloud_b3_v1.jsonl), labelled REPLAYED, or "not available"
when that run has no such task (it covers seeds 20260901-03 only).

Cost per task shown on screen is the ledger's (usd_ledger), and the cache-priced figure (the basis of the headline row
hybrid-route-at-start-measured) beside it.

PRE-JOB GATE (operator 2026-10-24: "gate the demo lane the same way on its own USD 10 ledger, USD 3 per session").
Before the session's first live call, src/cloud/console_gate.py::pre_job runs on the demo ledger: the session's
estimate is its session limit (USD 3 by default); a limit above USD 3 is refused at construction unless
operator_approval (the --operator-approval flag of complete_run / server, that command line only) names the job id
demo_live_session and an amount "USD <n>" at or above the limit, and the approval is written to the demo ledger as
its own row before the first call. Then the Console check: the operator's Anthropic Console reading must be under
24 h old and agree with the billed ledgers (main + demo, each row once, whichever checkout runs this) within
max(USD 0.50, 5%), and the demo ledger's billed headroom under its cap (USD 10 or lower) must cover the session
limit. A refusal makes every live call of the session REPLAYED, with the refusal as the reason (printed in full once).
An approval is accepted only from an interactive console and is confirmed there when the lane is built (at startup):
console_gate.confirm_approval refuses any automated launch (queue marker, Task Scheduler ancestry, AI agent session,
no terminal) and asks the operator to type "demo_live_session <amount>" on the console device.
ONE SESSION BUDGET for every live demo path in every process (this lane and the dashboard's Escalator): the session is
the demo ledger's latest demo_session row (escalate.open_session / session_remaining), its spend the demo ledger's
billed call rows since it started. Fakes without stub_mode=False (tests) are never gated.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "harness", REPO / "scripts", REPO / "src" / "dse", REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import demo_live as L  # noqa: E402
from demo.dashboard import escalate as ES  # noqa: E402

SESSION_LIMIT_USD = ES.DEMO_SESSION_LIMIT_USD
JOB_ID = "demo_live_session"
PROJECTION_SAFETY = 1.5
TASK_CAP_USD = 0.25   # per-task cap (src/cloud/task_budget.py, operator item C); under the session limit, never above
PRICES = REPO / "results" / "cloud_prices_anthropic_20261014.json"


def rates_for(model: str) -> dict:
    import cloud_redo as CR
    return CR.model_rates(CR.load_price_table(PRICES), model)


def projected_ledger_usd(task: dict, model: str, calls: int = 3) -> float:
    """Upper-side projection on the ledger basis (every input token at the input rate): the rebuilt prompt's
    calibrated size times `calls` provider requests, plus 2048 output tokens each, times PROJECTION_SAFETY."""
    import x2_r2_agent as ag
    r = rates_for(model)["short"]
    prompt = round(ag.prompt_tokens_est(task["history"], L.EV.tools_payload("v2")) * 1.25)
    return round(PROJECTION_SAFETY * calls * (prompt * r["input"] + 2048 * r["output"]) / 1e6, 6)


def lane_key(task: dict) -> str:
    return f"{task.get('seed')}:{task.get('task_id')}"


class CloudLane:
    def __init__(self, env=None, ledger: Path = ES.LEDGER, client_factory=None, session_limit: float = SESSION_LIMIT_USD,
                 all_cloud_rows=None, operator_approval: str | None = None, gate_fn=None, context_fn=None,
                 confirm_fn=None, task_cap_usd: float = TASK_CAP_USD):
        from src.cloud import console_gate as CG  # noqa: PLC0415
        # a session limit above USD 3 needs the operator's approval (raises JobApprovalRefused before anything runs);
        # an approval is accepted only from an interactive console, confirmed there at startup (never from automation)
        appr = CG.validate_approval(operator_approval, job_id=JOB_ID, estimate_usd=float(session_limit),
                                    estimate_basis="session limit", limit_usd=SESSION_LIMIT_USD)
        self.confirmed = CG.confirm_approval(appr, context_fn, confirm_fn) if appr is not None else None
        self.operator_approval = operator_approval
        self.gate_fn = gate_fn or ES.demo_pre_job
        self.gate = None
        self.gate_refusal = None
        self.env = os.environ if env is None else env
        self.ledger = Path(ledger)
        self.factory = client_factory or ES._default_client_factory
        self.session_limit = session_limit
        self.task_cap_usd = float(min(task_cap_usd, session_limit))
        self.client = None
        self.all_cloud = all_cloud_rows if all_cloud_rows is not None else L.read_jsonl(
            REPO / "results" / L.ALL_CLOUD_FILE)
        self.log = []

    def live_possible(self) -> tuple[bool, str]:
        if str(self.env.get(ES.ENV_LIVE, "")).strip() != "1":
            return False, f"live calls are off ({ES.ENV_LIVE} is not 1)"
        if not self.env.get(ES.KEY_ENV):
            return False, "no key in this process"
        return True, ""

    @property
    def session_spend(self) -> float:
        """This lane's own live spend (ledger USD of its calls); the shared session budget is ES.session_remaining."""
        return sum(float(x.get("usd_ledger") or 0.0) for x in self.log)

    def budget_left(self) -> float:
        """Cap headroom and the SHARED demo session budget (every process's demo calls since the session started)."""
        return max(0.0, min(ES.cap_usd(self.env) - ES.ledger_total(self.ledger),
                            ES.session_remaining(self.ledger, default_limit=self.session_limit)))

    def replay(self, task: dict, model: str, reason: str) -> dict:
        r = L.replayed_cloud(self.all_cloud, task, model)
        if r is None:
            return {"route": "cloud", "model": model, "label": "NOT AVAILABLE", "unavailable": True,
                    "end_state_ok": None, "reason": reason + "; no recorded all-cloud row for this task"}
        return {**r, "reason": reason}

    def _open_gate(self, client) -> None:
        """pre_job once per session, before its first live call; a refusal holds for the rest of the session."""
        from src.cloud import console_gate as CG  # noqa: PLC0415
        if self.gate is not None or self.gate_refusal is not None:
            return
        try:
            gate = self.gate_fn(client, job_id=JOB_ID, estimate_usd=float(self.session_limit),
                                estimate_basis="session limit (a task whose projection would pass it replays)",
                                confirmed=self.confirmed)
            ES.open_session(self.ledger, float(self.session_limit), approval_row=(gate or {}).get("approval"))
            self.gate = gate
        except CG.ConsoleCheckRefused as e:
            print(str(e), flush=True)
            self.gate_refusal = str(e).splitlines()[0]

    def __call__(self, lane: str, task: dict, model: str) -> dict:
        ok, why = self.live_possible()
        if not ok:
            return self.replay(task, model, why)
        if self.gate_refusal is not None:
            return self.replay(task, model, self.gate_refusal)
        proj = projected_ledger_usd(task, model)
        if proj > self.budget_left():
            return self.replay(task, model, f"projected ${proj:.4f} is above the budget left (${self.budget_left():.4f})")
        try:
            if self.client is None:
                self.client = self.factory(ES.cap_usd(self.env), self.ledger)
            self._open_gate(self.client)
            if self.gate_refusal is not None:
                return self.replay(task, model, self.gate_refusal)
            if proj > self.budget_left():   # again, on the opened shared session (another process may have spent)
                return self.replay(task, model, f"projected ${proj:.4f} is above the session budget left "
                                                f"(${self.budget_left():.4f})")
            from src.cloud import task_budget as TB  # noqa: PLC0415
            tbc = TB.BudgetedClient(self.client, self.task_cap_usd, ledger_path=self.ledger)
            tbc.start_task(f"{lane_key(task)}:{model}")
            try:
                res = L.run_cloud(tbc, task, model, rates_for(model))
            except TB.TaskBudgetExhausted as e:
                # hand-off, not a retry: the task stops on the cloud lane and is shown as such
                self.log.append({"lane": lane, "seed": task["seed"], "task_id": task["task_id"], "model": model,
                                 "usd_ledger": tbc.task_spent_usd, "projected": proj,
                                 "outcome": TB.OUTCOME_EXHAUSTED})
                return {"route": "cloud", "model": model, "label": "BUDGET EXHAUSTED", "end_state_ok": None,
                        "outcome": TB.OUTCOME_EXHAUSTED, "handed_off": True, "task_budget": e.record,
                        "reason": str(e)}
        except Exception as e:
            return self.replay(task, model, f"live call failed: {type(e).__name__}: {ES._scrub(e, self.env)}")
        self.log.append({"lane": lane, "seed": task["seed"], "task_id": task["task_id"], "model": model,
                         "usd_ledger": res.get("usd_ledger"), "projected": proj})
        return res
