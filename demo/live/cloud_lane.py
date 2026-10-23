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

SESSION_LIMIT_USD = 3.0
PROJECTION_SAFETY = 1.5
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


class CloudLane:
    def __init__(self, env=None, ledger: Path = ES.LEDGER, client_factory=None, session_limit: float = SESSION_LIMIT_USD,
                 all_cloud_rows=None):
        self.env = os.environ if env is None else env
        self.ledger = Path(ledger)
        self.factory = client_factory or ES._default_client_factory
        self.session_limit = session_limit
        self.session_spend = 0.0
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

    def budget_left(self) -> float:
        return max(0.0, min(ES.cap_usd(self.env) - ES.ledger_total(self.ledger),
                            self.session_limit - self.session_spend))

    def replay(self, task: dict, model: str, reason: str) -> dict:
        r = L.replayed_cloud(self.all_cloud, task, model)
        if r is None:
            return {"route": "cloud", "model": model, "label": "NOT AVAILABLE", "unavailable": True,
                    "end_state_ok": None, "reason": reason + "; no recorded all-cloud row for this task"}
        return {**r, "reason": reason}

    def __call__(self, lane: str, task: dict, model: str) -> dict:
        ok, why = self.live_possible()
        if not ok:
            return self.replay(task, model, why)
        proj = projected_ledger_usd(task, model)
        if proj > self.budget_left():
            return self.replay(task, model, f"projected ${proj:.4f} is above the budget left (${self.budget_left():.4f})")
        try:
            if self.client is None:
                self.client = self.factory(ES.cap_usd(self.env), self.ledger)
            res = L.run_cloud(self.client, task, model, rates_for(model))
        except Exception as e:
            return self.replay(task, model, f"live call failed: {type(e).__name__}: {ES._scrub(e, self.env)}")
        self.session_spend += float(res.get("usd_ledger") or 0.0)
        self.log.append({"lane": lane, "seed": task["seed"], "task_id": task["task_id"], "model": model,
                         "usd_ledger": res.get("usd_ledger"), "projected": proj})
        return res
