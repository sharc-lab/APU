"""Demo lane pre-job gate (operator 2026-10-24: "gate the demo lane the same way on its own USD 10 ledger, USD 3 per
session"): demo/live/cloud_lane.CloudLane and demo/dashboard/escalate.Escalator pass src/cloud/console_gate.py::pre_job
on the demo ledger before their first live call, and share ONE session budget through the demo ledger. Fake clients,
fake ledgers and fake console files under tmp_path; no network, no key, nothing is sent."""
from __future__ import annotations

import functools
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))

import demo_live as L  # noqa: E402
from demo.dashboard import escalate as ES  # noqa: E402
from demo.live import cloud_lane as CL  # noqa: E402
from src.cloud import console_gate as CG  # noqa: E402

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
T0 = datetime(2026, 10, 24, 9, 0, tzinfo=timezone.utc)
MODEL = "claude-sonnet-5-5"
ENV = {"ANTHROPIC_API_KEY": "k" * 20, "DEMO_CLOUD_LIVE": "1"}
INTERACTIVE = {"interactive": True, "automation_reasons": [], "stdin_tty": True, "console": True,
               "parent_chain": ["pwsh.exe", "explorer.exe"], "env_markers": {}}
APPROVE4 = "approve demo_live_session up to USD 4.00: rehearsal"


def _typed(job, amount):
    return f"{job} {amount:.2f}"


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    """main ledger billed 6.0, demo ledger billed 1.0 (cap 10), both in one (non-git) checkout: canonical."""
    _jsonl(tmp_path / "results" / "cloud_ledger.jsonl",
           [{"timestamp": T0.isoformat(), "model_id": MODEL, "cost_usd": 6.0, "cost_usd_upper_bound": 42.0}])
    demo = tmp_path / "demo" / "ledger" / "cloud_ledger_demo.jsonl"
    _jsonl(demo, [{"timestamp": (T0 + timedelta(minutes=1)).isoformat(), "model_id": MODEL, "cost_usd": 1.0,
                   "cost_usd_upper_bound": 2.0}])
    return tmp_path


def _demo(repo):
    return repo / "demo" / "ledger" / "cloud_ledger_demo.jsonl"


def _console(path, total):
    path.write_text(json.dumps({"console_total_usd": total, "read_at_utc": "2026-10-24T11:00:00Z",
                                "range_start_utc": "2026-10-24", "range_end_utc": "2026-10-24",
                                "read_by": "test operator"}), encoding="utf-8")
    return path


class FakeReal:
    """A real-mode client stand-in: never sends (prepare raises)."""
    stub_mode = False
    _reserved_usd = 0.0
    running_total_usd = 1.0

    def __init__(self, cap, ledger):
        self.spend_cap_usd, self.ledger_path = cap, Path(ledger)

    def prepare(self, **kw):
        raise AssertionError("nothing may be sent")


def _gate(console):
    return functools.partial(ES.demo_pre_job, check_path=console, now=NOW, head="h")


@pytest.fixture(scope="module")
def task8():
    return L.build_task(L.read_jsonl(REPO / "results" / L.B3_FILE), 20260901, "inv_rename")


def _rows(ledger, record):
    return [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()
            if json.loads(x).get("record") == record]


def _lane(repo, **kw):
    kw.setdefault("client_factory", FakeReal)
    kw.setdefault("all_cloud_rows", [])
    return CL.CloudLane(env=ENV, ledger=_demo(repo), **kw)


# -- (c) the USD 3 session limit refuses without the flag; nothing sent, no ledger row ---------------------------

def test_session_limit_above_three_refused_without_approval_nothing_written(repo):
    before = _demo(repo).read_text(encoding="utf-8")
    with pytest.raises(CG.JobApprovalRefused, match="USD 3.00 per-job limit"):
        _lane(repo, session_limit=4.0)
    with pytest.raises(CG.JobApprovalRefused, match="below the estimate"):
        _lane(repo, session_limit=4.0, operator_approval="approve demo_live_session up to USD 3.50")
    assert _demo(repo).read_text(encoding="utf-8") == before
    assert CL.SESSION_LIMIT_USD == 3.0 and ES.CAP_CEILING_USD == 10.0


def test_task_over_the_session_budget_replays_and_sends_nothing(repo, task8, monkeypatch):
    sent = []
    monkeypatch.setattr(CL.L, "run_cloud", lambda *a: sent.append(a) or {"usd_ledger": 0.0})
    before = _demo(repo).read_text(encoding="utf-8")
    lane = _lane(repo, session_limit=0.0, gate_fn=_gate(_console(repo / "c.json", 7.0)))
    r = lane("cloud", task8, MODEL)
    assert "budget left" in r["reason"] and sent == [] and _demo(repo).read_text(encoding="utf-8") == before


# -- (a) the approval cannot ride automation ------------------------------------------------------------------------

def test_lane_approval_refused_when_automated(repo):
    with pytest.raises(CG.JobApprovalRefused, match="looks automated"):
        _lane(repo, session_limit=4.0, operator_approval=APPROVE4,
              context_fn=lambda: dict(INTERACTIVE, interactive=False,
                                      automation_reasons=["launched by a queue runner or other automation "
                                                          "(APU_QUEUE_JOB_ID set)"]),
              confirm_fn=_typed)
    with pytest.raises(CG.JobApprovalRefused, match="did not match"):
        _lane(repo, session_limit=4.0, operator_approval=APPROVE4, context_fn=lambda: dict(INTERACTIVE),
              confirm_fn=lambda job, amt: "ok")


# -- the Console check on the demo lane --------------------------------------------------------------------------

def test_lane_replays_when_the_console_check_refuses_and_sends_nothing(repo, task8, capsys):
    lane = _lane(repo, gate_fn=_gate(repo / "missing.json"))
    r = lane("cloud", task8, MODEL)
    assert r["label"] == "NOT AVAILABLE" and "refused (console check)" in r["reason"]
    assert "WHAT TO DO" in capsys.readouterr().out
    r2 = lane("cloud", task8, MODEL)                                     # the refusal holds for the session
    assert "refused (console check)" in r2["reason"] and lane.log == []
    assert _rows(_demo(repo), ES.SESSION_RECORD) == []


def test_lane_refuses_on_disagreement_main_plus_demo_compared(repo, task8):
    lane = _lane(repo, gate_fn=_gate(_console(repo / "c.json", 6.0)))   # main 6 + demo 1 = 7 vs Console 6
    r = lane("cloud", task8, MODEL)
    assert "disagree" in r["reason"] and "the ledgers are higher" in r["reason"]


def test_lane_gate_passes_and_records_the_approval_before_the_first_call(repo, task8, monkeypatch):
    order = []
    monkeypatch.setattr(CL.L, "run_cloud",
                        lambda client, task, model, rates: order.append(len(_rows(_demo(repo), "operator_approval")))
                        or {"route": "cloud", "usd_ledger": 0.01})
    monkeypatch.setattr(CL, "projected_ledger_usd", lambda task, model: 0.5)
    lane = _lane(repo, session_limit=4.0, operator_approval=APPROVE4, context_fn=lambda: dict(INTERACTIVE),
                 confirm_fn=_typed, gate_fn=_gate(_console(repo / "c.json", 7.0)))
    lane("cloud", task8, MODEL)
    lane("cloud", task8, MODEL)
    assert order == [1, 1]                                               # recorded once, before the first call
    a = _rows(_demo(repo), "operator_approval")[0]
    assert a["job_id"] == "demo_live_session" and a["approved_usd"] == 4.0 and a["git_head"] == "h"
    assert a["launch_context"]["confirmed_on_console"] is True
    s = _rows(_demo(repo), ES.SESSION_RECORD)
    assert len(s) == 1 and s[0]["limit_usd"] == 4.0 and s[0]["approval_ts_utc"] == a["ts_utc"]
    assert lane.gate["headroom_billed_usd"] == pytest.approx(9.0) and lane.gate["cap_usd"] == 10.0


# -- one shared USD 3 session budget (CloudLane + Escalator, any process) ----------------------------------------

def _spend(ledger, usd, minutes):
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat(),
                             "model_id": MODEL, "cost_usd": usd, "cost_usd_upper_bound": usd}) + "\n")


def _fresh(tmp_path):
    p = tmp_path / "shared" / "cloud_ledger_demo.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text("", encoding="utf-8")
    return p


def _pass(client, **kw):
    return {"gated": False, "reason": "budget test (the gate itself is tested above)"}


def test_session_budget_is_shared_through_the_demo_ledger(tmp_path):
    demo = _fresh(tmp_path)
    st = ES.open_session(demo)                                           # e.g. the live screen's process
    assert st["limit_usd"] == 3.0 and ES.session_remaining(demo) == pytest.approx(3.0)
    _spend(demo, 2.5, 1)                                                 # e.g. the dashboard's process spends
    assert ES.open_session(demo)["session_id"] == st["session_id"]       # another process joins, same session
    assert ES.session_remaining(demo) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="needs an operator approval"):
        ES.open_session(demo, 4.0)                                       # no raise of the limit without approval


def test_escalator_and_lane_draw_on_one_budget(tmp_path, task8, monkeypatch):
    demo = _fresh(tmp_path)
    ES.open_session(demo)
    _spend(demo, 2.9, 1)                                                 # the lane already spent USD 2.9
    monkeypatch.setattr(CL, "projected_ledger_usd", lambda task, model: 0.5)
    lane = CL.CloudLane(env=ENV, ledger=demo, client_factory=FakeReal, all_cloud_rows=[], gate_fn=_pass)
    assert "budget left" in lane("cloud", task8, MODEL)["reason"]       # 0.5 > 0.1 left
    esc = ES.Escalator({"available": True, "redo": {}}, ledger=demo, env={ES.ENV_LIVE: "1"},
                       client_factory=FakeReal, gate_fn=_pass)
    made = []
    monkeypatch.setattr(esc, "client_factory", lambda cap, led: made.append(FakeReal(cap, led)) or made[-1])
    monkeypatch.setattr(ES, "_cloud_redo", lambda: (_ for _ in ()).throw(RuntimeError("stop before sending")))
    esc.run()
    # the Escalator's client cap was lowered to its billed total plus the USD 0.1 the session has left
    assert made and made[0].spend_cap_usd == pytest.approx(1.0 + 0.1)
    _spend(demo, 0.2, 2)                                                 # the session is now used up
    esc2 = ES.Escalator({"available": True, "redo": {}}, ledger=demo, env={ES.ENV_LIVE: "1"},
                        client_factory=FakeReal, gate_fn=_pass)
    out = esc2.run()
    assert "SessionBudgetUsed" in out["reason"]


def test_escalator_live_call_is_gated(repo):
    esc = ES.Escalator({"available": True, "redo": {}}, ledger=_demo(repo), env={ES.ENV_LIVE: "1"},
                       client_factory=FakeReal, gate_fn=_gate(repo / "missing.json"))
    out = esc.run()
    assert out["mode"] == ES.BEING_MEASURED and "ConsoleCheckRefused" in out["reason"]


def test_demo_ledger_is_canonical_in_the_main_checkout():
    assert ES.LEDGER == CG.canonical_demo_ledger(REPO)
    assert ES.LEDGER.parts[-3:] == ("demo", "ledger", "cloud_ledger_demo.jsonl")
