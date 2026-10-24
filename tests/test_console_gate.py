"""Billed cap basis and the Console check before a live cloud job (operator 2026-10-23, register row
cloud-spend-billed-vs-upper-bound; src/cloud/console_gate.py, src/cloud/client.py BILLED CAP BASIS AT START).

Fake ledgers, fake job files and fake console files under tmp_path only: no network, no key, no real Console figure.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.cloud import client as cc
from src.cloud import console_gate as g
from src.cloud.client import CloudClient, SpendCapExceeded

MODEL = "claude-sonnet-5-5"
R = cc.MODEL_PRICING[MODEL]["rates"]["short"]
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
T0 = datetime(2026, 10, 22, 10, 0, tzinfo=timezone.utc)


def _iso(t):
    return t.isoformat()


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _old_repo(tmp_path):
    """A repo whose ledger has 3 OLD rows (cost_usd only = upper bound, USD 14 each) whose usage is on file in a
    cloud job's result file: billed USD 2 each. Billed 6.0, upper bound 42.0."""
    ledger = tmp_path / "results" / "cloud_ledger.jsonl"
    _jsonl(ledger, [{"timestamp": _iso(T0 + timedelta(minutes=i + 1)), "model_id": MODEL, "cost_usd": 14.0,
                     "input_tokens": 1, "output_tokens": 1} for i in range(3)])
    job = [{"record": "run_start", "harness": "cloud_redo", "ts_utc": _iso(T0)}]
    job += [{"record": "cloud_redo_task", "ts_utc": _iso(T0 + timedelta(minutes=i + 1, seconds=1)), "usd": 2.0,
             "usd_ledger": 14.0, "n_calls": 1, "model_id": MODEL} for i in range(3)]
    job += [{"record": "run_end", "ts_utc": _iso(T0 + timedelta(minutes=10))}]
    _jsonl(tmp_path / "results" / "cloud_redo_v1.jsonl", job)
    return ledger


def _new_ledger(path, billed, upper, t=T0):
    _jsonl(path, [{"timestamp": _iso(t), "model_id": MODEL, "cost_usd": billed, "cost_usd_upper_bound": upper,
                   "input_tokens": 1, "output_tokens": 1}])
    return path


def _console(path, total, read_at=NOW - timedelta(hours=1), start="2026-10-22", end="2026-10-23", **extra):
    d = {"console_total_usd": total, "read_at_utc": read_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "range_start_utc": start, "range_end_utc": end, "read_by": "test operator", **extra}
    path.write_text(json.dumps(d), encoding="utf-8")
    return path


def _check(ledger, console, proj=1.0, cap=50.0, **kw):
    return g.check(proj, cap_usd=cap, ledger_path=ledger, check_path=console, now=NOW, **kw)


def _fake_fn(*, model_id, messages, api_key, **kwargs):
    return {"usage": {"prompt_tokens": 10, "completion_tokens": 5}, "choices": [{"message": {"content": "ok"}}]}


# -- billed cap basis -------------------------------------------------------------------------------------------

def test_start_total_is_the_register_billed_total_with_upper_bound_beside_it(tmp_path):
    ledger = _old_repo(tmp_path)
    c = CloudClient(mode="stub", ledger_path=ledger)
    assert c.running_total_usd == pytest.approx(6.0)
    assert c.running_total_usd_upper_bound == pytest.approx(42.0)
    assert c.cap_basis == "billed_with_recovered_usage"
    assert c._read_running_totals_per_row() == pytest.approx((42.0, 42.0))   # the old basis, for contrast
    s = c.spend_summary()
    assert "billed $6.0000" in s and "upper bound $42.0000" in s


def test_upper_bound_only_overrun_is_now_allowed(tmp_path):
    ledger = _old_repo(tmp_path)
    n_out = int(10.5e6 / R["output"])                  # a call whose upper-bound reservation is about USD 10.5
    proj = cc.reservation_cost_usd(MODEL, 2, n_out)
    c = CloudClient(api_key="sk-fake", ledger_path=ledger, role_override="controller", completion_fn=_fake_fn,
                    spend_cap_usd=50.0)
    assert 42.0 + proj > 50.0                          # on the old basis (or the upper bound) this was refused
    assert 6.0 + proj <= 50.0                          # on the billed basis it fits
    res = c.call(model_id=MODEL, messages=[{"role": "user", "content": "hi"}], expected_output_tokens=n_out)
    assert not res.stub
    last = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert last["running_total_basis"] == "billed_with_recovered_usage"
    assert last["running_total_usd"] == pytest.approx(6.0 + res.cost_usd)
    assert last["running_total_usd_upper_bound"] > 42.0


def test_billed_cap_hit_is_refused_and_prints_both_figures(tmp_path):
    ledger = _new_ledger(tmp_path / "l.jsonl", billed=48.0, upper=49.0)
    n_out = int(3.0e6 / R["output"])
    c = CloudClient(api_key="sk-fake", ledger_path=ledger, role_override="controller", completion_fn=_fake_fn,
                    spend_cap_usd=50.0)
    with pytest.raises(SpendCapExceeded) as e:
        c.call(model_id=MODEL, messages=[{"role": "user", "content": "hi"}], expected_output_tokens=n_out)
    msg = str(e.value)
    assert "$48.0000 billed" in msg and "upper bound $49.0000" in msg and "cap checks the billed total" in msg


def test_fallback_to_per_row_basis_when_billed_basis_fails(tmp_path, monkeypatch, capsys):
    ledger = _old_repo(tmp_path)
    monkeypatch.setattr(g, "ledger_spend", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    c = CloudClient(mode="stub", ledger_path=ledger)
    assert c.running_total_usd == pytest.approx(42.0) and c.cap_basis == "per_row_old_rows_at_upper_bound"
    assert "WARNING: billed cap basis unavailable" in capsys.readouterr().out


# -- console check ----------------------------------------------------------------------------------------------

def test_missing_file_refused_with_instructions(tmp_path):
    ledger = _old_repo(tmp_path)
    with pytest.raises(g.ConsoleCheckRefused) as e:
        _check(ledger, tmp_path / "results" / "console_usage_check.json")
    msg = str(e.value)
    assert "does not exist" in msg and "WHAT TO DO" in msg and "console_total_usd" in msg
    assert '"range_start_utc": "2026-10-22"' in msg and "Nothing was sent" in msg


def test_fresh_passes_stale_refused_and_window_can_only_tighten(tmp_path, monkeypatch):
    ledger = _old_repo(tmp_path)
    fresh = _console(tmp_path / "fresh.json", 6.0)
    rep = _check(ledger, fresh)
    assert rep["gated"] and rep["ledger_billed_in_range_usd"] == pytest.approx(6.0)
    assert rep["ledger_upper_bound_in_range_usd"] == pytest.approx(42.0)
    stale = _console(tmp_path / "stale.json", 6.0, read_at=NOW - timedelta(hours=25), start="2026-10-22",
                     end="2026-10-22")
    with pytest.raises(g.ConsoleCheckRefused, match="older than the 24 h freshness window"):
        _check(ledger, stale)
    with pytest.raises(g.ConsoleCheckRefused, match="tightened, never loosened"):
        _check(ledger, stale, freshness_h=48)                     # loosening is refused outright
    monkeypatch.setenv("APU_CONSOLE_FRESHNESS_HOURS", "48")       # no environment override exists any more
    with pytest.raises(g.ConsoleCheckRefused, match="older than the 24 h"):
        _check(ledger, stale)
    with pytest.raises(g.ConsoleCheckRefused, match="older than the 0.5 h"):
        _check(ledger, fresh, freshness_h=0.5)                    # tightening works (fresh is 1 h old)


def test_within_tolerance_passes_outside_refused(tmp_path):
    ledger = _old_repo(tmp_path)                                   # billed 6.0 in range
    assert _check(ledger, _console(tmp_path / "a.json", 6.45))["tolerance_usd"] == pytest.approx(0.50)
    with pytest.raises(g.ConsoleCheckRefused) as e:
        _check(ledger, _console(tmp_path / "b.json", 6.60))
    m = str(e.value)
    assert "ledgers billed $6.0000" in m and "Console $6.6000" in m and "the Console is higher, by $0.6000" in m
    with pytest.raises(g.ConsoleCheckRefused, match=r"the ledgers are higher, by \$0\.6000"):
        _check(ledger, _console(tmp_path / "c.json", 5.40))
    # the upper bound (42.0) must NOT be what is compared: a Console figure near it is refused
    with pytest.raises(g.ConsoleCheckRefused):
        _check(ledger, _console(tmp_path / "d.json", 42.0))


def test_tolerance_is_five_percent_above_ten_dollars(tmp_path):
    ledger = _new_ledger(tmp_path / "results" / "cloud_ledger.jsonl", billed=100.0, upper=300.0)
    assert g.tolerance_usd(100.0) == pytest.approx(5.0)
    assert _check(ledger, _console(tmp_path / "a.json", 104.0), cap=500.0)["gated"]
    with pytest.raises(g.ConsoleCheckRefused):
        _check(ledger, _console(tmp_path / "b.json", 106.0), cap=500.0)


def test_outside_repo_spend_needs_a_note_and_is_subtracted(tmp_path):
    ledger = _old_repo(tmp_path)
    with pytest.raises(g.ConsoleCheckRefused, match="outside_repo_note"):
        _check(ledger, _console(tmp_path / "a.json", 9.0, outside_repo_usd=3.0))
    rep = _check(ledger, _console(tmp_path / "b.json", 9.0, outside_repo_usd=3.0, outside_repo_note="other project"))
    assert rep["console_repo_usd"] == pytest.approx(6.0)


def test_calls_after_the_read_are_not_compared_and_range_must_reach_the_read_day(tmp_path):
    ledger = _old_repo(tmp_path)
    with ledger.open("a", encoding="utf-8") as fh:            # a USD 5 call after the Console was read
        fh.write(json.dumps({"timestamp": _iso(NOW - timedelta(minutes=10)), "model_id": MODEL, "cost_usd": 5.0,
                             "cost_usd_upper_bound": 5.0}) + "\n")
    rep = _check(ledger, _console(tmp_path / "a.json", 6.0))
    assert rep["ledger_billed_in_range_usd"] == pytest.approx(6.0) and rep["spent_billed_usd"] == pytest.approx(11.0)
    with pytest.raises(g.ConsoleCheckRefused, match="would go unchecked"):
        _check(ledger, _console(tmp_path / "b.json", 6.0, end="2026-10-22"))


def test_headroom_on_billed_basis(tmp_path):
    ledger = _old_repo(tmp_path)                                   # billed 6, upper 42, cap 50
    console = _console(tmp_path / "a.json", 6.0)
    rep = _check(ledger, console, proj=20.0)                       # over the upper-bound headroom (8), allowed
    assert rep["headroom_billed_usd"] == pytest.approx(44.0) and rep["headroom_upper_bound_usd"] == pytest.approx(8.0)
    with pytest.raises(g.ConsoleCheckRefused, match="exceeds the billed headroom"):
        _check(ledger, console, proj=45.0)


def test_template_or_invalid_file_refused(tmp_path):
    ledger = _old_repo(tmp_path)
    t = tmp_path / "t.json"
    t.write_text(json.dumps(g.TEMPLATE), encoding="utf-8")
    with pytest.raises(g.ConsoleCheckRefused, match="not valid"):
        _check(ledger, t)
    t.write_text("{not json", encoding="utf-8")
    with pytest.raises(g.ConsoleCheckRefused, match="not valid"):
        _check(ledger, t)


def test_stub_mode_is_never_gated(tmp_path):
    c = CloudClient(mode="stub", ledger_path=_old_repo(tmp_path))
    assert g.require_for_client(c, 1000.0, check_path=tmp_path / "missing.json") == \
        {"gated": False, "reason": "stub mode is never gated"}


def test_require_for_client_uses_the_client_cap_and_ledger(tmp_path):
    ledger = _old_repo(tmp_path)
    c = CloudClient(api_key="sk-fake", ledger_path=ledger, role_override="controller", completion_fn=_fake_fn,
                    spend_cap_usd=10.0)
    console = _console(tmp_path / "a.json", 6.0)
    rep = g.require_for_client(c, 3.0, 9.0, check_path=console, now=NOW)
    assert rep["cap_usd"] == 10.0 and rep["headroom_billed_usd"] == pytest.approx(4.0)
    assert "upper bound $9.0000" in g.format_report(rep)
    with pytest.raises(g.ConsoleCheckRefused):
        g.require_for_client(c, 5.0, check_path=console, now=NOW)



# -- per-job limit and operator approval (operator 2026-10-24) --------------------------------------------------

def _real(ledger, cap=50.0):
    return CloudClient(api_key="sk-fake", ledger_path=ledger, role_override="controller", completion_fn=_fake_fn,
                       spend_cap_usd=cap)


def _pre(client, console, est, approval=None, **kw):
    return g.pre_job(client, job_id="cloud_redo_test_v1", estimate_usd=est, estimate_basis="test",
                     projection_billed_usd=kw.pop("proj", est), approval=approval, check_path=console, now=NOW,
                     head="abc123", **kw)


def _approvals(ledger):
    return [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()
            if json.loads(x).get("record") == "operator_approval"]


def test_over_two_dollars_refused_without_approval_and_nothing_written(tmp_path):
    ledger = _old_repo(tmp_path)
    before = ledger.read_text(encoding="utf-8")
    with pytest.raises(g.JobApprovalRefused) as e:
        _pre(_real(ledger), tmp_path / "missing.json", 2.01)          # refused before the Console check
    m = str(e.value)
    assert "above the USD 2.00 per-job limit" in m and "$2.0100" in m
    assert '--operator-approval "approve cloud_redo_test_v1 up to USD 2.01: <reason>"' in m
    assert ledger.read_text(encoding="utf-8") == before


def test_at_or_under_two_dollars_needs_no_approval(tmp_path):
    ledger = _old_repo(tmp_path)
    rep = _pre(_real(ledger), _console(tmp_path / "a.json", 6.0), 2.0)
    assert rep["approval"] is None and _approvals(ledger) == []


def test_valid_approval_is_recorded_once_before_any_call(tmp_path):
    ledger = _old_repo(tmp_path)
    text = "approve cloud_redo_test_v1 up to USD 5.00: rerun of the 28 rollback tasks"
    rep = _pre(_real(ledger), _console(tmp_path / "a.json", 6.0), 4.5, approval=text)
    rows = _approvals(ledger)
    assert len(rows) == 1 and rows[0] == rep["approval"]
    r = rows[0]
    assert (r["approval_text"], r["approved_usd"], r["job_id"], r["git_head"], r["estimate_usd"]) == \
        (text, 5.0, "cloud_redo_test_v1", "abc123", 4.5)
    assert r["ts_utc"] == NOW.isoformat() and r["source"] == "console_gate" and "cost_usd" not in r
    # the approval row is not spend: totals unchanged
    c = CloudClient(mode="stub", ledger_path=ledger)
    assert c.running_total_usd == pytest.approx(6.0) and c.running_total_usd_upper_bound == pytest.approx(42.0)
    assert "operator approval recorded in the ledger: USD 5.00" in g.format_report(rep)


@pytest.mark.parametrize("text,why", [
    ("approve some_other_job up to USD 9.00", "does not name the job id"),
    ("approve cloud_redo_test_v1 up to USD 3.00", "below the estimate"),
    ("approve cloud_redo_test_v1 for 9 dollars", "exactly one amount"),
    ("approve cloud_redo_test_v1 up to USD 5 or USD 9", "exactly one amount"),
])
def test_invalid_approval_refused(tmp_path, text, why):
    ledger = _old_repo(tmp_path)
    with pytest.raises(g.JobApprovalRefused, match=why):
        _pre(_real(ledger), _console(tmp_path / "a.json", 6.0), 4.5, approval=text)
    assert _approvals(ledger) == []


def test_approval_never_skips_the_console_check_and_writes_nothing_on_refusal(tmp_path):
    ledger = _old_repo(tmp_path)
    text = "approve cloud_redo_test_v1 up to USD 5.00"
    with pytest.raises(g.ConsoleCheckRefused, match="does not exist"):
        _pre(_real(ledger), tmp_path / "missing.json", 4.5, approval=text)
    with pytest.raises(g.ConsoleCheckRefused, match="disagree"):
        _pre(_real(ledger), _console(tmp_path / "b.json", 20.0), 4.5, approval=text)
    assert _approvals(ledger) == []


def test_approval_never_raises_the_cap(tmp_path):
    ledger = _old_repo(tmp_path)                                         # billed 6.0
    text = "approve cloud_redo_test_v1 up to USD 40.00"
    with pytest.raises(g.ConsoleCheckRefused, match="exceeds the billed headroom"):
        _pre(_real(ledger, cap=10.0), _console(tmp_path / "a.json", 6.0), 40.0, approval=text)
    assert _approvals(ledger) == []


def test_approval_is_never_read_from_the_environment(tmp_path, monkeypatch):
    ledger = _old_repo(tmp_path)
    monkeypatch.setenv("APU_OPERATOR_APPROVAL", "approve cloud_redo_test_v1 up to USD 9.00")
    monkeypatch.setenv("OPERATOR_APPROVAL", "approve cloud_redo_test_v1 up to USD 9.00")
    with pytest.raises(g.JobApprovalRefused):
        _pre(_real(ledger), _console(tmp_path / "a.json", 6.0), 4.5)


def test_stub_mode_pre_job_is_never_gated(tmp_path):
    c = CloudClient(mode="stub", ledger_path=_old_repo(tmp_path))
    assert _pre(c, tmp_path / "missing.json", 1000.0)["gated"] is False


# -- main + demo streams, counted once across checkouts -----------------------------------------------------------

def _fake_checkout(root, gitfile_to=None):
    root.mkdir(parents=True, exist_ok=True)
    if gitfile_to is not None:
        (root / ".git").write_text(f"gitdir: {gitfile_to}\n", encoding="utf-8")
    return root


def test_main_and_demo_streams_counted_once_whichever_checkout_runs(tmp_path):
    main = tmp_path / "main"
    common = main / ".git"
    (common / "worktrees" / "wt").mkdir(parents=True)
    wt = tmp_path / "wt"
    _fake_checkout(wt, gitfile_to=common / "worktrees" / "wt")
    (common / "worktrees" / "wt" / "commondir").write_text("../..\n", encoding="utf-8")
    (common / "worktrees" / "wt" / "gitdir").write_text(str(wt / ".git") + "\n", encoding="utf-8")
    t = T0
    main_rows = [{"timestamp": _iso(t + timedelta(minutes=i)), "model_id": MODEL, "cost_usd": 1.0,
                  "cost_usd_upper_bound": 2.0} for i in range(3)]
    _jsonl(main / "results" / "cloud_ledger.jsonl", main_rows)
    _jsonl(wt / "results" / "cloud_ledger.jsonl", main_rows[:2])         # a stale committed copy: adds nothing
    _jsonl(wt / "demo" / "ledger" / "cloud_ledger_demo.jsonl",
           [{"timestamp": _iso(t + timedelta(minutes=5)), "model_id": MODEL, "cost_usd": 0.25,
             "cost_usd_upper_bound": 0.5}])
    for ledger in (main / "results" / "cloud_ledger.jsonl", wt / "results" / "cloud_ledger.jsonl",
                   wt / "demo" / "ledger" / "cloud_ledger_demo.jsonl"):
        side = g.console_side_spend(ledger)
        assert side["billed_usd"] == pytest.approx(3.25), ledger
        assert side["streams"]["main"]["ledger_calls"] == 3 and side["streams"]["demo"]["ledger_calls"] == 1
    assert g.main_checkout(wt) == main.resolve()


def test_demo_ledger_cap_and_session_limit(tmp_path):
    """The demo lane's gate: its own ledger and cap, limit USD 3 per session; the Console compares main + demo."""
    _jsonl(tmp_path / "results" / "cloud_ledger.jsonl",
           [{"timestamp": _iso(T0), "model_id": MODEL, "cost_usd": 6.0, "cost_usd_upper_bound": 42.0}])
    demo = tmp_path / "demo" / "ledger" / "cloud_ledger_demo.jsonl"
    _jsonl(demo, [{"timestamp": _iso(T0 + timedelta(minutes=1)), "model_id": MODEL, "cost_usd": 1.0,
                   "cost_usd_upper_bound": 2.0}])
    c = _real(demo, cap=10.0)
    console = _console(tmp_path / "a.json", 7.0)                         # main 6 + demo 1
    kw = dict(job_id="demo_live_session", estimate_basis="session limit", check_path=console, now=NOW, head="h",
              limit_usd=3.0)
    rep = g.pre_job(c, estimate_usd=3.0, projection_billed_usd=3.0, **kw)
    assert rep["spent_billed_usd"] == pytest.approx(1.0) and rep["headroom_billed_usd"] == pytest.approx(9.0)
    assert rep["ledger_billed_in_range_usd"] == pytest.approx(7.0)
    with pytest.raises(g.JobApprovalRefused, match="USD 3.00 per-job limit"):
        g.pre_job(c, estimate_usd=4.0, projection_billed_usd=4.0, **kw)
    rep = g.pre_job(c, estimate_usd=4.0, projection_billed_usd=4.0,
                    approval="approve demo_live_session up to USD 4.00: rehearsal", **kw)
    assert rep["approval"]["approved_usd"] == 4.0 and _approvals(demo)[0]["job_id"] == "demo_live_session"
