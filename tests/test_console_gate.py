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


def test_fresh_passes_stale_refused_and_window_is_configurable(tmp_path, monkeypatch):
    ledger = _old_repo(tmp_path)
    fresh = _console(tmp_path / "fresh.json", 6.0)
    rep = _check(ledger, fresh)
    assert rep["gated"] and rep["ledger_billed_in_range_usd"] == pytest.approx(6.0)
    assert rep["ledger_upper_bound_in_range_usd"] == pytest.approx(42.0)
    stale = _console(tmp_path / "stale.json", 6.0, read_at=NOW - timedelta(hours=25), start="2026-10-22",
                     end="2026-10-22")
    with pytest.raises(g.ConsoleCheckRefused, match="older than the 24 h freshness window"):
        _check(ledger, stale)
    assert _check(ledger, stale, freshness_h=48)["gated"]
    monkeypatch.setenv(g.FRESHNESS_ENV, "30")
    assert _check(ledger, stale)["freshness_h"] == 30.0


def test_within_tolerance_passes_outside_refused(tmp_path):
    ledger = _old_repo(tmp_path)                                   # billed 6.0 in range
    assert _check(ledger, _console(tmp_path / "a.json", 6.45))["tolerance_usd"] == pytest.approx(0.50)
    with pytest.raises(g.ConsoleCheckRefused, match="ledger is LOWER than the Console"):
        _check(ledger, _console(tmp_path / "b.json", 6.60))
    with pytest.raises(g.ConsoleCheckRefused, match="ledger is HIGHER than the Console"):
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
    assert rep["over_ask_first_usd"]
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

