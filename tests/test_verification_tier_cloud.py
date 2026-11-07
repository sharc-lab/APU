"""Verification tier Run V1 live script (scripts/verification_tier_cloud.py): fakes only. No network, no key, no real
provider call; a real-mode CloudClient is built only with a fake completion function and a test ledger."""
from __future__ import annotations

import ast
import functools
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "harness", REPO / "scripts", REPO / "src", REPO / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import verification_tier_cloud as V  # noqa: E402
from cloud import client as cc  # noqa: E402
from cloud import console_gate  # noqa: E402
from src.dse import verification_tier as VT  # noqa: E402

APPROVAL = "approve verification_tier_run_v1 up to USD 2.00: test"
EST = {"point": 0.1, "high": 0.15, "upper": 0.2, "n_calls": None, "source": "test"}
REPLY_ACCEPT = json.dumps({"verdict": "accept", "reason": "inventory/SKU-1 set as asked"})
REPLY_REJECT = json.dumps({"verdict": "reject", "reason": "inventory/SKU-1 unchanged"})


def _calls(n=6, wrong=(1,), big=(), seeds=None):
    out = []
    for i in range(n):
        user = "Request:\nChange SKU-1.\n" + ("x" * 50_000 if i in big else "")
        out.append({"config": "B3 v2", "tier": "ollama_default", "seed": (seeds or {}).get(i, 20260901),
                    "turn_idx": i + 1, "task_id": f"t{i}",
                    "payload": {"system": VT.VERIFIER_INSTRUCTION + "\n\nrules", "user": user},
                    "local_ok": i not in wrong})
    return out


def _est(calls):
    return dict(EST, n_calls=len(calls) + V.WARMUP_CALLS)


class Fake:
    """A provider stand-in: records every request; replies in turn from `replies` (str, Exception, or dict usage)."""

    def __init__(self, replies=None, usage=None):
        self.sent, self.replies, self.usage = [], list(replies or []), usage

    def __call__(self, **kw):
        self.sent.append(kw)
        r = self.replies.pop(0) if self.replies else REPLY_ACCEPT
        if isinstance(r, Exception):
            raise r
        return {"choices": [{"message": {"content": r}, "finish_reason": "stop"}],
                "usage": self.usage or {"prompt_tokens": 1200, "completion_tokens": 30}}


def _client(tmp_path, fake, cap=2.0):
    return cc.CloudClient(mode="real", api_key="test-not-a-key", role_override="controller", completion_fn=fake,
                          ledger_path=tmp_path / "ledger.jsonl", spend_cap_usd=cap)


def _run(tmp_path, calls, fake, cap=2.0, **kw):
    args = dict(out=tmp_path / "out.jsonl", live=True, client=_client(tmp_path, fake, cap), job_cap=2.0,
                ledger=tmp_path / "ledger.jsonl", approval=APPROVAL, calls=calls, estimate=_est(calls),
                gate_fn=lambda *a, **k: {"gated": True, "test": True}, emit_print=lambda *a: None)
    args.update(kw)
    return V.run(**args)


def _tasks(rows):
    return [r for r in rows if r["record"] == V.RECORD]


# ── happy path and scoring ─────────────────────────────────────────────────────────────────────────

def test_live_with_fakes_writes_start_warmup_calls_end_and_scores(tmp_path):
    calls = _calls(6, wrong=(1, 4), seeds={i: V.HELDOUT_SEED for i in range(6)})   # no futility stop here
    replies = [REPLY_ACCEPT, REPLY_ACCEPT, REPLY_REJECT, REPLY_REJECT, REPLY_ACCEPT, "not json", REPLY_ACCEPT]
    fake = Fake(replies)
    rows = _run(tmp_path, calls, fake)
    assert [r["record"] for r in rows][:2] == ["run_start", "vt_warmup"] and rows[-1]["record"] == "run_end"
    assert rows[1]["excluded_from_stats"] is True
    assert len(fake.sent) == len(calls) + 1                      # 1 warm-up + one per task
    for kw in fake.sent:
        assert kw["model_id"] == VT.VERIFIER_MODEL and kw["max_tokens"] == VT.MAX_TOKENS_VERDICT
        assert kw["thinking"] == {"type": "disabled"}
        assert kw["messages"][0]["role"] == "system"
        assert VT.REDERIVE_INSTRUCTION not in json.dumps(kw["messages"])
    t = _tasks(rows)
    assert all(x["sent"] and x["usd_billed"] > 0 and x["latency_s"] >= 0 and x["tokens"]["output_tokens"] == 30
               for x in t)
    s = rows[-1]["summary"]
    # replies after the warm-up: task 0 accept, 1 (wrong) reject = caught, 2 reject = false escalation, 3 accept,
    # 4 (wrong) unparseable = counts as a reject, so caught, 5 accept
    assert s["positives"] == 2 and s["negatives"] == 4
    assert s["caught"] == 2 and s["false_esc"] == 1 and s["unparseable"] == 1 and s["no_verdict"] == 0
    assert s["usd_billed_per_task_mean"] == pytest.approx(s["usd_billed_total"] / 6)
    assert s["usd_billed_incl_warmup"] > s["usd_billed_total"]
    assert V.score(tmp_path / "out.jsonl")["false_esc"] == 1
    text = V.score_text(V.score(tmp_path / "out.jsonl"))
    assert "caught 2/2" in text and "false escalations 1/4" in text and "%" not in text.split("REPLAY")[0]
    assert "REPLAY of the q4_0 28928 operating point" in text and "target >= 88/90" in text


def test_parse_verdict_is_strict():
    assert V.parse_verdict(REPLY_REJECT)[0] == "reject"
    assert V.parse_verdict("  " + REPLY_ACCEPT + "\n")[0] == "accept"
    for bad in ("", "accept", json.dumps({"verdict": "maybe"}), json.dumps([{"verdict": "accept"}]), None,
                "```json\n```json\n" + REPLY_ACCEPT + "\n```\n```", "Sure:\n```json\n" + REPLY_ACCEPT + "\n```",
                "```python\n" + REPLY_ACCEPT + "\n```", "```" + REPLY_ACCEPT + "```"):
        assert V.parse_verdict(bad) == (None, None), bad


def test_parse_verdict_strips_one_surrounding_fence():
    """Amendment 2026-11-07: one surrounding ``` or ```json fence is stripped before parsing."""
    assert V.parse_verdict("```json\n" + REPLY_REJECT + "\n```")[0] == "reject"
    assert V.parse_verdict("```\n" + REPLY_ACCEPT + "\n```\n")[0] == "accept"


class _HTTPError(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.status_code = code


class APITimeoutError(Exception):
    pass


def test_retryable_errors_are_timeout_429_and_5xx_only():
    assert V.retryable_error(_HTTPError(429)) and V.retryable_error(_HTTPError(503))
    assert V.retryable_error(APITimeoutError("t"))
    for e in (_HTTPError(400), _HTTPError(401), RuntimeError("boom"), ValueError("x")):
        assert not V.retryable_error(e)


def test_one_retry_on_provider_error_then_counts_every_failed_send(tmp_path):
    held = {i: V.HELDOUT_SEED for i in range(4)}
    # warm-up ok; task 0: 429 then accept (retried); task 1: 503 twice (gives up); task 2: 400 (no retry); task 3 ok
    fake = Fake([REPLY_ACCEPT, _HTTPError(429), REPLY_ACCEPT, _HTTPError(503), _HTTPError(503), _HTTPError(400),
                 REPLY_ACCEPT])
    rows = _run(tmp_path, _calls(4, wrong=(), seeds=held), fake)
    t = _tasks(rows)
    assert len(fake.sent) == 7
    assert t[0]["outcome"] == "sent" and len(t[0]["provider_errors"]) == 1 and t[0]["provider_errors"][0]["retryable"]
    assert t[1]["outcome"] == "provider_error" and len(t[1]["provider_errors"]) == 2
    assert t[2]["outcome"] == "provider_error" and len(t[2]["provider_errors"]) == 1
    assert not t[2]["provider_errors"][0]["retryable"] and t[3]["outcome"] == "sent"
    # failed sends 1 + 2 + 1 = 4: under the stop rule of 5
    assert rows[-1]["stopped"] is None and rows[-1]["summary"]["no_verdict"] == 2


def test_no_retry_on_unparseable_reply(tmp_path):
    held = {i: V.HELDOUT_SEED for i in range(2)}
    fake = Fake([REPLY_ACCEPT, "not json", REPLY_ACCEPT])
    rows = _run(tmp_path, _calls(2, wrong=(), seeds=held), fake)
    assert len(fake.sent) == 3 and _tasks(rows)[0]["parse_ok"] is False


def test_retry_errors_count_toward_the_stop_rule(tmp_path):
    held = {i: V.HELDOUT_SEED for i in range(5)}
    rows = _run(tmp_path, _calls(5, seeds=held), Fake([REPLY_ACCEPT] + [_HTTPError(500)] * 10))
    assert rows[-1]["stopped"] == "stop rule: 6 provider errors"     # task 2's second failed send passes 5


# ── refusals ───────────────────────────────────────────────────────────────────────────────────────

def test_refused_without_live(tmp_path, monkeypatch, capsys):
    """No --live: a dry run. The client is never used and nothing is written, approval or not."""
    calls = _calls(3)
    fake = Fake()
    rows = V.run(out=tmp_path / "out.jsonl", live=False, client=_client(tmp_path, fake), approval=APPROVAL,
                 calls=calls, estimate=_est(calls))
    assert len(rows) == 1 and rows[0]["record"] == "run_start" and not fake.sent
    assert not (tmp_path / "out.jsonl").exists()
    monkeypatch.setattr(V, "build_calls", lambda *a, **k: _calls(3))
    monkeypatch.setattr(V, "plan_estimate", lambda *a, **k: _est(_calls(3)))
    monkeypatch.setattr(V, "OUT", tmp_path / "out.jsonl")
    assert V.main([console_gate.APPROVAL_FLAG, APPROVAL]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "--live" in out and not (tmp_path / "out.jsonl").exists()


def test_live_refused_without_approval_key_console_file_or_8192_phase(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(V, "build_calls", lambda *a, **k: pytest.fail("payloads built on a refused live run"))
    out = str(tmp_path / "o.jsonl")
    assert V.main(["--live", "--out", out]) == 2                                   # no approval
    assert "operator's go" in capsys.readouterr().err
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert V.main(["--live", "--out", out, console_gate.APPROVAL_FLAG, APPROVAL]) == 2   # no key
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    monkeypatch.setattr(console_gate, "default_check_path", lambda: tmp_path / "missing_console_check.json")
    assert V.main(["--live", "--out", out, console_gate.APPROVAL_FLAG, APPROVAL]) == 2   # no console file
    err = capsys.readouterr().err
    assert "Console check file" in err and "nothing was sent" in err
    (tmp_path / "console.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(console_gate, "default_check_path", lambda: tmp_path / "console.json")
    monkeypatch.setattr(V, "REPO", tmp_path)                                         # no 8192 phase file here
    assert V.main(["--live", "--out", out, console_gate.APPROVAL_FLAG, APPROVAL]) == 2
    assert "8192 cloud phase" in capsys.readouterr().err
    assert not Path(out).exists()


def test_live_refused_by_the_real_console_gate_without_console_file(tmp_path):
    calls = _calls(3)
    fake = Fake()
    gate = functools.partial(console_gate.pre_job, check_path=tmp_path / "missing.json")
    with pytest.raises(console_gate.ConsoleCheckRefused):
        _run(tmp_path, calls, fake, gate_fn=gate)
    assert not fake.sent and not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize("kw,msg", [({"approval": None}, "operator's go"), ({"approval": " "}, "operator's go"),
                                    ({"job_cap": 2.5}, "job cap"), ({"job_cap": 0}, "job cap")])
def test_run_refusals(tmp_path, kw, msg):
    fake = Fake()
    with pytest.raises(SystemExit, match=msg):
        _run(tmp_path, _calls(3), fake, **kw)
    assert not fake.sent and not (tmp_path / "out.jsonl").exists()


def test_run_refuses_stub_client_and_existing_out(tmp_path):
    class Stub:
        stub_mode = True
    with pytest.raises(SystemExit, match="real-mode"):
        _run(tmp_path, _calls(3), Fake(), client=Stub())
    (tmp_path / "out.jsonl").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="exists"):
        _run(tmp_path, _calls(3), Fake())


def test_main_refuses_job_cap_above_2(capsys):
    assert V.main(["--job-cap", "2.01"]) == 2
    assert "job cap" in capsys.readouterr().err


# ── caps ───────────────────────────────────────────────────────────────────────────────────────────

def test_per_task_cap_refuses_before_sending(tmp_path):
    """A payload whose worst case (bytes as tokens at the highest input rate + max_tokens output) is over USD 0.005
    is refused before any request; the rest of the job runs."""
    calls = _calls(4, big=(2,))
    prices = V.load_prices()
    assert V.worst_case_usd(calls[2]["payload"], prices) > V.TASK_CAP_USD
    assert V.worst_case_usd(calls[0]["payload"], prices) < V.TASK_CAP_USD
    fake = Fake()
    rows = _run(tmp_path, calls, fake)
    t = _tasks(rows)
    assert t[2]["outcome"] == "refused_task_cap" and not t[2]["sent"]
    assert len(fake.sent) == 1 + 3                               # warm-up + the three other tasks
    assert all(len(json.dumps(k["messages"])) < 40_000 for k in fake.sent)
    assert rows[-1]["stopped"] is None and rows[-1]["summary"]["no_verdict"] == 1
    assert not rows[-1]["summary"]["complete"]


def test_billed_cost_over_task_cap_stops_the_job_and_is_recorded(tmp_path):
    calls = _calls(4)
    fake = Fake(usage={"prompt_tokens": 1200, "completion_tokens": 20_000})   # billed far above USD 0.005
    rows = _run(tmp_path, calls, fake)
    assert rows[1]["outcome"] == "task_cap_exceeded_billed"
    assert rows[-1]["stopped"].startswith("per-task cap")
    assert all(t["outcome"] == "not_run_stopped" for t in _tasks(rows))
    assert len(fake.sent) == 1


def test_job_cap_stop(tmp_path):
    """A client cap below one call: the client refuses before sending; every task is recorded not_run_job_cap."""
    calls = _calls(5)
    fake = Fake()
    rows = _run(tmp_path, calls, fake, cap=1e-6)
    assert not fake.sent and rows[-1]["stopped"].startswith("job cap")
    assert rows[1]["outcome"] == "not_run_job_cap"
    assert all(t["outcome"] == "not_run_job_cap" for t in _tasks(rows)) and len(_tasks(rows)) == 5


def test_job_cap_mid_run_stops_cleanly(tmp_path):
    calls = _calls(6)
    fake = Fake()
    probe = _client(tmp_path, Fake())
    one = probe.projected_cost_usd(model_id=VT.VERIFIER_MODEL, messages=VT.messages(calls[0]["payload"]),
                                   max_tokens=VT.MAX_TOKENS_VERDICT)
    rows = _run(tmp_path, calls, fake, cap=one * 2.5)
    outcomes = [t["outcome"] for t in _tasks(rows)]
    assert rows[-1]["stopped"].startswith("job cap") and "sent" in outcomes and outcomes[-1] == "not_run_job_cap"


def test_stop_after_five_unparseable_and_five_provider_errors(tmp_path):
    held = {i: V.HELDOUT_SEED for i in range(9)}       # held-out negatives: the futility rule does not apply
    rows = _run(tmp_path, _calls(9, seeds=held), Fake([REPLY_ACCEPT] + ["nope"] * 8))
    assert rows[-1]["stopped"].startswith("stop rule: 5 replies")
    assert sum(t["outcome"] == "not_run_stopped" for t in _tasks(rows)) == 4
    rows = _run(tmp_path, _calls(9), Fake([REPLY_ACCEPT] + [RuntimeError("boom")] * 8), out=tmp_path / "out2.jsonl")
    assert rows[-1]["stopped"].startswith("stop rule: 5 provider errors")


def test_futility_stops_on_development_false_escalations(tmp_path):
    seeds = {i: (20260905 if i >= 8 else 20260901) for i in range(10)}
    calls = _calls(10, wrong=(), seeds=seeds)         # 8 development negatives -> futility above 0.8, i.e. at 1
    rows = _run(tmp_path, calls, Fake([REPLY_ACCEPT, REPLY_REJECT]))
    assert rows[-1]["stopped"].startswith("futility")
    assert sum(t["sent"] for t in _tasks(rows)) == 1


# ── no re-derivation arm ───────────────────────────────────────────────────────────────────────────

def test_no_rederivation_arm(tmp_path, capsys):
    for arms in (("a", "c"), ("c",), ()):
        with pytest.raises(SystemExit, match="re-derivation"):
            V.check_arms(arms)
        with pytest.raises(SystemExit, match="re-derivation"):
            _run(tmp_path, _calls(2), Fake(), arms=arms)
    assert V.main(["--arm", "c"]) == 2 and "re-derivation" in capsys.readouterr().err
    tree = ast.parse((REPO / "scripts" / "verification_tier_cloud.py").read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
            {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"build_rederive_payload", "REDERIVE_INSTRUCTION", "MAX_TOKENS_REDERIVE", "payloads_for"}
    assert V.ARMS_ALLOWED == ("a",)


# ── dry run on the recorded files ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def built():
    return V.build_calls(REPO)


def test_operating_point_replay_reproduces_the_register_and_the_ceiling(built):
    """No verdicts: row operating-point-q4-28928-rolling-cache (85/90, 43.4%). A perfect verifier on the q4_0 28928
    turns: the CEILING's 90/90 (row verification-tier-ceiling)."""
    o = V.operating_point_replay([])
    assert (o["n"], o["correct"], o["correct_before"], o["no_reply"]) == (90, 85, 85, 81)
    assert round(o["pct_bill"], 1) == 43.4 == round(o["pct_bill_before"], 1)
    rows = [{"record": V.RECORD, "config": c["config"], "seed": c["seed"], "task_id": c["task_id"], "sent": True,
             "verdict": "accept" if c["local_ok"] else "reject", "usd_billed": 0.0}
            for c in built if c["config"] == "q4_0 28928"]
    o = V.operating_point_replay(rows)
    assert o["correct"] == 90 and o["rejected"] == o["rejected_fixed"] == 5 and o["rejected_were_correct"] == 0
    assert o["pct_bill"] > o["pct_bill_before"]


def test_dry_run_call_count_matches_the_preregistration(built, capsys):
    import verification_tier_plan as P
    est = V.plan_estimate(REPO)
    start = V.run(live=False, calls=built, estimate=est)[0]
    assert start["n_tasks"] == 260 and start["n_calls"] == 261 == est["n_calls"] == V.TARGETS["n_calls"]
    assert start["positives"] == 16 and start["negatives"] == 244 and start["matches_estimate_calls"]
    assert start["max_worst_case_usd"] < V.TASK_CAP_USD
    assert [V._key(c) for c in built] == [V._key(c) for c in P.plan(REPO)["calls"]]   # same run order
    text = V.dry_run_text(start)
    assert "261 calls" in text and "USD 0.1078" in text and "USD 0.1962" in text and V.live_command() in text


def test_payloads_equal_the_plan_modules_arm_a(built):
    import verification_tier_plan as P
    turns = P.accepted_turns(REPO)
    for i in (0, 57, 180, len(turns) - 1):
        t = turns[i]
        assert P.payloads_for(t["es"], t["prior"], t["row"])["a"] == built[i]["payload"]
    assert all(c["payload"]["system"].startswith(VT.VERIFIER_INSTRUCTION) for c in built)
    assert not any("Also, recall the exact value" in c["payload"]["user"] for c in built)


# ── register row ───────────────────────────────────────────────────────────────────────────────────

def _nr():
    spec = importlib.util.spec_from_file_location("numbers_register_vtc", REPO / "analysis" / "numbers_register.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_register_row_pending_without_data(tmp_path):
    nr = _nr()
    e = {x["claim_id"]: x for x in nr.NUMBER_ENTRIES}["verification-tier-run-v1"]
    assert e["pending_ok"] and e["require_run_end"] == [nr.VT_RUN_V1_FILE] == e["data_files"]
    assert nr._pending(e, tmp_path) == [nr.VT_RUN_V1_FILE]
    f = tmp_path / nr.VT_RUN_V1_FILE
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"record": "run_start"}) + "\n", encoding="utf-8")
    assert nr._pending(e, tmp_path) == [nr.VT_RUN_V1_FILE + " (partial: no run_end)"]
    with pytest.raises(RuntimeError, match="run_end"):
        e["compute"](tmp_path)


def test_register_row_computes_from_a_finished_file(tmp_path):
    nr = _nr()
    e = {x["claim_id"]: x for x in nr.NUMBER_ENTRIES}["verification-tier-run-v1"]
    out = tmp_path / nr.VT_RUN_V1_FILE
    out.parent.mkdir(parents=True)
    _run(tmp_path, _calls(4, wrong=(1,)), Fake([REPLY_ACCEPT, REPLY_ACCEPT, REPLY_REJECT]), out=out)
    assert nr._pending(e, tmp_path) == []
    v = e["compute"](tmp_path)
    assert "caught 1/1" in v["value"] and "false escalations 0/3" in v["value"] and v["n"] == "4 tasks + 1 warm-up"
