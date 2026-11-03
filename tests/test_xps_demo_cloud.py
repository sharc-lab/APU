"""XPS demo kit, cloud step (scripts/xps_demo_cloud.py): fakes only. No network, no key, no real client call."""
import argparse
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "harness", REPO / "scripts", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import xps_demo_cloud as XC  # noqa: E402
import xps_demo_kit as K  # noqa: E402

BASIS = {"model": "claude-sonnet-5-5", "all_cloud_usd_task": 0.01, "all_cloud_usd_task_recorded": 0.011,
         "all_cloud_n": 90, "all_cloud_correct": 90, "redo_usd_task": 0.05, "redo_tier": "ollama_default",
         "redo_tier_means": {"ollama_default": 0.05}, "redo_model": "claude-sonnet-5-5", "files": []}


class RealishClient:
    """Stands in for a real-mode CloudClient: stub_mode False, never sends anything (the fakes below never call it)."""
    stub_mode = False
    running_total_usd = 0.0

    def __init__(self, ledger):
        self.ledger_path = Path(ledger)

    def prepare(self, **kw):   # pragma: no cover - a test failing here means a real call was attempted
        raise AssertionError("no cloud call may be prepared in tests")


def _local(tmp_path, arms="b,c", window=3500, max_turns=6, name="local.jsonl"):
    ns = argparse.Namespace(arms=arms, seeds=[K.SEEDS[0]], max_turns=max_turns, runtime="ollama", base_url="fake",
                            model="qwen3:4b", allow_nondefault=None, pipeline_window=None,
                            out=str(tmp_path / name), note=None, runtime_exe=None, runtime_config_file=None,
                            declared_default_window=None)
    cfg = {"runtime": "ollama", "overrides": [], "unverified": [], "default_window": None}
    return Path(K.run(ns, runtime=K.FakeRuntime(window=window), cfg=cfg, log=lambda m: None)["out"])


def test_split_parts_keeps_each_part_under_the_cap():
    parts = XC.split_parts(list(range(90)), 0.04065, 2.00)
    assert [len(p) for p in parts] == [49, 41]
    assert all(len(p) * 0.04065 <= 2.00 for p in parts)
    assert XC.split_parts([], 0.05) == []


def test_cloud_tasks_per_arm(tmp_path):
    rows = XC.load_rows([_local(tmp_path)])
    b = XC.cloud_tasks(rows, "b")
    assert b and all(t["reason"] == "over_window" for t in b)
    c = XC.cloud_tasks(rows, "c")
    reasons = {t["reason"] for t in c}
    assert "route_at_start" in reasons and "post_run_check" in reasons
    assert [(t["seed"], t["turn_idx"]) for t in c] == sorted((t["seed"], t["turn_idx"]) for t in c)
    rc = XC.cloud_tasks(rows, "c", XC.recheck_fn(rows))
    assert all(t.get("decision_matches_xps", True) for t in rc)
    assert len(XC.cloud_tasks(rows, "d")) == 3


def test_plan_estimates(tmp_path):
    rows = XC.load_rows([_local(tmp_path)])
    p = XC.plan(rows, BASIS)
    assert p["d"]["estimate_usd"] == pytest.approx(3 * 30 * 0.01)
    assert p["b"]["estimate_usd"] == pytest.approx(p["b"]["n_cloud_units"] * 0.05)
    assert all(x["estimate_usd"] <= XC.JOB_CAP_MAX_USD for a in "bcd" for x in p[a]["parts"])


def _run(tmp_path, local, **kw):
    gate_calls = []

    def gate(client, **g):
        gate_calls.append(g)
        return {"gated": True, "fake": True}

    def live_turn(lane, model, es, built, rates, max_tokens, thinking):
        assert built["messages"][0]["role"] == "system" and built["messages"][-1]["role"] == "user"
        return {"usd": 0.02, "seconds": 3.0, "cloud_end_state_ok": True, "n_calls": 2}
    args = dict(arm="b", part=1, local_paths=[local], out=tmp_path / "cloud_b.jsonl",
                client=RealishClient(tmp_path / "ledger.jsonl"), approval="approve xps_demo_cloud_b up to USD 2.00: t",
                ledger=tmp_path / "ledger.jsonl", gate_fn=gate, basis=BASIS, log=lambda m: None, live_turn=live_turn)
    args.update(kw)
    return XC.run(**args), gate_calls


def test_run_b_happy_path_with_fakes(tmp_path):
    local = _local(tmp_path, arms="b")
    rows, gates = _run(tmp_path, local)
    tasks = [r for r in rows if r["record"] == XC.RECORD]
    assert rows[0]["record"] == "run_start" and rows[-1]["record"] == "run_end" and rows[-1]["stopped"] is None
    assert tasks and all(t["cloud"]["outcome"] == "cloud" and t["cloud"]["task_cap_usd"] == XC.TASK_CAP_USD
                         for t in tasks)
    assert len(gates) == 1 and gates[0]["estimate_usd"] == pytest.approx(len(tasks) * 0.05)
    assert gates[0]["approval"].startswith("approve xps_demo_cloud_b")
    s = XC.summarize(XC.load_rows([local]), rows)
    sb = s["standard_hybrid"]
    assert sb["correct_of"] == 90 and sb["jobs_recorded"] == 6 and not sb["complete"]
    assert sb["usd_total"] == pytest.approx(0.02 * len(tasks))


@pytest.mark.parametrize("kw,msg", [
    ({"approval": None}, "approval"),
    ({"approval": "  "}, "approval"),
    ({"job_cap": 2.5}, "above"),
    ({"arm": "a"}, "--arm"),
    ({"part": 3}, "part"),
])
def test_run_refusals(tmp_path, kw, msg):
    local = _local(tmp_path, arms="b")
    with pytest.raises(SystemExit, match=msg):
        _run(tmp_path, local, **kw)
    assert not (tmp_path / "cloud_b.jsonl").exists()


def test_run_refuses_stub_client_incomplete_local_and_existing_out(tmp_path):
    local = _local(tmp_path, arms="b")

    class Stub:
        stub_mode = True
    with pytest.raises(SystemExit, match="real-mode"):
        _run(tmp_path, local, client=Stub())
    partial = tmp_path / "partial.jsonl"
    partial.write_text("\n".join(x for x in local.read_text(encoding="utf-8").splitlines()
                                 if '"run_end"' not in x) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no run_end"):
        _run(tmp_path, partial)
    (tmp_path / "cloud_b.jsonl").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="exists"):
        _run(tmp_path, local)


def test_run_refuses_when_one_part_is_over_the_cap(tmp_path):
    local = _local(tmp_path, arms="b")
    big = dict(BASIS, redo_usd_task=2.5)          # one task alone is over USD 2
    with pytest.raises(SystemExit, match="above the job cap"):
        _run(tmp_path, local, basis=big)


def test_run_gate_refusal_sends_nothing(tmp_path):
    from cloud import console_gate
    local = _local(tmp_path, arms="b")

    def gate(client, **g):
        raise console_gate.ConsoleCheckRefused("refused (console check): stale")
    with pytest.raises(console_gate.ConsoleCheckRefused):
        _run(tmp_path, local, gate_fn=gate)
    assert not (tmp_path / "cloud_b.jsonl").exists()


def test_run_c_rolls_back_escalated_tasks(tmp_path):
    local = _local(tmp_path, arms="c")
    rows, _ = _run(tmp_path, local, arm="c", out=tmp_path / "cloud_c.jsonl",
                   approval="approve xps_demo_cloud_c up to USD 2.00: t")
    tasks = [r for r in rows if r["record"] == XC.RECORD]
    esc = [t for t in tasks if t["reason"] == "post_run_check"]
    assert esc and all("undone_calls" in t for t in tasks)


def test_run_d_uses_the_all_cloud_session(tmp_path):
    local = _local(tmp_path, arms="b")
    seen = []

    def run_session(rt, seed, emit, log, spend=None):
        seen.append(seed)
        for i in range(30):
            emit({"record": "r2a_turn", "arm_id": "cloud_all", "item_kind": "task", "seed": seed,
                  "task_id": f"t{i}", "end_state_ok": i != 0, "usd": 0.01, "turn_wall_s": 4.0})
    rows, gates = _run(tmp_path, local, arm="d", out=tmp_path / "cloud_d.jsonl", run_session=run_session,
                       approval="approve xps_demo_cloud_d up to USD 2.00: t")
    assert seen == list(K.SEEDS) and gates[0]["estimate_usd"] == pytest.approx(0.9)
    s = XC.summarize([], rows)["all_cloud"]
    assert s["correct"] == 87 and s["complete"] and s["usd_per_job_mean"] == pytest.approx(0.01)
    assert s["latency_s"]["median"] == 4.0


def test_summarize_counts_not_run_cloud_tasks_as_wrong():
    local = [
        {"record": "r2a_turn", "arm": "a", "item_kind": "task", "seed": 1, "task_id": "t1", "turn_idx": 1,
         "end_state_ok": True, "turn_wall_s": 10.0},
        {"record": "r2a_turn", "arm": "a", "item_kind": "task", "seed": 1, "task_id": "t2", "turn_idx": 2,
         "end_state_ok": False, "turn_wall_s": 20.0},
        {"record": K.DEFER_RECORD, "arm": "b", "item_kind": "task", "seed": 1, "task_id": "t1", "turn_idx": 1,
         "reason": "over_window", "local_seconds": 0.5},
        {"record": K.DEFER_RECORD, "arm": "b", "item_kind": "task", "seed": 1, "task_id": "t2", "turn_idx": 2,
         "reason": "hard_error", "local_seconds": 1.0},
        {"record": "r2a_turn", "arm": "c", "item_kind": "task", "seed": 1, "task_id": "t1", "turn_idx": 1,
         "end_state_ok": False, "turn_wall_s": 5.0, "local_seconds_with_check": 5.001,
         "prc_check": {"decision": "escalate"}},
    ]
    cloud = [
        {"record": XC.RECORD, "arm": "b", "seed": 1, "turn_idx": 1, "cloud": {"outcome": "cloud", "usd": 0.03,
                                                                             "seconds": 2.0,
                                                                             "cloud_end_state_ok": True}},
        {"record": XC.RECORD, "arm": "b", "seed": 1, "turn_idx": 2, "cloud": {"outcome": "not_run_job_cap"}},
        {"record": XC.RECORD, "arm": "c", "seed": 1, "turn_idx": 1, "cloud": {"outcome": "cloud", "usd": 0.04,
                                                                             "seconds": 3.0,
                                                                             "cloud_end_state_ok": True}},
    ]
    s = XC.summarize(local, cloud, n_jobs=2)
    a, b, c = s["local_default"], s["standard_hybrid"], s["full_pipeline"]
    assert (a["correct"], a["usd_per_job_mean"], a["latency_s"]["median"]) == (1, 0.0, 15.0)
    assert b["correct"] == 1 and b["not_run"] == 1 and not b["complete"]
    assert b["usd_per_job_mean"] == pytest.approx(0.015) and b["latency_s"]["mean"] == pytest.approx(2.5)
    assert c["correct"] == 1 and c["missing"] == 1 and c["latency_s"]["mean"] == pytest.approx(8.001)
    assert "| local_default | 1/2 |" in XC.format_table(s)


def test_estimate_basis_from_committed_files():
    b = XC.estimate_basis()
    assert b["model"] == "claude-sonnet-5-5" and b["all_cloud_n"] == 90
    assert 0 < b["all_cloud_usd_task"] < b["redo_usd_task"]
    assert b["redo_tier"] in b["redo_tier_means"]


def test_main_refuses_without_live_or_key(monkeypatch, tmp_path, capsys):
    local = _local(tmp_path, arms="b")
    assert XC.main(["run", "--arm", "b", "--local", str(local), "--out", str(tmp_path / "o.jsonl")]) == 2
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert XC.main(["run", "--arm", "b", "--local", str(local), "--out", str(tmp_path / "o.jsonl"), "--live"]) == 2
    assert "nothing was sent" in capsys.readouterr().err
    assert not (tmp_path / "o.jsonl").exists()
    out = tmp_path / "sum.json"
    assert XC.main(["summarize", "--local", str(local), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["standard_hybrid"]["correct_of"] == 90


def test_expand_wildcards_for_powershell(tmp_path):
    for n in ("xps_demo_local_v1.jsonl", "xps_demo_local_v1_part2.jsonl", "other.jsonl"):
        (tmp_path / n).write_text("", encoding="utf-8")
    got = XC.expand([str(tmp_path / "xps_demo_local_v1*.jsonl"), str(tmp_path / "missing.jsonl")])
    assert [Path(p).name for p in got] == ["xps_demo_local_v1.jsonl", "xps_demo_local_v1_part2.jsonl", "missing.jsonl"]
