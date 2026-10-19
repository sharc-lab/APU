"""B3 v2 task real run, x2_evalset_tasks_real_v2 (docs/FINDINGS.md 2026-10-19: deviation note and the protocol
consistency pre-registration): --plan tasks_v2, the num_keep tier (NumKeepSessionRuntime, host_measure), --preflight
v2_te_deviation, tasks_real_v2_outcome, the duration estimate, the queue insertion before x2_window_override_methods,
and the register rows. Fake runtime only; no network, no process launches, no queue."""
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "tests"))

import x2_r2_agent as ag  # noqa: E402
import x2_r2_evalset as ev  # noqa: E402
from test_x2_r2_evalset import FakeRuntime, perfect  # noqa: E402

M, L = "qwen3:14b", "llama3.1:8b"
SFX = "_call2_notools"
DEF, K16, K8, K4, NK = (t + SFX for t in ev.TASKS_V2_TIERS)
S_FAKE = 612


class RecRuntime(FakeRuntime):
    """Records options_extra on every chat call (what NumKeepSessionRuntime set on the base runtime)."""

    options_extra = None

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        resp = super().chat(model, messages, num_ctx, tools, think, extra)
        self.calls[-1]["options_extra"] = self.options_extra
        return resp


def fake_measure(s=S_FAKE):
    return lambda model, arm, seed, es: {"system_prompt_tokens": s, "variants": {}, "method": "fake"}


def drop_code_after(es, turn_from):
    """perfect(es), except the final answer leaves out the session code from turn_from on (rule 5 fails silently)."""
    pol = perfect(es)

    def policy(call_no, messages, turn_idx):
        content, native = pol(call_no, messages, turn_idx)
        if turn_idx >= turn_from and content:
            content = content.replace(es.session_code, "")
        return content, native
    return policy


def _v2_val_rows(models=(M,), te_fail=0):
    """A finished design-v2 tasks validation (perfect model); te_fail negative-control task turns per model marked
    end-state wrong, so the task end-state gate reads (90 - te_fail)/90."""
    rows = [{"record": "run_start", "mode": "validation", "call2_mode": "off", "call2_tools": False, "design": "v2"}]
    for model in models:
        for arm, seeds, turns in ev.validation_plan("evalset", False, "off", design="v2"):
            for seed in seeds:
                pos = "positive" in arm
                es = ev.build_session(seed, turns, "evalset", False, model, design="v2")
                ev.run_session(FakeRuntime(perfect(es, forget=pos), loaded=8192 if pos else 131072), model, arm, seed,
                               turns, rows.append, lambda m: None, "validation", "evalset", False, None, "v2")
    rows.append({"record": "run_end"})
    left = {m: te_fail for m in models}
    out = []
    for r in rows:
        if (r.get("record") == "r2a_turn" and r.get("item_kind") == "task" and ag.NEG_ARM in r["arm_id"]
                and left[r["model_id"]] > 0):
            left[r["model_id"]] -= 1
            r = dict(r, end_state_ok=False)
        out.append(r)
    return out


@pytest.fixture(scope="module")
def val_rows():
    return _v2_val_rows(te_fail=10)


def _real_rows(policy_for=None, drop=None, run_end=True, unsent_seed=None):
    """Rows of a tasks_v2 run through NumKeepSessionRuntime over a recording fake: llama3.1:8b refused, qwen3:14b
    every tier x seed. policy_for(arm, es) -> policy (default perfect)."""
    rows = [{"record": "refused_model", "model_id": L, "reasons": [f"{L}: baseline gate failed"]},
            {"record": "run_start", "mode": "real", "plan_name": ev.TASKS_V2_PLAN, "design": "v2",
             "rules_in_use": {M: list(ag.RULE_IDS)}, "call2_mode": "off"}]
    for arm, seeds, turns in ev.tasks_v2_plan():
        for seed in seeds:
            if drop == (arm, seed):
                continue
            es = ev.build_session(seed, turns, "evalset", False, M, design="v2")
            pol = (policy_for or (lambda a, e: perfect(e)))(arm, es)
            loaded = 40960 if arm == DEF else ag.ARMS[arm]["num_ctx"]
            base = RecRuntime(pol, loaded=loaded)
            rt = ev.NumKeepSessionRuntime(base, fake_measure(), rows.append)
            ev.run_session(rt, M, arm, seed, turns, rows.append, lambda m: None, "real", "evalset", False, None, "v2")
    if unsent_seed is not None:
        for r in rows:
            if r.get("record") == "r2a_turn" and r["arm_id"] == NK and r["seed"] == unsent_seed and r["turn_idx"] == 3:
                r["calls"][0].pop("num_keep", None)
    if run_end:
        rows.append({"record": "run_end"})
    return rows


@pytest.fixture(scope="module")
def real_rows():
    return _real_rows(policy_for=lambda arm, es: drop_code_after(es, 20) if arm == K4 else perfect(es))


# ── plan and CLI ────────────────────────────────────────────────────────────────────────────────────

def test_tasks_v2_plan_tiers_seeds_turns():
    plan = ev.tasks_v2_plan()
    assert plan == [(a, tuple(ag.SEEDS_STRONG), 30) for a in (DEF, K16, K8, K4, NK)]
    assert [ag.ARMS[a]["num_ctx"] for a, _, _ in plan] == [None, 16384, 8192, 4096, 4096]
    assert all(ag.ARMS[a]["call2_mode"] == "off" for a, _, _ in plan)
    assert NK in ev.NUMKEEP_ARMS and K4 not in ev.NUMKEEP_ARMS
    assert ev.TASKS_V2_MODELS == (M,) and ev.TE_DEVIATION == {M: (80, 90)}


def test_queued_command_parses_and_check_plan_prints_25_sessions(capsys, monkeypatch):
    import t2s_queue as tq
    monkeypatch.setattr(tq, "advance", lambda note: (_ for _ in ()).throw(AssertionError("advance called")))
    e = ev.queue_entry_tasks_v2("PY")
    assert e["cmd"][:2] == ["PY", "evalv2\\x2_r2_evalset.py"]
    a = ev.build_arg_parser().parse_args(e["cmd"][2:])
    assert (a.design, a.mode, a.workload, a.guardrails, a.plan, a.preflight, a.models) == (
        "v2", "real", "evalset", "off", "tasks_v2", "v2_te_deviation", None)
    assert a.rules_from.endswith("x2_evalset_tasks_val_v2.jsonl") and a.out.endswith("x2_evalset_tasks_real_v2.jsonl")
    assert ev.plan_for(a) == ev.tasks_v2_plan()
    assert ev.main(e["cmd"][2:] + ["--check-plan"]) == "checked"
    out = capsys.readouterr().out
    assert out.count('"task": 30') == 25 and '"guardrail": 0' in out and NK in out


@pytest.mark.parametrize("argv", [
    ["--mode", "real", "--design", "v2", "--plan", "tasks_v2", "--guardrails", "off", "--rules-from", "v", "--out", "o"],
    ["--mode", "real", "--design", "v1", "--plan", "tasks_v2", "--preflight", "v2_te_deviation", "--guardrails", "off",
     "--rules-from", "v", "--out", "o"],
    ["--mode", "real", "--design", "v2", "--plan", "tasks_v2", "--preflight", "v2_te_deviation", "--guardrails", "on",
     "--rules-from", "v", "--out", "o"],
    ["--mode", "real", "--design", "v2", "--plan", "tasks_v2", "--preflight", "v2_te_deviation", "--guardrails", "off",
     "--tiers", "4096", "--rules-from", "v", "--out", "o"],
    ["--mode", "real", "--design", "v2", "--plan", "tasks_v1", "--preflight", "v2_te_deviation", "--guardrails", "off",
     "--rules-from", "v", "--out", "o"],
    ["--mode", "validation", "--design", "v2", "--preflight", "v2_te_deviation", "--guardrails", "off", "--out", "o"],
])
def test_cli_errors(argv):
    with pytest.raises(SystemExit):
        ev.main(argv)


# ── preflight with the declared deviation ───────────────────────────────────────────────────────────

def test_preflight_waives_only_the_recorded_end_state_count(val_rows):
    plain = ev.validation_preflight_ext(val_rows, [M], "evalset", False, "off", "v2")
    assert not plain["ok"] and any("task end-state gate failed (80/90" in r for r in plain["refused"][M])
    pf = ev.validation_preflight_ext(val_rows, [M], "evalset", False, "off", "v2",
                                     waive_task_end_state=ev.TE_DEVIATION, allowed_models=ev.TASKS_V2_MODELS)
    assert pf["ok"] and pf["models_ok"] == [M]
    assert pf["waived"][M]["n_ok"] == 80 and pf["waived"][M]["n"] == 90
    assert pf["rules_in_use"][M] == list(ag.RULE_IDS)
    other = ev.validation_preflight_ext(val_rows, [M], "evalset", False, "off", "v2",
                                        waive_task_end_state={M: (79, 90)}, allowed_models=ev.TASKS_V2_MODELS)
    assert not other["ok"] and not other["waived"]


def test_preflight_refuses_another_count_another_design_and_llama():
    rows79 = _v2_val_rows(te_fail=11)
    pf = ev.validation_preflight_ext(rows79, [M], "evalset", False, "off", "v2",
                                     waive_task_end_state=ev.TE_DEVIATION, allowed_models=ev.TASKS_V2_MODELS)
    assert not pf["ok"] and any("79/90" in r for r in pf["refused"][M])
    both = _v2_val_rows(models=(L, M), te_fail=10)
    pf = ev.validation_preflight_ext(both, [L, M], "evalset", False, "off", "v2",
                                     waive_task_end_state=ev.TE_DEVIATION, allowed_models=ev.TASKS_V2_MODELS)
    assert pf["models_ok"] == [M] and any("not in this plan's model list" in r for r in pf["refused"][L])
    assert L not in pf["waived"]
    v1 = [dict(r, design="v1") if r.get("record") == "run_start" else r for r in _v2_val_rows(te_fail=10)]
    pf = ev.validation_preflight_ext(v1, [M], "evalset", False, "off", "v2",
                                     waive_task_end_state=ev.TE_DEVIATION, allowed_models=ev.TASKS_V2_MODELS)
    assert not pf["ok"] and any("design" in r for r in pf["refused"][M]) and not pf["waived"]


def test_preflight_on_the_committed_validation_file_refuses_llama_and_admits_qwen():
    rows = ag.read_rows(REPO / "results/x2_evalset_tasks_val_v2.jsonl")
    pf = ev.validation_preflight_ext(rows, [L, M], "evalset", False, "off", "v2",
                                     waive_task_end_state=ev.TE_DEVIATION, allowed_models=ev.TASKS_V2_MODELS)
    assert pf["models_ok"] == [M] and pf["waived"][M]["n_ok"] == 80
    assert any("negative control failed" in r for r in pf["refused"][L])
    assert any("task end-state gate failed (72/90" in r for r in pf["refused"][L])


# ── num_keep tier ───────────────────────────────────────────────────────────────────────────────────

def _one_session(arm, measure=None, turns=4):
    es = ev.build_session(20260901, turns, "evalset", False, M, design="v2")
    base = RecRuntime(perfect(es), loaded=4096)
    rows = []
    rt = ev.NumKeepSessionRuntime(base, measure or fake_measure(), rows.append)
    ev.run_session(rt, M, arm, 20260901, turns, rows.append, lambda m: None, "real", "evalset", False, None, "v2")
    return base, rows


def test_num_keep_sent_on_every_call_of_the_num_keep_tier_only():
    base, rows = _one_session(NK)
    assert all(c["options_extra"] == {"num_keep": S_FAKE} for c in base.calls) and base.options_extra is None
    nks = [r for r in rows if r["record"] == "num_keep_session"]
    assert len(nks) == 1 and nks[0]["num_keep"] == S_FAKE and nks[0]["arm_id"] == NK
    turns = [r for r in rows if r["record"] == "r2a_turn"]
    assert all(c["num_keep"] == {"sent": S_FAKE, "system_prompt_tokens": S_FAKE} for t in turns for c in t["calls"])
    assert all(t["num_keep_sent"] == S_FAKE and t["num_ctx_requested"] == 4096 for t in turns)
    base, rows = _one_session(K4)
    assert all(c["options_extra"] is None for c in base.calls)
    assert not any(r["record"] == "num_keep_session" for r in rows)
    assert all("num_keep" not in c for r in rows if r["record"] == "r2a_turn" for c in r["calls"])


def test_num_keep_refuses_a_non_positive_s():
    with pytest.raises(ValueError):
        _one_session(NK, measure=fake_measure(0))


def test_host_measure_renders_and_counts_the_session_system_prompt():
    class RenderBase:
        options_extra = {"num_keep": 1}
        seen = []

        def render_only(self, model, messages, num_ctx, tools, think):
            self.seen.append((self.options_extra, num_ctx, tools is not None, think))
            return "".join(f"<{m['role']}>{m['content']}" for m in messages), None
    b = RenderBase()
    rt = ev.NumKeepSessionRuntime(b, ev.host_measure(b, count_fn_for=lambda model: (lambda text: len(text) // 4)),
                                  lambda r: None)
    es = ev.build_session(20260901, 3, "evalset", False, M, design="v2")
    s = rt.begin_session(M, NK, 20260901, es)
    assert s == len("<system>" + es.system_prompt) // 4
    assert b.options_extra == {"num_keep": s}
    assert all(o is None and n == 4096 and th is False for o, n, _, th in b.seen)   # measured without num_keep
    assert {t for _, _, t, _ in b.seen} == {True, False}
    rt.end_session()
    assert b.options_extra is None


# ── outcome ─────────────────────────────────────────────────────────────────────────────────────────

def test_outcome_valid_and_wrong_rate_per_tier(real_rows):
    o = ev.tasks_real_v2_outcome(real_rows)
    assert o["run_finished"] and o["refused"] == {L: [f"{L}: baseline gate failed"]}
    assert o["models"][L]["refused"]
    d = o["models"][M]
    assert d["valid"], d["problems"]
    assert set(d["tiers"]) == set(ev.TASKS_V2_TIERS)
    assert all(i["n_task_turns"] == 150 and i["wrong"] == 0 for i in d["tiers"].values())
    assert d["tiers"][ev.NUMKEEP_ARM]["num_keep_per_seed"] == {s: S_FAKE for s in ag.SEEDS_STRONG}
    assert d["tiers"]["ollama_ctx_4096"]["loaded_context_values"] == [4096]
    b = d["excess_over_default"]
    assert b["unit"] == "session" and b["n_resamples"] == ev.BOOT_N and b["seed"] == ev.BOOT_SEED
    assert set(b["per_tier"]) == set(ev.TASKS_V2_TIERS) - {"ollama_default"}
    assert all(e["excess"] == 0 and e["reading"] == "no excess shown" for e in b["per_tier"].values())


def test_outcome_counts_wrong_turns_and_reads_the_excess():
    rows = _real_rows(policy_for=lambda arm, es: perfect(es, wrong_sku=True) if arm == K4 else perfect(es))
    d = ev.tasks_real_v2_outcome(rows)["models"][M]
    k4 = d["tiers"]["ollama_ctx_4096"]
    assert d["valid"] and k4["wrong"] == 5 and k4["silent_wrong"] == 5 and k4["surfaced_wrong"] == 0
    e = d["excess_over_default"]["per_tier"]["ollama_ctx_4096"]
    assert e["excess"] == pytest.approx(5 / 150) and e["ci95"][0] > 0
    assert e["reading"] == "ollama_ctx_4096 raises the wrong-task rate"


def test_outcome_invalid_on_a_missing_session_or_a_call_without_num_keep():
    d = ev.tasks_real_v2_outcome(_real_rows(drop=(K8, 20260903)))["models"][M]
    assert not d["valid"] and any("ollama_ctx_8192: sessions completed" in p for p in d["problems"])
    d = ev.tasks_real_v2_outcome(_real_rows(unsent_seed=20260902))["models"][M]
    assert not d["valid"] and any("without num_keep" in p for p in d["problems"])
    d = ev.tasks_real_v2_outcome(_real_rows(run_end=False))["models"][M]
    assert not d["valid"] and "no run_end" in d["problems"]


def test_session_bootstrap_matches_r2_real_report_on_binary_sessions():
    import r2_real_report as rr
    for k in range(6):
        cells = {"d": [(0, 1)] * 5, "t": [(1 if i < k else 0, 1) for i in range(5)]}
        mine = ev.session_bootstrap_excess(cells, "d")["per_tier"]["t"]
        table = {"ollama_default|q": {"sessions": [{"after_window_exceeded": False}] * 5},
                 "num_ctx_4096|q": {"sessions": [{"after_window_exceeded": i < k} for i in range(5)]}}
        theirs = rr.excess_over_default({"table": table})["per_tier"]["num_ctx_4096"]["q"]
        assert mine["excess"] == pytest.approx(theirs["excess"])
        assert mine["ci95"] == pytest.approx(theirs["ci95"])


# ── main() end to end, everything faked ─────────────────────────────────────────────────────────────

def test_main_runs_the_queued_command_end_to_end_on_fakes(monkeypatch, tmp_path):
    import socket
    import host_config as hc
    import t2s_queue as tq
    val = tmp_path / "x2_evalset_tasks_val_v2.jsonl"
    val.write_bytes((REPO / "results/x2_evalset_tasks_val_v2.jsonl").read_bytes())
    out = tmp_path / "x2_evalset_tasks_real_v2.jsonl"
    notes, sessions = [], {}

    class MainFake(RecRuntime):
        """One fake for the whole run: a perfect policy for whichever session the last user message belongs to."""

        def __init__(self):
            super().__init__(None, loaded=4096)

        def chat(self, model, messages, num_ctx, tools, think, extra=None):
            es = sessions[messages[0]["content"]]
            self.policy = perfect(es)
            self.loaded = num_ctx or 40960
            return super().chat(model, messages, num_ctx, tools, think, extra)

        def available_models(self):
            return [M, L]

        def version(self):
            return "fake"

    orig_build = ev.build_session

    def build(*a, **k):
        es = orig_build(*a, **k)
        sessions[es.system_prompt] = es
        return es
    monkeypatch.setattr(ev, "build_session", build)
    monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
    monkeypatch.setattr(hc, "require_host", lambda h: {"hw_id": "fake"})
    monkeypatch.setattr(hc, "enforce_or_record_interactive_session", lambda cfg: False)
    monkeypatch.setattr(hc, "wait_for_ollama_ready", lambda timeout_s=60: True)
    monkeypatch.setattr(hc, "stop_ollama_server", lambda: None)
    monkeypatch.setattr(ag, "_start_server", lambda hc_: None)
    monkeypatch.setattr(ag, "OllamaRuntime", MainFake)
    monkeypatch.setattr(ev, "host_measure", lambda base: fake_measure())
    monkeypatch.setattr(tq, "advance", notes.append)
    cmd = ev.queue_entry_tasks_v2("PY", results_dir=str(tmp_path))["cmd"][2:]
    note = ev.main(cmd)
    assert note == "completed" and notes == ["completed"]
    rows = ag.read_rows(out)
    start = next(r for r in rows if r["record"] == "run_start")
    assert start["models"] == [M] and start["rules_in_use"] == {M: list(ag.RULE_IDS)}
    assert start["num_keep_tiers"] == [NK] and start["validation_preflight"]["waived"][M]["n_ok"] == 80
    assert [r["model_id"] for r in rows if r["record"] == "refused_model"] == [L]
    assert sum(1 for r in rows if r["record"] == "num_keep_session") == 5
    rep = json.loads(Path(str(out) + ".report.json").read_text(encoding="utf-8"))
    d = rep["tasks_real_v2_outcome"]["models"][M]
    assert d["valid"], d["problems"]
    assert rep["tasks_real_v2_outcome"]["models"][L]["refused"]


# ── estimate and queue ──────────────────────────────────────────────────────────────────────────────

def test_projected_hours_is_the_register_estimate():
    e = ev.tasks_v2_hours_estimate(ag.read_rows(REPO / "results/x2_evalset_tasks_val_v2.jsonl"))
    assert e["n_sessions"] == 25 and e["n_default_sessions"] == 5
    assert round(e["point_h"], 1) == ev.TASKS_V2_PROJECTED_H
    assert e["low_h"] <= e["point_h"] <= e["high_h"]
    assert f"{ev.TASKS_V2_PROJECTED_H} h" in ev.queue_entry_tasks_v2("PY")["note"]


def _queue():
    ids = [("x2_r2_family_mistral_small32_real", "done"), ("x2_policy_compare", "running"),
           ("x2_vulkan_job_other_agent", "pending"), ("x2_window_override_methods", "pending")]
    return [{"id": i, "status": s} for i, s in ids]


def test_insert_before_the_anchor_keeps_every_other_entry():
    new = ev.insert_before(_queue(), ev.queue_entry_tasks_v2("PY"))
    assert [it["id"] for it in new] == ["x2_r2_family_mistral_small32_real", "x2_policy_compare",
                                        "x2_vulkan_job_other_agent", ev.TASKS_V2_JOB_ID, "x2_window_override_methods"]
    with pytest.raises(ValueError):
        ev.insert_before(new, ev.queue_entry_tasks_v2("PY"))
    with pytest.raises(ValueError):
        ev.insert_before([q for q in _queue() if q["id"] != ev.TASKS_V2_BEFORE], ev.queue_entry_tasks_v2("PY"))
    with pytest.raises(ValueError):
        ev.insert_before([dict(q, status="running") if q["id"] == ev.TASKS_V2_BEFORE else q for q in _queue()],
                         ev.queue_entry_tasks_v2("PY"))
    e = ev.queue_entry_tasks_v2("PY")
    assert e["status"] == "pending" and "tq.advance()" in e["note"] and "sets no pause" in e["note"]


def test_queue_insert_tasks_v2_reads_inside_the_pause_and_always_clears(monkeypatch):
    state = {"q": [q for q in _queue() if q["id"] != "x2_vulkan_job_other_agent"], "pause": [], "cleared": 0}

    def set_pause(reason):
        state["pause"].append(reason)   # another session's entry lands just before our read
        state["q"] = state["q"][:2] + [{"id": "x2_vulkan_job_other_agent", "status": "pending"}] + state["q"][2:]
    fake = types.SimpleNamespace(set_pause=set_pause, read_queue=lambda: state["q"],
                                 write_queue=lambda items: state.__setitem__("q", items),
                                 clear_pause=lambda: state.__setitem__("cleared", state["cleared"] + 1))
    monkeypatch.setitem(sys.modules, "t2s_queue", fake)
    ev.queue_insert_tasks_v2(log=lambda m: None)
    assert "released by clear_pause()" in state["pause"][0] and state["cleared"] == 1
    assert [it["id"] for it in state["q"]][2:] == ["x2_vulkan_job_other_agent", ev.TASKS_V2_JOB_ID,
                                                   "x2_window_override_methods"]
    with pytest.raises(ValueError):
        ev.queue_insert_tasks_v2(log=lambda m: None)
    assert state["cleared"] == 2


# ── register rows ───────────────────────────────────────────────────────────────────────────────────

def test_register_rows_pending_then_compute(tmp_path, monkeypatch, real_rows):
    import numbers_register as nr
    for cid in ("x2-evalset-tasks-real-v2-gated-excess", "x2-evalset-tasks-real-v2-wrong-rate"):
        e = next(x for x in nr.NUMBER_ENTRIES if x["claim_id"] == cid)
        assert e["pending_ok"] and nr.X2_EVALSET_TASKS_REAL_V2 in e["data_files"]
    (tmp_path / "results").mkdir()
    (tmp_path / nr.X2_EVALSET_TASKS_REAL_V2).write_text(
        "\n".join(json.dumps(r, default=str) for r in real_rows) + "\n", encoding="utf-8")
    (tmp_path / nr.X2_EVALSET_TASKS_VAL_V2).write_bytes((REPO / nr.X2_EVALSET_TASKS_VAL_V2).read_bytes())
    w = nr.compute_x2_evalset_tasks_real_v2_wrong_rate(tmp_path)["value"]
    assert f"{L}: refused" in w and f"{M}: valid True" in w and "ollama_ctx_4096 excess over default +0.0 points" in w
    orig = nr._r2_real_v1b_report
    monkeypatch.setattr(nr, "_r2_real_v1b_report", lambda repo: orig(nr.REPO))
    g = nr.compute_x2_evalset_tasks_real_v2_gated_excess(tmp_path)["value"]
    assert f"{M}: valid True" in g and "ollama_default 0/5 silent" in g
    # rule 5 dropped from turn 20 at 4096 only: every 4096 session is a gated silent failure after the window
    assert "num_ctx_4096: v2 5/5 vs default 0/5, excess +1.000" in g
    assert "v1b excess +0.800 (95% CI +0.400 to +1.000); reading: the v1b effect holds" in g
    assert "num_ctx_8192: v2 0/5 vs default 0/5" in g and "reading: no excess in either protocol" in g
    assert "num keep reading" not in g and "num_keep reading: num_keep removes the 4096 excess for qwen3:14b" in g


def test_protocol_reading_rule():
    import numbers_register as nr
    v1b = (0.8, (0.4, 1.0))
    assert nr.protocol_reading((0.2, 1.0), *v1b) == "the v1b effect holds"
    assert nr.protocol_reading((0.0, 0.6), *v1b) == "the v1b effect is a protocol artifact"
    assert nr.protocol_reading((0.0, 0.8), *v1b) == "inconclusive"
    assert nr.protocol_reading((0.2, 0.6), 0.0, (0.0, 0.0)) == "v2 shows an excess v1b did not"
    assert nr.protocol_reading((0.0, 0.4), 0.0, (0.0, 0.0)) == "no excess in either protocol"
