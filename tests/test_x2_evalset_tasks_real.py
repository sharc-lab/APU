"""B3 task real run, x2_evalset_tasks_real_v1 (pre-registered in docs/FINDINGS.md 2026-10-14): --plan tasks_v1,
--preflight end_state, tasks_real_outcome (wrong-task rate per window, excess with a task-cluster bootstrap CI, silent
vs surfaced), the anchored queue insertion of both B3 jobs, and the register rows. Fake runtime only; no network, no
process launches, no queue."""
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "tests"))

import x2_r2_agent as ag  # noqa: E402
import x2_r2_evalset as ev  # noqa: E402
from test_x2_r2_evalset import FakeRuntime, perfect, _rows  # noqa: E402

DEF, K4 = "ollama_default_call2_notools", "ollama_ctx_4096_call2_notools"
M = "qwen3:14b"


class ErrRuntime(FakeRuntime):
    """Call 1 of the given turns returns HTTP 500 (a surfaced failure)."""

    def __init__(self, policy, err_turns=(), **kw):
        super().__init__(policy, **kw)
        self.err_turns = set(err_turns)

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        turn_idx = sum(1 for m in messages if m["role"] == "user")
        if turn_idx in self.err_turns and messages[-1]["role"] == "user":
            self.calls.append({})
            return {"outcome": "http_error", "status": 500, "error": "boom", "duration_s": 0.1}
        return super().chat(model, messages, num_ctx, tools, think, extra)


def _real_rows(models=(M,), k4_flags=None, def_flags=None, err_turns=(), drop=None, run_end=True):
    rows = [{"record": "run_start", "mode": "real", "plan_name": ev.TASKS_PLAN}]
    for model in models:
        for arm, seeds, turns in ev.tasks_plan():
            for seed in seeds:
                if drop == (arm, seed):
                    continue
                es = ev.build_session(seed, turns, "evalset", False, model)
                flags = (k4_flags if arm == K4 else def_flags) or {}
                rt = ErrRuntime(perfect(es, **flags), err_turns=err_turns if arm == K4 else (),
                                loaded=4096 if arm == K4 else 40960)
                ev.run_session(rt, model, arm, seed, turns, rows.append, lambda m: None, "real", "evalset", False)
    if run_end:
        rows.append({"record": "run_end"})
    return rows


@pytest.fixture(scope="module")
def val_rows():
    return _rows(guardrails=False, models=(M,))


# ── plan and CLI ────────────────────────────────────────────────────────────────────────────────────

def test_tasks_plan_is_default_and_4096_three_seeds_30_turns():
    assert ev.tasks_plan() == [(DEF, tuple(ag.SEEDS), 30), (K4, tuple(ag.SEEDS), 30)]
    assert ag.ARMS[DEF]["num_ctx"] is None and ag.ARMS[K4]["num_ctx"] == 4096
    assert all(ag.ARMS[a]["call2_mode"] == "off" for a, _, _ in ev.tasks_plan())


def test_queued_command_parses_and_check_plan_prints_30_task_sessions(capsys, monkeypatch):
    import t2s_queue as tq
    monkeypatch.setattr(tq, "advance", lambda note: (_ for _ in ()).throw(AssertionError("advance called")))
    e = next(x for x in ev.queue_entries_b3("PY") if x["id"] == ev.TASKS_REAL_JOB_ID)
    assert e["cmd"][:2] == ["PY", "x2_r2_evalset.py"]
    a = ev.build_arg_parser().parse_args(e["cmd"][2:])
    assert (a.mode, a.workload, a.guardrails, a.plan, a.preflight) == ("real", "evalset", "off", "tasks_v1",
                                                                       "end_state")
    assert a.rules_from.endswith("x2_evalset_tasks_val_v1.jsonl") and a.out.endswith("x2_evalset_tasks_real_v1.jsonl")
    assert ev.plan_for(a) == ev.tasks_plan()
    assert ev.main(e["cmd"][2:] + ["--check-plan"]) == "checked"
    out = capsys.readouterr().out
    assert out.count('"task": 30') == 6 and '"guardrail": 0' in out


@pytest.mark.parametrize("argv", [
    ["--mode", "real", "--plan", "tasks_v1", "--guardrails", "on", "--rules-from", "v", "--out", "o"],
    ["--mode", "real", "--plan", "tasks_v1", "--guardrails", "off", "--tiers", "4096", "--rules-from", "v", "--out",
     "o"],
    ["--mode", "validation", "--plan", "tasks_v1", "--guardrails", "off", "--out", "o"],
    ["--mode", "real", "--preflight", "end_state", "--guardrails", "on", "--rules-from", "v", "--out", "o"],
    ["--out", "o"],
])
def test_cli_errors(argv):
    with pytest.raises(SystemExit):
        ev.main(argv)


# ── end-state preflight ─────────────────────────────────────────────────────────────────────────────

def test_end_state_preflight_passes_where_the_r2_preflight_refuses():
    rows = []   # the negative control misses a canary per session, as in x2_evalset_tasks_val_v1
    for r in _rows(guardrails=False, models=(M,)):
        if "131072" in str(r.get("arm_id")) and r.get("record") == "r2a_session":
            r = dict(r, canary_sys_misses=1)
        if "131072" in str(r.get("arm_id")) and r.get("record") == "r2a_turn" and r.get("canary_check"):
            r = dict(r, canary_sys_ok=False)
        rows.append(r)
    assert not ev.validation_preflight_ext(rows, [M], guardrails=False)["ok"]
    pf = ev.end_state_preflight(rows, [M])
    assert pf["ok"] and pf["models_ok"] == [M] and pf["preflight"] == "end_state"
    assert pf["r2_gates_not_required"][M]["negative_control"] is False
    assert len(pf["tasks_in_use"][M]) == 30
    assert all(len(v) == 30 for v in pf["tasks_in_use_per_seed"][M].values())


def test_end_state_preflight_refusals():
    rows = _rows(guardrails=False, models=(M,), run_end=False)
    pf = ev.end_state_preflight(rows, [M])
    assert not pf["ok"] and any("run_end" in r for r in pf["refused"][M])
    pf = ev.end_state_preflight(_rows(guardrails=False, models=(M,)), [M, "llama3.1:8b"])
    assert pf["models_ok"] == [M] and any("negative-control sessions" in r for r in pf["refused"]["llama3.1:8b"])


def test_end_state_preflight_refuses_a_model_with_no_correct_task(val_rows):
    rows = [dict(r, end_state_ok=False) if r.get("record") == "r2a_turn" else r for r in val_rows]
    pf = ev.end_state_preflight(rows, [M])
    assert not pf["ok"] and any("no task correct" in r for r in pf["refused"][M])


# ── outcome ─────────────────────────────────────────────────────────────────────────────────────────

def test_outcome_counts_wrong_silent_and_surfaced(val_rows):
    real = _real_rows(k4_flags={"wrong_sku": True}, err_turns=(1,))
    o = ev.tasks_real_outcome(real, val_rows)
    d = o["models"][M]
    assert o["run_finished"] and d["valid"] and not d["problems"]
    p = d["primary"]
    assert p["n_tasks"] == 30 and p["n_cells_in_use"] == 90
    assert p["default"]["n"] == 90 and p["default"]["wrong"] == 0 and p["default"]["rate"] == 0
    # 4096: inv_sell wrong on 3 seeds (silent) plus the turn-1 task of each seed failing on HTTP 500 (surfaced)
    first = {s: ev.build_session(s, 30, "evalset", False, M).items[0].task.task_id for s in ag.SEEDS}
    n_silent = sum(1 for s in ag.SEEDS if first[s] != "inv_sell")   # inv_sell cells not already failed by the 500
    assert p["4096"]["silent_wrong"] == n_silent
    assert p["4096"]["surfaced_wrong"] == 3
    assert p["4096"]["wrong"] == p["4096"]["silent_wrong"] + 3
    assert p["excess"] == pytest.approx(p["4096"]["wrong"] / 90)
    ci = p["excess_ci95"]
    assert ci["unit"] == "task" and ci["n_boot"] == ev.BOOT_N and ci["n_tasks"] == 30
    assert ci["lo"] <= p["excess"] <= ci["hi"]
    assert p["reading"] == ev.excess_reading(ci)
    assert d["windows"]["4096"]["loaded_context_values"] == [4096]
    assert d["secondary"]["n_cells_in_use"] == 90


def test_outcome_primary_not_estimable_when_no_task_in_use(val_rows):
    # seed 1 of the validation all wrong: no task is correct on every session, per-seed cells remain for seeds 2-3
    s1 = ag.SEEDS[0]
    rows = [dict(r, end_state_ok=False) if r.get("record") == "r2a_turn" and r.get("seed") == s1 else r
            for r in val_rows]
    o = ev.tasks_real_outcome(_real_rows(), rows)
    d = o["models"][M]
    assert d["primary"]["n_cells_in_use"] == 0 and d["primary"]["excess"] is None and d["primary"]["reading"] is None
    assert d["secondary"]["n_cells_in_use"] == 60
    assert d["secondary"]["default"]["not_in_use_n"] == 30


def test_outcome_flags_incomplete_and_refused(val_rows):
    o = ev.tasks_real_outcome(_real_rows(drop=(K4, ag.SEEDS[1]), run_end=False), val_rows)
    d = o["models"][M]
    assert not d["valid"] and "no run_end" in d["problems"]
    assert any("4096: sessions completed" in p for p in d["problems"])
    assert d["primary"]["4096"]["n_missing"] == 30
    rows = _real_rows() + [{"record": "refused_model", "model_id": "llama3.1:8b", "reasons": ["x"]}]
    assert ev.tasks_real_outcome(rows, val_rows)["models"]["llama3.1:8b"] == {"refused": True, "reasons": ["x"]}


def test_bootstrap_is_deterministic_and_reading_follows_the_ci():
    per = {f"t{i}": (0, 3, 3 if i < 10 else 0, 3) for i in range(30)}
    a, b = ev.cluster_bootstrap_excess(per), ev.cluster_bootstrap_excess(per)
    assert a == b and a["lo"] > 0 and ev.excess_reading(a) == "4096 raises the wrong-task rate"
    per0 = {f"t{i}": (1 if i < 3 else 0, 3, 1 if i in (5, 6, 7) else 0, 3) for i in range(30)}
    assert ev.excess_reading(ev.cluster_bootstrap_excess(per0)) == "no excess shown"
    neg = {f"t{i}": (3 if i < 10 else 0, 3, 0, 3) for i in range(30)}
    assert ev.excess_reading(ev.cluster_bootstrap_excess(neg)) == "lower at 4096"
    assert ev.cluster_bootstrap_excess({})["lo"] is None and ev.excess_reading({"lo": None}) is None
    assert ev._quantile([0.0, 1.0], 0.5) == 0.5 and ev._quantile([], 0.5) is None


# ── queue insertion ─────────────────────────────────────────────────────────────────────────────────

def _queue():
    ids = ["x2_evalset_guard_val_v1", "x2_num_keep_guardrail_test", "x2_model_pulls_scale",
           "x2_family_smoke_mistral_small32", "x2_scale_timing_probe"]
    return [{"id": i, "status": "running" if k == 0 else "pending"} for k, i in enumerate(ids)]


def test_insert_b3_goes_directly_after_the_anchor_in_order():
    new = ev.insert_b3_anchored(_queue(), ev.queue_entries_b3("PY"))
    assert [it["id"] for it in new] == ["x2_evalset_guard_val_v1", "x2_num_keep_guardrail_test",
                                        ev.OFFLOAD_JOB_ID, ev.TASKS_REAL_JOB_ID, "x2_model_pulls_scale",
                                        "x2_family_smoke_mistral_small32", "x2_scale_timing_probe"]
    with pytest.raises(ValueError):
        ev.insert_b3_anchored(new, ev.queue_entries_b3("PY"))
    with pytest.raises(ValueError):
        ev.insert_b3_anchored([q for q in _queue() if q["id"] != ev.B3_ANCHOR], ev.queue_entries_b3("PY"))


def test_offload_entry_runs_the_preregistered_validation_through_the_sidecar():
    e = next(x for x in ev.queue_entries_b3("PY") if x["id"] == ev.OFFLOAD_JOB_ID)
    assert e["cmd"][1] == "r2v32koff\\x2_r2_sidecar.py"
    a = ag.build_arg_parser().parse_args(e["cmd"][2:])
    assert (a.mode, a.call2_tools, a.models, a.neg_ctx, a.neg_sessions, a.num_gpu) == (
        "validation", "off", "llama3.1:8b", 32768, 5, 22)
    assert a.out.endswith("x2_r2_llama_val_32k_offload.jsonl")
    assert e["status"] == "pending" and "tq.advance()" in e["note"]


def test_queue_insert_b3_pauses_and_always_clears(monkeypatch):
    state = {"q": _queue(), "pause": [], "cleared": 0}
    fake = types.SimpleNamespace(set_pause=state["pause"].append, read_queue=lambda: state["q"],
                                 write_queue=lambda items: state.__setitem__("q", items),
                                 clear_pause=lambda: state.__setitem__("cleared", state["cleared"] + 1))
    monkeypatch.setitem(sys.modules, "t2s_queue", fake)
    ev.queue_insert_b3(log=lambda m: None)
    assert "released by clear_pause()" in state["pause"][0] and state["cleared"] == 1
    assert [it["id"] for it in state["q"]].index(ev.OFFLOAD_JOB_ID) == 2
    with pytest.raises(ValueError):   # second insert: nothing written, pause still cleared
        ev.queue_insert_b3(log=lambda m: None)
    assert state["cleared"] == 2


# ── register rows ───────────────────────────────────────────────────────────────────────────────────

def test_register_rows_pending_then_compute(tmp_path, val_rows):
    import numbers_register as nr
    for cid in ("x2-evalset-tasks-real-wrong-rate", "x2-evalset-tasks-real-silent"):
        e = next(x for x in nr.NUMBER_ENTRIES if x["claim_id"] == cid)
        assert e["pending_ok"] and e["data_files"] == [nr.X2_EVALSET_TASKS_REAL, nr.X2_EVALSET_TASKS_VAL]
    (tmp_path / "results").mkdir()
    for name, rows in ((nr.X2_EVALSET_TASKS_REAL, _real_rows(k4_flags={"wrong_sku": True})),
                       (nr.X2_EVALSET_TASKS_VAL, val_rows)):
        (tmp_path / name).write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
    w = nr.compute_x2_evalset_tasks_real_wrong_rate(tmp_path)["value"]
    assert "qwen3:14b: valid True" in w and "primary (30 tasks, 90 cells): wrong default 0/90 (0.0%)" in w
    assert "wrong 4096 3/90 (3.3%)" in w and "task-cluster bootstrap, 10000 resamples" in w
    s = nr.compute_x2_evalset_tasks_real_silent(tmp_path)["value"]
    assert "4096 wrong 3 = silent 3 + surfaced 0" in s
    v = nr.compute_x2_evalset_tasks_val_gates(tmp_path)["value"]
    assert "tasks in use (correct on every negative-control session) 30" in v
