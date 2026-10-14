"""H-scale jobs (harness/x2_r2_scale.py, pre-registered in docs/FINDINGS.md 2026-10-14): projection arithmetic, the
70B default-tier rule, the real-run order, queue insertion order, the pre-registered verdict on synthetic rows, and the
job driver end to end with a fake Ollama runtime, fake host_config and fake queue."""
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "tests"))

import x2_r2_agent as ag  # noqa: E402
import x2_r2_scale as sc  # noqa: E402
from test_x2_r2_agent import FakeRuntime, good_policy  # noqa: E402


# ── projection ──────────────────────────────────────────────────────────────────────────────────────

def _call(pe_n, pe_s, ev_n, ev_s, load=None, dur=None):
    return {"prompt_eval_count_info_only": pe_n, "prompt_eval_duration_s": pe_s, "completion_tokens": ev_n,
            "eval_duration_s": ev_s, "load_duration_s": load, "duration_s": dur}


def _turn(idx, calls, wall, attempt="a", **kw):
    return {"record": "r2a_turn", "turn_idx": idx, "calls": calls, "turn_wall_s": wall, "attempt_id": attempt, **kw}


def test_rates_totals_over_turns_1_and_2_and_turn1_load():
    turns = [_turn(1, [_call(1000, 2.0, 50, 1.0, load=5.0), _call(500, 1.0, 50, 1.0)], 10),
             _turn(2, [_call(300, 1.0, 100, 2.0)], 5),
             _turn(3, [_call(9999, 1.0, 9999, 1.0)], 5)]
    r = sc.rates(turns)
    assert r["pe_rate"] == pytest.approx(1800 / 4.0) and r["ev_rate"] == pytest.approx(200 / 4.0)
    assert r["load_s"] == 5.0


def test_projection_identity_and_scaling():
    turns = [_turn(1, [_call(1000, 2.0, 50, 1.0, load=5.0)], 12.0), _turn(2, [_call(100, 1.0, 50, 1.0)], 6.0)]
    ref = sc.rates(turns)
    assert sc.project_session_minutes(turns, ref, dict(ref)) == pytest.approx(18.0 / 60)
    slow = dict(ref, pe_rate=ref["pe_rate"] / 2, ev_rate=ref["ev_rate"] / 4, load_s=20.0)
    # prompt-eval 3 s doubles (+3), eval 2 s quadruples (+6), load 5 -> 20 (+15)
    assert sc.project_session_minutes(turns, ref, slow) == pytest.approx((18.0 + 3 + 6 + 15) / 60)


def _sessions_file(model, arm, seeds, turns, mode="real", pe=(1000, 2.0), ev=(50, 1.0), wall=10.0, loaded=4096):
    rows = []
    for s in seeds:
        att = f"{model}-{arm}-{s}"
        for t in range(1, turns + 1):
            rows.append(_turn(t, [_call(pe[0], pe[1], ev[0], ev[1], load=4.0 if t == 1 else None)], wall,
                              attempt=att, loaded_context=loaded))
        rows.append({"record": "r2a_session", "mode": mode, "model_id": model, "arm_id": arm, "seed": s,
                     "attempt_id": att})
    return rows


def test_project_all_and_default_tier_rule():
    ref_real, probe = [], []
    for tier in sc.PROBE_TIERS:
        ref_real += _sessions_file("qwen3:14b", sc.arm(tier), ag.SEEDS, 40)
        ref_real += _sessions_file("llama3.1:8b", sc.arm(tier), ag.SEEDS, 40)
        # 32B at half the reference speed on both rates, same load
        probe += _sessions_file("qwen3:32b", sc.arm(tier), [sc.PROBE_SEED], 2, mode="probe", pe=(1000, 4.0),
                                ev=(50, 2.0))
        # 70B four times slower
        probe += _sessions_file("llama3.3:70b", sc.arm(tier), [sc.PROBE_SEED], 2, mode="probe", pe=(1000, 8.0),
                                ev=(50, 4.0))
    ref_val = []
    for a, seeds, turns in ag.validation_plan("off"):
        ref_val += _sessions_file("qwen3:14b", a, seeds, turns, mode="validation")
    p = sc.project_all(probe, ref_real, ref_val)
    t = p["models"]["qwen3:32b"]["tiers"][sc.TIER_4096]
    # each reference turn: 10 s wall, pe 2 s -> 4 s, ev 1 s -> 2 s: 13 s; 40 turns = 520 s
    assert t["k_pe"] == pytest.approx(2.0) and t["k_ev"] == pytest.approx(2.0)
    assert t["minutes_per_session"] == pytest.approx(520 / 60)
    assert t["hours_for_seeds"] == pytest.approx(3 * 520 / 3600)
    assert p["models"]["qwen3:32b"]["validation_minutes"] == pytest.approx((3 * 10 + 15 + 3 * 10) * 13 / 60)
    assert "llama3.3:70b" in p["models"] and "validation_minutes" not in p["models"]["llama3.3:70b"]
    assert p["models"]["qwen3:32b"]["default_tier"]["include"] is True
    d70 = p["models"]["llama3.3:70b"]["default_tier"]
    # 70B: 10 + 6 + 3 = 19 s per turn, 40 turns, 3 seeds = 0.63 h <= 24 h
    assert d70["include"] is True and d70["projected_hours"] == pytest.approx(3 * 40 * 19 / 3600)
    assert p["real_hours_if_all_pass"]["per_model"]["llama3.3:70b"]["tiers"][-1] == sc.TIER_DEFAULT


def test_default_tier_decision_cap_and_no_probe():
    def proj(h):
        return {"models": {"llama3.3:70b": {"tiers": {sc.TIER_DEFAULT: {"hours_for_seeds": h}}}}}
    assert sc.default_tier_decision(proj(24.0), "llama3.3:70b")["include"] is True
    assert sc.default_tier_decision(proj(24.01), "llama3.3:70b")["include"] is False
    assert sc.default_tier_decision(None, "llama3.3:70b")["include"] is False
    assert sc.default_tier_decision(None, "qwen3:32b")["include"] is True
    assert sc.default_tier_decision(None, "qwen3-30b-a3b-2507")["include"] is True


def test_think_values_injected_without_overriding_existing():
    assert ag.MODELS["qwen3:32b"] is False and ag.MODELS["qwen3-30b-a3b-2507"] is False
    assert "llama3.3:70b" in ag.MODELS and ag.MODELS["llama3.3:70b"] is None
    assert ag.MODELS["llama3.1:8b"] is None and ag.MODELS["qwen3:14b"] is False


# ── real-run order ──────────────────────────────────────────────────────────────────────────────────

def test_real_items_deciding_tiers_first_seed_major_then_default():
    items = sc.real_items(["qwen3:32b", "llama3.3:70b"], {"qwen3:32b": True, "llama3.3:70b": False})
    deciding, default = items[:12], items[12:]
    assert all(ag.base_arm(a) in sc.DECIDING_TIERS for _, a, _, _ in deciding)
    assert [s for _, _, s, _ in deciding] == [s for s in ag.SEEDS for _ in range(4)]
    assert default == [("qwen3:32b", "ollama_default_call2_notools", s, 40) for s in ag.SEEDS]
    assert all(t == 40 for *_, t in items)


# ── queue insertion ─────────────────────────────────────────────────────────────────────────────────

def _q(*ids, status="pending"):
    return [{"id": i, "status": status} for i in ids]


def test_insert_jobs_empty_queue_appends_in_order():
    out = sc.insert_jobs(_q("old_done", status="done"), sc.queue_entries())
    assert [i["id"] for i in out] == ["old_done", "x2_scale_timing_probe", "x2_r2_scale_validation",
                                      "x2_r2_scale_real"]


def test_insert_jobs_before_later_items_after_earlier():
    q = _q("x2_num_keep_guardrail_test", "x2_r2_family_mistral_validation", "x2_policy_compare")
    out = [i["id"] for i in sc.insert_jobs(q, sc.queue_entries())]
    assert out == ["x2_num_keep_guardrail_test", "x2_scale_timing_probe", "x2_r2_scale_validation",
                   "x2_r2_scale_real", "x2_r2_family_mistral_validation", "x2_policy_compare"]


def test_insert_jobs_ignores_done_later_items_and_refuses_duplicates():
    q = _q("x2_policy_compare", status="done") + _q("x2_window_override_methods")
    out = [i["id"] for i in sc.insert_jobs(q, sc.queue_entries())]
    assert out[-3:] == ["x2_scale_timing_probe", "x2_r2_scale_validation", "x2_r2_scale_real"]
    with pytest.raises(ValueError):
        sc.insert_jobs(_q("x2_scale_timing_probe"), sc.queue_entries())


def test_queue_entries_commands_and_notes():
    e = sc.queue_entries()
    assert [x["cmd"][2:4] for x in e] == [["--mode", "probe"], ["--mode", "validation"], ["--mode", "real"]]
    assert all(x["cmd"][1] == "x2_r2_scale.py" and "STOP" not in x["note"] for x in e)
    assert all("tq.advance()" in x["note"] for x in e)


def test_queue_insert_pauses_and_always_clears(monkeypatch):
    log = []
    state = {"items": _q("a", status="done")}
    fake = types.SimpleNamespace(
        set_pause=lambda reason: log.append(("pause", reason)), clear_pause=lambda: log.append(("clear",)),
        read_queue=lambda: state["items"], write_queue=lambda items: state.update(items=items))
    monkeypatch.setitem(sys.modules, "t2s_queue", fake)
    sc.queue_insert(log=lambda m: None)
    assert log[0][0] == "pause" and "finally" in log[0][1] and log[-1] == ("clear",)
    assert len(state["items"]) == 4
    log.clear()
    with pytest.raises(ValueError):
        sc.queue_insert(log=lambda m: None)   # second insert: duplicate ids
    assert log[-1] == ("clear",)


# ── H-scale verdict ─────────────────────────────────────────────────────────────────────────────────

RULES = ["rule1_json_keys", "rule3_no_zebra"]


def _sess_rows(model, tier, seed, over_from=3, rule_fail_at=None, canary_miss_at=None, err_at=None, turns=40):
    att = f"{model}|{tier}|{seed}"
    rows = []
    for t in range(1, turns + 1):
        cc = t % 5 == 0
        miss = canary_miss_at is not None and cc and t >= canary_miss_at
        rows.append({"record": "r2a_turn", "attempt_id": att, "turn_idx": t, "model_id": model,
                     "over_loaded_window": over_from is not None and t >= over_from,
                     "any_http_error": err_at is not None and t == err_at,
                     "rule1_json_keys": not (rule_fail_at is not None and t >= rule_fail_at), "rule3_no_zebra": True,
                     "canary_check": cc, "canary_sys_ok": (not miss) if cc else None,
                     "canary_hist_ok": (not miss) if cc else None})
    rows.append({"record": "r2a_session", "mode": "real", "model_id": model, "arm_id": sc.arm(tier), "seed": seed,
                 "attempt_id": att})
    return rows


def _real(models_spec, run_end=True, extra=()):
    rows = [{"record": "run_start", "mode": "real", "rules_in_use": {m: RULES for m in models_spec}}]
    for m, spec in models_spec.items():
        for tier, kw in spec.items():
            for s in ag.SEEDS:
                rows += _sess_rows(m, tier, s, **(kw(s) if callable(kw) else kw))
    rows += list(extra)
    if run_end:
        rows.append({"record": "run_end"})
    return rows


HOLDS = {sc.TIER_4096: {"over_from": 3, "rule_fail_at": 5}, sc.TIER_32768: {"over_from": 23, "canary_miss_at": 25}}


def test_verdict_holds_and_overall():
    v = sc.h_scale_verdict(_real({"qwen3:32b": HOLDS}), [], models=["qwen3:32b"])
    e = v["per_model"]["qwen3:32b"]
    assert e["verdict"] == "H-scale holds" and e["R4096"] == 3 and e["H32768"] == 3
    assert v["overall"] == "H-scale holds" and v["tested"] == ["qwen3:32b"]


def test_verdict_resists_and_mixed():
    resists = {sc.TIER_4096: {"over_from": 3}, sc.TIER_32768: HOLDS[sc.TIER_32768]}
    v = sc.h_scale_verdict(_real({"qwen3:32b": resists}), [], models=["qwen3:32b"])
    assert v["per_model"]["qwen3:32b"]["verdict"] == "alternative (resists rule loss)"
    assert v["overall"] == "alternative supported"
    one = {sc.TIER_4096: lambda s: {"over_from": 3, "rule_fail_at": 5 if s == ag.SEEDS[0] else None},
           sc.TIER_32768: HOLDS[sc.TIER_32768]}
    v = sc.h_scale_verdict(_real({"qwen3:32b": one, "qwen3-30b-a3b-2507": HOLDS}), [],
                           models=["qwen3:32b", "qwen3-30b-a3b-2507"])
    assert v["per_model"]["qwen3:32b"]["verdict"] == "mixed" and v["per_model"]["qwen3:32b"]["R4096"] == 1
    assert v["overall"] == "mixed"


def test_verdict_rule_failure_within_window_or_after_error_is_not_silent_after_window():
    early = {sc.TIER_4096: {"over_from": 10, "rule_fail_at": 5}, sc.TIER_32768: HOLDS[sc.TIER_32768]}
    e = sc.h_scale_verdict(_real({"m": early}), [], models=["m"])["per_model"]["m"]
    # the rule keeps failing after turn 10 too (rule_fail_at means from that turn on), so still counted after window
    assert e["R4096"] == 3
    loud = {sc.TIER_4096: {"over_from": 3, "rule_fail_at": 5, "err_at": 4}, sc.TIER_32768: HOLDS[sc.TIER_32768]}
    e = sc.h_scale_verdict(_real({"m": loud}), [], models=["m"])["per_model"]["m"]
    assert e["R4096"] == 0 and e["at_4096"] == "mixed"   # window exceeded in all 3, but not silent -> not "resists"?
    # (an error before the failure is not silent; R4096 = 0 with the window exceeded reads as resists by the rule)


def test_verdict_32768_rules_failing_or_window_not_exceeded():
    broke = {sc.TIER_4096: HOLDS[sc.TIER_4096], sc.TIER_32768: {"over_from": 23, "canary_miss_at": 25,
                                                                "rule_fail_at": 30}}
    e = sc.h_scale_verdict(_real({"m": broke}), [], models=["m"])["per_model"]["m"]
    assert e["H32768"] == 0 and e["at_32768"] == "does not hold" and e["verdict"] == "mixed"
    never = {sc.TIER_4096: HOLDS[sc.TIER_4096], sc.TIER_32768: {"over_from": None}}
    e = sc.h_scale_verdict(_real({"m": never}), [], models=["m"])["per_model"]["m"]
    assert e["at_32768"].startswith("inconclusive")


def test_verdict_incomplete_refused_missing():
    short = {sc.TIER_4096: {"over_from": 3, "rule_fail_at": 5, "turns": 39}, sc.TIER_32768: HOLDS[sc.TIER_32768]}
    v = sc.h_scale_verdict(_real({"m": short}), [], models=["m"])
    assert v["per_model"]["m"]["verdict"] == "inconclusive (incomplete)" and v["overall"] == "no model tested"
    v = sc.h_scale_verdict(_real({"m": HOLDS}, run_end=False), [], models=["m"])
    assert v["per_model"]["m"]["verdict"] == "inconclusive (incomplete)"
    extra = [{"record": "refused_model", "model_id": "qwen3-30b-a3b-2507", "reasons": ["x: task tool"],
              "refusal_category": "failed"},
             {"record": "missing_model", "model_id": "llama3.3:70b"}]
    v = sc.h_scale_verdict(_real({"qwen3:32b": HOLDS}, extra=extra), [], models=list(sc.SCALE_MODELS))
    assert v["per_model"]["qwen3-30b-a3b-2507"]["verdict"] == "refused (not tested)"
    assert v["per_model"]["llama3.3:70b"]["verdict"] == "missing"
    assert v["overall"] == "H-scale holds" and v["tested"] == ["qwen3:32b"]


# ── job driver with fakes ───────────────────────────────────────────────────────────────────────────

class TimedFake(FakeRuntime):
    def __init__(self, avail):
        super().__init__(None, loaded=4096)
        self.avail = avail
        self._ags = {}

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        seed_code = messages[0]["content"]
        ags = self._ags.get(seed_code)
        if ags is None:
            ags = next(a for a in (ag.build_agent_session(s, 40) for s in ag.SEEDS)
                       if a.system_prompt == seed_code)
            self._ags[seed_code] = ags
        self.policy = good_policy(ags)
        resp = super().chat(model, messages, num_ctx, tools, think, extra)
        resp["prompt_eval_count"] = 1000
        resp["raw"].update({"prompt_eval_duration": 2e9, "eval_duration": 1e9, "load_duration": 3e9})
        return resp

    def available_models(self):
        return self.avail

    def version(self):
        return "0.34.4-fake"


def _fakes(monkeypatch, avail, notes, started):
    import socket
    monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
        start_ollama_server=lambda: started.append(1), stop_ollama_server=lambda: None,
        wait_for_ollama_ready=lambda timeout_s=0: True,
        require_host=lambda h: {"hw_id": "evo-x2", "interactive_guard": False},
        enforce_or_record_interactive_session=lambda cfg: {"logged_in": False}))
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))
    rt = TimedFake(avail)
    monkeypatch.setattr(ag, "OllamaRuntime", lambda: rt)
    return rt


def test_probe_job_end_to_end_with_missing_model(tmp_path, monkeypatch):
    notes, started = [], []
    rt = _fakes(monkeypatch, ["qwen3:32b", "qwen3-30b-a3b-2507:latest"], notes, started)
    out = tmp_path / sc.PROBE_FILE
    note = sc.main(["--mode", "probe", "--out", str(out)])
    assert note == "completed" and notes == ["completed"] and started == [1]
    rows = ag.read_rows(out)
    assert [r["model_id"] for r in rows if r["record"] == "missing_model"] == ["llama3.3:70b"]
    sess = [r for r in rows if r["record"] == "r2a_session"]
    assert len(sess) == 6 and all(r["n_turns"] == 2 and r["mode"] == "probe" for r in sess)
    assert {r["think"] for r in sess} == {False}
    start = rows[0]
    assert start["record"] == "run_start" and start["agent_blob"] == sc.git_blob_sha1(ag.__file__)
    proj = next(r for r in rows if r["record"] == "scale_projection")["projection"]
    # no reference files in tmp_path: rates measured, no minutes projected
    t = proj["models"]["qwen3:32b"]["tiers"][sc.TIER_4096]
    assert t["n_reference_sessions"] == 0 and t["probe_rates"]["pe_rate"] == pytest.approx(500.0)
    assert rows[-1]["record"] == "run_end"
    assert all(c["think"] is False for c in rt.calls)


def test_real_job_refuses_without_validation_and_starts_no_server(tmp_path, monkeypatch):
    notes, started = [], []
    _fakes(monkeypatch, [], notes, started)
    out = tmp_path / sc.REAL_FILE
    note = sc.main(["--mode", "real", "--out", str(out)])
    assert started == [] and len(notes) == 1 and note.startswith("completed; refused to start")
    assert "STOP" not in note and "stopped:" not in note
    recs = ag.read_rows(out)
    assert [r["record"] for r in recs] == ["run_start", "refused_model", "refused_model", "refused_model",
                                           "refused", "run_end"]
    assert {r["refusal_category"] for r in recs if r["record"] == "refused_model"} == {"missing"}
    assert all(d["include"] is False or m in sc.DEFAULT_TIER_MODELS for m, d in recs[0]["default_tier"].items())


def test_validation_then_real_with_fakes(tmp_path, monkeypatch):
    notes, started = [], []
    _fakes(monkeypatch, ["qwen3:32b"], notes, started)
    note = sc.main(["--mode", "validation", "--models", "qwen3:32b", "--out", str(tmp_path / sc.VALIDATION_FILE)])
    # the fake keeps a 4096 loaded window: the 131072 negative control is "over window"... the gates are computed
    # from the rows as recorded, whatever they say; the job itself completes and advances once
    assert note.startswith("completed; validation passed") and len(notes) == 1
    vrows = ag.read_rows(tmp_path / sc.VALIDATION_FILE)
    assert {r["arm_id"] for r in vrows if r.get("record") == "r2a_session"} == {a for a, _, _ in
                                                                              ag.validation_plan("off")}
    rep = json.loads((tmp_path / (sc.VALIDATION_FILE + ".report.json")).read_text(encoding="utf-8"))
    assert "qwen3:32b" in rep["preflight_per_model"]


def test_import_check_runs_without_server(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace())
    assert sc.main(["--import-check"], advance=False) == "import-check"
    out = json.loads(capsys.readouterr().out)
    assert out["think"] == {"qwen3:32b": False, "qwen3-30b-a3b-2507": False, "llama3.3:70b": None}


# ── register rows ───────────────────────────────────────────────────────────────────────────────────

def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")


def test_register_rows_pending_until_synced_then_compute(tmp_path):
    sys.path.insert(0, str(REPO))
    import analysis.numbers_register as nr
    entries = {e["claim_id"]: e for e in nr.NUMBER_ENTRIES}
    for cid in ("x2-scale-probe-hours", "x2-scale-validation", "H-scale-verdict"):
        assert entries[cid]["pending_ok"] and nr._pending(entries[cid], repo=tmp_path)
    ref, probe = [], []
    for tier in sc.PROBE_TIERS:
        ref += _sessions_file("qwen3:14b", sc.arm(tier), ag.SEEDS, 40)
        probe += _sessions_file("qwen3:32b", sc.arm(tier), [sc.PROBE_SEED], 2, mode="probe", pe=(1000, 4.0),
                                ev=(50, 2.0))
    probe.append({"record": "missing_model", "model_id": "llama3.3:70b"})
    _write(tmp_path / nr.R2_REAL_V1_FILE, ref)
    _write(tmp_path / nr.R2_REAL_V1B_FILE, [])
    _write(tmp_path / nr.R2_VALIDATION_V2_FILE, [])
    _write(tmp_path / nr.X2_SCALE_PROBE, probe)
    out = nr.compute_x2_scale_probe_hours(tmp_path)
    assert "qwen3:32b (ref qwen3:14b)" in out["value"] and "8.7 min/session" in out["value"]
    assert "missing from the store: ['llama3.3:70b']" in out["value"]
    _write(tmp_path / nr.X2_SCALE_REAL, _real({"qwen3:32b": HOLDS}))
    _write(tmp_path / nr.X2_SCALE_VALIDATION, [])
    v = nr.compute_h_scale_verdict(tmp_path)["value"]
    assert v.startswith("overall H-scale holds") and "R4096 3/3" in v and "H32768 3/3" in v
