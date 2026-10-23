"""harness/t2s_evalset.py (the evo-t2s wrapper around x2_r2_evalset.py), x2_r2_evalset's --server-env and
--plan tasks_v2_t2s_qwen3_14b, and the B3 v2 entries of queues/t2s_oct26_queue.json. A fake x2_r2_evalset stands in
wherever main() would run; nothing starts Ollama, no machine, no SSH."""
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "scripts"))

import t2s_evalset as te  # noqa: E402
import x2_r2_evalset as ev  # noqa: E402


class FakeHC:
    def __init__(self):
        self.stops = 0

    def stop_ollama_server(self):
        self.stops += 1


def _fake_ev(note="completed", rows=None, exit_code=None):
    calls = []

    def main(argv=None, advance=True):
        calls.append((list(argv), advance))
        if exit_code is not None:
            raise SystemExit(exit_code)
        out = argv[argv.index("--out") + 1]
        if rows:
            with open(out, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        return note
    return types.SimpleNamespace(main=main), calls


def _turn(loaded, arm="ollama_default_call2_notools"):
    return {"record": "r2a_turn", "model_id": "qwen3:14b", "arm_id": arm, "loaded_context": loaded}


def _args(rc, out):
    return ["--runtime-config", rc, "--design", "v2", "--mode", "real", "--workload", "evalset", "--guardrails",
            "off", "--models", "qwen3:14b", "--plan", ev.TASKS_V2_T2S_PLAN, "--preflight", ev.PREFLIGHT_STRICT,
            "--rules-from", "results/t2s_evalset_tasks_val_v2_x.jsonl", "--out", str(out)]


def test_main_advances_exactly_once_on_success_igpu_env_and_runtime_check(tmp_path):
    out = tmp_path / "t2s_evalset_tasks_real_v2_igpu.jsonl"
    fake, calls = _fake_ev(rows=[_turn(32768)])
    notes, hc = [], FakeHC()
    note = te.main(_args("igpu_enable", out), ev=fake, hc=hc, tq=types.SimpleNamespace(advance=notes.append),
                   hostname="evo-t2s")
    assert note == "completed" and notes == ["completed"]
    argv, adv = calls[0]
    assert adv is False and argv[argv.index("--server-env") + 1] == "OLLAMA_IGPU_ENABLE=1" and hc.stops == 1
    chk = [json.loads(x) for x in out.read_text().splitlines() if "t2s_runtime_check" in x]
    assert chk and chk[0]["ok"] and chk[0]["expected_default_ctx"] == 32768


def test_main_advances_exactly_once_cpu_without_env_and_flags_a_window_mismatch(tmp_path):
    out = tmp_path / "t2s_evalset_tasks_real_v2_cpu.jsonl"
    fake, calls = _fake_ev(rows=[_turn(8192)])
    notes = []
    note = te.main(_args("cpu_default", out), ev=fake, hc=FakeHC(), tq=types.SimpleNamespace(advance=notes.append),
                   hostname="EVO-T2S")
    assert "--server-env" not in calls[0][0]
    assert "runtime check" in note and "4096" in note and notes == [note]


def test_main_advances_exactly_once_on_wrong_host_bad_name_and_rejected_args(tmp_path):
    fake, calls = _fake_ev()
    notes = []
    note = te.main(_args("cpu_default", tmp_path / "t2s_x.jsonl"), ev=fake, hc=FakeHC(),
                   tq=types.SimpleNamespace(advance=notes.append), hostname="EVO-X2")
    assert note.startswith("stopped:") and "EVO-T2S only" in note and notes == [note] and calls == []
    notes.clear()
    note = te.main(_args("cpu_default", tmp_path / "x2_x.jsonl"), ev=fake, hc=FakeHC(),
                   tq=types.SimpleNamespace(advance=notes.append), hostname="EVO-T2S")
    assert note.startswith("stopped:") and "t2s_" in note and notes == [note] and calls == []
    notes.clear()
    bad, _ = _fake_ev(exit_code=2)
    note = te.main(_args("cpu_default", tmp_path / "t2s_x.jsonl"), ev=bad, hc=FakeHC(),
                   tq=types.SimpleNamespace(advance=notes.append), hostname="EVO-T2S")
    assert note.startswith("stopped: x2_r2_evalset.py rejected its arguments (exit 2)") and notes == [note]


def test_dry_run_uses_the_real_parser_and_plan():
    args, rest = te.build_parser().parse_known_args(_args("igpu_enable", "results/t2s_evalset_tasks_real_v2_igpu.jsonl"))
    rep = te.dry_run(args, rest, ev)
    assert rep["parse_ok"], rep["parse_error"]
    assert rep["plan"] == [["ollama_default_call2_notools", [20260901, 20260902, 20260903], 30]]
    # the strict plan without the strict preflight is refused by x2_r2_evalset's own checks
    bad = [a for a in rest if a not in ("--preflight", ev.PREFLIGHT_STRICT)]
    assert te.dry_run(args, bad, ev)["parse_ok"] is False


def test_server_env_is_parsed_and_malformed_values_are_refused():
    assert ev.build_arg_parser().parse_args(["--mode", "validation", "--out", "t2s_a.jsonl", "--server-env",
                                             "OLLAMA_IGPU_ENABLE=1"]).server_env == ["OLLAMA_IGPU_ENABLE=1"]
    import pytest
    with pytest.raises(SystemExit):
        ev.main(["--mode", "validation", "--out", "t2s_a.jsonl", "--server-env", "bad", "--check-plan"],
                advance=False)


def test_t2s_plan_shape_and_strict_models():
    assert ev.STRICT_PLAN_MODELS[ev.TASKS_V2_T2S_PLAN] == ("qwen3:14b",)
    assert ev.STRICT_PLAN_SHAPE[ev.TASKS_V2_T2S_PLAN] == (("ollama_default",), (20260901, 20260902, 20260903))


def test_oct26_b3_entries_order_gates_and_commands():
    import t2s_week_queue as wq
    o26 = wq.build_oct26()
    ids = [it["id"] for it in o26]
    b3 = ["t2s_o26_b3_val_cpu", "t2s_o26_b3_real_cpu", "t2s_o26_b3_val_igpu", "t2s_o26_b3_real_igpu"]
    # after the preflight and A5, before every B5 entry
    assert ids[:6] == ["t2s_o26_preflight", "t2s_o26_window_override_methods", *b3]
    by = {it["id"]: it for it in o26}
    for be, rc in (("cpu", "cpu_default"), ("igpu", "igpu_enable")):
        v, r = by[f"t2s_o26_b3_val_{be}"], by[f"t2s_o26_b3_real_{be}"]
        assert v["gate"]["requires_done"] == "t2s_o26_preflight" and r["gate"]["requires_done"] == v["id"]
        for it in (v, r):
            c = it["cmd"]
            assert c[1] == "t2s_evalset.py" and c[c.index("--runtime-config") + 1] == rc
            assert c[c.index("--models") + 1] == "qwen3:14b" and c[c.index("--design") + 1] == "v2"
            assert it["est_hours"] > 0 and it["est_basis"].startswith("ESTIMATE")
        assert "--plan" not in v["cmd"] and v["cmd"][v["cmd"].index("--mode") + 1] == "validation"
        c = r["cmd"]
        assert c[c.index("--plan") + 1] == ev.TASKS_V2_T2S_PLAN and c[c.index("--preflight") + 1] == "v2_strict"
        assert c[c.index("--rules-from") + 1] == v["expected_out"]


def test_b3_estimates_are_labelled_and_counted_before_the_qwen_decision():
    import t2s_window_stage as ws
    est = ws.b3_estimates()
    assert set(est) == {"b3_val_cpu", "b3_real_cpu", "b3_val_igpu", "b3_real_igpu"}
    for k, e in est.items():
        assert e["basis"].startswith("ESTIMATE") and "t2s-r2-session-minutes" in e["basis"]
        assert 0 < e["hours"] <= e["high_h"]
    assert "ASSUMED" in est["b3_val_cpu"]["basis"]
    q = ws.qwen_decision()
    assert q["b3_h"] == round(sum(round(e["hours"], 1) for e in est.values()), 1)
    assert q["before_qwen_h"] == round(q["a5_h"] + q["b3_h"] + sum(q["llama_h"].values()), 1)


def test_preflight_deploys_the_evalset_files():
    import t2s_week_preflight as pf
    for f in ("t2s_evalset.py", "x2_r2_evalset.py", "evalset_env.py", "agent_trace_replay.py"):
        assert f in pf.DEPLOYED_FILES


def _register():
    sys.path.insert(0, str(REPO / "analysis"))
    import numbers_register as nr
    return nr


def test_register_rows_read_only_after_run_end_and_report_a_refusal(tmp_path):
    import pytest
    nr = _register()
    real = tmp_path / nr.T2S_EVALSET_FILES["cpu"][1]
    real.parent.mkdir(parents=True)
    real.write_text(json.dumps({"record": "run_start", "harness": "x2_r2_evalset"}) + "\n")
    with pytest.raises(ValueError, match="no run_end"):
        nr.compute_t2s_evalset_tasks(tmp_path, "cpu")
    with open(real, "a") as f:
        f.write(json.dumps({"record": "refused_model", "model_id": "qwen3:14b",
                            "reasons": ["qwen3:14b: strict preflight, gates not passed: ['task_end_state']"]}) + "\n")
        f.write(json.dumps({"record": "run_end"}) + "\n")
    v = nr.compute_t2s_evalset_tasks(tmp_path, "cpu")["value"]
    assert "refused by the strict preflight" in v and "task_end_state" in v and "projection stays" in v


def test_register_has_the_pending_rows_and_the_eia_input_row():
    nr = _register()
    ids = {e["claim_id"]: e for e in nr.NUMBER_ENTRIES}
    for k in ("t2s-evalset-tasks-cpu", "t2s-evalset-tasks-igpu"):
        assert ids[k]["pending_ok"] and any("t2s_evalset_tasks_real_v2" in f for f in ids[k]["data_files"])
    assert "EXTERNAL INPUT" in ids["energy-price-eia-commercial"]["description"]
    v = nr.compute_energy_price_eia(REPO)["value"]
    assert v.startswith("EXTERNAL INPUT") and "14.53 cents/kWh for 2026-07" in v and "fetched 2026-10-22" in v
    for row in ("PX2-power", "cost-cut-config-time", "impact-ref-machine-price"):
        assert row in ids and row in nr.NET_SAVINGS_INPUT_ROWS
