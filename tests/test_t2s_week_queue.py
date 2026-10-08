"""queues/t2s_week_queue.json, scripts/t2s_week_queue.py (build / dry-run / inputs / load) and
harness/t2s_week_preflight.py. Fakes only: no SSH, no machine, no Ollama."""
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "scripts"))

import t2s_week_preflight as pf  # noqa: E402
import t2s_week_queue as wq  # noqa: E402


# ------------------------------------------------------------------ queue file
def test_committed_queue_file_is_exactly_build_queue():
    assert json.loads(wq.QUEUE_PATH.read_text(encoding="utf-8")) == wq.build_queue()


def test_queue_order_ids_and_gates():
    q = wq.build_queue()
    keys = [it["id"][len(wq.ID_PREFIX):] for it in q]
    assert keys == ["preflight", "r2_val_cpu", "r2_val_igpu", "r2_real_cpu", "r2_real_igpu", "r2_mitigation_cpu",
                    "px2i", "outcome_subset", "r2_mechanism_cpu"]
    assert [it["plan_step"] for it in q] == [1, 2, 2, 2, 2, 3, 4, 5, 6]
    assert all(it["status"] == "pending" for it in q)
    assert "gate" not in q[0]
    gates = {it["id"]: it["gate"]["requires_done"] for it in q[1:]}
    assert gates["t2s_wk_r2_real_cpu"] == "t2s_wk_r2_val_cpu"
    assert gates["t2s_wk_r2_real_igpu"] == "t2s_wk_r2_val_igpu"
    assert gates["t2s_wk_r2_mitigation_cpu"] == "t2s_wk_r2_val_cpu"
    assert gates["t2s_wk_px2i"] == gates["t2s_wk_outcome_subset"] == "t2s_wk_preflight"


def test_queue_entries_carry_hours_in_their_notes_and_a_running_total():
    q = wq.build_queue()
    total = 0.0
    for it in q:
        total += it["est_hours"]
        assert f"est {it['est_hours']:.1f} h" in it["note"]
        assert it["est_basis"] in ("measured", "scaled", "nominal")
        assert abs(it["cum_hours"] - total) < 0.02
        assert it["beyond_budget"] == (it["cum_hours"] > wq.BUDGET_H)


def test_r2_jobs_use_the_v1_protocol_five_seed_plan_and_validation_gates():
    q = {it["id"]: it["cmd"] for it in wq.build_queue()}
    for jid in ("t2s_wk_r2_real_cpu", "t2s_wk_r2_real_igpu"):
        c = q[jid]
        assert c[c.index("--call2-tools") + 1] == "off"
        assert c[c.index("--plan") + 1] == "strong"   # SEEDS_STRONG: 5 seeds x 40 turns
        assert c[c.index("--tiers") + 1] == "ollama_default"
        assert "--require-validation-gates" in c and "--per-model-refusal" in c
    assert q["t2s_wk_r2_real_igpu"][q["t2s_wk_r2_real_igpu"].index("--runtime-config") + 1] == "igpu_enable"
    assert q["t2s_wk_r2_real_cpu"][q["t2s_wk_r2_real_cpu"].index("--runtime-config") + 1] == "cpu_default"
    m = q["t2s_wk_r2_mitigation_cpu"]
    assert m[m.index("--mode") + 1] == "mitigation" and "--client-trim" in m
    assert m[m.index("--tiers") + 1] == "ollama_ctx_4096"


def test_outcome_job_uses_the_fixed_subset_canary_gate_and_two_models():
    c = next(it["cmd"] for it in wq.build_queue() if it["id"] == "t2s_wk_outcome_subset")
    assert c[1] == "t2s_outcome_table.py"
    assert c[c.index("--models") + 1] == "llama3.1:8b,qwen3-8b"
    assert c[c.index("--subset-n") + 1] == "100"
    import x2_outcome_table as x2ot
    assert c[c.index("--subset-seed") + 1] == str(x2ot.QWEN32B_SUBSET_SEED)
    assert "--canary-gate" in c


def test_px2i_job_selects_the_new_phase():
    c = next(it["cmd"] for it in wq.build_queue() if it["id"] == "t2s_wk_px2i")
    assert c[1] == "t2s_night2.py" and c[c.index("--phases") + 1] == "px2i"


# ------------------------------------------------------------------ estimates
def test_estimate_inputs_match_the_result_files():
    got = wq.inputs_from_files()
    for k, (v, _src, _how) in wq.ESTIMATE_INPUTS.items():
        assert got[k] == pytest.approx(v, rel=0.01), k


def test_cpu_factor_is_the_measured_per_call_ratio():
    assert wq.cpu_factor() == pytest.approx(49.27 / (6.95 * 60 / 80))


# ------------------------------------------------------------------ dry run
def test_dry_run_passes_and_reports_blocked_r2_entries():
    rep = wq.dry_run()
    assert rep["ok"], [e for e in rep["entries"] if not e["ok"]]
    assert rep["fresh_build_matches_file"]
    assert rep["total_hours"] == pytest.approx(sum(it["est_hours"] for it in wq.build_queue()), abs=0.1)
    assert all(e["parse_ok"] for e in rep["entries"])
    r2_ids = {it["id"] for it in wq.build_queue() if it["cmd"][1] == "t2s_r2_agent.py"}
    if any(e.get("pending_capabilities") for e in rep["entries"]):
        assert set(rep["blocked_until_x2_r2_agent_changes"]) <= r2_ids


def _item(**kw):
    base = {"id": "t2s_wk_x", "cmd": [wq.T2S_PY, "t2s_week_preflight.py"], "status": "pending"}
    base.update(kw)
    return base


def test_dry_run_catches_a_wrong_host_interpreter_and_an_x2_output_name():
    bad = _item(cmd=[wq.host_python("EVO-X2"), "t2s_outcome_table.py",
                     "--out", "results/x2_outcome_table_v3.jsonl"])
    res = wq.check_entry(bad, set(), {"x2_outcome_table_v3.jsonl"})
    text = " ".join(res["problems"])
    assert not res["ok"]
    assert "interpreter" in text and "ritz" in text and "collides" in text and "t2s_-prefixed" in text


def test_dry_run_catches_a_bad_flag_an_unknown_phase_and_a_forward_gate():
    r = wq.check_entry(_item(cmd=[wq.T2S_PY, "t2s_outcome_table.py", "--out", "results/t2s_x_new.jsonl",
                                  "--no-such-flag"]), set(), set())
    assert not r["ok"] and not r["parse_ok"]
    r = wq.check_entry(_item(cmd=[wq.T2S_PY, "t2s_night2.py", "--expect-blobs", "e.json", "--phases", "nope"]),
                       set(), set())
    assert any("unknown night2 phase" in p for p in r["problems"])
    r = wq.check_entry(_item(gate={"requires_done": "t2s_wk_later"}), set(), set())
    assert any("not an earlier entry" in p for p in r["problems"])


# ------------------------------------------------------------------ load (fake t2s_queue)
def _fake_tq(existing):
    state = {"items": [dict(x) for x in existing]}
    return types.SimpleNamespace(read_queue=lambda: [dict(x) for x in state["items"]],
                                 write_queue=lambda items: state.update(items=items)), state


def test_load_appends_pending_entries_after_the_existing_queue(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(wq.build_queue()), encoding="utf-8")
    tq, state = _fake_tq([{"id": "old", "cmd": ["x"], "status": "done"}])
    ids = wq.load(p, tq=tq)
    assert ids == [it["id"] for it in wq.build_queue()]
    assert state["items"][0] == {"id": "old", "cmd": ["x"], "status": "done"}
    assert all(it["status"] == "pending" for it in state["items"][1:])


def test_load_refuses_ids_already_in_the_queue_and_writes_nothing(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(wq.build_queue()), encoding="utf-8")
    tq, state = _fake_tq([{"id": "t2s_wk_preflight", "cmd": ["x"], "status": "running"}])
    with pytest.raises(SystemExit):
        wq.load(p, tq=tq)
    assert len(state["items"]) == 1


# ------------------------------------------------------------------ preflight
MODELS_DIR = r"D:\fake\.ollama\models"


class FakeProbe(pf.Probe):
    def __init__(self, files=(), ollama="ollama version is 0.34.4", llama="version: 10970 (bfdc3218)",
                 user="No User exists for *", host="EVO-T2S"):
        self.files, self.ollama, self.llama, self.user, self.host = set(map(str, files)), ollama, llama, user, host

    def hostname(self):
        return self.host

    def query_user(self):
        return self.user

    def exists(self, path):
        return str(path) in self.files

    def read_text(self, path):
        return json.dumps({"ts": "2026-10-06"})

    def ollama_exe(self):
        return r"C:\apu\bin\ollama-0.34.4\ollama.exe"

    def ollama_models_dir(self):
        return MODELS_DIR

    def run(self, cmd):
        return self.ollama if "ollama" in cmd[0] else self.llama

    def reboot_pending(self):
        return False

    def persistent_igpu_env(self):
        return {"hkcu": None, "hklm": None}

    def x2_missing(self):
        return []


def _all_files(deploy):
    files = [Path(deploy) / pf.HANDOVER_FILE.name]
    files += [pf.LLAMA_BIN_DIR / t for t in ("llama-server.exe", "llama-tokenize.exe")]
    files += [pf.manifest_path(t, MODELS_DIR) for t in pf.OLLAMA_TAGS]
    files += [pf.MODELS_DIR / g for g in pf.GGUFS]
    files += [Path(deploy) / f for f in pf.DEPLOYED_FILES] + [Path(deploy) / f for f in pf.DATA_FILES]
    return files


def test_preflight_advances_once(tmp_path):
    notes = []
    note = pf.main(probe=FakeProbe(_all_files(tmp_path)), tq=types.SimpleNamespace(advance=notes.append),
                   deploy=tmp_path)
    assert note == "completed" and notes == ["completed"]
    rows = [json.loads(x) for x in (tmp_path / pf.OUT_REL).read_text().splitlines()]
    assert rows[-1]["record"] == "t2s_preflight_summary" and rows[-1]["ok"] is True


def test_preflight_fails_on_a_different_ollama_version_and_still_advances_once(tmp_path):
    notes = []
    note = pf.main(probe=FakeProbe(_all_files(tmp_path), ollama="ollama version is 0.33.2"),
                   tq=types.SimpleNamespace(advance=notes.append), deploy=tmp_path)
    assert note.startswith("stopped: preflight failed: ollama_version") and notes == [note]


def test_preflight_fails_without_the_handover_file_or_with_a_user_logged_in(tmp_path):
    files = [f for f in _all_files(tmp_path) if f.name != pf.HANDOVER_FILE.name]
    checks = pf.run_checks(FakeProbe(files, user="USERNAME SESSIONNAME\n>zach console 1 Active"), deploy=tmp_path)
    failed = {c["check"] for c in checks if c["hard"] and not c["ok"]}
    assert failed == {"handover_file", "interactive_session"}


def test_preflight_soft_checks_never_fail_the_job(tmp_path):
    class P(FakeProbe):
        def reboot_pending(self):
            return True

        def x2_missing(self):
            return ["x2_r2_agent.py has no --tiers flag"]
    ok, note = pf.summarize(pf.run_checks(P(_all_files(tmp_path)), deploy=tmp_path))
    assert ok and "reboot_pending" in note and "x2_r2_agent_capabilities" in note


def test_version_parsers():
    assert pf.parse_ollama_version("ollama version is 0.34.4") == "0.34.4"
    assert pf.parse_ollama_version("Warning: could not connect\nWarning: client version is 0.33.2") == "0.33.2"
    assert pf.parse_llama_build("ggml_vulkan: ...\nversion: 10970 (bfdc3218)\nbuilt with ...") == "10970"
    assert pf.parse_llama_build("") is None


def test_apply_ollama_pin(tmp_path):
    exe = tmp_path / "ollama.exe"
    exe.write_text("")
    pin = tmp_path / "pin.json"
    env = {}
    assert pf.apply_ollama_pin(pin, env)["pinned"] is False and env == {}
    pin.write_text(json.dumps({"ollama_bin": str(exe), "version": "0.34.4"}))
    assert pf.apply_ollama_pin(pin, env)["pinned"] is True and env["OLLAMA_BIN"] == str(exe)
    pin.write_text(json.dumps({"ollama_bin": str(tmp_path / "gone.exe")}))
    env2 = {}
    assert pf.apply_ollama_pin(pin, env2)["pinned"] is False and env2 == {}


def test_preflight_deployed_files_exist_in_harness():
    for f in pf.DEPLOYED_FILES:
        assert (REPO / "harness" / f).exists(), f


def test_queue_script_spawns_nothing_on_the_controller():
    src = (REPO / "scripts" / "t2s_week_queue.py").read_text(encoding="utf-8")
    assert "subprocess" not in src and "ssh" not in src.lower().replace("no ssh", "")


def test_t2s_host_entry_keeps_the_interactive_guard_and_sharc_paths():
    import host_config as hc
    t = hc.HOSTS["EVO-T2S"]
    assert t["interactive_guard"] is True and t["user"] == "sharc" and t["gpu_vendor"] == "intel"
    assert "sharc" in wq.T2S_PY.lower() and r"Python312\python.exe" in wq.T2S_PY


# ------------------------------------------------------------------ night2 exit
def test_night2_main_advances_once():
    import t2s_night2 as n2
    src = inspect.getsource(n2.main)
    assert src.count("tq.advance(note)") == 1
    assert src.rindex("finally:") < src.index("tq.advance(note)")
