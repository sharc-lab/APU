"""harness/t2s_r2_agent.py (the evo-t2s wrapper around x2_r2_agent.py) and harness/argparse_probe.py. A fake
x2_r2_agent module stands in for the real one wherever main() would run; nothing starts Ollama."""
import argparse
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import argparse_probe  # noqa: E402
import t2s_r2_agent as w  # noqa: E402
import x2_r2_agent as x2  # noqa: E402


def _fake_x2(flags=("--server-env", "--tiers", "--seeds", "--client-trim"), modes=("validation", "real", "mitigation",
                                                                                     "mechanism"), note="completed",
             rows=None):
    calls = []

    def main(argv=None, advance=True):
        ap = argparse.ArgumentParser()
        ap.add_argument("--mode", choices=modes, required=True)
        ap.add_argument("--out", required=True)
        for f in ("--models", "--plan", "--call2-tools", "--rules-from", "--order", *flags):
            ap.add_argument(f, action="append" if f == "--server-env" else "store")
        ap.add_argument("--require-validation-gates", action="store_true")
        ap.add_argument("--per-model-refusal", action="store_true")
        args = ap.parse_args(argv)
        calls.append((list(argv), advance))
        if rows:
            with open(args.out, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        return note
    return types.SimpleNamespace(main=main), calls


class FakeHC:
    def __init__(self, logged_in=False):
        self.logged_in, self.stops = logged_in, 0

    def require_host(self, h):
        return {"hw_id": "evo-t2s", "interactive_guard": True}

    def enforce_or_record_interactive_session(self, cfg):
        if self.logged_in:
            raise SystemExit("another interactive session is logged in: zach")
        return {}

    def stop_ollama_server(self):
        self.stops += 1


def _turn(model, loaded, arm="ollama_default_call2_notools"):
    return {"record": "r2a_turn", "model_id": model, "arm_id": arm, "loaded_context": loaded}


def test_forwarded_argv_adds_igpu_env_and_keeps_unknown_flags():
    args, rest = w.build_parser().parse_known_args(
        ["--mode", "real", "--runtime-config", "igpu_enable", "--out", "results/t2s_a.jsonl", "--tiers",
         "ollama_default", "--plan", "strong", "--require-validation-gates"])
    fwd = w.forwarded_argv(args, rest)
    assert fwd[:4] == ["--mode", "real", "--out", "results/t2s_a.jsonl"]
    assert ["--server-env", "OLLAMA_IGPU_ENABLE=1"] == fwd[fwd.index("--server-env"):fwd.index("--server-env") + 2]
    assert fwd[-3:] == ["--plan", "strong", "--require-validation-gates"]
    args, rest = w.build_parser().parse_known_args(["--mode", "validation", "--runtime-config", "cpu_default",
                                                    "--out", "results/t2s_b.jsonl"])
    assert "--server-env" not in w.forwarded_argv(args, rest)


def test_capabilities_of_the_real_x2_r2_agent_are_read_without_running_it():
    caps = w.x2_capabilities(x2)
    assert caps["parsed"] and "--mode" in caps["flags"] and "validation" in caps["modes"]
    if not caps["host_generic"]:
        assert w.HOST_GUARD_LITERAL in __import__("inspect").getsource(x2.main)


def test_missing_capabilities_lists_flags_modes_and_the_host_guard():
    caps = {"host_generic": False, "flags": ["--mode", "--out"], "modes": ["validation"]}
    miss = w.missing_capabilities(caps, ["--mode", "mitigation", "--out", "o", "--tiers", "x"])
    assert any("EVO-X2" in m for m in miss) and any("--tiers" in m for m in miss)
    assert any("'mitigation'" in m for m in miss)
    caps = {"host_generic": True, "flags": ["--mode", "--out", "--tiers"], "modes": ["mitigation"]}
    assert w.missing_capabilities(caps, ["--mode", "mitigation", "--out", "o", "--tiers", "x"]) == []


def test_strip_unsupported_drops_flags_and_their_values_only():
    out = w.strip_unsupported(["--mode", "real", "--tiers", "a", "--flag", "--out", "o"], {"--mode", "--out"})
    assert out == ["--mode", "real", "--out", "o"]


def test_dry_run_against_the_real_parser_reports_pending_changes():
    args, rest = w.build_parser().parse_known_args(
        ["--mode", "mitigation", "--runtime-config", "cpu_default", "--out", "results/t2s_m.jsonl",
         "--client-trim", "margin=0.05", "--call2-tools", "off", "--dry-run"])
    rep = w.dry_run(args, rest, x2)
    assert rep["x2_parse_ok"] is True
    assert rep["out_name_problem"] is None


def test_out_name_must_be_t2s_prefixed():
    assert w.check_out_name("results/t2s_r2_x.jsonl") is None
    assert "t2s_" in w.check_out_name("results/x2_r2_real_v1.jsonl")


def test_main_advances_exactly_once_on_success_and_never_lets_x2_advance(tmp_path):
    out = tmp_path / "t2s_r2_real.jsonl"
    fake, calls = _fake_x2(rows=[_turn("llama3.1:8b", 32768), _turn("qwen3:14b", 32768)])
    notes, hc = [], FakeHC()
    note = w.main(["--mode", "real", "--runtime-config", "igpu_enable", "--out", str(out), "--tiers", "ollama_default"],
                  x2mod=fake, hc=hc, tq=types.SimpleNamespace(advance=notes.append), hostname="evo-t2s")
    assert note == "completed" and notes == ["completed"]
    assert calls[0][1] is False                       # x2_r2_agent.main(..., advance=False)
    assert "OLLAMA_IGPU_ENABLE=1" in calls[0][0] and hc.stops == 1
    chk = [json.loads(x) for x in out.read_text().splitlines() if "t2s_runtime_check" in x]
    assert chk and chk[0]["ok"] is True and chk[0]["expected_default_ctx"] == 32768


def test_main_advances_exactly_once_with_a_runtime_mismatch_note(tmp_path):
    out = tmp_path / "t2s_r2_real.jsonl"
    fake, _ = _fake_x2(rows=[_turn("llama3.1:8b", 4096)])
    notes = []
    note = w.main(["--mode", "real", "--runtime-config", "igpu_enable", "--out", str(out)], x2mod=fake, hc=FakeHC(),
                  tq=types.SimpleNamespace(advance=notes.append), hostname="EVO-T2S")
    assert "runtime check" in note and notes == [note] and not note.startswith("stopped")


def test_main_advances_exactly_once_when_x2_lacks_capabilities(tmp_path):
    fake, calls = _fake_x2(flags=())
    notes = []
    note = w.main(["--mode", "real", "--runtime-config", "igpu_enable", "--out", str(tmp_path / "t2s_r.jsonl"),
                   "--tiers", "ollama_default"], x2mod=fake, hc=FakeHC(),
                  tq=types.SimpleNamespace(advance=notes.append), hostname="EVO-T2S")
    assert note.startswith("stopped: deployed x2_r2_agent.py cannot run") and notes == [note]
    assert "--server-env" in note and calls == []


def test_main_advances_exactly_once_on_the_wrong_host_and_on_a_logged_in_user(tmp_path):
    fake, calls = _fake_x2()
    notes = []
    note = w.main(["--mode", "validation", "--runtime-config", "cpu_default", "--out", str(tmp_path / "t2s_v.jsonl")],
                  x2mod=fake, hc=FakeHC(), tq=types.SimpleNamespace(advance=notes.append), hostname="EVO-X2")
    assert note.startswith("stopped:") and "EVO-T2S only" in note and notes == [note]
    notes.clear()
    note = w.main(["--mode", "validation", "--runtime-config", "cpu_default", "--out", str(tmp_path / "t2s_v.jsonl")],
                  x2mod=fake, hc=FakeHC(logged_in=True), tq=types.SimpleNamespace(advance=notes.append),
                  hostname="EVO-T2S")
    assert "another interactive session" in note and notes == [note] and calls == []


def test_runtime_check_ignores_other_arms_and_records_per_model():
    rows = [_turn("llama3.1:8b", 4096), _turn("qwen3:14b", 4096), _turn("llama3.1:8b", 131072, "ollama_ctx_131072")]
    chk = w.runtime_check(rows, "cpu_default")
    assert chk["ok"] and chk["loaded_context_by_model"] == {"llama3.1:8b": [4096], "qwen3:14b": [4096]}
    assert w.runtime_check([], "cpu_default")["applicable"] is False


# ------------------------------------------------------------------ argparse_probe
def test_probe_parses_without_running_main():
    ran = []

    def main(argv=None):
        ap = argparse.ArgumentParser()
        ap.add_argument("--x", choices=("a", "b"))
        ap.parse_args(argv)
        ran.append(True)
    ok = argparse_probe.probe(main, ["--x", "a"])
    assert ok["ok"] and ok["namespace"] == {"x": "a"} and ok["options"]["--x"] == ("a", "b") and not ran
    bad = argparse_probe.probe(main, ["--x", "c"])
    assert not bad["ok"] and "invalid choice" in bad["error"] and not ran


def test_probe_with_sys_argv_restores_it():
    saved = list(sys.argv)

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--y")
        ap.parse_args()
    assert argparse_probe.probe(main, ["--y", "1"], use_sys_argv=True)["namespace"] == {"y": "1"}
    assert sys.argv == saved
