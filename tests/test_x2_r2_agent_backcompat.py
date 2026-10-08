"""Backward compatibility of harness/x2_r2_agent.py after the T2S changes (2026-10-08: host allowlist plus
interactive-session guard, --server-env, --tiers, --seeds / --allow-new-seeds).

tests/golden/x2_r2_agent_defaults_087e790.json was written by this file's __main__ block against x2_r2_agent.py as it
was at 087e790 (before those changes): for every mode (validation, real, strong, mitigation, mechanism,
toolchoice_check) and for the two reconstructed evo-x2 queue commands (x2_r2_real_v1b, x2_r2_mitigation_v1), the
parsed argv, the Ollama server start/stop calls, the run_start and runtime records and the exit note, captured by
running main() with fakes (no network, no process launch; the run stops with a sentinel at the first session).
The tests below run the same capture against the current module and require it to be identical, except for the new
run_start fields, which must be exactly NEW_RUN_START_FIELDS (hw_id and interactive_session from the host checks,
the rest null when the new flags are not given).

Regenerate only from the pre-change module: `git show 087e790:harness/x2_r2_agent.py` into a scratch harness copy,
then `py -3.12 tests/test_x2_r2_agent_backcompat.py <that harness dir> <out json>`.
"""
from __future__ import annotations

import importlib
import json
import socket
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "x2_r2_agent_defaults_087e790.json"

# results/ paths as the evo-x2 queue commands name them; the captures read the committed copies.
V2, V2B, V2C = ("results/x2_r2_validation_v2.jsonl", "results/x2_r2_validation_v2b.jsonl",
                "results/x2_r2_validation_v2c.jsonl")
V1 = "results/x2_r2_real_v1.jsonl"

# argv per case; "<OUT>" is replaced by a per-case output path under the capture's tmp dir.
CASES = {
    "validation": ["--mode", "validation", "--out", "<OUT>"],
    "validation_off": ["--mode", "validation", "--call2-tools", "off", "--out", "<OUT>"],
    "real": ["--mode", "real", "--out", "<OUT>"],
    "real_off": ["--mode", "real", "--call2-tools", "off", "--out", "<OUT>"],
    "strong": ["--mode", "real", "--plan", "strong", "--call2-tools", "off", "--out", "<OUT>"],
    "mitigation": ["--mode", "mitigation", "--client-trim", "margin=0.05", "--call2-tools", "off", "--out", "<OUT>"],
    "mechanism": ["--mode", "mechanism", "--call2-tools", "off", "--out", "<OUT>"],
    "toolchoice_check": ["--mode", "toolchoice_check", "--out", "<OUT>"],
    # Reconstructed from docs/STATE.md, docs/R2_DESIGN.md "Strengthened R2" and docs/FINDINGS.md's x2_r2_mitigation_v1
    # pre-registration (the exact queue_state.json lines live on evo-x2 only).
    "queued_x2_r2_real_v1b": ["--mode", "real", "--plan", "strong", "--call2-tools", "off",
                              "--rules-from", f"{V2},{V2B},{V2C}", "--require-validation-gates",
                              "--per-model-refusal", "--skip-done-from", V1, "--order", "seed_major",
                              "--out", "<OUT>"],
    "queued_x2_r2_mitigation_v1": ["--mode", "mitigation", "--client-trim", "margin=0.05", "--call2-tools", "off",
                                   "--rules-from", V2, "--require-validation-gates", "--out", "<OUT>"],
}

CONSOLE_STATE = {"console_session_state": "Active", "console_idle_s": 12.0, "user_active": True}
# run_start fields added by the change, with their value when no new flag is given (on EVO-X2).
NEW_RUN_START_FIELDS = {"hw_id": "evo-x2", "interactive_session": CONSOLE_STATE, "server_env_extra": None,
                        "tiers": None, "seeds": None, "allow_new_seeds": None}
NEW_ARGS = {"server_env": None, "tiers": None, "seeds": None, "allow_new_seeds": False}


class _Stop(Exception):
    """Raised by the stubbed session runners: the capture ends at the first session."""


def _norm(obj, subs):
    if isinstance(obj, dict):
        return {k: _norm(v, subs) for k, v in obj.items() if k != "ts_utc"}
    if isinstance(obj, (list, tuple)):
        return [_norm(v, subs) for v in obj]
    if isinstance(obj, str):
        for a, b in subs:
            obj = obj.replace(a, b)
        return obj
    return obj


def _subs(tmp: Path):
    out = []
    for p, tag in ((tmp, "<TMP>"), (REPO, "<REPO>")):
        out += [(str(p), tag), (str(p).replace("\\", "/"), tag)]
    return out


def _argv(case, tmp: Path):
    out = []
    for a in CASES[case]:
        if a == "<OUT>":
            a = str(tmp / f"{case}.jsonl")
        elif a.startswith("results/") or "," in a and a.split(",")[0].startswith("results/"):
            a = ",".join(str(REPO / p) for p in a.split(","))
        out.append(a)
    return out


def parse_snapshot(ag, case, tmp: Path) -> dict:
    return _norm(vars(ag.build_arg_parser().parse_args(_argv(case, tmp))), _subs(tmp))


def run_snapshot(ag, case, tmp: Path, mp, extra_argv=(), hostname="EVO-X2", logged_in=False) -> dict:
    """main() with fakes for host_config, the queue, the runtime and the session runners. extra_argv, hostname and
    logged_in are for the new-option tests below (the golden uses the defaults)."""
    import x2_r2_mechanism as mech
    tmp.mkdir(parents=True, exist_ok=True)
    events, notes, runtimes = [], [], []
    mech_log = tmp / f"{case}_serve.log"

    def start(*a, **kw):
        events.append({"call": "start_ollama_server", "args": list(a), "kwargs": dict(kw)})
        if kw.get("log_path"):
            Path(kw["log_path"]).write_text('level=INFO msg="server config" env="map[OLLAMA_DEBUG:DEBUG]"\n',
                                            encoding="utf-8")

    def stop(*a, **kw):
        events.append({"call": "stop_ollama_server", "args": list(a), "kwargs": dict(kw)})

    def require_host(h):  # host_config.HOSTS' keys and guard values
        hosts = {"EVO-X2": ("evo-x2", False), "EVO-T2S": ("evo-t2s", True), "RITZLAPTOP": ("blade", False)}
        if h.upper() not in hosts:
            raise SystemExit(f"unknown host {h!r}: not in harness/host_config.py HOSTS, refusing to run")
        return {"hw_id": hosts[h.upper()][0], "interactive_guard": hosts[h.upper()][1]}

    def enforce(cfg):
        if cfg["interactive_guard"] and logged_in:
            raise SystemExit("another interactive session is logged in: USERNAME zach console Active")
        return {} if cfg["interactive_guard"] else dict(CONSOLE_STATE)

    mp.setattr(socket, "gethostname", lambda: hostname)
    mp.setitem(sys.modules, "host_config", types.SimpleNamespace(
        start_ollama_server=start, stop_ollama_server=stop, wait_for_ollama_ready=lambda timeout_s=0: True,
        require_host=require_host, enforce_or_record_interactive_session=enforce))
    mp.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))
    mp.setattr(mech, "MECH_SERVE_LOG", str(mech_log))

    class RT:
        def __init__(self):
            self.server_env = self.server_log = None
            runtimes.append(self)

        def available_models(self):
            return ["llama3.1:8b", "qwen3:14b", "qwen3-4b-2507-tools:latest", "qwen3:8b", "qwen3-4b-2507:latest"]

        def version(self):
            return "0.34.4"

    def stop_run(*a, **kw):
        raise _Stop("first session")

    mp.setattr(ag, "OllamaRuntime", RT)
    for name in ("run_plan", "run_mitigation", "run_toolchoice_check"):
        mp.setattr(ag, name, stop_run)
    argv = _argv(case, tmp) + list(extra_argv)
    note = ag.main(argv)
    rows = ag.read_rows(tmp / f"{case}.jsonl")
    snap = {"note": note, "advance_notes": notes, "server_calls": events,
            "records": [r["record"] for r in rows],
            "run_start": next((r for r in rows if r["record"] == "run_start"), None),
            "runtime": next((r for r in rows if r["record"] == "runtime"), None),
            "runtime_obj": ({"server_env": runtimes[0].server_env, "server_log": runtimes[0].server_log}
                            if runtimes else None)}
    return _norm(json.loads(json.dumps(snap, default=str)), _subs(tmp) + [(str(mech_log), "<MECH_LOG>")])


def build_golden(ag, tmp_root: Path) -> dict:
    out = {}
    for case in CASES:
        with pytest.MonkeyPatch.context() as mp:
            out[case] = {"parse": parse_snapshot(ag, case, tmp_root / case),
                         "run": run_snapshot(ag, case, tmp_root / case, mp)}
    return out


# ── tests ──────────────────────────────────────────────────────────────────────────────────────────

def _current():
    sys.path.insert(0, str(REPO / "harness"))
    import x2_r2_agent as ag
    return ag


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_golden_covers_every_mode(golden):
    modes = {g["parse"]["mode"] for g in golden.values()}
    assert modes == {"validation", "real", "mitigation", "mechanism", "toolchoice_check"}
    assert any(g["parse"]["plan"] == "strong" for g in golden.values())
    assert set(golden) == set(CASES)


@pytest.mark.parametrize("case", list(CASES))
def test_default_argv_parses_identically(case, golden, tmp_path):
    ag = _current()
    new = parse_snapshot(ag, case, tmp_path / case)
    old = golden[case]["parse"]
    assert {k: v for k, v in new.items() if k not in NEW_ARGS} == old
    assert {k: new[k] for k in NEW_ARGS} == NEW_ARGS


@pytest.mark.parametrize("case", list(CASES))
def test_default_run_is_identical(case, golden, tmp_path, monkeypatch):
    ag = _current()
    new = run_snapshot(ag, case, tmp_path / case, monkeypatch)
    old = golden[case]["run"]
    assert new["note"] == old["note"] and new["advance_notes"] == old["advance_notes"]
    assert new["server_calls"] == old["server_calls"]
    assert new["records"] == old["records"]
    assert new["runtime"] == old["runtime"] and new["runtime_obj"] == old["runtime_obj"]
    rs_new, rs_old = new["run_start"], old["run_start"]
    assert {k: v for k, v in rs_new.items() if k not in NEW_RUN_START_FIELDS} == rs_old
    assert {k: rs_new[k] for k in NEW_RUN_START_FIELDS} == NEW_RUN_START_FIELDS
    assert rs_new["plan"] == rs_old["plan"]  # spelled out: the plan is the part the T2S flags change


def test_golden_reached_the_session_runner(golden):
    """Every capture got past the host check, the server start and the runtime record (so the comparison above
    covers them), and stopped at the first session."""
    for case, g in golden.items():
        assert g["run"]["note"] == "stopped: _Stop('first session')", case
        recs = g["run"]["records"]
        assert recs[0] == "run_start" and recs[-1] == "runtime", case
        assert set(recs[1:-1]) <= {"refused_model"}, case
        assert any(c["call"] == "start_ollama_server" for c in g["run"]["server_calls"]), case


# ── the new options (not part of the golden: each is off by default) ──────────────────────────────────

def test_resolve_tier_accepts_arm_ids_and_short_forms():
    ag = _current()
    assert ag.resolve_tier("ollama_ctx_4096") == "ollama_ctx_4096" == ag.resolve_tier("4096")
    assert ag.resolve_tier("default") == "ollama_default" == ag.resolve_tier("ollama_default")
    for bad in ("4097", "ctx_4096", "", "ollama_ctx_4096_call2_notools"):
        with pytest.raises(ValueError):
            ag.resolve_tier(bad)
    assert ag.base_arm("ollama_ctx_4096_call2_notools") == "ollama_ctx_4096"
    assert ag.base_arm("ollama_default_call2_forcednone_format") == "ollama_default"


def test_restrict_plan_tiers_and_seeds():
    ag = _current()
    strong = ag.strong_plan("off")
    assert ag.restrict_plan(strong) == strong
    p = ag.restrict_plan(strong, tiers=["ollama_default"])
    assert p == [("ollama_default_call2_notools", ag.SEEDS_STRONG, 40)]
    p = ag.restrict_plan(strong, tiers=["ollama_ctx_4096", "ollama_default"], seeds=[20260905, 20260901])
    assert p == [("ollama_default_call2_notools", (20260901, 20260905), 40),
                 ("ollama_ctx_4096_call2_notools", (20260901, 20260905), 40)]   # the plan's order, not the flag's
    with pytest.raises(ValueError, match="not in this mode's plan"):
        ag.restrict_plan(ag.mitigation_plan("off"), tiers=["ollama_default"])
    with pytest.raises(ValueError, match="allow-new-seeds"):
        ag.restrict_plan(ag.mitigation_plan("off"), seeds=[20260904])
    p = ag.restrict_plan(ag.mitigation_plan("off"), tiers=["ollama_ctx_4096"], seeds=[20260904],
                         allow_new_seeds=True)
    assert p == [("ollama_ctx_4096_call2_notools", (20260904,), 40)]


@pytest.mark.parametrize("extra,msg", [
    (["--tiers", "4097"], "unknown tier"),
    (["--tiers", "ollama_ctx_16384"], "not in this mode's plan"),      # mitigation plan: 4096 and 8192 only
    (["--tiers", "4096,ollama_ctx_4096"], "twice"),
    (["--seeds", "20260904"], "allow-new-seeds"),
    (["--seeds", "x"], "integers"),
    (["--allow-new-seeds"], "needs --seeds"),
    (["--server-env", "ollama_igpu_enable=1"], "KEY=VALUE"),
    (["--server-env", "OLLAMA_MODELS=C:/m"], "start_ollama_server itself"),
    (["--server-env", "OLLAMA_DEBUG=0"], "conflicts"),
])
def test_bad_new_options_are_parse_errors_before_any_server_call(extra, msg, tmp_path, monkeypatch, capsys):
    ag = _current()
    with pytest.raises(SystemExit) as ei:
        run_snapshot(ag, "mitigation", tmp_path, monkeypatch, extra_argv=extra)
    assert ei.value.code == 2 and msg in capsys.readouterr().err
    assert not (tmp_path / "mitigation.jsonl").exists()


def test_tiers_and_seeds_rejected_for_validation_and_toolchoice(tmp_path, monkeypatch, capsys):
    ag = _current()
    for case in ("validation", "toolchoice_check"):
        with pytest.raises(SystemExit):
            run_snapshot(ag, case, tmp_path / case, monkeypatch, extra_argv=["--tiers", "default"])
        assert "apply to --mode" in capsys.readouterr().err


def test_tiers_seeds_restrict_and_are_recorded(tmp_path, monkeypatch):
    ag = _current()
    snap = run_snapshot(ag, "strong", tmp_path, monkeypatch, extra_argv=["--tiers", "default,4096",
                                                                         "--seeds", "20260901,20260902"])
    rs = snap["run_start"]
    assert rs["plan"] == [["ollama_default_call2_notools", [20260901, 20260902], 40],
                          ["ollama_ctx_4096_call2_notools", [20260901, 20260902], 40]]
    assert rs["tiers"] == ["ollama_default", "ollama_ctx_4096"] and rs["seeds"] == [20260901, 20260902]
    assert rs["allow_new_seeds"] is False and rs["server_env_extra"] is None


def test_server_env_plain_mode_stops_first_then_starts_with_env(tmp_path, monkeypatch):
    ag = _current()
    snap = run_snapshot(ag, "strong", tmp_path, monkeypatch, extra_argv=["--server-env", "OLLAMA_IGPU_ENABLE=1",
                                                                         "--tiers", "ollama_default"])
    calls = snap["server_calls"]
    assert [c["call"] for c in calls] == ["stop_ollama_server", "start_ollama_server", "stop_ollama_server"]
    assert calls[1]["kwargs"] == {"env": {"OLLAMA_IGPU_ENABLE": "1"}, "log_path": None}
    assert snap["run_start"]["server_env_extra"] == {"OLLAMA_IGPU_ENABLE": "1"}
    assert snap["runtime"]["server_env"] == {"OLLAMA_IGPU_ENABLE": "1"}
    assert snap["runtime_obj"]["server_env"] == {"OLLAMA_IGPU_ENABLE": "1"}   # what recover() restarts with


def test_server_env_merges_with_the_debug_env(tmp_path, monkeypatch):
    ag = _current()
    snap = run_snapshot(ag, "mitigation", tmp_path, monkeypatch,
                        extra_argv=["--server-env", "OLLAMA_IGPU_ENABLE=1", "--tiers", "4096"])
    calls = snap["server_calls"]
    assert [c["call"] for c in calls] == ["stop_ollama_server", "start_ollama_server", "stop_ollama_server"]
    assert calls[1]["kwargs"]["env"] == {"OLLAMA_DEBUG": "1", "OLLAMA_IGPU_ENABLE": "1"}
    assert snap["runtime"]["server_env"] == {"OLLAMA_DEBUG": "1", "OLLAMA_IGPU_ENABLE": "1"}


def test_recover_restarts_with_the_server_env(monkeypatch):
    ag = _current()
    calls = []
    ready = iter([False, True])
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
        wait_for_ollama_ready=lambda timeout_s=0: next(ready), stop_ollama_server=lambda: calls.append("stop"),
        start_ollama_server=lambda env=None, log_path=None: calls.append(("start", env, log_path))))
    rt = ag.OllamaRuntime.__new__(ag.OllamaRuntime)
    rt.server_env, rt.server_log = {"OLLAMA_IGPU_ENABLE": "1"}, None
    assert rt.recover() is True
    assert calls == ["stop", ("start", {"OLLAMA_IGPU_ENABLE": "1"}, None)]


def test_evo_t2s_runs_and_records_its_hw_id(tmp_path, monkeypatch):
    ag = _current()
    snap = run_snapshot(ag, "mechanism", tmp_path, monkeypatch, hostname="EVO-T2S")
    assert snap["note"] == "stopped: _Stop('first session')"
    assert snap["run_start"]["hw_id"] == "evo-t2s" and snap["run_start"]["interactive_session"] == {}


def test_evo_t2s_refuses_while_someone_is_logged_in(tmp_path, monkeypatch):
    ag = _current()
    snap = run_snapshot(ag, "real", tmp_path, monkeypatch, hostname="EVO-T2S", logged_in=True)
    assert snap["note"].startswith("stopped: another interactive session")
    assert snap["advance_notes"] == [snap["note"]] and snap["server_calls"] == [] and snap["records"] == []


@pytest.mark.parametrize("host,msg", [("SOMEBOX", "unknown host"), ("RITZLAPTOP", "is not one of")])
def test_other_hosts_are_refused_without_a_server_call(host, msg, tmp_path, monkeypatch):
    ag = _current()
    snap = run_snapshot(ag, "real", tmp_path, monkeypatch, hostname=host)
    assert snap["note"].startswith("stopped: RuntimeError(") and msg in snap["note"]
    assert snap["advance_notes"] == [snap["note"]] and snap["server_calls"] == [] and snap["records"] == []


def test_t2s_wrapper_sees_every_capability():
    """harness/t2s_r2_agent.py's own check of the x2_r2_agent it would run: nothing missing now."""
    ag = _current()
    import t2s_r2_agent as w
    caps = w.x2_capabilities(ag)
    assert caps["host_generic"] and {"--server-env", "--tiers", "--seeds"} <= set(caps["flags"])
    fwd = ["--mode", "mitigation", "--out", "t2s_x.jsonl", "--tiers", "ollama_ctx_4096",
           "--server-env", "OLLAMA_IGPU_ENABLE=1", "--seeds", "20260901"]
    assert w.missing_capabilities(caps, fwd) == []


if __name__ == "__main__":
    harness_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(harness_dir))
    sys.path.insert(1, str(REPO / "harness"))
    mod = importlib.import_module("x2_r2_agent")
    assert Path(mod.__file__).resolve().parent == harness_dir.resolve(), mod.__file__
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        data = build_golden(mod, Path(td))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_json} ({len(data)} cases)")
