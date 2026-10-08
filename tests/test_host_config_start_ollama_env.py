"""host_config.start_ollama_server's optional env and log_path (2026-10-08, for x2_r2_agent's mechanism job, which
needs OLLAMA_DEBUG=1 on the server process and the server log in a file of its own). The command line is captured
through ps_fn; nothing is launched."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import host_config as hc  # noqa: E402


@pytest.fixture
def captured(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setattr(hc, "_resolve_ollama_exe_for_serve", lambda: r"C:\x\ollama.exe")
    monkeypatch.setattr(hc, "ollama_process_running", lambda ps_fn=None: False)
    monkeypatch.setattr(hc, "_this_host_entry", lambda: {"ollama_models": r"D:\store\models"})
    got = {}

    def ps_fn(cmd, timeout):
        got["cmd"] = cmd
        return "pid=4321"
    got["ps_fn"] = ps_fn
    return got


def test_default_command_line_is_unchanged(captured):
    assert hc.start_ollama_server(ps_fn=captured["ps_fn"]) == 4321
    assert captured["cmd"].startswith(
        "$cmd = 'cmd.exe /c set OLLAMA_KEEP_ALIVE=0 && set OLLAMA_MODELS=D:\\store\\models&& "
        "\"C:\\x\\ollama.exe\" serve > C:\\apu\\ovn\\ollama_serve.log 2>&1'; ")


def test_env_and_log_path(captured):
    hc.start_ollama_server(ps_fn=captured["ps_fn"], env={"OLLAMA_DEBUG": "1"},
                           log_path=r"C:\apu\ovn\ollama_serve_mechanism.log")
    cmd = captured["cmd"]
    assert 'set "OLLAMA_DEBUG=1"&& "C:\\x\\ollama.exe" serve' in cmd  # quoted: no trailing space in the value
    assert "> C:\\apu\\ovn\\ollama_serve_mechanism.log 2>&1'" in cmd
    assert cmd.index("OLLAMA_KEEP_ALIVE=0") < cmd.index("OLLAMA_DEBUG")


@pytest.mark.parametrize("env", [{"ollama_debug": "1"}, {"OLLAMA_DEBUG": "1 & calc"}, {"OLLAMA_DEBUG": "it's"},
                                 {"OLLAMA_KEEP_ALIVE": "5m"}, {"OLLAMA_MODELS": "x"}])
def test_rejects_unsafe_or_reserved_env(captured, env):
    with pytest.raises(ValueError):
        hc.start_ollama_server(ps_fn=captured["ps_fn"], env=env)
    assert "cmd" not in captured


def test_rejects_log_path_with_space(captured):
    with pytest.raises(ValueError):
        hc.start_ollama_server(ps_fn=captured["ps_fn"], log_path=r"C:\a b\x.log")


def test_still_idempotent_when_running(monkeypatch):
    monkeypatch.setattr(hc, "ollama_process_running", lambda ps_fn=None: True)
    assert hc.start_ollama_server(ps_fn=lambda c, t: "pid=1", env={"OLLAMA_DEBUG": "1"}) is None
