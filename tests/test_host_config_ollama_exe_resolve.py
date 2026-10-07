"""Unit tests for host_config._resolve_ollama_exe_for_serve (added 2026-10-01, T2S K1 v3 retry investigation):
found live that start_ollama_server's own WMI-launched "ollama serve" used the bare command name, which never
resolves on evo-t2s specifically (confirmed: `where ollama` also fails there in a plain interactive SSH
session), so the server never actually started and every subsequent pull/chat failed with connection-refused,
not a timeout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import host_config as hc  # noqa: E402


def test_resolve_prefers_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_BIN", r"C:\explicit\ollama.exe")
    assert hc._resolve_ollama_exe_for_serve() == r"C:\explicit\ollama.exe"


def test_resolve_falls_back_to_which(monkeypatch):
    import shutil
    monkeypatch.delenv("OLLAMA_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\which\ollama.exe" if name == "ollama" else None)
    assert hc._resolve_ollama_exe_for_serve() == r"C:\which\ollama.exe"


def test_resolve_falls_back_to_localappdata_when_which_fails(tmp_path, monkeypatch):
    import shutil
    monkeypatch.delenv("OLLAMA_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    ollama_dir = tmp_path / "Programs" / "Ollama"
    ollama_dir.mkdir(parents=True)
    exe_path = ollama_dir / "ollama.exe"
    exe_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert hc._resolve_ollama_exe_for_serve() == str(exe_path)


def test_resolve_bare_command_as_last_resort(tmp_path, monkeypatch):
    import shutil
    monkeypatch.delenv("OLLAMA_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))  # exists but no Programs/Ollama/ollama.exe inside it
    assert hc._resolve_ollama_exe_for_serve() == "ollama"


def test_start_ollama_server_quotes_the_resolved_exe_in_the_cmdline(monkeypatch):
    """The resolved exe path is embedded in a PowerShell string that is itself embedded in a cmd.exe command
    line -- found live elsewhere this session (t2s_queue.py's trailing-space bug) that this kind of nested
    quoting is exactly where real bugs hide. The exe path must be double-quoted in the cmd.exe line so a
    path containing a space still works."""
    monkeypatch.setattr(hc, "_resolve_ollama_exe_for_serve", lambda: r"C:\Program Files\Ollama\ollama.exe")
    monkeypatch.setattr(hc, "ollama_process_running", lambda ps_fn=None: False)
    captured = {}

    def fake_ps_fn(cmd, timeout):
        captured["cmd"] = cmd
        return "pid=1234"

    hc.start_ollama_server(ps_fn=fake_ps_fn)
    assert '"C:\\Program Files\\Ollama\\ollama.exe"' in captured["cmd"]


def test_resolve_uses_host_pinned_exe_when_localappdata_is_system_profile(tmp_path, monkeypatch):
    """2026-10-06, evo-x2: the queue watchdog runs as SYSTEM, so LOCALAPPDATA is the system profile and the
    per-user install is not under it. The host's pinned ollama_exe must win over the bare-name fallback."""
    import shutil
    monkeypatch.delenv("OLLAMA_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "systemprofile"))
    pinned = tmp_path / "ollama.exe"
    pinned.write_text("", encoding="utf-8")
    monkeypatch.setattr(hc, "_this_host_entry", lambda: {"ollama_exe": str(pinned)})
    assert hc._resolve_ollama_exe_for_serve() == str(pinned)


def test_start_ollama_server_sets_host_models_dir(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setattr(hc, "_resolve_ollama_exe_for_serve", lambda: r"C:\x\ollama.exe")
    monkeypatch.setattr(hc, "ollama_process_running", lambda ps_fn=None: False)
    monkeypatch.setattr(hc, "_this_host_entry", lambda: {"ollama_models": r"D:\store\models"})
    captured = {}
    hc.start_ollama_server(ps_fn=lambda cmd, t: captured.setdefault("cmd", cmd) and "pid=1")
    assert r"set OLLAMA_MODELS=D:\store\models&& " in captured["cmd"]


def test_this_host_entry_resolves_paths_under_the_host_users_profile(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "gethostname", lambda: "evo-x2")
    monkeypatch.setenv("SystemDrive", "C:")
    h = hc._this_host_entry()
    assert h["ollama_exe"].lower().endswith(r"\ritz\appdata\local\programs\ollama\ollama.exe")
    assert h["ollama_models"].lower().endswith(r"\ritz\.ollama\models")
    assert hc.HOSTS["EVO-X2"]["ollama_exe"] == r"AppData\Local\Programs\Ollama\ollama.exe"


def test_start_ollama_server_no_models_dir_on_unknown_host(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setattr(hc, "_resolve_ollama_exe_for_serve", lambda: r"C:\x\ollama.exe")
    monkeypatch.setattr(hc, "ollama_process_running", lambda ps_fn=None: False)
    monkeypatch.setattr(hc, "_this_host_entry", lambda: {})
    captured = {}
    hc.start_ollama_server(ps_fn=lambda cmd, t: captured.setdefault("cmd", cmd) and "pid=1")
    assert "OLLAMA_MODELS" not in captured["cmd"]
