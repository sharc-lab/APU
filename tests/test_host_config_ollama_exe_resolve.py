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
