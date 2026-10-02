import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import proc_util as pu  # noqa: E402


def test_run_hidden_sets_create_no_window_on_win32(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    captured = {}

    def fake_run(args, **kwargs):
        captured.update(kwargs)
        return "ran"

    monkeypatch.setattr(subprocess, "run", fake_run)
    pu.run_hidden(["echo", "hi"], capture_output=True)
    assert captured["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert captured["startupinfo"].dwFlags & subprocess.STARTF_USESHOWWINDOW
    assert captured["startupinfo"].wShowWindow == subprocess.SW_HIDE
    assert captured["capture_output"] is True  # caller's own kwargs preserved


def test_run_hidden_ors_in_caller_supplied_creationflags(monkeypatch):
    """A caller that already passes its own creationflags must not have CREATE_NO_WINDOW silently dropped,
    nor have its own flag silently dropped."""
    monkeypatch.setattr(sys, "platform", "win32")
    captured = {}
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: captured.update(kw))
    other_flag = 0x00000200  # CREATE_NEW_PROCESS_GROUP, picked arbitrarily as "some other real flag"
    pu.run_hidden(["echo"], creationflags=other_flag)
    assert captured["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert captured["creationflags"] & other_flag


def test_run_hidden_is_a_noop_passthrough_on_non_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    captured = {}
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: captured.update(kw) or "ran")
    pu.run_hidden(["free", "-b"], capture_output=True)
    assert "creationflags" not in captured
    assert "startupinfo" not in captured
    assert captured["capture_output"] is True


def test_popen_hidden_and_check_output_hidden_also_wrap(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    popen_kwargs = {}
    co_kwargs = {}
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kw: popen_kwargs.update(kw))
    monkeypatch.setattr(subprocess, "check_output", lambda args, **kw: co_kwargs.update(kw))
    pu.popen_hidden(["ssh", "host"])
    pu.check_output_hidden(["git", "status"])
    assert popen_kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert co_kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
