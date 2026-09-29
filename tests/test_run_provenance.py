"""Tests for harness/run_provenance.py's degrade-instead-of-crash behavior when git itself is unavailable in the
deploy environment (C:\\apu\\ovn is a copy of committed files, not a git clone -- see t2s_k1_ollama.py's 2026-09-29
crash, an uncaught FileNotFoundError before it ever reached its own try/finally, on evo-x2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import run_provenance as rp  # noqa: E402


def test_script_provenance_degrades_when_git_unavailable_and_not_required(monkeypatch):
    monkeypatch.setattr(rp, "_git_available", lambda: False)
    result = rp.script_provenance([__file__], require_committed=False)
    assert result["git_head"] is None
    assert result["committed"] is None
    assert result["files"] == []
    assert result["problems"]


def test_script_provenance_raises_when_git_unavailable_and_required(monkeypatch):
    monkeypatch.setattr(rp, "_git_available", lambda: False)
    try:
        rp.script_provenance([__file__], require_committed=True)
        assert False, "expected ProvenanceError"
    except rp.ProvenanceError:
        pass


def test_script_provenance_still_works_when_git_available(monkeypatch):
    """Sanity check against the real repo (this test file itself is committed), so the fallback path does not
    silently become the only path ever exercised."""
    result = rp.script_provenance([__file__], require_committed=False)
    assert result["git_head"] is not None
    assert result["committed"] in (True, False)


def test_git_available_false_when_git_binary_missing(monkeypatch):
    def fake_run(*a, **kw):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(rp.subprocess, "run", fake_run)
    assert rp._git_available() is False


def test_git_available_false_when_not_a_repo(monkeypatch):
    class FakeResult:
        returncode = 128  # git's own exit code for "not a git repository"

    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **kw: FakeResult())
    assert rp._git_available() is False
