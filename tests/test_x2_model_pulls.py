"""2026-10-02: x2_model_pulls is the queue job that guarantees x2_outcome_table_v2 never again silently runs
against a missing/mistagged Ollama model (qwen3:30b-a3b-instruct-2507 404'd all night; qwen3:32b was never
pulled at all). These tests cover the real logic (tag de-duplication, the already-present skip, pull/verify
result shape) with the real subprocess/network calls stubbed out."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import x2_model_pulls as mp  # noqa: E402


def test_required_tags_matches_model_map_deduplicated_in_order():
    import x2_outcome_table as x2ot
    tags = mp.required_tags()
    expected = []
    for ollama_tag, _gguf in x2ot.MODEL_MAP.values():
        if ollama_tag not in expected and ollama_tag not in x2ot.OLLAMA_CREATE_FROM_GGUF:
            expected.append(ollama_tag)
    assert tags == expected
    assert len(tags) == len(set(tags))  # no duplicates


def test_required_tags_is_the_real_corrected_list():
    """Locks in the specific 2026-10-02 fix: the bad tag must not reappear, the real one must be present."""
    tags = mp.required_tags()
    assert "qwen3:30b-a3b-instruct-2507" not in tags
    assert "qwen3:32b" in tags
    # 2026-10-07: the 30b leg now uses a model created locally from the Instruct-2507 GGUF; it must never be
    # pulled from the registry, where it does not exist.
    assert "qwen3-30b-a3b-2507" not in tags


def test_currently_present_tags_parses_real_api_tags_shape(monkeypatch):
    import urllib.request
    import json as _json

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
        def read(self):
            return _json.dumps(self._payload).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=10: FakeResp({"models": [{"name": "qwen3:8b"}, {"name": "qwen3:32b"}]}))
    assert mp.currently_present_tags() == {"qwen3:8b", "qwen3:32b"}


def test_currently_present_tags_normalizes_a_trailing_latest(monkeypatch):
    """2026-10-02 bug found live: Ollama's /api/tags reports a tagless custom model (e.g. a MODEL_MAP entry
    'qwen3-4b-2507') back with an explicit ':latest' suffix ('qwen3-4b-2507:latest'). Comparing the raw name
    against the bare required tag made an already-present model look missing and triggered a doomed pull
    from the public registry, which has no such model at all."""
    import urllib.request
    import json as _json

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
        def read(self):
            return _json.dumps(self._payload).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=10: FakeResp({"models": [{"name": "qwen3-4b-2507:latest"},
                                                                     {"name": "qwen3:8b"}]}))
    present = mp.currently_present_tags()
    assert "qwen3-4b-2507" in present  # the bare form, matching required_tags()'s own naming
    assert "qwen3-4b-2507:latest" not in present


def test_currently_present_tags_returns_empty_set_on_any_failure(monkeypatch):
    import urllib.request

    def raise_it(req, timeout=10):
        raise ConnectionRefusedError("no server")

    monkeypatch.setattr(urllib.request, "urlopen", raise_it)
    assert mp.currently_present_tags() == set()


def test_verify_one_reports_ok_on_a_successful_chat(monkeypatch):
    class FakeClient:
        def chat(self, model, prompt, max_tokens=64):
            return {"outcome": "ok", "message": "hi"}

    monkeypatch.setattr(mp.k1, "OllamaClient", FakeClient)
    result = mp.verify_one("qwen3:8b")
    assert result == {"verify_ok": True, "verify_error": None}


def test_verify_one_reports_failure_with_error_text(monkeypatch):
    class FakeClient:
        def chat(self, model, prompt, max_tokens=64):
            return {"outcome": "http_error", "error": "model not found"}

    monkeypatch.setattr(mp.k1, "OllamaClient", FakeClient)
    result = mp.verify_one("qwen3:30b-a3b")
    assert result == {"verify_ok": False, "verify_error": "model not found"}


def test_pull_one_streams_heartbeats_and_reports_real_exit_code(monkeypatch, tmp_path):
    """A pull that 'takes a while' must emit at least one heartbeat before finishing, and report the child's
    real exit code -- not just assume success."""
    calls = {"poll_n": 0}

    class FakeProc:
        stdout = None
        returncode = 0
        def poll(self):
            calls["poll_n"] += 1
            return None if calls["poll_n"] < 3 else 0

    monkeypatch.setattr(mp.subprocess, "Popen", lambda *a, **kw: FakeProc())
    # make the elapsed-time clock advance fast enough to trigger a heartbeat on the very first loop iteration
    times = iter([0, 0, 1000, 1000, 1000, 1000, 1000, 1000])
    monkeypatch.setattr(mp.time, "monotonic", lambda: next(times, 1000))
    monkeypatch.setattr(mp.time, "sleep", lambda s: None)
    heartbeats = []
    out_path = tmp_path / "out.jsonl"
    result = mp.pull_one("qwen3:32b", out_path, log=heartbeats.append)
    assert result["pull_exit_code"] == 0
    assert any("heartbeat" in h for h in heartbeats)


def test_pull_one_handles_a_real_child_that_floods_unicode_output_without_hanging_or_crashing(monkeypatch, tmp_path):
    """2026-10-02 bug found live: ollama's progress spinner writes Braille-pattern characters fast enough to
    fill the OS pipe buffer when nothing reads it, which blocks the child on its own write() (looked exactly
    like a hang: 9 straight minutes of unchanged heartbeats) and then crashed with UnicodeDecodeError once
    finally read under Windows' default console codepage. A REAL subprocess here (not a stub) that floods
    stdout with those exact characters faster than the old code's end-of-run-only read would drain them --
    if the fix regresses, this either hangs (pipe deadlock) or raises (decode error), not just reports a
    wrong value."""
    monkeypatch.setattr(mp, "OLLAMA_EXE", sys.executable)
    monkeypatch.setattr(mp, "HEARTBEAT_EVERY_S", 10_000)  # keep this test fast; heartbeat timing is covered elsewhere
    script = (
        "import sys\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"  # force the fake child itself past Windows' own default
                                                       # console codepage, so only pull_one's own decoding is
                                                       # under test here, not an unrelated child-side crash
        "for _ in range(5000):\n"
        "    sys.stdout.write('\\u283b\\u2819\\u2839 pulling manifest \\n')\n"
        "sys.stdout.flush()\n"
    )
    out_path = tmp_path / "out.jsonl"
    # mp.pull_one calls [OLLAMA_EXE, "pull", tag] -- OLLAMA_EXE is python here, "pull"/tag become argv it ignores
    # since the real work is `python -c <script>`; patch Popen's args through instead.
    real_popen = mp.subprocess.Popen
    monkeypatch.setattr(mp.subprocess, "Popen",
                        lambda args, **kw: real_popen([sys.executable, "-c", script], **kw))
    result = mp.pull_one("qwen3:32b", out_path, log=lambda s: None)
    assert result["pull_exit_code"] == 0


def test_pull_one_kills_and_reports_timeout_past_the_cap(monkeypatch, tmp_path):
    class FakeProc:
        stdout = None
        killed = False
        def poll(self):
            return None  # never finishes on its own
        def kill(self):
            FakeProc.killed = True

    monkeypatch.setattr(mp.subprocess, "Popen", lambda *a, **kw: FakeProc())
    times = iter([0, 0, mp.PULL_TIMEOUT_S + 1])
    monkeypatch.setattr(mp.time, "monotonic", lambda: next(times, mp.PULL_TIMEOUT_S + 1))
    monkeypatch.setattr(mp.time, "sleep", lambda s: None)
    out_path = tmp_path / "out.jsonl"
    result = mp.pull_one("qwen3:32b", out_path, log=lambda s: None)
    assert result["pull_exit_code"] is None
    assert "killed" in result["pull_error"]
    assert FakeProc.killed is True
