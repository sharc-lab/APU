"""harness/t2s_model_pull.py (queue job, evo-t2s): exactly one tq.advance() on every path, no pull when present, and the
pulls queue file matches its builder and passes the dry run. Fakes only."""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "scripts"))

import t2s_model_pull as mp  # noqa: E402
import t2s_week_queue as wq  # noqa: E402


class FakeTq:
    def __init__(self):
        self.notes = []

    def advance(self, note):
        self.notes.append(note)


def _host(monkeypatch, name="EVO-T2S", users="No User exists for *"):
    import host_config as hc
    monkeypatch.setattr(mp.socket, "gethostname", lambda: name)
    monkeypatch.setattr(hc, "_query_user_raw", lambda: users)
    monkeypatch.setattr(hc, "enforce_or_record_interactive_session",
                        lambda cfg, query_user_fn=None: (_ for _ in ()).throw(SystemExit("another interactive session"))
                        if "No User" not in users else {})


def test_main_advances_exactly_once_on_success(monkeypatch, tmp_path):
    _host(monkeypatch)
    monkeypatch.setattr(mp, "run", lambda tags, emit, log: (True, [{"model_tag": "qwen3:14b", "verify_ok": True}]))
    tq = FakeTq()
    assert mp.main(["--out", str(tmp_path / "o.jsonl")], tq=tq) == "completed" and tq.notes == ["completed"]


def test_main_advances_exactly_once_on_verify_failure(monkeypatch, tmp_path):
    _host(monkeypatch)
    monkeypatch.setattr(mp, "run", lambda tags, emit, log: (False, [{"model_tag": "qwen3:14b", "verify_ok": False}]))
    tq = FakeTq()
    note = mp.main(["--out", str(tmp_path / "o.jsonl")], tq=tq)
    assert note.startswith("stopped:") and tq.notes == [note]


def test_main_advances_exactly_once_on_exception(monkeypatch, tmp_path):
    _host(monkeypatch)

    def boom(*a):
        raise RuntimeError("ollama did not become ready")
    monkeypatch.setattr(mp, "run", boom)
    tq = FakeTq()
    note = mp.main(["--out", str(tmp_path / "o.jsonl")], tq=tq)
    assert "ollama did not become ready" in note and tq.notes == [note]


def test_main_advances_exactly_once_on_wrong_host(monkeypatch, tmp_path):
    _host(monkeypatch, name="SOMEBOX")
    called = []
    monkeypatch.setattr(mp, "run", lambda *a: called.append(1))
    tq = FakeTq()
    note = mp.main(["--out", str(tmp_path / "o.jsonl")], tq=tq)
    assert note.startswith("stopped:") and tq.notes == [note] and not called


def test_main_advances_exactly_once_when_someone_is_logged_in(monkeypatch, tmp_path):
    _host(monkeypatch, users="USERNAME SESSIONNAME\n zach console")
    called = []
    monkeypatch.setattr(mp, "run", lambda *a: called.append(1))
    tq = FakeTq()
    note = mp.main(["--out", str(tmp_path / "o.jsonl")], tq=tq)
    assert "another interactive session" in note and tq.notes == [note] and not called


def test_run_skips_pull_when_present(monkeypatch):
    import host_config as hc
    import t2s_week_preflight as pf
    calls = []
    monkeypatch.setattr(pf, "apply_ollama_pin", lambda: {"pinned": True})
    monkeypatch.setattr(hc, "_resolve_ollama_exe_for_serve", lambda: "ollama.exe")
    for n in ("stop_ollama_server", "start_ollama_server"):
        monkeypatch.setattr(hc, n, lambda *a, n=n, **k: calls.append(n))
    monkeypatch.setattr(hc, "wait_for_ollama_ready", lambda timeout_s=0: True)
    monkeypatch.setattr(mp, "present_tags", lambda: {"qwen3:14b"})
    monkeypatch.setattr(mp, "pull_one", lambda *a, **k: calls.append("PULL"))
    monkeypatch.setattr(mp, "verify_one", lambda tag: {"verify_ok": True, "verify_error": None})
    rows = []
    ok, res = mp.run(["qwen3:14b"], rows.append, lambda m: None)
    assert ok and "PULL" not in calls and calls[-1] == "stop_ollama_server"
    assert res[0]["already_present"] is True


def test_pulls_queue_file_matches_builder_and_dry_run_passes():
    items = json.loads(wq.PULLS_PATH.read_text(encoding="utf-8"))
    assert items == wq.build_pulls()
    rep = wq.dry_run(items)
    # The pull ran on evo-t2s at the start of the 2026-10-15 window (00:00 ET = 04:00 UTC) and its output synced into
    # results/. Accept the pre-launch "already exists" problem only for that run's own file; anything else fails.
    def own_synced_pull(problem):
        m = re.match(r"output '([^']+)' already exists in results/", problem)
        if not m:
            return False
        first = json.loads((wq.REPO / m.group(1)).read_text(encoding="utf-8").splitlines()[0])
        return first.get("record") == "run_start" and "tags" in first and first.get("ts_utc", "") >= "2026-10-15T04:00"
    bad = [e for e in rep["entries"] if not e["ok"] and not all(own_synced_pull(x) for x in e.get("problems", []))]
    assert not bad, rep
    ids = {it["id"] for it in json.loads(wq.QUEUE_PATH.read_text(encoding="utf-8"))}
    assert not ids & {it["id"] for it in items}


class FakeQueue:
    def __init__(self, items, paused=None):
        self.items, self.paused, self.log = items, paused, []

    def set_pause(self, reason):
        self.paused = reason
        self.log.append("pause")

    def clear_pause(self):
        self.paused = None
        self.log.append("clear")

    def read_queue(self):
        return [dict(it) for it in self.items]

    def write_queue(self, items):
        assert self.paused, "write outside the pause"
        self.items = items
        self.log.append("write")


def test_load_window_one_write_under_pause_in_order():
    fq = FakeQueue([{"id": "old", "status": "stopped_for_handover"}], paused={"reason": "T2S handover 2026-10-02"})
    ids = wq.load_window([wq.PULLS_PATH, wq.QUEUE_PATH], tq=fq)
    assert ids[0] == "t2s_wk_model_pull" and ids[1] == "t2s_wk_preflight" and len(ids) == 9
    assert fq.log == ["pause", "write", "clear"] and fq.paused is None
    assert [it["id"] for it in fq.items][:2] == ["old", "t2s_wk_model_pull"]
    assert all(it["status"] == "pending" for it in fq.items[1:])
    assert "finally" in wq.LOAD_PAUSE_REASON


def test_requeue_sets_an_error_entry_pending_under_pause_and_keeps_history():
    fq = FakeQueue([{"id": "a", "status": "done"},
                    {"id": "t2s_wk_preflight", "status": "error", "note": "stopped: preflight failed", "pid": 5},
                    {"id": "b", "status": "pending"}])
    it = wq.requeue("t2s_wk_preflight", "parser fix", tq=fq)
    assert it["status"] == "pending" and "pid" not in it and it["previous_runs"][0]["note"].startswith("stopped")
    assert fq.log == ["pause", "write", "clear"] and [x["id"] for x in fq.items] == ["a", "t2s_wk_preflight", "b"]


def test_requeue_refuses_while_running_or_on_a_done_entry():
    import pytest
    fq = FakeQueue([{"id": "x", "status": "running"}, {"id": "y", "status": "error"}])
    with pytest.raises(SystemExit):
        wq.requeue("y", "", tq=fq)
    fq2 = FakeQueue([{"id": "y", "status": "done"}])
    with pytest.raises(SystemExit):
        wq.requeue("y", "", tq=fq2)
    assert fq.log == ["pause", "clear"] and fq2.log == ["pause", "clear"]


def test_load_window_refuses_with_a_live_old_entry_and_still_clears():
    import pytest
    fq = FakeQueue([{"id": "old", "status": "pending"}])
    with pytest.raises(SystemExit):
        wq.load_window([wq.PULLS_PATH], tq=fq)
    assert fq.log == ["pause", "clear"] and fq.items == [{"id": "old", "status": "pending"}]
