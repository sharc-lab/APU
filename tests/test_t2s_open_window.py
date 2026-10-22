"""Start and end of an evo-t2s window (process rule 2026-10-20): the handover check, the controller's open-window job
that widens the results sync, arms the deadline and reverts the sync in its finally block, and sync_results'
expiring window file. Fakes only: no ssh, no real process list, no scheduled task, files under tmp_path."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "scripts"))

import t2s_deadline as tdl  # noqa: E402
import t2s_queue as q  # noqa: E402
import sync_results as sr  # noqa: E402

NO_LAUNCH = "2026-11-08T03:00:00Z"
HARD = "2026-11-08T03:30:00Z"
REASON = "handover to Zach 2026-11-08; resume only in the next confirmed window"
DONE = {"record": "t2s_deadline_handover", "ts_utc": "2026-10-19T03:21:00+00:00", "our_processes_remaining": []}
PAUSE = {"reason": "handover to Zach 2026-10-18; resume only in the next confirmed window"}


# ── handover check ──

def test_check_handover_ok():
    r = tdl.check_handover(DONE, PAUSE, [{"pid": 7, "name": "explorer.exe", "cmdline": ""}])
    assert r["ok"] and r["problems"] == []


@pytest.mark.parametrize("done,pause,procs,needle", [
    (None, PAUSE, [], "HANDOVER_DONE.json missing"),
    (DONE, None, [], "pause flag absent"),
    (DONE, {"reason": "manual"}, [], "does not name who releases"),
    (DONE, PAUSE, [{"pid": 9, "name": "python.exe", "cmdline": r"C:\apu\ovn\x2_r2_agent.py"}], "processes of ours"),
    (dict(DONE, our_processes_remaining=[{"pid": 1}]), PAUSE, [], "still running"),
])
def test_check_handover_failures(done, pause, procs, needle):
    r = tdl.check_handover(done, pause, procs)
    assert not r["ok"] and any(needle in p for p in r["problems"])


def test_check_handover_spares_own_chain():
    procs = [{"pid": 5, "name": "python.exe", "cmdline": r"C:\apu\ovn\t2s_deadline.py check-handover"}]
    assert tdl.check_handover(DONE, PAUSE, procs, mine={5})["ok"]


# ── open-window (controller) ──

class Remote:
    def __init__(self, check_ok=True, arm_rc=0):
        self.calls, self.check_ok, self.arm_rc = [], check_ok, arm_rc

    def __call__(self, argv):
        self.calls.append(argv)
        if argv[0] == "check-handover":
            return (0 if self.check_ok else 3), json.dumps({"ok": self.check_ok, "problems": [] if self.check_ok
                                                            else ["pause flag absent"]})
        if argv[0] == "arm":
            return self.arm_rc, json.dumps(tdl.build_record(argv[2], argv[4], argv[6]))
        raise AssertionError(argv)


def _clock(start, wf, seen):
    t = [start]

    def now():
        return t[0]

    def sleep(s):
        seen.append(Path(wf).exists())   # the window file exists while the job waits
        t[0] += s
    return now, sleep


def test_open_window_widens_arms_waits_and_reverts(tmp_path):
    wf = tmp_path / ".sync_host_window.json"
    rem, seen = Remote(), []
    now, sleep = _clock(q._epoch("2026-11-07T20:00:00Z"), wf, seen)
    res = tdl.open_window(NO_LAUNCH, HARD, REASON, remote=rem, window_file=wf, now_fn=now, sleep_fn=sleep,
                          log=lambda m: None)
    assert [c[0] for c in rem.calls] == ["check-handover", "arm"]    # check first, then arm
    assert res["opened"] and res["sync_widened"] and res["sync_reverted"]
    assert seen and all(seen)                                         # widened during the wait
    assert not wf.exists()                                            # reverted at the deadline (finally)
    assert now() >= q._epoch(HARD)


def test_open_window_refuses_without_touching_sync(tmp_path):
    wf = tmp_path / ".sync_host_window.json"
    rem = Remote(check_ok=False)
    res = tdl.open_window(NO_LAUNCH, HARD, REASON, remote=rem, window_file=wf, log=lambda m: None)
    assert not res["opened"] and not res["sync_widened"] and not wf.exists()
    assert [c[0] for c in rem.calls] == ["check-handover"]           # never armed


def test_open_window_reverts_when_arm_fails(tmp_path):
    wf = tmp_path / ".sync_host_window.json"
    with pytest.raises(RuntimeError):
        tdl.open_window(NO_LAUNCH, HARD, REASON, remote=Remote(arm_rc=1), window_file=wf, log=lambda m: None)
    assert not wf.exists()


def test_open_window_reverts_on_interrupt(tmp_path):
    wf = tmp_path / ".sync_host_window.json"

    def boom(_s):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        tdl.open_window(NO_LAUNCH, HARD, REASON, remote=Remote(), window_file=wf,
                        now_fn=lambda: q._epoch("2026-11-07T20:00:00Z"), sleep_fn=boom, log=lambda m: None)
    assert not wf.exists()


def test_open_window_validates_reason_before_any_remote_call(tmp_path):
    rem = Remote()
    with pytest.raises(ValueError):
        tdl.open_window(NO_LAUNCH, HARD, "no releaser named", remote=rem, window_file=tmp_path / "w.json")
    assert rem.calls == []


# ── sync_results: the window file widens evo-x2 to both only until it expires ──

def test_effective_host_widens_then_expires(tmp_path):
    wf = tmp_path / ".sync_host_window.json"
    assert sr.effective_host("evo-x2", wf)[0] == "evo-x2"            # no file
    tdl.write_sync_window(wf, "2026-11-08T03:30:00+00:00")
    assert sr.effective_host("evo-x2", wf, now=q._epoch("2026-11-01T00:00:00Z"))[0] == "both"
    assert sr.effective_host("evo-t2s", wf, now=q._epoch("2026-11-01T00:00:00Z"))[0] == "evo-t2s"
    host, why = sr.effective_host("evo-x2", wf, now=q._epoch("2026-11-08T03:30:00Z"))
    assert host == "evo-x2" and "expired" in why and not wf.exists()  # reverted at the deadline without the job


def test_effective_host_unreadable_never_widens(tmp_path):
    wf = tmp_path / ".sync_host_window.json"
    wf.write_text("{not json", encoding="utf-8")
    assert sr.effective_host("evo-x2", wf)[0] == "evo-x2"


def test_window_file_is_gitignored_and_default_path_matches():
    assert ".sync_host_window.json" in (REPO / ".gitignore").read_text(encoding="utf-8")
    assert sr.SYNC_WINDOW_FILE == REPO / ".sync_host_window.json"
