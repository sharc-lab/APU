import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import stale_server_cleanup as sc  # noqa: E402


def make_procs():
    return [
        {"pid": 1, "name": "ollama.exe", "ppid": 0, "cmdline": "C:\\Users\\X\\Ollama\\ollama.exe serve"},
        {"pid": 2, "name": "llama-server.exe", "ppid": 0,
         "cmdline": "C:\\apu\\bin\\llama-server.exe -m C:\\apu\\models\\x.gguf --port 8385"},
        {"pid": 3, "name": "llama-server.exe", "ppid": 0,
         "cmdline": "D:\\SomeoneElse\\llama-server.exe -m D:\\other\\model.gguf --port 9999"},
        {"pid": 4, "name": "notepad.exe", "ppid": 0, "cmdline": "notepad.exe"},
    ]


def test_dedicated_machine_rule_matches_any_ollama_or_llamaserver_by_name():
    procs = make_procs()
    matched = [p["pid"] for p in procs if sc.is_ours_dedicated_machine(p)]
    assert matched == [1, 2, 3]  # all 3 ollama/llama-server, regardless of path -- dedicated machine


def test_shared_machine_rule_only_matches_processes_pointing_into_our_paths():
    procs = make_procs()
    matched = [p["pid"] for p in procs if sc.is_ours_shared_machine(p, our_paths=("c:\\apu",))]
    assert matched == [2]  # only the one whose cmdline references C:\apu
    assert 1 not in matched  # ollama.exe with no C:\apu reference -- not clearly ours on a shared machine
    assert 3 not in matched  # someone else's llama-server entirely


def test_shared_machine_default_paths_recognize_ollamas_own_engine():
    """2026-10-02, found live: 16 orphaned Ollama-internal llama-server.exe processes had accumulated
    on evo-t2s, invisible to a 'C:\\apu'-only ownership check since Ollama's own internal engine never
    references C:\\apu at all (it runs from Ollama's own install dir and its own model blob cache),
    even though we are the only Ollama user on that machine. The default path list must catch this."""
    ollama_engine = {"pid": 99, "name": "llama-server.exe", "ppid": 0,
                     "cmdline": ("C:\\Users\\SHARC\\AppData\\Local\\Programs\\Ollama\\lib\\ollama\\"
                                 "llama-server.exe --model C:\\Users\\SHARC\\.ollama\\models\\blobs\\"
                                 "sha256-abc123")}
    assert sc.is_ours_shared_machine(ollama_engine) is True  # using the real default path list


def test_shared_machine_rule_also_matches_by_known_port():
    procs = make_procs()
    matched = [p["pid"] for p in procs if sc.is_ours_shared_machine(p, our_paths=(), our_ports=(9999,))]
    assert matched == [3]


def test_cleanup_kills_only_matched_processes_and_logs_each_one():
    procs = make_procs()
    killed_calls = []
    logs = []
    result = sc.cleanup_stale_servers(
        sc.is_ours_dedicated_machine,
        list_processes=lambda: procs,
        kill_pid=lambda pid: killed_calls.append(pid) or True,
        port_free=lambda port: True,
        log=logs.append,
    )
    assert set(killed_calls) == {1, 2, 3}
    assert 4 not in killed_calls  # notepad never touched
    assert len(logs) == 3  # one log line per killed process, nothing silent


def test_cleanup_waits_for_ports_to_actually_free():
    call_count = {"n": 0}

    def port_free(port):
        call_count["n"] += 1
        return call_count["n"] >= 3  # not free on first 2 checks, free on the 3rd

    slept = []
    result = sc.cleanup_stale_servers(
        sc.is_ours_dedicated_machine,
        list_processes=lambda: [],
        kill_pid=lambda pid: True,
        port_free=port_free,
        ports_to_wait_for=(8385,),
        wait_s=10, poll_s=1,
        sleep=slept.append,
    )
    assert result["ports_freed"] == {8385: True}
    assert len(slept) == 2  # slept twice before the port finally freed


def test_cleanup_reports_a_port_that_never_frees_without_raising():
    result = sc.cleanup_stale_servers(
        sc.is_ours_dedicated_machine,
        list_processes=lambda: [],
        kill_pid=lambda pid: True,
        port_free=lambda port: False,
        ports_to_wait_for=(8385,),
        wait_s=2, poll_s=1,
        sleep=lambda s: None,
    )
    assert result["ports_freed"] == {8385: False}
