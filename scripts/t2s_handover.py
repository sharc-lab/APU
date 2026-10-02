"""Self-executing handover for evo-t2s: leaves the machine clean for Zach even if the controller
session that queued tonight's work is gone by the time it runs.

Built 2026-10-02 because the 05:30 PT handover cannot depend on an operator (or an agent session)
being awake at that exact moment -- this script is installed as its own one-shot scheduled task on
evo-t2s so it runs regardless.

What it does, in order:
  1. Sets the queue maintenance lock (t2s_queue.set_pause) so the watchdog stops launching new work.
  2. Finds every queue_state.json entry with status "running" (these are, by construction, the only
     entries this script will touch -- ownership is the queue's own "running" bookkeeping, the same
     ownership test advance()/_caller_owns_running_job() uses elsewhere in this repo, not a guess).
  3. Waits up to --wait-s (default 600 = 10 minutes) for each running entry's recorded pid to exit on
     its own (a clean finish), polling every 15s.
  4. For any still alive after the wait, stops that pid's whole process tree (psutil, parent-to-child
     walk from the recorded pid -- never a pid not recorded in queue_state.json as ours).
  5. Stops every ollama.exe / "ollama app.exe" / llama-server.exe process found on the machine. This
     machine's entire Ollama install and every llama-server binary on it were set up for this project
     this week (see docs/T2S_CHANGELOG.md) -- there is no other owner for these two process names on
     evo-t2s, so a name match is the correct and complete ownership test here (unlike a shared
     controller laptop, where it would not be).
  6. Checks whether OLLAMA_IGPU_ENABLE was ever set PERSISTENTLY (HKCU/HKLM Environment registry
     keys, via setx or System Properties) rather than only transiently inside one cmd.exe invocation
     (the only way this session ever set it) -- removes it if found, records either way.
  7. Disables (never deletes) the APU-QueueWatchdog scheduled task.
  8. Writes C:\\apu\\ovn\\HANDOVER_DONE.json: timestamp, which pids/trees were stopped, the registry
     check result, the watchdog's new state, and a final process-list snapshot.

Never reboots. Never calls Stop-Process (or anything else) against a pid that is not one of (a) a
process recorded as "running" in queue_state.json, (b) a descendant of such a pid, or (c) an
ollama.exe/"ollama app.exe"/llama-server.exe process.

--dry-run prints every action it would take (which pids/trees, which ollama processes, the registry
check, the watchdog disable) without doing any of them, and without waiting the full --wait-s.

Usage: py -3.12 scripts\\t2s_handover.py [--dry-run] [--wait-s 600] [--poll-s 15]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# This script is deployed two ways: inside this repo (scripts/t2s_handover.py, with t2s_queue in
# ../harness/), and flat on evo-t2s itself (C:\apu\ovn\t2s_handover.py, with t2s_queue.py sitting
# right next to it -- that machine's real deployment layout, confirmed against how every other
# script this project uses there is laid out). Try the flat/sibling case first since that's where
# it actually runs in production; fall back to the repo layout for local dev/tests.
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parents[0] / "harness"))
import t2s_queue as tq  # noqa: E402

HANDOVER_DONE_PATH = Path(r"C:\apu\ovn\HANDOVER_DONE.json")
WATCHDOG_TASK_NAME = "APU-QueueWatchdog"
OLLAMA_PROCESS_NAMES = ("ollama.exe", "ollama app.exe", "llama-server.exe")
IGPU_ENV_VAR = "OLLAMA_IGPU_ENABLE"


def utc_iso():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ── injectable seams (real implementations below; tests supply fakes) ──────────────────────────


def real_list_processes():
    """Returns [{"pid": int, "ppid": int, "name": str}] for every process on this machine, via
    psutil (already a real dependency of this repo, see t2s_queue._pid_in_ancestor_chain)."""
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "ppid", "name"]):
        try:
            info = p.info
            out.append({"pid": info["pid"], "ppid": info.get("ppid"), "name": info.get("name") or ""})
        except Exception:
            continue
    return out


def real_kill_pid(pid):
    import psutil
    try:
        psutil.Process(pid).kill()
        return True
    except Exception:
        return False


def real_pid_alive(pid):
    import psutil
    return psutil.pid_exists(pid)


def real_check_persistent_igpu_enable():
    """Checks HKCU\\Environment and HKLM\\...\\Session Manager\\Environment for a persisted
    OLLAMA_IGPU_ENABLE value (set via setx or System Properties -- never how this session set it,
    which was always `set OLLAMA_IGPU_ENABLE=1 && ...` inside one cmd.exe invocation, scoped to that
    process only). Returns {"hkcu": str|None, "hklm": str|None}."""
    import winreg
    result = {"hkcu": None, "hklm": None}
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            try:
                result["hkcu"] = winreg.QueryValueEx(k, IGPU_ENV_VAR)[0]
            except FileNotFoundError:
                pass
    except OSError:
        pass
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as k:
            try:
                result["hklm"] = winreg.QueryValueEx(k, IGPU_ENV_VAR)[0]
            except FileNotFoundError:
                pass
    except OSError:
        pass
    return result


def real_clear_persistent_igpu_enable(scope):
    import winreg
    hive = winreg.HKEY_CURRENT_USER if scope == "hkcu" else winreg.HKEY_LOCAL_MACHINE
    path = "Environment" if scope == "hkcu" else r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    with winreg.OpenKey(hive, path, 0, winreg.KEY_SET_VALUE) as k:
        winreg.DeleteValue(k, IGPU_ENV_VAR)


def real_disable_scheduled_task(name):
    r = subprocess.run(["schtasks", "/Change", "/TN", name, "/DISABLE"],
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0, r.stdout + r.stderr


def real_query_scheduled_task_state(name):
    r = subprocess.run(["schtasks", "/Query", "/TN", name, "/FO", "LIST"],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        if line.strip().lower().startswith("status:"):
            return line.split(":", 1)[1].strip()
    return None


# ── core logic, all I/O injected so this is unit-testable without a real machine ───────────────


def find_descendants(root_pid, procs):
    by_ppid = {}
    for p in procs:
        by_ppid.setdefault(p["ppid"], []).append(p["pid"])
    out = []
    frontier = [root_pid]
    seen = set()
    while frontier:
        pid = frontier.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        frontier.extend(by_ppid.get(pid, []))
    return out


def run_handover(dry_run=False, wait_s=600, poll_s=15, *,
                 list_processes=real_list_processes, kill_pid=real_kill_pid, pid_alive=real_pid_alive,
                 check_igpu=real_check_persistent_igpu_enable, clear_igpu=real_clear_persistent_igpu_enable,
                 disable_task=real_disable_scheduled_task, query_task=real_query_scheduled_task_state,
                 read_queue=None, write_queue=None, set_pause=None, sleep=time.sleep, now=time.monotonic,
                 done_path=None):
    """Returns the HANDOVER_DONE record (dict) whether or not --dry-run; in dry-run mode nothing is
    actually stopped/disabled/written, and the record's "dry_run" key is True so a reader can tell."""
    read_queue = read_queue or tq.read_queue
    write_queue = write_queue or tq.write_queue
    set_pause = set_pause or tq.set_pause
    done_path = done_path if done_path is not None else HANDOVER_DONE_PATH

    actions = []

    def act(description, fn):
        actions.append(description)
        print(("[DRY RUN] " if dry_run else "") + description)
        if not dry_run:
            fn()

    # 1. maintenance lock
    act("set queue pause flag (handover in progress)", lambda: set_pause("T2S handover to Zach, 2026-10-02"))

    # 2-4. stop our own running queue job(s)
    items = read_queue()
    running = [it for it in items if it.get("status") == "running" and it.get("pid") is not None]
    stopped_trees = []
    for entry in running:
        pid = entry["pid"]
        waited = 0.0
        finished_on_own = False
        if not dry_run:
            while waited < wait_s:
                if not pid_alive(pid):
                    finished_on_own = True
                    break
                sleep(poll_s)
                waited += poll_s
        tree = find_descendants(pid, list_processes()) if (dry_run or not finished_on_own) else []
        if finished_on_own:
            print(f"queue entry {entry['id']!r} (pid {pid}) finished on its own after {waited:.0f}s")
            if not dry_run:
                entry["status"] = "done"
        else:
            act(f"stop queue entry {entry['id']!r}: pid {pid} and its process tree {tree}",
                lambda tree=tree: [kill_pid(p) for p in tree])
            if not dry_run:
                entry["status"] = "stopped_for_handover"
                entry["note"] = entry.get("note", "") + f" | stopped for T2S handover at {utc_iso()}"
            stopped_trees.append({"entry_id": entry["id"], "pid": pid, "tree": tree})
    if not dry_run and running:
        write_queue(items)

    # 5. stop ollama/llama-server
    procs = list_processes()
    ollama_pids = [p["pid"] for p in procs if p["name"] in OLLAMA_PROCESS_NAMES]
    if ollama_pids:
        act(f"stop ollama/llama-server processes: {ollama_pids}",
           lambda: [kill_pid(p) for p in ollama_pids])
    else:
        print("no ollama/llama-server processes found")

    # 6. persistent OLLAMA_IGPU_ENABLE check
    igpu_state = check_igpu()
    cleared = []
    for scope, val in igpu_state.items():
        if val is not None:
            act(f"clear persistent {IGPU_ENV_VAR} found in {scope} (value={val!r})",
               lambda scope=scope: clear_igpu(scope))
            cleared.append(scope)
    if not any(igpu_state.values()):
        print(f"{IGPU_ENV_VAR} is not set persistently (checked HKCU and HKLM Environment) -- confirmed clean")

    # 7. disable watchdog
    prior_state = query_task(WATCHDOG_TASK_NAME)
    act(f"disable scheduled task {WATCHDOG_TASK_NAME!r} (was: {prior_state})",
       lambda: disable_task(WATCHDOG_TASK_NAME))
    new_state = prior_state if dry_run else query_task(WATCHDOG_TASK_NAME)

    # 8. final process-list snapshot + record
    final_procs = list_processes()
    final_ollama = [p for p in final_procs if p["name"] in OLLAMA_PROCESS_NAMES]
    record = {
        "ts_utc": utc_iso(),
        "dry_run": dry_run,
        "actions": actions,
        "stopped_queue_entries": stopped_trees,
        "ollama_pids_stopped": ollama_pids,
        "persistent_igpu_enable_found": {k: v for k, v in igpu_state.items() if v is not None},
        "persistent_igpu_enable_cleared": cleared,
        "watchdog_prior_state": prior_state,
        "watchdog_new_state": new_state,
        "final_ollama_llamaserver_processes_remaining": final_ollama,
    }
    if not dry_run:
        done_path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait-s", type=int, default=600)
    ap.add_argument("--poll-s", type=int, default=15)
    args = ap.parse_args(argv)
    record = run_handover(dry_run=args.dry_run, wait_s=args.wait_s, poll_s=args.poll_s)
    print(json.dumps(record, indent=1, default=str))
    if final := record.get("final_ollama_llamaserver_processes_remaining"):
        print(f"WARNING: {len(final)} ollama/llama-server process(es) still present after handover", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
