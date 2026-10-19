"""Mechanical handover deadline for the evo-t2s queue (2026-10-14; first use: the window 2026-10-15 00:00 ET to the
handover to Zach on 2026-10-18 23:59 ET).

Nothing here depends on a person or an agent session being awake at the deadline. Three layers, all driven by one
file, C:\\apu\\ovn\\queue_deadline.json (t2s_queue.DEADLINE_FILE), written by `arm` below:

  1. no_launch_after_utc (2026-10-19T03:00Z = 23:00 ET): from then on t2s_queue.launch_next() launches nothing, from
     advance() and from the watchdog alike. A job that checks t2s_queue.deadline_reached() at its item boundaries
     (harness/x2_r2_agent.py run_plan, every R2 entry of the week) stops before its next session and advances the
     queue with a "stopped: handover deadline" note; its rows so far are kept (append-only JSONL).
  2. hard_deadline_utc (2026-10-19T03:30Z = 23:30 ET): queue_watchdog.tick() kills a job still running at its first
     tick at or after hard_deadline_utc - TICK_INTERVAL_S, so with a 10-minute tick the kill lands before the hard
     deadline. Rows already written are kept.
  3. As soon as nothing of ours is running after no_launch_after_utc (or at the kill), the watchdog calls handover():
     sets the queue pause flag with the recorded reason (who releases it), marks the running entry
     "stopped_for_handover", stops our Ollama / llama-server / python processes (ownership: stale_server_cleanup's
     shared-machine rule for the servers; for python, a command line or working directory under C:\\apu or the
     queue's own APU_QUEUE_JOB_ID variable; never the watchdog's own process chain), records whether
     OLLAMA_IGPU_ENABLE is set persistently (read only), and writes C:\\apu\\ovn\\HANDOVER_DONE.json (the previous one is
     kept as HANDOVER_DONE_prev_<stamp>.json). The pause makes every later watchdog tick a no-op, so it runs once.

The pause is deliberate and is NOT released automatically: its reason names who releases it (the next confirmed
window's operator). This is the one documented exception to "every pause has an automatic release" in CLAUDE.md: the
machine is handed back to its owner and must stay quiet.

Usage (on evo-t2s, deployed flat to C:\\apu\\ovn):
  python t2s_deadline.py arm --no-launch-after 2026-10-19T03:00:00Z --hard-deadline 2026-10-19T03:30:00Z \\
      --reason "handover to Zach 2026-10-18; resume only in the next confirmed window"
  python t2s_deadline.py status
Never reboots, never touches a scheduled task definition, never edits a queue entry other than the running one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
import t2s_queue as tq  # noqa: E402
import stale_server_cleanup as scc  # noqa: E402

HANDOVER_DONE_PATH = Path(r"C:\apu\ovn\HANDOVER_DONE.json")
TICK_INTERVAL_S = 600   # scripts/install_queue_watchdog.ps1: every 10 minutes
DEFAULT_REASON = "handover to Zach 2026-10-18; resume only in the next confirmed window"
OUR_PATH = "c:\\apu"
IGPU_ENV_VAR = "OLLAMA_IGPU_ENABLE"


def utc_iso(ts=None):
    return datetime.fromtimestamp(time.time() if ts is None else ts, timezone.utc).isoformat()


# ---------------------------------------------------------------- arm / status
def build_record(no_launch_after, hard_deadline, reason, armed_by="operator") -> dict:
    nl, hd = tq._epoch(no_launch_after), tq._epoch(hard_deadline)
    if hd < nl:
        raise ValueError("hard deadline is earlier than no-launch-after")
    if not reason.strip() or "resume" not in reason.lower():
        raise ValueError("the pause reason must say who resumes the queue and when")
    return {"no_launch_after_utc": utc_iso(nl), "hard_deadline_utc": utc_iso(hd),
            "kill_running_at_or_after_utc": utc_iso(hd - TICK_INTERVAL_S), "pause_reason": reason,
            "armed_ts_utc": utc_iso(), "armed_by": armed_by,
            "doc": "harness/t2s_deadline.py; the pause it sets is released only by the next confirmed window's operator"}


def arm(no_launch_after, hard_deadline, reason, path=None, armed_by="operator") -> dict:
    rec = build_record(no_launch_after, hard_deadline, reason, armed_by)
    Path(path or tq.DEADLINE_FILE).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def kill_threshold(deadline: dict) -> float:
    return tq._epoch(deadline["hard_deadline_utc"]) - TICK_INTERVAL_S


# ---------------------------------------------------------------- real seams
def real_list_processes():
    """[{pid, ppid, name, cmdline, cwd, job_env}] via psutil; cwd/job_env are None when not readable."""
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        try:
            info = p.info
            rec = {"pid": info["pid"], "ppid": info.get("ppid"), "name": info.get("name") or "",
                   "cmdline": " ".join(info.get("cmdline") or []), "cwd": None, "job_env": None}
            if rec["name"].lower().startswith("python"):
                try:
                    rec["cwd"] = p.cwd()
                except Exception:
                    pass
                try:
                    rec["job_env"] = p.environ().get(tq.JOB_ID_ENV_VAR)
                except Exception:
                    pass
            out.append(rec)
        except Exception:
            continue
    return out


def real_self_chain():
    """This process and its ancestors (the watchdog's own chain is never killed)."""
    import psutil
    chain = set()
    try:
        p = psutil.Process(os.getpid())
        while p is not None and len(chain) < 32:
            chain.add(p.pid)
            p = p.parent()
    except Exception:
        chain.add(os.getpid())
    return chain


def real_check_igpu():
    import winreg
    out = {"hkcu": None, "hklm": None}
    for scope, hive, key in (("hkcu", winreg.HKEY_CURRENT_USER, "Environment"),
                             ("hklm", winreg.HKEY_LOCAL_MACHINE,
                              r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
        try:
            with winreg.OpenKey(hive, key) as k:
                out[scope] = winreg.QueryValueEx(k, IGPU_ENV_VAR)[0]
        except OSError:
            pass
    return out


# ---------------------------------------------------------------- handover
def descendants(root_pid, procs):
    by_ppid = {}
    for p in procs:
        by_ppid.setdefault(p.get("ppid"), []).append(p["pid"])
    out, frontier, seen = [], [root_pid], set()
    while frontier:
        pid = frontier.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        frontier.extend(by_ppid.get(pid, []))
    return out


def is_our_python(p) -> bool:
    if not p["name"].lower().startswith("python"):
        return False
    return (OUR_PATH in (p.get("cmdline") or "").lower() or (p.get("cwd") or "").lower().startswith(OUR_PATH)
            or bool(p.get("job_env")))


def handover(*, list_processes=real_list_processes, kill_pid=scc.real_kill_pid, self_chain=real_self_chain,
             check_igpu=real_check_igpu, done_path=None, now=None, log=print) -> dict:
    """Runs the deadline handover once (see the module docstring). Returns the record it wrote."""
    done_path = Path(done_path) if done_path is not None else HANDOVER_DONE_PATH
    now = time.time() if now is None else now
    deadline = tq.read_deadline() or {}
    reason = deadline.get("pause_reason") or DEFAULT_REASON
    tq.set_pause(reason)   # first, so nothing launches while the rest runs
    log(f"t2s_deadline: pause set ({reason})")
    mine = set(self_chain())

    procs = list_processes()
    items = tq.read_queue()
    stopped_entries = []
    for it in items:
        if it.get("status") != "running":
            continue
        tree = [p for p in descendants(it["pid"], procs) if p not in mine] if it.get("pid") is not None else []
        killed = [p for p in tree if kill_pid(p)]
        it["status"] = "stopped_for_handover"
        it["finished_ts"] = now
        it["note"] = (str(it.get("note") or "") + f" | stopped for handover at {utc_iso(now)} "
                      f"(t2s_deadline, rows kept)").strip(" |")
        stopped_entries.append({"entry_id": it["id"], "pid": it.get("pid"), "tree": tree, "killed": killed})
        log(f"t2s_deadline: stopped running entry {it['id']} tree {tree}")
    if stopped_entries:
        tq.write_queue(items)

    procs = list_processes()
    targets = [p for p in procs if p["pid"] not in mine and (scc.is_ours_shared_machine(p) or is_our_python(p))]
    killed = []
    for p in targets:
        if kill_pid(p["pid"]):
            killed.append({"pid": p["pid"], "name": p["name"], "cmdline": (p.get("cmdline") or "")[:200]})
    log(f"t2s_deadline: stopped {len(killed)} of our processes")
    spared = [{"pid": p["pid"], "name": p["name"]} for p in procs
              if p["name"] in scc.OLLAMA_PROCESS_NAMES and p not in targets]

    try:
        igpu = check_igpu()
    except Exception as e:
        igpu = {"error": repr(e)}
    remaining = [{"pid": p["pid"], "name": p["name"]} for p in list_processes()
                 if p["pid"] not in mine and (scc.is_ours_shared_machine(p) or is_our_python(p))]
    record = {"record": "t2s_deadline_handover", "ts_utc": utc_iso(now), "deadline": deadline,
              "pause_reason": reason, "pause_set": tq.is_paused() is not None,
              "stopped_queue_entries": stopped_entries, "processes_stopped": killed,
              "ollama_like_processes_spared_not_ours": spared,
              "our_processes_remaining": remaining,
              "persistent_igpu_enable": igpu,
              "queue_after": [{"id": it["id"], "status": it.get("status")} for it in items],
              "release": "resume only in the next confirmed window: t2s_queue.clear_pause() by that window's operator"}
    if done_path.exists():
        stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            done_path.replace(done_path.with_name(f"{done_path.stem}_prev_{stamp}.json"))
        except OSError:
            pass
    done_path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--no-launch-after", required=True)
    a.add_argument("--hard-deadline", required=True)
    a.add_argument("--reason", default=DEFAULT_REASON)
    sub.add_parser("status")
    args = ap.parse_args(argv)
    if args.cmd == "arm":
        print(json.dumps(arm(args.no_launch_after, args.hard_deadline, args.reason), indent=1))
    else:
        d = tq.read_deadline()
        print(json.dumps({"deadline": d, "reached": tq.deadline_reached() is not None,
                          "paused": tq.is_paused(), "now_utc": utc_iso()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
