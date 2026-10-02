"""Independent watchdog for the evo-t2s/evo-x2 run queue. Runs as a Windows scheduled task (every 10 min, as SYSTEM)
so a crashed or hung orchestrator process cannot itself silently leave the queue stuck -- unlike t2s_queue.advance(),
which only runs from inside a finishing orchestrator's own process and therefore cannot recover if that process
never gets to its finally block at all (killed externally, powered off mid-run, or crashed hard enough to skip
cleanup).

2026-09-29 incident (see docs/T2S_CHANGELOG.md / docs/X2_CHANGELOG.md): the original version of this module trusted
a single PID-liveness check as authoritative. On evo-t2s that produced a false crash: the WMI-launched wrapper PID
died (root cause under separate investigation) while the actual measurement process kept running and legitimately
finished 2+ hours later, but the watchdog had already declared it "crashed" and cascaded through launching (or
failing to launch) every subsequent queued job. On evo-x2, the opposite failure hit the OLD run_end_written()
heuristic (guessing "the newest .jsonl file mtime after started_ts" when a job's cmd has no --resume flag): it
picked a DIFFERENT run's file that happened to have a run_end record, concluded the truly-dead x2_section0_retry
had already finished and advanced itself, and did nothing for 10+ hours while the job sat dead.

Fix: a job is declared dead only when BOTH signals agree it is dead.
  - pid check: is the recorded pid a live process.
  - heartbeat: how long since C:\\apu\\ovn\\queue_<id>.log (the job's own dedicated log file, created once by
    _launch and appended to for as long as the process runs) was last modified. This replaces the old "guess the
    newest jsonl across all stems" heuristic, which cannot tell one run's file from another's when several stems'
    files are touched in the same window. Stale = no update for HEARTBEAT_STALE_S (30 min), or
    HEARTBEAT_STALE_S_70B (60 min) for a 70B-model phase, which runs single calls that can legitimately take
    several minutes each.

Decision per tick, in order:
  1. If an entry is "running": pid alive -> do nothing. pid dead, heartbeat fresh -> log "pid_check_disagrees" and
     do nothing else (trust the heartbeat; a later tick will re-check). pid dead AND heartbeat stale -> mark
     "crashed", launch the next pending entry.
  2. Else if nothing is "running" and something is "pending" (and its gate, if any, is satisfied), launch it.
  3. Else if nothing is "running" and nothing eligible is "pending", write queue_empty.flag with a timestamp.
  4. A "running" entry with no pid ever recorded (its own launch failed) is never marked crashed by this watchdog --
     see t2s_queue.launch_next, which now reverts a failed launch straight back to "pending" instead of leaving a
     phantom "running, pid None" entry for the watchdog to find.

Every decision is logged to C:\\apu\\ovn\\watchdog.log, one line per tick, so a human reviewing the machine after the
fact can see what the watchdog saw and did, not just its final effect on queue_state.json.

Standing rule: every writer of queue_state.json uses t2s_queue.write_queue (this module does too, via
t2s_queue.launch_next / t2s_queue.write_queue) -- never a manual PowerShell edit, which is what produced the UTF-8
BOM that crashed advance() on 2026-09-29 in the first place.

Usage: python queue_watchdog.py   (no arguments; reads/writes C:\\apu\\ovn\\queue_state.json via t2s_queue)
Installed as a scheduled task by scripts/install_queue_watchdog.ps1.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
import t2s_queue as q  # noqa: E402
import stale_server_cleanup as scc  # noqa: E402

CIRCUIT_FAST_FAIL_S = 120  # a job that dies within this long of its own launch looks like a stale-state trip,
                          # not a genuine mid-run failure
CIRCUIT_CLEANUP_WAIT_S = 60
CIRCUIT_MAX_ATTEMPTS = 3
WATCHDOG_STATE_PATH = Path(r"C:\apu\ovn\watchdog_circuit_state.json")

WATCHDOG_LOG = Path(r"C:\apu\ovn\watchdog.log")
QUEUE_LOG_DIR = Path(r"C:\apu\ovn")
HEARTBEAT_STALE_S = 30 * 60
HEARTBEAT_STALE_S_70B = 60 * 60

# Progress-staleness thresholds (2026-09-30): the heartbeat above only proves the PROCESS is alive and its log file
# is still being touched -- a process can keep logging progress lines (or keep a stale log fresh via any periodic
# write) while producing zero actual result rows, e.g. stuck retrying a call that never completes, or looping on a
# guard failure. This is a real, distinct failure mode from a dead process: found on 2026-09-30 when two separate
# jobs on two separate machines were reported "running" well past when they had actually finished or errored out,
# because every check only re-confirmed pid-alive/heartbeat-fresh without ever checking whether a NEW RESULT ROW had
# actually been written recently. The thresholds here are deliberately longer than the heartbeat ones (60/120 min
# vs 30/60 min) since a real call can legitimately take several minutes without writing a row yet; this check exists
# to catch "stopped writing rows entirely for an hour-plus", not to second-guess normal per-call latency.
PROGRESS_STALE_S = 60 * 60
PROGRESS_STALE_S_LONG = 120 * 60


def _log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    try:
        with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line)


def pid_alive(pid, ps_fn=None):
    """ps_fn is injectable for tests (real usage calls t2s_lab.ps via a lazy import, so this module has no hard
    dependency on t2s_lab -- the watchdog must keep working even if that module fails to import for some reason)."""
    if pid is None:
        return False
    if ps_fn is None:
        import t2s_lab as L  # noqa: WPS433  (lazy: keep the watchdog's own import surface minimal)
        ps_fn = L.ps
    out = ps_fn(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue | Measure-Object).Count", 20)
    return out.strip() not in ("", "0")


def is_70b_phase(entry):
    """True if this entry's own cmd mentions a 70B model -- those phases run single calls that can legitimately
    take several minutes each (see P70's ~176s e2e_s per call), so they get the longer stale threshold."""
    cmd = entry.get("cmd") or []
    return any("70b" in str(c).lower() for c in cmd)


def heartbeat_age_s(entry, log_dir=QUEUE_LOG_DIR, now=None):
    """Seconds since C:\\apu\\ovn\\queue_<id>.log (this entry's own dedicated log file) was last modified, or None
    if that file does not exist. This is the one signal that reflects the actual measurement process's real
    progress regardless of whether the entry's cmd used --resume or which results stem it ended up writing to --
    see the module docstring for the 2026-09-29 incident this replaces run_end_written()'s guessing for."""
    log_path = Path(log_dir) / f"queue_{entry['id']}.log"
    try:
        mtime = log_path.stat().st_mtime
    except OSError:
        return None
    now = now if now is not None else time.time()
    return now - mtime


def is_stale(entry, age_s):
    if age_s is None:
        return True
    threshold = HEARTBEAT_STALE_S_70B if is_70b_phase(entry) else HEARTBEAT_STALE_S
    return age_s > threshold


def is_long_prompt_phase(entry):
    """True if this entry's own cmd mentions a 96K+ token prompt length -- K1 v3's own probe sweep (16K/32K/48K/
    96K/128K) is the current example, but this checks the cmd generically (any literal 96000+/128000/196608/262144-
    scale number as a standalone arg or inside a comma list) rather than hardcoding K1 v3's own constant names, so a
    future phase that probes similarly long prompts gets the longer threshold without this file needing an edit."""
    cmd = entry.get("cmd") or []
    for c in cmd:
        for tok in str(c).replace(",", " ").split():
            try:
                if int(tok) >= 96_000:
                    return True
            except ValueError:
                continue
    return False


def needs_long_progress_threshold(entry):
    return is_70b_phase(entry) or is_long_prompt_phase(entry)


def progress_stale_age_s(phases, now=None):
    """Seconds since the most recent result row across every phase bucket collect_phase_summaries produced (the
    real signal item 4 asks for: has ANY new RESULT ROW been written recently, not just a log line). Deliberately
    global across phases rather than trying to map a queue entry id to one specific phase/section key -- that
    mapping is not reliable (e.g. "r1b_controls" the queue id vs "R1b"/"R1d" the section values an entry's own rows
    actually carry), and a wrong guess here would be exactly the kind of silent misattribution this repo's own
    queue_watchdog history (see module docstring) has already been burned by once. If the job running right now is
    genuinely producing rows, some phase's last_ts_utc will be recent; if nothing has been written anywhere in over
    an hour while something is marked running, that is the real alert condition regardless of which phase it would
    have landed in. Returns None if no phase has ever recorded a last_ts_utc at all."""
    from datetime import datetime, timezone
    now = now if now is not None else time.time()
    timestamps = []
    for p in phases.values():
        ts = p.get("last_ts_utc")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            timestamps.append(dt.timestamp())
        except ValueError:
            continue
    if not timestamps:
        return None
    return now - max(timestamps)


def check_progress_stale(phases, queue_items, now=None):
    """Read-only progress check, alongside (not instead of) the heartbeat check above: if a queue entry is
    "running", is the last actual result row across all known phases older than PROGRESS_STALE_S (or
    PROGRESS_STALE_S_LONG for a 70B or 96K+-prompt phase)? Returns an ALERT string or None. Never touches pid_alive
    and never starts/stops/kills anything -- same read-only contract as check_idle in analysis/results_digest.py,
    which this is meant to be called alongside there."""
    now = now if now is not None else time.time()
    running = next((it for it in queue_items if it.get("status") == "running"), None)
    if running is None:
        return None
    age = progress_stale_age_s(phases, now=now)
    threshold = PROGRESS_STALE_S_LONG if needs_long_progress_threshold(running) else PROGRESS_STALE_S
    if age is None:
        return f"job {running['id']} is running but no phase has ever written a result row"
    if age > threshold:
        return (f"job {running['id']} is running (pid/heartbeat may look fine) but no new result row has been "
                f"written anywhere in {age/60:.0f} min (threshold {threshold//60} min)")
    return None


def _tail_error_signature(entry, log_dir=QUEUE_LOG_DIR, read_text=None):
    """A crude error signature for a crashed job: the last non-empty line of its own queue_<id>.log,
    used only to detect '2 consecutive jobs failed with the same error' (circuit breaker condition).
    Not meant to be a precise error parser -- a false match just means one extra cleanup+wait that
    was not strictly necessary, never a correctness problem. read_text is injectable for tests."""
    read_text = read_text or (lambda p: p.read_text(encoding="utf-8", errors="replace"))
    log_path = log_dir / f"queue_{entry['id']}.log"
    try:
        text = read_text(log_path)
    except Exception:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else None


def _load_circuit_state(path=WATCHDOG_STATE_PATH, read_text=None):
    read_text = read_text or (lambda p: p.read_text(encoding="utf-8"))
    try:
        return json.loads(read_text(path))
    except Exception:
        return {"last_crashed_id": None, "last_crash_signature": None}


def _save_circuit_state(state, path=WATCHDOG_STATE_PATH, write_text=None):
    write_text = write_text or (lambda p, s: p.write_text(s, encoding="utf-8"))
    write_text(path, json.dumps(state, indent=1))


def host_ownership_predicate(hostname=None):
    """evo-x2: name match is sufficient (dedicated machine). evo-t2s (and anything else): only a
    process whose command line points into C:\\apu is ours -- see stale_server_cleanup's own
    docstring for why these differ."""
    import socket
    hostname = (hostname or socket.gethostname()).lower()
    if "x2" in hostname:
        return scc.is_ours_dedicated_machine
    # 2026-10-02 bug found live: this used to hardcode our_paths=("c:\\apu",), which cannot see Ollama's own
    # internal engine (it runs from Ollama's own install dir against its own model blob cache, never
    # referencing C:\apu at all -- see stale_server_cleanup.DEFAULT_SHARED_MACHINE_PATHS's own docstring for
    # the exact paths) -- 16-18 such orphans on evo-t2s were invisible to this check. Falling through to scc's
    # own DEFAULT_SHARED_MACHINE_PATHS (which already includes those fragments) instead of re-narrowing it here.
    return scc.is_ours_shared_machine


def _circuit_breaker_check(running, elapsed_s, log_dir=QUEUE_LOG_DIR, read_text=None,
                           state=None, now=None):
    """Returns (should_run_circuit, error_signature). should_run_circuit is True if this crash looks
    like a stale-state trip worth an immediate cleanup+retry: either it died within
    CIRCUIT_FAST_FAIL_S of its own launch, or its error signature matches the immediately preceding
    crashed job's signature (2 consecutive jobs, same real error)."""
    state = state if state is not None else _load_circuit_state()
    sig = _tail_error_signature(running, log_dir=log_dir, read_text=read_text)
    fast_fail = elapsed_s is not None and elapsed_s < CIRCUIT_FAST_FAIL_S
    same_as_last = sig is not None and sig == state.get("last_crash_signature")
    return (fast_fail or same_as_last), sig


def tick(pid_alive_fn=pid_alive, heartbeat_age_fn=heartbeat_age_s, cleanup_fn=None, sleep_fn=time.sleep,
        now_fn=time.time, ownership_fn=None, load_circuit_state_fn=None, save_circuit_state_fn=None,
        tail_signature_fn=None):
    """One watchdog decision cycle. Injectable pid_alive_fn/heartbeat_age_fn for tests; production defaults hit the
    real machine. Returns a dict describing what happened, for logging and for tests to assert on.

    Maintenance lock (2026-10-01): if q.is_paused() returns a pause record, this tick does nothing at all --
    no crash detection, no launching -- and returns immediately. See t2s_queue.set_pause's docstring for why:
    an operator doing ad-hoc debug work (which may itself start/stop a runtime process) must be able to fully
    quiesce the watchdog first, not just block new launches, since the watchdog's crash-detection path could
    otherwise mark a deliberately-stopped job "crashed" and launch the next one mid-debug."""
    pause = q.is_paused()
    if pause is not None:
        return {"action": "paused", "reason": pause}

    # Resolved once, up front, so both the crash branch (circuit-breaker cleanup) and the pending branch
    # (t2s_queue.launch_next's own pre-launch cleanup, X1 2026-10-02) share the same injected test doubles --
    # a test that stubs cleanup_fn/ownership_fn for tick() must not have launch_next silently fall through to
    # the real process-killing defaults underneath it.
    cleanup_fn = cleanup_fn or (lambda ownership: scc.cleanup_stale_servers(ownership))
    ownership_fn = ownership_fn or host_ownership_predicate()
    launch_cleanup = lambda: cleanup_fn(ownership_fn)  # noqa: E731  (local adapter: launch_next's cleanup_fn takes no args)

    items = q.read_queue()
    running = next((it for it in items if it["status"] == "running"), None)

    if running is not None:
        if running.get("pid") is None:
            # A launch that never produced a pid should have been reverted to "pending" by launch_next() itself
            # (see t2s_queue.py); if one is still seen here, leave it alone rather than guess -- do not mark a
            # never-launched job crashed.
            return {"action": "none", "reason": f"entry {running['id']} has status running but no pid; leaving alone"}
        if pid_alive_fn(running.get("pid")):
            return {"action": "none", "reason": f"entry {running['id']} running, pid {running.get('pid')} alive"}
        age = heartbeat_age_fn(running)
        if not is_stale(running, age):
            age_str = f"{age:.0f}s" if age is not None else "unknown"
            reason = (f"entry {running['id']}: pid {running.get('pid')} check says dead but heartbeat age "
                      f"{age_str} is still fresh; trusting the heartbeat, leaving running")
            return {"action": "pid_check_disagrees", "reason": reason}
        # 2026-10-02 circuit breaker: a crashed job is never just left "crashed" (a real crash cascade
        # on evo-x2 burned through 5 queued jobs in under an hour because every one of them inherited
        # the same stale-process-guard trip from the first). Every crash is requeued (status back to
        # "pending", own attempt counter incremented) unless it has already used its 3rd attempt. A
        # crash that looks like a stale-state trip specifically -- died within CIRCUIT_FAST_FAIL_S of
        # its own launch, or its error signature matches the immediately preceding crash's -- also
        # gets an immediate cleanup (stop any of our own stray ollama/llama-server processes, wait for
        # the port(s) to free) and a CIRCUIT_CLEANUP_WAIT_S pause before that retry, and is placed
        # FIRST among pending entries so the very next launch is the retry itself, not unrelated work.
        load_state = load_circuit_state_fn or _load_circuit_state
        save_state = save_circuit_state_fn or _save_circuit_state
        tail_sig = tail_signature_fn or _tail_error_signature

        now = now_fn()
        started_ts = running.get("started_ts")
        elapsed_s = (now - started_ts) if started_ts else None
        state = load_state()
        sig = tail_sig(running)
        fast_fail = elapsed_s is not None and elapsed_s < CIRCUIT_FAST_FAIL_S
        same_as_last = sig is not None and sig == state.get("last_crash_signature")
        circuit_triggered = fast_fail or same_as_last

        attempt = int(running.get("attempt", 1))
        age_str = f"{age:.0f}s" if age is not None else "no heartbeat log found"
        base_note = f"watchdog: pid {running.get('pid')} dead and heartbeat stale ({age_str}), attempt {attempt}"
        save_state({"last_crashed_id": running["id"], "last_crash_signature": sig})

        if attempt >= CIRCUIT_MAX_ATTEMPTS:
            running["status"] = "failed_max_attempts"
            running["finished_ts"] = now
            running["note"] = base_note + f" -- {CIRCUIT_MAX_ATTEMPTS} attempts exhausted, giving up, not retrying"
            q.write_queue(items)
            launched = q.launch_next(items, cleanup_fn=launch_cleanup)
            result = {"action": "failed_max_attempts", "reason": running["note"]}
            if launched is not None:
                result["launched"] = launched["id"]
            return result

        cleanup_result = None
        if circuit_triggered:
            cleanup_result = cleanup_fn(ownership_fn)
            sleep_fn(CIRCUIT_CLEANUP_WAIT_S)

        running["status"] = "pending"
        running["attempt"] = attempt + 1
        running["note"] = base_note + (
            f" -- circuit breaker triggered (fast_fail={fast_fail}, same_error_as_last={same_as_last}), "
            f"ran cleanup and waited {CIRCUIT_CLEANUP_WAIT_S}s, retrying immediately"
            if circuit_triggered else " -- requeued for a normal retry (not an immediate-retry trigger)")
        for k in ("pid", "started_ts", "launch_result"):
            running.pop(k, None)
        items = [it for it in items if it is not running]
        if circuit_triggered:
            items = [running] + items  # immediate retry: first in line
        else:
            items = items + [running]  # ordinary requeue: back of the line, other work goes first
        q.write_queue(items)
        # A circuit-triggered retry already ran cleanup_fn above (lines just before); launch_next doing its own
        # pre-launch sweep again immediately after would be a redundant second real cleanup call for no benefit.
        # The ordinary (non-circuit) requeue path never ran cleanup_fn at all, so there launch_next's own sweep
        # is still the thing that clears any stray server this crash may have left before a possibly-different
        # pending entry launches.
        next_launch_cleanup = (lambda: None) if circuit_triggered else launch_cleanup
        launched = q.launch_next(items, cleanup_fn=next_launch_cleanup)
        result = {"action": "crashed_requeued", "reason": running["note"], "circuit_triggered": circuit_triggered,
                 "cleanup_result": cleanup_result, "attempt": running["attempt"]}
        if launched is not None:
            result["launched"] = launched["id"]
        else:
            q.write_empty_flag(items, f"entry {running['id']} requeued, no pending entry could launch")
        return result

    pending = next((it for it in items if it["status"] == "pending"), None)
    if pending is not None:
        launched = q.launch_next(items, cleanup_fn=launch_cleanup)
        if launched is None:
            # every remaining pending entry has an unsatisfied gate (see t2s_queue._gate_satisfied) -- nothing to
            # launch this tick, but this is not the same as an empty queue, so say so rather than claim "launched".
            still_pending = [it["id"] for it in items if it["status"] == "pending"]
            q.write_empty_flag(items, f"all {len(still_pending)} remaining pending entries are gate-blocked: {still_pending}")
            return {"action": "gate_blocked", "reason": still_pending}
        return {"action": "launched", "launched": launched["id"]}

    q.write_empty_flag(items, "watchdog: nothing running and nothing pending")
    return {"action": "empty_flag"}


def run_digest():
    """Calls analysis/results_digest.py as a subprocess (read-only; see that module's own docstring) so the digest
    stays fresh on the watchdog's own cadence without depending on a controller session being awake. Best-effort:
    a digest failure must never affect the watchdog's own tick() decision, so this is called after tick() and any
    exception here is only logged, not raised."""
    import subprocess
    script = DEPLOY / "results_digest.py"
    if not script.exists():
        return {"action": "skipped", "reason": "results_digest.py not deployed"}
    try:
        r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
        return {"action": "ran", "rc": r.returncode, "stdout": r.stdout.strip()[:300]}
    except Exception as e:
        return {"action": "error", "reason": repr(e)[:200]}


def main():
    result = tick()
    _log(json.dumps(result, default=str))
    digest_result = run_digest()
    _log(json.dumps({"digest": digest_result}, default=str))


if __name__ == "__main__":
    main()
