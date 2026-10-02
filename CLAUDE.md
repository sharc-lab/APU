# Controller-laptop rules

These apply to any agent or subagent working in this repo from the controller laptop (not evo-t2s/evo-x2,
which are separate remote machines reached only by SSH/SCP). A subagent does not inherit a parent session's
own promises or conversation context, so these are written down here instead.

## No visible windows on the controller

2026-10-02: terminal windows were flashing open and closing on the controller laptop during background/
scheduled work. Root cause: a console-less parent (pythonw.exe, used by the real APU-SyncResults scheduled
task) spawning a console-subsystem child (ssh.exe, scp.exe, git.exe, powershell.exe) with a bare
`subprocess.run`/`Popen`/`check_output` call -- the child has no console to inherit, so Windows allocates a
brand-new, visible one for every single call. Measured live: one sync_results.py run produced 109 new visible
windows (and a real scp.exe crash dialog) before the fix; 0 after.

- **From Python**: never call `subprocess.run`/`Popen`/`check_output`/`call` directly in a script that can run
  from a scheduled task or in the background on this laptop. Use `harness/proc_util.py`'s
  `run_hidden`/`popen_hidden`/`check_output_hidden` instead -- same signature, adds `CREATE_NO_WINDOW` and a
  hidden `STARTUPINFO`, no-ops on non-Windows. `tests/test_no_visible_console_windows.py` greps the known
  controller-side scripts (currently `scripts/sync_results.py`, `scripts/deploy_evo.py`,
  `scripts/t2s_responsiveness_probe.py`) for a bare call and fails the suite if one is found -- add any new
  controller-side script to that test's `CONTROLLER_SCRIPTS` list.
- **From an interactive agent session on this laptop**: never spawn a new visible console. No
  `Start-Process` without `-WindowStyle Hidden`. No `cmd /c start`. No launching a new `powershell.exe`/
  `cmd.exe` window. Run `ssh`/`scp`/`git` directly inside the tool's own shell process (the Bash/PowerShell
  tool already runs hidden; a separately spawned console does not).
- **No ad-hoc scheduled tasks.** Don't register a new Windows scheduled task from inside a session as a
  workaround for anything -- if recurring background work is genuinely needed, say so and use the existing
  pattern (`scripts/install_sync_results_task.ps1`, `scripts/install_queue_watchdog.ps1`) deliberately, not as
  a quick fix.

## Never leave a machine needing a manual resume

evo-t2s and evo-x2 run unattended for long stretches (overnight, over a weekend). 2026-10-02: `t2s_queue`'s
pause flag (`set_pause`/`clear_pause`, see its own docstring) was used to stop the watchdog while a model pull
ran by hand, and nothing else would have cleared it if the session doing that work had ended first -- the
machine would have sat idle, paused, until a human noticed.

- **Any pause you set must have an automatic release**, not just a plan to come back and clear it yourself.
  Either: (a) do the work as a queued job instead of by hand, so the job's own `tq.advance()` call on exit is
  the release, or (b) if you must pause for genuine by-hand debugging, clear it before you stop working this
  turn, not "later" or "next time you check in."
- **Record the release mechanism in the pause reason / queue note** -- e.g. "cleared once x2_model_pulls is
  queued ahead of x2_outcome_table_v2" -- so a later reader (human or agent) can see from the state itself
  whether the machine is waiting on a person or will resume on its own.
- Prefer turning one-off manual intervention into a real queued job (see `harness/x2_model_pulls.py` for the
  pattern: self-checking, safe to re-run, calls `tq.advance()` on exit) over doing it by hand and leaving a
  pause flag as the only thing standing between "paused" and "running."

## Research-reporting conventions

Not written down elsewhere in this repo, so stated here directly:

- **No AI/Claude attribution anywhere** -- no `Co-Authored-By`, no "Generated with" footer, in any commit
  message, PR description, or doc. Commits are authored and committed as the human researcher only.
- **No em dashes** in any new doc or commit message.
- **Report numbers from files only, never from memory.** A claim needs a real result file backing it
  (`results/*.jsonl` or equivalent) before it goes in a doc or a report; see `scripts/sync_results.py`'s own
  docstring for the incident this rule exists to prevent.
- **Statistics convention**: 1 warm-up call + N measured calls, unless a script's own docstring says otherwise.
