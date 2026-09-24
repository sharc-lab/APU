# RAM Cap Protocol — Boot-Time Memory Constraint (Primary Method)

## Why this replaces the in-process balloon as the primary method

Two balloon attempts (`memory_balloon.py` v1 and v2) both failed to produce a
genuinely constrained measurement, for two different reasons, both rooted in
the same fact: **any pressure enforced above the OS's own memory manager can
be undone by that memory manager.**

- **v1** (`results/memory_pressure_20260924T025611Z.jsonl`, INVALID): held
  `Available MBytes` at a moving target via a control loop. When the server
  allocated, the balloon's own code released memory to compensate. The
  server was never constrained at any of 7 tested levels.
- **v2** (`results/memory_pressure_v2_20260924T034948Z.jsonl`, INVALID): fixed
  the control-loop bug (allocation size never changes during a level) but
  the balloon was unlocked. Windows itself trimmed the balloon's resident
  working set under pressure (57866.9 MB → 28712.5 MB observed at one point)
  and handed the freed pages to the server — the identical net effect as v1,
  produced by the OS's memory manager instead of this script's own logic.

A **boot-time RAM cap** removes this failure mode entirely: the physical
ceiling is enforced by the boot loader before Windows's memory manager (or
any process, including a balloon) ever runs. There is no control loop to
have a bug, and no working set for the OS to trim, because the memory
genuinely does not exist as far as Windows is concerned for the entire
session. This is also why it requires a reboot per level and cannot be
scripted end-to-end the way the balloon was — the cap is a boot-time BCD
setting, not a runtime one.

---

## The `bcdedit truncatememory` command

```
bcdedit /set {current} truncatememory <bytes>
```

Restricts Windows to using physical memory addresses below `<bytes>`. This
is the same mechanism `msconfig.exe`'s Boot Advanced Options → "Maximum
memory" checkbox uses internally, exposed directly via `bcdedit` for exact,
scriptable byte values instead of the GUI's MB field.

**Byte values for the four levels** (binary GiB, matching how Windows itself
reports `TotalPhysicalMemory`/`TotalVisibleMemorySize`):

| Level | Bytes | Command |
|---|---|---|
| 24 GB | 25769803776 | `bcdedit /set {current} truncatememory 25769803776` |
| 16 GB | 17179869184 | `bcdedit /set {current} truncatememory 17179869184` |
| 12 GB | 12884901888 | `bcdedit /set {current} truncatememory 12884901888` |
| 10 GB | 10737418240 | `bcdedit /set {current} truncatememory 10737418240` |

Run from an **elevated** Command Prompt or PowerShell. Takes effect on the
**next boot** — reboot after setting it.

### Revert command (removes the cap entirely, restores full physical RAM)

```
bcdedit /deletevalue {current} truncatememory
```

Also requires a reboot to take effect. **Run this immediately after the
lowest level's measurements are done** — do not leave the machine capped
between sessions.

### Verifying the cap took effect after boot

Two separate queries, because they can legitimately disagree:

```powershell
(Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1MB   # OS-visible/usable, KB -> converted to "MB" here for readability
(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB       # installed hardware, as reported by SMBIOS
```

`TotalVisibleMemorySize` is the one that should reflect the cap (it is what
the OS actually made usable this boot). `TotalPhysicalMemory` may continue
to report the full installed amount on some systems/firmware, since it can
be querying hardware inventory rather than what the boot loader actually
mapped — **do not use `TotalPhysicalMemory` alone to confirm the cap; use
`TotalVisibleMemorySize`.** Expect `TotalVisibleMemorySize` to be a little
below the exact requested value (typically some tens to a few hundred MB
less), because a `truncatememory` ceiling counts from address 0 and a
handful of low physical address ranges are reserved for firmware/hardware
regardless of the cap — this is normal and expected, not a sign the cap
didn't apply. If `TotalVisibleMemorySize` is instead close to the *full*
uncapped amount, the cap did not take effect (commonly: forgot to reboot
after setting it, or set it on the wrong boot entry — `{current}` should be
correct for a single-boot-entry machine, but confirm with `bcdedit /enum`
if in doubt).

### Floor — never go below this

**8 GB (8589934592 bytes).** Windows 11's documented minimum is 4 GB, but
that is a bare-boot minimum, not a margin for reliably running SSH, a
monitoring/measurement script, and the model server together. 8 GB gives at
least 2 GB of headroom below the lowest *tested* level (10 GB) that is never
itself tested — it exists purely as a hard ceiling on how low this protocol
is ever allowed to go, automated or manual. If a future need arises to test
below 10 GB, that requires an explicit, separate decision to lower the floor
itself, not an extension of this sweep.

**Recovery if a cap ever prevents a usable boot:** since this runs on
EVO-X2 with physical keyboard access, recovery is: reboot, interrupt into
Windows Advanced Boot Options (hold Shift while clicking Restart, or repeated
F8/hold-power depending on firmware), reach a Command Prompt from Advanced
Startup Options (Troubleshoot → Advanced Options → Command Prompt, which
runs pre-boot and is unaffected by `truncatememory`), and run the revert
command there. This is the reason boot-time capping is being done on the
machine with physical access first, not on evo-t2s remotely — a bad cap on
a headless/remote box would require someone at *that* keyboard to recover,
which is what "do not run it on evo-t2s" is protecting against.

---

## Per-level measurement spec

Run on **EVO-X2 only**, once physically at the keyboard for each reboot.
The measurement script itself (run after each boot, before rebooting to the
next level) is a straightforward adaptation of the existing evo-t2s harness
pattern (`memory_pressure_experiment.py`'s server/probe logic) with the
balloon entirely removed — the cap is already in effect at boot, so there is
nothing to allocate or hold in-process. Not yet written; ready to build once
EVO-X2 SSH/Tailscale access is confirmed (see setup script below), since at
that point it can be launched and monitored the same way every other
experiment in this project has been.

**Grid:** 4 RAM-cap levels (24, 16, 12, 10 GB) × 2 KV precisions (f16, q4_0)
× ctx=32768, `-fa on`, qwen3-4b-instruct GGUF (sha256
`85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9`, verified
before every run, refuse to proceed on mismatch — same check already used in
`contention_experiment.py`).

**Per (level, precision) cell, record:**
- Server load success (yes/no) and load time
- TTFT and decode tok/s on a ~90%-fill ctx=32768 prompt (same F-NUM filler
  construction as `bw_saturation_sweep.py`, for direct comparability to the
  existing unconstrained baseline)
- 5 art probes (art_01-05, not all 10 — half the probe count of the invalid
  balloon smoke, since each full-context correctness call costs ~330-345s at
  baseline regardless of pressure, confirmed empirically above; 5 probes
  keeps one cell's correctness phase to ~28-30 min instead of ~56) at the
  same ~90%-fill ctx=32768 construction, `cache_prompt: false`
- Hard page faults/sec (`\Memory\Pages/sec` — the precise disk-paging
  signal, not the noisier `\Memory\Page Faults/sec`, per the same
  distinction already established for the balloon experiments)
- Pagefile usage (`Win32_PageFileUsage`: AllocatedBaseSize, CurrentUsage,
  PeakUsage) before and after the cell
- Server process private bytes and working set (`PrivateMemorySize64`,
  `WorkingSet64` via a single-process `Get-Process`/`Get-CimInstance` query
  on the server's own PID — never system-wide free memory, consistent with
  every other experiment in this project)
- iGPU shared-memory limit as Windows reports it (`Get-CimInstance
  Win32_VideoController` → `AdapterRAM`, and/or `dxdiag`/DirectX shared
  system memory figures — Strix Halo's iGPU draws its VRAM allocation from
  this same capped physical pool, so this figure should itself shrink as the
  RAM cap tightens; recording it confirms the cap is actually constraining
  the GPU-visible pool, not just CPU-visible system RAM)
- Exact error text on any failure (HTTP error body, process exit code +
  stderr, or explicit "hung past timeout" if neither)

**Total calls:** 4 levels × 2 precisions × (1 throughput + 5 correctness) =
**48 calls**, across **8 reboots** (4 levels, cap set once and both
precisions measured per boot before moving to the next level — no reason to
reboot between precisions within the same RAM level, since the OS-level cap
doesn't change).

**Classification** (post-hoc from logged data, same four categories used
throughout this project): `runs_normally` / `pages_and_slows` / `fails_loudly`
/ `fails_silently`.

---

## Secondary method (design only, not built): a genuinely locked balloon

For fine-grained steps that don't require a reboot per level (e.g. probing
exactly where a transition sits, once the RAM-cap sweep above has bounded
it), a locked in-process balloon is worth building correctly — the two
failed balloon attempts above establish precisely what "correctly" requires.

**Requirement:** pages must be pinned in physical RAM such that the OS
cannot trim, compress, or page them out under any pressure, for the whole
duration of a level. Two Windows APIs can do this; both need the same
underlying privilege.

### Option A — `VirtualLock` (already partially built)

`memory_balloon.py` v2 already attempts this (`FixedBalloon.allocate()` in
that script): raise the process's working-set maximum via
`SetProcessWorkingSetSizeEx`, then call `VirtualLock(address, size)` on the
full allocation. This is the simpler of the two options and requires no new
code beyond what already exists — it just needs the privilege below granted
to actually succeed, which it currently does not (confirmed empirically:
`AdjustTokenPrivileges` returns `ERROR_NOT_ALL_ASSIGNED` for `sharc`).

### Option B — `AllocateUserPhysicalPages` (AWE, stronger guarantee)

Address Windowing Extensions reserves physical pages directly, outside the
normal virtual-memory system entirely (the same mechanism SQL Server's "Lock
Pages in Memory" feature uses for its buffer pool). Pages allocated this way
are never subject to the ordinary working-set trimming that defeated
`VirtualLock`-without-privilege in principle, and are documented as a
stronger guarantee than `VirtualLock` even in some edge cases where the
latter's pinning can still be affected by extreme system-wide conditions.
Requires `AllocateUserPhysicalPages()` to reserve the physical page numbers,
then `MapUserPhysicalPages()` to map them into the process's address space.
More code than Option A for the same underlying effect; worth doing only if
Option A (once privileged) turns out to still be insufficient in practice.

### The privilege both options need: `SeLockMemoryPrivilege`

Neither API can lock more than a trivial amount of memory without it. To
grant it to a single account (`sharc`, or whichever account will run the
balloon):

```
secpol.msc → Local Policies → User Rights Assignment → "Lock pages in memory" → Add <account>
```

Then **log that account off and back on** — privileges are baked into a
logon token at logon time; granting the policy does not retroactively apply
to an already-open session. A machine reboot also works and is simpler to
reason about, though a plain logoff/logon of just that account is
sufficient.

Command-line equivalent (for scripting the grant itself, e.g. via a remote
session, still requires the subsequent logoff/logon): `secedit` can export,
modify, and re-import the local security policy non-interactively, but this
is a more fragile, multi-step process than the GUI for a one-time grant on
one account — the GUI path above is recommended unless there's a specific
need to script the grant itself.

**After granting:** re-run `memory_balloon.py` v2 unmodified (it already
attempts the lock and reports the outcome verbatim) — its own stdout/log
header will read `LOCK STATUS: LOCKED` instead of `UNLOCKED` once the
privilege is actually in effect, and that is the confirmation to trust, not
an assumption that granting the policy alone was sufficient.
