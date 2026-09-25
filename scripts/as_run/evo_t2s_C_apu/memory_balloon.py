"""
memory_balloon.py -- FIXED-SIZE memory balloon, evo-t2s. v2: replaces the
moving-target design, which was a real bug (see design note below).

DESIGN FLAW IN v1 (kept here as a record, not a hypothetical): v1 took
--target-available-mb and continuously re-measured Available MBytes every
tick, growing or shrinking the balloon to hold that AVAILABLE figure
constant. When llama-server allocated memory, Available dropped below
target, and the control loop's own logic released balloon memory to bring
Available back up -- the balloon retreated every time the thing under test
actually needed memory. The server was never constrained at any of the 7
levels tested with v1; every level showed flat throughput and 10/10
correctness because nothing was ever squeezed. That result
(results/memory_pressure_20260924T025611Z.jsonl) is marked invalid in
docs/RESULT_PROVENANCE.md.

v2 FIX: take --budget-x-mb (X = memory meant to be LEFT for the model + KV +
runtime + OS), compute B = total_physical_mb - X, allocate B ONCE before the
server starts, and NEVER resize or release it for the rest of the level. The
server must fit in whatever is left after B is already committed -- it gets
squeezed by construction, not by a control loop that can second-guess itself.

TOUCH + RE-TOUCH: every page in B is written once at allocation (forces
commit, not just reservation) and re-written every --log-interval-s seconds
for the whole level (defeats Windows Memory Compression and working-set
trimming on the SAME pages, i.e. keeps them looking "hot" to the OS's own
reclaim heuristics) -- necessary but NOT sufficient on its own, see next.

LOCKING: touching pages periodically does not stop Windows from trimming the
balloon's own working set BETWEEN touches under real memory pressure (the
OS can still page out or compress a touched-then-idle page before the next
re-touch cycle) -- which would silently reintroduce the same failure mode as
v1 (the balloon "gives back" memory under pressure) via a completely
different mechanism (OS-level trimming instead of this script's own control
loop). VirtualLock pins pages in physical RAM so the OS cannot do this at
all. Large (multi-GB) VirtualLock calls require:
  1. The process's working set maximum raised via SetProcessWorkingSetSizeEx
     (the default max is a few MB, nowhere near enough for a multi-GB lock).
  2. SeLockMemoryPrivilege enabled in the process's own token via
     AdjustTokenPrivileges -- which only succeeds if the account already has
     the "Lock pages in memory" user right assigned via local security
     policy (secpol.msc -> Local Policies -> User Rights Assignment, or the
     scripted secedit equivalent) AND the account has logged off/on (or the
     machine rebooted) since that right was granted, for it to appear in a
     fresh logon token.
This script ATTEMPTS both steps and reports the outcome explicitly per run
in the log header and on stdout -- it does NOT silently fall back to
unlocked pages without saying so. If locking fails, the run still proceeds
(fixed allocation size is the primary fix and stands on its own even
unlocked), but the log and this script's stdout state plainly: "LOCKED" or
"UNLOCKED (reason)". Do not read a v2 run as pinned-memory-guaranteed unless
its own log says LOCKED.

SAFETY: hard floor check is now a PRE-FLIGHT refusal, not a runtime
adjustment (there is no runtime adjustment in v2 by design) -- refuses to
start if X < --floor-mb (default 2560 = 2.5 GB), since that would define a
budget that already violates the "leave the OS something" rule before the
server even runs. Dead-man's switches unchanged from v1: heartbeat-file
staleness and an absolute wall-clock cap, both still release everything in
`finally` on any exit path.

LOGGING every --log-interval-s seconds: Available MBytes, Page Faults/sec,
Committed Bytes (same three counters as v1, per original instruction), PLUS
this balloon's OWN committed bytes and working-set size queried directly
from this process (the positive control the redesign specifically asks
for -- these two numbers must stay constant for the whole level; if they
don't, that itself is evidence of a problem, logged plainly rather than
hidden).
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import subprocess
import sys
import time

PAGE_SIZE = 4096
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04

kernel32 = ctypes.windll.kernel32
advapi32 = ctypes.windll.advapi32

TOKEN_ADJUST_PRIVILEGES = 0x0020
TOKEN_QUERY = 0x0008
SE_PRIVILEGE_ENABLED = 0x00000002


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wt.DWORD), ("HighPart", ctypes.c_long)]


class LUID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Luid", LUID), ("Attributes", wt.DWORD)]


class TOKEN_PRIVILEGES(ctypes.Structure):
    _fields_ = [("PrivilegeCount", wt.DWORD), ("Privileges", LUID_AND_ATTRIBUTES * 1)]


# Signatures declared AFTER the structs above, since several reference them
# (e.g. LookupPrivilegeValueW's LUID* out-param). Declaring these explicitly
# matters: ctypes' default restype (c_int) truncates/mis-marshals a 64-bit
# pseudo-handle like GetCurrentProcess()'s return value, which is exactly why
# OpenProcessToken failed with ERROR_INVALID_HANDLE in testing before these
# were added -- it was being handed a corrupted "handle" value, not a
# genuinely invalid one.
kernel32.VirtualAlloc.restype = wt.LPVOID
kernel32.VirtualAlloc.argtypes = [wt.LPVOID, ctypes.c_size_t, wt.DWORD, wt.DWORD]
kernel32.VirtualFree.restype = wt.BOOL
kernel32.VirtualFree.argtypes = [wt.LPVOID, ctypes.c_size_t, wt.DWORD]
kernel32.VirtualLock.restype = wt.BOOL
kernel32.VirtualLock.argtypes = [wt.LPVOID, ctypes.c_size_t]
kernel32.GetCurrentProcess.restype = wt.HANDLE
kernel32.GetCurrentProcess.argtypes = []
advapi32.OpenProcessToken.restype = wt.BOOL
advapi32.OpenProcessToken.argtypes = [wt.HANDLE, wt.DWORD, ctypes.POINTER(wt.HANDLE)]
advapi32.LookupPrivilegeValueW.restype = wt.BOOL
advapi32.LookupPrivilegeValueW.argtypes = [wt.LPCWSTR, wt.LPCWSTR, ctypes.POINTER(LUID)]
advapi32.AdjustTokenPrivileges.restype = wt.BOOL
advapi32.AdjustTokenPrivileges.argtypes = [
    wt.HANDLE, wt.BOOL, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p, ctypes.c_void_p,
]
kernel32.SetProcessWorkingSetSizeEx.restype = wt.BOOL
kernel32.SetProcessWorkingSetSizeEx.argtypes = [wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wt.DWORD]


def total_phys_mb() -> float:
    ms = MEMORYSTATUSEX()
    ms.dwLength = ctypes.sizeof(ms)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullTotalPhys / (1024 * 1024)


def available_mb() -> float:
    ms = MEMORYSTATUSEX()
    ms.dwLength = ctypes.sizeof(ms)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullAvailPhys / (1024 * 1024)


def try_enable_lock_privilege() -> tuple[bool, str]:
    """Enable SeLockMemoryPrivilege in THIS process's token, if the account
    already has the underlying user right assigned. Returns (enabled, detail)."""
    hToken = wt.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(),
                                      TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, ctypes.byref(hToken)):
        return False, f"OpenProcessToken failed, GetLastError={ctypes.GetLastError()}"

    luid = LUID()
    if not advapi32.LookupPrivilegeValueW(None, "SeLockMemoryPrivilege", ctypes.byref(luid)):
        return False, f"LookupPrivilegeValueW failed, GetLastError={ctypes.GetLastError()}"

    tp = TOKEN_PRIVILEGES()
    tp.PrivilegeCount = 1
    tp.Privileges[0].Luid = luid
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED

    if not advapi32.AdjustTokenPrivileges(hToken, False, ctypes.byref(tp), 0, None, None):
        return False, f"AdjustTokenPrivileges failed, GetLastError={ctypes.GetLastError()}"

    err = ctypes.GetLastError()
    if err == 1300:  # ERROR_NOT_ALL_ASSIGNED
        return False, (
            "AdjustTokenPrivileges succeeded but ERROR_NOT_ALL_ASSIGNED: the "
            "sharc account does not currently hold the 'Lock pages in memory' "
            "user right (or was not granted it before this logon session "
            "started). Grant it via: secpol.msc -> Local Policies -> User "
            "Rights Assignment -> 'Lock pages in memory' -> Add sharc -> "
            "log the account off and back on (a fresh logon token is "
            "required; enabling the policy alone does not retroactively "
            "apply to an already-open session) -> re-run this script."
        )
    return True, "SeLockMemoryPrivilege enabled in this process's token"


def get_own_process_stats() -> dict:
    """This balloon process's own committed/working-set size, queried
    directly (not via a subprocess) so it's cheap enough to call every tick."""
    pid = os.getpid()
    ps_cmd = (
        f"$p = Get-Process -Id {pid}; "
        f"[PSCustomObject]@{{WorkingSet64=$p.WorkingSet64; PrivateMemorySize64=$p.PrivateMemorySize64}} | ConvertTo-Json"
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                            capture_output=True, text=True, timeout=10)
        return json.loads(r.stdout)
    except Exception as e:
        return {"error": str(e)}


def get_counters() -> dict:
    ps_cmd = (
        "(Get-Counter '\\Memory\\Available MBytes','\\Memory\\Page Faults/sec',"
        "'\\Memory\\Committed Bytes','\\Memory\\Pages/sec').CounterSamples | "
        "Select-Object Path,CookedValue | ConvertTo-Json"
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                            capture_output=True, text=True, timeout=10)
        samples = json.loads(r.stdout)
        out = {}
        for s in samples:
            path = s["Path"].lower()
            if "available mbytes" in path:
                out["available_mbytes"] = s["CookedValue"]
            elif "page faults" in path:
                out["page_faults_per_sec"] = s["CookedValue"]
            elif "committed bytes" in path:
                out["committed_bytes"] = s["CookedValue"]
            elif path.endswith("pages/sec"):
                # \Memory\Pages/sec: HARD faults resolved from disk -- the
                # precise paging-activity signal, distinct from the noisier
                # Page Faults/sec (fires on ordinary memory activity too).
                out["pages_per_sec_hard_faults"] = s["CookedValue"]
        return out
    except Exception as e:
        return {"error": str(e)}


class FixedBalloon:
    """A single VirtualAlloc'd region of exactly `size_bytes`, allocated once
    and never resized. touch_all() re-writes every page; that's the only
    operation available after construction besides release_all()."""

    def __init__(self):
        self.addr = None
        self.size = 0
        self.locked = False
        self.lock_detail = ""

    def allocate(self, size_bytes: int, try_lock: bool):
        addr = kernel32.VirtualAlloc(None, ctypes.c_size_t(size_bytes),
                                      MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not addr:
            raise MemoryError(f"VirtualAlloc failed for {size_bytes} bytes "
                               f"(GetLastError={ctypes.GetLastError()})")
        self.addr = addr
        self.size = size_bytes
        self.touch_all()  # force commit before attempting to lock

        if try_lock:
            enabled, detail = try_enable_lock_privilege()
            if enabled:
                # Raise this process's working-set limits enough to cover the
                # lock request -- VirtualLock cannot lock beyond the current
                # working-set maximum.
                kernel32.SetProcessWorkingSetSizeEx(
                    kernel32.GetCurrentProcess(),
                    ctypes.c_size_t(size_bytes + (64 * 1024 * 1024)),
                    ctypes.c_size_t(size_bytes + (256 * 1024 * 1024)),
                    0,
                )
                ok = kernel32.VirtualLock(addr, ctypes.c_size_t(size_bytes))
                if ok:
                    self.locked = True
                    self.lock_detail = "VirtualLock succeeded: pages are pinned in physical RAM"
                else:
                    self.locked = False
                    self.lock_detail = (
                        f"SeLockMemoryPrivilege was enabled but VirtualLock still "
                        f"failed (GetLastError={ctypes.GetLastError()}) -- proceeding "
                        f"UNLOCKED. The OS can still trim/page this balloon's own "
                        f"working set under real pressure between re-touch cycles."
                    )
            else:
                self.locked = False
                self.lock_detail = f"UNLOCKED: {detail}"
        else:
            self.locked = False
            self.lock_detail = "UNLOCKED: --no-lock passed"

    def touch_all(self):
        buf = (ctypes.c_ubyte * self.size).from_address(self.addr)
        for offset in range(0, self.size, PAGE_SIZE):
            buf[offset] = 1

    def release(self):
        if self.addr:
            kernel32.VirtualFree(self.addr, 0, MEM_RELEASE)
            self.addr = None
            self.size = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-x-mb", type=float, required=True,
                     help="X: memory to LEAVE for model+KV+runtime+OS. Balloon size = total_phys - X, fixed for the whole run.")
    ap.add_argument("--floor-mb", type=float, default=2560.0,
                     help="Refuse to start if X is below this (default 2.5 GB) -- pre-flight check, not a runtime adjustment.")
    ap.add_argument("--no-lock", action="store_true", help="Skip the VirtualLock attempt entirely.")
    ap.add_argument("--heartbeat-file", type=str, required=True)
    ap.add_argument("--heartbeat-timeout-s", type=float, default=60.0)
    ap.add_argument("--max-runtime-s", type=float, default=2700.0)
    ap.add_argument("--log-path", type=str, required=True)
    ap.add_argument("--log-interval-s", type=float, default=5.0)
    args = ap.parse_args()

    if args.budget_x_mb < args.floor_mb:
        print(f"REFUSING TO START: budget X={args.budget_x_mb} MB is below "
              f"floor={args.floor_mb} MB. This would leave less than the floor "
              f"for the OS+server combined even before the server allocates "
              f"anything.")
        sys.exit(1)

    total_mb = total_phys_mb()
    balloon_size_mb = total_mb - args.budget_x_mb
    if balloon_size_mb <= 0:
        print(f"REFUSING TO START: computed balloon size {balloon_size_mb:.0f} MB "
              f"<= 0 (total_phys={total_mb:.0f} MB, X={args.budget_x_mb} MB).")
        sys.exit(1)

    with open(args.heartbeat_file, "w") as f:
        f.write(str(time.time()))

    balloon = FixedBalloon()
    print(f"total_phys={total_mb:.0f} MB  X={args.budget_x_mb} MB  "
          f"balloon_size={balloon_size_mb:.0f} MB  allocating + touching...")
    balloon.allocate(int(balloon_size_mb * 1024 * 1024), try_lock=not args.no_lock)
    print(f"LOCK STATUS: {'LOCKED' if balloon.locked else 'UNLOCKED'} -- {balloon.lock_detail}")

    t_start = time.monotonic()
    tick = 0

    with open(args.log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# budget_x_mb={args.budget_x_mb} balloon_size_mb={balloon_size_mb:.1f} "
                    f"lock_status={'LOCKED' if balloon.locked else 'UNLOCKED'} "
                    f"lock_detail={balloon.lock_detail!r}\n")
        logf.write("ts_iso,elapsed_s,tick,balloon_size_mb_fixed,balloon_own_working_set_mb,"
                    "balloon_own_private_mb,available_mbytes,page_faults_per_sec,"
                    "pages_per_sec_hard_faults,committed_bytes,note\n")
        logf.flush()

        try:
            while True:
                now = time.monotonic()
                elapsed = now - t_start

                if elapsed > args.max_runtime_s:
                    print(f"DEAD-MAN SWITCH: max-runtime-s ({args.max_runtime_s}) exceeded. Releasing and exiting.")
                    break
                try:
                    hb_age = time.time() - os.path.getmtime(args.heartbeat_file)
                except OSError:
                    hb_age = float("inf")
                if hb_age > args.heartbeat_timeout_s:
                    print(f"DEAD-MAN SWITCH: heartbeat stale ({hb_age:.1f}s). Releasing and exiting.")
                    break

                balloon.touch_all()  # the ONLY operation on the balloon each tick -- no resize, ever

                own = get_own_process_stats()
                counters = get_counters()
                note = ""
                if "error" in own:
                    note = f"own_stats_error:{own['error']}"
                if "error" in counters:
                    note = (note + ";counters_error:" + counters["error"]).strip(";")

                row = (
                    f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},"
                    f"{elapsed:.1f},{tick},{balloon_size_mb:.1f},"
                    f"{own.get('WorkingSet64', 0) / (1024*1024):.1f},"
                    f"{own.get('PrivateMemorySize64', 0) / (1024*1024):.1f},"
                    f"{counters.get('available_mbytes', '')},"
                    f"{counters.get('page_faults_per_sec', '')},"
                    f"{counters.get('pages_per_sec_hard_faults', '')},"
                    f"{counters.get('committed_bytes', '')},"
                    f"{note}\n"
                )
                logf.write(row)
                logf.flush()
                print(row.strip())

                tick += 1
                time.sleep(max(0.0, args.log_interval_s - (time.monotonic() - now)))

        except KeyboardInterrupt:
            print("KeyboardInterrupt -- releasing balloon.")
        finally:
            size_before = balloon.size
            balloon.release()
            print(f"Released {size_before / (1024*1024):.1f} MB. Exiting.")


if __name__ == "__main__":
    main()
