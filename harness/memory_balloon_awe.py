"""
memory_balloon_awe.py -- AWE-based locked memory balloon.

Two prior balloons both failed because Windows could take the balloon's own
memory back under pressure: v1's control loop released it voluntarily, v2's
unlocked VirtualAlloc pages got trimmed from its working set by the OS. This
one uses Address Windowing Extensions (AllocateUserPhysicalPages): physical
pages reserved this way are never part of any process's pageable working
set in the first place -- there is no working set for the OS to trim, and
nothing to page out, because these pages were never mapped into a virtual
address space at all. They do not need MapUserPhysicalPages ("mapping") to
serve as a balloon; simply holding the allocation via
AllocateUserPhysicalPages already removes that physical memory from the
system's available pool until FreeUserPhysicalPages is called.

Requires SeLockMemoryPrivilege, enabled on this process's own token at
startup via AdjustTokenPrivileges. If that fails, this script exits non-zero
immediately with the exact error -- it does not silently fall back to a
weaker mechanism when the primary one is unavailable for privilege reasons
(that was the whole failure mode of the earlier VirtualLock attempt: it
degraded silently-ish to "unlocked" and only the log said so). The ONLY
fallback here is VirtualLock-with-raised-working-set, used only if
AllocateUserPhysicalPages itself fails despite the privilege being present
(e.g. a page-array allocation limit) -- and which mechanism was actually
used is always in the log header, never assumed.

TARGET: bring Windows "Available MBytes" down to approximately S (MB),
allocating in chunks (default 2 GiB per AllocateUserPhysicalPages call --
large single calls are more failure-prone and harder to account for).
Once the target is reached (within --tolerance-mb, default 250), NEVER
allocate or free again until told to stop -- no control loop, no resizing,
matching the fixed-allocation lesson from v2 but now with pages the OS
genuinely cannot reclaim out from under it.

SAMPLER every --sample-interval-s (default 5s), for the whole run:
  - held page count (and MB, held_pages * page_size)
  - Available MBytes
  - \\Memory\\Pages/sec (hard faults -- the precise disk-paging signal)
  - \\PhysicalDisk(_Total)\\Disk Read Bytes/sec (weight re-reads from SSD; added 2026-09-25)
  - \\Paging File(_Total)\\% Usage
  - for --server-pid (if provided): PrivateMemorySize64, WorkingSet64,
    \\GPU Process Memory(pid_<PID>*)\\Shared Usage and Dedicated Usage
  - \\GPU Adapter Memory(*)\\Shared Usage

SAFETY VALVE: if Available MBytes stays below 1.0 GB for 120 consecutive
seconds, OR --heartbeat-file is not touched for 300s, free everything, log
"safety_valve", exit. Memory is ALWAYS freed in `finally` regardless of exit
path. AWE-locked pages are also released automatically if this process is
killed outright (Windows reclaims a terminated process's physical page
reservations) -- killing this PID is the manual escape hatch, same as any
other balloon design, but here it's a genuine backstop, not the primary
release mechanism.
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

kernel32 = ctypes.windll.kernel32
advapi32 = ctypes.windll.advapi32

TOKEN_ADJUST_PRIVILEGES = 0x0020
TOKEN_QUERY = 0x0008
SE_PRIVILEGE_ENABLED = 0x00000002
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
MEM_PHYSICAL = 0x00400000
PAGE_READWRITE = 0x04


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


class SYSTEM_INFO(ctypes.Structure):
    _fields_ = [
        ("wProcessorArchitecture", wt.WORD), ("wReserved", wt.WORD),
        ("dwPageSize", wt.DWORD), ("lpMinimumApplicationAddress", ctypes.c_void_p),
        ("lpMaximumApplicationAddress", ctypes.c_void_p), ("dwActiveProcessorMask", ctypes.c_void_p),
        ("dwNumberOfProcessors", wt.DWORD), ("dwProcessorType", wt.DWORD),
        ("dwAllocationGranularity", wt.DWORD), ("wProcessorLevel", wt.WORD),
        ("wProcessorRevision", wt.WORD),
    ]


kernel32.GetCurrentProcess.restype = wt.HANDLE
kernel32.GetCurrentProcess.argtypes = []
advapi32.OpenProcessToken.restype = wt.BOOL
advapi32.OpenProcessToken.argtypes = [wt.HANDLE, wt.DWORD, ctypes.POINTER(wt.HANDLE)]
advapi32.LookupPrivilegeValueW.restype = wt.BOOL
advapi32.LookupPrivilegeValueW.argtypes = [wt.LPCWSTR, wt.LPCWSTR, ctypes.POINTER(LUID)]
advapi32.AdjustTokenPrivileges.restype = wt.BOOL
advapi32.AdjustTokenPrivileges.argtypes = [wt.HANDLE, wt.BOOL, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p, ctypes.c_void_p]
kernel32.SetProcessWorkingSetSizeEx.restype = wt.BOOL
kernel32.SetProcessWorkingSetSizeEx.argtypes = [wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wt.DWORD]
kernel32.VirtualAlloc.restype = wt.LPVOID
kernel32.VirtualAlloc.argtypes = [wt.LPVOID, ctypes.c_size_t, wt.DWORD, wt.DWORD]
kernel32.VirtualFree.restype = wt.BOOL
kernel32.VirtualFree.argtypes = [wt.LPVOID, ctypes.c_size_t, wt.DWORD]
kernel32.VirtualLock.restype = wt.BOOL
kernel32.VirtualLock.argtypes = [wt.LPVOID, ctypes.c_size_t]
kernel32.GetSystemInfo.argtypes = [ctypes.POINTER(SYSTEM_INFO)]
kernel32.AllocateUserPhysicalPages.restype = wt.BOOL
kernel32.AllocateUserPhysicalPages.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
kernel32.FreeUserPhysicalPages.restype = wt.BOOL
kernel32.FreeUserPhysicalPages.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]


def enable_lock_privilege() -> tuple[bool, str]:
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
        return False, ("ERROR_NOT_ALL_ASSIGNED: account lacks the 'Lock pages in memory' "
                        "user right, or granted it without a fresh logon since. "
                        "See docs/RAM_CAP_PROTOCOL.md for the grant procedure.")
    return True, "SeLockMemoryPrivilege enabled on this process's token"


def get_page_size() -> int:
    si = SYSTEM_INFO()
    kernel32.GetSystemInfo(ctypes.byref(si))
    return si.dwPageSize


def available_mb() -> float:
    ms = MEMORYSTATUSEX()
    ms.dwLength = ctypes.sizeof(ms)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullAvailPhys / (1024 * 1024)


def ps(cmd: str, timeout: int = 15) -> str:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                            capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout
    except Exception as e:
        return json.dumps({"error": str(e)})


class AWEBalloon:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.chunks: list[ctypes.Array] = []  # each is a c_size_t array of page frame numbers
        self.n_pages_held = 0
        self.mechanism = None  # "AWE" or "VirtualLock"
        self.virtuallock_regions: list[tuple[int, int]] = []  # (addr, size) for the fallback

    def held_mb(self) -> float:
        if self.mechanism == "AWE":
            return self.n_pages_held * self.page_size / (1024 * 1024)
        else:
            return sum(size for _, size in self.virtuallock_regions) / (1024 * 1024)

    def allocate_awe_chunk(self, n_pages: int) -> int:
        """Returns actual pages allocated (may be less than requested)."""
        arr = (ctypes.c_size_t * n_pages)()
        count = ctypes.c_size_t(n_pages)
        hProcess = kernel32.GetCurrentProcess()
        ok = kernel32.AllocateUserPhysicalPages(hProcess, ctypes.byref(count), arr)
        if not ok:
            raise MemoryError(f"AllocateUserPhysicalPages failed for {n_pages} pages "
                               f"(GetLastError={ctypes.GetLastError()})")
        actual = count.value
        self.chunks.append(arr)
        self.n_pages_held += actual
        return actual

    def allocate_virtuallock_fallback(self, size_bytes: int):
        addr = kernel32.VirtualAlloc(None, ctypes.c_size_t(size_bytes), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not addr:
            raise MemoryError(f"VirtualAlloc fallback failed for {size_bytes} bytes "
                               f"(GetLastError={ctypes.GetLastError()})")
        buf = (ctypes.c_ubyte * size_bytes).from_address(addr)
        for off in range(0, size_bytes, self.page_size):
            buf[off] = 1
        kernel32.SetProcessWorkingSetSizeEx(
            kernel32.GetCurrentProcess(),
            ctypes.c_size_t(size_bytes + 64 * 1024 * 1024),
            ctypes.c_size_t(size_bytes + 256 * 1024 * 1024), 0,
        )
        ok = kernel32.VirtualLock(addr, ctypes.c_size_t(size_bytes))
        if not ok:
            kernel32.VirtualFree(addr, 0, MEM_RELEASE)
            raise MemoryError(f"VirtualLock fallback failed for {size_bytes} bytes "
                               f"(GetLastError={ctypes.GetLastError()})")
        self.virtuallock_regions.append((addr, size_bytes))

    def release_all(self):
        hProcess = kernel32.GetCurrentProcess()
        if self.mechanism == "AWE":
            for arr in self.chunks:
                n = len(arr)
                count = ctypes.c_size_t(n)
                kernel32.FreeUserPhysicalPages(hProcess, ctypes.byref(count), arr)
            self.chunks.clear()
            self.n_pages_held = 0
        else:
            for addr, size in self.virtuallock_regions:
                kernel32.VirtualFree(addr, 0, MEM_RELEASE)
            self.virtuallock_regions.clear()


def get_gpu_counters(server_pid: int | None) -> dict:
    if server_pid is None:
        return {}
    cmd = (
        "$out = @{}; "
        "try { "
        f"  $procSet = (Get-Counter -ListSet 'GPU Process Memory' -ErrorAction Stop).PathsWithInstances "
        f"    | Where-Object {{ $_ -match 'pid_{server_pid}_' }}; "
        "  if ($procSet) { "
        "    $samples = (Get-Counter -Counter $procSet -ErrorAction Stop).CounterSamples; "
        "    foreach ($s in $samples) { "
        "      if ($s.Path -match 'shared usage') { $out.gpu_shared_usage = $s.CookedValue } "
        "      if ($s.Path -match 'dedicated usage') { $out.gpu_dedicated_usage = ($out.gpu_dedicated_usage) + $s.CookedValue } "
        "    } "
        "  } "
        "} catch { $out.gpu_process_error = $_.Exception.Message } "
        "try { "
        "  $adapterSamples = (Get-Counter -Counter '\\GPU Adapter Memory(*)\\Shared Usage' -ErrorAction Stop).CounterSamples; "
        "  $out.gpu_adapter_shared_usage_total = ($adapterSamples | Measure-Object -Property CookedValue -Sum).Sum "
        "} catch { $out.gpu_adapter_error = $_.Exception.Message } "
        "$out | ConvertTo-Json"
    )
    out = ps(cmd, timeout=15)
    try:
        return json.loads(out)
    except Exception:
        return {"gpu_counter_parse_error": out[:300]}


def _ts() -> str:
    """True UTC. Earlier versions wrote local time with a false Z suffix (local was UTC-7 on evo-t2s)."""
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_system_counters() -> dict:
    cmd = (
        "$out = @{}; "
        "try { $out.pages_per_sec = (Get-Counter '\\Memory\\Pages/sec' -ErrorAction Stop).CounterSamples[0].CookedValue } catch { $out.pages_per_sec_error = $_.Exception.Message } "
        "try { $out.pagefile_pct_usage = (Get-Counter '\\Paging File(_Total)\\% Usage' -ErrorAction Stop).CounterSamples[0].CookedValue } catch { $out.pagefile_pct_usage_error = $_.Exception.Message } "
        "try { $out.disk_read_bytes_per_sec = (Get-Counter '\\PhysicalDisk(_Total)\\Disk Read Bytes/sec' -ErrorAction Stop).CounterSamples[0].CookedValue } catch { $out.disk_read_bytes_per_sec_error = $_.Exception.Message } "
        "$out | ConvertTo-Json"
    )
    out = ps(cmd, timeout=15)
    try:
        return json.loads(out)
    except Exception:
        return {"system_counter_parse_error": out[:300]}


def get_server_process_stats(server_pid: int | None) -> dict:
    if server_pid is None:
        return {}
    cmd = f"$p = Get-Process -Id {server_pid} -ErrorAction SilentlyContinue; if ($p) {{ @{{PrivateMemorySize64=$p.PrivateMemorySize64; WorkingSet64=$p.WorkingSet64}} | ConvertTo-Json }} else {{ '{{}}' }}"
    out = ps(cmd, timeout=10)
    try:
        return json.loads(out)
    except Exception:
        return {}


def touch_heartbeat(path: str):
    with open(path, "w") as f:
        f.write(str(time.time()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-available-mb", type=float, required=True)
    ap.add_argument("--tolerance-mb", type=float, default=250.0)
    ap.add_argument("--chunk-mb", type=float, default=2048.0)
    ap.add_argument("--server-pid", type=int, default=None)
    ap.add_argument("--heartbeat-file", type=str, required=True)
    ap.add_argument("--heartbeat-timeout-s", type=float, default=300.0)
    ap.add_argument("--low-available-floor-mb", type=float, default=1024.0)
    ap.add_argument("--low-available-max-consecutive-s", type=float, default=120.0)
    ap.add_argument("--max-runtime-s", type=float, default=10800.0)  # 3h, matches Phase D's own cap
    ap.add_argument("--log-path", type=str, required=True)
    ap.add_argument("--sample-interval-s", type=float, default=5.0)
    args = ap.parse_args()

    enabled, detail = enable_lock_privilege()
    print(f"SeLockMemoryPrivilege: {'ENABLED' if enabled else 'FAILED'} -- {detail}", flush=True)
    if not enabled:
        sys.exit(1)

    page_size = get_page_size()
    balloon = AWEBalloon(page_size)
    touch_heartbeat(args.heartbeat_file)

    with open(args.log_path, "w", encoding="utf-8") as logf:
        try:
            # ---- allocation phase: chunk toward target, then STOP forever ----
            chunk_bytes = int(args.chunk_mb * 1024 * 1024)
            chunk_pages = max(1, chunk_bytes // page_size)
            mechanism_used = None
            while True:
                cur_avail = available_mb()
                remaining_mb = cur_avail - args.target_available_mb
                if remaining_mb <= args.tolerance_mb:
                    break
                this_chunk_mb = min(args.chunk_mb, remaining_mb)
                this_chunk_pages = max(1, int(this_chunk_mb * 1024 * 1024) // page_size)
                try:
                    if mechanism_used in (None, "AWE"):
                        actual = balloon.allocate_awe_chunk(this_chunk_pages)
                        balloon.mechanism = "AWE"
                        mechanism_used = "AWE"
                        print(f"AWE chunk: requested {this_chunk_pages} pages, got {actual}. "
                              f"held_mb={balloon.held_mb():.1f} avail_mb={cur_avail:.1f}", flush=True)
                        if actual == 0:
                            # Returned success but made zero progress -- would
                            # spin forever otherwise (remaining_mb never
                            # shrinks). Treat as a hard failure of this
                            # mechanism, same as an outright API failure.
                            raise MemoryError("AllocateUserPhysicalPages returned TRUE but allocated 0 pages")
                    else:
                        raise MemoryError("mechanism already fell back to VirtualLock")
                except MemoryError as e:
                    if mechanism_used == "AWE":
                        print(f"AWE FAILED mid-run ({e}); this run does not mix mechanisms -- STOPPING allocation, "
                              f"held so far: {balloon.held_mb():.1f} MB via AWE.", flush=True)
                        break
                    print(f"AWE failed ({e}), falling back to VirtualLock for this and all further chunks.", flush=True)
                    balloon.mechanism = "VirtualLock"
                    mechanism_used = "VirtualLock"
                    balloon.allocate_virtuallock_fallback(int(this_chunk_mb * 1024 * 1024))
                    print(f"VirtualLock chunk: {this_chunk_mb:.1f} MB. held_mb={balloon.held_mb():.1f} "
                          f"avail_mb={cur_avail:.1f}", flush=True)
                touch_heartbeat(args.heartbeat_file)

            print(f"TARGET REACHED (or best effort). mechanism={balloon.mechanism} "
                  f"held_mb={balloon.held_mb():.1f} available_mb={available_mb():.1f} "
                  f"target_available_mb={args.target_available_mb}", flush=True)

            logf.write(f"# mechanism={balloon.mechanism} target_available_mb={args.target_available_mb} "
                       f"held_mb_final={balloon.held_mb():.1f} page_size={page_size}\n")
            logf.write("ts_iso,elapsed_s,held_mb,available_mb,pages_per_sec,pagefile_pct_usage,"
                       "server_private_mb,server_workingset_mb,gpu_shared_usage,gpu_dedicated_usage,"
                       "gpu_adapter_shared_usage_total,note,disk_read_bytes_per_sec,counter_read_s\n")
            logf.flush()

            # ---- hold + sample phase ----
            t_start = time.monotonic()
            low_avail_since = None
            while True:
                now = time.monotonic()
                elapsed = now - t_start
                if elapsed > args.max_runtime_s:
                    print(f"DEAD-MAN SWITCH: max-runtime-s exceeded.", flush=True)
                    break
                try:
                    hb_age = time.time() - os.path.getmtime(args.heartbeat_file)
                except OSError:
                    hb_age = float("inf")
                if hb_age > args.heartbeat_timeout_s:
                    print(f"SAFETY VALVE: heartbeat stale ({hb_age:.1f}s).", flush=True)
                    logf.write(f"{_ts()},{elapsed:.1f},{balloon.held_mb():.1f},,,,,,,,safety_valve_heartbeat\n")
                    break

                cur_avail = available_mb()
                if cur_avail < args.low_available_floor_mb:
                    if low_avail_since is None:
                        low_avail_since = now
                    elif now - low_avail_since > args.low_available_max_consecutive_s:
                        print(f"SAFETY VALVE: available < {args.low_available_floor_mb} MB for "
                              f">{args.low_available_max_consecutive_s}s.", flush=True)
                        logf.write(f"{_ts()},{elapsed:.1f},{balloon.held_mb():.1f},{cur_avail:.1f},,,,,,,safety_valve_low_available\n")
                        break
                else:
                    low_avail_since = None

                _t_read = time.monotonic()
                sys_counters = get_system_counters()
                srv_stats = get_server_process_stats(args.server_pid)
                gpu_counters = get_gpu_counters(args.server_pid)

                row = (
                    f"{_ts()},{elapsed:.1f},{balloon.held_mb():.1f},"
                    f"{cur_avail:.1f},{sys_counters.get('pages_per_sec', '')},"
                    f"{sys_counters.get('pagefile_pct_usage', '')},"
                    f"{(srv_stats.get('PrivateMemorySize64', 0) or 0) / (1024*1024):.1f},"
                    f"{(srv_stats.get('WorkingSet64', 0) or 0) / (1024*1024):.1f},"
                    f"{gpu_counters.get('gpu_shared_usage', '')},"
                    f"{gpu_counters.get('gpu_dedicated_usage', '')},"
                    f"{gpu_counters.get('gpu_adapter_shared_usage_total', '')},,"
                    f"{sys_counters.get('disk_read_bytes_per_sec', '')},{time.monotonic() - _t_read:.1f}\n"
                )
                logf.write(row)
                logf.flush()
                print(row.strip(), flush=True)

                time.sleep(max(0.0, args.sample_interval_s - (time.monotonic() - now)))

        except KeyboardInterrupt:
            print("KeyboardInterrupt.", flush=True)
        finally:
            held_before = balloon.held_mb()
            balloon.release_all()
            print(f"Released {held_before:.1f} MB (mechanism={balloon.mechanism}). Exiting.", flush=True)


if __name__ == "__main__":
    main()
