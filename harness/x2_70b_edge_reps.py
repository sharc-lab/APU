"""llama-3.3-70b load at the memory edge, 10 fresh-process reps with pre-rep memory/GPU/page-cache state
(evo-x2, P4, 2026-10-06). Follow-up to docs/FINDINGS.md's "CORRECTION (2026-10-06): MX2 crash boundary
re-tested" section: the fresh-process re-test of llama-3.3-70b at n_ctx=221,696 gave three different outcomes
in three reps (0xC0000409 at 347.2s, 0xC0000005 at 2.5s, and one run that neither crashed nor started within
600s, its log stalled directly after the `model buffer size` lines). That section names the open question:
is the boundary genuinely non-deterministic, or is elapsed time and outcome dominated by disk / memory state
(cold page cache, standby list, GPU memory not yet released by a previous process)? This job is built to
separate those.

Design:
  * Same server command line as recheck_70b_crash.py / mx2_validation.run_crash_repro (same -ctk/-ctv/-fa/-np/
    -t/--no-context-shift/-ngl flags, n_ctx=221696), 600s start budget per rep.
  * Each rep runs in a FRESH child Python process (this file, --single-rep), which launches a FRESH
    llama-server process. The parent orchestrator never launches a server itself, so no driver/process state
    accumulates in a long-lived process (the confound the 2026-10-06 correction found in mx2_validation.py).
  * Two arms, interleaved ABAB (rep 0 warm, rep 1 cold, rep 2 warm, ...), never blocked:
      - warm_read: one full sequential buffered read of the GGUF right before launch (page cache warm).
      - cold_purge: purge the Windows standby list (NtSetSystemInformation, MemoryPurgeStandbyList) right
        before launch so the GGUF is read from disk. If the purge is not permitted in the job's security
        context, the row records purge.ok=false and the arm is really "no warm-up read, cache as left by the
        previous rep" -- the residency probe below records what the cache actually held either way, so the
        analysis never has to trust the arm label alone.
  * Before every rep: wait (bounded) for GPU adapter dedicated usage to settle near idle, and refuse to run
    (STOP) if a llama-server we did not start is alive.

Logged per rep (all on one "x2_70b_edge_rep" row):
  * pre_arm and pre_launch snapshots: Available / Free+Zero / Standby (normal, reserve, core) / Modified page
    list bytes, committed bytes and commit limit; GPU Adapter Memory dedicated/shared/total-committed per
    adapter; GPU Process Memory dedicated/shared enumerated per process (every pid holding GPU memory, with
    its process name); any llama-server/ollama processes alive; a page-cache residency probe of the GGUF
    (timed 64 KiB buffered reads at 256 spread offsets; cached reads are tens of microseconds, disk reads are
    hundreds -- raw latency quantiles are recorded so the threshold can be re-applied later).
  * pre_launch only (heavier, once per rep): vulkaninfo memory heaps (size/budget/usage, i.e. the device-local
    pool and the shared pool sizes the driver advertises), and `llama-server --list-devices` total/free.
  * arm_prep: warm read bytes/seconds/GiB/s with standby bytes before/after, or purge result
    (privilege_ok, ntstatus) with standby bytes before/after.
  * the launch itself: outcome (STARTED / CRASH / STALL / EXIT_CLEAN_NO_START / LAUNCH_ERROR), exit code
    (decimal, hex, NTSTATUS name), the log failure signatures matched, the last load stage reached and the
    log's own timestamp of its last line (stall location), seconds the log was idle before the budget ran
    out, the server's own "(N MiB, M MiB free)" device line, and a trajectory sampled every 5s (server
    read_bytes / rss / private bytes / page faults, system available bytes, system disk read bytes) with
    GPU per-process counters every 30s.
  * post: a snapshot after the server is gone.

Usage: py -3.12 harness/x2_70b_edge_reps.py --out results/x2_70b_edge_reps.jsonl [--reps 10]
Queue job: calls t2s_queue.advance() exactly once, from the parent's finally block.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
if (_here / "analysis").is_dir():
    REPO = _here  # flat deployment on evo-x2
else:
    REPO = _here.parents[0]  # repo layout
sys.path.insert(0, str(_here))
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))

LLAMA_SERVER_EXE = r"C:\apu\bin\llama-b10970\llama-server.exe"
GGUF_PATH = r"C:\apu\models\Llama-3.3-70B-Instruct-Q4_K_M.gguf"
MODEL_ID = "llama-3.3-70b"
N_CTX = 221696  # the MX2 crash boundary, docs/FINDINGS.md 2026-10-06 correction
PORT = 58699  # distinct from mx2_validation (58399) and the by-hand recheck (58599)
START_BUDGET_S = 600
DEFAULT_REPS = 10
ARMS = ("warm_read", "cold_purge")

WARM_READ_CHUNK = 8 * 2 ** 20
WARM_READ_MAX_S = 900
RESIDENCY_SAMPLES = 256
RESIDENCY_CHUNK = 64 * 2 ** 10
RESIDENCY_CACHED_THRESHOLD_S = 100e-6
SETTLE_DEDICATED_MAX_BYTES = 4 * 2 ** 30  # GPU adapter dedicated usage at or below this counts as idle
SETTLE_MAX_WAIT_S = 180
TRAJ_EVERY_S = 5
TRAJ_GPU_EVERY_S = 30
CHILD_TIMEOUT_S = START_BUDGET_S + WARM_READ_MAX_S + SETTLE_MAX_WAIT_S + 600


def server_cmd(log_path):
    return [LLAMA_SERVER_EXE, "-m", GGUF_PATH, "--port", str(PORT), "-c", str(N_CTX), "-ctk", "f16", "-ctv",
            "f16", "-fa", "on", "-np", "1", "-t", "4", "--no-context-shift", "-ngl", "99",
            "--log-file", str(log_path), "--log-verbosity", "4"]


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def emit(out_path, row):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()


def arm_for_rep(rep):
    """ABAB interleave: even reps warm_read, odd reps cold_purge."""
    return ARMS[rep % 2]


# ------------------------------------------------------------------------------------------- pure parsing
NTSTATUS_NAMES = {
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",  # also __fastfail / abort paths in the MSVC CRT
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC0000017: "STATUS_NO_MEMORY",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC0000142: "STATUS_DLL_INIT_FAILED",
    0x40010004: "DBG_TERMINATE_PROCESS",
    3: "CRT_ABORT",
    1: "EXIT_FAILURE",
    0: "EXIT_SUCCESS",
}


def exit_code_info(code):
    if code is None:
        return {"exit_code": None, "exit_code_hex": None, "exit_code_name": None}
    u = code & 0xFFFFFFFF
    return {"exit_code": code, "exit_code_hex": f"0x{u:08X}", "exit_code_name": NTSTATUS_NAMES.get(u, "UNKNOWN")}


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# llama.cpp common/log.cpp prefix: minutes.seconds.millis.micros, e.g. "0.03.725.317"
LOG_TS_RE = re.compile(r"^\s*(\d+)\.(\d{2})\.(\d{3})\.(\d{3})\s")

# Ordered load stages; the stage reached is the highest index whose pattern appears anywhere in the log.
LOAD_STAGES = [
    ("model_meta", re.compile(r"llama_model_loader: loaded meta data")),
    ("print_info", re.compile(r"print_info:")),
    ("load_tensors_begin", re.compile(r"load_tensors: loading model tensors")),
    ("model_buffers_sized", re.compile(r"model buffer size\s*=")),
    ("context_init", re.compile(r"llama_context: constructing|llama_context: n_ctx\s*=")),
    ("kv_cache", re.compile(r"KV buffer size\s*=|llama_kv_cache")),
    ("compute_buffers", re.compile(r"compute buffer size\s*=")),
    ("warmup", re.compile(r"warming up the model")),
    ("listening", re.compile(r"server is listening|main: model loaded")),
]

FAIL_SIGNATURES = [
    ("vk_buffer_alloc_fail", re.compile(r"failed to allocate Vulkan\d*\s*\w* ?buffer", re.I)),
    ("compute_buffer_alloc_fail", re.compile(r"failed to allocate compute (pp )?buffers", re.I)),
    ("context_init_fail", re.compile(r"failed to initialize the context", re.I)),
    ("model_load_fail", re.compile(r"failed to load model|error loading model", re.I)),
    ("out_of_device_memory", re.compile(r"ErrorOutOfDeviceMemory|out of device memory", re.I)),
    ("out_of_host_memory", re.compile(r"ErrorOutOfHostMemory|out of host memory", re.I)),
    ("device_lost", re.compile(r"ErrorDeviceLost|device lost", re.I)),
    ("ggml_assert", re.compile(r"GGML_ASSERT|ggml_abort")),
    ("cpp_exception", re.compile(r"terminate called|what\(\):|std::exception|vk::\w+Error")),
]


def parse_log(text):
    """Load stage reached, last log timestamp (seconds since server start, from the log's own prefix),
    matched failure signatures, and the last few non-empty lines. Pure: takes the log text."""
    text = ANSI_RE.sub("", text or "")
    lines = [l for l in text.splitlines() if l.strip()]
    stage_idx = -1
    for i, (_name, pat) in enumerate(LOAD_STAGES):
        if pat.search(text):
            stage_idx = max(stage_idx, i)
    last_ts = None
    for l in reversed(lines):
        m = LOG_TS_RE.match(l)
        if m:
            mins, secs, ms, us = (int(g) for g in m.groups())
            last_ts = mins * 60 + secs + ms / 1e3 + us / 1e6
            break
    sigs = [name for name, pat in FAIL_SIGNATURES if pat.search(text)]
    sig_lines = [l.strip()[:300] for l in lines if any(p.search(l) for _n, p in FAIL_SIGNATURES)][-6:]
    return {"last_stage": LOAD_STAGES[stage_idx][0] if stage_idx >= 0 else None,
            "last_stage_index": stage_idx, "last_log_ts_s": last_ts, "fail_signatures": sigs,
            "fail_lines": sig_lines, "n_log_lines": len(lines), "log_tail": [l.strip()[:300] for l in lines[-8:]]}


def classify_outcome(started, exit_code, timed_out, parsed_log, elapsed_s, launch_error=None):
    """One label per rep. STARTED: /health 200 within budget. CRASH: exited on its own with a non-zero code.
    EXIT_CLEAN_NO_START: exited 0 without ever serving (unexpected, kept distinct). STALL: neither exited nor
    started within the budget; stall_stage is the last load stage the log reached and log_idle_s how long the
    log had been silent when the budget ran out. LAUNCH_ERROR: the process could not be started at all."""
    stage = (parsed_log or {}).get("last_stage")
    last_ts = (parsed_log or {}).get("last_log_ts_s")
    out = {"outcome": None, "stall_stage": None, "log_idle_s": None, "crash_stage": None}
    if launch_error:
        out["outcome"] = "LAUNCH_ERROR"
    elif started:
        out["outcome"] = "STARTED"
    elif exit_code is not None:
        out["outcome"] = "EXIT_CLEAN_NO_START" if exit_code == 0 else "CRASH"
        out["crash_stage"] = stage
    elif timed_out:
        out["outcome"] = "STALL"
        out["stall_stage"] = stage
        if last_ts is not None and elapsed_s is not None:
            out["log_idle_s"] = max(0.0, elapsed_s - last_ts)
    else:
        out["outcome"] = "UNKNOWN"
    return out


def parse_counter_samples(samples):
    """Get-Counter CounterSamples (as JSON: [{Path, InstanceName, CookedValue}]) -> {memory, gpu_adapters,
    gpu_processes}. memory: {counter_name: bytes}. gpu_adapters: {instance: {dedicated_usage, shared_usage,
    total_committed}}. gpu_processes: [{pid, luid, dedicated_bytes, shared_bytes}] for every process instance
    with any non-zero GPU memory, sorted by total descending. Pure."""
    if isinstance(samples, dict):
        samples = [samples]
    memory, adapters, procs = {}, {}, {}
    for s in samples or []:
        path = (s.get("Path") or "").lower()
        val = s.get("CookedValue")
        inst = s.get("InstanceName") or ""
        counter = path.rsplit("\\", 1)[-1]
        key = counter.replace(" & ", "_").replace(" ", "_")
        if "\\memory\\" in path:
            memory[key] = val
        elif "gpu adapter memory" in path:
            adapters.setdefault(inst, {})[key] = val
        elif "gpu process memory" in path:
            m = re.match(r"pid_(\d+)_(luid_\w+?)_phys", inst)
            if not m:
                continue
            pid, luid = int(m.group(1)), m.group(2)
            d = procs.setdefault((pid, luid), {"pid": pid, "luid": luid, "dedicated_bytes": 0.0,
                                               "shared_bytes": 0.0})
            if "dedicated" in counter:
                d["dedicated_bytes"] = val or 0.0
            elif "shared" in counter:
                d["shared_bytes"] = val or 0.0
    plist = [p for p in procs.values() if (p["dedicated_bytes"] or 0) + (p["shared_bytes"] or 0) > 0]
    plist.sort(key=lambda p: -((p["dedicated_bytes"] or 0) + (p["shared_bytes"] or 0)))
    return {"memory": memory, "gpu_adapters": adapters, "gpu_processes": plist}


def adapter_dedicated_total(parsed):
    vals = [a.get("dedicated_usage") for a in (parsed or {}).get("gpu_adapters", {}).values()]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def summarize_residency(latencies_s, threshold_s=RESIDENCY_CACHED_THRESHOLD_S):
    """Sampled-read latencies -> fraction under the cached threshold plus quantiles (seconds). Pure."""
    lat = sorted(x for x in latencies_s if x is not None)
    if not lat:
        return {"n": 0, "cached_fraction_est": None, "p10_s": None, "median_s": None, "p90_s": None,
                "threshold_s": threshold_s}
    def q(p):
        return lat[min(len(lat) - 1, int(p * (len(lat) - 1) + 0.5))]
    return {"n": len(lat), "cached_fraction_est": sum(1 for x in lat if x < threshold_s) / len(lat),
            "p10_s": q(0.10), "median_s": statistics.median(lat), "p90_s": q(0.90), "max_s": lat[-1],
            "threshold_s": threshold_s}


def summarize(rows):
    """Outcome counts per arm, from rep rows. Pure."""
    by_arm = {}
    for r in rows:
        a = by_arm.setdefault(r.get("arm"), {})
        a[r.get("outcome")] = a.get(r.get("outcome"), 0) + 1
    return by_arm


# -------------------------------------------------------------------------------------- live measurement
def _ps(cmd, timeout=60):
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout.strip()
    except Exception:
        return ""


MEMORY_COUNTERS = [r"\Memory\Available Bytes", r"\Memory\Free & Zero Page List Bytes",
                   r"\Memory\Standby Cache Normal Priority Bytes", r"\Memory\Standby Cache Reserve Bytes",
                   r"\Memory\Standby Cache Core Bytes", r"\Memory\Modified Page List Bytes",
                   r"\Memory\Committed Bytes", r"\Memory\Commit Limit"]
GPU_COUNTERS = [r"\GPU Adapter Memory(*)\Dedicated Usage", r"\GPU Adapter Memory(*)\Shared Usage",
                r"\GPU Adapter Memory(*)\Total Committed", r"\GPU Process Memory(*)\Dedicated Usage",
                r"\GPU Process Memory(*)\Shared Usage"]


def read_counters(include_gpu=True, ps_fn=_ps):
    names = MEMORY_COUNTERS + (GPU_COUNTERS if include_gpu else [])
    lst = ",".join("'" + n.replace("'", "''") + "'" for n in names)
    cmd = (f"(Get-Counter -Counter {lst} -ErrorAction SilentlyContinue).CounterSamples | "
           f"Select Path,InstanceName,CookedValue | ConvertTo-Json -Compress")
    out = ps_fn(cmd, 90)
    try:
        samples = json.loads(out) if out else []
    except Exception:
        samples = []
    parsed = parse_counter_samples(samples)
    parsed["counter_read_ok"] = bool(samples)
    return parsed


def standby_total(parsed):
    m = (parsed or {}).get("memory", {})
    vals = [m.get(k) for k in ("standby_cache_normal_priority_bytes", "standby_cache_reserve_bytes",
                               "standby_cache_core_bytes")]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def name_gpu_processes(parsed):
    try:
        import psutil
    except Exception:
        return parsed
    for p in parsed.get("gpu_processes", []):
        try:
            p["name"] = psutil.Process(p["pid"]).name()
        except Exception:
            p["name"] = None
    return parsed


def server_like_processes():
    """llama-server / ollama processes alive right now, with whether each is ours (our --port)."""
    try:
        import psutil
    except Exception:
        return None
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        n = (p.info.get("name") or "").lower()
        if "llama-server" in n or n.startswith("ollama"):
            cl = " ".join(p.info.get("cmdline") or [])
            out.append({"pid": p.info["pid"], "name": p.info.get("name"),
                        "ours": f"--port {PORT}" in cl, "cmdline": cl[:300]})
    return out


def residency_probe(path=GGUF_PATH, n=RESIDENCY_SAMPLES, chunk=RESIDENCY_CHUNK, seed=0):
    """Timed buffered reads at n spread offsets (one per equal slice, jittered within it). Touches n*chunk
    bytes (16 MiB by default, ~0.04% of the file), so it barely warms what it measures."""
    try:
        size = os.path.getsize(path)
        rng = random.Random(seed)
        slice_ = size // n
        lats = []
        with open(path, "rb", buffering=0) as f:
            for i in range(n):
                off = i * slice_ + rng.randrange(0, max(1, slice_ - chunk))
                t0 = time.perf_counter()
                f.seek(off)
                f.read(chunk)
                lats.append(time.perf_counter() - t0)
        return summarize_residency(lats)
    except Exception as e:
        return {"error": repr(e)[:300]}


def snapshot(label, rep, heavy=False):
    t0 = time.monotonic()
    snap = {"label": label, "ts_utc": utc_iso()}
    snap["counters"] = name_gpu_processes(read_counters())
    snap["standby_total_bytes"] = standby_total(snap["counters"])
    snap["adapter_dedicated_total_bytes"] = adapter_dedicated_total(snap["counters"])
    snap["server_like_processes"] = server_like_processes()
    snap["gguf_residency"] = residency_probe(seed=rep * 7 + (1 if label == "pre_launch" else 0))
    if heavy:
        snap["vulkan_heaps"] = read_vulkan_heaps()
        snap["list_devices"] = read_list_devices()
    snap["snapshot_s"] = time.monotonic() - t0
    return snap


def read_vulkan_heaps():
    try:
        import t2s_night2 as night2
        txt = night2.mx2_read_vulkaninfo()
        heaps = night2.mx2_parse_heaps(txt) if txt else []
        return {"heaps": heaps, "lines": night2.mx2_lines_from_heaps(heaps)}
    except Exception as e:
        return {"error": repr(e)[:300]}


def read_list_devices():
    try:
        p = subprocess.run([LLAMA_SERVER_EXE, "--list-devices"], capture_output=True, text=True,
                           errors="replace", timeout=180)
        txt = (p.stdout or "") + "\n" + (p.stderr or "")
        m = re.search(r"(\S+):\s*.*?(\d+)\s*MiB,\s*(\d+)\s*MiB free", txt)
        return {"device": m.group(1) if m else None, "total_mib": float(m.group(2)) if m else None,
                "free_mib": float(m.group(3)) if m else None, "raw": txt.strip()[:600]}
    except Exception as e:
        return {"error": repr(e)[:300]}


def warm_read(path=GGUF_PATH, max_s=WARM_READ_MAX_S):
    sb0 = standby_total(read_counters(include_gpu=False))
    buf = bytearray(WARM_READ_CHUNK)
    mv = memoryview(buf)
    total = 0
    t0 = time.monotonic()
    complete = True
    try:
        with open(path, "rb", buffering=0) as f:
            while True:
                n = f.readinto(mv)
                if not n:
                    break
                total += n
                if time.monotonic() - t0 > max_s:
                    complete = False
                    break
    except Exception as e:
        return {"error": repr(e)[:300], "bytes": total}
    dt = time.monotonic() - t0
    sb1 = standby_total(read_counters(include_gpu=False))
    return {"bytes": total, "seconds": dt, "gib_per_s": (total / 2 ** 30) / dt if dt > 0 else None,
            "complete": complete, "standby_before_bytes": sb0, "standby_after_bytes": sb1}


def purge_standby_list(dry_run=False):
    """Empties the Windows standby list (NtSetSystemInformation(SystemMemoryListInformation=80,
    MemoryPurgeStandbyList=4)). Needs SeProfileSingleProcessPrivilege (held by SYSTEM / elevated admin).
    Clean pages only -- nothing is lost, the cache just has to be refilled from disk. Never raises.
    dry_run=True stops after enabling the privilege (checks the job's security context, purges nothing)."""
    import ctypes
    from ctypes import wintypes
    res = {"privilege_ok": False, "ok": False, "ntstatus": None, "error": None}
    sb0 = standby_total(read_counters(include_gpu=False))
    try:
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll")

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

        class LUID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]

        class TOKEN_PRIVILEGES(ctypes.Structure):
            _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Privileges", LUID_AND_ATTRIBUTES * 1)]

        TOKEN_ADJUST_PRIVILEGES, TOKEN_QUERY, SE_PRIVILEGE_ENABLED = 0x20, 0x8, 0x2
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        advapi32.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.POINTER(LUID)]
        advapi32.AdjustTokenPrivileges.argtypes = [wintypes.HANDLE, wintypes.BOOL, ctypes.POINTER(TOKEN_PRIVILEGES),
                                                   wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
        ntdll.NtSetSystemInformation.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        ntdll.NtSetSystemInformation.restype = ctypes.c_long
        h = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                                         ctypes.byref(h)):
            res["error"] = f"OpenProcessToken failed: {ctypes.get_last_error()}"
            return res
        luid = LUID()
        if not advapi32.LookupPrivilegeValueW(None, "SeProfileSingleProcessPrivilege", ctypes.byref(luid)):
            res["error"] = f"LookupPrivilegeValueW failed: {ctypes.get_last_error()}"
            return res
        tp = TOKEN_PRIVILEGES(1, (LUID_AND_ATTRIBUTES * 1)(LUID_AND_ATTRIBUTES(luid, SE_PRIVILEGE_ENABLED)))
        ok = advapi32.AdjustTokenPrivileges(h, False, ctypes.byref(tp), 0, None, None)
        err = ctypes.get_last_error()
        kernel32.CloseHandle(h)
        if not ok or err == 1300:  # ERROR_NOT_ALL_ASSIGNED
            res["error"] = f"AdjustTokenPrivileges: privilege not held (ok={ok}, err={err})"
            return res
        res["privilege_ok"] = True
        if dry_run:
            res["dry_run"] = True
            return res
        cmd = ctypes.c_int(4)
        status = ntdll.NtSetSystemInformation(80, ctypes.byref(cmd), ctypes.sizeof(cmd)) & 0xFFFFFFFF
        res["ntstatus"] = f"0x{status:08X}"
        res["ok"] = status == 0
    except Exception as e:
        res["error"] = repr(e)[:300]
    finally:
        res["standby_before_bytes"] = sb0
        res["standby_after_bytes"] = standby_total(read_counters(include_gpu=False))
    return res


def wait_for_gpu_settle(max_wait_s=SETTLE_MAX_WAIT_S):
    t0 = time.monotonic()
    readings = []
    while True:
        ded = adapter_dedicated_total(read_counters())
        readings.append(ded)
        if ded is not None and ded <= SETTLE_DEDICATED_MAX_BYTES:
            return {"settled": True, "waited_s": time.monotonic() - t0, "last_dedicated_bytes": ded}
        if time.monotonic() - t0 > max_wait_s:
            return {"settled": False, "waited_s": time.monotonic() - t0, "last_dedicated_bytes": ded,
                    "readings_tail": readings[-5:]}
        time.sleep(10)


def _traj_sample(proc_ps, disk0, with_gpu, pid):
    import psutil
    s = {"t_s": None}
    try:
        mi = proc_ps.memory_info()
        io = proc_ps.io_counters()
        s.update({"rss_bytes": mi.rss, "private_bytes": getattr(mi, "private", None),
                  "page_faults": getattr(mi, "num_page_faults", None), "read_bytes": io.read_bytes})
    except Exception:
        s["proc_gone"] = True
    try:
        s["sys_available_bytes"] = psutil.virtual_memory().available
        d = psutil.disk_io_counters()
        s["sys_disk_read_bytes_delta"] = d.read_bytes - disk0 if d is not None and disk0 is not None else None
    except Exception:
        pass
    if with_gpu:
        c = read_counters(include_gpu=True)
        mine = [p for p in c.get("gpu_processes", []) if p["pid"] == pid]
        s["gpu_dedicated_bytes"] = sum(p["dedicated_bytes"] or 0 for p in mine) if mine else None
        s["gpu_shared_bytes"] = sum(p["shared_bytes"] or 0 for p in mine) if mine else None
        s["adapter_dedicated_total_bytes"] = adapter_dedicated_total(c)
    return s


def launch_and_watch(rep, log_dir):
    import psutil
    log_path = Path(log_dir) / f"x2_70b_edge_rep{rep}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    cmd = server_cmd(log_path)
    res = {"log_path": str(log_path), "cmd": cmd, "start_budget_s": START_BUDGET_S}
    try:
        d0 = psutil.disk_io_counters()
        disk0 = d0.read_bytes if d0 is not None else None
    except Exception:
        disk0 = None
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        res.update(classify_outcome(False, None, False, None, 0.0, launch_error=repr(e)))
        res["launch_error"] = repr(e)[:300]
        return res
    res["server_pid"] = proc.pid
    try:
        pps = psutil.Process(proc.pid)
    except Exception:
        pps = None
    import urllib.request
    started = False
    traj = []
    next_traj, next_gpu = 0.0, 0.0
    while True:
        el = time.monotonic() - t0
        if proc.poll() is not None or el >= START_BUDGET_S:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2.0) as r:
                if r.status == 200:
                    started = True
                    break
        except Exception:
            pass
        if pps is not None and el >= next_traj:
            with_gpu = el >= next_gpu
            s = _traj_sample(pps, disk0, with_gpu, proc.pid)
            s["t_s"] = round(time.monotonic() - t0, 2)
            traj.append(s)
            next_traj = el + TRAJ_EVERY_S
            if with_gpu:
                next_gpu = el + TRAJ_GPU_EVERY_S
        time.sleep(1.0)
    elapsed = time.monotonic() - t0
    exited = proc.poll() is not None
    exit_code = proc.returncode if exited and not started else None
    timed_out = (not started) and (not exited)
    if not exited:
        proc.kill()  # TerminateProcess; its exit code (1) is never reported as a crash code
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            res["kill_wait_timeout"] = True
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    parsed = parse_log(text)
    res.update({"started": started, "proc_exited_on_its_own": exited and not started, "timed_out": timed_out,
                "elapsed_s": elapsed, **exit_code_info(exit_code), **parsed,
                **classify_outcome(started, exit_code, timed_out, parsed, elapsed), "trajectory": traj})
    try:
        import t2s_lab as L
        sl = L.parse_server_log(log_path)
        res["server_device_line"] = {k: sl.get(k) for k in ("device_line", "device_total_mib", "device_free_mib",
                                                             "model_buffer_mib", "kv_buffer_mib",
                                                             "compute_buffer_mib")}
    except Exception as e:
        res["server_device_line"] = {"error": repr(e)[:200]}
    return res


def kill_our_servers():
    killed = []
    for p in server_like_processes() or []:
        if p["ours"] and "llama-server" in (p["name"] or "").lower():
            try:
                import psutil
                psutil.Process(p["pid"]).kill()
                killed.append(p["pid"])
            except Exception:
                pass
    return killed


def run_single_rep(rep, arm, out_path, log=print):
    """Child-process body: one rep, one fresh llama-server, one row."""
    row = {"record": "x2_70b_edge_rep", "rep": rep, "arm": arm, "model_id": MODEL_ID, "n_ctx": N_CTX,
           "gguf_path": GGUF_PATH, "ts_utc": utc_iso(), "child_pid": os.getpid()}
    try:
        row["gguf_bytes"] = os.path.getsize(GGUF_PATH)
    except Exception:
        row["gguf_bytes"] = None
    others = [p for p in (server_like_processes() or []) if "llama-server" in (p["name"] or "").lower()
              and not p["ours"]]
    if others:
        row.update({"outcome": "NOT_RUN", "error": f"STOP: a llama-server process we did not start is running: "
                                                    f"{others}"[:400]})
        emit(out_path, row)
        return row
    try:
        row["killed_stale_own_servers"] = kill_our_servers()
        row["gpu_settle"] = wait_for_gpu_settle()
        row["pre_arm"] = snapshot("pre_arm", rep)
        row["arm_prep"] = warm_read() if arm == "warm_read" else purge_standby_list()
        row["pre_launch"] = snapshot("pre_launch", rep, heavy=True)
        launch = launch_and_watch(rep, REPO / "results")
        row["launch"] = launch
        for k in ("outcome", "exit_code", "exit_code_hex", "exit_code_name", "elapsed_s", "stall_stage",
                  "crash_stage", "log_idle_s", "last_stage", "fail_signatures"):
            row[k] = launch.get(k)
        row["post"] = snapshot("post", rep)
    except Exception as e:
        row["outcome"] = row.get("outcome") or "CHILD_ERROR"
        row["error"] = f"driver exception: {e!r}"[:400]
        kill_our_servers()
    emit(out_path, row)
    if row["outcome"] == "CHILD_ERROR":
        log(f"rep {rep} [{arm}]: CHILD_ERROR {row['error']}")
        return row
    pre = row["pre_launch"].get("gguf_residency", {})
    log(f"rep {rep} [{arm}]: outcome={row['outcome']} exit={row['exit_code_hex']} ({row['exit_code_name']}) "
        f"elapsed_s={row['elapsed_s'] or 0:.1f} stage={row['last_stage']} sigs={row['fail_signatures']} "
        f"cached_frac_pre_launch={pre.get('cached_fraction_est')}")
    return row


# --------------------------------------------------------------------------------------------- parent
def read_rows(out_path):
    rows = []
    if not Path(out_path).exists():
        return rows
    for line in Path(out_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def done_reps(out_path):
    return {r["rep"] for r in read_rows(out_path)
            if r.get("record") == "x2_70b_edge_rep" and r.get("outcome") not in (None, "CHILD_FAILED")}


def run(out_path, reps, deadline_h=3.0, log=print, spawn=None):
    """Parent orchestrator. spawn(rep, arm) -> returncode runs one rep in a fresh child Python process;
    injectable for tests."""
    if spawn is None:
        def spawn(rep, arm):
            cmd = [sys.executable, str(Path(__file__).resolve()), "--single-rep", str(rep), "--arm", arm,
                   "--out", str(out_path)]
            try:
                return subprocess.run(cmd, timeout=CHILD_TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                return "timeout"
    done = done_reps(out_path)
    t_start = time.monotonic()
    emit(out_path, {"record": "run_start", "reps": reps, "n_ctx": N_CTX, "arms": list(ARMS),
                    "start_budget_s": START_BUDGET_S, "ts_utc": utc_iso()})
    for rep in range(reps):
        if rep in done:
            continue
        if time.monotonic() - t_start > deadline_h * 3600:
            log(f"deadline reached ({deadline_h}h), stopping")
            break
        arm = arm_for_rep(rep)
        emit(out_path, {"record": "heartbeat", "rep": rep, "arm": arm, "ts_utc": utc_iso()})
        rc = spawn(rep, arm)
        killed = kill_our_servers()
        rows = [r for r in read_rows(out_path) if r.get("record") == "x2_70b_edge_rep" and r.get("rep") == rep]
        if not rows:
            emit(out_path, {"record": "x2_70b_edge_rep", "rep": rep, "arm": arm, "outcome": "CHILD_FAILED",
                            "child_returncode": rc, "killed_orphan_servers": killed, "ts_utc": utc_iso()})
        elif str(rows[-1].get("error", "")).startswith("STOP:"):
            raise RuntimeError(rows[-1]["error"])
    reps_rows = [r for r in read_rows(out_path) if r.get("record") == "x2_70b_edge_rep"]
    summary = summarize(reps_rows)
    emit(out_path, {"record": "summary", "outcomes_by_arm": summary, "ts_utc": utc_iso()})
    log(f"outcomes by arm: {summary}")
    emit(out_path, {"record": "run_end", "ts_utc": utc_iso()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--deadline-h", type=float, default=3.0)
    ap.add_argument("--single-rep", type=int, default=None, help="internal: run one rep in this process")
    ap.add_argument("--arm", choices=ARMS, default=None)
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.single_rep is not None:
        run_single_rep(args.single_rep, args.arm or arm_for_rep(args.single_rep), out_path)
        return  # the child never touches the queue

    print(f"logging to {out_path}")
    note = "completed"
    try:
        run(out_path, args.reps, deadline_h=args.deadline_h)
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        print(note)
        traceback.print_exc()
    finally:
        try:
            import t2s_queue as tq
            tq.advance(note)
        except Exception as e:
            print(f"queue advance failed: {e!r}")


if __name__ == "__main__":
    main()
