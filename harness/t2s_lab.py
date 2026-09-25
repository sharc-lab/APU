"""Infrastructure for the evo-t2s overnight run (sections 0 and A to D live in t2s_overnight.py).

Reuses: server_guard (stale-server guard), run_provenance (blob-verified deploy), gguf_meta (KV bytes/token),
level_zero_sysman (iGPU frequency, power, temperature), t2s_m3_power_coupling (spin co-runner, affinity), the Phase D
AWE balloon script (memory_balloon_awe.py, launched as a subprocess, with output to a file this time).

Evo-t2s is another person's machine: only PIDs started here are ever killed, and every setting changed here
(power-plan PROCTHROTTLEMAX) is restored in a finally block and read back.
"""

from __future__ import annotations

import bisect
import base64
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
sys.path.insert(0, r"C:\apu\APU\harness")
import gguf_meta as gm  # noqa: E402
import level_zero_sysman as lz  # noqa: E402
import server_guard as sg  # noqa: E402
import t2s_m3_power_coupling as m3  # noqa: E402  (spin co-runner, kill_tree, read_ips)
import context as ctx_mod  # noqa: E402

PORT = 8385
SERVER_URL = f"http://127.0.0.1:{PORT}"
PYTHON = sys.executable
MODELS_DIR = r"C:\apu\models"
BINARIES = {"vulkan": r"C:\apu\bin\llama-b10970\llama-server.exe",
            "sycl": r"C:\apu\bin\llama-b10970-sycl\llama-server.exe"}
PID_FILE = r"C:\apu\ovn\cur_pid.txt"
PROBES_DIR = Path(r"C:\apu\APU\evaluation\probes")
FILE_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 7: "Q8_0", 12: "Q3_K_M", 14: "Q4_K_S", 15: "Q4_K_M", 17: "Q5_K_M", 18: "Q6_K"}
ERR_RE = re.compile(r"(error|failed|device lost|out of memory|abort|exception|cannot|unable)", re.I)


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    print(f"[{utc_iso()}] {msg}", flush=True)


def ps(cmd, timeout=30):
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout.strip()
    except Exception:
        return ""


class MEMSTATUS(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD), ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64), ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]


def avail_mb() -> float:
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMSTATUS)]
    k.GlobalMemoryStatusEx.restype = wt.BOOL
    s = MEMSTATUS()
    s.dwLength = ctypes.sizeof(MEMSTATUS)
    k.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 2 ** 20


def total_mb() -> float:
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMSTATUS)]
    k.GlobalMemoryStatusEx.restype = wt.BOOL
    s = MEMSTATUS()
    s.dwLength = ctypes.sizeof(MEMSTATUS)
    k.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullTotalPhys / 2 ** 20


class Jsonl:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, row):
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())


class Ring:
    """Time-ordered samples with window queries. Samples are dicts with an epoch 't'."""

    def __init__(self):
        self.t, self.rows = [], []
        self._lock = threading.Lock()

    def add(self, row):
        with self._lock:
            self.t.append(row["t"])
            self.rows.append(row)

    def window(self, t0, t1):
        with self._lock:
            i, j = bisect.bisect_left(self.t, t0), bisect.bisect_right(self.t, t1)
            return self.rows[i:j]

    def last(self):
        with self._lock:
            return self.rows[-1] if self.rows else None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def _max(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def iso_epoch(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


SYS_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$em = @()
try { $em = (Get-Counter -ListSet 'Energy Meter' -ErrorAction Stop).PathsWithInstances | Where-Object { $_ -like '*Power*' } } catch {}
$paths = @('\Memory\Available MBytes','\Memory\Committed Bytes','\Memory\Pages Input/sec','\Memory\Page Reads/sec',
           '\Memory\Page Faults/sec','\Processor Information(_Total)\% Processor Utility',
           '\GPU Adapter Memory(*)\Shared Usage','\GPU Adapter Memory(*)\Dedicated Usage') + $em
Get-Counter -Counter $paths -SampleInterval 1 -Continuous | ForEach-Object {
  $s = $_.CounterSamples
  $v = { param($like) ($s | Where-Object { $_.Path -like $like } | Select-Object -First 1).CookedValue }
  $sh = ($s | Where-Object { $_.Path -like '*gpu adapter memory*shared usage' } | Measure-Object CookedValue -Sum).Sum
  $de = ($s | Where-Object { $_.Path -like '*gpu adapter memory*dedicated usage' } | Measure-Object CookedValue -Sum).Sum
  $pk = ($s | Where-Object { $_.Path -like '*energy meter*rapl_package0_pkg*' } | Select-Object -First 1).CookedValue
  $o = [ordered]@{ ts = [DateTime]::UtcNow.ToString('o'); avail_mb = (& $v '*available mbytes'); committed_bytes = (& $v '*committed bytes');
                   pages_input = (& $v '*pages input/sec'); page_reads = (& $v '*page reads/sec'); page_faults = (& $v '*page faults/sec');
                   cpu_util = (& $v '*% processor utility'); adapter_shared = $sh; adapter_dedicated = $de; rapl_pkg_mw = $pk }
  [Console]::Out.WriteLine(($o | ConvertTo-Json -Compress)); [Console]::Out.Flush()
}
"""

GPU_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$pidFile = '__PIDFILE__'
while ($true) {
  $t0 = [DateTime]::UtcNow
  $p = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  $sh = $null; $de = $null
  if ($p) {
    try {
      $c = Get-Counter -Counter @("\GPU Process Memory(pid_$($p)_*)\Shared Usage","\GPU Process Memory(pid_$($p)_*)\Dedicated Usage") -ErrorAction Stop
      $sh = ($c.CounterSamples | Where-Object { $_.Path -like '*shared usage' } | Measure-Object CookedValue -Sum).Sum
      $de = ($c.CounterSamples | Where-Object { $_.Path -like '*dedicated usage' } | Measure-Object CookedValue -Sum).Sum
    } catch {}
  }
  $o = [ordered]@{ ts = $t0.ToString('o'); pid = $p; shared = $sh; dedicated = $de }
  [Console]::Out.WriteLine(($o | ConvertTo-Json -Compress)); [Console]::Out.Flush()
  $left = 1000 - ([DateTime]::UtcNow - $t0).TotalMilliseconds
  if ($left -gt 0) { Start-Sleep -Milliseconds $left }
}
"""


class Telemetry:
    """Sysman (1 s, in-process), Windows system counters (streaming), per-PID GPU memory (loop), own Available sampler."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.sysman = lz.Sysman()
        self.sys_ring, self.win_ring, self.gpu_ring, self.avail_ring = Ring(), Ring(), Ring(), Ring()
        self._procs = []
        self._stop = threading.Event()
        self._threads = []

    def _pump(self, proc, path, ring, conv):
        def run():
            with open(path, "a", encoding="utf-8") as f:
                for line in proc.stdout:
                    f.write(line if line.endswith("\n") else line + "\n")
                    f.flush()
                    try:
                        d = json.loads(line)
                        d["t"] = iso_epoch(d["ts"])
                        ring.add(d)
                    except Exception:
                        pass
        t = threading.Thread(target=run, daemon=True)
        t.start()
        self._threads.append(t)

    def _ps_proc(self, script):
        enc = base64.b64encode(script.encode("utf-16-le")).decode()
        return subprocess.Popen(["powershell", "-NoProfile", "-EncodedCommand", enc], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True, bufsize=1)

    def start(self):
        Path(PID_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(PID_FILE).write_text("")
        p1 = self._ps_proc(SYS_PS)
        p2 = self._ps_proc(GPU_PS.replace("__PIDFILE__", PID_FILE))
        self._procs += [p1, p2]
        self._pump(p1, self.prefix + "_winsys.jsonl", self.win_ring, None)
        self._pump(p2, self.prefix + "_wingpu.jsonl", self.gpu_ring, None)

        def sysman_loop():
            with open(self.prefix + "_sysman.csv", "w", encoding="utf-8") as f:
                f.write("ts_iso_utc,kind,domain,rc,actual_mhz,request_mhz,tdp_mhz,efficient_mhz,throttle_reasons,"
                        "energy_uj,timestamp_us,power_w,temp_c\n")
                nxt = time.monotonic()
                while not self._stop.is_set():
                    now = time.time()
                    ts = datetime.fromtimestamp(now, timezone.utc).isoformat()
                    for r in self.sysman.sample():
                        r["t"] = now
                        self.sys_ring.add(r)
                        f.write(",".join([ts, r["kind"], str(r["domain"]), str(r["rc"])] + [
                            "" if r.get(k) is None else str(r[k]) for k in
                            ("actual_mhz", "request_mhz", "tdp_mhz", "efficient_mhz", "throttle_reasons", "energy_uj",
                             "timestamp_us", "power_w", "temp_c")]) + "\n")
                    f.flush()
                    self.avail_ring.add({"t": now, "avail_mb": avail_mb()})
                    nxt += 1.0
                    self._stop.wait(max(0.0, nxt - time.monotonic()))
        t = threading.Thread(target=sysman_loop, daemon=True)
        t.start()
        self._threads.append(t)

    def set_pid(self, pid):
        Path(PID_FILE).write_text("" if pid is None else str(pid))

    def stop(self):
        self._stop.set()
        for p in self._procs:
            m3.kill_tree(p.pid)

    def temp_now(self):
        rows = [r for r in self.sys_ring.window(time.time() - 5, time.time()) if r["kind"] == "temp" and r.get("temp_c")]
        return max((r["temp_c"] for r in rows), default=None)

    def metrics(self, t0, t1):
        f = [r for r in self.sys_ring.window(t0, t1) if r["kind"] == "freq" and r["domain"] == 0]
        w = self.win_ring.window(t0, t1)
        g = self.gpu_ring.window(t0, t1)
        a = self.avail_ring.window(t0, t1)
        pk = [x.get("rapl_pkg_mw") for x in w]
        return {
            "igpu_mhz": _med([r["actual_mhz"] for r in f]),
            "igpu_throttle_bits": sorted({int(r["throttle_reasons"]) for r in f}) if f else None,
            "pkg_power_w": (_med(pk) / 1000) if _med(pk) is not None else None,
            "shared_usage_mib": (_max([x.get("shared") for x in g]) / 2 ** 20) if _max([x.get("shared") for x in g]) is not None else None,
            "total_committed_mib": (_max([x.get("committed_bytes") for x in w]) / 2 ** 20) if _max([x.get("committed_bytes") for x in w]) is not None else None,
            "pages_input_per_s": _max([x.get("pages_input") for x in w]),
            "hard_faults_per_s": _max([x.get("page_reads") for x in w]),
            "avail_mb_min": min((x["avail_mb"] for x in a), default=None),
            "avail_mb_max": max((x["avail_mb"] for x in a), default=None),
            "temp_c_max": _max([r.get("temp_c") for r in self.sys_ring.window(t0, t1) if r["kind"] == "temp"]),
        }

    def thermal_gate(self, idle_temp, tol=3.0, max_wait=300.0):
        t0 = time.monotonic()
        temp = self.temp_now()
        start = temp
        if idle_temp is None or temp is None:
            return {"thermal_wait_s": 0.0, "temp_at_gate_start": start, "temp_at_release": temp, "idle_temp": idle_temp,
                    "gate_released_by": "no_temperature"}
        released = "temp"
        while temp is not None and temp > idle_temp + tol:
            if time.monotonic() - t0 > max_wait:
                released = "timeout"
                break
            time.sleep(5)
            temp = self.temp_now()
        return {"thermal_wait_s": round(time.monotonic() - t0, 1), "temp_at_gate_start": start, "temp_at_release": temp,
                "idle_temp": idle_temp, "gate_released_by": released}


class ModelInfo:
    def __init__(self, model_id, path, sha256, thinking_hybrid, max_ctx_native, yarn_factor=1):
        self.model_id, self.path, self.sha256 = model_id, path, sha256
        self.thinking_hybrid, self.max_ctx_native, self.yarn_factor = thinking_hybrid, max_ctx_native, yarn_factor
        self.meta = gm.read_gguf_meta(path)
        self.p = gm.arch_params(self.meta)
        self.kv_bpt_meta = gm.kv_bytes_per_token_f16(self.meta)
        self.quant = FILE_TYPE.get(self.p["file_type"], str(self.p["file_type"]))
        self.file_bytes = os.path.getsize(path)


def parse_server_log(path):
    """Buffer sizes and the device memory line from a llama-server log (verbosity 3)."""
    out = {"kv_buffer_mib": None, "compute_buffer_mib": None, "model_buffer_mib": None, "device_line": None,
           "device_free_mib": None, "mmap_lines": []}
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    kv = re.findall(r"KV buffer size\s*=\s*([\d.]+)\s*MiB", txt)
    out["kv_buffer_mib"] = sum(float(x) for x in kv) if kv else None
    cb = re.findall(r"compute buffer size\s*=\s*([\d.]+)\s*MiB", txt)
    out["compute_buffer_mib"] = sum(float(x) for x in cb) if cb else None
    mb = re.findall(r"model buffer size\s*=\s*([\d.]+)\s*MiB", txt)
    out["model_buffer_mib"] = sum(float(x) for x in mb) if mb else None
    m = re.search(r"using device (Vulkan\d|SYCL\d)[^\n]*?-\s*([\d]+)\s*MiB free", txt)
    if m:
        out["device_line"] = m.group(0)
        out["device_free_mib"] = float(m.group(2))
    out["mmap_lines"] = [l.strip()[:200] for l in txt.splitlines() if "mmap" in l.lower()][:6]
    out["error_lines"] = [l.strip()[:300] for l in txt.splitlines() if ERR_RE.search(l)][-6:]
    return out


class Server:
    def __init__(self, lab, mi: ModelInfo, n_ctx, backend="vulkan", mmap=True, extra=(), tag="s"):
        self.lab, self.mi, self.n_ctx, self.backend, self.mmap, self.extra, self.tag = lab, mi, n_ctx, backend, mmap, list(extra), tag
        self.proc = None
        self.pid = None
        self.log_path = str(Path(lab.prefix + f"_srv_{tag}.txt"))
        self.guard = None
        self.props = {}
        self.start_info = {}

    def _cmd(self):
        c = [BINARIES[self.backend], "-m", self.mi.path, "--port", str(PORT), "-c", str(self.n_ctx), "-ctk", "f16",
             "-ctv", "f16", "-fa", "on", "-ngl", "99", "-np", "1", "-t", "4", "--no-context-shift",
             "--log-file", self.log_path, "--log-verbosity", "3"]
        if not self.mmap:
            c.append("--no-mmap")
        return c + self.extra

    def start(self, timeout=1500):
        """Returns a dict: ok, load_s, error, exit_code, pid, log parse. Never raises for a server that fails to load."""
        try:
            sg.assert_port_free(PORT)
        except sg.GuardError as e:
            raise RuntimeError("STOP guard: " + str(e))
        if ps("(Get-Process llama-server -ErrorAction SilentlyContinue | Measure-Object).Count") not in ("", "0"):
            raise RuntimeError("STOP: a llama-server process we did not start is running")
        Path(self.log_path).unlink(missing_ok=True)
        t0 = time.time()
        self.proc = subprocess.Popen(self._cmd(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        self.pid = self.proc.pid
        self.lab.tele.set_pid(self.pid)
        healthy = False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        healthy = True
                        break
            except Exception:
                pass
            time.sleep(1)
        t1 = time.time()
        info = {"t_start": t0, "t_end": t1, "load_s": t1 - t0, "pid": self.pid}
        lp = parse_server_log(self.log_path)
        info["log"] = lp
        if not healthy:
            code = self.proc.poll()
            err = "hung past load timeout" if code is None else f"process exited with code {code} ({code & 0xFFFFFFFF:#x})"
            tail = " | ".join(lp.get("error_lines", [])[-3:])
            info.update({"ok": False, "exit_code": code, "error": (err + (" ; log: " + tail if tail else ""))[:600]})
            self.stop()
            self.start_info = info
            return info
        try:
            exp = {"model_path": self.mi.path, "n_ctx": self.n_ctx, "ctk": "f16", "ctv": "f16", "fa": "on", "ngl": 99,
                   "np": 1, "threads": 4}
            rec = sg.assert_server_matches(SERVER_URL, PORT, self.pid, exp)
        except sg.GuardError as e:
            info.update({"ok": False, "exit_code": None, "error": "guard: " + str(e)[:500], "guard": e.record})
            self.stop()
            self.start_info = info
            return info
        self.guard = sg.RequestGuard(PORT, self.pid)
        self.props = rec
        info.update({"ok": True, "exit_code": None, "error": None, "guard": rec, "build": rec.get("props_build_info")})
        self.start_info = info
        return info

    def alive_and_ours(self):
        if self.proc is None or self.proc.poll() is not None:
            return False, []
        return self.guard.check()

    def stop(self):
        if self.proc is not None and self.pid:
            for _ in range(3):
                m3.kill_tree(self.pid)
                d = time.monotonic() + 45
                while time.monotonic() < d:
                    alive = ps(f"(Get-Process -Id {self.pid} -ErrorAction SilentlyContinue | Measure-Object).Count", 20) not in ("", "0")
                    if not alive and not sg.port_listeners(PORT):
                        self.lab.tele.set_pid(None)
                        return
                    time.sleep(2)
            raise RuntimeError(f"STOP: could not confirm server pid {self.pid} exited and port {PORT} freed")
        self.lab.tele.set_pid(None)

    # --- requests ---
    def tokenize(self, text):
        body = json.dumps({"content": text, "add_special": False}).encode()
        req = urllib.request.Request(f"{SERVER_URL}/tokenize", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return len(json.loads(r.read())["tokens"])

    def chat(self, prompt, max_tokens, ignore_eos, timeout=2700):
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
                "seed": 42, "stream": True, "cache_prompt": False, "stream_options": {"include_usage": True}}
        if ignore_eos:
            body["ignore_eos"] = True
        if self.mi.thinking_hybrid:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(f"{SERVER_URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        t_first = t_last = None
        n_content, usage_ct, pieces = 0, None, []
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                for raw in r:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    ch = chunk.get("choices") or []
                    c = ((ch[0] if ch else {}).get("delta") or {}).get("content")
                    if c:
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
                        n_content += 1
                        pieces.append(c)
                    if chunk.get("usage"):
                        usage_ct = chunk["usage"].get("completion_tokens")
        except Exception as e:
            return {"outcome": "timeout" if "timed out" in str(e).lower() else "error", "error": str(e)[:400], "output": None}
        e2e = time.perf_counter() - t0
        if t_first is None:
            return {"outcome": "error", "error": "no content tokens", "output": None, "e2e_s": e2e}
        ttft = t_first - t0
        ct = usage_ct if usage_ct is not None else n_content
        dec = (ct - 1) / (t_last - t_first) if t_last and t_last > t_first and ct > 1 else None
        out = "".join(pieces)
        return {"outcome": "ok", "error": None, "output": out, "ttft_s": ttft, "decode_tok_s": dec, "e2e_s": e2e,
                "completion_tokens": ct, "usage_reported": usage_ct is not None, "think_tag": "<think" in out}


class Balloon:
    """Phase D AWE balloon as a subprocess. Output to a file (a pipe nobody reads blocks the balloon), low-available
    valve off, light sampling, heartbeat touched here every 20 s. Killed by PID only."""

    def __init__(self, lab, script_path, tag):
        self.lab, self.script, self.tag = lab, script_path, tag
        self.proc = None
        self.hb = lab.prefix + f"_balloon_{tag}.hb"
        self.log_csv = lab.prefix + f"_balloon_{tag}.csv"
        self.out = lab.prefix + f"_balloon_{tag}.out.txt"
        self._stop_hb = threading.Event()

    def start(self, target_available_mb, tolerance_mb=250.0, reach_timeout=240):
        Path(self.hb).write_text(str(time.time()))
        cmd = [PYTHON, self.script, "--target-available-mb", str(target_available_mb), "--tolerance-mb", str(tolerance_mb),
               "--heartbeat-file", self.hb, "--heartbeat-timeout-s", "600", "--low-available-floor-mb", "0",
               "--max-runtime-s", "20000", "--log-path", self.log_csv, "--sample-interval-s", "5", "--light"]
        self.out_f = open(self.out, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.out_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

        def hb():
            while not self._stop_hb.is_set() and self.proc.poll() is None:
                try:
                    Path(self.hb).write_text(str(time.time()))
                except Exception:
                    pass
                self._stop_hb.wait(20)
        threading.Thread(target=hb, daemon=True).start()
        d = time.monotonic() + reach_timeout
        while time.monotonic() < d:
            if self.proc.poll() is not None:
                return {"ok": False, "why": f"balloon exited early rc={self.proc.returncode}", "out": Path(self.out).read_text(errors="replace")[-400:]}
            txt = Path(self.out).read_text(errors="replace") if os.path.exists(self.out) else ""
            if "TARGET REACHED" in txt:
                a = avail_mb()
                return {"ok": abs(a - target_available_mb) <= tolerance_mb * 2, "available_mb_after": a, "why": None}
            time.sleep(2)
        return {"ok": False, "why": "balloon did not reach target in time"}

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        self._stop_hb.set()
        if self.proc is not None:
            m3.kill_tree(self.proc.pid)
            try:
                self.proc.wait(timeout=30)
            except Exception:
                pass
        try:
            self.out_f.close()
        except Exception:
            pass


class PowerCap:
    """PROCTHROTTLEMAX on AC for the active scheme. Original value is restored and read back in restore()."""

    def __init__(self):
        q = ps("powercfg /getactivescheme")
        m = re.search(r"([0-9a-fA-F-]{36})", q)
        self.guid = m.group(1) if m else None
        self.original = self.read()
        self.current = self.original
        self.changed = False

    def read(self):
        out = ps("powercfg /query SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX")
        m = re.search(r"Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+)", out)
        return int(m.group(1), 16) if m else None

    def set(self, pct):
        ps(f"powercfg /setacvalueindex {self.guid} SUB_PROCESSOR PROCTHROTTLEMAX {int(pct)}; powercfg /setactive {self.guid}")
        self.changed = True
        self.current = self.read()
        return self.current

    def restore(self):
        if self.original is None:
            return {"restored": False, "why": "original value unknown"}
        ps(f"powercfg /setacvalueindex {self.guid} SUB_PROCESSOR PROCTHROTTLEMAX {self.original}; powercfg /setactive {self.guid}")
        back = self.read()
        self.current = back
        return {"restored": back == self.original, "original": self.original, "read_back": back}


def load_probes():
    import importlib.util
    spec = importlib.util.spec_from_file_location("probes_scorers", PROBES_DIR / "scorers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    segs = {}
    for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l)
            segs[d["id"]] = d
    return mod, [segs[f"art_{i:02d}"] for i in range(1, 6)]


def build_probe_prompts(server: Server, fill_tokens, probes):
    """filler, artifact, question (artifact adjacent to the question), total about fill_tokens tokens."""
    tk = server.tokenize
    min_aq = min(tk(p["artifact"].strip()) + tk(p["question"].strip()) for p in probes)
    filler_full = ctx_mod.build_filler(max(fill_tokens - min_aq, 256) + 64, seed=42, count_fn=tk)
    full_tokens = tk(filler_full)

    def left_trunc(text, target):
        if tk(text) <= target:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi) // 2
            if tk(text[mid:]) <= target:
                hi = mid
            else:
                lo = mid + 1
        return text[lo:]

    out = []
    for p in probes:
        a, q = p["artifact"].strip(), p["question"].strip()
        per = max(fill_tokens - tk(a) - tk(q), 64)
        filler = left_trunc(filler_full, min(per, full_tokens))
        prompt = f"{filler}\n\n{a}\n\n{q}"
        out.append((p, prompt, tk(prompt)))
    return out
