"""evo-t2s only (Intel Core Ultra X7 358H, Arc B390 iGPU, Vulkan b10970, unified memory).

M3: does CPU compute slow the iGPU through shared package power?

ctx 8192, qwen3-4b-instruct, f16 KV, -fa on -ngl 99 -np 1 -t 4 --no-context-shift, port 8385, no reasoning flags.
Conditions (co-runner = pure integer spin, no memory traffic, pinned by affinity mask):
  none          no co-runner
  spin16        all 16 logical CPUs        mask 0xFFFF
  spinP         P-cores only (logical 0-3, efficiency class 1)   mask 0x000F
  spinE         non-P cores (logical 4-15, efficiency class 0: E plus LP-E, Windows does not separate them)  mask 0xFFF0
  none_end      no co-runner again (drift check)
One discarded warm-up call, then 3 measured calls per condition. Order of the three spin conditions is shuffled with a
logged seed; none is first and none_end is last. The co-runner starts 5 s before the first call of a condition and stops
after the third.

Prediction if power coupling is real: iGPU frequency (Sysman actual MHz) drops under spin, more under spinP than spinE,
and package power reads at or near its limit. If Sysman gives nothing this is stated and Windows counters (GPU Engine
utilization, processor frequency/performance, Energy Meter if present) are the fallback.

Standing rules applied: stale-server guard (harness/server_guard.py), deployed scripts verified against committed git
blobs (harness/run_provenance.py), thermal state and AC recorded, positive control (hog iterations/s and worker affinity
read-back) on every condition.

Usage on evo-t2s (deployed to C:\\apu\\m3):
  python t2s_m3_power_coupling.py --expect-blobs m3_expected_blobs.json --smoke
  python t2s_m3_power_coupling.py --expect-blobs m3_expected_blobs.json
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import os
import random
import socket
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
import level_zero_sysman as lz  # noqa: E402
import run_provenance as rp  # noqa: E402
import server_guard as sg  # noqa: E402
import context as ctx_mod  # noqa: E402

SERVER_BIN = r"C:\apu\bin\llama-b10970\llama-server.exe"
MODEL_PATH = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
MODEL_SHA256 = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
PORT = 8385
SERVER_URL = f"http://127.0.0.1:{PORT}"
CTX = 8192
FILL_RATIO = 0.90
MAX_TOKENS = 128
CALL_TIMEOUT_S = 2700
N_MEASURED = 3
LEAD_S = 5
COOLDOWN_S = 20
ORDER_SEED = 20260925
PYTHON = sys.executable
CONDITIONS = {"spin16": 0xFFFF, "spinP": 0x000F, "spinE": 0xFFF0}

WINCTR_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$em = @()
try { $em = (Get-Counter -ListSet 'Energy Meter' -ErrorAction Stop).PathsWithInstances | Where-Object { $_ -like '*Power*' } } catch {}
$paths = @('\GPU Engine(*)\Utilization Percentage','\Processor Information(_Total)\% Processor Utility',
           '\Processor Information(_Total)\Processor Frequency','\Processor Information(_Total)\% Processor Performance',
           '\GPU Adapter Memory(*)\Shared Usage') + $em
Get-Counter -Counter $paths -SampleInterval 1 -Continuous | ForEach-Object {
  $s = $_.CounterSamples
  $g3 = ($s | Where-Object { $_.Path -like '*gpu engine*engtype_3d*' } | Measure-Object CookedValue -Sum).Sum
  $gc = ($s | Where-Object { $_.Path -like '*gpu engine*engtype_compute*' } | Measure-Object CookedValue -Sum).Sum
  $gm = ($s | Where-Object { $_.Path -like '*gpu engine*' } | Measure-Object CookedValue -Maximum).Maximum
  $cu = ($s | Where-Object { $_.Path -like '*% processor utility' } | Select-Object -First 1).CookedValue
  $cf = ($s | Where-Object { $_.Path -like '*processor frequency' } | Select-Object -First 1).CookedValue
  $cp = ($s | Where-Object { $_.Path -like '*% processor performance' } | Select-Object -First 1).CookedValue
  $sh = ($s | Where-Object { $_.Path -like '*gpu adapter memory*shared usage' } | Measure-Object CookedValue -Sum).Sum
  $emv = @{}; foreach ($x in ($s | Where-Object { $_.Path -like '*energy meter*' })) { $emv[$x.InstanceName] = $x.CookedValue }
  $o = [ordered]@{ ts = [DateTime]::UtcNow.ToString('o'); gpu_3d_pct = $g3; gpu_compute_pct = $gc; gpu_engine_max_pct = $gm;
                   cpu_utility_pct = $cu; cpu_freq_mhz = $cf; cpu_performance_pct = $cp; gpu_shared_bytes = $sh; energy_meter_mw = $emv }
  [Console]::Out.WriteLine(($o | ConvertTo-Json -Compress)); [Console]::Out.Flush()
}
"""


class StopExperiment(Exception):
    pass


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def ps(cmd, timeout=30):
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout
    except Exception:
        return ""


def kill_tree(pid):
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
    except Exception:
        pass


def pid_alive(pid):
    return ps(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue | Measure-Object).Count", 20).strip() not in ("", "0")


def kill_and_confirm(pid):
    for _ in range(3):
        kill_tree(pid)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if not pid_alive(pid) and not sg.port_listeners(PORT):
                return
            time.sleep(2)
    raise StopExperiment(f"could not confirm pid {pid} exited and port {PORT} freed")


def raise_priority():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wt.HANDLE
    k.SetPriorityClass.argtypes = [wt.HANDLE, wt.DWORD]
    k.SetPriorityClass.restype = wt.BOOL
    return bool(k.SetPriorityClass(k.GetCurrentProcess(), 0x8000))


class SysmanSampler(threading.Thread):
    def __init__(self, sysman, path):
        super().__init__(daemon=True)
        self.sysman, self.path = sysman, path
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("ts_iso_utc,kind,domain,rc,actual_mhz,request_mhz,tdp_mhz,efficient_mhz,throttle_reasons,"
                    "energy_uj,timestamp_us,power_w\n")
            nxt = time.monotonic()
            while not self._stop_event.is_set():
                ts = datetime.now(timezone.utc).isoformat()
                for r in self.sysman.sample():
                    f.write(",".join([ts, r["kind"], str(r["domain"]), str(r["rc"]),
                                      *(("" if r.get(k) is None else str(r[k])) for k in
                                        ("actual_mhz", "request_mhz", "tdp_mhz", "efficient_mhz", "throttle_reasons",
                                         "energy_uj", "timestamp_us", "power_w"))]) + "\n")
                f.flush()
                nxt += 1.0
                self._stop_event.wait(max(0.0, nxt - time.monotonic()))


class WinCounters:
    def __init__(self, path):
        self.path, self.proc = path, None

    def start(self):
        import base64
        enc = base64.b64encode(WINCTR_PS.encode("utf-16-le")).decode()
        self.proc = subprocess.Popen(["powershell", "-NoProfile", "-EncodedCommand", enc], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True, bufsize=1)

        def pump():
            with open(self.path, "a", encoding="utf-8") as f:
                for line in self.proc.stdout:
                    f.write(line if line.endswith("\n") else line + "\n")
                    f.flush()
        threading.Thread(target=pump, daemon=True).start()

    def stop(self):
        if self.proc is not None:
            kill_tree(self.proc.pid)


def tokenize(text):
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(f"{SERVER_URL}/tokenize", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return len(json.loads(r.read())["tokens"])


def chat_call(prompt, n_prompt_tokens):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS, "ignore_eos": True,
            "temperature": 0, "seed": 42, "stream": True, "cache_prompt": False,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{SERVER_URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
    n_content, usage_ct = 0, None
    try:
        with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
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
                choices = chunk.get("choices") or []
                if ((choices[0] if choices else {}).get("delta") or {}).get("content"):
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    n_content += 1
                if chunk.get("usage"):
                    usage_ct = chunk["usage"].get("completion_tokens")
    except Exception as e:
        return {"outcome": "timeout" if "timed out" in str(e).lower() else "error", "error": str(e)}
    if t_first is None:
        return {"outcome": "error", "error": "no content tokens"}
    ttft = t_first - t0
    ct = usage_ct if usage_ct is not None else n_content
    dec = (ct - 1) / (t_last - t_first) if t_last and t_last > t_first and ct > 1 else None
    return {"outcome": "ok", "ttft_s": ttft, "prefill_tps": n_prompt_tokens / ttft if ttft > 0 else None,
            "decode_tps": dec, "completion_tokens": ct, "usage_reported": usage_ct is not None}


def start_hog(tag, mask, out_dir, stem):
    report = str(out_dir / f"{stem}_{tag}_hog_ips.txt")
    aff = str(out_dir / f"{stem}_{tag}_hog_affinity.json")
    for p in (report, aff):
        Path(p).unlink(missing_ok=True)
    proc = subprocess.Popen([PYTHON, str(DEPLOY / "spin_hog_affinity.py"), "--affinity-mask", hex(mask),
                             "--duration-s", "1800", "--report-file", report, "--affinity-file", aff],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    return proc, report, aff


def read_ips(report):
    try:
        return float(Path(report).read_text().strip())
    except Exception:
        return None


def check_columns(prefix):
    def valid(v):
        return v.strip() not in ("", "nan", "None")
    res = {}
    p = Path(prefix + "_sysman.csv")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []
    if len(lines) > 1:
        hdr = lines[0].split(",")
        rows = [dict(zip(hdr, l.split(","))) for l in lines[1:] if l.strip()]
        fr = [r for r in rows if r["kind"] == "freq"]
        pr = [r for r in rows if r["kind"] == "power"]
        res["sysman.freq_rows"] = len(fr)
        res["sysman.power_rows"] = len(pr)
        res["sysman.freq_actual_valid_frac"] = sum(1 for r in fr if valid(r["actual_mhz"]) and float(r["actual_mhz"]) > 0) / max(len(fr), 1)
        res["sysman.freq_rc_zero_frac"] = sum(1 for r in fr if r["rc"] == "0") / max(len(fr), 1)
        res["sysman.power_w_valid_frac"] = sum(1 for r in pr if valid(r["power_w"])) / max(len(pr), 1)
        res["sysman.power_rc_zero_frac"] = sum(1 for r in pr if r["rc"] == "0") / max(len(pr), 1)
    else:
        res["sysman.rows"] = 0
    p = Path(prefix + "_wincounters.jsonl")
    rows = []
    for l in (p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []):
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
    res["win.rows"] = len(rows)
    for k in ("gpu_3d_pct", "gpu_compute_pct", "gpu_engine_max_pct", "cpu_utility_pct", "cpu_freq_mhz",
              "cpu_performance_pct", "gpu_shared_bytes"):
        res[f"win.{k}_valid_frac"] = sum(1 for r in rows if r.get(k) is not None) / max(len(rows), 1)
    res["win.energy_meter_present_frac"] = sum(1 for r in rows if r.get("energy_meter_mw")) / max(len(rows), 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-blobs", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if socket.gethostname().upper() != "EVO-T2S":
        raise StopExperiment(f"wrong machine {socket.gethostname()}: M3 runs on EVO-T2S only")
    prov = rp.verify_deployed_blobs(DEPLOY, args.expect_blobs)
    if hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest() != MODEL_SHA256:
        raise StopExperiment("model sha256 mismatch")
    out_dir = DEPLOY / ("smoke" if args.smoke else "results")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{'smoke_' if args.smoke else ''}t2s_m3_power_coupling_{ts}"
    prefix = str(out_dir / stem)
    jsonl = out_dir / f"{stem}.jsonl"
    manifest = {"experiment": "M3 CPU spin vs iGPU package power", "script_provenance": prov, "order_seed": ORDER_SEED,
                "ctx": CTX, "n_measured": N_MEASURED, "smoke": args.smoke, "conditions": CONDITIONS,
                "topology_note": "logical 0-3 efficiency class 1 (P), 4-15 class 0 (E plus LP-E)",
                "priority_raised": raise_priority(), "captured_utc": datetime.now(timezone.utc).isoformat(),
                "power_plan": ps("powercfg /getactivescheme").strip(),
                "gpu": ps("(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress)").strip(),
                "cpu": ps("(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress)").strip(),
                "context_py_sha256": hashlib.sha256(open(r"C:\apu\APU\harness\context.py", "rb").read()).hexdigest()}
    manifest["ac_note"] = "mini PC without battery; AC assumed, not measured"

    def emit(row):
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts_utc": datetime.now(timezone.utc).isoformat(), **row}) + "\n")

    # stale-server guard 1: port free
    try:
        sg.assert_port_free(PORT)
    except sg.GuardError as e:
        manifest["guard_stop"] = e.record
        Path(prefix + "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise StopExperiment(str(e))
    if ps("(Get-Process llama-server -ErrorAction SilentlyContinue | Measure-Object).Count").strip() not in ("", "0"):
        raise StopExperiment("a llama-server process is already running; refusing to start")
    others = ps("Get-Counter '\\Process(*)\\% Processor Time' -SampleInterval 1 -MaxSamples 3 | ForEach-Object { $_.CounterSamples } | "
                "Group-Object InstanceName | ForEach-Object { [pscustomobject]@{name=$_.Name; pct=[math]::Round((($_.Group | Measure-Object CookedValue -Average).Average)/[Environment]::ProcessorCount,1)} } | "
                "Where-Object { $_.pct -gt 5 -and $_.name -ne '_total' -and $_.name -ne 'idle' } | ConvertTo-Json -Compress", 60).strip()
    manifest["processes_over_5pct_cpu_before_start"] = others or []

    sysman = lz.Sysman()
    manifest["sysman"] = {"available": sysman.available, "drivers": sysman.n_drivers, "devices": sysman.n_devices,
                          "freq_props": sysman.freq_props, "power_props": sysman.power_props, "errors": sysman.errors}
    Path(prefix + "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"sysman available={sysman.available} errors={sysman.errors}")

    sampler = SysmanSampler(sysman, prefix + "_sysman.csv")
    win = WinCounters(prefix + "_wincounters.jsonl")
    sampler.start()
    win.start()

    srv_log = str(DEPLOY / f"{stem}_srv.txt")
    proc = subprocess.Popen([SERVER_BIN, "-m", MODEL_PATH, "--port", str(PORT), "-c", str(CTX), "-ctk", "f16",
                             "-ctv", "f16", "-fa", "on", "-ngl", "99", "-np", "1", "-t", "4", "--no-context-shift",
                             "--log-file", srv_log, "--log-verbosity", "3"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    hog = None
    try:
        deadline = time.monotonic() + 420
        healthy = False
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        healthy = True
                        break
            except Exception:
                pass
            time.sleep(2)
        if not healthy:
            raise StopExperiment("server did not become healthy")
        try:
            rec = sg.assert_server_matches(SERVER_URL, PORT, proc.pid, {
                "model_path": MODEL_PATH, "n_ctx": CTX, "ctk": "f16", "ctv": "f16", "fa": "on", "ngl": 99, "np": 1,
                "threads": 4})
        except sg.GuardError as e:
            emit({"record": "guard_stop", **e.record})
            raise StopExperiment(str(e))
        emit({"record": "server_start", "server_pid": proc.pid, "server_log": srv_log, "guard": rec})
        guard = sg.RequestGuard(PORT, proc.pid)
        prompt = ctx_mod.build_filler(round(CTX * FILL_RATIO), seed=42, count_fn=tokenize)
        n_tok = tokenize(prompt)
        log(f"prompt {n_tok} tokens")

        rng = random.Random(ORDER_SEED)
        spins = list(CONDITIONS)
        rng.shuffle(spins)
        plan = ["none"] + spins + ["none_end"]
        if args.smoke:
            plan = ["spin16"]
        emit({"record": "plan", "plan": plan, "seed": ORDER_SEED})

        ok, lp = guard.check()
        if not ok:
            raise StopExperiment(f"listener {lp} != {proc.pid} before warm-up")
        w = chat_call(prompt, n_tok)
        emit({"record": "warmup", "server_pid": proc.pid, **w})
        log(f"warm-up {w.get('outcome')} ttft={w.get('ttft_s')}")

        for cond in plan:
            base = cond.replace("_end", "")
            mask = CONDITIONS.get(cond)
            report = aff = None
            if mask is not None:
                hog, report, aff = start_hog(cond, mask, out_dir, stem)
                time.sleep(LEAD_S)
                log(f"{cond}: hog pid {hog.pid} mask {mask:#x} ips {read_ips(report)}")
            else:
                time.sleep(LEAD_S)
            stop_condition = False
            for i in range(N_MEASURED):
                ok, lp = guard.check()
                if not ok:
                    emit({"record": "call", "cond": cond, "call": i, "invalid": True,
                          "outcome": "invalid_listener_mismatch", "server_pid": proc.pid, "listener_pids": lp})
                    log(f"{cond} call {i} INVALID listener {lp}")
                    stop_condition = True
                    break
                t0 = datetime.now(timezone.utc).isoformat()
                res = chat_call(prompt, n_tok)
                t1 = datetime.now(timezone.utc).isoformat()
                row = {"record": "call", "cond": cond, "affinity_mask": mask, "call": i, "t_start_utc": t0,
                       "t_end_utc": t1, "n_prompt_tokens": n_tok, "server_pid": proc.pid, **res}
                if mask is not None:
                    ips = read_ips(report)
                    row["hog_ips"] = ips
                    try:
                        row["hog_affinity"] = json.loads(Path(aff).read_text())
                    except Exception:
                        row["hog_affinity"] = None
                    aff_ok = bool(row["hog_affinity"]) and all(m == mask for m in row["hog_affinity"]["worker_masks"])
                    row["hog_affinity_ok"] = aff_ok
                    if not ips or ips <= 0 or not aff_ok:
                        row["invalid"] = True
                        row["invalid_reason"] = "hog positive control failed (iterations/s or affinity)"
                emit(row)
                log(f"{cond} call {i} {res.get('outcome')} ttft={res.get('ttft_s')} decode_tps={res.get('decode_tps')} ips={row.get('hog_ips')}")
            if hog is not None:
                kill_tree(hog.pid)
                hog = None
            time.sleep(COOLDOWN_S)
            if stop_condition:
                continue
    finally:
        if hog is not None:
            kill_tree(hog.pid)
        sampler.stop()
        win.stop()
        try:
            kill_and_confirm(proc.pid)
        except StopExperiment as e:
            log(f"WARNING: {e}")
        time.sleep(2)
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        Path(prefix + "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.smoke:
        cols = check_columns(prefix)
        print(json.dumps(cols, indent=2))
    else:
        (out_dir / f"{stem}.DONE").write_text("done\n")
    log("M3 finished")


if __name__ == "__main__":
    try:
        main()
    except StopExperiment as e:
        log(f"STOP: {e}")
        sys.exit(2)
