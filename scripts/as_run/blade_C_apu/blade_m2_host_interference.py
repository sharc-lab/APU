"""
blade_m2_host_interference.py -- why does host memcpy slow a GPU-resident
model 7x on Blade? (ctx=8192, established fully in VRAM in Phase A/M1 data
-- GPU Process Memory Shared Usage checked again here, not assumed zero.)

PRIOR ART (searched before writing): "Characterizing CPU-Induced Slowdowns
in Multi-GPU LLM Inference" (arXiv 2603.22774) documents a real, related
phenomenon, but a DIFFERENT mechanism: multi-GPU tensor-parallel barrier
synchronization and tokenization control-plane bottlenecks under CPU
oversubscription. That paper's setup is 4-8 GPUs with inter-GPU collective
communication; ours is a single GPU with an unrelated co-running memcpy/spin
process competing for system RAM bandwidth. The specific question here --
does an independent host-memory-bandwidth hog slow a single-GPU, fully
VRAM-resident model, and through which of {host critical path / shared
power / PCIe bus contention} -- is not what that paper (or the general
PCIe-bandwidth-for-weight-offloading literature also found) addresses.
Legitimate, non-redundant local characterization.

DISCRIMINATING DESIGN:
  A. Baseline set: ctx=8192, -t 4 (the flags-always thread count), conditions
     {none, memcpy_16, spin_16}, 3 calls each. Telemetry: nvidia-smi dmon
     (power/clocks/utilization/PCIe rx-tx at ~1s) + GPU Process Memory
     Dedicated/Shared Usage for the server PID at ~1s.
  B. Thread-count discriminator: memcpy_16 condition rerun with the SERVER
     (not the co-runner) launched at -t 1 and -t 8 instead of -t 4.
     Prediction if the slowdown is a HOST-SIDE CRITICAL PATH effect (the
     server's own CPU threads starved of cycles/cache by the co-runner):
     -t 1 should show a LARGER slowdown than -t 8 is relatively protected
     by (or the reverse, if -t 8 spreads the server thinner and makes it
     MORE vulnerable to contention) -- either directional result is
     informative; a null result (both threading counts show identical
     slowdown) argues AGAINST a host-critical-path mechanism specifically
     tied to server thread scheduling.
  C. Dose-response: memcpy at 16 threads, throttled via
     bandwidth_hog_throttled.py to four target rates (5/10/15/20 GB/s
     aggregate), 3 calls each. Report slowdown vs. ACHIEVED (logged) GB/s,
     not the requested target -- pacing is approximate.

READING THE RESULT AGAINST THREE HYPOTHESES (stated up front, not fit after
the fact):
  - GPU utilization drops while clocks stay steady -> host-side critical
    path (GPU is idle waiting on the CPU, not power- or bus-limited).
  - SM clock or power draw drops -> shared power budget (the co-runner's
    CPU load is eating into a package-level power budget the GPU also
    draws from).
  - PCIe rx/tx (dmon rxpci/txpci) changes materially -> bus contention.
  This script does not pre-decide which is true; the FINAL REPORT states
  which hypothesis (if any) the data actually supports.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\Users\rithw\OneDrive\Documents\GitHub\APU")
RESULTS_DIR = REPO / "results"
SCRIPT_DIR = Path(r"C:\apu")

SERVER_BIN = r"C:\apu\bin\llama-b10970-cuda\llama-server.exe"
MODEL_PATH = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
MODEL_SHA256_EXPECTED = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
PORT = 8385
SERVER_URL = f"http://127.0.0.1:{PORT}"

CTX_SIZE = 8192
FILL_RATIO = 0.90
THROUGHPUT_MAX_TOKENS = 128
CALL_TIMEOUT_S = 600
SAMPLE_INTERVAL_S = 1.0
N_CALLS = 3
DOSE_TARGETS_GBPS = [5, 10, 15, 20]

sys.path.insert(0, str(REPO / "harness"))
import context as ctx_mod  # noqa: E402


def log_line(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def verify_model_sha256():
    h = hashlib.sha256()
    with open(MODEL_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != MODEL_SHA256_EXPECTED:
        raise RuntimeError(f"MODEL SHA256 MISMATCH: expected {MODEL_SHA256_EXPECTED}, got {got}")
    log_line(f"Model sha256 verified: {got}")


def ps(cmd: str, timeout: int = 15) -> str:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                            capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout
    except Exception:
        return ""


def kill_pid(pid: int):
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=15, stdin=subprocess.DEVNULL)
    except Exception:
        pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        r = ps(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue | Measure-Object).Count", timeout=10)
        if r.strip() in ("", "0"):
            return
        time.sleep(1)


def start_server(threads_flag: int, log_path: str) -> subprocess.Popen:
    cmd = [
        SERVER_BIN, "-m", MODEL_PATH, "--port", str(PORT), "-c", str(CTX_SIZE),
        "-ctk", "f16", "-ctv", "f16", "-fa", "on", "-ngl", "99", "-np", "1",
        "-t", str(threads_flag), "--no-context-shift", "--log-file", log_path, "--log-verbosity", "3",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


def wait_healthy(proc: subprocess.Popen, timeout: int = 180) -> bool:
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def tokenize(text: str) -> int:
    import urllib.request
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(f"{SERVER_URL}/tokenize", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return len(json.loads(r.read())["tokens"])


def chat_throughput_call(prompt: str, n_prompt_tokens: int) -> dict:
    import urllib.request
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": THROUGHPUT_MAX_TOKENS, "ignore_eos": True,
        "temperature": 0, "seed": 42, "stream": True, "cache_prompt": False,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{SERVER_URL}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first, t_last = None, None
    n_content = 0
    completion_tokens_usage = None
    try:
        with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
            for raw_line in r:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                choice0 = choices[0] if choices else {}
                delta = choice0.get("delta", {})
                content = delta.get("content", "")
                if content:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_content += 1
                usage = chunk.get("usage")
                if usage:
                    completion_tokens_usage = usage.get("completion_tokens")
        total_s = time.perf_counter() - t0
        if t_first is None:
            return {"status": "error", "error": "no content tokens"}
        ttft = t_first - t0
        completion_tokens = completion_tokens_usage if completion_tokens_usage is not None else n_content
        decode_s = (t_last - t_first) if (t_last and t_first and t_last > t_first) else None
        decode_tps = (completion_tokens - 1) / decode_s if decode_s and completion_tokens > 1 else None
        prefill_tps = n_prompt_tokens / ttft if ttft > 0 else None
        return {"status": "ok", "ttft_s": ttft, "total_s": total_s, "prefill_tps": prefill_tps, "decode_tps": decode_tps}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def build_prompt() -> str:
    target = round(CTX_SIZE * FILL_RATIO)
    return ctx_mod.build_filler(target, seed=42, count_fn=tokenize)


def get_gpu_mem(server_pid: int) -> dict:
    cmd = (
        "$out = @{}; try { "
        f"  $procSet = (Get-Counter -ListSet 'GPU Process Memory' -ErrorAction Stop).PathsWithInstances | Where-Object {{ $_ -match 'pid_{server_pid}_' }}; "
        "  if ($procSet) { $samples = (Get-Counter -Counter $procSet -ErrorAction Stop).CounterSamples; "
        "    foreach ($s in $samples) { if ($s.Path -match 'dedicated usage') { $out.dedicated = ($out.dedicated) + $s.CookedValue }; "
        "      if ($s.Path -match 'shared usage') { $out.shared = ($out.shared) + $s.CookedValue } } } "
        "} catch {}; $out | ConvertTo-Json"
    )
    out = ps(cmd, timeout=8)
    try:
        return json.loads(out) if out.strip() else {}
    except Exception:
        return {}


class DmonSampler(threading.Thread):
    """nvidia-smi dmon streams its own samples; we just tee its stdout to a
    file, and separately poll GPU Process Memory every ~1s into a second
    file (dmon doesn't report per-process VRAM usage)."""

    def __init__(self, server_pid: int, dmon_path: str, gpumem_path: str):
        super().__init__(daemon=True)
        self.server_pid = server_pid
        self.dmon_path = dmon_path
        self.gpumem_path = gpumem_path
        self._stop_event = threading.Event()
        self.dmon_proc = None

    def stop(self):
        self._stop_event.set()
        if self.dmon_proc is not None:
            try:
                self.dmon_proc.terminate()
            except Exception:
                pass

    def run(self):
        with open(self.dmon_path, "w", encoding="utf-8") as dmon_f:
            self.dmon_proc = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "pucvmt", "-d", "1"],
                stdout=dmon_f, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            )
            with open(self.gpumem_path, "w", encoding="utf-8") as memf:
                memf.write("ts_iso,gpu_dedicated_usage,gpu_shared_usage\n")
                while not self._stop_event.is_set():
                    gpu = get_gpu_mem(self.server_pid)
                    memf.write(f"{datetime.now(timezone.utc).isoformat()},{gpu.get('dedicated', '')},{gpu.get('shared', '')}\n")
                    memf.flush()
                    self._stop_event.wait(SAMPLE_INTERVAL_S)
            if self.dmon_proc.poll() is None:
                self.dmon_proc.terminate()


def start_corunner(co_type: str, thread_count: int, duration_s: float, target_gbps: float = 0.0) -> tuple[subprocess.Popen | None, str | None]:
    if co_type == "none":
        return None, None
    report_file = str(SCRIPT_DIR / f"m2_corunner_report_{time.time_ns()}.txt")
    if co_type == "memcpy":
        cmd = ["py", "-3.12", str(SCRIPT_DIR / "bandwidth_hog_throttled.py"),
               "--n-threads", str(thread_count), "--target-gbps", str(target_gbps),
               "--duration-s", str(duration_s), "--report-file", report_file, "--report-interval-s", "2"]
    elif co_type == "spin":
        cmd = ["py", "-3.12", str(SCRIPT_DIR / "spin_hog2.py"),
               "--n-threads", str(thread_count), "--duration-s", str(duration_s),
               "--report-file", report_file, "--report-interval-s", "2"]
    else:
        raise ValueError(co_type)
    proc = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    return proc, report_file


def read_report(path: str | None) -> float | None:
    if not path:
        return None
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return None


def run_condition(tag: str, threads_flag: int, co_type: str, thread_count: int,
                   target_gbps: float, all_rows: list, ts: str):
    log_line(f"=== {tag} (server -t {threads_flag}) ===")
    srv_log = str(SCRIPT_DIR / f"m2_srv_{tag}_{ts}.txt")
    server_proc = start_server(threads_flag, srv_log)
    if not wait_healthy(server_proc):
        log_line(f"  SERVER FAILED to start for {tag}")
        all_rows.append({"tag": tag, "server_started": False})
        return
    gpu0 = get_gpu_mem(server_proc.pid)
    log_line(f"  server up, pid={server_proc.pid}, initial GPU mem: {gpu0}")

    duration_s = 600.0
    corunner_proc, report_file = start_corunner(co_type, thread_count, duration_s, target_gbps)
    if corunner_proc is not None:
        time.sleep(5)

    dmon_path = str(SCRIPT_DIR / f"m2_dmon_{tag}_{ts}.txt")
    gpumem_path = str(SCRIPT_DIR / f"m2_gpumem_{tag}_{ts}.csv")
    sampler = DmonSampler(server_proc.pid, dmon_path, gpumem_path)
    sampler.start()

    prompt = build_prompt()
    n_tok = tokenize(prompt)

    achieved = None
    for i in range(N_CALLS):
        result = chat_throughput_call(prompt, n_tok)
        log_line(f"  [{tag}] call={i} ttft={result.get('ttft_s')} decode_tps={result.get('decode_tps')} status={result.get('status')}")
        if co_type == "memcpy":
            achieved = read_report(report_file)
        all_rows.append({
            "tag": tag, "server_threads_flag": threads_flag, "co_type": co_type,
            "co_thread_count": thread_count, "target_gbps": target_gbps,
            "achieved_gbps": achieved, "call": i, "n_prompt_tokens": n_tok,
            "dmon_file": Path(dmon_path).name, "gpumem_file": Path(gpumem_path).name,
            "gpu_mem_initial": gpu0, **result,
        })

    sampler.stop()
    sampler.join(timeout=10)
    if corunner_proc is not None:
        kill_pid(corunner_proc.pid)
        if report_file and Path(report_file).exists():
            try:
                Path(report_file).unlink()
            except OSError:
                pass
    kill_pid(server_proc.pid)
    time.sleep(3)


def main(smoke: bool = False):
    verify_model_sha256()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"blade_m2_host_interference_{ts}.jsonl"
    all_rows = []

    if smoke:
        run_condition("smoke_none", 4, "none", 0, 0.0, all_rows, ts)
        out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")
        log_line(f"SMOKE DONE -> {out_path.name}")
        return

    # A. baseline set at -t 4
    run_condition("A_none", 4, "none", 0, 0.0, all_rows, ts)
    run_condition("A_memcpy16", 4, "memcpy", 16, 0.0, all_rows, ts)
    run_condition("A_spin16", 4, "spin", 16, 0.0, all_rows, ts)
    out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")

    # B. thread-count discriminator (memcpy_16 only, server -t 1 and -t 8)
    run_condition("B_memcpy16_t1", 1, "memcpy", 16, 0.0, all_rows, ts)
    run_condition("B_memcpy16_t8", 8, "memcpy", 16, 0.0, all_rows, ts)
    out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")

    # C. dose-response (server -t 4, memcpy 16 threads throttled to targets)
    for target in DOSE_TARGETS_GBPS:
        run_condition(f"C_dose_{target}gbps", 4, "memcpy", 16, float(target), all_rows, ts)
        out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")

    log_line(f"DONE. {len(all_rows)} rows -> {out_path.name}")
    (RESULTS_DIR / f"{out_path.stem}.DONE").write_text("done\n")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
