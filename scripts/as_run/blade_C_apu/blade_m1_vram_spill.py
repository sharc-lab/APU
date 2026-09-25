"""
blade_m1_vram_spill.py -- does the RTX 4070 Laptop silently spill KV/context
into system RAM ("shared GPU memory") as ctx grows, and does TTFT jump when
it does? Blade only, no co-runner.

PRIOR ART (searched before writing this): NVIDIA's "CUDA Sysmem Fallback
Policy" (driver 536.40+, June 2023) is well-documented to silently spill
VRAM overflow into system RAM instead of erroring, with a reported 5-10x
slowdown and the GPU showing high utilization but LOW power draw (stalled
on PCIe transfers, not computing). The exact fix is already known:
NVIDIA Control Panel -> Manage 3D Settings -> CUDA - Sysmem Fallback Policy
-> "Prefer No Sysmem Fallback". This experiment is REPLICATING a documented
effect on this specific model/context-length combination, not discovering
new territory -- worth confirming locally (does it happen here, at what
ctx, how severe) before flipping the setting and re-testing per instruction.

GRID: ctx in {8192, 16384, 24576, 32768, 40960}, f16, -fa on, no reasoning
flags, sha256-verified GGUF, no co-runner. Per ctx: 3 throughput calls
(max_tokens=128, ignore_eos=true, seed=42, cache_prompt=false), 60s cooldown
between calls.

TELEMETRY every 1s for the whole ctx tier (server start through last call +
cooldown): nvidia-smi --query-gpu=memory.used,memory.total,clocks.sm,
clocks.mem,power.draw,temperature.gpu,utilization.gpu,
clocks_throttle_reasons.active (CSV), plus Windows
\\GPU Process Memory(pid_<server>*)\\Dedicated Usage and Shared Usage.

DISCRIMINATING READ (spill vs. thermal): thermal throttling shows up as
clock or throttle-reason changes at a given ctx; VRAM spill shows up as
Shared Usage > 0 while clocks stay steady (the GPU isn't being throttled,
it's just waiting on PCIe transfers for spilled data, which nvidia-smi
utilization.gpu can still report as "busy" even while power draw drops --
per the documented Sysmem Fallback signature above).
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

CTX_LEVELS = [8192, 16384, 24576, 32768, 40960]
FILL_RATIO = 0.90
THROUGHPUT_MAX_TOKENS = 128
CALL_TIMEOUT_S = 600
COOLDOWN_S = 60
SAMPLE_INTERVAL_S = 1.0

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


def start_server(ctx_size: int, log_path: str) -> subprocess.Popen:
    cmd = [
        SERVER_BIN, "-m", MODEL_PATH, "--port", str(PORT), "-c", str(ctx_size),
        "-ctk", "f16", "-ctv", "f16", "-fa", "on", "-ngl", "99", "-np", "1",
        "--no-context-shift", "--log-file", log_path, "--log-verbosity", "3",
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
        return {"status": "ok", "ttft_s": ttft, "total_s": total_s, "prefill_tps": prefill_tps,
                "decode_tps": decode_tps, "completion_tokens": completion_tokens}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def build_prompt(ctx_size: int) -> str:
    target = round(ctx_size * FILL_RATIO)
    return ctx_mod.build_filler(target, seed=42, count_fn=tokenize)


class Sampler(threading.Thread):
    """1s-interval telemetry: nvidia-smi CSV fields + Windows GPU Process
    Memory (Dedicated/Shared Usage) for the server PID."""

    def __init__(self, server_pid: int, out_path: str):
        super().__init__(daemon=True)
        self.server_pid = server_pid
        self.out_path = out_path
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        with open(self.out_path, "w", encoding="utf-8") as f:
            f.write("ts_iso,memory_used_mib,memory_total_mib,clocks_sm_mhz,clocks_mem_mhz,"
                    "power_draw_w,temperature_gpu_c,utilization_gpu_pct,throttle_reasons_active,"
                    "gpu_dedicated_usage,gpu_shared_usage\n")
            while not self._stop_event.is_set():
                try:
                    r = subprocess.run(
                        ["nvidia-smi",
                         "--query-gpu=memory.used,memory.total,clocks.sm,clocks.mem,power.draw,"
                         "temperature.gpu,utilization.gpu,clocks_throttle_reasons.active",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL,
                    )
                    smi_line = r.stdout.strip()
                except Exception as e:
                    smi_line = f"ERROR:{e}"

                gpu_cmd = (
                    "$out = @{}; try { "
                    f"  $procSet = (Get-Counter -ListSet 'GPU Process Memory' -ErrorAction Stop).PathsWithInstances | Where-Object {{ $_ -match 'pid_{self.server_pid}_' }}; "
                    "  if ($procSet) { $samples = (Get-Counter -Counter $procSet -ErrorAction Stop).CounterSamples; "
                    "    foreach ($s in $samples) { if ($s.Path -match 'dedicated usage') { $out.dedicated = ($out.dedicated) + $s.CookedValue }; "
                    "      if ($s.Path -match 'shared usage') { $out.shared = ($out.shared) + $s.CookedValue } } } "
                    "} catch {}; $out | ConvertTo-Json"
                )
                gpu_out = ps(gpu_cmd, timeout=8)
                try:
                    gpu = json.loads(gpu_out) if gpu_out.strip() else {}
                except Exception:
                    gpu = {}

                f.write(f"{datetime.now(timezone.utc).isoformat()},{smi_line},"
                        f"{gpu.get('dedicated', '')},{gpu.get('shared', '')}\n")
                f.flush()
                self._stop_event.wait(SAMPLE_INTERVAL_S)


def main(smoke: bool = False):
    verify_model_sha256()
    levels = CTX_LEVELS
    if smoke:
        levels = [8192]
        log_line("SMOKE MODE: ctx=8192 only, 1 call")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"blade_m1_vram_spill_{ts}.jsonl"
    all_rows = []

    for ctx_size in levels:
        log_line(f"=== ctx={ctx_size} ===")
        srv_log = str(SCRIPT_DIR / f"m1_srv_{ctx_size}_{ts}.txt")
        server_proc = start_server(ctx_size, srv_log)
        if not wait_healthy(server_proc):
            log_line(f"  SERVER FAILED to start at ctx={ctx_size}")
            all_rows.append({"ctx_size": ctx_size, "server_started": False})
            continue

        sampler_path = str(SCRIPT_DIR / f"m1_telemetry_{ctx_size}_{ts}.csv")
        sampler = Sampler(server_proc.pid, sampler_path)
        sampler.start()

        prompt = build_prompt(ctx_size)
        n_tok = tokenize(prompt)
        log_line(f"  prompt: {n_tok} tokens")

        n_calls = 1 if smoke else 3
        for i in range(n_calls):
            result = chat_throughput_call(prompt, n_tok)
            log_line(f"  call={i} ttft={result.get('ttft_s')} decode_tps={result.get('decode_tps')} status={result.get('status')}")
            all_rows.append({"ctx_size": ctx_size, "call": i, "n_prompt_tokens": n_tok,
                              "telemetry_file": Path(sampler_path).name, **result})
            out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")
            if i < n_calls - 1:
                time.sleep(COOLDOWN_S)

        sampler.stop()
        sampler.join(timeout=5)
        kill_pid(server_proc.pid)
        time.sleep(5)

    log_line(f"DONE. {len(all_rows)} rows -> {out_path.name}")
    (RESULTS_DIR / f"{out_path.stem}.DONE").write_text("done\n")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
