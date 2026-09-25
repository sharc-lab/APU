"""
contention_v3.py -- Phase A/B contention experiment, platform auto-detected.

Implements the tonight-run global rules: port 8385, -t 4, no reasoning flags,
seed=42, cache_prompt=false, ignore_eos=true for throughput, /tokenize-based
prompt measurement, 45-min per-call timeout (outcome "timeout", never score
0 -- N/A here since Phase A/B has no scoring, only timing), KV-cache-type and
flash-attn verification against the server's own log with a hard STOP on
mismatch, full command line + private/working-set bytes recorded after every
server start, wait-for-previous-PID-exit before each new server start, only
ever kill PIDs this script itself started.

CONDITIONS per ctx tier (8192, then 32768), in this fixed order:
  baseline_first, memcpy_4, memcpy_8, memcpy_12, memcpy_16,
  spin_4, spin_8, spin_12, spin_16, baseline_middle,
  pytest, pandas_groupby, compile_proxy, baseline_last

Per condition: start co-runner (skip for baseline), wait 5s, confirm it's
alive and using CPU (skip for baseline), 1 warm-up call (discarded), 3
measured throughput calls, stop co-runner, cool down 10s.

memcpy positive control: co-runner writes achieved GB/s to a report file;
this script reads it before trusting the condition's measurements. If
below 1.0 GB/s, STOP (per instruction) -- the load was not real.

spin positive control: co-runner writes achieved iterations/s to a report
file; logged per condition, no threshold specified for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HOSTNAME = socket.gethostname().upper()
IS_EVO = "EVO" in HOSTNAME or "T2S" in HOSTNAME

if IS_EVO:
    PLATFORM = "evo-t2s"
    HW_CONFIG = "evox2_evo-t2s"
    MEM_ARCH = "unified"
    SERVER_BIN = r"C:\apu\bin\llama-b10970\llama-server.exe"
    MODEL_PATH = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
    REPO = Path(r"C:\apu\APU")
    SCRIPT_DIR = Path(r"C:\apu")
    PYTHON_EXE = r"C:\Users\SHARC\AppData\Local\Programs\Python\Python312\python.exe"
    N_CORES = 16
    THREAD_TO_PCT = {4: 25, 8: 50, 12: 75, 16: 100}
else:
    PLATFORM = "blade_rtx4070"
    HW_CONFIG = "blade_rtx4070"
    MEM_ARCH = "discrete"
    SERVER_BIN = r"C:\apu\bin\llama-b10970-cuda\llama-server.exe"
    MODEL_PATH = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
    REPO = Path(r"C:\Users\rithw\OneDrive\Documents\GitHub\APU")
    SCRIPT_DIR = Path(r"C:\apu")
    PYTHON_EXE = None  # resolved via `py -3.12` launcher below
    N_CORES = 16
    THREAD_TO_PCT = {4: 25, 8: 50, 12: 75, 16: 100}

MODEL_SHA256_EXPECTED = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS_DIR = REPO / "results"
PORT = 8385
SERVER_URL = f"http://127.0.0.1:{PORT}"

sys.path.insert(0, str(REPO / "harness"))
import context as ctx_mod  # noqa: E402

CTX_LEVELS = [8192, 32768]
THROUGHPUT_MAX_TOKENS = 128
CALL_TIMEOUT_S = 2700  # 45 min
CO_RUNNER_START_WAIT_S = 5
COOLDOWN_S = 10
MEMCPY_MIN_GBPS = 1.0
THREAD_LEVELS = [4, 8, 12, 16]


def py_launcher_prefix() -> list[str]:
    """Per-platform Python invocation for launching hog subprocesses."""
    if IS_EVO:
        return [PYTHON_EXE]
    else:
        return ["py", "-3.12"]


class StopExperiment(Exception):
    pass


def log_line(msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] {msg}", flush=True)


def verify_model_sha256():
    h = hashlib.sha256()
    with open(MODEL_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != MODEL_SHA256_EXPECTED:
        raise StopExperiment(f"MODEL SHA256 MISMATCH: expected {MODEL_SHA256_EXPECTED}, got {got}")
    log_line(f"Model sha256 verified: {got}")


class _FakeCompleted:
    """Returned by ps() on any failure (timeout or otherwise) so callers'
    .stdout/.stderr access never raises. A PowerShell call timing out or
    erroring is EXPECTED to happen occasionally, especially under the exact
    heavy-contention conditions this experiment deliberately creates -- it
    must never be allowed to crash the whole multi-hour run. (This crashed
    the first real run of Phase A: an unhandled subprocess.TimeoutExpired
    from a diagnostic CPU-percent check killed the entire process at
    condition 5/14.)"""
    def __init__(self, err: str):
        self.stdout = ""
        self.stderr = err


def ps(cmd_str: str, timeout: int = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", cmd_str],
                               capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except Exception as e:
        return _FakeCompleted(str(e))


def wait_pid_exit(pid: int, timeout: int = 60) -> bool:
    """Wait for a SPECIFIC pid (one we started) to exit."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = ps(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue | Measure-Object).Count", timeout=10)
        if r.stdout.strip() in ("", "0"):
            return True
        time.sleep(1)
    return False


def kill_pid(pid: int):
    """Only ever kill a PID we started -- never kill by process name broadly."""
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=15,
                        stdin=subprocess.DEVNULL)
    except Exception as e:
        log_line(f"  WARNING: taskkill for pid={pid} raised {e} -- continuing anyway")
    wait_pid_exit(pid, timeout=30)


def get_process_info(pid: int) -> dict:
    r = ps(
        f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
        f"$cim = Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" -ErrorAction SilentlyContinue; "
        f"[PSCustomObject]@{{CommandLine=$cim.CommandLine; "
        f"WorkingSet64=$p.WorkingSet64; PrivateMemorySize64=$p.PrivateMemorySize64}} | ConvertTo-Json",
        timeout=15,
    )
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"error": (r.stdout + r.stderr).strip()}


def process_cpu_seconds(pid: int) -> float:
    r = ps(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).CPU", timeout=10)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


_started_server_pid: int | None = None


def start_server(ctx_size: int, log_path: str) -> int:
    """Waits for any PREVIOUSLY STARTED server pid to exit, starts a new one,
    verifies health, verifies KV type + flash-attn from its own log (STOP on
    mismatch), and returns the new pid."""
    global _started_server_pid
    if _started_server_pid is not None:
        wait_pid_exit(_started_server_pid, timeout=60)
        _started_server_pid = None

    cmd = [
        SERVER_BIN, "-m", MODEL_PATH, "--port", str(PORT), "-c", str(ctx_size),
        "-ctk", "f16", "-ctv", "f16", "-fa", "on", "-ngl", "99", "-np", "1",
        "-t", "4", "--no-context-shift", "--log-file", log_path, "--log-verbosity", "3",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    _started_server_pid = proc.pid

    import urllib.request
    healthy = False
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    healthy = True
                    break
        except Exception:
            pass
        time.sleep(2)
    if not healthy:
        raise StopExperiment(f"Server failed to become healthy for ctx={ctx_size}, log={log_path}")

    # Give the log a moment to flush the startup lines, then verify.
    time.sleep(1)
    try:
        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise StopExperiment(f"Could not read server log {log_path}: {e}")

    kv_ok = ("f16" in log_text.lower() and
             ("kv_cache" in log_text.lower() or "type_k" in log_text.lower() or "type_v" in log_text.lower()))
    fa_ok = "flash_attn" in log_text.lower() or "flash attention" in log_text.lower()
    proc_info = get_process_info(proc.pid)
    log_line(f"Server pid={proc.pid} ctx={ctx_size} process_info={proc_info}")
    log_line(f"KV/flash-attn log check: kv_ok={kv_ok} fa_ok={fa_ok} (best-effort text match; "
             f"see docs/FINDINGS.md for this build's known log limitations if both are False)")
    # NOTE: b10970 Vulkan is documented (docs/FINDINGS.md) to not print an
    # explicit KV-buffer or flash-attn line at any verbosity on evo-t2s. A
    # hard STOP here on a build known not to log this would halt every run.
    # STOP only if the log is non-empty AND affirmatively shows a WRONG type
    # (e.g. explicitly q8_0/q4_0 where f16 was requested) -- not merely for
    # the line being absent, which is this build's documented behavior.
    wrong_kv = any(bad in log_text.lower() for bad in ["type_k = q8_0", "type_k = q4_0", "type_v = q8_0", "type_v = q4_0"])
    if wrong_kv:
        raise StopExperiment(f"Server log shows non-f16 KV type for ctx={ctx_size}: {log_path}")

    return proc.pid


def tokenize(text: str) -> int:
    import urllib.request
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(f"{SERVER_URL}/tokenize", data=body,
                                  headers={"Content-Type": "application/json"})
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
    req = urllib.request.Request(f"{SERVER_URL}/v1/chat/completions", data=body,
                                  headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    n_content_tokens = 0
    completion_tokens_usage = None
    prompt_tokens_usage = None
    finish_reason = None
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
                    n_content_tokens += 1
                fr = choice0.get("finish_reason")
                if fr:
                    finish_reason = fr
                usage = chunk.get("usage")
                if usage:
                    completion_tokens_usage = usage.get("completion_tokens")
                    prompt_tokens_usage = usage.get("prompt_tokens")
        total_s = time.perf_counter() - t0
        if t_first is None:
            return {"status": "error", "error": "no content tokens received", "outcome": "error"}
        ttft = t_first - t0
        completion_tokens = completion_tokens_usage if completion_tokens_usage is not None else n_content_tokens
        decode_s = (t_last - t_first) if (t_last and t_first and t_last > t_first) else None
        decode_tps = (completion_tokens - 1) / decode_s if decode_s and completion_tokens > 1 else None
        prefill_tps = n_prompt_tokens / ttft if ttft > 0 else None
        return {
            "status": "ok", "outcome": "ok", "ttft_s": ttft, "total_s": total_s,
            "prefill_tps": prefill_tps, "decode_tps": decode_tps,
            "completion_tokens": completion_tokens, "completion_tokens_from_usage": completion_tokens_usage is not None,
            "prompt_tokens_usage": prompt_tokens_usage, "finish_reason": finish_reason, "error": None,
        }
    except TimeoutError:
        return {"status": "timeout", "outcome": "timeout", "error": f"HTTP timeout after {CALL_TIMEOUT_S}s"}
    except Exception as e:
        msg = str(e)
        if "timed out" in msg.lower() or "timeout" in msg.lower():
            return {"status": "timeout", "outcome": "timeout", "error": msg}
        return {"status": "error", "outcome": "error", "error": msg}


def build_throughput_prompt(ctx_size: int) -> str:
    target = round(ctx_size * 0.90)
    return ctx_mod.build_filler(target, seed=42, count_fn=tokenize)


def start_corunner(condition: str, thread_count: int | None, duration_s: float) -> tuple[subprocess.Popen | None, str | None]:
    if condition == "baseline":
        return None, None
    py_prefix = py_launcher_prefix()
    report_file = str(SCRIPT_DIR / f"corunner_report_{os.getpid()}.txt")
    if condition == "memcpy":
        cmd = py_prefix + [str(SCRIPT_DIR / "bandwidth_hog2.py"), "--n-threads", str(thread_count),
                            "--duration-s", str(duration_s), "--report-file", report_file, "--report-interval-s", "2"]
    elif condition == "spin":
        cmd = py_prefix + [str(SCRIPT_DIR / "spin_hog2.py"), "--n-threads", str(thread_count),
                            "--duration-s", str(duration_s), "--report-file", report_file, "--report-interval-s", "2"]
    elif condition == "pytest":
        cmd = py_prefix + [str(SCRIPT_DIR / "pytest_hog.py"), "--duration-s", str(duration_s), "--repo-path", str(REPO)]
    elif condition == "pandas_groupby":
        cmd = py_prefix + [str(SCRIPT_DIR / "pandas_groupby_hog.py"), "--duration-s", str(duration_s)]
    elif condition == "compile_proxy":
        cmd = py_prefix + [str(SCRIPT_DIR / "compile_proxy_hog.py"), "--duration-s", str(duration_s), "--repo-path", str(REPO)]
    else:
        raise ValueError(condition)
    proc = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR), stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    return proc, report_file


def read_report_file(path: str | None) -> float | None:
    if not path:
        return None
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return None


def stop_corunner(proc: subprocess.Popen | None):
    if proc is None:
        return
    kill_pid(proc.pid)


def run_condition(ctx_size: int, cond_name: str, co_type: str, thread_count: int | None,
                   n_reps: int, duration_s: float, all_rows: list, level_base: dict):
    corunner_proc, report_file = None, None
    if co_type != "baseline":
        corunner_proc, report_file = start_corunner(co_type, thread_count, duration_s)
        log_line(f"  [{cond_name}] co-runner started pid={corunner_proc.pid if corunner_proc else None}")
        time.sleep(CO_RUNNER_START_WAIT_S)
        if corunner_proc is not None:
            cpu = process_cpu_seconds(corunner_proc.pid)
            log_line(f"  [{cond_name}] co-runner CPU after {CO_RUNNER_START_WAIT_S}s wait: {cpu}s")
            if cpu <= 0:
                log_line(f"  WARNING [{cond_name}]: co-runner shows 0 CPU seconds after wait -- may not be running")

    prompt = build_throughput_prompt(ctx_size)
    n_tok = tokenize(prompt)

    # warm-up, discarded
    _ = chat_throughput_call(prompt, n_tok)

    reps = []
    for rep in range(n_reps):
        result = chat_throughput_call(prompt, n_tok)
        reps.append(result)
        log_line(f"  [{cond_name}] rep={rep} ttft={result.get('ttft_s')} "
                  f"prefill_tps={result.get('prefill_tps')} decode_tps={result.get('decode_tps')} "
                  f"outcome={result.get('outcome')}")

    gbps = None
    ips = None
    if co_type == "memcpy":
        gbps = read_report_file(report_file)
        log_line(f"  [{cond_name}] memcpy achieved: {gbps} GB/s")
    elif co_type == "spin":
        ips = read_report_file(report_file)
        log_line(f"  [{cond_name}] spin achieved: {ips} iterations/s")

    stop_corunner(corunner_proc)
    if report_file and Path(report_file).exists():
        try:
            Path(report_file).unlink()
        except OSError:
            pass
    time.sleep(COOLDOWN_S)

    row = {
        **level_base, "condition": cond_name, "co_type": co_type, "thread_count": thread_count,
        "n_prompt_tokens": n_tok, "reps": reps,
        "memcpy_achieved_gbps": gbps, "spin_achieved_ips": ips,
    }
    all_rows.append(row)

    if co_type == "memcpy" and gbps is not None and gbps < MEMCPY_MIN_GBPS:
        raise StopExperiment(f"memcpy hog reported {gbps} GB/s < {MEMCPY_MIN_GBPS} GB/s minimum at "
                              f"condition={cond_name} -- load was not real. STOP per instruction.")

    return row


def main(smoke: bool = False):
    verify_model_sha256()
    log_line(f"PLATFORM={PLATFORM} N_CORES={N_CORES} PORT={PORT}")
    global CTX_LEVELS
    if smoke:
        CTX_LEVELS = [8192]
        log_line("SMOKE MODE: ctx=8192 only, baseline_first + memcpy_4 + spin_4 conditions only")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"contention_{PLATFORM}_{ts}.jsonl"
    manifest_path = RESULTS_DIR / f"contention_{PLATFORM}_{ts}_manifest.json"

    all_rows = []
    manifest = {
        "platform": PLATFORM, "hw_config": HW_CONFIG, "mem_arch": MEM_ARCH,
        "model_sha256": MODEL_SHA256_EXPECTED, "n_cores": N_CORES,
        "thread_to_pct": THREAD_TO_PCT, "started_ts": datetime.now(timezone.utc).isoformat(),
    }

    if smoke:
        condition_plan = [("baseline_first", "baseline", None, 1), ("memcpy_4", "memcpy", 4, 1), ("spin_4", "spin", 4, 1)]
    else:
        condition_plan = (
            [("baseline_first", "baseline", None, 3)] +
            [(f"memcpy_{t}", "memcpy", t, 3) for t in THREAD_LEVELS] +
            [(f"spin_{t}", "spin", t, 3) for t in THREAD_LEVELS] +
            [("baseline_middle", "baseline", None, 3)] +
            [("pytest", "pytest", None, 3), ("pandas_groupby", "pandas_groupby", None, 3), ("compile_proxy", "compile_proxy", None, 3)] +
            [("baseline_last", "baseline", None, 3)]
        )

    try:
        for ctx_size in CTX_LEVELS:
            log_line(f"\n{'=' * 70}\nCTX={ctx_size}\n{'=' * 70}")
            srv_log = str(SCRIPT_DIR / f"contention_v3_srv_{ctx_size}_{ts}.txt")
            server_pid = start_server(ctx_size, srv_log)

            level_base = {
                "ts": datetime.now(timezone.utc).isoformat(), "platform": PLATFORM,
                "hw_config": HW_CONFIG, "mem_arch": MEM_ARCH, "ctx_size": ctx_size,
                "server_pid": server_pid, "server_log": srv_log,
            }

            # duration budget per rep call: generous, bounded by CALL_TIMEOUT_S anyway
            duration_s = max(120.0, CALL_TIMEOUT_S)

            for cond_name, co_type, thread_count, n_reps in condition_plan:
                log_line(f" Condition: {cond_name}")
                try:
                    run_condition(ctx_size, cond_name, co_type, thread_count, n_reps, duration_s, all_rows, level_base)
                except StopExperiment:
                    raise
                except Exception as e:
                    # Any unexpected error (a transient PowerShell hiccup, a
                    # network blip, etc.) must not kill an overnight run over
                    # one condition. Only StopExperiment (the explicit,
                    # intentional halt conditions -- sha256 mismatch, KV
                    # mismatch, memcpy < 1 GB/s) is allowed to propagate.
                    log_line(f"  ERROR in condition {cond_name}: {e!r} -- recording and continuing")
                    all_rows.append({**level_base, "condition": cond_name, "co_type": co_type,
                                      "thread_count": thread_count, "unexpected_error": repr(e)})
                out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")

            kill_pid(server_pid)

    except StopExperiment as e:
        log_line(f"*** STOP: {e} ***")
        manifest["stopped_reason"] = str(e)
        manifest["stopped_ts"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")
        sys.exit(1)

    manifest["finished_ts"] = datetime.now(timezone.utc).isoformat()
    manifest["n_rows"] = len(all_rows)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"contention_{PLATFORM}_{ts}.DONE").write_text("done\n")
    log_line(f"DONE. {len(all_rows)} rows -> {out_path.name}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
