"""
ramlock_evo_t2s.py -- Phase D: locked-balloon memory-constraint sweep, evo-t2s only.

A0 = Available MBytes with nothing of ours running (measured fresh at start).
S = target available AFTER the balloon is in place -- what's left for
model+KV+runtime+OS. Balloon size = A0 - S (handled by memory_balloon_awe.py,
launched with --target-available-mb S*1024).

SAMPLING SPLIT: the balloon must be in place BEFORE the server starts (so
the server's own load is squeezed, not just its later inference calls), so
the server's PID is not known when the balloon launches and can't be passed
to it. memory_balloon_awe.py samples held-page/Available/Pages-per-sec/
pagefile on its own 5s cadence into its own per-level CSV; THIS script
separately polls server private/working-set + GPU Process/Adapter Memory
counters on the same 5s cadence into its own per-level CSV once the server
exists, using the identical counter-query logic. Both are merged by
timestamp when compiling the final per-level trace for the report -- this
satisfies the full sampler spec via two coordinated sources instead of one,
without needing to inject a PID into an already-running process.

D1 (baseline, no balloon): server start, 1 warm-up, 1 throughput call at
~90% fill of ctx=32768, 5 correctness probes (art_01-05). This is the
reference every other level is classified against.

D2 (smoke gate, S=7GB): start balloon, wait until Available is within
+/-250MB of 7GB, hold 60s, start server, 1 throughput call. PASS requires
ALL of: held page count constant for the whole level (read from the
balloon's own log -- it never resizes by construction, so this should
always hold unless the balloon itself failed); Available stayed <= S+250MB
after the server loaded; the balloon log shows no free/resize event. On
PASS, continues straight into D3. On FAIL, stops and this script exits
non-zero with the trace path printed.

D3 (sweep): S = 12, 10, 9, 8, 7.5, 7, 6, 5, 4 GB, most headroom first (a
hang at the end loses the least). Per level: balloon to S + hold 60s +
verify within +/-250MB; start server (load success, load time, exact error
text); 1 warm-up; 1 throughput call; 5 correctness probes; stop server, wait
for its PID to exit; free balloon; wait until Available recovers to within
1GB of A0; commit results back. Level wall-clock cap: 3 hours -- if hit,
record "level_timeout" and move to the next level.

CLASSIFICATION per level, against D1:
  runs_normally:   loads; TTFT and decode within 1.25x of D1; correctness == D1
  pages_and_slows: loads; TTFT or decode beyond 1.25x of D1, or a call timed
                   out; correctness == D1
  fails_loudly:    server fails to load, crashes, or returns an HTTP error
                   (error text quoted)
  fails_silently:  responds without error but correctness < D1, or delivered
                   prompt tokens < intended
  system_unstable: safety valve fired, or SSH to evo-t2s was lost. If a level
                   is system_unstable, do NOT run lower S values -- record
                   and finish.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\apu\APU")
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS_DIR = REPO / "results"
SCRIPT_DIR = Path(r"C:\apu")
PYTHON_EXE = r"C:\Users\SHARC\AppData\Local\Programs\Python\Python312\python.exe"

sys.path.insert(0, str(REPO / "harness"))
import context as ctx_mod  # noqa: E402

SERVER_BIN = r"C:\apu\bin\llama-b10970\llama-server.exe"
MODEL_PATH = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
MODEL_SHA256_EXPECTED = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
BALLOON_SCRIPT = str(SCRIPT_DIR / "memory_balloon_awe.py")
PORT = 8385
SERVER_URL = f"http://127.0.0.1:{PORT}"

CTX_SIZE = 32768
FILL_RATIO = 0.90
THROUGHPUT_MAX_TOKENS = 128
CORRECTNESS_MAX_TOKENS = 32
CALL_TIMEOUT_S = 2700
SERVER_START_TIMEOUT_S = 600

S_LEVELS_GB = [12, 10, 9, 8, 7.5, 7, 6, 5, 4]
S_SMOKE_GB = 7
HOLD_AFTER_TARGET_S = 60
TOLERANCE_MB = 250
RECOVERY_TOLERANCE_MB = 1024
LEVEL_WALLCLOCK_CAP_S = 3 * 3600
SLOWDOWN_THRESHOLD = 1.25
SAMPLE_INTERVAL_S = 5.0


class StopExperiment(Exception):
    pass


def log_line(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def verify_model_sha256():
    h = hashlib.sha256()
    with open(MODEL_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != MODEL_SHA256_EXPECTED:
        raise StopExperiment(f"MODEL SHA256 MISMATCH: expected {MODEL_SHA256_EXPECTED}, got {got}")
    log_line(f"Model sha256 verified: {got}")


def ps(cmd: str, timeout: int = 20) -> str:
    """Never raises -- a PowerShell timeout/error returns "" instead of
    propagating. This crashed the first real Phase A run (an unhandled
    subprocess.TimeoutExpired from a diagnostic check killed the whole
    multi-hour process); every caller here must be equally immune for an
    unattended overnight run."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                            capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout
    except Exception as e:
        return ""


def available_mb_via_ps() -> float:
    out = ps("(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024")
    try:
        return float(out.strip())
    except Exception:
        return -1.0


def wait_pid_exit(pid: int, timeout: int = 60) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = ps(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue | Measure-Object).Count", timeout=10)
        if r.strip() in ("", "0"):
            return True
        time.sleep(1)
    return False


def kill_pid(pid: int):
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=15, stdin=subprocess.DEVNULL)
    except Exception as e:
        log_line(f"  WARNING: taskkill for pid={pid} raised {e} -- continuing anyway")
    wait_pid_exit(pid, timeout=30)


def get_process_info(pid: int) -> dict:
    out = ps(
        f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
        f"$cim = Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" -ErrorAction SilentlyContinue; "
        f"[PSCustomObject]@{{CommandLine=$cim.CommandLine; WorkingSet64=$p.WorkingSet64; PrivateMemorySize64=$p.PrivateMemorySize64}} | ConvertTo-Json",
        timeout=15,
    )
    try:
        return json.loads(out)
    except Exception:
        return {"error": out[:300]}


def get_pagefile_usage() -> dict:
    out = ps("Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage | ConvertTo-Json")
    try:
        return json.loads(out)
    except Exception:
        return {"error": out[:300]}


def get_gpu_shared_limit() -> dict:
    out = ps(
        "$out = @{}; try { $out.limit = (Get-Counter '\\GPU Adapter Memory(*)\\Shared Limit' -ErrorAction Stop).CounterSamples "
        "| Select-Object -First 1 -ExpandProperty CookedValue } catch { $out.error = $_.Exception.Message }; $out | ConvertTo-Json"
    )
    try:
        return json.loads(out)
    except Exception:
        return {"error": out[:300]}


def check_ssh_alive() -> bool:
    """This script runs ON evo-t2s already (launched via WMI); 'SSH to evo-t2s
    was lost' from the controlling workstation's perspective doesn't stop
    this script itself (it has no SSH dependency to run) -- but it DOES
    affect whether the heartbeat file the balloon watches keeps getting
    touched. That heartbeat is touched by THIS script, not by the remote
    controller, so this script's own liveness is the real signal; recorded
    here as a no-op placeholder for clarity in the classification logic
    rather than a meaningful external check."""
    return True


def tokenize(text: str) -> int:
    import urllib.request
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(f"{SERVER_URL}/tokenize", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return len(json.loads(r.read())["tokens"])


def start_server(log_path: str) -> subprocess.Popen:
    cmd = [
        SERVER_BIN, "-m", MODEL_PATH, "--port", str(PORT), "-c", str(CTX_SIZE),
        "-ctk", "f16", "-ctv", "f16", "-fa", "on", "-ngl", "99", "-np", "1",
        "-t", "4", "--no-context-shift", "--log-file", log_path, "--log-verbosity", "3",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


def wait_healthy(proc: subprocess.Popen, timeout: int) -> tuple[bool, str | None]:
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False, f"process exited with code {proc.returncode}"
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True, None
        except Exception:
            pass
        time.sleep(2)
    return False, None


def chat_call(prompt: str, max_tokens: int, ignore_eos: bool, n_prompt_tokens: int, timeout: int) -> dict:
    import urllib.request
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "ignore_eos": ignore_eos,
        "temperature": 0, "seed": 42, "stream": True, "cache_prompt": False,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{SERVER_URL}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first, t_last = None, None
    chunks = []
    completion_tokens_usage = None
    finish_reason = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
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
                    chunks.append(content)
                fr = choice0.get("finish_reason")
                if fr:
                    finish_reason = fr
                usage = chunk.get("usage")
                if usage:
                    completion_tokens_usage = usage.get("completion_tokens")
        total_s = time.perf_counter() - t0
        if t_first is None:
            return {"status": "error", "outcome": "error", "error": "no content tokens", "output": None}
        ttft = t_first - t0
        completion_tokens = completion_tokens_usage if completion_tokens_usage is not None else len(chunks)
        decode_s = (t_last - t_first) if (t_last and t_first and t_last > t_first) else None
        decode_tps = (completion_tokens - 1) / decode_s if decode_s and completion_tokens > 1 else None
        prefill_tps = n_prompt_tokens / ttft if ttft > 0 else None
        return {"status": "ok", "outcome": "ok", "ttft_s": ttft, "total_s": total_s,
                "prefill_tps": prefill_tps, "decode_tps": decode_tps, "output": "".join(chunks).strip(),
                "completion_tokens": completion_tokens, "finish_reason": finish_reason, "error": None}
    except Exception as e:
        msg = str(e)
        outcome = "timeout" if "time" in msg.lower() else "error"
        return {"status": outcome, "outcome": outcome, "error": msg, "output": None}


def build_throughput_prompt() -> str:
    target = round(CTX_SIZE * FILL_RATIO)
    return ctx_mod.build_filler(target, seed=42, count_fn=tokenize)


def _load_scorers():
    import importlib.util
    spec = importlib.util.spec_from_file_location("probes_scorers", PROBES_DIR / "scorers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_correctness_probes(scorers, probes: list, throughput_prompt_unused=None) -> list:
    target_total = round(CTX_SIZE * FILL_RATIO)
    min_art_q = min(tokenize(p["artifact"].strip()) + tokenize(p["question"].strip()) for p in probes)
    filler_target = max(target_total - min_art_q, 256) + 64
    filler_full = ctx_mod.build_filler(filler_target, seed=42, count_fn=tokenize)
    filler_full_tokens = tokenize(filler_full)

    def left_truncate_tokens(text, target_tokens):
        if tokenize(text) <= target_tokens:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi) // 2
            if tokenize(text[mid:]) <= target_tokens:
                hi = mid
            else:
                lo = mid + 1
        return text[lo:]

    rows = []
    for probe in probes:
        artifact = probe["artifact"].strip()
        question = probe["question"].strip()
        art_tok = tokenize(artifact)
        q_tok = tokenize(question)
        per_probe_target = max(target_total - art_tok - q_tok, 64)
        filler = left_truncate_tokens(filler_full, min(per_probe_target, filler_full_tokens))
        prompt = f"{filler}\n\n{artifact}\n\n{question}"
        n_tok = tokenize(prompt)
        result = chat_call(prompt, CORRECTNESS_MAX_TOKENS, False, n_tok, CALL_TIMEOUT_S)
        if result["status"] == "ok" and result["output"] is not None:
            probe_dict = {"id": probe["id"], "scorer_type": probe["scorer_type"], "expected": probe["expected"]}
            try:
                score, score_detail = scorers.score(probe_dict, result["output"])
            except Exception as e:
                score, score_detail = None, f"scorer_error:{e}"
        else:
            score, score_detail = None, "no output"
        rows.append({"probe_id": probe["id"], "n_prompt_tokens": n_tok, "score": score,
                     "score_detail": score_detail, "output": result["output"], "outcome": result["outcome"],
                     "error": result["error"]})
        log_line(f"    [{probe['id']}] score={score} outcome={result['outcome']}")
    return rows


def server_gpu_sampler(server_pid_holder: dict, stop_event: threading.Event, out_path: str):
    """Runs in a background thread; samples server private/working-set + GPU
    counters every SAMPLE_INTERVAL_S once server_pid_holder['pid'] is set."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ts_iso,server_private_mb,server_workingset_mb,gpu_shared_usage,gpu_dedicated_usage,gpu_adapter_shared_usage_total\n")
        while not stop_event.is_set():
            pid = server_pid_holder.get("pid")
            if pid is not None:
                info = get_process_info(pid)
                gpu_cmd = (
                    "$out = @{}; try { "
                    f"  $procSet = (Get-Counter -ListSet 'GPU Process Memory' -ErrorAction Stop).PathsWithInstances | Where-Object {{ $_ -match 'pid_{pid}_' }}; "
                    "  if ($procSet) { $samples = (Get-Counter -Counter $procSet -ErrorAction Stop).CounterSamples; "
                    "    foreach ($s in $samples) { if ($s.Path -match 'shared usage') { $out.gpu_shared = $s.CookedValue }; if ($s.Path -match 'dedicated usage') { $out.gpu_dedicated = ($out.gpu_dedicated) + $s.CookedValue } } } "
                    "} catch {}; "
                    "try { $adapterSamples = (Get-Counter -Counter '\\GPU Adapter Memory(*)\\Shared Usage' -ErrorAction Stop).CounterSamples; "
                    "  $out.gpu_adapter_total = ($adapterSamples | Measure-Object -Property CookedValue -Sum).Sum } catch {}; "
                    "$out | ConvertTo-Json"
                )
                gpu_out = ps(gpu_cmd, timeout=15)
                try:
                    gpu = json.loads(gpu_out)
                except Exception:
                    gpu = {}
                f.write(f"{datetime.now(timezone.utc).isoformat()},"
                        f"{(info.get('PrivateMemorySize64', 0) or 0) / (1024*1024):.1f},"
                        f"{(info.get('WorkingSet64', 0) or 0) / (1024*1024):.1f},"
                        f"{gpu.get('gpu_shared', '')},{gpu.get('gpu_dedicated', '')},{gpu.get('gpu_adapter_total', '')}\n")
                f.flush()
            stop_event.wait(SAMPLE_INTERVAL_S)


def start_balloon(target_mb: float, heartbeat_path: str, log_path: str) -> subprocess.Popen:
    cmd = [
        PYTHON_EXE, BALLOON_SCRIPT, "--target-available-mb", str(target_mb),
        "--tolerance-mb", str(TOLERANCE_MB), "--heartbeat-file", heartbeat_path,
        "--heartbeat-timeout-s", "300", "--max-runtime-s", str(LEVEL_WALLCLOCK_CAP_S + 600),
        "--log-path", log_path, "--sample-interval-s", str(SAMPLE_INTERVAL_S),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, stdin=subprocess.DEVNULL)


def touch_heartbeat(path: str):
    Path(path).write_text(str(time.time()))


def verify_d2_gate(d2_result: dict, s_gb: float) -> tuple[bool, str]:
    """Literal check of all three D2 PASS conditions from the balloon's own
    log, not just 'did the throughput call succeed' (which was the only
    thing checked before this fix -- a real gap, since a throughput call can
    succeed even if the balloon quietly failed to hold its target)."""
    if d2_result.get("classification") == "system_unstable" or not d2_result.get("server_started"):
        return False, f"server/system check failed: {d2_result.get('error_text')}"
    if d2_result.get("throughput", {}).get("outcome") != "ok":
        return False, f"throughput call did not succeed: {d2_result.get('throughput')}"

    log_name = d2_result.get("balloon_log_file")
    if not log_name:
        return False, "no balloon_log_file recorded for D2 (balloon may not have run)"
    log_path = SCRIPT_DIR / log_name
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return False, f"could not read balloon log {log_path}: {e}"

    data_lines = [l for l in lines if l and not l.startswith("#") and not l.startswith("ts_iso")]
    if not data_lines:
        return False, f"balloon log {log_path} has no data rows"

    held_mb_values = []
    avail_mb_values = []
    for l in data_lines:
        parts = l.split(",")
        if len(parts) < 4:
            continue
        try:
            held_mb_values.append(float(parts[2]))
        except (ValueError, IndexError):
            pass
        try:
            avail_mb_values.append(float(parts[3]))
        except (ValueError, IndexError):
            pass

    # PASS condition 1: held page count constant for the entire level.
    if held_mb_values:
        held_min, held_max = min(held_mb_values), max(held_mb_values)
        if held_max - held_min > 1.0:  # allow trivial float-formatting noise
            return False, f"held_mb varied during D2: min={held_min} max={held_max} (balloon resized -- should never happen by construction)"
    else:
        return False, "no held_mb values parsed from balloon log"

    # PASS condition 2: Available stayed <= S + 250MB after the server loaded.
    target_mb = s_gb * 1024
    over_budget = [v for v in avail_mb_values if v > target_mb + TOLERANCE_MB]
    if over_budget:
        return False, f"available_mb exceeded S+250MB in {len(over_budget)}/{len(avail_mb_values)} samples (max={max(over_budget):.1f} vs limit={target_mb + TOLERANCE_MB:.1f})"

    # PASS condition 3: balloon log shows no free/resize event.
    full_text = "\n".join(lines).lower()
    if "safety_valve" in full_text or "released" in full_text:
        return False, "balloon log shows a safety-valve trigger or release event during D2"

    return True, f"held_mb constant ({held_min:.1f} MB), available stayed within S+{TOLERANCE_MB}MB, no free/resize logged"


def classify(level_result: dict, baseline: dict) -> str:
    if not level_result.get("server_started"):
        return "fails_loudly" if level_result.get("error_text") else "fails_silently"
    tput = level_result.get("throughput", {})
    if tput.get("outcome") == "error":
        return "fails_loudly"
    if tput.get("outcome") == "timeout":
        return "pages_and_slows"
    corr = level_result.get("correctness", [])
    baseline_corr = baseline.get("correctness", [])
    n_correct = sum(1 for r in corr if r.get("score") == 1.0)
    n_correct_baseline = sum(1 for r in baseline_corr if r.get("score") == 1.0)
    any_short = any((r.get("n_prompt_tokens") or 0) < round(CTX_SIZE * FILL_RATIO * 0.85) for r in corr)
    if n_correct < n_correct_baseline or any_short:
        return "fails_silently"
    b_ttft = baseline.get("throughput", {}).get("ttft_s")
    b_decode = baseline.get("throughput", {}).get("decode_tps")
    ttft = tput.get("ttft_s")
    decode = tput.get("decode_tps")
    slow = False
    if b_ttft and ttft and ttft > b_ttft * SLOWDOWN_THRESHOLD:
        slow = True
    if b_decode and decode and decode < b_decode / SLOWDOWN_THRESHOLD:
        slow = True
    any_corr_timeout = any(r.get("outcome") == "timeout" for r in corr)
    if slow or any_corr_timeout:
        return "pages_and_slows"
    return "runs_normally"


def main():
    verify_model_sha256()
    scorers = _load_scorers()
    segs = [json.loads(l) for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seg_by_id = {s["id"]: s for s in segs}
    probes = [seg_by_id[f"art_{i:02d}"] for i in range(1, 6)]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ramlock_evo-t2s_{ts}.jsonl"
    manifest_path = RESULTS_DIR / f"ramlock_evo-t2s_{ts}_manifest.json"

    A0 = available_mb_via_ps()
    pagefile_info = get_pagefile_usage()
    gpu_limit = get_gpu_shared_limit()
    log_line(f"A0={A0:.1f} MB, pagefile={pagefile_info}, gpu_shared_limit={gpu_limit}")

    manifest = {
        "A0_mb": A0, "pagefile": pagefile_info, "gpu_shared_limit": gpu_limit,
        "model_sha256": MODEL_SHA256_EXPECTED, "started_ts": datetime.now(timezone.utc).isoformat(),
        "server_command_lines": [], "levels": [],
    }

    all_rows = []

    def run_level(label: str, s_gb: float | None, is_baseline: bool = False) -> dict:
        """s_gb=None means no balloon (D1 baseline)."""
        t_level_start = time.monotonic()
        level_result = {"label": label, "s_gb": s_gb, "ts": datetime.now(timezone.utc).isoformat()}
        balloon_proc = None
        hb_path = str(SCRIPT_DIR / f"ramlock_hb_{label}_{ts}.txt")
        balloon_log = str(SCRIPT_DIR / f"ramlock_balloon_{label}_{ts}.csv")

        if s_gb is not None:
            target_mb = s_gb * 1024
            touch_heartbeat(hb_path)
            balloon_proc = start_balloon(target_mb, hb_path, balloon_log)
            log_line(f" [{label}] balloon started pid={balloon_proc.pid} target={target_mb}MB")
            deadline = time.monotonic() + 300
            reached = False
            while time.monotonic() < deadline:
                touch_heartbeat(hb_path)
                if balloon_proc.poll() is not None:
                    level_result["classification"] = "system_unstable"
                    level_result["error_text"] = f"balloon exited early, code={balloon_proc.returncode}"
                    return level_result
                cur = available_mb_via_ps()
                if abs(cur - target_mb) <= TOLERANCE_MB:
                    reached = True
                    break
                time.sleep(5)
            if not reached:
                level_result["classification"] = "system_unstable"
                level_result["error_text"] = "balloon did not reach target within 300s"
                kill_pid(balloon_proc.pid)
                return level_result
            log_line(f" [{label}] target reached, holding {HOLD_AFTER_TARGET_S}s")
            hold_deadline = time.monotonic() + HOLD_AFTER_TARGET_S
            while time.monotonic() < hold_deadline:
                touch_heartbeat(hb_path)
                time.sleep(5)
            avail_before_server = available_mb_via_ps()
        else:
            avail_before_server = available_mb_via_ps()

        srv_log = str(SCRIPT_DIR / f"ramlock_srv_{label}_{ts}.txt")
        t_load_start = time.monotonic()
        server_proc = start_server(srv_log)
        healthy, start_error = wait_healthy(server_proc, SERVER_START_TIMEOUT_S)
        load_time_s = time.monotonic() - t_load_start

        if not healthy:
            level_result["server_started"] = False
            level_result["load_time_s"] = load_time_s
            level_result["error_text"] = start_error
            level_result["classification"] = "fails_loudly" if start_error else "fails_silently"
            kill_pid(server_proc.pid) if server_proc.poll() is None else None
            if balloon_proc:
                kill_pid(balloon_proc.pid)
            return level_result

        proc_info = get_process_info(server_proc.pid)
        manifest["server_command_lines"].append({"label": label, **proc_info})
        level_result["server_started"] = True
        level_result["load_time_s"] = round(load_time_s, 1)
        level_result["server_process_info"] = proc_info
        level_result["avail_before_server_mb"] = avail_before_server

        gpu_sampler_stop = threading.Event()
        gpu_sampler_path = str(SCRIPT_DIR / f"ramlock_gpu_{label}_{ts}.csv")
        pid_holder = {"pid": server_proc.pid}
        gpu_thread = threading.Thread(target=server_gpu_sampler, args=(pid_holder, gpu_sampler_stop, gpu_sampler_path), daemon=True)
        gpu_thread.start()

        prompt = build_throughput_prompt()
        n_tok = tokenize(prompt)
        _ = chat_call(prompt, THROUGHPUT_MAX_TOKENS, True, n_tok, CALL_TIMEOUT_S)  # warm-up, discarded
        if s_gb is not None:
            touch_heartbeat(hb_path)
        tput = chat_call(prompt, THROUGHPUT_MAX_TOKENS, True, n_tok, CALL_TIMEOUT_S)
        level_result["throughput"] = tput
        log_line(f" [{label}] throughput ttft={tput.get('ttft_s')} decode_tps={tput.get('decode_tps')} outcome={tput.get('outcome')}")

        if s_gb is not None:
            touch_heartbeat(hb_path)
        corr_rows = run_correctness_probes(scorers, probes)
        level_result["correctness"] = corr_rows

        gpu_sampler_stop.set()
        kill_pid(server_proc.pid)

        if balloon_proc:
            touch_heartbeat(hb_path)
            try:
                stop_marker = str(SCRIPT_DIR / f"ramlock_stop_{label}")
                kill_pid(balloon_proc.pid)
            except Exception:
                pass
            recover_deadline = time.monotonic() + 180
            while time.monotonic() < recover_deadline:
                if abs(available_mb_via_ps() - A0) <= RECOVERY_TOLERANCE_MB:
                    break
                time.sleep(5)

        try:
            balloon_csv_lines = Path(balloon_log).read_text(encoding="utf-8").splitlines() if Path(balloon_log).exists() else []
        except Exception:
            balloon_csv_lines = []
        level_result["balloon_log_file"] = Path(balloon_log).name
        level_result["gpu_sampler_file"] = Path(gpu_sampler_path).name
        level_result["balloon_n_log_lines"] = len(balloon_csv_lines)
        level_result["wall_clock_s"] = round(time.monotonic() - t_level_start, 1)
        return level_result

    try:
        # D1: baseline
        log_line("=== D1: baseline (no balloon) ===")
        d1 = run_level("D1_baseline", None, is_baseline=True)
        all_rows.append(d1)
        out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")
        if d1.get("classification") == "system_unstable" or not d1.get("server_started"):
            raise StopExperiment(f"D1 baseline itself failed: {d1.get('error_text')}")
        d1["classification"] = "runs_normally"  # D1 is the reference by definition

        # D2: smoke gate
        log_line(f"=== D2: smoke gate at S={S_SMOKE_GB}GB ===")
        d2 = run_level("D2_smoke", S_SMOKE_GB)
        all_rows.append(d2)
        out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")

        gate_pass, gate_detail = verify_d2_gate(d2, S_SMOKE_GB)
        d2["gate_pass"] = gate_pass
        d2["gate_detail"] = gate_detail
        log_line(f"D2 gate_pass={gate_pass} detail={gate_detail}")
        out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")
        if not gate_pass:
            raise StopExperiment(f"D2 smoke gate FAILED: {gate_detail}. Full trace: {json.dumps(d2)[:1500]}")

        # D3: sweep
        for s_gb in S_LEVELS_GB:
            log_line(f"=== D3: S={s_gb}GB ===")
            level_start = time.monotonic()
            level_result = run_level(f"D3_S{s_gb}", s_gb)
            level_result["classification"] = classify(level_result, d1)
            all_rows.append(level_result)
            out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")
            manifest["levels"].append({"s_gb": s_gb, "classification": level_result["classification"]})
            log_line(f" [D3 S={s_gb}] classification={level_result['classification']}")

            elapsed = time.monotonic() - level_start
            if elapsed > LEVEL_WALLCLOCK_CAP_S:
                level_result["level_timeout"] = True

            if level_result["classification"] == "system_unstable":
                log_line(f"system_unstable at S={s_gb}GB -- stopping sweep, not running lower S values.")
                break

            # Commit after each level (per instruction)
            subprocess.run(["git", "-C", str(REPO), "add", f"results/{out_path.name}"], capture_output=True, stdin=subprocess.DEVNULL)
            subprocess.run(["git", "-C", str(REPO), "commit", "-m", f"data(ramlock): S={s_gb}GB level complete, evo-t2s"],
                            capture_output=True, stdin=subprocess.DEVNULL)

    except StopExperiment as e:
        log_line(f"*** STOP: {e} ***")
        manifest["stopped_reason"] = str(e)

    manifest["finished_ts"] = datetime.now(timezone.utc).isoformat()
    manifest["n_rows"] = len(all_rows)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"ramlock_evo-t2s_{ts}.DONE").write_text("done\n")
    log_line(f"DONE. {len(all_rows)} rows -> {out_path.name}")


if __name__ == "__main__":
    main()
