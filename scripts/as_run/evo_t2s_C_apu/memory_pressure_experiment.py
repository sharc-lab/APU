"""
memory_pressure_experiment.py v2 -- real memory-constraint sweep, evo-t2s.

v1 used a moving-target balloon (held Available MBytes constant, which meant
it retreated whenever llama-server allocated -- the server was NEVER actually
constrained at any of its 7 levels; see docs/RESULT_PROVENANCE.md for the
full diagnosis of that invalid run, results/memory_pressure_20260924T025611Z.jsonl).

v2 uses memory_balloon.py v2: a FIXED-size allocation computed once as
(total_physical_mb - X) before the server starts, touched at allocation and
re-touched every tick, NEVER resized for the rest of the level. The server
starts into whatever's actually left -- it is squeezed by construction, not
by a control loop that can give memory back.

CONFIG (fixed for the whole run): qwen3-4b-instruct-85e4a5b7.gguf (sha256
verified), llama-server b10970 Vulkan, -fa on, ctx=32768, f16 KV (~7.6 GB
footprint: 2991 MiB model weight + 4516.56 MiB KV, per docs/KV_MEASUREMENT.md
and the bw_saturation header table).

LEVELS (X = MB left for model+KV+runtime+OS, descending): 12, 9, 8, 7.5, 7,
6, 5 GB, plus a final "OS floor" level at X=2.5 GB (=BALLOON_FLOOR_MB) as the
deliberately extreme end of the sweep -- expected to fail, characterizing
the floor itself rather than being a safety margin this time.

PER LEVEL:
  1. Start the fixed balloon at this X. Wait a few ticks, confirm from its
     own log that balloon_size_mb_fixed is holding constant (it always will
     by construction -- this is now a sanity check on the mechanism, not a
     control decision).
  2. Start llama-server fresh into whatever's left. Timeout 600s.
  3. Record: server command line + WorkingSet64/PrivateMemorySize64 (single-
     process query, never system-wide free memory), pagefile usage
     (Win32_PageFileUsage), the balloon's own log (committed/working-set
     positive control + \Memory\Pages/sec hard-fault signal).
  4. If server fails to start: exact stderr/exit code (fails_loudly) or a
     hang with nothing (fails_silently).
  5. If it starts: ONE 90%-fill ctx=32768 throughput call (TTFT, prefill,
     decode), then the 10 art_01-10 probes AT the same ~90%-fill ctx=32768
     construction (ADJACENT position: filler then artifact then question,
     filler sized against the smallest artifact+question pairing so
     per-probe trimming is always valid -- the kv_quality_sweep.py under-fill
     bug is not repeated here), cache_prompt=false throughout.
  6. Kill server, then balloon, in that order.

CLASSIFICATION (post-hoc from logged data, not decided live):
  - runs_normally / pages_and_slows / fails_loudly / fails_silently -- same
    four categories as v1, same caveat about Page Faults/sec being a noisy
    activity indicator versus Pages/sec (hard faults, now also logged by
    memory_balloon.py v2) being the precise paging-activity signal.

PAGEFILE: system-managed, 4096 MB base size at time of writing (confirmed via
Win32_PageFileUsage / Win32_ComputerSystem.AutomaticManagedPagefile=True).
Running with this as-is for this sweep (not disabled, not fixed-size) --
recorded per level in case Windows grows it under real pressure during the
run, which would itself be informative.

LOCKING: memory_balloon.py attempts VirtualLock; whether it succeeds is
recorded verbatim from the balloon's own reported LOCK STATUS line per
level (not assumed). As of this design, the sharc account does not hold
SeLockMemoryPrivilege (confirmed by direct test), so runs will show UNLOCKED
until that is granted -- see the exact grant procedure in memory_balloon.py's
try_enable_lock_privilege() error message, surfaced verbatim in this script's
own output.

CALL COUNT per level: 1 throughput + 10 correctness = 11 calls. 8 levels =
88 calls total, 8 server starts, 8 balloon holds.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"C:\apu\APU")
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS_DIR = REPO / "results"

sys.path.insert(0, str(REPO / "harness"))
import context as ctx_mod  # noqa: E402

SERVER_BIN = r"C:\apu\bin\llama-b10970\llama-server.exe"
MODEL_PATH = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
MODEL_SHA256_EXPECTED = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
PORT = 8384
SERVER_URL = f"http://127.0.0.1:{PORT}"
BALLOON_SCRIPT = r"C:\apu\memory_balloon.py"

CTX_SIZE = 32768
KV_PREC = "f16"
FILL_RATIO = 0.90
DECODE_TOKENS = 64
CORRECTNESS_MAX_TOKENS = 32

LEVELS_X_GB = [12, 9, 8, 7.5, 7, 6, 5, 2.5]  # last entry is the deliberate "OS floor" level

SERVER_START_TIMEOUT_S = 600
THROUGHPUT_CALL_TIMEOUT_S = 600
CORRECTNESS_CALL_TIMEOUT_S = 180
BALLOON_STABILIZE_TICKS = 2
BALLOON_LOG_INTERVAL_S = 5
BALLOON_FLOOR_MB = 2560
BALLOON_MAX_RUNTIME_S = 3600
HEARTBEAT_TIMEOUT_S = 90

BASELINE_THROUGHPUT_TTFT_S = 277.44
THROUGHPUT_SLOWDOWN_THRESHOLD = 2.0
SERVER_START_NORMAL_THRESHOLD_S = 60


def verify_model_sha256():
    import hashlib
    h = hashlib.sha256()
    with open(MODEL_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != MODEL_SHA256_EXPECTED:
        raise RuntimeError(f"MODEL SHA256 MISMATCH: expected {MODEL_SHA256_EXPECTED}, got {got}")
    print(f"Model sha256 verified: {got}")


def kill_by_name(name: str):
    subprocess.run(["powershell", "-NoProfile", "-Command",
                     f"Get-Process -Name {name} -ErrorAction SilentlyContinue | Stop-Process -Force"],
                    capture_output=True, timeout=15)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                             f"(Get-Process -Name {name} -ErrorAction SilentlyContinue | Measure-Object).Count"],
                            capture_output=True, text=True, timeout=10)
        if r.stdout.strip() in ("", "0"):
            return
        time.sleep(1)


def get_server_process_info(pid: int) -> dict:
    ps_cmd = (
        f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
        f"$cim = Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" -ErrorAction SilentlyContinue; "
        f"[PSCustomObject]@{{CommandLine=$cim.CommandLine; "
        f"WorkingSet64=$p.WorkingSet64; PrivateMemorySize64=$p.PrivateMemorySize64}} | ConvertTo-Json"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                        capture_output=True, text=True, timeout=15)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"error": (r.stdout + r.stderr).strip()}


def get_pagefile_usage() -> dict:
    ps_cmd = (
        "Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,"
        "CurrentUsage,PeakUsage | ConvertTo-Json"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                        capture_output=True, text=True, timeout=15)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"error": (r.stdout + r.stderr).strip()}


def touch_heartbeat(path: str):
    Path(path).write_text(str(time.time()))


class HeartbeatThread(threading.Thread):
    def __init__(self, path: str, interval_s: float = 5.0):
        super().__init__(daemon=True)
        self.path = path
        self.interval_s = interval_s
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            touch_heartbeat(self.path)
            self._stop.wait(self.interval_s)

    def stop(self):
        self._stop.set()


def start_balloon(x_mb: float, heartbeat_path: str, log_path: str) -> subprocess.Popen:
    cmd = [
        sys.executable, BALLOON_SCRIPT,
        "--budget-x-mb", str(x_mb), "--floor-mb", str(BALLOON_FLOOR_MB),
        "--heartbeat-file", heartbeat_path, "--heartbeat-timeout-s", str(HEARTBEAT_TIMEOUT_S),
        "--max-runtime-s", str(BALLOON_MAX_RUNTIME_S), "--log-path", log_path,
        "--log-interval-s", str(BALLOON_LOG_INTERVAL_S),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def start_server(log_path: str) -> subprocess.Popen:
    kill_by_name("llama-server")
    cmd = [
        SERVER_BIN, "-m", MODEL_PATH, "--port", str(PORT), "-c", str(CTX_SIZE),
        "-ctk", KV_PREC, "-ctv", KV_PREC, "-fa", "on", "-ngl", "99", "-np", "1",
        "--no-context-shift", "--log-file", log_path, "--log-verbosity", "3",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wait_healthy(proc: subprocess.Popen, timeout: int) -> tuple[bool, str | None]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            try:
                out, err = proc.communicate(timeout=5)
            except Exception:
                out, err = "", ""
            return False, f"process exited with code {proc.returncode}: {(err or out or '')[-2000:]}"
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True, None
        except Exception:
            pass
        time.sleep(2)
    return False, None


def _tokenize(text: str) -> int:
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(f"{SERVER_URL}/tokenize", data=body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return len(json.loads(r.read())["tokens"])


def left_truncate_tokens(text: str, target_tokens: int) -> str:
    if _tokenize(text) <= target_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if _tokenize(text[mid:]) <= target_tokens:
            hi = mid
        else:
            lo = mid + 1
    return text[lo:]


def build_throughput_prompt() -> str:
    target = round(CTX_SIZE * FILL_RATIO)
    return ctx_mod.build_filler(target, seed=42, count_fn=_tokenize)


def chat_call(prompt: str, max_tokens: int, timeout: int) -> dict:
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "cache_prompt": False, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{SERVER_URL}/v1/chat/completions", data=body,
                                  headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft_s = None
    chunks = []
    finish_reason = None
    tokens_out_seen = 0
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
                if content and ttft_s is None:
                    ttft_s = time.perf_counter() - t0
                if content:
                    chunks.append(content)
                    tokens_out_seen += 1
                fr = choice0.get("finish_reason")
                if fr:
                    finish_reason = fr
        total_s = time.perf_counter() - t0
        decode_s = total_s - (ttft_s or total_s)
        decode_tps = (tokens_out_seen - 1) / decode_s if decode_s > 0 and tokens_out_seen > 1 else None
        return {"output": "".join(chunks).strip(), "ttft_s": ttft_s, "total_s": total_s,
                "decode_tps": decode_tps, "finish_reason": finish_reason, "status": "ok", "error": None}
    except Exception as e:
        return {"output": None, "ttft_s": None, "total_s": time.perf_counter() - t0,
                "decode_tps": None, "finish_reason": None, "status": "error", "error": str(e)}


def _load_scorers():
    import importlib.util
    spec = importlib.util.spec_from_file_location("probes_scorers", PROBES_DIR / "scorers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(levels=None):
    verify_model_sha256()
    scorers = _load_scorers()
    segs = [json.loads(l) for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seg_by_id = {s["id"]: s for s in segs}
    probes = [seg_by_id[f"art_{i:02d}"] for i in range(1, 11)]

    levels = levels if levels is not None else LEVELS_X_GB
    print(f"Levels (X GB, descending): {levels}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"memory_pressure_v2_{ts}.jsonl"
    hb_path = r"C:\apu\mem_pressure_v2_hb.txt"

    touch_heartbeat(hb_path)
    hb_thread = HeartbeatThread(hb_path)
    hb_thread.start()

    pagefile_info = get_pagefile_usage()
    print(f"Pagefile (running as-is): {pagefile_info}")

    all_rows = []
    t_wall_start = time.monotonic()

    for x_gb in levels:
        x_mb = x_gb * 1024
        print(f"\n{'=' * 70}\nLEVEL: X={x_gb} GB budget ({x_mb} MB)\n{'=' * 70}")

        balloon_log = rf"C:\apu\balloon_v2_x{x_gb}gb_{ts}.csv"
        balloon_proc = start_balloon(x_mb, hb_path, balloon_log)
        balloon_startup_lines = []
        for _ in range(BALLOON_STABILIZE_TICKS + 1):
            touch_heartbeat(hb_path)
            time.sleep(BALLOON_LOG_INTERVAL_S)
        # Pull whatever the balloon has printed so far (allocation result + lock status).
        try:
            balloon_stdout_snapshot = Path(balloon_log).read_text(encoding="utf-8").splitlines()[:3]
        except Exception:
            balloon_stdout_snapshot = []
        print(f"  Balloon log header: {balloon_stdout_snapshot}")

        server_log = rf"C:\apu\mem_pressure_v2_srv_x{x_gb}gb_{ts}.txt"
        t_load_start = time.monotonic()
        server_proc = start_server(server_log)
        healthy, start_error = wait_healthy(server_proc, SERVER_START_TIMEOUT_S)
        load_time_s = time.monotonic() - t_load_start

        pagefile_after = get_pagefile_usage()
        level_row_base = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level_x_gb_target": x_gb, "budget_x_mb": x_mb,
            "ctx_size": CTX_SIZE, "kv_precision": KV_PREC, "flash_attn": "on (explicit)",
            "balloon_log_file": Path(balloon_log).name,
            "balloon_log_header": balloon_startup_lines,
            "pagefile_after": pagefile_after,
        }

        if not healthy:
            classification = "fails_loudly" if start_error else "fails_silently"
            print(f"  SERVER START FAILED after {load_time_s:.1f}s. classification={classification} error={start_error}")
            all_rows.append({**level_row_base, "phase": "server_start", "server_started": False,
                              "load_time_s": round(load_time_s, 1), "classification": classification,
                              "error_text": start_error})
            kill_by_name("llama-server")
            try:
                balloon_proc.terminate(); balloon_proc.wait(timeout=15)
            except Exception:
                try:
                    balloon_proc.kill()
                except Exception:
                    pass
            out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")
            continue

        proc_info = get_server_process_info(server_proc.pid)
        print(f"  Server up after {load_time_s:.1f}s (pid={server_proc.pid}). {proc_info}")
        all_rows.append({
            **level_row_base, "phase": "server_start", "server_started": True,
            "load_time_s": round(load_time_s, 1),
            "classification": "runs_normally" if load_time_s < SERVER_START_NORMAL_THRESHOLD_S else "pages_and_slows",
            "server_pid": server_proc.pid, "server_process_info": proc_info, "error_text": None,
        })

        # Throughput
        prompt = build_throughput_prompt()
        n_tok = _tokenize(prompt)
        result = chat_call(prompt, DECODE_TOKENS, THROUGHPUT_CALL_TIMEOUT_S)
        tput_class = ("fails_loudly" if result["status"] == "error" else
                      "runs_normally" if (result["ttft_s"] or 0) < BASELINE_THROUGHPUT_TTFT_S * THROUGHPUT_SLOWDOWN_THRESHOLD else
                      "pages_and_slows")
        print(f"  Throughput: ttft={result['ttft_s']} class={tput_class}")
        all_rows.append({**level_row_base, "phase": "throughput", "n_prompt_tokens": n_tok,
                          "ttft_s": result["ttft_s"], "total_s": round(result["total_s"], 3) if result["total_s"] else None,
                          "decode_tps": result["decode_tps"], "classification": tput_class, "error_text": result["error"]})

        # Correctness at full 90%-fill ctx=32768 (fixes v1's short-prompt limitation)
        target_total = round(CTX_SIZE * FILL_RATIO)
        min_art_q = min(_tokenize(p["artifact"].strip()) + _tokenize(p["question"].strip()) for p in probes)
        filler_target = max(target_total - min_art_q, 256) + 64
        filler_full = ctx_mod.build_filler(filler_target, seed=42, count_fn=_tokenize)
        filler_full_tokens = _tokenize(filler_full)

        for probe in probes:
            pid = probe["id"]
            artifact = probe["artifact"].strip()
            question = probe["question"].strip()
            art_tok = _tokenize(artifact)
            q_tok = _tokenize(question)
            per_probe_target = max(target_total - art_tok - q_tok, 64)
            filler = left_truncate_tokens(filler_full, min(per_probe_target, filler_full_tokens))
            c_prompt = f"{filler}\n\n{artifact}\n\n{question}"
            c_tok = _tokenize(c_prompt)

            result = chat_call(c_prompt, CORRECTNESS_MAX_TOKENS, CORRECTNESS_CALL_TIMEOUT_S)
            if result["status"] == "ok" and result["output"] is not None:
                probe_dict = {"id": pid, "scorer_type": probe["scorer_type"], "expected": probe["expected"]}
                try:
                    score, score_detail = scorers.score(probe_dict, result["output"])
                except Exception as e:
                    score, score_detail = None, f"scorer_error:{e}"
            else:
                score, score_detail = None, "no output"
            print(f"  [{pid}] score={score} tok={c_tok} ttft={result['ttft_s']} out={str(result['output'])[:40]!r}")
            all_rows.append({
                **level_row_base, "phase": "correctness", "probe_id": pid, "n_prompt_tokens": c_tok,
                "score": score, "score_detail": score_detail, "output": result["output"],
                "ttft_s": result["ttft_s"], "cache_prompt": False, "position": "ADJACENT",
                "classification": "fails_loudly" if result["status"] == "error" else "runs_normally",
                "error_text": result["error"],
            })
            out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n", encoding="utf-8")

        kill_by_name("llama-server")
        try:
            balloon_proc.terminate(); balloon_proc.wait(timeout=15)
        except Exception:
            try:
                balloon_proc.kill()
            except Exception:
                pass

    wall_s = time.monotonic() - t_wall_start
    print(f"\n{'=' * 70}\nDONE. {len(all_rows)} rows. Wall clock: {wall_s / 60:.1f} min\n{'=' * 70}")
    (RESULTS_DIR / f"{out_path.stem}.DONE").write_text("done\n")
    hb_thread.stop()


if __name__ == "__main__":
    if "--smoke7" in sys.argv:
        main(levels=[7])
    else:
        main()
