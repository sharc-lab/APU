#!/usr/bin/env python3
"""
bw_saturation_sweep.py
Sweep ctx-size and KV precision; measure prefill/decode throughput vs resident KV bytes.
Run on evo-t2s. Qwen3-4b-instruct Q4_K_M, llama-server b10970, Vulkan.

Usage: python bw_saturation_sweep.py [--dry-run]
  --dry-run  Print config table and exit without running anything.
"""

import json, os, sys, time, datetime, subprocess, re, math, argparse
import urllib.request, urllib.error

# ── config ────────────────────────────────────────────────────────────────────
RUN_LOG      = r"C:\apu\bw_sweep_run.log"   # written directly by Python
SERVER_BIN   = r"C:\apu\bin\llama-b10970\llama-server.exe"
MODEL_PATH   = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
PORT         = 8383
SERVER_URL   = f"http://127.0.0.1:{PORT}"
SRV_LOG      = r"C:\apu\bw_srv_log.txt"
OUT_DIR      = r"C:\apu\results"

CTX_SIZES    = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
KV_PRECISIONS = ["f16", "q8_0", "q4_0"]
FILL_RATIO   = 0.90
DECODE_TOKENS = 64       # max tokens to generate (small, for clean decode rate)

# B/token from llamaserver_feasibility.json (CUDA blade — to be validated here)
KNOWN_B_PER_TOK = {"f16": 144530, "q8_0": 81490, "q4_0": 44626}

# Filler: varied prose, ~47 tokens per unit, ~210 chars
FILLER_UNIT = (
    "Transformer models use key-value caches to avoid recomputing attention for prior "
    "tokens. On unified memory platforms, weights and KV cache share the same physical "
    "DRAM. Bandwidth to DRAM is the dominant bottleneck at large context lengths. "
    "Measuring throughput across a range of KV sizes reveals whether saturation is "
    "gradual or produces a sharp performance cliff near a bandwidth or pool boundary. "
)


# ── helpers ───────────────────────────────────────────────────────────────────

def sys_free_mib() -> float:
    """Return free physical memory in MiB using Windows GlobalMemoryStatusEx via ctypes."""
    try:
        import ctypes
        class MEMSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength",                ctypes.c_ulong),
                ("dwMemoryLoad",            ctypes.c_ulong),
                ("ullTotalPhys",            ctypes.c_ulonglong),
                ("ullAvailPhys",            ctypes.c_ulonglong),
                ("ullTotalPageFile",        ctypes.c_ulonglong),
                ("ullAvailPageFile",        ctypes.c_ulonglong),
                ("ullTotalVirtual",         ctypes.c_ulonglong),
                ("ullAvailVirtual",         ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        ms = MEMSTATUS()
        ms.dwLength = ctypes.sizeof(ms)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPhys / (1024 * 1024)
    except Exception:
        return -1.0


def kill_server():
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process -Name llama-server -ErrorAction SilentlyContinue | Stop-Process -Force"],
        capture_output=True, timeout=15
    )
    time.sleep(3)


def start_server(ctx_size: int, kv_prec: str) -> subprocess.Popen:
    kill_server()
    cmd = [
        SERVER_BIN,
        "-m",     MODEL_PATH,
        "--port", str(PORT),
        "-c",     str(ctx_size),
        "-ctk",   kv_prec,
        "-ctv",   kv_prec,
        "-ngl",   "99",
        "-np",    "1",
        "--no-context-shift",
        "--reasoning-format", "deepseek",
        "--reasoning-budget", "0",
        "--log-file",      SRV_LOG,
        "--log-verbosity", "3",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def wait_healthy(timeout: int = 180) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def parse_kv_log() -> dict:
    """Read server log, extract reported KV buffer sizes and build info."""
    result = {}
    try:
        with open(SRV_LOG, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        return {"log_error": str(e)}

    # Build: e.g. "build: Vulkan (10970)"  or "build: 10970"
    m = re.search(r"build[:\s]+[^\n]*\b(\d{4,6})\b", text, re.IGNORECASE)
    if m:
        result["build_id"] = m.group(1)

    # KV buffer lines (may appear in different formats across builds):
    # "llama_kv_cache_init:      Vulkan0 KV buffer size =  1234.56 MiB"
    # "llama_kv_cache: K ( f16): 1234.56 MiB"
    for line in text.splitlines():
        m = re.search(r"KV buffer size\s*[=:]\s*([\d.]+)\s*MiB", line, re.IGNORECASE)
        if m:
            result.setdefault("kv_buffer_mib_list", []).append(float(m.group(1)))
        m = re.search(r"llama_kv_cache:\s+K\s+\(\s*\w+\s*\):\s+([\d.]+)\s+MiB", line)
        if m:
            result["K_MiB"] = float(m.group(1))
        m = re.search(r"llama_kv_cache:\s+V\s+\(\s*\w+\s*\):\s+([\d.]+)\s+MiB", line)
        if m:
            result["V_MiB"] = float(m.group(1))

    # Try to sum K+V if present
    if "K_MiB" in result and "V_MiB" in result:
        result["KV_total_MiB"] = result["K_MiB"] + result["V_MiB"]
    elif "kv_buffer_mib_list" in result:
        result["KV_total_MiB"] = sum(result["kv_buffer_mib_list"])

    return result


def get_props() -> dict:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/props", timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def tokenize_count(text: str, timeout: int = 120) -> int:
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/tokenize", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return len(json.loads(r.read())["tokens"])


def build_prompt(target_tokens: int) -> tuple[str, int]:
    """Build a prompt of approximately target_tokens tokens. Returns (text, actual_count)."""
    unit_chars = len(FILLER_UNIT)

    # Initial estimate: calibrate chars/token from a small sample
    sample = FILLER_UNIT * 10
    sample_toks = tokenize_count(sample)
    chars_per_tok = len(sample) / sample_toks

    # Build initial text
    needed_chars = int(target_tokens * chars_per_tok * 1.05)  # 5% overshoot
    repeats = math.ceil(needed_chars / unit_chars) + 5
    text = FILLER_UNIT * repeats

    # Trim to be close to target
    actual = tokenize_count(text, timeout=180)
    if actual > target_tokens:
        # Scale down proportionally
        scale = target_tokens / actual
        text = text[:int(len(text) * scale)]
        actual = tokenize_count(text, timeout=180)

    # Fine-trim: remove a few units at a time until <= target
    while actual > target_tokens:
        chars_to_remove = int((actual - target_tokens) * chars_per_tok) + unit_chars
        text = text[:-chars_to_remove]
        if not text:
            break
        actual = tokenize_count(text, timeout=180)

    return text, actual


def run_inference(prompt_text: str) -> dict:
    """Stream one inference call; return timing and metadata."""
    messages = [{"role": "user", "content": prompt_text}]
    body = json.dumps({
        "messages": messages,
        "max_tokens": DECODE_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )

    t_send        = time.perf_counter()
    t_first       = None
    t_last_content = None
    decode_count  = 0
    tokens_in_api = 0
    finish_reason = None
    error_text    = None
    http_status   = 200

    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if t_first is None:
                            t_first = time.perf_counter()
                        t_last_content = time.perf_counter()
                        decode_count += 1
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr

                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens"):
                    tokens_in_api = usage["completion_tokens"]

    except urllib.error.HTTPError as e:
        http_status = e.code
        error_text = e.read().decode("utf-8", errors="replace")
    except Exception as e:
        error_text = str(e)

    t_done = time.perf_counter()

    ttft_s    = (t_first - t_send) if t_first else None
    dec_time  = (t_last_content - t_first) if (t_first and t_last_content and decode_count > 1) else None

    return {
        "http_status":         http_status,
        "error_text":          error_text,
        "ttft_s":              round(ttft_s, 4)  if ttft_s  else None,
        "total_s":             round(t_done - t_send, 4),
        "decode_count_stream": decode_count,
        "tokens_in_api":       tokens_in_api,
        "finish_reason":       finish_reason,
        "decode_time_s":       round(dec_time, 4) if dec_time else None,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path      = os.path.join(OUT_DIR, f"bw_saturation_{ts}.jsonl")
    manifest_path = os.path.join(OUT_DIR, f"bw_saturation_manifest_{ts}.json")

    # Print config table
    print(f"bw_saturation_sweep  {ts}")
    print(f"Server:  {SERVER_BIN}")
    print(f"Model:   {MODEL_PATH}")
    print(f"Output:  {out_path}")
    print()
    print(f"{'CTX':>8}  {'PREC':>6}  {'KV_bytes(computed)':>20}  {'KV_MiB(computed)':>18}")
    for kv in KV_PRECISIONS:
        b = KNOWN_B_PER_TOK[kv]
        for c in CTX_SIZES:
            kv_bytes = c * b
            print(f"{c:>8}  {kv:>6}  {kv_bytes:>20,}  {kv_bytes/1048576:>17.1f}")
        print()

    if dry_run:
        return

    rows = []

    for kv_prec in KV_PRECISIONS:
        for ctx_size in CTX_SIZES:
            tag = f"ctx={ctx_size:>7} prec={kv_prec}"
            print(f"\n{'='*60}")
            print(f"CONFIG: {tag}", flush=True)

            b_per_tok      = KNOWN_B_PER_TOK[kv_prec]
            computed_kv_b  = ctx_size * b_per_tok
            computed_kv_mib = computed_kv_b / 1048576

            mem0 = sys_free_mib()
            print(f"  sys_free_before_server = {mem0:.0f} MiB", flush=True)

            print(f"  Starting server...", flush=True)
            proc = start_server(ctx_size, kv_prec)

            healthy = wait_healthy(timeout=300)
            if not healthy:
                print(f"  FAIL: server did not become healthy", flush=True)
                # Read log for error
                try:
                    with open(SRV_LOG, errors="replace") as f:
                        log_tail = f.read()[-2000:]
                except Exception:
                    log_tail = ""
                row = {
                    "ts": datetime.datetime.utcnow().isoformat() + "Z",
                    "ctx_size": ctx_size, "kv_precision": kv_prec,
                    "computed_kv_bytes": computed_kv_b,
                    "computed_kv_mib": round(computed_kv_mib, 2),
                    "status": "server_start_fail",
                    "error_text": log_tail[-500:] if log_tail else "health_timeout",
                    "sys_free_before_server_mib": round(mem0, 1),
                }
                rows.append(row)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                kill_server()
                continue

            props = get_props()
            kv_log = parse_kv_log()
            mem1 = sys_free_mib()

            print(f"  Server up.  sys_free_after_server = {mem1:.0f} MiB  "
                  f"(delta = {mem0-mem1:.0f} MiB)", flush=True)
            print(f"  KV from log: {kv_log}", flush=True)

            # Build fill prompt
            target_toks = int(ctx_size * FILL_RATIO)
            print(f"  Building {target_toks}-token prompt (90% of {ctx_size})...", flush=True)
            build_error = None
            try:
                prompt_text, actual_toks = build_prompt(target_toks)
            except Exception as e:
                build_error = str(e)
                prompt_text, actual_toks = None, 0
                print(f"  FAIL: prompt build: {e}", flush=True)

            if not prompt_text:
                row = {
                    "ts": datetime.datetime.utcnow().isoformat() + "Z",
                    "ctx_size": ctx_size, "kv_precision": kv_prec,
                    "computed_kv_bytes": computed_kv_b,
                    "computed_kv_mib": round(computed_kv_mib, 2),
                    "status": "prompt_build_fail",
                    "error_text": build_error,
                    "kv_log": kv_log,
                    "sys_free_before_server_mib": round(mem0, 1),
                    "sys_free_after_server_mib": round(mem1, 1),
                }
                rows.append(row)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                kill_server()
                continue

            print(f"  Prompt: {actual_toks} tokens (target {target_toks})", flush=True)

            mem2 = sys_free_mib()
            print(f"  Running inference...", flush=True)
            infer = run_inference(prompt_text)
            mem3 = sys_free_mib()

            # Compute throughputs
            prefill_tps = None
            decode_tps  = None
            if infer["ttft_s"] and infer["ttft_s"] > 0 and actual_toks > 0:
                prefill_tps = round(actual_toks / infer["ttft_s"], 2)
            if infer["decode_time_s"] and infer["decode_time_s"] > 0 and infer["decode_count_stream"] > 1:
                decode_tps = round((infer["decode_count_stream"] - 1) / infer["decode_time_s"], 2)

            # Measured B/tok from KV log vs computed
            measured_kv_mib = kv_log.get("KV_total_MiB")
            measured_b_per_tok = None
            if measured_kv_mib and ctx_size:
                measured_b_per_tok = round(measured_kv_mib * 1048576 / ctx_size, 1)

            status = "ok" if infer["http_status"] == 200 and not infer["error_text"] else \
                     f"http_{infer['http_status']}"

            if status == "ok":
                print(f"  OK  TTFT={infer['ttft_s']:.3f}s  "
                      f"prefill={prefill_tps} tok/s  "
                      f"decode={decode_tps} tok/s", flush=True)
            else:
                print(f"  FAIL: {status}  {(infer['error_text'] or '')[:200]}", flush=True)

            row = {
                "ts":                        datetime.datetime.utcnow().isoformat() + "Z",
                "ctx_size":                  ctx_size,
                "kv_precision":              kv_prec,
                "computed_kv_bytes":         computed_kv_b,
                "computed_kv_mib":           round(computed_kv_mib, 2),
                "measured_kv_mib":           measured_kv_mib,
                "measured_b_per_tok":        measured_b_per_tok,
                "known_b_per_tok":           b_per_tok,
                "actual_prompt_tokens":      actual_toks,
                "target_prompt_tokens":      target_toks,
                "status":                    status,
                "error_text":                infer.get("error_text"),
                "http_status":               infer.get("http_status"),
                "ttft_s":                    infer.get("ttft_s"),
                "total_s":                   infer.get("total_s"),
                "prefill_tps":               prefill_tps,
                "decode_tps":                decode_tps,
                "decode_count":              infer.get("decode_count_stream"),
                "finish_reason":             infer.get("finish_reason"),
                "sys_free_before_server_mib": round(mem0, 1),
                "sys_free_after_server_mib":  round(mem1, 1),
                "sys_free_before_infer_mib":  round(mem2, 1),
                "sys_free_after_infer_mib":   round(mem3, 1),
                "kv_log":                    kv_log,
                "server_props_build":        props.get("build_info") or props.get("build"),
            }
            rows.append(row)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            kill_server()
            time.sleep(5)

    # Manifest
    manifest = {
        "sweep_type":     "bw_saturation",
        "ts":             ts,
        "platform":       "evo-t2s",
        "hw":             "Intel Arrow Lake, 64 GB LPDDR5X unified",
        "mem_arch":       "unified",
        "model":          "qwen3:4b-instruct Q4_K_M",
        "model_sha256":   "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9",
        "model_path":     MODEL_PATH,
        "server_bin":     SERVER_BIN,
        "server_build":   "b10970",
        "ctx_sizes":      CTX_SIZES,
        "kv_precisions":  KV_PRECISIONS,
        "fill_ratio":     FILL_RATIO,
        "decode_max_tokens": DECODE_TOKENS,
        "known_b_per_tok": KNOWN_B_PER_TOK,
        "rows_total":     len(rows),
        "rows_ok":        sum(1 for r in rows if r.get("status") == "ok"),
        "rows_fail":      sum(1 for r in rows if r.get("status") != "ok"),
        "out_path":       out_path,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. {len(rows)} rows -> {out_path}")
    print(f"Manifest -> {manifest_path}")

    # Print summary table
    print(f"\n{'CTX':>8}  {'PREC':>6}  {'STATUS':>18}  {'TTFT_s':>8}  "
          f"{'PF_tps':>8}  {'DEC_tps':>8}  {'KV_MiB':>9}  {'B/tok_meas':>10}")
    for r in rows:
        print(f"{r['ctx_size']:>8}  {r['kv_precision']:>6}  {r.get('status','?'):>18}  "
              f"{str(r.get('ttft_s') or ''):>8}  "
              f"{str(r.get('prefill_tps') or ''):>8}  "
              f"{str(r.get('decode_tps') or ''):>8}  "
              f"{str(r.get('computed_kv_mib') or ''):>9}  "
              f"{str(r.get('measured_b_per_tok') or ''):>10}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Route all output to RUN_LOG (works in both interactive and windowless Start-Process).
    if not args.dry_run:
        class Tee:
            encoding = "utf-8"
            errors   = "replace"
            closed   = False
            def __init__(self, *streams):
                self._s = [s for s in streams if s is not None]
            def write(self, s):
                for st in self._s:
                    try:
                        st.write(s)
                        st.flush()
                    except Exception:
                        pass
                return len(s)
            def flush(self):
                for st in self._s:
                    try:
                        st.flush()
                    except Exception:
                        pass
            def isatty(self):    return False
            def writable(self):  return True
            def readable(self):  return False
            def seekable(self):  return False
        _log_fh = open(RUN_LOG, "w", encoding="utf-8", buffering=1)
        _log_fh.write("LOG OPEN\n"); _log_fh.flush()
        sys.stdout = Tee(sys.__stdout__, _log_fh)
        sys.stderr = Tee(sys.__stderr__, _log_fh)

    try:
        main(dry_run=args.dry_run)
    except Exception as _e:
        import traceback as _tb
        _msg = f"\nFATAL: {_e}\n{_tb.format_exc()}\n"
        try:
            sys.stderr.write(_msg)
            sys.stderr.flush()
        except Exception:
            pass
        if not args.dry_run:
            try:
                _log_fh.write(_msg); _log_fh.flush()
            except Exception:
                pass
        sys.exit(1)
