"""MX2 validation (evo-x2): resolves the "guard: server does not match intended config" caveat in MX2's own
bisection and measures the real cost of the SILENT_SPILL regime plus the crash boundary, for llama-3.3-70b and
qwen3-32b. See docs/FINDINGS.md's "PRE-REGISTRATION: MX2 validation, evo-x2" section (written before this
script ran) for the methodology this implements and why: most of MX2's own bisection above a requested n_ctx
of ~123K (70B) / ~225K (32B) silently negotiated n_ctx=131072 regardless of the requested value (both models
are launched with RoPE/YaRN scaling capped at that effective ceiling), so those rows describe a repeat
measurement of the 131072 configuration, not the nominal context in their label.

Three jobs, each resumable and heartbeat-per-point:
  1. regime_point: for each model, 3 reps at 3 chosen n_ctx values (deep FITS, mid SILENT_SPILL, just below
     the crash onset) -- a fixed 2048-token prompt, n_predict=128, recording TTFT, prefill tok/s, decode
     tok/s, per-process GPU dedicated/shared usage, and the heap budgets. Every point records BOTH the
     requested n_ctx and the actual negotiated one (LlamaServerSession.n_ctx_slot, parsed from the server's
     own startup log) -- a point whose actual n_ctx differs from the requested one is labeled "CLAMPED" and
     reports the regime of its real (clamped) configuration, not the nominal one.
  2. crash_repro: for each model, 3 reps of the known crash-onset n_ctx (expected to fail): exit code, error
     text, and a responsiveness check (local round-trip timing of a trivial subprocess call, compared to a
     pre-run baseline) to see whether the crash degrades just the one process or the whole host.
  3. (implicit) a baseline responsiveness probe taken once before any crash reproduction.

Usage:
  py -3.12 harness/mx2_validation.py --out results/mx2_validation_<stem>.jsonl [--smoke] [--deadline-h 3]
"""
from __future__ import annotations

import argparse
import json
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

import t2s_night2 as night2  # noqa: E402  -- reused only for its standalone mx2_* heap-measurement helpers
import t2s_lab as L  # noqa: E402  -- reused only for parse_server_log's real model/kv/compute buffer regexes
from llama_server import LlamaServerConfig, LlamaServerSession  # noqa: E402

LLAMA_SERVER_EXE = r"C:\apu\bin\llama-b10970\llama-server.exe"
PORT = 58399

GGUF_PATHS = {
    "llama-3.3-70b": r"C:\apu\models\Llama-3.3-70B-Instruct-Q4_K_M.gguf",
    "qwen3-32b": r"C:\apu\models\Qwen3-32B-Q4_K_M.gguf",
}

# Chosen from the real, already-measured mx2_probe boundaries in results/t2s_night2_20261001T151340Z.jsonl,
# restricted to points whose own bisection row carried no "guard" mismatch text wherever a clean one existed
# (see docs/FINDINGS.md's pre-registration for the full reasoning). qwen3-32b has no clean SILENT_SPILL point
# at all in the prior data (every spill point there was already beyond the 131072 ceiling) -- its mid_spill
# point below is therefore expected to come back CLAMPED, and is kept anyway so the row itself documents that
# finding rather than silently avoiding it.
REGIME_POINTS = {
    "llama-3.3-70b": {"deep_fits": 20480, "mid_spill": 115200, "near_crash": 221440},
    "qwen3-32b": {"deep_fits": 20480, "mid_spill": 290000, "near_crash": 367104},
}
CRASH_POINTS = {"llama-3.3-70b": 221696, "qwen3-32b": 367360}

N_REPS = 3
FIXED_PROMPT_TOKENS = 2048
N_PREDICT = 128
CRASH_START_TIMEOUT_S = 60
HEARTBEAT_EVERY_S = 60


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def emit(out_path, row):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def already_done_keys(out_path):
    done = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("record") == "mx2v_regime_point":
            done.add(("regime_point", r["model_id"], r["point_type"], r["rep"]))
        elif r.get("record") == "mx2v_crash_repro":
            done.add(("crash_repro", r["model_id"], r["rep"]))
    return done


def build_fixed_prompt(n_tokens=FIXED_PROMPT_TOKENS):
    """A deterministic, repeatable filler prompt. Not tokenizer-exact (token count is confirmed per-call from
    the server's own response, not assumed) -- ~4 chars/token is this repo's own established estimate for a
    generic English filler (see harness/quality_suite.py's own char/token note), so this pads to roughly
    n_tokens*4 characters and the real prompt_eval_count (recorded per call) is the source of truth, not this
    estimate."""
    unit = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    text = (unit * (n_tokens * 4 // len(unit) + 1))[: n_tokens * 4]
    return text


FIXED_PROMPT = build_fixed_prompt()


def query_gpu_process_memory(pid, ps_fn=None):
    """Per-process GPU dedicated/shared usage via the same Windows performance-counter pattern already
    confirmed working on this exact machine (harness/lhm_x2_control.py's own mem_out query). Returns
    {"dedicated_mib", "shared_mib"}, both None if the counters could not be read (process already exited,
    or the counter category is unavailable)."""
    if ps_fn is None:
        ps_fn = L.ps  # t2s_lab's own PowerShell runner, already imported above
    cmd = (f'(Get-Counter -Counter "\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage",'
           f'"\\GPU Process Memory(pid_{pid}_*)\\Shared Usage" -ErrorAction SilentlyContinue).CounterSamples '
           f'| Select Path,CookedValue | ConvertTo-Json -Compress')
    try:
        out = ps_fn(cmd, 30)
        ctr = json.loads(out) if out else []
        ctr = ctr if isinstance(ctr, list) else [ctr]
    except Exception:
        ctr = []
    ded = next((c["CookedValue"] for c in ctr if "dedicated" in c.get("Path", "").lower()), None)
    shr = next((c["CookedValue"] for c in ctr if "shared" in c.get("Path", "").lower()), None)
    return {"dedicated_mib": None if ded is None else ded / 2 ** 20, "shared_mib": None if shr is None else shr / 2 ** 20}


def read_heap_lines():
    """The device-local / vulkan-total budget lines, via the exact same standalone readers MX2's own baseline
    used (t2s_night2.mx2_read_vulkaninfo -> mx2_parse_heaps -> mx2_lines_from_heaps), so this validation pass
    and the original bisection agree on what a heap is."""
    vk_txt = night2.mx2_read_vulkaninfo()
    heaps = night2.mx2_parse_heaps(vk_txt) if vk_txt else []
    return night2.mx2_lines_from_heaps(heaps)


def read_non_device_local_usage_mib():
    """Real, live usage_mib (not a budget/size) summed across every heap vulkaninfo reports as NOT
    device-local, at the moment this is called -- the direct spill signal classify_point uses. None if
    vulkaninfo could not be read at all (distinct from 0, a real reading of no usage)."""
    vk_txt = night2.mx2_read_vulkaninfo()
    if not vk_txt:
        return None
    heaps = night2.mx2_parse_heaps(vk_txt)
    non_local = [h for h in heaps if not h["device_local"]]
    return sum(h["usage_mib"] for h in non_local) if non_local else 0.0


def responsiveness_probe():
    """Local round-trip latency of a trivial subprocess call -- the proxy this validation pass uses for 'is
    the host still responsive', not a full health check. A crash that merely kills the llama-server process
    should leave this near its baseline; a crash that wedges the GPU driver (a TDR-style reset, or a shared-
    memory exhaustion that affects the whole session) should show up as a large increase."""
    t0 = time.monotonic()
    try:
        subprocess.run(["cmd", "/c", "echo", "hi"], capture_output=True, text=True, timeout=30)
        ok = True
    except Exception:
        ok = False
    return {"round_trip_s": time.monotonic() - t0, "ok": ok}


SPILL_USAGE_THRESHOLD_MIB = 256.0  # a non-device-local heap showing more than this much real usage, while the
                                   # server is loaded and serving, is treated as spill evidence -- small enough
                                   # that incidental driver/OS overhead on that heap does not false-positive,
                                   # large enough that it cannot be mistaken for noise


def classify_point(requested_n_ctx, actual_n_ctx, non_device_local_usage_mib, logged_mib=None, device_local_mib=None,
                   spill_margin_mib=512.0):
    """FITS / SILENT_SPILL, following the exact definitions in t2s_night2.mx2_classify's own docstring, but
    applied to the ACTUAL negotiated n_ctx -- a point is CLAMPED (actual != requested) in addition to whichever
    regime its real, achieved configuration falls into.

    Primary evidence is non_device_local_usage_mib: the REAL, live Vulkan heap usage on whichever heap is NOT
    device-local, queried via vulkaninfo while the server is loaded and has just served a real call -- direct
    confirmation that data moved off device-local memory, matching this validation pass's own pre-registered
    SILENT_SPILL criterion (docs/FINDINGS.md) more directly than llama-server's own startup-log buffer-size
    lines, which require --log-verbosity 4 to even appear (found live 2026-10-02: verbosity 3, this build's
    hardcoded default, never prints them at all). logged_mib/device_local_mib (from the log, when available)
    are kept as secondary, supplementary evidence only -- never required for the regime decision."""
    clamped = actual_n_ctx is not None and requested_n_ctx is not None and actual_n_ctx != requested_n_ctx
    evidence = []
    if non_device_local_usage_mib is not None and non_device_local_usage_mib > SPILL_USAGE_THRESHOLD_MIB:
        evidence.append("non_device_local_heap_usage_over_threshold")
    if device_local_mib is not None and logged_mib is not None and logged_mib > device_local_mib - spill_margin_mib:
        evidence.append("logged_buffers_over_device_local_line")
    regime = "SILENT_SPILL" if evidence else "FITS"
    return {"regime": regime, "clamped": clamped, "logged_mib": logged_mib, "spill_evidence": evidence,
            "device_local_line_mib": device_local_mib, "non_device_local_usage_mib": non_device_local_usage_mib}


def run_regime_point(model_id, point_type, requested_n_ctx, rep, out_path, log):
    cfg = LlamaServerConfig(exe=LLAMA_SERVER_EXE, model=GGUF_PATHS[model_id], ctx_size=requested_n_ctx, port=PORT,
                            n_gpu_layers=99, platform="evo-x2", backend="vulkan")
    row = {"record": "mx2v_regime_point", "model_id": model_id, "point_type": point_type,
          "requested_n_ctx": requested_n_ctx, "rep": rep, "ts_utc": utc_iso()}
    lines = read_heap_lines()
    try:
        with LlamaServerSession(cfg) as session:
            actual_n_ctx = session.n_ctx_slot
            # The memory-breakdown lines are written as part of model load, which /health can report ready
            # before fully finishing (found live 2026-10-02: the log was only 1642 bytes -- no buffer-size
            # lines at all -- read immediately after start()). Reading the log AFTER the actual call, once
            # the model has certainly finished loading and generated real output, avoids that race.
            text, latency_ms, ttft_ms, tokens_in, tokens_out, done_reason, _think = session.call(FIXED_PROMPT, N_PREDICT)
            non_local_usage = read_non_device_local_usage_mib()  # live, while the model is still loaded
            parsed = L.parse_server_log(session._log_path) if session._log_path else {}
            mbuf, kbuf, cbuf = parsed.get("model_buffer_mib"), parsed.get("kv_buffer_mib"), parsed.get("compute_buffer_mib")
            logged = sum(x for x in (mbuf, kbuf, cbuf) if x) or None
            cls = classify_point(requested_n_ctx, actual_n_ctx, non_local_usage, logged, lines.get("device_local_mib"))
            prefill_tok_s = (tokens_in / (ttft_ms / 1000)) if ttft_ms else None
            decode_s = (latency_ms - ttft_ms) / 1000 if (ttft_ms is not None) else None
            decode_tok_s = ((tokens_out - 1) / decode_s) if (decode_s and decode_s > 0 and tokens_out > 1) else None
            gpu_mem = query_gpu_process_memory(session._proc.pid if session._proc else None)
            row.update({"started": True, "actual_n_ctx": actual_n_ctx, "clamped": cls["clamped"],
                       "regime": cls["regime"], "spill_evidence": cls["spill_evidence"],
                       "logged_mib": cls["logged_mib"], "device_local_line_mib": cls["device_local_line_mib"],
                       "non_device_local_usage_mib": cls["non_device_local_usage_mib"],
                       "model_buffer_mib": mbuf, "kv_buffer_mib": kbuf, "compute_buffer_mib": cbuf,
                       "ttft_ms": ttft_ms, "latency_ms": latency_ms, "tokens_in": tokens_in, "tokens_out": tokens_out,
                       "prefill_tok_s": prefill_tok_s, "decode_tok_s": decode_tok_s, "done_reason": done_reason,
                       "gpu_dedicated_mib": gpu_mem["dedicated_mib"], "gpu_shared_mib": gpu_mem["shared_mib"]})
    except Exception as e:
        row.update({"started": False, "error": str(e)[:400]})
    emit(out_path, row)
    log(f"{model_id} {point_type} req={requested_n_ctx} rep={rep}: started={row.get('started')} "
        f"actual_n_ctx={row.get('actual_n_ctx')} regime={row.get('regime')} decode_tok_s={row.get('decode_tok_s')}")
    return row


def run_crash_repro(model_id, n_ctx, rep, out_path, log):
    """Launches llama-server directly (not via LlamaServerSession, which discards exit-code/log detail on a
    failed start) so a genuine crash's exit code and error text are captured, then runs the responsiveness
    probe immediately after."""
    row = {"record": "mx2v_crash_repro", "model_id": model_id, "n_ctx": n_ctx, "rep": rep, "ts_utc": utc_iso()}
    log_path = REPO / "results" / f"mx2v_crash_{model_id}_{n_ctx}_{rep}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Same flags as the original MX2 bisection's own Server._cmd() (t2s_lab.py) -- a "crash reproduction" with
    # different flags is not a reproduction at all; the original run's -ctk/-ctv/-fa/-np/-t/--no-context-shift
    # all affect the actual memory footprint, not just logging.
    cmd = [LLAMA_SERVER_EXE, "-m", GGUF_PATHS[model_id], "--port", str(PORT), "-c", str(n_ctx), "-ctk", "f16",
          "-ctv", "f16", "-fa", "on", "-np", "1", "-t", "4", "--no-context-shift", "-ngl", "99",
          "--log-file", str(log_path), "--log-verbosity", "4"]
    t0 = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    import httpx
    started = False
    while time.monotonic() - t0 < CRASH_START_TIMEOUT_S:
        if proc.poll() is not None:
            break
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2.0).status_code == 200:
                started = True
                break
        except Exception:
            pass
        time.sleep(1.0)
    killed_by_probe = False
    if started:
        # The server came up -- this n_ctx did NOT reproduce a crash this time. Killing it for cleanup sets
        # Windows' own default TerminateProcess exit code (1), which must never be reported as if it were a
        # genuine crash exit code -- real_exit_code stays None, and killed_by_probe says why the process is
        # gone at all.
        proc.kill()
        proc.wait(timeout=10)
        killed_by_probe = True
        real_exit_code = None
    else:
        try:
            proc.wait(timeout=max(1.0, CRASH_START_TIMEOUT_S - (time.monotonic() - t0)))
            real_exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            killed_by_probe = True
            real_exit_code = None  # never started, never exited on its own within the timeout -- a HANG, not
                                   # a captured crash code
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    row.update({"started": started, "exit_code": real_exit_code, "killed_by_probe": killed_by_probe,
               "error_tail": log_text[-800:]})
    responsiveness = responsiveness_probe()
    row["responsiveness"] = responsiveness
    emit(out_path, row)
    log(f"{model_id} crash_repro n_ctx={n_ctx} rep={rep}: started={started} exit_code={proc.returncode} "
        f"responsiveness_ok={responsiveness['ok']} rtt_s={responsiveness['round_trip_s']:.3f}")
    return row


def run(out_path, smoke=False, deadline_h=3.0, log=print):
    done = already_done_keys(out_path)
    t_start = time.monotonic()
    deadline_s = deadline_h * 3600
    baseline_resp = responsiveness_probe()
    emit(out_path, {"record": "mx2v_baseline_responsiveness", **baseline_resp, "ts_utc": utc_iso()})
    log(f"baseline responsiveness: ok={baseline_resp['ok']} rtt_s={baseline_resp['round_trip_s']:.3f}")

    models = list(REGIME_POINTS)
    if smoke:
        models = models[:1]

    for model_id in models:
        for point_type, n_ctx in REGIME_POINTS[model_id].items():
            if smoke and point_type != "deep_fits":
                continue
            reps = 1 if smoke else N_REPS
            for rep in range(reps):
                if time.monotonic() - t_start > deadline_s:
                    log(f"deadline reached ({deadline_h}h), stopping")
                    return
                key = ("regime_point", model_id, point_type, rep)
                if key in done:
                    continue
                emit(out_path, {"record": "heartbeat", "model_id": model_id, "point_type": point_type, "rep": rep,
                               "ts_utc": utc_iso()})
                run_regime_point(model_id, point_type, n_ctx, rep, out_path, log)

    for model_id in models:
        reps = 1 if smoke else N_REPS
        for rep in range(reps):
            if time.monotonic() - t_start > deadline_s:
                log(f"deadline reached ({deadline_h}h), stopping")
                return
            key = ("crash_repro", model_id, rep)
            if key in done:
                continue
            emit(out_path, {"record": "heartbeat", "model_id": model_id, "crash_rep": rep, "ts_utc": utc_iso()})
            run_crash_repro(model_id, CRASH_POINTS[model_id], rep, out_path, log)

    emit(out_path, {"record": "run_end", "ts_utc": utc_iso()})
    log("done")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--deadline-h", type=float, default=3.0)
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"logging to {out_path}")

    note = "completed"
    try:
        run(out_path, smoke=args.smoke, deadline_h=args.deadline_h)
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
