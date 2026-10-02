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


SPILL_SHARED_DELTA_THRESHOLD_MIB = 1024.0  # a server process's GPU Shared Usage more than this far above its
                                           # own deep-FITS baseline is treated as spill evidence


def classify_point(requested_n_ctx, actual_n_ctx, gpu_shared_mib, baseline_shared_mib,
                   non_device_local_usage_mib=None, logged_mib=None, device_local_mib=None):
    """FITS / SILENT_SPILL, matching this validation pass's own pre-registered criterion (docs/FINDINGS.md):
    confirmed SILENT_SPILL only if the load succeeded, the actual n_ctx equals the requested one (not
    CLAMPED), and the server process's own GPU Shared Usage exceeds its deep-FITS baseline by more than
    SPILL_SHARED_DELTA_THRESHOLD_MIB.

    2026-10-02, root cause of an earlier wrong classification: VK_EXT_memory_budget heap usage (what
    vulkaninfo reports) is per CALLING process -- vulkaninfo, run from this script's own process, can only
    ever see this script's own trivial usage, never llama-server's, so it is structurally incapable of
    detecting another process's spill no matter how it is read or timed. A real point (llama-3.3-70b,
    mid_spill, requested==actual==115200) showed non_device_local_usage_mib=0.0 (vulkaninfo, wrong) while
    gpu_shared_mib=14030.7 MiB vs a 633.7 MiB deep-FITS baseline (Windows' own per-process GPU Process Memory
    counter, right) -- a real, large spill vulkaninfo could never have seen. gpu_shared_mib/gpu_dedicated_mib
    (Get-Counter, per-process, already recorded on every row) are the only signal used here now.
    non_device_local_usage_mib/logged_mib/device_local_mib are still recorded on every row (useful provenance,
    and the device-local line is still needed to report where the budget sits) but never used to decide the
    regime."""
    clamped = actual_n_ctx is not None and requested_n_ctx is not None and actual_n_ctx != requested_n_ctx
    shared_delta_mib = None
    if gpu_shared_mib is not None and baseline_shared_mib is not None:
        shared_delta_mib = gpu_shared_mib - baseline_shared_mib
    is_spill = (not clamped) and shared_delta_mib is not None and shared_delta_mib > SPILL_SHARED_DELTA_THRESHOLD_MIB
    regime = "SILENT_SPILL" if is_spill else "FITS"
    return {"regime": regime, "clamped": clamped, "shared_delta_mib": shared_delta_mib, "logged_mib": logged_mib,
            "device_local_line_mib": device_local_mib, "non_device_local_usage_mib": non_device_local_usage_mib}


def run_regime_point(model_id, point_type, requested_n_ctx, rep, out_path, log, baseline_shared_mib=None):
    # reasoning_budget=0 suppresses thinking mode -- found live 2026-10-02: qwen3-32b defaults to thinking
    # mode (matching this repo's own documented pattern for the Qwen3 family, see t2s_k1_ollama.OllamaClient.
    # chat's own think= docstring), and without this, the whole n_predict=128 budget went into hidden
    # reasoning content with zero real content tokens ever streamed -- ttft_ms stayed None for every point on
    # this model despite tokens_out=128/done_reason="length", because _stream_chat's TTFT is stamped on the
    # first real content token, which never arrived. llama-3.3-70b is unaffected (not a reasoning model), so
    # this is harmless there (an already-non-thinking model has nothing to suppress).
    cfg = LlamaServerConfig(exe=LLAMA_SERVER_EXE, model=GGUF_PATHS[model_id], ctx_size=requested_n_ctx, port=PORT,
                            n_gpu_layers=99, platform="evo-x2", backend="vulkan", reasoning_budget=0)
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
            gpu_mem = query_gpu_process_memory(session._proc.pid if session._proc else None)
            non_local_usage = read_non_device_local_usage_mib()  # recorded for provenance only, see classify_point
            parsed = L.parse_server_log(session._log_path) if session._log_path else {}
            mbuf, kbuf, cbuf = parsed.get("model_buffer_mib"), parsed.get("kv_buffer_mib"), parsed.get("compute_buffer_mib")
            logged = sum(x for x in (mbuf, kbuf, cbuf) if x) or None
            cls = classify_point(requested_n_ctx, actual_n_ctx, gpu_mem["shared_mib"], baseline_shared_mib,
                                 non_local_usage, logged, lines.get("device_local_mib"))
            prefill_tok_s = (tokens_in / (ttft_ms / 1000)) if ttft_ms else None
            decode_s = (latency_ms - ttft_ms) / 1000 if (ttft_ms is not None) else None
            decode_tok_s = ((tokens_out - 1) / decode_s) if (decode_s and decode_s > 0 and tokens_out > 1) else None
            row.update({"started": True, "actual_n_ctx": actual_n_ctx, "clamped": cls["clamped"],
                       "regime": cls["regime"], "shared_delta_mib": cls["shared_delta_mib"],
                       "baseline_shared_mib": baseline_shared_mib,
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


def _wait_after_kill(proc, log, timeout_s=30):
    """proc.wait() after kill() is NOT guaranteed to return quickly -- found live 2026-10-02: a 32B model
    process, killed mid-load at a near-crash n_ctx (large VRAM/host allocation to tear down), took longer
    than a bare 10s wait, raising an uncaught TimeoutExpired that crashed the whole job (the top-level handler
    in main() caught it and advanced the queue correctly, but the job itself still stopped early, losing the
    rest of its own run). Never re-raises: the process has already been sent SIGKILL/TerminateProcess, so it
    WILL die eventually; this only logs if confirming that within timeout_s took longer than expected."""
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log(f"warning: process {proc.pid} did not exit within {timeout_s}s of being killed; continuing anyway")


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
        _wait_after_kill(proc, log)
        killed_by_probe = True
        real_exit_code = None
    else:
        try:
            proc.wait(timeout=max(1.0, CRASH_START_TIMEOUT_S - (time.monotonic() - t0)))
            real_exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            _wait_after_kill(proc, log)
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


def _seed_deep_fits_baselines(out_path):
    """On --resume, deep_fits points from an earlier run of this same file are not re-measured (see
    already_done_keys), so the baseline a later mid_spill/near_crash point needs must come from those
    already-written rows, not just ones produced this process's lifetime."""
    baselines = {}
    if not out_path.exists():
        return baselines
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (r.get("record") == "mx2v_regime_point" and r.get("point_type") == "deep_fits"
                and r.get("started") and r.get("gpu_shared_mib") is not None):
            baselines.setdefault(r["model_id"], r["gpu_shared_mib"])
    return baselines


def run(out_path, smoke=False, deadline_h=3.0, log=print):
    done = already_done_keys(out_path)
    deep_fits_baseline = _seed_deep_fits_baselines(out_path)
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
                row = run_regime_point(model_id, point_type, n_ctx, rep, out_path, log,
                                       baseline_shared_mib=deep_fits_baseline.get(model_id))
                if (point_type == "deep_fits" and row.get("started") and row.get("gpu_shared_mib") is not None):
                    deep_fits_baseline.setdefault(model_id, row["gpu_shared_mib"])

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


def reclassify_file(path):
    """Post-processing, no re-measurement: recomputes regime/clamped/shared_delta_mib for every already-
    written mx2v_regime_point row using the CURRENT classify_point and each model's own deep_fits
    gpu_shared_mib (already recorded on those rows) as the baseline -- every other field, including the raw
    gpu_shared_mib/gpu_dedicated_mib/ttft_ms/decode_tok_s measurements themselves, is left exactly as written.
    Idempotent (safe to run again, e.g. after a later resumed run added more rows). Returns the number of
    regime_point rows updated."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append(line)  # keep unparseable lines verbatim rather than dropping them
    baselines = {}
    for r in rows:
        if (isinstance(r, dict) and r.get("record") == "mx2v_regime_point" and r.get("point_type") == "deep_fits"
                and r.get("started") and r.get("gpu_shared_mib") is not None):
            baselines.setdefault(r["model_id"], r["gpu_shared_mib"])
    n_updated = 0
    for r in rows:
        if not isinstance(r, dict) or r.get("record") != "mx2v_regime_point" or not r.get("started"):
            continue
        baseline = baselines.get(r["model_id"])
        cls = classify_point(r.get("requested_n_ctx"), r.get("actual_n_ctx"), r.get("gpu_shared_mib"), baseline,
                             r.get("non_device_local_usage_mib"), r.get("logged_mib"), r.get("device_local_line_mib"))
        r["regime"], r["clamped"], r["shared_delta_mib"], r["baseline_shared_mib"] = (
            cls["regime"], cls["clamped"], cls["shared_delta_mib"], baseline)
        n_updated += 1
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((json.dumps(r) if isinstance(r, dict) else r) + "\n")
    return n_updated


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--deadline-h", type=float, default=3.0)
    ap.add_argument("--reclassify", action="store_true",
                    help="post-process --out in place: recompute regime/clamped from already-recorded "
                         "gpu_shared_mib, no re-measurement, no queue interaction")
    args = ap.parse_args(argv)
    out_path = Path(args.out)

    if args.reclassify:
        n = reclassify_file(out_path)
        print(f"reclassified {n} regime_point rows in {out_path}")
        return

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
