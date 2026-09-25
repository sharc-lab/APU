"""Blade only. C1: fine ctx sweep through the silent VRAM-spill regime. C2: correctness under spill.

Model: qwen3-4b-instruct GGUF (sha256 verified), llama-server b10970 CUDA, f16 KV, -fa on -ngl 99 -np 1 -t 4
--no-context-shift, no reasoning flags, port 8385. No co-runner.

C1 design
  Grid: 32768 (anchor / no-load baseline) plus 34816 .. 47104 step 2048. Per ctx: 1 discarded warm-up call and
  5 measured throughput calls (max_tokens 128, ignore_eos, cache_prompt false, temperature 0, seed 42), prompt
  filler sized to 90% of ctx. Grid order (excluding the anchor) is randomized with a logged seed; the anchor is
  re-measured after every 3 grid points and any anchor whose median TTFT drifts more than 10% from the first is
  flagged. A ctx larger than one that failed, or whose median TTFT exceeded 10x the length-scaled anchor TTFT
  (linear length scaling, which under-scales and so stops early rather than late), is skipped and recorded as such.
  Thermal gate before every call: wait up to 5 min for GPU temperature to be within 3 C of the idle temperature
  measured with the first server loaded and idle.

Telemetry: harness/blade_telemetry.py (nvidia-smi query, nvidia-smi dmon with PCIe, Windows GPU Process Memory
Dedicated/Shared Usage for the server PID, CPU total, adapter Shared Usage), 1 s cadence, one file set per run.

Usage:
  py -3.12 harness/blade_spill_sweep.py c1 --smoke      # one small condition, then telemetry column check
  py -3.12 harness/blade_spill_sweep.py c1
  py -3.12 harness/blade_spill_sweep.py c2 --ctx 32768 40960
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics as st
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
import blade_telemetry as bt  # noqa: E402
import server_guard as sg  # noqa: E402
import run_provenance as rp  # noqa: E402
import context as ctx_mod  # noqa: E402

RESULTS_DIR = REPO / "results"
SCRATCH_DIR = Path(r"C:\apu")
PROBES_DIR = REPO / "evaluation" / "probes"

SERVER_BIN = r"C:\apu\bin\llama-b10970-cuda\llama-server.exe"
MODEL_PATH = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
MODEL_SHA256 = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
PORT = 8385
SERVER_URL = f"http://127.0.0.1:{PORT}"

ANCHOR_CTX = 32768
GRID = [34816, 36864, 38912, 40960, 43008, 45056, 47104]
ORDER_SEED = 20260925
FILL_RATIO = 0.90
N_WARMUP = 1
N_MEASURED = 5
THROUGHPUT_MAX_TOKENS = 128
CORRECTNESS_MAX_TOKENS = 32
CALL_TIMEOUT_S = 2700
THERMAL_TOL_C = 3
THERMAL_MAX_WAIT_S = 300
STOP_TTFT_FACTOR = 10.0
DRIFT_FLAG = 0.10


class StopExperiment(Exception):
    pass


def log(msg: str):
    print(f"[{bt.utcnow_iso()}] {msg}", flush=True)


def verify_model_sha256():
    h = hashlib.sha256()
    with open(MODEL_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != MODEL_SHA256:
        raise StopExperiment(f"MODEL SHA256 MISMATCH {h.hexdigest()}")
    log(f"model sha256 verified {MODEL_SHA256}")


def ensure_port_free_and_no_stale_server(run=None):
    try:
        sg.assert_port_free(PORT)
    except sg.GuardError as e:
        if run is not None:
            run.emit({"record": "guard_stop", **e.record})
        raise StopExperiment(str(e))
    if bt.ps("(Get-Process llama-server -ErrorAction SilentlyContinue | Measure-Object).Count").strip() not in ("", "0"):
        raise StopExperiment("a llama-server process is already running (not listening on the port); refusing to start")


def start_server(ctx: int, log_path: str) -> subprocess.Popen:
    cmd = [SERVER_BIN, "-m", MODEL_PATH, "--port", str(PORT), "-c", str(ctx), "-ctk", "f16", "-ctv", "f16",
           "-fa", "on", "-ngl", "99", "-np", "1", "-t", "4", "--no-context-shift",
           "--log-file", log_path, "--log-verbosity", "3"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


def wait_healthy(proc: subprocess.Popen, timeout: int = 420) -> bool:
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


def kill_and_confirm(pid: int):
    """Terminate one server we started and confirm the process is gone and the port is free. Never proceeds silently."""
    for attempt in range(3):
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=30,
                           stdin=subprocess.DEVNULL)
        except Exception:
            pass
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if not bt.pid_alive(pid) and bt.listener_pid(PORT) is None:
                return
            time.sleep(2)
    raise StopExperiment(f"could not confirm server pid {pid} exited / port {PORT} freed")


def tokenize(text: str) -> int:
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(f"{SERVER_URL}/tokenize", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return len(json.loads(r.read())["tokens"])


def chat_call(prompt: str, max_tokens: int, ignore_eos: bool, n_prompt_tokens: int) -> dict:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "seed": 42, "stream": True, "cache_prompt": False, "stream_options": {"include_usage": True}}
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(f"{SERVER_URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
    n_content = 0
    usage_completion = None
    pieces = []
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
                delta = (choices[0] if choices else {}).get("delta", {})
                content = delta.get("content", "")
                if content:
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    n_content += 1
                    pieces.append(content)
                if chunk.get("usage"):
                    usage_completion = chunk["usage"].get("completion_tokens")
        total_s = time.perf_counter() - t0
    except Exception as e:
        msg = str(e)
        outcome = "timeout" if "timed out" in msg.lower() else "error"
        return {"outcome": outcome, "error": msg, "output": None}
    if t_first is None:
        return {"outcome": "error", "error": "no content tokens", "output": None}
    ttft = t_first - t0
    ct = usage_completion if usage_completion is not None else n_content
    decode_s = (t_last - t_first) if t_last and t_last > t_first else None
    return {"outcome": "ok", "error": None, "output": "".join(pieces), "ttft_s": ttft, "total_s": total_s,
            "prefill_tps": n_prompt_tokens / ttft if ttft > 0 else None,
            "decode_tps": (ct - 1) / decode_s if decode_s and ct > 1 else None,
            "completion_tokens": ct, "usage_reported": usage_completion is not None}


def thermal_gate(idle_temp):
    t0 = time.monotonic()
    t = bt.gpu_temp()
    start_t = t
    released = "temp"
    while idle_temp is not None and t is not None and t > idle_temp + THERMAL_TOL_C:
        if time.monotonic() - t0 > THERMAL_MAX_WAIT_S:
            released = "timeout"
            break
        time.sleep(5)
        t = bt.gpu_temp()
    return {"thermal_wait_s": round(time.monotonic() - t0, 1), "temp_at_gate_start": start_t,
            "temp_at_release": t, "idle_temp": idle_temp, "gate_released_by": released}


def measure_idle_temp() -> int | None:
    temps = []
    for _ in range(10):
        t = bt.gpu_temp()
        if t is not None:
            temps.append(t)
        time.sleep(1)
    return int(st.median(temps)) if temps else None


class Run:
    def __init__(self, name: str, out_dir: Path, scratch: bool):
        self.ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.name = name
        self.stem = f"{name}_{self.ts}"
        self.out_dir = out_dir
        self.jsonl = out_dir / f"{self.stem}.jsonl"
        self.manifest_path = out_dir / f"{self.stem}_manifest.json"
        self.tele = bt.Telemetry(str(out_dir / self.stem))
        self.scratch = scratch
        self.idle_temp = None
        self.manifest = {}

    def emit(self, row: dict):
        row = {"ts_utc": bt.utcnow_iso(), **row}
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")


def bring_up_server(run: Run, ctx: int, tag: str):
    """Start a server, prove it is OURS (listener pid == our pid), record its identity and memory."""
    ensure_port_free_and_no_stale_server(run)
    bt.require_ac()
    srv_log = str(SCRATCH_DIR / f"{run.stem}_srv_{tag}.txt")
    proc = start_server(ctx, srv_log)
    if not wait_healthy(proc):
        run.emit({"record": "server_start", "tag": tag, "ctx": ctx, "started": False, "server_log": srv_log})
        if proc.poll() is None:
            kill_and_confirm(proc.pid)
        return None, srv_log
    expected = {"model_path": MODEL_PATH, "n_ctx": ctx, "ctk": "f16", "ctv": "f16", "fa": "on", "ngl": 99,
                "np": 1, "threads": 4}
    try:
        guard_rec = sg.assert_server_matches(SERVER_URL, PORT, proc.pid, expected)
    except sg.GuardError as e:
        run.emit({"record": "guard_stop", "tag": tag, "ctx": ctx, **e.record})
        kill_and_confirm(proc.pid)
        raise StopExperiment(str(e))
    run.tele.start_winctr(proc.pid)
    time.sleep(3)
    info = bt.process_info(proc.pid)
    run.emit({"record": "server_start", "tag": tag, "ctx": ctx, "started": True, "server_pid": proc.pid,
              "server_log": srv_log, "guard": guard_rec, **info})
    return proc, srv_log


def run_throughput_ctx(run: Run, ctx: int, tag: str, kind: str, block: int) -> dict:
    proc, _ = bring_up_server(run, ctx, tag)
    if proc is None:
        return {"failed": True, "reason": "server_start_failed"}
    try:
        if run.idle_temp is None:
            time.sleep(15)
            run.idle_temp = measure_idle_temp()
            run.manifest["idle_temp_c"] = run.idle_temp
            run.manifest["idle_temp_definition"] = "median of 10 readings, first server loaded and idle, 15 s after health ok"
            run.save_manifest()
            log(f"idle temp {run.idle_temp} C")
        target = round(ctx * FILL_RATIO)
        prompt = ctx_mod.build_filler(target, seed=42, count_fn=tokenize)
        n_tok = tokenize(prompt)
        log(f"  {tag}: prompt {n_tok} tokens")
        measured = []
        failed = False
        for i in range(N_WARMUP + N_MEASURED):
            gate = thermal_gate(run.idle_temp)
            ok, lp = sg.RequestGuard(PORT, proc.pid).check()
            if not ok:
                run.emit({"record": "call", "tag": tag, "kind": kind, "block": block, "ctx": ctx, "call": i,
                          "warmup": i < N_WARMUP, "invalid": True, "outcome": "invalid_listener_mismatch",
                          "server_pid": proc.pid, "listener_pids": lp})
                log(f"  {tag} call={i} INVALID: listener {lp} != started pid {proc.pid}; stopping this condition")
                failed = True
                break
            t_start = bt.utcnow_iso()
            res = chat_call(prompt, THROUGHPUT_MAX_TOKENS, True, n_tok)
            row = {"record": "call", "tag": tag, "kind": kind, "block": block, "ctx": ctx, "n_prompt_tokens": n_tok,
                   "call": i, "warmup": i < N_WARMUP, "t_start_utc": t_start, "t_end_utc": bt.utcnow_iso(),
                   "server_pid": proc.pid, **gate, **{k: v for k, v in res.items() if k != "output"}}
            if res["outcome"] == "ok" and res.get("completion_tokens") != THROUGHPUT_MAX_TOKENS:
                row["completion_tokens_mismatch"] = True
            run.emit(row)
            log(f"  {tag} call={i}{' (warmup)' if i < N_WARMUP else ''} outcome={res['outcome']} "
                f"ttft={res.get('ttft_s')} decode_tps={res.get('decode_tps')} wait={gate['thermal_wait_s']}s")
            if res["outcome"] != "ok":
                failed = True
                break
            if i >= N_WARMUP:
                measured.append(res["ttft_s"])
        end_info = bt.process_info(proc.pid)
        run.emit({"record": "server_end", "tag": tag, "ctx": ctx, "server_pid": proc.pid, **end_info})
        return {"failed": failed, "median_ttft": st.median(measured) if measured else None, "n_tok": n_tok}
    finally:
        run.tele.stop_winctr()
        kill_and_confirm(proc.pid)


def check_columns(prefix: str) -> dict:
    """Smoke-test gate: every telemetry column must hold real values."""
    def valid(v):
        v = v.strip()
        return v not in ("", "[Unknown Error]", "[N/A]", "N/A", "-")
    res = {}
    p = Path(prefix + "_smi.csv")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []
    if len(lines) > 1:
        hdr = [h.strip() for h in lines[0].split(",")]
        rows = [l.split(",") for l in lines[1:] if l.strip()]
        for j, h in enumerate(hdr):
            res[f"smi.{h}"] = sum(1 for r in rows if len(r) > j and valid(r[j])) / max(len(rows), 1)
        res["smi.n_rows"] = len(rows)
    p = Path(prefix + "_dmon.txt")
    d = [l.split() for l in p.read_text(encoding="utf-8", errors="replace").splitlines()
         if l.strip() and not l.startswith("#")] if p.exists() else []
    d = [r for r in d if len(r) >= 21]
    if d:
        for name, idx in (("pwr", 3), ("sm", 6), ("pclk", 13), ("fb", 16), ("rxpci", 19), ("txpci", 20)):
            res[f"dmon.{name}"] = sum(1 for r in d if valid(r[idx])) / len(d)
        res["dmon.n_rows"] = len(d)
    p = Path(prefix + "_winctr.csv")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []
    if len(lines) > 1:
        hdr = lines[0].split(",")
        rows = [l.split(",") for l in lines[1:] if l.strip()]
        for j, h in enumerate(hdr):
            res[f"winctr.{h}"] = sum(1 for r in rows if len(r) > j and valid(r[j])) / max(len(rows), 1)
        res["winctr.n_rows"] = len(rows)
    return res


def _provenance(smoke: bool) -> dict:
    """Standing rule: refuse to produce a result from scripts that are not committed. Smoke runs may proceed
    dirty but the record says so."""
    here = Path(__file__).resolve().parent
    files = [__file__, here / "blade_telemetry.py", here / "server_guard.py", here / "run_provenance.py",
             here / "context.py", PROBES_DIR / "scorers.py"]
    try:
        return rp.script_provenance(files, require_committed=not smoke)
    except rp.ProvenanceError as e:
        raise StopExperiment(str(e))


def cmd_c1(smoke: bool):
    global N_MEASURED
    prov = _provenance(smoke)
    verify_model_sha256()
    env = bt.environment_manifest()
    out_dir = SCRATCH_DIR if smoke else RESULTS_DIR
    run = Run("smoke_c1" if smoke else "blade_c1_spill_sweep", out_dir, scratch=smoke)
    rng = random.Random(ORDER_SEED)
    grid = list(GRID)
    rng.shuffle(grid)
    blocks = [grid[i:i + 3] for i in range(0, len(grid), 3)]
    plan = [("baseline", ANCHOR_CTX, 0)]
    for b, pts in enumerate(blocks, start=1):
        plan += [("grid", c, b) for c in pts]
        plan.append(("baseline", ANCHOR_CTX, b))
    if smoke:
        plan = [("baseline", 8192, 0)]
    run.manifest = {"experiment": "C1 fine VRAM-spill sweep", "script_provenance": prov, "environment": env, "order_seed": ORDER_SEED,
                    "plan": plan, "anchor_ctx": ANCHOR_CTX, "grid": GRID, "n_warmup": N_WARMUP,
                    "n_measured": N_MEASURED, "call_timeout_s": CALL_TIMEOUT_S, "fill_ratio": FILL_RATIO,
                    "model_sha256": MODEL_SHA256, "server_bin": SERVER_BIN, "smoke": smoke,
                    "note_order": "grid order randomized with logged seed; anchor re-measured after every 3 grid points",
                    "note_nvidia_smi_glitch": "idle nvidia-smi can report impossible power values (~590 W); filter power > 200 W in analysis"}
    run.save_manifest()
    log(f"plan: {plan}")
    run.tele.start_gpu()
    fail_ctx = stop_ctx = 10 ** 9
    base_ttft = base_ntok = None
    if smoke:
        N_MEASURED = 1
    try:
        for step, (kind, ctx, block) in enumerate(plan):
            tag = f"{kind}{block}_ctx{ctx}"
            if kind == "grid" and ctx > min(fail_ctx, stop_ctx):
                run.emit({"record": "skipped", "tag": tag, "ctx": ctx, "reason": "beyond failure or 10x stop point"})
                log(f"skip {tag}")
                continue
            log(f"=== {tag} ===")
            res = run_throughput_ctx(run, ctx, tag, kind, block)
            time.sleep(5)
            if res.get("failed"):
                if kind == "grid":
                    fail_ctx = min(fail_ctx, ctx)
                continue
            if kind == "baseline":
                if base_ttft is None:
                    base_ttft, base_ntok = res["median_ttft"], res["n_tok"]
                else:
                    drift = res["median_ttft"] / base_ttft - 1
                    run.emit({"record": "baseline_check", "block": block, "median_ttft": res["median_ttft"],
                              "first_baseline_ttft": base_ttft, "drift_frac": drift, "flagged": abs(drift) > DRIFT_FLAG})
                    log(f"baseline block {block}: drift {drift:+.1%}{' FLAGGED' if abs(drift) > DRIFT_FLAG else ''}")
            elif base_ttft is not None:
                scaled = base_ttft * res["n_tok"] / base_ntok
                if res["median_ttft"] > STOP_TTFT_FACTOR * scaled:
                    stop_ctx = min(stop_ctx, ctx)
                    run.emit({"record": "stop_rule", "ctx": ctx, "median_ttft": res["median_ttft"],
                              "length_scaled_anchor": scaled, "factor": res["median_ttft"] / scaled})
                    log(f"10x stop rule reached at ctx={ctx}")
        run.manifest["completed_utc"] = bt.utcnow_iso()
    finally:
        run.tele.stop()
        time.sleep(2)
        run.save_manifest()
    if smoke:
        cols = check_columns(run.tele.prefix)
        print(json.dumps(cols, indent=2))
        bad = {k: v for k, v in cols.items() if not k.endswith("n_rows") and v < 0.8}
        print("SMOKE COLUMN CHECK:", "PASS" if not bad and cols else f"FAIL {bad}")
    else:
        (out_dir / f"{run.stem}.DONE").write_text("done\n")
    log("C1 finished")


def _load_scorers():
    spec = importlib.util.spec_from_file_location("probes_scorers", PROBES_DIR / "scorers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_c2(ctxs: list[int], smoke: bool = False):
    prov = _provenance(smoke)
    verify_model_sha256()
    env = bt.environment_manifest()
    scorers = _load_scorers()
    segs = {}
    for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l)
            segs[d["id"]] = d
    probes = [segs[f"art_{i:02d}"] for i in range(1, 6)]
    if smoke:
        probes = probes[:2]
    run = Run("smoke_c2" if smoke else "blade_c2_spill_correctness", SCRATCH_DIR if smoke else RESULTS_DIR, scratch=smoke)
    run.manifest = {"experiment": "C2 correctness under spill", "script_provenance": prov, "environment": env, "ctxs": ctxs,
                    "probes": [p["id"] for p in probes], "model_sha256": MODEL_SHA256,
                    "placement": "filler, then artifact, then question (artifact adjacent to question)",
                    "fill_ratio": FILL_RATIO, "max_tokens": CORRECTNESS_MAX_TOKENS,
                    "scorer": "evaluation/probes/scorers.py score(probe_dict, output)"}
    run.save_manifest()
    run.tele.start_gpu()
    try:
        for ctx in ctxs:
            tag = f"c2_ctx{ctx}"
            log(f"=== {tag} ===")
            proc, _ = bring_up_server(run, ctx, tag)
            if proc is None:
                continue
            try:
                if run.idle_temp is None:
                    time.sleep(15)
                    run.idle_temp = measure_idle_temp()
                    run.manifest["idle_temp_c"] = run.idle_temp
                    run.save_manifest()
                target_total = round(ctx * FILL_RATIO)
                min_aq = min(tokenize(p["artifact"].strip()) + tokenize(p["question"].strip()) for p in probes)
                filler_full = ctx_mod.build_filler(max(target_total - min_aq, 256) + 64, seed=42, count_fn=tokenize)
                full_tokens = tokenize(filler_full)

                def left_trunc(text, target_tokens):
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

                for probe in probes:
                    artifact, question = probe["artifact"].strip(), probe["question"].strip()
                    per_target = max(target_total - tokenize(artifact) - tokenize(question), 64)
                    filler = left_trunc(filler_full, min(per_target, full_tokens))
                    prompt = f"{filler}\n\n{artifact}\n\n{question}"
                    n_tok = tokenize(prompt)
                    gate = thermal_gate(run.idle_temp)
                    ok, lp = sg.RequestGuard(PORT, proc.pid).check()
                    if not ok:
                        run.emit({"record": "probe", "tag": tag, "ctx": ctx, "probe_id": probe["id"], "invalid": True,
                                  "outcome": "invalid_listener_mismatch", "server_pid": proc.pid, "listener_pids": lp})
                        log(f"  {tag} {probe['id']} INVALID: listener {lp} != {proc.pid}; stopping this condition")
                        break
                    t_start = bt.utcnow_iso()
                    res = chat_call(prompt, CORRECTNESS_MAX_TOKENS, False, n_tok)
                    score = detail = None
                    if res["outcome"] == "ok" and res.get("output") is not None:
                        try:
                            score, detail = scorers.score({"id": probe["id"], "scorer_type": probe["scorer_type"],
                                                           "expected": probe["expected"]}, res["output"])
                        except Exception as e:
                            detail = f"scorer_error:{e}"
                    run.emit({"record": "probe", "tag": tag, "ctx": ctx, "probe_id": probe["id"],
                              "n_prompt_tokens": n_tok, "t_start_utc": t_start, "t_end_utc": bt.utcnow_iso(),
                              "server_pid": proc.pid, "score": score, "score_detail": detail,
                              "output": res.get("output"), "outcome": res["outcome"], "error": res.get("error"),
                              "ttft_s": res.get("ttft_s"), **gate})
                    log(f"  {tag} {probe['id']} score={score} outcome={res['outcome']} ttft={res.get('ttft_s')}")
            finally:
                run.tele.stop_winctr()
                kill_and_confirm(proc.pid)
        run.manifest["completed_utc"] = bt.utcnow_iso()
    finally:
        run.tele.stop()
        time.sleep(2)
        run.save_manifest()
    if smoke:
        cols = check_columns(run.tele.prefix)
        print(json.dumps(cols, indent=2))
        bad = {k: v for k, v in cols.items() if not k.endswith("n_rows") and v < 0.8}
        print("SMOKE COLUMN CHECK:", "PASS" if not bad and cols else f"FAIL {bad}")
    else:
        (RESULTS_DIR / f"{run.stem}.DONE").write_text("done\n")
    log("C2 finished")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("c1")
    a.add_argument("--smoke", action="store_true")
    b = sub.add_parser("c2")
    b.add_argument("--ctx", type=int, nargs="+", required=True)
    b.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    try:
        if args.cmd == "c1":
            cmd_c1(args.smoke)
        else:
            cmd_c2(args.ctx, args.smoke)
    except StopExperiment as e:
        log(f"STOP: {e}")
        sys.exit(2)
