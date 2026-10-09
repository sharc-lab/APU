"""Blade K1 (night 1, job 1): Ollama's context policy on an 8 GB discrete NVIDIA GPU, plus llama.cpp CUDA defaults.

Measures context policy only, not agent behaviour (that is R2). Per model (default llama3.1:8b, qwen3:8b,
qwen3-4b-2507-tools; the 4B tools tag failed its R2 validation on evo-x2, register R2-install-path-4b-comparison, and
is here only for its context policy):

  1. device + VRAM: nvidia-smi totals, and Ollama's own server-log lines ("inference compute", "vram-based default
     context") from a server this job starts with its own log file.
  2. default context: one short chat with no num_ctx and keep_alive 10m, then GET /api/ps (context_length, size,
     size_vram; fully_on_gpu = size_vram >= 99% of size), nvidia-smi used, explicit unload.
  3. calibration: one ~4000-token marker prompt at num_ctx 8192 (fits, so nothing is cut); tokenizer_ratio =
     prompt_eval_count / target, used to turn every later target into the real token count sent.
  4. overflow at the default context: marker prompts of 4096 / 8192 / 16384 / 32768 target tokens, no num_ctx.
     Per call: HTTP status (400 = hard error), prompt_eval_count vs the real tokens sent (token_truncated), marker
     found, /api/ps context_length and size_vram at that point.
  5. overflow at a set window: num_ctx 4096 / 8192 / 16384 / 32768 each with a prompt of 1.5x the window; processed
     tokens vs the half-window rule floor(num_ctx/2)+2 (register ollama-overflow-keeps-half), status, size_vram
     (KV not fitting in 8 GB shows up as size_vram < size: Ollama moved layers to the CPU).
  6. llama.cpp CUDA defaults: llama-server b10970 CUDA with only -m (the model's own Ollama blob), --port, --log-file:
     /props n_ctx, and the log's layer offload, KV and buffer lines; then it is stopped and its exit confirmed.

Output: <out> (jsonl, records blade_k1_*) and <out stem>.summary.json, which blade_r2 reads to decide whether
qwen3:8b fits in 8 GB at its default context (fits = fully_on_gpu at the default context).

Usage:
  py -3.12 harness/blade_k1.py --out results/blade_k1_v1.jsonl
  py -3.12 harness/blade_k1.py --out results/blade_dryrun/blade_dryrun_k1.jsonl --models llama3.2 \
      --dry-run-seconds 60 --allow-version-mismatch --ollama-exe <installed ollama.exe>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

import blade_common as bc  # noqa: E402

DEFAULT_MODELS = ("llama3.1:8b", "qwen3:8b", "qwen3-4b-2507-tools")
DEFAULT_PROBES = (4096, 8192, 16384, 32768)
WINDOW_PROBES = (4096, 8192, 16384, 32768)
CALIB_TARGET, CALIB_NUM_CTX = 4000, 8192
OVERFLOW_FACTOR = 1.5
FULLY_ON_GPU_FRAC = 0.99
LLAMA_PORT = 8386
CALL_TIMEOUT_S = 3600
SEED = 20261008

_LOG_PATTERNS = {
    "inference_compute": re.compile(r'msg="inference compute".*'),
    "vram_default_ctx": re.compile(r'msg="vram-based default context" total_vram="([^"]*)" default_num_ctx=(\d+)'),
    "dropping_gpu": re.compile(r'msg="dropping [^"]*GPU[^"]*".*'),
    "offloaded": re.compile(r"offloaded (\d+)/(\d+) layers to GPU"),
}
_LLAMA_LOG = {
    "n_ctx": re.compile(r"llama_context: n_ctx\s*=\s*(\d+)"),
    "n_ctx_seq": re.compile(r"n_ctx_seq\s*\(?\s*=?\s*(\d+)"),
    "offloaded": re.compile(r"offloaded (\d+)/(\d+) layers to GPU"),
    "fit": re.compile(r"(?i)\bfit(?:ting)?\b.*"),
    "kv_buffer": re.compile(r"(\w+) KV buffer size\s*=\s*([\d.]+) MiB"),
    "model_buffer": re.compile(r"(\w+) model buffer size\s*=\s*([\d.]+) MiB"),
}


def think_for(tag: str):
    import x2_r2_agent as agent
    if tag in agent.MODELS:
        return agent.MODELS[tag]
    return False if tag.startswith("qwen3") else None


def parse_ollama_server_log(text: str) -> dict:
    out = {"inference_compute": [], "dropping_gpu": [], "vram_default_ctx": None, "total_vram": None,
           "offloaded": []}
    for line in (text or "").splitlines():
        if _LOG_PATTERNS["inference_compute"].search(line):
            out["inference_compute"].append(line.strip()[:400])
        if _LOG_PATTERNS["dropping_gpu"].search(line):
            out["dropping_gpu"].append(line.strip()[:400])
        m = _LOG_PATTERNS["vram_default_ctx"].search(line)
        if m:
            out["total_vram"], out["vram_default_ctx"] = m.group(1), int(m.group(2))
        m = _LOG_PATTERNS["offloaded"].search(line)
        if m:
            out["offloaded"].append([int(m.group(1)), int(m.group(2))])
    lib = re.findall(r"library=(\w+)", "\n".join(out["inference_compute"]))
    out["libraries"] = sorted(set(lib))
    out["device_guess"] = ("gpu" if any(l.lower() != "cpu" for l in lib) else "cpu") if lib else "unknown"
    return out


def parse_llama_server_log(text: str) -> dict:
    out = {"n_ctx": None, "offloaded": None, "kv_buffers_mib": {}, "model_buffers_mib": {}, "fit_lines": []}
    for line in (text or "").splitlines():
        m = _LLAMA_LOG["n_ctx"].search(line)
        if m:
            out["n_ctx"] = int(m.group(1))
        m = _LLAMA_LOG["offloaded"].search(line)
        if m:
            out["offloaded"] = [int(m.group(1)), int(m.group(2))]
        m = _LLAMA_LOG["kv_buffer"].search(line)
        if m:
            out["kv_buffers_mib"][m.group(1)] = float(m.group(2))
        m = _LLAMA_LOG["model_buffer"].search(line)
        if m:
            out["model_buffers_mib"][m.group(1)] = float(m.group(2))
        if "fit" in line.lower() and len(out["fit_lines"]) < 30:
            out["fit_lines"].append(line.strip()[:300])
    return out


def fully_on_gpu(ps_entry: dict | None) -> bool | None:
    if not ps_entry or not ps_entry.get("size"):
        return None
    return (ps_entry.get("size_vram") or 0) >= FULLY_ON_GPU_FRAC * ps_entry["size"]


def call_rates(res: dict) -> dict:
    """Prefill and decode rates of one Ollama /api/chat call, from its own counters (durations in ns)."""
    raw = res.get("raw") or {}
    pe, ped = raw.get("prompt_eval_count"), raw.get("prompt_eval_duration")
    ec, ed = raw.get("eval_count"), raw.get("eval_duration")
    return {"prompt_eval_count": pe, "prompt_eval_duration_s": ped / 1e9 if ped else None,
            "eval_count": ec, "eval_duration_s": ed / 1e9 if ed else None,
            "prefill_tps": pe / (ped / 1e9) if pe and ped else None,
            "decode_tps": ec / (ed / 1e9) if ec and ed else None}


def fits_at(model_summary: dict, ctxs) -> dict:
    """{ctx: fully_on_gpu} for each ctx in ctxs ("default" or an int), from summarize()'s per-model entry. A ctx
    K1 did not measure maps to None (unknown counts as not fitting)."""
    out = {}
    by = model_summary.get("fully_on_gpu_by_ctx") or {}
    for c in ctxs:
        if c == "default":
            out["default"] = model_summary.get("fully_on_gpu")
        else:
            out[str(int(c))] = by.get(str(int(c)))
    return out


def half_window(num_ctx: int) -> int:
    return num_ctx // 2 + 2


def summarize(rows: list[dict]) -> dict:
    """{"models": {tag: {...}}, "device": {...}}; pure, from the job's own rows."""
    out = {"models": {}, "device": None}
    for r in rows:
        rec = r.get("record")
        if rec == "blade_k1_device":
            out["device"] = {k: r.get(k) for k in ("vram_total_mib", "vram_free_mib", "vram_default_ctx",
                                                   "total_vram", "libraries", "device_guess")}
        elif rec == "blade_k1_default_ctx":
            out["models"].setdefault(r["model_tag"], {}).update({
                "default_ctx": r.get("context_length"), "size": r.get("size"), "size_vram": r.get("size_vram"),
                "fully_on_gpu": r.get("fully_on_gpu"), "chat_outcome": r.get("chat_outcome")})
        elif rec == "blade_k1_model_missing":
            out["models"].setdefault(r["model_tag"], {}).update({"missing": True})
        elif rec in ("blade_k1_overflow_default", "blade_k1_overflow_window"):
            key = "overflow_default" if rec.endswith("default") else "overflow_window"
            out["models"].setdefault(r["model_tag"], {}).setdefault(key, []).append(
                {k: r.get(k) for k in ("target_tokens", "num_ctx", "http_status", "prompt_eval_count",
                                       "sent_tokens_real", "token_truncated", "marker_found", "half_window_rule",
                                       "context_length", "size_vram", "size")})
        elif rec == "blade_k1_llamacpp_defaults":
            out["models"].setdefault(r["model_tag"], {})["llamacpp_defaults"] = {
                k: r.get(k) for k in ("started", "props_n_ctx", "log")}
    for tag, m in out["models"].items():
        m["fits_8gb_at_default"] = bool(m.get("fully_on_gpu")) and not m.get("missing")
        m["fully_on_gpu_by_ctx"] = {str(w["num_ctx"]): fully_on_gpu(w) for w in m.get("overflow_window", [])
                                    if w.get("num_ctx")}
    return out


class K1Job:
    def __init__(self, args, emit, log, ollama_client=None, server=None, run=None):
        self.args, self.emit, self.log = args, emit, log
        self.server = server
        self.run = run
        self.client = ollama_client

    def ps_entry(self, tag):
        import x2_r2_agent as agent
        for m in self.client.get_ps().get("models", []):
            if agent.same_tag(m.get("name") or m.get("model"), tag):
                return m
        return None

    def build_prompt(self, target, seed):
        import t2s_k1_ollama as k1
        return k1.qs_build_task(k1._EMPIRICAL_CTX_TASK_TYPE, target, seed)

    def chat(self, tag, prompt, num_ctx, max_tokens=32, keep_alive="10m"):
        return self.client.chat(tag, prompt, num_ctx=num_ctx, max_tokens=max_tokens, keep_alive=keep_alive,
                                think=think_for(tag), timeout=CALL_TIMEOUT_S)

    def unload(self, tag):
        self.client.unload(tag)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and self.ps_entry(tag):
            time.sleep(0.5)

    def model_present(self, tag, avail):
        import x2_r2_agent as agent
        return any(agent.same_tag(tag, a) for a in avail)

    def run_model(self, tag):
        e, log = self.emit, self.log
        res = self.chat(tag, "In one short sentence, name the capital of France.", None)
        ps = self.ps_entry(tag) or {}
        e({"record": "blade_k1_default_ctx", "model_tag": tag, "chat_outcome": res.get("outcome"),
           "http_status": res.get("status"), "error": res.get("error"),
           "context_length": ps.get("context_length"), "size": ps.get("size"), "size_vram": ps.get("size_vram"),
           "fully_on_gpu": fully_on_gpu(ps), "ps_raw": ps, **call_rates(res), **bc.gpu_memory(self.run)})
        log(f"{tag}: default ctx {ps.get('context_length')} size {ps.get('size')} vram {ps.get('size_vram')}")
        self.unload(tag)
        cal_prompt, _, _ = self.build_prompt(CALIB_TARGET, SEED)
        cal = self.chat(tag, cal_prompt, CALIB_NUM_CTX, max_tokens=1)
        pec = cal.get("prompt_eval_count")
        ratio = pec / CALIB_TARGET if (pec and cal.get("outcome") == "ok") else None
        e({"record": "blade_k1_calibration", "model_tag": tag, "target_tokens": CALIB_TARGET,
           "num_ctx": CALIB_NUM_CTX, "prompt_eval_count": pec, "tokenizer_ratio": ratio,
           "outcome": cal.get("outcome"), "error": cal.get("error")})
        self.unload(tag)
        for target in DEFAULT_PROBES:
            prompt, expected, _ = self.build_prompt(target, SEED + target)
            r = self.chat(tag, prompt, None)
            ps = self.ps_entry(tag) or {}
            sent = round(target * ratio) if ratio else None
            pec = r.get("prompt_eval_count")
            e({"record": "blade_k1_overflow_default", "model_tag": tag, "target_tokens": target, "num_ctx": None,
               "sent_tokens_real": sent, "http_status": r.get("status"), "outcome": r.get("outcome"),
               "error": r.get("error"), "prompt_eval_count": pec,
               "token_truncated": (pec is not None and sent is not None and pec < 0.99 * sent),
               "marker_found": bool(r.get("message")) and str(expected) in (r.get("message") or ""),
               "context_length": ps.get("context_length"), "size": ps.get("size"), "size_vram": ps.get("size_vram"),
               "duration_s": r.get("duration_s"), **call_rates(r), **bc.gpu_memory(self.run)})
        self.unload(tag)
        for n in WINDOW_PROBES:
            target = round(OVERFLOW_FACTOR * n / ratio) if ratio else round(OVERFLOW_FACTOR * n)
            prompt, expected, _ = self.build_prompt(target, SEED + n + 1)
            r = self.chat(tag, prompt, n)
            ps = self.ps_entry(tag) or {}
            pec = r.get("prompt_eval_count")
            e({"record": "blade_k1_overflow_window", "model_tag": tag, "target_tokens": target, "num_ctx": n,
               "sent_tokens_real": round(target * ratio) if ratio else None, "http_status": r.get("status"),
               "outcome": r.get("outcome"), "error": r.get("error"), "prompt_eval_count": pec,
               "half_window_rule": half_window(n), "matches_half_window": pec == half_window(n),
               "marker_found": bool(r.get("message")) and str(expected) in (r.get("message") or ""),
               "context_length": ps.get("context_length"), "size": ps.get("size"), "size_vram": ps.get("size_vram"),
               "duration_s": r.get("duration_s"), **call_rates(r), **bc.gpu_memory(self.run)})
            self.unload(tag)

    def llamacpp_defaults(self, tag, blob, popen=None):
        """llama-server with no context or offload flags; Ollama must be stopped first (VRAM)."""
        import subprocess
        if popen is None:
            from proc_util import popen_hidden as popen
        log_path = bc.LOG_DIR / f"{Path(self.args.out).stem}_llamacpp_{re.sub(r'[^A-Za-z0-9]+', '_', tag)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [bc.PINNED["llama_server_exe"], "-m", str(blob), "--port", str(LLAMA_PORT), "--log-file",
                str(log_path)]
        if self.args.llama_server_exe:
            argv[0] = self.args.llama_server_exe
        proc = popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        started, props = False, None
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=5) as r:
                        if json.loads(r.read()).get("status") == "ok":
                            started = True
                            break
                except Exception:
                    pass
                time.sleep(2)
            if started:
                with urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/props", timeout=30) as r:
                    props = json.loads(r.read())
            mem = bc.gpu_memory(self.run)
        finally:
            bc.kill_pid(proc.pid, self.run)
            t0 = time.monotonic()
            while proc.poll() is None and time.monotonic() - t0 < 30:
                time.sleep(0.5)
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        dgs = (props or {}).get("default_generation_settings") or {}
        self.emit({"record": "blade_k1_llamacpp_defaults", "model_tag": tag, "gguf": str(blob), "argv": argv,
                   "started": started, "exit_confirmed": proc.poll() is not None,
                   "props_n_ctx": dgs.get("n_ctx"), "props_total_slots": (props or {}).get("total_slots"),
                   "log": parse_llama_server_log(text), "log_path": str(log_path), **mem})


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--ollama-exe", default=None, help="default: the pinned side-by-side 0.34.4 install")
    ap.add_argument("--llama-server-exe", default=None)
    ap.add_argument("--skip-llamacpp", action="store_true")
    ap.add_argument("--dry-run-seconds", type=float, default=None)
    ap.add_argument("--allow-version-mismatch", action="store_true", help="dry runs only")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    dry = args.dry_run_seconds is not None
    if args.allow_version_mismatch and not dry:
        raise SystemExit("--allow-version-mismatch is for dry runs only")
    out = bc.out_path_ok(args.out, dry)
    clock = bc.DryRunClock(args.dry_run_seconds)
    emit = bc.jsonl_emitter(out, clock)

    def log(msg):
        print(f"[{bc.utc_iso()}] {msg}", flush=True)

    bc.require_blade()
    exe = args.ollama_exe or bc.PINNED["ollama_exe"]
    # every Ollama process (tray app, servers, runners) is stopped and confirmed gone BEFORE the version check, so
    # `<exe> --version` reports the pinned binary itself, never a server that was already running (night 1)
    versions, problems = bc.pinned_versions(exe)
    server = bc.LocalOllama(exe=exe, log_path=bc.LOG_DIR / f"{out.stem}.ollama_serve.log")
    rc, note = 0, "completed"
    try:
        emit({"record": "run_start", "job": "blade_k1", "argv": sys.argv, "dry_run": dry, "versions": versions,
              "version_problems": problems, "models": args.models.split(",")})
        if problems and not args.allow_version_mismatch:
            emit({"record": "refused", "reasons": problems})
            log(f"refused: {problems}")
            return 2
        import t2s_k1_ollama as k1
        server.start()
        if not server.wait_ready():
            raise RuntimeError("ollama did not become ready")
        served = server.server_version()
        sproblems = bc.server_version_problems(served)
        emit({"record": "blade_k1_server_version", "version": served, "problems": sproblems})
        if sproblems and not args.allow_version_mismatch:
            emit({"record": "refused", "reasons": sproblems})
            log(f"refused: {sproblems}")
            return 2
        client = k1.OllamaClient(timeout=CALL_TIMEOUT_S)
        job = K1Job(args, emit, log, client, server)
        time.sleep(2)
        dev = parse_ollama_server_log(server.log_path.read_text(encoding="utf-8", errors="replace"))
        emit({"record": "blade_k1_device", **dev, **bc.gpu_memory()})
        with urllib.request.urlopen(f"{bc.OLLAMA_URL}/api/tags", timeout=30) as r:
            avail = [m.get("name") for m in json.loads(r.read()).get("models", [])]
        present = []
        for tag in [m for m in args.models.split(",") if m]:
            if not job.model_present(tag, avail):
                emit({"record": "blade_k1_model_missing", "model_tag": tag, "available": avail})
                log(f"{tag}: not in the Ollama store, skipped")
                continue
            present.append(tag)
            try:
                job.run_model(tag)
            except bc.DryRunTimeUp:
                raise
            except Exception as ex:
                emit({"record": "blade_k1_model_failed", "model_tag": tag, "error": repr(ex)[:400]})
        dev2 = parse_ollama_server_log(server.log_path.read_text(encoding="utf-8", errors="replace"))
        emit({"record": "blade_k1_server_log_end", "offloaded": dev2["offloaded"],
              "inference_compute": dev2["inference_compute"]})
        server.stop()
        if not args.skip_llamacpp:
            import prompt_token_check as ptc
            for tag in present:
                job.llamacpp_defaults(tag, ptc.resolve_model_blob(bc.OLLAMA_STORE, tag))
    except bc.DryRunTimeUp as ex:
        note = f"dry run stopped: {ex}"
        log(note)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        note, rc = f"stopped: {ex!r}"[:400], 2
    finally:
        server.stop()
        for p in bc.harness_llama_servers(bc.list_processes()):
            if str(LLAMA_PORT) in p["cmd"]:
                bc.kill_pid(p["pid"])
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()] if out.exists() else []
        summary = summarize(rows)
        summary.update({"note": note, "dry_run": dry, "versions": versions})
        Path(str(out.with_suffix("")) + ".summary.json").write_text(json.dumps(summary, indent=1, default=str),
                                                                    encoding="utf-8")
        with open(out, "a", encoding="utf-8") as f:
            rec = "run_end" if (rc == 0 and not dry and note == "completed") else "run_stopped"
            f.write(json.dumps({"record": rec, "note": note, "ts_utc": bc.utc_iso()}) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
