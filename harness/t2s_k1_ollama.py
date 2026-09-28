"""K1: quality vs available memory through runtime policy (Ollama's automatic context/GPU-memory choices).

Question: does Ollama's own runtime policy (its default num_ctx choice, and how that choice changes when GPU memory
is already partly occupied) leave enough context headroom for realistic agent prompts, and does forcing a smaller
context onto llama-server directly (the same context Ollama picked) cost quality once a prompt is actually truncated.

Three phases, each independently resumable:

  1. tier          one Ollama chat call with no num_ctx override, then read `ollama ps` and (best effort) the Ollama
                   server log to see what context size and GPU memory figures it reports for the model's assigned
                   tier. See TIER DETECTION below.
  2. memory        start a llama-server occupying a target amount of GPU memory (t2s_lab.Server + Telemetry confirm
                   it), then repeat the tier probe against the Ollama model with that memory already in use.
  3. curves        for prompt lengths 3k/6k/12k/24k/48k/96k tokens, agent-layout prompts (rules+schema first, task
                   last), three arms: (a) Ollama default num_ctx, (b) Ollama with num_ctx sized to fit the prompt,
                   (c) llama-server directly with -c set to whatever Ollama's default context was. Reused/whether
                   arm (c) truncates is the load-bearing measurement for kill_criteria check (b).

kill_criteria(rows) implements exactly two checks (see its docstring) and main() prints/logs its result FIRST, before
any other summary, matching the "kill criterion reported before results" convention used elsewhere in this repo
(see harness/t2s_night2.py, harness/t2s_overnight.py).

This module is a DESIGN + CODE deliverable, written and tested without SSH access to evo-t2s or evo-x2. It has never
been run against a live Ollama server or a live BIOS-memory-limited iGPU, so several pieces are explicitly marked
TODO/ASSUMPTION below and must be confirmed against the real machines before any result from this script is trusted.
In particular: the Ollama Windows log path, the exact `ollama ps` column layout, and the 10 percent quality
tolerance in kill_criteria's truncation check.

CLI usage (never attempts SSH; run this directly on the target machine):
  python t2s_k1_ollama.py --phase tier,memory,curves --host evo-t2s --ollama-model qwen3:8b \
      --gguf-model qwen3-8b --memory-targets-gb 30

  python t2s_k1_ollama.py --phase tier,memory,curves --host evo-x2 --ollama-model qwen3:8b \
      --gguf-model qwen3-8b --memory-targets-gb 40,70

BIOS sweep hook: after changing the iGPU-memory BIOS setting at the keyboard, re-run only the tier and curve phases
against the same run (phases that do not depend on the BIOS-fixed memory ceiling, i.e. not the memory-pressure
phase, which reflects a runtime lock rather than the BIOS ceiling) with one command:
  python t2s_k1_ollama.py --phase tier,curves --host evo-x2 --ollama-model qwen3:8b --gguf-model qwen3-8b \
      --resume <prior-run-stem>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
import t2s_lab as L  # noqa: E402  (Server, Telemetry, ModelInfo, Jsonl, utc_iso, log, ps, avail_mb)
import run_provenance as rp  # noqa: E402

# ---------------------------------------------------------------------------------------------------- host config
# harness/host_config.py is the real source of truth for hw_id/user/python_exe/deploy_dir/models_dir/gpu_vendor/
# ssh_host (see require_host() there). It has no opinion on K1-specific policy values (which backend this script's
# occupying llama-server should use, how much GPU memory to occupy per host, or the native context to pass to
# ModelInfo when none is given), so K1_HOST_EXTRAS supplies just those, merged onto the real host_config dict by
# hw_id. Nothing here opens an SSH connection; --host only selects local config for a run that happens ON that
# machine.
import host_config as _hc  # noqa: E402

K1_HOST_EXTRAS = {
    "evo-t2s": {"backend": "vulkan", "memory_targets_gb": [30], "max_ctx_native": 32768},
    "evo-x2": {"backend": "vulkan", "memory_targets_gb": [40, 70], "max_ctx_native": 32768},
}


def require_host(name: str) -> dict:
    cfg = dict(_hc.require_host(name))
    cfg.setdefault("name", cfg["hw_id"])
    cfg.update({k: v for k, v in K1_HOST_EXTRAS.get(cfg["hw_id"], {}).items() if k not in cfg})
    return cfg


# ---------------------------------------------------------------------------------------------------- quality_suite
# quality_suite.py now exists (harness/quality_suite.py). qs_build_task/qs_score/qs_classify below normalize its
# Task-dataclass-based API (build_task -> Task, score_task(scorer_name, ...) -> (score, detail),
# classify_fabrication_or_refusal(output) -> str) down to the (prompt, expected, scorer) + plain-float-score shape
# this module's call sites use, since K1 drives Ollama's HTTP API directly for arms a/b (not quality_suite.run_task's
# own srv.chat flow). Delete the stub fallback branch once harness/quality_suite.py is guaranteed present.
try:
    import quality_suite as _qs  # type: ignore  # noqa: E402
    QUALITY_SUITE_SOURCE = "harness.quality_suite"

    def qs_build_task(task_type, target_tokens, seed):
        """Normalizes quality_suite.build_task's Task dataclass down to the (prompt, expected, scorer) triple this
        module's call sites need -- K1 drives Ollama's HTTP API directly for arms a/b (not srv.chat), so it cannot
        use quality_suite.run_task's own srv-calling flow and needs the prompt/expected/scorer split out instead."""
        t = _qs.build_task(task_type, target_tokens, seed)
        return t.prompt, t.expected, t.scorer

    def qs_score(scorer, output, expected):
        if output is None:
            return None
        score, _detail = _qs.score_task(scorer, output, expected)
        return score

    def qs_classify(output):
        return _qs.classify_fabrication_or_refusal(output)

except ImportError:
    QUALITY_SUITE_SOURCE = "stub in t2s_k1_ollama.py (harness/quality_suite.py not present yet)"

    _REFUSAL_RE = re.compile(
        r"\b(i (can'?t|cannot|don'?t have|do not have|am unable)|no (secret|code|value) (was|is) (provided|found)|"
        r"not (provided|available) in the (context|text|document))\b", re.I)

    def qs_build_task(task_type, target_tokens, seed):
        """Approx target_tokens using a words-per-token heuristic (~0.75 tokens/word for English prose); this is
        only a stub for local testing and is not the real quality_suite generator, so exact lengths are not
        load-bearing here. Layout is agent-style: schema/rules block first, the needle buried in filler, the task
        instruction last."""
        rng = random.Random(seed)
        needle = f"{rng.randint(100000, 999999)}"
        words_needed = max(int(target_tokens / 0.75) - 80, 40)
        filler_words = []
        pos = rng.randrange(words_needed)
        for i in range(words_needed):
            if i == pos:
                filler_words.append(f"[[SECRET_CODE:{needle}]]")
            else:
                filler_words.append(f"note{i % 997}")
        filler = " ".join(filler_words)
        schema = ("<schema>Respond with only the six-digit secret code, no other text.</schema>\n"
                  "<rules>Read the log below. Exactly one entry contains a token shaped like "
                  "[[SECRET_CODE:NNNNNN]]. That is the value to report.</rules>\n")
        task = "\n\nTASK: what is the secret code embedded in the log above? Reply with only the digits."
        prompt = schema + filler + task
        return prompt, needle, "stub_exact"

    def qs_score(scorer, output, expected):
        if not output:
            return 0.0
        return 1.0 if str(expected) in output else 0.0

    def qs_classify(output):
        if not output or not output.strip():
            return "refusal"
        if _REFUSAL_RE.search(output):
            return "refusal"
        return "fabrication"


# ---------------------------------------------------------------------------------------------------- Ollama client
OLLAMA_EXE = "ollama"
DEFAULT_OLLAMA_PORT = 11434

# TODO/ASSUMPTION: unverified against a live machine. Ollama for Windows (both the tray app and the service
# install) is documented upstream to log to %LOCALAPPDATA%\Ollama\server.log; some installs instead use
# %USERPROFILE%\.ollama\logs\server.log (the layout used on Linux/macOS installs, sometimes mirrored on Windows
# when OLLAMA_MODELS/OLLAMA_HOME env vars are customized). Confirm which applies on evo-t2s and evo-x2 with
# `Get-ChildItem` before trusting parse_ollama_log_context's output; fix the candidate list here if neither matches.
OLLAMA_LOG_PATH_CANDIDATES = [
    os.path.expandvars(r"%LOCALAPPDATA%\Ollama\server.log"),
    os.path.expandvars(r"%USERPROFILE%\.ollama\logs\server.log"),
]


def find_ollama_log():
    for p in OLLAMA_LOG_PATH_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


# TODO/ASSUMPTION: the exact log line formats below (num_ctx=, n_ctx=, "total"/"free" memory in MiB near a model
# load) are best-effort guesses at what an Ollama server.log looks like around a model load; they have not been
# checked against a real log. parse_ollama_log_context records every matched raw line so a human can correct the
# regexes after the first real run, rather than silently trusting a wrong parse.
_LOG_NUM_CTX_RE = re.compile(r"\b(?:num_ctx|n_ctx|context length)\D{0,3}(\d{3,7})", re.I)
_LOG_MEM_RE = re.compile(r"\b(total|free|available)\b[^0-9]{0,10}(\d+(?:\.\d+)?)\s*(MiB|MB|GiB|GB)", re.I)


def parse_ollama_log_context(path, since_pos=0):
    """Best-effort scan of the Ollama server log, starting at byte offset since_pos, for the context size and GPU
    memory figures it reports at model load. Returns {"available": bool, "num_ctx_seen": [...], "mem_seen": [...],
    "matched_lines": [...]}. Never raises; a log that cannot be read or parsed yields available=False with why."""
    if not path or not os.path.exists(path):
        return {"available": False, "why": "no log path found (see OLLAMA_LOG_PATH_CANDIDATES TODO)"}
    try:
        with open(path, "rb") as f:
            f.seek(max(since_pos, 0))
            text = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        return {"available": False, "why": f"could not read log: {e!r}"}
    num_ctx_seen = [int(m.group(1)) for m in _LOG_NUM_CTX_RE.finditer(text)]
    mem_seen = [{"kind": m.group(1).lower(), "value": float(m.group(2)), "unit": m.group(3)} for m in _LOG_MEM_RE.finditer(text)]
    matched = [l.strip()[:300] for l in text.splitlines() if _LOG_NUM_CTX_RE.search(l) or _LOG_MEM_RE.search(l)][-20:]
    return {"available": True, "num_ctx_seen": num_ctx_seen, "mem_seen": mem_seen, "matched_lines": matched}


def parse_ollama_ps(text):
    """Parse `ollama ps` table output. TODO/ASSUMPTION: unverified against a live install. The header-driven split
    is meant to survive `ollama` adding/removing/reordering columns (observed historically: NAME ID SIZE PROCESSOR
    UNTIL; newer builds may add a CONTEXT column) as long as columns stay separated by 2+ spaces, which is how
    Go's text/tabwriter (used by the ollama CLI) formats it. Confirm the header row on both machines before relying
    on context_len; if CONTEXT is absent, context_len is always None and phase rows record ollama_ps_raw for a
    human to reconcile against the log-based figure instead."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []
    header = [h.strip().upper() for h in re.split(r"\s{2,}", lines[0].strip())]
    rows = []
    for line in lines[1:]:
        fields = re.split(r"\s{2,}", line.strip())
        if len(fields) < len(header):
            fields += [None] * (len(header) - len(fields))
        d = dict(zip(header, fields))
        ctx = d.get("CONTEXT")
        ctx_n = None
        if ctx:
            m = re.search(r"\d+", ctx)
            ctx_n = int(m.group(0)) if m else None
        rows.append({"name": d.get("NAME"), "id": d.get("ID"), "size": d.get("SIZE"),
                     "processor": d.get("PROCESSOR"), "context_len": ctx_n, "until": d.get("UNTIL"), "raw": d})
    return rows


def run_ollama_ps(ollama_exe=OLLAMA_EXE, timeout=30):
    try:
        r = subprocess.run([ollama_exe, "ps"], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return parse_ollama_ps(r.stdout)
    except Exception as e:
        return []


class OllamaClient:
    """Thin client for the Ollama HTTP API (/api/chat). Durations are always monotonic; never a time.time() diff."""

    def __init__(self, host="127.0.0.1", port=DEFAULT_OLLAMA_PORT, timeout=1800):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def chat(self, model, prompt, num_ctx=None, max_tokens=64):
        options = {"num_predict": max_tokens, "temperature": 0, "seed": 42}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": options}
        req = urllib.request.Request(f"{self.base}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                status = r.status
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"outcome": "http_error", "status": e.code, "error": e.read().decode(errors="replace")[:400],
                    "duration_s": time.monotonic() - t0, "num_ctx_requested": num_ctx}
        except Exception as e:
            return {"outcome": "error", "status": None, "error": str(e)[:400], "duration_s": time.monotonic() - t0,
                    "num_ctx_requested": num_ctx}
        dt = time.monotonic() - t0
        return {"outcome": "ok", "status": status, "message": (data.get("message") or {}).get("content"),
                "prompt_eval_count": data.get("prompt_eval_count"), "eval_count": data.get("eval_count"),
                "duration_s": dt, "num_ctx_requested": num_ctx, "raw": data}


# ---------------------------------------------------------------------------------------------------- lab scaffold
class K1Lab:
    """Owns the output jsonl, provenance, host config, and resumption state. Not the Server/Telemetry lifecycle for
    the occupying process, which each phase manages itself in a finally block."""

    def __init__(self, args, host_cfg, prov):
        self.args, self.host_cfg, self.prov = args, host_cfg, prov
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stem = args.resume or f"t2s_k1_ollama_{host_cfg['name']}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        self.prefix = str(self.out_dir / self.stem)
        self.rows_path = self.prefix + ".jsonl"
        self.rows = L.Jsonl(self.rows_path)
        self.ollama = OllamaClient(port=args.ollama_port)
        self.done_phases = set()
        if args.resume and Path(self.rows_path).exists():
            for line in open(self.rows_path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("record") == "phase_done":
                    self.done_phases.add(r["phase"])

    def emit(self, row):
        row.setdefault("ts_utc", L.utc_iso())
        row.setdefault("host", self.host_cfg["name"])
        row.setdefault("gpu_vendor", self.host_cfg["gpu_vendor"])
        self.rows.write(row)
        return row

    def phase_done(self, phase):
        self.done_phases.add(phase)
        self.rows.write({"record": "phase_done", "phase": phase, "ts_utc": L.utc_iso()})

    def all_rows(self):
        if not Path(self.rows_path).exists():
            return []
        out = []
        for line in open(self.rows_path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out


TIER_PROBE_PROMPT = ("You are a helpful assistant. In one short sentence, name the capital of France.")


# ---------------------------------------------------------------------------------------------------- phase: tier
def phase_tier(lab: K1Lab, ollama_model: str, rep: int = 0, ollama=None, ps_fn=run_ollama_ps, log_finder=find_ollama_log):
    """Issue one Ollama chat with no num_ctx override, then read `ollama ps` and the Ollama server log for the
    context size and GPU memory it reports for this model's assigned tier. Records both to a row and returns it."""
    ollama = ollama or lab.ollama
    log_path = log_finder()
    log_pos = os.path.getsize(log_path) if log_path and os.path.exists(log_path) else 0
    res = ollama.chat(ollama_model, TIER_PROBE_PROMPT, num_ctx=None, max_tokens=32)
    ps_rows = ps_fn()
    match = next((r for r in ps_rows if r.get("name") == ollama_model or (r.get("name") or "").startswith(ollama_model.split(":")[0])), None)
    log_info = parse_ollama_log_context(log_path, log_pos) if log_path else {"available": False, "why": "no log path found"}
    log_ctx_candidates = log_info.get("num_ctx_seen") or []
    default_ctx = match.get("context_len") if match else None
    if default_ctx is None and log_ctx_candidates:
        default_ctx = max(log_ctx_candidates)
    row = lab.emit({
        "record": "tier", "phase": "tier", "model_tag": ollama_model, "rep": rep,
        "chat_outcome": res.get("outcome"), "chat_status": res.get("status"), "chat_error": res.get("error"),
        "chat_duration_s": res.get("duration_s"), "chat_prompt_eval_count": res.get("prompt_eval_count"),
        "ollama_ps_raw": match, "ollama_default_ctx": default_ctx, "ollama_log_path": log_path,
        "ollama_log_info": log_info, "mem_headroom_gb": None,
    })
    return row


# ---------------------------------------------------------------------------------------------------- phase: memory
def phase_memory_pressure(lab: K1Lab, ollama_model: str, occupier_mi, target_gb: float, occupier_n_ctx: int,
                          rep: int = 0, ollama=None, ps_fn=run_ollama_ps, log_finder=find_ollama_log,
                          server_factory=None, telemetry=None, avail_mb_fn=None, tolerance_gb: float = 2.0):
    """Start a llama-server (t2s_lab.Server-shaped) holding occupier_mi at occupier_n_ctx to occupy about target_gb
    of GPU memory, confirm via Telemetry/avail_mb that it actually holds that much, then repeat the tier probe
    against the Ollama model with that memory already in use. The occupying server is stopped in a finally block
    regardless of outcome. tolerance_gb follows the +-2x-tolerance-band pattern used by t2s_lab.Balloon for the
    same kind of "did we actually reach the target" check."""
    ollama = ollama or lab.ollama
    avail_mb_fn = avail_mb_fn or L.avail_mb
    server_factory = server_factory or (lambda mi, n_ctx, tag: L.Server(lab, mi, n_ctx, backend=lab.host_cfg.get("backend", "vulkan"), tag=tag))
    pre_avail_mb = avail_mb_fn()
    srv = server_factory(occupier_mi, occupier_n_ctx, f"k1_mem_{int(target_gb)}gb_{rep}")
    lab.resources_occupier = srv
    t_pre = time.time()
    info = srv.start()
    if not info.get("ok"):
        row = lab.emit({"record": "memory_pressure", "phase": "memory", "model_tag": ollama_model, "rep": rep,
                        "target_gb": target_gb, "occupier_n_ctx": occupier_n_ctx, "occupier_ok": False,
                        "occupier_error": info.get("error"), "held_gb": None, "held_confirmed": False,
                        "ollama_default_ctx": None})
        return row
    try:
        time.sleep(4)
        t_post = time.time()
        post_avail_mb = avail_mb_fn()
        held_gb = (pre_avail_mb - post_avail_mb) / 1024.0
        tele_shared_mib = None
        if telemetry is not None:
            try:
                m = telemetry.metrics(t_pre, t_post + 1.0)
                tele_shared_mib = m.get("shared_usage_mib")
            except Exception:
                tele_shared_mib = None
        held_confirmed = held_gb >= (target_gb - tolerance_gb)
        log_path = log_finder()
        log_pos = os.path.getsize(log_path) if log_path and os.path.exists(log_path) else 0
        res = ollama.chat(ollama_model, TIER_PROBE_PROMPT, num_ctx=None, max_tokens=32)
        ps_rows = ps_fn()
        match = next((r for r in ps_rows if r.get("name") == ollama_model or (r.get("name") or "").startswith(ollama_model.split(":")[0])), None)
        log_info = parse_ollama_log_context(log_path, log_pos) if log_path else {"available": False, "why": "no log path found"}
        default_ctx = match.get("context_len") if match else None
        if default_ctx is None and log_info.get("num_ctx_seen"):
            default_ctx = max(log_info["num_ctx_seen"])
        row = lab.emit({
            "record": "memory_pressure", "phase": "memory", "model_tag": ollama_model, "rep": rep,
            "target_gb": target_gb, "occupier_n_ctx": occupier_n_ctx, "occupier_ok": True,
            "occupier_load_s": info.get("load_s"), "pre_avail_mb": pre_avail_mb, "post_avail_mb": post_avail_mb,
            "held_gb": held_gb, "held_confirmed": held_confirmed, "telemetry_shared_usage_mib": tele_shared_mib,
            "chat_outcome": res.get("outcome"), "chat_status": res.get("status"),
            "chat_prompt_eval_count": res.get("prompt_eval_count"), "ollama_ps_raw": match,
            "ollama_default_ctx": default_ctx, "ollama_log_path": log_path, "ollama_log_info": log_info,
            "mem_headroom_gb": -target_gb,
        })
        return row
    finally:
        srv.stop()
        lab.resources_occupier = None


# ---------------------------------------------------------------------------------------------------- phase: curves
PROMPT_LENGTHS_TOKENS = (3000, 6000, 12000, 24000, 48000, 96000)
# Must be real quality_suite.TASK_TYPES names once that module is importable (qs_build_task dispatches on them
# directly); "needle_recall" only exists in the stub path. Default to one real task type (niah_multikey) so a plain
# `--phase curves` run works out of the box against the real suite; pass --task-types to run more of them.
TASK_TYPES = (_qs.TASK_TYPES[0],) if QUALITY_SUITE_SOURCE == "harness.quality_suite" else ("needle_recall",)


def _fit_num_ctx(length_tokens, headroom_tokens=1024, round_to=1024):
    n = length_tokens + headroom_tokens + 256  # +256 for the reply
    return int(-(-n // round_to)) * round_to


def phase_quality_curves(lab: K1Lab, ollama_model: str, gguf_mi, default_ctx: int, rep_count: int = 3,
                         lengths=PROMPT_LENGTHS_TOKENS, task_types=TASK_TYPES, seed: int = 20260928, ollama=None,
                         server_factory=None, max_tokens: int = 64):
    """Arm (a) Ollama default (no num_ctx), arm (b) Ollama with num_ctx fit to the prompt, arm (c) llama-server
    directly with -c == default_ctx (Ollama's own default, established once for the whole phase by the caller from
    an earlier phase_tier row: see build_default_ctx_from_rows). Three reps per condition. default_ctx is a
    parameter rather than re-derived here so the fixed-context arm (c) baseline is the SAME number across every
    prompt length in this phase, matching the task spec ("-c equal to whatever Ollama's default context was")."""
    ollama = ollama or lab.ollama
    server_factory = server_factory or (lambda mi, n_ctx, tag: L.Server(lab, mi, n_ctx, backend=lab.host_cfg.get("backend", "vulkan"), tag=tag))
    rows = []
    for length in lengths:
        for task_type in task_types:
            for rep in range(rep_count):
                prompt, expected, scorer = qs_build_task(task_type, length, seed + rep + length)

                ra = ollama.chat(ollama_model, prompt, num_ctx=None, max_tokens=max_tokens)
                score_a = qs_score(scorer, ra.get("message"), expected) if ra.get("outcome") == "ok" else None
                rows.append(lab.emit({
                    "record": "curve", "phase": "curves", "arm": "a", "model_tag": ollama_model, "task_type": task_type,
                    "prompt_len_target": length, "rep": rep, "num_ctx_requested": None,
                    "chat_outcome": ra.get("outcome"), "http_status": ra.get("status"),
                    "prompt_eval_count": ra.get("prompt_eval_count"), "score": score_a, "errored": ra.get("outcome") != "ok",
                    "fabrication_or_refusal": (qs_classify(ra.get("message"))
                                               if score_a is not None and score_a < 1.0 else None),
                }))

                fit_ctx = _fit_num_ctx(length)
                rb = ollama.chat(ollama_model, prompt, num_ctx=fit_ctx, max_tokens=max_tokens)
                score_b = qs_score(scorer, rb.get("message"), expected) if rb.get("outcome") == "ok" else None
                rows.append(lab.emit({
                    "record": "curve", "phase": "curves", "arm": "b", "model_tag": ollama_model, "task_type": task_type,
                    "prompt_len_target": length, "rep": rep, "num_ctx_requested": fit_ctx,
                    "chat_outcome": rb.get("outcome"), "http_status": rb.get("status"),
                    "prompt_eval_count": rb.get("prompt_eval_count"), "score": score_b, "errored": rb.get("outcome") != "ok",
                    "fabrication_or_refusal": (qs_classify(rb.get("message"))
                                               if score_b is not None and score_b < 1.0 else None),
                }))

                srv = server_factory(gguf_mi, default_ctx, f"k1_curve_{length}_{rep}")
                info = srv.start()
                if not info.get("ok"):
                    rows.append(lab.emit({
                        "record": "curve", "phase": "curves", "arm": "c", "model_tag": ollama_model, "task_type": task_type,
                        "prompt_len_target": length, "rep": rep, "num_ctx_requested": default_ctx, "chat_outcome": "start_error",
                        "http_status": None, "prompt_eval_count": None, "score": None, "errored": True,
                        "server_error": info.get("error"), "fabrication_or_refusal": None,
                    }))
                    continue
                try:
                    n_tok = srv.tokenize(prompt)
                    res = srv.chat(prompt, max_tokens, ignore_eos=False)
                    outcome = res.get("outcome")
                    errored = outcome != "ok"
                    score_c = qs_score(scorer, res.get("output"), expected) if not errored else None
                    rows.append(lab.emit({
                        "record": "curve", "phase": "curves", "arm": "c", "model_tag": ollama_model, "task_type": task_type,
                        "prompt_len_target": length, "rep": rep, "num_ctx_requested": default_ctx,
                        "prompt_tokens_tokenized": n_tok, "chat_outcome": outcome, "chat_error": res.get("error"),
                        "prompt_eval_count": n_tok if not errored else None, "score": score_c, "errored": errored,
                        "fabrication_or_refusal": (qs_classify(res.get("output"))
                                                   if score_c is not None and score_c < 1.0 else None),
                    }))
                finally:
                    srv.stop()
    return rows


def build_default_ctx_from_rows(rows, host=None):
    """Pull the most recent ollama_default_ctx from tier/memory_pressure rows (optionally filtered to one host),
    for wiring phase_quality_curves' fixed arm-(c) baseline without re-running phase_tier."""
    cands = [r for r in rows if r.get("record") in ("tier", "memory_pressure") and r.get("ollama_default_ctx") is not None
             and (host is None or r.get("host") == host)]
    return cands[-1]["ollama_default_ctx"] if cands else None


# ---------------------------------------------------------------------------------------------------- kill criteria
KILL_CRITERIA_MIN_DEFAULT_CTX = 48000
# 10% tolerance: chosen as a round number comfortably larger than ordinary greedy-decode scoring noise on a
# deterministic (temperature 0, fixed seed) single-fact recall task, while still catching a real collapse in
# quality from truncation (which, for a needle-recall task, tends to be near-total: the needle is simply gone).
# Revisit this number once real run-to-run variance for the actual quality_suite tasks has been measured.
KILL_CRITERIA_TRUNCATION_TOLERANCE = 0.10


def kill_criteria(rows: list) -> dict:
    """Exactly two named checks, matching the task spec verbatim:

    (a) "defaults cover realistic agent prompts (<= 48k) on both machines": Ollama's default num_ctx choice (from
        phase_tier / phase_memory_pressure rows) must be >= 48000 on every host tested.
    (b) "truncated cases keep their quality": for every curve row where arm (c) (llama-server with -c == Ollama's
        default) either errored or came back with prompt_eval_count less than the intended prompt length
        (truncation happened), the per-task score for that row must not be more than
        KILL_CRITERIA_TRUNCATION_TOLERANCE below the matched arm-(b) score at the same prompt length, task and rep.

    Returns {"defaults_cover_agent_prompts": {"ok": bool, "reason": str}, "truncated_cases_keep_quality": {...}}.
    """
    out = {}

    tier_rows = [r for r in rows if r.get("record") in ("tier", "memory_pressure") and r.get("ollama_default_ctx") is not None]
    if not tier_rows:
        out["defaults_cover_agent_prompts"] = {"ok": False, "reason": "no tier/memory_pressure rows with ollama_default_ctx found"}
    else:
        hosts = sorted({r.get("host") for r in tier_rows})
        fails = [r for r in tier_rows if r["ollama_default_ctx"] < KILL_CRITERIA_MIN_DEFAULT_CTX]
        if fails:
            worst = min(fails, key=lambda r: r["ollama_default_ctx"])
            out["defaults_cover_agent_prompts"] = {"ok": False, "reason": (
                f"{worst.get('host')} default num_ctx {worst['ollama_default_ctx']} < {KILL_CRITERIA_MIN_DEFAULT_CTX} "
                f"(hosts checked: {hosts})")}
        else:
            out["defaults_cover_agent_prompts"] = {"ok": True, "reason": (
                f"default num_ctx >= {KILL_CRITERIA_MIN_DEFAULT_CTX} on all {len(hosts)} host(s) checked ({hosts})")}

    curve_rows = [r for r in rows if r.get("record") == "curve"]
    by_key = {}
    for r in curve_rows:
        key = (r.get("prompt_len_target"), r.get("task_type"), r.get("rep"))
        by_key.setdefault(key, {})[r.get("arm")] = r
    checked, failures = 0, []
    for key, arms in by_key.items():
        c, b = arms.get("c"), arms.get("b")
        if c is None or b is None:
            continue
        length = key[0]
        truncated = bool(c.get("errored")) or (c.get("prompt_eval_count") is not None and length is not None
                                               and c["prompt_eval_count"] < length)
        if not truncated:
            continue
        checked += 1
        score_c, score_b = c.get("score"), b.get("score")
        if score_b is None:
            failures.append((key, "arm b has no score to compare against"))
        elif score_c is None or score_c < score_b - KILL_CRITERIA_TRUNCATION_TOLERANCE:
            failures.append((key, f"arm c score {score_c} more than {KILL_CRITERIA_TRUNCATION_TOLERANCE:.0%} below arm b score {score_b}"))
    if checked == 0:
        out["truncated_cases_keep_quality"] = {"ok": False, "reason": "no truncated arm-c case with a matched arm-b row was found to check"}
    elif failures:
        detail = "; ".join(f"len={k[0]} task={k[1]} rep={k[2]}: {why}" for k, why in failures[:5])
        out["truncated_cases_keep_quality"] = {"ok": False, "reason": f"{len(failures)}/{checked} truncated cases dropped quality: {detail}"}
    else:
        out["truncated_cases_keep_quality"] = {"ok": True, "reason": f"all {checked} truncated case(s) stayed within {KILL_CRITERIA_TRUNCATION_TOLERANCE:.0%} of the matched arm-b score"}

    return out


def report_kill_criteria(result: dict):
    """Printed/logged FIRST, before any other summary, in main(). Matches the repo convention of reporting the kill
    criterion ahead of results (see t2s_night2.py / t2s_overnight.py estimate + schedule logging)."""
    L.log("=== KILL CRITERIA (checked first) ===")
    for name, r in result.items():
        status = "OK" if r["ok"] else "FAIL"
        L.log(f"[{status}] {name}: {r['reason']}")
    L.log("=== end kill criteria ===")


# ---------------------------------------------------------------------------------------------------- CLI
def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="tier,memory,curves", help="comma-separated subset of tier,memory,curves")
    ap.add_argument("--host", required=True, choices=sorted(K1_HOST_EXTRAS), help="which local machine's config to use (never used to open an SSH connection)")
    ap.add_argument("--ollama-model", required=True, help="Ollama model tag, already pulled")
    ap.add_argument("--gguf-model", default=None, help="key into MODEL_FILES-style gguf path lookup for the llama-server arms; required for memory/curves phases")
    ap.add_argument("--gguf-path", default=None, help="explicit gguf path, overrides --gguf-model resolution against host_cfg models_dir")
    ap.add_argument("--gguf-sha256", default=None)
    ap.add_argument("--memory-targets-gb", default=None, help="comma-separated GB targets; defaults to host_cfg['memory_targets_gb']")
    ap.add_argument("--occupier-n-ctx", type=int, default=32768, help="context size for the occupying llama-server in the memory phase")
    ap.add_argument("--default-ctx", type=int, default=None, help="fixed arm-(c) context for the curves phase; if omitted, pulled from the most recent tier/memory_pressure row for this host")
    ap.add_argument("--ollama-port", type=int, default=DEFAULT_OLLAMA_PORT)
    ap.add_argument("--out-dir", default=str(DEPLOY.parent / "results"))
    ap.add_argument("--resume", default=None, help="prior run stem to resume/extend (see BIOS sweep hook in the module docstring)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--require-committed", action="store_true", help="refuse to run unless this script is committed (standing rule; off by default while this file is new/under review)")
    return ap


def resolve_gguf_mi(args, host_cfg):
    if args.gguf_path:
        path = args.gguf_path
    elif args.gguf_model:
        path = str(Path(host_cfg["models_dir"]) / args.gguf_model)
    else:
        return None
    return L.ModelInfo(args.gguf_model or Path(path).stem, path, args.gguf_sha256 or "unknown", False, host_cfg.get("max_ctx_native", 32768))


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    host_cfg = require_host(args.host)
    prov = rp.script_provenance([__file__], require_committed=args.require_committed)
    lab = K1Lab(args, host_cfg, prov)
    Path(lab.prefix + "_manifest.json").write_text(json.dumps({
        "launch_utc": L.utc_iso(), "host": host_cfg, "args": vars(args), "script_provenance": prov,
        "quality_suite_source": QUALITY_SUITE_SOURCE,
    }, indent=1, default=str), encoding="utf-8")

    phases = [p.strip() for p in args.phase.split(",") if p.strip()]
    targets = [float(x) for x in args.memory_targets_gb.split(",")] if args.memory_targets_gb else host_cfg.get("memory_targets_gb", [])
    gguf_mi = resolve_gguf_mi(args, host_cfg)

    if "tier" in phases and "phase_tier" not in lab.done_phases:
        for rep in range(args.reps):
            phase_tier(lab, args.ollama_model, rep=rep)
        lab.phase_done("phase_tier")

    if "memory" in phases and "phase_memory" not in lab.done_phases:
        if gguf_mi is None:
            L.log("memory phase requested but no --gguf-model/--gguf-path given; skipping")
        else:
            for target_gb in targets:
                for rep in range(args.reps):
                    phase_memory_pressure(lab, args.ollama_model, gguf_mi, target_gb, args.occupier_n_ctx, rep=rep)
            lab.phase_done("phase_memory")

    if "curves" in phases and "phase_curves" not in lab.done_phases:
        if gguf_mi is None:
            L.log("curves phase requested but no --gguf-model/--gguf-path given; skipping")
        else:
            default_ctx = args.default_ctx or build_default_ctx_from_rows(lab.all_rows(), host=host_cfg["name"])
            if default_ctx is None:
                L.log("curves phase requested but no default_ctx known (run --phase tier first, or pass --default-ctx); skipping")
            else:
                phase_quality_curves(lab, args.ollama_model, gguf_mi, default_ctx, rep_count=args.reps)
                lab.phase_done("phase_curves")

    all_rows = lab.all_rows()
    kc = kill_criteria(all_rows)
    report_kill_criteria(kc)
    L.log(f"run complete: stem={lab.stem} rows={len(all_rows)} phases_run={phases}")
    return kc


if __name__ == "__main__":
    main()
