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

K1 v3 (tier_v3 phase): fixes the K1 v2 model-ceiling confound (see K1_V3_MODELS / phase_tier_v3 above). Pulls all
three K1_V3_MODELS first (nothing else measures concurrently with the pulls), then for each model runs phase_tier()
once and the 5-length empirical probe sweep (K1_V3_PROBE_LENGTHS). --ollama-model/--gguf-model/--gguf-path are
unused by this phase (the model list is fixed, from K1_V3_MODELS, not the CLI). Two per-machine invocations
(x2_k1_tier_v3 / t2s_k1_tier_v3):

  python t2s_k1_ollama.py --phase tier_v3 --host evo-x2 --ollama-model unused
  python t2s_k1_ollama.py --phase tier_v3 --host evo-t2s --ollama-model unused
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
import t2s_queue as tq  # noqa: E402

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
_LOG_NUM_CTX_RE = re.compile(r"\b(?:num_ctx|n_ctx|context length|--ctx-size)\D{0,3}(\d{3,7})", re.I)
# The runner start line Ollama's server.log writes when it launches its internal llama.cpp-style subprocess: the
# full command line, including --ctx-size N. This is a narrower, more specific match than _LOG_NUM_CTX_RE above
# (which also matches looser phrases like "context length" in an unrelated log line); matched separately so
# phase_tier can report which signal actually fired.
_LOG_RUNNER_CTX_RE = re.compile(r"--ctx-size[= ](\d+)")
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


# TODO/ASSUMPTION (2026-10-01, same status as the ctx/mem regexes above: best-effort guess, not yet checked
# against a real evo-t2s server.log): Ollama's server.log reports its GPU/compute-device discovery near
# startup with lines resembling "inference compute" (library=cuda/rocm/vulkan/cpu, name=<device>) when a GPU
# backend is found, or "no compatible GPUs were discovered" / "falling back to CPU" when none is. This exists
# for A2 (2026-10-01): if evo-t2s's Ollama install runs qwen/llama models on the CPU because it never detected
# the Intel Arc iGPU (a Vulkan backend-support gap, not a memory-tier decision), the observed small default
# context tier there must be described as "Ollama did not detect a usable GPU" in every doc that cites it, not
# as a memory-driven tier choice -- those are two different mechanisms with the same symptom (a small n_ctx),
# and conflating them would misattribute the cause.
_LOG_GPU_DETECT_RE = re.compile(
    r"(inference compute|looking for compatible gpus?|no compatible gpus? (?:were |was )?discovered|"
    r"falling back to cpu|library=\w+|gpu(?:s)? (?:is|are) (?:not )?(?:compatible|available|discovered))",
    re.I,
)


def parse_ollama_log_device(path, since_pos=0):
    """Best-effort scan of the Ollama server log for its own GPU/compute-device detection lines (see
    _LOG_GPU_DETECT_RE's comment for format caveats and why this exists). Returns {"available": bool,
    "matched_lines": [...], "compute_device_guess": "gpu"|"cpu"|"unknown"}. compute_device_guess is "cpu" only
    if a CPU-fallback phrase is seen and no "library=" value other than cpu appears; "gpu" if any non-cpu
    library= value or an "inference compute" line appears; "unknown" if nothing matched at all -- never raises."""
    if not path or not os.path.exists(path):
        return {"available": False, "why": "no log path found", "compute_device_guess": "unknown"}
    try:
        with open(path, "rb") as f:
            f.seek(max(since_pos, 0))
            text = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        return {"available": False, "why": f"could not read log: {e!r}", "compute_device_guess": "unknown"}
    matched = [l.strip()[:300] for l in text.splitlines() if _LOG_GPU_DETECT_RE.search(l)]
    lower_all = "\n".join(matched).lower()
    if not matched:
        guess = "unknown"
    elif "no compatible gpu" in lower_all or "falling back to cpu" in lower_all:
        guess = "cpu"
    elif re.search(r"library=(?!cpu\b)\w+", lower_all) or "inference compute" in lower_all:
        guess = "gpu"
    else:
        guess = "unknown"
    return {"available": True, "matched_lines": matched[-20:], "compute_device_guess": guess}


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

    def chat(self, model, prompt, num_ctx=None, max_tokens=64, keep_alive=None, messages=None, tools=None,
             timeout=None, think=None):
        """messages: optional full conversation history ([{"role":.., "content":..}, ...]), for a caller (e.g.
        harness/t2s_r2_session_growth.py's growing agent sessions) that needs more than the single user-turn
        request every other caller of this method sends. None (the default, and every pre-existing call site's
        behavior) still sends exactly [{"role": "user", "content": prompt}] as before; prompt is then ignored only
        when messages is given, and is still recorded by call sites for logging.

        tools: optional OpenAI-style tools array ([{"type": "function", "function": {"name", "description",
        "parameters"}}, ...]), forwarded to /api/chat's `tools` field so a caller can use Ollama's native
        tool-calling support instead of asking the model to describe tool calls in free text (added for
        harness/t2s_r2_session_growth.py's rule-2 native-vs-text-fallback tool-call detection). None (the
        default, and every pre-existing call site) omits the field entirely, unchanged from before. The
        response's own message.tool_calls (if any) is returned verbatim under the "tool_calls" key.

        timeout: optional per-call override of self.timeout (added 2026-10-01 for K1 v3's long-prompt probes,
        which can legitimately take longer than the 1800s instance default on a slow prefill -- three real
        calls on evo-x2 errored at gaps within 2s of exactly 1800s, a client timeout, not a runtime failure).
        None (the default, every pre-existing call site) keeps self.timeout, unchanged.

        think: optional, forwarded to /api/chat's top-level "think" field (Ollama's hybrid-reasoning-model
        toggle). Found live 2026-10-01: K1 v3's marker probe reported qwen3:8b losing the marker at every
        length including 16K/32K, where token math shows no truncation at all. Root cause, confirmed by
        dumping the raw response: qwen3:8b defaults to thinking mode, and the probe's own max_tokens=32
        budget was being spent entirely on the hidden message.thinking field, with message.content (what
        marker_used actually checks) left empty every time (done_reason="length", reasoning never finished).
        think=False (verified live: real marker reproduced in 7 tokens, done_reason="stop") fixes this for
        any model where thinking is relevant; None (the default, every pre-existing call site) omits the
        field entirely, unchanged for models like qwen3-4b-instruct-2507 or llama3.1:8b that either have no
        thinking mode or already default to a non-thinking variant."""
        options = {"num_predict": max_tokens, "temperature": 0, "seed": 42}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        body = {"model": model, "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
                "stream": False, "options": options}
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        if tools is not None:
            body["tools"] = tools
        if think is not None:
            body["think"] = think
        req = urllib.request.Request(f"{self.base}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout if timeout is not None else self.timeout) as r:
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
                "tool_calls": (data.get("message") or {}).get("tool_calls"),
                "prompt_eval_count": data.get("prompt_eval_count"), "eval_count": data.get("eval_count"),
                "duration_s": dt, "num_ctx_requested": num_ctx, "raw": data}

    def get_ps(self):
        """GET /api/ps: every currently-loaded model, with every field Ollama returns (name, size, size_vram,
        expires_at, and context_length when the install exposes it -- confirmed 2026-09-29 that `ollama ps` (the
        CLI, parsed by parse_ollama_ps) returns nothing once OLLAMA_KEEP_ALIVE=0 unloads the model right after each
        call; this reads the same live state directly via HTTP instead of shelling out, and phase_tier calls it
        while the model is still deliberately kept alive (keep_alive="10m") so there is something to read)."""
        req = urllib.request.Request(f"{self.base}/api/ps")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            return {"outcome": "ok", "models": data.get("models") or []}
        except Exception as e:
            return {"outcome": "error", "error": str(e)[:400], "models": []}

    def unload(self, model):
        """Explicit unload: a chat call with keep_alive=0 and no real generation work (max_tokens=1), matching
        Ollama's own documented unload mechanism. Returns the chat() result; caller confirms via get_ps()."""
        return self.chat(model, "", num_ctx=None, max_tokens=1, keep_alive=0)

    def pull(self, tag, timeout=None):
        """POST /api/pull with stream=True: a real, blocking HTTP call, not a stub. Ollama streams one NDJSON
        object per line while it pulls (status progresses through e.g. "pulling manifest" -> "pulling <digest>"
        (with total/completed byte counts) -> "verifying sha256 digest" -> "success"; a failed pull instead sends a
        line with an "error" field). This reads the response line by line and returns as soon as a "success" status
        or an "error" is seen (or, if the connection closes with neither, once the stream ends), never assuming the
        pull finished just because the HTTP request returned. Returns
        {"outcome": "ok"|"error"|"incomplete"|"http_error", "final_status": str|None, "status_lines": [...]}."""
        body = {"model": tag, "stream": True}
        req = urllib.request.Request(f"{self.base}/api/pull", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        status_lines = []
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                for raw_line in r:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    status_lines.append(obj)
                    if obj.get("error"):
                        return {"outcome": "error", "final_status": obj.get("error"), "status_lines": status_lines}
                    if obj.get("status") == "success":
                        return {"outcome": "ok", "final_status": "success", "status_lines": status_lines}
            final = status_lines[-1].get("status") if status_lines else None
            return {"outcome": "ok" if final == "success" else "incomplete", "final_status": final,
                    "status_lines": status_lines}
        except urllib.error.HTTPError as e:
            return {"outcome": "http_error", "final_status": None, "status_lines": status_lines,
                    "error": e.read().decode(errors="replace")[:400]}
        except Exception as e:
            return {"outcome": "error", "final_status": None, "status_lines": status_lines, "error": str(e)[:400]}


def pull_model(ollama, tag):
    """Thin module-level wrapper around OllamaClient.pull, kept separate so call sites (and tests) can inject a
    fake pull_fn with this exact (ollama, tag) -> result shape, matching this file's dependency-injection
    convention elsewhere (ps_fn, log_finder, server_factory, ...)."""
    return ollama.pull(tag)


def _resolve_ollama_exe():
    """OLLAMA_BIN env var, then shutil.which("ollama"), then the Windows Ollama installer's own standard
    per-user install location (%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe), then the bare command name as a
    last resort.

    Found live 2026-10-01 (evo-t2s): both OLLAMA_BIN and shutil.which("ollama") failed even though
    `ollama serve` (launched the same way, via WMI Win32_Process Create's cmd.exe wrapper) was running at the
    time -- confirmed `where ollama` ALSO fails in a plain interactive SSH session on this host, while the exe
    is confirmed present at the standard install path. The most likely explanation: WMI-launched processes
    (and possibly OpenSSH's own non-interactive command execution) do not inherit the interactive user's
    PATH/profile the way a normal logon shell does, so neither PATH-based lookup is reliable in this call
    context regardless of what an interactive terminal reports. The LOCALAPPDATA fallback does not depend on
    PATH at all."""
    exe = os.environ.get("OLLAMA_BIN") or shutil.which("ollama")
    if exe:
        return exe
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidate = os.path.join(local_appdata, "Programs", "Ollama", "ollama.exe")
        if os.path.exists(candidate):
            return candidate
    return "ollama"


def create_model_from_gguf(name, gguf_path, run_fn=None):
    """"ollama create <name> -f Modelfile" with a Modelfile whose only line is "FROM <gguf_path>" -- the real CLI
    path (not a guessed HTTP API shape), matching exactly what a person would type. Ollama reads the model's native
    context straight from the GGUF's own metadata in this path; there is no registry tag lookup involved at all,
    which is the whole point of using this for a model (qwen3-4b-2507) whose registry tag turned out not to exist.
    run_fn is injectable for tests: (argv: list[str]) -> subprocess.CompletedProcess-shaped object with
    .returncode/.stdout/.stderr. Returns {"outcome": "ok"|"error", "returncode": int|None, "stdout": str,
    "stderr": str}; never raises."""
    import subprocess
    import tempfile
    # Found live 2026-10-01 (evo-t2s): subprocess.run(..., text=True) decodes stdout/stderr with the
    # console's default codepage (cp1252 on this machine) when no encoding is given. "ollama create"'s own
    # output contained a byte cp1252 cannot decode, which crashed the internal reader thread -- and that
    # left the whole call hanging indefinitely (observed live: the process sat idle, tiny steady memory, no
    # further log output, for minutes) rather than raising or returning. errors="replace" makes an
    # undecodable byte become a replacement character instead of killing the reader thread.
    run_fn = run_fn or (lambda argv: subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                                                     errors="replace", timeout=600))
    exe = _resolve_ollama_exe()
    with tempfile.TemporaryDirectory() as td:
        modelfile_path = os.path.join(td, "Modelfile")
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(f"FROM {gguf_path}\n")
        try:
            result = run_fn([exe, "create", name, "-f", modelfile_path])
        except Exception as e:
            return {"outcome": "error", "returncode": None, "stdout": "", "stderr": repr(e)[:500]}
    outcome = "ok" if getattr(result, "returncode", 1) == 0 else "error"
    return {"outcome": outcome, "returncode": getattr(result, "returncode", None),
            "stdout": (getattr(result, "stdout", "") or "")[:2000],
            "stderr": (getattr(result, "stderr", "") or "")[:2000]}


# ---------------------------------------------------------------------------------------------------- lab scaffold
class K1Lab:
    """Owns the output jsonl, provenance, host config, and resumption state. Not the Server/Telemetry lifecycle for
    the occupying process, which each phase manages itself in a finally block."""

    def __init__(self, args, host_cfg, prov):
        self.args, self.host_cfg, self.prov = args, host_cfg, prov
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # host_cfg has no "name" key (see HOSTS in host_config.py) -- the real per-host identifier is
        # "hw_id". Found live 2026-10-01: this fallback branch (args.resume not given) was never
        # exercised before run_positive_control() called K1Lab/R2SessionLab's __init__ with resume=None,
        # since every other caller always passes --resume.
        self.stem = args.resume or f"t2s_k1_ollama_{host_cfg['hw_id']}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
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
        row.setdefault("host", self.host_cfg["hw_id"])
        row.setdefault("gpu_vendor", self.host_cfg["gpu_vendor"])
        row.setdefault("ollama_model_loaded", _hc.get_ollama_loaded_model())
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

# ---------------------------------------------------------------------------------------------------- empirical context probe
# (c) A direct probe that needs no ollama-ps/log parsing at all: send prompts of known target length with no
# num_ctx, and use Ollama's own prompt_eval_count (returned for every /api/chat call, real ground truth for that
# specific request) to find where truncation starts. Added 2026-09-29 after phase_tier's ps/log signals both came
# back empty on evo-x2 (OLLAMA_KEEP_ALIVE=0 unloads the model right after each call, so `ollama ps` -- and the
# server.log runner line, if it is written to a location this file's candidates miss -- have nothing to show).
EMPIRICAL_CTX_PROBE_LENGTHS = (1000, 3000, 6000, 12000, 24000, 48000, 96000)
_EMPIRICAL_CTX_TASK_TYPE = "niah_multikey"  # has a marker key/value buried near the end plus a question last


def probe_effective_context_empirically(ollama, model, lengths=EMPIRICAL_CTX_PROBE_LENGTHS, seed=20260930):
    """For each target length, builds a niah_multikey task (quality_suite.build_task -- filler, a marker
    key/value pair, then a question asking for that exact value) and sends it with no num_ctx override. Records
    prompt_eval_count and whether the answer contains the correct marker value. The smallest length (1000 tokens)
    is assumed to fit under any real default context and calibrates the actual/target ratio for this specific
    build; each larger length's prompt_eval_count is compared against that calibrated expectation -- a shortfall
    means Ollama silently truncated the prompt before it reached the model. The largest length whose
    prompt_eval_count still matches the calibrated expectation (within 2%) is the effective_ctx_estimate; None if
    even the smallest probe does not return a usable prompt_eval_count."""
    results = []
    calibration_ratio = None
    for target in lengths:
        prompt, expected, _scorer = qs_build_task(_EMPIRICAL_CTX_TASK_TYPE, target, seed)
        res = ollama.chat(model, prompt, num_ctx=None, max_tokens=32)
        pec = res.get("prompt_eval_count")
        message = res.get("message") or ""
        marker_used = isinstance(message, str) and str(expected) in message
        if calibration_ratio is None and pec:
            calibration_ratio = pec / target
        expected_pec = target * calibration_ratio if calibration_ratio else None
        full_retention = (pec is not None and expected_pec is not None and pec >= expected_pec * 0.98)
        results.append({"target_tokens": target, "chat_outcome": res.get("outcome"), "prompt_eval_count": pec,
                        "expected_prompt_eval_count": round(expected_pec) if expected_pec else None,
                        "marker_used": marker_used, "full_retention": full_retention})
    retained = [r["target_tokens"] for r in results if r["full_retention"]]
    effective_ctx_estimate = max(retained) if retained else None
    return {"probes": results, "calibration_ratio": calibration_ratio, "effective_ctx_estimate": effective_ctx_estimate}


# ---------------------------------------------------------------------------------------------------- K1 v3: model-ceiling confound fix
# K1 v2 (qwen3:8b only) found evo-x2's tier-selection landed on 40960 and this was WRONGLY read as a memory-tier
# kill-criterion failure; it is qwen3:8b's own native max context (40960), so Ollama's choice was capped by the
# MODEL's own ceiling, not by available memory. v2 could never tell "runtime picked a small tier" apart from
# "runtime picked a tier bigger than what the model itself supports" whenever the test model's native ceiling sits
# below any tier a well-provisioned machine would plausibly pick. v3 fixes this by testing two models whose native
# ceiling exceeds any realistic memory tier, so the observed tier is unambiguously about memory, not the model.
QWEN3_4B_GGUF_PATH = r"C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"

K1_V3_MODELS = (
    # 2026-09-30: "qwen3:4b-instruct-2507" is not a real Ollama registry tag -- pulling it 404s on both machines
    # (found live, twice). Do not guess at a corrected tag name: create the model locally from the GGUF already on
    # disk (qwen3-4b-instruct-85e4a5b7.gguf, the same file t2s_overnight.MODEL_FILES uses for this model id), via
    # "ollama create <name> -f Modelfile" with a Modelfile whose only line is "FROM <gguf_path>". Ollama reads the
    # model's native context straight from the GGUF's own metadata in this path (there is no registry lookup
    # involved at all), so native_ctx=262144 below is still the right expectation, just sourced differently.
    {"tag": "qwen3-4b-2507", "native_ctx": 262144, "capped": False,
     "source": "create_from_gguf", "gguf_path": QWEN3_4B_GGUF_PATH},
    # native context 131072 -- also above any plausible tier. Already pulled on both machines.
    {"tag": "llama3.1:8b", "native_ctx": 131072, "capped": False, "source": "pull"},
    # Kept as a labeled control: native context 40960 is already known to be SMALLER than the tiers either machine
    # is expected to pick, so a run landing at/near 40960 for this model contrasts against, rather than confirms,
    # a genuine memory-availability effect. Every row this model produces below is stamped capped=True.
    {"tag": "qwen3:8b", "native_ctx": 40960, "capped": True, "source": "pull"},
)

K1_V3_PROBE_LENGTHS = (16000, 32000, 48000, 96000, 128000)


K1_V3_LONG_PROBE_THRESHOLD = 64_000  # at/above this target, use K1_V3_LONG_PROBE_TIMEOUT_S instead of the
                                      # client's own 1800s default
K1_V3_LONG_PROBE_TIMEOUT_S = 3600    # found live 2026-09-30/2026-10-01: three real K1 v3 probe calls on evo-x2
                                      # (qwen3-4b-2507 at 96K/128K, llama3.1:8b at 128K) errored out at gaps of
                                      # 1800.1s, 1802.0s and 1800.0s respectively -- within 2s of OllamaClient's
                                      # default 1800s timeout every single time. These were real prefill calls
                                      # still running past the client's own patience, not a runtime failure; a
                                      # client timeout must never be reported as a model/runtime error.


def probe_v3_effective_context(lab: K1Lab, ollama, model_tag: str, capped: bool, lengths=K1_V3_PROBE_LENGTHS,
                               seed: int = 20260930, native_ctx: int | None = None,
                               long_probe_threshold: int = K1_V3_LONG_PROBE_THRESHOLD,
                               long_probe_timeout_s: int = K1_V3_LONG_PROBE_TIMEOUT_S):
    """The 5-length empirical probe sweep required by K1 v3 (see K1_V3_PROBE_LENGTHS). For each target length:
    build a niah_multikey marker task sized to that many tokens (same construction as
    probe_effective_context_empirically, via qs_build_task), send it with no num_ctx override, then read BOTH
    signals at that same point in time: (1) GET /api/ps's context_length for this model (the runtime's own report),
    and (2) the empirical marker-probe result itself (did the buried marker value survive).

    sent_tokens_real / truncation criterion (fixed 2026-10-01, found live): `target` is the nominal prompt size
    the shared task-builder (qs_build_task) aimed for using ITS OWN tokenizer at build time, not necessarily what
    the model being probed actually counts the same text as -- comparing prompt_eval_count against `target`
    directly silently mislabeled every row from a model whose own tokenizer counts text differently (e.g.
    llama3.1:8b consistently at ~0.818x the nominal target across every length, confirmed not truncated once
    compared against its own real count instead). Fixed by a one-time calibration call per model: send the
    lengths[0] prompt (the shortest, least likely to risk truncation) with num_ctx forced to `native_ctx` (or
    left alone if not given) so it cannot be clipped, read that call's real prompt_eval_count, and scale it to
    every other length by simple proportion (lengths[0]_prompt and every other length's prompt are built by the
    same generator at a different scale, so the per-token expansion ratio between the build-time tokenizer and
    this model's own tokenizer is expected to hold approximately constant across lengths). A row counts as
    silently truncated only if processed tokens are more than 1% below this real sent-token estimate, OR the
    marker was lost -- not an OR with raw token-ratio alone, since a model can lose the marker for a reason
    unrelated to context truncation (see the module's own K1 v3 write-up: qwen3:8b lost the marker at every
    length including 16K/32K, where its own token math shows no truncation at all -- that is reported here as a
    separate marker_lost_unexplained flag, not folded into silently_truncated, so a real memory-wall truncation
    is never confused with an unrelated marker-matching miss).

    Calls at or above long_probe_threshold tokens use long_probe_timeout_s (not the client's own default) -- see
    K1_V3_LONG_PROBE_TIMEOUT_S's own comment for why this exists. Emits and returns exactly one row per length
    via lab.emit, plus one calibration row."""
    # Calibration row is emitted (lands in the jsonl) but intentionally NOT included in this function's returned
    # list -- callers/tests treat the return value as exactly one row per real probe length.
    cal_prompt, cal_expected, _ = qs_build_task(_EMPIRICAL_CTX_TASK_TYPE, lengths[0], seed + lengths[0])
    cal_res = ollama.chat(model_tag, cal_prompt, num_ctx=native_ctx, max_tokens=1, think=False)
    cal_pec = cal_res.get("prompt_eval_count")
    tokenizer_ratio = (cal_pec / lengths[0]) if (cal_pec and cal_res.get("outcome") == "ok") else None
    lab.emit({
        "record": "tier_v3_probe_calibration", "phase": "tier_v3", "model_tag": model_tag, "capped": capped,
        "calibration_target_tokens": lengths[0], "calibration_num_ctx": native_ctx,
        "calibration_prompt_eval_count": cal_pec, "tokenizer_ratio": tokenizer_ratio,
        "calibration_outcome": cal_res.get("outcome"), "calibration_error": cal_res.get("error"),
    })
    rows = []
    for target in lengths:
        prompt, expected, _scorer = qs_build_task(_EMPIRICAL_CTX_TASK_TYPE, target, seed + target)
        call_timeout = long_probe_timeout_s if target >= long_probe_threshold else None
        res = ollama.chat(model_tag, prompt, num_ctx=None, max_tokens=32, timeout=call_timeout, think=False)
        ps = ollama.get_ps()
        ps_match = next((m for m in ps.get("models", [])
                         if m.get("name") == model_tag or (m.get("name") or "").startswith(model_tag.split(":")[0])),
                        None)
        ps_context_length = ps_match.get("context_length") if ps_match else None
        pec = res.get("prompt_eval_count")
        message = res.get("message") or ""
        marker_used = isinstance(message, str) and str(expected) in message
        sent_tokens_real = round(target * tokenizer_ratio) if tokenizer_ratio else None
        token_truncated = (res.get("status") == 200 and pec is not None and sent_tokens_real is not None
                          and pec < 0.99 * sent_tokens_real)
        silently_truncated = token_truncated or not marker_used
        marker_lost_unexplained = (not marker_used) and not token_truncated
        rows.append(lab.emit({
            "record": "tier_v3_probe", "phase": "tier_v3", "model_tag": model_tag, "capped": capped,
            "probe_target_tokens": target, "sent_tokens_target": target, "sent_tokens_real": sent_tokens_real,
            "chat_outcome": res.get("outcome"), "chat_error": res.get("error"), "http_status": res.get("status"),
            "chat_timeout_s": call_timeout,
            "prompt_eval_count": pec, "marker_used": marker_used,
            "ps_context_length": ps_context_length, "silently_truncated": silently_truncated,
            "token_truncated": token_truncated, "marker_lost_unexplained": marker_lost_unexplained,
        }))
    return rows


def phase_tier_v3(lab: K1Lab, models=K1_V3_MODELS, ollama=None, log_finder=find_ollama_log, pull_fn=pull_model,
                  create_fn=create_model_from_gguf, probe_lengths=K1_V3_PROBE_LENGTHS):
    """K1 v3 orchestrator. Literal first action (before anything else, including phase_tier or any probe call):
    make every model in `models` available, one at a time, with nothing else measuring concurrently -- a registry
    pull for models with "source": "pull", or "ollama create ... -f Modelfile" from the local GGUF for models with
    "source": "create_from_gguf" (see create_model_from_gguf; this is how qwen3-4b-2507 is made available, since
    its registry tag does not exist -- found live, twice, 2026-09-30). Then, per model, one phase_tier() call
    (reused as-is -- the log/api-ps/empirical-probe signal logic is NOT reimplemented here) to capture the tier
    Ollama actually chose, a small model-meta row stamping capped/native_ctx for that model (phase_tier's own row
    shape is left untouched, per the task's backward-compatibility requirement), and the 5-length empirical probe
    sweep (probe_v3_effective_context).

    2026-09-30 fix: each model is now fully isolated -- wrapped in its own try/except so one model's pull/create/
    tier/probe failure is recorded and skipped, never aborting the models after it. This replaced the original
    all-or-nothing loop after a single real pull failure (qwen3-4b-2507's nonexistent tag) silently prevented
    llama3.1:8b and qwen3:8b from ever being measured in the same run, on both machines, twice.

    Hypothesis under test (this function does not force the result either way): if the memory-tier-vs-availability
    link holds, evo-x2 (larger unified memory) should land close to each model's own native ceiling (262144 for
    qwen3-4b-2507, 131072 for llama3.1:8b) while evo-t2s (smaller/different memory budget) should land well below
    either ceiling (something near 32768) for the SAME two models with the SAME native ceilings -- a comparison
    K1 v2 could never make, since qwen3:8b's 40960 ceiling was already below any plausible tier on either machine.
    """
    ollama = ollama or lab.ollama
    tier_rows, meta_rows, probe_rows, failures = [], [], [], []

    # A2 (2026-10-01): record Ollama's own GPU/compute-device detection once per run, before any model-specific
    # measurement, so a small default context tier on a given host can be correctly attributed to "no GPU
    # detected" (a backend-support gap) rather than silently assumed to be a memory-tier decision -- those are
    # different mechanisms with the same symptom.
    device_log_path = log_finder()
    device_info = parse_ollama_log_device(device_log_path)
    lab.emit({"record": "tier_v3_device_detect", "phase": "tier_v3", "ollama_log_path": device_log_path,
             **device_info})

    # Pass 1: make every model available (pull or create), one at a time, nothing else measuring concurrently --
    # this ordering (every model's availability step, in full, before any model's measurement starts) is the
    # literal requirement from the original K1 v3 spec ("pull them as the first step of the job itself, with
    # nothing else measuring"); each model is still isolated in its own try/except so one failing does not stop
    # the next model's availability step from being attempted.
    available = []
    for m in models:
        try:
            source = m.get("source", "pull")
            if source == "create_from_gguf":
                create_res = create_fn(m["tag"], m["gguf_path"])
                lab.emit({"record": "tier_v3_create", "phase": "tier_v3", "model_tag": m["tag"],
                          "capped": m["capped"], "native_ctx": m["native_ctx"], "gguf_path": m["gguf_path"],
                          "create_outcome": create_res.get("outcome"), "create_stderr": create_res.get("stderr")})
                if create_res.get("outcome") != "ok":
                    raise RuntimeError(f"ollama create failed for {m['tag']!r}: {create_res.get('stderr')!r}")
            else:
                pull_res = pull_fn(ollama, m["tag"])
                pull_outcome = pull_res.get("outcome") if isinstance(pull_res, dict) else None
                # Found live 2026-10-01 (evo-t2s): a pull that hits a real exception (OllamaClient.pull's
                # except Exception branch, e.g. a connection error) sets "error", not "final_status" --
                # final_status is only ever set on a real pull response (success or a mid-stream "error"
                # object). Reading only final_status silently discarded the real cause every time, logging
                # "pull failed for 'x': None" with no way to diagnose it after the fact.
                pull_error = (pull_res.get("error") or pull_res.get("final_status")) if isinstance(pull_res, dict) else None
                lab.emit({"record": "tier_v3_pull", "phase": "tier_v3", "model_tag": m["tag"],
                          "capped": m["capped"], "native_ctx": m["native_ctx"],
                          "pull_outcome": pull_outcome, "pull_error": pull_error})
                if pull_outcome != "ok":
                    raise RuntimeError(f"pull failed for {m['tag']!r}: {pull_error!r}")
            available.append(m)
        except Exception as e:
            L.log(f"K1 v3: {m['tag']!r} unavailable ({e!r}), skipping this model, continuing with the rest")
            lab.emit({"record": "tier_v3_model_failed", "phase": "tier_v3", "model_tag": m["tag"],
                      "stage": "make_available", "error": repr(e)[:500]})
            failures.append({"model_tag": m["tag"], "stage": "make_available", "error": repr(e)[:500]})

    # Pass 2: tier + probe measurement, only for models that actually became available above -- isolated the
    # same way, one model's measurement failure does not stop the next model's.
    for m in available:
        try:
            tier_rows.append(phase_tier(lab, m["tag"], rep=0, ollama=ollama, log_finder=log_finder))
            meta_rows.append(lab.emit({
                "record": "tier_v3_model_meta", "phase": "tier_v3", "model_tag": m["tag"],
                "capped": m["capped"], "native_ctx": m["native_ctx"],
            }))
            probe_rows.extend(probe_v3_effective_context(lab, ollama, m["tag"], m["capped"], lengths=probe_lengths,
                                                         native_ctx=m["native_ctx"]))
        except Exception as e:
            L.log(f"K1 v3: {m['tag']!r} tier/probe measurement failed ({e!r}), skipping, continuing with the rest")
            lab.emit({"record": "tier_v3_model_failed", "phase": "tier_v3", "model_tag": m["tag"],
                      "stage": "tier_or_probe", "error": repr(e)[:500]})
            failures.append({"model_tag": m["tag"], "stage": "tier_or_probe", "error": repr(e)[:500]})

    return {"tier_rows": tier_rows, "meta_rows": meta_rows, "probe_rows": probe_rows, "failures": failures}


# ---------------------------------------------------------------------------------------------------- phase: tier
def phase_tier(lab: K1Lab, ollama_model: str, rep: int = 0, ollama=None, log_finder=find_ollama_log,
               run_empirical=True):
    """Captures Ollama's default context choice for ollama_model by three independent signals, in order of cost:

    (a) /api/ps while the model is deliberately kept alive: chat with keep_alive="10m" (so OLLAMA_KEEP_ALIVE=0's
        immediate unload does not race the read), GET /api/ps recording every field the response has (name, size,
        size_vram, context_length when present, etc.), then explicitly unload (keep_alive=0) and confirm /api/ps
        comes back empty. Fixes the 2026-09-29 finding on evo-x2: with keep_alive left at the server's own
        OLLAMA_KEEP_ALIVE=0 default, the model was already gone by the time this phase queried ps.
    (b) the Ollama server log's runner start line (--ctx-size), parsed from %LOCALAPPDATA%\\Ollama\\server.log.
    (c) only if (a) and (b) both come back empty: an empirical probe (see probe_effective_context_empirically)
        that finds the effective default context directly from Ollama's own prompt_eval_count, no ps/log parsing
        needed at all.

    Raises RuntimeError if none of the three signals capture a context value -- this phase must never emit a row a
    kill-criteria check could read as a verdict when the underlying data is simply missing."""
    ollama = ollama or lab.ollama
    log_path = log_finder()
    log_pos = os.path.getsize(log_path) if log_path and os.path.exists(log_path) else 0

    res = ollama.chat(ollama_model, TIER_PROBE_PROMPT, num_ctx=None, max_tokens=32, keep_alive="10m")
    ps_after_chat = ollama.get_ps()
    ps_match = next((m for m in ps_after_chat.get("models", [])
                     if m.get("name") == ollama_model or (m.get("name") or "").startswith(ollama_model.split(":")[0])), None)
    unload_res = ollama.unload(ollama_model)
    ps_after_unload = ollama.get_ps()
    unload_confirmed_empty = not ps_after_unload.get("models")

    log_info = parse_ollama_log_context(log_path, log_pos) if log_path else {"available": False, "why": "no log path found"}
    runner_ctx = None
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, "rb") as f:
                f.seek(max(log_pos, 0))
                text = f.read().decode("utf-8", errors="replace")
            m = _LOG_RUNNER_CTX_RE.search(text)
            runner_ctx = int(m.group(1)) if m else None
        except Exception:
            runner_ctx = None
    log_ctx_candidates = log_info.get("num_ctx_seen") or []

    default_ctx_from_ps = ps_match.get("context_length") if ps_match else None
    default_ctx_from_log = runner_ctx or (max(log_ctx_candidates) if log_ctx_candidates else None)

    empirical = None
    default_ctx_from_empirical = None
    if run_empirical and default_ctx_from_ps is None and default_ctx_from_log is None:
        empirical = probe_effective_context_empirically(ollama, ollama_model)
        default_ctx_from_empirical = empirical.get("effective_ctx_estimate")

    default_ctx = default_ctx_from_ps if default_ctx_from_ps is not None else \
        (default_ctx_from_log if default_ctx_from_log is not None else default_ctx_from_empirical)
    ctx_source = ("api_ps" if default_ctx_from_ps is not None else
                  "server_log" if default_ctx_from_log is not None else
                  "empirical_probe" if default_ctx_from_empirical is not None else None)

    row = lab.emit({
        "record": "tier", "phase": "tier", "model_tag": ollama_model, "rep": rep,
        "chat_outcome": res.get("outcome"), "chat_status": res.get("status"), "chat_error": res.get("error"),
        "chat_duration_s": res.get("duration_s"), "chat_prompt_eval_count": res.get("prompt_eval_count"),
        "ollama_ps_raw": ps_match, "unload_outcome": unload_res.get("outcome"),
        "unload_confirmed_empty": unload_confirmed_empty,
        "ollama_default_ctx": default_ctx, "ollama_default_ctx_source": ctx_source,
        "ollama_default_ctx_from_ps": default_ctx_from_ps, "ollama_default_ctx_from_log": default_ctx_from_log,
        "ollama_default_ctx_from_empirical": default_ctx_from_empirical,
        "ollama_log_path": log_path, "ollama_log_info": log_info, "ollama_empirical_probe": empirical,
        "mem_headroom_gb": None,
    })
    if default_ctx is None:
        raise RuntimeError(f"K1 tier: could not capture ollama_default_ctx for {ollama_model} by any signal "
                            f"(api/ps, server.log, empirical probe); this run is INVALID, not a kill-criteria "
                            f"FAIL -- refusing to write a result a kill-criteria check could read as a verdict "
                            f"on missing data")
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
                         server_factory=None, max_tokens: int = 64, calibration_pass_set=None):
    """Arm (a) Ollama default (no num_ctx), arm (b) Ollama with num_ctx fit to the prompt, arm (c) llama-server
    directly with -c == default_ctx (Ollama's own default, established once for the whole phase by the caller from
    an earlier phase_tier row: see build_default_ctx_from_rows). Three reps per condition. default_ctx is a
    parameter rather than re-derived here so the fixed-context arm (c) baseline is the SAME number across every
    prompt length in this phase, matching the task spec ("-c equal to whatever Ollama's default context was").

    calibration_pass_set: an optional set of task_types that q0_token_calibration.py's own per-task check passed
    (see quality_suite.load_calibration_pass_set). None means "unknown, run every requested task_type unfiltered"
    (the old behavior). When given, any requested task_type not in the set is skipped and logged with the reason,
    rather than either blocking this whole phase on one failing task or silently running a known-miscalibrated one."""
    ollama = ollama or lab.ollama
    server_factory = server_factory or (lambda mi, n_ctx, tag: L.Server(lab, mi, n_ctx, backend=lab.host_cfg.get("backend", "vulkan"), tag=tag))
    rows = []
    if calibration_pass_set is not None:
        excluded = [t for t in task_types if t not in calibration_pass_set]
        for t in excluded:
            L.log(f"K1 curves: excluding task_type {t!r}, did not pass q0_token_calibration for this model")
        task_types = [t for t in task_types if t in calibration_pass_set]
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

    # A tier/memory_pressure row with ollama_default_ctx=None is INVALID (the context could not be captured by any
    # signal), not a fail -- phase_tier itself now raises rather than emit one (see phase_tier's (d)), so this only
    # matters when reading an older/historical run's rows. A fail requires a real captured-but-too-small context.
    tier_rows_any = [r for r in rows if r.get("record") in ("tier", "memory_pressure")]
    tier_rows = [r for r in tier_rows_any if r.get("ollama_default_ctx") is not None]
    if not tier_rows_any:
        out["defaults_cover_agent_prompts"] = {"ok": False, "reason": "no tier/memory_pressure rows found at all"}
    elif not tier_rows:
        out["defaults_cover_agent_prompts"] = {"ok": None, "status": "invalid_no_ctx_captured", "reason": (
            f"{len(tier_rows_any)} tier/memory_pressure row(s) found, but none captured ollama_default_ctx by any "
            f"signal (api/ps, server.log, empirical probe) -- this is INVALID, not a kill-criterion FAIL; the run "
            f"must be re-run with context capture working before this check means anything")}
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
        status = "OK" if r["ok"] is True else "INVALID" if r.get("status") == "invalid_no_ctx_captured" else "FAIL"
        L.log(f"[{status}] {name}: {r['reason']}")
    L.log("=== end kill criteria ===")


# ---------------------------------------------------------------------------------------------------- CLI
def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="tier,memory,curves",
                    help="comma-separated subset of tier,memory,curves,tier_v3 (tier_v3 is K1 v3: pulls "
                         "K1_V3_MODELS then runs phase_tier_v3 -- see module docstring)")
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
    ap.add_argument("--calibration-file", default=None, help="path to a q0_token_calibration.py run's own jsonl; "
                    "the curves phase runs only the task_types that passed calibration in it, per-task (not "
                    "whole-run) -- see quality_suite.load_calibration_pass_set. Omit to run every task unfiltered.")
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

    # This must always run, on every exit path, or the queue is left stuck on "running" until the watchdog's
    # heartbeat-staleness threshold expires (30-60 min wasted) -- found 2026-09-29 when this file turned out to
    # never call tq.advance() at all.
    note = "completed"
    kc = None
    try:
        # K1 is the only script (with K2) allowed to run Ollama at all (2026-09-29 contamination check, see
        # docs/RESULT_PROVENANCE.md); it starts its own server here and always stops it in the finally block below,
        # so Ollama never idles in the background once this job ends.
        started_pid = _hc.start_ollama_server()
        L.log(f"ollama server {'already running' if started_pid is None else f'started (pid={started_pid})'}")
        if started_pid is not None:
            # Found live 2026-10-01: the WMI-reported pid does not mean the HTTP server is listening yet --
            # phase_tier_v3's first pull on evo-t2s failed on a connection error 9s after this log line,
            # before Ollama had finished starting. See wait_for_ollama_ready's own docstring.
            ready = _hc.wait_for_ollama_ready()
            L.log(f"ollama server ready: {ready}" if ready else
                 "ollama server did NOT become ready within the wait timeout; proceeding anyway, expect pull/chat "
                 "failures")
        try:
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

            if "tier_v3" in phases and "phase_tier_v3" not in lab.done_phases:
                phase_tier_v3(lab)
                lab.phase_done("phase_tier_v3")

            if "curves" in phases and "phase_curves" not in lab.done_phases:
                if gguf_mi is None:
                    L.log("curves phase requested but no --gguf-model/--gguf-path given; skipping")
                else:
                    default_ctx = args.default_ctx or build_default_ctx_from_rows(lab.all_rows(), host=host_cfg["hw_id"])
                    if default_ctx is None:
                        L.log("curves phase requested but no default_ctx known (run --phase tier first, or pass --default-ctx); skipping")
                    else:
                        calibration_pass_set = None
                        if args.calibration_file:
                            if QUALITY_SUITE_SOURCE != "harness.quality_suite" or not hasattr(_qs, "load_calibration_pass_set"):
                                L.log(f"--calibration-file {args.calibration_file!r} given but {QUALITY_SUITE_SOURCE} has no load_calibration_pass_set; running every task_type unfiltered")
                            else:
                                calibration_pass_set = _qs.load_calibration_pass_set(args.calibration_file)
                            if calibration_pass_set is None:
                                L.log(f"--calibration-file {args.calibration_file!r} had no usable q0_calibration_summary; running every task_type unfiltered")
                            else:
                                L.log(f"K1 curves: calibration pass set from {args.calibration_file!r}: {sorted(calibration_pass_set)}")
                        phase_quality_curves(lab, args.ollama_model, gguf_mi, default_ctx, rep_count=args.reps,
                                             calibration_pass_set=calibration_pass_set)
                        lab.phase_done("phase_curves")
        finally:
            stop_result = _hc.stop_ollama_server()
            L.log(f"ollama server stopped: {stop_result}")

        all_rows = lab.all_rows()
        kc = kill_criteria(all_rows)
        report_kill_criteria(kc)
        L.log(f"run complete: stem={lab.stem} rows={len(all_rows)} phases_run={phases}")
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
        L.log(note)
        raise
    finally:
        try:
            tq.advance(note)
        except Exception as e:
            L.log(f"queue advance failed: {e!r}")
    return kc


if __name__ == "__main__":
    main()
