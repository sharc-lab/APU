"""Cross-runtime replication of the B3 v2 core result on llama.cpp server (docs/EXPERIMENTS.md, "PRE-REGISTRATION
(2026-10-26): cross-runtime replication on llama.cpp server"; queue id x2_xruntime_llamacpp). Register rows
xruntime-llamacpp-wrong-rate and xruntime-default-fixed read its file; row xruntime-llamacpp-hours is the ESTIMATE.

The B3 v2 task session of harness/x2_r2_evalset.py (design v2, tool loop, 30 task turns, guardrails off, the same
evalset_env tasks, checkers and end-state scoring, seeds 20260901-05) driven through x2_r2_evalset.run_session with a
thin runtime adapter (LlamaCppRuntime) that talks to llama-server's OpenAI-compatible /v1/chat/completions instead of
Ollama's /api/chat. Nothing in x2_r2_evalset or x2_r2_agent is changed; the arms are registered in x2_r2_agent.ARMS at
import, in the manner of x2_r2_evalset's own num_keep and own-compaction arms.

Arms (qwen3:14b weights: the official Qwen3-14B-Q4_K_M GGUF at GGUF, the file the llama_server leg of
harness/x2_outcome_table.py uses on evo-x2; llama-server b10970 Vulkan at LLAMA_SERVER_EXE, -ngl 99, -np 1, --jinja,
reasoning off), one fresh server process per session (the counterpart of the Ollama run's unload before each session):
  (a) llamacpp_default      : no -c; the window is whatever the server loads, read from /props and recorded
  (b) llamacpp_ctx_4096     : -c 4096
  (c) llamacpp_ctx_40960    : -c 40960, "default fixed": the GGUF's trained context (n_ctx_train 40960), which is also
                              the window Ollama's default loaded for qwen3:14b in x2_evalset_tasks_real_v2
Context shift is left at the server default (neither --context-shift nor --no-context-shift is passed); the server's
--help text for --ctx-size, --context-shift and --fit is printed by --check-plan and recorded in run_start.

Detector (ground truth for silent vs flagged): the server's own log. Each call's slice of the server log (byte offset
before the call to after it) is matched against REJECT_RE, SHIFT_RE, TRUNC_RE and CTX_STOP_RE (classify_log), and each
call is classed by call_class: flagged_reject (HTTP 4xx context-size error: the client is told), silent_cut (HTTP 200
and a context-shift or truncation line: tokens dropped, the client is not told), ctx_stop (HTTP 200, generation stopped
at the window: a stop line, or finish_reason "length" with prompt + completion at the window), none. A turn takes the
first of those classes any of its calls has (turn_class). A detector probe on its own server start (-c PROBE_CTX)
sends one prompt over the window and one generation over the window before any session; the detector is validated iff
both probes produce a detected event (detector_validated). Without that, the silent/flagged split is not read.

Stale-server guard (harness/server_guard.py, every llama-server harness): port free before each start; after each start
/props model path, slot count and (arms b, c) n_ctx plus the command line asserted; before every request the listening
PID must be the started PID, else the condition (the arm) stops and its partial session is not counted.

    python xrt\\xruntime_llamacpp.py --check-plan
        plan, sessions, server command lines, exe and GGUF presence and size, port listeners, the server's --help lines;
        no server start, no model call, no queue access
    python xrt\\xruntime_llamacpp.py --out C:\\apu\\ovn\\results\\xruntime_llamacpp_qwen3_14b.jsonl

Queue job: calls t2s_queue.advance() exactly once on exit (finally); sets no pause. Own port PORT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if not (_here / "t2s_queue.py").exists():
    sys.path.insert(1, str(_here.parent))

# imported before x2_r2_evalset (whose x2_r2_agent import puts C:\apu\ovn first on sys.path): this directory's copies
import server_guard as sg  # noqa: E402

try:
    from proc_util import popen_hidden, run_hidden  # noqa: E402
except ImportError:   # pragma: no cover - proc_util is deployed next to this file
    popen_hidden, run_hidden = subprocess.Popen, subprocess.run

import x2_r2_evalset as EV  # noqa: E402

ag = EV.ag

HARNESS = "xruntime_llamacpp"
LLAMA_SERVER_EXE = r"C:\apu\bin\llama-b10970\llama-server.exe"   # mx2_validation / x2_outcome_table on evo-x2
GGUF = r"C:\apu\models\Qwen3-14B-Q4_K_M.gguf"                      # x2_outcome_table.MODEL_MAP["qwen3-14b"]
GGUF_BYTES = 9_001_752_960                                         # docs/X2_CHANGELOG.md, hash table of 2026-09-28
GGUF_SHA256_HEAD, GGUF_SHA256_TAIL = "500a8806e", "ffeb6b81f0"    # the same table (published hash, abbreviated)
MODEL_ID = "qwen3:14b"     # row label, as in x2_evalset_tasks_real_v2 (same session builder, think False)
PORT = 58421
N_CTX_TRAIN = 40960
PROBE_CTX = 1024
SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
TURNS = 30
DESIGN = "v2"
CALL2 = "off"
SFX = ag.MODE_SUFFIX[CALL2]
BASES = {"default": ("llamacpp_default", None), "4096": ("llamacpp_ctx_4096", 4096),
         "fixed": ("llamacpp_ctx_40960", N_CTX_TRAIN)}
ARM = {k: b + SFX for k, (b, _n) in BASES.items()}
ARM_ORDER = ("default", "4096", "fixed")
for _k, (_b, _n) in BASES.items():
    ag.ARMS.setdefault(_b + SFX, {"num_ctx": _n, "call2_tools": False, "call2_mode": CALL2})
TOOL_PATH_MIN = 0.90       # gate: share of task turns at arm (c) with at least one executed tool round
STARTUP_TIMEOUT_S = 300.0
LOG_SETTLE_S = 0.3
OLLAMA_PS = "http://127.0.0.1:11434/api/ps"
OLLAMA_REF_FILE = "x2_evalset_tasks_real_v2.jsonl"
OLLAMA_DEFAULT_ARM = "ollama_default" + SFX

REJECT_RE = re.compile(r"exceed_context_size_error|exceeds the available context size|exceeds the context size",
                       re.IGNORECASE)
SHIFT_RE = re.compile(r"context shift(?! is disabled)|context_shift: ", re.IGNORECASE)
TRUNC_RE = re.compile(r"input truncated|prompt truncated|truncated prompt|truncating (the )?(input|prompt)",
                      re.IGNORECASE)
CTX_STOP_RE = re.compile(r"context shift is disabled|out of context|context capacity|n_past >= n_ctx",
                         re.IGNORECASE)
CLASS_ORDER = ("flagged_reject", "silent_cut", "ctx_stop", "error_other", "none")
HELP_KEYS = ("--ctx-size", "--context-shift", "--no-context-shift", "--fit", "--fit-ctx", "--parallel", "--jinja",
             "--reasoning-budget", "--cache-prompt")


def git_blob_sha1(path) -> str | None:
    try:
        b = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


# ── pure parts ────────────────────────────────────────────────────────────────────────────────────

def server_cmd(arm_key: str | None, log_path: str, port: int = PORT, exe: str = LLAMA_SERVER_EXE, gguf: str = GGUF,
               n_ctx: int | None = None) -> list[str]:
    """The launch command. arm_key None with n_ctx: the detector probe server. No context-shift flag either way."""
    if arm_key is not None:
        n_ctx = BASES[arm_key][1]
    cmd = [exe, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "--no-webui", "-ngl", "99", "-np", "1",
           "--jinja", "--reasoning-format", "deepseek", "--reasoning-budget", "0", "--log-file", log_path,
           "--log-verbosity", "3"]
    if n_ctx is not None:
        cmd += ["-c", str(n_ctx)]
    return cmd


def guard_expected(arm_key: str | None, gguf: str = GGUF, n_ctx: int | None = None) -> dict:
    exp = {"model_path": gguf, "np": 1, "ngl": 99}
    n = BASES[arm_key][1] if arm_key is not None else n_ctx
    if n is not None:
        exp["n_ctx"] = n
    return exp


def classify_log(text: str) -> dict:
    """Matched server log lines by kind (at most 3 kept per kind)."""
    out = {"reject": [], "shift": [], "trunc": [], "ctx_stop": []}
    for line in (text or "").splitlines():
        for k, rx in (("reject", REJECT_RE), ("shift", SHIFT_RE), ("trunc", TRUNC_RE), ("ctx_stop", CTX_STOP_RE)):
            if rx.search(line) and len(out[k]) < 3:
                out[k].append(line.strip()[:300])
    return out


def call_class(status, finish_reason, prompt_tokens, completion_tokens, n_ctx, log: dict, error_text=None) -> str:
    """flagged_reject | silent_cut | ctx_stop | error_other | none for one call (see the module docstring)."""
    if status is not None and 400 <= int(status) < 500 and (REJECT_RE.search(error_text or "") or log.get("reject")):
        return "flagged_reject"
    if status == 200 and (log.get("shift") or log.get("trunc")):
        return "silent_cut"
    if status == 200:
        at_window = (n_ctx is not None and prompt_tokens is not None and completion_tokens is not None
                     and prompt_tokens + completion_tokens >= n_ctx - 1)
        if log.get("ctx_stop") or (finish_reason == "length" and at_window):
            return "ctx_stop"
    if status != 200:
        return "error_other"   # an HTTP or connection error that is not a context-size rejection
    return "none"


def turn_class(call_events: list[dict]) -> str:
    classes = {e.get("class") for e in call_events or []}
    return next((c for c in CLASS_ORDER if c in classes), "none")


def detector_validated(probes: list[dict]) -> bool:
    """Both probes produced a detected event: the prompt over the window a reject or a silent cut, the generation over
    the window a silent cut (context shift) or a context stop."""
    by = {p.get("probe"): p.get("class") for p in probes}
    return (by.get("prompt_over_window") in ("flagged_reject", "silent_cut")
            and by.get("generation_over_window") in ("silent_cut", "ctx_stop"))


def native_tool_calls(tcs) -> list[dict] | None:
    """OpenAI tool_calls -> the Ollama-native shape the evalset turn reads ({"function": {"name", "arguments"}},
    arguments parsed to a dict when they are a JSON object)."""
    if not tcs:
        return None
    out = []
    for tc in tcs:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                args = parsed if isinstance(parsed, dict) else args
            except ValueError:
                pass
        out.append({"id": tc.get("id"), "function": {"name": fn.get("name"), "arguments": args}})
    return out


def chat_body(messages, tools, think, alias="local", max_tokens=None) -> dict:
    body = {"model": alias, "messages": ag.to_openai_messages(messages), "stream": False,
            "max_tokens": max_tokens or ag.MAX_TOKENS_PER_CALL, "temperature": 0, "seed": 42}
    if tools is not None:
        body["tools"] = tools
    if think is False:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def map_response(status, data, err, dt) -> dict:
    """llama-server response -> the OllamaClient.chat return shape x2_r2_agent._call_record reads."""
    if data is None:
        return {"outcome": "http_error" if status else "error", "status": status, "error": err, "duration_s": dt}
    ch = (data.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = data.get("usage") or {}
    raw = {"message": {"content": msg.get("content") or "", "thinking": msg.get("reasoning_content") or ""},
           "done_reason": ch.get("finish_reason"), "prompt_eval_count": usage.get("prompt_tokens"),
           "eval_count": usage.get("completion_tokens"), "timings": data.get("timings")}
    return {"outcome": "ok", "status": status, "message": msg.get("content") or "",
            "tool_calls": native_tool_calls(msg.get("tool_calls")), "prompt_eval_count": usage.get("prompt_tokens"),
            "eval_count": usage.get("completion_tokens"), "duration_s": dt, "raw": raw}


def help_lines(text: str) -> dict:
    """The --help lines (with their continuation lines) for HELP_KEYS."""
    out, cur = {}, None
    for line in (text or "").splitlines():
        s = line.strip()
        opt = line[:40] if s.startswith("-") else ""   # the option names (llama.cpp help: description at column 40)
        key = next((k for k in HELP_KEYS if re.search(rf"(^|[ ,]){re.escape(k)}([ ,=]|$)", opt)), None)
        if key:
            cur = key
            out[cur] = s
        elif cur and s and not s.startswith("-"):
            out[cur] += " " + s
        else:
            cur = None
    return out


def plan() -> list[tuple]:
    return [(ARM[k], SEEDS, TURNS) for k in ARM_ORDER]


class GuardMismatch(RuntimeError):
    pass


# ── runtime adapter ───────────────────────────────────────────────────────────────────────────────

def _post(url, body, timeout=ag.CALL_TIMEOUT_S):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), None, time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return e.code, None, e.read().decode(errors="replace")[:600], time.monotonic() - t0
    except Exception as e:
        return None, None, str(e)[:400], time.monotonic() - t0


def _read_from(path, offset) -> tuple[str, int]:
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            b = f.read()
        return b.decode("utf-8", errors="replace"), offset + len(b)
    except OSError:
        return "", offset


class LlamaCppRuntime:
    """The runtime interface x2_r2_evalset.run_session uses (chat, recover, loaded_context, unload, version), on one
    started server (`server`: url, n_ctx, log_path, guard, alive(), restart()). Per call: the request guard, the call,
    the call's server log slice and its class, kept until the turn row is emitted (pop_turn_events)."""

    def __init__(self, server, emit=None, post=None, settle_s=None):
        self.server, self.emit, self.post = server, emit or (lambda r: None), post or _post
        self.settle_s = LOG_SETTLE_S if settle_s is None else settle_s
        self._events = []
        self.max_tokens = None   # None: x2_r2_agent.MAX_TOKENS_PER_CALL (the Ollama run's num_predict); probes only

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        if extra:
            raise ValueError("call-2 forced modes are not part of this plan")
        ok, listeners = self.server.guard.check()
        if not ok:
            raise GuardMismatch(f"listener pids {listeners} != started pid {self.server.pid}; condition stopped")
        off = os.path.getsize(self.server.log_path) if os.path.exists(self.server.log_path) else 0
        status, data, err, dt = self.post(f"{self.server.url}/v1/chat/completions", chat_body(messages, tools, think,
                                                                                    max_tokens=self.max_tokens))
        time.sleep(self.settle_s)
        text, _ = _read_from(self.server.log_path, off)
        resp = map_response(status, data, err, dt)
        log = classify_log(text)
        raw = resp.get("raw") or {}
        ev = {"http_status": status, "finish_reason": raw.get("done_reason"),
              "prompt_tokens": raw.get("prompt_eval_count"), "completion_tokens": raw.get("eval_count"),
              "log_matches": {k: v for k, v in log.items() if v}, "listener_pids": listeners,
              "error": (err or "")[:300] or None}
        ev["class"] = call_class(status, ev["finish_reason"], ev["prompt_tokens"], ev["completion_tokens"],
                                 self.server.n_ctx, log, err)
        self._events.append(ev)
        return resp

    def pop_turn_events(self) -> list[dict]:
        ev, self._events = self._events, []
        return ev

    def recover(self):
        if self.server.alive():
            return True
        rec = self.server.restart()
        self.emit({"record": "server_restart", **rec, "ts_utc": ag.utc_iso()})
        return bool(rec.get("started"))

    def loaded_context(self, model):
        return self.server.n_ctx

    def unload(self, model, wait_s=30):
        return True   # a fresh server process per session: no KV state carried over (recorded in server_start)

    def version(self):
        return getattr(self.server, "build_info", None)


class LlamaServer:
    """One llama-server process for one arm (or the probe), guarded per harness/server_guard.py."""

    def __init__(self, arm_key, log_path, port=PORT, n_ctx=None, exe=LLAMA_SERVER_EXE, gguf=GGUF):
        self.arm_key, self.log_path, self.port, self.req_ctx = arm_key, str(log_path), port, n_ctx
        self.exe, self.gguf = exe, gguf
        self.url = f"http://127.0.0.1:{port}"
        self.proc = self.pid = self.n_ctx = self.build_info = self.guard = None

    def start(self) -> dict:
        rec = {"arm_key": self.arm_key, "port": self.port, "log_path": self.log_path}
        rec["port_guard"] = sg.assert_port_free(self.port)          # GuardError propagates: never reuse a listener
        cmd = server_cmd(self.arm_key, self.log_path, self.port, self.exe, self.gguf, self.req_ctx)
        rec["cmd"] = cmd
        self.proc = popen_hidden(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 stdin=subprocess.DEVNULL)
        self.pid = self.proc.pid
        t0 = time.monotonic()
        while time.monotonic() - t0 < STARTUP_TIMEOUT_S:
            if self.proc.poll() is not None:
                rec.update({"started": False, "error": f"process exited rc={self.proc.returncode}"})
                return rec
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1.0)
        else:
            self.stop()
            rec.update({"started": False, "error": f"not healthy within {STARTUP_TIMEOUT_S:.0f}s"})
            return rec
        rec["startup_s"] = time.monotonic() - t0
        m = sg.assert_server_matches(self.url, self.port, self.pid, guard_expected(self.arm_key, self.gguf,
                                                                                   self.req_ctx))
        self.n_ctx, self.build_info = m.get("props_n_ctx"), m.get("props_build_info")
        self.guard = sg.RequestGuard(self.port, self.pid)
        rec.update({"started": True, "server_pid": self.pid, "props_n_ctx": self.n_ctx, "server_guard": m,
                    "fresh_process_per_session": True})
        return rec

    def alive(self) -> bool:
        if self.proc is None or self.proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=30) as r:
                return r.status == 200
        except Exception:
            return False

    def restart(self) -> dict:
        self.stop()
        try:
            return self.start()
        except sg.GuardError as e:
            return {"started": False, "error": str(e)[:400], "guard_record": e.record}

    def stop(self) -> dict:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        return {"after_stop_listeners": sg.port_listeners(self.port)}


# ── outcome (pure; register rows xruntime-llamacpp-wrong-rate and xruntime-default-fixed) ─────────────

def _sessions(rows, arms):
    att = {}
    for s in ag.completed_sessions(rows):
        if s.get("mode") == "real" and s.get("model_id") == MODEL_ID and s.get("arm_id") in arms:
            att[s["attempt_id"]] = s
    return att


def per_session_cells(rows, arm) -> tuple[list, list, list]:
    """(sessions, task turns, [(wrong, n) per session]) for one arm of one file."""
    att = _sessions(rows, {arm})
    turns = [r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in att
             and r.get("item_kind") == "task"]
    ss = sorted(att.values(), key=lambda s: s["seed"])
    cells = []
    for s in ss:
        st = [r for r in turns if r["attempt_id"] == s["attempt_id"]]
        cells.append((sum(1 for r in st if r.get("end_state_ok") is False), len(st)))
    return ss, turns, cells


def _reading(ci, yes, no) -> str | None:
    lo = ci[0] if ci else None
    if lo is None:
        return None
    return yes if lo > 0 else no


def outcome(rows, ollama_rows=None, n_boot=EV.BOOT_N, seed=EV.BOOT_SEED) -> dict:
    """The pre-registered outcome: validity, per arm the wrong-task rate and the silent/flagged split from the server
    log, the primary excess (4096 over the server default, or over (c) if the default loaded 4096) with the session
    bootstrap and its reading, and the default-fixed arm against Ollama's default-window sessions."""
    finished = any(r.get("record") == "run_end" for r in rows)
    problems = [] if finished else ["no run_end"]
    starts = {}
    for r in rows:
        if r.get("record") == "server_start" and r.get("started"):
            starts.setdefault(r.get("arm_key"), set()).add(r.get("props_n_ctx"))
    if any(r.get("record") == "guard_stop" for r in rows):
        problems.append("request guard stopped a condition: " + "; ".join(
            str(r.get("arm_key")) for r in rows if r.get("record") == "guard_stop"))
    probes = [r for r in rows if r.get("record") == "detector_probe"]
    det_ok = detector_validated(probes)
    arms, cells = {}, {}
    for k in ARM_ORDER:
        ss, turns, c = per_session_cells(rows, ARM[k])
        cells[k] = c
        if [s["seed"] for s in ss] != list(SEEDS):
            problems.append(f"{k}: sessions completed for seeds {[s['seed'] for s in ss]}, expected {list(SEEDS)}")
        short = [s["seed"] for s, (_w, n) in zip(ss, c) if n != TURNS]
        if short:
            problems.append(f"{k}: seeds {short} without {TURNS} task turns")
        wrong = [r for r in turns if r.get("end_state_ok") is False]
        cls = {c_: sum(1 for r in wrong if r.get("context_event") == c_) for c_ in CLASS_ORDER}
        arms[k] = {"arm_id": ARM[k], "n_sessions": len(ss), "n_task_turns": len(turns), "wrong": len(wrong),
                   "rate": len(wrong) / len(turns) if turns else None,
                   "harness_silent_wrong": sum(1 for r in wrong if not r.get("any_http_error")),
                   "wrong_by_context_event": cls,
                   "silent_share": (cls["silent_cut"] / len(wrong)) if wrong and det_ok else None,
                   "context_event_turns": sum(1 for r in turns if r.get("context_event") not in (None, "none")),
                   "loaded_n_ctx": sorted(starts.get(k, set()), key=str),
                   "tool_turn_share": (sum(1 for r in turns if (r.get("tool_rounds") or 0) >= 1) / len(turns)
                                       if turns else None)}
    for k, want in (("4096", 4096), ("fixed", N_CTX_TRAIN)):
        if arms[k]["loaded_n_ctx"] and arms[k]["loaded_n_ctx"] != [want]:
            problems.append(f"{k}: loaded n_ctx {arms[k]['loaded_n_ctx']} != {want}")
    if len(arms["default"]["loaded_n_ctx"]) > 1:
        problems.append(f"default: loaded n_ctx differs between sessions {arms['default']['loaded_n_ctx']}")
    tool = arms["fixed"]["tool_turn_share"]
    tool_ok = tool is not None and tool >= TOOL_PATH_MIN
    if not tool_ok:
        problems.append(f"tool path gate: share of (c) task turns with a tool round {tool} < {TOOL_PATH_MIN}")
    default_ctx = arms["default"]["loaded_n_ctx"][0] if len(arms["default"]["loaded_n_ctx"]) == 1 else None
    comparator = "fixed" if default_ctx == 4096 else "default"
    boot = EV.session_bootstrap_excess({k: cells[k] for k in ARM_ORDER if cells[k]}, comparator, n_boot, seed)
    prim = (boot.get("per_tier") or {}).get("4096")
    valid = not problems
    primary = {"comparator": comparator, "comparator_reason": ("the server default loaded 4096, so (a) and (b) are one "
                                                               "config; pre-registered fallback to (c)")
               if comparator == "fixed" else "the server default (a)", "excess": prim}
    if prim is None:
        primary["reading"] = "not read (no sessions to compare)"
    elif not valid:
        primary["reading"] = "not read (invalid: " + "; ".join(problems)[:300] + ")"
    else:
        primary["reading"] = _reading(prim["ci95"], "replicated", "not replicated")
    out = {"run_finished": finished, "valid": valid, "problems": problems, "detector_validated": det_ok,
           "detector_probes": {p.get("probe"): p.get("class") for p in probes}, "default_loaded_n_ctx": default_ctx,
           "arms": arms, "primary": primary, "excess_all": boot}
    fixed = {"arm": arms["fixed"], "persists": None}
    if ollama_rows is not None:
        _oss, _ot, oc = per_session_cells(ollama_rows, OLLAMA_DEFAULT_ARM)
        b2 = EV.session_bootstrap_excess({"ollama_default": oc, "fixed": cells["fixed"]}, "ollama_default", n_boot,
                                         seed) if oc and cells["fixed"] else {}
        e = (b2.get("per_tier") or {}).get("fixed")
        reading = None
        if e is not None:
            lo, hi = e["ci95"]
            reading = ("matches Ollama's default-window rate" if lo is not None and lo <= 0 <= hi else
                       "higher on llama.cpp at the fixed window" if lo is not None and lo > 0 else
                       "lower on llama.cpp at the fixed window" if hi is not None else None)
            if not valid:
                reading = "not read (invalid: " + "; ".join(problems)[:300] + ")"
        fixed["persists"] = {"vs_ollama_default": e, "reading": reading, "ollama_sessions": len(oc)}
    out["default_fixed"] = fixed
    return out


def hours_estimate(b3_rows, outcome_rows, spill_rows) -> dict:
    """ESTIMATE (register row xruntime-llamacpp-hours): 150 task turns per arm at the B3 v2 Ollama run's measured mean
    turn wall (arms a and c: ollama_default, whose window is 40960; arm b: ollama_ctx_4096), times the measured
    llama_server / Ollama latency ratio for qwen3-14b on matched x2_outcome_table_v3 items, plus one start per session
    and the probe start at the largest measured start-and-stop overhead of the qwen3-32b spill sweep (a larger GGUF),
    plus PROBE_S (ASSUMED) for the two probe calls; high = point x HIGH_FACTOR (ASSUMED)."""
    walls = {}
    for r in b3_rows:
        if r.get("record") == "r2a_turn" and r.get("model_id") == MODEL_ID:
            walls.setdefault(r.get("arm_id"), []).append(r.get("turn_wall_s") or 0.0)
    wd = sum(walls[OLLAMA_DEFAULT_ARM]) / len(walls[OLLAMA_DEFAULT_ARM])
    a4 = "ollama_ctx_4096" + SFX
    w4 = sum(walls[a4]) / len(walls[a4])
    pair = {}
    for r in outcome_rows:
        if r.get("record") == "outcome_row" and r.get("model_id") == "qwen3-14b" and r.get("http_status") == 200 \
                and r.get("latency_s") is not None and r.get("config") in ("llama_server", "ollama_default"):
            pair.setdefault(r["item_id"], {})[r["config"]] = r["latency_s"]
    both = [p for p in pair.values() if len(p) == 2]
    ratio = sum(p["llama_server"] for p in both) / sum(p["ollama_default"] for p in both)
    calls = [r for r in spill_rows if r.get("record") == "sweep_call" and r.get("ok")]
    over = []
    for p in (r for r in spill_rows if r.get("record") == "sweep_point" and r.get("started")):
        lat = sum(c["latency_ms"] for c in calls if c["requested_n_ctx"] == p["requested_n_ctx"]) / 1000.0
        over.append(p["wall_s"] - lat)
    start_s = max(over)
    n_turns = len(SEEDS) * TURNS
    n_starts = len(SEEDS) * len(ARM_ORDER) + 1
    sess_h = ratio * n_turns * (wd + w4 + wd) / 3600.0
    point = sess_h + n_starts * start_s / 3600.0 + PROBE_S / 3600.0
    return {"mean_turn_wall_default_s": wd, "mean_turn_wall_4096_s": w4, "n_turns_per_arm": n_turns,
            "latency_ratio": ratio, "n_ratio_items": len(both), "start_overhead_s": start_s, "n_starts": n_starts,
            "probe_s_assumed": PROBE_S, "high_factor_assumed": HIGH_FACTOR, "session_h": sess_h, "point_h": point,
            "high_h": point * HIGH_FACTOR}


PROBE_S = 180.0
HIGH_FACTOR = 1.5


# ── probes, check-plan, main ──────────────────────────────────────────────────────────────────────

def run_probes(log_dir: Path, emit, log) -> list[dict]:
    """The detector probe on its own server start at -c PROBE_CTX: one prompt over the window, one generation over the
    window. Each probe's server log slice is classified exactly as a session call is."""
    srv = LlamaServer(None, log_dir / "xruntime_probe_server.log", n_ctx=PROBE_CTX)
    start = srv.start()
    emit({"record": "server_start", "purpose": "detector_probe", **start, "ts_utc": ag.utc_iso()})
    out = []
    try:
        if not start.get("started"):
            return out
        rt = LlamaCppRuntime(srv)
        long_text = " ".join(f"item{i}" for i in range(PROBE_CTX * 2))
        for name, msgs, max_tok in (
                ("prompt_over_window", [{"role": "user", "content": "Repeat the last word: " + long_text}], 16),
                ("generation_over_window", [{"role": "user", "content": "Count upward from 1, one number per line, "
                                                                        "and do not stop."}], PROBE_CTX * 2)):
            rt.max_tokens = max_tok
            rt.chat(MODEL_ID, msgs, None, None, False)
            ev = rt.pop_turn_events()[-1]
            row = {"record": "detector_probe", "probe": name, "max_tokens": max_tok, "n_ctx": srv.n_ctx, **ev,
                   "ts_utc": ag.utc_iso()}
            emit(row)
            out.append(row)
            log(f"probe {name}: http {ev['http_status']} class {ev['class']} matches {list(ev['log_matches'])}")
    finally:
        emit({"record": "server_stop", "purpose": "detector_probe", **srv.stop(), "ts_utc": ag.utc_iso()})
    return out


def server_help(exe=LLAMA_SERVER_EXE) -> dict:
    if not Path(exe).exists():
        return {"error": f"{exe} not found"}
    try:
        r = run_hidden([exe, "--help"], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        return help_lines((r.stdout or "") + "\n" + (r.stderr or ""))
    except Exception as e:
        return {"error": repr(e)[:300]}


def check_plan(run_help=True) -> dict:
    sessions = []
    for arm, seeds, turns in plan():
        for s in seeds:
            es = EV.build_session(s, turns, "evalset", False, model=MODEL_ID, design=DESIGN)
            sessions.append({"arm": arm, "seed": s, "turns": len(es.turns),
                             "task_turns": sum(1 for i in es.items if i.kind == "task"),
                             "projected_max_prompt": EV.projected_max_prompt(es)})
    g = Path(GGUF)
    return {"harness": HARNESS, "model_id": MODEL_ID, "plan": plan(), "arm_order": ARM_ORDER, "sessions": sessions,
            "server_cmds": {k: server_cmd(k, "<log>") for k in ARM_ORDER},
            "probe_cmd": server_cmd(None, "<log>", n_ctx=PROBE_CTX),
            "exe": LLAMA_SERVER_EXE, "exe_exists": Path(LLAMA_SERVER_EXE).exists(), "gguf": GGUF,
            "gguf_exists": g.exists(), "gguf_bytes": g.stat().st_size if g.exists() else None,
            "gguf_bytes_expected": GGUF_BYTES, "port": PORT, "port_listeners": sg.port_listeners(PORT),
            "server_help": server_help() if run_help else None, "tool_path_min": TOOL_PATH_MIN,
            "detector": {"reject": REJECT_RE.pattern, "shift": SHIFT_RE.pattern, "trunc": TRUNC_RE.pattern,
                         "ctx_stop": CTX_STOP_RE.pattern}}


def sha256_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ollama_loaded():
    try:
        with urllib.request.urlopen(OLLAMA_PS, timeout=5) as r:
            return [m.get("name") or m.get("model") for m in (json.loads(r.read()).get("models") or [])]
    except Exception:
        return None


def provenance() -> dict:
    mods = {"x2_r2_evalset": EV, "x2_r2_agent": ag, "evalset_env": EV.E, "server_guard": sg}
    return {"harness_file": str(Path(__file__).resolve()), "harness_blob": git_blob_sha1(__file__),
            "modules": {n: {"file": str(Path(m.__file__).resolve()), "blob": git_blob_sha1(m.__file__)}
                        for n, m in mods.items()}}


def run_arm_session(k, seed, out_path, emit, log, server_factory=LlamaServer, post=None, settle_s=None) -> str:
    """One session of arm k on a fresh server. 'ok' | 'guard_stop' | 'start_failed'."""
    srv = server_factory(k, out_path.parent / f"{out_path.stem}_{BASES[k][0]}_{seed}.log")
    try:
        start = srv.start()
    except sg.GuardError as e:
        emit({"record": "server_start", "arm_key": k, "seed": seed, "started": False, "error": str(e)[:400],
              "guard_record": e.record, "ts_utc": ag.utc_iso()})
        raise
    emit({"record": "server_start", "seed": seed, **start, "ts_utc": ag.utc_iso()})
    try:
        if not start.get("started"):
            return "start_failed"
        rt = LlamaCppRuntime(srv, emit, post=post, settle_s=settle_s)

        def emit_turn(row):
            if row.get("record") == "r2a_turn":
                ev = rt.pop_turn_events()
                row.update({"llamacpp_calls": ev, "context_event": turn_class(ev), "runtime": "llama-server",
                            "server_pid": srv.pid, "server_n_ctx": srv.n_ctx, "arm_key": k})
            emit(row)
        try:
            EV.run_session(rt, MODEL_ID, ARM[k], seed, TURNS, emit_turn, log, "real", "evalset", False, None, DESIGN)
        except GuardMismatch as e:
            emit({"record": "guard_stop", "arm_key": k, "seed": seed, "error": str(e), "ts_utc": ag.utc_iso()})
            return "guard_stop"
        return "ok"
    finally:
        emit({"record": "server_stop", "arm_key": k, "seed": seed, **srv.stop(), "ts_utc": ag.utc_iso()})


def main(argv=None, advance=True):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-plan", action="store_true")
    args = ap.parse_args(argv)
    if args.check_plan:
        print(json.dumps(check_plan(), indent=1, default=str))
        return "checked"
    if not args.out:
        ap.error("--out is required for a run")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def log(msg):
        print(f"[{ag.utc_iso()}] {msg}", flush=True)
    note = "completed"
    try:
        import host_config as hc
        hostname = socket.gethostname()
        if hostname.upper() != "EVO-X2":
            raise RuntimeError(f"{HARNESS} runs on EVO-X2 only, this is {hostname!r}")
        try:
            host_cfg = hc.require_host(hostname)
            interactive = hc.enforce_or_record_interactive_session(host_cfg)
        except SystemExit as e:
            note = f"stopped: {e}"[:400]
            log(note)
            return note
        if not Path(LLAMA_SERVER_EXE).exists() or not Path(GGUF).exists():
            raise RuntimeError(f"missing: exe {Path(LLAMA_SERVER_EXE).exists()} gguf {Path(GGUF).exists()}")
        size = Path(GGUF).stat().st_size
        digest = sha256_file(GGUF)
        gguf_ok = size == GGUF_BYTES and digest.startswith(GGUF_SHA256_HEAD) and digest.endswith(GGUF_SHA256_TAIL)
        emit({"record": "run_start", "harness": HARNESS, "model_id": MODEL_ID, "plan": plan(), "arm_order": ARM_ORDER,
              "arms": {k: {"arm_id": ARM[k], "n_ctx_flag": BASES[k][1]} for k in ARM_ORDER},
              "server_cmds": {k: server_cmd(k, "<log>") for k in ARM_ORDER}, "exe": LLAMA_SERVER_EXE, "gguf": GGUF,
              "gguf_bytes": size, "gguf_sha256": digest, "gguf_ok": gguf_ok, "port": PORT, "design": DESIGN,
              "workload": "evalset", "guardrails": False, "call2_mode": CALL2, "turn_protocol": EV.TURN_PROTOCOL[DESIGN],
              "max_tool_rounds": EV.MAX_TOOL_ROUNDS, "max_tokens_per_call": ag.MAX_TOKENS_PER_CALL,
              "sampling": {"temperature": 0, "seed": 42}, "think": False, "server_help": server_help(),
              "detector": check_plan(run_help=False)["detector"], "tool_path_min": TOOL_PATH_MIN,
              "ollama_loaded_models": ollama_loaded(), "hw_id": host_cfg.get("hw_id"),
              "interactive_session": interactive, **provenance(), "ts_utc": ag.utc_iso()})
        if not gguf_ok:
            raise RuntimeError(f"GGUF size/hash mismatch: {size} {digest}")
        done = {s["arm_id"] + "|" + str(s["seed"]) for s in _sessions(ag.read_rows(out_path), set(ARM.values()))
                .values()}
        if not any(r.get("record") == "detector_probe" for r in ag.read_rows(out_path)):
            probes = run_probes(out_path.parent, emit, log)
            emit({"record": "detector_check", "validated": detector_validated(probes), "ts_utc": ag.utc_iso()})
        for k in ARM_ORDER:
            for seed in SEEDS:
                if f"{ARM[k]}|{seed}" in done:
                    log(f"skip (done): {ARM[k]} seed={seed}")
                    continue
                log(f"session: {ARM[k]} seed={seed}")
                res = run_arm_session(k, seed, out_path, emit, log)
                if res != "ok":
                    log(f"{k}: {res}; condition stopped")
                    break
        rows = ag.read_rows(out_path)
        ref = out_path.parent / OLLAMA_REF_FILE
        report = outcome(rows + [{"record": "run_end"}], ag.read_rows(ref) if ref.exists() else None)
        Path(str(out_path) + ".report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        emit({"record": "run_end", "primary_reading": report["primary"]["reading"], "ts_utc": ag.utc_iso()})
        log(json.dumps(report["primary"], default=str)[:2000])
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        log(note)
        traceback.print_exc()
    finally:
        if advance:
            try:
                import t2s_queue as tq
                tq.advance(note)
            except Exception as e:
                log(f"queue advance failed: {e!r}")
    return note


if __name__ == "__main__":
    main()
