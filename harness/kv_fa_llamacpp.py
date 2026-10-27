"""KV precision x flash attention on llama-server (docs/EXPERIMENTS.md, "PRE-REGISTRATION (2026-10-26): KV precision x
flash attention on llama.cpp server"; queue id x2_kv_fa_llamacpp). Register row kv-fa-interaction-llamacpp reads its
file; row kv-fa-interaction-llamacpp-hours is the ESTIMATE.

Why: Ollama cannot run a quantized KV cache with flash attention off, so B5 on Ollama (policies (a) f16, (i) f16 + FA,
(g) q8_0 + FA, (h) q4_0 + FA) never had the fourth cell of the KV x FA square. llama-server takes -ctk, -ctv and -fa
separately, so the square can be filled where llama.cpp allows it.

Legal cells on b10970 (bfdc32183): its llama.dll holds the error string "quantized V cache requires flash_attn to be
enabled" (and "enabling flash_attn since it is required for quantized V cache" for -fa auto), so a quantized V cache
with -fa off is expected to be refused at load. The closest legal cell quantizes K only. Cells (key: -ctk/-ctv/-fa):
  f16_fa_off  f16 / f16  / off
  f16_fa_on   f16 / f16  / on
  q8k_fa_off  q8_0 / f16 / off   the legal stand-in for q8_0 with FA off (K quantized, V f16)
  q8k_fa_on   q8_0 / f16 / on    completes the K-only square: the primary interaction is tested on it
  q8_fa_on    q8_0 / q8_0 / on   the B5 (g) setting on llama.cpp (link to Ollama, and the operator's literal square)
  q8_fa_off   q8_0 / q8_0 / off  attempted once before any session (REFUSAL_CELL); recorded "refused" if the server
                                 does not load; if it loads (not expected), it is run as a sixth cell
Any cell whose start fails with a refusal line is recorded as refused (a result), not as a crash.

Workload: the B5 R2 session (harness/x2_r2_agent.run_session, call-2 mode off, temperature 0, seed 42, max_tokens 384,
seeds 20260901-03, qwen3:14b think off) cut at TURNS = 24 turns: a 24-turn session is the exact prefix of B5's 40-turn
session (same system prompt, user messages, tool plan; checked by tests), and turn 24 is the last turn whose prompt in
B5 on Ollama stayed at least 10% inside 40960 (B5 (a) max prompt_eval_count 36,596 at turn 24, 39,766 at turn 26, then
capped by Ollama's silent truncation). llama-server rejects an over-window prompt with HTTP 400 (row
xruntime-llamacpp-wrong-rate, arm b), so B5's full 40 turns would turn every late call into an instant error.
Window fixed at -c 40960 (the GGUF's n_ctx_train, B5's window for qwen3:14b), asserted from /props at every start.

Per start: the server log (--log-verbosity 4, the level at which this build writes the llama_context and llama_kv_cache
lines; x2_70b_edge_reps read KV buffer lines at 4 on evo-x2) is parsed for the K and V types, the KV buffer MiB and the
flash_attn line (parse_kv_log), checked against the cell before the session (kv_check); stderr goes to its own file.
Per call: llama-server's timings (prompt_n, prompt_ms, predicted_n, predicted_ms) for prefill and decode tok/s, and the
xruntime_llamacpp server-log call class (any class other than none, or any HTTP error, invalidates that cell).

Stale-server guard (harness/server_guard.py): port free before each start; /props model path, slot count and n_ctx plus
the command line (-ctk, -ctv, -fa) asserted after each start; the listening PID checked before every request.

    python kvfa\\kv_fa_llamacpp.py --check-plan
        plan, cells and run order, server command lines, exe and GGUF presence and size, port listeners, the server's
        --help lines for -fa, -ctk, -ctv; no server start, no model call, no queue access
    python kvfa\\kv_fa_llamacpp.py --out C:\\apu\\ovn\\results\\kv_fa_llamacpp_qwen3_14b.jsonl

Queue job: calls t2s_queue.advance() exactly once on exit (finally); sets no pause. Own port PORT.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import socket
import statistics as st
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if not (_here / "t2s_queue.py").exists():
    sys.path.insert(1, str(_here.parent))

import xruntime_llamacpp as XR  # noqa: E402  (imports server_guard, proc_util, x2_r2_evalset, x2_r2_agent)

sg, ag = XR.sg, XR.ag
popen_hidden, run_hidden = XR.popen_hidden, XR.run_hidden

HARNESS = "kv_fa_llamacpp"
LLAMA_SERVER_EXE, GGUF, GGUF_BYTES = XR.LLAMA_SERVER_EXE, XR.GGUF, XR.GGUF_BYTES
MODEL_ID = "qwen3:14b"
PORT = 58423                  # own port (xruntime_llamacpp uses 58421)
N_CTX = 40960
SEEDS = tuple(ag.SEEDS)       # 20260901, 20260902, 20260903 (B5's seeds)
TURNS = 24
B5_TURNS = 40
CALL2 = "off"
SFX = ag.MODE_SUFFIX[CALL2]
MODE = "kvfa"
CELLS = {
    "f16_fa_off": {"ctk": "f16", "ctv": "f16", "fa": "off"},
    "f16_fa_on": {"ctk": "f16", "ctv": "f16", "fa": "on"},
    "q8k_fa_off": {"ctk": "q8_0", "ctv": "f16", "fa": "off"},
    "q8k_fa_on": {"ctk": "q8_0", "ctv": "f16", "fa": "on"},
    "q8_fa_on": {"ctk": "q8_0", "ctv": "q8_0", "fa": "on"},
    "q8_fa_off": {"ctk": "q8_0", "ctv": "q8_0", "fa": "off"},
}
CELL_ORDER = ("f16_fa_off", "f16_fa_on", "q8k_fa_off", "q8k_fa_on", "q8_fa_on")
REFUSAL_CELL = "q8_fa_off"
SQUARE_K = {("f16", "off"): "f16_fa_off", ("f16", "on"): "f16_fa_on", ("q8", "off"): "q8k_fa_off",
            ("q8", "on"): "q8k_fa_on"}                      # primary: K-only square, V f16 throughout
SQUARE_OPERATOR = {("f16", "off"): "f16_fa_off", ("f16", "on"): "f16_fa_on", ("q8", "off"): "q8k_fa_off",
                   ("q8", "on"): "q8_fa_on"}                # the operator's square with the legal stand-in (confounded)
SQUARE_KV = {("f16", "off"): "f16_fa_off", ("f16", "on"): "f16_fa_on", ("q8", "off"): "q8_fa_off",
             ("q8", "on"): "q8_fa_on"}                      # only if REFUSAL_CELL loads
ARM = {k: f"llamacpp_kv_{k}{SFX}" for k in CELLS}
for _k in CELLS:
    ag.ARMS.setdefault(ARM[_k], {"num_ctx": N_CTX, "call2_tools": False, "call2_mode": CALL2})
TOL = 0.10                    # B5's tolerance (fa_reading); on the ratio scale: |ln ratio| <= ln 1.10
MEM_TOL_SHARE = 0.01          # memory interaction tolerance: 1% of the f16 FA-on KV buffer
BOOT_N, BOOT_SEED = XR.EV.BOOT_N, XR.EV.BOOT_SEED
STARTUP_TIMEOUT_S = 300.0
KV_LOG_WAIT_S = 30.0
REFUSAL_RE = re.compile(r"quantized V cache requires flash_attn|requires flash_attn to be enabled|"
                        r"does not support different K \(\S+\) and V \(\S+\) cache types|"
                        r"failed to create context|unsupported cache type", re.IGNORECASE)
KV_SIZE_RE = re.compile(r"llama_kv_cache:\s+size\s*=\s*([\d.]+)\s*MiB.*?K \((\w+)\):\s*([\d.]+)\s*MiB,\s*"
                        r"V \((\w+)\):\s*([\d.]+)\s*MiB")
KV_BUF_RE = re.compile(r"(\S+)\s+KV buffer size\s*=\s*([\d.]+)\s*MiB")
COMPUTE_BUF_RE = re.compile(r"(\S+)\s+compute buffer size\s*=\s*([\d.]+)\s*MiB")
FA_LINE_RE = re.compile(r"llama_context:\s+flash_attn\s*=\s*(\w+)")
HELP_KEYS = ("--flash-attn", "--cache-type-k", "--cache-type-v", "--ctx-size", "--parallel", "--jinja")
B5_FILE, B5_FA_FILE = "x2_policy_compare.jsonl", "x2_policy_compare_fa_control.jsonl"
B5_ARMS = {"a": "b5_a_default" + SFX, "i": "b5_i_kv_f16_fa" + SFX, "g": "b5_g_kv_q8_0" + SFX,
           "h": "b5_h_kv_q4_0" + SFX}
B5_RULES_QWEN = ("rule1_json_keys", "rule2_log_event_called", "rule3_no_zebra", "rule4_metres", "rule5_session_code")


# ── pure parts ────────────────────────────────────────────────────────────────────────────────────

def server_cmd(cell: str, log_path: str, port: int = PORT, exe: str = LLAMA_SERVER_EXE, gguf: str = GGUF) -> list:
    c = CELLS[cell]
    return [exe, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "--no-webui", "-ngl", "99", "-np", "1",
            "--jinja", "--reasoning-format", "deepseek", "--reasoning-budget", "0", "--log-file", log_path,
            "--log-verbosity", "4", "-c", str(N_CTX), "-ctk", c["ctk"], "-ctv", c["ctv"], "-fa", c["fa"]]


def guard_expected(cell: str, gguf: str = GGUF) -> dict:
    c = CELLS[cell]
    return {"model_path": gguf, "np": 1, "ngl": 99, "n_ctx": N_CTX, "ctk": c["ctk"], "ctv": c["ctv"], "fa": c["fa"]}


def run_order(seeds=SEEDS, cells=CELL_ORDER) -> list[tuple]:
    """Seed-major (a partial run keeps every cell balanced over the seeds done), the cell order rotated by one per seed
    so no cell always runs first or last within a seed."""
    out = []
    for i, s in enumerate(seeds):
        r = i % len(cells)
        out += [(c, s) for c in tuple(cells[r:]) + tuple(cells[:r])]
    return out


def _sum_dev(rx, text):
    d = {}
    for dev, val in rx.findall(text):
        d[dev] = float(val)
    return sum(d.values()) if d else None


def parse_kv_log(text: str) -> dict:
    """K and V cache types and sizes, the KV and compute buffer MiB (summed over devices, last value per device) and
    the flash_attn line, from a llama-server load log."""
    t = re.sub(r"\x1b\[[0-9;]*m", "", text or "")
    out = {"kv_size_mib": None, "k_type": None, "k_mib": None, "v_type": None, "v_mib": None,
           "kv_buffer_mib": _sum_dev(KV_BUF_RE, t), "compute_buffer_mib": _sum_dev(COMPUTE_BUF_RE, t),
           "flash_attn": None}
    m = None
    for m in KV_SIZE_RE.finditer(t):
        pass
    if m:
        out.update({"kv_size_mib": float(m.group(1)), "k_type": m.group(2), "k_mib": float(m.group(3)),
                    "v_type": m.group(4), "v_mib": float(m.group(5))})
    fa = FA_LINE_RE.findall(t)
    if fa:
        out["flash_attn"] = fa[-1].lower()
    return out


def kv_check(cell: str, parsed: dict) -> dict:
    """'ok' iff the log's K and V types are the cell's and its flash_attn line (when written) agrees with -fa;
    'absent' iff the llama_kv_cache line is missing; 'mismatch' otherwise."""
    c = CELLS[cell]
    if parsed.get("k_type") is None or parsed.get("v_type") is None:
        return {"status": "absent", "reason": "no llama_kv_cache size line in the log"}
    want_fa = {"on": "enabled", "off": "disabled"}[c["fa"]]
    bad = []
    if parsed["k_type"] != c["ctk"]:
        bad.append(f"K {parsed['k_type']} != {c['ctk']}")
    if parsed["v_type"] != c["ctv"]:
        bad.append(f"V {parsed['v_type']} != {c['ctv']}")
    if parsed.get("flash_attn") not in (None, want_fa):
        bad.append(f"flash_attn {parsed['flash_attn']} != {want_fa}")
    return {"status": "mismatch" if bad else "ok", "reason": "; ".join(bad) or None,
            "flash_attn_line_present": parsed.get("flash_attn") is not None}


def refusal_class(started: bool, log_text: str) -> dict:
    """refused (the server exited with a refusal line) | accepted (it loaded) | start_failed (exited, no refusal
    line: reported as a failure, not as a refusal)."""
    lines = [ln.strip()[:300] for ln in (log_text or "").splitlines() if REFUSAL_RE.search(ln)]
    if started:
        return {"result": "accepted", "lines": lines[:5]}
    return {"result": "refused" if lines else "start_failed", "lines": lines[:5]}


def call_timings(t) -> dict | None:
    if not t:
        return None
    return {k: t.get(k) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n")}


def help_lines(text: str) -> dict:
    keys_save = XR.HELP_KEYS
    try:
        XR.HELP_KEYS = HELP_KEYS
        return XR.help_lines(text)
    finally:
        XR.HELP_KEYS = keys_save


# ── runtime and server ────────────────────────────────────────────────────────────────────────────

class KvRuntime(XR.LlamaCppRuntime):
    """xruntime_llamacpp's adapter, plus llama-server's per-call timings on each call event."""

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        resp = super().chat(model, messages, num_ctx, tools, think, extra)
        if self._events:
            self._events[-1]["timings"] = call_timings((resp.get("raw") or {}).get("timings"))
        return resp


class KvServer(XR.LlamaServer):
    """One llama-server process for one KV x FA cell, guarded per harness/server_guard.py; stderr to its own file;
    the load log parsed for the KV lines after the start."""

    def __init__(self, cell, log_path, port=PORT, exe=LLAMA_SERVER_EXE, gguf=GGUF):
        super().__init__(cell, log_path, port=port, n_ctx=N_CTX, exe=exe, gguf=gguf)
        self.cell = cell
        self.stderr_path = str(log_path) + ".stderr.txt"

    def _log_text(self) -> str:
        out = []
        for p in (self.log_path, self.stderr_path):
            try:
                out.append(Path(p).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        return "\n".join(out)

    def start(self) -> dict:
        rec = {"cell": self.cell, **CELLS[self.cell], "port": self.port, "log_path": self.log_path}
        rec["port_guard"] = sg.assert_port_free(self.port)          # GuardError propagates: never reuse a listener
        cmd = server_cmd(self.cell, self.log_path, self.port, self.exe, self.gguf)
        rec["cmd"] = cmd
        with open(self.stderr_path, "w", encoding="utf-8") as ef:
            self.proc = popen_hidden(cmd, stdout=subprocess.DEVNULL, stderr=ef, stdin=subprocess.DEVNULL)
        self.pid = self.proc.pid
        t0 = time.monotonic()
        while time.monotonic() - t0 < STARTUP_TIMEOUT_S:
            if self.proc.poll() is not None:
                time.sleep(0.5)
                rec.update({"started": False, "exit_rc": self.proc.returncode,
                            "error": f"process exited rc={self.proc.returncode}",
                            "refusal": refusal_class(False, self._log_text())})
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
            rec.update({"started": False, "error": f"not healthy within {STARTUP_TIMEOUT_S:.0f}s",
                        "refusal": refusal_class(False, self._log_text())})
            return rec
        rec["startup_s"] = time.monotonic() - t0
        m = sg.assert_server_matches(self.url, self.port, self.pid, guard_expected(self.cell, self.gguf))
        self.n_ctx, self.build_info = m.get("props_n_ctx"), m.get("props_build_info")
        self.guard = sg.RequestGuard(self.port, self.pid)
        parsed = parse_kv_log(self._log_text())
        t1 = time.monotonic()
        while parsed["k_type"] is None and time.monotonic() - t1 < KV_LOG_WAIT_S:
            time.sleep(1.0)
            parsed = parse_kv_log(self._log_text())
        rec.update({"started": True, "server_pid": self.pid, "props_n_ctx": self.n_ctx, "server_guard": m,
                    "kv_log": parsed, "kv_check": kv_check(self.cell, parsed), "fresh_process_per_session": True,
                    "refusal": refusal_class(True, self._log_text())})
        return rec


# ── outcome (pure; register row kv-fa-interaction-llamacpp) ──────────────────────────────────────────

def _boot_ci(vals, lo=0.025, hi=0.975):
    v = sorted(vals)
    if not v:
        return [None, None]
    return [v[int(lo * (len(v) - 1))], v[int(math.ceil(hi * (len(v) - 1)))]]


def interaction(per_cell: dict, square: dict, n_boot=BOOT_N, seed=BOOT_SEED) -> dict | None:
    """I = ln(mean q8 / mean f16 at FA off) - ln(mean q8 / mean f16 at FA on) over per-session values, with a
    bootstrap that resamples sessions within each cell (stratified by cell). None if a cell has no sessions."""
    cells = {k: per_cell.get(c) or [] for k, c in square.items()}
    if not all(cells.values()) or any(v <= 0 for vs in cells.values() for v in vs):
        return None

    def stat(c):
        r_off = st.mean(c[("q8", "off")]) / st.mean(c[("f16", "off")])
        r_on = st.mean(c[("q8", "on")]) / st.mean(c[("f16", "on")])
        return math.log(r_off) - math.log(r_on), r_off, r_on
    i0, r_off, r_on = stat(cells)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        b = {k: [rng.choice(v) for _ in v] for k, v in cells.items()}
        boots.append(stat(b)[0])
    return {"I": i0, "ratio_fa_off": r_off, "ratio_fa_on": r_on, "ci95": _boot_ci(boots),
            "n_sessions": {f"{k[0]}_fa_{k[1]}": len(v) for k, v in cells.items()}, "n_boot": n_boot, "seed": seed}


def interaction_reading(e: dict | None, tol=TOL) -> str:
    if e is None:
        return "not read (a cell has no sessions)"
    lo, hi = e["ci95"]
    band = math.log(1 + tol)
    if (lo > 0 or hi < 0) and abs(e["I"]) > band:
        return ("interaction: the q8_0 effect differs with flash attention (" +
                ("q8_0 helps less or costs more with FA off" if e["I"] > 0 else "q8_0 helps more with FA off") + ")")
    if -band <= lo and hi <= band:
        return "no interaction shown: the q8_0 effect is the same with FA on and off within 10%"
    return "inconclusive"


def memory_interaction(kv: dict, square: dict, tol_share=MEM_TOL_SHARE) -> dict | None:
    """d_f = KV MiB(q8 at f) - KV MiB(f16 at f); interaction d_off - d_on; one deterministic value per cell."""
    v = {k: kv.get(c) for k, c in square.items()}
    if any(x is None for x in v.values()):
        return None
    d_off, d_on = v[("q8", "off")] - v[("f16", "off")], v[("q8", "on")] - v[("f16", "on")]
    diff = d_off - d_on
    tol = tol_share * v[("f16", "on")]
    return {"kv_mib": {f"{k[0]}_fa_{k[1]}": x for k, x in v.items()}, "d_fa_off_mib": d_off, "d_fa_on_mib": d_on,
            "interaction_mib": diff, "tolerance_mib": tol,
            "reading": ("no memory interaction: the q8_0 KV saving is the same with FA on and off" if abs(diff) <= tol
                        else "memory interaction: the q8_0 KV saving differs with flash attention")}


def _session_values(turns: list[dict], rules) -> dict:
    m = {"minutes": sum((t.get("turn_wall_s") or 0.0) for t in turns) / 60.0}
    pn = pm = dn = dm = 0.0
    for t in turns:
        for ev in t.get("llamacpp_calls") or []:
            tm = ev.get("timings") or {}
            if tm.get("prompt_ms"):
                pn, pm = pn + (tm.get("prompt_n") or 0), pm + tm["prompt_ms"]
            if tm.get("predicted_ms"):
                dn, dm = dn + (tm.get("predicted_n") or 0), dm + tm["predicted_ms"]
    m["prefill_tok_s"] = pn / pm * 1000.0 if pm else None
    m["decode_tok_s"] = dn / dm * 1000.0 if dm else None
    m["rule_pass"] = sum(1 for t in turns for r in rules if t.get(r) is True)
    m["rule_n"] = sum(1 for t in turns for r in rules if t.get(r) is not None)
    can = [t for t in turns if t.get("canary_check")]
    m["hist_ok"], m["canary_turns"] = sum(1 for t in can if t.get("canary_hist_ok") is True), len(can)
    m["bad_calls"] = sum(1 for t in turns for ev in t.get("llamacpp_calls") or [] if ev.get("class") != "none")
    m["http_error_turns"] = sum(1 for t in turns if t.get("any_http_error"))
    return m


def b5_prefix_minutes(b5_rows: list[dict], turns=TURNS) -> dict:
    """Ollama B5 reference (context, not tested): per policy (a), (i), (g), (h), per seed the minutes of turns 1..turns
    of qwen3:14b's 40-turn sessions, and the mean."""
    out = {}
    for p, arm in B5_ARMS.items():
        per = {}
        for r in b5_rows:
            if (r.get("record") == "r2a_turn" and r.get("model_id") == MODEL_ID and r.get("arm_id") == arm
                    and (r.get("turn_idx") or 0) <= turns):
                per.setdefault(r["seed"], []).append(r.get("turn_wall_s") or 0.0)
        per = {s: sum(v) / 60.0 for s, v in sorted(per.items()) if len(v) == turns}
        out[p] = {"per_seed_min": per, "mean_min": st.mean(per.values()) if per else None}
    return out


def outcome(rows: list[dict], b5_rows: list[dict] | None = None, n_boot=BOOT_N, seed=BOOT_SEED) -> dict:
    finished = any(r.get("record") == "run_end" for r in rows)
    problems = [] if finished else ["no run_end"]
    starts = [r for r in rows if r.get("record") == "server_start"]
    ref = next((r for r in rows if r.get("record") == "refusal_probe"), None)
    refusal = (ref or {}).get("result")
    if ref is None:
        problems.append("no refusal probe record")
    cells_run = list(CELL_ORDER) + ([REFUSAL_CELL] if refusal == "accepted" else [])
    if any(r.get("record") == "guard_stop" for r in rows):
        problems.append("request guard stopped a cell: " + "; ".join(
            str(r.get("cell")) for r in rows if r.get("record") == "guard_stop"))
    turns_by = {}
    for r in rows:
        if r.get("record") == "r2a_turn":
            turns_by.setdefault(r.get("attempt_id"), []).append(r)
    sessions = [s for s in ag.completed_sessions(rows) if s.get("mode") == MODE and s.get("model_id") == MODEL_ID]
    rules = list(B5_RULES_QWEN)
    vals_by, kv, cells = {}, {}, {}
    for c in cells_run:
        ss = {}
        for s in sessions:
            if s.get("arm_id") == ARM[c] and len(turns_by.get(s["attempt_id"], [])) == TURNS:
                ss[s["seed"]] = s["attempt_id"]
        cp = []
        if sorted(ss) != sorted(SEEDS):
            cp.append(f"sessions completed for seeds {sorted(ss)}, expected {list(SEEDS)}")
        vals = {sd: _session_values(turns_by[a], rules) for sd, a in sorted(ss.items())}
        vals_by[c] = vals
        cs = [r for r in starts if r.get("cell") == c and r.get("purpose") != "refusal_probe" and r.get("started")]
        stops = [r for r in rows if r.get("record") == "server_stop" and r.get("cell") == c and r.get("kv_log_after")]
        ctx = sorted({r.get("props_n_ctx") for r in cs}, key=str)
        if ctx and ctx != [N_CTX]:
            cp.append(f"loaded n_ctx {ctx} != {N_CTX}")
        chk = [(r.get("kv_check") or {}).get("status") for r in cs]
        chk_after = [(r.get("kv_check_after") or {}).get("status") for r in stops]
        if any(x == "mismatch" for x in chk + chk_after) or not any(x == "ok" for x in chk + chk_after) and cs:
            cp.append(f"KV check at start {sorted(set(map(str, chk)))}, after {sorted(set(map(str, chk_after)))}")
        kvs = sorted({(r.get("kv_log") or {}).get("kv_buffer_mib") for r in cs}
                     | {(r.get("kv_log_after") or {}).get("kv_buffer_mib") for r in stops}, key=str)
        kvs = [x for x in kvs if x is not None]
        if len(kvs) > 1:
            cp.append(f"KV buffer MiB differs between starts {kvs}")
        kv[c] = kvs[0] if len(kvs) == 1 else None
        bad = sum(v["bad_calls"] + v["http_error_turns"] for v in vals.values())
        if bad:
            cp.append(f"{bad} rejected, cut or failed calls or turns: the workload did not fit cleanly")
        mins = [v["minutes"] for v in vals.values()]
        rk, rn = sum(v["rule_pass"] for v in vals.values()), sum(v["rule_n"] for v in vals.values())
        hk, hn = sum(v["hist_ok"] for v in vals.values()), sum(v["canary_turns"] for v in vals.values())
        pre_v = [v["prefill_tok_s"] for v in vals.values() if v["prefill_tok_s"]]
        dec_v = [v["decode_tok_s"] for v in vals.values() if v["decode_tok_s"]]
        cells[c] = {"arm_id": ARM[c], **CELLS[c], "n_sessions": len(vals),
                    "per_seed_min": {sd: v["minutes"] for sd, v in vals.items()},
                    "mean_min": st.mean(mins) if mins else None,
                    "prefill_tok_s": st.mean(pre_v) if pre_v else None,
                    "decode_tok_s": st.mean(dec_v) if dec_v else None,
                    "rule_k": rk, "rule_n": rn, "rule_pass_rate": rk / rn if rn else None,
                    "hist_k": hk, "hist_n": hn, "history_recall": hk / hn if hn else None,
                    "kv_buffer_mib": kv[c],
                    "compute_buffer_mib": sorted({(r.get("kv_log") or {}).get("compute_buffer_mib") for r in cs}
                                                 - {None}),
                    "flash_attn_log": sorted({str((r.get("kv_log") or {}).get("flash_attn")) for r in cs}),
                    "problems": cp}
        problems += [f"{c}: {p}" for p in cp]
    valid = not problems

    def metric(name):
        return {c: [v[name] for v in vals_by[c].values() if v[name]] for c in cells_run}
    e = interaction(metric("minutes"), SQUARE_K, n_boot, seed)
    speed = {"estimate": e, "reading": interaction_reading(e) if valid else
             "not read (invalid: " + "; ".join(problems)[:300] + ")"}
    mem = memory_interaction(kv, SQUARE_K)
    if mem is not None and not valid:
        mem["reading"] = "not read (invalid: " + "; ".join(problems)[:300] + ")"
    secondary = {
        "prefill_tok_s": interaction(metric("prefill_tok_s"), SQUARE_K, n_boot, seed),
        "decode_tok_s": interaction(metric("decode_tok_s"), SQUARE_K, n_boot, seed),
        "operator_square_minutes": {"estimate": interaction(metric("minutes"), SQUARE_OPERATOR, n_boot, seed),
                                    "label": "confounded: K-only q8_0 at FA off vs K+V q8_0 at FA on"},
        "operator_square_memory": memory_interaction(kv, SQUARE_OPERATOR),
    }
    if refusal == "accepted":
        secondary["kv_square_minutes"] = interaction(metric("minutes"), SQUARE_KV, n_boot, seed)
        secondary["kv_square_memory"] = memory_interaction(kv, SQUARE_KV)
    out = {"run_finished": finished, "valid": valid, "problems": problems, "refusal_probe": {
        "cell": REFUSAL_CELL, **CELLS[REFUSAL_CELL], "result": refusal, "lines": (ref or {}).get("lines")},
        "window": N_CTX, "turns": TURNS, "cells": cells, "primary": {"speed": speed, "memory": mem},
        "secondary": secondary}
    if b5_rows is not None:
        out["ollama_b5_prefix"] = b5_prefix_minutes(b5_rows)
    return out


# ── hours (ESTIMATE; register row kv-fa-interaction-llamacpp-hours) ───────────────────────────────────

CELL_TO_B5 = {"f16_fa_off": "a", "f16_fa_on": "i", "q8k_fa_off": "a", "q8k_fa_on": "i", "q8_fa_on": "g"}
HIGH_FACTOR = 1.5


def hours_estimate(b5_rows, b5_fa_rows, xrt_rows, b3_rows) -> dict:
    """Per cell, 3 sessions at the measured Ollama B5 minutes of turns 1..24 of qwen3:14b (cell mapped to the B5 policy
    with the same KV and FA: f16 off -> (a), f16 on -> (i), q8_0 K+V on -> (g); the K-only cells have no Ollama
    counterpart and take their FA setting's f16 policy, ASSUMED no speedup), times the measured llama-server / Ollama
    turn-wall ratio at window 40960 on the B3 v2 sessions (xruntime_llamacpp arm c over x2_evalset_tasks_real_v2
    ollama_default); plus every server start (15 sessions + the refusal probe) at the largest measured stop-to-start
    gap of the xruntime_llamacpp run; high = point x HIGH_FACTOR (ASSUMED)."""
    ref = b5_prefix_minutes(list(b5_rows) + list(b5_fa_rows))
    wl = [r["turn_wall_s"] for r in xrt_rows if r.get("record") == "r2a_turn"
          and r.get("arm_id") == XR.ARM["fixed"]]
    wo = [r["turn_wall_s"] for r in b3_rows if r.get("record") == "r2a_turn" and r.get("model_id") == MODEL_ID
          and r.get("arm_id") == XR.OLLAMA_DEFAULT_ARM]
    ratio = st.mean(wl) / st.mean(wo)
    gaps, last_stop = [], None
    from datetime import datetime
    for r in xrt_rows:
        if r.get("record") == "server_stop":
            last_stop = datetime.fromisoformat(r["ts_utc"])
        elif r.get("record") == "server_start" and r.get("started") and last_stop is not None:
            gaps.append((datetime.fromisoformat(r["ts_utc"]) - last_stop).total_seconds())
            last_stop = None
    start_s = max(gaps)
    per_cell = {c: len(SEEDS) * ref[CELL_TO_B5[c]]["mean_min"] / 60.0 * ratio for c in CELL_ORDER}
    n_starts = len(SEEDS) * len(CELL_ORDER) + 1
    sess_h = sum(per_cell.values())
    point = sess_h + n_starts * start_s / 3600.0
    return {"b5_prefix_mean_min": {p: v["mean_min"] for p, v in ref.items()}, "latency_ratio": ratio,
            "n_ratio_turns": (len(wl), len(wo)), "per_cell_h": per_cell, "session_h": sess_h,
            "start_overhead_s": start_s, "n_starts": n_starts, "point_h": point, "high_h": point * HIGH_FACTOR,
            "high_factor_assumed": HIGH_FACTOR}


# ── refusal probe, sessions, check-plan, main ─────────────────────────────────────────────────────

def refusal_probe(out_dir: Path, emit, log, server_factory=KvServer) -> dict:
    srv = server_factory(REFUSAL_CELL, out_dir / f"kv_fa_refusal_{REFUSAL_CELL}.log")
    try:
        start = srv.start()
    except sg.GuardError as e:
        emit({"record": "server_start", "purpose": "refusal_probe", "cell": REFUSAL_CELL, "started": False,
              "error": str(e)[:400], "guard_record": e.record, "ts_utc": ag.utc_iso()})
        raise
    emit({"record": "server_start", "purpose": "refusal_probe", **start, "ts_utc": ag.utc_iso()})
    emit({"record": "server_stop", "purpose": "refusal_probe", **srv.stop(), "ts_utc": ag.utc_iso()})
    res = start.get("refusal") or refusal_class(bool(start.get("started")), "")
    row = {"record": "refusal_probe", "cell": REFUSAL_CELL, **CELLS[REFUSAL_CELL], "result": res["result"],
           "lines": res["lines"], "exit_rc": start.get("exit_rc"), "kv_log": start.get("kv_log"),
           "ts_utc": ag.utc_iso()}
    emit(row)
    log(f"refusal probe {REFUSAL_CELL}: {res['result']} {res['lines'][:1]}")
    return row


def run_cell_session(cell, seed, out_path, emit, log, server_factory=KvServer, post=None, settle_s=None) -> str:
    """One session of one cell on a fresh server. 'ok' | 'guard_stop' | 'start_failed' | 'refused' | 'kv_mismatch'."""
    srv = server_factory(cell, out_path.parent / f"{out_path.stem}_{cell}_{seed}.log")
    try:
        start = srv.start()
    except sg.GuardError as e:
        emit({"record": "server_start", "cell": cell, "seed": seed, "started": False, "error": str(e)[:400],
              "guard_record": e.record, "ts_utc": ag.utc_iso()})
        raise
    emit({"record": "server_start", "seed": seed, **start, "ts_utc": ag.utc_iso()})
    try:
        if not start.get("started"):
            ref = (start.get("refusal") or {}).get("result")
            return "refused" if ref == "refused" else "start_failed"
        if (start.get("kv_check") or {}).get("status") == "mismatch":
            return "kv_mismatch"
        rt = KvRuntime(srv, emit, post=post, settle_s=settle_s)

        def emit_turn(row):
            if row.get("record") == "r2a_turn":
                ev = rt.pop_turn_events()
                row.update({"llamacpp_calls": ev, "context_event": XR.turn_class(ev), "runtime": "llama-server",
                            "server_pid": srv.pid, "server_n_ctx": srv.n_ctx, "cell": cell})
            emit(row)
        try:
            ag.run_session(rt, MODEL_ID, ARM[cell], seed, TURNS, emit_turn, log, MODE)
        except XR.GuardMismatch as e:
            emit({"record": "guard_stop", "cell": cell, "seed": seed, "error": str(e), "ts_utc": ag.utc_iso()})
            return "guard_stop"
        return "ok"
    finally:
        after = parse_kv_log(srv._log_text()) if hasattr(srv, "_log_text") else None
        emit({"record": "server_stop", "cell": cell, "seed": seed, **srv.stop(),
              "kv_log_after": after, "kv_check_after": kv_check(cell, after) if after else None,
              "ts_utc": ag.utc_iso()})


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
    for c, s in run_order():
        a = ag.build_agent_session(s, TURNS)
        sessions.append({"cell": c, "seed": s, "turns": len(a.spec.turns),
                         "canary_turns": [t.idx for t in a.spec.turns if t.canary_check]})
    g = Path(GGUF)
    return {"harness": HARNESS, "model_id": MODEL_ID, "window": N_CTX, "turns": TURNS, "seeds": SEEDS,
            "cells": CELLS, "cell_order": CELL_ORDER, "refusal_cell": REFUSAL_CELL, "run_order": run_order(),
            "sessions": sessions, "server_cmds": {c: server_cmd(c, "<log>") for c in CELLS},
            "exe": LLAMA_SERVER_EXE, "exe_exists": Path(LLAMA_SERVER_EXE).exists(), "gguf": GGUF,
            "gguf_exists": g.exists(), "gguf_bytes": g.stat().st_size if g.exists() else None,
            "gguf_bytes_expected": GGUF_BYTES, "port": PORT, "port_listeners": sg.port_listeners(PORT),
            "server_help": server_help() if run_help else None, "rules": B5_RULES_QWEN}


def provenance() -> dict:
    p = XR.provenance()
    return {"harness_file": str(Path(__file__).resolve()), "harness_blob": XR.git_blob_sha1(__file__),
            "modules": {**p["modules"], "xruntime_llamacpp": {"file": p["harness_file"], "blob": p["harness_blob"]}}}


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
        digest = XR.sha256_file(GGUF)
        gguf_ok = (size == GGUF_BYTES and digest.startswith(XR.GGUF_SHA256_HEAD)
                   and digest.endswith(XR.GGUF_SHA256_TAIL))
        emit({"record": "run_start", "harness": HARNESS, "mode": MODE, "model_id": MODEL_ID, "window": N_CTX,
              "turns": TURNS, "seeds": SEEDS, "cells": CELLS, "cell_order": CELL_ORDER, "refusal_cell": REFUSAL_CELL,
              "run_order": run_order(), "server_cmds": {c: server_cmd(c, "<log>") for c in CELLS},
              "exe": LLAMA_SERVER_EXE, "gguf": GGUF, "gguf_bytes": size, "gguf_sha256": digest, "gguf_ok": gguf_ok,
              "port": PORT, "workload": "B5 R2 session, call-2 off, first 24 turns", "call2_mode": CALL2,
              "max_tokens_per_call": ag.MAX_TOKENS_PER_CALL, "sampling": {"temperature": 0, "seed": 42},
              "think": ag.MODELS.get(MODEL_ID), "rules_in_use": {MODEL_ID: list(B5_RULES_QWEN)},
              "server_help": server_help(), "ollama_loaded_models": XR.ollama_loaded(), "hw_id": host_cfg.get("hw_id"),
              "interactive_session": interactive, **provenance(), "ts_utc": ag.utc_iso()})
        if not gguf_ok:
            raise RuntimeError(f"GGUF size/hash mismatch: {size} {digest}")
        rows = ag.read_rows(out_path)
        ref = next((r for r in rows if r.get("record") == "refusal_probe"), None)
        if ref is None:
            ref = refusal_probe(out_path.parent, emit, log)
        cells = list(CELL_ORDER) + ([REFUSAL_CELL] if ref["result"] == "accepted" else [])
        done = {(s["arm_id"], s["seed"]) for s in ag.completed_sessions(ag.read_rows(out_path))
                if s.get("mode") == MODE}
        stopped = set()
        for cell, seed in run_order(cells=tuple(cells)):
            if cell in stopped:
                continue
            if (ARM[cell], seed) in done:
                log(f"skip (done): {cell} seed={seed}")
                continue
            log(f"session: {cell} seed={seed}")
            res = run_cell_session(cell, seed, out_path, emit, log)
            if res != "ok":
                log(f"{cell}: {res}; cell stopped")
                emit({"record": "cell_stopped", "cell": cell, "seed": seed, "result": res, "ts_utc": ag.utc_iso()})
                stopped.add(cell)
        rows = ag.read_rows(out_path)
        b5 = out_path.parent / B5_FILE
        bfa = out_path.parent / B5_FA_FILE
        b5_rows = (ag.read_rows(b5) if b5.exists() else []) + (ag.read_rows(bfa) if bfa.exists() else [])
        report = outcome(rows + [{"record": "run_end"}], b5_rows or None)
        Path(str(out_path) + ".report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        emit({"record": "run_end", "speed_reading": report["primary"]["speed"]["reading"],
              "refusal_probe": ref["result"], "ts_utc": ag.utc_iso()})
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
