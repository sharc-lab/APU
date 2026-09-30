"""K2: quality and failures under mid-session memory pressure (evo-t2s and evo-x2).

Starts one llama-server per (mmap arm, pressure arm) combination at n_ctx=16384, runs quality-suite items at
12k-token prompts back to back, then progressively removes available memory (targets +8, +4, +2, +1, 0, -1, -2 GB
relative to the model's footprint), 3 quality items per pressure step, while items keep running. Two mmap arms
(default load mode, explicit --load-mode mmap) x two pressure arms (AWE-locked balloon, ordinary pageable touch),
following the Section C / C1 memory-lock pattern in t2s_overnight.py / t2s_night2.py.

Reuses, does not reimplement: t2s_lab.Server (mmap/load_mode), t2s_lab.Balloon (AWE-locked balloon subprocess),
t2s_overnight.Lab, do_call, start_row (stale-server guard, telemetry window, one JSONL row per call), t2s_overnight's
need_mib/model-footprint arithmetic. The one genuinely new piece of memory-pressure code is pageable_touch.py
(ordinary, reclaimable pageable memory, pressure arm (b)); a repo search for an existing touch/pageable helper (see
that file's docstring) found none.

quality_suite.py (build_task(task_type, target_tokens, seed) -> (prompt, expected), plus a scorer) does not exist in
this checkout yet. This file codes against that documented interface and falls back to a small, self-contained task
(a planted reference code in a filler-padded 12k-token prompt) when the module is absent, so it is not blocked on
that merge; every call site is marked with a TODO to switch over.

Kill criterion (see kill_criterion() below): "nothing changes in quality or availability until a clean failure."
The two numeric tolerances that turns into (5% relative score, 2x responsiveness) are this file's own choice, not
given numerically in the K2 spec -- flagged there for a reviewer to confirm or replace.

Does not deploy, SSH, or run anything on evo-t2s/evo-x2 from here; this module is meant to be deployed and run there
the same way t2s_overnight.py and t2s_night2.py are (see main()'s host_config.require_host() guard, which also
selects gpu_vendor so Telemetry picks Level Zero Sysman on evo-t2s or the LHM feeder on evo-x2).

Usage (deployed to C:\\apu\\ovn on either machine):
  python t2s_k2_pressure.py --expect-blobs expected_blobs.json --deadline-h 6 --models qwen3-8b [--resume <stem>]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import socket
import statistics as st
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import browser_pressure as bap
import host_config as hc
import run_provenance as rp
import t2s_k1_ollama as k1
import t2s_lab as L
import t2s_overnight as ov
import t2s_queue as tq
from t2s_lab import log, ps, utc_iso

try:
    import quality_suite as qs
except ImportError:
    qs = None

# A real quality_suite.TASK_TYPES name once that module is importable (build_quality_task/score_output dispatch on it
# directly); "k2_long_context_fact_recall" only exists for the fallback task below.
TASK_TYPE = qs.TASK_TYPES[0] if qs is not None else "k2_long_context_fact_recall"

DEPLOY = Path(__file__).resolve().parent
PAGEABLE_TOUCH_SCRIPT = str(DEPLOY / "pageable_touch.py")

SEED = 20260928
N_CTX = 16384
TARGET_TOKENS = 12000
N_BASELINE_ITEMS = 3
N_ITEMS_PER_STEP = 3
LEVELS_GB = [8, 4, 2, 1, 0, -1, -2]
MMAP_ARMS = ("default", "mmap")
PRESSURE_ARMS = ("awe_balloon", "pageable_touch")
RESPONSIVENESS_INTERVAL_S = 30.0

# Pressure arm (c): everyday_apps (see docs/FINDINGS.md's "K2 arm (c), everyday_apps" section for the pre-registered
# kill criterion, and harness/browser_pressure.py for the page-generation/browser-lifecycle code this arm drives).
# Structurally different from arms (a)/(b): those sweep LEVELS_GB inside run_k2_run; this one runs a flat sequence
# of session turns and starts/stops the browser at fixed turns within that sequence, so it has its own run function
# (run_k2_everyday_apps_run) rather than going through run_k2_run's LEVELS_GB loop -- it still reuses do_call,
# build_quality_task/score_output and the same lab.row/lab.emit JSONL machinery every other arm uses.
EVERYDAY_APPS_ARM = "everyday_apps"
EVERYDAY_APPS_RUNTIMES = ("llama_server", "ollama")
# 2 models x 2 runtimes = 4 combinations, per the task. ollama_model is Ollama's own tag for the same weights;
# model_id is the ov.MODEL_FILES key used to start the llama-server leg directly against the local GGUF.
EVERYDAY_APPS_MODELS = (
    {"model_id": "llama31-8b", "ollama_model": "llama3.1:8b"},
    {"model_id": "qwen3-4b-2507", "ollama_model": "qwen3:4b-instruct-2507"},
)
EVERYDAY_APPS_START_TURN = bap.DEFAULT_START_TURN   # fixed turn the browser launches at (ASSUMPTION, see browser_pressure.py)
EVERYDAY_APPS_HOLD_TURNS = bap.DEFAULT_HOLD_TURNS   # turns the browser is held open for (given in the task: 10)
EVERYDAY_APPS_TAIL_TURNS = 3                        # turns run after close, to see whether quality/availability recover
# Minimum score drop (absolute, on quality_suite's 0..1 scale) from the pre-pressure baseline that counts as "a
# quality drop" for the pre-registered kill criterion (see everyday_apps_kill_criterion). Not given numerically in
# the task; this module's own choice, flagged the same way SCORE_TOL_REL/RESP_TOL_FACTOR already are above.
EVERYDAY_APPS_SCORE_DROP_ABS = 0.10

# Kill-criterion tolerances. Not given numerically in the K2 spec; reviewer must confirm these two numbers (or
# replace them) before this is trusted as a pass/fail gate on a real run.
SCORE_TOL_REL = 0.05          # median quality score must stay within 5% relative of the +8GB baseline
RESP_TOL_FACTOR = 2.0         # responsiveness median must stay within 2x of the +8GB baseline

REFUSAL_RE = re.compile(r"\b(i (don't|do not) know|cannot determine|not (sure|certain)|unable to (find|determine)"
                         r"|no (reference|such) code|insufficient (information|context))\b", re.I)


# ---------------------------------------------------------------- quality_suite interface (with fallback)
def _fallback_build_task(target_tokens, seed, tokenize_fn):
    """A planted reference code inside filler text, at ~target_tokens total, matching quality_suite's documented
    build_task(task_type, target_tokens, seed) -> (prompt, expected) shape. Reuses context.build_filler the same way
    t2s_lab.build_probe_prompts already does, rather than writing a second filler generator."""
    rng = random.Random(seed)
    code = f"{rng.randint(100000, 999999)}"
    fact = f"The reference code for this run is {code}."
    question = "What is the reference code mentioned above? Answer with only the digits, nothing else."
    fixed_tokens = tokenize_fn(fact) + tokenize_fn(question)
    filler_tokens = max(target_tokens - fixed_tokens - 32, 256)
    filler = L.ctx_mod.build_filler(filler_tokens, seed=seed, count_fn=tokenize_fn)
    prompt = f"{filler}\n\n{fact}\n\n{question}"
    return prompt, code


def _fallback_score(expected, output):
    text = (output or "").strip()
    if expected and expected in text:
        return 1.0, "correct"
    if not text or REFUSAL_RE.search(text):
        return 0.0, "refusal"
    return 0.0, "fabrication"


def build_quality_task(task_type, target_tokens, seed, tokenize_fn):
    """(prompt, expected, scorer_name). quality_suite.build_task returns a Task dataclass (.prompt/.expected/.scorer),
    not a bare tuple, so this unpacks it into the three values run_quality_item's call sites need. scorer_name is
    None in the fallback branch (score_output ignores it there)."""
    if qs is not None and hasattr(qs, "build_task"):
        t = qs.build_task(task_type, target_tokens, seed)
        return t.prompt, t.expected, t.scorer
    prompt, expected = _fallback_build_task(target_tokens, seed, tokenize_fn)
    return prompt, expected, None


def score_output(scorer_name, expected, output):
    """(score in [0, 1] or None, classification string). quality_suite.score_task(scorer_name, output, expected)
    returns (score, detail); detail is a human-readable string, not the {"correct","fabrication","refusal","other"}
    classification this function returns, so classify_fabrication_or_refusal is called separately for the label."""
    if qs is not None and hasattr(qs, "score_task"):
        score, _detail = qs.score_task(scorer_name, output, expected)
        if score is None:
            return None, "refusal"
        label = "correct" if score >= 1.0 else qs.classify_fabrication_or_refusal(output)
        return score, label
    return _fallback_score(expected, output)


# ---------------------------------------------------------------- responsiveness probe
class ResponsivenessSampler:
    """Local interactive-latency probe: launches `python -c "pass"` and times it with time.monotonic(), every
    interval_s, from start() to stop(). Runs in a background thread so it samples continuously through a pressure
    step regardless of how long the quality-suite calls in that step take. This class did not exist yet under
    t2s_night2.py when this file was written (searched, not found); it follows the same design intent that module's
    docstring describes (a local, subprocess-based interactive-latency signal) rather than any code copied from it."""

    def __init__(self, interval_s=RESPONSIVENESS_INTERVAL_S, python=None):
        self.interval_s = interval_s
        self.python = python or sys.executable
        self._stop = threading.Event()
        self._thread = None
        self.samples = []

    def _one_sample(self):
        t0 = time.monotonic()
        ok = False
        try:
            r = subprocess.run([self.python, "-c", "pass"], timeout=10, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok = r.returncode == 0
        except Exception:
            ok = False
        self.samples.append({"t_utc": utc_iso(), "latency_s": time.monotonic() - t0, "ok": ok})

    def start(self):
        self.samples = []
        self._stop.clear()
        self._one_sample()  # one sample immediately at step start, not only after the first interval

        def run():
            while not self._stop.wait(self.interval_s):
                self._one_sample()
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return list(self.samples)

    def median_latency_s(self):
        v = [s["latency_s"] for s in self.samples if s.get("ok")]
        return st.median(v) if v else None


# ---------------------------------------------------------------- pressure arm (b): ordinary pageable memory
class PageableToucher:
    """Ordinary, reclaimable pageable-memory pressure arm: launches pageable_touch.py as a subprocess. Counterpart to
    t2s_lab.Balloon's AWE-locked balloon; same start(target_available_mb)/alive()/stop() shape so phase_k2 can drive
    both pressure arms through one call site. Unlike the AWE balloon, Windows can trim or page this allocation, which
    is the point of the comparison."""

    def __init__(self, lab, script_path, tag):
        self.lab, self.script, self.tag = lab, script_path, tag
        self.proc = None
        self.hb = lab.prefix + f"_touch_{tag}.hb"
        self.log_path = lab.prefix + f"_touch_{tag}.jsonl"
        self.out = lab.prefix + f"_touch_{tag}.out.txt"
        self._stop_hb = threading.Event()

    def start(self, target_available_mb, tolerance_mb=250.0, reach_timeout=120):
        current_avail = L.avail_mb()
        consume_mb = max(current_avail - target_available_mb, 0.0)
        Path(self.hb).write_text(str(time.time()))
        cmd = [L.PYTHON, self.script, "--target-mb", str(consume_mb), "--heartbeat-file", self.hb,
               "--heartbeat-timeout-s", "300", "--log-path", self.log_path, "--sample-interval-s", "5"]
        self.out_f = open(self.out, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.out_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

        def hb():
            while not self._stop_hb.is_set() and self.proc.poll() is None:
                try:
                    Path(self.hb).write_text(str(time.time()))
                except Exception:
                    pass
                self._stop_hb.wait(20)
        threading.Thread(target=hb, daemon=True).start()
        d = time.monotonic() + reach_timeout
        while time.monotonic() < d:
            if self.proc.poll() is not None:
                return {"ok": False, "why": f"toucher exited early rc={self.proc.returncode}",
                        "out": Path(self.out).read_text(errors="replace")[-400:] if Path(self.out).exists() else ""}
            txt = Path(self.out).read_text(errors="replace") if Path(self.out).exists() else ""
            if "TARGET REACHED" in txt:
                # Ordinary pageable allocation is looser than the AWE lock (no control loop, one-shot estimate, the
                # OS can compact in between), so this arm's post-reach tolerance is wider than Balloon's.
                a = L.avail_mb()
                return {"ok": abs(a - target_available_mb) <= tolerance_mb * 4, "available_mb_after": a, "why": None}
            time.sleep(1)
        return {"ok": False, "why": "toucher did not reach target in time"}

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        self._stop_hb.set()
        if self.proc is not None:
            L.m3.kill_tree(self.proc.pid)
            try:
                self.proc.wait(timeout=30)
            except Exception:
                pass
        try:
            self.out_f.close()
        except Exception:
            pass


def _make_pressure(lab, pressure_arm, tag):
    if pressure_arm == "awe_balloon":
        return L.Balloon(lab, ov.BALLOON_SCRIPT, tag[:50])
    if pressure_arm == "pageable_touch":
        return PageableToucher(lab, PAGEABLE_TOUCH_SCRIPT, tag[:50])
    raise ValueError(f"unknown pressure_arm {pressure_arm!r}")


# ---------------------------------------------------------------- model footprint
def _need_mib(lab, mi, n_ctx):
    tab = lab.table.get(mi.model_id) if getattr(lab, "table", None) else None
    if tab and tab.get("ok"):
        return ov.need_mib(tab, n_ctx, mi)
    # No overnight model table loaded: same three terms need_mib uses, with a flat compute-buffer guess.
    return mi.file_bytes / 2 ** 20 + mi.kv_bpt_meta * n_ctx / 2 ** 20 + 800.0


# ---------------------------------------------------------------- one quality-suite item
def run_quality_item(lab, srv, mi, tag, k2_phase, level_gb, pressure_arm, rep):
    """One quality-suite call via t2s_overnight.do_call (stale-server guard, telemetry window, its own JSONL row),
    plus one linked quality_score row carrying the score/classification/crash fields do_call itself does not know
    about. extra below is a plain dict of new keys only (k2_phase, pressure_arm, task_type) -- none of do_call's own
    keyword names (n_ctx, prompt_tokens, mmap, load_mode, co_runner, rep, mem_headroom_gb, load_s, item_id, kind,
    warmup, server_pid, llama_build, or the thermal_gate keys) appear in it, so **extra inside do_call cannot collide
    with do_call's own explicit keywords."""
    prompt, expected, scorer = build_quality_task(TASK_TYPE, TARGET_TOKENS, SEED + rep, srv.tokenize)
    n_tok = srv.tokenize(prompt)
    item_id = f"{tag}_{k2_phase}_{rep}"
    extra = {"k2_phase": k2_phase, "task_type": TASK_TYPE}
    res = ov.do_call(lab, srv, mi, "K2", item_id, prompt, n_tok, warmup=False, rep=rep, extra=extra,
                      max_tokens=256, mem_headroom_gb=level_gb, co_runner="none", kind="quality")

    crash, exit_code, log_tail = False, None, None
    outcome = res.get("outcome") if res is not None else None
    if res is None or outcome != "ok":
        rc = srv.proc.poll() if getattr(srv, "proc", None) is not None else None
        if rc is not None:
            crash, exit_code = True, rc
            lp = L.parse_server_log(srv.log_path)
            log_tail = (lp.get("error_lines") or lp.get("mmap_lines") or [])[-10:]

    score = classification = None
    if outcome == "ok":
        score, classification = score_output(scorer, expected, res.get("output") or "")

    row = lab.row("K2", mi, srv.backend, item_id=item_id, kind="quality_score", k2_phase=k2_phase,
                   pressure_arm=pressure_arm, mem_headroom_gb=level_gb, task_type=TASK_TYPE, rep=rep,
                   score=score, classification=classification, ttft_s=(res or {}).get("ttft_s"),
                   decode_tok_s=(res or {}).get("decode_tok_s"), outcome=outcome, error=(res or {}).get("error"),
                   crash=crash, exit_code=exit_code, server_log_tail=log_tail)
    lab.emit(row)
    return row


# ---------------------------------------------------------------- one pressure step (3 items)
def summarize_step(tag, level_gb, pressure_arm, mmap_arm, step_rows, resp_samples, pressure_info):
    scores = [r["score"] for r in step_rows if r.get("score") is not None]
    resp_vals = [s["latency_s"] for s in resp_samples if s.get("ok")]
    clean_failure = any(r.get("crash") or r.get("outcome") in ("error", "timeout") for r in step_rows)
    exit_code = next((r.get("exit_code") for r in step_rows if r.get("exit_code") is not None), None)
    log_tail = next((r.get("server_log_tail") for r in step_rows if r.get("server_log_tail")), None)
    return {"record": "k2_step_summary", "item_tag": tag, "level_gb": level_gb, "pressure_arm": pressure_arm,
            "mmap_arm": mmap_arm, "n_items": len(step_rows), "median_score": st.median(scores) if scores else None,
            "responsiveness_median_s": st.median(resp_vals) if resp_vals else None,
            "responsiveness_n_samples": len(resp_samples), "clean_failure": bool(clean_failure),
            "exit_code": exit_code, "server_log_tail": log_tail, "pressure_reached": pressure_info.get("ok"),
            "pressure_start_info": pressure_info, "ts_utc": utc_iso()}


def run_k2_run(lab, mi, n_ctx, need_mib, mmap_arm, pressure_arm, responsiveness_interval_s=RESPONSIVENESS_INTERVAL_S):
    """One full run: one server (mmap_arm), 3 baseline items with no pressure applied, then the 7 pressure levels in
    LEVELS_GB order, 3 items per level, using pressure_arm as the pressure source. Returns the ordered list of
    k2_step_summary rows (one per level actually run; a level is skipped -- not summarized -- only if the server
    itself is already gone, see the alive_and_ours break below)."""
    tag = f"k2_{mi.model_id}_{mmap_arm}_{pressure_arm}"
    srv = L.Server(lab, mi, n_ctx, tag=tag, **({"mmap": True} if mmap_arm == "mmap" else {}))
    lab.resources["server"] = srv
    info = srv.start(timeout=1800)
    ov.start_row(lab, srv, mi, "K2", tag + "_start", info, {"pressure_arm": pressure_arm, "mmap_arm": mmap_arm,
                                                             "need_mib": need_mib})
    step_summaries = []
    if not info.get("ok"):
        lab.resources["server"] = None
        return step_summaries

    for rep in range(N_BASELINE_ITEMS):
        run_quality_item(lab, srv, mi, tag, "baseline", None, pressure_arm, rep)

    for level_gb in LEVELS_GB:
        lab.check()
        target_avail_mb = need_mib + level_gb * 1024
        pressure = _make_pressure(lab, pressure_arm, f"{tag}_{level_gb}")
        # Stored under the same "balloon" key ov.cleanup_partial already knows how to stop, regardless of which
        # pressure arm this is: t2s_lab.Balloon and PageableToucher both expose the same start/alive/stop shape.
        lab.resources["balloon"] = pressure
        pinfo = pressure.start(target_avail_mb)
        lab.emit({"record": "k2_pressure_start", "item_tag": tag, "level_gb": level_gb, "pressure_arm": pressure_arm,
                  "target_avail_mb": target_avail_mb, "info": pinfo, "ts_utc": utc_iso()})
        sampler = ResponsivenessSampler(responsiveness_interval_s)
        sampler.start()
        step_rows = []
        try:
            for rep in range(N_ITEMS_PER_STEP):
                step_rows.append(run_quality_item(lab, srv, mi, tag, f"level_{level_gb}", level_gb, pressure_arm, rep))
                ok, _lp = srv.alive_and_ours()
                if not ok:
                    break
        finally:
            resp_samples = sampler.stop()
            pressure.stop()
            lab.resources["balloon"] = None
        summary = summarize_step(tag, level_gb, pressure_arm, mmap_arm, step_rows, resp_samples, pinfo)
        lab.emit(summary)
        step_summaries.append(summary)
        if summary["clean_failure"]:
            break

    kc = kill_criterion(step_summaries)
    lab.emit({"record": "k2_kill_criterion", "item_tag": tag, "mmap_arm": mmap_arm, "pressure_arm": pressure_arm,
              **kc, "ts_utc": utc_iso()})
    log(f"K2 {tag}: kill criterion {'passed' if kc['ok'] else 'VIOLATED'} ({kc['reason']})")

    srv.stop()
    lab.resources["server"] = None
    return step_summaries


def phase_k2(lab, model_id, n_ctx=N_CTX, calibration_pass_set=None):
    """Two mmap arms x two pressure arms, run in full (not interleaved/trimmed -- a real deadline-bound overnight run
    may want to cut this cross down; that scheduling is out of scope here, same as it is out of scope for the
    C1 phase this borrows its memory-lock pattern from).

    calibration_pass_set: same per-task calibration gate as K1's phase_quality_curves (see
    quality_suite.load_calibration_pass_set). K2 only ever exercises one task_type (TASK_TYPE), so there is nothing
    to filter down to -- either it passed calibration for this model or it did not. None means "unknown, run
    unfiltered" (the old behavior, and the default when no --calibration-file is given). When given and TASK_TYPE is
    not in it, K2 does not run at all for this model and the job log states why, rather than silently running a
    whole memory-pressure sweep on a task_type known to be miscalibrated for this model."""
    if calibration_pass_set is not None and TASK_TYPE not in calibration_pass_set:
        log(f"K2: excluding model_id {model_id!r}, task_type {TASK_TYPE!r} did not pass q0_token_calibration for this model")
        lab.emit({"record": "k2_disabled", "model_id": model_id, "task_type": TASK_TYPE,
                  "reason": "task_type did not pass q0_token_calibration", "ts_utc": utc_iso()})
        return []
    mi = lab.models.get(model_id)
    if mi is None:
        lab.emit({"record": "k2_disabled", "model_id": model_id, "reason": "model not loaded", "ts_utc": utc_iso()})
        return []
    need = _need_mib(lab, mi, n_ctx)
    all_summaries = []
    for mmap_arm in MMAP_ARMS:
        for pressure_arm in PRESSURE_ARMS:
            lab.check()
            all_summaries += run_k2_run(lab, mi, n_ctx, need, mmap_arm, pressure_arm)
    return all_summaries


# ---------------------------------------------------------------- pressure arm (c): everyday_apps (Ollama adapter)
class OllamaServerAdapter:
    """Minimal t2s_lab.Server-shaped adapter around t2s_k1_ollama.OllamaClient, so do_call/run_everyday_apps_item
    can drive an Ollama-served model through exactly the same call/telemetry/JSONL-row path (ov.do_call) that
    drives a t2s_lab.Server-served one -- no second call-execution path. Implements only the subset of Server's
    public surface do_call actually touches (n_ctx, mmap, load_mode, backend, pid, start_info, proc, tokenize,
    alive_and_ours, chat) -- the same subset t2s_k2_pressure's own tests' StubServer implements.

    Ollama's /api/chat used here is not the streaming SSE endpoint t2s_lab.Server.chat uses, so ttft_s is not
    observable through it and is always None; decode_tok_s is derived instead from Ollama's own
    eval_count/duration_s, its one true per-call throughput figure. This is default fit/tier: num_ctx is left
    unset so Ollama picks its own default context, matching "Ollama (default fit/tier)" in the task."""

    def __init__(self, ollama, model_tag, n_ctx=None, keep_alive="10m"):
        self.ollama, self.model_tag, self.n_ctx, self.keep_alive = ollama, model_tag, n_ctx, keep_alive
        self.mmap, self.load_mode, self.backend = False, "ollama_default", "ollama"
        self.pid = None
        self.start_info = {"load_s": None, "build": "ollama"}
        self.log_path = None
        self.proc = types.SimpleNamespace(poll=lambda: None)  # Ollama manages its runner process out of band; it
                                                                 # is never observed as "crashed" through this field

    def tokenize(self, text):
        return max(len(text.split()), 1)  # word-count proxy only, for do_call's prompt_tokens field; the real
                                            # tokenizer's count comes back per-call as prompt_eval_count instead

    def alive_and_ours(self):
        return True, [self.pid]

    def chat(self, prompt, max_tokens, ignore_eos):
        res = self.ollama.chat(self.model_tag, prompt, num_ctx=self.n_ctx, max_tokens=max_tokens,
                                keep_alive=self.keep_alive)
        if res.get("outcome") != "ok":
            return {"outcome": "error", "error": res.get("error"), "output": None,
                    "http_status": res.get("status"), "prompt_eval_count": None}
        eval_count = res.get("eval_count") or 0
        duration_s = res.get("duration_s") or 0.0
        decode_tok_s = (eval_count / duration_s) if duration_s > 0 else None
        return {"outcome": "ok", "error": None, "output": res.get("message"), "ttft_s": None,
                "decode_tok_s": decode_tok_s, "e2e_s": duration_s, "completion_tokens": eval_count,
                "usage_reported": eval_count > 0, "think_tag": False, "http_status": res.get("status"),
                "prompt_eval_count": res.get("prompt_eval_count")}

    def stop(self):
        pass


def effective_context_signal(runtime, srv):
    """The effective-context signal for one everyday_apps turn, reusing K1's own signal rather than reimplementing
    it: for Ollama, GET /api/ps (phase_tier's cheapest and primary signal (a); (b) server.log and (c) the empirical
    probe are not re-run per turn here -- out of scope for a per-turn diagnostic, flagged for a reviewer who wants
    them). For llama-server, the context is not ambiguous the way Ollama's auto-tiering is: this arm starts the
    server with a known --ctx-size, so the effective context is simply srv.n_ctx, source 'configured'."""
    if runtime == "ollama":
        ps_res = srv.ollama.get_ps()
        match = next((m for m in ps_res.get("models", [])
                      if m.get("name") == srv.model_tag or (m.get("name") or "").startswith(srv.model_tag.split(":")[0])),
                     None)
        return {"effective_ctx": (match or {}).get("context_length"), "source": "api_ps", "raw": match}
    return {"effective_ctx": srv.n_ctx, "source": "configured", "raw": None}


def run_everyday_apps_item(lab, srv, mi, tag, runtime, turn_idx, browser_active, rep):
    """One everyday_apps-arm turn: same call/scoring path as run_quality_item (build_quality_task, ov.do_call,
    score_output -- do_call's own emit already writes the 'quality' call row with ttft_s/decode_tok_s/telemetry).
    This function additionally emits an 'everyday_apps_turn' row carrying the extra signals this arm's own
    diagnostics call for that run_quality_item's shared row shape does not: effective context, raw HTTP
    status/error text, and prompt_eval_count against tokens actually sent."""
    prompt, expected, scorer = build_quality_task(TASK_TYPE, TARGET_TOKENS, SEED + rep, srv.tokenize)
    n_tok = srv.tokenize(prompt)
    item_id = f"{tag}_turn{turn_idx}"
    extra = {"k2_phase": f"turn_{turn_idx}", "task_type": TASK_TYPE, "everyday_apps_turn": turn_idx,
             "everyday_apps_browser_active": browser_active}
    res = ov.do_call(lab, srv, mi, "K2", item_id, prompt, n_tok, warmup=False, rep=rep, extra=extra,
                      max_tokens=256, mem_headroom_gb=None, co_runner="none", kind="quality")
    outcome = res.get("outcome") if res is not None else None
    score = classification = None
    if outcome == "ok":
        score, classification = score_output(scorer, expected, res.get("output") or "")
    ctx_signal = effective_context_signal(runtime, srv)
    row = lab.row("K2", mi, srv.backend, item_id=item_id, kind="everyday_apps_turn", pressure_arm=EVERYDAY_APPS_ARM,
                   runtime=runtime, turn_idx=turn_idx, browser_active=browser_active, task_type=TASK_TYPE, rep=rep,
                   score=score, classification=classification, outcome=outcome,
                   http_status=(res or {}).get("http_status"), error=(res or {}).get("error"),
                   prompt_tokens_sent=n_tok, prompt_eval_count=(res or {}).get("prompt_eval_count"),
                   ttft_s=(res or {}).get("ttft_s"), decode_tok_s=(res or {}).get("decode_tok_s"),
                   effective_context=ctx_signal.get("effective_ctx"), effective_context_source=ctx_signal.get("source"))
    lab.emit(row)
    return row


def run_k2_everyday_apps_run(lab, mi, model_id, runtime, ollama_model=None, n_ctx=N_CTX,
                             start_turn=EVERYDAY_APPS_START_TURN, hold_turns=EVERYDAY_APPS_HOLD_TURNS,
                             tail_turns=EVERYDAY_APPS_TAIL_TURNS, everyday_apps_cls=bap.EverydayAppsPressure,
                             ollama_client_cls=None):
    """One full everyday_apps run for one (model, runtime) combination: start_turn + hold_turns + tail_turns turns
    total, the browser launching at turn start_turn and closing after hold_turns turns of holding it open. Reuses
    do_call (via run_everyday_apps_item), ov.start_row and lab.emit/lab.row exactly as run_k2_run does; the only
    new lifecycle here is the browser's own start/stop, which is the EverydayAppsPressure class's job."""
    tag = f"k2_everyday_apps_{model_id}_{runtime}"
    if runtime == "llama_server":
        srv = L.Server(lab, mi, n_ctx, tag=tag)  # default fit: no mmap/load_mode override
        lab.resources["server"] = srv
        info = srv.start(timeout=1800)
    elif runtime == "ollama":
        ollama = (ollama_client_cls or k1.OllamaClient)()
        srv = OllamaServerAdapter(ollama, ollama_model)
        now = time.time()
        info = {"ok": True, "pid": None, "load_s": None, "error": None, "build": "ollama", "t_start": now,
                "t_end": now, "log": {}}
        lab.resources["server"] = srv
    else:
        raise ValueError(f"unknown everyday_apps runtime {runtime!r}")

    ov.start_row(lab, srv, mi, "K2", tag + "_start", info, {"pressure_arm": EVERYDAY_APPS_ARM, "runtime": runtime})
    rows = []
    if not info.get("ok"):
        lab.resources["server"] = None
        return rows

    pressure = everyday_apps_cls(lab, tag)
    lab.resources["balloon"] = pressure  # same shared cleanup key ov.cleanup_partial already knows how to stop
    browser_active = False
    n_turns = start_turn + hold_turns + tail_turns
    try:
        for turn_idx in range(n_turns):
            lab.check()
            if turn_idx == start_turn:
                pinfo = pressure.start(None)
                browser_active = bool(pinfo.get("ok"))
                lab.emit({"record": "k2_everyday_apps_browser_start", "item_tag": tag, "turn_idx": turn_idx,
                          "info": pinfo, "ts_utc": utc_iso()})
            row = run_everyday_apps_item(lab, srv, mi, tag, runtime, turn_idx, browser_active, turn_idx)
            rows.append(row)
            if browser_active and turn_idx == start_turn + hold_turns - 1:
                pressure.stop()
                browser_active = False
                lab.emit({"record": "k2_everyday_apps_browser_stop", "item_tag": tag, "turn_idx": turn_idx,
                          "ts_utc": utc_iso()})
            ok, _lp = srv.alive_and_ours()
            if not ok:
                break
    finally:
        if pressure.alive():  # safety net: never leave the browser running if the loop above broke out early
            pressure.stop()
        lab.resources["balloon"] = None

    if runtime == "llama_server":
        srv.stop()
    lab.resources["server"] = None
    return rows


def everyday_apps_kill_criterion(rows, score_drop_abs=EVERYDAY_APPS_SCORE_DROP_ABS):
    """The pre-registered check from docs/FINDINGS.md's "K2 arm (c), everyday_apps" section (2026-09-29), written
    before this run-wiring code existed: the "degrades silently under everyday-app memory pressure" claim requires
    at least one turn whose score drops by more than score_drop_abs from the pre-pressure baseline (median score of
    turns before the browser started) while that same turn surfaced no error (outcome == 'ok' and error is falsy).
    If every drop of that size coincides with a surfaced error, or there is no such drop at all, the claim is not
    supported by this run.

    rows: ordered list of everyday_apps_turn row dicts (see run_everyday_apps_item), one per turn, turn_idx order.
    Returns {"claim_supported", "reason", "baseline_score", "silent_drop_turns"}."""
    baseline_rows = [r for r in rows if not r.get("browser_active") and r.get("score") is not None]
    if not baseline_rows:
        return {"claim_supported": False, "reason": "no pre-pressure baseline turn with a usable score",
                "baseline_score": None, "silent_drop_turns": []}
    baseline_score = st.median(r["score"] for r in baseline_rows)
    silent_drops = []
    for r in rows:
        if r.get("score") is None:
            continue
        if (baseline_score - r["score"]) > score_drop_abs:
            surfaced_error = r.get("outcome") != "ok" or bool(r.get("error"))
            if not surfaced_error:
                silent_drops.append(r.get("turn_idx"))
    claim_supported = len(silent_drops) > 0
    reason = (f"silent quality drop(s) found at turn(s) {silent_drops}" if claim_supported else
              "every quality drop observed coincided with a surfaced error (or no drop occurred)")
    return {"claim_supported": claim_supported, "reason": reason, "baseline_score": baseline_score,
            "silent_drop_turns": silent_drops}


def phase_k2_everyday_apps(lab, calibration_pass_set=None, models=EVERYDAY_APPS_MODELS, runtimes=EVERYDAY_APPS_RUNTIMES,
                           n_ctx=N_CTX):
    """Pressure arm (c) across every (model, runtime) combination -- 4 by default (2 models x 2 runtimes). Same
    per-task calibration gate as phase_k2: this arm only ever exercises TASK_TYPE, so either it passed calibration
    for a given model or the whole arm is skipped for that model, same reasoning as phase_k2's own gate."""
    if calibration_pass_set is not None and TASK_TYPE not in calibration_pass_set:
        log(f"K2 everyday_apps: excluding task_type {TASK_TYPE!r}, did not pass q0_token_calibration")
        lab.emit({"record": "k2_disabled", "pressure_arm": EVERYDAY_APPS_ARM, "task_type": TASK_TYPE,
                  "reason": "task_type did not pass q0_token_calibration", "ts_utc": utc_iso()})
        return []
    all_rows = []
    for model in models:
        model_id = model["model_id"]
        mi = lab.models.get(model_id)
        if mi is None:
            lab.emit({"record": "k2_disabled", "pressure_arm": EVERYDAY_APPS_ARM, "model_id": model_id,
                      "reason": "model not loaded", "ts_utc": utc_iso()})
            continue
        for runtime in runtimes:
            lab.check()
            rows = run_k2_everyday_apps_run(lab, mi, model_id, runtime, ollama_model=model.get("ollama_model"),
                                            n_ctx=n_ctx)
            kc = everyday_apps_kill_criterion(rows)
            lab.emit({"record": "k2_everyday_apps_kill_criterion", "model_id": model_id, "runtime": runtime,
                      **kc, "ts_utc": utc_iso()})
            log(f"K2 everyday_apps {model_id}/{runtime}: claim {'SUPPORTED' if kc['claim_supported'] else 'not supported'} ({kc['reason']})")
            all_rows += rows
    return all_rows


def everyday_apps_dry_run_call_counts(models=EVERYDAY_APPS_MODELS, runtimes=EVERYDAY_APPS_RUNTIMES,
                                      start_turn=EVERYDAY_APPS_START_TURN, hold_turns=EVERYDAY_APPS_HOLD_TURNS,
                                      tail_turns=EVERYDAY_APPS_TAIL_TURNS):
    """No real run, no I/O: the exact model-call count this arm would add per (model, runtime) combination, and in
    total, given the turn-scheduling constants above. One call per turn (n_turns = start_turn+hold_turns+tail_turns)."""
    n_turns = start_turn + hold_turns + tail_turns
    per_combo = {f"{m['model_id']}/{r}": n_turns for m in models for r in runtimes}
    return {"n_turns_per_combo": n_turns, "n_combinations": len(models) * len(runtimes), "per_combo": per_combo,
            "total_calls": n_turns * len(models) * len(runtimes)}


# ---------------------------------------------------------------- kill criterion
def kill_criterion(rows, score_tol_rel=SCORE_TOL_REL, resp_tol_factor=RESP_TOL_FACTOR):
    """"nothing changes in quality or availability until a clean failure": every step's median quality score and
    responsiveness must stay within tolerance of the +8GB baseline step until the first step with a clean failure
    (non-zero exit code, a chat call outcome of "error"/"timeout", or record()'s own crash flag); every step strictly
    before that first failure must already have passed, or this check is violated.

    Tolerance (reviewer must confirm -- not given numerically in the K2 spec):
      score_tol_rel   default 0.05  : median_score within 5% relative of the +8GB baseline score
      resp_tol_factor default 2.0   : responsiveness_median_s at most 2x the +8GB baseline (only a slowdown counts;
                                      responding faster than baseline is never a violation)

    rows: ordered list of k2_step_summary dicts (see summarize_step), one per level, in the order the levels were
    actually run (LEVELS_GB order: +8, +4, +2, +1, 0, -1, -2). Returns {"ok", "reason", "baseline",
    "first_failure_level_gb"}.
    """
    baseline = next((r for r in rows if r.get("level_gb") == 8), None)
    if baseline is None:
        return {"ok": False, "reason": "no +8GB baseline step found in rows", "baseline": None,
                "first_failure_level_gb": None}
    base_score, base_resp = baseline.get("median_score"), baseline.get("responsiveness_median_s")
    first_failure_gb = next((r.get("level_gb") for r in rows if r.get("clean_failure")), None)
    for r in rows:
        lv = r.get("level_gb")
        if first_failure_gb is not None and lv == first_failure_gb:
            break  # the failing step itself, and anything after it, is outside this check's scope
        score, resp = r.get("median_score"), r.get("responsiveness_median_s")
        if base_score is not None and score is not None:
            if abs(score - base_score) > score_tol_rel * max(abs(base_score), 1e-9):
                return {"ok": False, "baseline": {"score": base_score, "responsiveness_s": base_resp},
                        "first_failure_level_gb": first_failure_gb,
                        "reason": f"level {lv}GB: median score {score} deviates from the +8GB baseline "
                                  f"{base_score} by more than {score_tol_rel * 100:.0f}% before any clean failure"}
        if base_resp is not None and resp is not None:
            if resp > base_resp * resp_tol_factor:
                return {"ok": False, "baseline": {"score": base_score, "responsiveness_s": base_resp},
                        "first_failure_level_gb": first_failure_gb,
                        "reason": f"level {lv}GB: responsiveness median {resp}s exceeds {resp_tol_factor}x the "
                                  f"+8GB baseline ({base_resp}s) before any clean failure"}
    return {"ok": True, "reason": None, "baseline": {"score": base_score, "responsiveness_s": base_resp},
            "first_failure_level_gb": first_failure_gb}


# ---------------------------------------------------------------- CLI
def load_models(lab, wanted):
    """Same loader t2s_night2.load_models uses: skips (logs, does not raise) a wanted model whose hash isn't in
    downloads.jsonl yet, e.g. the 70B on evo-x2 while its download/verify is still in progress."""
    ov.read_downloads(lab)
    for mid in wanted:
        fn, hyb, mx, yf = ov.MODEL_FILES[mid]
        sha = ov.KNOWN_4B_SHA if mid == "qwen3-4b-2507" else lab.dl_sha.get(fn)
        if sha is None:
            log(f"load_models: {mid} ({fn}) not yet in downloads.jsonl, skipping for this run")
            continue
        lab.models[mid] = L.ModelInfo(mid, str(Path(L.MODELS_DIR) / fn), sha, hyb, mx, yf)


def make_lab(args, prov, gpu_vendor=None):
    ns = types.SimpleNamespace(smoke=False, deadline_h=args.deadline_h, resume=args.resume,
                                stem_prefix="t2s_k2_pressure", only=None, reserve_min=20, no_cap_arm=True,
                                max_items=0, gpu_vendor=gpu_vendor)
    return ov.Lab(ns, prov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-blobs", required=True)
    ap.add_argument("--deadline-h", type=float, default=6.0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--models", default="qwen3-8b", help="comma list of model ids to run K2 on")
    ap.add_argument("--n-ctx", type=int, default=N_CTX)
    ap.add_argument("--calibration-file", default=None, help="path to a q0_token_calibration.py run's own jsonl; "
                    "K2 runs only if its single task_type (TASK_TYPE) passed calibration in it, per-task (not "
                    "whole-run) -- see quality_suite.load_calibration_pass_set. Omit to run unfiltered.")
    ap.add_argument("--everyday-apps", action="store_true", help="also run pressure arm (c), everyday_apps, across "
                    "EVERYDAY_APPS_MODELS x EVERYDAY_APPS_RUNTIMES. Off by default: unlike arms (a)/(b), this arm's "
                    "Ollama-runtime leg deliberately starts Ollama, which the main per-model loop's contamination "
                    "guard below otherwise refuses to run alongside.")
    args = ap.parse_args()
    host_cfg = hc.require_host(socket.gethostname())
    hc.enforce_or_record_interactive_session(host_cfg)  # raises on evo-t2s if occupied; never raises on evo-x2
    prov = rp.verify_deployed_blobs(DEPLOY, args.expect_blobs)
    lab = make_lab(args, prov, gpu_vendor=host_cfg["gpu_vendor"])
    lab.identity["hw_id"] = host_cfg["hw_id"]
    lab.track_console = not host_cfg.get("interactive_guard", True)
    model_ids = args.models.split(",")
    load_models(lab, model_ids)
    if args.everyday_apps:
        load_models(lab, [m["model_id"] for m in EVERYDAY_APPS_MODELS if m["model_id"] not in model_ids])
    Path(lab.prefix + "_manifest.json").write_text(json.dumps({
        "launch_utc": utc_iso(), "script_provenance": prov, "identity": lab.identity, "models": model_ids,
        "n_ctx": args.n_ctx, "seed": SEED, "levels_gb": LEVELS_GB, "mmap_arms": MMAP_ARMS,
        "pressure_arms": PRESSURE_ARMS, "score_tol_rel": SCORE_TOL_REL, "resp_tol_factor": RESP_TOL_FACTOR,
        "everyday_apps": args.everyday_apps, "everyday_apps_models": [m["model_id"] for m in EVERYDAY_APPS_MODELS],
        "everyday_apps_runtimes": EVERYDAY_APPS_RUNTIMES},
        indent=1, default=str), encoding="utf-8")
    calibration_pass_set = None
    if args.calibration_file:
        if qs is None or not hasattr(qs, "load_calibration_pass_set"):
            log(f"--calibration-file {args.calibration_file!r} given but quality_suite is unavailable; running unfiltered")
        else:
            calibration_pass_set = qs.load_calibration_pass_set(args.calibration_file)
            if calibration_pass_set is None:
                log(f"--calibration-file {args.calibration_file!r} had no usable q0_calibration_summary; running unfiltered")
            else:
                log(f"K2: calibration pass set from {args.calibration_file!r}: {sorted(calibration_pass_set)}")
    lab.tele.start()
    note = "completed"
    try:
        time.sleep(15)
        pk = [lab.tele.pkg_now() for _ in range(6) if not time.sleep(1)]
        lab.idle_pkg = st.median([x for x in pk if x is not None]) if any(x is not None for x in pk) else None
        lab.idle_temp = None
        for mid in model_ids:
            if f"k2_done_{mid}" in lab.done:
                continue
            # K2 never uses Ollama itself; only K1 does (starting/stopping its own server). Per the 2026-09-29
            # contamination check (docs/RESULT_PROVENANCE.md), Ollama must not idle in the background during any
            # other phase, so abort rather than risk a model load racing against this measurement.
            if hc.ollama_process_running():
                raise RuntimeError(f"STOP: an ollama process is running; refusing to start K2 phase for {mid}")
            log(f"K2 phase: {mid}")
            phase_k2(lab, mid, args.n_ctx, calibration_pass_set=calibration_pass_set)
            lab.item_done(f"k2_done_{mid}")
        if args.everyday_apps and "k2_everyday_apps_done" not in lab.done:
            log("K2 phase: everyday_apps")
            phase_k2_everyday_apps(lab, calibration_pass_set=calibration_pass_set, n_ctx=args.n_ctx)
            lab.item_done("k2_everyday_apps_done")
    except ov.Deadline:
        note = "deadline reached"
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
        log(note)
    finally:
        ov.cleanup(lab)
        lab.emit({"record": "run_end", "note": note, "ts_utc": utc_iso()})
        (Path(lab.out_dir) / f"{lab.stem}.DONE").write_text(note + "\n")
        try:
            tq.advance(note)
        except Exception as e:
            log(f"queue advance failed: {e!r}")


if __name__ == "__main__":
    main()
