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

import host_config as hc
import run_provenance as rp
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


def phase_k2(lab, model_id, n_ctx=N_CTX):
    """Two mmap arms x two pressure arms, run in full (not interleaved/trimmed -- a real deadline-bound overnight run
    may want to cut this cross down; that scheduling is out of scope here, same as it is out of scope for the
    C1 phase this borrows its memory-lock pattern from)."""
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
    args = ap.parse_args()
    host_cfg = hc.require_host(socket.gethostname())
    hc.enforce_or_record_interactive_session(host_cfg)  # raises on evo-t2s if occupied; never raises on evo-x2
    prov = rp.verify_deployed_blobs(DEPLOY, args.expect_blobs)
    lab = make_lab(args, prov, gpu_vendor=host_cfg["gpu_vendor"])
    lab.identity["hw_id"] = host_cfg["hw_id"]
    lab.track_console = not host_cfg.get("interactive_guard", True)
    model_ids = args.models.split(",")
    load_models(lab, model_ids)
    Path(lab.prefix + "_manifest.json").write_text(json.dumps({
        "launch_utc": utc_iso(), "script_provenance": prov, "identity": lab.identity, "models": model_ids,
        "n_ctx": args.n_ctx, "seed": SEED, "levels_gb": LEVELS_GB, "mmap_arms": MMAP_ARMS,
        "pressure_arms": PRESSURE_ARMS, "score_tol_rel": SCORE_TOL_REL, "resp_tol_factor": RESP_TOL_FACTOR},
        indent=1, default=str), encoding="utf-8")
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
            phase_k2(lab, mid, args.n_ctx)
            lab.item_done(f"k2_done_{mid}")
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
