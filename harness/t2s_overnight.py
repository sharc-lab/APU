"""evo-t2s overnight run: model ladder, memory-budget crossing, power-coupling causal tests, memory lock with mmap arm.

Sections (priority order): 0 smoke and positive controls, A budget crossing, B power coupling (+ causal cap arm),
C memory lock with mmap on/off, D same hardware different runtime (SYCL). Work is a list of resumable items; a
planner estimates each item from the Section 0 timings and trims in reverse priority to fit the deadline, logging
exactly what was trimmed. Every call writes one JSONL row (fsync) carrying the join-key fields.

Usage on evo-t2s (deployed to C:\\apu\\ovn):
  python t2s_overnight.py --expect-blobs expected_blobs.json --deadline-h 11 [--smoke] [--resume <stem>] [--only 0|A|B|C|D]
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import statistics as st
import subprocess
import sys
import time
import zlib
from pathlib import Path

import t2s_lab as L
import run_provenance as rp
from t2s_lab import log, ps, utc_iso

DEPLOY = Path(__file__).resolve().parent
OUT_ROOT = Path(r"C:\apu\ovn")
BALLOON_SCRIPT = str(DEPLOY / "memory_balloon_awe.py")
SEED = 20260925
MODEL_FILES = {
    "qwen3-4b-2507": ("qwen3-4b-instruct-85e4a5b7.gguf", False, 262144, 1),
    "qwen3-8b": ("Qwen3-8B-Q4_K_M.gguf", True, 32768, 4),
    "llama31-8b": ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", False, 131072, 1),
    "qwen3-14b": ("Qwen3-14B-Q4_K_M.gguf", True, 32768, 4),
    "qwen3-30b-a3b-2507": ("Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", False, 262144, 1),
    "qwen3-32b": ("Qwen3-32B-Q4_K_M.gguf", True, 32768, 4),
}
PAGING_FILE = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
YARN_MODELS = ("qwen3-32b", "qwen3-14b", "qwen3-8b")
KNOWN_4B_SHA = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
MASKS = {"none": None, "p4": 0x000F, "e4": 0x00F0, "nonp12": 0xFFF0, "all16": 0xFFFF}
ROW_KEYS = ["hw_id", "cpu_name", "gpu_name", "gpu_driver_version", "backend", "llama_build", "model_id", "model_sha256",
            "quant", "kv_type", "flash_attn", "mmap", "n_ctx", "prompt_tokens", "co_runner", "proc_throttle_max",
            "mem_headroom_gb", "section", "rep", "seed", "git_sha", "script_sha", "load_s", "ttft_s", "decode_tok_s",
            "e2e_s", "output", "score", "igpu_mhz", "pkg_power_w", "shared_usage_mib", "total_committed_mib",
            "pages_input_per_s", "hard_faults_per_s", "error"]


class Deadline(Exception):
    pass


def crc(s):
    return zlib.crc32(s.encode()) % 100000


class Lab:
    def __init__(self, args, prov):
        self.args, self.prov = args, prov
        self.smoke = args.smoke
        self.deadline_ts = time.time() + args.deadline_h * 3600
        self.out_dir = OUT_ROOT / ("smoke" if args.smoke else "results")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stem = args.resume or f"t2s_overnight_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        self.prefix = str(self.out_dir / self.stem)
        self.rows = L.Jsonl(self.prefix + ".jsonl")
        self.tele = L.Telemetry(self.prefix)
        self.git_sha = prov["git_head"]
        self.script_sha = next((f["actual_git_blob_sha"] for f in prov["files"] if f["path"] == "t2s_overnight.py"), None)
        self.idle_temp = None
        self.idle_pkg = None
        self.models, self.table, self.done = {}, {}, set()
        self.dl_sha, self.dl_dropped = {}, set()
        self.scorers, self.probes = L.load_probes()
        self.resources = {"server": None, "balloon": None, "hog": None, "powercap": None}
        self.paging_ok = None
        self.knob_ok = None
        self.base_pkg = {}
        self.a_started = {}
        self.a_last_below = {}
        self.sycl_ok = None
        self.identity = {
            "hw_id": "evo-t2s",
            "cpu_name": ps("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"),
            "gpu_name": ps("(Get-CimInstance Win32_VideoController | Select-Object -First 1).Name"),
            "gpu_driver_version": ps("(Get-CimInstance Win32_VideoController | Select-Object -First 1).DriverVersion"),
        }
        if args.resume:
            for l in open(self.prefix + ".jsonl", encoding="utf-8"):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                if r.get("record") == "item_done":
                    self.done.add(r["item_id"])
                if r.get("record") == "section0_model":
                    self.table[r["model_id"]] = r

    def left(self):
        return self.deadline_ts - time.time()

    def check(self):
        if self.left() <= 0:
            raise Deadline()

    def cap(self):
        pc = self.resources.get("powercap")
        return pc.current if pc is not None else None

    def row(self, section, mi=None, backend="vulkan", **kw):
        r = {k: None for k in ROW_KEYS}
        r.update(self.identity)
        r.update({"backend": backend, "section": section, "seed": SEED, "git_sha": self.git_sha,
                  "script_sha": self.script_sha, "kv_type": "f16", "flash_attn": "on", "ts_utc": utc_iso(),
                  "proc_throttle_max": self.cap()})
        if mi is not None:
            r.update({"model_id": mi.model_id, "model_sha256": mi.sha256, "quant": mi.quant})
        r.update(kw)
        return r

    def emit(self, row):
        self.rows.write(row)

    def item_done(self, item_id):
        self.done.add(item_id)
        self.rows.write({"record": "item_done", "item_id": item_id, "ts_utc": utc_iso()})


# ---------------------------------------------------------------- shared call logic
def prompt_for(srv, fill_tokens):
    return L.ctx_mod.build_filler(fill_tokens, seed=42, count_fn=srv.tokenize)


def do_call(lab, srv, mi, section, item_id, prompt, n_tok, *, warmup, rep, extra, max_tokens=128, ignore_eos=True,
            mem_headroom_gb=None, co_runner="none", kind="call", probe=None):
    lab.check()
    gate = lab.tele.thermal_gate(lab.idle_temp, idle_pkg=lab.idle_pkg)
    ok, lp = srv.alive_and_ours()
    base = dict(n_ctx=srv.n_ctx, prompt_tokens=n_tok, mmap=srv.mmap, co_runner=co_runner, rep=rep,
                mem_headroom_gb=mem_headroom_gb, load_s=srv.start_info.get("load_s"), item_id=item_id, kind=kind,
                warmup=warmup, server_pid=srv.pid, llama_build=srv.start_info.get("build"), **gate, **extra)
    if not ok:
        r = lab.row(section, mi, srv.backend, **base)
        r.update({"valid": False, "invalid_reason": f"listener {lp} is not our server pid {srv.pid} or the server died",
                  "error": "stale-server guard: listener mismatch"})
        lab.emit(r)
        return None
    t0 = time.time()
    res = srv.chat(prompt, max_tokens, ignore_eos)
    t1 = time.time()
    time.sleep(1.5)
    m = lab.tele.metrics(t0, t1 + 1.0)
    r = lab.row(section, mi, srv.backend, **base)
    r.update({"t_start_utc": t0, "t_end_utc": t1, "ttft_s": res.get("ttft_s"), "decode_tok_s": res.get("decode_tok_s"),
              "e2e_s": res.get("e2e_s"), "output": (res.get("output") or "")[:200] if probe else None,
              "outcome": res.get("outcome"), "error": res.get("error"), "completion_tokens": res.get("completion_tokens"),
              "think_tag": res.get("think_tag"), "valid": res.get("outcome") == "ok",
              "igpu_mhz": m["igpu_mhz"], "igpu_throttle_bits": m["igpu_throttle_bits"], "pkg_power_w": m["pkg_power_w"],
              "shared_usage_mib": m["shared_usage_mib"], "total_committed_mib": m["total_committed_mib"],
              "pages_input_per_s": m["pages_input_per_s"], "hard_faults_per_s": m["hard_faults_per_s"],
              "avail_mb_min": m["avail_mb_min"], "avail_mb_max": m["avail_mb_max"], "temp_c_max": m["temp_c_max"]})
    if probe is not None:
        try:
            score, det = lab.scorers.score({"id": probe["id"], "scorer_type": probe["scorer_type"],
                                            "expected": probe["expected"]}, res.get("output") or "")
        except Exception as e:
            score, det = None, f"scorer_error:{e}"
        r.update({"score": score, "score_detail": det, "probe_id": probe["id"]})
    lab.emit(r)
    res.update({"pkg_power_w": m["pkg_power_w"], "igpu_mhz": m["igpu_mhz"]})
    if probe is not None:
        res["score"] = r.get("score")
    return res


def measured_sequence(lab, srv, mi, section, item_id, prompt, n_tok, *, extra, mem_headroom_gb=None, co_runner="none",
                      slow_s=90.0, max_tokens=128):
    """1 discarded warm-up then 5 measured calls (3 when a call exceeds slow_s, flagged n_reduced)."""
    w = do_call(lab, srv, mi, section, item_id, prompt, n_tok, warmup=True, rep=-1, extra=extra, max_tokens=max_tokens,
                mem_headroom_gb=mem_headroom_gb, co_runner=co_runner)
    if w is None or w.get("outcome") != "ok":
        return []
    slow = (w.get("e2e_s") or 0) > slow_s
    out = []
    for i in range(1 if lab.smoke else (3 if slow else 5)):
        r = do_call(lab, srv, mi, section, item_id, prompt, n_tok, warmup=False, rep=i, extra=dict(extra, n_reduced=slow),
                    max_tokens=max_tokens, mem_headroom_gb=mem_headroom_gb, co_runner=co_runner)
        if r is None or r.get("outcome") != "ok":
            break
        out.append(r)
    return out


def probe_sequence(lab, srv, mi, section, item_id, fill_tokens, *, extra, mem_headroom_gb=None, co_runner="none"):
    res = []
    for p, prompt, n_tok in L.build_probe_prompts(srv, fill_tokens, lab.probes):
        r = do_call(lab, srv, mi, section, item_id, prompt, n_tok, warmup=False, rep=0, extra=extra, max_tokens=32,
                    ignore_eos=False, mem_headroom_gb=mem_headroom_gb, co_runner=co_runner, kind="probe", probe=p)
        res.append(r)
        if r is None or r.get("outcome") != "ok":
            break
    return res


def start_row(lab, srv, mi, section, item_id, info, extra):
    r = lab.row(section, mi, srv.backend, n_ctx=srv.n_ctx, mmap=srv.mmap, load_s=info.get("load_s"), item_id=item_id,
                kind="start", error=info.get("error"), server_pid=info.get("pid"), llama_build=info.get("build"),
                valid=info.get("ok"), t_start_utc=info["t_start"], t_end_utc=info["t_end"], **extra)
    time.sleep(1.5)
    m = lab.tele.metrics(info["t_start"], info["t_end"] + 1.0)
    r.update({k: m[k] for k in ("igpu_mhz", "pkg_power_w", "shared_usage_mib", "total_committed_mib", "pages_input_per_s",
                                "hard_faults_per_s", "avail_mb_min", "avail_mb_max")})
    lg = info.get("log", {})
    r.update({"kv_buffer_mib_log": lg.get("kv_buffer_mib"), "compute_buffer_mib_log": lg.get("compute_buffer_mib"),
              "model_buffer_mib_log": lg.get("model_buffer_mib"), "device_line": lg.get("device_line"),
              "device_free_mib": lg.get("device_free_mib"), "mmap_lines": lg.get("mmap_lines"),
              "exit_code": info.get("exit_code"), "private_mib_at_load": info.get("private_mib"),
              "working_set_mib_at_load": info.get("working_set_mib")})
    lab.emit(r)


# ---------------------------------------------------------------- Section 0
def read_downloads(lab):
    p = OUT_ROOT / "downloads.jsonl"
    if not p.exists():
        return
    for l in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("event") == "done":
            lab.dl_sha[r["model"]] = r["sha256"]
        if r.get("event") in ("dropped", "sha_mismatch"):
            lab.dl_dropped.add(r["model"])


def paging_control(lab):
    """Evict the file cache with the balloon, then read 2 GB of a model file: pages input must be nonzero."""
    target = str(Path(L.MODELS_DIR) / PAGING_FILE)
    while not Path(target).exists() and lab.left() > 0:
        time.sleep(20)
    b = L.Balloon(lab, BALLOON_SCRIPT, "pagectl")
    info = b.start(8192)
    if not info.get("ok"):
        b.stop()
        return {"ok": False, "why": f"balloon control failed: {info}"}
    t0 = time.time()
    n = 0
    with open(target, "rb") as f:
        while n < 2 * 2 ** 30:
            c = f.read(4 * 2 ** 20)
            if not c:
                break
            n += len(c)
    t1 = time.time()
    time.sleep(3)
    m = lab.tele.metrics(t0, t1 + 3)
    b.stop()
    time.sleep(5)
    return {"ok": bool(m["pages_input_per_s"] and m["pages_input_per_s"] > 0), "pages_input_max": m["pages_input_per_s"],
            "page_reads_max": m["hard_faults_per_s"], "bytes_read": n, "balloon": info}


def section0_model(lab, mi):
    sid = f"s0_{mi.model_id}"
    srv = L.Server(lab, mi, 8192, tag=sid)
    lab.resources["server"] = srv
    pre_avail = L.avail_mb()
    info = srv.start()
    start_row(lab, srv, mi, "0", sid, info, {})
    tab = {"record": "section0_model", "ts_utc": utc_iso(), "model_id": mi.model_id, "ok": bool(info.get("ok")),
           "load_s": info.get("load_s"), "error": info.get("error"), "file_bytes": mi.file_bytes,
           "kv_bpt_meta": mi.kv_bpt_meta, "arch": mi.p, "max_ctx_native": mi.max_ctx_native, "yarn_factor": mi.yarn_factor,
           "pre_avail_mb": pre_avail}
    if not info.get("ok"):
        lab.table[mi.model_id] = tab
        srv.stop()
        lab.resources["server"] = None
        lab.emit(tab)
        return
    lg = info["log"]
    kv_log = lg.get("kv_buffer_mib")
    kv_meta_mib = mi.kv_bpt_meta * 8192 / 2 ** 20
    tab.update({"kv_buffer_mib_log_8192": kv_log, "kv_meta_mib_8192": kv_meta_mib,
                "kv_agree": (abs(kv_log - kv_meta_mib) / kv_meta_mib <= 0.02) if kv_log else None,
                "model_buffer_mib": lg.get("model_buffer_mib"), "compute_buffer_mib": lg.get("compute_buffer_mib"),
                "device_line": lg.get("device_line"), "device_free_mib": lg.get("device_free_mib"),
                "device_total_mib": lg.get("device_total_mib"),
                "props_build": info.get("build"), "mmap_lines": lg.get("mmap_lines")})
    time.sleep(4)
    g = lab.tele.gpu_ring.last()
    tab["shared_after_load_mib"] = (g["shared"] / 2 ** 20) if g and g.get("shared") else None
    expect = (lg.get("model_buffer_mib") or 0) + (kv_log or 0) + (lg.get("compute_buffer_mib") or 0)
    tab["shared_expected_mib"] = expect
    tab["shared_ok"] = bool(tab["shared_after_load_mib"] and expect and 0.6 <= tab["shared_after_load_mib"] / expect <= 1.6)
    pts = []
    fills = [800, 2600]
    for i in range(3):
        if i == 2:
            n1, t1 = pts[0][0], pts[0][1]
            n2, t2 = pts[1][0], pts[1][1]
            aa, bb = quad_fit([(n1, t1), (n2, t2)])
            fills.append(6500 if aa * 6500 + bb * 6500 ** 2 < 240 else 3500)
        fill = fills[i]
        prompt = prompt_for(srv, fill)
        n_tok = srv.tokenize(prompt)
        r = do_call(lab, srv, mi, "0", sid, prompt, n_tok, warmup=False, rep=0, extra={"purpose": "timing"}, max_tokens=64)
        if r and r.get("outcome") == "ok":
            pts.append((n_tok, r["ttft_s"], r["decode_tok_s"], r["e2e_s"], r.get("think_tag")))
        else:
            break
    tab["timing"] = pts
    if len(pts) >= 2:
        tab["ta"], tab["tb"] = quad_fit([(p[0], p[1]) for p in pts])
        tab["decode_tps"] = st.mean([p[2] for p in pts if p[2]])
        tab["prefill_tps"] = pts[-1][0] / pts[-1][1]
        tab["think_tag_seen"] = any(p[4] for p in pts)
    elif len(pts) == 1:
        tab["ta"], tab["tb"] = pts[0][1] / pts[0][0], 0.0
        tab["prefill_tps"] = pts[0][0] / pts[0][1]
        tab["decode_tps"] = pts[0][2]
    pr = probe_sequence(lab, srv, mi, "0", sid, 1500, extra={"purpose": "smoke_probe"})
    tab["probe_first_score"] = pr[0].get("score") if pr and pr[0] else None
    now = time.time()
    tab["sysman_ok"] = lab.tele.sysman.available
    tab["freq_seen"] = any(r["kind"] == "freq" and r.get("actual_mhz") for r in lab.tele.sys_ring.window(now - 90, now))
    tab["power_seen"] = any(r["kind"] == "power" and r.get("power_w") is not None for r in lab.tele.sys_ring.window(now - 90, now))
    tab["rapl_seen"] = any(x.get("rapl_pkg_mw") is not None for x in lab.tele.win_ring.window(now - 90, now))
    lab.table[mi.model_id] = tab
    srv.stop()
    lab.resources["server"] = None
    lab.emit(tab)


def section0(lab):
    read_downloads(lab)
    if lab.args.resume and "paging" in lab.done:
        lab.paging_ok = True
    else:
        ctrl = paging_control(lab)
        lab.paging_ok = ctrl.get("ok")
        lab.emit({"record": "paging_control", "ts_utc": utc_iso(), **ctrl})
        log(f"paging control: {ctrl}")
    pending = [m for m in MODEL_FILES if m not in lab.table]
    while pending:
        lab.check()
        read_downloads(lab)
        progressed = False
        for mid in list(pending):
            fn = MODEL_FILES[mid][0]
            if mid != "qwen3-4b-2507" and fn not in lab.dl_sha:
                if fn in lab.dl_dropped:
                    pending.remove(mid)
                    lab.emit({"record": "model_dropped", "model_id": mid, "ts_utc": utc_iso()})
                continue
            mi = L.ModelInfo(mid, str(Path(L.MODELS_DIR) / fn), KNOWN_4B_SHA if mid == "qwen3-4b-2507" else lab.dl_sha[fn],
                             MODEL_FILES[mid][1], MODEL_FILES[mid][2], MODEL_FILES[mid][3])
            lab.models[mid] = mi
            log(f"section 0: {mid}")
            section0_model(lab, mi)
            pending.remove(mid)
            progressed = True
        if pending and not progressed:
            time.sleep(30)
    for mid, (fn, hyb, mx, yf) in MODEL_FILES.items():
        if mid not in lab.models and mid in lab.table and (fn in lab.dl_sha or mid == "qwen3-4b-2507"):
            lab.models[mid] = L.ModelInfo(mid, str(Path(L.MODELS_DIR) / fn), KNOWN_4B_SHA if mid == "qwen3-4b-2507" else lab.dl_sha[fn], hyb, mx, yf)
    Path(lab.prefix + "_model_table.json").write_text(json.dumps(lab.table, indent=1, default=str), encoding="utf-8")


# ---------------------------------------------------------------- planning
def quad_fit(pts):
    """Least squares t = a*n + b*n^2 through the origin (b clamped to >= 0)."""
    s11 = sum(n * n for n, _ in pts)
    s12 = sum(n ** 3 for n, _ in pts)
    s22 = sum(n ** 4 for n, _ in pts)
    y1 = sum(n * t for n, t in pts)
    y2 = sum(n * n * t for n, t in pts)
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-9:
        return pts[0][1] / pts[0][0], 0.0
    a = (y1 * s22 - y2 * s12) / det
    b = (s11 * y2 - s12 * y1) / det
    if b < 0:
        return y1 / s11, 0.0
    return a, b


def prefill_s(tab, n):
    a, b = tab.get("ta"), tab.get("tb")
    if a is None:
        return n / (tab.get("prefill_tps") or 50.0)
    return a * n + b * n * n


def est_call_s(tab, fill_tokens, n_out=128):
    return prefill_s(tab, fill_tokens) + n_out / (tab.get("decode_tps") or 5.0)


def fill_cap_tokens(tab, n_ctx, cap_s):
    """Largest prompt (at most 90% of n_ctx) whose estimated prefill plus 128 decode tokens fits cap_s."""
    full = int(0.9 * n_ctx)
    budget = max(cap_s - 128 / (tab.get("decode_tps") or 5.0), 20.0)
    a, b = tab.get("ta"), tab.get("tb")
    if a is None:
        n = int(budget * (tab.get("prefill_tps") or 50.0))
    elif b and b > 0:
        n = int((-a + (a * a + 4 * b * budget) ** 0.5) / (2 * b))
    else:
        n = int(budget / a)
    return max(min(full, n), 512)


def budget_candidates(lab):
    """The llama-server device line gives (total, free); dxdiag (preflight) gave a third figure. Both llama-server
    numbers are tried because they disagree by more than 5%."""
    tot = [t.get("device_total_mib") for t in lab.table.values() if t.get("device_total_mib")]
    free = [t.get("device_free_mib") for t in lab.table.values() if t.get("device_free_mib")]
    out = {}
    if tot:
        out["device_total"] = max(tot)
    if free:
        out["device_free"] = max(free)
    return out


def budget_mib(lab):
    c = budget_candidates(lab)
    return c.get("device_free")


def need_mib(tab, n_ctx, mi):
    return (tab.get("model_buffer_mib") or mi.file_bytes / 2 ** 20) + mi.kv_bpt_meta * n_ctx / 2 ** 20 + \
        (tab.get("compute_buffer_mib") or 800.0)


def section_a_items(lab):
    """Grid around each candidate budget B (llama-server device total and device free; they disagree by >5%):
    3 points below and 5 above, KV growing by 512 MiB per step, capped at the model's supported context. A model whose
    n_ctx* exceeds its supported context for a candidate is recorded as 'budget not reachable' for that candidate."""
    items = []
    cands = budget_candidates(lab)
    lab.emit({"record": "budget", "ts_utc": utc_iso(), "candidates_mib": cands,
              "note": "dxdiag Shared Memory and the absent Shared Limit counter are in the preflight record"})
    for mid, extra_arm, prio in (("qwen3-32b", False, 10), ("qwen3-30b-a3b-2507", False, 11), ("qwen3-14b", False, 12),
                                 ("qwen3-4b-2507", True, 16), ("qwen3-8b", True, 17)):
        mi, tab = lab.models.get(mid), lab.table.get(mid)
        if not mi or not tab or not tab.get("ok") or not cands:
            continue
        w = tab.get("model_buffer_mib") or mi.file_bytes / 2 ** 20
        c = tab.get("compute_buffer_mib") or 800.0
        kv = mi.kv_bpt_meta / 2 ** 20
        cap_ctx = 10 ** 9 if extra_arm else mi.max_ctx_native * (mi.yarn_factor if mid in YARN_MODELS else 1)
        step = max(int(round(512 / kv / 256)) * 256, 256)
        grid_by = {}
        for label, B in cands.items():
            nstar = int((B - w - c) / kv)
            rec = {"record": "budget_plan", "model_id": mid, "budget_label": label, "B_mib": B, "weights_mib": w, "compute_mib": c,
                   "kv_bytes_per_token": mi.kv_bpt_meta, "n_ctx_star": nstar,
                   "max_ctx_supported": None if extra_arm else cap_ctx, "beyond_trained_context_arm": extra_arm, "ts_utc": utc_iso()}
            if nstar > cap_ctx:
                rec["result"] = "budget not reachable"
                rec["need_at_max_ctx_mib"] = w + c + kv * cap_ctx
                lab.emit(rec)
                continue
            g = [x for x in (nstar - 3 * step, nstar - 2 * step, nstar - step, nstar + step, nstar + 2 * step,
                             nstar + 3 * step, nstar + 4 * step, nstar + 5 * step) if 2048 <= x <= cap_ctx]
            rec["grid"] = g
            lab.emit(rec)
            for x in g:
                grid_by.setdefault(x, []).append((label, nstar))
        if not grid_by:
            continue
        grid = sorted(grid_by)
        yarn = ["--rope-scaling", "yarn", "--rope-scale", str(mi.yarn_factor), "--yarn-orig-ctx", str(mi.max_ctx_native)] \
            if (mid in YARN_MODELS and not extra_arm and max(grid) > mi.max_ctx_native) else []
        rng = random.Random(SEED + crc(mid))
        order = grid[:]
        rng.shuffle(order)
        lowest = min(g for g in grid)
        seq = []
        for i2, g in enumerate(order):
            seq.append(("grid", g))
            if (i2 + 1) % 3 == 0:
                seq.append(("anchor", lowest))
        for k, (kind, g) in enumerate(seq):
            fill = fill_cap_tokens(tab, g, 150.0)
            est = (tab.get("load_s") or 60) * 2 + 4 * est_call_s(tab, fill) + (0 if extra_arm else 5 * est_call_s(tab, fill, 32)) + 240
            labels = {lb: ns for lb, ns in grid_by[g]}
            items.append({"item_id": f"A_{mid}_{kind}_{g}_{k}", "section": "A", "prio": prio, "est_s": est, "model_id": mid,
                          "n_ctx": g, "kind": kind, "extra": yarn, "fill": fill, "nstar": labels, "beyond": extra_arm,
                          "budgets": cands})
        items.append({"item_id": f"A_{mid}_probe_max", "section": "A", "prio": prio + 0.5,
                      "est_s": 300 + 5 * est_call_s(tab, fill_cap_tokens(tab, max(grid), 150.0), 32), "model_id": mid,
                      "kind": "probe_max", "extra": yarn, "nstar": {}, "beyond": extra_arm, "budgets": cands, "n_ctx": None})
    return items


def section_b_items(lab):
    items = []
    order = [("qwen3-4b-2507", 20, True), ("qwen3-8b", 21, True), ("llama31-8b", 26, False), ("qwen3-14b", 27, False),
             ("qwen3-30b-a3b-2507", 32, False), ("qwen3-32b", 34, False)]
    for mid, prio, causal in order:
        mi, tab = lab.models.get(mid), lab.table.get(mid)
        if not mi or not tab or not tab.get("ok"):
            continue
        fill = fill_cap_tokens(tab, 8192, 150.0)
        cell_s = 6 * (est_call_s(tab, fill) + 45) + 40
        c100 = ["none", "nonp12"] if lab.smoke else ["none", "p4", "e4", "nonp12", "all16"]
        items.append({"item_id": f"B_{mid}_cap100", "section": "B", "prio": prio, "est_s": 240 + cell_s * 7, "model_id": mid,
                      "fill": fill, "cap": 100, "cells": c100})
        if causal:
            for j, cap in enumerate((50,) if lab.smoke else (70, 50, 30)):
                items.append({"item_id": f"B_{mid}_cap{cap}", "section": "B", "prio": prio + 0.1 * (j + 1) + 4, "est_s": 240 + cell_s * 2,
                              "model_id": mid, "fill": fill, "cap": cap, "cells": ["none", "nonp12"]})
    return items


def section_c_items(lab):
    items = []
    if not lab.paging_ok:
        lab.emit({"record": "section_c_disabled", "reason": "paging telemetry positive control did not pass", "ts_utc": utc_iso()})
        return items
    levels = [8, 0] if lab.smoke else [8, 4, 2, 1, 0, -1, -2]
    for mid, base_prio in (("qwen3-8b", 23), ("qwen3-14b", 29), ("qwen3-32b", 35)):
        mi, tab = lab.models.get(mid), lab.table.get(mid)
        if not mi or not tab or not tab.get("ok"):
            continue
        n_ctx = 16384
        need = need_mib(tab, n_ctx, mi)
        fill = fill_cap_tokens(tab, n_ctx, 150.0)
        call = est_call_s(tab, fill)
        rng = random.Random(SEED + 7 + crc(mid))
        cells = []
        for arm in (True, False):
            for lv in levels:
                target = need + lv * 1024
                if target < 3072:
                    continue
                for r in range(1 if lab.smoke else (3 if lv <= 1 else 1)):
                    cells.append((arm, lv, r, target))
        rng.shuffle(cells)
        seq = []
        for i, c in enumerate(cells):
            seq.append(("cell",) + c)
            if (i + 1) % 3 == 0:
                seq.append(("anchor", True, 8, 0, need + 8 * 1024))
        for k, (kind, arm, lv, rep, target) in enumerate(seq):
            est = 150 + (tab.get("load_s") or 60) * 1.5 + 6 * (call + 30) + 5 * (est_call_s(tab, fill, 32) + 30)
            core = kind == "anchor" or (lv <= 1 and rep == 0) or lv == 8
            prio = base_prio + (0 if core else 2) + (0 if arm else 0.3)
            items.append({"item_id": f"C_{mid}_{kind}_{'mm' if arm else 'nomm'}_{lv}_{rep}_{k}", "section": "C", "prio": prio,
                          "est_s": est, "model_id": mid, "n_ctx": n_ctx, "mmap": arm, "headroom_gb": lv, "rep": rep,
                          "target_avail_mb": target, "need_mib": need, "fill": fill, "kind": kind, "backend": "vulkan"})
    return items


def section_d_items(lab):
    items = [{"item_id": "D_prepare", "section": "D", "prio": 90, "est_s": 1800, "kind": "prepare"}]
    tab, mi = lab.table.get("qwen3-8b"), lab.models.get("qwen3-8b")
    if tab and tab.get("ok"):
        need = need_mib(tab, 16384, mi)
        fill = fill_cap_tokens(tab, 16384, 150.0)
        for k, lv in enumerate((1, 0, -1)):
            target = need + lv * 1024
            if target >= 3072:
                items.append({"item_id": f"D_C_qwen3-8b_mm_{lv}", "section": "D", "prio": 91 + k, "est_s": 900, "model_id": "qwen3-8b",
                              "n_ctx": 16384, "mmap": True, "headroom_gb": lv, "rep": 0, "target_avail_mb": target,
                              "need_mib": need, "fill": fill, "kind": "cell", "backend": "sycl"})
    return items


def trim(items, budget_s):
    kept, dropped, used = [], [], 0.0
    for it in sorted(items, key=lambda x: (x["prio"], x["item_id"])):
        if used + it["est_s"] <= budget_s:
            kept.append(it)
            used += it["est_s"]
        else:
            dropped.append(it)
    return kept, dropped, used


def build_plan(lab):
    items = section_a_items(lab) + section_b_items(lab) + section_c_items(lab) + section_d_items(lab)
    budget = lab.left() - lab.args.reserve_min * 60
    kept, dropped, used = trim(items, budget)
    kept_ids = {k["item_id"] for k in kept}
    kept = [k for k in kept if k.get("kind") != "cell" or k.get("backend") != "sycl" or "D_prepare" in kept_ids]
    lab.emit({"record": "schedule", "ts_utc": utc_iso(), "budget_s": budget, "planned_s": used, "kept": len(kept),
              "dropped": [{"item_id": d["item_id"], "est_s": d["est_s"], "prio": d["prio"]} for d in dropped]})
    log(f"schedule: {len(kept)} items kept ({used / 3600:.2f} h of {budget / 3600:.2f} h), {len(dropped)} trimmed")
    Path(lab.prefix + "_schedule.json").write_text(json.dumps({"kept": kept, "dropped": dropped}, indent=1, default=str), encoding="utf-8")
    return kept


# ---------------------------------------------------------------- Section A
def run_a_item(lab, it):
    mi, tab = lab.models[it["model_id"]], lab.table[it["model_id"]]
    if it["kind"] == "probe_max":
        started = sorted(lab.a_started.get(it["model_id"], []))
        if not started:
            return
        n_ctx = started[-1]
    else:
        n_ctx = it["n_ctx"]
    srv = L.Server(lab, mi, n_ctx, extra=it["extra"], tag=it["item_id"])
    lab.resources["server"] = srv
    info = srv.start(timeout=1800)
    fill = fill_cap_tokens(tab, n_ctx, 150.0)
    w = tab.get("model_buffer_mib") or mi.file_bytes / 2 ** 20
    need = w + (tab.get("compute_buffer_mib") or 800.0) + mi.kv_bpt_meta * n_ctx / 2 ** 20
    extra = {"need_mib": need, "budgets_mib": it["budgets"], "exceeds_budget": {k: need > v for k, v in it["budgets"].items()},
             "beyond_trained_ctx": it["beyond"], "anchor": it["kind"] == "anchor", "fill_capped": fill < int(0.9 * n_ctx),
             "rope_flags": it["extra"] or None}
    start_row(lab, srv, mi, "A", it["item_id"], info, extra)
    if info.get("ok"):
        lab.a_started.setdefault(it["model_id"], set()).add(n_ctx)
        prompt = prompt_for(srv, fill)
        n_tok = srv.tokenize(prompt)
        if it["kind"] != "probe_max":
            measured_sequence(lab, srv, mi, "A", it["item_id"], prompt, n_tok, extra=extra)
        below_all = [k for k, v in it["budgets"].items() if need <= v]
        lb = lab.a_last_below.setdefault(it["model_id"], 0)
        if it["kind"] == "probe_max" or (below_all and n_ctx >= lb and it["kind"] == "grid"):
            if it["kind"] != "probe_max":
                lab.a_last_below[it["model_id"]] = n_ctx
            probe_sequence(lab, srv, mi, "A", it["item_id"], fill, extra=extra)
    srv.stop()
    lab.resources["server"] = None


# ---------------------------------------------------------------- Section B
def run_b_item(lab, it):
    mi = lab.models[it["model_id"]]
    pc = lab.resources["powercap"]
    cap = it["cap"]
    if cap != 100 and lab.knob_ok is False:
        lab.emit({"record": "cap_arm_skipped", "item_id": it["item_id"], "ts_utc": utc_iso(),
                  "reason": "PROCTHROTTLEMAX knob did not take effect (positive control failed)"})
        return
    srv = L.Server(lab, mi, 8192, tag=it["item_id"])
    lab.resources["server"] = srv
    info = srv.start(timeout=1800)
    start_row(lab, srv, mi, "B", it["item_id"], info, {})
    if not info.get("ok"):
        srv.stop()
        lab.resources["server"] = None
        return
    prompt = prompt_for(srv, it["fill"])
    n_tok = srv.tokenize(prompt)
    rng = random.Random(SEED + 11 + crc(it["item_id"]))
    cells = it["cells"][:]
    rng.shuffle(cells)
    seq = []
    for i, c in enumerate(cells):
        seq.append((c, False))
        if cap == 100 and (i + 1) % 3 == 0:
            seq.append(("none", True))
    try:
        for cond, anchor in seq:
            lab.check()
            cell_id = f"{it['item_id']}_{cond}{'_anchor' if anchor else ''}"
            if cell_id in lab.done:
                continue
            if pc.current != cap:
                got = pc.set(cap)
                lab.emit({"record": "cap_set", "wanted": cap, "read_back": got, "ts_utc": utc_iso()})
                if got != cap:
                    lab.knob_ok = False
                    break
            mask = MASKS[cond]
            hog = None
            if mask is not None:
                hog = L.m3.start_hog(cond, mask, lab.out_dir, lab.stem + "_" + cell_id)
                lab.resources["hog"] = hog[0]
                time.sleep(5)
            res = measured_sequence(lab, srv, mi, "B", it["item_id"], prompt, n_tok,
                                    extra={"cell": cond, "cap": cap, "anchor": anchor}, co_runner=cond)
            if mask is not None:
                ips = L.m3.read_ips(hog[1])
                try:
                    aff = json.loads(Path(hog[2]).read_text())
                    aff_ok = all(m == mask for m in aff["worker_masks"])
                except Exception:
                    aff, aff_ok = None, False
                lab.emit({"record": "corunner_control", "cell": cell_id, "ips": ips, "affinity_ok": aff_ok, "affinity": aff,
                          "valid": bool(ips and ips > 0 and aff_ok), "ts_utc": utc_iso()})
                L.m3.kill_tree(hog[0].pid)
                lab.resources["hog"] = None
            pk = [r["pkg_power_w"] for r in res if r.get("pkg_power_w") is not None]
            if cond == "nonp12" and pk:
                m = st.median(pk)
                if cap == 100:
                    lab.base_pkg[it["model_id"]] = m
                elif it["model_id"] in lab.base_pkg:
                    ok = m < lab.base_pkg[it["model_id"]] * 0.97
                    lab.emit({"record": "knob_control", "model_id": it["model_id"], "cap": cap, "pkg_w": m,
                              "base_pkg_w": lab.base_pkg[it["model_id"]], "took_effect": ok, "ts_utc": utc_iso()})
                    if not ok:
                        lab.knob_ok = False
                    elif lab.knob_ok is None:
                        lab.knob_ok = True
            lab.item_done(cell_id)
            time.sleep(15)
    finally:
        srv.stop()
        lab.resources["server"] = None
        if cap != 100:
            pc.set(100) if pc.original == 100 else pc.restore()


# ---------------------------------------------------------------- Section C (and D via backend)
def run_c_item(lab, it):
    mi = lab.models[it["model_id"]]
    backend = it.get("backend", "vulkan")
    balloon = L.Balloon(lab, BALLOON_SCRIPT, it["item_id"][:50])
    lab.resources["balloon"] = balloon
    extra = {"target_avail_mb": it["target_avail_mb"], "need_mib": it["need_mib"], "anchor": it["kind"] == "anchor"}
    binfo = balloon.start(it["target_avail_mb"])
    lab.emit({"record": "balloon_start", "item_id": it["item_id"], "info": binfo, "ts_utc": utc_iso()})
    if not binfo.get("ok"):
        balloon.stop()
        lab.resources["balloon"] = None
        lab.emit(lab.row(it["section"], mi, backend, n_ctx=it["n_ctx"], mmap=it["mmap"], mem_headroom_gb=it["headroom_gb"],
                         rep=it["rep"], item_id=it["item_id"], kind="start", valid=False,
                         error="balloon did not reach target: " + str(binfo.get("why")), **extra))
        return
    time.sleep(20)
    srv = L.Server(lab, mi, it["n_ctx"], backend=backend, mmap=it["mmap"], tag=it["item_id"])
    lab.resources["server"] = srv
    extra["avail_before_server_mb"] = L.avail_mb()
    t_srv0 = time.time()
    info = srv.start(timeout=1800)
    start_row(lab, srv, mi, it["section"], it["item_id"], info, dict(extra, mem_headroom_gb=it["headroom_gb"], rep=it["rep"]))
    if info.get("ok"):
        prompt = prompt_for(srv, it["fill"])
        n_tok = srv.tokenize(prompt)
        measured_sequence(lab, srv, mi, it["section"], it["item_id"], prompt, n_tok, extra=extra, mem_headroom_gb=it["headroom_gb"])
        probe_sequence(lab, srv, mi, it["section"], it["item_id"], it["fill"], extra=extra, mem_headroom_gb=it["headroom_gb"])
    a = lab.tele.avail_ring.window(t_srv0 + (info.get("load_s") or 0), time.time())
    amax = max((x["avail_mb"] for x in a), default=None)
    valid, reason = True, None
    if not balloon.alive():
        valid, reason = False, "balloon not alive at end of measured phase"
    elif amax is not None and amax > it["target_avail_mb"] + 750:
        valid, reason = False, f"Available rose to {amax:.0f} MB above target {it['target_avail_mb']:.0f} MB (lock not held)"
    lab.emit({"record": "level_validity", "item_id": it["item_id"], "valid": valid, "reason": reason,
              "avail_max_after_load_mb": amax, "balloon_alive": balloon.alive(), "ts_utc": utc_iso()})
    srv.stop()
    lab.resources["server"] = None
    balloon.stop()
    lab.resources["balloon"] = None
    time.sleep(10)


def run_d_prepare(lab, it):
    """Download the b10970 win-sycl release, verify by sha256 of the zip, unpack, smoke a 4B start (20 min limit)."""
    dest = Path(L.BINARIES["sycl"]).parent
    info = {"record": "sycl_prepare", "ts_utc": utc_iso()}
    try:
        api = subprocess.run(["curl.exe", "-sL", "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b10970"],
                             capture_output=True, text=True, timeout=120).stdout
        assets = [(a["name"], a["browser_download_url"]) for a in json.loads(api).get("assets", [])]
        pick = [a for a in assets if "sycl" in a[0].lower() and "win" in a[0].lower()]
        if not pick:
            info.update({"ok": False, "why": "no win-sycl asset in release b10970", "assets": [a[0] for a in assets][:30]})
            lab.sycl_ok = False
            lab.emit(info)
            return
        name, url = pick[0]
        z = OUT_ROOT / name
        subprocess.run(["curl.exe", "-L", "-C", "-", "-sS", "-o", str(z), url], timeout=1500)
        import hashlib
        sha = hashlib.sha256(z.read_bytes()).hexdigest()
        dest.mkdir(parents=True, exist_ok=True)
        ps(f"Expand-Archive -Path '{z}' -DestinationPath '{dest}' -Force", timeout=600)
        info.update({"asset": name, "url": url, "zip_sha256": sha})
        if not Path(L.BINARIES["sycl"]).exists():
            cands = list(dest.rglob("llama-server.exe"))
            if cands:
                L.BINARIES["sycl"] = str(cands[0])
        mi = lab.models["qwen3-4b-2507"]
        srv = L.Server(lab, mi, 4096, backend="sycl", tag="D_smoke")
        lab.resources["server"] = srv
        r = srv.start(timeout=1200)
        info.update({"ok": bool(r.get("ok")), "load_s": r.get("load_s"), "error": r.get("error"), "build": r.get("build")})
        if r.get("ok"):
            prompt = prompt_for(srv, 800)
            res = srv.chat(prompt, 32, False)
            info["smoke_call_ok"] = res.get("outcome") == "ok"
            lab.sycl_ok = res.get("outcome") == "ok"
        else:
            lab.sycl_ok = False
        srv.stop()
        lab.resources["server"] = None
    except Exception as e:
        info.update({"ok": False, "why": repr(e)[:400]})
        lab.sycl_ok = False
    lab.emit(info)


# ---------------------------------------------------------------- main
def cleanup_partial(lab):
    r = lab.resources
    if r.get("hog") is not None:
        L.m3.kill_tree(r["hog"].pid)
        r["hog"] = None
    if r.get("server") is not None:
        try:
            r["server"].stop()
        except Exception as e:
            log(f"cleanup: server stop: {e}")
        r["server"] = None
    if r.get("balloon") is not None:
        r["balloon"].stop()
        r["balloon"] = None


def cleanup(lab):
    cleanup_partial(lab)
    pc = lab.resources.get("powercap")
    if pc is not None and pc.changed:
        rep = pc.restore()
        lab.emit({"record": "powercap_restore", **rep, "ts_utc": utc_iso()})
        log(f"powercap restore: {rep}")
    lab.tele.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-blobs", required=True)
    ap.add_argument("--deadline-h", type=float, default=11.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--only", default=None, help="run only section 0, A, B, C or D")
    ap.add_argument("--reserve-min", type=float, default=45.0, help="minutes kept free at the end for cleanup")
    ap.add_argument("--max-items", type=int, default=0, help="smoke only: run at most N items per section")
    ap.add_argument("--models", default=None, help="comma list of model ids to include (smoke and resume use)")
    args = ap.parse_args()
    if args.models:
        keep = set(args.models.split(","))
        for k in [k for k in MODEL_FILES if k not in keep]:
            del MODEL_FILES[k]
    if socket.gethostname().upper() != "EVO-T2S":
        raise SystemExit("EVO-T2S only")
    q = ps("try { (& query.exe user 2>&1) -join \"`n\" } catch { $_.Exception.Message }")
    if "No User exists" not in q:
        raise SystemExit("another interactive session is logged in: " + q)
    prov = rp.verify_deployed_blobs(DEPLOY, args.expect_blobs)
    lab = Lab(args, prov)
    lab.resources["powercap"] = L.PowerCap()
    Path(lab.prefix + "_manifest.json").write_text(json.dumps({
        "launch_utc": utc_iso(), "deadline_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(lab.deadline_ts)),
        "script_provenance": prov, "identity": lab.identity, "seed": SEED, "smoke": args.smoke,
        "power_cap_original": lab.resources["powercap"].original, "cpu_masks": MASKS}, indent=1, default=str), encoding="utf-8")
    lab.tele.start()
    note = "completed"
    try:
        time.sleep(15)
        temps = []
        for _ in range(5):
            time.sleep(1)
            t = lab.tele.temp_now()
            if t:
                temps.append(t)
        lab.idle_temp = st.median(temps) if temps else None
        pk = [lab.tele.pkg_now() for _ in range(6) if not time.sleep(1)]
        pk = [x for x in pk if x is not None]
        lab.idle_pkg = st.median(pk) if pk else None
        lab.emit({"record": "idle_temp", "idle_temp_c": lab.idle_temp, "idle_pkg_w": lab.idle_pkg,
                  "note": "no temperature sensor exposed; package-power proxy gate used" if lab.idle_temp is None else None,
                  "ts_utc": utc_iso()})
        section0(lab)
        if args.only == "0":
            return
        plan = build_plan(lab)
        runners = {"A": run_a_item, "B": run_b_item, "C": run_c_item, "D": run_c_item}
        for sec in ("A", "B", "C", "D"):
            if args.only and args.only != sec:
                continue
            sec_items = [p for p in plan if p["section"] == sec]
            if args.max_items:
                sec_items = [p for p in sec_items if p.get("kind") != "probe_max"][:args.max_items] +                             [p for p in sec_items if p.get("kind") == "probe_max"][:1]
            for it in sec_items:
                if it["item_id"] in lab.done:
                    continue
                if sec == "D" and it["kind"] != "prepare" and lab.sycl_ok is not True:
                    continue
                lab.check()
                log(f"item {it['item_id']} (est {it['est_s'] / 60:.0f} min, left {lab.left() / 3600:.2f} h)")
                try:
                    (run_d_prepare if it["kind"] == "prepare" else runners[sec])(lab, it)
                except Deadline:
                    raise
                except Exception as e:
                    lab.emit({"record": "item_error", "item_id": it["item_id"], "error": repr(e)[:500], "ts_utc": utc_iso()})
                    log(f"item error {it['item_id']}: {e!r}")
                    cleanup_partial(lab)
                    if "STOP" in str(e):
                        raise
                if sec != "B":
                    lab.item_done(it["item_id"])
            (Path(lab.out_dir) / f"{lab.stem}_section{sec}.DONE").write_text("done\n")
    except Deadline:
        note = "deadline reached"
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
        log(note)
    finally:
        cleanup(lab)
        lab.emit({"record": "run_end", "note": note, "ts_utc": utc_iso()})
        (Path(lab.out_dir) / f"{lab.stem}.DONE").write_text(note + "\n")


if __name__ == "__main__":
    main()
