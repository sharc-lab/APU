"""A-mech: why do the over-budget 32B configs fail on evo-t2s, and where exactly is the boundary?

Phases (each resumable through item_done records):
  rope     32B at n_ctx 4096 with no rope flags, --rope-scaling linear --rope-scale 4, and the overnight YaRN flags. Greedy
           first-token top-5 log-probabilities on the same prompt: yarn must differ numerically from linear at the same
           freq_scale, and the process command line is recorded, so "yarn" is shown by behaviour and not only by flags.
  vk       vulkaninfo limits (maxMemoryAllocationSize, maxBufferSize, memory heaps and flags), saved verbatim.
  bisect   server start only, no prompts. Exponential search then bisection on n_ctx in steps of 256 (the server pads
           n_ctx to a multiple of 256) between the last passing and first failing context. Reports llama.cpp's own
           projected device memory and the logged buffers at the boundary against the free-memory budget B.
           Models: qwen3-32b (primary, YaRN flags as overnight), qwen3-8b (same flags), then extras (30B-A3B, Llama-3.1-8B).
  arms     runtime policy at the first three failing 32B grid contexts: (a) -ngl 99, (b) -ngl unset with default fit on,
           (c) -ngl unset with -fit off. Same rope flags in all arms. If a server starts: layers on GPU, 5 calls (3 when a
           call is slow) and 5 art probes with the prompt filled to 90% of n_ctx, capped so one call is estimated at
           FILL_CALL_CAP_S or less (recorded as fill_capped).
  fill     32B at the last passing overnight context (111104): 1 warm-up + 3 calls and the probes with the same fill rule.
  fillmatched  32B at 111104 (ngl 99, YaRN) with the prompt forced to arm (b)'s measured 12802 tokens at 117248, so the
           two are a matched comparison at the same prompt length and rope flags.
  map      arm (b) only (ngl unset, default fit) across a 1024-token grid around each model's bisected boundary (32B,
           30B-A3B, 8B), 3 repeats in randomised order. Server start only plus one short call (512 in, 64 out) for decode
           speed when the server starts. Records outcome (runs / refused_at_context / crashed / hung), layers on GPU,
           llama.cpp's projected MiB, and the last 30 log lines plus a device-lost flag for every crash.

Usage on evo-t2s (deployed to C:\\apu\\ovn): python t2s_amech.py --expect-blobs expected_blobs.json --phases rope,vk,bisect,arms,fill,fillmatched,map
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
import time
import types
import urllib.request
import zlib
from pathlib import Path

import run_provenance as rp
import server_guard as sg
import t2s_lab as L
import t2s_overnight as ov
import t2s_queue as tq
from t2s_lab import log, ps, utc_iso

YARN = ["--rope-scaling", "yarn", "--rope-scale", "4", "--yarn-orig-ctx", "32768"]
LINEAR = ["--rope-scaling", "linear", "--rope-scale", "4"]
STEP = 256
BUDGET_MIB = 47865.0
FIT_MARGIN_MIB = 1024.0
GRID_FAIL = [117248, 119296, 121344]  # first three failing grid contexts of the overnight 32B ladder
LAST_PASS = 111104
FILL_CALL_CAP_S = 300.0
# Quadratic prefill fit from the overnight 32B GPU-only timings: t(n) = A n + Bq n^2 (seconds)
FIT_A, FIT_B = 0.00133, 1.285e-6
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip(txt):
    return ANSI.sub("", txt)


def est_prefill(n):
    return FIT_A * n + FIT_B * n * n


def read_logs(srv):
    txt = ""
    for p in (srv.log_path, srv.log_path + ".stdout.txt"):
        try:
            txt += "\n" + strip(Path(p).read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    return txt


def parse_extra(srv):
    """Everything the harness needs from one server log: llama.cpp's own projection, the layer split, the memory
    breakdown, and the exact refusal (Vulkan result text and the failing buffer size)."""
    txt = read_logs(srv)
    d = {}
    m = re.search(r"projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free device memory", txt)
    d["projected_mib"] = float(m.group(1)) if m else None
    d["fit_free_mib"] = float(m.group(2)) if m else None
    off = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", txt)
    d["layers_gpu"], d["layers_total"] = (int(off[-1][0]), int(off[-1][1])) if off else (None, None)
    lines = txt.splitlines()
    clean = lambda l: re.sub(r"^\S+\s+[A-Z]\s+", "", l).strip()[:240]
    d["breakdown"] = [clean(l) for l in lines if "common_memory_breakdown_print" in l][:6]
    d["fit_lines"] = [clean(l) for l in lines if "common_params_fit" in l or "common_fit_params" in l][:10]
    d["vk_errors"] = sorted({l.strip()[:200] for l in lines if re.search(r"vk::|ErrorOutOf|VK_ERROR", l)})
    d["alloc_failed"] = [(a, int(b)) for a, b in re.findall(r"failed to allocate (\S+) buffer of size (\d+)", txt)]
    d["error_lines"] = [clean(l) for l in lines if re.match(r"\S+\s+E\s", l)][-6:]
    d["fit_off_note"] = [clean(l) for l in lines if "fitting params" in l][:2]
    return d


def make_lab(args, prov):
    ns = types.SimpleNamespace(smoke=False, deadline_h=args.deadline_h, resume=args.resume, stem_prefix="t2s_amech",
                               only=None, reserve_min=0, no_cap_arm=True, max_items=0)
    lab = ov.Lab(ns, prov)
    lab.resources["powercap"] = None
    return lab


def load_models(lab, wanted):
    ov.read_downloads(lab)
    for mid in wanted:
        fn, hyb, mx, yf = ov.MODEL_FILES[mid]
        sha = ov.KNOWN_4B_SHA if mid == "qwen3-4b-2507" else lab.dl_sha[fn]
        lab.models[mid] = L.ModelInfo(mid, str(Path(L.MODELS_DIR) / fn), sha, hyb, mx, yf)


def am_row(lab, mi, srv, item_id, phase, **kw):
    return lab.row("AM", mi, srv.backend, n_ctx=srv.n_ctx, mmap=srv.mmap, load_mode=srv.load_mode, item_id=item_id, phase=phase,
                   ngl=srv.ngl, fit=srv.fit, **kw)


def start_and_record(lab, mi, n_ctx, tag, item_id, phase, extra_flags, ngl=99, fit=None, **kw):
    """Start one server (no prompt), write one start row, return (srv, info, extra-parse). The caller stops the server."""
    srv = L.Server(lab, mi, n_ctx, extra=extra_flags, tag=tag, ngl=ngl, fit=fit)
    lab.resources["server"] = srv
    info = srv.start(timeout=1800)
    px = parse_extra(srv)
    lg = info.get("log", {})
    r = am_row(lab, mi, srv, item_id, phase, kind="start", valid=info.get("ok"), error=info.get("error"), load_s=info.get("load_s"),
               server_pid=info.get("pid"), llama_build=info.get("build"), t_start_utc=info["t_start"], t_end_utc=info["t_end"],
               exit_code=info.get("exit_code"), rope_flags=extra_flags or None, cmdline=" ".join(srv._cmd()),
               kv_buffer_mib_log=lg.get("kv_buffer_mib"), compute_buffer_mib_log=lg.get("compute_buffer_mib"),
               model_buffer_mib_log=lg.get("model_buffer_mib"), device_line=lg.get("device_line"),
               device_free_mib=lg.get("device_free_mib"), **px, **kw)
    lab.emit(r)
    return srv, info, px


# ---------------------------------------------------------------- rope
def completion_top(srv, prompt, n_predict=16):
    ok, lp = srv.alive_and_ours()
    if not ok:
        raise RuntimeError(f"STOP: listener {lp} is not our server")
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "seed": 42, "n_probs": 5, "cache_prompt": False}
    req = urllib.request.Request(f"{L.SERVER_URL}/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read())
    probs = j.get("completion_probabilities") or []
    first = probs[0] if probs else {}
    top = first.get("top_logprobs") or first.get("top_probs") or []
    return {"text": j.get("content"), "first_top": [(t.get("token"), t.get("logprob", t.get("prob"))) for t in top][:5],
            "first_token": first.get("token"), "first_logprob": first.get("logprob"), "raw_first": json.dumps(first)[:600]}


def phase_rope(lab):
    mi = lab.models["qwen3-32b"]
    arms = (("none", []), ("linear", LINEAR), ("yarn", YARN))
    res = {}
    for tag, flags in arms:
        srv, info, px = start_and_record(lab, mi, 4096, f"rope_{tag}", f"AM_rope_{tag}", "rope", flags)
        if not info.get("ok"):
            srv.stop()
            lab.resources["server"] = None
            res[tag] = {"error": info.get("error")}
            continue
        cmdline = sg.process_cmdline(srv.pid)
        prompt = ov.prompt_for(srv, 1500)
        out = completion_top(srv, prompt)
        res[tag] = {**out, "cmdline_process": cmdline}
        lab.emit(am_row(lab, mi, srv, f"AM_rope_{tag}", "rope", kind="rope_probe", valid=True, rope_flags=flags or None,
                        prompt_tokens=srv.tokenize(prompt), output=out["text"], first_top=out["first_top"], cmdline_process=cmdline))
        srv.stop()
        lab.resources["server"] = None

    def lp(tag):
        try:
            return res[tag]["first_top"]
        except Exception:
            return None
    dif = lambda a, b: None if lp(a) is None or lp(b) is None else max(abs((x[1] or 0) - (y[1] or 0)) for x, y in zip(lp(a), lp(b)))
    verdict = {"record": "rope_verdict", "ts_utc": utc_iso(), "max_abs_logprob_diff_yarn_vs_linear": dif("yarn", "linear"),
               "max_abs_logprob_diff_yarn_vs_none": dif("yarn", "none"), "max_abs_logprob_diff_linear_vs_none": dif("linear", "none"),
               "text_equal": {"yarn_linear": res["yarn"].get("text") == res["linear"].get("text"),
                              "yarn_none": res["yarn"].get("text") == res["none"].get("text")},
               "cmdline_has_yarn": "--rope-scaling yarn" in (res["yarn"].get("cmdline_process") or "")}
    lab.emit(verdict)
    log(f"rope verdict: {verdict}")


# ---------------------------------------------------------------- vulkaninfo
def phase_vk(lab):
    out = lab.prefix + "_vulkaninfo.txt"
    p = subprocess.run(["vulkaninfo"], capture_output=True, text=True, errors="replace", timeout=180)
    Path(out).write_text(p.stdout, encoding="utf-8")
    txt = p.stdout
    rec = {"record": "vulkan_limits", "ts_utc": utc_iso(), "file": Path(out).name, "returncode": p.returncode}
    for k in ("maxMemoryAllocationSize", "maxBufferSize", "maxStorageBufferRange", "maxMemoryAllocationCount"):
        m = re.search(rf"{k}\s*=\s*(\S+)", txt)
        rec[k] = m.group(1) if m else None
    heaps = re.findall(r"memoryHeaps\[(\d+)\]:\s*\n?\s*size\s*=\s*(\d+)\s*\((0x[0-9a-f]+)\)[^\n]*\n\s*budget\s*=\s*(\d+)[^\n]*\n\s*usage\s*=\s*(\d+)[^\n]*\n\s*flags:\s*\n?([^\n]*(?:\n\s+MEMORY_HEAP[^\n]*)*)", txt)
    rec["heaps"] = [{"index": int(h[0]), "size_bytes": int(h[1]), "size_mib": int(h[1]) / 2 ** 20, "budget_mib": int(h[3]) / 2 ** 20,
                     "usage_mib": int(h[4]) / 2 ** 20, "flags": " ".join(h[5].split())} for h in heaps]
    if not heaps:
        rec["heaps_raw"] = [l.strip() for l in txt.splitlines() if re.search(r"memoryHeaps|heap|size\s*=", l, re.I)][:30]
    i = txt.find("VkPhysicalDeviceMemoryProperties")
    rec["memory_properties_text"] = [l.rstrip() for l in txt[i:i + 6000].splitlines()[:60]] if i >= 0 else None
    rec["device_name"] = (re.search(r"deviceName\s*=\s*(.+)", txt) or [None, None])[1]
    lab.emit(rec)
    log(f"vulkan limits: {json.dumps(rec)[:600]}")


# ---------------------------------------------------------------- bisection
class Prober:
    def __init__(self, lab, mi, flags, label, beyond_from=None):
        self.lab, self.mi, self.flags, self.label, self.cache, self.n = lab, mi, flags, label, {}, 0
        self.beyond_from = beyond_from

    def probe(self, n_ctx):
        n_ctx = int(round(n_ctx / STEP)) * STEP
        if n_ctx in self.cache:
            return self.cache[n_ctx]
        self.lab.check()
        self.n += 1
        srv, info, px = start_and_record(self.lab, self.mi, n_ctx, f"bis_{self.label}_{n_ctx}_{self.n}", f"AM_bis_{self.label}_{n_ctx}_{self.n}",
                                         "bisect", self.flags, bisect_label=self.label, beyond_trained_ctx=n_ctx > self.mi.max_ctx_native,
                                         kv_bpt_meta=self.mi.kv_bpt_meta)
        started, cap = bool(info.get("ok")), None
        m = re.match(r"guard: server does not match intended config: n_ctx (\d+) != (\d+)", info.get("error") or "")
        if not started and m and "llama_server: listening on" in read_logs(srv):
            # llama-server created the context (the KV buffer of the requested size is in its log) but /props reports a
            # smaller n_ctx (131072 with YaRN factor 4, 262144 native). For a memory test that is a start, not a refusal.
            started, cap = True, int(m.group(1))
        srv.stop()
        self.lab.resources["server"] = None
        self.lab.emit({"record": "bisect_probe", "label": self.label, "n_ctx": n_ctx, "started": started, "props_n_ctx_cap": cap,
                       "guard_error": info.get("error") if cap else None, "ts_utc": utc_iso()})
        lg = info.get("log", {})
        logged = sum(x for x in (lg.get("model_buffer_mib"), lg.get("kv_buffer_mib"), lg.get("compute_buffer_mib")) if x)
        res = {"ok": started, "props_cap": cap, "projected_mib": px.get("projected_mib"), "logged_mib": logged or None,
               "error": info.get("error"), "vk": px.get("vk_errors"), "alloc_failed": px.get("alloc_failed")}
        self.cache[n_ctx] = res
        log(f"bisect {self.label} n_ctx {n_ctx}: ok={res['ok']} projected={res['projected_mib']} logged={res['logged_mib']}")
        return res

    def find(self, n_start, step):
        """Exponential search from n_start, then bisection to STEP tokens. Returns (last_ok, first_fail)."""
        n = int(round(n_start / STEP)) * STEP
        ok = self.probe(n)["ok"]
        lo = hi = None
        if ok:
            lo = n
            while hi is None:
                n += step
                if self.probe(n)["ok"]:
                    lo = n
                    step *= 2
                else:
                    hi = n
        else:
            hi = n
            while lo is None:
                n = max(STEP, n - step)
                if self.probe(n)["ok"]:
                    lo = n
                else:
                    hi = n
                    step *= 2
                if n <= STEP and lo is None:
                    return None, hi
        while hi - lo > STEP:
            mid = (lo + hi) // 2 // STEP * STEP
            if mid <= lo:
                mid = lo + STEP
            if self.probe(mid)["ok"]:
                lo = mid
            else:
                hi = mid
        return lo, hi


def boundary_record(lab, mi, label, flags, pr, lo, hi, target_note=None):
    a, b = pr.cache.get(lo, {}), pr.cache.get(hi, {})
    kv_mib_step = mi.kv_bpt_meta * STEP / 2 ** 20
    rec = {"record": "bisect_result", "ts_utc": utc_iso(), "label": label, "model_id": mi.model_id, "rope_flags": flags or None,
           "last_ok_n_ctx": lo, "first_fail_n_ctx": hi, "step_tokens": STEP, "step_mib_of_kv": kv_mib_step,
           "projected_mib_last_ok": a.get("projected_mib"), "projected_mib_first_fail": b.get("projected_mib"),
           "logged_mib_last_ok": a.get("logged_mib"), "budget_B_mib": BUDGET_MIB, "fit_margin_mib": FIT_MARGIN_MIB,
           "beyond_trained_ctx": (hi or 0) > mi.max_ctx_native, "n_probes": pr.n, "first_fail_error": b.get("error"), "props_cap_at_last_ok": a.get("props_cap"),
           "first_fail_vk": b.get("vk"), "first_fail_alloc": b.get("alloc_failed"), "note": target_note}
    lab.emit(rec)
    log(f"boundary {label}: last_ok {lo} (projected {a.get('projected_mib')}) first_fail {hi} (projected {b.get('projected_mib')}) B={BUDGET_MIB}")
    return rec


def phase_bisect(lab, models):
    primary = None
    for mid in models:
        item = f"AM_bisect_v2_{mid}"
        if item in lab.done:
            continue
        mi = lab.models[mid]
        if mid == "qwen3-32b":
            flags, n0, step = YARN, LAST_PASS, 2048
        elif mid in ov.YARN_MODELS:
            flags = YARN
        else:
            flags = []
        pr = Prober(lab, mi, flags, mid)
        if mid != "qwen3-32b":
            # place the start where this model's KV needs the same total memory as the 32B boundary
            base = pr.probe(4096)
            target = (primary or {}).get("projected_mib_last_ok") or 46494.0
            n0 = 4096 + (target - (base["projected_mib"] or 0)) * 2 ** 20 / mi.kv_bpt_meta
            step = 4096
        lo, hi = pr.find(n0, step)
        if lo is None:
            lab.emit({"record": "bisect_result", "label": mid, "model_id": mid, "error": "no passing context found", "ts_utc": utc_iso()})
            lab.item_done(item)
            continue
        rec = boundary_record(lab, mi, mid, flags, pr, lo, hi)
        if mid == "qwen3-32b":
            primary = rec
            # each boundary config once more to see whether the flip is stable
            for n in (lo, hi):
                pr.cache.pop(n, None)
                r2 = pr.probe(n)
                lab.emit({"record": "bisect_repeat", "label": mid, "n_ctx": n, "ok": r2["ok"], "projected_mib": r2["projected_mib"],
                          "first_time_ok": (pr.cache.get(n) or {}).get("ok"), "ts_utc": utc_iso()})
        lab.item_done(item)


# ---------------------------------------------------------------- filled calls
def fill_for(t2k_s, n_ctx):
    """Largest fill <= 90% of n_ctx whose estimated single call is <= FILL_CALL_CAP_S, scaled from a measured 2K call."""
    scale = t2k_s / est_prefill(2000) if t2k_s and t2k_s > 0 else 1.0
    n = int(0.9 * n_ctx)
    while n > 2000 and est_prefill(n) * scale > FILL_CALL_CAP_S:
        n = int(n * 0.9)
    return max(2000, n // 100 * 100), scale


def filled_calls(lab, mi, srv, item_id, extra, n_calls=None):
    """Timing call at 2K, choose the fill, then measured calls and the five art probes at that fill."""
    p2 = ov.prompt_for(srv, 2000)
    t2 = ov.do_call(lab, srv, mi, "AM", item_id, p2, srv.tokenize(p2), warmup=True, rep=-2, extra=dict(extra, purpose="fill_timing"))
    t2k = (t2 or {}).get("ttft_s")
    fill, scale = fill_for(t2k, srv.n_ctx)
    ex = dict(extra, fill_target_90pct=int(0.9 * srv.n_ctx), fill_used=fill, fill_capped=fill < int(0.9 * srv.n_ctx),
              t2k_s=t2k, fill_est_call_s=est_prefill(fill) * scale, fill_call_cap_s=FILL_CALL_CAP_S)
    prompt = ov.prompt_for(srv, fill)
    n_tok = srv.tokenize(prompt)
    ov.measured_sequence(lab, srv, mi, "AM", item_id, prompt, n_tok, extra=ex, n_calls=n_calls)
    ov.probe_sequence(lab, srv, mi, "AM", item_id, fill, extra=ex)
    return ex


def phase_arms(lab):
    mi = lab.models["qwen3-32b"]
    lab.emit({"record": "fill_feasibility", "ts_utc": utc_iso(), "model_id": "qwen3-32b",
              "fit": {"A_s_per_tok": FIT_A, "B_s_per_tok2": FIT_B, "source": "overnight 32B GPU-only TTFT at 4.5K and 14.7K tokens"},
              "est_call_s_at_90pct": {str(n): est_prefill(int(0.9 * n)) for n in [LAST_PASS] + GRID_FAIL},
              "note": "a call filled to 90% of these contexts is estimated in hours; fills are capped, see fill_capped"})
    for n_ctx in GRID_FAIL:
        for arm, kw in (("a_ngl99", {"ngl": 99, "fit": None}), ("b_fit_default", {"ngl": None, "fit": None}),
                        ("c_fit_off", {"ngl": None, "fit": "off"})):
            item = f"AM_arm2_{arm}_{n_ctx}"
            if item in lab.done:
                continue
            lab.check()
            srv, info, px = start_and_record(lab, mi, n_ctx, item, item, "arms", YARN, arm=arm, **kw)
            log(f"arm {arm} n_ctx {n_ctx}: ok={info.get('ok')} layers={px.get('layers_gpu')}/{px.get('layers_total')} err={str(info.get('error'))[:80]}")
            if info.get("ok"):
                try:
                    filled_calls(lab, mi, srv, item, {"arm": arm, "layers_gpu": px.get("layers_gpu"), "layers_total": px.get("layers_total"),
                                                       "rope_flags": YARN})
                except ov.Deadline:
                    raise
                except Exception as e:
                    lab.emit({"record": "item_error", "item_id": item, "error": repr(e)[:400], "ts_utc": utc_iso()})
            srv.stop()
            lab.resources["server"] = None
            lab.item_done(item)


def phase_fill(lab):
    mi = lab.models["qwen3-32b"]
    item = f"AM_fill_{LAST_PASS}"
    if item in lab.done:
        return
    srv, info, px = start_and_record(lab, mi, LAST_PASS, item, item, "fill", YARN, arm="last_pass_ngl99")
    if info.get("ok"):
        filled_calls(lab, mi, srv, item, {"arm": "last_pass_ngl99", "layers_gpu": px.get("layers_gpu"), "rope_flags": YARN}, n_calls=3)
    srv.stop()
    lab.resources["server"] = None
    lab.item_done(item)


# ---------------------------------------------------------------- matched fill (same prompt length as arm b at 117248)
MATCHED_PROMPT_TOKENS = 12802  # arm2_b_fit_default_117248's measured prompt length


def phase_fillmatched(lab):
    """32B at the last passing ngl=99 context (111104), same YaRN flags, prompt matched to arm (b)'s 12802 tokens at
    117248, so the arm (b) slowdown is a matched TTFT/decode comparison and not confounded by prompt length."""
    mi = lab.models["qwen3-32b"]
    item = f"AM_fillmatch_{LAST_PASS}"
    if item in lab.done:
        return
    srv, info, px = start_and_record(lab, mi, LAST_PASS, item, item, "fillmatch", YARN, arm="matched_ngl99")
    if info.get("ok"):
        prompt = ov.prompt_for(srv, MATCHED_PROMPT_TOKENS)
        n_tok = srv.tokenize(prompt)
        ex = {"arm": "matched_ngl99", "rope_flags": YARN, "fill_used": n_tok, "matched_to": "AM_arm2_b_fit_default_117248",
              "matched_target_tokens": MATCHED_PROMPT_TOKENS, "layers_gpu": px.get("layers_gpu")}
        ov.measured_sequence(lab, srv, mi, "AM", item, prompt, n_tok, extra=ex, n_calls=3)
    else:
        lab.emit({"record": "item_error", "item_id": item, "error": info.get("error"), "ts_utc": utc_iso()})
    srv.stop()
    lab.resources["server"] = None
    lab.item_done(item)


# ---------------------------------------------------------------- fit-policy map (arm b only, cheap, start + one short call)
MAP_SPEC = [
    ("qwen3-32b", 115968, 125440, 1024, YARN),
    ("qwen3-30b-a3b-2507", 320512, 329728, 1024, []),
    ("qwen3-8b", 305408, 314432, 1024, YARN),
]


def classify_start(info):
    if info.get("ok"):
        return "runs"
    code = info.get("exit_code")
    if code is None:
        return "hung"
    if code == 1:
        return "refused_at_context"
    return "crashed"


def phase_map(lab):
    for mid, start_n, end_n, step, flags in MAP_SPEC:
        mi = lab.models.get(mid)
        if mi is None:
            lab.emit({"record": "map_skipped", "model_id": mid, "reason": "model not loaded", "ts_utc": utc_iso()})
            log(f"map {mid}: SKIPPED, model not loaded")
            continue
        points = list(range(start_n, end_n + 1, step))
        trials = [(n, r) for n in points for r in range(3)]
        rng = random.Random(20260925 + zlib.crc32(mid.encode()))
        rng.shuffle(trials)
        for n_ctx, rep in trials:
            item = f"AM_map_{mid}_{n_ctx}_{rep}"
            if item in lab.done:
                continue
            lab.check()
            srv, info, px = start_and_record(lab, mi, n_ctx, item, item, "map", flags, ngl=None, fit=None, rep=rep, arm="b_fit_default")
            outcome = classify_start(info)
            crash_tail = None
            if outcome == "crashed":
                crash_tail = read_logs(srv).splitlines()[-30:]
            lab.emit({"record": "map_probe", "model_id": mid, "n_ctx": n_ctx, "rep": rep, "outcome": outcome,
                      "exit_code": info.get("exit_code"), "layers_gpu": px.get("layers_gpu"), "layers_total": px.get("layers_total"),
                      "projected_mib": px.get("projected_mib"), "error": info.get("error"), "crash_log_tail": crash_tail,
                      "device_lost": bool(crash_tail and any("device lost" in l.lower() for l in crash_tail)), "ts_utc": utc_iso()})
            log(f"map {mid} n_ctx {n_ctx} rep {rep}: {outcome} layers={px.get('layers_gpu')}/{px.get('layers_total')} "
                f"projected={px.get('projected_mib')} exit_code={info.get('exit_code')}")
            if outcome == "runs":
                try:
                    prompt = ov.prompt_for(srv, 512)
                    n_tok = srv.tokenize(prompt)
                    ov.do_call(lab, srv, mi, "AM", item, prompt, n_tok, warmup=False, rep=0, max_tokens=64,
                               extra={"purpose": "map_decode_speed", "layers_gpu": px.get("layers_gpu"), "rope_flags": flags or None})
                except ov.Deadline:
                    raise
                except Exception as e:
                    lab.emit({"record": "item_error", "item_id": item, "error": repr(e)[:400], "ts_utc": utc_iso()})
            srv.stop()
            lab.resources["server"] = None
            lab.item_done(item)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-blobs", required=True)
    ap.add_argument("--deadline-h", type=float, default=9.0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--phases", default="rope,vk,bisect,arms,fill")
    ap.add_argument("--bisect-models", default="qwen3-32b,qwen3-8b,qwen3-30b-a3b-2507,llama31-8b")
    args = ap.parse_args()
    if socket.gethostname().upper() != "EVO-T2S":
        raise SystemExit("EVO-T2S only")
    q = ps("try { (& query.exe user 2>&1) -join \"`n\" } catch { $_.Exception.Message }")
    if "No User exists" not in q:
        raise SystemExit("another interactive session is logged in: " + q)
    prov = rp.verify_deployed_blobs(ov.DEPLOY, args.expect_blobs)
    lab = make_lab(args, prov)
    phases = args.phases.split(",")
    bm = args.bisect_models.split(",")
    map_models = [m for m, *_ in MAP_SPEC]
    extra_models = (bm if "bisect" in phases else []) + (map_models if "map" in phases else [])
    load_models(lab, sorted(set(["qwen3-32b"] + extra_models)))
    Path(lab.prefix + "_manifest.json").write_text(json.dumps({"launch_utc": utc_iso(), "script_provenance": prov, "identity": lab.identity,
                                                              "phases": phases, "bisect_models": bm, "yarn_flags": YARN, "linear_flags": LINEAR,
                                                              "budget_mib": BUDGET_MIB}, indent=1, default=str), encoding="utf-8")
    lab.tele.start()
    note = "completed"
    try:
        time.sleep(15)
        pk = [lab.tele.pkg_now() for _ in range(6) if not time.sleep(1)]
        lab.idle_pkg = st.median([x for x in pk if x is not None]) if any(x is not None for x in pk) else None
        lab.idle_temp = None
        for ph in phases:
            if f"AM_phase_{ph}" in lab.done:
                continue
            log(f"phase {ph}")
            {"rope": lambda: phase_rope(lab), "vk": lambda: phase_vk(lab), "bisect": lambda: phase_bisect(lab, bm),
             "arms": lambda: phase_arms(lab), "fill": lambda: phase_fill(lab), "fillmatched": lambda: phase_fillmatched(lab),
             "map": lambda: phase_map(lab)}[ph]()
            lab.item_done(f"AM_phase_{ph}")
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
