"""Night 2 on evo-t2s: core-set and duty-cycle power coupling (B1, B2, B3), memory-lock mmap arm at more headroom
levels (C1), and a PERFBOOSTMODE=0 arm, run in priority order (B1, C1, B2, B3, PERFBOOST) with backfill if any phase
finishes early. Chained onto the end of the A-mech run via harness/t2s_queue.py; also runs standalone.

Priority order (cheap, high-value phases first, so a deadline cuts the least valuable work): b1, b2, c1 (qwen3-8b only),
b3, c1b (qwen3-14b and qwen3-32b, 1 repeat each), perfboost. Each phase is wrapped in its own try/except in main(): an
exception inside one phase is logged and recorded as a phase_error, and the run moves on to the next phase. Only the
deadline and the interactive-session check stop the whole run.

Sections (each resumable through item_done records):
  B1  Core-set sweep on qwen3-4b-2507 and qwen3-8b: none, p4, e4, lp4, e8, p4e4, nonp12, all16 (context 8192, prompt
      7,368 tokens as in the overnight Section B). Per-core-type (P/E/LP-E) '% Processor Performance' and best-effort
      RAPL PP0/PP1 are sampled continuously by the same 1 s Windows-counter stream that already reports igpu_mhz and
      pkg_power_w (harness/t2s_lab.py SYS_PS), so every call's row carries the median over that call's own window, not
      a single sample taken before the call. Level Zero Sysman iGPU throttle reasons come from the same
      zesFrequencyGetState sample the overnight run used; a separate zesFrequencyGetThrottleReasons call was not
      added. Positive control: during the all16 condition, one direct Sysman sample is taken and its throttle-reason
      bitmask is required to be non-zero once the iGPU is at or below 1700 MHz; if the API returns nothing the control
      is recorded as unavailable and the phase continues.
  B2  Duty-cycle sweep (25/50/75/100%) of the nonp12 core set on qwen3-4b-2507 and qwen3-8b, same context and prompt.
  C1  mmap vs none memory lock, qwen3-8b only: headroom 0, -1, -2 GB, 3 repeats, randomised order.
  B3  none/p4/e4/nonp12 on llama31-8b, qwen3-14b, qwen3-30b-a3b-2507 and qwen3-32b (context 8192).
  c1b mmap vs none memory lock, qwen3-14b and qwen3-32b: headroom 0 and -1 GB, 1 repeat each tonight (cut first if the
      estimated total exceeds the deadline). Both C1 phases reuse the overnight model table (weights/KV/compute) for
      the headroom arithmetic; needs `results/t2s_overnight_*_model_table.json` from last night's run.
  perfboost  PERFBOOSTMODE=0 on AC for the qwen3-8b none vs nonp12 pair, restored in finally. Positive control:
      package power under nonp12 must drop versus the PERFBOOSTMODE default measured in B1.
  PX2 evo-x2 ONLY, opt in with `--phases px2`: separates the co-runner effect's power component from its memory-
      bandwidth component with a spin hog and a STREAM-triad hog on the SAME cores (S4 vs B4), plus an S-ladder for
      dose-response and CCD placement. Core sets come from real L2/L3 cache groups (harness/win_cpu_topology.py), so
      the phase records px2_disabled and returns on any part whose cache layout does not support that reading --
      including evo-t2s. Deploying it needs bw_hog.py and win_cpu_topology.py in the scripts/deploy_evo.py path list
      alongside spin_hog_affinity.py. See the PX2 section docstring below.

Usage on evo-t2s (deployed to C:\\apu\\ovn):
  python t2s_night2.py --expect-blobs expected_blobs.json --deadline-h 10 --overnight-table <model_table.json> [--resume <stem>]
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
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

import run_provenance as rp
import win_cpu_topology as wct  # PX2: real L2 (SMT sibling) and L3 (CCD) groups, not an assumed core map
import t2s_lab as L
import host_config as hc
import t2s_amech as am  # A70: reuses Prober/start_and_record/boundary_record (same ov.Lab shape, no reimplementation)
import t2s_overnight as ov
import t2s_queue as tq
from t2s_lab import log, ps, utc_iso
from stage_c_position_pressure import left_truncate as sc_left_truncate  # R1 check set: same char-truncation as stage C

try:
    import quality_suite as qs  # A70: the Q0 suite at 8k, once qs.build_task/run_suite exist
except ImportError:
    qs = None

PYTHON = sys.executable

SEED = 20260927
CTX_B = 8192
FILL_B = 7368
# Confirmed live on evo-t2s 2026-09-28 via win_cpu_topology.py (GetLogicalProcessorInformationEx, RelationCache,
# level 2): P-cores 0-3 each have an individual 3 MB L2; E-cores 4-7 share one 4 MB L2 (cluster A); E-cores 8-11
# share a second 4 MB L2 (cluster B); LP-E cores 12-15 share their own 4 MB L2. Matches the assumed 4-7/8-11 split
# exactly, so B4 uses it unchanged, but it was read, not assumed.
E_CLUSTER_A = list(range(4, 8))
E_CLUSTER_B = list(range(8, 12))
B4_MASKS = [("none", None), ("e4_clusterA", 0x00F0), ("e4_clusterB", 0x0F00), ("e4_split", 0x0330),
            ("e2", 0x0030), ("e6", 0x03F0), ("e8", 0x0FF0), ("lp4", 0xF000), ("e8_lp4", 0xFFF0)]

B1_MODELS = ["qwen3-4b-2507", "qwen3-8b"]
B1_MASKS = [("none", None), ("p4", 0x000F), ("e4", 0x00F0), ("lp4", 0xF000), ("e8", 0x0FF0),
            ("p4e4", 0x00FF), ("nonp12", 0xFFF0), ("all16", 0xFFFF)]
B2_DUTY = [25, 50, 75, 100]
B3_MODELS = ["llama31-8b", "qwen3-14b", "qwen3-30b-a3b-2507", "qwen3-32b"]
B3_CORESETS = [("none", None), ("p4", 0x000F), ("e4", 0x00F0), ("nonp12", 0xFFF0)]
CPU_CLASSES = {"p": list(range(0, 4)), "e": list(range(4, 12)), "lpe": list(range(12, 16))}
C1_SPEC_8B = [("qwen3-8b", [0, -1, -2], 3)]
C1_SPEC_REST = [("qwen3-14b", [0, -1], 2), ("qwen3-32b", [0, -1], 2)]
C1_SPEC = C1_SPEC_8B + C1_SPEC_REST

# R1 speed side (Addendum v3, item 2): matched core set across machines. TTFT(prompt_tokens) at these lengths builds
# each machine's speed curve for the Fig 6.1 envelope; joined against evo-x2's quality-side data per the addendum's
# machine(memory->context, speed) x model(context->quality) decomposition. llama-3.3-70b is skipped by load_models
# (and so by this phase) until its background download/verify actually lands in downloads.jsonl.
R1_SPEED_MODELS = ["qwen3-8b", "qwen3-14b", "qwen3-32b", "llama-3.3-70b"]
R1_SPEED_LENGTHS = [1024, 2048, 4096, 8192, 16384]
R1_SPEED_CTX = 20480  # fits the longest sweep point (16384) plus headroom for the 128-token completion

# R1 quality-side check set (Addendum v3, item 2): the exact Fig 6.1 spec in docs/PAPER_OUTLINE.md Section 6, ported
# onto the current lab/guard (see phase_r1_check below), run only on evo-t2s's check-set models -- the full ladder
# runs on evo-x2 instead (docs/CLAIMS_LEDGER.md J-01).
R1_CHECK_MODELS = ["qwen3-8b", "qwen3-14b"]
R1_CHECK_PROBE_IDS = ["rag_01", "rag_02", "rag_05", "sea_04", "sea_01"]
R1_CHECK_RATIOS = [1.20, 1.00, 0.85, 0.70, 0.55, 0.40]
R1_CHECK_ARMS = ["LATE", "EARLY"]
R1_CHECK_REPS = 3

# R1d (Addendum v3): the full stage_c_position_pressure.py spec -- all 11 rag_*/sea_* probes in segments.jsonl (R1a's
# 5-probe set is a subset of these same 11), same ratios/arms/reps/filler, so it reuses _phase_position_pressure
# directly with only the probe list and section tag different.
R1D_MODELS = ["qwen3-8b", "qwen3-14b"]
R1D_PROBE_IDS = ["rag_01", "rag_02", "rag_03", "rag_04", "rag_05", "rag_06", "sea_01", "sea_03", "sea_04", "sea_05", "sea_06"]
R1_CHECK_FILLER = 4000
R1_CHECK_CTX = 8192


def per_core_perf():
    """One-shot '% Processor Performance' per logical CPU, grouped into P/E/LP-E medians. Diagnostic only: the actual
    per-call telemetry no longer calls this (see the module docstring) -- it is sampled every second by the streaming
    SYS_PS counter in t2s_lab.py and windowed per call in Telemetry.metrics(), same as igpu_mhz. This function stays
    for the read-only sanity check and for interactive use. Returns None per group when the counter has no samples for
    it (the counter may not exist on this CPU/OS combination). On this machine InstanceName is "node,cpu" (e.g. "0,10"),
    confirmed by the read-only check before any real run; the logical CPU is the part after the last comma."""
    out = ps('try { (Get-Counter -Counter "\\Processor Information(*)\\% Processor Performance" '
             '-ErrorAction Stop).CounterSamples | Where-Object { $_.InstanceName -match "^[0-9]+(,[0-9]+)*$" } | '
             'ForEach-Object { $_.InstanceName + "=" + $_.CookedValue } } catch { "" }', 20)
    d = {}
    for tok in (out or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                d[int(k.rsplit(",", 1)[-1])] = float(v)
            except ValueError:
                pass
    def grp(idxs):
        vals = [d[i] for i in idxs if i in d]
        return st.median(vals) if vals else None
    return {"p_pct_perf": grp(CPU_CLASSES["p"]), "e_pct_perf": grp(CPU_CLASSES["e"]), "lpe_pct_perf": grp(CPU_CLASSES["lpe"]),
            "n_cores_read": len(d)}


def rapl_pp01():
    """Best-effort RAPL PP0 (cores)/PP1 (uncore/DRAM) via the same Energy Meter counter set the harness already reads
    for the package. Diagnostic only, same note as per_core_perf: the streaming counter already carries this into
    every row (rapl_pp0_w, rapl_pp1_w), null when the platform does not expose those instances."""
    out = ps('try { $s = (Get-Counter -ListSet "Energy Meter" -ErrorAction Stop).PathsWithInstances; '
             '($s | Where-Object { $_ -like "*rapl_package0_cores*" -or $_ -like "*rapl_package0_uncore*" -or $_ -like "*rapl_dram*" }) -join ";" } catch { "" }', 20)
    paths = [p for p in (out or "").split(";") if p]
    if not paths:
        return {"pp0_w": None, "pp1_w": None, "available": False, "instances_seen": (out or "")[:200]}
    vals = ps('(Get-Counter -Counter @(' + ",".join(f'"{p}"' for p in paths) + ') -ErrorAction SilentlyContinue).CounterSamples | '
              'ForEach-Object { $_.Path + "=" + $_.CookedValue }', 20)
    d = {}
    for l in (vals or "").splitlines():
        if "=" in l:
            k, v = l.rsplit("=", 1)
            try:
                d[k.strip()] = float(v) / 1000.0
            except ValueError:
                pass
    pp0 = next((v for k, v in d.items() if "cores" in k.lower()), None)
    pp1 = next((v for k, v in d.items() if "uncore" in k.lower() or "dram" in k.lower()), None)
    return {"pp0_w": pp0, "pp1_w": pp1, "available": True, "instances_seen": paths}


def make_lab(args, prov, gpu_vendor=None):
    ns = types.SimpleNamespace(smoke=False, deadline_h=args.deadline_h, resume=args.resume, stem_prefix="t2s_night2",
                               only=None, reserve_min=20, no_cap_arm=True, max_items=0, gpu_vendor=gpu_vendor)
    lab = ov.Lab(ns, prov)
    return lab


def load_models(lab, wanted):
    """Skips (logs, does not raise) a wanted model whose hash isn't in downloads.jsonl yet -- e.g. the 70B while its
    background download/verify is still in progress. Every phase already checks `lab.models.get(mid) is None` and
    skips that model, so a run launched before a large download finishes still executes its other models instead of
    crashing at startup."""
    ov.read_downloads(lab)
    for mid in wanted:
        fn, hyb, mx, yf = ov.MODEL_FILES[mid]
        sha = ov.KNOWN_4B_SHA if mid == "qwen3-4b-2507" else lab.dl_sha.get(fn)
        if sha is None:
            log(f"load_models: {mid} ({fn}) not yet in downloads.jsonl, skipping for this run")
            continue
        lab.models[mid] = L.ModelInfo(mid, str(Path(L.MODELS_DIR) / fn), sha, hyb, mx, yf)


def measured_with_extra(lab, srv, mi, section, item_id, prompt, n_tok, base_extra, co_runner, n_calls=5, slow_s=90.0, max_tokens=128):
    """Per-core frequency, RAPL PP0/PP1 and iGPU throttle bits are already in every do_call row (t2s_lab.py Telemetry
    streams and windows them the same way as igpu_mhz/pkg_power_w); base_extra only needs the condition labels."""
    w = ov.do_call(lab, srv, mi, section, item_id, prompt, n_tok, warmup=True, rep=-1, extra=base_extra, max_tokens=max_tokens, co_runner=co_runner)
    if w is None or w.get("outcome") != "ok":
        return []
    slow = (w.get("e2e_s") or 0) > slow_s
    out = []
    for i in range(3 if slow else n_calls):
        r = ov.do_call(lab, srv, mi, section, item_id, prompt, n_tok, warmup=False, rep=i, extra=dict(base_extra, n_reduced=slow),
                        max_tokens=max_tokens, co_runner=co_runner)
        if r is None or r.get("outcome") != "ok":
            break
        out.append(r)
    return out


def positive_control_throttle(lab, tag, calls):
    """Uses the igpu_mhz and igpu_throttle_bits already measured DURING the condition's own calls (do_call windows
    Telemetry.metrics over [t_start, t_end+1s]), not a fresh live Sysman sample. A live sample taken after
    measured_with_extra returns is not a valid control: on 2026-09-28 it read 2500 MHz and throttle_reasons 0 for
    qwen3-8b all16 only 1.5 s after that condition's last call ended at 1650 MHz -- the GPU had already boosted back
    to idle in that gap. See docs/RESULT_PROVENANCE.md."""
    mhz = [r.get("igpu_mhz") for r in calls if r and r.get("igpu_mhz") is not None]
    bits = set()
    for r in calls:
        for b in (r.get("igpu_throttle_bits") or []):
            bits.add(b)
    rec = {"record": "b1_positive_control", "tag": tag, "ts_utc": utc_iso(), "source": "median over the condition's own call rows"}
    if not mhz:
        rec.update({"available": False, "note": "no igpu_mhz on any call row for this condition"})
    else:
        med = st.median(mhz)
        rec.update({"available": True, "actual_mhz": med, "throttle_reasons": sorted(bits),
                    "pass": bool(med <= 1700 and bits)})
    lab.emit(rec)
    log(f"positive control ({tag}): {rec}")


# ---------------------------------------------------------------- B1
def phase_b1(lab):
    for mid in B1_MODELS:
        mi = lab.models.get(mid)
        if mi is None:
            continue
        item0 = f"B1_{mid}_start"
        srv = L.Server(lab, mi, CTX_B, tag=item0)
        lab.resources["server"] = srv
        info = srv.start()
        ov.start_row(lab, srv, mi, "B1", item0, info, {})
        if not info.get("ok"):
            srv.stop()
            lab.resources["server"] = None
            continue
        prompt = ov.prompt_for(srv, FILL_B)
        n_tok = srv.tokenize(prompt)
        for co, mask in B1_MASKS:
            item = f"B1_{mid}_{co}"
            if item in lab.done:
                continue
            lab.check()
            hog = None
            if mask is not None:
                hog, report, aff = L.m3.start_hog(f"night2_{item}", mask, Path(lab.prefix).parent, Path(lab.prefix).name + f"_{item}")
                time.sleep(5)
            calls = measured_with_extra(lab, srv, mi, "B1", item, prompt, n_tok, {"cpu_mask": hex(mask) if mask else None}, co)
            if co == "all16":
                positive_control_throttle(lab, item, calls)
            if hog is not None:
                L.m3.kill_tree(hog.pid)
                time.sleep(3)
            lab.item_done(item)
        srv.stop()
        lab.resources["server"] = None


# ---------------------------------------------------------------- B4 (mechanism test: which E-core grouping matters)
def _phase_b4(lab, models, label="B4"):
    for mid in models:
        mi = lab.models.get(mid)
        if mi is None:
            continue
        item0 = f"{label}_{mid}_start"
        srv = L.Server(lab, mi, CTX_B, tag=item0)
        lab.resources["server"] = srv
        info = srv.start()
        ov.start_row(lab, srv, mi, label, item0, info, {})
        if not info.get("ok"):
            srv.stop()
            lab.resources["server"] = None
            continue
        prompt = ov.prompt_for(srv, FILL_B)
        n_tok = srv.tokenize(prompt)
        for co, mask in B4_MASKS:
            item = f"{label}_{mid}_{co}"
            if item in lab.done:
                continue
            lab.check()
            hog = None
            if mask is not None:
                hog, report, aff = L.m3.start_hog(f"night3_{item}", mask, Path(lab.prefix).parent, Path(lab.prefix).name + f"_{item}")
                time.sleep(5)
            measured_with_extra(lab, srv, mi, label, item, prompt, n_tok, {"cpu_mask": hex(mask) if mask else None}, co)
            if hog is not None:
                L.m3.kill_tree(hog.pid)
                time.sleep(3)
            lab.item_done(item)
        srv.stop()
        lab.resources["server"] = None


def phase_b4(lab):
    _phase_b4(lab, ["qwen3-8b"], "B4")


def phase_b4_32b(lab):
    """Backlog: the same B4 mechanism test on qwen3-32b, runs only if time remains."""
    _phase_b4(lab, ["qwen3-32b"], "B4b")


def phase_b4_replicate(lab):
    """Standing-backlog default: an independent second run of the B4 mechanism test (qwen3-8b), same conditions,
    fresh server start and fresh item-id namespace ("B4r"). Queued behind night3 so the machine has a real next run
    once night3 finishes rather than sitting idle -- B4 is a brand-new test with only one run in night3, so a
    same-machine reproducibility replicate is the most directly justified backlog item, not a new experimental
    design. A different backlog should replace this in the queue once the user picks one."""
    _phase_b4(lab, ["qwen3-8b"], "B4r")


# ---------------------------------------------------------------- B2
def phase_b2(lab):
    for mid in B1_MODELS:
        mi = lab.models.get(mid)
        if mi is None:
            continue
        item0 = f"B2_{mid}_start"
        srv = L.Server(lab, mi, CTX_B, tag=item0)
        lab.resources["server"] = srv
        info = srv.start()
        ov.start_row(lab, srv, mi, "B2", item0, info, {})
        if not info.get("ok"):
            srv.stop()
            lab.resources["server"] = None
            continue
        prompt = ov.prompt_for(srv, FILL_B)
        n_tok = srv.tokenize(prompt)
        for duty in B2_DUTY:
            item = f"B2_{mid}_duty{duty}"
            if item in lab.done:
                continue
            lab.check()
            hog, report, aff = L.m3.start_hog(f"night2_{item}", 0xFFF0, Path(lab.prefix).parent, Path(lab.prefix).name + f"_{item}", duty_pct=float(duty))
            time.sleep(5)
            measured_with_extra(lab, srv, mi, "B2", item, prompt, n_tok, {"cpu_mask": "0xFFF0", "duty_cycle_pct": duty}, "nonp12")
            L.m3.kill_tree(hog.pid)
            time.sleep(3)
            lab.item_done(item)
        srv.stop()
        lab.resources["server"] = None


# ---------------------------------------------------------------- B3
def phase_b3(lab):
    for mid in B3_MODELS:
        mi = lab.models.get(mid)
        if mi is None:
            continue
        item0 = f"B3_{mid}_start"
        srv = L.Server(lab, mi, CTX_B, tag=item0)
        lab.resources["server"] = srv
        info = srv.start(timeout=1800)
        ov.start_row(lab, srv, mi, "B3", item0, info, {})
        if not info.get("ok"):
            srv.stop()
            lab.resources["server"] = None
            continue
        prompt = ov.prompt_for(srv, FILL_B)
        n_tok = srv.tokenize(prompt)
        for co, mask in B3_CORESETS:
            item = f"B3_{mid}_{co}"
            if item in lab.done:
                continue
            lab.check()
            hog = None
            if mask is not None:
                hog, report, aff = L.m3.start_hog(f"night2_{item}", mask, Path(lab.prefix).parent, Path(lab.prefix).name + f"_{item}")
                time.sleep(5)
            measured_with_extra(lab, srv, mi, "B3", item, prompt, n_tok, {"cpu_mask": hex(mask) if mask else None}, co)
            if hog is not None:
                L.m3.kill_tree(hog.pid)
                time.sleep(3)
            lab.item_done(item)
        srv.stop()
        lab.resources["server"] = None


# ---------------------------------------------------------------- R1 speed sweep
def phase_r1_speed(lab):
    """R1 speed side (Addendum v3): TTFT/decode at 1k/2k/4k/8k/16k prompt tokens, 1 warm-up + 3 measured calls each,
    no co-runner, tagged axis="latency". Skips a model load_models() didn't populate (e.g. the 70B before its
    download/verify lands)."""
    for mid in R1_SPEED_MODELS:
        mi = lab.models.get(mid)
        if mi is None:
            continue
        item0 = f"R1speed_{mid}_start"
        srv = L.Server(lab, mi, R1_SPEED_CTX, tag=item0)
        lab.resources["server"] = srv
        info = srv.start(timeout=1800)
        ov.start_row(lab, srv, mi, "R1speed", item0, info, {"axis": "latency"})
        if not info.get("ok"):
            srv.stop()
            lab.resources["server"] = None
            continue
        for tgt in R1_SPEED_LENGTHS:
            item = f"R1speed_{mid}_{tgt}"
            if item in lab.done:
                continue
            lab.check()
            prompt = ov.prompt_for(srv, tgt)
            n_tok = srv.tokenize(prompt)
            ov.measured_sequence(lab, srv, mi, "R1speed", item, prompt, n_tok,
                                 extra={"axis": "latency", "target_prompt_tokens": tgt}, n_calls=3)
            lab.item_done(item)
        srv.stop()
        lab.resources["server"] = None


# ---------------------------------------------------------------- R1 check set (Fig 6.1 quality side)
def load_position_pressure_probes(probe_ids):
    """Read-only load of the named probes from evaluation/probes/segments.jsonl -- never writes to that directory,
    per the standing rule."""
    segs = {}
    for l in (L.PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l)
            segs[d["id"]] = d
    missing = [pid for pid in probe_ids if pid not in segs]
    if missing:
        raise RuntimeError(f"position-pressure probes missing from segments.jsonl: {missing}")
    return [segs[pid] for pid in probe_ids]


def load_r1_check_probes():
    return load_position_pressure_probes(R1_CHECK_PROBE_IDS)


def _r1_arm_prompt(arm, filler, artifact, question):
    if arm == "LATE":
        return f"{filler}\n\n{artifact}\n\n{question}"
    return f"{artifact}\n\n{filler}\n\n{question}"


def _phase_position_pressure(lab, models, probe_ids, section):
    """Shared implementation for phase_r1_check (R1a, 5-probe Fig 6.1 subset) and phase_r1d (R1d, all 11 rag_*/sea_*
    probes): budget ratios x 2 position arms (LATE/EARLY) x 3 reps, ported from harness/fig61_stagec_sweep.py and
    harness/stage_c_position_pressure.py's raw-HTTP implementations onto the current L.Server/do_call machinery
    (stale-server guard, thermal gate) instead of talking to a hardcoded port directly. Same truncation function
    (stage_c_position_pressure.left_truncate), same filler (4000 tokens, F-NUM, seed=42), same budget-ratio/arm/
    positive-control definitions as both original scripts (which share these exact constants). Every row tagged
    axis="quality"."""
    probes = load_position_pressure_probes(probe_ids)
    for mid in models:
        mi = lab.models.get(mid)
        if mi is None:
            continue
        item0 = f"{section}_{mid}_start"
        srv = L.Server(lab, mi, R1_CHECK_CTX, tag=item0)
        lab.resources["server"] = srv
        info = srv.start(timeout=1800)
        ov.start_row(lab, srv, mi, section, item0, info, {"axis": "quality"})
        if not info.get("ok"):
            srv.stop()
            lab.resources["server"] = None
            continue
        filler = L.ctx_mod.build_filler(R1_CHECK_FILLER, seed=42, count_fn=srv.tokenize)
        full_tokens_cache = {p["id"]: srv.tokenize(_r1_arm_prompt("LATE", filler, p["artifact"].strip(), p["question"].strip()))
                             for p in probes}
        for ratio in R1_CHECK_RATIOS:
            for probe in probes:
                pid = probe["id"]
                artifact, question = probe["artifact"].strip(), probe["question"].strip()
                full_tokens = full_tokens_cache[pid]
                target_tokens = round(full_tokens * ratio)
                truncating = ratio < 1.0
                intended = min(target_tokens, full_tokens)
                for arm in R1_CHECK_ARMS:
                    for rep in range(R1_CHECK_REPS):
                        item = f"{section}_{mid}_{pid}_{ratio}_{arm}_{rep}"
                        if item in lab.done:
                            continue
                        lab.check()
                        full_prompt = _r1_arm_prompt(arm, filler, artifact, question)
                        prompt = sc_left_truncate(full_prompt, full_tokens, target_tokens) if truncating else full_prompt
                        n_tok = srv.tokenize(prompt)
                        chars_dropped = len(full_prompt) - len(prompt)
                        art_len = len(artifact)
                        if arm == "EARLY" and truncating and art_len > 0:
                            art_frac = max(0.0, 1.0 - min(chars_dropped, art_len) / art_len)
                        else:
                            art_frac = 1.0
                        pc_ok = abs(n_tok - intended) / max(intended, 1) <= 0.05
                        probe_dict = {"id": pid, "scorer_type": probe["scorer_type"], "expected": probe["expected"]}
                        ov.do_call(lab, srv, mi, section, item, prompt, n_tok, warmup=False, rep=rep, max_tokens=128,
                                   ignore_eos=False, kind="call", probe=probe_dict,
                                   extra={"axis": "quality", "budget_ratio": ratio, "arm": arm, "full_tokens": full_tokens,
                                          "target_tokens": target_tokens, "truncating": truncating,
                                          "chars_dropped": chars_dropped, "artifact_fraction_retained": round(art_frac, 4),
                                          "intended_budget_tokens": intended, "positive_control_ok": pc_ok})
                        lab.item_done(item)
        srv.stop()
        lab.resources["server"] = None


def phase_r1_check(lab):
    """R1a quality-side check set (Addendum v3): 5 probes x 6 budget ratios x 2 position arms x 3 reps = 180 calls
    per model. See _phase_position_pressure for the shared implementation."""
    _phase_position_pressure(lab, R1_CHECK_MODELS, R1_CHECK_PROBE_IDS, "R1check")


def phase_r1d(lab):
    """R1d position pressure (Addendum v3), the full stage_c_position_pressure.py spec: 11 probes x 6 budget ratios
    x 2 position arms x 3 reps = 396 calls per model. See _phase_position_pressure for the shared implementation."""
    _phase_position_pressure(lab, R1D_MODELS, R1D_PROBE_IDS, "R1d")


# ---------------------------------------------------------------- R1b (truncation cliff)
# Ported from harness/art_truncation.py (Ollama-based) onto the current L.Server/do_call machinery, same reasoning as
# phase_r1_check. Probe IDs come from evaluation/probes/artifact.jsonl (art_01..art_10); each probe's actual
# artifact/question/scorer_type/expected text is the matching art_* entry in segments.jsonl (art_truncation.py itself
# reads segments.jsonl for those fields, artifact.jsonl only for the id list and artifact_form), so this reuses
# load_r1_check_probes'-style read-only loading against the same file, just a different id list.
R1B_MODELS = ["qwen3-8b", "qwen3-14b"]
R1B_PROBE_IDS = [f"art_{i:02d}" for i in range(1, 11)]
R1B_RATIOS = [1.20, 0.98, 0.85, 0.40]
R1B_ARM_SUFFIXES = {
    "arm1_baseline": "",
    "arm3_self_report": ("\n\nFirst state whether the information needed to answer is present above, then answer. "
                         "Format: AVAILABLE: yes|no, then the answer."),
}
R1B_REPS = 3
R1B_FILLER = 4000
R1B_CTX = 8192


def load_r1b_probes():
    segs = {}
    for l in (L.PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l)
            segs[d["id"]] = d
    missing = [pid for pid in R1B_PROBE_IDS if pid not in segs]
    if missing:
        raise RuntimeError(f"R1b probes missing from segments.jsonl: {missing}")
    return [segs[pid] for pid in R1B_PROBE_IDS]


def phase_r1b(lab):
    """R1b truncation cliff (Addendum v3): 10 art_* probes x 4 budget ratios x 2 arms (baseline vs a self-report
    instruction appended after truncation) x 3 reps = 240 calls/model. Artifact always precedes filler precedes
    question (no LATE/EARLY position variation, unlike R1a/phase_r1_check) -- position is R1d's axis, not this one.
    Every row tagged axis="quality"."""
    probes = load_r1b_probes()
    for mid in R1B_MODELS:
        mi = lab.models.get(mid)
        if mi is None:
            continue
        item0 = f"R1b_{mid}_start"
        srv = L.Server(lab, mi, R1B_CTX, tag=item0)
        lab.resources["server"] = srv
        info = srv.start(timeout=1800)
        ov.start_row(lab, srv, mi, "R1b", item0, info, {"axis": "quality"})
        if not info.get("ok"):
            srv.stop()
            lab.resources["server"] = None
            continue
        filler = L.ctx_mod.build_filler(R1B_FILLER, seed=42, count_fn=srv.tokenize)
        full_tokens_cache = {p["id"]: srv.tokenize(f"{p['artifact'].strip()}\n\n{filler}\n\n{p['question'].strip()}")
                             for p in probes}
        for ratio in R1B_RATIOS:
            for probe in probes:
                pid = probe["id"]
                artifact, question = probe["artifact"].strip(), probe["question"].strip()
                full_tokens = full_tokens_cache[pid]
                target_tokens = round(full_tokens * ratio)
                truncating = target_tokens < full_tokens
                base_prompt = f"{artifact}\n\n{filler}\n\n{question}"
                truncated_base = sc_left_truncate(base_prompt, full_tokens, target_tokens) if truncating else base_prompt
                chars_dropped = len(base_prompt) - len(truncated_base)
                art_len = len(artifact)
                art_frac = max(0.0, 1.0 - min(chars_dropped, art_len) / art_len) if art_len else 0.0
                for arm, suffix in R1B_ARM_SUFFIXES.items():
                    full_prompt = truncated_base + suffix
                    n_tok = srv.tokenize(full_prompt)
                    for rep in range(R1B_REPS):
                        item = f"R1b_{mid}_{pid}_{ratio}_{arm}_{rep}"
                        if item in lab.done:
                            continue
                        lab.check()
                        probe_dict = {"id": pid, "scorer_type": probe["scorer_type"], "expected": probe["expected"]}
                        ov.do_call(lab, srv, mi, "R1b", item, full_prompt, n_tok, warmup=False, rep=rep, max_tokens=256,
                                   ignore_eos=False, kind="call", probe=probe_dict,
                                   extra={"axis": "quality", "arm": arm, "budget_ratio": ratio, "full_tokens": full_tokens,
                                          "target_tokens": target_tokens, "truncating": truncating,
                                          "chars_dropped": chars_dropped, "artifact_fraction_retained": round(art_frac, 4)})
                        lab.item_done(item)
        srv.stop()
        lab.resources["server"] = None


# ---------------------------------------------------------------- R1c (truncation cliff under position pressure)
# R1c is the one cell of the R1 quality design that the R1a/R1b/R1d set leaves empty. R1b runs the art_* truncation-
# cliff probes with the artifact always FIRST (artifact, filler, question) and sc_left_truncate drops characters from
# the left, so every truncating R1b cell eats the artifact itself: in R1a/R1d's vocabulary R1b measured the EARLY arm
# only, and its cliff therefore cannot be separated from "left truncation removes whatever happens to sit at the
# left". R1a/R1d do vary position (LATE/EARLY) but only over the rag_*/sea_* probes. R1c closes that cross term by
# running R1b's own 10 art_* probes through the position-pressure machinery unchanged, so at the same budget ratio the
# LATE arm (filler, artifact, question) keeps the artifact past the truncation point while EARLY still loses it. A
# cliff that survives the LATE arm is a budget effect; one that appears only in EARLY is an artifact-position effect.
# The EARLY arm at ratios 1.20/0.85/0.40 also reproduces R1b's own arm1_baseline cells at those ratios, which is a
# free cross-phase replication check. The art_* artifacts are 272-736 chars against a 4000-token filler, so the two
# arms really do differ in whether the artifact is inside the dropped span.
R1C_MODELS = ["qwen3-8b", "qwen3-14b"]
R1C_PROBE_IDS = R1B_PROBE_IDS


def phase_r1c(lab):
    """R1c truncation cliff under position pressure (Addendum v3): 10 art_* probes x 6 budget ratios x 2 position
    arms x 3 reps = 360 calls per model. A thin wrapper over _phase_position_pressure with R1b's probe id list and
    nothing else changed, exactly the way phase_r1_check (R1a) and phase_r1d already share that function."""
    _phase_position_pressure(lab, R1C_MODELS, R1C_PROBE_IDS, "R1c")


# ---------------------------------------------------------------- A70 (70B budget crossing, no YaRN)
A70_MODEL = "llama-3.3-70b"
A70_MATCHED_PROMPT_TOKENS = 8000
A70_ARMS = [("ngl99", 99, None), ("default_fit", None, None), ("fit_off", None, "off")]


def _a70_kv_step_tokens(mi, target_mib=512.0):
    """Tokens per ~target_mib of KV, rounded to a multiple of am.STEP (the server pads n_ctx to a multiple of 256
    anyway, so a step finer than that is not meaningful)."""
    kv_mib_per_step = mi.kv_bpt_meta * am.STEP / 2 ** 20
    if not kv_mib_per_step:
        return am.STEP
    mult = max(1, round(target_mib / kv_mib_per_step))
    return mult * am.STEP


def _a70_repeated_loads(lab, mi, n_ctx, label, n_reps=3):
    """Independent repeated loads at a single n_ctx, bypassing Prober.probe()'s cache (which would otherwise return
    the same cached result for every repeat, defeating the point -- the finalization needs 3 genuinely independent
    server starts per point to check for intermittency, not 3 reads of one cached outcome). Same arm/flags as the
    bisection itself (default fit, ngl=99, no rope flags, matching phase_a70's own Prober(lab, mi, [], "a70"))."""
    results = []
    for i in range(n_reps):
        lab.check()
        n_ctx_r = int(round(n_ctx / am.STEP)) * am.STEP
        srv, info, px = am.start_and_record(lab, mi, n_ctx_r, f"a70fin_{label}_{n_ctx_r}_{i}",
                                            f"A70_finalize_{label}_{n_ctx_r}_{i}", "a70_finalize", [])
        srv.stop()
        lab.resources["server"] = None
        lg = info.get("log", {})
        logged = sum(x for x in (lg.get("model_buffer_mib"), lg.get("kv_buffer_mib"), lg.get("compute_buffer_mib")) if x)
        results.append({"ok": bool(info.get("ok")), "error": info.get("error"), "logged_mib": logged or None,
                        "vk": px.get("vk_errors"), "rep": i})
    return results


def phase_a70(lab):
    """Budget crossing on the 70B at native context (no YaRN -- flags=[]), reusing t2s_amech.Prober/boundary_record/
    start_and_record directly (same ov.Lab shape, no reimplementation). Finds n_ctx* where weights+KV+compute cross
    am.BUDGET_MIB (47865 MiB), bisected to 256 tokens by Prober.find() itself; records 3 points below and 3 above at
    roughly 512 MiB of KV apart. Above the boundary, tries three fit arms (-ngl 99, default fit, -fit off); where a
    server starts, runs 1 warm-up + 3 calls with a matched 8000-token prompt plus the Q0 suite at 8k (if
    harness/quality_suite.py is importable; skipped with a note otherwise). Records exit codes and crash log tails
    for every arm, started or not."""
    mi = lab.models.get(A70_MODEL)
    if mi is None:
        log("A70: llama-3.3-70b not loaded (downloads.jsonl has no verified entry yet), skipping")
        return
    pr = am.Prober(lab, mi, [], "a70")
    lo, hi = pr.find(4096, 1024)
    if lo is None:
        lab.emit({"record": "a70_no_passing_context", "model_id": mi.model_id, "ts_utc": utc_iso()})
        return
    am.boundary_record(lab, mi, "a70", [], pr, lo, hi, target_note="native context, no YaRN, budget 47865 MiB")
    step = _a70_kv_step_tokens(mi)
    for i in range(1, 4):
        pr.probe(max(am.STEP, lo - i * step))
    for i in range(1, 4):
        pr.probe(hi + (i - 1) * step)

    # Finalization (2026-09-29 contamination check, corrected 2026-09-29 to 3 independent loads per side rather
    # than 1: the boundary above is reported as confirmed only if lo is 3/3 success and hi is 3/3 failure, with
    # Ollama confirmed absent before each load. This does not change lo/hi or the reported boundary value; it only
    # adds a reproduced/not-reproduced flag the report must check before trusting the boundary.
    _abort_if_ollama_running("a70_finalization")
    lo_loads = _a70_repeated_loads(lab, mi, lo, "lo")
    _abort_if_ollama_running("a70_finalization")
    hi_loads = _a70_repeated_loads(lab, mi, hi, "hi")
    lo_ok_count = sum(1 for r in lo_loads if r["ok"])
    hi_fail_count = sum(1 for r in hi_loads if not r["ok"])
    reproduced = lo_ok_count == 3 and hi_fail_count == 3
    lab.emit({"record": "a70_boundary_finalization", "model_id": mi.model_id, "lo": lo, "hi": hi,
              "lo_loads": lo_loads, "hi_loads": hi_loads, "lo_ok_count": lo_ok_count, "hi_fail_count": hi_fail_count,
              "reproduced": reproduced, "ollama_confirmed_absent": True, "ts_utc": utc_iso()})
    if not reproduced:
        log(f"A70 finalization: boundary lo={lo} hi={hi} did NOT reproduce 3/3 (lo_ok={lo_ok_count}/3, "
            f"hi_fail={hi_fail_count}/3); the boundary must be reported as unconfirmed, not as a result")

    for label, ngl, fit in A70_ARMS:
        item = f"A70_{mi.model_id}_{label}"
        if item in lab.done:
            continue
        lab.check()
        srv, info, px = am.start_and_record(lab, mi, hi, f"a70_{label}", item, "a70_arms", [], ngl=ngl, fit=fit)
        try:
            if not info.get("ok"):
                lg = info.get("log", {})
                lab.emit({"record": "a70_arm_result", "arm": label, "model_id": mi.model_id, "started": False,
                          "exit_code": info.get("exit_code"), "error": info.get("error"),
                          "error_lines": px.get("error_lines"), "vk_errors": px.get("vk_errors"),
                          "alloc_failed": px.get("alloc_failed"), "ts_utc": utc_iso()})
                lab.item_done(item)
                continue
            prompt = ov.prompt_for(srv, A70_MATCHED_PROMPT_TOKENS)
            n_tok = srv.tokenize(prompt)
            calls = ov.measured_sequence(lab, srv, mi, "A70", item, prompt, n_tok, extra={"arm": label}, n_calls=3)
            q0_rows = None
            if qs is not None and hasattr(qs, "run_suite"):
                try:
                    q0_rows = qs.run_suite(srv, A70_MATCHED_PROMPT_TOKENS)
                    for r in q0_rows:
                        lab.emit({"record": "a70_q0", "arm": label, "model_id": mi.model_id, **r, "ts_utc": utc_iso()})
                except Exception as e:
                    lab.emit({"record": "a70_q0_error", "arm": label, "error": repr(e)[:400], "ts_utc": utc_iso()})
            lab.emit({"record": "a70_arm_result", "arm": label, "model_id": mi.model_id, "started": True,
                      "n_calls_ok": len(calls), "q0_n": len(q0_rows) if q0_rows is not None else None,
                      "ts_utc": utc_iso()})
        finally:
            srv.stop()
            lab.resources["server"] = None
        lab.item_done(item)


def phase_a70_finalize(lab):
    """Standalone finalization phase for an A70 run that already completed under the OLD phase_a70 code, before the
    inline finalization step (2026-09-29 contamination check) existed. Reads the already-recorded boundary for
    A70_MODEL (this run's own 'bisect_result' row, label 'a70') instead of re-deriving it with a fresh, expensive
    bisection, then reruns 3 independent loads at lo and 3 at hi with Ollama confirmed absent, matching phase_a70's
    own inline finalization (corrected 2026-09-29 from 1 reprobe per side to 3: the boundary is confirmed only if
    lo is 3/3 success and hi is 3/3 failure). Queue this for a run whose a70 phase finished before this function
    existed, or whose earlier finalization record used the old 1-reprobe-per-side shape (no 'lo_ok_count' key)."""
    mi = lab.models.get(A70_MODEL)
    if mi is None:
        log("A70 finalize: llama-3.3-70b not loaded, skipping")
        return
    if any(r.get("record") == "a70_boundary_finalization" and r.get("model_id") == mi.model_id
           and "lo_ok_count" in r for r in lab.all_rows()):
        log("A70 finalize: already have a 3-load-per-side finalization record for this model, skipping")
        return
    boundary_rows = [r for r in lab.all_rows() if r.get("record") == "bisect_result" and r.get("label") == "a70"
                      and r.get("model_id") == mi.model_id]
    if not boundary_rows:
        log("A70 finalize: no bisect_result row found for a70, cannot finalize")
        return
    b = boundary_rows[-1]
    lo, hi = b.get("last_ok_n_ctx"), b.get("first_fail_n_ctx")
    if lo is None or hi is None:
        log(f"A70 finalize: bisect_result row missing last_ok_n_ctx/first_fail_n_ctx: {b}")
        return
    _abort_if_ollama_running("a70_finalize")
    lo_loads = _a70_repeated_loads(lab, mi, lo, "lo")
    _abort_if_ollama_running("a70_finalize")
    hi_loads = _a70_repeated_loads(lab, mi, hi, "hi")
    lo_ok_count = sum(1 for r in lo_loads if r["ok"])
    hi_fail_count = sum(1 for r in hi_loads if not r["ok"])
    reproduced = lo_ok_count == 3 and hi_fail_count == 3
    lab.emit({"record": "a70_boundary_finalization", "model_id": mi.model_id, "lo": lo, "hi": hi,
              "lo_loads": lo_loads, "hi_loads": hi_loads, "lo_ok_count": lo_ok_count, "hi_fail_count": hi_fail_count,
              "reproduced": reproduced, "ollama_confirmed_absent": True, "standalone": True, "ts_utc": utc_iso()})
    if not reproduced:
        log(f"A70 finalize: boundary lo={lo} hi={hi} did NOT reproduce 3/3 (lo_ok={lo_ok_count}/3, "
            f"hi_fail={hi_fail_count}/3); the boundary must be reported as unconfirmed, not as a result")


# ---------------------------------------------------------------- P70 (70B latency axis, none vs nonp12)
def phase_p70(lab):
    """None vs nonp12 co-runner on the 70B at context 8192, 1 warm-up + 3 calls each, fixed 20s co-runner settle
    (thermal_gate's corunner_active path) -- the B3 mechanism, reused for a single model/coreset pair rather than
    B3's full model x coreset cross."""
    mi = lab.models.get(A70_MODEL)
    if mi is None:
        log("P70: llama-3.3-70b not loaded (downloads.jsonl has no verified entry yet), skipping")
        return
    item0 = f"P70_{mi.model_id}_start"
    srv = L.Server(lab, mi, CTX_B, tag=item0)
    lab.resources["server"] = srv
    info = srv.start(timeout=1800)
    ov.start_row(lab, srv, mi, "P70", item0, info, {})
    if not info.get("ok"):
        srv.stop()
        lab.resources["server"] = None
        return
    prompt = ov.prompt_for(srv, FILL_B)
    n_tok = srv.tokenize(prompt)
    for co, mask in (("none", None), ("nonp12", 0xFFF0)):
        item = f"P70_{mi.model_id}_{co}"
        if item in lab.done:
            continue
        lab.check()
        hog = None
        if mask is not None:
            hog, report, aff = L.m3.start_hog(f"night3_{item}", mask, Path(lab.prefix).parent, Path(lab.prefix).name + f"_{item}")
            time.sleep(5)
        measured_with_extra(lab, srv, mi, "P70", item, prompt, n_tok, {"cpu_mask": hex(mask) if mask else None}, co, n_calls=3)
        if hog is not None:
            L.m3.kill_tree(hog.pid)
            time.sleep(3)
        lab.item_done(item)
    srv.stop()
    lab.resources["server"] = None


# ---------------------------------------------------------------- MX2 (evo-x2 two-line memory budget, unified pool)
# evo-x2 is a unified-memory APU: the whole Vulkan-visible pool is the same physical RAM, but Vulkan still reports it
# as several heaps, and only some of them carry DEVICE_LOCAL. That gives two budget lines rather than evo-t2s's one:
#   line 1 (device_local)  the sum of the DEVICE_LOCAL heaps. Past this line an allocation still SUCCEEDS -- the driver
#                          satisfies it out of the host-visible portion of the pool -- so the server starts and answers
#                          correctly, and the only symptom is time. That regime is SILENT_SPILL: it cannot be detected
#                          from the exit code, which is exactly why it needs its own bisected line.
#   line 2 (vulkan_total)  the sum of every heap. Past this line the allocation cannot be satisfied at all and the
#                          server either refuses/crashes (HARD_FAIL) or never becomes healthy (HANG).
# Both lines are MEASURED at run time from vulkaninfo in a single accounting view, deliberately not derived from the
# D3D Dedicated/Shared split: docs/X2_CHANGELOG.md (2026-09-29) records that Vulkan's device-local/host-visible split
# does not correspond 1:1 to D3D's Dedicated/Shared split on this machine, so mixing the two views would produce two
# lines that are not comparable to each other.
MX2_MODELS = ["llama-3.3-70b", "qwen3-32b"]
MX2_MATCHED_PROMPT_TOKENS = 8000
MX2_START_CTX = 8192
MX2_START_STEP = 4096
MX2_SPILL_MARGIN_MIB = 512.0  # one probe step's worth of slack before a logged total counts as over a line
MX2_ARM_CALLS = 3
# Planning figure only (estimate_hours), not a cap on the search. Derived, not guessed: the 70B's KV is 0.3125 MiB per
# token (80 layers x 8 KV heads x 128 x 2 x f16), so inside a ~76 GiB device_local line its ~40.5 GiB of weights leave
# room for roughly 114k tokens, and inside the ~114 GiB total line for roughly 236k. From MX2_START_CTX with
# MX2_START_STEP doubling, that is ~6 exponential probes + 8 bisection probes for line 1, then ~5 + 8 for line 2 with
# the overlapping ones served from the shared cache: ~26 server starts per model. The 32B works out about the same.
MX2_N_PROBES_PER_MODEL = 26
# Used only if vulkaninfo cannot be read at run time, and then recorded as fallback_lines with lines_measured=False so
# no analysis mistakes them for a measurement. Values from docs/X2_CHANGELOG.md's 2026-09-29 post-reboot heap dump
# (DEVICE_LOCAL heap 74.43 GiB; all heaps 111.65 GiB, matching llama-server --list-devices' 114326 MiB total).
MX2_FALLBACK_LINES = {"device_local_mib": 74.43 * 1024, "vulkan_total_mib": 114326.0}
MX2_HEAPS_RE = (r"memoryHeaps\[(\d+)\]:\s*\n?\s*size\s*=\s*(\d+)\s*\((0x[0-9a-f]+)\)[^\n]*\n\s*budget\s*=\s*(\d+)"
                r"[^\n]*\n\s*usage\s*=\s*(\d+)[^\n]*\n\s*flags:\s*\n?([^\n]*(?:\n\s+MEMORY_HEAP[^\n]*)*)")
# MX2 measures WHERE the memory lands and WHAT THAT COSTS IN TIME. Output correctness is deliberately out of scope:
# docs/FINDINGS.md already records that a silent spill moved KV into system RAM while the model returned the same
# answers, because spill changes where the KV lives and not the arithmetic. Running the Q0 suite here (as phase_a70
# does) would spend hours re-confirming a null result, so this phase records the scope decision instead.
MX2_QUALITY_SCOPE_NOTE = ("quality out of scope: spill changes where the KV cache lives, not the arithmetic "
                          "(docs/FINDINGS.md); this phase measures regime and time cost only, no Q0 suite")


def mx2_parse_heaps(txt):
    """vulkaninfo's memoryHeaps block -> [{index, size_mib, budget_mib, usage_mib, flags, device_local}]. Same regex
    shape as t2s_amech.phase_vk's own heap parse, so the two agree on what a heap is."""
    out = []
    for h in re.findall(MX2_HEAPS_RE, txt or ""):
        flags = " ".join(h[5].split())
        out.append({"index": int(h[0]), "size_mib": int(h[1]) / 2 ** 20, "budget_mib": int(h[3]) / 2 ** 20,
                    "usage_mib": int(h[4]) / 2 ** 20, "flags": flags, "device_local": "DEVICE_LOCAL" in flags})
    return out


def mx2_lines_from_heaps(heaps):
    """The two budget lines from one heap list. device_local_mib is None when no heap advertises DEVICE_LOCAL (rather
    than 0, which would read as a real line at zero and make every context look like a spill)."""
    if not heaps:
        return {"device_local_mib": None, "vulkan_total_mib": None}
    dl = [h for h in heaps if h["device_local"]]
    return {"device_local_mib": sum(h["size_mib"] for h in dl) if dl else None,
            "vulkan_total_mib": sum(h["size_mib"] for h in heaps)}


def mx2_read_vulkaninfo(run=subprocess.run):
    """Raw vulkaninfo text, or None. Injectable `run` so the whole baseline path is testable without a GPU."""
    try:
        p = run(["vulkaninfo"], capture_output=True, text=True, errors="replace", timeout=180)
        return p.stdout
    except Exception as e:
        log(f"MX2 baseline: vulkaninfo failed: {e!r}")
        return None


def mx2_read_list_devices(run=subprocess.run):
    """llama-server --list-devices' own view: {"total_mib", "free_mib", "device", "raw"}. A cross-check on the heap
    sum from a second, independent reader -- not the source of either line."""
    try:
        p = run([L.BINARIES["vulkan"], "--list-devices"], capture_output=True, text=True,
                errors="replace", timeout=180)
        txt = (p.stdout or "") + "\n" + (p.stderr or "")
    except Exception as e:
        log(f"MX2 baseline: --list-devices failed: {e!r}")
        return None
    m = re.search(r"(\S+):\s*.*?(\d+)\s*MiB,\s*(\d+)\s*MiB free", txt)
    return {"device": m.group(1) if m else None, "total_mib": float(m.group(2)) if m else None,
            "free_mib": float(m.group(3)) if m else None, "raw": txt.strip()[:1200]}


def mx2_read_registry_dedicated(ps_fn=ps):
    """HardwareInformation.qwMemorySize off the display adapter's Class key: the BIOS UMA reservation as the driver
    itself reports it. Recorded as provenance for the baseline, not used to compute a line (see the module comment on
    why the D3D/Vulkan accounting views are not mixed)."""
    cmd = (r"$k='HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000'; "
           r"try { 'qwMemorySize=' + (Get-ItemProperty -Path $k -ErrorAction Stop).'HardwareInformation.qwMemorySize' } "
           r"catch { 'qwMemorySize=unavailable' }")
    try:
        out = ps_fn(cmd, 30)
    except Exception as e:
        log(f"MX2 baseline: registry read failed: {e!r}")
        return None
    m = re.search(r"qwMemorySize=(\d+)", out or "")
    return {"qw_memory_size_bytes": int(m.group(1)) if m else None,
            "qw_memory_size_mib": int(m.group(1)) / 2 ** 20 if m else None, "raw": (out or "").strip()[:400]}


def mx2_record_baseline(lab, run=subprocess.run, ps_fn=ps):
    """Reads and records this machine's memory baseline, then returns the two budget lines. Emits one mx2_baseline
    row carrying the heaps verbatim, both independent cross-checks, and the standing caveat: docs/X2_CHANGELOG.md
    records that the Vulkan-visible pool grew about 50% at the 2026-09-29 reboot and that the growth is NOT fully
    explained, so a baseline taken now may be a baseline against a transient. Recorded per run rather than assumed,
    which is the whole reason this phase reads the numbers itself instead of importing them from docs/HARDWARE.md."""
    vk_txt = mx2_read_vulkaninfo(run=run)
    heaps = mx2_parse_heaps(vk_txt) if vk_txt else []
    lines = mx2_lines_from_heaps(heaps)
    measured = lines["device_local_mib"] is not None and lines["vulkan_total_mib"] is not None
    if not measured:
        lines = dict(MX2_FALLBACK_LINES)
    ld = mx2_read_list_devices(run=run)
    reg = mx2_read_registry_dedicated(ps_fn=ps_fn)
    heap_sum = sum(h["size_mib"] for h in heaps) if heaps else None
    ld_total = (ld or {}).get("total_mib")
    lab.emit({"record": "mx2_baseline", "heaps": heaps, "lines": lines, "lines_measured": measured,
              "fallback_lines": None if measured else dict(MX2_FALLBACK_LINES),
              "list_devices": ld, "registry_dedicated": reg,
              "heap_sum_vs_list_devices_mib": None if (heap_sum is None or ld_total is None) else heap_sum - ld_total,
              "spill_margin_mib": MX2_SPILL_MARGIN_MIB, "quality_scope": MX2_QUALITY_SCOPE_NOTE,
              "git_sha": getattr(lab, "git_sha", None), "script_sha": getattr(lab, "script_sha", None),
              "baseline_caveat": ("the Vulkan-visible pool grew about 50 percent at the 2026-09-29 reboot and that "
                                  "growth is not fully explained (docs/X2_CHANGELOG.md); these lines are this run's "
                                  "own measurement and may be a transient, not the machine's settled configuration"),
              "ts_utc": utc_iso()})
    log(f"MX2 baseline: device_local={lines['device_local_mib']} MiB vulkan_total={lines['vulkan_total_mib']} MiB "
        f"measured={measured} list_devices_total={ld_total}")
    return lines


def mx2_classify(info, srv, px, lines):
    """One probe's regime, plus the evidence it was decided on.

      FITS          the server started and its own logged/projected memory stays inside the device_local line
      SILENT_SPILL  the server started, but that memory crossed the device_local line -- the driver backed the
                    allocation out of the host-visible part of the pool. No error, no bad exit code, correct answers:
                    detectable only by comparing the numbers, which is why this regime exists as a label at all
      HARD_FAIL     the server did not start and exited with a code (refusal or crash)
      HANG          the server did not start and never exited (am.classify_start's "hung")

    A benign YaRN /props n_ctx cap counts as a start, not a refusal -- am.classify_start already encodes that, and
    reusing it here keeps MX2 from re-making the bug that bit the A-mech map phase and the bisection twice."""
    outcome, props_cap = am.classify_start(info, srv)
    lg = info.get("log", {}) or {}
    logged = sum(x for x in (lg.get("model_buffer_mib"), lg.get("kv_buffer_mib"), lg.get("compute_buffer_mib")) if x)
    logged = logged or None
    projected = px.get("projected_mib")
    if outcome != "runs":
        regime = "HANG" if outcome == "hung" else "HARD_FAIL"
        return regime, props_cap, {"outcome": outcome, "logged_mib": logged, "projected_mib": projected,
                                   "spill_evidence": [], "exit_code": info.get("exit_code")}
    dl = lines.get("device_local_mib")
    evidence = []
    if dl is not None:
        if logged is not None and logged > dl - MX2_SPILL_MARGIN_MIB:
            evidence.append("logged_buffers_over_device_local_line")
        if projected is not None and projected > dl:
            evidence.append("llama_cpp_projection_over_device_local_line")
    regime = "SILENT_SPILL" if evidence else "FITS"
    return regime, props_cap, {"outcome": outcome, "logged_mib": logged, "projected_mib": projected,
                               "spill_evidence": evidence, "exit_code": info.get("exit_code"),
                               "device_local_line_mib": dl}


class Mx2HeapProber(am.Prober):
    """t2s_amech.Prober with a four-way regime instead of a bare started/refused bit, and a selectable criterion so
    the SAME inherited exponential-search-then-bisect find() locates either line:

      criterion "no_spill"  ok = regime is FITS                    -> bisects line 1 (spill onset)
      criterion "start"     ok = regime is FITS or SILENT_SPILL     -> bisects line 2 (allocation failure)

    Two probers over ONE shared raw-result dict, so finding the second line never repeats a server start the first
    line already paid for -- a 70B load is minutes, and the two searches overlap heavily by construction. Nothing in
    Prober.find()/boundary_record() is reimplemented: find() reads res["ok"], boundary_record() reads pr.cache, and
    both are populated here in exactly the shapes they already expect."""

    CRITERIA = {"no_spill": ("FITS",), "start": ("FITS", "SILENT_SPILL")}

    def __init__(self, lab, mi, flags, label, lines, criterion, raw=None):
        super().__init__(lab, mi, flags, label)
        if criterion not in self.CRITERIA:
            raise ValueError(f"unknown MX2 criterion {criterion!r}")
        self.lines, self.criterion = lines, criterion
        self.raw = raw if raw is not None else {}
        self.regimes = {}

    def _probe_raw(self, n_ctx):
        self.lab.check()
        self.n += 1
        tag = f"mx2_{self.label}_{n_ctx}_{self.n}"
        srv, info, px = am.start_and_record(self.lab, self.mi, n_ctx, tag, f"MX2_probe_{self.label}_{n_ctx}_{self.n}",
                                            "mx2", self.flags, mx2_label=self.label,
                                            beyond_trained_ctx=n_ctx > self.mi.max_ctx_native,
                                            kv_bpt_meta=getattr(self.mi, "kv_bpt_meta", None))
        regime, props_cap, ev = mx2_classify(info, srv, px, self.lines)
        srv.stop()
        self.lab.resources["server"] = None
        res = {"ok": None, "regime": regime, "props_cap": props_cap, "projected_mib": ev["projected_mib"],
               "logged_mib": ev["logged_mib"], "error": info.get("error"), "vk": px.get("vk_errors"),
               "alloc_failed": px.get("alloc_failed"), "spill_evidence": ev["spill_evidence"],
               "exit_code": info.get("exit_code")}
        self.lab.emit({"record": "mx2_probe", "label": self.label, "model_id": self.mi.model_id, "n_ctx": n_ctx,
                       "regime": regime, "props_n_ctx_cap": props_cap, "projected_mib": ev["projected_mib"],
                       "logged_mib": ev["logged_mib"], "spill_evidence": ev["spill_evidence"],
                       "exit_code": info.get("exit_code"), "error": info.get("error"),
                       "device_local_line_mib": self.lines.get("device_local_mib"),
                       "vulkan_total_line_mib": self.lines.get("vulkan_total_mib"),
                       "beyond_trained_ctx": n_ctx > self.mi.max_ctx_native, "ts_utc": utc_iso()})
        log(f"mx2 {self.label} n_ctx {n_ctx}: {regime} projected={ev['projected_mib']} logged={ev['logged_mib']}")
        return res

    def probe(self, n_ctx):
        n_ctx = int(round(n_ctx / am.STEP)) * am.STEP
        if n_ctx not in self.raw:
            self.raw[n_ctx] = self._probe_raw(n_ctx)
        res = dict(self.raw[n_ctx])
        res["ok"] = res["regime"] in self.CRITERIA[self.criterion]
        self.regimes[n_ctx] = res["regime"]
        self.cache[n_ctx] = res  # boundary_record() reads pr.cache for the two endpoints
        return res


def _mx2_arm(lab, mi, n_ctx, label, flags, lines):
    """One measured arm at a fixed context: start, matched 8000-token prompt, 1 warm-up + 3 calls. No Q0 suite, by
    design (MX2_QUALITY_SCOPE_NOTE)."""
    item = f"MX2_{mi.model_id}_arm_{label}"
    if item in lab.done:
        return None
    lab.check()
    srv, info, px = am.start_and_record(lab, mi, n_ctx, f"mx2arm_{label}_{n_ctx}", item, "mx2_arms", flags)
    regime, props_cap, ev = mx2_classify(info, srv, px, lines)
    try:
        if not info.get("ok"):
            lab.emit({"record": "mx2_arm_result", "arm": label, "model_id": mi.model_id, "n_ctx": n_ctx,
                      "started": False, "regime": regime, "exit_code": info.get("exit_code"),
                      "error": info.get("error"), "error_lines": px.get("error_lines"),
                      "vk_errors": px.get("vk_errors"), "alloc_failed": px.get("alloc_failed"),
                      "quality_scope": MX2_QUALITY_SCOPE_NOTE, "ts_utc": utc_iso()})
            lab.item_done(item)
            return None
        prompt = ov.prompt_for(srv, MX2_MATCHED_PROMPT_TOKENS)
        n_tok = srv.tokenize(prompt)
        calls = ov.measured_sequence(lab, srv, mi, "MX2", item, prompt, n_tok,
                                     extra={"arm": label, "regime": regime, "mx2_n_ctx": n_ctx,
                                            "spill_evidence": ev["spill_evidence"]}, n_calls=MX2_ARM_CALLS)
        lab.emit({"record": "mx2_arm_result", "arm": label, "model_id": mi.model_id, "n_ctx": n_ctx, "started": True,
                  "regime": regime, "props_n_ctx_cap": props_cap, "n_calls_ok": len(calls),
                  "logged_mib": ev["logged_mib"], "projected_mib": ev["projected_mib"],
                  "spill_evidence": ev["spill_evidence"], "matched_prompt_tokens": MX2_MATCHED_PROMPT_TOKENS,
                  "quality_scope": MX2_QUALITY_SCOPE_NOTE, "ts_utc": utc_iso()})
    finally:
        srv.stop()
        lab.resources["server"] = None
    lab.item_done(item)
    return regime


def phase_mx2(lab, run=subprocess.run, ps_fn=ps):
    """Two-line memory budget on evo-x2's unified pool. For each model: bisect the spill-onset line (last context
    that FITS entirely inside the DEVICE_LOCAL heaps) and the allocation-failure line (last context that starts at
    all), both to am.STEP tokens, then measure what the window between them costs by running a matched 8000-token
    arm just below line 1 and another inside the spill window. Reuses t2s_amech.Prober (via Mx2HeapProber),
    am.start_and_record and am.boundary_record directly; the baseline is read and recorded per run, not imported
    from docs/HARDWARE.md."""
    _abort_if_ollama_running("mx2")
    lines = mx2_record_baseline(lab, run=run, ps_fn=ps_fn)
    for mid in MX2_MODELS:
        mi = lab.models.get(mid)
        if mi is None:
            lab.emit({"record": "mx2_skipped", "model_id": mid, "reason": "model not loaded", "ts_utc": utc_iso()})
            log(f"MX2: {mid} not loaded, skipping")
            continue
        item = f"MX2_{mid}_lines"
        if item in lab.done:
            continue
        flags = am.YARN if mid in ov.YARN_MODELS else []
        raw = {}  # shared across both searches: never start the same server twice for the two lines
        pr_fit = Mx2HeapProber(lab, mi, flags, f"{mid}_device_local", lines, "no_spill", raw)
        lo1, hi1 = pr_fit.find(MX2_START_CTX, MX2_START_STEP)
        if lo1 is None:
            lab.emit({"record": "mx2_no_fitting_context", "model_id": mid, "first_fail_n_ctx": hi1,
                      "lines": lines, "ts_utc": utc_iso()})
            log(f"MX2 {mid}: no context fits inside the device_local line at all; nothing to bisect above it")
            lab.item_done(item)
            continue
        am.boundary_record(lab, mi, f"mx2_{mid}_device_local", flags, pr_fit, lo1, hi1,
                           target_note=f"MX2 line 1 (spill onset): device_local {lines['device_local_mib']} MiB")
        pr_start = Mx2HeapProber(lab, mi, flags, f"{mid}_vulkan_total", lines, "start", raw)
        lo2, hi2 = pr_start.find(hi1, MX2_START_STEP)
        if lo2 is not None:
            am.boundary_record(lab, mi, f"mx2_{mid}_vulkan_total", flags, pr_start, lo2, hi2,
                               target_note=f"MX2 line 2 (allocation failure): vulkan_total {lines['vulkan_total_mib']} MiB")
        window = (lo2 - lo1) if lo2 is not None else None
        regimes = {n: r["regime"] for n, r in sorted(raw.items())}
        lab.emit({"record": "mx2_two_line_result", "model_id": mid, "rope_flags": flags or None, "lines": lines,
                  "device_local_last_fit_n_ctx": lo1, "device_local_first_spill_n_ctx": hi1,
                  "vulkan_total_last_start_n_ctx": lo2, "vulkan_total_first_fail_n_ctx": hi2,
                  "spill_window_tokens": window, "regimes_by_n_ctx": regimes,
                  "regime_counts": {r: sum(1 for x in regimes.values() if x == r) for r in sorted(set(regimes.values()))},
                  "n_server_starts": len(raw), "beyond_trained_ctx_at_first_fail": (hi2 or 0) > mi.max_ctx_native,
                  "max_ctx_native": mi.max_ctx_native, "yarn_factor": getattr(mi, "yarn_factor", None),
                  "quality_scope": MX2_QUALITY_SCOPE_NOTE, "ts_utc": utc_iso()})
        log(f"MX2 {mid}: fits<={lo1}, spills from {hi1}, starts<={lo2}, fails from {hi2}, window={window} tokens")
        arms = [("below_line1", lo1)]
        if window and window > am.STEP:
            mid_ctx = int(round(((lo1 + lo2) / 2) / am.STEP)) * am.STEP
            arms.append(("in_spill_window", min(max(mid_ctx, lo1 + am.STEP), lo2)))
        else:
            lab.emit({"record": "mx2_no_spill_window", "model_id": mid, "spill_window_tokens": window,
                      "note": "line 1 and line 2 are within one probe step: no context both starts and spills",
                      "ts_utc": utc_iso()})
        for label, n_ctx in arms:
            _mx2_arm(lab, mi, n_ctx, label, flags, lines)
        lab.item_done(item)


class ResponsivenessSampler:
    """Local interactive-latency probe for one C1 cell: every 30 s, times a trivial local subprocess
    (python -c "pass") with time.monotonic(). Local, not SSH, because this runs inside the harness process on
    evo-t2s itself -- it measures whether the machine is responsive to a new process launch under the memory lock,
    the same class of thing the external laptop-side SSH probe measures from outside. Started right after the
    balloon confirms the lock and stopped right before the balloon is released, so it spans the whole cell."""

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                subprocess.run([PYTHON, "-c", "pass"], timeout=25)
            except Exception:
                pass
            self.samples.append(time.monotonic() - t0)
            self._stop.wait(30.0)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self):
        if not self.samples:
            return {"resp_median_s": None, "resp_max_s": None, "resp_n": 0}
        return {"resp_median_s": round(st.median(self.samples), 3), "resp_max_s": round(max(self.samples), 3),
                "resp_n": len(self.samples)}


# ---------------------------------------------------------------- C1
def _phase_c1(lab, spec, label):
    if not lab.table:
        lab.emit({"record": "c1_disabled", "phase": label, "reason": "no overnight model table loaded", "ts_utc": utc_iso()})
        return
    n_ctx = 16384
    cells = []
    for mid, levels, reps in spec:
        mi, tab = lab.models.get(mid), lab.table.get(mid)
        if not mi or not tab or not tab.get("ok"):
            continue
        need = ov.need_mib(tab, n_ctx, mi)
        for rep in range(reps):
            for lv in levels:
                for arm in (True, False):
                    target = need + lv * 1024
                    if target >= 3072:
                        cells.append((mid, arm, lv, rep, target, need))
    rng = random.Random(SEED)
    rng.shuffle(cells)
    for k, (mid, arm, lv, rep, target, need) in enumerate(cells):
        item = f"C1_{mid}_{'mm' if arm else 'nomm'}_{lv}_{rep}_{k}"
        if item in lab.done:
            continue
        lab.check()
        mi, tab = lab.models[mid], lab.table[mid]
        fill = ov.fill_cap_tokens(tab, n_ctx, 60.0)
        b = L.Balloon(lab, ov.BALLOON_SCRIPT, item)
        binfo = b.start(target)
        lab.emit({"record": "balloon_start", "item_id": item, "info": binfo, "ts_utc": utc_iso()})
        if not binfo.get("ok"):
            b.stop()
            lab.item_done(item)
            continue
        resp = ResponsivenessSampler()
        resp.start()
        try:
            srv = L.Server(lab, mi, n_ctx, mmap=arm, tag=item)
            lab.resources["server"] = srv
            info = srv.start(timeout=1800)
            ov.start_row(lab, srv, mi, "C1", item, info, {"mem_headroom_gb": lv, "need_mib": need})
            if info.get("ok"):
                prompt = ov.prompt_for(srv, fill)
                n_tok = srv.tokenize(prompt)
                ov.measured_sequence(lab, srv, mi, "C1", item, prompt, n_tok, extra={}, mem_headroom_gb=lv)
                ov.probe_sequence(lab, srv, mi, "C1", item, fill, extra={}, mem_headroom_gb=lv)
            srv.stop()
            lab.resources["server"] = None
        finally:
            resp.stop()
            rsum = resp.summary()
            lab.emit({"record": "c1_responsiveness", "item_id": item, "mem_headroom_gb": lv, "mmap": arm, **rsum, "ts_utc": utc_iso()})
            log(f"c1 responsiveness {item}: {rsum}")
        b.stop()
        lab.item_done(item)


def phase_c1(lab):
    _phase_c1(lab, C1_SPEC_8B, "c1")


def phase_c1b(lab):
    _phase_c1(lab, C1_SPEC_REST, "c1b")


# ---------------------------------------------------------------- PERFBOOSTMODE
def phase_perfboost(lab):
    mi = lab.models.get("qwen3-8b")
    if mi is None:
        return
    pb = L.PowerSetting("SUB_PROCESSOR", "PERFBOOSTMODE")
    if pb.guid is None or pb.original is None:
        # Confirmed read-only, before this phase ever ran: PERFBOOSTMODE is not present under SUB_PROCESSOR on this
        # scheme at all -- not merely hidden. `powercfg -attributes SUB_PROCESSOR PERFBOOSTMODE -ATTRIB_SHOW` and the
        # same call with the setting's well-known GUID both returned "Invalid Parameters", and `powercfg /query
        # SCHEME_CURRENT SUB_PROCESSOR` lists only PROCTHROTTLEMIN and PROCTHROTTLEMAX. No value was ever set.
        lab.emit({"record": "perfboost_unavailable", "ts_utc": utc_iso(),
                  "reason": "PERFBOOSTMODE not present under SUB_PROCESSOR on this scheme (checked by alias and by its well-known GUID, both 'Invalid Parameters'); only PROCTHROTTLEMIN/MAX are listed"})
        log("perfboost: PERFBOOSTMODE is not available on this platform, skipping (nothing was set)")
        return
    try:
        pb.set(0)
        lab.emit({"record": "perfboost_set", "original": pb.original, "current": pb.current, "ts_utc": utc_iso()})
        srv = L.Server(lab, mi, CTX_B, tag="perfboost_start")
        lab.resources["server"] = srv
        info = srv.start()
        ov.start_row(lab, srv, mi, "PB", "perfboost_start", info, {})
        if info.get("ok"):
            prompt = ov.prompt_for(srv, FILL_B)
            n_tok = srv.tokenize(prompt)
            for co, mask in (("none", None), ("nonp12", 0xFFF0)):
                item = f"PB_qwen3-8b_{co}"
                if item in lab.done:
                    continue
                hog = None
                if mask is not None:
                    hog, report, aff = L.m3.start_hog(f"night2_{item}", mask, Path(lab.prefix).parent, Path(lab.prefix).name + f"_{item}")
                    time.sleep(5)
                measured_with_extra(lab, srv, mi, "PB", item, prompt, n_tok, {"cpu_mask": hex(mask) if mask else None, "perfboostmode": 0}, co)
                if hog is not None:
                    L.m3.kill_tree(hog.pid)
                    time.sleep(3)
                lab.item_done(item)
        srv.stop()
        lab.resources["server"] = None
    finally:
        rep = pb.restore()
        lab.emit({"record": "perfboost_restore", **rep, "ts_utc": utc_iso()})
        log(f"PERFBOOSTMODE restore: {rep}")


# ---------------------------------------------------------------- PX2
"""PX2: does the B1/B3 CPU co-runner power-coupling effect generalise to an AMD unified-memory part, and is it a power
effect or a memory-bandwidth effect?

The B1/B3/B4 result on evo-t2s is that a CPU co-runner slows iGPU inference, and B4 showed the effect tracks the E-core
*grouping*, not just the core count. Two mechanisms are still confounded there: the co-runner draws package power that
the shared power budget then denies the iGPU, and the co-runner also moves DRAM traffic that competes with the iGPU on
a unified memory controller. On evo-x2 (Ryzen AI Max+ 395, Strix Halo) the two can be separated, because a spin hog and
a bandwidth hog on the *same* four cores draw similar CPU power but move wildly different amounts of DRAM traffic:

  S4  4 physical cores, spin hog (harness/spin_hog_affinity.py) -- power load, near-zero DRAM traffic.
  B4  the SAME 4 physical cores, STREAM-triad hog (harness/bw_hog.py) -- power load plus saturating DRAM traffic.

S4-vs-B4 at a matched core count is therefore the power-versus-bandwidth contrast, and the S-ladder (S2/S4/S8/S8x/S14/
S28) gives the dose-response and the L3/CCD-placement contrast that B4 gave on Intel.

PLATFORM. Written for evo-x2, which host_config.py already supports: main() calls hc.require_host(gethostname()) and
hc.enforce_or_record_interactive_session(), so this file runs on EVO-X2 today with `--phases px2` and stamps
hw_id=evo-x2, gpu_vendor=amd. No new entry point is needed. PX2 rows must never be pooled with the evo-t2s B1/B2/B3/B4
rows: those are a separate arm (different vendor, different core taxonomy).

TOPOLOGY comes from harness/win_cpu_topology.py (GetLogicalProcessorInformationEx, RelationCache), not from
GetSystemCpuSetInformation. That module already reports both cache levels PX2 needs, and it was confirmed against real
hardware rather than assumed:
  l3_groups -- one entry per last-level-cache domain, i.e. one per CCD on a chiplet part. This is what defines a CCD;
      a core's private L2 says nothing about which cores share an L3.
  l2_groups -- on Zen the L2 is private to one physical core and shared only by that core's two SMT threads, so an L2
      group IS the SMT sibling set. Verified live while writing this: on the 8-core/16-thread development laptop
      win_cpu_topology.py returns 8 L2 groups of exactly 2 logical CPUs ([0,1], [2,3], ... [14,15]) at 1024 KB each,
      and a single 16384 KB L3 covering 0-15.
So GetSystemCpuSetInformation is NOT used here: it would add a second, unverified source for SMT pairing that
win_cpu_topology.py already gives, and px2_topology() refuses to guess instead -- if any L2 group spans more than two
logical CPUs (a shared-L2 cluster, which is exactly what evo-t2s's E-cores look like) it raises, because SMT pairing
cannot be derived from L2 on such a part. That refusal is deliberate: it makes PX2 a no-op-with-a-reason on evo-t2s and
on the single-CCD development laptop rather than silently measuring the wrong core sets.

PINNING. px2_pin_server() is genuinely new, and that was re-verified against the real code rather than assumed: the
only SetProcessAffinityMask call anywhere in harness/ is spin_hog_affinity.py's call on *itself*, phase_b4 and
phase_p70 pin only the hog (via L.m3.start_hog's --affinity-mask), and t2s_lab.Server passes `-t 4` worker threads but
sets no affinity at all. PX2 needs the server pinned in every condition, including the N0/N1 baselines, so that a hog
can never preempt the server's own dispatch threads and so that N0 and S28 differ only in the hog.

TELEMETRY actually available on evo-x2, by field name, from t2s_lab.Telemetry.metrics(). Corrected against the real
Telemetry class and its AMD LibreHardwareMonitor feeder (_start_lhm_feeder / lhm_loop, gated on gpu_vendor == "amd"):
  igpu_mhz              REAL on AMD. The LHM feeder maps the Radeon "GPU Core" Clock sensor into the same sys_ring
                        freq/domain-0 rows Sysman uses, so metrics() windows it per call with no AMD-specific branch.
                        The pre-registered "iGPU clock drop" criterion therefore does have a sensor behind it.
  igpu_power_w          REAL on AMD: the Radeon "GPU Core" Power sensor (source "lhm_gpu").
  temp_c_max            REAL on AMD: the Ryzen CPU Temperature sensor. It is also what gives lab.idle_temp a real
                        value, which switches Telemetry.thermal_gate onto its temperature path (tol 3 C) instead of
                        the package-power proxy Intel is stuck with.
  igpu_temp_c_max       REAL on AMD: the Radeon temperature sensor.
  pkg_power_w,          CPU package/cores/uncore power. These come only from the Windows "Energy Meter" counter set,
  rapl_pp0_w,           whose instances (rapl_package0_pkg/_cores/_uncore) are Intel RAPL names. lhm_loop does not map
  rapl_pp1_w            any CPU power sensor into sys_ring, so these are expected null on evo-x2. This is the one real
                        remaining sensor gap, and it is narrower than "no power telemetry": lhm_sensors.ps1 does
                        collect every CPU-group Power sensor into <prefix>_lhm.jsonl, so CPU package power is
                        recoverable offline from that raw file even though no per-call row field carries it.
  igpu_throttle_bits    PRESENT BUT UNINFORMATIVE on AMD: lhm_loop hardcodes throttle_reasons 0 on the freq rows it
                        synthesises, so this field is always [0] on evo-x2 and must not be read as "not throttling".
  cpu_{p,e,lpe}_pct_perf  Intel P/E/LP-E class medians of "% Processor Performance". This part has no P/E split, so
                        these groupings carry no meaning here and PX2 ignores them; px2_per_core_util() reads real
                        per-logical-CPU "% Processor Time" instead, for the hog-saturation check.
  PPT / STAPM / power-limit / throttle-reason registers: still not captured anywhere in this repo.
"""

PX2_SEED = 20260929
PX2_CTX = 8192
PX2_FILL = 2048
PX2_N_PREDICT = 128
PX2_SERVER_PHYS_CORES = 2  # 2 physical cores x 2 SMT threads = the 4 logical CPUs t2s_lab.Server's `-t 4` uses
PX2_SETTLE_S = 20.0
PX2_THERMAL_TOL_C = 3.0
PX2_THERMAL_MAX_S = 180.0
PX2_DRIFT_PCT = 3.0
PX2_BW_ARRAY_MIB = 320.0
PX2_HOG_DURATION_S = 1800
PX2_N_MEASURED = 5
PX2_N_MEASURED_CUT = 3
PX2_CUT_MODELS = ("llama-3.3-70b",)  # cut rule: the 70B gets 3 measured calls, not 5, so it cannot eat the deadline
PX2_MODELS = ["qwen3-8b", "qwen3-14b", "qwen3-32b", "llama-3.3-70b", "llama31-8b"]
# (label, physical cores taken from the hog CCD, from the server CCD, hog kind). None/None means "special": S14 is
# every free physical core, S28 is every free logical CPU (both SMT threads). B4 takes the SAME cores as S4 so the
# power-versus-bandwidth contrast is at a matched core count.
PX2_LADDER = [("S2", 2, 0, "spin"), ("S4", 4, 0, "spin"), ("S8", 8, 0, "spin"), ("S8x", 4, 4, "spin"),
              ("S14", None, None, "spin"), ("S28", None, None, "spin"), ("B4", 4, 0, "bw")]
PX2_HOG_KIND = {label: kind for label, _, _, kind in PX2_LADDER}
PX2_RANDOMISED = [label for label, *_ in PX2_LADDER]
# N0 always first and N1 always last; only the middle block is shuffled, so N1-vs-N0 is a clean session drift check.
PX2_CONDITIONS = ["N0"] + PX2_RANDOMISED + ["N1"]


def px2_topology(raw=None):
    """Physical cores and CCDs from win_cpu_topology.read_cache_groups(). `raw` lets a test inject a recorded topology
    document instead of reading the live machine.

    A physical core is one L2 group (see the section docstring: private L2 per core on Zen, shared only by the core's
    SMT siblings). A CCD is one L3 group. Raises rather than guessing when the part's cache layout does not support
    that reading, so PX2 refuses to run on a shared-L2-cluster part (evo-t2s) or a single-CCD part instead of silently
    measuring the wrong core sets."""
    d = raw if raw is not None else wct.read_cache_groups()
    if d.get("error"):
        raise RuntimeError(f"win_cpu_topology: {d['error']}")
    l2, l3 = d.get("l2_groups") or [], d.get("l3_groups") or []
    if not l2 or not l3:
        raise RuntimeError(f"win_cpu_topology returned {len(l2)} L2 and {len(l3)} L3 groups; need at least one of each")
    wide = [g["logical_cpus"] for g in l2 if len(g["logical_cpus"]) > 2]
    if wide:
        raise RuntimeError(f"L2 group(s) span more than 2 logical CPUs {wide}: this is a shared-L2 cluster part "
                           f"(e.g. Intel E-cores), so a physical core cannot be read off L2 here")
    cores = sorted((sorted(g["logical_cpus"]) for g in l2 if g["logical_cpus"]), key=lambda c: c[0])
    ccds, seen = [], set()
    for g in sorted(l3, key=lambda g: (g["logical_cpus"] or [-1])[0]):
        members = set(g["logical_cpus"])
        mine = [c for c in cores if set(c) <= members]
        if mine:
            ccds.append(mine)
            seen.update(tuple(c) for c in mine)
    orphans = [c for c in cores if tuple(c) not in seen]
    if orphans:
        raise RuntimeError(f"physical core(s) {orphans} belong to no L3 group: cannot assign them to a CCD")
    return {"ccds": ccds, "n_ccd": len(ccds), "n_physical": len(cores),
            "n_logical": sum(len(c) for c in cores),
            "l3_kb": [g["cache_size_kb"] for g in sorted(l3, key=lambda g: (g["logical_cpus"] or [-1])[0])],
            "l2_kb": sorted({g["cache_size_kb"] for g in l2})}


def px2_server_cpus(topo):
    """The logical CPUs llama-server is pinned to: every SMT thread of the first PX2_SERVER_PHYS_CORES physical cores
    of the first CCD. Derived, not hardcoded, so the mask is right whatever the real enumeration order is."""
    ccd0 = topo["ccds"][0]
    if len(ccd0) <= PX2_SERVER_PHYS_CORES:
        raise RuntimeError(f"CCD0 has only {len(ccd0)} physical cores: no free cores left after giving the server "
                           f"{PX2_SERVER_PHYS_CORES}")
    return sorted(l for core in ccd0[:PX2_SERVER_PHYS_CORES] for l in core)


def px2_condition_cpus(topo, server_cpus=None):
    """Logical-CPU set per condition, from px2_topology()'s real core/CCD map.

    The server lives on CCD0; the hog ladder is built out of CCD1 first (so a hog never shares the server's L3 unless
    the condition is specifically testing that), then spills onto CCD0's remaining cores. Any physical core holding one
    of the server's own logical CPUs is excluded *whole*, not just the shared logical, otherwise an S condition would
    put a hog thread on the SMT sibling of a core the server dispatches from.

    Every "physical" condition loads one thread per core (the core's lowest logical CPU) and never two SMT siblings;
    only S28 deliberately uses both siblings of every free core. Raises rather than silently shrinking a condition when
    the machine is too small for it -- a quietly undersized S8 would look like a real measurement.

    On the expected evo-x2 2 x 8 x 2 layout this gives S2=2, S4=4, S8=8, S8x=8 (4 per CCD), S14=14 (CCD1's 8 plus
    CCD0's 6 free, since 2 of CCD0's cores hold the server), S28=28 logical, and B4=S4's four cores."""
    if topo["n_ccd"] < 2:
        raise RuntimeError(f"px2 needs at least 2 CCDs (distinct last-level-cache domains), found {topo['n_ccd']}")
    server = sorted(server_cpus if server_cpus is not None else px2_server_cpus(topo))
    free = [[c for c in ccd if not any(l in server for l in c)] for ccd in topo["ccds"]]
    srv_ccd, hog_ccd = free[0], free[1]
    first = lambda cores: [c[0] for c in cores]

    def take(cores, n, label, where):
        if len(cores) < n:
            raise RuntimeError(f"px2 condition {label} needs {n} free physical cores on {where}, found {len(cores)}")
        return first(cores[:n])

    cond = {"N0": [], "N1": []}
    for label, n_hog, n_srv, _kind in PX2_LADDER:
        if label == "S14":
            cond[label] = first([c for ccd in free for c in ccd])
        elif label == "S28":
            cond[label] = sorted(l for ccd in free for c in ccd for l in c)
        else:
            cond[label] = take(hog_ccd, n_hog, label, "the hog CCD") + \
                          (take(srv_ccd, n_srv, label, "the server's CCD") if n_srv else [])
    for name, cpus in cond.items():
        bad = sorted(set(cpus) & set(server))
        if bad:
            raise RuntimeError(f"px2 condition {name} would use the server's own CPUs {bad}")
        if len(set(cpus)) != len(cpus):
            raise RuntimeError(f"px2 condition {name} lists a logical CPU twice: {cpus}")
    return cond


def px2_mask(cpus):
    """Affinity mask for a list of logical CPUs, the same hex-mask form start_hog and spin_hog_affinity.py take."""
    m = 0
    for c in cpus:
        m |= 1 << c
    return m


def px2_pin_server(pid, cpus):
    """Pin llama-server to `cpus`, and read the mask back. NEW mechanism, not a reuse -- see the section docstring:
    nothing in this repo pins the server itself today, only hogs. Uses the same SetProcessAffinityMask ctypes call
    spin_hog_affinity.py uses on itself, applied to the already-running server process."""
    mask = px2_mask(cpus)
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wt.HANDLE
    k.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    k.SetProcessAffinityMask.restype = wt.BOOL
    k.SetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.c_size_t]
    k.GetProcessAffinityMask.restype = wt.BOOL
    k.GetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k.CloseHandle.restype = wt.BOOL
    k.CloseHandle.argtypes = [wt.HANDLE]
    PROCESS_SET_INFORMATION, PROCESS_QUERY_INFORMATION = 0x0200, 0x0400
    h = k.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, False, int(pid))
    if not h:
        return {"pinned": False, "why": f"OpenProcess failed, error {ctypes.get_last_error()}", "mask": hex(mask),
                "server_cpus": list(cpus)}
    try:
        ok = bool(k.SetProcessAffinityMask(h, mask))
        proc, sysm = ctypes.c_size_t(), ctypes.c_size_t()
        k.GetProcessAffinityMask(h, ctypes.byref(proc), ctypes.byref(sysm))
        return {"pinned": bool(ok and proc.value == mask), "set_ok": ok, "mask": hex(mask),
                "read_back": hex(proc.value), "server_cpus": list(cpus)}
    finally:
        k.CloseHandle(h)


def px2_per_core_util(cpus):
    """Per-logical-CPU "% Processor Time" for the hog-saturation check. ADAPTATION, documented because it is one: the
    streaming SYS_PS telemetry carries "% Processor Utility" for _Total only, plus Intel P/E/LP-E class medians of
    "% Processor Performance" -- and that counter is a frequency ratio, not utilisation, so it cannot answer "is this
    core busy", and its class groupings are meaningless on a part with no P/E split. So this is a separate one-shot
    per-instance read, parsed exactly the way per_core_perf() above parses its instances (InstanceName is "node,cpu"
    on these machines, so the logical CPU is the part after the last comma). Returns {logical: pct} plus the minimum."""
    out = ps('try { (Get-Counter -Counter "\\Processor Information(*)\\% Processor Time" -ErrorAction Stop).CounterSamples | '
             'Where-Object { $_.InstanceName -match "^[0-9]+(,[0-9]+)*$" } | '
             'ForEach-Object { $_.InstanceName + "=" + $_.CookedValue } } catch { "" }', 30)
    d = {}
    for tok in (out or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                d[int(k.rsplit(",", 1)[-1])] = float(v)
            except ValueError:
                pass
    got = {c: d.get(c) for c in cpus}
    vals = [v for v in got.values() if v is not None]
    return {"per_cpu_pct": got, "min_pct": min(vals) if vals else None, "n_read": len(d),
            "all_at_95": bool(vals) and len(vals) == len(cpus) and min(vals) >= 95.0}


def px2_start_hog(lab, item, cpus, kind):
    """Start a co-runner on `cpus`. The spin arm goes through L.m3.start_hog unchanged (same launcher, same report and
    affinity file naming, same kill_tree cleanup). The bandwidth arm launches bw_hog.py the same way and with the same
    argument names, so the report file is a single float either way and L.m3.read_ips reads both -- iterations/s for
    spin, GB/s for bandwidth."""
    mask = px2_mask(cpus)
    out_dir, stem = Path(lab.prefix).parent, Path(lab.prefix).name + f"_{item}"
    if kind == "spin":
        hog, report, aff = L.m3.start_hog(f"px2_{item}", mask, out_dir, stem)
        return hog, report, aff, mask
    report = str(out_dir / f"{stem}_px2_{item}_hog_gbps.txt")
    aff = str(out_dir / f"{stem}_px2_{item}_hog_affinity.json")
    for p in (report, aff):
        Path(p).unlink(missing_ok=True)
    hog = subprocess.Popen([PYTHON, str(ov.DEPLOY / "bw_hog.py"), "--affinity-mask", hex(mask),
                            "--duration-s", str(PX2_HOG_DURATION_S), "--report-file", report, "--affinity-file", aff,
                            "--array-mib", str(PX2_BW_ARRAY_MIB)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    return hog, report, aff, mask


def px2_bw_calibration(lab, cpus):
    """bw_hog's own achieved GB/s with nothing else running, once per session before the sweep, so B4's in-sweep rate
    can be read against a solo ceiling. Runs on the same cores B4 uses."""
    item = "PX2_bw_calibration"
    hog, report, aff, mask = px2_start_hog(lab, item, cpus, "bw")
    rec = {"record": "px2_bw_calibration", "item_id": item, "cpu_mask": hex(mask), "cpus": list(cpus),
           "array_mib": PX2_BW_ARRAY_MIB, "ts_utc": utc_iso()}
    try:
        time.sleep(45)  # 3 x 320 MiB allocated and first-faulted per worker, then a settled rate window
        rec["util"] = px2_per_core_util(cpus)
        rec["gbps_solo"] = L.m3.read_ips(report)
        try:
            rec["hog_affinity"] = json.loads(Path(aff).read_text())
        except Exception:
            rec["hog_affinity"] = None
    finally:
        L.m3.kill_tree(hog.pid)
        time.sleep(3)
    lab.emit(rec)
    log(f"px2 bandwidth calibration: {rec.get('gbps_solo')} GB/s solo on {list(cpus)}")
    return rec.get("gbps_solo")


def px2_order(mid):
    """Shuffled middle block for one model, with the seed that produced it so the order is reproducible from the row."""
    seed = PX2_SEED + ov.crc(mid)
    order = list(PX2_RANDOMISED)
    random.Random(seed).shuffle(order)
    return ["N0"] + order + ["N1"], seed


def px2_drift(ttft_by_cond):
    """N1 versus N0 median TTFT for one model. Flags drift when they differ by more than PX2_DRIFT_PCT, measured
    against N0 (the clean baseline taken before any hog ran for this model)."""
    n0 = [t for t in ttft_by_cond.get("N0", []) if t is not None]
    n1 = [t for t in ttft_by_cond.get("N1", []) if t is not None]
    if not n0 or not n1:
        return {"drift_flag": None, "drift_pct": None, "why": "N0 or N1 has no valid TTFT"}
    m0, m1 = st.median(n0), st.median(n1)
    pct = (m1 - m0) / m0 * 100.0 if m0 else None
    return {"drift_flag": bool(pct is not None and abs(pct) > PX2_DRIFT_PCT), "drift_pct": pct,
            "n0_ttft_median_s": m0, "n1_ttft_median_s": m1, "drift_threshold_pct": PX2_DRIFT_PCT}


def phase_px2(lab):
    """One server per model, held up across all 9 conditions (the B3/P70 pattern: only a model change restarts it), so
    a condition's effect is never confounded with a reload. Order per model is N0, shuffled S/B block, N1."""
    try:
        topo = px2_topology()
        server_cpus = px2_server_cpus(topo)
        cond_cpus = px2_condition_cpus(topo, server_cpus)
    except Exception as e:
        # Recorded, not raised: on evo-t2s (shared-L2 E-core clusters) and on a single-CCD part this is the correct,
        # expected outcome, and a phase that refuses with a reason is better than one that measures the wrong cores.
        lab.emit({"record": "px2_disabled", "reason": f"topology unusable: {e!r}"[:400], "ts_utc": utc_iso()})
        log(f"px2 disabled: {e!r}")
        return
    lab.emit({"record": "px2_topology", "n_ccd": topo["n_ccd"], "n_physical": topo["n_physical"],
              "n_logical": topo["n_logical"], "l3_kb": topo["l3_kb"], "l2_kb": topo["l2_kb"],
              "ccd_cores": topo["ccds"], "server_cpus": server_cpus,
              "condition_cpus": dict(cond_cpus),
              "condition_masks": {k: hex(px2_mask(v)) for k, v in cond_cpus.items() if v},
              "hog_kind_by_cond": dict(PX2_HOG_KIND), "ts_utc": utc_iso()})
    gbps_solo = None
    if "PX2_bw_calibration" not in lab.done:
        gbps_solo = px2_bw_calibration(lab, cond_cpus["B4"])
        lab.item_done("PX2_bw_calibration")
    for mid in PX2_MODELS:
        mi = lab.models.get(mid)
        if mi is None:
            # Same shape as phase_p70: ov.MODEL_FILES has a real entry for every PX2 model including llama-3.3-70b
            # (Llama-3.3-70B-Instruct-Q4_K_M.gguf), so a miss here means load_models() found no verified hash in
            # downloads.jsonl yet on this machine, not that the repo lacks a model definition.
            log(f"px2: {mid} not loaded (no verified entry in downloads.jsonl yet), skipping")
            continue
        order, seed = px2_order(mid)
        n_measured = PX2_N_MEASURED_CUT if mid in PX2_CUT_MODELS else PX2_N_MEASURED
        item0 = f"PX2_{mid}_start"
        srv = L.Server(lab, mi, PX2_CTX, tag=item0)
        lab.resources["server"] = srv
        info = srv.start(timeout=1800)
        pin = px2_pin_server(info["pid"], server_cpus) if info.get("ok") and info.get("pid") \
            else {"pinned": False, "why": "server did not start", "server_cpus": server_cpus}
        ov.start_row(lab, srv, mi, "PX2", item0, info, {"px2_order": order, "px2_seed": seed,
                                                       "server_affinity": pin, "n_measured": n_measured,
                                                       "n_reduced_cut_rule": mid in PX2_CUT_MODELS,
                                                       "bw_gbps_solo": gbps_solo})
        if not info.get("ok"):
            srv.stop()
            lab.resources["server"] = None
            continue
        log(f"px2 {mid}: order {order} (seed {seed}), server pin {pin}, {n_measured} measured calls per condition")
        prompt = ov.prompt_for(srv, PX2_FILL)
        n_tok = srv.tokenize(prompt)
        ttft_by_cond = {}
        for cond in order:
            item = f"PX2_{mid}_{cond}"
            if item in lab.done:
                continue
            lab.check()
            cpus = cond_cpus[cond]
            kind = PX2_HOG_KIND.get(cond, "none") if cpus else "none"
            gate = lab.tele.thermal_gate(lab.idle_temp, tol=PX2_THERMAL_TOL_C, max_wait=PX2_THERMAL_MAX_S,
                                         idle_pkg=lab.idle_pkg)
            lab.emit({"record": "px2_condition_gate", "item_id": item, "model_id": mid, "cond": cond,
                      "gate": gate, "ts_utc": utc_iso()})
            hog, report, aff, mask = None, None, None, None
            util = None
            try:
                if cpus:
                    hog, report, aff, mask = px2_start_hog(lab, item, cpus, kind)
                    time.sleep(PX2_SETTLE_S)
                    util = px2_per_core_util(cpus)
                    lab.emit({"record": "px2_hog_settled", "item_id": item, "cond": cond, "hog_kind": kind,
                              "cpu_mask": hex(mask), "cpus": list(cpus), "settle_s": PX2_SETTLE_S, "util": util,
                              "rate_at_settle": L.m3.read_ips(report), "ts_utc": utc_iso()})
                # NOTE the px2_gate_* prefixes: ov.do_call merges its OWN per-call thermal_gate() result into every row
                # under the plain names (thermal_wait_s, gate_released_by, ...), which are reserved keys -- reusing
                # them here would be a duplicate-keyword TypeError (see tests/test_no_duplicate_row_kwargs.py). These
                # two are the separate pre-condition gate taken before the hog started.
                extra = {"cpu_mask": hex(mask) if mask else None, "cond": cond, "hog_kind": kind,
                         "hog_cpus": list(cpus), "px2_seed": seed, "px2_order": order,
                         "server_affinity_mask": hex(px2_mask(server_cpus)), "server_pinned": pin.get("pinned"),
                         "hog_cpus_all_at_95": (util or {}).get("all_at_95"),
                         "hog_min_core_pct": (util or {}).get("min_pct"),
                         "n_measured_planned": n_measured, "n_reduced_cut_rule": mid in PX2_CUT_MODELS,
                         "bw_gbps_solo": gbps_solo, "px2_gate_released_by": gate.get("gate_released_by"),
                         "px2_gate_wait_s": gate.get("thermal_wait_s")}
                rows = measured_with_extra(lab, srv, mi, "PX2", item, prompt, n_tok, extra, cond,
                                           n_calls=n_measured, max_tokens=PX2_N_PREDICT)
                ttft_by_cond[cond] = [r.get("ttft_s") for r in rows]
                lab.emit({"record": "px2_condition_done", "item_id": item, "model_id": mid, "cond": cond,
                          "hog_kind": kind, "n_rows": len(rows),
                          "hog_rate_during_calls": L.m3.read_ips(report) if report else None,
                          "hog_rate_units": {"bw": "gbps", "spin": "iterations_per_s"}.get(kind),
                          "util": util, "ts_utc": utc_iso()})
            finally:
                # Same cleanup-on-exception contract as phase_b1/b3/b4: the hog is killed even if the calls raise, so
                # no condition's hog can survive into the next one or outlive a deadline abort.
                if hog is not None:
                    L.m3.kill_tree(hog.pid)
                    time.sleep(3)
            lab.item_done(item)
        drift = px2_drift(ttft_by_cond)
        lab.emit({"record": "px2_model_summary", "model_id": mid, "px2_seed": seed, "px2_order": order,
                  "n_measured": n_measured, "n_reduced_cut_rule": mid in PX2_CUT_MODELS, "bw_gbps_solo": gbps_solo,
                  "ttft_median_by_cond": {c: (st.median([t for t in v if t is not None])
                                              if any(t is not None for t in v) else None)
                                          for c, v in ttft_by_cond.items()},
                  **drift, "ts_utc": utc_iso()})
        log(f"px2 {mid} drift check: {drift}")
        srv.stop()
        lab.resources["server"] = None


PRIO = {"b1": 1, "b2": 2, "c1": 3, "b3": 4, "c1b": 5, "b4": 6, "b4_32b": 7, "b4_replicate": 8, "r1_speed": 10,
        "r1_check": 11, "a70": 12, "a70_finalize": 12.5, "p70": 13, "r1b": 14, "r1d": 15, "r1c": 16, "mx2": 17,
        "px2": 18, "perfboost": 9}
PHASE_FN = {"b1": phase_b1, "b2": phase_b2, "c1": phase_c1, "b3": phase_b3, "c1b": phase_c1b, "b4": phase_b4,
           "b4_32b": phase_b4_32b, "b4_replicate": phase_b4_replicate, "r1_speed": phase_r1_speed,
           "r1_check": phase_r1_check, "a70": phase_a70, "a70_finalize": phase_a70_finalize, "p70": phase_p70,
           "r1b": phase_r1b, "r1d": phase_r1d, "r1c": phase_r1c, "mx2": phase_mx2, "px2": phase_px2,
           "perfboost": phase_perfboost}
PHASE_ORDER = "b1,b2,c1,b3,c1b,perfboost"  # mx2/px2 are evo-x2 only: opt in with --phases, never in the default order

# Phases that must pass a 1-item live smoke (server start, stale-server guard, one call, row-schema check) before
# their first real run in a given resumed stem, per the standing rule added after two duplicate-keyword crashes: a
# dry run against a stub lab catches code bugs, but only a real machine catches a bad deploy, a missing dependency
# file, or a wrong assumption about what the live server actually returns.
SMOKE_GATED_PHASES = {"r1_speed", "r1_check", "a70", "p70", "r1b", "r1c", "r1d", "mx2", "px2"}


class SmokeFailure(Exception):
    pass


def _is_stale_server_error(e):
    """True for the error shapes t2s_lab's own port-8385/stale-listener guards raise (assert_port_free's GuardError,
    wrapped as 'STOP guard: ...'; Server.stop()'s 'STOP: could not confirm server pid ... exited'; the unwanted-
    llama-server guard in Server.start()) -- the class of error a stale leftover process causes, and the only class
    this phase loop retries once for. Not matched: ov.Deadline (handled separately, never retried) or any other
    exception (a real bug in the phase's own logic should not be silently retried and masked)."""
    msg = str(e)
    return ("port" in msg and "listener" in msg) or "could not confirm server pid" in msg or \
           "a llama-server process we did not start" in msg or "could not clear stale listener" in msg


def _abort_if_ollama_running(ph):
    """No phase in this script is an Ollama phase (K1/K2 are separate scripts that start and stop Ollama themselves).
    Per the 2026-09-29 contamination check (docs/RESULT_PROVENANCE.md), Ollama must never idle in the background
    during any other phase, so every phase here checks first and aborts the whole run rather than risk a model load
    racing against a measurement. Message contains 'STOP' so t2s_queue.advance() halts instead of auto-advancing."""
    if hc.ollama_process_running():
        raise SmokeFailure(f"STOP: an ollama process is running; refusing to start phase {ph}")


SMOKE_REQUIRED_ROW_KEYS = {"ttft_s", "decode_tok_s", "e2e_s", "outcome"}


def smoke_check_phase(lab, phase, model_id="qwen3-8b"):
    """1-item live smoke for `phase`: start a server (exercises the stale-server guard), run exactly one measured
    call via ov.measured_sequence(n_calls=1), and check the call's result carries the fields every real phase depends
    on (ov.do_call always writes item_id/kind/model_id/server_pid into the emitted row itself -- that construction is
    fixed code, not something a live run can get subtly wrong; ttft_s/decode_tok_s/e2e_s/outcome, in contrast, come
    from the live chat call and are exactly what a bad deploy, a missing dependency, or a wrong assumption about the
    live server's response shape would actually break). Returns (ok, reason). Never raises on a server-side failure
    (that IS the failure being tested for); only a bug in this function itself would raise."""
    mi = lab.models.get(model_id)
    if mi is None:
        return False, f"{model_id} not loaded (check downloads.jsonl)"
    item0 = f"smoke_{phase}_start"
    srv = L.Server(lab, mi, 8192, tag=item0)
    lab.resources["server"] = srv
    try:
        info = srv.start(timeout=1800)
        ov.start_row(lab, srv, mi, "SMOKE", item0, info, {"smoke_for_phase": phase})
        if not info.get("ok"):
            return False, f"server start failed: {info.get('error')}"
        prompt = ov.prompt_for(srv, 256)
        n_tok = srv.tokenize(prompt)
        rows = ov.measured_sequence(lab, srv, mi, "SMOKE", f"smoke_{phase}_call", prompt, n_tok,
                                    extra={"smoke_for_phase": phase}, n_calls=1)
        if not rows:
            return False, "measured_sequence returned no rows (warm-up call failed)"
        row = rows[0]
        missing = SMOKE_REQUIRED_ROW_KEYS - set(row.keys())
        if missing:
            return False, f"row missing expected keys: {sorted(missing)}"
        if row.get("outcome") != "ok":
            return False, f"call outcome {row.get('outcome')!r}, error={row.get('error')}"
        return True, "ok"
    finally:
        srv.stop()
        lab.resources["server"] = None


def _call_s(tab, mid, fill, n_out=128):
    t = tab.get(mid) or {}
    return ov.est_call_s(t, fill, n_out) if t.get("ok") else 45.0  # no overnight timing for this model: a flat guess


def _load_s(tab, mid):
    return (tab.get(mid) or {}).get("load_s") or 15.0


def load_night2_overheads(path):
    """Real per-call overheads (server load, non-warmup call e2e, thermal-gate wait) measured directly from a prior
    night2 results file, medians by model_id. Used to replace the overnight model-table prefill/decode fit for
    night3's pre-launch estimate, per the user's instruction to use real per-call overheads measured in night2 (gate,
    load, settle) rather than the old formula-based estimator. Returns {} if path is missing or unreadable -- callers
    fall back to the overnight-table fit in that case."""
    import statistics as st
    if not path or not Path(path).exists():
        return {}
    loads, calls, gates = {}, {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            mid = r.get("model_id")
            if not mid:
                continue
            if r.get("kind") == "start" and r.get("load_s") is not None:
                loads.setdefault(mid, []).append(r["load_s"])
            elif r.get("kind") == "call" and not r.get("warmup"):
                if r.get("e2e_s") is not None:
                    calls.setdefault(mid, []).append(r["e2e_s"])
                if r.get("thermal_wait_s") is not None:
                    gates.append(r["thermal_wait_s"])
    return {
        "load_s": {m: st.median(v) for m, v in loads.items()},
        "call_e2e_s": {m: st.median(v) for m, v in calls.items()},
        "gate_wait_s": st.median(gates) if gates else 90.0,
    }


def estimate_hours(lab, overheads=None):
    """Hours per phase for the pre-launch printout and the cut-C1b-first rule. When `overheads` (from
    load_night2_overheads) has a real median for a given model, that measured load_s/e2e_s is used in place of the
    overnight model-table's prefill/decode fit, since the fit does not capture actual gate/load/settle time observed
    on this machine. Falls back to the table-fit estimate per-model when no real measurement exists for it (e.g. a
    model night2 never actually ran). Co-runner hog start/stop and memory-lock balloon overhead are still flat
    guesses -- neither source measures those directly per-call."""
    tab = lab.table or {}
    ov_ = overheads or {}
    real_load = ov_.get("load_s", {})
    real_call = ov_.get("call_e2e_s", {})
    gate_s = ov_.get("gate_wait_s", 90.0)

    def load_s(mid):
        return real_load.get(mid) or _load_s(tab, mid)

    def call_s(mid, fill, n_out=128):
        # real medians were measured at whatever fill/n_out night2 actually used (FILL_B/128 for B1-B3, larger for
        # C1); only use the real median for the "normal" 128-token call shape, otherwise fall back to the table fit
        # so probe-length (n_out=32) and long-fill C1 calls still get a fill-aware estimate.
        if n_out == 128 and mid in real_call:
            return real_call[mid]
        return _call_s(tab, mid, fill, n_out)

    est = {}
    for ph in ("b1", "b2", "b3"):
        models, conds = (B1_MODELS, len(B1_MASKS)) if ph == "b1" else (B1_MODELS, len(B2_DUTY)) if ph == "b2" else (B3_MODELS, len(B3_CORESETS))
        est[ph] = sum(load_s(m) + conds * (6 * call_s(m, FILL_B) + gate_s) for m in models) / 3600
    for ph, spec in (("c1", C1_SPEC_8B), ("c1b", C1_SPEC_REST)):
        s = 0.0
        for mid, levels, reps in spec:
            fill = ov.fill_cap_tokens(tab.get(mid) or {}, 16384, 60.0) if tab.get(mid, {}).get("ok") else 2000
            n_cells = reps * len(levels) * 2
            s += n_cells * (load_s(mid) + 6 * call_s(mid, fill) + 5 * call_s(mid, min(fill, 1500), 32) + 90)
        est[ph] = s / 3600
    for ph, models in (("b4", ["qwen3-8b"]), ("b4_32b", ["qwen3-32b"]), ("b4_replicate", ["qwen3-8b"])):
        # B4: 1 server load + len(B4_MASKS) conditions, 5 measured calls/condition, fixed 20s co-runner settle
        # (thermal_gate's corunner_active path) plus 5s hog start + 3s hog kill per non-"none" condition.
        s = 0.0
        for mid in models:
            n_hog = sum(1 for _, mask in B4_MASKS if mask is not None)
            s += load_s(mid) + len(B4_MASKS) * (5 * call_s(mid, FILL_B) + 20) + n_hog * 8
        est[ph] = s / 3600
    est["perfboost"] = (load_s("qwen3-8b") + 2 * (6 * call_s("qwen3-8b", FILL_B) + gate_s)) / 3600
    s = 0.0
    for mid in R1_SPEED_MODELS:
        if mid not in tab and mid not in real_load:
            continue  # not on this machine / not downloaded yet
        s += load_s(mid) + sum(4 * call_s(mid, tgt) for tgt in R1_SPEED_LENGTHS)  # 1 warm-up + 3 measured, per length
    est["r1_speed"] = s / 3600
    s = 0.0
    for mid in R1_CHECK_MODELS:
        n_calls = len(R1_CHECK_PROBE_IDS) * len(R1_CHECK_RATIOS) * len(R1_CHECK_ARMS) * R1_CHECK_REPS  # 180
        s += load_s(mid) + n_calls * call_s(mid, R1_CHECK_FILLER)
    est["r1_check"] = s / 3600
    # A70/P70: no overnight-table or night2 timing exists for the 70B at all (it has never run), so this is a flat
    # guess scaled off the 32B's own load_s/call_s in the same table -- a genuine unknown until A70's own smoke runs.
    a70_load = load_s(A70_MODEL) if A70_MODEL in tab or A70_MODEL in real_load else 3.5 * load_s("qwen3-32b")
    a70_call = call_s(A70_MODEL, A70_MATCHED_PROMPT_TOKENS) if A70_MODEL in real_call else 3.0 * call_s("qwen3-32b", A70_MATCHED_PROMPT_TOKENS)
    n_bisect_probes = 12  # exponential search + bisection to 256 tokens; server-start-only, no measured call
    n_extra_points = 6    # 3 below + 3 above the boundary, also server-start-only
    est["a70"] = (a70_load + (n_bisect_probes + n_extra_points) * a70_load + len(A70_ARMS) * (a70_load + 4 * a70_call)) / 3600
    est["p70"] = (a70_load + 2 * (4 * a70_call + gate_s)) / 3600
    s = 0.0
    for mid in R1B_MODELS:
        n_calls = len(R1B_PROBE_IDS) * len(R1B_RATIOS) * len(R1B_ARM_SUFFIXES) * R1B_REPS  # 240
        s += load_s(mid) + n_calls * call_s(mid, R1B_FILLER, 256)
    est["r1b"] = s / 3600
    s = 0.0
    for mid in R1D_MODELS:
        n_calls = len(R1D_PROBE_IDS) * len(R1_CHECK_RATIOS) * len(R1_CHECK_ARMS) * R1_CHECK_REPS  # 396
        s += load_s(mid) + n_calls * call_s(mid, R1_CHECK_FILLER)
    est["r1d"] = s / 3600
    s = 0.0
    for mid in R1C_MODELS:
        n_calls = len(R1C_PROBE_IDS) * len(R1_CHECK_RATIOS) * len(R1_CHECK_ARMS) * R1_CHECK_REPS  # 360
        s += load_s(mid) + n_calls * call_s(mid, R1_CHECK_FILLER)
    est["r1c"] = s / 3600
    # MX2: server-start-only probes plus two measured arms per model. The two lines share one raw-result cache, so the
    # second search only pays for the probes the first did not already do -- MX2_N_PROBES_PER_MODEL is that shared
    # total (see the comment on the constant), not two independent bisections.
    s = 0.0
    for mid in MX2_MODELS:
        if mid not in tab and mid not in real_load:
            continue  # not on this machine / not downloaded yet
        s += MX2_N_PROBES_PER_MODEL * load_s(mid) + 2 * (load_s(mid) + (1 + MX2_ARM_CALLS) * call_s(mid, MX2_MATCHED_PROMPT_TOKENS))
    est["mx2"] = s / 3600
    # PX2: one server load per model, then len(PX2_CONDITIONS) conditions of (1 warm-up + n_measured) calls, plus the
    # pre-condition gate, a PX2_SETTLE_S hog settle and a ~3 s hog kill on every non-baseline condition, plus the
    # one-off 48 s bandwidth calibration. n_measured follows the cut rule (3 for PX2_CUT_MODELS, else 5).
    s = 48.0
    n_hog_conds = sum(1 for c in PX2_CONDITIONS if c not in ("N0", "N1"))
    for mid in PX2_MODELS:
        n_measured = PX2_N_MEASURED_CUT if mid in PX2_CUT_MODELS else PX2_N_MEASURED
        s += load_s(mid) + len(PX2_CONDITIONS) * ((1 + n_measured) * call_s(mid, PX2_FILL, PX2_N_PREDICT) + gate_s) \
            + n_hog_conds * (PX2_SETTLE_S + 3)
    est["px2"] = s / 3600
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-blobs", required=True)
    ap.add_argument("--deadline-h", type=float, default=10.0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--overnight-table", default=None)
    ap.add_argument("--prior-results", default=None,
                     help="a previous night2 results JSONL to pull real per-call load_s/e2e_s/gate-wait medians from "
                          "for the pre-launch hour estimate, in place of the overnight-table prefill/decode fit")
    ap.add_argument("--phases", default=PHASE_ORDER)
    ap.add_argument("--no-auto-cut", action="store_true", help="do not drop c1b even if the estimate exceeds the deadline")
    ap.add_argument("--r1-check-models", default=None,
                     help="comma list overriding R1_CHECK_MODELS for this run (e.g. the evo-x2 full ladder instead "
                          "of evo-t2s's qwen3-8b,qwen3-14b check set)")
    ap.add_argument("--r1b-models", default=None,
                     help="comma list overriding R1B_MODELS for this run (e.g. the evo-x2 full ladder instead of "
                          "evo-t2s's qwen3-8b,qwen3-14b check set)")
    ap.add_argument("--r1d-models", default=None,
                     help="comma list overriding R1D_MODELS for this run (e.g. the evo-x2 full ladder instead of "
                          "evo-t2s's qwen3-8b,qwen3-14b check set)")
    ap.add_argument("--r1c-models", default=None,
                     help="comma list overriding R1C_MODELS for this run (e.g. the evo-x2 full ladder instead of "
                          "evo-t2s's qwen3-8b,qwen3-14b check set)")
    args = ap.parse_args()
    host_cfg = hc.require_host(socket.gethostname())
    hc.enforce_or_record_interactive_session(host_cfg)  # raises on evo-t2s if occupied; never raises on evo-x2
    if args.r1_check_models:
        globals()["R1_CHECK_MODELS"] = args.r1_check_models.split(",")
    if args.r1b_models:
        globals()["R1B_MODELS"] = args.r1b_models.split(",")
    if args.r1d_models:
        globals()["R1D_MODELS"] = args.r1d_models.split(",")
    if args.r1c_models:
        globals()["R1C_MODELS"] = args.r1c_models.split(",")
    prov = rp.verify_deployed_blobs(ov.DEPLOY, args.expect_blobs)
    lab = make_lab(args, prov, gpu_vendor=host_cfg["gpu_vendor"])
    lab.identity["hw_id"] = host_cfg["hw_id"]
    lab.track_console = not host_cfg.get("interactive_guard", True)
    lab.resources["powercap"] = None
    if args.overnight_table and Path(args.overnight_table).exists():
        lab.table = json.load(open(args.overnight_table, encoding="utf-8"))
    all_models = sorted(set(B1_MODELS + B3_MODELS + R1_SPEED_MODELS + R1_CHECK_MODELS + R1B_MODELS + R1D_MODELS
                            + R1C_MODELS + PX2_MODELS + [m for m, *_ in C1_SPEC]))
    load_models(lab, all_models)
    phases = args.phases.split(",")
    overheads = load_night2_overheads(args.prior_results)
    est = estimate_hours(lab, overheads)
    est_total = sum(est.get(p, 0.0) for p in phases)
    src = f"real medians from {args.prior_results} ({len(overheads.get('load_s', {}))} models)" if overheads else "overnight-table fit (no --prior-results given or file unreadable)"
    log(f"estimated hours per phase [{src}]: {json.dumps({p: round(est.get(p, 0.0), 2) for p in phases})}; total {est_total:.2f} h of {args.deadline_h} h deadline")
    cut = []
    if est_total > args.deadline_h and not args.no_auto_cut and "c1b" in phases:
        phases = [p for p in phases if p != "c1b"]
        cut.append("c1b")
        est_total = sum(est.get(p, 0.0) for p in phases)
        log(f"estimate ({est_total + est.get('c1b', 0):.2f} h) exceeds the {args.deadline_h} h deadline: cut c1b first; remaining estimate {est_total:.2f} h")
    lab.emit({"record": "phase_estimate", "estimated_hours": est, "phases_kept": phases, "phases_cut": cut,
              "deadline_h": args.deadline_h, "ts_utc": utc_iso()})
    Path(lab.prefix + "_manifest.json").write_text(json.dumps({"launch_utc": utc_iso(), "script_provenance": prov,
                                                              "identity": lab.identity, "phases": phases, "phases_cut": cut,
                                                              "estimated_hours": est, "estimate_source": src, "seed": SEED,
                                                              "overnight_table": args.overnight_table,
                                                              "prior_results": args.prior_results},
                                                             indent=1, default=str), encoding="utf-8")
    lab.tele.start()
    note = "completed"
    try:
        time.sleep(15)
        pk = [lab.tele.pkg_now() for _ in range(6) if not time.sleep(1)]
        lab.idle_pkg = st.median([x for x in pk if x is not None]) if any(x is not None for x in pk) else None
        tmp = [lab.tele.temp_now() for _ in range(5) if not time.sleep(1)]
        tmp = [x for x in tmp if x is not None]
        # A real idle_temp switches Telemetry.thermal_gate to the temp-based path (within tol=3.0 C of idle, see
        # t2s_lab.py); on AMD this comes from the LHM CPU package sensor. On Intel (no sensor) it stays None and the
        # gate falls back to the package-power proxy, unchanged.
        lab.idle_temp = st.median(tmp) if tmp else None
        for ph in phases:
            if f"phase_{ph}" in lab.done:
                continue
            _abort_if_ollama_running(ph)
            if ph in SMOKE_GATED_PHASES and f"smoke_ok_{ph}" not in lab.done:
                ok, reason = smoke_check_phase(lab, ph)
                lab.emit({"record": "phase_smoke", "phase": ph, "ok": ok, "reason": reason, "ts_utc": utc_iso()})
                if not ok:
                    # Deliberately outside the per-phase try/except below: a smoke failure must stop the whole run
                    # (not just skip this phase), reach main()'s outer handler, and produce a note containing "STOP"
                    # so t2s_queue.advance() treats it as a halt (writes queue_empty.flag, does not auto-launch the
                    # next queued run) instead of quietly moving on.
                    raise SmokeFailure(f"STOP: smoke failed before phase {ph}: {reason}")
                lab.item_done(f"smoke_ok_{ph}")
                log(f"phase {ph} smoke passed")
            log(f"phase {ph}")
            succeeded = False
            for retry in (False, True):
                try:
                    PHASE_FN[ph](lab)
                    succeeded = True
                    break
                except ov.Deadline:
                    raise
                except Exception as e:
                    stale = _is_stale_server_error(e)
                    lab.emit({"record": "phase_error", "phase": ph, "error": repr(e)[:500], "stale_server": stale,
                             "retried": retry, "ts_utc": utc_iso()})
                    log(f"phase {ph} error: {e!r}")
                    ov.cleanup_partial(lab)
                    if stale and not retry:
                        # A stale port-8385 listener (see the PID 7408 incident, 2026-09-29): the next Server.start()
                        # call inside PHASE_FN[ph] will run ensure_port_free_or_cleanup() itself, so simply retrying
                        # the phase once gives the cleanup-and-retry the spec asks for, instead of failing the whole
                        # phase over what is usually a one-time leftover-process condition.
                        log(f"phase {ph}: stale-server error, retrying once after cleanup")
                        continue
                    break  # not a stale-server error, or already retried once: give up on this phase
            if not succeeded:
                continue
            lab.item_done(f"phase_{ph}")
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
