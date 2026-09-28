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

Usage on evo-t2s (deployed to C:\\apu\\ovn):
  python t2s_night2.py --expect-blobs expected_blobs.json --deadline-h 10 --overnight-table <model_table.json> [--resume <stem>]
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import statistics as st
import time
import types
from pathlib import Path

import run_provenance as rp
import t2s_lab as L
import t2s_overnight as ov
import t2s_queue as tq
from t2s_lab import log, ps, utc_iso

SEED = 20260927
CTX_B = 8192
FILL_B = 7368
B1_MODELS = ["qwen3-4b-2507", "qwen3-8b"]
B1_MASKS = [("none", None), ("p4", 0x000F), ("e4", 0x00F0), ("lp4", 0xF000), ("e8", 0x0FF0),
            ("p4e4", 0x00FF), ("nonp12", 0xFFF0), ("all16", 0xFFFF)]
B2_DUTY = [25, 50, 75, 100]
B3_MODELS = ["llama31-8b", "qwen3-14b", "qwen3-30b-a3b-2507", "qwen3-32b"]
B3_CORESETS = [("none", None), ("p4", 0x000F), ("e4", 0x00F0), ("nonp12", 0xFFF0)]
CPU_CLASSES = {"p": list(range(0, 4)), "e": list(range(4, 12)), "lpe": list(range(12, 16))}
C1_SPEC_8B = [("qwen3-8b", [0, -1, -2], 3)]
C1_SPEC_REST = [("qwen3-14b", [0, -1], 1), ("qwen3-32b", [0, -1], 1)]
C1_SPEC = C1_SPEC_8B + C1_SPEC_REST


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


def make_lab(args, prov):
    ns = types.SimpleNamespace(smoke=False, deadline_h=args.deadline_h, resume=args.resume, stem_prefix="t2s_night2",
                               only=None, reserve_min=20, no_cap_arm=True, max_items=0)
    lab = ov.Lab(ns, prov)
    return lab


def load_models(lab, wanted):
    ov.read_downloads(lab)
    for mid in wanted:
        fn, hyb, mx, yf = ov.MODEL_FILES[mid]
        sha = ov.KNOWN_4B_SHA if mid == "qwen3-4b-2507" else lab.dl_sha[fn]
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
        srv = L.Server(lab, mi, n_ctx, mmap=arm, tag=item)
        lab.resources["server"] = srv
        info = srv.start(timeout=1800)
        ov.start_row(lab, srv, mi, "C1", item, info, {"mem_headroom_gb": lv, "need_mib": need})
        if info.get("ok"):
            prompt = ov.prompt_for(srv, fill)
            n_tok = srv.tokenize(prompt)
            ov.measured_sequence(lab, srv, mi, "C1", item, prompt, n_tok, extra={"mem_headroom_gb": lv}, mem_headroom_gb=lv)
            ov.probe_sequence(lab, srv, mi, "C1", item, fill, extra={"mem_headroom_gb": lv}, mem_headroom_gb=lv)
        srv.stop()
        lab.resources["server"] = None
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


PRIO = {"b1": 1, "b2": 2, "c1": 3, "b3": 4, "c1b": 5, "perfboost": 9}
PHASE_FN = {"b1": phase_b1, "b2": phase_b2, "c1": phase_c1, "b3": phase_b3, "c1b": phase_c1b, "perfboost": phase_perfboost}
PHASE_ORDER = "b1,b2,c1,b3,c1b,perfboost"


def _call_s(tab, mid, fill, n_out=128):
    t = tab.get(mid) or {}
    return ov.est_call_s(t, fill, n_out) if t.get("ok") else 45.0  # no overnight timing for this model: a flat guess


def _load_s(tab, mid):
    return (tab.get(mid) or {}).get("load_s") or 15.0


def estimate_hours(lab):
    """Rough hours per phase from the overnight model table's own prefill/decode fit, for the pre-launch printout and
    the cut-C1b-first rule. Not exact: co-runner and memory-lock overhead (hog/balloon start, waits) are flat guesses."""
    tab = lab.table or {}
    est = {}
    for ph in ("b1", "b2", "b3"):
        models, conds = (B1_MODELS, len(B1_MASKS)) if ph == "b1" else (B1_MODELS, len(B2_DUTY)) if ph == "b2" else (B3_MODELS, len(B3_CORESETS))
        est[ph] = sum(_load_s(tab, m) + conds * (6 * _call_s(tab, m, FILL_B) + 10) for m in models) / 3600
    for ph, spec in (("c1", C1_SPEC_8B), ("c1b", C1_SPEC_REST)):
        s = 0.0
        for mid, levels, reps in spec:
            fill = ov.fill_cap_tokens(tab.get(mid) or {}, 16384, 60.0) if tab.get(mid, {}).get("ok") else 2000
            n_cells = reps * len(levels) * 2
            s += n_cells * (_load_s(tab, mid) + 6 * _call_s(tab, mid, fill) + 5 * _call_s(tab, mid, min(fill, 1500), 32) + 90)
        est[ph] = s / 3600
    est["perfboost"] = (_load_s(tab, "qwen3-8b") + 2 * (6 * _call_s(tab, "qwen3-8b", FILL_B) + 10)) / 3600
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-blobs", required=True)
    ap.add_argument("--deadline-h", type=float, default=10.0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--overnight-table", default=None)
    ap.add_argument("--phases", default=PHASE_ORDER)
    ap.add_argument("--no-auto-cut", action="store_true", help="do not drop c1b even if the estimate exceeds the deadline")
    args = ap.parse_args()
    if socket.gethostname().upper() != "EVO-T2S":
        raise SystemExit("EVO-T2S only")
    q = ps("try { (& query.exe user 2>&1) -join \"`n\" } catch { $_.Exception.Message }")
    if "No User exists" not in q:
        raise SystemExit("another interactive session is logged in: " + q)
    prov = rp.verify_deployed_blobs(ov.DEPLOY, args.expect_blobs)
    lab = make_lab(args, prov)
    lab.resources["powercap"] = None
    if args.overnight_table and Path(args.overnight_table).exists():
        lab.table = json.load(open(args.overnight_table, encoding="utf-8"))
    all_models = sorted(set(B1_MODELS + B3_MODELS + [m for m, *_ in C1_SPEC]))
    load_models(lab, all_models)
    phases = args.phases.split(",")
    est = estimate_hours(lab)
    est_total = sum(est.get(p, 0.0) for p in phases)
    log(f"estimated hours per phase: {json.dumps({p: round(est.get(p, 0.0), 2) for p in phases})}; total {est_total:.2f} h of {args.deadline_h} h deadline")
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
                                                              "estimated_hours": est, "seed": SEED,
                                                              "overnight_table": args.overnight_table},
                                                             indent=1, default=str), encoding="utf-8")
    lab.tele.start()
    note = "completed"
    try:
        time.sleep(15)
        pk = [lab.tele.pkg_now() for _ in range(6) if not time.sleep(1)]
        lab.idle_pkg = st.median([x for x in pk if x is not None]) if any(x is not None for x in pk) else None
        lab.idle_temp = None
        for ph in phases:
            if f"phase_{ph}" in lab.done:
                continue
            log(f"phase {ph}")
            try:
                PHASE_FN[ph](lab)
            except ov.Deadline:
                raise
            except Exception as e:
                lab.emit({"record": "phase_error", "phase": ph, "error": repr(e)[:500], "ts_utc": utc_iso()})
                log(f"phase {ph} error: {e!r}")
                ov.cleanup_partial(lab)
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
