"""qwen3:32b AMD memory-boundary spill sweep on evo-x2 between n_ctx 20480 and 40960 (docs/FINDINGS.md 2026-10-22,
"PRE-REGISTRATION: qwen3-32b spill point and spill cost on evo-x2"). Queue id x2_mx2_qwen3_32b_spill_sweep.

Same server path and launch flags as the MX2 runs (harness/mx2_validation.run_regime_point, mx2_spill_stats):
llama-server b10970 Vulkan, LlamaServerConfig(n_gpu_layers=99, platform evo-x2, backend vulkan, reasoning_budget=0;
-np 1, --no-context-shift, reasoning format deepseek), the fixed MX2 prompt (about 2048 tokens), n_predict 128, the
per-process GPU dedicated/shared usage counters (mx2_validation.query_gpu_process_memory) and MX2's spill threshold
(shared usage more than SPILL_SHARED_DELTA_THRESHOLD_MIB above the 20480 baseline, unclamped). Differences, all stated
in the pre-registration: one server start per point with 1 warm-up + MEASURED_CALLS measured calls (repo statistics
convention) instead of one call per start, a distinct first line per call so the server's prompt cache cannot serve
a call from the previous one, its own port, and the stale-server guard (harness/server_guard.py) before each start,
after each start and before every request.

Points: GRID (4096 apart, 20480 to 40960, every request at or below the GGUF's trained context so nothing is clamped);
then, if the first spill is at a grid point G above 20480, REFINE_STEPS below G (G-3072, G-2048, G-1024). A point
whose served n_ctx differs from the request is labelled clamped and never counts as a spill or a no-spill point.

    python spill32\\mx2_qwen32b_spill_sweep.py --out C:\\apu\\ovn\\results\\x2_mx2_qwen3_32b_spill_sweep.jsonl
    python spill32\\mx2_qwen32b_spill_sweep.py --check     imports, files, port, Ollama loaded-model check; no start
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
_front = [str(HERE)] + ([str(HERE.parent)] if (HERE.parent / "mx2_validation.py").exists() else [])
if (HERE.parent / "harness").is_dir():          # repo layout
    _front = [str(HERE)]
sys.path[:] = _front + [p for p in sys.path if p not in _front]

MODEL_ID = "qwen3-32b"
PORT = 58411                                    # its own port (MX2 used 58399)
GRID = (20480, 24576, 28672, 32768, 36864, 40960)
REFINE_STEPS = (3072, 2048, 1024)
BASELINE_CTX = 20480
MEASURED_CALLS = 5
WARMUP_CALLS = 1
N_PREDICT = 128
OLLAMA_PS = "http://127.0.0.1:11434/api/ps"
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def git_blob_sha1(path) -> str | None:
    try:
        b = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


# ── pure parts ────────────────────────────────────────────────────────────────────────────────────

def refine_points(first_spill: int | None, grid=GRID, steps=REFINE_STEPS) -> list[int]:
    """Points below the first spilling grid point (ascending); none if no spill, or if the first spill is the
    baseline itself."""
    if first_spill is None or first_spill <= grid[0]:
        return []
    return sorted(first_spill - s for s in steps)


def mean_ci95(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    m = st.mean(v)
    if len(v) < 2:
        return m, None, len(v)
    return m, T_CRIT.get(len(v) - 1, 1.96) * st.stdev(v) / math.sqrt(len(v)), len(v)


def welch_diff_ci(a, b):
    """Mean(b) - mean(a) with a 95% Welch CI (df floored, conservative). None when either side has < 2 values."""
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = math.sqrt(va + vb)
    d = st.mean(b) - st.mean(a)
    if se == 0:
        return {"diff": d, "lo": d, "hi": d, "df": None}
    df = (va + vb) ** 2 / ((va ** 2) / (len(a) - 1) + (vb ** 2) / (len(b) - 1))
    t = T_CRIT.get(max(1, int(math.floor(df))), 1.96)
    return {"diff": d, "lo": d - t * se, "hi": d + t * se, "df": df}


def classify(point: dict, baseline_shared: float | None, threshold: float) -> str:
    """"clamped" | "spill" | "no_spill" | "unknown" for one point (median measured shared usage vs the baseline)."""
    if point.get("clamped"):
        return "clamped"
    sh = point.get("shared_mib_median")
    if sh is None or baseline_shared is None:
        return "unknown"
    return "spill" if sh - baseline_shared > threshold else "no_spill"


def summarize(rows: list[dict], threshold: float) -> dict:
    """The pre-registered outcome from the rows: per point the classification, the spill point (smallest unclamped
    spilling n_ctx), the largest unclamped no-spill n_ctx below it, and the spill cost (decode tok/s and TTFT, first
    spill vs last no-spill: means with 95% CIs, Welch difference CI, ratio)."""
    pts = {}
    for r in rows:
        if r.get("record") == "sweep_call" and not r.get("warmup") and r.get("ok"):
            p = pts.setdefault(r["requested_n_ctx"], {"decode": [], "ttft": [], "shared": [], "dedicated": [],
                                                      "clamped": False})
            p["decode"].append(r.get("decode_tok_s"))
            p["ttft"].append(r.get("ttft_ms"))
            p["shared"].append(r.get("gpu_shared_mib"))
            p["dedicated"].append(r.get("gpu_dedicated_mib"))
        if r.get("record") == "sweep_point":
            p = pts.setdefault(r["requested_n_ctx"], {"decode": [], "ttft": [], "shared": [], "dedicated": [],
                                                      "clamped": False})
            p["clamped"] = bool(r.get("clamped"))
            p["served_n_ctx"] = r.get("served_n_ctx")
            p["started"] = r.get("started")
    for p in pts.values():
        sh = [x for x in p["shared"] if x is not None]
        p["shared_mib_median"] = st.median(sh) if sh else None
        de = [x for x in p["dedicated"] if x is not None]
        p["dedicated_mib_median"] = st.median(de) if de else None
        p["n_measured"] = len([x for x in p["decode"] if x is not None])
    base = (pts.get(BASELINE_CTX) or {}).get("shared_mib_median") if not (pts.get(BASELINE_CTX) or {}).get(
        "clamped") else None
    for n, p in pts.items():
        p["class"] = classify(p, base, threshold)
        p["shared_delta_mib"] = (None if p["shared_mib_median"] is None or base is None
                                 else p["shared_mib_median"] - base)
    order = sorted(pts)
    spill = next((n for n in order if pts[n]["class"] == "spill"), None)
    below = [n for n in order if pts[n]["class"] == "no_spill" and (spill is None or n < spill)]
    last_ok = max(below) if below else None
    cost = None
    if spill is not None and last_ok is not None:
        a, b = pts[last_ok], pts[spill]
        dm = (mean_ci95(a["decode"]), mean_ci95(b["decode"]))
        tm = (mean_ci95(a["ttft"]), mean_ci95(b["ttft"]))
        dd, td = welch_diff_ci(a["decode"], b["decode"]), welch_diff_ci(a["ttft"], b["ttft"])
        cost = {"decode_no_spill": dm[0], "decode_spill": dm[1], "decode_ratio": (dm[1][0] / dm[0][0]
                                                                                 if dm[0][0] else None),
                "decode_diff_ci": dd, "ttft_no_spill": tm[0], "ttft_spill": tm[1],
                "ttft_ratio": (tm[1][0] / tm[0][0] if tm[0][0] else None), "ttft_diff_ci": td,
                "decode_reading": cost_reading(dd, "decode", lower_is_worse=True),
                "ttft_reading": cost_reading(td, "TTFT", lower_is_worse=False)}
    return {"points": {n: {k: v for k, v in pts[n].items() if k not in ("decode", "ttft", "shared", "dedicated")}
                       | {"decode": mean_ci95(pts[n]["decode"]), "ttft": mean_ci95(pts[n]["ttft"])}
                       for n in order},
            "baseline_shared_mib": base, "spill_point": spill, "last_no_spill": last_ok, "cost": cost,
            "threshold_mib": threshold}


def cost_reading(ci, name, lower_is_worse):
    """Pre-registered: a cost is shown when the Welch 95% CI of (spill - no-spill) excludes 0 in the worse direction."""
    if ci is None:
        return f"{name}: not read (fewer than 2 calls on a side)"
    worse = ci["hi"] < 0 if lower_is_worse else ci["lo"] > 0
    better = ci["lo"] > 0 if lower_is_worse else ci["hi"] < 0
    return f"{name}: " + ("spill cost shown" if worse else "faster after the spill point" if better
                          else "no spill cost shown")


# ── runtime ───────────────────────────────────────────────────────────────────────────────────────

def ollama_loaded_models() -> list[str] | None:
    """Models Ollama has loaded right now; [] if Ollama answers with none; None if Ollama is not running."""
    try:
        with urllib.request.urlopen(OLLAMA_PS, timeout=5) as r:
            return [m.get("name") or m.get("model") for m in (json.loads(r.read()).get("models") or [])]
    except Exception:
        return None


def call_prompt(mx, k: int) -> str:
    return f"Request {k}.\n" + mx.FIXED_PROMPT


def run_point(n_ctx, emit, log, mx, sg, LlamaServerConfig, LlamaServerSession, call_index):
    """One server start at n_ctx: guard, 1 warm-up + MEASURED_CALLS calls, counters after each call."""
    t0 = time.monotonic()
    point = {"record": "sweep_point", "model_id": MODEL_ID, "requested_n_ctx": n_ctx, "ts_utc": utc_iso()}
    try:
        point["port_guard"] = sg.assert_port_free(PORT)
    except sg.GuardError as e:
        point.update({"started": False, "error": str(e)[:400], "guard_record": e.record})
        emit(point)
        raise
    cfg = LlamaServerConfig(exe=mx.LLAMA_SERVER_EXE, model=mx.GGUF_PATHS[MODEL_ID], ctx_size=n_ctx, port=PORT,
                            n_gpu_layers=99, platform="evo-x2", backend="vulkan", reasoning_budget=0)
    try:
        with LlamaServerSession(cfg) as s:
            pid = s._proc.pid
            served = s.n_ctx_slot
            m = sg.assert_server_matches(f"http://127.0.0.1:{PORT}", PORT, pid,
                                         {"model_path": cfg.model, "n_ctx": n_ctx, "np": 1, "ngl": 99})
            point.update({"started": True, "server_pid": pid, "served_n_ctx": served, "clamped": served != n_ctx,
                          "server_guard": m, "thinking_suppressed": s.thinking_suppression_verified})
            rg = sg.RequestGuard(PORT, pid)
            for i in range(WARMUP_CALLS + MEASURED_CALLS):
                ok_pid, listeners = rg.check()
                row = {"record": "sweep_call", "model_id": MODEL_ID, "requested_n_ctx": n_ctx, "served_n_ctx": served,
                       "warmup": i < WARMUP_CALLS, "call": i, "server_pid": pid, "listener_pids": listeners}
                if not ok_pid:
                    row.update({"ok": False, "error": f"listener pids {listeners} != {pid}; condition stopped"})
                    emit(row)
                    raise RuntimeError(row["error"])
                call_index[0] += 1
                text, lat, ttft, tin, tout, reason, think = s.call(call_prompt(mx, call_index[0]), N_PREDICT)
                mem = mx.query_gpu_process_memory(pid)
                decode_s = (lat - ttft) / 1000 if ttft is not None else None
                row.update({"ok": True, "latency_ms": lat, "ttft_ms": ttft, "tokens_in": tin, "tokens_out": tout,
                            "done_reason": reason, "thinking_chars": think,
                            "decode_tok_s": ((tout - 1) / decode_s) if decode_s and decode_s > 0 and tout > 1 else None,
                            "prefill_tok_s": (tin / (ttft / 1000)) if ttft else None,
                            "gpu_dedicated_mib": mem["dedicated_mib"], "gpu_shared_mib": mem["shared_mib"],
                            "ts_utc": utc_iso()})
                emit(row)
            import t2s_lab as L
            parsed = L.parse_server_log(s._log_path) if s._log_path else {}
            point.update({"model_buffer_mib": parsed.get("model_buffer_mib"), "kv_buffer_mib": parsed.get("kv_buffer_mib"),
                          "compute_buffer_mib": parsed.get("compute_buffer_mib"), "server_log": s._log_path})
    except sg.GuardError as e:
        point.update({"started": point.get("started", False), "error": str(e)[:400], "guard_record": e.record})
    except Exception as e:
        point.update({"started": point.get("started", False), "error": repr(e)[:400]})
    point["after_stop_listeners"] = sg.port_listeners(PORT)
    point["wall_s"] = time.monotonic() - t0
    emit(point)
    log(f"n_ctx={n_ctx} started={point.get('started')} served={point.get('served_n_ctx')} "
        f"error={point.get('error')} wall={point['wall_s']:.0f}s")
    return point


def provenance(mods) -> dict:
    return {"harness_file": str(Path(__file__).resolve()), "harness_blob": git_blob_sha1(__file__),
            "modules": {n: {"file": str(Path(m.__file__).resolve()), "blob": git_blob_sha1(m.__file__)}
                        for n, m in mods.items()}}


def main(argv=None, advance=True) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    import mx2_validation as mx
    import server_guard as sg
    import llama_server
    mods = {"mx2_validation": mx, "server_guard": sg, "llama_server": llama_server}
    if args.check:
        print(json.dumps({"provenance": provenance(mods), "exe_exists": Path(mx.LLAMA_SERVER_EXE).exists(),
                          "gguf_exists": Path(mx.GGUF_PATHS[MODEL_ID]).exists(),
                          "port_listeners": sg.port_listeners(PORT), "ollama_loaded": ollama_loaded_models(),
                          "grid": GRID, "refine_if_spill_at_32768": refine_points(32768),
                          "threshold_mib": mx.SPILL_SHARED_DELTA_THRESHOLD_MIB}, indent=1, default=str))
        return "checked"
    if not args.out:
        ap.error("--out is required for a run")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def log(msg):
        print(f"[{utc_iso()}] {msg}", flush=True)
    note = "completed"
    try:
        import socket
        if socket.gethostname().upper() != "EVO-X2":
            raise RuntimeError(f"runs on EVO-X2 only, this is {socket.gethostname()!r}")
        loaded = ollama_loaded_models()
        emit({"record": "run_start", "model_id": MODEL_ID, "port": PORT, "grid": GRID, "refine_steps": REFINE_STEPS,
              "warmup_calls": WARMUP_CALLS, "measured_calls": MEASURED_CALLS, "n_predict": N_PREDICT,
              "threshold_mib": mx.SPILL_SHARED_DELTA_THRESHOLD_MIB, "exe": mx.LLAMA_SERVER_EXE,
              "gguf": mx.GGUF_PATHS[MODEL_ID], "ollama_loaded_models": loaded, **provenance(mods),
              "ts_utc": utc_iso()})
        if loaded:
            raise RuntimeError(f"Ollama has models loaded {loaded}: GPU memory is not free for the sweep; refused")
        from llama_server import LlamaServerConfig, LlamaServerSession
        idx = [0]
        for n in GRID:
            p = run_point(n, emit, log, mx, sg, LlamaServerConfig, LlamaServerSession, idx)
            if p.get("guard_record", {}).get("guard") == "port_not_free":
                raise RuntimeError("port not free; stopped")
        s = summarize(ag_rows(out), mx.SPILL_SHARED_DELTA_THRESHOLD_MIB)
        for n in refine_points(s["spill_point"]):
            run_point(n, emit, log, mx, sg, LlamaServerConfig, LlamaServerSession, idx)
        s = summarize(ag_rows(out), mx.SPILL_SHARED_DELTA_THRESHOLD_MIB)
        Path(str(out) + ".report.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
        emit({"record": "run_end", "spill_point": s["spill_point"], "last_no_spill": s["last_no_spill"],
              "ts_utc": utc_iso()})
        log(json.dumps({k: s[k] for k in ("spill_point", "last_no_spill", "baseline_shared_mib")}))
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


def ag_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


if __name__ == "__main__":
    main()
