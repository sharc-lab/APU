"""Queue job x2_family_smoke_mistral_small32 (evo-x2): B2 model-family smoke test, pre-registered in docs/FINDINGS.md
("llama3.3:70b library pull and the B2 model family", 2026-10-14). Queued immediately after x2_model_pulls_scale.

One R2 session of SMOKE_TURNS turns for the family model through x2_r2_agent.run_session unchanged (seed 20260901,
num_ctx 32768, call-2 mode "off", the R2 tool schema, "think" omitted: the model has no thinking mode). The measured
calls are the call-1 requests (the tools-on call of each turn). Per call: native (message.tool_calls non-empty,
detection method "native") and valid (x2_r2_agent.tool_call_valid on every call in it). Mechanical pass: at least
PASS_MIN of the SMOKE_TURNS call-1 requests return a native, valid tool call; a text-embedded call is not native. No
warm-up call (capability check, not a timing measurement).

Writes r2a_turn / r2a_session rows (mode "family_smoke"), one family_smoke_result row per model, run_end, and a
.report.json. x2_r2_scale.py --mode real --smoke-file reads the family_smoke_result row (smoke_ok) as part of the family
real job's per-model preflight. Calls t2s_queue.advance() exactly once on exit; sets no pause.

Usage (evo-x2, from C:\\apu\\ovn): python x2_family_smoke.py --out results\\x2_family_smoke_mistral_small32.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if (_here.parent / "harness").is_dir():
    sys.path.insert(0, str(_here.parent / "harness"))

import x2_r2_agent as ag  # noqa: E402
import x2_r2_scale as sc  # noqa: E402   (sets ag.MODELS think values for the family model)

FAMILY_MODEL = sc.FAMILY_MODEL
SMOKE_TURNS = 5
PASS_MIN = 4
SMOKE_SEED = ag.SEEDS[0]
SMOKE_ARM = sc.arm(sc.TIER_32768)
SMOKE_MODE = "family_smoke"
JOB_ID = "x2_family_smoke_mistral_small32"
OUT_FILE = "x2_family_smoke_mistral_small32.jsonl"


# ── pure parts (unit-tested; also used by the register and by x2_r2_scale's family preflight) ──────

def smoke_summary(rows: list[dict], model: str, n_calls=SMOKE_TURNS, pass_min=PASS_MIN) -> dict:
    """From the latest completed smoke session of `model`: per call-1 request whether a native tool call came back and
    whether it was valid; the count of native-and-valid calls and the mechanical pass."""
    sess = [s for s in ag.completed_sessions(rows) if s.get("mode") == SMOKE_MODE and s.get("model_id") == model]
    if not sess:
        return {"model_id": model, "n_calls": 0, "n_native_valid": 0, "pass_min": pass_min, "passed": False,
                "per_call": [], "reason": "no completed smoke session"}
    att = sess[-1]["attempt_id"]
    turns = sorted((r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") == att),
                   key=lambda r: r["turn_idx"])
    per = []
    for t in turns[:n_calls]:
        c1 = (t.get("calls") or [{}])[0]
        calls = c1.get("tool_calls_parsed") or []
        native = c1.get("tool_detection_method") == "native" and bool(calls)
        per.append({"turn_idx": t["turn_idx"], "tool_task": t.get("tool_task"), "native_tool_call": native,
                    "detection_method": c1.get("tool_detection_method"), "n_tool_calls": len(calls),
                    "tool_names": [c.get("name") for c in calls],
                    "valid": native and all(ag.tool_call_valid(c) for c in calls),
                    "http_error": c1.get("outcome") != "ok" or bool(c1.get("error"))})
    n_ok = sum(1 for p in per if p["valid"])
    out = {"model_id": model, "n_calls": len(per), "n_native_valid": n_ok,
           "n_native": sum(1 for p in per if p["native_tool_call"]), "pass_min": pass_min,
           "passed": len(per) == n_calls and n_ok >= pass_min, "per_call": per, "attempt_id": att}
    if len(per) < n_calls:
        out["reason"] = f"only {len(per)} of {n_calls} calls recorded"
    return out


def smoke_ok(rows: list[dict], model: str) -> tuple[bool, list[str]]:
    """The family real job's smoke preflight: the latest family_smoke_result row for `model` says passed."""
    res = [r for r in rows if r.get("record") == "family_smoke_result" and r.get("model_id") == model]
    if not res:
        return False, [f"{model}: smoke test: no smoke result in the smoke file"]
    r = res[-1]
    if not r.get("passed"):
        return False, [f"{model}: smoke test failed ({r.get('n_native_valid')}/{r.get('n_calls')} native valid tool "
                       f"calls, need {r.get('pass_min')}{'; ' + r['reason'] if r.get('reason') else ''})"]
    return True, []


# ── job ─────────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None, advance=True):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=FAMILY_MODEL)
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models.split(",") if m]

    def log(msg):
        print(f"[{ag.utc_iso()}] {msg}", flush=True)

    emit = sc.row_emitter(out_path)
    note = "completed"
    import host_config as hc
    started = False
    try:
        import socket
        hostname = socket.gethostname()
        if hostname.upper() != "EVO-X2":
            raise RuntimeError(f"x2_family_smoke: host {hostname!r} is not EVO-X2")
        host_cfg = hc.require_host(hostname)
        emit({"record": "run_start", "mode": SMOKE_MODE, "job_id": JOB_ID, "models_requested": models,
              "arm_id": SMOKE_ARM, "seed": SMOKE_SEED, "turns": SMOKE_TURNS, "pass_min": PASS_MIN,
              "think": {m: ag.MODELS.get(m) for m in models}, "agent_blob": sc.git_blob_sha1(ag.__file__),
              "hw_id": host_cfg.get("hw_id"), "ts_utc": ag.utc_iso()})
        ag._start_server(hc)
        started = True
        if not hc.wait_for_ollama_ready(timeout_s=60):
            raise RuntimeError("ollama did not become ready")
        runtime = ag.OllamaRuntime()
        avail = runtime.available_models()
        emit({"record": "runtime", "ollama_version": runtime.version(), "ts_utc": ag.utc_iso()})
        results = {}
        for m in models:
            if not any(ag.same_tag(m, a) for a in avail):
                emit({"record": "missing_model", "model_id": m, "available": avail, "ts_utc": ag.utc_iso()})
                res = {"model_id": m, "n_calls": 0, "n_native_valid": 0, "pass_min": PASS_MIN, "passed": False,
                       "per_call": [], "reason": "not in the Ollama store"}
            else:
                ag.run_session(runtime, m, SMOKE_ARM, SMOKE_SEED, SMOKE_TURNS, emit, log, SMOKE_MODE)
                res = smoke_summary(ag.read_rows(out_path), m)
            emit(dict(res, record="family_smoke_result", ts_utc=ag.utc_iso()))
            results[m] = res
            log(f"{m}: {res['n_native_valid']}/{res['n_calls']} native valid, passed={res['passed']}")
        Path(str(out_path) + ".report.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
        note = "completed; smoke " + "; ".join(f"{m} {'pass' if r['passed'] else 'fail'} "
                                                f"{r['n_native_valid']}/{r['n_calls']}" for m, r in results.items())
        emit({"record": "run_end", "note": note, "ts_utc": ag.utc_iso()})
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        log(note)
        traceback.print_exc()
    finally:
        if started:
            try:
                hc.stop_ollama_server()
            except Exception as e:
                log(f"ollama stop failed: {e!r}")
        if advance:
            try:
                import t2s_queue as tq
                tq.advance(note)
            except Exception as e:
                log(f"queue advance failed: {e!r}")
    return note


if __name__ == "__main__":
    main()
