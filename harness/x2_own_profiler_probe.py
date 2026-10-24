"""own-profiler-probe (operator item 3, 2026-10-23): the MEASURED probe that replaces the own-profiler ratio prediction.
Pre-registered in docs/EXPERIMENTS.md "2026-10-23: own-profiler-probe (measured probe) and own-dse-v1 trivial cells";
rows own-profiler-probe and own-profiler-probe-hours. Queue job x2_own_profiler_probe_v1 on evo-x2.

Run on evo-x2 from a side directory (C:\\apu\\ovn\\ownprobe, next to copies of x2_r2_evalset.py and the modules it
imports first), so x2_r2_agent, t2s_queue and host_config come from the parent C:\\apu\\ovn:

    py -3.12 ownprobe\\x2_own_profiler_probe.py --out C:\\apu\\ovn\\results\\x2_own_profiler_probe_v1.jsonl
    py -3.12 ownprobe\\x2_own_profiler_probe.py --out ... --check-plan     (no server, no model call, no queue)

For each window x KV config in CONFIGS, in order: one call of x2_r2_evalset.main (advance=False) with
--plan own_profiler_probe_v1 --tiers <tier>, seed 20260901, the config's --server-env, and its own output file
<out stem>_<tier>.jsonl. x2_r2_evalset gives each config the job's own Ollama server (stopped and restarted so the KV
env takes effect; refuses if the tray app runs), OLLAMA_DEBUG=1 with its own log, detector v2 on every call
(own_compaction.CutLogRuntime), the llama_kv_cache lines (record kv_cache_log), loaded_context per turn, run_start /
run_end in the config file, and the session stopped by x2_r2_evalset.PROBE_WALL_S (600 s) of cumulative turn wall
clock (stop_before_turn: 10 minutes PER CONFIG). A config whose file already has run_end is skipped (resume). This
driver writes run_start, one probe_config record per config and run_end to --out, and calls tq.advance() once on exit
(never a pause).

Pure parts (probe_features, reference_features, score, probe_text, hours_estimate) read rows only and are unit-tested
with fake records in tests/test_own_profiler_probe.py; nothing at import time starts a server or touches the queue.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if not (_here / "t2s_queue.py").exists():
    sys.path.insert(1, str(_here.parent))

PROBE_SEED = 20260901
PROBE_BUDGET_S = 600.0          # equal to x2_r2_evalset.PROBE_WALL_S (asserted in main and in the tests)
TARGET_PCT = 10.0
MODEL = "qwen3:14b"
X2_DEFAULT_WINDOW = 40960       # qwen3:14b loaded context at ollama_default on evo-x2 (x2_evalset_tasks_real_v2 rows)
RESULTS = r"C:\apu\ovn\results"
RULES_FROM = RESULTS + r"\x2_evalset_tasks_val_v2.jsonl"
B3V2_FILE = "results/x2_evalset_tasks_real_v2.jsonl"
REF_Q8 = "results/x2_own_config_probe_q8_15360.jsonl"
REF_Q4 = "results/x2_own_config_probe_q4_28928.jsonl"
PROBE_FILE = "results/x2_own_profiler_probe_v1.jsonl"
_FA = ["OLLAMA_FLASH_ATTENTION=1"]
# (tier, KV type, requested window, extra server env, committed reference file)
CONFIGS = [
    {"tier": "ollama_default", "kv": "f16", "window": X2_DEFAULT_WINDOW, "env": [], "ref": B3V2_FILE},
    {"tier": "ollama_ctx_16384", "kv": "f16", "window": 16384, "env": [], "ref": B3V2_FILE},
    {"tier": "ollama_ctx_8192", "kv": "f16", "window": 8192, "env": [], "ref": B3V2_FILE},
    {"tier": "ollama_ctx_4096", "kv": "f16", "window": 4096, "env": [], "ref": B3V2_FILE},
    {"tier": "ollama_ctx_15360", "kv": "q8_0", "window": 15360, "env": ["OLLAMA_KV_CACHE_TYPE=q8_0"] + _FA, "ref": REF_Q8},
    {"tier": "ollama_ctx_28928", "kv": "q4_0", "window": 28928, "env": ["OLLAMA_KV_CACHE_TYPE=q4_0"] + _FA, "ref": REF_Q4},
]
# hours ESTIMATE: per config overhead beyond the session (server stop / start / ready, unload, KV-log read). Point is an
# ASSUMPTION (60 s); high = x2_r2_evalset's own ready timeouts (60 s, then a restart and 90 s) plus 30 s.
OVERHEAD_S = {"point": 60.0, "high": 180.0}
SESSION_TURNS = 30


def config_file(out: str | Path, tier: str) -> Path:
    out = Path(out)
    return out.with_name(f"{out.stem}_{tier}{out.suffix}")


def config_argv(cfg: dict, out: str | Path, rules_from: str = RULES_FROM) -> list[str]:
    argv = ["--design", "v2", "--mode", "real", "--workload", "evalset", "--guardrails", "off", "--models", MODEL,
            "--preflight", "v2_te_deviation", "--rules-from", rules_from, "--plan", "own_profiler_probe_v1",
            "--tiers", cfg["tier"], "--seeds", str(PROBE_SEED)]
    for e in cfg["env"]:
        argv += ["--server-env", e]
    return argv + ["--out", str(config_file(out, cfg["tier"]))]


# ── pure scoring ─────────────────────────────────────────────────────────────────────────────────────

def _base(arm: str) -> str:
    for sfx in ("_call2_notools", "_call2_forcednone_format", "_call2_forcednone"):
        if arm.endswith(sfx):
            return arm[:-len(sfx)]
    return arm


def has_run_end(rows) -> bool:
    return any(r.get("record") == "run_end" for r in rows)


def probe_prefix(turns, budget_s=PROBE_BUDGET_S):
    """Longest prefix whose cumulative turn_wall_s is <= budget_s (as src/dse/own_profiler.probe_prefix)."""
    out, cum = [], 0.0
    for t in turns:
        if cum + t["turn_wall_s"] > budget_s:
            break
        cum += t["turn_wall_s"]
        out.append(t)
    return out


def _mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None


def probe_features(rows, cfg, seed=PROBE_SEED, budget_s=PROBE_BUDGET_S) -> dict:
    """The probe of one config from its file's rows: the latest attempt of the probe seed, scored prefix, seconds per
    task, accuracy, detector-v2 overflow rate, loaded windows and KV lines, and whether it is valid (pre-registered:
    run_end, >= 1 scored task turn, every scored turn loaded the requested window, KV line shows the requested type)."""
    turns = [r for r in rows if r.get("record") == "r2a_turn" and r.get("mode") == "real" and r.get("seed") == seed
             and _base(r.get("arm_id", "")) == cfg["tier"]]
    last = turns[-1]["attempt_id"] if turns else None
    turns = sorted((r for r in turns if r["attempt_id"] == last), key=lambda r: r["turn_idx"])
    scored = [t for t in probe_prefix(turns, budget_s) if t.get("item_kind", "task") == "task"]
    kv_lines = [ln for r in rows if r.get("record") == "kv_cache_log" for ln in (r.get("lines") or [])]
    loaded = sorted({t.get("loaded_context") for t in scored}, key=str)
    out = {"tier": cfg["tier"], "kv": cfg["kv"], "window": cfg["window"], "run_end": has_run_end(rows),
           "n_turns_run": len(turns), "n_scored": len(scored), "loaded": loaded, "kv_lines": kv_lines,
           "seconds": _mean(t["turn_wall_s"] for t in scored),
           "acc": _mean(float(bool(t.get("end_state_ok"))) for t in scored),
           "overflow_rate": _mean(float(bool(t.get("cut_v2_flagged"))) for t in scored),
           "flagged": sum(1 for t in scored if t.get("cut_v2_flagged")),
           "no_log_turns": sum(1 for t in scored if t.get("cut_v2_no_log_calls")),
           "stopped_by_budget": any(r.get("stopped_by_budget") for r in rows if r.get("record") == "probe_session")}
    why = []
    if not out["run_end"]:
        why.append("no run_end")
    if not scored:
        why.append("no scored task turn")
    if scored and loaded != [cfg["window"]]:
        why.append(f"loaded {loaded} != {cfg['window']}")
    if not any(f"K ({cfg['kv']})" in ln for ln in kv_lines):
        why.append(f"no llama_kv_cache line with K ({cfg['kv']})")
    out["valid"], out["invalid_reason"] = not why, "; ".join(why)
    return out


def reference_features(rows, tier, probe_seed=PROBE_SEED) -> dict:
    """The full-run reference of one config: every task turn of every completed session (r2a_session present; latest
    attempt per seed) whose seed is not the probe seed. Overflow by detector v2 if the rows carry cut_v2_flagged,
    else the over_loaded_window projection (detector v1, reported, not counted in the agreement)."""
    done = {r.get("attempt_id") for r in rows if r.get("record") == "r2a_session" and _base(r.get("arm_id", "")) == tier}
    by_seed = {}
    for r in rows:
        if (r.get("record") == "r2a_turn" and r.get("attempt_id") in done and r.get("seed") != probe_seed
                and _base(r.get("arm_id", "")) == tier and r.get("item_kind", "task") == "task"):
            by_seed.setdefault(r["seed"], {}).setdefault(r["attempt_id"], []).append(r)
    turns = [t for atts in by_seed.values() for t in list(atts.values())[-1]]
    v2 = any("cut_v2_flagged" in t for t in turns)
    key = "cut_v2_flagged" if v2 else "over_loaded_window"
    return {"run_end": has_run_end(rows), "seeds": sorted(by_seed), "n": len(turns),
            "seconds": _mean(t["turn_wall_s"] for t in turns),
            "acc": _mean(float(bool(t.get("end_state_ok"))) for t in turns),
            "overflow_rate": _mean(float(bool(t.get(key))) for t in turns),
            "overflow_detector": "v2" if v2 else "v1 projection (over_loaded_window)"}


def ape(pred, actual):
    if actual is None or actual == 0:
        return None
    if pred is None:
        return math.inf
    return abs(pred - actual) / abs(actual) * 100.0


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def score(probe_rows, ref_rows_by_file, configs=CONFIGS) -> dict:
    """Per config: probe, reference, APEs and overflow agreement; medians and the pre-registered verdict.
    probe_rows: {tier: rows of that config's file} (missing tier = no probe); ref_rows_by_file: {repo path: rows}
    (missing = no reference)."""
    per = []
    for cfg in configs:
        p = probe_features(probe_rows.get(cfg["tier"], []), cfg)
        rrows = ref_rows_by_file.get(cfg["ref"])
        r = reference_features(rrows, cfg["tier"]) if rrows is not None else None
        if r is not None and (not r["run_end"] or not r["n"]):
            r = None
        e = {"cfg": cfg, "probe": p, "ref": r, "scored": r is not None}
        if r is not None:
            e["err_s"] = ape(p["seconds"] if p["valid"] else None, r["seconds"])
            e["err_acc"] = ape(p["acc"] if p["valid"] else None, r["acc"])
            if r["overflow_detector"] == "v2" and p["valid"]:
                e["overflow_agree"] = (p["flagged"] > 0) == (r["overflow_rate"] > 0)
        per.append(e)
    sc = [e for e in per if e["scored"]]
    med_s = _med([e["err_s"] for e in sc])
    agree = [e["overflow_agree"] for e in sc if "overflow_agree" in e]
    return {"configs": per, "n_scored": len(sc), "med_s": med_s, "med_acc": _med([e["err_acc"] for e in sc]),
            "met": med_s is not None and med_s < TARGET_PCT, "agree": sum(agree), "agree_n": len(agree)}


def _p(x):
    return "n/a" if x is None else ("inf" if x == math.inf else f"{x:.1f}%")


def _f(x, fmt="{:.2f}"):
    return "n/a" if x is None else fmt.format(x)


def probe_text(d) -> str:
    parts = []
    for e in d["configs"]:
        c, p, r = e["cfg"], e["probe"], e["ref"]
        head = f"{c['kv']} {c['tier']}"
        pt = (f"probe {p['n_scored']} task turns in 600 s ({p['n_turns_run']} run"
              + (", stopped by the budget" if p["stopped_by_budget"] else "") + f"), {_f(p['seconds'])} s/task, "
              f"accuracy {_f(p['acc'], '{:.3f}')}, detector v2 flagged {p['flagged']}/{p['n_scored']}, loaded "
              f"{p['loaded']}, KV line {'K (' + c['kv'] + ') present' if any('K (' + c['kv'] + ')' in x for x in p['kv_lines']) else 'missing'}"
              + ("" if p["valid"] else f", INVALID ({p['invalid_reason']})"))
        if r is None:
            parts.append(f"{head}: {pt}; no reference ({c['ref']} missing, without run_end or without held-out "
                         f"sessions), not scored")
            continue
        ov = (f"overflow agree {e['overflow_agree']}" if "overflow_agree" in e else
              f"overflow not compared ({r['overflow_detector']} reference, {_f(r['overflow_rate'], '{:.3f}')})")
        parts.append(f"{head}: {pt}; reference {r['n']} task turns (seeds {r['seeds']}) {_f(r['seconds'])} s/task, "
                     f"accuracy {_f(r['acc'], '{:.3f}')}, overflow rate {_f(r['overflow_rate'], '{:.3f}')} "
                     f"({r['overflow_detector']}); seconds APE {_p(e['err_s'])}, accuracy APE {_p(e['err_acc'])}; {ov}")
    return (f"measured probe on evo-x2 (x2_own_profiler_probe_v1): qwen3:14b, B3 v2 tasks, seed {PROBE_SEED}, at most "
            f"600 s of session per config, no ratios; " + "; ".join(parts)
            + f"; scored configs {d['n_scored']}/{len(d['configs'])}: seconds per task median APE {_p(d['med_s'])}, "
            f"accuracy median APE {_p(d['med_acc'])} (secondary); overflow agreement (detector-v2 references) "
            f"{d['agree']}/{d['agree_n']}; pre-registered target seconds median APE < 10%: "
            + ("MET" if d["met"] else "MISSED") + "; own-profiler-loo (ratio prediction, 70.2%) stays as the recorded miss")


# ── hours ESTIMATE (row own-profiler-probe-hours) ────────────────────────────────────────────────────

def hours_estimate(b3v2_rows, configs=CONFIGS) -> dict:
    """ESTIMATE per config and in total (hours). Session: point = min(600 s, 30 x the tier's B3 v2 mean turn_wall_s)
    (q8_0 / q4_0: the 16384 tier's, as own_compaction.hours_estimate does for the config probes); high = 600 s + the
    slowest single B3 v2 turn of the tier (q8_0 / q4_0: of any f16 tier), since the stop rule can overrun by one turn.
    Plus OVERHEAD_S per config."""
    walls = {}
    for r in b3v2_rows:
        if r.get("record") == "r2a_turn" and r.get("item_kind") == "task":
            walls.setdefault(_base(r["arm_id"]), []).append(float(r["turn_wall_s"]))
    any_max = max(max(v) for v in walls.values())
    per = {}
    for c in configs:
        w = walls.get(c["tier"]) or walls["ollama_ctx_16384"]
        mx = max(walls[c["tier"]]) if c["tier"] in walls else any_max
        per[c["tier"]] = {"point_h": (min(PROBE_BUDGET_S, SESSION_TURNS * statistics.mean(w)) + OVERHEAD_S["point"]) / 3600,
                          "high_h": (PROBE_BUDGET_S + mx + OVERHEAD_S["high"]) / 3600}
    return {"per": per, "point_h": sum(v["point_h"] for v in per.values()),
            "high_h": sum(v["high_h"] for v in per.values())}


# ── job ──────────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rules-from", default=RULES_FROM)
    ap.add_argument("--check-plan", action="store_true")
    args = ap.parse_args(argv)
    import x2_r2_evalset as ev
    assert ev.PROBE_WALL_S == PROBE_BUDGET_S and set(ev.OWN_PROFILER_TIERS) == {c["tier"] for c in CONFIGS}
    assert all(ev.OWN_PROFILER_TIERS[c["tier"]] == c["kv"] for c in CONFIGS)
    if args.check_plan:
        out = []
        for c in CONFIGS:
            argv_c = config_argv(c, args.out, args.rules_from)
            out.append({"tier": c["tier"], "kv": c["kv"], "window": c["window"], "argv": argv_c,
                        "check": ev.main(argv_c + ["--check-plan"], advance=False)})
        print(json.dumps({"configs": out, "budget_s_per_config": PROBE_BUDGET_S, "seed": PROBE_SEED,
                          "evalset_file": ev.__file__, "own_compaction_file": ev.OCX.__file__}, indent=1))
        return "checked"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def utc():
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    note = "completed"
    try:
        emit({"record": "run_start", "harness": "x2_own_profiler_probe", "configs": CONFIGS, "seed": PROBE_SEED,
              "budget_s_per_config": PROBE_BUDGET_S, "rules_from": args.rules_from,
              "harness_blob": ev.git_blob_sha1(__file__), "evalset_blob": ev.git_blob_sha1(ev.__file__),
              "ts_utc": utc()})
        notes = []
        for c in CONFIGS:
            f = config_file(out_path, c["tier"])
            prior = ev.ag.read_rows(f) if f.exists() else []
            if has_run_end(prior):
                emit({"record": "probe_config", "tier": c["tier"], "kv": c["kv"], "file": str(f),
                      "note": "skipped (file already has run_end)", "ts_utc": utc()})
                continue
            t0 = time.monotonic()
            n = ev.main(config_argv(c, out_path, args.rules_from), advance=False)
            rows = ev.ag.read_rows(f) if f.exists() else []
            emit({"record": "probe_config", "tier": c["tier"], "kv": c["kv"], "window": c["window"], "file": str(f),
                  "note": n, "run_end": has_run_end(rows), "wall_s": time.monotonic() - t0, "ts_utc": utc()})
            notes.append(f"{c['tier']}: {n}")
        bad = [x for x in notes if not x.endswith(": completed")]
        if bad:
            note = ("completed with stops: " + "; ".join(bad))[:400]
        emit({"record": "run_end", "note": note, "ts_utc": utc()})
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
        print(note, flush=True)
    finally:
        try:
            import t2s_queue as tq
            tq.advance(note)
        except Exception as e:
            print(f"queue advance failed: {e!r}", flush=True)
    return note


if __name__ == "__main__":
    main()
