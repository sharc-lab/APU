"""The evo-t2s week queue (docs/T2S_WEEK_PLAN.md): build it, dry-run it, load it.

  py -3.12 scripts/t2s_week_queue.py build      # controller: (re)writes queues/t2s_week_queue.json from build_queue()
  py -3.12 scripts/t2s_week_queue.py dry-run    # controller: walks every entry with fakes, no SSH, no machine
  py -3.12 scripts/t2s_week_queue.py inputs     # controller: recomputes every estimate input from results/ files
  python t2s_week_queue.py load t2s_week_queue.json   # ON evo-t2s (deployed flat to C:\\apu\\ovn): appends the
                                                       # entries to queue_state.json as pending

Nothing here spawns a process or opens a connection; `load` only reads and writes queue_state.json through
harness/t2s_queue.py, refuses an id that is already in the queue, and never touches a running entry.

OPERATOR CUT (2026-10-08, docs/T2S_WEEK_PLAN.md): steps 1 to 4 plus the R2 mechanism run are queued; the outcome-table
subset is deferred (DEFERRED below: kept as a spec so its command stays tested, never written to the queue file).

HOUR ESTIMATES. Every estimate is computed in estimates() from ESTIMATE_INPUTS, and every input is a number that
inputs_from_files() recomputes from a committed result file (tests/test_t2s_week_queue.py checks the two agree).
"measured" = the same job shape was timed on evo-t2s; "scaled" = an evo-x2 timing times a measured T2S/X2 ratio
(which ratio is named per entry); "nominal" = no timing exists, a stated allowance.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(REPO / "harness"))

QUEUE_PATH = REPO / "queues" / "t2s_week_queue.json"
import host_config as _hc  # noqa: E402


def host_python(host_key):
    """The host's interpreter as an absolute path: HOSTS python_exe with %USERPROFILE% expanded to that host user's
    profile dir. Absolute because the watchdog launches as SYSTEM, whose %USERPROFILE% is the system profile."""
    import ntpath
    h = _hc.HOSTS[host_key]
    profile = ntpath.join("C:" + ntpath.sep, "Users", h["user"])  # as host_config._this_host_entry builds it
    return h["python_exe"].replace("%USERPROFILE%", profile)


T2S_PY = host_python("EVO-T2S")
ID_PREFIX = "t2s_wk_"
BUDGET_H = 72.0   # "about 2-3 days machine time": the cut line is drawn at 3 days
R2_MODELS = "llama3.1:8b,qwen3:14b"
SUBSET_SEED = 20261007

# name -> (value, source file under results/, how it is computed). Recomputed by inputs_from_files().
ESTIMATE_INPUTS = {
    "x2_val_v2_total_min": (32.1, "x2_r2_validation_v2.jsonl",
                            "first to last ts_utc of the file (llama3.1:8b + qwen3:14b, all validation arms)"),
    "x2_real_4096_llama_min": (6.95, "x2_r2_real_v1.jsonl",
                               "median session minutes, llama3.1:8b, ollama_ctx_4096_call2_notools, 40 turns"),
    "x2_real_4096_qwen14_min": (12.24, "x2_r2_real_v1.jsonl",
                                "median session minutes, qwen3:14b, ollama_ctx_4096_call2_notools, 40 turns"),
    "x2_real_32768_llama_min": (22.73, "x2_r2_real_v1.jsonl",
                                "median session minutes, llama3.1:8b, ollama_ctx_32768_call2_notools"),
    "x2_real_32768_qwen14_min": (105.57, "x2_r2_real_v1.jsonl",
                                 "median session minutes, qwen3:14b, ollama_ctx_32768_call2_notools"),
    "x2_mech_4096_min": (9.17, "x2_r2_mechanism.jsonl",
                         "session minutes, llama3.1:8b, ollama_ctx_4096 (OLLAMA_DEBUG=1, render checks)"),
    "t2s_cpu_call_s": (49.27, "t2s_outcome_table_full.jsonl",
                       "median latency_s of HTTP-200 ollama_default rows (evo-t2s CPU, 4096 window, 2050 processed)"),
    "t2s_igpu_over_x2_ollama": (3.31, "t2s_outcome_table_full.jsonl + x2_outcome_table_v3.jsonl",
                                "median per-item latency ratio, llama3.1:8b, T2S ollama_igpu_enable / X2 "
                                "ollama_default, HTTP-200 rows on both sides"),
    "t2s_vulkan_over_x2_llama": (1.99, "t2s_outcome_table_full.jsonl + x2_outcome_table_v3.jsonl",
                                 "median per-item latency ratio, llama3.1:8b, T2S llama_server_vulkan / X2 "
                                 "llama_server"),
    "x2_px2_qwen8_min": (32.77, "t2s_night2_20260930T135145Z.jsonl",
                         "first to last ts_utc of PX2_* item rows, qwen3-8b, 9 conditions, evo-x2"),
    "x2_px2_qwen14_min": (35.99, "t2s_night2_20260930T135145Z.jsonl",
                          "first to last ts_utc of PX2_* item rows, qwen3-14b, 9 conditions, evo-x2"),
    "t2s_outcome_min_per_item": (9.98, "t2s_outcome_table_full.jsonl",
                                 "file span minutes / 40 items, llama3.1:8b, all three configs incl. server starts"),
}
# T2S CPU per call vs X2 per call at the same 4096 window (llama3.1:8b, 80 calls per 40-turn session).
CPU_CALLS_PER_SESSION = 80


def _v(name):
    return ESTIMATE_INPUTS[name][0]


def cpu_factor():
    """T2S CPU seconds per call / X2 seconds per call at num_ctx 4096 (llama3.1:8b)."""
    return _v("t2s_cpu_call_s") / (_v("x2_real_4096_llama_min") * 60 / CPU_CALLS_PER_SESSION)


def estimates() -> dict:
    """id -> (hours, basis, how). Order-independent; build_queue() attaches them to entries."""
    cf, gf = cpu_factor(), _v("t2s_igpu_over_x2_ollama")
    llama_cpu = _v("x2_real_4096_llama_min") * cf
    qwen_cpu = _v("x2_real_4096_qwen14_min") * cf
    mech_over = _v("x2_mech_4096_min") / _v("x2_real_4096_llama_min")
    return {
        "preflight": (0.1, "nominal", "read-only checks, a few version calls"),
        "r2_val_cpu": (_v("x2_val_v2_total_min") * cf / 60, "scaled",
                       f"x2_r2_validation_v2 {_v('x2_val_v2_total_min')} min x CPU factor {cf:.2f}"),
        "r2_val_igpu": (_v("x2_val_v2_total_min") * gf / 60, "scaled",
                        f"x2_r2_validation_v2 {_v('x2_val_v2_total_min')} min x iGPU ratio {gf}"),
        "r2_real_cpu": (5 * (llama_cpu + qwen_cpu) / 60, "scaled",
                        f"5 seeds x (llama {llama_cpu:.0f} + qwen3:14b {qwen_cpu:.0f}) min; X2 4096 sessions x CPU "
                        f"factor {cf:.2f}"),
        "r2_real_igpu": (5 * (_v("x2_real_32768_llama_min") + _v("x2_real_32768_qwen14_min")) * gf / 60, "scaled",
                         f"5 seeds x X2 32768 sessions ({_v('x2_real_32768_llama_min')} + "
                         f"{_v('x2_real_32768_qwen14_min')} min) x iGPU ratio {gf}"),
        "r2_mitigation_cpu": (3 * (llama_cpu + qwen_cpu) * mech_over / 60, "scaled",
                              f"3 seeds x CPU 4096 sessions x render/log overhead {mech_over:.2f} (x2_r2_mechanism "
                              f"4096 vs x2_r2_real_v1 4096)"),
        "px2i": ((_v("x2_px2_qwen8_min") + _v("x2_px2_qwen14_min")) * 4 / 9 * _v("t2s_vulkan_over_x2_llama") / 60
                 + 0.1, "scaled",
                 f"X2 PX2 per-condition time x 4 conditions x Vulkan ratio {_v('t2s_vulkan_over_x2_llama')}, "
                 f"+0.1 h smoke and calibration"),
        "outcome_subset": (100 * _v("t2s_outcome_min_per_item") * (1 + 1) / 60 + 1.0, "measured",
                           f"100 items x 2 models x {_v('t2s_outcome_min_per_item')} min per item-model (evo-t2s, "
                           f"t2s_outcome_table_full, same 3 configs) + 1 h canaries; qwen3-8b assumed equal to "
                           f"llama3.1:8b"),
        "r2_mechanism_cpu": (llama_cpu * mech_over / 60, "scaled",
                             f"one llama3.1:8b CPU 4096 session {llama_cpu:.0f} min x overhead {mech_over:.2f}"),
    }


def _r2(mode, runtime_config, out, *extra):
    return [T2S_PY, "t2s_r2_agent.py", "--mode", mode, "--runtime-config", runtime_config, "--out", out, *extra]


# Capabilities of harness/x2_r2_agent.py the R2 entries use (on main since the 2026-10-08 T2S change; the dry run fails
# an R2 entry if the repo's x2_r2_agent.py lacks one, see t2s_r2_agent.missing_capabilities).
X2_HOST = "x2_r2_agent host allowlist + interactive guard"
X2_ENV, X2_TIERS = "x2_r2_agent --server-env", "x2_r2_agent --tiers"
# key -> why it is not queued (operator decision); the spec stays in _all_specs() so its command keeps being checked.
DEFERRED = {"outcome_subset": "operator cut 2026-10-08: the evo-x2 outcome table covers quality; deferred, not queued"}


def _all_specs() -> list[dict]:
    """Every job spec, queued or deferred, in plan order. gate: the id that must be done first (requires_done)."""
    val_cpu, val_igpu = "results/t2s_r2_validation_cpu_v1.jsonl", "results/t2s_r2_validation_igpu_v1.jsonl"
    gates = ["--require-validation-gates", "--per-model-refusal"]
    return [
        {"key": "preflight", "step": 1, "gate": None,
         "cmd": [T2S_PY, "t2s_week_preflight.py"], "out": "results/t2s_week_preflight.jsonl"},
        {"key": "r2_val_cpu", "step": 2, "gate": "preflight",
         "cmd": _r2("validation", "cpu_default", val_cpu, "--models", R2_MODELS, "--call2-tools", "off"),
         "out": val_cpu, "depends": [X2_HOST]},
        {"key": "r2_val_igpu", "step": 2, "gate": "preflight",
         "cmd": _r2("validation", "igpu_enable", val_igpu, "--models", R2_MODELS, "--call2-tools", "off"),
         "out": val_igpu, "depends": [X2_HOST, X2_ENV]},
        {"key": "r2_real_cpu", "step": 2, "gate": "r2_val_cpu",
         "cmd": _r2("real", "cpu_default", "results/t2s_r2_real_cpu_v1.jsonl", "--models", R2_MODELS,
                    "--plan", "strong", "--tiers", "ollama_default", "--call2-tools", "off", "--order", "seed_major",
                    "--rules-from", val_cpu, *gates),
         "out": "results/t2s_r2_real_cpu_v1.jsonl",
         "depends": [X2_HOST, X2_TIERS]},
        {"key": "r2_real_igpu", "step": 2, "gate": "r2_val_igpu",
         "cmd": _r2("real", "igpu_enable", "results/t2s_r2_real_igpu_v1.jsonl", "--models", R2_MODELS,
                    "--plan", "strong", "--tiers", "ollama_default", "--call2-tools", "off", "--order", "seed_major",
                    "--rules-from", val_igpu, *gates),
         "out": "results/t2s_r2_real_igpu_v1.jsonl",
         "depends": [X2_HOST, X2_ENV, X2_TIERS]},
        {"key": "r2_mitigation_cpu", "step": 3, "gate": "r2_val_cpu",
         "cmd": _r2("mitigation", "cpu_default", "results/t2s_r2_mitigation_cpu_v1.jsonl", "--models", R2_MODELS,
                    "--tiers", "ollama_ctx_4096", "--client-trim", "margin=0.05", "--call2-tools", "off",
                    "--rules-from", val_cpu, *gates),
         "out": "results/t2s_r2_mitigation_cpu_v1.jsonl",
         "depends": [X2_HOST, X2_TIERS,
                     "x2_r2_mitigation_v1 flags (--mode mitigation, --client-trim; merged in 4b46765)"]},
        {"key": "px2i", "step": 4, "gate": "preflight",
         "cmd": [T2S_PY, "t2s_night2.py", "--expect-blobs", "expected_blobs.json", "--deadline-h", "4",
                 "--phases", "px2i"],
         "out": "results/t2s_night2_<launch stem>.jsonl (PX2I rows)"},
        {"key": "outcome_subset", "step": None, "gate": "preflight",
         "cmd": [T2S_PY, "t2s_outcome_table.py", "--out", "results/t2s_outcome_subset_v1.jsonl",
                 "--models", "llama3.1:8b,qwen3-8b", "--subset-n", "100", "--subset-seed", str(SUBSET_SEED),
                 "--canary-gate", "--deadline-h", "36"],
         "out": "results/t2s_outcome_subset_v1.jsonl"},
        {"key": "r2_mechanism_cpu", "step": 5, "gate": "preflight",
         "cmd": _r2("mechanism", "cpu_default", "results/t2s_r2_mechanism_cpu_v1.jsonl", "--tiers",
                    "ollama_default", "--call2-tools", "off"),
         "out": "results/t2s_r2_mechanism_cpu_v1.jsonl",
         "depends": [X2_HOST, X2_TIERS]},
    ]


def job_specs() -> list[dict]:
    """The queued jobs in queue order (every spec not in DEFERRED)."""
    return [s for s in _all_specs() if s["key"] not in DEFERRED]


def deferred_specs() -> list[dict]:
    """The deferred jobs (DEFERRED), never written to the queue file."""
    return [s for s in _all_specs() if s["key"] in DEFERRED]


def build_queue() -> list[dict]:
    est = estimates()
    items, cum = [], 0.0
    for spec in job_specs():
        h, basis, how = est[spec["key"]]
        cum += h
        item = {"id": ID_PREFIX + spec["key"], "cmd": spec["cmd"], "status": "pending",
                "note": (f"T2S week step {spec['step']}: est {h:.1f} h ({basis}: {how}); cumulative {cum:.1f} h"
                         + ("; BEYOND the 72 h budget line, see docs/T2S_WEEK_PLAN.md cut options"
                            if cum > BUDGET_H else "")),
                "plan_step": spec["step"], "est_hours": round(h, 2), "est_basis": basis,
                "cum_hours": round(cum, 2), "beyond_budget": cum > BUDGET_H, "expected_out": spec["out"],
                "depends_on_changes": spec.get("depends", [])}
        if spec["gate"]:
            item["gate"] = {"requires_done": ID_PREFIX + spec["gate"]}
        items.append(item)
    return items


# ------------------------------------------------------------------ inputs recomputed from files
def _rows(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _span_min(rows):
    ts = [_ts(r["ts_utc"]) for r in rows if r.get("ts_utc")]
    return (max(ts) - min(ts)).total_seconds() / 60


def session_minutes(rows):
    """{(model, arm): [minutes]} from each heartbeat to its own r2a_session row."""
    start, out = {}, {}
    for r in rows:
        key = (r.get("model_id"), r.get("arm_id"), r.get("seed"))
        if r.get("record") == "heartbeat":
            start[key] = _ts(r["ts_utc"])
        elif r.get("record") == "r2a_session" and key in start:
            out.setdefault(key[:2], []).append((_ts(r["ts_utc"]) - start.pop(key)).total_seconds() / 60)
    return out


def _latency_median(rows, config):
    return st.median(r["latency_s"] for r in rows if r.get("record") == "outcome_row"
                     and r.get("config") == config and r.get("http_status") == 200 and r.get("latency_s"))


def _ratio(t2s_rows, x2_rows, t2s_cfg, x2_cfg):
    x2 = {}
    for r in x2_rows:
        if (r.get("record") == "outcome_row" and r.get("model_id") == "llama3.1:8b" and r.get("config") == x2_cfg
                and r.get("http_status") == 200 and r.get("latency_s")):
            x2.setdefault(r["item_id"], []).append(r["latency_s"])
    ratios = [r["latency_s"] / st.median(x2[r["item_id"]]) for r in t2s_rows
              if r.get("record") == "outcome_row" and r.get("config") == t2s_cfg and r.get("http_status") == 200
              and r.get("latency_s") and r["item_id"] in x2]
    return st.median(ratios)


def _px2_model_minutes(rows, model):
    mine = [r for r in rows if r.get("model_id") == model and str(r.get("item_id", "")).startswith("PX2_")
            and r.get("ts_utc")]
    return _span_min(mine)


def inputs_from_files(results=REPO / "results") -> dict:
    results = Path(results)
    real = session_minutes(_rows(results / "x2_r2_real_v1.jsonl"))
    mech = session_minutes(_rows(results / "x2_r2_mechanism.jsonl"))
    t2s = _rows(results / "t2s_outcome_table_full.jsonl")
    x2ot = _rows(results / "x2_outcome_table_v3.jsonl")
    px2 = _rows(results / "t2s_night2_20260930T135145Z.jsonl")
    sfx = "_call2_notools"
    return {
        "x2_val_v2_total_min": _span_min(_rows(results / "x2_r2_validation_v2.jsonl")),
        "x2_real_4096_llama_min": st.median(real[("llama3.1:8b", "ollama_ctx_4096" + sfx)]),
        "x2_real_4096_qwen14_min": st.median(real[("qwen3:14b", "ollama_ctx_4096" + sfx)]),
        "x2_real_32768_llama_min": st.median(real[("llama3.1:8b", "ollama_ctx_32768" + sfx)]),
        "x2_real_32768_qwen14_min": st.median(real[("qwen3:14b", "ollama_ctx_32768" + sfx)]),
        "x2_mech_4096_min": st.median(mech[("llama3.1:8b", "ollama_ctx_4096" + sfx)]),
        "t2s_cpu_call_s": _latency_median(t2s, "ollama_default"),
        "t2s_igpu_over_x2_ollama": _ratio(t2s, x2ot, "ollama_igpu_enable", "ollama_default"),
        "t2s_vulkan_over_x2_llama": _ratio(t2s, x2ot, "llama_server_vulkan", "llama_server"),
        "x2_px2_qwen8_min": _px2_model_minutes(px2, "qwen3-8b"),
        "x2_px2_qwen14_min": _px2_model_minutes(px2, "qwen3-14b"),
        "t2s_outcome_min_per_item": _span_min(t2s) / 40,
    }


# ------------------------------------------------------------------ dry run (fakes only)
SCRIPT_DIR = REPO / "harness"
ADVANCING_SCRIPTS = {  # script -> the test that proves one tq.advance() on exit
    "t2s_week_preflight.py": "tests/test_t2s_week_queue.py::test_preflight_advances_once",
    "t2s_r2_agent.py": "tests/test_t2s_r2_agent.py::test_main_advances_exactly_once_*",
    "t2s_night2.py": "tests/test_t2s_week_queue.py::test_night2_main_advances_once",
    "t2s_outcome_table.py": "tests/test_t2s_outcome_subset.py::test_main_advances_once_when_queued",
}


def check_entry(item, earlier_ids, x2_result_names) -> dict:
    """Every check for one queue entry, with the harness's own parser and no side effects."""
    import argparse_probe
    problems, info = [], {}
    cmd = item["cmd"]
    if not item["id"].startswith(ID_PREFIX):
        problems.append(f"id {item['id']!r} lacks prefix {ID_PREFIX!r}")
    gate = (item.get("gate") or {}).get("requires_done")
    if gate and gate not in earlier_ids:
        problems.append(f"gate {gate!r} is not an earlier entry")
    if cmd[0] != T2S_PY:
        problems.append(f"interpreter {cmd[0]!r} is not evo-t2s's {T2S_PY!r}")
    script, argv = cmd[1], cmd[2:]
    if "\\" in script or "/" in script:
        problems.append(f"script {script!r} is not a flat C:\\apu\\ovn name")
    if not (SCRIPT_DIR / script).exists():
        problems.append(f"script {script!r} not found in harness/")
    import t2s_week_preflight as pf
    if script not in pf.DEPLOYED_FILES:
        problems.append(f"script {script!r} is not in t2s_week_preflight.DEPLOYED_FILES")
    if script not in ADVANCING_SCRIPTS:
        problems.append(f"script {script!r} has no tq.advance() test")
    joined = " ".join(cmd).lower()
    for bad in ("ritz", "evo-x2", "100.118.33.76"):
        if bad in joined:
            problems.append(f"evo-x2 reference {bad!r} in the command")
    outs = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--out"]
    for o in outs:
        name = Path(o).name
        if not name.startswith("t2s_"):
            problems.append(f"output {o!r} is not t2s_-prefixed")
        if name in x2_result_names or name.startswith("x2_"):
            problems.append(f"output {o!r} collides with an evo-x2 result name")
        if (REPO / o).exists():
            problems.append(f"output {o!r} already exists in results/ (would append to an old file)")
    if script == "t2s_r2_agent.py":
        import t2s_r2_agent as w
        import x2_r2_agent as x2
        rep = w.dry_run(*w.build_parser().parse_known_args(argv), x2)
        info["forwarded_argv"] = rep["forwarded_argv"]
        info["pending_capabilities"] = rep["missing_capabilities"]
        info["parse_ok"] = rep["x2_parse_ok"]
        if rep["missing_capabilities"]:
            problems.append("the repo's x2_r2_agent.py lacks: " + "; ".join(rep["missing_capabilities"]))
        if not rep["x2_parse_ok"]:
            problems.append(f"x2_r2_agent rejects the forwarded argv: {rep['x2_parse_error']}")
        if rep["out_name_problem"]:
            problems.append(rep["out_name_problem"])
    elif script == "t2s_outcome_table.py":
        import t2s_outcome_table as ot
        res = argparse_probe.probe(ot.main, argv)
        info["parse_ok"] = res["ok"]
        if not res["ok"]:
            problems.append(f"t2s_outcome_table rejects argv: {res['error']}")
        else:
            for m in (res["namespace"].get("models") or "").split(","):
                if m and m not in ot.MODEL_MAP:
                    problems.append(f"unknown outcome-table model {m!r}")
    elif script == "t2s_night2.py":
        import t2s_night2 as n2
        res = argparse_probe.probe(n2.main, argv, use_sys_argv=True, prog="t2s_night2.py")
        info["parse_ok"] = res["ok"]
        if not res["ok"]:
            problems.append(f"t2s_night2 rejects argv: {res['error']}")
        else:
            for ph in res["namespace"]["phases"].split(","):
                if ph not in n2.PHASE_FN:
                    problems.append(f"unknown night2 phase {ph!r}")
    elif script == "t2s_week_preflight.py":
        info["parse_ok"] = not argv
        if argv:
            problems.append("t2s_week_preflight.py takes no arguments")
    return {"id": item["id"], "ok": not problems, "problems": problems, **info}


def dry_run(items=None) -> dict:
    items = items if items is not None else json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    x2_names = {p.name for p in (REPO / "results").glob("x2_*")}
    ids, results = [], []
    if len({it["id"] for it in items}) != len(items):
        results.append({"id": "<queue>", "ok": False, "problems": ["duplicate ids"]})
    for it in items:
        results.append(check_entry(it, set(ids), x2_names))
        ids.append(it["id"])
    total = sum(it.get("est_hours", 0) for it in items)
    within = sum(it.get("est_hours", 0) for it in items if not it.get("beyond_budget"))
    return {"ok": all(r["ok"] for r in results), "entries": results, "total_hours": round(total, 1),
            "hours_within_budget": round(within, 1), "budget_hours": BUDGET_H,
            "deferred": {ID_PREFIX + k: v for k, v in DEFERRED.items()},
            "blocked_until_x2_r2_agent_changes": [r["id"] for r in results if r.get("pending_capabilities")],
            "fresh_build_matches_file": items == build_queue()}


# ------------------------------------------------------------------ load (on evo-t2s)
def load(queue_json, tq=None) -> list[str]:
    """Appends every entry of queue_json to queue_state.json as pending. Refuses (no write at all) if any id is
    already present. Never edits an existing entry."""
    if tq is None:
        import t2s_queue as tq
    new = json.loads(Path(queue_json).read_text(encoding="utf-8-sig"))
    cur = tq.read_queue()
    clash = sorted({it["id"] for it in new} & {it["id"] for it in cur})
    if clash:
        raise SystemExit(f"refusing to load: ids already in queue_state.json: {clash}")
    for it in new:
        it["status"] = "pending"
    tq.write_queue(cur + new)
    return [it["id"] for it in new]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "dry-run"
    if cmd == "build":
        QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUEUE_PATH.write_text(json.dumps(build_queue(), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {QUEUE_PATH}")
    elif cmd == "dry-run":
        rep = dry_run()
        print(json.dumps(rep, indent=1))
        return 0 if rep["ok"] and rep["fresh_build_matches_file"] else 1
    elif cmd == "inputs":
        got = inputs_from_files()
        for k, (v, src, how) in ESTIMATE_INPUTS.items():
            print(f"{k:28s} recorded {v:>8} recomputed {got[k]:>10.3f}  {src}: {how}")
    elif cmd == "load":
        ids = load(argv[1])
        print(f"loaded {len(ids)} pending entries: {ids}")
    else:
        raise SystemExit(f"unknown command {cmd!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
