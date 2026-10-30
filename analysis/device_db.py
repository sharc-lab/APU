"""Device database (docs/EXPERIMENTS.md "PRE-REGISTRATION (2026-10-29)", G1): one schema over every existing
device x runtime x model x window x KV result, each value with its source row or file; missing is null, never guessed.

    python analysis/device_db.py          # writes results/device_db.json and docs/DEVICE_DB.md

Schema (FIELDS): device, runtime, version, model, window, kv, backend, measured_ceiling, wrong_rate, s_per_task, plus
workload, metric notes and `sources` (field -> source row or file). wrong_rate is a task wrong rate (B3 v2 tasks,
long-horizon tasks or the profiler probe's B3 v2 tasks) as a fraction with its k/n; R2 sessions carry
r2_gated_silent (k/n of sessions with a gated silent rule failure after the window was exceeded) instead, never a
wrong rate. s_per_task is the mean wall seconds per task turn (R2: per session turn).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = "results/device_db.json"
OUT_MD = "docs/DEVICE_DB.md"
FIELDS = ("device", "runtime", "version", "model", "window", "kv", "backend", "measured_ceiling", "wrong_rate",
          "s_per_task")
B3_TIERS_KV_PROBE = {"ollama_default", "ollama_ctx_16384", "ollama_ctx_8192", "ollama_ctx_4096"}
B5_KV = {"a": "f16", "c": "f16", "d": "f16", "f": "f16", "g": "q8_0", "h": "q4_0", "i": "f16"}
R2_X2_FILES = ["results/x2_r2_real_v1.jsonl", "results/x2_r2_real_v1b.jsonl", "results/x2_r2_v1b_seeds_06_10.jsonl",
               "results/x2_r2_qwen3_8b_seeds_06_10.jsonl"]
R2_X2_VALIDATION = ["results/x2_r2_validation_v2.jsonl", "results/x2_r2_validation_v2b.jsonl"]
T2S_R2 = {"evo-t2s Intel CPU": ("results/t2s_r2_real_cpu_v1.jsonl", "results/t2s_r2_validation_cpu_v1.jsonl",
                                "t2s-r2-real-cpu", "cpu"),
          "evo-t2s Intel iGPU": ("results/t2s_r2_real_igpu_v1.jsonl", "results/t2s_r2_validation_igpu_v1.jsonl",
                                 "t2s-r2-real-igpu", "igpu")}
T2S_B3 = {"evo-t2s Intel CPU": ("results/t2s_evalset_tasks_real_v2_cpu.jsonl", "t2s-evalset-tasks-cpu"),
          "evo-t2s Intel iGPU": ("results/t2s_evalset_tasks_real_v2_igpu.jsonl", "t2s-evalset-tasks-igpu")}
NOT_INCLUDED = ("Blade 14 (separate experiment arm; its spill sweep and validation are in the blade-* rows, not "
                "joined here so the arms are never mixed)",
                "OpenVINO (row xruntime-openvino-ceiling PENDING, awaiting Zach's files)")


def _nr():
    p = str(REPO / "analysis")
    if p not in sys.path:
        sys.path.insert(0, p)
    import numbers_register as nr
    return nr


def _entry(**kw) -> dict:
    e = {f: kw.pop(f, None) for f in FIELDS}
    e["sources"] = kw.pop("sources", {})
    e.update(kw)
    return e


def _rate(k, n) -> dict | None:
    return {"k": k, "n": n, "rate": k / n} if n else None


def _runtime_version(rows) -> str | None:
    rt = [r for r in rows if r.get("record") == "runtime" and r.get("ollama_version")]
    return rt[-1]["ollama_version"] if rt else None


def _turn_wall(rows, attempts) -> float | None:
    w = [r["turn_wall_s"] for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in attempts
         and r.get("turn_wall_s") is not None]
    return statistics.fmean(w) if w else None


def _sessions(nr, rows, modes=(None, "real")):
    return [s for s in nr._r2_agent_module().completed_sessions(rows) if s.get("mode") in modes]


def _find_key(obj, key) -> list:
    """Every value of key anywhere inside a nested dict/list."""
    if isinstance(obj, dict):
        return ([obj[key]] if key in obj else []) + [v for x in obj.values() for v in _find_key(x, key)]
    if isinstance(obj, list):
        return [v for x in obj for v in _find_key(x, key)]
    return []


def b3v2_entries(repo) -> list[dict]:
    nr = _nr()
    ev = nr._harness_mod(repo, "x2_r2_evalset")
    f = nr.X2_EVALSET_TASKS_REAL_V2
    rows = ev.ag.read_rows(repo / f)
    o = ev.tasks_real_v2_outcome(rows, models=(nr.V2_PROTOCOL_MODEL,))
    ver = _runtime_version(rows)
    out = []
    for tier, i in o["models"][nr.V2_PROTOCOL_MODEL]["tiers"].items():
        att = {s["attempt_id"] for s in _sessions(nr, rows) if s["arm_id"].removesuffix("_call2_notools") == tier}
        kv_src = None
        if tier in B3_TIERS_KV_PROBE:
            kv_src = f"file {Path(nr._own_probe_module().config_file(nr.OWN_PROBE_FILE, tier)).as_posix()} KV line (same tier, same tasks)"
        out.append(_entry(device="evo-x2", runtime="ollama", version=ver, model=nr.V2_PROTOCOL_MODEL,
                          window=i["loaded_context_values"][0] if len(i["loaded_context_values"]) == 1 else None,
                          kv="f16" if kv_src else None, wrong_rate=_rate(i["wrong"], i["n_task_turns"]),
                          s_per_task=_turn_wall(rows, att), workload="B3 v2 tasks, 5 seeds", setting=tier,
                          silent_wrong=i["silent_wrong"],
                          sources={"wrong_rate": "row x2-evalset-tasks-real-v2-wrong-rate", "window":
                                   "row x2-evalset-tasks-real-v2-wrong-rate", "version": f"file {f} runtime record",
                                   "s_per_task": f"file {f} r2a_turn turn_wall_s", **({"kv": kv_src} if kv_src else {})}))
    return out


def r2_entries(repo) -> list[dict]:
    nr = _nr()
    sys.path.insert(0, str(repo / "analysis"))
    import r2_real_report as rr
    ag = nr._r2_agent_module()
    groups = [("evo-x2", R2_X2_FILES, R2_X2_VALIDATION, "rows R2-v1b-excess-10seed, R2-qwen3-8b-reversal-10seed",
               None)]
    groups += [(dev, [real], [val], f"row {row}", back) for dev, (real, val, row, back) in T2S_R2.items()]
    out = []
    for dev, files, vals, row, back in groups:
        rep = rr.build_combined([repo / f for f in files], validation_paths=[repo / v for v in vals])
        rows = [r for f in files for r in ag.read_rows(repo / f)]
        ver = _runtime_version(rows)
        sess = _sessions(nr, rows)
        for key, c in sorted(rep["table"].items()):
            tier, model = key.split("|")
            arm = ("ollama_default" if tier == "ollama_default" else f"ollama_ctx_{tier.split('_')[-1]}") \
                + "_call2_notools"
            ss = [s for s in sess if s["model_id"] == model and s["arm_id"] == arm]
            loaded = sorted({v for s in ss for v in (s.get("loaded_context_values") or [])})
            k = sum(1 for e in c["sessions"] if e["after_window_exceeded"])
            out.append(_entry(device=dev, runtime="ollama", version=ver, model=model,
                              window=loaded[0] if len(loaded) == 1 else None, backend=back,
                              s_per_task=_turn_wall(rows, {s["attempt_id"] for s in ss}),
                              workload="R2 40-turn sessions", setting=tier,
                              r2_gated_silent={"k": k, "n": c["n"]},
                              sources={"r2_gated_silent": row, "window": "files " + ", ".join(files)
                                       + " r2a_session loaded_context_values",
                                       "version": "runtime record of " + ", ".join(files),
                                       "s_per_task": "r2a_turn turn_wall_s of " + ", ".join(files),
                                       **({"backend": f"{row} (file name)"} if back else {})}))
    return out


def b5_entries(repo) -> list[dict]:
    nr = _nr()
    ag = nr._r2_agent_module()
    out = []
    for f in (nr.B5_X2_FILE, nr.B5_FA_FILE):
        rows = ag.read_rows(repo / f)
        ver = _runtime_version(rows)
        sess = _sessions(nr, rows, modes=("b5",))
        for model, arm in sorted({(s["model_id"], s["arm_id"]) for s in sess}):
            ss = [s for s in sess if s["model_id"] == model and s["arm_id"] == arm]
            pol = arm.split("_")[1]
            loaded = sorted({v for s in ss for v in (s.get("loaded_context_values") or [])})
            out.append(_entry(device="evo-x2", runtime="ollama", version=ver, model=model,
                              window=loaded[0] if len(loaded) == 1 else None, kv=B5_KV.get(pol),
                              s_per_task=_turn_wall(rows, {s["attempt_id"] for s in ss}),
                              workload="R2 40-turn sessions (B5 policy compare, 3 seeds)", setting=arm,
                              sources={"kv": "row kv-precision-evidence (policies a, i, g, h); harness/policy_compare.py "
                                             "policy env (c, d, f: no KV env)",
                                       "window": f"file {f} r2a_session loaded_context_values",
                                       "version": f"file {f} runtime record", "s_per_task": f"file {f} turn_wall_s"}))
    return out


def xruntime_entries(repo) -> list[dict]:
    nr = _nr()
    ag = nr._r2_agent_module()
    f = nr.XRT_FILE
    rows = ag.read_rows(repo / f)
    builds = sorted({b for r in rows if r.get("record") == "server_start" for b in _find_key(r, "props_build_info")} - {None})
    sw = [r for r in ag.read_rows(repo / nr.SPILL32_FILE) if r.get("record") == "run_start"]
    st = [r for r in rows if r.get("record") == "run_start"][-1]
    cmds = st.get("server_cmds") or {}
    same_exe = bool(sw) and any(sw[-1].get("exe") in (c or []) for c in cmds.values())
    out = []
    for arm in sorted({s["arm_id"] for s in _sessions(nr, rows)}):
        ss = [s for s in _sessions(nr, rows) if s["arm_id"] == arm]
        att = {s["attempt_id"] for s in ss}
        t = [r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in att]
        loaded = sorted({v for s in ss for v in (s.get("loaded_context_values") or [])})
        k = sum(1 for r in t if not r.get("end_state_ok"))
        out.append(_entry(device="evo-x2", runtime="llama-server", version=builds[0] if len(builds) == 1 else None,
                          model="qwen3:14b (GGUF)", window=loaded[0] if len(loaded) == 1 else None,
                          backend="Vulkan" if same_exe else None, wrong_rate=_rate(k, len(t)),
                          s_per_task=_turn_wall(rows, att), workload="B3 v2 tasks, 5 seeds", setting=arm,
                          sources={"wrong_rate": "row xruntime-llamacpp-wrong-rate", "version":
                                   f"file {f} server_start props_build_info", "window": "row xruntime-llamacpp-wrong-rate",
                                   "s_per_task": f"file {f} turn_wall_s",
                                   **({"backend": f"same llama-server.exe as {nr.SPILL32_FILE} run_start exe (row "
                                                  "fh1-amd-qwen3-32b-spill-sweep, Vulkan heaps measured)"}
                                      if same_exe else {})}))
    return out


def probe_entries(repo) -> list[dict]:
    nr = _nr()
    d = nr.own_profiler_probe_data(repo)
    out = []
    for c in d["configs"]:
        p = c.get("probe") or {}
        if not p:
            continue
        kvl = " ".join(p.get("kv_lines") or [])
        back = "ROCm" if "ROCm" in kvl else ("Vulkan" if "Vulkan" in kvl else None)
        n = p.get("n_scored") or 0
        k = round((1 - p["acc"]) * n) if p.get("acc") is not None and n else None
        out.append(_entry(device="evo-x2", runtime="ollama", model="qwen3:14b", window=p.get("loaded", [None])[0],
                          kv=p.get("kv"), backend=back, wrong_rate=_rate(k, n) if k is not None else None,
                          s_per_task=p.get("seconds"), workload="B3 v2 tasks, seed 20260901, <= 600 s probe",
                          setting=p.get("tier"),
                          sources={"wrong_rate": "row own-profiler-probe", "s_per_task": "row own-profiler-probe",
                                   "window": "row own-profiler-probe", "kv": "row own-profiler-probe",
                                   **({"backend": "row own-profiler-probe KV line ("
                                                  + Path(nr._own_probe_module().config_file(nr.OWN_PROBE_FILE, p["tier"])).as_posix() + ")"}
                                      if back else {})}))
    return out


def spill_entries(repo) -> list[dict]:
    nr = _nr()
    sys.path.insert(0, str(repo / "harness"))
    import mx2_qwen32b_spill_sweep as sw
    rows = sw.ag_rows(repo / nr.SPILL32_FILE)
    rs = next((r for r in rows if r.get("record") == "run_start"), {})
    s = sw.summarize(rows, rs.get("threshold_mib") or 1024.0)
    nospill = [n for n, p in s["points"].items() if p["class"] == "no_spill" and p.get("served_n_ctx") == n]
    builds = sorted({(r.get("server_guard") or {}).get("props_build_info") for r in rows
                     if r.get("record") == "sweep_point"} - {None})
    out = [_entry(device="evo-x2", runtime="llama-server", version=builds[0] if len(builds) == 1 else None,
                  model="qwen3-32b (GGUF)", window=max(nospill) if nospill else None, backend="Vulkan",
                  measured_ceiling=max(nospill) if nospill else None, workload="spill sweep 20480 to 40960, -ngl 99",
                  ceiling_note="largest tested n_ctx without spill; no spill point up to it",
                  sources={"measured_ceiling": "row fh1-amd-qwen3-32b-spill-sweep", "version":
                           f"file {nr.SPILL32_FILE} sweep_point props_build_info", "backend":
                           "row fh1-amd-qwen3-32b-spill-sweep (llama-server Vulkan, Vulkan heaps)"})]
    for model, row in (("llama-3.3-70b", "fh1-amd-boundary-llama-3.3-70b"), ("qwen3-32b", "fh1-amd-boundary-qwen3-32b")):
        dd = nr._amd_boundary(repo, model)["detail"]
        lf = dd.get("last_fit") or {}
        out.append(_entry(device="evo-x2", runtime="llama-server", model=f"{model} (GGUF)", backend="Vulkan",
                          measured_ceiling=lf.get("actual"), workload="FH1 AMD spill boundary",
                          first_spill_n_ctx=(dd.get("first_spill") or {}).get("actual"),
                          ceiling_note="last unclamped no-spill n_ctx measured (" + str(lf.get("source")) + ")",
                          sources={"measured_ceiling": f"row {row}", "first_spill_n_ctx": f"row {row}",
                                   "backend": f"row {row} (llama-server Vulkan)"}))
    return out


def longtask_entries(repo) -> list[dict]:
    nr = _nr()
    _lt, xl = nr._longtask_modules()
    rows = nr._finished_attempt_rows(nr._read_jsonl(repo / nr.LONGTASK_LOCAL_FILE))
    out = []
    for a in xl.ARM_ORDER:
        g = xl.task_rows(rows, xl.ARMS[a]["arm_id"])
        if not g:
            continue
        lc = sorted({r.get("loaded_context") for r in g} - {None})
        out.append(_entry(device="evo-x2", runtime="ollama", model="qwen3:14b", window=lc[0] if len(lc) == 1 else None,
                          kv=xl.ARMS[a]["kv"] or ("f16" if a == "ollama_default" else None),
                          wrong_rate=_rate(sum(1 for r in g if not r["correct"]), len(g)),
                          s_per_task=statistics.fmean(r["task_wall_s"] for r in g),
                          workload="long-horizon synthetic tasks, 20 x 3 seeds", setting=a,
                          sources={"wrong_rate": "row long-task-local-accuracy", "window": "row long-task-local-accuracy",
                                   "kv": "harness/x2_longtask.py ARMS (kv_q8_0 env); ollama_default = B5 policy (a), "
                                         "row kv-precision-evidence",
                                   "s_per_task": f"file {nr.LONGTASK_LOCAL_FILE} task_wall_s"}))
    return out


def t2s_b3_entries(repo) -> list[dict]:
    nr = _nr()
    ag = nr._r2_agent_module()
    out = []
    for dev, (f, row) in T2S_B3.items():
        if not (repo / f).exists():
            continue
        rows = ag.read_rows(repo / f)
        refused = [r for r in rows if r.get("record") == "refused_model"]
        sess = _sessions(nr, rows)
        out.append(_entry(device=dev, runtime="ollama", version=_runtime_version(rows),
                          model=", ".join(sorted({r.get("model_id") for r in refused} - {None})) or None,
                          workload="B3 v2 tasks", completed_sessions=len(sess),
                          status="refused at preflight" if refused and not sess else None,
                          sources={"status": f"row {row}; file {f} refused_model"}))
    return out


def build(repo=REPO) -> dict:
    repo = Path(repo)
    entries = (b3v2_entries(repo) + r2_entries(repo) + b5_entries(repo) + xruntime_entries(repo)
               + probe_entries(repo) + spill_entries(repo) + longtask_entries(repo) + t2s_b3_entries(repo))
    for e in entries:   # a value without a source is dropped to null, never kept unsourced
        for f in ("version", "window", "kv", "backend", "measured_ceiling", "wrong_rate", "s_per_task"):
            if e[f] is not None and f not in e["sources"]:
                e[f] = None
    return {"schema": list(FIELDS), "note": " ".join(__doc__.split("\n\n")[2].split()),
            "not_included": list(NOT_INCLUDED), "entries": entries}


def coverage(db: dict) -> dict:
    es = db["entries"]
    return {"n": len(es), "devices": sorted({e["device"] for e in es}), "runtimes": sorted({e["runtime"] for e in es}),
            "models": sorted({e["model"] for e in es if e["model"]}),
            "filled": {f: sum(1 for e in es if e[f] is not None) for f in FIELDS}}


def _fmt(v):
    if v is None:
        return "null"
    if isinstance(v, dict) and "k" in v:
        return f"{v['k']}/{v['n']}" + (f" ({100 * v['rate']:.1f}%)" if "rate" in v else "")
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def write_md(db: dict, path: Path) -> None:
    cov = coverage(db)
    lines = ["# Device Database", "",
             "Generated by `analysis/device_db.py` (also `results/device_db.json`); do not edit by hand. Every value "
             "cites its source row or file in the JSON's `sources`; missing is null, never guessed. Pre-registered in "
             "docs/EXPERIMENTS.md, \"PRE-REGISTRATION (2026-10-29)\", G1.", "",
             db["note"], "",
             f"{cov['n']} entries; devices {', '.join(cov['devices'])}; runtimes {', '.join(cov['runtimes'])}.", "",
             "Not included: " + "; ".join(db["not_included"]) + ".", "",
             "| device | runtime | version | model | setting | workload | window | KV | backend | measured ceiling | "
             "wrong rate | R2 gated silent | s/task |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in db["entries"]:
        lines.append("| " + " | ".join(_fmt(x).replace("|", "/") for x in (
            e["device"], e["runtime"], e["version"], e["model"], e.get("setting"), e.get("workload"), e["window"],
            e["kv"], e["backend"], e["measured_ceiling"], e["wrong_rate"], e.get("r2_gated_silent"),
            e["s_per_task"])) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    db = build(REPO)
    (REPO / OUT_JSON).write_text(json.dumps(db, indent=1, default=str) + "\n", encoding="utf-8")
    write_md(db, REPO / OUT_MD)
    cov = coverage(db)
    print(f"wrote {OUT_JSON} and {OUT_MD}: {cov['n']} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
