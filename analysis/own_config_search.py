"""own-config-search and hybrid-route-at-start-v2 (operator items of 2026-10-23; pre-registered in docs/EXPERIMENTS.md
"own-config-search" and "hybrid-route-at-start-v2").

own-config-search: for the 8192-constrained AMD profile (evo-x2, qwen3:14b, the KV memory f16 uses at num_ctx 8192)
and the Intel profiles (evo-t2s CPU and iGPU), search KV precision (f16 / q8_0 / q4_0) x window x backend for the
setting that keeps the most B3 v2 tasks local at matched accuracy, from MEASURED probes only.
  - memory: f16 KV bytes per token measured from llama.cpp's own line (KV_LOG_FILE); q8_0 / q4_0 bytes per token are
    the ggml block sizes (34 and 18 bytes per 32 values against f16's 64), NOT measured; each probe job records the
    server's own llama_kv_cache line (record kv_cache_log) and replaces the arithmetic once synced.
  - window per KV type: the largest multiple of 256 whose KV fits the f16-at-8192 budget.
  - kept local: route at start on the 3 all-cloud seeds (90 tasks): a task stays local iff its projected task-turn
    prompt (impact_model.projected_task_prompts) <= the window.
  - matched accuracy: the setting's local accuracy on its kept tasks is not below the Ollama-default-window accuracy on
    the same tasks by more than MATCH_TOL_POINTS (5; 0 also shown). Measured only where a B3 v2 run at that exact
    (KV, window) exists; otherwise "not measured" (the default-window outcome of the same tasks is shown as a proxy,
    never counted).
  - backend: on evo-x2 Ollama runs the iGPU via ROCm only (x2-device-detect-mechanism: Vulkan drops the AMD iGPU,
    no opt-out of ROCm), so there is one backend; on evo-t2s the CPU (default 4096) and the iGPU with
    OLLAMA_IGPU_ENABLE=1 (32768) are the two (t2s-r2-real-cpu, t2s-r2-real-igpu); KV quantization on evo-t2s is not
    measured and not staged (B5 there runs policies a to f only, t2s-b5-plan-hours).

hybrid-route-at-start-v2: the replay of hybrid-route-at-start-measured with items 1 and 2 in place; PENDING until the
own-compaction and probe files exist with run_end. Policies are in V2_POLICIES; target >= 50/90 kept local at >= 97%
hybrid accuracy against claude-sonnet-5-5 all-cloud.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KV_LOG_FILE = "results/demo_live_logs/acceptance/serve_8192.log"   # evo-x2 Ollama, Qwen3 14B, 2026-10-23
B3_V2_FILE = "results/x2_evalset_tasks_real_v2.jsonl"
KV_RATIO_FORMAT = {"f16": 1.0, "q8_0": 34 / 64, "q4_0": 18 / 64}    # ggml block bytes vs f16; not measured
BUDGET_CTX = 8192
MATCH_TOL_POINTS = 5
PROBE_FILES = {("q8_0", 15360): "results/x2_own_config_probe_q8_15360.jsonl",
               ("q4_0", 28928): "results/x2_own_config_probe_q4_28928.jsonl"}
OWN_COMPACTION_FILES = ("results/x2_own_compaction_v1_a.jsonl", "results/x2_own_compaction_v1_b.jsonl")
T2S_PROFILES = {"evo-t2s CPU (default)": {"window": 4096, "rows": "t2s-r2-real-cpu",
                                          "b3_file": "results/t2s_evalset_tasks_real_v2_cpu.jsonl"},
                "evo-t2s iGPU (OLLAMA_IGPU_ENABLE=1)": {"window": 32768, "rows": "t2s-r2-real-igpu",
                                                       "b3_file": "results/t2s_evalset_tasks_real_v2_igpu.jsonl"}}
TARGET_KEPT, TARGET_ACC = 50, 0.97
_KV_RE = re.compile(r"llama_kv_cache: size =\s*([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers.*K \((\w+)\)")


def _paths():
    sys.path.insert(0, str(REPO / "analysis"))
    sys.path.insert(0, str(REPO / "harness"))


def kv_line(repo=REPO, path=KV_LOG_FILE) -> dict:
    """The first f16 llama_kv_cache size line of the log: MiB, cells, layers, bytes per token."""
    for ln in (Path(repo) / path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _KV_RE.search(ln)
        if m and m.group(4) == "f16":
            mib, cells = float(m.group(1)), int(m.group(2))
            return {"mib": mib, "cells": cells, "layers": int(m.group(3)), "type": "f16",
                    "bytes_per_token": mib * 1048576 / cells, "line": ln.strip()[:200]}
    raise ValueError(f"no f16 llama_kv_cache line in {path}")


def window_for(kv: str, f16_bpt: float, budget_ctx: int = BUDGET_CTX) -> int:
    budget = f16_bpt * budget_ctx
    return int(budget / (f16_bpt * KV_RATIO_FORMAT[kv]) // 256 * 256)


def projections(repo=REPO, seeds=(20260901, 20260902, 20260903)) -> dict:
    import contextlib
    import io
    _paths()
    import impact_model as im
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for s in seeds:
            for tid, (_i, p) in im.projected_task_prompts(repo, s).items():
                out[(s, tid)] = p
    return out


def finished(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path, encoding="utf-8") as f:
        return any('"run_end"' in ln and json.loads(ln).get("record") == "run_end" for ln in f if ln.strip())


def local_outcomes(path: Path, arm_prefix: str | None = None) -> dict:
    """{(seed, task_id): {"ok", "flag", "flag_v2", "wall_s"}} over task turns of completed sessions of a file."""
    return local_outcomes_rows([json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()], arm_prefix)


def local_outcomes_rows(rows: list, arm_prefix: str | None = None) -> dict:
    done = {(r["arm_id"], r["seed"], r.get("attempt_id")) for r in rows if r.get("record") == "r2a_session"}
    out = {}
    for r in rows:
        if r.get("record") != "r2a_turn" or r.get("item_kind") != "task" or r.get("end_state_ok") is None:
            continue
        if (r["arm_id"], r["seed"], r.get("attempt_id")) not in done:
            continue
        if arm_prefix and not r["arm_id"].startswith(arm_prefix):
            continue
        out[(r["seed"], r["task_id"])] = {"ok": r["end_state_ok"] is True, "flag": bool(r.get("over_loaded_window")),
                                          "flag_v2": bool(r.get("cut_v2_flagged")), "wall_s": r.get("turn_wall_s")}
    return out


def matched(kept, local, default) -> dict:
    n = len(kept)
    if not n:
        return {"n_kept": 0, "acc": None, "default_acc": None, "diff_points": None, "matched_5": False, "matched_0": False}
    a = sum(local[k]["ok"] for k in kept) / n
    d = sum(default[k]["ok"] for k in kept) / n
    return {"n_kept": n, "acc": a, "default_acc": d, "diff_points": 100 * (a - d),
            "matched_5": 100 * (d - a) <= MATCH_TOL_POINTS, "matched_0": a >= d}


def config_search(repo=REPO) -> dict:
    _paths()
    import impact_model as im
    kv = kv_line(repo)
    proj = projections(repo)
    local = im.replay_local_tasks(repo)
    default = local["ollama_default"]
    cands = []
    for k in ("f16", "q8_0", "q4_0"):
        w = window_for(k, kv["bytes_per_token"])
        kept = sorted(t for t in proj if proj[t] <= w and t in default)
        c = {"profile": "evo-x2 8192-constrained", "kv": k, "window": w, "backend": "ROCm (Ollama)",
             "kv_mib": round(w * kv["bytes_per_token"] * KV_RATIO_FORMAT[k] / 1048576, 1),
             "kv_bytes": "measured" if k == "f16" else "ggml block arithmetic (not measured)",
             "kept": len(kept), "n": len(proj)}
        if k == "f16" and w == 8192:
            c.update({"accuracy_source": f"measured ({B3_V2_FILE} ollama_ctx_8192)",
                      **matched(kept, local["ollama_ctx_8192"], default)})
        elif (k, w) in PROBE_FILES and finished(Path(repo) / PROBE_FILES[(k, w)]):
            lo = local_outcomes(Path(repo) / PROBE_FILES[(k, w)])
            kk = [t for t in kept if t in lo]
            c.update({"accuracy_source": f"measured ({PROBE_FILES[(k, w)]})", **matched(kk, lo, default)})
        else:
            m = matched(kept, default, default)
            c.update({"accuracy_source": "NOT MEASURED", "n_acc": len(kept), "proxy_default_acc": m["default_acc"],
                      "matched_5": None, "matched_0": None,
                      "probe": f"x2_own_config_probe_{k.replace('_0', '')}_{w}: B3 v2 tasks, qwen3:14b, KV {k}, num_ctx "
                               f"{w}, seeds 20260901-03 ({PROBE_FILES.get((k, w), 'no file')})"})
        cands.append(c)
    ok = [c for c in cands if c.get("matched_5")]
    best = max(ok, key=lambda c: c["kept"]) if ok else None
    t2s = []
    for name, p in T2S_PROFILES.items():
        kept = sum(1 for v in proj.values() if v <= p["window"])
        meas = finished(Path(repo) / p["b3_file"])
        t2s.append({"profile": name, "kv": "f16", "window": p["window"], "kept": kept, "n": len(proj),
                    "accuracy_source": f"measured ({p['b3_file']})" if meas else
                    f"NOT MEASURED (B3 v2 on evo-t2s staged for the 2026-10-26 window: {p['b3_file']})",
                    "window_row": p["rows"]})
    return {"kv_line": kv, "candidates": cands, "best_measured": best, "intel": t2s, "before": 21}


def config_text(d) -> str:
    kv = d["kv_line"]
    parts = []
    for c in d["candidates"]:
        acc = (f"local accuracy on kept {c['acc'] * 100:.1f}% vs default window {c['default_acc'] * 100:.1f}% on the "
               f"same tasks ({c['diff_points']:+.1f} points; matched at 5 points {c['matched_5']}, at 0 "
               f"{c['matched_0']}); {c['accuracy_source']}" if c.get("acc") is not None else
               f"accuracy {c['accuracy_source']} (proxy, not counted: default-window accuracy on the same tasks "
               f"{(c.get('proxy_default_acc') or 0) * 100:.1f}%); minimal probe: {c.get('probe')}")
        parts.append(f"{c['kv']} window {c['window']} ({c['kv_mib']} MiB KV, {c['kv_bytes']}, {c['backend']}): kept "
                     f"local by projection {c['kept']}/{c['n']}; {acc}")
    best = d["best_measured"]
    b = (f"best measured setting {best['kv']} {best['window']}: kept local {best['kept']}/{best['n']} (before "
         f"{d['before']}/90)" if best else f"no measured setting beyond f16 8192; before {d['before']}/90, after "
         "PENDING the probes")
    intel = "; ".join(f"{t['profile']} window {t['window']} ({t['window_row']}): kept local by projection {t['kept']}/"
                      f"{t['n']}, accuracy {t['accuracy_source']}; KV q8_0/q4_0 on evo-t2s not measured and not staged"
                      for t in d["intel"])
    return (f"search over measured probes, B3 v2 tasks, qwen3:14b, seeds 20260901-03 (90 tasks), route at start by "
            f"projected task-turn prompt; KV budget = f16 at 8192 = {kv['mib']:.0f} MiB ({kv['bytes_per_token'] / 1024:.0f} "
            f"KiB/token, measured: '{kv['line']}', {KV_LOG_FILE}); matched = local accuracy on kept tasks within "
            f"{MATCH_TOL_POINTS} points of the default window on the same tasks / AMD 8192-constrained: "
            + " / ".join(parts) + f"; {b}; backend: Ollama on evo-x2 uses ROCm only (x2-device-detect-mechanism), "
            "windows set by num_ctx per request (window-override-methods) / Intel: " + intel)


# ── hybrid-route-at-start-v2 ─────────────────────────────────────────────────────────────────────────

V2_POLICIES = ("config", "compaction_all", "fit_plain_else_compaction", "fit_plain_else_trim")


def hybrid_v2_policy(policy, tasks, proj, w0, plain, comp, trim, cfg, cfg_w, cloud, esc_usd) -> dict:
    """Pure. Per task (seed, task_id) with cloud claude-sonnet-5-5 rows:
      config:                    projected <= cfg_w: local at the config probe's outcome (over-window flag escalates);
                                 else the cloud from the first call.
      compaction_all:            every task local under own compaction at w0; detector-v2 flag escalates.
      fit_plain_else_compaction: projected <= w0: plain local at w0 (over-window flag escalates); else own compaction
                                 at w0 (detector-v2 flag escalates). Same server and KV; only the client policy differs.
      fit_plain_else_trim:       as above with client trim in place of compaction.
    Escalated tasks: outcome of the claude-sonnet-5-5 all-cloud row of the task, USD = esc_usd (the tier mean redo USD,
    IMPUTED: no redo exists for these histories), labelled. Kept local = tasks whose outcome is local."""
    res = {"n": 0, "kept_local": 0, "correct": 0, "usd": 0.0, "escalated": 0, "routed": 0, "local_correct": 0}
    for k in tasks:
        c = cloud[k]
        if policy == "config":
            src, flagf = (cfg, "flag") if proj[k] <= cfg_w else (None, None)
        elif policy == "compaction_all":
            src, flagf = comp, "flag_v2"
        else:
            other = comp if policy == "fit_plain_else_compaction" else trim
            src, flagf = (plain, "flag") if proj[k] <= w0 else (other, "flag_v2")
        res["n"] += 1
        if src is None:
            res["routed"] += 1
            res["correct"] += c["ok"]
            res["usd"] += c["usd"]
        elif src[k][flagf]:
            res["escalated"] += 1
            res["correct"] += c["ok"]
            res["usd"] += esc_usd
        else:
            res["kept_local"] += 1
            res["correct"] += src[k]["ok"]
            res["local_correct"] += src[k]["ok"]
    res["accuracy"] = res["correct"] / res["n"] if res["n"] else None
    res["usd_task"] = res["usd"] / res["n"] if res["n"] else None
    res["meets_target"] = res["kept_local"] >= TARGET_KEPT and (res["accuracy"] or 0) >= TARGET_ACC
    return res


def hybrid_v2(repo=REPO, cfg_best=None) -> dict:
    _paths()
    import impact_model as im
    files = [Path(repo) / f for f in OWN_COMPACTION_FILES]
    if not all(finished(f) for f in files):
        return {"status": "PENDING", "pending": [str(f.name) for f in files if not finished(f)]}
    rows = [json.loads(ln) for f in files for ln in open(f, encoding="utf-8") if ln.strip()]
    comp = {w: local_outcomes_rows(rows, f"own_compaction_{w}") for w in (4096, 8192)}
    trim = {w: local_outcomes_rows(rows, f"client_trim_{w}") for w in (4096, 8192)}
    local = im.replay_local_tasks(repo)
    ac = im.replay_cloud_tasks(repo)
    redo, _m = im.replay_redo_tasks(repo)
    cloud = ac["models"]["claude-sonnet-5-5"]
    proj = projections(repo)
    cfg = cfg_best or config_search(repo)["best_measured"]
    cfg_out = None
    if cfg and (cfg["kv"], cfg["window"]) in PROBE_FILES:
        cfg_out = local_outcomes(Path(repo) / PROBE_FILES[(cfg["kv"], cfg["window"])])
    elif cfg and cfg["kv"] == "f16":
        cfg_out = local["ollama_ctx_8192"]
    cells = []
    for w in (8192, 4096):
        plain = local[f"ollama_ctx_{w}"]
        t_redo = [v["usd"] for (tr, _s, _t), v in redo.items() if tr == f"ollama_ctx_{w}"]
        esc = statistics.fmean(t_redo)
        tasks = sorted(k for k in proj if k in cloud and k in plain and k in comp[w] and k in trim[w])
        for pol in V2_POLICIES:
            if pol == "config" and (w != 8192 or cfg_out is None):
                continue
            r = hybrid_v2_policy(pol, [k for k in tasks if pol != "config" or k in cfg_out], proj, w, plain,
                                 comp[w], trim[w], cfg_out, cfg["window"] if cfg else None, cloud, esc)
            cells.append({"tier": w, "policy": pol, **r, "esc_usd": esc})
    allc = {"correct": sum(cloud[k]["ok"] for k in proj if k in cloud),
            "usd_task": statistics.fmean(cloud[k]["usd"] for k in proj if k in cloud)}
    return {"status": "VERIFIED", "cells": cells, "all_cloud": allc, "config": cfg}


def hybrid_v2_text(d) -> str:
    if d["status"] != "VERIFIED":
        return "PENDING: " + ", ".join(d["pending"])
    parts = [f"{c['tier']} {c['policy']}: kept local {c['kept_local']}/{c['n']}, accuracy {c['correct']}/{c['n']} "
             f"({100 * c['accuracy']:.1f}%), cloud USD {c['usd_task']:.5f}/task (escalated {c['escalated']} at the "
             f"imputed tier mean redo USD {c['esc_usd']:.5f}, routed {c['routed']}); target met {c['meets_target']}"
             for c in d["cells"]]
    return ("measured by replay of recorded runs, claude-sonnet-5-5, seeds 20260901-03; all-cloud "
            f"{d['all_cloud']['correct']}/90, USD {d['all_cloud']['usd_task']:.5f}/task; target >= {TARGET_KEPT}/90 "
            f"kept local at >= {100 * TARGET_ACC:.0f}% / " + " / ".join(parts))
