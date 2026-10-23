"""Live demo: scenarios, the per-task lane sequence and the tally (pure; runners are injected).

Scenarios (operator decisions of 2026-10-23):
  amd8192  "AMD workstation, constrained memory (8192)", the headline scenario (register row
           hybrid-route-at-start-measured): Default local runs every task at num_ctx 8192; Ours routes per task with
           src/dse/route_at_start on the evo-x2-8192 profile (local at 8192 if the projected task-turn prompt fits,
           else the whole task on claude-sonnet-5-5 from its first call) and escalates a local task whose cut-off
           detector fired; All cloud runs every task on claude-sonnet-5-5. A real run on evo-x2 with a simulated
           device window.
  intel    "Intel laptop default, simulated on AMD": Default local at num_ctx 4096 (the Intel laptop's measured
           default); Ours on the evo-t2s profile (local at 32768, the measured Intel graphics-chip window, labelled
           "Intel graphics-chip setting, simulated": window only, base server, OLLAMA_IGPU_ENABLE is not set on evo-x2;
           else claude-haiku-5-5 from the first call); All cloud on claude-haiku-5-5.

Lanes run sequentially per task: default, ours, cloud. One Ollama server per configuration (server env, num_ctx).
Ours uses the headline row's rule: need = the projected task-turn prompt from the task definition
(demo_live.projected_task_prompt), local iff need <= window (margin 0).

Runner interfaces:
  local(lane, task, num_ctx) -> result dict with end_state_ok, flagged, calls, seconds, ...   (demo_live.run_local)
  cloud(lane, task, model)   -> result dict with end_state_ok, usd_ledger, label LIVE/REPLAYED, or
                                {"pending": True} when this process cannot call the cloud (evo-x2 holds no key; the
                                controller completes those parts, complete_cloud below).
"""
from __future__ import annotations

SCENARIOS = {
    "amd8192": {"label": "AMD workstation, constrained memory (8192)", "device": "evo-x2-8192", "default_ctx": 8192,
                "cloud_model": "claude-sonnet-5-5", "simulated": "simulated device window (8192) on the AMD workstation",
                "default_label": "Default local (memory 8,192)", "setting_labels": {"default": "memory 8,192"},
                "headline_row": "hybrid-route-at-start-measured"},
    "intel": {"label": "Intel laptop default, simulated on AMD", "device": "evo-t2s", "default_ctx": 4096,
              "cloud_model": "claude-haiku-5-5", "simulated": "Intel laptop default, simulated on AMD",
              "default_label": "Default local (Intel laptop default 4,096, simulated on AMD)",
              "setting_labels": {"igpu": "Intel graphics-chip setting, simulated", "default": "memory 4,096"},
              "headline_row": None},
}
LANES = ("default", "ours", "cloud")
NEED_SOURCE = "projected task-turn prompt (hybrid-route-at-start-measured rule)"


def configurations(scenario: str, profile: dict) -> list[dict]:
    """Distinct (server env, num_ctx) pairs the scenario can use locally: the default lane's window and every
    measured local setting of the device (all on the base server env on evo-x2; window-only simulation)."""
    sc = SCENARIOS[scenario]
    dev = profile["devices"][sc["device"]]
    ctxs = [sc["default_ctx"]] + [s["window"] for s in dev["settings"]]
    out, seen = [], set()
    for c in ctxs:
        if c not in seen:
            seen.add(c)
            out.append({"num_ctx": c, "server_env": {}})
    return out


def decide(scenario: str, profile: dict, task: dict, budget_left: float) -> dict:
    import route_at_start as R
    sc = SCENARIOS[scenario]
    d = R.route(profile, sc["device"], task["projected_prompt"], budget_left, margin=0.0, need_source=NEED_SOURCE)
    if d["route"] == "local":
        lab = sc["setting_labels"].get(d["setting"], d.get("setting_label"))
        if lab != d.get("setting_label"):
            d["reason"] = d["reason"].replace(d["setting_label"], lab)
        d["setting_label"] = lab
    return d


def run_task(scenario: str, profile: dict, task: dict, local, cloud, budget_left, emit=lambda ev: None) -> dict:
    """The three lanes on one task, in order. budget_left: callable -> USD left under the demo cap."""
    sc = SCENARIOS[scenario]
    out = {"scenario": scenario, "seed": task["seed"], "task_id": task["task_id"], "turn_idx": task["turn_idx"],
           "projected_prompt": task["projected_prompt"], "history_turns": task["history_turns"], "lanes": {}}
    emit({"event": "lane_start", "lane": "default", **_key(out)})
    r = local("default", task, sc["default_ctx"])
    out["lanes"]["default"] = {**r, "route": "local", "num_ctx": sc["default_ctx"]}
    emit({"event": "lane_end", "lane": "default", "result": out["lanes"]["default"], **_key(out)})

    emit({"event": "lane_start", "lane": "ours", **_key(out)})
    d = decide(scenario, profile, task, budget_left())
    emit({"event": "decision", "lane": "ours", "decision": d, **_key(out)})
    if d["route"] == "local":
        r = local("ours", task, d["window"])
        res = {**r, "route": "local", "num_ctx": d["window"], "decision": d}
        if r.get("flagged"):
            emit({"event": "escalate", "lane": "ours", **_key(out)})
            c = cloud("ours", task, sc["cloud_model"])
            res = {**res, "escalated": True, "local_end_state_ok": r.get("end_state_ok"), "cloud": c,
                   "end_state_ok": c.get("end_state_ok"), "pending": c.get("pending", False)}
    else:
        c = cloud("ours", task, d["model"])
        res = {**c, "route": "cloud", "decision": d}
    out["lanes"]["ours"] = res
    emit({"event": "lane_end", "lane": "ours", "result": res, **_key(out)})

    emit({"event": "lane_start", "lane": "cloud", **_key(out)})
    c = cloud("cloud", task, sc["cloud_model"])
    out["lanes"]["cloud"] = {**c, "route": "cloud"}
    emit({"event": "lane_end", "lane": "cloud", "result": out["lanes"]["cloud"], **_key(out)})
    return out


def _key(o):
    return {"seed": o["seed"], "task_id": o["task_id"], "scenario": o["scenario"]}


def lane_usd(res: dict) -> float:
    if res.get("route") == "cloud":
        return float(res.get("usd_ledger") or 0.0)
    if res.get("escalated"):
        return float((res.get("cloud") or {}).get("usd_ledger") or 0.0)
    return 0.0


def lane_usd_cache(res: dict) -> float:
    if res.get("route") == "cloud":
        return float(res.get("usd_cache_priced") or 0.0)
    if res.get("escalated"):
        return float((res.get("cloud") or {}).get("usd_cache_priced") or 0.0)
    return 0.0


def tally(task_results: list[dict]) -> dict:
    """Per lane: tasks, correct, wrong silent / wrong flagged (the detector's flag on the local call that produced
    the answer), kept local, cloud USD from the ledger (and cache-priced, the headline's basis), seconds, and how many
    cloud results were LIVE vs REPLAYED vs still pending."""
    out = {}
    for lane in LANES:
        rs = [t["lanes"][lane] for t in task_results if lane in t["lanes"]]
        done = [r for r in rs if not r.get("pending") and not r.get("unavailable")
                and not (r.get("cloud") or {}).get("unavailable")]
        wrong = [r for r in done if r.get("end_state_ok") is not True]
        local_answer = [r for r in done if r.get("route") == "local" and not r.get("escalated")]
        n = len(done)
        usd = sum(lane_usd(r) for r in done)
        out[lane] = {
            "tasks": len(rs), "graded": n, "correct": n - len(wrong), "wrong": len(wrong),
            "wrong_flagged": sum(1 for r in wrong if r.get("route") == "local" and not r.get("escalated")
                                 and r.get("flagged")),
            "wrong_silent": sum(1 for r in wrong if not (r.get("route") == "local" and not r.get("escalated")
                                                       and r.get("flagged"))),
            "kept_local": len(local_answer), "escalated": sum(1 for r in done if r.get("escalated")),
            "flagged_calls_tasks": sum(1 for r in done if r.get("flagged")),
            "usd_ledger_total": round(usd, 6), "usd_ledger_per_task": round(usd / n, 6) if n else None,
            "usd_cache_priced_per_task": round(sum(lane_usd_cache(r) for r in done) / n, 6) if n else None,
            "seconds": round(sum(float(r.get("seconds") or 0) + float((r.get("cloud") or {}).get("seconds") or 0)
                                 for r in done), 1),
            "labels": sorted({(r.get("label") or (r.get("cloud") or {}).get("label") or "LIVE") for r in done}),
            "pending": sum(1 for r in rs if r.get("pending")),
            "unavailable": len(rs) - n - sum(1 for r in rs if r.get("pending")),
        }
    return out
