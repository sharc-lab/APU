"""apu_bench audit one-page report ("bring your own logs" MVP, operator brief 2026-10-29 item H).

From an audit result (apu_bench.audit.run_audit) it writes one page, in Markdown and HTML, with:
  - the hidden failure rate per setting, and whether it rests on the runtime's own cut-off log (ground truth: a server
    log replayed here, or a verdict recorded from that log at run time) or on the chars/4 estimate (PREDICTED);
  - the cloud cost of the same sessions (floor and upper bound; billed when provider usage was recorded), and the
    escalation cost of a setting: its hidden-failure tasks redone in the cloud at the mean cloud cost per task;
  - a recommended setting (window and KV cache type), chosen by the stated rule below.

Recommendation rule (RULE, printed on the page):
  1. Candidates are the settings whose verdicts are ground truth (server_log or recorded_verdict). Estimate-only
     settings are shown but never recommended. If no setting has ground truth, no recommendation is made.
  2. If a cost bound is given (USD per 100 tasks), drop the candidates whose escalation cost per 100 tasks exceeds it.
     A candidate with no computable cost (no token counts in the logs) is kept only when no bound is given.
  3. Pick the lowest hidden failure rate. Ties: lower wrong rate (when outcomes are graded), then lower escalation
     cost, then the larger window, then the config name.
  4. If every candidate was dropped by the bound, the page says so ("none within the bound: run in the cloud").
A hidden failure rate of 0 does not mean the tasks were right: the page prints "wrong without a cut" next to it.

KV cache type per setting: the profile's window entry "kv" if present, else a q8_0 / q4_0 / f16 token in the config
label, else "not recorded" (the runtime default; on Ollama 0.34.4 that is f16).

    py -3.12 -m apu_bench audit ... --onepage [--cost-bound USD_PER_100_TASKS]
    py -3.12 -m apu_bench examples --out <dir>      the one-page report on our own recorded runs (offline)
"""
from __future__ import annotations

import html
import re

RULE = ("Among settings with runtime cut-off ground truth (server log, or verdict recorded from it), within the cost "
        "bound on escalation cost per 100 tasks if one is given: lowest hidden failure rate; ties by lower wrong "
        "rate, then lower escalation cost, then larger window, then config name.")


def kv_of(group: dict, prof: dict) -> str:
    for w in prof.get("windows", []):
        if w.get("window") == group["window"] and w.get("tier") == group["config"] and w.get("kv"):
            return w["kv"]
    m = re.search(r"(q8_0|q4_0|f16)", group["config"])
    return m.group(1) if m else "not recorded"


def setting_rows(res: dict) -> list[dict]:
    prof = res["device_profile"]
    out = []
    for g in res["groups"]:
        c = g.get("cloud_cost") or {}
        per_task = (c["floor_usd"] / g["tasks"]) if c.get("floor_usd") is not None and g["tasks"] else None
        esc100 = (100 * g["hidden_failure_rate"] * per_task
                  if per_task is not None and g["hidden_failure_rate"] is not None else None)
        out.append({"config": g["config"], "window": g["window"], "kv": kv_of(g, prof), "method": g["method"],
                    "ground_truth": g["ground_truth"], "tasks": g["tasks"],
                    "hidden": g["tasks_hidden_failure"], "hidden_rate": g["hidden_failure_rate"],
                    "graded": g["tasks_graded"], "wrong": g["wrong"],
                    "wrong_rate": (g["wrong"] / g["tasks_graded"]) if g["tasks_graded"] else None,
                    "wrong_hidden": g["wrong_with_hidden_cut"], "wrong_no_cut": g["wrong_without_cut"],
                    "floor_usd": c.get("floor_usd"), "upper_usd": c.get("upper_bound_usd"),
                    "billed_usd": c.get("billed_usd"), "token_basis": c.get("token_basis"),
                    "cloud_per_task_usd": per_task, "escalation_usd_per_100": esc100})
    return out


def recommend(rows: list[dict], cost_bound: float | None = None) -> dict:
    cands = [r for r in rows if r["ground_truth"] and r["hidden_rate"] is not None]
    if not cands:
        return {"setting": None, "reason": "no setting has runtime cut-off ground truth; no recommendation"}
    if cost_bound is not None:
        kept = [r for r in cands if r["escalation_usd_per_100"] is not None and r["escalation_usd_per_100"] <= cost_bound]
        if not kept:
            return {"setting": None, "reason": f"none within the bound of USD {cost_bound:.4f} per 100 tasks: run in "
                                               "the cloud"}
        cands = kept

    def key(r):
        return (r["hidden_rate"], r["wrong_rate"] if r["wrong_rate"] is not None else 2.0,
                r["escalation_usd_per_100"] if r["escalation_usd_per_100"] is not None else float("inf"),
                -r["window"], r["config"])
    best = sorted(cands, key=key)[0]
    tied = [r for r in cands if r["hidden_rate"] == best["hidden_rate"]]
    why = f"lowest hidden failure rate ({_pct(best['hidden_rate'])})"
    if len(tied) > 1:
        why += f", tied with {len(tied) - 1} other setting(s), broken by the rule's tie order"
    return {"setting": best, "reason": why}


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _usd(x):
    return "n/a" if x is None else f"{x:.4f}"


def _table(rows):
    hdr = ["config", "window", "KV", "ground truth", "tasks", "hidden failures", "hidden rate", "wrong",
           "wrong with hidden cut", "wrong without a cut", "cloud USD floor", "cloud USD upper", "cloud USD billed",
           "escalation USD per 100 tasks"]
    body = []
    for r in rows:
        gt = r["method"] + ("" if r["ground_truth"] else " (PREDICTED)")
        wrong = f"{r['wrong']}/{r['graded']}" if r["graded"] else "not graded"
        body.append([r["config"], str(r["window"]), r["kv"], gt, str(r["tasks"]), str(r["hidden"]),
                     _pct(r["hidden_rate"]), wrong, str(r["wrong_hidden"]), str(r["wrong_no_cut"]),
                     _usd(r["floor_usd"]), _usd(r["upper_usd"]), _usd(r["billed_usd"]),
                     _usd(r["escalation_usd_per_100"])])
    return hdr, body


def page(res: dict, cost_bound: float | None = None, title: str | None = None, source: str | None = None) -> dict:
    """Returns {"md", "html", "recommendation", "rows"}."""
    prof = res["device_profile"]
    rows = setting_rows(res)
    rec = recommend(rows, cost_bound)
    rt = prof.get("runtime") or {}
    hw = prof.get("hardware") or {}
    title = title or f"APU audit: {prof.get('profile_id')}"
    bases = sorted({b for r in rows for b in (r["token_basis"] or [])})
    facts = [("Device", f"{hw.get('cpu') or 'not recorded'}"),
             ("Runtime", f"{rt.get('name')} {rt.get('version') or ''}".strip()),
             ("Model", (prof.get("model") or {}).get("name") or "not recorded"),
             ("Cloud model priced", res.get("cloud_model") or "none"),
             ("Token basis for cost", ", ".join(bases) if bases else "no token counts: cost not computed"),
             ("Cost bound", "none" if cost_bound is None else f"USD {cost_bound:.4f} per 100 tasks (escalation cost)")]
    if source:
        facts.insert(0, ("Logs", source))
    s = rec["setting"]
    rec_line = (f"window {s['window']}, KV {s['kv']} (config {s['config']}): {rec['reason']}; hidden failure rate "
                f"{_pct(s['hidden_rate'])}, wrong {s['wrong']}/{s['graded']}, escalation USD "
                f"{_usd(s['escalation_usd_per_100'])} per 100 tasks" if s else rec["reason"])
    hdr, body = _table(rows)
    notes = ["Hidden failure: the runtime cut the context (its own cut-off log line) while the client saw HTTP 200 and "
             "no error; a task is a hidden failure when any of its requests is. PREDICTED rows come from the chars/4 "
             "estimate, not a log, and are never recommended.",
             "A hidden failure rate of 0 does not mean the tasks were right: see \"wrong without a cut\".",
             "Cloud cost: the same sessions priced on the committed price tables (floor = best caching, upper = no "
             "caching; billed only when provider usage was recorded). Escalation cost = hidden failure rate x mean "
             "floor cost per task x 100. Where completion tokens were not recorded, cost is input only."]
    md = [f"# {title}", ""] + [f"- **{k}:** {v}" for k, v in facts] + [
        "", f"**Recommended setting.** {rec_line}", "", f"Rule: {RULE}", "",
        "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(b) + " |" for b in body] + [
        ""] + [f"- {n}" for n in notes] + [""]
    e = html.escape
    h = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width, initial-scale=1">', f"<title>{e(title)}</title>",
         "<style>body{font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;"
         "color:#1b1b1b;background:#fff}table{border-collapse:collapse;width:100%;font-size:12px}"
         "th,td{border:1px solid #ccc;padding:3px 6px;text-align:left}th{background:#f2f2f2}"
         ".rec{border-left:4px solid #2a6;padding:6px 10px;background:#f4fbf6}.wrap{overflow-x:auto}"
         "@media print{body{margin:0}}</style></head><body>", f"<h1>{e(title)}</h1>", "<ul>"]
    h += [f"<li><b>{e(k)}:</b> {e(v)}</li>" for k, v in facts]
    h += ["</ul>", f'<p class="rec"><b>Recommended setting.</b> {e(rec_line)}</p>', f"<p><i>Rule:</i> {e(RULE)}</p>",
          '<div class="wrap"><table><thead><tr>' + "".join(f"<th>{e(x)}</th>" for x in hdr) + "</tr></thead><tbody>"]
    h += ["<tr>" + "".join(f"<td>{e(x)}</td>" for x in b) + "</tr>" for b in body]
    h += ["</tbody></table></div>", "<ul>"] + [f"<li>{e(n)}</li>" for n in notes] + ["</ul>", "</body></html>", ""]
    return {"md": "\n".join(md), "html": "\n".join(h), "recommendation": rec, "rows": rows}


# ── our own recorded runs ────────────────────────────────────────────────────────────────────────────

def examples(repo, out_dir, cost_bound: float | None = None) -> dict:
    """The one-page report on (1) the demo detector check v2 (raw Ollama logs replayed) and (2) the own-compaction
    runs (detector v2 verdicts recorded per turn). Offline; reads committed files only."""
    import json
    import sys
    from pathlib import Path
    from apu_bench import adapters as AD, audit as A, profile as P
    repo, out_dir = Path(repo), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prof, _sha, _missing = P.load(repo / "apu_bench" / "profiles" / "evo-x2-qwen3-14b.json")
    rows = [json.loads(x) for x in (repo / "results" / "demo_detector_check_v2.jsonl").read_text(encoding="utf-8")
            .splitlines() if x.strip()]
    reqs = AD.demo_detector_requests(rows, "num_ctx_4096") + AD.demo_detector_requests(rows, "default")
    logd = repo / "results" / "demo_live_logs" / "detector_check_v2"
    logs = {("ollama_ctx_4096", 4096): (logd / "serve_4096.log").read_text(encoding="utf-8", errors="replace"),
            ("ollama_default", 40960): (logd / "serve_default.log").read_text(encoding="utf-8", errors="replace")}
    done = {}
    res = A.run_audit(reqs, prof, logs)
    done["demo_detector_check_v2"] = page(
        res, cost_bound, "APU audit: demo detector check v2 (evo-x2, qwen3:14b)",
        "results/demo_detector_check_v2.jsonl with results/demo_live_logs/detector_check_v2/serve_*.log (replayed)")
    an = str(repo / "analysis")
    if an not in sys.path:
        sys.path.insert(0, an)
    import numbers_register as nr
    ag = nr._r2_agent_module()
    oc = [r for f in nr.OC_PARTS.values() for r in nr._finished_attempt_rows(ag.read_rows(repo / f))]
    res2 = A.run_audit(AD.r2a_turn_requests(oc), prof)
    done["own_compaction_v1"] = page(
        res2, cost_bound, "APU audit: own compaction and client trim (evo-x2, qwen3:14b)",
        "results/x2_own_compaction_v1_a.jsonl and _b.jsonl (detector v2 verdicts recorded per turn; logs not kept)")
    for name, p in done.items():
        (out_dir / f"{name}.md").write_text(p["md"], encoding="utf-8")
        (out_dir / f"{name}.html").write_text(p["html"], encoding="utf-8")
    return done
