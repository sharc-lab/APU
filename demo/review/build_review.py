"""Generate the data fragments for demo/review/demo_review.tex and demo/review/leave_behind.tex.

    py -3.12 demo/review/build_review.py --tests "2440 passed, 14 skipped"

Every number in both PDFs comes from the dashboard's own API (the committed cache, built from
docs/NUMBERS_REGISTER.md and committed result files), from the register itself (verbatim appendix), from the two
cloud ledgers (read-only sums) or from git (scope check). Nothing numeric is typed into the .tex files. Writes
demo/review/generated/*.tex. No network, no cloud call (the escalation check runs the stub path with an empty
environment).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "harness"))

from demo.dashboard import escalate, server, views  # noqa: E402
from proc_util import check_output_hidden  # noqa: E402

GEN = HERE / "generated"
ROUTE_DEFAULT = views.ROUTE_DEFAULT_TOL
STATUS = {"measured": "measured", "projected": "projected", "simulated": "simulated",
          "priced, not executed": "priced, not executed", "not measured": "not measured",
          views.PENDING_FA: "pending", views.BEING_MEASURED: "being measured", None: "context"}
SCOPE_OK = ("demo/", "src/dse/", "tests/test_dashboard", "tests/dashboard_views_smoke.js")
ASCII = {"\u2014": "-", "\u2013": "-", "\u2212": "-", "\u2265": ">=", "\u2264": "<=", "\u00d7": "x", "\u2192": "->",
         "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00b1": "+/-", "\u2248": "~",
         "\u00a0": " ", "\u03bc": "u"}


def tex(s) -> str:
    s = "" if s is None else str(s)
    s = "".join(ASCII.get(c, c) if ord(c) > 127 else c for c in s)
    s = s.encode("ascii", "replace").decode("ascii")
    out = []
    for c in s:
        out.append({"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
                    "^": r"\^{}", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}", "<": r"\textless{}",
                    ">": r"\textgreater{}", "|": r"\textbar{}"}.get(c, c))
    return "".join(out)


def texbreak(s) -> str:
    """tex() in typewriter type with break points after / _ - . : so long ids and paths wrap."""
    t = tex(s)
    for a in ("/", r"\_", "-", ".", ":"):
        t = t.replace(a, a + r"\allowbreak{}")
    return r"\texttt{" + t + "}"


def rid_of(src) -> str:
    if not src:
        return ""
    if src.get("register"):
        return src["register"]
    if src.get("file"):
        return src["file"] + (f":{src['line']}" if src.get("line") else "")
    return ""


def row(*cells) -> str:
    return " & ".join(cells) + r" \\" + "\n"


def numbers_table(rows: list[tuple[str, str, str]], caption: str, label: str) -> str:
    """rows: (value text, register id or file, status) -> a longtable."""
    out = [r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.54\textwidth}>{\raggedright\arraybackslash}"
           r"p{0.25\textwidth}>{\raggedright\arraybackslash}p{0.13\textwidth}}",
           rf"\caption{{{tex(caption)}}}\label{{{label}}}\\", r"\toprule",
           row(r"\textbf{Value shown}", r"\textbf{Register id / file}", r"\textbf{Status}") + r"\midrule",
           r"\endfirsthead", r"\toprule", row(r"\textbf{Value shown}", r"\textbf{Register id / file}",
                                              r"\textbf{Status}") + r"\midrule", r"\endhead"]
    for v, r_, s in rows:
        out.append(row(r"\small " + tex(v), r"{\footnotesize " + texbreak(r_) + "}", r"\small " + tex(s)))
    out += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(out) + "\n"


def get(app, path):
    r = app.handle("GET", path)
    assert r.status == 200, (path, r.status)
    return r.json()


def view_a(app) -> tuple[list, dict]:
    s = get(app, "/api/silent")
    rows = []
    for dev, name in (("cpu", "evo-t2s Intel CPU"), ("igpu", "evo-t2s Intel iGPU")):
        d = s["device_pair"][dev]
        for m in d["models"]:
            rows.append((f"{name}, {m['model']}: loaded window {m['loaded_window']}", d["register_id"], "measured"))
            rows.append((f"{name}, {m['model']}: {m['gated_silent']}/{m['n']} sessions silently lost a rule, "
                         f"{m['silent_after_window']} after the window filled", d["register_id"], "measured"))
        # (2026-10-27, docs/SILENT_SCOPING_20261027.md: the "N of M calls surfaced an error" row is removed; no register
        # row backs it)
    wr = s["wrong_rate"]
    for t in wr["tiers"]:
        rows.append((f"qwen3:14b {t['tier']}: wrong {t['wrong']}/{t['n']} ({t['pct']:.1f}%), silent {t['silent']}, "
                     f"surfaced {t['surfaced']}", wr["register_id"], "measured"))
        if t["excess_points"] is not None:
            rows.append((f"{t['tier']}: excess {t['excess_points']:+.1f} points (95% CI {t['ci_lo']:+.1f} to "
                         f"{t['ci_hi']:+.1f})", wr["register_id"], "measured"))
    for c in s["v1b_per_model"]["cells"]:
        rows.append((f"evo-x2 {c['tier']} {c['model']}: {c['gated_silent']}/{c['n']} silent, "
                     f"{c['silent_after_window']} after window", s["v1b_per_model"]["register_id"], "measured"))
    return rows, s


def view_b(app) -> tuple[list, dict]:
    rows, recs = [], {}
    for dev, model, wl in (("evo-x2", "qwen3:14b", "r2_agent"), ("evo-x2", "llama3.1:8b", "r2_agent"),
                           ("evo-t2s-cpu", "qwen3:14b", "r2_agent"), ("evo-t2s-cpu", "llama3.1:8b", "r2_agent"),
                           ("evo-t2s-igpu", "qwen3:14b", "r2_agent"), ("evo-x2", "qwen3:14b", "b3_tasks"),
                           ("evo-x2", "llama3.1:8b", "single_shot"), ("blade", "qwen3:14b", "r2_agent")):
        r = get(app, f"/api/device/recommend?device={dev}&model={model}&workload={wl}")
        recs[(dev, model, wl)] = r
        head = f"{dev} / {model} / {wl}"
        for k, s in r["settings"].items():
            if s["value"] == "not measured" and k != "window":
                continue
            rows.append((f"{head}: {k} = {s['value']}", rid_of(s.get("src")) or "none", STATUS.get(s["label"],
                                                                                                 s["label"])))
        for o in r["outcome"]:
            rows.append((f"{head}: {o['text']}", rid_of(o.get("src")), STATUS.get(o["label"], o["label"])))
        if r["status"] != "measured":
            rows.append((f"{head}: overall status", ", ".join(r["register_ids"]) or "none",
                         STATUS.get(r["status"], r["status"])))
        d = r.get("decision") or {}
        if d.get("text"):
            rows.append((f"{head}: {d['text']}", ", ".join(d.get("basis") or []) or "none",
                         STATUS.get(d.get("label"), d.get("label")) or "context"))
            for t in d.get("tolerances") or []:
                rows.append((f"{head}: would route local only at tolerance >= {t['tolerance_local']:.1f} points vs "
                             f"{t['model']} (local {t['local']} vs all-cloud {t['cloud']}; {t['basis']})",
                             "hybrid-route-at-start", "measured"))
            rc = d.get("route_cost")
            if rc and not d.get("tolerances"):
                rows.append((f"{head}: route at start: {rc['text']}", rid_of(rc.get("src")) or "none",
                             STATUS.get(rc["label"], rc["label"])))
    return rows, recs


def view_c(app) -> tuple[list, dict]:
    c = get(app, "/api/cost")
    rows = [(c["lead"]["text"], rid_of(c["lead"]["src"]), STATUS[c["lead"]["label"]])]
    rows += [(f"{i['what']}: {i['text']}", rid_of(i["src"]), STATUS.get(i["label"], i["label"])) for i in c["items"]]
    rows.append((c["router"]["text"], rid_of(c["router"]["src"]), "not shown"))
    return rows, c


def view_d(app) -> tuple[list, dict]:
    e = get(app, "/api/escalation")
    t = e["task"]
    loc, rd = t["local"], t["redo"]
    got = next((c["arguments"] for c in loc["tool_calls"] if c["name"] == t["expected_calls"][0]["name"]), {})
    rows = [(f"task {t['task_id']}, {t['local_model']} at {t['tier']}, seed {t['seed']}, turn {t['turn_idx']}",
             rid_of(loc["src"]), "measured"),
            (f"local tool call {t['expected_calls'][0]['name']}: {json.dumps(got)}", rid_of(loc["src"]), "measured"),
            (f"expected: {json.dumps(t['expected_calls'][0]['arguments'])}", rid_of(loc["src"]), "measured"),
            (f"local end state wrong: {loc['end_state_ok'] is False}; HTTP error: {loc['any_http_error']}",
             rid_of(loc["src"]), "measured"),
            (f"spend cap for the live call: USD {e['status']['cap_usd']:.2f}; live calls per run "
             f"{e['status']['live_calls_allowed']}", "demo/dashboard/escalate.py", "configuration")]
    if rd.get("usd") is not None:
        rows.insert(4, (f"recorded full-history redo of this task ({rd.get('model')}): USD {rd['usd']:.4f}, "
                        f"{rd['seconds']:.1f} s", rid_of(rd.get("src")), STATUS.get(rd.get("label"), rd.get("label"))))
    else:
        rows.insert(4, (rd.get("text") or "cost to fix: being measured", "none",
                        STATUS.get(rd.get("label"), rd.get("label"))))
    return rows, e


def v1_figures() -> list[str]:
    """Strings only a v1 redo figure or answer would produce (same rule as tests/test_dashboard_views.py)."""
    rows = [json.loads(l) for l in (REPO / "results" / "cloud_redo_v1.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    s = next(r for r in rows if r.get("record") == "run_end")["summary"]
    out = [f"{s['mean_usd']:.4f}", f"{s['median_usd']:.4f}", f"{s['total_usd']:.4f}"]
    esc = json.loads((REPO / "demo" / "dashboard" / "cache" / "escalation.json").read_text(encoding="utf-8"))
    t = next((r for r in rows if r.get("record") == "cloud_redo_task" and r.get("attempt_id") == esc.get("attempt_id")
              and r.get("turn_idx") == esc.get("turn_idx")), None)
    if t:
        out += [f"{t['usd']:.4f}", f"{t['input_tokens']:,}"] + ([t["cloud_final_text"][:60]]
                                                             if t.get("cloud_final_text") else [])
    return out


def ledger_total(path: Path) -> tuple[float, int] | None:
    if not path.exists():
        return None
    total, n = 0.0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("record") == "alert":
            continue
        total += float(r.get("cost_usd") or 0.0)
        n += 1
    return total, n


def git(*args) -> str:
    return check_output_hidden(["git", *args], cwd=str(REPO), text=True).strip()


def checklist(app, a, recs, c, tests: str, main_ledger, demo_ledger, scope_files) -> list[tuple[str, bool, str]]:
    items = []
    cpu = a["device_pair"]["cpu"]["models"]
    ok = bool(cpu) and all(m["gated_silent"] == m["n"] == 5 and m["loaded_window"] == [4096] for m in cpu)
    items.append(("Silent failure: 5/5 on the Intel CPU default (4096) for both models", ok,
                  "; ".join(f"{m['model']} {m['gated_silent']}/{m['n']} at {m['loaded_window']}" for m in cpu)))
    tiers = {t["tier"]: t for t in a["wrong_rate"]["tiers"]}
    ok = tiers.get("ollama_default", {}).get("pct") == 11.3 and tiers.get("ollama_ctx_4096", {}).get("pct") == 42.0
    items.append(("Silent failure: qwen3:14b wrong-task rate 11.3% to 42.0%, 0 surfaced", ok and all(
        t["surfaced"] == 0 for t in tiers.values()), f"default {tiers['ollama_default']['pct']}%, 4096 "
                                                     f"{tiers['ollama_ctx_4096']['pct']}%"))
    q14, l8 = recs[("evo-x2", "qwen3:14b", "r2_agent")], recs[("evo-x2", "llama3.1:8b", "r2_agent")]
    # The flash-attention rule: pending until b5-fa-control reads; with the verified reading (KV precision, not flash
    # attention) q8_0 / q4_0 are recommended as measured with that attribution.
    read = all(views.fa_control_for(app.cache.register, m) for m in ("qwen3:14b", "llama3.1:8b"))
    want = "measured" if read else views.PENDING_FA
    ok = all(r["status"] == want and r["settings"]["kv_precision"]["label"] == want for r in (q14, l8)) and (
        q14["settings"]["kv_precision"]["value"], l8["settings"]["kv_precision"]["value"]) == ("q8_0", "q4_0")
    if read:
        ok = ok and all(any("KV precision, not flash attention" in c for c in r["caveats"]) for r in (q14, l8))
    items.append(("Flash-attention rule: q8_0 (qwen3:14b) / q4_0 (llama3.1:8b) " +
                  ("measured, attributed to KV precision by b5-fa-control" if read else
                   "labelled pending flash-attention control"), ok,
                  f"qwen3:14b {q14['settings']['kv_precision']['value']}: {q14['status']}; llama3.1:8b "
                  f"{l8['settings']['kv_precision']['value']}: {l8['status']}"))
    blade = recs[("blade", "qwen3:14b", "r2_agent")]
    ok = blade["status"] == "not measured" and all(s["value"] == "not measured" for s in blade["settings"].values())
    items.append(('"not measured" where there is no data (Blade, every field)', ok, f"status {blade['status']}"))
    story = get(app, "/api/story")
    ok_lead = (c["lead"]["label"] == "measured" and c["lead"]["src"] == {"register": "cloud-redo-full-context"}
               and c["lead"]["text"].startswith("Fixing a wrong answer in the cloud costs about USD 0.02"))
    items.append(("Cost panel leads with the measured cost to fix a wrong answer (full-history redo, 2048 stated)",
                  ok_lead and "2048" in (c["lead"].get("note") or ""), c["lead"]["text"]))
    texts = [c["lead"]["text"], c["lead"].get("note") or ""] + [i["text"] for i in c["items"]] + [
        i["text"] for st in story["steps"] for i in st["items"]]
    rates = [t for t in texts if re.search(r"\d% fixed|fixed \d+/\d+", t)]
    ok = bool(rates) and all("Wilson CI" in t and "inherited the local model's earlier executed actions" in t
                             for t in rates)
    items.append(("Fix rate shown only with its 95% Wilson CI and the inherited-actions caveat", ok,
                  f"{len(rates)} place(s) show a fix rate; each carries the CI and the caveat" if ok else
                  "a fix rate without its CI or caveat"))
    money = next(st for st in story["steps"] if st["n"] == 4)
    mains = " ".join(t for st in story["steps"] for t in st["visible"]).lower()
    ok = ("ceiling" not in mains and not any(i["text"].lower().startswith("ceiling") for i in money["items"])
          and set((money.get("tradeoff") or {}).get("featured") or {}) >= {"all_cloud", "naive_local", "hybrid_flag"})
    items.append(("The detector ceiling is never a headline (not on any main screen; in Details only as a side note)",
                  ok, "no main screen mentions it; Details of step 4 keep the trade-off with the ceiling as a dotted "
                      "line / side note"))
    # story rules (operator 2026-10-22)
    words = {st["n"]: sum(len(t.split()) for t in st["visible"]) for st in story["steps"]}
    ok = all(w <= (views.STEP4_WORD_LIMIT if n == 4 else views.STEP_WORD_LIMIT) for n, w in words.items())
    items.append(("Story: about 25 visible words per step (step 4: the operator's three sentences)", ok,
                  "visible words per step " + ", ".join(f"{n}: {w}" for n, w in words.items())))
    bad = re.findall(r"\b(qwen\S*|llama\S*|claude\S*|evo-\S*|num_ctx|window|surfaced|excess|results/\S*)", mains)
    nums = [c for st in story["steps"] for k in ("cards", "tiles", "numbers") for c in st["main"].get(k, [])]
    nums += [b for st in story["steps"] for b in st["main"].get("bars", [])]
    ok = not bad and all(c.get("tag") and c.get("src") for c in nums)
    items.append(("Story: no model tags, ids, paths or jargon on main screens; each number has one tag and a "
                  "VERIFIED source in Details", ok, f"{len(nums)} numbers tagged" + (f"; found {bad}" if bad else "")))
    css = (REPO / "demo" / "dashboard" / "static" / "app.css").read_text(encoding="utf-8")
    big = int(re.search(r"\.s-big\s*\{[^}]*font-size:\s*(\d+)px", css)[1])
    body = int(re.search(r"#story\s*\{[^}]*font-size:\s*(\d+)px", css)[1])
    items.append(("Story: presenter sizes (headline numbers >= 72px, body >= 24px); Details closed by default",
                  big >= 72 and body >= 24 and '<details class="s-details">' in
                  (REPO / "demo" / "dashboard" / "static" / "views.js").read_text(encoding="utf-8"),
                  f"headline numbers {big}px, body {body}px; Details rendered without the open attribute"))
    sl = money["main"]["slider"]
    expect = {0: "cloud", 5: "cloud", 12: None, 20: "local"}
    got = {t: [views.route_at(e, t)["route"] for e in sl["entries"]] for t in (0, 5, 12, 20)}
    row = {t: ["local" if t >= e["tolerance_local"] else "cloud" for e in sl["entries"]] for t in (0, 5, 12, 20)}
    ok = sl["default"] == 5 and got == row and all(expect[t] is None or set(got[t]) == {expect[t]} for t in got)
    items.append(("Accuracy-tolerance slider: default 5; decisions match row hybrid-route-at-start at 0, 5, 12, 20",
                  ok, "; ".join(f"{t}: " + ", ".join(f"{e['machine']} vs {e['model']} {d}" for e, d in
                                                      zip(sl["entries"], got[t])) for t in (0, 5, 12, 20))))
    html = app.handle("GET", "/").body.decode("utf-8")
    js = (REPO / "demo" / "dashboard" / "static" / "views.js").read_text(encoding="utf-8")
    tool = next(st for st in story["steps"] if st["kind"] == "tool")
    ok = ('tasks a day ${TAG("projected")}' in js and 'tasks a day <span class="s-big">' in js
          and "Saves per month at 1,000 tasks a day" in tool["visible"])
    items.append(("Story: every volume-based dollar figure says 'at 1,000 tasks a day' and is tagged projected",
                  ok, "step 4 slider lines and the step 5 card"))
    items.append(("Story: a single Engineering view link (top right)",
                  html.count("Engineering view") == 1 and 'id="story-explore"' not in html,
                  f"{html.count('Engineering view')} link"))
    probe = server.App(app.cache, app.backend, escalator=escalate.Escalator(
        app.cache.escalation, env={}, ledger=HERE / "generated" / "_unused_ledger.jsonl"))
    fb = probe.handle("POST", "/api/fix-now").json()
    money_main = next(st for st in story["steps"] if st["kind"] == "money")["main"]
    items.append(("Story step 4: live button present; without a live call it shows the recorded redo, tagged recorded",
                  money_main.get("live_button") == views.LIVE_BUTTON and fb.get("label") == "recorded"
                  and fb.get("usd") is not None,
                  f"button '{money_main.get('live_button')}'; fallback {fb.get('label')}, USD {fb.get('usd')}"))
    items.append(("Landing page is story mode; the engineering view is behind a link (?explore=1)",
                  'id="eng-link"' in html and 'href="?explore=1"' in html and 'params.get("explore") !== "1"' in js,
                  "story section first, engineering view link in its corner and on the step bar"))
    fr = get(app, "/api/frontier")
    ok = fr["cloud"] == [] and fr["cloud_measured"]["rows"] and all(
        "gpt-" not in json.dumps(get(app, p)) for p in ("/api/meta", "/api/frontier", "/api/recommend"))
    items.append(("Engineering view: no OpenAI stub rows; measured Anthropic cloud rows instead", bool(ok),
                  f"{len(fr['cloud_measured']['rows'])} cloud rows from " + ", ".join(sorted(
                      {r['src']['register'] for r in fr['cloud_measured']['rows']}))))
    meta = get(app, "/api/meta")
    items.append(("Engineering view: unmeasured hardware only in one 'Not measured' line",
                  {h["machine"] for h in meta["hardware_measured"]} == set(meta["machines"]),
                  "not measured: " + ", ".join(h["name"] for h in meta["hardware_not_measured"])))
    ev = get(app, "/api/scenario")["evidence"]
    appjs = (REPO / "demo" / "dashboard" / "static" / "app.js").read_text(encoding="utf-8")
    items.append(("Engineering view: every verbatim register block in a closed Details with a one-sentence summary",
                  bool(ev) and all(e["summary"] and e["verbatim"] for e in ev) and "reg[id].value" not in appjs
                  and '<details class="reg-details" open' not in appjs,
                  f"{len(ev)} register blocks, each summarised above a closed Details"))
    figs = v1_figures()
    bodies = [json.dumps(get(app, p)) for p in ("/api/cost", "/api/story", "/api/escalation", "/api/register")]
    leaks = [f for f in figs for b in bodies if re.search(r"(?<![\w.])" + re.escape(f), b)]
    items.append(("No figure or answer from the v1 cloud redo (results/cloud_redo_v1.jsonl) anywhere", not leaks,
                  f"{len(figs)} v1 figures checked against cost, story, escalation and the register dump" +
                  (f"; found: {leaks}" if leaks else "")))
    items.append(("Cost panel shows no router savings", c["router"]["shown"] is False, c["router"]["text"]))
    stub = escalate.Escalator(app.cache.escalation, env={}, ledger=HERE / "generated" / "_unused_ledger.jsonl").run()
    rd = app.cache.escalation.get("redo") or {}
    items.append(("Live view fallback shows the task's recorded full-history redo (no live call)",
                  stub["mode"] == "recorded redo" and stub["label"] == "measured"
                  and str((rd.get("src") or {}).get("file", "")).startswith("results/cloud_redo_full_context"),
                  f"mode {stub['mode']}, USD {stub['usd']:.4f}, {stub['seconds']:.1f} s, from "
                  f"{(rd.get('src') or {}).get('file')}" if stub.get("usd") is not None else f"mode {stub['mode']}"))
    bad = [f for f in scope_files if not f.startswith(SCOPE_OK)]
    items.append(("Scope: every changed file vs main is in demo/, src/dse/ or the dashboard tests", not bad,
                  f"{len(scope_files)} files changed" + (f"; outside scope: {', '.join(bad)}" if bad else "")))
    m = re.search(r"(\d+) passed", tests or "")
    failed = re.search(r"(\d+) failed", tests or "")
    items.append(("Full test suite", bool(m) and not failed, tests or "not run"))
    items.append(("Cloud spend so far (read-only ledgers)", True,
                  (f"main results/cloud_ledger.jsonl USD {main_ledger[0]:.4f} over {main_ledger[1]} rows"
                   if main_ledger else "main ledger not found") + "; " +
                  (f"demo ledger USD {demo_ledger[0]:.4f} over {demo_ledger[1]} call(s)" if demo_ledger
                   else "demo ledger empty")))
    sys.path.insert(0, str(REPO / "harness"))
    import demo_run as _dr
    live_html = (REPO / "demo" / "live" / "static" / "index.html").read_text(encoding="utf-8")
    items.append(("Live run screen: cut-off signal is Ollama's own truncation / context-shift lines (detector v2, "
                  "re-check passed); detector v1 marked not used (false alarms at default, see "
                  "detector-precision-recall)",
                  _dr.DETECTOR == "v2" and "Detector v1 (token-count check) not used" in live_html,
                  f"demo_run.DETECTOR = {_dr.DETECTOR}"))
    return items


def md_to_tex(md: str) -> str:
    """The demo script (Markdown subset: headings, bullets, code fences, one table) as LaTeX."""
    out, in_code, in_list, table = [], False, False, []

    def inline(s):
        parts = re.split(r"(`[^`]+`)", s)
        return "".join(texbreak(p[1:-1]) if p.startswith("`") and p.endswith("`") else
                       re.sub(r"\\textbackslash\{\}", r"\\textbackslash{}", tex(p)) for p in parts)

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r"\end{itemize}")
            in_list = False

    def flush_table():
        if not table:
            return
        out.append(r"\begin{longtable}{p{0.5\textwidth}p{0.18\textwidth}p{0.24\textwidth}}\toprule")
        for i, cells in enumerate(table):
            if set("".join(cells)) <= set("-: "):
                continue
            out.append(" & ".join((r"\textbf{" + inline(c) + "}") if i == 0 else inline(c) for c in cells) +
                       r" \\" + (r" \midrule" if i == 0 else ""))
        out.append(r"\bottomrule\end{longtable}")
        table.clear()

    for line in md.splitlines():
        if line.strip().startswith("```"):
            close_list()
            out.append(r"\end{verbatim}" if in_code else r"\begin{verbatim}")
            in_code = not in_code
            continue
        if in_code:
            out.append(line.strip())
            continue
        if line.startswith("|"):
            close_list()
            table.append([c.strip() for c in line.strip().strip("|").split("|")])
            continue
        flush_table()
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            close_list()
            out.append(r"\subsection*{" + inline(line[3:]) + "}")
        elif re.match(r"\s*- ", line):
            if not in_list:
                out.append(r"\begin{itemize}\setlength\itemsep{1pt}")
                in_list = True
            out.append(r"\item " + inline(re.sub(r"^\s*- ", "", line)))
        elif re.match(r"\s*\d+\. ", line):
            close_list()
            out.append(r"\par\noindent " + inline(line.strip()))
        elif not line.strip():
            close_list()
            out.append("")
        else:
            out.append(inline(line.strip()))
    close_list()
    flush_table()
    return "\n".join(out) + "\n"


def register_appendix(app, ids: list[str]) -> str:
    out = []
    for i in ids:
        r = app.cache.register.get(i)
        if not r:
            out.append(r"\paragraph{" + tex(i) + "} not in the register.")
            continue
        out.append(r"\paragraph{\texttt{" + tex(i) + "}} (" + tex(r["status"]) + ", n " + tex(r["n"]) + ")")
        out.append(r"{\footnotesize\raggedright " + tex(r["value"]) + r"\par}")
    return "\n\n".join(out) + "\n"


def leave_behind(app) -> str:
    """The five story points with labels; register ids collected into a numbered source list."""
    s = get(app, "/api/story")
    refs: list[str] = []
    out = []

    def mark(src):
        r_ = rid_of(src)
        if not r_:
            return ""
        if r_ not in refs:
            refs.append(r_)
        return r"\textsuperscript{" + str(refs.index(r_) + 1) + "}"

    for st in s["steps"]:  # the main screen of each step, as the audience sees it, with its tag and a source mark
        m = st["main"]
        out.append(r"\storystep{" + str(st["n"]) + "}{" + tex(st["title"]) + "}{leave_step" + str(st["n"]) + "}")
        out.append(r"\begin{itemize}\setlength\itemsep{0pt}")
        for c in m.get("cards", []) + m.get("tiles", []) + m.get("numbers", []):
            text = c["text"] if (not c.get("big") or c["big"] in c["text"]) else f"{c['big']} {c['text']}"
            out.append(r"\item \lab{" + tex(c["tag"]) + "} " + tex(text) + mark(c.get("src")) +
                       "".join(mark(s) for s in c.get("also") or []))
        for b in m.get("bars", []):
            out.append(r"\item \lab{" + tex(b["tag"]) + "} " + tex(f"{b['label']}: {b['value']}% wrong") +
                       mark(b.get("src")))
        if m.get("caption"):
            out.append(r"\item \lab{measured} " + tex(m["caption"]) + mark(m.get("bars", [{}])[0].get("src")))
        if m.get("takeaway"):
            out.append(r"\item " + tex(m["takeaway"]))
        if st["kind"] == "tool":
            for e in m.get("entries", []):
                if e["model"] != m.get("default_tier"):
                    continue
                d = views.route_at(e, m.get("tolerance", ROUTE_DEFAULT))
                out.append(r"\item \lab{measured} " + tex(
                    f"{e['machine']}: " + ("run locally with the recommended setting" if d["route"] == "local"
                                           else "send to the cloud from the start") +
                    f" at an accuracy gap of {ROUTE_DEFAULT} points") + mark({"register": "hybrid-route-at-start"}))
            missing = [x for x in m["machines"] if x not in {e["machine"] for e in m["entries"]}]
            if missing:
                out.append(r"\item \lab{not measured} " + tex(", ".join(missing)))
        out.append(r"\end{itemize}")
        out.append(r"\end{minipage}\end{tabular}\par\medskip")
    srcs = (r"\par\medskip\noindent{\footnotesize\raggedright\textbf{Sources} (row ids in docs/NUMBERS\_REGISTER.md, "
            r"or result file and line): " + "; ".join(f"{i + 1}~{texbreak(r_)}" for i, r_ in enumerate(refs)) +
            r".\par}")
    return "\n".join(out) + "\n", srcs + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="", help='full-suite summary, e.g. "2440 passed, 14 skipped"')
    ap.add_argument("--main-checkout", default=str(REPO.parents[2]) if len(REPO.parents) > 2 else str(REPO))
    a = ap.parse_args(argv)
    GEN.mkdir(exist_ok=True)
    app = server.App()
    rows_a, sa = view_a(app)
    rows_b, recs = view_b(app)
    rows_c, sc = view_c(app)
    rows_d, sd = view_d(app)
    (GEN / "view_a.tex").write_text(numbers_table(rows_a, "View A: every number shown", "tab:a"), encoding="utf-8")
    (GEN / "view_b.tex").write_text(numbers_table(rows_b, "View B: every number shown, for the eight "
                                                  "device / model / workload choices reviewed", "tab:b"),
                                    encoding="utf-8")
    (GEN / "view_c.tex").write_text(numbers_table(rows_c, "View C: every number shown", "tab:c"), encoding="utf-8")
    (GEN / "view_d.tex").write_text(numbers_table(rows_d, "View D: every number shown", "tab:d"), encoding="utf-8")
    main_ledger = ledger_total(Path(a.main_checkout) / "results" / "cloud_ledger.jsonl")
    demo_ledger = ledger_total(escalate.LEDGER)
    scope_files = [f for f in git("diff", "--name-only", "main...HEAD").splitlines() if f]
    items = checklist(app, sa, recs, sc, a.tests, main_ledger, demo_ledger, scope_files)
    ck = [r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.38\textwidth}p{0.07\textwidth}"
          r">{\raggedright\arraybackslash}p{0.47\textwidth}}",
          r"\caption{Review checklist, evaluated by demo/review/build\_review.py against the dashboard API}"
          r"\label{tab:check}\\\toprule",
          row(r"\textbf{Check}", r"\textbf{Result}", r"\textbf{Evidence}") + r"\midrule\endhead"]
    for name, ok, ev in items:
        ck.append(row(tex(name), r"\textbf{PASS}" if ok else r"\textbf{FAIL}", tex(ev)))
    ck += [r"\bottomrule", r"\end{longtable}"]
    (GEN / "checklist.tex").write_text("\n".join(ck) + "\n", encoding="utf-8")
    (GEN / "scope.tex").write_text(r"{\footnotesize\raggedright " + "; ".join(r"\texttt{" + tex(f) + "}"
                                                                          for f in scope_files) + r"\par}" + "\n",
                                   encoding="utf-8")
    (GEN / "script.tex").write_text(md_to_tex((REPO / "demo" / "DEMO_SCRIPT.md").read_text(encoding="utf-8")),
                                    encoding="utf-8")
    ids = sorted(set(views.SILENT_IDS + views.DEVICE_IDS + views.COST_IDS + views.STORY_IDS))
    (GEN / "register.tex").write_text(register_appendix(app, ids), encoding="utf-8")
    lb, srcs = leave_behind(app)
    (GEN / "leave_steps.tex").write_text(lb, encoding="utf-8")
    (GEN / "leave_sources.tex").write_text(srcs, encoding="utf-8")
    head = git("rev-parse", "--short", "HEAD")
    branch = git("branch", "--show-current")
    meta = {"branch": branch, "commit": head, "date": date.today().isoformat(), "tests": a.tests or "not run",
            "fails": sum(1 for _n, ok, _e in items if not ok), "nchecks": len(items)}
    (GEN / "meta.tex").write_text("".join(rf"\newcommand{{\meta{k.capitalize()}}}{{{tex(v)}}}" + "\n"
                                          for k, v in meta.items()), encoding="utf-8")
    print(json.dumps({"checks": [(n, ok) for n, ok, _ in items], **meta}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
