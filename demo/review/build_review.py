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
STATUS = {"measured": "measured", "projected": "projected", "simulated": "simulated",
          "priced, not executed": "priced, not executed", "not measured": "not measured",
          views.PENDING_FA: "pending", None: "context"}
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
        t = d["evidence"]["totals"]
        rows.append((f"{name}: {t['calls_surfaced_error']} of {t['calls']} calls surfaced an error",
                     d["evidence"]["file"], "measured"))
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
            (f"recorded cloud redo ({t['cloud_model']}): USD {rd['usd']:.4f}", rid_of(rd["src"]),
             STATUS[rd["label"]]),
            (f"recorded cloud redo seconds: {rd['seconds']:.1f} s" if rd.get("seconds") is not None else
             "recorded cloud redo seconds: not measured", rid_of(rd["src"]), STATUS[rd["label"]]),
            (f"spend cap for the live call: USD {e['status']['cap_usd']:.2f}; live calls per run "
             f"{e['status']['live_calls_allowed']}", "demo/dashboard/escalate.py", "configuration")]
    return rows, e


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
    ok = all(r["status"] == views.PENDING_FA and r["settings"]["kv_precision"]["label"] == views.PENDING_FA
             for r in (q14, l8)) and (q14["settings"]["kv_precision"]["value"], l8["settings"]["kv_precision"][
                 "value"]) == ("q8_0", "q4_0")
    items.append(('"pending flash-attention control" shown on q8_0 (qwen3:14b) and q4_0 (llama3.1:8b)', ok,
                  f"qwen3:14b {q14['settings']['kv_precision']['value']}: {q14['status']}; llama3.1:8b "
                  f"{l8['settings']['kv_precision']['value']}: {l8['status']}"))
    blade = recs[("blade", "qwen3:14b", "r2_agent")]
    ok = blade["status"] == "not measured" and all(s["value"] == "not measured" for s in blade["settings"].values())
    items.append(('"not measured" where there is no data (Blade, every field)', ok, f"status {blade['status']}"))
    shown = json.dumps({"lead": c["lead"], "items": c["items"]})  # what the panel renders
    ok_lead = c["lead"]["text"].startswith("Fixing a wrong answer in the cloud costs about USD 0.02")
    no_fix = (not re.search(r"cloud_correct|\bfixed\b|\d+/\d+ \([\d.]+%\) of these", shown)
              and not re.search(r"cloud correct \d+/\d+", json.dumps(c)))  # nor in the click-through register rows
    items.append(("Cost panel leads with the USD 0.02 line", ok_lead, c["lead"]["text"]))
    items.append(("Cost panel shows no cloud fix rate", no_fix,
                  "no fix-rate field or text in the panel; the clause is withheld in the click-through register row"))
    items.append(("Cost panel shows no router savings", c["router"]["shown"] is False, c["router"]["text"]))
    stub = escalate.Escalator(app.cache.escalation, env={}, ledger=HERE / "generated" / "_unused_ledger.jsonl").run()
    items.append(("Live view shows the recorded redo when not sent live", stub["mode"] == "recorded redo" and
                  stub["label"] == "measured", f"mode {stub['mode']}, USD {stub['usd']:.4f}, label {stub['label']}"))
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
    for st in s["steps"]:
        out.append(r"\storystep{" + str(st["n"]) + "}{" + tex(st["title"]) + "}{leave_step" + str(st["n"]) + "}")
        out.append(r"\begin{itemize}\setlength\itemsep{0pt}")
        for it in st["items"]:
            r_ = rid_of(it.get("src"))
            mark = ""
            if r_:
                if r_ not in refs:
                    refs.append(r_)
                mark = r"\textsuperscript{" + str(refs.index(r_) + 1) + "}"
            lab = STATUS.get(it["label"], it["label"])
            badge = r"\lab{" + tex(lab) + "} " if it["label"] else ""
            out.append(r"\item " + badge + tex(it["text"]) + mark)
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
