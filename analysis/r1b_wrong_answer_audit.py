"""R1b evaluation audit: per-probe correctness grid, refusal dumps, self-report rescoring, and truncation-awareness
check, run against a t2s_night2.py R1b/R1d result JSONL (either machine).

Usage: py -3.12 analysis/r1b_wrong_answer_audit.py <t2s_jsonl> <x2_jsonl>

Why this exists rather than trusting the stored `score` field directly: the self-report arm's on-disk score can be
stale if the process that wrote it held an in-memory copy of evaluation/probes/scorers.py from before a scoring fix
was deployed (a running Python process never re-reads a module from disk). This script always rescores the
self-report arm from the raw `output` string via the current scorers.score_exact + strip_available_prefix, and
reports both the stored score and the rescored one side by side, so a stale in-process scorer is caught rather than
silently trusted. See docs/FINDINGS.md, "R1b evaluation audit, both machines, from the live Oct-1-cut runs
(2026-09-30)" for the write-up this script produced.
"""
import json
import re
import sys
import collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "probes"))
import scorers as sc  # noqa: E402

REFUSAL_RE = re.compile(
    r"\b(i (can'?t|cannot|don'?t have|do not have|am unable)|no [a-z ]{0,40} (was|were|is|are) (not )?"
    r"(provided|found|available|recorded|given|specified|mentioned)|"
    r"not (provided|available|found|recorded|specified) in the (context|text|document|log|passage)|"
    r"(does not|doesn'?t) (include|contain|mention|specify|provide)|"
    r"unable to determine|insufficient (information|context))\b", re.I)

TRUNCATION_AWARE_RE = re.compile(
    r"\b(truncat|incomplete|cut off|cut short|appears to (be )?(cut|end)|seems (incomplete|cut)|"
    r"input (looks|appears|seems) (incomplete|truncated|cut))\b", re.I)


def load_r1b_rows(path):
    """Every real R1b art_* call row (baseline or self_report arm), skipping smoke-test rows and R1d/R1c rows
    that reuse the same file (they carry sea_*/rag_* probe ids and LATE/EARLY arms, not art_*/arm1_baseline or
    arm3_self_report, so the probe_id/arm filter below excludes them without needing a separate phase marker)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "call" and str(r.get("probe_id", "")).startswith("art_") \
               and r.get("arm") in ("arm1_baseline", "arm3_self_report"):
                rows.append(r)
    return rows


def classify_refusal_or_other(output):
    if output is None or not str(output).strip():
        return "refusal_empty"
    if REFUSAL_RE.search(str(output)):
        return "refusal"
    return "other_wrong"


def extract_want_from_score_detail(score_detail):
    """score_detail's wrong-answer format is "got='...' want='...'" (see evaluation/probes/scorers.py
    score_exact); this pulls the want= value back out so a row can be rescored without needing the original
    probe's expected value threaded through separately."""
    if not score_detail or "want=" not in score_detail:
        return None
    tail = score_detail.split("want=", 1)[1]
    if len(tail) >= 2 and tail[0] in ("'", '"'):
        quote = tail[0]
        inner = tail[1:]
        if inner.endswith(quote):
            inner = inner[:-1]
        return inner
    return tail


def rescore_self_report(rows):
    """Returns {ratio: (old_mean, new_mean, n, n_changed)} using the CURRENT (imported) scorers module, regardless
    of what the row's own stored `score` field says -- this is the check for a stale in-process scorer."""
    by_ratio_old = collections.defaultdict(list)
    by_ratio_new = collections.defaultdict(list)
    changed = collections.Counter()
    for r in rows:
        if r.get("arm") != "arm3_self_report":
            continue
        old_score = r.get("score") or 0.0
        if old_score == 1.0:
            new_score = 1.0
        else:
            want = extract_want_from_score_detail(r.get("score_detail"))
            if want is None:
                continue
            new_score, _detail = sc.score_exact(sc.strip_available_prefix(r.get("output")), want)
        ratio = r["budget_ratio"]
        by_ratio_old[ratio].append(old_score)
        by_ratio_new[ratio].append(new_score)
        if new_score != old_score:
            changed[ratio] += 1
    return {ratio: (sum(by_ratio_old[ratio]) / len(by_ratio_old[ratio]),
                    sum(by_ratio_new[ratio]) / len(by_ratio_new[ratio]),
                    len(by_ratio_old[ratio]), changed[ratio])
            for ratio in sorted(by_ratio_old)}


def per_probe_grid(rows, arm="arm1_baseline"):
    """{probe_id: {ratio: (mean_score, mean_artifact_fraction_retained)}} for one arm."""
    by_probe = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("arm") == arm:
            by_probe[r["probe_id"]][r["budget_ratio"]].append(r)
    out = {}
    for probe, by_ratio in by_probe.items():
        out[probe] = {}
        for ratio, cell in by_ratio.items():
            mean_score = sum(c["score"] for c in cell if c.get("score") is not None) / max(1, len(cell))
            mean_afr = sum(c.get("artifact_fraction_retained") or 0 for c in cell) / max(1, len(cell))
            out[probe][ratio] = (mean_score, mean_afr)
    return out


def refusal_share_by_model(rows):
    """{model_id: (n_refusal, n_wrong)} among score==0.0 rows -- the item-3 pre-planned "does scale change
    fabricate vs refuse" question."""
    by_model = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("score") == 0.0:
            by_model[r["model_id"]][1] += 1
            if classify_refusal_or_other(r.get("output")) in ("refusal", "refusal_empty"):
                by_model[r["model_id"]][0] += 1
    return dict(by_model)


def report(path, label):
    rows = load_r1b_rows(path)
    print(f"\n===== {label}: {len(rows)} R1b art_* rows =====")

    print("\n-- per-probe grid (baseline arm): mean_score/mean_artifact_fraction_retained --")
    grid = per_probe_grid(rows)
    ratios = sorted({ratio for by_ratio in grid.values() for ratio in by_ratio})
    print("probe   " + "".join(f"{r:>10}" for r in ratios))
    for probe in sorted(grid):
        cells = "".join(f"{grid[probe].get(r, (None, None))[0]!s:>5}/{grid[probe].get(r, (None, None))[1]!s:<4}"
                         if r in grid[probe] else f"{'--':>10}" for r in ratios)
        print(f"{probe:8}{cells}")

    print("\n-- refusal-classified rows --")
    refusals = [r for r in rows if r.get("score") == 0.0 and classify_refusal_or_other(r.get("output")) in ("refusal", "refusal_empty")]
    print(f"{len(refusals)} / {len(rows)} total")
    for r in refusals[:10]:
        print(f"  [{r['model_id']} {r['probe_id']} ratio={r['budget_ratio']} arm={r['arm']}] output={r.get('output')!r}")

    print("\n-- refusal share of wrong answers, by model --")
    for model, (ref, tot) in sorted(refusal_share_by_model(rows).items()):
        pct = 100.0 * ref / tot if tot else 0.0
        print(f"  {model}: {ref}/{tot} ({pct:.0f}%)")

    print("\n-- self-report: stored score vs rescored (stale-in-process-scorer check) --")
    for ratio, (old, new, n, changed) in rescore_self_report(rows).items():
        flag = "  <-- MISMATCH, stale in-process scorer" if changed else ""
        print(f"  ratio {ratio}: stored={old:.3f} rescored={new:.3f} n={n} changed={changed}{flag}")

    print("\n-- self-report truncation-awareness --")
    sr = [r for r in rows if r.get("arm") == "arm3_self_report"]
    aware = [r for r in sr if r.get("output") and TRUNCATION_AWARE_RE.search(str(r["output"]))]
    print(f"  {len(aware)} / {len(sr)} outputs mention truncation/incompleteness")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <t2s_jsonl> <x2_jsonl>")
    report(sys.argv[1], "evo-t2s")
    report(sys.argv[2], "evo-x2")


if __name__ == "__main__":
    main()
