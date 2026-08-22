"""Analyse results/art_truncation.json and write summary to results/art_truncation_analysis.json."""

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).parent.parent
RESULTS = REPO / "results"

rows = json.loads((RESULTS / "art_truncation.json").read_text(encoding="utf-8"))["rows"]

FORMS = ["structured", "narrative", "mixed"]
MODELS = ["qwen3:4b-instruct", "llama3.1:8b"]
RATIOS = [1.20, 0.98, 0.85, 0.40]
ARMS = ["arm1_baseline", "arm3_self_report"]
FILLER_TAIL = "93850"

probe_ids = list(dict.fromkeys(r["probe_id"] for r in rows))
art_form = {r["probe_id"]: r["artifact_form"] for r in rows}


def mean_score_table():
    """Mean score by form × model × arm × ratio."""
    table = {}
    for form in FORMS:
        for model in MODELS:
            for arm in ARMS:
                cell = defaultdict(list)
                for r in rows:
                    if (r["artifact_form"] == form and r["model"] == model
                            and r["arm"] == arm and r["score"] is not None):
                        cell[r["budget_ratio"]].append(r["score"])
                key = f"{form}|{model}|{arm}"
                table[key] = {str(rt): round(sum(v)/len(v), 4) for rt, v in cell.items()}
    return table


def fab_abs_table():
    """Fabrication/abstention/correct among truncated arm1 rows (budget_ratio < 1.0)."""
    table = {}
    for form in FORMS:
        trunc = [r for r in rows if r["artifact_form"] == form
                 and r["arm"] == "arm1_baseline" and r["truncating"]
                 and r["budget_ratio"] < 1.0]
        n = len(trunc)
        if not n:
            continue
        fab = sum(1 for r in trunc if r["outcome"] == "incorrect")
        abs_ = sum(1 for r in trunc if r["outcome"] == "abstained")
        correct = sum(1 for r in trunc if r["outcome"] == "correct")
        table[form] = {
            "n": n, "fabricated": fab, "abstained": abs_, "correct": correct,
            "fab_rate": round(fab / n, 4), "abs_rate": round(abs_ / n, 4),
            "correct_rate": round(correct / n, 4),
        }
    return table


def fab_source_table():
    """For arm1 truncated incorrect rows: did the fabricated value come from the filler tail?"""
    table = {}
    for form in FORMS:
        incorr = [r for r in rows if r["artifact_form"] == form
                  and r["arm"] == "arm1_baseline" and r["truncating"]
                  and r["outcome"] == "incorrect" and r["output"]]
        from_filler = sum(1 for r in incorr if FILLER_TAIL in r["output"])
        from_prior = len(incorr) - from_filler
        table[form] = {
            "n_incorrect": len(incorr),
            "from_filler": from_filler,
            "from_prior": from_prior,
        }
    return table


def arm3_available_table():
    """AVAILABLE:yes rate for arm3 truncated rows, per form."""
    table = {}
    for form in FORMS:
        arm3 = [r for r in rows if r["artifact_form"] == form
                and r["arm"] == "arm3_self_report" and r["truncating"]
                and r["budget_ratio"] < 1.0]
        n = len(arm3)
        if not n:
            continue
        yes = sum(1 for r in arm3 if r.get("available_field") == "yes")
        no_ = sum(1 for r in arm3 if r.get("available_field") == "no")
        none_ = n - yes - no_
        table[form] = {
            "n": n, "yes": yes, "no": no_, "none": none_,
            "yes_rate": round(yes / n, 4),
        }
    return table


def per_probe_collapse():
    """Score at each ratio per probe, arm1 only, both models pooled."""
    table = {}
    for pid in probe_ids:
        cell = defaultdict(list)
        for r in rows:
            if r["probe_id"] == pid and r["arm"] == "arm1_baseline" and r["score"] is not None:
                cell[r["budget_ratio"]].append(r["score"])
        table[pid] = {
            "form": art_form[pid],
            "scores": {str(rt): round(sum(v)/len(v), 4) for rt, v in cell.items()}
        }
    return table


def per_probe_fabricated_values():
    """What did the model output for arm1 truncated incorrect rows, by probe?"""
    table = {}
    for pid in probe_ids:
        incorr = [r for r in rows if r["probe_id"] == pid
                  and r["arm"] == "arm1_baseline" and r["truncating"]
                  and r["outcome"] == "incorrect"]
        if incorr:
            table[pid] = {
                "form": art_form[pid],
                "outputs": list(dict.fromkeys(r["output"] for r in incorr if r["output"]))
            }
    return table


def r098_partial_artifact_detail():
    """At r=0.98, which probes had partial artifact and what happened?"""
    rows098 = [r for r in rows if r["budget_ratio"] == 0.98 and r["arm"] == "arm1_baseline"]
    table = {}
    for pid in probe_ids:
        pid_rows = [r for r in rows098 if r["probe_id"] == pid]
        if not pid_rows:
            continue
        af = pid_rows[0]["artifact_fraction"]
        outcomes = [r["outcome"] for r in pid_rows]
        scores = [r["score"] for r in pid_rows if r["score"] is not None]
        outputs = list(dict.fromkeys(r["output"] for r in pid_rows if r["output"]))
        mean_s = round(sum(scores) / len(scores), 4) if scores else None
        table[pid] = {
            "form": art_form[pid], "artifact_fraction": af,
            "mean_score": mean_s, "outcomes": outcomes, "outputs": outputs,
        }
    return table


ms = mean_score_table()
fa = fab_abs_table()
fs = fab_source_table()
av = arm3_available_table()
pp = per_probe_collapse()
pf = per_probe_fabricated_values()
r98 = r098_partial_artifact_detail()

analysis = {
    "n_rows": len(rows),
    "mean_score_by_form_model_arm_ratio": ms,
    "fab_abs_by_form": fa,
    "fabricated_value_source_by_form": fs,
    "arm3_available_yes_rate_by_form": av,
    "per_probe_scores_arm1": pp,
    "per_probe_fabricated_outputs_arm1": pf,
    "r098_partial_artifact_detail": r98,
}

out = RESULTS / "art_truncation_analysis.json"
out.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
print(f"Written → {out}")

print("\n=== MEAN SCORE (arm1_baseline) by form × model × ratio ===")
for form in FORMS:
    print(f"\n  {form.upper()}")
    for model in MODELS:
        key = f"{form}|{model}|arm1_baseline"
        scores = ms.get(key, {})
        parts = " | ".join(f"r={rt:.2f}:{scores.get(str(rt),'?'):.2f}" for rt in RATIOS)
        print(f"    {model[:18]}: {parts}")

print("\n=== FABRICATION / ABSTENTION (arm1, truncated, r<1.0) ===")
for form in FORMS:
    d = fa.get(form, {})
    print(f"  {form:<12}: n={d.get('n',0)}  "
          f"fab={d.get('fabricated',0)}({d.get('fab_rate',0):.0%})  "
          f"abs={d.get('abstained',0)}({d.get('abs_rate',0):.0%})  "
          f"correct={d.get('correct',0)}({d.get('correct_rate',0):.0%})")

print("\n=== FABRICATED VALUE SOURCE (arm1, truncated, incorrect rows) ===")
for form in FORMS:
    d = fs.get(form, {})
    print(f"  {form:<12}: n_incorrect={d.get('n_incorrect',0)}  "
          f"from_filler={d.get('from_filler',0)}  "
          f"from_prior={d.get('from_prior',0)}")

print("\n=== ARM3 AVAILABLE:yes RATE (truncated, r<1.0) ===")
for form in FORMS:
    d = av.get(form, {})
    print(f"  {form:<12}: yes={d.get('yes',0)}/{d.get('n',0)} ({d.get('yes_rate',0):.0%})")

print("\n=== r=0.98 PARTIAL-ARTIFACT DETAIL (arm1) ===")
for pid in probe_ids:
    d = r98.get(pid, {})
    print(f"  {pid} ({d.get('form','?'):<12}) af={d.get('artifact_fraction',0):.2f}"
          f"  mean_score={d.get('mean_score','?')}  outputs={d.get('outputs',[][:2])}")

print("\n=== PER-PROBE FABRICATED VALUES (arm1, truncated, incorrect) ===")
for pid in probe_ids:
    d = pf.get(pid, {})
    if d:
        print(f"  {pid} ({d['form']}): {d['outputs'][:4]}")
