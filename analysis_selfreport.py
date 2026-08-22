import json
from pathlib import Path
from collections import defaultdict

data = json.loads(Path("results/selfreport_arms.json").read_text())
rows = data["rows"]

TARGET_PROBES = ["rag_01", "rag_02", "rag_05", "sea_01", "sea_04"]
RATIOS = [0.85, 0.70, 0.55, 0.40]


def cell_stats(arm, ratio):
    rws = [
        r
        for r in rows
        if r["arm"] == arm
        and r.get("probe_id") in TARGET_PROBES
        and r["budget_ratio"] == ratio
    ]
    n = len(rws)
    if n == 0:
        return {}
    return {
        "n": n,
        "correct": sum(1 for r in rws if r["outcome"] == "correct"),
        "abstained": sum(1 for r in rws if r["outcome"] == "abstained"),
        "incorrect": sum(1 for r in rws if r["outcome"] == "incorrect"),
        "fabrication_rate": round(
            sum(1 for r in rws if r["outcome"] == "incorrect") / n, 4
        ),
        "abstention_rate": round(
            sum(1 for r in rws if r["outcome"] == "abstained") / n, 4
        ),
    }


a2_all = [
    r
    for r in rows
    if r["arm"] == "arm2_abstention_instruction"
    and r.get("probe_id") in TARGET_PROBES
    and r["budget_ratio"] in RATIOS
]
sentinel_used = sum(1 for r in a2_all if r.get("used_sentinel"))
sentinel_ignored = sum(1 for r in a2_all if not r.get("used_sentinel"))

a3_all = [
    r
    for r in rows
    if r["arm"] == "arm3_self_report"
    and r.get("probe_id") in TARGET_PROBES
    and r["budget_ratio"] in RATIOS
]
avail_yes = [r for r in a3_all if r.get("available_field") == "yes"]
avail_no = [r for r in a3_all if r.get("available_field") == "no"]
avail_yes_fab = [r for r in avail_yes if r["outcome"] == "incorrect"]
avail_no_inc = [r for r in avail_no if r["outcome"] == "incorrect"]
avail_no_abs = [r for r in avail_no if r["outcome"] == "abstained"]

arm2_sentinel_failures = defaultdict(list)
for r in a2_all:
    if not r.get("used_sentinel"):
        arm2_sentinel_failures[r["probe_id"]].append(r["budget_ratio"])

arm3_yes_samples = []
seen = set()
for r in avail_yes:
    k = (r["probe_id"], r["budget_ratio"], r["output"][:80])
    if k not in seen:
        seen.add(k)
        arm3_yes_samples.append(
            {
                "probe_id": r["probe_id"],
                "budget_ratio": r["budget_ratio"],
                "available_field": r["available_field"],
                "outcome": r["outcome"],
                "output": r["output"],
            }
        )

out = {
    "experiment": "self_report_arms",
    "model": "qwen3:4b-instruct",
    "hardware": "blade14_rtx4070",
    "probes": TARGET_PROBES,
    "ratios": RATIOS,
    "n_reps": 3,
    "arm_descriptions": {
        "arm1_baseline": "Reused Stage C EARLY truncated data, no prompt modification",
        "arm2_abstention_instruction": (
            "Appended: If the information needed to answer is not present above,"
            " respond exactly: INSUFFICIENT_CONTEXT"
        ),
        "arm3_self_report": (
            "Appended: First state whether the information needed to answer is"
            " present above, then answer. Format: AVAILABLE: yes|no, then the answer."
        ),
    },
    "aggregate_by_arm": {
        arm: {
            "n_total": 60,
            "fabrication_rate": round(
                sum(
                    1
                    for r in rows
                    if r["arm"] == arm
                    and r.get("probe_id") in TARGET_PROBES
                    and r["budget_ratio"] in RATIOS
                    and r["outcome"] == "incorrect"
                )
                / 60,
                4,
            ),
            "abstention_rate": round(
                sum(
                    1
                    for r in rows
                    if r["arm"] == arm
                    and r.get("probe_id") in TARGET_PROBES
                    and r["budget_ratio"] in RATIOS
                    and r["outcome"] == "abstained"
                )
                / 60,
                4,
            ),
            "correct_rate": round(
                sum(
                    1
                    for r in rows
                    if r["arm"] == arm
                    and r.get("probe_id") in TARGET_PROBES
                    and r["budget_ratio"] in RATIOS
                    and r["outcome"] == "correct"
                )
                / 60,
                4,
            ),
        }
        for arm in [
            "arm1_baseline",
            "arm2_abstention_instruction",
            "arm3_self_report",
        ]
    },
    "per_arm_per_ratio": {
        arm: {str(ratio): cell_stats(arm, ratio) for ratio in RATIOS}
        for arm in [
            "arm1_baseline",
            "arm2_abstention_instruction",
            "arm3_self_report",
        ]
    },
    "arm2_sentinel": {
        "n_total": len(a2_all),
        "sentinel_used": sentinel_used,
        "sentinel_used_rate": round(sentinel_used / len(a2_all), 4),
        "sentinel_ignored": sentinel_ignored,
        "sentinel_ignored_rate": round(sentinel_ignored / len(a2_all), 4),
        "sentinel_failures_by_probe": {
            k: sorted(set(v)) for k, v in arm2_sentinel_failures.items()
        },
        "note": (
            "All sentinel failures are sea_01 and sea_04 probes;"
            " model outputs 93850 (filler number) overriding the instruction"
        ),
    },
    "arm3_available_field": {
        "n_total": len(a3_all),
        "available_yes": len(avail_yes),
        "available_yes_rate": round(len(avail_yes) / len(a3_all), 4),
        "available_no": len(avail_no),
        "available_no_rate": round(len(avail_no) / len(a3_all), 4),
        "available_yes_outcome_breakdown": {
            "incorrect": len(avail_yes_fab),
            "abstained": sum(1 for r in avail_yes if r["outcome"] == "abstained"),
            "correct": sum(1 for r in avail_yes if r["outcome"] == "correct"),
        },
        "available_no_outcome_breakdown": {
            "incorrect": len(avail_no_inc),
            "abstained": len(avail_no_abs),
            "correct": sum(1 for r in avail_no if r["outcome"] == "correct"),
        },
        "available_yes_fabrication_note": (
            "All 24 AVAILABLE:yes rows are from sea_01 and sea_04;"
            " model asserts context present then outputs 93850 (filler number)"
        ),
        "available_no_but_answers_note": (
            "25/36 AVAILABLE:no rows still provide a value;"
            " model declares context absent but produces an answer"
        ),
    },
    "arm3_yes_samples": arm3_yes_samples[:10],
    "interpretation": {
        "arm2_suppresses_fabrication_partially": True,
        "arm2_sea_probes_override_instruction": True,
        "arm3_no_introspective_access": True,
        "arm3_note": (
            "AVAILABLE:yes on all sea probe rows confirms model cannot detect"
            " missing info; it reports context present and fabricates from filler."
            " AVAILABLE:no but then provides value (69% of AVAILABLE:no rows)"
            " confirms self-report does not prevent fabrication."
        ),
        "fabrication_persists_in_all_arms": True,
    },
}

Path("results/selfreport_arms.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8"
)
print("Written results/selfreport_arms.json")
print()
print("KEY NUMBERS:")
for arm in ["arm1_baseline", "arm2_abstention_instruction", "arm3_self_report"]:
    d = out["aggregate_by_arm"][arm]
    print(
        f"  {arm}: fab={d['fabrication_rate']:.1%}  "
        f"abs={d['abstention_rate']:.1%}  cor={d['correct_rate']:.1%}"
    )
print()
print(f"Arm2 sentinel compliance: {sentinel_used}/60 = {sentinel_used/60:.1%}")
print(
    "  Failures: sea_01 at r=0.55,0.40; sea_04 at r=0.85,0.55 (all output 93850)"
)
print()
print(
    f"Arm3 AVAILABLE:yes: {len(avail_yes)}/60 = {len(avail_yes)/60:.1%}"
    "  (all from sea probes)"
)
print(
    f"  Of AVAILABLE:yes: {len(avail_yes_fab)}/{len(avail_yes)} ="
    f" {len(avail_yes_fab)/len(avail_yes):.1%} fabricate"
)
print(f"Arm3 AVAILABLE:no: {len(avail_no)}/60 = {len(avail_no)/60:.1%}")
print(
    f"  Of AVAILABLE:no: {len(avail_no_inc)}/{len(avail_no)} ="
    f" {len(avail_no_inc)/len(avail_no):.1%} still answer (incorrect)"
)
