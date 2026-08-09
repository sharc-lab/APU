#!/usr/bin/env python3
"""
Validates the eval set BEFORE any model is run against it.

Three classes of check:
  A. POSITIVE  - a known-correct answer must score 1.0 through the real scorer.
  B. NEGATIVE  - a known-wrong answer must NOT score 1.0. (A scorer that always
                 returns 1.0 is worse than no scorer; this catches that.)
  C. INDEPENDENT - arithmetic/logic answers are recomputed from the problem
                 statement in Python, not copied from the `expected` field.

If this file passes, a score drop during the real run is attributable to the
model, not to a wrong answer key.
"""

import json
import pathlib
import sys
from collections import Counter

import scorers

ROOT = pathlib.Path(__file__).parent
PROBES = {json.loads(l)["id"]: json.loads(l)
          for l in (ROOT / "prompts.jsonl").read_text().splitlines() if l.strip()}

fails = []


def check(name, cond, detail=""):
    if not cond:
        fails.append(f"{name}: {detail}")
    return cond


# ============================================================ C. INDEPENDENT
# Recompute each deterministic answer from the problem statement.

def independent_answers():
    a = {}

    # rea_01 catch-up
    gap = 60 * (45 / 60)                     # miles of head start
    a["rea_01"] = str(int(gap / (80 - 60) * 60))

    # rea_03 fill rates
    from fractions import Fraction
    rate = Fraction(1, 6) + Fraction(1, 4) - Fraction(1, 12)
    a["rea_03"] = str(int(1 / rate))

    # rea_04 same-color probability
    p = Fraction(4, 10) * Fraction(3, 9) + Fraction(6, 10) * Fraction(5, 9)
    a["rea_04"] = f"{p.numerator}/{p.denominator}"

    # rea_05 flow rate
    a["rea_05"] = str(int(250 * 8 * 60 / 1000))

    # rea_06 constraint puzzle, brute force
    import itertools
    sol = None
    for perm in itertools.permutations(["tea", "coffee", "juice", "water"]):
        ana, ben, cleo, dev = perm
        if ana in ("tea", "coffee"):
            continue
        if ben in ("juice", "water"):
            continue
        if cleo != "water":
            continue
        if dev == "coffee":
            continue
        sol = {"Ana": ana, "Ben": ben, "Cleo": cleo, "Dev": dev}
    a["rea_06"] = [k for k, v in sol.items() if v == "coffee"][0]

    # rag_01 / rag_03 / rag_04 / rag_05 policy logic
    a["rag_01"] = "0.15%"
    a["rag_03"] = str(sum([62 > 75, False]))                 # dinner<=75, mileage exempt
    a["rag_04"] = str(sum([True, False]))                    # intl taxi yes, mileage exempt
    a["rag_05"] = f"{200 * 0.67:.2f}"

    # search haystack, parsed from the same literal used to build the prompt
    rows = []
    for line in PROBES["sea_01"]["prompt"].splitlines():
        if line.startswith("RECORD"):
            parts = [p.strip() for p in line.split("|")]
            d = {}
            for p in parts[1:]:
                k, v = p.split("=")
                d[k.strip()] = v.strip()
            d["units"] = int(d["units"])
            d["defect_rate"] = float(d["defect_rate"])
            rows.append(d)
    check("haystack parsed", len(rows) == 8, f"got {len(rows)} rows")
    a["sea_01"] = max(rows, key=lambda r: r["defect_rate"])["lot"]
    a["sea_03"] = str(sum(r["units"] for r in rows if r["region"] == "SE"))
    a["sea_04"] = max(rows, key=lambda r: r["units"])["region"]
    a["sea_05"] = str(sum(r["units"] for r in rows if r["defect_rate"] > 0.015))
    a["sea_06"] = str(len({r["lot"][0] for r in rows}))

    # long_horizon
    xs = [3, 1, 4, 1, 5, 9, 2, 6]
    seen, dedup = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); dedup.append(x)
    v = sorted(dedup)[1:]
    a["lon_01"] = ",".join(str(x * 2) for x in v)

    c = 100
    c -= 18; c //= 2; c += 9; c *= 3; c -= 20; c //= 5
    a["lon_02"] = str(c)

    dur = {"A": 5, "B": 3, "C": 4}
    a["lon_05"] = str(sum(dur.values()))                     # single worker, serial

    costs = {"classify": 10, "local_call": 25, "cloud_call": 120, "cache_hit": 0}
    trace = ["classify", "cloud_call", "classify", "local_call", "classify",
             "cache_hit", "classify", "local_call", "classify", "cloud_call"]
    a["lon_06"] = str(500 - sum(costs[s] for s in trace))

    # chained_tools
    a["cha_02"] = min({"h1": 0.82, "h2": 0.31, "h3": 0.67}.items(),
                      key=lambda kv: kv[1])[0]
    a["cha_03"] = f"{24.00 * 3 * 1.10:.2f}"
    a["cha_04"] = str(["fail", "fail", "success"].count("fail") * 200)
    nodes = {"n1": {"mem_gb": 16, "status": "drain"},
             "n2": {"mem_gb": 32, "status": "ready"}}
    a["cha_05"] = str(sum(n["mem_gb"] for n in nodes.values()
                          if n["status"] == "ready"))
    a["cha_06"] = str((20480 + 51200) // 1024)

    # fan_out
    a["fan_02"] = ",".join(str(x) for x in [17 * 4, 144 // 12, 2 ** 7, 91 - 38])
    a["fan_03"] = ",".join(str(w.count("e")) for w in
                           ["engineering", "resident", "cheese", "latency"])

    def is_prime(n):
        return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))
    a["fan_04"] = str(sum([is_prime(7), is_prime(51), True, is_prime(1),
                           is_prime(97)]))
    return a


IND = independent_answers()

for pid, computed in IND.items():
    stated = PROBES[pid]["expected"]
    check(f"INDEPENDENT {pid}", str(stated) == str(computed),
          f"answer key says {stated!r}, recomputed {computed!r}")


# ============================================================ A/B unit_test

for pid, p in PROBES.items():
    if p["scorer_type"] != "unit_test":
        continue
    ref = (ROOT / f"reference/ref_{pid}.py").read_text()
    s, d = scorers.score_unit_test(ref, p["expected"])
    check(f"POSITIVE {pid}", s == 1.0, f"reference scored {s} ({d})")

    bad = "def _noop():\n    return None\n"
    s2, _ = scorers.score_unit_test(bad, p["expected"])
    check(f"NEGATIVE {pid}", s2 < 1.0, f"garbage scored {s2}")


# ============================================================ A/B schema

GOOD_JSON = {
    "str_01": '{"name":"Marcus Delgado","age":34,"city":"Lisbon"}',
    "str_02": '{"total":84.5,"items":[{"sku":"SKU-9931","qty":2},'
              '{"sku":"SKU-4402","qty":1},{"sku":"SKU-1180","qty":5}]}',
    "str_03": '{"status":"open","priority":2,"tags":["infra","postgres"],'
              '"assignee":null}',
    "str_04": '{"events":[{"name":"design review","day":4,"attendees":12},'
              '{"name":"sprint planning","day":11,"attendees":8},'
              '{"name":"all-hands","day":25,"attendees":140}]}',
    "str_05": '{"deps":{"alpha":["beta","gamma"],"beta":["delta"],'
              '"gamma":["delta","epsilon"],"delta":[],"epsilon":[]}}',
    "lon_03": '{"a":7,"c":8}',
    "fan_05": '{"a":132,"b":"KV-CACHE","c":true}',
}

BAD_JSON = {
    "str_01": '{"name":"Marcus Delgado","age":"34","city":"Lisbon"}',   # wrong type
    "str_02": '{"total":84.5,"items":[{"sku":"SKU-9931","qty":2}]}',    # too few
    "str_03": '{"status":"closed","priority":2,"tags":["x"],"assignee":null}',
    "str_04": '{"events":[{"name":"x","day":4,"attendees":12}]}',        # too few
    "str_05": '{"deps":{"alpha":["beta"],"beta":["delta"],'
              '"gamma":["delta","epsilon"],"delta":[],"epsilon":[]}}',   # alpha short
    "lon_03": '{"a":5,"c":8}',                                          # stale a
    "fan_05": '{"a":132,"b":"kv-cache","c":true}',                      # wrong case
}

for pid, p in PROBES.items():
    if p["scorer_type"] != "schema":
        continue
    check(f"schema fixture {pid}", pid in GOOD_JSON, "no good fixture")
    if pid not in GOOD_JSON:
        continue
    s, d = scorers.score_schema(GOOD_JSON[pid], p["expected"])
    check(f"POSITIVE {pid}", s == 1.0, f"good JSON scored {s} ({d})")
    s2, _ = scorers.score_schema(BAD_JSON[pid], p["expected"])
    check(f"NEGATIVE {pid}", s2 < 1.0, f"bad JSON scored {s2}")

    # fenced + prose-wrapped variants must still parse
    s3, _ = scorers.score_schema(
        "Here you go:\n```json\n" + GOOD_JSON[pid] + "\n```", p["expected"])
    check(f"ROBUST {pid}", s3 == 1.0, "fenced JSON not extracted")


# ============================================================ A/B exact

for pid, p in PROBES.items():
    if p["scorer_type"] != "exact":
        continue
    s, d = scorers.score_exact(p["expected"], p["expected"])
    check(f"POSITIVE {pid}", s == 1.0, f"identity scored {s} ({d})")
    s2, _ = scorers.score_exact("banana", p["expected"])
    check(f"NEGATIVE {pid}", s2 < 1.0, "wrong answer scored 1.0")
    # models commonly wrap the answer; these must still pass
    s3, _ = scorers.score_exact(f"The answer is {p['expected']}.", p["expected"])
    check(f"ROBUST {pid}", s3 == 1.0, "wrapped answer rejected")


# ============================================================ A/B span_match

SPAN_GOOD = {
    "rea_07": "This is Simpson's paradox, caused by unequal sample size across "
              "the country strata.",
    "rag_02": "Meridian-3 still shows that drift because it retains the "
              "original firmware.",
    "rag_06": "The documents do not state the sampling rate.",
}
SPAN_BAD = {
    "rea_07": "It is a rounding error in the dashboard.",
    "rag_02": "Meridian-4 continues to drift at that rate.",
    "rag_06": "The sampling rate was 40 Hz.",
}

for pid, p in PROBES.items():
    if p["scorer_type"] != "span_match":
        continue
    s, d = scorers.score_span_match(SPAN_GOOD[pid], p["expected"])
    check(f"POSITIVE {pid}", s == 1.0, f"good answer scored {s} ({d})")
    s2, _ = scorers.score_span_match(SPAN_BAD[pid], p["expected"])
    check(f"NEGATIVE {pid}", s2 < 1.0, f"bad answer scored {s2}")


# ============================================================ structural

for pid, p in PROBES.items():
    check(f"struct {pid} ctx-indep", p["context_independent"] is True, "")
    check(f"struct {pid} maxtok", 0 < p["max_tokens"] <= 800, "")
    check(f"struct {pid} difficulty",
          p["difficulty"] in ("easy", "medium", "hard"), "")

# judge probes must carry a rubric
for pid, p in PROBES.items():
    if p["scorer_type"] == "judge":
        check(f"rubric {pid}", isinstance(p["expected"], dict)
              and len(p["expected"].get("rubric", "")) > 40, "thin rubric")


# ============================================================ report

print(f"probes            : {len(PROBES)}")
print(f"by category       : {dict(Counter(p['category'] for p in PROBES.values()))}")
print(f"by difficulty     : {dict(Counter(p['difficulty'] for p in PROBES.values()))}")
print(f"by scorer         : {dict(Counter(p['scorer_type'] for p in PROBES.values()))}")
det = sum(1 for p in PROBES.values() if p["scorer_type"] != "judge")
print(f"deterministic     : {det}/{len(PROBES)}")
print(f"independent recomp: {len(IND)} answers")
print()

if fails:
    print(f"FAILED ({len(fails)}):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
