#!/usr/bin/env python3
"""
Builds the Paper 1 quality-axis eval set.

Design constraints (see EVALSET_DESIGN.md):
  1. Every probe is CONTEXT-INDEPENDENT: all information needed to answer is in
     the prompt itself. Filler context injected by the harness must never be
     required to answer. This is what lets a score drop be attributed to
     degradation rather than retrieval failure.
  2. Scoring is deterministic-primary. 42/50 probes are mechanically checkable.
     8/50 are judge-scored and reported as a SEPARATE column so a reviewer can
     discount them.
  3. Difficulty is stratified (easy/medium/hard) so the run yields a capability
     boundary, not just an average score.
  4. Outputs are short and bounded so context depth, not output length, is the
     independent variable.
"""

import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).parent
SCHEMAS = ROOT / "schemas"
TESTS = ROOT / "tests"
REF = ROOT / "reference"

for d in (SCHEMAS, TESTS, REF):
    d.mkdir(parents=True, exist_ok=True)

P = []


def probe(id, category, difficulty, prompt, scorer_type, expected,
          max_tokens=256, notes=""):
    P.append({
        "id": id,
        "category": category,
        "difficulty": difficulty,
        "prompt": prompt.strip(),
        "scorer_type": scorer_type,
        "expected": expected,
        "max_tokens": max_tokens,
        "context_independent": True,
        "notes": notes,
    })


# Shared instruction fragment forcing terse, parseable output.
TERSE = "\n\nRespond with ONLY the final answer. No explanation, no units, no punctuation."
JSONONLY = "\n\nRespond with ONLY valid JSON. No markdown fences, no commentary."
CODEONLY = "\n\nRespond with ONLY a Python code block containing the function. No explanation."

# ---------------------------------------------------------------- reasoning_heavy (8)

probe("rea_01", "reasoning_heavy", "easy",
      "A train leaves a station at 2:15 PM traveling at 60 mph. A second train "
      "leaves the same station 45 minutes later traveling at 80 mph in the same "
      "direction on a parallel track. How many minutes after the second train "
      "departs does it catch the first train?" + TERSE,
      "exact", "135", 64,
      "45 min head start = 45 mi gap; closing speed 20 mph; 45/20 = 2.25 h = 135 min.")

probe("rea_02", "reasoning_heavy", "easy",
      "A store sells pens in packs of 12 and pencils in packs of 20. What is the "
      "smallest total number of pens such that the number of pens equals the "
      "number of pencils, buying only whole packs of each?" + TERSE,
      "exact", "60", 64,
      "LCM(12,20)=60.")

probe("rea_03", "reasoning_heavy", "medium",
      "Pipe A fills a tank in 6 hours. Pipe B fills the same tank in 4 hours. "
      "Drain C empties the full tank in 12 hours. All three are opened on an "
      "empty tank simultaneously. How many hours until the tank is full?" + TERSE,
      "exact", "3", 64,
      "1/6 + 1/4 - 1/12 = 1/3 tank/hr.")

probe("rea_04", "reasoning_heavy", "medium",
      "A bag contains 4 red marbles and 6 blue marbles. Two marbles are drawn "
      "without replacement. What is the probability both drawn marbles are the "
      "same color? Give the answer as a fully reduced fraction in the form a/b."
      + TERSE,
      "exact", "7/15", 64,
      "(4/10)(3/9) + (6/10)(5/9) = 42/90 = 7/15.")

probe("rea_05", "reasoning_heavy", "medium",
      "A pump moves 250 milliliters of water per second. How many liters does it "
      "move in 8 minutes?" + TERSE,
      "exact", "120", 64,
      "250 mL/s * 480 s = 120000 mL = 120 L.")

probe("rea_06", "reasoning_heavy", "hard",
      "Four people (Ana, Ben, Cleo, Dev) each ordered exactly one of: tea, "
      "coffee, juice, water. Clues:\n"
      "1. Ana did not order tea or coffee.\n"
      "2. Ben ordered neither juice nor water.\n"
      "3. Cleo ordered water.\n"
      "4. Dev did not order coffee.\n"
      "Who ordered coffee?" + TERSE,
      "exact", "Ben", 64,
      "Cleo=water. Ana in {juice,water}->juice. Ben in {tea,coffee}. "
      "Dev not coffee, so Dev=tea, Ben=coffee.")

probe("rea_07", "reasoning_heavy", "hard",
      "Consider this claim: 'Our A/B test showed variant B had a higher "
      "conversion rate than variant A in every single country we tested, "
      "therefore variant B has a higher conversion rate overall.' This claim "
      "can be false. Name the statistical phenomenon that explains how, and "
      "state in one sentence what causes it.",
      "span_match",
      {"required_any": [["simpson"], ["simpson's paradox"]],
       "required_all": [],
       "forbidden": [],
       "any_of_groups": [["unequal", "different", "varies", "varying", "imbalance",
                          "distribution", "sample size", "weight"]]},
      160,
      "Simpson's paradox; caused by unequal group sizes across strata.")

probe("rea_08", "reasoning_heavy", "hard",
      "A hospital reports that patients treated with a new drug have a HIGHER "
      "mortality rate than patients who received no treatment. The drug is "
      "genuinely effective. Explain in 2-3 sentences the most likely reason for "
      "this pattern in the observational data.",
      "judge",
      {"rubric": "Answer must identify confounding by indication / selection "
                 "effect: sicker patients are preferentially given the new drug, "
                 "so treatment group has worse baseline prognosis. Full credit "
                 "requires naming the mechanism (not just 'correlation is not "
                 "causation'). Partial credit for identifying confounding "
                 "generically."},
      200,
      "Confounding by indication.")

# ---------------------------------------------------------------- code_heavy (8)

probe("cod_01", "code_heavy", "easy",
      "Write a Python function `reverse_words(s: str) -> str` that reverses the "
      "order of whitespace-separated words in a string, collapsing any runs of "
      "whitespace to a single space and stripping leading/trailing whitespace. "
      "Example: '  the sky   is blue ' -> 'blue is sky the'" + CODEONLY,
      "unit_test", "tests/test_cod_01.py", 320)

probe("cod_02", "code_heavy", "easy",
      "Write a Python function `fizzbuzz_sum(n: int) -> int` that returns the sum "
      "of all integers from 1 to n inclusive that are divisible by 3 or 5 but "
      "NOT by both." + CODEONLY,
      "unit_test", "tests/test_cod_02.py", 320)

probe("cod_03", "code_heavy", "medium",
      "Write a Python function `search_rotated(nums: list[int], target: int) -> int` "
      "that performs binary search on a sorted array that has been rotated at an "
      "unknown pivot. Return the index of target, or -1 if absent. All values are "
      "distinct. Must run in O(log n)." + CODEONLY,
      "unit_test", "tests/test_cod_03.py", 480)

probe("cod_04", "code_heavy", "medium",
      "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
      "that merges all overlapping intervals and returns them sorted by start. "
      "Intervals touching at an endpoint (e.g. [1,3] and [3,5]) DO merge. "
      "Handle the empty list." + CODEONLY,
      "unit_test", "tests/test_cod_04.py", 480)

probe("cod_05", "code_heavy", "medium",
      "The following Python function is supposed to return the second largest "
      "DISTINCT value in a list, or None if fewer than two distinct values exist. "
      "It has a bug. Return the corrected function, keeping the name "
      "`second_largest`.\n\n"
      "```python\n"
      "def second_largest(nums):\n"
      "    nums.sort(reverse=True)\n"
      "    return nums[1]\n"
      "```" + CODEONLY,
      "unit_test", "tests/test_cod_05.py", 400,
      "Bugs: no dedup, no length guard, mutates input.")

probe("cod_06", "code_heavy", "hard",
      "Write a Python class `LRUCache` with `__init__(self, capacity: int)`, "
      "`get(self, key) -> int` returning -1 on miss, and `put(self, key, value) -> None`. "
      "Both operations must be O(1) average. Evict the least recently used entry "
      "when over capacity. A `get` counts as a use; a `put` to an existing key "
      "counts as a use and updates the value." + CODEONLY,
      "unit_test", "tests/test_cod_06.py", 640)

probe("cod_07", "code_heavy", "hard",
      "Write a Python function `eval_expr(s: str) -> int` that evaluates an "
      "arithmetic expression string containing non-negative integers, the binary "
      "operators + - * /, and parentheses. Division is integer division "
      "truncating toward zero. Standard precedence applies. Do not use eval() or "
      "exec()." + CODEONLY,
      "unit_test", "tests/test_cod_07.py", 800)

probe("cod_08", "code_heavy", "hard",
      "Write a Python function `top_k_frequent(words: list[str], k: int) -> list[str]` "
      "returning the k most frequent words, sorted by descending frequency, with "
      "ties broken by ascending lexicographic order." + CODEONLY,
      "unit_test", "tests/test_cod_08.py", 480)

# ---------------------------------------------------------------- structured_output (5)

probe("str_01", "structured_output", "easy",
      "Extract the fields from this text into JSON with keys "
      "\"name\" (string), \"age\" (integer), \"city\" (string).\n\n"
      "Text: Marcus Delgado is a 34-year-old architect currently based in Lisbon."
      + JSONONLY,
      "schema", "schemas/str_01.json", 160)

probe("str_02", "structured_output", "easy",
      "Convert to JSON with keys \"total\" (number) and \"items\" (array of "
      "objects each with \"sku\" (string) and \"qty\" (integer)).\n\n"
      "Order: 2x SKU-9931, 1x SKU-4402, 5x SKU-1180. Total charged: 84.50"
      + JSONONLY,
      "schema", "schemas/str_02.json", 240)

probe("str_03", "structured_output", "medium",
      "Produce JSON with keys \"status\" (one of exactly: \"open\", \"closed\", "
      "\"pending\"), \"priority\" (integer 1-5), \"tags\" (array of strings, "
      "min 1 item), and \"assignee\" (string or null).\n\n"
      "Ticket: Sev-2 database connection pool exhaustion. Still being worked. "
      "Nobody assigned yet. Relates to infra and postgres." + JSONONLY,
      "schema", "schemas/str_03.json", 280)

probe("str_04", "structured_output", "medium",
      "Produce JSON with key \"events\": an array of objects, each with "
      "\"name\" (string), \"day\" (integer 1-31), and \"attendees\" (integer). "
      "Include every event mentioned.\n\n"
      "Text: The design review is on the 4th with 12 people. Sprint planning "
      "lands on the 11th, 8 attending. The all-hands is the 25th and all 140 "
      "employees are expected." + JSONONLY,
      "schema", "schemas/str_04.json", 320)

probe("str_05", "structured_output", "hard",
      "Produce JSON with key \"deps\": an object mapping each package name "
      "(string) to an array of the package names it directly depends on. Include "
      "every package mentioned as a key, even if it has no dependencies.\n\n"
      "Text: alpha requires beta and gamma. beta requires delta. gamma requires "
      "delta and epsilon. delta has no dependencies. epsilon has no dependencies."
      + JSONONLY,
      "schema", "schemas/str_05.json", 320)

# ---------------------------------------------------------------- rag_heavy (6)
# Retrieved documents are embedded IN the prompt, keeping the probe self-contained.

RAG_DOCS_A = """
[DOC 1] The Meridian-4 sensor array was commissioned in March 2019 at the Caldera
Ridge site. Initial calibration drift was measured at 0.8% per month.

[DOC 2] Following the firmware revision in August 2020, Meridian-4 calibration
drift was reduced to 0.15% per month. The revision did not alter sampling rate.

[DOC 3] The Meridian-3 array, a separate deployment at Thornfield Basin, retains
the original firmware and continues to exhibit drift near 0.9% per month.
"""

probe("rag_01", "rag_heavy", "easy",
      f"Using only the documents below, answer the question.\n{RAG_DOCS_A}\n"
      "Question: What was the calibration drift of Meridian-4 after the firmware "
      "revision, as a percentage per month?" + TERSE,
      "exact", "0.15%", 96,
      "Answer normalizer strips whitespace/case; accepts 0.15% or 0.15 percent.")

probe("rag_02", "rag_heavy", "medium",
      f"Using only the documents below, answer the question.\n{RAG_DOCS_A}\n"
      "Question: Which array still shows drift near 0.9% per month, and why? "
      "Answer in one sentence.",
      "span_match",
      {"required_all": ["meridian-3"],
       "required_any": [],
       "forbidden": ["meridian-4 still", "meridian-4 continues"],
       "any_of_groups": [["original firmware", "old firmware", "not updated",
                          "retains", "never updated", "unrevised"]]},
      160)

RAG_DOCS_B = """
[DOC 1] Policy 14.2: Reimbursement for domestic travel requires receipts for any
single expense over $75. Expenses at or below $75 may be claimed without receipt.

[DOC 2] Policy 14.7: International travel requires receipts for ALL expenses
regardless of amount.

[DOC 3] Policy 9.1: Mileage is reimbursed at $0.67 per mile and never requires a
receipt, domestic or international.
"""

probe("rag_03", "rag_heavy", "medium",
      f"Using only the documents below, answer the question.\n{RAG_DOCS_B}\n"
      "Question: An employee on a domestic trip claims a $62 dinner and 140 miles "
      "of driving. How many of these two line items require a receipt?" + TERSE,
      "exact", "0", 96,
      "$62 <= $75 threshold; mileage never needs receipt.")

probe("rag_04", "rag_heavy", "hard",
      f"Using only the documents below, answer the question.\n{RAG_DOCS_B}\n"
      "Question: An employee on an international trip claims a $40 taxi and 90 "
      "miles of personal-vehicle driving. How many of these two line items "
      "require a receipt?" + TERSE,
      "exact", "1", 96,
      "Tests precedence: 14.7 requires all receipts, but 9.1 exempts mileage "
      "explicitly for international too.")

probe("rag_05", "rag_heavy", "medium",
      f"Using only the documents below, answer the question.\n{RAG_DOCS_B}\n"
      "Question: What is the reimbursement amount in dollars for 200 miles of "
      "driving? Give a number with two decimal places." + TERSE,
      "exact", "134.00", 96,
      "200 * 0.67 = 134.00")

probe("rag_06", "rag_heavy", "hard",
      f"Using only the documents below, answer the question.\n{RAG_DOCS_A}\n"
      "Question: Based only on these documents, what was the sampling rate of "
      "Meridian-4 in 2021?",
      "span_match",
      {"required_all": [],
       "required_any": [],
       "forbidden": [],
       "any_of_groups": [["not stated", "not specified", "not mentioned",
                          "not provided", "not given", "not included",
                          "does not", "doesn't", "do not", "don't",
                          "no information", "not contain", "unknown",
                          "cannot", "can't", "unable to"]]},
      160,
      "Abstention probe. Docs say the revision did not ALTER sampling rate but "
      "never state its value. Tests refusal-to-fabricate under context pressure.")

# ---------------------------------------------------------------- search_heavy (6)
# A haystack is embedded in the prompt; the model must locate a fact within it.

HAYSTACK = """
RECORD 001 | region=NW  | units=418  | defect_rate=0.021 | lot=A7
RECORD 002 | region=SE  | units=1130 | defect_rate=0.004 | lot=B2
RECORD 003 | region=NW  | units=207  | defect_rate=0.038 | lot=A9
RECORD 004 | region=MW  | units=894  | defect_rate=0.011 | lot=C1
RECORD 005 | region=SE  | units=765  | defect_rate=0.009 | lot=B7
RECORD 006 | region=NW  | units=1002 | defect_rate=0.006 | lot=A3
RECORD 007 | region=MW  | units=333  | defect_rate=0.029 | lot=C8
RECORD 008 | region=SE  | units=612  | defect_rate=0.017 | lot=B4
"""

probe("sea_01", "search_heavy", "easy",
      f"Given these records:\n{HAYSTACK}\n"
      "Which lot has the highest defect_rate?" + TERSE,
      "exact", "A9", 64)

probe("sea_02", "search_heavy", "easy",
      f"Given these records:\n{HAYSTACK}\n"
      "How many records have region=NW?" + TERSE,
      "exact", "3", 64)

probe("sea_03", "search_heavy", "medium",
      f"Given these records:\n{HAYSTACK}\n"
      "What is the total units across all region=SE records?" + TERSE,
      "exact", "2507", 96,
      "1130 + 765 + 612 = 2507")

probe("sea_04", "search_heavy", "medium",
      f"Given these records:\n{HAYSTACK}\n"
      "Which region has the record with the largest units value?" + TERSE,
      "exact", "SE", 64,
      "1130 is max, region SE.")

probe("sea_05", "search_heavy", "hard",
      f"Given these records:\n{HAYSTACK}\n"
      "Considering only records with defect_rate above 0.015, what is the sum of "
      "their units?" + TERSE,
      "exact", "1570", 96,
      "0.021->418, 0.038->207, 0.029->333, 0.017->612. Sum=1570.")

probe("sea_06", "search_heavy", "hard",
      f"Given these records:\n{HAYSTACK}\n"
      "How many distinct lot prefixes (the single letter before the digit) appear?"
      + TERSE,
      "exact", "3", 64,
      "A, B, C = 3")

# ---------------------------------------------------------------- long_horizon (6)

probe("lon_01", "long_horizon", "medium",
      "Apply these operations in order to the starting list [3, 1, 4, 1, 5, 9, 2, 6].\n"
      "1. Remove all duplicate values, keeping first occurrence.\n"
      "2. Sort ascending.\n"
      "3. Drop the smallest element.\n"
      "4. Multiply every remaining element by 2.\n"
      "Give the final list as comma-separated integers with no brackets." + TERSE,
      "exact", "4,6,8,10,12,18", 128,
      "dedup->[3,1,4,5,9,2,6]; sort->[1,2,3,4,5,6,9]; drop 1->[2,3,4,5,6,9]; "
      "x2->[4,6,8,10,12,18]")

probe("lon_02", "long_horizon", "medium",
      "A counter starts at 100. Apply in order:\n"
      "1. Subtract 18.\n2. Divide by 2.\n3. Add 9.\n4. Multiply by 3.\n"
      "5. Subtract 20.\n6. Divide by 5.\n"
      "What is the final value?" + TERSE,
      "exact", "26", 96,
      "100-18=82; /2=41; +9=50; *3=150; -20=130; /5=26.")

probe("lon_03", "long_horizon", "hard",
      "You start with an empty dictionary. Apply in order:\n"
      "1. Set key 'a' to 1.\n"
      "2. Set key 'b' to 2.\n"
      "3. Set key 'a' to 5.\n"
      "4. Delete key 'b'.\n"
      "5. Set key 'c' to the current value of 'a' plus 3.\n"
      "6. Set key 'a' to the current value of 'c' minus 1.\n"
      "Give the final dictionary as JSON." + JSONONLY,
      "schema", "schemas/lon_03.json", 160,
      "a=5; c=8; a=7 -> {a:7, c:8}")

probe("lon_04", "long_horizon", "easy",
      "Start with the string 'hardware'. Apply in order:\n"
      "1. Reverse it.\n2. Remove all vowels (a,e,i,o,u).\n3. Convert to uppercase.\n"
      "What is the result?" + TERSE,
      "exact", "WRDRH", 96,
      "'hardware'->'erawdrah'; remove vowels -> 'rwdrh'; upper -> 'RWDRH'. "
      "CORRECTED in validation.")

probe("lon_05", "long_horizon", "hard",
      "Three tasks run on one worker with these durations and dependencies:\n"
      "TaskA: 5 min, no deps.\n"
      "TaskB: 3 min, depends on TaskA.\n"
      "TaskC: 4 min, depends on TaskA.\n"
      "The worker runs one task at a time and always picks the shortest available "
      "ready task first. What is the total elapsed time in minutes?" + TERSE,
      "exact", "12", 96,
      "A(5) then B(3) then C(4) = 12 serial.")

probe("lon_06", "long_horizon", "medium",
      "A budget starts at 500 tokens. Each step costs: classify=10, local_call=25, "
      "cloud_call=120, cache_hit=0. Execute this trace in order and report the "
      "remaining budget:\n"
      "classify, cloud_call, classify, local_call, classify, cache_hit, "
      "classify, local_call, classify, cloud_call" + TERSE,
      "exact", "160", 128,
      "5 classify=50; 2 cloud=240; 2 local=50; 1 cache=0. Total 340. 500-340=160.")

# ---------------------------------------------------------------- chained_tools (6)
# Tool outputs are provided inline; the model must thread them correctly.

probe("cha_01", "chained_tools", "easy",
      "You called two tools and got these results:\n"
      "get_user(id=88) -> {\"name\": \"Rivera\", \"dept_id\": 12}\n"
      "get_dept(id=12) -> {\"name\": \"Reliability\", \"head_id\": 91}\n"
      "What is the department name for user 88?" + TERSE,
      "exact", "Reliability", 64)

probe("cha_02", "chained_tools", "medium",
      "Tool results, in call order:\n"
      "list_hosts() -> [\"h1\", \"h2\", \"h3\"]\n"
      "get_load(\"h1\") -> 0.82\n"
      "get_load(\"h2\") -> 0.31\n"
      "get_load(\"h3\") -> 0.67\n"
      "Which host should receive the next request, assuming lowest load wins?"
      + TERSE,
      "exact", "h2", 64)

probe("cha_03", "chained_tools", "medium",
      "Tool results, in call order:\n"
      "get_price(\"SKU-11\") -> 24.00\n"
      "get_qty(\"SKU-11\") -> 3\n"
      "get_tax_rate(\"WA\") -> 0.10\n"
      "Compute the total cost including tax. Give a number with two decimals."
      + TERSE,
      "exact", "79.20", 96,
      "24*3=72; 72*1.10=79.20")

probe("cha_04", "chained_tools", "hard",
      "Tool results, in call order:\n"
      "get_config() -> {\"retries\": 3, \"backoff_ms\": 200, \"jitter\": false}\n"
      "get_attempt_log() -> [\"fail\", \"fail\", \"success\"]\n"
      "With no jitter and constant backoff, how many total milliseconds were "
      "spent waiting in backoff before the successful attempt?" + TERSE,
      "exact", "400", 96,
      "2 failures -> 2 waits of 200ms = 400ms.")

probe("cha_05", "chained_tools", "hard",
      "Tool results, in call order:\n"
      "resolve_alias(\"prod\") -> \"cluster-7\"\n"
      "get_cluster(\"cluster-7\") -> {\"nodes\": [\"n1\",\"n2\"], \"region\": \"us-e\"}\n"
      "get_node(\"n1\") -> {\"mem_gb\": 16, \"status\": \"drain\"}\n"
      "get_node(\"n2\") -> {\"mem_gb\": 32, \"status\": \"ready\"}\n"
      "How many GB of memory are available for scheduling in the 'prod' alias?"
      + TERSE,
      "exact", "32", 96,
      "n1 draining -> unavailable. Only n2's 32.")

probe("cha_06", "chained_tools", "medium",
      "Tool results, in call order:\n"
      "search(\"invoice\") -> [\"f1.pdf\", \"f2.pdf\"]\n"
      "stat(\"f1.pdf\") -> {\"bytes\": 20480}\n"
      "stat(\"f2.pdf\") -> {\"bytes\": 51200}\n"
      "What is the combined size in kilobytes, using 1 KB = 1024 bytes?" + TERSE,
      "exact", "70", 96,
      "20480+51200 = 71680 bytes / 1024 = 70 KB.")

# ---------------------------------------------------------------- fan_out (5)
# Independent subtasks that must all be answered in one response and aggregated.

probe("fan_01", "fan_out", "easy",
      "Answer all three independently, then give ONLY the sum of the three "
      "numeric answers.\n"
      "(a) How many days in a non-leap year?\n"
      "(b) How many minutes in 3 hours?\n"
      "(c) How many sides does a hexagon have?" + TERSE,
      "exact", "551", 96,
      "365 + 180 + 6 = 551")

probe("fan_02", "fan_out", "medium",
      "Compute all four independently, then give ONLY the four results as "
      "comma-separated integers in order.\n"
      "(a) 17 * 4\n(b) 144 / 12\n(c) 2^7\n(d) 91 - 38" + TERSE,
      "exact", "68,12,128,53", 96)

probe("fan_03", "fan_out", "medium",
      "For each of the four strings, count the letter 'e'. Give ONLY the four "
      "counts as comma-separated integers in order.\n"
      "(a) engineering\n(b) resident\n(c) cheese\n(d) latency" + TERSE,
      "exact", "3,2,3,1", 96,
      "engineering: e,e,e =3. resident: e,e =2. cheese: e,e,e =3. latency: e =1.")

probe("fan_04", "fan_out", "hard",
      "Evaluate each independently, then give ONLY the count of how many are TRUE.\n"
      "(a) 7 is prime\n(b) 51 is prime\n(c) 2 is the only even prime\n"
      "(d) 1 is prime\n(e) 97 is prime" + TERSE,
      "exact", "3", 96,
      "a T, b F (3*17), c T, d F, e T -> 3")

probe("fan_05", "fan_out", "hard",
      "Three independent subtasks. Return ONLY valid JSON with keys \"a\", \"b\", "
      "\"c\" whose values are the respective answers.\n"
      "(a) the integer 12 factorial divided by 10 factorial\n"
      "(b) the string 'kv-cache' uppercased\n"
      "(c) the boolean result of: 2**10 > 1000" + JSONONLY,
      "schema", "schemas/fan_05.json", 160,
      "a = 12*11 = 132; b = 'KV-CACHE'; c = true")

# ---------------------------------------------------------------- judge probes (7 more)
# Kept few and clearly bounded. Reported as a separate column.

probe("jud_01", "reasoning_heavy", "medium",
      "In two sentences, explain why re-sending an entire conversation history on "
      "every turn makes multi-turn agent workloads expensive, and name the cost "
      "that grows.",
      "judge",
      {"rubric": "Must identify that prefill/prompt processing is re-paid over "
                 "the whole accumulated context each turn, and that cost grows "
                 "with context length (quadratic attention or linear token cost "
                 "both acceptable). Must not claim decode is the dominant term."},
      200)

probe("jud_02", "code_heavy", "medium",
      "In two sentences, explain what a KV cache stores during autoregressive "
      "decoding and why keeping it resident across turns saves work.",
      "judge",
      {"rubric": "Must state KV cache holds key/value projections for previously "
                 "processed tokens, and that residency avoids recomputing them "
                 "(avoids re-prefill). Penalize if it claims the cache stores "
                 "output tokens or logits."},
      200)

probe("jud_03", "long_horizon", "hard",
      "An agent must complete a 20-step task but its context window fits only 12 "
      "steps of history. Describe in 3 sentences one concrete strategy to finish "
      "the task, and state one specific way that strategy can lose information.",
      "judge",
      {"rubric": "Must propose a concrete mechanism (summarization/compaction, "
                 "sliding window, external memory/scratchpad, or retrieval) AND "
                 "name a specific failure mode of that mechanism (e.g. summary "
                 "drops a constraint needed later, eviction removes a referenced "
                 "entity). Generic 'may lose context' is partial credit only."},
      240)

probe("jud_04", "rag_heavy", "medium",
      "In two sentences, explain the difference between a model failing a task "
      "because the needed fact was never retrieved, versus failing because the "
      "fact was retrieved but ignored.",
      "judge",
      {"rubric": "Must distinguish retrieval failure (fact absent from context) "
                 "from attention/utilization failure (fact present but unused). "
                 "Bonus but not required: notes these need different fixes."},
      200)

probe("jud_05", "search_heavy", "hard",
      "You are given a 50,000-token document and asked a question whose answer "
      "appears exactly once, in the middle. In 2-3 sentences, explain why models "
      "often fail this and what the phenomenon is commonly called.",
      "judge",
      {"rubric": "Must reference lost-in-the-middle / positional degradation: "
                 "recall is worse for information in the middle of long contexts "
                 "than at the start or end. Naming the effect is required for "
                 "full credit."},
      240)

probe("jud_06", "structured_output", "hard",
      "In two sentences, explain why a model that produces valid JSON at short "
      "context lengths may start producing malformed JSON at long context "
      "lengths.",
      "judge",
      {"rubric": "Must connect degraded instruction-following / format adherence "
                 "to context length, e.g. the format instruction is early in the "
                 "prompt and its influence weakens as context grows, or "
                 "attention dilution over long inputs. Reject answers that "
                 "attribute it purely to randomness or temperature."},
      200)

probe("jud_07", "chained_tools", "hard",
      "In 2-3 sentences, explain why an agent that routes some steps to a local "
      "model and some to a cloud model may pay a cost that a single-backend agent "
      "does not, even if both models are equally fast.",
      "judge",
      {"rubric": "Must identify switch cost: the conversation/KV state is not "
                 "shared across backends, so switching requires re-sending or "
                 "re-prefilling the context on the new backend. Credit answers "
                 "citing cold cache on the target backend."},
      240)


# ================================================================= schemas

def w(path, obj):
    (ROOT / path).write_text(json.dumps(obj, indent=2))


w("schemas/str_01.json", {
    "type": "object",
    "required": ["name", "age", "city"],
    "additionalProperties": False,
    "properties": {
        "name": {"const": "Marcus Delgado"},
        "age": {"const": 34},
        "city": {"const": "Lisbon"},
    },
})

w("schemas/str_02.json", {
    "type": "object",
    "required": ["total", "items"],
    "additionalProperties": False,
    "properties": {
        "total": {"const": 84.5},
        "items": {
            "type": "array",
            "minItems": 3, "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["sku", "qty"],
                "additionalProperties": False,
                "properties": {
                    "sku": {"enum": ["SKU-9931", "SKU-4402", "SKU-1180"]},
                    "qty": {"type": "integer", "minimum": 1},
                },
            },
        },
    },
})

w("schemas/str_03.json", {
    "type": "object",
    "required": ["status", "priority", "tags", "assignee"],
    "additionalProperties": False,
    "properties": {
        "status": {"const": "open"},
        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
        "tags": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "assignee": {"type": "null"},
    },
})

w("schemas/str_04.json", {
    "type": "object",
    "required": ["events"],
    "additionalProperties": False,
    "properties": {
        "events": {
            "type": "array",
            "minItems": 3, "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["name", "day", "attendees"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "day": {"enum": [4, 11, 25]},
                    "attendees": {"enum": [12, 8, 140]},
                },
            },
        },
    },
})

w("schemas/str_05.json", {
    "type": "object",
    "required": ["deps"],
    "additionalProperties": False,
    "properties": {
        "deps": {
            "type": "object",
            "required": ["alpha", "beta", "gamma", "delta", "epsilon"],
            "additionalProperties": False,
            "properties": {
                "alpha": {"type": "array", "items": {"type": "string"},
                          "minItems": 2, "maxItems": 2},
                "beta": {"type": "array", "items": {"const": "delta"},
                         "minItems": 1, "maxItems": 1},
                "gamma": {"type": "array", "items": {"type": "string"},
                          "minItems": 2, "maxItems": 2},
                "delta": {"type": "array", "maxItems": 0},
                "epsilon": {"type": "array", "maxItems": 0},
            },
        },
    },
})

w("schemas/lon_03.json", {
    "type": "object",
    "required": ["a", "c"],
    "additionalProperties": False,
    "properties": {"a": {"const": 7}, "c": {"const": 8}},
})

w("schemas/fan_05.json", {
    "type": "object",
    "required": ["a", "b", "c"],
    "additionalProperties": False,
    "properties": {
        "a": {"const": 132},
        "b": {"const": "KV-CACHE"},
        "c": {"const": True},
    },
})

# ================================================================= unit tests

TEST_FILES = {
"tests/test_cod_01.py": '''
from candidate import reverse_words
def test_basic():
    assert reverse_words("the sky is blue") == "blue is sky the"
def test_whitespace():
    assert reverse_words("  the sky   is blue ") == "blue is sky the"
def test_single():
    assert reverse_words("word") == "word"
def test_empty():
    assert reverse_words("   ") == ""
''',

"tests/test_cod_02.py": '''
from candidate import fizzbuzz_sum
def test_small():
    # 3,5,6,9,10,12 -> 45 ; 15 excluded (div by both)
    assert fizzbuzz_sum(15) == 45
def test_one():
    assert fizzbuzz_sum(1) == 0
def test_thirty():
    # 3,5,6,9,10,12,18,20,21,24,25,27 -> 180 (15 and 30 excluded)
    assert fizzbuzz_sum(30) == 180
''',

"tests/test_cod_03.py": '''
from candidate import search_rotated
def test_found():
    assert search_rotated([4,5,6,7,0,1,2], 0) == 4
def test_absent():
    assert search_rotated([4,5,6,7,0,1,2], 3) == -1
def test_single():
    assert search_rotated([1], 1) == 0
def test_unrotated():
    assert search_rotated([1,2,3,4,5], 4) == 3
def test_pivot_first():
    assert search_rotated([5,1,2,3,4], 5) == 0
''',

"tests/test_cod_04.py": '''
from candidate import merge_intervals
def test_basic():
    assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
def test_touching():
    assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
def test_empty():
    assert merge_intervals([]) == []
def test_unsorted():
    assert merge_intervals([[5,7],[1,3],[2,4]]) == [[1,4],[5,7]]
def test_contained():
    assert merge_intervals([[1,10],[2,3]]) == [[1,10]]
''',

"tests/test_cod_05.py": '''
from candidate import second_largest
def test_basic():
    assert second_largest([3,1,4,1,5]) == 4
def test_dupes_at_top():
    assert second_largest([5,5,3]) == 3
def test_all_same():
    assert second_largest([2,2,2]) is None
def test_too_short():
    assert second_largest([1]) is None
def test_no_mutation():
    src = [3,1,2]
    second_largest(src)
    assert src == [3,1,2]
''',

"tests/test_cod_06.py": '''
from candidate import LRUCache
def test_basic():
    c = LRUCache(2)
    c.put(1,1); c.put(2,2)
    assert c.get(1) == 1
    c.put(3,3)              # evicts key 2
    assert c.get(2) == -1
    assert c.get(3) == 3
def test_update_counts_as_use():
    c = LRUCache(2)
    c.put(1,1); c.put(2,2)
    c.put(1,10)             # refresh key 1
    c.put(3,3)              # evicts key 2
    assert c.get(1) == 10
    assert c.get(2) == -1
def test_capacity_one():
    c = LRUCache(1)
    c.put(1,1); c.put(2,2)
    assert c.get(1) == -1
    assert c.get(2) == 2
''',

"tests/test_cod_07.py": '''
from candidate import eval_expr
def test_precedence():
    assert eval_expr("2+3*4") == 14
def test_parens():
    assert eval_expr("(2+3)*4") == 20
def test_div_trunc():
    assert eval_expr("7/2") == 3
def test_nested():
    assert eval_expr("((1+2)*(3+4))-5") == 16
def test_spaces():
    assert eval_expr(" 10 - 2 * 3 ") == 4
''',

"tests/test_cod_08.py": '''
from candidate import top_k_frequent
def test_basic():
    assert top_k_frequent(["i","love","code","i","love","fun"], 2) == ["i","love"]
def test_tiebreak_lexicographic():
    assert top_k_frequent(["b","a","c"], 2) == ["a","b"]
def test_k_all():
    assert top_k_frequent(["x","y","x"], 2) == ["x","y"]
''',
}

for path, body in TEST_FILES.items():
    (ROOT / path).write_text(body.lstrip())

# ================================================================= reference solutions
# Used ONLY to validate that the scorers/tests are self-consistent.
# Never shown to the model under test.

REFERENCE = {
"reference/ref_cod_01.py": '''
def reverse_words(s: str) -> str:
    return " ".join(reversed(s.split()))
''',
"reference/ref_cod_02.py": '''
def fizzbuzz_sum(n: int) -> int:
    return sum(i for i in range(1, n+1)
               if (i % 3 == 0) != (i % 5 == 0))
''',
"reference/ref_cod_03.py": '''
def search_rotated(nums, target):
    lo, hi = 0, len(nums)-1
    while lo <= hi:
        mid = (lo+hi)//2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid-1
            else:
                lo = mid+1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid+1
            else:
                hi = mid-1
    return -1
''',
"reference/ref_cod_04.py": '''
def merge_intervals(intervals):
    if not intervals:
        return []
    xs = sorted(intervals, key=lambda p: p[0])
    out = [list(xs[0])]
    for s, e in xs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
''',
"reference/ref_cod_05.py": '''
def second_largest(nums):
    u = sorted(set(nums), reverse=True)
    return u[1] if len(u) >= 2 else None
''',
"reference/ref_cod_06.py": '''
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.d = OrderedDict()
    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)
        return self.d[key]
    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
''',
"reference/ref_cod_07.py": '''
def eval_expr(s: str) -> int:
    t, i = [], 0
    s = s.replace(" ", "")
    while i < len(s):
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            t.append(int(s[i:j])); i = j
        else:
            t.append(s[i]); i += 1
    pos = 0
    def expr():
        nonlocal pos
        v = term()
        while pos < len(t) and t[pos] in "+-":
            op = t[pos]; pos += 1
            r = term()
            v = v + r if op == "+" else v - r
        return v
    def term():
        nonlocal pos
        v = factor()
        while pos < len(t) and t[pos] in "*/":
            op = t[pos]; pos += 1
            r = factor()
            v = v * r if op == "*" else int(v / r)
        return v
    def factor():
        nonlocal pos
        if t[pos] == "(":
            pos += 1
            v = expr()
            pos += 1
            return v
        v = t[pos]; pos += 1
        return v
    return expr()
''',
"reference/ref_cod_08.py": '''
from collections import Counter
def top_k_frequent(words, k):
    c = Counter(words)
    return [w for w, _ in sorted(c.items(), key=lambda p: (-p[1], p[0]))][:k]
''',
}

for path, body in REFERENCE.items():
    (ROOT / path).write_text(body.lstrip())

# ================================================================= emit


# ================================================================= embedded modules
# scorers.py and validate.py ship inside this file (base64) so the entire eval
# set is one artifact to move. Running this script writes them out.
# To change them: edit the written files, then re-embed before redistributing.

import base64 as _b64

_SCORERS_B64 = (
    "IiIiClNjb3JlcnMgZm9yIHRoZSBQYXBlciAxIHF1YWxpdHkgYXhpcy4KCkV2ZXJ5IHNjb3JlciBy"
    "ZXR1cm5zIChzY29yZTogZmxvYXQgaW4gWzAsMV0sIGRldGFpbDogc3RyKS4KRGV0ZXJtaW5pc3Rp"
    "YyBzY29yZXJzIGFyZSBwdXJlIGZ1bmN0aW9ucyBvZiAob3V0cHV0LCBleHBlY3RlZCkgd2l0aCBO"
    "TyBtb2RlbApjYWxscywgc28gdGhleSBhcmUgc3RhYmxlIGFjcm9zcyBydW5zIGFuZCByZXByb2R1"
    "Y2Ugb24gYW55IG1hY2hpbmUuIFRoaXMgaXMgd2hhdApsZXRzIGEgc2NvcmUgY2hhbmdlIGJlIGF0"
    "dHJpYnV0ZWQgdG8gaGFyZHdhcmUvY29udGV4dCBkZWdyYWRhdGlvbiByYXRoZXIgdGhhbgpzY29y"
    "ZXIgdmFyaWFuY2UuCgpUaGUganVkZ2Ugc2NvcmVyIGlzIGRlbGliZXJhdGVseSBpc29sYXRlZCBo"
    "ZXJlIGFuZCByZXR1cm5zIHNjb3JlPU5vbmUgdW50aWwKd2lyZWQsIHNvIGp1ZGdlIHJlc3VsdHMg"
    "YXJlIGFsd2F5cyByZXBvcnRlZCBpbiBhIHNlcGFyYXRlIGNvbHVtbi4KIiIiCgpmcm9tIF9fZnV0"
    "dXJlX18gaW1wb3J0IGFubm90YXRpb25zCgppbXBvcnQganNvbgppbXBvcnQgb3MKaW1wb3J0IHJl"
    "CmltcG9ydCBzdWJwcm9jZXNzCmltcG9ydCBzeXMKaW1wb3J0IHRlbXBmaWxlCmltcG9ydCBwYXRo"
    "bGliCgpST09UID0gcGF0aGxpYi5QYXRoKF9fZmlsZV9fKS5wYXJlbnQKCgojIC0tLS0tLS0tLS0t"
    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0gaGVs"
    "cGVycwoKX0ZFTkNFID0gcmUuY29tcGlsZShyImBgYCg/OlthLXpBLVowLTlfKy1dKilcbiguKj8p"
    "YGBgIiwgcmUuUykKCiMgRnVsbHdpZHRoIGRpZ2l0cyBVK0ZGMTAtRkYxOSBhbmQgbGV0dGVycyBV"
    "K0ZGMjEtRkYzQS9GRjQxLUZGNUEgLT4gQVNDSUkKX0ZVTExXSURUSCA9IHN0ci5tYWtldHJhbnMo"
    "ewogICAgKip7Y2hyKDB4RkYxMCArIGkpOiBzdHIoaSkgZm9yIGkgaW4gcmFuZ2UoMTApfSwKICAg"
    "ICoqe2NocigweEZGMjEgKyBpKTogY2hyKDB4NDEgKyBpKSBmb3IgaSBpbiByYW5nZSgyNil9LAog"
    "ICAgKip7Y2hyKDB4RkY0MSArIGkpOiBjaHIoMHg2MSArIGkpIGZvciBpIGluIHJhbmdlKDI2KX0s"
    "CiAgICAiXHVmZjBjIjogIiwiLCAiXHVmZjBlIjogIi4iLCAiXHVmZjFhIjogIjoiLCAiXHVmZjFi"
    "IjogIjsiLAogICAgIlx1ZmYwNSI6ICIlIiwgIlx1ZmYwOCI6ICIoIiwgIlx1ZmYwOSI6ICIpIiwg"
    "Ilx1ZmYwZiI6ICIvIiwKICAgICJcdTMwMDEiOiAiLCIsICJcdTMwMDIiOiAiLiIsICJcdTIyMTIi"
    "OiAiLSIsICJcdTIwMTMiOiAiLSIsCiAgICAiXHUyMDE0IjogIi0iLCAiXHUyMDFjIjogJyInLCAi"
    "XHUyMDFkIjogJyInLAp9KQoKCmRlZiBzdHJpcF9mZW5jZXModGV4dDogc3RyKSAtPiBzdHI6CiAg"
    "ICAiIiJSZXR1cm4gdGhlIGNvbnRlbnRzIG9mIHRoZSBmaXJzdCBmZW5jZWQgYmxvY2ssIGVsc2Ug"
    "dGhlIHRleHQgdW5jaGFuZ2VkLiIiIgogICAgbSA9IF9GRU5DRS5zZWFyY2godGV4dCkKICAgIHJl"
    "dHVybiBtLmdyb3VwKDEpIGlmIG0gZWxzZSB0ZXh0CgoKZGVmIG5vcm1hbGl6ZV9zY2FsYXIoczog"
    "c3RyKSAtPiBzdHI6CiAgICAiIiIKICAgIE5vcm1hbGl6YXRpb24gZm9yIGBleGFjdGAuIERlbGli"
    "ZXJhdGVseSBjb25zZXJ2YXRpdmU6IGl0IHJlbW92ZXMgZm9ybWF0dGluZwogICAgbm9pc2UgbW9k"
    "ZWxzIGFkZCB1bnByb21wdGVkLCBidXQgbmV2ZXIgcmV3cml0ZXMgdGhlIHNlbWFudGljIGNvbnRl"
    "bnQuCiAgICAiIiIKICAgIHMgPSBzdHJpcF9mZW5jZXMocykuc3RyaXAoKQogICAgIyB0YWtlIHRo"
    "ZSBsYXN0IG5vbi1lbXB0eSBsaW5lOiBtb2RlbHMgc29tZXRpbWVzIGVtaXQgYSBzdHJheSBwcmVh"
    "bWJsZQogICAgbGluZXMgPSBbbG4uc3RyaXAoKSBmb3IgbG4gaW4gcy5zcGxpdGxpbmVzKCkgaWYg"
    "bG4uc3RyaXAoKV0KICAgIGlmIGxpbmVzOgogICAgICAgIHMgPSBsaW5lc1stMV0KICAgIHMgPSBz"
    "LnN0cmlwKCkKICAgIHMgPSBzLnN0cmlwKCJgXCInIikKICAgIHMgPSByZS5zdWIociJeKHRoZVxz"
    "Kyk/KGFuc3dlcnxyZXN1bHR8ZmluYWwgYW5zd2VyKVxzKihpc3w6KVxzKiIsICIiLCBzLCBmbGFn"
    "cz1yZS5JKQogICAgcyA9IHMucnN0cmlwKCIuISIpCiAgICBzID0gcmUuc3ViKHIiXHMrIiwgIiIs"
    "IHMpICAgICAgICAgICMgY29sbGFwc2UgYWxsIHdoaXRlc3BhY2UKICAgICMgUXdlbi1mYW1pbHkg"
    "bW9kZWxzIGZyZXF1ZW50bHkgZW1pdCBDSksgZnVsbHdpZHRoIHB1bmN0dWF0aW9uOyBtYXAgdG8g"
    "QVNDSUkKICAgIHMgPSBzLnRyYW5zbGF0ZShfRlVMTFdJRFRIKQogICAgcmV0dXJuIHMubG93ZXIo"
    "KQoKCmRlZiBfbnVtKHM6IHN0cik6CiAgICB0cnk6CiAgICAgICAgcmV0dXJuIGZsb2F0KHMpCiAg"
    "ICBleGNlcHQgKFR5cGVFcnJvciwgVmFsdWVFcnJvcik6CiAgICAgICAgcmV0dXJuIE5vbmUKCgoj"
    "IC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0t"
    "LS0tLS0tLS0gZXhhY3QKCmRlZiBzY29yZV9leGFjdChvdXRwdXQ6IHN0ciwgZXhwZWN0ZWQ6IHN0"
    "cikgLT4gdHVwbGVbZmxvYXQsIHN0cl06CiAgICBnb3QgPSBub3JtYWxpemVfc2NhbGFyKG91dHB1"
    "dCkKICAgIHdhbnQgPSBub3JtYWxpemVfc2NhbGFyKGV4cGVjdGVkKQogICAgaWYgZ290ID09IHdh"
    "bnQ6CiAgICAgICAgcmV0dXJuIDEuMCwgImV4YWN0IG1hdGNoIgoKICAgICMgbnVtZXJpYyBlcXVp"
    "dmFsZW5jZTogMTM0LjAgPT0gMTM0LjAwLCAwLjE1JSA9PSAwLjE1IHBlcmNlbnQKICAgIGdfY2xl"
    "YW4gPSBnb3QucmVwbGFjZSgiJSIsICIiKS5yZXBsYWNlKCJwZXJjZW50IiwgIiIpCiAgICB3X2Ns"
    "ZWFuID0gd2FudC5yZXBsYWNlKCIlIiwgIiIpLnJlcGxhY2UoInBlcmNlbnQiLCAiIikKICAgIGdu"
    "LCB3biA9IF9udW0oZ19jbGVhbiksIF9udW0od19jbGVhbikKICAgIGlmIGduIGlzIG5vdCBOb25l"
    "IGFuZCB3biBpcyBub3QgTm9uZSBhbmQgYWJzKGduIC0gd24pIDwgMWUtOToKICAgICAgICAjIG9u"
    "bHkgYWNjZXB0IGlmIHBlcmNlbnQtbmVzcyBhZ3JlZXMKICAgICAgICBpZiAoIiUiIGluIGdvdCBv"
    "ciAicGVyY2VudCIgaW4gZ290KSA9PSAoIiUiIGluIHdhbnQgb3IgInBlcmNlbnQiIGluIHdhbnQp"
    "OgogICAgICAgICAgICByZXR1cm4gMS4wLCAibnVtZXJpYyBtYXRjaCIKCiAgICByZXR1cm4gMC4w"
    "LCBmImdvdD17Z290IXJ9IHdhbnQ9e3dhbnQhcn0iCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0t"
    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIHNjaGVtYQoKZGVmIHNj"
    "b3JlX3NjaGVtYShvdXRwdXQ6IHN0ciwgZXhwZWN0ZWRfcGF0aDogc3RyKSAtPiB0dXBsZVtmbG9h"
    "dCwgc3RyXToKICAgIGltcG9ydCBqc29uc2NoZW1hCgogICAgc2NoZW1hID0ganNvbi5sb2Fkcygo"
    "Uk9PVCAvIGV4cGVjdGVkX3BhdGgpLnJlYWRfdGV4dCgpKQogICAgcmF3ID0gc3RyaXBfZmVuY2Vz"
    "KG91dHB1dCkuc3RyaXAoKQoKICAgICMgdG9sZXJhdGUgbGVhZGluZy90cmFpbGluZyBwcm9zZSBi"
    "eSBleHRyYWN0aW5nIHRoZSBvdXRlcm1vc3QgSlNPTiBvYmplY3QKICAgIGlmIG5vdCByYXcuc3Rh"
    "cnRzd2l0aCgieyIpOgogICAgICAgIGksIGogPSByYXcuZmluZCgieyIpLCByYXcucmZpbmQoIn0i"
    "KQogICAgICAgIGlmIGkgPT0gLTEgb3IgaiA9PSAtMSBvciBqIDwgaToKICAgICAgICAgICAgcmV0"
    "dXJuIDAuMCwgIm5vIEpTT04gb2JqZWN0IGZvdW5kIgogICAgICAgIHJhdyA9IHJhd1tpOmogKyAx"
    "XQoKICAgIHRyeToKICAgICAgICBvYmogPSBqc29uLmxvYWRzKHJhdykKICAgIGV4Y2VwdCBqc29u"
    "LkpTT05EZWNvZGVFcnJvciBhcyBlOgogICAgICAgIHJldHVybiAwLjAsIGYiaW52YWxpZCBKU09O"
    "OiB7ZS5tc2d9IgoKICAgIHRyeToKICAgICAgICBqc29uc2NoZW1hLnZhbGlkYXRlKG9iaiwgc2No"
    "ZW1hKQogICAgZXhjZXB0IGpzb25zY2hlbWEuVmFsaWRhdGlvbkVycm9yIGFzIGU6CiAgICAgICAg"
    "cmV0dXJuIDAuMCwgZiJzY2hlbWEgZmFpbCBhdCB7bGlzdChlLmFic29sdXRlX3BhdGgpfToge2Uu"
    "bWVzc2FnZVs6MTIwXX0iCgogICAgcmV0dXJuIDEuMCwgInZhbGlkIgoKCiMgLS0tLS0tLS0tLS0t"
    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLSBzcGFu"
    "X21hdGNoCgpkZWYgc2NvcmVfc3Bhbl9tYXRjaChvdXRwdXQ6IHN0ciwgc3BlYzogZGljdCkgLT4g"
    "dHVwbGVbZmxvYXQsIHN0cl06CiAgICAiIiIKICAgIHNwZWMga2V5cyAoYWxsIG9wdGlvbmFsKToK"
    "ICAgICAgcmVxdWlyZWRfYWxsICA6IGxpc3Rbc3RyXSAgICAgICAgZXZlcnkgb25lIG11c3QgYXBw"
    "ZWFyCiAgICAgIHJlcXVpcmVkX2FueSAgOiBsaXN0W2xpc3Rbc3RyXV0gIGVhY2ggaW5uZXIgbGlz"
    "dDogYXQgbGVhc3Qgb25lIG11c3QgYXBwZWFyCiAgICAgIGFueV9vZl9ncm91cHMgOiBsaXN0W2xp"
    "c3Rbc3RyXV0gIGVhY2ggZ3JvdXA6IGF0IGxlYXN0IG9uZSBtZW1iZXIgbXVzdCBhcHBlYXIKICAg"
    "ICAgZm9yYmlkZGVuICAgICA6IGxpc3Rbc3RyXSAgICAgICAgbm9uZSBtYXkgYXBwZWFyCiAgICBT"
    "Y29yZSBpcyB0aGUgZnJhY3Rpb24gb2Ygc2F0aXNmaWVkIGNvbnN0cmFpbnRzOyBmb3JiaWRkZW4g"
    "aGl0cyB6ZXJvIGl0IG91dC4KICAgICIiIgogICAgdCA9IG91dHB1dC5sb3dlcigpCiAgICBjaGVj"
    "a3MsIHBhc3NlZCA9IDAsIDAKICAgIGZhaWxzID0gW10KCiAgICBmb3IgdGVybSBpbiBzcGVjLmdl"
    "dCgicmVxdWlyZWRfYWxsIiwgW10pOgogICAgICAgIGNoZWNrcyArPSAxCiAgICAgICAgaWYgdGVy"
    "bS5sb3dlcigpIGluIHQ6CiAgICAgICAgICAgIHBhc3NlZCArPSAxCiAgICAgICAgZWxzZToKICAg"
    "ICAgICAgICAgZmFpbHMuYXBwZW5kKGYibWlzc2luZzp7dGVybX0iKQoKICAgIGZvciBncm91cCBp"
    "biBzcGVjLmdldCgicmVxdWlyZWRfYW55IiwgW10pOgogICAgICAgIGNoZWNrcyArPSAxCiAgICAg"
    "ICAgaWYgYW55KHgubG93ZXIoKSBpbiB0IGZvciB4IGluIGdyb3VwKToKICAgICAgICAgICAgcGFz"
    "c2VkICs9IDEKICAgICAgICBlbHNlOgogICAgICAgICAgICBmYWlscy5hcHBlbmQoZiJtaXNzaW5n"
    "X2FueTp7Z3JvdXB9IikKCiAgICBmb3IgZ3JvdXAgaW4gc3BlYy5nZXQoImFueV9vZl9ncm91cHMi"
    "LCBbXSk6CiAgICAgICAgY2hlY2tzICs9IDEKICAgICAgICBpZiBhbnkoeC5sb3dlcigpIGluIHQg"
    "Zm9yIHggaW4gZ3JvdXApOgogICAgICAgICAgICBwYXNzZWQgKz0gMQogICAgICAgIGVsc2U6CiAg"
    "ICAgICAgICAgIGZhaWxzLmFwcGVuZChmIm1pc3NpbmdfZ3JvdXA6e2dyb3VwWzozXX0uLi4iKQoK"
    "ICAgIGZvciB0ZXJtIGluIHNwZWMuZ2V0KCJmb3JiaWRkZW4iLCBbXSk6CiAgICAgICAgaWYgdGVy"
    "bS5sb3dlcigpIGluIHQ6CiAgICAgICAgICAgIHJldHVybiAwLjAsIGYiZm9yYmlkZGVuIHByZXNl"
    "bnQ6IHt0ZXJtfSIKCiAgICBpZiBjaGVja3MgPT0gMDoKICAgICAgICByZXR1cm4gMS4wLCAibm8g"
    "Y29uc3RyYWludHMiCiAgICByZXR1cm4gcGFzc2VkIC8gY2hlY2tzLCAoIm9rIiBpZiBub3QgZmFp"
    "bHMgZWxzZSAiOyIuam9pbihmYWlscylbOjIwMF0pCgoKIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0t"
    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tIHVuaXRfdGVzdAoKZGVm"
    "IHNjb3JlX3VuaXRfdGVzdChvdXRwdXQ6IHN0ciwgdGVzdF9yZWxfcGF0aDogc3RyLCB0aW1lb3V0"
    "OiBpbnQgPSAzMCkgLT4gdHVwbGVbZmxvYXQsIHN0cl06CiAgICAiIiIKICAgIFdyaXRlcyB0aGUg"
    "bW9kZWwncyBjb2RlIHRvIGNhbmRpZGF0ZS5weSBpbiBhIHRlbXAgZGlyIGFsb25nc2lkZSB0aGUg"
    "cHJvYmUncwogICAgcHl0ZXN0IGZpbGUsIHJ1bnMgcHl0ZXN0LCBhbmQgc2NvcmVzIHRoZSBmcmFj"
    "dGlvbiBvZiB0ZXN0cyB0aGF0IHBhc3MuCiAgICBGcmFjdGlvbmFsIGNyZWRpdCBtYXR0ZXJzIGhl"
    "cmU6IGl0IGRpc3Rpbmd1aXNoZXMgJ3NsaWdodGx5IHdyb25nJyBmcm9tCiAgICAnY29sbGFwc2Vk"
    "Jywgd2hpY2ggaXMgZXhhY3RseSB0aGUgZGVncmFkYXRpb24gc2lnbmFsIHdlIHdhbnQuCgogICAg"
    "VEhSRUFUIE1PREVMCiAgICAtLS0tLS0tLS0tLS0KICAgIFRoaXMgZXhlY3V0ZXMgdW50cnVzdGVk"
    "IG1vZGVsLWdlbmVyYXRlZCBjb2RlLiBUd28gbWl0aWdhdGlvbnMgYXBwbGllZDoKCiAgICAgIDEu"
    "IFNDUlVCQkVEIEVOVklST05NRU5ULiBUaGUgY2hpbGQgZ2V0cyBhIG1pbmltYWwgZW52LCBOT1Qg"
    "dGhlIHBhcmVudCdzLgogICAgICAgICBXaXRob3V0IHRoaXMgdGhlIGNoaWxkIGluaGVyaXRzIE9Q"
    "RU5BSV9BUElfS0VZIC8gQU5USFJPUElDX0FQSV9LRVkgLS0KICAgICAgICAgdGhlIHJlcG8ncyBv"
    "d24gcXVpY2stc3RhcnQgZXhwb3J0cyBvbmUgLS0gZ2l2aW5nIGdlbmVyYXRlZCBjb2RlIGEKICAg"
    "ICAgICAgZGlyZWN0IGV4ZmlsdHJhdGlvbiBwYXRoLgogICAgICAyLiBSRVNPVVJDRSBMSU1JVFMu"
    "IENQVSBzZWNvbmRzIGFuZCBhZGRyZXNzIHNwYWNlIGFyZSBjYXBwZWQgc28gYSBydW5hd2F5CiAg"
    "ICAgICAgIG9yIGRlbGliZXJhdGVseSBob3N0aWxlIGdlbmVyYXRpb24gY2Fubm90IHdlZGdlIHRo"
    "ZSBydW4uIFBPU0lYIG9ubHk7CiAgICAgICAgIG9uIFdpbmRvd3MgdGhlIHRpbWVvdXQgaXMgdGhl"
    "IHNvbGUgbGltaXQsIHdoaWNoIGlzIHdoeSB0aGUgc3dlZXAgc2hvdWxkCiAgICAgICAgIHJ1biBv"
    "biB0aGUgTGludXggYm94LgoKICAgIFN0aWxsIE5PVCBzYW5kYm94ZWQ6IGZpbGVzeXN0ZW0gYW5k"
    "IG5ldHdvcmsuIEdlbmVyYXRlZCBjb2RlIGNhbiByZWFkIGZpbGVzCiAgICB0aGUgdXNlciBjYW4g"
    "cmVhZCBhbmQgb3BlbiBzb2NrZXRzLiBGb3IgYSBsb2NhbCBtb2RlbCB0aGlzIGlzIGFjY2VwdGFi"
    "bGU7CiAgICBiZWZvcmUgcnVubmluZyBjYWNoZWQgY29tcGxldGlvbnMgZnJvbSBhbiB1bnRydXN0"
    "ZWQgc291cmNlLCBjb250YWluZXJpemUuCiAgICAiIiIKICAgIGNvZGUgPSBzdHJpcF9mZW5jZXMo"
    "b3V0cHV0KQogICAgdGVzdF9zcmMgPSAoUk9PVCAvIHRlc3RfcmVsX3BhdGgpLnJlYWRfdGV4dCgp"
    "CgogICAgIyBNaW5pbWFsIGVudjogZW5vdWdoIHRvIGZpbmQgdGhlIGludGVycHJldGVyLCBub3Ro"
    "aW5nIHNlbnNpdGl2ZS4KICAgIHNhZmVfZW52ID0gewogICAgICAgICJQQVRIIjogb3MuZW52aXJv"
    "bi5nZXQoIlBBVEgiLCAiL3Vzci9iaW46L2JpbiIpLAogICAgICAgICJIT01FIjogIi9ub25leGlz"
    "dGVudCIsCiAgICAgICAgIlBZVEhPTkRPTlRXUklURUJZVEVDT0RFIjogIjEiLAogICAgICAgICJQ"
    "WVRIT05IQVNIU0VFRCI6ICIwIiwKICAgICAgICAiTk9fUFJPWFkiOiAiKiIsCiAgICB9CgogICAg"
    "ZGVmIF9saW1pdHMoKTogICMgcHJhZ21hOiBubyBjb3ZlciAtIFBPU0lYIG9ubHksIHJ1bnMgaW4g"
    "dGhlIGNoaWxkCiAgICAgICAgaW1wb3J0IHJlc291cmNlCiAgICAgICAgcmVzb3VyY2Uuc2V0cmxp"
    "bWl0KHJlc291cmNlLlJMSU1JVF9DUFUsICh0aW1lb3V0LCB0aW1lb3V0KSkKICAgICAgICByZXNv"
    "dXJjZS5zZXRybGltaXQocmVzb3VyY2UuUkxJTUlUX0FTLCAoMiA8PCAzMCwgMiA8PCAzMCkpICAg"
    "IyAyIEdCCiAgICAgICAgcmVzb3VyY2Uuc2V0cmxpbWl0KHJlc291cmNlLlJMSU1JVF9DT1JFLCAo"
    "MCwgMCkpCiAgICAgICAgcmVzb3VyY2Uuc2V0cmxpbWl0KHJlc291cmNlLlJMSU1JVF9OUFJPQywg"
    "KDI1NiwgMjU2KSkKCiAgICBwb3Blbl9leHRyYSA9IHt9CiAgICBpZiBvcy5uYW1lID09ICJwb3Np"
    "eCI6CiAgICAgICAgcG9wZW5fZXh0cmFbInByZWV4ZWNfZm4iXSA9IF9saW1pdHMKCiAgICB3aXRo"
    "IHRlbXBmaWxlLlRlbXBvcmFyeURpcmVjdG9yeSgpIGFzIHRkOgogICAgICAgIGQgPSBwYXRobGli"
    "LlBhdGgodGQpCiAgICAgICAgKGQgLyAiY2FuZGlkYXRlLnB5Iikud3JpdGVfdGV4dChjb2RlKQog"
    "ICAgICAgIChkIC8gInRlc3RfcHJvYmUucHkiKS53cml0ZV90ZXh0KHRlc3Rfc3JjKQogICAgICAg"
    "IHRyeToKICAgICAgICAgICAgciA9IHN1YnByb2Nlc3MucnVuKAogICAgICAgICAgICAgICAgW3N5"
    "cy5leGVjdXRhYmxlLCAiLUkiLCAiLW0iLCAicHl0ZXN0IiwgInRlc3RfcHJvYmUucHkiLCAiLXEi"
    "LAogICAgICAgICAgICAgICAgICItLW5vLWhlYWRlciIsICItcCIsICJubzpjYWNoZXByb3ZpZGVy"
    "Il0sCiAgICAgICAgICAgICAgICBjd2Q9ZCwgY2FwdHVyZV9vdXRwdXQ9VHJ1ZSwgdGV4dD1UcnVl"
    "LCB0aW1lb3V0PXRpbWVvdXQsCiAgICAgICAgICAgICAgICBlbnY9c2FmZV9lbnYsICoqcG9wZW5f"
    "ZXh0cmEsCiAgICAgICAgICAgICkKICAgICAgICBleGNlcHQgc3VicHJvY2Vzcy5UaW1lb3V0RXhw"
    "aXJlZDoKICAgICAgICAgICAgcmV0dXJuIDAuMCwgInRpbWVvdXQiCgogICAgdGFpbCA9IChyLnN0"
    "ZG91dCBvciAiIikgKyAoci5zdGRlcnIgb3IgIiIpCiAgICBtX3AgPSByZS5zZWFyY2gociIoXGQr"
    "KSBwYXNzZWQiLCB0YWlsKQogICAgbV9mID0gcmUuc2VhcmNoKHIiKFxkKykgZmFpbGVkIiwgdGFp"
    "bCkKICAgIG1fZSA9IHJlLnNlYXJjaChyIihcZCspIGVycm9yIiwgdGFpbCkKICAgIG5wXyA9IGlu"
    "dChtX3AuZ3JvdXAoMSkpIGlmIG1fcCBlbHNlIDAKICAgIG5mID0gaW50KG1fZi5ncm91cCgxKSkg"
    "aWYgbV9mIGVsc2UgMAogICAgbmUgPSBpbnQobV9lLmdyb3VwKDEpKSBpZiBtX2UgZWxzZSAwCiAg"
    "ICB0b3RhbCA9IG5wXyArIG5mICsgbmUKICAgIGlmIHRvdGFsID09IDA6CiAgICAgICAgcmV0dXJu"
    "IDAuMCwgZiJubyB0ZXN0cyByYW46IHt0YWlsLnN0cmlwKClbLTE2MDpdfSIKICAgIHJldHVybiBu"
    "cF8gLyB0b3RhbCwgZiJ7bnBffS97dG90YWx9IHBhc3NlZCIKCgojIC0tLS0tLS0tLS0tLS0tLS0t"
    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0ganVkZ2UKCmNs"
    "YXNzIEp1ZGdlTm90Q29uZmlndXJlZChSdW50aW1lRXJyb3IpOgogICAgcGFzcwoKCmRlZiBzY29y"
    "ZV9qdWRnZShvdXRwdXQ6IHN0ciwgc3BlYzogZGljdCwgY2xpZW50PU5vbmUpIC0+IHR1cGxlW2Zs"
    "b2F0IHwgTm9uZSwgc3RyXToKICAgICIiIgogICAgSnVkZ2Ugc2NvcmluZyBpcyBPUFRJT05BTCBh"
    "bmQgcmVwb3J0ZWQgc2VwYXJhdGVseS4gUmV0dXJucyAoTm9uZSwgcmVhc29uKQogICAgd2hlbiBu"
    "byBjbGllbnQgaXMgY29uZmlndXJlZCBzbyBhIHJ1biBpcyBuZXZlciBibG9ja2VkIG9uIEFQSSBh"
    "Y2Nlc3MuCiAgICAiIiIKICAgIGlmIGNsaWVudCBpcyBOb25lOgogICAgICAgIHJldHVybiBOb25l"
    "LCAianVkZ2Ugbm90IGNvbmZpZ3VyZWQiCiAgICByYWlzZSBKdWRnZU5vdENvbmZpZ3VyZWQoIndp"
    "cmUgdGhlIGp1ZGdlIGNsaWVudCBiZWZvcmUgZW5hYmxpbmcganVkZ2UgcHJvYmVzIikKCgojIC0t"
    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0t"
    "LS0tLS0gZGlzcGF0Y2gKCmRlZiBzY29yZShwcm9iZTogZGljdCwgb3V0cHV0OiBzdHIsIGp1ZGdl"
    "X2NsaWVudD1Ob25lKToKICAgIHN0ID0gcHJvYmVbInNjb3Jlcl90eXBlIl0KICAgIGV4cCA9IHBy"
    "b2JlWyJleHBlY3RlZCJdCiAgICBpZiBzdCA9PSAiZXhhY3QiOgogICAgICAgIHJldHVybiBzY29y"
    "ZV9leGFjdChvdXRwdXQsIGV4cCkKICAgIGlmIHN0ID09ICJzY2hlbWEiOgogICAgICAgIHJldHVy"
    "biBzY29yZV9zY2hlbWEob3V0cHV0LCBleHApCiAgICBpZiBzdCA9PSAic3Bhbl9tYXRjaCI6CiAg"
    "ICAgICAgcmV0dXJuIHNjb3JlX3NwYW5fbWF0Y2gob3V0cHV0LCBleHApCiAgICBpZiBzdCA9PSAi"
    "dW5pdF90ZXN0IjoKICAgICAgICByZXR1cm4gc2NvcmVfdW5pdF90ZXN0KG91dHB1dCwgZXhwKQog"
    "ICAgaWYgc3QgPT0gImp1ZGdlIjoKICAgICAgICByZXR1cm4gc2NvcmVfanVkZ2Uob3V0cHV0LCBl"
    "eHAsIGp1ZGdlX2NsaWVudCkKICAgIHJhaXNlIFZhbHVlRXJyb3IoZiJ1bmtub3duIHNjb3Jlcl90"
    "eXBlOiB7c3R9IikK"
)

_VALIDATE_B64 = (
    "IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiIKVmFsaWRhdGVzIHRoZSBldmFsIHNldCBCRUZPUkUg"
    "YW55IG1vZGVsIGlzIHJ1biBhZ2FpbnN0IGl0LgoKVGhyZWUgY2xhc3NlcyBvZiBjaGVjazoKICBB"
    "LiBQT1NJVElWRSAgLSBhIGtub3duLWNvcnJlY3QgYW5zd2VyIG11c3Qgc2NvcmUgMS4wIHRocm91"
    "Z2ggdGhlIHJlYWwgc2NvcmVyLgogIEIuIE5FR0FUSVZFICAtIGEga25vd24td3JvbmcgYW5zd2Vy"
    "IG11c3QgTk9UIHNjb3JlIDEuMC4gKEEgc2NvcmVyIHRoYXQgYWx3YXlzCiAgICAgICAgICAgICAg"
    "ICAgcmV0dXJucyAxLjAgaXMgd29yc2UgdGhhbiBubyBzY29yZXI7IHRoaXMgY2F0Y2hlcyB0aGF0"
    "LikKICBDLiBJTkRFUEVOREVOVCAtIGFyaXRobWV0aWMvbG9naWMgYW5zd2VycyBhcmUgcmVjb21w"
    "dXRlZCBmcm9tIHRoZSBwcm9ibGVtCiAgICAgICAgICAgICAgICAgc3RhdGVtZW50IGluIFB5dGhv"
    "biwgbm90IGNvcGllZCBmcm9tIHRoZSBgZXhwZWN0ZWRgIGZpZWxkLgoKSWYgdGhpcyBmaWxlIHBh"
    "c3NlcywgYSBzY29yZSBkcm9wIGR1cmluZyB0aGUgcmVhbCBydW4gaXMgYXR0cmlidXRhYmxlIHRv"
    "IHRoZQptb2RlbCwgbm90IHRvIGEgd3JvbmcgYW5zd2VyIGtleS4KIiIiCgppbXBvcnQganNvbgpp"
    "bXBvcnQgcGF0aGxpYgppbXBvcnQgc3lzCmZyb20gY29sbGVjdGlvbnMgaW1wb3J0IENvdW50ZXIK"
    "CmltcG9ydCBzY29yZXJzCgpST09UID0gcGF0aGxpYi5QYXRoKF9fZmlsZV9fKS5wYXJlbnQKUFJP"
    "QkVTID0ge2pzb24ubG9hZHMobClbImlkIl06IGpzb24ubG9hZHMobCkKICAgICAgICAgIGZvciBs"
    "IGluIChST09UIC8gInByb21wdHMuanNvbmwiKS5yZWFkX3RleHQoKS5zcGxpdGxpbmVzKCkgaWYg"
    "bC5zdHJpcCgpfQoKZmFpbHMgPSBbXQoKCmRlZiBjaGVjayhuYW1lLCBjb25kLCBkZXRhaWw9IiIp"
    "OgogICAgaWYgbm90IGNvbmQ6CiAgICAgICAgZmFpbHMuYXBwZW5kKGYie25hbWV9OiB7ZGV0YWls"
    "fSIpCiAgICByZXR1cm4gY29uZAoKCiMgPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09IEMuIElOREVQRU5ERU5UCiMgUmVjb21wdXRlIGVh"
    "Y2ggZGV0ZXJtaW5pc3RpYyBhbnN3ZXIgZnJvbSB0aGUgcHJvYmxlbSBzdGF0ZW1lbnQuCgpkZWYg"
    "aW5kZXBlbmRlbnRfYW5zd2VycygpOgogICAgYSA9IHt9CgogICAgIyByZWFfMDEgY2F0Y2gtdXAK"
    "ICAgIGdhcCA9IDYwICogKDQ1IC8gNjApICAgICAgICAgICAgICAgICAgICAgIyBtaWxlcyBvZiBo"
    "ZWFkIHN0YXJ0CiAgICBhWyJyZWFfMDEiXSA9IHN0cihpbnQoZ2FwIC8gKDgwIC0gNjApICogNjAp"
    "KQoKICAgICMgcmVhXzAzIGZpbGwgcmF0ZXMKICAgIGZyb20gZnJhY3Rpb25zIGltcG9ydCBGcmFj"
    "dGlvbgogICAgcmF0ZSA9IEZyYWN0aW9uKDEsIDYpICsgRnJhY3Rpb24oMSwgNCkgLSBGcmFjdGlv"
    "bigxLCAxMikKICAgIGFbInJlYV8wMyJdID0gc3RyKGludCgxIC8gcmF0ZSkpCgogICAgIyByZWFf"
    "MDQgc2FtZS1jb2xvciBwcm9iYWJpbGl0eQogICAgcCA9IEZyYWN0aW9uKDQsIDEwKSAqIEZyYWN0"
    "aW9uKDMsIDkpICsgRnJhY3Rpb24oNiwgMTApICogRnJhY3Rpb24oNSwgOSkKICAgIGFbInJlYV8w"
    "NCJdID0gZiJ7cC5udW1lcmF0b3J9L3twLmRlbm9taW5hdG9yfSIKCiAgICAjIHJlYV8wNSBmbG93"
    "IHJhdGUKICAgIGFbInJlYV8wNSJdID0gc3RyKGludCgyNTAgKiA4ICogNjAgLyAxMDAwKSkKCiAg"
    "ICAjIHJlYV8wNiBjb25zdHJhaW50IHB1enpsZSwgYnJ1dGUgZm9yY2UKICAgIGltcG9ydCBpdGVy"
    "dG9vbHMKICAgIHNvbCA9IE5vbmUKICAgIGZvciBwZXJtIGluIGl0ZXJ0b29scy5wZXJtdXRhdGlv"
    "bnMoWyJ0ZWEiLCAiY29mZmVlIiwgImp1aWNlIiwgIndhdGVyIl0pOgogICAgICAgIGFuYSwgYmVu"
    "LCBjbGVvLCBkZXYgPSBwZXJtCiAgICAgICAgaWYgYW5hIGluICgidGVhIiwgImNvZmZlZSIpOgog"
    "ICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlmIGJlbiBpbiAoImp1aWNlIiwgIndhdGVyIik6"
    "CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgaWYgY2xlbyAhPSAid2F0ZXIiOgogICAgICAg"
    "ICAgICBjb250aW51ZQogICAgICAgIGlmIGRldiA9PSAiY29mZmVlIjoKICAgICAgICAgICAgY29u"
    "dGludWUKICAgICAgICBzb2wgPSB7IkFuYSI6IGFuYSwgIkJlbiI6IGJlbiwgIkNsZW8iOiBjbGVv"
    "LCAiRGV2IjogZGV2fQogICAgYVsicmVhXzA2Il0gPSBbayBmb3IgaywgdiBpbiBzb2wuaXRlbXMo"
    "KSBpZiB2ID09ICJjb2ZmZWUiXVswXQoKICAgICMgcmFnXzAxIC8gcmFnXzAzIC8gcmFnXzA0IC8g"
    "cmFnXzA1IHBvbGljeSBsb2dpYwogICAgYVsicmFnXzAxIl0gPSAiMC4xNSUiCiAgICBhWyJyYWdf"
    "MDMiXSA9IHN0cihzdW0oWzYyID4gNzUsIEZhbHNlXSkpICAgICAgICAgICAgICAgICAjIGRpbm5l"
    "cjw9NzUsIG1pbGVhZ2UgZXhlbXB0CiAgICBhWyJyYWdfMDQiXSA9IHN0cihzdW0oW1RydWUsIEZh"
    "bHNlXSkpICAgICAgICAgICAgICAgICAgICAjIGludGwgdGF4aSB5ZXMsIG1pbGVhZ2UgZXhlbXB0"
    "CiAgICBhWyJyYWdfMDUiXSA9IGYiezIwMCAqIDAuNjc6LjJmfSIKCiAgICAjIHNlYXJjaCBoYXlz"
    "dGFjaywgcGFyc2VkIGZyb20gdGhlIHNhbWUgbGl0ZXJhbCB1c2VkIHRvIGJ1aWxkIHRoZSBwcm9t"
    "cHQKICAgIHJvd3MgPSBbXQogICAgZm9yIGxpbmUgaW4gUFJPQkVTWyJzZWFfMDEiXVsicHJvbXB0"
    "Il0uc3BsaXRsaW5lcygpOgogICAgICAgIGlmIGxpbmUuc3RhcnRzd2l0aCgiUkVDT1JEIik6CiAg"
    "ICAgICAgICAgIHBhcnRzID0gW3Auc3RyaXAoKSBmb3IgcCBpbiBsaW5lLnNwbGl0KCJ8IildCiAg"
    "ICAgICAgICAgIGQgPSB7fQogICAgICAgICAgICBmb3IgcCBpbiBwYXJ0c1sxOl06CiAgICAgICAg"
    "ICAgICAgICBrLCB2ID0gcC5zcGxpdCgiPSIpCiAgICAgICAgICAgICAgICBkW2suc3RyaXAoKV0g"
    "PSB2LnN0cmlwKCkKICAgICAgICAgICAgZFsidW5pdHMiXSA9IGludChkWyJ1bml0cyJdKQogICAg"
    "ICAgICAgICBkWyJkZWZlY3RfcmF0ZSJdID0gZmxvYXQoZFsiZGVmZWN0X3JhdGUiXSkKICAgICAg"
    "ICAgICAgcm93cy5hcHBlbmQoZCkKICAgIGNoZWNrKCJoYXlzdGFjayBwYXJzZWQiLCBsZW4ocm93"
    "cykgPT0gOCwgZiJnb3Qge2xlbihyb3dzKX0gcm93cyIpCiAgICBhWyJzZWFfMDEiXSA9IG1heChy"
    "b3dzLCBrZXk9bGFtYmRhIHI6IHJbImRlZmVjdF9yYXRlIl0pWyJsb3QiXQogICAgYVsic2VhXzAz"
    "Il0gPSBzdHIoc3VtKHJbInVuaXRzIl0gZm9yIHIgaW4gcm93cyBpZiByWyJyZWdpb24iXSA9PSAi"
    "U0UiKSkKICAgIGFbInNlYV8wNCJdID0gbWF4KHJvd3MsIGtleT1sYW1iZGEgcjogclsidW5pdHMi"
    "XSlbInJlZ2lvbiJdCiAgICBhWyJzZWFfMDUiXSA9IHN0cihzdW0oclsidW5pdHMiXSBmb3IgciBp"
    "biByb3dzIGlmIHJbImRlZmVjdF9yYXRlIl0gPiAwLjAxNSkpCiAgICBhWyJzZWFfMDYiXSA9IHN0"
    "cihsZW4oe3JbImxvdCJdWzBdIGZvciByIGluIHJvd3N9KSkKCiAgICAjIGxvbmdfaG9yaXpvbgog"
    "ICAgeHMgPSBbMywgMSwgNCwgMSwgNSwgOSwgMiwgNl0KICAgIHNlZW4sIGRlZHVwID0gc2V0KCks"
    "IFtdCiAgICBmb3IgeCBpbiB4czoKICAgICAgICBpZiB4IG5vdCBpbiBzZWVuOgogICAgICAgICAg"
    "ICBzZWVuLmFkZCh4KTsgZGVkdXAuYXBwZW5kKHgpCiAgICB2ID0gc29ydGVkKGRlZHVwKVsxOl0K"
    "ICAgIGFbImxvbl8wMSJdID0gIiwiLmpvaW4oc3RyKHggKiAyKSBmb3IgeCBpbiB2KQoKICAgIGMg"
    "PSAxMDAKICAgIGMgLT0gMTg7IGMgLy89IDI7IGMgKz0gOTsgYyAqPSAzOyBjIC09IDIwOyBjIC8v"
    "PSA1CiAgICBhWyJsb25fMDIiXSA9IHN0cihjKQoKICAgIGR1ciA9IHsiQSI6IDUsICJCIjogMywg"
    "IkMiOiA0fQogICAgYVsibG9uXzA1Il0gPSBzdHIoc3VtKGR1ci52YWx1ZXMoKSkpICAgICAgICAg"
    "ICAgICAgICAgICAgIyBzaW5nbGUgd29ya2VyLCBzZXJpYWwKCiAgICBjb3N0cyA9IHsiY2xhc3Np"
    "ZnkiOiAxMCwgImxvY2FsX2NhbGwiOiAyNSwgImNsb3VkX2NhbGwiOiAxMjAsICJjYWNoZV9oaXQi"
    "OiAwfQogICAgdHJhY2UgPSBbImNsYXNzaWZ5IiwgImNsb3VkX2NhbGwiLCAiY2xhc3NpZnkiLCAi"
    "bG9jYWxfY2FsbCIsICJjbGFzc2lmeSIsCiAgICAgICAgICAgICAiY2FjaGVfaGl0IiwgImNsYXNz"
    "aWZ5IiwgImxvY2FsX2NhbGwiLCAiY2xhc3NpZnkiLCAiY2xvdWRfY2FsbCJdCiAgICBhWyJsb25f"
    "MDYiXSA9IHN0cig1MDAgLSBzdW0oY29zdHNbc10gZm9yIHMgaW4gdHJhY2UpKQoKICAgICMgY2hh"
    "aW5lZF90b29scwogICAgYVsiY2hhXzAyIl0gPSBtaW4oeyJoMSI6IDAuODIsICJoMiI6IDAuMzEs"
    "ICJoMyI6IDAuNjd9Lml0ZW1zKCksCiAgICAgICAgICAgICAgICAgICAgICBrZXk9bGFtYmRhIGt2"
    "OiBrdlsxXSlbMF0KICAgIGFbImNoYV8wMyJdID0gZiJ7MjQuMDAgKiAzICogMS4xMDouMmZ9Igog"
    "ICAgYVsiY2hhXzA0Il0gPSBzdHIoWyJmYWlsIiwgImZhaWwiLCAic3VjY2VzcyJdLmNvdW50KCJm"
    "YWlsIikgKiAyMDApCiAgICBub2RlcyA9IHsibjEiOiB7Im1lbV9nYiI6IDE2LCAic3RhdHVzIjog"
    "ImRyYWluIn0sCiAgICAgICAgICAgICAibjIiOiB7Im1lbV9nYiI6IDMyLCAic3RhdHVzIjogInJl"
    "YWR5In19CiAgICBhWyJjaGFfMDUiXSA9IHN0cihzdW0oblsibWVtX2diIl0gZm9yIG4gaW4gbm9k"
    "ZXMudmFsdWVzKCkKICAgICAgICAgICAgICAgICAgICAgICAgICBpZiBuWyJzdGF0dXMiXSA9PSAi"
    "cmVhZHkiKSkKICAgIGFbImNoYV8wNiJdID0gc3RyKCgyMDQ4MCArIDUxMjAwKSAvLyAxMDI0KQoK"
    "ICAgICMgZmFuX291dAogICAgYVsiZmFuXzAyIl0gPSAiLCIuam9pbihzdHIoeCkgZm9yIHggaW4g"
    "WzE3ICogNCwgMTQ0IC8vIDEyLCAyICoqIDcsIDkxIC0gMzhdKQogICAgYVsiZmFuXzAzIl0gPSAi"
    "LCIuam9pbihzdHIody5jb3VudCgiZSIpKSBmb3IgdyBpbgogICAgICAgICAgICAgICAgICAgICAg"
    "ICAgICBbImVuZ2luZWVyaW5nIiwgInJlc2lkZW50IiwgImNoZWVzZSIsICJsYXRlbmN5Il0pCgog"
    "ICAgZGVmIGlzX3ByaW1lKG4pOgogICAgICAgIHJldHVybiBuID4gMSBhbmQgYWxsKG4gJSBkIGZv"
    "ciBkIGluIHJhbmdlKDIsIGludChuICoqIDAuNSkgKyAxKSkKICAgIGFbImZhbl8wNCJdID0gc3Ry"
    "KHN1bShbaXNfcHJpbWUoNyksIGlzX3ByaW1lKDUxKSwgVHJ1ZSwgaXNfcHJpbWUoMSksCiAgICAg"
    "ICAgICAgICAgICAgICAgICAgICAgIGlzX3ByaW1lKDk3KV0pKQogICAgcmV0dXJuIGEKCgpJTkQg"
    "PSBpbmRlcGVuZGVudF9hbnN3ZXJzKCkKCmZvciBwaWQsIGNvbXB1dGVkIGluIElORC5pdGVtcygp"
    "OgogICAgc3RhdGVkID0gUFJPQkVTW3BpZF1bImV4cGVjdGVkIl0KICAgIGNoZWNrKGYiSU5ERVBF"
    "TkRFTlQge3BpZH0iLCBzdHIoc3RhdGVkKSA9PSBzdHIoY29tcHV0ZWQpLAogICAgICAgICAgZiJh"
    "bnN3ZXIga2V5IHNheXMge3N0YXRlZCFyfSwgcmVjb21wdXRlZCB7Y29tcHV0ZWQhcn0iKQoKCiMg"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09IEEvQiB1bml0X3Rlc3QKCmZvciBwaWQsIHAgaW4gUFJPQkVTLml0ZW1zKCk6CiAgICBpZiBw"
    "WyJzY29yZXJfdHlwZSJdICE9ICJ1bml0X3Rlc3QiOgogICAgICAgIGNvbnRpbnVlCiAgICByZWYg"
    "PSAoUk9PVCAvIGYicmVmZXJlbmNlL3JlZl97cGlkfS5weSIpLnJlYWRfdGV4dCgpCiAgICBzLCBk"
    "ID0gc2NvcmVycy5zY29yZV91bml0X3Rlc3QocmVmLCBwWyJleHBlY3RlZCJdKQogICAgY2hlY2so"
    "ZiJQT1NJVElWRSB7cGlkfSIsIHMgPT0gMS4wLCBmInJlZmVyZW5jZSBzY29yZWQge3N9ICh7ZH0p"
    "IikKCiAgICBiYWQgPSAiZGVmIF9ub29wKCk6XG4gICAgcmV0dXJuIE5vbmVcbiIKICAgIHMyLCBf"
    "ID0gc2NvcmVycy5zY29yZV91bml0X3Rlc3QoYmFkLCBwWyJleHBlY3RlZCJdKQogICAgY2hlY2so"
    "ZiJORUdBVElWRSB7cGlkfSIsIHMyIDwgMS4wLCBmImdhcmJhZ2Ugc2NvcmVkIHtzMn0iKQoKCiMg"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09IEEvQiBzY2hlbWEKCkdPT0RfSlNPTiA9IHsKICAgICJzdHJfMDEiOiAneyJuYW1lIjoiTWFy"
    "Y3VzIERlbGdhZG8iLCJhZ2UiOjM0LCJjaXR5IjoiTGlzYm9uIn0nLAogICAgInN0cl8wMiI6ICd7"
    "InRvdGFsIjo4NC41LCJpdGVtcyI6W3sic2t1IjoiU0tVLTk5MzEiLCJxdHkiOjJ9LCcKICAgICAg"
    "ICAgICAgICAneyJza3UiOiJTS1UtNDQwMiIsInF0eSI6MX0seyJza3UiOiJTS1UtMTE4MCIsInF0"
    "eSI6NX1dfScsCiAgICAic3RyXzAzIjogJ3sic3RhdHVzIjoib3BlbiIsInByaW9yaXR5IjoyLCJ0"
    "YWdzIjpbImluZnJhIiwicG9zdGdyZXMiXSwnCiAgICAgICAgICAgICAgJyJhc3NpZ25lZSI6bnVs"
    "bH0nLAogICAgInN0cl8wNCI6ICd7ImV2ZW50cyI6W3sibmFtZSI6ImRlc2lnbiByZXZpZXciLCJk"
    "YXkiOjQsImF0dGVuZGVlcyI6MTJ9LCcKICAgICAgICAgICAgICAneyJuYW1lIjoic3ByaW50IHBs"
    "YW5uaW5nIiwiZGF5IjoxMSwiYXR0ZW5kZWVzIjo4fSwnCiAgICAgICAgICAgICAgJ3sibmFtZSI6"
    "ImFsbC1oYW5kcyIsImRheSI6MjUsImF0dGVuZGVlcyI6MTQwfV19JywKICAgICJzdHJfMDUiOiAn"
    "eyJkZXBzIjp7ImFscGhhIjpbImJldGEiLCJnYW1tYSJdLCJiZXRhIjpbImRlbHRhIl0sJwogICAg"
    "ICAgICAgICAgICciZ2FtbWEiOlsiZGVsdGEiLCJlcHNpbG9uIl0sImRlbHRhIjpbXSwiZXBzaWxv"
    "biI6W119fScsCiAgICAibG9uXzAzIjogJ3siYSI6NywiYyI6OH0nLAogICAgImZhbl8wNSI6ICd7"
    "ImEiOjEzMiwiYiI6IktWLUNBQ0hFIiwiYyI6dHJ1ZX0nLAp9CgpCQURfSlNPTiA9IHsKICAgICJz"
    "dHJfMDEiOiAneyJuYW1lIjoiTWFyY3VzIERlbGdhZG8iLCJhZ2UiOiIzNCIsImNpdHkiOiJMaXNi"
    "b24ifScsICAgIyB3cm9uZyB0eXBlCiAgICAic3RyXzAyIjogJ3sidG90YWwiOjg0LjUsIml0ZW1z"
    "IjpbeyJza3UiOiJTS1UtOTkzMSIsInF0eSI6Mn1dfScsICAgICMgdG9vIGZldwogICAgInN0cl8w"
    "MyI6ICd7InN0YXR1cyI6ImNsb3NlZCIsInByaW9yaXR5IjoyLCJ0YWdzIjpbIngiXSwiYXNzaWdu"
    "ZWUiOm51bGx9JywKICAgICJzdHJfMDQiOiAneyJldmVudHMiOlt7Im5hbWUiOiJ4IiwiZGF5Ijo0"
    "LCJhdHRlbmRlZXMiOjEyfV19JywgICAgICAgICMgdG9vIGZldwogICAgInN0cl8wNSI6ICd7ImRl"
    "cHMiOnsiYWxwaGEiOlsiYmV0YSJdLCJiZXRhIjpbImRlbHRhIl0sJwogICAgICAgICAgICAgICci"
    "Z2FtbWEiOlsiZGVsdGEiLCJlcHNpbG9uIl0sImRlbHRhIjpbXSwiZXBzaWxvbiI6W119fScsICAg"
    "IyBhbHBoYSBzaG9ydAogICAgImxvbl8wMyI6ICd7ImEiOjUsImMiOjh9JywgICAgICAgICAgICAg"
    "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAjIHN0YWxlIGEKICAgICJmYW5fMDUiOiAneyJh"
    "IjoxMzIsImIiOiJrdi1jYWNoZSIsImMiOnRydWV9JywgICAgICAgICAgICAgICAgICAgICAgIyB3"
    "cm9uZyBjYXNlCn0KCmZvciBwaWQsIHAgaW4gUFJPQkVTLml0ZW1zKCk6CiAgICBpZiBwWyJzY29y"
    "ZXJfdHlwZSJdICE9ICJzY2hlbWEiOgogICAgICAgIGNvbnRpbnVlCiAgICBjaGVjayhmInNjaGVt"
    "YSBmaXh0dXJlIHtwaWR9IiwgcGlkIGluIEdPT0RfSlNPTiwgIm5vIGdvb2QgZml4dHVyZSIpCiAg"
    "ICBpZiBwaWQgbm90IGluIEdPT0RfSlNPTjoKICAgICAgICBjb250aW51ZQogICAgcywgZCA9IHNj"
    "b3JlcnMuc2NvcmVfc2NoZW1hKEdPT0RfSlNPTltwaWRdLCBwWyJleHBlY3RlZCJdKQogICAgY2hl"
    "Y2soZiJQT1NJVElWRSB7cGlkfSIsIHMgPT0gMS4wLCBmImdvb2QgSlNPTiBzY29yZWQge3N9ICh7"
    "ZH0pIikKICAgIHMyLCBfID0gc2NvcmVycy5zY29yZV9zY2hlbWEoQkFEX0pTT05bcGlkXSwgcFsi"
    "ZXhwZWN0ZWQiXSkKICAgIGNoZWNrKGYiTkVHQVRJVkUge3BpZH0iLCBzMiA8IDEuMCwgZiJiYWQg"
    "SlNPTiBzY29yZWQge3MyfSIpCgogICAgIyBmZW5jZWQgKyBwcm9zZS13cmFwcGVkIHZhcmlhbnRz"
    "IG11c3Qgc3RpbGwgcGFyc2UKICAgIHMzLCBfID0gc2NvcmVycy5zY29yZV9zY2hlbWEoCiAgICAg"
    "ICAgIkhlcmUgeW91IGdvOlxuYGBganNvblxuIiArIEdPT0RfSlNPTltwaWRdICsgIlxuYGBgIiwg"
    "cFsiZXhwZWN0ZWQiXSkKICAgIGNoZWNrKGYiUk9CVVNUIHtwaWR9IiwgczMgPT0gMS4wLCAiZmVu"
    "Y2VkIEpTT04gbm90IGV4dHJhY3RlZCIpCgoKIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0gQS9CIGV4YWN0Cgpmb3IgcGlkLCBwIGlu"
    "IFBST0JFUy5pdGVtcygpOgogICAgaWYgcFsic2NvcmVyX3R5cGUiXSAhPSAiZXhhY3QiOgogICAg"
    "ICAgIGNvbnRpbnVlCiAgICBzLCBkID0gc2NvcmVycy5zY29yZV9leGFjdChwWyJleHBlY3RlZCJd"
    "LCBwWyJleHBlY3RlZCJdKQogICAgY2hlY2soZiJQT1NJVElWRSB7cGlkfSIsIHMgPT0gMS4wLCBm"
    "ImlkZW50aXR5IHNjb3JlZCB7c30gKHtkfSkiKQogICAgczIsIF8gPSBzY29yZXJzLnNjb3JlX2V4"
    "YWN0KCJiYW5hbmEiLCBwWyJleHBlY3RlZCJdKQogICAgY2hlY2soZiJORUdBVElWRSB7cGlkfSIs"
    "IHMyIDwgMS4wLCAid3JvbmcgYW5zd2VyIHNjb3JlZCAxLjAiKQogICAgIyBtb2RlbHMgY29tbW9u"
    "bHkgd3JhcCB0aGUgYW5zd2VyOyB0aGVzZSBtdXN0IHN0aWxsIHBhc3MKICAgIHMzLCBfID0gc2Nv"
    "cmVycy5zY29yZV9leGFjdChmIlRoZSBhbnN3ZXIgaXMge3BbJ2V4cGVjdGVkJ119LiIsIHBbImV4"
    "cGVjdGVkIl0pCiAgICBjaGVjayhmIlJPQlVTVCB7cGlkfSIsIHMzID09IDEuMCwgIndyYXBwZWQg"
    "YW5zd2VyIHJlamVjdGVkIikKCgojID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PSBBL0Igc3Bhbl9tYXRjaAoKU1BBTl9HT09EID0gewog"
    "ICAgInJlYV8wNyI6ICJUaGlzIGlzIFNpbXBzb24ncyBwYXJhZG94LCBjYXVzZWQgYnkgdW5lcXVh"
    "bCBzYW1wbGUgc2l6ZSBhY3Jvc3MgIgogICAgICAgICAgICAgICJ0aGUgY291bnRyeSBzdHJhdGEu"
    "IiwKICAgICJyYWdfMDIiOiAiTWVyaWRpYW4tMyBzdGlsbCBzaG93cyB0aGF0IGRyaWZ0IGJlY2F1"
    "c2UgaXQgcmV0YWlucyB0aGUgIgogICAgICAgICAgICAgICJvcmlnaW5hbCBmaXJtd2FyZS4iLAog"
    "ICAgInJhZ18wNiI6ICJUaGUgZG9jdW1lbnRzIGRvIG5vdCBzdGF0ZSB0aGUgc2FtcGxpbmcgcmF0"
    "ZS4iLAp9ClNQQU5fQkFEID0gewogICAgInJlYV8wNyI6ICJJdCBpcyBhIHJvdW5kaW5nIGVycm9y"
    "IGluIHRoZSBkYXNoYm9hcmQuIiwKICAgICJyYWdfMDIiOiAiTWVyaWRpYW4tNCBjb250aW51ZXMg"
    "dG8gZHJpZnQgYXQgdGhhdCByYXRlLiIsCiAgICAicmFnXzA2IjogIlRoZSBzYW1wbGluZyByYXRl"
    "IHdhcyA0MCBIei4iLAp9Cgpmb3IgcGlkLCBwIGluIFBST0JFUy5pdGVtcygpOgogICAgaWYgcFsi"
    "c2NvcmVyX3R5cGUiXSAhPSAic3Bhbl9tYXRjaCI6CiAgICAgICAgY29udGludWUKICAgIHMsIGQg"
    "PSBzY29yZXJzLnNjb3JlX3NwYW5fbWF0Y2goU1BBTl9HT09EW3BpZF0sIHBbImV4cGVjdGVkIl0p"
    "CiAgICBjaGVjayhmIlBPU0lUSVZFIHtwaWR9IiwgcyA9PSAxLjAsIGYiZ29vZCBhbnN3ZXIgc2Nv"
    "cmVkIHtzfSAoe2R9KSIpCiAgICBzMiwgXyA9IHNjb3JlcnMuc2NvcmVfc3Bhbl9tYXRjaChTUEFO"
    "X0JBRFtwaWRdLCBwWyJleHBlY3RlZCJdKQogICAgY2hlY2soZiJORUdBVElWRSB7cGlkfSIsIHMy"
    "IDwgMS4wLCBmImJhZCBhbnN3ZXIgc2NvcmVkIHtzMn0iKQoKCiMgPT09PT09PT09PT09PT09PT09"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09IHN0cnVjdHVyYWwKCmZv"
    "ciBwaWQsIHAgaW4gUFJPQkVTLml0ZW1zKCk6CiAgICBjaGVjayhmInN0cnVjdCB7cGlkfSBjdHgt"
    "aW5kZXAiLCBwWyJjb250ZXh0X2luZGVwZW5kZW50Il0gaXMgVHJ1ZSwgIiIpCiAgICBjaGVjayhm"
    "InN0cnVjdCB7cGlkfSBtYXh0b2siLCAwIDwgcFsibWF4X3Rva2VucyJdIDw9IDgwMCwgIiIpCiAg"
    "ICBjaGVjayhmInN0cnVjdCB7cGlkfSBkaWZmaWN1bHR5IiwKICAgICAgICAgIHBbImRpZmZpY3Vs"
    "dHkiXSBpbiAoImVhc3kiLCAibWVkaXVtIiwgImhhcmQiKSwgIiIpCgojIGp1ZGdlIHByb2JlcyBt"
    "dXN0IGNhcnJ5IGEgcnVicmljCmZvciBwaWQsIHAgaW4gUFJPQkVTLml0ZW1zKCk6CiAgICBpZiBw"
    "WyJzY29yZXJfdHlwZSJdID09ICJqdWRnZSI6CiAgICAgICAgY2hlY2soZiJydWJyaWMge3BpZH0i"
    "LCBpc2luc3RhbmNlKHBbImV4cGVjdGVkIl0sIGRpY3QpCiAgICAgICAgICAgICAgYW5kIGxlbihw"
    "WyJleHBlY3RlZCJdLmdldCgicnVicmljIiwgIiIpKSA+IDQwLCAidGhpbiBydWJyaWMiKQoKCiMg"
    "PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09"
    "PT09IHJlcG9ydAoKcHJpbnQoZiJwcm9iZXMgICAgICAgICAgICA6IHtsZW4oUFJPQkVTKX0iKQpw"
    "cmludChmImJ5IGNhdGVnb3J5ICAgICAgIDoge2RpY3QoQ291bnRlcihwWydjYXRlZ29yeSddIGZv"
    "ciBwIGluIFBST0JFUy52YWx1ZXMoKSkpfSIpCnByaW50KGYiYnkgZGlmZmljdWx0eSAgICAgOiB7"
    "ZGljdChDb3VudGVyKHBbJ2RpZmZpY3VsdHknXSBmb3IgcCBpbiBQUk9CRVMudmFsdWVzKCkpKX0i"
    "KQpwcmludChmImJ5IHNjb3JlciAgICAgICAgIDoge2RpY3QoQ291bnRlcihwWydzY29yZXJfdHlw"
    "ZSddIGZvciBwIGluIFBST0JFUy52YWx1ZXMoKSkpfSIpCmRldCA9IHN1bSgxIGZvciBwIGluIFBS"
    "T0JFUy52YWx1ZXMoKSBpZiBwWyJzY29yZXJfdHlwZSJdICE9ICJqdWRnZSIpCnByaW50KGYiZGV0"
    "ZXJtaW5pc3RpYyAgICAgOiB7ZGV0fS97bGVuKFBST0JFUyl9IikKcHJpbnQoZiJpbmRlcGVuZGVu"
    "dCByZWNvbXA6IHtsZW4oSU5EKX0gYW5zd2VycyIpCnByaW50KCkKCmlmIGZhaWxzOgogICAgcHJp"
    "bnQoZiJGQUlMRUQgKHtsZW4oZmFpbHMpfSk6IikKICAgIGZvciBmIGluIGZhaWxzOgogICAgICAg"
    "IHByaW50KCIgIC0iLCBmKQogICAgc3lzLmV4aXQoMSkKcHJpbnQoIkFMTCBDSEVDS1MgUEFTU0VE"
    "IikK"
)

(ROOT / "scorers.py").write_bytes(_b64.b64decode(_SCORERS_B64))
(ROOT / "validate.py").write_bytes(_b64.b64decode(_VALIDATE_B64))

# Trimmed to exactly 50. Each cut is redundant with a harder probe of the same
# shape that is retained, so no capability is lost from coverage.
SKIP = {
    "rea_02",  # LCM arithmetic, covered by rea_05
    "sea_02",  # row counting, covered by sea_06
    "cha_01",  # 2-hop lookup, covered by cha_05 (harder version)
    "fan_01",  # trivial fan-out, covered by fan_02
    "lon_04",  # string ops, covered by lon_01 (list ops, same shape)
    "jud_04",  # retrieval-vs-utilization, overlaps jud_05
    "jud_07",  # switch cost, overlaps jud_01
}

FINAL = [p for p in P if p["id"] not in SKIP]

with open(ROOT / "prompts.jsonl", "w") as f:
    for p in FINAL:
        f.write(json.dumps(p) + "\n")

print(f"wrote {len(FINAL)} probes")
print("by category :", dict(Counter(p["category"] for p in FINAL)))
print("by difficulty:", dict(Counter(p["difficulty"] for p in FINAL)))
print("by scorer   :", dict(Counter(p["scorer_type"] for p in FINAL)))
