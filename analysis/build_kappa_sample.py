"""Builds the kappa (inter-rater agreement) study sample for the probe scorer vs human judgment.

Source data: results/labeling/r1b_wrong_sample.csv (committed, commit c7cb214), itself a 150-row
stratified sample of real R1b art_* wrong-answer rows built by analysis/build_r1b_wrong_sample.py.
That file is the only committed artifact in this repo with verbatim R1b art_* model outputs; the raw
JSONL files it was drawn from were never committed (see docs/FINDINGS.md, "A3/A6 audit" section).
This script does not re-sample from scratch -- it reuses that existing, real, already-stratified
150-row population rather than inventing a new, smaller one, so the stratification below is the one
build_r1b_wrong_sample.py already performed (round-robin across model x ratio x arm x machine strata
present in the wrong-answer population, capped at 150 rows).

Automated scorer classification: computed by calling evaluation/outcome.py's classify() -- the real,
tested, current four-way outcome classifier in this repo (CORRECT / REFUSED / FABRICATED /
UNCLASSIFIABLE), loaded by file path exactly as outcome.py itself loads scorers.py, to avoid pulling
in evaluation/__init__.py's optional-dependency chain (openai). Each row's probe_id is recovered from
its sample id (format r1bw_<idx>_<machine>_<model>_<probe>) to determine scorer_type (span_match for
art_05, exact for all other art_* probes) via evaluation/probes/artifact.jsonl. `score` is passed as
0.0 for every row: this is safe because build_r1b_wrong_sample.py already excluded every score==1.0
row from this population, and classify()'s only use of the numeric score value is the `score == 1.0`
check -- any other value routes to the same abstention/fabrication logic regardless of magnitude.

Annotation (first-pass "your_labels"): computed independently by
classify_human_label() below, which is deliberately NOT a call into scorers.py or evaluation/outcome.py
-- it is a separately-written reading of each row's stripped output against its expected answer, using
its own abstention vocabulary and its own exact/embedded-match check. This is what lets the resulting
kappa be a real inter-rater comparison instead of a tautology (comparing a function to itself).

NOTE: the `annotator_claude` column is this operator's own annotation (produced by this script's
classify_human_label(), a Claude session's independent read of each row), not a real external human
rater. It was named `human_label` until 2026-10-01; renamed to `annotator_claude` so the column name
does not imply an external human annotator.

Usage: py -3.12 analysis/build_kappa_sample.py
Writes:
  results/labeling/kappa_sample.csv            -- full sheet: both raters' labels, for auditing
  results/labeling/kappa_sample_blinded.csv    -- scorer_classification removed, annotator_claude
                                                   column empty, for any *new* rater to fill in unanchored
  results/labeling/kappa_sample_your_labels.csv -- same shape as the blinded file, annotator_claude
                                                   filled in with this script's first-pass labels (this
                                                   session's own read, serving as "the annotator" rater)
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_CSV = ROOT / "results" / "labeling" / "r1b_wrong_sample.csv"
OUT_DIR = ROOT / "results" / "labeling"


def _load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


outcome = _load_module("outcome", "evaluation/outcome.py")


def _load_scorer_types():
    """probe_id -> scorer_type, from the real probe definitions."""
    types = {}
    path = ROOT / "evaluation" / "probes" / "artifact.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        types[d["id"]] = d["scorer_type"]
    return types


def probe_id_from_sample_id(sample_id: str) -> str:
    """r1bw_000_evo-x2_qwen3-14b_art_01 -> art_01 (probe ids are always art_NN, two tokens)."""
    parts = sample_id.split("_")
    return f"{parts[-2]}_{parts[-1]}"


# ---------------------------------------------------------------- automated scorer classification

def scorer_classification(row: dict, scorer_types: dict) -> str:
    probe_id = probe_id_from_sample_id(row["id"])
    scorer_type = scorer_types.get(probe_id, "exact")
    result = outcome.classify(
        output=row["output"],
        expected=row["expected"],
        scorer_type=scorer_type,
        score=0.0,  # safe: see module docstring -- every row here already has score != 1.0
        done_reason=None,
    )
    return result["outcome_class"]


# ---------------------------------------------------------------- independent human classification

_AVAILABLE_RE = re.compile(r"^\s*AVAILABLE\s*:\s*(?:yes|no)\s*[,:]?\s*", re.I)
_BARE_UNAVAILABLE_RE = re.compile(r"^\s*AVAILABLE\s*:\s*no\s*[,.]?\s*$", re.I)

_REFUSAL_PATTERNS = [
    r"\bnot (provided|present|mentioned|specified|stated|given|available|found|contained|included)\b",
    r"\bno (information|data|mention|record|value|details)\b",
    r"\b(does not|doesn'?t|do not) (mention|contain|include|specify|provide|state)\b",
    r"\b(cannot|can'?t|unable to) (determine|find|identify|answer|tell)\b",
    r"\bimpossible to (determine|answer|tell)\b",
    r"\binsufficient (information|data|context)\b",
    r"\bnot enough information\b",
    r"\bisn'?t (mentioned|provided|specified|available)\b",
    # "no <noun phrase> was/were/is/are <provided/recorded/...>" -- a common
    # denial construction distinct from the simpler patterns above.
    r"\bno\b[\w\s,.()*-]{0,100}\b(?:was|were|is|are)\b[\w\s]{0,25}\b"
    r"(?:recorded|provided|mentioned|given|available|found|stated|specified|included|present)\b",
    r"\bthere is no answer\b",
    r"\bnone of\b[\w\s,.()*-]{0,60}\bmentions?\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.I)

# A hedge that is followed by a concrete value anyway -- still a fabrication, not a genuine refusal,
# even though refusal language appears somewhere in the output. Deliberately requires an adversative
# conjunction (but/however/though/yet/still) or an explicit answer-declaration phrase before the
# value, NOT just any occurrence of a word like "about" -- an earlier version of this regex matched
# "about" inside ordinary refusal prose ("no information about X is provided"), which is not a hedge
# at all and produced false FABRICATED labels on genuine refusals.
_ANSWER_AFTER_HEDGE_RE = re.compile(
    r"\b(?:but|however|though|yet|still)\b[^.]{0,80}?\b"
    r"(?:likely|probably|approximately|around|would be|is about)\s*[0-9A-Za-z]"
    r"|\b(?:the answer is|answer would be|result is|value is|that is)\s*[0-9A-Za-z]",
    re.I,
)


def _strip_available(s: str) -> str:
    return _AVAILABLE_RE.sub("", s or "", count=1)


def _norm_scalar(s: str) -> str:
    s = (s or "").strip().strip("`\"'")
    s = re.sub(r"^(the\s+)?(answer|result|final answer)\s*(is|:)\s*", "", s, flags=re.I)
    s = s.rstrip(".!")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def classify_human_label(row: dict) -> tuple[str, str]:
    """Returns (label, reason). label is one of CORRECT/REFUSED/FABRICATED/UNCLASSIFIABLE, using the
    same vocabulary as evaluation/outcome.py so the two raters' labels sit on the same scale -- but
    computed by an independently written reading of the row, not by calling that module."""
    raw_output = row["output"] or ""
    expected_raw = row["expected"]
    stripped = _strip_available(raw_output).strip()

    if _BARE_UNAVAILABLE_RE.match(raw_output):
        return "REFUSED", "bare 'AVAILABLE: no' with no further content -- explicit unavailability signal"

    if not stripped:
        return "UNCLASSIFIABLE", "empty output after stripping AVAILABLE: prefix, and not a bare AVAILABLE:no"

    if expected_raw.strip().startswith("{"):
        try:
            spec = ast.literal_eval(expected_raw)
        except Exception:
            spec = None
        if spec:
            t = stripped.lower()
            req = spec.get("required_all", [])
            if req and all(term.lower() in t for term in req):
                return "CORRECT", "all required_all terms present"
    else:
        last_line = stripped.splitlines()[-1] if stripped.splitlines() else stripped
        if _norm_scalar(last_line) == _norm_scalar(expected_raw):
            return "CORRECT", "normalized output matches expected scalar"
        exp_n = _norm_scalar(expected_raw)
        out_n = _norm_scalar(stripped)
        if exp_n and len(exp_n) >= 3 and exp_n in out_n:
            return "CORRECT", "expected value found embedded in output"

    if _REFUSAL_RE.search(stripped):
        if _ANSWER_AFTER_HEDGE_RE.search(stripped):
            return "FABRICATED", "refusal language present but a concrete value follows a hedge"
        return "REFUSED", "declines to answer, no concrete value given"

    return "FABRICATED", "gives a specific, confident, wrong-looking answer with no refusal language"


# ---------------------------------------------------------------- main

FULL_FIELDS = [
    "id", "model", "ratio", "arm", "machine", "question", "expected", "output",
    "visible_prompt_excerpt_near_answer", "legacy_heuristic_label",
    "scorer_classification", "annotator_claude", "annotator_claude_reason",
]
BLIND_FIELDS = [
    "id", "model", "ratio", "arm", "machine", "question", "expected", "output",
    "visible_prompt_excerpt_near_answer", "annotator_claude",
]


def main():
    scorer_types = _load_scorer_types()
    with open(SRC_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 150:
        print(f"WARNING: expected 150 rows in {SRC_CSV}, found {len(rows)}", file=sys.stderr)

    full_rows = []
    for r in rows:
        sc = scorer_classification(r, scorer_types)
        hl, reason = classify_human_label(r)
        full_rows.append({
            "id": r["id"], "model": r["model"], "ratio": r["ratio"], "arm": r["arm"],
            "machine": r["machine"], "question": r["question"], "expected": r["expected"],
            "output": r["output"], "visible_prompt_excerpt_near_answer": r["visible_prompt_excerpt_near_answer"],
            "legacy_heuristic_label": r["heuristic_label"],
            "scorer_classification": sc, "annotator_claude": hl, "annotator_claude_reason": reason,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "kappa_sample.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FULL_FIELDS)
        w.writeheader()
        w.writerows(full_rows)

    with open(OUT_DIR / "kappa_sample_blinded.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BLIND_FIELDS)
        w.writeheader()
        for r in full_rows:
            blind = {k: r[k] for k in BLIND_FIELDS if k != "annotator_claude"}
            blind["annotator_claude"] = ""
            w.writerow(blind)

    with open(OUT_DIR / "kappa_sample_your_labels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BLIND_FIELDS)
        w.writeheader()
        for r in full_rows:
            w.writerow({k: r[k] for k in BLIND_FIELDS})

    print(f"wrote {len(full_rows)} rows to kappa_sample.csv, kappa_sample_blinded.csv, "
          f"kappa_sample_your_labels.csv")


if __name__ == "__main__":
    main()
