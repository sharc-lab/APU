"""Stage 1.4 — Resident set measurement by span ablation.

For each art probe, identify the minimum set of spans that must be present
for a correct answer by ablating spans individually rather than truncating
from one end.

Five ablation conditions per probe:
  baseline            — full artifact, confirms ground truth
  no_answer           — artifact with answer span removed (must fail)
  answer_plus_header  — preamble + data header + answer span only
  answer_no_header    — preamble + answer span, no data header
  answer_plus_adjacent — preamble + header + adjacent distractor + answer span

Output per probe: which conditions pass, approximate span_tokens, span
position within the artifact. No filler — this measures the artifact
structure alone, not filler pressure.

10 probes × 5 conditions × 5 reps = 250 calls.
Model: qwen3:4b-instruct.

Usage:
  python -u harness/span_ablation.py
  python -u harness/span_ablation.py --resume
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).parent.parent
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "harness"))

MODEL = "qwen3:4b-instruct"
HOST = "http://localhost:11434"
N_REPS = 5
MAX_TOKENS = 128
OUTFILE = RESULTS / "span_ablation.jsonl"

TARGET_PROBES = [
    "art_01", "art_02", "art_03", "art_04", "art_05",
    "art_06", "art_07", "art_08", "art_09", "art_10",
]

CONDITIONS = [
    "baseline",             # full artifact
    "no_answer",            # artifact minus answer span
    "answer_plus_header",   # preamble + data header + answer span
    "answer_no_header",     # preamble + answer span (no header)
    "answer_plus_adjacent", # preamble + header + adjacent span + answer span
]

# Per-probe span definitions.
# preamble: first framing line of the artifact
# header: data-type header line(s), including table column labels where needed
# answer_text: the exact substring(s) of the artifact that contain the answer value.
#   For multi-line spans this is the full block. The no_answer condition removes
#   this text from the full artifact.
# adjacent_text: a nearby span that could confuse the model (distractor test).
# answer_text_sub: for probes where answer_text is a clause within a longer line,
#   this is the clause to retain; the remainder of the line becomes the adjacent.
#   Set to None when answer_text is the full line or block.
PROBE_SPANS: dict[str, dict] = {
    "art_01": {
        "preamble": "Using only the configuration below, answer the question.",
        "header": "[CONFIG: service=invoice-relay v2.4]",
        "answer_text": '  "listen_port": 51847,',
        "adjacent_text": '  "max_payload_kb": 192,',
        "answer_text_sub": None,
    },
    "art_02": {
        "preamble": "Using only the calibration record below, answer the question.",
        "header": "[CALIBRATION RECORD: unit=CHROM-7, facility=Kessler-B]",
        "answer_text": "alert_threshold_ppb : 0.0073",
        "adjacent_text": "baseline_ppm        : 12.4",
        "answer_text_sub": None,
    },
    "art_03": {
        "preamble": "Using only the job summary table below, answer the question.",
        # header includes the column label row so the model can interpret the table
        "header": "[JOB SUMMARY: facility=KILO-3, period=2025-Q1]\nJOB-ID  | STATUS   | CPU_HOURS | QUEUE",
        "answer_text": "J-10894 | complete | 8.9       | priority",
        "adjacent_text": "J-10893 | complete | 27.1      | normal",
        "answer_text_sub": None,
    },
    "art_04": {
        "preamble": "Using only the access record below, answer the question.",
        "header": "[ACCESS RECORD: system=payroll-v3, date=2025-03-08]",
        "answer_text": "14:02  DELETE  tcosta",
        "adjacent_text": "10:33  QUERY   tcosta",
        "answer_text_sub": None,
    },
    "art_05": {
        "preamble": "Using only the field survey report below, answer the question.",
        "header": (
            "[FIELD SURVEY REPORT: site=Halcourt Peninsula, Sector 7]\n"
            "Date: 2025-03-14. Investigator: Dr. K. Tamboli."
        ),
        # Sentence containing the grid reference answer
        "answer_text": (
            "The sediment core extracted at grid reference N47.3 / W12.8 shows\n"
            "laminar deposition consistent with seasonal flooding."
        ),
        "adjacent_text": (
            "Layer analysis\n"
            "indicates an average clay fraction of 0.31 in the lower 40 cm."
        ),
        "answer_text_sub": None,
    },
    "art_06": {
        "preamble": "Using only the meeting minutes below, answer the question.",
        # header includes the attendees line and item topic so the question is
        # interpretable; answer_text is just the Second clause
        "header": (
            "[ENGINEERING REVIEW BOARD — Session 2025-02-11]\n"
            "Attendees: D. Okonkwo (chair), S. Nakagawa, V. Herrera, R. Mäkinen, T. Blum\n"
            "\n"
            "Item 3: Proposal to retire the DELPHI-2 indexing service by Q3 2025."
        ),
        # The full line is "  Motion: V. Herrera. Second: T. Blum."
        # answer_text is the second-clause only; no_answer removes it from the line,
        # leaving "  Motion: V. Herrera."
        "answer_text": "Second: T. Blum.",
        # adjacent is the motion clause on the same line (the confusable distractor)
        "adjacent_text": "Motion: V. Herrera.",
        "answer_text_sub": None,
    },
    "art_07": {
        "preamble": "Using only the release notes below, answer the question.",
        "header": "[RELEASE NOTES — Ferrite ORM]",
        # Full 3.11.9 version block (the answer is the version number "3.11.9")
        "answer_text": (
            "Version 3.11.9 (2024-12-05)\n"
            "  - Patched CVE-2024-51022: stack overflow in schema diff with cyclic references\n"
            "  - Cursor iteration no longer holds an open transaction on connection return"
        ),
        # Adjacent: the 3.12.0 block immediately above
        "adjacent_text": (
            "Version 3.12.0 (2025-01-30)\n"
            "  - Added support for nullable composite foreign keys (GH-8841)\n"
            "  - Bulk insert now respects on_conflict=REPLACE for SQLite targets\n"
            "  - Fixed incorrect OFFSET behaviour when combined with GROUP BY on MySQL 8.4+"
        ),
        "answer_text_sub": None,
    },
    "art_08": {
        "preamble": "Using only the inventory report below, answer the question.",
        # header includes the table section label and column header rows
        "header": (
            "DISCONTINUED SKUS\n"
            "| SKU      | Description                        | Last shipment |\n"
            "|----------|------------------------------------|---------------|"
        ),
        "answer_text": "| PN-38901 | Molex-to-SATA adapter, legacy type | Mar 28        |",
        "adjacent_text": "| PN-38847 | 4-pin PWM splitter, 90 cm cable    | Apr 17        |",
        "answer_text_sub": None,
    },
    "art_09": {
        "preamble": "Using only the deployment manifest below, answer the question.",
        "header": (
            "[DEPLOYMENT MANIFEST: svc-inference-v4]\n"
            "Generated: 2025-03-02T11:42:17Z"
        ),
        "answer_text": "keepalive_timeout_s:     73",
        # adjacent: the comment block immediately above the answer field
        "adjacent_text": (
            "# The keep-alive timeout is set below the upstream proxy timeout of 95 s\n"
            "# to prevent truncated responses at the proxy layer. Do not increase\n"
            "# without coordinating with the network team."
        ),
        "answer_text_sub": None,
    },
    "art_10": {
        "preamble": "Using only the diff output below, answer the question.",
        # header includes the diff file headers and function context line
        "header": (
            "[DIFF: repo=flux-collector, branch=hotfix/metric-clamp]\n"
            "--- a/src/collector/pipeline.py\n"
            "+++ b/src/collector/pipeline.py\n"
            "@@ -118,7 +118,7 @@\n"
            "     def _flush_buffer(self, force: bool = False) -> int:\n"
            "         if not force and len(self._buf) < self._threshold:\n"
            "             return 0"
        ),
        "answer_text": '+        clamp_val = self._config.get("metric_clamp", 1847293)',
        # adjacent: the removed (-) line that shows what was there before
        "adjacent_text": "-        clamp_val = MAX_METRIC_VALUE",
        "answer_text_sub": None,
    },
}


def _count_fn(text: str) -> int:
    payload = json.dumps({
        "model": MODEL,
        "prompt": text,
        "options": {"num_ctx": 8192, "num_predict": 1, "temperature": 0},
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["prompt_eval_count"]


def _call(prompt: str):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": 8192, "num_predict": MAX_TOKENS, "temperature": 0},
        "stream": False,
        "think": False,
    }).encode()
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
        content = data["message"]["content"].strip()
        return content, time.monotonic() - t0, data.get("eval_count"), data.get("prompt_eval_count"), data.get("done_reason")
    except Exception as e:
        print(f"    [CALL ERROR: {e}]")
        return None, time.monotonic() - t0, None, None, None


def build_condition_artifact(condition: str, full_artifact: str, spans: dict) -> str:
    """Return artifact text for the given ablation condition."""
    preamble = spans["preamble"]
    header = spans["header"]
    answer_text = spans["answer_text"]
    adjacent_text = spans["adjacent_text"]

    if condition == "baseline":
        return full_artifact

    if condition == "no_answer":
        # Remove the answer_text substring from the full artifact.
        # Try exact match first; if not found, warn.
        if answer_text in full_artifact:
            ablated = full_artifact.replace(answer_text, "")
            # Clean up any double blank lines left by the removal
            while "\n\n\n" in ablated:
                ablated = ablated.replace("\n\n\n", "\n\n")
            return ablated.strip()
        else:
            print(f"  [WARN] answer_text not found in artifact for no_answer ablation", flush=True)
            return full_artifact

    if condition == "answer_plus_header":
        return f"{preamble}\n\n{header}\n{answer_text}"

    if condition == "answer_no_header":
        return f"{preamble}\n\n{answer_text}"

    if condition == "answer_plus_adjacent":
        return f"{preamble}\n\n{header}\n{adjacent_text}\n{answer_text}"

    raise ValueError(f"Unknown condition: {condition!r}")


def _load_completed_jsonl(outfile, keys):
    completed = set()
    if not outfile.exists():
        return completed
    for line in outfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            completed.add(tuple(r[k] for k in keys))
        except (json.JSONDecodeError, KeyError):
            pass
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip cells already in OUTFILE")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if OUTFILE.exists() and not args.resume:
        n = sum(1 for l in OUTFILE.read_text(encoding="utf-8").splitlines() if l.strip())
        if n > 0:
            print(f"ERROR: {OUTFILE.name} has {n} existing rows. Pass --resume to continue, or delete to restart.", flush=True)
            sys.exit(1)

    spec = importlib.util.spec_from_file_location("scorers", PROBES_DIR / "scorers.py")
    scorers_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorers_mod)

    segments = {
        s["id"]: s
        for s in [
            json.loads(l)
            for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        if s["id"] in TARGET_PROBES
    }

    print("Stage 1.4 — Span ablation")
    print(f"Model : {MODEL}")
    print(f"Probes: {TARGET_PROBES}")
    print(f"Conditions: {CONDITIONS}")
    print(f"N_reps: {N_REPS}")
    print()

    # Measure token counts for each probe × condition (no model calls — just counting)
    print("Measuring span token counts...")
    span_tokens: dict[str, dict[str, int]] = {}
    for pid in TARGET_PROBES:
        seg = segments[pid]
        spans = PROBE_SPANS[pid]
        span_tokens[pid] = {}
        full_artifact = seg["artifact"]
        for cond in CONDITIONS:
            artifact_text = build_condition_artifact(cond, full_artifact, spans)
            prompt = f"{artifact_text}\n\n{seg['question']}"
            try:
                tok = _count_fn(prompt)
                span_tokens[pid][cond] = tok
            except Exception as e:
                print(f"  [token count error {pid} {cond}: {e}]", flush=True)
                span_tokens[pid][cond] = -1
        # Also compute baseline answer_text token count
        try:
            span_tokens[pid]["answer_span_only"] = _count_fn(spans["answer_text"])
        except Exception as e:
            span_tokens[pid]["answer_span_only"] = -1
        print(
            f"  {pid}: "
            + " | ".join(f"{c}={span_tokens[pid].get(c,'?')}" for c in CONDITIONS)
            + f" | answer_span={span_tokens[pid].get('answer_span_only','?')}",
            flush=True,
        )
    print()

    total_calls = len(TARGET_PROBES) * len(CONDITIONS) * N_REPS
    _CELL_KEYS = ["probe_id", "condition", "rep"]
    completed = _load_completed_jsonl(OUTFILE, _CELL_KEYS) if args.resume else set()
    print(f"Total calls: {total_calls}  (completed: {len(completed)})")
    print()

    call_n = 0
    written = 0

    with OUTFILE.open("a", encoding="utf-8") as out_fh:
        for pid in TARGET_PROBES:
            seg = segments[pid]
            spans = PROBE_SPANS[pid]
            full_artifact = seg["artifact"]
            probe_dict = {
                "id": pid,
                "scorer_type": seg["scorer_type"],
                "expected": seg["expected"],
            }

            for cond in CONDITIONS:
                artifact_text = build_condition_artifact(cond, full_artifact, spans)
                prompt = f"{artifact_text}\n\n{seg['question']}"
                cond_tokens = span_tokens[pid].get(cond, -1)
                answer_span_tokens = span_tokens[pid].get("answer_span_only", -1)

                # Compute position of answer span in full artifact (char offset)
                answer_char_start = full_artifact.find(spans["answer_text"])
                answer_char_end = answer_char_start + len(spans["answer_text"]) if answer_char_start >= 0 else -1
                full_chars = len(full_artifact)
                answer_position = round(answer_char_start / full_chars, 4) if answer_char_start >= 0 else -1

                for rep in range(N_REPS):
                    call_n += 1
                    cell = (pid, cond, rep)
                    if cell in completed:
                        print(f"[{call_n:3d}/{total_calls}] SKIP {pid} {cond} rep={rep}")
                        continue

                    output, latency, eval_count, prompt_eval_count, done_reason = _call(prompt)
                    score, score_detail = (
                        scorers_mod.score(probe_dict, output) if output is not None
                        else (None, "no output")
                    )
                    outcome = scorers_mod.outcome_class(output, score, truncated=False)

                    row = {
                        "probe_id": pid,
                        "model": MODEL,
                        "condition": cond,
                        "rep": rep,
                        "answer_span_tokens": answer_span_tokens,
                        "condition_prompt_chars": len(prompt),
                        "answer_position": answer_position,
                        "output": output,
                        "score": score,
                        "score_detail": score_detail,
                        "outcome": outcome,
                        "eval_count": eval_count,
                        "prompt_eval_count": prompt_eval_count,
                        "done_reason": done_reason,
                        "latency_s": round(latency, 3),
                        "hardware": "blade14_rtx4070",
                    }
                    out_fh.write(json.dumps(row) + "\n")
                    out_fh.flush()
                    written += 1

                    status = "✓" if score == 1.0 else ("A" if outcome == "abstained" else "✗")
                    print(
                        f"[{call_n:3d}/{total_calls}] {pid} {cond:<22} rep={rep}"
                        f" {status} {latency*1000:.0f}ms  {repr((output or '')[:45])}",
                        flush=True,
                    )
                    if call_n % 10 == 0:
                        print(f"--- progress: {call_n}/{total_calls} calls, {written} written ---", flush=True)

    print(f"\nDone. {written} new rows → {OUTFILE}")
    _print_summary(OUTFILE, span_tokens)


def _print_summary(outfile: Path, span_tokens: dict) -> None:
    rows = []
    for line in outfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    from collections import defaultdict
    print("\n=== SCORE BY PROBE × CONDITION (mean over reps) ===")
    print(f"  {'probe':<8} " + " ".join(f"{c[:5]:>6}" for c in CONDITIONS))
    for pid in TARGET_PROBES:
        parts = []
        for cond in CONDITIONS:
            cell_rows = [r for r in rows if r["probe_id"] == pid and r["condition"] == cond]
            scores = [r["score"] for r in cell_rows if r["score"] is not None]
            mean = sum(scores) / len(scores) if scores else float("nan")
            parts.append(f"{mean:6.2f}")
        print(f"  {pid:<8} " + " ".join(parts))

    print("\n=== REQUIRED SPAN ANALYSIS ===")
    print(f"  {'probe':<8} {'answer_span_tok':>15}  baseline_pass  no_answer_pass  min_cond")
    for pid in TARGET_PROBES:
        baseline_scores = [r["score"] for r in rows if r["probe_id"] == pid and r["condition"] == "baseline" and r["score"] is not None]
        no_ans_scores = [r["score"] for r in rows if r["probe_id"] == pid and r["condition"] == "no_answer" and r["score"] is not None]
        baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else float("nan")
        no_ans_mean = sum(no_ans_scores) / len(no_ans_scores) if no_ans_scores else float("nan")
        span_tok = span_tokens.get(pid, {}).get("answer_span_only", "?")

        # Find the minimum sufficient condition (first non-baseline that passes)
        min_cond = "none"
        for cond in ["answer_plus_header", "answer_plus_adjacent", "answer_no_header"]:
            cond_scores = [r["score"] for r in rows if r["probe_id"] == pid and r["condition"] == cond and r["score"] is not None]
            if cond_scores and sum(cond_scores) / len(cond_scores) >= 1.0:
                min_cond = cond
                break

        print(f"  {pid:<8} {str(span_tok):>15}  {baseline_mean:>13.2f}  {no_ans_mean:>14.2f}  {min_cond}")

    print("\n=== FAILURES IN BASELINE CONDITION ===")
    baseline_failures = [r for r in rows if r["condition"] == "baseline" and (r["score"] or 0) < 1.0]
    if not baseline_failures:
        print("  None.")
    for r in baseline_failures:
        print(f"  {r['probe_id']} rep={r['rep']} score={r['score']} out={repr((r['output'] or '')[:50])}")

    # Write summary JSON
    out = RESULTS / "span_ablation.json"
    cond_means: dict[str, dict[str, float]] = defaultdict(dict)
    for pid in TARGET_PROBES:
        for cond in CONDITIONS:
            cell_rows = [r for r in rows if r["probe_id"] == pid and r["condition"] == cond]
            scores = [r["score"] for r in cell_rows if r["score"] is not None]
            cond_means[pid][cond] = round(sum(scores) / len(scores), 4) if scores else None

    summary = {
        "experiment": "span_ablation",
        "stage": "1.4",
        "date_run": time.strftime("%Y-%m-%d"),
        "model": MODEL,
        "n_reps": N_REPS,
        "conditions": CONDITIONS,
        "total_rows": len(rows),
        "span_tokens": span_tokens,
        "condition_mean_scores": dict(cond_means),
        "rows": rows,
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary JSON → {out}")


if __name__ == "__main__":
    main()
