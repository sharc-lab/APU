"""Truncation experiment for art_* artifact-bearing probes.

Both models, EARLY arm, 4 ratios, 2 arms, 3 reps.
  10 probes × 4 ratios × 2 models × 2 arms × 3 reps = 480 calls.

Ratios:
  1.20 — untruncated baseline (confirms headroom under experimental condition)
  0.98 — partial-artifact window (from Stage 2: most probes extinct by r=0.96)
  0.85 — deep truncation (Stage C equivalent)
  0.40 — very deep truncation

Arms:
  arm1_baseline    — no modification to prompt
  arm3_self_report — appends: "First state whether the information needed to
                     answer is present above, then answer.
                     Format: AVAILABLE: yes|no, then the answer."

Results written to results/art_truncation.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).parent.parent
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "harness"))

import context as ctx_mod

MODELS = ["qwen3:4b-instruct", "llama3.1:8b"]
HOST = "http://localhost:11434"
FILLER_TARGET = 4000
BUDGET_RATIOS = [1.20, 0.98, 0.85, 0.40]
N_REPS = 3
MAX_TOKENS = 256
OUTFILE = RESULTS / "art_truncation.jsonl"

ARM1_SUFFIX = ""
ARM3_SUFFIX = (
    "\n\nFirst state whether the information needed to answer is present above, "
    "then answer. Format: AVAILABLE: yes|no, then the answer."
)

_AVAIL_RE = re.compile(r"AVAILABLE\s*:\s*(yes|no)", re.I)


def _count_fn(text: str, model: str = "qwen3:4b-instruct") -> int:
    payload = json.dumps({
        "model": model,
        "prompt": text,
        "options": {"num_ctx": 8192, "num_predict": 1, "temperature": 0},
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["prompt_eval_count"]


def _call(prompt, model):
    extra = {}
    if "qwen" in model:
        extra["think"] = False
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": 8192, "num_predict": MAX_TOKENS, "temperature": 0},
        "stream": False,
        **extra,
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


def left_truncate(prompt: str, full_tokens: int, target_tokens: int) -> str:
    if target_tokens >= full_tokens:
        return prompt
    chars_to_keep = round(len(prompt) * target_tokens / full_tokens)
    chars_to_keep = max(1, min(chars_to_keep, len(prompt)))
    return prompt[len(prompt) - chars_to_keep:]


def artifact_survival(artifact: str, chars_dropped: int):
    a_chars = len(artifact)
    surviving = max(0, a_chars - chars_dropped)
    fraction = surviving / a_chars if a_chars > 0 else 0.0
    return surviving, round(fraction, 6)


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


def parse_available(output: str | None) -> str | None:
    if output is None:
        return None
    m = _AVAIL_RE.search(output)
    return m.group(1).lower() if m else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip cells already in OUTFILE")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("scorers", PROBES_DIR / "scorers.py")
    scorers_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorers_mod)

    art_probes = [
        json.loads(l)
        for l in (PROBES_DIR / "artifact.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    segments = {
        s["id"]: s
        for s in [
            json.loads(l)
            for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        if s["id"].startswith("art_")
    }

    probe_ids = [p["id"] for p in art_probes]
    # artifact_form lookup
    art_form = {p["id"]: p["artifact_form"] for p in art_probes}

    print("Building F-NUM filler...")
    filler = ctx_mod.build_filler(FILLER_TARGET, seed=42, count_fn=_count_fn, variant="F-NUM")
    filler_tokens = _count_fn(filler)
    print(f"Filler: {filler_tokens} tokens ({len(filler)} chars)")
    print()

    print("Measuring full EARLY prompt sizes...")
    full_tokens_by_probe: dict[str, int] = {}
    artifact_tokens_by_probe: dict[str, int] = {}
    for pid in probe_ids:
        seg = segments[pid]
        early = f"{seg['artifact']}\n\n{filler}\n\n{seg['question']}"
        ft = _count_fn(early)
        at = _count_fn(seg["artifact"])
        full_tokens_by_probe[pid] = ft
        artifact_tokens_by_probe[pid] = at
        extinction = 1.0 - len(seg["artifact"]) / len(early)
        print(f"  {pid} ({art_form[pid]:<12}): full={ft} tok  artifact={at} tok  "
              f"extinction≈{extinction:.3f}")
    print()

    arms = [("arm1_baseline", ARM1_SUFFIX), ("arm3_self_report", ARM3_SUFFIX)]
    total_calls = len(probe_ids) * len(BUDGET_RATIOS) * len(MODELS) * len(arms) * N_REPS
    _CELL_KEYS = ["probe_id", "model", "arm", "budget_ratio", "rep"]
    completed = _load_completed_jsonl(OUTFILE, _CELL_KEYS) if args.resume else set()
    print(f"Total calls: {total_calls}  (completed: {len(completed)})")
    print()

    rows: list[dict] = []
    call_n = 0
    written = 0

    with OUTFILE.open("a", encoding="utf-8") as out_fh:
        for ratio in BUDGET_RATIOS:
            print(f"=== ratio {ratio:.2f} ===")
            for model in MODELS:
                for arm_name, arm_suffix in arms:
                    for pid in probe_ids:
                        seg = segments[pid]
                        full_tokens = full_tokens_by_probe[pid]
                        artifact_tokens_total = artifact_tokens_by_probe[pid]
                        target_tokens = round(full_tokens * ratio)
                        truncating = target_tokens < full_tokens

                        base_prompt = f"{seg['artifact']}\n\n{filler}\n\n{seg['question']}"
                        truncated_base = left_truncate(base_prompt, full_tokens, target_tokens)
                        full_prompt = truncated_base + arm_suffix

                        chars_dropped = len(base_prompt) - len(truncated_base)
                        a_chars_surviving, a_fraction = artifact_survival(seg["artifact"], chars_dropped)
                        a_tokens_surviving = round(artifact_tokens_total * a_fraction)

                        probe_dict = {
                            "id": pid,
                            "scorer_type": seg["scorer_type"],
                            "expected": seg["expected"],
                        }

                        for rep in range(N_REPS):
                            call_n += 1
                            cell = (pid, model, arm_name, ratio, rep)
                            if cell in completed:
                                print(f"[{call_n:3d}/{total_calls}] SKIP {pid} {arm_name[:5]} {model[:14]} r{rep}")
                                continue
                            output, latency, eval_count, prompt_eval_count, done_reason = _call(full_prompt, model)

                            score, score_detail = (
                                scorers_mod.score(probe_dict, output) if output is not None
                                else (None, "no output")
                            )
                            outcome = scorers_mod.outcome_class(output, score, truncated=truncating)
                            available_field = parse_available(output) if arm_name == "arm3_self_report" else None

                            row: dict = {
                                "probe_id": pid,
                                "artifact_form": art_form[pid],
                                "model": model,
                                "arm": arm_name,
                                "budget_ratio": ratio,
                                "target_tokens": target_tokens,
                                "full_tokens": full_tokens,
                                "truncating": truncating,
                                "chars_dropped": chars_dropped,
                                "artifact_tokens_total": artifact_tokens_total,
                                "artifact_tokens_surviving": a_tokens_surviving,
                                "artifact_fraction": a_fraction,
                                "rep": rep,
                                "output": output,
                                "score": score,
                                "score_detail": score_detail,
                                "outcome": outcome,
                                "available_field": available_field,
                                "eval_count": eval_count,
                                "prompt_eval_count": prompt_eval_count,
                                "done_reason": done_reason,
                                "latency_s": round(latency, 3),
                                "hardware": "blade14_rtx4070",
                            }
                            rows.append(row)
                            out_fh.write(json.dumps(row) + "\n")
                            out_fh.flush()
                            written += 1

                            status = "✓" if score == 1.0 else ("A" if outcome == "abstained" else "✗")
                            avail_tag = f" AVAIL:{available_field}" if available_field else ""
                            print(
                                f"[{call_n:3d}/{total_calls}] {pid} {arm_name[:5]} "
                                f"{model[:14]} r{rep}"
                                f" af={a_fraction:.2f} {status}{avail_tag}"
                                f" {latency*1000:.0f}ms  {repr((output or '')[:45])}"
                            )
                            if call_n % 10 == 0:
                                print(f"--- progress: {call_n}/{total_calls} calls, {written} written ---")

    out = RESULTS / "art_truncation.json"
    out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWritten {len(rows)} rows → {out} (legacy JSON)")
    print(f"Incremental JSONL: {written} rows → {OUTFILE}")

    # Quick summary
    _print_summary(rows, probe_ids, art_form, BUDGET_RATIOS, MODELS, arms)


def _print_summary(rows, probe_ids, art_form, ratios, models, arms):
    from collections import defaultdict

    print("\n=== MEAN SCORE by form × model × arm × ratio (arm1 baseline) ===")
    # arm1 only, truncating rows only
    forms = ["structured", "narrative", "mixed"]
    for form in forms:
        form_pids = [p for p in probe_ids if art_form[p] == form]
        for model in models:
            for arm_name, _ in arms:
                cell = defaultdict(list)
                for r in rows:
                    if (r["artifact_form"] == form and r["model"] == model
                            and r["arm"] == arm_name and r["score"] is not None):
                        cell[r["budget_ratio"]].append(r["score"])
                scores_str = " | ".join(
                    f"r={rt:.2f}:{sum(v)/len(v):.2f}" for rt in ratios
                    if cell[rt]
                )
                print(f"  {form:<12} {model[:14]} {arm_name[:5]}: {scores_str}")

    print("\n=== FABRICATION / ABSTENTION among truncated rows (arm1) ===")
    for form in forms:
        trunc_rows = [r for r in rows if r["artifact_form"] == form
                      and r["arm"] == "arm1_baseline" and r["truncating"]
                      and r["budget_ratio"] < 1.0]
        n = len(trunc_rows)
        fab = sum(1 for r in trunc_rows if r["outcome"] == "incorrect")
        abs_ = sum(1 for r in trunc_rows if r["outcome"] == "abstained")
        correct = sum(1 for r in trunc_rows if r["outcome"] == "correct")
        if n:
            print(f"  {form:<12}: n={n}  fab={fab}({fab/n:.0%})  abs={abs_}({abs_/n:.0%})  "
                  f"correct={correct}({correct/n:.0%})")

    print("\n=== FABRICATED VALUE SOURCE (arm1 truncated, incorrect rows) ===")
    filler_tail = "93850"
    for form in forms:
        incorr = [r for r in rows if r["artifact_form"] == form
                  and r["arm"] == "arm1_baseline" and r["truncating"]
                  and r["outcome"] == "incorrect" and r["output"]]
        from_filler = sum(1 for r in incorr if filler_tail in (r["output"] or ""))
        from_prior = len(incorr) - from_filler
        print(f"  {form:<12}: {len(incorr)} fabrications  "
              f"from_filler={from_filler}  from_prior={from_prior}")

    print("\n=== ARM3 AVAILABLE:yes RATE by form (truncated rows) ===")
    for form in forms:
        arm3_rows = [r for r in rows if r["artifact_form"] == form
                     and r["arm"] == "arm3_self_report" and r["truncating"]
                     and r["budget_ratio"] < 1.0]
        yes_rows = [r for r in arm3_rows if r.get("available_field") == "yes"]
        n = len(arm3_rows)
        if n:
            print(f"  {form:<12}: AVAILABLE:yes {len(yes_rows)}/{n} ({len(yes_rows)/n:.0%})")


if __name__ == "__main__":
    main()
