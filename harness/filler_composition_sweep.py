"""Filler composition control sweep.

Three filler variants (F-NUM, F-PROSE, F-STRUCT-NONNUM) × 5 probes ×
2 budget ratios × 2 arms (baseline, self-report) × 3 reps = 180 calls.

EARLY arm only. Truncation identical to Stage C (harness-side left-char).
Records filler_variant, full raw output, and verbatim-in-filler flag on
every row.
"""

from __future__ import annotations

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

import context as ctx_mod

MODEL = "qwen3:4b-instruct"
HOST = "http://localhost:11434"
FILLER_TARGET = 4000
TARGET_PROBES = ["rag_01", "rag_02", "rag_05", "sea_01", "sea_04"]
VARIANTS = ["F-NUM", "F-PROSE", "F-STRUCT-NONNUM"]
BUDGET_RATIOS = [0.70, 0.40]
N_REPS = 3
MAX_TOKENS = 128

ARM3_SUFFIX = (
    "\nFirst state whether the information needed to answer is present above,"
    " then answer. Format: AVAILABLE: yes|no, then the answer."
)


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


def _call(prompt: str) -> tuple[str | None, float]:
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
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return data["message"]["content"].strip(), time.monotonic() - t0
    except Exception as e:
        return None, time.monotonic() - t0


def left_truncate(prompt: str, full_tokens: int, target_tokens: int) -> str:
    if target_tokens >= full_tokens:
        return prompt
    chars_to_keep = round(len(prompt) * target_tokens / full_tokens)
    chars_to_keep = max(1, min(chars_to_keep, len(prompt)))
    return prompt[len(prompt) - chars_to_keep:]


def classify_available(output: str) -> str:
    t = output.lower()
    if "available: no" in t or "available:no" in t:
        return "no"
    if "available: yes" in t or "available:yes" in t:
        return "yes"
    return "absent"


def source_of_fabrication(output: str | None, truncated_prompt: str) -> str:
    """Classify where a fabricated answer came from.

    Returns one of:
      verbatim_in_context  — normalized output appears literally in the
                             truncated prompt (case-insensitive)
      partial_in_context   — a token of >=4 chars from the output appears in
                             the truncated prompt (weaker signal)
      prior_only           — no match found; model drew from weights
    """
    if output is None:
        return "error"
    normed = output.strip()
    if not normed:
        return "error"
    tp_lower = truncated_prompt.lower()
    if normed.lower() in tp_lower:
        return "verbatim_in_context"
    # Check individual tokens of >=4 chars
    tokens = [t for t in normed.split() if len(t) >= 4]
    if tokens and any(t.lower() in tp_lower for t in tokens):
        return "partial_in_context"
    return "prior_only"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location(
        "scorers", PROBES_DIR / "scorers.py"
    )
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

    print("Filler Composition Sweep")
    print(f"Model  : {MODEL}")
    print(f"Probes : {TARGET_PROBES}")
    print(f"Variants: {VARIANTS}")
    print(f"Ratios : {BUDGET_RATIOS}")
    print()

    # Build fillers and verify
    fillers: dict[str, str] = {}
    filler_tokens: dict[str, int] = {}
    filler_samples: dict[str, str] = {}

    for variant in VARIANTS:
        print(f"Building {variant}...")
        f = ctx_mod.build_filler(FILLER_TARGET, seed=42, count_fn=_count_fn, variant=variant)
        tok = _count_fn(f)
        fillers[variant] = f
        filler_tokens[variant] = tok
        filler_samples[variant] = f[:300]
        print(f"  {variant}: {tok} tokens, {len(f)} chars")
        print(f"  Sample: {f[:120]!r}")
        print()

    # Precompute full EARLY prompts × variant
    # full_tokens is computed once per probe (arm1, same filler F-NUM for anchor)
    print("Measuring full EARLY prompt token counts (F-NUM anchor)...")
    full_tokens_by_probe: dict[str, int] = {}
    for pid in TARGET_PROBES:
        seg = segments[pid]
        early = f"{seg['artifact']}\n\n{fillers['F-NUM']}\n\n{seg['question']}"
        full_tokens_by_probe[pid] = _count_fn(early)
        print(f"  {pid}: {full_tokens_by_probe[pid]} tokens")
    print()

    total_calls = len(TARGET_PROBES) * len(VARIANTS) * len(BUDGET_RATIOS) * 2 * N_REPS
    print(f"Total calls: {total_calls}")
    print()

    rows: list[dict] = []
    call_n = 0

    for variant in VARIANTS:
        filler = fillers[variant]
        for pid in TARGET_PROBES:
            seg = segments[pid]
            full_tokens = full_tokens_by_probe[pid]

            for ratio in BUDGET_RATIOS:
                target_tokens = round(full_tokens * ratio)

                for arm_name, question in [
                    ("arm1_baseline", seg["question"]),
                    ("arm3_self_report", seg["question"] + ARM3_SUFFIX),
                ]:
                    full_prompt = f"{seg['artifact']}\n\n{filler}\n\n{question}"
                    truncated = left_truncate(full_prompt, full_tokens, target_tokens)
                    chars_dropped = len(full_prompt) - len(truncated)

                    for rep in range(N_REPS):
                        call_n += 1
                        output, latency = _call(truncated)

                        probe_dict = {
                            "id": pid,
                            "scorer_type": seg["scorer_type"],
                            "expected": seg["expected"],
                        }
                        score, score_detail = scorers_mod.score(probe_dict, output)
                        outcome = scorers_mod.outcome_class(output, score, truncated=True)

                        fab_source = source_of_fabrication(output, truncated)

                        row: dict = {
                            "probe_id": pid,
                            "filler_variant": variant,
                            "arm": arm_name,
                            "budget_ratio": ratio,
                            "target_tokens": target_tokens,
                            "full_tokens": full_tokens,
                            "chars_dropped": chars_dropped,
                            "rep": rep,
                            "output": output,
                            "score": score,
                            "score_detail": score_detail,
                            "outcome": outcome,
                            "fabrication_source": fab_source,
                            "latency_s": round(latency, 3),
                            "model": MODEL,
                            "hardware": "blade_rtx4070",
                        }

                        if arm_name == "arm3_self_report":
                            row["available_field"] = classify_available(output or "")

                        rows.append(row)

                        fab_flag = fab_source[0].upper() if outcome == "incorrect" else " "
                        avail = row.get("available_field", "")
                        status = "✓" if score == 1.0 else ("A" if outcome == "abstained" else "✗")
                        print(
                            f"[{call_n:3d}/{total_calls}] {variant:16s} {pid} "
                            f"r={ratio:.2f} {arm_name[:5]} rep={rep} "
                            f"{status} src={fab_flag} av={avail:3s} "
                            f"{latency*1000:.0f}ms  {repr((output or '')[:50])}"
                        )

    out = Path("results/filler_composition.json")
    out.write_text(
        json.dumps(
            {
                "experiment": "filler_composition",
                "model": MODEL,
                "hardware": "blade_rtx4070",
                "probes": TARGET_PROBES,
                "variants": VARIANTS,
                "ratios": BUDGET_RATIOS,
                "n_reps": N_REPS,
                "filler_tokens": filler_tokens,
                "filler_samples": filler_samples,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten {len(rows)} rows to {out}")

    # Quick summary
    from collections import defaultdict
    cell: dict = defaultdict(list)
    for r in rows:
        if r["outcome"] in ("incorrect", "abstained", "correct"):
            cell[(r["filler_variant"], r["arm"], r["budget_ratio"])].append(r)

    print("\n=== Fabrication rate by variant × arm × ratio ===")
    for variant in VARIANTS:
        for arm in ["arm1_baseline", "arm3_self_report"]:
            for ratio in BUDGET_RATIOS:
                rws = cell[(variant, arm, ratio)]
                n = len(rws)
                fab = sum(1 for r in rws if r["outcome"] == "incorrect")
                ab = sum(1 for r in rws if r["outcome"] == "abstained")
                print(f"  {variant:16s} {arm[:5]} r={ratio:.2f}: "
                      f"fab={fab}/{n} ({fab/n:.0%})  abs={ab}/{n}")

    print("\n=== Fabrication source ===")
    for variant in VARIANTS:
        rws = [r for r in rows if r["filler_variant"] == variant and r["outcome"] == "incorrect"]
        from collections import Counter
        src = Counter(r["fabrication_source"] for r in rws)
        print(f"  {variant}: {dict(src)}")

    print("\n=== Arm3 AVAILABLE:yes by variant ===")
    for variant in VARIANTS:
        rws = [r for r in rows if r["filler_variant"] == variant and r.get("available_field") is not None]
        yes = sum(1 for r in rws if r["available_field"] == "yes")
        total = len(rws)
        print(f"  {variant}: {yes}/{total} ({yes/total:.0%}) AVAILABLE:yes")


if __name__ == "__main__":
    main()
