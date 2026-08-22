"""Stage 2 — fine-grained partial truncation sweep.

qwen3:4b-instruct, EARLY arm only, 9 budget ratios (1.00 → 0.85),
5 reps, with artifact survival instrumentation.

5 probes × 9 ratios × 5 reps = 225 calls.

Artifact survival fields recorded per row:
  artifact_tokens_total    — full artifact token count (pre-truncation)
  artifact_tokens_surviving — tokens of the artifact surviving in the
                             truncated prompt (character-proportional estimate)
  artifact_fraction        — surviving / total (0.0 when artifact fully gone)

Note on the estimate: artifact_tokens_surviving is derived from character
counts (artifact_chars_surviving / artifact_chars_total * artifact_tokens_total).
Tokenisation is not strictly linear in characters, so this is an approximation.
The artifact_fraction value is the primary analysis variable.
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
BUDGET_RATIOS = [1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.85]
N_REPS = 5
MAX_TOKENS = 128


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


def _call(prompt):
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
    except Exception:
        return None, time.monotonic() - t0, None, None, None


def left_truncate(prompt: str, full_tokens: int, target_tokens: int) -> str:
    if target_tokens >= full_tokens:
        return prompt
    chars_to_keep = round(len(prompt) * target_tokens / full_tokens)
    chars_to_keep = max(1, min(chars_to_keep, len(prompt)))
    return prompt[len(prompt) - chars_to_keep:]


def artifact_survival(artifact: str, chars_dropped: int) -> tuple[int, float]:
    """Return (artifact_chars_surviving, artifact_fraction)."""
    a_chars = len(artifact)
    surviving = max(0, a_chars - chars_dropped)
    fraction = surviving / a_chars if a_chars > 0 else 0.0
    return surviving, round(fraction, 6)


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

    print("Stage 2 — Fine-grained partial truncation sweep")
    print(f"Model : {MODEL}")
    print(f"Probes: {TARGET_PROBES}")
    print(f"Ratios: {BUDGET_RATIOS}")
    print(f"N_reps: {N_REPS}")
    print()

    print("Building F-NUM filler...")
    filler = ctx_mod.build_filler(FILLER_TARGET, seed=42, count_fn=_count_fn, variant="F-NUM")
    filler_tokens = _count_fn(filler)
    print(f"Filler: {filler_tokens} tokens ({len(filler)} chars)")
    print()

    print("Measuring full EARLY prompt token counts and artifact sizes...")
    full_tokens_by_probe: dict[str, int] = {}
    artifact_tokens_by_probe: dict[str, int] = {}
    for pid in TARGET_PROBES:
        seg = segments[pid]
        early = f"{seg['artifact']}\n\n{filler}\n\n{seg['question']}"
        ft = _count_fn(early)
        at = _count_fn(seg["artifact"])
        full_tokens_by_probe[pid] = ft
        artifact_tokens_by_probe[pid] = at
        # Extinction ratio: artifact fully gone when chars_dropped >= len(artifact)
        # chars_dropped = len(full_prompt) * (1 - ratio)
        # Extinction at: ratio = 1 - len(artifact) / len(full_prompt)
        full_prompt = f"{seg['artifact']}\n\n{filler}\n\n{seg['question']}"
        extinction_ratio = 1.0 - len(seg["artifact"]) / len(full_prompt)
        print(f"  {pid}: full={ft} tok  artifact={at} tok ({len(seg['artifact'])} chars)"
              f"  extinction_ratio≈{extinction_ratio:.3f}")
    print()

    total_calls = len(TARGET_PROBES) * len(BUDGET_RATIOS) * N_REPS
    print(f"Total calls: {total_calls}")
    print()

    rows: list[dict] = []
    call_n = 0

    for ratio in BUDGET_RATIOS:
        print(f"--- ratio {ratio:.2f} ---")
        for pid in TARGET_PROBES:
            seg = segments[pid]
            full_tokens = full_tokens_by_probe[pid]
            artifact_tokens_total = artifact_tokens_by_probe[pid]
            target_tokens = round(full_tokens * ratio)
            truncating = target_tokens < full_tokens

            full_prompt = f"{seg['artifact']}\n\n{filler}\n\n{seg['question']}"
            truncated = left_truncate(full_prompt, full_tokens, target_tokens)
            chars_dropped = len(full_prompt) - len(truncated)

            # Artifact survival
            a_chars_surviving, a_fraction = artifact_survival(seg["artifact"], chars_dropped)
            a_tokens_surviving = round(artifact_tokens_total * a_fraction)

            for rep in range(N_REPS):
                call_n += 1
                output, latency, eval_count, prompt_eval_count, done_reason = _call(truncated)

                probe_dict = {
                    "id": pid,
                    "scorer_type": seg["scorer_type"],
                    "expected": seg["expected"],
                }
                score, score_detail = scorers_mod.score(probe_dict, output)
                outcome = scorers_mod.outcome_class(output, score, truncated=truncating)

                row: dict = {
                    "probe_id": pid,
                    "model": MODEL,
                    "model_variant": "instruct",
                    "arm": "EARLY",
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
                    "eval_count": eval_count,
                    "prompt_eval_count": prompt_eval_count,
                    "done_reason": done_reason,
                    "latency_s": round(latency, 3),
                    "hardware": "blade14_rtx4070",
                }
                rows.append(row)

                status = "✓" if score == 1.0 else ("A" if outcome == "abstained" else "✗")
                print(
                    f"[{call_n:3d}/{total_calls}] {pid} r={ratio:.2f} rep={rep}"
                    f" af={a_fraction:.2f} {status}"
                    f" {latency*1000:.0f}ms  {repr((output or '')[:55])}"
                )

    out = RESULTS / "partial_truncation.json"
    out.write_text(json.dumps({"model": MODEL, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWritten {len(rows)} rows to {out}")

    # Quick summary
    from collections import defaultdict
    print("\n=== Score by probe × ratio ===")
    cell: dict = defaultdict(list)
    for r in rows:
        if r.get("score") is not None:
            cell[(r["probe_id"], r["budget_ratio"])].append(r["score"])
    for pid in TARGET_PROBES:
        vals = [f"r={rt:.2f}:{sum(cell[(pid,rt)])/len(cell[(pid,rt)]):.2f}"
                for rt in BUDGET_RATIOS if cell[(pid, rt)]]
        print(f"  {pid}: {' | '.join(vals)}")

    print("\n=== artifact_fraction by probe × ratio ===")
    af_cell: dict = defaultdict(list)
    for r in rows:
        af_cell[(r["probe_id"], r["budget_ratio"])].append(r["artifact_fraction"])
    for pid in TARGET_PROBES:
        vals = [f"r={rt:.2f}:af={af_cell[(pid,rt)][0]:.3f}"
                for rt in BUDGET_RATIOS if af_cell[(pid, rt)]]
        print(f"  {pid}: {' | '.join(vals)}")


if __name__ == "__main__":
    main()
