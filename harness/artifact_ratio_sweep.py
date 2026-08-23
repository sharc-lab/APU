"""Artifact-to-prompt ratio sweep.

Question: Is the apparent step-function collapse in art_* partly a design
artifact of the small artifact/filler ratio? With 4,000 tokens of filler the
artifact is <4% of the prompt, so the partial regime (where artifact partially
survives) is only ~2 percentage points of budget_ratio wide. Wider partial
regimes (from smaller filler) would produce more data points inside the
transition and might reveal graded degradation rather than a step.

Design:
  Probes:       art_01 (structured, 272-char artifact)
                art_07 (narrative,  692-char artifact)
                art_08 (mixed,      736-char artifact)
  Filler sizes: 500, 2000, 4000, 8000 tokens
  Ratios:       7 per probe×filler_size, centred on that config's extinction
                point (analytically computed from char counts)
  Model:        qwen3:4b-instruct only
  Reps:         3
  Total:        3 x 4 x 7 x 3 = 252 calls

Key metric: partial regime width = 1 - extinction_ratio (in budget_ratio units).
            Does non-zero artifact_fraction predict non-zero score?

Results written to results/artifact_ratio_sweep.json.
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

import context as ctx_mod

TARGET_PROBES = ["art_01", "art_07", "art_08"]
FILLER_SIZES = [500, 2000, 4000, 8000]
MODEL = "qwen3:4b-instruct"
HOST = "http://localhost:11434"
N_REPS = 3
MAX_TOKENS = 256
OUTFILE = RESULTS / "artifact_ratio_sweep.jsonl"

# Number of ratios to sample on each side of the extinction point
RATIOS_ABOVE = [0.03, 0.01]     # extinction + delta (partial regime)
RATIOS_BELOW = [0.01, 0.04]     # extinction - delta (fully extinct)
FIXED_RATIOS = [1.20, 0.40]     # always include untruncated baseline and deep cut


def _count_fn(text: str) -> int:
    payload = json.dumps({
        "model": MODEL,
        "prompt": text,
        "options": {"num_ctx": 16384, "num_predict": 1, "temperature": 0},
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
        "options": {"num_ctx": 16384, "num_predict": MAX_TOKENS, "temperature": 0},
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
        with urllib.request.urlopen(req, timeout=300) as r:
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


def artifact_survival(artifact: str, chars_dropped: int) -> float:
    a = len(artifact)
    surviving = max(0, a - chars_dropped)
    return round(surviving / a if a else 0.0, 6)


def compute_extinction_ratio(artifact_chars: int, full_prompt_chars: int) -> float:
    """Budget ratio below which the artifact is fully removed.

    Left-truncation drops chars from the left. The artifact (also at the left)
    is fully gone when chars_dropped >= artifact_chars:
      chars_dropped = full_prompt_chars * (1 - ratio)
      1 - ratio >= artifact_chars / full_prompt_chars
      ratio <= 1 - artifact_chars / full_prompt_chars
    So extinction_ratio = 1 - artifact_chars / full_prompt_chars.
    """
    return 1.0 - artifact_chars / full_prompt_chars


def build_ratio_grid(extinction: float) -> list[float]:
    """7 ratios centred on extinction point."""
    ratios = set(FIXED_RATIOS)
    for delta in RATIOS_ABOVE:
        r = round(extinction + delta, 4)
        if 0.01 < r < 1.50:
            ratios.add(r)
    for delta in RATIOS_BELOW:
        r = round(extinction - delta, 4)
        if 0.01 < r < 1.50:
            ratios.add(r)
    # always include extinction itself (or nearest representable float)
    ratios.add(round(extinction, 4))
    return sorted(ratios, reverse=True)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip cells already in OUTFILE")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("scorers", PROBES_DIR / "scorers.py")
    scorers_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorers_mod)

    segments = {
        s["id"]: s
        for s in [json.loads(l) for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if s["id"] in TARGET_PROBES
    }

    # Build fillers at each size
    fillers: dict[int, str] = {}
    print("Building fillers...")
    for size in FILLER_SIZES:
        f = ctx_mod.build_filler(size, seed=42, count_fn=_count_fn, variant="F-NUM")
        tok = _count_fn(f)
        print(f"  filler={size}: {tok} tok ({len(f)} chars)")
        fillers[size] = f

    # Per probe × filler_size: compute full_tokens, extinction, ratio grid
    print("\nComputing extinction points...")
    configs: list[dict] = []
    for pid in TARGET_PROBES:
        seg = segments[pid]
        artifact_chars = len(seg["artifact"])
        for fsize in FILLER_SIZES:
            filler = fillers[fsize]
            full_prompt = f"{seg['artifact']}\n\n{filler}\n\n{seg['question']}"
            full_chars = len(full_prompt)
            full_tokens = _count_fn(full_prompt)
            extinction = compute_extinction_ratio(artifact_chars, full_chars)
            partial_width = round(1.0 - extinction, 4)
            ratios = build_ratio_grid(extinction)
            print(f"  {pid} filler={fsize:5d}: full={full_tokens} tok  "
                  f"extinction={extinction:.4f}  partial_width={partial_width:.4f}  "
                  f"ratios={[round(r,3) for r in ratios]}")
            configs.append({
                "probe_id": pid,
                "filler_size": fsize,
                "full_tokens": full_tokens,
                "full_chars": full_chars,
                "artifact_chars": artifact_chars,
                "extinction_ratio": round(extinction, 6),
                "partial_regime_width": partial_width,
                "ratios": ratios,
            })

    total_calls = sum(len(c["ratios"]) for c in configs) * N_REPS
    _CELL_KEYS = ["probe_id", "filler_size", "budget_ratio", "rep"]
    completed = _load_completed_jsonl(OUTFILE, _CELL_KEYS) if args.resume else set()
    print(f"\nTotal calls: {total_calls}  (completed: {len(completed)})")

    rows = []
    call_n = 0
    written = 0

    with OUTFILE.open("a", encoding="utf-8") as out_fh:
        for cfg in configs:
            pid = cfg["probe_id"]
            fsize = cfg["filler_size"]
            seg = segments[pid]
            filler = fillers[fsize]
            full_tokens = cfg["full_tokens"]
            extinction = cfg["extinction_ratio"]

            for ratio in cfg["ratios"]:
                target_tokens = round(full_tokens * ratio)
                truncating = target_tokens < full_tokens
                base_prompt = f"{seg['artifact']}\n\n{filler}\n\n{seg['question']}"
                truncated = left_truncate(base_prompt, full_tokens, target_tokens)
                chars_dropped = len(base_prompt) - len(truncated)
                a_fraction = artifact_survival(seg["artifact"], chars_dropped)

                probe_dict = {
                    "id": pid,
                    "scorer_type": seg["scorer_type"],
                    "expected": seg["expected"],
                }

                for rep in range(N_REPS):
                    call_n += 1
                    cell = (pid, fsize, round(ratio, 4), rep)
                    if cell in completed:
                        print(f"[{call_n:3d}/{total_calls}] SKIP {pid} f={fsize} r={ratio:.4f} rep={rep}")
                        continue
                    output, latency, eval_count, prompt_eval_count, done_reason = _call(truncated)
                    score, score_detail = (
                        scorers_mod.score(probe_dict, output) if output is not None
                        else (None, "no output")
                    )
                    outcome = scorers_mod.outcome_class(output, score, truncated=truncating)

                    row = {
                        "probe_id": pid,
                        "filler_size": fsize,
                        "budget_ratio": round(ratio, 4),
                        "extinction_ratio": extinction,
                        "partial_regime_width": cfg["partial_regime_width"],
                        "full_tokens": full_tokens,
                        "target_tokens": target_tokens,
                        "truncating": truncating,
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
                    out_fh.write(json.dumps(row) + "\n")
                    out_fh.flush()
                    written += 1

                    tag = "✓" if score == 1.0 else ("A" if outcome == "abstained" else "✗")
                    zone = "partial" if 0 < a_fraction < 1 else ("full" if a_fraction == 1.0 else "extinct")
                    print(
                        f"[{call_n:3d}/{total_calls}] {pid} f={fsize:5d} "
                        f"r={ratio:.3f} ({zone:<7}) af={a_fraction:.2f} {tag} "
                        f"{latency*1000:.0f}ms  {repr((output or '')[:35])}"
                    )
                    if call_n % 10 == 0:
                        print(f"--- progress: {call_n}/{total_calls} calls, {written} written ---")

    out = RESULTS / "artifact_ratio_sweep.json"
    out.write_text(json.dumps({"rows": rows, "configs": configs}, indent=2), encoding="utf-8")
    print(f"\nWritten {len(rows)} rows → {out} (legacy JSON)")
    print(f"Incremental JSONL: {written} rows → {OUTFILE}")
    _print_summary(rows, configs)


def _print_summary(rows, configs):
    print("\n=== PARTIAL REGIME WIDTH by probe × filler_size ===")
    print(f"  {'probe':<8} {'filler':>6}  width   extinction")
    for cfg in configs:
        print(f"  {cfg['probe_id']:<8} {cfg['filler_size']:>6}  "
              f"{cfg['partial_regime_width']:.4f}  {cfg['extinction_ratio']:.4f}")

    print("\n=== SCORE IN PARTIAL REGIME vs EXTINCT (per probe × filler_size) ===")
    for cfg in configs:
        pid, fsize = cfg["probe_id"], cfg["filler_size"]
        partial = [r for r in rows if r["probe_id"] == pid and r["filler_size"] == fsize
                   and 0 < r["artifact_fraction"] < 1]
        extinct = [r for r in rows if r["probe_id"] == pid and r["filler_size"] == fsize
                   and r["artifact_fraction"] == 0.0 and r["truncating"]]
        def mean_s(lst):
            s = [r["score"] for r in lst if r["score"] is not None]
            return f"{sum(s)/len(s):.2f}" if s else "n/a"
        def outcomes(lst):
            return (
                f"fab={sum(1 for r in lst if r['outcome']=='incorrect')}"
                f" abs={sum(1 for r in lst if r['outcome']=='abstained')}"
                f" ok={sum(1 for r in lst if r['outcome']=='correct')}"
            )
        print(f"  {pid} f={fsize:5d}:  partial n={len(partial)} score={mean_s(partial)} {outcomes(partial)} | "
              f"extinct n={len(extinct)} score={mean_s(extinct)} {outcomes(extinct)}")


if __name__ == "__main__":
    main()
