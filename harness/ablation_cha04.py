"""Ablation runner for cha_04 substitution hypothesis.

Runs cha_04 (control) + cha_04_ablate + cha_04_swap at depths [0, 8000, 32000],
5 reps each (45 calls total). Prints a table of raw answer strings with counts.
Does NOT write to results/; outputs go to stdout only.

Usage:
    py -3.11 harness/ablation_cha04.py
    py -3.11 harness/ablation_cha04.py --host http://localhost:11434
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from harness import context
from harness.context import DEFAULT_FILLER_MODE

MODEL = "qwen3:4b-instruct"
HOST  = "http://localhost:11434"
DEPTHS = [0, 8_000, 32_000]
REPS   = 5

CONTROL_ID = "cha_04"
ABLATION_FILE = REPO / "evaluation" / "probes" / "ablation.jsonl"
PROMPTS_FILE  = REPO / "evaluation" / "probes" / "prompts.jsonl"


def load_probes() -> list[dict]:
    probes = []
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["id"] == CONTROL_ID:
                probes.append(p)
    with open(ABLATION_FILE, encoding="utf-8") as f:
        for line in f:
            probes.append(json.loads(line))
    return probes  # order: cha_04, cha_04_ablate, cha_04_swap


def count_fn(text: str, host: str) -> int:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "stream": True,
        "options": {"num_predict": 1, "temperature": 0},
    }
    with httpx.stream("POST", f"{host}/api/chat", json=payload, timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            raw = raw.strip()
            if not raw:
                continue
            c = json.loads(raw)
            if c.get("done"):
                return max(0, c.get("prompt_eval_count", 0) - 8)
    return 0


def call_model(prompt: str, max_tokens: int, host: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }
    text = ""
    with httpx.stream("POST", f"{host}/api/chat", json=payload, timeout=600) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            raw = raw.strip()
            if not raw:
                continue
            c = json.loads(raw)
            text += c.get("message", {}).get("content", "")
            if c.get("done"):
                break
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    args = parser.parse_args()

    probes = load_probes()
    probe_ids = [p["id"] for p in probes]
    print(f"Probes : {probe_ids}")
    print(f"Depths : {DEPTHS}")
    print(f"Reps   : {REPS}")
    print(f"Total  : {len(probes) * len(DEPTHS) * REPS} calls")
    print()

    # Build filler cache (calibrated, same as main runner)
    print("Calibrating filler...")
    filler_cache: dict[tuple[int, int], str] = {}
    for d in DEPTHS:
        for r in range(REPS):
            if d == 0:
                filler_cache[(d, r)] = context.build_filler(0, seed=r, count_fn=None)
            else:
                fn = lambda text, _d=d, _r=r: count_fn(text, args.host)
                filler_cache[(d, r)] = context.build_filler(d, seed=r, count_fn=fn)
                print(f"  d={d} r={r}: {len(filler_cache[(d,r)])} chars", flush=True)
    print("Filler ready.\n")

    # raw_answers[probe_id][depth] -> list of answer strings
    raw_answers: dict[str, dict[int, list[str]]] = {
        p["id"]: {d: [] for d in DEPTHS} for p in probes
    }

    total = len(probes) * len(DEPTHS) * REPS
    done = 0
    w = len(str(total))

    for d in DEPTHS:
        for r in range(REPS):
            filler = filler_cache[(d, r)]
            for probe in probes:
                prompt = context.wrap_prompt(filler, probe["prompt"], filler_mode=DEFAULT_FILLER_MODE)
                t0 = time.perf_counter()
                answer = call_model(prompt, probe["max_tokens"], args.host)
                elapsed = time.perf_counter() - t0
                raw_answers[probe["id"]][d].append(answer)
                done += 1
                correct = answer == probe["expected"]
                print(
                    f"[{done:{w}}/{total}] {probe['id']} d={d:>5} r={r}"
                    f"  got={answer!r}  want={probe['expected']!r}"
                    f"  {'OK' if correct else 'FAIL'}  {elapsed:.1f}s",
                    flush=True,
                )

    # ── RESULTS TABLE ──────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("RAW ANSWER TABLE  (counts across 5 reps)")
    print("=" * 72)
    print(f"{'Probe':<18} {'Depth':>6}  {'Answer':>8}  {'Count':>5}  {'Score'}")
    print("-" * 72)

    for probe in probes:
        pid = probe["id"]
        want = probe["expected"]
        for d in DEPTHS:
            answers = raw_answers[pid][d]
            tally: dict[str, int] = defaultdict(int)
            for a in answers:
                tally[a] += 1
            for ans in sorted(tally, key=lambda x: -tally[x]):
                correct = "✓" if ans == want else f"✗ (want {want})"
                print(f"  {pid:<16} {d:>6}  {ans:>8}  {tally[ans]:>5}  {correct}")
        print()

    # ── INTERPRETATION ─────────────────────────────────────────────────
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    swap_answers_depth = {d: raw_answers["cha_04_swap"][d] for d in [8_000, 32_000]}
    ablate_answers_depth = {d: raw_answers["cha_04_ablate"][d] for d in [8_000, 32_000]}

    swap_wrong_set = set()
    for d in [8_000, 32_000]:
        for a in swap_answers_depth[d]:
            if a != "400":
                swap_wrong_set.add(a)

    ablate_correct = all(
        a == "400"
        for d in [8_000, 32_000]
        for a in ablate_answers_depth[d]
    )
    ablate_wrong_set = set()
    for d in [8_000, 32_000]:
        for a in ablate_answers_depth[d]:
            if a != "400":
                ablate_wrong_set.add(a)

    swap_tracks_retries = "1400" in swap_wrong_set

    print()
    if swap_tracks_retries and ablate_correct:
        print("VERDICT: SUBSTITUTION CONFIRMED (decisive)")
        print("  cha_04_swap returns 1400 (7×200) at depth>0 — model tracks retries field.")
        print("  cha_04_ablate returns 400 at depth>0 — removing the distractor restores correct counting.")
    elif swap_tracks_retries:
        print("VERDICT: SUBSTITUTION CONFIRMED (swap arm only)")
        print(f"  cha_04_swap returns 1400 at depth>0.")
        print(f"  cha_04_ablate: wrong answers = {ablate_wrong_set or 'none (correct)'}")
    elif ablate_correct:
        print("VERDICT: SUBSTITUTION CONFIRMED (ablate arm only)")
        print(f"  cha_04_ablate returns 400 at depth>0 — removing retries restores correct counting.")
        print(f"  cha_04_swap wrong answers: {swap_wrong_set} (not 1400 — partial evidence)")
    elif not swap_wrong_set and not ablate_wrong_set:
        print("VERDICT: BOTH VARIANTS CORRECT AT DEPTH — substitution field was the sole cause.")
    else:
        print("VERDICT: MIXED / INCONCLUSIVE")
        print(f"  cha_04_swap wrong answers at depth>0: {swap_wrong_set}")
        print(f"  cha_04_ablate wrong answers at depth>0: {ablate_wrong_set}")
        print("  Does not cleanly confirm or refute substitution hypothesis.")
        print("  Report exact table above; do not force a conclusion.")
    print()


if __name__ == "__main__":
    main()
