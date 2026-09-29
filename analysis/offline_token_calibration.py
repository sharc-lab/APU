"""Offline Q0 token calibration against real Hugging Face tokenizers, run on the controller (no lab machine time).

Checks quality_suite.build_task's actual/target token ratio at 2000/8000/24000/96000 for every TASK_TYPES task, once
per tokenizer family (Qwen3, Llama 3.1/3.3), using the real tokenizer.encode() count rather than a live server's
/tokenize endpoint. This is the same 0.95-1.05 bar q0_token_calibration.py checks on the lab machines, but for the
model families that script has not yet been run against (Llama), and it costs no GPU time.

Llama 3.1/3.3's own tokenizer repos on Hugging Face are gated; this uses ungated mirrors that re-upload the
identical, unmodified tokenizer files (NousResearch/Meta-Llama-3.1-8B-Instruct, unsloth/Llama-3.3-70B-Instruct) --
confirmed by exact vocab_size match (128000) and identical special-token layout.

Usage:
  python analysis/offline_token_calibration.py --out results/offline_token_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import quality_suite as qs  # noqa: E402

TARGET_LENGTHS = (2000, 8000, 24000, 96000)
SEED = 42
OK_LOW, OK_HIGH = 0.95, 1.05

# Tokenizer repo id per model family. Llama uses ungated mirrors of the gated official repos (see module docstring).
TOKENIZERS = {
    "qwen3": "Qwen/Qwen3-8B",
    "llama-3.1": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "llama-3.3": "unsloth/Llama-3.3-70B-Instruct",
}


def load_tokenizers(families=None):
    from transformers import AutoTokenizer
    families = families or TOKENIZERS
    return {name: AutoTokenizer.from_pretrained(repo) for name, repo in families.items()}


def calibrate(tokenizers, task_types=None, target_lengths=TARGET_LENGTHS, seed=SEED, use_count_fn=True):
    """tokenizers: {family_name: a tokenizer object with .encode(text) -> list}. Returns
    {family: {task_type: [{"target", "actual", "ratio", "ok"}, ...]}}.

    use_count_fn=True (default) passes the real tokenizer's own count into build_task as count_fn, so filler
    sizing iteratively refines toward that family's real token count (see quality_suite.build_task's docstring) --
    this is what fixed the systematic ~0.81 ratio Llama 3.1/3.3 showed on 2026-09-29 when count_fn was not
    threaded through. False disables it, to still be able to reproduce/inspect the old (broken) behavior."""
    task_types = task_types or qs.TASK_TYPES
    out = {}
    for family, tok in tokenizers.items():
        count_fn = (lambda text, tok=tok: len(tok.encode(text))) if use_count_fn else None
        fam_out = {}
        for task_type in task_types:
            cells = []
            for target in target_lengths:
                task = qs.build_task(task_type, target, seed, count_fn=count_fn)
                actual = len(tok.encode(task.prompt))
                ratio = actual / target
                cells.append({"target": target, "actual": actual, "ratio": round(ratio, 4),
                             "ok": OK_LOW <= ratio <= OK_HIGH})
            fam_out[task_type] = cells
        out[family] = fam_out
    return out


def summarize(results):
    """{family: {"all_ok": bool, "worst": {"task_type","target","ratio"}, "fail_cells": [...]}}"""
    out = {}
    for family, tasks in results.items():
        fails = []
        worst = None
        for task_type, cells in tasks.items():
            for c in cells:
                if not c["ok"]:
                    fails.append({"task_type": task_type, **c})
                if worst is None or abs(c["ratio"] - 1.0) > abs(worst["ratio"] - 1.0):
                    worst = {"task_type": task_type, **c}
        out[family] = {"all_ok": not fails, "worst": worst, "fail_cells": fails}
    return out


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="write full results + summary JSON here")
    ap.add_argument("--families", default=None, help="comma list restricting which tokenizer families to load (default: all)")
    return ap


def main():
    args = build_arg_parser().parse_args()
    families = {k: TOKENIZERS[k] for k in args.families.split(",")} if args.families else None
    tokenizers = load_tokenizers(families)
    results = calibrate(tokenizers)
    summary = summarize(results)
    payload = {"results": results, "summary": summary, "task_types": list(qs.TASK_TYPES),
              "target_lengths": list(TARGET_LENGTHS), "tokenizer_repos": TOKENIZERS}
    text = json.dumps(payload, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    for family, s in summary.items():
        status = "ALL OK" if s["all_ok"] else f"{len(s['fail_cells'])} FAIL"
        print(f"{family}: {status}; worst ratio {s['worst']['ratio']} ({s['worst']['task_type']}@{s['worst']['target']})")


if __name__ == "__main__":
    main()
