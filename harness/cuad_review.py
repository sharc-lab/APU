"""CUAD contract review subset with a deterministic grader (docs/EXPERIMENTS.md "PRE-REGISTRATION (2026-10-29)",
F2 cuad-pipeline). Pure functions plus a manifest builder; no model call lives here.

Task material stays out of the repo: the CUAD data (data.zip from https://github.com/TheAtticusProject/cuad) is read
from a folder outside the repo, and only results/cuad_manifest_v1.json (titles, sha256 hashes, counts) is committed.

Task: the contract is split into numbered segments (non-empty lines); the model answers a JSON list of
{"category": <one of CATEGORIES>, "segment": <int>}. Gold = every (category, segment) whose segment overlaps a CUAD
gold answer span of that category. Correct iff the recorded set equals the gold set exactly.

    python harness/cuad_review.py --build-manifest <dir with test.json> [--zip <data.zip>] --out results/cuad_manifest_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

CATEGORIES = ("Governing Law", "Expiration Date", "Renewal Term", "Termination For Convenience", "Anti-Assignment",
              "Exclusivity", "Non-Compete", "Cap On Liability", "Audit Rights", "License Grant")
ORDER_SALT = "20261029"
N_FIT, N_OVER = 40, 10
FIT_TOKENS = 30000           # estimated prompt must fit this to be in the "fits" stratum
WINDOW = 40960               # qwen3:14b Ollama default loaded window (row x2-evalset-tasks-real-v2-wrong-rate)
CHARS_PER_TOKEN = 4.0        # ASSUMPTION (labelled wherever used)
PROMPT_OVERHEAD_TOKENS = 600  # ASSUMPTION: instructions + category list
SEGMENT_ID_TOKENS = 3        # ASSUMPTION: "[123] " prefix per segment
OUTPUT_TOKENS = {"point": 300, "high": 800}   # ASSUMPTION
MANIFEST_VERSION = "cuad_manifest_v1"


def segment(context: str) -> list[tuple[int, int]]:
    """(start, end) char offsets of the non-empty lines of the contract, in order; segment ids are 1-based."""
    out, pos = [], 0
    for line in context.split("\n"):
        s, e = pos, pos + len(line)
        if line.strip():
            out.append((s, e))
        pos = e + 1
    return out


def gold_pairs(context: str, answers: dict[str, list[tuple[int, str]]], categories=CATEGORIES) -> set[tuple[str, int]]:
    """Every (category, segment id) whose segment overlaps a gold span [start, start + len(text)) of that category."""
    segs = segment(context)
    out = set()
    for cat in categories:
        for start, text in answers.get(cat, ()):
            end = start + len(text)
            for i, (s, e) in enumerate(segs, 1):
                if s < end and start < e:
                    out.add((cat, i))
    return out


def parse_answer(text: str | None, n_segments: int, categories=CATEGORIES) -> tuple[set[tuple[str, int]] | None, str | None]:
    """The model's JSON list as a set of (category, segment); (None, reason) when the shape check fails (not JSON, not a
    list of objects, unknown category, segment out of range). The first [...] block in the text is read."""
    if text is None:
        return None, "no text"
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None, "no JSON list"
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "not JSON"
    out = set()
    for it in items:
        if not isinstance(it, dict) or "category" not in it or "segment" not in it:
            return None, "item not an object with category and segment"
        c, s = it["category"], it["segment"]
        if c not in categories:
            return None, f"unknown category {c!r}"
        if isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= n_segments:
            return None, f"segment {s!r} out of range"
        out.add((c, s))
    return out, None


def grade(recorded: set | None, gold: set) -> dict:
    """Exact-set grader plus pair-level counts. recorded None (shape check failed) is wrong with 0 true positives."""
    rec = recorded or set()
    tp = len(rec & gold)
    p = tp / len(rec) if rec else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else (1.0 if not rec else 0.0)
    return {"correct": recorded is not None and rec == gold, "tp": tp, "fp": len(rec - gold), "fn": len(gold - rec),
            "precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if p + r else 0.0}


def load_squad(path) -> list[dict]:
    """CUAD SQuAD-format file -> [{title, context, answers: {category: [(start, text)]}}]."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    out = []
    for c in d:
        para = c["paragraphs"][0]
        ans = {}
        for q in para["qas"]:
            cat = q["id"].split("__", 1)[1]
            ans[cat] = [(a["answer_start"], a["text"]) for a in q["answers"]]
        out.append({"title": c["title"], "context": para["context"], "answers": ans})
    return out


def est_prompt_tokens(context: str) -> float:
    """ASSUMPTION: chars / 4 + 3 tokens per segment id + 600 tokens of instructions."""
    return len(context) / CHARS_PER_TOKEN + SEGMENT_ID_TOKENS * len(segment(context)) + PROMPT_OVERHEAD_TOKENS


def order_key(title: str) -> str:
    return hashlib.sha256(f"{ORDER_SALT}:{title}".encode("utf-8")).hexdigest()


def select_subset(contracts: list[dict]) -> list[dict]:
    """First N_FIT contracts (by order_key) whose estimate fits FIT_TOKENS, then the first N_OVER over WINDOW."""
    fit, over = [], []
    for c in sorted(contracts, key=lambda c: order_key(c["title"])):
        t = est_prompt_tokens(c["context"])
        if t <= FIT_TOKENS and len(fit) < N_FIT:
            fit.append({**c, "stratum": "fits"})
        elif t > WINDOW and len(over) < N_OVER:
            over.append({**c, "stratum": "over_window"})
    return fit + over


def manifest(contracts: list[dict], zip_sha256: str | None, n_split: int) -> dict:
    """The committed manifest: no contract or clause text."""
    rows = []
    for c in select_subset(contracts):
        g = gold_pairs(c["context"], c["answers"])
        rows.append({"title": c["title"], "stratum": c["stratum"],
                     "context_sha256": hashlib.sha256(c["context"].encode("utf-8")).hexdigest(),
                     "n_chars": len(c["context"]), "n_segments": len(segment(c["context"])),
                     "est_prompt_tokens": round(est_prompt_tokens(c["context"])),
                     "gold_pairs": len(g), "gold_empty": not g})
    over_in_split = sum(1 for c in contracts if est_prompt_tokens(c["context"]) > WINDOW)
    return {"record": MANIFEST_VERSION, "dataset": "CUAD v1 (The Atticus Project), test split (test.json of data.zip)",
            "licence": "CC BY 4.0 (results/cuad_licence_20261029.json)", "source_zip_sha256": zip_sha256,
            "source_url": "https://github.com/TheAtticusProject/cuad/raw/main/data.zip",
            "local_copy_note": "data.zip and its extracted files are kept outside the repo (the session scratchpad); "
                               "nothing from them but this manifest is committed",
            "categories": list(CATEGORIES), "order_salt": ORDER_SALT, "n_fit_target": N_FIT, "n_over_target": N_OVER,
            "fit_tokens": FIT_TOKENS, "window": WINDOW, "n_contracts_in_split": n_split,
            "n_over_window_in_split": over_in_split,
            "token_estimate": f"ASSUMPTION: chars / {CHARS_PER_TOKEN:g} + {SEGMENT_ID_TOKENS} per segment + "
                              f"{PROMPT_OVERHEAD_TOKENS}",
            "contracts": rows}


def estimate(man: dict, haiku_rates: dict, prefill_tok_s: float, decode_tok_s: float) -> dict:
    """ESTIMATE of cloud USD (upper bound, all input uncached) and local evo-x2 hours per arm, one call per contract.
    prefill_tok_s and decode_tok_s are measured inputs passed in; high case = prefill at half the rate and the high
    output tokens (ASSUMPTION)."""
    def call_usd(prompt, out):
        above = haiku_rates.get("long_context_above")
        r = haiku_rates["long"] if above is not None and prompt > above else haiku_rates["short"]
        return (prompt * r["input"] + out * r["output"]) / 1e6
    cs = man["contracts"]
    res = {"n": len(cs), "n_over": sum(1 for c in cs if c["stratum"] == "over_window")}
    for case in ("point", "high"):
        out_t = OUTPUT_TOKENS[case]
        pre = prefill_tok_s if case == "point" else prefill_tok_s / 2
        cloud_all = sum(call_usd(c["est_prompt_tokens"], out_t) for c in cs)
        cloud_routed = sum(call_usd(c["est_prompt_tokens"], out_t) for c in cs if c["est_prompt_tokens"] > man["window"])
        local_s = sum(min(c["est_prompt_tokens"], man["window"]) / pre + out_t / decode_tok_s for c in cs)
        local_fit_s = sum(c["est_prompt_tokens"] / pre + out_t / decode_tok_s for c in cs
                          if c["est_prompt_tokens"] <= man["window"])
        res[case] = {"cloud_arm_usd": cloud_all, "pipeline_routed_usd": cloud_routed,
                     "local_arm_h": local_s / 3600, "pipeline_local_h": local_fit_s / 3600}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-manifest", required=True, help="folder holding CUAD test.json (outside the repo)")
    ap.add_argument("--zip", help="data.zip, for its sha256")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cs = load_squad(Path(a.build_manifest) / "test.json")
    zsha = hashlib.sha256(Path(a.zip).read_bytes()).hexdigest() if a.zip else None
    m = manifest(cs, zsha, len(cs))
    Path(a.out).write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out}: {len(m['contracts'])} contracts "
          f"({sum(1 for c in m['contracts'] if c['stratum'] == 'over_window')} over window)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
