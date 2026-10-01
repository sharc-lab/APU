"""Shared helpers for building results/workload_pack/.

Tokenizer note
--------------
This repo's own long-context tooling (harness/model2_truncation.py,
harness/stage_a_scale.py) counts tokens with a *local* model tokenizer
(llama3.1:8b / qwen3:4b-instruct via Ollama), not transformers.AutoTokenizer.
No file named analysis/agent_traces.py exists anywhere in this repository
(verified with a full-repo search before writing this pack), so there is no
established AutoTokenizer convention to copy. We use transformers.AutoTokenizer
with Qwen/Qwen2.5-7B-Instruct's real tokenizer (same model family -- Qwen3 --
as configs/default.yaml's "qwen3:4b" default model) because it is a real,
reachable, BPE tokenizer in the same family, and the task calls for a real
tokenizer rather than a word-count heuristic.

This module must be run with the Python 3.8 interpreter at C:\\Python38\\python.exe,
which has transformers==4.46.3 and datasets==3.1.0 installed. The repo's main
Python (3.12) has neither installed and they are not in pyproject.toml.
"""
from __future__ import annotations

import functools
import json
import random
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parent.parent
ITEMS_DIR = PACK_ROOT / "items"

TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"


@functools.lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rng_for(seed_key: str) -> random.Random:
    """Deterministic per-item RNG so the pack is reproducible byte-for-byte."""
    seed = 0
    for ch in seed_key:
        seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
    return random.Random(seed)
