"""Fixed seeded document corpus for E-FILTER C2 large-payload tool returns.

Context growth under the Stage-1 BFCL slice was ~164 tokens/step because tool returns were
near-constant tiny strings. Prefill never reached parity with decode (peak context 1,826 vs
~21k needed). This corpus supplies high-variance 500-2,000 token chunks so trajectories can
reach the 20k-30k regime the filter hypothesis actually concerns.

Token counts use the same ``len(text) // 4`` proxy as the router (C2.3 lands ``+ scaffold``
separately). Chunk bodies and retrieval selections are fully determined by
``(corpus_seed, …)`` so two runs with the same seed are byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "CHUNK_TOKEN_MAX",
    "CHUNK_TOKEN_MIN",
    "DocumentChunk",
    "DocumentCorpus",
    "proxy_tokens",
]

CHUNK_TOKEN_MIN: Final = 500
CHUNK_TOKEN_MAX: Final = 2000

#: Default chunk count: enough that several retrieve/read steps can accumulate 20k-30k context.
_DEFAULT_N_CHUNKS: Final = 48

_FILLER_UNITS: Final = (
    "latency envelope",
    "resident KV bytes",
    "prefill throughput",
    "decode throughput",
    "escalation filter",
    "deadline grid",
    "arithmetic intensity",
    "cache residency",
    "context trajectory",
    "silicon partition",
    "tool payload variance",
    "bootstrap confidence",
)


def proxy_tokens(text: str) -> int:
    """Router-compatible token proxy: character length divided by four."""
    return len(text) // 4


def _stable_digest(*parts: str | int) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    path: str
    topic: str
    text: str
    token_proxy: int
    facts: dict[str, str]

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "topic": self.topic,
            "token_proxy": self.token_proxy,
            "facts": dict(self.facts),
        }


class DocumentCorpus:
    """Deterministic corpus of large document chunks."""

    def __init__(
        self,
        *,
        seed: int,
        n_chunks: int = _DEFAULT_N_CHUNKS,
        planted_facts: dict[str, dict[str, str]] | None = None,
    ) -> None:
        if n_chunks < 8:
            raise ValueError(f"n_chunks must be >= 8 for 20k+ trajectories, got {n_chunks}")
        self.seed = int(seed)
        self.n_chunks = int(n_chunks)
        self._planted = planted_facts or {}
        self._chunks: tuple[DocumentChunk, ...] = tuple(
            self._build_chunk(index) for index in range(self.n_chunks)
        )
        self._by_path = {chunk.path: chunk for chunk in self._chunks}
        self._by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._call_index = 0

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        return self._chunks

    @property
    def paths(self) -> list[str]:
        return [chunk.path for chunk in self._chunks]

    @property
    def total_token_proxy(self) -> int:
        return sum(chunk.token_proxy for chunk in self._chunks)

    def get_by_path(self, path: str) -> DocumentChunk | None:
        return self._by_path.get(path)

    def reset_call_index(self) -> None:
        """Reset per-trajectory retrieve call counter (one world may serve many tasks)."""
        self._call_index = 0

    def retrieve(self, query: str, *, call_index: int | None = None) -> list[DocumentChunk]:
        """Return 1-3 chunks, deterministic from ``(corpus_seed, query, call_index)``."""
        if call_index is None:
            call_index = self._call_index
            self._call_index += 1
        digest = _stable_digest(self.seed, query.strip().casefold(), call_index)
        rng = random.Random(digest)
        count = 1 + (digest % 3)  # 1..3
        # Prefer topic-keyword overlap when present, then fill from the full set.
        query_tokens = {tok for tok in query.casefold().replace("/", " ").split() if len(tok) > 2}
        ranked = sorted(
            self._chunks,
            key=lambda chunk: (
                -sum(
                    1
                    for tok in query_tokens
                    if tok in chunk.topic.casefold() or tok in chunk.text[:400].casefold()
                ),
                chunk.chunk_id,
            ),
        )
        # Mix: take the top keyword hits, then scramble within a seeded window for variance.
        window = ranked[: max(count * 4, count)]
        picks = rng.sample(window, k=min(count, len(window)))
        picks.sort(key=lambda chunk: chunk.chunk_id)
        return picks

    def catalog_text(self) -> str:
        """Short index listing topic → path (not a large payload)."""
        lines = ["CORPUS CATALOG", f"seed={self.seed}", f"n_chunks={self.n_chunks}", ""]
        for chunk in self._chunks:
            fact_keys = ",".join(sorted(chunk.facts)) if chunk.facts else "-"
            lines.append(
                f"{chunk.chunk_id} path={chunk.path} topic={chunk.topic} "
                f"tokens~{chunk.token_proxy} facts={fact_keys}"
            )
        return "\n".join(lines)

    def world_overlay(self) -> dict[str, Any]:
        """Directories/files fragment that ToolWorld merges for corpus paths."""
        basenames = ["catalog.txt", *[path.rsplit("/", 1)[-1] for path in self.paths]]
        return {
            "directories": {"/corpus": basenames},
            "files": {"/corpus/catalog.txt": self.catalog_text()},
            "corpus_seed": self.seed,
            "corpus_n_chunks": self.n_chunks,
            "corpus_total_token_proxy": self.total_token_proxy,
        }

    def _build_chunk(self, index: int) -> DocumentChunk:
        chunk_id = f"D{index:03d}"
        path = f"/corpus/{chunk_id}.txt"
        topic = _FILLER_UNITS[index % len(_FILLER_UNITS)]
        target_tokens = CHUNK_TOKEN_MIN + (
            _stable_digest(self.seed, "size", index) % (CHUNK_TOKEN_MAX - CHUNK_TOKEN_MIN + 1)
        )
        facts = dict(self._planted.get(chunk_id, {}))
        # Every chunk carries a stable numeric marker the research tasks can ask for.
        facts.setdefault("doc_index", str(index))
        facts.setdefault("marker", str(1000 + (_stable_digest(self.seed, "marker", index) % 9000)))

        header = (
            f"DOCUMENT {chunk_id}\n"
            f"path: {path}\n"
            f"topic: {topic}\n"
            f"FACTS_JSON: {json.dumps(facts, sort_keys=True)}\n"
            f"---\n"
        )
        body = self._filler_body(index=index, topic=topic, target_chars=target_tokens * 4)
        text = header + body
        # Trim/pad to land inside the token band after the header.
        while proxy_tokens(text) > CHUNK_TOKEN_MAX:
            text = text[: len(text) - 32].rstrip() + "\n"
        while proxy_tokens(text) < CHUNK_TOKEN_MIN:
            text += (
                f" pad-{index}-{proxy_tokens(text)} {_FILLER_UNITS[index % len(_FILLER_UNITS)]}."
            )
        return DocumentChunk(
            chunk_id=chunk_id,
            path=path,
            topic=topic,
            text=text,
            token_proxy=proxy_tokens(text),
            facts=facts,
        )

    def _filler_body(self, *, index: int, topic: str, target_chars: int) -> str:
        rng = random.Random(_stable_digest(self.seed, "body", index))
        parts: list[str] = []
        while sum(len(p) for p in parts) < target_chars:
            unit = rng.choice(_FILLER_UNITS)
            n = rng.randint(10, 40)
            parts.append(
                f"Section on {topic}: measured {unit} sample-{index}-{len(parts)} "
                f"with {n} replicates under seed {self.seed}. "
            )
        return "".join(parts)[:target_chars]
