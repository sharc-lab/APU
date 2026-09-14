"""First-turn residency render equivalence (DISPATCH K) - offline, no generate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

pytest.importorskip("openvino_genai")
pytest.importorskip("transformers")

from bfcl_feasibility_probe import (  # noqa: E402
    MODEL_DIR,
    assert_first_turn_token_equivalence,
    messages_for_entry,
    select_multi_turn_entries,
    tools_for_entry,
)


@pytest.fixture(scope="module")
def tok_pair():
    import openvino_genai as ov_genai
    from transformers import AutoTokenizer

    if not MODEL_DIR.is_dir():
        pytest.skip(f"model dir missing: {MODEL_DIR}")
    return (
        AutoTokenizer.from_pretrained(str(MODEL_DIR)),
        ov_genai.Tokenizer(str(MODEL_DIR)),
        ov_genai,
    )


def test_first_turn_resident_nonresident_tokens_match(tok_pair):
    hf_tok, genai_tok, ov_genai = tok_pair
    entries = select_multi_turn_entries()[:3]
    assert entries, "expected multi_turn_base entries"
    for entry in entries:
        row = assert_first_turn_token_equivalence(
            hf_tokenizer=hf_tok,
            genai_tokenizer=genai_tok,
            ov_genai=ov_genai,
            messages=messages_for_entry(entry),
            tools=tools_for_entry(entry),
            entry_id=str(entry["id"]),
        )
        assert row["identical"] is True
        assert row["n_tokens"] > 1000
        assert row["resident_sha256"] == row["non_resident_sha256"] == row["hf_sha256"]
