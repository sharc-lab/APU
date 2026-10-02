"""2026-10-02: "qwen3:30b-a3b-instruct-2507" does not exist on the Ollama registry (confirmed live against
the registry's own manifest endpoint: 404) -- every ollama_default call for this model 404'd for the whole
first night of the X2 weekend outcome-table run, silently producing zero real data for that model's ollama
leg across all 340 items. The real tag (also confirmed live: 200) is plain "qwen3:30b-a3b". This test locks
that mapping down so a future edit cannot silently reintroduce the bad tag."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

import x2_outcome_table as x2  # noqa: E402


def test_qwen3_30b_a3b_ollama_tag_is_the_real_registry_tag():
    ollama_tag, _gguf = x2.MODEL_MAP["qwen3-30b-a3b"]
    assert ollama_tag == "qwen3:30b-a3b"
    assert "instruct-2507" not in ollama_tag  # the confirmed-404 tag


def test_every_model_map_entry_has_an_ollama_tag_and_a_gguf_path():
    for model_key, (ollama_tag, gguf_path) in x2.MODEL_MAP.items():
        assert ollama_tag, f"{model_key} has an empty ollama tag"
        assert gguf_path.lower().endswith(".gguf"), f"{model_key}'s path is not a .gguf: {gguf_path}"
