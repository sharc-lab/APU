"""Token counting from the server or the model's tokenizer (src/dse/token_count.py) and the spend reservation's
upper bound (src/cloud/client.upper_bound_input_tokens), operator decision 2026-11-13."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cloud import client as cc  # noqa: E402
from src.dse import token_count as TC  # noqa: E402

HUB = Path.home() / ".cache" / "huggingface" / "hub"
TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a record by id.",
          "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}}]
MSGS = [
    {"role": "system", "content": "You are an operations assistant. Follow the rules.\nRule 1: answer in JSON."},
    {"role": "user", "content": "Find record REC-0042 and tell me its owner."},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": "lookup", "arguments": {"id": "REC-0042"}}}]},
    {"role": "tool", "content": json.dumps({"id": "REC-0042", "owner": "Dana", "status": "open"})},
    {"role": "assistant", "content": "The owner of REC-0042 is Dana."},
    {"role": "user", "content": "Thanks. Now close it, and log the event with code HTAG-77Q2."},
]


# ── llama-server counter (fake server: the "template" joins contents, a "token" is a character)
def _fake_server(calls):
    def post(url, payload, timeout):
        calls.append(url.rsplit("/", 1)[-1])
        if url.endswith("/apply-template"):
            body = "".join(f"<{m['role']}>{m.get('content') or ''}" for m in payload["messages"])
            if payload.get("tools"):
                body = json.dumps(payload["tools"]) + body
            return {"prompt": body + "<gen>"}
        return {"tokens": list(range(len(payload["content"])))}
    return post


def test_server_counter_sums_to_the_whole_render_and_caches():
    calls = []
    c = TC.LlamaServerCounter("http://x:8080", post=_fake_server(calls))
    counts, framing = c.counts(MSGS, TOOLS)
    assert sum(counts) + framing == c.n_prompt(MSGS, TOOLS)
    assert all(n >= 0 for n in counts) and len(counts) == len(MSGS)
    n_before = len(calls)
    c.counts(MSGS, TOOLS)                                  # same request again: all cached
    assert len(calls) == n_before
    c.counts(MSGS + [{"role": "user", "content": "more"}], TOOLS)
    assert len(calls) - n_before == 4                      # only the new prefix and the new whole render


def test_make_counter_prefers_the_server_and_never_falls_back_to_chars4():
    srv = TC.LlamaServerCounter("http://x:8080", post=_fake_server([]))
    assert TC.make_counter("llama_server", "qwen3:14b", server_counter=srv) is srv
    with pytest.raises(TC.TokenCountUnavailable):
        TC.make_counter("ollama", "qwen3:14b")                       # Ollama cannot count, no tokenizer given
    with pytest.raises(TC.TokenCountUnavailable):
        TC.make_counter("ollama", "llama3.3:70b", tokenizer_json="t.json")   # llama3 framing not verified

    def down(url, payload, timeout):
        raise OSError("connection refused")
    with pytest.raises(TC.TokenCountUnavailable):
        TC.make_counter("llama_server", "mistral-small3.2:24b",
                        server_counter=TC.LlamaServerCounter("http://x:1", post=down))


# ── tokenizer counter against the model's own chat template (Qwen3, real tokenizer and template)
def _qwen3():
    snaps = glob.glob(str(HUB / "models--Qwen--Qwen3-8B" / "snapshots" / "*"))
    if not snaps:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")
    pytest.importorskip("tokenizers")
    pytest.importorskip("jinja2")
    return Path(snaps[0])


@pytest.mark.parametrize("tools", [TOOLS, None])
def test_chatml_counter_equals_the_qwen3_template(tools):
    import jinja2
    from tokenizers import Tokenizer
    snap = _qwen3()
    cfg = json.loads((snap / "tokenizer_config.json").read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, indent=None: json.dumps(x, ensure_ascii=False, indent=indent)
    kw = {"messages": MSGS, "add_generation_prompt": True}
    if tools:
        kw["tools"] = tools
    rendered = env.from_string(cfg["chat_template"]).render(**kw)
    real = len(tok.encode(rendered, add_special_tokens=False).ids)
    counts, framing = TC.TokenizerCounter("chatml", tokenizer_json=str(snap / "tokenizer.json")).counts(MSGS, tools)
    assert sum(counts) + framing == real


# ── the router uses a counter object for messages and tools together
def test_router_routes_on_the_counter():
    from src.dse import router as R

    class Big:
        def counts(self, messages, tools=None):
            return [5000] * len(messages), 100

    class Small:
        def counts(self, messages, tools=None):
            return [10] * len(messages), 5

    step = {"messages": MSGS, "tools": TOOLS, "num_predict": 256}
    kw = dict(envelope=None, budget_usd=0.0, quality_floor=0.0, latency_target_ms=None, hardware="evo-x2",
              runtime="ollama_ctx_8192", model="qwen3:14b")
    big = R.Router(token_counter=Big(), **kw)
    small = R.Router(token_counter=Small(), **kw)
    assert big._count(step) == ([5000] * len(MSGS), 100)
    assert small.decide(step).target == "local"
    assert big.decide(step).target != "local"


# ── spend reservation upper bound
def test_upper_bound_never_below_measured_anthropic_counts():
    rows = [json.loads(x) for x in (REPO / "results" / "token_reservation_check_v1.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 40
    assert all(r["new_upper_bound"] >= r["anthropic_tokens"] for r in rows)
    assert all(r["old_word_count"] < r["anthropic_tokens"] for r in rows)   # the defect the bound replaces


def test_upper_bound_counts_tools_framing_and_media_but_not_schema_types():
    m = [{"role": "user", "content": "hi"}]
    base = cc.upper_bound_input_tokens(m)
    with_tools = cc.upper_bound_input_tokens(m, {"tools": TOOLS})
    assert with_tools - base >= cc.UPPER_TOOL_USE_SYSTEM_TOKENS + len(json.dumps(TOOLS))
    assert with_tools - base < cc.UPPER_NON_TEXT_BLOCK_TOKENS                # "type": "object" is not a media block
    img = [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://e/x.png"}},
                                        {"type": "text", "text": "what is this"}]}]
    assert cc.upper_bound_input_tokens(img) >= cc.UPPER_NON_TEXT_BLOCK_TOKENS
    s = "é" * 100                                                       # 2 UTF-8 bytes per character
    assert cc.upper_bound_input_tokens([{"role": "user", "content": s}]) >= 200


def test_reservation_uses_the_upper_bound():
    msgs = [{"role": "user", "content": "{\"a\":1,\"b\":[1,2,3]}" * 500}]          # one whitespace-free "word"
    tok, _usd = cc.CloudClient._reservation("claude-sonnet-5-5", msgs, 256, {})
    assert tok == cc.upper_bound_input_tokens(msgs)
    assert tok > len(msgs[0]["content"])
