"""src/cloud/prompt_cache.py: the rolling cache breakpoint (live message builder) and its pricing model, plus the
offline repricing of the recorded B3 v2 runs (analysis/rolling_cache_reprice.py). No network, no call."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "analysis", REPO / "harness", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cloud import prompt_cache as pc  # noqa: E402

EPH = {"type": "ephemeral"}
PRICES = json.loads((REPO / "results" / "cloud_prices_anthropic_20261014.json").read_text(encoding="utf-8"))
SONNET = PRICES["models"]["claude-sonnet-5-5"]["standard"]["short"]


def _bp(m):
    c = m.get("content")
    return [b for b in c if isinstance(b, dict) and b.get("cache_control")] if isinstance(c, list) else []


def _tool_loop():
    return [{"role": "system", "content": "sys"},
            {"role": "user", "content": "earlier request"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "work order"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_4_0", "type": "function",
                                                                 "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_4_0", "content": "{\"ok\": true}"}]


# ── live message builder ─────────────────────────────────────────────────────────────────────────────

def test_latest_turn_carries_the_breakpoint_not_the_last_user_message():
    msgs = _tool_loop()
    out = pc.with_rolling_breakpoints(msgs)
    assert _bp(out[0]) and _bp(out[-1])                       # system prompt and the latest turn (a tool result)
    assert out[-1]["content"][0]["cache_control"] == EPH and out[-1]["content"][0]["text"] == "{\"ok\": true}"
    assert out[3]["content"] == "work order"                  # the last user message no longer carries one
    assert pc.count_breakpoints(out) == 2
    assert msgs[-1]["content"] == "{\"ok\": true}"            # the input is not mutated


def test_breakpoint_moves_forward_as_the_conversation_grows():
    msgs = _tool_loop()
    first = pc.with_rolling_breakpoints(msgs)
    grown = msgs + [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "next task"}]
    second = pc.with_rolling_breakpoints(grown)
    assert _bp(first[5]) and not _bp(second[5]) and _bp(second[-1])
    assert pc.count_breakpoints(second) == 2


def test_empty_latest_message_falls_back_to_the_latest_with_text():
    msgs = _tool_loop()[:5]                                    # last message: assistant tool call with empty content
    out = pc.with_rolling_breakpoints(msgs)
    assert not _bp(out[4]) and out[4]["content"] == ""        # an empty text block would be rejected
    assert _bp(out[3])


def test_never_more_than_the_allowed_breakpoints():
    pre = [{"role": "system", "content": [{"type": "text", "text": "s", "cache_control": EPH}]}] + [
        {"role": "user", "content": [{"type": "text", "text": f"u{i}", "cache_control": EPH}]} for i in range(3)]
    pre.append({"role": "user", "content": "latest"})
    out = pc.with_rolling_breakpoints(pre)
    assert pc.count_breakpoints(out) == pc.MAX_BREAKPOINTS == 4
    assert out[-1]["content"] == "latest"                      # no fifth breakpoint
    three = pre[:3] + [pre[-1]]
    out3 = pc.with_rolling_breakpoints(three)
    assert pc.count_breakpoints(out3) == 4 and _bp(out3[-1])  # room for exactly one more: the latest turn
    for n in range(1, 8):
        msgs = [{"role": "system", "content": "s"}] + [{"role": "user", "content": f"u{i}"} for i in range(n)]
        assert pc.count_breakpoints(pc.with_rolling_breakpoints(msgs)) <= pc.MAX_BREAKPOINTS


def test_no_system_prompt_marks_only_the_latest_and_list_content_keeps_its_blocks():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    out = pc.with_rolling_breakpoints(msgs)
    assert out[0]["content"][0].get("cache_control") is None and out[0]["content"][1]["cache_control"] == EPH
    assert pc.count_breakpoints(out) == 1


def test_live_builder_in_cloud_redo_is_the_rolling_one():
    spec = importlib.util.spec_from_file_location("cloud_redo_pc", REPO / "scripts" / "cloud_redo.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    out = cr.with_cache_control(_tool_loop())
    assert _bp(out[0]) and _bp(out[-1]) and not _bp(out[3])
    assert cr.CACHE_BREAKPOINTS == "system prompt and latest message (rolling)"


def test_litellm_keeps_the_breakpoint_on_a_tool_result():
    litellm = pytest.importorskip("litellm")
    from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
    out = pc.with_rolling_breakpoints(_tool_loop())[1:]
    conv = anthropic_messages_pt(out, model="claude-sonnet-5-5", llm_provider="anthropic")
    last = conv[-1]["content"][-1]
    assert last["type"] == "tool_result"
    assert any(isinstance(b, dict) and b.get("cache_control") == EPH for b in last["content"])
    assert litellm is not None


# ── pricing model ────────────────────────────────────────────────────────────────────────────────────

def test_rolling_split_reads_previous_prompt_and_writes_only_new_tokens():
    assert pc.rolling_split([1000, 1500, 2100], 512) == [(0, 1000, 0), (1000, 500, 0), (1500, 600, 0)]
    assert pc.rolling_split([300, 800], 512) == [(0, 0, 300), (0, 800, 0)]       # under the minimum: not cached
    assert pc.rolling_split([2000, 1500], 512) == [(0, 2000, 0), (0, 1500, 0)]   # history rewritten: no prefix read
    assert pc.rolling_split([1000, 1200], 512, first=(990, 6, 4)) == [(990, 6, 4), (1000, 200, 0)]
    with pytest.raises(ValueError):
        pc.rolling_split([1000], 512, first=(1, 1, 1))


def test_pricing_at_the_anthropic_price_file():
    """Cache reads at the cache-hit price, writes at the 5-minute cache-write price, from the committed price file."""
    assert (SONNET["cached_input"], SONNET["cache_write"], SONNET["input"]) == (0.10, 2.50, 2.00)
    prompts, outputs = [10_000, 12_000], [100, 50]
    split = pc.rolling_split(prompts, 512)
    usd = pc.split_cost(SONNET, split, prompts, outputs)
    assert usd == pytest.approx((10_000 * 2.50 + 10_000 * 0.10 + 2_000 * 2.50 + 150 * 10.0) / 1e6)
    # the earlier placement on the same tokens: the tool round (2,000) sent uncached, never written
    old = pc.split_cost(SONNET, [(0, 10_000, 0), (10_000, 0, 2_000)], prompts, outputs)
    assert old == pytest.approx((10_000 * 2.50 + 10_000 * 0.10 + 2_000 * 2.00 + 150 * 10.0) / 1e6)
    # a third call: rolling reads the whole previous prompt, the earlier placement re-sends both rounds uncached
    p3 = prompts + [14_000]
    o3 = outputs + [0]
    new3 = pc.split_cost(SONNET, pc.rolling_split(p3, 512), p3, o3)
    old3 = pc.split_cost(SONNET, [(0, 10_000, 0), (10_000, 0, 2_000), (10_000, 0, 4_000)], p3, o3)
    assert new3 < old3


def test_long_context_rate_applies_per_call():
    haiku = {"short": PRICES["models"]["claude-haiku-5-5"]["standard"]["short"],
             "long": PRICES["models"]["claude-haiku-5-5"]["standard"]["long"], "long_context_above": 100_000}
    usd = pc.split_cost(haiku, [(0, 0, 150_000)], [150_000], [0])
    assert usd == pytest.approx(150_000 * haiku["long"]["input"] / 1e6)


def test_first_split_from_totals_inverts_the_earlier_placement():
    # earlier placement: call 1 writes 15,846 (cold), call 2 reads it and sends 401 uncached (a recorded redo row)
    assert pc.first_split_from_totals([15_850, 16_247], 15_846, 15_846) == (0.0, 15_846.0, 4.0)
    assert pc.first_split_from_totals([23_214, 23_538, 23_900, 24_100], 69_630, 23_210) == (0.0, 23_210.0, 4.0)
    with pytest.raises(ValueError):
        pc.first_split_from_totals([1000, 1200], 5000, 0)


def test_reprice_recorded_calls_keeps_a_recorded_first_call():
    calls = [{"input_tokens": 5509, "output_tokens": 117, "cache_read_tokens": 5505, "cache_write_tokens": 0,
              "uncached_input_tokens": 4},
             {"input_tokens": 5654, "output_tokens": 77, "cache_read_tokens": 5505, "cache_write_tokens": 0,
              "uncached_input_tokens": 149}]
    r = pc.reprice_recorded_calls(calls, SONNET, 512)
    assert r["split"] == [(5505, 0, 4), (5509, 145, 0)]
    assert r["uncached"] == 4


# ── cloud_task_cost rolling case ──────────────────────────────────────────────────────────────────────

def test_cloud_task_cost_rolling_case():
    import cloud_task_cost as ct
    p = {"models": {"m": {"min_cacheable_tokens": 100, "standard": {
        "short": {"input": 1.0, "cached_input": 0.1, "cache_write": 1.25, "output": 5.0},
        "long": {"input": 1.0, "cached_input": 0.1, "cache_write": 1.25, "output": 5.0}}}}}
    assert ct.task_cost(p, "m", [1000, 1500], [0, 0], "rolling") == pytest.approx(
        ct.task_cost(p, "m", [1000, 1500], [0, 0], "best"))
    # a shrinking step (condensed history): rolling reads nothing, best reads the shorter prompt
    roll = ct.task_cost(p, "m", [1000, 800], [0, 0], "rolling")
    assert roll == pytest.approx((1000 * 1.25 + 800 * 1.25) / 1e6)
    assert roll > ct.task_cost(p, "m", [1000, 800], [0, 0], "best")
    assert ct.ROLLING_CASE not in ct.CASES and ct.ROLLING_CASE in ct.ALL_CASES


# ── recorded runs, repriced (committed result files) ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def reprice():
    import rolling_cache_reprice as rr
    ac = rr.all_cloud_rolling(REPO)
    rd = rr.redo_rolling(REPO)
    return rr, ac, rd


def test_recorded_split_reproduces_each_rows_own_usd(reprice):
    _rr, ac, rd = reprice
    for m in ac.values():
        for v in m.values():
            assert v["usd_old"] == pytest.approx(v["usd_recorded"], abs=1e-9)
        # per turn NEW can exceed OLD (a tool round is now written on its own turn, where OLD wrote it on the next
        # turn's first call), but over the run it is lower
        assert sum(v["usd_new"] for v in m.values()) < sum(v["usd_old"] for v in m.values())
    for v in list(rd["redo"].values()) + list(rd["rb"].values()):
        assert v["usd_old"] == pytest.approx(v["usd_recorded"], abs=1e-9)


def test_all_cloud_repriced_keeps_tokens_and_outcomes(reprice):
    rr, ac, _rd = reprice
    s = rr.b3_all_cloud_summary(ac)
    for v in s.values():
        assert v["read_old"] + v["write_old"] + v["uncached_old"] == v["read_new"] + v["write_new"] + v["uncached_new"]
        assert v["uncached_new"] < v["uncached_old"] and v["usd_new"] < v["usd_old"]
    assert s["claude-sonnet-5-5"]["correct"] == 90 and s["claude-sonnet-5-5"]["n"] == 90
