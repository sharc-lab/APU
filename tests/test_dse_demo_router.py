"""Tests for src/dse/router.Router (the per-step demo router), src/dse/r2_replay.py and src/dse/envelope_data.json.

Every cloud client here is a stub on a temporary ledger; the router never calls the cloud (a client whose call path
raises proves it). The R2 tests replay the real x2_r2_real_v1 sessions.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
from src.cloud.client import CloudClient  # noqa: E402
from src.dse import r2_replay  # noqa: E402
from src.dse.router import (CONTEXT_MARGIN, ENVELOPE_DATA_PATH, Envelope, RouteDecision, Router,  # noqa: E402
                            norm_model, parse_runtime, r2_message_tokens_raw, split_turns)

R2_FILE = REPO / "results" / "x2_r2_real_v1.jsonl"
ENV = Envelope.load()


class NoCallClient(CloudClient):
    """A stub client whose call path fails the test if the router ever uses it."""

    def call(self, *a, **k):  # pragma: no cover - must never run
        raise AssertionError("router must not call the cloud")

    def prepare(self, *a, **k):  # pragma: no cover - must never run
        raise AssertionError("router must not call the cloud")


def _client(tmp_path, name="ledger.jsonl"):
    c = NoCallClient(mode="stub", ledger_path=Path(tmp_path) / name)
    assert c.stub_mode
    return c


def _router(tmp_path, runtime="ollama_ctx_1000", hardware="evo-x2", model="llama3.1:8b", budget=5.0, floor=0.9,
            latency=None, **kw):
    return Router(ENV, budget, floor, latency, hardware, runtime, model, cloud_client=_client(tmp_path), **kw)


def _conv(n_turns, tool_turns=()):
    """system + n_turns turns; turn k (1-based) in tool_turns has an assistant tool call and a tool result."""
    msgs = [{"role": "system", "content": "rules"}]
    for k in range(1, n_turns + 1):
        msgs.append({"role": "user", "content": f"task {k}"})
        if k in tool_turns:
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f"}}]})
            msgs.append({"role": "tool", "content": "ok"})
        msgs.append({"role": "assistant", "content": f"answer {k}"})
    return msgs


def _step(msgs, per_msg=100, num_predict=100, **kw):
    return {"messages": msgs, "num_predict": num_predict, "token_counts": [per_msg] * len(msgs), **kw}


# ── envelope data ────────────────────────────────────────────────────────────────────────────────────────────────

def test_envelope_data_matches_register_rebuild():
    import build_envelope_data as bed
    import numbers_register as nr
    for cid in ("T2S-vs-X2-default-ctx", "K1v3-X2-table", "R2-real-v1-loaded-ctx", "ttft-physical-fit-per-machine",
                "decode-rate-per-machine", "B3-corunner-6model", "PX2-full-ratio-table", "A-24-budget-boundary"):
        entry = next(e for e in nr.NUMBER_ENTRIES if e["claim_id"] == cid)
        missing = [f for f in entry["data_files"] if "*" not in f and not (REPO / f).exists()]
        if missing:
            pytest.skip(f"register data not present: {missing}")
    assert bed.dumps(bed.build()) == ENVELOPE_DATA_PATH.read_text(encoding="utf-8"), \
        "src/dse/envelope_data.json is stale or hand-edited; run py -3.12 analysis/build_envelope_data.py"


def test_envelope_every_block_cites_a_register_row():
    data = json.loads(ENVELOPE_DATA_PATH.read_text(encoding="utf-8"))
    import numbers_register as nr
    ids = {e["claim_id"] for e in nr.NUMBER_ENTRIES}
    assert set(data["sources"]) <= ids
    for hw, m in data["machines"].items():
        for model, v in m["ollama_default_ctx"].items():
            assert v["claim_ids"] and set(v["claim_ids"]) <= ids
        assert m["co_runner"]["claim_id"] in ids


def test_ollama_default_ctx_comes_from_envelope(tmp_path):
    for hw in ("evo-t2s", "evo-x2"):
        for model, v in ENV.data["machines"][hw]["ollama_default_ctx"].items():
            r = Router(ENV, 1.0, 0.9, None, hw, "ollama", model, cloud_client=_client(tmp_path))
            assert r.num_ctx == v["ctx"]
            assert r.usable_ctx == math.floor(v["ctx"] * (1 - CONTEXT_MARGIN))
    with pytest.raises(ValueError):
        Router(ENV, 1.0, 0.9, None, "evo-t2s", "ollama", "no-such-model", cloud_client=_client(tmp_path))


def test_parse_runtime():
    assert parse_runtime("ollama") == ("ollama", None)
    assert parse_runtime("ollama_default_call2_notools") == ("ollama", None)
    assert parse_runtime("ollama_ctx_4096_call2_notools") == ("ollama", 4096)
    assert parse_runtime("ollama:8192") == ("ollama", 8192)
    assert parse_runtime("llama_server:32768") == ("llama_server", 32768)
    assert parse_runtime("llama-server_ctx_16384") == ("llama_server", 16384)
    with pytest.raises(ValueError):
        parse_runtime("llama_server")
    with pytest.raises(ValueError):
        parse_runtime("vllm")
    assert norm_model("llama3.1:8b") == norm_model("llama31-8b")


# ── turn structure and trimming ──────────────────────────────────────────────────────────────────────────────────

def test_split_turns_keeps_tool_round_trip_in_its_turn():
    msgs = _conv(3, tool_turns=(2,))
    sys_idx, turns = split_turns(msgs)
    assert sys_idx == [0]
    assert [[msgs[i]["role"] for i in t] for t in turns] == [
        ["user", "assistant"], ["user", "assistant", "tool", "assistant"], ["user", "assistant"]]


def test_fits_untrimmed_routes_local(tmp_path):
    d = _router(tmp_path).decide(_step(_conv(2), per_msg=50))
    assert isinstance(d, RouteDecision)
    assert d.target == "local" and d.dropped_turns == 0 and d.kept_turns == 2 and d.kept_turn_ids == [1, 2]
    assert d.tokens == {"system": 50, "kept_history": 200, "answer_budget": 100, "total": 350,
                        "full_history": 200, "full_total": 350}
    assert d.est_cost_usd == 0.0 and d.stub is True


def test_over_window_trims_whole_oldest_turns_and_fits(tmp_path):
    r = _router(tmp_path)  # num_ctx 1000, usable 950
    msgs = _conv(5, tool_turns=(1, 3))
    d = r.decide(_step(msgs, per_msg=90))
    assert d.target == "local_trimmed"
    assert d.tokens["total"] <= r.usable_ctx < r.num_ctx
    assert d.dropped_turn_ids == list(range(1, d.dropped_turns + 1))  # oldest first, contiguous
    assert 5 in d.kept_turn_ids  # the current turn is never dropped
    sys_idx, turns = split_turns(msgs)
    kept_msgs = [msgs[i] for t in d.kept_turn_ids for i in turns[t - 1]]
    assert kept_msgs[0]["role"] == "user"  # no orphaned tool result, no assistant without its user
    assert d.tokens["system"] == 90  # system prompt kept
    assert "dropped turn" in d.reason and "kept system prompt" in d.reason
    assert "\n" not in d.reason


def test_needed_turn_dropped_goes_to_cloud_within_budget(tmp_path):
    r = _router(tmp_path)
    d = r.decide(_step(_conv(5), per_msg=90, needs_turns=[1]))
    assert d.target == "cloud" and d.machine == "cloud" and d.model == "gpt-4o-mini"
    assert d.est_cost_usd > 0 and d.budget_remaining_usd == pytest.approx(5.0 - d.est_cost_usd)
    assert "needs" in d.reason
    # budget is state: a second cloud decision lowers it again
    d2 = r.decide(_step(_conv(5), per_msg=90, needs_turns=[1]))
    assert d2.budget_remaining_usd == pytest.approx(5.0 - 2 * d.est_cost_usd)


def test_needed_turn_dropped_without_budget_trims_and_flags_floor(tmp_path):
    r = _router(tmp_path, budget=0.0)
    d = r.decide(_step(_conv(5), per_msg=90, needs_turns=[1]))
    assert d.target == "local_trimmed" and d.quality_floor_met is False
    assert d.tokens["total"] <= r.num_ctx
    assert "NOT met" in d.reason


def test_quality_floor_zero_accepts_trim_and_require_full_history_forces_cloud(tmp_path):
    d = _router(tmp_path, floor=0.0).decide(_step(_conv(5), per_msg=90, needs_turns=[1]))
    assert d.target == "local_trimmed" and d.quality_floor_met
    d = _router(tmp_path, require_full_history=True).decide(_step(_conv(5), per_msg=90))
    assert d.target == "cloud"


def test_identifier_reference_detects_needed_turn(tmp_path):
    msgs = _conv(5)
    msgs[1]["content"] = "order ORD-77812 is delayed"
    msgs[-2]["content"] = "what was the status of ORD-77812?"
    assert _router(tmp_path).decide(_step(msgs, per_msg=90)).target == "cloud"
    msgs[0]["content"] = "known orders: ORD-77812"  # still in the kept system prompt: not lost
    assert _router(tmp_path).decide(_step(msgs, per_msg=90)).target == "local_trimmed"


def test_current_turn_too_big_routes_cloud_or_refuses(tmp_path):
    step = _step(_conv(1), per_msg=600)
    assert _router(tmp_path).decide(step).target == "cloud"
    d = _router(tmp_path, budget=0.0).decide(step)
    assert d.target == "refuse" and d.est_cost_usd == 0.0


def test_llama_server_over_measured_memory_budget_routes_cloud(tmp_path):
    mem = ENV.data["machines"]["evo-t2s"]["llama_server_memory"]
    model = sorted(mem["last_ok_n_ctx"])[0]
    last_ok = mem["last_ok_n_ctx"][model]
    ok = _router(tmp_path, runtime=f"llama_server:{last_ok}", hardware="evo-t2s", model=model)
    assert ok.decide(_step(_conv(1), per_msg=10)).memory_status == "FITS"
    over = _router(tmp_path, runtime=f"llama_server:{last_ok + 256}", hardware="evo-t2s", model=model)
    d = over.decide(_step(_conv(1), per_msg=10))
    assert d.target == "cloud" and d.memory_status == "OVER_BUDGET" and "A-24" in d.reason


def test_latency_and_co_runner_term(tmp_path):
    model = "qwen3-8b"
    base = _router(tmp_path, runtime="ollama_ctx_8192", hardware="evo-t2s", model=model)
    hog = _router(tmp_path, runtime="ollama_ctx_8192", hardware="evo-t2s", model=model, co_runner=True)
    step = _step(_conv(2), per_msg=500)
    a, b = base.decide(step), hog.decide(step)
    assert a.target == b.target == "local"
    assert b.predicted_latency_ms > a.predicted_latency_ms
    ratio = ENV.data["machines"]["evo-t2s"]["co_runner"]["ttft_ratio"][model]
    fit = ENV.data["machines"]["evo-t2s"]["ttft_fit"][model]
    n = a.tokens["system"] + a.tokens["kept_history"]
    ttft_ms = (fit["a"] * n + fit["b"] * n * n) * 1000
    assert b.predicted_latency_ms - a.predicted_latency_ms == pytest.approx(ttft_ms * (ratio - 1), rel=1e-3, abs=0.2)
    tight = _router(tmp_path, runtime="ollama_ctx_8192", hardware="evo-t2s", model=model, latency=1.0)
    d = tight.decide(step)
    assert d.target == "cloud" and "target" in d.reason


def test_default_client_is_stub(tmp_path):
    r = Router(ENV, 1.0, 0.9, None, "evo-x2", "ollama_ctx_4096", "llama3.1:8b")
    assert r.stub and r.client.stub_mode


def test_default_token_count_is_the_r2_harness_estimate(tmp_path):
    import x2_r2_agent as agent  # via r2_replay's sys.path setup
    msgs = _conv(3, tool_turns=(2,))
    assert sum(r2_message_tokens_raw(m) for m in msgs) == agent.transcript_tokens(msgs)
    r = _router(tmp_path, token_calib_ratio=1.0, runtime="ollama_ctx_100000")
    d = r.decide({"messages": msgs, "num_predict": 10})
    assert d.tokens["system"] + d.tokens["kept_history"] == agent.transcript_tokens(msgs)


# ── R2 replay: the real x2_r2_real_v1 sessions ──────────────────────────────────────────────────────────────────

def _r2_sessions():
    if not R2_FILE.exists():
        pytest.skip("results/x2_r2_real_v1.jsonl not present")
    return r2_replay.r2_sessions(r2_replay.read_rows(R2_FILE))


def test_r2_reconstruction_is_exact_and_no_local_decision_overflows(tmp_path):
    sessions = _r2_sessions()
    assert sessions
    for i, srows in enumerate(sessions.values()):
        router = r2_replay.router_for_session(srows, cloud_client=_client(tmp_path, f"l{i}.jsonl"))
        for d in router.replay_calls(srows):
            o = d.observed
            assert o["reconstructed_prompt_tokens_est"] == o["recorded_prompt_tokens_est"], (o, d.turn_idx)
            if d.target in ("local", "local_trimmed"):
                assert d.tokens["total"] <= router.usable_ctx <= d.num_ctx, d.reason
            if o["real_prompt_tokens_calibrated"] + 384 > o["loaded_context"]:
                assert d.target != "local", (d.turn_idx, d.call_idx, d.reason)


def test_r2_4096_router_acts_before_the_real_overflow(tmp_path):
    sessions = {k: v for k, v in _r2_sessions().items() if k[1].startswith("ollama_ctx_4096")}
    assert len(sessions) == 6  # 2 models x 3 seeds
    for i, srows in enumerate(sessions.values()):
        first_over = min(r["turn_idx"] for r in srows if r.get("over_loaded_window"))
        router = r2_replay.router_for_session(srows, cloud_client=_client(tmp_path, f"l{i}.jsonl"))
        assert router.num_ctx == 4096
        ds = router.replay_calls(srows)
        first_action = min(d.turn_idx for d in ds if d.target != "local")
        assert first_action <= first_over
        at_first_over = [d for d in ds if d.turn_idx == first_over]
        assert at_first_over and all(d.target in ("local_trimmed", "cloud") for d in at_first_over)
        for d in ds:
            if d.target != "cloud":
                assert d.tokens["total"] <= 4096
            if d.observed["canary_check"] and d.dropped_turns == 0 and d.target == "cloud":
                assert "turn 1" in d.reason  # the history tags live in turn 1


def _strip_like_dashboard_cache(srows):
    """demo/dashboard/build_cache.py drops final_text and each call's content and native_tool_calls."""
    out = []
    for r in srows:
        r = {k: v for k, v in r.items() if k != "final_text"}
        r["calls"] = [{k: v for k, v in c.items() if k not in ("content", "native_tool_calls")} for c in r["calls"]]
        out.append(r)
    return out


def test_r2_residual_reconstruction_matches_content_reconstruction():
    sessions = _r2_sessions()
    srows = next(v for k, v in sessions.items() if k[1].startswith("ollama_ctx_4096"))
    full = list(r2_replay.session_steps(srows))
    stripped = list(r2_replay.session_steps(_strip_like_dashboard_cache(srows)))
    assert len(full) == len(stripped)
    for a, b in zip(full, stripped):
        assert a["observed"]["reconstruction"] == "content" and b["observed"]["reconstruction"] == "residual"
        assert a["token_counts"] == b["token_counts"], (a["turn_idx"], a["call_idx"])
        assert a["tools_tokens"] == b["tools_tokens"]


def test_replay_session_one_decision_per_turn_row_dashboard_call(tmp_path):
    """The dashboard's call: Router(envelope, budget, floor, latency, "evo-x2", "ollama", model, None) on cached
    rows of the num_ctx 4096 arm. The rows' num_ctx (Ollama per-request option) must govern, not the default."""
    sessions = _r2_sessions()
    for (model, arm, seed, _), srows in sessions.items():
        if not arm.startswith("ollama_ctx_4096"):
            continue
        rows = _strip_like_dashboard_cache(srows)
        router = Router(None, 50.0, 0.9, 30000.0, "evo-x2", "ollama", model, _client(tmp_path, f"{seed}.jsonl"))
        ds = router.replay_session([dict(r) for r in rows])
        assert [d.turn_idx for d in ds] == [r["turn_idx"] for r in rows]
        calls = router.replay_calls(rows)  # same router, second pass: only structure is compared
        assert len(calls) == sum(len(r["calls"]) for r in rows)
        for d in ds:
            assert d.num_ctx == 4096
            assert len(d.observed["call_targets"]) == d.call_idx
            if d.target != "cloud":
                assert d.tokens["total"] <= 4096


def test_sample_decisions_file_is_current():
    out = REPO / "results" / "router_sample_decisions.txt"
    if not R2_FILE.exists() or not out.exists():
        pytest.skip("R2 file or sample output not present")
    import router_sample_decisions as rsd
    assert rsd.run() == out.read_text(encoding="utf-8")
