"""timeout_latency (2026-10-08 operator decision): rows whose cause is timeout are labelled timeout_latency in analysis
only, excluded from accuracy denominators (register x2-v3-scores, src/dse/pareto.py, the dashboard fake), kept in
progress / error-cause counts, and reported in the x2-v3-usability and qwen3-32b-timeout-count register rows.
All rows here are synthetic."""
from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

import x2_outcome_table as x2  # noqa: E402
import analysis.numbers_register as nr  # noqa: E402


def orow(item, model, config, score=1.0, status=200, error=None, family="longdoc_qa", **kw):
    r = {"record": "outcome_row", "item_id": item, "family": family, "model_id": model, "config": config,
         "score": score, "http_status": status, "error": error, "latency_s": 1.0}
    r.update(kw)
    return r


TIMEOUT = dict(score=0.0, status=None, error="timed out")


# ------------------------------------------------------------------------------------------------ harness function
@pytest.mark.parametrize("model", x2.DEFAULT_MODELS)
@pytest.mark.parametrize("config", x2.CONFIGS)
def test_timeout_rows_are_timeout_latency_for_every_model_and_config(model, config):
    r = orow("longdoc_32000_00", model, config, **TIMEOUT)
    assert x2.classify_error_cause(r) == "timeout"
    assert x2.analysis_outcome(r) == x2.TIMEOUT_LATENCY == "timeout_latency"
    assert x2.counts_toward_accuracy(r) is False


@pytest.mark.parametrize("r", [
    orow("a", "m", "llama_server"),                                                     # none
    orow("a", "m", "llama_server", score=0.0),                                          # wrong answer
    orow("a", "m", "llama_server", score=0.0, status=400, error="exceeds the context"),  # context_overflow
    orow("a", "m", "llama_server", score=0.0, status=None, error="connection refused"),  # connection
    orow("a", "m", "ollama_default", score=0.0, status=500, error="out of memory"),      # other
])
def test_non_timeout_rows_are_not_relabelled(r):
    assert x2.analysis_outcome(r) is None
    assert x2.counts_toward_accuracy(r) is True


def test_analysis_outcome_is_pure():
    r = orow("a", "m", "llama_server", **TIMEOUT)
    before = copy.deepcopy(r)
    x2.analysis_outcome(r)
    x2.counts_toward_accuracy(r)
    assert r == before and "outcome" not in r


def test_context_tier_buckets():
    assert x2.context_tier("longdoc_qa", 32001) == "longdoc_32000"
    assert x2.context_tier("longdoc_qa", 80006) == "longdoc_80000"
    assert x2.context_tier("trace_length_mix", 31090) == "trace_mix_24000"
    assert x2.context_tier("gsm8k", 120) == "gsm8k_0"
    assert x2.context_tier("longdoc_qa", None) == "longdoc_unknown"


def test_timeout_constant_is_the_harness_default_and_behaviour_is_unchanged():
    assert x2.DEFAULT_CALL_TIMEOUT_S == 900
    assert inspect.signature(x2.run).parameters["call_timeout_s"].default == x2.DEFAULT_CALL_TIMEOUT_S
    assert x2.SCORER_VERSION == 2
    # still a valid, cached outcome row for the harness (not retried)
    assert x2.row_is_valid(orow("a", "m", "llama_server", **TIMEOUT))


# ------------------------------------------------------------------------------------------------ register rows
ITEMS = [{"item_id": "longdoc_32000_00", "family": "longdoc_qa", "prompt_tokens": 32001},
         {"item_id": "longdoc_32000_01", "family": "longdoc_qa", "prompt_tokens": 32010},
         {"item_id": "trace_mix_038", "family": "trace_length_mix", "prompt_tokens": 31090},
         {"item_id": "g1", "family": "gsm8k", "prompt_tokens": 80}]
WEIGHTS = {"longdoc_32000_00": 0.4, "longdoc_32000_01": 0.3, "trace_mix_038": 0.2, "g1": 0.1}


def _fake_cells(rows, plan_items):
    def fake(_repo):
        cells = {}
        for r in rows:
            cells.setdefault((r["model_id"], r["config"]), {})[r["item_id"]] = r
        plan_all = {it["item_id"]: it["family"] for it in ITEMS}
        plan = {m: set(plan_items) for m in x2.DEFAULT_MODELS}
        return x2, cells, plan, plan_all, WEIGHTS
    return fake


@pytest.fixture
def synthetic(monkeypatch):
    def install(rows, plan_items=None):
        monkeypatch.setattr(nr, "_x2_v3_cells", _fake_cells(rows, plan_items or [it["item_id"] for it in ITEMS]))
        monkeypatch.setattr(nr, "_x2_bare_cells", lambda repo, rows=None: set())
        monkeypatch.setattr(x2, "load_items_trace_weighted", lambda repo: list(ITEMS))
    return install


ROWS = [
    # qwen3-32b / llama_server: two 32K longdoc items and one trace-mix item time out, one scored row
    orow("longdoc_32000_00", "qwen3-32b", "llama_server", **TIMEOUT),
    orow("longdoc_32000_01", "qwen3-32b", "llama_server", **TIMEOUT),
    orow("trace_mix_038", "qwen3-32b", "llama_server", family="trace_length_mix", **TIMEOUT),
    orow("g1", "qwen3-32b", "llama_server", family="gsm8k", score=1.0),
    # qwen3-32b / ollama_default: completes everything
    orow("longdoc_32000_00", "qwen3-32b", "ollama_default", score=1.0),
    orow("longdoc_32000_01", "qwen3-32b", "ollama_default", score=0.0),
    # another model: one timeout, one context overflow (overflow still scores 0 in the denominator)
    orow("longdoc_32000_00", "qwen3-8b", "llama_server", **TIMEOUT),
    orow("longdoc_32000_01", "qwen3-8b", "llama_server", score=0.0, status=400, error="exceeds the context"),
    orow("g1", "qwen3-8b", "llama_server", family="gsm8k", score=1.0),
]


def test_scores_exclude_timeout_latency_from_every_denominator(synthetic):
    synthetic(ROWS)
    v = nr.compute_x2_v3_scores(Path("."))["value"]
    q32 = next(p for p in v.split(" / ") if p.startswith("qwen3-32b/llama_server"))
    assert "longdoc_qa" not in q32 and "trace_length_mix" not in q32  # every row of those families timed out
    assert "gsm8k 1.00 (n=1)" in q32 and "trace-weighted 1.00" in q32 and "3 timeout_latency excluded" in q32
    q8 = next(p for p in v.split(" / ") if p.startswith("qwen3-8b/llama_server"))
    # overflow stays in at 0: longdoc 0.00 (n=1); weights renormalized over longdoc_32000_01 and g1 only
    assert "longdoc_qa 0.00 (n=1)" in q8 and "1 timeout_latency excluded" in q8
    assert f"trace-weighted {0.1 / 0.4:.2f}" in q8
    qo = next(p for p in v.split(" / ") if p.startswith("qwen3-32b/ollama_default"))
    assert "excluded" not in qo and "longdoc_qa 0.50 (n=2)" in qo


def test_progress_and_error_causes_still_count_timeout_latency(synthetic):
    synthetic(ROWS)
    prog = nr.compute_x2_v3_progress(Path("."))["value"]
    assert "qwen3-32b/llama_server 4/4" in prog
    causes = nr.compute_x2_v3_error_causes(Path("."))["value"]
    assert "qwen3-32b/llama_server 1/0/3/0/0" in causes
    assert "qwen3-8b/llama_server 1/1/1/0/0" in causes


def test_usability_row(synthetic):
    synthetic(ROWS)
    out = nr.compute_x2_v3_usability(Path("."))
    v = out["value"]
    assert v.startswith(f"per-call timeout {x2.DEFAULT_CALL_TIMEOUT_S} s")
    assert "qwen3-32b/llama_server 3/4 (75.0%) [longdoc_32000 x2, trace_mix_24000 x1]" in v
    assert "qwen3-32b/ollama_default 0/2 (0.0%)" in v
    assert "qwen3-8b/llama_server 1/3 (33.3%) [longdoc_32000 x1]" in v
    assert out["n"] == 4


def test_qwen3_32b_timeout_count_provisional_then_final(synthetic):
    synthetic(ROWS)
    v = nr.compute_qwen3_32b_timeout_count(Path("."))
    assert v["value"].startswith("llama_server 3/4, ollama_default 0/2 (timeout_latency/done); provisional: "
                                 "outcome table v3 not finished")
    assert v["n"] == 3
    # every planned (model, config) cell done: the flag turns final by itself
    full = [orow(it["item_id"], m, c, family=it["family"],
                 **(TIMEOUT if (m, c, it["item_id"]) == ("qwen3-32b", "llama_server", "longdoc_32000_00") else {}))
            for m in x2.DEFAULT_MODELS for c in x2.CONFIGS for it in ITEMS]
    synthetic(full)
    v = nr.compute_qwen3_32b_timeout_count(Path("."))
    assert v["value"].startswith("llama_server 1/4, ollama_default 0/4 (timeout_latency/done); final: ")
    assert v["n"] == 1


def test_new_register_entries_exist():
    ids = {e["claim_id"]: e for e in nr.NUMBER_ENTRIES}
    assert ids["x2-v3-usability"]["compute"] is nr.compute_x2_v3_usability
    assert ids["qwen3-32b-timeout-count"]["compute"] is nr.compute_qwen3_32b_timeout_count


# ------------------------------------------------------------------------------------------------ dashboard fake
def test_dashboard_fake_quality_excludes_timeout_latency(tmp_path, monkeypatch):
    from demo.dashboard import fakes
    monkeypatch.setattr(fakes, "_template_sources", lambda: {})
    f = tmp_path / "x2_outcome_table_v3.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in [
        orow("a", "m", "llama_server", score=1.0),
        orow("b", "m", "llama_server", **TIMEOUT),
        orow("c", "m", "llama_server", score=0.0, status=400, error="exceeds the context"),
    ]), encoding="utf-8")
    pts = [p for p in fakes.load_points([("evo-x2", str(f))], cloud_source="none") if not p.stub]
    assert len(pts) == 1 and pts[0].quality == pytest.approx(0.5) and pts[0].quality_n == 2
