"""Silent truncation vs latency (operator 2026-10-10): register rows qwen3-32b-truncation-vs-latency and
x2-v3-truncation-all-models. All rows here are synthetic."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

import x2_outcome_table as x2  # noqa: E402
import analysis.numbers_register as nr  # noqa: E402


def orow(item, model, config, score=1.0, status=200, error=None, sent=None, processed=None, latency=10.0, **kw):
    r = {"record": "outcome_row", "item_id": item, "family": "longdoc_qa", "model_id": model, "config": config,
         "score": score, "http_status": status, "error": error, "latency_s": latency}
    if sent is not None:
        r.update({"sent_tokens": sent, "processed_tokens": processed,
                  "silently_truncated": bool(processed is not None and processed < sent)})
    r.update(kw)
    return r


TIMEOUT = dict(score=0.0, status=None, error="timed out", latency=None)
OVERFLOW = dict(score=0.0, status=None, latency=None,
                error="server guard mismatch: server does not match intended config n_ctx 40960 != 48640")

ITEMS = [{"item_id": "s1", "family": "longdoc_qa", "prompt_tokens": 8000},
         {"item_id": "s2", "family": "longdoc_qa", "prompt_tokens": 16000},
         {"item_id": "m1", "family": "longdoc_qa", "prompt_tokens": 32000},
         {"item_id": "l1", "family": "longdoc_qa", "prompt_tokens": 48000},
         {"item_id": "l2", "family": "longdoc_qa", "prompt_tokens": 64000}]


# ------------------------------------------------------------------------------------------------ pure functions
@pytest.mark.parametrize("sent,processed,expected", [
    (8000, 8016, "full"),          # template overhead: processed above sent
    (8000, 7969, "full"),          # 31 short: inside max(32, 80) tolerance
    (8000, 7920, "full"),          # exactly at the 1% edge (80): not truncated
    (8000, 7919, "truncated"),     # 81 short
    (1000, 967, "truncated"),      # small prompt: the 32-token floor applies (1% would be 10); 33 short
    (1000, 968, "full"),           # exactly 32 short: at the edge, not truncated
    (1000, 975, "full"),           # 25 short, inside the 32 floor although above 1%
    (48000, 20482, "truncated"),   # Ollama keep-half
])
def test_truncation_status_tolerance(sent, processed, expected):
    assert nr.truncation_status(orow("a", "m", "ollama_default", sent=sent, processed=processed), x2=x2) == expected


def test_truncation_status_errors_and_unmeasured():
    assert nr.truncation_status(orow("a", "m", "llama_server", **TIMEOUT), x2=x2) == "timeout"
    assert nr.truncation_status(orow("a", "m", "llama_server", **OVERFLOW), x2=x2) == "context_overflow"
    assert nr.truncation_status(orow("a", "m", "ollama_default", status=404, error="model not found", sent=10),
                                x2=x2) == "other"
    assert nr.truncation_status(orow("a", "m", "ollama_default"), x2=x2) == "unmeasured"
    # sent falls back to the pack's prompt_tokens when the row has none
    r = orow("a", "m", "ollama_default", processed_tokens=20482)
    assert nr.truncation_status(r, sent=48000, x2=x2) == "truncated"


def test_wilson_and_score_ci():
    lo, hi = nr.wilson_ci(6, 15)
    assert lo == pytest.approx(0.198, abs=0.002) and hi == pytest.approx(0.643, abs=0.002)
    assert nr.wilson_ci(0, 0) == (0.0, 0.0)
    assert nr._score_ci([1.0, 0.0, 1.0])[3] == "Wilson"
    m, lo, hi, method = nr._score_ci([0.5, 1.0, 0.25, 0.75])
    assert method == "bootstrap" and lo <= m <= hi and m == pytest.approx(0.625)
    assert nr._score_ci([]) == (None, None, None, "n/a")


# ------------------------------------------------------------------------------------------------ register rows
ROWS = [
    # llama_server: serves s1, s2 in full; times out on m1; refuses l1, l2 (n_ctx_train clamp)
    orow("s1", "qwen3-32b", "llama_server", sent=8000, processed=8012, latency=60.0),
    orow("s2", "qwen3-32b", "llama_server", sent=16000, processed=16012, latency=200.0),
    orow("m1", "qwen3-32b", "llama_server", **TIMEOUT),
    orow("l1", "qwen3-32b", "llama_server", **OVERFLOW),
    orow("l2", "qwen3-32b", "llama_server", **OVERFLOW),
    # ollama_default: everything HTTP 200; l1 and l2 silently cut to 20482, one wrong
    orow("s1", "qwen3-32b", "ollama_default", sent=8000, processed=8016, latency=30.0),
    orow("s2", "qwen3-32b", "ollama_default", sent=16000, processed=16016, latency=100.0),
    orow("m1", "qwen3-32b", "ollama_default", sent=32000, processed=32016, latency=300.0),
    orow("l1", "qwen3-32b", "ollama_default", sent=48000, processed=20482, latency=140.0, score=1.0),
    orow("l2", "qwen3-32b", "ollama_default", sent=64000, processed=20482, latency=150.0, score=0.0),
    # another model, nothing truncated
    orow("s1", "qwen3-8b", "ollama_default", sent=8000, processed=8016),
]


@pytest.fixture
def synthetic(monkeypatch):
    def install(rows, plan_items=None):
        plan_items = plan_items or [it["item_id"] for it in ITEMS]

        def fake(_repo):
            cells = {}
            for r in rows:
                cells.setdefault((r["model_id"], r["config"]), {})[r["item_id"]] = r
            plan_all = {it["item_id"]: it["family"] for it in ITEMS}
            return x2, cells, {m: set(plan_items) for m in x2.DEFAULT_MODELS}, plan_all, {}
        monkeypatch.setattr(nr, "_x2_v3_cells", fake)
        monkeypatch.setattr(x2, "load_items_trace_weighted", lambda repo: list(ITEMS))
    return install


def test_qwen3_32b_truncation_vs_latency(synthetic):
    synthetic(ROWS)
    out = nr.compute_qwen3_32b_truncation_vs_latency(Path("."))
    v = out["value"]
    assert v.startswith(nr.TRUNC_DEFINITION)
    assert "ollama_default: silently truncated 2/5 HTTP-200 rows (40.0%) (evaluated tokens 20482)" in v
    lo, hi = nr.wilson_ci(1, 2)
    assert f"truncated score 0.50 (1/2, Wilson 95% CI {lo:.2f}-{hi:.2f}), latency median 145 s (range 140-150)" in v
    assert "full score 1.00 (3/3, Wilson" in v and "latency median 100 s (range 30-300)" in v
    assert "llama_server: silently truncated 0/2 HTTP-200 rows (0.0%); truncated n=0" in v
    assert "context_overflow 2, timeout 1" in v
    assert "harness silently_truncated agrees on 5/5" in v
    assert "llama_server context_overflow items on ollama_default: truncated 2 (of 2)" in v
    assert "llama_server timeout_latency items on ollama_default: full 1 (of 1)" in v
    # s1 30/60 = 0.5, s2 100/200 = 0.5
    assert "ollama_default/llama_server latency ratio median 0.50 (range 0.50-0.50, n=2)" in v
    assert "provisional: outcome table v3 not finished" in v
    assert out["n"] == 10


def test_timeout_rows_stay_out_of_accuracy_groups(synthetic):
    synthetic(ROWS)
    per = nr.compute_qwen3_32b_truncation_vs_latency(Path("."))["detail"]["per"]
    ls = per["llama_server"]
    assert ls["groups"]["full"]["n"] == 2 and ls["groups"]["truncated"]["n"] == 0
    assert ls["counts"] == {"full": 2, "timeout": 1, "context_overflow": 2}


def test_harness_flag_disagreement_is_counted(synthetic):
    # strict harness flag (processed < sent) fires on a 10-token shortfall; the tolerance does not
    rows = [orow("s1", "qwen3-32b", "ollama_default", sent=8000, processed=7990)]
    synthetic(rows, plan_items=["s1"])
    v = nr.compute_qwen3_32b_truncation_vs_latency(Path("."))["value"]
    assert "silently truncated 0/1" in v and "harness silently_truncated agrees on 0/1" in v


def test_truncation_flag_turns_final(synthetic):
    full = [orow(it["item_id"], m, c, sent=it["prompt_tokens"], processed=it["prompt_tokens"] + 12)
            for m in x2.DEFAULT_MODELS for c in x2.CONFIGS for it in ITEMS]
    synthetic(full)
    assert "; final: outcome table v3 finished" in nr.compute_qwen3_32b_truncation_vs_latency(Path("."))["value"]
    assert nr.compute_x2_v3_truncation_all_models(Path("."))["value"].endswith(
        f"final ({len(full)}/{len(full)} done)")


def test_all_models_row(synthetic):
    synthetic(ROWS)
    v = nr.compute_x2_v3_truncation_all_models(Path("."))["value"]
    assert "qwen3-32b/ollama_default 2/5, score truncated 0.50 (n=2) vs full 1.00 (n=3)" in v
    assert "qwen3-32b/llama_server 0/2, score full 1.00 (n=2)" in v
    assert "qwen3-8b/ollama_default 0/1, score full 1.00 (n=1)" in v


def test_missing_model_raises(synthetic):
    synthetic([ROWS[-1]])
    with pytest.raises(FileNotFoundError):
        nr.compute_qwen3_32b_truncation_vs_latency(Path("."))


def test_register_entries_exist():
    ids = {e["claim_id"]: e for e in nr.NUMBER_ENTRIES}
    assert ids["qwen3-32b-truncation-vs-latency"]["compute"] is nr.compute_qwen3_32b_truncation_vs_latency
    assert ids["x2-v3-truncation-all-models"]["compute"] is nr.compute_x2_v3_truncation_all_models
