"""analysis/r2_session_cost: wall-clock, token split and cloud cost range per R2 session, on synthetic rows."""
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))

import r2_session_cost as sc  # noqa: E402

OPENAI_FILE = "results/cloud_prices_20261008.json"


def _turn(model, arm, seed, i, calls, wall, cum):
    return {"record": "r2a_turn", "model_id": model, "arm_id": arm, "seed": seed, "turn_idx": i,
            "turn_wall_s": wall, "session_tokens_billed_cumulative": cum,
            "calls": [{"prompt_tokens_est": p, "completion_tokens": c, "content": "x" * 40,
                       "prompt_eval_duration_s": 1.0, "eval_duration_s": 0.5} for p, c in calls]}


def _session(model, arm, seed, turns):
    rows, cum = [], 0
    for i, (calls, wall) in enumerate(turns, start=1):
        cum += sum(p + (c if c is not None else 10) for p, c in calls)
        rows.append(_turn(model, arm, seed, i, calls, wall, cum))
    return rows


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "results").mkdir()
    shutil.copy(REPO / OPENAI_FILE, tmp_path / OPENAI_FILE)
    a = "ollama_default_call2_notools"
    v1 = ([{"record": "run_start"}]
          + _session("m", a, 1, [([(1000, 20), (1100, 30)], 60.0), ([(2000, 10), (2100, None)], 120.0)])
          + _session("m", a, 2, [([(1000, 20)], 30.0), ([(3000, 40)], 30.0)]))
    v1b = _session("m", a, 3, [([(500, 5)], 600.0)]) + _session("m", a, 1, [([(9, 9)], 1.0)])  # dup key
    _write(tmp_path / sc.V1_FILE, v1)
    _write(tmp_path / sc.V1B_FILE, v1b)
    return tmp_path


def test_session_metrics_split_billed_into_input_and_output(repo):
    ss = sc.sessions(sc.read_rows(repo / sc.V1_FILE))
    s = ss[("m", "ollama_default", 1)]
    assert s["wall_min"] == pytest.approx(3.0)
    # completion_tokens None falls back to chars/4 of the content (40 chars -> 10), as the harness counter does
    assert s["output"] == 20 + 30 + 10 + 10
    assert s["billed"] == 1000 + 1100 + 2000 + 2100 + 70
    assert s["input"] == s["input_sum_check"] == 6200
    assert s["final_prompt"] == 2100
    assert s["prefill_s"] == pytest.approx(4.0) and s["decode_s"] == pytest.approx(2.0)


def test_cost_range_no_cache_and_best_cache(repo):
    prices = sc.load_prices(repo)
    assert list(prices) == ["openai"] and prices["openai"]["models"] == ("gpt-6.1-sol", "gpt-6-luna")
    s = sc.sessions(sc.read_rows(repo / sc.V1_FILE))[("m", "ollama_default", 1)]
    raw = json.loads((repo / OPENAI_FILE).read_text(encoding="utf-8"))
    r = raw["models"]["gpt-6.1-sol"]["standard"]["short"]
    c = sc.session_cost(prices["openai"]["rates"]["gpt-6.1-sol"], s)
    assert c["no_cache"] == pytest.approx((6200 * r["input"] + 70 * r["output"]) / 1e6)
    assert c["best_cache"] == pytest.approx((2100 * r["input"] + 4100 * r["cached_input"] + 70 * r["output"]) / 1e6)
    assert c["best_cache"] < c["no_cache"]


def test_build_per_file_combined_and_duplicates(repo):
    rep = sc.build(repo)
    v1 = rep["per_file"][sc.V1_FILE][("m", "ollama_default")]
    assert v1["n"] == 2 and v1["mean"]["wall_min"] == pytest.approx((3.0 + 1.0) / 2)
    comb = rep["combined"][("m", "ollama_default")]
    assert comb["n"] == 3 and set(comb["seeds"]) == {1, 2, 3}
    assert comb["seeds"][1]["wall_min"] == pytest.approx(3.0)   # the v1 session wins over the v1b duplicate
    assert rep["duplicates"] == [(sc.V1B_FILE, "m", "ollama_default", 1)]
    w, c = sc.wallclock_str(rep), sc.cost_str(rep)
    assert "[x2_r2_real_v1] m ollama_default: mean 2.0 min (3.0, 1.0; n 2)" in w
    assert "[combined v1+v1b]" in w and "gpt-6-luna USD" in c and "gpt-6.1-sol USD" in c
    assert "openai: results/cloud_prices_20261008.json" in c


def _anthropic_table(fetched="2026-10-14T00:00Z", strong_in=3.0):
    return {"provider": "anthropic", "source_url": "https://claude.com/pricing", "fetched_utc": fetched,
            "selection": {"cheap_small": "claude-cheap", "mid_tier": "claude-mid"},
            "models": {"claude-cheap": {"standard": {"input": 1.0, "cache_read": 0.1, "cache_write_5m": 1.25,
                                                     "output": 5.0}},
                       "claude-mid": {"standard": {"input": strong_in, "cache_read": 0.3, "output": 15.0}}}}


def test_anthropic_price_file_is_picked_up_and_both_providers_reported(repo):
    old = _anthropic_table(fetched="old", strong_in=99.0)
    (repo / "results/cloud_prices_anthropic_20261001.json").write_text(json.dumps(old), encoding="utf-8")
    (repo / "results/cloud_prices_anthropic_20261014.json").write_text(json.dumps(_anthropic_table()),
                                                                      encoding="utf-8")
    prices = sc.load_prices(repo)
    assert list(prices) == ["anthropic", "openai"]
    assert prices["anthropic"]["path"] == "results/cloud_prices_anthropic_20261014.json"   # latest file wins
    assert prices["anthropic"]["models"] == ("claude-mid", "claude-cheap")
    s = sc.sessions(sc.read_rows(repo / sc.V1_FILE))[("m", "ollama_default", 1)]
    c = sc.session_cost(prices["anthropic"]["rates"]["claude-mid"], s)
    assert c["no_cache"] == pytest.approx((6200 * 3.0 + 70 * 15.0) / 1e6)
    assert c["best_cache"] == pytest.approx((2100 * 3.0 + 4100 * 0.3 + 70 * 15.0) / 1e6)
    rep = sc.build(repo)
    cs = sc.cost_str(rep)
    assert "claude-mid USD" in cs and "gpt-6.1-sol USD" in cs and "anthropic: results/cloud_prices_anthropic_20261014" in cs
    assert set(rep["combined"][("m", "ollama_default")]["mean"]["cost"]) == {
        "claude-mid", "claude-cheap", "gpt-6.1-sol", "gpt-6-luna"}


def test_no_price_file_is_an_error(tmp_path):
    (tmp_path / "results").mkdir()
    with pytest.raises(FileNotFoundError):
        sc.load_prices(tmp_path)


def test_expectation_check_reports_mismatch_without_adjusting(repo, monkeypatch):
    monkeypatch.setattr(sc, "EXPECTED", (("m default min", "m", "ollama_default", "wall_min", 2, 0),
                                         ("m default min wrong", "m", "ollama_default", "wall_min", 5, 0),
                                         ("unpriced USD", "m", "ollama_default", ("no-such-model", "no_cache"), 1, 2)))
    rep = sc.build(repo)
    out = sc.expectation_check(rep)
    assert out[0]["v1"]["match"] is True and out[0]["v1"]["mean"] == pytest.approx(2.0)
    assert out[1]["v1"]["match"] is False and out[1]["v1"]["mean"] == pytest.approx(2.0)
    assert out[2]["v1"] is None
    assert "MISMATCH" in sc.expectation_str(rep, "wallclock") and "unpriced" not in sc.expectation_str(rep, "wallclock")
    assert sc.expectation_str(rep, "cost").startswith("unpriced USD")
