"""analysis/cloud_task_cost.py: cloud USD per real agent task, three caching cases, two price tiers, measured output
tokens per step and the Anthropic/qwen calibration. No network: the count_tokens client is faked."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import cloud_task_cost as ct  # noqa: E402

PRICES = {"models": {"m": {"min_cacheable_tokens": 100, "standard": {
    "short": {"input": 1.0, "cached_input": 0.1, "cache_write": 1.25, "output": 5.0},
    "long": {"input": 1.0, "cached_input": 0.1, "cache_write": 1.25, "output": 5.0}}}}}


def test_none_bills_every_resent_prompt_uncached_and_each_steps_output():
    usd = ct.task_cost(PRICES, "m", [1000, 1500, 2000], [10, 20, 30], "none")
    assert usd == pytest.approx(((1000 + 1500 + 2000) * 1.0 + 60 * 5.0) / 1e6)


def test_best_reads_resent_prefix_and_writes_only_new_tokens():
    usd = ct.task_cost(PRICES, "m", [1000, 1500], [0, 0], "best")
    assert usd == pytest.approx((1000 * 1.25 + 1000 * 0.1 + 500 * 1.25) / 1e6)


def test_realistic_hit_rate_splits_resent_prefix():
    usd = ct.task_cost(PRICES, "m", [1000, 1500], [0, 0], "realistic", hit_rate=0.7)
    assert usd == pytest.approx((1000 * 1.25 + 700 * 0.1 + (500 + 300) * 1.25) / 1e6)
    assert ct.task_cost(PRICES, "m", [1000, 1500], [0, 0], "realistic", hit_rate=1.0) == pytest.approx(
        ct.task_cost(PRICES, "m", [1000, 1500], [0, 0], "best"))


def test_prompt_under_min_cacheable_is_plain_input():
    assert ct.task_cost(PRICES, "m", [50, 80], [0, 0], "best") == pytest.approx(130 * 1.0 / 1e6)


def test_calibration_ratio_per_set_and_all():
    recs = [{"record": "count_tokens", "dataset": "SWE-Gym/OpenHands-Sampled-Trajectories", "qwen_tokens": 100,
             "anthropic_tokens": a} for a in (110, 120, 130)]
    recs.append({"record": "count_tokens", "dataset": "nebius/SWE-rebench-openhands-trajectories", "qwen_tokens": 200,
                 "anthropic_tokens": 300})
    c = ct.calibration(recs)
    assert c["SWE-Gym"]["median"] == pytest.approx(1.2) and c["SWE-Gym"]["n"] == 3
    assert c["nebius-rebench-openhands"]["median"] == pytest.approx(1.5) and c["all"]["n"] == 4


def test_summarize_applies_factor_to_input_and_output():
    trajs = {"SWE-Gym": [([1000, 1500], [10, 20])]}
    one = ct.summarize(trajs, {"models": {m: PRICES["models"]["m"] for m in ct.TIERS}}, {"SWE-Gym": 1.0})
    two = ct.summarize(trajs, {"models": {m: PRICES["models"]["m"] for m in ct.TIERS}}, {"SWE-Gym": 2.0})
    assert two["SWE-Gym"]["input_median"] == pytest.approx(2 * one["SWE-Gym"]["input_median"])
    assert two["SWE-Gym"]["output_median"] == pytest.approx(60)
    k = (ct.TIERS[0], "none")
    assert two["SWE-Gym"]["usd"][k]["median"] == pytest.approx(2 * one["SWE-Gym"]["usd"][k]["median"])


def _script():
    spec = importlib.util.spec_from_file_location("toc", REPO / "scripts" / "trace_output_calibration.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_count_phase_records_calls_without_network_and_caps(tmp_path, monkeypatch):
    m = _script()
    monkeypatch.setattr(m, "CALIB_FILE", tmp_path / "calib.jsonl")
    for i in range(3):
        (tmp_path / f"s{i}.json").write_text(json.dumps({"dataset": "SWE-Gym/OpenHands-Sampled-Trajectories",
                                                         "trajectory_id": f"t{i}", "qwen_tokens": 10, "text": "x" * i}))
    calls = []

    class Fake:
        @staticmethod
        def count_tokens(model_id, messages):
            calls.append(model_id)
            return 12

    m.count(tmp_path, client=Fake())
    m.count(tmp_path, client=Fake())  # already recorded: no new calls
    recs = [json.loads(l) for l in (tmp_path / "calib.jsonl").read_text().splitlines()]
    assert len(calls) == 3 and len(recs) == 3 and all(r["anthropic_tokens"] == 12 for r in recs)
    assert all("text" not in r for r in recs)
    monkeypatch.setattr(m, "MAX_CALLS", 2)
    with pytest.raises(SystemExit):
        m.count(tmp_path, client=Fake())


def test_capped_client_count_tokens_checks_mode_and_provider():
    sys.path.insert(0, str(REPO))
    from src.cloud import client as cc
    with pytest.raises(cc.MissingApiKeyError):
        cc.CloudClient(mode="stub").count_tokens(model_id="claude-sonnet-5-5", messages=[])
    real = cc.CloudClient(mode="real", api_key="test-not-a-key", role_override="controller")
    seen = []
    n = real.count_tokens(model_id="claude-sonnet-5-5", messages=[{"role": "user", "content": "hi"}],
                          count_fn=lambda model_id, messages: seen.append(model_id) or 7)
    assert n == 7 and seen == ["claude-sonnet-5-5"]
    with pytest.raises(cc.UnknownModelError):
        real.count_tokens(model_id="not-a-model", messages=[], count_fn=lambda **k: 1)


def test_sample_is_fixed_and_bounded():
    m = _script()
    ids = [f"t{i}" for i in range(36)]
    assert m.sample_ids(ids) == m.sample_ids(list(reversed(ids))) and len(m.sample_ids(ids)) == 20
    assert len(m.sample_ids(ids[:5])) == 5


def test_real_files_case_ordering():
    if not (REPO / ct.OUTPUT_FILE).exists() or not (REPO / ct.CALIB_FILE).exists():
        pytest.skip("measured output or calibration file not present")
    res = ct.compute(REPO)
    for e in res["sets"].values():
        for m in ct.TIERS:
            u = e["usd"]
            assert u[(m, "best")]["median"] <= u[(m, "realistic")]["median"] <= u[(m, "none")]["median"]
    txt = ct.value_text(res)
    assert "MEASURED" in txt and "ASSUMPTION" in txt and "count_tokens" in txt


def test_tiers_include_opus_and_fable_priced_from_the_shared_file():
    prices = ct.load_prices()
    assert ct.TIERS == ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1")
    assert ct.BASE_TIERS == ("claude-haiku-5-5", "claude-sonnet-5-5")
    for m in ct.TIERS:
        r = prices["models"][m]["standard"]["short"]
        assert all(r[k] > 0 for k in ("input", "cache_write", "cached_input", "output"))
    assert prices["source_url"] == "https://platform.claude.com/docs/en/about-claude/pricing"


def test_all_tiers_real_files_sonnet_reproduces_existing_row():
    if not (REPO / ct.OUTPUT_FILE).exists() or not (REPO / ct.CALIB_FILE).exists():
        pytest.skip("measured output or calibration file not present")
    res = ct.compute(REPO)
    u = res["sets"]["nebius-rebench-openhands"]["usd"]
    assert f"{u[('claude-sonnet-5-5', 'realistic')]['median']:.4f}" == "2.5307"
    for m in ct.TIERS:
        assert u[(m, "best")]["median"] <= u[(m, "realistic")]["median"] <= u[(m, "none")]["median"]
    base, full = ct.value_text(res), ct.all_tiers_text(res)
    assert "claude-opus-5-5" not in base and "claude-fable-5-1" not in base  # existing row's text unchanged
    assert "claude-opus-5-5:" in full and "claude-fable-5-1:" in full and "realistic $2.5307" in full
