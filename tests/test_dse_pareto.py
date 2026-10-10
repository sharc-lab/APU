"""src/dse/pareto.py on synthetic outcome rows: quality weighting, latency percentiles, validity filter, bare-template
flagging, STUB cloud labelling, frontier, recommendation, and re-running unchanged with an added machine file and
with real cloud rows / a ledger replacing the stub."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dse import pareto as P

PRICES = {"models": {"cheap-m": {"standard": {"short": {"input": 1.0, "cached_input": 0.1, "cache_write": 1.25,
                                                          "output": 2.0},
                                                "long": {"input": 2.0, "cached_input": 0.2, "cache_write": 2.5,
                                                         "output": 4.0}}},
                     "mid-m": {"standard": {"short": {"input": 10.0, "cached_input": 1.0, "cache_write": 12.5,
                                                        "output": 20.0},
                                              "long": {"input": 20.0, "cached_input": 2.0, "cache_write": 25.0,
                                                       "output": 40.0}}}}}
WEIGHTS = {"a": 0.5, "b": 0.3, "c": 0.2}
TOKENS = {"a": (1000, 100.0), "b": (2000, 100.0), "c": (4000, 200.0)}
STEPS = ["a", "b", "c"]


def row(item, model, config, score, latency_s=10.0, status=200, **kw):
    r = {"record": "outcome_row", "item_id": item, "family": "fam", "model_id": model, "config": config,
         "score": score, "latency_s": latency_s, "http_status": status, "error": None}
    r.update(kw)
    return r


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def load(files, cloud_source="stub", tmp=None, **kw):
    import cloud_budget_forecast as cbf
    orig = (cbf.CHEAP, cbf.MID)
    cbf.CHEAP, cbf.MID = "cheap-m", "mid-m"  # stub models priced in PRICES
    try:
        return P.load_points(files, cloud_source=cloud_source, repo=tmp, weights=WEIGHTS, prices=PRICES,
                             stub_token_counts=TOKENS, step_items=STEPS, template_facts={}, **kw)
    finally:
        cbf.CHEAP, cbf.MID = orig


def by_id(points):
    return {p.config_id: p for p in points}


@pytest.fixture
def x2_file(tmp_path):
    return write(tmp_path / "results" / "x2.jsonl", [
        {"record": "heartbeat"},
        # model-a / llama_server: a=1, b=0 (overwritten later by 1), c=0 timeout
        row("a", "model-a", "llama_server", 1.0, 10.0, chat_template_source="gguf_embedded_llama_server"),
        row("b", "model-a", "llama_server", 0.0, 20.0, chat_template_source="gguf_embedded_llama_server"),
        row("b", "model-a", "llama_server", 1.0, 30.0, chat_template_source="gguf_embedded_llama_server"),
        row("c", "model-a", "llama_server", 0.0, 900.0, status=None, error="call timed out",
            chat_template_source="gguf_embedded_llama_server"),
        # an infra failure (connection) is not a valid row and must not count
        row("a", "model-b", "ollama_default", 0.0, 1.0, status=None, error="connection refused",
            chat_template_source="ollama_library"),
        row("b", "model-b", "ollama_default", 1.0, 5.0, chat_template_source="ollama_library"),
        row("c", "model-b", "ollama_default", 1.0, 7.0, chat_template_source="ollama_library"),
        # bare template cell: perfect and fast, but an install-path result
        row("a", "model-c", "ollama_default", 1.0, 1.0, chat_template_source="bare_gguf_ollama_create"),
        row("b", "model-c", "ollama_default", 1.0, 1.0, chat_template_source="bare_gguf_ollama_create"),
        # tagged invalid row is ignored
        row("c", "model-c", "ollama_default", 0.0, 1.0, invalid_race=True,
            chat_template_source="bare_gguf_ollama_create"),
    ])


def test_quality_latency_and_validity(x2_file, tmp_path):
    pts = by_id(load([("evo-x2", str(x2_file))], tmp=tmp_path))
    a = pts["evo-x2:llama_server:model-a"]
    # the timed-out item c is timeout_latency: out of the quality denominator, still in n_rows and error_causes
    assert a.quality_n == 2 and a.n_rows == 3
    assert a.quality == pytest.approx((0.5 * 1 + 0.3 * 1) / 0.8)
    assert a.weight_coverage == pytest.approx(0.8)
    assert a.latency_p50_ms == pytest.approx(20_000)  # completed rows only: 10 s and 30 s
    assert a.latency_p90_ms == pytest.approx(28_000)
    assert "errors:timeout=1" in a.flags and a.error_causes == {"none": 2, "timeout": 1}
    assert "timeout_latency_excluded=1" in a.flags
    assert a.usd_per_1k_steps == 0.0 and not a.stub
    b = pts["evo-x2:ollama_default:model-b"]
    assert b.quality_n == 2 and b.quality == pytest.approx(1.0)  # connection row dropped, weights renormalized
    assert b.weight_coverage == pytest.approx(0.5)
    c = pts["evo-x2:ollama_default:model-c"]
    assert c.quality_n == 2 and P.BARE_FLAG in c.flags and "[bare template]" in c.label()
    assert c.template_source == "bare_gguf_ollama_create"


def test_stub_cloud_points_are_labelled_and_priced(x2_file, tmp_path):
    pts = load([("evo-x2", str(x2_file))], tmp=tmp_path)
    stubs = [p for p in pts if p.stub]
    assert {p.model for p in stubs} == {"cheap-m", "mid-m"}
    for p in stubs:
        assert p.machine == P.CLOUD_MACHINE and p.quality is None and p.latency_p50_ms is None
        assert P.STUB in p.flags and P.STUB in p.config_id and P.STUB in p.label() and p.to_dict()["stub"]
    cheap = next(p for p in stubs if p.model == "cheap-m")
    per = {i: (TOKENS[i][0] * 1.0 + TOKENS[i][1] * 2.0) / 1e6 for i in STEPS}
    assert cheap.usd_per_1k_steps == pytest.approx(sum(WEIGHTS[i] * per[i] for i in STEPS) * 1000)
    assert all(not p.stub for p in P.frontier(pts))  # no measured quality: never on a frontier


def test_frontier_excludes_bare_and_respects_cost(x2_file, tmp_path):
    pts = load([("evo-x2", str(x2_file))], tmp=tmp_path)
    f = P.frontier(pts, "evo-x2")
    assert [p.config_id for p in f] == ["evo-x2:ollama_default:model-b"]
    f2 = P.frontier(pts, "evo-x2", include_flagged=True)
    assert f2[0].config_id == "evo-x2:ollama_default:model-c"  # same cost and quality, lower latency
    # with an amortized hardware cost the slower cell costs more, the frontier is still the dominant one
    pts_hw = load([("evo-x2", str(x2_file))], tmp=tmp_path, hw_usd_per_hour={"evo-x2": 3.6})
    b = by_id(pts_hw)["evo-x2:ollama_default:model-b"]
    assert b.usd_per_1k_steps == pytest.approx(3.6 * 6.0 * 1000 / 3600) and "amortized_hw_cost" in b.flags


def test_recommend_cheapest_meeting_floor_and_latency(x2_file, tmp_path):
    pts = load([("evo-x2", str(x2_file))], tmp=tmp_path)
    rec = P.recommend(pts, budget_usd=50, quality_floor=0.9, latency_target_ms=30_000, hardware=["evo-x2"])
    assert rec.chosen.config_id == "evo-x2:ollama_default:model-b"
    cheap = next(p for p in pts if p.stub and p.model == "cheap-m")
    assert rec.all_cloud_reference is cheap and rec.stub and P.STUB in rec.reason
    assert rec.savings_vs_all_cloud_usd_per_1k == pytest.approx(cheap.usd_per_1k_steps)
    assert "\n" not in rec.reason
    none = P.recommend(pts, budget_usd=50, quality_floor=0.9, latency_target_ms=1_000, hardware=["evo-x2"])
    assert none.chosen is None and none.savings_vs_all_cloud_usd_per_1k is None and "rejected" in none.reason
    other = P.recommend(pts, budget_usd=50, quality_floor=0.9, latency_target_ms=30_000, hardware=["evo-t2s"])
    assert other.chosen is None  # hardware filter: evo-x2 cells are not candidates


def test_machine_matching():
    assert P.machine_matches("evo-x2", "evox2_strix_halo_128gb")
    assert P.machine_matches("EVO-X2", "evo-x2")
    assert not P.machine_matches("evo-x2", "evo-t2s")


def test_rerun_with_an_added_machine_file(x2_file, tmp_path):
    t2s = write(tmp_path / "results" / "t2s.jsonl", [
        row("a", "model-a", "ollama_igpu_enable", 1.0, 4.0),
        row("b", "model-a", "ollama_igpu_enable", 1.0, 4.0),
        row("c", "model-a", "ollama_igpu_enable", 1.0, 4.0),
        row("a", "model-z", "llama_server_vulkan", 0.5, 2.0),
    ])
    one = load([("evo-x2", str(x2_file))], tmp=tmp_path)
    two = load([("evo-x2", str(x2_file)), ("evo-t2s", str(t2s))], tmp=tmp_path)
    assert {p.machine for p in two} - {p.machine for p in one} == {"evo-t2s"}
    assert len([p for p in two if p.machine == "evo-t2s"]) == 2
    t = by_id(two)["evo-t2s:ollama_igpu_enable:model-a"]
    assert t.template_source is None and "template_unknown" in t.flags
    assert [p.config_id for p in P.frontier(two, "evo-t2s")] == ["evo-t2s:ollama_igpu_enable:model-a"]
    rec = P.recommend(two, 50, 0.9, 30_000, ["evo-x2", "evo-t2s"])
    assert rec.chosen.config_id == "evo-t2s:ollama_igpu_enable:model-a"  # same cost, faster
    # a second file for the same machine supersedes the first per (item, model, config)
    newer = write(tmp_path / "results" / "x2_new.jsonl", [row("a", "model-a", "llama_server", 0.0, 10.0)])
    p3 = by_id(load([("evo-x2", str(x2_file)), ("evo-x2", str(newer))], tmp=tmp_path))
    # a=0 (superseded), b=1, c timeout_latency (excluded): weights renormalized over a and b
    assert p3["evo-x2:llama_server:model-a"].quality == pytest.approx(0.3 / 0.8)


def test_real_cloud_rows_replace_the_stub(x2_file, tmp_path):
    cloud = write(tmp_path / "results" / "cloud_outcome.jsonl", [
        row("a", "cheap-m", "api", 1.0, 2.0, cost_usd=0.002),
        row("b", "cheap-m", "api", 1.0, 2.0, cost_usd=0.004),
        row("c", "cheap-m", "api", 0.0, 2.0, input_tokens=1000, output_tokens=1000),
    ])
    pts = load([("evo-x2", str(x2_file)), (P.CLOUD_MACHINE, str(cloud))], tmp=tmp_path)
    cheap = [p for p in pts if p.model == "cheap-m"]
    assert len(cheap) == 1 and not cheap[0].stub and P.STUB not in cheap[0].label()
    assert cheap[0].quality == pytest.approx(0.8)
    c_cost = (1000 * 1.0 + 1000 * 2.0) / 1e6
    assert cheap[0].usd_per_1k_steps == pytest.approx((0.5 * 0.002 + 0.3 * 0.004 + 0.2 * c_cost) * 1000)
    assert any(p.stub and p.model == "mid-m" for p in pts)  # no real rows for mid-m yet: still a STUB point
    rec = P.recommend(pts, 50, 0.9, 30_000, [])
    assert rec.chosen is None  # cloud only: cheap-m is below the floor, mid-m unmeasured
    rec2 = P.recommend(pts, 50, 0.7, 30_000, ["evo-x2"])
    assert rec2.all_cloud_reference.model == "cheap-m" and not rec2.all_cloud_reference.stub


def test_ledger_cloud_source(x2_file, tmp_path):
    ledger = write(tmp_path / "results" / "cloud_ledger.jsonl", [
        {"timestamp": "t", "model_id": "gpt-x", "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01,
         "running_total_usd": 0.01},
        {"timestamp": "t", "model_id": "gpt-x", "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.03,
         "running_total_usd": 0.04},
        {"record": "alert", "source": "cloud_client"},
    ])
    pts = load([("evo-x2", str(x2_file))], cloud_source=str(ledger), tmp=tmp_path)
    cloud = [p for p in pts if p.is_cloud]
    assert len(cloud) == 1 and not cloud[0].stub and cloud[0].usd_per_1k_steps == pytest.approx(20.0)
    # "auto" picks up the default ledger path when it has call rows, otherwise falls back to the stub
    auto = load([("evo-x2", str(x2_file))], cloud_source="auto", tmp=tmp_path)
    assert [p.model for p in auto if p.is_cloud] == ["gpt-x"]
    ledger.unlink()
    auto2 = load([("evo-x2", str(x2_file))], cloud_source="auto", tmp=tmp_path)
    assert all(p.stub for p in auto2 if p.is_cloud) and any(p.is_cloud for p in auto2)


def test_template_derived_from_matching_host_facts(tmp_path):
    import x2_outcome_table as x2
    tag, gguf = x2.MODEL_MAP["qwen3-4b-2507"]
    f = write(tmp_path / "results" / "x2.jsonl", [
        row("a", "qwen3-4b-2507", "ollama_default", 1.0, 1.0, ollama_tag=tag),
        row("a", "qwen3-4b-2507", "llama_server", 1.0, 1.0)])
    facts = {"evox2": {("ollama", tag): {"chat_template_source": "bare_gguf_ollama_create",
                                         "chat_template_sha256": "x"}}}
    import cloud_budget_forecast  # noqa: F401
    pts = by_id(P.load_points([("evo-x2", str(f))], cloud_source="none", repo=tmp_path, weights=WEIGHTS,
                              prices=PRICES, template_facts=facts))
    assert P.BARE_FLAG in pts["evo-x2:ollama_default:qwen3-4b-2507"].flags
    assert pts["evo-x2:llama_server:qwen3-4b-2507"].template_source == "gguf_embedded_llama_server"
    other = by_id(P.load_points([("evo-t2s", str(f))], cloud_source="none", repo=tmp_path, weights=WEIGHTS,
                                prices=PRICES, template_facts=facts))
    assert other["evo-t2s:ollama_default:qwen3-4b-2507"].template_source is None  # facts are per host


def test_points_table_is_json_serializable(x2_file, tmp_path):
    pts = load([("evo-x2", str(x2_file))], tmp=tmp_path)
    t = P.points_table(pts)
    json.dumps(t)
    assert t["frontier_by_machine"]["evo-x2"] == ["evo-x2:ollama_default:model-b"]
    assert set(t["stub_points"]) == {p.config_id for p in pts if p.stub}
    assert any("STUB" in line for line in P.format_table(pts))


def test_percentile_matches_linear_interpolation():
    assert P.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert P.percentile([10], 90) == 10
    assert P.percentile([], 50) is None


# ── usability constraint: timeout_latency share (operator 2026-10-10) ──

def _pt(cid, share, n_rows=20, quality=1.0, p50=10_000.0, machine="evo-x2"):
    n_tl = round(share * n_rows) if share is not None else 0
    return P.Point(machine=machine, runtime=cid, model="m", config_id=f"{machine}:{cid}:m", quality=quality,
                   quality_n=n_rows - n_tl, usd_per_1k_steps=0.0, latency_p50_ms=p50, latency_p90_ms=p50,
                   stub=False, template_source=None, n_rows=n_rows, n_timeout_latency=n_tl,
                   timeout_latency_share=share)


def test_timeout_latency_share_is_on_the_point(x2_file, tmp_path):
    pts = by_id(load([("evo-x2", str(x2_file))], tmp=tmp_path))
    a = pts["evo-x2:llama_server:model-a"]
    assert a.n_timeout_latency == 1 and a.timeout_latency_share == pytest.approx(1 / 3)
    assert pts["evo-x2:ollama_default:model-b"].timeout_latency_share == 0.0
    assert all(p.timeout_latency_share is None for p in pts.values() if p.stub)  # no rows, no share
    assert "timeout_latency_share" in a.to_dict()


def test_default_max_timeout_share_is_0_05():
    import inspect
    assert P.DEFAULT_MAX_TIMEOUT_SHARE == 0.05
    assert inspect.signature(P.recommend).parameters["max_timeout_share"].default == 0.05
    assert inspect.signature(P.points_table).parameters["max_timeout_share"].default == 0.05
    # without the argument recommend() applies 0.05: a 0.10 config is excluded even though it is the fastest
    rec = P.recommend([_pt("fast", 0.10, p50=1_000.0), _pt("slow", 0.0)], 50, 0.9, 30_000, ["evo-x2"])
    assert rec.chosen.runtime == "slow" and [e["config_id"] for e in rec.excluded] == ["evo-x2:fast:m"]
    assert rec.inputs["max_timeout_share"] == 0.05


def test_share_exactly_at_the_limit_is_eligible():
    at = _pt("at", 0.05)  # 1 of 20 done rows
    assert at.timeout_latency_share == 0.05 and P.usability_exclusion(at, 0.05) is None
    above = _pt("above", 0.10)
    assert P.usability_exclusion(above, 0.05) is not None
    rec = P.recommend([at, above], 50, 0.9, 30_000, ["evo-x2"], max_timeout_share=0.05)
    assert rec.chosen is at and [e["config_id"] for e in rec.excluded] == [above.config_id]
    # the same share computed from rows (1 timeout in 20) is exactly the boundary too
    assert P.usability_exclusion(_pt("r", 1 / 20), 0.05) is None
    assert P.frontier([at, above], "evo-x2", max_timeout_share=0.05) == [at]


def test_exclusion_reason_is_returned_not_silently_dropped():
    bad = _pt("bad", 0.15)  # 3 of 20
    rec = P.recommend([bad], 50, 0.9, 30_000, ["evo-x2"])
    assert rec.chosen is None and "usability 1" in rec.reason
    (e,) = rec.excluded
    assert e["reason"] == "excluded: timeout_latency share 0.150 > 0.05 (3/20 done rows)"
    assert e["timeout_latency_share"] == pytest.approx(0.15) and e["config_id"] == bad.config_id
    assert rec.to_dict()["excluded"] == rec.excluded
    assert e["reason"] in P.recommend([bad, _pt("ok", 0.0)], 50, 0.9, 30_000, ["evo-x2"]).reason
    # constraint off: the config is a candidate again and nothing is excluded
    off = P.recommend([bad], 50, 0.9, 30_000, ["evo-x2"], max_timeout_share=None)
    assert off.chosen is bad and off.excluded == [] and "timeout_latency" not in off.reason
    # frontier() keeps every config by default (existing frontier rows unchanged)
    assert P.frontier([bad], "evo-x2") == [bad] and P.frontier([bad], "evo-x2", max_timeout_share=0.05) == []


def test_points_table_carries_share_and_exclusions(x2_file, tmp_path):
    pts = load([("evo-x2", str(x2_file))], tmp=tmp_path)
    t = P.points_table(pts)
    json.dumps(t)
    a = next(d for d in t["points"] if d["config_id"] == "evo-x2:llama_server:model-a")
    assert a["timeout_latency_share"] == pytest.approx(1 / 3)
    assert t["max_timeout_share"] == 0.05
    assert [e["config_id"] for e in t["usability_excluded"]] == ["evo-x2:llama_server:model-a"]
    assert t["usability_excluded"][0]["reason"].startswith("excluded: timeout_latency share 0.333 > 0.05")
    assert "evo-x2:llama_server:model-a" not in t["frontier_by_machine"]["evo-x2"]
    assert any("excluded: timeout_latency share" in line for line in P.format_table(pts))
