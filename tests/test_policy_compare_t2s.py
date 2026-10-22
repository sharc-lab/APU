"""B5 on evo-t2s Intel CPU and Intel iGPU (pre-registration 2026-10-22): the CPU platform, the handover-deadline stop at a
session boundary, the pre-registered T2S verdict on synthetic cells, and the estimate from measured evo-t2s sessions."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "tests"))

import x2_r2_agent as ag  # noqa: E402
import policy_compare as pc  # noqa: E402
from test_policy_compare import FakeServer, _job_fakes  # noqa: E402

needs_results = pytest.mark.skipif(not (REPO / "results" / "t2s_r2_real_cpu_v1.jsonl").exists(),
                                   reason="evo-t2s result files not present")


def test_cpu_platform_is_stock_ollama_on_evo_t2s():
    c = pc.PLATFORMS["evo-t2s-cpu"]
    assert c["host"] == "EVO-T2S" and c["prefix"] == "t2s_" and c["base_env"] == {} and c["queue"]
    assert c["rules_from"] == ("t2s_r2_validation_cpu_v1.jsonl",)
    assert pc.PLATFORMS["evo-t2s"]["base_env"] == {"OLLAMA_IGPU_ENABLE": "1"}
    assert pc.T2S_SEEDS == (20260901, 20260902, 20260903, 20260904, 20260905)


def test_handover_deadline_stops_before_the_first_session(tmp_path, monkeypatch):
    notes = []
    deps, _ = _job_fakes(monkeypatch, notes)
    deps.update(server=FakeServer(), hostname="EVO-T2S", make_router=pc.make_router_factory("evo-t2s", 4096,
                                                                                           ledger_dir=tmp_path),
                deadline_reached=lambda: {"hard_deadline_utc": "2026-11-09T04:30:00+00:00"})
    deps["hc"].require_host = lambda h: {"hw_id": "evo-t2s"}
    out = tmp_path / "t2s_policy_compare_cpu_llama31_8b_v1.jsonl"
    note = pc.main(["--platform", "evo-t2s-cpu", "--out", str(out), "--seeds", "20260901", "--models", "llama3.1:8b"],
                   deps=deps)
    assert "handover deadline" in note and notes == [note]
    rows = ag.read_rows(out)
    assert not [r for r in rows if r["record"] == "r2a_session"] and rows[-1]["record"] == "run_end"


def test_handover_deadline_helper():
    assert pc.handover_deadline_reached("evo-t2s", {"deadline_reached": lambda: {"x": 1}})
    assert not pc.handover_deadline_reached("evo-t2s", {"deadline_reached": lambda: None})
    assert not pc.handover_deadline_reached("blade", {"deadline_reached": lambda: {"x": 1}})


def _pol(rule, rec, wall, usd=0.0):
    return {"rule_pass_rate": rule, "history_recall": rec, "wall_min": wall, "cloud_usd": usd}


def _cells(**by_model):
    return {m: {"seeds_used": list(pc.T2S_SEEDS), "problems": [], "policies": p, "complete": True}
            for m, p in by_model.items()}


def test_t2s_verdict_supported_refuted_inconclusive(monkeypatch):
    good = {"a": _pol(0.6, 0.2, 100), "b": _pol(1.0, 1.0, 600), "d": _pol(0.95, 0.9, 120, 0.5)}
    no_gain = {"a": _pol(0.6, 0.2, 100), "b": _pol(1.0, 1.0, 600), "d": _pol(0.6, 0.2, 100, 0.5)}
    table = {}
    monkeypatch.setattr(pc, "_t2s_cells", lambda rows, prices=None: table[rows[0]])
    table.update(cpu=_cells(**{"llama3.1:8b": good}), igpu=_cells(**{"llama3.1:8b": good}))
    v = pc.b5_t2s_verdict({"evo-t2s-cpu": ["cpu"], "evo-t2s": ["igpu"]})
    assert v["verdict"].startswith("supported")
    table.update(cpu=_cells(**{"llama3.1:8b": no_gain}))
    assert pc.b5_t2s_verdict({"evo-t2s-cpu": ["cpu"], "evo-t2s": ["igpu"]})["verdict"].startswith("refuted (router dom")
    assert pc.b5_t2s_verdict({"evo-t2s": ["igpu"]})["verdict"].startswith("inconclusive (no complete CPU cell")
    no_gain_not_dominated = {"a": _pol(0.6, 0.2, 100), "d": _pol(0.6, 0.2, 90, 0.5)}
    table.update(cpu=_cells(**{"llama3.1:8b": no_gain_not_dominated}))
    assert "does not beat the default" in pc.b5_t2s_verdict({"evo-t2s-cpu": ["cpu"]})["verdict"]


def test_router_beats_default():
    assert pc.router_beats_default({"a": _pol(0.5, 0.5, 1), "d": _pol(0.5, 0.6, 9)})
    assert not pc.router_beats_default({"a": _pol(0.5, 0.5, 1), "d": _pol(0.6, 0.4, 9)})
    assert not pc.router_beats_default({"a": _pol(0.5, 0.5, 1)})


@needs_results
def test_estimate_from_measured_t2s_sessions():
    cpu, igpu = pc.estimate_t2s("evo-t2s-cpu"), pc.estimate_t2s("evo-t2s")
    for e, D in ((cpu, 4096), (igpu, 32768)):
        assert e["label"].startswith("ESTIMATE") and e["seeds"] == list(pc.T2S_SEEDS)
        for m in ("llama3.1:8b", "qwen3:14b"):
            assert e["models"][m]["D"] == D and e["models"][m]["measured_n"]["default"] == 5
            assert e["models"][m]["total_h"] > 0
    # amendment 2026-10-21 22:30: evo-t2s runs a, c, d, e, f; (e) on the iGPU has its own sessions (no (b) to alias)
    for e in (cpu, igpu):
        assert e["policies"] == list(pc.T2S_POLICIES)
        assert not {"b", "g", "h"} & set(e["models"]["llama3.1:8b"]["min_per_session"])
    assert "e" not in cpu["models"]["llama3.1:8b"]["min_per_session"] and cpu["models"]["llama3.1:8b"]["aliases"] == {}
    assert igpu["models"]["llama3.1:8b"]["aliases"] == {} and igpu["models"]["llama3.1:8b"]["min_per_session"]["e"] > 0
    assert pc.estimate_platform("evo-t2s-cpu")["total_h"] == pytest.approx(cpu["total_h"])
    full = pc.estimate_t2s("evo-t2s", policies=tuple(pc.POLICIES))
    assert full["models"]["llama3.1:8b"]["aliases"] == {"e": "b"} and full["total_h"] > igpu["total_h"]


def test_t2s_policy_set_and_parse():
    assert pc.T2S_POLICIES == ("a", "c", "d", "e", "f")
    assert pc.platform_policies("evo-t2s") == pc.platform_policies("evo-t2s-cpu") == pc.T2S_POLICIES
    assert pc.platform_policies("evo-x2") == tuple(pc.POLICIES) == pc.platform_policies("blade")
    assert pc.parse_policy_set(None, "evo-t2s-cpu") == pc.T2S_POLICIES
    assert pc.parse_policy_set("f,a,d", "evo-x2") == ("a", "d", "f")
    with pytest.raises(ValueError):
        pc.parse_policy_set("a,c", "evo-t2s")            # no router
    with pytest.raises(ValueError):
        pc.parse_policy_set("a,d,z", "evo-t2s")
    assert pc.probe_envs(pc.T2S_POLICIES) == ["base"] and pc.recall_precisions(pc.T2S_POLICIES) == ["f16"]
    assert pc.probe_envs(tuple(pc.POLICIES)) == ["base", "q8_0", "q4_0"]
    assert pc.recall_precisions(tuple(pc.POLICIES)) == list(pc.RECALL_PRECISIONS)


def test_resolve_with_the_t2s_set_skips_b_g_h_and_e_does_not_alias_b():
    win = {"D": 32768, "L": {"base": 131072}, "G": 131072}
    cfg = pc.resolve_policies(win, policies=pc.T2S_POLICIES)
    for p in ("b", "g", "h"):
        assert cfg[p]["skip"] == pc.NOT_IN_SET
    assert cfg["e"]["skip"] is None and cfg["e"]["alias_of"] is None and cfg["e"]["effective_ctx"] == 131072
    assert cfg["d"]["proxy_ctx"] == 131072                 # the router's full-history proxy still uses L(base)
    assert pc.distinct_policies(cfg) == ["a", "c", "d", "e", "f"]
    assert pc.resolve_policies(win)["e"]["alias_of"] == "b"   # evo-x2 / blade: unchanged
    cpu = pc.resolve_policies({"D": 4096, "L": {"base": 131072}, "G": None}, policies=pc.T2S_POLICIES)
    assert pc.distinct_policies(cpu) == ["a", "c", "d", "f"]  # no GPU: (e) has no window (2026-10-14 rule)


def test_cpu_job_runs_only_the_t2s_set(tmp_path, monkeypatch):
    notes = []
    deps, _ = _job_fakes(monkeypatch, notes)
    server = FakeServer()
    deps.update(server=server, hostname="EVO-T2S", deadline_reached=lambda: None,
                make_router=pc.make_router_factory("evo-t2s", 4096, ledger_dir=tmp_path))
    deps["hc"].require_host = lambda h: {"hw_id": "evo-t2s"}
    out = tmp_path / "t2s_policy_compare_cpu_llama31_8b_v1.jsonl"
    note = pc.main(["--platform", "evo-t2s-cpu", "--out", str(out), "--seeds", "20260901", "--models", "llama3.1:8b"],
                   deps=deps)
    assert note == "completed" and notes == [note]
    rows = ag.read_rows(out)
    assert rows[0]["policy_set"] == list(pc.T2S_POLICIES)
    res = next(r for r in rows if r["record"] == "b5_resolution")
    assert set(res["windows"]["L"]) == {"base"} and all(res["configs"][p]["skip"] == pc.NOT_IN_SET for p in "bgh")
    sess = sorted(pc.policy_of_arm(r["arm_id"]) for r in rows if r["record"] == "r2a_session")
    assert sess == ["a", "c", "d", "f"]                 # (e) = G = D here, an alias of (a)
    assert {s.get("OLLAMA_KV_CACHE_TYPE") for s in server.starts} == {None, "f16"}   # f16 only for the recall probe
    assert not [r for r in rows if r["record"] == "b5_kv_check"]
    assert {r["precision"] for r in rows if r["record"] == "b5_recall"} == {"f16"}
    m = pc.metrics_for_platform(rows, seeds=(20260901,), turns=10)["llama3.1:8b"]
    assert set(m["policies"]) == {"a", "c", "d", "e", "f"}


def test_policies_flag_overrides_and_is_validated(tmp_path, monkeypatch):
    notes = []
    deps, _ = _job_fakes(monkeypatch, notes)
    with pytest.raises(SystemExit):
        pc.main(["--platform", "evo-t2s", "--out", str(tmp_path / "t2s_x.jsonl"), "--policies", "a,c"], deps=deps)
