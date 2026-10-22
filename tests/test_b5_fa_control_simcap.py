"""B5 follow-ups of 2026-10-22 (docs/FINDINGS.md pre-registrations): the flash-attention control (policy i, f16 KV +
flash attention at L(f16), llama3.1:8b f16 recall rerun with a longer timeout) and B5 under a simulated device window
(--window-cap). Fakes only: no server, no model call."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import test_policy_compare as T
from tests.test_policy_compare import ag, pc


def test_fa_reading_rules():
    assert pc.fa_reading(7.0, 6.75, 43.7)["reading"] == "the B5 speedup is flash attention, not KV precision"
    assert pc.fa_reading(42.0, 6.75, 43.7)["reading"] == "the B5 speedup is KV precision, not flash attention"
    r = pc.fa_reading(25.0, 6.75, 43.7)
    assert r["reading"] == "both contribute" and abs(r["fa_share"] - (43.7 - 25.0) / (43.7 - 6.75)) < 1e-9
    assert pc.fa_reading(3.0, 6.75, 43.7)["reading"] == "neither reading"
    assert pc.fa_reading(60.0, 6.75, 43.7)["reading"] == "neither reading"
    assert pc.fa_reading(None, 6.75, 43.7)["reading"].startswith("not read")
    # the 10% boundary is inclusive
    assert pc.fa_reading(6.75 * 1.0999, 6.75, 43.7)["reading"].startswith("the B5 speedup is flash attention")


def test_fa_policy_arm_is_outside_b5_policies():
    assert pc.arm_id("i") == "b5_i_kv_f16_fa_call2_notools" and pc.policy_of_arm(pc.arm_id("i")) == "i"
    assert "i" not in pc.POLICIES and pc.KV_ENV["f16"] == {"OLLAMA_KV_CACHE_TYPE": "f16",
                                                           "OLLAMA_FLASH_ATTENTION": "1"}


def test_base_fa_check_reads_config_and_llama_lines():
    log = ('msg="server config" env="map[OLLAMA_FLASH_ATTENTION:false OLLAMA_KV_CACHE_TYPE:]"\n'
           "llama_context: flash_attn = 0\n")
    c = pc.base_fa_check(log)
    assert c["config_flash_attention_values"] == ["false"] and any("flash_attn = 0" in x for x in
                                                                   c["flash_attention_lines"])
    assert pc.base_fa_check("")["config_flash_attention_values"] == []


def test_resolve_with_cap_caps_every_window_and_sends_it_for_a_and_f():
    win = {"D": 131072, "L": {"base": 131072, "q8_0": 131072, "q4_0": 131072}, "G": 131072}
    c = pc.resolve_policies(win, cap=4096)
    for p in pc.POLICIES:
        assert c[p]["effective_ctx"] == 4096 and c[p]["skip"] is None
    assert c["a"]["num_ctx_sent"] == 4096 and c["f"]["num_ctx_sent"] == 4096
    assert c["b"]["alias_of"] == "a" and c["e"]["alias_of"] == "a"
    assert c["d"]["proxy_ctx"] == 131072          # the cloud proxy is not bound by the device window
    assert c["g"]["env"] == "q8_0" and c["h"]["env"] == "q4_0"
    # no cap: unchanged B5 behaviour
    u = pc.resolve_policies(win)
    assert u["a"]["num_ctx_sent"] is None and u["d"]["proxy_ctx"] == 131072 and u["a"]["effective_ctx"] == 131072
    # a window below the cap is kept
    small = pc.resolve_policies({"D": 2048, "L": {"base": 8192, "q8_0": 8192, "q4_0": 8192}, "G": 2048}, cap=4096)
    assert small["a"]["effective_ctx"] == 2048 and small["b"]["effective_ctx"] == 4096


def _b5_then(tmp_path, monkeypatch):
    notes = []
    deps, rt = T._job_fakes(monkeypatch, notes)
    deps["server"] = T.FakeServer()
    deps["make_router"] = pc.make_router_factory("evo-x2", 4096, ledger_dir=tmp_path)
    b5 = tmp_path / "x2_policy_compare.jsonl"
    assert pc.main(["--platform", "evo-x2", "--out", str(b5), "--seeds", "20260901", "--b1-models", "",
                    "--no-recall"], deps=deps) == "completed"
    return deps, notes


def test_fa_control_job_end_to_end_with_fakes(tmp_path, monkeypatch):
    deps, notes = _b5_then(tmp_path, monkeypatch)
    server = T.FakeServer()
    out = tmp_path / "x2_policy_compare_fa_control.jsonl"
    seen_timeouts = []
    orig = pc.run_recall

    def spy(runtime, model, num_ctx, precision, emit, log):
        seen_timeouts.append((model, precision, num_ctx, ag.CALL_TIMEOUT_S))
        return orig(runtime, model, num_ctx, precision, emit, log)
    monkeypatch.setattr(pc, "run_recall", spy)
    before = ag.CALL_TIMEOUT_S
    note = pc.main(["--platform", "evo-x2", "--fa-control", "--out", str(out), "--seeds", "20260901"],
                   deps=dict(deps, server=server))
    assert note == "completed" and notes[-1] == "completed" and ag.CALL_TIMEOUT_S == before
    rows = ag.read_rows(out)
    rs = rows[0]
    assert rs["record"] == "run_start" and rs["job_kind"] == "fa_control" and rs["harness_blob"] == \
        pc.git_blob_sha1(pc.__file__) and rs["agent_blob"] == pc.git_blob_sha1(ag.__file__)
    assert [s for s in server.starts if s == {}] and [s for s in server.starts if s == pc.KV_ENV["f16"]]
    assert [r for r in rows if r["record"] == "b5_base_fa_check"]
    sess = [r for r in rows if r["record"] == "r2a_session"]
    assert [pc.policy_of_arm(s["arm_id"]) for s in sess] == ["i"] and sess[0]["mode"] == "b5"
    t1 = next(r for r in rows if r["record"] == "r2a_turn" and r["turn_idx"] == 1)
    assert t1["num_ctx_requested"] == 8192
    assert seen_timeouts == [("llama3.1:8b", "f16", 8192, pc.FA_RECALL_TIMEOUT_S)]
    rec = [r for r in rows if r["record"] == "b5_recall"]
    assert len(rec) == 4 and sum(r["warmup"] for r in rec) == 1
    assert rows[-1]["record"] == "run_end"
    rep = json.loads(Path(str(out) + ".report.json").read_text(encoding="utf-8"))
    fa = rep["fa_control"]["llama3.1:8b"]
    assert fa["valid"] and fa["window"] == 8192 and fa["m_i"] is not None and fa["m_gh"] is not None
    assert fa["reading"] in ("the B5 speedup is flash attention, not KV precision",
                             "the B5 speedup is KV precision, not flash attention", "both contribute",
                             "neither reading")


def test_fa_control_refuses_a_model_when_flash_attention_is_not_on(tmp_path, monkeypatch):
    deps, notes = _b5_then(tmp_path, monkeypatch)

    class NoFA(T.FakeServer):
        def log_text(self):
            kv = (self.env or {}).get("OLLAMA_KV_CACHE_TYPE")
            return f'msg="server config" env="map[OLLAMA_KV_CACHE_TYPE:{kv or ""}]"\n' + \
                (f"llama_kv_cache: type_k = '{kv}'" if kv else "")
    out = tmp_path / "x2_policy_compare_fa_control.jsonl"
    pc.main(["--platform", "evo-x2", "--fa-control", "--out", str(out), "--seeds", "20260901"],
            deps=dict(deps, server=NoFA()))
    rows = ag.read_rows(out)
    res = next(r for r in rows if r["record"] == "b5_fa_resolution")
    assert res["config"]["skip"] == "flash attention not on in the server config line"
    assert not [r for r in rows if r["record"] == "r2a_session"]
    assert not [r for r in rows if r["record"] == "b5_recall"]


def test_window_cap_job_end_to_end_with_fakes(tmp_path, monkeypatch):
    notes = []
    deps, rt = T._job_fakes(monkeypatch, notes)
    deps["server"] = T.FakeServer()
    deps["make_router"] = pc.make_router_factory("evo-x2", 2048, ledger_dir=tmp_path)
    out = tmp_path / "x2_policy_compare_simcap2048.jsonl"
    note = pc.main(["--platform", "evo-x2", "--window-cap", "2048", "--out", str(out), "--seeds", "20260901"],
                   deps=deps)
    assert note == "completed"
    rows = ag.read_rows(out)
    assert rows[0]["window_cap"] == 2048 and rows[0]["label"] == "simulated constraint"
    res = next(r for r in rows if r["record"] == "b5_resolution")
    assert {c["effective_ctx"] for c in res["configs"].values()} == {2048}
    assert res["configs"]["d"]["proxy_ctx"] == 8192
    turns = {pc.policy_of_arm(r["arm_id"]): r for r in rows if r["record"] == "r2a_turn" and r["turn_idx"] == 1}
    assert turns["a"]["num_ctx_requested"] == 2048 and turns["f"]["num_ctx_requested"] == 2048
    assert {r["num_ctx"] for r in rows if r["record"] == "b5_recall"} == {2048}
    m = pc.metrics_for_platform(rows, seeds=(20260901,), turns=10)["llama3.1:8b"]
    assert m["complete"] and set(m["policies"]) >= {"a", "c", "d", "f", "g", "h"}


def test_fa_control_and_window_cap_are_separate_and_evo_x2_only(tmp_path):
    with pytest.raises(SystemExit):
        pc.main(["--platform", "evo-x2", "--fa-control", "--window-cap", "4096", "--out",
                 str(tmp_path / "x2_a.jsonl")])
    with pytest.raises(SystemExit):
        pc.main(["--platform", "blade", "--fa-control", "--out", str(tmp_path / "blade_a.jsonl")])
