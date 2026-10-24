"""own-profiler-probe (harness/x2_own_profiler_probe.py and --plan own_profiler_probe_v1 in x2_r2_evalset): the stop
rule, the plan and server-env checks, a budget-stopped session on a fake runtime, scoring on fake records, the
--check-plan dry mode, and the queue entry. No network, no server, no queue access."""
from __future__ import annotations

import json
import math
import sys
import time as _time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "tests"))

import own_compaction as oc  # noqa: E402
import x2_own_profiler_probe as opp  # noqa: E402
import x2_r2_agent as ag  # noqa: E402
import x2_r2_evalset as ev  # noqa: E402

M = "qwen3:14b"
SFX = "_call2_notools"


# ── stop rule, plan, server env ──────────────────────────────────────────────────────────────────────

def test_stop_rule_never_starts_a_turn_expected_past_the_budget():
    assert ev.stop_before_turn([], 600) is False            # the first turn always runs
    assert ev.stop_before_turn([700.0], 600) is True          # first turn alone over budget: nothing more
    assert ev.stop_before_turn([100.0] * 4, 600) is False     # 400 + 100 <= 600
    assert ev.stop_before_turn([100.0] * 5, 600) is False     # 500 + 100 = 600, not over
    assert ev.stop_before_turn([100.0] * 5 + [1.0], 600) is False
    assert ev.stop_before_turn([290.0, 290.0], 600) is True
    assert ev.stop_before_turn([10.0] * 50, None) is False


def test_profiler_plan_tiers_seed_and_budget():
    assert ev.PROBE_WALL_S == opp.PROBE_BUDGET_S == 600.0
    assert ev.OWN_PROFILER_PLAN in ev.CTX_POLICY_PLANS
    assert [c["tier"] for c in opp.CONFIGS] == list(ev.OWN_PROFILER_TIERS)
    for c in opp.CONFIGS:
        assert ev.OWN_PROFILER_TIERS[c["tier"]] == c["kv"]
        p = ev.ctx_policy_plan(ev.OWN_PROFILER_PLAN, c["tier"])
        assert p == [(c["tier"] + SFX, (20260901,), 30)]
        n = ag.ARMS[c["tier"] + SFX]["num_ctx"]
        assert n == (None if c["tier"] == "ollama_default" else c["window"])
    with pytest.raises(ValueError):   # one config per call
        ev.ctx_policy_plan(ev.OWN_PROFILER_PLAN, "ollama_ctx_8192,ollama_ctx_4096")
    with pytest.raises(ValueError):   # probe seed only
        ev.ctx_policy_plan(ev.OWN_PROFILER_PLAN, "ollama_ctx_8192", "20260902")
    with pytest.raises(ValueError):
        ev.ctx_policy_plan(ev.OWN_PROFILER_PLAN, "ollama_ctx_4096_numkeep")


def test_server_env_rules():
    P = ev.OWN_PROFILER_PLAN
    assert ev.server_env_problem(P, "ollama_ctx_8192", []) is None
    assert ev.server_env_problem(P, "ollama_ctx_8192", ["OLLAMA_KV_CACHE_TYPE=f16"])   # f16 = server defaults
    assert ev.server_env_problem(P, "ollama_default", ["OLLAMA_FLASH_ATTENTION=1"])
    assert ev.server_env_problem(P, "ollama_ctx_15360", ["OLLAMA_KV_CACHE_TYPE=q8_0"])  # needs flash attention
    assert ev.server_env_problem(P, "ollama_ctx_15360", ["OLLAMA_KV_CACHE_TYPE=q8_0",
                                                        "OLLAMA_FLASH_ATTENTION=1"]) is None
    assert ev.server_env_problem(P, "ollama_ctx_28928", ["OLLAMA_KV_CACHE_TYPE=q8_0", "OLLAMA_FLASH_ATTENTION=1"])
    assert ev.server_env_problem(ev.OWN_CONFIG_PLAN, "ollama_ctx_15360", []) is not None
    assert ev.server_env_problem(ev.OWN_COMPACTION_PLAN, "own_compaction_4096", []) is None


def test_config_argv_and_files():
    out = r"C:\apu\ovn\results\x2_own_profiler_probe_v1.jsonl"
    a = opp.config_argv(opp.CONFIGS[4], out)
    assert a[a.index("--plan") + 1] == "own_profiler_probe_v1" and a[a.index("--tiers") + 1] == "ollama_ctx_15360"
    assert a[a.index("--seeds") + 1] == "20260901"
    assert "OLLAMA_KV_CACHE_TYPE=q8_0" in a and "OLLAMA_FLASH_ATTENTION=1" in a
    assert a[-1].endswith("x2_own_profiler_probe_v1_ollama_ctx_15360.jsonl")
    assert "--server-env" not in opp.config_argv(opp.CONFIGS[0], out)


# ── a budget-stopped session on a fake runtime ───────────────────────────────────────────────────────

class _Clock:
    """ev.time stand-in: monotonic advances 40 s per read; everything else is the real time module."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        self.t += 40.0
        return self.t

    def __getattr__(self, name):
        return getattr(_time, name)


def test_session_stops_by_budget_and_writes_no_complete_session(tmp_path, monkeypatch):
    from test_own_compaction import SessionFake
    log = tmp_path / "serve.log"
    log.write_text("OLLAMA_DEBUG:DEBUG\n", encoding="utf-8")
    es = ev.build_session(20260901, 30, "evalset", False, M, design="v2")
    rows = []
    rt = oc.PolicySessionRuntime(SessionFake(es, log, 8192), log, rows.append, ev.ctx_policy_of, sleep=lambda s: None)
    rt.cut.gin_wait_s = 0.0
    monkeypatch.setattr(ev, "time", _Clock())
    ev.run_session(rt, M, "ollama_ctx_8192" + SFX, 20260901, 30, rows.append, lambda m: None, "real", "evalset",
                   False, None, "v2", wall_budget_s=600.0)
    turns = [r for r in rows if r.get("record") == "r2a_turn"]
    ps = [r for r in rows if r.get("record") == "probe_session"]
    assert 0 < len(turns) < 30 and len(ps) == 1 and ps[0]["stopped_by_budget"] is True
    assert not any(r.get("record") == "r2a_session" for r in rows)
    walls = [t["turn_wall_s"] for t in turns]
    assert sum(walls[:-1]) + walls[-2] <= 600.0     # the last turn was started only when expected to fit
    assert all("cut_v2_flagged" in t for t in turns)


def test_session_without_budget_is_unchanged(tmp_path):
    from test_own_compaction import SessionFake
    log = tmp_path / "serve.log"
    log.write_text("OLLAMA_DEBUG:DEBUG\n", encoding="utf-8")
    es = ev.build_session(20260901, 30, "evalset", False, M, design="v2")
    rows = []
    rt = oc.PolicySessionRuntime(SessionFake(es, log, 8192), log, rows.append, ev.ctx_policy_of, sleep=lambda s: None)
    rt.cut.gin_wait_s = 0.0
    ev.run_session(rt, M, "ollama_ctx_8192" + SFX, 20260901, 30, rows.append, lambda m: None, "real", "evalset",
                   False, None, "v2")
    assert sum(1 for r in rows if r.get("record") == "r2a_turn") == 30
    assert any(r.get("record") == "r2a_session" for r in rows)
    assert not any(r.get("record") == "probe_session" for r in rows)


# ── scoring on fake records ──────────────────────────────────────────────────────────────────────────

def _turn(tier, seed, idx, wall, ok=True, flagged=None, loaded=None, att="a", over=None):
    r = {"record": "r2a_turn", "mode": "real", "item_kind": "task", "arm_id": tier + SFX, "seed": seed,
         "attempt_id": att, "turn_idx": idx, "turn_wall_s": wall, "end_state_ok": ok, "loaded_context": loaded}
    if flagged is not None:
        r["cut_v2_flagged"] = flagged
    if over is not None:
        r["over_loaded_window"] = over
    return r


def _probe_rows(cfg, walls, kv=None, flagged=(), loaded=None, run_end=True):
    rows = [{"record": "run_start"}]
    for i, w in enumerate(walls, 1):
        rows.append(_turn(cfg["tier"], 20260901, i, w, ok=i != 2, flagged=i in flagged,
                          loaded=cfg["window"] if loaded is None else loaded))
    rows.append({"record": "kv_cache_log", "lines": [f"llama_kv_cache: size = 1.0 MiB, K ({kv or cfg['kv']}): x"]})
    if run_end:
        rows.append({"record": "run_end"})
    return rows


def _ref_rows(tier, per_seed, seeds=(20260901, 20260902, 20260903), v2=False, flag_some=False):
    rows = []
    for s in seeds:
        for i in range(1, 4):
            rows.append(_turn(tier, s, i, per_seed[s], ok=i != 3,
                              flagged=(flag_some and i == 3) if v2 else None,
                              over=None if v2 else (i == 3)))
        rows.append({"record": "r2a_session", "arm_id": tier + SFX, "seed": s, "attempt_id": "a"})
    rows.append({"record": "run_end"})
    return rows


def test_probe_prefix_scores_only_turns_inside_600_s():
    cfg = opp.CONFIGS[2]
    p = opp.probe_features(_probe_rows(cfg, [100, 200, 250, 100]), cfg)
    assert p["n_turns_run"] == 4 and p["n_scored"] == 3 and p["seconds"] == pytest.approx(550 / 3)
    assert p["acc"] == pytest.approx(2 / 3) and p["valid"]


def test_probe_validity_rules():
    cfg = opp.CONFIGS[4]
    assert not opp.probe_features(_probe_rows(cfg, [10, 10], kv="f16"), cfg)["valid"]        # wrong KV line
    assert not opp.probe_features(_probe_rows(cfg, [10, 10], loaded=8192), cfg)["valid"]     # wrong window
    assert not opp.probe_features(_probe_rows(cfg, [10, 10], run_end=False), cfg)["valid"]
    assert not opp.probe_features(_probe_rows(cfg, [700]), cfg)["valid"]                     # nothing scored
    assert opp.probe_features(_probe_rows(cfg, [10, 10]), cfg)["valid"]


def test_reference_excludes_the_probe_seed_and_picks_the_detector():
    r = opp.reference_features(_ref_rows("ollama_ctx_8192", {20260901: 99.0, 20260902: 10.0, 20260903: 20.0}),
                               "ollama_ctx_8192")
    assert r["seeds"] == [20260902, 20260903] and r["n"] == 6 and r["seconds"] == pytest.approx(15.0)
    assert r["overflow_detector"].startswith("v1") and r["overflow_rate"] == pytest.approx(1 / 3)
    r2 = opp.reference_features(_ref_rows("ollama_ctx_15360", {20260901: 1, 20260902: 1, 20260903: 1}, v2=True),
                                "ollama_ctx_15360")
    assert r2["overflow_detector"] == "v2" and r2["overflow_rate"] == 0


def test_score_medians_misses_agreement_and_verdict():
    C = {c["tier"]: c for c in opp.CONFIGS}
    probe = {
        "ollama_default": _probe_rows(C["ollama_default"], [10.0] * 30),
        "ollama_ctx_16384": _probe_rows(C["ollama_ctx_16384"], [11.0] * 30),
        "ollama_ctx_8192": _probe_rows(C["ollama_ctx_8192"], [30.0] * 20),
        "ollama_ctx_4096": _probe_rows(C["ollama_ctx_4096"], [20.0] * 20, kv="q8_0"),   # invalid: miss
        "ollama_ctx_15360": _probe_rows(C["ollama_ctx_15360"], [12.0] * 30, flagged=(5,)),
    }   # q4 probe missing entirely
    b3 = []
    for tier, s in (("ollama_default", 10.0), ("ollama_ctx_16384", 10.0), ("ollama_ctx_8192", 24.0),
                    ("ollama_ctx_4096", 18.0)):
        b3 += _ref_rows(tier, {20260901: s, 20260902: s, 20260903: s})[:-1]
    b3.append({"record": "run_end"})
    refs = {opp.B3V2_FILE: b3,
            opp.REF_Q8: _ref_rows("ollama_ctx_15360", {20260901: 12, 20260902: 12, 20260903: 12}, v2=True)}
    d = opp.score(probe, refs)
    by = {e["cfg"]["tier"]: e for e in d["configs"]}
    assert by["ollama_default"]["err_s"] == pytest.approx(0.0)
    assert by["ollama_ctx_16384"]["err_s"] == pytest.approx(10.0)
    assert by["ollama_ctx_8192"]["err_s"] == pytest.approx(25.0)
    assert by["ollama_ctx_4096"]["err_s"] == math.inf
    assert by["ollama_ctx_15360"]["overflow_agree"] is False      # probe flagged, reference none
    assert by["ollama_ctx_28928"]["scored"] is False               # no reference file
    assert d["n_scored"] == 5 and d["med_s"] == pytest.approx(10.0) and d["met"] is False
    assert (d["agree"], d["agree_n"]) == (0, 1)
    t = opp.probe_text(d)
    assert "MISSED" in t and "no reference" in t and "INVALID" in t and "\u2014" not in t


def test_reference_without_run_end_is_not_scored():
    cfg = opp.CONFIGS[5]
    ref = _ref_rows(cfg["tier"], {20260901: 5, 20260902: 5, 20260903: 5}, v2=True)[:-1]
    d = opp.score({cfg["tier"]: _probe_rows(cfg, [5.0] * 30)}, {opp.REF_Q4: ref}, configs=[cfg])
    assert d["n_scored"] == 0 and d["med_s"] is None and d["met"] is False


# ── dry mode, queue entry, hours ─────────────────────────────────────────────────────────────────────

def test_check_plan_dry_mode(tmp_path, capsys):
    out = tmp_path / "x2_own_profiler_probe_v1.jsonl"
    assert opp.main(["--out", str(out), "--rules-from", "v.jsonl", "--check-plan"]) == "checked"
    printed = capsys.readouterr().out
    j = json.loads(printed[printed.rindex('{\n "configs"'):])
    assert [c["check"] for c in j["configs"]] == ["checked"] * 6
    assert not out.exists() and not any(tmp_path.iterdir())


def test_queue_entry_and_hours():
    import x2_queue_fill as qf
    e = qf.entry("x2_own_profiler_probe_v1")
    assert e["cmd"][1] == "ownprobe\\x2_own_profiler_probe.py" and "tq.advance()" in e["note"]
    assert e["cmd"][-1].endswith("x2_own_profiler_probe_v1.jsonl") and "gate" not in e
    est = opp.hours_estimate(ag.read_rows(REPO / opp.B3V2_FILE))
    assert qf.OP_HOURS == (round(est["point_h"], 2), round(est["high_h"], 2))
    assert 0 < est["point_h"] <= est["high_h"] < 2
    q = [{"id": j, "status": "pending"} for j in ("x2_own_compaction_v1_a", "x2_own_compaction_v1_b",
                                                  "x2_own_config_probe_q8_15360", "x2_own_config_probe_q4_28928")]
    new = qf.append_entries(q, ["x2_own_profiler_probe_v1"])
    assert [x["id"] for x in new][-2:] == ["x2_own_config_probe_q4_28928", "x2_own_profiler_probe_v1"]
    with pytest.raises(ValueError):
        qf.append_entries(new, ["x2_own_profiler_probe_v1"])
