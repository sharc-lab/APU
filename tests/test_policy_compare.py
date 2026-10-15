"""B5 policy comparison (harness/policy_compare.py): pure parts, wrappers, metrics/verdict, and the job with fakes."""
import json
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "analysis"))

import x2_r2_agent as ag  # noqa: E402
import policy_compare as pc  # noqa: E402
from test_x2_r2_agent import FakeRuntime, good_policy  # noqa: E402


# ── windows ─────────────────────────────────────────────────────────────────────────────────────────

def test_ladder_caps_at_native_and_includes_it():
    assert pc.ladder_for(40960) == [40960, 32768, 16384, 8192, 4096]
    assert pc.ladder_for(100000) == [100000, 65536, 40960, 32768, 16384, 8192, 4096]
    assert pc.ladder_for(None)[0] == 262144


def _pt(ok=True, loaded=None, size=100, vram=100):
    return {"ok": ok, "loaded_context": loaded, "size": size, "size_vram": vram}


def test_descend_stops_at_first_full_gpu_and_windows():
    table = {32768: _pt(False), 16384: _pt(True, 16384, 100, 60), 8192: _pt(True, 8192, 100, 100),
             4096: _pt(True, 4096)}
    seen = []

    def probe(c):
        seen.append(c)
        p = dict(table[c])
        if p["loaded_context"] is None and p["ok"]:
            p["loaded_context"] = c
        return p
    pts = pc.descend(probe, [32768, 16384, 8192, 4096])
    assert seen == [32768, 16384, 8192]
    assert pc.windows_from_points(pts) == {"largest_starts": 16384, "largest_full_gpu": 8192}
    seen.clear()
    pts = pc.descend(probe, [32768, 16384, 8192, 4096], need_full_gpu=False)
    assert seen == [32768, 16384] and pc.windows_from_points(pts)["largest_starts"] == 16384


def test_a_point_whose_loaded_context_differs_did_not_start():
    assert not pc.point_starts({"ok": True, "num_ctx": 8192, "loaded_context": 4096})
    assert not pc.point_full_gpu({"ok": True, "num_ctx": 8192, "loaded_context": 8192, "size": 0, "size_vram": 0})


# ── policy resolution ───────────────────────────────────────────────────────────────────────────────

def test_resolve_x2_like_aliases_b_and_e_to_a():
    c = pc.resolve_policies({"D": 131072, "L": {"base": 131072, "q8_0": 131072, "q4_0": 131072}, "G": 131072})
    assert c["b"]["alias_of"] == "a" and c["e"]["alias_of"] == "a"
    assert pc.distinct_policies(c) == ["a", "c", "d", "f", "g", "h"]
    assert c["a"]["num_ctx_sent"] is None and c["c"]["num_ctx_sent"] == 131072 and c["f"]["num_ctx_sent"] is None
    assert c["d"]["proxy_ctx"] == 131072 and c["g"]["env"] == "q8_0" and c["h"]["env"] == "q4_0"
    assert c["c"]["alias_of"] is None and c["g"]["alias_of"] is None   # a wrapper or another env is never an alias


def test_resolve_t2s_like_and_missing_windows():
    c = pc.resolve_policies({"D": 32768, "L": {"base": 131072, "q8_0": None, "q4_0": 65536}, "G": 131072})
    assert c["b"]["alias_of"] is None and c["e"]["alias_of"] == "b"
    assert c["g"]["skip"] and "q8_0" in c["g"]["skip"] and c["h"]["num_ctx_sent"] == 65536
    c = pc.resolve_policies({"D": None, "L": {"base": None}, "G": None})
    assert pc.distinct_policies(c) == []


def test_session_items_seed_major_in_run_order():
    cfg = pc.resolve_policies({"D": 4096, "L": {"base": 8192, "q8_0": 8192, "q4_0": 8192}, "G": 4096})
    items = pc.session_items(["m1", "m2"], {"m1": cfg, "m2": cfg}, seeds=(1, 2))
    assert items[:7] == [(1, "m1", p) for p in ("a", "c", "f", "d", "g", "h", "b")]
    assert items[7][1] == "m2" and items[14][0] == 2 and len(items) == 28


def test_server_env_for_kv():
    assert pc.server_env_for({"OLLAMA_IGPU_ENABLE": "1"}, "q4_0") == {
        "OLLAMA_IGPU_ENABLE": "1", "OLLAMA_KV_CACHE_TYPE": "q4_0", "OLLAMA_FLASH_ATTENTION": "1"}
    assert pc.server_env_for({}, "base") == {}


# ── KV check ────────────────────────────────────────────────────────────────────────────────────────

def test_kv_check():
    ok = 'level=INFO msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true OLLAMA_KV_CACHE_TYPE:q8_0]"\n' \
         "llama_kv_cache: type_k = 'q8_0', type_v = 'q8_0'"
    r = pc.kv_check(ok, "q8_0")
    assert r["valid"] and r["verified"] and r["config_flash_attention"]
    r = pc.kv_check(ok.replace("type_k = 'q8_0'", "type_k = 'f16'"), "q8_0")
    assert not r["valid"] and "f16" in r["reason"]
    r = pc.kv_check('msg="server config" env="map[OLLAMA_KV_CACHE_TYPE:q4_0]"', "q4_0")
    assert r["valid"] and not r["verified"]
    assert not pc.kv_check("no config line", "q4_0")["valid"]
    assert pc.kv_check("", None)["valid"]


# ── recall probe ────────────────────────────────────────────────────────────────────────────────────

def test_recall_prompt_size_needles_and_score():
    prompt, needles = pc.build_recall_prompt(1, 20000, 1.0)
    assert len(needles) == pc.RECALL_NEEDLES and all(v in prompt for v in needles.values())
    assert 0.9 * 20000 <= ag.est_tokens(prompt) <= 1.1 * 20000
    p2, n2 = pc.build_recall_prompt(1, 20000, 1.0)
    assert p2 == prompt and n2 == needles and pc.recall_needles(2) != needles
    s = pc.score_recall(" ".join(list(needles.values())[:3]), needles)
    assert s["found"] == 3 and s["recall"] == pytest.approx(3 / 8)


# ── wrappers ────────────────────────────────────────────────────────────────────────────────────────

def test_kept_by_turn_ids():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "t1"}, {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"}]
    assert [m["content"] for m in pc.kept_by_turn_ids(msgs, [2, 3])] == ["s", "u2", "a2", "u3"]
    assert [m["content"] for m in pc.kept_by_turn_ids(msgs, [3])] == ["s", "u3"]


class _Dec:
    def __init__(self, target, kept=None):
        self.target, self.kept_turn_ids = target, kept or []
        self.reason, self.tokens, self.dropped_turn_ids, self.needs_turns = target, {"system": 10,
                                                                                    "full_history": 2000}, [], []
        self.quality_floor_met, self.est_cost_usd, self.budget_remaining_usd, self.stub = True, 0.001, 4.9, True


class _Rec:
    def __init__(self):
        self.calls = []
        self.unloads = []

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        self.calls.append((len(messages), num_ctx))
        return {"outcome": "ok", "status": 200, "message": "x", "duration_s": 2.0, "eval_count": 40,
                "prompt_eval_count": 100}

    def loaded_context(self, model):
        return 4096

    def unload(self, model):
        return True

    def recover(self):
        return True


def test_router_runtime_routes_and_emits():
    targets = iter(["local", "local_trimmed", "cloud", "refuse"])
    steps = []

    class R:
        def decide(self, step):
            steps.append(step)
            t = next(targets)
            return _Dec(t, kept=[2] if t == "local_trimmed" else None)
    inner, rows = _Rec(), []
    rt = pc.RouterPolicyRuntime(inner, lambda m: R(), 4096, 8192, rows.append)
    rt.begin_session("m", "arm", 1)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a"}, {"role": "user", "content": "u2"}]
    for _ in range(3):
        rt.chat("m", msgs, 4096, None, None)
    msgs5 = msgs + [{"role": "user", "content": f"u{i}"} for i in range(3, 6)]
    resp = rt.chat("m", msgs5, 4096, None, None)
    assert inner.calls == [(4, 4096), (2, 4096), (4, 8192)]
    assert resp["outcome"] == "router_refused" and resp["status"] is None
    assert [r["target"] for r in rows] == ["local", "local_trimmed", "cloud", "refuse"]
    assert [r["call_idx"] for r in rows] == [1, 2, 3, 1] and rows[3]["turn_idx"] == 5
    assert steps[3]["needs_turns"] == [1] and steps[0]["needs_turns"] == []   # turn 5 is a canary turn
    assert rows[2]["sent_num_ctx"] == 8192 and rows[2]["eval_count"] == 40


def test_num_keep_runtime_sets_options_extra_per_session():
    class Base(_Rec):
        options_extra = None
    base, rows = Base(), []
    rt = pc.NumKeepPolicyRuntime(base, lambda m, a, s: {"system_prompt_tokens": 321, "method": "fake"}, rows.append)
    with pytest.raises(RuntimeError):
        rt.chat("m", [], None, None, None)
    rt.begin_session("m", "arm", 1)
    assert base.options_extra == {"num_keep": 321}
    r = rt.chat("m", [{"role": "user", "content": "u"}], None, None, None)
    assert r["num_keep"]["sent"] == 321 and rows[0]["record"] == "b5_num_keep"
    rt.end_session()
    assert base.options_extra is None


def test_router_factory_builds_a_router_for_every_platform(tmp_path):
    for plat in pc.PLATFORMS:
        r = pc.make_router_factory(plat, 4096, ledger_dir=tmp_path)("llama3.1:8b")
        assert r.num_ctx == 4096 and r.stub
    assert not list(tmp_path.iterdir())    # the stub client writes no ledger


# ── metrics / cost / Pareto ─────────────────────────────────────────────────────────────────────────

def test_cloud_cost_caching_matches_the_shared_price_table():
    import cloud_budget_forecast as cbf
    prices = cbf.load_prices()
    r1 = {"target": "cloud", "tokens": {"system": 1000, "full_history": 3000}, "eval_count": 100,
          "ts_utc": "2026-10-20T00:00:00+00:00"}
    r2 = dict(r1, tokens={"system": 1000, "full_history": 5000}, ts_utc="2026-10-20T00:10:00+00:00")
    r3 = dict(r1, tokens={"system": 1000, "full_history": 6000}, ts_utc="2026-10-20T01:00:00+00:00")  # > 30 min
    got = pc.cloud_session_cost([r1, {"target": "local"}, r2, r3], prices)
    m, t = pc.CLOUD_PRICE_MODEL, pc.CLOUD_PRICE_TIER
    exp = (cbf.call_cost(prices, m, t, 4000, 100, cached=0, write=4000)
           + cbf.call_cost(prices, m, t, 6000, 100, cached=4000, write=2000)
           + cbf.call_cost(prices, m, t, 7000, 100, cached=0, write=7000))
    assert got["n_cloud_calls"] == 3 and got["usd"] == pytest.approx(exp)
    nc = pc.cloud_session_cost([r1, r2, r3], prices, caching=False)["usd"]
    assert nc == pytest.approx(sum(cbf.call_cost(prices, m, t, p, 100) for p in (4000, 6000, 7000)))


def test_session_metrics_retimes_cloud_steps():
    turns = [{"turn_idx": i, "turn_wall_s": 60.0, "rule1_json_keys": i != 2, "canary_check": i == 5,
              "canary_hist_ok": True} for i in range(1, 6)]
    routes = [{"target": "cloud", "duration_s": 30.0, "eval_count": 100, "tokens": {}, "ts_utc": None}]
    m = pc.session_metrics(turns, ["rule1_json_keys"], routes)
    assert m["rule_pass"] == 4 and m["rule_n"] == 5 and m["hist_ok"] == 1 and m["canary_turns"] == 1
    assert m["wall_min_proxy_measured"] == pytest.approx(5.0)
    assert m["wall_min"] == pytest.approx((300 - 30 + pc.cloud_step_wall_s(100)) / 60)
    assert m["targets"]["cloud"] == 1


def test_dominance_and_frontier():
    a = {"rule_pass_rate": 0.9, "history_recall": 0.5, "wall_min": 10, "cloud_usd": 0}
    d = {"rule_pass_rate": 1.0, "history_recall": 1.0, "wall_min": 12, "cloud_usd": 0.1}
    b = {"rule_pass_rate": 0.9, "history_recall": 0.5, "wall_min": 20, "cloud_usd": 0}
    assert pc.dominates(a, b) and not pc.dominates(a, d) and not pc.dominates(d, a)
    assert not pc.dominates(a, dict(a))                      # equal vectors do not dominate
    assert sorted(pc.pareto_frontier({"a": a, "b": b, "d": d})) == ["a", "d"]
    assert not pc.dominates(dict(a, history_recall=None), b)


# synthetic rows for one platform: per policy, per seed, `turns` turns with given rule/hist/wall
def _rows(policies: dict, model="m", seeds=(1, 2), turns=5, finished=True, cloud=None):
    rows = [{"record": "run_start", "mode": "b5", "rules_in_use": {model: ["rule1_json_keys"]}}]
    win = {"D": 4096, "L": {"base": 8192, "q8_0": 8192, "q4_0": 8192}, "G": 4096}
    cfg = pc.resolve_policies(win)
    rows.append({"record": "b5_resolution", "model_id": model, "configs": cfg})
    for p, (rule_ok, hist_ok, wall) in policies.items():
        arm = pc.arm_id(p)
        for s in seeds:
            att = f"{p}{s}"
            if cloud and p == "d":
                rows.append({"record": "b5_route", "model_id": model, "arm_id": arm, "seed": s, "session_uid": att,
                             "turn_idx": 1, "call_idx": 1, "target": "cloud", "duration_s": 0.0, "eval_count": 0,
                             "tokens": {"system": 2000, "full_history": 2000}, "ts_utc": "2026-10-20T00:00:00+00:00"})
            for t in range(1, turns + 1):
                rows.append({"record": "r2a_turn", "mode": "b5", "model_id": model, "arm_id": arm, "seed": s,
                             "attempt_id": att, "turn_idx": t, "turn_wall_s": wall * 60 / turns,
                             "rule1_json_keys": rule_ok, "canary_check": t == turns, "canary_hist_ok": hist_ok})
            rows.append({"record": "r2a_session", "mode": "b5", "model_id": model, "arm_id": arm, "seed": s,
                         "attempt_id": att})
    if finished:
        rows.append({"record": "run_end"})
    return rows


ALL = ("a", "c", "d", "f", "g", "h", "b")   # e aliases a in _rows' windows


def _pol(**over):
    base = {p: (True, False, 10.0) for p in ALL}
    base.update(over)
    return base


def test_metrics_balanced_seeds_and_alias():
    rows = _rows(_pol(d=(True, True, 12.0)), seeds=(1, 2))
    m = pc.metrics_for_platform(rows, seeds=(1, 2), turns=5)["m"]
    assert m["complete"] and m["balanced_seeds"] == [1, 2]
    assert m["policies"]["d"]["history_recall"] == 1.0 and m["policies"]["a"]["history_recall"] == 0.0
    assert m["policies"]["e"]["alias_of"] == "a" and m["policies"]["e"]["wall_min"] == m["policies"]["a"]["wall_min"]
    # a policy missing a seed unbalances the cell
    rows = [r for r in rows if not (r.get("arm_id") == pc.arm_id("g") and r.get("seed") == 2)]
    m = pc.metrics_for_platform(rows, seeds=(1, 2), turns=5)["m"]
    assert not m["complete"] and m["balanced_seeds"] == [1]


def test_verdict_supported_refuted_inconclusive():
    import cloud_budget_forecast as cbf
    prices = cbf.load_prices()
    # router: best history at some USD; the fixed settings trade wall clock differently per platform
    x2 = _rows(_pol(a=(True, False, 5.0), d=(True, True, 12.0), b=(True, True, 30.0)), cloud=True)
    t2s = _rows(_pol(c=(True, False, 4.0), a=(True, False, 9.0), d=(True, True, 12.0), b=(True, True, 30.0)),
                cloud=True)
    # on the Blade the largest window spills and loses its rules (b dominated there)
    blade = _rows(_pol(f=(True, False, 3.0), d=(True, True, 12.0), b=(False, False, 30.0)), cloud=True)
    v = pc.b5_verdict({"evo-x2": x2, "evo-t2s": t2s, "blade": blade}, prices, seeds=(1, 2), turns=5)
    assert v["verdict"] == "supported", v
    assert v["platforms"]["evo-x2"]["cells"]["m"]["router_on_frontier"]
    # router dominated on one platform: b has full history, same rules, faster, zero USD
    t2s_bad = _rows(_pol(c=(True, False, 4.0), d=(True, True, 12.0), b=(True, True, 11.0)), cloud=True)
    v = pc.b5_verdict({"evo-x2": x2, "evo-t2s": t2s_bad, "blade": blade}, prices, seeds=(1, 2), turns=5)
    assert v["verdict"].startswith("refuted (router dominated in evo-t2s/m")
    # one fixed setting on the frontier everywhere (a is the fastest on every platform)
    same = _rows(_pol(a=(True, False, 2.0), d=(True, True, 12.0), b=(True, True, 30.0)), cloud=True)
    v = pc.b5_verdict({"evo-x2": same, "evo-t2s": same, "blade": same}, prices, seeds=(1, 2), turns=5)
    assert v["verdict"].startswith("refuted (fixed setting(s)") and "a" in v["fixed_on_frontier_everywhere"]
    # missing platforms: inconclusive (the router not dominated where it ran)
    v = pc.b5_verdict({"evo-x2": x2}, prices, seeds=(1, 2), turns=5)
    assert v["verdict"] == "inconclusive (platforms not complete: evo-t2s, blade)"
    assert v["platforms"]["blade"]["status"] == "not run"


def test_recall_summary():
    rows = [{"record": "b5_recall", "model_id": "m", "precision": "q8_0", "num_ctx": 8192, "warmup": True,
             "outcome": "ok", "score": {"recall": 0.0}},
            {"record": "b5_recall", "model_id": "m", "precision": "q8_0", "num_ctx": 8192, "warmup": False,
             "outcome": "ok", "score": {"recall": 0.5}, "truncated": False},
            {"record": "b5_recall", "model_id": "m", "precision": "q8_0", "num_ctx": 8192, "warmup": False,
             "outcome": "ok", "score": {"recall": 1.0}, "truncated": True}]
    s = pc.recall_summary(rows)["m"]["q8_0"]
    assert s["recall_mean"] == 0.75 and s["calls"] == 2 and s["truncated"] == 1


# ── estimates, dry run, queue ───────────────────────────────────────────────────────────────────────

needs_results = pytest.mark.skipif(not (REPO / "results" / "x2_r2_real_v1.jsonl").exists(),
                                   reason="R2 result files not synced")


@needs_results
def test_estimates_are_labelled_and_cover_every_platform():
    for plat in pc.PLATFORMS:
        e = pc.estimate_platform(plat)
        assert e["label"].startswith("ESTIMATE") and e["total_h"] > 0
        for m in e["models"].values():
            assert not m["unknown_minutes"]
    x2 = pc.estimate_platform("evo-x2")
    assert x2["models"]["llama3.1:8b"]["aliases"] == {"b": "a", "e": "a"}
    b = pc.estimate_platform("blade", deadline_h=3.0)
    assert 0 < len(b["sessions_within_deadline"]) < b["sessions_total"]


@needs_results
def test_dry_run_prints_plan_and_touches_nothing(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "host_config", None)    # a dry run must not import it
    assert pc.main(["--dry-run", "--platform", "evo-x2"], advance=False) == "dry-run"
    out = json.loads(capsys.readouterr().out)
    assert out["label"].startswith("DRY RUN") and "ESTIMATE" in out["label"]
    assert out["platforms"]["evo-x2"]["queue_entry"][1:] == ["b5\\policy_compare.py", "--platform", "evo-x2", "--out",
                                                             "C:\\apu\\ovn\\results\\x2_policy_compare.jsonl"]


def test_queue_insert_appends_and_refuses_twice(monkeypatch):
    state = {"items": [{"id": "x2_r2_scale_real", "status": "pending"},
                       {"id": "x2_r2_family_mistral_small32_real", "status": "pending"},
                       {"id": "x2_evalset_replay_val_v1", "status": "pending"}], "pause": []}
    tq = types.SimpleNamespace(
        set_pause=lambda r: state["pause"].append(("set", r)), clear_pause=lambda: state["pause"].append(("clear",)),
        read_queue=lambda: [dict(i) for i in state["items"]], write_queue=lambda it: state.update(items=it))
    monkeypatch.setitem(sys.modules, "t2s_queue", tq)
    pc.queue_insert(log=lambda m: None)
    assert [i["id"] for i in state["items"]][-1] == pc.JOB_ID
    assert state["pause"][0][0] == "set" and "clear_pause" in state["pause"][0][1] and state["pause"][-1] == ("clear",)
    with pytest.raises(ValueError):
        pc.queue_insert(log=lambda m: None)
    assert state["pause"][-1] == ("clear",)                   # cleared even when the insert refuses


# ── the job with fakes ──────────────────────────────────────────────────────────────────────────────

class FakeServer:
    def __init__(self, bad_kv=None):
        self.env, self.starts, self.stopped, self.bad_kv = None, [], 0, bad_kv

    def start(self, env):
        self.env = dict(env)
        self.starts.append(dict(env))

    def stop(self):
        self.stopped += 1

    def log_text(self):
        kv = (self.env or {}).get("OLLAMA_KV_CACHE_TYPE")
        if not kv:
            return 'msg="server config" env="map[]"'
        seen = "f16" if kv == self.bad_kv else kv
        return f'msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true OLLAMA_KV_CACHE_TYPE:{kv}]"\n' \
               f"llama_kv_cache: type_k = '{seen}'"


class FakeOllama(FakeRuntime):
    """R2 answers from good_policy; windows: native 8192, default 4096, 8192 starts but only 4096 is fully in GPU."""
    NATIVE, DEFAULT, GPU_MAX = 8192, 4096, 4096

    def __init__(self, avail):
        super().__init__(None, loaded=None)
        self.avail, self._ags, self.ps, self.options_extra = avail, {}, None, None
        self.client = types.SimpleNamespace(get_ps=lambda: {"models": [self.ps] if self.ps else []})

    def _load(self, model, num_ctx):
        ctx = num_ctx or self.DEFAULT
        self.ps = {"name": model, "context_length": ctx, "size": 100, "size_vram": 100 if ctx <= self.GPU_MAX else 60}
        self.loaded = ctx

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        code = messages[0]["content"]
        if code not in self._ags:
            self._ags[code] = next(a for a in (ag.build_agent_session(s, 40) for s in ag.SEEDS)
                                   if a.system_prompt == code)
        self.policy = good_policy(self._ags[code])
        self._load(model, num_ctx)
        resp = super().chat(model, messages, num_ctx, tools, think, extra)
        resp["prompt_eval_count"] = 1000
        return resp

    def _post(self, path, body):
        if path == "/api/show":
            return 200, {"model_info": {"llama.context_length": self.NATIVE}}, None, 0.01
        ctx = body["options"].get("num_ctx")
        if ctx is not None and ctx > self.NATIVE:
            return 500, None, "alloc failed", 0.01
        self._load(body["model"], ctx)
        return 200, {"message": {"content": "OK"}}, None, 0.01

    def _post_native(self, body):
        self._load(body["model"], body["options"].get("num_ctx"))
        prompt = body["messages"][0]["content"]
        codes = re.findall(r"vault V\d+ is (\w+)\.", prompt)
        return {"outcome": "ok", "status": 200, "message": json.dumps(codes), "prompt_eval_count": ag.est_tokens(prompt),
                "duration_s": 0.1}

    def unload(self, model, wait_s=0):
        self.ps, self.loaded = None, None
        return True

    def loaded_context(self, model):
        return self.ps["context_length"] if self.ps else None

    def available_models(self):
        return self.avail

    def version(self):
        return "0.34.4-fake"

    def render_only(self, *a, **k):
        return "", None


def _job_fakes(monkeypatch, notes, avail=("llama3.1:8b",), refuse=False):
    hc = types.SimpleNamespace(require_host=lambda h: {"hw_id": "evo-x2"},
                               enforce_or_record_interactive_session=lambda cfg: {"logged_in": False})
    pf = {"ok": not refuse, "models_ok": [] if refuse else ["llama3.1:8b"],
          "refused": {"llama3.1:8b": ["llama3.1:8b: no validation rows"]} if refuse else {},
          "rules_in_use": {} if refuse else {"llama3.1:8b": ["rule1_json_keys", "rule5_session_code"]},
          "source": {}}
    monkeypatch.setattr(ag, "validation_preflight_per_model", lambda files, c2, models, neg_arm=None: pf)
    monkeypatch.setattr(pc, "TURNS", 10)
    rt = FakeOllama(list(avail))
    return {"hc": hc, "hostname": "EVO-X2", "runtime": rt,
            "tq": types.SimpleNamespace(advance=notes.append),
            "counter": lambda model, msgs, num_ctx, tools, think, extra=None: ag.transcript_tokens(msgs),
            "measure": lambda m, a, s: {"system_prompt_tokens": 700, "method": "fake"}}, rt


def test_job_end_to_end_with_fakes(tmp_path, monkeypatch):
    notes = []
    deps, rt = _job_fakes(monkeypatch, notes)
    server = FakeServer(bad_kv="q4_0")
    deps["server"] = server
    deps["make_router"] = pc.make_router_factory("evo-x2", 4096, ledger_dir=tmp_path)
    out = tmp_path / "x2_policy_compare.jsonl"
    note = pc.main(["--platform", "evo-x2", "--out", str(out), "--seeds", "20260901", "--b1-models", ""], deps=deps)
    assert note == "completed" and notes == ["completed"] and server.stopped == 1
    rows = ag.read_rows(out)
    res = next(r for r in rows if r["record"] == "b5_resolution")
    assert res["windows"]["D"] == 4096 and res["windows"]["L"] == {"base": 8192, "q8_0": 8192, "q4_0": 8192}
    assert res["windows"]["G"] == 4096
    assert res["configs"]["e"]["alias_of"] == "a" and res["configs"]["h"]["skip"].startswith("KV type not applied")
    sess = [r for r in rows if r["record"] == "r2a_session"]
    assert sorted(pc.policy_of_arm(s["arm_id"]) for s in sess) == ["a", "b", "c", "d", "f", "g"]
    assert all(s["n_turns"] == 10 and s["mode"] == "b5" for s in sess)
    turns = {pc.policy_of_arm(r["arm_id"]): r for r in rows if r["record"] == "r2a_turn" and r["turn_idx"] == 1}
    assert turns["a"]["num_ctx_requested"] is None and turns["b"]["num_ctx_requested"] == 8192
    assert turns["c"]["calls"][0]["client_trim"]["applied"] and turns["f"]["calls"][0]["num_keep"]["sent"] == 700
    routes = [r for r in rows if r["record"] == "b5_route"]
    assert len(routes) == sum(r["n_calls"] for r in rows if r["record"] == "r2a_turn"
                              and pc.policy_of_arm(r["arm_id"]) == "d")
    assert {r["sent_num_ctx"] for r in routes} <= {4096, 8192}
    kv = {r["env"]: r["check"]["valid"] for r in rows if r["record"] == "b5_kv_check"}
    assert kv == {"q8_0": True, "q4_0": False}
    assert [s for s in server.starts if s.get("OLLAMA_KV_CACHE_TYPE") == "q8_0"]
    rec = [r for r in rows if r["record"] == "b5_recall"]
    assert {r["precision"] for r in rec} == {"f16", "q8_0", "q4_0"} and sum(r["warmup"] for r in rec) == 3
    assert all(r["score"]["recall"] == 1.0 for r in rec)
    assert rows[-1]["record"] == "run_end"
    m = pc.metrics_for_platform(rows, seeds=(20260901,), turns=10)["llama3.1:8b"]
    assert set(m["policies"]) == {"a", "b", "c", "d", "e", "f", "g"} and m["balanced_seeds"] == [20260901]
    rep = json.loads((tmp_path / "x2_policy_compare.jsonl.report.json").read_text(encoding="utf-8"))
    assert rep["platform"] == "evo-x2"
    # resumable: a second run skips every completed session
    n_before = len(sess)
    pc.main(["--platform", "evo-x2", "--out", str(out), "--seeds", "20260901", "--b1-models", "", "--no-recall"],
            deps=dict(deps, server=FakeServer(bad_kv="q4_0")))
    assert len([r for r in ag.read_rows(out) if r["record"] == "r2a_session"]) == n_before


def test_job_refuses_without_validation_and_starts_no_server(tmp_path, monkeypatch):
    notes = []
    deps, _ = _job_fakes(monkeypatch, notes, refuse=True)
    server = FakeServer()
    deps.update(server=server, make_router=lambda m: None)
    out = tmp_path / "x2_policy_compare.jsonl"
    note = pc.main(["--platform", "evo-x2", "--out", str(out), "--b1-models", ""], deps=deps)
    assert note.startswith("completed; refused to start") and notes == [note] and server.starts == []
    assert [r["record"] for r in ag.read_rows(out)] == ["run_start", "refused_model", "refused", "run_end"]


def test_job_rejects_wrong_host_and_wrong_prefix(tmp_path, monkeypatch):
    notes = []
    deps, _ = _job_fakes(monkeypatch, notes)
    deps.update(server=FakeServer(), make_router=lambda m: None, hostname="EVO-T2S")
    note = pc.main(["--platform", "evo-x2", "--out", str(tmp_path / "x2_p.jsonl")], deps=deps)
    assert note.startswith("stopped:") and "not EVO-X2" in note and notes == [note]
    with pytest.raises(SystemExit):
        pc.main(["--platform", "evo-x2", "--out", str(tmp_path / "t2s_p.jsonl")], deps=deps)


def test_blade_job_returns_exit_code_and_never_advances(tmp_path, monkeypatch):
    notes = []
    deps, _ = _job_fakes(monkeypatch, notes, refuse=True)
    deps.update(server=FakeServer(), make_router=lambda m: None, hostname="RITZLAPTOP")
    rc = pc.main(["--platform", "blade", "--out", str(tmp_path / "blade_policy_compare_x.jsonl")], deps=deps)
    assert rc == 0 and notes == []


def test_b1_models_added_only_with_a_projection_within_the_cap(tmp_path):
    proj = {"models": {"qwen3:32b": {"tiers": {"ollama_default": {"minutes_per_session": 60.0}}},
                       "llama3.3:70b": {"tiers": {"ollama_default": {"minutes_per_session": 400.0}}}}}
    (tmp_path / "x2_scale_timing_probe.jsonl").write_text(json.dumps({"record": "scale_projection",
                                                                       "projection": proj}) + "\n")
    d = pc.b1_models_to_add("evo-x2", list(pc.B1_MODELS), tmp_path)
    assert d["qwen3:32b"]["add"] and not d["llama3.3:70b"]["add"] and not d["qwen3-30b-a3b-2507"]["add"]
    assert not pc.b1_models_to_add("blade", ["qwen3:32b"], tmp_path)["qwen3:32b"]["add"]


def test_register_rows_pending_until_synced_then_compute(tmp_path):
    sys.path.insert(0, str(REPO))
    import shutil
    import analysis.numbers_register as nr
    entries = {e["claim_id"]: e for e in nr.NUMBER_ENTRIES}
    for cid in ("b5-policy-metrics", "b5-verdict", "b5-kv-recall"):
        assert entries[cid]["pending_ok"] and nr._pending(entries[cid], repo=tmp_path)
    assert not entries["b5-plan-hours"].get("pending_ok")
    (tmp_path / "results").mkdir()
    shutil.copy(REPO / "results" / "cloud_prices_20261008.json", tmp_path / "results")
    rows = _rows(_pol(d=(True, True, 12.0)), seeds=pc.SEEDS, turns=pc.TURNS, cloud=True)
    rows.insert(-1, {"record": "b5_recall", "model_id": "m", "precision": "q8_0", "num_ctx": 8192, "warmup": False,
                     "outcome": "ok", "score": {"recall": 1.0}})
    (tmp_path / "results" / "x2_policy_compare.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert not nr._pending(entries["b5-verdict"], repo=tmp_path)
    v = nr.compute_b5_verdict(tmp_path)
    assert v["value"].startswith("inconclusive (platforms not complete: evo-t2s, blade)")
    assert "evo-x2 m:" in nr.compute_b5_policy_metrics(tmp_path)["value"]
    assert "q8_0 ctx 8192 recall 100" in nr.compute_b5_kv_recall(tmp_path)["value"]


def test_import_check_resolves_the_router_stack(capsys):
    assert pc.main(["--import-check"], advance=False) == "import-check"
    out = json.loads(capsys.readouterr().out)
    assert out["agent_has_options_extra"] and "src.dse.router" in out["modules"]


def test_blade_deadline_constants_agree():
    import blade_hours_estimate as bhe
    import blade_queue as bq
    assert bhe.B5_DEADLINE_H == bq.BLADE_B5_DEADLINE_H


@needs_results
def test_t2s_window_staging_file_is_fresh_and_passes_its_dry_run():
    sys.path.insert(0, str(REPO / "scripts"))
    import t2s_window_stage as ws
    week = json.loads(ws.wq.QUEUE_PATH.read_text(encoding="utf-8"))
    staged = json.loads(ws.STAGE_PATH.read_text(encoding="utf-8"))
    rep = ws.check_staged(staged, [it["id"] for it in week])
    assert rep["ok"] and rep["fresh_build_matches_file"], rep
    assert {it["track"] for it in staged} == {"B5", "B6"}
    d = ws.displacement(staged, week)
    assert d["recommended_cut_h"] <= ws.BUDGET_H and set(ws.RECOMMENDED_CUT) <= {i["id"] for i in week + staged}
    assert all(it["est_basis"].endswith("(ESTIMATE)") for it in staged)
