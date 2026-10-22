"""demo/dashboard four demo views (A silent failure, B device recommendation, C cost, D live escalation).

Parsers and builders run on fake register rows and fake result files written to tmp_path; the endpoint tests use a
cache built from the committed files with the fake DSE backend. No network: the escalation tests inject a fake
completion function into src/cloud/client.py's CloudClient, and a guard fails any test that opens a socket to a
non-loopback address.
"""
import json
import socket
from pathlib import Path

import pytest

from demo.dashboard import build_cache, data, server, views

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    real = socket.socket.connect

    def guard(self, addr):
        host = addr[0] if isinstance(addr, tuple) else addr
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"network access attempted: {addr}")
        return real(self, addr)

    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture(scope="module")
def fake_backend():
    return data.load_backend(force_fake=True)


@pytest.fixture(scope="module")
def app(tmp_path_factory, fake_backend):
    d = tmp_path_factory.mktemp("views_cache")
    build_cache.build(out_dir=d, backend=fake_backend)
    return server.App(data.Cache(d), fake_backend)


def get(app, target, status=200):
    r = app.handle("GET", target)
    assert r.status == status, (target, r.status, r.body[:300])
    return r.json()


def reg(**rows):
    return {k: {"id": k, "status": "VERIFIED", "value": v, "n": "", "files": []} for k, v in rows.items()}


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


T2S_CPU = ("llama3.1:8b: loaded window [4096], first over window turn [3]; gated 5/5 silent, 5 after window exceeded, "
           "0 baseline-type (gate excludes recall); intact 0/0/0/0/0/0/0/0 / qwen3:14b: loaded window [4096], first "
           "over window turn [3]; gated 4/5 silent, 4 after window exceeded, 0 baseline-type (gate excludes recall); "
           "intact 1/0/0/0/0/0/0/0; pre-registered kill criterion killed=False")
T2S_IGPU = ("llama3.1:8b: loaded window [32768], first over window turn [23]; gated 0/5 silent, 0 after window exceeded,"
            " 0 baseline-type (gate excludes nothing); intact 5/5/5/4/4/0/0/0")
WRONG = ("run finished True; qwen3:14b: valid True; ollama_default: wrong 17/150 (11.3%) (silent 17, surfaced 0), loaded "
         "context [40960]; ollama_ctx_4096: wrong 63/150 (42.0%) (silent 63, surfaced 0), loaded context [4096]; "
         "ollama_ctx_4096 excess over default +30.7 points (95% CI +26.0 to +34.7, 5 vs 5 sessions)")
GATED = ("ollama_default qwen3:8b: 0/5 silent, 0 after window exceeded, 0 baseline-type; intact 5/5/5/4/4/0/0/0; "
         "num_ctx_4096 qwen3:8b: 1/5 silent, 1 after window exceeded, 0 baseline-type; intact 0/0/0/0/0/0/0/0")


# ── A: silent failure ──

def test_parse_t2s_and_wrong_rate_and_gated():
    cpu = views.parse_t2s_real(T2S_CPU)
    assert [m["model"] for m in cpu] == ["llama3.1:8b", "qwen3:14b"]
    assert cpu[0]["loaded_window"] == [4096] and cpu[1]["gated_silent"] == 4 and cpu[1]["intact"][0] == 1
    assert views.parse_t2s_real(T2S_IGPU)[0]["loaded_window"] == [32768]
    w = views.parse_wrong_rate(WRONG)
    assert [t["tier"] for t in w] == ["ollama_default", "ollama_ctx_4096"]
    assert w[0]["excess_points"] is None and w[1]["excess_points"] == 30.7 and w[1]["ci_lo"] == 26.0
    assert all(t["surfaced"] == 0 for t in w)
    g = views.parse_gated_per_model(GATED)
    assert g[1] == {"tier": "num_ctx_4096", "model": "qwen3:8b", "gated_silent": 1, "n": 5, "silent_after_window": 1,
                    "baseline_type": 0, "intact": [0] * 8}
    assert views.parse_t2s_real("reworded row") == [] and views.parse_wrong_rate(None) == []


def test_build_silent_from_fake_files(tmp_path):
    turn = {"record": "r2a_turn", "attempt_id": "a1", "over_loaded_window": True,
            "calls": [{"http_status": 200, "error": None}, {"http_status": 200, "error": None}]}
    bad = dict(turn, calls=[{"http_status": 500, "error": "boom"}])
    sess = {"record": "r2a_session", "attempt_id": "a1", "model_id": "m", "seed": 1, "arm_id": "ollama_default",
            "loaded_context_values": [4096], "first_over_window_turn": 3}
    write_jsonl(tmp_path / views.T2S_FILES["cpu"], [turn, turn, sess])
    write_jsonl(tmp_path / views.T2S_FILES["igpu"], [turn, bad, sess])
    s = views.build_silent(reg(**{"t2s-r2-real-cpu": T2S_CPU, "t2s-r2-real-igpu": T2S_IGPU,
                                  "x2-evalset-tasks-real-v2-wrong-rate": WRONG, "R2-v1b-gated-per-model": GATED}),
                           tmp_path)
    cpu, igpu = s["device_pair"]["cpu"], s["device_pair"]["igpu"]
    assert cpu["label"] == views.MEASURED and cpu["evidence"]["totals"] == {"sessions": 1, "calls": 4,
                                                                            "calls_surfaced_error": 0}
    assert igpu["evidence"]["totals"]["calls_surfaced_error"] == 1
    assert cpu["evidence"]["sessions"][0]["src"] == {"file": views.T2S_FILES["cpu"], "line": 3}
    assert s["wrong_rate"]["parsed"] and s["v1b_per_model"]["parsed"]
    # a missing row degrades to "not measured", never to an invented number
    s2 = views.build_silent({}, tmp_path)
    assert s2["device_pair"]["cpu"]["label"] == views.NOT_MEASURED and s2["wrong_rate"]["tiers"] == []


def test_silent_endpoint_on_committed_files(app):
    s = get(app, "/api/silent")
    cpu, igpu = s["device_pair"]["cpu"], s["device_pair"]["igpu"]
    assert cpu["register_id"] == "t2s-r2-real-cpu" and igpu["register_id"] == "t2s-r2-real-igpu"
    assert {m["model"] for m in cpu["models"]} == {m["model"] for m in igpu["models"]}  # same models
    assert all(m["loaded_window"] == [4096] for m in cpu["models"])
    assert all(m["loaded_window"] == [32768] for m in igpu["models"])
    assert cpu["evidence"]["totals"]["calls_surfaced_error"] == 0  # silent: no call surfaced an error
    tiers = {t["tier"]: t for t in s["wrong_rate"]["tiers"]}
    assert set(tiers) >= {"ollama_default", "ollama_ctx_4096"}
    assert all(t["surfaced"] == 0 and t["silent"] == t["wrong"] for t in tiers.values())
    for tier, fc in s["wrong_rate"]["file_check"].items():  # the file agrees with the register row
        assert fc["wrong_rows"] == tiers[tier]["wrong"] and fc["http_error_rows"] == 0
    assert s["v1b_per_model"]["parsed"] and len(s["v1b_per_model"]["cells"]) == 15
    assert set(s["register"]) >= {"t2s-r2-real-cpu", "R2-v1b-gated-per-model"}


def test_static_views_js_served(app):
    r = app.handle("GET", "/static/views.js")
    assert r.status == 200 and b"renderSilent" in r.body
    assert b"/static/views.js" in app.handle("GET", "/").body


# ── B: per-device recommendation ──

B5 = ("evo-x2 m1: a rule 100.0% hist 50.0% 40.0 min $0.0000; b rule 100.0% hist 50.0% 40.0 min $0.0000 (=a); "
      "g rule 100.0% hist 75.0% 10.0 min $0.0000; frontier ['g']; complete True / evo-t2s: not run yet")
KV = "evo-x2 m1: f16 ctx 8192 recall 50.0% (2 calls; KV check n/a); q8_0 ctx 8192 recall 60.0% (3 calls; KV check True)"


def fake_b5_file(root: Path):
    write_jsonl(root / views.B5_FILE, [
        {"record": "b5_resolution", "model_id": "m1", "windows": {},
         "configs": {"a": {"effective_ctx": 8192, "env": "base", "wrapper": None, "num_ctx_sent": None},
                     "g": {"effective_ctx": 8192, "env": "q8_0", "wrapper": None, "num_ctx_sent": 8192}}},
        {"record": "b5_kv_check", "model_id": "m1", "env": "q8_0",
         "check": {"config_flash_attention": True, "observed": ["q8_0"], "verified": True}}])


def test_device_rules_on_fakes(tmp_path):
    fake_b5_file(tmp_path)
    r = reg(**{"b5-policy-metrics": B5, "b5-kv-recall": KV, "R2-v1b-gated-per-model": GATED,
               "x2-evalset-tasks-real-v2-wrong-rate": WRONG, "t2s-r2-real-cpu": T2S_CPU,
               "t2s-r2-mitigation-cpu": "overall refuted; num_ctx_4096 llama3.1:8b recovered; num_ctx_4096 qwen3:14b "
                                        "not_recovered"})
    r["b5-fa-control"] = {"id": "b5-fa-control", "status": "PENDING", "value": "", "files": []}
    dv = views.build_device(r, tmp_path, {"sessions": [{"model": "qwen3:8b", "advertised_ctx": {
        "ctx": 40960, "file": "f.jsonl", "line": 7}}]})
    b5 = views.device_recommend(dv, "evo-x2", "m1", "r2_agent")
    assert b5["status"] == views.MEASURED
    assert b5["settings"]["kv_precision"]["value"] == "q8_0"
    assert b5["settings"]["flash_attention"]["value"].startswith("on")
    assert "PENDING" in b5["settings"]["flash_attention"]["note"]
    assert b5["settings"]["window"]["value"] == 8192
    assert any("saves 30.0 min" in o["text"] for o in b5["outcome"])
    assert any("60.0%" in o["text"] for o in b5["outcome"])  # recall at the chosen precision
    q8 = views.device_recommend(dv, "evo-x2", "qwen3:8b", "r2_agent")  # no B5 row: the gated R2 windows
    assert q8["settings"]["window"]["value"] == 40960 and q8["settings"]["kv_precision"]["label"] == views.NOT_MEASURED
    b3 = views.device_recommend(dv, "evo-x2", "qwen3:14b", "b3_tasks")
    assert b3["settings"]["window"]["value"] == 40960 and "11.3%" in b3["outcome"][0]["text"]
    assert views.device_recommend(dv, "evo-x2", "llama3.1:8b", "b3_tasks")["status"] == views.NOT_MEASURED
    cpu_l = views.device_recommend(dv, "evo-t2s-cpu", "llama3.1:8b", "r2_agent")
    assert cpu_l["status"] == views.MEASURED and "trimming" in cpu_l["settings"]["context_handling"]["value"]
    cpu_q = views.device_recommend(dv, "evo-t2s-cpu", "qwen3:14b", "r2_agent")
    assert cpu_q["status"] == views.NOT_MEASURED and cpu_q["outcome"]  # shows what failed, recommends nothing
    blade = views.device_recommend(dv, "blade", "qwen3:14b", "r2_agent")
    assert blade["status"] == views.NOT_MEASURED
    assert all(s["value"] == "not measured" for s in blade["settings"].values())
    with pytest.raises(KeyError):
        views.device_recommend(dv, "toaster", "m1", "r2_agent")


def test_fa_control_reading_only_when_read():
    unread = reg(**{"b5-fa-control": "llama3.1:8b: (i) f16+FA 44.7 min; valid False (no run_end); reading: not read "
                                      "(incomplete) / qwen3:14b: reading: not read (incomplete)"})
    assert views.fa_control_reading(unread) is None  # VERIFIED row, but no reading: attribution stays open
    read = reg(**{"b5-fa-control": "llama3.1:8b: valid True; reading: the B5 speedup is KV precision, not flash "
                                    "attention; qwen3:14b: x"})
    assert views.fa_control_reading(read) == "the B5 speedup is KV precision, not flash attention"
    pending = reg(**{"b5-fa-control": "the B5 speedup is flash attention"})
    pending["b5-fa-control"]["status"] = "PENDING"
    assert views.fa_control_reading(pending) is None


def test_single_shot_timeout_floor():
    pts = [{"machine": "evo-x2", "runtime": "llama_server", "model": "m-1", "config_id": "ls", "quality": 1.0,
            "quality_n": 10, "latency_p50_ms": 1000, "timeout_latency_share": 0.11, "excluded_reason": "timeout 0.11"},
           {"machine": "evo-x2", "runtime": "ollama_default", "model": "m:1", "config_id": "od", "quality": 0.9,
            "quality_n": 10, "latency_p50_ms": 2000, "timeout_latency_share": 0.02, "excluded_reason": None}]
    r = views.device_recommend({}, "evo-x2", "m:1", "single_shot", pts, 0.05)
    assert r["outcome"][0]["text"].startswith("quality 0.900") and r["excluded"][0]["config_id"] == "ls"
    none = views.device_recommend({}, "evo-x2", "m:1", "single_shot", [pts[0]], 0.05)
    assert none["status"] == views.NOT_MEASURED and none["excluded"]


def test_device_endpoints_on_committed_files(app):
    o = get(app, "/api/device")
    assert {d["id"] for d in o["devices"]} == {"evo-x2", "evo-t2s-cpu", "evo-t2s-igpu", "blade"}
    assert o["default_max_timeout_share"] == 0.05
    r = get(app, "/api/device/recommend?device=evo-x2&model=qwen3:14b&workload=r2_agent")
    assert r["status"] == "measured" and r["settings"]["window"]["label"] == "measured"
    assert "b5-policy-metrics" in r["register"]
    for k in ("window", "num_keep", "kv_precision", "flash_attention", "context_handling"):
        assert r["settings"][k]["label"] in views.LABELS
    t2s = get(app, "/api/device/recommend?device=evo-t2s-igpu&model=llama3.1:8b&workload=r2_agent")
    assert t2s["settings"]["window"]["value"] == 32768
    tight = get(app, "/api/device/recommend?device=evo-x2&model=llama3.1:8b&workload=single_shot")
    loose = get(app, "/api/device/recommend?device=evo-x2&model=llama3.1:8b&workload=single_shot&max_timeout_share=1")
    assert tight["excluded"] and not loose["excluded"]
    assert get(app, "/api/device/recommend?device=blade&model=qwen3:14b&workload=b3_tasks")["status"] == "not measured"
    get(app, "/api/device/recommend?device=toaster", 404)
    get(app, "/api/device/recommend?max_timeout_share=7", 400)


# ── C: cost panel ──

REDO = ("mode priced, not executed (stub: no API call); model claude-sonnet-5-5 (prices.json), prompt caching on; USD per "
        "wrong task mean $0.0283, median $0.0283, total $5.5681 (all input uncached: $8.7552); per tier: ollama_default "
        "n=17 mean $0.0275 median $0.0299 total $0.4680; ollama_ctx_4096 n=63 mean $0.0281 median $0.0278 total $1.7681")
CONFIG_TIME = ("MEASURED; evo-x2 qwen3:14b: default (a) 214.8 min per session, rule 100.0% hist 62.5%; fastest measured "
               "setting (g) kv_q8_0 75.4 min, rule 100.0% hist 62.5%, window 40960; saved 139.4 min per session (65%), "
               "over 3 seeds")


def test_cost_lead_order_and_router_gate():
    r = reg(**{"cloud-redo-cost-per-wrong-task": REDO, "x2-evalset-tasks-real-v2-wrong-rate": WRONG,
               "cost-cut-config-time": CONFIG_TIME,
               "impact-v2-P1": "profile evo-x2, 40 sessions/day (ASSUMED volume), x; status mixed, cloud part priced, "
                               "not executed; profile cell USD/day 10.33 to 137.18",
               "b5-verdict": "refuted (router dominated in evo-x2/llama3.1:8b)"})
    c = views.build_cost(r)
    assert c["lead"]["text"] == ("Fixing a wrong answer in the cloud costs about USD 0.028 (median per wrong task; "
                                 "mean USD 0.028); the cost is that the failure is silent.")
    assert c["lead"]["label"] == views.PRICED and c["lead"]["src"] == {"register": "cloud-redo-cost-per-wrong-task"}
    assert c["lead"]["cloud_correct"] is None  # the priced run graded nothing
    assert any("cloud part priced, not executed" in i["text"] for i in c["items"])
    labels = [i["label"] for i in c["items"]]
    assert labels == sorted(labels, key=lambda l: views._ORDER[l])  # measured parts first
    assert labels[0] == views.MEASURED and views.PROJECTED in labels
    assert any("139.4 min" in i["text"] for i in c["items"])
    assert c["router"]["shown"] is False and "refuted" in c["router"]["text"]
    assert not any("saving" in i["what"].lower() and "router" in i["what"].lower() for i in c["items"])
    # a router savings figure appears only behind a VERIFIED non-refuted b5-verdict
    assert views.router_savings_supported(reg(**{"b5-verdict": "supported (router on frontier)"}))
    assert not views.router_savings_supported({})
    pending = reg(**{"b5-verdict": "supported"})
    pending["b5-verdict"]["status"] = "PENDING"
    assert not views.router_savings_supported(pending)
    # missing register rows: no lead number, no invented items
    empty = views.build_cost({})
    assert empty["lead"] is None and empty["items"] == []


REDO_LIVE = ("mode executed; model claude-sonnet-5-5 (prices.json), prompt caching on (cold cache per task); USD per wrong "
             "task mean $0.0201, median $0.0122, total $3.9520 (all input uncached: $13.8989); per tier: ollama_default "
             "n=17 mean $0.0364 median $0.0297 total $0.6193; local regrade reproduces end_state_ok false for 197/197 "
             "(100.0%); cloud correct 79/197 (40.1%), mean 6.89 s per task")
RANK = ("one profile: x; ranking by main-case USD/day (P2: machine capacity; the other rows: time at the full hour "
        "value; a PENDING row is not ranked): P1 > P5 > P3 > P4; unranked: P2, P6, P7")


def test_cost_lead_from_executed_redo():
    c = views.build_cost(reg(**{"cloud-redo-cost-per-wrong-task": REDO_LIVE, "impact-ranking-v2": RANK}))
    L = c["lead"]
    assert L["text"] == ("Fixing a wrong answer in the cloud costs about USD 0.012 (median per wrong task; mean USD "
                         "0.020); the cost is that the failure is silent.")
    assert L["label"] == views.MEASURED and L["statistic"] == "median" and L["executed"] is True
    assert (L["cloud_correct"], L["cloud_graded"], L["cloud_correct_pct"], L["seconds_mean"]) == (79, 197, 40.1, 6.89)
    fixed = next(i for i in c["items"] if i["what"] == "Did the cloud fix them?")
    assert "fixed 79 of 197 (40.1%)" in fixed["text"] and fixed["label"] == views.MEASURED
    assert any(i["text"] == "P1 > P5 > P3 > P4" for i in c["items"])


def test_cost_docs_absent_then_present(tmp_path):
    absent = views.doc_summary(tmp_path, views.COST_SUMMARY_DOC)
    assert absent == {"path": views.COST_SUMMARY_DOC, "present": False}
    p = tmp_path / views.COST_SUMMARY_DOC
    p.parent.mkdir(parents=True)
    p.write_text("# Oct 29 cost summary\n\nMeasured: X. Priced: Y.\nSecond line.\n\n## Section one\n\ntext\n",
                 encoding="utf-8")
    d = views.doc_summary(tmp_path, views.COST_SUMMARY_DOC)
    assert d["present"] and d["title"] == "Oct 29 cost summary"
    assert d["headline"] == "Measured: X. Priced: Y. Second line." and d["headline_line"] == 3
    assert d["sections"] == [{"line": 6, "text": "Section one"}]


def test_cost_endpoint_picks_up_summary_without_rebuild(app, tmp_path):
    c = get(app, "/api/cost")
    assert c["lead"]["text"].startswith("Fixing a wrong answer in the cloud costs about USD 0.0")
    assert c["router"]["shown"] is False
    docs = {d["path"]: d for d in c["docs"]}
    assert docs[views.IMPACT_DOC]["present"] is True
    other = server.App(app.cache, app.backend, repo=tmp_path)
    assert {d["path"]: d["present"] for d in get(other, "/api/cost")["docs"]} == {
        views.COST_SUMMARY_DOC: False, views.IMPACT_DOC: False}
    (tmp_path / "docs").mkdir()
    (tmp_path / views.COST_SUMMARY_DOC).write_text("# S\n\nHello.\n", encoding="utf-8")
    assert {d["path"]: d["present"] for d in get(other, "/api/cost")["docs"]}[views.COST_SUMMARY_DOC] is True


# ── D: live escalation (fake provider; no network) ──

FAKE_KEY = "sk-ant-test-0000-not-a-real-key-0000"


def fake_completion(calls, record):
    def fn(*, model_id, messages, api_key, **kw):
        record.append({"model_id": model_id, "n_messages": len(messages), "tools": bool(kw.get("tools")),
                       "provider": kw.get("custom_llm_provider")})
        return {"choices": [{"message": {"content": '{"answer": "booked"}', "tool_calls": [
            {"id": f"t{i}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for i, c in enumerate(calls)]}}], "usage": {"prompt_tokens": 9000, "completion_tokens": 80}}
    return fn


def make_escalator(app, tmp_path, completion, env_live=True, **kw):
    from demo.dashboard import escalate
    from src.cloud import client as cc
    env = {escalate.ENV_LIVE: "1"} if env_live else {}

    def factory(cap, ledger):
        return cc.CloudClient(mode="real", api_key=FAKE_KEY, spend_cap_usd=cap, ledger_path=ledger,
                              role_override="controller", completion_fn=completion)

    return escalate.Escalator(app.cache.escalation, ledger=tmp_path / "ledger.jsonl", env=env,
                              client_factory=factory, **kw)


def test_escalation_task_snapshot(app):
    t = get(app, "/api/escalation")["task"]
    assert t["available"] and t["tier"] == "ollama_ctx_4096" and t["local"]["end_state_ok"] is False
    assert t["local"]["any_http_error"] is False  # the wrong answer came back as a normal answer
    rd = t["redo"]
    assert rd["usd"] > 0 and rd["src"]["file"] == "results/cloud_redo_v1.jsonl" and rd["src"]["line"]
    # the label follows the redo file: measured once it ran on the cloud, priced before that
    assert rd["label"] == ("measured" if rd["mode"] == "executed" else "priced, not executed")
    if rd["mode"] == "executed":
        assert rd["seconds"] is not None and rd["cloud_end_state_ok"] in (True, False)


STUB_MODES = ("recorded redo", "priced, not executed")


def test_escalation_stub_when_live_off(app, tmp_path):
    rec = []
    e = make_escalator(app, tmp_path, fake_completion([], rec), env_live=False)
    r = e.run()
    rd = app.cache.escalation["redo"]
    assert r["mode"] in STUB_MODES and r["label"] == rd["label"] and "live calls are off" in r["reason"]
    assert r["usd"] == rd["usd"] and r["usd_src"] == rd["src"]
    if rd["mode"] == "executed":
        assert r["seconds"] == rd["seconds"] and r["cloud_end_state_ok"] == rd["cloud_end_state_ok"]
    else:
        assert r["seconds"] is None and r["answer"] is None
    assert rec == [] and not (tmp_path / "ledger.jsonl").exists()


def test_escalation_stub_priced_vs_executed_redo(app, tmp_path):
    from demo.dashboard import escalate
    base = dict(app.cache.escalation)
    priced = escalate.Escalator(dict(base, redo={"mode": "priced_not_executed", "label": "priced, not executed",
                                                 "usd": 0.01, "seconds": None, "src": {"file": "f", "line": 1}}),
                                ledger=tmp_path / "l.jsonl", env={}).run()
    assert priced["mode"] == "priced, not executed" and priced["seconds"] is None and priced["cloud_end_state_ok"] is None
    done = escalate.Escalator(dict(base, redo={"mode": "executed", "label": "measured", "usd": 0.012, "seconds": 4.8,
                                               "cloud_end_state_ok": True, "cloud_final_text": "ok",
                                               "src": {"file": "f", "line": 2}}),
                              ledger=tmp_path / "l.jsonl", env={}).run()
    assert done["mode"] == "recorded redo" and done["label"] == "measured"
    assert (done["usd"], done["seconds"], done["cloud_end_state_ok"], done["answer"]) == (0.012, 4.8, True, "ok")


def test_escalation_one_live_call_then_stub(app, tmp_path):
    rec = []
    expected = app.cache.escalation["expected_calls"]
    calls = [{"name": "log_event", "arguments": {"event": "booking"}}] + expected
    e = make_escalator(app, tmp_path, fake_completion(calls, rec))
    r = e.run()
    assert r["mode"] == "executed" and r["label"] == "measured", r.get("reason")
    assert r["cloud_end_state_ok"] is True and r["usd"] > 0 and r["seconds"] >= 0
    assert rec == [{"model_id": app.cache.escalation["cloud_model"], "n_messages": app.cache.escalation["n_messages"],
                    "tools": True, "provider": "anthropic"}]
    ledger = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert FAKE_KEY not in ledger and FAKE_KEY not in json.dumps(r)
    assert r["ledger_total_usd"] == pytest.approx(r["usd"]) and r["cap_usd"] == 10.0
    again = e.run()  # one live call per demo run
    assert again["mode"] in STUB_MODES and "used" in again["reason"] and len(rec) == 1


def test_escalation_failure_falls_back_and_scrubs_key(app, tmp_path):
    def boom(**kw):
        raise RuntimeError(f"auth failed for key {kw['api_key']}")
    from demo.dashboard import escalate
    e = make_escalator(app, tmp_path, boom)
    e.env = {escalate.ENV_LIVE: "1", "ANTHROPIC_API_KEY": FAKE_KEY}
    r = e.run()
    assert r["mode"] in STUB_MODES and "live call failed" in r["reason"]
    assert FAKE_KEY not in json.dumps(r) and "[redacted]" in r["reason"]


def test_escalation_cap_and_endpoint(app, tmp_path):
    from demo.dashboard import escalate
    assert escalate.cap_usd({}) == 10.0
    assert escalate.cap_usd({escalate.ENV_CAP: "50"}) == 10.0  # never above this agent's cap
    assert escalate.cap_usd({escalate.ENV_CAP: "2.5"}) == 2.5
    assert escalate.cap_usd({escalate.ENV_CAP: "junk"}) == 10.0
    other = server.App(app.cache, app.backend, escalator=make_escalator(app, tmp_path, fake_completion([], []),
                                                                         env_live=False))
    assert other.handle("GET", "/api/escalate").status == 405
    r = other.handle("POST", "/api/escalate")
    assert r.status == 200 and r.json()["mode"] in STUB_MODES
    s = get(other, "/api/escalation")["status"]
    assert s["live_requested"] is False and s["cap_usd"] == 10.0 and "key" not in json.dumps(s).replace("key_present", "")


def test_views_js_renders_every_view(app, tmp_path):
    """Run static/views.js under node with a stub DOM, fed by this App's own JSON (skipped without node)."""
    import shutil
    import sys
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    sys.path.insert(0, str(REPO / "harness"))
    from proc_util import run_hidden
    dump = {p.split("?")[0]: get(app, p) for p in (
        "/api/silent", "/api/device", "/api/device/recommend?device=evo-x2&model=qwen3:14b&workload=r2_agent",
        "/api/cost", "/api/escalation")}
    dump["POST"] = {"mode": "executed", "label": "measured", "reason": "one live call", "answer": "",
                    "tool_calls": [{"name": "create_event"}], "seconds": 14.0, "usd": 0.0174, "usd_basis": "ledger",
                    "cloud_end_state_ok": True, "live_requested": True, "key_present": True, "cap_usd": 10.0,
                    "live_calls_used": 1, "live_calls_allowed": 1, "ledger": "demo/ledger/x.jsonl",
                    "ledger_total_usd": 0.0174}
    (tmp_path / "dump.json").write_text(json.dumps(dump), encoding="utf-8")
    r = run_hidden([node, str(REPO / "tests" / "dashboard_views_smoke.js"),
                    str(REPO / "demo" / "dashboard" / "static" / "views.js"), str(tmp_path / "dump.json")],
                   capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert all(not v.startswith("Error") and v for v in out.values()), out
    assert "4096" in out["#silent-body"] and "32768" in out["#silent-body"]
    lead = get(app, "/api/cost")["lead"]
    assert lead["text"] in out["#cost-body"]
    if lead["cloud_correct"] is not None:
        assert f"{lead['cloud_correct']} of {lead['cloud_graded']}" in out["#cost-body"]
    assert "not shown" in out["#cost-body"]
    assert "KV cache precision" in out["#device-body"]
    assert "USD 0.0174" in out["#esc-result"] and "14.0 s" in out["#esc-result"]


def test_recommendation_hides_router_savings(app):
    r = get(app, "/api/recommend")
    assert r["savings_shown"] is False and r["savings_vs_all_cloud_usd_per_1k"] is None
    assert "b5-verdict" in r["savings_note"]
