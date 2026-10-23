"""demo/dashboard four demo views (A silent failure, B device recommendation, C cost, D live escalation).

Parsers and builders run on fake register rows and fake result files written to tmp_path; the endpoint tests use a
cache built from the committed files with the fake DSE backend. No network: the escalation tests inject a fake
completion function into src/cloud/client.py's CloudClient, and a guard fails any test that opens a socket to a
non-loopback address.
"""
import json
import re
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
    assert b5["status"] == views.PENDING_FA  # KV q8_0 + flash attention waits on the flash-attention control
    assert b5["settings"]["kv_precision"]["label"] == views.PENDING_FA
    assert b5["settings"]["flash_attention"]["label"] == views.PENDING_FA
    assert b5["settings"]["window"]["label"] == views.MEASURED
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


def _fa_dv(tmp_path, fa_value, status="VERIFIED"):
    fake_b5_file(tmp_path)
    r = reg(**{"b5-policy-metrics": B5, "b5-kv-recall": KV, "b5-fa-control": fa_value})
    r["b5-fa-control"]["status"] = status
    return views.build_device(r, tmp_path, {"sessions": []})


def test_device_follows_fa_control_reading(tmp_path):
    tail = "(g)/(h) 10.0 min, (a) 40.0 min, window 8192; rule 1.0, hist 0.75; base-env x; valid True; reading: "
    fa = _fa_dv(tmp_path, f"m1: (i) f16+FA 10.5 min per session (per seed 1: 10.5), {tail}{views.FA_READ_FA} "
                          "(tolerance 10%)")
    r = views.device_recommend(fa, "evo-x2", "m1", "r2_agent")
    assert r["status"] == views.MEASURED and r["settings"]["kv_precision"]["value"] == "f16 (runtime default)"
    assert r["settings"]["flash_attention"]["value"].startswith("on") and r["settings"]["window"]["value"] == 8192
    assert any("10.5 min" in o["text"] for o in r["outcome"]) and "flash attention alone" in r["basis"]
    kv = _fa_dv(tmp_path, f"m1: (i) f16+FA 39.0 min per session (per seed 1: 39.0), {tail}{views.FA_READ_KV}")
    r = views.device_recommend(kv, "evo-x2", "m1", "r2_agent")
    assert r["status"] == views.MEASURED and r["settings"]["kv_precision"] == {
        **r["settings"]["kv_precision"], "value": "q8_0", "label": views.MEASURED}
    assert any(views.FA_READ_KV in c for c in r["caveats"])
    # VERIFIED but the run is incomplete (the 2026-10-21 state): still pending
    unread = _fa_dv(tmp_path, "m1: (i) f16+FA 44.7 min per session (per seed 1: 44.7), (g)/(h) 6.8 min, (a) 43.7 "
                              "min, window 8192; valid False (no run_end); reading: not read (incomplete)")
    r = views.device_recommend(unread, "evo-x2", "m1", "r2_agent")
    assert r["status"] == views.PENDING_FA and "not read yet" in r["settings"]["flash_attention"]["note"]
    # PENDING row with a reading in it: still pending
    pend = _fa_dv(tmp_path, f"m1: (i) f16+FA 10.5 min per session, {tail}{views.FA_READ_FA}", status="PENDING")
    assert views.device_recommend(pend, "evo-x2", "m1", "r2_agent")["status"] == views.PENDING_FA


def test_fa_control_reading_only_when_read():
    unread = reg(**{"b5-fa-control": "llama3.1:8b: (i) f16+FA 44.7 min; valid False (no run_end); reading: not read "
                                      "(incomplete) / qwen3:14b: reading: not read (incomplete)"})
    assert views.fa_control_reading(unread) is None  # VERIFIED row, but no reading: attribution stays open
    read = reg(**{"b5-fa-control": "llama3.1:8b: (i) f16+FA 43.0 min per session, (g)/(h) 6.8 min, (a) 43.7 min, "
                                    "window 131072; valid True; reading: the B5 speedup is KV precision, not flash "
                                    "attention / qwen3:14b: (i) f16+FA None min per session; valid False; reading: "
                                    "not read (incomplete) (tolerance 10%)"})
    assert views.fa_control_reading(read) == "llama3.1:8b: the B5 speedup is KV precision, not flash attention"
    p = views.parse_fa_control(read["b5-fa-control"]["value"])
    assert p["llama3.1:8b"] == {"minutes_i": 43.0, "window": 131072, "valid": True, "reading": views.FA_READ_KV}
    assert p["qwen3:14b"]["minutes_i"] is None and views.fa_control_for(read, "qwen3:14b") is None
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
    assert r["settings"]["window"]["label"] == "measured"
    # pending until b5-fa-control reads for qwen3:14b (VERIFIED but incomplete on 2026-10-21)
    expect = "measured" if views.fa_control_for(app.cache.register, "qwen3:14b") else views.PENDING_FA
    assert r["status"] == expect
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
    # 2026-10-22: the lead carries no figure from the v1 redo, whatever the redo row says
    assert c["lead"]["text"] == "Cost to fix a wrong answer: being measured"
    assert c["lead"]["label"] == views.BEING_MEASURED and c["lead"]["src"] is None
    blob = json.dumps(c)
    assert "0.0283" not in blob and "10.33" not in blob and "137.18" not in blob
    labels = [i["label"] for i in c["items"]]
    assert labels == sorted(labels, key=lambda l: views._ORDER[l])  # measured parts first
    assert labels[0] == views.MEASURED and views.BEING_MEASURED in labels
    assert any("139.4 min" in i["text"] for i in c["items"])
    assert c["router"]["shown"] is False and "refuted" in c["router"]["text"]
    assert not any("saving" in i["what"].lower() and "router" in i["what"].lower() for i in c["items"])
    # a router savings figure appears only behind a VERIFIED non-refuted b5-verdict
    assert views.router_savings_supported(reg(**{"b5-verdict": "supported (router on frontier)"}))
    assert not views.router_savings_supported({})
    pending = reg(**{"b5-verdict": "supported"})
    pending["b5-verdict"]["status"] = "PENDING"
    assert not views.router_savings_supported(pending)
    # missing register rows: no invented numbers (only the 'being measured' placeholders)
    empty = views.build_cost({})
    assert all(i["label"] == views.BEING_MEASURED for i in empty["items"])
    assert not re.search(r"\d\.\d|USD \d", json.dumps(empty["items"]))


REDO_LIVE = ("mode executed; model claude-sonnet-5-5 (prices.json), prompt caching on (cold cache per task); USD per wrong "
             "task mean $0.0201, median $0.0122, total $3.9520 (all input uncached: $13.8989); per tier: ollama_default "
             "n=17 mean $0.0364 median $0.0297 total $0.6193; local regrade reproduces end_state_ok false for 197/197 "
             "(100.0%); cloud correct 79/197 (40.1%), mean 6.89 s per task")
RANK = ("one profile: x; ranking by main-case USD/day (P2: machine capacity; the other rows: time at the full hour "
        "value; a PENDING row is not ranked): P1 > P5 > P3 > P4; unranked: P2, P6, P7")


def test_cost_panel_shows_no_v1_redo_figure():
    c = views.build_cost(reg(**{"cloud-redo-cost-per-wrong-task": REDO_LIVE, "impact-ranking-v2": RANK}))
    assert c["lead"]["text"] == "Cost to fix a wrong answer: being measured"
    blob = json.dumps(c)
    for fig in ("0.0201", "0.0122", "3.9520", "13.8989", "6.89", "79/197", "40.1", "P1 > P5"):
        assert fig not in blob, fig  # neither the v1 redo nor the ranking built on it
    # the register rows behind it are withheld whole in every API response
    red = views.redact_register(reg(**{"cloud-redo-cost-per-wrong-task": REDO_LIVE, "impact-ranking-v2": RANK,
                                       "impact-v2-P1": "cloud redo USD/day 5.47 (USD 0.0149 per wrong task)"}))
    for rid in ("cloud-redo-cost-per-wrong-task", "impact-ranking-v2", "impact-v2-P1"):
        assert red[rid]["redacted"] and "withheld" in red[rid]["value"] and not re.search(r"\d\.\d", red[rid]["value"])


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
    # 2026-10-22 final pass: measured by the full-history redo, with the protocol difference, CI and caveat
    assert c["lead"]["text"].startswith("Fixing a wrong answer in the cloud costs about USD 0.0")
    assert c["lead"]["label"] == "measured" and c["lead"]["src"] == {"register": "cloud-redo-full-context"}
    note = c["lead"]["note"]
    assert "2048" in note and "384" in note and "Wilson CI" in note and "inherited" in note
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
    # the task's full-history redo (measured); never the v1 redo
    assert rd["label"] == "measured" and rd["usd"] > 0 and rd["seconds"] > 0 and rd["cloud_final_text"]
    assert rd["src"]["file"].startswith("results/cloud_redo_full_context_v1") and rd["src"]["line"]
    assert "2048" in rd["note"] and "cloud_redo_v1" not in json.dumps(rd)


def test_full_redo_of_missing_task_is_being_measured(tmp_path):
    from demo.dashboard import escalate
    rd = escalate.full_redo_of(tmp_path, "nope", 1)
    assert rd["label"] == "being measured" and "usd" not in rd
    stub = escalate.Escalator({"available": True, "redo": rd}, ledger=tmp_path / "l.jsonl", env={}).run()
    assert stub["mode"] == "being measured" and stub["usd"] is None and stub["answer"] is None


STUB_MODES = ("recorded redo", "being measured")


def v1_redo_figures() -> list[str]:
    """Strings that would only appear if a figure or answer from results/cloud_redo_v1.jsonl were shown: the run's
    summary (mean, median, total, mean seconds) and the demo task's own row (USD, seconds, tokens, answer)."""
    rows = [json.loads(l) for l in (REPO / "results" / "cloud_redo_v1.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    s = next(r for r in rows if r.get("record") == "run_end")["summary"]
    out = [f"{s['mean_usd']:.4f}", f"{s['median_usd']:.4f}", f"{s['total_usd']:.4f}"]
    if s.get("seconds_mean"):
        out.append(f"{s['seconds_mean']:.2f} s")
    esc = json.loads((REPO / "demo" / "dashboard" / "cache" / "escalation.json").read_text(encoding="utf-8"))
    task = next(r for r in rows if r.get("record") == "cloud_redo_task" and r.get("attempt_id") == esc["attempt_id"]
                and r.get("turn_idx") == esc["turn_idx"])
    out += [f"{task['usd']:.4f}", f"{task['usd_all_uncached']:.4f}", f"{task['input_tokens']:,}"]
    if task.get("seconds") is not None:
        out.append(f"{task['seconds']:.1f} s")
    if task.get("cloud_final_text"):
        out.append(task["cloud_final_text"][:60])
    return out


def test_no_v1_redo_figure_anywhere(app, tmp_path):
    figs = v1_redo_figures()
    assert len(figs) >= 6
    stub = make_escalator(app, tmp_path, fake_completion([], []), env_live=False).run()
    bodies = {p: json.dumps(get(app, p)) for p in ("/api/cost", "/api/story", "/api/escalation", "/api/register",
                                                   "/api/register?id=cloud-redo-cost-per-wrong-task")}
    bodies["POST /api/escalate (stub)"] = json.dumps(stub)
    for name in ("cost.json", "story.json", "escalation.json"):
        bodies["cache/" + name] = (REPO / "demo" / "dashboard" / "cache" / name).read_text(encoding="utf-8")
    for where, body in bodies.items():
        for f in figs:  # whole-figure match: "4.8 s" must not match inside "314.8 s"
            assert not re.search(r"(?<![\w.])" + re.escape(f), body), (where, f)
    assert "cloud-redo-full-context" in bodies["/api/cost"]  # the measured basis that replaced it


def test_fix_rate_only_with_ci_and_inherited_caveat(app):
    """Wherever a cloud fix rate is shown (story, cost panel), the same text carries its 95% Wilson CI and the
    inherited-actions caveat."""
    texts = [i["text"] for st in get(app, "/api/story")["steps"] for i in st["items"]]
    c = get(app, "/api/cost")
    texts += [c["lead"]["text"], c["lead"].get("note") or ""] + [i["text"] for i in c["items"]]
    shown = [t for t in texts if re.search(r"\d% fixed|fixed \d+/\d+", t)]
    assert shown, "the fix rate should now be shown"
    for t in shown:
        assert "Wilson CI" in t and "inherited the local model's earlier executed actions" in t, t


def test_escalation_stub_when_live_off(app, tmp_path):
    rec = []
    e = make_escalator(app, tmp_path, fake_completion([], rec), env_live=False)
    r = e.run()
    rd = app.cache.escalation["redo"]
    assert r["mode"] == "recorded redo" and r["label"] == "measured" and "live calls are off" in r["reason"]
    assert (r["usd"], r["seconds"], r["answer"], r["usd_src"]) == (rd["usd"], rd["seconds"], rd["cloud_final_text"],
                                                                    rd["src"])
    assert rec == [] and not (tmp_path / "ledger.jsonl").exists()


def test_escalation_one_live_call_then_stub(app, tmp_path):
    rec = []
    expected = app.cache.escalation["expected_calls"]
    calls = [{"name": "log_event", "arguments": {"event": "booking"}}] + expected
    e = make_escalator(app, tmp_path, fake_completion(calls, rec))
    r = e.run()
    assert r["mode"] == "executed" and r["label"] == "measured", r.get("reason")
    assert r["usd"] > 0 and r["seconds"] >= 0 and r["answer"] == '{"answer": "booked"}'
    assert [c["name"] for c in r["tool_calls"]] == [c["name"] for c in calls]
    assert "cloud_end_state_ok" not in r  # cost, seconds, answer; no graded result (fix rate under diagnosis)
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
        "/api/cost", "/api/escalation", "/api/story")}
    dump["POST"] = {"mode": "executed", "label": "measured", "reason": "one live call", "answer": "",
                    "tool_calls": [{"name": "create_event"}], "seconds": 14.0, "usd": 0.0174, "usd_basis": "ledger",
                    "live_requested": True, "key_present": True, "cap_usd": 10.0,
                    "live_calls_used": 1, "live_calls_allowed": 1, "ledger": "demo/ledger/x.jsonl",
                    "ledger_total_usd": 0.0174}
    (tmp_path / "dump.json").write_text(json.dumps(dump), encoding="utf-8")
    r = run_hidden([node, str(REPO / "tests" / "dashboard_views_smoke.js"),
                    str(REPO / "demo" / "dashboard" / "static" / "views.js"), str(tmp_path / "dump.json")],
                   capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert all(not v.startswith("Error") and v for v in out.values() if isinstance(v, str)), out
    assert "4096" in out["#silent-body"] and "32768" in out["#silent-body"]
    lead = get(app, "/api/cost")["lead"]
    assert lead["text"] in out["#cost-body"]
    assert "not shown" in out["#cost-body"]
    assert "KV cache precision" in out["#device-body"]
    assert "USD 0.0174" in out["#esc-result"] and "14.0 s" in out["#esc-result"]
    for k in ("#cost-body", "#esc-task", "#esc-result"):  # no cloud fix rate or graded cloud result on screen
        assert "40.1%" not in out[k] and "Graded end state" not in out[k], k  # never the v1 fix rate
    assert "Wilson CI" in out["#cost-body"] and "inherited" in out["#cost-body"]
    # story mode: starts on step 1 in story mode, Next walks 1..5, Back steps back, Next past 5 leaves story mode
    w = out["story"]
    assert [s["indicator"].split(",")[0] for s in w[:5]] == [f"Step {i} of 5" for i in range(1, 6)]
    assert w[0]["story_on"] and w[0]["back_disabled"] and not w[1]["back_disabled"]
    assert w[4]["next"].startswith("Finish") and w[5]["indicator"].startswith("Step 4 of 5")
    assert w[0]["panel"] == "#silent-body" and w[4]["panel"] == "#esc-wrap"
    assert views.ROUTE_TAKEAWAY in w[2]["items"]
    # the slider (JS routeAt) agrees with views.route_at at 0, 5, 12 and 20 points, and renders at the default 5
    route = next(st for st in get(app, "/api/story")["steps"] if st["panel"] == "route")["route"]
    for t in ("0", "5", "12", "20"):
        assert [d["route"] for d in out["route"][t]] == [views.route_at(e, float(t))["route"] for e in route["entries"]]
    assert "5.0" in out["routeHTML5"] and "measured by R2 session pass rate (not B3 task accuracy)" in out["routeHTML5"]
    assert w[6]["story_on"] is False  # free exploration after step 5
    assert "5/5" in w[0]["items"] and "42.0%" in w[0]["items"] and "30.0%" in w[1]["items"]
    assert "Fix after the fact: about 2 cents" in w[3]["items"] and "40.1" not in w[3]["items"]
    assert "ceiling, no such detector" in w[3]["items"]
    assert out["nav"] == [{"index": 0, "explore": False}, {"index": 1, "explore": False},
                          {"index": 4, "explore": True}, {"index": 4, "explore": False}, {"index": 3, "explore": True}]


# ── story mode (backend) ──

def test_story_steps_from_committed_files(app):
    s = get(app, "/api/story")
    steps = s["steps"]
    assert [st["n"] for st in steps] == [1, 2, 3, 4, 5] and s["total_minutes"] == 5
    assert [st["panel"] for st in steps] == ["silent", None, "route", "tradeoff", "escalate"]
    allowed = set(views.LABELS) | {None}
    for st in steps:
        assert st["items"], st["title"]
        for it in st["items"]:
            assert it["label"] in allowed, it
            if it["label"] not in (None, views.BEING_MEASURED):
                assert it["src"], it  # every labelled number links to its source
    text = " ".join(it["text"] for st in steps for it in st["items"])
    assert "4096" in text and "32768" in text and "5/5" in text and "11.3%" in text and "42.0%" in text
    assert "30.0%" in text and "two-call pattern" in text and "resists rule loss" in text
    money = steps[3]["items"]
    # step 4 leads with the measured trade-off (the prepared IMPACT.md text), then the fix-after-the-fact line
    assert money[0]["label"] == "measured" and money[0]["src"] == {"register": "hybrid-savings-tradeoff"}
    assert money[0]["text"].startswith("Local saves money")
    fix = next(i for i in money if i["text"].startswith("Fix after the fact: about 2 cents"))
    assert fix["label"] == "measured" and fix["src"] == {"register": "cloud-redo-full-context"}
    assert "mean USD 0.0227" in fix["text"] and "84.8% fixed" in fix["text"] and "Wilson CI 79.1% to 89.1%" in fix["text"]
    assert "28 of the 30 unfixed inherited" in fix["text"] and "decide up front instead" in fix["text"]
    assert "2048" in fix["text"]
    cpu = next(i for i in money if i["text"].startswith("evo-t2s Intel CPU at 4096"))
    assert "MORE than all-cloud" in cpu["text"] and cpu["label"] == "projected"
    assert any("interim" in i["text"] and "claude-haiku-5-5" in i["text"] for i in money)
    assert any(i["label"] == "priced, not executed" and "USD 2.53 median" in i["text"] for i in money)
    late = next(i for i in money if i["text"].startswith("Late full-history escalation"))
    assert late["label"] in ("being measured", "measured")
    # the ceiling is never a headline item and never an option in the trade-off table
    assert not any(i["text"].lower().startswith("ceiling") for i in money)
    t = steps[3]["tradeoff"]["featured"]
    assert set(t) >= {"all_cloud", "naive_local", "hybrid_flag", "ceiling"} and t["model"] == "claude-sonnet-5-5"
    assert t["naive_local"]["silent"] == 10 and t["device_label"] == "measured"
    assert "40.1" not in text and "savings" not in text.lower()
    # step 3: the decision at session start and the kept-local label; step 5: the recorded full-history redo
    assert views.ROUTE_TAKEAWAY in text and "measured by R2 session pass rate (not B3 task accuracy)" in text
    assert any(i["text"].startswith("Recorded full-history cloud redo of this task") for i in steps[4]["items"])


def test_story_tradeoff_parsers():
    row = ("x / evo-x2 (ollama_default, default = best) [measured] vs claude-sonnet-5-5, B3 v2 tasks (30 tasks x 3 "
           "seeds), median [measured; accuracy measured]: all-cloud saves 0, accuracy 90/90 (100.0%); naive local saves "
           "30.61 / 306.10 / 3,060.99, accuracy 80/90 (88.9%), 10 wrong answers silent; hybrid flag saves 30.61 / 306.10 "
           "/ 3,060.99, accuracy 80/90 (88.9%), escalated 0, 10 wrong answers silent; ceiling, no such detector: saves "
           "23.04 / 230.43 / 2,304.29, accuracy 90/90 (100.0%) / evo-t2s Intel CPU (default, loaded window 4096) "
           "[projected] vs claude-sonnet-5-5, B3 v2 tasks (30 tasks x 3 seeds), median [measured]: all-cloud saves 0, "
           "accuracy 90/90 (100.0%); naive local saves 30.61 / 306.10 / 3,060.99, accuracy 53/90 (58.9%), 37 wrong "
           "answers silent; hybrid flag saves -35.98 / -359.80 / -3,597.97, accuracy 90/90 (100.0%), escalated 88, 0 "
           "wrong answers silent; ceiling, no such detector: saves 2.61 / 26.12 / 261.20, accuracy 90/90 (100.0%)")
    t = views.parse_tradeoff("escalation cost interim for claude-haiku-5-5: x; " + row)
    assert [r["device"] for r in t] == ["evo-x2", "evo-t2s Intel CPU"]
    assert t[0]["naive_local"]["saves"][1000] == 306.10 and t[0]["ceiling"]["saves"][10000] == 2304.29
    assert t[1]["hybrid_flag"]["saves"][1000] == -359.80 and t[1]["hybrid_flag"]["escalated"] == 88
    assert not any(r["escalation_interim"] for r in t)  # interim applies to haiku only
    fr = views.parse_full_redo("model claude-sonnet-5-5, x; part 2: max_tokens 2048 (max_tokens 2048 per cloud call vs "
                               "the local run's 384); fixed 167/197 (84.8%) of the wrong tasks, 95% Wilson CI over tasks "
                               "[79.1%, 89.1%] (x); USD per task (cache-priced) mean $0.0227, median $0.0169, total "
                               "$4.4721; seconds per task mean 6.54, median 5.89; still wrong by class: "
                               "thinking_empty_at_cap 0, earlier_state_repair 28, cloud_model_error 2")
    assert (fr["fixed"], fr["n"], fr["ci_lo"], fr["ci_hi"], fr["mean_usd"], fr["max_tokens"]) == (
        167, 197, 79.1, 89.1, 0.0227, 2048)
    assert fr["classes"] == {"thinking_empty_at_cap": 0, "earlier_state_repair": 28, "cloud_model_error": 2}
    line = views.fix_after_the_fact(fr)
    assert line.startswith("Fix after the fact: about 2 cents (mean USD 0.0227 per task, cache-priced), 84.8% fixed "
                           "(95% Wilson CI 79.1% to 89.1%), but executed actions cannot be undone; decide up front")
    assert "28 of the 30 unfixed inherited" in line
    assert views.optional_row({}, "hybrid-route-at-start")["label"] == "being measured"
    cfg = views.parse_t2s_config("naive: Intel CPU, Ollama default qwen3:14b: loaded window 4096, gated silent 5/5 "
                                 "sessions, kept local (measured by R2 session pass rate (not B3 task accuracy)) 0/5 = "
                                 "0.00; vs claude-sonnet-5-5: saves USD -37.49 at 100/day, -374.93 at 1,000/day, x")
    assert cfg["cpu|qwen3:14b"]["kept"] == 0 and cfg["cpu|qwen3:14b"]["saves_1000"]["claude-sonnet-5-5"] == -374.93


def test_device_decision_and_kept_local_label(app):
    x2 = get(app, "/api/device/recommend?device=evo-x2&model=qwen3:14b&workload=r2_agent")
    # the decision follows hybrid-route-at-start: at matched quality the measured answer is cloud from the start,
    # with the tolerance at which the device would route local
    assert x2["decision"]["route"] == "cloud" and x2["status"] == "measured"
    assert x2["decision"]["src"] == {"register": "hybrid-route-at-start"}
    tol = {t["model"]: t["tolerance_local"] for t in x2["decision"]["tolerances"]}
    assert tol == {"claude-haiku-5-5": 7.8, "claude-sonnet-5-5": 11.1}
    igpu = get(app, "/api/device/recommend?device=evo-t2s-igpu&model=qwen3:14b&workload=r2_agent")["decision"]
    assert {t["model"]: t["tolerance_local"] for t in igpu["tolerances"]} == {"claude-haiku-5-5": 16.7,
                                                                               "claude-sonnet-5-5": 20.0}
    assert all("R2 session pass rate" in t["basis"] for t in igpu["tolerances"])
    # the verified flash-attention control reads KV precision: q8_0 measured, with that attribution
    assert x2["settings"]["kv_precision"] ["value"] == "q8_0" and x2["settings"]["kv_precision"]["label"] == "measured"
    assert any("KV precision, not flash attention" in c for c in x2["caveats"])
    cpu = get(app, "/api/device/recommend?device=evo-t2s-cpu&model=qwen3:14b&workload=r2_agent")
    assert cpu["decision"]["route"] == "cloud"
    assert any("measured by R2 session pass rate (not B3 task accuracy)" in o["text"] for o in cpu["outcome"])
    assert cpu["kept_local"]["label"] == "measured by R2 session pass rate (not B3 task accuracy)"
    assert cpu["decision"]["route_cost"]["label"] == "not measured"  # the route row covers the iGPU, not the CPU


def test_route_and_late_escalation_parsers():
    route = views.parse_route_at_start(
        "rule x / evo-x2 vs claude-haiku-5-5: recommended ollama_default, default = best, local accuracy 80/90 (88.9%) "
        "[measured, B3 task accuracy] vs all-cloud 87/90 (96.7%), gap 7.8 points (routes local at tolerance >= 7.8); "
        "tolerance 0: cloud, saves 0.00 / evo-t2s vs claude-sonnet-5-5: recommended Intel iGPU, Ollama default, loaded "
        "window 32768, local accuracy 4/5 (80.0%) [measured by R2 session pass rate (not B3 task accuracy)] vs "
        "all-cloud 90/90 (100.0%), gap 20.0 points (routes local at tolerance >= 20.0); tolerance 0: cloud, saves 0")
    assert [(r["device"], r["model"], r["tolerance_local"], r["at_matched"]) for r in route] == [
        ("evo-x2", "claude-haiku-5-5", 7.8, "cloud"), ("evo-t2s", "claude-sonnet-5-5", 20.0, "cloud")]
    le = views.parse_late_escalation(
        "claude-sonnet-5-5, USD per task: x; ratio 2.22x on the mean, 1.87x on the median; matched on (seed, task_id) "
        "where both exist: 119 redo rows over 53 distinct tasks, redo mean $0.0226 vs fresh mean $0.0129, ratio 1.76x "
        "mean, 1.64x median; incomplete")
    assert (le["matched_mean"], le["matched_median"], le["unmatched_mean"], le["unmatched_median"],
            le["matched_rows"], le["matched_tasks"]) == (1.76, 1.64, 2.22, 1.87, 119, 53)
    assert views.late_escalation_text(le).startswith("Late full-history escalation costs 1.76x a fresh cloud run")
    assert views.parse_late_escalation("") is None and views.parse_route_at_start("") == []


def test_story_step3_tolerance_slider(app):
    st = get(app, "/api/story")["steps"][2]
    text = " ".join(i["text"] for i in st["items"])
    assert views.ROUTE_TAKEAWAY == ("How much accuracy gap you accept decides where it runs; we measure that gap per "
                                    "device.")
    assert st["items"][0]["text"] == views.ROUTE_TAKEAWAY and "being measured" not in text
    r = st["route"]
    assert r["default_tolerance"] == 5 and (r["min"], r["max"]) == (0, 20) and r["volume"] == 1000
    assert r["src"] == {"register": "hybrid-route-at-start"} and "not measured" in r["not_measured"]
    gaps = {(e["device"], e["model"]): e["tolerance_local"] for e in r["entries"]}
    assert gaps == {("evo-x2", "claude-haiku-5-5"): 7.8, ("evo-x2", "claude-sonnet-5-5"): 11.1,
                    ("evo-t2s", "claude-haiku-5-5"): 16.7, ("evo-t2s", "claude-sonnet-5-5"): 20.0}
    for e in r["entries"]:
        # the decision flips exactly at each entry's gap; cloud saves 0, local saves the tier's whole bill
        below, at = views.route_at(e, e["tolerance_local"] - 0.1), views.route_at(e, e["tolerance_local"])
        assert below == {"route": "cloud", "saves_1000": 0.0}
        assert at["route"] == "local" and at["saves_1000"] == e["saves_local"]["1000"] > 0
        assert e["r2_basis"] == (e["device"] == "evo-t2s")
        if e["r2_basis"]:
            assert e["basis"] == "measured by R2 session pass rate (not B3 task accuracy)"
    at5 = {(e["device"], e["model"]): views.route_at(e, 5)["route"] for e in r["entries"]}
    assert set(at5.values()) == {"cloud"}
    at12 = {k: views.route_at(e, 12)["route"] for k, e in zip(gaps, r["entries"])}
    assert at12[("evo-x2", "claude-haiku-5-5")] == "local" and at12[("evo-x2", "claude-sonnet-5-5")] == "local"
    assert at12[("evo-t2s", "claude-haiku-5-5")] == "cloud"
    assert all(views.route_at(e, 20)["route"] == "local" for e in r["entries"])


def test_route_parser_picks_up_a_new_local_option():
    seg = ("evo-x2 llama3.3:70b vs claude-sonnet-5-5: recommended ollama_default, local accuracy 86/90 (95.6%) "
           "[measured, B3 task accuracy] vs all-cloud 90/90 (100.0%), gap 4.4 points (routes local at tolerance >= "
           "4.4); tolerance 0: cloud, saves 0.00 / 0.00 / 0.00, accuracy 90/90 (100.0%); tolerance 5: local, saves "
           "30.61 / 306.10 / 3,060.99, accuracy 86/90 (95.6%)")
    r = views.parse_route_at_start("rule / " + seg)
    assert len(r) == 1 and r[0]["device"] == "evo-x2 llama3.3:70b" and r[0]["saves_local"]["1000"] == 306.10
    assert views.route_at(r[0], 5) == {"route": "local", "saves_1000": 306.10}


def test_story_step4_late_escalation(app):
    s4 = " ".join(i["text"] for i in get(app, "/api/story")["steps"][3]["items"])
    assert "1.76x" in s4 and "1.64x" in s4 and "2.22x" in s4 and "being measured" not in s4
    blade = get(app, "/api/device/recommend?device=blade&model=qwen3:14b&workload=r2_agent")
    assert blade["decision"]["label"] == "not measured"


def test_story_parsers():
    assert views.parse_scale("x; qwen3:32b: R4096 2/3 (window exceeded 3/3), H32768 1/3; verdict mixed / "
                             "llama3.3:70b: R4096 0/3 (gated 3/3); resists rule loss; verdict alternative "
                             "(resists rule loss)") == [
        {"model": "qwen3:32b", "silent": 2, "n": 3, "verdict": "mixed", "note": None},
        {"model": "llama3.3:70b", "silent": 0, "n": 3, "verdict": "alternative", "note": "resists rule loss"}]
    rc = views.parse_reconciliation_rule2("v1: per seed 1 rule2 40/40 / 2 rule2 40/40 // mitigation: per seed 1 rule2 "
                                          "24/40 / 2 rule2 25/40 // first trimmed rule2 failure seed 1 turn 3: call 1 "
                                          "tools trimmed ['lookup_fact'], untrimmed v1 ['log_event', 'lookup_fact']")
    assert rc["v1"] == [(40, 40), (40, 40)] and rc["trimmed"] == [(24, 40), (25, 40)]
    assert rc["first_failure"]["turn"] == 3
    tc = views.parse_task_cost("USD per real agent task, priced, not executed: long-set (n=36 tasks, median 62 steps; "
                               "x): claude-sonnet-5-5: none $3.4761 / $6.0913, realistic $1.5367 / $2.6348, best "
                               "$0.3378 / $0.5060 | short-set (n=600 tasks, median 6 steps)")
    assert tc["median_steps"] == 62 and tc["models"]["claude-sonnet-5-5"]["realistic"] == (1.5367, 2.6348)
    assert tc["priced"] is True
    assert views.parse_task_cost("") == {}


def test_fix_rate_redacted_everywhere(app):
    r = {"cloud-redo-cost-per-wrong-task": {"id": "x", "status": "VERIFIED", "value": "mean $0.0201; cloud correct "
                                                                                        "79/197 (40.1%), mean 6.89 s"}}
    red = views.redact_register(r)["cloud-redo-cost-per-wrong-task"]
    assert "79/197" not in red["value"] and "withheld" in red["value"] and red["redacted"] is True
    assert "6.89" not in red["value"]  # 2026-10-22: the whole v1 redo row is withheld
    assert r["cloud-redo-cost-per-wrong-task"]["value"].count("79/197") == 1  # the input is not mutated
    for path in ("/api/register?id=cloud-redo-cost-per-wrong-task", "/api/cost", "/api/story", "/api/escalation",
                 "/api/register"):
        body = json.dumps(get(app, path))
        assert not re.search(r"cloud correct \d+/\d+", body), path


def test_register_row_with_pipe_in_value(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("| claim id | status | value | n | reported | files | script | commit | date |\n"
                 "| a | VERIFIED | left part | right part | 3 | x | f1; f2 | s | c | d |\n", encoding="utf-8")
    r = build_cache.parse_register(p)
    assert r["a"]["value"] == "left part | right part" and r["a"]["files"] == ["f1", "f2"]


def test_recommendation_hides_router_savings(app):
    r = get(app, "/api/recommend")
    assert r["savings_shown"] is False and r["savings_vs_all_cloud_usd_per_1k"] is None
    assert "b5-verdict" in r["savings_note"]
