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
