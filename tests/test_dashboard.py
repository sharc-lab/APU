"""demo/dashboard: one test-client test per endpoint, the side-by-side scenario's data contract, offline-ness,
and (when src/dse/router.py and src/dse/pareto.py expose the agreed interface) the real-backend integration."""
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from demo.dashboard import build_cache, data, naive, server

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "demo" / "dashboard" / "static"


@pytest.fixture(scope="module")
def fake_backend():
    return data.load_backend(force_fake=True)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory, fake_backend):
    d = tmp_path_factory.mktemp("dash_cache")
    build_cache.build(out_dir=d, backend=fake_backend)
    return d


@pytest.fixture(scope="module")
def app(cache_dir, fake_backend):
    return server.App(data.Cache(cache_dir), fake_backend)


def get(app, target, status=200):
    r = app.handle("GET", target)
    assert r.status == status, (target, r.status, r.body[:300])
    return r


def first_session(app):
    return app.cache.r2_sessions["sessions"][0]["key"]


# ── endpoints ──

def test_index_and_static(app):
    r = get(app, "/")
    assert r.content_type.startswith("text/html") and b"/static/app.js" in r.body
    assert get(app, "/static/app.js").content_type.startswith(("text/javascript", "application/javascript"))
    assert get(app, "/static/app.css").content_type.startswith("text/css")
    get(app, "/static/../server.py", 404)
    get(app, "/static/nope.js", 404)


def test_meta(app):
    m = get(app, "/api/meta").json()
    assert m["backend"]["fake"] is True
    assert set(m["machines"]) == {"evo-x2", "evo-t2s"}
    assert any(w["scenario"] for w in m["workloads"])
    assert {h["name"] for h in m["hardware"]} >= {"evo_t2s", "evox2_strix_halo_128gb"}
    assert m["mitigation"]["label"] or m["mitigation"]["measured"]


def test_frontier(app):
    f = get(app, "/api/frontier?hardware=evo-x2").json()
    assert list(f["machines"]) == ["evo-x2"]
    pts = f["machines"]["evo-x2"]["points"]
    assert pts and all(p["src"]["file"].startswith("results/") for p in pts)
    assert f["machines"]["evo-x2"]["frontier_ids"]
    assert all(p["stub"] for p in f["cloud"]) and f["cloud"]
    assert any("bare template" in p["flags"] for p in pts)


def test_recommend(app):
    r = get(app, "/api/recommend?budget=50&floor=0.5&latency=60000&hardware=evo-x2,evo-t2s").json()
    assert r["chosen"] and r["chosen"]["src"]["file"].startswith("results/")
    assert r["savings_stub"] is True
    none = get(app, "/api/recommend?floor=1.01&hardware=evo-x2").json()
    assert none["chosen"] is None
    get(app, "/api/recommend?budget=abc", 400)


def test_recommend_moves_with_floor(app):
    lo = get(app, "/api/recommend?floor=0.5&latency=200000&hardware=evo-x2").json()["chosen"]["id"]
    hi = get(app, "/api/recommend?floor=0.5&latency=20000&hardware=evo-x2").json()["chosen"]["id"]
    assert lo != hi  # the recommender is a live function of the inputs


def test_scenario(app):
    s = get(app, "/api/scenario").json()
    assert s["arm"] == "ollama_ctx_4096_call2_notools"
    assert len(s["sessions"]) == 6 and all(x["num_ctx"] == 4096 for x in s["sessions"])
    assert "R2-real-v1-gated-kill" in s["register"]
    assert s["register"]["R2-real-v1-gated-kill"]["status"] == "VERIFIED"


def test_replay(app):
    r = get(app, f"/api/replay?session={first_session(app)}").json()
    assert len(r["steps"]) == 40
    for st in r["steps"]:
        assert st["naive"]["src"]["file"] == "results/x2_r2_real_v1.jsonl" and st["naive"]["src"]["line"]
        assert st["ours"]["decision"]["target"] in ("local", "local_trimmed", "cloud")
    get(app, "/api/replay?session=nope", 404)


def test_replay_stream(app):
    r = get(app, f"/api/replay/stream?session={first_session(app)}&delay_ms=0")
    assert r.content_type.startswith("text/event-stream")
    body = b"".join(r.stream).decode()
    events = [blk.split("\n")[0][len("event: "):] for blk in body.strip().split("\n\n")]
    assert events[0] == "start" and events[-1] == "done" and events.count("step") == 40


def test_register(app):
    allrows = get(app, "/api/register").json()
    assert "R2-mechanism-verdict" in allrows
    row = get(app, "/api/register?id=R2-mechanism-verdict").json()
    assert row["files"] == ["results/x2_r2_mechanism.jsonl"]
    get(app, "/api/register?id=not-a-row", 404)


def test_source(app):
    line = app.cache.r2_sessions["sessions"][0]["turns"][0]["line"]
    s = get(app, f"/api/source?file=results/x2_r2_real_v1.jsonl&line={line}").json()
    assert json.loads(s["text"])["record"] == "r2a_turn" and s["sha256_at_build"]
    m = get(app, "/api/source?file=results/x2_outcome_table_v3.jsonl&match=record:outcome_row,config:llama_server").json()
    assert m["n_matching_rows"] > 0
    get(app, "/api/source?file=../CLAUDE.md", 403)
    get(app, "/api/source?file=docs/STATE.md", 403)


def test_unknown_route_and_method(app):
    get(app, "/api/nope", 404)
    assert app.handle("POST", "/api/meta").status == 405


# ── scenario contract ──

def test_naive_side_is_the_recorded_silent_failure(app):
    """At 4096 every session fails silently (register R2-real-v1-gated-kill): the naive router keeps every step
    local, every call returned HTTP 200 with no error, and at least one step is a silent failure."""
    for s in app.cache.r2_sessions["sessions"]:
        r = data.replay(app.cache, app.backend, s["key"])
        locals_ = [st["naive"] for st in r["steps"] if st["naive"]["target"] == "local"]
        assert locals_, s["key"]
        for n in locals_:
            assert all(str(h) == "200" for h in n["outcome"]["http_status"]) and not n["outcome"]["errors"]
        assert any(n["outcome"]["silent"] for n in locals_), s["key"]


def test_naive_never_reads_num_ctx():
    step = {"task_words": 10, "transcript_tokens_calibrated": 30000, "num_ctx_requested": 4096,
            "loaded_context": 4096, "answer_budget_tokens": 384}
    assert naive.decide(step, 131072)["target"] == "local"
    assert naive.decide(dict(step, num_ctx_requested=10**9), 131072)["target"] == "local"


def test_ours_quality_is_projected_until_mitigation_measured(app):
    r = data.replay(app.cache, app.backend, first_session(app))
    if not app.cache.mitigation["measured"]:
        assert all(not st["ours"]["quality"]["measured"] and "projected" in st["ours"]["quality"]["label"]
                   for st in r["steps"])


def test_fake_router_keeps_system_prompt_and_fits(app):
    r = data.replay(app.cache, app.backend, first_session(app))
    for st in r["steps"]:
        d = st["ours"]["decision"]
        if d["target"] != "cloud":
            assert d["tokens"]["total"] <= d["num_ctx"] and d["tokens"]["system"] > 0


# ── offline, headless, committed snapshot ──

def test_static_has_no_external_urls():
    for p in STATIC.iterdir():
        text = p.read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, p.name


def test_no_browser_or_console_launch():
    for p in (REPO / "demo").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "webbrowser" not in text and "Start-Process" not in text, p


def test_committed_cache_loads():
    c = data.Cache()
    assert c.r2_sessions["sessions"] and c.points["points"] and c.register
    for f in c.manifest["sources"]:
        assert (REPO / f).exists(), f


def test_live_server_binds_localhost(app):
    srv = server.make_server(0, app)
    assert srv.server_address[0] == "127.0.0.1"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/meta", timeout=10) as resp:
            assert json.loads(resp.read())["backend"]["fake"] is True
        url = f"http://127.0.0.1:{port}/api/replay/stream?session={first_session(app)}&delay_ms=0"
        with urllib.request.urlopen(url, timeout=30) as resp:
            assert resp.read().decode().count("event: step") == 40
    finally:
        srv.shutdown()
        srv.server_close()


# ── integration with the real src/dse modules (skips until they are on this branch) ──

def test_real_backend_integration(cache_dir):
    b = data.load_backend()
    if b.is_fake:
        pytest.skip(f"real backend not available: router={b.router_source}; pareto={b.pareto_source}")
    d = cache_dir.parent / "dash_cache_real"
    build_cache.build(out_dir=d, backend=b)
    app = server.App(data.Cache(d), b)
    assert get(app, "/api/frontier").json()["machines"]
    get(app, "/api/recommend?floor=0.5&latency=60000")
    r = get(app, f"/api/replay?session={first_session(app)}").json()
    fields = {"target", "machine", "runtime", "model", "num_ctx", "tokens", "kept_turns", "dropped_turns",
              "predicted_latency_ms", "est_cost_usd", "budget_remaining_usd", "stub", "reason"}
    for st in r["steps"]:
        assert fields <= set(st["ours"]["decision"])
