"""Live demo engine without a model or a network: task rebuild from the recorded B3 v2 session, the lane sequence and
escalation (demo_run), the tally, the cloud lane's REPLAYED fallback and budget checks, the screen's API, the agent
protocol, and the lane server's stale-server guard."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))

import demo_live as L  # noqa: E402
import demo_run as DR  # noqa: E402
import demo_servers as S  # noqa: E402
import route_at_start as R  # noqa: E402


@pytest.fixture(scope="module")
def b3():
    return L.read_jsonl(REPO / "results" / L.B3_FILE)


@pytest.fixture(scope="module")
def prof():
    return R.load_profiles()


@pytest.fixture(scope="module")
def task8(b3):
    return L.build_task(b3, 20260901, "inv_rename")


def test_history_rebuild_matches_the_recorded_session(b3, task8):
    t = task8
    assert t["turn_idx"] == 8 and t["history_turns"] == 7
    assert t["history_check"]["rebuilt_est"] == t["history_check"]["recorded_est"]
    assert t["history"][0]["role"] == "system" and sum(m["role"] == "user" for m in t["history"]) == 7
    assert t["built"]["env_before"] == t["env_before"]
    assert t["projected_prompt"] == 8708
    assert set(t["recorded"]) >= {"ollama_ctx_4096_call2_notools", "ollama_default_call2_notools"}
    late = L.build_task(b3, 20260901, "led_balance_info", with_cloud=False)
    assert late["built"] is None and late["turn_idx"] == 21
    assert late["history_check"]["rebuilt_est"] == late["history_check"]["recorded_est"]


def test_projection_equals_the_headline_rows_projection(b3):
    sys.path.insert(0, str(REPO / "analysis"))
    import impact_model as im
    proj = im.projected_task_prompts(REPO, 20260902)
    es = L.EV.build_session(20260902, None, "evalset", False, L.MODEL, None, "v2")
    for tid, (idx, p) in list(proj.items())[:6]:
        assert L.projected_task_prompt(es, tid) == (idx, p)


def _fake_local(results):
    def run(lane, task, num_ctx):
        r = dict(results[(lane, num_ctx)])
        r.setdefault("flagged_v2", r.get("flagged"))     # these fakes stand for a real server cut (both detectors)
        return {"calls": [], "seconds": 1.0, **r}
    return run


class FakeCloud:
    def __init__(self, ok=True, usd=0.05):
        self.calls, self.ok, self.usd = [], ok, usd

    def __call__(self, lane, task, model):
        self.calls.append((lane, model))
        return {"route": "cloud", "model": model, "label": "LIVE", "end_state_ok": self.ok, "usd_ledger": self.usd,
                "usd_cache_priced": self.usd / 8, "seconds": 2.0}


def test_amd8192_routes_a_long_task_to_the_cloud_and_runs_default_locally(prof, task8):
    cloud = FakeCloud()
    res = DR.run_task("amd8192", prof, task8, _fake_local({("default", 8192): {"end_state_ok": False, "flagged": True}}),
                      cloud, lambda: 3.0)
    assert res["lanes"]["default"]["route"] == "local" and res["lanes"]["default"]["num_ctx"] == 8192
    o = res["lanes"]["ours"]
    assert o["route"] == "cloud" and o["model"] == "claude-sonnet-5-5"
    assert o["decision"]["need_tokens"] == 8708 and o["decision"]["margin"] == 0.0
    assert sorted(cloud.calls) == [("cloud", "claude-sonnet-5-5"), ("ours", "claude-sonnet-5-5")]
    assert list(res["lanes"]) == ["default", "ours", "cloud"]
    t = DR.tally([res])
    assert t["default"]["wrong_flagged"] == 1 and t["default"]["kept_local"] == 1
    assert t["ours"]["correct"] == 1 and t["ours"]["kept_local"] == 0 and t["ours"]["usd_ledger_per_task"] == 0.05


def test_intel_runs_ours_locally_at_the_graphics_chip_window_and_escalates_on_a_flag(prof, task8):
    cloud = FakeCloud(ok=True, usd=0.004)
    local = _fake_local({("default", 4096): {"end_state_ok": False, "flagged": False},
                         ("ours", 32768): {"end_state_ok": False, "flagged": True}})
    res = DR.run_task("intel", prof, task8, local, cloud, lambda: 3.0)
    o = res["lanes"]["ours"]
    assert o["route"] == "local" and o["num_ctx"] == 32768 and o["escalated"] and o["end_state_ok"] is True
    assert "Intel graphics-chip setting, simulated" in o["decision"]["reason"]
    assert o["decision"]["setting_label"] == "Intel graphics-chip setting, simulated"
    t = DR.tally([res])
    assert t["default"]["wrong_silent"] == 1 and t["ours"]["escalated"] == 1 and t["ours"]["kept_local"] == 0
    assert t["ours"]["usd_ledger_total"] == 0.004
    assert DR.configurations("intel", prof) == [{"num_ctx": 4096, "server_env": {}},
                                                {"num_ctx": 32768, "server_env": {}}]
    assert DR.configurations("amd8192", prof) == [{"num_ctx": 8192, "server_env": {}}]


def test_detector_switch_drives_escalation_and_flags(prof, task8, monkeypatch):
    local = _fake_local({("default", 4096): {"end_state_ok": False, "flagged": True, "flagged_v2": True},
                         ("ours", 32768): {"end_state_ok": True, "flagged": True, "flagged_v2": False}})
    assert DR.DETECTOR == "v2"          # the v2 re-check passed on evo-x2 (row demo-detector-check-v2)
    monkeypatch.setattr(DR, "DETECTOR", "v1")
    r1 = DR.run_task("intel", prof, task8, local, FakeCloud(), lambda: 3.0)
    assert r1["lanes"]["ours"].get("escalated")
    monkeypatch.setattr(DR, "DETECTOR", "v2")
    r2 = DR.run_task("intel", prof, task8, local, FakeCloud(), lambda: 3.0)
    assert not r2["lanes"]["ours"].get("escalated") and r2["lanes"]["ours"]["route"] == "local"
    t = DR.tally([r2])
    assert t["ours"]["kept_local"] == 1 and t["default"]["wrong_flagged"] == 1
    from demo.live import server as SV
    html = (SV.STATIC / "index.html").read_text(encoding="utf-8")
    assert "Detector v1 (token-count check) not used: false alarms at default (see detector-precision-recall)" in html
    assert "the runtime's own signal" in html and "known false alarm" in html


def test_pending_cloud_parts_are_not_graded(prof, task8):
    pend = lambda lane, task, model: {"pending": True, "model": model}  # noqa: E731
    res = DR.run_task("amd8192", prof, task8, _fake_local({("default", 8192): {"end_state_ok": True,
                                                                                 "flagged": False}}), pend, lambda: 3)
    t = DR.tally([res])
    assert t["ours"]["pending"] == 1 and t["ours"]["graded"] == 0 and t["default"]["graded"] == 1


def test_cloud_lane_replays_when_live_is_off_and_never_shows_the_key(task8, tmp_path):
    from demo.live import cloud_lane as CL
    ac = L.read_jsonl(REPO / "results" / L.ALL_CLOUD_FILE)
    lane = CL.CloudLane(env={"ANTHROPIC_API_KEY": "sk-test-SECRETSECRET"}, ledger=tmp_path / "l.jsonl",
                        all_cloud_rows=ac)
    r = lane("cloud", task8, "claude-sonnet-5-5")
    assert r["label"] == "REPLAYED" and "DEMO_CLOUD_LIVE" in r["reason"] and r["end_state_ok"] is not None

    class Boom:
        def prepare(self, **kw):
            raise RuntimeError("provider said sk-test-SECRETSECRET is bad")
    lane2 = CL.CloudLane(env={"ANTHROPIC_API_KEY": "sk-test-SECRETSECRET", "DEMO_CLOUD_LIVE": "1"},
                         ledger=tmp_path / "l2.jsonl", client_factory=lambda cap, led: Boom(), all_cloud_rows=ac)
    r2 = lane2("cloud", task8, "claude-sonnet-5-5")
    assert r2["label"] == "REPLAYED" and "SECRET" not in json.dumps(r2) and "[redacted]" in r2["reason"]
    lane3 = CL.CloudLane(env={"ANTHROPIC_API_KEY": "k" * 20, "DEMO_CLOUD_LIVE": "1"}, ledger=tmp_path / "l3.jsonl",
                         session_limit=0.0, all_cloud_rows=ac)
    assert "budget left" in lane3("cloud", task8, "claude-sonnet-5-5")["reason"]
    assert CL.projected_ledger_usd(task8, "claude-sonnet-5-5") > 0


def test_screen_api_replay_mode(tmp_path):
    from demo.live import server as SV
    run = tmp_path / "run.jsonl"
    res = {"scenario": "amd8192", "seed": 1, "task_id": "t", "turn_idx": 3, "history_turns": 2,
           "projected_prompt": 5000, "lanes": {
               "default": {"route": "local", "end_state_ok": False, "flagged": True, "flagged_v2": True,
                           "calls": [{"n": 1, "seconds": 0, "cut": True, "used_tokens": 8000, "own_tokens": 9000}]},
               "ours": {"route": "cloud", "label": "LIVE", "end_state_ok": True, "usd_ledger": 0.05, "seconds": 0,
                        "decision": {"route": "cloud", "reason": "Cloud from the start"}},
               "cloud": {"route": "cloud", "label": "LIVE", "end_state_ok": True, "usd_ledger": 0.05, "seconds": 0}}}
    run.write_text(json.dumps({"record": "task_result", **res}) + "\n", encoding="utf-8")
    app = SV.App("replay", str(run))
    code, _, body = app.handle("GET", "/api/meta")
    meta = json.loads(body)
    assert code == 200 and meta["headline"]["ours"]["correct"] == 88 and meta["headline"]["cloud"]["usd_task"] == 0.0102
    assert json.loads(app.handle("POST", "/api/run?scenario=amd8192")[2])["started"]
    import time
    for _ in range(100):
        if app.run.done:
            break
        time.sleep(0.05)
    evs = json.loads(app.handle("GET", "/api/events?since=0")[2])["events"]
    kinds = [e["event"] for e in evs]
    assert kinds[0] == "run_start" and evs[0]["label"] == "REPLAYED" and kinds[-1] == "run_end"
    tally = [e for e in evs if e["event"] == "tally"][-1]["tally"]
    assert tally["default"]["wrong_flagged"] == 1 and tally["ours"]["correct"] == 1
    assert app.handle("POST", "/api/run?scenario=nope")[0] == 400
    html = app.handle("GET", "/")[2].decode("utf-8")
    assert "cut off" in html.lower() and "memory for the conversation" in html and "—" not in html


def test_agent_protocol_with_a_fake_process():
    from demo.live import runner as RN
    lines = [{"record": "agent_ready"}, {"record": "call", "lane": "default", "n": 1, "cut": False},
             {"record": "local_result", "result": {"end_state_ok": True}}]

    class P:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("".join(json.dumps(x) + "\n" for x in lines))

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass
    seen = []
    a = RN.Agent(["x"], popen=lambda *a, **k: P(), on_line=seen.append)
    r = a.local("default", {"seed": 1, "task_id": "t"}, 4096)
    assert r == {"end_state_ok": True} and seen[0]["record"] == "call"
    assert json.loads(a.proc.stdin.getvalue().splitlines()[0])["num_ctx"] == 4096
    a.close()


def test_frozen_picks_match_the_rule_and_the_source(b3):
    from demo.live import picks as P
    import hashlib
    frozen = json.loads((REPO / "demo" / "live" / "picks_v1.json").read_text(encoding="utf-8"))
    r = P.rule_v1(b3)
    assert r["enough"] and not r["filled_beyond_rule"]
    assert [(p["seed"], p["task_id"]) for p in frozen["picks"]] == [(p["seed"], p["task_id"]) for p in r["picks"]]
    assert frozen["source_sha256"] == hashlib.sha256((REPO / "results" / L.B3_FILE).read_bytes()).hexdigest()
    assert frozen["rule"] == P.RULE_V1 and "—" not in frozen["rule"]
    fail = [p for p in frozen["picks"] if p["class"].startswith("wrong")]
    assert len(fail) == 3 and all(sum(x is False for x in p["recorded"]["ollama_ctx_4096_call2_notools"]) >= 3
                                  and sum(x is True for x in p["recorded"]["ollama_default_call2_notools"]) >= 4
                                  for p in fail)
    ok = [p for p in frozen["picks"] if p["class"] == "right in every arm"]
    assert len(ok) == 2 and all(sum(v) >= 4 for p in ok for v in p["recorded"].values())
    with pytest.raises(SystemExit):
        P.freeze(REPO / "demo" / "live" / "picks_v1.json", b3)


def test_register_rows_for_the_demo(tmp_path, prof, task8):
    sys.path.insert(0, str(REPO / "analysis"))
    import numbers_register as nr
    res = DR.run_task("amd8192", prof, task8, _fake_local({("default", 8192): {"end_state_ok": False,
                                                                                 "flagged": True}}), FakeCloud(),
                      lambda: 3.0)
    (tmp_path / "results").mkdir()
    (tmp_path / nr.DEMO_LIVE_RUN_FILE).write_text("\n".join(json.dumps(x) for x in [
        {"record": "run_start"}, {"record": "task_result", **res}, {"record": "run_end"}]) + "\n", encoding="utf-8")
    v = nr.compute_demo_live_run(tmp_path)["value"]
    assert "default: 0/1 correct (wrong 1: silent 0, flagged 1)" in v and "Ours beats Default local" in v
    assert "Ours does NOT beat All cloud" in v
    det = [{"record": "detector_task", "arm": a, "flagged": f, "max_own_tokens": 8000, "max_used_tokens": u,
            "task_id": "t", "seed": 1, "turn_idx": 9, "end_state_ok": True,
            "calls": [{"used_tokens": u, "prompt_eval_count": u}]}
           for a, f, u in (("num_ctx_4096", True, 4000), ("default", False, 8000))]
    (tmp_path / nr.DEMO_DETECTOR_FILE).write_text("\n".join(json.dumps(x) for x in det + [{"record": "run_end"}]),
                                                 encoding="utf-8")
    assert nr.compute_demo_detector_check(tmp_path)["value"].startswith("detector check passed")


def test_queue_insert_after_places_demo_jobs_directly_after_the_running_b5_job():
    import x2_queue_fill as qf
    q = [{"id": "done_job", "status": "done"}, {"id": "x2_b5_simcap_8192", "status": "running"},
         {"id": "x2_b5_simcap_4096", "status": "pending"}]
    new = qf.insert_entries_after(q, "x2_b5_simcap_8192", ["x2_demo_detector_check", "x2_demo_live_acceptance"])
    assert [it["id"] for it in new] == ["done_job", "x2_b5_simcap_8192", "x2_demo_detector_check",
                                         "x2_demo_live_acceptance", "x2_b5_simcap_4096"]
    for e in new[2:4]:
        assert e["status"] == "pending" and "--queued" in e["cmd"] and "tq.advance()" in e["note"]
        assert "sets no pause" in e["note"] and "demo\\picks_v1.json" in e["cmd"]
    with pytest.raises(ValueError):
        qf.insert_entries_after(q, "done_job", ["x2_demo_detector_check"])
    with pytest.raises(ValueError):
        qf.insert_entries_after(new, "x2_b5_simcap_8192", ["x2_demo_detector_check"])


def test_lane_server_guard_refuses_a_held_port(tmp_path):
    srv = S.LaneServer(11999, {}, str(tmp_path / "s.log"), popen=lambda *a, **k: None, listeners=lambda p: [1234])
    with pytest.raises(S.sg.GuardError):
        srv.start()
    with pytest.raises(ValueError):
        S.LaneServer(11999, {"OLLAMA_HOST": "x"}, str(tmp_path / "s.log"))
    env = S.LaneServer(11999, {"OLLAMA_KV_CACHE_TYPE": "q8_0"}, str(tmp_path / "s.log")).env()
    assert env["OLLAMA_HOST"] == "127.0.0.1:11999" and env["OLLAMA_DEBUG"] == "1" and env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
