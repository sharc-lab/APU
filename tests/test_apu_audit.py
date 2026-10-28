"""apu_bench audit (operator brief 2026-10-26, item 6): replay from recorded logs reproduces the recorded hidden-failure
counts of our own runs, offline (no live runtime).

  - results/demo_detector_check_v2.jsonl + the committed server logs results/demo_live_logs/detector_check_v2/: the
    audit slices the raw Ollama debug logs itself and must reproduce the per-call detector v2 verdicts recorded at run
    time and register row demo-detector-check-v2 (fired on 5/5 tasks at num_ctx 4096, 0/5 at the default; 10 calls
    flagged);
  - results/demo_detector_check.jsonl and the acceptance run with their logs: the log ground-truth cut counts of
    register row detector-precision-recall (TP + FN: 10 of 20 calls, 12 of 32 calls);
  - results/x2_own_compaction_v1_{a,b}.jsonl (detector v2 verdicts recorded per turn, logs not kept): register rows
    own-compaction-4096 / -8192 (wrong, flagged, silent, turns flagged per arm)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from apu_bench import adapters as AD, audit as A

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "results" / "demo_live_logs"
PROFILE = {"schema": "apu-device-profile/1", "profile_id": "evo-x2-test", "hardware": {}, "model": {"name": "qwen3:14b"},
           "runtime": {"name": "ollama", "version": "0.34.4"}, "cloud_model": "claude-haiku-5-5",
           "windows": [{"tier": "ollama_default", "window": 40960}, {"tier": "ollama_ctx_8192", "window": 8192},
                       {"tier": "ollama_ctx_4096", "window": 4096}]}


def _rows(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _reg(rid):
    sys.path.insert(0, str(REPO / "src" / "dse"))
    from src.dse.route_at_start import register_rows
    return register_rows((REPO / "docs" / "NUMBERS_REGISTER.md").read_text(encoding="utf-8"))[rid]["value"]


def _log(*p):
    return (LOGS.joinpath(*p)).read_text(encoding="utf-8", errors="replace")


def test_slicer_and_verdict_units():
    text = "\n".join([
        'time=1 level=DEBUG msg="truncating input messages which exceed context length" truncated=3',
        "slot | task 0 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 4000",
        '[GIN] 2026/10/23 - 20:02:10 | 200 |   1s |       127.0.0.1 | POST     "/api/chat"',
        '[GIN] 2026/10/23 - 20:02:10 | 200 |   0s |       127.0.0.1 | GET      "/api/ps"',
        "slot | task 1 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 1000",
        '[GIN] 2026/10/23 - 20:02:12 | 500 |   1s |       127.0.0.1 | POST     "/api/chat"'])
    s = A.slice_server_log(text)
    assert [x["http_status"] for x in s] == [200, 500]
    assert A.log_verdict(s[0]["text"], None)["cut"] is True
    assert A.log_verdict(s[0]["text"], 10)["cut"] is True        # rendered path: first kept = 10 - 3 = 7 > 0
    assert A.log_verdict(s[0]["text"], 3)["cut"] is False        # kept all 3
    assert A.log_verdict(s[1]["text"], None)["cut"] is False


def test_demo_detector_check_v2_reproduced_from_raw_logs():
    rows = _rows(REPO / "results" / "demo_detector_check_v2.jsonl")
    reqs = AD.demo_detector_requests(rows, "num_ctx_4096") + AD.demo_detector_requests(rows, "default")
    logs = {("ollama_ctx_4096", 4096): _log("detector_check_v2", "serve_4096.log"),
            ("ollama_default", 40960): _log("detector_check_v2", "serve_default.log")}
    res = A.run_audit(reqs, PROFILE, logs)
    g = {x["config"]: x for x in res["groups"]}
    assert set(g) == {"ollama_ctx_4096", "ollama_default"}
    for cfg, x in g.items():
        assert x["method"] == "server_log"
        mine = [v["cut"] for v in x["verdicts"]]
        recorded = [r["recorded_cut_v2"] for r in sorted((r for r in reqs if r["ran_at"]["config"] == cfg),
                                                         key=lambda r: r["_order"])]
        assert mine == recorded
    v = _reg("demo-detector-check-v2")
    fired = int(re.search(r"num_ctx_4096: fired on (\d+)/5 tasks", v).group(1))
    calls_flagged = int(re.search(r"num_ctx_4096: .*?calls 10, flagged by a server cut line .*? on (\d+)", v).group(1))
    assert g["ollama_ctx_4096"]["tasks_hidden_failure"] == fired == 5
    assert g["ollama_ctx_4096"]["requests_hidden_cut"] == calls_flagged == 10
    assert g["ollama_default"]["tasks_hidden_failure"] == 0
    assert g["ollama_ctx_4096"]["hidden_failure_rate"] == 1.0
    # cost basis: the server's own token count; no completion tokens recorded, priced on input only
    assert g["ollama_ctx_4096"]["cloud_cost"]["floor_usd"] <= g["ollama_ctx_4096"]["cloud_cost"]["upper_bound_usd"]
    assert g["ollama_ctx_4096"]["cloud_cost"]["billed_usd"] is None


def test_log_ground_truth_counts_match_detector_precision_recall():
    v = _reg("detector-precision-recall")
    tp, fn = map(int, re.search(r"demo detector check: TP (\d+) FP \d+ FN (\d+)", v).groups())
    tp2, fn2 = map(int, re.search(r"demo acceptance run: TP (\d+) FP \d+ FN (\d+)", v).groups())
    cut = 0
    for lane, log in (("num_ctx_4096", "serve_4096.log"), ("default", "serve_default.log")):
        cut += sum(A.log_verdict(s["text"], None)["cut"] for s in A.slice_server_log(_log("detector_check", log)))
    assert cut == tp + fn == 10
    acc = sum(A.log_verdict(s["text"], None)["cut"] for ctx in (4096, 8192, 32768)
              for s in A.slice_server_log(_log("acceptance", f"serve_{ctx}.log")))
    assert acc == tp2 + fn2 == 12


@pytest.mark.parametrize("w", [4096, 8192])
def test_own_compaction_reproduced_from_recorded_verdicts(w):
    sys.path.insert(0, str(REPO / "analysis"))
    import numbers_register as nr
    ag = nr._r2_agent_module()
    rows = [r for f in nr.OC_PARTS.values() for r in nr._finished_attempt_rows(ag.read_rows(REPO / f))]
    res = A.run_audit(AD.r2a_turn_requests(rows), PROFILE, cloud_model=None)
    g = {x["config"]: x for x in res["groups"]}
    v = _reg(f"own-compaction-{w}")
    for arm in (f"own_compaction_{w}", f"client_trim_{w}"):
        m = re.search(arm + r": wrong (\d+)/(\d+) \([\d.]+%\), flagged \(detector v2\) (\d+), silent (\d+); turns flagged "
                      r"(\d+), turns without a log slice (\d+)", v)
        wrong, n, flagged, silent, turns_flagged, no_log = map(int, m.groups())
        x = g[arm]
        assert x["method"] == "recorded_verdict" and x["window"] == w
        assert (x["tasks"], x["wrong"], x["wrong_with_hidden_cut"], x["wrong_without_cut"]) == (n, wrong, flagged, silent)
        assert x["tasks_hidden_failure"] == turns_flagged and x["requests_no_log"] == no_log


def test_estimate_mode_and_billed_cost():
    msgs = [{"role": "system", "content": "x" * 4000}, {"role": "user", "content": "y" * 20000}]
    reqs = [{"session_id": "s", "request_idx": 0, "messages": msgs, "prompt_tokens": 6000, "completion_tokens": 100,
             "cached_prompt_tokens": 0, "outcome": {"ok": False}},
            {"session_id": "s", "request_idx": 1, "messages": msgs[:1], "prompt_tokens": 1000, "completion_tokens": 50,
             "cached_prompt_tokens": 1000}]
    res = A.run_audit(reqs, PROFILE, calib=1.0)
    g = {x["window"]: x for x in res["groups"]}
    assert all(x["method"] == "estimate" and not x["ground_truth"] for x in g.values())
    assert g[4096]["requests_cut"] == 1 and g[40960]["requests_cut"] == 0       # 6000 est > 4096; nothing > 40960
    assert g[4096]["wrong_with_hidden_cut"] == 1
    c = g[4096]["cloud_cost"]
    assert c["billed_usd"] is not None and c["floor_usd"] <= c["billed_usd"] <= c["upper_bound_usd"] + 1e-12
    assert "PREDICTED" in A.report_md(res)


def test_cli_audit_roundtrip(tmp_path):
    from apu_bench.__main__ import main
    rows = _rows(REPO / "results" / "demo_detector_check_v2.jsonl")
    reqs = AD.demo_detector_requests(rows, "num_ctx_4096")
    sp = tmp_path / "sessions.jsonl"
    sp.write_text(json.dumps({"schema": "apu-audit-session/1"}) + "\n" + "".join(json.dumps(r) + "\n" for r in reqs),
                  encoding="utf-8")
    pp = tmp_path / "profile.json"
    pp.write_text(json.dumps(PROFILE), encoding="utf-8")
    rc = main(["audit", "--profile", str(pp), "--sessions", str(sp), "--out", str(tmp_path / "o"),
               "--server-log", f"4096={LOGS / 'detector_check_v2' / 'serve_4096.log'}"])
    assert rc == 0
    res = json.loads((tmp_path / "o" / "audit.json").read_text(encoding="utf-8"))
    (g,) = res["groups"]
    assert g["tasks_hidden_failure"] == 5 and g["method"] == "server_log"
