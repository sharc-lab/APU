"""Per-task route-at-start plus window flag, by replay (analysis/impact_model.py::route_measured_policy and helpers;
rows hybrid-route-at-start-measured and task-latency-by-policy)."""
import contextlib
import io
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "harness"))

import impact_model as im  # noqa: E402


def _cl(ok, usd, s):
    return {"ok": ok, "usd": usd, "seconds": s}


def _lo(flag, ok, wall):
    return {"flag": flag, "ok": ok, "wall_s": wall}


def test_tier_window():
    assert im.tier_window("ollama_ctx_8192") == 8192
    assert im.tier_window("ollama_ctx_4096_numkeep") == 4096
    with pytest.raises(ValueError):
        im.tier_window("ollama_default")


def test_policy_routes_keeps_and_escalates():
    k = [(1, "a"), (1, "b"), (1, "c"), (1, "d"), (2, "a")]
    local = {k[0]: _lo(False, True, 10.0), k[1]: _lo(True, False, 20.0), k[2]: _lo(True, True, 30.0),
             k[3]: _lo(False, False, 40.0), k[4]: _lo(False, True, 50.0)}
    cloud = {x: _cl(True, 0.01, 2.0) for x in k}
    proj = {k[0]: 100, k[1]: 100, k[2]: 100, k[3]: 999, k[4]: 100}
    redo = {k[1]: _cl(True, 0.05, 5.0)}
    r = im.route_measured_policy(local, cloud, proj, 500, seeds=[1], redo=redo, redo_tier_usd=0.04)
    t = r["tasks"]
    assert r["n"] == 4  # seed 2 excluded
    assert t[k[0]]["src"] == "local" and t[k[0]]["usd"] == 0 and t[k[0]]["seconds"] == 10.0
    assert t[k[1]]["src"] == "escalated_redo" and t[k[1]]["usd"] == 0.05 and t[k[1]]["seconds"] == 25.0
    assert t[k[2]]["src"] == "escalated_imputed" and t[k[2]]["usd"] == 0.04 and t[k[2]]["seconds"] == 32.0
    assert t[k[3]]["src"] == "routed" and t[k[3]]["ok"] and t[k[3]]["seconds"] == 2.0
    assert r["correct"] == 4 and r["kept_local"] == 1 and r["escalated"] == 2 and r["routed"] == 1
    assert abs(r["usd_task"] - (0.05 + 0.04 + 0.01) / 4) < 1e-12
    assert r["pred_fit_but_flagged"] == 2 and r["pred_over_but_not_flagged"] == 1


def test_policy_without_redo_is_interim_at_the_cloud_row():
    k = (1, "a")
    r = im.route_measured_policy({k: _lo(True, False, 10.0)}, {k: _cl(False, 0.002, 1.0)}, {k: 1}, 10, [1])
    assert r["escalated_interim"] == 1 and r["usd_total"] == 0.002 and r["correct"] == 0


def test_quantiles():
    xs = list(range(1, 11))
    assert im._q(xs, 50) == 5.5
    assert abs(im._q(xs, 90) - 9.1) < 1e-9
    assert im._q([3.0], 90) == 3.0


def test_projection_per_turn_matches_the_harness_max():
    import x2_r2_evalset as ev
    p = im.projected_task_prompts(REPO, 20260901)
    with contextlib.redirect_stdout(io.StringIO()):
        es = ev.build_session(20260901, workload="evalset", guardrails=False, design="v2")
    assert len(p) == 30
    assert max(v for _i, v in p.values()) == ev.projected_max_prompt(es)
    assert sorted(i for i, _v in p.values()) == list(range(1, 31))


def test_pending_texts():
    assert im.route_measured_text({"status": "PENDING", "pending": ["x"]}).startswith("PENDING")
    assert im.latency_text({"status": "PENDING", "pending": ["x"]}).startswith("PENDING")
