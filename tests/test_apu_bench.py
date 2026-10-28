"""apu_bench run (operator brief 2026-10-26, item 5): fake-runtime report end to end, schema, provenance, live plan."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apu_bench import RESULT_SCHEMA, bench as B, profile as P, taskset as TS
from apu_bench.__main__ import main

REPO = Path(__file__).resolve().parents[1]
FAKE = REPO / "apu_bench" / "profiles" / "example-fake.json"
X2 = REPO / "apu_bench" / "profiles" / "evo-x2-qwen3-14b.json"


def test_profiles_validate():
    _p, sha, missing = P.load(FAKE)
    assert missing == [] and len(sha) == 64
    _p, _sha, missing = P.load(X2)
    assert "hardware.gpu" in missing and "model.quant" in missing and "hardware.cpu" not in missing


@pytest.mark.parametrize("bad", [
    {"schema": "x"},
    {"windows": []},
    {"windows": [{"tier": "ollama_ctx_4096", "window": 8192}]},
    {"windows": [{"tier": "bogus", "window": 1}]},
])
def test_profile_rejects(bad):
    p = json.loads(FAKE.read_text(encoding="utf-8"))
    p.update(bad)
    with pytest.raises(P.ProfileError):
        P.validate(p)


def test_taskset_hash_stable_and_matches_projection():
    a, b = TS.build("b3v2-small"), TS.build("b3v2-small")
    assert a["sha256"] == b["sha256"] and a["n_task_turns"] == 30
    assert TS.build("b3v2-3seed")["sha256"] != a["sha256"]
    import sys
    sys.path.insert(0, str(REPO / "analysis"))
    import impact_model as im
    proj = im.projected_task_prompts(REPO, 20260901)
    mine = {t["item_id"]: t["projected_prompt_tokens"] for t in a["sessions"][0]["turns"] if t["kind"] == "task"}
    assert mine == {k: v[1] for k, v in proj.items()}


def test_fake_run_writes_full_report(tmp_path):
    rc = main(["run", "--profile", str(FAKE), "--subset", "b3v2-small", "--out", str(tmp_path)])
    assert rc == 0
    res = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert res["schema"] == RESULT_SCHEMA and res["mode"] == "fake" and res["label"] == B.FAKE_LABEL
    assert res["task_set"]["sha256"] == TS.build("b3v2-small")["sha256"]
    assert res["records_sha256"] == B.sha256_file(tmp_path / "records.jsonl")
    assert res["device_profile_sha256"] == B.sha256_file(FAKE)
    assert "git_head" in res["provenance"] and res["provenance"]["harness_blobs"]["harness/x2_r2_evalset.py"]
    assert [s["step"] for s in res["commands"]["client"]] == ["validation", "real"]
    assert "--tiers" in res["commands"]["client"][1]["argv"]
    for r in res["results"]:
        assert r["n_tasks"] == 30 and r["loaded_windows"] == [r["window"]]
        assert r["wrong"] == r["wrong_silent"] + r["wrong_flagged"]
        assert r["seconds_per_task"]["median"] > 0
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert B.FAKE_LABEL in md and "| 4096 | ollama_ctx_4096 | 30 |" in md and res["task_set"]["sha256"] in md


def test_fake_flag_is_the_fits_locally_baseline(tmp_path):
    from src.dse.fits_locally import fits_locally
    prof, _s, _m = P.load(FAKE)
    ts = TS.build("b3v2-small")
    recs = B.fake_records(prof, ts)
    for r in recs:
        if r.get("record") == "r2a_turn":
            assert r["over_loaded_window"] is (not fits_locally(r["final_call_prompt_tokens_calibrated"],
                                                               r["loaded_context"])["fits"])


def test_summary_on_hand_records():
    prof = {"windows": [{"tier": "ollama_ctx_4096", "window": 4096}]}
    base = {"record": "r2a_turn", "arm_id": "ollama_ctx_4096_call2_notools", "item_kind": "task", "attempt_id": "a",
            "seed": 1, "loaded_context": 4096}
    recs = [dict(base, end_state_ok=True, turn_wall_s=10, over_loaded_window=False),
            dict(base, end_state_ok=False, turn_wall_s=20, over_loaded_window=True),
            dict(base, end_state_ok=False, turn_wall_s=30, over_loaded_window=False),
            dict(base, end_state_ok=False, turn_wall_s=40, over_loaded_window=False, cut_v2_flagged=True),
            dict(base, end_state_ok=False, turn_wall_s=50, attempt_id="incomplete"),
            {"record": "r2a_session", "attempt_id": "a"}]
    (r,) = B.summarize(recs, prof, [1])
    assert (r["n_tasks"], r["wrong"], r["wrong_flagged"], r["wrong_silent"], r["flagged"]) == (4, 3, 2, 1, 2)
    assert r["seconds_per_task"]["mean"] == 25


def test_live_without_execute_only_writes_a_plan(tmp_path):
    plan = B.run(X2, "b3v2-small", tmp_path, runtime="live")
    assert plan["plan_only"] and (tmp_path / "plan.json").exists() and not (tmp_path / "records.jsonl").exists()
    real = plan["steps"][-1]["argv"]
    assert real[real.index("--tiers") + 1] == "ollama_default,ollama_ctx_16384,ollama_ctx_8192,ollama_ctx_4096"
    plan2 = B.run(X2, "b3v2-small", tmp_path, runtime="live", rules_from="val.jsonl")
    assert [s["step"] for s in plan2["steps"]] == ["real"]
