"""apu_bench audit one-page report (apu_bench/onepage.py): the recommendation rule on synthetic rows, and the page on
our own recorded runs (demo detector check v2 with its raw logs; own-compaction recorded verdicts), offline."""
from __future__ import annotations

from pathlib import Path

from apu_bench import onepage as OP

REPO = Path(__file__).resolve().parents[1]


def _row(cfg, w, hidden_rate, wrong_rate=None, esc=None, gt=True):
    return {"config": cfg, "window": w, "kv": "not recorded", "ground_truth": gt, "hidden_rate": hidden_rate,
            "wrong_rate": wrong_rate, "escalation_usd_per_100": esc, "wrong": 0, "graded": 0}


def test_rule_lowest_hidden_then_ties():
    rows = [_row("a", 4096, 0.5, 0.4, 1.0), _row("b", 8192, 0.0, 0.3, 0.0), _row("c", 16384, 0.0, 0.2, 0.0),
            _row("est", 40960, 0.0, 0.0, 0.0, gt=False)]
    rec = OP.recommend(rows)
    assert rec["setting"]["config"] == "c"            # tie on hidden rate broken by lower wrong rate; estimate excluded


def test_rule_cost_bound_and_none_within():
    rows = [_row("a", 4096, 0.1, 0.4, 5.0), _row("b", 8192, 0.2, 0.3, 1.0)]
    assert OP.recommend(rows, cost_bound=2.0)["setting"]["config"] == "b"
    r = OP.recommend(rows, cost_bound=0.5)
    assert r["setting"] is None and "run in the cloud" in r["reason"]
    assert OP.recommend([_row("x", 4096, 0.0, gt=False)])["setting"] is None


def test_examples_on_our_recorded_runs(tmp_path):
    done = OP.examples(REPO, tmp_path)
    demo = {r["config"]: r for r in done["demo_detector_check_v2"]["rows"]}
    assert demo["ollama_ctx_4096"]["hidden"] == 5 and demo["ollama_ctx_4096"]["method"] == "server_log"
    assert demo["ollama_default"]["hidden"] == 0
    assert done["demo_detector_check_v2"]["recommendation"]["setting"]["window"] == 40960
    oc = done["own_compaction_v1"]
    assert all(r["method"] == "recorded_verdict" for r in oc["rows"])
    assert oc["recommendation"]["setting"]["config"] == "own_compaction_8192"
    for name in done:
        md = (tmp_path / f"{name}.md").read_text(encoding="utf-8")
        h = (tmp_path / f"{name}.html").read_text(encoding="utf-8")
        assert "Recommended setting" in md and "Rule:" in md and "wrong without a cut" in md
        assert h.startswith("<!doctype html>") and "<table>" in h
