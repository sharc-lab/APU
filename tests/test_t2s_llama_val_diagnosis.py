"""Tests for analysis/t2s_llama_val_diagnosis.py: the classifier on synthetic rows, and on the committed evo-t2s R2
validation files the reconciliation of the class counts with the harness's own baseline gate rates. No network."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "analysis"))
import x2_r2_agent as ag  # noqa: E402
import t2s_llama_val_diagnosis as T  # noqa: E402

CODE = "SC-ABCDEFGH"


def _row(final_text="", native1=None, content1="", tool_task="lookup_fact", **kw):
    r = {"record": "r2a_turn", "session_code": CODE, "tool_task": tool_task, "final_text": final_text,
         "calls": [{"content": content1, "native_tool_calls": native1},
                   {"content": final_text, "native_tool_calls": None}]}
    r.update(kw)
    return r


def _native(name, **args):
    return [{"id": "x", "function": {"index": 0, "name": name, "arguments": args}}]


class TestClassifier:
    def test_parse_json_strict_matches_harness(self):
        for s in ('{"answer": "a", "source": "b"}', '```json\n{"answer": "a"}\n```', "nope", "", "[1, 2]"):
            assert T.parse_json_strict(s) == ag.parse_json_strict(s)

    def test_rule2_other_tool_only(self):
        r = _row('{"answer": "x", "source": "y"}', _native("lookup_fact", key="REC-0001"))
        assert T.classify_rule2(r)[0] == "omitted"
        assert T.classify_rule2(r)[1].startswith("other_tool_only")

    def test_rule2_claimed_name_only_is_omitted(self):
        r = _row('{"answer": "x", "source": "y", "tool_calls": ["log_event", "lookup_fact"]}',
                 _native("lookup_fact", key="REC-0001"))
        cls, det = T.classify_rule2(r)
        assert cls == "omitted" and det.startswith("claimed_in_answer_report")

    def test_rule2_report_with_arguments_is_text(self):
        for rep in ({"log_event": "event text"}, [{"name": "log_event", "arguments": {"event": "e"}}]):
            r = _row(json.dumps({"answer": "x", "source": "y", "tool_calls": rep}), _native("lookup_fact", key="K"))
            assert T.classify_rule2(r)[0] == "tool_call_as_text"

    def test_rule2_whole_content_text_call(self):
        r = _row("ok", None, content1='{"name": "log_event", "arguments": {"event": "e"}}')
        assert T.classify_rule2(r)[0] == "tool_call_as_text"

    def test_rule2_native_log_event_is_scoring(self):
        assert T.classify_rule2(_row("x", _native("log_event", event="e")))[0] == "scoring_problem"

    def test_rule2_unknown_tool_is_wrong_value(self):
        assert T.classify_rule2(_row("x", _native("logEvent", event="e")))[0] == "wrong_value"

    def test_rule2_no_call(self):
        assert T.classify_rule2(_row("x"))[1].startswith("no_call")

    @pytest.mark.parametrize("text,cls", [
        ('{"answer": "value", "source": "s"}', "omitted"),
        ("", "omitted"),
        ('{"name": "lookup_fact", "arguments": {"key": "K"}}', "tool_call_as_text"),
        (f'{{"answer": "value {CODE}.", "source": "s"}}', "scoring_problem"),
        (f'{{"answer": "{CODE} then more text", "source": "s"}}', "wrong_value"),
        ('{"answer": "value SC-ZZZZZZZZ", "source": "s"}', "wrong_value"),
        (f'{{"answer": "value", "source": "{CODE}"}}', "wrong_value"),
    ])
    def test_rule5(self, text, cls):
        assert T.classify_rule5(_row(text))[0] == cls

    def test_every_class_is_known(self):
        for be in T.FILES:
            rows = ag.read_rows(ROOT / T.FILES[be])
            for m in T.MODELS:
                assert all(f[2] in T.CLASSES for f in T.failures(rows, m))


@pytest.fixture(scope="module")
def data():
    return {be: ag.read_rows(ROOT / fn) for be, fn in T.FILES.items()}


class TestCommittedFiles:

    def test_counts_reconcile_with_gate_rates(self, data):
        """Every failed rule on every baseline turn is classified exactly once, and per rule the number of classified
        failures is the harness gate's (1 - rate) x n, both as recomputed and as in the committed report."""
        for be, rows in data.items():
            rep = ag.evaluate_gates(rows, call2_tools="off")
            committed = json.loads((ROOT / (T.FILES[be] + ".report.json")).read_text(encoding="utf-8"))
            assert rep["gates"]["baseline"] == committed["gates"]["baseline"]
            for m in T.MODELS:
                f = T.failures(rows, m)
                table = rep["baseline_table"][m]["rules"]
                for rule in T.RULES:
                    fails = [x for x in f if x[1] == rule]
                    rate, n = table[rule]["rate"], table[rule]["n"]
                    assert len(fails) == round((1 - rate) * n), (be, m, rule)
                    assert T.rule_rates(rows, m)[rule] == (len(fails), n)
                s = T.summary()[be][m]
                assert sum(s["classes"].values()) == s["failed_rules"] == len(f)
                assert sum(sum(c.values()) for c in s["by_rule"].values()) == len(f)
                # one class per (turn, rule)
                keys = [(x[0]["attempt_id"], x[0]["turn_idx"], x[1]) for x in f]
                assert len(keys) == len(set(keys))
                failing = [r for r in T.RULES if table[r]["status"] == "fail"]
                assert failing == rep["gates"]["baseline"][m]["failing_rules"]

    def test_llama_gate_and_finding(self, data):
        s = T.summary()
        for be in T.FILES:
            assert s[be]["llama3.1:8b"]["canary_misses"] == 0
            assert set(s[be]["llama3.1:8b"]["by_rule"]) == {"rule2_log_event_called", "rule5_session_code"}
        assert T.dominant_class(s) == "omitted"
        assert "dominant class llama3.1:8b both backends: omitted" in T.register_value()["value"]

    def test_meta(self, data):
        for be, rows in data.items():
            meta = T.run_meta(rows)
            assert meta["hw_id"] == "evo-t2s"
            assert meta["seeds"] == [20260901, 20260902, 20260903]
            assert meta["digest_recorded"] is False
