"""Verification tier payload (src/dse/verification_tier.py, analysis/verification_tier_plan.py): grader isolation and
the recall-probe strip (operator decisions 2026-11-03). Offline; reads committed result files only."""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for sub in ("", "analysis", "harness"):
    p = str(REPO / sub) if sub else str(REPO)
    if p not in sys.path:
        sys.path.insert(0, p)

from src.dse import verification_tier as VT  # noqa: E402

VT_FILE = REPO / "src" / "dse" / "verification_tier.py"
FORBIDDEN = ("check_task", "reference", "answer_values", "end_state", "expected", "evalset_env", "x2_r2_evalset",
             "numbers_register", "grade", "state_ok", "answer_ok", "cloud_redo", "all_cloud", "TASK_CATALOG",
             "build_tasks", "Task")
ALLOWED_IMPORTS = {"__future__", "json", "src"}
HARNESS_MODULES = ("evalset_env", "x2_r2_evalset", "x2_r2_agent", "numbers_register", "cloud_redo",
                   "hybrid_full_pipeline", "post_run_check_replay")
GRADER_NAMES = ("end_state_ok", "end_state_diff", "state_ok", "answer_ok", "task_tool_ok", "n_args_correct",
                "tool_args_all", "recall_ok", "expected", "reference")
SENTINEL = "SENTINEL_GRADER_7Q"


def _code_tokens(tree):
    """Identifiers, attributes, import names and string constants, docstrings excluded (as tests/test_post_run_check)."""
    doc_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(
                node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
            doc_nodes.add(id(node.body[0].value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, ast.arg):
            out.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.append(node.name)
        elif isinstance(node, ast.alias):
            out.append(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_nodes:
            out.append(node.value)
    return out


# ── static isolation of the payload builder ─────────────────────────────────────────────────────────

def test_builder_static_never_touches_grader():
    tree = ast.parse(VT_FILE.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("__import__", "open", "exec", "eval"):
            pytest.fail(f"payload builder calls {node.func.id}")
    assert imports <= ALLOWED_IMPORTS, imports
    toks = _code_tokens(tree)
    for bad in FORBIDDEN:
        hits = [t for t in toks if re.search(rf"(?<![A-Za-z]){re.escape(bad)}", t)]
        assert not hits, (bad, hits)


def test_builder_fresh_import_loads_no_harness_module():
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("_vt_fresh", VT_FILE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    new = set(sys.modules) - before
    assert not any(n.split(".")[-1] in HARNESS_MODULES for n in new), new


def test_builder_signatures_take_no_task_object():
    def params(f):
        return list(f.__code__.co_varnames[:f.__code__.co_argcount])
    assert params(VT.build_verdict_payload) == ["task_text", "recall_key", "standing", "rules", "state_before",
                                                "trace", "final_text", "changes", "secrets"]
    assert params(VT.build_rederive_payload) == ["task_text", "recall_key", "standing", "rules", "state_before",
                                                 "secrets"]


def test_messages_carry_the_rolling_breakpoints():
    msgs = VT.messages({"system": "s" * 10, "user": "u"})
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert all(m["content"][-1].get("cache_control") == {"type": "ephemeral"} for m in msgs)


# ── recall-probe suffix ──────────────────────────────────────────────────────────────────────────

def test_recall_suffix_matches_harness():
    import x2_r2_evalset as EV
    assert VT.RECALL_SUFFIX == EV._RECALL_SUFFIX


def test_recall_suffix_stripped_on_every_probe_turn():
    import x2_r2_evalset as EV
    n = 0
    for seed in (20260901, 20260902, 20260903, 20260904, 20260905):
        with contextlib.redirect_stdout(io.StringIO()):
            es = EV.build_session(seed, None, "evalset", False, "qwen3:14b", None, "v2")
        for t, it in zip(es.turns, es.items):
            out = VT.strip_recall_suffix(t.user_text, t.recall_key if t.is_recall else None)
            assert out == it.text
            assert "recall the exact value" not in out
            n += t.is_recall
    assert n > 0
    with pytest.raises(ValueError):
        VT.strip_recall_suffix("Change the unit price of SKU-1 to 2.00.", "ALPHA-CACHE")


# ── dynamic: the payloads the plan builds ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def accepted():
    import verification_tier_plan as P
    return P, P.accepted_turns(REPO)


def _text(p):
    return p["system"] + "\n" + p["user"]


def test_plan_stream_is_accepted_local_results_only(accepted):
    P, turns = accepted
    acc = P.accepted_b3(REPO)
    b3 = [t for t in turns if t["config"] == "B3 v2"]
    assert {(t["tier"], t["seed"], t["turn_idx"]) for t in b3} == set(acc)
    assert all(t["config"] in ("B3 v2", P.Q4_CONFIG) for t in turns)


def test_payload_never_contains_grader_fields_or_reference_calls(accepted):
    P, turns = accepted
    for t in turns:
        p = P.payloads_for(t["es"], t["prior"], t["row"])
        for arm in ("a", "c"):
            txt = _text(p[arm])
            for name in GRADER_NAMES:
                assert name not in txt, (t["tier"], t["seed"], t["turn_idx"], arm, name)
    # sentinel: grader fields on the rows and the session's reference calls / answer values replaced by a sentinel;
    # the payload must not change and must not carry the sentinel
    wrong = [t for t in turns if t["full"]["end_state_ok"] is not True]
    assert wrong
    for t in wrong:
        base = P.payloads_for(t["es"], t["prior"], t["row"])
        tasks = {k: dataclasses.replace(v, reference=({"name": "update_record", "arguments": {"value": SENTINEL}},),
                                        answer_values=(SENTINEL,)) for k, v in t["es"].tasks.items()}
        es2 = dataclasses.replace(t["es"], tasks=tasks)
        poison = {"end_state_ok": SENTINEL, "end_state_diff": [SENTINEL], "state_ok": SENTINEL,
                  "answer_ok": SENTINEL, "expected": SENTINEL}
        row2 = {**t["full"], **poison}
        prior2 = [{**r, **poison} for r in t["prior"]]
        got = P.payloads_for(es2, prior2, row2)
        for arm in ("a", "c"):
            assert got[arm] == base[arm]
            assert SENTINEL not in _text(got[arm])


def test_recall_probe_stripped_and_secrets_redacted_in_payload(accepted):
    P, turns = accepted
    import evalset_env as E
    probes = 0
    for t in turns:
        p = P.payloads_for(t["es"], t["prior"], t["row"])
        turn = p["view"]["turn"]
        for arm in ("a", "c"):
            assert "Also, recall the exact value" not in _text(p[arm])
            assert VT.strip_recall_suffix(turn.user_text, turn.recall_key if turn.is_recall else None).strip() \
                in p[arm]["user"]
            for s in E.secrets_of(p["view"]["env_before"]):
                assert s not in _text(p[arm])
        probes += bool(turn.is_recall)
    assert probes > 0


def test_payload_static_part_is_cacheable_and_shared_within_a_session(accepted):
    P, turns = accepted
    prices = json.loads((REPO / P.PRICE_FILE).read_text(encoding="utf-8"))
    enc = P._enc()
    seen = {}
    for t in turns[:40]:
        p = P.payloads_for(t["es"], t["prior"], t["row"])["a"]
        seen.setdefault((t["config"], t["tier"], t["seed"]), set()).add(p["system"])
        assert len(enc.encode(p["system"])) >= prices["models"][VT.VERIFIER_MODEL]["min_cacheable_tokens"]
    assert all(len(v) == 1 for v in seen.values())
