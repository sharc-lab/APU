"""queues/t2s_oct26_queue.json: staged for the requested 2026-10-26 to 2026-11-08 evo-t2s window (A5, B5, B6),
never part of the week's queue, file equal to its builder, dry run passing. No machine, no SSH."""
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "scripts"))

import t2s_week_queue as wq  # noqa: E402
import policy_compare as pc  # noqa: E402


def test_oct26_file_matches_builder_and_dry_run_passes():
    items = json.loads(wq.OCT26_PATH.read_text(encoding="utf-8"))
    assert items == wq.build_oct26()
    rep = wq.dry_run(items, prefix=wq.OCT26_PREFIX)
    # The window was loaded on 2026-10-28 and its outputs now sync into results/; "already exists" is expected for
    # entries that have run, so only other problems fail the staged file.
    bad = [(e["id"], p) for e in rep["entries"] for p in e.get("problems", []) if "already exists" not in p]
    assert not bad, bad


def test_oct26_is_separate_from_the_week_and_gated_on_its_own_preflight():
    o26 = wq.build_oct26()
    week = json.loads(wq.QUEUE_PATH.read_text(encoding="utf-8"))
    assert not {it["id"] for it in o26} & {it["id"] for it in week}
    assert all(it["id"].startswith("t2s_o26_") for it in o26)
    assert "gate" not in o26[0] and all(it["gate"]["requires_done"].startswith("t2s_o26_") for it in o26[1:])
    for it in o26:
        if "_b5_" in it["id"]:
            assert it["est_hours"] > 0 and it["est_basis"].endswith("(ESTIMATE)")
        elif "_b6_" in it["id"]:   # amendment 2026-10-21 (later): measured-rate estimates, labelled, or nominal
            assert it["est_hours"] is None or (it["est_hours"] > 0 and it["est_basis"].startswith(("ESTIMATE",
                                                                                                   "nominal")))
        elif "_b3_" in it["id"]:   # B3 v2 Intel task accuracy (2026-10-22): measured-rate estimates, labelled
            assert it["est_hours"] > 0 and it["est_basis"].startswith("ESTIMATE")
        else:
            assert it["est_hours"] is None and it["est_basis"] == "none"


def test_oct26_b6_covers_both_models_at_4096_and_the_igpu_window():
    by = {it["id"]: it["cmd"] for it in wq.oct26_all_specs()}   # every B6 spec, staged or not
    cpu, igpu = by["t2s_o26_b6_real_cpu_4096"], by["t2s_o26_b6_real_igpu_default"]
    for cmd in (cpu, igpu):
        assert "qwen3:32b,llama3.3:70b" in cmd and "--require-validation-gates" in cmd
    assert cpu[cpu.index("--tiers") + 1] == "ollama_ctx_4096" and "cpu_default" in cpu
    assert igpu[igpu.index("--tiers") + 1] == "ollama_default" and "igpu_enable" in igpu
    assert wq.OCT26_PENDING == {}


def test_oct26_b5_entries_cover_both_platforms_and_models_before_b6():
    o26 = wq.build_oct26()
    ids = [it["id"] for it in o26]
    b5 = [it for it in o26 if "_b5_" in it["id"]]
    cells = set()
    for it in b5:
        c = it["cmd"]
        assert c[1] == "policy_compare.py" and it["gate"]["requires_done"] == "t2s_o26_preflight"
        cell = (c[c.index("--platform") + 1], c[c.index("--models") + 1])
        want = ("20260901,20260902,20260903" if cell == ("evo-t2s-cpu", "qwen3:14b")
                else "20260901,20260902,20260903,20260904,20260905")
        assert c[c.index("--seeds") + 1] == want
        # amendment 2026-10-22: per-cell sets, the llama3.1:8b cells a-f, the qwen3:14b CPU cell without (b)
        want_p = "a,c,d,e,f" if cell == ("evo-t2s-cpu", "qwen3:14b") else ",".join(pc.T2S_POLICIES)
        assert c[c.index("--policies") + 1] == want_p
        cells.add(cell)
    # amendment 2026-10-21 (later): llama3.1:8b CPU then iGPU at 5 seeds first; qwen3:14b only on the CPU at 3 seeds
    # and only when it fits the launch budget; never the iGPU qwen3:14b cell
    import t2s_window_stage as ws
    llama = [("evo-t2s-cpu", "llama3.1:8b"), ("evo-t2s", "llama3.1:8b")]
    assert [b["id"] for b in b5][:2] == ["t2s_o26_b5_cpu_llama31_8b", "t2s_o26_b5_igpu_llama31_8b"]
    assert cells == set(llama) | ({("evo-t2s-cpu", "qwen3:14b")} if ws.qwen_decision()["fits"] else set())
    assert max(ids.index(it["id"]) for it in b5) < min(i for i, x in enumerate(ids) if "_b6_" in x)
    assert ids[:2] == ["t2s_o26_preflight", "t2s_o26_window_override_methods"]


def test_t2s_r2_agent_registers_think_values_for_the_large_models():
    import t2s_r2_agent as w
    fake = types.SimpleNamespace(MODELS={"llama3.1:8b": None})
    w.register_extra_models(fake)
    assert fake.MODELS["qwen3:32b"] is False and "llama3.3:70b" in fake.MODELS
    import x2_r2_scale as sc
    for m, t in w.EXTRA_MODEL_THINK.items():
        assert sc.SCALE_THINK[m] == t


def test_oct26_qwen_decision_is_computed_from_the_hours_left():
    import t2s_window_stage as ws
    q = ws.qwen_decision()
    assert q["qwen_seeds"] == 3 and q["policies"] == ["a", "c", "d", "e", "f"]
    assert q["llama_policies"] == ["a", "b", "c", "d", "e", "f"]
    assert q["qwen_cell"] == "evo-t2s-cpu qwen3:14b" and q["slack_h"] == 24.0
    assert q["budget_h"] == round(q["available_h"] - 24.0, 1)
    assert q["left_h"] == round(ws.budget_h() - q["before_qwen_h"], 1)
    assert q["fits"] == (q["qwen_h"] <= ws.budget_h() - q["before_qwen_h"])
    assert set(q["llama_h"]) == {"evo-t2s-cpu llama3.1:8b", "evo-t2s llama3.1:8b"}


def test_select_b6_fills_in_order_respects_gates_and_puts_unestimated_last(monkeypatch):
    import t2s_window_stage as ws
    monkeypatch.setattr(ws, "budget_h", lambda: 100.0)
    specs = [{"id": "p", "gate": "pre", "est": 1.0}, {"id": "vc", "gate": "p", "est": 30.0},
             {"id": "vg", "gate": "p", "est": None}, {"id": "rc", "gate": "vc", "est": 50.0},
             {"id": "rg", "gate": "vg", "est": 10.0}, {"id": "big", "gate": "p", "est": 500.0}]
    sel = ws.select_b6(specs, start_h=10.0)
    assert sel["staged"] == ["p", "vc", "rc", "vg", "rg"] and set(sel["not_staged"]) == {"big"}
    assert sel["not_staged"]["big"].startswith("does not fit")
    # a failed gate drops its dependants
    sel = ws.select_b6([{"id": "vc", "gate": "pre", "est": 95.0}, {"id": "rc", "gate": "vc", "est": 1.0}],
                       start_h=10.0)
    assert sel["staged"] == [] and sel["not_staged"]["rc"] == "gate vc not staged"


def test_oct26_plan_every_estimated_entry_ends_within_the_slack_budget():
    import t2s_window_stage as ws
    p = ws.plan()
    assert p["slack_h"] == 24.0 and p["budget_h"] == round(p["available_h"] - 24.0, 1)
    for r in p["entries"]:
        if r.get("end_h") is not None:
            assert r["end_h"] <= p["budget_h"] and r["fits"] == "yes" and r["slack_h"] >= 24.0
    staged = {r["id"] for r in p["entries"]}
    assert not staged & {x["id"] for x in p["not_staged"]}
    every_b6 = {s["id"] for s in wq.oct26_all_specs() if "_b6_" in s["id"]}
    assert every_b6 == {r["id"] for r in p["entries"] if "_b6_" in r["id"]} | {x["id"] for x in p["not_staged"]}


def test_b6_estimates_are_labelled_with_their_factor():
    import t2s_window_stage as ws
    est = ws.b6_estimates()
    assert est["b6_model_pull"]["basis"].startswith("nominal")
    for k in ("b6_val_cpu", "b6_val_igpu", "b6_real_cpu_4096", "b6_real_igpu_default"):
        if est[k]["hours"] is not None:
            assert est[k]["basis"].startswith("ESTIMATE") and "factor" in est[k]["basis"]
    if est["b6_real_igpu_default"]["hours"] is not None:
        assert "ASSUMED 32768" in est["b6_real_igpu_default"]["basis"]


def test_oct26_per_cell_policy_sets_and_plan_labels():
    import t2s_window_stage as ws
    assert pc.T2S_POLICIES == ("a", "b", "c", "d", "e", "f")
    assert wq.oct26_cell_policies("evo-t2s-cpu", "qwen3:14b") == "a,c,d,e,f"
    assert wq.oct26_cell_policies("evo-t2s-cpu", "llama3.1:8b") == wq.oct26_cell_policies("evo-t2s", "llama3.1:8b") \
        == "a,b,c,d,e,f"
    p = ws.plan()
    for r in p["entries"]:
        if "_b5_" in r["id"]:
            assert r["policies"] == ("a,c,d,e,f" if r["id"] == "t2s_o26_b5_cpu_qwen3_14b" else "a,b,c,d,e,f")
    if ws.qwen_decision()["fits"]:
        assert p["b5_policies"]["t2s_o26_b5_cpu_qwen3_14b"] == "a,c,d,e,f"
