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


def test_oct26_file_matches_builder_and_dry_run_passes():
    items = json.loads(wq.OCT26_PATH.read_text(encoding="utf-8"))
    assert items == wq.build_oct26()
    rep = wq.dry_run(items, prefix=wq.OCT26_PREFIX)
    assert rep["ok"], [e for e in rep["entries"] if not e["ok"]]


def test_oct26_is_separate_from_the_week_and_gated_on_its_own_preflight():
    o26 = wq.build_oct26()
    week = json.loads(wq.QUEUE_PATH.read_text(encoding="utf-8"))
    assert not {it["id"] for it in o26} & {it["id"] for it in week}
    assert all(it["id"].startswith("t2s_o26_") for it in o26)
    assert "gate" not in o26[0] and all(it["gate"]["requires_done"].startswith("t2s_o26_") for it in o26[1:])
    for it in o26:
        if "_b5_" in it["id"]:
            assert it["est_hours"] > 0 and it["est_basis"].endswith("(ESTIMATE)")
        else:
            assert it["est_hours"] is None and it["est_basis"] == "none"


def test_oct26_b6_covers_both_models_at_4096_and_the_igpu_window():
    by = {it["id"]: it["cmd"] for it in wq.build_oct26()}
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
        assert c[c.index("--seeds") + 1] == "20260901,20260902,20260903,20260904,20260905"
        cells.add((c[c.index("--platform") + 1], c[c.index("--models") + 1]))
    assert cells == {(p, m) for p in ("evo-t2s-cpu", "evo-t2s") for m in ("llama3.1:8b", "qwen3:14b")}
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
