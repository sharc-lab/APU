"""Pure-logic tests for analysis/make_paper_figures.py's helpers and the A-24 dedup rule. Does not invoke matplotlib
(the fig_* functions are exercised end to end by actually running the script against real pulled data, not here)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import make_paper_figures as mpf  # noqa: E402


def test_med_ignores_none_and_returns_median():
    assert mpf._med([1.0, 2.0, 3.0]) == 2.0
    assert mpf._med([1.0, None, 3.0]) == 2.0
    assert mpf._med([]) is None
    assert mpf._med([None, None]) is None


def test_a24_dedup_keeps_last_bisect_result_per_label(tmp_path):
    """Regression: the amech file has two bisect_result rows for qwen3-8b -- an earlier probe that hit its
    max_ctx_native ceiling (~23,100 MiB, a different boundary entirely) before a later, correctly-configured probe
    reached the real memory-limited boundary (~47,770 MiB, the value docs/CLAIMS_LEDGER.md A-24 actually cites).
    Taking the last bisect_result per label must select the second one, not silently plot both or pick the first."""
    lines = [
        '{"record": "bisect_result", "label": "qwen3-8b", "projected_mib_last_ok": 23103.0, "projected_mib_first_fail": 23139.0, "budget_B_mib": 47865.0}',
        '{"record": "bisect_result", "label": "qwen3-8b", "projected_mib_last_ok": 47753.0, "projected_mib_first_fail": 47789.0, "budget_B_mib": 47865.0}',
        '{"record": "bisect_result", "label": "qwen3-32b", "projected_mib_last_ok": 47650.0, "projected_mib_first_fail": 47714.0, "budget_B_mib": 47865.0}',
    ]
    p = tmp_path / "amech.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = mpf._load(p)
    boundaries = [r for r in rows if r.get("record") == "bisect_result"]
    seen = {}
    for r in boundaries:
        seen[r.get("label")] = r
    deduped = list(seen.values())
    qwen8b = next(r for r in deduped if r["label"] == "qwen3-8b")
    assert qwen8b["projected_mib_last_ok"] == 47753.0
    assert len(deduped) == 2


def test_load_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    rows = mpf._load(p)
    assert len(rows) == 2
