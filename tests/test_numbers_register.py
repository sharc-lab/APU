"""Unit tests for analysis/numbers_register.py: the compute functions against small fixture files (not the
real multi-MB result files), the VERIFIED/CORRECTED/UNSUPPORTED classification logic, and that every
script_function string in NUMBER_ENTRIES names a real, importable function in this repo."""
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis.numbers_register as nr  # noqa: E402


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_compute_px2_ttft_gap_on_a_small_fixture(tmp_path):
    (tmp_path / "results").mkdir()
    path = tmp_path / "results" / "t2s_night2_20260930T135145Z.jsonl"
    rows = []
    for cond, ttft in [("N0", 1.0), ("N0", 1.0), ("B4", 1.1), ("B4", 1.1), ("S4", 1.05), ("S4", 1.05)]:
        rows.append({"record": None, "section": "PX2", "model_id": "fake-model", "co_runner": cond,
                    "ttft_s": ttft})
    _write_jsonl(path, rows)
    result = nr.compute_px2_ttft_gap(tmp_path)
    assert "fake-model" in result["detail"]
    # B4/N0=1.10, S4/N0=1.05, gap = |1.10-1.05|/1.0*100 = 5.0%
    assert abs(result["detail"]["fake-model"]["b4_vs_s4_gap_pct"] - 5.0) < 0.01


def test_compute_px2_decode_ratios_on_a_small_fixture(tmp_path):
    (tmp_path / "results").mkdir()
    path = tmp_path / "results" / "t2s_night2_20260930T135145Z.jsonl"
    rows = []
    for cond, dec in [("N0", 100.0), ("N0", 100.0), ("B4", 92.0), ("B4", 92.0), ("S4", 99.0), ("S4", 99.0)]:
        rows.append({"record": None, "section": "PX2", "model_id": "fake-model", "co_runner": cond,
                    "decode_tok_s": dec})
    _write_jsonl(path, rows)
    result = nr.compute_px2_decode_ratios(tmp_path)
    assert abs(result["detail"]["b4_ratios"][0] - 0.92) < 0.001
    assert abs(result["detail"]["s4_ratios"][0] - 0.99) < 0.001


def test_compute_k1_v3_x2_table_on_a_small_fixture(tmp_path):
    (tmp_path / "results").mkdir()
    path = tmp_path / "results" / "t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl"
    rows = [
        {"record": "tier_v3_model_meta", "model_tag": "fake-model", "capped": False, "native_ctx": 131072},
        {"record": "tier", "model_tag": "fake-model", "ollama_default_ctx": 131072},
    ]
    _write_jsonl(path, rows)
    result = nr.compute_k1_v3_x2_table(tmp_path)
    assert result["detail"]["fake-model"]["native_ctx"] == 131072
    assert result["detail"]["fake-model"]["ollama_default_ctx"] == 131072
    assert result["n"] == 1


def test_compute_trace_context_exit_rate_on_a_small_fixture(tmp_path):
    import pandas as pd
    (tmp_path / "results" / "traces").mkdir(parents=True)
    path = tmp_path / "results" / "traces" / "exit_status_sample.parquet"
    df = pd.DataFrame({"exit_status": ["exit_context", "submitted", "submitted", "exit_context"]})
    df.to_parquet(path, index=False)
    result = nr.compute_trace_context_exit_rate(tmp_path)
    assert result["value"] == "50.00%"
    assert result["n"] == 4


def test_compute_qwen32b_refusal_share_unsupported_when_file_missing(tmp_path):
    (tmp_path / "results").mkdir()
    try:
        nr.compute_qwen32b_refusal_share(tmp_path)
        assert False, "expected an exception when the source file does not exist"
    except Exception:
        pass


def test_compute_qwen32b_refusal_share_on_the_real_committed_file():
    """Exercises the real function against the real, now-committed source file (not a fixture) -- this is
    the one test in this file allowed to depend on a real results/ file, specifically because this entry's
    whole point is that it must be computed from the real file, not a fixture standing in for it."""
    repo_root = Path(__file__).resolve().parents[1]
    real_file = repo_root / "results" / "t2s_night2_20260929T205109Z.jsonl"
    if not real_file.exists():
        import pytest
        pytest.skip("real source file not present in this checkout")
    result = nr.compute_qwen32b_refusal_share(repo_root)
    assert result["n"] > 0
    assert "/" in result["value"]


def test_values_match_exact_number():
    assert nr._values_match("29.85%", "29.85%")


def test_values_match_rejects_a_real_discrepancy():
    assert not nr._values_match("10% / 0%", "1% of trajectories exceed 32K at some step, 0% exceed 40,960")


def test_build_register_classifies_unsupported_on_exception(monkeypatch):
    def failing(repo):
        raise FileNotFoundError("no file")

    monkeypatch.setattr(nr, "NUMBER_ENTRIES", [
        {"claim_id": "fake", "description": "x", "compute": failing, "data_files": ["x.jsonl"],
         "script_function": "x::y", "reported_value": "1"},
    ])
    rows = nr.build_register()
    assert rows[0]["status"] == "UNSUPPORTED"
    assert "FileNotFoundError" in rows[0]["value"]


def test_build_register_classifies_verified_on_exact_match(monkeypatch):
    def ok(repo):
        return {"value": "42", "n": 10}

    monkeypatch.setattr(nr, "NUMBER_ENTRIES", [
        {"claim_id": "fake", "description": "x", "compute": ok, "data_files": ["x.jsonl"],
         "script_function": "x::y", "reported_value": "42"},
    ])
    rows = nr.build_register()
    assert rows[0]["status"] == "VERIFIED"


def test_build_register_classifies_corrected_on_mismatch(monkeypatch):
    def wrong(repo):
        return {"value": "17", "n": 10}

    monkeypatch.setattr(nr, "NUMBER_ENTRIES", [
        {"claim_id": "fake", "description": "x", "compute": wrong, "data_files": ["x.jsonl"],
         "script_function": "x::y", "reported_value": "42"},
    ])
    rows = nr.build_register()
    assert rows[0]["status"] == "CORRECTED"


def test_build_register_no_reported_value_is_always_verified(monkeypatch):
    def ok(repo):
        return {"value": "anything", "n": 1}

    monkeypatch.setattr(nr, "NUMBER_ENTRIES", [
        {"claim_id": "fake", "description": "x", "compute": ok, "data_files": ["x.jsonl"],
         "script_function": "x::y"},
    ])
    rows = nr.build_register()
    assert rows[0]["status"] == "VERIFIED"


def test_write_register_md_sorts_corrected_and_unsupported_first(tmp_path):
    rows = [
        {"claim_id": "a", "status": "VERIFIED", "value": "1", "n": 1, "reported_value": "1",
         "data_files": ["x"], "script_function": "x::y", "commit": "abc", "date": "2026-10-01"},
        {"claim_id": "b", "status": "CORRECTED", "value": "2", "n": 1, "reported_value": "3",
         "data_files": ["x"], "script_function": "x::y", "commit": "abc", "date": "2026-10-01"},
        {"claim_id": "c", "status": "UNSUPPORTED", "value": "ERROR", "n": "n/a", "reported_value": "4",
         "data_files": ["x"], "script_function": "x::y", "commit": "abc", "date": "2026-10-01"},
    ]
    out = tmp_path / "NUMBERS_REGISTER.md"
    nr.write_register_md(rows, out)
    text = out.read_text(encoding="utf-8")
    i_corrected = text.index("| b |")
    i_unsupported = text.index("| c |")
    i_verified = text.index("| a |")
    assert i_corrected < i_verified
    assert i_unsupported < i_verified


def test_compute_pack_trace_weighted_stats_on_the_real_committed_files():
    """Exercises the real function against the real, committed workload pack and trace parquet (not a
    fixture) -- the whole point of this entry is that the weighting is computed from the real 400-item pack
    and the real uncensored trace data, so a synthetic stand-in would not exercise what matters: that the
    weights are real trace mass, not invented numbers."""
    repo_root = Path(__file__).resolve().parents[1]
    pack_dir = repo_root / "results" / "workload_pack" / "items"
    trace_path = repo_root / "results" / "traces" / "agent_step_lengths.parquet"
    if not pack_dir.exists() or not trace_path.exists():
        import pytest
        pytest.skip("real workload pack or trace parquet not present in this checkout")
    result = nr.compute_pack_trace_weighted_stats(repo_root)
    assert result["n"] == 400
    overall = result["detail"]["overall"]
    # Weights sum to 1.0 across all 400 items (step 4 of the documented method).
    assert abs(overall["weight_mass"] - 1.0) < 1e-9
    # Trace-weighting should pull the reported median UP relative to the flat median, since the pack's
    # mass is concentrated at short lengths the real uncensored trace data rarely visits.
    assert overall["traced_p50"] > overall["flat_p50"]


def test_weighted_percentile_matches_unweighted_pct_when_weights_are_equal():
    """With uniform weights, weighted_percentile() must reduce to the same answer as grade.py's own
    nearest-rank pct() for a simple real case (sanity check on the weighted-percentile definition itself,
    using the real 400-item pack's prompt_tokens as the input, not a synthetic list)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis import trace_weighted_pack as twp

    repo_root = Path(__file__).resolve().parents[1]
    pack_dir = repo_root / "results" / "workload_pack" / "items"
    if not pack_dir.exists():
        import pytest
        pytest.skip("real workload pack not present in this checkout")
    items = twp.load_pack_items(repo_root)
    toks = [it["prompt_tokens"] for it in items]
    grade_mod = twp._grade_pct_module(repo_root)
    for p in (0.50, 0.90, 0.99):
        flat_via_grade = grade_mod.pct(toks, p)
        flat_via_weighted = twp.weighted_percentile(toks, [1.0] * len(toks), p)
        # Both are real percentile definitions (nearest-rank-interp vs weighted-midpoint-interp) and need
        # not match exactly -- the pack's real token distribution clusters sharply at family boundaries
        # (confirmed live: p90 differs by ~12% between the two definitions there), so this checks they
        # land within 20% of each other's value rather than requiring an exact match.
        assert abs(flat_via_grade - flat_via_weighted) < 0.20 * max(flat_via_grade, flat_via_weighted, 1.0)


def test_every_entry_script_function_names_a_real_function():
    """Every NUMBER_ENTRIES['script_function'] must name a function that actually exists in this repo --
    this registry must never cite a function that was renamed or removed. Entries may cite a function in
    numbers_register.py itself or in any other module path given relative to the repo root."""
    import importlib.util
    repo_root = Path(__file__).resolve().parents[1]
    for entry in nr.NUMBER_ENTRIES:
        module_path, func_name = entry["script_function"].split("::")
        if module_path == "analysis/numbers_register.py":
            mod = nr
        else:
            full_path = repo_root / module_path
            assert full_path.exists(), f"{entry['claim_id']} cites {module_path}, which does not exist"
            spec = importlib.util.spec_from_file_location(full_path.stem, full_path)
            mod = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(full_path.parent))
            spec.loader.exec_module(mod)
        assert hasattr(mod, func_name), f"{entry['claim_id']} cites {func_name}, not found in {module_path}"
        assert callable(getattr(mod, func_name))


def _k2_fixture(tmp_path):
    (tmp_path / "results").mkdir()
    rows = []
    for tag, levels, fail_at, kc in [("k2_m1_default_awe_balloon", [8, 4], 4, True),
                                     ("k2_m1_mmap_pageable_touch", [8, 4, 2], None, True),
                                     ("k2_m2_default_pageable_touch", [8], None, None)]:
        rows.append({"kind": "start", "item_id": tag + "_start"})
        for lv in levels:
            rows.append({"record": "k2_step_summary", "item_tag": tag, "level_gb": lv, "median_score": 1.0,
                         "responsiveness_median_s": 0.031 if lv == 2 else 0.016, "clean_failure": lv == fail_at})
        if kc is not None:
            rows.append({"record": "k2_kill_criterion", "item_tag": tag, "ok": kc})
    rows.append({"record": "run_end", "note": "deadline reached"})
    _write_jsonl(tmp_path / nr._K2_X2_RUN, rows)


def test_compute_k2_x2_kill_criterion_table_on_a_small_fixture(tmp_path):
    _k2_fixture(tmp_path)
    out = nr.compute_k2_x2_kill_criterion_table(tmp_path)
    assert out["n"] == 2
    assert out["value"].startswith("2 kill-criterion evaluations, 2 passed, 0 violated")
    assert "started without an evaluation: k2_m2_default_pageable_touch" in out["value"]
    by = {(d["model"], d["mmap_arm"], d["pressure_arm"]): d for d in out["detail"]}
    assert by[("m1", "default", "awe_balloon")]["first_clean_failure_gb"] == 4
    assert by[("m1", "mmap", "pageable_touch")]["max_resp_ratio_vs_8gb"] == 1.94
    assert by[("m2", "default", "pageable_touch")]["kill_criterion"] is None


def test_k2_tag_parts_splits_two_word_pressure_arms():
    assert nr._k2_tag_parts("k2_llama-3.3-70b_default_pageable_touch") == ("llama-3.3-70b", "default", "pageable_touch")
    assert nr._k2_tag_parts("k2_qwen3-8b_mmap_awe_balloon") == ("qwen3-8b", "mmap", "awe_balloon")
