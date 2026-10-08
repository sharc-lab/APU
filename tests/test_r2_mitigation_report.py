"""analysis/r2_mitigation_report.py and its register rows, on synthetic mitigation / real-run rows (fakes, no data
files from the repo)."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
sys.path.insert(0, str(REPO / "harness"))
import r2_mitigation_report as mr  # noqa: E402
import numbers_register as nr  # noqa: E402

SEEDS = (20260901, 20260902, 20260903)
RIU = {"llama3.1:8b": ["rule1_json_keys", "rule3_no_zebra", "rule4_metres"],
       "qwen3:14b": ["rule1_json_keys", "rule2_log_event_called", "rule3_no_zebra", "rule4_metres",
                     "rule5_session_code"]}
THINK = {"llama3.1:8b": None, "qwen3:14b": False}


def _obs(shifts=0, drops=False):
    return {"message_level_truncation": drops, "token_level_cut": False, "context_shifts": shifts,
            "exceed_context_error": False, "log_slice_complete": True, "client_exact_vs_server_diff": 0}


def _turn(mode, model, arm, seed, t, fail_rules=(), shifts=0, think="default", h_miss=True):
    row = {"record": "r2a_turn", "mode": mode, "model_id": model, "arm_id": arm, "seed": seed, "turn_idx": t,
           "attempt_id": f"{mode}-{model}-{arm}-{seed}", "think": THINK[model] if think == "default" else think,
           "call2_mode": "off", "call2_tools": False, "canary_check": t % 5 == 0, "any_http_error": False,
           "thinking_present": False}
    for r in ("rule1_json_keys", "rule2_log_event_called", "rule3_no_zebra", "rule5_session_code"):
        row[r] = r not in fail_rules
    row["rule4_metres"] = (r4 := ("rule4_metres" not in fail_rules)) if t % 3 == 0 else None
    if t % 5 == 0:
        row["canary_sys_ok"], row["canary_hist_ok"] = True, not h_miss
    calls = []
    for i in (1, 2):
        c = {"content": ""}
        if mode == "mitigation":
            c["client_trim"] = {"applied": True, "count_method": "render_tokenize", "over_budget": False,
                                "kept_prompt_tokens_exact": 3000, "budget_prompt_tokens": 3507, "num_predict": 384,
                                "kept_turns": [t], "n_turns_dropped": t - 1,
                                "observed": _obs(shifts if (i == 2 and t == 7) else 0)}
        calls.append(c)
    row["calls"] = calls
    return row


def _session(mode, model, arm, seed):
    return {"record": "r2a_session", "mode": mode, "model_id": model, "arm_id": arm, "seed": seed,
            "attempt_id": f"{mode}-{model}-{arm}-{seed}", "call2_mode": "off", "call2_tools": False}


def _write(path, mode, arms, fail=None, turns=40, shifts=0, think="default", r2m=True, seeds=SEEDS):
    fail = fail or {}
    rows = [{"record": "run_start", "mode": mode, "rules_in_use": RIU}]
    for arm in arms:
        for model in RIU:
            for seed in seeds:
                for t in range(1, turns + 1):
                    fr = fail.get((model, arm), {}).get(t, ())
                    rows.append(_turn(mode, model, arm, seed, t, fr, shifts, think))
                    if mode == "mitigation" and r2m:
                        for i in (1, 2):
                            rows.append({"record": "r2m_call", "mode": "mitigation", "model_id": model,
                                         "arm_id": arm, "seed": seed, "turn_idx": t, "call_idx": i})
                rows.append(_session(mode, model, arm, seed))
    rows.append({"record": "run_end"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


A4, A8 = "ollama_ctx_4096_call2_notools", "ollama_ctx_8192_call2_notools"


def _build(tmp_path, mit_kw=None, real4=True, real8=False):
    mit = _write(tmp_path / "mit.jsonl", "mitigation", [A4, A8], **(mit_kw or {}))
    # the real run at 4096: llama rule 1 fails from turn 3 on in every session (the v1 pattern, made up here)
    fail_real = {("llama3.1:8b", A4): {t: ("rule1_json_keys",) for t in range(3, 41)}}
    paths = {}
    if real4:
        paths["num_ctx_4096"] = _write(tmp_path / "v1.jsonl", "real", [A4, "ollama_default_call2_notools"],
                                       fail=fail_real)
    else:
        paths["num_ctx_4096"] = tmp_path / "absent_v1.jsonl"
    paths["num_ctx_8192"] = (_write(tmp_path / "v1b.jsonl", "real", [A8], seeds=SEEDS + (20260904,)) if real8
                             else tmp_path / "x2_r2_real_v1b.jsonl")
    return mr.build(mit, real_paths=paths, validation_path=tmp_path / "no_validation.jsonl")


def test_recovered_and_supported_with_real_comparison(tmp_path):
    rep = _build(tmp_path, real8=True)
    assert rep["overall_verdict"] == "supported"
    c = rep["cells"]["num_ctx_4096|llama3.1:8b"]
    assert c["verdict"]["verdict"] == "recovered" and c["mitigation"]["rules"]["rule1_json_keys"]["rate"] == 1.0
    assert c["observed"]["context_shift_events"] == 0 and c["observed"]["r2m_call_rows"] == 240
    real = c["real"]["metrics"]
    assert c["real"]["available"] and real["seeds"] == list(SEEDS)
    assert real["rules"]["rule1_json_keys"]["first_fail_per_seed"] == [3, 3, 3]
    assert real["rules"]["rule1_json_keys"]["rate"] == pytest.approx(6 / 120)
    assert real["first_rule_failure_per_seed"] == [3, 3, 3]
    # the 8192 comparison only uses the mitigation seeds (20260904 in the real file is ignored)
    assert rep["cells"]["num_ctx_8192|qwen3:14b"]["real"]["metrics"]["seeds"] == list(SEEDS)
    # H misses are expected (memory loss) and do not affect the verdict
    assert c["mitigation"]["canary_hist_misses"] == 24 and c["mitigation"]["canary_sys_misses"] == 0
    md = mr.format_markdown(rep)
    assert "supported" in md and "mitigation" in md and "real (v1.jsonl)" in md


def test_not_recovered_refutes(tmp_path):
    fail = {("qwen3:14b", A4): {t: ("rule5_session_code",) for t in range(5, 41, 2)}}
    rep = _build(tmp_path, mit_kw={"fail": fail})
    c = rep["cells"]["num_ctx_4096|qwen3:14b"]
    assert c["verdict"] == {"verdict": "not_recovered", "valid": True, "reasons": [],
                            "rules_below_threshold": ["rule5_session_code"]}
    assert rep["overall_verdict"] == "refuted"
    assert c["mitigation"]["rules"]["rule5_session_code"]["first_fail_per_seed"] == [5, 5, 5]


def test_context_shift_makes_the_cell_invalid_and_overall_inconclusive(tmp_path):
    rep = _build(tmp_path, mit_kw={"shifts": 2})
    v = rep["cells"]["num_ctx_4096|llama3.1:8b"]["verdict"]
    assert v["verdict"] == "invalid" and any("context shifts" in r for r in v["reasons"])
    assert rep["overall_verdict"] == "inconclusive"


def test_think_not_false_for_qwen_is_invalid(tmp_path):
    rep = _build(tmp_path, mit_kw={"think": None})
    v = rep["cells"]["num_ctx_4096|qwen3:14b"]["verdict"]
    assert v["verdict"] == "invalid" and any("think false" in r for r in v["reasons"])
    assert rep["cells"]["num_ctx_4096|llama3.1:8b"]["verdict"]["verdict"] == "recovered"


def test_missing_log_rows_and_seeds_are_invalid(tmp_path):
    rep = _build(tmp_path, mit_kw={"r2m": False})
    assert any("r2m_call rows" in r for r in rep["cells"]["num_ctx_4096|llama3.1:8b"]["verdict"]["reasons"])
    rep = _build(tmp_path, mit_kw={"seeds": SEEDS[:2]})
    assert any("expected" in r for r in rep["cells"]["num_ctx_4096|llama3.1:8b"]["verdict"]["reasons"])


def test_missing_real_file_is_not_available(tmp_path):
    rep = _build(tmp_path, real4=False)
    assert rep["cells"]["num_ctx_4096|llama3.1:8b"]["real"]["available"] is False
    assert rep["cells"]["num_ctx_8192|llama3.1:8b"]["real"] == {"available": False, "file": "x2_r2_real_v1b.jsonl"}
    assert "not available" in mr.format_markdown(rep)


def test_register_rows_compute_and_pending(tmp_path):
    res = tmp_path / "results"
    _write(res / "x2_r2_mitigation_v1.jsonl", "mitigation", [A4, A8])
    _write(res / "x2_r2_real_v1.jsonl", "real", [A4])
    v = nr.compute_r2_mitigation_v1_verdict(tmp_path)
    assert v["value"].startswith("pre-registered verdict at num_ctx_4096: supported")
    assert "llama3.1:8b recovered" in v["value"]
    c4 = nr.compute_r2_mitigation_v1_4096(tmp_path)["value"]
    assert "llama3.1:8b mitigation" in c4 and "real (x2_r2_real_v1.jsonl)" in c4
    o = nr.compute_r2_mitigation_v1_observed(tmp_path)["value"]
    assert "0 context shifts" in o
    with pytest.raises(FileNotFoundError):
        nr.compute_r2_mitigation_v1_8192(tmp_path)
    entries = {e["claim_id"]: e for e in nr.NUMBER_ENTRIES}
    for cid in ("R2-mitigation-v1-verdict", "R2-mitigation-v1-4096", "R2-mitigation-v1-8192",
                "R2-mitigation-v1-observed"):
        assert entries[cid]["pending_ok"] and "results/x2_r2_mitigation_v1.jsonl" in entries[cid]["data_files"]
        assert nr._pending(entries[cid], repo=tmp_path / "empty")
    assert "results/x2_r2_real_v1b.jsonl" in entries["R2-mitigation-v1-8192"]["data_files"]
