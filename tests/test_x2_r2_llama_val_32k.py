"""x2_r2_llama_val_32k (Blade confound check on evo-x2, pre-registered in docs/FINDINGS.md 2026-10-10): the agent's
--neg-ctx / --neg-sessions plan against harness/blade_r2.py's, the sidecar launcher's import check, and the register
row's pre-registered verdict on synthetic rows (plus on Blade run 2's own file, whose arm and seeds are the same)."""
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))

import analysis.numbers_register as nr  # noqa: E402
import blade_r2  # noqa: E402
import x2_r2_agent as ag  # noqa: E402
import x2_r2_sidecar as sidecar  # noqa: E402

ARM = "ollama_ctx_32768_negative_control_call2_notools"
SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
JOB_ARGV = ["--mode", "validation", "--call2-tools", "off", "--models", "llama3.1:8b", "--neg-ctx", "32768",
            "--neg-sessions", "5", "--out", "C:\\apu\\ovn\\results\\x2_r2_llama_val_32k.jsonl"]


# ── plan: the same as Blade run 2 ────────────────────────────────────────────────────────────────────

def test_neg_ctx_plan_equals_blade_run2_plan():
    plan, neg_arm = ag.validation_plan_neg("off", 32768, 5)
    assert neg_arm == blade_r2.BLADE_NEG_ARM == "ollama_ctx_32768_negative_control"
    assert plan == blade_r2.plan_for("validation", [], neg_sessions=5)
    assert plan[0] == (ARM, SEEDS, 10)
    assert plan[1] == ("ollama_ctx_8192_positive_control_call2_notools", (20260901,), 15)
    assert plan[2] == ("ollama_ctx_32768_negative_control", tuple(ag.SEEDS), 10)
    assert ag.ARMS[ARM] == {"num_ctx": 32768, "call2_tools": False, "call2_mode": "off"}


def test_neg_defaults_are_the_old_plan():
    assert ag.validation_plan_neg("off") == (ag.validation_plan("off"), ag.NEG_ARM)
    assert ag.validation_plan_neg("off", 131072)[1] == ag.NEG_ARM
    with pytest.raises(ValueError):
        ag.validation_plan_neg("off", 32768, 6)


def test_job_argv_parses_and_neg_flags_refused_outside_validation(tmp_path):
    args = ag.build_arg_parser().parse_args(JOB_ARGV)
    assert (args.neg_ctx, args.neg_sessions, args.call2_tools, args.models) == (32768, 5, "off", "llama3.1:8b")
    with pytest.raises(SystemExit):
        ag.main(["--mode", "real", "--neg-ctx", "32768", "--out", str(tmp_path / "x.jsonl")], advance=False)


# ── sidecar launcher ─────────────────────────────────────────────────────────────────────────────────

def test_sidecar_import_check_loads_the_side_copy(tmp_path):
    side = tmp_path / "ovn" / "r2v32k"
    side.mkdir(parents=True)
    shutil.copy(REPO / "harness" / "x2_r2_agent.py", side / "x2_r2_agent.py")
    saved_path, saved_mod = list(sys.path), sys.modules.get("x2_r2_agent")
    try:
        out = sidecar.import_check(JOB_ARGV, here=side)
        assert Path(out["agent_file"]).parent == side.resolve()
        # the agent re-inserts its own directory at the front on import; the side dir stays ahead of the parent
        assert sys.path[0] == str(side) and sys.path.index(str(side)) < sys.path.index(str(side.parent))
        assert out["neg_arm"] == "ollama_ctx_32768_negative_control"
        assert list(out["plan"][0]) == [ARM, list(SEEDS), 10, 32768]
        assert all(Path(p).parent == (REPO / "harness").resolve() for p in out["modules"].values())
    finally:
        sys.path[:] = saved_path
        if saved_mod is not None:
            sys.modules["x2_r2_agent"] = saved_mod


# ── register row: the pre-registered verdict ─────────────────────────────────────────────────────────

ANSWER = '{"answer": "CANARY-X HTAG-Y", "source": "s"}'
TOOLCALL = '{"name": "lookup_fact", "parameters": {"key":"REC-0005"}}'


def synth(misses=(), seeds=SEEDS, run_end=True, turns=10, num_ctx=32768):
    """misses: {(seed, turn): (sys_ok, hist_ok, final_text)} on canary turns 5 and 10."""
    misses = dict(misses)
    rows = [{"record": "run_start"}]
    for seed in seeds:
        att = f"a{seed}"
        sm = hm = 0
        for t in range(1, turns + 1):
            chk = t % 5 == 0
            sys_ok, hist_ok, text = misses.get((seed, t), (True, True, ANSWER)) if chk else (None, None, ANSWER)
            sm += chk and not sys_ok
            hm += chk and not hist_ok
            rows.append({"record": "r2a_turn", "attempt_id": att, "seed": seed, "turn_idx": t, "arm_id": ARM,
                         "model_id": "llama3.1:8b", "mode": "validation", "canary_check": chk, "canary_sys_ok": sys_ok,
                         "canary_hist_ok": hist_ok, "final_text": text, "num_ctx_requested": num_ctx,
                         "over_loaded_window": False})
        rows.append({"record": "r2a_session", "attempt_id": att, "seed": seed, "arm_id": ARM, "model_id": "llama3.1:8b",
                     "mode": "validation", "canary_sys_misses": sm, "canary_hist_misses": hm,
                     "canary_checks": turns // 5})
    if run_end:
        rows.append({"record": "run_end"})
    return rows


def verdict(rows):
    return nr.x2_r2_llama_val_32k_verdict(rows, ag)


def test_zero_misses_is_backend_or_offload_effect():
    v = verdict(synth())
    assert v["verdict"] == "backend or offload effect" and v["valid"]
    assert v["per_session"] == {s: 0 for s in SEEDS} and v["n_canaries"] == 20


def test_turn5_text_tool_calls_are_a_window_effect():
    v = verdict(synth({(20260903, 5): (False, False, TOOLCALL), (20260905, 5): (False, False, "  " + TOOLCALL)}))
    assert v["verdict"] == "window effect"
    assert v["per_session"][20260903] == 2 and v["n_missed_canaries"] == 4
    assert [(m["seed"], m["turn"], m["tool_name"]) for m in v["misses"]] == [(20260903, 5, "lookup_fact"),
                                                                             (20260905, 5, "lookup_fact")]


@pytest.mark.parametrize("miss", [
    {(20260903, 10): (False, False, TOOLCALL)},                       # text tool call, but not on turn 5
    {(20260903, 5): (False, True, ANSWER)},                           # a real canary miss in an answer
    {(20260903, 5): (False, False, TOOLCALL), (20260904, 10): (True, False, ANSWER)},   # mixed
])
def test_other_patterns_are_inconclusive_with_the_pattern(miss):
    v = verdict(synth(miss))
    assert v["verdict"] == "inconclusive" and v["valid"]
    for (seed, turn) in miss:
        assert f"seed {seed} turn {turn}" in v["pattern"]


@pytest.mark.parametrize("kw,needle", [({"seeds": SEEDS[:4]}, "expected"), ({"run_end": False}, "no run_end"),
                                       ({"turns": 9}, "9 turns"), ({"num_ctx": 131072}, "num_ctx_requested")])
def test_incomplete_or_wrong_data_is_inconclusive_incomplete(kw, needle):
    v = verdict(synth(**kw))
    assert v["verdict"].startswith("inconclusive (incomplete:") and needle in v["verdict"] and not v["valid"]


def test_text_tool_call_pattern():
    assert nr.text_tool_call_name(TOOLCALL) == "lookup_fact"
    assert nr.text_tool_call_name('\n {"name": 3}') == ""
    assert nr.text_tool_call_name(ANSWER) is None
    assert nr.text_tool_call_name('text {"name": "x"}') is None
    assert nr.text_tool_call_name(None) is None


def test_register_row_is_pending_until_synced_and_computes_on_real_format_rows(tmp_path):
    entry = next(e for e in nr.NUMBER_ENTRIES if e["claim_id"] == "x2-r2-llama-val-32k")
    assert entry["pending_ok"] and nr._pending(entry, repo=tmp_path) == [nr.X2_R2_LLAMA_VAL_32K]
    blade = REPO / "results" / "blade_r2_validation_v2.jsonl"
    if not blade.exists():
        pytest.skip("Blade run 2 file not present")
    # Blade run 2 has the same arm name, seeds and turn plan: its rows exercise the full compute path.
    (tmp_path / "results").mkdir()
    shutil.copy(blade, tmp_path / nr.X2_R2_LLAMA_VAL_32K)
    r = nr.compute_x2_r2_llama_val_32k(tmp_path)
    assert r["detail"]["verdict"]["verdict"] == "window effect"
    assert "20260903: 2" in r["value"] and "20260905: 2" in r["value"] and "verdict: window effect" in r["value"]
    json.dumps(r["value"])
