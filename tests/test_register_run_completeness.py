"""The register must not mark a row VERIFIED from a partial run file (analysis/numbers_register.py::_incomplete and
run_file_incomplete): single files need a run_end after the last run_start (or a refused-to-start preflight),
multi-part runs are judged by their last part, and the opt-out list is explicit."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis.numbers_register as nr  # noqa: E402


def _w(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _entry(*files):
    return {"claim_id": "x", "data_files": list(files)}


def test_single_file_needs_run_end(tmp_path):
    _w(tmp_path / "results/a.jsonl", [{"record": "run_start"}, {"record": "r2a_turn"}])
    assert nr._incomplete(_entry("results/a.jsonl"), tmp_path) == ["partial: no run_end in results/a.jsonl"]
    _w(tmp_path / "results/a.jsonl", [{"record": "run_start"}, {"record": "r2a_turn"}, {"record": "run_end"}])
    assert nr._incomplete(_entry("results/a.jsonl"), tmp_path) == []


def test_run_end_before_a_later_run_start_is_partial(tmp_path):
    _w(tmp_path / "results/a.jsonl", [{"record": "run_start"}, {"record": "run_end"}, {"record": "run_start"}])
    assert nr._incomplete(_entry("results/a.jsonl"), tmp_path)


def test_files_without_run_start_and_non_jsonl_are_not_runs(tmp_path):
    _w(tmp_path / "results/b.jsonl", [{"record": "x"}, {"record": "run_end"}])
    _w(tmp_path / "results/c.jsonl", [{"record": "x"}])
    (tmp_path / "results/d.json").write_text('{"record": "run_start"}', encoding="utf-8")
    assert nr._incomplete(_entry("results/b.jsonl", "results/c.jsonl", "results/d.json", "results/none.jsonl"),
                          tmp_path) == []


def test_refused_to_start_counts_as_complete(tmp_path):
    _w(tmp_path / "results/r.jsonl", [{"record": "refused_model"},
                                      {"record": "run_start", "models": [], "validation_preflight": {"ok": False}}])
    assert nr._incomplete(_entry("results/r.jsonl"), tmp_path) == []
    _w(tmp_path / "results/r.jsonl", [{"record": "run_start", "models": ["m"], "validation_preflight": {"ok": True}}])
    assert nr._incomplete(_entry("results/r.jsonl"), tmp_path)


def test_globs_are_expanded(tmp_path):
    _w(tmp_path / "results/g_1.jsonl", [{"record": "run_start"}, {"record": "run_end"}])
    _w(tmp_path / "results/g_2.jsonl", [{"record": "run_start"}])
    assert nr._incomplete(_entry("results/g_*.jsonl"), tmp_path) == ["partial: no run_end in results/g_2.jsonl"]


def test_multi_part_judged_by_last_part(tmp_path):
    _w(tmp_path / "results/m.jsonl", [{"record": "run_start"}, {"record": "run_end", "stopped": "crash"}])
    _w(tmp_path / "results/m_part2.jsonl", [{"record": "run_start"}])
    _w(tmp_path / "results/m_part3.jsonl", [{"record": "run_start"}, {"record": "run_end", "stopped": None}])
    for f in ("results/m.jsonl", "results/m_part2.jsonl", "results/m_part3.jsonl", "results/m*.jsonl"):
        assert nr._incomplete(_entry(f), tmp_path) == [], f
    _w(tmp_path / "results/m_part3.jsonl", [{"record": "run_start"}])
    assert nr._incomplete(_entry("results/m.jsonl"), tmp_path) == [
        "partial: no run_end in the last part results/m_part3.jsonl"]
    _w(tmp_path / "results/m_part3.jsonl", [{"record": "run_start"}, {"record": "run_end", "stopped": "cap"}])
    assert "stopped" in nr._incomplete(_entry("results/m_part2.jsonl"), tmp_path)[0]


def test_opt_out_list_is_explicit_and_documented(tmp_path, monkeypatch):
    for pat, why in nr.RUN_END_OPT_OUT.items():
        assert pat.startswith("results/") and len(why) > 20
    _w(tmp_path / "results/blade_dryrun/x.jsonl", [{"record": "run_start"}, {"record": "run_stopped"}])
    assert nr._incomplete(_entry("results/blade_dryrun/x.jsonl"), tmp_path) == []


def test_build_register_marks_partial_rows_pending(tmp_path, monkeypatch):
    _w(tmp_path / "results/p.jsonl", [{"record": "run_start"}])
    called = []
    entry = {"claim_id": "p-row", "description": "", "data_files": ["results/p.jsonl"],
             "compute": lambda repo: called.append(1) or {"value": "1", "n": 1}, "script_function": "x::y"}
    monkeypatch.setattr(nr, "REPO", tmp_path)
    monkeypatch.setattr(nr, "NUMBER_ENTRIES", [entry])
    monkeypatch.setattr(nr, "_git_head", lambda: "h")
    rows = nr.build_register()
    assert rows[0]["status"] == "PENDING" and "partial: no run_end" in rows[0]["value"] and not called
    _w(tmp_path / "results/p.jsonl", [{"record": "run_start"}, {"record": "run_end"}])
    rows = nr.build_register()
    assert rows[0]["status"] == "VERIFIED" and called


def test_real_cloud_redo_parts_complete_and_middle_part_not_flagged():
    repo = nr.REPO
    if not (repo / "results/cloud_redo_full_context_v1_part3.jsonl").exists():
        return
    # part 2 has no run_end of its own and part 1 stopped; part 3 resumed both and ended unstopped with every task
    assert nr.run_file_incomplete(repo, "results/cloud_redo_full_context_v1_part2.jsonl") is None
    assert nr.run_file_incomplete(repo, "results/cloud_redo_full_context_v1.jsonl") is None
