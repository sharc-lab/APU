"""Tests for analysis/results_digest.py. Read-only end to end: no test here starts, stops, or kills a process."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import results_digest as rd  # noqa: E402


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------- collect_phase_summaries
def test_collect_groups_by_section_and_counts_cells(tmp_path):
    f = tmp_path / "run.jsonl"
    _write_jsonl(f, [
        {"section": "R1speed", "kind": "call", "ttft_s": 1.5, "ts_utc": "2026-09-29T01:00:00Z"},
        {"section": "R1speed", "kind": "call", "ttft_s": 2.5, "ts_utc": "2026-09-29T02:00:00Z"},
        {"section": "P70", "kind": "call", "ttft_s": 100.0, "ts_utc": "2026-09-29T03:00:00Z"},
    ])
    phases = rd.collect_phase_summaries([f])
    assert phases["R1speed"]["n"] == 2
    assert phases["P70"]["n"] == 1
    assert phases["R1speed"]["last_ts_utc"] == "2026-09-29T02:00:00Z"


def test_collect_falls_back_to_record_when_no_section(tmp_path):
    f = tmp_path / "run.jsonl"
    _write_jsonl(f, [{"record": "bisect_result", "ts_utc": "2026-09-29T01:00:00Z"}])
    phases = rd.collect_phase_summaries([f])
    assert "bisect_result" in phases


def test_collect_unclassified_when_neither_present(tmp_path):
    f = tmp_path / "run.jsonl"
    _write_jsonl(f, [{"foo": "bar"}])
    phases = rd.collect_phase_summaries([f])
    assert "unclassified" in phases


def test_collect_summarizes_numeric_metrics(tmp_path):
    f = tmp_path / "run.jsonl"
    _write_jsonl(f, [
        {"section": "R1speed", "ttft_s": 1.0}, {"section": "R1speed", "ttft_s": 2.0},
        {"section": "R1speed", "ttft_s": 3.0},
    ])
    phases = rd.collect_phase_summaries([f])
    m = phases["R1speed"]["metrics"]["ttft_s"]
    assert m == {"n": 3, "median": 2.0, "min": 1.0, "max": 3.0}


def test_collect_ignores_non_numeric_and_bool_metric_values(tmp_path):
    f = tmp_path / "run.jsonl"
    _write_jsonl(f, [{"section": "X", "ttft_s": None}, {"section": "X", "score": True}])
    phases = rd.collect_phase_summaries([f])
    assert "ttft_s" not in phases["X"]["metrics"]
    assert "score" not in phases["X"]["metrics"]  # bool score must not be counted as a numeric metric


def test_collect_counts_contamination_tag_fields(tmp_path):
    f = tmp_path / "run.jsonl"
    _write_jsonl(f, [
        {"section": "C1", "contaminated_watchdog_overlap": True},
        {"section": "C1", "contaminated_watchdog_overlap": True},
        {"section": "C1"},
    ])
    phases = rd.collect_phase_summaries([f])
    assert phases["C1"]["contamination_tags"]["contaminated_watchdog_overlap"] == 2


def test_collect_counts_contamination_tag_value_field(tmp_path):
    f = tmp_path / "run.jsonl"
    _write_jsonl(f, [{"section": "C1", "tag": "contaminated_ollama_resident"}])
    phases = rd.collect_phase_summaries([f])
    assert phases["C1"]["contamination_tags"]["contaminated_ollama_resident"] == 1


def test_collect_tracks_outcomes(tmp_path):
    f = tmp_path / "run.jsonl"
    _write_jsonl(f, [{"section": "A70", "started": True}, {"section": "A70", "started": False}])
    phases = rd.collect_phase_summaries([f])
    assert phases["A70"]["outcomes"]["started=True"] == 1
    assert phases["A70"]["outcomes"]["started=False"] == 1


def test_collect_skips_blank_and_malformed_lines(tmp_path):
    f = tmp_path / "run.jsonl"
    f.write_text('{"section": "X"}\n\nnot json\n{"section": "X"}\n', encoding="utf-8")
    phases = rd.collect_phase_summaries([f])
    assert phases["X"]["n"] == 2


def test_collect_missing_file_does_not_crash(tmp_path):
    phases = rd.collect_phase_summaries([tmp_path / "does_not_exist.jsonl"])
    assert phases == {}


def test_collect_on_real_committed_result_file():
    """Per the standing instruction to unit-test against the existing result files, not just synthetic ones."""
    real = Path(__file__).resolve().parents[1] / "results" / "t2s_amech_20260926T181456Z.jsonl"
    if not real.exists():
        import pytest
        pytest.skip("real result file not present in this checkout")
    phases = rd.collect_phase_summaries([real])
    assert phases  # at least one phase bucket, and it did not crash on real data
    assert sum(p["n"] for p in phases.values()) > 0


# ---------------------------------------------------------------------------------------------------- check_idle
def test_check_idle_none_when_nothing_pending_and_nothing_running():
    assert rd.check_idle([], Path("C:/apu/ovn")) is None


def test_check_idle_alerts_when_pending_and_nothing_running():
    items = [{"id": "a", "status": "pending"}]
    alert = rd.check_idle(items, Path("C:/apu/ovn"))
    assert alert is not None
    assert "pending" in alert


def test_check_idle_none_when_running_with_fresh_heartbeat(tmp_path):
    log = tmp_path / "queue_a.log"
    log.write_text("hi", encoding="utf-8")
    items = [{"id": "a", "status": "running"}]
    assert rd.check_idle(items, tmp_path, now=time.time()) is None


def test_check_idle_alerts_when_running_with_stale_heartbeat(tmp_path):
    import os
    log = tmp_path / "queue_a.log"
    log.write_text("hi", encoding="utf-8")
    old = time.time() - 3600
    os.utime(log, (old, old))
    items = [{"id": "a", "status": "running"}]
    alert = rd.check_idle(items, tmp_path, now=time.time())
    assert alert is not None
    assert "a" in alert


def test_check_idle_none_at_40min_for_a_70b_phase_but_stale_for_others(tmp_path):
    """Not a flat 15-minute threshold: a 70B-model phase gets the watchdog's own 60-minute stale threshold, since
    a single 70B call can legitimately take minutes each."""
    import os
    age_s = 40 * 60  # stale for a non-70B phase (30 min), fresh for a 70B phase (60 min)

    log70 = tmp_path / "queue_a70job.log"
    log70.write_text("hi", encoding="utf-8")
    old = time.time() - age_s
    os.utime(log70, (old, old))
    items70 = [{"id": "a70job", "status": "running", "cmd": ["python", "x.py", "--models", "llama-3.3-70b"]}]
    assert rd.check_idle(items70, tmp_path, now=time.time()) is None

    log8b = tmp_path / "queue_8bjob.log"
    log8b.write_text("hi", encoding="utf-8")
    os.utime(log8b, (old, old))
    items8b = [{"id": "8bjob", "status": "running", "cmd": ["python", "x.py", "--models", "qwen3-8b"]}]
    alert = rd.check_idle(items8b, tmp_path, now=time.time())
    assert alert is not None


def test_check_idle_alerts_when_running_with_no_log_file(tmp_path):
    items = [{"id": "a", "status": "running"}]
    alert = rd.check_idle(items, tmp_path, now=time.time())
    assert alert is not None
    assert "no log file found" in alert


# ---------------------------------------------------------------------------------------------------- render_markdown / read-only
def test_render_markdown_puts_alert_first():
    md = rd.render_markdown("evo-t2s", {}, idle_alert="job x stale")
    assert md.startswith("**ALERT: evo-t2s idle since check, reason: job x stale**")


def test_render_markdown_no_alert_line_when_none():
    md = rd.render_markdown("evo-t2s", {})
    assert "ALERT" not in md


def test_render_markdown_includes_progress_alert():
    md = rd.render_markdown("evo-x2", {}, progress_alert="job r1b_controls stale, 90 min")
    assert "ALERT: evo-x2 progress stale, reason: job r1b_controls stale, 90 min" in md


def test_render_markdown_shows_both_alerts_when_both_present():
    md = rd.render_markdown("evo-x2", {}, idle_alert="idle reason", progress_alert="progress reason")
    assert "idle since check" in md
    assert "progress stale" in md


# ------------------------------------------------------------------------------------------- check_progress_stale wiring
def test_main_surfaces_progress_alert_in_output(tmp_path, monkeypatch, capsys):
    """End to end through main(): a queue with a running job and a phase whose last row is over an hour old must
    produce a progress alert in both the written digest and the printed summary line."""
    import json as _json
    from datetime import datetime, timedelta, timezone
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stale_ts = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
    (results_dir / "run.jsonl").write_text(
        _json.dumps({"section": "R1b", "ts_utc": stale_ts, "model_id": "qwen3-8b"}) + "\n", encoding="utf-8")
    queue_file = tmp_path / "queue_state.json"
    queue_file.write_text(_json.dumps([{"id": "r1b_controls", "status": "running", "cmd": ["...", "qwen3-8b"]}]),
                          encoding="utf-8")
    out_path = tmp_path / "DIGEST.md"
    monkeypatch.setattr(sys, "argv", ["results_digest.py", "--results-dir", str(results_dir),
                                      "--queue-file", str(queue_file), "--out", str(out_path),
                                      "--machine", "evo-x2"])
    rd.main()
    printed = capsys.readouterr().out
    assert "r1b_controls" in printed
    md = out_path.read_text(encoding="utf-8")
    assert "progress stale" in md
    assert "r1b_controls" in md


def test_render_markdown_includes_phase_table():
    phases = {"R1speed": {"n": 3, "last_ts_utc": "2026-09-29T01:00:00Z", "metrics": {"ttft_s": {"n": 3, "median": 2.0, "min": 1.0, "max": 3.0}},
                          "contamination_tags": {}, "outcomes": {}, "files": ["x.jsonl"]}}
    md = rd.render_markdown("evo-t2s", phases)
    assert "## R1speed" in md
    assert "cells: 3" in md
    assert "ttft_s" in md


def test_module_never_imports_anything_that_starts_stops_or_kills_processes():
    """Static guard: this module must not import subprocess or t2s_lab (which owns Server.start/stop) -- it is
    meant to be safe to run at any time without touching a live measurement."""
    import inspect
    src = inspect.getsource(rd)
    assert "import subprocess" not in src
    assert "import t2s_lab" not in src
