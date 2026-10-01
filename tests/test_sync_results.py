"""Unit tests for scripts/sync_results.py's pure logic (plan_pulls, local_dest_path, manifest round-trip).
No real SSH; the enumerate/pull functions that touch the network are exercised only via dependency injection
in the one integration-shaped test below, with a fake SSH/SCP layer."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sync_results as sr  # noqa: E402


def test_enumerate_remote_files_skips_malformed_lines(monkeypatch):
    """Found live 2026-10-01: one of 977 real enumerated lines parsed to a non-dict JSON value (a malformed
    one-line-per-file JSON fragment, likely from an unusual filename), crashing plan_pulls with
    'string indices must be integers'. enumerate_remote_files must skip anything that isn't a well-formed
    file-record dict instead of propagating a bad line into the rest of the pipeline."""
    lines = [
        '{"root":"C:\\\\apu\\\\ovn\\\\results","rel":"a.jsonl","size":10,"sha256":"aaa"}',
        '"just a stray string"',
        'not even valid json {{{',
        '{"root":"C:\\\\apu\\\\ovn\\\\results","rel":"b.jsonl","size":20,"sha256":"bbb"}',
    ]
    monkeypatch.setattr(sr, "_ssh_lines", lambda host_str, script, timeout=600: lines)
    out = sr.enumerate_remote_files("user@host")
    assert len(out) == 2
    assert {f["rel"] for f in out} == {"a.jsonl", "b.jsonl"}


def test_plan_pulls_new_file_goes_to_to_pull():
    remote = [{"root": r"C:\apu\ovn\results", "rel": "a.jsonl", "size": 100, "sha256": "aaa"}]
    to_pull, too_large, unchanged = sr.plan_pulls("evo-x2", remote, manifest={})
    assert to_pull == remote
    assert too_large == []
    assert unchanged == []


def test_plan_pulls_unchanged_file_is_skipped():
    remote = [{"root": r"C:\apu\ovn\results", "rel": "a.jsonl", "size": 100, "sha256": "aaa"}]
    manifest = {"evo-x2::C:\\apu\\ovn\\results::a.jsonl": "aaa"}
    to_pull, too_large, unchanged = sr.plan_pulls("evo-x2", remote, manifest)
    assert to_pull == []
    assert unchanged == remote


def test_plan_pulls_changed_hash_goes_to_to_pull_again():
    remote = [{"root": r"C:\apu\ovn\results", "rel": "a.jsonl", "size": 100, "sha256": "bbb"}]
    manifest = {"evo-x2::C:\\apu\\ovn\\results::a.jsonl": "aaa"}
    to_pull, too_large, unchanged = sr.plan_pulls("evo-x2", remote, manifest)
    assert to_pull == remote


def test_plan_pulls_oversized_file_is_flagged_not_pulled():
    remote = [{"root": r"C:\apu\ovn\results", "rel": "huge.jsonl", "size": 60 * 1024 * 1024, "sha256": "ccc"}]
    to_pull, too_large, unchanged = sr.plan_pulls("evo-x2", remote, manifest={})
    assert to_pull == []
    assert too_large == remote


def test_local_dest_path_default_root_is_flat_under_results():
    f = {"root": r"C:\apu\ovn\results", "rel": "t2s_night2_x.jsonl", "size": 1, "sha256": "x"}
    dest = sr.local_dest_path(f)
    assert dest == sr.RESULTS_DIR / "t2s_night2_x.jsonl"


def test_local_dest_path_alt_root_gets_prefixed_to_avoid_collision():
    f = {"root": r"C:\apu\results", "rel": "t2s_k1_ollama_x.jsonl", "size": 1, "sha256": "x"}
    dest = sr.local_dest_path(f)
    assert dest.name == "apu_results__t2s_k1_ollama_x.jsonl"


def test_manifest_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "MANIFEST_PATH", tmp_path / "manifest.json")
    m = {"evo-x2::C:\\apu\\ovn\\results::a.jsonl": "aaa"}
    sr.save_manifest(m)
    loaded = sr.load_manifest()
    assert loaded == m


def test_load_manifest_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "MANIFEST_PATH", tmp_path / "does_not_exist.json")
    assert sr.load_manifest() == {}


def test_load_manifest_corrupt_file_returns_empty_not_a_crash(tmp_path, monkeypatch):
    p = tmp_path / "manifest.json"
    p.write_text("not valid json {{{", encoding="utf-8")
    monkeypatch.setattr(sr, "MANIFEST_PATH", p)
    assert sr.load_manifest() == {}


def test_pull_files_updates_manifest_only_for_successful_pulls(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "REPO", tmp_path)
    monkeypatch.setattr(sr, "RESULTS_DIR", tmp_path)
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    to_pull = [{"root": r"C:\apu\ovn\results", "rel": "a.jsonl", "size": 10, "sha256": "aaa"}]
    manifest = {}
    pulled = sr.pull_files("user@host", "evo-x2", to_pull, manifest)
    assert len(pulled) == 1
    assert manifest["evo-x2::C:\\apu\\ovn\\results::a.jsonl"] == "aaa"


def test_pull_files_does_not_update_manifest_on_scp_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "RESULTS_DIR", tmp_path)

    def fake_run(cmd, capture_output, text, timeout):
        class R:
            returncode = 1
            stderr = "connection refused"
        return R()

    monkeypatch.setattr(sr.subprocess, "run", fake_run)
    to_pull = [{"root": r"C:\apu\ovn\results", "rel": "a.jsonl", "size": 10, "sha256": "aaa"}]
    manifest = {}
    pulled = sr.pull_files("user@host", "evo-x2", to_pull, manifest)
    assert pulled == []
    assert manifest == {}
