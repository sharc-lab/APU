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


def test_git_add_in_batches_splits_a_large_file_list(tmp_path, monkeypatch):
    """Found live 2026-10-01: git add with 979 individual path args hit Windows' command-line length limit
    (WinError 206, 'The filename or extension is too long'). Must split into batches."""
    calls = []
    monkeypatch.setattr(sr, "git", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(sr, "GIT_ADD_BATCH_SIZE", 3)
    paths = [f"file{i}.jsonl" for i in range(7)]
    sr._git_add_in_batches(paths)
    assert len(calls) == 3  # 3 + 3 + 1
    assert calls[0] == (("add", "file0.jsonl", "file1.jsonl", "file2.jsonl"), {"check": False})
    assert calls[2] == (("add", "file6.jsonl"), {"check": False})


def test_git_add_in_batches_tolerates_a_gitignored_path_in_the_batch(tmp_path, monkeypatch):
    """2026-10-02, found live: a real pulled batch (7 files) included 3 paths matched by .gitignore
    (*.log sidecars). Plain `git add` exits non-zero for the WHOLE batch when any path is ignored,
    which commit_pulled's check=True then raised as a commit failure for every file in that batch,
    not just the ignored ones. Real git repo, real .gitignore, real add -- not mocked, since the bug
    is specifically in git's own exit-code behavior for this case."""
    import subprocess
    repo = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (repo / "real.jsonl").write_text("{}", encoding="utf-8")
    (repo / "ignored.log").write_text("log text", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    monkeypatch.setattr(sr, "REPO", repo)
    monkeypatch.setattr(sr, "GIT_ADD_BATCH_SIZE", 10)
    sr._git_add_in_batches(["real.jsonl", "ignored.log"])  # must not raise
    status = subprocess.run(["git", "status", "--porcelain", "real.jsonl"], cwd=repo,
                            capture_output=True, text=True, check=True).stdout
    assert status.strip().startswith("A")  # real.jsonl really got staged despite ignored.log in the batch


def test_commit_pulled_batches_both_add_and_status(monkeypatch):
    monkeypatch.setattr(sr, "GIT_ADD_BATCH_SIZE", 2)
    monkeypatch.setattr(sr, "repo_busy_reason", lambda: None)
    add_calls = []
    status_calls = []

    def fake_git(*a, **kw):
        class R:
            stdout = ""
        if a[0] == "add":
            add_calls.append(a)
        elif a[0] == "status":
            status_calls.append(a)
            r = R()
            r.stdout = " M file0.jsonl\0"
            return r
        elif a[0] == "commit":
            return R()
        elif a[0] == "rev-parse":
            r = R()
            r.stdout = "abc123\n"
            return r
        return R()

    monkeypatch.setattr(sr, "git", fake_git)
    paths = [f"file{i}.jsonl" for i in range(5)]
    sha = sr.commit_pulled(paths)
    assert len(add_calls) == 3  # ceil(6/2): 5 pulled files + the manifest
    assert len(status_calls) == 3
    assert sha == "abc123"


def test_sync_host_does_not_update_manifest_when_commit_fails(tmp_path, monkeypatch):
    """The manifest must only be saved after a successful commit -- a commit failure must leave every
    pulled file eligible for retry on the next run, not silently marked done."""
    monkeypatch.setattr(sr, "REPO", tmp_path)
    monkeypatch.setattr(sr, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sr, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(hc_module := sr.hc, "ALIASES", {"evo-x2": "EVO-X2"})
    monkeypatch.setattr(sr.hc, "HOSTS", {"EVO-X2": {"ssh_host": "user@host"}})
    monkeypatch.setattr(sr, "enumerate_remote_files", lambda host_str: [
        {"root": r"C:\apu\ovn\results", "rel": "a.jsonl", "size": 10, "sha256": "aaa"}])
    monkeypatch.setattr(sr, "pull_files", lambda host_str, key, to_pull, manifest: (
        manifest.update({"evo-x2::C:\\apu\\ovn\\results::a.jsonl": "aaa"}), ["a.jsonl"])[1])

    def failing_commit(pulled_paths):
        raise RuntimeError("git add failed: command line too long")

    monkeypatch.setattr(sr, "commit_pulled", failing_commit)
    result = sr.sync_host("evo-x2", do_commit=True)
    assert result.get("commit_failed") is True
    assert sr.load_manifest() == {}  # never saved -- a.jsonl stays eligible for retry


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

    def fake_run(cmd, capture_output, text, timeout, **kwargs):
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


def test_pull_files_survives_a_timeout_and_continues_to_the_next_file(tmp_path, monkeypatch):
    """Found live 2026-10-01: subprocess.run(..., timeout=...) raises TimeoutExpired rather than returning a
    non-zero-returncode result; an uncaught timeout on file N of many crashed the whole sync before any later
    file was attempted. pull_files must catch it and keep going."""
    monkeypatch.setattr(sr, "REPO", tmp_path)
    monkeypatch.setattr(sr, "RESULTS_DIR", tmp_path)
    calls = {"n": 0}

    def flaky_run(cmd, capture_output, text, timeout, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sr.subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr(sr.subprocess, "run", flaky_run)
    to_pull = [
        {"root": r"C:\apu\ovn\results", "rel": "a.jsonl", "size": 10, "sha256": "aaa"},
        {"root": r"C:\apu\ovn\results", "rel": "b.jsonl", "size": 10, "sha256": "bbb"},
    ]
    manifest = {}
    pulled = sr.pull_files("user@host", "evo-x2", to_pull, manifest)
    assert pulled == ["b.jsonl"]  # a.jsonl timed out and was skipped; b.jsonl still got pulled
    assert "evo-x2::C:\\apu\\ovn\\results::a.jsonl" not in manifest
    assert manifest["evo-x2::C:\\apu\\ovn\\results::b.jsonl"] == "bbb"


def test_pull_files_does_not_update_manifest_on_scp_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "RESULTS_DIR", tmp_path)

    def fake_run(cmd, capture_output, text, timeout, **kwargs):
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
