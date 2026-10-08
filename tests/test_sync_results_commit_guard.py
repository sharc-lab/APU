"""Tests for scripts/sync_results.py's commit guard (2026-10-08): the APU-SyncResults scheduled task committed
in the middle of a merge in the primary checkout. Real tmp git repos, fake SSH/SCP (enumerate_remote_files and
pull_files are replaced; nothing touches the network). Covers: skip on merge / rebase / cherry-pick / revert in
progress, lock files and unmerged paths, with the reason logged and nothing staged; the fixed commit message
"results sync"; only the pulled files plus the manifest staged and committed."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sync_results as sr  # noqa: E402

PULLED = ["results/a.jsonl", "results/sub/b.jsonl"]
MANIFEST_REL = "results/.sync_manifest.json"


def _git(repo, *a, check=True):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=check)


def _init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@t.com")
    _git(path, "config", "user.name", "t")
    _git(path, "config", "commit.gpgsign", "false")
    (path / "results").mkdir()
    (path / "results" / ".gitkeep").write_text("", encoding="utf-8")
    (path / MANIFEST_REL).write_text("{}", encoding="utf-8")
    (path / "README.txt").write_text("base\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")
    return path


def _point_sr_at(monkeypatch, repo):
    monkeypatch.setattr(sr, "REPO", repo)
    monkeypatch.setattr(sr, "RESULTS_DIR", repo / "results")
    monkeypatch.setattr(sr, "MANIFEST_PATH", repo / MANIFEST_REL)
    monkeypatch.setattr(sr.hc, "ALIASES", {"evo-x2": "EVO-X2"})
    monkeypatch.setattr(sr.hc, "HOSTS", {"EVO-X2": {"ssh_host": "user@host"}})
    monkeypatch.setattr(sr, "enumerate_remote_files", lambda host_str: [
        {"root": r"C:\apu\ovn\results", "rel": "a.jsonl", "size": 10, "sha256": "aaa"},
        {"root": r"C:\apu\ovn\results", "rel": "sub\\b.jsonl", "size": 10, "sha256": "bbb"}])

    def fake_pull(host_str, key, to_pull, manifest):
        out = []
        for f in to_pull:
            dest = repo / "results" / f["rel"].replace("\\", "/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps({"sha": f["sha256"]}) + "\n", encoding="utf-8")
            manifest[f"{key}::{f['root']}::{f['rel']}"] = f["sha256"]
            out.append(dest.relative_to(repo).as_posix())
        return out

    monkeypatch.setattr(sr, "pull_files", fake_pull)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = _init_repo(tmp_path / "repo")
    _point_sr_at(monkeypatch, r)
    return r


def _git_dir(repo):
    rel = _git(repo, "rev-parse", "--path-format=relative", "--git-dir").stdout.strip()
    return (Path(repo) / rel).resolve()


def _assert_skipped(repo, result, capsys, reason_fragment):
    head_before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert result["commit"] is None
    assert reason_fragment in result["commit_skipped"]
    out = capsys.readouterr().out
    assert "SKIPPED commit" in out and reason_fragment in out
    assert _git(repo, "log", "-1", "--format=%s").stdout.strip() == "init"
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head_before
    # pulled files are left on disk, but nothing was staged
    for p in PULLED:
        assert (repo / p).exists()
    assert _git(repo, "diff", "--cached", "--name-only").stdout.strip() == ""
    # manifest not advanced, so the files are retried next cycle
    assert json.loads((repo / MANIFEST_REL).read_text(encoding="utf-8")) == {}


def _make_merge_head(repo):
    (_git_dir(repo) / "MERGE_HEAD").write_text(_git(repo, "rev-parse", "HEAD").stdout, encoding="utf-8")


@pytest.mark.parametrize("make, fragment", [
    (_make_merge_head, "merge in progress"),
    (lambda r: (_git_dir(r) / "CHERRY_PICK_HEAD").write_text("x\n"), "cherry-pick in progress"),
    (lambda r: (_git_dir(r) / "REVERT_HEAD").write_text("x\n"), "revert in progress"),
    (lambda r: (_git_dir(r) / "rebase-merge").mkdir(), "rebase in progress"),
    (lambda r: (_git_dir(r) / "rebase-apply").mkdir(), "rebase in progress"),
    (lambda r: (_git_dir(r) / "index.lock").write_text(""), "index.lock"),
    (lambda r: (_git_dir(r) / "HEAD.lock").write_text(""), "HEAD.lock"),
    (lambda r: (_git_dir(r) / "refs" / "heads" / "main.lock").write_text(""), "main.lock"),
])
def test_sync_skips_commit_when_repo_is_busy(repo, capsys, make, fragment):
    make(repo)
    result = sr.sync_host("evo-x2", do_commit=True)
    _assert_skipped(repo, result, capsys, fragment)


def test_sync_skips_commit_with_unmerged_paths(repo, capsys):
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "README.txt").write_text("side\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "side")
    _git(repo, "checkout", "-q", "main")
    (repo / "README.txt").write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "main change")
    assert _git(repo, "merge", "side", check=False).returncode != 0  # real conflict
    (_git_dir(repo) / "MERGE_HEAD").unlink()  # isolate the unmerged-paths check from the MERGE_HEAD one
    assert "unmerged path" in sr.repo_busy_reason()
    head_before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    result = sr.sync_host("evo-x2", do_commit=True)
    assert "unmerged path" in result["commit_skipped"]
    assert "SKIPPED commit" in capsys.readouterr().out
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head_before
    for p in PULLED:
        assert (repo / p).exists()


def test_linked_worktree_git_dir_is_resolved_via_git(tmp_path, monkeypatch, capsys):
    """A linked worktree's .git is a file, not a directory: MERGE_HEAD lives under <common>/worktrees/<name>."""
    main = _init_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", "-b", "wtb", str(wt))
    assert (wt / ".git").is_file()
    _point_sr_at(monkeypatch, wt)
    assert sr.repo_busy_reason() is None
    _make_merge_head(wt)
    assert "worktrees" in str(_git_dir(wt))
    result = sr.sync_host("evo-x2", do_commit=True)
    _assert_skipped(wt, result, capsys, "merge in progress")


def test_clean_repo_commits_only_pulled_files_and_manifest_with_fixed_message(repo, capsys):
    # unrelated human work: one file staged, one modified-unstaged, one untracked; none may enter the commit
    (repo / "staged.txt").write_text("human\n", encoding="utf-8")
    _git(repo, "add", "staged.txt")
    (repo / "README.txt").write_text("edited\n", encoding="utf-8")
    (repo / "results" / "untracked_other.jsonl").write_text("{}\n", encoding="utf-8")

    result = sr.sync_host("evo-x2", do_commit=True)
    assert result["commit"]
    assert "commit_skipped" not in result
    msg = _git(repo, "log", "-1", "--format=%B").stdout
    assert msg.strip() == "results sync"
    assert msg.rstrip("\n") == "results sync"
    files = set(_git(repo, "show", "--name-only", "--format=", "HEAD").stdout.split())
    assert files == set(PULLED) | {MANIFEST_REL}
    manifest = json.loads((repo / MANIFEST_REL).read_text(encoding="utf-8"))
    assert set(manifest.values()) == {"aaa", "bbb"}
    # the human's staged file is still staged, the rest untouched
    assert _git(repo, "diff", "--cached", "--name-only").stdout.split() == ["staged.txt"]
    status = _git(repo, "status", "--porcelain").stdout
    assert " M README.txt" in status and "?? results/untracked_other.jsonl" in status


def test_installer_sync_host_param_defaults_to_evo_x2():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "install_sync_results_task.ps1").read_text(
        encoding="utf-8")
    assert "[ValidateSet('evo-x2', 'evo-t2s', 'both')]" in src
    assert "[string]$SyncHost = 'evo-x2'" in src
    assert "--host $SyncHost" in src and "--host both" not in src


def test_never_uses_git_add_all():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "sync_results.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert '"add", "-A"' not in code and '"add", "."' not in code and '"--all"' not in code
