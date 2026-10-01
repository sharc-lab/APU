"""Pulls every results file from both evo-t2s and evo-x2 to the controller, checksummed and incremental, and
commits whatever changed. Built 2026-10-01 after a hand-typed number (qwen3-32b "41/82 (50%)") was reported in
a chat message with no backing file anywhere in this repo -- the real file for that claim (and several others
found the same day: the real PX2 result file, the real X2 K1 v3 tier file) existed only on a remote machine and
was never pulled or committed, so a subagent (or a human) working from the repo alone could not check it, and in
at least one case a number got reported from memory instead. This script exists so "the file is only on the
remote machine" stops being possible: every subagent and every report works from synced, committed files only.

Both result directories are synced on each host -- C:\\apu\\ovn\\results (the one most scripts' --out-dir
defaults to) AND C:\\apu\\results (K1's own DEPLOY.parent-based default, found live 2026-09-30/10-01, a real
inconsistency between scripts, not fixed here -- this script works around it by syncing both locations rather
than picking one).

Incremental: a local manifest (results/.sync_manifest.json) records the SHA-256 this script last pulled for
every (host, remote_path) pair; a file is re-pulled only if the remote's current hash differs (or it was never
pulled before). Files over 50 MB are flagged for Git LFS rather than committed directly -- see the printed
report for which, if any; this script does not configure LFS itself (not installed on this controller as of
2026-10-01), it only ever refuses to `git add` an oversized file and says so.

Usage: py -3.12 scripts/sync_results.py [--host evo-t2s|evo-x2|both] [--commit/--no-commit]
Intended to also run unattended every 2 hours via the digest task (queue_watchdog.py's run_digest, or a
separate scheduled task -- see this repo's install_queue_watchdog.ps1 for the pattern to copy).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
MANIFEST_PATH = RESULTS_DIR / ".sync_manifest.json"
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
SCP = r"C:\Windows\System32\OpenSSH\scp.exe"
LFS_SIZE_LIMIT_BYTES = 50 * 1024 * 1024

sys.path.insert(0, str(REPO / "harness"))
import host_config as hc  # noqa: E402

REMOTE_RESULT_DIRS = (r"C:\apu\ovn\results", r"C:\apu\results")

# One PowerShell command enumerates both remote result directories' files (recursively), each as one JSON
# object per line (ConvertTo-Json -Compress, then joined with newlines by the caller, not Format-List --
# a single giant ConvertTo-Json array over possibly thousands of files risks a truncated/invalid blob on a
# slow link; one line per file is robust to a partial transfer, and trivial to parse incrementally).
_ENUM_PS = r"""
$dirs = @('{dir1}', '{dir2}')
foreach ($d in $dirs) {{
  if (-not (Test-Path $d)) {{ continue }}
  Get-ChildItem -Path $d -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {{
    $h = (Get-FileHash -Path $_.FullName -Algorithm SHA256 -ErrorAction SilentlyContinue).Hash
    [PSCustomObject]@{{ root = $d; rel = $_.FullName.Substring($d.Length + 1); size = $_.Length; sha256 = $h }} | ConvertTo-Json -Compress
  }}
}}
"""


def _ssh_lines(host_str, ps_script, timeout=600):
    p = subprocess.run([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host_str,
                       "powershell", "-NoProfile", "-Command", ps_script],
                       capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0 and not p.stdout.strip():
        raise RuntimeError(f"ssh enumerate failed: rc={p.returncode} stderr={p.stderr[:500]}")
    return [l for l in p.stdout.splitlines() if l.strip()]


def enumerate_remote_files(host_str):
    """Returns a list of {root, rel, size, sha256} dicts for every file under either remote results dir."""
    script = _ENUM_PS.format(dir1=REMOTE_RESULT_DIRS[0], dir2=REMOTE_RESULT_DIRS[1])
    out = []
    skipped = 0
    for line in _ssh_lines(host_str, script):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(obj, dict) or not {"root", "rel", "size"} <= obj.keys():
            # A malformed/partial line (e.g. a path containing a character that broke the one-line-per-file
            # JSON framing) parses to something other than the expected object -- skip it rather than crash
            # the whole sync; the file just doesn't get synced this run and will be retried next time.
            skipped += 1
            continue
        out.append(obj)
    if skipped:
        print(f"  (skipped {skipped} unparseable/malformed enumeration line(s))", flush=True)
    return out


def load_manifest():
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_manifest(manifest):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")


def plan_pulls(host_key, remote_files, manifest):
    """Returns (to_pull, too_large, unchanged) -- to_pull is a list of remote file dicts whose sha256 differs
    from (or is absent from) the manifest; too_large is to_pull entries over LFS_SIZE_LIMIT_BYTES, split out
    so the caller can skip committing them while still reporting their existence."""
    to_pull, too_large, unchanged = [], [], []
    for f in remote_files:
        key = f"{host_key}::{f['root']}::{f['rel']}"
        prior = manifest.get(key)
        if prior == f["sha256"]:
            unchanged.append(f)
            continue
        if f["size"] > LFS_SIZE_LIMIT_BYTES:
            too_large.append(f)
        else:
            to_pull.append(f)
    return to_pull, too_large, unchanged


def local_dest_path(f):
    """Both remote roots land in the same local results/ tree, flattened by relative path -- a collision
    (same rel path under both roots) is resolved by prefixing the C:\\apu\\results-rooted ones (the K1
    default-out-dir location) with 'apu_results__' so neither ever silently overwrites the other."""
    rel = f["rel"].replace("\\", "/")
    if f["root"] == r"C:\apu\results":
        parts = rel.split("/")
        parts[-1] = "apu_results__" + parts[-1] if len(parts) == 1 else parts[-1]
        rel = "/".join(parts) if len(parts) > 1 else parts[0]
        if "/" not in rel and not rel.startswith("apu_results__"):
            rel = "apu_results__" + rel
    return RESULTS_DIR / rel


def pull_files(host_str, host_key, to_pull, manifest):
    pulled = []
    for f in to_pull:
        remote_path = f["root"] + "\\" + f["rel"]
        dest = local_dest_path(f)
        dest.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.run([SCP, "-q", "-o", "BatchMode=yes", f"{host_str}:{remote_path}", str(dest)],
                           capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            print(f"  FAILED to pull {remote_path}: {p.stderr[:300]}", file=sys.stderr)
            continue
        key = f"{host_key}::{f['root']}::{f['rel']}"
        manifest[key] = f["sha256"]
        pulled.append(str(dest.relative_to(REPO)))
    return pulled


def git(*a, check=True):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=check)


def commit_pulled(pulled_paths):
    if not pulled_paths:
        return None
    git("add", *pulled_paths)
    status = git("status", "--porcelain", "--", *pulled_paths).stdout.strip()
    if not status:
        return None  # nothing actually changed (re-pulled identical content)
    msg = f"sync_results: pull {len(pulled_paths)} result file(s) from the remote machines\n\n" + "\n".join(
        f"- {p}" for p in pulled_paths[:50])
    if len(pulled_paths) > 50:
        msg += f"\n- ... and {len(pulled_paths) - 50} more"
    git("commit", "-m", msg)
    return git("rev-parse", "HEAD").stdout.strip()


def sync_host(alias, do_commit=True):
    key = hc.ALIASES.get(alias.lower())
    if key is None:
        raise SystemExit(f"unknown host {alias!r}")
    host_str = hc.HOSTS[key]["ssh_host"]
    print(f"=== {key} ===", flush=True)
    remote_files = enumerate_remote_files(host_str)
    print(f"{key}: {len(remote_files)} remote files found across {REMOTE_RESULT_DIRS}", flush=True)
    manifest = load_manifest()
    to_pull, too_large, unchanged = plan_pulls(key, remote_files, manifest)
    print(f"{key}: {len(to_pull)} to pull, {len(too_large)} too large for direct commit, "
         f"{len(unchanged)} unchanged", flush=True)
    if too_large:
        print(f"{key}: files over {LFS_SIZE_LIMIT_BYTES / 1024 / 1024:.0f} MB, need Git LFS "
             f"(not installed on this controller as of 2026-10-01) -- NOT pulled:", flush=True)
        for f in too_large:
            print(f"    {f['root']}\\{f['rel']} ({f['size'] / 1024 / 1024:.1f} MB)", flush=True)
    pulled = pull_files(host_str, key, to_pull, manifest)
    save_manifest(manifest)
    print(f"{key}: pulled {len(pulled)} files", flush=True)
    commit_sha = commit_pulled(pulled) if do_commit else None
    if commit_sha:
        print(f"{key}: committed {commit_sha}", flush=True)
    elif pulled:
        print(f"{key}: pulled but not committed (do_commit=False or nothing actually changed)", flush=True)
    return {"host": key, "remote_files": len(remote_files), "pulled": pulled, "too_large": too_large,
            "unchanged": len(unchanged), "commit": commit_sha}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="both", choices=["evo-t2s", "evo-x2", "both"])
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args(argv)
    hosts = ["evo-t2s", "evo-x2"] if args.host == "both" else [args.host]
    results = [sync_host(h, do_commit=not args.no_commit) for h in hosts]
    print(json.dumps({"results": [{k: v for k, v in r.items() if k != "too_large"} for r in results]}, indent=1))
    return results


if __name__ == "__main__":
    main()
