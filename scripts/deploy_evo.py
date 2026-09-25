"""Deploy committed repo files to evo-t2s byte-exact (from `git show HEAD:<path>`), and write the expected-blob list
that harness/run_provenance.verify_deployed_blobs checks on the machine.

Usage: py -3.12 scripts/deploy_evo.py <remote_dir> <repo_path> [<repo_path> ...]
Refuses if any listed path has uncommitted changes. Uses C:\\Windows\\System32\\OpenSSH\\ssh.exe and scp.exe, BatchMode.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
SCP = r"C:\Windows\System32\OpenSSH\scp.exe"
HOST = "sharc@100.72.40.24"


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def main():
    remote, paths = sys.argv[1], sys.argv[2:]
    dirty = git("status", "--porcelain", "--", *paths).decode().strip()
    if dirty:
        raise SystemExit("refusing to deploy uncommitted files:\n" + dirty)
    host = subprocess.run([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", HOST, "hostname"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if host.upper() != "EVO-T2S":
        raise SystemExit(f"wrong host {host}")
    subprocess.run([SSH, "-o", "BatchMode=yes", HOST, f"New-Item -ItemType Directory -Force {remote} | Out-Null"],
                   check=True)
    head = git("rev-parse", "HEAD").decode().strip()
    blobs = {}
    with tempfile.TemporaryDirectory() as td:
        for p in paths:
            name = Path(p).name
            data = git("show", f"HEAD:{p}")
            (Path(td) / name).write_bytes(data)
            blobs[name] = git("rev-parse", f"HEAD:{p}").decode().strip()
            subprocess.run([SCP, "-q", "-o", "BatchMode=yes", str(Path(td) / name), f"{HOST}:{remote}/{name}"], check=True)
        exp = Path(td) / "expected_blobs.json"
        exp.write_text(json.dumps({"git_head": head, "blobs": blobs}, indent=1))
        subprocess.run([SCP, "-q", "-o", "BatchMode=yes", str(exp), f"{HOST}:{remote}/expected_blobs.json"], check=True)
    print(json.dumps({"deployed": list(blobs), "git_head": head}))


if __name__ == "__main__":
    main()
