"""Deploy committed repo files to a target machine byte-exact (from `git show HEAD:<path>`), and write the
expected-blob list that harness/run_provenance.verify_deployed_blobs checks on the machine.

Usage: py -3.12 scripts/deploy_evo.py <remote_dir> <repo_path> [<repo_path> ...] [--host evo-t2s|evo-x2]
--host defaults to evo-t2s. Refuses if any listed path has uncommitted changes, or if the machine that answers on the
configured SSH host does not report the expected hostname (harness/host_config.py HOSTS). Uses
C:\\Windows\\System32\\OpenSSH\\ssh.exe and scp.exe, BatchMode.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
SCP = r"C:\Windows\System32\OpenSSH\scp.exe"
sys.path.insert(0, str(REPO / "harness"))
import host_config as hc  # noqa: E402
from proc_util import run_hidden  # noqa: E402


def git(*a):
    return run_hidden(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def main():
    args = sys.argv[1:]
    alias = "evo-t2s"
    if "--host" in args:
        i = args.index("--host")
        alias = args[i + 1]
        del args[i:i + 2]
    remote, paths = args[0], args[1:]
    key = hc.ALIASES.get(alias.lower())
    if key is None:
        raise SystemExit(f"unknown --host {alias!r}: not in harness/host_config.py ALIASES")
    HOST = hc.HOSTS[key]["ssh_host"]
    dirty = git("status", "--porcelain", "--", *paths).decode().strip()
    if dirty:
        raise SystemExit("refusing to deploy uncommitted files:\n" + dirty)
    host = run_hidden([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", HOST, "hostname"],
                      capture_output=True, text=True, check=True).stdout.strip()
    if host.upper() != key:
        raise SystemExit(f"wrong host {host}, expected {key}")
    run_hidden([SSH, "-o", "BatchMode=yes", HOST, f"New-Item -ItemType Directory -Force {remote} | Out-Null"],
              check=True)
    head = git("rev-parse", "HEAD").decode().strip()
    blobs = {}
    with tempfile.TemporaryDirectory() as td:
        for p in paths:
            name = Path(p).name
            data = git("show", f"HEAD:{p}")
            (Path(td) / name).write_bytes(data)
            blobs[name] = git("rev-parse", f"HEAD:{p}").decode().strip()
            run_hidden([SCP, "-q", "-o", "BatchMode=yes", str(Path(td) / name), f"{HOST}:{remote}/{name}"], check=True)
        exp = Path(td) / "expected_blobs.json"
        exp.write_text(json.dumps({"git_head": head, "blobs": blobs}, indent=1))
        run_hidden([SCP, "-q", "-o", "BatchMode=yes", str(exp), f"{HOST}:{remote}/expected_blobs.json"], check=True)
    print(json.dumps({"deployed": list(blobs), "git_head": head, "host": key}))


if __name__ == "__main__":
    main()
