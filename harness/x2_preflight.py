"""Reboot-survival and pre-run preflight for evo-x2. Read-only checks, with two narrowly-scoped self-healing actions
(restart sshd/Tailscale if stopped -- the same idempotent fix as scripts/x2_boot_task.ps1, restart the download
window or a stalled run queue if either died) explicitly requested by the user rather than run silently.

Usage on evo-x2: python x2_preflight.py [--fix]
Without --fix, every check is report-only. With --fix, sshd/Tailscale are restarted if stopped, dl.ps1 is relaunched
if C:\\apu\\models\\sha256.txt does not end with ALL DONE and no dl.ps1 process is running, and the queue's current
"running" entry is relaunched if its process is gone but its status is still "running" (a crash, not a normal
run_end -- t2s_queue.advance() already moved a normal completion to "done" or "error").
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

OVN = r"C:\apu\ovn"
MODELS = r"C:\apu\models"


def ps(cmd, timeout=30):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=timeout).stdout.strip()


def check_service(name):
    out = ps(f'Get-Service {name} -ErrorAction SilentlyContinue | Select Status,StartType | ConvertTo-Json -Compress')
    try:
        d = json.loads(out) if out else None
    except Exception:
        d = None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()
    report = {}

    sshd = check_service("sshd")
    report["sshd"] = sshd
    report["sshd_ok"] = bool(sshd and sshd.get("Status") == 4 and sshd.get("StartType") == 2)  # Running, Automatic
    if args.fix and sshd and not report["sshd_ok"]:
        ps("Set-Service sshd -StartupType Automatic; Start-Service sshd")
        report["sshd_fixed"] = check_service("sshd")

    ts = check_service("Tailscale")
    report["tailscale"] = ts
    report["tailscale_ok"] = bool(ts and ts.get("Status") == 4)
    if args.fix and ts and not report["tailscale_ok"]:
        ps("Start-Service Tailscale")
        report["tailscale_fixed"] = check_service("Tailscale")

    listen = ps('(Get-NetTCPConnection -LocalPort 22 -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count')
    report["port_22_listening"] = listen.strip() not in ("", "0")

    fw = ps('(Get-NetFirewallRule -Enabled True -Direction Inbound -Action Allow -ErrorAction SilentlyContinue | '
            'Get-NetFirewallPortFilter -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -eq "22" } | Measure-Object).Count')
    report["firewall_rule_present"] = fw.strip() not in ("", "0")

    wu_expiry = ps('(Get-ItemProperty -Path "HKLM:\\SOFTWARE\\Microsoft\\WindowsUpdate\\UX\\Settings" -Name PauseUpdatesExpiryTime '
                   '-ErrorAction SilentlyContinue).PauseUpdatesExpiryTime')
    report["windows_update_paused_until"] = wu_expiry or None
    report["windows_update_paused"] = bool(wu_expiry)

    reboot_pending = ps('Test-Path "HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Component Based Servicing\\RebootPending"')
    report["reboot_pending"] = reboot_pending.strip() == "True"

    sha_path = Path(MODELS) / "sha256.txt"
    sha_text = sha_path.read_text(encoding="utf-8", errors="replace") if sha_path.exists() else ""
    report["downloads_all_done"] = "ALL DONE" in sha_text
    dl_running = ps('(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*dl.ps1*" } | Measure-Object).Count')
    report["dl_ps1_running"] = dl_running.strip() not in ("", "0")
    if args.fix and not report["downloads_all_done"] and not report["dl_ps1_running"]:
        ps(r"Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File C:\apu\dl.ps1'")
        report["dl_ps1_relaunched"] = True

    q_path = Path(OVN) / "queue_state.json"
    if q_path.exists():
        queue = json.loads(q_path.read_text(encoding="utf-8"))
        running = next((it for it in queue if it["status"] == "running"), None)
        report["queue_running_entry"] = running["id"] if running else None
        if running:
            cmd0 = running["cmd"][0]
            alive = ps(f'(Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like "*{Path(running["cmd"][1]).name}*" }} | Measure-Object).Count')
            report["queue_running_process_alive"] = alive.strip() not in ("", "0")
            if args.fix and running and alive.strip() in ("", "0"):
                quoted = " ".join(f'"{c}"' if " " in c else c for c in running["cmd"])
                log_path = f"{OVN}\\queue_{running['id']}_relaunch.log"
                full_cmdline = f'cmd.exe /c cd /d {OVN} && {quoted} > "{log_path}" 2>&1'
                ps(f"$cmd = '{full_cmdline}'; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine=$cmd; CurrentDirectory='{OVN}'}}")
                report["queue_relaunched"] = running["id"]
    else:
        report["queue_running_entry"] = None

    print(json.dumps(report, indent=1))
    sys.exit(0 if (report["sshd_ok"] and report["tailscale_ok"] and report["port_22_listening"] and
                   report["firewall_rule_present"] and report["windows_update_paused"] and not report["reboot_pending"]) else 1)


if __name__ == "__main__":
    main()
