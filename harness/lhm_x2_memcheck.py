"""Re-runs only check 5 (GPU memory cross-check) from lhm_x2_control.py, now that the PID bug is fixed (the server was
launched via a cmd.exe wrapper, so proc.pid was the wrapper's PID, not llama-server.exe's; \\GPU Process Memory(pid_*)
therefore never matched). Reuses its helpers rather than duplicating them.

Usage on evo-x2: python lhm_x2_memcheck.py <LibreHardwareMonitorLib.dll> <output prefix> <llama-server path>
  <model path> <port>
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lhm_x2_control as base  # noqa: E402


def main():
    dll, prefix, llama, model, port = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    out = prefix + "_lhm_x2.jsonl"
    Path(out).unlink(missing_ok=True)
    rd = base.start_reader(dll, out)
    time.sleep(8)
    proc = base.subprocess.Popen([llama, "-m", model, "--port", port, "-ctk", "f16", "-ctv", "f16",
                                  "-fa", "on", "-ngl", "99", "-np", "1", "-t", "4", "--no-context-shift",
                                  "--log-file", prefix + "_memck_srv.log", "--log-verbosity", "4"],
                                 stdout=base.subprocess.DEVNULL, stderr=base.subprocess.DEVNULL, stdin=base.subprocess.DEVNULL)
    healthy = False
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if json.loads(r.read()).get("status") == "ok":
                    healthy = True
                    break
        except Exception:
            pass
        time.sleep(1)
    if not healthy:
        print(json.dumps({"pass": False, "why": "server never became healthy"}))
        base.m3.kill_tree(proc.pid)
        base.m3.kill_tree(rd.pid)
        sys.exit(2)

    prompt = ("Repeat the following sentence forty times, incrementing the counter each time, one per line. "
              "Sentence: 'The quick brown fox jumps over the lazy dog, testing token number %d.'\n" * 150)
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 128, "temperature": 0, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
    t1 = time.time()
    srv_pid = proc.pid
    mem_out = base.ps(f'(Get-Counter -Counter "\\GPU Process Memory(pid_{srv_pid}_*)\\Dedicated Usage","\\GPU Process Memory(pid_{srv_pid}_*)\\Shared Usage" '
                      f'-ErrorAction SilentlyContinue).CounterSamples | Select Path,CookedValue | ConvertTo-Json -Compress')
    rows = base.read_rows(out)
    is_gpu_mem = lambda k: "radeon" in k.lower() and ("memory" in k.lower() or "vram" in k.lower())
    lhm_mem = base.med_window(rows, t1 - 10, t1, is_gpu_mem)
    try:
        ctr = json.loads(mem_out) if mem_out else []
        ctr = ctr if isinstance(ctr, list) else [ctr]
    except Exception:
        ctr = []
    ctr_ded = next((c["CookedValue"] for c in ctr if "dedicated" in c.get("Path", "").lower()), None)
    ctr_shr = next((c["CookedValue"] for c in ctr if "shared" in c.get("Path", "").lower()), None)
    ctr_total_mib = ((ctr_ded or 0) + (ctr_shr or 0)) / 2 ** 20
    agree = None
    if lhm_mem and ctr_total_mib:
        agree = abs(lhm_mem - ctr_total_mib) / max(lhm_mem, ctr_total_mib) <= 0.05
    result = {"srv_pid": srv_pid, "lhm_mib": lhm_mem, "counter_dedicated_mib": (ctr_ded or 0) / 2 ** 20,
              "counter_shared_mib": (ctr_shr or 0) / 2 ** 20, "counter_total_mib": ctr_total_mib,
              "counter_raw": mem_out, "available": lhm_mem is not None, "pass": bool(agree)}
    print(json.dumps(result, indent=1))
    base.m3.kill_tree(proc.pid)
    time.sleep(3)
    base.m3.kill_tree(rd.pid)
    sys.exit(0 if agree else 2)


if __name__ == "__main__":
    main()
