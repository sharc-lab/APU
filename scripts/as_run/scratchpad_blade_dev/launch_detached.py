"""
Launch fig61_stagec_sweep.py --full as a fully detached Windows process.
Run: py -3 launch_detached.py
Returns immediately after printing the PID.
"""
import subprocess, sys, os, time

REPO  = r"C:\apu\APU"
SCRIPT = REPO + r"\harness\fig61_stagec_sweep.py"
PYTHON = sys.executable
STDOUT = REPO + r"\results\fig61_full_sweep2_stdout.txt"
STDERR = REPO + r"\results\fig61_full_sweep2_stderr.txt"

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUNBUFFERED"] = "1"

with open(STDOUT, "w") as out, open(STDERR, "w") as err:
    proc = subprocess.Popen(
        [PYTHON, "-u", SCRIPT, "--full"],
        stdout=out,
        stderr=err,
        env=env,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )

print(f"PID {proc.pid}")
print(f"stdout -> {STDOUT}")
print(f"stderr -> {STDERR}")
