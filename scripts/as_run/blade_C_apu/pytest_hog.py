"""pytest_hog.py -- repeatedly runs this repo's test suite as a co-runner
workload. A single pytest invocation finishes in well under our measurement
window, so this loops it for --duration-s to sustain load throughout."""

import argparse
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--repo-path", type=str, required=True)
    args = ap.parse_args()

    t_end = time.monotonic() + args.duration_s
    iters = 0
    while time.monotonic() < t_end:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "-x", "--timeout=60"],
            cwd=args.repo_path, capture_output=True, timeout=120, stdin=subprocess.DEVNULL,
        )
        iters += 1
    print(f"pytest_hog: {iters} full test-suite passes in {args.duration_s}s")


if __name__ == "__main__":
    main()
