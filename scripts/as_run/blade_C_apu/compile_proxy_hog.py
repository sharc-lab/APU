"""compile_proxy_hog.py -- repeated Python bytecode compilation of the repo
tree as a co-runner workload.

HONEST LABELING: this is NOT a C/C++ compiler. Neither evo-t2s nor the Blade
14 workstation was confirmed to have a ready-to-build C/C++ project plus
toolchain available for this experiment, and standing this up correctly
(consistent compiler, consistent project, on both machines) was judged not
worth the time against the actual question (does a CPU+disk+memory-churning
background task slow the co-resident LLM server). `compileall` genuinely
exercises CPU (parsing, bytecode generation), disk I/O (reading every .py
file, writing every .pyc), and allocator churn (one AST + code-object tree
per file) -- a real if different memory access pattern from a C++ compiler's
translation-unit-at-a-time allocation-heavy behavior. Report this as "Python
bytecode compilation of the repo tree," not as "compiling C++," in any
results table."""

import argparse
import compileall
import time
from pathlib import Path

REPO = Path(__file__).parent  # overridden by caller passing the real repo root as cwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--repo-path", type=str, required=True)
    args = ap.parse_args()

    t_end = time.monotonic() + args.duration_s
    iters = 0
    while time.monotonic() < t_end:
        compileall.compile_dir(args.repo_path, quiet=2, force=True, workers=0)
        iters += 1
    print(f"compile_proxy_hog: {iters} full-tree compileall passes in {args.duration_s}s")


if __name__ == "__main__":
    main()
