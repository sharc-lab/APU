"""pandas_groupby_hog.py -- a large groupby-aggregate loop as a realistic
agent-tool-work co-runner. Runs for --duration-s, repeatedly regenerating and
aggregating a large DataFrame (not reusing one across iterations, so this
also churns allocator/memory activity like a real data-analysis task would,
not just CPU)."""

import argparse
import time

import numpy as np
import pandas as pd

N_ROWS = 5_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, required=True)
    args = ap.parse_args()

    t_end = time.monotonic() + args.duration_s
    iters = 0
    while time.monotonic() < t_end:
        df = pd.DataFrame({
            "key": np.random.randint(0, 10_000, N_ROWS),
            "a": np.random.randn(N_ROWS),
            "b": np.random.randn(N_ROWS),
        })
        _ = df.groupby("key").agg({"a": ["mean", "std", "sum"], "b": ["mean", "max"]})
        iters += 1
    print(f"pandas_groupby_hog: {iters} iterations in {args.duration_s}s")


if __name__ == "__main__":
    main()
