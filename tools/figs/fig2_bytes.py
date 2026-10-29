"""Fig 2: where context bytes go. Bytes per token, KV (k) vs non-KV prefill working set (a),
per KV precision: predicted bars, with the measured CC-BYTES / CC-ALLOC values when present."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.figs._common import CC_DIR, DEFAULT_OUT, Inputs, measured, save, setup


def make(out_dir: Path) -> dict[str, Any]:
    plt = setup()
    inp = Inputs()
    base = inp.json(CC_DIR / "CC_PREDICTIONS.json")
    bl = inp.json(CC_DIR / "CC_BL_PREDICTIONS.json")
    meas = measured(inp).get("bytes_per_token", {})
    k = base["inputs"]["k_kv_B_per_token"]
    a = bl["hypotheses"]["H_act"]["a_B_per_token"]
    precs = ["f16", "u8", "u4"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = range(len(precs))
    ax.bar(x, [k[p] / 1024 for p in precs], color="C0", label="KV k (shapes)")
    ax.bar(
        x,
        [a / 1024] * 3,
        bottom=[k[p] / 1024 for p in precs],
        color="C3",
        label="non-KV a (fitted)",
    )
    for i, p in enumerate(precs):
        m = meas.get(p)
        if m:
            ax.scatter(
                [i + 0.25],
                [m["total_KiB"]],
                color="k",
                zorder=5,
                label="measured" if i == 0 else None,
            )
        else:
            ax.text(i + 0.25, (k[p] + a) / 1024, "weekend", fontsize=6, color="0.4", rotation=90)
    ax.set_xticks(list(x), precs)
    ax.set_ylabel("KiB per context token (uncapped prefill)")
    ax.legend(fontsize=6)
    ax.set_title("Where context bytes go (XPS, Qwen3-4B)", fontsize=8)
    return save(
        fig,
        "fig2_bytes",
        inp,
        out_dir,
        ["bars are DERIVED; measured points from analysis/cc_20261026/measured.json"],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    make(ap.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
