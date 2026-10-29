"""Task BK: A31 (compiled KV == requested, backend PA) and A32 (failure class and phase) over every
child result of every CC cell (unsealed work/ copies; same run_ids). Writes provenance.json."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent

from tools.cc_failure import classify  # noqa: E402

WANT = {"gpu_only_f16": "f16", "gpu_only_u4": "u4", "gpu_only_i8": "i8"}


def main() -> int:
    cells = json.loads((OUT / "cells.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {"loads": Counter(), "mismatch": [], "failures": [], "classes": Counter()}
    seen: set[str] = set()
    for c in cells:
        if not c.get("run_id"):
            continue
        work = Path(str(ROOT / c["dir"]).replace("__sealed_", "__")) / "work"
        for f in sorted(work.glob("*.result.json")):
            key = str(f)
            if key in seen:
                continue
            seen.add(key)
            ch = json.loads(f.read_text(encoding="utf-8"))
            for ld in ch.get("loads") or []:
                comp = ld.get("compiled") or {}
                want = WANT.get(str(c.get("arm")))
                got = comp.get("kv_cache_precision")
                be = comp.get("backend")
                out["loads"][f"{want}->{got}:{be}"] += 1
                if want and (got != want or be != "PA"):
                    out["mismatch"].append([c["label"], c["run_id"], f.name, want, got, be])
            if ch.get("failure_mode"):
                k = classify(ch)
                row = [
                    c["host"],
                    c["label"],
                    c["run_id"],
                    f.name,
                    ch.get("failure_mode"),
                    k.get("class"),
                    k.get("phase"),
                ]
                out["failures"].append(row)
                out["classes"][f"{c['host']}:{k.get('class')}:{k.get('phase')}"] += 1
    out["loads"] = dict(out["loads"])
    out["classes"] = dict(out["classes"])
    (OUT / "provenance.json").write_text(json.dumps(out, indent=1) + "\n", encoding="ascii")
    print(
        json.dumps(
            {"loads": out["loads"], "mismatch": out["mismatch"][:10], "classes": out["classes"]},
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
