"""
Read-only audit: count rows where ttft_ms == latency_ms exactly.

Background
----------
Before 2026-09-14, harness/runner.py line 174 returned
`ttft_ms or latency_ms`, meaning that whenever no content token was
received (empty response) the returned ttft_ms silently equalled the
total latency.  Those rows never had a real TTFT measurement — the field
just duplicated latency_ms.

This script scans every .json and .jsonl file under results/ and counts
rows where both fields are present, both are non-null, and their values
are exactly equal.  The count per file is the number of suspect rows.

THIS SCRIPT DOES NOT MODIFY ANY FILE.  It only reads and prints.

Rows with ttft_source == "streamed" are excluded from the count: those
were written after the fix (or in a future re-run) and their equality
would be genuinely coincidental (theoretically possible but not the
fallback artifact).  Rows lacking ttft_source (old rows written before
the field existed) are included in the suspect count if
ttft_ms == latency_ms.

Usage
-----
    py -3.12 analysis/audit_ttft_suspect.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _iter_rows(path: Path):
    """Yield dicts from a .json or .jsonl file (best-effort)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                pass
    else:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
        elif isinstance(obj, dict):
            yield obj


def _is_suspect(row: dict) -> bool:
    """True if ttft_ms is non-null, latency_ms is non-null, they are equal,
    and ttft_source is not "streamed" (exclude post-fix rows)."""
    ttft = row.get("ttft_ms")
    lat = row.get("latency_ms")
    if ttft is None or lat is None:
        return False
    if row.get("ttft_source") == "streamed":
        return False
    return ttft == lat


def main() -> None:
    files = sorted(RESULTS_DIR.glob("*.json")) + sorted(RESULTS_DIR.glob("*.jsonl"))
    if not files:
        print(f"No result files found under {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    total_suspect = 0
    total_rows = 0
    any_suspect = False

    print(f"{'File':<55}  {'rows':>7}  {'suspect':>7}  {'%':>6}")
    print("-" * 80)

    for path in files:
        rows = list(_iter_rows(path))
        suspect = [r for r in rows if _is_suspect(r)]
        n_rows = len(rows)
        n_suspect = len(suspect)
        total_rows += n_rows
        total_suspect += n_suspect
        pct = (n_suspect / n_rows * 100) if n_rows > 0 else 0.0
        flag = "  <-- ALL SUSPECT" if n_suspect == n_rows and n_rows > 0 else ""
        print(f"{path.name:<55}  {n_rows:>7}  {n_suspect:>7}  {pct:>5.1f}%{flag}")
        if n_suspect > 0:
            any_suspect = True

    print("-" * 80)
    total_pct = (total_suspect / total_rows * 100) if total_rows > 0 else 0.0
    print(f"{'TOTAL':<55}  {total_rows:>7}  {total_suspect:>7}  {total_pct:>5.1f}%")

    if any_suspect:
        print()
        print("WARNING: suspect rows have ttft_ms == latency_ms.")
        print("These rows predate the 2026-09-14 fix that removed the")
        print("`ttft_ms or latency_ms` fallback in harness/runner.py.")
        print("Any TTFT analysis over these rows is unreliable.")
        print("Filter on ttft_source == 'streamed' for valid measurements only.")


if __name__ == "__main__":
    main()
