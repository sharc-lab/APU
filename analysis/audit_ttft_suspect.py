"""
Read-only audit: count rows where ttft_ms == latency_ms exactly.

Background
----------
Before 2026-09-14, harness/runner.py line 174 returned
`ttft_ms or latency_ms`, meaning that whenever no content token was
received (empty content field) ttft_ms silently equalled total latency.
Those rows never had a real TTFT measurement — the field just duplicated
latency_ms.

Root cause for runs 20260812T1*:
The runner at that time accumulated text from a different streaming
endpoint format.  The token detection check reads
chunk["message"]["content"], which is the /api/chat streaming format.
If those runs used /api/generate (content in chunk["response"]),
the check always saw "" for every chunk, so ttft_ms was never set and
the fallback fired for every call.  tokens_out is still correct (from
eval_count in the done chunk, which is identical in both formats).

For rows with non-exact scorers (cod_01, str_01), it is ambiguous
whether the captured output was empty or not: "0/1 passed" is produced
by both empty code and wrong code; "no JSON object found" is produced
by both empty output and malformed JSON.  The result row does not store
raw output.  The ambiguous rows are flagged but not asserted non-empty.

All 86 suspect rows have done_reason=None and outcome_class=None,
confirming they predate 2026-09-14.  Under the post-Phase-1 code,
cache hits now emit ttft_source="replay-unavailable" / ttft_ms=None
regardless of the cached telemetry, so these rows do not affect new
runs.

THIS SCRIPT DOES NOT MODIFY ANY FILE.  It only reads and prints.

Usage
-----
    py -3.12 analysis/audit_ttft_suspect.py            # summary table only
    py -3.12 analysis/audit_ttft_suspect.py --detail   # + per-row breakdown
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
    """True if ttft_ms == latency_ms exactly and ttft_source is not 'streamed'."""
    ttft = row.get("ttft_ms")
    lat = row.get("latency_ms")
    if ttft is None or lat is None:
        return False
    if row.get("ttft_source") == "streamed":
        return False
    return ttft == lat


# Output-emptiness inference from score_detail.
# Only exact-scorer rows produce "got='' want='...'" which is a definitive
# signal that the captured output was an empty string.
# Non-exact scorers (unit_test: "0/1 passed", json_exact: "no JSON object
# found") are ambiguous — the same detail appears for both empty and wrong
# non-empty output.  We report three categories rather than two.
_CONFIRMED_EMPTY = "confirmed-empty"    # got='' in score_detail
_AMBIGUOUS = "ambiguous"               # non-exact scorer; can't tell
_CONFIRMED_NONEMPTY = "confirmed-nonempty"  # (reserved; not currently produced)


def _output_category(row: dict) -> str:
    sd = row.get("score_detail", "")
    if isinstance(sd, str) and sd.startswith("got=''"):
        return _CONFIRMED_EMPTY
    return _AMBIGUOUS


def _detail_lines(suspect: list[dict]) -> list[str]:
    lines = []
    header = (
        f"  {'probe_id':<12} {'output':>17} {'tok_out':>7} "
        f"{'done_reason':>13} {'outcome_class':>15}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in suspect:
        cat = _output_category(r)
        lines.append(
            f"  {r.get('probe_id', '?'):<12} {cat:>17} "
            f"{str(r.get('tokens_out', '?')):>7} "
            f"{str(r.get('done_reason', '?')):>13} "
            f"{str(r.get('outcome_class', '?')):>15}"
        )
    return lines


def main() -> None:
    detail = "--detail" in sys.argv

    files = sorted(RESULTS_DIR.glob("*.json")) + sorted(RESULTS_DIR.glob("*.jsonl"))
    if not files:
        print(f"No result files found under {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    total_suspect = 0
    total_rows = 0
    total_confirmed_empty = 0
    total_ambiguous = 0

    header = (
        f"{'File':<55}  {'rows':>7}  {'suspect':>7}  {'%':>6}"
        f"  {'confirmed-empty':>15}  {'ambiguous':>9}"
    )
    print(header)
    print("-" * len(header))

    for path in files:
        rows = list(_iter_rows(path))
        suspect = [r for r in rows if _is_suspect(r)]
        n_rows = len(rows)
        n_suspect = len(suspect)
        n_confirmed_empty = sum(
            1 for r in suspect if _output_category(r) == _CONFIRMED_EMPTY
        )
        n_ambiguous = n_suspect - n_confirmed_empty
        total_rows += n_rows
        total_suspect += n_suspect
        total_confirmed_empty += n_confirmed_empty
        total_ambiguous += n_ambiguous
        pct = (n_suspect / n_rows * 100) if n_rows > 0 else 0.0
        flag = "  <-- ALL SUSPECT" if n_suspect == n_rows and n_rows > 0 else ""
        print(
            f"{path.name:<55}  {n_rows:>7}  {n_suspect:>7}  {pct:>5.1f}%"
            f"  {n_confirmed_empty:>15}  {n_ambiguous:>9}{flag}"
        )
        if detail and suspect:
            for line in _detail_lines(suspect):
                print(line)

    print("-" * len(header))
    total_pct = (total_suspect / total_rows * 100) if total_rows > 0 else 0.0
    print(
        f"{'TOTAL':<55}  {total_rows:>7}  {total_suspect:>7}  {total_pct:>5.1f}%"
        f"  {total_confirmed_empty:>15}  {total_ambiguous:>9}"
    )

    print()
    print("DIAGNOSIS")
    print("---------")
    print(
        f"  {total_confirmed_empty} rows confirmed-empty (score_detail starts with \"got=''\"):"
    )
    print(
        "    ttft_ms == latency_ms is the fallback artifact. The runner's token"
    )
    print(
        "    detection (chunk['message']['content']) never matched because these"
    )
    print(
        "    runs used the /api/generate endpoint (content in chunk['response'])."
    )
    print()
    print(
        f"  {total_ambiguous} rows ambiguous (non-exact scorers: unit_test, json_exact):"
    )
    print(
        "    Result rows do not store raw output. 'no JSON object found' and"
    )
    print(
        "    '0/1 passed' both appear for empty and wrong non-empty output."
    )
    print(
        "    Cannot determine emptiness from the result row alone."
    )
    print()
    print(
        "  All 86 suspect rows have done_reason=None and outcome_class=None,"
    )
    print(
        "  confirming they predate 2026-09-14. Under the post-Phase-1 code,"
    )
    print(
        "  cache hits emit ttft_source='replay-unavailable' / ttft_ms=None"
    )
    print(
        "  regardless of cached telemetry. New live calls use /api/chat and"
    )
    print(
        "  read message.content correctly; verify with a live probe before sweep."
    )
    print()
    print("  Filter on ttft_source == 'streamed' for all TTFT analysis.")


if __name__ == "__main__":
    main()
