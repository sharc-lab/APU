"""Cloud spend so far, BILLED vs UPPER BOUND, per job, without any live call (operator 2026-10-22).

Before 2026-10-22 src/cloud/client.py wrote every call to results/cloud_ledger.jsonl at the UPPER BOUND (every
input token at the uncached input rate). The billed cost (cache reads at the hit rate, cache writes at the
5-minute write rate) is recoverable only for calls whose provider usage was recorded somewhere: the per-task rows
of the cloud jobs' result files carry it (cache_read_tokens, cache_write_tokens, uncached_input_tokens, and the
harness's own billed figure "usd" computed from them with the shared price file; "usd_ledger" is what the ledger
charged for the same calls, the upper bound).

Per job (one result file = one job run):
  billed       = sum of "usd" over the job's leaf call rows (LEAF_RECORDS; session roll-up rows are skipped, they
                 repeat their turns), plus each warm-up call at its ledger cost (a warm-up row records no cache
                 breakdown; its prompt is under every model's minimum cacheable length, so no cache read or write
                 can occur and billed = upper bound for it)
  upper bound  = the ledger rows whose timestamp falls in the job's window [run_start ts, run_end ts or the file's
                 last ts; a run with no run_end row (aborted) keeps every ledger call up to the next job's start,
                 since its unfinished last turn left no row]; scripts/all_cloud_b3.py makes its warm-up calls before it writes run_start, so its window
                 opens the warm-ups' seconds plus B3_WARMUP_SLACK_S earlier; cross-checked against the sum of the
                 rows' "usd_ledger"
  unrecorded   = ledger cost in the window not covered by the job's rows (calls with no usage on file): counted
                 at the upper bound in both columns
Ledger rows outside every job window are calls with no usage on file at all (the smoke calls and anything else):
upper bound only, counted at the upper bound in both columns. New-shape ledger rows (cost_usd_upper_bound present)
are taken as written: cost_usd billed, cost_usd_upper_bound upper bound.

Also cross-checks each sonnet row's "usd" against src/cloud/client.py::billed_cost_usd on the row's token counts
(sonnet has one context tier, so a per-row aggregate prices the same as per call).

Usage: py -3.12 analysis/cloud_spend_billed.py [--root DIR ...]
  --root (repeatable): extra directories whose results/ also hold job files (read-only); the ledger is read from
  the first root that has one. Default: this repo only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

JOB_GLOBS = ("cloud_redo_v1.jsonl", "cloud_redo_full_context_v1*.jsonl", "all_cloud_b3_v1*.jsonl")
LEAF_RECORDS = ("cloud_redo_task", "r2a_turn", "pilot_turn")
LEDGER = "results/cloud_ledger.jsonl"
B3_WARMUP_SLACK_S = 2.0


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict):
                rows.append(r)
    return rows


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def job_files(roots: list[Path]) -> list[Path]:
    """Job result files by name, first root wins for a name present in several."""
    seen: dict[str, Path] = {}
    for root in roots:
        for g in JOB_GLOBS:
            for p in sorted((root / "results").glob(g)):
                seen.setdefault(p.name, p)
    return [seen[k] for k in sorted(seen)]


def _in_range(t: datetime | None, start: datetime | None, end: datetime | None) -> bool:
    """start <= t < end; no range means every row. A row with no timestamp is in range only when no range is set."""
    if start is None and end is None:
        return True
    if t is None:
        return False
    return (start is None or t >= start) and (end is None or t < end)


def _job(path: Path, rng_start: datetime | None = None, rng_end: datetime | None = None) -> dict:
    """One job file. The job's window comes from all its rows; with a range [start, end) only the leaf and warm-up
    rows whose ts_utc falls in it are summed (a row is stamped when its turn finishes, so a turn that straddles a
    range edge is counted on the side its row lands on)."""
    rows = _read_jsonl(path)
    start = next((r for r in rows if r.get("record") == "run_start"), {})
    end = next((r for r in rows if r.get("record") == "run_end"), None)
    stamps = [_ts(r["ts_utc"]) for r in rows if r.get("ts_utc")]
    leaf_all = [r for r in rows if r.get("record") in LEAF_RECORDS and "usd" in r]
    warm_all = [r for r in rows if r.get("record") == "warmup"]
    leaf = [r for r in leaf_all if _in_range(_ts(r.get("ts_utc")), rng_start, rng_end)]
    warm = [r for r in warm_all if _in_range(_ts(r.get("ts_utc")), rng_start, rng_end)]
    t0 = _ts(start.get("ts_utc")) or min(stamps)
    if start.get("harness") == "all_cloud_b3":
        # scripts/all_cloud_b3.py runs its warm-up calls BEFORE it writes run_start (cloud_redo.py after)
        t0 -= timedelta(seconds=sum(float(r.get("seconds") or 0) for r in warm_all) + B3_WARMUP_SLACK_S)
    return {
        "job": path.stem, "file": path, "t0": t0,
        "t1": _ts(end.get("ts_utc")) if end and end.get("ts_utc") else max(stamps),
        "has_end": bool(end and end.get("ts_utc")),
        "n_leaf": len(leaf), "n_calls": sum(int(r.get("n_calls") or len(r.get("calls") or []) or 1) for r in leaf),
        "billed_rows": sum(float(r["usd"]) for r in leaf),
        "upper_rows": sum(float(r.get("usd_ledger") or 0) for r in leaf),
        "warmup_usd": sum(float(r.get("usd_ledger") or 0) for r in warm),
        "leaf": leaf, "model_ids": sorted({r.get("model_id") for r in leaf_all + warm_all if r.get("model_id")}
                                          | ({start["model_id"]} if start.get("model_id") else set())),
    }


def _ledger_rows(path: Path) -> list[dict]:
    out = []
    for r in _read_jsonl(path):
        if r.get("record") == "alert" or "cost_usd" not in r:
            continue
        ub = r.get("cost_usd_upper_bound")
        out.append({"t": _ts(r.get("timestamp")), "model_id": r.get("model_id"),
                    "upper": float(ub if ub is not None else r["cost_usd"]),
                    "billed_known": float(r["cost_usd"]) if ub is not None else None})
    return out


def _check_billed(leaf: list[dict]) -> tuple[int, int]:
    """(rows checked, rows matching within 1e-9 USD) for claude-sonnet-5-5 rows: the harness's usd vs
    src/cloud/client.py::billed_cost_usd on the row's token counts."""
    from src.cloud.client import billed_cost_usd  # noqa: PLC0415
    n = ok = 0
    for r in leaf:
        if r.get("model_id") != "claude-sonnet-5-5" or r.get("uncached_input_tokens") is None:
            continue
        u = {k: int(r.get(k) or 0) for k in ("input_tokens", "uncached_input_tokens", "cache_read_tokens",
                                             "cache_write_tokens", "output_tokens")}
        n += 1
        ok += abs(billed_cost_usd("claude-sonnet-5-5", u) - float(r["usd"])) < 1e-9
    return n, ok


def _rel_or_name(path: Path, roots: list[Path]) -> str:
    root = next((r for r in roots if path.is_relative_to(r)), None)
    return (str(path.relative_to(root)) if root is not None else str(path)).replace("\\", "/")


def compute(roots: list[Path] | None = None, ledger_path: Path | None = None,
            start: datetime | None = None, end: datetime | None = None) -> dict:
    """The register row's figures. ledger_path: read this ledger instead of the first root's results/cloud_ledger.jsonl
    (src/cloud/client.py's billed cap basis passes its own ledger); then the job files come from the ledger's
    repo (its parent's parent, when the ledger sits in a results/ directory) unless roots is given, and a ledger
    anywhere else has no job files, so every row is taken as written (cost_usd billed when the row is new-shape,
    else its upper bound). start/end (aware datetimes): count only ledger rows and job rows stamped in [start, end)
    (the console check compares the Console's date range); job windows still come from the whole job file."""
    if ledger_path is not None:
        ledger_path = Path(ledger_path).resolve()
        if roots is None:
            roots = [ledger_path.parent.parent] if ledger_path.parent.name == "results" else []
        roots = [Path(r).resolve() for r in roots]
        if not ledger_path.exists():
            raise FileNotFoundError(str(ledger_path))
    else:
        roots = [Path(r) for r in (roots or [REPO])]
        ledger_path = next((r / LEDGER for r in roots if (r / LEDGER).exists()), None)
        if ledger_path is None:
            raise FileNotFoundError(LEDGER)
    ledger = [r for r in _ledger_rows(ledger_path) if _in_range(r["t"], start, end)]
    jobs = [_job(p, start, end) for p in job_files(roots)]
    jobs.sort(key=lambda j: j["t0"])
    for j, nxt in zip(jobs, jobs[1:]):
        if not j["has_end"]:   # an aborted run: calls after its last row (an unfinished turn) are still its own
            j["t1"] = nxt["t0"] - timedelta(microseconds=1)
    used = [False] * len(ledger)
    out_jobs = []
    n_checked = n_ok = 0
    for j in jobs:
        idx = [i for i, row in enumerate(ledger) if row["t"] and j["t0"] <= row["t"] <= j["t1"] and not used[i]]
        for i in idx:
            used[i] = True
        upper_ledger = sum(ledger[i]["upper"] for i in idx)
        covered = j["upper_rows"] + j["warmup_usd"]
        unrecorded = max(0.0, upper_ledger - covered)
        if unrecorded < 1e-6:
            unrecorded = 0.0
        billed = j["billed_rows"] + j["warmup_usd"] + unrecorded
        c, k = _check_billed(j["leaf"])
        n_checked, n_ok = n_checked + c, n_ok + k
        out_jobs.append({"job": j["job"], "window_utc": [j["t0"].isoformat(), j["t1"].isoformat()],
                         "ledger_calls": len(idx), "leaf_rows": j["n_leaf"], "calls_with_usage": j["n_calls"],
                         "models": j["model_ids"], "billed_usd": billed, "upper_bound_usd": upper_ledger,
                         "upper_bound_rows_usd": covered, "unrecorded_upper_usd": unrecorded,
                         "warmup_usd": j["warmup_usd"], "file": _rel_or_name(j["file"], roots)})
    rest = [ledger[i] for i in range(len(ledger)) if not used[i]]
    rest_upper = sum(r["upper"] for r in rest)
    rest_billed = sum(r["billed_known"] if r["billed_known"] is not None else r["upper"] for r in rest)
    rest_unrecorded = sum(r["upper"] for r in rest if r["billed_known"] is None)
    total_upper = sum(r["upper"] for r in ledger)
    total_billed = sum(j["billed_usd"] for j in out_jobs) + rest_billed
    unrecorded_total = sum(j["unrecorded_upper_usd"] for j in out_jobs) + rest_unrecorded + \
        sum(j["warmup_usd"] for j in out_jobs)
    return {"ledger": _rel_or_name(ledger_path, roots),
            "ledger_calls": len(ledger), "jobs": out_jobs,
            "unattributed": {"ledger_calls": len(rest), "billed_usd": rest_billed, "upper_bound_usd": rest_upper,
                             "windows_utc": sorted({r["t"].isoformat()[:16] for r in rest if r["t"]})[:3]
                             + (["..."] if len(rest) > 3 else []),
                             "models": sorted({r["model_id"] for r in rest if r["model_id"]})},
            "total_billed_usd": total_billed, "total_upper_bound_usd": total_upper,
            "upper_bound_only_usd": unrecorded_total, "sonnet_rows_checked": n_checked, "sonnet_rows_match": n_ok}


def summary_value(res: dict) -> str:
    parts = [f"{j['job']}: billed USD {j['billed_usd']:.4f} vs upper bound USD {j['upper_bound_usd']:.4f}"
             f" ({j['ledger_calls']} ledger calls)" for j in res["jobs"]]
    u = res["unattributed"]
    parts.append(f"calls outside every job window (no result file on disk): USD {u['upper_bound_usd']:.4f} upper bound only"
                 f" ({u['ledger_calls']} ledger calls)")
    return (f"cloud spend so far, billed USD {res['total_billed_usd']:.4f} vs upper bound USD "
            f"{res['total_upper_bound_usd']:.4f} ({res['ledger_calls']} ledger calls in {res['ledger']}; "
            f"of the billed total, USD {res['upper_bound_only_usd']:.4f} is counted at the upper bound because its "
            f"usage is not recorded); " + "; ".join(parts) +
            f"; billed check: {res['sonnet_rows_match']}/{res['sonnet_rows_checked']} sonnet rows match "
            "src/cloud/client.py::billed_cost_usd")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", action="append", default=None)
    a = ap.parse_args()
    res = compute([Path(r) for r in a.root] if a.root else None)
    print(summary_value(res))
    print()
    print("| job | window (UTC) | ledger calls | billed USD | upper bound USD | rows upper USD | unrecorded USD |")
    print("|---|---|---|---|---|---|---|")
    for j in res["jobs"]:
        print(f"| {j['job']} | {j['window_utc'][0][11:19]} to {j['window_utc'][1][11:19]} | {j['ledger_calls']} | "
              f"{j['billed_usd']:.4f} | {j['upper_bound_usd']:.4f} | {j['upper_bound_rows_usd']:.4f} | "
              f"{j['unrecorded_upper_usd']:.4f} |")
    u = res["unattributed"]
    print(f"| no usage on file | {u['windows_utc']} | {u['ledger_calls']} | {u['billed_usd']:.4f} | "
          f"{u['upper_bound_usd']:.4f} | n/a | {u['upper_bound_usd']:.4f} |")


if __name__ == "__main__":
    main()
