"""Pre-job gate for every live cloud job: per-job spend limit with operator approval, then the Console check
(operator 2026-10-23 and 2026-10-24, register row cloud-spend-billed-vs-upper-bound).

The cap is on BILLED spend (src/cloud/client.py, BILLED CAP BASIS AT START). pre_job() runs before a live job's first
call and refuses it unless, in this order:

  1. PER-JOB LIMIT (operator 2026-10-24: "hard refusal in code above USD 2 estimated per job, unless run with an
     explicit per-job approval flag that records my approval in the ledger"). The job's estimate (each entry point
     states its basis) must be at most limit_usd (ASK_FIRST_USD = 2.00 on the main ledger; the demo lane passes its
     own USD 3 per session), or the job's command line carries --operator-approval "<text>" whose text names the job
     id and exactly one amount "USD <n>" at or above the estimate. The approval never raises a cap and never skips
     the Console check; it is only ever a function argument (the entry point passes its own command-line flag), never
     read from an environment variable or a file. Once every check below passes, it is appended to the job's ledger
     as its own {"record": "operator_approval", ...} row (text, amount, job id, UTC time, git head) before the first
     call; it has no cost_usd, so no spend total counts it.
  2. CONSOLE CHECK. Code cannot read the Anthropic Console; the operator reads it and writes
     results/console_usage_check.json in the main checkout (TEMPLATE below). Refused unless:
     (a) the file exists, parses, and was read within FRESHNESS_HOURS_DEFAULT = 24 h (confirmed by the operator
         2026-10-24). A caller may pass a SHORTER window (tighter); a longer one is refused, so the window cannot be
         loosened at all (no environment variable, approval or not);
     (b) the ledgers' billed total for the same range agrees with the Console total within
         max(TOLERANCE_FLOOR_USD = 0.50, TOLERANCE_FRACTION = 5% of the Console figure) (confirmed 2026-10-24; not
         configurable). "The same range" is [range_start_utc 00:00, range_end_utc + 1 day 00:00) cut off at
         read_at_utc; the range must reach the day the Console was read. On disagreement: refuse and report both
         figures and which side is higher;
     (c) the cap headroom on the billed basis (cap - billed total - in-flight reservations) covers the projection.

WHICH LEDGERS THE CONSOLE FIGURE IS COMPARED WITH. The Console sees every call on the key: the main ledger
(results/cloud_ledger.jsonl) and the demo lane's (demo/ledger/cloud_ledger_demo.jsonl, gitignored). Each is a
STREAM; the files of a stream are looked for in every checkout of this repository (the main working tree and every
git worktree, found from the .git file or directory of the checkout that holds the job's ledger, by reading files,
no git command), and a stream's rows are the union of those files with identical rows counted once. So a stale
committed copy of the main ledger in a worktree adds nothing, and main plus demo spend is counted once whichever
checkout runs the job. The main stream is priced by analysis/cloud_spend_billed.py::compute with the main checkout's
job files; demo rows are taken as written (they are all billed rows).

Stub-mode clients are never gated. Nothing here makes a network call or reads a key.
"""
from __future__ import annotations

import importlib.util
import json
import re
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CONSOLE_CHECK_NAME = "results/console_usage_check.json"
FRESHNESS_HOURS_DEFAULT = 24.0
TOLERANCE_FLOOR_USD = 0.50
TOLERANCE_FRACTION = 0.05
CLOCK_SKEW_S = 300.0          # a read_at_utc up to 5 minutes in the future is accepted (clock skew), not more
ASK_FIRST_USD = 2.00          # operator 2026-10-24: hard refusal above USD 2 estimated per job without approval
STREAMS = {"main": "results/cloud_ledger.jsonl", "demo": "demo/ledger/cloud_ledger_demo.jsonl"}
REQUIRED_FIELDS = ("console_total_usd", "read_at_utc", "range_start_utc", "range_end_utc", "read_by")
APPROVAL_FLAG = "--operator-approval"
APPROVAL_RECORD = "operator_approval"
_USD_RE = re.compile(r"\bUSD\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

TEMPLATE = {
    "console_total_usd": 0.0,
    "read_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
    "range_start_utc": "YYYY-MM-DD",
    "range_end_utc": "YYYY-MM-DD",
    "read_by": "name of the person who read the Console",
    "outside_repo_usd": 0.0,
    "outside_repo_note": "",
}


class ConsoleCheckRefused(RuntimeError):
    """A live job may not start; str(e) holds the reason and what the operator must do."""


class JobApprovalRefused(ConsoleCheckRefused):
    """The job's estimate is over the per-job limit and no valid operator approval was given."""


# -- checkouts and ledger streams ------------------------------------------------------------------------------

def checkout_root(ledger_path: str | Path) -> Path | None:
    """The checkout holding a stream ledger (results/cloud_ledger.jsonl or demo/ledger/cloud_ledger_demo.jsonl), else
    None (a ledger anywhere else, e.g. a test's, stands alone)."""
    p = Path(ledger_path).resolve()
    for rel in STREAMS.values():
        n = len(Path(rel).parts)
        if tuple(x.lower() for x in p.parts[-n:]) == tuple(x.lower() for x in Path(rel).parts):
            return Path(*p.parts[:-n])
    return None


def checkouts(root: Path) -> list[Path]:
    """Every working tree of root's repository (root itself, the main working tree, each git worktree), from the
    .git directory or file by reading files only. A root that is not a git checkout is returned alone."""
    root = Path(root).resolve()
    g = root / ".git"
    common = None
    try:
        if g.is_dir():
            common = g
        elif g.is_file():
            gitdir = Path(g.read_text(encoding="utf-8").split("gitdir:", 1)[1].strip())
            gitdir = gitdir if gitdir.is_absolute() else (root / gitdir)
            cd = gitdir / "commondir"
            common = (gitdir / cd.read_text(encoding="utf-8").strip()).resolve() if cd.exists() else gitdir
    except (OSError, IndexError):
        common = None
    out = {root}
    if common is not None:
        out.add(common.resolve().parent)
        for f in (common / "worktrees").glob("*/gitdir"):
            try:
                out.add(Path(f.read_text(encoding="utf-8").strip()).resolve().parent)
            except OSError:
                continue
    return sorted(r for r in out if r.exists())


def main_checkout(root: Path) -> Path:
    """The main working tree of root's repository (the common .git directory's parent), else root."""
    root = Path(root).resolve()
    g = root / ".git"
    if g.is_file():
        try:
            gitdir = Path(g.read_text(encoding="utf-8").split("gitdir:", 1)[1].strip())
            gitdir = gitdir if gitdir.is_absolute() else (root / gitdir)
            cd = gitdir / "commondir"
            if cd.exists():
                return (gitdir / cd.read_text(encoding="utf-8").strip()).resolve().parent
        except (OSError, IndexError):
            pass
    return root


def default_check_path() -> Path:
    """results/console_usage_check.json in the main checkout, whichever checkout runs the job."""
    return main_checkout(REPO) / CONSOLE_CHECK_NAME


def _read_rows(path: Path) -> list[str]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(json.dumps(row, sort_keys=True))
    return out


def stream_files(ledger_path: str | Path) -> dict[str, list[Path]]:
    """{stream: [existing files]} across every checkout of the ledger's repository (just the ledger itself as the
    main stream when it is not a stream ledger)."""
    root = checkout_root(ledger_path)
    if root is None:
        lp = Path(ledger_path).resolve()
        return {"main": [lp] if lp.exists() else [], "demo": []}
    roots = checkouts(root)
    out = {}
    for name, rel in STREAMS.items():
        seen, files = set(), []
        for r in roots:
            f = (r / rel).resolve()
            if f.exists() and f not in seen:
                seen.add(f)
                files.append(f)
        out[name] = files
    return out


def _union(files: list[Path]) -> list[str]:
    seen, rows = set(), []
    for f in files:
        for r in _read_rows(f):
            if r not in seen:
                seen.add(r)
                rows.append(r)
    return rows


def _spend_module():
    spec = importlib.util.spec_from_file_location("cloud_spend_billed", REPO / "analysis" / "cloud_spend_billed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _empty(ledger: str) -> dict:
    return {"billed_usd": 0.0, "upper_bound_usd": 0.0, "upper_bound_only_usd": 0.0, "ledger_calls": 0,
            "ledger": ledger}


def ledger_spend(ledger_path: str | Path, start: datetime | None = None, end: datetime | None = None) -> dict:
    """Billed and upper-bound totals of ONE ledger file, exactly as register row cloud-spend-billed-vs-upper-bound
    computes them (analysis/cloud_spend_billed.py::compute), optionally only rows stamped in [start, end)."""
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return _empty(str(ledger_path))
    res = _spend_module().compute(ledger_path=ledger_path, start=start, end=end)
    return {"billed_usd": res["total_billed_usd"], "upper_bound_usd": res["total_upper_bound_usd"],
            "upper_bound_only_usd": res["upper_bound_only_usd"], "ledger_calls": res["ledger_calls"],
            "ledger": str(ledger_path)}


def _rows_spend(rows: list[str], job_roots: list[Path], start, end, label: str) -> dict:
    if not rows:
        return _empty(label)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "union.jsonl"
        p.write_text("\n".join(rows) + "\n", encoding="utf-8")
        res = _spend_module().compute(roots=job_roots, ledger_path=p, start=start, end=end)
    return {"billed_usd": res["total_billed_usd"], "upper_bound_usd": res["total_upper_bound_usd"],
            "upper_bound_only_usd": res["upper_bound_only_usd"], "ledger_calls": res["ledger_calls"],
            "ledger": label}


def console_side_spend(ledger_path: str | Path, start=None, end=None) -> dict:
    """Billed and upper-bound totals of every stream (each row once) over [start, end): what the Console sees."""
    files = stream_files(ledger_path)
    root = checkout_root(ledger_path)
    job_roots = [] if root is None else [main_checkout(root), root]
    per = {}
    for name, fl in files.items():
        label = "; ".join(str(f) for f in fl) or f"(no {name} ledger)"
        per[name] = _rows_spend(_union(fl), job_roots if name == "main" else [], start, end, label)
    return {"billed_usd": sum(x["billed_usd"] for x in per.values()),
            "upper_bound_usd": sum(x["upper_bound_usd"] for x in per.values()),
            "streams": per, "files": {k: [str(f) for f in v] for k, v in files.items()}}


def _first_call_date(ledger_path: Path) -> date | None:
    dates = []
    for fl in stream_files(ledger_path).values():
        for f in fl:
            for r in _read_rows(f):
                row = json.loads(r)
                if row.get("cost_usd") is not None and row.get("timestamp"):
                    dates.append(datetime.fromisoformat(row["timestamp"]).astimezone(timezone.utc).date())
                    break
    return min(dates) if dates else None


# -- freshness and tolerance -----------------------------------------------------------------------------------

def freshness_hours(value: float | None = None) -> float:
    """24 h, or a SHORTER caller-given window. A longer one would loosen the confirmed value and is refused."""
    if value is None:
        return FRESHNESS_HOURS_DEFAULT
    v = float(value)
    if not 0 < v <= FRESHNESS_HOURS_DEFAULT:
        raise ConsoleCheckRefused(f"refused (console check): the freshness window must be in "
                                  f"(0, {FRESHNESS_HOURS_DEFAULT:g}] h (operator 2026-10-24: 24 h confirmed; it can be "
                                  f"tightened, never loosened), got {v:g}")
    return v


def tolerance_usd(console_total_usd: float) -> float:
    return max(TOLERANCE_FLOOR_USD, TOLERANCE_FRACTION * abs(float(console_total_usd)))


# -- refusals --------------------------------------------------------------------------------------------------

def what_to_do(check_path: Path, ledger_path: Path, now: datetime, problem: str) -> str:
    first = _first_call_date(Path(ledger_path))
    t = dict(TEMPLATE, range_start_utc=first.isoformat() if first else "YYYY-MM-DD",
             range_end_utc=now.date().isoformat())
    return (
        f"WHAT TO DO ({problem}):\n"
        "  1. Open the Anthropic Console usage/cost page for the key this repo uses.\n"
        f"  2. Set the date range from {t['range_start_utc']} (the ledgers' first call) to {t['range_end_utc']} "
        "(today, UTC) and read the total cost in USD.\n"
        f"  3. Write {check_path} with exactly these fields (read_at_utc = when you read it, UTC):\n"
        + "\n".join("     " + ln for ln in json.dumps(t, indent=2).splitlines()) + "\n"
        "     outside_repo_usd: Console spend you know is not from this repo (other projects on the same key or "
        "org); say what it is in outside_repo_note. Leave 0.0 if none.\n"
        "  4. Re-run the job. Nothing was sent."
    )


def _refuse(reason: str, check_path: Path, ledger_path: Path, now: datetime, problem: str) -> None:
    raise ConsoleCheckRefused(f"refused (console check): {reason}\n{what_to_do(check_path, ledger_path, now, problem)}")


def _parse_ts(s: Any) -> datetime:
    t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("read_at_utc needs a UTC offset (end it with Z)")
    return t.astimezone(timezone.utc)


def load_console_check(check_path: Path, ledger_path: Path, now: datetime) -> dict:
    if not check_path.exists():
        _refuse(f"{check_path} does not exist; the Console total has not been recorded", check_path, ledger_path,
                now, "missing file")
    try:
        d = json.loads(check_path.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED_FIELDS if k not in d or d[k] in (None, "")]
        if missing:
            raise ValueError(f"missing field(s) {missing}")
        out = {"console_total_usd": float(d["console_total_usd"]), "read_at": _parse_ts(d["read_at_utc"]),
               "range_start": date.fromisoformat(str(d["range_start_utc"])),
               "range_end": date.fromisoformat(str(d["range_end_utc"])), "read_by": str(d["read_by"]),
               "outside_repo_usd": float(d.get("outside_repo_usd") or 0.0),
               "outside_repo_note": str(d.get("outside_repo_note") or "")}
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        _refuse(f"{check_path} is not valid: {e}", check_path, ledger_path, now, "invalid file")
    if out["read_by"].startswith("name of the person"):
        _refuse(f"{check_path} still holds the template's read_by", check_path, ledger_path, now, "template not filled")
    if out["outside_repo_usd"] and not out["outside_repo_note"].strip():
        _refuse("outside_repo_usd is set but outside_repo_note does not say what it is", check_path, ledger_path,
                now, "unexplained outside spend")
    if out["range_end"] < out["range_start"]:
        _refuse("range_end_utc is before range_start_utc", check_path, ledger_path, now, "invalid range")
    return out


# -- the Console check -----------------------------------------------------------------------------------------

def check(projection_billed_usd: float, *, cap_usd: float, ledger_path: str | Path,
          projection_upper_usd: float | None = None, reserved_usd: float = 0.0,
          check_path: str | Path | None = None, now: datetime | None = None,
          freshness_h: float | None = None) -> dict:
    """(a) fresh, (b) agrees within tolerance, (c) billed headroom covers the projection; else ConsoleCheckRefused.
    Returns the report (every figure the decision used) when the job may start."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    check_path = Path(check_path) if check_path is not None else default_check_path()
    ledger_path = Path(ledger_path)
    fresh_h = freshness_hours(freshness_h)
    c = load_console_check(check_path, ledger_path, now)

    age_h = (now - c["read_at"]).total_seconds() / 3600.0
    if age_h < -CLOCK_SKEW_S / 3600.0:
        _refuse(f"read_at_utc {c['read_at'].isoformat()} is in the future", check_path, ledger_path, now,
                "read time in the future")
    if age_h > fresh_h:
        _refuse(f"the Console was read {age_h:.1f} h ago ({c['read_at'].isoformat()} by {c['read_by']}), older than "
                f"the {fresh_h:g} h freshness window", check_path, ledger_path, now, "stale reading")
    if c["range_end"] < c["read_at"].date():
        _refuse(f"the Console range ends {c['range_end']} but it was read on {c['read_at'].date()}; spend after "
                f"{c['range_end']} would go unchecked", check_path, ledger_path, now, "range does not reach the read day")

    start = datetime.combine(c["range_start"], datetime.min.time(), timezone.utc)
    end = min(datetime.combine(c["range_end"] + timedelta(days=1), datetime.min.time(), timezone.utc),
              c["read_at"])
    side = console_side_spend(ledger_path, start, end)
    ledger_billed, ledger_upper = side["billed_usd"], side["upper_bound_usd"]
    console_repo = c["console_total_usd"] - c["outside_repo_usd"]
    diff = ledger_billed - console_repo
    tol = tolerance_usd(console_repo)
    first = _first_call_date(ledger_path)

    total = ledger_spend(ledger_path)
    headroom = cap_usd - total["billed_usd"] - reserved_usd
    headroom_upper = cap_usd - total["upper_bound_usd"] - reserved_usd
    rep = {
        "gated": True, "check_file": str(check_path), "read_by": c["read_by"], "read_at_utc": c["read_at"].isoformat(),
        "read_age_h": age_h, "freshness_h": fresh_h, "range_utc": [start.isoformat(), end.isoformat()],
        "range_starts_after_first_ledger_call": bool(first and c["range_start"] > first),
        "console_total_usd": c["console_total_usd"], "outside_repo_usd": c["outside_repo_usd"],
        "console_repo_usd": console_repo, "ledger_billed_in_range_usd": ledger_billed,
        "ledger_upper_bound_in_range_usd": ledger_upper, "ledger_files": side["files"],
        "ledger_billed_in_range_by_stream_usd": {k: v["billed_usd"] for k, v in side["streams"].items()},
        "diff_ledger_minus_console_usd": diff, "tolerance_usd": tol,
        "cap_usd": cap_usd, "cap_ledger": str(ledger_path), "spent_billed_usd": total["billed_usd"],
        "spent_upper_bound_usd": total["upper_bound_usd"], "reserved_usd": reserved_usd,
        "headroom_billed_usd": headroom, "headroom_upper_bound_usd": headroom_upper,
        "projection_billed_usd": projection_billed_usd, "projection_upper_bound_usd": projection_upper_usd,
    }
    if abs(diff) > tol + 1e-9:
        higher = "the Console is higher" if diff < 0 else "the ledgers are higher"
        why = ("spend the ledgers do not see (another project on the key, or calls made outside CloudClient); find "
               "it before any job, and if it is genuinely not this repo's, record it in outside_repo_usd with a note"
               if diff < 0 else
               "the Console may lag (wait an hour and re-read), or the ledgers over-count (rows without usage are "
               "counted at the upper bound)")
        _refuse(f"billed ledgers and Console disagree: ledgers billed ${ledger_billed:.4f} (upper bound "
                f"${ledger_upper:.4f}) vs Console ${console_repo:.4f}"
                + (f" (${c['console_total_usd']:.4f} less ${c['outside_repo_usd']:.4f} outside this repo)"
                   if c["outside_repo_usd"] else "")
                + f" for {start.date()} to {end.isoformat()}; {higher}, by ${abs(diff):.4f}, outside the tolerance "
                f"${tol:.4f} = max(${TOLERANCE_FLOOR_USD:.2f}, {TOLERANCE_FRACTION:.0%} of the Console figure); "
                f"likely cause: {why}", check_path, ledger_path, now, "ledgers and Console disagree")
    if projection_billed_usd > headroom + 1e-9:
        raise ConsoleCheckRefused(
            f"refused (console check): projected ${projection_billed_usd:.4f}"
            + (f" (upper bound ${projection_upper_usd:.4f})" if projection_upper_usd is not None else "")
            + f" exceeds the billed headroom ${headroom:.4f} under the ${cap_usd:.2f} cap (spent billed "
            f"${total['billed_usd']:.4f}, upper bound ${total['upper_bound_usd']:.4f}; upper-bound headroom "
            f"${headroom_upper:.4f}).\nWHAT TO DO (cap headroom): shrink the job (--limit) or ask the operator to "
            "raise this job's cap (never above the hard cap). Nothing was sent.")
    return rep


def require_for_client(client: Any, projection_billed_usd: float, projection_upper_usd: float | None = None,
                       **kw: Any) -> dict:
    """The Console check for a CloudClient job entry point. Stub mode is never gated."""
    if getattr(client, "stub_mode", True):
        return {"gated": False, "reason": "stub mode is never gated"}
    return check(projection_billed_usd, cap_usd=client.spend_cap_usd, ledger_path=client.ledger_path,
                 projection_upper_usd=projection_upper_usd, reserved_usd=getattr(client, "_reserved_usd", 0.0), **kw)


# -- per-job limit and operator approval -----------------------------------------------------------------------

def approval_syntax(job_id: str, amount_usd: float) -> str:
    return f'{APPROVAL_FLAG} "approve {job_id} up to USD {amount_usd:.2f}: <reason>"'


def validate_approval(approval: str | None, *, job_id: str, estimate_usd: float, estimate_basis: str,
                      limit_usd: float = ASK_FIRST_USD) -> dict | None:
    """None when the estimate is within the limit and no approval was given; the parsed approval when one was given
    and is valid; JobApprovalRefused otherwise. A valid approval names job_id and exactly one amount "USD <n>" with
    n >= the estimate. Given when not needed, it is still checked (and recorded), never ignored."""
    over = estimate_usd > limit_usd + 1e-9
    suggest = max(estimate_usd, limit_usd)
    if not approval:
        if not over:
            return None
        raise JobApprovalRefused(
            f"refused (per-job limit): job {job_id} is estimated at ${estimate_usd:.4f} ({estimate_basis}), above the "
            f"USD {limit_usd:.2f} per-job limit (operator rule 2026-10-24: hard refusal above it without an explicit "
            f"per-job approval).\nWHAT TO DO: only the operator approves. To run this one job, add to its command "
            f"line:\n  {approval_syntax(job_id, suggest)}\nThe approval is recorded in the ledger before the first "
            "call; it does not raise any cap or skip the Console check. Nothing was sent.")
    text = str(approval).strip()
    problems = []
    if job_id not in text:
        problems.append(f"it does not name the job id '{job_id}'")
    amounts = sorted({float(x) for x in _USD_RE.findall(text)})
    if len(amounts) != 1:
        problems.append(f"it must state exactly one amount as 'USD <n>' (found {len(amounts)})")
    elif amounts[0] + 1e-9 < estimate_usd:
        problems.append(f"its amount USD {amounts[0]:.2f} is below the estimate ${estimate_usd:.4f}")
    if problems:
        raise JobApprovalRefused(
            f"refused (per-job limit): the {APPROVAL_FLAG} text for job {job_id} is not a valid approval: "
            + "; ".join(problems) + f".\nWHAT TO DO: the operator re-runs with\n  {approval_syntax(job_id, suggest)}\n"
            "Nothing was sent.")
    return {"approval_text": text, "approved_usd": amounts[0], "job_id": job_id, "estimate_usd": estimate_usd,
            "estimate_basis": estimate_basis, "limit_usd": limit_usd, "over_limit": over}


def git_head() -> str:
    """HEAD of the checkout this code runs from (hidden child process, harness/proc_util.py), or 'unknown'."""
    try:
        spec = importlib.util.spec_from_file_location("proc_util", REPO / "harness" / "proc_util.py")
        pu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pu)
        out = pu.check_output_hidden(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, timeout=10)
        return out.strip() or "unknown"
    except Exception:  # noqa: BLE001 - the record says unknown rather than blocking on git
        return "unknown"


def write_approval_record(ledger_path: str | Path, appr: dict, *, head: str, now: datetime) -> dict:
    """Append the approval to the job's ledger as its own row (no cost_usd: no spend total counts it)."""
    ts = now.astimezone(timezone.utc).isoformat()
    row = {"record": APPROVAL_RECORD, "source": "console_gate", **appr, "git_head": head, "ts_utc": ts,
           "timestamp": ts}
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def pre_job(client: Any, *, job_id: str, estimate_usd: float, estimate_basis: str, projection_billed_usd: float,
            projection_upper_usd: float | None = None, approval: str | None = None, limit_usd: float = ASK_FIRST_USD,
            freshness_h: float | None = None, check_path: str | Path | None = None, now: datetime | None = None,
            head: str | None = None) -> dict:
    """The whole pre-job gate, before a live job's first call: per-job limit / approval, Console check, then the
    approval record. Stub mode is never gated. Raises ConsoleCheckRefused (JobApprovalRefused for the limit)."""
    if getattr(client, "stub_mode", True):
        return {"gated": False, "reason": "stub mode is never gated"}
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    appr = validate_approval(approval, job_id=job_id, estimate_usd=estimate_usd, estimate_basis=estimate_basis,
                             limit_usd=limit_usd)
    rep = require_for_client(client, projection_billed_usd, projection_upper_usd, check_path=check_path, now=now,
                             freshness_h=freshness_h)
    rep.update({"job_id": job_id, "estimate_usd": estimate_usd, "estimate_basis": estimate_basis,
                "limit_usd": limit_usd, "approval": None})
    if appr is not None:
        rep["approval"] = write_approval_record(client.ledger_path, appr, head=head or git_head(), now=now)
    return rep


def format_report(rep: dict) -> str:
    if not rep.get("gated"):
        return f"console check: not gated ({rep.get('reason')})"
    lines = [
        f"console check passed: Console ${rep['console_total_usd']:.4f} read {rep['read_age_h']:.1f} h ago by "
        f"{rep['read_by']} (window {rep['freshness_h']:g} h)"
        + (f", less ${rep['outside_repo_usd']:.4f} outside this repo" if rep["outside_repo_usd"] else ""),
        f"  ledgers billed ${rep['ledger_billed_in_range_usd']:.4f} (upper bound "
        f"${rep['ledger_upper_bound_in_range_usd']:.4f}) for {rep['range_utc'][0]} to {rep['range_utc'][1]}; "
        f"diff ${rep['diff_ledger_minus_console_usd']:+.4f}, tolerance ${rep['tolerance_usd']:.4f}",
        f"  cap ${rep['cap_usd']:.2f}: spent billed ${rep['spent_billed_usd']:.4f} (upper bound "
        f"${rep['spent_upper_bound_usd']:.4f}); headroom billed ${rep['headroom_billed_usd']:.4f} (upper bound "
        f"${rep['headroom_upper_bound_usd']:.4f})",
        f"  projection billed ${rep['projection_billed_usd']:.4f}"
        + (f" (upper bound ${rep['projection_upper_bound_usd']:.4f})"
           if rep["projection_upper_bound_usd"] is not None else ""),
    ]
    if "estimate_usd" in rep:
        a = rep.get("approval")
        lines.append(f"  job {rep['job_id']}: estimate ${rep['estimate_usd']:.4f} ({rep['estimate_basis']}), limit "
                     f"USD {rep['limit_usd']:.2f}; "
                     + (f"operator approval recorded in the ledger: USD {a['approved_usd']:.2f}, \"{a['approval_text']}\""
                        if a else "no approval needed"))
    if rep["range_starts_after_first_ledger_call"]:
        lines.append("  note: the Console range starts after the ledgers' first call, so earlier spend is not "
                     "verified by this reading")
    return "\n".join(lines)
