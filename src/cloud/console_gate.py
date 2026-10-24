"""Console check before each live cloud job (operator 2026-10-23, register row cloud-spend-billed-vs-upper-bound).

The cap is on BILLED spend (src/cloud/client.py, BILLED CAP BASIS AT START). Billed spend is reconstructed from the
ledger and the cloud jobs' result files, so before a live job spends anything it is checked against the one figure
the provider itself reports: the Anthropic Console usage total. Code cannot read the Console; the operator reads it
and writes results/console_usage_check.json (TEMPLATE below). check() refuses the job unless all of:

  (a) the file exists, parses, and was read within the freshness window (FRESHNESS_HOURS_DEFAULT = 24 h, override
      with freshness_hours=... or the APU_CONSOLE_FRESHNESS_HOURS environment variable);
  (b) the ledger's billed total for the same range agrees with the Console total within the tolerance,
      max(TOLERANCE_FLOOR_USD = 0.50, TOLERANCE_FRACTION = 5% of the Console total). "The same range" is
      [range_start_utc 00:00, range_end_utc + 1 day 00:00) cut off at read_at_utc, since calls after the Console
      was read cannot be in its total; the range must reach the day the Console was read. The ledgers compared are
      the cap's ledger plus any other ledger of this repo that exists (EXTRA_LEDGERS: the demo lane's), because the
      Console sees every call on the key. Console spend the operator knows is not from this repo goes in
      outside_repo_usd (with a note) and is subtracted before the comparison;
  (c) the cap headroom on the billed basis (cap - billed total - in-flight reservations) covers the job's projection.

On a refusal ConsoleCheckRefused carries the reason and exactly what the operator must do. Stub-mode clients are
never gated (require_for_client returns at once), and nothing here is called from tests except with fake files.
Nothing here makes a network call or reads a key.
"""
from __future__ import annotations

import importlib.util
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CONSOLE_CHECK_PATH = REPO / "results" / "console_usage_check.json"
FRESHNESS_HOURS_DEFAULT = 24.0
FRESHNESS_ENV = "APU_CONSOLE_FRESHNESS_HOURS"
TOLERANCE_FLOOR_USD = 0.50
TOLERANCE_FRACTION = 0.05
CLOCK_SKEW_S = 300.0          # a read_at_utc up to 5 minutes in the future is accepted (clock skew), not more
ASK_FIRST_USD = 2.00          # operator rule: stop and ask before anything over USD 2 (printed, not enforced here)
EXTRA_LEDGERS = ("demo/ledger/cloud_ledger_demo.jsonl",)
REQUIRED_FIELDS = ("console_total_usd", "read_at_utc", "range_start_utc", "range_end_utc", "read_by")

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


def _spend_module():
    spec = importlib.util.spec_from_file_location("cloud_spend_billed", REPO / "analysis" / "cloud_spend_billed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ledger_spend(ledger_path: str | Path, start: datetime | None = None, end: datetime | None = None) -> dict:
    """Billed and upper-bound totals of one ledger, exactly as register row cloud-spend-billed-vs-upper-bound
    computes them (analysis/cloud_spend_billed.py::compute), optionally only rows stamped in [start, end)."""
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return {"billed_usd": 0.0, "upper_bound_usd": 0.0, "upper_bound_only_usd": 0.0, "ledger_calls": 0,
                "ledger": str(ledger_path)}
    res = _spend_module().compute(ledger_path=ledger_path, start=start, end=end)
    return {"billed_usd": res["total_billed_usd"], "upper_bound_usd": res["total_upper_bound_usd"],
            "upper_bound_only_usd": res["upper_bound_only_usd"], "ledger_calls": res["ledger_calls"],
            "ledger": str(ledger_path)}


def _first_call_date(ledger_path: Path) -> date | None:
    if not ledger_path.exists():
        return None
    with ledger_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("record") != "alert" and row.get("timestamp"):
                return datetime.fromisoformat(row["timestamp"]).astimezone(timezone.utc).date()
    return None


def freshness_hours(value: float | None = None) -> float:
    if value is not None:
        return float(value)
    env = os.environ.get(FRESHNESS_ENV)
    return float(env) if env else FRESHNESS_HOURS_DEFAULT


def tolerance_usd(console_total_usd: float) -> float:
    return max(TOLERANCE_FLOOR_USD, TOLERANCE_FRACTION * abs(float(console_total_usd)))


def compare_ledgers(ledger_path: str | Path) -> list[Path]:
    """The cap's ledger plus every EXTRA_LEDGERS file that exists in the same repo (the ledger's results/ parent)."""
    ledger_path = Path(ledger_path).resolve()
    root = ledger_path.parent.parent if ledger_path.parent.name == "results" else None
    out = [ledger_path]
    if root is not None:
        out += [root / x for x in EXTRA_LEDGERS if (root / x).exists() and (root / x).resolve() != ledger_path]
    return out


def what_to_do(check_path: Path, ledger_path: Path, now: datetime, problem: str) -> str:
    first = _first_call_date(Path(ledger_path))
    t = dict(TEMPLATE, range_start_utc=first.isoformat() if first else "YYYY-MM-DD",
             range_end_utc=now.date().isoformat())
    return (
        f"WHAT TO DO ({problem}):\n"
        "  1. Open the Anthropic Console usage/cost page for the key this repo uses.\n"
        f"  2. Set the date range from {t['range_start_utc']} (the ledger's first call) to {t['range_end_utc']} "
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


def check(projection_billed_usd: float, *, cap_usd: float, ledger_path: str | Path,
          projection_upper_usd: float | None = None, reserved_usd: float = 0.0,
          check_path: str | Path = CONSOLE_CHECK_PATH, now: datetime | None = None,
          freshness_h: float | None = None, ledgers: list[Path] | None = None) -> dict:
    """(a) fresh, (b) agrees within tolerance, (c) billed headroom covers the projection; else ConsoleCheckRefused.
    Returns the report (every figure the decision used) when the job may start."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    check_path, ledger_path = Path(check_path), Path(ledger_path)
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
    ledgers = ledgers if ledgers is not None else compare_ledgers(ledger_path)
    per = [ledger_spend(p, start, end) for p in ledgers]
    ledger_billed = sum(x["billed_usd"] for x in per)
    ledger_upper = sum(x["upper_bound_usd"] for x in per)
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
        "ledger_upper_bound_in_range_usd": ledger_upper, "ledgers": [str(p) for p in ledgers],
        "diff_ledger_minus_console_usd": diff, "tolerance_usd": tol,
        "cap_usd": cap_usd, "spent_billed_usd": total["billed_usd"], "spent_upper_bound_usd": total["upper_bound_usd"],
        "reserved_usd": reserved_usd, "headroom_billed_usd": headroom, "headroom_upper_bound_usd": headroom_upper,
        "projection_billed_usd": projection_billed_usd, "projection_upper_bound_usd": projection_upper_usd,
        "over_ask_first_usd": projection_billed_usd > ASK_FIRST_USD,
    }
    if abs(diff) > tol + 1e-9:
        side = ("the ledger is LOWER than the Console: spend the ledger does not see (another project on the key, "
                "or calls made outside CloudClient); find it before any job, and if it is genuinely not this repo's, "
                "record it in outside_repo_usd with a note" if diff < 0 else
                "the ledger is HIGHER than the Console: the Console may lag (wait an hour and re-read), or the "
                "ledger over-counts (rows without usage are counted at the upper bound)")
        _refuse(f"ledger billed ${ledger_billed:.4f} (upper bound ${ledger_upper:.4f}) for {start.date()} to "
                f"{end.isoformat()} vs Console ${console_repo:.4f}"
                + (f" (${c['console_total_usd']:.4f} less ${c['outside_repo_usd']:.4f} outside this repo)"
                   if c["outside_repo_usd"] else "")
                + f": differ by ${diff:+.4f}, outside the tolerance ${tol:.4f} = max(${TOLERANCE_FLOOR_USD:.2f}, "
                f"{TOLERANCE_FRACTION:.0%} of the Console figure); {side}", check_path, ledger_path, now,
                "ledger and Console disagree")
    if projection_billed_usd > headroom + 1e-9:
        raise ConsoleCheckRefused(
            f"refused (console check): projected ${projection_billed_usd:.4f}"
            + (f" (upper bound ${projection_upper_usd:.4f})" if projection_upper_usd is not None else "")
            + f" exceeds the billed headroom ${headroom:.4f} under the ${cap_usd:.2f} cap (spent billed "
            f"${total['billed_usd']:.4f}, upper bound ${total['upper_bound_usd']:.4f}; upper-bound headroom "
            f"${headroom_upper:.4f}).\nWHAT TO DO (cap headroom): shrink the job (--limit) or ask the operator to "
            "raise this job's cap (never above the USD 50 hard cap). Nothing was sent.")
    return rep


def require_for_client(client: Any, projection_billed_usd: float, projection_upper_usd: float | None = None,
                       **kw: Any) -> dict:
    """The pre-job gate for a CloudClient job entry point. Stub mode is never gated."""
    if getattr(client, "stub_mode", True):
        return {"gated": False, "reason": "stub mode is never gated"}
    return check(projection_billed_usd, cap_usd=client.spend_cap_usd, ledger_path=client.ledger_path,
                 projection_upper_usd=projection_upper_usd, reserved_usd=getattr(client, "_reserved_usd", 0.0), **kw)


def format_report(rep: dict) -> str:
    if not rep.get("gated"):
        return f"console check: not gated ({rep.get('reason')})"
    lines = [
        f"console check passed: Console ${rep['console_total_usd']:.4f} read {rep['read_age_h']:.1f} h ago by "
        f"{rep['read_by']} (window {rep['freshness_h']:g} h)"
        + (f", less ${rep['outside_repo_usd']:.4f} outside this repo" if rep["outside_repo_usd"] else ""),
        f"  ledger billed ${rep['ledger_billed_in_range_usd']:.4f} (upper bound "
        f"${rep['ledger_upper_bound_in_range_usd']:.4f}) for {rep['range_utc'][0]} to {rep['range_utc'][1]}; "
        f"diff ${rep['diff_ledger_minus_console_usd']:+.4f}, tolerance ${rep['tolerance_usd']:.4f}",
        f"  cap ${rep['cap_usd']:.2f}: spent billed ${rep['spent_billed_usd']:.4f} (upper bound "
        f"${rep['spent_upper_bound_usd']:.4f}); headroom billed ${rep['headroom_billed_usd']:.4f} (upper bound "
        f"${rep['headroom_upper_bound_usd']:.4f})",
        f"  projection billed ${rep['projection_billed_usd']:.4f}"
        + (f" (upper bound ${rep['projection_upper_bound_usd']:.4f})"
           if rep["projection_upper_bound_usd"] is not None else ""),
    ]
    if rep["range_starts_after_first_ledger_call"]:
        lines.append("  note: the Console range starts after the ledger's first call, so earlier spend is not "
                     "verified by this reading")
    if rep["over_ask_first_usd"]:
        lines.append(f"  note: the projection is over USD {ASK_FIRST_USD:.2f}; the operator's rule is to stop and ask "
                     "before running it")
    return "\n".join(lines)
