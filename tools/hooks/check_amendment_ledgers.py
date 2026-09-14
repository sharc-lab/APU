"""Pre-commit hook: fail when AMENDMENTS.md and Blueprint §14 diverge on an ID.

The dual-definition episode (AMENDMENTS AM-025/AM-027 vs Blueprint §14 AM-025/AM-027)
happened because nothing compared the two ledgers. This hook does.

Living definition rule
----------------------
A tombstone is not a living definition. An AMENDMENTS.md heading whose title contains
``TOMBSTONE`` (case-insensitive), or whose immediately following status line marks
``RETIRED``, does not compete with Blueprint §14. Identifiers are never reused; they
are retired. Only living definitions are compared.

Parsers
-------
- AMENDMENTS.md: ``## AM-NNN — Title`` section headings (index table is ignored).
- Blueprint §14: amendment-log table rows whose Change cell opens with
  ``**AM-NNN — …**`` (bold identifier + em/en dash + one-line summary).

Exit 0 = no divergent dual living definitions, 1 = collision.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AMENDMENTS = REPO_ROOT / "AMENDMENTS.md"
BLUEPRINT = REPO_ROOT / "docs" / "SEAM_research_blueprint.md"

# Em dash / en dash / hyphen-minus (unicode escapes: ruff RUF001).
_DASH_CLASS = "[\u2014\u2013-]"
HEADING_RE = re.compile(
    rf"^#{{2,3}}\s+(AM-\d+)\s*{_DASH_CLASS}\s*(.+?)\s*$",
)
STATUS_LINE_RE = re.compile(
    r"\*\*Status:\*\*\s*([^*\n]+)",
    re.IGNORECASE,
)
BLUEPRINT_AM_RE = re.compile(
    rf"\*\*(AM-\d+)\s*{_DASH_CLASS}\s*(.+?)\*\*",
)


def _normalize_summary(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text)
    return " ".join(text.split())


def parse_amendments(path: Path) -> dict[str, dict[str, str]]:
    """Return living AMENDMENTS definitions: id -> {title, summary, status}."""
    if not path.is_file():
        return {}

    living: dict[str, dict[str, str]] = {}
    current_id: str | None = None
    current_title = ""
    current_status: str | None = None
    retired = False

    def flush() -> None:
        nonlocal current_id, current_title, current_status, retired
        if current_id and not retired:
            living[current_id] = {
                "title": current_title.strip(),
                "summary": current_title.strip(),
                "status": (current_status or "").strip(),
            }
        current_id = None
        current_title = ""
        current_status = None
        retired = False

    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            current_id = match.group(1)
            current_title = match.group(2)
            if "tombstone" in current_title.lower():
                retired = True
            continue
        if current_id is None:
            continue
        status_match = STATUS_LINE_RE.search(line)
        if status_match and current_status is None:
            current_status = status_match.group(1).strip()
            status_upper = current_status.upper()
            if "RETIRED" in status_upper or "TOMBSTONE" in status_upper:
                retired = True

    flush()
    return living


def parse_blueprint_section14(path: Path) -> dict[str, dict[str, str]]:
    """Return Blueprint §14 amendment-log definitions."""
    if not path.is_file():
        return {}

    text = path.read_text(encoding="utf-8")
    # Prefer the amendment-log section; fall back to whole file if heading moves.
    section_match = re.search(
        r"^##\s+14\.\s+Amendment log\s*$",
        text,
        re.MULTILINE,
    )
    if section_match:
        start = section_match.start()
        next_h2 = re.search(r"^##\s+", text[start + 1 :], re.MULTILINE)
        section = text[start : start + 1 + next_h2.start()] if next_h2 else text[start:]
    else:
        section = text

    living: dict[str, dict[str, str]] = {}
    status_only = re.compile(
        r"^(RESOLVED|OPEN|DEFERRED|WITHDRAWN|RETIRED|N/?A)\b",
        re.IGNORECASE,
    )
    for match in BLUEPRINT_AM_RE.finditer(section):
        am_id = match.group(1)
        summary = match.group(2).strip()
        # Ignore bold status tags like **AM-004 — RESOLVED.** that appear in
        # prose outside the amendment-log definition row.
        if status_only.match(summary):
            continue
        if len(summary) < 24:
            continue
        # Keep the first bold AM-NNN definition in §14 for each id.
        if am_id not in living:
            living[am_id] = {"title": summary, "summary": summary, "status": "RESOLVED"}
    return living


def _summaries_diverge(a: str, b: str) -> bool:
    na, nb = _normalize_summary(a), _normalize_summary(b)
    if not na or not nb:
        return True
    # Exact normalized match is agreement.
    if na == nb:
        return False
    # Token overlap: treat as divergent when the shorter summary's tokens are
    # not substantially covered by the longer (guards paraphrase vs collision).
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return True
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    coverage = len(shorter & longer) / len(shorter)
    return coverage < 0.6


def main() -> int:
    amendments = parse_amendments(AMENDMENTS)
    blueprint = parse_blueprint_section14(BLUEPRINT)

    if not AMENDMENTS.is_file():
        sys.stderr.write("note: AMENDMENTS.md absent — ledger cross-check skipped.\n")
        return 0
    if not BLUEPRINT.is_file():
        sys.stderr.write(
            "note: docs/SEAM_research_blueprint.md absent — ledger cross-check skipped.\n"
        )
        return 0

    shared = sorted(set(amendments) & set(blueprint))
    collisions: list[tuple[str, str, str]] = []
    for am_id in shared:
        a_sum = amendments[am_id]["summary"]
        b_sum = blueprint[am_id]["summary"]
        if _summaries_diverge(a_sum, b_sum):
            collisions.append((am_id, a_sum, b_sum))

    if collisions:
        sys.stderr.write(
            "\nBLOCKED — amendment identifier defined in both ledgers with divergent content:\n"
        )
        for am_id, a_sum, b_sum in collisions:
            sys.stderr.write(f"  {am_id}\n")
            sys.stderr.write(f"    AMENDMENTS.md : {a_sum}\n")
            sys.stderr.write(f"    Blueprint §14 : {b_sum}\n")
        sys.stderr.write(
            "\nIdentifiers are never reused. Tombstone the AMENDMENTS.md entry and\n"
            "reissue above high-water, or reconcile content so both living definitions\n"
            "agree. A TOMBSTONE heading / RETIRED status is not a living definition.\n\n"
        )
        return 1

    sys.stderr.write(
        f"amendment ledgers consistent "
        f"(AMENDMENTS living={len(amendments)}, Blueprint §14={len(blueprint)}, "
        f"shared_ok={len(shared)})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
