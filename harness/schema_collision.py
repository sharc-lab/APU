"""Schema collision experiment.

Hypothesis: the art_02/F-TYPED interference effect is explained by schema match,
not value type. Only art_02's F-TYPED filler replicates the artifact's record
structure (CALIBRATION RECORD with the same field names). Test by building a
third filler condition — F-SCHEMA — that uses the artifact's exact record format
with different entity identifiers for art_01, art_06, and art_07.

Design note: art_01 and art_06 F-TYPED fillers are ALREADY schema-matched
(same CONFIG JSON keys for art_01; same Engineering Review session format for
art_06). The key discriminating case is art_07: F-TYPED uses other ORM names
(Granite, Quartz, etc.); F-SCHEMA uses Ferrite ORM (same ORM as artifact) with
different version numbers and different CVEs. If F-SCHEMA induces failure where
F-TYPED did not, same-namespace records cause the confusion.

Design:
  Probes:  art_01, art_06, art_07 (art_02 already studied)
  Filler:  F-NUM, F-TYPED (reusing existing builders, seed=42), F-SCHEMA (new)
  Ratio:   1.20 (artifact fully present, af=1.00)
  Models:  qwen3:4b-instruct, llama3.1:8b, gpt-oss:120b-cloud
  Reps:    5
  Total:   3 x 3 x 3 x 5 = 135 calls

For wrong answers, the harness reports which filler field the value came from.

Results written incrementally to results/schema_collision.jsonl (one row per
line, flushed after each call). Supports --resume to skip already-completed
cells when restarting after interruption.

Usage:
  python -u harness/schema_collision.py
  python -u harness/schema_collision.py --resume
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).parent.parent
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS = REPO / "results"
OUTFILE = RESULTS / "schema_collision.jsonl"
sys.path.insert(0, str(REPO / "harness"))

import context as ctx_mod

TARGET_PROBES = ["art_01", "art_06", "art_07"]
MODELS = ["qwen3:4b-instruct", "llama3.1:8b", "gpt-oss:120b-cloud"]
COUNT_MODEL = "qwen3:4b-instruct"
HOST = "http://localhost:11434"
FILLER_TOKENS = 4000
BUDGET_RATIO = 1.20
N_REPS = 5
MAX_TOKENS = 256
MIN_PREDICT_120B = 1024

# Chars-per-token estimate calibrated from smoke tests (matches context.py).
_CHARS_PER_TOKEN = 5.03
# Maximum entries to accumulate before accepting the corpus as-is.
_FILLER_BUILD_CAP = 600

# ---------------------------------------------------------------------------
# Watchdog: abort if no model call starts within 5 min of process start.
# Set _first_call_started before issuing the first HTTP request.
# ---------------------------------------------------------------------------
_first_call_started = threading.Event()


def _start_watchdog(timeout: float = 300.0) -> None:
    def _watchdog():
        if not _first_call_started.wait(timeout=timeout):
            print(
                f"\n[WATCHDOG] No model call started within {timeout:.0f}s of process start.",
                flush=True,
            )
            print(
                "[WATCHDOG] Process is likely stuck in filler building. Aborting.",
                flush=True,
            )
            import os
            os._exit(2)

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Filler trim helper — capped at max_iter count_fn calls, always returns.
# ---------------------------------------------------------------------------

def _trim_with_cap(
    raw: str,
    target_tokens: int,
    count_fn,
    label: str = "filler",
    max_iter: int = 8,
) -> tuple[str, int, bool]:
    """Trim raw to ~target_tokens.

    Returns (text, realized_tok, converged).
    Logs every iteration. Accepts the best result when max_iter is exhausted.
    """
    target_chars = min(int(target_tokens * _CHARS_PER_TOKEN), len(raw))
    text = raw[:target_chars].strip()
    if count_fn is None:
        return text, target_tokens, True

    best = text
    best_err = float("inf")
    best_tok = target_tokens

    for i in range(max_iter):
        actual = count_fn(text)
        err = abs(actual - target_tokens) / max(target_tokens, 1)
        is_best = err < best_err
        if is_best:
            best, best_err, best_tok = text, err, actual
        print(f"  [{label} trim {i+1}/{max_iter}] {actual} tok  err={err:.1%}{' *' if is_best else ''}", flush=True)
        if err <= 0.02:
            return text, actual, True
        new_chars = min(int(len(text) * target_tokens / actual), len(raw))
        if new_chars == len(text):
            break
        text = raw[:new_chars].strip()

    converged = best_err <= 0.02
    if not converged:
        print(
            f"  [{label}] accepting best: {best_tok} tok (err={best_err:.1%}, not converged in {max_iter} iters)",
            flush=True,
        )
    return best, best_tok, converged


# ---------------------------------------------------------------------------
# F-TYPED builders (same schema as interference_r120.py, seed=42)
# ---------------------------------------------------------------------------

def _build_port_filler(target_tokens, seed, count_fn):
    """F-TYPED for art_01: same CONFIG JSON schema, different service names."""
    rng = random.Random(seed)
    EXCLUDE = {51847}
    services = [
        "billing-relay", "audit-logger", "rate-limiter", "metrics-sink",
        "event-bus", "config-svc", "health-proxy", "session-store",
        "telemetry-api", "webhook-router", "data-pipeline", "auth-gateway",
        "cache-proxy", "log-aggregator", "file-upload", "search-api",
    ]
    schemes = ["HMAC-SHA3", "Bearer", "mTLS", "API-Key", "OIDC", "HMAC-SHA256"]
    liftable = []
    other_fields = {"auth_schemes": set(), "max_payload_kb": set()}

    def make_entry():
        port = rng.randint(10240, 65535)
        while port in EXCLUDE:
            port = rng.randint(10240, 65535)
        EXCLUDE.add(port)
        liftable.append(str(port))
        svc = rng.choice(services)
        ver = f"{rng.randint(1,5)}.{rng.randint(0,12)}"
        kb = rng.choice([64, 128, 192, 256, 512])
        scheme = rng.choice(schemes)
        other_fields["auth_schemes"].add(scheme)
        other_fields["max_payload_kb"].add(str(kb))
        return (
            f"[CONFIG: service={svc} v{ver}]\n"
            f"{{\n"
            f'  "listen_port": {port},\n'
            f'  "max_payload_kb": {kb},\n'
            f'  "auth_scheme": "{scheme}",\n'
            f'  "flush_interval_ms": {rng.randint(500,5000)},\n'
            f'  "retry_backoff_base_ms": {rng.randint(100,2000)},\n'
            f'  "circuit_breaker_threshold": {rng.randint(4,30)}\n'
            f"}}"
        )

    needed_chars = int(target_tokens * _CHARS_PER_TOKEN * 1.5)
    entries: list[str] = []
    total_chars = 0
    while total_chars < needed_chars:
        e = make_entry()
        entries.append(e)
        total_chars += len(e) + 2
        if len(entries) >= _FILLER_BUILD_CAP:
            print(f"  [art_01/F-TYPED] capped at {_FILLER_BUILD_CAP} entries", flush=True)
            break

    print(f"  [art_01/F-TYPED] {len(entries)} entries, {total_chars} chars, trimming...", flush=True)
    raw = "\n\n".join(entries)
    text, tok, converged = _trim_with_cap(raw, target_tokens, count_fn, label="art_01/F-TYPED")
    return text, liftable, {k: list(v) for k, v in other_fields.items()}, tok, converged


def _build_surname_filler(target_tokens, seed, count_fn):
    """F-TYPED for art_06: Engineering Review session format, different people."""
    rng = random.Random(seed)
    EXCLUDE = {"Blum", "blum"}
    firstnames = ["R.", "T.", "P.", "M.", "J.", "S.", "K.", "A.", "C.", "D.",
                  "H.", "F.", "G.", "L.", "N.", "E.", "W.", "I.", "O.", "B."]
    surnames = [
        "Okafor", "Lindström", "Watanabe", "Ferreira", "Nkosi", "Patel",
        "Kowalski", "Bergström", "Nakamura", "Herrera", "Abramov", "Fischer",
        "Tremblay", "Yamamoto", "Oyelaran", "Kovacs", "Santana", "Eriksen",
        "Mehrotra", "Castillo", "Svensson", "Mwangi", "Petrov", "Reyes",
        "Thornton", "Nakagawa", "Guerrero", "Andersen", "Hashimoto", "Moreau",
    ]
    items = [
        "Migration to IPv6 for internal services",
        "Deployment freeze policy for Q4 releases",
        "On-call rotation extension proposal",
        "Incident post-mortem review process update",
        "New API gateway vendor evaluation",
        "Database replication lag threshold revision",
        "SLA targets for edge node latency",
        "Capacity planning for peak season traffic",
        "Security audit findings — network layer",
        "Load balancer failover procedure amendment",
    ]
    liftable = []
    other_fields = {"movers": set(), "chairs": set()}

    def make_entry():
        session_surnames = [s for s in rng.sample(surnames, 5) if s not in EXCLUDE][:5]
        if len(session_surnames) < 5:
            session_surnames.append("Ivanova")
        chair = session_surnames[0]
        mover = session_surnames[rng.randint(1, 4)]
        seconder = session_surnames[rng.randint(1, 4)]
        if seconder == mover:
            seconder = session_surnames[1] if mover != session_surnames[1] else session_surnames[2]
        liftable.append(seconder)
        liftable.append(mover)
        other_fields["movers"].add(mover)
        other_fields["chairs"].add(chair)
        item = rng.choice(items)
        yes = rng.randint(2, 4)
        no = 5 - yes - rng.randint(0, 1)
        abst = 5 - yes - no
        outcome = "PASSED" if yes > no else "FAILED"
        date = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        chair_str = f"{rng.choice(firstnames)} {chair} (chair)"
        attendee_str = chair_str + ", " + ", ".join(
            f"{rng.choice(firstnames)} {s}" for s in session_surnames[1:]
        )
        votes_str = "  ".join(
            f"{s}: {rng.choice(['yes', 'yes', 'no', 'abstain'])}" for s in session_surnames
        )
        return (
            f"[ENGINEERING REVIEW — Session {date}]\n"
            f"Attendees: {attendee_str}\n\n"
            f"Item {rng.randint(1,6)}: {item}.\n"
            f"  Motion: {rng.choice(firstnames)} {mover}. Second: {rng.choice(firstnames)} {seconder}.\n"
            f"  Votes — {votes_str}\n"
            f"  Outcome: {outcome} ({yes} yes / {no} no / {abst} abstain)"
        )

    needed_chars = int(target_tokens * _CHARS_PER_TOKEN * 1.5)
    entries: list[str] = []
    total_chars = 0
    while total_chars < needed_chars:
        e = make_entry()
        entries.append(e)
        total_chars += len(e) + 2
        if len(entries) >= _FILLER_BUILD_CAP:
            print(f"  [art_06/F-TYPED] capped at {_FILLER_BUILD_CAP} entries", flush=True)
            break

    print(f"  [art_06/F-TYPED] {len(entries)} entries, {total_chars} chars, trimming...", flush=True)
    raw = "\n\n".join(entries)
    text, tok, converged = _trim_with_cap(raw, target_tokens, count_fn, label="art_06/F-TYPED")
    return text, list(set(liftable)), {k: list(v) for k, v in other_fields.items()}, tok, converged


def _build_version_filler(target_tokens, seed, count_fn):
    """F-TYPED for art_07: same changelog format, OTHER ORM names (not Ferrite)."""
    rng = random.Random(seed)
    EXCLUDE_VERSIONS: set[str] = {"3.11.9", "3.12.0", "3.11.8"}
    orm_names = ["Granite ORM", "Quartz ORM", "Basalt ORM", "Obsidian ORM",
                 "Feldspar ORM", "Schist ORM"]
    cve_changes = [
        "stack overflow in schema diff with cyclic references",
        "heap overflow in connection pool under burst load",
        "integer overflow in bulk-insert batch counter",
        "use-after-free in cursor iteration on connection return",
        "off-by-one in migration rollback with FK constraints",
        "NULL dereference in idle eviction under concurrent writes",
    ]
    feature_changes = [
        "Added support for nullable composite foreign keys",
        "Bulk insert now respects on_conflict=REPLACE for SQLite targets",
        "Fixed incorrect OFFSET behaviour with GROUP BY on MySQL 8.4+",
        "Cursor iteration no longer holds open transaction on connection return",
        "Added connection pool eviction for idle connections exceeding 300 s",
        "Corrected UTC offset handling for timestamps in Oracle time zones",
        "Improved index hint generation for correlated subqueries",
        "Batch fetch now honours per-relation prefetch limit",
    ]
    liftable = []
    other_fields = {"cve_ids": set()}

    def make_entry():
        for _attempt in range(100):
            major = rng.randint(2, 5)
            minor = rng.randint(0, 14)
            patch_hi = rng.randint(1, 15)
            patch_lo = patch_hi - 1
            ver_hi = f"{major}.{minor}.{patch_hi}"
            ver_lo = f"{major}.{minor}.{patch_lo}"
            if ver_hi not in EXCLUDE_VERSIONS and ver_lo not in EXCLUDE_VERSIONS:
                break
        EXCLUDE_VERSIONS.add(ver_hi)
        EXCLUDE_VERSIONS.add(ver_lo)
        liftable.extend([ver_hi, ver_lo])
        cve_num = f"CVE-2024-{rng.randint(10000,99999)}"
        other_fields["cve_ids"].add(cve_num)
        date_hi = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        date_lo = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        orm = rng.choice(orm_names)
        return (
            f"[RELEASE NOTES — {orm}]\n\n"
            f"Version {ver_hi} ({date_hi})\n"
            f"  - {rng.choice(feature_changes)}\n"
            f"  - {rng.choice(feature_changes)}\n\n"
            f"Version {ver_lo} ({date_lo})\n"
            f"  - Patched {cve_num}: {rng.choice(cve_changes)}\n"
            f"  - Cursor iteration now releases lock on connection return"
        )

    needed_chars = int(target_tokens * _CHARS_PER_TOKEN * 1.5)
    entries: list[str] = []
    total_chars = 0
    while total_chars < needed_chars:
        e = make_entry()
        entries.append(e)
        total_chars += len(e) + 2
        if len(entries) >= _FILLER_BUILD_CAP:
            print(f"  [art_07/F-TYPED] capped at {_FILLER_BUILD_CAP} entries", flush=True)
            break

    print(f"  [art_07/F-TYPED] {len(entries)} entries, {total_chars} chars, trimming...", flush=True)
    raw = "\n\n".join(entries)
    text, tok, converged = _trim_with_cap(raw, target_tokens, count_fn, label="art_07/F-TYPED")
    return text, list(set(liftable)), {k: list(v) for k, v in other_fields.items()}, tok, converged


# ---------------------------------------------------------------------------
# F-SCHEMA builders — same record format as each artifact, different entities
# ---------------------------------------------------------------------------

def _build_art01_schema(target_tokens, seed, count_fn):
    """F-SCHEMA for art_01: exact same CONFIG JSON format, invoice/relay-adjacent
    service names (not invoice-relay). Same field names: listen_port,
    max_payload_kb, auth_scheme, flush_interval_ms, retry_backoff_base_ms,
    circuit_breaker_threshold."""
    rng = random.Random(seed)
    EXCLUDE_PORTS = {51847}
    services = [
        "invoice-gateway", "invoice-processor", "invoice-validator",
        "invoice-archiver", "invoice-normalizer", "invoice-router",
        "billing-relay", "payment-relay", "settlement-relay", "audit-relay",
        "sync-relay", "ledger-api", "statement-api", "receipt-processor",
        "charge-router", "refund-handler",
    ]
    schemes = ["HMAC-SHA3", "Bearer", "mTLS", "API-Key", "OIDC", "HMAC-SHA256"]
    liftable = []
    other_fields = {"auth_schemes": set(), "max_payload_kb": set()}

    def make_entry():
        port = rng.randint(10240, 65535)
        while port in EXCLUDE_PORTS:
            port = rng.randint(10240, 65535)
        EXCLUDE_PORTS.add(port)
        liftable.append(str(port))
        svc = rng.choice(services)
        ver = f"{rng.randint(1,5)}.{rng.randint(0,12)}"
        kb = rng.choice([64, 128, 192, 256, 512])
        scheme = rng.choice(schemes)
        other_fields["auth_schemes"].add(scheme)
        other_fields["max_payload_kb"].add(str(kb))
        return (
            f"[CONFIG: service={svc} v{ver}]\n"
            f"{{\n"
            f'  "listen_port": {port},\n'
            f'  "max_payload_kb": {kb},\n'
            f'  "auth_scheme": "{scheme}",\n'
            f'  "flush_interval_ms": {rng.randint(500,5000)},\n'
            f'  "retry_backoff_base_ms": {rng.randint(100,2000)},\n'
            f'  "circuit_breaker_threshold": {rng.randint(4,30)}\n'
            f"}}"
        )

    needed_chars = int(target_tokens * _CHARS_PER_TOKEN * 1.5)
    entries: list[str] = []
    total_chars = 0
    while total_chars < needed_chars:
        e = make_entry()
        entries.append(e)
        total_chars += len(e) + 2
        if len(entries) >= _FILLER_BUILD_CAP:
            print(f"  [art_01/F-SCHEMA] capped at {_FILLER_BUILD_CAP} entries", flush=True)
            break

    print(f"  [art_01/F-SCHEMA] {len(entries)} entries, {total_chars} chars, trimming...", flush=True)
    raw = "\n\n".join(entries)
    text, tok, converged = _trim_with_cap(raw, target_tokens, count_fn, label="art_01/F-SCHEMA")
    return text, liftable, {k: list(v) for k, v in other_fields.items()}, tok, converged


def _build_art06_schema(target_tokens, seed, count_fn):
    """F-SCHEMA for art_06: exact same ENGINEERING REVIEW BOARD format (matching
    the artifact's board name verbatim), different sessions. Attendee pool
    includes artifact names (Blum, Nakagawa, Herrera, Makinen, Okonkwo) so
    Blum can appear as seconder in filler items."""
    rng = random.Random(seed)
    firstnames = ["R.", "T.", "P.", "M.", "J.", "S.", "K.", "A.", "C.", "D.",
                  "H.", "F.", "G.", "L.", "N.", "E.", "W.", "I.", "O.", "B."]
    surnames = [
        "Okonkwo", "Nakagawa", "Herrera", "Makinen", "Blum",
        "Lindström", "Watanabe", "Ferreira", "Nkosi", "Patel",
        "Kowalski", "Bergström", "Nakamura", "Abramov", "Fischer",
        "Tremblay", "Yamamoto", "Oyelaran", "Kovacs", "Santana",
        "Mehrotra", "Castillo", "Svensson", "Mwangi", "Petrov",
        "Thornton", "Guerrero", "Andersen", "Hashimoto", "Moreau",
    ]
    items = [
        "Upgrade monitoring stack to Prometheus 3.x",
        "Extend disaster recovery RTO from 4 h to 2 h",
        "Consolidate CI pipelines across three product teams",
        "Adopt OpenTelemetry as the standard tracing library",
        "Increase on-call stipend by 15% effective Q3",
        "Migrate remaining HTTP/1.1 internal services to HTTP/2",
        "Approve new incident severity classification matrix",
        "Retire ATLAS-1 batch processing service by year-end",
        "Mandate TLS 1.3 minimum across all internal endpoints",
        "Adopt Infrastructure-as-Code policy for all new deployments",
        "Review and update data retention policy for observability logs",
        "Approve budget for additional SRE headcount in H2",
    ]
    liftable = []
    other_fields = {"movers": set(), "chairs": set()}

    def make_entry():
        session_surnames = rng.sample(surnames, 5)
        chair = session_surnames[0]
        mover = session_surnames[rng.randint(1, 4)]
        seconder = session_surnames[rng.randint(1, 4)]
        if seconder == mover:
            seconder = session_surnames[1] if mover != session_surnames[1] else session_surnames[2]
        liftable.append(seconder)
        liftable.append(mover)
        other_fields["movers"].add(mover)
        other_fields["chairs"].add(chair)
        item = rng.choice(items)
        yes = rng.randint(2, 4)
        no = 5 - yes - rng.randint(0, 1)
        abst = 5 - yes - no
        outcome = "PASSED" if yes > no else "FAILED"
        date = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        chair_str = f"{rng.choice(firstnames)} {chair} (chair)"
        attendee_str = chair_str + ", " + ", ".join(
            f"{rng.choice(firstnames)} {s}" for s in session_surnames[1:]
        )
        votes_str = "  ".join(
            f"{s}: {rng.choice(['yes', 'yes', 'no', 'abstain'])}" for s in session_surnames
        )
        return (
            f"[ENGINEERING REVIEW BOARD — Session {date}]\n"
            f"Attendees: {attendee_str}\n\n"
            f"Item {rng.randint(1,6)}: {item}.\n"
            f"  Motion: {rng.choice(firstnames)} {mover}. Second: {rng.choice(firstnames)} {seconder}.\n"
            f"  Votes — {votes_str}\n"
            f"  Outcome: {outcome} ({yes} yes / {no} no / {abst} abstain)"
        )

    needed_chars = int(target_tokens * _CHARS_PER_TOKEN * 1.5)
    entries: list[str] = []
    total_chars = 0
    while total_chars < needed_chars:
        e = make_entry()
        entries.append(e)
        total_chars += len(e) + 2
        if len(entries) >= _FILLER_BUILD_CAP:
            print(f"  [art_06/F-SCHEMA] capped at {_FILLER_BUILD_CAP} entries", flush=True)
            break

    print(f"  [art_06/F-SCHEMA] {len(entries)} entries, {total_chars} chars, trimming...", flush=True)
    raw = "\n\n".join(entries)
    text, tok, converged = _trim_with_cap(raw, target_tokens, count_fn, label="art_06/F-SCHEMA")
    return text, list(set(liftable)), {k: list(v) for k, v in other_fields.items()}, tok, converged


def _build_art07_schema(target_tokens, seed, count_fn):
    """F-SCHEMA for art_07: RELEASE NOTES for Ferrite ORM specifically (same ORM
    as artifact), with version numbers outside the artifact's range and different
    CVEs (not CVE-2024-51022). The model must find CVE-2024-51022 among many
    Ferrite ORM entries. Tests whether same-ORM records cause version confusion.

    Version space: minor in [1..10, 13, 14, 15] (avoids artifact's minor 11/12),
    patch_hi in [1..15]. Inner loop capped at 100 attempts; if space exhausted,
    accepts a potentially-repeated combination rather than hanging.
    """
    rng = random.Random(seed)
    EXCLUDE_VERSIONS: set[str] = {"3.11.9", "3.12.0", "3.11.8"}
    # Expanded minor range: avoids 11 and 12 (artifact's minor versions).
    # 13 values × 15 patch values = 195 unique (minor, patch_hi) pairs — sufficient.
    minor_ranges = list(range(1, 11)) + [13, 14, 15]
    EXCLUDE_CVES: set[str] = {"CVE-2024-51022"}
    cve_changes = [
        "stack overflow in schema diff with cyclic references",
        "heap overflow in connection pool under burst load",
        "integer overflow in bulk-insert batch counter",
        "use-after-free in cursor iteration on connection return",
        "off-by-one in migration rollback with FK constraints",
        "NULL dereference in idle eviction under concurrent writes",
        "race condition in savepoint rollback under high concurrency",
        "double-free in prepared statement cache on cursor close",
    ]
    feature_changes = [
        "Added support for nullable composite foreign keys",
        "Bulk insert now respects on_conflict=REPLACE for SQLite targets",
        "Fixed incorrect OFFSET behaviour with GROUP BY on MySQL 8.4+",
        "Cursor iteration no longer holds open transaction on connection return",
        "Added connection pool eviction for idle connections exceeding 300 s",
        "Corrected UTC offset handling for timestamps in Oracle time zones",
        "Improved index hint generation for correlated subqueries",
        "Batch fetch now honours per-relation prefetch limit",
        "Added read-replica routing for SELECT queries",
        "Schema migration now validates foreign key consistency before apply",
    ]
    liftable = []
    other_fields = {"cve_ids": set()}

    def make_entry():
        # Cap inner search at 100 attempts; if version space is exhausted,
        # accept a repeated combination rather than hanging.
        for _attempt in range(100):
            minor = rng.choice(minor_ranges)
            patch_hi = rng.randint(1, 15)
            patch_lo = patch_hi - 1
            ver_hi = f"3.{minor}.{patch_hi}"
            ver_lo = f"3.{minor}.{patch_lo}"
            if ver_hi not in EXCLUDE_VERSIONS and ver_lo not in EXCLUDE_VERSIONS and ver_hi != ver_lo:
                break
        # Add unconditionally — even if a repeated pair slips through,
        # liftable tracking still works (artifact versions excluded from EXCLUDE_CVES).
        EXCLUDE_VERSIONS.add(ver_hi)
        EXCLUDE_VERSIONS.add(ver_lo)
        liftable.extend([ver_hi, ver_lo])
        cve_num = f"CVE-2024-{rng.randint(10000,99999)}"
        while cve_num in EXCLUDE_CVES:
            cve_num = f"CVE-2024-{rng.randint(10000,99999)}"
        EXCLUDE_CVES.add(cve_num)
        other_fields["cve_ids"].add(cve_num)
        date_hi = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        date_lo = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        return (
            f"[RELEASE NOTES — Ferrite ORM]\n\n"
            f"Version {ver_hi} ({date_hi})\n"
            f"  - {rng.choice(feature_changes)}\n"
            f"  - {rng.choice(feature_changes)}\n\n"
            f"Version {ver_lo} ({date_lo})\n"
            f"  - Patched {cve_num}: {rng.choice(cve_changes)}\n"
            f"  - Cursor iteration now releases lock on connection return"
        )

    needed_chars = int(target_tokens * _CHARS_PER_TOKEN * 1.5)
    entries: list[str] = []
    total_chars = 0
    while total_chars < needed_chars:
        e = make_entry()
        entries.append(e)
        total_chars += len(e) + 2
        if len(entries) >= _FILLER_BUILD_CAP:
            print(f"  [art_07/F-SCHEMA] capped at {_FILLER_BUILD_CAP} entries", flush=True)
            break

    print(f"  [art_07/F-SCHEMA] {len(entries)} entries, {total_chars} chars, trimming...", flush=True)
    raw = "\n\n".join(entries)
    text, tok, converged = _trim_with_cap(raw, target_tokens, count_fn, label="art_07/F-SCHEMA")
    return text, list(set(liftable)), {k: list(v) for k, v in other_fields.items()}, tok, converged


TYPED_BUILDERS = {
    "art_01": _build_port_filler,
    "art_06": _build_surname_filler,
    "art_07": _build_version_filler,
}

SCHEMA_BUILDERS = {
    "art_01": _build_art01_schema,
    "art_06": _build_art06_schema,
    "art_07": _build_art07_schema,
}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _count_fn(text):
    payload = json.dumps({
        "model": COUNT_MODEL,
        "prompt": text,
        "options": {"num_ctx": 8192, "num_predict": 1, "temperature": 0},
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["prompt_eval_count"]


def _call(prompt, model):
    _first_call_started.set()  # disarms watchdog
    is_thinking = model == "gpt-oss:120b-cloud"
    num_predict = MIN_PREDICT_120B if is_thinking else MAX_TOKENS
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_predict": num_predict, "temperature": 0},
        "stream": False,
    }
    if "qwen" in model or is_thinking:
        body["think"] = False
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        content = data["message"]["content"].strip()
        thinking = (data["message"].get("thinking") or "").strip() if is_thinking else None
        return (content, thinking, time.monotonic() - t0,
                data.get("eval_count"), data.get("prompt_eval_count"), data.get("done_reason"))
    except Exception as e:
        print(f"    [CALL ERROR: {e}]")
        return None, None, time.monotonic() - t0, None, None, None


def value_in_text(value, text):
    return bool(re.search(
        r"(?<![0-9a-zA-Z._-])" + re.escape(value) + r"(?![0-9a-zA-Z._-])", text
    ))


def lifted_from_filler(output, liftable_values):
    if not output:
        return False
    return any(value_in_text(v, output) for v in liftable_values)


def classify_wrong_field(output, probe_id, other_fields):
    if not output:
        return None
    if probe_id == "art_01":
        if any(value_in_text(v, output) for v in other_fields.get("max_payload_kb", [])):
            return "max_payload_kb"
        for scheme in other_fields.get("auth_schemes", []):
            if scheme.lower() in output.lower():
                return "auth_scheme"
        return "listen_port_or_unknown"
    if probe_id == "art_06":
        if any(value_in_text(v, output) for v in other_fields.get("movers", [])):
            return "mover"
        if any(value_in_text(v, output) for v in other_fields.get("chairs", [])):
            return "chair"
        return "seconder_or_unknown"
    if probe_id == "art_07":
        for cve in other_fields.get("cve_ids", []):
            if cve in output:
                return "cve_id"
        return "version_or_unknown"
    return "unknown"


def _load_completed(outfile):
    """Return set of (probe_id, filler_type, model, rep) already in outfile."""
    completed = set()
    if not outfile.exists():
        return completed
    for line in outfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            completed.add((r["probe_id"], r["filler_type"], r["model"], r["rep"]))
        except (json.JSONDecodeError, KeyError):
            pass
    return completed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells already present in the output file.")
    args = parser.parse_args()

    # Filler building makes ~50-70 count_fn calls (~5s each) before the first
    # model call, so 300s is too tight. 900s catches runaway loops (which took
    # 15+ hours before) without false-positives on normal filler building.
    _start_watchdog(timeout=900.0)

    RESULTS.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("scorers", PROBES_DIR / "scorers.py")
    scorers_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorers_mod)

    segments = {
        s["id"]: s
        for s in [
            json.loads(l)
            for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        if s["id"] in TARGET_PROBES
    }

    completed = _load_completed(OUTFILE) if args.resume else set()
    if completed:
        print(f"--resume: {len(completed)} cells already completed, skipping them.")

    print(f"Building F-NUM filler ({FILLER_TOKENS} tok)...")
    filler_num = ctx_mod.build_filler(FILLER_TOKENS, seed=42, count_fn=_count_fn, variant="F-NUM")
    filler_num_tok = _count_fn(filler_num)
    print(f"  F-NUM: {filler_num_tok} tok ({len(filler_num)} chars)")

    print("Building F-TYPED fillers (seed=42)...")
    filler_typed: dict[str, tuple] = {}
    filler_typed_meta: dict[str, tuple[int, bool]] = {}
    for pid in TARGET_PROBES:
        text, liftable, other, tok, converged = TYPED_BUILDERS[pid](FILLER_TOKENS, seed=42, count_fn=_count_fn)
        if not converged:
            print(f"  WARNING: {pid} F-TYPED filler did not converge (realized={tok} tok)")
        print(f"  {pid} F-TYPED: {tok} tok, {len(liftable)} liftable values", flush=True)
        filler_typed[pid] = (text, liftable, other)
        filler_typed_meta[pid] = (tok, converged)

    print("Building F-SCHEMA fillers (seed=42)...")
    filler_schema: dict[str, tuple] = {}
    filler_schema_meta: dict[str, tuple[int, bool]] = {}
    for pid in TARGET_PROBES:
        text, liftable, other, tok, converged = SCHEMA_BUILDERS[pid](FILLER_TOKENS, seed=42, count_fn=_count_fn)
        if not converged:
            print(f"  WARNING: {pid} F-SCHEMA filler did not converge (realized={tok} tok)")
        print(f"  {pid} F-SCHEMA: {tok} tok, {len(liftable)} liftable values", flush=True)
        filler_schema[pid] = (text, liftable, other)
        filler_schema_meta[pid] = (tok, converged)

    print("Measuring full prompt token counts...")
    for pid in TARGET_PROBES:
        seg = segments[pid]
        for fname, ftext in [
            ("F-NUM",    filler_num),
            ("F-TYPED",  filler_typed[pid][0]),
            ("F-SCHEMA", filler_schema[pid][0]),
        ]:
            ft = _count_fn(f"{seg['artifact']}\n\n{ftext}\n\n{seg['question']}")
            print(f"  {pid} {fname}: {ft} tok")

    # Build per-filler realized_tokens and convergence_failed metadata.
    filler_realized: dict[tuple, int] = {}
    filler_converged: dict[tuple, bool] = {}
    filler_realized[("F-NUM", "*")] = filler_num_tok
    filler_converged[("F-NUM", "*")] = True
    for pid in TARGET_PROBES:
        filler_realized[("F-TYPED",  pid)] = filler_typed_meta[pid][0]
        filler_converged[("F-TYPED",  pid)] = filler_typed_meta[pid][1]
        filler_realized[("F-SCHEMA", pid)] = filler_schema_meta[pid][0]
        filler_converged[("F-SCHEMA", pid)] = filler_schema_meta[pid][1]

    fillers_config = [
        ("F-NUM",    lambda pid: (filler_num,              [],                           {})),
        ("F-TYPED",  lambda pid: (filler_typed[pid][0],   filler_typed[pid][1],         filler_typed[pid][2])),
        ("F-SCHEMA", lambda pid: (filler_schema[pid][0],  filler_schema[pid][1],        filler_schema[pid][2])),
    ]

    total_calls = len(TARGET_PROBES) * len(fillers_config) * len(MODELS) * N_REPS
    skipped = len(completed)
    print(f"\nTotal cells: {total_calls}  Completed: {skipped}  Remaining: {total_calls - skipped}")
    print(f"Output: {OUTFILE}")

    call_n = 0
    written = 0

    with OUTFILE.open("a", encoding="utf-8") as out_fh:
        for filler_name, filler_fn in fillers_config:
            for model in MODELS:
                for pid in TARGET_PROBES:
                    seg = segments[pid]
                    probe_dict = {
                        "id": pid,
                        "scorer_type": seg["scorer_type"],
                        "expected": seg["expected"],
                    }
                    filler_text, liftable, other_fields = filler_fn(pid)
                    r_tok = filler_realized.get((filler_name, pid), filler_realized.get((filler_name, "*"), 0))
                    r_conv = filler_converged.get((filler_name, pid), filler_converged.get((filler_name, "*"), True))

                    for rep in range(N_REPS):
                        call_n += 1
                        cell = (pid, filler_name, model, rep)

                        if cell in completed:
                            print(f"[{call_n:3d}/{total_calls}] SKIP {pid} {filler_name:8} {model[:16]} r{rep}")
                            continue

                        prompt = f"{seg['artifact']}\n\n{filler_text}\n\n{seg['question']}"
                        output, thinking, latency, eval_count, prompt_eval_count, done_reason = _call(prompt, model)

                        score, score_detail = (
                            scorers_mod.score(probe_dict, output) if output is not None
                            else (None, "no output")
                        )
                        outcome = scorers_mod.outcome_class(output, score, truncated=False)
                        is_lifted = lifted_from_filler(output, liftable) if outcome == "incorrect" else False

                        wrong_field = None
                        if outcome == "incorrect" and output:
                            if is_lifted:
                                wrong_field = {
                                    "art_01": "listen_port",
                                    "art_06": "seconder_or_mover",
                                    "art_07": "version",
                                }.get(pid)
                            else:
                                wrong_field = classify_wrong_field(output, pid, other_fields)

                        row = {
                            "probe_id": pid,
                            "model": model,
                            "filler_type": filler_name,
                            "filler_realized_tokens": r_tok,
                            "convergence_failed": not r_conv,
                            "budget_ratio": BUDGET_RATIO,
                            "truncating": False,
                            "artifact_fraction": 1.0,
                            "rep": rep,
                            "output": output,
                            "score": score,
                            "score_detail": score_detail,
                            "outcome": outcome,
                            "lifted_from_filler": is_lifted,
                            "wrong_field": wrong_field,
                            "eval_count": eval_count,
                            "prompt_eval_count": prompt_eval_count,
                            "done_reason": done_reason,
                            "latency_s": round(latency, 3),
                            "hardware": "blade14_rtx4070",
                        }
                        if thinking is not None:
                            row["thinking"] = thinking

                        out_fh.write(json.dumps(row) + "\n")
                        out_fh.flush()
                        written += 1

                        tag = "✓" if score == 1.0 else ("L" if is_lifted else "✗")
                        field_tag = f" [{wrong_field}]" if wrong_field else ""
                        print(
                            f"[{call_n:3d}/{total_calls}] {pid} {filler_name:8} "
                            f"{model[:16]} r{rep} {tag}{field_tag} {latency*1000:.0f}ms  "
                            f"{repr((output or '')[:50])}"
                        )

                        if call_n % 10 == 0:
                            print(f"--- progress: {call_n}/{total_calls} calls, {written} written ---")

    print(f"\nDone. {written} rows written to {OUTFILE}")
    _print_summary(OUTFILE)


def _print_summary(outfile):
    rows = []
    for line in Path(outfile).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    filler_names = ["F-NUM", "F-TYPED", "F-SCHEMA"]
    print("\n=== SCHEMA COLLISION SUMMARY (r=1.20, af=1.00) ===")
    print(f"{'probe':8} {'filler':10} {'model':20}  score  lift  wrong_answers")
    for filler_name in filler_names:
        for model in MODELS:
            for pid in TARGET_PROBES:
                s = [r for r in rows
                     if r["probe_id"] == pid and r["filler_type"] == filler_name and r["model"] == model]
                if not s:
                    continue
                scores = [r["score"] for r in s if r["score"] is not None]
                lifts = [r for r in s if r.get("lifted_from_filler")]
                mean_score = sum(scores) / len(scores) if scores else float("nan")
                lift_rate = len(lifts) / len(s)
                wrong = [r for r in s if (r["score"] or 0) < 1.0 and r.get("output")]
                wrong_strs = ", ".join(
                    f"{repr((r['output'] or '')[:20])}({r.get('wrong_field','?')})"
                    for r in wrong
                ) if wrong else "—"
                conv_flag = ""
                if any(r.get("convergence_failed") for r in s):
                    conv_flag = " [conv-fail]"
                print(
                    f"  {pid:8} {filler_name:10} {model:20}  {mean_score:.2f}   "
                    f"{lift_rate:.0%}  {wrong_strs}{conv_flag}"
                )

    print("\n=== ALL FAILURES (score < 1.0) ===")
    any_failure = False
    for r in rows:
        if (r["score"] or 0) < 1.0:
            any_failure = True
            print(
                f"  {r['probe_id']} {r['filler_type']:8} {r['model'][:16]} rep={r['rep']}"
                f"  score={r['score']}  lift={r['lifted_from_filler']}"
                f"  field={r.get('wrong_field','?')}"
                f"  out={repr((r['output'] or '')[:60])}"
            )
    if not any_failure:
        print("  (none)")


if __name__ == "__main__":
    main()
