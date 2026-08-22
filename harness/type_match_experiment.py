"""Type-match experiment: does type-matched filler raise fabrication rate?

Hypothesis: model lifts from context when surviving filler is type-matched to
the expected answer shape. sea probes lifted (filler=numbered records,
answer=record value); art probes did not (filler=admin logs, answer varies).
If type-matched filler causes verbatim lifting, the mechanism is confirmed.

Design:
  Probes: art_01 (structured, port int), art_02 (structured, ppb float),
          art_06 (narrative, surname), art_07 (narrative, version string)
  Filler A: F-NUM  — current admin-log filler (dissimilar to artifact type)
  Filler B: F-TYPED — filler matching each probe's artifact type and answer shape
  Ratios:   1.20 (baseline), 0.98, 0.85, 0.40
  Models:   qwen3:4b-instruct, llama3.1:8b
  Reps:     3
  Total:    4 x 2 x 4 x 2 x 3 = 192 calls

Key output per fabricated row: whether the fabricated value appears verbatim
in the filler content.

Results written to results/type_match.json.
"""

from __future__ import annotations

import importlib.util
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).parent.parent
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "harness"))

import context as ctx_mod

TARGET_PROBES = ["art_01", "art_02", "art_06", "art_07"]
MODELS = ["qwen3:4b-instruct", "llama3.1:8b"]
HOST = "http://localhost:11434"
FILLER_TOKENS = 4000
BUDGET_RATIOS = [1.20, 0.98, 0.85, 0.40]
N_REPS = 3
MAX_TOKENS = 256


# ---------------------------------------------------------------------------
# Type-matched filler generators
# ---------------------------------------------------------------------------

def _build_port_filler(target_tokens: int, seed: int, count_fn) -> tuple[str, list]:
    """JSON config blocks full of listen_port values. Excludes 51847."""
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

    def make_entry():
        port = rng.randint(10240, 65535)
        while port in EXCLUDE:
            port = rng.randint(10240, 65535)
        EXCLUDE.add(port)
        liftable.append(str(port))
        svc = rng.choice(services)
        ver = f"{rng.randint(1, 5)}.{rng.randint(0, 12)}"
        return (
            f"[CONFIG: service={svc} v{ver}]\n"
            f"{{\n"
            f'  "listen_port": {port},\n'
            f'  "max_payload_kb": {rng.choice([64, 128, 192, 256, 512])},\n'
            f'  "auth_scheme": "{rng.choice(schemes)}",\n'
            f'  "flush_interval_ms": {rng.randint(500, 5000)},\n'
            f'  "retry_backoff_base_ms": {rng.randint(100, 2000)},\n'
            f'  "circuit_breaker_threshold": {rng.randint(4, 30)}\n'
            f"}}"
        )

    entries = []
    while True:
        entries.append(make_entry())
        text = "\n\n".join(entries)
        if count_fn(text) >= target_tokens:
            break
    return text, liftable


def _build_ppb_filler(target_tokens: int, seed: int, count_fn) -> tuple[str, list]:
    """Calibration records full of alert_threshold_ppb values. Excludes 0.0073."""
    rng = random.Random(seed)
    EXCLUDE_STR = {"0.0073"}
    facilities = [
        "Aldridge-C", "Morrow-A", "Kessler-B", "Weston-F", "Travers-D",
        "Holbrook-E", "Pemberton-G", "Slade-H", "Carver-J", "Norwood-K",
    ]
    lots = ["XK-4419", "XK-4420", "XK-4421", "XK-4418", "YM-5501",
            "YM-5502", "ZL-0032", "ZL-0033", "ZL-0034", "AM-9917"]
    liftable = []

    def make_entry():
        unit_id = rng.randint(1, 20)
        # generate ppb float in range 0.0010 – 0.0200, 4 sig figs, exclude 0.0073
        while True:
            base = rng.randint(10, 200)
            val_str = f"0.00{base:02d}" if base < 100 else f"0.0{base}"
            val_str = f"{int(base)/10000:.4f}".lstrip('0') or '0'
            val_str = f"{base / 10000:.4f}"
            if val_str not in EXCLUDE_STR:
                break
        EXCLUDE_STR.add(val_str)
        liftable.append(val_str)
        fac = rng.choice(facilities)
        lot = rng.choice(lots)
        return (
            f"[CALIBRATION RECORD: unit=CHROM-{unit_id}, facility={fac}]\n"
            f"baseline_ppm        : {rng.randint(80, 180) / 10:.1f}\n"
            f"alert_threshold_ppb : {val_str}\n"
            f"sample_interval_s   : {rng.choice([180, 240, 300, 360, 600])}\n"
            f"reagent_lot         : {lot}\n"
            f"correction_factor   : {1 + rng.randint(-100, 200)/10000:.4f}\n"
            f"last_verified       : 2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        )

    entries = []
    while True:
        entries.append(make_entry())
        text = "\n\n".join(entries)
        if count_fn(text) >= target_tokens:
            break
    return text, liftable


def _build_surname_filler(target_tokens: int, seed: int, count_fn) -> tuple[str, list]:
    """Meeting minutes full of surnames. Excludes 'Blum'."""
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
    outcomes = ["PASSED", "PASSED", "PASSED", "FAILED", "DEFERRED"]
    liftable = []

    def make_entry():
        session_surnames = [s for s in rng.sample(surnames, 5) if s not in EXCLUDE][:5]
        if len(session_surnames) < 5:
            session_surnames.append("Ivanova")
        chair = session_surnames[0]
        attendee_str = ", ".join(
            f"{rng.choice(firstnames)} {s}" for s in session_surnames
        )
        chair_str = f"{rng.choice(firstnames)} {chair} (chair)"
        attendee_str = chair_str + ", " + ", ".join(
            f"{rng.choice(firstnames)} {s}" for s in session_surnames[1:]
        )
        mover = session_surnames[rng.randint(1, 4)]
        seconder = session_surnames[rng.randint(1, 4)]
        if seconder == mover:
            seconder = session_surnames[1] if mover != session_surnames[1] else session_surnames[2]
        liftable.append(seconder)
        liftable.append(mover)
        item = rng.choice(items)
        yes = rng.randint(2, 4)
        no = 5 - yes - rng.randint(0, 1)
        abst = 5 - yes - no
        outcome = "PASSED" if yes > no else "FAILED"
        date = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        votes_str = "  ".join(
            f"{s}: {rng.choice(['yes','yes','no','abstain'])}"
            for s in session_surnames
        )
        return (
            f"[ENGINEERING REVIEW — Session {date}]\n"
            f"Attendees: {attendee_str}\n\n"
            f"Item {rng.randint(1,6)}: {item}.\n"
            f"  Motion: {rng.choice(firstnames)} {mover}. Second: {rng.choice(firstnames)} {seconder}.\n"
            f"  Votes — {votes_str}\n"
            f"  Outcome: {outcome} ({yes} yes / {no} no / {abst} abstain)"
        )

    entries = []
    while True:
        entries.append(make_entry())
        text = "\n\n".join(entries)
        if count_fn(text) >= target_tokens:
            break
    return text, list(set(liftable))


def _build_version_filler(target_tokens: int, seed: int, count_fn) -> tuple[str, list]:
    """Changelog entries full of version strings and CVE IDs. Excludes 3.11.9."""
    rng = random.Random(seed)
    EXCLUDE_VERSIONS = {"3.11.9"}
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

    def make_version(major, minor, patch):
        return f"{major}.{minor}.{patch}"

    def make_entry():
        orm = rng.choice(orm_names)
        major = rng.randint(2, 5)
        minor = rng.randint(0, 14)
        patch_hi = rng.randint(2, 8)
        patch_lo = patch_hi - rng.randint(1, 2)
        ver_hi = make_version(major, minor, patch_hi)
        ver_lo = make_version(major, minor, patch_lo)
        # keep retrying until neither version is excluded
        while ver_hi in EXCLUDE_VERSIONS or ver_lo in EXCLUDE_VERSIONS:
            major = rng.randint(2, 5)
            minor = rng.randint(0, 14)
            patch_hi = rng.randint(2, 8)
            patch_lo = patch_hi - 1
            ver_hi = make_version(major, minor, patch_hi)
            ver_lo = make_version(major, minor, patch_lo)
        EXCLUDE_VERSIONS.add(ver_hi)
        EXCLUDE_VERSIONS.add(ver_lo)
        liftable.extend([ver_hi, ver_lo])
        cve_num = f"CVE-2024-{rng.randint(10000, 99999)}"
        date_hi = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        date_lo = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        feat1 = rng.choice(feature_changes)
        feat2 = rng.choice(feature_changes)
        cve_desc = rng.choice(cve_changes)
        return (
            f"[RELEASE NOTES — {orm}]\n\n"
            f"Version {ver_hi} ({date_hi})\n"
            f"  - {feat1}\n"
            f"  - {feat2}\n\n"
            f"Version {ver_lo} ({date_lo})\n"
            f"  - Patched {cve_num}: {cve_desc}\n"
            f"  - Cursor iteration now releases lock on connection return"
        )

    entries = []
    while True:
        entries.append(make_entry())
        text = "\n\n".join(entries)
        if count_fn(text) >= target_tokens:
            break
    return text, list(set(liftable))


TYPED_BUILDERS = {
    "art_01": _build_port_filler,
    "art_02": _build_ppb_filler,
    "art_06": _build_surname_filler,
    "art_07": _build_version_filler,
}


# ---------------------------------------------------------------------------
# Shared inference helpers (same as art_truncation.py)
# ---------------------------------------------------------------------------

def _count_fn(text: str, model: str = "qwen3:4b-instruct") -> int:
    payload = json.dumps({
        "model": model,
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


def _call(prompt: str, model: str) -> tuple[str | None, float]:
    extra = {"think": False} if "qwen" in model else {}
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_ctx": 8192, "num_predict": MAX_TOKENS, "temperature": 0},
        "stream": False,
        **extra,
    }).encode()
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
        return data["message"]["content"].strip(), time.monotonic() - t0
    except Exception as e:
        print(f"    [CALL ERROR: {e}]")
        return None, time.monotonic() - t0


def left_truncate(prompt: str, full_tokens: int, target_tokens: int) -> str:
    if target_tokens >= full_tokens:
        return prompt
    chars_to_keep = round(len(prompt) * target_tokens / full_tokens)
    chars_to_keep = max(1, min(chars_to_keep, len(prompt)))
    return prompt[len(prompt) - chars_to_keep:]


def artifact_survival(artifact: str, chars_dropped: int):
    a_chars = len(artifact)
    surviving = max(0, a_chars - chars_dropped)
    fraction = surviving / a_chars if a_chars > 0 else 0.0
    return surviving, round(fraction, 6)


def value_in_text(value: str, text: str) -> bool:
    """Return True if value appears as a standalone token in text."""
    return bool(re.search(r"(?<![0-9a-zA-Z._-])" + re.escape(value) + r"(?![0-9a-zA-Z._-])", text))


def lifted_from_filler(output: str, liftable_values: list[str]) -> bool:
    if output is None:
        return False
    for v in liftable_values:
        if value_in_text(v, output):
            return True
    return False


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("scorers", PROBES_DIR / "scorers.py")
    scorers_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorers_mod)

    segments = {
        s["id"]: s
        for s in [json.loads(l) for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if s["id"] in TARGET_PROBES
    }

    # Build F-NUM filler (shared across all probes)
    print("Building F-NUM filler (filler_a)...")
    filler_a = ctx_mod.build_filler(FILLER_TOKENS, seed=42, count_fn=_count_fn, variant="F-NUM")
    tokens_a = _count_fn(filler_a)
    print(f"  F-NUM: {tokens_a} tokens ({len(filler_a)} chars)")

    # Build F-TYPED fillers (one per probe)
    print("Building F-TYPED fillers (filler_b)...")
    filler_b_data: dict[str, tuple[str, list]] = {}
    for pid in TARGET_PROBES:
        builder = TYPED_BUILDERS[pid]
        text, liftable = builder(FILLER_TOKENS, seed=42, count_fn=_count_fn)
        tok = _count_fn(text)
        print(f"  {pid} F-TYPED: {tok} tokens ({len(text)} chars), {len(liftable)} liftable values")
        filler_b_data[pid] = (text, liftable)

    # Measure full_tokens for each probe with F-NUM filler
    print("\nMeasuring full prompt sizes (F-NUM)...")
    full_tokens_by_probe: dict[str, int] = {}
    for pid in TARGET_PROBES:
        seg = segments[pid]
        early = f"{seg['artifact']}\n\n{filler_a}\n\n{seg['question']}"
        ft = _count_fn(early)
        ext = 1.0 - len(seg["artifact"]) / len(early)
        full_tokens_by_probe[pid] = ft
        print(f"  {pid}: {ft} tok  extinction≈{ext:.3f}")

    fillers = [
        ("filler_a_F-NUM", filler_a, {pid: [] for pid in TARGET_PROBES}),
        ("filler_b_F-TYPED", None, {pid: filler_b_data[pid][1] for pid in TARGET_PROBES}),
    ]

    total_calls = len(TARGET_PROBES) * 2 * len(BUDGET_RATIOS) * len(MODELS) * N_REPS
    print(f"\nTotal calls: {total_calls}")

    rows = []
    call_n = 0

    for ratio in BUDGET_RATIOS:
        print(f"\n=== ratio {ratio:.2f} ===")
        for model in MODELS:
            for filler_name, filler_text_shared, liftable_map in fillers:
                for pid in TARGET_PROBES:
                    seg = segments[pid]
                    full_tokens = full_tokens_by_probe[pid]
                    target_tokens = round(full_tokens * ratio)
                    truncating = target_tokens < full_tokens

                    if filler_name.startswith("filler_a"):
                        filler_text = filler_text_shared
                        liftable = liftable_map[pid]
                    else:
                        filler_text, liftable = filler_b_data[pid]

                    # Recompute full_tokens for this filler (filler B may differ slightly)
                    if filler_name.startswith("filler_b"):
                        base_probe_ft = _count_fn(
                            f"{seg['artifact']}\n\n{filler_text}\n\n{seg['question']}"
                        )
                        t_tokens = round(base_probe_ft * ratio)
                        truncating = t_tokens < base_probe_ft
                    else:
                        base_probe_ft = full_tokens
                        t_tokens = target_tokens

                    base_prompt = f"{seg['artifact']}\n\n{filler_text}\n\n{seg['question']}"
                    truncated_base = left_truncate(base_prompt, base_probe_ft, t_tokens)
                    chars_dropped = len(base_prompt) - len(truncated_base)
                    _, a_fraction = artifact_survival(seg["artifact"], chars_dropped)
                    a_tokens_surviving = round(_count_fn(seg["artifact"]) * a_fraction)

                    probe_dict = {
                        "id": pid,
                        "scorer_type": seg["scorer_type"],
                        "expected": seg["expected"],
                    }

                    for rep in range(N_REPS):
                        call_n += 1
                        output, latency = _call(truncated_base, model)
                        score, score_detail = (
                            scorers_mod.score(probe_dict, output) if output is not None
                            else (None, "no output")
                        )
                        outcome = scorers_mod.outcome_class(output, score, truncated=truncating)
                        lifted = lifted_from_filler(output, liftable) if outcome == "incorrect" else False

                        row = {
                            "probe_id": pid,
                            "model": model,
                            "filler_type": filler_name,
                            "budget_ratio": ratio,
                            "truncating": truncating,
                            "artifact_fraction": a_fraction,
                            "artifact_tokens_surviving": a_tokens_surviving,
                            "rep": rep,
                            "output": output,
                            "score": score,
                            "score_detail": score_detail,
                            "outcome": outcome,
                            "lifted_from_filler": lifted,
                            "latency_s": round(latency, 3),
                            "hardware": "blade14_rtx4070",
                        }
                        rows.append(row)

                        tag = "✓" if score == 1.0 else ("A" if outcome == "abstained" else ("L" if lifted else "✗"))
                        print(
                            f"[{call_n:3d}/{total_calls}] {pid} {filler_name[:7]} "
                            f"{model[:14]} r{rep} af={a_fraction:.2f} {tag} "
                            f"{latency*1000:.0f}ms  {repr((output or '')[:40])}"
                        )

    out = RESULTS / "type_match.json"
    out.write_text(json.dumps({"rows": rows, "liftable_values": {pid: filler_b_data[pid][1] for pid in TARGET_PROBES}}, indent=2), encoding="utf-8")
    print(f"\nWritten {len(rows)} rows → {out}")
    _print_summary(rows)


def _print_summary(rows):
    from collections import defaultdict
    print("\n=== FABRICATION RATE by probe × filler_type (truncating rows, arm1) ===")
    for pid in TARGET_PROBES:
        for ft in ["filler_a_F-NUM", "filler_b_F-TYPED"]:
            trunc = [r for r in rows if r["probe_id"] == pid and r["filler_type"] == ft
                     and r["truncating"] and r["budget_ratio"] < 1.0]
            n = len(trunc)
            if not n:
                continue
            fab = sum(1 for r in trunc if r["outcome"] == "incorrect")
            lifted = sum(1 for r in trunc if r.get("lifted_from_filler"))
            abs_ = sum(1 for r in trunc if r["outcome"] == "abstained")
            print(f"  {pid} {ft[:7]}: n={n}  fab={fab}({fab/n:.0%})  "
                  f"lifted={lifted}({lifted/n:.0%})  abs={abs_}({abs_/n:.0%})")

    print("\n=== MEAN SCORE at r=1.20 by probe × filler_type (headroom check) ===")
    for pid in TARGET_PROBES:
        for ft in ["filler_a_F-NUM", "filler_b_F-TYPED"]:
            base_rows = [r for r in rows if r["probe_id"] == pid and r["filler_type"] == ft
                         and r["budget_ratio"] == 1.20 and r["score"] is not None]
            if base_rows:
                mean_s = sum(r["score"] for r in base_rows) / len(base_rows)
                print(f"  {pid} {ft[:7]}: {mean_s:.2f} (n={len(base_rows)})")

    print("\n=== FABRICATED VALUES (incorrect, truncating rows) ===")
    for pid in TARGET_PROBES:
        for ft in ["filler_a_F-NUM", "filler_b_F-TYPED"]:
            incorr = [r for r in rows if r["probe_id"] == pid and r["filler_type"] == ft
                      and r["truncating"] and r["outcome"] == "incorrect" and r["output"]]
            if incorr:
                unique_outs = list(dict.fromkeys(r["output"][:60] for r in incorr))
                lifted = [r for r in incorr if r.get("lifted_from_filler")]
                print(f"  {pid} {ft[:7]}: {unique_outs[:3]}  lifted={len(lifted)}/{len(incorr)}")


if __name__ == "__main__":
    main()
