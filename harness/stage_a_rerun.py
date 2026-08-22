"""Rerun Stage A rows that produced empty output.

Reads results/stage_a_scale.json, identifies rows where output == "",
reruns them with MIN_PREDICT=1024, and writes the results back in-place
(replacing those rows). The un-rerun rows receive classification_method:
"unavailable" (they lack done_reason because they predate the harness fix).
Rerun rows receive classification_method: "done_reason" and the done_reason
field is used to distinguish budget_exhausted vs null_response:
  done_reason == "length" -> budget_exhausted (hit token ceiling mid-thinking)
  done_reason == "stop" and output == "" -> null_response
  done_reason == "stop" and output != "" -> regular output (valid)

Results written back to results/stage_a_scale.json.
"""

from __future__ import annotations

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

MODEL = "gpt-oss:120b-cloud"
COUNT_MODEL = "qwen3:4b-instruct"
HOST = "http://localhost:11434"
FILLER_TOKENS = 4000
MIN_PREDICT = 1024


# ---------------------------------------------------------------------------
# Filler generators (identical to stage_a_scale.py, same seed=42)
# ---------------------------------------------------------------------------

def _build_port_filler(target_tokens, seed, count_fn):
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
        ver = f"{rng.randint(1,5)}.{rng.randint(0,12)}"
        return (
            f"[CONFIG: service={svc} v{ver}]\n"
            f"{{\n"
            f'  "listen_port": {port},\n'
            f'  "max_payload_kb": {rng.choice([64,128,192,256,512])},\n'
            f'  "auth_scheme": "{rng.choice(schemes)}",\n'
            f'  "flush_interval_ms": {rng.randint(500,5000)},\n'
            f'  "retry_backoff_base_ms": {rng.randint(100,2000)},\n'
            f'  "circuit_breaker_threshold": {rng.randint(4,30)}\n'
            f"}}"
        )
    entries = []
    while True:
        entries.append(make_entry())
        text = "\n\n".join(entries)
        if count_fn(text) >= target_tokens:
            break
    return text, liftable


def _build_ppb_filler(target_tokens, seed, count_fn):
    rng = random.Random(seed)
    EXCLUDE_STR = {"0.0073"}
    facilities = [
        "Aldridge-C", "Morrow-A", "Kessler-B", "Weston-F", "Travers-D",
        "Holbrook-E", "Pemberton-G", "Slade-H", "Carver-J", "Norwood-K",
    ]
    lots = ["XK-4419","XK-4420","XK-4421","XK-4418","YM-5501",
            "YM-5502","ZL-0032","ZL-0033","ZL-0034","AM-9917"]
    liftable = []
    def make_entry():
        unit_id = rng.randint(1, 20)
        while True:
            base = rng.randint(10, 200)
            val_str = f"{base / 10000:.4f}"
            if val_str not in EXCLUDE_STR:
                break
        EXCLUDE_STR.add(val_str)
        liftable.append(val_str)
        fac = rng.choice(facilities)
        lot = rng.choice(lots)
        return (
            f"[CALIBRATION RECORD: unit=CHROM-{unit_id}, facility={fac}]\n"
            f"baseline_ppm        : {rng.randint(80,180)/10:.1f}\n"
            f"alert_threshold_ppb : {val_str}\n"
            f"sample_interval_s   : {rng.choice([180,240,300,360,600])}\n"
            f"reagent_lot         : {lot}\n"
            f"correction_factor   : {1+rng.randint(-100,200)/10000:.4f}\n"
            f"last_verified       : 2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        )
    entries = []
    while True:
        entries.append(make_entry())
        text = "\n\n".join(entries)
        if count_fn(text) >= target_tokens:
            break
    return text, liftable


def _build_surname_filler(target_tokens, seed, count_fn):
    rng = random.Random(seed)
    EXCLUDE = {"Blum", "blum"}
    firstnames = ["R.","T.","P.","M.","J.","S.","K.","A.","C.","D.",
                  "H.","F.","G.","L.","N.","E.","W.","I.","O.","B."]
    surnames = [
        "Okafor","Lindström","Watanabe","Ferreira","Nkosi","Patel",
        "Kowalski","Bergström","Nakamura","Herrera","Abramov","Fischer",
        "Tremblay","Yamamoto","Oyelaran","Kovacs","Santana","Eriksen",
        "Mehrotra","Castillo","Svensson","Mwangi","Petrov","Reyes",
        "Thornton","Nakagawa","Guerrero","Andersen","Hashimoto","Moreau",
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
    def make_entry():
        session_surnames = [s for s in rng.sample(surnames, 5) if s not in EXCLUDE][:5]
        if len(session_surnames) < 5:
            session_surnames.append("Ivanova")
        chair = session_surnames[0]
        mover = session_surnames[rng.randint(1,4)]
        seconder = session_surnames[rng.randint(1,4)]
        if seconder == mover:
            seconder = session_surnames[1] if mover != session_surnames[1] else session_surnames[2]
        liftable.append(seconder)
        liftable.append(mover)
        item = rng.choice(items)
        yes = rng.randint(2,4); no = 5 - yes - rng.randint(0,1); abst = 5 - yes - no
        outcome = "PASSED" if yes > no else "FAILED"
        date = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        chair_str = f"{rng.choice(firstnames)} {chair} (chair)"
        attendee_str = chair_str + ", " + ", ".join(f"{rng.choice(firstnames)} {s}" for s in session_surnames[1:])
        votes_str = "  ".join(f"{s}: {rng.choice(['yes','yes','no','abstain'])}" for s in session_surnames)
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


def _build_version_filler(target_tokens, seed, count_fn):
    rng = random.Random(seed)
    EXCLUDE_VERSIONS = {"3.11.9"}
    orm_names = ["Granite ORM","Quartz ORM","Basalt ORM","Obsidian ORM",
                 "Feldspar ORM","Schist ORM"]
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
    def make_entry():
        orm = rng.choice(orm_names)
        major = rng.randint(2,5); minor = rng.randint(0,14)
        patch_hi = rng.randint(2,8); patch_lo = patch_hi - rng.randint(1,2)
        ver_hi = f"{major}.{minor}.{patch_hi}"
        ver_lo = f"{major}.{minor}.{patch_lo}"
        while ver_hi in EXCLUDE_VERSIONS or ver_lo in EXCLUDE_VERSIONS:
            major = rng.randint(2,5); minor = rng.randint(0,14)
            patch_hi = rng.randint(2,8); patch_lo = patch_hi - 1
            ver_hi = f"{major}.{minor}.{patch_hi}"; ver_lo = f"{major}.{minor}.{patch_lo}"
        EXCLUDE_VERSIONS.add(ver_hi); EXCLUDE_VERSIONS.add(ver_lo)
        liftable.extend([ver_hi, ver_lo])
        cve_num = f"CVE-2024-{rng.randint(10000,99999)}"
        date_hi = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        date_lo = f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        return (
            f"[RELEASE NOTES — {orm}]\n\n"
            f"Version {ver_hi} ({date_hi})\n"
            f"  - {rng.choice(feature_changes)}\n"
            f"  - {rng.choice(feature_changes)}\n\n"
            f"Version {ver_lo} ({date_lo})\n"
            f"  - Patched {cve_num}: {rng.choice(cve_changes)}\n"
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

TARGET_PROBES = ["art_01", "art_02", "art_06", "art_07"]


# ---------------------------------------------------------------------------
# Helpers
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


def _call(prompt):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_predict": MIN_PREDICT, "temperature": 0},
        "stream": False,
        "think": False,
    }).encode()
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
        thinking = (data["message"].get("thinking") or "").strip()
        return content, thinking, time.monotonic() - t0, data.get("eval_count"), data.get("prompt_eval_count"), data.get("done_reason")
    except Exception as e:
        print(f"    [CALL ERROR: {e}]")
        return None, None, time.monotonic() - t0, None, None, None


def left_truncate(prompt, full_tokens, target_tokens):
    if target_tokens >= full_tokens:
        return prompt
    chars_to_keep = round(len(prompt) * target_tokens / full_tokens)
    chars_to_keep = max(1, min(chars_to_keep, len(prompt)))
    return prompt[len(prompt) - chars_to_keep:]


def value_in_text(value, text):
    return bool(re.search(r"(?<![0-9a-zA-Z._-])" + re.escape(value) + r"(?![0-9a-zA-Z._-])", text))


def lifted_from_filler(output, liftable_values):
    if not output:
        return False
    return any(value_in_text(v, output) for v in liftable_values)


def classify_done_reason(output, done_reason):
    if done_reason == "length":
        return "budget_exhausted"
    if done_reason == "stop" and not output:
        return "null_response"
    return "done_reason"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    result_path = RESULTS / "stage_a_scale.json"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    rows = data["rows"]
    liftable_values = data["liftable_values"]

    empty_indices = [i for i, r in enumerate(rows) if not (r.get("output") or "").strip()]
    print(f"Found {len(empty_indices)} empty-output rows to rerun (MIN_PREDICT={MIN_PREDICT})")

    import importlib.util
    spec = importlib.util.spec_from_file_location("scorers", PROBES_DIR / "scorers.py")
    scorers_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorers_mod)

    segments = {
        s["id"]: s
        for s in [json.loads(l) for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if s["id"] in TARGET_PROBES
    }

    print("Building F-NUM filler (seed=42)...")
    filler_a = ctx_mod.build_filler(FILLER_TOKENS, seed=42, count_fn=_count_fn, variant="F-NUM")

    print("Building F-TYPED fillers (seed=42)...")
    filler_b_data = {}
    for pid in TARGET_PROBES:
        text, liftable = TYPED_BUILDERS[pid](FILLER_TOKENS, seed=42, count_fn=_count_fn)
        filler_b_data[pid] = (text, liftable)
        print(f"  {pid}: {len(liftable)} liftable values")

    # Precompute full token counts for each probe×filler combination
    print("Measuring full prompt token counts...")
    full_tokens_cache = {}
    for pid in TARGET_PROBES:
        seg = segments[pid]
        key_fnum = (pid, "filler_a_F-NUM")
        key_ftyped = (pid, "filler_b_F-TYPED")
        if key_fnum not in full_tokens_cache:
            ft = _count_fn(f"{seg['artifact']}\n\n{filler_a}\n\n{seg['question']}")
            full_tokens_cache[key_fnum] = ft
        if key_ftyped not in full_tokens_cache:
            ft = _count_fn(f"{seg['artifact']}\n\n{filler_b_data[pid][0]}\n\n{seg['question']}")
            full_tokens_cache[key_ftyped] = ft
    print(f"  Cached {len(full_tokens_cache)} prompt sizes")

    results_by_done_reason = {"budget_exhausted": 0, "null_response": 0, "became_nonempty": 0}
    call_n = 0

    for idx in empty_indices:
        row = rows[idx]
        pid = row["probe_id"]
        filler_name = row["filler_type"]
        ratio = row["budget_ratio"]
        rep = row["rep"]
        seg = segments[pid]

        if filler_name.startswith("filler_a"):
            filler_text = filler_a
            liftable = []
        else:
            filler_text, liftable = filler_b_data[pid]

        full_tokens = full_tokens_cache[(pid, filler_name)]
        t_tokens = round(full_tokens * ratio)
        truncating = t_tokens < full_tokens
        base_prompt = f"{seg['artifact']}\n\n{filler_text}\n\n{seg['question']}"
        prompt = left_truncate(base_prompt, full_tokens, t_tokens)

        call_n += 1
        print(f"[{call_n}/{len(empty_indices)}] {pid} {filler_name[8:13]} r={ratio:.2f} rep={rep}")
        output, thinking, latency, eval_count, prompt_eval_count, done_reason = _call(prompt)

        score, score_detail = (
            scorers_mod.score({"id": pid, "scorer_type": seg["scorer_type"], "expected": seg["expected"]}, output)
            if output is not None else (None, "no output")
        )
        outcome = scorers_mod.outcome_class(output, score, truncated=truncating)
        lifted = lifted_from_filler(output, liftable) if outcome == "incorrect" else False

        classification_method = classify_done_reason(output, done_reason)
        if (output or "").strip():
            results_by_done_reason["became_nonempty"] += 1
        elif done_reason == "length":
            results_by_done_reason["budget_exhausted"] += 1
        else:
            results_by_done_reason["null_response"] += 1

        print(f"  done_reason={done_reason!r}  output={repr((output or '')[:60])}  classification={classification_method}")

        rows[idx].update({
            "output": output,
            "thinking": thinking or "",
            "score": score,
            "score_detail": score_detail,
            "outcome": outcome,
            "lifted_from_filler": lifted,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "done_reason": done_reason,
            "latency_s": round(latency, 3),
            "classification_method": classification_method,
            "rerun": True,
        })

    # Mark all non-rerun rows (those without classification_method)
    marked = 0
    for row in rows:
        if "classification_method" not in row:
            row["classification_method"] = "unavailable"
            marked += 1
    print(f"\nMarked {marked} non-rerun rows with classification_method='unavailable'")

    data["rows"] = rows
    result_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Written {len(rows)} rows → {result_path}")

    print("\n=== RERUN SUMMARY ===")
    print(f"  budget_exhausted (done_reason=length):        {results_by_done_reason['budget_exhausted']}")
    print(f"  null_response (done_reason=stop, empty):      {results_by_done_reason['null_response']}")
    print(f"  became_nonempty (valid output at 1024 tok):   {results_by_done_reason['became_nonempty']}")
    total = sum(results_by_done_reason.values())
    print(f"  total rerun: {total}")

    if results_by_done_reason["became_nonempty"] > 0:
        print("\nWARNING: Some previously-empty rows produced output at 1024 tokens.")
        print("If these are fabrications, the implicit-abstention finding must be dropped.")
        print("Outputs:")
        for idx in empty_indices:
            row = rows[idx]
            if row.get("rerun") and (row.get("output") or "").strip():
                print(f"  {row['probe_id']} {row['filler_type'][8:13]} r={row['budget_ratio']} rep={row['rep']}: {repr(row['output'][:80])}")


if __name__ == "__main__":
    main()
