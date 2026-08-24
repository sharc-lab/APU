"""Interference run at r=1.20 — 120 calls.

Tests whether type-matched filler (F-TYPED) interferes with correct retrieval
even when the full artifact is present in context (af=1.00). Compares all
three models side-by-side at the headroom ratio.

Design:
  Probes:  art_01, art_02, art_06, art_07
  Filler:  F-NUM (dissimilar) and F-TYPED (type-matched)
  Ratio:   1.20 (artifact fully present, af=1.00)
  Models:  qwen3:4b-instruct, llama3.1:8b, gpt-oss:120b-cloud
  Reps:    5
  Total:   4 x 2 x 3 x 5 = 120 calls

gpt-oss:120b-cloud is a thinking model; MIN_PREDICT_120B controls its token
budget. Other models use MAX_TOKENS.

Results written to results/interference_r120.json.
"""

from __future__ import annotations

import argparse
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
MODELS = ["qwen3:4b-instruct", "llama3.1:8b", "gpt-oss:120b-cloud"]
COUNT_MODEL = "qwen3:4b-instruct"
HOST = "http://localhost:11434"
FILLER_TOKENS = 4000
BUDGET_RATIO = 1.20
N_REPS = 5
MAX_TOKENS = 256       # for qwen3 and llama
MIN_PREDICT_120B = 1024  # gpt-oss:120b-cloud: covers thinking + answer


# ---------------------------------------------------------------------------
# Filler generators (same as stage_a_scale.py, seed=42)
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


OUTFILE = RESULTS / "interference_r120.jsonl"


def _load_completed_jsonl(outfile, keys):
    """Return set of cell tuples already written to outfile."""
    completed = set()
    if not outfile.exists():
        return completed
    for line in outfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            completed.add(tuple(r[k] for k in keys))
        except (json.JSONDecodeError, KeyError):
            pass
    return completed


TYPED_BUILDERS = {
    "art_01": _build_port_filler,
    "art_02": _build_ppb_filler,
    "art_06": _build_surname_filler,
    "art_07": _build_version_filler,
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
        return content, thinking, time.monotonic() - t0, data.get("eval_count"), data.get("prompt_eval_count"), data.get("done_reason")
    except Exception as e:
        print(f"    [CALL ERROR: {e}]")
        return None, None, time.monotonic() - t0, None, None, None


def value_in_text(value, text):
    return bool(re.search(r"(?<![0-9a-zA-Z._-])" + re.escape(value) + r"(?![0-9a-zA-Z._-])", text))


def lifted_from_filler(output, liftable_values):
    if not output:
        return False
    return any(value_in_text(v, output) for v in liftable_values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells already present in the output file.")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if OUTFILE.exists() and not args.resume:
        n = sum(1 for l in OUTFILE.read_text(encoding="utf-8").splitlines() if l.strip())
        if n > 0:
            print(f"ERROR: {OUTFILE.name} has {n} existing rows. Pass --resume to continue, or delete to restart.", flush=True)
            sys.exit(1)

    spec = importlib.util.spec_from_file_location("scorers", PROBES_DIR / "scorers.py")
    scorers_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorers_mod)

    segments = {
        s["id"]: s
        for s in [json.loads(l) for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if s["id"] in TARGET_PROBES
    }

    _CELL_KEYS = ["probe_id", "filler_type", "model", "rep"]
    completed = _load_completed_jsonl(OUTFILE, _CELL_KEYS) if args.resume else set()
    if completed:
        print(f"--resume: {len(completed)} cells already completed.")

    print(f"Building F-NUM filler ({FILLER_TOKENS} tok)...")
    filler_a = ctx_mod.build_filler(FILLER_TOKENS, seed=42, count_fn=_count_fn, variant="F-NUM")
    print(f"  F-NUM: {_count_fn(filler_a)} tok ({len(filler_a)} chars)")

    print("Building F-TYPED fillers...")
    filler_b_data = {}
    all_liftable = {}
    for pid in TARGET_PROBES:
        text, liftable = TYPED_BUILDERS[pid](FILLER_TOKENS, seed=42, count_fn=_count_fn)
        tok = _count_fn(text)
        print(f"  {pid} F-TYPED: {tok} tok ({len(text)} chars), {len(liftable)} liftable values")
        filler_b_data[pid] = (text, liftable)
        all_liftable[pid] = liftable

    # Measure full prompt sizes for both filler types
    print("\nMeasuring full prompt token counts (r=1.20, no truncation)...")
    full_tokens = {}
    for pid in TARGET_PROBES:
        seg = segments[pid]
        ft_fnum = _count_fn(f"{seg['artifact']}\n\n{filler_a}\n\n{seg['question']}")
        ft_ftyp = _count_fn(f"{seg['artifact']}\n\n{filler_b_data[pid][0]}\n\n{seg['question']}")
        full_tokens[(pid, "filler_a_F-NUM")] = ft_fnum
        full_tokens[(pid, "filler_b_F-TYPED")] = ft_ftyp
        print(f"  {pid} F-NUM={ft_fnum} tok  F-TYPED={ft_ftyp} tok")

    total_calls = len(TARGET_PROBES) * 2 * len(MODELS) * N_REPS
    print(f"\nTotal cells: {total_calls}  Completed: {len(completed)}  Remaining: {total_calls - len(completed)}")
    print(f"Budget ratio: {BUDGET_RATIO} (artifact fully present, no truncation)")
    print(f"Output: {OUTFILE}")

    rows = []
    call_n = 0

    fillers = [
        ("filler_a_F-NUM",   filler_a,  {pid: []                    for pid in TARGET_PROBES}),
        ("filler_b_F-TYPED", None,      {pid: filler_b_data[pid][1] for pid in TARGET_PROBES}),
    ]

    with OUTFILE.open("a", encoding="utf-8") as out_fh:
      for filler_name, filler_text_shared, liftable_map in fillers:
        for model in MODELS:
            for pid in TARGET_PROBES:
                seg = segments[pid]
                probe_dict = {"id": pid, "scorer_type": seg["scorer_type"], "expected": seg["expected"]}

                if filler_name.startswith("filler_a"):
                    filler_text = filler_text_shared
                    liftable = liftable_map[pid]
                else:
                    filler_text, liftable = filler_b_data[pid]

                ft = full_tokens[(pid, filler_name)]
                t_tokens = round(ft * BUDGET_RATIO)
                prompt = f"{seg['artifact']}\n\n{filler_text}\n\n{seg['question']}"
                truncating = t_tokens < ft

                for rep in range(N_REPS):
                    call_n += 1
                    cell = (pid, filler_name, model, rep)
                    if cell in completed:
                        print(f"[{call_n:3d}/{total_calls}] SKIP {pid} {filler_name[8:13]} {model[:16]} r{rep}")
                        continue
                    output, thinking, latency, eval_count, prompt_eval_count, done_reason = _call(prompt, model)
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
                        "budget_ratio": BUDGET_RATIO,
                        "truncating": truncating,
                        "artifact_fraction": 1.0,
                        "rep": rep,
                        "output": output,
                        "score": score,
                        "score_detail": score_detail,
                        "outcome": outcome,
                        "lifted_from_filler": lifted,
                        "eval_count": eval_count,
                        "prompt_eval_count": prompt_eval_count,
                        "done_reason": done_reason,
                        "latency_s": round(latency, 3),
                        "hardware": "blade14_rtx4070",
                    }
                    if thinking is not None:
                        row["thinking"] = thinking

                    rows.append(row)
                    out_fh.write(json.dumps(row) + "\n")
                    out_fh.flush()

                    tag = "✓" if score == 1.0 else ("A" if outcome == "abstained" else ("L" if lifted else "✗"))
                    print(
                        f"[{call_n:3d}/{total_calls}] {pid} {filler_name[8:13]} "
                        f"{model[:16]} r{rep} {tag} {latency*1000:.0f}ms  "
                        f"{repr((output or '')[:50])}"
                    )
                    if call_n % 10 == 0:
                        print(f"--- progress: {call_n}/{total_calls} calls ---")

    print(f"\nDone. {len(rows)} rows written to {OUTFILE}")
    out = RESULTS / "interference_r120.json"
    out.write_text(
        json.dumps({"rows": rows, "liftable_values": all_liftable, "budget_ratio": BUDGET_RATIO}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWritten {len(rows)} rows -> {out}")
    _print_summary(rows)


def _print_summary(rows):
    print("\n=== INTERFERENCE SUMMARY (r=1.20, af=1.00) ===")
    print(f"{'probe':8} {'filler':14} {'model':20}  score  lift")
    for filler_name in ["filler_a_F-NUM", "filler_b_F-TYPED"]:
        for model in MODELS:
            for pid in TARGET_PROBES:
                s = [r for r in rows if r["probe_id"] == pid and r["filler_type"] == filler_name and r["model"] == model]
                if not s:
                    continue
                scores = [r["score"] for r in s if r["score"] is not None]
                lifts = [r for r in s if r.get("lifted_from_filler")]
                mean_score = sum(scores) / len(scores) if scores else float("nan")
                lift_rate = len(lifts) / len(s)
                print(f"  {pid:8} {filler_name[8:13]:14} {model:20}  {mean_score:.2f}   {lift_rate:.0%}")

    print("\n=== F-TYPED INTERFERENCE FAILURES (score < 1.0) ===")
    for r in rows:
        if r["filler_type"] == "filler_b_F-TYPED" and (r["score"] or 0) < 1.0:
            print(f"  {r['probe_id']} {r['model'][:16]} rep={r['rep']} score={r['score']} lift={r['lifted_from_filler']}  {repr((r['output'] or '')[:60])}")


if __name__ == "__main__":
    main()
