"""Fig 6.1 draft sweep — budget_ratio × position arm, evo-t2s llama-server.

budget_ratio = target_tokens / full_prompt_tokens  (same as stage C)
At ratio < 1.0: harness-side left-char truncation. LATE arm (filler first) loses
filler; EARLY arm (artifact first) loses the artifact. This is the signal.
At ratio >= 1.0: full prompt sent, no truncation.

Run this on evo-t2s where llama-server is listening on 127.0.0.1:8383.
SCP segments.jsonl and scorers.py to the same directory as this script (or pass
--segments / --scorers), then:
    py -3 fig61_sweep.py --smoke        # 1 probe x 6 ratios x 2 arms x 1 rep
    py -3 fig61_sweep.py                # 5 probes x 6 ratios x 2 arms x 3 reps
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── server config ──────────────────────────────────────────────────────────────
SERVER_URL   = "http://127.0.0.1:8383"
N_CTX_SLOT   = 8192
MODEL_ALIAS  = "qwen3:4b-instruct"
MAX_TOKENS   = 128
BUILD_ID     = "b10970"
PLATFORM     = "evo-t2s"
HW_CONFIG    = "evox2_evo-t2s"
MEM_ARCH     = "unified"

# ── sweep parameters ────────────────────────────────────────────────────────────
BUDGET_RATIOS   = [1.20, 1.00, 0.85, 0.70, 0.55, 0.40]
ARMS            = ["LATE", "EARLY"]
PROBE_IDS_FULL  = ["rag_01", "rag_02", "rag_05", "sea_04", "sea_01"]
FILLER_TARGET   = 4000
FILLER_SEED     = 42
_CHARS_PER_TOK  = 5.03

# ── filler template (F-NUM, from harness/context.py) ───────────────────────────
_TEMPLATE = (
    "Administrative log entry {n}: The oversight committee reviewed all "
    "submitted documentation for compliance period {n} and confirmed that "
    "operational metrics remained within established baseline parameters. "
    "No anomalies were recorded in district {n} during the reference interval. "
    "Budget allocations for cycle {n} were processed according to standing "
    "procedure without escalation. Routine maintenance of infrastructure "
    "segment {n} was completed on schedule and filed under reference {n}. "
)


# ── stage C mechanics (inlined from stage_c_position_pressure.py) ──────────────

def build_arm_prompt(arm: str, filler: str, artifact: str, question: str) -> str:
    if arm == "LATE":
        return f"{filler}\n\n{artifact}\n\n{question}"
    else:
        return f"{artifact}\n\n{filler}\n\n{question}"


def left_truncate(prompt: str, full_tokens: int, target_tokens: int) -> str:
    if target_tokens >= full_tokens:
        return prompt
    chars_to_keep = round(len(prompt) * target_tokens / full_tokens)
    chars_to_keep = max(1, min(chars_to_keep, len(prompt)))
    return prompt[len(prompt) - chars_to_keep:]


# ── server HTTP helpers ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> int:
    """Count tokens via /tokenize (add_special=False)."""
    body = json.dumps({"content": text, "add_special": False}).encode()
    req  = urllib.request.Request(
        f"{SERVER_URL}/tokenize", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return len(json.loads(r.read())["tokens"])


def _chat(prompt: str, max_tokens: int) -> tuple[str, float, int, int, str | None]:
    """POST /v1/chat/completions (non-streaming).
    Returns (text, latency_ms, tokens_in, tokens_out, finish_reason).
    """
    payload = json.dumps({
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    latency_ms = (time.perf_counter() - t0) * 1000.0

    choice      = resp["choices"][0]
    text        = choice["message"]["content"] or ""
    finish      = choice.get("finish_reason")
    usage       = resp.get("usage", {})
    tokens_in   = usage.get("prompt_tokens", 0)
    tokens_out  = usage.get("completion_tokens", 0)
    return text, latency_ms, tokens_in, tokens_out, finish


# ── filler builder ─────────────────────────────────────────────────────────────

def _build_filler(target: int, seed: int = 42) -> tuple[str, int]:
    """Build F-NUM filler calibrated to target tokens via /tokenize.
    Returns (filler_text, actual_token_count).
    """
    rng   = random.Random(seed)
    start = rng.randint(10_000, 99_999)
    needed_chars = int(target * _CHARS_PER_TOK * 1.5)
    chunks: list[str] = []
    total = 0
    i = start
    while total < needed_chars:
        chunk = _TEMPLATE.format(n=i)
        chunks.append(chunk)
        total += len(chunk)
        i += 1
    raw = "".join(chunks)

    filler = raw[:int(target * _CHARS_PER_TOK)].strip()
    best, best_err = filler, float("inf")
    for it in range(8):
        actual = _tokenize(filler)
        err    = abs(actual - target) / max(target, 1)
        if err < best_err:
            best, best_err = filler, err
        print(f"  [filler calibration {it+1}/8] {actual} tok  err={err:.1%}")
        if err <= 0.02:
            break
        new_chars = min(int(len(filler) * target / max(actual, 1)), len(raw))
        if new_chars == len(filler):
            break
        filler = raw[:new_chars].strip()
    if best_err > 0.02:
        print(f"  [filler] WARNING: did not converge (best err={best_err:.1%})")
    actual_final = _tokenize(best)
    return best, actual_final


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="segments.jsonl")
    ap.add_argument("--scorers",  default="scorers.py")
    ap.add_argument("--out-dir",  default="C:\\apu\\fig61_results")
    ap.add_argument("--reps",     type=int, default=3)
    ap.add_argument("--smoke",    action="store_true",
                    help="Smoke test: rag_01 only, 1 rep, all ratios+arms (12 calls)")
    args = ap.parse_args()

    # Load scorers
    spec = importlib.util.spec_from_file_location("scorers", args.scorers)
    scorers_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorers_mod)

    # Load segments
    raw_segs = [json.loads(l) for l in Path(args.segments).read_text("utf-8").splitlines() if l.strip()]
    probe_ids = ["rag_01"] if args.smoke else PROBE_IDS_FULL
    segs = [s for s in raw_segs if s["id"] in probe_ids]
    if not segs:
        sys.exit(f"ERROR: no segments found for {probe_ids}")

    n_reps    = 1 if args.smoke else args.reps
    mode_tag  = "smoke" if args.smoke else "full"
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path  = out_dir / f"fig61_{mode_tag}_{ts}.jsonl"
    mani_path = out_dir / f"fig61_{mode_tag}_{ts}_manifest.json"

    print(f"Fig 6.1 {mode_tag.upper()} sweep  server={SERVER_URL}  n_ctx={N_CTX_SLOT}")
    print(f"Probes : {[s['id'] for s in segs]}   Ratios: {BUDGET_RATIOS}   Reps: {n_reps}")

    # Verify server health
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=8) as r:
            print(f"Health : HTTP {r.status}")
    except urllib.error.HTTPError as e:
        print(f"Health : HTTP {e.code} (may be normal for /health returning 503 when loaded)")
    except Exception as e:
        sys.exit(f"ERROR: server not reachable — {e}")

    # Build filler
    print(f"\nBuilding filler (target={FILLER_TARGET} tok, seed={FILLER_SEED})...")
    filler_text, filler_tokens = _build_filler(FILLER_TARGET, FILLER_SEED)
    print(f"Filler : {filler_tokens} tokens, {len(filler_text)} chars\n")

    # Pre-measure full prompt tokens per probe
    print("Measuring full prompt tokens per probe...")
    for seg in segs:
        late_p  = build_arm_prompt("LATE",  filler_text, seg["artifact"], seg["question"])
        early_p = build_arm_prompt("EARLY", filler_text, seg["artifact"], seg["question"])
        seg["_late_prompt"]  = late_p
        seg["_early_prompt"] = early_p
        # LATE and EARLY prompts have the same total token count (same components, different order)
        seg["_full_tokens"] = _tokenize(late_p)
        print(f"  {seg['id']:10s}: {seg['_full_tokens']} tokens")
    print()

    total_calls = len(BUDGET_RATIOS) * len(segs) * len(ARMS) * n_reps
    print(f"Total calls: {total_calls}   Output: {out_path}\n")

    rows: list[dict] = []
    call_n = 0

    for ratio in BUDGET_RATIOS:
        for seg in segs:
            full_tokens  = seg["_full_tokens"]
            target_toks  = round(full_tokens * ratio)
            truncating   = target_toks < full_tokens
            effective    = min(target_toks, full_tokens)

            for arm in ARMS:
                full_prompt = seg["_late_prompt"] if arm == "LATE" else seg["_early_prompt"]

                for rep in range(n_reps):
                    call_n += 1

                    # ── ORCH SETUP: prompt assembly + truncation ──────────────
                    t_orch = time.perf_counter_ns()
                    if truncating:
                        prompt_sent   = left_truncate(full_prompt, full_tokens, target_toks)
                        chars_dropped = len(full_prompt) - len(prompt_sent)
                    else:
                        prompt_sent   = full_prompt
                        chars_dropped = 0
                    orch_setup_ns = time.perf_counter_ns() - t_orch

                    # artifact_fraction_retained
                    if arm == "EARLY" and truncating and chars_dropped > 0:
                        art_len   = len(seg["artifact"])
                        art_drop  = min(chars_dropped, art_len)
                        art_frac  = max(0.0, 1.0 - art_drop / art_len)
                    else:
                        art_frac  = 1.0

                    # ── HTTP CLIENT: inference ────────────────────────────────
                    t_http = time.perf_counter_ns()
                    error: str | None = None
                    try:
                        output, latency_ms, tokens_in, tokens_out, finish = \
                            _chat(prompt_sent, MAX_TOKENS)
                    except Exception as exc:
                        output    = None
                        latency_ms = None
                        tokens_in = 0
                        tokens_out = 0
                        finish    = None
                        error     = str(exc)
                    http_client_ns = time.perf_counter_ns() - t_http

                    # ── scoring ───────────────────────────────────────────────
                    if output is not None:
                        probe_d = {
                            "id":           seg["id"],
                            "scorer_type":  seg["scorer_type"],
                            "expected":     seg["expected"],
                        }
                        score_val, score_det = scorers_mod.score(probe_d, output)
                    else:
                        score_val, score_det = None, f"ERROR: {error}"

                    # ── positive control ──────────────────────────────────────
                    pc_ok = (
                        tokens_in > 0
                        and abs(tokens_in - effective) / max(effective, 1) <= 0.05
                    )

                    row = {
                        # identity
                        "probe_id":    seg["id"],
                        "category":    seg["category"],
                        "difficulty":  seg["difficulty"],
                        "arm":         arm,
                        "rep":         rep,
                        "run":         ts,
                        # budget
                        "budget_ratio":        ratio,
                        "target_tokens":       target_toks,
                        "full_tokens":         full_tokens,
                        "filler_tokens":       filler_tokens,
                        "truncating":          truncating,
                        "truncation_method":   "harness_left_char" if truncating else "none",
                        "chars_dropped":       chars_dropped,
                        "artifact_fraction_retained": round(art_frac, 3),
                        # quality axis
                        "score":        score_val,
                        "score_detail": score_det,
                        "output":       output,
                        # latency axis
                        "latency_ms":      round(latency_ms, 1) if latency_ms is not None else None,
                        "orch_setup_ns":   orch_setup_ns,
                        "http_client_ns":  http_client_ns,
                        "tool_compute_ns": 0,
                        # positive control
                        "n_prompt_tokens_actual": tokens_in,
                        "n_ctx_slot":             N_CTX_SLOT,
                        "intended_budget_tokens":  effective,
                        "positive_control_ok":     pc_ok,
                        # metadata
                        "tokens_out":       tokens_out,
                        "finish_reason":    finish,
                        "count_method":     "llamaserver_tokenize",
                        "model":            "qwen3:4b-q4_k_m",
                        "build_id":         BUILD_ID,
                        "platform":         PLATFORM,
                        "hardware_config":  HW_CONFIG,
                        "memory_architecture": MEM_ARCH,
                        "error":            error,
                    }
                    rows.append(row)
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")

                    # progress line
                    s_tag  = "1.0" if score_val == 1.0 else ("0.0" if score_val == 0.0 else str(score_val))
                    t_tag  = "T" if truncating else " "
                    pc_tag = "PC+" if pc_ok else "PC!"
                    lat_s  = f"{round(latency_ms):.0f}ms" if latency_ms is not None else "ERR"
                    print(
                        f"[{call_n:4d}/{total_calls}] {seg['id']:10s} {arm:5s} "
                        f"r={ratio:.2f} rep={rep} [{t_tag}] "
                        f"score={s_tag:5s} lat={lat_s:7s} {pc_tag} "
                        f"art={art_frac:.2f}"
                    )

    # ── summary ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Done: {len(rows)} rows  ->  {out_path}")

    pc_pass = sum(1 for r in rows if r["positive_control_ok"])
    print(f"Positive control: {pc_pass}/{len(rows)} rows OK")

    # Score table
    cell: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if r["score"] is not None:
            cell[(r["budget_ratio"], r["arm"])].append(r["score"])

    print(f"\n{'Ratio':>8}  {'LATE':>8}  {'EARLY':>8}  {'delta':>8}  N")
    for ratio in BUDGET_RATIOS:
        late  = cell[(ratio, "LATE")]
        early = cell[(ratio, "EARLY")]
        lm    = sum(late)  / len(late)  if late  else float("nan")
        em    = sum(early) / len(early) if early else float("nan")
        delta = em - lm if late and early else float("nan")
        n     = len(late) + len(early)
        print(f"{ratio:>8.2f}  {lm:>8.3f}  {em:>8.3f}  {delta:>+8.3f}  {n}")

    # Manifest
    manifest = {
        "run":                ts,
        "mode":               mode_tag,
        "server_url":         SERVER_URL,
        "n_ctx_slot":         N_CTX_SLOT,
        "filler_tokens":      filler_tokens,
        "probe_ids":          [s["id"] for s in segs],
        "budget_ratios":      BUDGET_RATIOS,
        "arms":               ARMS,
        "n_reps":             n_reps,
        "total_calls":        total_calls,
        "rows_written":       len(rows),
        "out_path":           str(out_path),
        "hardware_config":    HW_CONFIG,
        "memory_architecture": MEM_ARCH,
        "platform":           PLATFORM,
        "build_id":           BUILD_ID,
        "model":              "qwen3:4b-q4_k_m",
    }
    mani_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {mani_path}")


if __name__ == "__main__":
    main()
