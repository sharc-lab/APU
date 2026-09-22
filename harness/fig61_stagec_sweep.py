"""Fig 6.1 Stage C replication — evo-t2s, qwen3:4b-instruct checkpoint.

11 probes, 6 ratios, 2 arms, 3 reps = 396 calls.

Matches stage_c_position_pressure.py exactly:
  - max_tokens=128, temperature=0
  - char-based left truncation
  - context.build_filler(4000, seed=42, count_fn=_tokenize)   <- same algorithm
  - scorer call: score(probe_dict, output)                     <- same signature

Differences from stage C (Blade):
  - llama-server /v1/chat/completions instead of Ollama /api/chat
  - streaming=True to capture ttft_ms
  - cache_prompt=false per request to prevent KV reuse across reps
  - stream_options.include_usage=true (reported if server honors it)
  - /tokenize instead of /api/generate for count_fn
  - n_prompt_tokens_actual measured via /tokenize on exact prompt string,
    not from API usage (which streaming does not reliably return)
  - build_id live-queried from /props at startup
  - platform = evo-t2s

Rep note: at temperature=0 with an identical prompt, reps are near-deterministic.
Per-cell variance across reps is not a meaningful error estimate (same as stage C).
"""

from __future__ import annotations

import importlib.util
import json
import time
import sys
import socket
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent   # C:/apu/APU
PROBES_DIR = REPO / "evaluation" / "probes"
RESULTS_DIR = REPO / "results"

sys.path.insert(0, str(REPO / "harness"))
import context as ctx_mod   # same module stage_c uses

SERVER_URL = "http://127.0.0.1:8383"
N_CTX_SLOT  = 8192
MAX_TOKENS  = 128        # matches stage C
TEMPERATURE = 0          # matches stage C

BUDGET_RATIOS = [1.20, 1.00, 0.85, 0.70, 0.55, 0.40]
ARMS          = ["LATE", "EARLY"]
N_REPS        = 3
FILLER_TARGET = 4000
FILLER_SEED   = 42

# Stage C probe set (11 probes confirmed from 396-row result file)
PROBE_IDS = [
    "rag_01", "rag_02", "rag_03", "rag_04", "rag_05", "rag_06",
    "sea_01", "sea_03", "sea_04", "sea_05", "sea_06",
]

HW_CONFIG  = "evox2_evo-t2s"
MEM_ARCH   = "unified"
PLATFORM   = "evo-t2s"

# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> int:
    """Count tokens via llama-server /tokenize (used as count_fn for filler)."""
    body = json.dumps({"content": text, "add_special": False}).encode()
    req  = urllib.request.Request(
        f"{SERVER_URL}/tokenize",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return len(json.loads(r.read())["tokens"])


def _query_props() -> dict:
    req = urllib.request.Request(f"{SERVER_URL}/props", method="GET")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def build_arm_prompt(arm: str, filler: str, artifact: str, question: str) -> str:
    if arm == "LATE":
        return f"{filler}\n\n{artifact}\n\n{question}"
    else:
        return f"{artifact}\n\n{filler}\n\n{question}"


def left_truncate(prompt: str, full_tokens: int, target_tokens: int) -> str:
    """Char-based left truncation matching stage_c_position_pressure.py exactly."""
    if target_tokens >= full_tokens:
        return prompt
    chars_to_keep = round(len(prompt) * target_tokens / full_tokens)
    chars_to_keep = max(1, min(chars_to_keep, len(prompt)))
    return prompt[len(prompt) - chars_to_keep:]


def _chat_streaming(prompt: str) -> tuple[str, float, float, int, int, str | None]:
    """POST to /v1/chat/completions with stream=True, cache disabled.

    Returns (text, latency_ms, ttft_ms, tokens_in_api, tokens_out, finish_reason).
    tokens_in_api comes from the usage chunk if the server returns it; 0 otherwise.
    cache_prompt=false prevents KV reuse across reps so TTFT reflects real prefill.
    stream_options.include_usage=true requests a trailing usage chunk (honored if supported).
    """
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
        "cache_prompt": False,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0            = time.perf_counter()
    ttft_ms       = None
    chunks        = []
    tokens_in_api = 0
    tokens_out    = 0
    finish_reason = None

    with urllib.request.urlopen(req, timeout=120) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or []
            choice0 = choices[0] if choices else {}
            delta   = choice0.get("delta", {})
            content = delta.get("content", "")
            if content and ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            if content:
                chunks.append(content)

            fr = choice0.get("finish_reason")
            if fr:
                finish_reason = fr

            usage = chunk.get("usage")
            if usage:
                tokens_in_api = usage.get("prompt_tokens",     tokens_in_api)
                tokens_out    = usage.get("completion_tokens", tokens_out)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = "".join(chunks).strip()
    return text, latency_ms, ttft_ms or latency_ms, tokens_in_api, tokens_out, finish_reason


def _verify_cache_disabled(sample_prompt: str) -> None:
    """Run 2 identical calls and assert TTFT is consistent (< 10x spread).

    With cache_prompt=false, both calls must do full prefill every time.
    The 70x spread seen in the 20260922T191031Z run (5177ms vs 75ms) was
    KV cache reuse, not prompt size variation.
    """
    ttfts = []
    for i in range(2):
        _, _, ttft_ms, _, _, _ = _chat_streaming(sample_prompt)
        ttfts.append(ttft_ms)
        print(f"  cache_check call {i}: ttft={ttft_ms:.0f}ms")
    ratio = max(ttfts) / max(min(ttfts), 1.0)
    if ratio > 10.0:
        raise RuntimeError(
            f"TTFT spread {ratio:.1f}x ({ttfts[0]:.0f}ms vs {ttfts[1]:.0f}ms) "
            f"-- cache_prompt=false not honored. STOP."
        )
    print(f"  cache_check OK: {ttfts[0]:.0f}ms vs {ttfts[1]:.0f}ms  ratio={ratio:.1f}x")


def _load_scorers():
    spec = importlib.util.spec_from_file_location(
        "probes_scorers", PROBES_DIR / "scorers.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── main ──────────────────────────────────────────────────────────────────────

def main(smoke: bool = False, gate: bool = False, blade: bool = False):
    hostname = socket.gethostname().upper()
    if blade:
        platform     = "blade_rtx4070"
        hw_config    = "blade_rtx4070"
        mem_arch     = "discrete"
        print(f"MODE: --blade (Razer Blade 14, RTX 4070, CUDA b10970)")
    else:
        assert "EVO" in hostname or "T2S" in hostname, \
            f"WRONG HOST: {hostname}. Must run on EVO-T2S."
        platform     = PLATFORM
        hw_config    = HW_CONFIG
        mem_arch     = MEM_ARCH

    props    = _query_props()
    build_id = props.get("build_info", props.get("build", "UNKNOWN"))
    print(f"HOSTNAME : {hostname}")
    print(f"BUILD    : {build_id}")
    print()

    scorers = _load_scorers()

    all_segs  = [
        json.loads(l)
        for l in (PROBES_DIR / "segments.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    seg_by_id = {s["id"]: s for s in all_segs}
    missing   = [pid for pid in PROBE_IDS if pid not in seg_by_id]
    if missing:
        raise RuntimeError(f"Probes missing from segments.jsonl: {missing}")
    probes = [seg_by_id[pid] for pid in PROBE_IDS]

    print(f"Probes   : {len(probes)}")
    print(f"Building filler ({FILLER_TARGET} tokens, seed={FILLER_SEED})...")
    filler        = ctx_mod.build_filler(FILLER_TARGET, seed=FILLER_SEED, count_fn=_tokenize)
    filler_tokens = _tokenize(filler)
    print(f"  filler_tokens={filler_tokens} ({len(filler)} chars)")
    print()

    # Verify cache_prompt=false is honored before running any real calls.
    print("Verifying cache disabled (2 identical calls, expect TTFT within 10x)...")
    _sample_probe   = probes[0]
    _sample_prompt  = build_arm_prompt(
        "LATE",
        filler,
        _sample_probe["artifact"].strip(),
        _sample_probe["question"].strip(),
    )
    _verify_cache_disabled(_sample_prompt)
    print()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if smoke:
        out_path     = RESULTS_DIR / f"fig61_stagec_smoke_{ts}.jsonl"
        run_tag      = f"stagec_smoke_{ts}"
        probe_subset = [probes[0]]
        ratio_list   = BUDGET_RATIOS
        n_reps       = 1
        total        = len(ratio_list) * 2 * n_reps
        print(f"SMOKE TEST: 1 probe × {len(ratio_list)} ratios × 2 arms × 1 rep = {total} calls")
    elif gate:
        out_path     = RESULTS_DIR / f"fig61_stagec_gate_{ts}.jsonl"
        run_tag      = f"stagec_gate_{ts}"
        probe_subset = probes
        ratio_list   = [1.20]
        n_reps       = 1
        total        = len(probe_subset) * 2 * n_reps
        print(f"BASELINE GATE: {len(probe_subset)} probes × 1 ratio × 2 arms × 1 rep = {total} calls")
    else:
        out_path     = RESULTS_DIR / f"fig61_stagec_full_{ts}.jsonl"
        run_tag      = f"stagec_full_{ts}"
        probe_subset = probes
        ratio_list   = BUDGET_RATIOS
        n_reps       = N_REPS
        total        = len(probe_subset) * len(ratio_list) * 2 * n_reps
        print(f"FULL SWEEP: {len(probe_subset)} probes × {len(ratio_list)} ratios "
              f"× 2 arms × {n_reps} reps = {total} calls")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-measure full token counts (LATE arm = canonical)
    full_tokens_cache: dict[str, int] = {}
    for probe in probe_subset:
        pid      = probe["id"]
        artifact = probe["artifact"].strip()
        question = probe["question"].strip()
        full_p   = build_arm_prompt("LATE", filler, artifact, question)
        ft       = _tokenize(full_p)
        full_tokens_cache[pid] = ft
        print(f"  {pid}: {ft} full tokens")
    print()

    rows   = []
    call_n = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for ratio in ratio_list:
            for probe in probe_subset:
                pid           = probe["id"]
                artifact      = probe["artifact"].strip()
                question      = probe["question"].strip()
                expected      = probe["expected"]
                scorer_type   = probe["scorer_type"]
                full_tokens   = full_tokens_cache[pid]
                target_tokens = round(full_tokens * ratio)
                truncating    = ratio < 1.0
                intended      = min(target_tokens, full_tokens)

                for arm in ARMS:
                    for rep in range(n_reps):
                        call_n    += 1
                        t_setup    = time.perf_counter_ns()

                        full_prompt = build_arm_prompt(arm, filler, artifact, question)
                        if truncating:
                            prompt        = left_truncate(full_prompt, full_tokens, target_tokens)
                            chars_dropped = len(full_prompt) - len(prompt)
                        else:
                            prompt        = full_prompt
                            chars_dropped = 0

                        art_len = len(artifact)
                        if arm == "EARLY" and truncating and art_len > 0:
                            art_frac = max(0.0, 1.0 - min(chars_dropped, art_len) / art_len)
                        else:
                            art_frac = 1.0

                        orch_setup_ns = time.perf_counter_ns() - t_setup

                        # Direct token count on exact prompt string (does not depend on
                        # server reporting usage in streaming mode).
                        n_prompt_tokens_actual = _tokenize(prompt)

                        t_http = time.perf_counter_ns()

                        try:
                            output, latency_ms, ttft_ms, tokens_in_api, tokens_out, finish_reason = \
                                _chat_streaming(prompt)
                            error = None
                        except Exception as e:
                            output         = ""
                            latency_ms     = 0.0
                            ttft_ms        = None
                            tokens_in_api  = 0
                            tokens_out     = 0
                            finish_reason  = None
                            error          = str(e)

                        http_client_ns = time.perf_counter_ns() - t_http

                        # PC: direct /tokenize measurement vs intended budget tokens.
                        pc_ok = abs(n_prompt_tokens_actual - intended) / max(intended, 1) <= 0.05

                        # ── score: same call as stage_c_position_pressure.py ──
                        probe_dict = {
                            "id":          pid,
                            "scorer_type": scorer_type,
                            "expected":    expected,
                        }
                        try:
                            score, score_detail = scorers.score(probe_dict, output)
                        except Exception as e:
                            score        = None
                            score_detail = f"scorer_error:{e}"

                        row = {
                            "probe_id":                   pid,
                            "category":                   probe["category"],
                            "difficulty":                 probe.get("difficulty", ""),
                            "arm":                        arm,
                            "rep":                        rep,
                            "run":                        run_tag,
                            "budget_ratio":               ratio,
                            "target_tokens":              target_tokens,
                            "full_tokens":                full_tokens,
                            "filler_tokens":              filler_tokens,
                            "truncating":                 truncating,
                            "truncation_method":          "left_char" if truncating else "none",
                            "chars_dropped":              chars_dropped,
                            "artifact_fraction_retained": round(art_frac, 4),
                            "score":                      score,
                            "score_detail":               score_detail,
                            "output":                     output,
                            "latency_ms":                 round(latency_ms, 1),
                            "ttft_ms":                    round(ttft_ms, 1) if ttft_ms else None,
                            "orch_setup_ns":              orch_setup_ns,
                            "http_client_ns":             http_client_ns,
                            "n_prompt_tokens_actual":     n_prompt_tokens_actual,
                            "tokens_in_api":              tokens_in_api,
                            "intended_budget_tokens":     intended,
                            "positive_control_ok":        pc_ok,
                            "tokens_out":                 tokens_out,
                            "finish_reason":              finish_reason,
                            "n_ctx_slot":                 N_CTX_SLOT,
                            "count_method":               "llamaserver_tokenize",
                            "model":                      "qwen3:4b-instruct",
                            "build_id":                   build_id,
                            "platform":                   platform,
                            "hardware_config":            hw_config,
                            "memory_architecture":        mem_arch,
                            "error":                      error,
                        }
                        rows.append(row)
                        fout.write(json.dumps(row) + "\n")
                        fout.flush()

                        pc_flag  = "PC+" if pc_ok else "PC-"
                        api_flag = f"api={tokens_in_api}" if tokens_in_api else "api=?"
                        print(f"  [{call_n:3d}/{total}] {pid} {arm} r={ratio:.2f} rep={rep} "
                              f"score={score} tok={n_prompt_tokens_actual}({api_flag}) "
                              f"ttft={ttft_ms:.0f}ms lat={latency_ms:.0f}ms "
                              f"{pc_flag} finish={finish_reason}")

    print(f"\nWrote {len(rows)} rows -> {out_path.name}")

    pc_pass = sum(1 for r in rows if r["positive_control_ok"])
    manifest = {
        "schema_version":   1,
        "run_tag":          run_tag,
        "result_file":      out_path.name,
        "n_rows":           len(rows),
        "platform":         platform,
        "hardware_config":  hw_config,
        "memory_architecture": mem_arch,
        "server": {
            "build_id":        build_id,
            "ctx_size":        N_CTX_SLOT,
            "n_parallel":      1,
            "port":            8383,
            "reasoning_flags": "none",
        },
        "model": {
            "alias":       "qwen3:4b-instruct",
            "gguf_sha256": "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9",
        },
        "generation": {
            "max_tokens":                  MAX_TOKENS,
            "temperature":                 TEMPERATURE,
            "cache_prompt":                False,
            "stream_options_include_usage": True,
        },
        "filler": {
            "target_tokens": FILLER_TARGET,
            "actual_tokens": filler_tokens,
            "seed":          FILLER_SEED,
            "variant":       "F-NUM",
            "count_fn":      "llamaserver_tokenize",
        },
        "truncation": {
            "method":                    "left_char",
            "chars_per_token_heuristic": 5.03,
        },
        "n_prompt_tokens_source": "llamaserver_tokenize",
        "sweep": {
            "n_probes":       len(probe_subset),
            "n_budget_ratios": len(ratio_list),
            "budget_ratios":  ratio_list,
            "n_arms":         2,
            "n_reps":         n_reps,
            "total_calls":    total,
        },
        "positive_control": {
            "tolerance_pct": 5.0,
            "n_pass":        pc_pass,
            "n_fail":        len(rows) - pc_pass,
            "pass_rate_pct": round(pc_pass / len(rows) * 100, 1) if rows else 0,
        },
    }
    manifest_path = out_path.with_name(out_path.name.replace("_full_", "_manifest_")
                                                      .replace("_smoke_", "_manifest_")
                                                      .replace("_gate_", "_manifest_")
                                                      .replace(".jsonl", ".json"))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest -> {manifest_path.name}")
    return rows, out_path


def run_gate_check(gate_rows: list, stage_c_path: Path) -> tuple[int, list]:
    """Compare gate rows (ratio=1.20) to stage C means. Threshold: diff > 0.5."""
    from collections import defaultdict
    sc_rows = [
        json.loads(l)
        for l in stage_c_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    sc_cells: dict[tuple, list] = defaultdict(list)
    for r in sc_rows:
        if abs(r["budget_ratio"] - 1.20) < 0.01:
            sc_cells[(r["probe_id"], r["arm"])].append(r["score"])
    sc_mean = {k: sum(v) / len(v) for k, v in sc_cells.items() if v}

    disagreements = []
    for r in gate_rows:
        key    = (r["probe_id"], r["arm"])
        sc_val = sc_mean.get(key)
        if sc_val is None:
            continue
        new_val = r["score"] or 0.0
        if abs(new_val - sc_val) > 0.5:
            disagreements.append({
                "probe_id":     r["probe_id"],
                "arm":          r["arm"],
                "stage_c_mean": round(sc_val, 3),
                "evo_t2s":      new_val,
                "output":       (r.get("output") or "")[:200],
            })
    return len(disagreements), disagreements


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gate",  action="store_true")
    ap.add_argument("--full",  action="store_true")
    ap.add_argument("--blade", action="store_true",
                    help="Run on Razer Blade 14 (RTX 4070, CUDA). "
                         "Skips EVO-T2S hostname check and sets blade_rtx4070 metadata.")
    args = ap.parse_args()

    STAGE_C = RESULTS_DIR / "stage_c_20260818T040408Z.jsonl"

    if args.smoke:
        smoke_rows, _ = main(smoke=True, blade=args.blade)
        pc_fails      = [r for r in smoke_rows if not r["positive_control_ok"]]
        print(f"\nSMOKE: PC failures={len(pc_fails)}")
        for r in pc_fails:
            print(f"  PC FAIL {r['probe_id']} {r['arm']} r={r['budget_ratio']} "
                  f"intended={r['intended_budget_tokens']} "
                  f"actual={r['n_prompt_tokens_actual']} api={r['tokens_in_api']}")
        print(f"SMOKE {'PASS' if not pc_fails else 'FAIL'}")

    elif args.gate:
        gate_rows, _      = main(gate=True, blade=args.blade)
        n_disagree, diffs = run_gate_check(gate_rows, STAGE_C)
        print(f"\nGATE: {n_disagree}/22 cells disagree with stage C at ratio=1.20")
        for d in diffs:
            print(f"  DISAGREE {d['probe_id']} {d['arm']}  "
                  f"stage_C={d['stage_c_mean']} evo_t2s={d['evo_t2s']}")
            print(f"    out: {repr(d['output'][:120])}")
        if n_disagree > 4:
            print(f"GATE FAIL: {n_disagree} > 4. STOP.")
            sys.exit(1)
        else:
            print(f"GATE PASS ({n_disagree} <= 4). Run full sweep.")

    elif args.full:
        rows, path = main(blade=args.blade)
        print(f"Full sweep: {len(rows)} rows -> {path.name}")
